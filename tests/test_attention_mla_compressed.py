from __future__ import annotations

import inspect
import math

import torch

import b12x.attention.mla.compressed_api as compressed_api_impl
import b12x.attention.mla.split as mla_split_impl
from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.attention.workspace import (
    B12XAttentionArena,
    B12XAttentionArenaCaps,
    B12XAttentionWorkspace,
)
from b12x.attention.mla.compressed_reference import (
    COMPRESSED_MLA_C128_PAGE_SIZE,
    COMPRESSED_MLA_C4_PAGE_SIZE,
    COMPRESSED_MLA_DSV4_PAGE_SIZE,
    COMPRESSED_MLA_SWA_TOKENS,
    compressed_mla_page_nbytes,
    compressed_sparse_mla_reference,
    gather_compressed_mla_kv_cache_reference,
    pack_compressed_mla_kv_cache_reference,
)
from b12x.attention.mla.compressed_config import (
    compressed_mla_split_config_for_contract,
)
from b12x.integration.mla import (
    clear_mla_caches,
    compressed_mla_decode_forward,
    compressed_mla_split_chunks_for_contract,
)
from b12x.cute.compiler import clear_compile_cache, compile_cache_info

from .helpers import require_sm120


_COMPRESSED_HEAD_DIM = 512
_SHARED_CORE_HEAD_DIM = 576
_SHARED_CORE_V_HEAD_DIM = 512
_LOCAL_Q_HEADS = 32
_SM_SCALE = 1.0 / math.sqrt(_COMPRESSED_HEAD_DIM)


def _make_split_merge_tensors(
    *,
    rows: int,
    heads: int,
    chunks: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    tmp_storage = torch.randn(
        rows * heads * chunks * _COMPRESSED_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=gen,
    )
    tmp_output = tmp_storage.as_strided(
        (rows, heads, chunks, _COMPRESSED_HEAD_DIM),
        (
            heads * _COMPRESSED_HEAD_DIM,
            _COMPRESSED_HEAD_DIM,
            rows * heads * _COMPRESSED_HEAD_DIM,
            1,
        ),
    )
    tmp_lse = torch.randn(
        (rows, heads, chunks),
        dtype=torch.float32,
        device=device,
        generator=gen,
    )
    output = torch.empty(
        (rows, heads, _COMPRESSED_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    num_chunks_ptr = torch.tensor([chunks], dtype=torch.int32, device=device)
    attn_sink = torch.zeros((heads,), dtype=torch.float32, device=device)
    return tmp_output, tmp_lse, num_chunks_ptr, attn_sink, output


@torch.inference_mode()
def test_split_sink_merge_live_rows_do_not_resolve_new_kernel() -> None:
    device = require_sm120()
    clear_compile_cache()
    mla_split_impl.clear_sparse_mla_split_kernel_cache()

    warm_args = _make_split_merge_tensors(
        rows=128,
        heads=2,
        chunks=2,
        device=device,
        seed=6120,
    )
    mla_split_impl.run_sparse_mla_split_decode_merge(
        tmp_output=warm_args[0],
        tmp_lse=warm_args[1],
        num_chunks_ptr=warm_args[2],
        attn_sink=warm_args[3],
        output=warm_args[4],
    )
    torch.cuda.synchronize(device)
    warm_misses = compile_cache_info()["compile_misses"]

    live_args = _make_split_merge_tensors(
        rows=113,
        heads=2,
        chunks=1,
        device=device,
        seed=6121,
    )
    freeze_kernel_resolution(
        "split sink merge live rows and chunks should reuse padded capture"
    )
    try:
        mla_split_impl.run_sparse_mla_split_decode_merge(
            tmp_output=live_args[0],
            tmp_lse=live_args[1],
            num_chunks_ptr=live_args[2],
            attn_sink=live_args[3],
            output=live_args[4],
        )
        torch.cuda.synchronize(device)
    finally:
        unfreeze_kernel_resolution()

    assert compile_cache_info()["compile_misses"] == warm_misses
    assert torch.isfinite(live_args[4].float()).all()


def _make_workspace(
    *,
    device: torch.device | str,
    rows: int,
    topk: int,
    max_kv_rows: int,
    use_cuda_graph: bool = False,
    head_dim: int = _COMPRESSED_HEAD_DIM,
    v_head_dim: int = _COMPRESSED_HEAD_DIM,
    max_chunks_per_row: int = 64,
    max_page_table_width: int | None = None,
) -> B12XAttentionWorkspace:
    return B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=_LOCAL_Q_HEADS,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        topk=topk,
        max_page_table_width=max_page_table_width,
        max_total_q=rows,
        max_batch=rows,
        max_kv_rows=max_kv_rows,
        use_cuda_graph=use_cuda_graph,
        max_chunks_per_row=max_chunks_per_row,
        reserve_compressed_mla_staging=True,
    )


def _make_cache(
    *,
    tokens: int,
    page_size: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k_nope = (
        torch.randn((tokens, 448), generator=gen, dtype=torch.float32, device=device)
        * 0.05
    )
    k_rope = (
        torch.randn((tokens, 64), generator=gen, dtype=torch.float32, device=device)
        * 0.05
    )
    return pack_compressed_mla_kv_cache_reference(
        k_nope,
        k_rope.to(dtype=torch.bfloat16),
        page_size=page_size,
    )


def _make_q(*, rows: int, seed: int, device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = (
        torch.randn(
            (rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
            generator=gen,
            dtype=torch.float32,
            device=device,
        )
        * 0.04
    )
    return q.to(dtype=torch.bfloat16)


def test_compressed_mla_page_byte_widths_match_padded_layout() -> None:
    assert COMPRESSED_MLA_DSV4_PAGE_SIZE == 256
    assert COMPRESSED_MLA_SWA_TOKENS == 128
    assert COMPRESSED_MLA_C4_PAGE_SIZE == COMPRESSED_MLA_DSV4_PAGE_SIZE // 4
    assert COMPRESSED_MLA_C128_PAGE_SIZE == COMPRESSED_MLA_DSV4_PAGE_SIZE // 128
    assert compressed_mla_page_nbytes(COMPRESSED_MLA_DSV4_PAGE_SIZE) == 149760
    assert compressed_mla_page_nbytes(COMPRESSED_MLA_C4_PAGE_SIZE) == 37440
    assert compressed_mla_page_nbytes(COMPRESSED_MLA_C128_PAGE_SIZE) == 1728


def test_compressed_mla_decode_does_not_pin_flash_tp2_heads_by_default() -> None:
    signature = inspect.signature(compressed_mla_decode_forward)
    assert signature.parameters["expected_num_q_heads"].default is None


def test_compressed_mla_mtp_graph_rows_keep_decode_split_contract() -> None:
    graph_rows_cfg = compressed_mla_split_config_for_contract(
        rows=192,
        width=384,
        max_chunks=6,
    )
    larger_prefill_cfg = compressed_mla_split_config_for_contract(
        rows=257,
        width=384,
        max_chunks=6,
    )

    assert graph_rows_cfg.chunk_size == 64
    assert graph_rows_cfg.num_chunks == 6
    assert larger_prefill_cfg.chunk_size == 1024
    assert larger_prefill_cfg.num_chunks == 1


def test_compressed_mla_arena_scratch_uses_contract_q_chunks() -> None:
    device = require_sm120()
    selected_widths = (128, 640, 2880)
    graph_q_rows = 16
    compressed_prefill_q = 8192
    max_chunks_per_row = max(
        compressed_mla_split_chunks_for_contract(rows=graph_q_rows, width=width)
        for width in selected_widths
    )
    mla_max_q_chunks = max(
        max(graph_q_rows, compressed_prefill_q)
        * compressed_mla_split_chunks_for_contract(
            rows=max(graph_q_rows, compressed_prefill_q),
            width=width,
        )
        for width in selected_widths
    )
    decode_q_chunks = graph_q_rows * max_chunks_per_row
    mla_max_q_chunks = max(mla_max_q_chunks, decode_q_chunks)

    base_caps = dict(
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=_LOCAL_Q_HEADS,
        indexer_num_q_heads=64,
        head_dim=_COMPRESSED_HEAD_DIM,
        max_v_head_dim=_COMPRESSED_HEAD_DIM,
        topk=max(selected_widths),
        indexer_topk=512,
        max_page_table_width=4160,
        extend_max_total_q=compressed_prefill_q,
        extend_max_batch=compressed_prefill_q,
        extend_max_kv_rows=0,
        paged_max_q_rows=4096,
        paged_max_batch=4096,
        mla_max_total_q=max(graph_q_rows, compressed_prefill_q),
        page_size=64,
        max_chunks_per_row=max_chunks_per_row,
        reserve_extend_indexer_logits=False,
        reserve_paged_indexer_logits=True,
        reserve_compressed_mla_staging=True,
        reserve_mhc=True,
        mhc_max_tokens=4096,
        mhc_hidden_size=4096,
        paged_indexer_logits_q_rows=graph_q_rows,
        paged_indexer_logits_k_rows=4160 * 64,
        paged_indexer_tile_logits_k_rows=32768,
    )
    capped = B12XAttentionArena.required_nbytes(
        B12XAttentionArenaCaps(**base_caps, mla_max_q_chunks=mla_max_q_chunks)
    )
    layout = B12XAttentionArena._layout(
        B12XAttentionArenaCaps(**base_caps, mla_max_q_chunks=mla_max_q_chunks)
    )
    legacy_ragged = B12XAttentionArena.required_nbytes(
        B12XAttentionArenaCaps(
            **{
                **base_caps,
                "extend_max_kv_rows": compressed_prefill_q * max(selected_widths),
            },
            mla_max_q_chunks=mla_max_q_chunks,
        )
    )

    assert max_chunks_per_row == 240
    assert layout.ragged_kv_nbytes <= 1024
    assert layout.output_buffer_nbytes == 0
    assert layout.compressed_q_stage_nbytes > 0
    assert layout.compressed_index_stage_nbytes > 0
    assert legacy_ragged > capped * 3
    assert capped < int(2.25 * (1 << 30))


def test_compressed_mla_reference_pack_gathers_across_padded_pages() -> None:
    device = require_sm120()
    gen = torch.Generator(device=device)
    gen.manual_seed(31)

    for page_size in (COMPRESSED_MLA_C4_PAGE_SIZE, COMPRESSED_MLA_C128_PAGE_SIZE):
        tokens = page_size * 2 + 1
        k_nope = (
            torch.randn(
                (tokens, 448), generator=gen, dtype=torch.float32, device=device
            )
            * 0.05
        )
        k_rope = (
            torch.randn((tokens, 64), generator=gen, dtype=torch.float32, device=device)
            * 0.05
        ).to(torch.bfloat16)
        cache = pack_compressed_mla_kv_cache_reference(
            k_nope, k_rope, page_size=page_size
        )
        indices = torch.tensor(
            [0, page_size - 1, page_size, page_size + 1, tokens - 1],
            dtype=torch.int32,
            device=device,
        )

        gathered, _ = gather_compressed_mla_kv_cache_reference(
            cache, indices, page_size=page_size
        )
        expected_rope = k_rope[indices.to(torch.long)].float()
        assert torch.count_nonzero(gathered[2:]).item() > 0
        torch.testing.assert_close(gathered[:, 448:], expected_rope, atol=0, rtol=0)
        torch.testing.assert_close(
            gathered[:, :448], k_nope[indices.to(torch.long)], atol=0.01, rtol=0.12
        )


@torch.inference_mode()
def test_compressed_mla_fixed_workspace_split_plan_uses_contract_not_live_shape(
    monkeypatch,
) -> None:
    device = require_sm120()
    clear_mla_caches()

    # ESCAPE HATCH (mapped indexed_page_table): this test passes a mapped extra-cache
    # indexed_page_table, which is genuinely-upstream-unsupported on the unified SM120
    # backend (upstream FlashInfer addresses the extra cache by raw slot id only; the
    # unified path correctly RAISEs). Pin the legacy escape hatch so the test exercises
    # the legacy split internals it was written for, post P10 default-flip.
    monkeypatch.setenv("B12X_MLA_SM120_UNIFIED", "0")

    contract_rows = 128
    contract_width = 2304
    live_rows = 1
    live_width = 512
    page_table_width = 4097
    max_chunks_per_row = compressed_mla_split_chunks_for_contract(
        rows=live_rows,
        width=contract_width,
    )
    workspace = _make_workspace(
        device=device,
        rows=contract_rows,
        topk=contract_width,
        max_kv_rows=1,
        use_cuda_graph=True,
        max_chunks_per_row=max_chunks_per_row,
        max_page_table_width=page_table_width,
    )

    q = _make_q(rows=live_rows, seed=121, device=device)
    swa_cache = torch.empty(
        (1, compressed_mla_page_nbytes(COMPRESSED_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    swa_indices = torch.zeros((live_rows, live_width), dtype=torch.int32, device=device)
    swa_lengths = torch.zeros((live_rows,), dtype=torch.int32, device=device)
    indexed_cache = torch.empty(
        (1, compressed_mla_page_nbytes(COMPRESSED_MLA_C4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    indexed_indices = torch.full(
        (live_rows, live_width), -1, dtype=torch.int32, device=device
    )
    indexed_lengths = torch.zeros((live_rows,), dtype=torch.int32, device=device)
    indexed_page_table = torch.full(
        (live_rows, page_table_width),
        -1,
        dtype=torch.int32,
        device=device,
    )

    calls: dict[str, int | bool] = {}

    def fake_forward(**kwargs) -> None:
        binding = kwargs["binding"]
        calls["launch_num_chunks"] = int(binding.launch_num_chunks)
        calls["direct_output"] = bool(binding.direct_output)
        calls["swa_cache_is_u8"] = binding.swa_k_cache.dtype == torch.uint8
        assert binding.q_all.shape[0] == contract_rows
        assert binding.swa_indices.shape[0] == contract_rows
        assert binding.swa_lengths.shape == (contract_rows,)
        assert binding.indexed_indices.shape == (contract_rows, live_width)
        assert binding.indexed_lengths.shape == (contract_rows,)
        assert binding.indexed_page_table.shape == (contract_rows, page_table_width)

    def fake_merge(**kwargs) -> None:
        binding = kwargs["binding"]
        assert binding.output.shape[0] == contract_rows
        binding.output.zero_()
        calls["merge"] = True

    monkeypatch.setattr(
        compressed_api_impl,
        "run_compressed_mla_split_decode_forward",
        fake_forward,
    )
    monkeypatch.setattr(
        compressed_api_impl,
        "run_sparse_mla_split_decode_merge",
        fake_merge,
    )

    actual = compressed_api_impl.compressed_mla_decode_forward(
        q_all=q,
        swa_k_cache=swa_cache.view(torch.float8_e4m3fn),
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices,
        indexed_topk_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        workspace=workspace,
        sm_scale=_SM_SCALE,
    )

    assert workspace.kv_chunk_size_value == 1024
    assert workspace.num_chunks_value == 3
    assert calls["launch_num_chunks"] == max_chunks_per_row
    assert calls["direct_output"] is False
    assert calls["swa_cache_is_u8"] is True
    assert calls["merge"] is True
    assert actual.shape == (live_rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM)


@torch.inference_mode()
def test_compressed_mla_shared_core_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    q = _make_q(rows=1, seed=21, device=device)
    swa_cache_bytes = _make_cache(
        tokens=32, page_size=COMPRESSED_MLA_DSV4_PAGE_SIZE, seed=22, device=device
    )
    indexed_cache_bytes = _make_cache(
        tokens=32, page_size=COMPRESSED_MLA_C128_PAGE_SIZE, seed=23, device=device
    )
    swa_cache = swa_cache_bytes.view(torch.float8_e4m3fn)
    indexed_cache = indexed_cache_bytes.view(torch.float8_e4m3fn)
    swa_indices = torch.arange(16, dtype=torch.int32, device=device).unsqueeze(0)
    indexed_indices = torch.arange(16, dtype=torch.int32, device=device).unsqueeze(0)
    swa_lengths = torch.tensor([11], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([7], dtype=torch.int32, device=device)
    attn_sink = torch.nn.Parameter(
        torch.linspace(-0.1, 0.1, _LOCAL_Q_HEADS, dtype=torch.float32, device=device)
    )
    workspace = _make_workspace(
        device=device,
        rows=8,
        topk=swa_indices.shape[1] + indexed_indices.shape[1],
        max_kv_rows=8 * (swa_indices.shape[1] + indexed_indices.shape[1]),
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_mla_decode_forward(
            q_all=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_lengths,
            indexed_k_cache=indexed_cache,
            indexed_indices=indexed_indices,
            indexed_topk_lengths=indexed_lengths,
            indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
            attn_sink=attn_sink,
            workspace=workspace,
            sm_scale=_SM_SCALE,
        )
        return captured_out

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_out is not None

    expected = compressed_sparse_mla_reference(
        q,
        swa_cache_bytes,
        swa_indices,
        swa_lengths,
        extra_k_cache=indexed_cache_bytes,
        extra_indices=indexed_indices,
        extra_topk_lengths=indexed_lengths,
        extra_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )
    max_abs = (captured_out.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_c128_pv_row_swizzle_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    width = 32
    q = torch.zeros(
        (1, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    k_nope = torch.zeros((width, 448), dtype=torch.bfloat16, device=device)
    k_nope[20, 0] = 1
    k_rope = torch.zeros((width, 64), dtype=torch.bfloat16, device=device)
    swa_cache = torch.empty(
        (0, compressed_mla_page_nbytes(COMPRESSED_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    indexed_cache = pack_compressed_mla_kv_cache_reference(
        k_nope,
        k_rope,
        page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
    )
    swa_indices = torch.empty((1, 0), dtype=torch.int32, device=device)
    indexed_indices = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
    swa_lengths = torch.zeros((1,), dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([width], dtype=torch.int32, device=device)
    workspace = _make_workspace(
        device=device,
        rows=1,
        topk=width,
        max_kv_rows=width,
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_mla_decode_forward(
            q_all=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_lengths,
            indexed_k_cache=indexed_cache,
            indexed_indices=indexed_indices,
            indexed_topk_lengths=indexed_lengths,
            indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
            workspace=workspace,
            sm_scale=1.0,
        )
        return captured_out

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_out is not None

    expected = torch.zeros_like(captured_out.float())
    expected[:, :, 0] = 1.0 / width
    max_abs = (captured_out.float() - expected).abs().max().item()
    assert max_abs <= 1e-4


@torch.inference_mode()
def test_compressed_mla_swa_page_size_256_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    swa_page_size = 256
    q = _make_q(rows=1, seed=91, device=device)
    swa_cache = _make_cache(tokens=300, page_size=swa_page_size, seed=92, device=device)
    swa_indices = torch.tensor(
        [[126, 127, 128, 129, 130, 255, 256, 257]],
        dtype=torch.int32,
        device=device,
    )
    swa_lengths = torch.tensor([8], dtype=torch.int32, device=device)
    attn_sink = torch.linspace(
        -0.08, 0.12, _LOCAL_Q_HEADS, dtype=torch.float32, device=device
    )
    workspace = _make_workspace(
        device=device,
        rows=q.shape[0],
        topk=swa_indices.shape[1],
        max_kv_rows=q.shape[0] * swa_indices.shape[1],
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_mla_decode_forward(
            q_all=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_lengths,
            swa_page_size=swa_page_size,
            attn_sink=attn_sink,
            workspace=workspace,
            sm_scale=_SM_SCALE,
        )
        return captured_out

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_out is not None

    expected = compressed_sparse_mla_reference(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        swa_page_size=swa_page_size,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )
    max_abs = (captured_out.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_prefill_swa_only_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    rows = 8
    width = 8
    q = _make_q(rows=rows, seed=81, device=device)
    swa_cache = _make_cache(
        tokens=32, page_size=COMPRESSED_MLA_DSV4_PAGE_SIZE, seed=82, device=device
    )
    swa_indices = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    swa_lengths = torch.empty((rows,), dtype=torch.int32, device=device)
    for row in range(rows):
        length = min(width, row + 1)
        swa_indices[row, :length] = torch.arange(
            row, row - length, -1, dtype=torch.int32, device=device
        )
        swa_lengths[row] = length
    attn_sink = torch.linspace(
        -0.2, 0.15, _LOCAL_Q_HEADS, dtype=torch.float32, device=device
    )
    workspace = _make_workspace(
        device=device,
        rows=rows,
        topk=width,
        max_kv_rows=rows * width,
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_mla_decode_forward(
            q_all=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_lengths,
            attn_sink=attn_sink,
            workspace=workspace,
            sm_scale=_SM_SCALE,
        )
        return captured_out

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_out is not None

    expected = compressed_sparse_mla_reference(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )
    max_abs = (captured_out.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_clamp_to_one_negative_extra_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    q = _make_q(rows=1, seed=71, device=device)
    swa_cache = _make_cache(
        tokens=32, page_size=COMPRESSED_MLA_DSV4_PAGE_SIZE, seed=72, device=device
    )
    indexed_cache = _make_cache(
        tokens=4, page_size=COMPRESSED_MLA_C128_PAGE_SIZE, seed=73, device=device
    )
    swa_indices = torch.arange(8, dtype=torch.int32, device=device).unsqueeze(0)
    indexed_indices = torch.full((1, 4), -1, dtype=torch.int32, device=device)
    swa_lengths = torch.tensor([6], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([1], dtype=torch.int32, device=device)
    workspace = _make_workspace(
        device=device,
        rows=q.shape[0],
        topk=swa_indices.shape[1] + indexed_indices.shape[1],
        max_kv_rows=q.shape[0] * (swa_indices.shape[1] + indexed_indices.shape[1]),
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_mla_decode_forward(
            q_all=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_lengths,
            indexed_k_cache=indexed_cache,
            indexed_indices=indexed_indices,
            indexed_topk_lengths=indexed_lengths,
            indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
            workspace=workspace,
            sm_scale=_SM_SCALE,
        )
        return captured_out

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_out is not None

    expected = compressed_sparse_mla_reference(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        extra_k_cache=indexed_cache,
        extra_indices=indexed_indices,
        extra_topk_lengths=indexed_lengths,
        extra_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
        sm_scale=_SM_SCALE,
    )
    max_abs = (captured_out.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_page_table_width_does_not_resolve_new_kernel(monkeypatch) -> None:
    device = require_sm120()
    clear_mla_caches()

    # ESCAPE HATCH (mapped indexed_page_table): passes a mapped extra-cache page table,
    # genuinely-upstream-unsupported on the unified SM120 backend (raw-slot-id only;
    # unified correctly RAISEs). Pin the legacy escape hatch so this exercises the
    # legacy split internals it was written for, post P10 default-flip.
    monkeypatch.setenv("B12X_MLA_SM120_UNIFIED", "0")

    q = _make_q(rows=1, seed=181, device=device)
    swa_cache = torch.empty(
        (0, compressed_mla_page_nbytes(COMPRESSED_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    indexed_cache = _make_cache(
        tokens=64,
        page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        seed=182,
        device=device,
    )
    swa_indices = torch.empty((1, 0), dtype=torch.int32, device=device)
    swa_lengths = torch.zeros((1,), dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([8], dtype=torch.int32, device=device)
    workspace = _make_workspace(
        device=device,
        rows=1,
        topk=8,
        max_kv_rows=8,
        use_cuda_graph=True,
    )

    indexed_indices_a = torch.arange(8, dtype=torch.int32, device=device).unsqueeze(0)
    page_table_a = torch.arange(2, dtype=torch.int32, device=device).unsqueeze(0)
    compressed_mla_decode_forward(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices_a,
        indexed_topk_lengths=indexed_lengths,
        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        indexed_page_table=page_table_a,
        workspace=workspace,
        sm_scale=_SM_SCALE,
    )
    torch.cuda.synchronize(device)

    indexed_indices_b = torch.tensor(
        [[0, 16, 17, 32, 33, 47, 48, 63]],
        dtype=torch.int32,
        device=device,
    )
    page_table_b = torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0)
    freeze_kernel_resolution("compressed MLA page-table width must be dynamic")
    try:
        actual = compressed_mla_decode_forward(
            q_all=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_lengths,
            indexed_k_cache=indexed_cache,
            indexed_indices=indexed_indices_b,
            indexed_topk_lengths=indexed_lengths,
            indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
            indexed_page_table=page_table_b,
            workspace=workspace,
            sm_scale=_SM_SCALE,
        )
        torch.cuda.synchronize(device)
    finally:
        unfreeze_kernel_resolution()

    expected = compressed_sparse_mla_reference(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        extra_k_cache=indexed_cache,
        extra_indices=indexed_indices_b,
        extra_topk_lengths=indexed_lengths,
        extra_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        sm_scale=_SM_SCALE,
    )
    max_abs = (actual.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        actual.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_accepts_row_shared_page_table(monkeypatch) -> None:
    device = require_sm120()
    clear_mla_caches()

    # ESCAPE HATCH (mapped indexed_page_table): passes a row-shared mapped extra-cache
    # page table, genuinely-upstream-unsupported on the unified SM120 backend
    # (raw-slot-id only; unified correctly RAISEs). Pin the legacy escape hatch so this
    # exercises the legacy split internals it was written for, post P10 default-flip.
    monkeypatch.setenv("B12X_MLA_SM120_UNIFIED", "0")

    rows = 3
    q = _make_q(rows=rows, seed=186, device=device)
    swa_cache = torch.empty(
        (0, compressed_mla_page_nbytes(COMPRESSED_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    indexed_cache = _make_cache(
        tokens=128,
        page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        seed=187,
        device=device,
    )
    swa_indices = torch.empty((rows, 0), dtype=torch.int32, device=device)
    swa_lengths = torch.zeros((rows,), dtype=torch.int32, device=device)
    indexed_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
        ],
        dtype=torch.int32,
        device=device,
    )
    indexed_lengths = torch.full(
        (rows,), indexed_indices.shape[1], dtype=torch.int32, device=device
    )
    indexed_page_table = (
        torch.arange(2, dtype=torch.int32, device=device).unsqueeze(0).expand(rows, -1)
    )
    workspace = _make_workspace(
        device=device,
        rows=rows,
        topk=indexed_indices.shape[1],
        max_kv_rows=rows * indexed_indices.shape[1],
        use_cuda_graph=True,
    )

    actual = compressed_mla_decode_forward(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices,
        indexed_topk_lengths=indexed_lengths,
        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        indexed_page_table=indexed_page_table,
        workspace=workspace,
        sm_scale=_SM_SCALE,
    )
    torch.cuda.synchronize(device)

    expected = compressed_sparse_mla_reference(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        extra_k_cache=indexed_cache,
        extra_indices=indexed_indices,
        extra_topk_lengths=indexed_lengths,
        extra_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        sm_scale=_SM_SCALE,
    )
    max_abs = (actual.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        actual.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_live_rows_do_not_resolve_new_split_forward_kernel(monkeypatch) -> None:
    device = require_sm120()
    clear_mla_caches()

    # ESCAPE HATCH (mapped indexed_page_table): passes a mapped extra-cache page table,
    # genuinely-upstream-unsupported on the unified SM120 backend (raw-slot-id only;
    # unified correctly RAISEs). Pin the legacy escape hatch so this exercises the
    # legacy split internals it was written for, post P10 default-flip.
    monkeypatch.setenv("B12X_MLA_SM120_UNIFIED", "0")

    workspace = B12XAttentionWorkspace.for_contract(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=_LOCAL_Q_HEADS,
        head_dim=_COMPRESSED_HEAD_DIM,
        v_head_dim=_COMPRESSED_HEAD_DIM,
        topk=8,
        max_total_q=3,
        max_batch=3,
        max_kv_rows=8,
        use_cuda_graph=False,
    )
    indexed_cache = _make_cache(
        tokens=64,
        page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        seed=192,
        device=device,
    )
    swa_cache = torch.empty(
        (0, compressed_mla_page_nbytes(COMPRESSED_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )

    q3 = _make_q(rows=3, seed=193, device=device)
    swa_indices3 = torch.empty((3, 0), dtype=torch.int32, device=device)
    swa_lengths3 = torch.zeros((3,), dtype=torch.int32, device=device)
    indexed_indices3 = torch.arange(8, dtype=torch.int32, device=device).repeat(3, 1)
    indexed_lengths3 = torch.full((3,), 8, dtype=torch.int32, device=device)
    page_table3 = torch.arange(2, dtype=torch.int32, device=device).repeat(3, 1)
    compressed_mla_decode_forward(
        q_all=q3,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices3,
        swa_topk_lengths=swa_lengths3,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices3,
        indexed_topk_lengths=indexed_lengths3,
        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        indexed_page_table=page_table3,
        workspace=workspace,
        sm_scale=_SM_SCALE,
    )
    torch.cuda.synchronize(device)

    q1 = _make_q(rows=1, seed=194, device=device)
    swa_indices1 = torch.empty((1, 0), dtype=torch.int32, device=device)
    swa_lengths1 = torch.zeros((1,), dtype=torch.int32, device=device)
    indexed_indices1 = torch.tensor(
        [[0, 16, 17, 32, 33, 47, 48, 63]],
        dtype=torch.int32,
        device=device,
    )
    indexed_lengths1 = torch.tensor([8], dtype=torch.int32, device=device)
    page_table1 = torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0)

    freeze_kernel_resolution("compressed MLA live rows must be runtime")
    try:
        actual = compressed_mla_decode_forward(
            q_all=q1,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices1,
            swa_topk_lengths=swa_lengths1,
            indexed_k_cache=indexed_cache,
            indexed_indices=indexed_indices1,
            indexed_topk_lengths=indexed_lengths1,
            indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
            indexed_page_table=page_table1,
            workspace=workspace,
            sm_scale=_SM_SCALE,
        )
        torch.cuda.synchronize(device)
    finally:
        unfreeze_kernel_resolution()

    expected = compressed_sparse_mla_reference(
        q1,
        swa_cache,
        swa_indices1,
        swa_lengths1,
        extra_k_cache=indexed_cache,
        extra_indices=indexed_indices1,
        extra_topk_lengths=indexed_lengths1,
        extra_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        sm_scale=_SM_SCALE,
    )
    max_abs = (actual.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        actual.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995
