from __future__ import annotations

import inspect
import math

import torch

import b12x.attention.mla.compressed_api as compressed_api_impl
from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.attention.workspace import (
    B12XAttentionArena,
    B12XAttentionArenaCaps,
    B12XAttentionWorkspace,
)
from b12x.attention.mla.compressed_reference import (
    COMPRESSED_MLA_C128_PAGE_SIZE,
    COMPRESSED_MLA_C4_PAGE_SIZE,
    COMPRESSED_MLA_SWA_PAGE_SIZE,
    compressed_mla_page_nbytes,
    compressed_sparse_mla_reference,
    gather_compressed_mla_kv_cache_reference,
    pack_compressed_mla_kv_cache_reference,
)
from b12x.integration.mla import (
    clear_mla_caches,
    compressed_mla_decode_forward,
    compressed_mla_split_chunks_for_contract,
)

from .helpers import require_sm120


_COMPRESSED_HEAD_DIM = 512
_SHARED_CORE_HEAD_DIM = 576
_SHARED_CORE_V_HEAD_DIM = 512
_LOCAL_Q_HEADS = 32
_SM_SCALE = 1.0 / math.sqrt(_COMPRESSED_HEAD_DIM)


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
    k_nope = torch.randn((tokens, 448), generator=gen, dtype=torch.float32, device=device) * 0.05
    k_rope = torch.randn((tokens, 64), generator=gen, dtype=torch.float32, device=device) * 0.05
    return pack_compressed_mla_kv_cache_reference(
        k_nope,
        k_rope.to(dtype=torch.bfloat16),
        page_size=page_size,
    )


def _make_q(*, rows: int, seed: int, device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn(
        (rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    ) * 0.04
    return q.to(dtype=torch.bfloat16)


def test_compressed_mla_page_byte_widths_match_padded_layout() -> None:
    assert compressed_mla_page_nbytes(COMPRESSED_MLA_SWA_PAGE_SIZE) == 74880
    assert compressed_mla_page_nbytes(COMPRESSED_MLA_C4_PAGE_SIZE) == 37440
    assert compressed_mla_page_nbytes(COMPRESSED_MLA_C128_PAGE_SIZE) == 1728


def test_compressed_mla_decode_does_not_pin_flash_tp2_heads_by_default() -> None:
    signature = inspect.signature(compressed_mla_decode_forward)
    assert signature.parameters["expected_num_q_heads"].default is None


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
        k_nope = torch.randn((tokens, 448), generator=gen, dtype=torch.float32, device=device) * 0.05
        k_rope = (
            torch.randn((tokens, 64), generator=gen, dtype=torch.float32, device=device) * 0.05
        ).to(torch.bfloat16)
        cache = pack_compressed_mla_kv_cache_reference(k_nope, k_rope, page_size=page_size)
        indices = torch.tensor(
            [0, page_size - 1, page_size, page_size + 1, tokens - 1],
            dtype=torch.int32,
            device=device,
        )

        gathered, _ = gather_compressed_mla_kv_cache_reference(cache, indices, page_size=page_size)
        expected_rope = k_rope[indices.to(torch.long)].float()
        assert torch.count_nonzero(gathered[2:]).item() > 0
        torch.testing.assert_close(gathered[:, 448:], expected_rope, atol=0, rtol=0)
        torch.testing.assert_close(gathered[:, :448], k_nope[indices.to(torch.long)], atol=0.01, rtol=0.12)


@torch.inference_mode()
def test_compressed_mla_fixed_workspace_split_plan_uses_contract_not_live_shape(monkeypatch) -> None:
    device = require_sm120()
    clear_mla_caches()

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
        (1, compressed_mla_page_nbytes(COMPRESSED_MLA_SWA_PAGE_SIZE)),
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
    indexed_indices = torch.full((live_rows, live_width), -1, dtype=torch.int32, device=device)
    indexed_lengths = torch.zeros((live_rows,), dtype=torch.int32, device=device)
    indexed_page_table = torch.full(
        (live_rows, page_table_width),
        -1,
        dtype=torch.int32,
        device=device,
    )

    calls: dict[str, int | bool] = {}

    def fake_forward(**kwargs) -> None:
        calls["launch_num_chunks"] = int(kwargs["launch_num_chunks"])
        calls["direct_output"] = bool(kwargs["direct_output"])
        calls["swa_cache_is_u8"] = kwargs["swa_k_cache"].dtype == torch.uint8
        assert kwargs["q_all"].shape[0] == contract_rows
        assert kwargs["swa_indices"].shape[0] == contract_rows
        assert kwargs["swa_lengths"].shape == (contract_rows,)
        assert kwargs["indexed_indices"].shape == (contract_rows, live_width)
        assert kwargs["indexed_lengths"].shape == (contract_rows,)
        assert kwargs["indexed_page_table"].shape == (contract_rows, page_table_width)

    def fake_merge(**kwargs) -> None:
        assert kwargs["output"].shape[0] == contract_rows
        kwargs["output"].zero_()
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
    swa_cache_bytes = _make_cache(tokens=32, page_size=COMPRESSED_MLA_SWA_PAGE_SIZE, seed=22, device=device)
    indexed_cache_bytes = _make_cache(tokens=32, page_size=COMPRESSED_MLA_C128_PAGE_SIZE, seed=23, device=device)
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
    cos = torch.nn.functional.cosine_similarity(captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0)
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_c128_pv_row_swizzle_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    width = 32
    q = torch.zeros((1, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM), dtype=torch.bfloat16, device=device)
    k_nope = torch.zeros((width, 448), dtype=torch.bfloat16, device=device)
    k_nope[20, 0] = 1
    k_rope = torch.zeros((width, 64), dtype=torch.bfloat16, device=device)
    swa_cache = torch.empty(
        (0, compressed_mla_page_nbytes(COMPRESSED_MLA_SWA_PAGE_SIZE)),
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
    attn_sink = torch.linspace(-0.08, 0.12, _LOCAL_Q_HEADS, dtype=torch.float32, device=device)
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
    cos = torch.nn.functional.cosine_similarity(captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0)
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_prefill_swa_only_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    rows = 8
    width = 8
    q = _make_q(rows=rows, seed=81, device=device)
    swa_cache = _make_cache(tokens=32, page_size=COMPRESSED_MLA_SWA_PAGE_SIZE, seed=82, device=device)
    swa_indices = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    swa_lengths = torch.empty((rows,), dtype=torch.int32, device=device)
    for row in range(rows):
        length = min(width, row + 1)
        swa_indices[row, :length] = torch.arange(row, row - length, -1, dtype=torch.int32, device=device)
        swa_lengths[row] = length
    attn_sink = torch.linspace(-0.2, 0.15, _LOCAL_Q_HEADS, dtype=torch.float32, device=device)
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
    cos = torch.nn.functional.cosine_similarity(captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0)
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_clamp_to_one_negative_extra_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    q = _make_q(rows=1, seed=71, device=device)
    swa_cache = _make_cache(tokens=32, page_size=COMPRESSED_MLA_SWA_PAGE_SIZE, seed=72, device=device)
    indexed_cache = _make_cache(tokens=4, page_size=COMPRESSED_MLA_C128_PAGE_SIZE, seed=73, device=device)
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
    cos = torch.nn.functional.cosine_similarity(captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0)
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_mla_page_table_width_does_not_resolve_new_kernel() -> None:
    device = require_sm120()
    clear_mla_caches()

    q = _make_q(rows=1, seed=181, device=device)
    swa_cache = torch.empty(
        (0, compressed_mla_page_nbytes(COMPRESSED_MLA_SWA_PAGE_SIZE)),
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
