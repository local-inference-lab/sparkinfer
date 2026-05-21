from __future__ import annotations

import math

import torch

from b12x.integration.mla import (
    B12XAttentionWorkspace,
    COMPRESSED_MLA_C128_PAGE_SIZE,
    COMPRESSED_MLA_C4_PAGE_SIZE,
    COMPRESSED_MLA_SWA_PAGE_SIZE,
    clear_mla_caches,
    compressed_mla_decode_forward,
    compressed_mla_page_nbytes,
    compressed_sparse_mla_reference,
    gather_compressed_mla_kv_cache_reference,
    pack_compressed_mla_kv_cache_reference,
    prepare_compressed_mla_core_inputs,
    sparse_mla_reference,
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
) -> B12XAttentionWorkspace:
    workspace_factory = (
        B12XAttentionWorkspace.for_contract
        if torch.device(device).type == "cpu"
        else B12XAttentionWorkspace.for_fixed_capacity
    )
    return workspace_factory(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=_LOCAL_Q_HEADS,
        head_dim=_SHARED_CORE_HEAD_DIM,
        v_head_dim=_SHARED_CORE_V_HEAD_DIM,
        topk=topk,
        max_total_q=rows,
        max_batch=rows,
        max_kv_rows=max_kv_rows,
        use_cuda_graph=use_cuda_graph,
    )


def _make_cache(
    *,
    tokens: int,
    page_size: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    k_nope = torch.randn((tokens, 448), generator=gen, dtype=torch.float32) * 0.05
    k_rope = torch.randn((tokens, 64), generator=gen, dtype=torch.float32) * 0.05
    return pack_compressed_mla_kv_cache_reference(
        k_nope.to(device=device),
        k_rope.to(dtype=torch.bfloat16, device=device),
        page_size=page_size,
    )


def _make_q(*, rows: int, seed: int, device: torch.device | str) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q = torch.randn((rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM), generator=gen, dtype=torch.float32) * 0.04
    return q.to(dtype=torch.bfloat16, device=device)


def test_compressed_mla_page_byte_widths_match_padded_layout() -> None:
    assert compressed_mla_page_nbytes(COMPRESSED_MLA_SWA_PAGE_SIZE) == 74880
    assert compressed_mla_page_nbytes(COMPRESSED_MLA_C4_PAGE_SIZE) == 37440
    assert compressed_mla_page_nbytes(COMPRESSED_MLA_C128_PAGE_SIZE) == 1728


def test_compressed_mla_reference_pack_gathers_across_padded_pages() -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(31)

    for page_size in (COMPRESSED_MLA_C4_PAGE_SIZE, COMPRESSED_MLA_C128_PAGE_SIZE):
        tokens = page_size * 2 + 1
        k_nope = torch.randn((tokens, 448), generator=gen, dtype=torch.float32) * 0.05
        k_rope = (torch.randn((tokens, 64), generator=gen, dtype=torch.float32) * 0.05).to(torch.bfloat16)
        cache = pack_compressed_mla_kv_cache_reference(k_nope, k_rope, page_size=page_size)
        indices = torch.tensor([0, page_size - 1, page_size, page_size + 1, tokens - 1], dtype=torch.int32)

        gathered, _ = gather_compressed_mla_kv_cache_reference(cache, indices, page_size=page_size)
        expected_rope = k_rope[indices.to(torch.long)].float()
        assert torch.count_nonzero(gathered[2:]).item() > 0
        torch.testing.assert_close(gathered[:, 448:], expected_rope, atol=0, rtol=0)
        torch.testing.assert_close(gathered[:, :448], k_nope[indices.to(torch.long)], atol=0.01, rtol=0.12)


def test_compressed_mla_decode_uses_shared_core_contract_on_cpu() -> None:
    device = torch.device("cpu")
    q = _make_q(rows=2, seed=11, device=device)
    swa_cache = _make_cache(tokens=18, page_size=COMPRESSED_MLA_SWA_PAGE_SIZE, seed=12, device=device)
    indexed_cache = _make_cache(tokens=12, page_size=COMPRESSED_MLA_C4_PAGE_SIZE, seed=13, device=device)
    swa_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [8, 7, 6, 5, 4],
        ],
        dtype=torch.int32,
        device=device,
    )
    indexed_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [6, 7, 8, 9],
        ],
        dtype=torch.int32,
        device=device,
    )
    swa_lengths = torch.tensor([5, 3], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([2, 4], dtype=torch.int32, device=device)

    core = prepare_compressed_mla_core_inputs(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices,
        indexed_topk_lengths=indexed_lengths,
        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
    )
    workspace = _make_workspace(
        device=device,
        rows=q.shape[0],
        topk=core.page_table_1.shape[1],
        max_kv_rows=core.kv_cache.shape[0],
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
        workspace=workspace,
        sm_scale=_SM_SCALE,
    )
    expected_core = sparse_mla_reference(
        q_all=core.q_all,
        kv_cache=core.kv_cache,
        page_table_1=core.page_table_1,
        active_token_counts=core.nsa_cache_seqlens_int32,
        sm_scale=_SM_SCALE,
        v_head_dim=core.v_head_dim,
    )
    assert torch.equal(actual, expected_core)

    expected_algorithm = compressed_sparse_mla_reference(
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
    diff = (actual.float() - expected_algorithm.float()).abs()
    cos = torch.nn.functional.cosine_similarity(actual.float().reshape(-1), expected_algorithm.float().reshape(-1), dim=0)
    assert diff.max().item() <= 0.04
    assert cos.item() >= 0.999


def test_compressed_mla_decode_attention_sink_matches_reference_on_cpu() -> None:
    device = torch.device("cpu")
    q = _make_q(rows=2, seed=51, device=device)
    swa_cache = _make_cache(tokens=12, page_size=COMPRESSED_MLA_SWA_PAGE_SIZE, seed=52, device=device)
    indexed_cache = _make_cache(tokens=12, page_size=COMPRESSED_MLA_C4_PAGE_SIZE, seed=53, device=device)
    swa_indices = torch.tensor([[0, 1, 2], [6, 5, 4]], dtype=torch.int32, device=device)
    indexed_indices = torch.tensor([[0, 1, 2, 3], [6, 7, 8, 9]], dtype=torch.int32, device=device)
    swa_lengths = torch.tensor([3, 2], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([2, 4], dtype=torch.int32, device=device)
    attn_sink = torch.linspace(-0.3, 0.2, _LOCAL_Q_HEADS, dtype=torch.float32, device=device)

    workspace = _make_workspace(
        device=device,
        rows=q.shape[0],
        topk=swa_indices.shape[1] + indexed_indices.shape[1],
        max_kv_rows=q.shape[0] * (swa_indices.shape[1] + indexed_indices.shape[1]),
    )
    actual, actual_lse = compressed_mla_decode_forward(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices,
        indexed_topk_lengths=indexed_lengths,
        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        attn_sink=attn_sink,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        return_lse=True,
        lse_scale="natural",
    )

    expected, expected_lse = compressed_sparse_mla_reference(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        extra_k_cache=indexed_cache,
        extra_indices=indexed_indices,
        extra_topk_lengths=indexed_lengths,
        extra_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
        return_lse=True,
    )
    diff = (actual.float() - expected.float()).abs()
    cos = torch.nn.functional.cosine_similarity(actual.float().reshape(-1), expected.float().reshape(-1), dim=0)
    assert diff.max().item() <= 0.04
    assert cos.item() >= 0.999
    torch.testing.assert_close(actual_lse, expected_lse, atol=0.02, rtol=0.02)


def test_compressed_mla_clamp_to_one_negative_extra_is_ignored_on_cpu() -> None:
    device = torch.device("cpu")
    q = _make_q(rows=1, seed=61, device=device)
    swa_cache = _make_cache(tokens=12, page_size=COMPRESSED_MLA_SWA_PAGE_SIZE, seed=62, device=device)
    indexed_cache = _make_cache(tokens=4, page_size=COMPRESSED_MLA_C128_PAGE_SIZE, seed=63, device=device)
    swa_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
    indexed_indices = torch.tensor([[-1, 0, 1, 2]], dtype=torch.int32, device=device)
    swa_lengths = torch.tensor([3], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([1], dtype=torch.int32, device=device)

    core = prepare_compressed_mla_core_inputs(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices,
        indexed_topk_lengths=indexed_lengths,
        indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
    )
    assert int(core.nsa_cache_seqlens_int32[0].item()) == int(swa_lengths[0].item())

    workspace = _make_workspace(
        device=device,
        rows=q.shape[0],
        topk=swa_indices.shape[1] + indexed_indices.shape[1],
        max_kv_rows=q.shape[0] * (swa_indices.shape[1] + indexed_indices.shape[1]),
    )
    actual = compressed_mla_decode_forward(
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
    expected = compressed_sparse_mla_reference(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        sm_scale=_SM_SCALE,
    )
    diff = (actual.float() - expected.float()).abs()
    cos = torch.nn.functional.cosine_similarity(actual.float().reshape(-1), expected.float().reshape(-1), dim=0)
    assert diff.max().item() <= 0.04
    assert cos.item() >= 0.999


@torch.inference_mode()
def test_compressed_mla_shared_core_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    q = _make_q(rows=1, seed=21, device=device)
    swa_cache = _make_cache(tokens=32, page_size=COMPRESSED_MLA_SWA_PAGE_SIZE, seed=22, device=device)
    indexed_cache = _make_cache(tokens=32, page_size=COMPRESSED_MLA_C128_PAGE_SIZE, seed=23, device=device)
    swa_indices = torch.arange(16, dtype=torch.int32, device=device).unsqueeze(0)
    indexed_indices = torch.arange(16, dtype=torch.int32, device=device).unsqueeze(0)
    swa_lengths = torch.tensor([11], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([7], dtype=torch.int32, device=device)
    attn_sink = torch.nn.Parameter(
        torch.linspace(-0.1, 0.1, _LOCAL_Q_HEADS, dtype=torch.float32, device=device)
    )
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
        extra_k_cache=indexed_cache,
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
