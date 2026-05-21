from __future__ import annotations

import pytest
import torch

from b12x.attention.nsa_indexer.kernel import (
    PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT,
    _split_index_k_cache_runtime_views,
)
from b12x.attention.nsa_indexer.reference import (
    sparse_nsa_extend_logits_reference,
    sparse_nsa_paged_logits_reference,
)
from b12x.integration.nsa_indexer import (
    NSAIndexerExtendLogitsMetadata,
    NSAIndexerPagedDecodeMetadata,
    clear_nsa_indexer_caches,
    get_paged_mqa_logits_metadata,
    pack_nsa_index_k_cache_reference,
    resolve_sparse_nsa_extend_prefill_block_k,
    sparse_nsa_index_decode_logits_paged,
    sparse_nsa_index_extend_logits,
    sparse_nsa_index_extend_tiled_topk,
    uses_paged_mqa_schedule_metadata,
)


_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _make_real_page_table(
    *,
    page_starts: list[int],
    seqlens: list[int],
    width_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    real_page_table = torch.full(
        (len(seqlens), width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row_idx, (page_start, seq_len) in enumerate(zip(page_starts, seqlens, strict=True)):
        block_count = (int(seq_len) + 63) // 64
        if block_count:
            real_page_table[row_idx, :block_count] = torch.arange(
                page_start,
                page_start + block_count,
                dtype=torch.int32,
                device=device,
            )
    return real_page_table


def _quantize_rows_to_kv_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = k.abs().amax(dim=1) / _FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (k / scale.unsqueeze(1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return quant, scale.to(torch.float32)


def _assert_logits_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def _paged_mqa_schedule_reference(
    context_lens: torch.Tensor,
    *,
    block_kv: int,
    num_sms: int,
) -> torch.Tensor:
    rows = context_lens[:, -1].tolist() if context_lens.ndim == 2 else context_lens.tolist()
    split_kv = block_kv * PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT
    prefix_sum: list[int] = []
    total = 0
    for row in rows:
        total += max((int(row) + split_kv - 1) // split_kv, 0)
        prefix_sum.append(total)

    q, r = divmod(total, num_sms)
    out: list[list[int]] = []
    for sm_idx in range(num_sms + 1):
        seg_start = sm_idx * q + min(sm_idx, r)
        q_idx = 0
        while q_idx < len(prefix_sum) and prefix_sum[q_idx] <= seg_start:
            q_idx += 1
        kv_split_idx = seg_start if q_idx == 0 else seg_start - prefix_sum[q_idx - 1]
        out.append([q_idx, kv_split_idx])
    return torch.tensor(out, dtype=torch.int32)


def test_sparse_nsa_index_runtime_views_preserve_page_stride() -> None:
    device = torch.device("cpu")
    page_count = 3
    page_bytes = 64 * (128 + 4)
    index_k_cache = torch.arange(page_count * page_bytes, dtype=torch.uint8, device=device).view(
        page_count,
        page_bytes,
    )

    quant, scales = _split_index_k_cache_runtime_views(index_k_cache)

    assert quant.shape == (page_count, 64, 128)
    assert quant.stride() == (page_bytes, 128, 1)
    assert quant.untyped_storage().data_ptr() == index_k_cache.untyped_storage().data_ptr()
    assert quant[1, 2, 127].item() == index_k_cache[1, 2 * 128 + 127].item()

    data_bytes = 64 * 128
    assert scales.shape == (page_count, 64)
    assert scales.stride() == (page_bytes // 4, 1)
    assert scales.untyped_storage().data_ptr() == index_k_cache.untyped_storage().data_ptr()
    scale_bytes = scales.view(torch.uint8).view(page_count, 64, 4)
    assert scale_bytes[1, 2, 0].item() == index_k_cache[1, data_bytes + 2 * 4].item()


def test_get_paged_mqa_logits_metadata_matches_deepgemm_partitioning() -> None:
    context_lens_1d = torch.tensor([0, 64, 4096, 4097, 16384], dtype=torch.int32)
    schedule_1d = get_paged_mqa_logits_metadata(context_lens_1d, 64, 5)
    expected_1d = _paged_mqa_schedule_reference(context_lens_1d, block_kv=64, num_sms=5)
    assert schedule_1d.shape == (6, 2)
    assert schedule_1d.dtype == torch.int32
    assert schedule_1d.is_contiguous()
    assert torch.equal(schedule_1d.cpu(), expected_1d)

    context_lens_2d = torch.tensor([[64, 65], [0, 8192], [128, 129]], dtype=torch.int32)
    schedule_2d = get_paged_mqa_logits_metadata(context_lens_2d, 64, 7)
    expected_2d = _paged_mqa_schedule_reference(context_lens_2d, block_kv=64, num_sms=7)
    assert schedule_2d.shape == (8, 2)
    assert schedule_2d.dtype == torch.int32
    assert schedule_2d.is_contiguous()
    assert torch.equal(schedule_2d.cpu(), expected_2d)


def test_uses_paged_mqa_schedule_metadata_only_for_long_rows() -> None:
    assert not uses_paged_mqa_schedule_metadata(q_rows=0, max_pages=2048)
    assert not uses_paged_mqa_schedule_metadata(q_rows=1, max_pages=128)
    assert not uses_paged_mqa_schedule_metadata(q_rows=2, max_pages=512)
    assert not uses_paged_mqa_schedule_metadata(q_rows=9, max_pages=2048)
    assert uses_paged_mqa_schedule_metadata(q_rows=1, max_pages=2048)
    assert uses_paged_mqa_schedule_metadata(q_rows=2, max_pages=2048)
    assert uses_paged_mqa_schedule_metadata(q_rows=8, max_pages=2048)


def test_sparse_nsa_extend_prefill_block_k_auto_targets_long_bs1_prefill(monkeypatch) -> None:
    monkeypatch.delenv("B12X_NSA_EXTEND_PREFILL_THRESHOLD", raising=False)
    monkeypatch.delenv("B12X_NSA_EXTEND_PREFILL_BLOCK_K", raising=False)

    assert (
        resolve_sparse_nsa_extend_prefill_block_k(
            valid_q_rows=2048,
            k_rows=65536,
            num_heads=32,
        )
        == 512
    )
    assert (
        resolve_sparse_nsa_extend_prefill_block_k(
            valid_q_rows=512,
            k_rows=65536,
            num_heads=64,
        )
        == 256
    )
    assert (
        resolve_sparse_nsa_extend_prefill_block_k(
            valid_q_rows=128,
            k_rows=65536,
            num_heads=64,
        )
        is None
    )


def test_sparse_nsa_extend_prefill_block_k_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("B12X_NSA_EXTEND_PREFILL_THRESHOLD", raising=False)
    monkeypatch.setenv("B12X_NSA_EXTEND_PREFILL_BLOCK_K", "256")
    assert (
        resolve_sparse_nsa_extend_prefill_block_k(
            valid_q_rows=2048,
            k_rows=65536,
            num_heads=32,
        )
        == 256
    )

    monkeypatch.setenv("B12X_NSA_EXTEND_PREFILL_BLOCK_K", "512")
    assert (
        resolve_sparse_nsa_extend_prefill_block_k(
            valid_q_rows=2048,
            k_rows=65536,
            num_heads=32,
        )
        == 512
    )
    with pytest.raises(ValueError, match="unsupported"):
        resolve_sparse_nsa_extend_prefill_block_k(
            valid_q_rows=512,
            k_rows=65536,
            num_heads=64,
        )

    monkeypatch.setenv("B12X_NSA_EXTEND_PREFILL_BLOCK_K", "bad")
    with pytest.raises(ValueError, match="auto, 256, or 512"):
        resolve_sparse_nsa_extend_prefill_block_k(
            valid_q_rows=2048,
            k_rows=65536,
            num_heads=64,
        )


def test_sparse_nsa_index_decode_logits_paged_cpu_hard_fails_without_fallback() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_100)

    q_rows = 3
    num_heads = 4
    page_starts = [1, 3, 5]
    width_blocks = 3
    num_tokens = (max(page_starts) + width_blocks) * 64
    seqlens = torch.tensor([65, 128, 150], dtype=torch.int32, device=device)
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=device,
    )
    q_fp8 = (
        torch.randn((q_rows + 1, num_heads, 128), generator=gen, dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows + 1, num_heads), generator=gen, dtype=torch.float32, device=device)
    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32, device=device) / 3
    )

    with pytest.raises(NotImplementedError, match="refusing to run the reference fallback"):
        sparse_nsa_index_decode_logits_paged(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=NSAIndexerPagedDecodeMetadata(
                real_page_table=real_page_table,
                cache_seqlens_int32=seqlens,
                paged_mqa_schedule_metadata=get_paged_mqa_logits_metadata(seqlens, 64, 8),
            ),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for paged kernel coverage")
def test_sparse_nsa_index_decode_logits_paged_cuda_kernel_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_101)

    q_rows = 4
    num_heads = 8
    page_starts = [2, 8, 12, 16]
    num_tokens = (max(page_starts) + 3) * 64
    seqlens = torch.tensor([65, 96, 128, 191], dtype=torch.int32, device=device)
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=3,
        device=device,
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    )

    actual = sparse_nsa_index_decode_logits_paged(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerPagedDecodeMetadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=seqlens,
            paged_mqa_schedule_metadata=get_paged_mqa_logits_metadata(seqlens, 64, 8),
        ),
    )
    expected = sparse_nsa_paged_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )

    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for paged kernel coverage")
def test_sparse_nsa_index_decode_logits_paged_cuda_schedule_kernel_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_111)

    q_rows = 2
    num_heads = 8
    width_blocks = 1024
    page_starts = [2, 1100]
    num_tokens = (max(page_starts) + 40) * 64
    seqlens = torch.tensor([2048, 2304], dtype=torch.int32, device=device)
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=device,
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    )

    actual = sparse_nsa_index_decode_logits_paged(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerPagedDecodeMetadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=seqlens,
            paged_mqa_schedule_metadata=get_paged_mqa_logits_metadata(seqlens, 64, 8),
        ),
    )
    expected = sparse_nsa_paged_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )

    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture coverage")
def test_sparse_nsa_index_decode_logits_paged_cuda_graph_replay_tracks_live_width_without_stale_output() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_102)

    rows = 2
    num_heads = 8
    num_tokens = 1024
    graph_width_blocks = 4
    live_width_blocks = 3

    q_fp8 = (
        torch.randn((rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    )
    live_real_page_table0 = _make_real_page_table(
        page_starts=[2, 8],
        seqlens=[150, 129],
        width_blocks=live_width_blocks,
        device=device,
    )
    live_real_page_table1 = _make_real_page_table(
        page_starts=[4, 9],
        seqlens=[65, 64],
        width_blocks=live_width_blocks,
        device=device,
    )
    graph_real_page_table = torch.full(
        (rows, graph_width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    graph_schedule_metadata = torch.empty((9, 2), dtype=torch.int32, device=device)

    def prepare(page_table: torch.Tensor, seqlens: torch.Tensor) -> None:
        graph_real_page_table[:, :live_width_blocks].copy_(page_table)
        graph_seqlens.copy_(seqlens)
        get_paged_mqa_logits_metadata(graph_seqlens, 64, 8, out=graph_schedule_metadata)

    metadata = NSAIndexerPagedDecodeMetadata(
        real_page_table=graph_real_page_table,
        cache_seqlens_int32=graph_seqlens,
        paged_mqa_schedule_metadata=graph_schedule_metadata,
    )

    clear_nsa_indexer_caches()
    prepare(live_real_page_table0, torch.tensor([150, 129], dtype=torch.int32, device=device))
    sparse_nsa_index_decode_logits_paged(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = sparse_nsa_index_decode_logits_paged(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    actual0 = captured_out.clone()
    expected0 = sparse_nsa_paged_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
    )
    _assert_logits_close(actual0, expected0)

    prepare(live_real_page_table1, torch.tensor([65, 64], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize(device)
    actual1 = captured_out.clone()
    expected1 = sparse_nsa_paged_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
    )
    _assert_logits_close(actual1, expected1)
    assert torch.isneginf(actual1[:, 65:]).all()


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []),
)
def test_sparse_nsa_index_extend_logits_matches_reference(device: torch.device) -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_103)

    q_rows = 5
    num_heads = 3
    k_rows = 64
    q_fp8 = (
        torch.randn((q_rows + 1, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows + 1, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor([0, 5, 12, 12, 40], dtype=torch.int32, device=device)
    k_end = torch.tensor([8, 16, 20, 12, 55], dtype=torch.int32, device=device)

    actual = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=NSAIndexerExtendLogitsMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    expected = sparse_nsa_extend_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    _assert_logits_close(actual, expected)
    assert torch.isneginf(actual[-1]).all()


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []),
)
def test_sparse_nsa_index_extend_logits_matches_reference_for_sparse_tile_ranges(
    device: torch.device,
) -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_104)

    q_rows = 40
    num_heads = 4
    k_rows = 130
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor(([0] * 32) + ([128] * 8), dtype=torch.int32, device=device)
    k_end = torch.tensor(([32] * 32) + ([130] * 8), dtype=torch.int32, device=device)

    actual = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=NSAIndexerExtendLogitsMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    expected = sparse_nsa_extend_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    _assert_logits_close(actual, expected)
    assert torch.isneginf(actual[:32, 32:]).all()
    assert torch.isneginf(actual[32:, :128]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for extend kernel coverage")
@pytest.mark.parametrize("num_heads", [16, 32, 64])
def test_sparse_nsa_index_extend_logits_cuda_matches_reference_for_large_head_counts(
    num_heads: int,
) -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_105 + num_heads)

    q_rows = 8
    k_rows = 257
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor(
        [0, 192, 16, 128, 32, 224, 0, 64],
        dtype=torch.int32,
        device=device,
    )
    k_end = torch.tensor(
        [33, 257, 80, 192, 96, 257, 1, 65],
        dtype=torch.int32,
        device=device,
    )

    actual = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=NSAIndexerExtendLogitsMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    expected = sparse_nsa_extend_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)
    assert torch.isneginf(actual[0, 33:192]).all()
    assert torch.isneginf(actual[1, :192]).all()
    assert torch.isneginf(actual[6, 1:]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for extend kernel coverage")
@pytest.mark.parametrize(
    "q_rows, k_rows",
    [
        (256, 4096),   # shortest prefill q, >2048 k — crosses the K-tile budget.
        (512, 8192),   # multi-Q-tile prefill over long context.
        (1024, 3072),  # many Q-tiles, mid-length K.
    ],
)
def test_sparse_nsa_index_extend_logits_cuda_matches_reference_for_long_prefill(
    q_rows: int, k_rows: int
) -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_200 + q_rows * 17 + k_rows)

    num_heads = 64
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)

    # Causal ragged ranges: row q sees k ∈ [0, q+1). Spans the full k_rows range
    # for tail rows and exercises per-row sparse-range liveness for early rows.
    positions = torch.arange(q_rows, dtype=torch.int32, device=device)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.clamp(positions + 1, max=k_rows).to(torch.int32)

    actual = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=NSAIndexerExtendLogitsMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    actual_no_fill = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=NSAIndexerExtendLogitsMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
        preinitialize_invalid_logits=False,
    )
    expected = sparse_nsa_extend_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)
    _assert_logits_close(actual_no_fill, expected)
    # Out-of-range positions must stay -inf all the way out to k_rows.
    for q in (0, q_rows // 2, q_rows - 1):
        ke = min(q + 1, k_rows)
        if ke < k_rows:
            assert torch.isneginf(actual[q, ke:]).all(), f"row {q} leaked non-neginf beyond k_end={ke}"
            assert torch.isneginf(actual_no_fill[q, ke:]).all(), (
                f"no-fill row {q} leaked non-neginf beyond k_end={ke}"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for extend kernel coverage")
@pytest.mark.parametrize(
    "q_rows, k_rows",
    [
        (256, 3072),   # slightly over the old hardcoded 2048 K-tile budget.
        (256, 8192),   # well past it — exercises full K-grid scaling.
    ],
)
def test_sparse_nsa_index_extend_logits_cuda_matches_reference_for_dense_long_prefill(
    q_rows: int, k_rows: int
) -> None:
    """Dense (non-causal) long-K prefill: every q row sees every k row.

    Exercises the K-tile grid scaling — with a fixed K_GROUPS=4 launcher and a fixed
    inner K-tile loop of 4, the kernel can only cover 2048 K-rows per Q-tile; anything
    beyond that silently stays at -inf and this test catches it.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_400 + q_rows * 31 + k_rows)

    num_heads = 64
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.full((q_rows,), k_rows, dtype=torch.int32, device=device)

    actual = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=NSAIndexerExtendLogitsMetadata(k_start=k_start, k_end=k_end),
    )
    expected = sparse_nsa_extend_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )
    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)
    # No position should have silently fallen back to -inf.
    assert torch.isfinite(actual).all(), "kernel left finite positions as -inf"


def test_sparse_nsa_index_extend_tiled_topk_cpu_matches_reference() -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_610)

    q_rows = 4
    num_heads = 3
    k_rows = 17
    topk = 6
    q_fp8 = (torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32) / 2).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor([0, 2, 7, 16], dtype=torch.int32)
    k_end = torch.tensor([9, 12, 17, 17], dtype=torch.int32)
    metadata = NSAIndexerExtendLogitsMetadata(k_start=k_start, k_end=k_end)
    lengths = torch.empty((q_rows,), dtype=torch.int32)
    output_indices = torch.empty((q_rows, topk), dtype=torch.int32)

    actual = sparse_nsa_index_extend_tiled_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=metadata,
        topk=topk,
        lengths=lengths,
        output_indices=output_indices,
    )
    logits = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=metadata,
    )
    topk_pos = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :topk]
    topk_values = torch.gather(logits, 1, topk_pos)
    expected = torch.where(
        torch.isfinite(topk_values),
        topk_pos.to(torch.int32),
        torch.full_like(topk_pos, -1, dtype=torch.int32),
    )

    assert actual.data_ptr() == output_indices.data_ptr()
    assert torch.equal(actual, expected)
    assert torch.equal(lengths, k_end - k_start)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for tiled topk coverage")
def test_sparse_nsa_index_extend_tiled_topk_matches_scatter_logits(monkeypatch) -> None:
    monkeypatch.setenv("B12X_NSA_EXTEND_TOPK_SUPERTILE_K", "3072")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_620)

    q_rows = 256
    num_heads = 8
    k_rows = 4096
    topk = 2048
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.full((q_rows,), k_rows, dtype=torch.int32, device=device)
    metadata = NSAIndexerExtendLogitsMetadata(k_start=k_start, k_end=k_end)

    actual = sparse_nsa_index_extend_tiled_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=metadata,
        topk=topk,
    )
    logits = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=metadata,
        preinitialize_invalid_logits=False,
    )
    expected = torch.topk(logits, k=topk, dim=1, largest=True, sorted=False).indices.to(torch.int32)

    torch.cuda.synchronize(device)
    assert actual.shape == (q_rows, topk)
    assert torch.equal(torch.sort(actual, dim=1).values, torch.sort(expected, dim=1).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for BK512 prefill coverage")
def test_sparse_nsa_index_extend_logits_cuda_prefill512_sampled_logits(monkeypatch) -> None:
    monkeypatch.setenv("B12X_NSA_EXTEND_PREFILL_BLOCK_K", "512")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_512)

    q_rows = 1024
    k_rows = 32768
    num_heads = 32
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)

    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.zeros(q_rows, dtype=torch.int32, device=device)
    ranges = {
        0: (0, k_rows),
        31: (256, 1024),
        32: (512, 1536),
        255: (8192, 12288),
        512: (16384, k_rows),
        1023: (k_rows - 1024, k_rows),
    }
    for q_idx, (start, end) in ranges.items():
        k_start[q_idx] = start
        k_end[q_idx] = end

    actual = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=NSAIndexerExtendLogitsMetadata(k_start=k_start, k_end=k_end),
    )
    torch.cuda.synchronize(device)

    k_quant, k_scale = kv_fp8

    def assert_sampled_logits(q_idx: int, cols: list[int]) -> None:
        k_cols = torch.tensor(cols, dtype=torch.long, device=device)
        scores = torch.matmul(q_fp8[q_idx].to(torch.float32), k_quant[k_cols].to(torch.float32).T)
        expected = (torch.relu(scores) * weights[q_idx].unsqueeze(1)).sum(dim=0) * k_scale[k_cols]
        torch.testing.assert_close(actual[q_idx, k_cols], expected, atol=1e-4, rtol=1e-4)

    assert_sampled_logits(0, [0, 255, 256, 511, 512, 8191, 16384, 32767])
    assert_sampled_logits(31, [256, 511, 512, 1023])
    assert_sampled_logits(32, [512, 1024, 1535])
    assert_sampled_logits(255, [8192, 8193, 12287])
    assert_sampled_logits(512, [16384, 32767])
    assert_sampled_logits(1023, [31744, 32767])

    assert torch.isneginf(actual[31, torch.tensor([0, 1024], device=device)]).all()
    assert torch.isneginf(actual[32, torch.tensor([511, 1536], device=device)]).all()
    assert torch.isneginf(actual[255, torch.tensor([8191, 12288], device=device)]).all()
    assert torch.isneginf(actual[512, torch.tensor([0, 16383], device=device)]).all()
    assert torch.isneginf(actual[1023, torch.tensor([31743], device=device)]).all()
    assert torch.isneginf(actual[1]).all()
    assert torch.isneginf(actual[64]).all()
