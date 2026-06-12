from __future__ import annotations

import pytest
import torch

from b12x.attention.indexer import (
    IndexerContiguousMetadata,
    IndexerPagedDecodeMetadata,
    clear_indexer_caches,
    msa_contiguous_block_scores,
    msa_decode_query_positions,
    msa_paged_decode_block_scores,
    msa_q2k_indices_decode,
    msa_q2k_indices_prefill,
    quantize_msa_q_fp8,
)
from b12x.attention.indexer.msa_reference import (
    msa_contiguous_block_scores_reference,
    msa_paged_decode_block_scores_reference,
    msa_q2k_indices_reference,
)
from b12x.attention.indexer.reference import pack_index_k_cache_reference
from b12x.attention.indexer.scratch import (
    B12XIndexerContiguousScratchCaps,
    B12XIndexerPagedScratchCaps,
    plan_indexer_contiguous_scratch,
    plan_indexer_paged_scratch,
)


def _one_scratch(plan):
    (spec,) = plan.scratch_specs()
    return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def _make_real_page_table(
    *,
    rows: int,
    width_pages: int,
    device: torch.device,
) -> torch.Tensor:
    page_ids = torch.arange(width_pages, dtype=torch.int32, device=device)
    return page_ids.unsqueeze(0).expand(rows, width_pages).contiguous()


def _make_msa_decode_case(
    *,
    rows: int,
    heads: int,
    width_pages: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, IndexerPagedDecodeMetadata]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn((rows, heads, 128), generator=gen, dtype=torch.float32, device=device) / 3
    q_fp8, q_scale = quantize_msa_q_fp8(q)
    k_rows = max(width_pages * 64, 1)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32, device=device) / 3
    if rows and heads:
        # Ensure at least one finite block max is negative, which catches accidental ReLU.
        q0 = torch.ones((128,), dtype=torch.float32, device=device) / 4
        k0 = -torch.ones((128, 128), dtype=torch.float32, device=device) / 4
        q_fp8[:1, :1], q_scale[:1, :1] = quantize_msa_q_fp8(q0.view(1, 1, 128))
        k[:128].copy_(k0)
    index_k_cache = pack_index_k_cache_reference(k)
    real_page_table = _make_real_page_table(rows=rows, width_pages=width_pages, device=device)
    seqlens = torch.full((rows,), width_pages * 64, dtype=torch.int32, device=device)
    if rows >= 3:
        seqlens[1] = max(width_pages * 64 - 1, 0)
        seqlens[2] = min(width_pages * 64, 129)
    metadata = IndexerPagedDecodeMetadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
    )
    return q_fp8, q_scale, index_k_cache, metadata


def _quantize_rows_to_kv_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scale = k.abs().amax(dim=1) / fp8_max
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (k / scale.unsqueeze(1)).clamp(-fp8_max, fp8_max)
    return quant.to(torch.float8_e4m3fn), scale.to(torch.float32)


def _make_msa_contiguous_case(
    *,
    rows: int,
    heads: int,
    k_rows: int,
    device: torch.device,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    IndexerContiguousMetadata,
]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn((rows, heads, 128), generator=gen, dtype=torch.float32, device=device) / 3
    q_fp8, q_scale = quantize_msa_q_fp8(q)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32, device=device) / 3
    if rows and heads and k_rows >= 128:
        q0 = torch.ones((128,), dtype=torch.float32, device=device) / 4
        k[:128].fill_(-0.25)
        q_fp8[:1, :1], q_scale[:1, :1] = quantize_msa_q_fp8(q0.view(1, 1, 128))
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros((rows,), dtype=torch.int32, device=device)
    k_end = torch.arange(1, rows + 1, dtype=torch.int32, device=device).clamp(max=k_rows)
    if rows >= 3:
        k_start[2:] = 128
        k_end[2:] = torch.arange(129, 129 + rows - 2, dtype=torch.int32, device=device).clamp(max=k_rows)
    metadata = IndexerContiguousMetadata(k_start=k_start, k_end=k_end)
    return q_fp8, q_scale, kv_fp8, metadata


def test_msa_decode_query_positions() -> None:
    seqlens = torch.tensor([1, 128, 129], dtype=torch.int32)
    assert torch.equal(msa_decode_query_positions(seqlens), torch.tensor([0, 127, 128]))


def test_msa_paged_decode_block_scores_cpu_matches_reference() -> None:
    device = torch.device("cpu")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=3,
        heads=4,
        width_pages=5,
        device=device,
        seed=92_001,
    )

    actual = msa_paged_decode_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        metadata=metadata,
    )
    expected = msa_paged_decode_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
    )

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    finite = expected[0, 0].isfinite()
    assert torch.any(expected[0, 0, finite] < 0)


def test_msa_q2k_indices_decode_cpu_matches_reference_and_out() -> None:
    device = torch.device("cpu")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=2,
        heads=1,
        width_pages=5,
        device=device,
        seed=92_101,
    )
    out = torch.empty((1, 2, 6), dtype=torch.int32, device=device)

    actual = msa_q2k_indices_decode(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=6,
        out_indices=out,
    )
    expected = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        query_positions=metadata.cache_seqlens_int32 - 1,
        topk=6,
    )

    assert actual.data_ptr() == out.data_ptr()
    assert torch.equal(actual, expected)


def test_msa_paged_scratch_binding_owns_decode_outputs() -> None:
    device = torch.device("cpu")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=2,
        heads=4,
        width_pages=5,
        device=device,
        seed=92_155,
    )
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=4,
            max_q_rows=2,
            max_page_table_width=5,
            topk=16,
            score_mode="msa",
        )
    )
    binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
    )

    actual = msa_q2k_indices_decode(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        binding=binding,
    )
    expected = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        query_positions=metadata.cache_seqlens_int32 - 1,
    )

    assert actual.data_ptr() == binding.q2k_indices.data_ptr()
    assert binding.block_scores is not None
    assert binding.page_scores is not None
    assert binding.block_scores.shape == (4, 2, 3)
    assert binding.page_scores.shape == (4, 2, 6)
    assert torch.equal(actual, expected)


def test_msa_paged_binding_rejects_duplicate_outputs_and_non_msa_scratch() -> None:
    device = torch.device("cpu")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=2,
        heads=4,
        width_pages=5,
        device=device,
        seed=92_156,
    )
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=4,
            max_q_rows=2,
            max_page_table_width=5,
            topk=16,
            score_mode="msa",
        )
    )
    binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
    )
    out_scores = torch.empty((4, 2, 3), dtype=torch.float32, device=device)
    out_indices = torch.empty((4, 2, 16), dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="binding owns metadata"):
        msa_paged_decode_block_scores(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=index_k_cache,
            metadata=metadata,
            binding=binding,
        )
    with pytest.raises(ValueError, match="binding owns metadata and block-score buffers"):
        msa_paged_decode_block_scores(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=index_k_cache,
            out=out_scores,
            binding=binding,
        )
    with pytest.raises(ValueError, match="binding owns metadata and q2k output"):
        msa_q2k_indices_decode(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=index_k_cache,
            out_indices=out_indices,
            binding=binding,
        )

    nsa_plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=1,
            max_q_rows=2,
            max_page_table_width=5,
            topk=16,
        )
    )
    with pytest.raises(RuntimeError, match="score_mode='msa'"):
        nsa_plan.bind_msa(
            scratch=_one_scratch(nsa_plan),
            real_page_table=metadata.real_page_table,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
        )


def test_msa_contiguous_block_scores_cpu_matches_reference_and_prefill_indices() -> None:
    device = torch.device("cpu")
    q_fp8, q_scale, kv_fp8, metadata = _make_msa_contiguous_case(
        rows=5,
        heads=4,
        k_rows=384,
        device=device,
        seed=92_151,
    )

    actual_scores = msa_contiguous_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        metadata=metadata,
    )
    expected_scores = msa_contiguous_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )
    torch.testing.assert_close(actual_scores, expected_scores, atol=0.0, rtol=0.0)

    actual_indices = msa_q2k_indices_prefill(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        metadata=metadata,
        topk=6,
    )
    expected_indices = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        k_start=metadata.k_start,
        k_end=metadata.k_end,
        query_positions=metadata.k_end - 1,
        block_base=torch.div(metadata.k_start, 128, rounding_mode="floor"),
        topk=6,
    )
    assert torch.equal(actual_indices, expected_indices)


def test_msa_contiguous_scratch_binding_owns_prefill_outputs() -> None:
    device = torch.device("cpu")
    q_fp8, q_scale, kv_fp8, metadata = _make_msa_contiguous_case(
        rows=3,
        heads=4,
        k_rows=257,
        device=device,
        seed=92_255,
    )
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=4,
            max_q_rows=3,
            max_k_rows=257,
            topk=16,
            score_mode="msa",
        )
    )
    binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )

    actual = msa_q2k_indices_prefill(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        binding=binding,
    )
    expected = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        k_start=metadata.k_start,
        k_end=metadata.k_end,
        query_positions=metadata.k_end - 1,
        block_base=torch.div(metadata.k_start, 128, rounding_mode="floor"),
    )

    assert actual.data_ptr() == binding.q2k_indices.data_ptr()
    assert binding.block_scores is not None
    assert binding.block_scores.shape == (4, 3, 3)
    assert torch.equal(actual, expected)


def test_msa_contiguous_binding_rejects_duplicate_outputs_and_non_msa_scratch() -> None:
    device = torch.device("cpu")
    q_fp8, q_scale, kv_fp8, metadata = _make_msa_contiguous_case(
        rows=3,
        heads=4,
        k_rows=257,
        device=device,
        seed=92_256,
    )
    plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=4,
            max_q_rows=3,
            max_k_rows=257,
            topk=16,
            score_mode="msa",
        )
    )
    binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )
    out_scores = torch.empty((4, 3, 3), dtype=torch.float32, device=device)
    out_indices = torch.empty((4, 3, 16), dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="binding owns metadata"):
        msa_contiguous_block_scores(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            metadata=metadata,
            binding=binding,
        )
    with pytest.raises(ValueError, match="binding owns metadata and block-score buffers"):
        msa_contiguous_block_scores(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            out=out_scores,
            binding=binding,
        )
    with pytest.raises(ValueError, match="binding owns metadata and q2k output"):
        msa_q2k_indices_prefill(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            out_indices=out_indices,
            binding=binding,
        )

    nsa_plan = plan_indexer_contiguous_scratch(
        B12XIndexerContiguousScratchCaps(
            device=device,
            num_q_heads=1,
            max_q_rows=3,
            max_k_rows=257,
            topk=16,
        )
    )
    with pytest.raises(RuntimeError, match="score_mode='msa'"):
        nsa_plan.bind_msa(
            scratch=_one_scratch(nsa_plan),
            k_start=metadata.k_start,
            k_end=metadata.k_end,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for MSA decode scorer coverage",
)
@pytest.mark.parametrize(
    ("rows", "heads", "width_pages"),
    [
        (2, 1, 8),
        (1, 4, 1024),
        (4, 4, 1024),
    ],
)
def test_msa_paged_decode_block_scores_cuda_matches_reference(
    rows: int,
    heads: int,
    width_pages: int,
) -> None:
    clear_indexer_caches()
    device = torch.device("cuda")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=rows,
        heads=heads,
        width_pages=width_pages,
        device=device,
        seed=92_201 + rows * 17 + heads,
    )

    actual = msa_paged_decode_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        metadata=metadata,
    )
    expected = msa_paged_decode_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    finite = expected[0, 0].isfinite()
    assert torch.any(actual[0, 0, finite] < 0)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for MSA decode scorer coverage",
)
def test_msa_q2k_indices_decode_cuda_matches_reference() -> None:
    clear_indexer_caches()
    device = torch.device("cuda")
    q_fp8, q_scale, index_k_cache, metadata = _make_msa_decode_case(
        rows=4,
        heads=4,
        width_pages=9,
        device=device,
        seed=92_301,
    )

    actual = msa_q2k_indices_decode(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=8,
    )
    expected = msa_q2k_indices_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        query_positions=metadata.cache_seqlens_int32 - 1,
        topk=8,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for MSA contiguous scorer coverage",
)
@pytest.mark.parametrize("rows", [4, 257])
def test_msa_contiguous_block_scores_cuda_matches_reference(rows: int) -> None:
    clear_indexer_caches()
    device = torch.device("cuda")
    q_fp8, q_scale, kv_fp8, metadata = _make_msa_contiguous_case(
        rows=rows,
        heads=4,
        k_rows=384,
        device=device,
        seed=92_401 + rows,
    )

    actual = msa_contiguous_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        metadata=metadata,
    )
    expected = msa_contiguous_block_scores_reference(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert actual.shape[0] == 4
