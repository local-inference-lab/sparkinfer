from __future__ import annotations

import math

import pytest
import torch

from sparkinfer.attention.paged._scratch import (
    SPARKINFERPagedAttentionScratchCaps,
    plan_paged_attention_scratch,
)
from sparkinfer.attention.paged.planner import plan_decode_graph_capacity
from sparkinfer.attention.paged.reference import paged_attention_reference
from sparkinfer.attention.paged.traits import select_paged_forward_traits_from_plan

from tests._reference.helpers import require_sparkinfer


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _require_exact_laguna_device() -> torch.device:
    device = require_sparkinfer()
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("the exact Laguna KV128 specialization requires SM120")
    return device


def _make_laguna_graph_plan(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table_width: int,
):
    batch = int(q.shape[0])
    capacity = plan_decode_graph_capacity(
        device=q.device,
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        num_q_heads=int(q.shape[1]),
        num_kv_heads=int(k_cache.shape[2]),
        head_dim_qk=int(q.shape[2]),
        head_dim_vo=int(v_cache.shape[3]),
        page_size=int(k_cache.shape[1]),
        batch=batch,
        max_cache_page_count=page_table_width,
        force_split_kv=True,
    )
    scratch_plan = plan_paged_attention_scratch(
        SPARKINFERPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=int(q.shape[1]),
            num_kv_heads=int(k_cache.shape[2]),
            head_dim_qk=int(q.shape[2]),
            head_dim_vo=int(v_cache.shape[3]),
            page_size=int(k_cache.shape[1]),
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=capacity.max_work_items,
            max_partial_rows=capacity.max_partial_rows,
            num_cache_pages=int(k_cache.shape[0]),
            use_cuda_graph=True,
            copy_runtime_metadata=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=batch,
        total_q_capacity=batch,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
        force_split_kv=True,
    )
    assert scratch_plan.plan.split_kv is True
    traits = select_paged_forward_traits_from_plan(scratch_plan.plan)
    assert traits.cta_tile_kv == 128
    assert traits.num_mma_kv == 2

    (scratch_spec,) = scratch_plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    return scratch_plan, scratch


def _assert_laguna_result_matches_reference(
    *,
    output: torch.Tensor,
    lse_log2: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    relative_attention_bias: torch.Tensor | None = None,
) -> None:
    reference_output, reference_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        k_descale=k_descale,
        v_descale=v_descale,
        relative_attention_bias=relative_attention_bias,
    )
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(lse_log2).all().item()
    assert output.abs().max().item() > 0
    torch.testing.assert_close(
        output.to(torch.float32),
        reference_output.to(torch.float32),
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        lse_log2.to(torch.float32) * math.log(2.0),
        reference_lse.to(torch.float32),
        atol=5e-2,
        rtol=5e-2,
    )
    assert _cosine_similarity(output, reference_output) >= 0.9999


@torch.inference_mode()
def test_laguna_kv128_graph_replay_supports_relative_attention_bias() -> None:
    device = _require_exact_laguna_device()
    torch.manual_seed(20260725)

    page_size = 128
    head_dim = 128
    q_heads = 36
    kv_heads = 4
    page_table_width = 4
    combined_kv_cache = (
        torch.randn(
            (page_table_width, 2, page_size, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        / 4
    ).to(torch.float8_e4m3fn)
    k_cache = combined_kv_cache[:, 0]
    v_cache = combined_kv_cache[:, 1]
    q = torch.randn(
        (1, q_heads, head_dim), dtype=torch.bfloat16, device=device
    ) / 4
    page_table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([257], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    k_descale = torch.ones((1, kv_heads), dtype=torch.float32, device=device)
    v_descale = torch.ones((1, kv_heads), dtype=torch.float32, device=device)
    relative_attention_bias = torch.randn(
        (1, q_heads, 512), dtype=torch.bfloat16, device=device
    )
    output = torch.empty_like(q)

    scratch_plan, scratch = _make_laguna_graph_plan(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table_width=page_table_width,
    )

    def bind():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=1,
            k_descale=k_descale,
            v_descale=v_descale,
            relative_attention_bias=relative_attention_bias,
        )

    bind().run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_binding = bind()
        _, captured_lse = captured_binding.run()

    output.fill_(torch.nan)
    captured_lse.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize(device)
    _assert_laguna_result_matches_reference(
        output=output,
        lse_log2=captured_lse,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        relative_attention_bias=relative_attention_bias,
    )


@torch.inference_mode()
def test_laguna_kv128_b8_regular_graph_replay_handles_heterogeneous_lengths() -> None:
    device = _require_exact_laguna_device()
    torch.manual_seed(20260724)

    batch = 8
    page_size = 128
    page_table_width = 64
    head_dim = 128
    q_heads = 36
    kv_heads = 4
    num_cache_pages = batch * page_table_width
    combined_kv_cache = (
        torch.randn(
            (num_cache_pages, 2, page_size, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        / 4
    ).to(torch.float8_e4m3fn)
    k_cache = combined_kv_cache[:, 0]
    v_cache = combined_kv_cache[:, 1]
    assert not k_cache.is_contiguous()
    assert not v_cache.is_contiguous()
    assert k_cache.stride(0) == 2 * page_size * kv_heads * head_dim

    q = torch.randn(
        (batch, q_heads, head_dim), dtype=torch.bfloat16, device=device
    ) / 4
    page_table = torch.stack(
        tuple(
            torch.roll(
                torch.arange(
                    request_idx * page_table_width,
                    (request_idx + 1) * page_table_width,
                    dtype=torch.int32,
                    device=device,
                ),
                shifts=request_idx + 1,
            )
            for request_idx in range(batch)
        )
    )
    initial_lengths = (1, 129, 1024, 1025, 2113, 4223, 6145, 8129)
    permuted_lengths = (8129, 1, 4223, 129, 6145, 1025, 2113, 1024)
    cache_seqlens = torch.tensor(
        initial_lengths, dtype=torch.int32, device=device
    )
    cu_seqlens_q = torch.arange(
        batch + 1, dtype=torch.int32, device=device
    )
    k_descale = torch.linspace(
        0.55, 1.25, steps=batch * kv_heads, dtype=torch.float32, device=device
    ).reshape(batch, kv_heads)
    v_descale = torch.linspace(
        1.35, 0.65, steps=batch * kv_heads, dtype=torch.float32, device=device
    ).reshape(batch, kv_heads)
    assert torch.unique(k_descale).numel() == batch * kv_heads
    assert torch.unique(v_descale).numel() == batch * kv_heads
    output = torch.empty_like(q)

    scratch_plan, scratch = _make_laguna_graph_plan(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table_width=page_table_width,
    )

    def bind():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=batch,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    warm_binding = bind()
    warm_binding.run()
    torch.cuda.synchronize(device)
    assert warm_binding.scratch.plan.split_kv is True
    assert warm_binding.scratch._use_regular_decode_graph_replay
    assert warm_binding.scratch._uses_plan_owned_decode_graph_metadata
    assert scratch_plan._decode_graph_max_chunks_per_req == 8

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_binding = bind()
        _, captured_lse = captured_binding.run()

    tmp_output = captured_binding.scratch.tmp_output
    tmp_lse = captured_binding.scratch.tmp_lse
    assert tmp_output is not None
    assert tmp_lse is not None
    assert tmp_output.shape[0] == batch * 8
    assert tmp_lse.shape[0] == batch * 8

    stable_ptrs = (
        int(q.data_ptr()),
        int(combined_kv_cache.data_ptr()),
        int(page_table.data_ptr()),
        int(cache_seqlens.data_ptr()),
        int(cu_seqlens_q.data_ptr()),
        int(k_descale.data_ptr()),
        int(v_descale.data_ptr()),
        int(scratch.data_ptr()),
        int(tmp_output.data_ptr()),
        int(tmp_lse.data_ptr()),
        int(output.data_ptr()),
        int(captured_lse.data_ptr()),
    )
    q_before = q.clone()
    cache_before = combined_kv_cache.clone()
    page_table_before = page_table.clone()
    cu_seqlens_q_before = cu_seqlens_q.clone()
    k_descale_before = k_descale.clone()
    v_descale_before = v_descale.clone()

    expected_chunk_counts = (1, 1, 1, 2, 3, 5, 7, 8)
    for replay_idx, lengths in enumerate((initial_lengths, permuted_lengths)):
        cache_seqlens.copy_(
            torch.tensor(lengths, dtype=torch.int32, device=device)
        )
        reference_output, reference_lse = paged_attention_reference(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            causal=True,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        output.fill_(torch.nan)
        captured_lse.fill_(torch.nan)
        tmp_output.fill_(torch.nan)
        tmp_lse.fill_(torch.nan)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        graph.replay()
        torch.cuda.synchronize(device)
        if replay_idx > 0:
            assert torch.cuda.memory_allocated(device) == allocated_before
            assert torch.cuda.memory_reserved(device) == reserved_before

        torch.testing.assert_close(
            output.to(torch.float32),
            reference_output.to(torch.float32),
            atol=5e-2,
            rtol=5e-2,
        )
        torch.testing.assert_close(
            captured_lse.to(torch.float32) * math.log(2.0),
            reference_lse.to(torch.float32),
            atol=5e-2,
            rtol=5e-2,
        )
        assert _cosine_similarity(output, reference_output) >= 0.9999
        assert torch.isfinite(output).all().item()
        assert torch.isfinite(captured_lse).all().item()
        assert output.abs().max().item() > 0

        # The longest request keeps the runtime policy at eight pages per
        # chunk.  Every full chunk therefore performs eight KV128 iterations.
        chunk_tokens = int(captured_binding.scratch.kv_chunk_size_ptr.item())
        assert chunk_tokens == 8 * page_size
        length_to_chunks = dict(zip(initial_lengths, expected_chunk_counts))
        live_chunk_counts = tuple(length_to_chunks[length] for length in lengths)
        compact_prefix = [0]
        active_fixed_rows = []
        for request_idx, chunk_count in enumerate(live_chunk_counts):
            compact_prefix.append(compact_prefix[-1] + chunk_count)
            active_fixed_rows.extend(
                request_idx * 8 + chunk_idx for chunk_idx in range(chunk_count)
            )
        assert captured_binding.scratch.merge_indptr.cpu().tolist() == compact_prefix
        assert captured_binding.scratch.o_indptr.cpu().tolist() == compact_prefix

        active_rows = torch.tensor(
            active_fixed_rows, dtype=torch.int64, device=device
        )
        inactive_mask = torch.ones(batch * 8, dtype=torch.bool, device=device)
        inactive_mask[active_rows] = False
        assert torch.isfinite(tmp_output.index_select(0, active_rows)).all().item()
        assert torch.isfinite(tmp_lse.index_select(0, active_rows)).all().item()
        assert torch.isnan(tmp_output[inactive_mask]).all().item()
        assert torch.isnan(tmp_lse[inactive_mask]).all().item()

        assert stable_ptrs == (
            int(q.data_ptr()),
            int(combined_kv_cache.data_ptr()),
            int(page_table.data_ptr()),
            int(cache_seqlens.data_ptr()),
            int(cu_seqlens_q.data_ptr()),
            int(k_descale.data_ptr()),
            int(v_descale.data_ptr()),
            int(scratch.data_ptr()),
            int(tmp_output.data_ptr()),
            int(tmp_lse.data_ptr()),
            int(output.data_ptr()),
            int(captured_lse.data_ptr()),
        )

    assert torch.equal(q, q_before)
    assert torch.equal(combined_kv_cache, cache_before)
    assert torch.equal(page_table, page_table_before)
    assert torch.equal(cu_seqlens_q, cu_seqlens_q_before)
    assert torch.equal(k_descale, k_descale_before)
    assert torch.equal(v_descale, v_descale_before)


@torch.inference_mode()
def test_page128_graph_replay_handles_page_ids_past_int32_byte_offset() -> None:
    device = require_sparkinfer()
    torch.manual_seed(20260721)

    page_size = 128
    head_dim = 128
    q_heads = 8
    kv_heads = 1
    cache_seqlen = page_size + 65
    live_page_count = math.ceil(cache_seqlen / page_size)

    # Model the strided K/V views used by vLLM's combined
    # [num_pages, 2, page_size, kv_heads, head_dim] allocation.  This keeps the
    # mostly-uninitialized high-pid pool to about 2 GiB while placing every live
    # page strictly beyond the signed-Int32 byte-offset boundary.
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    page_stride_bytes = 2 * page_size * kv_heads * head_dim * element_size
    int32_max = torch.iinfo(torch.int32).max
    high_page_id = int32_max // page_stride_bytes + 2
    num_cache_pages = high_page_id + live_page_count
    combined_kv_cache = torch.empty(
        (num_cache_pages, 2, page_size, kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = combined_kv_cache[:, 0]
    v_cache = combined_kv_cache[:, 1]
    assert k_cache.stride(0) * k_cache.element_size() == page_stride_bytes
    assert high_page_id * page_stride_bytes > int32_max

    k_cache[high_page_id:num_cache_pages].normal_().div_(4)
    v_cache[high_page_id:num_cache_pages].normal_().div_(4)
    page_table = torch.arange(
        high_page_id,
        num_cache_pages,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    live_page_ids = page_table[0, :live_page_count].to(torch.int64)
    assert int(live_page_ids.min().item()) * page_stride_bytes > int32_max

    # Page-128 TMA flattens a combined-cache entry into four physical 64-row
    # tiles.  Verify the exact coordinate/stride product exercised below also
    # crosses 2^31, rather than merely constructing a large unused allocation.
    tile_rows = 64
    tile_stride_bytes = tile_rows * k_cache.stride(1) * k_cache.element_size()
    page_tiles_per_entry = k_cache.stride(0) // (tile_rows * k_cache.stride(1))
    assert page_tiles_per_entry == 4
    assert high_page_id * page_tiles_per_entry * tile_stride_bytes > int32_max

    q = torch.randn(
        (1, q_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    ) / 4
    cache_seqlens = torch.tensor(
        [cache_seqlen], dtype=torch.int32, device=device
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty_like(q)

    scratch_plan = plan_paged_attention_scratch(
        SPARKINFERPagedAttentionScratchCaps(
            device=device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_page_table_width=live_page_count,
            max_work_items=4,
            max_partial_rows=0,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=True,
            copy_runtime_metadata=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=1,
        total_q_capacity=1,
        max_page_table_width=live_page_count,
        max_cache_page_count=live_page_count,
        force_split_kv=False,
    )
    (scratch_spec,) = scratch_plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )

    def bind():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            disable_split_kv=True,
            active_total_q=1,
        )

    warm_binding = bind()
    warm_binding.run()
    torch.cuda.synchronize(device)
    assert warm_binding.scratch.plan.split_kv is False

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_binding = bind()
        _, captured_lse = captured_binding.run()

    output.fill_(torch.nan)
    captured_lse.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize(device)

    reference_output, reference_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(captured_lse).all().item()
    assert output.abs().max().item() > 0
    torch.testing.assert_close(
        output.to(torch.float32),
        reference_output.to(torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        captured_lse.to(torch.float32) * math.log(2.0),
        reference_lse.to(torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )
    assert _cosine_similarity(output, reference_output) >= 0.99999


@torch.inference_mode()
def test_laguna_kv128_graph_replay_handles_live_length_tails() -> None:
    device = _require_exact_laguna_device()
    torch.manual_seed(20260722)

    page_size = 128
    head_dim = 128
    q_heads = 36
    kv_heads = 4
    page_table_width = 4
    combined_kv_cache = (
        torch.randn(
            (page_table_width, 2, page_size, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        / 4
    ).to(torch.float8_e4m3fn)
    k_cache = combined_kv_cache[:, 0]
    v_cache = combined_kv_cache[:, 1]
    assert not k_cache.is_contiguous()
    assert not v_cache.is_contiguous()

    q = torch.randn(
        (1, q_heads, head_dim), dtype=torch.bfloat16, device=device
    ) / 4
    page_table = torch.tensor(
        [[2, 0, 3, 1]], dtype=torch.int32, device=device
    )
    cache_seqlens = torch.tensor([257], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    k_descale = torch.ones((1, kv_heads), dtype=torch.float32, device=device)
    v_descale = torch.ones((1, kv_heads), dtype=torch.float32, device=device)
    output = torch.empty_like(q)

    scratch_plan, scratch = _make_laguna_graph_plan(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table_width=page_table_width,
    )

    def bind():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=1,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    warm_binding = bind()
    warm_binding.run()
    torch.cuda.synchronize(device)
    assert warm_binding.scratch.plan.split_kv is True
    assert warm_binding.scratch._uses_plan_owned_decode_graph_metadata

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_binding = bind()
        _, captured_lse = captured_binding.run()

    stable_ptrs = (
        int(q.data_ptr()),
        int(page_table.data_ptr()),
        int(cache_seqlens.data_ptr()),
        int(cu_seqlens_q.data_ptr()),
        int(output.data_ptr()),
        int(captured_lse.data_ptr()),
    )
    q_before = q.clone()
    cache_before = combined_kv_cache.clone()
    page_table_before = page_table.clone()
    k_descale_before = k_descale.clone()
    v_descale_before = v_descale.clone()

    # One capture must cover tile-internal and physical-page tails.  Only the
    # stable device length scalar changes; split/chunk selection stays inside
    # the captured metadata updater and paged kernels.
    for cache_seqlen in (1, 63, 64, 65, 127, 128, 129, 257):
        cache_seqlens.fill_(cache_seqlen)
        output.fill_(torch.nan)
        captured_lse.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize(device)

        _assert_laguna_result_matches_reference(
            output=output,
            lse_log2=captured_lse,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        assert stable_ptrs == (
            int(q.data_ptr()),
            int(page_table.data_ptr()),
            int(cache_seqlens.data_ptr()),
            int(cu_seqlens_q.data_ptr()),
            int(output.data_ptr()),
            int(captured_lse.data_ptr()),
        )
        assert torch.equal(q, q_before)
        assert torch.equal(combined_kv_cache, cache_before)
        assert torch.equal(page_table, page_table_before)
        assert torch.equal(k_descale, k_descale_before)
        assert torch.equal(v_descale, v_descale_before)


@torch.inference_mode()
def test_laguna_kv128_graph_replay_handles_fp8_page_ids_past_int32_offset() -> None:
    device = _require_exact_laguna_device()
    torch.manual_seed(20260723)

    page_size = 128
    head_dim = 128
    q_heads = 36
    kv_heads = 4
    cache_seqlen = page_size + 65
    live_page_count = math.ceil(cache_seqlen / page_size)

    # This is the production combined FP8 layout.  At four KV heads its page
    # stride is 128 KiB, so page ids just past 16K already require Int64 page
    # scaling even though each individual tensor dimension fits in Int32.
    element_size = torch.empty((), dtype=torch.float8_e4m3fn).element_size()
    page_stride_bytes = 2 * page_size * kv_heads * head_dim * element_size
    int32_max = torch.iinfo(torch.int32).max
    high_page_id = int32_max // page_stride_bytes + 2
    num_cache_pages = high_page_id + live_page_count
    combined_kv_cache = torch.empty(
        (num_cache_pages, 2, page_size, kv_heads, head_dim),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    k_cache = combined_kv_cache[:, 0]
    v_cache = combined_kv_cache[:, 1]
    assert not k_cache.is_contiguous()
    assert not v_cache.is_contiguous()
    assert k_cache.stride(0) * element_size == page_stride_bytes
    assert high_page_id * page_stride_bytes > int32_max

    # Seed the low wrapped-address region with a conspicuously different value
    # so an Int32 wrap cannot accidentally resemble the live high-pid pages.
    low_poison_page_count = live_page_count + 1
    low_poison = torch.empty(
        (low_poison_page_count, 2, page_size, kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    low_poison[:, 0].fill_(4)
    low_poison[:, 1].fill_(-4)
    combined_kv_cache[:low_poison_page_count].copy_(
        low_poison.to(torch.float8_e4m3fn)
    )
    live_pages = (
        torch.randn(
            (live_page_count, 2, page_size, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        / 4
    ).to(torch.float8_e4m3fn)
    combined_kv_cache[high_page_id:num_cache_pages].copy_(live_pages)

    page_table = torch.arange(
        high_page_id,
        num_cache_pages,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    live_page_ids = page_table[0, :live_page_count].to(torch.int64)
    assert int(live_page_ids.min().item()) * page_stride_bytes > int32_max

    # KV128 consumes one physical token page per loop.  In a combined K/V
    # allocation each cache entry is therefore two flattened 128-row tiles;
    # assert the exact tile-coordinate product used by TMA also crosses 2^31.
    stage_tile_rows = 128
    tile_stride_bytes = (
        stage_tile_rows * k_cache.stride(1) * k_cache.element_size()
    )
    page_tiles_per_entry = k_cache.stride(0) // (
        stage_tile_rows * k_cache.stride(1)
    )
    assert page_tiles_per_entry == 2
    assert high_page_id * page_tiles_per_entry * tile_stride_bytes > int32_max

    q = torch.randn(
        (1, q_heads, head_dim), dtype=torch.bfloat16, device=device
    ) / 4
    cache_seqlens = torch.tensor(
        [cache_seqlen], dtype=torch.int32, device=device
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    k_descale = torch.ones((1, kv_heads), dtype=torch.float32, device=device)
    v_descale = torch.ones((1, kv_heads), dtype=torch.float32, device=device)
    output = torch.empty_like(q)

    scratch_plan, scratch = _make_laguna_graph_plan(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table_width=live_page_count,
    )

    def bind():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=1,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    warm_binding = bind()
    warm_binding.run()
    torch.cuda.synchronize(device)
    assert warm_binding.scratch.plan.split_kv is True
    assert warm_binding.scratch._uses_plan_owned_decode_graph_metadata

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_binding = bind()
        _, captured_lse = captured_binding.run()

    low_poison_before = combined_kv_cache[:low_poison_page_count].clone()
    live_pages_before = combined_kv_cache[high_page_id:num_cache_pages].clone()
    q_before = q.clone()
    page_table_before = page_table.clone()
    k_descale_before = k_descale.clone()
    v_descale_before = v_descale.clone()
    output.fill_(torch.nan)
    captured_lse.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize(device)

    _assert_laguna_result_matches_reference(
        output=output,
        lse_log2=captured_lse,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    assert torch.equal(q, q_before)
    assert torch.equal(page_table, page_table_before)
    assert torch.equal(k_descale, k_descale_before)
    assert torch.equal(v_descale, v_descale_before)
    assert torch.equal(
        combined_kv_cache[:low_poison_page_count], low_poison_before
    )
    assert torch.equal(
        combined_kv_cache[high_page_id:num_cache_pages], live_pages_before
    )
