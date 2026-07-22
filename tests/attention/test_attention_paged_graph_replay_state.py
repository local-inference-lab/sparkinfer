from __future__ import annotations

import gc
import weakref

import pytest
import torch

from sparkinfer.attention.paged._scratch import (
    SPARKINFERPagedAttentionScratchCaps,
    plan_paged_attention_scratch,
)
from sparkinfer.attention.paged.reference import paged_attention_reference

from tests._reference.helpers import require_sparkinfer
from tests._reference.paged_attention_helpers import make_paged_inputs


_SCHEDULE_FIELDS = (
    "request_indices",
    "qo_tile_indices",
    "kv_tile_indices",
    "merge_indptr",
    "o_indptr",
    "block_valid_mask",
    "kv_window_start_tokens",
    "kv_chunk_size_ptr",
    "total_num_rows_ptr",
)
_STATIC_REGULAR_FIELDS = (
    "request_indices",
    "qo_tile_indices",
    "kv_tile_indices",
    "block_valid_mask",
)


def _make_decode_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    batch: int,
    page_table_width: int,
    window_left: int,
    max_work_items: int = 512,
    max_partial_rows: int = 512,
):
    plan = plan_paged_attention_scratch(
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
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
            num_cache_pages=int(k_cache.shape[0]),
            use_cuda_graph=True,
            copy_runtime_metadata=True,
        )
    )
    plan.prepare_decode_graph_replay_state(
        batch=batch,
        total_q_capacity=batch,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
        window_left=window_left,
    )
    assert plan.plan.split_kv
    assert plan._decode_graph_max_chunks_per_req is not None
    query_tiles_per_request = (
        plan.plan.gqa_group_size + plan.plan.cta_tile_q - 1
    ) // plan.plan.cta_tile_q
    assert plan.plan.padded_batch_size == (
        batch * query_tiles_per_request * plan._decode_graph_max_chunks_per_req
    )
    assert plan.plan.padded_batch_size <= plan.caps.max_work_items
    return plan


def _schedule_ptrs(binding) -> dict[str, int]:
    return {
        name: int(getattr(binding.scratch, name).data_ptr())
        for name in _SCHEDULE_FIELDS
    }


def _assert_plan_runtime_schedule(
    plan,
    cache_seqlens: torch.Tensor,
    *,
    window_left: int,
) -> None:
    cache = plan._plan_metadata_cache
    assert cache is not None
    page_size = int(plan.caps.page_size)
    chunk_tokens = int(cache.kv_chunk_size_ptr.item())
    assert chunk_tokens > 0 and chunk_tokens % page_size == 0
    chunk_pages = chunk_tokens // page_size
    lengths = [int(value) for value in cache_seqlens.cpu().tolist()]
    expected_window_starts = []
    expected_prefix = [0]
    for cache_len in lengths:
        window_start_token = (
            max(cache_len - 1 - window_left, 0) if window_left >= 0 else 0
        )
        window_start_page = window_start_token // page_size
        expected_window_starts.append(window_start_page * page_size)
        num_pages = max((cache_len + page_size - 1) // page_size, 1)
        effective_pages = max(num_pages - window_start_page, 1)
        num_chunks = max((effective_pages + chunk_pages - 1) // chunk_pages, 1)
        expected_prefix.append(expected_prefix[-1] + num_chunks)

    assert cache.merge_indptr.cpu().tolist() == expected_prefix
    assert cache.o_indptr.cpu().tolist() == expected_prefix
    assert cache.kv_window_start_tokens.cpu().tolist() == expected_window_starts
    assert int(cache.total_num_rows_ptr.item()) == len(lengths)


def _assert_matches_reference(
    output: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    window_left: int,
) -> None:
    expected, _ = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
    )
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output).item() > 0
    assert torch.allclose(
        output.to(torch.float32),
        expected.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    cosine = torch.nn.functional.cosine_similarity(
        output.float().reshape(-1), expected.float().reshape(-1), dim=0
    ).item()
    assert cosine >= 0.99999


@torch.inference_mode()
def test_compact_decode_graph_metadata_covers_every_query_tile() -> None:
    """GQA 128 / D256 has eight metadata query tiles at CTA tile Q=16."""
    require_sparkinfer()
    batch = 2
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
        make_paged_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=[4096, 3072],
            page_size=64,
            seed=4871,
            q_heads=128,
            kv_heads=1,
            head_dim=256,
            page_table_width=66,
            num_pages=256,
        )
    )
    output = torch.empty_like(q)
    plan = _make_decode_plan(
        q,
        k_cache,
        v_cache,
        batch=batch,
        page_table_width=int(page_table.shape[1]),
        window_left=-1,
        # Two requests x eight query tiles x two chunks is the entire work grid.
        # Partial rows are per (request, chunk), not per query tile.
        max_work_items=32,
        max_partial_rows=4,
    )
    assert plan.plan.gqa_group_size == 128
    assert plan.plan.cta_tile_q == 16
    assert not plan._use_regular_decode_graph_replay
    max_q_tiles_per_req = 8
    assert plan.plan.padded_batch_size == (
        batch * max_q_tiles_per_req * plan._decode_graph_max_chunks_per_req
    )
    assert plan._decode_graph_max_chunks_per_req == 2
    assert plan._decode_graph_chunk_pages_lut is not None
    # Exercise the in-kernel capacity guard.  A one-page LUT entry would ask
    # for 64 chunks at replay time; graph execution must enlarge the shared
    # chunk size without dropping any KV pages or overflowing the fixed grid.
    plan._decode_graph_chunk_pages_lut.fill_(1)

    (scratch_spec,) = plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )

    def bind():
        return plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=batch,
        )

    # Compile the metadata updater before capture.  The forward specialization
    # for this uncommon shape is intentionally outside this scheduler test.
    bind()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bind()
    assert captured.scratch._uses_plan_owned_decode_graph_metadata

    cache_seqlens.fill_(4096)
    graph.replay()
    torch.cuda.synchronize()

    cache = plan._plan_metadata_cache
    assert cache is not None
    chunk_tokens = int(cache.kv_chunk_size_ptr.item())
    chunk_pages = chunk_tokens // int(plan.caps.page_size)
    lengths = [4096, 4096]
    chunks_per_req = [
        max(
            (
                (length + int(plan.caps.page_size) - 1)
                // int(plan.caps.page_size)
                + chunk_pages
                - 1
            )
            // chunk_pages,
            1,
        )
        for length in lengths
    ]
    assert chunks_per_req == [2, 2]
    expected_requests: list[int] = []
    expected_q_tiles: list[int] = []
    expected_kv_tiles: list[int] = []
    for req_idx, num_chunks in enumerate(chunks_per_req):
        for q_tile_idx in range(max_q_tiles_per_req):
            for kv_tile_idx in range(num_chunks):
                expected_requests.append(req_idx)
                expected_q_tiles.append(q_tile_idx)
                expected_kv_tiles.append(kv_tile_idx)
    active_work = len(expected_requests)
    assert cache.request_indices[:active_work].cpu().tolist() == expected_requests
    assert cache.qo_tile_indices[:active_work].cpu().tolist() == expected_q_tiles
    assert cache.kv_tile_indices[:active_work].cpu().tolist() == expected_kv_tiles
    assert cache.block_valid_mask[:active_work].cpu().tolist() == [1] * active_work
    assert torch.count_nonzero(cache.block_valid_mask[active_work:]).item() == 0
    expected_prefix = [0]
    for num_chunks in chunks_per_req:
        expected_prefix.append(expected_prefix[-1] + num_chunks)
    assert cache.merge_indptr.cpu().tolist() == expected_prefix
    assert cache.o_indptr.cpu().tolist() == expected_prefix


@torch.inference_mode()
def test_prebound_decode_graph_run_pins_owning_plan_replay_state() -> None:
    """Capturing only ``binding.run()`` must pin the Plan's addresses."""
    require_sparkinfer()
    batch = 2
    page_table_width = 66
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
        make_paged_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=[4096, 3072],
            page_size=64,
            seed=4903,
            q_heads=8,
            kv_heads=1,
            head_dim=256,
            page_table_width=page_table_width,
            num_pages=256,
        )
    )
    output = torch.empty_like(q)
    plan = _make_decode_plan(
        q,
        k_cache,
        v_cache,
        batch=batch,
        page_table_width=page_table_width,
        window_left=-1,
    )
    (scratch_spec,) = plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )

    def bind():
        return plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=batch,
        )

    # An eager-only binding does not freeze policy: the owning Plan may still
    # replace its replay state before any address has escaped into a graph.
    eager_binding = bind()
    assert eager_binding.scratch._owner_scratch_plan is plan
    eager_binding.run()
    torch.cuda.synchronize()
    assert not plan._decode_graph_replay_state_captured
    old_cache = plan._plan_metadata_cache
    plan.prepare_decode_graph_replay_state(
        batch=batch,
        total_q_capacity=batch,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
    )
    assert plan._plan_metadata_cache is not old_cache

    # Bind and warm outside capture, then capture only run().  The materialized
    # scratch's owner link must close the lifetime gap left by bind-time capture
    # detection and pin the exact metadata cache used by the graph.
    binding = bind()
    binding.run()
    torch.cuda.synchronize()
    assert not plan._decode_graph_replay_state_captured
    captured_cache = plan._plan_metadata_cache
    assert captured_cache is not None
    captured_ptrs = _schedule_ptrs(binding)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.run()

    # Reusing the same prepared state for a second capture must emit a fresh
    # updater node.  The capture-lifetime marker pins addresses, but is not an
    # updater-deduplication flag.
    second_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(second_graph):
        binding.run()

    assert binding.scratch._decode_graph_metadata_captured_in_graph
    assert plan._decode_graph_replay_state_captured
    assert plan._plan_metadata_cache is captured_cache
    assert _schedule_ptrs(binding) == captured_ptrs
    with pytest.raises(RuntimeError, match="cannot replace decode graph replay state"):
        plan.prepare_decode_graph_replay_state(
            batch=batch,
            total_q_capacity=batch,
            max_page_table_width=page_table_width,
            max_cache_page_count=page_table_width,
        )

    # This binding was deliberately created before capture.  With
    # copy_runtime_metadata=True, its captured updater reads the Plan's
    # fixed-address metadata copy, so update that bound input explicitly.
    assert binding.scratch.cache_seqlens is not None
    for replay_graph, replay_lengths in (
        (graph, (512, 2048)),
        (second_graph, (3072, 1024)),
    ):
        replay_cache_seqlens = torch.tensor(
            replay_lengths, dtype=torch.int32, device=q.device
        )
        cache_seqlens.copy_(replay_cache_seqlens)
        binding.scratch.cache_seqlens.copy_(replay_cache_seqlens)
        output.fill_(float("nan"))
        replay_graph.replay()
        torch.cuda.synchronize()
        _assert_matches_reference(
            output,
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            window_left=-1,
        )


@torch.inference_mode()
def test_decode_graph_plan_owned_replay_state_survives_shared_scratch_and_big_pid() -> None:
    """Two Plans may share numerical scratch without sharing replay metadata."""
    require_sparkinfer()
    batch = 2
    page_size = 64
    page_table_width = 66
    q, small_k, small_v, page_table, cache_seqlens, cu_seqlens_q = (
        make_paged_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=[4096, 3072],
            page_size=page_size,
            seed=4817,
            q_heads=8,
            kv_heads=1,
            head_dim=256,
            page_table_width=page_table_width,
            num_pages=256,
        )
    )

    # page_stride = 64 * 1 * 256 * sizeof(bf16) = 32768 bytes.  Starting live
    # pages at 65537 puts every pool-scaled byte offset past signed Int32.
    high_page_base = 65_537
    k_cache = torch.empty(
        (high_page_base + int(small_k.shape[0]), *small_k.shape[1:]),
        dtype=small_k.dtype,
        device=small_k.device,
    )
    v_cache = torch.empty(
        (high_page_base + int(small_v.shape[0]), *small_v.shape[1:]),
        dtype=small_v.dtype,
        device=small_v.device,
    )
    k_cache[high_page_base:].copy_(small_k)
    v_cache[high_page_base:].copy_(small_v)
    page_table.add_(high_page_base)
    del small_k, small_v

    q_full = q
    q_window = q.clone().mul_(0.875)
    page_table_full = page_table
    page_table_window = page_table.clone()
    seqlens_full = cache_seqlens
    seqlens_window = cache_seqlens.clone()
    cu_full = cu_seqlens_q
    cu_window = cu_seqlens_q.clone()
    output_full = torch.empty_like(q_full)
    output_window = torch.empty_like(q_window)
    window_left = 1024

    plan_full = _make_decode_plan(
        q_full,
        k_cache,
        v_cache,
        batch=batch,
        page_table_width=page_table_width,
        window_left=-1,
    )
    plan_window = _make_decode_plan(
        q_window,
        k_cache,
        v_cache,
        batch=batch,
        page_table_width=page_table_width,
        window_left=window_left,
    )
    assert plan_full._use_regular_decode_graph_replay
    assert plan_window._use_regular_decode_graph_replay

    (scratch_spec,) = plan_full.scratch_specs()
    assert scratch_spec == plan_window.scratch_specs()[0]
    shared_scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )

    def bind_full():
        return plan_full.bind(
            scratch=shared_scratch,
            q=q_full,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output_full,
            page_table=page_table_full,
            cache_seqlens=seqlens_full,
            cu_seqlens_q=cu_full,
            window_left=-1,
            active_total_q=batch,
        )

    def bind_window():
        return plan_window.bind(
            scratch=shared_scratch,
            q=q_window,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output_window,
            page_table=page_table_window,
            cache_seqlens=seqlens_window,
            cu_seqlens_q=cu_window,
            window_left=window_left,
            active_total_q=batch,
        )

    # Compile and initialize all paths before capture.
    bind_full().run()
    bind_window().run()
    torch.cuda.synchronize()
    shared_scratch.fill_(0xA5)

    static_full = {
        name: getattr(plan_full._plan_metadata_cache, name).clone()
        for name in _STATIC_REGULAR_FIELDS
    }
    static_window = {
        name: getattr(plan_window._plan_metadata_cache, name).clone()
        for name in _STATIC_REGULAR_FIELDS
    }
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_full = bind_full()
        captured_full.run()
        captured_window = bind_window()
        captured_window.run()

    assert captured_full.scratch._uses_plan_owned_decode_graph_metadata
    assert captured_window.scratch._uses_plan_owned_decode_graph_metadata
    assert captured_full.scratch.block_valid_mask.shape[0] == (
        plan_full.plan.padded_batch_size
    )
    assert captured_full.scratch.tmp_output.shape[0] == (
        plan_full.plan.padded_batch_size
    )
    full_ptrs = _schedule_ptrs(captured_full)
    window_ptrs = _schedule_ptrs(captured_window)
    assert all(full_ptrs[name] != window_ptrs[name] for name in _SCHEDULE_FIELDS)
    scratch_start = int(shared_scratch.data_ptr())
    scratch_end = scratch_start + int(shared_scratch.numel())
    assert all(
        not (scratch_start <= ptr < scratch_end)
        for ptr in (*full_ptrs.values(), *window_ptrs.values())
    )
    for name in _SCHEDULE_FIELDS:
        assert full_ptrs[name] == int(
            getattr(plan_full._plan_metadata_cache, name).data_ptr()
        )
        assert window_ptrs[name] == int(
            getattr(plan_window._plan_metadata_cache, name).data_ptr()
        )

    # The Plan, rather than an ephemeral binding, owns every address captured
    # by the graph.  Re-preparation is fail-closed once those addresses escape.
    live_merge_ref = weakref.ref(plan_full._plan_metadata_cache.merge_indptr)
    del captured_full, captured_window
    gc.collect()
    assert live_merge_ref() is not None
    assert int(live_merge_ref().data_ptr()) == full_ptrs["merge_indptr"]
    with pytest.raises(RuntimeError, match="cannot replace decode graph replay state"):
        plan_full.prepare_decode_graph_replay_state(
            batch=batch,
            total_q_capacity=batch,
            max_page_table_width=page_table_width,
            max_cache_page_count=page_table_width,
        )

    def replay_and_check(
        full_lengths: tuple[int, int],
        window_lengths: tuple[int, int],
        *,
        check_allocator: bool,
    ) -> None:
        seqlens_full.copy_(
            torch.tensor(full_lengths, dtype=torch.int32, device=q.device)
        )
        seqlens_window.copy_(
            torch.tensor(window_lengths, dtype=torch.int32, device=q.device)
        )
        expected_full, _ = paged_attention_reference(
            q_full,
            k_cache,
            v_cache,
            page_table_full,
            seqlens_full,
            cu_full,
            causal=True,
            window_left=-1,
        )
        expected_window, _ = paged_attention_reference(
            q_window,
            k_cache,
            v_cache,
            page_table_window,
            seqlens_window,
            cu_window,
            causal=True,
            window_left=window_left,
        )
        output_full.fill_(float("nan"))
        output_window.fill_(float("nan"))
        # Poison every byte of the shared allocation between replays.  Plan
        # schedule state is outside this allocation; captured runtime copies
        # and attention kernels overwrite the live numerical regions.
        shared_scratch.fill_(0x5A)
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        graph.replay()
        torch.cuda.synchronize()
        if check_allocator:
            assert torch.cuda.memory_allocated() == allocated_before
            assert torch.cuda.memory_reserved() == reserved_before
        assert torch.allclose(
            output_full.float(), expected_full.float(), atol=2e-2, rtol=2e-2
        )
        assert torch.allclose(
            output_window.float(), expected_window.float(), atol=2e-2, rtol=2e-2
        )
        assert torch.isfinite(output_full).all()
        assert torch.isfinite(output_window).all()

        # Inspect graph-produced state before another bind can launch an eager
        # metadata refresh into the same Plan-owned tensors.
        _assert_plan_runtime_schedule(plan_full, seqlens_full, window_left=-1)
        _assert_plan_runtime_schedule(
            plan_window, seqlens_window, window_left=window_left
        )

        # A later rematerialization may refresh metadata, but it must preserve
        # all Plan-owned addresses.
        rebound_full = bind_full()
        rebound_window = bind_window()
        assert _schedule_ptrs(rebound_full) == full_ptrs
        assert _schedule_ptrs(rebound_window) == window_ptrs

    # First replay settles any one-time CUDA graph/runtime bookkeeping; the
    # following replay is the allocation invariant.
    replay_and_check((512, 2048), (1536, 4096), check_allocator=False)
    replay_and_check((4096, 1024), (2048, 3072), check_allocator=True)

    for name, expected in static_full.items():
        assert torch.equal(getattr(plan_full._plan_metadata_cache, name), expected)
    for name, expected in static_window.items():
        assert torch.equal(getattr(plan_window._plan_metadata_cache, name), expected)

    # Explicit oracle helper keeps cosine/nonzero gates visible in this
    # regression in addition to the exact replay-specific checks above.
    _assert_matches_reference(
        output_full,
        q_full,
        k_cache,
        v_cache,
        page_table_full,
        seqlens_full,
        cu_full,
        window_left=-1,
    )
    _assert_matches_reference(
        output_window,
        q_window,
        k_cache,
        v_cache,
        page_table_window,
        seqlens_window,
        cu_window,
        window_left=window_left,
    )
