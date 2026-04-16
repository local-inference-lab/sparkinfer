from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _compact_topk_ids_kernel(
    topk_ids_ptr,
    compact_topk_ids_ptr,
    weight_expert_ids_ptr,
    active_expert_count_ptr,
    total_pairs,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs
    ids = tl.load(topk_ids_ptr + pair_slots, mask=valid, other=-1).to(tl.int32)

    row_slots = pair_slots[:, None]
    col_slots = pair_slots[None, :]
    row_valid = valid[:, None]
    col_valid = valid[None, :]

    same_id = ids[:, None] == ids[None, :]
    prior_same = row_valid & col_valid & same_id & (col_slots < row_slots)

    first_flags = valid & (tl.sum(prior_same.to(tl.int32), axis=1) == 0)
    first_prefix = tl.cumsum(first_flags.to(tl.int32), axis=0)

    prior_slots = tl.where(prior_same, col_slots, BLOCK)
    first_match = tl.min(prior_slots, axis=1)
    first_slot = tl.where(first_match < BLOCK, first_match, pair_slots)
    first_slot_mask = col_slots == first_slot[:, None]
    compact_id = tl.sum(tl.where(first_slot_mask, first_prefix[None, :], 0), axis=1) - 1

    tl.store(compact_topk_ids_ptr + pair_slots, compact_id, mask=valid)
    tl.store(weight_expert_ids_ptr + compact_id, ids, mask=valid & first_flags)

    active_expert_count = tl.sum(first_flags.to(tl.int32), axis=0)
    tl.store(active_expert_count_ptr, active_expert_count)


@triton.jit
def _compact_route_metadata_kernel(
    compact_topk_ids_ptr,
    route_row_indices_ptr,
    row_counts_ptr,
    total_pairs,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs
    ids = tl.load(compact_topk_ids_ptr + pair_slots, mask=valid, other=-1).to(tl.int32)

    row_slots = pair_slots[:, None]
    col_slots = pair_slots[None, :]
    row_valid = valid[:, None]
    col_valid = valid[None, :]

    same_id = row_valid & col_valid & (ids[:, None] == ids[None, :])
    prior_same = same_id & (col_slots < row_slots)
    route_row = tl.sum(prior_same.to(tl.int32), axis=1)
    total_same = tl.sum(same_id.to(tl.int32), axis=1)
    first_flags = valid & (route_row == 0)

    tl.store(route_row_indices_ptr + pair_slots, route_row, mask=valid)
    tl.store(row_counts_ptr + ids, total_same, mask=valid & first_flags)


@triton.jit
def _compact_route_sorted_state_kernel(
    topk_ids_ptr,
    flat_token_indices_ptr,
    compact_topk_ids_ptr,
    route_row_indices_ptr,
    row_counts_ptr,
    weight_expert_ids_ptr,
    active_expert_count_ptr,
    route_order_ptr,
    sorted_flat_ids_ptr,
    sorted_flat_token_indices_ptr,
    total_pairs,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs
    ids = tl.load(topk_ids_ptr + pair_slots, mask=valid, other=-1).to(tl.int32)
    token_indices = tl.load(
        flat_token_indices_ptr + pair_slots, mask=valid, other=0
    ).to(tl.int64)

    row_slots = pair_slots[:, None]
    col_slots = pair_slots[None, :]
    row_valid = valid[:, None]
    col_valid = valid[None, :]

    same_id = row_valid & col_valid & (ids[:, None] == ids[None, :])
    prior_same = same_id & (col_slots < row_slots)
    route_row = tl.sum(prior_same.to(tl.int32), axis=1)
    total_same = tl.sum(same_id.to(tl.int32), axis=1)
    first_flags = valid & (route_row == 0)

    first_prefix = tl.cumsum(first_flags.to(tl.int32), axis=0)
    prior_slots = tl.where(prior_same, col_slots, BLOCK)
    first_match = tl.min(prior_slots, axis=1)
    first_slot = tl.where(first_match < BLOCK, first_match, pair_slots)
    first_slot_mask = col_slots == first_slot[:, None]
    compact_id = tl.sum(
        tl.where(first_slot_mask, first_prefix[None, :], 0), axis=1
    ) - 1

    expert_counts = tl.where(first_flags, total_same, 0)
    expert_offsets = tl.cumsum(expert_counts, axis=0) - expert_counts
    compact_base = tl.sum(
        tl.where(first_slot_mask, expert_offsets[None, :], 0), axis=1
    ).to(tl.int32)
    sorted_pos = compact_base + route_row
    sorted_pos_i64 = sorted_pos.to(tl.int64)

    tl.store(compact_topk_ids_ptr + pair_slots, compact_id, mask=valid)
    tl.store(route_row_indices_ptr + pair_slots, route_row, mask=valid)
    tl.store(weight_expert_ids_ptr + compact_id, ids, mask=valid & first_flags)
    tl.store(row_counts_ptr + compact_id, total_same, mask=valid & first_flags)
    tl.store(route_order_ptr + sorted_pos_i64, pair_slots.to(tl.int64), mask=valid)
    tl.store(sorted_flat_ids_ptr + sorted_pos_i64, ids, mask=valid)
    tl.store(
        sorted_flat_token_indices_ptr + sorted_pos_i64,
        token_indices,
        mask=valid,
    )

    active_expert_count = tl.sum(first_flags.to(tl.int32), axis=0)
    tl.store(active_expert_count_ptr, active_expert_count)


@triton.jit
def _compact_route_sorted_direct_state_kernel(
    topk_ids_ptr,
    flat_token_indices_ptr,
    route_order_ptr,
    sorted_flat_ids_ptr,
    sorted_flat_token_indices_ptr,
    total_pairs,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs
    ids = tl.load(topk_ids_ptr + pair_slots, mask=valid, other=-1).to(tl.int32)
    token_indices = tl.load(
        flat_token_indices_ptr + pair_slots, mask=valid, other=0
    ).to(tl.int64)

    row_slots = pair_slots[:, None]
    col_slots = pair_slots[None, :]
    row_valid = valid[:, None]
    col_valid = valid[None, :]

    same_id = row_valid & col_valid & (ids[:, None] == ids[None, :])
    prior_same = same_id & (col_slots < row_slots)
    route_row = tl.sum(prior_same.to(tl.int32), axis=1)
    total_same = tl.sum(same_id.to(tl.int32), axis=1)
    first_flags = valid & (route_row == 0)

    expert_counts = tl.where(first_flags, total_same, 0)
    expert_offsets = tl.cumsum(expert_counts, axis=0) - expert_counts
    prior_slots = tl.where(prior_same, col_slots, BLOCK)
    first_match = tl.min(prior_slots, axis=1)
    first_slot = tl.where(first_match < BLOCK, first_match, pair_slots)
    first_slot_mask = col_slots == first_slot[:, None]
    compact_base = tl.sum(
        tl.where(first_slot_mask, expert_offsets[None, :], 0), axis=1
    ).to(tl.int32)
    sorted_pos = compact_base + route_row
    sorted_pos_i64 = sorted_pos.to(tl.int64)

    tl.store(route_order_ptr + sorted_pos_i64, pair_slots.to(tl.int64), mask=valid)
    tl.store(sorted_flat_ids_ptr + sorted_pos_i64, ids, mask=valid)
    tl.store(
        sorted_flat_token_indices_ptr + sorted_pos_i64,
        token_indices,
        mask=valid,
    )


@triton.jit
def _compact_route_sorted_singleton_direct_state_kernel(
    topk_ids_ptr,
    flat_token_indices_ptr,
    route_order_ptr,
    sorted_singleton_flat_ids_ptr,
    sorted_flat_token_indices_ptr,
    total_pairs,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs
    ids = tl.load(topk_ids_ptr + pair_slots, mask=valid, other=-1).to(tl.int32)
    token_indices = tl.load(
        flat_token_indices_ptr + pair_slots, mask=valid, other=0
    ).to(tl.int64)

    row_slots = pair_slots[:, None]
    col_slots = pair_slots[None, :]
    row_valid = valid[:, None]
    col_valid = valid[None, :]

    same_id = row_valid & col_valid & (ids[:, None] == ids[None, :])
    prior_same = same_id & (col_slots < row_slots)
    route_row = tl.sum(prior_same.to(tl.int32), axis=1)
    total_same = tl.sum(same_id.to(tl.int32), axis=1)
    first_flags = valid & (route_row == 0)

    expert_counts = tl.where(first_flags, total_same, 0)
    expert_offsets = tl.cumsum(expert_counts, axis=0) - expert_counts
    prior_slots = tl.where(prior_same, col_slots, BLOCK)
    first_match = tl.min(prior_slots, axis=1)
    first_slot = tl.where(first_match < BLOCK, first_match, pair_slots)
    first_slot_mask = col_slots == first_slot[:, None]
    compact_base = tl.sum(
        tl.where(first_slot_mask, expert_offsets[None, :], 0), axis=1
    ).to(tl.int32)
    sorted_pos = compact_base + route_row
    sorted_pos_i64 = sorted_pos.to(tl.int64)
    singleton_ids = tl.where(total_same == 1, ids, -1)

    tl.store(route_order_ptr + sorted_pos_i64, pair_slots.to(tl.int64), mask=valid)
    tl.store(sorted_singleton_flat_ids_ptr + sorted_pos_i64, singleton_ids, mask=valid)
    tl.store(
        sorted_flat_token_indices_ptr + sorted_pos_i64,
        token_indices,
        mask=valid,
    )


def compact_topk_ids(
    topk_ids: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    weight_expert_ids: torch.Tensor,
    active_expert_count: torch.Tensor,
) -> None:
    total_pairs = topk_ids.numel()
    if total_pairs == 0:
        active_expert_count.zero_()
        return
    if compact_topk_ids.numel() < total_pairs:
        raise ValueError("compact_topk_ids must have at least total_pairs elements")
    if weight_expert_ids.numel() <= 0:
        raise ValueError("weight_expert_ids must have positive capacity")
    if active_expert_count.numel() != 1:
        raise ValueError("active_expert_count must have shape [1]")

    block = triton.next_power_of_2(total_pairs)
    num_warps = 1 if block <= 16 else 2
    _compact_topk_ids_kernel[(1,)](
        topk_ids,
        compact_topk_ids,
        weight_expert_ids,
        active_expert_count,
        total_pairs,
        BLOCK=block,
        num_warps=num_warps,
    )


def build_compact_route_metadata(
    compact_topk_ids: torch.Tensor,
    route_row_indices: torch.Tensor,
    row_counts: torch.Tensor,
) -> None:
    total_pairs = compact_topk_ids.numel()
    if total_pairs == 0:
        row_counts.zero_()
        return
    if total_pairs > 256:
        raise ValueError(
            "build_compact_route_metadata currently supports at most 256 routed pairs"
        )
    if route_row_indices.numel() < total_pairs:
        raise ValueError("route_row_indices must have at least total_pairs elements")
    if row_counts.numel() <= 0:
        raise ValueError("row_counts must have positive capacity")

    row_counts[:total_pairs].zero_()
    block = triton.next_power_of_2(total_pairs)
    # Blackwell likes more warp-level parallelism for the compact sorted-route
    # builder once the routed-pair block reaches the Nemotron bs=4/8 regime.
    num_warps = 1 if block <= 16 else 2 if block <= 64 else 4 if block <= 128 else 8
    _compact_route_metadata_kernel[(1,)](
        compact_topk_ids,
        route_row_indices,
        row_counts,
        total_pairs,
        BLOCK=block,
        num_warps=num_warps,
    )


def build_compact_route_sorted_state(
    topk_ids: torch.Tensor,
    flat_token_indices: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    route_row_indices: torch.Tensor,
    row_counts: torch.Tensor,
    weight_expert_ids: torch.Tensor,
    active_expert_count: torch.Tensor,
    route_order: torch.Tensor,
    sorted_flat_ids: torch.Tensor,
    sorted_flat_token_indices: torch.Tensor,
) -> None:
    total_pairs = topk_ids.numel()
    if total_pairs == 0:
        row_counts.zero_()
        active_expert_count.zero_()
        return
    if total_pairs > 256:
        raise ValueError(
            "build_compact_route_sorted_state currently supports at most 256 routed pairs"
        )
    if flat_token_indices.numel() < total_pairs:
        raise ValueError("flat_token_indices must have at least total_pairs elements")
    if compact_topk_ids.numel() < total_pairs:
        raise ValueError("compact_topk_ids must have at least total_pairs elements")
    if route_row_indices.numel() < total_pairs:
        raise ValueError("route_row_indices must have at least total_pairs elements")
    if row_counts.numel() <= 0:
        raise ValueError("row_counts must have positive capacity")
    if weight_expert_ids.numel() <= 0:
        raise ValueError("weight_expert_ids must have positive capacity")
    if active_expert_count.numel() != 1:
        raise ValueError("active_expert_count must have shape [1]")
    if route_order.numel() < total_pairs:
        raise ValueError("route_order must have at least total_pairs elements")
    if sorted_flat_ids.numel() < total_pairs:
        raise ValueError("sorted_flat_ids must have at least total_pairs elements")
    if sorted_flat_token_indices.numel() < total_pairs:
        raise ValueError(
            "sorted_flat_token_indices must have at least total_pairs elements"
        )

    row_counts.zero_()
    block = triton.next_power_of_2(total_pairs)
    num_warps = 1 if block <= 16 else 2 if block <= 64 else 4
    _compact_route_sorted_state_kernel[(1,)](
        topk_ids,
        flat_token_indices,
        compact_topk_ids,
        route_row_indices,
        row_counts,
        weight_expert_ids,
        active_expert_count,
        route_order,
        sorted_flat_ids,
        sorted_flat_token_indices,
        total_pairs,
        BLOCK=block,
        num_warps=num_warps,
    )


def build_compact_route_sorted_direct_state(
    topk_ids: torch.Tensor,
    flat_token_indices: torch.Tensor,
    route_order: torch.Tensor,
    sorted_flat_ids: torch.Tensor,
    sorted_flat_token_indices: torch.Tensor,
) -> None:
    total_pairs = topk_ids.numel()
    if total_pairs == 0:
        return
    if total_pairs > 256:
        raise ValueError(
            "build_compact_route_sorted_direct_state currently supports at most 256 routed pairs"
        )
    if flat_token_indices.numel() < total_pairs:
        raise ValueError("flat_token_indices must have at least total_pairs elements")
    if route_order.numel() < total_pairs:
        raise ValueError("route_order must have at least total_pairs elements")
    if sorted_flat_ids.numel() < total_pairs:
        raise ValueError("sorted_flat_ids must have at least total_pairs elements")
    if sorted_flat_token_indices.numel() < total_pairs:
        raise ValueError(
            "sorted_flat_token_indices must have at least total_pairs elements"
        )

    block = triton.next_power_of_2(total_pairs)
    num_warps = 1 if block <= 16 else 2 if block <= 64 else 4
    _compact_route_sorted_direct_state_kernel[(1,)](
        topk_ids,
        flat_token_indices,
        route_order,
        sorted_flat_ids,
        sorted_flat_token_indices,
        total_pairs,
        BLOCK=block,
        num_warps=num_warps,
    )


def build_compact_route_sorted_singleton_direct_state(
    topk_ids: torch.Tensor,
    flat_token_indices: torch.Tensor,
    route_order: torch.Tensor,
    sorted_singleton_flat_ids: torch.Tensor,
    sorted_flat_token_indices: torch.Tensor,
) -> None:
    total_pairs = topk_ids.numel()
    if total_pairs == 0:
        return
    if total_pairs > 256:
        raise ValueError(
            "build_compact_route_sorted_singleton_direct_state currently supports at most 256 routed pairs"
        )
    if flat_token_indices.numel() < total_pairs:
        raise ValueError("flat_token_indices must have at least total_pairs elements")
    if route_order.numel() < total_pairs:
        raise ValueError("route_order must have at least total_pairs elements")
    if sorted_singleton_flat_ids.numel() < total_pairs:
        raise ValueError(
            "sorted_singleton_flat_ids must have at least total_pairs elements"
        )
    if sorted_flat_token_indices.numel() < total_pairs:
        raise ValueError(
            "sorted_flat_token_indices must have at least total_pairs elements"
        )

    block = triton.next_power_of_2(total_pairs)
    num_warps = 1 if block <= 16 else 2 if block <= 64 else 4
    _compact_route_sorted_singleton_direct_state_kernel[(1,)](
        topk_ids,
        flat_token_indices,
        route_order,
        sorted_singleton_flat_ids,
        sorted_flat_token_indices,
        total_pairs,
        BLOCK=block,
        num_warps=num_warps,
    )


@triton.jit
def _compact_token_map_kernel(
    compact_topk_ids_ptr,
    route_row_indices_ptr,
    flat_token_indices_ptr,
    flat_weights_ptr,
    token_map_ptr,
    token_weights_ptr,
    total_pairs,
    max_rows,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs

    compact_ids = tl.load(compact_topk_ids_ptr + pair_slots, mask=valid, other=0).to(tl.int32)
    route_rows = tl.load(route_row_indices_ptr + pair_slots, mask=valid, other=0).to(tl.int32)
    token_indices = tl.load(flat_token_indices_ptr + pair_slots, mask=valid, other=0).to(tl.int32)
    weights = tl.load(flat_weights_ptr + pair_slots, mask=valid, other=0.0)

    out_offsets = compact_ids.to(tl.int64) * max_rows + route_rows.to(tl.int64)
    tl.store(token_map_ptr + out_offsets, token_indices, mask=valid)
    tl.store(token_weights_ptr + out_offsets, weights, mask=valid)


@triton.jit
def _global_to_local_expert_kernel(
    weight_expert_ids_ptr,
    global_to_local_expert_ptr,
    total_slots,
    BLOCK: tl.constexpr,
):
    local_slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = local_slots < total_slots
    global_ids = tl.load(weight_expert_ids_ptr + local_slots, mask=valid, other=-1).to(tl.int32)
    store_mask = valid & (global_ids >= 0)
    tl.store(global_to_local_expert_ptr + global_ids, local_slots.to(tl.int32), mask=store_mask)


def build_compact_token_map(
    compact_topk_ids: torch.Tensor,
    route_row_indices: torch.Tensor,
    flat_token_indices: torch.Tensor,
    flat_weights: torch.Tensor,
    token_map: torch.Tensor,
    token_weights: torch.Tensor,
) -> None:
    total_pairs = compact_topk_ids.numel()
    if total_pairs == 0:
        token_map.fill_(-1)
        token_weights.zero_()
        return
    if token_map.ndim != 2 or token_weights.ndim != 2:
        raise ValueError("token_map/token_weights must be rank-2")
    if token_map.shape != token_weights.shape:
        raise ValueError("token_map/token_weights shape mismatch")
    if token_map.shape[0] <= 0:
        raise ValueError("token_map must have positive expert capacity")
    if token_map.shape[1] <= 0:
        raise ValueError("token_map must have a positive max_rows dimension")

    token_map.fill_(-1)
    token_weights.zero_()
    block = min(256, triton.next_power_of_2(total_pairs))
    num_warps = 1 if block <= 16 else 2 if block <= 64 else 4
    grid = (triton.cdiv(total_pairs, block),)
    _compact_token_map_kernel[grid](
        compact_topk_ids,
        route_row_indices,
        flat_token_indices,
        flat_weights,
        token_map,
        token_weights,
        total_pairs,
        token_map.shape[1],
        BLOCK=block,
        num_warps=num_warps,
    )


def build_global_to_local_expert(
    weight_expert_ids: torch.Tensor,
    global_to_local_expert: torch.Tensor,
) -> None:
    total_slots = weight_expert_ids.numel()
    global_to_local_expert.fill_(-1)
    if total_slots == 0:
        return
    block = min(256, triton.next_power_of_2(total_slots))
    num_warps = 1 if block <= 16 else 2 if block <= 64 else 4
    grid = (triton.cdiv(total_slots, block),)
    _global_to_local_expert_kernel[grid](
        weight_expert_ids,
        global_to_local_expert,
        total_slots,
        BLOCK=block,
        num_warps=num_warps,
    )


@triton.jit
def _mask_expert_ids_by_row_count_kernel(
    row_counts_ptr,
    expert_ids_ptr,
    masked_expert_ids_ptr,
    bucket_rows,
    total_slots,
    BLOCK: tl.constexpr,
):
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = slots < total_slots
    row_counts = tl.load(row_counts_ptr + slots, mask=valid, other=0).to(tl.int32)
    expert_ids = tl.load(expert_ids_ptr + slots, mask=valid, other=-1).to(tl.int32)
    masked_ids = tl.where(row_counts == bucket_rows, expert_ids, -1)
    tl.store(masked_expert_ids_ptr + slots, masked_ids, mask=valid)


def mask_expert_ids_by_row_count(
    row_counts: torch.Tensor,
    expert_ids: torch.Tensor,
    bucket_rows: int,
    masked_expert_ids: torch.Tensor,
) -> None:
    total_slots = row_counts.numel()
    if expert_ids.numel() < total_slots:
        raise ValueError("expert_ids must have at least total_slots elements")
    if masked_expert_ids.numel() < total_slots:
        raise ValueError("masked_expert_ids must have at least total_slots elements")
    if total_slots == 0:
        return

    block = min(256, triton.next_power_of_2(total_slots))
    num_warps = 1 if block <= 16 else 2 if block <= 64 else 4
    grid = (triton.cdiv(total_slots, block),)
    _mask_expert_ids_by_row_count_kernel[grid](
        row_counts,
        expert_ids,
        masked_expert_ids,
        bucket_rows,
        total_slots,
        BLOCK=block,
        num_warps=num_warps,
    )


@triton.jit
def _build_bucketed_compact_route_kernel(
    row_counts_ptr,
    expert_ids_ptr,
    token_map_ptr,
    token_weights_ptr,
    compact_expert_ids_ptr,
    compact_token_map_ptr,
    compact_token_weights_ptr,
    total_slots,
    token_map_stride0,
    token_map_stride1,
    compact_token_map_stride0,
    compact_token_map_stride1,
    bucket_rows,
    bucket_capacity,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    slots = tl.arange(0, BLOCK_SLOTS)
    valid_slots = slots < total_slots
    row_counts = tl.load(row_counts_ptr + slots, mask=valid_slots, other=0).to(tl.int32)
    expert_ids = tl.load(expert_ids_ptr + slots, mask=valid_slots, other=-1).to(tl.int32)

    row_slots = slots[:, None]
    col_slots = slots[None, :]
    match = valid_slots & (row_counts == bucket_rows)
    prior_match = match[:, None] & match[None, :] & (col_slots < row_slots)
    compact_pos = tl.sum(prior_match.to(tl.int32), axis=1)
    store_slot = match & (compact_pos < bucket_capacity)

    tl.store(compact_expert_ids_ptr + compact_pos, expert_ids, mask=store_slot)

    cols = tl.arange(0, BLOCK_ROWS)
    valid_cols = cols < bucket_rows
    src_offsets = slots[:, None] * token_map_stride0 + cols[None, :] * token_map_stride1
    dst_offsets = (
        compact_pos[:, None] * compact_token_map_stride0
        + cols[None, :] * compact_token_map_stride1
    )
    store_mask = store_slot[:, None] & valid_cols[None, :]
    token_ids = tl.load(token_map_ptr + src_offsets, mask=store_mask, other=-1).to(tl.int32)
    token_weights = tl.load(token_weights_ptr + src_offsets, mask=store_mask, other=0.0)
    tl.store(compact_token_map_ptr + dst_offsets, token_ids, mask=store_mask)
    tl.store(compact_token_weights_ptr + dst_offsets, token_weights, mask=store_mask)


def build_bucketed_compact_route(
    row_counts: torch.Tensor,
    expert_ids: torch.Tensor,
    token_map: torch.Tensor,
    token_weights: torch.Tensor,
    bucket_rows: int,
    compact_expert_ids: torch.Tensor,
    compact_token_map: torch.Tensor,
    compact_token_weights: torch.Tensor,
) -> None:
    total_slots = row_counts.numel()
    if expert_ids.numel() < total_slots:
        raise ValueError("expert_ids must have at least total_slots elements")
    if token_map.ndim != 2 or token_weights.ndim != 2:
        raise ValueError("token_map and token_weights must be rank-2")
    if token_map.shape != token_weights.shape:
        raise ValueError("token_map and token_weights must have the same shape")
    if token_map.shape[0] < total_slots:
        raise ValueError("token_map/token_weights must have at least total_slots rows")
    if bucket_rows <= 0:
        raise ValueError("bucket_rows must be positive")
    if bucket_rows > token_map.shape[1]:
        raise ValueError("bucket_rows exceeds available token-map columns")
    if compact_token_map.shape != compact_token_weights.shape:
        raise ValueError("compact token outputs must have matching shapes")
    if compact_expert_ids.ndim != 1:
        raise ValueError("compact_expert_ids must be rank-1")
    if compact_token_map.ndim != 2:
        raise ValueError("compact_token_map must be rank-2")
    if compact_expert_ids.shape[0] != compact_token_map.shape[0]:
        raise ValueError("compact expert/token-map row counts must match")
    if compact_token_map.shape[1] < bucket_rows:
        raise ValueError("compact token-map output must have at least bucket_rows columns")

    compact_expert_ids.fill_(-1)
    compact_token_map.fill_(-1)
    compact_token_weights.zero_()
    if total_slots == 0:
        return
    if total_slots > 256:
        raise ValueError(
            "build_bucketed_compact_route currently supports at most 256 expert slots"
        )

    block_slots = triton.next_power_of_2(total_slots)
    block_rows = triton.next_power_of_2(bucket_rows)
    num_warps = 1 if block_slots <= 16 else 2 if block_slots <= 64 else 4 if block_slots <= 128 else 8
    _build_bucketed_compact_route_kernel[(1,)](
        row_counts,
        expert_ids,
        token_map,
        token_weights,
        compact_expert_ids,
        compact_token_map,
        compact_token_weights,
        total_slots,
        token_map.stride(0),
        token_map.stride(1),
        compact_token_map.stride(0),
        compact_token_map.stride(1),
        bucket_rows,
        compact_expert_ids.shape[0],
        BLOCK_SLOTS=block_slots,
        BLOCK_ROWS=block_rows,
        num_warps=num_warps,
    )
