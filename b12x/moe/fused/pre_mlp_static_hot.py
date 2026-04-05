"""Stage-2 hot producer path for the compact/static unified pre-MLP flow.

V1 focuses on the compact static metadata path:
 - consumes validated topk_ids/topk_weights
 - preserves first-appearance active-expert ordering
 - writes the exact compact workspace metadata contract

The FC1 packing path can layer on top of this module without changing the
workspace contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


MAX_STAGE2_TOTAL_PAIRS = 1024


@dataclass(frozen=True, kw_only=True)
class Stage2CompactProducerContract:
    """Stage-2 producer boundary for the compact/static path."""

    weight_E: int
    max_rows: int
    num_topk: int


@triton.jit
def _compact_route_metadata_kernel(
    flat_ids_ptr,
    flat_weights_ptr,
    active_expert_count_ptr,
    weight_expert_ids_ptr,
    global_to_local_expert_ptr,
    row_counts_ptr,
    token_map_ptr,
    token_weights_ptr,
    max_rows,
    num_topk,
    total_pairs,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs
    ids = tl.load(flat_ids_ptr + pair_slots, mask=valid, other=-1).to(tl.int32)
    weights = tl.load(flat_weights_ptr + pair_slots, mask=valid, other=0.0).to(tl.float32)
    token_idx = (pair_slots // num_topk).to(tl.int32)

    row_slots = pair_slots[:, None]
    col_slots = pair_slots[None, :]
    row_valid = valid[:, None]
    col_valid = valid[None, :]

    same_id = ids[:, None] == ids[None, :]
    prior_same = row_valid & col_valid & same_id & (col_slots < row_slots)
    same_id_all = row_valid & col_valid & same_id

    first_flags = valid & (tl.sum(prior_same.to(tl.int32), axis=1) == 0)
    first_prefix = tl.cumsum(first_flags.to(tl.int32), axis=0)

    prior_slots = tl.where(prior_same, col_slots, BLOCK)
    first_match = tl.min(prior_slots, axis=1)
    first_slot = tl.where(first_match < BLOCK, first_match, pair_slots)
    first_slot_mask = col_slots == first_slot[:, None]
    compact_id = tl.sum(tl.where(first_slot_mask, first_prefix[None, :], 0), axis=1) - 1
    row_idx = tl.sum(prior_same.to(tl.int32), axis=1)
    counts = tl.sum(same_id_all.to(tl.int32), axis=1)

    tl.store(weight_expert_ids_ptr + compact_id, ids, mask=valid & first_flags)
    tl.store(global_to_local_expert_ptr + ids, compact_id, mask=valid & first_flags)
    tl.store(row_counts_ptr + compact_id, counts, mask=valid & first_flags)

    map_offsets = compact_id * max_rows + row_idx
    tl.store(token_map_ptr + map_offsets, token_idx, mask=valid)
    tl.store(token_weights_ptr + map_offsets, weights, mask=valid)

    active_expert_count = tl.sum(first_flags.to(tl.int32), axis=0)
    tl.store(active_expert_count_ptr, active_expert_count)


def stage2_compact_route_metadata(
    workspace,
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("topk_ids and topk_weights must both have rank 2")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            "topk_ids/topk_weights shape mismatch: "
            f"{tuple(topk_ids.shape)} vs {tuple(topk_weights.shape)}"
        )
    if topk_ids.device != topk_weights.device:
        raise ValueError("topk_ids and topk_weights must be on the same device")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("stage2_compact_route_metadata expects contiguous routing tensors")

    total_pairs = int(topk_ids.numel())
    if total_pairs > int(workspace.token_map.shape[1]):
        raise ValueError(
            "workspace token_map capacity mismatch: "
            f"expected at least {total_pairs}, got {workspace.token_map.shape[1]}"
        )
    if total_pairs == 0:
        workspace.row_counts.zero_()
        workspace.token_map.zero_()
        workspace.token_weights.zero_()
        workspace.active_expert_count.zero_()
        workspace.weight_expert_ids.zero_()
        workspace.global_to_local_expert.fill_(-1)
        return
    if total_pairs > MAX_STAGE2_TOTAL_PAIRS:
        raise ValueError(
            "stage2 compact metadata kernel currently supports at most "
            f"{MAX_STAGE2_TOTAL_PAIRS} routed pairs, got {total_pairs}"
        )

    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    if flat_ids.dtype != torch.int32:
        flat_ids = flat_ids.to(torch.int32)
    if flat_weights.dtype != torch.float32:
        flat_weights = flat_weights.to(torch.float32)

    workspace.row_counts.zero_()
    workspace.token_map.zero_()
    workspace.token_weights.zero_()
    workspace.active_expert_count.zero_()
    workspace.weight_expert_ids.zero_()
    workspace.global_to_local_expert.fill_(-1)

    block = triton.next_power_of_2(total_pairs)
    num_warps = 1 if block <= 16 else 2 if block <= 64 else 4
    _compact_route_metadata_kernel[(1,)](
        flat_ids,
        flat_weights,
        workspace.active_expert_count,
        workspace.weight_expert_ids,
        workspace.global_to_local_expert,
        workspace.row_counts,
        workspace.token_map,
        workspace.token_weights,
        workspace.token_map.shape[1],
        topk_ids.shape[1],
        total_pairs,
        BLOCK=block,
        num_warps=num_warps,
    )


def stage2_quantize_fc1_inputs(
    workspace,
    *,
    normalized_hidden_states: torch.Tensor,
    expert_input_scale: torch.Tensor,
    expert_alpha: torch.Tensor,
    fc1_tile_amax: bool = False,
) -> None:
    if fc1_tile_amax:
        raise NotImplementedError("stage2_quantize_fc1_inputs currently supports fc1_tile_amax=False only")
    if normalized_hidden_states.ndim != 2:
        raise ValueError("normalized_hidden_states must be [num_tokens, hidden_size]")
    if expert_input_scale.ndim != 1 or expert_alpha.ndim != 1:
        raise ValueError("expert_input_scale and expert_alpha must both be rank-1")
    if expert_input_scale.numel() != int(workspace.weight_E):
        raise ValueError(
            "expert_input_scale expert mismatch: expected "
            f"{workspace.weight_E}, got {expert_input_scale.numel()}"
        )
    if expert_alpha.numel() != int(workspace.weight_E):
        raise ValueError(
            "expert_alpha expert mismatch: expected "
            f"{workspace.weight_E}, got {expert_alpha.numel()}"
        )

    from b12x.cute.fp4 import quantize_grouped_nvfp4_torch
    from b12x.integration.tp_moe import _grouped_scale_view_to_swizzled_u8

    device = normalized_hidden_states.device
    hidden_size = int(normalized_hidden_states.shape[1])
    cols_pad_k = int(workspace.packed_input_scale.shape[-1])

    workspace.packed_input.zero_()
    workspace.packed_input_scale.zero_()
    workspace.fc1_tile_scale.zero_()
    workspace.fc1_tile_alpha.zero_()

    active_expert_count = int(workspace.active_expert_count.item())
    total_tiles = 0
    tile_bases: list[tuple[int, int, int, int, float]] = []
    # (global_tile_idx, local_idx, expert_tile_idx, row_count, expert_alpha_value)
    for local_idx in range(active_expert_count):
        row_count = int(workspace.row_counts[local_idx].item())
        if row_count == 0:
            continue
        expert_idx = int(workspace.weight_expert_ids[local_idx].item())
        num_tiles = (row_count + 128 - 1) // 128
        alpha_value = float(expert_alpha[expert_idx].item())
        for expert_tile_idx in range(num_tiles):
            valid_rows = min(128, row_count - expert_tile_idx * 128)
            tile_bases.append((total_tiles, local_idx, expert_tile_idx, valid_rows, alpha_value))
            total_tiles += 1

    if total_tiles == 0:
        return

    tile_rows = torch.zeros((total_tiles, 128, hidden_size), dtype=torch.float32, device=device)
    tile_row_counts = torch.empty((total_tiles,), dtype=torch.int32, device=device)
    tile_scale = torch.empty((total_tiles,), dtype=torch.float32, device=device)

    tile_cursor = 0
    for local_idx in range(active_expert_count):
        row_count = int(workspace.row_counts[local_idx].item())
        if row_count == 0:
            continue
        expert_idx = int(workspace.weight_expert_ids[local_idx].item())
        token_idx = workspace.token_map[local_idx, :row_count].to(torch.long)
        rows_f32 = normalized_hidden_states.index_select(0, token_idx).float()
        num_tiles = (row_count + 128 - 1) // 128
        rows_pad = num_tiles * 128
        tile_rows[tile_cursor : tile_cursor + num_tiles].view(rows_pad, hidden_size)[:row_count].copy_(rows_f32)
        tile_row_counts[tile_cursor : tile_cursor + num_tiles].fill_(128)
        tile_row_counts[tile_cursor + num_tiles - 1] = row_count - (num_tiles - 1) * 128
        expert_scale_value = float(expert_input_scale[expert_idx].item())
        effective_scale = expert_scale_value if expert_scale_value > 0.0 else 1.0
        tile_scale[tile_cursor : tile_cursor + num_tiles].fill_(effective_scale)
        tile_cursor += num_tiles

    packed_grouped, scale_view = quantize_grouped_nvfp4_torch(tile_rows, tile_row_counts, tile_scale)
    packed_tiles = packed_grouped.permute(2, 0, 1).contiguous().view(total_tiles * 128, hidden_size // 2)
    swizzled_tiles = _grouped_scale_view_to_swizzled_u8(
        scale_view,
        rows=128,
        cols=hidden_size,
    ).view(total_tiles * 128, cols_pad_k)

    for global_tile_idx, local_idx, expert_tile_idx, valid_rows, alpha_value in tile_bases:
        row_base = expert_tile_idx * 128
        tile_row_base = global_tile_idx * 128
        workspace.packed_input[local_idx, row_base : row_base + valid_rows].copy_(
            packed_tiles[tile_row_base : tile_row_base + valid_rows]
        )
        workspace.packed_input_scale[local_idx, row_base : row_base + 128].copy_(
            swizzled_tiles[tile_row_base : tile_row_base + 128]
        )
        workspace.fc1_tile_scale[local_idx, expert_tile_idx] = tile_scale[global_tile_idx]
        workspace.fc1_tile_alpha[local_idx, expert_tile_idx] = alpha_value


__all__ = [
    "MAX_STAGE2_TOTAL_PAIRS",
    "Stage2CompactProducerContract",
    "stage2_compact_route_metadata",
    "stage2_quantize_fc1_inputs",
]
