"""Shared compact route/pack metadata helpers for the unified pre-MLP path."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class CompactRouteMetadata:
    weight_expert_ids: torch.Tensor
    local_of_global: torch.Tensor
    counts: torch.Tensor
    sorted_local: torch.Tensor
    sorted_tokens: torch.Tensor
    sorted_weights: torch.Tensor
    row_idx: torch.Tensor

    @property
    def active_expert_count(self) -> int:
        return int(self.weight_expert_ids.numel())


def flatten_routing_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    flat_ids = topk_ids.reshape(-1)
    if flat_ids.dtype not in (torch.int32, torch.int64):
        return flat_ids.to(torch.int32)
    if not flat_ids.is_contiguous():
        return flat_ids.contiguous()
    return flat_ids


def flatten_routing_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    flat_weights = topk_weights.reshape(-1)
    if flat_weights.dtype != torch.float32:
        return flat_weights.to(torch.float32)
    if not flat_weights.is_contiguous():
        return flat_weights.contiguous()
    return flat_weights


def flatten_token_indices(topk_ids: torch.Tensor) -> torch.Tensor:
    num_tokens, top_k = topk_ids.shape
    return torch.arange(num_tokens, device=topk_ids.device, dtype=torch.int32).repeat_interleave(top_k)


def build_compact_route_metadata(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weight_E: int,
) -> CompactRouteMetadata:
    device = topk_ids.device
    flat_ids = flatten_routing_ids(topk_ids)
    flat_weights = flatten_routing_weights(topk_weights)
    flat_tokens = flatten_token_indices(topk_ids)
    total_pairs = flat_ids.numel()
    if total_pairs == 0:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
        empty_f32 = torch.empty((0,), dtype=torch.float32, device=device)
        return CompactRouteMetadata(
            weight_expert_ids=empty_i32,
            local_of_global=torch.full((weight_E,), -1, dtype=torch.int32, device=device),
            counts=empty_i32,
            sorted_local=empty_i64,
            sorted_tokens=empty_i32,
            sorted_weights=empty_f32,
            row_idx=empty_i64,
        )

    pair_pos = torch.arange(total_pairs, device=device, dtype=torch.int64)
    first_pos = torch.full((weight_E,), total_pairs, dtype=torch.int64, device=device)
    first_pos.scatter_reduce_(
        0,
        flat_ids.to(torch.int64),
        pair_pos,
        reduce="amin",
        include_self=True,
    )
    active_mask = first_pos < total_pairs
    active_global = torch.arange(weight_E, device=device, dtype=torch.int64)[active_mask]
    active_order = torch.argsort(first_pos[active_mask], stable=True)
    weight_expert_ids = active_global[active_order].to(torch.int32)
    local_of_global = torch.full((weight_E,), -1, dtype=torch.int32, device=device)
    local_of_global[weight_expert_ids.to(torch.int64)] = torch.arange(
        weight_expert_ids.numel(),
        device=device,
        dtype=torch.int32,
    )
    local_ids = local_of_global[flat_ids.to(torch.int64)]
    sort_idx = torch.argsort(local_ids.to(torch.int64), stable=True)
    sorted_local = local_ids[sort_idx].to(torch.int64)
    sorted_tokens = flat_tokens[sort_idx]
    sorted_weights = flat_weights[sort_idx]
    counts = torch.bincount(sorted_local, minlength=weight_expert_ids.numel()).to(torch.int32)
    offsets = torch.cumsum(counts.to(torch.int64), dim=0) - counts.to(torch.int64)
    row_idx = torch.arange(total_pairs, device=device, dtype=torch.int64) - torch.repeat_interleave(
        offsets,
        counts.to(torch.int64),
    )
    return CompactRouteMetadata(
        weight_expert_ids=weight_expert_ids,
        local_of_global=local_of_global,
        counts=counts,
        sorted_local=sorted_local,
        sorted_tokens=sorted_tokens,
        sorted_weights=sorted_weights,
        row_idx=row_idx,
    )


__all__ = [
    "CompactRouteMetadata",
    "build_compact_route_metadata",
    "flatten_routing_ids",
    "flatten_routing_weights",
    "flatten_token_indices",
]
