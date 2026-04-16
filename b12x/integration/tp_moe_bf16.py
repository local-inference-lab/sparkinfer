"""Tensor-parallel BF16 MoE entrypoints."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from b12x.integration.triton_compact import (
    build_compact_token_map as triton_build_compact_token_map,
)
from b12x.integration.triton_compact import (
    build_compact_route_metadata as triton_build_compact_route_metadata,
)
from b12x.integration.triton_compact import (
    build_compact_route_sorted_state as triton_build_compact_route_sorted_state,
)
from b12x.integration.triton_compact import (
    build_global_to_local_expert as triton_build_global_to_local_expert,
)
from b12x.integration.triton_compact import compact_topk_ids as triton_compact_topk_ids
from b12x.moe.fused.bf16.relu2 import (
    MoEDynamicKernelRelu2,
    MoEMicroKernelRelu2,
    MoEStaticKernelRelu2,
)
from b12x.moe.fused.bf16.silu import (
    MoEDynamicKernelSilu,
    MoEMicroKernelSilu,
    MoEStaticKernelSilu,
)


_BF16_MICRO_CUTOVER_PAIRS = 128
_BF16_STATIC_CUTOVER_PAIRS = 4096
# Nemotron BF16 relu2 now uses one shipped compact-route execution shape.
# On the current tree, forcing the micro backend is slightly worse than the
# static backend across the bs=1/2/4/8 regime, so auto-selection should stay
# on static unless the user explicitly overrides `B12X_BF16_BACKEND=micro`.
_BF16_RELU2_MICRO_CUTOVER_PAIRS = 0
_BF16_RELU2_STATIC_CUTOVER_PAIRS = _BF16_STATIC_CUTOVER_PAIRS
_BF16_SF_VEC_SIZE = 16
_BF16_MMA_TILER_MN = (128, 128)
_BF16_OUTPUT_TILE_COUNT_N = 1
_BF16_RELU2_STATIC_FALLBACK_TAIL_EXPERTS = 4
_BF16_RELU2_COMPACT_MAX_ROUTED_ROWS = 256


@dataclass(frozen=True)
class _ActivationKernelSpec:
    activation: str
    is_gated: bool
    micro_kernel_cls: type
    static_kernel_cls: type
    dynamic_kernel_cls: type

    def w1_rows(self, n: int) -> int:
        return (2 if self.is_gated else 1) * n

    def make_micro_kernel(self):
        return self.micro_kernel_cls(
            _BF16_SF_VEC_SIZE,
            _BF16_MMA_TILER_MN,
            _BF16_OUTPUT_TILE_COUNT_N,
        )

    def make_static_kernel(self, *, num_topk: int):
        return self.static_kernel_cls(
            _BF16_SF_VEC_SIZE,
            _BF16_MMA_TILER_MN,
            _BF16_OUTPUT_TILE_COUNT_N,
            exact_mma_m_tiles=(self.is_gated and num_topk == 1),
        )

    def make_dynamic_kernel(self):
        return self.dynamic_kernel_cls(
            _BF16_SF_VEC_SIZE,
            _BF16_MMA_TILER_MN,
        )


_ACTIVATION_KERNEL_SPECS = {
    "silu": _ActivationKernelSpec(
        activation="silu",
        is_gated=True,
        micro_kernel_cls=MoEMicroKernelSilu,
        static_kernel_cls=MoEStaticKernelSilu,
        dynamic_kernel_cls=MoEDynamicKernelSilu,
    ),
    "relu2": _ActivationKernelSpec(
        activation="relu2",
        is_gated=False,
        micro_kernel_cls=MoEMicroKernelRelu2,
        static_kernel_cls=MoEStaticKernelRelu2,
        dynamic_kernel_cls=MoEDynamicKernelRelu2,
    ),
}

_MICRO_BACKEND_CACHE: Dict[Tuple[str], object] = {}
_STATIC_BACKEND_CACHE: Dict[Tuple[str, int], object] = {}
_DYNAMIC_BACKEND_CACHE: Dict[Tuple[str], object] = {}


@dataclass(frozen=True)
class _CompactRouteState:
    kind: str
    flat_ids_i32: torch.Tensor  # [routed_rows] int32
    flat_weights: torch.Tensor  # [routed_rows] float32
    flat_token_indices: torch.Tensor  # [routed_rows] int32/int64
    compact_topk_ids: torch.Tensor  # [routed_rows] int32
    compact_topk_ids_i64: torch.Tensor | None  # [routed_rows] int64
    route_row_indices: torch.Tensor  # [routed_rows] int32
    route_row_indices_i64: torch.Tensor | None  # [routed_rows] int64
    row_counts: torch.Tensor  # [routed_rows_capacity] int32
    active_expert_count: torch.Tensor  # [1] int32
    weight_expert_ids: torch.Tensor  # [routed_rows_capacity] int32
    weight_expert_ids_i64: torch.Tensor | None  # [routed_rows_capacity] int64
    kernel_weight_expert_ids: torch.Tensor  # [routed_rows_capacity] int32
    kernel_weight_expert_ids_i64: torch.Tensor | None  # [routed_rows_capacity] int64
    routed_rows: int
    global_to_local_expert: torch.Tensor | None = None  # [weight_E] int32
    token_map: torch.Tensor | None = None  # [routed_rows_capacity, num_tokens] int32
    token_weights: torch.Tensor | None = None  # [routed_rows_capacity, num_tokens] float32
    sorted_route_order_i64: torch.Tensor | None = None  # [routed_rows] int32/int64
    sorted_flat_ids_i32: torch.Tensor | None = None  # [routed_rows] int32
    sorted_flat_token_indices: torch.Tensor | None = None  # [routed_rows] int32/int64


@dataclass(kw_only=True)
class TPMoEBF16Workspace:
    implementation: str
    max_tokens: int
    hidden_size: int
    intermediate_size: int
    weight_E: int
    num_topk: int
    activation: str
    expert_chunk_size: int
    device: torch.device
    routed_input: torch.Tensor
    sorted_weights: torch.Tensor
    routed_output_sorted: torch.Tensor
    routed_output_unsorted: torch.Tensor
    routed_input_chunk: torch.Tensor
    fc1_output_chunk: torch.Tensor
    intermediate_chunk: torch.Tensor
    fc2_output_chunk: torch.Tensor
    accum_output: torch.Tensor
    micro_routed_input_flat: torch.Tensor | None = None
    micro_sorted_weights_flat: torch.Tensor | None = None
    micro_fc1_output_flat: torch.Tensor | None = None
    micro_intermediate_flat: torch.Tensor | None = None
    micro_fc2_output_flat: torch.Tensor | None = None
    micro_accum_output_float: torch.Tensor | None = None
    micro_topk_weights_bf16: torch.Tensor | None = None
    micro_row1_routed_input_chunk: torch.Tensor | None = None
    micro_row1_fc1_output_chunk: torch.Tensor | None = None
    micro_row1_fc2_output_chunk: torch.Tensor | None = None
    direct_route_expert_ids_i32: torch.Tensor | None = None
    direct_w1_view: torch.Tensor | None = None
    direct_w2_view: torch.Tensor | None = None
    direct_weight_key: Tuple[int, int] | None = None
    compact_flat_ids_i32: torch.Tensor | None = None
    compact_flat_weights: torch.Tensor | None = None
    compact_flat_token_indices: torch.Tensor | None = None
    compact_topk_ids: torch.Tensor | None = None
    compact_topk_ids_i64: torch.Tensor | None = None
    compact_route_row_indices: torch.Tensor | None = None
    compact_route_row_indices_i64: torch.Tensor | None = None
    compact_row_counts: torch.Tensor | None = None
    compact_active_expert_count: torch.Tensor | None = None
    compact_weight_expert_ids: torch.Tensor | None = None
    compact_weight_expert_ids_i64: torch.Tensor | None = None
    compact_kernel_weight_expert_ids: torch.Tensor | None = None
    compact_kernel_weight_expert_ids_i64: torch.Tensor | None = None
    compact_row_counts_i64: torch.Tensor | None = None
    compact_expert_offsets_i64: torch.Tensor | None = None
    compact_route_positions_i64: torch.Tensor | None = None
    compact_route_order_i64: torch.Tensor | None = None
    compact_route_order_i32: torch.Tensor | None = None
    compact_sorted_flat_ids_i32: torch.Tensor | None = None
    compact_sorted_flat_token_indices: torch.Tensor | None = None
    compact_sorted_flat_token_indices_i32: torch.Tensor | None = None
    compact_pair_arange_i64: torch.Tensor | None = None
    compact_global_to_local_expert: torch.Tensor | None = None
    compact_token_map: torch.Tensor | None = None
    compact_token_weights: torch.Tensor | None = None
    compact_bucket_expert_ids: torch.Tensor | None = None
    compact_bucket_token_map: torch.Tensor | None = None
    compact_bucket_token_weights: torch.Tensor | None = None


@dataclass
class TPMoEBF16WorkspacePool:
    workspaces: Dict[Tuple, TPMoEBF16Workspace] = field(default_factory=dict)

    def clear(self) -> None:
        self.workspaces.clear()


def clear_tp_moe_bf16_caches() -> None:
    _MICRO_BACKEND_CACHE.clear()
    _STATIC_BACKEND_CACHE.clear()
    _DYNAMIC_BACKEND_CACHE.clear()


def allocate_tp_moe_bf16_workspace_pool() -> TPMoEBF16WorkspacePool:
    return TPMoEBF16WorkspacePool()


def _get_activation_kernel_spec(activation: str) -> _ActivationKernelSpec:
    try:
        return _ACTIVATION_KERNEL_SPECS[activation]
    except KeyError as exc:
        raise ValueError(f"unsupported activation {activation!r}") from exc


def _backend_cutovers(*, activation: str) -> tuple[int, int]:
    if activation == "relu2":
        return _BF16_RELU2_MICRO_CUTOVER_PAIRS, _BF16_RELU2_STATIC_CUTOVER_PAIRS
    return _BF16_MICRO_CUTOVER_PAIRS, _BF16_STATIC_CUTOVER_PAIRS


def _select_tp_moe_backend(*, num_tokens: int, num_topk: int, activation: str) -> str:
    override = os.environ.get("B12X_BF16_BACKEND")
    if override is not None:
        if override not in {"micro", "static", "dynamic"}:
            raise ValueError(f"unsupported B12X_BF16_BACKEND={override!r}")
        return override
    routed_rows = num_tokens * num_topk
    micro_cutover, static_cutover = _backend_cutovers(activation=activation)
    if routed_rows <= micro_cutover:
        return "micro"
    if routed_rows <= static_cutover:
        return "static"
    return "dynamic"


def _expert_chunk_size(backend, *, num_tokens: int, num_topk: int) -> int:
    env = os.environ.get("B12X_BF16_EXPERT_CHUNK")
    if env is not None:
        return max(1, int(env))
    resolve_chunk_size = getattr(backend, "resolve_expert_chunk_size", None)
    if resolve_chunk_size is not None:
        return max(1, int(resolve_chunk_size(num_tokens=num_tokens, num_topk=num_topk)))
    return backend.expert_chunk_size


def _compact_route_state_e(*, routed_rows: int, weight_E: int) -> int:
    return min(weight_E, routed_rows)


def _compact_route_alloc_e(*, routed_rows: int, weight_E: int) -> int:
    return max(1, _compact_route_state_e(routed_rows=routed_rows, weight_E=weight_E))


def _flatten_route_ids_i32(
    workspace: TPMoEBF16Workspace,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
        return topk_ids.view(-1)
    flat_ids_i32 = workspace.compact_flat_ids_i32[: topk_ids.numel()]
    flat_ids_i32.copy_(topk_ids.reshape(-1).to(torch.int32))
    return flat_ids_i32


def _flatten_route_weights(
    workspace: TPMoEBF16Workspace,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    if topk_weights.dtype == torch.float32 and topk_weights.is_contiguous():
        return topk_weights.view(-1)
    flat_weights = workspace.compact_flat_weights[: topk_weights.numel()]
    flat_weights.copy_(topk_weights.reshape(-1).to(torch.float32))
    return flat_weights


def _get_micro_kernel(*, activation: str):
    key = (activation,)
    kernel = _MICRO_BACKEND_CACHE.get(key)
    if kernel is None:
        kernel = _get_activation_kernel_spec(activation).make_micro_kernel()
        _MICRO_BACKEND_CACHE[key] = kernel
    return kernel


def _get_static_kernel(*, activation: str, num_topk: int):
    key = (activation, num_topk)
    kernel = _STATIC_BACKEND_CACHE.get(key)
    if kernel is None:
        kernel = _get_activation_kernel_spec(activation).make_static_kernel(
            num_topk=num_topk
        )
        _STATIC_BACKEND_CACHE[key] = kernel
    return kernel


def _get_dynamic_kernel(*, activation: str):
    key = (activation,)
    kernel = _DYNAMIC_BACKEND_CACHE.get(key)
    if kernel is None:
        kernel = _get_activation_kernel_spec(activation).make_dynamic_kernel()
        _DYNAMIC_BACKEND_CACHE[key] = kernel
    return kernel


def _resolve_backend(*, implementation: str, activation: str, num_topk: int):
    if implementation == "micro":
        return _get_micro_kernel(activation=activation)
    if implementation == "static":
        return _get_static_kernel(activation=activation, num_topk=num_topk)
    return _get_dynamic_kernel(activation=activation)


def _alloc_batched_matrix(
    mode0: int,
    mode1: int,
    batch: int,
    *,
    mode0_major: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if mode0_major:
        return torch.empty(
            (batch, mode1, mode0), dtype=dtype, device=device
        ).permute(2, 1, 0)
    return torch.empty((batch, mode0, mode1), dtype=dtype, device=device).permute(
        1, 2, 0
    )


def _round_up_tile_m(rows: int) -> int:
    tile_m = _BF16_MMA_TILER_MN[0]
    return max(tile_m, ((rows + tile_m - 1) // tile_m) * tile_m)


def _workspace_chunk_rows(
    *,
    implementation: str,
    activation: str,
    max_rows_any_chunk: int,
) -> int:
    rows = max(1, max_rows_any_chunk)
    if implementation == "micro" and activation != "relu2":
        return rows
    return _round_up_tile_m(rows)


def _validate_inputs(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
) -> tuple[int, int, int, int]:
    if a.dtype != torch.bfloat16:
        raise TypeError(f"expected a.dtype=torch.bfloat16, got {a.dtype}")
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 expert weights, got w1={w1.dtype}, w2={w2.dtype}")
    if topk_weights.dtype not in {torch.float32, torch.bfloat16}:
        raise TypeError(f"expected topk_weights float32/bfloat16, got {topk_weights.dtype}")
    if topk_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError(f"expected topk_ids int32/int64, got {topk_ids.dtype}")
    if a.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError(f"invalid tensor ranks: a={a.ndim}, w1={w1.ndim}, w2={w2.ndim}")
    if topk_weights.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("expected rank-2 topk tensors")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"topk shape mismatch: {tuple(topk_weights.shape)} vs {tuple(topk_ids.shape)}"
        )
    m, k = a.shape
    experts, w1_rows, w1_k = w1.shape
    w2_experts, hidden_k, n = w2.shape
    if w1_k != k:
        raise ValueError(f"expected w1.shape[2] == {k}, got {w1_k}")
    if hidden_k != k:
        raise ValueError(f"expected w2.shape[1] == {k}, got {hidden_k}")
    if w2_experts != experts:
        raise ValueError(f"expert count mismatch: {experts} vs {w2_experts}")
    if topk_ids.shape[0] != m:
        raise ValueError(f"topk batch mismatch: expected {m}, got {topk_ids.shape[0]}")
    expected_w1_rows = _get_activation_kernel_spec(activation).w1_rows(n)
    if w1_rows != expected_w1_rows:
        raise ValueError(
            f"expected w1.shape[1] == {expected_w1_rows} for activation {activation!r}, got {w1_rows}"
        )
    return m, experts, k, n

@dataclass(frozen=True)
class _WorkspaceSizing:
    max_rows_any_chunk: int
    max_chunk_experts: int


def _alloc_workspace(
    *,
    implementation: str,
    max_tokens: int,
    k: int,
    n: int,
    weight_E: int,
    num_topk: int,
    activation: str,
    expert_chunk_size: int,
    device: torch.device,
    max_rows_any_chunk: int,
    max_chunk_experts: int,
) -> TPMoEBF16Workspace:
    w1_rows = 2 * n if activation == "silu" else n
    routed_rows = max(1, max_tokens * num_topk)
    chunk_rows = _workspace_chunk_rows(
        implementation=implementation,
        activation=activation,
        max_rows_any_chunk=max_rows_any_chunk,
    )
    micro_flat_rows = max(1, chunk_rows * max(1, max_chunk_experts))
    # ReLU2's backend-owned direct route path executes row1 expert-id GEMMs in
    # routed-pair order, so its row1 scratch must be sized by routed rows, not
    # by active-expert chunk capacity.
    row1_routed_rows = routed_rows if activation == "relu2" else max(1, max_chunk_experts)
    workspace = TPMoEBF16Workspace(
        implementation=implementation,
        max_tokens=max_tokens,
        hidden_size=k,
        intermediate_size=n,
        weight_E=weight_E,
        num_topk=num_topk,
        activation=activation,
        expert_chunk_size=expert_chunk_size,
        device=device,
        routed_input=torch.empty(routed_rows, k, dtype=torch.bfloat16, device=device),
        sorted_weights=torch.empty(routed_rows, dtype=torch.float32, device=device),
        routed_output_sorted=torch.empty(
            routed_rows, k, dtype=torch.bfloat16, device=device
        ),
        routed_output_unsorted=torch.empty(
            routed_rows, k, dtype=torch.bfloat16, device=device
        ),
        routed_input_chunk=_alloc_batched_matrix(
            chunk_rows,
            k,
            max(1, max_chunk_experts),
            mode0_major=True,
            dtype=torch.bfloat16,
            device=device,
        ),
        fc1_output_chunk=_alloc_batched_matrix(
            chunk_rows,
            w1_rows,
            max(1, max_chunk_experts),
            mode0_major=True,
            dtype=torch.bfloat16,
            device=device,
        ),
        intermediate_chunk=_alloc_batched_matrix(
            chunk_rows,
            n,
            max(1, max_chunk_experts),
            mode0_major=True,
            dtype=torch.bfloat16,
            device=device,
        ),
        fc2_output_chunk=_alloc_batched_matrix(
            chunk_rows,
            k,
            max(1, max_chunk_experts),
            mode0_major=True,
            dtype=torch.bfloat16,
            device=device,
        ),
        accum_output=torch.empty(max_tokens, k, dtype=torch.float32, device=device),
        micro_routed_input_flat=(
            torch.empty(micro_flat_rows, k, dtype=torch.bfloat16, device=device)
            if implementation == "micro"
            else None
        ),
        micro_sorted_weights_flat=(
            torch.empty(micro_flat_rows, dtype=torch.float32, device=device)
            if implementation == "micro"
            else None
        ),
        micro_fc1_output_flat=(
            torch.empty(micro_flat_rows, w1_rows, dtype=torch.bfloat16, device=device)
            if implementation == "micro"
            else None
        ),
        micro_intermediate_flat=(
            torch.empty(micro_flat_rows, n, dtype=torch.bfloat16, device=device)
            if implementation == "micro" and activation == "silu"
            else None
        ),
        micro_fc2_output_flat=(
            torch.empty(micro_flat_rows, k, dtype=torch.bfloat16, device=device)
            if implementation == "micro"
            else None
        ),
        micro_accum_output_float=(
            torch.empty(max_tokens, k, dtype=torch.float32, device=device)
            if implementation == "micro"
            else None
        ),
        micro_topk_weights_bf16=(
            torch.empty(max_tokens, num_topk, dtype=torch.bfloat16, device=device)
            if activation == "relu2"
            else None
        ),
        micro_row1_routed_input_chunk=(
            _alloc_batched_matrix(
                1,
                k,
                row1_routed_rows,
                mode0_major=True,
                dtype=torch.bfloat16,
                device=device,
            )
            if activation == "relu2"
            else None
        ),
        micro_row1_fc1_output_chunk=(
            _alloc_batched_matrix(
                1,
                w1_rows,
                row1_routed_rows,
                mode0_major=True,
                dtype=torch.bfloat16,
                device=device,
            )
            if activation == "relu2"
            else None
        ),
        micro_row1_fc2_output_chunk=(
            _alloc_batched_matrix(
                1,
                k,
                row1_routed_rows,
                mode0_major=True,
                dtype=torch.bfloat16,
                device=device,
            )
            if activation == "relu2"
            else None
        ),
        direct_route_expert_ids_i32=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
            if activation == "relu2"
            else None
        ),
        compact_flat_ids_i32=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
        ),
        compact_flat_weights=(
            torch.empty(max_tokens * num_topk, dtype=torch.float32, device=device)
        ),
        compact_flat_token_indices=(
            torch.arange(max_tokens, device=device, dtype=torch.int32).repeat_interleave(
                num_topk
            )
        ),
        compact_topk_ids=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
        ),
        compact_topk_ids_i64=(
            torch.empty(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_route_row_indices=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
        ),
        compact_route_row_indices_i64=(
            torch.empty(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_row_counts=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
        ),
        compact_active_expert_count=(
            torch.empty(1, dtype=torch.int32, device=device)
        ),
        compact_weight_expert_ids=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
        ),
        compact_weight_expert_ids_i64=(
            torch.empty(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_kernel_weight_expert_ids=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
        ),
        compact_kernel_weight_expert_ids_i64=(
            torch.empty(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_row_counts_i64=(
            torch.empty(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_expert_offsets_i64=(
            torch.empty(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_route_positions_i64=(
            torch.empty(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_route_order_i64=(
            torch.empty(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_route_order_i32=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
        ),
        compact_sorted_flat_ids_i32=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
        ),
        compact_sorted_flat_token_indices=(
            torch.empty(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_sorted_flat_token_indices_i32=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
        ),
        compact_pair_arange_i64=(
            torch.arange(max_tokens * num_topk, dtype=torch.int64, device=device)
        ),
        compact_bucket_expert_ids=(
            torch.empty(max_tokens * num_topk, dtype=torch.int32, device=device)
            if activation == "relu2"
            else None
        ),
        compact_bucket_token_map=(
            torch.empty(
                max_tokens * num_topk,
                max_tokens,
                dtype=torch.int32,
                device=device,
            )
            if activation == "relu2"
            else None
        ),
        compact_bucket_token_weights=(
            torch.empty(
                max_tokens * num_topk,
                max_tokens,
                dtype=torch.float32,
                device=device,
            )
            if activation == "relu2"
            else None
        ),
    )
    return workspace


def _workspace_key(
    *,
    implementation: str,
    max_tokens: int,
    k: int,
    n: int,
    weight_E: int,
    num_topk: int,
    activation: str,
    expert_chunk_size: int,
    device: torch.device,
) -> tuple:
    return (
        implementation,
        max_tokens,
        k,
        n,
        weight_E,
        num_topk,
        activation,
        expert_chunk_size,
        device.index or 0,
    )


def _resolve_workspace(
    workspace: TPMoEBF16Workspace | TPMoEBF16WorkspacePool | None,
    *,
    implementation: str,
    max_tokens: int,
    k: int,
    n: int,
    weight_E: int,
    num_topk: int,
    activation: str,
    expert_chunk_size: int,
    device: torch.device,
    routing_layout: _WorkspaceSizing,
) -> TPMoEBF16Workspace:
    required_rows = _workspace_chunk_rows(
        implementation=implementation,
        activation=activation,
        max_rows_any_chunk=routing_layout.max_rows_any_chunk,
    )
    required_chunk_experts = max(1, routing_layout.max_chunk_experts)

    if workspace is None:
        return _alloc_workspace(
            implementation=implementation,
            max_tokens=max_tokens,
            k=k,
            n=n,
            weight_E=weight_E,
            num_topk=num_topk,
            activation=activation,
            expert_chunk_size=expert_chunk_size,
            device=device,
            max_rows_any_chunk=required_rows,
            max_chunk_experts=required_chunk_experts,
        )

    if isinstance(workspace, TPMoEBF16Workspace):
        resolved = workspace
    elif isinstance(workspace, TPMoEBF16WorkspacePool):
        key = _workspace_key(
            implementation=implementation,
            max_tokens=max_tokens,
            k=k,
            n=n,
            weight_E=weight_E,
            num_topk=num_topk,
            activation=activation,
            expert_chunk_size=expert_chunk_size,
            device=device,
        )
        resolved = workspace.workspaces.get(key)
        if resolved is None:
            resolved = _alloc_workspace(
                implementation=implementation,
                max_tokens=max_tokens,
                k=k,
                n=n,
                weight_E=weight_E,
                num_topk=num_topk,
                activation=activation,
                expert_chunk_size=expert_chunk_size,
                device=device,
                max_rows_any_chunk=required_rows,
                max_chunk_experts=required_chunk_experts,
            )
            workspace.workspaces[key] = resolved
    else:
        raise TypeError("workspace must be None, TPMoEBF16Workspace, or TPMoEBF16WorkspacePool")

    needs_growth = (
        resolved.routed_input.shape[0] < max_tokens * num_topk
        or resolved.routed_output_sorted.shape[0] < max_tokens * num_topk
        or resolved.routed_input_chunk.shape[0] < required_rows
        or resolved.routed_input_chunk.shape[2] < required_chunk_experts
        or resolved.accum_output.shape[0] < max_tokens
        or (
            implementation == "micro"
            and (
                resolved.micro_routed_input_flat is None
                or resolved.micro_routed_input_flat.shape[0]
                < required_rows * required_chunk_experts
                or resolved.micro_accum_output_float is None
                or resolved.micro_accum_output_float.shape[0] < max_tokens
            )
        )
        or (
            activation == "relu2"
            and (
                resolved.micro_topk_weights_bf16 is None
                or resolved.micro_topk_weights_bf16.shape[0] < max_tokens
                or resolved.micro_row1_routed_input_chunk is None
                or resolved.micro_row1_routed_input_chunk.shape[2]
                < max_tokens * num_topk
                or resolved.micro_row1_fc1_output_chunk is None
                or resolved.micro_row1_fc1_output_chunk.shape[2]
                < max_tokens * num_topk
                or resolved.micro_row1_fc2_output_chunk is None
                or resolved.micro_row1_fc2_output_chunk.shape[2]
                < max_tokens * num_topk
                or resolved.direct_route_expert_ids_i32 is None
                or resolved.direct_route_expert_ids_i32.shape[0] < max_tokens * num_topk
            )
        )
        or (
            (
                resolved.compact_flat_ids_i32 is None
                or resolved.compact_flat_ids_i32.shape[0] < max_tokens * num_topk
                or resolved.compact_flat_weights is None
                or resolved.compact_flat_weights.shape[0] < max_tokens * num_topk
                or resolved.compact_flat_token_indices is None
                or resolved.compact_flat_token_indices.shape[0] < max_tokens * num_topk
                or resolved.compact_topk_ids is None
                or resolved.compact_topk_ids.shape[0] < max_tokens * num_topk
                or resolved.compact_topk_ids_i64 is None
                or resolved.compact_topk_ids_i64.shape[0] < max_tokens * num_topk
                or resolved.compact_route_row_indices is None
                or resolved.compact_route_row_indices.shape[0] < max_tokens * num_topk
                or resolved.compact_route_row_indices_i64 is None
                or resolved.compact_route_row_indices_i64.shape[0] < max_tokens * num_topk
                or resolved.compact_row_counts is None
                or resolved.compact_row_counts.shape[0] < max_tokens * num_topk
                or resolved.compact_active_expert_count is None
                or resolved.compact_weight_expert_ids is None
                or resolved.compact_weight_expert_ids.shape[0] < max_tokens * num_topk
                or resolved.compact_weight_expert_ids_i64 is None
                or resolved.compact_weight_expert_ids_i64.shape[0] < max_tokens * num_topk
                or resolved.compact_kernel_weight_expert_ids is None
                or resolved.compact_kernel_weight_expert_ids.shape[0] < max_tokens * num_topk
                or resolved.compact_kernel_weight_expert_ids_i64 is None
                or resolved.compact_kernel_weight_expert_ids_i64.shape[0] < max_tokens * num_topk
                or resolved.compact_row_counts_i64 is None
                or resolved.compact_row_counts_i64.shape[0] < max_tokens * num_topk
                or resolved.compact_expert_offsets_i64 is None
                or resolved.compact_expert_offsets_i64.shape[0] < max_tokens * num_topk
                or resolved.compact_route_positions_i64 is None
                or resolved.compact_route_positions_i64.shape[0] < max_tokens * num_topk
                or resolved.compact_route_order_i64 is None
                or resolved.compact_route_order_i64.shape[0] < max_tokens * num_topk
                or resolved.compact_route_order_i32 is None
                or resolved.compact_route_order_i32.shape[0] < max_tokens * num_topk
                or resolved.compact_sorted_flat_ids_i32 is None
                or resolved.compact_sorted_flat_ids_i32.shape[0] < max_tokens * num_topk
                or resolved.compact_sorted_flat_token_indices is None
                or resolved.compact_sorted_flat_token_indices.shape[0] < max_tokens * num_topk
                or resolved.compact_sorted_flat_token_indices_i32 is None
                or resolved.compact_sorted_flat_token_indices_i32.shape[0]
                < max_tokens * num_topk
                or resolved.compact_pair_arange_i64 is None
                or resolved.compact_pair_arange_i64.shape[0] < max_tokens * num_topk
            )
        )
        or (
            activation == "relu2"
            and (
                resolved.compact_bucket_expert_ids is None
                or resolved.compact_bucket_expert_ids.shape[0] < max_tokens * num_topk
                or resolved.compact_bucket_token_map is None
                or resolved.compact_bucket_token_map.shape[0] < max_tokens * num_topk
                or resolved.compact_bucket_token_map.shape[1] < max_tokens
                or resolved.compact_bucket_token_weights is None
                or resolved.compact_bucket_token_weights.shape[0] < max_tokens * num_topk
                or resolved.compact_bucket_token_weights.shape[1] < max_tokens
            )
        )
    )
    if needs_growth:
        grown = _alloc_workspace(
            implementation=implementation,
            max_tokens=max(max_tokens, resolved.max_tokens),
            k=k,
            n=n,
            weight_E=weight_E,
            num_topk=num_topk,
            activation=activation,
            expert_chunk_size=expert_chunk_size,
            device=device,
            max_rows_any_chunk=max(required_rows, resolved.routed_input_chunk.shape[0]),
            max_chunk_experts=max(required_chunk_experts, resolved.routed_input_chunk.shape[2]),
        )
        if isinstance(workspace, TPMoEBF16WorkspacePool):
            workspace.workspaces[key] = grown
        return grown
    return resolved


def _prepare_compact_route_state(
    workspace: TPMoEBF16Workspace,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    route_kind: str,
    build_fp4_metadata: bool,
) -> _CompactRouteState:
    if (
        workspace.compact_flat_ids_i32 is None
        or workspace.compact_flat_weights is None
        or workspace.compact_flat_token_indices is None
        or workspace.compact_topk_ids is None
        or workspace.compact_topk_ids_i64 is None
        or workspace.compact_route_row_indices is None
        or workspace.compact_route_row_indices_i64 is None
        or workspace.compact_row_counts is None
        or workspace.compact_active_expert_count is None
        or workspace.compact_weight_expert_ids is None
        or workspace.compact_weight_expert_ids_i64 is None
        or workspace.compact_kernel_weight_expert_ids is None
        or workspace.compact_kernel_weight_expert_ids_i64 is None
    ):
        raise RuntimeError("compact-route workspace scratch is not initialized")

    routed_rows = topk_ids.numel()
    state_e = _compact_route_state_e(
        routed_rows=routed_rows,
        weight_E=workspace.weight_E,
    )
    skip_small_relu2_direct_i64 = (
        workspace.activation == "relu2"
        and routed_rows <= MoEStaticKernelRelu2.compact_direct_routed_rows_limit
    )
    flat_ids_i32 = _flatten_route_ids_i32(workspace, topk_ids)
    flat_weights = _flatten_route_weights(workspace, topk_weights)
    flat_token_indices = workspace.compact_flat_token_indices[:routed_rows]
    compact_topk_ids = workspace.compact_topk_ids[:routed_rows]
    compact_topk_ids_i64 = (
        None
        if skip_small_relu2_direct_i64
        else workspace.compact_topk_ids_i64[:routed_rows]
    )
    route_row_indices = workspace.compact_route_row_indices[:routed_rows]
    route_row_indices_i64 = (
        None
        if skip_small_relu2_direct_i64
        else workspace.compact_route_row_indices_i64[:routed_rows]
    )
    row_counts = workspace.compact_row_counts[:state_e]
    active_expert_count = workspace.compact_active_expert_count
    weight_expert_ids = workspace.compact_weight_expert_ids[:state_e]
    weight_expert_ids_i64 = (
        None
        if skip_small_relu2_direct_i64
        else workspace.compact_weight_expert_ids_i64[:state_e]
    )
    kernel_weight_expert_ids = workspace.compact_kernel_weight_expert_ids[:state_e]
    kernel_weight_expert_ids_i64 = (
        None
        if skip_small_relu2_direct_i64
        else workspace.compact_kernel_weight_expert_ids_i64[:state_e]
    )
    token_map = None
    token_weights = None
    global_to_local_expert = None
    sorted_route_order_i64 = None
    sorted_flat_ids_i32 = None
    sorted_flat_token_indices = None
    if build_fp4_metadata:
        token_map, token_weights, global_to_local_expert = (
            _ensure_fp4_shaped_route_metadata_buffers(
                workspace,
                state_e=state_e,
                num_tokens=topk_ids.shape[0],
            )
        )

    # Keep unused expert slots valid for the indexed-dense kernels. Their
    # routed activations remain zero, so the exact id does not matter.
    weight_expert_ids.fill_(-1)
    if routed_rows <= 256:
        if (
            workspace.compact_route_order_i32 is not None
            and workspace.compact_sorted_flat_ids_i32 is not None
            and workspace.compact_sorted_flat_token_indices_i32 is not None
        ):
            sorted_route_order_i64 = workspace.compact_route_order_i32[:routed_rows]
            sorted_flat_ids_i32 = workspace.compact_sorted_flat_ids_i32[:routed_rows]
            sorted_flat_token_indices = workspace.compact_sorted_flat_token_indices_i32[
                :routed_rows
            ]
            triton_build_compact_route_sorted_state(
                flat_ids_i32,
                flat_token_indices,
                compact_topk_ids,
                route_row_indices,
                row_counts,
                weight_expert_ids,
                active_expert_count,
                sorted_route_order_i64,
                sorted_flat_ids_i32,
                sorted_flat_token_indices,
            )
        else:
            triton_compact_topk_ids(
                flat_ids_i32,
                compact_topk_ids,
                weight_expert_ids,
                active_expert_count,
            )
            triton_build_compact_route_metadata(
                compact_topk_ids,
                route_row_indices,
                row_counts,
            )
    else:
        order = torch.argsort(flat_ids_i32.to(torch.int64), stable=True)
        sorted_ids = flat_ids_i32.index_select(0, order)
        new_segment = torch.ones(routed_rows, device=topk_ids.device, dtype=torch.bool)
        if routed_rows > 1:
            new_segment[1:] = sorted_ids[1:] != sorted_ids[:-1]
        sorted_local_ids = (
            torch.cumsum(new_segment.to(torch.int32), dim=0) - 1
        ).to(compact_topk_ids.dtype)
        compact_topk_ids.scatter_(0, order, sorted_local_ids)
        row_counts.zero_()
        counts = torch.bincount(
            sorted_local_ids.to(torch.int64),
            minlength=state_e,
        ).to(torch.int32)
        row_counts.copy_(counts)
        unique_ids = sorted_ids[new_segment].to(weight_expert_ids.dtype)
        weight_expert_ids[: unique_ids.numel()].copy_(unique_ids)
        active_expert_count.copy_(
            new_segment.to(torch.int32).sum().reshape(1).to(active_expert_count.dtype)
        )
        sorted_positions = torch.arange(
            routed_rows,
            device=topk_ids.device,
            dtype=torch.int32,
        )
        segment_starts = torch.where(
            new_segment,
            sorted_positions,
            torch.zeros_like(sorted_positions),
        )
        segment_starts = torch.cummax(segment_starts, dim=0).values
        route_row_sorted = (sorted_positions - segment_starts).to(route_row_indices.dtype)
        route_row_indices.scatter_(0, order, route_row_sorted)
    if compact_topk_ids_i64 is not None:
        compact_topk_ids_i64.copy_(compact_topk_ids)
    if route_row_indices_i64 is not None:
        route_row_indices_i64.copy_(route_row_indices)
    if weight_expert_ids_i64 is not None:
        weight_expert_ids_i64.copy_(weight_expert_ids)
    torch.clamp_min(weight_expert_ids, 0, out=kernel_weight_expert_ids)
    if kernel_weight_expert_ids_i64 is not None:
        kernel_weight_expert_ids_i64.copy_(kernel_weight_expert_ids)
    if (
        build_fp4_metadata
        and token_map is not None
        and token_weights is not None
        and global_to_local_expert is not None
    ):
        triton_build_global_to_local_expert(
            weight_expert_ids,
            global_to_local_expert,
        )
        triton_build_compact_token_map(
            compact_topk_ids,
            route_row_indices,
            flat_token_indices,
            flat_weights,
            token_map,
            token_weights,
        )
    return _CompactRouteState(
        kind=route_kind,
        flat_ids_i32=flat_ids_i32,
        flat_weights=flat_weights,
        flat_token_indices=flat_token_indices,
        compact_topk_ids=compact_topk_ids,
        compact_topk_ids_i64=compact_topk_ids_i64,
        route_row_indices=route_row_indices,
        route_row_indices_i64=route_row_indices_i64,
        row_counts=row_counts,
        active_expert_count=active_expert_count,
        weight_expert_ids=weight_expert_ids,
        weight_expert_ids_i64=weight_expert_ids_i64,
        kernel_weight_expert_ids=kernel_weight_expert_ids,
        kernel_weight_expert_ids_i64=kernel_weight_expert_ids_i64,
        routed_rows=routed_rows,
        global_to_local_expert=global_to_local_expert,
        token_map=token_map,
        token_weights=token_weights,
        sorted_route_order_i64=sorted_route_order_i64,
        sorted_flat_ids_i32=sorted_flat_ids_i32,
        sorted_flat_token_indices=sorted_flat_token_indices,
    )


def _prepare_compact_route_layout(
    workspace: TPMoEBF16Workspace,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> _CompactRouteState:
    return _prepare_compact_route_state(
        workspace,
        topk_ids,
        topk_weights,
        route_kind="compact",
        # The shipped relu2 compact path now consumes token-map metadata
        # directly inside the static backend's small-row scheduler.
        build_fp4_metadata=True,
    )


def _resolve_output_buffer(a: torch.Tensor, output: torch.Tensor | None) -> torch.Tensor:
    if output is None:
        output = torch.empty_like(a)
    if output.shape != a.shape or output.dtype != torch.bfloat16 or output.device != a.device:
        raise ValueError(
            "output must match the input activation shape/device and use torch.bfloat16"
        )
    return output


def _should_use_compact_relu2(
    *,
    a: torch.Tensor,
    num_topk: int,
    implementation: str,
    activation: str,
) -> bool:
    if activation != "relu2" or implementation != "static":
        return False
    routed_rows = a.shape[0] * num_topk
    # Ship the compact small-row scheduler for the real Nemotron regime
    # directly. The hidden env override still exists for wider bring-up.
    if routed_rows <= MoEStaticKernelRelu2.compact_direct_routed_rows_limit:
        return True
    return (
        os.environ.get("B12X_BF16_ENABLE_COMPACT_RELU2") == "1"
        and routed_rows <= _BF16_RELU2_COMPACT_MAX_ROUTED_ROWS
    )


def _should_use_fp4_shaped_relu2_route(
    *,
    implementation: str,
    activation: str,
) -> bool:
    # The long-term BF16 target should mirror FP4's device-owned routing flow.
    # The missing piece is a route-aware BF16 scatter/gather kernel; the
    # current PyTorch chunk runner is only for bring-up and is not yet correct
    # enough to ship as the default path.
    return (
        os.environ.get("B12X_BF16_ENABLE_FP4_SHAPED_RELU2") == "1"
        and activation == "relu2"
        and implementation in {"static", "dynamic"}
    )


def _compact_relu2_workspace_sizing(
    *,
    num_tokens: int,
    num_topk: int,
    weight_E: int,
) -> _WorkspaceSizing:
    routed_rows = num_tokens * num_topk
    return _WorkspaceSizing(
        # Compact relu2 should follow the same route shape as FP4 static: rows
        # are per-expert token rows, not routed-pair count.
        max_rows_any_chunk=max(1, num_tokens),
        max_chunk_experts=_compact_route_alloc_e(
            routed_rows=routed_rows,
            weight_E=weight_E,
        ),
    )


def _fp4_shaped_static_workspace_sizing(
    *,
    num_tokens: int,
    num_topk: int,
    weight_E: int,
) -> _WorkspaceSizing:
    routed_rows = num_tokens * num_topk
    return _WorkspaceSizing(
        # Keep the FP4-shaped relu2 route workspace fully device-owned and
        # graph-safe: row capacity comes straight from token count instead of
        # a host-side max(row_counts) probe or a synthetic small-row clamp.
        max_rows_any_chunk=max(1, num_tokens),
        max_chunk_experts=_compact_route_alloc_e(
            routed_rows=routed_rows,
            weight_E=weight_E,
        ),
    )


def _route_state_workspace_sizing(
    *,
    num_tokens: int,
    num_topk: int,
    expert_chunk_size: int,
    weight_E: int,
) -> _WorkspaceSizing:
    routed_rows = num_tokens * num_topk
    return _WorkspaceSizing(
        max_rows_any_chunk=max(1, num_tokens),
        max_chunk_experts=max(
            1,
            min(expert_chunk_size, _compact_route_alloc_e(routed_rows=routed_rows, weight_E=weight_E)),
        ),
    )


def _fp4_shaped_relu2_workspace_sizing(
    *,
    num_tokens: int,
    num_topk: int,
    expert_chunk_size: int,
    weight_E: int,
) -> _WorkspaceSizing:
    routed_rows = num_tokens * num_topk
    return _WorkspaceSizing(
        # BF16 should follow FP4's device-owned routing model, but unlike the
        # Nemotron serving path our generic tests can repeat the same expert
        # multiple times within one token row. Keep the workspace sized to the
        # full routed-row budget so route_row_indices stay valid without any
        # Python-side max-row planning.
        max_rows_any_chunk=max(1, routed_rows),
        max_chunk_experts=max(1, min(expert_chunk_size, weight_E)),
    )


def _ensure_fp4_shaped_route_metadata_buffers(
    workspace: TPMoEBF16Workspace,
    *,
    state_e: int,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alloc_rows = max(1, state_e)
    alloc_tokens = max(1, num_tokens)
    token_map = workspace.compact_token_map
    token_weights = workspace.compact_token_weights
    global_to_local_expert = workspace.compact_global_to_local_expert
    if (
        token_map is None
        or token_map.shape[0] < state_e
        or token_map.shape[1] < num_tokens
    ):
        token_map = torch.empty(
            (alloc_rows, alloc_tokens),
            dtype=torch.int32,
            device=workspace.device,
        )
        workspace.compact_token_map = token_map
    if (
        token_weights is None
        or token_weights.shape[0] < state_e
        or token_weights.shape[1] < num_tokens
    ):
        token_weights = torch.empty(
            (alloc_rows, alloc_tokens),
            dtype=torch.float32,
            device=workspace.device,
        )
        workspace.compact_token_weights = token_weights
    if (
        global_to_local_expert is None
        or global_to_local_expert.numel() < workspace.weight_E
    ):
        global_to_local_expert = torch.empty(
            workspace.weight_E,
            dtype=torch.int32,
            device=workspace.device,
        )
        workspace.compact_global_to_local_expert = global_to_local_expert
    return (
        token_map[:state_e, :num_tokens],
        token_weights[:state_e, :num_tokens],
        global_to_local_expert[: workspace.weight_E],
    )


def _prepare_fp4_shaped_relu2_route_layout(
    workspace: TPMoEBF16Workspace,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    build_fp4_metadata: bool = True,
) -> _CompactRouteState:
    return _prepare_compact_route_state(
        workspace,
        topk_ids,
        topk_weights,
        # Match FP4 compact-static behavior: the route state can carry richer
        # FP4-style metadata while still executing as one compact active-expert
        # set instead of an integration-invented chunked pseudo-kind.
        route_kind="compact",
        build_fp4_metadata=build_fp4_metadata,
    )


def b12x_moe_bf16(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    apply_router_weight_on_input: bool = False,
    workspace: TPMoEBF16Workspace | TPMoEBF16WorkspacePool | None = None,
    output: torch.Tensor | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    if apply_router_weight_on_input:
        raise NotImplementedError("apply_router_weight_on_input=True is not supported")

    m, weight_E, k, n = _validate_inputs(
        a,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=activation,
    )
    implementation = _select_tp_moe_backend(
        num_tokens=m,
        num_topk=topk_ids.shape[1],
        activation=activation,
    )
    backend = _resolve_backend(
        implementation=implementation,
        activation=activation,
        num_topk=topk_ids.shape[1],
    )
    expert_chunk_size = _expert_chunk_size(
        backend,
        num_tokens=m,
        num_topk=topk_ids.shape[1],
    )
    output = _resolve_output_buffer(a, output)

    if _should_use_fp4_shaped_relu2_route(
        implementation=implementation,
        activation=activation,
    ):
        route_implementation = implementation
        route_backend = backend
        fp4_shaped_workspace_sizing = (
            _fp4_shaped_static_workspace_sizing(
                num_tokens=m,
                num_topk=topk_ids.shape[1],
                weight_E=weight_E,
            )
            if route_implementation == "static"
            else _fp4_shaped_relu2_workspace_sizing(
                num_tokens=m,
                num_topk=topk_ids.shape[1],
                expert_chunk_size=expert_chunk_size,
                weight_E=weight_E,
            )
        )
        resolved_workspace = _resolve_workspace(
            workspace,
            implementation=route_implementation,
            max_tokens=m,
            k=k,
            n=n,
            weight_E=weight_E,
            num_topk=topk_ids.shape[1],
            activation=activation,
            expert_chunk_size=expert_chunk_size,
            device=a.device,
            routing_layout=fp4_shaped_workspace_sizing,
        )
        flat_route = _prepare_fp4_shaped_relu2_route_layout(
            resolved_workspace,
            topk_ids,
            topk_weights,
            build_fp4_metadata=True,
        )
        return route_backend.run_compact_route(
            a=a,
            w1=w1,
            w2=w2,
            topk_ids=topk_ids,
            route=flat_route,
            workspace=resolved_workspace,
            output=output,
            expert_chunk_size=expert_chunk_size,
        )

    if _should_use_compact_relu2(
        a=a,
        num_topk=topk_ids.shape[1],
        implementation=implementation,
        activation=activation,
    ):
        compact_workspace_sizing = _compact_relu2_workspace_sizing(
            num_tokens=m,
            num_topk=topk_ids.shape[1],
            weight_E=weight_E,
        )
        resolved_workspace = _resolve_workspace(
            workspace,
            implementation=implementation,
            max_tokens=m,
            k=k,
            n=n,
            weight_E=weight_E,
            num_topk=topk_ids.shape[1],
            activation=activation,
            expert_chunk_size=expert_chunk_size,
            device=a.device,
            routing_layout=compact_workspace_sizing,
        )
        compact_route = _prepare_compact_route_layout(
            resolved_workspace,
            topk_ids,
            topk_weights,
        )
        return backend.run_compact_route(
            a=a,
            w1=w1,
            w2=w2,
            topk_ids=topk_ids,
            route=compact_route,
            workspace=resolved_workspace,
            output=output,
            expert_chunk_size=expert_chunk_size,
        )

    # Keep the shared compact-route control plane single-shaped. ReLU2 micro
    # now uses the same route-state/backend contract as static, including the
    # backend-owned direct small-route fast path. Silu micro still reuses the
    # static compact-route executor because its dedicated micro backend has not
    # been taught the route-state path yet. Dynamic also continues to reuse the
    # static executor until it grows a distinct route-aware scheduler.
    use_static_route_backend = implementation == "dynamic" or (
        implementation == "micro" and activation != "relu2"
    )
    route_implementation = "static" if use_static_route_backend else implementation
    route_backend = (
        _get_static_kernel(activation=activation, num_topk=topk_ids.shape[1])
        if use_static_route_backend
        else backend
    )
    route_workspace_sizing = _route_state_workspace_sizing(
        num_tokens=m,
        num_topk=topk_ids.shape[1],
        expert_chunk_size=expert_chunk_size,
        weight_E=weight_E,
    )
    resolved_workspace = _resolve_workspace(
        workspace,
        implementation=route_implementation,
        max_tokens=m,
        k=k,
        n=n,
        weight_E=weight_E,
        num_topk=topk_ids.shape[1],
        activation=activation,
        expert_chunk_size=expert_chunk_size,
        device=a.device,
        routing_layout=route_workspace_sizing,
    )
    route_state = _prepare_compact_route_state(
        resolved_workspace,
        topk_ids,
        topk_weights,
        # Keep the BF16 control plane single-shaped and device-owned, but do
        # not force the shipped path through the heavier token_map consume
        # side yet. The backend still performs best here when it reads compact
        # route-row metadata directly and owns the chunking policy internally.
        route_kind="chunked",
        build_fp4_metadata=False,
    )
    return route_backend.run_compact_route(
        a=a,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        route=route_state,
        workspace=resolved_workspace,
        output=output,
        expert_chunk_size=expert_chunk_size,
    )
