"""Tensor-parallel MoE entrypoints backed by fused CuTe DSL kernels."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from torch.profiler import record_function

from b12x.cute.fp4 import align_up, as_grouped_scale_view
from b12x.cute.utils import current_cuda_stream, get_max_active_clusters, get_num_sm, make_ptr
from b12x.integration.triton_compact import compact_topk_ids as triton_compact_topk_ids
from b12x.integration.triton_route import route_topk as triton_route_topk
from b12x.moe.fused import MoEDynamicKernel, MoEMicroKernel, MoEStaticKernel
from b12x.moe.tuning import lookup_max_active_clusters

_NVFP4_BLOCK_SIZE = 16
_RUNTIME_MEMREF_LIMIT = (1 << 31) - 1
_LEVEL_TILE_M = 128
_LEVEL_TILE_N = 128
_DYNAMIC_SLICE_CHUNK = 2


@dataclass(kw_only=True)
class TPMoEWorkspace:
    """Reusable scratch buffers for one `b12x_moe_fp4` shape family."""
    implementation: str
    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    dtype: torch.dtype
    row_counts: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    packed_input: torch.Tensor
    packed_input_scale: torch.Tensor
    barrier_count: torch.Tensor
    barrier_epoch: torch.Tensor
    packed_a_view: object = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None
    packed_a_storage_ptr: object = None
    route_workspace: "_TPRouteWorkspace | None" = None


@dataclass(kw_only=True)
class TPCompactStaticWorkspace(TPMoEWorkspace):
    routed_rows_capacity: int
    active_expert_count: torch.Tensor
    weight_expert_ids: torch.Tensor
    global_to_local_expert: torch.Tensor
    compact_topk_ids: torch.Tensor


@dataclass(kw_only=True)
class TPDynamicWorkspace(TPMoEWorkspace):
    routed_rows_capacity: int
    physical_tiles_capacity: int
    task_capacity: int
    expert_write_rows: torch.Tensor
    expert_tile_base: torch.Tensor
    input_gs: torch.Tensor
    down_input_scale: torch.Tensor
    pair_head: torch.Tensor
    producers_done_count: torch.Tensor
    all_work_published: torch.Tensor
    task_head: torch.Tensor
    task_tail: torch.Tensor
    task_ready: torch.Tensor
    task_expert: torch.Tensor
    task_m_tile: torch.Tensor
    task_slice_begin: torch.Tensor
    task_slice_count: torch.Tensor
    task_valid_rows: torch.Tensor
    tile_write_count: torch.Tensor
    input_gs_src_ptr: int = 0
    down_input_scale_src_ptr: int = 0


@dataclass
class TPMoEWorkspacePool:
    """Caller-owned capacity-based workspace cache partitioned by CUDA stream.

    A single explicit pool may be shared across multiple layers, but overlapping
    launches on different CUDA streams must still use distinct scratch buffers.
    The pool therefore keys allocations by both launch shape and current stream.
    """

    workspaces: Dict[Tuple, TPMoEWorkspace] = field(default_factory=dict)
    route_workspaces: Dict[Tuple, "_TPRouteWorkspace"] = field(default_factory=dict)

    def clear(self) -> None:
        self.workspaces.clear()
        self.route_workspaces.clear()


@dataclass(frozen=True, kw_only=True)
class B12XFP4ExpertWeights:
    """Packaged FP4 expert tensors for routed-expert MoE entrypoints."""

    a1_gscale: torch.Tensor
    w1_fp4: torch.Tensor
    w1_blockscale: torch.Tensor
    w1_alphas: torch.Tensor
    a2_gscale: torch.Tensor
    w2_fp4: torch.Tensor
    w2_blockscale: torch.Tensor
    w2_alphas: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class B12XTopKRouting:
    """Top-k routing selection for sparse-block MoE wrappers."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor | None = None
    flat_ids: torch.Tensor | None = None
    flat_weights: torch.Tensor | None = None


@dataclass(kw_only=True)
class _TPRouteWorkspace:
    router_logits: torch.Tensor
    topk_logits: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class TPMoEPlan:
    """Logical launch plan shared by the static and dynamic backends."""

    implementation: str
    state_E: int
    weight_E: int
    routed_rows: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    dtype: torch.dtype
    max_tokens_per_launch: int
    dynamic_physical_tiles: int | None = None
    dynamic_task_capacity: int | None = None


@dataclass(frozen=True, kw_only=True)
class _TPMoEWorkspacePolicy:
    can_chunk: bool
    eager_exact_dynamic: bool


@dataclass
class _WeightViews:
    """Cached weight views for the concatenated expert-weight layout."""
    w13: torch.Tensor        # [2*n, k//2, E] uint8 (permuted view, no copy)
    down: torch.Tensor       # [k, n//2, E] uint8 (permuted view, no copy)
    w13_sf: torch.Tensor     # 6D MMA view for concatenated w13 scale factors
    down_sf: torch.Tensor    # [E, down_sf_rows, sf_cols] uint8 (view)
    w1_alpha: torch.Tensor   # [E] float32 contiguous tensor in plain CUDA storage
    w2_alpha: torch.Tensor   # [E] float32 contiguous tensor in plain CUDA storage
    # Pre-computed fp4 views and CuTe pointers
    w13_fp4: object = None
    down_fp4: object = None
    sfb_w13_ptr: object = None
    sfb_down_ptr: object = None

_WEIGHT_CACHE: Dict[Tuple[int, int, int], _WeightViews] = {}
_MICRO_KERNEL_CACHE: Dict[Tuple, Tuple] = {}
_STATIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}
_DYNAMIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}
_MAC_CACHE: Dict[Tuple[int, str], int] = {}  # (device_idx, impl) → max_active_clusters
_PLAIN_PARAM_CACHE: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...], torch.dtype, torch.dtype, int], torch.Tensor] = {}
_MICRO_COMPACT_CUTOVER_PAIRS_DEFAULT = 20
_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK_DEFAULT = 40
_STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT = 640
_MICRO_COMPACT_CUTOVER_PAIRS_CACHE: int | None = None
_STATIC_COMPACT_CUTOVER_PAIRS_CACHE: int | None = None
_DYNAMIC_MULTICTA_CACHE: bool | None = None
_DYNAMIC_CHUNK_MULTIPLIER_CACHE: int | None = None
_LAST_WEIGHTS: Tuple = (None, None)  # (cache_key, views)
_LAST_KERNEL: Tuple = (None, None)  # (cache_key, (compiled, mac))


def clear_tp_moe_caches() -> None:
    """Clear runtime caches owned by `tp_moe`.

    Explicit workspaces and workspace pools are caller-owned and intentionally
    unaffected by this helper.
    """
    global _LAST_WEIGHTS
    global _LAST_KERNEL
    global _MICRO_COMPACT_CUTOVER_PAIRS_CACHE
    global _STATIC_COMPACT_CUTOVER_PAIRS_CACHE
    global _DYNAMIC_MULTICTA_CACHE
    global _DYNAMIC_CHUNK_MULTIPLIER_CACHE
    _WEIGHT_CACHE.clear()
    _MICRO_KERNEL_CACHE.clear()
    _STATIC_KERNEL_CACHE.clear()
    _DYNAMIC_KERNEL_CACHE.clear()
    _MAC_CACHE.clear()
    _PLAIN_PARAM_CACHE.clear()
    _MICRO_COMPACT_CUTOVER_PAIRS_CACHE = None
    _STATIC_COMPACT_CUTOVER_PAIRS_CACHE = None
    _DYNAMIC_MULTICTA_CACHE = None
    _DYNAMIC_CHUNK_MULTIPLIER_CACHE = None
    _LAST_WEIGHTS = (None, None)
    _LAST_KERNEL = (None, None)


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("", "0", "false", "False")


_FAST_MATH_DEFAULT = _env_flag("B12X_FAST_MATH", default=True)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def _get_static_compact_cutover_pairs() -> int:
    global _STATIC_COMPACT_CUTOVER_PAIRS_CACHE
    if _STATIC_COMPACT_CUTOVER_PAIRS_CACHE is None:
        cutover = _first_env(
            "B12X_STATIC_COMPACT_CUTOVER_PAIRS",
            "B12X_DYNAMIC_STATIC_CUTOVER_PAIRS",
            "B12X_LEVEL10_STATIC_CUTOVER_PAIRS",
        )
        if cutover is None:
            _STATIC_COMPACT_CUTOVER_PAIRS_CACHE = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
        else:
            _STATIC_COMPACT_CUTOVER_PAIRS_CACHE = max(0, int(cutover))
    return _STATIC_COMPACT_CUTOVER_PAIRS_CACHE


def _get_micro_compact_cutover_pairs() -> int:
    global _MICRO_COMPACT_CUTOVER_PAIRS_CACHE
    if _MICRO_COMPACT_CUTOVER_PAIRS_CACHE is None:
        cutover = _first_env(
            "B12X_MICRO_COMPACT_CUTOVER_PAIRS",
            "B12X_MICRO_CUTOVER_TOKENS",
        )
        if cutover is None:
            _MICRO_COMPACT_CUTOVER_PAIRS_CACHE = _MICRO_COMPACT_CUTOVER_PAIRS_DEFAULT
        else:
            _MICRO_COMPACT_CUTOVER_PAIRS_CACHE = max(0, int(cutover))
    return _MICRO_COMPACT_CUTOVER_PAIRS_CACHE


def _dynamic_multicta_enabled() -> bool:
    global _DYNAMIC_MULTICTA_CACHE
    if _DYNAMIC_MULTICTA_CACHE is None:
        multicta_env = _first_env(
            "B12X_DYNAMIC_ENABLE_MULTICTA",
            "B12X_LEVEL10_ENABLE_MULTICTA",
        )
        if multicta_env is None:
            multicta_env = "1"
        _DYNAMIC_MULTICTA_CACHE = multicta_env == "1"
    return _DYNAMIC_MULTICTA_CACHE


def _get_dynamic_chunk_multiplier() -> int:
    global _DYNAMIC_CHUNK_MULTIPLIER_CACHE
    if _DYNAMIC_CHUNK_MULTIPLIER_CACHE is None:
        mult_env = os.environ.get("B12X_DYNAMIC_CHUNK_MULTIPLIER", "1")
        _DYNAMIC_CHUNK_MULTIPLIER_CACHE = max(1, int(mult_env))
    return _DYNAMIC_CHUNK_MULTIPLIER_CACHE


def _flatten_routing_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    with record_function("tp_moe.flatten_routing_ids"):
        flat_ids = topk_ids.view(-1)
        if flat_ids.dtype not in (torch.int32, torch.int64):
            with record_function("tp_moe.flatten_routing_ids.cast_int32"):
                return flat_ids.to(torch.int32)
        if not flat_ids.is_contiguous():
            with record_function("tp_moe.flatten_routing_ids.contiguous"):
                return flat_ids.contiguous()
        return flat_ids


def _flatten_routing_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    with record_function("tp_moe.flatten_routing_weights"):
        flat_weights = topk_weights.view(-1)
        if flat_weights.dtype != torch.float32:
            with record_function("tp_moe.flatten_routing_weights.cast_fp32"):
                return flat_weights.to(torch.float32)
        if not flat_weights.is_contiguous():
            with record_function("tp_moe.flatten_routing_weights.contiguous"):
                return flat_weights.contiguous()
        return flat_weights


def _prepare_expert_scale(scale: torch.Tensor, weight_E: int) -> torch.Tensor:
    with record_function("tp_moe.prepare_expert_scale"):
        if scale.numel() == 1:
            with record_function("tp_moe.prepare_expert_scale.expand_scalar"):
                return scale.expand(weight_E).to(torch.float32).contiguous()
        if scale.numel() != weight_E:
            raise ValueError(f"expected expert scale with {weight_E} elements, got {scale.numel()}")
        return _get_plain_cuda_tensor(scale, dtype=torch.float32)


def _get_plain_cuda_tensor(t: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    with record_function("tp_moe.get_plain_cuda_tensor"):
        target_dtype = t.dtype if dtype is None else dtype
        key = (
            t.data_ptr(),
            tuple(t.shape),
            tuple(t.stride()),
            t.dtype,
            target_dtype,
            int(t._version),
        )
        cached = _PLAIN_PARAM_CACHE.get(key)
        if cached is not None:
            return cached
        plain = torch.empty(tuple(t.shape), dtype=target_dtype, device=t.device)
        with record_function("tp_moe.get_plain_cuda_tensor.copy"):
            plain.copy_(t.to(target_dtype) if t.dtype != target_dtype else t)
        _PLAIN_PARAM_CACHE[key] = plain
        return plain


def _safe_max_rows_per_launch(E: int, k: int, n: int) -> int:
    """Largest padded row count that fits within CuTe runtime memref limits."""
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)
    limits = [
        _RUNTIME_MEMREF_LIMIT // max(1, E * (k // 2)),
        _RUNTIME_MEMREF_LIMIT // max(1, E * cols_pad_k),
        _RUNTIME_MEMREF_LIMIT // max(1, E * n),
        _RUNTIME_MEMREF_LIMIT // max(1, E),
    ]
    max_rows = min(limits)
    return max_rows - (max_rows % 128)


def _safe_token_chunk(E: int, k: int, n: int, num_topk: int) -> int:
    """Largest token chunk that keeps all per-launch work buffers in range."""
    safe_rows = _safe_max_rows_per_launch(E, k, n)
    if safe_rows <= 0:
        return 1
    max_tokens = max(1, safe_rows // max(1, num_topk))
    while max_tokens > 1 and align_up(max_tokens * num_topk, 128) > safe_rows:
        max_tokens -= 1
    return max_tokens


def _safe_dynamic_max_rows_per_launch(E: int, k: int, _n: int) -> int:
    """Largest graph-safe routed-row budget for the compact dynamic workspace.

    Dynamic now stores routed activations in a compact physical-tile pool, so
    the dominant CuTe memref extents scale with `rows_padded` rather than
    `E * max_rows`. Graph-safe chunking still has to budget for the worst-case
    active-expert envelope, so it reserves `E - 1` extra 128-row tiles in that
    large-row regime.
    """
    rows_padded_limit = _dynamic_rows_padded_limit(k)
    extra_rows = max(0, E - 1) * _LEVEL_TILE_M
    safe_rows = rows_padded_limit - extra_rows
    if safe_rows <= 0:
        return _LEVEL_TILE_M
    return max(_LEVEL_TILE_M, safe_rows - (safe_rows % _LEVEL_TILE_M))


def _dynamic_rows_padded_limit(k: int) -> int:
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)
    rows_padded_limit = min(
        _RUNTIME_MEMREF_LIMIT // max(1, k // 2),
        _RUNTIME_MEMREF_LIMIT // max(1, cols_pad_k),
    )
    return rows_padded_limit - (rows_padded_limit % _LEVEL_TILE_M)


def _safe_dynamic_token_chunk(E: int, k: int, n: int, num_topk: int) -> int:
    """Largest token chunk that fits the compact dynamic launch ABI."""
    safe_rows = _safe_dynamic_max_rows_per_launch(E, k, n)
    max_tokens = max(1, safe_rows // max(1, num_topk))
    while max_tokens > 1 and align_up(max_tokens * num_topk, _LEVEL_TILE_M) > safe_rows:
        max_tokens -= 1
    return max_tokens


def _dynamic_token_chunk_limit(E: int, k: int, n: int, num_topk: int) -> int:
    """Dynamic chunk limit with a compatibility clamp for the old multiplier knob."""
    compact_limit = _safe_dynamic_token_chunk(E, k, n, num_topk)
    legacy_env = os.environ.get("B12X_DYNAMIC_CHUNK_MULTIPLIER")
    if legacy_env is None:
        return compact_limit
    legacy_limit = _safe_token_chunk(E, k, n, num_topk) * _get_dynamic_chunk_multiplier()
    return min(compact_limit, legacy_limit)


def _eager_dynamic_token_chunk_limit(
    topk_ids: torch.Tensor,
    *,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
) -> int:
    """Largest eager token chunk whose exact routed tile pool fits in one launch."""
    rows_padded_limit = _dynamic_rows_padded_limit(k)
    total_tokens = topk_ids.shape[0]
    exact_tiles, _, _ = _dynamic_task_geometry_from_routing(topk_ids, weight_E=weight_E, n=n)
    if exact_tiles * _LEVEL_TILE_M <= rows_padded_limit:
        exact_limit = total_tokens
    else:
        lo = 1
        hi = total_tokens
        exact_limit = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            prefix_tiles, _, _ = _dynamic_task_geometry_from_routing(
                topk_ids[:mid],
                weight_E=weight_E,
                n=n,
            )
            if prefix_tiles * _LEVEL_TILE_M <= rows_padded_limit:
                exact_limit = mid
                lo = mid + 1
            else:
                hi = mid - 1

    legacy_env = os.environ.get("B12X_DYNAMIC_CHUNK_MULTIPLIER")
    if legacy_env is None:
        return exact_limit
    legacy_limit = _safe_token_chunk(weight_E, k, n, num_topk) * _get_dynamic_chunk_multiplier()
    return min(exact_limit, legacy_limit)


def _workspace_policy(
    workspace: TPMoEWorkspace | TPMoEWorkspacePool,
) -> _TPMoEWorkspacePolicy:
    is_pool = isinstance(workspace, TPMoEWorkspacePool)
    return _TPMoEWorkspacePolicy(
        can_chunk=is_pool,
        eager_exact_dynamic=is_pool and not torch.cuda.is_current_stream_capturing(),
    )


def select_tp_moe_backend(
    *,
    num_tokens: int,
    num_topk: int,
) -> str:
    """Pick the fused MoE backend from the intrinsic routed workload shape."""
    routed_rows = num_tokens * num_topk
    if routed_rows <= _get_static_compact_cutover_pairs():
        return "static"
    return "dynamic"


def _dynamic_task_geometry(E: int, n: int, routed_rows: int) -> tuple[int, int, int]:
    routed_rows = max(1, routed_rows)
    base_m_tiles = align_up(routed_rows, _LEVEL_TILE_M) // _LEVEL_TILE_M
    # At most one new physical tile is introduced per active expert beyond the
    # first, and the routed workload cannot touch more experts than routed rows.
    active_expert_upper_bound = min(E, routed_rows)
    max_m_tiles = max(1, base_m_tiles + active_expert_upper_bound - 1)
    gate_tile_cnt = max(1, (n + _LEVEL_TILE_N - 1) // _LEVEL_TILE_N)
    slice_groups = max(1, (gate_tile_cnt + _DYNAMIC_SLICE_CHUNK - 1) // _DYNAMIC_SLICE_CHUNK)
    max_tasks = max_m_tiles * slice_groups
    return max_m_tiles, gate_tile_cnt, max_tasks


def _dynamic_task_geometry_from_routing(
    topk_ids: torch.Tensor,
    *,
    weight_E: int,
    n: int,
) -> tuple[int, int, int]:
    flat_ids = topk_ids.reshape(-1)
    if flat_ids.dtype != torch.int64:
        flat_ids = flat_ids.to(torch.int64)
    counts = torch.bincount(flat_ids, minlength=weight_E)
    tiles_per_expert = (counts + (_LEVEL_TILE_M - 1)) // _LEVEL_TILE_M
    exact_tiles = max(1, int(tiles_per_expert.sum().item()))
    gate_tile_cnt = max(1, (n + _LEVEL_TILE_N - 1) // _LEVEL_TILE_N)
    slice_groups = max(1, (gate_tile_cnt + _DYNAMIC_SLICE_CHUNK - 1) // _DYNAMIC_SLICE_CHUNK)
    max_tasks = exact_tiles * slice_groups
    return exact_tiles, gate_tile_cnt, max_tasks


def _refresh_dynamic_workspace_scales(
    workspace: TPDynamicWorkspace,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    *,
    input_scales_static: bool,
) -> None:
    a1_src_ptr = a1_gscale.data_ptr()
    a2_src_ptr = a2_gscale.data_ptr()
    if (
        not input_scales_static
        or workspace.input_gs_src_ptr != a1_src_ptr
        or workspace.down_input_scale_src_ptr != a2_src_ptr
    ):
        workspace.input_gs.copy_(a1_gscale.expand(workspace.weight_E))
        workspace.down_input_scale.copy_(a2_gscale.expand(workspace.weight_E))
        workspace.input_gs_src_ptr = a1_src_ptr if input_scales_static else 0
        workspace.down_input_scale_src_ptr = a2_src_ptr if input_scales_static else 0


def _finalize_workspace_views(workspace: TPMoEWorkspace) -> None:
    sf_dtype = cutlass.Float8E4M3FN
    # Keep as uint8 — the float4 element type is conveyed to CUTLASS via
    # _gptr / compile-time dtype, and dlpack does not support float4.
    workspace.packed_a_view = workspace.packed_input.permute(1, 2, 0)
    workspace.packed_a_flat = workspace.packed_input.view(-1)
    workspace.scale_flat = workspace.packed_input_scale.view(-1)
    workspace.sfa_ptr = make_ptr(
        sf_dtype,
        workspace.packed_input_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    workspace.packed_a_storage_ptr = make_ptr(
        cutlass.Uint8,
        workspace.packed_input.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )


def _alloc_workspace(
    implementation: str,
    state_E: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    *,
    routed_rows: int,
    max_rows: int,
    input_scales_static: bool,
    dynamic_physical_tiles: int | None = None,
    dynamic_task_capacity: int | None = None,
) -> TPMoEWorkspace:
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)
    common_kwargs = dict(
        implementation=implementation,
        state_E=state_E,
        weight_E=weight_E,
        max_rows=max_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        dtype=dtype,
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
    )

    if implementation == "static":
        static_rows_pad_k = align_up(max_rows, 128)
        workspace = TPCompactStaticWorkspace(
            **common_kwargs,
            routed_rows_capacity=routed_rows,
            token_map=torch.zeros(state_E, max_rows, dtype=torch.int32, device=device),
            token_weights=torch.zeros(state_E, max_rows, dtype=torch.float32, device=device),
            packed_input=torch.empty(state_E, max_rows, k // 2, dtype=torch.uint8, device=device),
            packed_input_scale=torch.empty(state_E, static_rows_pad_k, cols_pad_k, dtype=torch.uint8, device=device),
            active_expert_count=torch.zeros(1, dtype=torch.int32, device=device),
            weight_expert_ids=torch.arange(state_E, dtype=torch.int32, device=device),
            global_to_local_expert=torch.empty(weight_E, dtype=torch.int32, device=device),
            compact_topk_ids=torch.empty(state_E, dtype=torch.int32, device=device),
        )
        _finalize_workspace_views(workspace)
        return workspace

    if dynamic_physical_tiles is None or dynamic_task_capacity is None:
        dynamic_tiles, _, dynamic_max_tasks = _dynamic_task_geometry(state_E, n, routed_rows)
    else:
        dynamic_tiles = dynamic_physical_tiles
        dynamic_max_tasks = dynamic_task_capacity
    dynamic_rows_padded = dynamic_tiles * _LEVEL_TILE_M
    workspace = TPDynamicWorkspace(
        **common_kwargs,
        routed_rows_capacity=routed_rows,
        physical_tiles_capacity=dynamic_tiles,
        task_capacity=dynamic_max_tasks,
        token_map=torch.zeros(dynamic_rows_padded, dtype=torch.int32, device=device),
        token_weights=torch.zeros(dynamic_rows_padded, dtype=torch.float32, device=device),
        packed_input=torch.empty(1, dynamic_rows_padded, k // 2, dtype=torch.uint8, device=device),
        packed_input_scale=torch.empty(dynamic_rows_padded, cols_pad_k, dtype=torch.uint8, device=device),
        expert_write_rows=torch.zeros(state_E, dtype=torch.int32, device=device),
        expert_tile_base=torch.zeros(state_E + 1, dtype=torch.int32, device=device),
        input_gs=torch.empty(weight_E, dtype=torch.float32, device=device),
        down_input_scale=torch.empty(weight_E, dtype=torch.float32, device=device),
        pair_head=torch.zeros(1, dtype=torch.int32, device=device),
        producers_done_count=torch.zeros(1, dtype=torch.int32, device=device),
        all_work_published=torch.zeros(1, dtype=torch.int32, device=device),
        task_head=torch.zeros(1, dtype=torch.int32, device=device),
        task_tail=torch.zeros(1, dtype=torch.int32, device=device),
        task_ready=torch.zeros(dynamic_max_tasks, dtype=torch.int32, device=device),
        task_expert=torch.zeros(dynamic_max_tasks, dtype=torch.int32, device=device),
        task_m_tile=torch.zeros(dynamic_max_tasks, dtype=torch.int32, device=device),
        task_slice_begin=torch.zeros(dynamic_max_tasks, dtype=torch.int32, device=device),
        task_slice_count=torch.zeros(dynamic_max_tasks, dtype=torch.int32, device=device),
        task_valid_rows=torch.zeros(dynamic_max_tasks, dtype=torch.int32, device=device),
        tile_write_count=torch.zeros(dynamic_tiles, dtype=torch.int32, device=device),
    )
    _refresh_dynamic_workspace_scales(
        workspace,
        a1_gscale,
        a2_gscale,
        input_scales_static=input_scales_static,
    )
    _finalize_workspace_views(workspace)
    return workspace


def _get_weight_views(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
    *,
    is_gated: bool = True,
) -> _WeightViews:
    """Create weight views from the expert-weight layout.

    For gated (SiLU): w1 is ``[E, 2*n, k//2]`` (gate+up concatenated).
    For non-gated (relu2): w1 is ``[E, n, k//2]`` (single FC1 weight).
    """
    global _LAST_WEIGHTS
    key = (
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_alphas.data_ptr(),
        is_gated,
    )
    last_wkey, last_wval = _LAST_WEIGHTS
    if last_wkey == key:
        return last_wval
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        _LAST_WEIGHTS = (key, cached)
        return cached

    # Permute [E, w1_n, k//2] → [w1_n, k//2, E] (view, no copy!)
    w13 = w1_fp4.permute(1, 2, 0)     # [w1_n, k//2, E]
    down = w2_fp4.permute(1, 2, 0)    # [k, n//2, E]

    # w13 scale factors (6D MMA view, small contiguous copy)
    w1_n = 2 * n if is_gated else n
    bs_u8 = w1_blockscale.view(torch.uint8)
    w13_sf = as_grouped_scale_view(bs_u8, w1_n, k)
    down_sf = as_grouped_scale_view(w2_blockscale.view(torch.uint8), k, n)

    sf_dtype = cutlass.Float8E4M3FN
    views = _WeightViews(
        w13=w13, down=down,
        w13_sf=w13_sf, down_sf=down_sf,
        w1_alpha=_get_plain_cuda_tensor(w1_alphas),
        w2_alpha=_get_plain_cuda_tensor(w2_alphas),
    )
    # Keep as uint8 for dlpack compatibility — torch float4 types are not
    # supported by dlpack, and sglang may load weights as native float4.
    # The CUTLASS kernel receives the element type via _gptr / compile-time
    # dtype, not from the torch tensor dtype.
    views.w13_fp4 = w13.view(torch.uint8)
    views.down_fp4 = down.view(torch.uint8)
    views.sfb_w13_ptr = make_ptr(sf_dtype, w13_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    views.sfb_down_ptr = make_ptr(sf_dtype, down_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    _WEIGHT_CACHE[key] = views
    _LAST_WEIGHTS = (key, views)
    return views


def _resolve_workspace_layout(
    *,
    num_tokens: int,
    weight_E: int,
    num_topk: int,
) -> tuple[str, int, int]:
    routed_rows = num_tokens * num_topk
    implementation = select_tp_moe_backend(
        num_tokens=num_tokens,
        num_topk=num_topk,
    )
    if implementation == "static":
        return implementation, max(1, routed_rows), max(1, routed_rows)
    return implementation, weight_E, ((routed_rows + 127) // 128) * 128


def _make_workspace_plan(
    *,
    num_tokens: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
    topk_ids: torch.Tensor | None = None,
    eager_exact_dynamic: bool = False,
) -> TPMoEPlan:
    routed_rows = num_tokens * num_topk
    implementation, state_E, max_rows = _resolve_workspace_layout(
        num_tokens=num_tokens,
        weight_E=weight_E,
        num_topk=num_topk,
    )
    dynamic_physical_tiles = None
    dynamic_task_capacity = None
    max_tokens_per_launch = num_tokens
    if implementation == "dynamic":
        if eager_exact_dynamic:
            if topk_ids is None:
                raise ValueError("routing-aware dynamic planning requires topk_ids")
            dynamic_physical_tiles, _, dynamic_task_capacity = _dynamic_task_geometry_from_routing(
                topk_ids,
                weight_E=weight_E,
                n=n,
            )
            max_tokens_per_launch = _eager_dynamic_token_chunk_limit(
                topk_ids,
                weight_E=weight_E,
                k=k,
                n=n,
                num_topk=num_topk,
            )
        else:
            dynamic_physical_tiles, _, dynamic_task_capacity = _dynamic_task_geometry(
                state_E,
                n,
                routed_rows,
            )
            max_tokens_per_launch = _dynamic_token_chunk_limit(weight_E, k, n, num_topk)
    return TPMoEPlan(
        implementation=implementation,
        state_E=state_E,
        weight_E=weight_E,
        routed_rows=routed_rows,
        max_rows=max_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        dtype=dtype,
        max_tokens_per_launch=max_tokens_per_launch,
        dynamic_physical_tiles=dynamic_physical_tiles,
        dynamic_task_capacity=dynamic_task_capacity,
    )


def _validate_workspace(
    workspace: TPMoEWorkspace,
    *,
    plan: TPMoEPlan,
) -> None:
    expected = (
        plan.implementation,
        plan.weight_E,
        plan.k,
        plan.n,
        plan.num_topk,
        plan.device,
        plan.dtype,
    )
    actual = (
        workspace.implementation,
        workspace.weight_E,
        workspace.k,
        workspace.n,
        workspace.num_topk,
        workspace.device,
        workspace.dtype,
    )
    if actual != expected:
        raise ValueError(
            "workspace metadata mismatch: "
            f"expected {(plan.implementation, plan.weight_E, plan.k, plan.n, plan.num_topk, plan.device, plan.dtype)}, "
            f"got {actual}"
        )
    if workspace.state_E < plan.state_E:
        raise ValueError(
            "workspace expert capacity mismatch: "
            f"expected at least {plan.state_E}, got {workspace.state_E}"
        )
    if workspace.max_rows < plan.max_rows:
        raise ValueError(
            "workspace row capacity mismatch: "
            f"expected at least {plan.max_rows}, got {workspace.max_rows}"
        )
    if plan.implementation == "static" and not isinstance(workspace, TPCompactStaticWorkspace):
        raise TypeError("expected a TPCompactStaticWorkspace for the compact static backend")
    if plan.implementation == "dynamic" and not isinstance(workspace, TPDynamicWorkspace):
        raise TypeError("expected a TPDynamicWorkspace for the dynamic backend")
    if isinstance(workspace, TPCompactStaticWorkspace) and workspace.routed_rows_capacity < plan.routed_rows:
        raise ValueError(
            "workspace routed-row capacity mismatch: "
            f"expected at least {plan.routed_rows}, got {workspace.routed_rows_capacity}"
        )
    if isinstance(workspace, TPDynamicWorkspace) and workspace.routed_rows_capacity < plan.routed_rows:
        raise ValueError(
            "workspace routed-row capacity mismatch: "
            f"expected at least {plan.routed_rows}, got {workspace.routed_rows_capacity}"
        )
    if (
        isinstance(workspace, TPDynamicWorkspace)
        and plan.dynamic_physical_tiles is not None
        and workspace.physical_tiles_capacity < plan.dynamic_physical_tiles
    ):
        raise ValueError(
            "workspace physical-tile capacity mismatch: "
            f"expected at least {plan.dynamic_physical_tiles}, got {workspace.physical_tiles_capacity}"
        )
    if (
        isinstance(workspace, TPDynamicWorkspace)
        and plan.dynamic_task_capacity is not None
        and workspace.task_capacity < plan.dynamic_task_capacity
    ):
        raise ValueError(
            "workspace task capacity mismatch: "
            f"expected at least {plan.dynamic_task_capacity}, got {workspace.task_capacity}"
        )


def _workspace_pool_key(
    implementation: str,
    *,
    stream_key: int,
    state_E: int,
    weight_E: int,
    max_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    # Pool-backed static and dynamic workspaces are capacity-based. Avoid
    # exact-shape keys here or long-tail prompt lengths will accumulate one
    # retained workspace per distinct routed-row count.
    if implementation in ("static", "dynamic"):
        state_E = -1
        max_rows = -1
    return (
        implementation,
        stream_key,
        state_E,
        weight_E,
        max_rows,
        k,
        n,
        num_topk,
        device.index or 0,
        dtype,
    )


def _resolve_workspace(
    workspace: TPMoEWorkspace | TPMoEWorkspacePool,
    *,
    plan: TPMoEPlan,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    input_scales_static: bool,
) -> TPMoEWorkspace:
    if isinstance(workspace, TPMoEWorkspace):
        _validate_workspace(workspace, plan=plan)
        if isinstance(workspace, TPDynamicWorkspace):
            _refresh_dynamic_workspace_scales(
                workspace,
                a1_gscale,
                a2_gscale,
                input_scales_static=input_scales_static,
            )
        return workspace

    if not isinstance(workspace, TPMoEWorkspacePool):
        raise TypeError(
            "workspace must be a TPMoEWorkspace or TPMoEWorkspacePool"
        )

    stream_key = int(torch.cuda.current_stream(plan.device).stream_id)
    key = _workspace_pool_key(
        plan.implementation,
        stream_key=stream_key,
        state_E=plan.state_E,
        weight_E=plan.weight_E,
        max_rows=plan.max_rows,
        k=plan.k,
        n=plan.n,
        num_topk=plan.num_topk,
        device=plan.device,
        dtype=plan.dtype,
    )
    resolved = workspace.workspaces.get(key)
    if resolved is None:
        resolved = _alloc_workspace(
            plan.implementation,
            plan.state_E,
            plan.weight_E,
            plan.k,
            plan.n,
            plan.num_topk,
            plan.device,
            plan.dtype,
            a1_gscale,
            a2_gscale,
            routed_rows=plan.routed_rows,
            max_rows=plan.max_rows,
            input_scales_static=input_scales_static,
            dynamic_physical_tiles=plan.dynamic_physical_tiles,
            dynamic_task_capacity=plan.dynamic_task_capacity,
        )
        workspace.workspaces[key] = resolved
        return resolved

    needs_growth = (
        resolved.state_E < plan.state_E
        or resolved.max_rows < plan.max_rows
        or (
            isinstance(resolved, (TPDynamicWorkspace, TPCompactStaticWorkspace))
            and resolved.routed_rows_capacity < plan.routed_rows
        )
        or (
            isinstance(resolved, TPDynamicWorkspace)
            and plan.dynamic_physical_tiles is not None
            and resolved.physical_tiles_capacity < plan.dynamic_physical_tiles
        )
        or (
            isinstance(resolved, TPDynamicWorkspace)
            and plan.dynamic_task_capacity is not None
            and resolved.task_capacity < plan.dynamic_task_capacity
        )
    )
    if needs_growth:
        dynamic_tiles = plan.dynamic_physical_tiles
        dynamic_tasks = plan.dynamic_task_capacity
        if isinstance(resolved, TPDynamicWorkspace):
            dynamic_tiles = max(dynamic_tiles or 0, resolved.physical_tiles_capacity)
            dynamic_tasks = max(dynamic_tasks or 0, resolved.task_capacity)
        resolved = _alloc_workspace(
            plan.implementation,
            max(plan.state_E, resolved.state_E),
            plan.weight_E,
            plan.k,
            plan.n,
            plan.num_topk,
            plan.device,
            plan.dtype,
            a1_gscale,
            a2_gscale,
            routed_rows=max(plan.routed_rows, getattr(resolved, "routed_rows_capacity", 0)),
            max_rows=max(plan.max_rows, resolved.max_rows),
            input_scales_static=input_scales_static,
            dynamic_physical_tiles=dynamic_tiles,
            dynamic_task_capacity=dynamic_tasks,
        )
        workspace.workspaces[key] = resolved
        return resolved

    if isinstance(resolved, TPDynamicWorkspace):
        _refresh_dynamic_workspace_scales(
            resolved,
            a1_gscale,
            a2_gscale,
            input_scales_static=input_scales_static,
        )
    return resolved


def allocate_tp_moe_workspace(
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    input_scales_static: bool = False,
) -> TPMoEWorkspace:
    """Allocate reusable scratch covering one unchunked `b12x_moe_fp4` call."""
    if a.ndim != 2:
        raise ValueError(f"expected input activations with rank 2, got shape {tuple(a.shape)}")
    if topk_ids.ndim != 2:
        raise ValueError(f"expected topk_ids with rank 2, got shape {tuple(topk_ids.shape)}")
    m, k = a.shape
    if topk_ids.shape[0] != m:
        raise ValueError(f"topk_ids batch mismatch: expected {m}, got {topk_ids.shape[0]}")
    weight_E = w1_fp4.shape[0]
    n = w2_fp4.shape[2] * 2
    num_topk = topk_ids.shape[1]
    plan = _make_workspace_plan(
        num_tokens=m,
        weight_E=weight_E,
        k=k,
        n=n,
        num_topk=num_topk,
        device=a.device,
        dtype=a.dtype,
    )
    effective_input_scales_static = (
        input_scales_static
        or (a1_gscale.numel() == 1 and a2_gscale.numel() == 1)
    )
    return _alloc_workspace(
        plan.implementation,
        plan.state_E,
        plan.weight_E,
        plan.k,
        plan.n,
        plan.num_topk,
        plan.device,
        plan.dtype,
        a1_gscale,
        a2_gscale,
        routed_rows=plan.routed_rows,
        max_rows=plan.max_rows,
        input_scales_static=effective_input_scales_static,
        dynamic_physical_tiles=plan.dynamic_physical_tiles,
        dynamic_task_capacity=plan.dynamic_task_capacity,
    )


def allocate_tp_moe_workspace_pool() -> TPMoEWorkspacePool:
    """Allocate an explicit caller-owned workspace pool."""
    return TPMoEWorkspacePool()


def _get_kernel_cache(impl: str) -> Dict[Tuple, Tuple]:
    if impl == "micro":
        return _MICRO_KERNEL_CACHE
    if impl == "static":
        return _STATIC_KERNEL_CACHE
    if impl == "dynamic":
        return _DYNAMIC_KERNEL_CACHE
    raise ValueError(f"unsupported implementation {impl!r}")


def _get_impl_mac(impl: str, *, routed_rows: int | None = None) -> int:
    dev_idx = torch.cuda.current_device()
    key = (dev_idx, impl)
    mac = _MAC_CACHE.get(key)
    sm_count = get_num_sm(torch.device("cuda"))
    mac_limit = min(get_max_active_clusters(1), sm_count)
    override_name = f"B12X_{impl.upper()}_MAX_ACTIVE_CLUSTERS"
    if impl == "dynamic":
        mac_override = _first_env(override_name, "B12X_LEVEL10_MAX_ACTIVE_CLUSTERS")
    else:
        mac_override = _first_env(override_name)
    if mac is None:
        if mac_override is not None:
            mac = max(1, min(int(mac_override), mac_limit))
        else:
            mac = mac_limit
        _MAC_CACHE[key] = mac
    if mac_override is not None:
        return mac
    if routed_rows is not None:
        tuned_mac = lookup_max_active_clusters(
            regime="decode",
            backend=impl,
            routed_rows=int(routed_rows),
        )
        if tuned_mac is not None:
            return max(1, min(int(tuned_mac), mac_limit))
    return mac


def _select_micro_mma_tiler_mn(max_rows: int, n: int) -> tuple[int, int]:
    if os.environ.get("B12X_MOE_TILE_MN"):
        return tuple(int(x) for x in os.environ["B12X_MOE_TILE_MN"].split("x"))
    sm_count = get_num_sm(torch.device("cuda"))
    coarse_tile = (128, 128)
    coarse_tiles = ((max_rows + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    if max_rows <= 128 and coarse_tiles < max(1, sm_count // 2):
        return (64, 128)
    return (128, 128)


def _get_static_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    mac_override: int | None = None,
    activation: str = "silu",
):
    sf_vec_size = 16
    mac = mac_override if mac_override is not None else _get_impl_mac("static")
    routed_rows = m * num_topk
    mma_tiler_mn = (128, 128)
    if num_topk > 1:
        mma_tiler_mn = _select_micro_mma_tiler_mn(routed_rows, n)

    global _LAST_KERNEL
    cache_key = (
        "static", state_E, weight_E, m, k, n, num_topk, max_rows, mac, mma_tiler_mn, topk_ids_dtype,
        input_scales_are_reciprocal, fast_math, activation,
    )
    last_kkey, last_kval = _LAST_KERNEL
    if last_kkey == cache_key:
        return last_kval
    reuse_compiled = os.environ.get("B12X_STATIC_REUSE_COMPILED", "1") != "0"
    if reuse_compiled:
        cached = _STATIC_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            _LAST_KERNEL = (cache_key, cached)
            return cached


    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    if activation == "relu2":
        from b12x.moe.fused import MoEStaticKernelRelu2
        _KernelCls = MoEStaticKernelRelu2
    else:
        _KernelCls = MoEStaticKernel

    kernel = _KernelCls(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1]),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
    )

    rows_pad_k = align_up(max_rows, 128)
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)

    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16,
    )
    topk_ids_cutlass_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype, (m * num_topk,), assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m * num_topk,), assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (max_rows, k, state_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * max_rows * (k // 2),), assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * rows_pad_k * cols_pad_k,), assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    w1_n = n if activation == "relu2" else 2 * n
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (w1_n, k, weight_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (k, n, weight_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (weight_E,), assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E, max_rows), stride_order=(1, 0), assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (state_E, max_rows), stride_order=(1, 0), assumed_align=16,
    )
    compiled = cute.compile(
        kernel,
        a_input_fake, topk_ids_fake, topk_weights_fake,
        packed_a_fake, sfa_fake,
        packed_a_storage_fake, scale_storage_fake,
        barrier_count_fake, barrier_epoch_fake,
        b_w13_fake, sfb_w13_fake,
        b_down_fake, sfb_down_fake,
        row_counts_fake, active_expert_count_fake, weight_expert_ids_fake, global_to_local_expert_fake,
        input_gs_fake, alpha_fake, down_alpha_fake, global_scale_fake,
        scatter_fake, token_map_fake, token_weights_fake,
        mac, current_cuda_stream(),
    )

    result = (compiled, mac)
    if reuse_compiled:
        _STATIC_KERNEL_CACHE[cache_key] = result
    _LAST_KERNEL = (cache_key, result)
    return result


def _get_micro_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    mac_override: int | None = None,
    activation: str = "silu",
):
    sf_vec_size = 16
    mac = mac_override if mac_override is not None else _get_impl_mac("micro")
    routed_rows = m * num_topk
    mma_tiler_mn = _select_micro_mma_tiler_mn(routed_rows, n)

    global _LAST_KERNEL
    cache_key = (
        "micro", state_E, weight_E, m, k, n, num_topk, max_rows, mac, mma_tiler_mn, topk_ids_dtype,
        input_scales_are_reciprocal, fast_math, activation,
    )
    last_kkey, last_kval = _LAST_KERNEL
    if last_kkey == cache_key:
        return last_kval
    reuse_compiled = os.environ.get("B12X_MICRO_REUSE_COMPILED", "1") != "0"
    if reuse_compiled:
        cached = _MICRO_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            _LAST_KERNEL = (cache_key, cached)
            return cached

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    if activation == "relu2":
        from b12x.moe.fused import MoEMicroKernelRelu2
        _KernelCls = MoEMicroKernelRelu2
    else:
        _KernelCls = MoEMicroKernel

    kernel = _KernelCls(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1]),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
    )

    rows_pad_k = align_up(max_rows, 128)
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)

    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16,
    )
    topk_ids_cutlass_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype, (m * num_topk,), assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m * num_topk,), assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (max_rows, k, state_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * max_rows * (k // 2),), assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * rows_pad_k * cols_pad_k,), assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    w1_n = n if activation == "relu2" else 2 * n
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (w1_n, k, weight_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (k, n, weight_E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (weight_E,), assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E, max_rows), stride_order=(1, 0), assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (state_E, max_rows), stride_order=(1, 0), assumed_align=16,
    )
    compiled = cute.compile(
        kernel,
        a_input_fake, topk_ids_fake, topk_weights_fake,
        packed_a_fake, sfa_fake,
        packed_a_storage_fake, scale_storage_fake,
        barrier_count_fake, barrier_epoch_fake,
        b_w13_fake, sfb_w13_fake,
        b_down_fake, sfb_down_fake,
        row_counts_fake, active_expert_count_fake, weight_expert_ids_fake, global_to_local_expert_fake,
        input_gs_fake, alpha_fake, down_alpha_fake, global_scale_fake,
        scatter_fake, token_map_fake, token_weights_fake,
        mac, current_cuda_stream(),
    )

    result = (compiled, mac)
    if reuse_compiled:
        _MICRO_KERNEL_CACHE[cache_key] = result
    _LAST_KERNEL = (cache_key, result)
    return result


class _DynamicMoELaunch:
    """Thin wrapper that makes num_tokens and max_rows runtime Int32."""

    def __init__(self, kernel, k, num_topk):
        self._kernel = kernel
        self._k = k
        self._half_k = k // 2
        self._num_topk = num_topk
        self._cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        packed_a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        packed_a_storage_ptr: cute.Pointer,
        scale_storage_ptr: cute.Pointer,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        producers_done_count: cute.Tensor,
        all_work_published: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_ready_ptr: cute.Pointer,
        task_expert_ptr: cute.Pointer,
        task_m_tile_ptr: cute.Pointer,
        task_slice_begin_ptr: cute.Pointer,
        task_slice_count_ptr: cute.Pointer,
        task_valid_rows_ptr: cute.Pointer,
        tile_write_count_ptr: cute.Pointer,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_ptr: cute.Pointer,
        token_map_ptr: cute.Pointer,
        token_weights_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        max_rows: cutlass.Int32,
        rows_padded: cutlass.Int32,
        max_tasks: cutlass.Int32,
        max_phys_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        a_input = cute.make_tensor(a_ptr, layout=cute.make_layout(
            (num_tokens, self._k), stride=(self._k, 1)))
        topk_ids = cute.make_tensor(topk_ids_ptr, layout=cute.make_layout(
            (num_tokens * self._num_topk,), stride=(1,)))
        topk_weights_t = cute.make_tensor(topk_weights_ptr, layout=cute.make_layout(
            (num_tokens * self._num_topk,), stride=(1,)))
        scatter_output = cute.make_tensor(scatter_ptr, layout=cute.make_layout(
            (num_tokens, self._k), stride=(self._k, 1)))
        packed_a = cute.make_tensor(packed_a_ptr, layout=cute.make_layout(
            (rows_padded, self._k, 1), stride=(self._k, 1, rows_padded * self._k)))
        packed_a_storage = cute.make_tensor(packed_a_storage_ptr, layout=cute.make_layout(
            (rows_padded * self._half_k,), stride=(1,)))
        scale_storage = cute.make_tensor(scale_storage_ptr, layout=cute.make_layout(
            (rows_padded * self._cols_pad_k,), stride=(1,)))
        token_map = cute.make_tensor(token_map_ptr, layout=cute.make_layout(
            (rows_padded,), stride=(1,)))
        token_weights_t = cute.make_tensor(token_weights_ptr, layout=cute.make_layout(
            (rows_padded,), stride=(1,)))
        task_ready = cute.make_tensor(task_ready_ptr, layout=cute.make_layout(
            (max_tasks,), stride=(1,)))
        task_expert = cute.make_tensor(task_expert_ptr, layout=cute.make_layout(
            (max_tasks,), stride=(1,)))
        task_m_tile = cute.make_tensor(task_m_tile_ptr, layout=cute.make_layout(
            (max_tasks,), stride=(1,)))
        task_slice_begin = cute.make_tensor(task_slice_begin_ptr, layout=cute.make_layout(
            (max_tasks,), stride=(1,)))
        task_slice_count = cute.make_tensor(task_slice_count_ptr, layout=cute.make_layout(
            (max_tasks,), stride=(1,)))
        task_valid_rows = cute.make_tensor(task_valid_rows_ptr, layout=cute.make_layout(
            (max_tasks,), stride=(1,)))
        tile_write_count = cute.make_tensor(tile_write_count_ptr, layout=cute.make_layout(
            (max_phys_tiles,), stride=(1,)))
        self._kernel(
            a_input, topk_ids, topk_weights_t,
            packed_a, sfa_ptr, packed_a_storage, scale_storage,
            barrier_count, barrier_epoch,
            pair_head, producers_done_count, all_work_published,
            task_head, task_tail, task_ready,
            task_expert, task_m_tile,
            task_slice_begin, task_slice_count, task_valid_rows,
            tile_write_count,
            b_w13, sfb_w13_ptr,
            b_down, sfb_down_ptr,
            row_counts, expert_write_rows, expert_tile_base,
            input_global_scale, alpha, down_alpha, global_scale,
            scatter_output, token_map, token_weights_t,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )


def _get_dynamic_kernel(
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    mac_override: int | None = None,
    activation: str = "silu",
):
    sf_vec_size = 16
    mac = mac_override if mac_override is not None else _get_impl_mac("dynamic")

    global _LAST_KERNEL
    cache_key = (
        "dynamic", E, k, n, num_topk, mac, topk_ids_dtype,
        input_scales_are_reciprocal, fast_math, activation,
    )
    last_kkey, last_kval = _LAST_KERNEL
    if last_kkey == cache_key:
        return last_kval
    reuse_compiled = _first_env("B12X_DYNAMIC_REUSE_COMPILED", "B12X_LEVEL10_REUSE_COMPILED")
    if reuse_compiled is None:
        reuse_compiled = "1"
    reuse_compiled = reuse_compiled != "0"
    if reuse_compiled:
        cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            _LAST_KERNEL = (cache_key, cached)
            return cached

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    if activation == "relu2":
        from b12x.moe.fused import MoEDynamicKernelRelu2
        _KernelCls = MoEDynamicKernelRelu2
    else:
        _KernelCls = MoEDynamicKernel

    kernel = _KernelCls(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=(_LEVEL_TILE_M, _LEVEL_TILE_N),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
    )
    launch = _DynamicMoELaunch(kernel, k=k, num_topk=num_topk)

    topk_ids_cutlass_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8

    # a_input, topk_ids, topk_weights, scatter_output are pointers — shapes
    # are constructed at runtime from num_tokens Int32.
    a_input_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_ids_fake = make_ptr(topk_ids_cutlass_dtype, topk_ids_align, cute.AddressSpace.gmem, assumed_align=topk_ids_align)
    topk_weights_fake = make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4)

    packed_a_fake = make_ptr(ab_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    scale_storage_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    pair_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    producers_done_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    all_work_published_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    task_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    task_tail_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4,
    )
    tasks_ph = 1
    tiles_ph = 1
    task_ready_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    task_expert_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    task_m_tile_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    task_slice_begin_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    task_slice_count_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    task_valid_rows_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    tile_write_count_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    w1_n = n if activation == "relu2" else 2 * n
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (w1_n, k, E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (k, n, E), stride_order=(1, 0, 2), assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4,
    )
    expert_write_rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4,
    )
    expert_tile_base_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E + 1,), assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16,
    )
    scatter_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    compiled = cute.compile(
        launch,
        a_input_fake, topk_ids_fake, topk_weights_fake,
        packed_a_fake, sfa_fake,
        packed_a_storage_fake, scale_storage_fake,
        barrier_count_fake, barrier_epoch_fake,
        pair_head_fake, producers_done_count_fake, all_work_published_fake,
        task_head_fake, task_tail_fake, task_ready_fake,
        task_expert_fake, task_m_tile_fake,
        task_slice_begin_fake, task_slice_count_fake, task_valid_rows_fake,
        tile_write_count_fake,
        b_w13_fake, sfb_w13_fake,
        b_down_fake, sfb_down_fake,
        row_counts_fake, expert_write_rows_fake, expert_tile_base_fake,
        input_gs_fake, alpha_fake, down_alpha_fake, global_scale_fake,
        scatter_fake, token_map_fake, token_weights_fake,
        1, 1, 1, 1, 1, mac, current_cuda_stream(),
    )

    result = (compiled, mac)
    if reuse_compiled:
        _DYNAMIC_KERNEL_CACHE[cache_key] = result
    _LAST_KERNEL = (cache_key, result)
    return result


def _launch_dynamic(
    *,
    workspace: TPDynamicWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    scatter_output: torch.Tensor,
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    routed_rows: int,
    max_rows: int,
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    stream,
    activation: str = "silu",
) -> None:
    effective_mac = _get_impl_mac("dynamic", routed_rows=routed_rows)
    if not _dynamic_multicta_enabled():
        effective_mac = 1
    compiled, mac = _get_dynamic_kernel(
        E, m, k, n, num_topk, max_rows,
        topk_ids_dtype=topk_ids_dtype,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        mac_override=effective_mac,
        activation=activation,
    )
    _gptr = lambda dtype, t, align=16: make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)
    ids_cutlass_dtype = cutlass.Int32 if flat_ids.dtype == torch.int32 else cutlass.Int64
    ids_align = 4 if flat_ids.dtype == torch.int32 else 8
    compiled(
        _gptr(cutlass.BFloat16, a),
        _gptr(ids_cutlass_dtype, flat_ids, ids_align),
        _gptr(cutlass.Float32, flat_weights, 4),
        _gptr(cutlass.Float4E2M1FN, workspace.packed_a_view),
        workspace.sfa_ptr,
        _gptr(cutlass.Uint8, workspace.packed_a_flat),
        _gptr(cutlass.Uint8, workspace.scale_flat),
        workspace.barrier_count, workspace.barrier_epoch,
        workspace.pair_head, workspace.producers_done_count, workspace.all_work_published,
        workspace.task_head, workspace.task_tail,
        _gptr(cutlass.Int32, workspace.task_ready, 4),
        _gptr(cutlass.Int32, workspace.task_expert, 4),
        _gptr(cutlass.Int32, workspace.task_m_tile, 4),
        _gptr(cutlass.Int32, workspace.task_slice_begin, 4),
        _gptr(cutlass.Int32, workspace.task_slice_count, 4),
        _gptr(cutlass.Int32, workspace.task_valid_rows, 4),
        _gptr(cutlass.Int32, workspace.tile_write_count, 4),
        weights.w13_fp4, weights.sfb_w13_ptr,
        weights.down_fp4, weights.sfb_down_ptr,
        workspace.row_counts, workspace.expert_write_rows, workspace.expert_tile_base,
        workspace.input_gs, weights.w1_alpha, weights.w2_alpha, workspace.down_input_scale,
        _gptr(cutlass.BFloat16, scatter_output),
        _gptr(cutlass.Int32, workspace.token_map, 4),
        _gptr(cutlass.Float32, workspace.token_weights, 4),
        m, max_rows,
        workspace.physical_tiles_capacity * _LEVEL_TILE_M,
        workspace.task_capacity,
        workspace.physical_tiles_capacity,
        mac, stream,
    )


def _launch_compact_static(
    *,
    workspace: TPCompactStaticWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    routed_rows: int,
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    stream,
    activation: str = "silu",
) -> None:
    micro_cutover_pairs = _get_micro_compact_cutover_pairs()
    if (
        micro_cutover_pairs == _MICRO_COMPACT_CUTOVER_PAIRS_DEFAULT
        and num_topk > 1
    ):
        micro_cutover_pairs = _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK_DEFAULT
    use_micro = routed_rows <= micro_cutover_pairs
    static_mac = _get_impl_mac("static", routed_rows=routed_rows)
    if not use_micro and routed_rows < 40:
        # Tiny compact launches have very little FC2 tile work, so capping
        # resident clusters avoids idle CTA participation in the barrier phases.
        static_mac = min(static_mac, 64)
    if use_micro:
        # Micro work can cover at most one m-tile per routed pair and one
        # FC2 output tile per N tile. Launching more persistent CTAs than that
        # upper bound only creates idle clusters that sit through the grid
        # barriers without owning useful work.
        micro_work_tiles = max(1, routed_rows * max(1, (n + 128 - 1) // 128))
        micro_mac = min(_get_impl_mac("micro", routed_rows=routed_rows), micro_work_tiles)
        compact_ids = workspace.compact_topk_ids[: flat_ids.numel()]
        triton_compact_topk_ids(
            flat_ids,
            compact_ids,
            workspace.weight_expert_ids,
            workspace.active_expert_count,
        )
        compiled, mac = _get_micro_kernel(
            workspace.state_E, weight_E, m, k, n, num_topk, workspace.max_rows,
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            mac_override=micro_mac,
            activation=activation,
        )
        launch_ids = compact_ids
    else:
        compiled, mac = _get_static_kernel(
            workspace.state_E, weight_E, m, k, n, num_topk, workspace.max_rows,
            topk_ids_dtype=topk_ids_dtype,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            mac_override=static_mac,
            activation=activation,
        )
        launch_ids = flat_ids
    compiled(
        a, launch_ids, flat_weights,
        workspace.packed_a_view, workspace.sfa_ptr,
        workspace.packed_a_flat, workspace.scale_flat,
        workspace.barrier_count, workspace.barrier_epoch,
        weights.w13_fp4, weights.sfb_w13_ptr,
        weights.down_fp4, weights.sfb_down_ptr,
        workspace.row_counts, workspace.active_expert_count, workspace.weight_expert_ids, workspace.global_to_local_expert,
        input_gs, weights.w1_alpha, weights.w2_alpha, down_input_scale,
        scatter_output, workspace.token_map, workspace.token_weights,
        mac, stream,
    )


@torch._dynamo.disable
def b12x_moe_fp4(
    a: torch.Tensor,           # [m, k] bf16 activations
    a1_gscale: torch.Tensor,   # [E] or scalar — input quant scale
    w1_fp4: torch.Tensor,      # [E, 2*n, k//2] uint8
    w1_blockscale: torch.Tensor,  # [E, ...] float8_e4m3fn swizzled
    w1_alphas: torch.Tensor,   # [E] float32
    a2_gscale: torch.Tensor,   # [E] or scalar — intermediate quant scale
    w2_fp4: torch.Tensor,      # [E, k, n//2] uint8
    w2_blockscale: torch.Tensor,  # [E, ...] float8_e4m3fn swizzled
    w2_alphas: torch.Tensor,   # [E] float32
    topk_weights: torch.Tensor,  # [m, topk] float
    topk_ids: torch.Tensor,    # [m, topk] int
    apply_router_weight_on_input: bool = False,
    *,
    workspace: TPMoEWorkspace | TPMoEWorkspacePool,
    output: torch.Tensor | None = None,
    input_scales_are_reciprocal: bool = False,
    input_scales_static: bool = False,
    fast_math: bool | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    """MoE with shape-selected fused static or dynamic kernels.

    Compact workloads use the graph-safe static backend. All larger routed
    workloads use dynamic. Large token batches are chunked only when the chosen
    backend cannot describe the required work buffers in a single launch.
    """
    if activation not in {"silu", "relu2"}:
        raise ValueError(f"unsupported activation {activation!r}")
    m, k = a.shape
    E = w1_fp4.shape[0]
    weight_E = E
    n = w2_fp4.shape[2] * 2  # intermediate_size
    expected_w1_rows = n if activation == "relu2" else 2 * n
    if w1_fp4.shape[1] != expected_w1_rows:
        raise ValueError(
            f"expected w1_fp4.shape[1] == {expected_w1_rows} for activation "
            f"{activation!r}, got {w1_fp4.shape[1]}"
        )
    num_topk = topk_ids.shape[1]
    routed_rows = m * num_topk
    device = a.device
    workspace_policy = _workspace_policy(workspace)
    plan = _make_workspace_plan(
        num_tokens=m,
        weight_E=weight_E,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        dtype=a.dtype,
        topk_ids=topk_ids,
        eager_exact_dynamic=workspace_policy.eager_exact_dynamic,
    )

    if apply_router_weight_on_input:
        raise NotImplementedError("apply_router_weight_on_input is not implemented in b12x_moe_fp4")
    if fast_math is None:
        fast_math = _FAST_MATH_DEFAULT
    # Shared scalar input scales are weight-side constants in the benchmarked
    # path, so treat them as static and avoid re-expanding them every launch.
    effective_input_scales_static = (
        input_scales_static
        or (a1_gscale.numel() == 1 and a2_gscale.numel() == 1)
    )

    impl = plan.implementation
    max_rows = plan.max_rows
    if impl == "dynamic" and m > plan.max_tokens_per_launch:
        if not workspace_policy.can_chunk:
            raise ValueError(
                "chunked requests require a TPMoEWorkspacePool; "
                "an exact TPMoEWorkspace only supports one launch shape"
            )
        chunk_output = output
        if chunk_output is None:
            chunk_output = torch.empty(m, k, dtype=a.dtype, device=device)
        for start in range(0, m, plan.max_tokens_per_launch):
            end = min(start + plan.max_tokens_per_launch, m)
            b12x_moe_fp4(
                a[start:end],
                a1_gscale,
                w1_fp4,
                w1_blockscale,
                w1_alphas,
                a2_gscale,
                w2_fp4,
                w2_blockscale,
                w2_alphas,
                topk_weights[start:end],
                topk_ids[start:end],
                apply_router_weight_on_input=apply_router_weight_on_input,
                output=chunk_output[start:end],
                workspace=workspace,
                input_scales_are_reciprocal=input_scales_are_reciprocal,
                input_scales_static=effective_input_scales_static,
                fast_math=fast_math,
                activation=activation,
            )
        return chunk_output

    s = _resolve_workspace(
        workspace,
        plan=plan,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        input_scales_static=effective_input_scales_static,
    )

    # CUDA graph capture may run on a non-default stream, so the launch stream
    # must be fetched per-call rather than cached per-device.
    stream = current_cuda_stream()

    if impl == "static":
        assert isinstance(s, TPCompactStaticWorkspace)
        flat_ids = _flatten_routing_ids(topk_ids)
        flat_weights = _flatten_routing_weights(topk_weights)

        wv = _get_weight_views(
            w1_fp4,
            w1_blockscale,
            w2_fp4,
            w2_blockscale,
            w1_alphas,
            w2_alphas,
            n,
            k,
            is_gated=(activation != "relu2"),
        )
        input_gs = _prepare_expert_scale(a1_gscale, weight_E)
        down_input_scale = _prepare_expert_scale(a2_gscale, weight_E)
    else:
        assert isinstance(s, TPDynamicWorkspace)
        wv = _get_weight_views(
            w1_fp4,
            w1_blockscale,
            w2_fp4,
            w2_blockscale,
            w1_alphas,
            w2_alphas,
            n,
            k,
            is_gated=(activation != "relu2"),
        )
        input_gs = s.input_gs
        down_input_scale = s.down_input_scale
        flat_ids = _flatten_routing_ids(topk_ids)
        flat_weights = _flatten_routing_weights(topk_weights)

    if output is None:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("CUDA graph capture requires a caller-owned output buffer")
        scatter_output = torch.zeros(m, k, dtype=a.dtype, device=device)
    else:
        scatter_output = output
    if scatter_output.shape != (m, k):
        raise ValueError(f"output must have shape {(m, k)}, got {tuple(scatter_output.shape)}")
    if scatter_output.dtype != a.dtype:
        raise ValueError(f"output must have dtype {a.dtype}, got {scatter_output.dtype}")
    if scatter_output.device != device:
        raise ValueError(f"output must be on device {device}, got {scatter_output.device}")
    if not scatter_output.is_contiguous():
        raise ValueError("output must be contiguous")

    if impl == "dynamic":
        _launch_dynamic(
            workspace=s,
            weights=wv,
            a=a,
            flat_ids=flat_ids,
            flat_weights=flat_weights,
            scatter_output=scatter_output,
            E=E,
            m=m,
            k=k,
            n=n,
            num_topk=num_topk,
            routed_rows=routed_rows,
            max_rows=max_rows,
            topk_ids_dtype=flat_ids.dtype,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            stream=stream,
            activation=activation,
        )
    else:
        _launch_compact_static(
            workspace=s,
            weights=wv,
            a=a,
            flat_ids=flat_ids,
            flat_weights=flat_weights,
            input_gs=input_gs,
            down_input_scale=down_input_scale,
            scatter_output=scatter_output,
            weight_E=weight_E,
            m=m,
            k=k,
            n=n,
            num_topk=num_topk,
            routed_rows=routed_rows,
            topk_ids_dtype=flat_ids.dtype,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            stream=stream,
            activation=activation,
        )
    return scatter_output


def _validate_sparse_routing(hidden_states: torch.Tensor, routing: B12XTopKRouting) -> None:
    if routing.topk_ids.ndim != 2:
        raise ValueError(
            f"expected topk_ids with rank 2, got shape {tuple(routing.topk_ids.shape)}"
        )
    if routing.topk_weights.ndim != 2:
        raise ValueError(
            "expected topk_weights with rank 2, got shape "
            f"{tuple(routing.topk_weights.shape)}"
        )
    if routing.topk_ids.shape != routing.topk_weights.shape:
        raise ValueError(
            "topk_ids and topk_weights must have the same shape, got "
            f"{tuple(routing.topk_ids.shape)} and {tuple(routing.topk_weights.shape)}"
        )
    if routing.topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "routing batch mismatch: expected "
            f"{hidden_states.shape[0]}, got {routing.topk_ids.shape[0]}"
        )
    if routing.router_logits is not None and routing.router_logits.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "router_logits batch mismatch: expected "
            f"{hidden_states.shape[0]}, got {routing.router_logits.shape[0]}"
        )
    if routing.flat_ids is not None and routing.flat_ids.numel() != routing.topk_ids.numel():
        raise ValueError(
            "flat_ids size mismatch: expected "
            f"{routing.topk_ids.numel()}, got {routing.flat_ids.numel()}"
        )
    if routing.flat_weights is not None and routing.flat_weights.numel() != routing.topk_weights.numel():
        raise ValueError(
            "flat_weights size mismatch: expected "
            f"{routing.topk_weights.numel()}, got {routing.flat_weights.numel()}"
        )


def _alloc_route_workspace(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    logits_dtype: torch.dtype,
) -> _TPRouteWorkspace:
    return _TPRouteWorkspace(
        router_logits=torch.empty(num_tokens, num_experts, device=device, dtype=logits_dtype),
        topk_logits=torch.empty(num_tokens, top_k, device=device, dtype=torch.float32),
        topk_ids=torch.empty(num_tokens, top_k, device=device, dtype=torch.int32),
        topk_weights=torch.empty(num_tokens, top_k, device=device, dtype=torch.float32),
    )


def _get_route_workspace(
    hidden_states: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    logits_dtype: torch.dtype,
    workspace: TPMoEWorkspace | TPMoEWorkspacePool | None,
) -> _TPRouteWorkspace | None:
    if workspace is None:
        return None

    m = hidden_states.shape[0]
    device = hidden_states.device

    if isinstance(workspace, TPMoEWorkspacePool):
        key = (
            int(torch.cuda.current_stream(device=device).stream_id),
            device.index,
            m,
            num_experts,
            top_k,
            logits_dtype,
        )
        route_workspace = workspace.route_workspaces.get(key)
        if route_workspace is None:
            route_workspace = _alloc_route_workspace(
                num_tokens=m,
                num_experts=num_experts,
                top_k=top_k,
                device=device,
                logits_dtype=logits_dtype,
            )
            workspace.route_workspaces[key] = route_workspace
        return route_workspace

    route_workspace = workspace.route_workspace
    if (
        route_workspace is None
        or route_workspace.router_logits.shape != (m, num_experts)
        or route_workspace.topk_logits.shape != (m, top_k)
        or route_workspace.router_logits.dtype != logits_dtype
        or route_workspace.router_logits.device != device
    ):
        route_workspace = _alloc_route_workspace(
            num_tokens=m,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            logits_dtype=logits_dtype,
        )
        workspace.route_workspace = route_workspace
    return route_workspace


def _select_experts_reference(
    hidden_states: torch.Tensor,
    *,
    top_k: int,
    gate_weight: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
    renormalize: bool = True,
) -> B12XTopKRouting:
    """Reference routing selection for sparse-block MoE wrappers.

    Keep this path simple and obviously correct. Optimized routing should live
    in a separate public fast path rather than accreting special cases here.
    """

    if hidden_states.ndim != 2:
        raise ValueError(
            "expected hidden_states with rank 2, got shape "
            f"{tuple(hidden_states.shape)}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if router_logits is not None and gate_weight is not None:
        raise ValueError("pass either router_logits or gate_weight, not both")
    if router_logits is None and gate_weight is None:
        raise ValueError("expected router_logits or gate_weight")

    if router_logits is None:
        assert gate_weight is not None
        if gate_weight.ndim != 2:
            raise ValueError(
                f"expected gate_weight with rank 2, got shape {tuple(gate_weight.shape)}"
            )
        if gate_weight.shape[1] != hidden_states.shape[1]:
            raise ValueError(
                "gate_weight hidden-size mismatch: expected "
                f"{hidden_states.shape[1]}, got {gate_weight.shape[1]}"
            )
        if gate_bias is not None:
            if gate_bias.ndim != 1:
                raise ValueError(
                    f"expected gate_bias with rank 1, got shape {tuple(gate_bias.shape)}"
                )
            if gate_bias.shape[0] != gate_weight.shape[0]:
                raise ValueError(
                    "gate_bias expert mismatch: expected "
                    f"{gate_weight.shape[0]}, got {gate_bias.shape[0]}"
                )
        router_logits = F.linear(hidden_states, gate_weight, gate_bias)
    else:
        if router_logits.ndim != 2:
            raise ValueError(
                "expected router_logits with rank 2, got shape "
                f"{tuple(router_logits.shape)}"
            )
        if router_logits.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "router_logits batch mismatch: expected "
                f"{hidden_states.shape[0]}, got {router_logits.shape[0]}"
            )

    num_experts = router_logits.shape[1]
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")

    topk_logits, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
    if renormalize:
        topk_weights = torch.softmax(topk_logits.to(torch.float32), dim=-1)
    else:
        topk_weights = topk_logits.to(torch.float32)
    return B12XTopKRouting(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )


def b12x_route_experts_fast(
    hidden_states: torch.Tensor,
    *,
    top_k: int,
    gate_weight: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
    renormalize: bool = True,
    workspace: TPMoEWorkspace | TPMoEWorkspacePool | None = None,
) -> B12XTopKRouting:
    """Public sparse-routing entrypoint for higher-level integrations.

    This is the optimization seam for future fast routing work. The current
    implementation preserves the simple reference math, but when a caller-owned
    workspace is available it reuses route scratch buffers for the gate logits
    and top-k outputs. Returned tensors may therefore alias mutable workspace
    scratch and should be cloned by callers that want to retain them across
    subsequent launches on the same workspace.
    """
    if hidden_states.ndim != 2:
        raise ValueError(
            "expected hidden_states with rank 2, got shape "
            f"{tuple(hidden_states.shape)}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if router_logits is not None and gate_weight is not None:
        raise ValueError("pass either router_logits or gate_weight, not both")
    if router_logits is None and gate_weight is None:
        raise ValueError("expected router_logits or gate_weight")

    if router_logits is None:
        assert gate_weight is not None
        if gate_weight.ndim != 2:
            raise ValueError(
                f"expected gate_weight with rank 2, got shape {tuple(gate_weight.shape)}"
            )
        if gate_weight.shape[1] != hidden_states.shape[1]:
            raise ValueError(
                "gate_weight hidden-size mismatch: expected "
                f"{hidden_states.shape[1]}, got {gate_weight.shape[1]}"
            )
        if gate_bias is not None:
            if gate_bias.ndim != 1:
                raise ValueError(
                    f"expected gate_bias with rank 1, got shape {tuple(gate_bias.shape)}"
                )
            if gate_bias.shape[0] != gate_weight.shape[0]:
                raise ValueError(
                    "gate_bias expert mismatch: expected "
                    f"{gate_weight.shape[0]}, got {gate_bias.shape[0]}"
                )
        num_experts = gate_weight.shape[0]
        logits_dtype = torch.result_type(hidden_states, gate_weight)
    else:
        if router_logits.ndim != 2:
            raise ValueError(
                "expected router_logits with rank 2, got shape "
                f"{tuple(router_logits.shape)}"
            )
        if router_logits.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "router_logits batch mismatch: expected "
                f"{hidden_states.shape[0]}, got {router_logits.shape[0]}"
            )
        num_experts = router_logits.shape[1]
        logits_dtype = router_logits.dtype

    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")

    if not hidden_states.is_cuda or num_experts > 1024:
        selected = _select_experts_reference(
            hidden_states,
            top_k=top_k,
            gate_weight=gate_weight,
            gate_bias=gate_bias,
            router_logits=router_logits,
            renormalize=renormalize,
        )
        topk_ids_i32 = selected.topk_ids.to(torch.int32)
        return B12XTopKRouting(
            topk_weights=selected.topk_weights,
            topk_ids=topk_ids_i32,
            router_logits=selected.router_logits,
            flat_ids=topk_ids_i32.view(-1),
            flat_weights=selected.topk_weights.reshape(-1),
        )

    route_workspace = _get_route_workspace(
        hidden_states,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
        workspace=workspace,
    )
    if route_workspace is None:
        route_workspace = _alloc_route_workspace(
            num_tokens=hidden_states.shape[0],
            num_experts=num_experts,
            top_k=top_k,
            device=hidden_states.device,
            logits_dtype=logits_dtype,
        )

    if router_logits is None:
        assert gate_weight is not None
        torch.mm(hidden_states, gate_weight.t(), out=route_workspace.router_logits)
        if gate_bias is not None:
            route_workspace.router_logits.add_(gate_bias.to(route_workspace.router_logits.dtype))
        router_logits = route_workspace.router_logits
    else:
        if not router_logits.is_contiguous():
            route_workspace.router_logits.copy_(router_logits)
            router_logits = route_workspace.router_logits

    triton_route_topk(
        router_logits,
        route_workspace.topk_logits,
        route_workspace.topk_ids,
        route_workspace.topk_weights,
        renormalize=renormalize,
    )
    topk_ids = route_workspace.topk_ids
    topk_weights = route_workspace.topk_weights

    return B12XTopKRouting(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
        flat_ids=topk_ids.view(-1),
        flat_weights=topk_weights.view(-1),
    )


def b12x_sparse_moe_fp4(
    hidden_states: torch.Tensor,
    *,
    experts: B12XFP4ExpertWeights,
    workspace: TPMoEWorkspace | TPMoEWorkspacePool,
    routing: B12XTopKRouting | None = None,
    top_k: int | None = None,
    gate_weight: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
    renormalize_topk: bool = True,
    routed_scaling_factor: float = 1.0,
    output: torch.Tensor | None = None,
    return_routing: bool = False,
    input_scales_are_reciprocal: bool = False,
    input_scales_static: bool = False,
    fast_math: bool | None = None,
    activation: str = "silu",
) -> torch.Tensor | tuple[torch.Tensor, B12XTopKRouting]:
    """Sparse-block FP4 MoE wrapper above the routed-expert TP primitive.

    This additive entrypoint preserves `b12x_moe_fp4(...)` as the stable
    low-level contract while giving higher-level integrations a single call that
    can own `gate -> topk -> routed experts` at the sparse MoE block seam.
    """

    if routing is not None:
        if top_k is not None or gate_weight is not None or gate_bias is not None or router_logits is not None:
            raise ValueError(
                "routing is mutually exclusive with top_k/gate_weight/gate_bias/router_logits"
            )
        selected = routing
    else:
        if top_k is None:
            raise ValueError("top_k is required when routing is not provided")
        selected = b12x_route_experts_fast(
            hidden_states,
            top_k=top_k,
            gate_weight=gate_weight,
            gate_bias=gate_bias,
            router_logits=router_logits,
            renormalize=renormalize_topk,
            workspace=workspace,
        )

    _validate_sparse_routing(hidden_states, selected)

    routed_output = b12x_moe_fp4(
        hidden_states,
        experts.a1_gscale,
        experts.w1_fp4,
        experts.w1_blockscale,
        experts.w1_alphas,
        experts.a2_gscale,
        experts.w2_fp4,
        experts.w2_blockscale,
        experts.w2_alphas,
        selected.topk_weights,
        selected.topk_ids,
        workspace=workspace,
        output=output,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        input_scales_static=input_scales_static,
        fast_math=fast_math,
        activation=activation,
    )
    if routed_scaling_factor != 1.0:
        routed_output.mul_(routed_scaling_factor)
    if return_routing:
        return routed_output, selected
    return routed_output
