"""Tensor-parallel MoE entrypoints backed by fused CuTe DSL kernels."""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from torch.profiler import record_function

from b12x.cute.fp4 import align_up, as_grouped_scale_view
from b12x.cute.utils import (
    current_cuda_stream,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
)
from cutlass.cutlass_dsl import Int32
from b12x.integration.triton_route import route_topk as triton_route_topk
from b12x.moe.fused.relu2 import (
    MoEDynamicKernelRelu2,
    MoEMicroKernelRelu2,
    MoEStaticKernelRelu2,
)
from b12x.moe.fused.silu import (
    MoEDynamicKernelSilu,
    MoEMicroKernelSilu,
    MoEStaticKernelSilu,
)
from b12x.moe.fused.micro import (
    _BLOCK_DIM as _DIRECT_MICRO_BLOCK_DIM,
    _direct_k_segments_for_k,
    _direct_k_segments_supported,
)
from b12x.moe.tuning import lookup_max_active_clusters
from b12x.runtime_control import raise_if_kernel_resolution_frozen

_NVFP4_BLOCK_SIZE = 16
_RUNTIME_MEMREF_LIMIT = (1 << 31) - 1
_LEVEL_TILE_M = 128
_LEVEL_TILE_N = 128
_DYNAMIC_SLICE_CHUNK = 1
_MOE_FORCE_A16_ENV = "B12X_MOE_FORCE_A16"
_FP4_SOURCE_FORMATS = {
    "modelopt": "modelopt",
    "compressed_tensors": "compressed_tensors",
    "compressed-tensors": "compressed_tensors",
    "ct": "compressed_tensors",
}


@dataclass(kw_only=True)
class TPMoEWorkspace:
    """Reusable scratch buffers for one `b12x_moe_fp4` shape family."""

    implementation: str
    quant_mode: str
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
    volatile_launch_state: bool = False


@dataclass(kw_only=True)
class TPCompactStaticWorkspace(TPMoEWorkspace):
    routed_rows_capacity: int
    active_expert_count: torch.Tensor
    weight_expert_ids: torch.Tensor
    global_to_local_expert: torch.Tensor
    compact_topk_ids: torch.Tensor
    micro_intermediate: torch.Tensor


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


@dataclass(kw_only=True)
class TPW4A16Workspace:
    implementation: str
    quant_mode: str
    activation: str
    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    dtype: torch.dtype
    routed_rows_capacity: int
    intermediate_cache13: torch.Tensor
    intermediate_cache2: torch.Tensor
    fc1_c_tmp: torch.Tensor
    fc2_c_tmp: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    route_workspace: "_TPRouteWorkspace | None" = None
    volatile_launch_state: bool = False


@dataclass
class TPMoEWorkspacePool:
    """Caller-owned capacity-based workspace cache for one execution lane.

    A single explicit pool may be shared across layers in a lane. Independent
    overlapping lanes must use distinct pools; internal fork/join streams share
    the lane pool and therefore the same scratch arena.
    """

    workspaces: Dict[Tuple, object] = field(default_factory=dict)
    route_workspaces: Dict[Tuple, "_TPRouteWorkspace"] = field(default_factory=dict)
    core_arenas: Dict[Tuple, "_TPCoreArena"] = field(default_factory=dict)
    shared_arena: torch.Tensor | None = None
    shared_arena_nbytes: int = 0
    route_workspace_nbytes: int = 0
    core_arena_offset_bytes: int = 0
    core_arena_nbytes: int = 0
    frozen: bool = False

    def clear(self) -> None:
        self.workspaces.clear()
        self.route_workspaces.clear()
        self.core_arenas.clear()

    def bind_shared_arena(
        self,
        shared_arena: torch.Tensor,
        *,
        route_workspace_nbytes: int,
        core_workspace_nbytes: int,
        frozen: bool = True,
    ) -> None:
        if shared_arena.dtype != torch.uint8:
            raise TypeError(
                f"shared_arena must have dtype torch.uint8, got {shared_arena.dtype}"
            )
        route_workspace_nbytes = align_up(max(int(route_workspace_nbytes), 0), 16)
        core_workspace_nbytes = max(int(core_workspace_nbytes), 0)
        required = route_workspace_nbytes + core_workspace_nbytes
        if shared_arena.numel() < max(required, 1):
            raise ValueError(
                f"shared_arena has {shared_arena.numel()} bytes, but MoE workspace requires {required}"
            )
        self.clear()
        self.shared_arena = shared_arena
        self.shared_arena_nbytes = int(shared_arena.numel())
        self.route_workspace_nbytes = route_workspace_nbytes
        self.core_arena_offset_bytes = route_workspace_nbytes
        self.core_arena_nbytes = core_workspace_nbytes
        self.frozen = bool(frozen)


@dataclass(frozen=True, kw_only=True)
class B12XFP4ExpertWeights:
    """Packaged FP4 expert tensors for routed-expert MoE entrypoints."""

    a1_gscale: torch.Tensor  # reciprocal activation global scale for FC1 input
    w1_fp4: torch.Tensor
    w1_blockscale: torch.Tensor
    w1_alphas: torch.Tensor
    a2_gscale: torch.Tensor  # reciprocal activation global scale for FC2 input
    w2_fp4: torch.Tensor
    w2_blockscale: torch.Tensor
    w2_alphas: torch.Tensor
    source_format: str = "modelopt"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_format",
            _normalize_fp4_source_format(self.source_format),
        )


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


@dataclass(frozen=True)
class _TensorAllocSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    init: str = "empty"


@dataclass(frozen=True, kw_only=True)
class _TPCoreWorkspacePlan:
    implementation: str
    quant_mode: str
    activation: str
    state_E: int
    weight_E: int
    routed_rows: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    dtype: torch.dtype
    dynamic_physical_tiles: int | None = None
    dynamic_task_capacity: int | None = None
    tensor_specs: Tuple[_TensorAllocSpec, ...] = ()


@dataclass
class _TPCoreArena:
    plan: _TPCoreWorkspacePlan
    shared_arena: torch.Tensor
    tensors: Dict[str, torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class TPMoEArenaLayout:
    route_workspace_nbytes: int
    core_workspace_nbytes: int
    total_nbytes: int
    core_token_counts: tuple[int, ...] = ()


@dataclass(frozen=True, kw_only=True)
class TPMoEPlan:
    """Logical launch plan shared by the static and dynamic backends."""

    implementation: str
    quant_mode: str
    activation: str
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


@dataclass
class _WeightViews:
    """Cached weight views for the concatenated expert-weight layout."""

    w13: torch.Tensor  # [2*n, k//2, E] uint8 (permuted view, no copy)
    down: torch.Tensor  # [k, n//2, E] uint8 (permuted view, no copy)
    w13_sf: torch.Tensor  # 6D MMA view for concatenated w13 scale factors
    down_sf: torch.Tensor  # [E, down_sf_rows, sf_cols] uint8 (view)
    w1_alpha: torch.Tensor  # [E] float32 contiguous tensor in plain CUDA storage
    w2_alpha: torch.Tensor  # [E] float32 contiguous tensor in plain CUDA storage
    w1_storage: torch.Tensor  # original [E, w1_n, k//2] tensor for direct micro
    w1_scale_storage: torch.Tensor
    w2_storage: torch.Tensor  # original [E, k, n//2] tensor for direct micro
    w2_scale_storage: torch.Tensor
    # Pre-computed fp4 views and CuTe pointers
    w13_fp4: object = None
    down_fp4: object = None
    sfb_w13_ptr: object = None
    sfb_down_ptr: object = None


@dataclass(frozen=True)
class _ExactRelu2Bs1NemotronLauncher:
    plan: TPMoEPlan
    weights: _WeightViews
    input_gs: torch.Tensor
    down_input_scale: torch.Tensor
    compiled: object
    mac: int


@dataclass(frozen=True)
class _ActivationKernelSpec:
    activation: str
    is_gated: bool
    micro_kernel_cls: type
    static_kernel_cls: type
    dynamic_kernel_cls: type

    def w1_rows(self, n: int) -> int:
        return (2 if self.is_gated else 1) * n

    def make_micro_kernel(self, **kernel_kwargs):
        return self.micro_kernel_cls(**kernel_kwargs)

    def make_static_kernel(self, *, num_topk: int, **kernel_kwargs):
        if self.is_gated:
            kernel_kwargs["exact_mma_m_tiles"] = num_topk == 1
        return self.static_kernel_cls(**kernel_kwargs)

    def make_dynamic_kernel(self, **kernel_kwargs):
        return self.dynamic_kernel_cls(**kernel_kwargs)


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


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("", "0", "false", "False")


def default_moe_quant_mode() -> str:
    return "w4a16" if _env_flag(_MOE_FORCE_A16_ENV, default=False) else "nvfp4"


def _normalize_quant_mode(quant_mode: str | None) -> str:
    if quant_mode is None:
        return default_moe_quant_mode()
    normalized = quant_mode.lower()
    if normalized not in {"nvfp4", "w4a16"}:
        raise ValueError(f"unsupported quant_mode {quant_mode!r}")
    return normalized


def _normalize_fp4_source_format(source_format: str) -> str:
    try:
        return _FP4_SOURCE_FORMATS[source_format.lower()]
    except KeyError as exc:
        raise ValueError(
            "source_format must be one of 'modelopt' or 'compressed_tensors', "
            f"got {source_format!r}"
        ) from exc


def _validate_fp4_source_format_for_quant_mode(
    *, source_format: str, quant_mode: str
) -> None:
    if source_format == "compressed_tensors" and quant_mode != "w4a16":
        raise ValueError(
            "source_format='compressed_tensors' is only supported with "
            "quant_mode='w4a16'; the NVFP4 kernels currently support only "
            "source_format='modelopt'"
        )


def _assert_reciprocal_input_scale_contract(
    input_scales_are_reciprocal: bool | None,
) -> None:
    assert input_scales_are_reciprocal is None or input_scales_are_reciprocal is True, (
        "input_scales_are_reciprocal is deprecated; b12x always expects reciprocal input scales"
    )


def _get_activation_kernel_spec(
    activation: str,
    *,
    quant_mode: str = "nvfp4",
) -> _ActivationKernelSpec:
    if _normalize_quant_mode(quant_mode) == "w4a16":
        raise ValueError("W4A16 dispatch uses b12x.moe.fused.w4a16.kernel directly")
    try:
        return _ACTIVATION_KERNEL_SPECS[activation]
    except KeyError as exc:
        raise ValueError(f"unsupported activation {activation!r}") from exc


def _activation_w1_rows(activation: str, n: int) -> int:
    if activation == "silu":
        return 2 * n
    if activation == "relu2":
        return n
    raise ValueError(f"unsupported activation {activation!r}")


def _dynamic_tile_n(quant_mode: str = "nvfp4") -> int:
    _normalize_quant_mode(quant_mode)
    return _LEVEL_TILE_N


def _dynamic_tile_m(quant_mode: str = "nvfp4") -> int:
    _normalize_quant_mode(quant_mode)
    return _LEVEL_TILE_M


_WEIGHT_CACHE: Dict[Tuple[int, int, int], _WeightViews] = {}
_W4A16_PACKED_WEIGHT_CACHE: Dict[Tuple[object, ...], object] = {}
_MICRO_KERNEL_CACHE: Dict[Tuple, Tuple] = {}
_STATIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}
_DYNAMIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}
_MAC_CACHE: Dict[Tuple[int, str], int] = {}  # (device_idx, impl) → max_active_clusters
_PLAIN_PARAM_CACHE: Dict[
    Tuple[int, Tuple[int, ...], Tuple[int, ...], torch.dtype, torch.dtype, int],
    torch.Tensor,
] = {}
_W4A16_ALPHA_CACHE: Dict[Tuple, torch.Tensor] = {}
_MICRO_COMPACT_CUTOVER_PAIRS_DEFAULT = 20
_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK_DEFAULT = 80
_STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT = 640
_MICRO_COMPACT_CUTOVER_PAIRS_CACHE: int | None = None
_STATIC_COMPACT_CUTOVER_PAIRS_CACHE: Dict[str, int] = {}
_DYNAMIC_MULTICTA_CACHE: bool | None = None
_DYNAMIC_CHUNK_MULTIPLIER_CACHE: int | None = None
_DYNAMIC_DOWN_SCALE_CACHE: bool | None = None
_LAST_WEIGHTS: Tuple = (None, None)  # (cache_key, views)
_LAST_KERNEL: Tuple = (None, None)  # (cache_key, (compiled, mac))
_MICRO_DIRECT_LAUNCH_CAP_CACHE: Dict[Tuple[int, int], bool] = {}
_EXACT_RELU2_BS1_NEMOTRON_CACHE: Dict[Tuple, _ExactRelu2Bs1NemotronLauncher] = {}
_LAST_EXACT_RELU2_BS1_NEMOTRON: Tuple = (None, None)  # (cache_key, launcher)
_CURRENT_DISPATCH_STAGE: str | None = None
_DIRECT_MICRO_SHAPE_ATTR = "_b12x_direct_micro_shape"


def _tensor_version(t: torch.Tensor) -> int:
    try:
        return int(t._version)
    except RuntimeError:
        # Inference tensors intentionally do not track a version counter.
        # Model weights/scales are expected to be immutable during serving.
        return 0


@contextmanager
def b12x_moe_dispatch_context(stage: str | None):
    global _CURRENT_DISPATCH_STAGE
    previous_stage = _CURRENT_DISPATCH_STAGE
    _CURRENT_DISPATCH_STAGE = stage
    try:
        yield
    finally:
        _CURRENT_DISPATCH_STAGE = previous_stage


def clear_tp_moe_caches() -> None:
    """Clear runtime caches owned by `tp_moe`.

    Explicit workspaces and workspace pools are caller-owned and intentionally
    unaffected by this helper.
    """
    from b12x.moe.fused.w4a16.kernel import clear_w4a16_kernel_cache

    global _LAST_WEIGHTS
    global _LAST_KERNEL
    global _LAST_EXACT_RELU2_BS1_NEMOTRON
    global _MICRO_COMPACT_CUTOVER_PAIRS_CACHE
    global _STATIC_COMPACT_CUTOVER_PAIRS_CACHE
    global _DYNAMIC_MULTICTA_CACHE
    global _DYNAMIC_CHUNK_MULTIPLIER_CACHE
    global _DYNAMIC_DOWN_SCALE_CACHE
    _WEIGHT_CACHE.clear()
    _W4A16_PACKED_WEIGHT_CACHE.clear()
    clear_w4a16_kernel_cache()
    _MICRO_KERNEL_CACHE.clear()
    _STATIC_KERNEL_CACHE.clear()
    _DYNAMIC_KERNEL_CACHE.clear()
    _MAC_CACHE.clear()
    _MICRO_DIRECT_LAUNCH_CAP_CACHE.clear()
    _EXACT_RELU2_BS1_NEMOTRON_CACHE.clear()
    _PLAIN_PARAM_CACHE.clear()
    _W4A16_ALPHA_CACHE.clear()
    _MICRO_COMPACT_CUTOVER_PAIRS_CACHE = None
    _STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
    _DYNAMIC_MULTICTA_CACHE = None
    _DYNAMIC_CHUNK_MULTIPLIER_CACHE = None
    _DYNAMIC_DOWN_SCALE_CACHE = None
    _LAST_WEIGHTS = (None, None)
    _LAST_KERNEL = (None, None)
    _LAST_EXACT_RELU2_BS1_NEMOTRON = (None, None)


_FAST_MATH_DEFAULT = _env_flag("B12X_FAST_MATH", default=True)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def _get_static_compact_cutover_pairs(quant_mode: str = "nvfp4") -> int:
    quant_mode = _normalize_quant_mode(quant_mode)
    cached = _STATIC_COMPACT_CUTOVER_PAIRS_CACHE.get(quant_mode)
    if cached is None:
        cutover = _first_env(
            "B12X_STATIC_COMPACT_CUTOVER_PAIRS",
            "B12X_DYNAMIC_STATIC_CUTOVER_PAIRS",
            "B12X_LEVEL10_STATIC_CUTOVER_PAIRS",
        )
        if cutover is None:
            cached = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
        else:
            cached = max(0, int(cutover))
        _STATIC_COMPACT_CUTOVER_PAIRS_CACHE[quant_mode] = cached
    return cached


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


def _dynamic_down_scale_enabled() -> bool:
    global _DYNAMIC_DOWN_SCALE_CACHE
    if _DYNAMIC_DOWN_SCALE_CACHE is None:
        _DYNAMIC_DOWN_SCALE_CACHE = _env_flag(
            "B12X_ENABLE_DYNAMIC_DOWN_SCALE", default=False
        )
    return _DYNAMIC_DOWN_SCALE_CACHE


def _get_dynamic_chunk_multiplier() -> int:
    global _DYNAMIC_CHUNK_MULTIPLIER_CACHE
    if _DYNAMIC_CHUNK_MULTIPLIER_CACHE is None:
        mult_env = os.environ.get("B12X_DYNAMIC_CHUNK_MULTIPLIER", "1")
        _DYNAMIC_CHUNK_MULTIPLIER_CACHE = max(1, int(mult_env))
    return _DYNAMIC_CHUNK_MULTIPLIER_CACHE


def _get_relu2_bs1_spark_micro_cap() -> int:
    cap = _first_env("B12X_RELU2_BS1_SPARK_MICRO_CAP")
    if cap is None:
        return 42
    return max(1, int(cap))


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
                return _get_plain_cuda_tensor(
                    scale.expand(weight_E), dtype=torch.float32
                )
        if scale.numel() != weight_E:
            raise ValueError(
                f"expected expert scale with {weight_E} elements, got {scale.numel()}"
            )
        return _get_plain_cuda_tensor(scale, dtype=torch.float32)


def _get_plain_cuda_tensor(
    t: torch.Tensor, *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    with record_function("tp_moe.get_plain_cuda_tensor"):
        target_dtype = t.dtype if dtype is None else dtype
        key = (
            t.data_ptr(),
            tuple(t.shape),
            tuple(t.stride()),
            t.dtype,
            target_dtype,
            _tensor_version(t),
        )
        cached = _PLAIN_PARAM_CACHE.get(key)
        if cached is not None:
            return cached
        plain = torch.empty(tuple(t.shape), dtype=target_dtype, device=t.device)
        with record_function("tp_moe.get_plain_cuda_tensor.copy"):
            plain.copy_(t.to(target_dtype) if t.dtype != target_dtype else t)
        _PLAIN_PARAM_CACHE[key] = plain
        return plain


def _tensor_cache_key(
    t: torch.Tensor,
) -> Tuple[int, Tuple[int, ...], Tuple[int, ...], torch.dtype, int]:
    return (
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        t.dtype,
        _tensor_version(t),
    )


def _w4a16_default_alpha(
    alpha: torch.Tensor,
    input_scale: torch.Tensor,
    weight_E: int,
) -> torch.Tensor:
    key = (
        _tensor_cache_key(alpha),
        _tensor_cache_key(input_scale),
        weight_E,
    )
    cached = _W4A16_ALPHA_CACHE.get(key)
    if cached is not None:
        return cached

    alpha_plain = _get_plain_cuda_tensor(alpha, dtype=torch.float32)
    scale_plain = _prepare_expert_scale(input_scale, weight_E)
    adjusted = torch.empty_like(alpha_plain)
    torch.mul(alpha_plain, scale_plain, out=adjusted)
    _W4A16_ALPHA_CACHE[key] = adjusted
    return adjusted


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


def _safe_dynamic_max_rows_per_launch(
    E: int,
    k: int,
    _n: int,
    quant_mode: str = "nvfp4",
) -> int:
    """Largest graph-safe routed-row budget for the compact dynamic workspace.

    Dynamic now stores routed activations in a compact physical-tile pool, so
    the dominant CuTe memref extents scale with `rows_padded` rather than
    `E * max_rows`. Graph-safe chunking still has to budget for the worst-case
    active-expert envelope, so it reserves `E - 1` extra 128-row tiles in that
    large-row regime.
    """
    tile_m = _dynamic_tile_m(quant_mode)
    rows_padded_limit = _dynamic_rows_padded_limit(k, quant_mode=quant_mode)
    extra_rows = max(0, E - 1) * tile_m
    safe_rows = rows_padded_limit - extra_rows
    if safe_rows <= 0:
        return tile_m
    return max(tile_m, safe_rows - (safe_rows % tile_m))


def _dynamic_rows_padded_limit(k: int, *, quant_mode: str = "nvfp4") -> int:
    tile_m = _dynamic_tile_m(quant_mode)
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)
    input_cols = k if _normalize_quant_mode(quant_mode) == "w4a16" else k // 2
    rows_padded_limit = min(
        _RUNTIME_MEMREF_LIMIT // max(1, input_cols),
        _RUNTIME_MEMREF_LIMIT // max(1, cols_pad_k),
    )
    return rows_padded_limit - (rows_padded_limit % tile_m)


def _safe_dynamic_token_chunk(
    E: int,
    k: int,
    n: int,
    num_topk: int,
    quant_mode: str = "nvfp4",
) -> int:
    """Largest token chunk that fits the compact dynamic launch ABI."""
    tile_m = _dynamic_tile_m(quant_mode)
    safe_rows = _safe_dynamic_max_rows_per_launch(E, k, n, quant_mode)
    max_tokens = max(1, safe_rows // max(1, num_topk))
    while max_tokens > 1 and align_up(max_tokens * num_topk, tile_m) > safe_rows:
        max_tokens -= 1
    return max_tokens


def _dynamic_token_chunk_limit(
    E: int,
    k: int,
    n: int,
    num_topk: int,
    quant_mode: str = "nvfp4",
) -> int:
    """Dynamic chunk limit with a compatibility clamp for the old multiplier knob."""
    compact_limit = _safe_dynamic_token_chunk(E, k, n, num_topk, quant_mode)
    legacy_env = os.environ.get("B12X_DYNAMIC_CHUNK_MULTIPLIER")
    if legacy_env is None:
        return compact_limit
    legacy_limit = (
        _safe_token_chunk(E, k, n, num_topk) * _get_dynamic_chunk_multiplier()
    )
    return min(compact_limit, legacy_limit)


def _workspace_policy(
    workspace: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool,
) -> _TPMoEWorkspacePolicy:
    is_pool = isinstance(workspace, TPMoEWorkspacePool)
    return _TPMoEWorkspacePolicy(
        can_chunk=is_pool,
    )


def select_tp_moe_backend(
    *,
    num_tokens: int,
    num_topk: int,
    quant_mode: str = "nvfp4",
) -> str:
    """Pick the fused MoE backend from the intrinsic routed workload shape."""
    routed_rows = num_tokens * num_topk
    if routed_rows <= _get_static_compact_cutover_pairs(quant_mode):
        return "static"
    return "dynamic"


def _dynamic_task_geometry(
    E: int,
    n: int,
    routed_rows: int,
    tile_m: int = _LEVEL_TILE_M,
    tile_n: int = _LEVEL_TILE_N,
) -> tuple[int, int, int]:
    routed_rows = max(1, routed_rows)
    base_m_tiles = align_up(routed_rows, tile_m) // tile_m
    # At most one new physical tile is introduced per active expert beyond the
    # first, and the routed workload cannot touch more experts than routed rows.
    active_expert_upper_bound = min(E, routed_rows)
    max_m_tiles = max(1, base_m_tiles + active_expert_upper_bound - 1)
    gate_tile_cnt = max(1, (n + tile_n - 1) // tile_n)
    slice_groups = max(
        1, (gate_tile_cnt + _DYNAMIC_SLICE_CHUNK - 1) // _DYNAMIC_SLICE_CHUNK
    )
    max_tasks = max_m_tiles * slice_groups
    return max_m_tiles, gate_tile_cnt, max_tasks


def _refresh_dynamic_workspace_scales(
    workspace: TPDynamicWorkspace,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    *,
    input_scales_static: bool,
    force: bool = False,
) -> None:
    a1_src_ptr = a1_gscale.data_ptr()
    a2_src_ptr = a2_gscale.data_ptr()
    if (
        force
        or not input_scales_static
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


def _reset_volatile_launch_state(workspace: TPMoEWorkspace) -> None:
    if not workspace.volatile_launch_state:
        return
    # Shared execution-lane arenas overlay MoE scratch with attention/indexer
    # scratch. The resident-grid barrier scalars are launch state, so refresh
    # them after any previous phase may have overwritten them.
    workspace.barrier_count.zero_()
    workspace.barrier_epoch.zero_()


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _tensor_numel(shape: Tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _plan_core_workspace(
    implementation: str,
    quant_mode: str,
    state_E: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    routed_rows: int,
    max_rows: int,
    activation: str = "silu",
    dynamic_physical_tiles: int | None = None,
    dynamic_task_capacity: int | None = None,
) -> _TPCoreWorkspacePlan:
    quant_mode = _normalize_quant_mode(quant_mode)
    if implementation == "w4a16":
        from b12x.moe.fused.w4a16.host import (
            _W4A16_ALLOWED_ROUTED_SIZES,
            max_packed_route_slots,
            packed_gemm_scratch_elements,
        )

        routed_capacity = max(int(routed_rows), 1)
        fc1_cols = _activation_w1_rows(activation, int(n))
        route_slots_capacity = 1
        route_blocks_capacity = 1
        fc1_c_tmp_elements = 1
        fc2_c_tmp_elements = 1
        sms = max(1, int(get_num_sm(device)))
        for block_size in _W4A16_ALLOWED_ROUTED_SIZES:
            route_slots = max_packed_route_slots(
                routed_capacity,
                int(block_size),
                int(weight_E),
            )
            route_blocks = (route_slots + int(block_size) - 1) // int(block_size)
            route_slots_capacity = max(route_slots_capacity, route_slots)
            route_blocks_capacity = max(route_blocks_capacity, route_blocks)
            fc1_c_tmp_elements = max(
                fc1_c_tmp_elements,
                packed_gemm_scratch_elements(
                    size_n=fc1_cols,
                    route_slots=route_slots,
                    moe_block_size=int(block_size),
                    sms=sms,
                ),
            )
            fc2_c_tmp_elements = max(
                fc2_c_tmp_elements,
                packed_gemm_scratch_elements(
                    size_n=int(k),
                    route_slots=route_slots,
                    moe_block_size=int(block_size),
                    sms=sms,
                ),
            )
        return _TPCoreWorkspacePlan(
            implementation=implementation,
            quant_mode=quant_mode,
            activation=activation,
            state_E=state_E,
            weight_E=weight_E,
            routed_rows=routed_capacity,
            max_rows=max(max_rows, routed_capacity),
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            dtype=dtype,
            tensor_specs=(
                _TensorAllocSpec(
                    "intermediate_cache13",
                    (routed_capacity * max(fc1_cols, int(k)),),
                    dtype,
                ),
                _TensorAllocSpec(
                    "intermediate_cache2",
                    (routed_capacity, int(n)),
                    dtype,
                ),
                _TensorAllocSpec("fc1_c_tmp", (fc1_c_tmp_elements,), torch.float32),
                _TensorAllocSpec("fc2_c_tmp", (fc2_c_tmp_elements,), torch.float32),
                _TensorAllocSpec(
                    "packed_route_indices", (route_slots_capacity,), torch.int32
                ),
                _TensorAllocSpec(
                    "block_expert_ids", (route_blocks_capacity,), torch.int32
                ),
                _TensorAllocSpec("packed_route_count", (1,), torch.int32),
                _TensorAllocSpec("expert_offsets", (int(weight_E) + 1,), torch.int32),
            ),
        )

    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)

    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)
    direct_micro_tokens = max(1, routed_rows // max(1, num_topk))
    direct_micro_k_supported = (
        k > 0
        and k % _NVFP4_BLOCK_SIZE == 0
        and k % 128 == 0
        and _direct_k_segments_supported(_direct_k_segments_for_k(k))
    )
    direct_micro_token_supported = direct_micro_tokens in (1, 2, 4, 8)
    direct_micro_candidate = (
        implementation == "static"
        and n % _NVFP4_BLOCK_SIZE == 0
        and direct_micro_k_supported
        and 0 < num_topk <= 32
        and weight_E > 0
        and routed_rows == direct_micro_tokens * num_topk
        and direct_micro_token_supported
    )
    barrier_slots = max(1, routed_rows)
    if direct_micro_candidate:
        barrier_slots = max(barrier_slots, routed_rows + direct_micro_tokens * 16)
    common_specs = (
        _TensorAllocSpec("row_counts", (state_E,), torch.int32, init="zeros"),
        _TensorAllocSpec("barrier_count", (barrier_slots,), torch.int32, init="zeros"),
        _TensorAllocSpec("barrier_epoch", (barrier_slots,), torch.int32, init="zeros"),
    )
    if implementation == "static":
        static_rows_pad_k = align_up(max_rows, 128)
        packed_input_shape = (state_E, max_rows, k // 2)
        packed_input_dtype = torch.uint8
        micro_intermediate_elements = state_E * n
        if direct_micro_candidate:
            fc2_n_chunks = (n // 2 + 127) // 128
            micro_intermediate_elements = max(
                micro_intermediate_elements,
                direct_micro_tokens * num_topk * k
                + direct_micro_tokens * num_topk * fc2_n_chunks * 128,
            )
        return _TPCoreWorkspacePlan(
            implementation=implementation,
            quant_mode=quant_mode,
            activation=activation_spec.activation,
            state_E=state_E,
            weight_E=weight_E,
            routed_rows=routed_rows,
            max_rows=max_rows,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            dtype=dtype,
            tensor_specs=common_specs
            + (
                _TensorAllocSpec(
                    "token_map", (state_E, max_rows), torch.int32, init="zeros"
                ),
                _TensorAllocSpec(
                    "token_weights", (state_E, max_rows), torch.float32, init="zeros"
                ),
                _TensorAllocSpec(
                    "packed_input", packed_input_shape, packed_input_dtype
                ),
                _TensorAllocSpec(
                    "packed_input_scale",
                    (state_E, static_rows_pad_k, cols_pad_k),
                    torch.uint8,
                ),
                _TensorAllocSpec(
                    "active_expert_count", (1,), torch.int32, init="zeros"
                ),
                _TensorAllocSpec(
                    "weight_expert_ids", (state_E,), torch.int32, init="arange"
                ),
                _TensorAllocSpec("global_to_local_expert", (weight_E,), torch.int32),
                _TensorAllocSpec("compact_topk_ids", (state_E,), torch.int32),
                _TensorAllocSpec(
                    "micro_intermediate",
                    (micro_intermediate_elements,),
                    torch.float32,
                    init="zeros",
                ),
            ),
        )

    if dynamic_physical_tiles is None or dynamic_task_capacity is None:
        dynamic_tile_m = _dynamic_tile_m(quant_mode)
        dynamic_tiles, _, dynamic_max_tasks = _dynamic_task_geometry(
            state_E,
            n,
            routed_rows,
            tile_m=dynamic_tile_m,
            tile_n=_dynamic_tile_n(quant_mode),
        )
    else:
        dynamic_tiles = dynamic_physical_tiles
        dynamic_max_tasks = dynamic_task_capacity
        dynamic_tile_m = _dynamic_tile_m(quant_mode)
    dynamic_rows_padded = dynamic_tiles * dynamic_tile_m
    packed_input_shape = (1, dynamic_rows_padded, k // 2)
    packed_input_dtype = torch.uint8
    return _TPCoreWorkspacePlan(
        implementation=implementation,
        quant_mode=quant_mode,
        activation=activation_spec.activation,
        state_E=state_E,
        weight_E=weight_E,
        routed_rows=routed_rows,
        max_rows=max_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        dtype=dtype,
        dynamic_physical_tiles=dynamic_tiles,
        dynamic_task_capacity=dynamic_max_tasks,
        tensor_specs=common_specs
        + (
            _TensorAllocSpec(
                "token_map", (dynamic_rows_padded,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "token_weights", (dynamic_rows_padded,), torch.float32, init="zeros"
            ),
            _TensorAllocSpec("packed_input", packed_input_shape, packed_input_dtype),
            _TensorAllocSpec(
                "packed_input_scale", (dynamic_rows_padded, cols_pad_k), torch.uint8
            ),
            _TensorAllocSpec(
                "expert_write_rows", (state_E,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "expert_tile_base", (state_E + 1,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec("input_gs", (weight_E,), torch.float32),
            _TensorAllocSpec("down_input_scale", (weight_E,), torch.float32),
            _TensorAllocSpec("pair_head", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec("producers_done_count", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec("all_work_published", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec("task_head", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec("task_tail", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec(
                "task_ready", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_expert", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_m_tile", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_slice_begin", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_slice_count", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_valid_rows", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "tile_write_count", (dynamic_tiles,), torch.int32, init="zeros"
            ),
        ),
    )


def _allocate_arena_tensor(
    shared_arena: torch.Tensor,
    offset: int,
    spec: _TensorAllocSpec,
) -> tuple[torch.Tensor, int]:
    alignment = max(16, _dtype_nbytes(spec.dtype))
    offset = align_up(offset, alignment)
    nbytes = _tensor_numel(spec.shape) * _dtype_nbytes(spec.dtype)
    storage = shared_arena.narrow(0, offset, nbytes)
    if spec.dtype == torch.uint8:
        tensor = storage.view(spec.shape)
    else:
        tensor = storage.view(spec.dtype).view(spec.shape)
    if spec.init == "zeros":
        tensor.zero_()
    elif spec.init == "arange":
        tensor.copy_(
            torch.arange(tensor.numel(), dtype=tensor.dtype, device=tensor.device).view(
                spec.shape
            )
        )
    elif spec.init != "empty":
        raise ValueError(f"unsupported tensor init mode {spec.init!r}")
    return tensor, offset + nbytes


def _core_workspace_nbytes(plan: _TPCoreWorkspacePlan) -> int:
    arena_nbytes = 0
    for spec in plan.tensor_specs:
        arena_nbytes = align_up(arena_nbytes, max(16, _dtype_nbytes(spec.dtype)))
        arena_nbytes += _tensor_numel(spec.shape) * _dtype_nbytes(spec.dtype)
    return int(arena_nbytes)


def _emit_core_workspace_stats(
    plan: _TPCoreWorkspacePlan,
    *,
    storage: str,
    required_nbytes: int,
    capacity_nbytes: int | None = None,
) -> None:
    return


def _materialize_core_arena(
    plan: _TPCoreWorkspacePlan,
    shared_arena: torch.Tensor,
    *,
    offset_bytes: int = 0,
    capacity_nbytes: int | None = None,
) -> _TPCoreArena:
    arena_nbytes = _core_workspace_nbytes(plan)
    offset_bytes = int(offset_bytes)
    if capacity_nbytes is None:
        capacity_nbytes = shared_arena.numel() - offset_bytes
    if capacity_nbytes < arena_nbytes:
        raise ValueError(
            f"MoE core arena requires {arena_nbytes} bytes, but only {capacity_nbytes} are available"
        )
    relative_offset = 0
    tensors: Dict[str, torch.Tensor] = {}
    for spec in plan.tensor_specs:
        tensor, absolute_next = _allocate_arena_tensor(
            shared_arena,
            offset_bytes + relative_offset,
            spec,
        )
        tensors[spec.name] = tensor
        relative_offset = absolute_next - offset_bytes
    return _TPCoreArena(plan=plan, shared_arena=shared_arena, tensors=tensors)


def _allocate_core_arena(plan: _TPCoreWorkspacePlan) -> _TPCoreArena:
    arena_nbytes = _core_workspace_nbytes(plan)
    shared_arena = torch.empty(
        arena_nbytes,
        dtype=torch.uint8,
        device=plan.device,
    )
    arena = _materialize_core_arena(plan, shared_arena)
    _emit_core_workspace_stats(
        plan,
        storage="standalone",
        required_nbytes=arena_nbytes,
    )
    return arena


def _materialize_workspace_from_core_arena(
    plan: _TPCoreWorkspacePlan,
    arena: _TPCoreArena,
    *,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    input_scales_static: bool,
    volatile_launch_state: bool = False,
) -> TPMoEWorkspace | TPW4A16Workspace:
    tensors = arena.tensors
    if plan.implementation == "w4a16":
        return TPW4A16Workspace(
            implementation=plan.implementation,
            quant_mode=plan.quant_mode,
            activation=plan.activation,
            state_E=plan.state_E,
            weight_E=plan.weight_E,
            max_rows=plan.max_rows,
            k=plan.k,
            n=plan.n,
            num_topk=plan.num_topk,
            device=plan.device,
            dtype=plan.dtype,
            routed_rows_capacity=plan.routed_rows,
            intermediate_cache13=tensors["intermediate_cache13"],
            intermediate_cache2=tensors["intermediate_cache2"],
            fc1_c_tmp=tensors["fc1_c_tmp"],
            fc2_c_tmp=tensors["fc2_c_tmp"],
            packed_route_indices=tensors["packed_route_indices"],
            block_expert_ids=tensors["block_expert_ids"],
            packed_route_count=tensors["packed_route_count"],
            expert_offsets=tensors["expert_offsets"],
            volatile_launch_state=bool(volatile_launch_state),
        )

    common_kwargs = dict(
        implementation=plan.implementation,
        quant_mode=plan.quant_mode,
        state_E=plan.state_E,
        weight_E=plan.weight_E,
        max_rows=plan.max_rows,
        k=plan.k,
        n=plan.n,
        num_topk=plan.num_topk,
        device=plan.device,
        dtype=plan.dtype,
        row_counts=tensors["row_counts"],
        barrier_count=tensors["barrier_count"],
        barrier_epoch=tensors["barrier_epoch"],
        volatile_launch_state=bool(volatile_launch_state),
    )
    if plan.implementation == "static":
        workspace = TPCompactStaticWorkspace(
            **common_kwargs,
            routed_rows_capacity=plan.routed_rows,
            token_map=tensors["token_map"],
            token_weights=tensors["token_weights"],
            packed_input=tensors["packed_input"],
            packed_input_scale=tensors["packed_input_scale"],
            active_expert_count=tensors["active_expert_count"],
            weight_expert_ids=tensors["weight_expert_ids"],
            global_to_local_expert=tensors["global_to_local_expert"],
            compact_topk_ids=tensors["compact_topk_ids"],
            micro_intermediate=tensors["micro_intermediate"],
        )
        _finalize_workspace_views(workspace)
        return workspace

    assert plan.dynamic_physical_tiles is not None
    assert plan.dynamic_task_capacity is not None
    workspace = TPDynamicWorkspace(
        **common_kwargs,
        routed_rows_capacity=plan.routed_rows,
        physical_tiles_capacity=plan.dynamic_physical_tiles,
        task_capacity=plan.dynamic_task_capacity,
        token_map=tensors["token_map"],
        token_weights=tensors["token_weights"],
        packed_input=tensors["packed_input"],
        packed_input_scale=tensors["packed_input_scale"],
        expert_write_rows=tensors["expert_write_rows"],
        expert_tile_base=tensors["expert_tile_base"],
        input_gs=tensors["input_gs"],
        down_input_scale=tensors["down_input_scale"],
        pair_head=tensors["pair_head"],
        producers_done_count=tensors["producers_done_count"],
        all_work_published=tensors["all_work_published"],
        task_head=tensors["task_head"],
        task_tail=tensors["task_tail"],
        task_ready=tensors["task_ready"],
        task_expert=tensors["task_expert"],
        task_m_tile=tensors["task_m_tile"],
        task_slice_begin=tensors["task_slice_begin"],
        task_slice_count=tensors["task_slice_count"],
        task_valid_rows=tensors["task_valid_rows"],
        tile_write_count=tensors["tile_write_count"],
    )
    _refresh_dynamic_workspace_scales(
        workspace,
        a1_gscale,
        a2_gscale,
        input_scales_static=input_scales_static,
        force=volatile_launch_state,
    )
    _finalize_workspace_views(workspace)
    return workspace


def _alloc_workspace(
    implementation: str,
    quant_mode: str,
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
    activation: str = "silu",
    dynamic_physical_tiles: int | None = None,
    dynamic_task_capacity: int | None = None,
    pool: TPMoEWorkspacePool | None = None,
    storage_key: tuple | None = None,
) -> TPMoEWorkspace | TPW4A16Workspace:
    plan = _plan_core_workspace(
        implementation,
        quant_mode,
        state_E,
        weight_E,
        k,
        n,
        num_topk,
        device,
        dtype,
        routed_rows=routed_rows,
        max_rows=max_rows,
        activation=activation,
        dynamic_physical_tiles=dynamic_physical_tiles,
        dynamic_task_capacity=dynamic_task_capacity,
    )
    if pool is not None:
        if storage_key is None:
            raise ValueError(
                "storage_key is required when allocating from a workspace pool"
            )
        arena = pool.core_arenas.get(storage_key)
        if arena is None or arena.plan != plan:
            if pool.shared_arena is None:
                arena = _allocate_core_arena(plan)
            else:
                if pool.shared_arena.device != plan.device:
                    raise ValueError(
                        f"MoE pool arena device {pool.shared_arena.device} does not match plan device {plan.device}"
                    )
                arena = _materialize_core_arena(
                    plan,
                    pool.shared_arena,
                    offset_bytes=pool.core_arena_offset_bytes,
                    capacity_nbytes=pool.core_arena_nbytes,
                )
                _emit_core_workspace_stats(
                    plan,
                    storage="shared",
                    required_nbytes=_core_workspace_nbytes(plan),
                    capacity_nbytes=pool.core_arena_nbytes,
                )
            pool.core_arenas[storage_key] = arena
    else:
        arena = _allocate_core_arena(plan)
    return _materialize_workspace_from_core_arena(
        plan,
        arena,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        input_scales_static=input_scales_static,
        volatile_launch_state=bool(pool is not None and pool.shared_arena is not None),
    )


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
    activation_spec: _ActivationKernelSpec,
) -> _WeightViews:
    """Create weight views from the expert-weight layout.

    For gated SwiGLU kernels, ``w1_fp4`` is `[E, 2*n, k//2]`.
    For relu2 kernels, ``w1_fp4`` is `[E, n, k//2]`.
    """
    global _LAST_WEIGHTS
    key = (
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_alphas.data_ptr(),
        activation_spec.activation,
    )
    last_wkey, last_wval = _LAST_WEIGHTS
    if last_wkey == key:
        return last_wval
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        _LAST_WEIGHTS = (key, cached)
        return cached

    # Permute [E, w1_n, k//2] → [w1_n, k//2, E] (view, no copy!)
    w13 = w1_fp4.permute(1, 2, 0)  # [w1_n, k//2, E]
    down = w2_fp4.permute(1, 2, 0)  # [k, n//2, E]

    # Compact contiguous scale storage for the FC1 weights.
    w1_n = activation_spec.w1_rows(n)
    bs_u8 = w1_blockscale.view(torch.uint8)
    w13_sf = as_grouped_scale_view(bs_u8, w1_n, k)
    down_sf = as_grouped_scale_view(w2_blockscale.view(torch.uint8), k, n)

    sf_dtype = cutlass.Float8E4M3FN
    views = _WeightViews(
        w13=w13,
        down=down,
        w13_sf=w13_sf,
        down_sf=down_sf,
        w1_alpha=_get_plain_cuda_tensor(w1_alphas),
        w2_alpha=_get_plain_cuda_tensor(w2_alphas),
        w1_storage=w1_fp4,
        w1_scale_storage=w1_blockscale,
        w2_storage=w2_fp4,
        w2_scale_storage=w2_blockscale,
    )
    # Keep as uint8 for dlpack compatibility — torch float4 types are not
    # supported by dlpack, and sglang may load weights as native float4.
    # The CUTLASS kernel receives the element type via _gptr / compile-time
    # dtype, not from the torch tensor dtype.
    views.w13_fp4 = w13.view(torch.uint8)
    views.down_fp4 = down.view(torch.uint8)
    views.sfb_w13_ptr = make_ptr(
        sf_dtype, w13_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    views.sfb_down_ptr = make_ptr(
        sf_dtype, down_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    _WEIGHT_CACHE[key] = views
    _LAST_WEIGHTS = (key, views)
    return views


def _get_w4a16_packed_weights(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype,
    source_format: str = "modelopt",
):
    from b12x.moe.fused.w4a16.prepare import prepare_w4a16_packed_weights

    source_format = _normalize_fp4_source_format(source_format)
    key = (
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w2_alphas.data_ptr(),
        activation,
        params_dtype,
        source_format,
    )
    cached = _W4A16_PACKED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    prepared = prepare_w4a16_packed_weights(
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        activation=activation,
        params_dtype=params_dtype,
        source_format=source_format,
    )
    _W4A16_PACKED_WEIGHT_CACHE[key] = prepared
    return prepared


def _resolve_workspace_layout(
    *,
    num_tokens: int,
    weight_E: int,
    num_topk: int,
    quant_mode: str = "nvfp4",
) -> tuple[str, int, int]:
    routed_rows = num_tokens * num_topk
    if _normalize_quant_mode(quant_mode) == "w4a16":
        return "w4a16", weight_E, max(1, routed_rows)
    implementation = select_tp_moe_backend(
        num_tokens=num_tokens,
        num_topk=num_topk,
        quant_mode=quant_mode,
    )
    if implementation == "static":
        return implementation, max(1, routed_rows), max(1, routed_rows)
    return implementation, weight_E, align_up(routed_rows, _dynamic_tile_m(quant_mode))


def _make_workspace_plan(
    *,
    num_tokens: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
    quant_mode: str = "nvfp4",
    activation: str = "silu",
) -> TPMoEPlan:
    quant_mode = _normalize_quant_mode(quant_mode)
    if quant_mode == "w4a16":
        _activation_w1_rows(activation, 1)
    else:
        activation = _get_activation_kernel_spec(
            activation, quant_mode=quant_mode
        ).activation
    routed_rows = num_tokens * num_topk
    implementation, state_E, max_rows = _resolve_workspace_layout(
        num_tokens=num_tokens,
        weight_E=weight_E,
        num_topk=num_topk,
        quant_mode=quant_mode,
    )
    dynamic_physical_tiles = None
    dynamic_task_capacity = None
    max_tokens_per_launch = num_tokens
    if implementation == "dynamic":
        dynamic_tile_m = _dynamic_tile_m(quant_mode)
        dynamic_tile_n = _dynamic_tile_n(quant_mode)
        dynamic_physical_tiles, _, dynamic_task_capacity = _dynamic_task_geometry(
            state_E,
            n,
            routed_rows,
            tile_m=dynamic_tile_m,
            tile_n=dynamic_tile_n,
        )
        max_tokens_per_launch = _dynamic_token_chunk_limit(
            weight_E,
            k,
            n,
            num_topk,
            quant_mode,
        )
    return TPMoEPlan(
        implementation=implementation,
        quant_mode=quant_mode,
        activation=activation,
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


def _make_exact_relu2_bs1_nemotron_plan(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_tokens: int = 1,
) -> TPMoEPlan:
    num_topk = 22
    total_pairs = num_topk * num_tokens
    return TPMoEPlan(
        implementation="static",
        quant_mode="nvfp4",
        activation="relu2",
        state_E=total_pairs,
        weight_E=512,
        routed_rows=total_pairs,
        max_rows=total_pairs,
        k=1024,
        n=2688,
        num_topk=num_topk,
        device=device,
        dtype=dtype,
        max_tokens_per_launch=num_tokens,
    )


def _validate_workspace(
    workspace: object,
    *,
    plan: TPMoEPlan,
) -> None:
    def _canonical_device(device: torch.device) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    expected = (
        plan.implementation,
        plan.quant_mode,
        plan.weight_E,
        plan.k,
        plan.n,
        plan.num_topk,
        _canonical_device(plan.device),
        plan.dtype,
    )
    actual = (
        workspace.implementation,
        workspace.quant_mode,
        workspace.weight_E,
        workspace.k,
        workspace.n,
        workspace.num_topk,
        _canonical_device(workspace.device),
        workspace.dtype,
    )
    if actual != expected:
        raise ValueError(
            "workspace metadata mismatch: "
            f"expected {(plan.implementation, plan.quant_mode, plan.weight_E, plan.k, plan.n, plan.num_topk, plan.device, plan.dtype)}, "
            f"got {actual}"
        )
    if plan.implementation == "w4a16":
        if not isinstance(workspace, TPW4A16Workspace):
            raise TypeError("expected a TPW4A16Workspace for the W4A16 backend")
        if workspace.activation != plan.activation:
            raise ValueError("workspace activation mismatch")
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
        if workspace.routed_rows_capacity < plan.routed_rows:
            raise ValueError(
                "workspace routed-row capacity mismatch: "
                f"expected at least {plan.routed_rows}, got {workspace.routed_rows_capacity}"
            )
        return
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
    if plan.implementation == "static" and not isinstance(
        workspace, TPCompactStaticWorkspace
    ):
        raise TypeError(
            "expected a TPCompactStaticWorkspace for the compact static backend"
        )
    if plan.implementation == "dynamic" and not isinstance(
        workspace, TPDynamicWorkspace
    ):
        raise TypeError("expected a TPDynamicWorkspace for the dynamic backend")
    if (
        isinstance(workspace, TPCompactStaticWorkspace)
        and workspace.routed_rows_capacity < plan.routed_rows
    ):
        raise ValueError(
            "workspace routed-row capacity mismatch: "
            f"expected at least {plan.routed_rows}, got {workspace.routed_rows_capacity}"
        )
    if (
        isinstance(workspace, TPDynamicWorkspace)
        and workspace.routed_rows_capacity < plan.routed_rows
    ):
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
    quant_mode: str,
    activation: str,
    state_E: int,
    weight_E: int,
    max_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    # Pool-backed workspaces are capacity-based. Avoid
    # exact-shape keys here or long-tail prompt lengths will accumulate one
    # retained workspace per distinct routed-row count.
    if implementation in ("static", "dynamic", "w4a16"):
        state_E = -1
        max_rows = -1
    return (
        implementation,
        quant_mode,
        activation if implementation == "w4a16" else "",
        state_E,
        weight_E,
        max_rows,
        k,
        n,
        num_topk,
        device.index or 0,
        dtype,
    )


def _lookup_capture_static_workspace(
    workspace: TPMoEWorkspacePool,
    *,
    plan: TPMoEPlan,
) -> TPCompactStaticWorkspace | None:
    if plan.implementation != "static":
        return None
    for candidate in workspace.workspaces.values():
        if not isinstance(candidate, TPCompactStaticWorkspace):
            continue
        if (
            candidate.implementation != plan.implementation
            or candidate.quant_mode != plan.quant_mode
            or candidate.weight_E != plan.weight_E
            or candidate.k != plan.k
            or candidate.n != plan.n
            or candidate.num_topk != plan.num_topk
            or candidate.device != plan.device
            or candidate.dtype != plan.dtype
        ):
            continue
        if candidate.state_E < plan.state_E:
            continue
        if candidate.max_rows < plan.max_rows:
            continue
        if candidate.routed_rows_capacity < plan.routed_rows:
            continue
        return candidate
    return None


def _resolve_workspace(
    workspace: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool,
    *,
    plan: TPMoEPlan,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    input_scales_static: bool,
) -> object:
    if isinstance(workspace, (TPMoEWorkspace, TPW4A16Workspace)):
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
            "workspace must be a TPMoEWorkspace, TPW4A16Workspace, or TPMoEWorkspacePool"
        )

    key = _workspace_pool_key(
        plan.implementation,
        state_E=plan.state_E,
        weight_E=plan.weight_E,
        max_rows=plan.max_rows,
        k=plan.k,
        n=plan.n,
        num_topk=plan.num_topk,
        device=plan.device,
        dtype=plan.dtype,
        quant_mode=plan.quant_mode,
        activation=plan.activation,
    )
    resolved = workspace.workspaces.get(key)
    if resolved is None and torch.cuda.is_current_stream_capturing():
        capture_static = _lookup_capture_static_workspace(workspace, plan=plan)
        if capture_static is not None:
            # Capture may switch to a dedicated stream, but the compact static
            # workspace is stream-agnostic scratch. Reuse the warmed eager
            # workspace instead of allocating a fresh one inside capture.
            workspace.workspaces[key] = capture_static
            resolved = capture_static
    if resolved is None:
        if plan.implementation == "w4a16" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "W4A16 workspace is not initialized for CUDA graph capture; "
                "run a warmup with the workspace pool or allocate a sufficient workspace before capture"
            )
        resolved = _alloc_workspace(
            plan.implementation,
            plan.quant_mode,
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
            activation=plan.activation,
            dynamic_physical_tiles=plan.dynamic_physical_tiles,
            dynamic_task_capacity=plan.dynamic_task_capacity,
            pool=workspace,
            storage_key=key,
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
            isinstance(resolved, TPW4A16Workspace)
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
        if plan.implementation == "w4a16" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "W4A16 workspace capacity is too small for CUDA graph capture; "
                "run an eager warmup with a larger routed-row budget before capture"
            )
        dynamic_tiles = plan.dynamic_physical_tiles
        dynamic_tasks = plan.dynamic_task_capacity
        if isinstance(resolved, TPDynamicWorkspace):
            dynamic_tiles = max(dynamic_tiles or 0, resolved.physical_tiles_capacity)
            dynamic_tasks = max(dynamic_tasks or 0, resolved.task_capacity)
        resolved = _alloc_workspace(
            plan.implementation,
            plan.quant_mode,
            max(plan.state_E, resolved.state_E),
            plan.weight_E,
            plan.k,
            plan.n,
            plan.num_topk,
            plan.device,
            plan.dtype,
            a1_gscale,
            a2_gscale,
            routed_rows=max(
                plan.routed_rows, getattr(resolved, "routed_rows_capacity", 0)
            ),
            max_rows=max(plan.max_rows, resolved.max_rows),
            input_scales_static=input_scales_static,
            activation=plan.activation,
            dynamic_physical_tiles=dynamic_tiles,
            dynamic_task_capacity=dynamic_tasks,
            pool=workspace,
            storage_key=key,
        )
        workspace.workspaces[key] = resolved
        return resolved

    if isinstance(resolved, TPDynamicWorkspace):
        _refresh_dynamic_workspace_scales(
            resolved,
            a1_gscale,
            a2_gscale,
            input_scales_static=input_scales_static,
            force=resolved.volatile_launch_state,
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
    quant_mode: str | None = None,
    activation: str = "silu",
) -> TPMoEWorkspace | TPW4A16Workspace:
    """Allocate reusable scratch covering one unchunked `b12x_moe_fp4` call."""
    quant_mode = _normalize_quant_mode(quant_mode)
    if a.ndim != 2:
        raise ValueError(
            f"expected input activations with rank 2, got shape {tuple(a.shape)}"
        )
    if topk_ids.ndim != 2:
        raise ValueError(
            f"expected topk_ids with rank 2, got shape {tuple(topk_ids.shape)}"
        )
    m, k = a.shape
    if topk_ids.shape[0] != m:
        raise ValueError(
            f"topk_ids batch mismatch: expected {m}, got {topk_ids.shape[0]}"
        )
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
        quant_mode=quant_mode,
        activation=activation,
    )
    effective_input_scales_static = input_scales_static or (
        a1_gscale.numel() == 1 and a2_gscale.numel() == 1
    )
    return _alloc_workspace(
        plan.implementation,
        plan.quant_mode,
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
        activation=plan.activation,
        dynamic_physical_tiles=plan.dynamic_physical_tiles,
        dynamic_task_capacity=plan.dynamic_task_capacity,
    )


def plan_tp_moe_arena_layout(
    *,
    max_tokens: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device | str,
    dtype: torch.dtype,
    core_token_counts: tuple[int, ...] | None = None,
    route_num_experts: int | None = None,
    route_logits_dtype: torch.dtype | None = None,
    quant_mode: str | None = None,
    activation: str = "silu",
) -> TPMoEArenaLayout:
    """Compute the byte layout needed by one lane-owned MoE pool."""
    quant_mode = _normalize_quant_mode(quant_mode)
    device = torch.device(device)
    max_tokens = max(int(max_tokens), 1)
    weight_E = max(int(weight_E), 1)
    k = max(int(k), 1)
    n = max(int(n), 1)
    num_topk = max(int(num_topk), 1)
    if core_token_counts is None:
        core_token_counts = (max_tokens,)
    else:
        core_token_counts = tuple(
            max(int(token_count), 1) for token_count in core_token_counts
        )
        if max_tokens not in core_token_counts:
            core_token_counts = (max_tokens, *core_token_counts)
    static_cutover_pairs = _get_static_compact_cutover_pairs(quant_mode)
    max_static_tokens = static_cutover_pairs // num_topk
    if max_static_tokens >= 1:
        static_boundary_tokens = min(max_tokens, max_static_tokens)
        if static_boundary_tokens not in core_token_counts:
            core_token_counts = (*core_token_counts, static_boundary_tokens)
    route_num_experts = int(
        route_num_experts if route_num_experts is not None else weight_E
    )
    route_logits_dtype = route_logits_dtype or dtype
    core_nbytes = 0
    for token_count in core_token_counts:
        plan = _make_workspace_plan(
            num_tokens=token_count,
            weight_E=weight_E,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            dtype=dtype,
            quant_mode=quant_mode,
            activation=activation,
        )
        core_plan = _plan_core_workspace(
            plan.implementation,
            plan.quant_mode,
            plan.state_E,
            plan.weight_E,
            plan.k,
            plan.n,
            plan.num_topk,
            plan.device,
            plan.dtype,
            routed_rows=plan.routed_rows,
            max_rows=plan.max_rows,
            activation=plan.activation,
            dynamic_physical_tiles=plan.dynamic_physical_tiles,
            dynamic_task_capacity=plan.dynamic_task_capacity,
        )
        core_nbytes = max(core_nbytes, _core_workspace_nbytes(core_plan))
    route_nbytes = _route_workspace_nbytes(
        num_tokens=max_tokens,
        num_experts=route_num_experts,
        top_k=num_topk,
        logits_dtype=route_logits_dtype,
    )
    route_nbytes = align_up(route_nbytes, 16)
    return TPMoEArenaLayout(
        route_workspace_nbytes=route_nbytes,
        core_workspace_nbytes=core_nbytes,
        total_nbytes=max(route_nbytes + core_nbytes, 1),
        core_token_counts=core_token_counts,
    )


def allocate_tp_moe_workspace_pool(
    *,
    shared_arena: torch.Tensor | None = None,
    route_workspace_nbytes: int = 0,
    core_workspace_nbytes: int = 0,
    frozen: bool = False,
) -> TPMoEWorkspacePool:
    """Allocate an explicit caller-owned workspace pool for one execution lane."""
    pool = TPMoEWorkspacePool()
    if shared_arena is not None:
        pool.bind_shared_arena(
            shared_arena,
            route_workspace_nbytes=route_workspace_nbytes,
            core_workspace_nbytes=core_workspace_nbytes,
            frozen=frozen,
        )
    return pool


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


def _select_micro_mma_tiler_mn(
    max_rows: int,
    n: int,
    *,
    resident_clusters: int | None = None,
) -> tuple[int, int]:
    if os.environ.get("B12X_MOE_TILE_MN"):
        return tuple(int(x) for x in os.environ["B12X_MOE_TILE_MN"].split("x"))
    sm_count = get_num_sm(torch.device("cuda"))
    coarse_tile = (128, 128)
    if max_rows <= 32 and n <= 256:
        return (64, 128)
    if resident_clusters is not None and resident_clusters < sm_count:
        return coarse_tile
    coarse_tiles = ((max_rows + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    # Single-token decode often lands exactly on the "half the machine" boundary.
    # Keeping the coarse 128x128 tile there leaves the M dimension badly underfilled.
    if max_rows <= 64 or (max_rows <= 128 and coarse_tiles <= max(1, sm_count // 2)):
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
    fast_math: bool,
    mac_override: int | None = None,
    activation: str = "silu",
    single_token: bool = False,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    quant_mode: str = "nvfp4",
):
    quant_mode = _normalize_quant_mode(quant_mode)
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    sf_vec_size = 16
    mac = mac_override if mac_override is not None else _get_impl_mac("static")
    routed_rows = m * num_topk
    mma_tiler_mn = (128, 128)
    dynamic_down_scale = _dynamic_down_scale_enabled()
    if num_topk > 1:
        mma_tiler_mn = _select_micro_mma_tiler_mn(routed_rows, n, resident_clusters=mac)

    global _LAST_KERNEL
    cache_key = (
        quant_mode,
        "static",
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        fast_math,
        activation,
        single_token,
        share_input_across_experts,
        share_expert_scales,
        dynamic_down_scale,
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

    weight_dtype = cutlass.Float4E2M1FN
    a_scratch_dtype = weight_dtype
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel_kwargs = dict(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1]),
        fast_math=fast_math,
        dynamic_down_scale=dynamic_down_scale,
    )
    kernel = activation_spec.make_static_kernel(
        **kernel_kwargs,
        num_topk=num_topk,
        single_token=single_token,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
    )

    rows_pad_k = align_up(max_rows, 128)
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)

    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype,
        (m * num_topk,),
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (m * num_topk,),
        assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        a_scratch_dtype,
        (max_rows, k, state_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_elements = state_E * max_rows * (k // 2)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (packed_a_storage_elements,),
        assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * rows_pad_k * cols_pad_k,),
        assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    w1_n = activation_spec.w1_rows(n)
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (w1_n, k, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (k, n, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (weight_E,),
        assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=16,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = cute.compile(
        kernel,
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        b_w13_fake,
        sfb_w13_fake,
        b_down_fake,
        sfb_down_fake,
        row_counts_fake,
        active_expert_count_fake,
        weight_expert_ids_fake,
        global_to_local_expert_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        mac,
        current_cuda_stream(),
    )

    result = (compiled, mac)
    if reuse_compiled:
        _STATIC_KERNEL_CACHE[cache_key] = result
    _LAST_KERNEL = (cache_key, result)
    return result


def _get_micro_kernel(
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    *,
    topk_ids_dtype: torch.dtype,
    fast_math: bool,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    single_token: bool = False,
    mac_override: int | None = None,
    activation: str = "silu",
    device: torch.device | None = None,
    quant_mode: str = "nvfp4",
):
    quant_mode = _normalize_quant_mode(quant_mode)
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    mac = mac_override if mac_override is not None else _get_impl_mac("micro")
    dynamic_down_scale = _dynamic_down_scale_enabled()

    global _LAST_KERNEL
    cache_key = (
        quant_mode,
        "micro_direct",
        m,
        k,
        n,
        num_topk,
        weight_E,
        topk_ids_dtype,
        fast_math,
        share_input_across_experts,
        share_expert_scales,
        single_token,
        activation,
        dynamic_down_scale,
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

    kernel = activation_spec.make_micro_kernel(
        sf_vec_size=16,
        mma_tiler_mn=(64, 128),
        output_tile_count_n=1,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
        dynamic_down_scale=dynamic_down_scale,
    )
    kernel.configure(m, k, n, num_topk, weight_E, max_active_ctas=mac, device=device)

    def dummy(dt):
        return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)

    ids_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    barrier_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compile_kwargs = {}
    compile_options = os.environ.get("B12X_DIRECT_CUTE_OPTIONS", "")
    if compile_options:
        compile_kwargs["options"] = compile_options
    compiled = cute.compile(
        kernel,
        dummy(cutlass.BFloat16),  # x_ptr
        dummy(cutlass.Uint8),  # w1_ptr
        dummy(cutlass.Uint8),  # w1s_ptr
        dummy(cutlass.Float32),  # w1a_ptr
        dummy(cutlass.Float32),  # a1_ptr
        dummy(cutlass.Float32),  # a2_ptr
        dummy(cutlass.Uint32),  # inter_ptr
        dummy(cutlass.Uint8),  # w2_ptr
        dummy(cutlass.Uint8),  # w2s_ptr
        dummy(cutlass.Float32),  # w2a_ptr
        dummy(ids_dtype),  # tid_ptr
        dummy(cutlass.Float32),  # tw_ptr
        dummy(cutlass.BFloat16),  # out_ptr
        barrier_fake,  # barrier_count
        barrier_fake,  # barrier_epoch
        Int32(m),  # m_val
        Int32(kernel.grid_x),  # grid_x
        current_cuda_stream(),  # stream
        **compile_kwargs,
    )
    with suppress(Exception):
        setattr(
            compiled,
            _DIRECT_MICRO_SHAPE_ATTR,
            (quant_mode, int(m), int(k), int(n), int(num_topk), int(weight_E)),
        )

    result = (compiled, kernel.grid_x)
    if reuse_compiled:
        _MICRO_KERNEL_CACHE[cache_key] = result
    _LAST_KERNEL = (cache_key, result)
    return result


def _direct_micro_shape_accepts_block_dim(compiled, block_dim: int) -> bool:
    return True


def _compiled_direct_micro_accepts_block_dim(compiled, block_dim: int) -> bool:
    """Return whether the compiled direct micro kernel can launch `block_dim` threads."""
    cache_key = (
        id(compiled),
        int(block_dim),
        getattr(compiled, _DIRECT_MICRO_SHAPE_ATTR, None),
    )
    cached = _MICRO_DIRECT_LAUNCH_CAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not _direct_micro_shape_accepts_block_dim(compiled, block_dim):
        _MICRO_DIRECT_LAUNCH_CAP_CACHE[cache_key] = False
        return False

    accepted = False
    try:
        from cuda.bindings import driver, runtime

        executor = compiled.to(None)
        kernel_info = getattr(compiled, "kernel_info", None) or {}
        kernel_name = next(iter(kernel_info.keys()), None)
        if kernel_name is None and hasattr(compiled, "_get_name"):
            kernel_name = compiled._get_name()
        if isinstance(kernel_name, str):
            kernel_name = kernel_name.encode()
        if kernel_name is None:
            raise RuntimeError("compiled micro kernel did not expose a kernel name")

        jit_module = getattr(executor, "jit_module", None)
        cuda_library = getattr(jit_module, "cuda_library", None)
        if isinstance(cuda_library, (list, tuple)):
            cuda_library = cuda_library[0] if cuda_library else None
        if cuda_library is None:
            cuda_library = getattr(executor, "kernel", None)
        if cuda_library is None:
            raise RuntimeError("compiled micro kernel did not expose a CUDA library")

        err, kernel = runtime.cudaLibraryGetKernel(cuda_library, kernel_name)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaLibraryGetKernel failed with {err}")
        cu_kernel = driver.CUkernel(int(kernel))
        err, max_threads = driver.cuKernelGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            cu_kernel,
            0,
        )
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuKernelGetAttribute failed with {err}")
        accepted = int(max_threads) >= int(block_dim)
    except Exception:
        accepted = False

    _MICRO_DIRECT_LAUNCH_CAP_CACHE[cache_key] = accepted
    return accepted


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
        a_input = cute.make_tensor(
            a_ptr, layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1))
        )
        topk_ids = cute.make_tensor(
            topk_ids_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        topk_weights_t = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        scatter_output = cute.make_tensor(
            scatter_ptr,
            layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1)),
        )
        packed_a = cute.make_tensor(
            packed_a_ptr,
            layout=cute.make_layout(
                (rows_padded, self._k, 1), stride=(self._k, 1, rows_padded * self._k)
            ),
        )
        packed_a_storage = cute.make_tensor(
            packed_a_storage_ptr,
            layout=cute.make_layout((rows_padded * self._half_k,), stride=(1,)),
        )
        scale_storage = cute.make_tensor(
            scale_storage_ptr,
            layout=cute.make_layout((rows_padded * self._cols_pad_k,), stride=(1,)),
        )
        token_map = cute.make_tensor(
            token_map_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        token_weights_t = cute.make_tensor(
            token_weights_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        task_ready = cute.make_tensor(
            task_ready_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_expert = cute.make_tensor(
            task_expert_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_m_tile = cute.make_tensor(
            task_m_tile_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_begin = cute.make_tensor(
            task_slice_begin_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_count = cute.make_tensor(
            task_slice_count_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_valid_rows = cute.make_tensor(
            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        tile_write_count = cute.make_tensor(
            tile_write_count_ptr,
            layout=cute.make_layout((max_phys_tiles,), stride=(1,)),
        )
        self._kernel(
            a_input,
            topk_ids,
            topk_weights_t,
            packed_a,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            pair_head,
            producers_done_count,
            all_work_published,
            task_head,
            task_tail,
            task_ready,
            task_expert,
            task_m_tile,
            task_slice_begin,
            task_slice_count,
            task_valid_rows,
            tile_write_count,
            b_w13,
            sfb_w13_ptr,
            b_down,
            sfb_down_ptr,
            row_counts,
            expert_write_rows,
            expert_tile_base,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights_t,
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
    fast_math: bool,
    mac_override: int | None = None,
    activation: str = "silu",
    quant_mode: str = "nvfp4",
    share_input_across_experts: bool = False,
):
    quant_mode = _normalize_quant_mode(quant_mode)
    share_input_across_experts = bool(
        share_input_across_experts and quant_mode == "nvfp4"
    )
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    sf_vec_size = 16
    mac = mac_override if mac_override is not None else _get_impl_mac("dynamic")
    dynamic_down_scale = _dynamic_down_scale_enabled()
    mma_tiler_mn = (
        _dynamic_tile_m(quant_mode),
        _dynamic_tile_n(quant_mode),
    )

    global _LAST_KERNEL
    cache_key = (
        quant_mode,
        "dynamic",
        E,
        k,
        n,
        num_topk,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        fast_math,
        activation,
        dynamic_down_scale,
        share_input_across_experts,
    )
    last_kkey, last_kval = _LAST_KERNEL
    if last_kkey == cache_key:
        return last_kval
    reuse_compiled = _first_env(
        "B12X_DYNAMIC_REUSE_COMPILED", "B12X_LEVEL10_REUSE_COMPILED"
    )
    if reuse_compiled is None:
        reuse_compiled = "1"
    reuse_compiled = reuse_compiled != "0"
    if reuse_compiled:
        cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            _LAST_KERNEL = (cache_key, cached)
            return cached

    weight_dtype = cutlass.Float4E2M1FN
    a_scratch_dtype = weight_dtype
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel_kwargs = dict(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        fast_math=fast_math,
        dynamic_down_scale=dynamic_down_scale,
    )
    kernel_kwargs["share_input_across_experts"] = share_input_across_experts
    kernel = activation_spec.make_dynamic_kernel(**kernel_kwargs)
    launch = _DynamicMoELaunch(kernel, k=k, num_topk=num_topk)

    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8

    # a_input, topk_ids, topk_weights, scatter_output are pointers — shapes
    # are constructed at runtime from num_tokens Int32.
    a_input_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_ids_fake = make_ptr(
        topk_ids_cutlass_dtype,
        topk_ids_align,
        cute.AddressSpace.gmem,
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )

    packed_a_fake = make_ptr(
        a_scratch_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    scale_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    pair_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    producers_done_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    all_work_published_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    task_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    task_tail_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    task_ready_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_expert_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_m_tile_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_begin_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_valid_rows_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    tile_write_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    w1_n = activation_spec.w1_rows(n)
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (w1_n, k, E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (k, n, E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (E,),
        assumed_align=4,
    )
    expert_write_rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (E,),
        assumed_align=4,
    )
    expert_tile_base_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (E + 1,),
        assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (E,),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (E,),
        assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (E,),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (E,),
        assumed_align=16,
    )
    scatter_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(
        alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    compiled = cute.compile(
        launch,
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        pair_head_fake,
        producers_done_count_fake,
        all_work_published_fake,
        task_head_fake,
        task_tail_fake,
        task_ready_fake,
        task_expert_fake,
        task_m_tile_fake,
        task_slice_begin_fake,
        task_slice_count_fake,
        task_valid_rows_fake,
        tile_write_count_fake,
        b_w13_fake,
        sfb_w13_fake,
        b_down_fake,
        sfb_down_fake,
        row_counts_fake,
        expert_write_rows_fake,
        expert_tile_base_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        1,
        1,
        1,
        1,
        1,
        mac,
        current_cuda_stream(),
    )

    result = (compiled, mac)
    if reuse_compiled:
        _DYNAMIC_KERNEL_CACHE[cache_key] = result
    _LAST_KERNEL = (cache_key, result)
    return result


def _is_exact_relu2_bs1_nemotron_case(
    *,
    activation: str,
    a: torch.Tensor,
    w1_fp4: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    return False
    if not (
        activation == "relu2"
        and a.dtype == torch.bfloat16
        and a.dim() == 2
        and a.shape[1] == 1024
        and w1_fp4.shape == (512, 2688, 512)
        and w2_fp4.shape == (512, 1024, 1344)
        and topk_ids.dim() == 2
        and topk_ids.shape[1] == 22
        and topk_weights.shape == topk_ids.shape
        and a1_gscale.numel() == 1
        and a2_gscale.numel() == 1
    ):
        return False
    if os.environ.get("B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS", "1") == "0":
        return False
    # This exact launcher compiles the micro-kernel's single-token shared-input
    # path. Multi-token batches must use the generic static/dynamic path.
    return a.shape[0] == 1 and topk_ids.shape[0] == 1


def _get_exact_relu2_bs1_nemotron_launcher(
    *,
    a: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids_dtype: torch.dtype,
    fast_math: bool,
) -> _ExactRelu2Bs1NemotronLauncher:
    global _LAST_EXACT_RELU2_BS1_NEMOTRON
    num_tokens = int(a.shape[0])
    plan = _make_exact_relu2_bs1_nemotron_plan(
        device=a.device,
        dtype=a.dtype,
        num_tokens=num_tokens,
    )
    cache_key = (
        plan.device.index or 0,
        plan.dtype,
        topk_ids_dtype,
        fast_math,
        num_tokens,
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w2_alphas.data_ptr(),
        a1_gscale.data_ptr(),
        _tensor_version(a1_gscale),
        a2_gscale.data_ptr(),
        _tensor_version(a2_gscale),
    )
    last_key, last_launcher = _LAST_EXACT_RELU2_BS1_NEMOTRON
    if last_key == cache_key:
        return last_launcher
    cached = _EXACT_RELU2_BS1_NEMOTRON_CACHE.get(cache_key)
    if cached is not None:
        _LAST_EXACT_RELU2_BS1_NEMOTRON = (cache_key, cached)
        return cached

    weights = _get_weight_views(
        w1_fp4,
        w1_blockscale,
        w2_fp4,
        w2_blockscale,
        w1_alphas,
        w2_alphas,
        plan.n,
        plan.k,
        activation_spec=_ACTIVATION_KERNEL_SPECS["relu2"],
    )
    input_gs = _prepare_expert_scale(a1_gscale, plan.weight_E)
    down_input_scale = _prepare_expert_scale(a2_gscale, plan.weight_E)
    static_work_tiles = plan.routed_rows * max(1, (plan.n + 127) // 128)
    static_mac = min(
        _get_impl_mac("static", routed_rows=plan.routed_rows), static_work_tiles
    )
    if get_num_sm(plan.device) <= 96:
        static_mac = min(static_mac, _get_relu2_bs1_spark_micro_cap())
    compiled, mac = _get_static_kernel(
        plan.state_E,
        plan.weight_E,
        num_tokens,
        plan.k,
        plan.n,
        plan.num_topk,
        plan.max_rows,
        topk_ids_dtype=topk_ids_dtype,
        fast_math=fast_math,
        share_input_across_experts=True,
        share_expert_scales=False,
        single_token=True,
        mac_override=static_mac,
        activation="relu2",
    )
    launcher = _ExactRelu2Bs1NemotronLauncher(
        plan=plan,
        weights=weights,
        input_gs=input_gs,
        down_input_scale=down_input_scale,
        compiled=compiled,
        mac=mac,
    )
    _EXACT_RELU2_BS1_NEMOTRON_CACHE[cache_key] = launcher
    _LAST_EXACT_RELU2_BS1_NEMOTRON = (cache_key, launcher)
    return launcher


def _resolve_scatter_output(
    *,
    a: torch.Tensor,
    output: torch.Tensor | None,
    device: torch.device,
    m: int,
    k: int,
) -> torch.Tensor:
    if output is None:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("CUDA graph capture requires a caller-owned output buffer")
        scatter_output = torch.zeros(m, k, dtype=a.dtype, device=device)
    else:
        scatter_output = output
    if scatter_output.shape != (m, k):
        raise ValueError(
            f"output must have shape {(m, k)}, got {tuple(scatter_output.shape)}"
        )
    if scatter_output.dtype != a.dtype:
        raise ValueError(
            f"output must have dtype {a.dtype}, got {scatter_output.dtype}"
        )
    if scatter_output.device != device:
        raise ValueError(
            f"output must be on device {device}, got {scatter_output.device}"
        )
    if not scatter_output.is_contiguous():
        raise ValueError("output must be contiguous")
    return scatter_output


def _launch_exact_relu2_bs1_nemotron(
    *,
    workspace: TPMoEWorkspace | TPMoEWorkspacePool,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scatter_output: torch.Tensor,
    fast_math: bool,
    input_scales_static: bool,
) -> torch.Tensor:
    flat_ids = _flatten_routing_ids(topk_ids)
    flat_weights = _flatten_routing_weights(topk_weights)
    launcher = _get_exact_relu2_bs1_nemotron_launcher(
        a=a,
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=w1_alphas,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_alphas,
        topk_ids_dtype=flat_ids.dtype,
        fast_math=fast_math,
    )
    resolved = _resolve_workspace(
        workspace,
        plan=launcher.plan,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        input_scales_static=input_scales_static,
    )
    assert isinstance(resolved, TPCompactStaticWorkspace)
    _reset_volatile_launch_state(resolved)
    launcher.compiled(
        a,
        flat_ids,
        flat_weights,
        resolved.packed_a_view,
        resolved.sfa_ptr,
        resolved.packed_a_flat,
        resolved.scale_flat,
        resolved.barrier_count,
        resolved.barrier_epoch,
        launcher.weights.w13_fp4,
        launcher.weights.sfb_w13_ptr,
        launcher.weights.down_fp4,
        launcher.weights.sfb_down_ptr,
        resolved.row_counts,
        resolved.active_expert_count,
        resolved.weight_expert_ids,
        resolved.global_to_local_expert,
        launcher.input_gs,
        launcher.weights.w1_alpha,
        launcher.weights.w2_alpha,
        launcher.down_input_scale,
        scatter_output,
        resolved.token_map,
        resolved.token_weights,
        current_cuda_stream(),
    )
    return scatter_output


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
    fast_math: bool,
    stream,
    activation: str = "silu",
    quant_mode: str = "nvfp4",
    share_input_across_experts: bool = False,
) -> None:
    quant_mode = _normalize_quant_mode(quant_mode)
    effective_mac = _get_impl_mac("dynamic", routed_rows=routed_rows)
    if not _dynamic_multicta_enabled():
        effective_mac = 1
    compiled, mac = _get_dynamic_kernel(
        E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        topk_ids_dtype=topk_ids_dtype,
        fast_math=fast_math,
        mac_override=effective_mac,
        activation=activation,
        quant_mode=quant_mode,
        share_input_across_experts=share_input_across_experts,
    )
    _reset_volatile_launch_state(workspace)
    def _gptr(dtype, t, align=16):
        return make_ptr(
            dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
        )
    ids_cutlass_dtype = (
        cutlass.Int32 if flat_ids.dtype == torch.int32 else cutlass.Int64
    )
    ids_align = 4 if flat_ids.dtype == torch.int32 else 8
    compiled(
        _gptr(cutlass.BFloat16, a),
        _gptr(ids_cutlass_dtype, flat_ids, ids_align),
        _gptr(cutlass.Float32, flat_weights, 4),
        _gptr(cutlass.Float4E2M1FN, workspace.packed_a_view),
        workspace.sfa_ptr,
        _gptr(cutlass.Uint8, workspace.packed_a_flat),
        _gptr(cutlass.Uint8, workspace.scale_flat),
        workspace.barrier_count,
        workspace.barrier_epoch,
        workspace.pair_head,
        workspace.producers_done_count,
        workspace.all_work_published,
        workspace.task_head,
        workspace.task_tail,
        _gptr(cutlass.Int32, workspace.task_ready, 4),
        _gptr(cutlass.Int32, workspace.task_expert, 4),
        _gptr(cutlass.Int32, workspace.task_m_tile, 4),
        _gptr(cutlass.Int32, workspace.task_slice_begin, 4),
        _gptr(cutlass.Int32, workspace.task_slice_count, 4),
        _gptr(cutlass.Int32, workspace.task_valid_rows, 4),
        _gptr(cutlass.Int32, workspace.tile_write_count, 4),
        weights.w13_fp4,
        weights.sfb_w13_ptr,
        weights.down_fp4,
        weights.sfb_down_ptr,
        workspace.row_counts,
        workspace.expert_write_rows,
        workspace.expert_tile_base,
        workspace.input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        workspace.down_input_scale,
        _gptr(cutlass.BFloat16, scatter_output),
        _gptr(cutlass.Int32, workspace.token_map, 4),
        _gptr(cutlass.Float32, workspace.token_weights, 4),
        m,
        max_rows,
        workspace.physical_tiles_capacity * _dynamic_tile_m(quant_mode),
        workspace.task_capacity,
        workspace.physical_tiles_capacity,
        stream,
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
    fast_math: bool,
    stream,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    activation: str = "silu",
    quant_mode: str = "nvfp4",
    unit_scale_contract: bool = False,
) -> None:
    quant_mode = _normalize_quant_mode(quant_mode)
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    micro_cls = activation_spec.micro_kernel_cls
    use_micro_direct = quant_mode == "nvfp4" and micro_cls.is_supported(
        m=m,
        k=k,
        n=n,
        num_topk=num_topk,
        weight_E=weight_E,
    )
    if use_micro_direct:
        if flat_ids.dtype in (torch.int32, torch.int64) and flat_ids.is_contiguous():
            launch_ids = flat_ids
        else:
            launch_ids = workspace.compact_topk_ids[: flat_ids.numel()]
            launch_ids.copy_(flat_ids.to(torch.int32))
        compiled, grid_x = _get_micro_kernel(
            weight_E,
            m,
            k,
            n,
            num_topk,
            topk_ids_dtype=topk_ids_dtype,
            fast_math=fast_math,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=(m == 1),
            activation=activation,
            device=a.device,
            quant_mode=quant_mode,
        )
        if _compiled_direct_micro_accepts_block_dim(
            compiled,
            _DIRECT_MICRO_BLOCK_DIM,
        ):
            _reset_volatile_launch_state(workspace)
            micro_cls.launch(
                compiled,
                x=a,
                w1_fp4=weights.w1_storage,
                w1_blockscale=weights.w1_scale_storage,
                w1_alphas=weights.w1_alpha,
                a1_gscale=input_gs,
                a2_gscale=down_input_scale,
                inter_fp32=workspace.micro_intermediate,
                w2_fp4=weights.w2_storage,
                w2_blockscale=weights.w2_scale_storage,
                w2_alphas=weights.w2_alpha,
                topk_ids=launch_ids.view(m, num_topk),
                topk_weights=flat_weights.view(m, num_topk),
                out=scatter_output,
                barrier_count=workspace.barrier_count,
                barrier_epoch=workspace.barrier_epoch,
                m=m,
                grid_x=grid_x,
            )
            return

    static_mac = _get_impl_mac("static", routed_rows=routed_rows)
    if routed_rows <= 16:
        static_mac = min(static_mac, 32)
    elif routed_rows < 40:
        # Tiny compact launches have very little FC2 tile work, so capping
        # resident clusters avoids idle CTA participation in the barrier phases.
        static_mac = min(static_mac, 64)

    compiled, mac = _get_static_kernel(
        workspace.state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        workspace.max_rows,
        topk_ids_dtype=topk_ids_dtype,
        fast_math=fast_math,
        mac_override=static_mac,
        activation=activation,
        single_token=(m == 1),
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        quant_mode=quant_mode,
    )
    launch_ids = flat_ids
    _reset_volatile_launch_state(workspace)
    compiled(
        a,
        launch_ids,
        flat_weights,
        workspace.packed_a_view,
        workspace.sfa_ptr,
        workspace.packed_a_flat,
        workspace.scale_flat,
        workspace.barrier_count,
        workspace.barrier_epoch,
        weights.w13_fp4,
        weights.sfb_w13_ptr,
        weights.down_fp4,
        weights.sfb_down_ptr,
        workspace.row_counts,
        workspace.active_expert_count,
        workspace.weight_expert_ids,
        workspace.global_to_local_expert,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output,
        workspace.token_map,
        workspace.token_weights,
        stream,
    )


@torch._dynamo.disable
def b12x_moe_fp4(
    a: torch.Tensor,  # [m, k] bf16 activations
    a1_gscale: torch.Tensor,  # [E] or scalar — reciprocal input quant global scale
    w1_fp4: torch.Tensor,  # [E, 2*n, k//2] uint8
    w1_blockscale: torch.Tensor,  # [E, ...] float8_e4m3fn swizzled
    w1_alphas: torch.Tensor,  # [E] float32
    a2_gscale: torch.Tensor,  # [E] or scalar — reciprocal intermediate quant global scale
    w2_fp4: torch.Tensor,  # [E, k, n//2] uint8
    w2_blockscale: torch.Tensor,  # [E, ...] float8_e4m3fn swizzled
    w2_alphas: torch.Tensor,  # [E] float32
    topk_weights: torch.Tensor,  # [m, topk] float
    topk_ids: torch.Tensor,  # [m, topk] int
    apply_router_weight_on_input: bool = False,
    *,
    workspace: TPMoEWorkspace | TPMoEWorkspacePool,
    output: torch.Tensor | None = None,
    input_scales_are_reciprocal: bool | None = None,
    input_scales_static: bool = False,
    fast_math: bool | None = None,
    activation: str = "silu",
    quant_mode: str | None = None,
    unit_scale_contract: bool = False,
    source_format: str = "modelopt",
    prepared_w4a16: object | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """MoE with shape-selected fused static or dynamic kernels.

    Compact workloads use the graph-safe static backend. All larger routed
    workloads use dynamic. Large token batches are chunked only when the chosen
    backend cannot describe the required work buffers in a single launch.
    """
    _assert_reciprocal_input_scale_contract(input_scales_are_reciprocal)
    quant_mode_arg = quant_mode
    quant_mode = _normalize_quant_mode(quant_mode_arg)
    source_format = _normalize_fp4_source_format(source_format)
    _validate_fp4_source_format_for_quant_mode(
        source_format=source_format,
        quant_mode=quant_mode,
    )
    num_topk = topk_ids.shape[1]
    m, k = a.shape
    device = a.device
    if prepared_w4a16 is not None:
        if quant_mode != "w4a16":
            raise ValueError("prepared_w4a16 requires quant_mode='w4a16'")
        prepared_hidden = int(getattr(prepared_w4a16, "hidden_size"))
        if prepared_hidden != k:
            raise ValueError(
                f"prepared_w4a16 hidden_size mismatch: expected {k}, got {prepared_hidden}"
            )
        prepared_dtype = getattr(prepared_w4a16, "params_dtype", a.dtype)
        if prepared_dtype != a.dtype:
            raise TypeError(
                f"prepared_w4a16 was built for {prepared_dtype}, but a has dtype {a.dtype}"
            )
        weight_E = int(getattr(prepared_w4a16, "num_experts"))
        n = int(getattr(prepared_w4a16, "intermediate_size"))
    else:
        weight_E = w1_fp4.shape[0]
        n = w2_fp4.shape[2] * 2  # intermediate_size
        expected_w1_rows = _activation_w1_rows(activation, n)
        if w1_fp4.shape[1] != expected_w1_rows:
            raise ValueError(
                f"expected w1_fp4.shape[1] == {expected_w1_rows} for activation "
                f"{activation!r}, got {w1_fp4.shape[1]}"
            )
    routed_rows = m * num_topk
    if apply_router_weight_on_input and quant_mode != "w4a16":
        raise NotImplementedError(
            "apply_router_weight_on_input is not implemented in b12x_moe_fp4"
        )
    if swiglu_limit is not None and quant_mode != "w4a16":
        raise NotImplementedError("swiglu_limit is implemented only for W4A16 MoE")
    if fast_math is None:
        fast_math = _FAST_MATH_DEFAULT
    if prepared_w4a16 is None and quant_mode_arg is None and quant_mode == "w4a16":
        w1_alphas = _w4a16_default_alpha(
            w1_alphas,
            a1_gscale,
            weight_E,
        )
        w2_alphas = _w4a16_default_alpha(
            w2_alphas,
            a2_gscale,
            weight_E,
        )
    # Shared scalar input scales are weight-side constants in the benchmarked
    # path, so treat them as static and avoid re-expanding them every launch.
    effective_input_scales_static = input_scales_static or (
        a1_gscale.numel() == 1 and a2_gscale.numel() == 1
    )
    if quant_mode == "w4a16":
        from b12x.moe.fused.w4a16.kernel import run_w4a16_moe

        if output is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires a caller-owned output buffer"
                )
            scatter_output = torch.empty(m, k, dtype=a.dtype, device=device)
        else:
            scatter_output = output
        if scatter_output.shape != (m, k):
            raise ValueError(
                f"output must have shape {(m, k)}, got {tuple(scatter_output.shape)}"
            )
        if scatter_output.dtype != a.dtype:
            raise ValueError(
                f"output must have dtype {a.dtype}, got {scatter_output.dtype}"
            )
        if scatter_output.device != device:
            raise ValueError(
                f"output must be on device {device}, got {scatter_output.device}"
            )
        if not scatter_output.is_contiguous():
            raise ValueError("output must be contiguous")

        prepared = prepared_w4a16
        if prepared is None:
            prepared = _get_w4a16_packed_weights(
                w1_fp4,
                w1_blockscale,
                w1_alphas,
                w2_fp4,
                w2_blockscale,
                w2_alphas,
                activation=activation,
                params_dtype=a.dtype,
                source_format=source_format,
            )
        plan = _make_workspace_plan(
            num_tokens=m,
            weight_E=weight_E,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            dtype=a.dtype,
            quant_mode=quant_mode,
            activation=activation,
        )
        w4a16_workspace = _resolve_workspace(
            workspace,
            plan=plan,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            input_scales_static=effective_input_scales_static,
        )
        if not isinstance(w4a16_workspace, TPW4A16Workspace):
            raise TypeError("expected a TPW4A16Workspace for the W4A16 backend")
        if not topk_weights.is_contiguous():
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires contiguous W4A16 topk_weights"
                )
            topk_weights = topk_weights.contiguous()
        if not topk_ids.is_contiguous():
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires contiguous W4A16 topk_ids"
                )
            topk_ids = topk_ids.contiguous()
        return run_w4a16_moe(
            a,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            fast_math=fast_math,
            intermediate_cache13=w4a16_workspace.intermediate_cache13,
            intermediate_cache2=w4a16_workspace.intermediate_cache2,
            output=scatter_output,
            fc1_c_tmp=w4a16_workspace.fc1_c_tmp,
            fc2_c_tmp=w4a16_workspace.fc2_c_tmp,
            packed_route_indices=w4a16_workspace.packed_route_indices,
            block_expert_ids=w4a16_workspace.block_expert_ids,
            packed_route_count=w4a16_workspace.packed_route_count,
            expert_offsets=w4a16_workspace.expert_offsets,
            swiglu_limit=swiglu_limit,
        )
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    if quant_mode == "nvfp4" and _is_exact_relu2_bs1_nemotron_case(
        activation=activation,
        a=a,
        w1_fp4=w1_fp4,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        w2_fp4=w2_fp4,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    ):
        scatter_output = _resolve_scatter_output(
            a=a,
            output=output,
            device=device,
            m=m,
            k=k,
        )
        return _launch_exact_relu2_bs1_nemotron(
            workspace=workspace,
            a=a,
            a1_gscale=a1_gscale,
            w1_fp4=w1_fp4,
            w1_blockscale=w1_blockscale,
            w1_alphas=w1_alphas,
            a2_gscale=a2_gscale,
            w2_fp4=w2_fp4,
            w2_blockscale=w2_blockscale,
            w2_alphas=w2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            scatter_output=scatter_output,
            fast_math=fast_math,
            input_scales_static=effective_input_scales_static,
        )
    workspace_policy = _workspace_policy(workspace)
    plan = _make_workspace_plan(
        num_tokens=m,
        weight_E=weight_E,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        dtype=a.dtype,
        quant_mode=quant_mode,
        activation=activation,
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
                input_scales_static=effective_input_scales_static,
                fast_math=fast_math,
                activation=activation,
                quant_mode=quant_mode,
                unit_scale_contract=unit_scale_contract,
                swiglu_limit=swiglu_limit,
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
            activation_spec=activation_spec,
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
            activation_spec=activation_spec,
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
        raise ValueError(
            f"output must have shape {(m, k)}, got {tuple(scatter_output.shape)}"
        )
    if scatter_output.dtype != a.dtype:
        raise ValueError(
            f"output must have dtype {a.dtype}, got {scatter_output.dtype}"
        )
    if scatter_output.device != device:
        raise ValueError(
            f"output must be on device {device}, got {scatter_output.device}"
        )
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
            fast_math=fast_math,
            stream=stream,
            activation=activation,
            quant_mode=quant_mode,
            share_input_across_experts=(
                quant_mode == "nvfp4" and a1_gscale.numel() == 1
            ),
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
            fast_math=fast_math,
            stream=stream,
            share_input_across_experts=(
                activation in ("relu2", "silu")
                and m == 1
                and a1_gscale.numel() == 1
                and os.environ.get("B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS", "1") != "0"
            ),
            share_expert_scales=(
                activation in ("relu2", "silu")
                and a1_gscale.numel() == 1
                and a2_gscale.numel() == 1
            ),
            activation=activation,
            quant_mode=quant_mode,
            unit_scale_contract=unit_scale_contract,
        )
    return scatter_output


def _validate_sparse_routing(
    hidden_states: torch.Tensor, routing: B12XTopKRouting
) -> None:
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
    if (
        routing.router_logits is not None
        and routing.router_logits.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError(
            "router_logits batch mismatch: expected "
            f"{hidden_states.shape[0]}, got {routing.router_logits.shape[0]}"
        )
    if (
        routing.flat_ids is not None
        and routing.flat_ids.numel() != routing.topk_ids.numel()
    ):
        raise ValueError(
            "flat_ids size mismatch: expected "
            f"{routing.topk_ids.numel()}, got {routing.flat_ids.numel()}"
        )
    if (
        routing.flat_weights is not None
        and routing.flat_weights.numel() != routing.topk_weights.numel()
    ):
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
    required = _route_workspace_nbytes(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
    )
    _emit_route_workspace_stats(
        storage="standalone",
        required_nbytes=required,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        device=device,
        logits_dtype=logits_dtype,
    )
    return _TPRouteWorkspace(
        router_logits=torch.empty(
            num_tokens, num_experts, device=device, dtype=logits_dtype
        ),
        topk_logits=torch.empty(num_tokens, top_k, device=device, dtype=torch.float32),
        topk_ids=torch.empty(num_tokens, top_k, device=device, dtype=torch.int32),
        topk_weights=torch.empty(num_tokens, top_k, device=device, dtype=torch.float32),
    )


def _route_workspace_specs(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    logits_dtype: torch.dtype,
) -> tuple[_TensorAllocSpec, ...]:
    return (
        _TensorAllocSpec("router_logits", (num_tokens, num_experts), logits_dtype),
        _TensorAllocSpec("topk_logits", (num_tokens, top_k), torch.float32),
        _TensorAllocSpec("topk_ids", (num_tokens, top_k), torch.int32),
        _TensorAllocSpec("topk_weights", (num_tokens, top_k), torch.float32),
    )


def _route_workspace_nbytes(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    logits_dtype: torch.dtype,
) -> int:
    offset = 0
    for spec in _route_workspace_specs(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
    ):
        offset = align_up(offset, max(16, _dtype_nbytes(spec.dtype)))
        offset += _tensor_numel(spec.shape) * _dtype_nbytes(spec.dtype)
    return int(offset)


def _emit_route_workspace_stats(
    *,
    storage: str,
    required_nbytes: int,
    capacity_nbytes: int | None = None,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    logits_dtype: torch.dtype,
) -> None:
    return


def _materialize_route_workspace(
    shared_arena: torch.Tensor,
    *,
    offset_bytes: int,
    capacity_nbytes: int,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    logits_dtype: torch.dtype,
) -> _TPRouteWorkspace:
    required = _route_workspace_nbytes(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
    )
    if capacity_nbytes < required:
        raise ValueError(
            f"MoE route workspace requires {required} bytes, but only {capacity_nbytes} are available"
        )
    _emit_route_workspace_stats(
        storage="shared",
        required_nbytes=required,
        capacity_nbytes=capacity_nbytes,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        device=shared_arena.device,
        logits_dtype=logits_dtype,
    )
    offset = int(offset_bytes)
    tensors: Dict[str, torch.Tensor] = {}
    for spec in _route_workspace_specs(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
    ):
        tensors[spec.name], offset = _allocate_arena_tensor(shared_arena, offset, spec)
    return _TPRouteWorkspace(
        router_logits=tensors["router_logits"],
        topk_logits=tensors["topk_logits"],
        topk_ids=tensors["topk_ids"],
        topk_weights=tensors["topk_weights"],
    )


def _slice_route_workspace(
    route_workspace: _TPRouteWorkspace, num_tokens: int
) -> _TPRouteWorkspace:
    if route_workspace.router_logits.shape[0] == num_tokens:
        return route_workspace
    return _TPRouteWorkspace(
        router_logits=route_workspace.router_logits[:num_tokens],
        topk_logits=route_workspace.topk_logits[:num_tokens],
        topk_ids=route_workspace.topk_ids[:num_tokens],
        topk_weights=route_workspace.topk_weights[:num_tokens],
    )


def _get_route_workspace(
    hidden_states: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    logits_dtype: torch.dtype,
    workspace: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool | None,
) -> _TPRouteWorkspace | None:
    if workspace is None:
        return None

    m = hidden_states.shape[0]
    device = hidden_states.device

    if isinstance(workspace, TPMoEWorkspacePool):
        key = (
            device.index,
            num_experts,
            top_k,
            logits_dtype,
        )
        route_workspace = workspace.route_workspaces.get(key)
        needs_growth = (
            route_workspace is None
            or route_workspace.router_logits.shape[0] < m
            or route_workspace.router_logits.shape[1] != num_experts
            or route_workspace.topk_ids.shape[1] != top_k
            or route_workspace.router_logits.dtype != logits_dtype
            or route_workspace.router_logits.device != device
        )
        if needs_growth:
            if workspace.shared_arena is None:
                route_workspace = _alloc_route_workspace(
                    num_tokens=m,
                    num_experts=num_experts,
                    top_k=top_k,
                    device=device,
                    logits_dtype=logits_dtype,
                )
            else:
                if workspace.shared_arena.device != device:
                    raise ValueError(
                        f"MoE pool arena device {workspace.shared_arena.device} does not match hidden_states device {device}"
                    )
                route_workspace = _materialize_route_workspace(
                    workspace.shared_arena,
                    offset_bytes=0,
                    capacity_nbytes=workspace.route_workspace_nbytes,
                    num_tokens=m,
                    num_experts=num_experts,
                    top_k=top_k,
                    logits_dtype=logits_dtype,
                )
            workspace.route_workspaces[key] = route_workspace
        return _slice_route_workspace(route_workspace, m)

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
            route_workspace.router_logits.add_(
                gate_bias.to(route_workspace.router_logits.dtype)
            )
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
    input_scales_are_reciprocal: bool | None = None,
    input_scales_static: bool = False,
    fast_math: bool | None = None,
    activation: str = "silu",
    quant_mode: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, B12XTopKRouting]:
    """Sparse-block FP4 MoE wrapper above the routed-expert TP primitive.

    This additive entrypoint preserves `b12x_moe_fp4(...)` as the stable
    low-level contract while giving higher-level integrations a single call that
    can own `gate -> topk -> routed experts` at the sparse MoE block seam.
    """

    _assert_reciprocal_input_scale_contract(input_scales_are_reciprocal)
    quant_mode_arg = quant_mode
    quant_mode_normalized = _normalize_quant_mode(quant_mode_arg)
    _validate_fp4_source_format_for_quant_mode(
        source_format=experts.source_format,
        quant_mode=quant_mode_normalized,
    )

    if routing is not None:
        if (
            top_k is not None
            or gate_weight is not None
            or gate_bias is not None
            or router_logits is not None
        ):
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
        input_scales_static=input_scales_static,
        fast_math=fast_math,
        activation=activation,
        quant_mode=quant_mode_arg,
        source_format=experts.source_format,
    )
    if routed_scaling_factor != 1.0:
        routed_output.mul_(routed_scaling_factor)
    if return_routing:
        return routed_output, selected
    return routed_output
