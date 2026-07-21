"""Implementation of the public planned full-rotation Trellis MoE op."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import prod

import torch

from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from .._shared.kernels.w4a16.host import (
    W4A16BufferPlan,
    max_packed_route_slots,
    plan_w4a16_buffers,
)
from .._shared.kernels.w4a16.kernel import (
    W4A16FusedMoeCompileResult,
    W4A16TopKSumCompileResult,
    clear_w4a16_kernel_cache,
    compile_w4a16_fused_moe,
    compile_w4a16_topk_sum,
    run_w4a16_moe,
)
from .._shared.kernels.w4a16.prepare import (
    PreparedNF3MoeWeights,
    _normalize_trellis256_codebook,
    prepare_trellis256_moe_weights,
)


_ALLOWED_BLOCK_M = (8, 16, 32, 48, 64)
_ALLOWED_BITS = (3, 4, 5, 6)
_DEFAULT_TILE_CONFIG = (64, 256, 64, 256)
_MCG_SENTINEL = 0xCBAC1FED
_ARENA_ALIGNMENT = 256


def _align_up(value: int, alignment: int = _ARENA_ALIGNMENT) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _normalize_tile_config(value: Sequence[int]) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError(
            "tile_config must be (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n)"
        )
    tile_config = tuple(int(item) for item in value)
    if any(item <= 0 or item % 16 != 0 for item in tile_config):
        raise ValueError(
            "every tile_config value must be a positive multiple of 16, got "
            f"{tile_config}"
        )
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = tile_config
    if fc1_tile_k * fc1_tile_n != fc2_tile_k * fc2_tile_n:
        raise ValueError(
            "FC1 and FC2 tile_config entries must select the same CTA thread count"
        )
    return tile_config


def _normalize_input_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "Trellis MoE input_dtype must be torch.bfloat16 or torch.float16, "
            f"got {dtype}"
        )
    return dtype


def _input_dtype_name(dtype: torch.dtype) -> str:
    return "bf16" if dtype == torch.bfloat16 else "fp16"


def _resolve_cuda_device(device: torch.device | str | int) -> torch.device:
    if isinstance(device, int):
        result = torch.device("cuda", int(device))
    else:
        result = torch.device(device)
    if result.type != "cuda":
        raise ValueError(f"Trellis MoE requires a CUDA device, got {result}")
    if result.index is None and torch.cuda.is_available():
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@dataclass(frozen=True, kw_only=True)
class TrellisMoECaps:
    """Fixed serving capacity and compile policy for one Trellis MoE shape."""

    max_tokens: int
    num_topk: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    device: torch.device | str | int
    input_dtype: torch.dtype = torch.bfloat16
    route_num_experts: int | None = None
    block_size_m: int = 8
    trellis_bits: int = 3
    tile_config: tuple[int, int, int, int] = _DEFAULT_TILE_CONFIG
    activation: str = "silu"
    fast_math: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_tokens",
            "num_topk",
            "num_experts",
            "hidden_size",
            "intermediate_size",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "full-rotation Trellis MoE requires hidden_size and "
                "intermediate_size divisible by 128"
            )
        object.__setattr__(self, "device", _resolve_cuda_device(self.device))
        object.__setattr__(
            self, "input_dtype", _normalize_input_dtype(self.input_dtype)
        )
        route_num_experts = (
            self.num_experts
            if self.route_num_experts is None
            else int(self.route_num_experts)
        )
        if route_num_experts <= 0:
            raise ValueError(
                f"route_num_experts must be positive, got {route_num_experts}"
            )
        if self.num_topk > route_num_experts:
            raise ValueError(
                f"num_topk={self.num_topk} exceeds route_num_experts={route_num_experts}"
            )
        object.__setattr__(self, "route_num_experts", route_num_experts)
        block_size_m = int(self.block_size_m)
        if block_size_m not in _ALLOWED_BLOCK_M:
            raise ValueError(
                f"block_size_m must be one of {_ALLOWED_BLOCK_M}, got {block_size_m}"
            )
        object.__setattr__(self, "block_size_m", block_size_m)
        trellis_bits = int(self.trellis_bits)
        if trellis_bits not in _ALLOWED_BITS:
            raise ValueError(
                f"trellis_bits must be one of {_ALLOWED_BITS}, got {trellis_bits}"
            )
        object.__setattr__(self, "trellis_bits", trellis_bits)
        tile_config = _normalize_tile_config(self.tile_config)
        fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = tile_config
        if self.hidden_size % fc1_tile_k != 0:
            raise ValueError("hidden_size must be divisible by FC1 tile K")
        if self.intermediate_size % fc1_tile_n != 0:
            raise ValueError(
                "projection-major intermediate_size must be divisible by FC1 tile N"
            )
        if self.intermediate_size % fc2_tile_k != 0:
            raise ValueError("intermediate_size must be divisible by FC2 tile K")
        if self.hidden_size % fc2_tile_n != 0:
            raise ValueError("hidden_size must be divisible by FC2 tile N")
        object.__setattr__(self, "tile_config", tile_config)
        if str(self.activation).strip().lower() != "silu":
            raise ValueError("the validated full-rotation Trellis recipe requires silu")
        object.__setattr__(self, "activation", "silu")
        object.__setattr__(self, "fast_math", bool(self.fast_math))

    @property
    def is_gated(self) -> bool:
        return True


@dataclass(frozen=True, eq=False)
class TrellisMoEWeights:
    """Zero-copy native tensors and persistent full-rotation tables."""

    w13: torch.Tensor
    w2: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    intermediate_rotations: torch.Tensor
    down_svh: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    trellis_bits: int
    tile_config: tuple[int, int, int, int]
    device: torch.device
    _prepared: PreparedNF3MoeWeights = field(repr=False)


@dataclass(frozen=True)
class _ArenaViewSpec:
    name: str
    offset_bytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return int(prod(self.shape)) * _dtype_nbytes(self.dtype)


@dataclass(frozen=True)
class _ArenaLayout:
    nbytes: int
    views: tuple[_ArenaViewSpec, ...]

    def materialize(self, scratch: torch.Tensor) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for spec in self.views:
            raw = scratch.narrow(0, spec.offset_bytes, spec.nbytes)
            result[spec.name] = raw.view(spec.dtype).view(spec.shape)
        return result


def _make_arena_layout(
    caps: TrellisMoECaps,
    buffers: W4A16BufferPlan,
    *,
    sms: int,
) -> _ArenaLayout:
    assert caps.route_num_experts is not None
    specs = (
        (
            "intermediate_cache13",
            (buffers.intermediate_cache13_elements,),
            torch.float16,
        ),
        ("intermediate_cache2", (buffers.intermediate_cache2_elements,), torch.float16),
        ("output", (caps.max_tokens, caps.hidden_size), torch.float32),
        ("fc1_c_tmp", (buffers.fc1_c_tmp_elements,), torch.float32),
        ("fc2_c_tmp", (buffers.fc2_c_tmp_elements,), torch.float32),
        ("packed_route_indices", (buffers.route_slots,), torch.int32),
        ("block_expert_ids", (buffers.route_blocks,), torch.int32),
        ("packed_route_count", (1,), torch.int32),
        ("expert_offsets", (caps.route_num_experts + 1,), torch.int32),
        ("expert_counts", (caps.route_num_experts,), torch.int32),
        ("rotation_a_gate", (buffers.rotation_a_elements,), torch.float16),
        ("rotation_a_up", (buffers.rotation_a_elements,), torch.float16),
        ("kernel_workspace", (int(sms) * 4 + 2,), torch.int32),
    )
    cursor = 0
    views: list[_ArenaViewSpec] = []
    for name, shape, dtype in specs:
        cursor = _align_up(cursor, max(_ARENA_ALIGNMENT, _dtype_nbytes(dtype)))
        spec = _ArenaViewSpec(
            name=name,
            offset_bytes=cursor,
            shape=tuple(int(dim) for dim in shape),
            dtype=dtype,
        )
        views.append(spec)
        cursor += spec.nbytes
    return _ArenaLayout(nbytes=max(_align_up(cursor), 1), views=tuple(views))


@dataclass(frozen=True)
class TrellisMoEPlan:
    """Compiled launches and byte-arena layout for one fixed capacity."""

    caps: TrellisMoECaps
    buffer_plan: W4A16BufferPlan
    fused_launch: W4A16FusedMoeCompileResult = field(repr=False)
    identity_sums: tuple[W4A16TopKSumCompileResult, ...] = field(repr=False)
    mapped_sums: tuple[W4A16TopKSumCompileResult, ...] = field(repr=False)
    _arena_layout: _ArenaLayout = field(repr=False)
    _scratch_specs: tuple[ScratchBufferSpec, ...] = field(repr=False)

    @property
    def scratch_nbytes(self) -> int:
        return self._arena_layout.nbytes

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(self, **kwargs) -> "TrellisMoEBinding":
        return bind_trellis_moe(self, **kwargs)

    def topk_sum_launch(
        self, ids_dtype: torch.dtype, *, mapped: bool
    ) -> W4A16TopKSumCompileResult:
        launches = self.mapped_sums if mapped else self.identity_sums
        for launch in launches:
            if launch.route_ids_dtype == ids_dtype:
                return launch
        raise TypeError(f"Trellis MoE does not have a top-k sum for {ids_dtype}")


@dataclass(frozen=True, kw_only=True)
class TrellisMoEBinding:
    """Stable tensor views for one launch; safe to retain across graph replay."""

    plan: TrellisMoEPlan
    weights: TrellisMoEWeights
    a: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    output: torch.Tensor
    route_expert_map: torch.Tensor | None
    output_expert_map: torch.Tensor | None
    prepared: PreparedNF3MoeWeights = field(repr=False)
    intermediate_cache13: torch.Tensor = field(repr=False)
    intermediate_cache2: torch.Tensor = field(repr=False)
    fc1_c_tmp: torch.Tensor = field(repr=False)
    fc2_c_tmp: torch.Tensor = field(repr=False)
    packed_route_indices: torch.Tensor = field(repr=False)
    block_expert_ids: torch.Tensor = field(repr=False)
    packed_route_count: torch.Tensor = field(repr=False)
    expert_offsets: torch.Tensor = field(repr=False)
    expert_counts: torch.Tensor = field(repr=False)
    rotation_a_gate: torch.Tensor = field(repr=False)
    rotation_a_up: torch.Tensor = field(repr=False)
    topk_sum_launch: W4A16TopKSumCompileResult = field(repr=False)

    def run(self) -> torch.Tensor:
        return run_trellis_moe(binding=self)


def _validate_rotation_table(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor.dtype != torch.float16:
        raise TypeError(f"{name} must be torch.float16, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_mcg(codebook: str | int, mcg: torch.Tensor | int | None) -> None:
    normalized = _normalize_trellis256_codebook(codebook)
    if normalized != "mcg":
        raise NotImplementedError(
            "the production Trellis MoE decoder accepts only the MCG codebook"
        )
    if mcg is None:
        return
    if isinstance(mcg, torch.Tensor):
        if mcg.numel() != 1 or mcg.dtype not in (torch.int32, torch.uint32):
            raise ValueError("mcg must be a scalar int32/uint32 tensor")
        marker = int(mcg.item()) & 0xFFFFFFFF
    else:
        marker = int(mcg) & 0xFFFFFFFF
    if marker != _MCG_SENTINEL:
        raise ValueError(
            f"unexpected MCG marker {marker:#010x}; expected {_MCG_SENTINEL:#010x}"
        )


def prepare_trellis_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    codebook: str | int = "mcg",
    mcg: torch.Tensor | int | None = None,
    tile_config: tuple[int, int, int, int] = _DEFAULT_TILE_CONFIG,
    dummy_scale: torch.Tensor | None = None,
) -> TrellisMoEWeights:
    """Validate and wrap projection-major native EXL3 tensors without copying."""
    _validate_mcg(codebook, mcg)
    tile_config = _normalize_tile_config(tile_config)
    if w13.ndim != 5 or int(w13.shape[0]) != 2:
        raise ValueError(
            "w13 must be projection-major [2,E,H/16,I/16,16*bits] int16 "
            "or the byte-identical [...,8*bits] int32 view"
        )
    if w2.ndim != 4:
        raise ValueError(
            "w2 must be [E,I/16,H/16,16*bits] int16 or the byte-identical "
            "[...,8*bits] int32 view"
        )
    num_experts = int(w13.shape[1])
    hidden_size = int(w13.shape[2]) * 16
    intermediate_size = int(w13.shape[3]) * 16
    if tuple(w2.shape[:3]) != (
        num_experts,
        intermediate_size // 16,
        hidden_size // 16,
    ):
        raise ValueError(
            "w2 geometry does not match projection-major w13: "
            f"got {tuple(w2.shape[:3])}"
        )
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError(
            "full-rotation Trellis weights require hidden and intermediate "
            "dimensions divisible by 128"
        )
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = tile_config
    if (
        hidden_size % fc1_tile_k != 0
        or intermediate_size % fc1_tile_n != 0
        or intermediate_size % fc2_tile_k != 0
        or hidden_size % fc2_tile_n != 0
    ):
        raise ValueError(
            f"tile_config={tile_config} does not divide H={hidden_size}, "
            f"I={intermediate_size}"
        )
    device = w13.device
    if w2.device != device:
        raise ValueError("w13 and w2 must be on the same device")
    _validate_rotation_table(
        "gate_suh", gate_suh, shape=(num_experts, hidden_size), device=device
    )
    _validate_rotation_table(
        "up_suh", up_suh, shape=(num_experts, hidden_size), device=device
    )
    _validate_rotation_table(
        "intermediate_rotations",
        intermediate_rotations,
        shape=(num_experts, 3 * intermediate_size),
        device=device,
    )
    _validate_rotation_table(
        "down_svh", down_svh, shape=(num_experts, hidden_size), device=device
    )
    prepared = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        activation="silu",
        params_dtype=torch.float16,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_n=fc2_tile_n,
        w13_layout="trellis3_t256_proj",
        codebook=codebook,
        dummy_scale=dummy_scale,
        gate_suh=gate_suh,
        up_suh=up_suh,
    )
    if prepared.trellis_codebook != "mcg":
        raise RuntimeError("Trellis preparation did not preserve the MCG contract")
    if (
        prepared.w13.data_ptr() != w13.data_ptr()
        or prepared.w2.data_ptr() != w2.data_ptr()
    ):
        raise RuntimeError("Trellis preparation unexpectedly copied native weights")
    return TrellisMoEWeights(
        w13=w13,
        w2=w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        trellis_bits=int(prepared.trellis_bits),
        tile_config=tile_config,
        device=device,
        _prepared=prepared,
    )


def plan_trellis_moe(caps: TrellisMoECaps) -> TrellisMoEPlan:
    """Compile every launch and produce the fixed caller-scratch layout."""
    if not isinstance(caps, TrellisMoECaps):
        raise TypeError("caps must be a TrellisMoECaps")
    device = _resolve_cuda_device(caps.device)
    if device.index is None:
        raise RuntimeError("CUDA must be available before planning Trellis MoE")
    if device != caps.device:
        caps = replace(caps, device=device)
    with torch.cuda.device(device):
        props = torch.cuda.get_device_properties(device)
        sms = int(props.multi_processor_count)
        max_shared_mem = int(getattr(props, "shared_memory_per_block_optin", 101_376))
        buffer_plan = plan_w4a16_buffers(
            caps,
            m=caps.max_tokens,
            topk=caps.num_topk,
            route_num_experts=caps.route_num_experts,
            sms=sms,
            full_rotation=True,
            block_size_m=caps.block_size_m,
        )
        route_slots = max_packed_route_slots(
            caps.max_tokens * caps.num_topk,
            caps.block_size_m,
            caps.route_num_experts,
        )
        max_m_blocks = (route_slots + caps.block_size_m - 1) // caps.block_size_m
        fused_launch = compile_w4a16_fused_moe(
            size_m=caps.max_tokens,
            hidden_size=caps.hidden_size,
            intermediate_size=caps.intermediate_size,
            num_experts=caps.num_experts,
            top_k=caps.num_topk,
            activation=caps.activation,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=caps.block_size_m,
            max_m_blocks=max_m_blocks,
            element_dtype="fp16",
            fast_math=caps.fast_math,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout="trellis3_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis3_t256_proj",
            trellis_bits=caps.trellis_bits,
            force_tile_config=caps.tile_config,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype=_input_dtype_name(caps.input_dtype),
        )
        identity_sums = tuple(
            compile_w4a16_topk_sum(
                m=caps.max_tokens,
                topk=caps.num_topk,
                hidden_size=caps.hidden_size,
                element_dtype="fp16",
                full_rotation=True,
                num_experts=caps.num_experts,
                route_num_experts=0,
                route_ids_dtype=ids_dtype,
                use_expert_map=False,
            )
            for ids_dtype in (torch.int32, torch.int64)
        )
        mapped_sums = tuple(
            compile_w4a16_topk_sum(
                m=caps.max_tokens,
                topk=caps.num_topk,
                hidden_size=caps.hidden_size,
                element_dtype="fp16",
                full_rotation=True,
                num_experts=caps.num_experts,
                route_num_experts=caps.route_num_experts,
                route_ids_dtype=ids_dtype,
                use_expert_map=True,
            )
            for ids_dtype in (torch.int32, torch.int64)
        )
    arena_layout = _make_arena_layout(caps, buffer_plan, sms=sms)
    specs = (
        scratch_buffer_spec(
            "trellis_moe",
            nbytes=arena_layout.nbytes,
            device=device,
        ),
    )
    return TrellisMoEPlan(
        caps=caps,
        buffer_plan=buffer_plan,
        fused_launch=fused_launch,
        identity_sums=identity_sums,
        mapped_sums=mapped_sums,
        _arena_layout=arena_layout,
        _scratch_specs=specs,
    )


def _validate_runtime_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_expert_map(
    name: str,
    tensor: torch.Tensor | None,
    *,
    caps: TrellisMoECaps,
) -> None:
    if tensor is None:
        return
    assert caps.route_num_experts is not None
    _validate_runtime_tensor(
        name,
        tensor,
        shape=(caps.route_num_experts,),
        dtype=torch.int32,
        device=caps.device,
    )


def bind_trellis_moe(
    plan: TrellisMoEPlan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    a: torch.Tensor,
    weights: TrellisMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    route_expert_map: torch.Tensor | None = None,
    output_expert_map: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> TrellisMoEBinding:
    """Bind runtime tensors by carving views from ``scratch`` only."""
    if not isinstance(plan, TrellisMoEPlan):
        raise TypeError("plan must come from trellis_moe.plan")
    if not isinstance(weights, TrellisMoEWeights):
        raise TypeError("weights must come from trellis_moe.prepare_weights")
    caps = plan.caps
    if (
        weights.hidden_size != caps.hidden_size
        or weights.intermediate_size != caps.intermediate_size
        or weights.num_experts != caps.num_experts
        or weights.trellis_bits != caps.trellis_bits
        or weights.tile_config != caps.tile_config
        or weights.device != caps.device
    ):
        raise ValueError("weights do not match the Trellis MoE plan")
    if a.ndim != 2:
        raise ValueError(f"a must be rank 2, got shape {tuple(a.shape)}")
    tokens = int(a.shape[0])
    if tokens < 1 or tokens > caps.max_tokens:
        raise ValueError(
            f"input tokens must be in [1, {caps.max_tokens}], got {tokens}"
        )
    _validate_runtime_tensor(
        "a",
        a,
        shape=(tokens, caps.hidden_size),
        dtype=caps.input_dtype,
        device=caps.device,
    )
    _validate_runtime_tensor(
        "topk_weights",
        topk_weights,
        shape=(tokens, caps.num_topk),
        dtype=torch.float32,
        device=caps.device,
    )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be torch.int32 or torch.int64")
    _validate_runtime_tensor(
        "topk_ids",
        topk_ids,
        shape=(tokens, caps.num_topk),
        dtype=topk_ids.dtype,
        device=caps.device,
    )
    _validate_expert_map("route_expert_map", route_expert_map, caps=caps)
    _validate_expert_map("output_expert_map", output_expert_map, caps=caps)

    scratch_storage = scratch_tensor(scratch, plan._scratch_specs, owner="Trellis MoE")
    if int(scratch_storage.data_ptr()) % _ARENA_ALIGNMENT != 0:
        raise ValueError(f"Trellis MoE scratch must be {_ARENA_ALIGNMENT}-byte aligned")
    views = plan._arena_layout.materialize(scratch_storage)
    views["kernel_workspace"].zero_()
    if output is None:
        output_view = views["output"][:tokens]
    else:
        if output.ndim != 2 or tuple(output.shape) not in (
            (tokens, caps.hidden_size),
            (caps.max_tokens, caps.hidden_size),
        ):
            raise ValueError(
                "output must be the live or capacity FP32 view: expected "
                f"{(tokens, caps.hidden_size)} or {(caps.max_tokens, caps.hidden_size)}, "
                f"got {tuple(output.shape)}"
            )
        if output.dtype != torch.float32:
            raise TypeError(f"output must be torch.float32, got {output.dtype}")
        if output.device != caps.device or not output.is_contiguous():
            raise ValueError("output must be contiguous on the planned CUDA device")
        output_view = output[:tokens]
    prepared = replace(weights._prepared, workspace=views["kernel_workspace"])
    return TrellisMoEBinding(
        plan=plan,
        weights=weights,
        a=a,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output_view,
        route_expert_map=route_expert_map,
        output_expert_map=output_expert_map,
        prepared=prepared,
        intermediate_cache13=views["intermediate_cache13"],
        intermediate_cache2=views["intermediate_cache2"],
        fc1_c_tmp=views["fc1_c_tmp"],
        fc2_c_tmp=views["fc2_c_tmp"],
        packed_route_indices=views["packed_route_indices"],
        block_expert_ids=views["block_expert_ids"],
        packed_route_count=views["packed_route_count"],
        expert_offsets=views["expert_offsets"],
        expert_counts=views["expert_counts"],
        rotation_a_gate=views["rotation_a_gate"],
        rotation_a_up=views["rotation_a_up"],
        topk_sum_launch=plan.topk_sum_launch(
            topk_ids.dtype, mapped=output_expert_map is not None
        ),
    )


def run_trellis_moe(*, binding: TrellisMoEBinding) -> torch.Tensor:
    """Run the preplanned full-rotation path into ``binding.output``."""
    if not isinstance(binding, TrellisMoEBinding):
        raise TypeError("binding must come from trellis_moe.bind")
    caps = binding.plan.caps
    return run_w4a16_moe(
        binding.a,
        binding.prepared,
        binding.topk_weights,
        binding.topk_ids,
        activation=caps.activation,
        intermediate_cache13=binding.intermediate_cache13,
        intermediate_cache2=binding.intermediate_cache2,
        output=binding.output,
        fc1_c_tmp=binding.fc1_c_tmp,
        fc2_c_tmp=binding.fc2_c_tmp,
        packed_route_indices=binding.packed_route_indices,
        block_expert_ids=binding.block_expert_ids,
        packed_route_count=binding.packed_route_count,
        expert_offsets=binding.expert_offsets,
        expert_counts=binding.expert_counts,
        expert_map=binding.route_expert_map,
        output_expert_map=binding.output_expert_map,
        apply_router_weight_on_input=False,
        fast_math=caps.fast_math,
        fused_launch=binding.plan.fused_launch,
        topk_sum_launch=binding.topk_sum_launch,
        intermediate_rotation_scales=binding.weights.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=binding.weights.gate_suh,
        suh_up_table=binding.weights.up_suh,
        svh_table=binding.weights.down_svh,
        rotation_a_gate=binding.rotation_a_gate,
        rotation_a_up=binding.rotation_a_up,
    )


def clear_trellis_moe_caches() -> None:
    """Clear the shared W4A16 compile caches used by this planned op."""
    clear_w4a16_kernel_cache()


__all__ = [
    "TrellisMoEBinding",
    "TrellisMoECaps",
    "TrellisMoEPlan",
    "TrellisMoEWeights",
    "bind_trellis_moe",
    "clear_trellis_moe_caches",
    "plan_trellis_moe",
    "prepare_trellis_moe_weights",
    "run_trellis_moe",
]
