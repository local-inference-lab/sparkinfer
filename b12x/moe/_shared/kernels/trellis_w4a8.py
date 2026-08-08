"""End-to-end TP12 E4M3 trellis W4A8 MoE path.

The route-major reference path keeps the encoded weights compact throughout:
EXL transforms produce FP16 rows, rows are quantized to MXFP8, and the P24/P33
trellis payload is decoded directly into E4M3 MMA registers.  The caller owns
all scratch so the complete sequence is CUDA-graph safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.utils import cuda_stream_to_int
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8Rows,
    empty_mxfp8_rows_for_dense_gemm,
)
from b12x.moe._shared.kernels.trellis_w4a8_pair import (
    run_trellis_w4a8_fc1_routes,
    run_trellis_w4a8_fc2_routes,
)
from b12x.moe._shared.kernels.trellis_w4a8_transform import (
    run_trellis_w4a8_activation_rotation_quant,
    run_trellis_w4a8_input_rotation_quant,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _w4a16_topk_sum_launch_flat,
)


@dataclass(frozen=True)
class TrellisW4A8MoeScratch:
    """Caller-owned intermediate storage for one fixed ``(M, top-k)`` shape."""

    route_experts: torch.Tensor
    gate_rotated: torch.Tensor
    up_rotated: torch.Tensor
    # Broadcast checkpoints quantize one FC1 operand per token.  Per-expert
    # checkpoints quantize one operand per route because each route applies a
    # different input rotation.
    gate_quantized: MXFP8Rows
    up_quantized: MXFP8Rows
    gate_fc1: torch.Tensor
    up_fc1: torch.Tensor
    activated_rotated: torch.Tensor
    activated_quantized: MXFP8Rows
    fc2: torch.Tensor
    output: torch.Tensor


def make_trellis_w4a8_moe_scratch(
    *,
    m: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device | str,
    shared_suh: bool = True,
) -> TrellisW4A8MoeScratch:
    """Allocate the fixed buffers needed by :func:`run_trellis_w4a8_moe`."""

    m = int(m)
    topk = int(topk)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    shared_suh = bool(shared_suh)
    if m <= 0 or topk <= 0:
        raise ValueError("m and topk must be positive")
    if hidden_size <= 0 or hidden_size % 128:
        raise ValueError("hidden_size must be a positive multiple of 128")
    if intermediate_size != 256:
        raise ValueError(
            "the initial TP12 QSRT W4A8 path requires "
            f"intermediate_size=256, got {intermediate_size}"
        )
    device = torch.device(device)
    routes = m * topk
    input_rows = m if shared_suh else routes
    gate_rotated = torch.empty(
        (routes, hidden_size), dtype=torch.float16, device=device
    )
    up_rotated = torch.empty_like(gate_rotated)
    gate_fc1 = torch.empty(
        (routes, intermediate_size), dtype=torch.float16, device=device
    )
    up_fc1 = torch.empty_like(gate_fc1)
    activated_rotated = torch.empty_like(gate_fc1)
    fc2 = torch.empty_like(gate_rotated)
    return TrellisW4A8MoeScratch(
        route_experts=torch.empty((routes,), dtype=torch.int32, device=device),
        gate_rotated=gate_rotated,
        up_rotated=up_rotated,
        gate_quantized=empty_mxfp8_rows_for_dense_gemm(
            input_rows, hidden_size, device=device
        ),
        up_quantized=empty_mxfp8_rows_for_dense_gemm(
            input_rows, hidden_size, device=device
        ),
        gate_fc1=gate_fc1,
        up_fc1=up_fc1,
        activated_rotated=activated_rotated,
        activated_quantized=empty_mxfp8_rows_for_dense_gemm(
            routes, intermediate_size, device=device
        ),
        fc2=fc2,
        output=torch.empty((m, hidden_size), dtype=torch.float32, device=device),
    )


def _mxfp8_rows_prefix(rows: MXFP8Rows, m: int, *, name: str) -> MXFP8Rows:
    if rows.values.ndim != 2 or int(rows.values.shape[0]) < m:
        raise ValueError(f"{name} does not have capacity for {m} rows")
    if rows.scale_rows.ndim != 3 or int(rows.scale_rows.shape[1]) < m:
        raise ValueError(f"{name}.scale_rows does not have capacity for {m} rows")
    m_tiles = (m + 127) // 128
    if rows.scale_mma.ndim != 6 or int(rows.scale_mma.shape[2]) < m_tiles:
        raise ValueError(f"{name}.scale_mma does not have capacity for {m} rows")
    return MXFP8Rows(
        values=rows.values[:m],
        scale_rows=rows.scale_rows[:, :m],
        scale_mma=rows.scale_mma[:, :, :m_tiles],
    )


def view_trellis_w4a8_moe_scratch(
    scratch: TrellisW4A8MoeScratch,
    *,
    m: int,
    topk: int,
    shared_suh: bool = True,
) -> TrellisW4A8MoeScratch:
    """Return allocation-free prefixes of a capacity-sized scratch binding."""

    m = int(m)
    topk = int(topk)
    shared_suh = bool(shared_suh)
    if m <= 0 or topk <= 0:
        raise ValueError("m and topk must be positive")
    routes = m * topk
    input_rows = m if shared_suh else routes

    def prefix(name: str, tensor: torch.Tensor, rows: int) -> torch.Tensor:
        if tensor.ndim == 0 or int(tensor.shape[0]) < rows:
            raise ValueError(f"scratch.{name} does not have capacity for {rows} rows")
        return tensor[:rows]

    return TrellisW4A8MoeScratch(
        route_experts=prefix("route_experts", scratch.route_experts, routes),
        gate_rotated=prefix("gate_rotated", scratch.gate_rotated, routes),
        up_rotated=prefix("up_rotated", scratch.up_rotated, routes),
        gate_quantized=_mxfp8_rows_prefix(
            scratch.gate_quantized, input_rows, name="scratch.gate_quantized"
        ),
        up_quantized=_mxfp8_rows_prefix(
            scratch.up_quantized, input_rows, name="scratch.up_quantized"
        ),
        gate_fc1=prefix("gate_fc1", scratch.gate_fc1, routes),
        up_fc1=prefix("up_fc1", scratch.up_fc1, routes),
        activated_rotated=prefix(
            "activated_rotated", scratch.activated_rotated, routes
        ),
        activated_quantized=_mxfp8_rows_prefix(
            scratch.activated_quantized,
            routes,
            name="scratch.activated_quantized",
        ),
        fc2=prefix("fc2", scratch.fc2, routes),
        output=prefix("output", scratch.output, m),
    )


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_mxfp8_rows(
    name: str,
    rows: MXFP8Rows,
    *,
    m: int,
    k: int,
    device: torch.device,
) -> None:
    _check_tensor(
        f"{name}.values",
        rows.values,
        shape=(m, k),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _check_tensor(
        f"{name}.scale_rows",
        rows.scale_rows,
        shape=(1, m, k // 32),
        dtype=torch.float8_e8m0fnu,
        device=device,
    )
    expected_scale_mma = (
        32,
        4,
        (m + 127) // 128,
        4,
        (k + 127) // 128,
        1,
    )
    if tuple(rows.scale_mma.shape) != expected_scale_mma:
        raise ValueError(
            f"{name}.scale_mma must have shape {expected_scale_mma}, "
            f"got {tuple(rows.scale_mma.shape)}"
        )
    if rows.scale_mma.dtype != torch.float8_e8m0fnu:
        raise TypeError(
            f"{name}.scale_mma must have dtype torch.float8_e8m0fnu, "
            f"got {rows.scale_mma.dtype}"
        )
    if rows.scale_mma.device != device:
        raise ValueError(
            f"{name}.scale_mma must be on {device}, got {rows.scale_mma.device}"
        )


def run_trellis_w4a8_moe(
    source: torch.Tensor,
    prepared,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scratch: TrellisW4A8MoeScratch,
    *,
    expert_map: torch.Tensor | None = None,
    fast_math: bool = False,
) -> torch.Tensor:
    """Run a homogeneous compact E4M3 trellis tier through native W4A8 MMA.

    ``topk_ids`` are local indices into ``prepared`` unless ``expert_map`` is
    supplied.  In that case it is an int32 global-to-local table; ``-1`` map
    entries are inactive routes and contribute exactly zero.
    """

    if source.ndim != 2 or source.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("source must be contiguous BF16/FP16 [M,H]")
    if not source.is_contiguous():
        raise ValueError("source must be contiguous")
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be contiguous int32/int64 [M,top-k]")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if (
        topk_weights.shape != topk_ids.shape
        or topk_weights.dtype != torch.float32
    ):
        raise TypeError("topk_weights must be contiguous FP32 with topk_ids shape")
    if not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous")
    if topk_ids.device != source.device or topk_weights.device != source.device:
        raise ValueError("source and routing tensors must share one CUDA device")

    m, hidden_size = map(int, source.shape)
    topk = int(topk_ids.shape[1])
    if int(topk_ids.shape[0]) != m:
        raise ValueError(
            f"topk_ids must have one row per source token, got {topk_ids.shape[0]} and {m}"
        )
    routes = m * topk
    intermediate_size = int(prepared.intermediate_size)
    if int(prepared.hidden_size) != hidden_size:
        raise ValueError(
            f"source hidden size {hidden_size} != prepared {prepared.hidden_size}"
        )
    if intermediate_size != 256:
        raise ValueError("the initial TP12 W4A8 path requires local intermediate 256")
    trellis_codebook = str(getattr(prepared, "trellis_codebook", "")).lower()
    if trellis_codebook != "sqg_xor_cheb_t12":
        raise NotImplementedError(
            "native W4A8 trellis execution requires SQG-XOR-Cheb-T12, "
            f"got {trellis_codebook!r}"
        )
    activation = str(getattr(prepared, "activation", "")).lower()
    if activation not in {"silu", "situ"}:
        raise ValueError(
            "native W4A8 trellis execution requires activation 'silu' or "
            f"'situ', got {activation!r}"
        )
    num_experts = int(prepared.num_experts)
    for name in ("gate_suh", "up_suh"):
        suh = getattr(prepared, name, None)
        if (
            not isinstance(suh, torch.Tensor)
            or suh.ndim != 2
            or int(suh.shape[1]) != hidden_size
            or suh.dtype != torch.float16
            or suh.device != source.device
            or not suh.is_contiguous()
        ):
            raise ValueError(
                f"prepared.{name} must be contiguous fp16 [1|E,{hidden_size}] "
                f"on {source.device}"
            )
    shared_suh = getattr(prepared, "shared_suh", None)
    if type(shared_suh) is not bool:
        raise ValueError("prepared weights omit the explicit shared_suh contract")
    gate_suh_rows = int(prepared.gate_suh.shape[0])
    up_suh_rows = int(prepared.up_suh.shape[0])
    expected_suh_rows = 1 if shared_suh else num_experts
    if gate_suh_rows != expected_suh_rows or up_suh_rows != expected_suh_rows:
        raise ValueError(
            "prepared gate_suh/up_suh rows disagree with shared_suh; "
            f"got {gate_suh_rows}, {up_suh_rows}, expected {expected_suh_rows}"
        )
    input_rows = m if shared_suh else routes
    _check_mxfp8_rows(
        "scratch.gate_quantized",
        scratch.gate_quantized,
        m=input_rows,
        k=hidden_size,
        device=source.device,
    )
    _check_mxfp8_rows(
        "scratch.up_quantized",
        scratch.up_quantized,
        m=input_rows,
        k=hidden_size,
        device=source.device,
    )
    _check_mxfp8_rows(
        "scratch.activated_quantized",
        scratch.activated_quantized,
        m=routes,
        k=intermediate_size,
        device=source.device,
    )

    _check_tensor(
        "scratch.route_experts",
        scratch.route_experts,
        shape=(routes,),
        dtype=torch.int32,
        device=source.device,
    )
    for name, tensor, shape in (
        ("gate_rotated", scratch.gate_rotated, (routes, hidden_size)),
        ("up_rotated", scratch.up_rotated, (routes, hidden_size)),
        ("gate_fc1", scratch.gate_fc1, (routes, intermediate_size)),
        ("up_fc1", scratch.up_fc1, (routes, intermediate_size)),
        ("activated_rotated", scratch.activated_rotated, (routes, intermediate_size)),
        ("fc2", scratch.fc2, (routes, hidden_size)),
    ):
        _check_tensor(
            f"scratch.{name}",
            tensor,
            shape=shape,
            dtype=torch.float16,
            device=source.device,
        )
    _check_tensor(
        "scratch.output",
        scratch.output,
        shape=(m, hidden_size),
        dtype=torch.float32,
        device=source.device,
    )

    if expert_map is None:
        scratch.route_experts.copy_(topk_ids.reshape(-1))
    else:
        if (
            expert_map.ndim != 1
            or expert_map.dtype != torch.int32
            or expert_map.device != source.device
            or not expert_map.is_contiguous()
        ):
            raise TypeError(
                "expert_map must be contiguous int32 [global experts] on source device"
            )
        torch.index_select(
            expert_map,
            0,
            topk_ids.reshape(-1),
            out=scratch.route_experts,
        )
    run_trellis_w4a8_input_rotation_quant(
        source,
        scratch.route_experts,
        prepared,
        scratch.gate_quantized,
        scratch.up_quantized,
        topk=topk,
    )
    run_trellis_w4a8_fc1_routes(
        scratch.gate_quantized,
        scratch.up_quantized,
        prepared,
        scratch.route_experts,
        scratch.gate_fc1,
        scratch.up_fc1,
        topk=topk,
        shared_input=shared_suh,
        decode_both=False,
        sqg_direct_lut=True,
    )
    run_trellis_w4a8_activation_rotation_quant(
        scratch.gate_fc1,
        scratch.up_fc1,
        scratch.route_experts,
        prepared,
        scratch.activated_quantized,
        fast_math=fast_math,
    )
    run_trellis_w4a8_fc2_routes(
        scratch.activated_quantized,
        prepared,
        scratch.route_experts,
        scratch.fc2,
        sqg_direct_lut=True,
    )
    _w4a16_topk_sum_launch_flat(
        scratch.fc2,
        scratch.output,
        m,
        topk,
        hidden_size,
        "fp16",
        int(cuda_stream_to_int(torch.cuda.current_stream())),
        full_rotation=True,
        num_experts=num_experts,
        topk_weights=topk_weights,
        route_expert_ids=scratch.route_experts,
        svh_table=prepared.down_svh,
    )
    return scratch.output


__all__ = [
    "TrellisW4A8MoeScratch",
    "make_trellis_w4a8_moe_scratch",
    "view_trellis_w4a8_moe_scratch",
    "run_trellis_w4a8_moe",
]
