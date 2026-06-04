from __future__ import annotations

from dataclasses import dataclass

import torch
from b12x.cute.fp4 import fp4_quantize_values_torch


@dataclass(frozen=True)
class OracleMetrics:
    max_abs: float
    rmse: float
    mean_abs: float
    cos: float


@dataclass(frozen=True)
class MoERouteTrace:
    token_idx: int
    route_idx: int
    expert_idx: int
    activation: str
    router_weight: float
    alpha_fc1: float
    alpha_fc2: float
    gs_fc1: float
    gs_fc2: float
    x_dequant: torch.Tensor
    fc1_out: torch.Tensor | None
    gate_out: torch.Tensor | None
    up_out: torch.Tensor | None
    intermediate: torch.Tensor
    int_dequant: torch.Tensor
    down_out: torch.Tensor
    routed_out: torch.Tensor
    routed_out_accum: torch.Tensor


@dataclass(frozen=True)
class FlashInferTrtllmFP4E8M0K32Weights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


_E8M0_K32_BF16_MAX_SCALE_BYTE = 247


def compare_to_reference(actual: torch.Tensor, reference: torch.Tensor) -> OracleMetrics:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    diff = actual_fp32 - reference_fp32
    actual_rows = actual_fp32.reshape(actual_fp32.shape[0], -1)
    reference_rows = reference_fp32.reshape(reference_fp32.shape[0], -1)
    dot = (actual_rows * reference_rows).sum(dim=1)
    actual_norm = actual_rows.norm(dim=1)
    reference_norm = reference_rows.norm(dim=1)
    denom = actual_norm * reference_norm
    both_zero = (actual_norm <= 1e-12) & (reference_norm <= 1e-12)
    cos_rows = torch.where(
        both_zero,
        torch.ones_like(dot),
        torch.where(denom > 1e-24, dot / denom, torch.zeros_like(dot)),
    )
    cos = cos_rows.mean().item()
    return OracleMetrics(
        max_abs=diff.abs().max().item(),
        rmse=diff.square().mean().sqrt().item(),
        mean_abs=diff.abs().mean().item(),
        cos=cos,
    )


def unswizzle_block_scale(swizzled_scale: torch.Tensor, rows: int, cols_blocks: int) -> torch.Tensor:
    cols_padded = ((cols_blocks + 3) // 4) * 4
    rows_padded = ((rows + 127) // 128) * 128
    unswizzled = swizzled_scale.view(torch.float8_e4m3fn).reshape(
        rows_padded // 128, cols_padded // 4, 32, 4, 4,
    )
    unswizzled = unswizzled.permute(0, 3, 2, 1, 4).contiguous()
    unswizzled = unswizzled.reshape(rows_padded, cols_padded)
    return unswizzled[:rows, :cols_blocks].to(torch.float32)


def _block_scale_interleave_128x4_torch(unswizzled_sf: torch.Tensor) -> torch.Tensor:
    """Byte-preserving torch equivalent of FlashInfer/TRT-LLM block_scale_interleave."""
    if unswizzled_sf.dtype not in {torch.uint8, torch.bfloat16}:
        raise TypeError(f"expected uint8 or bfloat16 scale tensor, got {unswizzled_sf.dtype}")
    if unswizzled_sf.dim() == 2:
        sf = unswizzled_sf.reshape(1, unswizzled_sf.shape[0], unswizzled_sf.shape[1])
    elif unswizzled_sf.dim() == 3:
        sf = unswizzled_sf
    else:
        raise ValueError(f"expected 2D or 3D scale tensor, got shape {tuple(unswizzled_sf.shape)}")

    batches, rows, cols = sf.shape
    rows_padded = ((int(rows) + 127) // 128) * 128
    cols_padded = ((int(cols) + 3) // 4) * 4
    if rows_padded != rows or cols_padded != cols:
        padded = sf.new_zeros((batches, rows_padded, cols_padded))
        padded[:, :rows, :cols] = sf
        sf = padded
    else:
        sf = sf.contiguous()

    swizzled = sf.reshape(batches, rows_padded // 128, 4, 32, cols_padded // 4, 4)
    swizzled = swizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    return swizzled.reshape(-1)


def _flashinfer_block_scale_interleave(unswizzled_sf: torch.Tensor) -> torch.Tensor:
    try:
        from flashinfer.fp4_quantization import nvfp4_block_scale_interleave

        return nvfp4_block_scale_interleave(unswizzled_sf)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "nvcc" not in msg and "cuda_home" not in msg:
            raise
        return _block_scale_interleave_128x4_torch(unswizzled_sf)


def _validate_reference_inputs(
    w1_fp4: torch.Tensor,
    I_tp: int,
    activation: str,
) -> None:
    if activation not in {"silu", "relu2"}:
        raise ValueError(f"unsupported activation {activation!r}")
    expected_w1_rows = 2 * I_tp if activation == "silu" else I_tp
    if w1_fp4.shape[1] != expected_w1_rows:
        raise ValueError(
            f"expected w1_fp4.shape[1] == {expected_w1_rows} for activation "
            f"{activation!r}, got {w1_fp4.shape[1]}"
        )


def _make_fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _dequant_fp4(
    packed_u8: torch.Tensor,
    rows: int,
    cols: int,
    fp4_lut: torch.Tensor,
) -> torch.Tensor:
    if packed_u8.dtype == torch.int8:
        packed_u8 = packed_u8.view(torch.uint8)
    lo = (packed_u8 & 0x0F).to(torch.int64)
    hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
    return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)


def _apply_block_scales(
    raw: torch.Tensor,
    sf_f32: torch.Tensor,
    rows: int,
    cols: int,
    *,
    block_size: int,
) -> torch.Tensor:
    n_blocks = (int(cols) + int(block_size) - 1) // int(block_size)
    sf = sf_f32[:rows, :n_blocks]
    padded_cols = n_blocks * int(block_size)
    if padded_cols != int(cols):
        padded = raw.new_zeros((int(rows), padded_cols))
        padded[:, : int(cols)] = raw
        raw = padded
    scaled = raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(
        rows,
        padded_cols,
    )
    return scaled[:, : int(cols)]


def _e8m0_scales_to_float(scales: torch.Tensor) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scales.dtype == torch.uint8:
        if e8m0_dtype is None:
            raise TypeError("uint8 E8M0 scales require torch.float8_e8m0fnu")
        scales = scales.view(e8m0_dtype)
    return scales.to(torch.float32)


def _e8m0_scale_bytes(
    scales: torch.Tensor,
    *,
    scale_byte_clamp: int | None = None,
) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scales.dtype == torch.uint8:
        scale_bytes = scales
    elif e8m0_dtype is not None and scales.dtype == e8m0_dtype:
        scale_bytes = scales.view(torch.uint8)
    else:
        raise TypeError("E8M0 K/32 scales must be torch.uint8 or torch.float8_e8m0fnu")
    if scale_byte_clamp is None:
        return scale_bytes.contiguous()
    return scale_bytes.clamp(max=int(scale_byte_clamp)).contiguous()


def _normalize_w13_layout(w13_layout: str) -> str:
    layout = str(w13_layout).lower()
    if layout not in {"w13", "w31"}:
        raise ValueError(f"unsupported W13 layout {w13_layout!r}")
    return layout


def _interleave_flashinfer_w1_w3_rows(
    tensor: torch.Tensor,
    *,
    intermediate_size: int,
    activation: str,
) -> torch.Tensor:
    if activation != "silu":
        return tensor.contiguous()
    w1 = tensor[:, :intermediate_size]
    w3 = tensor[:, intermediate_size:]
    return torch.stack([w3, w1], dim=2).reshape(tensor.shape).contiguous()


def prepare_flashinfer_trtllm_fp4_e8m0_k32_weights(
    w13_fp4: torch.Tensor,
    w13_e8m0_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
    scale_byte_clamp: int | None = None,
) -> FlashInferTrtllmFP4E8M0K32Weights:
    """Prepare b12x FP4/E8M0 K/32 source tensors for FlashInfer TRT-LLM MXFP4.

    The source W13 row contract is vLLM DeepSeek native loading order,
    [w1/gate, w3/up].  FlashInfer TRT-LLM expects the vLLM DeepSeek conversion
    style: [up0, gate0, up1, gate1, ...], then the TRT-LLM row permutation.
    Scale bytes stay E8M0 bytes; the final float8_e4m3fn dtype is only
    FlashInfer's ABI carrier for the interleaved byte storage.
    """
    _validate_reference_inputs(w13_fp4, I_tp, activation)
    if not w13_fp4.is_cuda or not w2_fp4.is_cuda:
        raise RuntimeError("FlashInfer TRT-LLM FP4 preparation requires CUDA tensors")
    if int(K) % 32 != 0 or int(I_tp) % 32 != 0:
        raise ValueError(f"FlashInfer MXFP4 prep requires K and I_tp divisible by 32, got K={K}, I_tp={I_tp}")
    rows_w13 = 2 * int(I_tp) if activation == "silu" else int(I_tp)
    if tuple(w13_e8m0_scale.shape) != (int(w13_fp4.shape[0]), rows_w13, int(K) // 32):
        raise ValueError(
            f"w13_e8m0_scale must have shape {(int(w13_fp4.shape[0]), rows_w13, int(K) // 32)}, "
            f"got {tuple(w13_e8m0_scale.shape)}"
        )
    if tuple(w2_e8m0_scale.shape) != (int(w2_fp4.shape[0]), int(K), int(I_tp) // 32):
        raise ValueError(
            f"w2_e8m0_scale must have shape {(int(w2_fp4.shape[0]), int(K), int(I_tp) // 32)}, "
            f"got {tuple(w2_e8m0_scale.shape)}"
        )

    try:
        from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("FlashInfer TRT-LLM FP4 prep requires flashinfer") from exc

    w13_u8 = w13_fp4.view(torch.uint8).contiguous()
    w2_u8 = w2_fp4.view(torch.uint8).contiguous()
    w13_s_u8 = _e8m0_scale_bytes(w13_e8m0_scale, scale_byte_clamp=scale_byte_clamp)
    w2_s_u8 = _e8m0_scale_bytes(w2_e8m0_scale, scale_byte_clamp=scale_byte_clamp)

    w13_u8 = _interleave_flashinfer_w1_w3_rows(
        w13_u8,
        intermediate_size=int(I_tp),
        activation=activation,
    )
    w13_s_u8 = _interleave_flashinfer_w1_w3_rows(
        w13_s_u8,
        intermediate_size=int(I_tp),
        activation=activation,
    )

    cache: dict = {}
    epilogue_tile_m = 128
    w13_perm = get_w2_permute_indices_with_cache(
        cache,
        w13_u8[0],
        epilogue_tile_m,
    ).to(w13_u8.device)
    w13_out = w13_u8[:, w13_perm].contiguous()

    w13_sf_perm = get_w2_permute_indices_with_cache(
        cache,
        w13_s_u8[0],
        epilogue_tile_m,
        num_elts_per_sf=16,
    ).to(w13_s_u8.device)
    w13_s = w13_s_u8[:, w13_sf_perm].contiguous()
    E, N_s, K_s = w13_s.shape
    w13_scale_out = (
        _flashinfer_block_scale_interleave(w13_s.reshape(E * N_s, K_s))
        .reshape(E, rows_w13, int(K) // 32)
        .view(torch.float8_e4m3fn)
    )

    w2_perm = get_w2_permute_indices_with_cache(
        cache,
        w2_u8[0],
        epilogue_tile_m,
    ).to(w2_u8.device)
    w2_out = w2_u8[:, w2_perm].contiguous()

    w2_sf_perm = get_w2_permute_indices_with_cache(
        cache,
        w2_s_u8[0],
        epilogue_tile_m,
        num_elts_per_sf=16,
    ).to(w2_s_u8.device)
    w2_s = w2_s_u8[:, w2_sf_perm].contiguous()
    E2, N2_s, K2_s = w2_s.shape
    w2_scale_out = (
        _flashinfer_block_scale_interleave(w2_s.reshape(E2 * N2_s, K2_s))
        .reshape(E2, int(K), int(I_tp) // 32)
        .view(torch.float8_e4m3fn)
    )

    return FlashInferTrtllmFP4E8M0K32Weights(
        w13=w13_out,
        w13_scale=w13_scale_out,
        w2=w2_out,
        w2_scale=w2_scale_out,
    )


def _per_expert_float32(
    scale: torch.Tensor,
    *,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    scale = scale.to(device=device, dtype=torch.float32)
    if scale.numel() == 1:
        return scale.reshape(1).expand(num_experts).contiguous()
    if scale.numel() != num_experts:
        raise ValueError(f"expected scalar or {num_experts} per-expert scales, got {scale.numel()}")
    return scale.reshape(num_experts).contiguous()


def pack_flashinfer_trtllm_topk_ids_weights(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Pack top-k ids/weights exactly like vLLM/SGLang's TRT-LLM wrappers."""
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(f"shape mismatch: topk_ids={tuple(topk_ids.shape)} topk_weights={tuple(topk_weights.shape)}")
    weight_bits = (
        topk_weights.contiguous().to(torch.bfloat16).view(torch.int16).to(torch.int32)
        & 0xFFFF
    )
    return (topk_ids.contiguous().to(torch.int32) << 16) | weight_bits


def moe_reference_w4a16_fp4_e8m0_k32_flashinfer(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_e8m0_scale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    scale_byte_clamp: int | None = _E8M0_K32_BF16_MAX_SCALE_BYTE,
) -> torch.Tensor:
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    if int(E) != int(w1_fp4.shape[0]) or int(E) != int(w2_fp4.shape[0]):
        raise ValueError("E must match the expert dimension of w1_fp4 and w2_fp4")
    prepared = prepare_flashinfer_trtllm_fp4_e8m0_k32_weights(
        w1_fp4,
        w1_e8m0_scale,
        w2_fp4,
        w2_e8m0_scale,
        K,
        I_tp,
        activation=activation,
        scale_byte_clamp=scale_byte_clamp,
    )
    return moe_reference_w4a16_fp4_e8m0_k32_flashinfer_prepared(
        x,
        prepared,
        w1_alphas,
        w2_alphas,
        topk_ids,
        topk_weights,
        E,
        K,
        I_tp,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )


def moe_reference_w4a16_fp4_e8m0_k32_flashinfer_prepared(
    x: torch.Tensor,
    prepared: FlashInferTrtllmFP4E8M0K32Weights,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        raise TypeError(f"FlashInfer W4A16 oracle expects BF16 activations, got {x.dtype}")
    if int(E) != int(prepared.w13.shape[0]) or int(E) != int(prepared.w2.shape[0]):
        raise ValueError("E must match the expert dimension of FlashInfer prepared weights")

    try:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer import ActivationType, RoutingMethodType
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("FlashInfer TRT-LLM FP4 oracle requires flashinfer") from exc

    packed_topk = pack_flashinfer_trtllm_topk_ids_weights(topk_ids, topk_weights)
    output = torch.empty(x.shape[0], K, dtype=torch.bfloat16, device=x.device)

    g1_scale = _per_expert_float32(w1_alphas, num_experts=int(E), device=x.device)
    g2_scale = _per_expert_float32(w2_alphas, num_experts=int(E), device=x.device)
    clamp_limit = None
    if swiglu_limit is not None:
        if activation != "silu":
            raise ValueError("swiglu_limit requires a gated W4A16 activation")
        clamp_limit = torch.full((int(E),), float(swiglu_limit), dtype=torch.float32, device=x.device)

    if activation == "silu":
        activation_type = int(ActivationType.Swiglu)
    elif activation == "relu2":
        activation_type = int(ActivationType.Relu2)
    else:
        raise ValueError(f"unsupported activation {activation!r}")

    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=packed_topk,
        routing_bias=None,
        hidden_states=x.contiguous(),
        hidden_states_scale=None,
        gemm1_weights=prepared.w13,
        gemm1_weights_scale=prepared.w13_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=clamp_limit,
        gemm2_weights=prepared.w2,
        gemm2_weights_scale=prepared.w2_scale,
        gemm2_bias=None,
        output1_scale_scalar=g1_scale,
        output1_scale_gate_scalar=g1_scale,
        output2_scale_scalar=g2_scale,
        num_experts=int(E),
        top_k=int(topk_ids.shape[1]),
        n_group=1,
        topk_group=1,
        intermediate_size=int(I_tp),
        local_expert_offset=0,
        local_num_experts=int(E),
        routed_scaling_factor=1.0,
        routing_method_type=int(RoutingMethodType.TopK),
        do_finalize=True,
        activation_type=activation_type,
        output=output,
        tune_max_num_tokens=max(16, int(x.shape[0])),
    )
    return result[0]


def _quantize_vec_to_fp4_dequant(
    vals_f32: torch.Tensor,
    global_scale: float,
    *,
    block_size: int,
    fp8_e4m3_max: float,
) -> torch.Tensor:
    cols = vals_f32.shape[0]
    n_blocks = cols // block_size
    blocked = vals_f32.reshape(n_blocks, block_size)
    block_max = blocked.abs().amax(dim=-1)

    raw_scale = (block_max * global_scale / 6.0).clamp(max=fp8_e4m3_max)
    sf_e4m3 = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)

    sf_times_gs = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols) / global_scale
    scaled = vals_f32 / sf_times_gs.clamp(min=1e-30)
    quant = fp4_quantize_values_torch(scaled)
    sf_only = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols)
    return quant * sf_only


def _trace_nvfp4_route(
    *,
    x_f32: torch.Tensor,
    w1_fp4_eid: torch.Tensor,
    w1_blockscale_eid: torch.Tensor,
    alpha_fc1: float,
    w2_fp4_eid: torch.Tensor,
    w2_blockscale_eid: torch.Tensor,
    alpha_fc2: float,
    gs_fc1: float,
    gs_fc2: float,
    K: int,
    I_tp: int,
    token_idx: int,
    route_idx: int,
    expert_idx: int,
    router_weight: float,
    activation: str,
) -> MoERouteTrace:
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)
    fp4_lut = _make_fp4_lut(x_f32.device)
    is_gated = activation == "silu"

    x_dequant = _quantize_vec_to_fp4_dequant(
        x_f32,
        gs_fc1,
        block_size=block_size,
        fp8_e4m3_max=fp8_e4m3_max,
    )
    w2_sf = unswizzle_block_scale(w2_blockscale_eid, K, I_tp // block_size)

    fc1_out = None
    gate_out = None
    up_out = None
    if is_gated:
        w13_sf = unswizzle_block_scale(w1_blockscale_eid, 2 * I_tp, K // block_size)
        up_dequant = _apply_block_scales(
            _dequant_fp4(w1_fp4_eid[:I_tp], I_tp, K, fp4_lut),
            w13_sf[:I_tp],
            I_tp,
            K,
            block_size=block_size,
        )
        gate_dequant = _apply_block_scales(
            _dequant_fp4(w1_fp4_eid[I_tp:], I_tp, K, fp4_lut),
            w13_sf[I_tp:],
            I_tp,
            K,
            block_size=block_size,
        )
        gate_out = (gate_dequant @ x_dequant) * alpha_fc1
        up_out = (up_dequant @ x_dequant) * alpha_fc1
        intermediate = (torch.sigmoid(gate_out) * gate_out * up_out).to(torch.bfloat16).float()
    else:
        w1_sf = unswizzle_block_scale(w1_blockscale_eid, I_tp, K // block_size)
        fc1_dequant = _apply_block_scales(
            _dequant_fp4(w1_fp4_eid[:I_tp], I_tp, K, fp4_lut),
            w1_sf[:I_tp],
            I_tp,
            K,
            block_size=block_size,
        )
        fc1_out = (fc1_dequant @ x_dequant) * alpha_fc1
        intermediate = torch.square(torch.relu(fc1_out)).to(torch.bfloat16).float()

    int_dequant = _quantize_vec_to_fp4_dequant(
        intermediate,
        gs_fc2,
        block_size=block_size,
        fp8_e4m3_max=fp8_e4m3_max,
    )
    down_dequant = _apply_block_scales(
        _dequant_fp4(w2_fp4_eid, K, I_tp, fp4_lut),
        w2_sf,
        K,
        I_tp,
        block_size=block_size,
    )
    down_out_f32 = (down_dequant @ int_dequant) * alpha_fc2
    down_out = down_out_f32.to(torch.bfloat16)
    routed_out = (router_weight * down_out.float()).to(torch.bfloat16)
    routed_out_accum = down_out_f32 * router_weight
    return MoERouteTrace(
        token_idx=token_idx,
        route_idx=route_idx,
        expert_idx=expert_idx,
        activation=activation,
        router_weight=router_weight,
        alpha_fc1=alpha_fc1,
        alpha_fc2=alpha_fc2,
        gs_fc1=gs_fc1,
        gs_fc2=gs_fc2,
        x_dequant=x_dequant,
        fc1_out=fc1_out,
        gate_out=gate_out,
        up_out=up_out,
        intermediate=intermediate,
        int_dequant=int_dequant,
        down_out=down_out,
        routed_out=routed_out,
        routed_out_accum=routed_out_accum,
    )


def trace_moe_reference_nvfp4_route(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    token_idx: int,
    route_idx: int,
    activation: str = "silu",
) -> MoERouteTrace:
    del E
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    if token_idx < 0 or token_idx >= x.shape[0]:
        raise IndexError(f"token_idx {token_idx} is out of range for batch {x.shape[0]}")
    if route_idx < 0 or route_idx >= topk_ids.shape[1]:
        raise IndexError(f"route_idx {route_idx} is out of range for top_k {topk_ids.shape[1]}")

    x_f32 = x[token_idx].float()
    expert_idx = int(topk_ids[token_idx, route_idx].item())
    router_weight = float(topk_weights[token_idx, route_idx].item())
    alpha_fc1 = float(w1_alphas[expert_idx].item())
    alpha_fc2 = float(w2_alphas[expert_idx].item())
    gs_fc1 = float(a1_gscale[expert_idx].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
    gs_fc2 = float(a2_gscale[expert_idx].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())
    return _trace_nvfp4_route(
        x_f32=x_f32,
        w1_fp4_eid=w1_fp4[expert_idx],
        w1_blockscale_eid=w1_blockscale[expert_idx],
        alpha_fc1=alpha_fc1,
        w2_fp4_eid=w2_fp4[expert_idx],
        w2_blockscale_eid=w2_blockscale[expert_idx],
        alpha_fc2=alpha_fc2,
        gs_fc1=gs_fc1,
        gs_fc2=gs_fc2,
        K=K,
        I_tp=I_tp,
        token_idx=token_idx,
        route_idx=route_idx,
        expert_idx=expert_idx,
        router_weight=router_weight,
        activation=activation,
    )


def moe_reference_f32(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    del E
    is_gated = activation == "silu"
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)

    fp4_lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
    def dequant_fp4(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        lo = (packed_u8 & 0x0F).to(torch.int64)
        hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
        return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)

    def apply_block_scales(raw: torch.Tensor, sf_f32: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        n_blocks = cols // block_size
        sf = sf_f32[:rows, :n_blocks]
        return raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)

    def quantize_vec_to_fp4_dequant(vals_f32: torch.Tensor, global_scale: float) -> torch.Tensor:
        cols = vals_f32.shape[0]
        n_blocks = cols // block_size
        blocked = vals_f32.reshape(n_blocks, block_size)
        block_max = blocked.abs().amax(dim=-1)

        raw_scale = (block_max * global_scale / 6.0).clamp(max=fp8_e4m3_max)
        sf_e4m3 = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)

        sf_times_gs = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols) / global_scale
        scaled = vals_f32 / sf_times_gs.clamp(min=1e-30)
        quant = fp4_quantize_values_torch(scaled)
        sf_only = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols)
        return quant * sf_only

    device = x.device
    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.float32, device=device)

    for t in range(m):
        x_f32 = x[t].float()
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            alpha_fc1 = float(w1_alphas[eid].item())
            alpha_fc2 = float(w2_alphas[eid].item())

            gs_fc1 = float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
            gs_fc2 = float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())

            x_dequant = quantize_vec_to_fp4_dequant(x_f32, gs_fc1)

            w2_sf = unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)

            if is_gated:
                w13_sf = unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
                up_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w13_sf[:I_tp], I_tp, K,
                )
                gate_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K), w13_sf[I_tp:], I_tp, K,
                )
                gate_out = (gate_dequant @ x_dequant) * alpha_fc1
                up_out = (up_dequant @ x_dequant) * alpha_fc1
                intermediate = torch.sigmoid(gate_out) * gate_out * up_out
            else:
                w1_sf = unswizzle_block_scale(w1_blockscale[eid], I_tp, K // block_size)
                fc1_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w1_sf[:I_tp], I_tp, K,
                )
                fc1_out = (fc1_dequant @ x_dequant) * alpha_fc1
                intermediate = torch.square(torch.relu(fc1_out))

            int_dequant = quantize_vec_to_fp4_dequant(intermediate, gs_fc2)
            down_dequant = apply_block_scales(
                dequant_fp4(w2_fp4[eid], K, I_tp), w2_sf, K, I_tp,
            )
            down_out = (down_dequant @ int_dequant) * alpha_fc2
            output[t] += router_w * down_out

    return output


def moe_reference_w4a16_f32(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    del E
    is_gated = activation == "silu"
    if swiglu_limit is not None and not is_gated:
        raise ValueError("swiglu_limit requires a gated W4A16 activation")

    block_size = 16
    fp4_lut = _make_fp4_lut(x.device)
    device = x.device
    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.float32, device=device)

    for t in range(m):
        x_f32 = x[t].float()
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            alpha_fc1 = float(w1_alphas[eid].item())
            alpha_fc2 = float(w2_alphas[eid].item())

            w2_sf = unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)

            if is_gated:
                w13_sf = unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
                up_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K, fp4_lut),
                    w13_sf[:I_tp],
                    I_tp,
                    K,
                    block_size=block_size,
                )
                gate_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K, fp4_lut),
                    w13_sf[I_tp:],
                    I_tp,
                    K,
                    block_size=block_size,
                )
                gate_out = (gate_dequant @ x_f32) * alpha_fc1
                up_out = (up_dequant @ x_f32) * alpha_fc1
                if swiglu_limit is not None:
                    gate_out = torch.clamp(gate_out, max=swiglu_limit)
                    up_out = torch.clamp(up_out, min=-swiglu_limit, max=swiglu_limit)
                intermediate = torch.sigmoid(gate_out) * gate_out * up_out
            else:
                w1_sf = unswizzle_block_scale(w1_blockscale[eid], I_tp, K // block_size)
                fc1_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K, fp4_lut),
                    w1_sf[:I_tp],
                    I_tp,
                    K,
                    block_size=block_size,
                )
                fc1_out = (fc1_dequant @ x_f32) * alpha_fc1
                intermediate = torch.square(torch.relu(fc1_out))

            down_dequant = _apply_block_scales(
                _dequant_fp4(w2_fp4[eid], K, I_tp, fp4_lut),
                w2_sf,
                K,
                I_tp,
                block_size=block_size,
            )
            down_out = (down_dequant @ intermediate) * alpha_fc2
            output[t] += router_w * down_out

    return output


def moe_reference_w4a16_fp4_e8m0_k32(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_e8m0_scale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    w13_layout: str = "w31",
) -> torch.Tensor:
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    if int(E) != int(w1_fp4.shape[0]) or int(E) != int(w2_fp4.shape[0]):
        raise ValueError("E must match the expert dimension of w1_fp4 and w2_fp4")
    is_gated = activation == "silu"
    w13_layout = _normalize_w13_layout(w13_layout)
    if swiglu_limit is not None and not is_gated:
        raise ValueError("swiglu_limit requires a gated W4A16 activation")

    block_size = 32
    if int(K) % block_size != 0:
        raise ValueError(f"E8M0 K/32 oracle requires K divisible by 32, got K={K}")
    if int(I_tp) % 8 != 0:
        raise ValueError(
            f"E8M0 K/32 oracle requires compact I_tp divisible by 8, got I_tp={I_tp}"
        )
    w1_rows = 2 * int(I_tp) if is_gated else int(I_tp)
    expected_w1_scale = (int(E), w1_rows, int(K) // block_size)
    expected_w2_scale = (int(E), int(K), (int(I_tp) + block_size - 1) // block_size)
    if tuple(w1_e8m0_scale.shape) != expected_w1_scale:
        raise ValueError(
            f"w1_e8m0_scale must have shape {expected_w1_scale}, got {tuple(w1_e8m0_scale.shape)}"
        )
    if tuple(w2_e8m0_scale.shape) != expected_w2_scale:
        raise ValueError(
            f"w2_e8m0_scale must have shape {expected_w2_scale}, got {tuple(w2_e8m0_scale.shape)}"
        )

    fp4_lut = _make_fp4_lut(x.device)
    output = torch.zeros(x.shape[0], K, dtype=torch.float32, device=x.device)
    top_k = topk_ids.shape[1]

    for t in range(x.shape[0]):
        x_f32 = x[t].float()
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            alpha_fc1 = float(w1_alphas[eid].item()) if w1_alphas.numel() > 1 else float(w1_alphas.item())
            alpha_fc2 = float(w2_alphas[eid].item()) if w2_alphas.numel() > 1 else float(w2_alphas.item())

            w2_sf = _e8m0_scales_to_float(w2_e8m0_scale[eid])
            if is_gated:
                w13_sf = _e8m0_scales_to_float(w1_e8m0_scale[eid])
                if w13_layout == "w31":
                    gate_rows = slice(0, I_tp)
                    up_rows = slice(I_tp, 2 * I_tp)
                else:
                    up_rows = slice(0, I_tp)
                    gate_rows = slice(I_tp, 2 * I_tp)
                up_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, up_rows], I_tp, K, fp4_lut),
                    w13_sf[up_rows],
                    I_tp,
                    K,
                    block_size=block_size,
                )
                gate_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, gate_rows], I_tp, K, fp4_lut),
                    w13_sf[gate_rows],
                    I_tp,
                    K,
                    block_size=block_size,
                )
                gate_out = (gate_dequant @ x_f32) * alpha_fc1
                up_out = (up_dequant @ x_f32) * alpha_fc1
                if swiglu_limit is not None:
                    gate_out = torch.clamp(gate_out, max=swiglu_limit)
                    up_out = torch.clamp(up_out, min=-swiglu_limit, max=swiglu_limit)
                intermediate = torch.sigmoid(gate_out) * gate_out * up_out
            else:
                w1_sf = _e8m0_scales_to_float(w1_e8m0_scale[eid])
                fc1_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K, fp4_lut),
                    w1_sf[:I_tp],
                    I_tp,
                    K,
                    block_size=block_size,
                )
                fc1_out = (fc1_dequant @ x_f32) * alpha_fc1
                intermediate = torch.square(torch.relu(fc1_out))

            down_dequant = _apply_block_scales(
                _dequant_fp4(w2_fp4[eid], K, I_tp, fp4_lut),
                w2_sf,
                K,
                I_tp,
                block_size=block_size,
            )
            down_out = (down_dequant @ intermediate) * alpha_fc2
            output[t] += router_w * down_out

    return output


def moe_reference_nvfp4(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.float32, device=x.device)

    for t in range(m):
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            trace = trace_moe_reference_nvfp4_route(
                x,
                w1_fp4,
                w1_blockscale,
                w1_alphas,
                w2_fp4,
                w2_blockscale,
                w2_alphas,
                a1_gscale,
                a2_gscale,
                topk_ids,
                topk_weights,
                E,
                K,
                I_tp,
                token_idx=t,
                route_idx=k_idx,
                activation=activation,
            )
            assert trace.expert_idx == eid
            output[t] += trace.routed_out_accum

    return output.to(torch.bfloat16)
