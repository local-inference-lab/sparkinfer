from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
NVFP4_BLOCK_SIZE = 16

E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def require_sm120() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for b12x tests")
    return torch.device("cuda")


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def cast_from_fp4(x: torch.Tensor) -> torch.Tensor:
    v_lo = x.to(torch.uint8) & 0xF
    v_hi = (x.to(torch.uint8) >> 4) & 0xF
    combined = torch.stack((v_lo, v_hi), dim=-1)
    new_shape = combined.shape[:-2] + (combined.shape[-2] * combined.shape[-1],)
    lookup = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32, device=x.device)
    return lookup[combined.to(torch.long)].reshape(new_shape)


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x = torch.abs(x.clone())
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def _reciprocal(x: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.zeros_like(x), 1.0 / x)
    if x == 0:
        return 0.0
    return 1.0 / x


def ref_fp4_quant(
    x: torch.Tensor,
    global_scale: torch.Tensor | float,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sliced_shape = x.shape[:-1] + (x.shape[-1] // block_size, block_size)
    sliced_x = x.reshape(sliced_shape)
    vec_max = torch.max(torch.abs(sliced_x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * _reciprocal(FLOAT4_E2M1_MAX))
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = _reciprocal(scale * _reciprocal(global_scale))
    scaled_x = sliced_x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(x.shape)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def ref_grouped_fp4_quantize(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, rows, cols = input_tensor.shape
    quantized = torch.zeros(
        (num_groups, rows, cols), dtype=torch.float32, device=input_tensor.device
    )
    scales = torch.zeros(
        (num_groups, rows, cols // NVFP4_BLOCK_SIZE),
        dtype=torch.float32,
        device=input_tensor.device,
    )
    for group_idx in range(num_groups):
        valid_rows = int(row_counts[group_idx].item())
        if valid_rows == 0:
            continue
        quantized[group_idx, :valid_rows], scales[group_idx, :valid_rows] = ref_fp4_quant(
            input_tensor[group_idx, :valid_rows].float(),
            global_scale[group_idx],
            NVFP4_BLOCK_SIZE,
        )
    return quantized, scales


def ref_grouped_silu_mul_quantize(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cols = input_tensor.shape[-1] // 2
    left = input_tensor[..., :cols].float()
    right = input_tensor[..., cols:].float()
    activated = (F.silu(left) * right).to(input_tensor.dtype).to(torch.float32)
    return ref_grouped_fp4_quantize(activated, row_counts, global_scale)


def swizzle_block_scale_reference(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        scale = scale.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    batch, rows, cols = scale.shape
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols, 4)
    padded = torch.zeros(
        (batch, rows_padded, cols_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:, :rows, :cols] = scale
    swizzled = padded.reshape(batch, rows_padded // 128, 4, 32, cols_padded // 4, 4)
    swizzled = swizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    swizzled = swizzled.reshape(batch, rows_padded, cols_padded)
    return swizzled[0] if squeeze_batch else swizzled


def recover_grouped_e4m3_scales(
    scale_view: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    num_groups = scale_view.shape[-1]
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols // NVFP4_BLOCK_SIZE, 4)
    swizzled = scale_view.permute(5, 2, 4, 0, 1, 3).contiguous()
    swizzled = swizzled.reshape(num_groups, rows_padded, cols_padded)
    unswizzled = swizzled.view(
        num_groups,
        rows_padded // 128,
        cols_padded // 4,
        32,
        4,
        4,
    )
    unswizzled = unswizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    unswizzled = unswizzled.reshape(num_groups, rows_padded, cols_padded)
    return unswizzled[:, :rows, : cols // NVFP4_BLOCK_SIZE].to(torch.float32)


def dequantize_grouped_nvfp4(
    packed: torch.Tensor,
    scale_view: torch.Tensor,
    cols: int,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    if global_scale.numel() == 1:
        global_scale = global_scale.expand(packed.shape[0]).contiguous()
    packed_fp32 = cast_from_fp4(packed.view(torch.uint8)).view(
        packed.shape[0], packed.shape[1], cols
    )
    scales = recover_grouped_e4m3_scales(scale_view, packed.shape[1], cols)
    values = packed_fp32.view(packed.shape[0], packed.shape[1], cols // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    return (
        values * scales.unsqueeze(-1) / global_scale.view(-1, 1, 1, 1)
    ).reshape(packed.shape[0], packed.shape[1], cols)


def dequantize_token_major_nvfp4(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    *,
    hidden_size: int,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    x_fp4_float = cast_from_fp4(x_fp4.view(torch.uint8))
    num_tokens = x_fp4_float.shape[0]
    x_fp4_float = x_fp4_float.view(num_tokens, hidden_size // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    scales = x_sf.float().view(num_tokens, hidden_size // NVFP4_BLOCK_SIZE, 1)
    return (x_fp4_float * scales).view(num_tokens, hidden_size) / global_scale.item()


def compute_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32)
    value = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    return torch.tensor([value], dtype=torch.float32, device=x.device)


def compute_per_group_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=(1, 2)).to(torch.float32)
    numerator = torch.full_like(amax, FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
    return torch.where(amax > 0, numerator / amax, torch.ones_like(amax))


def llama_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
