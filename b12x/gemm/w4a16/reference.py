"""Python reference for W4A16 dense GEMM.

Mirrors `b12x.moe.fused.w4a16.reference.moe_reference_w4a16` without
routing, expert dim, FC2, or intermediate quant. Operates in fp32 and
casts the final result to bf16.

Convention (matches the MoE W4A16 path):

* ``w_fp4``: packed FP4 weight, shape ``[N, K // 2]`` ``uint8``.
* ``w_blockscale``: swizzled FP8 block scales, shape
  ``[rows_padded, cols_padded]`` ``float8_e4m3fn`` where
  ``rows_padded = ceil(N / 128) * 128``,
  ``cols_padded = ceil(K / 16 / 4) * 4``.  The swizzle is the inverse of
  ``b12x.moe.fused.w4a16.reference.unswizzle_block_scale``.
* ``w_alpha``: scalar fp32 (``weight_scale_2``).  Dequant is
  ``packed_fp4 * unswizzled_sf * w_alpha``.
"""

from __future__ import annotations

import torch

from b12x.cute.fp4 import (
    _fp4_quantize_values,
    pack_grouped_fp4_values,
    swizzle_block_scale,
)
from b12x.moe.fused.reference import (
    _apply_block_scales,
    _dequant_fp4,
    _make_fp4_lut,
    unswizzle_block_scale,
)

_SF_VEC_SIZE = 16
_FP4_E2M1_MAX = 6.0


def dense_reference_w4a16(
    x: torch.Tensor,
    *,
    w_fp4: torch.Tensor | None = None,
    w_blockscale: torch.Tensor | None = None,
    w_alpha: torch.Tensor | float = 1.0,
    w_bf16: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense W4A16 reference matmul.

    Either pass ``(w_fp4, w_blockscale, w_alpha)`` for the production
    quantized path, or ``w_bf16`` for a debug-only float path.

    Returns ``[M, N]`` bf16.
    """
    if w_bf16 is not None:
        if w_fp4 is not None or w_blockscale is not None:
            raise ValueError("Pass w_bf16 alone, not with w_fp4/w_blockscale.")
        out_fp32 = x.float() @ w_bf16.float().t()
        return out_fp32.to(torch.bfloat16)

    if w_fp4 is None or w_blockscale is None:
        raise ValueError("Must provide (w_fp4, w_blockscale, w_alpha) or w_bf16.")

    n = w_fp4.shape[0]
    k = w_fp4.shape[1] * 2  # FP4 packed two-per-byte
    cols_blocks = k // _SF_VEC_SIZE

    fp4_lut = _make_fp4_lut(w_fp4.device)
    raw_w = _dequant_fp4(w_fp4, rows=n, cols=k, fp4_lut=fp4_lut)
    sf_f32 = unswizzle_block_scale(w_blockscale, rows=n, cols_blocks=cols_blocks)
    w_dequant_fp32 = _apply_block_scales(
        raw_w, sf_f32, rows=n, cols=k, block_size=_SF_VEC_SIZE
    )
    alpha = float(w_alpha) if not isinstance(w_alpha, torch.Tensor) else float(w_alpha.item())
    w_dequant_fp32 = w_dequant_fp32 * alpha

    out_fp32 = x.float() @ w_dequant_fp32.t()
    return out_fp32.to(torch.bfloat16)


def quantize_dense_weight_to_fp4(
    w_bf16: torch.Tensor,
    *,
    weight_scale_2: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a ``[N, K]`` bf16 weight to b12x's W4A16 storage format.

    Returns ``(w_fp4, w_blockscale, w_alpha)`` where:

    * ``w_fp4`` is ``[N, K // 2]`` ``uint8`` (FP4 packed two-per-byte).
    * ``w_blockscale`` is ``[rows_padded, cols_padded]``
      ``float8_e4m3fn`` (swizzled).
    * ``w_alpha`` is a 0-d ``float32`` tensor equal to
      ``weight_scale_2``.

    The chosen ``weight_scale_2`` is folded into the kernel's epilogue
    multiplier; the on-device per-block FP8 scale is sized so
    ``packed_fp4 * sf * weight_scale_2 ≈ w_bf16`` to FP4 precision.
    """
    if w_bf16.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(w_bf16.shape)}")
    if w_bf16.dtype != torch.bfloat16:
        raise ValueError(f"expected bf16 weight, got {w_bf16.dtype}")
    n, k = w_bf16.shape
    if k % _SF_VEC_SIZE != 0:
        raise ValueError(f"K={k} must be divisible by {_SF_VEC_SIZE}")

    device = w_bf16.device
    cols_blocks = k // _SF_VEC_SIZE

    # Per-block absmax.  Reshape `(N, K) -> (N, K/16, 16)`.
    blocks = w_bf16.float().view(n, cols_blocks, _SF_VEC_SIZE)
    block_absmax = blocks.abs().amax(dim=-1)  # [N, K/16]

    # Per-block FP8 scale, chosen so absmax / (sf_eff * weight_scale_2) == 6.
    raw_scale = (block_absmax / _FP4_E2M1_MAX / weight_scale_2).clamp_min(1e-30)
    sf_fp8 = raw_scale.to(torch.float8_e4m3fn)
    sf_eff_fp32 = sf_fp8.to(torch.float32) * weight_scale_2  # multiplier the kernel will apply

    # Quantize values: normalize by sf_eff, clamp to FP4 range, snap to grid.
    sf_eff_expanded = sf_eff_fp32.unsqueeze(-1).expand(n, cols_blocks, _SF_VEC_SIZE).clamp_min(1e-30)
    normalized = (blocks / sf_eff_expanded).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    quantized = _fp4_quantize_values(normalized.reshape(n, k))  # values in {0, ±0.5, ±1, ..., ±6}

    # Pack two nibbles per byte.  Use the existing grouped packer with
    # G=1 (the helper returns shape (rows, cols/2, G), so we squeeze).
    packed_3d = pack_grouped_fp4_values(quantized.unsqueeze(0))  # (N, K/2, 1)
    w_fp4 = packed_3d.squeeze(-1).contiguous()

    # Swizzle block scales to the kernel storage layout.
    w_blockscale = swizzle_block_scale(sf_fp8)  # (rows_padded, cols_padded) fp8_e4m3fn

    w_alpha = torch.tensor(weight_scale_2, dtype=torch.float32, device=device)
    return w_fp4, w_blockscale, w_alpha


__all__ = ["dense_reference_w4a16", "quantize_dense_weight_to_fp4"]
