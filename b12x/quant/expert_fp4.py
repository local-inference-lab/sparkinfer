"""Compatibility exports for the expert FP4 helper utilities."""

from b12x.cute.fp4 import (
    align_up,
    as_grouped_scale_view as _as_grouped_scale_view,
    quantize_grouped_nvfp4_torch,
    relu2_quantize_grouped_nvfp4_torch,
    silu_mul_quantize_grouped_nvfp4_torch,
    swizzle_block_scale,
)

__all__ = [
    "_as_grouped_scale_view",
    "align_up",
    "quantize_grouped_nvfp4_torch",
    "relu2_quantize_grouped_nvfp4_torch",
    "silu_mul_quantize_grouped_nvfp4_torch",
    "swizzle_block_scale",
]
