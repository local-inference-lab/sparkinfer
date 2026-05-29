"""Unified SM120 sparse-MLA CuTeDSL backend (opt-in, parallel to the defaults).

A NEW FlashInfer-style 288-thread warp-specialized backend specialized at trace
time by ``cutlass.const_expr`` on three int-enum keys (``ModelType`` /
``ComputeMode`` / ``ScaleFormat``). Targets DSV4 (compressed, first) and GLM_NSA
(uncompressed); DSV3.2 / POW2_FP32 are DROPPED. Existing kernels remain default
and untouched. Gated by ``B12X_MLA_SM120_UNIFIED`` (see launch.py).
"""

from __future__ import annotations

from .launch import (
    run_unified_decode,
    run_unified_merge,
    run_unified_prefill,
)
from .traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
    UnifiedMLATraits,
    infer_model_type,
    make_unified_traits,
)

__all__ = [
    # enums (int-valued const_expr keys)
    "ModelType",
    "ComputeMode",
    "ScaleFormat",
    # traits
    "UnifiedMLATraits",
    "make_unified_traits",
    "infer_model_type",
    # launch / dispatch
    "run_unified_decode",
    "run_unified_prefill",
    "run_unified_merge",
]
