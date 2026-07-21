"""GEMM ops for sparkinfer.

- ``mm``: block-scaled dense matrix multiplication over ``[M,K,L]`` stacks.
- ``bmm``: dtype-dispatched batched matrix multiplication.
- ``blockscaled``: one-shot dense block-scaled GEMM (NVFP4 / MXFP4 / MXFP8).
- ``block_fp8_linear``: DeepSeek-style serialized block-FP8 linear via MXFP8.
- ``mxfp8_linear``: ModelOpt MXFP8 linear (one-shot).
- ``wo_projection``: fused MLA WO-A/WO-B projections (+ inverse-RoPE variant).
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = (
    "blockscaled",
    "block_fp8_linear",
    "mxfp8_linear",
    "wo_projection",
)
_FUNCTIONS = {
    "mm": (".._lib.dense_gemm", "dense_gemm"),
    "bmm": ("._bmm.api", "bmm"),
    "prewarm_bmm": ("._bmm.api", "prewarm_bmm"),
    "can_implement_bmm": ("._bmm.api", "can_implement_bmm"),
    "is_bmm_supported": ("._bmm.api", "is_bmm_supported"),
}


def __getattr__(name: str) -> Any:
    if name in _OP_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _FUNCTIONS:
        module_name, attribute = _FUNCTIONS[name]
        value = getattr(importlib.import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*_OP_MODULES, *_FUNCTIONS])


__all__ = [*_FUNCTIONS, *_OP_MODULES]
