"""sparkinfer — consumer-Blackwell (SM120/SM121) kernels.

CuTe-DSL kernels for NVFP4/MXFP4/MXFP8 GEMM, fused MoE, attention (paged,
sparse/compressed MLA, NSA indexing), quantization, mHC residual, and PCIe
collectives, ported from the sparkinfer project.  One grammar everywhere:

- ops live at ``sparkinfer.<group>.<op>`` and declare themselves via ``META``;
- planned ops share the lifecycle ``Caps -> plan() -> bind() ->
  run*()`` (``plan`` may allocate; ``bind`` builds views only and never
  allocates; ``run*`` is CUDA-graph-capture safe);
- one-shot ops are plain functions; comm collectives are classes.

Serving controls (`freeze_kernel_resolution` & friends) live here at the
arch root because they guard the shared compiler: warm every kernel shape,
then freeze so a cache miss raises instead of compiling inside a live
request or graph capture.

Importing this module is cheap and side-effect free; kernels, cutlass, and
torch custom ops load on first op use.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from ._lib.meta import OpMeta
from ._lib.runtime_control import (
    KernelResolutionFrozenError,
    compilation_frozen,
    freeze_compilation,
    freeze_kernel_resolution,
    kernel_resolution_frozen,
    unfreeze_compilation,
    unfreeze_kernel_resolution,
)

# Static logical-op registry, kept in lockstep with public op directories and
# the explicit private-module overrides below by tests/test_registry.py.
_OPS: tuple[str, ...] = (
    "attention.paged",
    "attention.sparse_mla",
    "attention.compressed_mla",
    "attention.nsa_indexer",
    "attention.varlen",
    "comm.pcie",
    "gemm.blockscaled",
    "gemm.block_fp8_linear",
    "gemm.bmm",
    "gemm.mxfp8_linear",
    "gemm.wo_projection",
    "moe.fused_moe",
    "moe.ep_moe",
    "norm.mhc",
    "quantization.mxfp8",
    "quantization.nvfp4",
)

# A group-level function cannot share its name with an imported child module.
# These registry entries keep their public qualname while their metadata and
# implementation live under a private package.
_OP_MODULE_OVERRIDES: dict[str, str] = {
    "gemm.bmm": "gemm._bmm",
}
_CACHE_CLEAR_OVERRIDES: dict[str, str] = {
    "gemm.bmm": "clear_bmm_caches",
}

_GROUPS = ("attention", "comm", "gemm", "moe", "norm", "quantization")
_LAZY_ROOT_ATTRS: dict[str, tuple[str, str]] = {
    # public name -> (module, attribute)
    "ScratchBufferSpec": ("._lib.scratch", "ScratchBufferSpec"),
}


def _op_module_path(qualname: str) -> str:
    return _OP_MODULE_OVERRIDES.get(qualname, qualname)


def list_ops() -> tuple[OpMeta, ...]:
    """Import every op's (cheap) ``__init__`` and return their ``META``s."""
    return tuple(
        importlib.import_module(f".{_op_module_path(op_path)}", __name__).META
        for op_path in _OPS
    )


def find_op(qualname: str) -> OpMeta:
    """Look up one op's ``META`` by ``"<group>.<op>"`` qualname."""
    if qualname not in _OPS:
        raise KeyError(
            f"unknown experimental sparkinfer op {qualname!r}; known ops: {sorted(_OPS)}"
        )
    return importlib.import_module(f".{_op_module_path(qualname)}", __name__).META


def clear_all_caches() -> None:
    """Clear caches of every op already imported; never forces imports."""
    for op_path in _OPS:
        module_path = _op_module_path(op_path)
        api = sys.modules.get(f"{__name__}.{module_path}.api")
        clear_name = _CACHE_CLEAR_OVERRIDES.get(op_path, "clear_caches")
        clear = getattr(api, clear_name, None) if api is not None else None
        if clear is not None:
            clear()
    compiler = sys.modules.get(f"{__name__}._lib.compiler")
    if compiler is not None:
        compiler.clear_compile_cache()


def __getattr__(name: str) -> Any:
    if name in _GROUPS:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _LAZY_ROOT_ATTRS:
        module_name, attr = _LAZY_ROOT_ATTRS[name]
        value = getattr(importlib.import_module(module_name, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, *_GROUPS])


__all__ = [
    "KernelResolutionFrozenError",
    "OpMeta",
    "ScratchBufferSpec",
    "clear_all_caches",
    "compilation_frozen",
    "find_op",
    "freeze_compilation",
    "freeze_kernel_resolution",
    "kernel_resolution_frozen",
    "list_ops",
    "unfreeze_compilation",
    "unfreeze_kernel_resolution",
]
