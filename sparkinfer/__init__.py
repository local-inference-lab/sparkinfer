"""Public sparkinfer package surface."""

from .cute.runtime_patches import apply_cutlass_runtime_patches
from .cute.runtime_control import (
    KernelResolutionFrozenError,
    compilation_frozen,
    freeze_compilation,
    freeze_kernel_resolution,
    kernel_resolution_frozen,
    unfreeze_compilation,
    unfreeze_kernel_resolution,
)
from . import cute, gemm, integration

apply_cutlass_runtime_patches()

__all__ = [
    "cute",
    "gemm",
    "integration",
    "KernelResolutionFrozenError",
    "compilation_frozen",
    "freeze_compilation",
    "freeze_kernel_resolution",
    "kernel_resolution_frozen",
    "unfreeze_compilation",
    "unfreeze_kernel_resolution",
]
