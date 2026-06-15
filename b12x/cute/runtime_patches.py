from __future__ import annotations

import os
from functools import wraps

_COMPILE_ONLY_CACHE_WARNING = "Cache is disabled as user wants to compile only."
_WARNING_PATCHED = False
_MEMORY_DEBUG_PATCHED = False
_MEMORY_DEBUG_SNAPSHOT = {
    "free": None,
    "total": None,
    "used": None,
    "torch_allocated": None,
    "torch_reserved": None,
    "external": None,
    "device": None,
}


def apply_cutlass_runtime_patches() -> None:
    _apply_compile_only_warning_patch()
    _apply_memory_debug_patch()


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _apply_compile_only_warning_patch() -> None:
    global _WARNING_PATCHED
    if _WARNING_PATCHED:
        return

    try:
        from cutlass.base_dsl.dsl import BaseDSL
    except Exception:
        return

    original_print_warning = BaseDSL.print_warning
    original_print_warning_once = BaseDSL.print_warning_once

    @wraps(original_print_warning)
    def patched_print_warning(self, message):
        if message == _COMPILE_ONLY_CACHE_WARNING:
            return None
        return original_print_warning(self, message)

    @wraps(original_print_warning_once)
    def patched_print_warning_once(self, message):
        if message == _COMPILE_ONLY_CACHE_WARNING:
            return None
        return original_print_warning_once(self, message)

    BaseDSL.print_warning = patched_print_warning
    BaseDSL.print_warning_once = patched_print_warning_once
    _WARNING_PATCHED = True


def _apply_memory_debug_patch() -> None:
    global _MEMORY_DEBUG_PATCHED
    if _MEMORY_DEBUG_PATCHED:
        return
    if _env_flag("CUTLASS_DSL_CUDA_MEMORY_DEBUG", default=False):
        return

    try:
        from cutlass.base_dsl.runtime import cuda as cuda_helpers
    except Exception:
        return
    if not hasattr(cuda_helpers, "_memory_debug_snapshot") or not hasattr(
        cuda_helpers, "_memory_debug_log"
    ):
        _MEMORY_DEBUG_PATCHED = True
        return
    if getattr(cuda_helpers, "_b12x_memory_debug_patched", False):
        _MEMORY_DEBUG_PATCHED = True
        return

    if not hasattr(cuda_helpers, "_b12x_original_memory_debug_snapshot"):
        cuda_helpers._b12x_original_memory_debug_snapshot = (
            cuda_helpers._memory_debug_snapshot
        )
    if not hasattr(cuda_helpers, "_b12x_original_memory_debug_log"):
        cuda_helpers._b12x_original_memory_debug_log = cuda_helpers._memory_debug_log

    def _empty_memory_debug_snapshot() -> dict[str, int | None]:
        return dict(_MEMORY_DEBUG_SNAPSHOT)

    def _empty_memory_debug_log(
        label: str, before: dict[str, int | None] | None = None
    ) -> None:
        return None

    cuda_helpers._memory_debug_snapshot = _empty_memory_debug_snapshot
    cuda_helpers._memory_debug_log = _empty_memory_debug_log
    cuda_helpers._b12x_memory_debug_patched = True
    _MEMORY_DEBUG_PATCHED = True
