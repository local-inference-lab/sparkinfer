from __future__ import annotations

from functools import wraps

_COMPILE_ONLY_CACHE_WARNING = "Cache is disabled as user wants to compile only."
_PATCHED = False


def apply_cutlass_runtime_patches() -> None:
    global _PATCHED
    if _PATCHED:
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
    _PATCHED = True
