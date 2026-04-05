"""Opt-in profiling helpers for b12x and serve runtime code."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import ContextManager

from torch.profiler import record_function as _torch_record_function

_PROFILE_RANGE_ENV = "B12X_ENABLE_PROFILE_RANGES"
_ENABLED_VALUES = {"1", "true", "yes", "on"}
_PROFILE_RANGES_ENABLED = (
    os.environ.get(_PROFILE_RANGE_ENV, "").strip().lower() in _ENABLED_VALUES
)


def profile_ranges_enabled() -> bool:
    """Return whether named runtime profile ranges are enabled."""
    return _PROFILE_RANGES_ENABLED


def record_function(name: str) -> ContextManager[None]:
    """Return a named profiler range only when explicitly enabled."""
    if not _PROFILE_RANGES_ENABLED:
        return nullcontext()
    return _torch_record_function(name)
