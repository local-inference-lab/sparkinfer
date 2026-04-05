"""Tests for opt-in runtime profiling ranges."""

from __future__ import annotations

import importlib
from contextlib import nullcontext

import b12x.profiling as profiling


class _SentinelRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


def _reload_profiling():
    return importlib.reload(profiling)


def test_profile_ranges_disabled_by_default(monkeypatch):
    monkeypatch.delenv("B12X_ENABLE_PROFILE_RANGES", raising=False)
    module = _reload_profiling()

    assert module.profile_ranges_enabled() is False
    assert type(module.record_function("disabled")) is nullcontext


def test_profile_ranges_require_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("B12X_ENABLE_PROFILE_RANGES", "1")
    module = _reload_profiling()
    monkeypatch.setattr(module, "_torch_record_function", lambda name: _SentinelRange(name))

    assert module.profile_ranges_enabled() is True
    ctx = module.record_function("enabled")
    assert isinstance(ctx, _SentinelRange)
    assert ctx.name == "enabled"

    monkeypatch.delenv("B12X_ENABLE_PROFILE_RANGES", raising=False)
    _reload_profiling()
