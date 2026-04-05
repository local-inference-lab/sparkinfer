"""Tests for serving-engine configuration policy helpers."""

from serve.engine.serving import (
    _should_enable_layer_compile,
    _should_enable_prefix_cache,
)


def test_prefix_cache_enabled_for_non_hybrid_models():
    assert _should_enable_prefix_cache(
        is_hybrid=False,
        has_state_snapshot_slots=False,
    )


def test_prefix_cache_enabled_for_single_gpu_hybrid_with_snapshots():
    assert _should_enable_prefix_cache(
        is_hybrid=True,
        has_state_snapshot_slots=True,
    )


def test_prefix_cache_disabled_for_single_gpu_hybrid_without_snapshots():
    assert not _should_enable_prefix_cache(
        is_hybrid=True,
        has_state_snapshot_slots=False,
    )


def test_prefix_cache_enabled_for_tp_hybrid_with_snapshots():
    assert _should_enable_prefix_cache(
        is_hybrid=True,
        has_state_snapshot_slots=True,
    )


def test_layer_compile_disabled_when_explicitly_disabled():
    assert not _should_enable_layer_compile(
        is_hybrid=False,
        compile_layers=False,
    )


def test_layer_compile_enabled_for_hybrid_models_when_requested():
    assert _should_enable_layer_compile(
        is_hybrid=True,
        compile_layers=True,
    )


def test_layer_compile_enabled_only_when_requested_for_non_hybrid():
    assert _should_enable_layer_compile(
        is_hybrid=False,
        compile_layers=True,
    )
