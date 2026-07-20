#!/usr/bin/env python3
"""Offline fail-closed checks for the shared exact-cache ABBA timer.

This exercises benchmark control flow and artifact validation only. It is not
a CPU kernel test and cannot satisfy any CUTLASS migration acceptance gate;
release evidence still comes exclusively from physical GPUs 4 and 5.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import sparkinfer.cute.compiler as cute_compiler
from validation.cutlass_migration.core.exact_cache_abba import (
    _PreparedAbbaEventPool,
    _balanced_duration_precondition,
    _prepare_abba_event_pool,
    _validate_timing_mode_stability,
    load_exact,
    time_abba,
    time_conditions,
    validate_time_abba_aggregation,
)
from validation.cutlass_migration.integrity_checks.release_aggregate import _timing


_LABELS = ("cutlass-4.5.2", "cutlass-4.6.0")


def _expect_failure(
    callback: object,
    error_type: type[Exception],
    message: str,
) -> None:
    if not callable(callback):
        raise AssertionError("negative-test callback is not callable")
    try:
        callback()
    except error_type as error:
        if message not in str(error):
            raise AssertionError(
                f"expected failure containing {message!r}, got {str(error)!r}"
            ) from error
    else:
        raise AssertionError(f"invalid timer contract passed: {message}")


def _mode_snapshot(
    captured_ns: int,
    *,
    pstate: str = "P1",
    sm_clock: int = 2300,
    throttle_reasons: str = "0x0000000000000000",
) -> dict[str, object]:
    return {
        "available": True,
        "captured_unix_ns": captured_ns,
        "fields": {
            "index": "4",
            "uuid": "GPU-static-selftest",
            "pstate": pstate,
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
            "power.limit": "600.00 W",
            "clocks_throttle_reasons.active": throttle_reasons,
            "clocks.current.sm": f"{sm_clock} MHz",
            "clocks.current.memory": "13365 MHz",
        },
    }


class _NoOpGraph:
    def replay(self) -> None:
        pass


class _NoOpStream:
    def synchronize(self) -> None:
        pass


def _validate_exact_load_staging() -> None:
    key = "0" * 64
    spec_hash = "1" * 64
    object_bytes = b"immutable-exact-cache-object"
    with tempfile.TemporaryDirectory(prefix="sparkinfer-exact-cache-selftest-") as raw:
        cache = Path(raw) / "cache"
        shard = cache / key[:2]
        shard.mkdir(parents=True)
        object_path = shard / f"{key}.o"
        manifest_path = shard / f"{key}.json"
        object_path.write_bytes(object_bytes)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": "sparkinfer.cute.compile_manifest.v3",
                    "cache_key": key,
                    "compile_spec_hash": spec_hash,
                    "object_sha256": hashlib.sha256(object_bytes).hexdigest(),
                    "object_bytes": len(object_bytes),
                    "compile_spec_json": "{}",
                    "semantic_key": "2" * 64,
                    "kernel_id": "integrity.fake",
                    "package_fingerprint": "3" * 64,
                    "toolchain": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        staged: dict[str, object] = {}

        def fake_load(observed_key: str) -> object:
            stage_cache = Path(os.environ["SPARKINFER_CUTE_COMPILE_CACHE_DIR"])
            stage_object = stage_cache / observed_key[:2] / f"{observed_key}.o"
            staged["separate_cache"] = stage_cache != cache
            staged["before"] = stage_object.read_bytes()
            stage_object.write_bytes(stage_object.read_bytes() + b":loader-patch")
            staged["after"] = stage_object.read_bytes()
            return object()

        previous_cache = os.environ.get("SPARKINFER_CUTE_COMPILE_CACHE_DIR")
        with patch(
            "validation.cutlass_migration.core.exact_cache_abba."
            "cute_compiler._load_cute_compile_from_disk",
            fake_load,
        ):
            compiled, provenance = load_exact(cache, spec_hash)
        restored_cache = os.environ.get("SPARKINFER_CUTE_COMPILE_CACHE_DIR")

        if compiled is None or provenance["object_bytes"] != len(object_bytes):
            raise AssertionError("staged exact-cache load did not return provenance")
        if object_path.read_bytes() != object_bytes:
            raise AssertionError("exact-cache source object changed during staged load")
        if not (
            staged.get("separate_cache") is True
            and staged.get("before") == object_bytes
            and staged.get("after") == object_bytes + b":loader-patch"
        ):
            raise AssertionError(
                "exact-cache loader did not mutate only its staging copy"
            )
        if restored_cache != previous_cache:
            raise AssertionError(
                "exact-cache staged load did not restore cache environment"
            )


def _validate_compiler_disk_load_staging() -> None:
    """Prove every compiler disk hit protects its content-addressed object."""

    key = "4" * 64
    object_bytes = b"immutable-compiler-cache-object"
    with tempfile.TemporaryDirectory(prefix="sparkinfer-compiler-cache-selftest-") as raw:
        cache = Path(raw) / "cache"
        object_path = cache / key[:2] / f"{key}.o"
        object_path.parent.mkdir(parents=True)
        object_path.write_bytes(object_bytes)
        observed: dict[str, object] = {}
        sentinel = object()

        class _PatchingExternalBinaryModule:
            def __init__(self, path: str) -> None:
                staged_object = Path(path)
                observed["path"] = staged_object
                observed["separate_object"] = staged_object != object_path
                observed["before"] = staged_object.read_bytes()
                staged_object.write_bytes(staged_object.read_bytes() + b":loader-patch")
                observed["after"] = staged_object.read_bytes()

            def __getattr__(self, _name: str) -> object:
                return sentinel

        previous_cache = os.environ.get("SPARKINFER_CUTE_COMPILE_CACHE_DIR")
        os.environ["SPARKINFER_CUTE_COMPILE_CACHE_DIR"] = str(cache)
        try:
            with patch(
                "cutlass.base_dsl.export.external_binary_module.ExternalBinaryModule",
                _PatchingExternalBinaryModule,
            ):
                compiled = cute_compiler._load_cute_compile_from_disk(key)
        finally:
            if previous_cache is None:
                os.environ.pop("SPARKINFER_CUTE_COMPILE_CACHE_DIR", None)
            else:
                os.environ["SPARKINFER_CUTE_COMPILE_CACHE_DIR"] = previous_cache

        staged_path = observed.get("path")
        if compiled is not sentinel:
            raise AssertionError("compiler disk load did not return staged module")
        if object_path.read_bytes() != object_bytes:
            raise AssertionError("compiler disk load mutated its canonical object")
        if not (
            observed.get("separate_object") is True
            and observed.get("before") == object_bytes
            and observed.get("after") == object_bytes + b":loader-patch"
            and isinstance(staged_path, Path)
            and not staged_path.exists()
        ):
            raise AssertionError("compiler disk load did not isolate loader mutation")


def _validate_event_pool_contract() -> None:
    for replays in (1, 2):
        if not validate_time_abba_aggregation(_timing(replays=replays), labels=_LABELS):
            raise AssertionError(f"valid K={replays} aggregate timing did not pass")
    for initialized in (None, False):
        mutated = deepcopy(_timing(replays=2))
        if initialized is None:
            mutated["event_pool"].pop("initialized_before_target_graph_preconditioning")
        else:
            mutated["event_pool"]["initialized_before_target_graph_preconditioning"] = (
                initialized
            )
        _expect_failure(
            lambda mutated=mutated: validate_time_abba_aggregation(
                mutated, labels=_LABELS
            ),
            AssertionError,
            "event-pool policy is inconsistent",
        )


def _validate_timer_inputs() -> None:
    for cycles in (0, 1, 3, True):
        _expect_failure(
            lambda cycles=cycles: _prepare_abba_event_pool(
                cycles=cycles,
                event_batch_cycles=1,
                replays_per_reported_sample=1,
                stream=None,  # rejected before any CUDA operation
            ),
            ValueError,
            "cycles must",
        )
    for event_batch_cycles in (0, -1, True, 1.5):
        _expect_failure(
            lambda event_batch_cycles=event_batch_cycles: _prepare_abba_event_pool(
                cycles=2,
                event_batch_cycles=event_batch_cycles,
                replays_per_reported_sample=1,
                stream=None,  # rejected before any CUDA operation
            ),
            ValueError,
            "event_batch_cycles must be a positive integer",
        )
    mismatched_pool = _PreparedAbbaEventPool(
        event_pairs=[],
        metadata={},
        cycles=2,
        event_batch_cycles=1,
        replays_per_reported_sample=1,
    )
    _expect_failure(
        lambda: time_abba(
            {},
            labels=("a", "b"),
            cycles=4,
            event_batch_cycles=1,
            stream=None,
            flush=None,
            prepared_event_pool=mismatched_pool,
        ),
        ValueError,
        "does not match timing parameters",
    )


def _validate_required_release_controls() -> None:
    signature = inspect.signature(time_conditions)
    required_controls = (
        "precondition_seconds",
        "maximum_precondition_seconds",
        "mode_snapshot",
        "required_pstate",
        "max_sm_clock_delta_mhz",
    )
    for name in required_controls:
        if signature.parameters[name].default is not inspect.Parameter.empty:
            raise AssertionError(f"time_conditions.{name} must not have a default")
    throttle_control = signature.parameters["required_active_throttle_reasons"]
    if throttle_control.default != 0:
        raise AssertionError(
            "time_conditions.required_active_throttle_reasons must default to zero"
        )

    base = {
        "precondition_seconds": 5.0,
        "maximum_precondition_seconds": 30.0,
        "mode_snapshot": lambda: _mode_snapshot(1_000),
        "required_pstate": "P1",
        "max_sm_clock_delta_mhz": 60.0,
    }
    mutations = (
        ("precondition_seconds", 4.999, "at least 5 seconds"),
        ("maximum_precondition_seconds", 60.001, "at most 60"),
        ("mode_snapshot", None, "physical-GPU callback"),
        ("required_pstate", "P8", "must be P1"),
        ("max_sm_clock_delta_mhz", 0.0, "must be in (0, 60]"),
        (
            "required_active_throttle_reasons",
            0x20,
            "must be exactly 0 or 0x4",
        ),
        (
            "required_active_throttle_reasons",
            0x80,
            "must be exactly 0 or 0x4",
        ),
    )
    for field, value, expected in mutations:
        controls = {**base, field: value}
        _expect_failure(
            lambda controls=controls: time_conditions(
                {"a": _NoOpGraph(), "b": _NoOpGraph()},
                labels=("a", "b"),
                precondition=1,
                warmup=1,
                cycles=2,
                event_batch_cycles=1,
                stream=None,  # validation fails before any CUDA operation
                cold_l2=False,
                l2_flush_bytes=0,
                **controls,
            ),
            ValueError,
            expected,
        )


def _validate_preconditioning_limits() -> None:
    graphs = {"a": _NoOpGraph(), "b": _NoOpGraph()}
    with (
        patch(
            "validation.cutlass_migration.core.exact_cache_abba.torch.cuda.stream",
            lambda _stream: nullcontext(),
        ),
        patch(
            "validation.cutlass_migration.core.exact_cache_abba.time.monotonic",
            side_effect=(0.0, 1.1),
        ),
    ):
        _expect_failure(
            lambda: _balanced_duration_precondition(
                graphs,
                labels=("a", "b"),
                minimum_cycles=0,
                minimum_seconds=0.0,
                maximum_seconds=1.0,
                stream=_NoOpStream(),
                flush=None,
                mode_snapshot=None,
                required_pstate=None,
            ),
            RuntimeError,
            "exceeded its maximum",
        )

    for bad_probe in (
        _mode_snapshot(1_000, pstate="P8"),
        _mode_snapshot(1_000, throttle_reasons="0x4"),
    ):
        with (
            patch(
                "validation.cutlass_migration.core.exact_cache_abba.torch.cuda.stream",
                lambda _stream: nullcontext(),
            ),
            patch(
                "validation.cutlass_migration.core.exact_cache_abba.time.monotonic",
                side_effect=(0.0, 0.4, 0.4, 1.1),
            ),
        ):
            _expect_failure(
                lambda bad_probe=bad_probe: _balanced_duration_precondition(
                    graphs,
                    labels=("a", "b"),
                    minimum_cycles=0,
                    minimum_seconds=0.0,
                    maximum_seconds=1.0,
                    stream=_NoOpStream(),
                    flush=None,
                    mode_snapshot=lambda: bad_probe,
                    required_pstate="P1",
                ),
                RuntimeError,
                "exceeded its maximum",
            )

    with (
        patch(
            "validation.cutlass_migration.core.exact_cache_abba.torch.cuda.stream",
            lambda _stream: nullcontext(),
        ),
        patch(
            "validation.cutlass_migration.core.exact_cache_abba.time.monotonic",
            side_effect=(0.0, 0.1),
        ),
    ):
        result, probe = _balanced_duration_precondition(
            graphs,
            labels=("a", "b"),
            minimum_cycles=0,
            minimum_seconds=0.0,
            maximum_seconds=1.0,
            stream=_NoOpStream(),
            flush=None,
            mode_snapshot=lambda: _mode_snapshot(1_000, throttle_reasons="0x4"),
            required_pstate="P1",
            required_active_throttle_reasons=0x4,
        )
    if (
        result["required_active_throttle_reasons"] != 0x4
        or probe is None
        or probe["fields"]["clocks_throttle_reasons.active"] != "0x4"
    ):
        raise AssertionError("explicit SW-power-cap preconditioning was not retained")

    replay_count = [0]

    class _CountingGraph:
        def replay(self) -> None:
            replay_count[0] += 1

    adaptive_graphs = {"a": _CountingGraph(), "b": _CountingGraph()}
    with (
        patch(
            "validation.cutlass_migration.core.exact_cache_abba.torch.cuda.stream",
            lambda _stream: nullcontext(),
        ),
        patch(
            "validation.cutlass_migration.core.exact_cache_abba.time.monotonic",
            lambda: replay_count[0] * 0.005,
        ),
    ):
        adaptive, _ = _balanced_duration_precondition(
            adaptive_graphs,
            labels=("a", "b"),
            minimum_cycles=1,
            minimum_seconds=5.0,
            maximum_seconds=30.0,
            stream=_NoOpStream(),
            flush=None,
            mode_snapshot=lambda: _mode_snapshot(1_000),
            required_pstate="P1",
        )
    if not (
        5.0 <= adaptive["observed_active_seconds"] <= 30.0
        and max(adaptive["batch_cycle_counts"]) < 1024
        and sum(adaptive["batch_cycle_counts"]) == adaptive["completed_cycles"]
    ):
        raise AssertionError("adaptive long-graph preconditioning changed")


def _validate_mode_envelope() -> None:
    before = _mode_snapshot(1_000)
    after = _mode_snapshot(2_000)
    result = _validate_timing_mode_stability(
        before,
        after,
        required_pstate="P1",
        required_active_throttle_reasons=0,
        max_sm_clock_delta_mhz=60.0,
    )
    if (
        result["observed_sm_clock_delta_mhz"] != 0.0
        or result["required_active_throttle_reasons"] != 0
    ):
        raise AssertionError("zero clock delta/unthrottled positive case changed")

    sw_power_cap = _validate_timing_mode_stability(
        _mode_snapshot(1_000, throttle_reasons="0x4"),
        _mode_snapshot(2_000, throttle_reasons="0x0000000000000004"),
        required_pstate="P1",
        required_active_throttle_reasons=0x4,
        max_sm_clock_delta_mhz=60.0,
    )
    if (
        sw_power_cap["required_active_throttle_reasons"] != 0x4
        or sw_power_cap["observed_before_active_throttle_reasons"] != 0x4
        or sw_power_cap["observed_after_active_throttle_reasons"] != 0x4
    ):
        raise AssertionError("explicit stable SW-power-cap evidence was not retained")

    sw_power_cap_transition = _validate_timing_mode_stability(
        _mode_snapshot(1_000, throttle_reasons="0x0"),
        _mode_snapshot(2_000, throttle_reasons="0x4"),
        required_pstate="P1",
        required_active_throttle_reasons=0,
        max_sm_clock_delta_mhz=60.0,
        allow_sw_power_cap_transition=True,
    )
    if not (
        sw_power_cap_transition["allow_sw_power_cap_transition"] is True
        and sw_power_cap_transition["active_throttle_reasons_transition_observed"]
        is True
        and sw_power_cap_transition["allowed_observed_active_throttle_reasons"]
        == [0, 0x4]
    ):
        raise AssertionError("diagnostic SW-power-cap transition was not retained")

    _expect_failure(
        lambda: _validate_timing_mode_stability(
            _mode_snapshot(1_000, pstate="P8"),
            _mode_snapshot(2_000, pstate="P8"),
            required_pstate="P1",
            required_active_throttle_reasons=0,
            max_sm_clock_delta_mhz=60.0,
        ),
        AssertionError,
        "requires stable P1",
    )
    for location in ("before", "after"):
        throttled_before = deepcopy(before)
        throttled_after = deepcopy(after)
        target = throttled_before if location == "before" else throttled_after
        target["fields"]["clocks_throttle_reasons.active"] = "0x4"
        _expect_failure(
            lambda throttled_before=throttled_before, throttled_after=throttled_after: (
                _validate_timing_mode_stability(
                    throttled_before,
                    throttled_after,
                    required_pstate="P1",
                    required_active_throttle_reasons=0,
                    max_sm_clock_delta_mhz=60.0,
                )
            ),
            AssertionError,
            "exact active clock-throttle reasons mask 0x0",
        )

    for unsupported_mask in (0x1, 0x20, 0x80, 0x84):
        _expect_failure(
            lambda unsupported_mask=unsupported_mask: (
                _validate_timing_mode_stability(
                    _mode_snapshot(1_000, throttle_reasons=f"{unsupported_mask:#x}"),
                    _mode_snapshot(2_000, throttle_reasons=f"{unsupported_mask:#x}"),
                    required_pstate="P1",
                    required_active_throttle_reasons=unsupported_mask,
                    max_sm_clock_delta_mhz=60.0,
                )
            ),
            ValueError,
            "must be exactly 0 or 0x4",
        )

    _expect_failure(
        lambda: _validate_timing_mode_stability(
            _mode_snapshot(1_000, throttle_reasons="0x4"),
            _mode_snapshot(2_000, throttle_reasons="0x0"),
            required_pstate="P1",
            required_active_throttle_reasons=0x4,
            max_sm_clock_delta_mhz=60.0,
        ),
        AssertionError,
        "exact active clock-throttle reasons mask 0x4",
    )


def main() -> int:
    _validate_compiler_disk_load_staging()
    _validate_exact_load_staging()
    _validate_event_pool_contract()
    _validate_timer_inputs()
    _validate_required_release_controls()
    _validate_preconditioning_limits()
    _validate_mode_envelope()
    print(
        "status=pass positive=compiler-staged-immutable-load,"
        "exact-cache-staged-immutable-load,aggregate-k1-k2,"
        "zero-clock-delta,zero-throttle,"
        "explicit-stable-sw-power-cap,diagnostic-sw-power-cap-transition "
        "negative=event-pool-order,cycles,event-batch-cycles,pool-mismatch,"
        "required-duration-pstate-mode-clock-controls,"
        "precondition-overshoot,p8,throttle-before,throttle-after,"
        "precondition-probe,thermal,hw-power-brake,composite-mask,asymmetric-mask"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
