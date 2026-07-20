#!/usr/bin/env python3
"""Offline fail-closed checks for aggregate ABBA release evidence.

This is a manual evidence-tool self-test, not a CPU kernel test.  It verifies
that the release index reconstructs every timing statistic from nested CUDA
event samples and rejects incomplete or diagnostic provenance.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Callable

from validation.cutlass_migration.acceptance.e2e.release_index import (
    ReleaseValidationError,
    _SUPPORTED_ABBA_SCHEMAS,
    _recompute_abba_summary,
    _scan_abba_paths,
    _validate_aggregated_timing,
    _validate_artifact_timing_mode_policy,
    _validate_evidence_status,
    _validate_smem_contract_finalization,
)


def _timing(*, replays: int) -> dict[str, object]:
    labels = ("cutlass-4.5.2", "cutlass-4.6.0")
    orders = (
        (labels[0], labels[1], labels[1], labels[0]),
        (labels[1], labels[0], labels[0], labels[1]),
    )
    groups_per_position = 3
    raw = {
        f"{order_index}:{position}:{label}": [
            [
                10.0
                + order_index
                + position / 10.0
                + sample_index / 100.0
                + replay_index / 1000.0
                for replay_index in range(replays)
            ]
            for sample_index in range(groups_per_position)
        ]
        for order_index, order in enumerate(orders)
        for position, label in enumerate(order)
    }
    position_samples = {
        key: [sum(group) / len(group) for group in groups]
        for key, groups in raw.items()
    }
    by_label = {label: [] for label in labels}
    for cycle in range(2 * groups_per_position):
        order_index = cycle & 1
        order = orders[order_index]
        sample_index = cycle // 2
        for position, label in enumerate(order):
            key = f"{order_index}:{position}:{label}"
            by_label[label].append(position_samples[key][sample_index])
    summaries = {
        label: _recompute_abba_summary(samples) for label, samples in by_label.items()
    }
    total_cycles = 2 * groups_per_position
    event_batch_cycles = 4
    batch_cycle_capacity = min(total_cycles, event_batch_cycles)
    pair_count = batch_cycle_capacity * len(orders[0]) * replays
    batch_count = (total_cycles + event_batch_cycles - 1) // event_batch_cycles
    return {
        "orders": orders,
        "cold_l2": False,
        "replays_per_reported_sample": replays,
        "aggregation": {
            "reported_sample": "arithmetic_mean_us",
            "inner_event_bracketing": "independent_per_graph_replay",
            "inner_schedule": "full_abba_order_per_repetition",
            "flush_before_every_inner_replay": False,
            "flush_inside_timed_interval": False,
        },
        "event_pool": {
            "schema": "sparkinfer.cuda_event_pool.v1",
            "allocation_phase": "before_reported_samples",
            "prewarm_phase": "before_reported_samples",
            "prewarm_each_event": True,
            "one_pair_per_inner_replay": True,
            "event_creation_inside_sample_schedule": False,
            "reuse_boundary": "after_stream_synchronize_and_elapsed_query",
            "initialized_before_target_graph_preconditioning": True,
            "event_batch_cycles": event_batch_cycles,
            "batch_cycle_capacity": batch_cycle_capacity,
            "pair_count": pair_count,
            "event_count": 2 * pair_count,
            "unique_event_handle_count": 2 * pair_count,
            "event_handle_sha256": "a" * 64,
            "prewarm_elapsed_query_count": pair_count,
            "prewarm_elapsed_sha256": "b" * 64,
            "batch_count": batch_count,
            "reuse_count": max(0, batch_count - 1),
        },
        "inner_samples_by_position": raw,
        "inner_sample_count_by_label": {
            label: len(samples) * replays for label, samples in by_label.items()
        },
        "summaries": summaries,
        "position_summaries": {
            key: _recompute_abba_summary(samples)
            for key, samples in position_samples.items()
        },
    }


_GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _smem_allocator_row(
    *, source_category: str, path: str, method: str
) -> dict[str, object]:
    tensor = method == "allocate_tensor"
    return {
        "kind": "allocator",
        "source_category": source_category,
        "path": path,
        "line": 10,
        "column": 8,
        "scope": "Kernel.kernel",
        "allocator_name": "smem",
        "constructor_argument_count": 0,
        "constructor_keyword_count": 0,
        "allocator_store_count": 1,
        "allocator_store_lines": [10],
        "allocation_count": 1,
        "allocation_methods": [method],
        "allocation_lines": [11],
        "allocation_argument_counts": [0 if tensor else 1],
        "allocation_keyword_counts": [3 if tensor else 0],
        "allocation_argument_sources": ["synthetic"],
        "allocation_result_names": ["storage"],
        "typed_allocation": True,
        "allocation_after_constructor": True,
        "allocation_result_bound_locally": True,
        "violations": [],
        "passed": True,
    }


def _smem_report() -> dict[str, object]:
    rows = [
        _smem_allocator_row(
            source_category="production",
            path="sparkinfer/production_kernel.py",
            method="allocate",
        ),
        _smem_allocator_row(
            source_category="infrastructure",
            path="tests/test_probe.py",
            method="allocate_tensor",
        ),
        {
            "kind": "private_memrange",
            "source_category": "production",
            "path": "sparkinfer/cute/smem.py",
            "line": 17,
            "column": 11,
            "scope": "",
            "identifier": "_MemRangeData",
            "centralized": True,
            "violations": [],
            "passed": True,
        },
    ]
    production = {
        "python_file_count": 2,
        "allocator_count": 1,
        "allocator_pass_count": 1,
        "allocator_fail_count": 0,
        "allocation_call_count": 1,
        "allocate_call_count": 1,
        "allocate_tensor_call_count": 0,
        "private_memrange_identifier_count": 1,
        "private_memrange_centralized_count": 1,
        "private_memrange_outside_count": 0,
        "parse_error_count": 0,
        "violation_count": 0,
        "row_count": 2,
    }
    infrastructure = {
        "python_file_count": 1,
        "allocator_count": 1,
        "allocator_pass_count": 1,
        "allocator_fail_count": 0,
        "allocation_call_count": 1,
        "allocate_call_count": 0,
        "allocate_tensor_call_count": 1,
        "private_memrange_identifier_count": 0,
        "private_memrange_centralized_count": 0,
        "private_memrange_outside_count": 0,
        "parse_error_count": 0,
        "violation_count": 0,
        "row_count": 1,
    }
    counts = {field: production[field] + infrastructure[field] for field in production}
    return {
        "schema": "sparkinfer.cute.smem_contracts.v1",
        "root": "/synthetic/source",
        "audited_source_roots": {
            "production": ["sparkinfer"],
            "infrastructure": ["benchmarks", "tests", "validation"],
        },
        "central_private_memrange_path": "sparkinfer/cute/smem.py",
        "rows": rows,
        "counts": counts,
        "source_counts": {
            "production": production,
            "infrastructure": infrastructure,
        },
        "passed": True,
    }


def _write_smem_fixture(
    root: Path,
) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    root.mkdir(parents=True)
    report = _smem_report()
    smem_path = root / "smem-contracts.final.json"
    smem_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_sha = hashlib.sha256(smem_path.read_bytes()).hexdigest()
    auditor_sha = "c" * 64
    gate = {
        "schema": "sparkinfer.cute.migration.smem_contract_gate.v1",
        "passed": True,
        "auditor": {
            "path": "validation/cutlass_migration/evidence/smem_contracts.py",
            "sha256": auditor_sha,
        },
        "report_schema": "sparkinfer.cute.smem_contracts.v1",
        "report_sha256": report_sha,
        "counts": report["counts"],
        "report": report,
    }
    artifact = {
        "path": str(smem_path),
        "sha256": report_sha,
        "schema": "sparkinfer.cute.smem_contracts.v1",
    }
    trace = {
        "source_root": "/synthetic/source",
        "static_validation": {
            "hashes": {"smem_contract_auditor_sha256": auditor_sha},
            "smem_contracts": gate,
        },
        "smem_contracts": {
            "schema": "sparkinfer.cute.migration.smem_contract_finalization.v1",
            "passed": True,
            "static_final_reports_equal": True,
            "gate": gate,
            "artifact": artifact,
        },
        "artifacts": {"smem_contracts": artifact},
    }
    manifest = {
        "inputs": {
            "smem_contract_auditor": {
                "path": "validation/cutlass_migration/evidence/smem_contracts.py",
                "sha256": auditor_sha,
                "size_bytes": 1234,
            }
        }
    }
    return root / "case-trace.json", trace, manifest, report


def _refresh_smem_fixture(
    trace_path: Path,
    trace: dict[str, object],
    report: dict[str, object],
) -> None:
    smem_path = trace_path.parent / "smem-contracts.final.json"
    smem_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(smem_path.read_bytes()).hexdigest()
    gate = trace["smem_contracts"]["gate"]
    gate["report_sha256"] = digest
    gate["counts"] = report["counts"]
    artifact = trace["smem_contracts"]["artifact"]
    artifact["sha256"] = digest


def _mode_snapshot(
    *,
    captured_ns: int,
    pstate: str = "P1",
    sm_clock: int = 2300,
    memory_clock: int = 13365,
    active_throttle_reasons: int = 0,
) -> dict[str, object]:
    return {
        "available": True,
        "captured_unix_ns": captured_ns,
        "torch_uuid": _GPU_UUID,
        "nvidia_smi_uuid": _GPU_UUID,
        "fields": {
            "index": "4",
            "uuid": _GPU_UUID,
            "pstate": pstate,
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
            "clocks.current.sm": f"{sm_clock} MHz",
            "clocks.current.memory": f"{memory_clock} MHz",
            "clocks_throttle_reasons.active": (f"0x{active_throttle_reasons:016x}"),
            "power.draw": "200.00 W",
            "power.limit": "600.00 W",
            "temperature.gpu": "55",
        },
    }


def _condition(
    timing: dict[str, object],
    *,
    required_active_throttle_reasons: int = 0,
) -> dict[str, object]:
    summaries = timing["summaries"]
    if not isinstance(summaries, dict):
        raise AssertionError("synthetic summaries changed type")
    before = _mode_snapshot(
        captured_ns=2_000,
        sm_clock=2300,
        active_throttle_reasons=required_active_throttle_reasons,
    )
    after = _mode_snapshot(
        captured_ns=3_000,
        sm_clock=2320,
        active_throttle_reasons=required_active_throttle_reasons,
    )
    return {
        "cold_l2": False,
        "l2_flush_bytes": 0,
        "preconditioning": {
            "policy": "balanced_abba_target_graph_duration",
            "minimum_cycles": 4,
            "minimum_active_seconds": 5.0,
            "maximum_active_seconds": 10.0,
            "completed_cycles": 4,
            "observed_active_seconds": 5.5,
            "target_graph_replays_by_label": {label: 8 for label in summaries},
            "cold_l2_flush_before_every_replay": False,
            "flush_inside_timed_interval": False,
            "required_pstate": "P1",
            "required_active_throttle_reasons": (required_active_throttle_reasons),
            "mode_probes": [
                _mode_snapshot(
                    captured_ns=1_000,
                    active_throttle_reasons=required_active_throttle_reasons,
                )
            ],
        },
        "allocator_before": {"allocated": 1024, "reserved": 2048},
        "allocator_after": {"allocated": 1024, "reserved": 2048},
        "allocator_stable": True,
        "gpu_mode_before_timing": before,
        "gpu_mode_after_timing": after,
        "gpu_mode_stability": {
            "schema": "sparkinfer.gpu_mode_stability.v1",
            "required_pstate": "P1",
            "required_memory_clock_equality": True,
            "required_active_throttle_reasons": required_active_throttle_reasons,
            "max_sm_clock_delta_mhz": 60.0,
            "observed_sm_clock_delta_mhz": 20.0,
            "observed_before_sm_clock_mhz": 2300.0,
            "observed_after_sm_clock_mhz": 2320.0,
            "observed_memory_clock_mhz": 13365.0,
            "observed_before_active_throttle_reasons": (
                required_active_throttle_reasons
            ),
            "observed_after_active_throttle_reasons": (
                required_active_throttle_reasons
            ),
            "stable_identity_and_mode_fields": [
                "index",
                "uuid",
                "persistence_mode",
                "compute_mode",
                "power.limit",
            ],
            "passed": True,
        },
        "timings": timing,
    }


def _timing_mode_policy(required_mask: int) -> dict[str, object]:
    return {
        "schema": "sparkinfer.gpu_timing_mode_policy.v1",
        "required_pstate": "P1",
        "required_active_throttle_reasons": required_mask,
        "active_throttle_reasons_match": "exact",
        "permitted_required_active_throttle_reasons": [0, 0x4],
        "required_memory_clock_equality": True,
        "max_sm_clock_delta_mhz": 60.0,
    }


def _expect_failure(
    name: str,
    callback: Callable[[], object],
    expected: str,
) -> None:
    try:
        callback()
    except ReleaseValidationError as exc:
        if expected not in str(exc):
            raise AssertionError(
                f"{name}: expected {expected!r}, got {str(exc)!r}"
            ) from exc
    else:
        raise AssertionError(f"{name}: invalid release evidence unexpectedly passed")


def _validate(
    timing: dict[str, object], condition: dict[str, object] | None = None
) -> int:
    summaries = timing["summaries"]
    if not isinstance(summaries, dict):
        raise AssertionError("synthetic summaries changed type")
    return _validate_aggregated_timing(
        Path("/synthetic/abba.json"),
        ("conditions", "warm_l2", "timings"),
        timing,
        summaries,
        condition=_condition(timing) if condition is None else condition,
        gpu=4,
        gpu_uuid=_GPU_UUID,
    )


def _selftest_smem_finalization() -> None:
    with tempfile.TemporaryDirectory(prefix="sparkinfer-release-smem-selftest-") as raw:
        base = Path(raw)

        trace_path, trace, manifest, _ = _write_smem_fixture(base / "positive")
        record = _validate_smem_contract_finalization(trace_path, trace, manifest)
        if record.get("schema") != "sparkinfer.cute.smem_contracts.v1":
            raise AssertionError("valid final SMEM artifact did not pass")

        trace_path, trace, manifest, _ = _write_smem_fixture(base / "missing")
        (trace_path.parent / "smem-contracts.final.json").unlink()
        _expect_failure(
            "missing final SMEM file",
            lambda: _validate_smem_contract_finalization(trace_path, trace, manifest),
            "lacks smem-contracts.final.json",
        )

        trace_path, trace, manifest, _ = _write_smem_fixture(base / "bytes")
        smem_path = trace_path.parent / "smem-contracts.final.json"
        smem_path.write_bytes(smem_path.read_bytes() + b" ")
        _expect_failure(
            "mutated final SMEM bytes",
            lambda: _validate_smem_contract_finalization(trace_path, trace, manifest),
            "artifact SHA-256 mismatch",
        )

        trace_path, trace, manifest, report = _write_smem_fixture(base / "counts")
        report["source_counts"]["production"]["allocator_count"] += 1
        report["counts"]["allocator_count"] += 1
        _refresh_smem_fixture(trace_path, trace, report)
        _expect_failure(
            "mutated final SMEM count",
            lambda: _validate_smem_contract_finalization(trace_path, trace, manifest),
            "production SMEM counts do not reconstruct from rows",
        )

        trace_path, trace, manifest, _ = _write_smem_fixture(base / "sha")
        trace["smem_contracts"]["artifact"]["sha256"] = "d" * 64
        _expect_failure(
            "mutated final SMEM SHA",
            lambda: _validate_smem_contract_finalization(trace_path, trace, manifest),
            "artifact SHA-256 mismatch",
        )

        trace_path, trace, manifest, report = _write_smem_fixture(base / "schema")
        report["schema"] = "sparkinfer.cute.smem_contracts.invalid"
        _refresh_smem_fixture(trace_path, trace, report)
        _expect_failure(
            "mutated final SMEM schema",
            lambda: _validate_smem_contract_finalization(trace_path, trace, manifest),
            "invalid final SMEM report schema/root/policy",
        )


def _selftest_bf16_schema_transition() -> None:
    old_schema = "sparkinfer.bf16_to_fp4_tma.cache_abba.v3"
    new_schema = "sparkinfer.bf16_to_fp4_tma.cache_abba.v4"
    if (
        old_schema in _SUPPORTED_ABBA_SCHEMAS
        or new_schema not in _SUPPORTED_ABBA_SCHEMAS
    ):
        raise AssertionError("BF16 aggregate schema transition is not fail-closed")
    with tempfile.TemporaryDirectory(prefix="sparkinfer-release-bf16-schema-") as raw:
        old_path = Path(raw) / "old-v3.json"
        old_path.write_text(json.dumps({"schema": old_schema}), encoding="utf-8")
        _expect_failure(
            "old BF16 custom timing schema",
            lambda: _scan_abba_paths([old_path]),
            "unsupported ABBA schema",
        )

    timing = _timing(replays=1)
    condition = _condition(timing)
    if _validate(timing, condition) != 1:
        raise AssertionError("new BF16 v4-style aggregate envelope did not pass")
    missing_envelope = deepcopy(condition)
    missing_envelope.pop("gpu_mode_stability")
    _expect_failure(
        "new BF16 aggregate envelope mutation",
        lambda: _validate(timing, missing_envelope),
        "GPU mode-stability policy",
    )


def _selftest_contiguous_schema_transition() -> None:
    old_schema = "sparkinfer.contiguous_attention.cache_abba.v1"
    new_schema = "sparkinfer.contiguous_attention.cache_abba.v2"
    if (
        old_schema in _SUPPORTED_ABBA_SCHEMAS
        or new_schema not in _SUPPORTED_ABBA_SCHEMAS
    ):
        raise AssertionError(
            "contiguous aggregate schema transition is not fail-closed"
        )
    with tempfile.TemporaryDirectory(prefix="sparkinfer-release-contiguous-schema-") as raw:
        old_path = Path(raw) / "old-v1.json"
        old_path.write_text(json.dumps({"schema": old_schema}), encoding="utf-8")
        _expect_failure(
            "old contiguous custom timing schema",
            lambda: _scan_abba_paths([old_path]),
            "unsupported ABBA schema",
        )


def _selftest_w4a8_schema_transition() -> None:
    old_schema = "sparkinfer.w4a8.dynamic.cache_abba.v1"
    new_schema = "sparkinfer.w4a8.dynamic.cache_abba.v2"
    if (
        old_schema in _SUPPORTED_ABBA_SCHEMAS
        or new_schema not in _SUPPORTED_ABBA_SCHEMAS
    ):
        raise AssertionError("W4A8 aggregate schema transition is not fail-closed")
    with tempfile.TemporaryDirectory(prefix="sparkinfer-release-w4a8-schema-") as raw:
        old_path = Path(raw) / "old-v1.json"
        old_path.write_text(json.dumps({"schema": old_schema}), encoding="utf-8")
        _expect_failure(
            "old W4A8 custom timing schema",
            lambda: _scan_abba_paths([old_path]),
            "unsupported ABBA schema",
        )


def _selftest_remaining_timer_schema_transitions() -> None:
    transitions = (
        (
            "MLA decode/merge",
            "sparkinfer.attention.mla.decode_merge.exact_cache_abba.v1",
            "sparkinfer.attention.mla.decode_merge.exact_cache_abba.v2",
        ),
        (
            "MLA prefill-MG",
            "sparkinfer.attention.mla.prefill_mg.exact_cache_abba.v3",
            "sparkinfer.attention.mla.prefill_mg.exact_cache_abba.v4",
        ),
        (
            "TP-MoE dynamic",
            "sparkinfer.tp_moe.dynamic.cache_abba.v1",
            "sparkinfer.tp_moe.dynamic.cache_abba.v2",
        ),
        (
            "W4A16 serving timing-mode policy",
            "sparkinfer.w4a16.serving.cache_abba.v1",
            "sparkinfer.w4a16.serving.cache_abba.v2",
        ),
    )
    for name, old_schema, new_schema in transitions:
        if (
            old_schema in _SUPPORTED_ABBA_SCHEMAS
            or new_schema not in _SUPPORTED_ABBA_SCHEMAS
        ):
            raise AssertionError(f"{name} schema transition is not fail-closed")
        with tempfile.TemporaryDirectory(prefix="sparkinfer-release-timer-schema-") as raw:
            old_path = Path(raw) / "old.json"
            old_path.write_text(json.dumps({"schema": old_schema}), encoding="utf-8")
            _expect_failure(
                f"old {name} custom timing schema",
                lambda old_path=old_path: _scan_abba_paths([old_path]),
                "unsupported ABBA schema",
            )


def main() -> int:
    timing = _timing(replays=2)
    timing_k1 = _timing(replays=1)
    zero_delta_condition = _condition(timing_k1)
    zero_delta_condition["gpu_mode_after_timing"]["fields"]["clocks.current.sm"] = (
        "2300 MHz"
    )
    zero_delta_condition["gpu_mode_stability"]["observed_sm_clock_delta_mhz"] = 0.0
    zero_delta_condition["gpu_mode_stability"]["observed_after_sm_clock_mhz"] = 2300.0
    sw_power_cap_condition = _condition(timing, required_active_throttle_reasons=0x4)
    if (
        _validate(timing) != 2
        or _validate(timing_k1, zero_delta_condition) != 1
        or _validate(timing, sw_power_cap_condition) != 2
    ):
        raise AssertionError("valid K=1/K=2 aggregate timing did not pass")

    policy_artifact = {
        "schema": "sparkinfer.w4a16.serving.cache_abba.v2",
        "timing_mode_policy": _timing_mode_policy(0x4),
    }
    validated_policy = _validate_artifact_timing_mode_policy(
        Path("/synthetic/w4-v2.json"),
        policy_artifact,
        [{"required_active_throttle_reasons": 0x4}],
    )
    if validated_policy["required_active_throttle_reasons"] != 0x4:
        raise AssertionError("top-level SW-power-cap policy was not retained")
    for name, artifact, rows, expected in (
        (
            "missing W4 timing policy",
            {"schema": "sparkinfer.w4a16.serving.cache_abba.v2"},
            [{"required_active_throttle_reasons": 0x4}],
            "lacks its top-level timing-mode policy",
        ),
        (
            "mismatched W4 timing policy",
            {
                "schema": "sparkinfer.w4a16.serving.cache_abba.v2",
                "timing_mode_policy": _timing_mode_policy(0),
            },
            [{"required_active_throttle_reasons": 0x4}],
            "does not match condition evidence",
        ),
        (
            "mixed W4 condition policies",
            policy_artifact,
            [
                {"required_active_throttle_reasons": 0},
                {"required_active_throttle_reasons": 0x4},
            ],
            "do not bind one consistent",
        ),
    ):
        _expect_failure(
            name,
            lambda artifact=artifact, rows=rows: (
                _validate_artifact_timing_mode_policy(
                    Path("/synthetic/w4-v2.json"), artifact, rows
                )
            ),
            expected,
        )

    incomplete = deepcopy(timing)
    incomplete.pop("aggregation")
    _expect_failure(
        "missing aggregate contract",
        lambda: _validate(incomplete),
        "complete aggregate timing contract even when K=1",
    )

    missing_position = deepcopy(timing)
    missing_key = next(iter(missing_position["inner_samples_by_position"]))
    missing_position["inner_samples_by_position"].pop(missing_key)
    missing_position["position_summaries"].pop(missing_key)
    _expect_failure(
        "missing ABBA position",
        lambda: _validate(missing_position),
        "all eight exact ABBA position keys",
    )

    order_bias = deepcopy(timing)
    biased_key = next(iter(order_bias["inner_samples_by_position"]))
    biased_groups = order_bias["inner_samples_by_position"][biased_key]
    biased_groups.pop()
    biased_samples = [sum(group) / len(group) for group in biased_groups]
    order_bias["position_summaries"][biased_key] = _recompute_abba_summary(
        biased_samples
    )
    _expect_failure(
        "unbalanced order counts",
        lambda: _validate(order_bias),
        "position/order counts are unbalanced",
    )

    bad_position_summary = deepcopy(timing)
    position_key = next(iter(bad_position_summary["position_summaries"]))
    bad_position_summary["position_summaries"][position_key]["mean_us"] += 1.0
    _expect_failure(
        "altered position summary",
        lambda: _validate(bad_position_summary),
        "mean_us does not reconstruct from inner events",
    )

    bad_arm_summary = deepcopy(timing)
    arm = next(iter(bad_arm_summary["summaries"]))
    bad_arm_summary["summaries"][arm]["median_us"] += 1.0
    _expect_failure(
        "altered arm summary",
        lambda: _validate(bad_arm_summary),
        "median_us does not reconstruct from inner events",
    )

    bad_orders = deepcopy(timing)
    bad_orders["orders"] = (bad_orders["orders"][0], bad_orders["orders"][0])
    _expect_failure(
        "altered ABBA order",
        lambda: _validate(bad_orders),
        "exact ABBA/BAAB orders",
    )

    missing_pool = deepcopy(timing)
    missing_pool.pop("event_pool")
    _expect_failure(
        "missing event pool",
        lambda: _validate(missing_pool),
        "complete aggregate timing contract even when K=1",
    )

    bad_pool_count = deepcopy(timing)
    bad_pool_count["event_pool"]["pair_count"] += 1
    _expect_failure(
        "event pool count",
        lambda: _validate(bad_pool_count),
        "event-pool counts are inconsistent",
    )

    bad_pool_digest = deepcopy(timing)
    bad_pool_digest["event_pool"]["event_handle_sha256"] = "not-a-digest"
    _expect_failure(
        "event pool digest",
        lambda: _validate(bad_pool_digest),
        "malformed event-pool digests",
    )

    for initialized in (None, False):
        bad_pool_order = deepcopy(timing)
        if initialized is None:
            bad_pool_order["event_pool"].pop(
                "initialized_before_target_graph_preconditioning"
            )
        else:
            bad_pool_order["event_pool"][
                "initialized_before_target_graph_preconditioning"
            ] = initialized
        _expect_failure(
            f"event pool ordering {initialized!r}",
            lambda bad_pool_order=bad_pool_order: _validate(bad_pool_order),
            "invalid CUDA event-pool provenance",
        )

    missing_precondition = _condition(timing)
    missing_precondition.pop("preconditioning")
    _expect_failure(
        "missing preconditioning",
        lambda: _validate(timing, missing_precondition),
        "balanced target-graph duration preconditioning",
    )

    tiny_duration = _condition(timing)
    tiny_duration["preconditioning"]["minimum_active_seconds"] = 1.0e-9
    _expect_failure(
        "tiny preconditioning duration",
        lambda: _validate(timing, tiny_duration),
        "active-duration envelope",
    )

    unbounded_duration = _condition(timing)
    unbounded_duration["preconditioning"]["maximum_active_seconds"] = 61.0
    _expect_failure(
        "unbounded preconditioning duration",
        lambda: _validate(timing, unbounded_duration),
        "active-duration envelope",
    )

    unbalanced_precondition = _condition(timing)
    replay_counts = unbalanced_precondition["preconditioning"][
        "target_graph_replays_by_label"
    ]
    replay_counts[next(iter(replay_counts))] += 1
    _expect_failure(
        "unbalanced preconditioning",
        lambda: _validate(timing, unbalanced_precondition),
        "preconditioning is not balanced",
    )

    p8_condition = _condition(timing)
    p8_condition["gpu_mode_before_timing"]["fields"]["pstate"] = "P8"
    _expect_failure(
        "P8 timing",
        lambda: _validate(timing, p8_condition),
        "timing did not remain in P1",
    )

    clock_condition = _condition(timing)
    clock_condition["gpu_mode_after_timing"]["fields"]["clocks.current.sm"] = "2400 MHz"
    clock_condition["gpu_mode_stability"]["observed_sm_clock_delta_mhz"] = 100.0
    clock_condition["gpu_mode_stability"]["observed_after_sm_clock_mhz"] = 2400.0
    _expect_failure(
        "unstable SM clock",
        lambda: _validate(timing, clock_condition),
        "timing clocks are not release-stable",
    )

    for throttle_location, throttle_mask in (
        ("before", 0x20),
        ("after", 0x80),
        ("probe", 0x84),
    ):
        throttled_condition = _condition(timing)
        if throttle_location == "before":
            snapshot = throttled_condition["gpu_mode_before_timing"]
        elif throttle_location == "after":
            snapshot = throttled_condition["gpu_mode_after_timing"]
        else:
            snapshot = throttled_condition["preconditioning"]["mode_probes"][-1]
        snapshot["fields"]["clocks_throttle_reasons.active"] = f"{throttle_mask:#x}"
        _expect_failure(
            f"active throttle reasons {throttle_location}",
            lambda throttled_condition=throttled_condition: _validate(
                timing, throttled_condition
            ),
            "does not match its exact requested active clock-throttle reasons mask",
        )

    for unsupported_mask in (0x20, 0x80, 0x84):
        unsupported_condition = _condition(
            timing, required_active_throttle_reasons=unsupported_mask
        )
        _expect_failure(
            f"unsupported requested throttle mask {unsupported_mask:#x}",
            lambda unsupported_condition=unsupported_condition: _validate(
                timing, unsupported_condition
            ),
            "must be exactly 0 or 0x4",
        )

    asymmetric_condition = _condition(timing, required_active_throttle_reasons=0x4)
    asymmetric_condition["gpu_mode_after_timing"]["fields"][
        "clocks_throttle_reasons.active"
    ] = "0x0"
    _expect_failure(
        "asymmetric explicit SW-power-cap state",
        lambda: _validate(timing, asymmetric_condition),
        "does not match its exact requested active clock-throttle reasons mask",
    )

    missing_allocator = _condition(timing)
    missing_allocator.pop("allocator_after")
    _expect_failure(
        "missing allocator proof",
        lambda: _validate(timing, missing_allocator),
        "stable timing allocator proof",
    )

    _validate_evidence_status(
        Path("/synthetic/final.json"), {"evidence_status": "final-source"}
    )
    for status in (None, "diagnostic-non-final"):
        artifact = {} if status is None else {"evidence_status": status}
        _expect_failure(
            f"evidence status {status!r}",
            lambda artifact=artifact: _validate_evidence_status(
                Path("/synthetic/non-final.json"), artifact
            ),
            "explicit final-source evidence status is required",
        )

    _selftest_smem_finalization()
    _selftest_bf16_schema_transition()
    _selftest_contiguous_schema_transition()
    _selftest_w4a8_schema_transition()
    _selftest_remaining_timer_schema_transitions()

    print(
        "status=pass positive=aggregate-k1-k2,explicit-final-source,"
        "explicit-stable-sw-power-cap,top-level-timing-policy "
        "negative=missing-contract,missing-position,order-bias,position-summary,"
        "arm-summary,orders,event-pool-count,event-pool-digest,event-pool-order,"
        "preconditioning,duration,p8,clock,throttle,thermal,hw-power-brake,"
        "composite-mask,asymmetric-mask,policy-mismatch,allocator,missing-status,"
        "diagnostic-status,smem-missing,smem-bytes,smem-count,smem-sha,"
        "smem-schema,bf16-v3-schema,bf16-v4-envelope,contiguous-v1-schema,"
        "w4a8-v1-schema,mla-decode-v1-schema,mla-prefill-v3-schema,"
        "tp-moe-v1-schema"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
