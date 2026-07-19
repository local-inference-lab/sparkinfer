#!/usr/bin/env python3
"""Synthetic fail-closed validation for the separate-source E2E release gate.

This is a manual evidence-tool self-test, not a CPU substitute for any GPU
kernel test.  It constructs all 104 required process artifacts, including an
explicitly reviewed cross-arm topology change and arm-specific PTXAS versions,
then proves incomplete coverage, invalid artifact provenance, unreviewed
topology drift, and a performance regression are rejected.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable

from validation.cutlass_migration.acceptance.e2e.index import (
    BASELINE_RUNTIME_OVERLAY_PATHS,
    CONTRACT_SCHEMA,
    EVIDENCE_SET_SCHEMA,
    EXPECTED_PACKAGES,
    PHYSICAL_GPUS,
    POSITION_ARM,
    PRODUCTION_SOURCE_SCHEMA,
    REQUIRED_FAMILIES,
    RUN_SCHEMA,
    SEQUENCE,
    EndToEndValidationError,
    _canonical_sha256,
    _sha256_file,
    _validate_conditions,
    build_index,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _record(path: str, label: str) -> dict[str, object]:
    return {"path": path, "sha256": _sha(label), "size_bytes": len(label)}


def _package(label: str, files: list[dict[str, object]]) -> dict[str, object]:
    return {
        "root": "b12x",
        "fingerprint": _sha(f"{label}-content"),
        "records_sha256": _canonical_sha256(files),
        "file_count": len(files),
        "files": files,
    }


def _source_manifest(root: Path, side: str) -> dict[str, object]:
    common = _record("common.py", f"{side}-common")
    if side == "baseline":
        production_files = [
            common,
            _record("cute/compiler.py", "baseline-production-compiler"),
            _record("cute/runtime_patches.py", "baseline-production-runtime"),
        ]
        runtime_files = [
            common,
            _record("cute/compiler.py", "baseline-instrumented-compiler"),
            _record("cute/runtime_patches.py", "baseline-instrumented-runtime"),
        ]
        policy = "instrumentation-only"
        allowed_paths = list(BASELINE_RUNTIME_OVERLAY_PATHS)
        production_status: list[str] = []
        runtime_status = [
            " M b12x/cute/compiler.py",
            " M b12x/cute/runtime_patches.py",
        ]
    else:
        production_files = [
            common,
            _record("cute/compiler.py", "current-compiler"),
            _record("cute/runtime_patches.py", "current-runtime"),
        ]
        runtime_files = deepcopy(production_files)
        policy = "none"
        allowed_paths = []
        production_status = []
        runtime_status = []
    production_files.sort(key=lambda record: str(record["path"]))
    runtime_files.sort(key=lambda record: str(record["path"]))
    production_by_path = {
        f"b12x/{record['path']}": record for record in production_files
    }
    runtime_by_path = {f"b12x/{record['path']}": record for record in runtime_files}
    changed_paths = sorted(
        path
        for path in set(production_by_path) | set(runtime_by_path)
        if production_by_path.get(path) != runtime_by_path.get(path)
    )
    details = {
        path: {
            "production": production_by_path.get(path),
            "runtime": runtime_by_path.get(path),
        }
        for path in changed_paths
    }
    commit = "1" * 40 if side == "baseline" else "2" * 40
    payload = {
        "schema": PRODUCTION_SOURCE_SCHEMA,
        "side": side,
        "source_id": f"synthetic-{side}",
        "production": {
            "repo_root": str((root / f"{side}-production").resolve()),
            "git": {"commit": commit, "status": production_status},
            "b12x_package": _package(f"{side}-production", production_files),
        },
        "runtime": {
            "repo_root": str((root / f"{side}-runtime").resolve()),
            "git": {"commit": commit, "status": runtime_status},
            "b12x_package": _package(f"{side}-runtime", runtime_files),
        },
        "runtime_overlay": {
            "policy": policy,
            "allowed_paths": allowed_paths,
            "changed_paths": changed_paths,
            "details_sha256": _canonical_sha256(details),
        },
    }
    return {**payload, "manifest_sha256": _canonical_sha256(payload)}


def _topology(family: str, side: str) -> dict[str, object]:
    changed_family = REQUIRED_FAMILIES[0]
    topology_label = (
        f"{family}-topology-{side}"
        if family == changed_family
        else f"{family}-topology"
    )
    node_count = 4 if family == changed_family and side == "current" else 3
    return {
        "topology_sha256": _sha(topology_label),
        "node_count": node_count,
        "kernel_node_count": node_count - 1,
    }


def _compile_identity(family: str, side: str, role: str) -> dict[str, str]:
    compile_spec_json = json.dumps(
        {"family": family, "role": role, "shape": [128, 4096], "side": side},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "role": role,
        "kernel_id": f"synthetic.{family}",
        "compile_spec_hash": hashlib.sha256(compile_spec_json.encode()).hexdigest(),
        "compile_spec_json": compile_spec_json,
    }


def _compile_side_contract(
    family: str,
    side: str,
    *,
    kernel_node_count: int,
) -> dict[str, object]:
    if family == REQUIRED_FAMILIES[1]:
        if kernel_node_count != 2:
            raise AssertionError("mixed-node synthetic family must have two kernels")
        artifact = _compile_identity(family, side, "stage-0")
        kernel_name = f"synthetic_torch_setup_{side}"
        return {
            "artifacts": [artifact],
            "launch_plan": [
                {
                    "node_index": 1,
                    "artifact_role": "stage-0",
                    "kernel_id": artifact["kernel_id"],
                    "compile_spec_hash": artifact["compile_spec_hash"],
                    "multiplicity_index": 1,
                }
            ],
            "source_owned_kernel_nodes": [
                {
                    "node_index": 0,
                    "role": "torch-setup",
                    "implementation": "torch_cuda",
                    "kernel_name": kernel_name,
                    "kernel_name_sha256": hashlib.sha256(
                        kernel_name.encode("utf-8")
                    ).hexdigest(),
                    "grid": [1, 1, 1],
                    "block": [32, 1, 1],
                    "dynamic_smem_bytes": 0,
                    "source_files": [
                        {
                            "path": f"validation/cutlass_migration/acceptance/single_arm/{family}.py",
                            "sha256": _sha(f"producer-{family}"),
                        }
                    ],
                }
            ],
        }
    artifacts = [
        _compile_identity(family, side, role) for role in ("stage-0", "stage-1")
    ]
    by_role = {artifact["role"]: artifact for artifact in artifacts}
    roles = ["stage-0", "stage-1"]
    if kernel_node_count == 3:
        roles.append("stage-0")
    if len(roles) != kernel_node_count:
        raise AssertionError(f"unsupported synthetic kernel count {kernel_node_count}")
    multiplicities: dict[str, int] = {}
    launch_plan = []
    for node_index, role in enumerate(roles):
        multiplicity = multiplicities.get(role, 0) + 1
        multiplicities[role] = multiplicity
        launch_plan.append(
            {
                "node_index": node_index,
                "artifact_role": role,
                "kernel_id": by_role[role]["kernel_id"],
                "compile_spec_hash": by_role[role]["compile_spec_hash"],
                "multiplicity_index": multiplicity,
            }
        )
    return {
        "artifacts": artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": [],
    }


def _contract() -> dict[str, object]:
    harness_files = [
        _record(
            f"validation/cutlass_migration/acceptance/single_arm/{family}.py",
            f"producer-{family}",
        )
        for family in REQUIRED_FAMILIES
    ]
    harness_files.sort(key=lambda record: str(record["path"]))
    families: dict[str, object] = {}
    for family in REQUIRED_FAMILIES:
        baseline_topology = _topology(family, "baseline")
        current_topology = _topology(family, "current")
        changed = baseline_topology != current_topology
        case_payload = {
            "case_id": f"{family}/synthetic-prefill",
            "input_sha256": _sha(f"{family}-input"),
            "required_correctness_gates": ["torch-reference"],
            "cross_arm_output_policy": "bit-exact",
            "graph_topology_contract": {
                "disposition": "changed-reviewed" if changed else "equal",
                "reason": "migration planner adds one kernel node" if changed else "",
                "baseline": baseline_topology,
                "current": current_topology,
            },
            "compile_artifact_contract": {
                side: _compile_side_contract(
                    family,
                    side,
                    kernel_node_count=int(
                        (baseline_topology if side == "baseline" else current_topology)[
                            "kernel_node_count"
                        ]
                    ),
                )
                for side in ("baseline", "current")
            },
        }
        case = {
            **case_payload,
            "case_contract_sha256": _canonical_sha256(case_payload),
        }
        producer = f"validation/cutlass_migration/acceptance/single_arm/{family}.py"
        producer_sha = next(
            record["sha256"] for record in harness_files if record["path"] == producer
        )
        family_payload = {
            "producer": producer,
            "producer_sha256": producer_sha,
            "cases": [case],
        }
        families[family] = {
            **family_payload,
            "family_contract_sha256": _canonical_sha256(family_payload),
        }
    payload = {
        "schema": CONTRACT_SCHEMA,
        "corpus_id": "synthetic-complete-e2e",
        "version": "1",
        "harness": {
            "files": harness_files,
            "file_count": len(harness_files),
            "tree_fingerprint": _canonical_sha256(harness_files),
        },
        "arm_toolchains": {
            "baseline": {
                "cutlass_packages": EXPECTED_PACKAGES["baseline"],
                "ptxas_version": "13.1.66",
            },
            "current": {
                "cutlass_packages": EXPECTED_PACKAGES["current"],
                "ptxas_version": "13.3.27",
            },
        },
        "required_families": list(REQUIRED_FAMILIES),
        "families": families,
    }
    return {**payload, "contract_sha256": _canonical_sha256(payload)}


def _mode_snapshot(
    gpu: int,
    uuid: str,
    captured_ns: int,
    *,
    active_throttle_reasons: int = 0,
) -> dict[str, object]:
    return {
        "available": True,
        "torch_uuid": uuid,
        "nvidia_smi_uuid": uuid,
        "captured_unix_ns": captured_ns,
        "fields": {
            "index": str(gpu),
            "uuid": uuid,
            "pstate": "P1",
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
            "clocks.current.sm": "2400 MHz",
            "clocks.current.memory": "14000 MHz",
            "clocks_throttle_reasons.active": f"0x{active_throttle_reasons:x}",
            "power.draw": "300 W",
            "power.limit": "600 W",
            "temperature.gpu": "45",
        },
    }


def _condition(
    sample: float,
    *,
    cold_l2: bool,
    gpu: int,
    uuid: str,
    captured_base_ns: int,
    required_active_throttle_reasons: int = 0,
) -> dict[str, object]:
    replays_per_reported_sample = 2
    inner_samples_us = [[sample - 0.01, sample + 0.01] for _ in range(1_000)]
    event_batch_replays = 25
    batch_replay_capacity = min(len(inner_samples_us), event_batch_replays)
    pair_count = batch_replay_capacity * replays_per_reported_sample
    batch_count = math.ceil(len(inner_samples_us) / event_batch_replays)
    mode_probe = _mode_snapshot(
        gpu,
        uuid,
        captured_base_ns,
        active_throttle_reasons=required_active_throttle_reasons,
    )
    mode_before = _mode_snapshot(
        gpu,
        uuid,
        captured_base_ns + 1,
        active_throttle_reasons=required_active_throttle_reasons,
    )
    mode_after = _mode_snapshot(
        gpu,
        uuid,
        captured_base_ns + 2,
        active_throttle_reasons=required_active_throttle_reasons,
    )
    return {
        "l2_flushed": cold_l2,
        "l2_flush_bytes": 200 if cold_l2 else 0,
        "preconditioning": {
            "policy": "single_exact_target_graph_duration",
            "minimum_replays": 2_000,
            "minimum_active_seconds": 5.0,
            "maximum_active_seconds": 30.0,
            "completed_replays": 2_048,
            "observed_active_seconds": 5.5,
            "target_graph_replays": 2_048,
            "cold_l2_flush_before_every_replay": cold_l2,
            "flush_inside_timed_interval": False,
            "required_pstate": "P1",
            "required_active_throttle_reasons": required_active_throttle_reasons,
            "mode_probes": [mode_probe],
        },
        "event_pool": {
            "schema": "b12x.cuda_event_pool.v1",
            "allocation_phase": "before_reported_samples",
            "prewarm_phase": "before_reported_samples",
            "prewarm_each_event": True,
            "one_pair_per_inner_replay": True,
            "event_creation_inside_sample_schedule": False,
            "initialized_before_target_graph_preconditioning": True,
            "reuse_boundary": "after_stream_synchronize_and_elapsed_query",
            "event_batch_replays": event_batch_replays,
            "batch_replay_capacity": batch_replay_capacity,
            "pair_count": pair_count,
            "event_count": 2 * pair_count,
            "unique_event_handle_count": 2 * pair_count,
            "event_handle_sha256": "a" * 64,
            "prewarm_elapsed_query_count": pair_count,
            "prewarm_elapsed_sha256": "b" * 64,
            "batch_count": batch_count,
            "reuse_count": max(0, batch_count - 1),
        },
        "gpu_mode_before_timing": mode_before,
        "gpu_mode_after_timing": mode_after,
        "gpu_mode_stability": {
            "schema": "b12x.gpu_mode_stability.v1",
            "required_pstate": "P1",
            "required_memory_clock_equality": True,
            "max_sm_clock_delta_mhz": 60.0,
            "observed_sm_clock_delta_mhz": 0.0,
            "observed_before_sm_clock_mhz": 2400.0,
            "observed_after_sm_clock_mhz": 2400.0,
            "observed_memory_clock_mhz": 14000.0,
            "required_active_throttle_reasons": required_active_throttle_reasons,
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
        "replays_per_reported_sample": replays_per_reported_sample,
        "aggregation": {
            "reported_sample": "arithmetic_mean_us",
            "inner_event_bracketing": "independent_per_graph_replay",
            "inner_schedule": "same_exact_graph_replay_per_repetition",
            "flush_before_every_inner_replay": cold_l2,
            "flush_inside_timed_interval": False,
        },
        "inner_samples_us": inner_samples_us,
        "inner_sample_count": len(inner_samples_us) * replays_per_reported_sample,
        "samples_us": [sample] * len(inner_samples_us),
    }


def _run(
    *,
    family: str,
    gpu: int,
    position: str,
    ordinal: int,
    contract: dict[str, Any],
    source_manifest: dict[str, Any],
    source_artifact_sha: str,
) -> dict[str, object]:
    arm = POSITION_ARM[position]
    case = contract["families"][family]["cases"][0]
    producer = contract["families"][family]["producer"]
    topology = case["graph_topology_contract"][arm]
    compile_contract = case["compile_artifact_contract"][arm]
    uuid = f"GPU-{gpu:064x}"
    started = 1_000_000_000 + ordinal * 10_000
    baseline = arm == "baseline"
    warm_sample = 10.0 if baseline else 9.98
    cold_sample = 12.0 if baseline else 11.97
    toolchain = [["cutlass_dsl", "4.5.2" if baseline else "4.6.0"]]
    artifacts: list[dict[str, object]] = []
    by_role: dict[str, dict[str, object]] = {}
    for compile_identity in compile_contract["artifacts"]:
        role = str(compile_identity["role"])
        cache_root = Path(f"/synthetic-cache/gpu{gpu}/{arm}/{role}")
        cache_key = _sha(f"cache-{family}-{gpu}-{arm}-{role}")
        cache_stem = cache_root / cache_key[:2] / cache_key
        manifest_sha256 = _sha(f"manifest-{family}-{gpu}-{arm}-{role}")
        object_sha256 = _sha(f"object-{family}-{gpu}-{arm}-{role}")
        object_bytes = 65_536 + len(family) + len(role)
        verification = {
            "passed": True,
            "manifest_sha256": manifest_sha256,
            "object_sha256": object_sha256,
            "object_bytes": object_bytes,
        }
        evidence = {
            "cache_root": str(cache_root),
            "cache_key": cache_key,
            "manifest_path": str(cache_stem.with_suffix(".json")),
            "manifest_sha256": manifest_sha256,
            "frontend_ptx_sidecar_path": str(cache_stem.with_suffix(".ptx.json")),
            "frontend_ptx_sidecar_sha256": _sha(
                f"ptx-sidecar-{family}-{gpu}-{arm}-{role}"
            ),
            "object_path": str(cache_stem.with_suffix(".o")),
            "object_sha256": object_sha256,
            "object_bytes": object_bytes,
            "compile_spec_hash": compile_identity["compile_spec_hash"],
            "compile_spec_json": compile_identity["compile_spec_json"],
            "semantic_key": _sha(f"semantic-{family}-{gpu}-{arm}-{role}"),
            "kernel_id": compile_identity["kernel_id"],
            "package_fingerprint": source_manifest["runtime"]["b12x_package"][
                "fingerprint"
            ],
            "toolchain": toolchain,
            "toolchain_sha256": _canonical_sha256(toolchain),
            "verification_before": verification,
            "verification_after": deepcopy(verification),
        }
        binding = {
            "role": role,
            "kernel_id": evidence["kernel_id"],
            "compile_spec_hash": evidence["compile_spec_hash"],
            "object_sha256": evidence["object_sha256"],
            "evidence": evidence,
        }
        artifacts.append(binding)
        by_role[role] = binding
    launch_plan = [
        {
            **binding,
            "object_sha256": by_role[str(binding["artifact_role"])]["object_sha256"],
        }
        for binding in compile_contract["launch_plan"]
    ]
    case_result = {
        "case_id": case["case_id"],
        "case_contract_sha256": case["case_contract_sha256"],
        "input_sha256": case["input_sha256"],
        "artifacts": artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": deepcopy(
            compile_contract["source_owned_kernel_nodes"]
        ),
        "correctness": {
            "independent_oracle": True,
            "oracle": "torch-reference",
            "passed": True,
            "finite": True,
            "nonzero_count": 128,
            "gates": {"torch-reference": True},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": case["input_sha256"],
            "output_sha256": _sha(f"{family}-output"),
        },
        "graph": {
            "capture_passed": True,
            "replay_passed": True,
            "topology_stable": True,
            "addresses_stable": True,
            "live_input_changed_output": True,
            "poison_overwrite_passed": True,
            **topology,
        },
        "allocation": {
            "fixed_workspace_capacity": True,
            "workspace_capacity_bytes": 4096 if baseline else 8192,
            "stable_addresses": True,
            "allocator_stable": True,
            "zero_replay_allocations": True,
            "allocated_bytes_before": 65536,
            "allocated_bytes_after": 65536,
            "reserved_bytes_before": 131072,
            "reserved_bytes_after": 131072,
            "condition_counters": {
                "warm_l2": {
                    "allocated_bytes_before": 65536,
                    "allocated_bytes_after": 65536,
                    "reserved_bytes_before": 131072,
                    "reserved_bytes_after": 131072,
                },
                "cold_l2": {
                    "allocated_bytes_before": 262144,
                    "allocated_bytes_after": 262144,
                    "reserved_bytes_before": 524288,
                    "reserved_bytes_after": 524288,
                },
            },
        },
        "conditions": {
            "warm_l2": _condition(
                warm_sample,
                cold_l2=False,
                gpu=gpu,
                uuid=uuid,
                captured_base_ns=started + 100,
            ),
            "cold_l2": _condition(
                cold_sample,
                cold_l2=True,
                gpu=gpu,
                uuid=uuid,
                captured_base_ns=started + 400,
            ),
        },
    }
    payload = {
        "schema": RUN_SCHEMA,
        "family": family,
        "arm": arm,
        "sequence_position": position,
        "evidence_status": "final-source",
        "invocation": {
            "process_id": _sha(f"process-{family}-{gpu}-{position}"),
            "pid": ordinal + 1,
            "started_unix_ns": started,
            "finished_unix_ns": started + 5_000,
            "command": ["python", producer, "--case", case["case_id"]],
            "worktree": source_manifest["runtime"]["repo_root"],
        },
        "source": {
            "manifest_sha256": source_manifest["manifest_sha256"],
            "manifest_artifact_sha256": source_artifact_sha,
            "production_fingerprint": source_manifest["production"]["b12x_package"][
                "fingerprint"
            ],
            "runtime_package_fingerprint": source_manifest["runtime"]["b12x_package"][
                "fingerprint"
            ],
        },
        "producer": {
            "path": producer,
            "sha256": contract["families"][family]["producer_sha256"],
        },
        "harness_case_contract_sha256": contract["contract_sha256"],
        "cutlass_packages": EXPECTED_PACKAGES[arm],
        "runtime": {
            "python_version": "3.12.9",
            "torch_version": "2.8.0",
            "torch_cuda_version": "13.0",
            "cuda_driver_version": "590.1",
            "ptxas_version": "13.1.66" if baseline else "13.3.27",
            "raw_environment_sha256": _sha(f"raw-environment-{arm}"),
            "comparison_environment_sha256": _sha("normalized-measurement-environment"),
        },
        "gpu": {
            "physical_ordinal": gpu,
            "name": "NVIDIA RTX PRO 6000 Blackwell",
            "uuid": uuid,
            "capability": [12, 0],
            "l2_cache_bytes": 100,
            "mode_before": _mode_snapshot(gpu, uuid, started + 1),
            "mode_after": _mode_snapshot(gpu, uuid, started + 4_999),
        },
        "cases": [case_result],
    }
    return {**payload, "result_sha256": _canonical_sha256(payload)}


def _expect_failure(
    name: str,
    callback: Callable[[], object],
    expected_text: str,
) -> None:
    try:
        callback()
    except EndToEndValidationError as exc:
        if expected_text not in str(exc):
            raise AssertionError(
                f"{name}: wrong failure: expected {expected_text!r} in {str(exc)!r}"
            ) from exc
    else:
        raise AssertionError(f"{name}: invalid evidence unexpectedly passed")


def _validate_timing_mode_policy_contract() -> None:
    gpu = 4
    uuid = f"GPU-{gpu:064x}"

    def conditions(required_mask: int) -> dict[str, object]:
        return {
            "warm_l2": _condition(
                10.0,
                cold_l2=False,
                gpu=gpu,
                uuid=uuid,
                captured_base_ns=1_000,
                required_active_throttle_reasons=required_mask,
            ),
            "cold_l2": _condition(
                12.0,
                cold_l2=True,
                gpu=gpu,
                uuid=uuid,
                captured_base_ns=2_000,
                required_active_throttle_reasons=required_mask,
            ),
        }

    validated = _validate_conditions(
        conditions(0x4),
        location="synthetic.conditions",
        l2_cache_bytes=100,
        physical_gpu=gpu,
        gpu_uuid=uuid,
    )
    if any(
        condition["preconditioning"]["required_active_throttle_reasons"] != 0x4
        for condition in validated.values()
    ):
        raise AssertionError("explicit SW-power-cap policy was not retained")

    for unsupported_mask in (0x20, 0x80, 0x84):
        _expect_failure(
            f"unsupported requested throttle mask {unsupported_mask:#x}",
            lambda unsupported_mask=unsupported_mask: _validate_conditions(
                conditions(unsupported_mask),
                location="synthetic.conditions",
                l2_cache_bytes=100,
                physical_gpu=gpu,
                gpu_uuid=uuid,
            ),
            "must be exactly 0 or 0x4",
        )

    asymmetric = conditions(0x4)
    asymmetric["warm_l2"]["gpu_mode_after_timing"]["fields"][
        "clocks_throttle_reasons.active"
    ] = "0x0"
    _expect_failure(
        "asymmetric explicit SW-power-cap state",
        lambda: _validate_conditions(
            asymmetric,
            location="synthetic.conditions",
            l2_cache_bytes=100,
            physical_gpu=gpu,
            gpu_uuid=uuid,
        ),
        "required stable P1/throttle-mask policy",
    )

    mixed_conditions = conditions(0)
    mixed_conditions["cold_l2"] = conditions(0x4)["cold_l2"]
    _expect_failure(
        "warm/cold timing policy mismatch",
        lambda: _validate_conditions(
            mixed_conditions,
            location="synthetic.conditions",
            l2_cache_bytes=100,
            physical_gpu=gpu,
            gpu_uuid=uuid,
        ),
        "warm/cold timing conditions request different",
    )


def _publish_mutated_evidence(
    *,
    root: Path,
    evidence_payload: dict[str, object],
    entries: list[dict[str, object]],
    entry: dict[str, object],
    run: dict[str, Any],
    stem: str,
) -> Path:
    run["result_sha256"] = _canonical_sha256(
        {key: value for key, value in run.items() if key != "result_sha256"}
    )
    run_path = root / f"{stem}-run.json"
    _write_json(run_path, run)
    entry["path"] = run_path.name
    entry["sha256"] = _sha256_file(run_path)
    payload = {**evidence_payload, "runs": entries}
    evidence = {**payload, "evidence_set_sha256": _canonical_sha256(payload)}
    path = root / f"{stem}-evidence.json"
    _write_json(path, evidence)
    return path


def _publish_first_run_mutation(
    *,
    root: Path,
    evidence_payload: dict[str, object],
    entries: list[dict[str, object]],
    stem: str,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    mutated_entries = deepcopy(entries)
    entry = mutated_entries[0]
    run = json.loads((root / str(entry["path"])).read_text(encoding="utf-8"))
    mutate(run)
    return _publish_mutated_evidence(
        root=root,
        evidence_payload=evidence_payload,
        entries=mutated_entries,
        entry=entry,
        run=run,
        stem=stem,
    )


def main() -> int:
    _validate_timing_mode_policy_contract()
    with tempfile.TemporaryDirectory(prefix="b12x-e2e-selftest-") as raw_root:
        root = Path(raw_root)
        sources: dict[str, dict[str, Any]] = {}
        source_paths: dict[str, Path] = {}
        for side in ("baseline", "current"):
            sources[side] = _source_manifest(root, side)
            source_paths[side] = root / f"{side}-source.json"
            _write_json(source_paths[side], sources[side])
        contract = _contract()
        contract_path = root / "contract.json"
        _write_json(contract_path, contract)

        entries: list[dict[str, object]] = []
        ordinal = 0
        for family in REQUIRED_FAMILIES:
            for gpu in PHYSICAL_GPUS:
                for position in SEQUENCE:
                    arm = POSITION_ARM[position]
                    run = _run(
                        family=family,
                        gpu=gpu,
                        position=position,
                        ordinal=ordinal,
                        contract=contract,
                        source_manifest=sources[arm],
                        source_artifact_sha=_sha256_file(source_paths[arm]),
                    )
                    run_path = root / f"{family}-{gpu}-{position}.json"
                    _write_json(run_path, run)
                    entries.append(
                        {
                            "family": family,
                            "physical_gpu": gpu,
                            "position": position,
                            "path": run_path.name,
                            "sha256": _sha256_file(run_path),
                        }
                    )
                    ordinal += 1
        evidence_payload = {
            "schema": EVIDENCE_SET_SCHEMA,
            "contract_sha256": contract["contract_sha256"],
            "runs": entries,
        }
        evidence = {
            **evidence_payload,
            "evidence_set_sha256": _canonical_sha256(evidence_payload),
        }
        evidence_path = root / "evidence.json"
        _write_json(evidence_path, evidence)

        index, rows = build_index(
            baseline_source_manifest=source_paths["baseline"],
            current_source_manifest=source_paths["current"],
            contract_path=contract_path,
            evidence_set_path=evidence_path,
        )
        assert index["coverage"] == {
            "family_count": 13,
            "case_count": 13,
            "process_result_count": 104,
            "performance_row_count": 52,
        }
        assert len(rows) == 52
        assert {row["topology_disposition"] for row in rows} == {
            "equal",
            "changed-reviewed",
        }

        missing_status_entries = deepcopy(entries)
        missing_status_entry = missing_status_entries[0]
        missing_status_run = json.loads(
            (root / str(missing_status_entry["path"])).read_text(encoding="utf-8")
        )
        missing_status_run.pop("evidence_status")
        missing_status_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=missing_status_entries,
            entry=missing_status_entry,
            run=missing_status_run,
            stem="missing-evidence-status",
        )
        _expect_failure(
            "missing evidence status",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=missing_status_path,
            ),
            "explicit final-source evidence status is required",
        )

        diagnostic_status_entries = deepcopy(entries)
        diagnostic_status_entry = diagnostic_status_entries[0]
        diagnostic_status_run = json.loads(
            (root / str(diagnostic_status_entry["path"])).read_text(encoding="utf-8")
        )
        diagnostic_status_run["evidence_status"] = "diagnostic-non-final"
        diagnostic_status_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=diagnostic_status_entries,
            entry=diagnostic_status_entry,
            run=diagnostic_status_run,
            stem="diagnostic-evidence-status",
        )
        _expect_failure(
            "diagnostic evidence status",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=diagnostic_status_path,
            ),
            "explicit final-source evidence status is required",
        )

        incomplete_payload = {**evidence_payload, "runs": entries[:-1]}
        incomplete = {
            **incomplete_payload,
            "evidence_set_sha256": _canonical_sha256(incomplete_payload),
        }
        incomplete_path = root / "incomplete-evidence.json"
        _write_json(incomplete_path, incomplete)
        _expect_failure(
            "incomplete family/GPU/position coverage",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=incomplete_path,
            ),
            "incomplete family/GPU/process coverage",
        )

        omitted_artifact_entries = deepcopy(entries)
        omitted_entry = omitted_artifact_entries[0]
        omitted_run_path = root / str(omitted_entry["path"])
        omitted_run = json.loads(omitted_run_path.read_text(encoding="utf-8"))
        omitted_run["cases"][0]["artifacts"].pop()
        omitted_run["result_sha256"] = _canonical_sha256(
            {key: value for key, value in omitted_run.items() if key != "result_sha256"}
        )
        omitted_replacement = root / "omitted-artifact.json"
        _write_json(omitted_replacement, omitted_run)
        omitted_entry["path"] = omitted_replacement.name
        omitted_entry["sha256"] = _sha256_file(omitted_replacement)
        omitted_payload = {**evidence_payload, "runs": omitted_artifact_entries}
        omitted_evidence = {
            **omitted_payload,
            "evidence_set_sha256": _canonical_sha256(omitted_payload),
        }
        omitted_path = root / "omitted-artifact-evidence.json"
        _write_json(omitted_path, omitted_evidence)
        _expect_failure(
            "omitted exact artifact",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=omitted_path,
            ),
            "missing or unused exact artifacts",
        )

        mutated_artifact_entries = deepcopy(entries)
        mutated_entry = mutated_artifact_entries[1]
        mutated_run_path = root / str(mutated_entry["path"])
        mutated_run = json.loads(mutated_run_path.read_text(encoding="utf-8"))
        mutated_run["cases"][0]["artifacts"][0]["evidence"]["verification_after"][
            "object_sha256"
        ] = _sha("mutated-object")
        mutated_run["result_sha256"] = _canonical_sha256(
            {key: value for key, value in mutated_run.items() if key != "result_sha256"}
        )
        mutated_replacement = root / "mutated-artifact.json"
        _write_json(mutated_replacement, mutated_run)
        mutated_entry["path"] = mutated_replacement.name
        mutated_entry["sha256"] = _sha256_file(mutated_replacement)
        mutated_payload = {**evidence_payload, "runs": mutated_artifact_entries}
        mutated_evidence = {
            **mutated_payload,
            "evidence_set_sha256": _canonical_sha256(mutated_payload),
        }
        mutated_path = root / "mutated-artifact-evidence.json"
        _write_json(mutated_path, mutated_evidence)
        _expect_failure(
            "mutated exact artifact",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=mutated_path,
            ),
            "verification differs from exact artifact identity",
        )

        substituted_artifact_entries = deepcopy(entries)
        target_entry = next(
            entry
            for entry in substituted_artifact_entries
            if entry["family"] == REQUIRED_FAMILIES[0]
            and entry["physical_gpu"] == PHYSICAL_GPUS[0]
            and entry["position"] == "a1"
        )
        donor_entry = next(
            entry
            for entry in entries
            if entry["family"] == REQUIRED_FAMILIES[1]
            and entry["physical_gpu"] == PHYSICAL_GPUS[0]
            and entry["position"] == "a1"
        )
        target_run = json.loads((root / str(target_entry["path"])).read_text())
        donor_run = json.loads((root / str(donor_entry["path"])).read_text())
        target_run["cases"][0]["artifacts"][0] = deepcopy(
            donor_run["cases"][0]["artifacts"][0]
        )
        target_run["result_sha256"] = _canonical_sha256(
            {key: value for key, value in target_run.items() if key != "result_sha256"}
        )
        substituted_replacement = root / "substituted-artifact.json"
        _write_json(substituted_replacement, target_run)
        target_entry["path"] = substituted_replacement.name
        target_entry["sha256"] = _sha256_file(substituted_replacement)
        substituted_payload = {**evidence_payload, "runs": substituted_artifact_entries}
        substituted_evidence = {
            **substituted_payload,
            "evidence_set_sha256": _canonical_sha256(substituted_payload),
        }
        substituted_path = root / "substituted-artifact-evidence.json"
        _write_json(substituted_path, substituted_evidence)
        _expect_failure(
            "cross-case exact artifact substitution",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=substituted_path,
            ),
            "differs from the reviewed compile contract",
        )

        source_owned_entries = deepcopy(entries)
        source_owned_entry = next(
            entry
            for entry in source_owned_entries
            if entry["family"] == REQUIRED_FAMILIES[1]
            and entry["physical_gpu"] == PHYSICAL_GPUS[0]
            and entry["position"] == "a1"
        )
        source_owned_run = json.loads(
            (root / str(source_owned_entry["path"])).read_text(encoding="utf-8")
        )
        source_owned_run["cases"][0]["source_owned_kernel_nodes"][0]["source_files"][0][
            "sha256"
        ] = _sha("substituted-source-owned-file")
        source_owned_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=source_owned_entries,
            entry=source_owned_entry,
            run=source_owned_run,
            stem="source-owned-substitution",
        )
        _expect_failure(
            "source-owned graph-node substitution",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=source_owned_path,
            ),
            "observed source-owned nodes differ from review",
        )

        duplicate_entries = deepcopy(entries)
        duplicate_entry = duplicate_entries[0]
        duplicate_run = json.loads(
            (root / str(duplicate_entry["path"])).read_text(encoding="utf-8")
        )
        duplicate_run["cases"][0]["artifacts"].append(
            deepcopy(duplicate_run["cases"][0]["artifacts"][0])
        )
        duplicate_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=duplicate_entries,
            entry=duplicate_entry,
            run=duplicate_run,
            stem="duplicate-object",
        )
        _expect_failure(
            "duplicate exact object",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=duplicate_path,
            ),
            "duplicate artifact role",
        )

        unused_entries = deepcopy(entries)
        unused_entry = unused_entries[0]
        unused_run = json.loads(
            (root / str(unused_entry["path"])).read_text(encoding="utf-8")
        )
        unused_case = unused_run["cases"][0]
        first_launch = deepcopy(unused_case["launch_plan"][0])
        first_launch["node_index"] = 1
        first_launch["multiplicity_index"] = 2
        unused_case["launch_plan"][1] = first_launch
        unused_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=unused_entries,
            entry=unused_entry,
            run=unused_run,
            stem="unused-object",
        )
        _expect_failure(
            "unused exact object",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=unused_path,
            ),
            "exact artifact is unused by the graph",
        )

        reordered_entries = deepcopy(entries)
        reordered_entry = reordered_entries[0]
        reordered_run = json.loads(
            (root / str(reordered_entry["path"])).read_text(encoding="utf-8")
        )
        reordered_plan = reordered_run["cases"][0]["launch_plan"]
        reordered_plan[0], reordered_plan[1] = reordered_plan[1], reordered_plan[0]
        for node_index, binding in enumerate(reordered_plan):
            binding["node_index"] = node_index
        reordered_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=reordered_entries,
            entry=reordered_entry,
            run=reordered_run,
            stem="reordered-launch",
        )
        _expect_failure(
            "reordered graph launch",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=reordered_path,
            ),
            "observed launch order or multiplicity differs from review",
        )

        multiplicity_entries = deepcopy(entries)
        multiplicity_entry = multiplicity_entries[0]
        multiplicity_run = json.loads(
            (root / str(multiplicity_entry["path"])).read_text(encoding="utf-8")
        )
        multiplicity_run["cases"][0]["launch_plan"][0]["multiplicity_index"] = 2
        multiplicity_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=multiplicity_entries,
            entry=multiplicity_entry,
            run=multiplicity_run,
            stem="multiplicity-drift",
        )
        _expect_failure(
            "launch multiplicity drift",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=multiplicity_path,
            ),
            "launch multiplicity is not exact",
        )

        topology_entries = deepcopy(entries)
        topology_entry = next(
            entry
            for entry in topology_entries
            if entry["family"] == REQUIRED_FAMILIES[0]
            and entry["physical_gpu"] == PHYSICAL_GPUS[0]
            and entry["position"] == "b2"
        )
        original_topology_path = root / str(topology_entry["path"])
        bad_topology_run = json.loads(
            original_topology_path.read_text(encoding="utf-8")
        )
        bad_topology_run["cases"][0]["graph"]["topology_sha256"] = _sha(
            "unreviewed-topology"
        )
        bad_topology_run["result_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in bad_topology_run.items()
                if key != "result_sha256"
            }
        )
        bad_topology_path = root / "bad-topology.json"
        _write_json(bad_topology_path, bad_topology_run)
        topology_entry["path"] = bad_topology_path.name
        topology_entry["sha256"] = _sha256_file(bad_topology_path)
        topology_payload = {**evidence_payload, "runs": topology_entries}
        topology_evidence = {
            **topology_payload,
            "evidence_set_sha256": _canonical_sha256(topology_payload),
        }
        topology_path = root / "topology-evidence.json"
        _write_json(topology_path, topology_evidence)
        _expect_failure(
            "unreviewed topology drift",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=topology_path,
            ),
            "repeat graph topology changed",
        )

        allocator_entries = deepcopy(entries)
        allocator_entry = next(
            entry
            for entry in allocator_entries
            if entry["family"] == REQUIRED_FAMILIES[0]
            and entry["physical_gpu"] == PHYSICAL_GPUS[0]
            and entry["position"] == "b1"
        )
        allocator_run_path = root / str(allocator_entry["path"])
        allocator_run = json.loads(allocator_run_path.read_text(encoding="utf-8"))
        allocator_run["cases"][0]["allocation"]["condition_counters"]["cold_l2"][
            "allocated_bytes_after"
        ] += 4096
        allocator_run["result_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in allocator_run.items()
                if key != "result_sha256"
            }
        )
        allocator_replacement = root / "allocator-instability.json"
        _write_json(allocator_replacement, allocator_run)
        allocator_entry["path"] = allocator_replacement.name
        allocator_entry["sha256"] = _sha256_file(allocator_replacement)
        allocator_payload = {**evidence_payload, "runs": allocator_entries}
        allocator_evidence = {
            **allocator_payload,
            "evidence_set_sha256": _canonical_sha256(allocator_payload),
        }
        allocator_path = root / "allocator-instability-evidence.json"
        _write_json(allocator_path, allocator_evidence)
        _expect_failure(
            "cold-L2 allocator instability",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=allocator_path,
            ),
            "cold_l2 allocator counters changed during replay",
        )

        inner_sample_entries = deepcopy(entries)
        inner_sample_entry = inner_sample_entries[0]
        inner_sample_run = json.loads(
            (root / str(inner_sample_entry["path"])).read_text(encoding="utf-8")
        )
        inner_sample_run["cases"][0]["conditions"]["warm_l2"]["inner_samples_us"][0][
            0
        ] += 1.0
        inner_sample_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=inner_sample_entries,
            entry=inner_sample_entry,
            run=inner_sample_run,
            stem="inner-sample-mutation",
        )
        _expect_failure(
            "aggregate inner-sample mutation",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=inner_sample_path,
            ),
            "reported sample is not its inner arithmetic mean",
        )

        inner_count_entries = deepcopy(entries)
        inner_count_entry = inner_count_entries[0]
        inner_count_run = json.loads(
            (root / str(inner_count_entry["path"])).read_text(encoding="utf-8")
        )
        inner_count_run["cases"][0]["conditions"]["warm_l2"]["inner_sample_count"] -= 1
        inner_count_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=inner_count_entries,
            entry=inner_count_entry,
            run=inner_count_run,
            stem="inner-count-mutation",
        )
        _expect_failure(
            "aggregate inner-count mutation",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=inner_count_path,
            ),
            "declared inner timing count differs",
        )

        aggregate_policy_entries = deepcopy(entries)
        aggregate_policy_entry = aggregate_policy_entries[0]
        aggregate_policy_run = json.loads(
            (root / str(aggregate_policy_entry["path"])).read_text(encoding="utf-8")
        )
        aggregate_policy_run["cases"][0]["conditions"]["cold_l2"]["aggregation"][
            "flush_before_every_inner_replay"
        ] = False
        aggregate_policy_path = _publish_mutated_evidence(
            root=root,
            evidence_payload=evidence_payload,
            entries=aggregate_policy_entries,
            entry=aggregate_policy_entry,
            run=aggregate_policy_run,
            stem="aggregate-policy-mutation",
        )
        _expect_failure(
            "aggregate cold-flush policy mutation",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=aggregate_policy_path,
            ),
            "aggregate timing policy differs from review",
        )

        old_schema_path = _publish_first_run_mutation(
            root=root,
            evidence_payload=evidence_payload,
            entries=entries,
            stem="old-timer-schema",
            mutate=lambda run: run.__setitem__(
                "schema", "b12x.cute.migration.end_to_end_process_result.v2"
            ),
        )
        _expect_failure(
            "old timer schema",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=old_schema_path,
            ),
            "invalid process-result schema",
        )

        missing_pool_path = _publish_first_run_mutation(
            root=root,
            evidence_payload=evidence_payload,
            entries=entries,
            stem="missing-event-pool",
            mutate=lambda run: run["cases"][0]["conditions"]["warm_l2"].pop(
                "event_pool"
            ),
        )
        _expect_failure(
            "missing event pool",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=missing_pool_path,
            ),
            "missing=['event_pool']",
        )

        invalid_pool_path = _publish_first_run_mutation(
            root=root,
            evidence_payload=evidence_payload,
            entries=entries,
            stem="invalid-event-pool-order",
            mutate=lambda run: run["cases"][0]["conditions"]["warm_l2"][
                "event_pool"
            ].__setitem__("initialized_before_target_graph_preconditioning", False),
        )
        _expect_failure(
            "event pool initialized after target conditioning",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=invalid_pool_path,
            ),
            "CUDA event-pool evidence is inconsistent",
        )

        short_duration_path = _publish_first_run_mutation(
            root=root,
            evidence_payload=evidence_payload,
            entries=entries,
            stem="short-precondition-duration",
            mutate=lambda run: run["cases"][0]["conditions"]["warm_l2"][
                "preconditioning"
            ].__setitem__("minimum_active_seconds", 4.0),
        )
        _expect_failure(
            "short preconditioning duration",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=short_duration_path,
            ),
            "preconditioning duration envelope is invalid",
        )

        precondition_throttle_path = _publish_first_run_mutation(
            root=root,
            evidence_payload=evidence_payload,
            entries=entries,
            stem="precondition-throttle",
            mutate=lambda run: run["cases"][0]["conditions"]["warm_l2"][
                "preconditioning"
            ]["mode_probes"][-1]["fields"].__setitem__(
                "clocks_throttle_reasons.active", "0x20"
            ),
        )
        _expect_failure(
            "throttled final preconditioning probe",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=precondition_throttle_path,
            ),
            "final preconditioning probe does not match",
        )

        p8_timing_path = _publish_first_run_mutation(
            root=root,
            evidence_payload=evidence_payload,
            entries=entries,
            stem="p8-timing",
            mutate=lambda run: run["cases"][0]["conditions"]["warm_l2"][
                "gpu_mode_before_timing"
            ]["fields"].__setitem__("pstate", "P8"),
        )
        _expect_failure(
            "P8 timing",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=p8_timing_path,
            ),
            "timing did not remain in the required stable P1/throttle-mask policy",
        )

        timing_throttle_path = _publish_first_run_mutation(
            root=root,
            evidence_payload=evidence_payload,
            entries=entries,
            stem="timing-throttle",
            mutate=lambda run: run["cases"][0]["conditions"]["warm_l2"][
                "gpu_mode_after_timing"
            ]["fields"].__setitem__("clocks_throttle_reasons.active", "0x80"),
        )
        _expect_failure(
            "timing throttle",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=timing_throttle_path,
            ),
            "timing did not remain in the required stable P1/throttle-mask policy",
        )

        clock_delta_path = _publish_first_run_mutation(
            root=root,
            evidence_payload=evidence_payload,
            entries=entries,
            stem="sm-clock-delta",
            mutate=lambda run: run["cases"][0]["conditions"]["warm_l2"][
                "gpu_mode_after_timing"
            ]["fields"].__setitem__("clocks.current.sm", "2500 MHz"),
        )
        _expect_failure(
            "SM clock delta",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=clock_delta_path,
            ),
            "GPU mode-stability evidence is inconsistent",
        )

        regression_entries = deepcopy(entries)
        for position in ("b1", "b2"):
            entry = next(
                item
                for item in regression_entries
                if item["family"] == REQUIRED_FAMILIES[1]
                and item["physical_gpu"] == PHYSICAL_GPUS[0]
                and item["position"] == position
            )
            run_path = root / str(entry["path"])
            regression_run = json.loads(run_path.read_text(encoding="utf-8"))
            regression_gpu = int(regression_run["gpu"]["physical_ordinal"])
            regression_uuid = str(regression_run["gpu"]["uuid"])
            regression_started = int(regression_run["invocation"]["started_unix_ns"])
            regression_run["cases"][0]["conditions"]["warm_l2"] = _condition(
                10.2,
                cold_l2=False,
                gpu=regression_gpu,
                uuid=regression_uuid,
                captured_base_ns=regression_started + 100,
            )
            regression_run["cases"][0]["conditions"]["cold_l2"] = _condition(
                12.3,
                cold_l2=True,
                gpu=regression_gpu,
                uuid=regression_uuid,
                captured_base_ns=regression_started + 400,
            )
            regression_run["result_sha256"] = _canonical_sha256(
                {
                    key: value
                    for key, value in regression_run.items()
                    if key != "result_sha256"
                }
            )
            replacement = root / f"regression-{position}.json"
            _write_json(replacement, regression_run)
            entry["path"] = replacement.name
            entry["sha256"] = _sha256_file(replacement)
        regression_payload = {**evidence_payload, "runs": regression_entries}
        regression_evidence = {
            **regression_payload,
            "evidence_set_sha256": _canonical_sha256(regression_payload),
        }
        regression_path = root / "regression-evidence.json"
        _write_json(regression_path, regression_evidence)
        _expect_failure(
            "performance regression",
            lambda: build_index(
                baseline_source_manifest=source_paths["baseline"],
                current_source_manifest=source_paths["current"],
                contract_path=contract_path,
                evidence_set_path=regression_path,
            ),
            "regression",
        )

    print(
        "status=pass positive=complete-104-process-gate,explicit-stable-sw-power-cap "
        "negative=missing-evidence-status,diagnostic-evidence-status,"
        "incomplete-coverage,artifact-omission,artifact-mutation,"
        "artifact-substitution,duplicate-object,unused-object,reordered-launch,"
        "multiplicity-drift,topology-drift,allocator-instability,"
        "inner-sample-mutation,inner-count-mutation,aggregate-policy-mutation,"
        "old-timer-schema,missing-event-pool,event-pool-order,"
        "short-precondition-duration,precondition-throttle,p8-timing,"
        "timing-throttle,thermal,hw-power-brake,composite-mask,asymmetric-mask,"
        "mixed-condition-policy,sm-clock-delta,"
        "performance-regression"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
