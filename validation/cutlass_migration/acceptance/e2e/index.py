#!/usr/bin/env python3
"""Build the separate-source CUTLASS 4.5.2 -> 4.6.0 release gate.

This validator is intentionally separate from the identical-final-source
exact-cache gate.  It consumes one immutable result artifact per process and
requires an A1/B1/B2/A2 sequence for every reviewed family on physical GPUs 4
and 5.  It is offline: importing this module does not import torch or CUTLASS.

The v1 contract is deliberately closed.  In particular, the required family
set and release ceilings are constants rather than caller-selectable options.
An omitted family, GPU, sequence position, case, correctness proof, graph
proof, or allocation proof is a validation failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import pairwise
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any


PRODUCTION_SOURCE_SCHEMA = "sparkinfer.cute.migration.production_source_snapshot.v1"
CONTRACT_SCHEMA = "sparkinfer.cute.migration.end_to_end_contract.v3"
EVIDENCE_SET_SCHEMA = "sparkinfer.cute.migration.end_to_end_evidence_set.v2"
RUN_SCHEMA = "sparkinfer.cute.migration.end_to_end_process_result.v4"
INDEX_SCHEMA = "sparkinfer.cute.migration.end_to_end_release_index.v2"
ROW_SCHEMA = "sparkinfer.cute.migration.end_to_end_performance_row.v3"

# Keep this aligned with the production schemas accepted by `python -m
# validation.cutlass_migration acceptance release-index`.  The end-to-end gate
# must not be made green by submitting only the families that already have a
# producer.
REQUIRED_FAMILIES = (
    "bf16_to_fp4_tma",
    "compute_exceptions",
    "contiguous_attention",
    "mla_decode_merge",
    "mla_prefill_mg",
    "nsa_indexer",
    "paged_attention",
    "residual_composite",
    "residual_prefill_partial",
    "tp_moe_dynamic",
    "w4a16_serving",
    "w4a16_topk_sum",
    "w4a8_dynamic",
)
PHYSICAL_GPUS = (4, 5)
SEQUENCE = ("a1", "b1", "b2", "a2")
POSITION_ARM = {
    "a1": "baseline",
    "b1": "current",
    "b2": "current",
    "a2": "baseline",
}
EXPECTED_PACKAGES = {
    "baseline": {
        "nvidia-cutlass-dsl": "4.5.2",
        "nvidia-cutlass-dsl-libs-base": "4.5.2",
        "nvidia-cutlass-dsl-libs-core": "missing",
        "nvidia-cutlass-dsl-libs-cu12": "missing",
        "nvidia-cutlass-dsl-libs-cu13": "4.5.2",
    },
    "current": {
        "nvidia-cutlass-dsl": "4.6.0",
        "nvidia-cutlass-dsl-libs-base": "4.6.0",
        "nvidia-cutlass-dsl-libs-core": "4.6.0",
        "nvidia-cutlass-dsl-libs-cu12": "4.6.0",
        "nvidia-cutlass-dsl-libs-cu13": "4.6.0",
    },
}

# The 4.5.2 evidence runtime carries only the compiler/resource-capture
# instrumentation needed to emit the same evidence as the final tree.  The
# production snapshot remains the pristine pre-migration package; these two
# files are recorded separately as the reviewed runtime overlay.
BASELINE_RUNTIME_OVERLAY_PATHS = (
    "sparkinfer/cute/compiler.py",
    "sparkinfer/cute/runtime_patches.py",
)

MINIMUM_SAMPLES_PER_PROCESS_CONDITION = 1_000
MINIMUM_L2_FLUSH_MULTIPLE = 2.0
MAX_MEAN_REGRESSION_PCT = 0.5
MAX_MEDIAN_REGRESSION_PCT = 0.5
MAX_P95_REGRESSION_PCT = 1.0
MAX_RUN_MEAN_DRIFT_PCT = 1.0
CUDA_EVENT_POOL_SCHEMA = "sparkinfer.cuda_event_pool.v1"
GPU_MODE_STABILITY_SCHEMA = "sparkinfer.gpu_mode_stability.v1"
REQUIRED_TIMING_PSTATE = "P1"
MINIMUM_PRECONDITION_SECONDS = 5.0
MAXIMUM_PRECONDITION_SECONDS = 60.0
MAX_TIMING_SM_CLOCK_DELTA_MHZ = 60.0
PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS = frozenset((0, 0x4))

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_ARTIFACT_ROLE_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_SOURCE_OWNED_IMPLEMENTATIONS = {"torch_cuda", "triton"}
_GPU_MODE_FIELDS = (
    "index",
    "uuid",
    "pstate",
    "persistence_mode",
    "compute_mode",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks_throttle_reasons.active",
    "power.draw",
    "power.limit",
    "temperature.gpu",
)
_GPU_STABLE_FIELDS = (
    "index",
    "uuid",
    "persistence_mode",
    "compute_mode",
    "power.limit",
)
_TIMING_STABLE_MODE_FIELDS = (
    "index",
    "uuid",
    "persistence_mode",
    "compute_mode",
    "power.limit",
)
_RUNTIME_CROSS_ARM_STABLE_FIELDS = (
    "python_version",
    "torch_version",
    "torch_cuda_version",
    "cuda_driver_version",
    "comparison_environment_sha256",
)
_RUNTIME_ARM_STABLE_FIELDS = (
    *_RUNTIME_CROSS_ARM_STABLE_FIELDS,
    "ptxas_version",
    "raw_environment_sha256",
)
_CSV_FIELDS = (
    "row_schema",
    "family",
    "case_id",
    "physical_gpu",
    "gpu_uuid",
    "condition",
    "required_active_throttle_reasons",
    "max_sm_clock_delta_mhz",
    "samples_a1",
    "samples_b1",
    "samples_b2",
    "samples_a2",
    "a1_mean_us",
    "b1_mean_us",
    "b2_mean_us",
    "a2_mean_us",
    "a_run_mean_drift_pct",
    "b_run_mean_drift_pct",
    "baseline_mean_us",
    "current_mean_us",
    "mean_ratio_current_over_baseline",
    "mean_regression_pct",
    "baseline_median_us",
    "current_median_us",
    "median_ratio_current_over_baseline",
    "median_regression_pct",
    "baseline_p95_us",
    "current_p95_us",
    "p95_ratio_current_over_baseline",
    "p95_regression_pct",
    "baseline_topology_sha256",
    "current_topology_sha256",
    "baseline_node_count",
    "current_node_count",
    "baseline_kernel_node_count",
    "current_kernel_node_count",
    "topology_disposition",
    "status",
)


class EndToEndValidationError(RuntimeError):
    """An immutable end-to-end evidence invariant failed."""


def _fail(message: str) -> None:
    raise EndToEndValidationError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EndToEndValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{path}: expected a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _exact_keys(value: object, expected: set[str], location: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{location}: expected an object")
    actual = set(value)
    _require(
        actual == expected,
        f"{location}: fields differ; missing={sorted(expected - actual)!r}, "
        f"unexpected={sorted(actual - expected)!r}",
    )
    return value


def _positive_int(value: object, location: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{location}: expected a positive integer",
    )
    return int(value)


def _nonnegative_int(value: object, location: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{location}: expected a nonnegative integer",
    )
    return int(value)


def _normalize_uuid(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def _artifact(path: Path, *, schema: str = "") -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema": schema,
    }


def _validate_file_records(value: object, location: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and value, f"{location}: no file records")
    records: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, raw in enumerate(value):
        record = _exact_keys(
            raw,
            {"path", "sha256", "size_bytes"},
            f"{location}[{index}]",
        )
        path = record["path"]
        _require(isinstance(path, str) and path, f"{location}[{index}]: empty path")
        _require(_is_sha256(record["sha256"]), f"{location}[{index}]: invalid SHA")
        _nonnegative_int(record["size_bytes"], f"{location}[{index}].size_bytes")
        records.append(record)
        paths.append(path)
    _require(paths == sorted(set(paths)), f"{location}: paths are not sorted/unique")
    return records


def _validate_package_tree(value: object, location: str) -> dict[str, Any]:
    package = _exact_keys(
        value,
        {"root", "fingerprint", "records_sha256", "file_count", "files"},
        location,
    )
    _require(package["root"] == "sparkinfer", f"{location}: package root is not sparkinfer")
    _require(
        _is_sha256(package["fingerprint"]), f"{location}: invalid content fingerprint"
    )
    files = _validate_file_records(package["files"], f"{location}.files")
    _require(
        package["file_count"] == len(files),
        f"{location}: file count differs from manifest rows",
    )
    _require(
        package["records_sha256"] == _canonical_sha256(files),
        f"{location}: file-record fingerprint mismatch",
    )
    return {**package, "files": files}


def _validate_source_endpoint(value: object, location: str) -> dict[str, Any]:
    endpoint = _exact_keys(value, {"repo_root", "git", "sparkinfer_package"}, location)
    repo_root = endpoint["repo_root"]
    _require(
        isinstance(repo_root, str) and Path(repo_root).is_absolute(),
        f"{location}: repo_root must be absolute provenance",
    )
    git = _exact_keys(endpoint["git"], {"commit", "status"}, f"{location}.git")
    _require(
        isinstance(git["commit"], str)
        and bool(_GIT_COMMIT_RE.fullmatch(git["commit"])),
        f"{location}: invalid git commit",
    )
    status = git["status"]
    _require(
        isinstance(status, list)
        and status == sorted(set(status))
        and all(isinstance(line, str) and line for line in status),
        f"{location}: git status is not sorted/unique string provenance",
    )
    package = _validate_package_tree(
        endpoint["sparkinfer_package"], f"{location}.sparkinfer_package"
    )
    return {**endpoint, "git": git, "sparkinfer_package": package}


def _overlay_payload(
    production_files: list[dict[str, Any]],
    runtime_files: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    production = {f"sparkinfer/{record['path']}": record for record in production_files}
    runtime = {f"sparkinfer/{record['path']}": record for record in runtime_files}
    changed_paths = sorted(
        path
        for path in set(production) | set(runtime)
        if production.get(path) != runtime.get(path)
    )
    details = {
        path: {
            "production": production.get(path),
            "runtime": runtime.get(path),
        }
        for path in changed_paths
    }
    return changed_paths, details


def _validate_source_manifest(path: Path, side: str) -> dict[str, Any]:
    manifest = _exact_keys(
        _load_json(path),
        {
            "schema",
            "side",
            "source_id",
            "production",
            "runtime",
            "runtime_overlay",
            "manifest_sha256",
        },
        str(path),
    )
    _require(
        manifest["schema"] == PRODUCTION_SOURCE_SCHEMA,
        f"{path}: unsupported production-source snapshot schema",
    )
    _require(manifest["side"] == side, f"{path}: source side is not {side}")
    _require(
        isinstance(manifest["source_id"], str) and manifest["source_id"],
        f"{path}: empty source_id",
    )
    recorded = manifest.get("manifest_sha256")
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    computed = _canonical_sha256(payload)
    _require(
        _is_sha256(recorded) and recorded == computed,
        f"{path}: source manifest canonical hash mismatch",
    )
    production = _validate_source_endpoint(manifest["production"], f"{path}.production")
    runtime = _validate_source_endpoint(manifest["runtime"], f"{path}.runtime")
    _require(
        production["git"]["commit"] == runtime["git"]["commit"],
        f"{path}: production and evidence runtime do not share a git base",
    )
    overlay = _exact_keys(
        manifest["runtime_overlay"],
        {"policy", "allowed_paths", "changed_paths", "details_sha256"},
        f"{path}.runtime_overlay",
    )
    changed_paths, details = _overlay_payload(
        production["sparkinfer_package"]["files"],
        runtime["sparkinfer_package"]["files"],
    )
    _require(
        overlay["changed_paths"] == changed_paths,
        f"{path}: runtime overlay does not match the frozen package records",
    )
    _require(
        overlay["details_sha256"] == _canonical_sha256(details),
        f"{path}: runtime-overlay detail fingerprint mismatch",
    )
    if side == "baseline":
        expected_overlay = list(BASELINE_RUNTIME_OVERLAY_PATHS)
        _require(
            production["git"]["status"] == []
            and overlay["policy"] == "instrumentation-only"
            and overlay["allowed_paths"] == expected_overlay
            and changed_paths == expected_overlay,
            f"{path}: baseline runtime overlay is not the exact reviewed instrumentation set",
        )
    else:
        _require(
            overlay["policy"] == "none"
            and overlay["allowed_paths"] == []
            and changed_paths == [],
            f"{path}: current runtime must exactly equal current production source",
        )
    return {
        "side": side,
        "manifest": manifest,
        "manifest_sha256": computed,
        "manifest_artifact_sha256": _sha256_file(path),
        "production_fingerprint": str(production["sparkinfer_package"]["fingerprint"]),
        "runtime_package_fingerprint": str(runtime["sparkinfer_package"]["fingerprint"]),
        "artifact": _artifact(path, schema=PRODUCTION_SOURCE_SCHEMA),
    }


def _validate_compile_identity(value: object, location: str) -> dict[str, Any]:
    identity = _exact_keys(
        value,
        {"role", "kernel_id", "compile_spec_hash", "compile_spec_json"},
        location,
    )
    role = identity["role"]
    _require(
        isinstance(role, str) and bool(_ARTIFACT_ROLE_RE.fullmatch(role)),
        f"{location}: invalid artifact role",
    )
    _require(
        isinstance(identity["kernel_id"], str) and bool(identity["kernel_id"]),
        f"{location}: kernel ID is empty",
    )
    compile_spec_json = identity["compile_spec_json"]
    _require(
        isinstance(compile_spec_json, str) and bool(compile_spec_json),
        f"{location}: compile-spec JSON is empty",
    )
    try:
        parsed_compile_spec = json.loads(compile_spec_json)
    except json.JSONDecodeError as exc:
        _fail(f"{location}: compile-spec JSON is invalid: {exc}")
    _require(
        isinstance(parsed_compile_spec, dict),
        f"{location}: compile spec is not an object",
    )
    _require(
        identity["compile_spec_hash"]
        == hashlib.sha256(compile_spec_json.encode()).hexdigest(),
        f"{location}: compile-spec hash mismatch",
    )
    return identity


def _validate_source_owned_kernel_nodes(
    value: object,
    *,
    location: str,
    kernel_node_count: int,
) -> list[dict[str, Any]]:
    """Validate reviewed non-CUTLASS nodes without erasing graph ordinals."""

    _require(isinstance(value, list), f"{location}: expected a list")
    normalized: list[dict[str, Any]] = []
    previous_node_index = -1
    for index, raw_record in enumerate(value):
        record_location = f"{location}[{index}]"
        record = _exact_keys(
            raw_record,
            {
                "node_index",
                "role",
                "implementation",
                "kernel_name",
                "kernel_name_sha256",
                "grid",
                "block",
                "dynamic_smem_bytes",
                "source_files",
            },
            record_location,
        )
        node_index = _nonnegative_int(
            record["node_index"], f"{record_location}.node_index"
        )
        _require(
            node_index < kernel_node_count and node_index > previous_node_index,
            f"{record_location}: source-owned node ordinals must be in-range and "
            "strictly increasing",
        )
        previous_node_index = node_index
        _require(
            isinstance(record["role"], str)
            and bool(_ARTIFACT_ROLE_RE.fullmatch(record["role"])),
            f"{record_location}: invalid source-owned role",
        )
        _require(
            isinstance(record["implementation"], str)
            and record["implementation"] in _SOURCE_OWNED_IMPLEMENTATIONS,
            f"{record_location}: unsupported source-owned implementation",
        )
        kernel_name = record["kernel_name"]
        _require(
            isinstance(kernel_name, str) and bool(kernel_name),
            f"{record_location}: kernel name is empty",
        )
        _require(
            record["kernel_name_sha256"]
            == hashlib.sha256(kernel_name.encode("utf-8")).hexdigest(),
            f"{record_location}: kernel-name SHA mismatch",
        )
        for field in ("grid", "block"):
            dimensions = record[field]
            _require(
                isinstance(dimensions, list)
                and len(dimensions) == 3
                and all(
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension > 0
                    for dimension in dimensions
                ),
                f"{record_location}.{field}: expected three positive integers",
            )
        _nonnegative_int(
            record["dynamic_smem_bytes"],
            f"{record_location}.dynamic_smem_bytes",
        )
        source_files = record["source_files"]
        _require(
            isinstance(source_files, list) and source_files,
            f"{record_location}.source_files: no source records",
        )
        source_paths: list[str] = []
        for file_index, raw_file in enumerate(source_files):
            file_location = f"{record_location}.source_files[{file_index}]"
            source_file = _exact_keys(raw_file, {"path", "sha256"}, file_location)
            source_path = source_file["path"]
            _require(
                isinstance(source_path, str)
                and bool(source_path)
                and not Path(source_path).is_absolute()
                and Path(source_path).as_posix() == source_path
                and ".." not in Path(source_path).parts,
                f"{file_location}: source path is not normalized repo-relative",
            )
            _require(
                _is_sha256(source_file["sha256"]),
                f"{file_location}: invalid source SHA",
            )
            source_paths.append(source_path)
        _require(
            source_paths == sorted(set(source_paths)),
            f"{record_location}.source_files: paths are not sorted/unique",
        )
        normalized.append(record)
    return normalized


def _validate_compile_side_contract(
    value: object,
    *,
    location: str,
    kernel_node_count: int,
) -> dict[str, Any]:
    contract = _exact_keys(
        value,
        {"artifacts", "launch_plan", "source_owned_kernel_nodes"},
        location,
    )
    raw_artifacts = contract["artifacts"]
    _require(
        isinstance(raw_artifacts, list) and raw_artifacts,
        f"{location}: compile artifact list is empty",
    )
    artifacts = [
        _validate_compile_identity(item, f"{location}.artifacts[{index}]")
        for index, item in enumerate(raw_artifacts)
    ]
    by_role = {str(item["role"]): item for item in artifacts}
    _require(
        len(by_role) == len(artifacts),
        f"{location}: duplicate artifact role",
    )
    compile_identities = {
        (str(item["kernel_id"]), str(item["compile_spec_hash"])) for item in artifacts
    }
    _require(
        len(compile_identities) == len(artifacts),
        f"{location}: duplicate compile artifact identity",
    )

    raw_plan = contract["launch_plan"]
    _require(
        isinstance(raw_plan, list) and raw_plan,
        f"{location}: launch plan is empty",
    )
    launch_plan: list[dict[str, Any]] = []
    next_multiplicity: dict[str, int] = {}
    used_roles: set[str] = set()
    previous_node_index = -1
    for index, raw_binding in enumerate(raw_plan):
        binding_location = f"{location}.launch_plan[{index}]"
        binding = _exact_keys(
            raw_binding,
            {
                "node_index",
                "artifact_role",
                "kernel_id",
                "compile_spec_hash",
                "multiplicity_index",
            },
            binding_location,
        )
        node_index = _nonnegative_int(
            binding["node_index"], f"{binding_location}.node_index"
        )
        _require(
            node_index < kernel_node_count and node_index > previous_node_index,
            f"{binding_location}: exact node ordinals must be in-range and "
            "strictly increasing",
        )
        previous_node_index = node_index
        role = binding["artifact_role"]
        _require(
            isinstance(role, str) and role in by_role,
            f"{binding_location}: launch references an unknown artifact role",
        )
        artifact = by_role[role]
        _require(
            binding["kernel_id"] == artifact["kernel_id"]
            and binding["compile_spec_hash"] == artifact["compile_spec_hash"],
            f"{binding_location}: launch identity differs from its artifact role",
        )
        expected_multiplicity = next_multiplicity.get(role, 0) + 1
        _require(
            binding["multiplicity_index"] == expected_multiplicity,
            f"{binding_location}: launch multiplicity is not exact",
        )
        next_multiplicity[role] = expected_multiplicity
        used_roles.add(role)
        launch_plan.append(binding)
    _require(
        used_roles == set(by_role),
        f"{location}: unused or missing compile artifact roles; "
        f"unused={sorted(set(by_role) - used_roles)!r}",
    )
    source_owned = _validate_source_owned_kernel_nodes(
        contract["source_owned_kernel_nodes"],
        location=f"{location}.source_owned_kernel_nodes",
        kernel_node_count=kernel_node_count,
    )
    covered_indices = sorted(
        [int(binding["node_index"]) for binding in launch_plan]
        + [int(record["node_index"]) for record in source_owned]
    )
    _require(
        covered_indices == list(range(kernel_node_count)),
        f"{location}: exact and source-owned nodes do not partition the graph; "
        f"observed={covered_indices!r}",
    )
    return {
        "artifacts": artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": source_owned,
    }


def _validate_contract(path: Path) -> dict[str, Any]:
    contract = _exact_keys(
        _load_json(path),
        {
            "schema",
            "corpus_id",
            "version",
            "harness",
            "arm_toolchains",
            "required_families",
            "families",
            "contract_sha256",
        },
        str(path),
    )
    _require(contract["schema"] == CONTRACT_SCHEMA, f"{path}: invalid contract schema")
    _require(
        isinstance(contract["corpus_id"], str) and contract["corpus_id"],
        f"{path}: empty corpus_id",
    )
    _require(
        isinstance(contract["version"], str) and contract["version"],
        f"{path}: empty contract version",
    )
    _require(
        contract["required_families"] == list(REQUIRED_FAMILIES),
        f"{path}: required family list is incomplete or reordered; "
        f"expected={list(REQUIRED_FAMILIES)!r}",
    )

    arm_toolchains = _exact_keys(
        contract["arm_toolchains"],
        {"baseline", "current"},
        f"{path}.arm_toolchains",
    )
    normalized_toolchains: dict[str, dict[str, Any]] = {}
    for side in ("baseline", "current"):
        toolchain = _exact_keys(
            arm_toolchains[side],
            {"cutlass_packages", "ptxas_version"},
            f"{path}.arm_toolchains.{side}",
        )
        _require(
            toolchain["cutlass_packages"] == EXPECTED_PACKAGES[side],
            f"{path}: {side} CUTLASS package contract is not exact",
        )
        _require(
            isinstance(toolchain["ptxas_version"], str) and toolchain["ptxas_version"],
            f"{path}: {side} ptxas version is empty",
        )
        normalized_toolchains[side] = toolchain
    families = contract["families"]
    _require(
        isinstance(families, dict) and set(families) == set(REQUIRED_FAMILIES),
        f"{path}: family contract set is incomplete; "
        f"missing={sorted(set(REQUIRED_FAMILIES) - set(families or {}))!r}, "
        f"unexpected={sorted(set(families or {}) - set(REQUIRED_FAMILIES))!r}",
    )

    harness = _exact_keys(
        contract["harness"],
        {"files", "file_count", "tree_fingerprint"},
        f"{path}.harness",
    )
    harness_files = _validate_file_records(harness["files"], f"{path}.harness.files")
    _require(
        harness["file_count"] == len(harness_files),
        f"{path}: harness file count mismatch",
    )
    _require(
        harness["tree_fingerprint"] == _canonical_sha256(harness_files),
        f"{path}: harness tree fingerprint mismatch",
    )
    harness_by_path = {record["path"]: record for record in harness_files}

    all_case_ids: set[str] = set()
    normalized_families: dict[str, dict[str, Any]] = {}
    for family in REQUIRED_FAMILIES:
        location = f"{path}.families.{family}"
        record = _exact_keys(
            families[family],
            {"producer", "producer_sha256", "cases", "family_contract_sha256"},
            location,
        )
        producer = record["producer"]
        _require(isinstance(producer, str) and producer, f"{location}: empty producer")
        _require(
            _is_sha256(record["producer_sha256"]), f"{location}: invalid producer SHA"
        )
        _require(
            producer in harness_by_path
            and harness_by_path[producer]["sha256"] == record["producer_sha256"],
            f"{location}: producer is not bound to the frozen harness",
        )
        cases = record["cases"]
        _require(isinstance(cases, list) and cases, f"{location}: no reviewed cases")
        indexed_cases: dict[str, dict[str, Any]] = {}
        for index, raw_case in enumerate(cases):
            case_location = f"{location}.cases[{index}]"
            case = _exact_keys(
                raw_case,
                {
                    "case_id",
                    "input_sha256",
                    "required_correctness_gates",
                    "cross_arm_output_policy",
                    "graph_topology_contract",
                    "compile_artifact_contract",
                    "case_contract_sha256",
                },
                case_location,
            )
            case_id = case["case_id"]
            _require(
                isinstance(case_id, str) and bool(_CASE_ID_RE.fullmatch(case_id)),
                f"{case_location}: invalid stable external case_id",
            )
            _require(
                case_id not in all_case_ids,
                f"{case_location}: duplicate global case_id",
            )
            all_case_ids.add(case_id)
            _require(
                _is_sha256(case["input_sha256"]), f"{case_location}: invalid input SHA"
            )
            gates = case["required_correctness_gates"]
            _require(
                isinstance(gates, list)
                and gates
                and gates == sorted(set(gates))
                and all(isinstance(gate, str) and gate for gate in gates),
                f"{case_location}: correctness gates are not sorted/unique/nonempty",
            )
            _require(
                case["cross_arm_output_policy"] in {"bit-exact", "oracle-only"},
                f"{case_location}: invalid cross-arm output policy",
            )
            topology = _exact_keys(
                case["graph_topology_contract"],
                {"disposition", "reason", "baseline", "current"},
                f"{case_location}.graph_topology_contract",
            )
            _require(
                topology["disposition"] in {"equal", "changed-reviewed"},
                f"{case_location}: invalid graph-topology disposition",
            )
            _require(
                isinstance(topology["reason"], str),
                f"{case_location}: graph-topology reason must be a string",
            )
            signatures: dict[str, dict[str, Any]] = {}
            for side in ("baseline", "current"):
                signature = _exact_keys(
                    topology[side],
                    {"topology_sha256", "node_count", "kernel_node_count"},
                    f"{case_location}.graph_topology_contract.{side}",
                )
                _require(
                    _is_sha256(signature["topology_sha256"]),
                    f"{case_location}: invalid {side} topology SHA",
                )
                node_count = _positive_int(
                    signature["node_count"],
                    f"{case_location}.graph_topology_contract.{side}.node_count",
                )
                kernel_node_count = _positive_int(
                    signature["kernel_node_count"],
                    f"{case_location}.graph_topology_contract.{side}.kernel_node_count",
                )
                _require(
                    kernel_node_count <= node_count,
                    f"{case_location}: {side} kernel nodes exceed graph nodes",
                )
                signatures[side] = signature
            signatures_equal = signatures["baseline"] == signatures["current"]
            if topology["disposition"] == "equal":
                _require(
                    topology["reason"] == "" and signatures_equal,
                    f"{case_location}: equal topology requires equal signatures and no reason",
                )
            else:
                _require(
                    bool(topology["reason"].strip()) and not signatures_equal,
                    f"{case_location}: changed topology requires a reason and differing signatures",
                )
            compile_contract = _exact_keys(
                case["compile_artifact_contract"],
                {"baseline", "current"},
                f"{case_location}.compile_artifact_contract",
            )
            for side in ("baseline", "current"):
                _validate_compile_side_contract(
                    compile_contract[side],
                    location=f"{case_location}.compile_artifact_contract.{side}",
                    kernel_node_count=int(signatures[side]["kernel_node_count"]),
                )
            case_payload = {
                key: value
                for key, value in case.items()
                if key != "case_contract_sha256"
            }
            _require(
                case["case_contract_sha256"] == _canonical_sha256(case_payload),
                f"{case_location}: case contract SHA mismatch",
            )
            indexed_cases[case_id] = case
        family_payload = {
            key: value
            for key, value in record.items()
            if key != "family_contract_sha256"
        }
        _require(
            record["family_contract_sha256"] == _canonical_sha256(family_payload),
            f"{location}: family contract SHA mismatch",
        )
        normalized_families[family] = {**record, "case_by_id": indexed_cases}

    payload = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    computed = _canonical_sha256(payload)
    _require(
        contract["contract_sha256"] == computed,
        f"{path}: combined harness/case-contract SHA mismatch",
    )
    return {
        "value": contract,
        "sha256": computed,
        "harness_fingerprint": str(harness["tree_fingerprint"]),
        "arm_toolchains": normalized_toolchains,
        "families": normalized_families,
        "artifact": _artifact(path, schema=CONTRACT_SCHEMA),
    }


def _validate_gpu_mode_snapshot(
    value: object,
    *,
    location: str,
    physical_gpu: int,
    gpu_uuid: str,
) -> dict[str, Any]:
    snapshot = _exact_keys(
        value,
        {"available", "torch_uuid", "nvidia_smi_uuid", "captured_unix_ns", "fields"},
        location,
    )
    _require(snapshot["available"] is True, f"{location}: GPU mode unavailable")
    fields = snapshot["fields"]
    _require(isinstance(fields, dict), f"{location}: mode fields are not an object")
    missing = [field for field in _GPU_MODE_FIELDS if not str(fields.get(field, ""))]
    _require(not missing, f"{location}: missing GPU mode fields {missing!r}")
    _require(
        str(fields["index"]) == str(physical_gpu), f"{location}: GPU index mismatch"
    )
    normalized_uuid = _normalize_uuid(gpu_uuid)
    _require(
        normalized_uuid
        and _normalize_uuid(fields["uuid"]) == normalized_uuid
        and _normalize_uuid(snapshot["torch_uuid"]) == normalized_uuid
        and _normalize_uuid(snapshot["nvidia_smi_uuid"]) == normalized_uuid,
        f"{location}: GPU UUID provenance mismatch",
    )
    _positive_int(snapshot["captured_unix_ns"], f"{location}.captured_unix_ns")
    return snapshot


def _timing_clock_mhz(snapshot: dict[str, Any], field: str, *, location: str) -> float:
    raw = snapshot["fields"].get(field)
    try:
        value = float(str(raw).split()[0])
    except (IndexError, ValueError):
        _fail(f"{location}: invalid {field}: {raw!r}")
    _require(
        math.isfinite(value) and value > 0.0,
        f"{location}: nonpositive {field}: {raw!r}",
    )
    return value


def _active_throttle_reasons(snapshot: dict[str, Any], *, location: str) -> int:
    raw = snapshot["fields"].get("clocks_throttle_reasons.active")
    try:
        value = int(str(raw).strip(), 0)
    except ValueError:
        _fail(f"{location}: invalid active throttle reasons: {raw!r}")
    _require(value >= 0, f"{location}: negative active throttle reasons: {raw!r}")
    return value


def _required_active_throttle_reasons(value: object, *, location: str) -> int:
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS,
        f"{location}: required active throttle reasons must be exactly 0 or 0x4",
    )
    return int(value)


def _validate_runtime(value: object, location: str) -> dict[str, Any]:
    runtime = _exact_keys(
        value,
        {
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "cuda_driver_version",
            "ptxas_version",
            "raw_environment_sha256",
            "comparison_environment_sha256",
        },
        location,
    )
    for field in (
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "cuda_driver_version",
        "ptxas_version",
    ):
        _require(
            isinstance(runtime[field], str) and runtime[field],
            f"{location}: empty {field}",
        )
    for field in ("raw_environment_sha256", "comparison_environment_sha256"):
        _require(_is_sha256(runtime[field]), f"{location}: invalid {field}")
    return runtime


def _validate_correctness(
    value: object,
    *,
    location: str,
    required_gates: list[str],
) -> dict[str, Any]:
    correctness = _exact_keys(
        value,
        {
            "independent_oracle",
            "oracle",
            "passed",
            "finite",
            "nonzero_count",
            "gates",
            "read_only_inputs_immutable",
            "read_only_inputs_sha256",
            "output_sha256",
        },
        location,
    )
    _require(
        correctness["independent_oracle"] is True,
        f"{location}: oracle is not independent",
    )
    oracle = correctness["oracle"]
    _require(
        isinstance(oracle, str)
        and oracle
        and "arm" not in oracle.lower()
        and "equal" not in oracle.lower(),
        f"{location}: invalid independent oracle name",
    )
    _require(
        correctness["passed"] is True and correctness["finite"] is True,
        f"{location}: correctness did not pass",
    )
    _positive_int(correctness["nonzero_count"], f"{location}.nonzero_count")
    gates = correctness["gates"]
    _require(
        isinstance(gates, dict)
        and all(
            isinstance(name, str) and isinstance(passed, bool)
            for name, passed in gates.items()
        ),
        f"{location}: invalid correctness-gate map",
    )
    missing = sorted(set(required_gates) - set(gates))
    failed = sorted(name for name in required_gates if gates.get(name) is not True)
    _require(
        not missing and not failed,
        f"{location}: missing={missing!r}, failed={failed!r}",
    )
    _require(
        correctness["read_only_inputs_immutable"] is True,
        f"{location}: read-only inputs changed",
    )
    for field in ("read_only_inputs_sha256", "output_sha256"):
        _require(_is_sha256(correctness[field]), f"{location}: invalid {field}")
    return correctness


def _validate_artifact_verification(
    value: object,
    *,
    location: str,
    manifest_sha256: str,
    object_sha256: str,
    object_bytes: int,
) -> dict[str, Any]:
    verification = _exact_keys(
        value,
        {"passed", "manifest_sha256", "object_sha256", "object_bytes"},
        location,
    )
    _require(verification["passed"] is True, f"{location}: verification did not pass")
    _require(
        verification["manifest_sha256"] == manifest_sha256
        and verification["object_sha256"] == object_sha256
        and verification["object_bytes"] == object_bytes,
        f"{location}: verification differs from exact artifact identity",
    )
    return verification


def _validate_exact_artifact(
    value: object,
    *,
    location: str,
    expected_package_fingerprint: str,
    compile_contract: dict[str, Any],
) -> dict[str, Any]:
    artifact = _exact_keys(
        value,
        {
            "cache_root",
            "cache_key",
            "manifest_path",
            "manifest_sha256",
            "frontend_ptx_sidecar_path",
            "frontend_ptx_sidecar_sha256",
            "object_path",
            "object_sha256",
            "object_bytes",
            "compile_spec_hash",
            "compile_spec_json",
            "semantic_key",
            "kernel_id",
            "package_fingerprint",
            "toolchain",
            "toolchain_sha256",
            "verification_before",
            "verification_after",
        },
        location,
    )
    for field in (
        "cache_key",
        "manifest_sha256",
        "frontend_ptx_sidecar_sha256",
        "object_sha256",
        "compile_spec_hash",
        "semantic_key",
        "package_fingerprint",
        "toolchain_sha256",
    ):
        _require(_is_sha256(artifact[field]), f"{location}: invalid {field}")
    cache_root = Path(str(artifact["cache_root"]))
    manifest_path = Path(str(artifact["manifest_path"]))
    sidecar_path = Path(str(artifact["frontend_ptx_sidecar_path"]))
    object_path = Path(str(artifact["object_path"]))
    _require(
        all(
            path.is_absolute()
            for path in (cache_root, manifest_path, sidecar_path, object_path)
        ),
        f"{location}: cache artifact paths must be absolute",
    )
    cache_key = str(artifact["cache_key"])
    expected_stem = cache_root / cache_key[:2] / cache_key
    _require(
        manifest_path == expected_stem.with_suffix(".json")
        and sidecar_path == expected_stem.with_suffix(".ptx.json")
        and object_path == expected_stem.with_suffix(".o"),
        f"{location}: cache-key/path binding mismatch",
    )
    object_bytes = _positive_int(artifact["object_bytes"], f"{location}.object_bytes")
    compile_spec_json = artifact["compile_spec_json"]
    _require(
        isinstance(compile_spec_json, str) and bool(compile_spec_json),
        f"{location}: compile-spec JSON is empty",
    )
    try:
        parsed_compile_spec = json.loads(compile_spec_json)
    except json.JSONDecodeError as exc:
        _fail(f"{location}: compile-spec JSON is invalid: {exc}")
    _require(
        isinstance(parsed_compile_spec, dict),
        f"{location}: compile spec is not an object",
    )
    _require(
        artifact["compile_spec_hash"]
        == hashlib.sha256(compile_spec_json.encode()).hexdigest(),
        f"{location}: compile-spec hash does not match its exact JSON",
    )
    _require(
        artifact["kernel_id"] == compile_contract["kernel_id"]
        and artifact["compile_spec_hash"] == compile_contract["compile_spec_hash"]
        and artifact["compile_spec_json"] == compile_contract["compile_spec_json"],
        f"{location}: exact object differs from the reviewed compile contract",
    )
    _require(
        artifact["package_fingerprint"] == expected_package_fingerprint,
        f"{location}: exact object package fingerprint differs from frozen source",
    )
    toolchain = artifact["toolchain"]
    _require(
        isinstance(toolchain, (dict, list)) and bool(toolchain),
        f"{location}: exact object toolchain is empty",
    )
    _require(
        artifact["toolchain_sha256"] == _canonical_sha256(toolchain),
        f"{location}: exact object toolchain hash mismatch",
    )
    before = _validate_artifact_verification(
        artifact["verification_before"],
        location=f"{location}.verification_before",
        manifest_sha256=str(artifact["manifest_sha256"]),
        object_sha256=str(artifact["object_sha256"]),
        object_bytes=object_bytes,
    )
    after = _validate_artifact_verification(
        artifact["verification_after"],
        location=f"{location}.verification_after",
        manifest_sha256=str(artifact["manifest_sha256"]),
        object_sha256=str(artifact["object_sha256"]),
        object_bytes=object_bytes,
    )
    _require(before == after, f"{location}: exact artifact changed during timing")
    return artifact


def _validate_case_artifacts(
    artifacts_value: object,
    launch_plan_value: object,
    source_owned_value: object,
    *,
    location: str,
    expected_package_fingerprint: str,
    compile_contract: dict[str, Any],
    graph_kernel_node_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_artifacts = artifacts_value
    _require(
        isinstance(raw_artifacts, list) and raw_artifacts,
        f"{location}.artifacts: exact artifact list is empty",
    )
    expected_artifacts = compile_contract["artifacts"]
    expected_by_role = {
        str(artifact["role"]): artifact for artifact in expected_artifacts
    }
    artifacts: list[dict[str, Any]] = []
    by_role: dict[str, dict[str, Any]] = {}
    cache_object_identities: set[tuple[str, str]] = set()
    for index, raw_binding in enumerate(raw_artifacts):
        binding_location = f"{location}.artifacts[{index}]"
        binding = _exact_keys(
            raw_binding,
            {
                "role",
                "kernel_id",
                "compile_spec_hash",
                "object_sha256",
                "evidence",
            },
            binding_location,
        )
        role = binding["role"]
        _require(
            isinstance(role, str) and role in expected_by_role,
            f"{binding_location}: unreviewed artifact role",
        )
        _require(role not in by_role, f"{binding_location}: duplicate artifact role")
        expected = expected_by_role[role]
        evidence = _validate_exact_artifact(
            binding["evidence"],
            location=f"{binding_location}.evidence",
            expected_package_fingerprint=expected_package_fingerprint,
            compile_contract=expected,
        )
        _require(
            binding["kernel_id"] == evidence["kernel_id"]
            and binding["compile_spec_hash"] == evidence["compile_spec_hash"]
            and binding["object_sha256"] == evidence["object_sha256"],
            f"{binding_location}: explicit artifact identity differs from evidence",
        )
        cache_object_identity = (
            str(evidence["cache_key"]),
            str(evidence["object_sha256"]),
        )
        _require(
            cache_object_identity not in cache_object_identities,
            f"{binding_location}: duplicate exact cache object",
        )
        cache_object_identities.add(cache_object_identity)
        normalized = {**binding, "evidence": evidence}
        by_role[str(role)] = normalized
        artifacts.append(normalized)
    _require(
        set(by_role) == set(expected_by_role),
        f"{location}: missing or unused exact artifacts; "
        f"missing={sorted(set(expected_by_role) - set(by_role))!r}, "
        f"unexpected={sorted(set(by_role) - set(expected_by_role))!r}",
    )

    raw_plan = launch_plan_value
    _require(
        isinstance(raw_plan, list) and raw_plan,
        f"{location}.launch_plan: observed launch plan is empty",
    )
    launch_plan: list[dict[str, Any]] = []
    normalized_plan: list[dict[str, Any]] = []
    next_multiplicity: dict[str, int] = {}
    used_roles: set[str] = set()
    previous_node_index = -1
    for index, raw_binding in enumerate(raw_plan):
        binding_location = f"{location}.launch_plan[{index}]"
        binding = _exact_keys(
            raw_binding,
            {
                "node_index",
                "artifact_role",
                "kernel_id",
                "compile_spec_hash",
                "object_sha256",
                "multiplicity_index",
            },
            binding_location,
        )
        node_index = _nonnegative_int(
            binding["node_index"], f"{binding_location}.node_index"
        )
        _require(
            node_index < graph_kernel_node_count and node_index > previous_node_index,
            f"{binding_location}: exact node ordinals must be in-range and "
            "strictly increasing",
        )
        previous_node_index = node_index
        role = binding["artifact_role"]
        _require(
            isinstance(role, str) and role in by_role,
            f"{binding_location}: launch references an unknown artifact role",
        )
        artifact = by_role[role]
        _require(
            binding["kernel_id"] == artifact["kernel_id"]
            and binding["compile_spec_hash"] == artifact["compile_spec_hash"]
            and binding["object_sha256"] == artifact["object_sha256"],
            f"{binding_location}: launch does not bind its exact artifact object",
        )
        expected_multiplicity = next_multiplicity.get(role, 0) + 1
        _require(
            binding["multiplicity_index"] == expected_multiplicity,
            f"{binding_location}: launch multiplicity is not exact",
        )
        next_multiplicity[role] = expected_multiplicity
        used_roles.add(role)
        launch_plan.append(binding)
        normalized_plan.append(
            {
                key: binding[key]
                for key in (
                    "node_index",
                    "artifact_role",
                    "kernel_id",
                    "compile_spec_hash",
                    "multiplicity_index",
                )
            }
        )
    _require(
        used_roles == set(by_role),
        f"{location}.launch_plan: exact artifact is unused by the graph",
    )
    _require(
        normalized_plan == compile_contract["launch_plan"],
        f"{location}.launch_plan: observed launch order or multiplicity differs from review",
    )
    source_owned = _validate_source_owned_kernel_nodes(
        source_owned_value,
        location=f"{location}.source_owned_kernel_nodes",
        kernel_node_count=graph_kernel_node_count,
    )
    _require(
        source_owned == compile_contract["source_owned_kernel_nodes"],
        f"{location}.source_owned_kernel_nodes: observed source-owned nodes "
        "differ from review",
    )
    covered_indices = sorted(
        [int(binding["node_index"]) for binding in launch_plan]
        + [int(record["node_index"]) for record in source_owned]
    )
    _require(
        covered_indices == list(range(graph_kernel_node_count)),
        f"{location}: exact and source-owned nodes do not partition the graph; "
        f"observed={covered_indices!r}",
    )
    return artifacts, launch_plan, source_owned


def _validate_graph(value: object, location: str) -> dict[str, Any]:
    graph = _exact_keys(
        value,
        {
            "capture_passed",
            "replay_passed",
            "topology_stable",
            "addresses_stable",
            "live_input_changed_output",
            "poison_overwrite_passed",
            "node_count",
            "kernel_node_count",
            "topology_sha256",
        },
        location,
    )
    for field in (
        "capture_passed",
        "replay_passed",
        "topology_stable",
        "addresses_stable",
        "live_input_changed_output",
        "poison_overwrite_passed",
    ):
        _require(graph[field] is True, f"{location}: graph gate {field} did not pass")
    node_count = _positive_int(graph["node_count"], f"{location}.node_count")
    kernel_count = _positive_int(
        graph["kernel_node_count"], f"{location}.kernel_node_count"
    )
    _require(
        kernel_count <= node_count, f"{location}: kernel node count exceeds graph nodes"
    )
    _require(_is_sha256(graph["topology_sha256"]), f"{location}: invalid topology SHA")
    return graph


def _validate_allocation(value: object, location: str) -> dict[str, Any]:
    allocation = _exact_keys(
        value,
        {
            "fixed_workspace_capacity",
            "workspace_capacity_bytes",
            "stable_addresses",
            "allocator_stable",
            "zero_replay_allocations",
            "allocated_bytes_before",
            "allocated_bytes_after",
            "reserved_bytes_before",
            "reserved_bytes_after",
            "condition_counters",
        },
        location,
    )
    for field in (
        "fixed_workspace_capacity",
        "stable_addresses",
        "allocator_stable",
        "zero_replay_allocations",
    ):
        _require(
            allocation[field] is True, f"{location}: allocation gate {field} failed"
        )
    for field in (
        "workspace_capacity_bytes",
        "allocated_bytes_before",
        "allocated_bytes_after",
        "reserved_bytes_before",
        "reserved_bytes_after",
    ):
        _nonnegative_int(allocation[field], f"{location}.{field}")
    _require(
        allocation["allocated_bytes_before"] == allocation["allocated_bytes_after"]
        and allocation["reserved_bytes_before"] == allocation["reserved_bytes_after"],
        f"{location}: allocator counters changed during timing",
    )
    condition_counters = _exact_keys(
        allocation["condition_counters"],
        {"warm_l2", "cold_l2"},
        f"{location}.condition_counters",
    )
    validated_counters: dict[str, dict[str, Any]] = {}
    counter_fields = {
        "allocated_bytes_before",
        "allocated_bytes_after",
        "reserved_bytes_before",
        "reserved_bytes_after",
    }
    for condition in ("warm_l2", "cold_l2"):
        counters = _exact_keys(
            condition_counters[condition],
            counter_fields,
            f"{location}.condition_counters.{condition}",
        )
        for field in counter_fields:
            _nonnegative_int(
                counters[field],
                f"{location}.condition_counters.{condition}.{field}",
            )
        _require(
            counters["allocated_bytes_before"] == counters["allocated_bytes_after"]
            and counters["reserved_bytes_before"] == counters["reserved_bytes_after"],
            f"{location}: {condition} allocator counters changed during replay",
        )
        validated_counters[condition] = counters
    _require(
        all(
            allocation[field] == validated_counters["warm_l2"][field]
            for field in counter_fields
        ),
        f"{location}: summary counters differ from warm-L2 timing counters",
    )
    allocation["condition_counters"] = validated_counters
    return allocation


def _validate_conditions(
    value: object,
    *,
    location: str,
    l2_cache_bytes: int,
    physical_gpu: int,
    gpu_uuid: str,
) -> dict[str, dict[str, Any]]:
    conditions = _exact_keys(value, {"warm_l2", "cold_l2"}, location)
    validated: dict[str, dict[str, Any]] = {}
    observed_required_throttle_reasons: set[int] = set()
    for name in ("warm_l2", "cold_l2"):
        condition = _exact_keys(
            conditions[name],
            {
                "l2_flushed",
                "l2_flush_bytes",
                "preconditioning",
                "event_pool",
                "gpu_mode_before_timing",
                "gpu_mode_after_timing",
                "gpu_mode_stability",
                "replays_per_reported_sample",
                "aggregation",
                "inner_samples_us",
                "inner_sample_count",
                "samples_us",
            },
            f"{location}.{name}",
        )
        expected_flush = name == "cold_l2"
        _require(
            condition["l2_flushed"] is expected_flush,
            f"{location}.{name}: incorrect L2 condition marker",
        )
        flush_bytes = _nonnegative_int(
            condition["l2_flush_bytes"], f"{location}.{name}.l2_flush_bytes"
        )
        if expected_flush:
            _require(
                flush_bytes >= math.ceil(MINIMUM_L2_FLUSH_MULTIPLE * l2_cache_bytes),
                f"{location}.{name}: L2 eviction is too small",
            )
        else:
            _require(
                flush_bytes == 0, f"{location}.{name}: warm condition has an L2 flush"
            )

        preconditioning = _exact_keys(
            condition["preconditioning"],
            {
                "policy",
                "minimum_replays",
                "minimum_active_seconds",
                "maximum_active_seconds",
                "completed_replays",
                "observed_active_seconds",
                "target_graph_replays",
                "cold_l2_flush_before_every_replay",
                "flush_inside_timed_interval",
                "required_pstate",
                "required_active_throttle_reasons",
                "mode_probes",
            },
            f"{location}.{name}.preconditioning",
        )
        minimum_replays = _positive_int(
            preconditioning["minimum_replays"],
            f"{location}.{name}.preconditioning.minimum_replays",
        )
        completed_replays = _positive_int(
            preconditioning["completed_replays"],
            f"{location}.{name}.preconditioning.completed_replays",
        )
        target_replays = _positive_int(
            preconditioning["target_graph_replays"],
            f"{location}.{name}.preconditioning.target_graph_replays",
        )
        required_throttle_reasons = _required_active_throttle_reasons(
            preconditioning["required_active_throttle_reasons"],
            location=(
                f"{location}.{name}.preconditioning.required_active_throttle_reasons"
            ),
        )
        observed_required_throttle_reasons.add(required_throttle_reasons)
        _require(
            preconditioning["policy"] == "single_exact_target_graph_duration"
            and minimum_replays >= 2_000
            and completed_replays >= minimum_replays
            and target_replays == completed_replays
            and preconditioning["cold_l2_flush_before_every_replay"] is expected_flush
            and preconditioning["flush_inside_timed_interval"] is False
            and preconditioning["required_pstate"] == REQUIRED_TIMING_PSTATE,
            f"{location}.{name}: invalid target-graph preconditioning policy",
        )
        duration: dict[str, float] = {}
        for field in (
            "minimum_active_seconds",
            "maximum_active_seconds",
            "observed_active_seconds",
        ):
            raw = preconditioning[field]
            _require(
                isinstance(raw, (int, float))
                and not isinstance(raw, bool)
                and math.isfinite(float(raw))
                and float(raw) > 0.0,
                f"{location}.{name}.preconditioning.{field}: invalid duration",
            )
            duration[field] = float(raw)
        _require(
            duration["minimum_active_seconds"] >= MINIMUM_PRECONDITION_SECONDS
            and duration["maximum_active_seconds"] <= MAXIMUM_PRECONDITION_SECONDS
            and duration["minimum_active_seconds"]
            <= duration["observed_active_seconds"]
            <= duration["maximum_active_seconds"],
            f"{location}.{name}: preconditioning duration envelope is invalid",
        )
        mode_probes = preconditioning["mode_probes"]
        _require(
            isinstance(mode_probes, list) and mode_probes,
            f"{location}.{name}: preconditioning has no GPU-mode probe",
        )
        normalized_probes = [
            _validate_gpu_mode_snapshot(
                probe,
                location=f"{location}.{name}.preconditioning.mode_probes[{index}]",
                physical_gpu=physical_gpu,
                gpu_uuid=gpu_uuid,
            )
            for index, probe in enumerate(mode_probes)
        ]
        final_probe = normalized_probes[-1]
        final_probe_throttle_reasons = _active_throttle_reasons(
            final_probe,
            location=f"{location}.{name}.preconditioning.mode_probes[-1]",
        )
        _require(
            final_probe["fields"]["pstate"] == REQUIRED_TIMING_PSTATE
            and final_probe_throttle_reasons == required_throttle_reasons,
            f"{location}.{name}: final preconditioning probe does not match the "
            "required P1/throttle-mask policy",
        )

        mode_before = _validate_gpu_mode_snapshot(
            condition["gpu_mode_before_timing"],
            location=f"{location}.{name}.gpu_mode_before_timing",
            physical_gpu=physical_gpu,
            gpu_uuid=gpu_uuid,
        )
        mode_after = _validate_gpu_mode_snapshot(
            condition["gpu_mode_after_timing"],
            location=f"{location}.{name}.gpu_mode_after_timing",
            physical_gpu=physical_gpu,
            gpu_uuid=gpu_uuid,
        )
        _require(
            int(final_probe["captured_unix_ns"])
            < int(mode_before["captured_unix_ns"])
            < int(mode_after["captured_unix_ns"]),
            f"{location}.{name}: precondition/timing mode timestamps are not ordered",
        )
        before_throttle_reasons = _active_throttle_reasons(
            mode_before, location=f"{location}.{name}.gpu_mode_before_timing"
        )
        after_throttle_reasons = _active_throttle_reasons(
            mode_after, location=f"{location}.{name}.gpu_mode_after_timing"
        )
        _require(
            mode_before["fields"]["pstate"] == REQUIRED_TIMING_PSTATE
            and mode_after["fields"]["pstate"] == REQUIRED_TIMING_PSTATE
            and before_throttle_reasons == required_throttle_reasons
            and after_throttle_reasons == required_throttle_reasons
            and all(
                mode_before["fields"][field] == mode_after["fields"][field]
                for field in _TIMING_STABLE_MODE_FIELDS
            ),
            f"{location}.{name}: timing did not remain in the required stable "
            "P1/throttle-mask policy",
        )
        before_sm = _timing_clock_mhz(
            mode_before,
            "clocks.current.sm",
            location=f"{location}.{name}.gpu_mode_before_timing",
        )
        after_sm = _timing_clock_mhz(
            mode_after,
            "clocks.current.sm",
            location=f"{location}.{name}.gpu_mode_after_timing",
        )
        before_memory = _timing_clock_mhz(
            mode_before,
            "clocks.current.memory",
            location=f"{location}.{name}.gpu_mode_before_timing",
        )
        after_memory = _timing_clock_mhz(
            mode_after,
            "clocks.current.memory",
            location=f"{location}.{name}.gpu_mode_after_timing",
        )
        stability = _exact_keys(
            condition["gpu_mode_stability"],
            {
                "schema",
                "required_pstate",
                "required_memory_clock_equality",
                "max_sm_clock_delta_mhz",
                "observed_sm_clock_delta_mhz",
                "observed_before_sm_clock_mhz",
                "observed_after_sm_clock_mhz",
                "observed_memory_clock_mhz",
                "required_active_throttle_reasons",
                "observed_before_active_throttle_reasons",
                "observed_after_active_throttle_reasons",
                "stable_identity_and_mode_fields",
                "passed",
            },
            f"{location}.{name}.gpu_mode_stability",
        )
        configured_sm_delta = stability["max_sm_clock_delta_mhz"]
        _require(
            isinstance(configured_sm_delta, (int, float))
            and not isinstance(configured_sm_delta, bool)
            and 0.0 < float(configured_sm_delta) <= MAX_TIMING_SM_CLOCK_DELTA_MHZ,
            f"{location}.{name}: invalid SM-clock stability ceiling",
        )
        observed_sm_delta = abs(after_sm - before_sm)
        expected_stability_values = {
            "observed_sm_clock_delta_mhz": observed_sm_delta,
            "observed_before_sm_clock_mhz": before_sm,
            "observed_after_sm_clock_mhz": after_sm,
            "observed_memory_clock_mhz": before_memory,
        }
        _require(
            stability["schema"] == GPU_MODE_STABILITY_SCHEMA
            and stability["required_pstate"] == REQUIRED_TIMING_PSTATE
            and stability["required_memory_clock_equality"] is True
            and stability["required_active_throttle_reasons"]
            == required_throttle_reasons
            and stability["observed_before_active_throttle_reasons"]
            == before_throttle_reasons
            and stability["observed_after_active_throttle_reasons"]
            == after_throttle_reasons
            and stability["stable_identity_and_mode_fields"]
            == list(_TIMING_STABLE_MODE_FIELDS)
            and stability["passed"] is True
            and observed_sm_delta <= float(configured_sm_delta)
            and before_memory == after_memory
            and all(
                isinstance(stability[field], (int, float))
                and not isinstance(stability[field], bool)
                and math.isclose(
                    float(stability[field]), expected, rel_tol=1e-12, abs_tol=1e-12
                )
                for field, expected in expected_stability_values.items()
            ),
            f"{location}.{name}: GPU mode-stability evidence is inconsistent",
        )

        replays_per_sample = _positive_int(
            condition["replays_per_reported_sample"],
            f"{location}.{name}.replays_per_reported_sample",
        )
        aggregation = _exact_keys(
            condition["aggregation"],
            {
                "reported_sample",
                "inner_event_bracketing",
                "inner_schedule",
                "flush_before_every_inner_replay",
                "flush_inside_timed_interval",
            },
            f"{location}.{name}.aggregation",
        )
        _require(
            aggregation
            == {
                "reported_sample": "arithmetic_mean_us",
                "inner_event_bracketing": "independent_per_graph_replay",
                "inner_schedule": "same_exact_graph_replay_per_repetition",
                "flush_before_every_inner_replay": expected_flush,
                "flush_inside_timed_interval": False,
            },
            f"{location}.{name}: aggregate timing policy differs from review",
        )
        samples = condition["samples_us"]
        _require(
            isinstance(samples, list)
            and len(samples) >= MINIMUM_SAMPLES_PER_PROCESS_CONDITION,
            f"{location}.{name}: requires at least "
            f"{MINIMUM_SAMPLES_PER_PROCESS_CONDITION} raw samples",
        )
        normalized: list[float] = []
        for index, sample in enumerate(samples):
            _require(
                isinstance(sample, (int, float))
                and not isinstance(sample, bool)
                and math.isfinite(float(sample))
                and float(sample) > 0,
                f"{location}.{name}.samples_us[{index}]: invalid sample",
            )
            normalized.append(float(sample))
        raw_groups = condition["inner_samples_us"]
        _require(
            isinstance(raw_groups, list) and len(raw_groups) == len(normalized),
            f"{location}.{name}: inner timing group count differs from reported samples",
        )
        normalized_raw: list[list[float]] = []
        for sample_index, raw_group in enumerate(raw_groups):
            _require(
                isinstance(raw_group, list) and len(raw_group) == replays_per_sample,
                f"{location}.{name}.inner_samples_us[{sample_index}]: "
                "inner replay count differs",
            )
            normalized_group: list[float] = []
            for inner_index, raw_sample in enumerate(raw_group):
                _require(
                    isinstance(raw_sample, (int, float))
                    and not isinstance(raw_sample, bool)
                    and math.isfinite(float(raw_sample))
                    and float(raw_sample) > 0,
                    f"{location}.{name}.inner_samples_us[{sample_index}]"
                    f"[{inner_index}]: invalid inner sample",
                )
                normalized_group.append(float(raw_sample))
            reconstructed = math.fsum(normalized_group) / replays_per_sample
            _require(
                math.isclose(
                    reconstructed,
                    normalized[sample_index],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                ),
                f"{location}.{name}.samples_us[{sample_index}]: "
                "reported sample is not its inner arithmetic mean",
            )
            normalized_raw.append(normalized_group)
        inner_sample_count = _positive_int(
            condition["inner_sample_count"],
            f"{location}.{name}.inner_sample_count",
        )
        _require(
            inner_sample_count == len(normalized) * replays_per_sample,
            f"{location}.{name}: declared inner timing count differs",
        )
        event_pool = _exact_keys(
            condition["event_pool"],
            {
                "schema",
                "allocation_phase",
                "prewarm_phase",
                "prewarm_each_event",
                "one_pair_per_inner_replay",
                "event_creation_inside_sample_schedule",
                "initialized_before_target_graph_preconditioning",
                "reuse_boundary",
                "event_batch_replays",
                "batch_replay_capacity",
                "pair_count",
                "event_count",
                "unique_event_handle_count",
                "event_handle_sha256",
                "prewarm_elapsed_query_count",
                "prewarm_elapsed_sha256",
                "batch_count",
                "reuse_count",
            },
            f"{location}.{name}.event_pool",
        )
        event_batch_replays = _positive_int(
            event_pool["event_batch_replays"],
            f"{location}.{name}.event_pool.event_batch_replays",
        )
        batch_replay_capacity = min(len(normalized), event_batch_replays)
        pair_count = batch_replay_capacity * replays_per_sample
        batch_count = math.ceil(len(normalized) / event_batch_replays)
        expected_pool_counts = {
            "batch_replay_capacity": batch_replay_capacity,
            "pair_count": pair_count,
            "event_count": 2 * pair_count,
            "unique_event_handle_count": 2 * pair_count,
            "prewarm_elapsed_query_count": pair_count,
            "batch_count": batch_count,
            "reuse_count": max(0, batch_count - 1),
        }
        _require(
            event_pool["schema"] == CUDA_EVENT_POOL_SCHEMA
            and event_pool["allocation_phase"] == "before_reported_samples"
            and event_pool["prewarm_phase"] == "before_reported_samples"
            and event_pool["prewarm_each_event"] is True
            and event_pool["one_pair_per_inner_replay"] is True
            and event_pool["event_creation_inside_sample_schedule"] is False
            and event_pool["initialized_before_target_graph_preconditioning"] is True
            and event_pool["reuse_boundary"]
            == "after_stream_synchronize_and_elapsed_query"
            and _is_sha256(event_pool["event_handle_sha256"])
            and _is_sha256(event_pool["prewarm_elapsed_sha256"])
            and all(
                event_pool[field] == expected
                for field, expected in expected_pool_counts.items()
            ),
            f"{location}.{name}: CUDA event-pool evidence is inconsistent",
        )
        validated[name] = {
            **condition,
            "preconditioning": preconditioning,
            "event_pool": event_pool,
            "gpu_mode_before_timing": mode_before,
            "gpu_mode_after_timing": mode_after,
            "gpu_mode_stability": stability,
            "inner_samples_us": normalized_raw,
            "samples_us": normalized,
        }
    _require(
        len(observed_required_throttle_reasons) == 1,
        f"{location}: warm/cold timing conditions request different active "
        "throttle-reason masks",
    )
    return validated


def _validate_run(
    path: Path,
    *,
    family: str,
    physical_gpu: int,
    position: str,
    contract: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    loaded = _load_json(path)
    _require(
        loaded.get("evidence_status") == "final-source",
        f"{path}: explicit final-source evidence status is required; "
        "diagnostic, missing, or non-final evidence cannot enter the release index",
    )
    run = _exact_keys(
        loaded,
        {
            "schema",
            "family",
            "arm",
            "sequence_position",
            "evidence_status",
            "invocation",
            "source",
            "producer",
            "harness_case_contract_sha256",
            "cutlass_packages",
            "runtime",
            "gpu",
            "cases",
            "result_sha256",
        },
        str(path),
    )
    _require(run["schema"] == RUN_SCHEMA, f"{path}: invalid process-result schema")
    _require(run["family"] == family, f"{path}: family differs from evidence set")
    expected_arm = POSITION_ARM[position]
    _require(run["arm"] == expected_arm, f"{path}: wrong arm for {position}")
    _require(run["sequence_position"] == position, f"{path}: wrong sequence position")
    payload = {key: value for key, value in run.items() if key != "result_sha256"}
    _require(
        run["result_sha256"] == _canonical_sha256(payload),
        f"{path}: result SHA mismatch",
    )

    invocation = _exact_keys(
        run["invocation"],
        {
            "process_id",
            "pid",
            "started_unix_ns",
            "finished_unix_ns",
            "command",
            "worktree",
        },
        f"{path}.invocation",
    )
    _require(_is_sha256(invocation["process_id"]), f"{path}: invalid process identity")
    _positive_int(invocation["pid"], f"{path}.invocation.pid")
    started = _positive_int(
        invocation["started_unix_ns"], f"{path}.invocation.started_unix_ns"
    )
    finished = _positive_int(
        invocation["finished_unix_ns"], f"{path}.invocation.finished_unix_ns"
    )
    _require(finished > started, f"{path}: process timestamps are not ordered")
    _require(
        isinstance(invocation["command"], list)
        and invocation["command"]
        and all(isinstance(item, str) and item for item in invocation["command"]),
        f"{path}: command provenance is invalid",
    )
    _require(
        isinstance(invocation["worktree"], str) and invocation["worktree"],
        f"{path}: worktree provenance is empty",
    )

    source = _exact_keys(
        run["source"],
        {
            "manifest_sha256",
            "manifest_artifact_sha256",
            "production_fingerprint",
            "runtime_package_fingerprint",
        },
        f"{path}.source",
    )
    expected_source = sources[expected_arm]
    _require(
        source["manifest_sha256"] == expected_source["manifest_sha256"]
        and source["manifest_artifact_sha256"]
        == expected_source["manifest_artifact_sha256"]
        and source["production_fingerprint"]
        == expected_source["production_fingerprint"]
        and source["runtime_package_fingerprint"]
        == expected_source["runtime_package_fingerprint"],
        f"{path}: process is not bound to the frozen {expected_arm} production source",
    )
    _require(
        str(Path(invocation["worktree"]).resolve())
        == expected_source["manifest"]["runtime"]["repo_root"],
        f"{path}: invocation worktree differs from the frozen evidence runtime",
    )
    producer = _exact_keys(
        run["producer"],
        {"path", "sha256"},
        f"{path}.producer",
    )
    family_contract = contract["families"][family]
    _require(
        producer["path"] == family_contract["producer"]
        and producer["sha256"] == family_contract["producer_sha256"],
        f"{path}: process producer differs from the frozen family harness",
    )
    _require(
        run["harness_case_contract_sha256"] == contract["sha256"],
        f"{path}: process used a different harness/case contract",
    )
    _require(
        run["cutlass_packages"]
        == contract["arm_toolchains"][expected_arm]["cutlass_packages"],
        f"{path}: CUTLASS package map is not exact for {expected_arm}",
    )
    runtime = _validate_runtime(run["runtime"], f"{path}.runtime")
    _require(
        runtime["ptxas_version"]
        == contract["arm_toolchains"][expected_arm]["ptxas_version"],
        f"{path}: ptxas version differs from the reviewed {expected_arm} toolchain",
    )

    gpu = _exact_keys(
        run["gpu"],
        {
            "physical_ordinal",
            "name",
            "uuid",
            "capability",
            "l2_cache_bytes",
            "mode_before",
            "mode_after",
        },
        f"{path}.gpu",
    )
    _require(gpu["physical_ordinal"] == physical_gpu, f"{path}: physical GPU mismatch")
    _require(isinstance(gpu["name"], str) and gpu["name"], f"{path}: GPU name is empty")
    _require(_normalize_uuid(gpu["uuid"]), f"{path}: GPU UUID is empty")
    _require(gpu["capability"] == [12, 0], f"{path}: expected SM120 capability")
    l2_cache_bytes = _positive_int(gpu["l2_cache_bytes"], f"{path}.gpu.l2_cache_bytes")
    before = _validate_gpu_mode_snapshot(
        gpu["mode_before"],
        location=f"{path}.gpu.mode_before",
        physical_gpu=physical_gpu,
        gpu_uuid=str(gpu["uuid"]),
    )
    after = _validate_gpu_mode_snapshot(
        gpu["mode_after"],
        location=f"{path}.gpu.mode_after",
        physical_gpu=physical_gpu,
        gpu_uuid=str(gpu["uuid"]),
    )
    _require(
        after["captured_unix_ns"] > before["captured_unix_ns"],
        f"{path}: GPU mode timestamps are not ordered",
    )
    _require(
        all(
            before["fields"][field] == after["fields"][field]
            for field in _GPU_STABLE_FIELDS
        ),
        f"{path}: stable GPU mode changed during the process",
    )

    cases = run["cases"]
    expected_cases = family_contract["case_by_id"]
    _require(isinstance(cases, list), f"{path}: cases are not a list")
    by_case: dict[str, dict[str, Any]] = {}
    run_timing_mode_policies: set[tuple[str, int, float]] = set()
    for index, raw_case in enumerate(cases):
        location = f"{path}.cases[{index}]"
        case = _exact_keys(
            raw_case,
            {
                "case_id",
                "case_contract_sha256",
                "input_sha256",
                "artifacts",
                "launch_plan",
                "source_owned_kernel_nodes",
                "correctness",
                "graph",
                "allocation",
                "conditions",
            },
            location,
        )
        case_id = case["case_id"]
        _require(
            case_id in expected_cases, f"{location}: unreviewed case_id {case_id!r}"
        )
        _require(case_id not in by_case, f"{location}: duplicate case_id")
        reviewed = expected_cases[case_id]
        _require(
            case["case_contract_sha256"] == reviewed["case_contract_sha256"]
            and case["input_sha256"] == reviewed["input_sha256"],
            f"{location}: case/input contract differs from review",
        )
        graph = _validate_graph(case["graph"], f"{location}.graph")
        artifacts, launch_plan, source_owned_kernel_nodes = _validate_case_artifacts(
            case["artifacts"],
            case["launch_plan"],
            case["source_owned_kernel_nodes"],
            location=location,
            expected_package_fingerprint=str(source["runtime_package_fingerprint"]),
            compile_contract=reviewed["compile_artifact_contract"][expected_arm],
            graph_kernel_node_count=int(graph["kernel_node_count"]),
        )
        correctness = _validate_correctness(
            case["correctness"],
            location=f"{location}.correctness",
            required_gates=reviewed["required_correctness_gates"],
        )
        allocation = _validate_allocation(case["allocation"], f"{location}.allocation")
        conditions = _validate_conditions(
            case["conditions"],
            location=f"{location}.conditions",
            l2_cache_bytes=l2_cache_bytes,
            physical_gpu=physical_gpu,
            gpu_uuid=str(gpu["uuid"]),
        )
        for condition in conditions.values():
            preconditioning = condition["preconditioning"]
            stability = condition["gpu_mode_stability"]
            run_timing_mode_policies.add(
                (
                    str(preconditioning["required_pstate"]),
                    int(preconditioning["required_active_throttle_reasons"]),
                    float(stability["max_sm_clock_delta_mhz"]),
                )
            )
        by_case[str(case_id)] = {
            **case,
            "artifacts": artifacts,
            "launch_plan": launch_plan,
            "source_owned_kernel_nodes": source_owned_kernel_nodes,
            "correctness": correctness,
            "graph": graph,
            "allocation": allocation,
            "conditions": conditions,
        }
    _require(
        set(by_case) == set(expected_cases),
        f"{path}: incomplete case coverage; "
        f"missing={sorted(set(expected_cases) - set(by_case))!r}, "
        f"unexpected={sorted(set(by_case) - set(expected_cases))!r}",
    )
    _require(
        len(run_timing_mode_policies) == 1,
        f"{path}: cases do not bind one consistent GPU timing-mode policy",
    )
    return {
        "path": path,
        "value": run,
        "invocation": invocation,
        "runtime": runtime,
        "gpu": gpu,
        "cases": by_case,
        "artifact": _artifact(path, schema=RUN_SCHEMA),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio(numerator: float, denominator: float) -> float:
    _require(denominator > 0, "timing denominator is not positive")
    return numerator / denominator


def _performance_row(
    *,
    family: str,
    case_id: str,
    physical_gpu: int,
    condition: str,
    runs: dict[str, dict[str, Any]],
    topology_disposition: str,
) -> dict[str, object]:
    timing_mode_policies = {
        (
            str(
                runs[position]["cases"][case_id]["conditions"][condition][
                    "preconditioning"
                ]["required_pstate"]
            ),
            int(
                runs[position]["cases"][case_id]["conditions"][condition][
                    "preconditioning"
                ]["required_active_throttle_reasons"]
            ),
            float(
                runs[position]["cases"][case_id]["conditions"][condition][
                    "gpu_mode_stability"
                ]["max_sm_clock_delta_mhz"]
            ),
        )
        for position in SEQUENCE
    }
    _require(
        len(timing_mode_policies) == 1,
        f"GPU{physical_gpu} {family}/{case_id}/{condition}: A1/B1/B2/A2 "
        "timing-mode policies differ",
    )
    required_pstate, required_throttle_reasons, max_sm_clock_delta_mhz = next(
        iter(timing_mode_policies)
    )
    _require(
        required_pstate == REQUIRED_TIMING_PSTATE,
        f"GPU{physical_gpu} {family}/{case_id}/{condition}: timing policy is not P1",
    )
    samples = {
        position: runs[position]["cases"][case_id]["conditions"][condition][
            "samples_us"
        ]
        for position in SEQUENCE
    }
    means = {position: statistics.fmean(values) for position, values in samples.items()}
    baseline = [*samples["a1"], *samples["a2"]]
    current = [*samples["b1"], *samples["b2"]]
    baseline_mean = statistics.fmean(baseline)
    current_mean = statistics.fmean(current)
    baseline_median = statistics.median(baseline)
    current_median = statistics.median(current)
    baseline_p95 = _percentile(baseline, 0.95)
    current_p95 = _percentile(current, 0.95)
    a_drift = (_ratio(means["a2"], means["a1"]) - 1.0) * 100.0
    b_drift = (_ratio(means["b2"], means["b1"]) - 1.0) * 100.0
    mean_regression = (_ratio(current_mean, baseline_mean) - 1.0) * 100.0
    median_regression = (_ratio(current_median, baseline_median) - 1.0) * 100.0
    p95_regression = (_ratio(current_p95, baseline_p95) - 1.0) * 100.0
    failures = []
    if abs(a_drift) > MAX_RUN_MEAN_DRIFT_PCT:
        failures.append(f"baseline run drift {a_drift:.6f}%")
    if abs(b_drift) > MAX_RUN_MEAN_DRIFT_PCT:
        failures.append(f"current run drift {b_drift:.6f}%")
    if mean_regression > MAX_MEAN_REGRESSION_PCT:
        failures.append(f"mean regression {mean_regression:.6f}%")
    if median_regression > MAX_MEDIAN_REGRESSION_PCT:
        failures.append(f"median regression {median_regression:.6f}%")
    if p95_regression > MAX_P95_REGRESSION_PCT:
        failures.append(f"p95 regression {p95_regression:.6f}%")
    _require(
        not failures,
        f"GPU{physical_gpu} {family}/{case_id}/{condition}: " + "; ".join(failures),
    )
    baseline_topology = runs["a1"]["cases"][case_id]["graph"]["topology_sha256"]
    current_topology = runs["b1"]["cases"][case_id]["graph"]["topology_sha256"]
    baseline_graph = runs["a1"]["cases"][case_id]["graph"]
    current_graph = runs["b1"]["cases"][case_id]["graph"]
    return {
        "row_schema": ROW_SCHEMA,
        "family": family,
        "case_id": case_id,
        "physical_gpu": physical_gpu,
        "gpu_uuid": runs["a1"]["gpu"]["uuid"],
        "condition": condition,
        "required_active_throttle_reasons": required_throttle_reasons,
        "max_sm_clock_delta_mhz": max_sm_clock_delta_mhz,
        "samples_a1": len(samples["a1"]),
        "samples_b1": len(samples["b1"]),
        "samples_b2": len(samples["b2"]),
        "samples_a2": len(samples["a2"]),
        "a1_mean_us": means["a1"],
        "b1_mean_us": means["b1"],
        "b2_mean_us": means["b2"],
        "a2_mean_us": means["a2"],
        "a_run_mean_drift_pct": a_drift,
        "b_run_mean_drift_pct": b_drift,
        "baseline_mean_us": baseline_mean,
        "current_mean_us": current_mean,
        "mean_ratio_current_over_baseline": _ratio(current_mean, baseline_mean),
        "mean_regression_pct": mean_regression,
        "baseline_median_us": baseline_median,
        "current_median_us": current_median,
        "median_ratio_current_over_baseline": _ratio(current_median, baseline_median),
        "median_regression_pct": median_regression,
        "baseline_p95_us": baseline_p95,
        "current_p95_us": current_p95,
        "p95_ratio_current_over_baseline": _ratio(current_p95, baseline_p95),
        "p95_regression_pct": p95_regression,
        "baseline_topology_sha256": baseline_topology,
        "current_topology_sha256": current_topology,
        "baseline_node_count": baseline_graph["node_count"],
        "current_node_count": current_graph["node_count"],
        "baseline_kernel_node_count": baseline_graph["kernel_node_count"],
        "current_kernel_node_count": current_graph["kernel_node_count"],
        "topology_disposition": topology_disposition,
        "status": "pass",
    }


def _validate_group(
    *,
    family: str,
    physical_gpu: int,
    runs: dict[str, dict[str, Any]],
    family_contract: dict[str, Any],
) -> list[dict[str, object]]:
    process_ids = [
        str(runs[position]["invocation"]["process_id"]) for position in SEQUENCE
    ]
    _require(
        len(set(process_ids)) == len(SEQUENCE),
        f"GPU{physical_gpu} {family}: A1/B1/B2/A2 were not separate processes",
    )
    for before_position, after_position in pairwise(SEQUENCE):
        before = runs[before_position]["invocation"]
        after = runs[after_position]["invocation"]
        _require(
            before["finished_unix_ns"] < after["started_unix_ns"],
            f"GPU{physical_gpu} {family}: process sequence is not A1/B1/B2/A2",
        )
    gpu_identities = {
        (
            int(run["gpu"]["physical_ordinal"]),
            str(run["gpu"]["name"]),
            _normalize_uuid(run["gpu"]["uuid"]),
            tuple(run["gpu"]["capability"]),
            int(run["gpu"]["l2_cache_bytes"]),
        )
        for run in runs.values()
    }
    _require(
        len(gpu_identities) == 1,
        f"GPU{physical_gpu} {family}: A1/B1/B2/A2 GPU identities differ",
    )
    stable_modes = {
        tuple(
            str(run["gpu"]["mode_before"]["fields"][field])
            for field in _GPU_STABLE_FIELDS
        )
        for run in runs.values()
    }
    _require(
        len(stable_modes) == 1,
        f"GPU{physical_gpu} {family}: stable GPU mode differs across processes",
    )
    runtime_values = {
        field: {str(run["runtime"][field]) for run in runs.values()}
        for field in _RUNTIME_CROSS_ARM_STABLE_FIELDS
    }
    differing_runtime = {
        field: values for field, values in runtime_values.items() if len(values) != 1
    }
    _require(
        not differing_runtime,
        f"GPU{physical_gpu} {family}: runtime/build environment differs: "
        f"{differing_runtime!r}",
    )
    for arm, positions in (("baseline", ("a1", "a2")), ("current", ("b1", "b2"))):
        arm_runtime_values = {
            field: {str(runs[position]["runtime"][field]) for position in positions}
            for field in _RUNTIME_ARM_STABLE_FIELDS
        }
        differing_arm_runtime = {
            field: values
            for field, values in arm_runtime_values.items()
            if len(values) != 1
        }
        _require(
            not differing_arm_runtime,
            f"GPU{physical_gpu} {family}: {arm} repeat runtime/build environment "
            f"differs: {differing_arm_runtime!r}",
        )

    rows: list[dict[str, object]] = []
    for case_id, case_contract in sorted(family_contract["case_by_id"].items()):
        case_runs = {
            position: runs[position]["cases"][case_id] for position in SEQUENCE
        }
        _require(
            len({case["input_sha256"] for case in case_runs.values()}) == 1,
            f"GPU{physical_gpu} {family}/{case_id}: process inputs differ",
        )
        _require(
            len(
                {
                    case["correctness"]["read_only_inputs_sha256"]
                    for case in case_runs.values()
                }
            )
            == 1,
            f"GPU{physical_gpu} {family}/{case_id}: read-only inputs differ",
        )
        baseline_outputs = {
            case_runs[position]["correctness"]["output_sha256"]
            for position in ("a1", "a2")
        }
        current_outputs = {
            case_runs[position]["correctness"]["output_sha256"]
            for position in ("b1", "b2")
        }
        _require(
            len(baseline_outputs) == 1 and len(current_outputs) == 1,
            f"GPU{physical_gpu} {family}/{case_id}: repeat outputs are not deterministic",
        )
        if case_contract["cross_arm_output_policy"] == "bit-exact":
            _require(
                baseline_outputs == current_outputs,
                f"GPU{physical_gpu} {family}/{case_id}: cross-arm output is not bit-exact",
            )
        for arm, positions in (("baseline", ("a1", "a2")), ("current", ("b1", "b2"))):
            artifact_bindings = {
                _canonical_sha256(
                    sorted(
                        case_runs[position]["artifacts"],
                        key=lambda artifact: str(artifact["role"]),
                    )
                )
                for position in positions
            }
            _require(
                len(artifact_bindings) == 1,
                f"GPU{physical_gpu} {family}/{case_id}: {arm} repeat exact artifact changed",
            )
        topology_contract = case_contract["graph_topology_contract"]
        for arm, positions in (("baseline", ("a1", "a2")), ("current", ("b1", "b2"))):
            topology_signatures = {
                (
                    case_runs[position]["graph"]["topology_sha256"],
                    int(case_runs[position]["graph"]["node_count"]),
                    int(case_runs[position]["graph"]["kernel_node_count"]),
                )
                for position in positions
            }
            _require(
                len(topology_signatures) == 1,
                f"GPU{physical_gpu} {family}/{case_id}: {arm} repeat graph topology changed",
            )
            observed_topology, observed_nodes, observed_kernel_nodes = next(
                iter(topology_signatures)
            )
            expected_signature = topology_contract[arm]
            _require(
                observed_topology == expected_signature["topology_sha256"]
                and observed_nodes == expected_signature["node_count"]
                and observed_kernel_nodes == expected_signature["kernel_node_count"],
                f"GPU{physical_gpu} {family}/{case_id}: {arm} topology differs from "
                "the reviewed case contract",
            )
        for arm, positions in (("baseline", ("a1", "a2")), ("current", ("b1", "b2"))):
            capacities = {
                case_runs[position]["allocation"]["workspace_capacity_bytes"]
                for position in positions
            }
            _require(
                len(capacities) == 1,
                f"GPU{physical_gpu} {family}/{case_id}: {arm} workspace capacity changed",
            )
        for condition in ("warm_l2", "cold_l2"):
            rows.append(
                _performance_row(
                    family=family,
                    case_id=case_id,
                    physical_gpu=physical_gpu,
                    condition=condition,
                    runs=runs,
                    topology_disposition=topology_contract["disposition"],
                )
            )
    return rows


def build_index(
    *,
    baseline_source_manifest: Path,
    current_source_manifest: Path,
    contract_path: Path,
    evidence_set_path: Path,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    baseline_source_manifest = baseline_source_manifest.resolve()
    current_source_manifest = current_source_manifest.resolve()
    contract_path = contract_path.resolve()
    evidence_set_path = evidence_set_path.resolve()
    sources = {
        "baseline": _validate_source_manifest(baseline_source_manifest, "baseline"),
        "current": _validate_source_manifest(current_source_manifest, "current"),
    }
    _require(
        sources["baseline"]["production_fingerprint"]
        != sources["current"]["production_fingerprint"],
        "baseline and current production-source fingerprints are identical; use the "
        "compiler-only release gate for a same-source comparison",
    )
    _require(
        sources["baseline"]["manifest_sha256"] != sources["current"]["manifest_sha256"],
        "baseline and current frozen source manifests are identical",
    )
    _require(
        sources["baseline"]["manifest"]["source_id"]
        != sources["current"]["manifest"]["source_id"],
        "baseline and current source IDs are identical",
    )
    contract = _validate_contract(contract_path)
    evidence_set = _exact_keys(
        _load_json(evidence_set_path),
        {"schema", "contract_sha256", "runs", "evidence_set_sha256"},
        str(evidence_set_path),
    )
    _require(
        evidence_set["schema"] == EVIDENCE_SET_SCHEMA,
        f"{evidence_set_path}: invalid evidence-set schema",
    )
    _require(
        evidence_set["contract_sha256"] == contract["sha256"],
        f"{evidence_set_path}: evidence set binds a different harness/case contract",
    )
    evidence_payload = {
        key: value
        for key, value in evidence_set.items()
        if key != "evidence_set_sha256"
    }
    _require(
        evidence_set["evidence_set_sha256"] == _canonical_sha256(evidence_payload),
        f"{evidence_set_path}: evidence-set SHA mismatch",
    )
    entries = evidence_set["runs"]
    _require(isinstance(entries, list), f"{evidence_set_path}: runs are not a list")
    expected_keys = {
        (family, gpu, position)
        for family in REQUIRED_FAMILIES
        for gpu in PHYSICAL_GPUS
        for position in SEQUENCE
    }
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    paths: set[Path] = set()
    for index, raw in enumerate(entries):
        location = f"{evidence_set_path}.runs[{index}]"
        entry = _exact_keys(
            raw, {"family", "physical_gpu", "position", "path", "sha256"}, location
        )
        family = entry["family"]
        gpu = entry["physical_gpu"]
        position = entry["position"]
        _require(
            isinstance(family, str)
            and isinstance(gpu, int)
            and not isinstance(gpu, bool)
            and isinstance(position, str),
            f"{location}: invalid evidence key",
        )
        key = (family, gpu, position)
        _require(key in expected_keys, f"{location}: unreviewed evidence key {key!r}")
        _require(key not in by_key, f"{location}: duplicate evidence key {key!r}")
        raw_path = entry["path"]
        _require(isinstance(raw_path, str) and raw_path, f"{location}: empty path")
        run_path = Path(raw_path)
        if not run_path.is_absolute():
            run_path = (evidence_set_path.parent / run_path).resolve()
        else:
            run_path = run_path.resolve()
        _require(
            run_path.is_file(), f"{location}: run artifact does not exist: {run_path}"
        )
        _require(run_path not in paths, f"{location}: run artifact path is reused")
        paths.add(run_path)
        _require(_is_sha256(entry["sha256"]), f"{location}: invalid artifact SHA")
        _require(
            _sha256_file(run_path) == entry["sha256"],
            f"{location}: artifact SHA mismatch",
        )
        by_key[key] = {**entry, "resolved_path": run_path}
    _require(
        set(by_key) == expected_keys,
        f"{evidence_set_path}: incomplete family/GPU/process coverage; "
        f"missing={sorted(expected_keys - set(by_key))!r}, "
        f"unexpected={sorted(set(by_key) - expected_keys)!r}",
    )

    validated_runs: dict[tuple[str, int, str], dict[str, Any]] = {}
    for key in sorted(expected_keys):
        family, gpu, position = key
        entry = by_key[key]
        validated_runs[key] = _validate_run(
            entry["resolved_path"],
            family=family,
            physical_gpu=gpu,
            position=position,
            contract=contract,
            sources=sources,
        )
    process_ids = [
        str(validated_runs[key]["invocation"]["process_id"])
        for key in sorted(validated_runs)
    ]
    _require(
        len(set(process_ids)) == len(process_ids),
        "process identity is reused across end-to-end run artifacts",
    )
    rows: list[dict[str, object]] = []
    for family in REQUIRED_FAMILIES:
        for gpu in PHYSICAL_GPUS:
            group = {
                position: validated_runs[(family, gpu, position)]
                for position in SEQUENCE
            }
            rows.extend(
                _validate_group(
                    family=family,
                    physical_gpu=gpu,
                    runs=group,
                    family_contract=contract["families"][family],
                )
            )

    expected_row_count = (
        sum(
            len(contract["families"][family]["case_by_id"])
            for family in REQUIRED_FAMILIES
        )
        * len(PHYSICAL_GPUS)
        * 2
    )
    _require(
        len(rows) == expected_row_count, "internal performance-row coverage mismatch"
    )
    index: dict[str, Any] = {
        "schema": INDEX_SCHEMA,
        "status": "pass",
        "comparison": "pre-migration-4.5.2-source-vs-final-4.6.0-source",
        "contract_sha256": contract["sha256"],
        "harness_fingerprint": contract["harness_fingerprint"],
        "required_families": list(REQUIRED_FAMILIES),
        "physical_gpus": list(PHYSICAL_GPUS),
        "sequence": list(SEQUENCE),
        "thresholds": {
            "minimum_samples_per_process_condition": MINIMUM_SAMPLES_PER_PROCESS_CONDITION,
            "minimum_l2_flush_multiple": MINIMUM_L2_FLUSH_MULTIPLE,
            "max_mean_regression_pct": MAX_MEAN_REGRESSION_PCT,
            "max_median_regression_pct": MAX_MEDIAN_REGRESSION_PCT,
            "max_p95_regression_pct": MAX_P95_REGRESSION_PCT,
            "max_run_mean_drift_pct": MAX_RUN_MEAN_DRIFT_PCT,
            "p95_definition": "linear interpolation at 0.95 * (n - 1)",
        },
        "sources": {
            side: {
                key: source[key]
                for key in (
                    "manifest_sha256",
                    "manifest_artifact_sha256",
                    "production_fingerprint",
                    "runtime_package_fingerprint",
                    "artifact",
                )
            }
            for side, source in sources.items()
        },
        "artifacts": {
            "contract": contract["artifact"],
            "evidence_set": _artifact(evidence_set_path, schema=EVIDENCE_SET_SCHEMA),
            "process_results": [
                validated_runs[key]["artifact"] for key in sorted(validated_runs)
            ],
        },
        "coverage": {
            "family_count": len(REQUIRED_FAMILIES),
            "case_count": sum(
                len(contract["families"][family]["case_by_id"])
                for family in REQUIRED_FAMILIES
            ),
            "process_result_count": len(validated_runs),
            "performance_row_count": len(rows),
        },
        "performance_rows_sha256": _canonical_sha256(rows),
        "performance_rows": rows,
    }
    index["result_sha256"] = _canonical_sha256(index)
    return index, rows


def _write_outputs(
    *,
    index: dict[str, Any],
    rows: list[dict[str, object]],
    output_json: Path,
    output_csv: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    with output_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source-manifest", type=Path, required=True)
    parser.add_argument("--current-source-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence-set", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        args.baseline_source_manifest.resolve(),
        args.current_source_manifest.resolve(),
        args.contract.resolve(),
        args.evidence_set.resolve(),
    }
    outputs = {args.output_json.resolve(), args.output_csv.resolve()}
    if len(outputs) != 2 or inputs & outputs:
        parser.error("outputs must be distinct and must not overwrite an input")
    return args


def main() -> int:
    args = _args()
    try:
        index, rows = build_index(
            baseline_source_manifest=args.baseline_source_manifest,
            current_source_manifest=args.current_source_manifest,
            contract_path=args.contract,
            evidence_set_path=args.evidence_set,
        )
    except EndToEndValidationError as exc:
        print(f"end-to-end migration gate failed: {exc}", file=sys.stderr)
        return 1
    _write_outputs(
        index=index,
        rows=rows,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(
        f"status=pass families={index['coverage']['family_count']} "
        f"cases={index['coverage']['case_count']} "
        f"process_results={index['coverage']['process_result_count']} "
        f"performance_rows={index['coverage']['performance_row_count']} "
        f"result_sha256={index['result_sha256']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
