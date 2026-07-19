#!/usr/bin/env python3
# ruff: noqa: SIM905
"""Validate and index the final CUTLASS 4.5.2/4.6 migration evidence.

This is an offline, benchmark-artifact-only release gate.  It imports neither
torch nor CUTLASS and never selects or probes a GPU.  It deliberately requires
the complete four-corpus matrix, per-GPU resource/SASS/accounting reports, and
two independent exact-cache ABBA runs for every production-bound compile spec
and performance condition.  Accounting exceptions are an additional binding
gate, not the boundary of performance coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any

from validation.cutlass_migration.acceptance.corpus.source_snapshot import (
    SOURCE_SNAPSHOT_SCHEMA,
)
from validation.cutlass_migration.evidence.compare_resources import (
    _RESOURCE_REPORT_SCHEMA,
    _read as _read_resource_report,
)
from validation.cutlass_migration.evidence.compare_sass_register_sets import (
    _DELTA_SCHEMA as _SASS_DELTA_SCHEMA,
    _OUTPUT_FIELDS as _SASS_DELTA_FIELDS,
)
from validation.cutlass_migration.evidence.register_accounting import (
    _ACCOUNTING_SCHEMA,
    _DELTA_REPORT_FIELDS,
    _DELTA_REPORT_SCHEMA,
    _exception_fields,
    _require_exact_pair,
    _require_sass_delta_pair,
    _sass_accounting_fields,
    _validate_and_recompute,
)


_INDEX_SCHEMA = "b12x.cute.migration.release_artifact_index.v1"
_EXCEPTION_INDEX_SCHEMA = "b12x.cute.migration.release_exception_index.v1"
_TRACE_SCHEMA = "b12x.cute.migration.case_trace.v1"
_MATRIX_SCHEMA = "b12x.cute.migration.corpus_matrix.v1"
_CORPUS_ID = "b12x-cutlass-45-46-full-gpu-corpus"
_CORPUS_VERSION = "1"
_ARCHITECTURE = "sm_120a"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CUDA_EVENT_POOL_SCHEMA = "b12x.cuda_event_pool.v1"
_GPU_MODE_STABILITY_SCHEMA = "b12x.gpu_mode_stability.v1"
_GPU_TIMING_MODE_POLICY_SCHEMA = "b12x.gpu_timing_mode_policy.v1"
_REQUIRED_TIMING_PSTATE = "P1"
_MAX_TIMING_SM_CLOCK_DELTA_MHZ = 60.0
_MIN_TIMING_PRECONDITION_SECONDS = 5.0
_MAX_TIMING_PRECONDITION_SECONDS = 60.0
_PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS = frozenset((0, 0x4))
_TIMING_STABLE_MODE_FIELDS = [
    "index",
    "uuid",
    "persistence_mode",
    "compute_mode",
    "power.limit",
]
_SMEM_CONTRACT_SCHEMA = "b12x.cute.smem_contracts.v1"
_SMEM_GATE_SCHEMA = "b12x.cute.migration.smem_contract_gate.v1"
_SMEM_FINALIZATION_SCHEMA = "b12x.cute.migration.smem_contract_finalization.v1"
_SMEM_AUDITOR_PATH = "validation/cutlass_migration/evidence/smem_contracts.py"
_SMEM_SOURCE_ROOTS = {
    "production": ["b12x"],
    "infrastructure": ["benchmarks", "tests", "validation"],
}
_SMEM_COUNT_FIELDS = {
    "python_file_count",
    "allocator_count",
    "allocator_pass_count",
    "allocator_fail_count",
    "allocation_call_count",
    "allocate_call_count",
    "allocate_tensor_call_count",
    "private_memrange_identifier_count",
    "private_memrange_centralized_count",
    "private_memrange_outside_count",
    "parse_error_count",
    "violation_count",
    "row_count",
}
_SMEM_ALLOCATOR_ROW_FIELDS = {
    "kind",
    "source_category",
    "path",
    "line",
    "column",
    "scope",
    "allocator_name",
    "constructor_argument_count",
    "constructor_keyword_count",
    "allocator_store_count",
    "allocator_store_lines",
    "allocation_count",
    "allocation_methods",
    "allocation_lines",
    "allocation_argument_counts",
    "allocation_keyword_counts",
    "allocation_argument_sources",
    "allocation_result_names",
    "typed_allocation",
    "allocation_after_constructor",
    "allocation_result_bound_locally",
    "violations",
    "passed",
}
_SMEM_PRIVATE_ROW_FIELDS = {
    "kind",
    "source_category",
    "path",
    "line",
    "column",
    "scope",
    "identifier",
    "centralized",
    "violations",
    "passed",
}

_EXPECTED_PACKAGES = {
    "4.5.2": {
        "nvidia-cutlass-dsl": "4.5.2",
        "nvidia-cutlass-dsl-libs-base": "4.5.2",
        "nvidia-cutlass-dsl-libs-core": "missing",
        "nvidia-cutlass-dsl-libs-cu12": "missing",
        "nvidia-cutlass-dsl-libs-cu13": "4.5.2",
    },
    "4.6.0": {
        "nvidia-cutlass-dsl": "4.6.0",
        "nvidia-cutlass-dsl-libs-base": "4.6.0",
        "nvidia-cutlass-dsl-libs-core": "4.6.0",
        "nvidia-cutlass-dsl-libs-cu12": "4.6.0",
        "nvidia-cutlass-dsl-libs-cu13": "4.6.0",
    },
}
_TOOLCHAIN_PACKAGE_NAMES = {
    "cutlass_dsl": "nvidia-cutlass-dsl",
    "cutlass_dsl_libs_base": "nvidia-cutlass-dsl-libs-base",
    "cutlass_dsl_libs_core": "nvidia-cutlass-dsl-libs-core",
    "cutlass_dsl_libs_cu12": "nvidia-cutlass-dsl-libs-cu12",
    "cutlass_dsl_libs_cu13": "nvidia-cutlass-dsl-libs-cu13",
}

_SUPPORTED_ABBA_SCHEMAS = frozenset(
    {
        "b12x.bf16_to_fp4_tma.cache_abba.v4",
        "b12x.compute_exceptions.cache_abba.v1",
        "b12x.contiguous_attention.cache_abba.v2",
        "b12x.attention.mla.decode_merge.exact_cache_abba.v2",
        "b12x.attention.mla.prefill_mg.exact_cache_abba.v4",
        "b12x.attention.indexer.exact_cache_abba.v1",
        "b12x.attention.paged.exact_cache_abba.v1",
        "b12x.residual_prefill_partial.cache_abba.v4",
        "b12x.residual.composite_exact_cache_abba.v1",
        "b12x.tp_moe.dynamic.cache_abba.v2",
        "b12x.w4a16.serving.cache_abba.v2",
        "b12x.w4a16.topk_sum.cache_abba.v1",
        "b12x.w4a8.dynamic.cache_abba.v2",
    }
)

_BINDING_FIELDS = tuple(
    (
        "case_id,branch,shared_manifest_group,evidence_kind,nodeid,"
        "test_nvtx_label,cache_key,semantic_key,kernel_id,"
        "compile_spec_version,compile_spec_hash,compile_spec_json,kernel,"
        "launch_count,registers,frame_bytes,min_stack_bytes,"
        "local_load_instructions,local_store_instructions,threads_per_cta,"
        "launch_dynamic_smem_bytes,occupancy_active_ctas_per_sm"
    ).split(",")
)

_EXCEPTION_CSV_FIELDS = (
    "release_exception_index_schema",
    "comparison_semantic_key",
    "symbol_sha256",
    "family",
    "kernel",
    "compile_spec_hash",
    "exception_fields",
    "cause",
    "disposition",
    "performance_status",
    "gpu4_artifacts_json",
    "gpu5_artifacts_json",
    "gpu4_conditions_json",
    "gpu5_conditions_json",
    "max_mean_regression_pct",
    "max_median_regression_pct",
    "max_p95_regression_pct",
    "max_run_mean_drift_pct",
    "status",
)

_SASS_PROVENANCE_ONLY_FIELDS = frozenset(
    {
        "baseline_semantic_key",
        "current_semantic_key",
        "baseline_package_fingerprint",
        "current_package_fingerprint",
        "baseline_object_sha256",
        "current_object_sha256",
    }
)


class ReleaseValidationError(RuntimeError):
    """A final-release artifact or cross-artifact invariant failed."""


def _fail(message: str) -> None:
    raise ReleaseValidationError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise ReleaseValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{path}: expected a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _normalize_uuid(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("gpu-"):
        raw = raw[4:]
    return raw


def _artifact_record(path: Path, *, schema: str = "") -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema": schema,
    }


def _csv_rows(
    path: Path,
    expected_fields: tuple[str, ...],
    *,
    description: str,
    delimiter: str = ",",
) -> list[dict[str, str]]:
    try:
        source = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ReleaseValidationError(
            f"cannot read {description} {path}: {exc}"
        ) from exc
    with source:
        reader = csv.DictReader(source, delimiter=delimiter)
        observed = tuple(reader.fieldnames or ())
        if observed != expected_fields:
            _fail(
                f"{path}: expected exact {description} header; "
                f"expected={expected_fields!r}, observed={observed!r}"
            )
        rows = [
            {key: (value or "").strip() for key, value in row.items()} for row in reader
        ]
    _require(rows, f"{path}: {description} has no rows")
    return rows


def _validate_source_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest = _load_json(path)
    _require(
        manifest.get("schema") == SOURCE_SNAPSHOT_SCHEMA,
        f"{path}: expected source schema {SOURCE_SNAPSHOT_SCHEMA}",
    )
    recorded = manifest.get("manifest_sha256")
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    computed = _canonical_sha256(payload)
    _require(
        recorded == computed,
        f"{path}: source manifest canonical hash mismatch "
        f"recorded={recorded!r}, computed={computed}",
    )
    package = manifest.get("b12x_package")
    _require(isinstance(package, dict), f"{path}: missing b12x_package")
    fingerprint = package.get("fingerprint")
    _require(_is_sha256(fingerprint), f"{path}: invalid b12x package fingerprint")
    return manifest, str(fingerprint)


def _validate_smem_count_map(
    path: Path,
    value: object,
    *,
    description: str,
) -> dict[str, int]:
    _require(
        isinstance(value, dict)
        and set(value) == _SMEM_COUNT_FIELDS
        and all(
            isinstance(value[field], int)
            and not isinstance(value[field], bool)
            and value[field] >= 0
            for field in _SMEM_COUNT_FIELDS
        ),
        f"{path}: {description} has invalid exact SMEM summary counts",
    )
    return value


def _validate_smem_contract_report(
    path: Path,
    report: object,
    *,
    source_root: str,
) -> dict[str, Any]:
    _require(isinstance(report, dict), f"{path}: SMEM report is not an object")
    _require(
        set(report)
        == {
            "schema",
            "root",
            "audited_source_roots",
            "central_private_memrange_path",
            "rows",
            "counts",
            "source_counts",
            "passed",
        }
        and report.get("schema") == _SMEM_CONTRACT_SCHEMA
        and report.get("root") == source_root
        and report.get("audited_source_roots") == _SMEM_SOURCE_ROOTS
        and report.get("central_private_memrange_path") == "b12x/cute/smem.py"
        and report.get("passed") is True,
        f"{path}: invalid final SMEM report schema/root/policy",
    )
    rows = report.get("rows")
    _require(
        isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows),
        f"{path}: final SMEM report lacks machine-readable rows",
    )
    counts = _validate_smem_count_map(path, report.get("counts"), description="total")
    source_counts_raw = report.get("source_counts")
    _require(
        isinstance(source_counts_raw, dict)
        and set(source_counts_raw) == set(_SMEM_SOURCE_ROOTS),
        f"{path}: final SMEM report lacks exact production/infrastructure counts",
    )
    source_counts = {
        category: _validate_smem_count_map(
            path,
            source_counts_raw[category],
            description=f"{category} source",
        )
        for category in _SMEM_SOURCE_ROOTS
    }
    _require(
        all(
            counts[field]
            == source_counts["production"][field]
            + source_counts["infrastructure"][field]
            for field in _SMEM_COUNT_FIELDS
        )
        and source_counts["production"]["python_file_count"] > 0
        and source_counts["infrastructure"]["python_file_count"] > 0,
        f"{path}: final SMEM total/source counts do not reconcile",
    )

    allocator_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    parse_rows: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        prefix = f"{path}: SMEM row {number}"
        source_category = row.get("source_category")
        row_path = row.get("path")
        _require(
            source_category in _SMEM_SOURCE_ROOTS
            and isinstance(row_path, str)
            and row_path
            and (
                (source_category == "production" and row_path.startswith("b12x/"))
                or (
                    source_category == "infrastructure"
                    and row_path.startswith(("benchmarks/", "tests/", "validation/"))
                )
            )
            and row.get("passed") is True
            and row.get("violations") == [],
            f"{prefix}: invalid source classification or fail-open status",
        )
        kind = row.get("kind")
        if kind == "allocator":
            allocator_rows.append(row)
            _require(
                set(row) == _SMEM_ALLOCATOR_ROW_FIELDS,
                f"{prefix}: allocator row fields are not exact",
            )
            methods = row.get("allocation_methods")
            method_shape = (
                methods == ["allocate"]
                and row.get("allocation_argument_counts") == [1]
                and row.get("allocation_keyword_counts") == [0]
            ) or (
                methods == ["allocate_tensor"]
                and row.get("allocation_argument_counts") == [0]
                and row.get("allocation_keyword_counts") == [3]
            )
            _require(
                all(
                    isinstance(row.get(field), int)
                    and not isinstance(row.get(field), bool)
                    and row[field] == expected
                    for field, expected in (
                        ("constructor_argument_count", 0),
                        ("constructor_keyword_count", 0),
                        ("allocator_store_count", 1),
                        ("allocation_count", 1),
                    )
                )
                and isinstance(row.get("line"), int)
                and not isinstance(row.get("line"), bool)
                and row["line"] > 0
                and isinstance(row.get("column"), int)
                and not isinstance(row.get("column"), bool)
                and row["column"] >= 0
                and isinstance(row.get("allocator_name"), str)
                and bool(row["allocator_name"])
                and isinstance(row.get("allocator_store_lines"), list)
                and len(row["allocator_store_lines"]) == 1
                and all(
                    type(value) is int and value > 0
                    for value in row["allocator_store_lines"]
                )
                and isinstance(row.get("allocation_lines"), list)
                and len(row["allocation_lines"]) == 1
                and all(
                    type(value) is int and value > 0
                    for value in row["allocation_lines"]
                )
                and isinstance(row.get("allocation_argument_sources"), list)
                and len(row["allocation_argument_sources"]) == 1
                and all(
                    isinstance(value, str) and bool(value)
                    for value in row["allocation_argument_sources"]
                )
                and isinstance(row.get("allocation_result_names"), list)
                and len(row["allocation_result_names"]) == 1
                and all(
                    isinstance(value, str) and bool(value)
                    for value in row["allocation_result_names"]
                )
                and row.get("typed_allocation") is True
                and row.get("allocation_after_constructor") is True
                and row.get("allocation_result_bound_locally") is True
                and method_shape,
                f"{prefix}: invalid CUTLASS 4.6 allocator contract",
            )
        elif kind == "private_memrange":
            private_rows.append(row)
            _require(
                set(row) == _SMEM_PRIVATE_ROW_FIELDS
                and source_category == "production"
                and row_path == "b12x/cute/smem.py"
                and isinstance(row.get("line"), int)
                and not isinstance(row.get("line"), bool)
                and row["line"] > 0
                and isinstance(row.get("column"), int)
                and not isinstance(row.get("column"), bool)
                and row["column"] >= 0
                and isinstance(row.get("identifier"), str)
                and row["identifier"].startswith("_MemRange")
                and row.get("centralized") is True,
                f"{prefix}: private _MemRange escaped its central bridge",
            )
        elif kind == "parse_error":
            parse_rows.append(row)
        else:
            _fail(f"{prefix}: unknown SMEM audit row kind {kind!r}")

    _require(not parse_rows, f"{path}: final SMEM report contains parse errors")
    for source_category, category_counts in source_counts.items():
        category_rows = [
            row for row in rows if row["source_category"] == source_category
        ]
        category_allocators = [
            row for row in allocator_rows if row["source_category"] == source_category
        ]
        category_private = [
            row for row in private_rows if row["source_category"] == source_category
        ]
        recomputed = {
            "allocator_count": len(category_allocators),
            "allocator_pass_count": len(category_allocators),
            "allocator_fail_count": 0,
            "allocation_call_count": sum(
                row["allocation_count"] for row in category_allocators
            ),
            "allocate_call_count": sum(
                row["allocation_methods"] == ["allocate"] for row in category_allocators
            ),
            "allocate_tensor_call_count": sum(
                row["allocation_methods"] == ["allocate_tensor"]
                for row in category_allocators
            ),
            "private_memrange_identifier_count": len(category_private),
            "private_memrange_centralized_count": sum(
                row["centralized"] is True for row in category_private
            ),
            "private_memrange_outside_count": sum(
                row["centralized"] is not True for row in category_private
            ),
            "parse_error_count": 0,
            "violation_count": 0,
            "row_count": len(category_rows),
        }
        _require(
            all(category_counts[field] == value for field, value in recomputed.items()),
            f"{path}: {source_category} SMEM counts do not reconstruct from rows",
        )
    _require(
        counts["allocator_count"] > 0
        and source_counts["production"]["allocator_count"] > 0
        and source_counts["infrastructure"]["allocator_count"] > 0
        and source_counts["infrastructure"]["allocate_tensor_call_count"] > 0,
        f"{path}: final SMEM report does not cover both allocation APIs/surfaces",
    )
    return report


def _validate_smem_contract_finalization(
    trace_path: Path,
    trace: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, object]:
    """Bind the final SMEM bytes to the frozen auditor, trace, and source root."""

    smem_path = trace_path.parent / "smem-contracts.final.json"
    _require(smem_path.is_file(), f"corpus {trace_path.parent} lacks {smem_path.name}")
    manifest_inputs = manifest.get("inputs")
    _require(isinstance(manifest_inputs, dict), f"{trace_path}: source inputs missing")
    auditor_input = manifest_inputs.get("smem_contract_auditor")
    _require(
        isinstance(auditor_input, dict)
        and set(auditor_input) == {"path", "sha256", "size_bytes"}
        and auditor_input.get("path") == _SMEM_AUDITOR_PATH
        and _is_sha256(auditor_input.get("sha256"))
        and isinstance(auditor_input.get("size_bytes"), int)
        and not isinstance(auditor_input.get("size_bytes"), bool)
        and auditor_input["size_bytes"] > 0,
        f"{trace_path}: frozen SMEM auditor input is invalid",
    )
    static = trace.get("static_validation")
    _require(isinstance(static, dict), f"{trace_path}: static validation missing")
    static_hashes = static.get("hashes")
    _require(
        isinstance(static_hashes, dict)
        and static_hashes.get("smem_contract_auditor_sha256")
        == auditor_input["sha256"],
        f"{trace_path}: static validation used a different SMEM auditor",
    )
    finalization = trace.get("smem_contracts")
    _require(
        isinstance(finalization, dict)
        and set(finalization)
        == {"schema", "passed", "static_final_reports_equal", "gate", "artifact"}
        and finalization.get("schema") == _SMEM_FINALIZATION_SCHEMA
        and finalization.get("passed") is True
        and finalization.get("static_final_reports_equal") is True,
        f"{trace_path}: invalid final SMEM finalization envelope",
    )
    gate = finalization.get("gate")
    artifact = finalization.get("artifact")
    _require(
        isinstance(gate, dict)
        and set(gate)
        == {
            "schema",
            "passed",
            "auditor",
            "report_schema",
            "report_sha256",
            "counts",
            "report",
        }
        and gate.get("schema") == _SMEM_GATE_SCHEMA
        and gate.get("passed") is True
        and gate.get("auditor")
        == {"path": _SMEM_AUDITOR_PATH, "sha256": auditor_input["sha256"]}
        and gate.get("report_schema") == _SMEM_CONTRACT_SCHEMA,
        f"{trace_path}: final SMEM gate/auditor binding is invalid",
    )
    _require(
        static.get("smem_contracts") == gate,
        f"{trace_path}: static/final SMEM reports are not byte-identical",
    )
    trace_artifacts = trace.get("artifacts")
    _require(
        isinstance(artifact, dict)
        and set(artifact) == {"path", "sha256", "schema"}
        and Path(str(artifact.get("path", ""))).resolve() == smem_path.resolve()
        and artifact.get("schema") == _SMEM_CONTRACT_SCHEMA
        and isinstance(trace_artifacts, dict)
        and trace_artifacts.get("smem_contracts") == artifact,
        f"{trace_path}: final SMEM artifact binding is invalid",
    )
    artifact_sha256 = _sha256_file(smem_path)
    _require(
        artifact.get("sha256") == artifact_sha256
        and gate.get("report_sha256") == artifact_sha256,
        f"{trace_path}: final SMEM artifact SHA-256 mismatch",
    )
    report = _load_json(smem_path)
    _require(
        gate.get("report") == report and gate.get("counts") == report.get("counts"),
        f"{trace_path}: final SMEM bytes/report/count binding differs",
    )
    _validate_smem_contract_report(
        smem_path,
        report,
        source_root=str(trace.get("source_root", "")),
    )
    return _artifact_record(smem_path, schema=_SMEM_CONTRACT_SCHEMA)


def _validate_trace_cases(trace_path: Path, trace: dict[str, Any]) -> dict[str, str]:
    cases = trace.get("cases")
    _require(isinstance(cases, list) and cases, f"{trace_path}: no corpus cases")
    statuses: dict[str, str] = {}
    for index, case in enumerate(cases):
        _require(isinstance(case, dict), f"{trace_path}: case {index} is not an object")
        case_id = str(case.get("id", f"index-{index}"))
        _require(
            case_id and case_id not in statuses,
            f"{trace_path}: duplicate case {case_id}",
        )
        status = str(case.get("status", ""))
        _require(
            status in {"production", "diagnostic"},
            f"{trace_path}: case {case_id} has invalid status {status!r}",
        )
        statuses[case_id] = status
        correctness = case.get("correctness")
        _require(
            isinstance(correctness, dict) and correctness.get("status") == "passed",
            f"{trace_path}: case {case_id} correctness did not pass",
        )
        graph = case.get("graph")
        _require(
            isinstance(graph, dict), f"{trace_path}: case {case_id} lacks graph gate"
        )
        graph_status = graph.get("status")
        _require(
            graph_status in {"passed", "not-applicable"},
            f"{trace_path}: case {case_id} graph status is {graph_status!r}",
        )
        if graph_status == "not-applicable":
            _require(
                status == "diagnostic",
                f"{trace_path}: production case {case_id} has no graph gate",
            )
        if status == "production":
            _require(
                graph_status == "passed",
                f"{trace_path}: production case {case_id} did not pass graph replay",
            )
    return statuses


def _validate_binding_table(
    path: Path,
) -> tuple[list[dict[str, str]], set[tuple[str, str]]]:
    rows = _csv_rows(
        path,
        _BINDING_FIELDS,
        description="case-resource binding",
        delimiter="\t",
    )
    pairs: set[tuple[str, str]] = set()
    for number, row in enumerate(rows, start=2):
        prefix = f"{path}:{number}"
        for field in (
            "case_id",
            "branch",
            "evidence_kind",
            "nodeid",
            "cache_key",
            "semantic_key",
            "kernel_id",
            "compile_spec_version",
            "compile_spec_hash",
            "compile_spec_json",
            "kernel",
        ):
            _require(row[field], f"{prefix}: empty {field}")
        for field in ("cache_key", "semantic_key", "compile_spec_hash"):
            _require(_is_sha256(row[field]), f"{prefix}: invalid {field}")
        _require(
            hashlib.sha256(row["compile_spec_json"].encode()).hexdigest()
            == row["compile_spec_hash"],
            f"{prefix}: compile spec hash mismatch",
        )
        pairs.add((row["compile_spec_hash"], row["kernel"]))
    return rows, pairs


def _validate_corpus(
    root: Path,
    *,
    gpu: int,
    cutlass: str,
) -> dict[str, Any]:
    root = root.resolve()
    _require(root.is_dir(), f"corpus root is not a directory: {root}")
    trace_path = root / "case-trace.json"
    source_path = root / "frozen-source-manifest.json"
    resources_path = root / "kernel-resources.final.csv"
    bindings_path = root / "case-resource-binding.tsv"
    smem_path = root / "smem-contracts.final.json"
    for path in (trace_path, source_path, resources_path, bindings_path, smem_path):
        _require(path.is_file(), f"corpus {root} lacks {path.name}")

    manifest, fingerprint = _validate_source_manifest(source_path)
    trace = _load_json(trace_path)
    _require(trace.get("schema") == _TRACE_SCHEMA, f"{trace_path}: invalid schema")
    _require(trace.get("complete") is True, f"{trace_path}: corpus is incomplete")
    _require(trace.get("corpus_id") == _CORPUS_ID, f"{trace_path}: wrong corpus id")
    _require(
        str(trace.get("corpus_version")) == _CORPUS_VERSION,
        f"{trace_path}: wrong corpus version",
    )
    trace_gpu = trace.get("gpu")
    _require(isinstance(trace_gpu, dict), f"{trace_path}: missing GPU identity")
    _require(
        trace_gpu.get("physical_ordinal") == gpu,
        f"{trace_path}: expected physical GPU {gpu}, got {trace_gpu!r}",
    )
    gpu_uuid = str(trace_gpu.get("uuid", ""))
    _require(_normalize_uuid(gpu_uuid), f"{trace_path}: missing GPU UUID")
    _require(
        trace.get("cutlass_packages") == _EXPECTED_PACKAGES[cutlass],
        f"{trace_path}: CUTLASS package map is not exact for {cutlass}",
    )
    source_snapshot = trace.get("source_snapshot")
    _require(isinstance(source_snapshot, dict), f"{trace_path}: missing source binding")
    _require(
        source_snapshot.get("manifest") == manifest,
        f"{trace_path}: embedded source manifest differs from artifact",
    )
    binding = source_snapshot.get("binding")
    _require(isinstance(binding, dict), f"{trace_path}: invalid source binding")
    _require(
        binding.get("schema") == SOURCE_SNAPSHOT_SCHEMA
        and binding.get("manifest_sha256") == manifest["manifest_sha256"]
        and binding.get("b12x_package_fingerprint") == fingerprint,
        f"{trace_path}: source binding does not match frozen manifest",
    )
    artifact = source_snapshot.get("artifact")
    _require(isinstance(artifact, dict), f"{trace_path}: missing source artifact hash")
    _require(
        artifact.get("sha256") == _sha256_file(source_path),
        f"{trace_path}: frozen source artifact SHA mismatch",
    )
    static = trace.get("static_validation")
    _require(isinstance(static, dict), f"{trace_path}: missing static validation")
    _require(static.get("gap_case_count") == 0, f"{trace_path}: corpus has gap cases")
    _require(static.get("gaps") == [], f"{trace_path}: corpus has static gaps")
    _require(
        static.get("uncovered_shape_branches") == [],
        f"{trace_path}: corpus has uncovered shape branches",
    )
    smem_artifact = _validate_smem_contract_finalization(
        trace_path,
        trace,
        manifest,
    )
    frontend = trace.get("frontend_ptx_capture")
    _require(
        isinstance(frontend, dict)
        and frontend.get("status") == "ok"
        and frontend.get("errors") == [],
        f"{trace_path}: frontend PTX capture is incomplete",
    )
    case_statuses = _validate_trace_cases(trace_path, trace)

    groups, row_count, missing = _read_resource_report(resources_path, strict=True)
    _require(row_count > 0 and missing == 0, f"{resources_path}: incomplete resources")
    resource_rows = [row for rows in groups.values() for row in rows]
    resource_keys: set[tuple[str, str]] = set()
    for row in resource_rows:
        key = (str(row["comparison_semantic_key"]), str(row["kernel"]))
        _require(key not in resource_keys, f"{resources_path}: duplicate row {key!r}")
        resource_keys.add(key)
        _require(
            row["package_fingerprint"] == fingerprint,
            f"{resources_path}: package fingerprint mismatch for {key!r}",
        )
        _require(
            row["architecture"] == _ARCHITECTURE,
            f"{resources_path}: non-SM120a row {key!r}",
        )
        _require(
            row["occupancy_device_ordinal"] == 0,
            f"{resources_path}: expected visible ordinal zero for {key!r}",
        )
        _require(
            _normalize_uuid(row["occupancy_gpu_uuid"]) == _normalize_uuid(gpu_uuid),
            f"{resources_path}: GPU UUID mismatch for {key!r}",
        )
        observed_packages = {
            distribution: str(row[field])
            for distribution, field in (
                ("nvidia-cutlass-dsl", "cutlass_dsl_version"),
                ("nvidia-cutlass-dsl-libs-base", "cutlass_dsl_libs_base_version"),
                ("nvidia-cutlass-dsl-libs-core", "cutlass_dsl_libs_core_version"),
                ("nvidia-cutlass-dsl-libs-cu12", "cutlass_dsl_libs_cu12_version"),
                ("nvidia-cutlass-dsl-libs-cu13", "cutlass_dsl_libs_cu13_version"),
            )
        }
        _require(
            observed_packages == _EXPECTED_PACKAGES[cutlass],
            f"{resources_path}: row {key!r} has a mixed package map",
        )

    binding_rows, binding_pairs = _validate_binding_table(bindings_path)
    for number, row in enumerate(binding_rows, start=2):
        _require(
            row["case_id"] in case_statuses,
            f"{bindings_path}:{number}: unknown case_id {row['case_id']!r}",
        )
    resource_binding_pairs = {
        (str(row["compile_spec_hash"]), str(row["kernel"])) for row in resource_rows
    }
    _require(
        binding_pairs == resource_binding_pairs,
        f"{bindings_path}: every resource specialization/symbol must be case-bound; "
        f"orphan_resources={sorted(resource_binding_pairs - binding_pairs)!r}, "
        f"orphan_bindings={sorted(binding_pairs - resource_binding_pairs)!r}",
    )
    production_binding_pairs = {
        (row["compile_spec_hash"], row["kernel"])
        for row in binding_rows
        if case_statuses[row["case_id"]] == "production"
    }
    production_spec_hashes = {spec_hash for spec_hash, _ in production_binding_pairs}
    _require(production_spec_hashes, f"{bindings_path}: no production-bound specs")
    return {
        "root": str(root),
        "gpu": gpu,
        "gpu_name": str(trace_gpu.get("name", "")),
        "gpu_uuid": gpu_uuid,
        "cutlass": cutlass,
        "packages": _EXPECTED_PACKAGES[cutlass],
        "source_fingerprint": fingerprint,
        "source_manifest": manifest,
        "resource_rows": resource_rows,
        "resource_keys": resource_keys,
        "binding_pairs": binding_pairs,
        "binding_rows": binding_rows,
        "case_statuses": case_statuses,
        "production_binding_pairs": production_binding_pairs,
        "production_spec_hashes": production_spec_hashes,
        "all_spec_hashes": {spec_hash for spec_hash, _ in resource_binding_pairs},
        "case_count": len(trace["cases"]),
        "binding_count": len(binding_rows),
        "artifacts": {
            "trace": _artifact_record(trace_path, schema=_TRACE_SCHEMA),
            "source_manifest": _artifact_record(
                source_path, schema=SOURCE_SNAPSHOT_SCHEMA
            ),
            "resources": _artifact_record(
                resources_path, schema=_RESOURCE_REPORT_SCHEMA
            ),
            "bindings": _artifact_record(bindings_path),
            "smem_contracts": smem_artifact,
        },
    }


def _resource_map(corpus: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["comparison_semantic_key"]), str(row["kernel"])): row
        for row in corpus["resource_rows"]
    }


def _validate_resource_delta(
    path: Path,
    *,
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    rows = _csv_rows(path, _DELTA_REPORT_FIELDS, description="resource delta")
    baseline_rows = _resource_map(baseline)
    current_rows = _resource_map(current)
    expected_keys = set(baseline_rows)
    _require(
        expected_keys == set(current_rows),
        f"{path}: baseline/current resource key sets differ",
    )
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for number, row in enumerate(rows, start=2):
        _require_exact_pair(row, number)
        _require(
            row.get("delta_report_schema") == _DELTA_REPORT_SCHEMA,
            f"{path}:{number}: invalid resource delta schema",
        )
        key = (row["comparison_semantic_key"], row["current_kernel"])
        _require(key not in indexed, f"{path}:{number}: duplicate delta key {key!r}")
        _require(key in expected_keys, f"{path}:{number}: unknown resource key {key!r}")
        base = baseline_rows[key]
        curr = current_rows[key]
        exact_pairs = (
            ("baseline_semantic_key", base["semantic_key"]),
            ("current_semantic_key", curr["semantic_key"]),
            ("baseline_object_sha256", base["object_sha256"]),
            ("current_object_sha256", curr["object_sha256"]),
            ("baseline_compile_spec_hash", base["compile_spec_hash"]),
            ("current_compile_spec_hash", curr["compile_spec_hash"]),
            ("baseline_package_fingerprint", base["package_fingerprint"]),
            ("current_package_fingerprint", curr["package_fingerprint"]),
            ("baseline_occupancy_gpu_uuid", base["occupancy_gpu_uuid"]),
            ("current_occupancy_gpu_uuid", curr["occupancy_gpu_uuid"]),
        )
        for field, expected in exact_pairs:
            _require(
                row[field] == str(expected),
                f"{path}:{number}: {field} disagrees with corpus report",
            )
        indexed[key] = row
    _require(set(indexed) == expected_keys, f"{path}: resource delta is incomplete")
    return rows, indexed


def _canonicalize_csv_value(field: str, value: str) -> object:
    if field.endswith("_json") and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ReleaseValidationError(f"invalid JSON in {field}: {value!r}") from exc
    return value


def _validate_sass_delta(
    path: Path,
    *,
    resource_delta: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]], str]:
    rows = _csv_rows(path, tuple(_SASS_DELTA_FIELDS), description="SASS delta")
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    normalized: list[dict[str, object]] = []
    for number, row in enumerate(rows, start=2):
        _require(
            row.get("sass_register_set_delta_schema") == _SASS_DELTA_SCHEMA,
            f"{path}:{number}: invalid SASS delta schema",
        )
        key = (row["comparison_semantic_key"], row["kernel"])
        _require(key not in indexed, f"{path}:{number}: duplicate SASS key {key!r}")
        delta = resource_delta.get(key)
        _require(delta is not None, f"{path}:{number}: no resource delta for {key!r}")
        _require_sass_delta_pair(delta, row, number)
        _sass_accounting_fields(row, row_number=number)
        indexed[key] = row
        normalized.append(
            {
                field: _canonicalize_csv_value(field, row[field])
                for field in _SASS_DELTA_FIELDS
                if field not in _SASS_PROVENANCE_ONLY_FIELDS
            }
        )
    _require(
        set(indexed) == set(resource_delta),
        f"{path}: exact SASS delta key set is incomplete",
    )
    normalized.sort(
        key=lambda row: (str(row["comparison_semantic_key"]), str(row["kernel"]))
    )
    return rows, indexed, _canonical_sha256(normalized)


def _read_accounting(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    try:
        source = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ReleaseValidationError(f"cannot read accounting {path}: {exc}") from exc
    with source:
        reader = csv.DictReader(source)
        fields = tuple(reader.fieldnames or ())
        required = {
            "accounting_schema",
            "family",
            "comparison_semantic_key",
            "baseline_semantic_key",
            "current_semantic_key",
            "symbol_sha256",
            "baseline_package_fingerprint",
            "current_package_fingerprint",
            "compile_spec_json",
            "kernel",
            "architecture",
            "baseline_cutlass_dsl",
            "current_cutlass_dsl",
            "occupancy_gpu_uuid",
            "exception_fields",
            "cause",
            "disposition",
            "evidence",
            "performance_status",
        }
        missing = sorted(required - set(fields))
        _require(not missing, f"{path}: accounting header lacks {missing!r}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()} for row in reader
        ]
    _require(rows, f"{path}: accounting has no rows")
    return rows, fields


def _validate_accounting(
    path: Path,
    *,
    resource_delta: dict[tuple[str, str], dict[str, str]],
    sass_delta: dict[tuple[str, str], dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
    tuple[str, ...],
]:
    rows, fields = _read_accounting(path)
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    allowed_dispositions = {
        "fixed-residual-flagged",
        "retained-beneficial",
        "retained-neutral",
    }
    for number, row in enumerate(rows, start=2):
        prefix = f"{path}:{number}"
        _require(row["accounting_schema"] == _ACCOUNTING_SCHEMA, f"{prefix}: schema")
        key = (row["comparison_semantic_key"], row["kernel"])
        _require(key not in indexed, f"{prefix}: duplicate key {key!r}")
        delta = resource_delta.get(key)
        sass = sass_delta.get(key)
        _require(delta is not None and sass is not None, f"{prefix}: unmatched key")
        expected_symbol = hashlib.sha256(row["kernel"].encode()).hexdigest()
        _require(row["symbol_sha256"] == expected_symbol, f"{prefix}: symbol SHA")
        _require(row["symbol_sha256"] == delta["symbol_sha256"], f"{prefix}: symbol")
        exact_fields = (
            ("baseline_semantic_key", delta["baseline_semantic_key"]),
            ("current_semantic_key", delta["current_semantic_key"]),
            ("baseline_package_fingerprint", delta["baseline_package_fingerprint"]),
            ("current_package_fingerprint", delta["current_package_fingerprint"]),
            ("compile_spec_json", delta["current_compile_spec_json"]),
            ("architecture", delta["current_architecture"]),
            ("baseline_cutlass_dsl", delta["baseline_cutlass_dsl_version"]),
            ("current_cutlass_dsl", delta["current_cutlass_dsl_version"]),
            ("occupancy_gpu_uuid", delta["current_occupancy_gpu_uuid"]),
        )
        for field, expected in exact_fields:
            _require(row[field] == expected, f"{prefix}: {field} mismatch")
        derived = _validate_and_recompute(delta)
        _, sass_exceptions, _ = _sass_accounting_fields(sass, row_number=number)
        expected_exceptions = tuple(
            dict.fromkeys((*_exception_fields(derived), *sass_exceptions))
        )
        observed_exceptions = tuple(
            field for field in row["exception_fields"].split(";") if field
        )
        _require(
            observed_exceptions == expected_exceptions,
            f"{prefix}: exception fields do not match resource/SASS evidence",
        )
        if expected_exceptions:
            for field in ("cause", "disposition", "evidence", "performance_status"):
                _require(row[field], f"{prefix}: exception lacks {field}")
            _require(
                not row["cause"].lower().startswith("unresolved"),
                f"{prefix}: unresolved exception cause",
            )
            _require(
                row["disposition"] in allowed_dispositions,
                f"{prefix}: non-final disposition {row['disposition']!r}",
            )
            _require(
                row["performance_status"] in {"pass:gpu4,gpu5", "beneficial:gpu4,gpu5"},
                f"{prefix}: exception lacks both-GPU passing performance status",
            )
        indexed[key] = row
    _require(set(indexed) == set(resource_delta), f"{path}: accounting is incomplete")
    return rows, indexed, fields


def _validate_analysis_set(
    *,
    gpu: int,
    baseline: dict[str, Any],
    current: dict[str, Any],
    resource_delta_path: Path,
    sass_delta_path: Path,
    accounting_path: Path,
) -> dict[str, Any]:
    resource_rows, resource_delta = _validate_resource_delta(
        resource_delta_path,
        baseline=baseline,
        current=current,
    )
    sass_rows, sass_delta, sass_structural_sha = _validate_sass_delta(
        sass_delta_path,
        resource_delta=resource_delta,
    )
    accounting_rows, accounting, accounting_fields = _validate_accounting(
        accounting_path,
        resource_delta=resource_delta,
        sass_delta=sass_delta,
    )
    return {
        "gpu": gpu,
        "resource_rows": resource_rows,
        "resource_delta": resource_delta,
        "sass_rows": sass_rows,
        "sass_delta": sass_delta,
        "sass_structural_sha256": sass_structural_sha,
        "accounting_rows": accounting_rows,
        "accounting": accounting,
        "accounting_fields": accounting_fields,
        "artifacts": {
            "resource_delta": _artifact_record(
                resource_delta_path, schema=_DELTA_REPORT_SCHEMA
            ),
            "sass_delta": _artifact_record(sass_delta_path, schema=_SASS_DELTA_SCHEMA),
            "accounting": _artifact_record(accounting_path, schema=_ACCOUNTING_SCHEMA),
        },
    }


def _compare_analysis_sets(gpu4: dict[str, Any], gpu5: dict[str, Any]) -> None:
    keys4 = set(gpu4["resource_delta"])
    keys5 = set(gpu5["resource_delta"])
    _require(keys4 == keys5, "GPU4/GPU5 analysis key sets differ")
    _require(
        gpu4["sass_structural_sha256"] == gpu5["sass_structural_sha256"],
        "GPU4/GPU5 normalized SASS structures differ",
    )
    _require(
        gpu4["accounting_fields"] == gpu5["accounting_fields"],
        "GPU4/GPU5 accounting headers differ",
    )
    ignored = {
        "baseline_semantic_key",
        "current_semantic_key",
        "occupancy_gpu_uuid",
    }
    for key in sorted(keys4):
        row4 = gpu4["accounting"][key]
        row5 = gpu5["accounting"][key]
        normalized4 = {
            field: value for field, value in row4.items() if field not in ignored
        }
        normalized5 = {
            field: value for field, value in row5.items() if field not in ignored
        }
        _require(
            normalized4 == normalized5,
            f"GPU4/GPU5 accounting structures differ for {key!r}",
        )


def _walk(value: object, path: tuple[str, ...] = ()) -> Any:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))


def _get_path(value: dict[str, Any], dotted: str) -> object:
    current: object = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _path_is_true(value: dict[str, Any], *paths: str) -> bool:
    return any(_get_path(value, path) is True for path in paths)


def _serving_units(path: Path, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    schema = str(artifact.get("schema", ""))
    collection = {
        "b12x.compute_exceptions.cache_abba.v1": "cases",
        "b12x.attention.mla.decode_merge.exact_cache_abba.v2": "rows",
        "b12x.attention.mla.prefill_mg.exact_cache_abba.v4": "rows",
        "b12x.residual.composite_exact_cache_abba.v1": "shapes",
        "b12x.tp_moe.dynamic.cache_abba.v2": "cases",
        "b12x.w4a16.serving.cache_abba.v2": "cases",
        "b12x.w4a16.topk_sum.cache_abba.v1": "cases",
    }.get(schema)
    if collection is None:
        return [artifact]
    raw = artifact.get(collection)
    _require(
        isinstance(raw, list) and raw,
        f"{path}: schema {schema} has no nonempty {collection} collection",
    )
    _require(
        all(isinstance(item, dict) for item in raw),
        f"{path}: schema {schema} has a malformed {collection} collection",
    )
    return list(raw)


def _unit_fixed_capacity(unit: dict[str, Any]) -> bool:
    if _path_is_true(
        unit,
        "fixed_allocation",
        "fixed_workspace",
        "fixed_workspace_capacity",
        "runtime_contract.fixed_preplanned_workspace",
    ):
        return True
    workspace = unit.get("workspace")
    if isinstance(workspace, dict):
        return workspace.get("fixed") is True and workspace.get("preplanned") is True
    fixed_workspace = unit.get("fixed_workspace")
    return isinstance(fixed_workspace, dict) and fixed_workspace.get("verified") is True


def _unit_allocator_stable(unit: dict[str, Any]) -> bool:
    if _path_is_true(
        unit,
        "allocator_stable",
        "allocator_stable_during_timing",
        "zero_replay_allocations",
    ):
        return True
    correctness = unit.get("correctness")
    comparisons = 0
    for _, observed in _walk(correctness):
        if not isinstance(observed, dict):
            continue
        if "allocator_before" in observed or "allocator_after" in observed:
            _require(
                "allocator_before" in observed and "allocator_after" in observed,
                "allocator evidence must contain both before and after counters",
            )
            _require(
                observed["allocator_before"] == observed["allocator_after"],
                "allocator counters changed across a correctness replay",
            )
            comparisons += 1
    return comparisons > 0


def _unit_input_immutable(unit: dict[str, Any]) -> bool:
    if _path_is_true(
        unit,
        "input_immutable",
        "read_only_inputs_unchanged",
        "read_only_inputs_immutable",
    ):
        return True
    read_only = unit.get("read_only_inputs")
    if not isinstance(read_only, dict) or read_only.get("unchanged") is not True:
        return False
    if "sha256_before" in read_only or "sha256_after" in read_only:
        _require(
            read_only.get("sha256_before") == read_only.get("sha256_after"),
            "read-only input hashes changed",
        )
    timed_live = read_only.get("timed_live_scenario_0")
    if isinstance(timed_live, dict):
        _require(timed_live.get("unchanged") is True, "timed live input changed")
        _require(
            timed_live.get("sha256_before") == timed_live.get("sha256_after"),
            "timed live input hashes changed",
        )
    return True


_ORACLE_METRIC_KEYS = frozenset(
    {
        "cosine",
        "max_abs",
        "max_abs_error",
        "max_rel",
        "normalized_rmse",
        "relative_l2",
        "rmse",
        "topk_equal",
    }
)


def _unit_has_oracle(unit: dict[str, Any]) -> bool:
    correctness = unit.get("correctness")
    if not isinstance(correctness, (dict, list)) or not correctness:
        return False
    metric_count = 0
    for observed_path, observed in _walk(correctness):
        if not observed_path:
            continue
        key = observed_path[-1]
        if key in {"passed", "finite"}:
            _require(
                observed is True, f"correctness gate {'.'.join(observed_path)} failed"
            )
        if key == "nonzero":
            _require(
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed > 0,
                f"correctness gate {'.'.join(observed_path)} is not positive",
            )
        if key == "topk_equal":
            _require(
                observed is True, f"oracle metric {'.'.join(observed_path)} failed"
            )
            metric_count += 1
        elif key in _ORACLE_METRIC_KEYS:
            _require(
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isfinite(float(observed)),
                f"invalid oracle metric {'.'.join(observed_path)}",
            )
            metric_count += 1
    return metric_count > 0


_ARM_BIT_EXACT_KEYS = frozenset(
    {
        "arms_bit_exact",
        "all_arms_bit_exact",
        "all_arm_outputs_bit_exact",
        "output_arms_bit_exact",
    }
)


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _strict_arm_policy_passed(policy: object) -> bool:
    return (
        isinstance(policy, dict)
        and policy.get("kind") == "strict-bit-exact"
        and isinstance(policy.get("manifest_deterministic_output"), bool)
        and policy.get("empirically_bit_exact_required") is True
        and _finite_nonnegative(policy.get("canonical_cross_arm_max_abs"))
        and policy.get("passed") is True
    )


def _nondeterministic_arm_policy_passed(policy: object) -> bool:
    if not isinstance(policy, dict):
        return False
    if (
        policy.get("kind") != "manifest-declared-nondeterministic-bf16-max-abs-envelope"
        or policy.get("manifest_deterministic_output") is not False
        or policy.get("passed") is not True
        or policy.get("scalar_abs_envelope_passed") is not True
        or policy.get("elementwise_ulp_diagnostic_passed") is not True
    ):
        return False
    replays = policy.get("replays_per_arm")
    rounds = policy.get("sentinel_rounds")
    if (
        not isinstance(replays, int)
        or isinstance(replays, bool)
        or replays < 6
        or not isinstance(rounds, int)
        or isinstance(rounds, bool)
        or rounds < 2
        or replays != rounds * 3
    ):
        return False
    if policy.get("canonical_arms_bit_exact") not in (True, False):
        return False
    finite_fields = (
        "canonical_cross_arm_max_abs",
        "cross_arm_union_max_abs_range",
        "same_arm_union_max_abs_range",
        "exact_bf16_ulp_at_cross_max",
        "scalar_abs_envelope_limit",
    )
    if not all(_finite_nonnegative(policy.get(field)) for field in finite_fields):
        return False
    if float(policy["cross_arm_union_max_abs_range"]) > float(
        policy["scalar_abs_envelope_limit"]
    ):
        return False
    integer_fields = (
        "cross_arm_union_max_bf16_ulp_range",
        "same_arm_union_max_bf16_ulp_range",
        "elementwise_violation_count",
    )
    if not all(
        isinstance(policy.get(field), int)
        and not isinstance(policy.get(field), bool)
        and int(policy[field]) >= 0
        for field in integer_fields
    ):
        return False
    if (
        policy.get("exact_bf16_ulp_margin") != 1
        or policy.get("elementwise_violation_count") != 0
        or policy.get("violation_sample_coordinates") != []
    ):
        return False
    endpoints = policy.get("cross_arm_max_abs_endpoints")
    coordinate = policy.get("cross_arm_max_abs_coordinate")
    if (
        not isinstance(endpoints, list)
        or len(endpoints) != 2
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in endpoints
        )
        or not isinstance(coordinate, list)
        or not coordinate
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in coordinate
        )
    ):
        return False
    same_arm = policy.get("same_arm_envelopes")
    if not isinstance(same_arm, dict) or len(same_arm) != 2:
        return False
    for envelope in same_arm.values():
        if not isinstance(envelope, dict):
            return False
        if envelope.get("replay_count") != replays:
            return False
        if not _finite_nonnegative(envelope.get("max_abs_range")):
            return False
        for field in ("max_bf16_ulp_range", "varying_elements"):
            value = envelope.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return False
    return True


def _arm_equality_records(value: object) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _ARM_BIT_EXACT_KEYS:
                yield child, value.get("arm_comparison_policy")
            if key != "arm_comparison_policy":
                yield from _arm_equality_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _arm_equality_records(child)


def _unit_arm_equality(unit: dict[str, Any]) -> bool:
    records = list(_arm_equality_records(unit.get("correctness")))
    if records:
        for bit_exact, policy in records:
            if bit_exact is True:
                if policy is not None and not (
                    _strict_arm_policy_passed(policy)
                    or _nondeterministic_arm_policy_passed(policy)
                ):
                    return False
                continue
            if bit_exact is not False or not _nondeterministic_arm_policy_passed(
                policy
            ):
                return False
        return True
    if _path_is_true(
        unit,
        "arms_bit_exact",
        "all_arms_bit_exact",
        "all_arm_outputs_bit_exact",
    ):
        return True
    if unit.get("all_arm_output_policies_passed") is not True:
        return False
    policies = [
        observed
        for observed_path, observed in _walk(unit)
        if observed_path and observed_path[-1] == "arm_comparison_policy"
    ]
    return bool(policies) and all(
        _strict_arm_policy_passed(policy) or _nondeterministic_arm_policy_passed(policy)
        for policy in policies
    )


def _unit_output_overwrite(unit: dict[str, Any]) -> bool:
    return _path_is_true(
        unit,
        "full_output_overwrite",
        "poisoned_output_overwritten",
        "poisoned_outputs_overwritten",
        "poisoned_checked_regions_overwritten",
        "poisoned_replay_exact",
    )


def _unit_declares_live_inputs(unit: dict[str, Any]) -> bool:
    return any(
        _get_path(unit, dotted) is not None
        for dotted in (
            "live_input_mutation",
            "live_input_scenarios_distinct",
            "read_only_inputs.timed_live_scenario_0",
        )
    ) or any(
        path and "live" in path[-1].lower() and "timing" in path[-1].lower()
        for path, _ in _walk(unit.get("correctness"))
    )


def _unit_live_input_changed_output(unit: dict[str, Any]) -> bool:
    return _path_is_true(
        unit,
        "live_input_mutation_changed_output",
        "live_input_mutation.changed_output",
    )


def _unit_live_input_changed_input(unit: dict[str, Any]) -> bool:
    return _path_is_true(
        unit,
        "live_input_mutation_changed_input",
        "live_input_mutation.changed_input",
    )


def _unit_live_input_scenarios_distinct(unit: dict[str, Any]) -> bool:
    return _path_is_true(
        unit,
        "live_input_scenarios_distinct",
        "live_input_mutation.scenarios_distinct",
    )


def _unit_live_input_hash_proof(unit: dict[str, Any]) -> bool:
    mutation = unit.get("live_input_mutation")
    if not isinstance(mutation, dict):
        return False
    if not all(
        mutation.get(key) is True
        for key in (
            "captured_graph_reused",
            "in_place",
            "same_addresses",
            "allocation_addresses_stable",
        )
    ):
        return False
    changed_inputs = mutation.get("changed_inputs")
    changed_outputs = mutation.get("changed_outputs")
    if not isinstance(changed_inputs, dict) or not changed_inputs:
        return False
    if not isinstance(changed_outputs, dict) or not changed_outputs:
        return False
    _require(
        all(value is True for value in changed_inputs.values()),
        "captured live-input hash proof contains an unchanged input",
    )
    _require(
        all(value is True for value in changed_outputs.values()),
        "captured live-input hash proof contains an unchanged arm output",
    )
    scenario_0 = mutation.get("scenario_0_sha256", mutation.get("scenario_0"))
    scenario_1 = mutation.get("scenario_1_sha256", mutation.get("scenario_1"))
    if not isinstance(scenario_0, dict) or not scenario_0:
        return False
    if not isinstance(scenario_1, dict) or not scenario_1:
        return False
    _require(
        set(scenario_0) == set(changed_inputs) == set(scenario_1),
        "live-input scenario hashes do not cover the declared changed inputs",
    )
    for name in sorted(changed_inputs):
        _require(
            _is_sha256(scenario_0[name]) and _is_sha256(scenario_1[name]),
            f"live-input scenario hash is invalid for {name}",
        )
        _require(
            scenario_0[name] != scenario_1[name],
            f"live-input scenario hash is unchanged for {name}",
        )
    output_0 = mutation.get("scenario_0_output_sha256")
    output_1 = mutation.get("scenario_1_output_sha256")
    if not isinstance(output_0, dict) or not output_0:
        return False
    if not isinstance(output_1, dict) or not output_1:
        return False
    _require(
        set(output_0) == set(changed_outputs) == set(output_1),
        "live-input output hashes do not cover the declared changed arms",
    )
    for name in sorted(changed_outputs):
        _require(
            _is_sha256(output_0[name]) and _is_sha256(output_1[name]),
            f"live-input output hash is invalid for arm {name}",
        )
        _require(
            output_0[name] != output_1[name],
            f"live-input output hash is unchanged for arm {name}",
        )
    return True


def _unit_read_only_hash_proof(unit: dict[str, Any]) -> bool:
    read_only = unit.get("read_only_inputs")
    if not isinstance(read_only, dict) or read_only.get("unchanged") is not True:
        return False
    before = read_only.get("sha256_before")
    after = read_only.get("sha256_after")
    if not isinstance(before, dict) or not before:
        return False
    if not isinstance(after, dict) or not after:
        return False
    _require(
        set(before) == set(after)
        and all(_is_sha256(value) for value in before.values())
        and all(_is_sha256(value) for value in after.values()),
        "read-only input hash proof is malformed",
    )
    _require(before == after, "read-only input hashes changed")
    timed_live = read_only.get("timed_live_scenario_0")
    if not isinstance(timed_live, dict) or timed_live.get("unchanged") is not True:
        return False
    timed_before = timed_live.get("sha256_before")
    timed_after = timed_live.get("sha256_after")
    if not isinstance(timed_before, dict) or not timed_before:
        return False
    if not isinstance(timed_after, dict) or not timed_after:
        return False
    _require(
        set(timed_before) == set(timed_after)
        and all(_is_sha256(value) for value in timed_before.values())
        and all(_is_sha256(value) for value in timed_after.values()),
        "timed live-input hash proof is malformed",
    )
    _require(timed_before == timed_after, "timed live-input hashes changed")
    return True


def _allocator_check_records(value: object) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if "before" in value or "after" in value:
            return [value]
        records: list[dict[str, Any]] = []
        for nested in value.values():
            records.extend(_allocator_check_records(nested))
        return records
    if isinstance(value, list):
        records = []
        for nested in value:
            records.extend(_allocator_check_records(nested))
        return records
    return []


def _unit_replay_allocator_proof(unit: dict[str, Any]) -> bool:
    if unit.get("zero_replay_allocations") is not True:
        return False
    checks = [
        observed
        for path, observed in _walk(unit)
        if path and path[-1] == "allocator_checks"
    ]
    records = [
        record
        for checks_value in checks
        for record in _allocator_check_records(checks_value)
    ]
    if not records:
        return False
    for record in records:
        _require(
            "before" in record and "after" in record,
            "replay allocator proof omits before/after counters",
        )
        _require(
            record["before"] == record["after"],
            "allocator counters changed during a correctness replay",
        )
        for phase in ("before", "after"):
            counters = record[phase]
            _require(
                isinstance(counters, dict)
                and set(counters) == {"allocated", "reserved"}
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in counters.values()
                ),
                f"replay allocator {phase} counters are malformed",
            )
    return True


def _validate_serving_gates(path: Path, artifact: dict[str, Any]) -> None:
    units = _serving_units(path, artifact)
    root_or_all: tuple[tuple[str, tuple[str, ...], Any], ...] = (
        (
            "cuda_graph_replay",
            ("graph_replay", "cuda_graph_replay", "runtime_contract.cuda_graph_replay"),
            lambda unit: _path_is_true(
                unit, "graph_replay", "cuda_graph_replay", "graph.cuda_graph_replay"
            ),
        ),
        (
            "same_addresses",
            (
                "same_address_arms",
                "same_address_across_arms",
                "runtime_contract.same_addresses_across_arms",
            ),
            lambda unit: _path_is_true(
                unit,
                "same_address_arms",
                "same_address_across_arms",
                "same_addresses.verified",
                "graph.same_addresses_across_arms",
            ),
        ),
        (
            "fixed_capacity",
            (
                "fixed_allocation",
                "fixed_workspace",
                "fixed_workspace_capacity",
                "runtime_contract.fixed_preplanned_workspace",
            ),
            _unit_fixed_capacity,
        ),
        (
            "graph_topology_equal",
            ("cuda_graph_topology_equal", "all_graph_topologies_equal"),
            lambda unit: _path_is_true(
                unit, "cuda_graph_topology_equal", "graph.topologies_equal"
            ),
        ),
        (
            "allocator_stable",
            (
                "allocator_stable",
                "allocator_stable_during_timing",
                "all_zero_replay_allocations",
            ),
            _unit_allocator_stable,
        ),
        (
            "input_immutable",
            (
                "input_immutable",
                "read_only_inputs_unchanged",
                "read_only_inputs_immutable",
            ),
            _unit_input_immutable,
        ),
        (
            "arm_equality",
            (
                "arms_bit_exact",
                "all_arms_bit_exact",
                "all_arm_outputs_bit_exact",
                "all_arm_output_policies_passed",
            ),
            _unit_arm_equality,
        ),
        (
            "poisoned_output_overwrite",
            (
                "full_output_overwrite",
                "poisoned_output_overwritten",
                "poisoned_outputs_overwritten",
                "poisoned_checked_regions_overwritten",
                "poisoned_replay_exact",
            ),
            _unit_output_overwrite,
        ),
    )
    for gate, root_paths, unit_validator in root_or_all:
        for dotted in root_paths:
            observed = _get_path(artifact, dotted)
            if gate == "arm_equality" and dotted in _ARM_BIT_EXACT_KEYS:
                continue
            _require(
                observed is not False,
                f"{path}: explicit serving gate {dotted} is false",
            )
        root_passed = _path_is_true(artifact, *root_paths)
        unit_passed = all(unit_validator(unit) for unit in units)
        _require(
            root_passed or unit_passed,
            f"{path}: missing independent serving gate {gate}",
        )
    for unit_index, unit in enumerate(units):
        for dotted in (
            "graph_replay",
            "cuda_graph_replay",
            "graph.cuda_graph_replay",
            "same_address_arms",
            "same_address_across_arms",
            "same_addresses.verified",
            "graph.same_addresses_across_arms",
            "fixed_allocation",
            "fixed_workspace",
            "fixed_workspace_capacity",
            "cuda_graph_topology_equal",
            "graph.topologies_equal",
            "allocator_stable",
            "allocator_stable_during_timing",
            "zero_replay_allocations",
            "input_immutable",
            "read_only_inputs_unchanged",
            "read_only_inputs_immutable",
            "full_output_overwrite",
            "poisoned_output_overwritten",
            "poisoned_outputs_overwritten",
            "poisoned_checked_regions_overwritten",
            "poisoned_replay_exact",
        ):
            _require(
                _get_path(unit, dotted) is not False,
                f"{path}: serving unit {unit_index} has false gate {dotted}",
            )
        _require(
            _unit_arm_equality(unit),
            f"{path}: serving unit {unit_index} arm equality policy failed",
        )
    _require(
        all(_unit_has_oracle(unit) for unit in units),
        f"{path}: missing real GPU-oracle metrics independent of poison/arm equality",
    )
    if "all_correct" in artifact:
        _require(artifact["all_correct"] is True, f"{path}: all_correct is not true")
    mandatory_live_schemas = {
        "b12x.attention.paged.exact_cache_abba.v1",
        "b12x.attention.indexer.exact_cache_abba.v1",
        "b12x.bf16_to_fp4_tma.cache_abba.v4",
        "b12x.contiguous_attention.cache_abba.v2",
        "b12x.residual_prefill_partial.cache_abba.v4",
        "b12x.tp_moe.dynamic.cache_abba.v2",
        "b12x.w4a16.serving.cache_abba.v2",
        "b12x.w4a16.topk_sum.cache_abba.v1",
        "b12x.w4a8.dynamic.cache_abba.v2",
    }
    schema = str(artifact.get("schema", ""))
    if schema in mandatory_live_schemas:
        _require(
            all(_unit_live_input_scenarios_distinct(unit) for unit in units),
            f"{path}: schema requires an explicit distinct live-input scenario",
        )
        _require(
            all(_unit_live_input_changed_input(unit) for unit in units),
            f"{path}: schema requires an explicit changed-input proof",
        )
        _require(
            all(_unit_live_input_changed_output(unit) for unit in units),
            f"{path}: schema requires an explicit changed-output proof",
        )
        strict_live_schemas = {
            "b12x.attention.paged.exact_cache_abba.v1",
            "b12x.attention.indexer.exact_cache_abba.v1",
            "b12x.bf16_to_fp4_tma.cache_abba.v4",
            "b12x.contiguous_attention.cache_abba.v2",
            "b12x.residual_prefill_partial.cache_abba.v4",
            "b12x.tp_moe.dynamic.cache_abba.v2",
            "b12x.w4a16.serving.cache_abba.v2",
            "b12x.w4a16.topk_sum.cache_abba.v1",
            "b12x.w4a8.dynamic.cache_abba.v2",
        }
        if schema in strict_live_schemas:
            _require(
                all(_unit_live_input_hash_proof(unit) for unit in units),
                f"{path}: schema requires in-place live-input/output hash proof",
            )
            _require(
                all(_unit_read_only_hash_proof(unit) for unit in units),
                f"{path}: schema requires read-only and timed-input hash proof",
            )
            _require(
                all(_unit_replay_allocator_proof(unit) for unit in units),
                f"{path}: schema requires per-replay allocator counter proof",
            )
    else:
        live_units = [unit for unit in units if _unit_declares_live_inputs(unit)]
        if live_units:
            _require(
                all(_unit_live_input_changed_output(unit) for unit in live_units),
                f"{path}: live-input benchmark lacks an explicit changed-output proof",
            )


def _toolchain_package_map(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, list):
        raw = {
            str(item[0]): item[1]
            for item in value
            if isinstance(item, list) and len(item) >= 2
        }
    else:
        return {}
    return {
        distribution: str(raw.get(toolchain_key, ""))
        for toolchain_key, distribution in _TOOLCHAIN_PACKAGE_NAMES.items()
    }


def _provenance_records(
    path: Path,
    artifact: dict[str, Any],
    *,
    source_fingerprint: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], set[str]]:
    by_spec: dict[str, dict[str, dict[str, Any]]] = {}
    object_hashes: set[str] = set()
    for record_path, value in _walk(artifact):
        if not isinstance(value, dict) or "compile_spec_hash" not in value:
            continue
        required = {
            "compile_spec_hash",
            "compile_spec_json",
            "kernel_id",
            "package_fingerprint",
            "toolchain",
            "object_sha256",
            "manifest_sha256",
        }
        if not required <= set(value):
            continue
        spec_hash = value.get("compile_spec_hash")
        _require(_is_sha256(spec_hash), f"{path}: invalid provenance spec hash")
        compile_json = value.get("compile_spec_json")
        _require(isinstance(compile_json, str), f"{path}: missing compile spec JSON")
        _require(
            hashlib.sha256(compile_json.encode()).hexdigest() == spec_hash,
            f"{path}: provenance compile spec hash mismatch",
        )
        _require(
            value.get("package_fingerprint") == source_fingerprint,
            f"{path}: object package fingerprint differs from frozen source",
        )
        packages = _toolchain_package_map(value.get("toolchain"))
        cutlass = packages.get("nvidia-cutlass-dsl", "")
        _require(
            cutlass in _EXPECTED_PACKAGES,
            f"{path}: object provenance lacks a supported CUTLASS arm",
        )
        _require(
            packages == _EXPECTED_PACKAGES[cutlass],
            f"{path}: object provenance has a mixed CUTLASS package map",
        )
        object_sha = value.get("object_sha256")
        manifest_sha = value.get("manifest_sha256")
        _require(_is_sha256(object_sha), f"{path}: invalid object SHA")
        _require(_is_sha256(manifest_sha), f"{path}: invalid manifest SHA")
        object_hashes.add(str(object_sha))
        arm_records = by_spec.setdefault(str(spec_hash), {})
        normalized = {
            "compile_spec_json": compile_json,
            "kernel_id": str(value.get("kernel_id", "")),
            "package_fingerprint": source_fingerprint,
            "packages": packages,
            "object_sha256": str(object_sha),
            "manifest_sha256": str(manifest_sha),
        }
        existing = arm_records.get(cutlass)
        if existing is not None:
            existing_without_paths = {
                key: item for key, item in existing.items() if key != "paths"
            }
            if existing_without_paths != normalized:
                _fail(f"{path}: ambiguous {cutlass} provenance for spec {spec_hash}")
            existing["paths"].append(".".join(record_path))
        else:
            arm_records[cutlass] = {
                **normalized,
                "paths": [".".join(record_path)],
            }
    _require(by_spec, f"{path}: no exact-object compile provenance records")
    for spec_hash, arms in by_spec.items():
        _require(
            set(arms) == set(_EXPECTED_PACKAGES),
            f"{path}: spec {spec_hash} lacks exact provenance for both arms",
        )
        _require(
            arms["4.5.2"]["compile_spec_json"] == arms["4.6.0"]["compile_spec_json"],
            f"{path}: arm compile specs differ for {spec_hash}",
        )
        _require(
            arms["4.5.2"]["kernel_id"] == arms["4.6.0"]["kernel_id"],
            f"{path}: arm kernel IDs differ for {spec_hash}",
        )
    return by_spec, object_hashes


def _verification_identities(
    path: Path,
    value: object,
    *,
    description: str,
) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for record_path, observed in _walk(value):
        if not isinstance(observed, dict):
            continue
        present = {"manifest_sha256", "object_sha256"} & set(observed)
        if not present:
            continue
        _require(
            present == {"manifest_sha256", "object_sha256"},
            f"{path}: {description} has a partial artifact hash record at "
            f"{'.'.join(record_path)}",
        )
        manifest_sha = observed["manifest_sha256"]
        object_sha = observed["object_sha256"]
        _require(
            _is_sha256(manifest_sha) and _is_sha256(object_sha),
            f"{path}: {description} has invalid artifact hashes at "
            f"{'.'.join(record_path)}",
        )
        if "object_bytes" in observed:
            _require(
                isinstance(observed["object_bytes"], int)
                and not isinstance(observed["object_bytes"], bool)
                and observed["object_bytes"] > 0,
                f"{path}: {description} has invalid object_bytes",
            )
        if "passed" in observed:
            _require(
                observed["passed"] is True,
                f"{path}: {description} verification status is not true",
            )
        identities.add((str(manifest_sha), str(object_sha)))
    _require(identities, f"{path}: {description} has no exact artifact hash records")
    return identities


def _validate_artifact_integrity(
    path: Path,
    artifact: dict[str, Any],
    provenance: dict[str, dict[str, dict[str, Any]]],
) -> None:
    expected = {
        (record["manifest_sha256"], record["object_sha256"])
        for arms in provenance.values()
        for record in arms.values()
    }
    proven: set[tuple[str, str]] = set()
    proof_pairs = 0
    for parent_path, observed in _walk(artifact):
        if not isinstance(observed, dict):
            continue
        has_before = "artifact_verification_before" in observed
        has_after = "artifact_verification_after" in observed
        if not has_before and not has_after:
            continue
        _require(
            has_before and has_after,
            f"{path}: {'.'.join(parent_path) or '<root>'} must record both "
            "artifact_verification_before and artifact_verification_after",
        )
        before = observed["artifact_verification_before"]
        after = observed["artifact_verification_after"]
        _require(
            before == after,
            f"{path}: exact-cache artifact verification changed at "
            f"{'.'.join(parent_path) or '<root>'}",
        )
        before_ids = _verification_identities(
            path,
            before,
            description=f"{'.'.join(parent_path)} artifact verification before",
        )
        after_ids = _verification_identities(
            path,
            after,
            description=f"{'.'.join(parent_path)} artifact verification after",
        )
        _require(before_ids == after_ids, f"{path}: before/after artifact IDs differ")
        proven.update(before_ids)
        proof_pairs += 1

    integrity = artifact.get("artifact_integrity")
    if integrity is not None:
        _require(isinstance(integrity, dict), f"{path}: malformed artifact_integrity")
        _require(
            integrity.get("immutable") is True,
            f"{path}: artifact_integrity.immutable is not true",
        )
        initial = integrity.get("initial")
        final = integrity.get("final")
        _require(
            initial is not None and final is not None and initial == final,
            f"{path}: artifact_integrity initial/final records differ or are missing",
        )
        proven.update(
            _verification_identities(path, initial, description="integrity initial")
        )
        _require(
            _verification_identities(path, final, description="integrity final")
            == _verification_identities(path, initial, description="integrity initial"),
            f"{path}: artifact_integrity initial/final identities differ",
        )
        by_rows = integrity.get("by_rows", {})
        _require(isinstance(by_rows, dict), f"{path}: malformed integrity by_rows")
        for row_name, row_proof in by_rows.items():
            _require(
                isinstance(row_proof, dict)
                and "before" in row_proof
                and "after" in row_proof
                and row_proof["before"] == row_proof["after"],
                f"{path}: artifact integrity changed or is missing for row {row_name}",
            )
            proven.update(
                _verification_identities(
                    path,
                    row_proof["before"],
                    description=f"integrity row {row_name} before",
                )
            )
        proof_pairs += 1

    _require(proof_pairs > 0, f"{path}: no exact before/after artifact proof")
    _require(
        proven == expected,
        f"{path}: artifact proof does not exactly cover provenance; "
        f"missing={sorted(expected - proven)!r}, extra={sorted(proven - expected)!r}",
    )


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


def _validate_gpu_mode_snapshot(
    path: Path,
    snapshot: object,
    *,
    gpu: int,
    gpu_uuid: str,
    description: str,
) -> dict[str, Any]:
    _require(isinstance(snapshot, dict), f"{path}: {description} is not an object")
    _require(snapshot.get("available") is True, f"{path}: {description} unavailable")
    fields = snapshot.get("fields")
    _require(isinstance(fields, dict), f"{path}: {description} lacks mode fields")
    missing = [field for field in _GPU_MODE_FIELDS if not str(fields.get(field, ""))]
    _require(not missing, f"{path}: {description} lacks GPU mode fields {missing!r}")
    _require(
        str(fields["index"]) == str(gpu), f"{path}: {description} GPU index mismatch"
    )
    _require(
        _normalize_uuid(str(fields["uuid"])) == _normalize_uuid(gpu_uuid),
        f"{path}: {description} GPU UUID mismatch",
    )
    for key in ("torch_uuid", "nvidia_smi_uuid"):
        _require(
            _normalize_uuid(str(snapshot.get(key, ""))) == _normalize_uuid(gpu_uuid),
            f"{path}: {description} {key} mismatch",
        )
    captured = snapshot.get("captured_unix_ns")
    _require(
        isinstance(captured, int) and not isinstance(captured, bool) and captured > 0,
        f"{path}: {description} has no capture timestamp",
    )
    return snapshot


def _validate_gpu_mode_pairs(
    path: Path,
    artifact: dict[str, Any],
    *,
    gpu: int,
    gpu_uuid: str,
) -> None:
    key_pairs = (
        ("gpu_mode_initial", "gpu_mode_final"),
        ("gpu_mode_before", "gpu_mode_after"),
        ("gpu_mode_before_timing", "gpu_mode_after_timing"),
        ("mode_initial", "mode_final"),
    )
    pair_count = 0
    stable_fields = ("index", "uuid", "persistence_mode", "compute_mode", "power.limit")
    for parent_path, observed in _walk(artifact):
        if not isinstance(observed, dict):
            continue
        for before_key, after_key in key_pairs:
            if before_key not in observed and after_key not in observed:
                continue
            location = ".".join(parent_path) or "<root>"
            _require(
                before_key in observed and after_key in observed,
                f"{path}: {location} lacks a complete {before_key}/{after_key} pair",
            )
            before = _validate_gpu_mode_snapshot(
                path,
                observed[before_key],
                gpu=gpu,
                gpu_uuid=gpu_uuid,
                description=f"{location}.{before_key}",
            )
            after = _validate_gpu_mode_snapshot(
                path,
                observed[after_key],
                gpu=gpu,
                gpu_uuid=gpu_uuid,
                description=f"{location}.{after_key}",
            )
            _require(
                int(after["captured_unix_ns"]) > int(before["captured_unix_ns"]),
                f"{path}: {location} GPU mode timestamps are not ordered",
            )
            before_fields = before["fields"]
            after_fields = after["fields"]
            _require(
                all(
                    before_fields[field] == after_fields[field]
                    for field in stable_fields
                ),
                f"{path}: {location} benchmark GPU mode identity changed",
            )
            pair_count += 1
    _require(pair_count > 0, f"{path}: no before/after physical GPU mode snapshots")


def _validate_result_hash(path: Path, artifact: dict[str, Any]) -> None:
    if "result_sha256" not in artifact:
        return
    payload = {key: value for key, value in artifact.items() if key != "result_sha256"}
    _require(
        artifact["result_sha256"] == _canonical_sha256(payload),
        f"{path}: result_sha256 is invalid",
    )


def _physical_gpu(artifact: dict[str, Any]) -> tuple[int | None, str, list[Any]]:
    gpu = artifact.get("gpu")
    if not isinstance(gpu, dict):
        return None, "", []
    physical: int | None = None
    for key in ("physical_index", "expected_physical_index", "physical_ordinal"):
        value = gpu.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            physical = value
            break
    if physical is None:
        for key in ("visible_devices", "cuda_visible_devices"):
            value = str(gpu.get(key, "")).strip()
            if value in {"4", "5"}:
                physical = int(value)
                break
    return physical, str(gpu.get("uuid", "")), list(gpu.get("capability", []))


def _summary_arm_versions(
    artifact: dict[str, Any], summaries: dict[str, Any]
) -> dict[str, str]:
    labels = artifact.get("labels")
    result: dict[str, str] = {}
    if isinstance(labels, dict):
        a = labels.get("a")
        b = labels.get("b")
        if isinstance(a, str):
            result[a] = "4.5.2"
        if isinstance(b, str):
            result[b] = "4.6.0"
    for label in summaries:
        if "4.5.2" in label:
            result[label] = "4.5.2"
        elif "4.6.0" in label:
            result[label] = "4.6.0"
    _require(
        set(result) == set(summaries)
        and set(result.values()) == set(_EXPECTED_PACKAGES),
        "timing summary labels do not unambiguously identify 4.5.2 and 4.6.0",
    )
    return result


def _percentile_linear(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = percentile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _adapter_reported_p95(samples: list[float], _schema: str) -> float:
    ordered = sorted(samples)
    index = int(0.95 * (len(ordered) - 1))
    return ordered[index]


def _value_at_path(artifact: object, path: tuple[str, ...]) -> object:
    current = artifact
    for part in path:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            return None
    return current


def _explicit_spec_hashes(value: object, known: set[str]) -> set[str]:
    hashes: set[str] = set()
    accepted_keys = {
        "compile_spec_hash",
        "compile_spec_hashes",
        "target_spec_hashes",
        "all_graph_spec_hashes",
        "spec_hash",
        "spec_hashes",
    }
    for path, observed in _walk(value):
        if not path or path[-1] not in accepted_keys:
            continue
        candidates = observed if isinstance(observed, list) else [observed]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate in known:
                hashes.add(candidate)
    return hashes


def _timing_spec_hashes(
    artifact: dict[str, Any],
    timing_path: tuple[str, ...],
    known: set[str],
) -> set[str]:
    for prefix_length in range(len(timing_path), -1, -1):
        ancestor = _value_at_path(artifact, timing_path[:prefix_length])
        hashes = _explicit_spec_hashes(ancestor, known)
        if hashes:
            return hashes
    return set()


def _timing_condition(
    artifact: dict[str, Any], path: tuple[str, ...]
) -> tuple[str, int]:
    if "warm_l2" in path:
        condition = "warm_l2"
    elif "cold_l2" in path:
        condition = "cold_l2"
    else:
        l2_policy = str(artifact.get("l2_policy", "")).lower()
        if l2_policy in {"warm", "warm_l2"} or artifact.get("cold_l2") is False:
            condition = "warm_l2"
        elif l2_policy in {"cold", "cold_l2"} or artifact.get("cold_l2") is True:
            condition = "cold_l2"
        else:
            flush = artifact.get("l2_flush_bytes")
            if isinstance(flush, int) and not isinstance(flush, bool):
                condition = "cold_l2" if flush > 0 else "warm_l2"
            else:
                _fail(f"cannot classify L2 condition for timing path {'.'.join(path)}")
    flush_bytes = 0
    for prefix_length in range(len(path), -1, -1):
        current: object = artifact
        for part in path[:prefix_length]:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                current = None
                break
        if isinstance(current, dict):
            raw = current.get("l2_flush_bytes")
            if isinstance(raw, int) and not isinstance(raw, bool):
                flush_bytes = raw
                break
    if not flush_bytes:
        raw = artifact.get("l2_flush_bytes")
        if isinstance(raw, int) and not isinstance(raw, bool):
            flush_bytes = raw
    _require(
        condition != "cold_l2" or flush_bytes > 0,
        f"cold-L2 timing path {'.'.join(path)} lacks a positive flush size",
    )
    return condition, flush_bytes


def _recompute_abba_summary(samples: list[float]) -> dict[str, object]:
    _require(bool(samples), "cannot summarize an empty ABBA sample set")
    ordered = sorted(samples)
    trim = int(0.01 * len(ordered))
    trimmed = ordered[trim:-trim] if trim else ordered
    return {
        "count": len(samples),
        "mean_us": statistics.mean(samples),
        "trimmed_mean_us": statistics.mean(trimmed),
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "p05_us": ordered[int(0.05 * (len(ordered) - 1))],
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))],
        "samples_us": samples,
    }


def _require_abba_summary_matches(
    observed: object,
    samples: list[float],
    *,
    path: Path,
    context: str,
    exact_sample_order: bool,
) -> None:
    expected = _recompute_abba_summary(samples)
    expected_fields = set(expected)
    _require(
        isinstance(observed, dict) and set(observed) == expected_fields,
        f"{path}: {context} summary fields are incomplete or unexpected",
    )
    _require(
        observed["count"] == expected["count"],
        f"{path}: {context} reported sample count differs",
    )
    observed_samples = observed["samples_us"]
    _require(
        isinstance(observed_samples, list) and len(observed_samples) == len(samples),
        f"{path}: {context} reported samples are malformed",
    )
    normalized_samples: list[float] = []
    for index, value in enumerate(observed_samples):
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0,
            f"{path}: {context} sample {index} is invalid",
        )
        normalized_samples.append(float(value))
    expected_samples = samples if exact_sample_order else sorted(samples)
    comparable_samples = (
        normalized_samples if exact_sample_order else sorted(normalized_samples)
    )
    _require(
        len(comparable_samples) == len(expected_samples)
        and all(
            math.isclose(observed_value, expected_value, rel_tol=1e-12, abs_tol=1e-12)
            for observed_value, expected_value in zip(
                comparable_samples, expected_samples, strict=True
            )
        ),
        f"{path}: {context} samples do not reconstruct from inner events",
    )
    for field in (
        "mean_us",
        "trimmed_mean_us",
        "median_us",
        "min_us",
        "p05_us",
        "p95_us",
    ):
        value = observed[field]
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and math.isclose(
                float(value),
                float(expected[field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"{path}: {context} {field} does not reconstruct from inner events",
        )


def _timing_clock_mhz(
    path: Path,
    snapshot: dict[str, Any],
    field: str,
    *,
    description: str,
) -> float:
    fields = snapshot.get("fields")
    _require(
        isinstance(fields, dict),
        f"{path}: {description} lacks physical GPU mode fields",
    )
    raw = fields.get(field)
    try:
        value = float(str(raw).split()[0])
    except (IndexError, ValueError):
        _fail(f"{path}: {description} has an invalid {field}: {raw!r}")
    _require(
        math.isfinite(value) and value > 0.0,
        f"{path}: {description} has a nonpositive {field}: {raw!r}",
    )
    return value


def _timing_active_throttle_reasons(
    path: Path,
    snapshot: dict[str, Any],
    *,
    description: str,
) -> int:
    fields = snapshot.get("fields")
    _require(
        isinstance(fields, dict),
        f"{path}: {description} lacks physical GPU mode fields",
    )
    raw = fields.get("clocks_throttle_reasons.active")
    try:
        value = int(str(raw).strip(), 0)
    except ValueError:
        _fail(f"{path}: {description} has invalid active throttle reasons: {raw!r}")
    _require(
        value >= 0,
        f"{path}: {description} has negative active throttle reasons: {raw!r}",
    )
    return value


def _timing_required_active_throttle_reasons(
    path: Path,
    value: object,
    *,
    description: str,
) -> int:
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in _PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS,
        f"{path}: {description} must be exactly 0 or 0x4",
    )
    return int(value)


def _validate_timing_condition_envelope(
    path: Path,
    timing_path: tuple[str, ...],
    timing: dict[str, Any],
    summaries: dict[str, Any],
    condition: object,
    *,
    gpu: int,
    gpu_uuid: str,
) -> None:
    """Validate the condition state surrounding one aggregate sample schedule."""

    location = ".".join(timing_path)
    _require(
        timing_path and timing_path[-1] == "timings" and isinstance(condition, dict),
        f"{path}: {location} must be nested under one complete timing condition",
    )
    cold_l2 = timing.get("cold_l2")
    _require(
        isinstance(cold_l2, bool) and condition.get("cold_l2") is cold_l2,
        f"{path}: {location} condition/timing cold-L2 states differ",
    )
    flush_bytes = condition.get("l2_flush_bytes")
    _require(
        isinstance(flush_bytes, int)
        and not isinstance(flush_bytes, bool)
        and ((cold_l2 and flush_bytes > 0) or (not cold_l2 and flush_bytes == 0)),
        f"{path}: {location} has an invalid condition-level L2 flush size",
    )

    allocator_before = condition.get("allocator_before")
    allocator_after = condition.get("allocator_after")
    allocator_keys = {"allocated", "reserved"}
    _require(
        isinstance(allocator_before, dict)
        and isinstance(allocator_after, dict)
        and set(allocator_before) == allocator_keys
        and set(allocator_after) == allocator_keys
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for counters in (allocator_before, allocator_after)
            for value in counters.values()
        )
        and allocator_before == allocator_after
        and condition.get("allocator_stable") is True,
        f"{path}: {location} lacks an exact stable timing allocator proof",
    )

    preconditioning = condition.get("preconditioning")
    _require(
        isinstance(preconditioning, dict)
        and preconditioning.get("policy") == "balanced_abba_target_graph_duration",
        f"{path}: {location} lacks balanced target-graph duration preconditioning",
    )
    minimum_cycles = preconditioning.get("minimum_cycles")
    completed_cycles = preconditioning.get("completed_cycles")
    _require(
        isinstance(minimum_cycles, int)
        and not isinstance(minimum_cycles, bool)
        and minimum_cycles > 0
        and isinstance(completed_cycles, int)
        and not isinstance(completed_cycles, bool)
        and completed_cycles >= minimum_cycles,
        f"{path}: {location} has invalid preconditioning cycle counts",
    )
    duration_values: dict[str, float] = {}
    for field in (
        "minimum_active_seconds",
        "maximum_active_seconds",
        "observed_active_seconds",
    ):
        value = preconditioning.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0,
            f"{path}: {location} has an invalid preconditioning {field}",
        )
        duration_values[field] = float(value)
    _require(
        duration_values["observed_active_seconds"]
        >= duration_values["minimum_active_seconds"]
        and duration_values["maximum_active_seconds"]
        >= duration_values["observed_active_seconds"]
        and duration_values["minimum_active_seconds"]
        >= _MIN_TIMING_PRECONDITION_SECONDS
        and duration_values["maximum_active_seconds"]
        <= _MAX_TIMING_PRECONDITION_SECONDS,
        f"{path}: {location} did not satisfy its active-duration envelope",
    )
    labels = set(summaries)
    replay_counts = preconditioning.get("target_graph_replays_by_label")
    _require(
        isinstance(replay_counts, dict)
        and set(replay_counts) == labels
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in replay_counts.values()
        )
        and len(set(replay_counts.values())) == 1
        and next(iter(replay_counts.values())) == 2 * completed_cycles,
        f"{path}: {location} preconditioning is not balanced across exact arms",
    )
    _require(
        preconditioning.get("cold_l2_flush_before_every_replay") is cold_l2
        and preconditioning.get("flush_inside_timed_interval") is False
        and preconditioning.get("required_pstate") == _REQUIRED_TIMING_PSTATE,
        f"{path}: {location} preconditioning has an invalid cache/P-state policy",
    )
    required_throttle_reasons = _timing_required_active_throttle_reasons(
        path,
        preconditioning.get("required_active_throttle_reasons"),
        description=(f"{location}.preconditioning.required_active_throttle_reasons"),
    )

    before = _validate_gpu_mode_snapshot(
        path,
        condition.get("gpu_mode_before_timing"),
        gpu=gpu,
        gpu_uuid=gpu_uuid,
        description=f"{location}.gpu_mode_before_timing",
    )
    after = _validate_gpu_mode_snapshot(
        path,
        condition.get("gpu_mode_after_timing"),
        gpu=gpu,
        gpu_uuid=gpu_uuid,
        description=f"{location}.gpu_mode_after_timing",
    )
    _require(
        int(after["captured_unix_ns"]) > int(before["captured_unix_ns"]),
        f"{path}: {location} timing mode timestamps are not ordered",
    )
    before_fields = before["fields"]
    after_fields = after["fields"]
    _require(
        before_fields["pstate"] == _REQUIRED_TIMING_PSTATE
        and after_fields["pstate"] == _REQUIRED_TIMING_PSTATE,
        f"{path}: {location} timing did not remain in P1",
    )
    mode_probes = preconditioning.get("mode_probes")
    _require(
        isinstance(mode_probes, list) and mode_probes,
        f"{path}: {location} preconditioning lacks physical GPU mode probes",
    )
    last_probe = _validate_gpu_mode_snapshot(
        path,
        mode_probes[-1],
        gpu=gpu,
        gpu_uuid=gpu_uuid,
        description=f"{location}.preconditioning.mode_probes[-1]",
    )
    _require(
        last_probe["fields"]["pstate"] == _REQUIRED_TIMING_PSTATE
        and int(last_probe["captured_unix_ns"]) < int(before["captured_unix_ns"]),
        f"{path}: {location} preconditioning did not establish P1 before timing",
    )
    before_throttle = _timing_active_throttle_reasons(
        path, before, description=f"{location} before timing"
    )
    after_throttle = _timing_active_throttle_reasons(
        path, after, description=f"{location} after timing"
    )
    probe_throttle = _timing_active_throttle_reasons(
        path,
        last_probe,
        description=f"{location} final preconditioning probe",
    )
    _require(
        before_throttle
        == after_throttle
        == probe_throttle
        == required_throttle_reasons,
        f"{path}: {location} does not match its exact requested active "
        "clock-throttle reasons mask",
    )
    _require(
        all(
            before_fields[field] == after_fields[field]
            for field in _TIMING_STABLE_MODE_FIELDS
        ),
        f"{path}: {location} timing identity/mode fields changed",
    )

    before_sm = _timing_clock_mhz(
        path, before, "clocks.current.sm", description=f"{location} before timing"
    )
    after_sm = _timing_clock_mhz(
        path, after, "clocks.current.sm", description=f"{location} after timing"
    )
    before_memory = _timing_clock_mhz(
        path,
        before,
        "clocks.current.memory",
        description=f"{location} before timing",
    )
    after_memory = _timing_clock_mhz(
        path,
        after,
        "clocks.current.memory",
        description=f"{location} after timing",
    )
    sm_delta = abs(after_sm - before_sm)
    stability = condition.get("gpu_mode_stability")
    _require(
        isinstance(stability, dict)
        and stability.get("schema") == _GPU_MODE_STABILITY_SCHEMA
        and stability.get("required_pstate") == _REQUIRED_TIMING_PSTATE
        and stability.get("required_memory_clock_equality") is True
        and stability.get("required_active_throttle_reasons")
        == required_throttle_reasons
        and stability.get("max_sm_clock_delta_mhz") == _MAX_TIMING_SM_CLOCK_DELTA_MHZ
        and stability.get("stable_identity_and_mode_fields")
        == _TIMING_STABLE_MODE_FIELDS
        and stability.get("passed") is True,
        f"{path}: {location} has an invalid GPU mode-stability policy",
    )
    _require(
        sm_delta <= _MAX_TIMING_SM_CLOCK_DELTA_MHZ and before_memory == after_memory,
        f"{path}: {location} timing clocks are not release-stable",
    )
    _require(
        all(
            isinstance(stability.get(field), int)
            and not isinstance(stability.get(field), bool)
            and stability[field] == expected
            for field, expected in (
                ("observed_before_active_throttle_reasons", before_throttle),
                ("observed_after_active_throttle_reasons", after_throttle),
            )
        ),
        f"{path}: {location} throttle-reason report does not match its snapshots",
    )
    observed_positive_clock_fields = {
        "observed_before_sm_clock_mhz": before_sm,
        "observed_after_sm_clock_mhz": after_sm,
        "observed_memory_clock_mhz": before_memory,
    }
    _require(
        all(
            isinstance(stability.get(field), (int, float))
            and not isinstance(stability.get(field), bool)
            and math.isfinite(float(stability[field]))
            and float(stability[field]) > 0.0
            and math.isclose(
                float(stability[field]), expected, rel_tol=1e-12, abs_tol=1e-12
            )
            for field, expected in observed_positive_clock_fields.items()
        )
        and isinstance(stability.get("observed_sm_clock_delta_mhz"), (int, float))
        and not isinstance(stability.get("observed_sm_clock_delta_mhz"), bool)
        and math.isfinite(float(stability["observed_sm_clock_delta_mhz"]))
        and float(stability["observed_sm_clock_delta_mhz"]) >= 0.0
        and math.isclose(
            float(stability["observed_sm_clock_delta_mhz"]),
            sm_delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        f"{path}: {location} reported GPU clocks do not match its snapshots",
    )


def _validate_aggregated_timing(
    path: Path,
    timing_path: tuple[str, ...],
    timing: dict[str, Any],
    summaries: dict[str, Any],
    *,
    condition: object,
    gpu: int,
    gpu_uuid: str,
) -> int:
    """Reconstruct aggregate samples from every independently timed replay."""

    aggregate_fields = {
        "orders",
        "replays_per_reported_sample",
        "aggregation",
        "event_pool",
        "inner_samples_by_position",
        "inner_sample_count_by_label",
    }
    present_fields = aggregate_fields & set(timing)
    _require(
        present_fields == aggregate_fields,
        f"{path}: {'.'.join(timing_path)} must include the complete aggregate "
        "timing contract even when K=1",
    )
    replays = timing["replays_per_reported_sample"]
    _require(
        isinstance(replays, int) and not isinstance(replays, bool) and replays >= 1,
        f"{path}: {'.'.join(timing_path)} has an invalid aggregate replay count",
    )
    cold_l2 = timing.get("cold_l2")
    _require(
        isinstance(cold_l2, bool),
        f"{path}: {'.'.join(timing_path)} omits aggregate cold-L2 state",
    )
    _require(
        timing["aggregation"]
        == {
            "reported_sample": "arithmetic_mean_us",
            "inner_event_bracketing": "independent_per_graph_replay",
            "inner_schedule": "full_abba_order_per_repetition",
            "flush_before_every_inner_replay": cold_l2,
            "flush_inside_timed_interval": False,
        },
        f"{path}: {'.'.join(timing_path)} has an invalid aggregate timing policy",
    )
    _validate_timing_condition_envelope(
        path,
        timing_path,
        timing,
        summaries,
        condition,
        gpu=gpu,
        gpu_uuid=gpu_uuid,
    )
    event_pool = timing["event_pool"]
    _require(
        isinstance(event_pool, dict)
        and event_pool.get("schema") == _CUDA_EVENT_POOL_SCHEMA
        and event_pool.get("allocation_phase") == "before_reported_samples"
        and event_pool.get("prewarm_phase") == "before_reported_samples"
        and event_pool.get("prewarm_each_event") is True
        and event_pool.get("one_pair_per_inner_replay") is True
        and event_pool.get("event_creation_inside_sample_schedule") is False
        and event_pool.get("reuse_boundary")
        == "after_stream_synchronize_and_elapsed_query"
        and event_pool.get("initialized_before_target_graph_preconditioning") is True,
        f"{path}: {'.'.join(timing_path)} has invalid CUDA event-pool provenance",
    )
    event_batch_cycles = event_pool.get("event_batch_cycles")
    _require(
        isinstance(event_batch_cycles, int)
        and not isinstance(event_batch_cycles, bool)
        and event_batch_cycles > 0,
        f"{path}: {'.'.join(timing_path)} has an invalid event-pool batch size",
    )
    _require(
        _is_sha256(event_pool.get("event_handle_sha256"))
        and _is_sha256(event_pool.get("prewarm_elapsed_sha256")),
        f"{path}: {'.'.join(timing_path)} has malformed event-pool digests",
    )
    orders = timing["orders"]
    _require(
        isinstance(orders, (list, tuple))
        and len(orders) == 2
        and all(
            isinstance(order, (list, tuple)) and len(order) == 4 for order in orders
        ),
        f"{path}: {'.'.join(timing_path)} must declare exactly two four-position orders",
    )
    order_0 = list(orders[0])
    order_1 = list(orders[1])
    arm_a, arm_b = order_0[0], order_0[1]
    _require(
        isinstance(arm_a, str)
        and isinstance(arm_b, str)
        and arm_a != arm_b
        and order_0 == [arm_a, arm_b, arm_b, arm_a]
        and order_1 == [arm_b, arm_a, arm_a, arm_b]
        and set(summaries) == {arm_a, arm_b},
        f"{path}: {'.'.join(timing_path)} does not declare exact ABBA/BAAB orders",
    )
    raw_by_position = timing["inner_samples_by_position"]
    position_summaries = timing.get("position_summaries")
    inner_counts = timing["inner_sample_count_by_label"]
    _require(
        isinstance(raw_by_position, dict)
        and isinstance(position_summaries, dict)
        and isinstance(inner_counts, dict)
        and set(raw_by_position) == set(position_summaries)
        and set(inner_counts) == set(summaries),
        f"{path}: {'.'.join(timing_path)} aggregate maps are incomplete",
    )
    expected_position_keys = {
        f"{order_index}:{position}:{label}"
        for order_index, order in enumerate((order_0, order_1))
        for position, label in enumerate(order)
    }
    _require(
        set(raw_by_position) == expected_position_keys,
        f"{path}: {'.'.join(timing_path)} must contain all eight exact ABBA "
        "position keys",
    )
    reported_by_label: dict[str, list[float]] = {str(label): [] for label in summaries}
    observed_inner_counts = {str(label): 0 for label in summaries}
    recomputed_by_position: dict[str, list[float]] = {}
    position_counts: set[int] = set()
    for position_key, raw_groups in raw_by_position.items():
        _require(
            isinstance(position_key, str),
            f"{path}: aggregate position key is not a string",
        )
        label = position_key.rsplit(":", 1)[-1]
        _require(
            label in reported_by_label,
            f"{path}: aggregate position {position_key!r} has an unknown arm",
        )
        position_summary = position_summaries[position_key]
        _require(
            isinstance(position_summary, dict)
            and isinstance(position_summary.get("samples_us"), list)
            and isinstance(raw_groups, list)
            and len(raw_groups) == len(position_summary["samples_us"]),
            f"{path}: aggregate position {position_key!r} counts differ",
        )
        recomputed: list[float] = []
        for reported_sample, raw_group in zip(
            position_summary["samples_us"], raw_groups, strict=True
        ):
            _require(
                isinstance(raw_group, list)
                and len(raw_group) == replays
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0.0
                    for value in raw_group
                ),
                f"{path}: aggregate position {position_key!r} has invalid inner samples",
            )
            inner_mean = statistics.fmean(float(value) for value in raw_group)
            _require(
                isinstance(reported_sample, (int, float))
                and not isinstance(reported_sample, bool)
                and math.isclose(
                    float(reported_sample),
                    inner_mean,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"{path}: aggregate position {position_key!r} sample is not its "
                "declared inner mean",
            )
            recomputed.append(inner_mean)
        _require_abba_summary_matches(
            position_summary,
            recomputed,
            path=path,
            context=f"aggregate position {position_key!r}",
            exact_sample_order=True,
        )
        recomputed_by_position[position_key] = recomputed
        position_counts.add(len(recomputed))
        reported_by_label[label].extend(recomputed)
        observed_inner_counts[label] += len(recomputed) * replays
    _require(
        len(position_counts) == 1 and next(iter(position_counts)) > 0,
        f"{path}: {'.'.join(timing_path)} position/order counts are unbalanced",
    )
    samples_per_position = next(iter(position_counts))
    total_cycles = 2 * samples_per_position
    batch_cycle_capacity = min(total_cycles, event_batch_cycles)
    pair_count = batch_cycle_capacity * len(order_0) * replays
    batch_count = math.ceil(total_cycles / event_batch_cycles)
    expected_pool_counts = {
        "batch_cycle_capacity": batch_cycle_capacity,
        "pair_count": pair_count,
        "event_count": 2 * pair_count,
        "unique_event_handle_count": 2 * pair_count,
        "prewarm_elapsed_query_count": pair_count,
        "batch_count": batch_count,
        "reuse_count": max(0, batch_count - 1),
    }
    _require(
        all(
            isinstance(event_pool.get(field), int)
            and not isinstance(event_pool.get(field), bool)
            and event_pool[field] == expected
            for field, expected in expected_pool_counts.items()
        ),
        f"{path}: {'.'.join(timing_path)} event-pool counts are inconsistent",
    )
    ordered_by_label = {arm_a: [], arm_b: []}
    for cycle in range(2 * samples_per_position):
        order_index = cycle & 1
        order = order_0 if order_index == 0 else order_1
        sample_index = cycle // 2
        for position, label in enumerate(order):
            key = f"{order_index}:{position}:{label}"
            ordered_by_label[label].append(recomputed_by_position[key][sample_index])
    for label, label_summary in summaries.items():
        _require(
            isinstance(label_summary, dict)
            and isinstance(label_summary.get("samples_us"), list),
            f"{path}: aggregate arm {label!r} summary is malformed",
        )
        _require_abba_summary_matches(
            label_summary,
            ordered_by_label[str(label)],
            path=path,
            context=f"aggregate arm {label!r}",
            exact_sample_order=True,
        )
        _require(
            inner_counts[label] == observed_inner_counts[str(label)],
            f"{path}: aggregate arm {label!r} declared counts differ",
        )
    return replays


def _timing_rows(
    path: Path,
    artifact: dict[str, Any],
    *,
    gpu: int,
    gpu_uuid: str,
    spec_hashes: set[str],
    minimum_samples: int,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for timing_path, value in _walk(artifact):
        if not isinstance(value, dict):
            continue
        summaries = value.get("summaries")
        if not isinstance(summaries, dict) or len(summaries) != 2:
            continue
        if not all(
            isinstance(summary, dict) and isinstance(summary.get("samples_us"), list)
            for summary in summaries.values()
        ):
            continue
        timing_condition = _value_at_path(artifact, timing_path[:-1])
        replays_per_reported_sample = _validate_aggregated_timing(
            path,
            timing_path,
            value,
            summaries,
            condition=timing_condition,
            gpu=gpu,
            gpu_uuid=gpu_uuid,
        )
        _require(
            isinstance(timing_condition, dict)
            and isinstance(timing_condition.get("preconditioning"), dict),
            f"{path}: timing path {'.'.join(timing_path)} lacks timing-mode policy",
        )
        required_throttle_reasons = _timing_required_active_throttle_reasons(
            path,
            timing_condition["preconditioning"].get("required_active_throttle_reasons"),
            description=(
                f"{'.'.join(timing_path)} requested active throttle-reasons mask"
            ),
        )
        timing_spec_hashes = _timing_spec_hashes(
            artifact,
            timing_path,
            spec_hashes,
        )
        _require(
            timing_spec_hashes,
            f"{path}: timing path {'.'.join(timing_path)} has no exact compile-spec "
            "binding",
        )
        arm_versions = _summary_arm_versions(artifact, summaries)
        by_version: dict[str, dict[str, Any]] = {}
        for label, summary in summaries.items():
            samples_raw = summary["samples_us"]
            samples: list[float] = []
            for sample in samples_raw:
                _require(
                    isinstance(sample, (int, float))
                    and not isinstance(sample, bool)
                    and math.isfinite(float(sample))
                    and float(sample) > 0.0,
                    f"{path}: invalid raw timing sample at {'.'.join(timing_path)}",
                )
                samples.append(float(sample))
            _require(
                len(samples) >= minimum_samples,
                f"{path}: {'.'.join(timing_path)} has {len(samples)} samples for "
                f"{label}; minimum is {minimum_samples}",
            )
            mean = statistics.fmean(samples)
            median = statistics.median(samples)
            p95 = _percentile_linear(samples, 0.95)
            if "count" in summary:
                _require(
                    summary["count"] == len(samples), f"{path}: summary count mismatch"
                )
            for field, expected in (("mean_us", mean), ("median_us", median)):
                if field in summary:
                    observed = float(summary[field])
                    _require(
                        math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9),
                        f"{path}: reported {field} disagrees with raw samples",
                    )
            reported_p95 = _adapter_reported_p95(
                samples, str(artifact.get("schema", ""))
            )
            for field in ("p95_us", "p95"):
                if field not in summary:
                    continue
                observed = summary[field]
                _require(
                    isinstance(observed, (int, float))
                    and not isinstance(observed, bool)
                    and math.isfinite(float(observed))
                    and math.isclose(
                        float(observed),
                        reported_p95,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ),
                    f"{path}: reported {field} disagrees with raw samples",
                )
            by_version[arm_versions[label]] = {
                "label": label,
                "sample_count": len(samples),
                "samples_sha256": _canonical_sha256(samples),
                "mean_us": mean,
                "median_us": median,
                "p95_us": p95,
                "inner_sample_count": len(samples) * replays_per_reported_sample,
            }
        _require(
            by_version["4.5.2"]["sample_count"] == by_version["4.6.0"]["sample_count"],
            f"{path}: unequal arm sample counts at {'.'.join(timing_path)}",
        )
        regressions = {
            metric: 100.0
            * (
                by_version["4.6.0"][f"{metric}_us"]
                / by_version["4.5.2"][f"{metric}_us"]
                - 1.0
            )
            for metric in ("mean", "median", "p95")
        }
        for metric in ("mean", "median", "p95"):
            _require(
                regressions[metric] <= thresholds[metric],
                f"{path}: {'.'.join(timing_path)} {metric} regression is "
                f"{regressions[metric]:.6f}%, limit={thresholds[metric]:.6f}%",
            )
        condition, flush_bytes = _timing_condition(artifact, timing_path)
        result.append(
            {
                "timing_path": ".".join(timing_path),
                "condition": condition,
                "l2_flush_bytes": flush_bytes,
                "spec_hashes": sorted(timing_spec_hashes),
                "replays_per_reported_sample": replays_per_reported_sample,
                "required_active_throttle_reasons": required_throttle_reasons,
                "arms": by_version,
                "regression_pct": regressions,
            }
        )
    _require(result, f"{path}: no raw ABBA timing summaries")
    return result


def _scan_abba_paths(roots: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    seen: set[Path] = set()
    selected: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        resolved = root.resolve()
        _require(resolved.exists(), f"ABBA artifact root does not exist: {resolved}")
        candidates = (
            [resolved] if resolved.is_file() else sorted(resolved.rglob("*.json"))
        )
        for path in candidates:
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            artifact = _load_json(path)
            schema = artifact.get("schema")
            if schema in _SUPPORTED_ABBA_SCHEMAS:
                selected.append((path, artifact))
            elif isinstance(schema, str) and "abba" in schema.lower():
                _fail(f"{path}: unsupported ABBA schema {schema!r}")
    _require(selected, "no supported exact-cache ABBA artifacts were found")
    return selected


def _validate_artifact_timing_mode_policy(
    path: Path,
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, object]:
    requested_masks = {int(row["required_active_throttle_reasons"]) for row in rows}
    _require(
        len(requested_masks) == 1,
        f"{path}: timing conditions do not bind one consistent active "
        "throttle-reasons policy",
    )
    requested_mask = next(iter(requested_masks))
    raw_policy = artifact.get("timing_mode_policy")
    requires_top_level_policy = (
        artifact.get("schema") == "b12x.w4a16.serving.cache_abba.v2"
    )
    if raw_policy is None:
        _require(
            not requires_top_level_policy,
            f"{path}: W4A16 v2 evidence lacks its top-level timing-mode policy",
        )
        return {
            "schema": _GPU_TIMING_MODE_POLICY_SCHEMA,
            "required_pstate": _REQUIRED_TIMING_PSTATE,
            "required_active_throttle_reasons": requested_mask,
            "active_throttle_reasons_match": "exact",
            "permitted_required_active_throttle_reasons": sorted(
                _PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS
            ),
            "required_memory_clock_equality": True,
            "max_sm_clock_delta_mhz": _MAX_TIMING_SM_CLOCK_DELTA_MHZ,
        }

    required_fields = {
        "schema",
        "required_pstate",
        "required_active_throttle_reasons",
        "active_throttle_reasons_match",
        "permitted_required_active_throttle_reasons",
        "required_memory_clock_equality",
        "max_sm_clock_delta_mhz",
    }
    _require(
        isinstance(raw_policy, dict) and set(raw_policy) == required_fields,
        f"{path}: top-level timing-mode policy fields are incomplete or unexpected",
    )
    policy_mask = _timing_required_active_throttle_reasons(
        path,
        raw_policy["required_active_throttle_reasons"],
        description="top-level required active throttle-reasons mask",
    )
    _require(
        raw_policy["schema"] == _GPU_TIMING_MODE_POLICY_SCHEMA
        and raw_policy["required_pstate"] == _REQUIRED_TIMING_PSTATE
        and raw_policy["active_throttle_reasons_match"] == "exact"
        and raw_policy["permitted_required_active_throttle_reasons"]
        == sorted(_PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS)
        and raw_policy["required_memory_clock_equality"] is True
        and raw_policy["max_sm_clock_delta_mhz"] == _MAX_TIMING_SM_CLOCK_DELTA_MHZ
        and policy_mask == requested_mask,
        f"{path}: top-level timing-mode policy does not match condition evidence",
    )
    return dict(raw_policy)


def _validate_evidence_status(path: Path, artifact: dict[str, Any]) -> None:
    _require(
        artifact.get("evidence_status") == "final-source",
        f"{path}: explicit final-source evidence status is required; "
        "diagnostic, missing, or non-final evidence cannot enter the release index",
    )


def _validate_abba_artifacts(
    roots: list[Path],
    *,
    gpu: int,
    gpu_uuid: str,
    source_fingerprint: str,
    minimum_samples: int,
    thresholds: dict[str, float],
    required_conditions: set[str],
    allowed_spec_hashes: set[str],
    production_spec_hashes: set[str],
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    timing_runs: list[dict[str, Any]] = []
    seen_artifact_hashes: set[str] = set()
    for path, artifact in _scan_abba_paths(roots):
        schema = str(artifact["schema"])
        _validate_evidence_status(path, artifact)
        _validate_result_hash(path, artifact)
        _validate_serving_gates(path, artifact)
        physical, observed_uuid, capability = _physical_gpu(artifact)
        _require(
            physical == gpu, f"{path}: expected physical GPU {gpu}, got {physical}"
        )
        _require(
            _normalize_uuid(observed_uuid) == _normalize_uuid(gpu_uuid),
            f"{path}: GPU UUID does not match corpus GPU {gpu}",
        )
        _require(capability == [12, 0], f"{path}: expected SM120 capability")
        _validate_gpu_mode_pairs(
            path,
            artifact,
            gpu=gpu,
            gpu_uuid=gpu_uuid,
        )
        provenance, object_hashes = _provenance_records(
            path,
            artifact,
            source_fingerprint=source_fingerprint,
        )
        _validate_artifact_integrity(path, artifact, provenance)
        spec_hashes = set(provenance)
        _require(
            spec_hashes <= allowed_spec_hashes,
            f"{path}: ABBA provenance contains specs absent from the corpus: "
            f"{sorted(spec_hashes - allowed_spec_hashes)!r}",
        )
        rows = _timing_rows(
            path,
            artifact,
            gpu=gpu,
            gpu_uuid=gpu_uuid,
            spec_hashes=spec_hashes,
            minimum_samples=minimum_samples,
            thresholds=thresholds,
        )
        timing_mode_policy = _validate_artifact_timing_mode_policy(path, artifact, rows)
        timed_spec_hashes = {
            spec_hash for row in rows for spec_hash in row["spec_hashes"]
        }
        _require(
            timed_spec_hashes == spec_hashes,
            f"{path}: exact-object provenance and timing coverage differ; "
            f"untimed={sorted(spec_hashes - timed_spec_hashes)!r}, "
            f"unproven={sorted(timed_spec_hashes - spec_hashes)!r}",
        )
        record = _artifact_record(path, schema=schema)
        artifact_sha = str(record["sha256"])
        _require(
            artifact_sha not in seen_artifact_hashes,
            f"duplicate ABBA artifact content supplied: {path}",
        )
        seen_artifact_hashes.add(artifact_sha)
        artifact_index = len(artifacts)
        artifacts.append(
            {
                **record,
                "gpu": gpu,
                "gpu_uuid": gpu_uuid,
                "source_fingerprint": source_fingerprint,
                "spec_hashes": sorted(spec_hashes),
                "object_sha256s": sorted(object_hashes),
                "provenance": provenance,
                "timing_mode_policy": timing_mode_policy,
            }
        )
        for row in rows:
            logical_key = _canonical_sha256(
                {
                    "schema": schema,
                    "timing_path": row["timing_path"],
                    "condition": row["condition"],
                    "spec_hashes": row["spec_hashes"],
                    "required_active_throttle_reasons": row[
                        "required_active_throttle_reasons"
                    ],
                }
            )
            timing_runs.append(
                {
                    **row,
                    "schema": schema,
                    "logical_key": logical_key,
                    "artifact_index": artifact_index,
                    "artifact_path": str(path),
                    "artifact_sha256": artifact_sha,
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in timing_runs:
        grouped.setdefault(run["logical_key"], []).append(run)
    performance_groups: list[dict[str, Any]] = []
    for logical_key, runs in sorted(grouped.items()):
        _require(
            len(runs) >= 2,
            f"GPU{gpu} timing group {logical_key} has {len(runs)} run(s); "
            "two independent ABBA artifacts are required for run drift",
        )
        drifts: dict[str, float] = {}
        for cutlass in ("4.5.2", "4.6.0"):
            means = [float(run["arms"][cutlass]["mean_us"]) for run in runs]
            drift = 100.0 * (max(means) / min(means) - 1.0)
            _require(
                drift <= thresholds["run_drift"],
                f"GPU{gpu} timing group {logical_key} {cutlass} run drift is "
                f"{drift:.6f}%, limit={thresholds['run_drift']:.6f}%",
            )
            drifts[cutlass] = drift
        performance_groups.append(
            {
                "logical_key": logical_key,
                "schema": runs[0]["schema"],
                "timing_path": runs[0]["timing_path"],
                "condition": runs[0]["condition"],
                "required_active_throttle_reasons": runs[0][
                    "required_active_throttle_reasons"
                ],
                "spec_hashes": runs[0]["spec_hashes"],
                "run_count": len(runs),
                "run_mean_drift_pct": drifts,
                "runs": [
                    {
                        key: run[key]
                        for key in (
                            "artifact_path",
                            "artifact_sha256",
                            "l2_flush_bytes",
                            "arms",
                            "regression_pct",
                        )
                    }
                    for run in runs
                ],
            }
        )

    coverage: dict[str, dict[str, list[int]]] = {}
    for group_index, group in enumerate(performance_groups):
        condition = str(group["condition"])
        for spec_hash in group["spec_hashes"]:
            coverage.setdefault(spec_hash, {}).setdefault(condition, []).append(
                group_index
            )
    for spec_hash, conditions in coverage.items():
        missing = sorted(required_conditions - set(conditions))
        _require(
            not missing,
            f"GPU{gpu} performance spec {spec_hash} lacks conditions {missing!r}",
        )
    missing_production = sorted(production_spec_hashes - set(coverage))
    _require(
        not missing_production,
        f"GPU{gpu} lacks ABBA evidence for production-bound specs "
        f"{missing_production!r}",
    )
    for spec_hash in sorted(production_spec_hashes):
        missing = sorted(required_conditions - set(coverage[spec_hash]))
        _require(
            not missing,
            f"GPU{gpu} production spec {spec_hash} lacks conditions {missing!r}",
        )
    return {
        "gpu": gpu,
        "roots": [str(path.resolve()) for path in roots],
        "artifacts": artifacts,
        "performance_groups": performance_groups,
        "coverage": coverage,
    }


def _exception_bindings(
    gpu4_analysis: dict[str, Any],
    gpu5_analysis: dict[str, Any],
    gpu4_perf: dict[str, Any],
    gpu5_perf: dict[str, Any],
    corpora: dict[str, dict[str, Any]],
    *,
    required_conditions: set[str],
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    json_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    for key in sorted(gpu4_analysis["accounting"]):
        row4 = gpu4_analysis["accounting"][key]
        row5 = gpu5_analysis["accounting"][key]
        if not row4["exception_fields"]:
            continue
        spec_hash = hashlib.sha256(row4["compile_spec_json"].encode()).hexdigest()
        kernel = row4["kernel"]
        binding_pair = (spec_hash, kernel)
        for corpus_name in (
            "gpu4_cutlass45",
            "gpu4_cutlass46",
            "gpu5_cutlass45",
            "gpu5_cutlass46",
        ):
            _require(
                binding_pair in corpora[corpus_name]["binding_pairs"],
                f"exception {key!r} is not bound by {corpus_name} case evidence",
            )
        gpu_evidence: dict[int, dict[str, Any]] = {}
        for gpu, perf in ((4, gpu4_perf), (5, gpu5_perf)):
            conditions = perf["coverage"].get(spec_hash, {})
            missing = sorted(required_conditions - set(conditions))
            _require(
                not missing,
                f"exception {key!r} lacks GPU{gpu} ABBA conditions {missing!r}",
            )
            group_indices = sorted(
                {
                    index
                    for condition in required_conditions
                    for index in conditions[condition]
                }
            )
            artifact_paths = sorted(
                {
                    run["artifact_path"]
                    for index in group_indices
                    for run in perf["performance_groups"][index]["runs"]
                }
            )
            gpu_evidence[gpu] = {
                "conditions": sorted(required_conditions),
                "performance_group_indices": group_indices,
                "artifact_paths": artifact_paths,
            }
        record = {
            "comparison_semantic_key": row4["comparison_semantic_key"],
            "symbol_sha256": row4["symbol_sha256"],
            "family": row4["family"],
            "kernel": kernel,
            "compile_spec_hash": spec_hash,
            "exception_fields": row4["exception_fields"].split(";"),
            "cause": row4["cause"],
            "disposition": row4["disposition"],
            "evidence": row4["evidence"],
            "performance_status": row4["performance_status"],
            "gpu4": gpu_evidence[4],
            "gpu5": gpu_evidence[5],
            "status": "pass",
        }
        json_rows.append(record)
        csv_rows.append(
            {
                "release_exception_index_schema": _EXCEPTION_INDEX_SCHEMA,
                "comparison_semantic_key": row4["comparison_semantic_key"],
                "symbol_sha256": row4["symbol_sha256"],
                "family": row4["family"],
                "kernel": kernel,
                "compile_spec_hash": spec_hash,
                "exception_fields": row4["exception_fields"],
                "cause": row4["cause"],
                "disposition": row4["disposition"],
                "performance_status": row4["performance_status"],
                "gpu4_artifacts_json": json.dumps(
                    gpu_evidence[4]["artifact_paths"], separators=(",", ":")
                ),
                "gpu5_artifacts_json": json.dumps(
                    gpu_evidence[5]["artifact_paths"], separators=(",", ":")
                ),
                "gpu4_conditions_json": json.dumps(
                    gpu_evidence[4]["conditions"], separators=(",", ":")
                ),
                "gpu5_conditions_json": json.dumps(
                    gpu_evidence[5]["conditions"], separators=(",", ":")
                ),
                "max_mean_regression_pct": str(thresholds["mean"]),
                "max_median_regression_pct": str(thresholds["median"]),
                "max_p95_regression_pct": str(thresholds["p95"]),
                "max_run_mean_drift_pct": str(thresholds["run_drift"]),
                "status": "pass",
            }
        )
        _require(
            row4["exception_fields"] == row5["exception_fields"],
            f"GPU4/GPU5 exception classification differs for {key!r}",
        )
    return json_rows, csv_rows


def _compare_performance_coverage(gpu4: dict[str, Any], gpu5: dict[str, Any]) -> None:
    def normalized(performance: dict[str, Any]) -> dict[str, tuple[str, ...]]:
        return {
            spec_hash: tuple(sorted(conditions))
            for spec_hash, conditions in performance["coverage"].items()
        }

    _require(
        normalized(gpu4) == normalized(gpu5),
        "GPU4/GPU5 ABBA compile-spec/condition coverage differs",
    )


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=_EXCEPTION_CSV_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _public_corpus_record(corpus: dict[str, Any]) -> dict[str, Any]:
    return {
        key: corpus[key]
        for key in (
            "root",
            "gpu",
            "gpu_name",
            "gpu_uuid",
            "cutlass",
            "packages",
            "source_fingerprint",
            "case_count",
            "binding_count",
            "artifacts",
        )
    } | {
        "resource_row_count": len(corpus["resource_rows"]),
        "all_compile_spec_count": len(corpus["all_spec_hashes"]),
        "production_compile_spec_count": len(corpus["production_spec_hashes"]),
        "case_status_counts": {
            status: sum(
                observed == status for observed in corpus["case_statuses"].values()
            )
            for status in ("production", "diagnostic")
        },
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for gpu in (4, 5):
        for arm, version in (("cutlass45", "4.5.2"), ("cutlass46", "4.6.0")):
            parser.add_argument(
                f"--gpu{gpu}-{arm}-corpus",
                type=Path,
                required=True,
                help=f"complete CUTLASS {version} corpus root from physical GPU {gpu}",
            )
        parser.add_argument(f"--gpu{gpu}-resource-delta", type=Path, required=True)
        parser.add_argument(f"--gpu{gpu}-sass-delta", type=Path, required=True)
        parser.add_argument(f"--gpu{gpu}-accounting", type=Path, required=True)
        parser.add_argument(
            f"--gpu{gpu}-abba-root",
            type=Path,
            action="append",
            required=True,
            help=(
                "exact-cache ABBA JSON artifact or directory; repeat for disjoint "
                "release artifact sets"
            ),
        )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--minimum-samples-per-arm", type=int, default=1000)
    parser.add_argument("--max-mean-regression-pct", type=float, default=0.5)
    parser.add_argument("--max-median-regression-pct", type=float, default=0.5)
    parser.add_argument("--max-p95-regression-pct", type=float, default=1.0)
    parser.add_argument("--max-run-mean-drift-pct", type=float, default=1.0)
    args = parser.parse_args()
    if args.minimum_samples_per_arm < 1000:
        parser.error(
            "--minimum-samples-per-arm cannot weaken the release floor of 1000"
        )
    release_ceilings = {
        "max_mean_regression_pct": 0.5,
        "max_median_regression_pct": 0.5,
        "max_p95_regression_pct": 1.0,
        "max_run_mean_drift_pct": 1.0,
    }
    for field, release_ceiling in release_ceilings.items():
        value = getattr(args, field)
        if not math.isfinite(value) or not 0.0 <= value <= release_ceiling:
            parser.error(
                f"--{field.replace('_', '-')} must be finite, nonnegative, and cannot "
                f"exceed the release ceiling of {release_ceiling}"
            )
    args.required_conditions = {"warm_l2", "cold_l2"}
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    if output_json == output_csv:
        parser.error("--output-json and --output-csv must differ")
    input_paths = {
        value.resolve()
        for key, value in vars(args).items()
        if isinstance(value, Path) and key not in {"output_json", "output_csv"}
    }
    abba_roots = [
        path.resolve()
        for gpu in (4, 5)
        for path in getattr(args, f"gpu{gpu}_abba_root")
    ]
    input_paths.update(path for path in abba_roots if path.is_file())
    if output_json in input_paths or output_csv in input_paths:
        parser.error("outputs must differ from every file input")
    for output in (output_json, output_csv):
        if any(root.is_dir() and root in output.parents for root in abba_roots):
            parser.error("outputs must be outside every ABBA artifact directory")
    return args


def main() -> None:
    args = _args()
    thresholds = {
        "mean": args.max_mean_regression_pct,
        "median": args.max_median_regression_pct,
        "p95": args.max_p95_regression_pct,
        "run_drift": args.max_run_mean_drift_pct,
    }
    try:
        corpora = {
            "gpu4_cutlass45": _validate_corpus(
                args.gpu4_cutlass45_corpus, gpu=4, cutlass="4.5.2"
            ),
            "gpu4_cutlass46": _validate_corpus(
                args.gpu4_cutlass46_corpus, gpu=4, cutlass="4.6.0"
            ),
            "gpu5_cutlass45": _validate_corpus(
                args.gpu5_cutlass45_corpus, gpu=5, cutlass="4.5.2"
            ),
            "gpu5_cutlass46": _validate_corpus(
                args.gpu5_cutlass46_corpus, gpu=5, cutlass="4.6.0"
            ),
        }
        source_manifests = [
            _canonical_json_bytes(corpus["source_manifest"])
            for corpus in corpora.values()
        ]
        _require(
            all(value == source_manifests[0] for value in source_manifests[1:]),
            "four-corpus frozen source manifests are not identical",
        )
        source_fingerprint = corpora["gpu4_cutlass45"]["source_fingerprint"]
        _require(
            all(
                corpus["source_fingerprint"] == source_fingerprint
                for corpus in corpora.values()
            ),
            "four-corpus b12x package fingerprints differ",
        )
        expected_resource_keys = corpora["gpu4_cutlass45"]["resource_keys"]
        _require(
            all(
                corpus["resource_keys"] == expected_resource_keys
                for corpus in corpora.values()
            ),
            "four-corpus resource specialization/kernel key sets differ",
        )
        allowed_spec_hashes = corpora["gpu4_cutlass45"]["all_spec_hashes"]
        production_spec_hashes = corpora["gpu4_cutlass45"]["production_spec_hashes"]
        _require(
            all(
                corpus["all_spec_hashes"] == allowed_spec_hashes
                for corpus in corpora.values()
            ),
            "four-corpus compile-spec sets differ",
        )
        _require(
            all(
                corpus["production_spec_hashes"] == production_spec_hashes
                for corpus in corpora.values()
            ),
            "four-corpus production compile-spec coverage differs",
        )

        analyses = {
            "gpu4": _validate_analysis_set(
                gpu=4,
                baseline=corpora["gpu4_cutlass45"],
                current=corpora["gpu4_cutlass46"],
                resource_delta_path=args.gpu4_resource_delta,
                sass_delta_path=args.gpu4_sass_delta,
                accounting_path=args.gpu4_accounting,
            ),
            "gpu5": _validate_analysis_set(
                gpu=5,
                baseline=corpora["gpu5_cutlass45"],
                current=corpora["gpu5_cutlass46"],
                resource_delta_path=args.gpu5_resource_delta,
                sass_delta_path=args.gpu5_sass_delta,
                accounting_path=args.gpu5_accounting,
            ),
        }
        _compare_analysis_sets(analyses["gpu4"], analyses["gpu5"])

        performance = {
            "gpu4": _validate_abba_artifacts(
                args.gpu4_abba_root,
                gpu=4,
                gpu_uuid=corpora["gpu4_cutlass46"]["gpu_uuid"],
                source_fingerprint=source_fingerprint,
                minimum_samples=args.minimum_samples_per_arm,
                thresholds=thresholds,
                required_conditions=args.required_conditions,
                allowed_spec_hashes=allowed_spec_hashes,
                production_spec_hashes=production_spec_hashes,
            ),
            "gpu5": _validate_abba_artifacts(
                args.gpu5_abba_root,
                gpu=5,
                gpu_uuid=corpora["gpu5_cutlass46"]["gpu_uuid"],
                source_fingerprint=source_fingerprint,
                minimum_samples=args.minimum_samples_per_arm,
                thresholds=thresholds,
                required_conditions=args.required_conditions,
                allowed_spec_hashes=allowed_spec_hashes,
                production_spec_hashes=production_spec_hashes,
            ),
        }
        _compare_performance_coverage(performance["gpu4"], performance["gpu5"])
        exception_rows, exception_csv_rows = _exception_bindings(
            analyses["gpu4"],
            analyses["gpu5"],
            performance["gpu4"],
            performance["gpu5"],
            corpora,
            required_conditions=args.required_conditions,
            thresholds=thresholds,
        )

        csv_payload = _csv_bytes(exception_csv_rows)
        index: dict[str, Any] = {
            "schema": _INDEX_SCHEMA,
            "status": "pass",
            "source_fingerprint": source_fingerprint,
            "source_manifest_sha256": corpora["gpu4_cutlass45"]["source_manifest"][
                "manifest_sha256"
            ],
            "thresholds": {
                "minimum_samples_per_arm": args.minimum_samples_per_arm,
                "required_conditions": sorted(args.required_conditions),
                "max_mean_regression_pct": thresholds["mean"],
                "max_median_regression_pct": thresholds["median"],
                "max_p95_regression_pct": thresholds["p95"],
                "max_run_mean_drift_pct": thresholds["run_drift"],
                "p95_definition": "linear interpolation at 0.95 * (n - 1)",
                "run_drift_definition": (
                    "100 * (max independent-run arm mean / min independent-run "
                    "arm mean - 1)"
                ),
            },
            "corpora": {
                name: _public_corpus_record(corpus)
                for name, corpus in sorted(corpora.items())
            },
            "analysis": {
                name: {
                    "gpu": analysis["gpu"],
                    "row_count": len(analysis["resource_rows"]),
                    "exception_count": sum(
                        bool(row["exception_fields"])
                        for row in analysis["accounting_rows"]
                    ),
                    "sass_structural_sha256": analysis["sass_structural_sha256"],
                    "artifacts": analysis["artifacts"],
                }
                for name, analysis in sorted(analyses.items())
            },
            "performance": performance,
            "exceptions": exception_rows,
            "exception_csv": {
                "path": str(args.output_csv.resolve()),
                "schema": _EXCEPTION_INDEX_SCHEMA,
                "sha256": _sha256_bytes(csv_payload),
                "size_bytes": len(csv_payload),
                "row_count": len(exception_csv_rows),
            },
        }
        index["index_sha256"] = _canonical_sha256(index)
        json_payload = (
            json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_csv.write_bytes(csv_payload)
        args.output_json.write_bytes(json_payload)
    except (OSError, ValueError, ReleaseValidationError) as exc:
        print(f"release artifact validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(
        json.dumps(
            {
                "status": "pass",
                "output_json": str(args.output_json.resolve()),
                "output_csv": str(args.output_csv.resolve()),
                "index_sha256": index["index_sha256"],
                "resource_rows": len(expected_resource_keys),
                "exceptions": len(exception_rows),
                "gpu4_abba_artifacts": len(performance["gpu4"]["artifacts"]),
                "gpu5_abba_artifacts": len(performance["gpu5"]["artifacts"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
