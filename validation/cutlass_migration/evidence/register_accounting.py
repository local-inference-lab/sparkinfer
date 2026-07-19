#!/usr/bin/env python3
# ruff: noqa: SIM905
"""Build a reviewable all-kernel accounting table from a strict delta report.

The primary input is the CSV emitted by the
``python -m validation.cutlass_migration evidence compare-resources`` command;
the exact SASS-set delta from
``python -m validation.cutlass_migration evidence compare-sass`` is also
required and joined one-to-one.  This script deliberately retains every
comparison specialization and CUDA entry point, including rows whose allocated
register count is unchanged or lower.  A separate annotation CSV can attach
causal evidence to exceptional rows without making the resource measurements
themselves hand-maintained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from validation.cutlass_migration.core.comparison_identity import (
    normalize_comparison_compile_options,
    normalize_comparison_compile_environment,
    normalize_comparison_compile_kwargs,
)
from validation.cutlass_migration.evidence.compare_sass_register_sets import (
    _DELTA_SCHEMA as _SASS_DELTA_SCHEMA,
    _FAMILIES as _SASS_FAMILIES,
    _OUTPUT_FIELDS as _SASS_DELTA_FIELDS,
    _RECONFIGURATION_FIELDS as _SASS_RECONFIGURATION_FIELDS,
    _RESOURCE_FIELDS as _SASS_RESOURCE_FIELDS,
)

_ANNOTATION_KEY = ("comparison_semantic_key", "symbol_sha256")
_ANNOTATION_FIELDS = (
    "cause",
    "disposition",
    "evidence",
    "performance_status",
)
_ANNOTATION_TEMPLATE_SCHEMA = "b12x.cute.kernel_register_annotation_template.v2"
_COMMON_PTXAS_DELTA_SCHEMA = "b12x.cute.common_ptxas_positive_delta.v2"
_DEFAULT_ANNOTATION = {
    "cause": "unresolved",
    "disposition": "pending-review",
    # Keep generated templates from satisfying --require-annotations-for-exceptions
    # until a reviewer attaches concrete evidence.
    "evidence": "",
    "performance_status": "pending-benchmark",
}
_ANNOTATION_TEMPLATE_FIELDS = (
    "annotation_template_schema",
    "comparison_semantic_key",
    "baseline_semantic_key",
    "current_semantic_key",
    "symbol_sha256",
    "family",
    "kernel_id",
    "kernel",
    "compile_spec_hash",
    "compile_spec_json",
    "architecture",
    "baseline_package_fingerprint",
    "current_package_fingerprint",
    "baseline_cache_key",
    "current_cache_key",
    "baseline_cutlass_dsl",
    "current_cutlass_dsl",
    "baseline_ptxas",
    "current_ptxas",
    "exception_fields",
    "native_delta_summary_json",
    "common_ptxas_status",
    "common_ptxas_sha256",
    "common_ptxas_version_output_sha256",
    "common_ptxas_flags_json",
    "common_ptxas_selection_reasons_json",
    "common_ptxas_delta_summary_json",
    "annotation_origin",
    *_ANNOTATION_FIELDS,
)
_COMMON_PTXAS_REQUIRED_FIELDS = (
    "common_ptxas_delta_schema",
    "comparison_semantic_key",
    "kernel",
    "symbol_sha256",
    "selection_reasons_json",
    "baseline_cache_key",
    "current_cache_key",
    "common_ptxas_sha256",
    "common_ptxas_version_output_sha256",
    "common_ptxas_flags_json",
    "common_any_register_usage_increase",
    "common_any_local_memory_increase",
    "common_any_static_shared_memory_increase",
)
_DELTA_REPORT_SCHEMA = "b12x.cute.kernel_resource_delta.v4"
_RESOURCE_REPORT_SCHEMA = "b12x.cute.kernel_resources.v4"
_ACCOUNTING_SCHEMA = "b12x.cute.kernel_register_accounting.v5"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DELTA_REPORT_FIELDS = (
    *(
        "delta_report_schema,baseline_resource_report_schema,current_resource_report_schema,"
        "family,identity_kind,comparison_semantic_key,baseline_semantic_key,"
        "current_semantic_key,symbol_sha256,pairing,pair_ordinal,"
        "baseline_identity_count,current_identity_count,baseline_manifest_status,"
        "current_manifest_status,baseline_cache_key,current_cache_key,"
        "baseline_object_sha256,current_object_sha256,baseline_target,current_target,"
        "baseline_kernel_id,current_kernel_id,baseline_compile_spec_version,"
        "current_compile_spec_version,baseline_compile_spec_hash,"
        "current_compile_spec_hash,baseline_compile_spec_json,current_compile_spec_json,"
        "baseline_compile_kwargs_json,current_compile_kwargs_json,"
        "baseline_package_fingerprint,current_package_fingerprint,baseline_toolchain_json,"
        "current_toolchain_json,baseline_compile_options_json,current_compile_options_json,"
        "baseline_compile_environment_json,current_compile_environment_json,"
        "baseline_architecture,current_architecture,architecture_change,"
        "baseline_ptxas_version,current_ptxas_version,baseline_ptxas_flags,"
        "current_ptxas_flags,baseline_cutlass_dsl_version,current_cutlass_dsl_version,"
        "baseline_cutlass_dsl_libs_base_version,current_cutlass_dsl_libs_base_version,"
        "baseline_cutlass_dsl_libs_core_version,current_cutlass_dsl_libs_core_version,"
        "baseline_cutlass_dsl_libs_cu12_version,current_cutlass_dsl_libs_cu12_version,"
        "baseline_cutlass_dsl_libs_cu13_version,current_cutlass_dsl_libs_cu13_version,"
        "baseline_threads_per_cta,current_threads_per_cta,threads_per_cta_delta,"
        "baseline_threads_x,current_threads_x,baseline_threads_y,current_threads_y,"
        "baseline_threads_z,current_threads_z,baseline_cubin_shared_section_bytes,"
        "current_cubin_shared_section_bytes,cubin_shared_section_delta,"
        "cubin_shared_section_change,cubin_shared_section_increase,"
        "baseline_launch_dynamic_smem_status,current_launch_dynamic_smem_status,"
        "baseline_launch_dynamic_smem_count,current_launch_dynamic_smem_count,"
        "baseline_launch_dynamic_smem_values_json,current_launch_dynamic_smem_values_json,"
        "baseline_launch_metadata_source,current_launch_metadata_source,"
        "baseline_launch_metadata_reason,current_launch_metadata_reason,"
        "launch_dynamic_smem_comparable,baseline_launch_dynamic_smem_bytes,"
        "current_launch_dynamic_smem_bytes,launch_dynamic_smem_delta,"
        "launch_dynamic_smem_change,launch_dynamic_smem_increase,"
        "baseline_max_register_count,current_max_register_count,baseline_parameter_bytes,"
        "current_parameter_bytes,launch_metadata_change,baseline_occupancy_status,"
        "current_occupancy_status,baseline_occupancy_device_ordinal,"
        "current_occupancy_device_ordinal,baseline_occupancy_gpu_name,"
        "current_occupancy_gpu_name,baseline_occupancy_gpu_uuid,current_occupancy_gpu_uuid,"
        "driver_occupancy_comparable,baseline_occupancy_active_ctas_per_sm,"
        "current_occupancy_active_ctas_per_sm,occupancy_active_ctas_per_sm_delta,"
        "driver_occupancy_decrease,baseline_occupancy_active_threads_per_sm,"
        "current_occupancy_active_threads_per_sm,occupancy_active_threads_per_sm_delta,"
        "baseline_driver_resource_validation_status,"
        "current_driver_resource_validation_status,baseline_driver_registers,"
        "current_driver_registers,baseline_driver_local_bytes,current_driver_local_bytes,"
        "baseline_driver_static_shared_bytes,current_driver_static_shared_bytes,"
        "driver_static_shared_bytes_delta,driver_static_shared_bytes_change,"
        "driver_static_shared_bytes_increase,"
        "baseline_driver_max_threads_per_block,current_driver_max_threads_per_block,"
        "baseline_registers,current_registers,register_delta,register_increase,"
        "baseline_sass_uniform_registers_used,current_sass_uniform_registers_used,"
        "sass_uniform_registers_used_delta,sass_uniform_registers_used_increase,"
        "baseline_sass_uniform_register_span,current_sass_uniform_register_span,"
        "sass_uniform_register_span_delta,sass_uniform_register_span_increase,"
        "baseline_sass_predicate_registers_used,current_sass_predicate_registers_used,"
        "sass_predicate_registers_used_delta,sass_predicate_registers_used_increase,"
        "baseline_sass_predicate_register_span,current_sass_predicate_register_span,"
        "sass_predicate_register_span_delta,sass_predicate_register_span_increase,"
        "baseline_sass_uniform_predicate_registers_used,"
        "current_sass_uniform_predicate_registers_used,"
        "sass_uniform_predicate_registers_used_delta,"
        "sass_uniform_predicate_registers_used_increase,"
        "baseline_sass_uniform_predicate_register_span,"
        "current_sass_uniform_predicate_register_span,"
        "sass_uniform_predicate_register_span_delta,"
        "sass_uniform_predicate_register_span_increase,any_register_usage_increase,"
        "new_register_ceiling,baseline_frame_bytes,current_frame_bytes,frame_delta,"
        "baseline_stack_bytes,current_stack_bytes,stack_delta,baseline_local_loads,"
        "current_local_loads,local_load_delta,baseline_local_stores,current_local_stores,"
        "local_store_delta,baseline_local_memory,current_local_memory,new_local_memory,"
        "local_memory_increase,baseline_sass_instructions,current_sass_instructions,"
        "sass_instruction_delta,baseline_code_bytes,current_code_bytes,code_bytes_delta,"
        "baseline_kernel,current_kernel"
    ).split(","),
)


def _true(row: dict[str, str], field: str) -> bool:
    value = row.get(field, "").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"row has invalid {field} value {value!r}")
    return value == "true"


def _integer(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"matched row has no {field}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"row has invalid {field} value {value!r}") from exc


def _normalized_compile_options(raw: str, *, side: str) -> tuple[str, ...]:
    """Normalize cross-arm operational options without weakening raw identity."""

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{side} compile_options_json is invalid") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{side} compile_options_json is not a string list")
    return tuple(normalize_comparison_compile_options(value))


def _normalized_compile_environment(
    raw: str, *, side: str
) -> tuple[tuple[str, str], ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{side} compile_environment_json is invalid") from exc
    if not isinstance(value, list) or any(
        not isinstance(entry, list)
        or len(entry) != 2
        or not isinstance(entry[0], str)
        or not isinstance(entry[1], str)
        for entry in value
    ):
        raise ValueError(f"{side} compile_environment_json is malformed")
    semantic = normalize_comparison_compile_environment(value)
    return tuple((entry[0], entry[1]) for entry in semantic)


def _normalized_compile_kwargs(raw: str, *, side: str) -> object:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{side} compile_kwargs_json is invalid") from exc
    return normalize_comparison_compile_kwargs(value)


def _bool_text(value: bool | int) -> str:
    return "true" if bool(value) else "false"


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _read_annotations(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None:
        return {}
    annotations: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        required = {*_ANNOTATION_KEY, *_ANNOTATION_FIELDS}
        if not required.issubset(fields):
            raise ValueError(
                f"{path}: annotation columns must include {sorted(required)!r}"
            )
        for row_number, row in enumerate(reader, start=2):
            key = tuple(row[field].strip() for field in _ANNOTATION_KEY)
            if not all(key):
                raise ValueError(f"{path}:{row_number}: annotation key is empty")
            if key in annotations:
                raise ValueError(f"{path}:{row_number}: duplicate annotation {key!r}")
            annotations[key] = {field: row[field].strip() for field in required}
    return annotations


def _read_common_ptxas_deltas(
    path: Path | None,
) -> dict[tuple[str, str], dict[str, str]]:
    if path is None:
        return {}
    deltas: dict[tuple[str, str], dict[str, str]] = {}
    provenance: tuple[str, str] | None = None
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
        if not rows and fieldnames == ("common_ptxas_delta_schema",):
            return {}
        missing = sorted(set(_COMMON_PTXAS_REQUIRED_FIELDS) - set(fieldnames))
        if missing:
            raise ValueError(f"{path}: common-PTXAS columns are missing {missing!r}")
        for row_number, raw in enumerate(rows, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            if row.get("common_ptxas_delta_schema") != _COMMON_PTXAS_DELTA_SCHEMA:
                raise ValueError(
                    f"{path}:{row_number}: invalid common-PTXAS delta schema"
                )
            key = _annotation_key(row)
            if not all(_SHA256_RE.fullmatch(value) for value in key):
                raise ValueError(f"{path}:{row_number}: invalid common-PTXAS exact key")
            kernel = row.get("kernel", "")
            if not kernel or hashlib.sha256(kernel.encode()).hexdigest() != key[1]:
                raise ValueError(
                    f"{path}:{row_number}: common-PTXAS symbol hash mismatch"
                )
            for field in (
                "baseline_cache_key",
                "current_cache_key",
                "common_ptxas_sha256",
                "common_ptxas_version_output_sha256",
            ):
                if not _SHA256_RE.fullmatch(row.get(field, "")):
                    raise ValueError(
                        f"{path}:{row_number}: invalid common-PTXAS {field}"
                    )
            try:
                selection_reasons = json.loads(row["selection_reasons_json"])
                common_ptxas_flags = json.loads(row["common_ptxas_flags_json"])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{row_number}: invalid common-PTXAS JSON"
                ) from exc
            if (
                not isinstance(selection_reasons, list)
                or any(
                    not isinstance(reason, str) or not reason
                    for reason in selection_reasons
                )
                or selection_reasons != sorted(set(selection_reasons))
            ):
                raise ValueError(
                    f"{path}:{row_number}: invalid common-PTXAS selection reasons"
                )
            if not isinstance(common_ptxas_flags, list) or any(
                not isinstance(flag, str) or not flag for flag in common_ptxas_flags
            ):
                raise ValueError(f"{path}:{row_number}: invalid common-PTXAS flags")
            row["selection_reasons_json"] = _compact_json(selection_reasons)
            row["common_ptxas_flags_json"] = _compact_json(common_ptxas_flags)
            for field in (
                "common_any_register_usage_increase",
                "common_any_local_memory_increase",
                "common_any_static_shared_memory_increase",
            ):
                _true(row, field)
            for field, value in row.items():
                if field.startswith("common_") and field.endswith("_delta"):
                    try:
                        int(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"{path}:{row_number}: invalid {field} value {value!r}"
                        ) from exc
            row_provenance = (
                row["common_ptxas_sha256"],
                row["common_ptxas_version_output_sha256"],
            )
            if provenance is None:
                provenance = row_provenance
            elif row_provenance != provenance:
                raise ValueError(
                    f"{path}:{row_number}: common-PTXAS provenance changed within report"
                )
            if key in deltas:
                raise ValueError(
                    f"{path}:{row_number}: duplicate common-PTXAS delta {key!r}"
                )
            deltas[key] = row
    return deltas


def _read_sass_deltas(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    deltas: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != tuple(_SASS_DELTA_FIELDS):
            raise ValueError(f"{path}: expected exact {_SASS_DELTA_SCHEMA} header")
        for row_number, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            if row.get("sass_register_set_delta_schema") != _SASS_DELTA_SCHEMA:
                raise ValueError(f"{path}:{row_number}: invalid SASS delta schema")
            key = (
                row.get("comparison_semantic_key", ""),
                row.get("kernel", ""),
            )
            if not all(key):
                raise ValueError(f"{path}:{row_number}: empty SASS delta key")
            if key in deltas:
                raise ValueError(f"{path}:{row_number}: duplicate SASS delta {key!r}")
            deltas[key] = row
    if not deltas:
        raise ValueError(f"{path}: SASS delta report has no rows")
    return deltas


def _require_sass_delta_pair(
    row: dict[str, str],
    sass: dict[str, str],
    row_number: int,
) -> None:
    key = (row["comparison_semantic_key"], row.get("current_kernel", ""))
    if key != (
        sass.get("comparison_semantic_key", ""),
        sass.get("kernel", ""),
    ):
        raise ValueError(f"delta row {row_number} has mismatched exact SASS key")
    expected_symbol_sha256 = hashlib.sha256(key[1].encode()).hexdigest()
    if row["symbol_sha256"] != expected_symbol_sha256:
        raise ValueError(f"delta row {row_number} has an invalid symbol SHA-256")

    for exact_field, main_field in (
        ("baseline_semantic_key", "baseline_semantic_key"),
        ("current_semantic_key", "current_semantic_key"),
        ("baseline_cutlass_dsl_version", "baseline_cutlass_dsl_version"),
        ("current_cutlass_dsl_version", "current_cutlass_dsl_version"),
        ("baseline_package_fingerprint", "baseline_package_fingerprint"),
        ("current_package_fingerprint", "current_package_fingerprint"),
        ("baseline_object_sha256", "baseline_object_sha256"),
        ("current_object_sha256", "current_object_sha256"),
    ):
        if sass.get(exact_field, "") != row.get(main_field, ""):
            raise ValueError(
                f"delta row {row_number} exact SASS {exact_field} disagrees "
                f"with {main_field}"
            )
    for exact_field, main_field in (
        ("target", "current_target"),
        ("kernel_id", "current_kernel_id"),
        ("compile_spec_version", "current_compile_spec_version"),
        ("compile_spec_hash", "current_compile_spec_hash"),
        ("compile_spec_json", "current_compile_spec_json"),
        ("architecture", "current_architecture"),
    ):
        if sass.get(exact_field, "") != row.get(main_field, ""):
            raise ValueError(
                f"delta row {row_number} exact SASS {exact_field} disagrees "
                f"with {main_field}"
            )

    resource_map = {
        "allocated_registers": "registers",
        "eiattr_max_register_count": "max_register_count",
        "frame_bytes": "frame_bytes",
        "min_stack_bytes": "stack_bytes",
        "local_load_instructions": "local_loads",
        "local_store_instructions": "local_stores",
        "driver_local_bytes": "driver_local_bytes",
        "cubin_shared_section_bytes": "cubin_shared_section_bytes",
        "driver_static_shared_bytes": "driver_static_shared_bytes",
        "launch_dynamic_smem_bytes": "launch_dynamic_smem_bytes",
    }
    for side in ("baseline", "current"):
        for exact_resource, main_resource in resource_map.items():
            exact_value = _integer(sass, f"{side}_{exact_resource}")
            main_value = _integer(row, f"{side}_{main_resource}")
            if exact_value != main_value:
                raise ValueError(
                    f"delta row {row_number} exact SASS {side}_{exact_resource} "
                    f"is {exact_value}, expected {main_value}"
                )
        expected_total_shared = _integer(
            row, f"{side}_driver_static_shared_bytes"
        ) + _integer(row, f"{side}_launch_dynamic_smem_bytes")
        if _integer(sass, f"{side}_total_launch_shared_bytes") != expected_total_shared:
            raise ValueError(
                f"delta row {row_number} has inconsistent exact total launch SMEM"
            )
    for resource in _SASS_RESOURCE_FIELDS:
        baseline = _integer(sass, f"baseline_{resource}")
        current = _integer(sass, f"current_{resource}")
        delta = current - baseline
        if _integer(sass, f"{resource}_delta") != delta:
            raise ValueError(
                f"delta row {row_number} has inconsistent exact {resource} delta"
            )
        if _true(sass, f"{resource}_increase") != (delta > 0):
            raise ValueError(
                f"delta row {row_number} has inconsistent exact {resource} flag"
            )

    exact_local_increase = any(
        _true(sass, f"{resource}_increase")
        for resource in (
            "frame_bytes",
            "min_stack_bytes",
            "local_load_instructions",
            "local_store_instructions",
            "driver_local_bytes",
        )
    )
    exact_shared_increase = any(
        _true(sass, f"{resource}_increase")
        for resource in (
            "cubin_shared_section_bytes",
            "driver_static_shared_bytes",
            "launch_dynamic_smem_bytes",
            "total_launch_shared_bytes",
        )
    )
    if _true(sass, "any_local_memory_increase") != exact_local_increase:
        raise ValueError(
            f"delta row {row_number} has inconsistent exact local-memory summary"
        )
    if _true(sass, "any_shared_memory_increase") != exact_shared_increase:
        raise ValueError(
            f"delta row {row_number} has inconsistent exact shared-memory summary"
        )


def _sass_accounting_fields(
    sass: dict[str, str],
    *,
    row_number: int,
) -> tuple[dict[str, str | int], tuple[str, ...], bool]:
    fields: dict[str, str | int] = {}
    exceptions: list[str] = []
    any_set_change = False
    any_set_addition = False
    any_family_usage_increase = False

    resource_increases: dict[str, bool] = {}
    for resource in _SASS_RESOURCE_FIELDS:
        baseline = _integer(sass, f"baseline_{resource}")
        current = _integer(sass, f"current_{resource}")
        delta = current - baseline
        if _integer(sass, f"{resource}_delta") != delta:
            raise ValueError(
                f"SASS delta row {row_number} has inconsistent {resource}_delta"
            )
        increase = delta > 0
        if _true(sass, f"{resource}_increase") != increase:
            raise ValueError(
                f"SASS delta row {row_number} has inconsistent {resource}_increase"
            )
        fields.update(
            {
                f"baseline_{resource}": baseline,
                f"current_{resource}": current,
                f"{resource}_delta": delta,
                f"{resource}_increase": _bool_text(increase),
            }
        )
        resource_increases[resource] = increase

    baseline_reconfiguration = _true(sass, "baseline_register_reconfiguration")
    current_reconfiguration = _true(sass, "current_register_reconfiguration")
    reconfiguration_change = baseline_reconfiguration != current_reconfiguration
    if _true(sass, "register_reconfiguration_change") != reconfiguration_change:
        raise ValueError(
            f"SASS delta row {row_number} has inconsistent "
            "register_reconfiguration_change"
        )
    fields.update(
        {
            "baseline_register_reconfiguration": _bool_text(baseline_reconfiguration),
            "current_register_reconfiguration": _bool_text(current_reconfiguration),
            "register_reconfiguration_change": _bool_text(reconfiguration_change),
        }
    )
    if reconfiguration_change:
        exceptions.append("register_reconfiguration_change")

    target_change = False
    target_addition = False
    target_increase = False
    for target_field in _SASS_RECONFIGURATION_FIELDS:
        stem = target_field.removesuffix("_json")
        try:
            baseline_targets = json.loads(sass[f"baseline_{target_field}"])
            current_targets = json.loads(sass[f"current_{target_field}"])
            reported_added = json.loads(sass[f"{stem}_added_json"])
            reported_removed = json.loads(sass[f"{stem}_removed_json"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SASS delta row {row_number} has invalid {stem} JSON"
            ) from exc
        for label, targets in (
            ("baseline", baseline_targets),
            ("current", current_targets),
            ("added", reported_added),
            ("removed", reported_removed),
        ):
            if (
                not isinstance(targets, list)
                or any(
                    isinstance(target, bool)
                    or not isinstance(target, int)
                    or target <= 0
                    or target > 256
                    for target in targets
                )
                or targets != sorted(set(targets))
            ):
                raise ValueError(
                    f"SASS delta row {row_number} has invalid {label} {stem}"
                )
        added = sorted(set(current_targets) - set(baseline_targets))
        removed = sorted(set(baseline_targets) - set(current_targets))
        if added != reported_added or removed != reported_removed:
            raise ValueError(
                f"SASS delta row {row_number} has inconsistent {stem} delta"
            )
        changed = bool(added or removed)
        maximum_increase = max(current_targets, default=0) > max(
            baseline_targets, default=0
        )
        if (
            _true(sass, f"{stem}_set_change") != changed
            or _true(sass, f"{stem}_maximum_increase") != maximum_increase
        ):
            raise ValueError(
                f"SASS delta row {row_number} has inconsistent {stem} flags"
            )
        fields.update(
            {
                f"baseline_{target_field}": json.dumps(
                    baseline_targets, separators=(",", ":")
                ),
                f"current_{target_field}": json.dumps(
                    current_targets, separators=(",", ":")
                ),
                f"{stem}_added_json": json.dumps(added, separators=(",", ":")),
                f"{stem}_removed_json": json.dumps(removed, separators=(",", ":")),
                f"{stem}_set_change": _bool_text(changed),
                f"{stem}_maximum_increase": _bool_text(maximum_increase),
            }
        )
        target_change |= changed
        target_addition |= bool(added)
        target_increase |= maximum_increase
    for field, expected in (
        ("any_register_reconfiguration_target_change", target_change),
        ("any_register_reconfiguration_target_addition", target_addition),
        ("any_register_reconfiguration_target_increase", target_increase),
    ):
        if _true(sass, field) != expected:
            raise ValueError(f"SASS delta row {row_number} has inconsistent {field}")
        fields[field] = _bool_text(expected)
    if target_change:
        exceptions.append("register_reconfiguration_target_change")

    for family in _SASS_FAMILIES:
        try:
            baseline_indices = json.loads(sass[f"baseline_{family}_indices_json"])
            current_indices = json.loads(sass[f"current_{family}_indices_json"])
            reported_added = json.loads(sass[f"{family}_added_indices_json"])
            reported_removed = json.loads(sass[f"{family}_removed_indices_json"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SASS delta row {row_number} has invalid {family.upper()} JSON"
            ) from exc
        for label, indices in (
            ("baseline", baseline_indices),
            ("current", current_indices),
            ("added", reported_added),
            ("removed", reported_removed),
        ):
            if (
                not isinstance(indices, list)
                or any(not isinstance(index, int) or index < 0 for index in indices)
                or indices != sorted(set(indices))
            ):
                raise ValueError(
                    f"SASS delta row {row_number} has invalid {label} "
                    f"{family.upper()} set"
                )
        added = sorted(set(current_indices) - set(baseline_indices))
        removed = sorted(set(baseline_indices) - set(current_indices))
        if reported_added != added or reported_removed != removed:
            raise ValueError(
                f"SASS delta row {row_number} has inconsistent {family.upper()} set delta"
            )
        set_change = bool(added or removed)
        set_addition = bool(added)
        if _true(sass, f"{family}_set_change") != set_change:
            raise ValueError(
                f"SASS delta row {row_number} has inconsistent "
                f"{family.upper()} set-change flag"
            )

        fields.update(
            {
                f"baseline_{family}_indices_json": json.dumps(
                    baseline_indices, separators=(",", ":")
                ),
                f"current_{family}_indices_json": json.dumps(
                    current_indices, separators=(",", ":")
                ),
                f"{family}_added_indices_json": json.dumps(
                    added, separators=(",", ":")
                ),
                f"{family}_removed_indices_json": json.dumps(
                    removed, separators=(",", ":")
                ),
                f"{family}_set_change": _bool_text(set_change),
            }
        )

        family_increases: list[bool] = []
        for metric in ("count", "min", "max", "span"):
            baseline = (
                len(baseline_indices)
                if metric == "count"
                else min(baseline_indices)
                if metric == "min" and baseline_indices
                else max(baseline_indices)
                if metric == "max" and baseline_indices
                else max(baseline_indices) - min(baseline_indices) + 1
                if metric == "span" and baseline_indices
                else -1
                if metric in {"min", "max"}
                else 0
            )
            current = (
                len(current_indices)
                if metric == "count"
                else min(current_indices)
                if metric == "min" and current_indices
                else max(current_indices)
                if metric == "max" and current_indices
                else max(current_indices) - min(current_indices) + 1
                if metric == "span" and current_indices
                else -1
                if metric in {"min", "max"}
                else 0
            )
            delta = current - baseline
            for field, expected in (
                (f"baseline_{family}_{metric}", baseline),
                (f"current_{family}_{metric}", current),
                (f"{family}_{metric}_delta", delta),
            ):
                if _integer(sass, field) != expected:
                    raise ValueError(
                        f"SASS delta row {row_number} has inconsistent {field}"
                    )
            fields.update(
                {
                    f"baseline_{family}_{metric}": baseline,
                    f"current_{family}_{metric}": current,
                    f"{family}_{metric}_delta": delta,
                }
            )
            if metric in {"count", "max", "span"}:
                increase = delta > 0
                if _true(sass, f"{family}_{metric}_increase") != increase:
                    raise ValueError(
                        f"SASS delta row {row_number} has inconsistent "
                        f"{family}_{metric}_increase"
                    )
                fields[f"{family}_{metric}_increase"] = _bool_text(increase)
                family_increases.append(increase)
                if increase:
                    exceptions.append(f"sass_{family}_{metric}_increase")

        baseline_index_span = max(baseline_indices) + 1 if baseline_indices else 0
        current_index_span = max(current_indices) + 1 if current_indices else 0
        index_span_delta = current_index_span - baseline_index_span
        index_span_increase = index_span_delta > 0
        for field, expected in (
            (f"baseline_{family}_index_span", baseline_index_span),
            (f"current_{family}_index_span", current_index_span),
            (f"{family}_index_span_delta", index_span_delta),
        ):
            if _integer(sass, field) != expected:
                raise ValueError(
                    f"SASS delta row {row_number} has inconsistent {field}"
                )
        if _true(sass, f"{family}_index_span_increase") != index_span_increase:
            raise ValueError(
                f"SASS delta row {row_number} has inconsistent "
                f"{family}_index_span_increase"
            )
        fields.update(
            {
                f"baseline_{family}_index_span": baseline_index_span,
                f"current_{family}_index_span": current_index_span,
                f"{family}_index_span_delta": index_span_delta,
                f"{family}_index_span_increase": _bool_text(index_span_increase),
            }
        )
        family_increases.append(index_span_increase)
        if index_span_increase:
            exceptions.append(f"sass_{family}_index_span_increase")
        family_usage_increase = any(family_increases)
        if _true(sass, f"{family}_usage_increase") != family_usage_increase:
            raise ValueError(
                f"SASS delta row {row_number} has inconsistent "
                f"{family.upper()} usage-increase flag"
            )
        fields[f"{family}_usage_increase"] = _bool_text(family_usage_increase)
        any_set_change |= set_change
        any_set_addition |= set_addition
        any_family_usage_increase |= family_usage_increase

    expected_usage_increase = any(
        (
            resource_increases["allocated_registers"],
            resource_increases["eiattr_max_register_count"],
            resource_increases["effective_register_index_ceiling"],
            target_increase,
            any_family_usage_increase,
        )
    )
    if _true(sass, "any_register_set_change") != any_set_change:
        raise ValueError(
            f"SASS delta row {row_number} has inconsistent set-change summary"
        )
    if _true(sass, "any_register_set_addition") != any_set_addition:
        raise ValueError(
            f"SASS delta row {row_number} has inconsistent set-addition summary"
        )
    if _true(sass, "any_register_usage_increase") != expected_usage_increase:
        raise ValueError(
            f"SASS delta row {row_number} has inconsistent usage-increase summary"
        )
    fields.update(
        {
            "any_register_set_change": _bool_text(any_set_change),
            "any_register_set_addition": _bool_text(any_set_addition),
            "exact_sass_any_register_usage_increase": _bool_text(
                expected_usage_increase
            ),
        }
    )
    return fields, tuple(exceptions), expected_usage_increase


def _require_boolean(row: dict[str, str], field: str, expected: bool) -> None:
    actual = _true(row, field)
    if actual != expected:
        raise ValueError(
            f"derived field {field} is {actual}, expected {expected} from raw values"
        )


def _require_delta(row: dict[str, str], baseline: str, current: str, delta: str) -> int:
    expected = _integer(row, current) - _integer(row, baseline)
    reported = _integer(row, delta)
    if reported != expected:
        raise ValueError(f"derived field {delta} is {reported}, expected {expected}")
    return expected


def _local_memory(row: dict[str, str], side: str) -> bool:
    return any(
        _integer(row, f"{side}_{field}") > 0
        for field in ("frame_bytes", "stack_bytes", "local_loads", "local_stores")
    )


def _validate_and_recompute(row: dict[str, str]) -> dict[str, bool | int]:
    derived: dict[str, bool | int] = {}
    delta_fields = (
        (
            "baseline_threads_per_cta",
            "current_threads_per_cta",
            "threads_per_cta_delta",
        ),
        (
            "baseline_cubin_shared_section_bytes",
            "current_cubin_shared_section_bytes",
            "cubin_shared_section_delta",
        ),
        (
            "baseline_launch_dynamic_smem_bytes",
            "current_launch_dynamic_smem_bytes",
            "launch_dynamic_smem_delta",
        ),
        (
            "baseline_occupancy_active_ctas_per_sm",
            "current_occupancy_active_ctas_per_sm",
            "occupancy_active_ctas_per_sm_delta",
        ),
        (
            "baseline_occupancy_active_threads_per_sm",
            "current_occupancy_active_threads_per_sm",
            "occupancy_active_threads_per_sm_delta",
        ),
        (
            "baseline_driver_static_shared_bytes",
            "current_driver_static_shared_bytes",
            "driver_static_shared_bytes_delta",
        ),
        ("baseline_registers", "current_registers", "register_delta"),
        (
            "baseline_sass_uniform_registers_used",
            "current_sass_uniform_registers_used",
            "sass_uniform_registers_used_delta",
        ),
        (
            "baseline_sass_uniform_register_span",
            "current_sass_uniform_register_span",
            "sass_uniform_register_span_delta",
        ),
        (
            "baseline_sass_predicate_registers_used",
            "current_sass_predicate_registers_used",
            "sass_predicate_registers_used_delta",
        ),
        (
            "baseline_sass_predicate_register_span",
            "current_sass_predicate_register_span",
            "sass_predicate_register_span_delta",
        ),
        (
            "baseline_sass_uniform_predicate_registers_used",
            "current_sass_uniform_predicate_registers_used",
            "sass_uniform_predicate_registers_used_delta",
        ),
        (
            "baseline_sass_uniform_predicate_register_span",
            "current_sass_uniform_predicate_register_span",
            "sass_uniform_predicate_register_span_delta",
        ),
        ("baseline_frame_bytes", "current_frame_bytes", "frame_delta"),
        ("baseline_stack_bytes", "current_stack_bytes", "stack_delta"),
        ("baseline_local_loads", "current_local_loads", "local_load_delta"),
        ("baseline_local_stores", "current_local_stores", "local_store_delta"),
        (
            "baseline_sass_instructions",
            "current_sass_instructions",
            "sass_instruction_delta",
        ),
        ("baseline_code_bytes", "current_code_bytes", "code_bytes_delta"),
    )
    for baseline, current, delta in delta_fields:
        derived[delta] = _require_delta(row, baseline, current, delta)
    architecture_change = row.get("baseline_architecture") != row.get(
        "current_architecture"
    )
    _require_boolean(row, "architecture_change", architecture_change)
    derived["architecture_change"] = architecture_change

    increase_fields = (
        ("register_delta", "register_increase"),
        ("sass_uniform_registers_used_delta", "sass_uniform_registers_used_increase"),
        ("sass_uniform_register_span_delta", "sass_uniform_register_span_increase"),
        (
            "sass_predicate_registers_used_delta",
            "sass_predicate_registers_used_increase",
        ),
        ("sass_predicate_register_span_delta", "sass_predicate_register_span_increase"),
        (
            "sass_uniform_predicate_registers_used_delta",
            "sass_uniform_predicate_registers_used_increase",
        ),
        (
            "sass_uniform_predicate_register_span_delta",
            "sass_uniform_predicate_register_span_increase",
        ),
    )
    for delta, flag in increase_fields:
        expected = int(derived[delta]) > 0
        _require_boolean(row, flag, expected)
        derived[flag] = expected

    # Code growth is not itself a correctness failure, but it is compiler
    # resource evidence that must remain visible and causally annotated.  In
    # particular, equal register sets can still hide extra address/predicate
    # lowering in a hot loop.
    derived["sass_instruction_increase"] = int(derived["sass_instruction_delta"]) > 0
    derived["code_bytes_increase"] = int(derived["code_bytes_delta"]) > 0

    register_flags = tuple(flag for _, flag in increase_fields)
    any_register_increase = any(bool(derived[field]) for field in register_flags)
    _require_boolean(row, "any_register_usage_increase", any_register_increase)
    derived["any_register_usage_increase"] = any_register_increase
    new_ceiling = (
        _integer(row, "current_registers") >= 255
        and _integer(row, "baseline_registers") < 255
    )
    _require_boolean(row, "new_register_ceiling", new_ceiling)
    derived["new_register_ceiling"] = new_ceiling

    baseline_local = _local_memory(row, "baseline")
    current_local = _local_memory(row, "current")
    new_local = current_local and not baseline_local
    local_increase = any(
        int(derived[field]) > 0
        for field in (
            "frame_delta",
            "stack_delta",
            "local_load_delta",
            "local_store_delta",
        )
    )
    for field, expected in (
        ("baseline_local_memory", baseline_local),
        ("current_local_memory", current_local),
        ("new_local_memory", new_local),
        ("local_memory_increase", local_increase),
    ):
        _require_boolean(row, field, expected)
        derived[field] = expected

    cubin_change = int(derived["cubin_shared_section_delta"]) != 0
    cubin_increase = int(derived["cubin_shared_section_delta"]) > 0
    launch_change = int(derived["launch_dynamic_smem_delta"]) != 0
    launch_increase = int(derived["launch_dynamic_smem_delta"]) > 0
    occupancy_decrease = int(derived["occupancy_active_ctas_per_sm_delta"]) < 0
    driver_static_change = int(derived["driver_static_shared_bytes_delta"]) != 0
    driver_static_increase = int(derived["driver_static_shared_bytes_delta"]) > 0
    for field, expected in (
        ("cubin_shared_section_change", cubin_change),
        ("cubin_shared_section_increase", cubin_increase),
        ("launch_dynamic_smem_comparable", True),
        ("launch_dynamic_smem_change", launch_change),
        ("launch_dynamic_smem_increase", launch_increase),
        ("driver_occupancy_comparable", True),
        ("driver_occupancy_decrease", occupancy_decrease),
        ("driver_static_shared_bytes_change", driver_static_change),
        ("driver_static_shared_bytes_increase", driver_static_increase),
    ):
        _require_boolean(row, field, expected)
        derived[field] = expected

    launch_fields = (
        "architecture",
        "threads_x",
        "threads_y",
        "threads_z",
        "max_register_count",
        "parameter_bytes",
        "launch_dynamic_smem_status",
        "launch_dynamic_smem_bytes",
        "launch_dynamic_smem_count",
        "launch_dynamic_smem_values_json",
        "launch_metadata_source",
        "launch_metadata_reason",
    )
    launch_metadata_change = any(
        row.get(f"baseline_{field}", "") != row.get(f"current_{field}", "")
        for field in launch_fields
    )
    _require_boolean(row, "launch_metadata_change", launch_metadata_change)
    derived["launch_metadata_change"] = launch_metadata_change
    return derived


def _exception_fields(derived: dict[str, bool | int]) -> tuple[str, ...]:
    fields = []
    for field in (
        "register_increase",
        "sass_uniform_registers_used_increase",
        "sass_uniform_register_span_increase",
        "sass_predicate_registers_used_increase",
        "sass_predicate_register_span_increase",
        "sass_uniform_predicate_registers_used_increase",
        "sass_uniform_predicate_register_span_increase",
        "new_register_ceiling",
        "driver_occupancy_decrease",
        "driver_static_shared_bytes_change",
        "local_memory_increase",
        "cubin_shared_section_change",
        "launch_dynamic_smem_change",
        "launch_metadata_change",
        "sass_instruction_increase",
        "code_bytes_increase",
    ):
        if bool(derived[field]):
            fields.append(field)
    return tuple(fields)


def _annotation_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("comparison_semantic_key", ""), row.get("symbol_sha256", "")


def _require_exact_pair(row: dict[str, str], row_number: int) -> None:
    if row.get("delta_report_schema") != _DELTA_REPORT_SCHEMA:
        raise ValueError(f"delta row {row_number} has an invalid schema")
    if (
        row.get("baseline_resource_report_schema") != _RESOURCE_REPORT_SCHEMA
        or row.get("current_resource_report_schema") != _RESOURCE_REPORT_SCHEMA
    ):
        raise ValueError(f"delta row {row_number} does not pair v4 resource reports")
    if row.get("identity_kind") != "comparison-semantic-kernel":
        raise ValueError(
            f"delta row {row_number} is not a comparison-semantic-kernel row"
        )
    if row.get("pairing") != "exact-comparison-semantic-kernel":
        raise ValueError(
            f"delta row {row_number} is not an exact one-to-one pair: "
            f"{row.get('pairing')!r}"
        )
    if (
        _integer(row, "pair_ordinal") != 0
        or _integer(row, "baseline_identity_count") != 1
        or _integer(row, "current_identity_count") != 1
    ):
        raise ValueError(f"delta row {row_number} is not a unique ordinal-zero pair")
    for field in _ANNOTATION_KEY:
        if not row.get(field):
            raise ValueError(f"delta row {row_number} has no {field}")
        if not _SHA256_RE.fullmatch(row[field]):
            raise ValueError(f"delta row {row_number} has invalid {field}")
    for side in ("baseline", "current"):
        if row.get(f"{side}_manifest_status") != "ok":
            raise ValueError(f"delta row {row_number} has an invalid {side} manifest")
        if not row.get(f"{side}_package_fingerprint"):
            raise ValueError(f"delta row {row_number} has no {side} fingerprint")
        if row.get(f"{side}_launch_dynamic_smem_status") != "exact":
            raise ValueError(f"delta row {row_number} has inexact {side} launch SMEM")
        if row.get(f"{side}_occupancy_status") != "exact-driver-query":
            raise ValueError(f"delta row {row_number} has inexact {side} occupancy")
        if row.get(f"{side}_driver_resource_validation_status") != "exact-match":
            raise ValueError(
                f"delta row {row_number} has unvalidated {side} driver resources"
            )
        if row.get(f"{side}_launch_metadata_source") != (
            "cutlass-final-llvm-launch-config-field-2"
        ):
            raise ValueError(
                f"delta row {row_number} has an invalid {side} launch source"
            )
        for field in (
            "semantic_key",
            "package_fingerprint",
            "cache_key",
            "object_sha256",
            "compile_spec_hash",
        ):
            if not _SHA256_RE.fullmatch(row.get(f"{side}_{field}", "")):
                raise ValueError(f"delta row {row_number} has invalid {side}_{field}")
    if row.get("baseline_architecture") != row.get("current_architecture"):
        raise ValueError(f"delta row {row_number} changes architecture")
    if not _true(row, "driver_occupancy_comparable"):
        raise ValueError(
            f"delta row {row_number} lacks same-GPU exact driver occupancy"
        )
    baseline_uuid = row.get("baseline_occupancy_gpu_uuid", "")
    if not baseline_uuid or baseline_uuid != row.get("current_occupancy_gpu_uuid"):
        raise ValueError(f"delta row {row_number} does not use one exact GPU UUID")
    for field in (
        "target",
        "kernel_id",
        "compile_spec_version",
        "compile_spec_hash",
        "compile_spec_json",
        "ptxas_flags",
        "kernel",
    ):
        if row.get(f"baseline_{field}") != row.get(f"current_{field}"):
            raise ValueError(f"delta row {row_number} changes {field}")
    baseline_kwargs = _normalized_compile_kwargs(
        row.get("baseline_compile_kwargs_json", ""), side="baseline"
    )
    current_kwargs = _normalized_compile_kwargs(
        row.get("current_compile_kwargs_json", ""), side="current"
    )
    if baseline_kwargs != current_kwargs:
        raise ValueError(f"delta row {row_number} changes comparison compile kwargs")
    baseline_environment = _normalized_compile_environment(
        row.get("baseline_compile_environment_json", ""), side="baseline"
    )
    current_environment = _normalized_compile_environment(
        row.get("current_compile_environment_json", ""), side="current"
    )
    if baseline_environment != current_environment:
        raise ValueError(
            f"delta row {row_number} changes semantic compile environment: "
            f"baseline={baseline_environment!r} current={current_environment!r}"
        )
    baseline_options = _normalized_compile_options(
        row.get("baseline_compile_options_json", ""), side="baseline"
    )
    current_options = _normalized_compile_options(
        row.get("current_compile_options_json", ""), side="current"
    )
    if baseline_options != current_options:
        raise ValueError(
            f"delta row {row_number} changes semantic compile options: "
            f"baseline={baseline_options!r} current={current_options!r}"
        )
    for side in ("baseline", "current"):
        if not row.get(f"{side}_ptxas_version", "").strip():
            raise ValueError(f"delta row {row_number} has no {side} PTXAS version")
    for side in ("baseline", "current"):
        threads = _integer(row, f"{side}_threads_per_cta")
        dimensions = (
            _integer(row, f"{side}_threads_x"),
            _integer(row, f"{side}_threads_y"),
            _integer(row, f"{side}_threads_z"),
        )
        if threads != dimensions[0] * dimensions[1] * dimensions[2]:
            raise ValueError(f"delta row {row_number} has invalid {side} CTA shape")
        active_ctas = _integer(row, f"{side}_occupancy_active_ctas_per_sm")
        active_threads = _integer(row, f"{side}_occupancy_active_threads_per_sm")
        if active_ctas <= 0 or active_threads != active_ctas * threads:
            raise ValueError(
                f"delta row {row_number} has invalid {side} driver occupancy"
            )
        launch_count = _integer(row, f"{side}_launch_dynamic_smem_count")
        try:
            launch_values = json.loads(
                row.get(f"{side}_launch_dynamic_smem_values_json", "")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"delta row {row_number} has invalid {side} launch values"
            ) from exc
        if (
            not isinstance(launch_values, list)
            or len(launch_values) != launch_count
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in launch_values
            )
            or max(launch_values, default=-1)
            != _integer(row, f"{side}_launch_dynamic_smem_bytes")
        ):
            raise ValueError(
                f"delta row {row_number} has inconsistent {side} launch values"
            )
        if _integer(row, f"{side}_driver_registers") != _integer(
            row, f"{side}_registers"
        ):
            raise ValueError(f"delta row {row_number} has invalid {side} driver GPRs")
        if _integer(row, f"{side}_driver_local_bytes") != _integer(
            row, f"{side}_frame_bytes"
        ):
            raise ValueError(
                f"delta row {row_number} has invalid {side} driver local bytes"
            )
        if _integer(row, f"{side}_driver_max_threads_per_block") < threads:
            raise ValueError(
                f"delta row {row_number} has invalid {side} max-thread limit"
            )


def _accounting_row(
    row: dict[str, str],
    annotation: dict[str, str],
    exceptions: tuple[str, ...],
    derived: dict[str, bool | int],
    sass_fields: dict[str, str | int],
    exact_register_usage_increase: bool,
) -> dict[str, str | int]:
    baseline_threads = _integer(row, "baseline_threads_per_cta")
    current_threads = _integer(row, "current_threads_per_cta")
    baseline_registers = _integer(row, "baseline_registers")
    current_registers = _integer(row, "current_registers")
    return {
        "accounting_schema": _ACCOUNTING_SCHEMA,
        "family": row.get("family", ""),
        "comparison_semantic_key": row["comparison_semantic_key"],
        "baseline_semantic_key": row["baseline_semantic_key"],
        "current_semantic_key": row["current_semantic_key"],
        "symbol_sha256": row["symbol_sha256"],
        "baseline_package_fingerprint": row["baseline_package_fingerprint"],
        "current_package_fingerprint": row["current_package_fingerprint"],
        "compile_spec_json": row.get("current_compile_spec_json", ""),
        "kernel": row.get("current_kernel", ""),
        "architecture": row.get("current_architecture", ""),
        "baseline_cutlass_dsl": row.get("baseline_cutlass_dsl_version", ""),
        "current_cutlass_dsl": row.get("current_cutlass_dsl_version", ""),
        "baseline_ptxas": row.get("baseline_ptxas_version", ""),
        "current_ptxas": row.get("current_ptxas_version", ""),
        "baseline_compile_options_json": row.get("baseline_compile_options_json", ""),
        "current_compile_options_json": row.get("current_compile_options_json", ""),
        "baseline_threads_per_cta": baseline_threads,
        "current_threads_per_cta": current_threads,
        "baseline_registers_per_thread": baseline_registers,
        "current_registers_per_thread": current_registers,
        "register_delta": int(derived["register_delta"]),
        "register_increase": _bool_text(derived["register_increase"]),
        "baseline_sass_uniform_registers_used": _integer(
            row, "baseline_sass_uniform_registers_used"
        ),
        "current_sass_uniform_registers_used": _integer(
            row, "current_sass_uniform_registers_used"
        ),
        "sass_uniform_registers_used_delta": int(
            derived["sass_uniform_registers_used_delta"]
        ),
        "sass_uniform_registers_used_increase": _bool_text(
            derived["sass_uniform_registers_used_increase"]
        ),
        "baseline_sass_uniform_register_span": _integer(
            row, "baseline_sass_uniform_register_span"
        ),
        "current_sass_uniform_register_span": _integer(
            row, "current_sass_uniform_register_span"
        ),
        "sass_uniform_register_span_delta": int(
            derived["sass_uniform_register_span_delta"]
        ),
        "sass_uniform_register_span_increase": _bool_text(
            derived["sass_uniform_register_span_increase"]
        ),
        "baseline_sass_predicate_registers_used": _integer(
            row, "baseline_sass_predicate_registers_used"
        ),
        "current_sass_predicate_registers_used": _integer(
            row, "current_sass_predicate_registers_used"
        ),
        "sass_predicate_registers_used_delta": int(
            derived["sass_predicate_registers_used_delta"]
        ),
        "sass_predicate_registers_used_increase": _bool_text(
            derived["sass_predicate_registers_used_increase"]
        ),
        "baseline_sass_predicate_register_span": _integer(
            row, "baseline_sass_predicate_register_span"
        ),
        "current_sass_predicate_register_span": _integer(
            row, "current_sass_predicate_register_span"
        ),
        "sass_predicate_register_span_delta": int(
            derived["sass_predicate_register_span_delta"]
        ),
        "sass_predicate_register_span_increase": _bool_text(
            derived["sass_predicate_register_span_increase"]
        ),
        "baseline_sass_uniform_predicate_registers_used": _integer(
            row, "baseline_sass_uniform_predicate_registers_used"
        ),
        "current_sass_uniform_predicate_registers_used": _integer(
            row, "current_sass_uniform_predicate_registers_used"
        ),
        "sass_uniform_predicate_registers_used_delta": int(
            derived["sass_uniform_predicate_registers_used_delta"]
        ),
        "sass_uniform_predicate_registers_used_increase": _bool_text(
            derived["sass_uniform_predicate_registers_used_increase"]
        ),
        "baseline_sass_uniform_predicate_register_span": _integer(
            row, "baseline_sass_uniform_predicate_register_span"
        ),
        "current_sass_uniform_predicate_register_span": _integer(
            row, "current_sass_uniform_predicate_register_span"
        ),
        "sass_uniform_predicate_register_span_delta": int(
            derived["sass_uniform_predicate_register_span_delta"]
        ),
        "sass_uniform_predicate_register_span_increase": _bool_text(
            derived["sass_uniform_predicate_register_span_increase"]
        ),
        "coarse_any_register_usage_increase": _bool_text(
            derived["any_register_usage_increase"]
        ),
        **sass_fields,
        "any_register_usage_increase": _bool_text(exact_register_usage_increase),
        "new_register_ceiling": _bool_text(derived["new_register_ceiling"]),
        "baseline_register_footprint_per_cta": (baseline_registers * baseline_threads),
        "current_register_footprint_per_cta": current_registers * current_threads,
        "register_footprint_delta": (
            current_registers * current_threads - baseline_registers * baseline_threads
        ),
        "baseline_max_register_count": _integer(row, "baseline_max_register_count"),
        "current_max_register_count": _integer(row, "current_max_register_count"),
        "occupancy_gpu_uuid": row.get("current_occupancy_gpu_uuid", ""),
        "baseline_active_ctas_per_sm": _integer(
            row, "baseline_occupancy_active_ctas_per_sm"
        ),
        "current_active_ctas_per_sm": _integer(
            row, "current_occupancy_active_ctas_per_sm"
        ),
        "active_ctas_per_sm_delta": int(derived["occupancy_active_ctas_per_sm_delta"]),
        "driver_occupancy_decrease": _bool_text(derived["driver_occupancy_decrease"]),
        "baseline_active_threads_per_sm": _integer(
            row, "baseline_occupancy_active_threads_per_sm"
        ),
        "current_active_threads_per_sm": _integer(
            row, "current_occupancy_active_threads_per_sm"
        ),
        "active_threads_per_sm_delta": int(
            derived["occupancy_active_threads_per_sm_delta"]
        ),
        "baseline_driver_registers": _integer(row, "baseline_driver_registers"),
        "current_driver_registers": _integer(row, "current_driver_registers"),
        "baseline_driver_local_bytes": _integer(row, "baseline_driver_local_bytes"),
        "current_driver_local_bytes": _integer(row, "current_driver_local_bytes"),
        "baseline_driver_static_shared_bytes": _integer(
            row, "baseline_driver_static_shared_bytes"
        ),
        "current_driver_static_shared_bytes": _integer(
            row, "current_driver_static_shared_bytes"
        ),
        "driver_static_shared_bytes_delta": int(
            derived["driver_static_shared_bytes_delta"]
        ),
        "driver_static_shared_bytes_change": _bool_text(
            derived["driver_static_shared_bytes_change"]
        ),
        "baseline_frame_bytes_per_thread": _integer(row, "baseline_frame_bytes"),
        "current_frame_bytes_per_thread": _integer(row, "current_frame_bytes"),
        "frame_delta": int(derived["frame_delta"]),
        "baseline_stack_bytes_per_thread": _integer(row, "baseline_stack_bytes"),
        "current_stack_bytes_per_thread": _integer(row, "current_stack_bytes"),
        "stack_delta": int(derived["stack_delta"]),
        "baseline_local_loads": _integer(row, "baseline_local_loads"),
        "current_local_loads": _integer(row, "current_local_loads"),
        "local_load_delta": int(derived["local_load_delta"]),
        "baseline_local_stores": _integer(row, "baseline_local_stores"),
        "current_local_stores": _integer(row, "current_local_stores"),
        "local_store_delta": int(derived["local_store_delta"]),
        "baseline_local_memory": _bool_text(derived["baseline_local_memory"]),
        "current_local_memory": _bool_text(derived["current_local_memory"]),
        "new_local_memory": _bool_text(derived["new_local_memory"]),
        "local_memory_increase": _bool_text(derived["local_memory_increase"]),
        "baseline_launch_dynamic_smem_bytes": _integer(
            row, "baseline_launch_dynamic_smem_bytes"
        ),
        "current_launch_dynamic_smem_bytes": _integer(
            row, "current_launch_dynamic_smem_bytes"
        ),
        "launch_dynamic_smem_delta": int(derived["launch_dynamic_smem_delta"]),
        "launch_dynamic_smem_change": _bool_text(derived["launch_dynamic_smem_change"]),
        "baseline_cubin_shared_section_bytes": _integer(
            row, "baseline_cubin_shared_section_bytes"
        ),
        "current_cubin_shared_section_bytes": _integer(
            row, "current_cubin_shared_section_bytes"
        ),
        "cubin_shared_section_delta": int(derived["cubin_shared_section_delta"]),
        "cubin_shared_section_change": _bool_text(
            derived["cubin_shared_section_change"]
        ),
        "baseline_sass_instructions": _integer(row, "baseline_sass_instructions"),
        "current_sass_instructions": _integer(row, "current_sass_instructions"),
        "sass_instruction_delta": int(derived["sass_instruction_delta"]),
        "sass_instruction_increase": _bool_text(derived["sass_instruction_increase"]),
        "baseline_code_bytes": _integer(row, "baseline_code_bytes"),
        "current_code_bytes": _integer(row, "current_code_bytes"),
        "code_bytes_delta": int(derived["code_bytes_delta"]),
        "code_bytes_increase": _bool_text(derived["code_bytes_increase"]),
        "launch_metadata_change": _bool_text(derived["launch_metadata_change"]),
        "exception_fields": ";".join(exceptions),
        **{field: annotation.get(field, "") for field in _ANNOTATION_FIELDS},
    }


def _annotation_template_context(row: dict[str, str]) -> dict[str, str]:
    return {
        "kernel_id": row.get("current_kernel_id", ""),
        "compile_spec_hash": row.get("current_compile_spec_hash", ""),
        "baseline_cache_key": row.get("baseline_cache_key", ""),
        "current_cache_key": row.get("current_cache_key", ""),
    }


def _delta_summary(
    row: dict[str, str | int],
    *,
    field_prefix: str = "",
) -> dict[str, object]:
    deltas: dict[str, int] = {}
    set_changes: dict[str, object] = {}
    for field, raw in row.items():
        if field.startswith(field_prefix) and field.endswith("_delta"):
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid review delta {field}={raw!r}") from exc
            if value:
                deltas[field] = value
        if not (
            field.startswith(field_prefix)
            and field.endswith("_json")
            and ("_added_" in field or "_removed_" in field)
        ):
            continue
        try:
            values = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid review set JSON {field}={raw!r}") from exc
        if not isinstance(values, list):
            raise ValueError(f"invalid review set {field}={raw!r}")
        if values:
            set_changes[field] = values
    summary: dict[str, object] = {"deltas": deltas}
    if set_changes:
        summary["set_changes"] = set_changes
    return summary


def _native_delta_summary(row: dict[str, str | int]) -> str:
    summary = _delta_summary(row)
    if str(row.get("register_reconfiguration_change", "")).lower() == "true":
        summary["register_reconfiguration"] = {
            "baseline": str(row.get("baseline_register_reconfiguration", "")),
            "current": str(row.get("current_register_reconfiguration", "")),
        }
    return _compact_json(summary)


def _common_ptxas_delta_summary(row: dict[str, str]) -> str:
    summary = _delta_summary(row, field_prefix="common_")
    summary["increase_flags"] = {
        field: _true(row, field)
        for field in (
            "common_any_register_usage_increase",
            "common_any_local_memory_increase",
            "common_any_static_shared_memory_increase",
        )
    }
    return _compact_json(summary)


def _build_annotation_template(
    accounting: list[dict[str, str | int]],
    contexts: dict[tuple[str, str], dict[str, str]],
    annotations: dict[tuple[str, str], dict[str, str]],
    common_ptxas_deltas: dict[tuple[str, str], dict[str, str]],
    *,
    common_ptxas_requested: bool,
) -> list[dict[str, str | int]]:
    exception_rows: dict[tuple[str, str], dict[str, str | int]] = {}
    for row in accounting:
        if not str(row.get("exception_fields", "")):
            continue
        key = _annotation_key(
            {field: str(row.get(field, "")) for field in _ANNOTATION_KEY}
        )
        if not all(key):
            raise ValueError("exception accounting row has an empty annotation key")
        if key in exception_rows:
            raise ValueError(f"duplicate exception accounting key {key!r}")
        exception_rows[key] = row

    exception_keys = set(exception_rows)
    stale_annotations = sorted(set(annotations) - exception_keys)
    if stale_annotations:
        raise ValueError(
            "annotation-template annotations do not match current exception rows: "
            f"{stale_annotations!r}"
        )
    if common_ptxas_requested:
        missing_common = sorted(exception_keys - set(common_ptxas_deltas))
        stale_common = sorted(set(common_ptxas_deltas) - exception_keys)
        if missing_common or stale_common:
            raise ValueError(
                "common-PTXAS exact keys do not match current exception rows: "
                f"missing={missing_common!r} stale={stale_common!r}"
            )

    template: list[dict[str, str | int]] = []
    for key, row in exception_rows.items():
        context = contexts.get(key)
        if context is None:
            raise ValueError(f"exception accounting row has no source context: {key!r}")
        common = common_ptxas_deltas.get(key)
        if common is not None:
            if common.get("kernel", "") != str(row.get("kernel", "")):
                raise ValueError(f"common-PTXAS kernel is stale for {key!r}")
            for field in ("baseline_cache_key", "current_cache_key"):
                if common.get(field, "") != context[field]:
                    raise ValueError(f"common-PTXAS {field} is stale for {key!r}")
        if key in annotations:
            review = {field: annotations[key][field] for field in _ANNOTATION_FIELDS}
            annotation_origin = "preserved-exact-key"
        else:
            review = dict(_DEFAULT_ANNOTATION)
            annotation_origin = "generated-default"
        template.append(
            {
                "annotation_template_schema": _ANNOTATION_TEMPLATE_SCHEMA,
                "comparison_semantic_key": key[0],
                "baseline_semantic_key": row.get("baseline_semantic_key", ""),
                "current_semantic_key": row.get("current_semantic_key", ""),
                "symbol_sha256": key[1],
                "family": row.get("family", ""),
                "kernel_id": context["kernel_id"],
                "kernel": row.get("kernel", ""),
                "compile_spec_hash": context["compile_spec_hash"],
                "compile_spec_json": row.get("compile_spec_json", ""),
                "architecture": row.get("architecture", ""),
                "baseline_package_fingerprint": row.get(
                    "baseline_package_fingerprint", ""
                ),
                "current_package_fingerprint": row.get(
                    "current_package_fingerprint", ""
                ),
                "baseline_cache_key": context["baseline_cache_key"],
                "current_cache_key": context["current_cache_key"],
                "baseline_cutlass_dsl": row.get("baseline_cutlass_dsl", ""),
                "current_cutlass_dsl": row.get("current_cutlass_dsl", ""),
                "baseline_ptxas": row.get("baseline_ptxas", ""),
                "current_ptxas": row.get("current_ptxas", ""),
                "exception_fields": row.get("exception_fields", ""),
                "native_delta_summary_json": _native_delta_summary(row),
                "common_ptxas_status": (
                    "exact-match" if common is not None else "not-provided"
                ),
                "common_ptxas_sha256": (
                    common.get("common_ptxas_sha256", "") if common else ""
                ),
                "common_ptxas_version_output_sha256": (
                    common.get("common_ptxas_version_output_sha256", "")
                    if common
                    else ""
                ),
                "common_ptxas_flags_json": (
                    common.get("common_ptxas_flags_json", "") if common else ""
                ),
                "common_ptxas_selection_reasons_json": (
                    common.get("selection_reasons_json", "") if common else ""
                ),
                "common_ptxas_delta_summary_json": (
                    _common_ptxas_delta_summary(common) if common else ""
                ),
                "annotation_origin": annotation_origin,
                **review,
            }
        )
    template.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["comparison_semantic_key"]),
            str(row["symbol_sha256"]),
        )
    )
    return template


def _write_annotation_template(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=_ANNOTATION_TEMPLATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delta_report", type=Path)
    parser.add_argument(
        "--sass-register-delta",
        type=Path,
        required=True,
        help=(
            "exact R/UR/P/UP delta CSV emitted by `python -m "
            "validation.cutlass_migration evidence compare-sass`"
        ),
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--annotation-template-output",
        type=Path,
        help=(
            "write a deterministic exception-only annotation CSV; exact-key "
            "annotations are preserved and all new rows remain unresolved"
        ),
    )
    parser.add_argument(
        "--common-ptxas-delta",
        type=Path,
        help=(
            "optionally join an exact common-PTXAS positive-delta CSV into the "
            "annotation template"
        ),
    )
    parser.add_argument(
        "--require-annotations-for-exceptions",
        action="store_true",
        help=(
            "fail unless every register/local-memory/SMEM/launch exception has "
            "a nonempty cause, disposition, evidence, and performance_status"
        ),
    )
    args = parser.parse_args()
    if args.common_ptxas_delta and not args.annotation_template_output:
        parser.error("--common-ptxas-delta requires --annotation-template-output")
    if args.annotation_template_output:
        template_path = args.annotation_template_output.resolve()
        protected_paths = {
            path.resolve()
            for path in (
                args.delta_report,
                args.sass_register_delta,
                args.annotations,
                args.common_ptxas_delta,
                args.output,
            )
            if path is not None
        }
        if template_path in protected_paths:
            parser.error(
                "--annotation-template-output must differ from every input and "
                "the accounting output"
            )

    try:
        annotations = _read_annotations(args.annotations)
        common_ptxas_deltas = _read_common_ptxas_deltas(args.common_ptxas_delta)
        sass_deltas = _read_sass_deltas(args.sass_register_delta)
        template_contexts: dict[tuple[str, str], dict[str, str]] = {}
        with args.delta_report.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != _DELTA_REPORT_FIELDS:
                raise ValueError(
                    f"{args.delta_report}: expected exact {_DELTA_REPORT_SCHEMA} header"
                )
            accounting: list[dict[str, str | int]] = []
            used_annotations: set[tuple[str, str]] = set()
            used_sass_deltas: set[tuple[str, str]] = set()
            exceptions_count = 0
            for row_number, row in enumerate(reader, start=2):
                _require_exact_pair(row, row_number)
                sass_key = (
                    row["comparison_semantic_key"],
                    row["current_kernel"],
                )
                sass = sass_deltas.get(sass_key)
                if sass is None:
                    raise ValueError(
                        f"delta row {row_number} has no exact SASS sidecar pair"
                    )
                _require_sass_delta_pair(row, sass, row_number)
                sass_fields, sass_exceptions, exact_register_increase = (
                    _sass_accounting_fields(sass, row_number=row_number)
                )
                used_sass_deltas.add(sass_key)
                key = _annotation_key(row)
                annotation = annotations.get(key, {})
                if annotation:
                    used_annotations.add(key)
                derived = _validate_and_recompute(row)
                if bool(derived["any_register_usage_increase"]) and not (
                    exact_register_increase
                ):
                    raise ValueError(
                        f"delta row {row_number} exact SASS report loses a "
                        "resource-report register increase"
                    )
                exceptions = tuple(
                    dict.fromkeys((*_exception_fields(derived), *sass_exceptions))
                )
                exceptions_count += bool(exceptions)
                if args.require_annotations_for_exceptions and exceptions:
                    missing = [
                        field
                        for field in _ANNOTATION_FIELDS
                        if not annotation.get(field, "")
                    ]
                    if missing:
                        raise ValueError(
                            f"delta row {row_number} exception {exceptions!r} has "
                            f"no complete annotation; missing {missing!r}"
                        )
                accounting_row = _accounting_row(
                    row,
                    annotation,
                    exceptions,
                    derived,
                    sass_fields,
                    exact_register_increase,
                )
                accounting.append(accounting_row)
                if args.annotation_template_output:
                    if key in template_contexts:
                        raise ValueError(
                            f"duplicate annotation-template source key {key!r}"
                        )
                    template_contexts[key] = _annotation_template_context(row)
        if not accounting:
            raise ValueError(f"{args.delta_report}: delta report has no rows")
        unused = sorted(set(annotations) - used_annotations)
        if unused:
            raise ValueError(f"annotations do not match delta rows: {unused!r}")
        unmatched_sass = sorted(set(sass_deltas) - used_sass_deltas)
        if unmatched_sass:
            raise ValueError(
                f"exact SASS rows do not match delta rows: {unmatched_sass!r}"
            )
        annotation_template = (
            _build_annotation_template(
                accounting,
                template_contexts,
                annotations,
                common_ptxas_deltas,
                common_ptxas_requested=args.common_ptxas_delta is not None,
            )
            if args.annotation_template_output
            else []
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    accounting.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["compile_spec_json"]),
            str(row["kernel"]),
        )
    )
    fieldnames = list(accounting[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    output = (
        args.output.open("w", newline="", encoding="utf-8")
        if args.output
        else sys.stdout
    )
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accounting)
    finally:
        if args.output:
            output.close()
    if args.annotation_template_output:
        _write_annotation_template(args.annotation_template_output, annotation_template)

    increases = sum(int(row["register_delta"]) > 0 for row in accounting)
    any_register_usage_increases = sum(
        str(row["any_register_usage_increase"]).lower() == "true" for row in accounting
    )
    ceilings = sum(
        int(row["current_registers_per_thread"]) >= 255 for row in accounting
    )
    local_rows = sum(
        any(
            int(row[field]) > 0
            for field in (
                "current_frame_bytes_per_thread",
                "current_stack_bytes_per_thread",
                "current_local_loads",
                "current_local_stores",
            )
        )
        for row in accounting
    )
    summary = (
        f"rows={len(accounting)} register_increases={increases} "
        f"any_register_usage_increases={any_register_usage_increases} "
        f"register_ceilings={ceilings} current_local_rows={local_rows} "
        f"exception_rows={exceptions_count}"
    )
    if args.annotation_template_output:
        summary += f" annotation_template_rows={len(annotation_template)}"
    print(summary, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
