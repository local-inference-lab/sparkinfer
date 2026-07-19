#!/usr/bin/env python3
# ruff: noqa: SIM905
"""Compare resource reports emitted by audit_cute_kernel_resources.py.

New reports are paired by a benchmark-only cross-toolchain comparison key and
the exact CUDA entry-point symbol within that object. Raw production semantic
keys remain arm-specific evidence and are never treated as equal. Legacy reports
remain readable, but their objects are paired only as explicitly labelled
symbol multisets; a legacy row is never guessed to match a row carrying
semantic metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from validation.cutlass_migration.core.comparison_identity import (
    comparison_semantic_key_from_resource_row,
    normalize_comparison_compile_environment,
)
from validation.cutlass_migration.evidence.kernel_resources import (
    _read_contract_metadata,
    _read_specialization_contract,
)

_RESOURCE_REPORT_SCHEMA = "b12x.cute.kernel_resources.v4"
_DELTA_REPORT_SCHEMA = "b12x.cute.kernel_resource_delta.v4"
_RESOURCE_REPORT_FIELDS = (
    *(
        "resource_report_schema,object_file,object_sha256,cache_key,manifest_status,"
        "manifest_schema,semantic_key,comparison_semantic_key,target,kernel_id,compile_spec_version,"
        "compile_spec_hash,compile_spec_json,compile_kwargs_json,package_fingerprint,"
        "python_version,torch_version,torch_cuda_version,cutlass_dsl_version,"
        "cutlass_dsl_libs_base_version,cutlass_dsl_libs_core_version,"
        "cutlass_dsl_libs_cu12_version,cutlass_dsl_libs_cu13_version,toolchain_json,"
        "compile_options_json,compile_environment_json,kernel,architecture,ptxas_version,"
        "ptxas_flags,threads_x,threads_y,threads_z,threads_per_cta,"
        "cubin_shared_section_bytes,launch_dynamic_smem_bytes,launch_dynamic_smem_count,"
        "launch_dynamic_smem_values_json,launch_dynamic_smem_status,"
        "launch_metadata_source,launch_metadata_reason,occupancy_status,"
        "occupancy_device_ordinal,occupancy_gpu_name,occupancy_gpu_uuid,"
        "occupancy_active_ctas_per_sm,occupancy_active_threads_per_sm,"
        "driver_resource_validation_status,driver_registers,driver_local_bytes,"
        "driver_static_shared_bytes,driver_max_threads_per_block,max_register_count,"
        "parameter_bytes,registers,sass_uniform_registers_used,"
        "sass_uniform_register_span,sass_predicate_registers_used,"
        "sass_predicate_register_span,sass_uniform_predicate_registers_used,"
        "sass_uniform_predicate_register_span,frame_bytes,min_stack_bytes,"
        "local_load_instructions,local_store_instructions,sass_instructions,code_bytes,"
        "register_ceiling,local_memory"
    ).split(","),
)
_REQUIRED_INTEGER_FIELDS = (
    "threads_x",
    "threads_y",
    "threads_z",
    "threads_per_cta",
    "cubin_shared_section_bytes",
    "launch_dynamic_smem_count",
    "max_register_count",
    "parameter_bytes",
    "registers",
    "sass_uniform_registers_used",
    "sass_uniform_register_span",
    "sass_predicate_registers_used",
    "sass_predicate_register_span",
    "sass_uniform_predicate_registers_used",
    "sass_uniform_predicate_register_span",
    "frame_bytes",
    "min_stack_bytes",
    "local_load_instructions",
    "local_store_instructions",
    "sass_instructions",
    "code_bytes",
)
_NULLABLE_INTEGER_FIELDS = (
    "launch_dynamic_smem_bytes",
    "occupancy_device_ordinal",
    "occupancy_active_ctas_per_sm",
    "occupancy_active_threads_per_sm",
    "driver_registers",
    "driver_local_bytes",
    "driver_static_shared_bytes",
    "driver_max_threads_per_block",
)
_INTEGER_FIELDS = _REQUIRED_INTEGER_FIELDS + _NULLABLE_INTEGER_FIELDS
_CUTLASS_PACKAGE_FIELDS = {
    "nvidia-cutlass-dsl": "cutlass_dsl_version",
    "nvidia-cutlass-dsl-libs-base": "cutlass_dsl_libs_base_version",
    "nvidia-cutlass-dsl-libs-core": "cutlass_dsl_libs_core_version",
    "nvidia-cutlass-dsl-libs-cu12": "cutlass_dsl_libs_cu12_version",
    "nvidia-cutlass-dsl-libs-cu13": "cutlass_dsl_libs_cu13_version",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESOURCE_SORT_FIELDS = (
    "registers",
    "sass_uniform_register_span",
    "sass_predicate_register_span",
    "sass_uniform_predicate_register_span",
    "frame_bytes",
    "min_stack_bytes",
    "local_load_instructions",
    "local_store_instructions",
    "cubin_shared_section_bytes",
    "code_bytes",
)


def _manifest_ok(row: dict[str, Any]) -> bool:
    return (
        row.get("manifest_status") == "ok"
        and bool(row.get("semantic_key"))
        and bool(row.get("comparison_semantic_key"))
    )


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    if _manifest_ok(row):
        # A single compiled host object can contain multiple CUDA entry points.
        # Pair each emitted cubin kernel independently so a resource increase in
        # one entry point cannot be hidden by multiset ordering within the same
        # semantic compile specialization.
        return (
            "comparison-semantic-kernel",
            f"{row['comparison_semantic_key']}\0{row['kernel']}",
        )
    return "legacy-symbol", str(row["kernel"])


def _validate_strict_resource_row(
    path: Path, row_number: int, row: dict[str, Any]
) -> None:
    prefix = f"{path}:{row_number}"
    required_nonempty = (
        "object_file",
        "object_sha256",
        "cache_key",
        "manifest_status",
        "manifest_schema",
        "semantic_key",
        "comparison_semantic_key",
        "target",
        "kernel_id",
        "compile_spec_version",
        "compile_spec_hash",
        "compile_spec_json",
        "package_fingerprint",
        "python_version",
        "torch_version",
        "torch_cuda_version",
        *_CUTLASS_PACKAGE_FIELDS.values(),
        "toolchain_json",
        "compile_options_json",
        "compile_environment_json",
        "kernel",
        "architecture",
        "ptxas_version",
        "ptxas_flags",
        "launch_dynamic_smem_status",
        "launch_metadata_source",
        "occupancy_status",
        "driver_resource_validation_status",
    )
    empty = [field for field in required_nonempty if not row.get(field)]
    if empty:
        raise ValueError(f"{prefix}: empty required fields {empty!r}")
    for field in (
        "object_sha256",
        "cache_key",
        "semantic_key",
        "comparison_semantic_key",
        "compile_spec_hash",
        "package_fingerprint",
    ):
        if not _SHA256_RE.fullmatch(str(row[field])):
            raise ValueError(f"{prefix}: {field} is not SHA-256")
    if row["manifest_status"] != "ok":
        raise ValueError(f"{prefix}: semantic manifest is not valid")
    if row["manifest_schema"] != "b12x.cute.compile_manifest.v3":
        raise ValueError(f"{prefix}: unsupported semantic manifest schema")
    if comparison_semantic_key_from_resource_row(row) != row["comparison_semantic_key"]:
        raise ValueError(f"{prefix}: comparison semantic key mismatch")
    if (
        hashlib.sha256(str(row["compile_spec_json"]).encode("utf-8")).hexdigest()
        != row["compile_spec_hash"]
    ):
        raise ValueError(f"{prefix}: compile spec hash mismatch")
    try:
        compile_spec = json.loads(str(row["compile_spec_json"]))
        if row.get("compile_kwargs_json"):
            json.loads(str(row["compile_kwargs_json"]))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{prefix}: invalid compile JSON") from exc
    if not isinstance(compile_spec, dict):
        raise ValueError(f"{prefix}: compile spec is not an object")
    if compile_spec.get("kernel") != row["kernel_id"]:
        raise ValueError(f"{prefix}: kernel id disagrees with compile spec")
    if str(compile_spec.get("version", "")) != row["compile_spec_version"]:
        raise ValueError(f"{prefix}: version disagrees with compile spec")
    for field in ("toolchain_json", "compile_options_json", "compile_environment_json"):
        try:
            json.loads(str(row[field]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{prefix}: invalid JSON in {field}") from exc
    if row["threads_per_cta"] != (
        row["threads_x"] * row["threads_y"] * row["threads_z"]
    ):
        raise ValueError(f"{prefix}: threads_per_cta disagrees with dimensions")
    if min(row["threads_x"], row["threads_y"], row["threads_z"]) <= 0:
        raise ValueError(f"{prefix}: nonpositive thread dimension")
    if row["registers"] <= 0 or row["sass_instructions"] <= 0 or row["code_bytes"] <= 0:
        raise ValueError(f"{prefix}: nonpositive register/code accounting")
    for field in ("register_ceiling", "local_memory"):
        if row.get(field) not in {"true", "false", "True", "False"}:
            raise ValueError(f"{prefix}: invalid boolean {field}={row.get(field)!r}")
    register_ceiling = row["registers"] >= 255
    local_memory = any(
        row[field] > 0
        for field in (
            "frame_bytes",
            "min_stack_bytes",
            "local_load_instructions",
            "local_store_instructions",
        )
    )
    if str(row["register_ceiling"]).lower() != _bool(register_ceiling):
        raise ValueError(f"{prefix}: register_ceiling is inconsistent")
    if str(row["local_memory"]).lower() != _bool(local_memory):
        raise ValueError(f"{prefix}: local_memory is inconsistent")

    if row["launch_dynamic_smem_status"] == "exact":
        if row["launch_dynamic_smem_bytes"] is None:
            raise ValueError(f"{prefix}: exact launch SMEM has no byte count")
        if row["launch_dynamic_smem_count"] <= 0:
            raise ValueError(f"{prefix}: exact launch SMEM has no observations")
        if row["launch_metadata_source"] != "cutlass-final-llvm-launch-config-field-2":
            raise ValueError(f"{prefix}: exact launch SMEM has unknown source")
        try:
            launch_values = json.loads(str(row["launch_dynamic_smem_values_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{prefix}: invalid launch SMEM values JSON") from exc
        if (
            not isinstance(launch_values, list)
            or len(launch_values) != row["launch_dynamic_smem_count"]
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in launch_values
            )
            or max(launch_values, default=-1) != row["launch_dynamic_smem_bytes"]
        ):
            raise ValueError(f"{prefix}: inconsistent launch SMEM observations")

    if row["occupancy_status"] == "exact-driver-query":
        occupancy_fields = (
            "occupancy_device_ordinal",
            "occupancy_active_ctas_per_sm",
            "occupancy_active_threads_per_sm",
            "driver_registers",
            "driver_local_bytes",
            "driver_static_shared_bytes",
            "driver_max_threads_per_block",
        )
        if any(row[field] is None for field in occupancy_fields):
            raise ValueError(f"{prefix}: exact occupancy has missing driver fields")
        if not row.get("occupancy_gpu_name") or not row.get("occupancy_gpu_uuid"):
            raise ValueError(f"{prefix}: exact occupancy has no GPU identity")
        if row["occupancy_active_ctas_per_sm"] <= 0:
            raise ValueError(f"{prefix}: nonpositive active CTAs/SM")
        if row["occupancy_active_threads_per_sm"] != (
            row["occupancy_active_ctas_per_sm"] * row["threads_per_cta"]
        ):
            raise ValueError(f"{prefix}: inconsistent active threads/SM")
        if row["driver_resource_validation_status"] != "exact-match":
            raise ValueError(f"{prefix}: driver resource validation did not match")
        if (
            row["driver_registers"] != row["registers"]
            or row["driver_local_bytes"] != row["frame_bytes"]
            or row["driver_max_threads_per_block"] < row["threads_per_cta"]
        ):
            raise ValueError(f"{prefix}: driver resources disagree with report")


def _read(
    path: Path,
    *,
    strict: bool = False,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], int, int]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows_read = 0
    missing_manifests = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if strict and tuple(reader.fieldnames or ()) != _RESOURCE_REPORT_FIELDS:
            raise ValueError(
                f"{path}: expected exact {_RESOURCE_REPORT_SCHEMA} columns; "
                f"got {reader.fieldnames!r}"
            )
        for row_number, raw_row in enumerate(reader, start=2):
            row: dict[str, Any] = dict(raw_row)
            # Reports emitted before the field was named precisely called the
            # cubin .nv.shared section size "static_shared_bytes".  Preserve
            # report readability without continuing that misleading name.
            if not row.get("cubin_shared_section_bytes"):
                row["cubin_shared_section_bytes"] = raw_row.get(
                    "static_shared_bytes", ""
                )
            if strict:
                if row.get("resource_report_schema") != _RESOURCE_REPORT_SCHEMA:
                    raise ValueError(
                        f"{path}:{row_number}: invalid resource_report_schema"
                    )
                for field in _REQUIRED_INTEGER_FIELDS:
                    raw = row.get(field, "")
                    if raw == "":
                        raise ValueError(f"{path}:{row_number}: empty integer {field}")
                    try:
                        value = int(raw)
                    except ValueError as exc:
                        raise ValueError(
                            f"{path}:{row_number}: invalid integer {field}={raw!r}"
                        ) from exc
                    if value < 0:
                        raise ValueError(f"{path}:{row_number}: negative {field}")
                    row[field] = value
                for field in _NULLABLE_INTEGER_FIELDS:
                    raw = row.get(field, "")
                    try:
                        value = int(raw) if raw != "" else None
                    except ValueError as exc:
                        raise ValueError(
                            f"{path}:{row_number}: invalid integer {field}={raw!r}"
                        ) from exc
                    if value is not None and value < 0:
                        raise ValueError(f"{path}:{row_number}: negative {field}")
                    row[field] = value
                _validate_strict_resource_row(path, row_number, row)
            else:
                for field in _REQUIRED_INTEGER_FIELDS:
                    row[field] = int(row[field]) if row.get(field) else 0
                for field in _NULLABLE_INTEGER_FIELDS:
                    row[field] = int(row[field]) if row.get(field) else None
            groups[_identity(row)].append(row)
            rows_read += 1
            missing_manifests += not _manifest_ok(row)
    for rows in groups.values():
        rows.sort(
            key=lambda row: (
                str(row.get("kernel", "")),
                *(int(row[field]) for field in _RESOURCE_SORT_FIELDS),
            )
        )
    return groups, rows_read, missing_manifests


def _family(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    kernel_id = str(row.get("kernel_id", ""))
    if kernel_id:
        return kernel_id
    target = str(row.get("target", ""))
    if target:
        return target
    kernel = str(row.get("kernel", ""))
    match = re.search(r"kernel_b12x(.*?)_object_at", kernel)
    return match.group(1) if match else kernel.split("__", 1)[0]


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _value(row: dict[str, Any] | None, field: str) -> Any:
    return row.get(field, "") if row is not None else ""


def _number(row: dict[str, Any] | None, field: str) -> int | str:
    if row is None or row.get(field) is None:
        return ""
    return int(row[field])


def _delta(
    old: dict[str, Any] | None, new: dict[str, Any] | None, field: str
) -> int | str:
    if old is None or new is None:
        return ""
    if old.get(field) is None or new.get(field) is None:
        return ""
    return int(new[field]) - int(old[field])


def _exact_launch_dynamic_smem(row: dict[str, Any] | None) -> bool:
    return bool(
        row is not None
        and row.get("launch_dynamic_smem_status") == "exact"
        and row.get("launch_dynamic_smem_bytes") is not None
    )


def _exact_driver_occupancy(row: dict[str, Any] | None) -> bool:
    return bool(
        row is not None
        and row.get("occupancy_status") == "exact-driver-query"
        and int(row.get("occupancy_active_ctas_per_sm", 0)) > 0
        and row.get("occupancy_gpu_uuid")
    )


def _local_memory(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    return any(
        int(row[field]) > 0
        for field in (
            "frame_bytes",
            "min_stack_bytes",
            "local_load_instructions",
            "local_store_instructions",
        )
    )


def _field_values(
    groups: dict[tuple[str, str], list[dict[str, Any]]], field: str
) -> set[str]:
    return {
        str(row.get(field, ""))
        for rows in groups.values()
        for row in rows
        if row.get(field)
    }


def _comparison_compile_environment_values(
    groups: dict[tuple[str, str], list[dict[str, Any]]], side: str
) -> set[str]:
    values: set[str] = set()
    for rows in groups.values():
        for row in rows:
            raw = row.get("compile_environment_json")
            if not raw:
                raise ValueError(f"{side} report has empty compile_environment_json")
            try:
                environment = json.loads(str(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{side} report has invalid compile_environment_json"
                ) from exc
            if not isinstance(environment, list) or any(
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
                for entry in environment
            ):
                raise ValueError(
                    f"{side} report has malformed compile_environment_json"
                )
            semantic = normalize_comparison_compile_environment(environment)
            values.add(json.dumps(semantic, separators=(",", ":")))
    return values


def _parse_cutlass_package_map(values: list[str], side: str) -> dict[str, str] | None:
    if not values:
        return None
    packages: dict[str, str] = {}
    for value in values:
        name, separator, version = value.partition("=")
        if not separator or not name or not version:
            raise ValueError(
                f"invalid {side} CUTLASS package {value!r}; expected NAME=VERSION"
            )
        if name not in _CUTLASS_PACKAGE_FIELDS:
            raise ValueError(f"unknown {side} CUTLASS package {name!r}")
        if name in packages:
            raise ValueError(f"duplicate {side} CUTLASS package {name!r}")
        packages[name] = version
    missing = sorted(set(_CUTLASS_PACKAGE_FIELDS) - set(packages))
    if missing:
        raise ValueError(
            f"exact {side} CUTLASS package map is incomplete; missing "
            + ", ".join(missing)
        )
    return packages


def _require_uniform_value(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    field: str,
    expected: str,
    side: str,
) -> None:
    values = _field_values(groups, field)
    if values != {expected}:
        raise ValueError(
            f"{side} {field} differs from expected {expected!r}: "
            f"observed={sorted(values)!r}"
        )


def _require_cutlass_package_map(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    packages: dict[str, str],
    side: str,
) -> None:
    for package, expected in packages.items():
        _require_uniform_value(groups, _CUTLASS_PACKAGE_FIELDS[package], expected, side)


def _non_cutlass_toolchains(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> set[str]:
    values: set[str] = set()
    for rows in groups.values():
        for row in rows:
            raw = row.get("toolchain_json")
            if not raw:
                raise ValueError("resource report has empty toolchain_json")
            try:
                toolchain = json.loads(str(raw))
            except json.JSONDecodeError as exc:
                raise ValueError("resource report has invalid toolchain_json") from exc
            if not isinstance(toolchain, list):
                raise ValueError("resource report toolchain_json is not a list")
            non_cutlass = []
            for entry in toolchain:
                if not isinstance(entry, list) or len(entry) < 2:
                    raise ValueError("resource report has malformed toolchain entry")
                if not str(entry[0]).startswith("cutlass_dsl"):
                    non_cutlass.append(entry)
            values.add(json.dumps(non_cutlass, sort_keys=True, separators=(",", ":")))
    return values


def _require_matching_build_environment(
    baseline: dict[tuple[str, str], list[dict[str, Any]]],
    current: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    fields = (
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "architecture",
        "ptxas_version",
    )
    for field in fields:
        baseline_values = _field_values(baseline, field)
        current_values = _field_values(current, field)
        if (
            len(baseline_values) != 1
            or len(current_values) != 1
            or baseline_values != current_values
        ):
            raise ValueError(
                f"build environment field {field} differs or is non-uniform: "
                f"baseline={sorted(baseline_values)!r} "
                f"current={sorted(current_values)!r}"
            )
    baseline_environments = _comparison_compile_environment_values(baseline, "baseline")
    current_environments = _comparison_compile_environment_values(current, "current")
    if (
        len(baseline_environments) != 1
        or len(current_environments) != 1
        or baseline_environments != current_environments
    ):
        raise ValueError(
            "comparison build environment differs or is non-uniform: "
            f"baseline={sorted(baseline_environments)!r} "
            f"current={sorted(current_environments)!r}"
        )
    baseline_toolchains = _non_cutlass_toolchains(baseline)
    current_toolchains = _non_cutlass_toolchains(current)
    if (
        len(baseline_toolchains) != 1
        or len(current_toolchains) != 1
        or baseline_toolchains != current_toolchains
    ):
        raise ValueError(
            "non-CUTLASS runtime toolchains differ or are non-uniform: "
            f"baseline={sorted(baseline_toolchains)!r} "
            f"current={sorted(current_toolchains)!r}"
        )
    baseline_ptxas_flags = {
        (identity, str(row.get("ptxas_flags", "")))
        for identity, rows in baseline.items()
        for row in rows
    }
    current_ptxas_flags = {
        (identity, str(row.get("ptxas_flags", "")))
        for identity, rows in current.items()
        for row in rows
    }
    if any(not flags for _, flags in baseline_ptxas_flags | current_ptxas_flags):
        raise ValueError("resource report has empty PTXAS flags")
    if baseline_ptxas_flags != current_ptxas_flags:
        raise ValueError(
            "PTXAS flags differ for the exact comparison-specialization set"
        )


def _require_exact_resource_contract(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    required_rows: set[tuple[str, ...]],
    required_counts: dict[str, int],
    side: str,
) -> None:
    rows = [row for grouped_rows in groups.values() for row in grouped_rows]
    observed = {
        (
            str(row["kernel_id"]),
            str(row["compile_spec_version"]),
            str(row["compile_spec_hash"]),
            str(row["compile_spec_json"]),
            str(row["semantic_key"]),
            str(row["comparison_semantic_key"]),
            str(row["target"]),
            str(row.get("compile_kwargs_json", "")),
            str(row["kernel"]),
        )
        for row in rows
    }
    identities = [
        (str(row["comparison_semantic_key"]), str(row["kernel"])) for row in rows
    ]
    semantic_objects: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    cache_semantics: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        semantic_objects[str(row["semantic_key"])].add(
            (
                str(row["cache_key"]),
                str(row["object_sha256"]),
                str(row["object_file"]),
            )
        )
        cache_semantics[str(row["cache_key"])].add(str(row["semantic_key"]))
    object_identities = {
        identity for objects in semantic_objects.values() for identity in objects
    }
    failures = {
        "missing": len(required_rows - observed),
        "unexpected": len(observed - required_rows),
        "duplicate": len(identities) - len(set(identities)),
        "multi_object": sum(len(objects) != 1 for objects in semantic_objects.values()),
        "multi_semantic_cache": sum(
            len(semantics) != 1 for semantics in cache_semantics.values()
        ),
        "object_count_delta": len(object_identities) - required_counts["object_count"],
    }
    if any(failures.values()):
        raise ValueError(f"{side} exact resource-row contract mismatch: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--require-exact-baseline-specialization-contract",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--require-exact-baseline-specialization-contract-metadata",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--require-exact-current-specialization-contract",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--require-exact-current-specialization-contract-metadata",
        type=Path,
        required=True,
    )
    parser.add_argument("--require-corpus-driver", type=Path, required=True)
    parser.add_argument("--require-shape-matrix", type=Path, required=True)
    parser.add_argument("--require-source-inventory", type=Path, required=True)
    parser.add_argument("--require-corpus-id", required=True)
    parser.add_argument("--require-corpus-version", required=True)
    parser.add_argument(
        "--fail-on-register-increase",
        action="store_true",
        help=(
            "fail on any GPR, SASS uniform-register, predicate-register, or "
            "uniform-predicate register-usage increase"
        ),
    )
    parser.add_argument("--fail-on-new-register-ceiling", action="store_true")
    parser.add_argument("--fail-on-local-memory-increase", action="store_true")
    parser.add_argument("--fail-on-cubin-shared-section-increase", action="store_true")
    parser.add_argument("--fail-on-cubin-shared-section-change", action="store_true")
    parser.add_argument("--fail-on-driver-static-shared-increase", action="store_true")
    parser.add_argument("--fail-on-driver-static-shared-change", action="store_true")
    parser.add_argument("--fail-on-launch-dynamic-smem-increase", action="store_true")
    parser.add_argument("--fail-on-launch-dynamic-smem-change", action="store_true")
    parser.add_argument("--fail-on-launch-metadata-change", action="store_true")
    parser.add_argument("--fail-on-driver-occupancy-decrease", action="store_true")
    parser.add_argument("--fail-on-unmatched", action="store_true")
    parser.add_argument(
        "--require-semantic-manifest",
        action="store_true",
        help="fail when either input contains a legacy or invalid-manifest row",
    )
    parser.add_argument(
        "--require-exact-launch-dynamic-smem",
        action="store_true",
        help="fail when either input lacks exact launch dynamic-SMEM accounting",
    )
    parser.add_argument(
        "--require-matching-package-fingerprint",
        action="store_true",
        help=(
            "require each report to contain one nonempty b12x package fingerprint "
            "and require the baseline and current fingerprints to match"
        ),
    )
    parser.add_argument(
        "--expected-baseline-package-fingerprint",
        help=(
            "end-to-end mode: require one explicit source/package fingerprint "
            "uniformly across the baseline report"
        ),
    )
    parser.add_argument(
        "--expected-current-package-fingerprint",
        help=(
            "end-to-end mode: require one explicit source/package fingerprint "
            "uniformly across the current report"
        ),
    )
    parser.add_argument(
        "--expected-baseline-cutlass-package",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help=(
            "require a baseline CUTLASS distribution version; repeat for the "
            "exact five-package map and use VERSION=missing for absence"
        ),
    )
    parser.add_argument(
        "--expected-current-cutlass-package",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help=(
            "require a current CUTLASS distribution version; repeat for the "
            "exact five-package map and use VERSION=missing for absence"
        ),
    )
    parser.add_argument(
        "--require-architecture",
        default="sm_120a",
        help="require this exact cubin target in both reports (default: sm_120a)",
    )
    parser.add_argument(
        "--require-matching-driver-occupancy",
        action="store_true",
        help=(
            "require each exact pair to have CUDA driver occupancy measured on "
            "the same GPU UUID"
        ),
    )
    parser.add_argument(
        "--require-matching-build-environment",
        action="store_true",
        help=(
            "require identical Python/Torch/CUDA runtime, compile environment, "
            "architecture, PTXAS version/flags, and non-CUTLASS toolchain"
        ),
    )
    parser.add_argument(
        "--fail-on-resource-regression",
        action="store_true",
        help=(
            "fail on any register or local-memory increase, a new 255-register "
            "kernel, any cubin shared-section or exact launch dynamic-SMEM change, "
            "a lower CUDA driver CTA-residency bound, or changed launch metadata"
        ),
    )
    args = parser.parse_args()

    end_to_end_fingerprints = (
        args.expected_baseline_package_fingerprint,
        args.expected_current_package_fingerprint,
    )
    if bool(end_to_end_fingerprints[0]) != bool(end_to_end_fingerprints[1]):
        parser.error(
            "both --expected-baseline-package-fingerprint and "
            "--expected-current-package-fingerprint are required together"
        )
    if args.require_matching_package_fingerprint and end_to_end_fingerprints[0]:
        parser.error(
            "compiler-only --require-matching-package-fingerprint conflicts with "
            "end-to-end expected side-specific fingerprints"
        )
    if not args.require_matching_package_fingerprint and not end_to_end_fingerprints[0]:
        parser.error(
            "select compiler-only --require-matching-package-fingerprint or provide "
            "explicit end-to-end baseline/current fingerprints"
        )
    try:
        baseline_required_resource_rows = _read_specialization_contract(
            args.require_exact_baseline_specialization_contract
        )
        baseline_required_contract_counts = _read_contract_metadata(
            args.require_exact_baseline_specialization_contract_metadata,
            contract_path=args.require_exact_baseline_specialization_contract,
            corpus_driver=args.require_corpus_driver,
            shape_matrix=args.require_shape_matrix,
            source_inventory=args.require_source_inventory,
            expected_corpus_id=args.require_corpus_id,
            expected_corpus_version=args.require_corpus_version,
            resource_rows=baseline_required_resource_rows,
        )
        current_required_resource_rows = _read_specialization_contract(
            args.require_exact_current_specialization_contract
        )
        current_required_contract_counts = _read_contract_metadata(
            args.require_exact_current_specialization_contract_metadata,
            contract_path=args.require_exact_current_specialization_contract,
            corpus_driver=args.require_corpus_driver,
            shape_matrix=args.require_shape_matrix,
            source_inventory=args.require_source_inventory,
            expected_corpus_id=args.require_corpus_id,
            expected_corpus_version=args.require_corpus_version,
            resource_rows=current_required_resource_rows,
        )
        baseline_cutlass_packages = _parse_cutlass_package_map(
            args.expected_baseline_cutlass_package, "baseline"
        )
        current_cutlass_packages = _parse_cutlass_package_map(
            args.expected_current_cutlass_package, "current"
        )
    except ValueError as exc:
        parser.error(str(exc))
    if bool(baseline_cutlass_packages) != bool(current_cutlass_packages):
        parser.error(
            "baseline and current exact CUTLASS package maps are both required"
        )
    if baseline_cutlass_packages is None or current_cutlass_packages is None:
        parser.error(
            "exact five-package baseline and current CUTLASS maps are required"
        )

    strict = any(
        value
        for name, value in vars(args).items()
        if name.startswith(("fail_on_", "require_", "expected_"))
    )
    try:
        baseline, baseline_count, baseline_missing = _read(args.baseline, strict=strict)
        current, current_count, current_missing = _read(args.current, strict=strict)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if baseline_count == 0 or current_count == 0:
        empty = []
        if baseline_count == 0:
            empty.append(str(args.baseline))
        if current_count == 0:
            empty.append(str(args.current))
        parser.error("resource report contains no kernel rows: " + ", ".join(empty))
    try:
        _require_exact_resource_contract(
            baseline,
            baseline_required_resource_rows,
            baseline_required_contract_counts,
            "baseline",
        )
        _require_exact_resource_contract(
            current,
            current_required_resource_rows,
            current_required_contract_counts,
            "current",
        )
        baseline_pair_rows = {
            identity: len(rows)
            for identity, rows in baseline.items()
            if identity[0] == "comparison-semantic-kernel"
        }
        current_pair_rows = {
            identity: len(rows)
            for identity, rows in current.items()
            if identity[0] == "comparison-semantic-kernel"
        }
        if baseline_pair_rows != current_pair_rows or any(
            count != 1 for count in baseline_pair_rows.values()
        ):
            raise ValueError(
                "baseline/current exact contracts do not form the same one-to-one "
                "comparison-specialization + CUDA-symbol set"
            )
    except ValueError as exc:
        parser.error(str(exc))
    if args.require_matching_package_fingerprint:
        baseline_fingerprints = _field_values(baseline, "package_fingerprint")
        current_fingerprints = _field_values(current, "package_fingerprint")
        if len(baseline_fingerprints) != 1 or len(current_fingerprints) != 1:
            parser.error(
                "each report must contain exactly one nonempty b12x package "
                "fingerprint: "
                f"baseline={sorted(baseline_fingerprints)!r} "
                f"current={sorted(current_fingerprints)!r}"
            )
        if baseline_fingerprints != current_fingerprints:
            parser.error(
                "baseline and current b12x package fingerprints differ: "
                f"baseline={next(iter(baseline_fingerprints))!r} "
                f"current={next(iter(current_fingerprints))!r}"
            )
    if end_to_end_fingerprints[0]:
        try:
            _require_uniform_value(
                baseline,
                "package_fingerprint",
                str(end_to_end_fingerprints[0]),
                "baseline",
            )
            _require_uniform_value(
                current,
                "package_fingerprint",
                str(end_to_end_fingerprints[1]),
                "current",
            )
        except ValueError as exc:
            parser.error(str(exc))
    if baseline_cutlass_packages is not None and current_cutlass_packages is not None:
        try:
            _require_cutlass_package_map(
                baseline, baseline_cutlass_packages, "baseline"
            )
            _require_cutlass_package_map(current, current_cutlass_packages, "current")
        except ValueError as exc:
            parser.error(str(exc))
    for side, groups in (("baseline", baseline), ("current", current)):
        architectures = _field_values(groups, "architecture")
        if architectures != {args.require_architecture}:
            parser.error(
                f"{side} architecture differs from {args.require_architecture!r}: "
                f"observed={sorted(architectures)!r}"
            )
    if args.require_matching_build_environment:
        try:
            _require_matching_build_environment(baseline, current)
        except ValueError as exc:
            parser.error(str(exc))
    launch_dynamic_smem_unknown_rows = sum(
        not _exact_launch_dynamic_smem(row)
        for groups in (baseline, current)
        for rows in groups.values()
        for row in rows
    )
    fieldnames = [
        "delta_report_schema",
        "baseline_resource_report_schema",
        "current_resource_report_schema",
        "family",
        "identity_kind",
        "comparison_semantic_key",
        "baseline_semantic_key",
        "current_semantic_key",
        "symbol_sha256",
        "pairing",
        "pair_ordinal",
        "baseline_identity_count",
        "current_identity_count",
        "baseline_manifest_status",
        "current_manifest_status",
        "baseline_cache_key",
        "current_cache_key",
        "baseline_object_sha256",
        "current_object_sha256",
        "baseline_target",
        "current_target",
        "baseline_kernel_id",
        "current_kernel_id",
        "baseline_compile_spec_version",
        "current_compile_spec_version",
        "baseline_compile_spec_hash",
        "current_compile_spec_hash",
        "baseline_compile_spec_json",
        "current_compile_spec_json",
        "baseline_compile_kwargs_json",
        "current_compile_kwargs_json",
        "baseline_package_fingerprint",
        "current_package_fingerprint",
        "baseline_toolchain_json",
        "current_toolchain_json",
        "baseline_compile_options_json",
        "current_compile_options_json",
        "baseline_compile_environment_json",
        "current_compile_environment_json",
        "baseline_architecture",
        "current_architecture",
        "architecture_change",
        "baseline_ptxas_version",
        "current_ptxas_version",
        "baseline_ptxas_flags",
        "current_ptxas_flags",
        "baseline_cutlass_dsl_version",
        "current_cutlass_dsl_version",
        "baseline_cutlass_dsl_libs_base_version",
        "current_cutlass_dsl_libs_base_version",
        "baseline_cutlass_dsl_libs_core_version",
        "current_cutlass_dsl_libs_core_version",
        "baseline_cutlass_dsl_libs_cu12_version",
        "current_cutlass_dsl_libs_cu12_version",
        "baseline_cutlass_dsl_libs_cu13_version",
        "current_cutlass_dsl_libs_cu13_version",
        "baseline_threads_per_cta",
        "current_threads_per_cta",
        "threads_per_cta_delta",
        "baseline_threads_x",
        "current_threads_x",
        "baseline_threads_y",
        "current_threads_y",
        "baseline_threads_z",
        "current_threads_z",
        "baseline_cubin_shared_section_bytes",
        "current_cubin_shared_section_bytes",
        "cubin_shared_section_delta",
        "cubin_shared_section_change",
        "cubin_shared_section_increase",
        "baseline_launch_dynamic_smem_status",
        "current_launch_dynamic_smem_status",
        "baseline_launch_dynamic_smem_count",
        "current_launch_dynamic_smem_count",
        "baseline_launch_dynamic_smem_values_json",
        "current_launch_dynamic_smem_values_json",
        "baseline_launch_metadata_source",
        "current_launch_metadata_source",
        "baseline_launch_metadata_reason",
        "current_launch_metadata_reason",
        "launch_dynamic_smem_comparable",
        "baseline_launch_dynamic_smem_bytes",
        "current_launch_dynamic_smem_bytes",
        "launch_dynamic_smem_delta",
        "launch_dynamic_smem_change",
        "launch_dynamic_smem_increase",
        "baseline_max_register_count",
        "current_max_register_count",
        "baseline_parameter_bytes",
        "current_parameter_bytes",
        "launch_metadata_change",
        "baseline_occupancy_status",
        "current_occupancy_status",
        "baseline_occupancy_device_ordinal",
        "current_occupancy_device_ordinal",
        "baseline_occupancy_gpu_name",
        "current_occupancy_gpu_name",
        "baseline_occupancy_gpu_uuid",
        "current_occupancy_gpu_uuid",
        "driver_occupancy_comparable",
        "baseline_occupancy_active_ctas_per_sm",
        "current_occupancy_active_ctas_per_sm",
        "occupancy_active_ctas_per_sm_delta",
        "driver_occupancy_decrease",
        "baseline_occupancy_active_threads_per_sm",
        "current_occupancy_active_threads_per_sm",
        "occupancy_active_threads_per_sm_delta",
        "baseline_driver_resource_validation_status",
        "current_driver_resource_validation_status",
        "baseline_driver_registers",
        "current_driver_registers",
        "baseline_driver_local_bytes",
        "current_driver_local_bytes",
        "baseline_driver_static_shared_bytes",
        "current_driver_static_shared_bytes",
        "driver_static_shared_bytes_delta",
        "driver_static_shared_bytes_change",
        "driver_static_shared_bytes_increase",
        "baseline_driver_max_threads_per_block",
        "current_driver_max_threads_per_block",
        "baseline_registers",
        "current_registers",
        "register_delta",
        "register_increase",
        "baseline_sass_uniform_registers_used",
        "current_sass_uniform_registers_used",
        "sass_uniform_registers_used_delta",
        "sass_uniform_registers_used_increase",
        "baseline_sass_uniform_register_span",
        "current_sass_uniform_register_span",
        "sass_uniform_register_span_delta",
        "sass_uniform_register_span_increase",
        "baseline_sass_predicate_registers_used",
        "current_sass_predicate_registers_used",
        "sass_predicate_registers_used_delta",
        "sass_predicate_registers_used_increase",
        "baseline_sass_predicate_register_span",
        "current_sass_predicate_register_span",
        "sass_predicate_register_span_delta",
        "sass_predicate_register_span_increase",
        "baseline_sass_uniform_predicate_registers_used",
        "current_sass_uniform_predicate_registers_used",
        "sass_uniform_predicate_registers_used_delta",
        "sass_uniform_predicate_registers_used_increase",
        "baseline_sass_uniform_predicate_register_span",
        "current_sass_uniform_predicate_register_span",
        "sass_uniform_predicate_register_span_delta",
        "sass_uniform_predicate_register_span_increase",
        "any_register_usage_increase",
        "new_register_ceiling",
        "baseline_frame_bytes",
        "current_frame_bytes",
        "frame_delta",
        "baseline_stack_bytes",
        "current_stack_bytes",
        "stack_delta",
        "baseline_local_loads",
        "current_local_loads",
        "local_load_delta",
        "baseline_local_stores",
        "current_local_stores",
        "local_store_delta",
        "baseline_local_memory",
        "current_local_memory",
        "new_local_memory",
        "local_memory_increase",
        "baseline_sass_instructions",
        "current_sass_instructions",
        "sass_instruction_delta",
        "baseline_code_bytes",
        "current_code_bytes",
        "code_bytes_delta",
        "baseline_kernel",
        "current_kernel",
    ]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open("w", newline="") if args.output else sys.stdout
    matched = 0
    unmatched = 0
    register_increases = 0
    uniform_register_usage_increases = 0
    predicate_register_usage_increases = 0
    uniform_predicate_register_usage_increases = 0
    any_register_usage_increases = 0
    new_register_ceilings = 0
    local_memory_increases = 0
    cubin_shared_section_changes = 0
    cubin_shared_section_increases = 0
    launch_dynamic_smem_changes = 0
    launch_dynamic_smem_increases = 0
    launch_dynamic_smem_unknown = 0
    launch_metadata_changes = 0
    driver_occupancy_decreases = 0
    driver_occupancy_unknown_or_mismatched = 0
    driver_static_shared_changes = 0
    driver_static_shared_increases = 0
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for identity in sorted(set(baseline) | set(current)):
            identity_kind, identity_value = identity
            old_rows = baseline.get(identity, [])
            new_rows = current.get(identity, [])
            count = max(len(old_rows), len(new_rows))
            if identity_kind == "comparison-semantic-kernel":
                base_pairing = (
                    "exact-comparison-semantic-kernel"
                    if count == 1
                    else "duplicate-comparison-semantic-kernel"
                )
            else:
                base_pairing = (
                    "legacy-exact-symbol" if count == 1 else "legacy-symbol-multiset"
                )
            for index in range(count):
                old = old_rows[index] if index < len(old_rows) else None
                new = new_rows[index] if index < len(new_rows) else None
                paired = old is not None and new is not None
                if paired:
                    matched += 1
                else:
                    unmatched += 1

                register_increase = bool(
                    paired and int(new["registers"]) > int(old["registers"])
                )
                uniform_registers_used_increase = bool(
                    paired
                    and int(new["sass_uniform_registers_used"])
                    > int(old["sass_uniform_registers_used"])
                )
                uniform_register_span_increase = bool(
                    paired
                    and int(new["sass_uniform_register_span"])
                    > int(old["sass_uniform_register_span"])
                )
                predicate_registers_used_increase = bool(
                    paired
                    and int(new["sass_predicate_registers_used"])
                    > int(old["sass_predicate_registers_used"])
                )
                predicate_register_span_increase = bool(
                    paired
                    and int(new["sass_predicate_register_span"])
                    > int(old["sass_predicate_register_span"])
                )
                uniform_predicate_registers_used_increase = bool(
                    paired
                    and int(new["sass_uniform_predicate_registers_used"])
                    > int(old["sass_uniform_predicate_registers_used"])
                )
                uniform_predicate_register_span_increase = bool(
                    paired
                    and int(new["sass_uniform_predicate_register_span"])
                    > int(old["sass_uniform_predicate_register_span"])
                )
                uniform_register_usage_increase = bool(
                    uniform_registers_used_increase or uniform_register_span_increase
                )
                predicate_register_usage_increase = bool(
                    predicate_registers_used_increase
                    or predicate_register_span_increase
                )
                uniform_predicate_register_usage_increase = bool(
                    uniform_predicate_registers_used_increase
                    or uniform_predicate_register_span_increase
                )
                any_register_usage_increase = bool(
                    register_increase
                    or uniform_register_usage_increase
                    or predicate_register_usage_increase
                    or uniform_predicate_register_usage_increase
                )
                new_register_ceiling = bool(
                    paired
                    and int(new["registers"]) >= 255
                    and int(old["registers"]) < 255
                )
                local_memory_increase = bool(
                    paired
                    and any(
                        int(new[field]) > int(old[field])
                        for field in (
                            "frame_bytes",
                            "min_stack_bytes",
                            "local_load_instructions",
                            "local_store_instructions",
                        )
                    )
                )
                cubin_shared_section_increase = bool(
                    paired
                    and int(new["cubin_shared_section_bytes"])
                    > int(old["cubin_shared_section_bytes"])
                )
                cubin_shared_section_change = bool(
                    paired
                    and int(new["cubin_shared_section_bytes"])
                    != int(old["cubin_shared_section_bytes"])
                )
                launch_dynamic_smem_comparable = bool(
                    paired
                    and _exact_launch_dynamic_smem(old)
                    and _exact_launch_dynamic_smem(new)
                )
                launch_dynamic_smem_delta = (
                    int(new["launch_dynamic_smem_bytes"])
                    - int(old["launch_dynamic_smem_bytes"])
                    if launch_dynamic_smem_comparable
                    else ""
                )
                launch_dynamic_smem_increase = bool(
                    launch_dynamic_smem_comparable
                    and int(new["launch_dynamic_smem_bytes"])
                    > int(old["launch_dynamic_smem_bytes"])
                )
                launch_dynamic_smem_change = bool(
                    launch_dynamic_smem_comparable
                    and int(new["launch_dynamic_smem_bytes"])
                    != int(old["launch_dynamic_smem_bytes"])
                )
                launch_metadata_change = bool(
                    paired
                    and (
                        any(
                            new.get(field, "") != old.get(field, "")
                            for field in (
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
                        )
                    )
                )
                driver_occupancy_comparable = bool(
                    paired
                    and _exact_driver_occupancy(old)
                    and _exact_driver_occupancy(new)
                    and old.get("occupancy_gpu_uuid") == new.get("occupancy_gpu_uuid")
                )
                driver_occupancy_decrease = bool(
                    driver_occupancy_comparable
                    and int(new["occupancy_active_ctas_per_sm"])
                    < int(old["occupancy_active_ctas_per_sm"])
                )
                driver_static_shared_comparable = bool(
                    paired
                    and old.get("driver_static_shared_bytes") is not None
                    and new.get("driver_static_shared_bytes") is not None
                )
                driver_static_shared_delta = (
                    int(new["driver_static_shared_bytes"])
                    - int(old["driver_static_shared_bytes"])
                    if driver_static_shared_comparable
                    else ""
                )
                driver_static_shared_change = bool(
                    driver_static_shared_comparable and driver_static_shared_delta != 0
                )
                driver_static_shared_increase = bool(
                    driver_static_shared_comparable and driver_static_shared_delta > 0
                )
                register_increases += register_increase
                uniform_register_usage_increases += uniform_register_usage_increase
                predicate_register_usage_increases += predicate_register_usage_increase
                uniform_predicate_register_usage_increases += (
                    uniform_predicate_register_usage_increase
                )
                any_register_usage_increases += any_register_usage_increase
                new_register_ceilings += new_register_ceiling
                local_memory_increases += local_memory_increase
                cubin_shared_section_changes += cubin_shared_section_change
                cubin_shared_section_increases += cubin_shared_section_increase
                launch_dynamic_smem_changes += launch_dynamic_smem_change
                launch_dynamic_smem_increases += launch_dynamic_smem_increase
                launch_dynamic_smem_unknown += (
                    paired and not launch_dynamic_smem_comparable
                )
                launch_metadata_changes += launch_metadata_change
                driver_occupancy_decreases += driver_occupancy_decrease
                driver_occupancy_unknown_or_mismatched += (
                    paired and not driver_occupancy_comparable
                )
                driver_static_shared_changes += driver_static_shared_change
                driver_static_shared_increases += driver_static_shared_increase

                old_kernel = str(_value(old, "kernel"))
                new_kernel = str(_value(new, "kernel"))
                hash_kernel = (
                    old_kernel
                    if old_kernel == new_kernel
                    else f"{old_kernel}\0{new_kernel}"
                )
                writer.writerow(
                    {
                        "delta_report_schema": _DELTA_REPORT_SCHEMA,
                        "baseline_resource_report_schema": _value(
                            old, "resource_report_schema"
                        ),
                        "current_resource_report_schema": _value(
                            new, "resource_report_schema"
                        ),
                        "family": _family(new or old),
                        "identity_kind": identity_kind,
                        "comparison_semantic_key": identity_value.partition("\0")[0]
                        if identity_kind == "comparison-semantic-kernel"
                        else "",
                        "baseline_semantic_key": _value(old, "semantic_key"),
                        "current_semantic_key": _value(new, "semantic_key"),
                        "symbol_sha256": hashlib.sha256(
                            hash_kernel.encode()
                        ).hexdigest(),
                        "pairing": base_pairing
                        if paired
                        else f"unmatched-{identity_kind}",
                        "pair_ordinal": index,
                        "baseline_identity_count": len(old_rows),
                        "current_identity_count": len(new_rows),
                        "baseline_manifest_status": _value(old, "manifest_status"),
                        "current_manifest_status": _value(new, "manifest_status"),
                        "baseline_cache_key": _value(old, "cache_key"),
                        "current_cache_key": _value(new, "cache_key"),
                        "baseline_object_sha256": _value(old, "object_sha256"),
                        "current_object_sha256": _value(new, "object_sha256"),
                        "baseline_target": _value(old, "target"),
                        "current_target": _value(new, "target"),
                        "baseline_kernel_id": _value(old, "kernel_id"),
                        "current_kernel_id": _value(new, "kernel_id"),
                        "baseline_compile_spec_version": _value(
                            old, "compile_spec_version"
                        ),
                        "current_compile_spec_version": _value(
                            new, "compile_spec_version"
                        ),
                        "baseline_compile_spec_hash": _value(old, "compile_spec_hash"),
                        "current_compile_spec_hash": _value(new, "compile_spec_hash"),
                        "baseline_compile_spec_json": _value(old, "compile_spec_json"),
                        "current_compile_spec_json": _value(new, "compile_spec_json"),
                        "baseline_compile_kwargs_json": _value(
                            old, "compile_kwargs_json"
                        ),
                        "current_compile_kwargs_json": _value(
                            new, "compile_kwargs_json"
                        ),
                        "baseline_package_fingerprint": _value(
                            old, "package_fingerprint"
                        ),
                        "current_package_fingerprint": _value(
                            new, "package_fingerprint"
                        ),
                        "baseline_toolchain_json": _value(old, "toolchain_json"),
                        "current_toolchain_json": _value(new, "toolchain_json"),
                        "baseline_compile_options_json": _value(
                            old, "compile_options_json"
                        ),
                        "current_compile_options_json": _value(
                            new, "compile_options_json"
                        ),
                        "baseline_compile_environment_json": _value(
                            old, "compile_environment_json"
                        ),
                        "current_compile_environment_json": _value(
                            new, "compile_environment_json"
                        ),
                        "baseline_architecture": _value(old, "architecture"),
                        "current_architecture": _value(new, "architecture"),
                        "architecture_change": _bool(
                            bool(paired and new["architecture"] != old["architecture"])
                        ),
                        "baseline_ptxas_version": _value(old, "ptxas_version"),
                        "current_ptxas_version": _value(new, "ptxas_version"),
                        "baseline_ptxas_flags": _value(old, "ptxas_flags"),
                        "current_ptxas_flags": _value(new, "ptxas_flags"),
                        "baseline_cutlass_dsl_version": _value(
                            old, "cutlass_dsl_version"
                        ),
                        "current_cutlass_dsl_version": _value(
                            new, "cutlass_dsl_version"
                        ),
                        "baseline_cutlass_dsl_libs_base_version": _value(
                            old, "cutlass_dsl_libs_base_version"
                        ),
                        "current_cutlass_dsl_libs_base_version": _value(
                            new, "cutlass_dsl_libs_base_version"
                        ),
                        "baseline_cutlass_dsl_libs_core_version": _value(
                            old, "cutlass_dsl_libs_core_version"
                        ),
                        "current_cutlass_dsl_libs_core_version": _value(
                            new, "cutlass_dsl_libs_core_version"
                        ),
                        "baseline_cutlass_dsl_libs_cu12_version": _value(
                            old, "cutlass_dsl_libs_cu12_version"
                        ),
                        "current_cutlass_dsl_libs_cu12_version": _value(
                            new, "cutlass_dsl_libs_cu12_version"
                        ),
                        "baseline_cutlass_dsl_libs_cu13_version": _value(
                            old, "cutlass_dsl_libs_cu13_version"
                        ),
                        "current_cutlass_dsl_libs_cu13_version": _value(
                            new, "cutlass_dsl_libs_cu13_version"
                        ),
                        "baseline_threads_per_cta": _number(old, "threads_per_cta"),
                        "current_threads_per_cta": _number(new, "threads_per_cta"),
                        "threads_per_cta_delta": _delta(old, new, "threads_per_cta"),
                        "baseline_threads_x": _number(old, "threads_x"),
                        "current_threads_x": _number(new, "threads_x"),
                        "baseline_threads_y": _number(old, "threads_y"),
                        "current_threads_y": _number(new, "threads_y"),
                        "baseline_threads_z": _number(old, "threads_z"),
                        "current_threads_z": _number(new, "threads_z"),
                        "baseline_cubin_shared_section_bytes": _number(
                            old, "cubin_shared_section_bytes"
                        ),
                        "current_cubin_shared_section_bytes": _number(
                            new, "cubin_shared_section_bytes"
                        ),
                        "cubin_shared_section_delta": _delta(
                            old, new, "cubin_shared_section_bytes"
                        ),
                        "cubin_shared_section_change": _bool(
                            cubin_shared_section_change
                        ),
                        "cubin_shared_section_increase": _bool(
                            cubin_shared_section_increase
                        ),
                        "baseline_launch_dynamic_smem_status": _value(
                            old, "launch_dynamic_smem_status"
                        ),
                        "current_launch_dynamic_smem_status": _value(
                            new, "launch_dynamic_smem_status"
                        ),
                        "baseline_launch_dynamic_smem_count": _number(
                            old, "launch_dynamic_smem_count"
                        ),
                        "current_launch_dynamic_smem_count": _number(
                            new, "launch_dynamic_smem_count"
                        ),
                        "baseline_launch_dynamic_smem_values_json": _value(
                            old, "launch_dynamic_smem_values_json"
                        ),
                        "current_launch_dynamic_smem_values_json": _value(
                            new, "launch_dynamic_smem_values_json"
                        ),
                        "baseline_launch_metadata_source": _value(
                            old, "launch_metadata_source"
                        ),
                        "current_launch_metadata_source": _value(
                            new, "launch_metadata_source"
                        ),
                        "baseline_launch_metadata_reason": _value(
                            old, "launch_metadata_reason"
                        ),
                        "current_launch_metadata_reason": _value(
                            new, "launch_metadata_reason"
                        ),
                        "launch_dynamic_smem_comparable": _bool(
                            launch_dynamic_smem_comparable
                        ),
                        "baseline_launch_dynamic_smem_bytes": _value(
                            old, "launch_dynamic_smem_bytes"
                        ),
                        "current_launch_dynamic_smem_bytes": _value(
                            new, "launch_dynamic_smem_bytes"
                        ),
                        "launch_dynamic_smem_delta": launch_dynamic_smem_delta,
                        "launch_dynamic_smem_change": _bool(launch_dynamic_smem_change),
                        "launch_dynamic_smem_increase": _bool(
                            launch_dynamic_smem_increase
                        ),
                        "baseline_max_register_count": _number(
                            old, "max_register_count"
                        ),
                        "current_max_register_count": _number(
                            new, "max_register_count"
                        ),
                        "baseline_parameter_bytes": _number(old, "parameter_bytes"),
                        "current_parameter_bytes": _number(new, "parameter_bytes"),
                        "launch_metadata_change": _bool(launch_metadata_change),
                        "baseline_occupancy_status": _value(old, "occupancy_status"),
                        "current_occupancy_status": _value(new, "occupancy_status"),
                        "baseline_occupancy_device_ordinal": _number(
                            old, "occupancy_device_ordinal"
                        ),
                        "current_occupancy_device_ordinal": _number(
                            new, "occupancy_device_ordinal"
                        ),
                        "baseline_occupancy_gpu_name": _value(
                            old, "occupancy_gpu_name"
                        ),
                        "current_occupancy_gpu_name": _value(new, "occupancy_gpu_name"),
                        "baseline_occupancy_gpu_uuid": _value(
                            old, "occupancy_gpu_uuid"
                        ),
                        "current_occupancy_gpu_uuid": _value(new, "occupancy_gpu_uuid"),
                        "driver_occupancy_comparable": _bool(
                            driver_occupancy_comparable
                        ),
                        "baseline_occupancy_active_ctas_per_sm": _number(
                            old, "occupancy_active_ctas_per_sm"
                        ),
                        "current_occupancy_active_ctas_per_sm": _number(
                            new, "occupancy_active_ctas_per_sm"
                        ),
                        "occupancy_active_ctas_per_sm_delta": _delta(
                            old, new, "occupancy_active_ctas_per_sm"
                        ),
                        "driver_occupancy_decrease": _bool(driver_occupancy_decrease),
                        "baseline_occupancy_active_threads_per_sm": _number(
                            old, "occupancy_active_threads_per_sm"
                        ),
                        "current_occupancy_active_threads_per_sm": _number(
                            new, "occupancy_active_threads_per_sm"
                        ),
                        "occupancy_active_threads_per_sm_delta": _delta(
                            old, new, "occupancy_active_threads_per_sm"
                        ),
                        "baseline_driver_resource_validation_status": _value(
                            old, "driver_resource_validation_status"
                        ),
                        "current_driver_resource_validation_status": _value(
                            new, "driver_resource_validation_status"
                        ),
                        "baseline_driver_registers": _number(old, "driver_registers"),
                        "current_driver_registers": _number(new, "driver_registers"),
                        "baseline_driver_local_bytes": _number(
                            old, "driver_local_bytes"
                        ),
                        "current_driver_local_bytes": _number(
                            new, "driver_local_bytes"
                        ),
                        "baseline_driver_static_shared_bytes": _number(
                            old, "driver_static_shared_bytes"
                        ),
                        "current_driver_static_shared_bytes": _number(
                            new, "driver_static_shared_bytes"
                        ),
                        "driver_static_shared_bytes_delta": (
                            driver_static_shared_delta
                        ),
                        "driver_static_shared_bytes_change": _bool(
                            driver_static_shared_change
                        ),
                        "driver_static_shared_bytes_increase": _bool(
                            driver_static_shared_increase
                        ),
                        "baseline_driver_max_threads_per_block": _number(
                            old, "driver_max_threads_per_block"
                        ),
                        "current_driver_max_threads_per_block": _number(
                            new, "driver_max_threads_per_block"
                        ),
                        "baseline_registers": _number(old, "registers"),
                        "current_registers": _number(new, "registers"),
                        "register_delta": _delta(old, new, "registers"),
                        "register_increase": _bool(register_increase),
                        "baseline_sass_uniform_registers_used": _number(
                            old, "sass_uniform_registers_used"
                        ),
                        "current_sass_uniform_registers_used": _number(
                            new, "sass_uniform_registers_used"
                        ),
                        "sass_uniform_registers_used_delta": _delta(
                            old, new, "sass_uniform_registers_used"
                        ),
                        "sass_uniform_registers_used_increase": _bool(
                            uniform_registers_used_increase
                        ),
                        "baseline_sass_uniform_register_span": _number(
                            old, "sass_uniform_register_span"
                        ),
                        "current_sass_uniform_register_span": _number(
                            new, "sass_uniform_register_span"
                        ),
                        "sass_uniform_register_span_delta": _delta(
                            old, new, "sass_uniform_register_span"
                        ),
                        "sass_uniform_register_span_increase": _bool(
                            uniform_register_span_increase
                        ),
                        "baseline_sass_predicate_registers_used": _number(
                            old, "sass_predicate_registers_used"
                        ),
                        "current_sass_predicate_registers_used": _number(
                            new, "sass_predicate_registers_used"
                        ),
                        "sass_predicate_registers_used_delta": _delta(
                            old, new, "sass_predicate_registers_used"
                        ),
                        "sass_predicate_registers_used_increase": _bool(
                            predicate_registers_used_increase
                        ),
                        "baseline_sass_predicate_register_span": _number(
                            old, "sass_predicate_register_span"
                        ),
                        "current_sass_predicate_register_span": _number(
                            new, "sass_predicate_register_span"
                        ),
                        "sass_predicate_register_span_delta": _delta(
                            old, new, "sass_predicate_register_span"
                        ),
                        "sass_predicate_register_span_increase": _bool(
                            predicate_register_span_increase
                        ),
                        "baseline_sass_uniform_predicate_registers_used": _number(
                            old, "sass_uniform_predicate_registers_used"
                        ),
                        "current_sass_uniform_predicate_registers_used": _number(
                            new, "sass_uniform_predicate_registers_used"
                        ),
                        "sass_uniform_predicate_registers_used_delta": _delta(
                            old, new, "sass_uniform_predicate_registers_used"
                        ),
                        "sass_uniform_predicate_registers_used_increase": _bool(
                            uniform_predicate_registers_used_increase
                        ),
                        "baseline_sass_uniform_predicate_register_span": _number(
                            old, "sass_uniform_predicate_register_span"
                        ),
                        "current_sass_uniform_predicate_register_span": _number(
                            new, "sass_uniform_predicate_register_span"
                        ),
                        "sass_uniform_predicate_register_span_delta": _delta(
                            old, new, "sass_uniform_predicate_register_span"
                        ),
                        "sass_uniform_predicate_register_span_increase": _bool(
                            uniform_predicate_register_span_increase
                        ),
                        "any_register_usage_increase": _bool(
                            any_register_usage_increase
                        ),
                        "new_register_ceiling": _bool(new_register_ceiling),
                        "baseline_frame_bytes": _number(old, "frame_bytes"),
                        "current_frame_bytes": _number(new, "frame_bytes"),
                        "frame_delta": _delta(old, new, "frame_bytes"),
                        "baseline_stack_bytes": _number(old, "min_stack_bytes"),
                        "current_stack_bytes": _number(new, "min_stack_bytes"),
                        "stack_delta": _delta(old, new, "min_stack_bytes"),
                        "baseline_local_loads": _number(old, "local_load_instructions"),
                        "current_local_loads": _number(new, "local_load_instructions"),
                        "local_load_delta": _delta(old, new, "local_load_instructions"),
                        "baseline_local_stores": _number(
                            old, "local_store_instructions"
                        ),
                        "current_local_stores": _number(
                            new, "local_store_instructions"
                        ),
                        "local_store_delta": _delta(
                            old, new, "local_store_instructions"
                        ),
                        "baseline_local_memory": _bool(_local_memory(old)),
                        "current_local_memory": _bool(_local_memory(new)),
                        "new_local_memory": _bool(
                            paired and _local_memory(new) and not _local_memory(old)
                        ),
                        "local_memory_increase": _bool(local_memory_increase),
                        "baseline_sass_instructions": _number(old, "sass_instructions"),
                        "current_sass_instructions": _number(new, "sass_instructions"),
                        "sass_instruction_delta": _delta(old, new, "sass_instructions"),
                        "baseline_code_bytes": _number(old, "code_bytes"),
                        "current_code_bytes": _number(new, "code_bytes"),
                        "code_bytes_delta": _delta(old, new, "code_bytes"),
                        "baseline_kernel": old_kernel,
                        "current_kernel": new_kernel,
                    }
                )
    finally:
        if args.output:
            output.close()

    print(
        f"baseline_rows={baseline_count} current_rows={current_count} "
        f"matched={matched} unmatched={unmatched} "
        f"missing_manifests={baseline_missing + current_missing} "
        f"register_increases={register_increases} "
        f"uniform_register_usage_increases={uniform_register_usage_increases} "
        f"predicate_register_usage_increases={predicate_register_usage_increases} "
        f"uniform_predicate_register_usage_increases="
        f"{uniform_predicate_register_usage_increases} "
        f"any_register_usage_increases={any_register_usage_increases} "
        f"new_register_ceilings={new_register_ceilings} "
        f"local_memory_increases={local_memory_increases} "
        f"cubin_shared_section_changes={cubin_shared_section_changes} "
        f"cubin_shared_section_increases={cubin_shared_section_increases} "
        f"launch_dynamic_smem_changes={launch_dynamic_smem_changes} "
        f"launch_dynamic_smem_increases={launch_dynamic_smem_increases} "
        f"launch_dynamic_smem_unknown={launch_dynamic_smem_unknown} "
        f"launch_dynamic_smem_unknown_rows={launch_dynamic_smem_unknown_rows} "
        f"launch_metadata_changes={launch_metadata_changes} "
        f"driver_occupancy_decreases={driver_occupancy_decreases} "
        f"driver_occupancy_unknown_or_mismatched="
        f"{driver_occupancy_unknown_or_mismatched} "
        f"driver_static_shared_changes={driver_static_shared_changes} "
        f"driver_static_shared_increases={driver_static_shared_increases}",
        file=sys.stderr,
    )

    resource_regression = any(
        (
            any_register_usage_increases,
            new_register_ceilings,
            local_memory_increases,
            cubin_shared_section_changes,
            launch_dynamic_smem_changes,
            driver_static_shared_changes,
            launch_metadata_changes,
            driver_occupancy_decreases,
        )
    )
    fail = (
        (args.fail_on_register_increase and any_register_usage_increases > 0)
        or (args.fail_on_new_register_ceiling and new_register_ceilings > 0)
        or (args.fail_on_local_memory_increase and local_memory_increases > 0)
        or (
            args.fail_on_cubin_shared_section_increase
            and cubin_shared_section_increases > 0
        )
        or (
            args.fail_on_cubin_shared_section_change
            and cubin_shared_section_changes > 0
        )
        or (
            args.fail_on_launch_dynamic_smem_increase
            and launch_dynamic_smem_increases > 0
        )
        or (args.fail_on_launch_dynamic_smem_change and launch_dynamic_smem_changes > 0)
        or (args.fail_on_launch_metadata_change and launch_metadata_changes > 0)
        or (
            args.fail_on_driver_static_shared_increase
            and driver_static_shared_increases > 0
        )
        or (
            args.fail_on_driver_static_shared_change
            and driver_static_shared_changes > 0
        )
        or (args.fail_on_driver_occupancy_decrease and driver_occupancy_decreases > 0)
        or (args.fail_on_unmatched and unmatched > 0)
        or (args.require_semantic_manifest and baseline_missing + current_missing > 0)
        or (
            args.require_exact_launch_dynamic_smem
            and launch_dynamic_smem_unknown_rows > 0
        )
        or (
            args.require_matching_driver_occupancy
            and driver_occupancy_unknown_or_mismatched > 0
        )
        or (args.fail_on_resource_regression and resource_regression)
    )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
