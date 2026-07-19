#!/usr/bin/env python3
"""Merge strict CuTe resource CSVs without hiding conflicting duplicates."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from validation.cutlass_migration.evidence.compare_resources import (
    _RESOURCE_REPORT_FIELDS,
    _RESOURCE_REPORT_SCHEMA,
    _read as _read_strict_resource_report,
)
from validation.cutlass_migration.evidence.kernel_resources import (
    _read_contract_metadata,
    _read_specialization_contract,
)


def _row_identity(row: dict[str, str]) -> tuple[str, str]:
    comparison_key = row.get("comparison_semantic_key", "")
    kernel = row.get("kernel", "")
    if row.get("manifest_status") != "ok" or not comparison_key or not kernel:
        raise ValueError(
            "all merged rows require a valid semantic manifest and CUDA symbol"
        )
    return comparison_key, kernel


def _comparable_row(row: dict[str, str]) -> dict[str, str]:
    # The same cache object can be audited from multiple shard directories.
    # Its absolute path is not semantic; every other field must agree exactly.
    return {key: value for key, value in row.items() if key != "object_file"}


def _contract_row(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["kernel_id"],
        row["compile_spec_version"],
        row["compile_spec_hash"],
        row["compile_spec_json"],
        row["semantic_key"],
        row["comparison_semantic_key"],
        row["target"],
        row["compile_kwargs_json"],
        row["kernel"],
    )


def _require_one_value(rows: dict[tuple[str, str], dict[str, str]], field: str) -> str:
    values = {row.get(field, "") for row in rows.values()}
    if "" in values:
        raise ValueError(f"merged rows have an empty {field}")
    if len(values) != 1:
        rendered = ", ".join(repr(value) for value in sorted(values))
        raise ValueError(f"merged rows mix {field} values: {rendered}")
    return next(iter(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--require-exact-specialization-contract",
        type=Path,
        required=True,
        metavar="PATH",
        help="require the merged exact resource-row set to equal PATH",
    )
    parser.add_argument(
        "--require-exact-specialization-contract-metadata",
        type=Path,
        required=True,
    )
    parser.add_argument("--require-corpus-driver", type=Path, required=True)
    parser.add_argument("--require-shape-matrix", type=Path, required=True)
    parser.add_argument("--require-source-inventory", type=Path, required=True)
    parser.add_argument("--require-corpus-id", required=True)
    parser.add_argument("--require-corpus-version", required=True)
    args = parser.parse_args()

    fieldnames: list[str] | None = None
    merged: dict[tuple[str, str], dict[str, str]] = {}
    input_rows = 0
    duplicate_rows = 0
    try:
        required_resource_rows = _read_specialization_contract(
            args.require_exact_specialization_contract
        )
        required_contract_counts = _read_contract_metadata(
            args.require_exact_specialization_contract_metadata,
            contract_path=args.require_exact_specialization_contract,
            corpus_driver=args.require_corpus_driver,
            shape_matrix=args.require_shape_matrix,
            source_inventory=args.require_source_inventory,
            expected_corpus_id=args.require_corpus_id,
            expected_corpus_version=args.require_corpus_version,
            resource_rows=required_resource_rows,
        )
        for path in args.reports:
            _read_strict_resource_report(path, strict=True)
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                current_fields = list(reader.fieldnames or ())
                if tuple(current_fields) != _RESOURCE_REPORT_FIELDS:
                    raise ValueError(
                        f"{path}: expected exact {_RESOURCE_REPORT_SCHEMA} columns"
                    )
                if fieldnames is None:
                    fieldnames = current_fields
                elif current_fields != fieldnames:
                    raise ValueError(f"{path}: resource report columns differ")
                rows_in_report = 0
                for row in reader:
                    rows_in_report += 1
                    input_rows += 1
                    identity = _row_identity(row)
                    if row.get("resource_report_schema") != _RESOURCE_REPORT_SCHEMA:
                        raise ValueError(f"{path}: invalid resource report schema")
                    if row.get("launch_dynamic_smem_status") != "exact":
                        raise ValueError(f"{path}: launch dynamic SMEM is not exact")
                    if row.get("occupancy_status") != "exact-driver-query":
                        raise ValueError(f"{path}: driver occupancy is not exact")
                    if row.get("driver_resource_validation_status") != "exact-match":
                        raise ValueError(
                            f"{path}: driver resource validation is not exact"
                        )
                    previous = merged.get(identity)
                    if previous is None:
                        merged[identity] = row
                    elif _comparable_row(previous) == _comparable_row(row):
                        duplicate_rows += 1
                    else:
                        raise ValueError(
                            f"{path}: conflicting duplicate semantic kernel "
                            f"{identity[0]} {identity[1]}"
                        )
                if rows_in_report == 0:
                    raise ValueError(f"{path}: resource report has no kernel rows")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if fieldnames is None or not merged:
        parser.error("no resource rows were read")

    observed_resource_rows = {_contract_row(row) for row in merged.values()}
    missing_resource_rows = required_resource_rows - observed_resource_rows
    unexpected_resource_rows = observed_resource_rows - required_resource_rows
    semantic_objects: dict[str, set[tuple[str, str, str]]] = {}
    cache_semantics: dict[str, set[str]] = {}
    for row in merged.values():
        semantic_objects.setdefault(row["semantic_key"], set()).add(
            (row["cache_key"], row["object_sha256"], row["object_file"])
        )
        cache_semantics.setdefault(row["cache_key"], set()).add(row["semantic_key"])
    multi_object_semantics = {
        key: values for key, values in semantic_objects.items() if len(values) != 1
    }
    multi_semantic_caches = {
        key: values for key, values in cache_semantics.items() if len(values) != 1
    }
    object_identities = {
        identity for objects in semantic_objects.values() for identity in objects
    }
    object_count_mismatch = (
        len(object_identities) != required_contract_counts["object_count"]
    )
    if (
        missing_resource_rows
        or unexpected_resource_rows
        or multi_object_semantics
        or multi_semantic_caches
        or object_count_mismatch
    ):
        parser.error(
            "exact resource-row contract mismatch: "
            f"missing={len(missing_resource_rows)} "
            f"unexpected={len(unexpected_resource_rows)} "
            f"multi_object={len(multi_object_semantics)} "
            f"multi_semantic_cache={len(multi_semantic_caches)} "
            f"objects={len(object_identities)}/"
            f"{required_contract_counts['object_count']}"
        )

    # A final corpus is evidence for one source snapshot under one toolchain.
    # Without these gates, independently valid shards compiled before and after
    # a source fix (or with different transitive CUTLASS wheels) could be merged
    # into a report that never existed as one reproducible build.
    try:
        package_fingerprint = _require_one_value(merged, "package_fingerprint")
        toolchain_fields = (
            "toolchain_json",
            "compile_options_json",
            "compile_environment_json",
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "cutlass_dsl_version",
            "cutlass_dsl_libs_base_version",
            "cutlass_dsl_libs_core_version",
            "cutlass_dsl_libs_cu12_version",
            "cutlass_dsl_libs_cu13_version",
            "architecture",
            "ptxas_version",
            "ptxas_flags",
            "occupancy_device_ordinal",
            "occupancy_gpu_name",
            "occupancy_gpu_uuid",
        )
        toolchain = {
            field: _require_one_value(merged, field) for field in toolchain_fields
        }
    except ValueError as exc:
        parser.error(str(exc))

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
        for identity in sorted(merged):
            writer.writerow(merged[identity])
    finally:
        if args.output:
            output.close()

    print(
        f"reports={len(args.reports)} input_rows={input_rows} "
        f"merged_rows={len(merged)} duplicate_rows={duplicate_rows} "
        f"package_fingerprint={package_fingerprint} "
        f"toolchain={toolchain}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
