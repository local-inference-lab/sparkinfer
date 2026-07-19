#!/usr/bin/env python3
"""Emit exact SASS register sets for every row in a CuTe resource report.

``audit_cute_kernel_resources.py`` remains the stable, backward-compatible
resource report.  This companion sidecar re-opens those exact cache objects,
disassembles the same CUDA entry points, and records the physical register
indices named by SASS for all four register files: R, UR, P, and UP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from validation.cutlass_migration.evidence.kernel_resources import (
    _CUDA_ELF_MAGIC,
    _RESOURCE_REPORT_SCHEMA,
    _TEXT_KERNEL_SECTION_RE,
    _attribute_blocks,
    _kernel_code,
    _kernel_info,
    _resource_values,
)


_SIDECAR_SCHEMA = "b12x.cute.sass_register_sets.v3"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INSTRUCTION_LINE_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/.*;\s*$",
    re.MULTILINE,
)
_REGISTER_PATTERNS = {
    # The negative lookbehind prevents R/P from being counted inside UR/UP.
    "r": re.compile(r"(?<![A-Z0-9_])R(?P<index>[0-9]+)\b"),
    "ur": re.compile(r"(?<![A-Z0-9_])UR(?P<index>[0-9]+)\b"),
    "p": re.compile(r"(?<![A-Z0-9_])P(?P<index>[0-9]+)\b"),
    "up": re.compile(r"(?<![A-Z0-9_])UP(?P<index>[0-9]+)\b"),
}
_SETMAXREG_RE = re.compile(
    r"\bUSETMAXREG\.(?P<operation>[A-Z_]+)\.CTAPOOL\s+"
    r"(?:(?:UP(?:T|[0-9]+)|P(?:T|[0-9]+)),\s*)?"
    r"(?P<target>0x[0-9a-fA-F]+|[0-9]+)\b"
)
_IDENTITY_FIELDS = (
    "resource_report_schema",
    "object_file",
    "object_sha256",
    "cache_key",
    "manifest_status",
    "semantic_key",
    "comparison_semantic_key",
    "target",
    "kernel_id",
    "compile_spec_version",
    "compile_spec_hash",
    "compile_spec_json",
    "package_fingerprint",
    "cutlass_dsl_version",
    "architecture",
    "kernel",
)
_RESOURCE_FIELDS = (
    "registers",
    "max_register_count",
    "frame_bytes",
    "min_stack_bytes",
    "local_load_instructions",
    "local_store_instructions",
    "cubin_shared_section_bytes",
    "driver_local_bytes",
    "driver_static_shared_bytes",
    "launch_dynamic_smem_bytes",
    "sass_uniform_registers_used",
    "sass_uniform_register_span",
    "sass_predicate_registers_used",
    "sass_predicate_register_span",
    "sass_uniform_predicate_registers_used",
    "sass_uniform_predicate_register_span",
)
_OUTPUT_FIELDS = (
    "sass_register_set_report_schema",
    *_IDENTITY_FIELDS,
    "allocated_registers",
    "eiattr_max_register_count",
    "register_reconfiguration",
    "register_reconfiguration_instruction_count",
    "register_deallocation_targets_json",
    "register_deallocation_instruction_count",
    "register_allocation_targets_json",
    "register_allocation_instruction_count",
    "register_reconfiguration_min_target",
    "register_reconfiguration_max_target",
    "effective_register_index_ceiling",
    "frame_bytes",
    "min_stack_bytes",
    "local_load_instructions",
    "local_store_instructions",
    "driver_local_bytes",
    "cubin_shared_section_bytes",
    "driver_static_shared_bytes",
    "launch_dynamic_smem_bytes",
    "total_launch_shared_bytes",
    *(
        field
        for family in _REGISTER_PATTERNS
        for field in (
            f"{family}_indices_json",
            f"{family}_count",
            f"{family}_min",
            f"{family}_max",
            f"{family}_span",
            f"{family}_index_span",
        )
    ),
)


def _integer(row: dict[str, str], field: str, *, row_number: int) -> int:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"resource row {row_number} has no {field}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"resource row {row_number} has invalid {field}={value!r}"
        ) from exc


def _read_resource_report(path: Path) -> list[dict[str, str]]:
    try:
        source = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read resource report {path}: {exc}") from exc
    with source:
        reader = csv.DictReader(source)
        expected = {*_IDENTITY_FIELDS, *_RESOURCE_FIELDS}
        missing = sorted(expected - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: resource report is missing columns {missing}")
        rows: list[dict[str, str]] = []
        identities: set[tuple[str, str]] = set()
        object_kernels: set[tuple[str, str]] = set()
        for row_number, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            if row["resource_report_schema"] != _RESOURCE_REPORT_SCHEMA:
                raise ValueError(
                    f"{path}:{row_number}: expected {_RESOURCE_REPORT_SCHEMA}, got "
                    f"{row['resource_report_schema']!r}"
                )
            if row["manifest_status"] != "ok":
                raise ValueError(
                    f"{path}:{row_number}: manifest status is "
                    f"{row['manifest_status']!r}, expected 'ok'"
                )
            for field in _IDENTITY_FIELDS:
                if not row[field]:
                    raise ValueError(f"{path}:{row_number}: empty {field}")
            for field in (
                "object_sha256",
                "cache_key",
                "semantic_key",
                "comparison_semantic_key",
                "compile_spec_hash",
            ):
                if not _SHA256_RE.fullmatch(row[field]):
                    raise ValueError(
                        f"{path}:{row_number}: invalid SHA-256 field {field}"
                    )
            for field in _RESOURCE_FIELDS:
                _integer(row, field, row_number=row_number)
            identity = (row["comparison_semantic_key"], row["kernel"])
            if identity in identities:
                raise ValueError(
                    f"{path}:{row_number}: duplicate identity {identity!r}"
                )
            identities.add(identity)
            object_kernel = (row["object_sha256"], row["kernel"])
            if object_kernel in object_kernels:
                raise ValueError(
                    f"{path}:{row_number}: duplicate object/kernel {object_kernel!r}"
                )
            object_kernels.add(object_kernel)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: resource report is empty")
    return rows


def _resolve_object_path(raw: str, report_path: Path) -> Path:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(report_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"resource object does not exist: {raw}")


def _disassemble_object(path: Path, nvdisasm: str) -> tuple[bytes, str]:
    object_bytes = path.read_bytes()
    embedded_cubin_count = object_bytes.count(_CUDA_ELF_MAGIC)
    if embedded_cubin_count != 1:
        raise ValueError(
            f"{path}: expected exactly one embedded CUDA ELF, found "
            f"{embedded_cubin_count}"
        )
    cubin = object_bytes[object_bytes.find(_CUDA_ELF_MAGIC) :]
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(cubin)
        cubin_file.flush()
        result = subprocess.run(
            [nvdisasm, cubin_file.name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return object_bytes, result.stdout


def _register_indices(code: str, family: str) -> list[int]:
    instruction_text = "\n".join(_INSTRUCTION_LINE_RE.findall(code))
    if not instruction_text:
        raise ValueError("nvdisasm kernel section contains no SASS instructions")
    return sorted(
        {
            int(match.group("index"))
            for match in _REGISTER_PATTERNS[family].finditer(instruction_text)
        }
    )


def _register_reconfiguration(
    disassembly: str,
    kernel: str,
    code: str,
) -> dict[str, str | int]:
    """Return exact warp-role register-reconfiguration evidence.

    ``EIATTR_REGCOUNT`` is the launch/static allocation reported by the CUDA
    driver.  Warp-specialized kernels can subsequently redistribute the CTA
    pool with ``USETMAXREG`` and legitimately name registers above that static
    count.  Preserve both facts and validate named R indices against the
    largest explicit dynamic target, not against REGCOUNT alone.
    """

    instruction_text = "\n".join(_INSTRUCTION_LINE_RE.findall(code))
    operations: list[tuple[str, int]] = []
    for match in _SETMAXREG_RE.finditer(instruction_text):
        operation = match.group("operation")
        target = int(match.group("target"), 0)
        if target <= 0 or target > 256:
            raise ValueError(
                f"{kernel}: invalid USETMAXREG target {target} in {operation}"
            )
        operations.append((operation, target))

    info = _kernel_info(disassembly, kernel)
    metadata_blocks = _attribute_blocks(info, "EIATTR_REG_RECONFIG")
    if len(metadata_blocks) > 1:
        raise ValueError(f"{kernel}: duplicate EIATTR_REG_RECONFIG metadata")
    metadata_present = bool(metadata_blocks)
    instructions_present = bool(operations)
    if metadata_present != instructions_present:
        raise ValueError(
            f"{kernel}: EIATTR_REG_RECONFIG/USETMAXREG disagreement: "
            f"metadata={metadata_present} instructions={len(operations)}"
        )

    deallocation_targets: list[int] = []
    allocation_targets: list[int] = []
    for operation, target in operations:
        if operation == "DEALLOC":
            deallocation_targets.append(target)
        elif "ALLOC" in operation:
            allocation_targets.append(target)
        else:
            raise ValueError(
                f"{kernel}: unsupported USETMAXREG operation {operation!r}"
            )
    targets = [target for _, target in operations]
    return {
        "register_reconfiguration": "true" if operations else "false",
        "register_reconfiguration_instruction_count": len(operations),
        "register_deallocation_targets_json": json.dumps(
            sorted(set(deallocation_targets)), separators=(",", ":")
        ),
        "register_deallocation_instruction_count": len(deallocation_targets),
        "register_allocation_targets_json": json.dumps(
            sorted(set(allocation_targets)), separators=(",", ":")
        ),
        "register_allocation_instruction_count": len(allocation_targets),
        "register_reconfiguration_min_target": min(targets, default=0),
        "register_reconfiguration_max_target": max(targets, default=0),
    }


def _set_fields(family: str, indices: list[int]) -> dict[str, str | int]:
    minimum = min(indices) if indices else -1
    maximum = max(indices) if indices else -1
    return {
        f"{family}_indices_json": json.dumps(indices, separators=(",", ":")),
        f"{family}_count": len(indices),
        f"{family}_min": minimum,
        f"{family}_max": maximum,
        f"{family}_span": maximum - minimum + 1 if indices else 0,
        # This matches the historical audit's max(index)+1 "span" metric.
        f"{family}_index_span": maximum + 1 if indices else 0,
    }


def _build_rows(
    report_path: Path,
    resource_rows: list[dict[str, str]],
    nvdisasm: str,
) -> list[dict[str, str | int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in resource_rows:
        grouped[row["object_file"]].append(row)

    output: list[dict[str, str | int]] = []
    for object_file, rows in sorted(grouped.items()):
        object_path = _resolve_object_path(object_file, report_path)
        object_bytes, disassembly = _disassemble_object(object_path, nvdisasm)
        actual_sha256 = hashlib.sha256(object_bytes).hexdigest()
        reported_hashes = {row["object_sha256"] for row in rows}
        if reported_hashes != {actual_sha256}:
            raise ValueError(
                f"{object_path}: object SHA-256 differs from resource report: "
                f"reported={sorted(reported_hashes)!r} actual={actual_sha256}"
            )
        disassembled_kernels = {
            match.group("kernel")
            for match in _TEXT_KERNEL_SECTION_RE.finditer(disassembly)
        }
        reported_kernels = {row["kernel"] for row in rows}
        if disassembled_kernels != reported_kernels:
            raise ValueError(
                f"{object_path}: resource-report kernel set differs from cubin: "
                f"missing={sorted(disassembled_kernels - reported_kernels)!r} "
                f"unexpected={sorted(reported_kernels - disassembled_kernels)!r}"
            )
        allocated_by_kernel = _resource_values(disassembly, "EIATTR_REGCOUNT")
        if set(allocated_by_kernel) != disassembled_kernels:
            raise ValueError(f"{object_path}: EIATTR_REGCOUNT kernel set differs")

        for row in rows:
            kernel = row["kernel"]
            code = _kernel_code(disassembly, kernel)
            if not code:
                raise ValueError(f"{object_path}: missing text for {kernel}")
            allocated = allocated_by_kernel[kernel]
            if allocated != _integer(row, "registers", row_number=0):
                raise ValueError(
                    f"{object_path}:{kernel}: allocated REG differs from resource report"
                )
            register_reconfiguration = _register_reconfiguration(
                disassembly, kernel, code
            )
            effective_register_index_ceiling = max(
                allocated,
                int(register_reconfiguration["register_reconfiguration_max_target"]),
            )
            register_fields: dict[str, str | int] = {}
            register_sets: dict[str, list[int]] = {}
            for family in _REGISTER_PATTERNS:
                indices = _register_indices(code, family)
                register_sets[family] = indices
                register_fields.update(_set_fields(family, indices))
            if (
                register_sets["r"]
                and register_sets["r"][-1] >= effective_register_index_ceiling
            ):
                raise ValueError(
                    f"{object_path}:{kernel}: SASS names R{register_sets['r'][-1]} "
                    "but its effective static/dynamic register-index ceiling is "
                    f"{effective_register_index_ceiling} "
                    f"(EIATTR_REGCOUNT={allocated}, "
                    "USETMAXREG maximum="
                    f"{register_reconfiguration['register_reconfiguration_max_target']})"
                )

            historical = {
                "ur": (
                    "sass_uniform_registers_used",
                    "sass_uniform_register_span",
                ),
                "p": (
                    "sass_predicate_registers_used",
                    "sass_predicate_register_span",
                ),
                "up": (
                    "sass_uniform_predicate_registers_used",
                    "sass_uniform_predicate_register_span",
                ),
            }
            for family, (count_field, span_field) in historical.items():
                count = int(register_fields[f"{family}_count"])
                index_span = int(register_fields[f"{family}_index_span"])
                if count != _integer(row, count_field, row_number=0):
                    raise ValueError(
                        f"{object_path}:{kernel}: {family.upper()} count differs "
                        "from resource report"
                    )
                if index_span != _integer(row, span_field, row_number=0):
                    raise ValueError(
                        f"{object_path}:{kernel}: {family.upper()} index span "
                        "differs from resource report"
                    )

            static_shared = _integer(row, "driver_static_shared_bytes", row_number=0)
            dynamic_shared = _integer(row, "launch_dynamic_smem_bytes", row_number=0)
            output.append(
                {
                    "sass_register_set_report_schema": _SIDECAR_SCHEMA,
                    **{field: row[field] for field in _IDENTITY_FIELDS},
                    "allocated_registers": allocated,
                    "eiattr_max_register_count": _integer(
                        row, "max_register_count", row_number=0
                    ),
                    **register_reconfiguration,
                    "effective_register_index_ceiling": (
                        effective_register_index_ceiling
                    ),
                    **{
                        field: _integer(row, field, row_number=0)
                        for field in (
                            "frame_bytes",
                            "min_stack_bytes",
                            "local_load_instructions",
                            "local_store_instructions",
                            "driver_local_bytes",
                            "cubin_shared_section_bytes",
                            "driver_static_shared_bytes",
                            "launch_dynamic_smem_bytes",
                        )
                    },
                    "total_launch_shared_bytes": static_shared + dynamic_shared,
                    **register_fields,
                }
            )
    return sorted(
        output,
        key=lambda row: (
            str(row["comparison_semantic_key"]),
            str(row["kernel"]),
        ),
    )


def _write_rows(
    rows: list[dict[str, str | int]], output_path: Path | None, output_format: str
) -> None:
    output = (
        output_path.open("w", newline="", encoding="utf-8")
        if output_path is not None
        else sys.stdout
    )
    try:
        if output_format == "json":
            json.dump(rows, output, indent=2)
            output.write("\n")
        else:
            writer = csv.DictWriter(output, fieldnames=_OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    finally:
        if output_path is not None:
            output.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource_report", type=Path)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    nvdisasm = shutil.which("nvdisasm")
    if nvdisasm is None:
        parser.error("nvdisasm is required")
    try:
        resource_rows = _read_resource_report(args.resource_report)
        rows = _build_rows(args.resource_report, resource_rows, nvdisasm)
        _write_rows(rows, args.output, args.format)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
