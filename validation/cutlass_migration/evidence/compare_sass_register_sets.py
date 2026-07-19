#!/usr/bin/env python3
"""Compare exact per-specialization SASS register-set sidecars.

Every comparison specialization and CUDA entry point is retained, including
unchanged rows.  In addition to allocated GPR changes, the report exposes exact
R/UR/P/UP additions and removals and flags increases in count, maximum index,
or inclusive used-index span.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


_SIDECAR_SCHEMA = "b12x.cute.sass_register_sets.v3"
_DELTA_SCHEMA = "b12x.cute.sass_register_set_delta.v3"
_FAMILIES = ("r", "ur", "p", "up")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_KEY = ("comparison_semantic_key", "kernel")
_STABLE_FIELDS = (
    "resource_report_schema",
    "target",
    "kernel_id",
    "compile_spec_version",
    "compile_spec_hash",
    "compile_spec_json",
    "architecture",
)
_RESOURCE_FIELDS = (
    "allocated_registers",
    "eiattr_max_register_count",
    "register_reconfiguration_instruction_count",
    "register_deallocation_instruction_count",
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
)
_RECONFIGURATION_FIELDS = (
    "register_deallocation_targets_json",
    "register_allocation_targets_json",
)
_OUTPUT_FIELDS = (
    "sass_register_set_delta_schema",
    "comparison_semantic_key",
    "baseline_semantic_key",
    "current_semantic_key",
    "kernel",
    "target",
    "kernel_id",
    "compile_spec_version",
    "compile_spec_hash",
    "compile_spec_json",
    "architecture",
    "baseline_cutlass_dsl_version",
    "current_cutlass_dsl_version",
    "baseline_package_fingerprint",
    "current_package_fingerprint",
    "baseline_object_sha256",
    "current_object_sha256",
    *(
        field
        for resource in _RESOURCE_FIELDS
        for field in (
            f"baseline_{resource}",
            f"current_{resource}",
            f"{resource}_delta",
            f"{resource}_increase",
        )
    ),
    "baseline_register_reconfiguration",
    "current_register_reconfiguration",
    "register_reconfiguration_change",
    *(
        field
        for target_field in _RECONFIGURATION_FIELDS
        for field in (
            f"baseline_{target_field}",
            f"current_{target_field}",
            f"{target_field.removesuffix('_json')}_added_json",
            f"{target_field.removesuffix('_json')}_removed_json",
            f"{target_field.removesuffix('_json')}_set_change",
            f"{target_field.removesuffix('_json')}_maximum_increase",
        )
    ),
    "any_register_reconfiguration_target_change",
    "any_register_reconfiguration_target_addition",
    "any_register_reconfiguration_target_increase",
    *(
        field
        for family in _FAMILIES
        for field in (
            f"baseline_{family}_indices_json",
            f"current_{family}_indices_json",
            f"{family}_added_indices_json",
            f"{family}_removed_indices_json",
            f"{family}_set_change",
            f"baseline_{family}_count",
            f"current_{family}_count",
            f"{family}_count_delta",
            f"{family}_count_increase",
            f"baseline_{family}_min",
            f"current_{family}_min",
            f"{family}_min_delta",
            f"baseline_{family}_max",
            f"current_{family}_max",
            f"{family}_max_delta",
            f"{family}_max_increase",
            f"baseline_{family}_span",
            f"current_{family}_span",
            f"{family}_span_delta",
            f"{family}_span_increase",
            f"baseline_{family}_index_span",
            f"current_{family}_index_span",
            f"{family}_index_span_delta",
            f"{family}_index_span_increase",
            f"{family}_usage_increase",
        )
    ),
    "any_register_set_change",
    "any_register_set_addition",
    "any_register_usage_increase",
    "any_local_memory_increase",
    "any_shared_memory_increase",
)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _integer(row: dict[str, str], field: str, *, location: str) -> int:
    raw = row.get(field, "").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{location}: invalid {field}={raw!r}") from exc


def _indices(row: dict[str, str], family: str, *, location: str) -> list[int]:
    field = f"{family}_indices_json"
    try:
        value = json.loads(row.get(field, ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}: invalid {field}") from exc
    if (
        not isinstance(value, list)
        or any(not isinstance(index, int) or index < 0 for index in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{location}: {field} is not a sorted unique index list")
    count = len(value)
    minimum = min(value) if value else -1
    maximum = max(value) if value else -1
    expected = {
        f"{family}_count": count,
        f"{family}_min": minimum,
        f"{family}_max": maximum,
        f"{family}_span": maximum - minimum + 1 if value else 0,
        f"{family}_index_span": maximum + 1 if value else 0,
    }
    for metric, expected_value in expected.items():
        actual = _integer(row, metric, location=location)
        if actual != expected_value:
            raise ValueError(
                f"{location}: {metric}={actual}, expected {expected_value}"
            )
    return value


def _targets(row: dict[str, str], field: str, *, location: str) -> list[int]:
    try:
        value = json.loads(row.get(field, ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}: invalid {field}") from exc
    if (
        not isinstance(value, list)
        or any(
            isinstance(target, bool)
            or not isinstance(target, int)
            or target <= 0
            or target > 256
            for target in value
        )
        or value != sorted(set(value))
    ):
        raise ValueError(f"{location}: {field} is not a sorted unique target list")
    return value


def _read(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    try:
        source = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    with source:
        reader = csv.DictReader(source)
        required = {
            "sass_register_set_report_schema",
            "semantic_key",
            "comparison_semantic_key",
            "kernel",
            "cutlass_dsl_version",
            "package_fingerprint",
            "object_sha256",
            *_STABLE_FIELDS,
            *_RESOURCE_FIELDS,
            "register_reconfiguration",
            *_RECONFIGURATION_FIELDS,
            *(
                field
                for family in _FAMILIES
                for field in (
                    f"{family}_indices_json",
                    f"{family}_count",
                    f"{family}_min",
                    f"{family}_max",
                    f"{family}_span",
                    f"{family}_index_span",
                )
            ),
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows: dict[tuple[str, str], dict[str, str]] = {}
        for row_number, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            location = f"{path}:{row_number}"
            if row["sass_register_set_report_schema"] != _SIDECAR_SCHEMA:
                raise ValueError(f"{location}: invalid sidecar schema")
            key = tuple(row[field] for field in _PAIR_KEY)
            if not all(key):
                raise ValueError(f"{location}: empty pair key")
            for field in (
                "semantic_key",
                "comparison_semantic_key",
                "package_fingerprint",
                "object_sha256",
                "compile_spec_hash",
            ):
                if not _SHA256_RE.fullmatch(row.get(field, "")):
                    raise ValueError(f"{location}: invalid SHA-256 field {field}")
            if key in rows:
                raise ValueError(f"{location}: duplicate pair key {key!r}")
            for field in _RESOURCE_FIELDS:
                _integer(row, field, location=location)
            for family in _FAMILIES:
                _indices(row, family, location=location)
            raw_reconfiguration = row["register_reconfiguration"].lower()
            if raw_reconfiguration not in {"true", "false"}:
                raise ValueError(
                    f"{location}: invalid register_reconfiguration="
                    f"{row['register_reconfiguration']!r}"
                )
            deallocation_targets = _targets(
                row, "register_deallocation_targets_json", location=location
            )
            allocation_targets = _targets(
                row, "register_allocation_targets_json", location=location
            )
            reconfiguration_count = _integer(
                row, "register_reconfiguration_instruction_count", location=location
            )
            deallocation_count = _integer(
                row, "register_deallocation_instruction_count", location=location
            )
            allocation_count = _integer(
                row, "register_allocation_instruction_count", location=location
            )
            targets = [*deallocation_targets, *allocation_targets]
            expected_minimum = min(targets, default=0)
            expected_maximum = max(targets, default=0)
            if (
                deallocation_count < len(deallocation_targets)
                or allocation_count < len(allocation_targets)
                or reconfiguration_count != deallocation_count + allocation_count
                or (raw_reconfiguration == "true") != bool(reconfiguration_count)
                or _integer(
                    row, "register_reconfiguration_min_target", location=location
                )
                != expected_minimum
                or _integer(
                    row, "register_reconfiguration_max_target", location=location
                )
                != expected_maximum
            ):
                raise ValueError(
                    f"{location}: inconsistent register-reconfiguration accounting"
                )
            allocated = _integer(row, "allocated_registers", location=location)
            effective_ceiling = _integer(
                row, "effective_register_index_ceiling", location=location
            )
            if effective_ceiling != max(allocated, expected_maximum):
                raise ValueError(
                    f"{location}: inconsistent effective register-index ceiling"
                )
            r_indices = _indices(row, "r", location=location)
            if r_indices and r_indices[-1] >= effective_ceiling:
                raise ValueError(
                    f"{location}: R{r_indices[-1]} exceeds effective exclusive "
                    f"ceiling {effective_ceiling}"
                )
            rows[key] = row
    if not rows:
        raise ValueError(f"{path}: sidecar is empty")
    return rows


def _resource_delta(
    old: dict[str, str], new: dict[str, str], field: str
) -> dict[str, int | str]:
    baseline = int(old[field])
    current = int(new[field])
    delta = current - baseline
    return {
        f"baseline_{field}": baseline,
        f"current_{field}": current,
        f"{field}_delta": delta,
        f"{field}_increase": _bool(delta > 0),
    }


def _target_delta(
    old: dict[str, str], new: dict[str, str], field: str
) -> tuple[dict[str, str], bool, bool, bool]:
    baseline_targets = json.loads(old[field])
    current_targets = json.loads(new[field])
    added = sorted(set(current_targets) - set(baseline_targets))
    removed = sorted(set(baseline_targets) - set(current_targets))
    stem = field.removesuffix("_json")
    changed = bool(added or removed)
    maximum_increase = max(current_targets, default=0) > max(
        baseline_targets, default=0
    )
    return (
        {
            f"baseline_{field}": json.dumps(baseline_targets, separators=(",", ":")),
            f"current_{field}": json.dumps(current_targets, separators=(",", ":")),
            f"{stem}_added_json": json.dumps(added, separators=(",", ":")),
            f"{stem}_removed_json": json.dumps(removed, separators=(",", ":")),
            f"{stem}_set_change": _bool(changed),
            f"{stem}_maximum_increase": _bool(maximum_increase),
        },
        changed,
        bool(added),
        maximum_increase,
    )


def _family_delta(
    old: dict[str, str], new: dict[str, str], family: str
) -> tuple[dict[str, int | str], bool, bool, bool]:
    baseline_indices = json.loads(old[f"{family}_indices_json"])
    current_indices = json.loads(new[f"{family}_indices_json"])
    baseline_set = set(baseline_indices)
    current_set = set(current_indices)
    added = sorted(current_set - baseline_set)
    removed = sorted(baseline_set - current_set)
    result: dict[str, int | str] = {
        f"baseline_{family}_indices_json": json.dumps(
            baseline_indices, separators=(",", ":")
        ),
        f"current_{family}_indices_json": json.dumps(
            current_indices, separators=(",", ":")
        ),
        f"{family}_added_indices_json": json.dumps(added, separators=(",", ":")),
        f"{family}_removed_indices_json": json.dumps(removed, separators=(",", ":")),
        f"{family}_set_change": _bool(bool(added or removed)),
    }
    increases = []
    for metric in ("count", "min", "max", "span", "index_span"):
        baseline = int(old[f"{family}_{metric}"])
        current = int(new[f"{family}_{metric}"])
        delta = current - baseline
        result.update(
            {
                f"baseline_{family}_{metric}": baseline,
                f"current_{family}_{metric}": current,
                f"{family}_{metric}_delta": delta,
            }
        )
        if metric in {"count", "max", "span", "index_span"}:
            increase = delta > 0
            result[f"{family}_{metric}_increase"] = _bool(increase)
            increases.append(increase)
    usage_increase = any(increases)
    result[f"{family}_usage_increase"] = _bool(usage_increase)
    return result, bool(added or removed), bool(added), usage_increase


def _compare(
    baseline: dict[tuple[str, str], dict[str, str]],
    current: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, int | str]]:
    if set(baseline) != set(current):
        raise ValueError(
            "sidecar specialization sets differ: "
            f"missing={sorted(set(baseline) - set(current))!r} "
            f"unexpected={sorted(set(current) - set(baseline))!r}"
        )
    rows: list[dict[str, int | str]] = []
    for key in sorted(baseline):
        old = baseline[key]
        new = current[key]
        changed_stable = [field for field in _STABLE_FIELDS if old[field] != new[field]]
        if changed_stable:
            raise ValueError(f"{key!r}: stable fields changed: {changed_stable}")
        row: dict[str, int | str] = {
            "sass_register_set_delta_schema": _DELTA_SCHEMA,
            "comparison_semantic_key": key[0],
            "baseline_semantic_key": old["semantic_key"],
            "current_semantic_key": new["semantic_key"],
            "kernel": key[1],
            **{
                field: old[field]
                for field in _STABLE_FIELDS
                if field != "resource_report_schema"
            },
            "baseline_cutlass_dsl_version": old["cutlass_dsl_version"],
            "current_cutlass_dsl_version": new["cutlass_dsl_version"],
            "baseline_package_fingerprint": old["package_fingerprint"],
            "current_package_fingerprint": new["package_fingerprint"],
            "baseline_object_sha256": old["object_sha256"],
            "current_object_sha256": new["object_sha256"],
        }
        resource_increases: dict[str, bool] = {}
        for field in _RESOURCE_FIELDS:
            delta = _resource_delta(old, new, field)
            row.update(delta)
            resource_increases[field] = delta[f"{field}_increase"] == "true"

        baseline_reconfiguration = old["register_reconfiguration"].lower() == "true"
        current_reconfiguration = new["register_reconfiguration"].lower() == "true"
        row.update(
            {
                "baseline_register_reconfiguration": _bool(baseline_reconfiguration),
                "current_register_reconfiguration": _bool(current_reconfiguration),
                "register_reconfiguration_change": _bool(
                    baseline_reconfiguration != current_reconfiguration
                ),
            }
        )
        reconfiguration_target_change = False
        reconfiguration_target_addition = False
        reconfiguration_target_increase = False
        for field in _RECONFIGURATION_FIELDS:
            target_fields, changed, added, increased = _target_delta(old, new, field)
            row.update(target_fields)
            reconfiguration_target_change |= changed
            reconfiguration_target_addition |= added
            reconfiguration_target_increase |= increased
        row.update(
            {
                "any_register_reconfiguration_target_change": _bool(
                    reconfiguration_target_change
                ),
                "any_register_reconfiguration_target_addition": _bool(
                    reconfiguration_target_addition
                ),
                "any_register_reconfiguration_target_increase": _bool(
                    reconfiguration_target_increase
                ),
            }
        )

        set_change = False
        set_addition = False
        register_usage_increase = any(
            (
                resource_increases["allocated_registers"],
                resource_increases["eiattr_max_register_count"],
                resource_increases["effective_register_index_ceiling"],
                reconfiguration_target_increase,
            )
        )
        for family in _FAMILIES:
            family_fields, changed, added, increased = _family_delta(old, new, family)
            row.update(family_fields)
            set_change |= changed
            set_addition |= added
            register_usage_increase |= increased
        local_increase = any(
            resource_increases[field]
            for field in (
                "frame_bytes",
                "min_stack_bytes",
                "local_load_instructions",
                "local_store_instructions",
                "driver_local_bytes",
            )
        )
        shared_increase = any(
            resource_increases[field]
            for field in (
                "cubin_shared_section_bytes",
                "driver_static_shared_bytes",
                "launch_dynamic_smem_bytes",
                "total_launch_shared_bytes",
            )
        )
        row.update(
            {
                "any_register_set_change": _bool(set_change),
                "any_register_set_addition": _bool(set_addition),
                "any_register_usage_increase": _bool(register_usage_increase),
                "any_local_memory_increase": _bool(local_increase),
                "any_shared_memory_increase": _bool(shared_increase),
            }
        )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--fail-on-register-usage-increase", action="store_true")
    parser.add_argument("--fail-on-local-memory-increase", action="store_true")
    parser.add_argument("--fail-on-shared-memory-increase", action="store_true")
    args = parser.parse_args()

    try:
        rows = _compare(_read(args.baseline), _read(args.current))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = (
        args.output.open("w", newline="", encoding="utf-8")
        if args.output is not None
        else sys.stdout
    )
    try:
        writer = csv.DictWriter(output, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output is not None:
            output.close()

    if args.fail_on_register_usage_increase and any(
        row["any_register_usage_increase"] == "true" for row in rows
    ):
        return 1
    if args.fail_on_local_memory_increase and any(
        row["any_local_memory_increase"] == "true" for row in rows
    ):
        return 1
    if args.fail_on_shared_memory_increase and any(
        row["any_shared_memory_increase"] == "true" for row in rows
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
