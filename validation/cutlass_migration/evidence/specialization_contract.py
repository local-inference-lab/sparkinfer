#!/usr/bin/env python3
"""Build an exact CuTe compile-specialization contract from resource CSVs.

The generated contract is intentionally source-independent: it identifies each
CUDA resource row by kernel id, compile-spec hash, raw semantic key, explicit
cross-toolchain comparison key, and exact CUDA entry-point symbol. Review the
corpus shape driver before checking the file in;
generating a contract from a partial corpus does not make that corpus complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


_FIELDS = [
    "kernel_id",
    "compile_spec_version",
    "compile_spec_hash",
    "compile_spec_json",
    "semantic_key",
    "comparison_semantic_key",
    "target",
    "compile_kwargs_json",
    "kernel",
]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_METADATA_SCHEMA = "sparkinfer.cute.resource_row_contract.v2"


def _file_record(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _validated_entry(
    path: Path, line_number: int, row: dict[str, str]
) -> tuple[str, ...]:
    entry = tuple(str(row.get(field, "")).strip() for field in _FIELDS)
    values = dict(zip(_FIELDS, entry, strict=True))
    required = set(_FIELDS) - {"compile_kwargs_json"}
    empty = sorted(field for field in required if not values[field])
    if empty:
        raise ValueError(f"{path}:{line_number}: empty identity fields {empty}")
    for field in (
        "compile_spec_hash",
        "semantic_key",
        "comparison_semantic_key",
    ):
        if not _SHA256_RE.fullmatch(values[field]):
            raise ValueError(f"{path}:{line_number}: {field} is not SHA-256")
    if (
        hashlib.sha256(values["compile_spec_json"].encode("utf-8")).hexdigest()
        != values["compile_spec_hash"]
    ):
        raise ValueError(f"{path}:{line_number}: compile_spec_hash mismatch")
    try:
        compile_spec = json.loads(values["compile_spec_json"])
        if values["compile_kwargs_json"]:
            json.loads(values["compile_kwargs_json"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: invalid compile JSON") from exc
    if not isinstance(compile_spec, dict):
        raise ValueError(f"{path}:{line_number}: compile spec is not an object")
    if compile_spec.get("kernel") != values["kernel_id"]:
        raise ValueError(f"{path}:{line_number}: kernel_id disagrees with spec")
    if str(compile_spec.get("version", "")) != values["compile_spec_version"]:
        raise ValueError(f"{path}:{line_number}: spec version disagrees with spec")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--corpus-version", required=True)
    parser.add_argument("--corpus-driver", type=Path, required=True)
    parser.add_argument("--shape-matrix", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    args = parser.parse_args()

    entries: dict[tuple[str, str], tuple[str, ...]] = {}
    semantic_objects: dict[str, set[tuple[str, str, str]]] = {}
    comparison_semantic_keys: set[str] = set()
    cache_semantics: dict[str, set[str]] = {}
    try:
        for path in args.reports:
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                missing_fields = set(_FIELDS) - set(reader.fieldnames or ())
                if missing_fields:
                    raise ValueError(
                        f"{path}: missing resource columns {sorted(missing_fields)}"
                    )
                rows = 0
                for line_number, row in enumerate(reader, 2):
                    rows += 1
                    if row.get("manifest_status") != "ok":
                        raise ValueError(
                            f"{path}:{line_number}: invalid semantic manifest"
                        )
                    entry = _validated_entry(path, line_number, row)
                    semantic_key = entry[_FIELDS.index("semantic_key")]
                    comparison_semantic_key = entry[
                        _FIELDS.index("comparison_semantic_key")
                    ]
                    kernel = entry[_FIELDS.index("kernel")]
                    identity = (comparison_semantic_key, kernel)
                    if identity in entries:
                        raise ValueError(
                            f"{path}:{line_number}: duplicate resource-row identity "
                            f"{identity}"
                        )
                    entries[identity] = entry
                    object_identity = tuple(
                        str(row.get(field, "")).strip()
                        for field in ("cache_key", "object_sha256", "object_file")
                    )
                    if any(not value for value in object_identity):
                        raise ValueError(
                            f"{path}:{line_number}: incomplete object identity"
                        )
                    semantic_objects.setdefault(semantic_key, set()).add(
                        object_identity
                    )
                    comparison_semantic_keys.add(comparison_semantic_key)
                    cache_semantics.setdefault(object_identity[0], set()).add(
                        semantic_key
                    )
                if rows == 0:
                    raise ValueError(f"{path}: resource report has no rows")
        multi_object = {
            key: values for key, values in semantic_objects.items() if len(values) != 1
        }
        multi_semantic_cache = {
            key: values for key, values in cache_semantics.items() if len(values) != 1
        }
        if multi_object or multi_semantic_cache:
            raise ValueError(
                "resource rows do not form one object per semantic key: "
                f"multi_object={len(multi_object)} "
                f"multi_semantic_cache={len(multi_semantic_cache)}"
            )
        for artifact in (
            args.corpus_driver,
            args.shape_matrix,
            args.source_inventory,
        ):
            if not artifact.is_file():
                raise ValueError(
                    f"required contract artifact is not a file: {artifact}"
                )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open("w", newline="", encoding="utf-8")
    try:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(_FIELDS)
        writer.writerows(sorted(entries.values()))
    finally:
        output.close()
    unique_objects = {next(iter(objects)) for objects in semantic_objects.values()}
    metadata = {
        "schema": _METADATA_SCHEMA,
        "corpus_id": args.corpus_id,
        "corpus_version": args.corpus_version,
        "contract": {
            "path": str(args.output),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "fields": _FIELDS,
            "resource_row_count": len(entries),
            "semantic_key_count": len(semantic_objects),
            "comparison_semantic_key_count": len(comparison_semantic_keys),
            "object_count": len(unique_objects),
        },
        "reviewed_artifacts": {
            "contract_builder": _file_record(Path(__file__).resolve()),
            "corpus_driver": _file_record(args.corpus_driver),
            "shape_matrix": _file_record(args.shape_matrix),
            "source_inventory": _file_record(args.source_inventory),
        },
        "origin_reports": [_file_record(path) for path in args.reports],
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"resource_rows={len(entries)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
