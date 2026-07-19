#!/usr/bin/env python3
"""Build the closed 104-process CUTLASS migration evidence-set manifest.

The end-to-end index intentionally consumes an explicit immutable manifest,
not a best-effort directory scan.  This command is the production constructor
for that manifest: it accepts dedicated run files/directories, rejects every
non-run or duplicate artifact, verifies each embedded result hash, and requires
the complete family x GPU x A1/B1/B2/A2 product before publishing anything.

This module is offline and must not import Torch or CUTLASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

from validation.cutlass_migration.acceptance.e2e.index import (
    CONTRACT_SCHEMA,
    EVIDENCE_SET_SCHEMA,
    PHYSICAL_GPUS,
    POSITION_ARM,
    REQUIRED_FAMILIES,
    RUN_SCHEMA,
    SEQUENCE,
)


class EvidenceSetBuildError(RuntimeError):
    """A process artifact set is not admissible for release validation."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, location: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceSetBuildError(f"{location}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceSetBuildError(f"{location}: JSON root is not an object")
    return value


def _require_regular_file(path: Path, *, location: str) -> Path:
    if path.is_symlink():
        raise EvidenceSetBuildError(
            f"{location}: not a regular non-symlink file: {path}"
        )
    path = path.resolve()
    if not path.is_file():
        raise EvidenceSetBuildError(
            f"{location}: not a regular non-symlink file: {path}"
        )
    return path


def _contract_sha256(path: Path) -> str:
    path = _require_regular_file(path, location="contract")
    contract = _load_json(path, location=str(path))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise EvidenceSetBuildError(f"{path}: unsupported contract schema")
    recorded = contract.get("contract_sha256")
    payload = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if not isinstance(recorded, str) or recorded != _canonical_sha256(payload):
        raise EvidenceSetBuildError(f"{path}: contract canonical hash mismatch")
    return recorded


def _collect_paths(
    *, run_paths: Sequence[Path], run_roots: Sequence[Path], output: Path
) -> list[Path]:
    collected: list[Path] = []
    for raw_path in run_paths:
        collected.append(_require_regular_file(raw_path, location="--run"))
    for raw_root in run_roots:
        if raw_root.is_symlink():
            raise EvidenceSetBuildError(
                f"--run-root is not a regular non-symlink directory: {raw_root}"
            )
        root = raw_root.resolve()
        if not root.is_dir():
            raise EvidenceSetBuildError(
                f"--run-root is not a regular non-symlink directory: {root}"
            )
        if output == root or root in output.parents:
            raise EvidenceSetBuildError("--output must be outside every --run-root")
        found = sorted(root.rglob("*.json"))
        if not found:
            raise EvidenceSetBuildError(f"--run-root contains no JSON files: {root}")
        for path in found:
            collected.append(_require_regular_file(path, location=str(root)))
    if not collected:
        raise EvidenceSetBuildError("provide at least one --run or --run-root")
    if output in collected:
        raise EvidenceSetBuildError("--output must not overwrite a run artifact")
    if len(set(collected)) != len(collected):
        raise EvidenceSetBuildError("the same run artifact was supplied more than once")
    return collected


def _run_key(
    path: Path, value: Mapping[str, Any], *, contract_sha256: str
) -> tuple[str, int, str]:
    recorded_result = value.get("result_sha256")
    result_payload = {
        key: item for key, item in value.items() if key != "result_sha256"
    }
    if value.get("schema") != RUN_SCHEMA:
        raise EvidenceSetBuildError(f"{path}: unsupported process-result schema")
    if not isinstance(recorded_result, str) or recorded_result != _canonical_sha256(
        result_payload
    ):
        raise EvidenceSetBuildError(f"{path}: process-result canonical hash mismatch")
    family = value.get("family")
    position = value.get("sequence_position")
    arm = value.get("arm")
    gpu_record = value.get("gpu")
    gpu = gpu_record.get("physical_ordinal") if isinstance(gpu_record, dict) else None
    if (
        not isinstance(family, str)
        or family not in REQUIRED_FAMILIES
        or not isinstance(gpu, int)
        or isinstance(gpu, bool)
        or gpu not in PHYSICAL_GPUS
        or not isinstance(position, str)
        or position not in SEQUENCE
    ):
        raise EvidenceSetBuildError(f"{path}: unreviewed family/GPU/position identity")
    if arm != POSITION_ARM[position]:
        raise EvidenceSetBuildError(f"{path}: arm does not match sequence position")
    if value.get("evidence_status") != "final-source":
        raise EvidenceSetBuildError(f"{path}: diagnostic evidence cannot enter release")
    if value.get("harness_case_contract_sha256") != contract_sha256:
        raise EvidenceSetBuildError(f"{path}: run binds a different contract")
    return family, gpu, position


def _published_path(path: Path, *, output_parent: Path) -> str:
    try:
        return path.relative_to(output_parent).as_posix()
    except ValueError:
        return str(path)


def build_evidence_set(
    *,
    contract_path: Path,
    run_paths: Sequence[Path],
    run_roots: Sequence[Path],
    output_path: Path,
) -> dict[str, object]:
    """Return one canonical, complete evidence-set value."""

    output = output_path.resolve()
    contract_sha256 = _contract_sha256(contract_path)
    paths = _collect_paths(run_paths=run_paths, run_roots=run_roots, output=output)
    expected = {
        (family, gpu, position)
        for family in REQUIRED_FAMILIES
        for gpu in PHYSICAL_GPUS
        for position in SEQUENCE
    }
    by_key: dict[tuple[str, int, str], Path] = {}
    for path in paths:
        value = _load_json(path, location=str(path))
        key = _run_key(path, value, contract_sha256=contract_sha256)
        if key in by_key:
            raise EvidenceSetBuildError(
                f"duplicate process identity {key!r}: {by_key[key]} and {path}"
            )
        by_key[key] = path
    if set(by_key) != expected:
        raise EvidenceSetBuildError(
            "incomplete process-result product: "
            f"missing={sorted(expected - set(by_key))!r}, "
            f"unexpected={sorted(set(by_key) - expected)!r}"
        )
    order = {
        (family, gpu, position): (family_index, gpu_index, position_index)
        for family_index, family in enumerate(REQUIRED_FAMILIES)
        for gpu_index, gpu in enumerate(PHYSICAL_GPUS)
        for position_index, position in enumerate(SEQUENCE)
    }
    entries = []
    for family, gpu, position in sorted(by_key, key=order.__getitem__):
        path = by_key[(family, gpu, position)]
        entries.append(
            {
                "family": family,
                "physical_gpu": gpu,
                "position": position,
                "path": _published_path(path, output_parent=output.parent),
                "sha256": _sha256_file(path),
            }
        )
    payload: dict[str, object] = {
        "schema": EVIDENCE_SET_SCHEMA,
        "contract_sha256": contract_sha256,
        "runs": entries,
    }
    return {**payload, "evidence_set_sha256": _canonical_sha256(payload)}


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                value,
                temporary,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=[],
        help="one process-result JSON; repeat for individual artifacts",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="dedicated tree containing only process-result JSON; repeat if needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    try:
        value = build_evidence_set(
            contract_path=args.contract,
            run_paths=args.run,
            run_roots=args.run_root,
            output_path=args.output,
        )
        output = args.output.resolve()
        _atomic_write_json(output, value)
        if _load_json(output, location=str(output)) != value:
            raise EvidenceSetBuildError("published evidence set failed readback")
    except (EvidenceSetBuildError, OSError, ValueError) as exc:
        print(f"evidence-set build failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"status=pass runs={len(value['runs'])} "
        f"evidence_set_sha256={value['evidence_set_sha256']} output={output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
