#!/usr/bin/env python3
"""Audit readiness of the fail-closed CUTLASS migration E2E producers.

This durable orchestrator intentionally refuses to launch a partial corpus.
Every required family must first have a one-arm, one-process GPU producer;
paired in-process comparators are retained as diagnostics but cannot satisfy
the separate-source A1/B1/B2/A2 release contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from validation.cutlass_migration.acceptance.e2e.index import REQUIRED_FAMILIES
from validation.cutlass_migration.paths import REPO_ROOT


SCHEMA = "sparkinfer.cute.migration.end_to_end_producer_readiness.v1"


class ProducerBinding(NamedTuple):
    paired_diagnostic: str
    single_arm: str | None


REGISTRY = {
    family: ProducerBinding(
        f"validation/cutlass_migration/diagnostics/paired/{family}.py",
        f"validation/cutlass_migration/acceptance/single_arm/{family}.py",
    )
    for family in REQUIRED_FAMILIES
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional deterministic JSON readiness report",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(repo_root: Path, relative_path: str) -> dict[str, object]:
    path = repo_root / relative_path
    return {
        "path": relative_path,
        "exists": path.is_file(),
        "sha256": _sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def build_readiness(repo_root: Path) -> dict[str, object]:
    required = set(REQUIRED_FAMILIES)
    registered = set(REGISTRY)
    if registered != required:
        raise RuntimeError(
            "producer registry differs from the closed family contract: "
            f"missing={sorted(required - registered)}, "
            f"unexpected={sorted(registered - required)}"
        )
    families: dict[str, object] = {}
    missing_single_arm: list[str] = []
    invalid_paired: list[str] = []
    invalid_single_arm: list[str] = []
    for family in REQUIRED_FAMILIES:
        binding = REGISTRY[family]
        paired = _file_record(repo_root, binding.paired_diagnostic)
        if paired["exists"] is not True:
            invalid_paired.append(family)
        if binding.single_arm is None:
            single_arm: dict[str, object] = {
                "path": None,
                "exists": False,
                "sha256": None,
                "size_bytes": None,
                "contract_markers_present": False,
            }
            missing_single_arm.append(family)
        else:
            single_arm = _file_record(repo_root, binding.single_arm)
            producer_path = repo_root / binding.single_arm
            source = (
                producer_path.read_text(encoding="utf-8")
                if producer_path.is_file()
                else ""
            )
            markers_present = all(
                marker in source
                for marker in (
                    "begin_single_arm_session",
                    "finish_single_arm_session",
                    "exact_artifact_evidence",
                    "time_single_graph_conditions",
                )
            )
            single_arm["contract_markers_present"] = markers_present
            if single_arm["exists"] is not True or not markers_present:
                invalid_single_arm.append(family)
        families[family] = {
            "paired_diagnostic": paired,
            "single_arm_producer": single_arm,
            "ready": (
                paired["exists"] is True
                and single_arm["exists"] is True
                and single_arm["contract_markers_present"] is True
            ),
        }
    blockers = sorted(set(missing_single_arm) | set(invalid_paired) | set(invalid_single_arm))
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "required_families": list(REQUIRED_FAMILIES),
        "policy": {
            "partial_launch_allowed": False,
            "paired_in_process_counts_as_e2e": False,
            "physical_gpus": [4, 5],
            "sequence": ["a1", "b1", "b2", "a2"],
        },
        "families": families,
        "summary": {
            "required_family_count": len(REQUIRED_FAMILIES),
            "ready_family_count": len(REQUIRED_FAMILIES) - len(blockers),
            "missing_single_arm_families": missing_single_arm,
            "invalid_paired_diagnostic_families": invalid_paired,
            "invalid_single_arm_families": invalid_single_arm,
            "blockers": blockers,
            "ready": not blockers,
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return {**payload, "readiness_sha256": hashlib.sha256(encoded).hexdigest()}


def main() -> int:
    args = _args()
    report = build_readiness(REPO_ROOT)
    encoded = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["summary"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
