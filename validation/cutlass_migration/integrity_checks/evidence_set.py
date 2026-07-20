#!/usr/bin/env python3
"""Mutation tests for the offline E2E evidence-set constructor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

from validation.cutlass_migration.acceptance.e2e.evidence_set import (
    EvidenceSetBuildError,
    build_evidence_set,
)
from validation.cutlass_migration.acceptance.e2e.index import (
    CONTRACT_SCHEMA,
    PHYSICAL_GPUS,
    POSITION_ARM,
    REQUIRED_FAMILIES,
    RUN_SCHEMA,
    SEQUENCE,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _hashed(payload: dict[str, object], field: str) -> dict[str, object]:
    return {**payload, field: _canonical_sha256(payload)}


def _expect_failure(label: str, function, needle: str) -> None:
    try:
        function()
    except EvidenceSetBuildError as exc:
        if needle not in str(exc):
            raise AssertionError(f"{label}: wrong failure: {exc}") from exc
    else:
        raise AssertionError(f"{label}: mutation unexpectedly passed")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sparkinfer-evidence-set-integrity-") as raw:
        root = Path(raw)
        contract = _hashed(
            {
                "schema": CONTRACT_SCHEMA,
                "corpus_id": "integrity",
                "version": "1",
            },
            "contract_sha256",
        )
        contract_path = root / "contract.json"
        _write(contract_path, contract)
        run_root = root / "runs"
        runs: list[Path] = []
        for ordinal, family in enumerate(REQUIRED_FAMILIES):
            for gpu in PHYSICAL_GPUS:
                for position in SEQUENCE:
                    payload: dict[str, object] = {
                        "schema": RUN_SCHEMA,
                        "family": family,
                        "arm": POSITION_ARM[position],
                        "sequence_position": position,
                        "evidence_status": "final-source",
                        "harness_case_contract_sha256": contract["contract_sha256"],
                        "gpu": {"physical_ordinal": gpu},
                        "ordinal": ordinal,
                    }
                    run = _hashed(payload, "result_sha256")
                    path = run_root / f"gpu{gpu}" / position / f"{family}.json"
                    _write(path, run)
                    runs.append(path)
        output = root / "evidence.json"
        value = build_evidence_set(
            contract_path=contract_path,
            run_paths=[],
            run_roots=[run_root],
            output_path=output,
        )
        assert len(value["runs"]) == 104
        assert value["contract_sha256"] == contract["contract_sha256"]

        missing = runs[-1]
        saved = missing.read_bytes()
        missing.unlink()
        _expect_failure(
            "missing run",
            lambda: build_evidence_set(
                contract_path=contract_path,
                run_paths=[],
                run_roots=[run_root],
                output_path=output,
            ),
            "incomplete process-result product",
        )
        missing.write_bytes(saved)

        first = runs[0]
        first_value = json.loads(first.read_text(encoding="utf-8"))
        corrupted = copy.deepcopy(first_value)
        corrupted["arm"] = "current"
        _write(first, corrupted)
        _expect_failure(
            "corrupt hash",
            lambda: build_evidence_set(
                contract_path=contract_path,
                run_paths=[],
                run_roots=[run_root],
                output_path=output,
            ),
            "canonical hash mismatch",
        )
        _write(first, first_value)

        diagnostic_payload = {
            key: item for key, item in first_value.items() if key != "result_sha256"
        }
        diagnostic_payload["evidence_status"] = "diagnostic-non-final"
        _write(first, _hashed(diagnostic_payload, "result_sha256"))
        _expect_failure(
            "diagnostic run",
            lambda: build_evidence_set(
                contract_path=contract_path,
                run_paths=[],
                run_roots=[run_root],
                output_path=output,
            ),
            "diagnostic evidence cannot enter release",
        )
        _write(first, first_value)

        duplicate = root / "duplicate.json"
        _write(duplicate, first_value)
        _expect_failure(
            "duplicate run",
            lambda: build_evidence_set(
                contract_path=contract_path,
                run_paths=[duplicate],
                run_roots=[run_root],
                output_path=output,
            ),
            "duplicate process identity",
        )

        symlink = root / "run-link.json"
        symlink.symlink_to(first)
        _expect_failure(
            "symlink run",
            lambda: build_evidence_set(
                contract_path=contract_path,
                run_paths=[symlink],
                run_roots=[],
                output_path=output,
            ),
            "non-symlink file",
        )
    print("evidence-set integrity checks: pass (104 closed process identities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
