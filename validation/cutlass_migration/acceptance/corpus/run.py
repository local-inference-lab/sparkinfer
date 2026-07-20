#!/usr/bin/env python3
"""Run and bind the GPU-only CUTLASS migration specialization corpus.

The coordinator deliberately does not import torch, cutlass, or sparkinfer.  Every
case runs in a fresh Python process, on one explicitly selected physical GPU,
through the package's pytest launcher. Nsight Systems supplies the
independent launch proof: a compile-manifest row is credited to a case only
when its exact CUDA entry point appears inside that case's NVTX range.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from validation.cutlass_migration.paths import (
    CORPUS_ROOT,
    DATA_ROOT,
    EVIDENCE_ROOT,
    REPO_ROOT,
    repo_relative,
)

_ROOT = REPO_ROOT
_MATRIX = DATA_ROOT / "cute_migration_corpus_matrix.json"
_INVENTORY = DATA_ROOT / "cute_kernel_source_inventory.tsv"
_FAMILY_PATTERNS = DATA_ROOT / "cute_production_kernel_coverage.txt"
_SYMBOL_PATTERNS = DATA_ROOT / "cute_kernel_symbol_coverage.txt"
_PLUGIN = "validation.cutlass_migration.acceptance.corpus.pytest_plugin"
_LAUNCHER = "validation.cutlass_migration.acceptance.corpus.pytest_launcher"
_AUDITOR_MODULE = "validation.cutlass_migration.evidence.kernel_resources"
_SOURCE_AUDITOR_MODULE = "validation.cutlass_migration.evidence.source_inventory"
_SMEM_CONTRACT_AUDITOR_MODULE = (
    "validation.cutlass_migration.evidence.smem_contracts"
)
_CONTRACT_BUILDER_MODULE = (
    "validation.cutlass_migration.evidence.specialization_contract"
)
_DRIVER_PATH = Path(__file__).resolve()
_PLUGIN_PATH = CORPUS_ROOT / "pytest_plugin.py"
_LAUNCHER_PATH = CORPUS_ROOT / "pytest_launcher.py"
_AUDITOR = EVIDENCE_ROOT / "kernel_resources.py"
_SOURCE_AUDITOR = EVIDENCE_ROOT / "source_inventory.py"
_SMEM_CONTRACT_AUDITOR = EVIDENCE_ROOT / "smem_contracts.py"
_CONTRACT_BUILDER = EVIDENCE_ROOT / "specialization_contract.py"
_PTX_CAPTURE = CORPUS_ROOT / "ptx_capture.py"
_SOURCE_ATTESTATION = CORPUS_ROOT / "source_snapshot.py"
_PROJECT_CONFIG = _ROOT / "pyproject.toml"
_MATRIX_SCHEMA = "sparkinfer.cute.migration.corpus_matrix.v1"
_TRACE_SCHEMA = "sparkinfer.cute.migration.case_trace.v1"
_MANIFEST_SCHEMA = "sparkinfer.cute.compile_manifest.v3"
_SOURCE_SNAPSHOT_SCHEMA = "sparkinfer.cute.migration.source_snapshot.v2"
_SMEM_CONTRACT_SCHEMA = "sparkinfer.cute.smem_contracts.v1"
_SMEM_GATE_SCHEMA = "sparkinfer.cute.migration.smem_contract_gate.v1"
_SOURCE_SNAPSHOT_ENV = "CORPUS_FROZEN_SOURCE_MANIFEST"
_SOURCE_SNAPSHOT_SHA256_ENV = "CORPUS_FROZEN_SOURCE_MANIFEST_SHA256"
_SOURCE_SNAPSHOT_FINGERPRINT_ENV = "CORPUS_FROZEN_SOURCE_SNAPSHOT_FINGERPRINT"
_SPARKINFER_FINGERPRINT_ENV = "CORPUS_EXPECTED_SPARKINFER_PACKAGE_FINGERPRINT"
_COMPILE_EVENT_RE = re.compile(
    # Pytest's progress marker can share the line with captured ``-s`` output
    # (for example ``.[sparkinfer cute.compile]``), so the structured marker is the
    # record boundary; a physical-line anchor would silently drop events.
    r"\[sparkinfer cute\.compile\] "
    r"(?P<event>miss|disk-hit|disk-hit-after-wait)\b.*?"
    r"\bcache_key=(?P<prefix>[0-9a-f]{16})\b",
    re.MULTILINE,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CUTLASS_PACKAGES = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
)
_COMPILE_ENV_EXACT = {
    "CC",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CUDAHOSTCXX",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_TOOLKIT_PATH",
    "CUDACXX",
    "CXX",
    "LIBRARY_PATH",
    "NVCC",
}
_COMPILE_ENV_PREFIXES = (
    "SPARKINFER_",
    "CUDA_",
    "CUTE_",
    "CUTLASS_",
    "NVCC_",
    "PTXAS_",
)


class CorpusError(RuntimeError):
    """A corpus completeness or evidence gate failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_ROOT))
    except ValueError:
        return str(resolved)


def _source_tree_snapshot(root: Path, *, root_label: str) -> dict[str, Any]:
    resolved_root = root.resolve()
    if not resolved_root.is_dir() or root.is_symlink():
        raise CorpusError(f"source tree is not a directory: {root}")
    paths = []
    for path in resolved_root.rglob("*"):
        if path.is_symlink():
            raise CorpusError(f"source tree contains a symlink: {path}")
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        paths.append(path)
    paths.sort()

    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in paths:
        relative = str(path.relative_to(resolved_root))
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    return {
        "root": root_label,
        "fingerprint": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _sparkinfer_package_snapshot() -> dict[str, Any]:
    """Reproduce the exact package fingerprint used by the object cache."""
    return _source_tree_snapshot(_ROOT / "sparkinfer", root_label="sparkinfer")


def _source_input_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or path.is_symlink():
        raise CorpusError(f"source input is not a regular file: {path}")
    content = resolved.read_bytes()
    return {
        "path": _source_path(resolved),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _compute_source_snapshot(matrix_path: Path) -> dict[str, Any]:
    payload = {
        "schema": _SOURCE_SNAPSHOT_SCHEMA,
        "sparkinfer_package": _sparkinfer_package_snapshot(),
        # Freeze the complete test and evidence-tool trees.  This deliberately
        # exceeds the current import closure so a helper/auditor edit cannot
        # make the two compiler arms execute different corpus logic.
        "source_trees": {
            "benchmarks": _source_tree_snapshot(
                _ROOT / "benchmarks", root_label="benchmarks"
            ),
            "tests": _source_tree_snapshot(_ROOT / "tests", root_label="tests"),
            "validation": _source_tree_snapshot(
                _ROOT / "validation", root_label="validation"
            ),
        },
        "inputs": {
            "source_inventory": _source_input_record(_INVENTORY),
            "matrix": _source_input_record(matrix_path),
            "driver": _source_input_record(_DRIVER_PATH),
            "plugin": _source_input_record(_PLUGIN_PATH),
            "launcher": _source_input_record(_LAUNCHER_PATH),
            "frontend_ptx_capture": _source_input_record(_PTX_CAPTURE),
            "source_attestation": _source_input_record(_SOURCE_ATTESTATION),
            "family_patterns": _source_input_record(_FAMILY_PATTERNS),
            "symbol_patterns": _source_input_record(_SYMBOL_PATTERNS),
            "resource_auditor": _source_input_record(_AUDITOR),
            "source_auditor": _source_input_record(_SOURCE_AUDITOR),
            "smem_contract_auditor": _source_input_record(_SMEM_CONTRACT_AUDITOR),
            "contract_builder": _source_input_record(_CONTRACT_BUILDER),
            "project_config": _source_input_record(_PROJECT_CONFIG),
        },
    }
    return {**payload, "manifest_sha256": _canonical_sha256(payload)}


def _snapshot_differences(
    expected: dict[str, Any], observed: dict[str, Any]
) -> list[str]:
    differences: list[str] = []
    expected_package = expected["sparkinfer_package"]
    observed_package = observed["sparkinfer_package"]
    if expected_package["fingerprint"] != observed_package["fingerprint"]:
        differences.append(
            "sparkinfer_package:"
            f"{expected_package['fingerprint']}->{observed_package['fingerprint']}"
        )
    expected_inputs = expected["inputs"]
    observed_inputs = observed["inputs"]
    for name in sorted(set(expected_inputs) | set(observed_inputs)):
        old = expected_inputs.get(name, {}).get("sha256", "missing")
        new = observed_inputs.get(name, {}).get("sha256", "missing")
        if old != new:
            differences.append(f"{name}:{old}->{new}")
    if not differences and expected != observed:
        differences.append("source snapshot metadata changed")
    return differences


def _assert_source_snapshot(
    expected: dict[str, Any], matrix_path: Path, *, stage: str
) -> None:
    observed = _compute_source_snapshot(matrix_path)
    if observed != expected:
        raise CorpusError(
            f"frozen source snapshot changed at {stage}: "
            + "; ".join(_snapshot_differences(expected, observed))
        )


def _assert_source_snapshot_artifact(
    path: Path, expected: dict[str, Any], *, stage: str
) -> None:
    observed = _read_json(path)
    if observed != expected:
        raise CorpusError(f"frozen source manifest changed at {stage}: {path}")


def _source_snapshot_environment(
    path: Path, snapshot: dict[str, Any]
) -> dict[str, str]:
    return {
        _SOURCE_SNAPSHOT_ENV: str(path),
        _SOURCE_SNAPSHOT_SHA256_ENV: _sha256(path),
        _SOURCE_SNAPSHOT_FINGERPRINT_ENV: str(snapshot["manifest_sha256"]),
        _SPARKINFER_FINGERPRINT_ENV: str(snapshot["sparkinfer_package"]["fingerprint"]),
    }


def _assert_source_snapshot_environment(
    env: dict[str, str], path: Path, snapshot: dict[str, Any], *, stage: str
) -> None:
    expected = _source_snapshot_environment(path, snapshot)
    mismatches = {
        name: {"expected": value, "observed": env.get(name)}
        for name, value in expected.items()
        if env.get(name) != value
    }
    if mismatches:
        raise CorpusError(
            f"frozen source launcher environment changed at {stage}: {mismatches}"
        )
    _assert_source_snapshot_artifact(path, snapshot, stage=stage)


def _source_snapshot_binding(path: Path, snapshot: dict[str, Any]) -> dict[str, str]:
    return {
        "schema": _SOURCE_SNAPSHOT_SCHEMA,
        "manifest_sha256": str(snapshot["manifest_sha256"]),
        "manifest_artifact_sha256": _sha256(path),
        "sparkinfer_package_fingerprint": str(snapshot["sparkinfer_package"]["fingerprint"]),
    }


def _validate_launcher_source_binding(
    evidence: Any,
    case_id: str,
    snapshot_path: Path,
    snapshot: dict[str, Any],
) -> None:
    if not isinstance(evidence, dict):
        raise CorpusError(f"case {case_id}: invalid launcher evidence")
    artifacts = evidence.get("artifacts")
    launcher = artifacts.get("launcher", {}) if isinstance(artifacts, dict) else {}
    expected = snapshot["inputs"]["launcher"]
    if (
        not isinstance(launcher, dict)
        or Path(str(launcher.get("path", ""))).resolve() != _LAUNCHER_PATH
        or launcher.get("sha256") != expected["sha256"]
    ):
        raise CorpusError(
            f"case {case_id}: launcher does not match frozen source snapshot: "
            f"{launcher!r}"
        )
    attestations = evidence.get("source_attestation")
    if not isinstance(attestations, dict):
        raise CorpusError(f"case {case_id}: launcher lacks source attestations")
    _validate_child_source_attestation(
        attestations.get("pre_runtime"),
        case_id=case_id,
        expected_stage="launcher_pre_runtime",
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
    _validate_child_source_attestation(
        attestations.get("post_pytest"),
        case_id=case_id,
        expected_stage="launcher_post_pytest",
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )


def _validate_child_source_attestation(
    attestation: Any,
    *,
    case_id: str,
    expected_stage: str,
    snapshot_path: Path,
    snapshot: dict[str, Any],
) -> None:
    expected_file_count = int(snapshot["sparkinfer_package"]["file_count"]) + sum(
        int(tree["file_count"]) for tree in snapshot["source_trees"].values()
    )
    expected = {
        "status": "verified",
        "stage": expected_stage,
        "schema": _SOURCE_SNAPSHOT_SCHEMA,
        "repo_root": str(_ROOT.resolve()),
        "manifest_path": str(snapshot_path.resolve()),
        "manifest_artifact_sha256": _sha256(snapshot_path),
        "manifest_sha256": str(snapshot["manifest_sha256"]),
        "sparkinfer_package_fingerprint": str(snapshot["sparkinfer_package"]["fingerprint"]),
        "verified_file_count": expected_file_count,
        "verified_input_count": len(snapshot["inputs"]),
    }
    if not isinstance(attestation, dict) or attestation != expected:
        raise CorpusError(
            f"case {case_id}: invalid {expected_stage} source attestation: "
            f"expected={expected!r} observed={attestation!r}"
        )


def _validate_telemetry_source_binding(
    telemetry: Any,
    *,
    case_id: str,
    snapshot_path: Path,
    snapshot: dict[str, Any],
) -> None:
    source_attestation = (
        telemetry.get("source_attestation") if isinstance(telemetry, dict) else None
    )
    if not isinstance(source_attestation, dict):
        raise CorpusError(f"case {case_id}: telemetry lacks source attestations")
    _validate_child_source_attestation(
        source_attestation.get("pre_collection"),
        case_id=case_id,
        expected_stage="pytest_pre_collection",
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
    _validate_child_source_attestation(
        source_attestation.get("session_finish"),
        case_id=case_id,
        expected_stage="pytest_session_finish",
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )


def _assert_case_source_bindings(
    cases: list[dict[str, Any]], path: Path, snapshot: dict[str, Any]
) -> None:
    expected = _source_snapshot_binding(path, snapshot)
    for case in cases:
        case_id = str(case.get("id", ""))
        if case.get("source_snapshot") != expected:
            raise CorpusError(
                f"case {case_id}: source snapshot binding differs from runner"
            )
        prewarm = case.get("prewarm")
        if not isinstance(prewarm, dict) or prewarm.get("source_snapshot") != expected:
            raise CorpusError(
                f"case {case_id}: prewarm source snapshot binding differs from runner"
            )
        prewarm_evidence = (
            prewarm.get("artifacts", {}).get("launcher_evidence", {}).get("evidence")
        )
        profile_evidence = (
            case.get("artifacts", {}).get("launcher_evidence", {}).get("evidence")
        )
        _validate_launcher_source_binding(prewarm_evidence, case_id, path, snapshot)
        _validate_launcher_source_binding(profile_evidence, case_id, path, snapshot)
        manifests = case.get("manifests")
        if not isinstance(manifests, dict) or any(
            not isinstance(manifest, dict)
            or manifest.get("package_fingerprint")
            != expected["sparkinfer_package_fingerprint"]
            for manifest in manifests.values()
        ):
            raise CorpusError(
                f"case {case_id}: compile manifest differs from frozen sparkinfer source"
            )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"cannot read JSON {path}: {exc}") from exc


def _validate_launcher_ptx_capture(evidence: Any, case_id: str) -> None:
    if not isinstance(evidence, dict):
        raise CorpusError(f"case {case_id}: invalid launcher evidence")
    capture = evidence.get("frontend_ptx_capture_environment")
    if (
        not isinstance(capture, dict)
        or capture.get("enabled") is not True
        or capture.get("verified_before_cutlass_import") is not True
        or "ptx" not in capture.get("cute_dsl_keep_tokens", [])
    ):
        raise CorpusError(
            f"case {case_id}: PTX retention was not verified before CUTLASS import: "
            f"{capture!r}"
        )
    dump_dir = Path(str(capture.get("cute_dsl_dump_dir", "")))
    if not dump_dir.is_absolute():
        raise CorpusError(f"case {case_id}: invalid CUTLASS PTX dump directory")


def _read_patterns(path: Path) -> list[str]:
    try:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except OSError as exc:
        raise CorpusError(f"cannot read pattern file {path}: {exc}") from exc


def _read_inventory() -> list[dict[str, str]]:
    try:
        with _INVENTORY.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
    except OSError as exc:
        raise CorpusError(f"cannot read source inventory: {exc}") from exc
    expected = {
        "status",
        "path",
        "qualified_name",
        "kernel_symbol_glob",
        "reason",
    }
    if not rows or set(rows[0]) != expected:
        raise CorpusError("source inventory is empty or has unexpected columns")
    return rows


def _is_cute_kernel(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Attribute)
        and decorator.attr == "kernel"
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "cute"
    )


def _discover_source_kernels() -> set[str]:
    found: set[str] = set()
    for path in sorted((_ROOT / "sparkinfer").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                names.append(node.name)
                self.generic_visit(node)
                names.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if any(_is_cute_kernel(item) for item in node.decorator_list):
                    relative = path.relative_to(_ROOT)
                    found.add(f"{relative}:{'.'.join((*names, node.name))}")
                names.append(node.name)
                self.generic_visit(node)
                names.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

        Visitor().visit(tree)
    return found


def _require_string_list(case: dict[str, Any], field: str) -> list[str]:
    value = case.get(field)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise CorpusError(f"case {case.get('id')!r}: {field} must be a string list")
    return value


def _validate_nodeid(nodeid: str) -> None:
    file_text, separator, test_text = nodeid.partition("::")
    if not separator or not file_text.startswith("tests/"):
        raise CorpusError(f"invalid GPU pytest node id {nodeid!r}")
    path = _ROOT / file_text
    if not path.is_file():
        raise CorpusError(f"pytest node file does not exist: {nodeid}")
    test_name = test_text.split("::")[-1].split("[", 1)[0]
    # Static source and naming conventions cannot prove execution. This check
    # only resolves the closed nodeid; `_nsys_trace` later requires a CUDA
    # event inside this nodeid's exact, nonoverlapping NVTX range.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if test_name not in functions:
        raise CorpusError(f"pytest function does not exist: {nodeid}")


def validate_matrix(
    matrix_path: Path = _MATRIX,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = _read_json(matrix_path)
    if not isinstance(matrix, dict) or matrix.get("schema") != _MATRIX_SCHEMA:
        raise CorpusError(f"unsupported matrix schema in {matrix_path}")
    if not isinstance(matrix.get("corpus_id"), str) or not matrix["corpus_id"]:
        raise CorpusError("matrix corpus_id is missing")
    if not isinstance(matrix.get("version"), str) or not matrix["version"]:
        raise CorpusError("matrix version is missing")
    requirements = matrix.get("requirements")
    cases = matrix.get("cases")
    if not isinstance(requirements, dict) or not isinstance(cases, list) or not cases:
        raise CorpusError("matrix requirements/cases are missing")
    if requirements.get("physical_gpus") != [4, 5]:
        raise CorpusError("matrix must restrict execution to physical GPUs 4 and 5")
    if requirements.get("compute_capability") != [12, 0]:
        raise CorpusError("matrix must require SM120")
    expected_contract_paths = {
        "source_inventory": repo_relative(_INVENTORY),
        "kernel_id_patterns": repo_relative(_FAMILY_PATTERNS),
        "kernel_symbol_patterns": repo_relative(_SYMBOL_PATTERNS),
    }
    observed_contract_paths = {
        name: requirements.get(name) for name in expected_contract_paths
    }
    if observed_contract_paths != expected_contract_paths:
        raise CorpusError(
            "matrix data paths differ from the packaged contracts: "
            f"expected={expected_contract_paths!r}, "
            f"observed={observed_contract_paths!r}"
        )
    required_branches = requirements.get("shape_branches")
    if not isinstance(required_branches, list) or len(set(required_branches)) != len(
        required_branches
    ):
        raise CorpusError("requirements.shape_branches must be unique")

    inventory = _read_inventory()
    inventory_records = {
        f"{row['path']}:{row['qualified_name']}": row for row in inventory
    }
    if len(inventory_records) != len(inventory):
        raise CorpusError("source inventory contains duplicate entries")
    source_kernels = _discover_source_kernels()
    if source_kernels != set(inventory_records):
        missing = sorted(source_kernels - set(inventory_records))
        stale = sorted(set(inventory_records) - source_kernels)
        raise CorpusError(f"source inventory drift: missing={missing} stale={stale}")
    for source, row in inventory_records.items():
        status = row["status"]
        symbol_glob = row["kernel_symbol_glob"]
        reason = row["reason"]
        if status == "production":
            valid = bool(symbol_glob) and not reason
        elif status == "diagnostic":
            valid = bool(symbol_glob) and bool(reason)
        elif status == "archaeology":
            valid = "/legacy/" in row["path"] and not symbol_glob and bool(reason)
        else:
            raise CorpusError(f"source {source}: invalid inventory status {status!r}")
        if not valid:
            raise CorpusError(
                f"source {source}: invalid {status!r} inventory contract"
            )
    active_inventory_records = {
        source: row
        for source, row in inventory_records.items()
        if row["status"] in {"production", "diagnostic"}
    }

    ids: set[str] = set()
    case_sources: list[str] = []
    case_families: list[str] = []
    case_symbols: list[str] = []
    case_branches: list[str] = []
    gaps: list[dict[str, str]] = []
    nodeids: set[str] = set()
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            raise CorpusError("every case must be an object")
        case_id = raw_case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise CorpusError(f"invalid or duplicate case id {case_id!r}")
        ids.add(case_id)
        status = raw_case.get("status")
        state = raw_case.get("coverage_state")
        if status not in {"production", "diagnostic"}:
            raise CorpusError(f"case {case_id}: invalid status {status!r}")
        if state not in {"ready", "gap"}:
            raise CorpusError(f"case {case_id}: invalid coverage_state {state!r}")
        gap_reason = raw_case.get("gap_reason", "")
        if state == "gap":
            if not isinstance(gap_reason, str) or not gap_reason:
                raise CorpusError(f"case {case_id}: gap requires a reason")
            gaps.append({"id": case_id, "reason": gap_reason})
        elif gap_reason:
            raise CorpusError(f"case {case_id}: ready case cannot have gap_reason")

        sources = _require_string_list(raw_case, "source_kernels")
        families = _require_string_list(raw_case, "kernel_id_patterns")
        symbols = _require_string_list(raw_case, "kernel_symbol_patterns")
        branches = _require_string_list(raw_case, "shape_branches")
        tests = _require_string_list(raw_case, "pytest_nodeids")
        if len(set(tests)) != len(tests):
            raise CorpusError(f"case {case_id}: duplicate pytest nodeids")
        correctness = raw_case.get("correctness")
        graph = raw_case.get("graph")
        if not isinstance(correctness, dict) or not isinstance(graph, dict):
            raise CorpusError(f"case {case_id}: correctness/graph records are required")
        branch_evidence = raw_case.get("shape_branch_evidence", {})
        if not isinstance(branch_evidence, dict) or any(
            not isinstance(key, str) or not isinstance(value, dict)
            for key, value in branch_evidence.items()
        ):
            raise CorpusError(f"case {case_id}: invalid shape_branch_evidence")
        if not set(branch_evidence) <= set(branches):
            raise CorpusError(
                f"case {case_id}: evidence names an unassigned shape branch"
            )
        if state == "ready" and set(branch_evidence) != set(branches):
            raise CorpusError(
                f"ready case {case_id}: every shape branch needs machine evidence"
            )
        for branch, evidence in branch_evidence.items():
            allowed_evidence_fields = {
                "correctness_nodeids",
                "graph_nodeids",
                "manifest_predicates",
                "shared_manifest_group",
            }
            if not set(evidence) <= allowed_evidence_fields:
                raise CorpusError(
                    f"case {case_id}: unsupported evidence field for {branch}"
                )
            if "nodeids" in evidence:
                raise CorpusError(
                    f"case {case_id}: branch {branch} uses ambiguous legacy nodeids; "
                    "use correctness_nodeids and graph_nodeids"
                )
            correctness_nodeids = evidence.get("correctness_nodeids", [])
            graph_nodeids = evidence.get("graph_nodeids", [])
            predicates = evidence.get("manifest_predicates", [])
            shared_manifest_group = evidence.get("shared_manifest_group")
            if (
                not isinstance(correctness_nodeids, list)
                or any(
                    not isinstance(nodeid, str) or nodeid not in tests
                    for nodeid in correctness_nodeids
                )
                or len(set(correctness_nodeids)) != len(correctness_nodeids)
                or not isinstance(graph_nodeids, list)
                or any(
                    not isinstance(nodeid, str) or nodeid not in tests
                    for nodeid in graph_nodeids
                )
                or len(set(graph_nodeids)) != len(graph_nodeids)
                or not isinstance(predicates, list)
                or any(not isinstance(item, dict) or not item for item in predicates)
                or (
                    shared_manifest_group is not None
                    and (
                        not isinstance(shared_manifest_group, str)
                        or not shared_manifest_group
                    )
                )
            ):
                raise CorpusError(f"case {case_id}: invalid evidence for {branch}")
            if not correctness_nodeids and not graph_nodeids and not predicates:
                raise CorpusError(f"case {case_id}: empty evidence for {branch}")
            if state == "ready" and status == "production":
                case_correctness_nodeids = correctness.get("evidence_nodeids", [])
                case_graph_nodeids = graph.get("evidence_nodeids", [])
                if (
                    not correctness_nodeids
                    or not graph_nodeids
                    or not predicates
                    or not set(correctness_nodeids) <= set(case_correctness_nodeids)
                    or not set(graph_nodeids) <= set(case_graph_nodeids)
                ):
                    raise CorpusError(
                        f"ready production case {case_id}: branch {branch} must bind "
                        "nonempty case-level correctness_nodeids, graph_nodeids, "
                        "and manifest_predicates"
                    )
            allowed_predicates = {
                "semantic_key",
                "compile_spec_hash",
                "compile_spec_version",
                "kernel_id",
                "kernel_id_glob",
                "target",
                "target_glob",
                "facts_prefix",
            }
            for predicate in predicates:
                if not set(predicate) <= allowed_predicates:
                    raise CorpusError(
                        f"case {case_id}: unsupported manifest predicate for {branch}"
                    )
        if status == "diagnostic" and len(sources) != 1:
            raise CorpusError(
                f"diagnostic case {case_id} must contain one source entry"
            )
        for source in sources:
            row = active_inventory_records.get(source)
            if row is None:
                raise CorpusError(f"case {case_id}: unknown source kernel {source}")
            if row["status"] != status:
                raise CorpusError(
                    f"case {case_id}: source {source} has inventory status "
                    f"{row['status']!r}, not {status!r}"
                )
        if status == "production":
            if correctness.get("required") is not True:
                raise CorpusError(f"case {case_id}: GPU correctness is mandatory")
            if correctness.get("oracle") not in {
                "gpu_reference",
                "exact_gpu_reference",
                "gpu_dequant_reference",
            }:
                raise CorpusError(f"case {case_id}: invalid GPU oracle")
            if graph.get("required") is not True:
                raise CorpusError(f"case {case_id}: CUDA graph proof is mandatory")
        elif correctness.get("serving_claim") is not False:
            raise CorpusError(f"diagnostic case {case_id} cannot make a serving claim")
        if state == "ready":
            if not tests:
                raise CorpusError(f"ready case {case_id} has no GPU tests")
            if int(raw_case.get("min_compile_events", 0)) <= 0:
                raise CorpusError(f"ready case {case_id} requires compile evidence")
            for evidence_field in ("evidence_nodeids",):
                correctness_nodes = correctness.get(evidence_field)
                graph_nodes = graph.get(evidence_field)
                if status == "production" and (
                    not isinstance(correctness_nodes, list)
                    or not correctness_nodes
                    or not isinstance(graph_nodes, list)
                    or not graph_nodes
                ):
                    raise CorpusError(
                        f"case {case_id}: correctness and graph evidence nodeids required"
                    )
        for nodeid in tests:
            _validate_nodeid(nodeid)
            nodeids.add(nodeid)
        case_sources.extend(sources)
        case_families.extend(families)
        case_symbols.extend(symbols)
        case_branches.extend(branches)

    duplicate_sources = sorted(
        item for item, count in Counter(case_sources).items() if count != 1
    )
    if set(case_sources) != set(active_inventory_records) or duplicate_sources:
        raise CorpusError(
            "matrix must assign every source kernel exactly once: "
            f"duplicates={duplicate_sources} "
            f"missing={sorted(set(active_inventory_records) - set(case_sources))}"
        )
    expected_families = _read_patterns(_FAMILY_PATTERNS)
    if Counter(case_families) != Counter(expected_families):
        raise CorpusError(
            "matrix compile-family patterns differ from reviewed coverage"
        )
    expected_symbols = _read_patterns(_SYMBOL_PATTERNS)
    if Counter(case_symbols) != Counter(expected_symbols):
        raise CorpusError("matrix symbol patterns differ from reviewed coverage")
    if Counter(case_branches) != Counter(required_branches):
        raise CorpusError("matrix cases do not assign every required shape branch once")

    hashes = {
        "sparkinfer_package_fingerprint": _sparkinfer_package_snapshot()["fingerprint"],
        "matrix_raw_sha256": _sha256(matrix_path),
        "matrix_canonical_sha256": _canonical_sha256(matrix),
        "source_inventory_sha256": _sha256(_INVENTORY),
        "kernel_id_patterns_sha256": _sha256(_FAMILY_PATTERNS),
        "kernel_symbol_patterns_sha256": _sha256(_SYMBOL_PATTERNS),
        "runner_sha256": _sha256(_DRIVER_PATH),
        "launcher_sha256": _sha256(_LAUNCHER_PATH),
        "plugin_sha256": _sha256(_PLUGIN_PATH),
        "frontend_ptx_capture_sha256": _sha256(_PTX_CAPTURE),
        "smem_contract_auditor_sha256": _sha256(_SMEM_CONTRACT_AUDITOR),
    }
    report = {
        "schema": "sparkinfer.cute.migration.static_validation.v1",
        "corpus_id": matrix["corpus_id"],
        "corpus_version": matrix["version"],
        "case_count": len(cases),
        "ready_case_count": len(cases) - len(gaps),
        "gap_case_count": len(gaps),
        "production_source_kernel_count": sum(
            row["status"] == "production" for row in inventory
        ),
        "diagnostic_source_kernel_count": sum(
            row["status"] == "diagnostic" for row in inventory
        ),
        "archaeology_source_kernel_count": sum(
            row["status"] == "archaeology" for row in inventory
        ),
        "compile_family_pattern_count": len(expected_families),
        "kernel_symbol_pattern_count": len(expected_symbols),
        "shape_branch_count": len(required_branches),
        "declared_ready_shape_branch_evidence_count": sum(
            len(case.get("shape_branch_evidence", {}))
            for case in cases
            if case["coverage_state"] == "ready"
        ),
        "uncovered_shape_branches": sorted(
            branch
            for case in cases
            if case["coverage_state"] != "ready"
            for branch in case["shape_branches"]
        ),
        "pytest_nodeid_count": len(nodeids),
        "gpu_execution_contract": {
            "static_nodeid_validation": "resolvable_only",
            "runtime_proof": "nsys_exact_per_test_nvtx_cuda_event",
            "runtime_proof_required": True,
        },
        "gaps": gaps,
        "hashes": hashes,
    }
    return matrix, report


def _assert_validation_matches_snapshot(
    validation: dict[str, Any], snapshot: dict[str, Any]
) -> None:
    hashes = validation.get("hashes", {})
    inputs = snapshot["inputs"]
    expected = {
        "sparkinfer_package_fingerprint": snapshot["sparkinfer_package"]["fingerprint"],
        "matrix_raw_sha256": inputs["matrix"]["sha256"],
        "source_inventory_sha256": inputs["source_inventory"]["sha256"],
        "runner_sha256": inputs["driver"]["sha256"],
        "launcher_sha256": inputs["launcher"]["sha256"],
        "plugin_sha256": inputs["plugin"]["sha256"],
        "frontend_ptx_capture_sha256": inputs["frontend_ptx_capture"]["sha256"],
        "smem_contract_auditor_sha256": inputs["smem_contract_auditor"]["sha256"],
        "kernel_id_patterns_sha256": inputs["family_patterns"]["sha256"],
        "kernel_symbol_patterns_sha256": inputs["symbol_patterns"]["sha256"],
    }
    mismatches = {
        name: {"expected": value, "observed": hashes.get(name)}
        for name, value in expected.items()
        if hashes.get(name) != value
    }
    if mismatches:
        raise CorpusError(
            f"static validation used a different source snapshot: {mismatches}"
        )


def _ensure_empty(path: Path, label: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise CorpusError(f"{label} is not a directory: {path}")
        if any(path.iterdir()):
            raise CorpusError(f"{label} must be verified empty: {path}")
    else:
        path.mkdir(parents=True)


def _canonical_subprocess_environment(
    source: dict[str, str], *, cuda_path: Path, ptx_dump_dir: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    """Remove ambient compiler controls, then install one declared map."""
    removed: dict[str, str] = {}
    env: dict[str, str] = {}
    for name, value in source.items():
        compile_control = name in _COMPILE_ENV_EXACT or name.startswith(
            _COMPILE_ENV_PREFIXES
        )
        if compile_control:
            removed[name] = value
        else:
            env[name] = value
    canonical = {
        "CUDA_PATH": str(cuda_path),
        "CUTE_DSL_ARCH": "sm_120a",
        # CUTLASS's function-derived dump filename is only a staging path.  The
        # pytest plugin copies each final PTX immediately to its sparkinfer cache-key
        # sidecar.  The capture hook excludes KEEP/DUMP_DIR from semantic
        # compile identity because they do not change generated code.
        "CUTE_DSL_KEEP": "ptx",
        "CUTE_DSL_DUMP_DIR": str(ptx_dump_dir),
        "CORPUS_RETAIN_FRONTEND_PTX": "1",
        "CORPUS_COMMON_PTXAS": str(cuda_path / "bin" / "ptxas"),
        "CORPUS_NVDISASM": str(cuda_path / "bin" / "nvdisasm"),
    }
    env.update(canonical)
    record = {
        "schema": "sparkinfer.cute.migration.canonical_compile_environment.v1",
        "canonical": canonical,
        "removed_names": sorted(removed),
        "removed_value_sha256": {
            name: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for name, value in sorted(removed.items())
        },
        "explicitly_absent": sorted(
            {
                "CUTE_DSL_LIBS",
                "CUDA_HOME",
                "CUDA_TOOLKIT_PATH",
                "CUDACXX",
                "NVCC_APPEND_FLAGS",
                "NVCC_PREPEND_FLAGS",
            }
            - set(canonical)
        ),
        "measurement_only_semantic_identity_exclusions": [
            "CUTE_DSL_DUMP_DIR",
            "CUTE_DSL_KEEP",
        ],
    }
    return env, record


def _run_checked(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if log_path is not None:
        log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise CorpusError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{tail}"
        )
    return result


def _run_smem_contract_audit() -> tuple[dict[str, Any], str]:
    """Run and independently validate the source-only CUTLASS 4.6 SMEM gate."""

    result = _run_checked(
        [
            sys.executable,
            "-m",
            _SMEM_CONTRACT_AUDITOR_MODULE,
            "--root",
            str(_ROOT),
            "--format",
            "json",
        ]
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CorpusError(
            f"SMEM contract auditor did not emit JSON: {exc}"
        ) from exc
    if not isinstance(report, dict) or report.get("schema") != _SMEM_CONTRACT_SCHEMA:
        raise CorpusError("SMEM contract auditor emitted an invalid schema")
    if report.get("root") != str(_ROOT.resolve()):
        raise CorpusError(
            "SMEM contract auditor inspected the wrong source root: "
            f"{report.get('root')!r}"
        )
    if report.get("passed") is not True:
        raise CorpusError("SMEM contract audit did not pass")
    rows = report.get("rows")
    counts = report.get("counts")
    source_counts = report.get("source_counts")
    if (
        not isinstance(rows, list)
        or not isinstance(counts, dict)
        or not isinstance(source_counts, dict)
        or report.get("audited_source_roots")
        != {
            "production": ["sparkinfer"],
            "infrastructure": ["benchmarks", "tests", "validation"],
        }
    ):
        raise CorpusError("SMEM contract audit lacks machine-readable rows/counts")
    required_counts = {
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
    if set(counts) != required_counts or any(
        type(counts[name]) is not int or counts[name] < 0 for name in required_counts
    ):
        raise CorpusError("SMEM contract audit has invalid summary counts")
    if set(source_counts) != {"production", "infrastructure"} or any(
        not isinstance(category_counts, dict)
        or set(category_counts) != required_counts
        or any(
            type(category_counts[name]) is not int or category_counts[name] < 0
            for name in required_counts
        )
        for category_counts in source_counts.values()
    ):
        raise CorpusError("SMEM contract audit has invalid per-source counts")
    if any(not isinstance(row, dict) for row in rows):
        raise CorpusError("SMEM contract audit contains a non-object row")
    allocator_rows = [row for row in rows if row.get("kind") == "allocator"]
    private_rows = [row for row in rows if row.get("kind") == "private_memrange"]
    parse_rows = [row for row in rows if row.get("kind") == "parse_error"]
    expected_relations = (
        counts["allocator_count"] == len(allocator_rows),
        counts["allocator_pass_count"] == len(allocator_rows),
        counts["allocator_fail_count"] == 0,
        counts["allocation_call_count"] == len(allocator_rows),
        counts["allocate_call_count"]
        == sum(row.get("allocation_methods") == ["allocate"] for row in allocator_rows),
        counts["allocate_tensor_call_count"]
        == sum(
            row.get("allocation_methods") == ["allocate_tensor"]
            for row in allocator_rows
        ),
        counts["allocation_call_count"]
        == counts["allocate_call_count"] + counts["allocate_tensor_call_count"],
        counts["private_memrange_identifier_count"] == len(private_rows),
        counts["private_memrange_centralized_count"] == len(private_rows),
        counts["private_memrange_outside_count"] == 0,
        counts["parse_error_count"] == len(parse_rows) == 0,
        counts["violation_count"] == 0,
        counts["row_count"] == len(rows),
        len(allocator_rows) > 0,
        source_counts["production"]["allocator_count"] > 0,
        source_counts["infrastructure"]["allocator_count"] > 0,
        source_counts["infrastructure"]["allocate_tensor_call_count"] > 0,
    )
    if not all(expected_relations):
        raise CorpusError(f"SMEM contract audit counts are inconsistent: {counts}")
    for source_category, category_counts in source_counts.items():
        category_rows = [
            row for row in rows if row.get("source_category") == source_category
        ]
        category_allocators = [
            row for row in category_rows if row.get("kind") == "allocator"
        ]
        category_private = [
            row for row in category_rows if row.get("kind") == "private_memrange"
        ]
        category_parse = [
            row for row in category_rows if row.get("kind") == "parse_error"
        ]
        recomputed = {
            "allocator_count": len(category_allocators),
            "allocator_pass_count": sum(
                row.get("passed") is True for row in category_allocators
            ),
            "allocator_fail_count": sum(
                row.get("passed") is not True for row in category_allocators
            ),
            "allocation_call_count": sum(
                int(row.get("allocation_count", -1)) for row in category_allocators
            ),
            "allocate_call_count": sum(
                row.get("allocation_methods") == ["allocate"]
                for row in category_allocators
            ),
            "allocate_tensor_call_count": sum(
                row.get("allocation_methods") == ["allocate_tensor"]
                for row in category_allocators
            ),
            "private_memrange_identifier_count": len(category_private),
            "private_memrange_centralized_count": sum(
                row.get("centralized") is True for row in category_private
            ),
            "private_memrange_outside_count": sum(
                row.get("centralized") is not True for row in category_private
            ),
            "parse_error_count": len(category_parse),
            "violation_count": sum(
                len(row.get("violations", [])) for row in category_rows
            ),
            "row_count": len(category_rows),
        }
        if any(category_counts[name] != value for name, value in recomputed.items()):
            raise CorpusError(
                "SMEM contract audit per-source counts are inconsistent: "
                f"{source_category}={category_counts}"
            )
    if any(
        counts[name]
        != source_counts["production"][name] + source_counts["infrastructure"][name]
        for name in required_counts
    ):
        raise CorpusError("SMEM contract audit total/per-source counts differ")
    for row in allocator_rows:
        allocation_methods = row.get("allocation_methods")
        method_shape_passed = (
            allocation_methods == ["allocate"]
            and row.get("allocation_argument_counts") == [1]
            and row.get("allocation_keyword_counts") == [0]
        ) or (
            allocation_methods == ["allocate_tensor"]
            and row.get("allocation_argument_counts") == [0]
            and row.get("allocation_keyword_counts") == [3]
        )
        if not (
            row.get("passed") is True
            and row.get("violations") == []
            and row.get("source_category") in {"production", "infrastructure"}
            and row.get("constructor_argument_count") == 0
            and row.get("constructor_keyword_count") == 0
            and row.get("allocator_store_count") == 1
            and row.get("allocation_count") == 1
            and row.get("typed_allocation") is True
            and row.get("allocation_after_constructor") is True
            and row.get("allocation_result_bound_locally") is True
            and method_shape_passed
        ):
            raise CorpusError(
                "SMEM contract audit has a fail-open allocator row: "
                f"{row.get('path')}:{row.get('line')}"
            )
    if any(
        row.get("passed") is not True
        or row.get("centralized") is not True
        or row.get("violations") != []
        for row in private_rows
    ):
        raise CorpusError("SMEM contract audit has a private _MemRange escape")
    report_sha256 = hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()
    binding = {
        "schema": _SMEM_GATE_SCHEMA,
        "passed": True,
        "auditor": {
            "path": _source_path(_SMEM_CONTRACT_AUDITOR),
            "sha256": _sha256(_SMEM_CONTRACT_AUDITOR),
        },
        "report_schema": _SMEM_CONTRACT_SCHEMA,
        "report_sha256": report_sha256,
        "counts": counts,
        "report": report,
    }
    return binding, result.stdout


def _gpu_record(physical_gpu: int) -> dict[str, Any]:
    result = _run_checked(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ]
    )
    records: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3:
            records[int(parts[0])] = {"uuid": parts[1], "name": parts[2]}
    if physical_gpu not in records:
        raise CorpusError(f"physical GPU {physical_gpu} was not reported by nvidia-smi")
    return {"physical_ordinal": physical_gpu, **records[physical_gpu]}


def _cache_manifests(cache_dir: Path) -> dict[str, Path]:
    manifests: dict[str, Path] = {}
    for path in cache_dir.rglob("*.json"):
        if _SHA256_RE.fullmatch(path.stem):
            manifests[path.stem] = path
    return manifests


def _resolve_event_prefixes(prefixes: list[str], keys: set[str]) -> list[str]:
    resolved: list[str] = []
    for prefix in prefixes:
        matches = sorted(key for key in keys if key.startswith(prefix))
        if len(matches) != 1:
            raise CorpusError(
                f"compile event prefix {prefix!r} resolves to {len(matches)} cache keys"
            )
        resolved.append(matches[0])
    return resolved


def _validate_manifest(
    path: Path, cache_key: str, *, expected_package_fingerprint: str
) -> dict[str, Any]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema") != _MANIFEST_SCHEMA:
        raise CorpusError(f"invalid compile manifest schema: {path}")
    if manifest.get("cache_key") != cache_key:
        raise CorpusError(f"manifest/cache-key mismatch: {path}")
    if manifest.get("package_fingerprint") != expected_package_fingerprint:
        raise CorpusError(
            f"manifest package fingerprint differs from frozen source: {path}"
        )
    for field in (
        "semantic_key",
        "target",
        "kernel_id",
        "compile_spec_version",
        "compile_spec_hash",
        "compile_spec_json",
    ):
        if manifest.get(field) in {None, ""}:
            raise CorpusError(f"manifest {path} lacks {field}")
    object_path = path.with_suffix(".o")
    if not object_path.is_file():
        raise CorpusError(f"manifest has no object: {path}")
    if manifest.get("object_sha256") != _sha256(object_path):
        raise CorpusError(f"manifest object SHA-256 mismatch: {path}")
    return manifest


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _test_nvtx_label(case_id: str, nodeid: str) -> str:
    identity = hashlib.sha256(f"{case_id}\0{nodeid}".encode()).hexdigest()
    return f"sparkinfer-cute-corpus-test:{case_id}:{identity}"


def _nsys_trace(
    sqlite_path: Path,
    case_id: str,
    expected_test_ranges: list[dict[str, Any]],
) -> dict[str, Any]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"}
        if not required <= tables:
            raise CorpusError(f"Nsight SQLite lacks tables {sorted(required - tables)}")
        strings = {
            int(row[0]): str(row[1])
            for row in connection.execute("SELECT id, value FROM StringIds")
        }

        def resolve(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, int):
                return strings.get(value, str(value))
            text = str(value)
            if text.isdigit() and int(text) in strings:
                return strings[int(text)]
            return text

        nvtx_columns = _sqlite_columns(connection, "NVTX_EVENTS")
        nvtx_rows = connection.execute("SELECT * FROM NVTX_EVENTS").fetchall()
        case_label = f"sparkinfer-cute-corpus-case:{case_id}"
        nvtx_by_label: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in nvtx_rows:
            nvtx_text = str(row["text"] or "") if "text" in nvtx_columns else ""
            if not nvtx_text and "textId" in nvtx_columns:
                nvtx_text = resolve(row["textId"])
            if nvtx_text and row["start"] is not None and row["end"] is not None:
                nvtx_by_label[nvtx_text].append((int(row["start"]), int(row["end"])))
        case_ranges = nvtx_by_label.get(case_label, [])
        if len(case_ranges) != 1:
            raise CorpusError(
                f"expected one completed NVTX case range, found {case_ranges}"
            )
        range_start, range_end = case_ranges[0]

        expected_labels: dict[str, str] = {}
        for record in expected_test_ranges:
            if not isinstance(record, dict):
                raise CorpusError(f"case {case_id}: invalid test NVTX telemetry")
            nodeid = record.get("nodeid")
            test_label = record.get("label")
            if (
                not isinstance(nodeid, str)
                or not nodeid
                or not isinstance(test_label, str)
                or test_label != _test_nvtx_label(case_id, nodeid)
                or record.get("completed") is not True
                or nodeid in expected_labels
            ):
                raise CorpusError(
                    f"case {case_id}: invalid test NVTX record {record!r}"
                )
            expected_labels[nodeid] = test_label
        if len(set(expected_labels.values())) != len(expected_labels):
            raise CorpusError(f"case {case_id}: duplicate test NVTX labels")
        observed_test_labels = {
            nvtx_label
            for nvtx_label in nvtx_by_label
            if nvtx_label.startswith(f"sparkinfer-cute-corpus-test:{case_id}:")
        }
        if observed_test_labels != set(expected_labels.values()):
            raise CorpusError(
                f"case {case_id}: test NVTX label mismatch: "
                f"missing={sorted(set(expected_labels.values()) - observed_test_labels)} "
                f"unexpected={sorted(observed_test_labels - set(expected_labels.values()))}"
            )
        test_ranges: dict[str, dict[str, Any]] = {}
        for nodeid, test_label in expected_labels.items():
            matches = nvtx_by_label[test_label]
            if len(matches) != 1:
                raise CorpusError(
                    f"case {case_id}: expected one completed range for {nodeid}, "
                    f"found {matches}"
                )
            test_start, test_end = matches[0]
            if not (range_start <= test_start < test_end <= range_end):
                raise CorpusError(
                    f"case {case_id}: test range {nodeid} is not nested in case range"
                )
            test_ranges[nodeid] = {
                "nvtx_label": test_label,
                "range_start_ns": test_start,
                "range_end_ns": test_end,
                "cuda_events": [],
            }
        ordered_test_ranges = sorted(
            (
                int(record["range_start_ns"]),
                int(record["range_end_ns"]),
                nodeid,
            )
            for nodeid, record in test_ranges.items()
        )
        for previous, current in zip(
            ordered_test_ranges, ordered_test_ranges[1:], strict=False
        ):
            if current[0] < previous[1]:
                raise CorpusError(
                    f"case {case_id}: overlapping test NVTX ranges "
                    f"{previous[2]!r} and {current[2]!r}"
                )

        kernel_columns = _sqlite_columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
        events: list[dict[str, Any]] = []
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            start = int(row["start"])
            end = int(row["end"])
            if end <= start:
                raise CorpusError(
                    f"case {case_id}: invalid CUDA event interval "
                    f"start={start} end={end}"
                )
            overlaps_case = start < range_end and end > range_start
            if overlaps_case and not (range_start <= start and end <= range_end):
                raise CorpusError(
                    f"case {case_id}: CUDA event crosses the case NVTX boundary: "
                    f"start={start} end={end}"
                )
            if not overlaps_case:
                continue
            event: dict[str, Any] = {"start": start, "end": end}
            for field in ("shortName", "demangledName", "mangledName"):
                event[field] = resolve(row[field]) if field in kernel_columns else ""
            for field in (
                "deviceId",
                "contextId",
                "streamId",
                "gridX",
                "gridY",
                "gridZ",
                "blockX",
                "blockY",
                "blockZ",
                "staticSharedMemory",
                "dynamicSharedMemory",
                "localMemoryPerThread",
                "localMemoryTotal",
                "registersPerThread",
            ):
                if field in kernel_columns and row[field] is not None:
                    event[field] = int(row[field])
            events.append(event)
            overlapping = [
                nodeid
                for test_start, test_end, nodeid in ordered_test_ranges
                if start < test_end and end > test_start
            ]
            contained = [
                nodeid
                for test_start, test_end, nodeid in ordered_test_ranges
                if test_start <= start and end <= test_end
            ]
            if overlapping and len(contained) != 1:
                raise CorpusError(
                    f"case {case_id}: CUDA event crosses or ambiguously occupies "
                    f"test ranges: start={start} end={end} overlap={overlapping}"
                )
            if contained:
                test_ranges[contained[0]]["cuda_events"].append(event)
        if not events:
            raise CorpusError(f"case {case_id} has no CUDA launches in its NVTX range")
        empty_test_ranges = sorted(
            nodeid
            for nodeid, record in test_ranges.items()
            if not record["cuda_events"]
        )
        if empty_test_ranges:
            raise CorpusError(
                f"case {case_id}: GPU-only corpus tests have no CUDA launches in "
                f"their exact ranges: {empty_test_ranges}"
            )
        return {
            "nvtx_label": case_label,
            "range_start_ns": range_start,
            "range_end_ns": range_end,
            "cuda_events": events,
            "test_ranges": test_ranges,
        }
    finally:
        connection.close()


def _reports_cover_nodeid(reports: list[dict[str, Any]], expected: str) -> bool:
    return any(
        str(report.get("nodeid", "")) == expected
        and report.get("when") == "call"
        and report.get("outcome") == "passed"
        for report in reports
    )


def _manifest_matches_predicate(
    manifest: dict[str, Any], predicate: dict[str, Any]
) -> bool:
    for field in (
        "semantic_key",
        "compile_spec_hash",
        "compile_spec_version",
        "kernel_id",
        "target",
    ):
        expected = predicate.get(field)
        if expected is not None and str(manifest.get(field, "")) != str(expected):
            return False
    kernel_glob = predicate.get("kernel_id_glob")
    if kernel_glob is not None and not fnmatch.fnmatchcase(
        str(manifest.get("kernel_id", "")), str(kernel_glob)
    ):
        return False
    target_glob = predicate.get("target_glob")
    if target_glob is not None and not fnmatch.fnmatchcase(
        str(manifest.get("target", "")), str(target_glob)
    ):
        return False
    facts_prefix = predicate.get("facts_prefix")
    if facts_prefix is not None:
        try:
            compile_spec = json.loads(str(manifest["compile_spec_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            return False
        facts = compile_spec.get("facts") if isinstance(compile_spec, dict) else None
        if not isinstance(facts, list) or not isinstance(facts_prefix, list):
            return False
        if facts[: len(facts_prefix)] != facts_prefix:
            return False
    return True


def _validate_shape_branch_evidence(
    case: dict[str, Any],
    reports: list[dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    proof: dict[str, Any] = {}
    for branch in case["shape_branches"]:
        evidence = case.get("shape_branch_evidence", {}).get(branch)
        if not isinstance(evidence, dict):
            raise CorpusError(f"case {case['id']}: branch {branch} has no evidence")
        correctness_nodeids = evidence.get("correctness_nodeids", [])
        graph_nodeids = evidence.get("graph_nodeids", [])
        missing_correctness_nodeids = [
            nodeid
            for nodeid in correctness_nodeids
            if not _reports_cover_nodeid(reports, nodeid)
        ]
        missing_graph_nodeids = [
            nodeid
            for nodeid in graph_nodeids
            if not _reports_cover_nodeid(reports, nodeid)
        ]
        if missing_correctness_nodeids or missing_graph_nodeids:
            raise CorpusError(
                f"case {case['id']}: branch {branch} missing correctness nodeids "
                f"{missing_correctness_nodeids} or graph nodeids "
                f"{missing_graph_nodeids}"
            )
        predicate_matches: list[dict[str, Any]] = []
        selected_cache_keys: list[str] = []
        for predicate in evidence.get("manifest_predicates", []):
            keys = sorted(
                key
                for key, manifest in manifests.items()
                if _manifest_matches_predicate(manifest, predicate)
            )
            if len(keys) != 1:
                raise CorpusError(
                    f"case {case['id']}: branch {branch} predicate must match "
                    f"exactly one used manifest, matched {keys}: {predicate}"
                )
            predicate_matches.append({"predicate": predicate, "cache_keys": keys})
            selected_cache_keys.extend(keys)
        duplicate_keys = sorted(
            key for key, count in Counter(selected_cache_keys).items() if count != 1
        )
        if duplicate_keys:
            raise CorpusError(
                f"case {case['id']}: branch {branch} predicates multiply select "
                f"cache keys {duplicate_keys}"
            )
        proof[branch] = {
            "passed_correctness_nodeids": correctness_nodeids,
            "passed_graph_nodeids": graph_nodeids,
            "manifest_predicate_matches": predicate_matches,
            "selected_cache_keys": sorted(selected_cache_keys),
            "shared_manifest_group": evidence.get("shared_manifest_group"),
        }
    return proof


def _pytest_launcher_command(case: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        _LAUNCHER,
        "-q",
        "-s",
        "-p",
        _PLUGIN,
        *case["pytest_nodeids"],
    ]


def _prewarm_case(
    case: dict[str, Any],
    *,
    cache_dir: Path,
    case_dir: Path,
    base_env: dict[str, str],
    matrix_path: Path,
    source_snapshot_path: Path,
    source_snapshot: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """Compile the exact case outside Nsight, then prove its cache identity.

    Profiling a fresh compile tree injects CUPTI into compiler helper processes.
    On the supported host this can strand a helper after it exits and leave
    ``subprocess.communicate`` blocked forever.  A separate process also makes
    the profiled run exercise the production disk-cache load path rather than
    inheriting in-memory compiled callables.
    """
    before = _cache_manifests(cache_dir)
    telemetry_path = case_dir / "prewarm-pytest-telemetry.json"
    launcher_evidence_path = case_dir / "prewarm-launcher-evidence.json"
    log_path = case_dir / "prewarm-pytest.log"
    env = dict(base_env)
    env.update(
        {
            "CORPUS_CASE_ID": str(case["id"]),
            "CORPUS_TELEMETRY": str(telemetry_path),
            "CORPUS_LAUNCHER_EVIDENCE": str(launcher_evidence_path),
        }
    )
    case_id = str(case["id"])
    _assert_source_snapshot(
        source_snapshot, matrix_path, stage=f"case {case_id} prewarm before launcher"
    )
    _assert_source_snapshot_environment(
        env,
        source_snapshot_path,
        source_snapshot,
        stage=f"case {case_id} prewarm before launcher",
    )
    _run_checked(
        _pytest_launcher_command(case),
        env=env,
        timeout=timeout,
        log_path=log_path,
    )
    _assert_source_snapshot(
        source_snapshot, matrix_path, stage=f"case {case_id} prewarm after launcher"
    )
    _assert_source_snapshot_environment(
        env,
        source_snapshot_path,
        source_snapshot,
        stage=f"case {case_id} prewarm after launcher",
    )
    telemetry = _read_json(telemetry_path)
    launcher_evidence = _read_json(launcher_evidence_path)
    if (
        telemetry.get("schema") != "sparkinfer.cute.migration.pytest_telemetry.v3"
        or int(telemetry.get("exitstatus", -1)) != 0
    ):
        raise CorpusError(f"case {case['id']}: invalid prewarm pytest telemetry")
    prewarm_capture = telemetry.get("frontend_ptx_capture", {})
    if (
        prewarm_capture.get("status") != "ok"
        or prewarm_capture.get("redisassembled") is not True
    ):
        raise CorpusError(f"case {case['id']}: incomplete prewarm frontend PTX capture")
    _validate_launcher_ptx_capture(launcher_evidence, str(case["id"]))
    _validate_launcher_source_binding(
        launcher_evidence, case_id, source_snapshot_path, source_snapshot
    )
    _validate_telemetry_source_binding(
        telemetry,
        case_id=case_id,
        snapshot_path=source_snapshot_path,
        snapshot=source_snapshot,
    )

    after = _cache_manifests(cache_dir)
    log_text = log_path.read_text(encoding="utf-8")
    events = [match.groupdict() for match in _COMPILE_EVENT_RE.finditer(log_text)]
    event_keys = _resolve_event_prefixes(
        [event["prefix"] for event in events], set(after)
    )
    used_keys = sorted((set(after) - set(before)) | set(event_keys))
    if len(events) < int(case["min_compile_events"]) or not used_keys:
        raise CorpusError(f"case {case['id']}: insufficient prewarm cache evidence")
    return {
        "compile_event_count": len(events),
        "miss_event_count": sum(event["event"] == "miss" for event in events),
        "disk_hit_event_count": sum(
            event["event"].startswith("disk-hit") for event in events
        ),
        "used_cache_keys": used_keys,
        "new_cache_keys": sorted(set(after) - set(before)),
        "source_snapshot": _source_snapshot_binding(
            source_snapshot_path, source_snapshot
        ),
        "artifacts": {
            "pytest_log": {"path": str(log_path), "sha256": _sha256(log_path)},
            "pytest_telemetry": {
                "path": str(telemetry_path),
                "sha256": _sha256(telemetry_path),
            },
            "launcher_evidence": {
                "path": str(launcher_evidence_path),
                "sha256": _sha256(launcher_evidence_path),
                "evidence": launcher_evidence,
            },
        },
    }


def _run_case(
    case: dict[str, Any],
    *,
    cache_dir: Path,
    output_dir: Path,
    base_env: dict[str, str],
    expected_gpu: dict[str, Any],
    matrix_path: Path,
    source_snapshot_path: Path,
    source_snapshot: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    case_id = str(case["id"])
    _assert_source_snapshot(
        source_snapshot, matrix_path, stage=f"case {case_id} before case"
    )
    _assert_source_snapshot_environment(
        base_env,
        source_snapshot_path,
        source_snapshot,
        stage=f"case {case_id} before case",
    )
    case_dir = output_dir / "cases" / case_id
    case_dir.mkdir(parents=True)
    prewarm = _prewarm_case(
        case,
        cache_dir=cache_dir,
        case_dir=case_dir,
        base_env=base_env,
        matrix_path=matrix_path,
        source_snapshot_path=source_snapshot_path,
        source_snapshot=source_snapshot,
        timeout=timeout,
    )
    before = _cache_manifests(cache_dir)
    telemetry_path = case_dir / "pytest-telemetry.json"
    launcher_evidence_path = case_dir / "launcher-evidence.json"
    nsys_prefix = case_dir / "nsys"
    log_path = case_dir / "pytest-nsys.log"
    env = dict(base_env)
    env.update(
        {
            "CORPUS_CASE_ID": case_id,
            "CORPUS_TELEMETRY": str(telemetry_path),
            "CORPUS_LAUNCHER_EVIDENCE": str(launcher_evidence_path),
            "CORPUS_UNDER_NSYS": "1",
            # Triton's platform key normally shells out to file(1).  Resolve
            # the exact native value before Nsight injection and let the
            # measurement-only launcher install it before importing Triton.
            "CORPUS_NSYS_PLATFORM_ARCHITECTURE": json.dumps(
                list(platform.architecture())
            ),
        }
    )
    command = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--output",
        str(nsys_prefix),
        *_pytest_launcher_command(case),
    ]
    _assert_source_snapshot(
        source_snapshot, matrix_path, stage=f"case {case_id} profile before launcher"
    )
    _assert_source_snapshot_environment(
        env,
        source_snapshot_path,
        source_snapshot,
        stage=f"case {case_id} profile before launcher",
    )
    _run_checked(command, env=env, timeout=timeout, log_path=log_path)
    _assert_source_snapshot(
        source_snapshot, matrix_path, stage=f"case {case_id} profile after launcher"
    )
    _assert_source_snapshot_environment(
        env,
        source_snapshot_path,
        source_snapshot,
        stage=f"case {case_id} profile after launcher",
    )
    rep_path = nsys_prefix.with_suffix(".nsys-rep")
    if not rep_path.is_file():
        raise CorpusError(f"Nsight report is missing for case {case_id}")
    sqlite_path = case_dir / "nsys.sqlite"
    export_log = case_dir / "nsys-export.log"
    _run_checked(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(rep_path),
        ],
        log_path=export_log,
    )
    if not sqlite_path.is_file():
        appended = sqlite_path.with_suffix(sqlite_path.suffix + ".sqlite")
        if appended.is_file():
            sqlite_path = appended
        else:
            raise CorpusError(f"Nsight SQLite export is missing for case {case_id}")

    telemetry = _read_json(telemetry_path)
    launcher_evidence = _read_json(launcher_evidence_path)
    if telemetry.get("schema") != "sparkinfer.cute.migration.pytest_telemetry.v3":
        raise CorpusError(f"case {case_id}: invalid pytest telemetry")
    if int(telemetry.get("exitstatus", -1)) != 0:
        raise CorpusError(f"case {case_id}: pytest telemetry reports failure")
    profiled_capture = telemetry.get("frontend_ptx_capture", {})
    if (
        profiled_capture.get("status") != "ok"
        or profiled_capture.get("redisassembled") is not False
    ):
        raise CorpusError(f"case {case_id}: incomplete frontend PTX capture")
    telemetry_gpu = telemetry.get("gpu")
    expected_identity = {
        "physical_ordinal": expected_gpu["physical_ordinal"],
        "uuid": expected_gpu["uuid"],
        "nvidia_smi_uuid": expected_gpu["uuid"],
        "name": expected_gpu["name"],
        "nvidia_smi_name": expected_gpu["name"],
    }
    if not isinstance(telemetry_gpu, dict) or any(
        telemetry_gpu.get(field) != value for field, value in expected_identity.items()
    ):
        raise CorpusError(
            f"case {case_id}: telemetry GPU identity {telemetry_gpu!r} "
            f"differs from nvidia-smi {expected_identity!r}"
        )
    if telemetry_gpu.get("capability") != [12, 0]:
        raise CorpusError(f"case {case_id}: telemetry capability is not SM120")
    reports = telemetry.get("reports")
    if not isinstance(reports, list) or not reports:
        raise CorpusError(f"case {case_id}: pytest emitted no reports")
    expected_nodeids = list(case["pytest_nodeids"])
    expected_nodeid_set = set(expected_nodeids)
    report_nodeids = {
        str(report.get("nodeid", "")) for report in reports if isinstance(report, dict)
    }
    phase_counts = Counter(
        (str(report.get("nodeid", "")), str(report.get("when", "")))
        for report in reports
        if isinstance(report, dict)
    )
    expected_phases = {
        (nodeid, phase)
        for nodeid in expected_nodeids
        for phase in ("setup", "call", "teardown")
    }
    if (
        any(not isinstance(report, dict) for report in reports)
        or report_nodeids != expected_nodeid_set
        or set(phase_counts) != expected_phases
        or any(count != 1 for count in phase_counts.values())
    ):
        raise CorpusError(
            f"case {case_id}: pytest report/node selection mismatch: "
            f"expected={expected_nodeids!r} observed={sorted(report_nodeids)!r} "
            f"phases={dict(phase_counts)!r}"
        )
    bad_reports = [
        report
        for report in reports
        if report.get("outcome") != "passed" or report.get("wasxfail")
    ]
    if bad_reports:
        raise CorpusError(f"case {case_id}: skip/xfail/failure reports: {bad_reports}")
    test_nvtx_ranges = telemetry.get("test_nvtx_ranges")
    if not isinstance(test_nvtx_ranges, list):
        raise CorpusError(f"case {case_id}: missing test NVTX telemetry")
    nvtx_nodeids = [
        record.get("nodeid") for record in test_nvtx_ranges if isinstance(record, dict)
    ]
    if (
        len(nvtx_nodeids) != len(test_nvtx_ranges)
        or any(not isinstance(nodeid, str) for nodeid in nvtx_nodeids)
        or Counter(nvtx_nodeids) != Counter(expected_nodeids)
        or any(
            record.get("label")
            != _test_nvtx_label(case_id, str(record.get("nodeid", "")))
            or record.get("completed") is not True
            for record in test_nvtx_ranges
        )
    ):
        raise CorpusError(
            f"case {case_id}: test NVTX/node selection mismatch: "
            f"expected={expected_nodeids!r} observed={nvtx_nodeids!r}"
        )
    for evidence in (
        case["correctness"].get("evidence_nodeids", []),
        case["graph"].get("evidence_nodeids", []),
    ):
        for nodeid in evidence:
            if not _reports_cover_nodeid(reports, nodeid):
                raise CorpusError(f"case {case_id}: missing passed evidence {nodeid}")
    if launcher_evidence.get("schema") != ("sparkinfer.cute.migration.launcher_evidence.v2"):
        raise CorpusError(f"case {case_id}: invalid launcher evidence")
    _validate_launcher_ptx_capture(launcher_evidence, case_id)
    _validate_launcher_source_binding(
        launcher_evidence, case_id, source_snapshot_path, source_snapshot
    )
    _validate_telemetry_source_binding(
        telemetry,
        case_id=case_id,
        snapshot_path=source_snapshot_path,
        snapshot=source_snapshot,
    )
    patch_status = launcher_evidence.get("runtime_patch_status")
    if not isinstance(patch_status, dict) or not all(patch_status.values()):
        raise CorpusError(f"case {case_id}: incomplete runtime patches")
    architecture_override = launcher_evidence.get("nsys_platform_architecture_override")
    if (
        not isinstance(architecture_override, dict)
        or architecture_override.get("installed") is not True
        or architecture_override.get("value") != list(platform.architecture())
    ):
        raise CorpusError(
            f"case {case_id}: invalid Nsight platform architecture override: "
            f"{architecture_override!r}"
        )
    sparkinfer_record = launcher_evidence.get("artifacts", {}).get("sparkinfer_package", {})
    expected_sparkinfer = (_ROOT / "sparkinfer" / "__init__.py").resolve()
    if Path(str(sparkinfer_record.get("path", ""))).resolve() != expected_sparkinfer:
        raise CorpusError(
            f"case {case_id}: imported sparkinfer from {sparkinfer_record.get('path')!r}, "
            f"expected current source {expected_sparkinfer}"
        )
    if sparkinfer_record.get("sha256") != _sha256(expected_sparkinfer):
        raise CorpusError(f"case {case_id}: current-source sparkinfer hash mismatch")

    after = _cache_manifests(cache_dir)
    new_keys = set(after) - set(before)
    log_text = log_path.read_text(encoding="utf-8")
    events = [match.groupdict() for match in _COMPILE_EVENT_RE.finditer(log_text)]
    miss_events = [event for event in events if event["event"] == "miss"]
    disk_events = [event for event in events if event["event"].startswith("disk-hit")]
    cache_info = telemetry.get("compile_cache", {})
    if int(cache_info.get("compile_misses", -1)) != len(miss_events):
        raise CorpusError(f"case {case_id}: compile miss log/counter mismatch")
    if int(cache_info.get("disk_cache_hits", -1)) != len(disk_events):
        raise CorpusError(f"case {case_id}: disk hit log/counter mismatch")
    event_keys = _resolve_event_prefixes(
        [event["prefix"] for event in events], set(after)
    )
    used_keys = sorted(new_keys | set(event_keys))
    if len(events) < int(case["min_compile_events"]) or not used_keys:
        raise CorpusError(f"case {case_id}: insufficient compile/cache evidence")
    if miss_events or new_keys:
        raise CorpusError(
            f"case {case_id}: profiled run compiled under Nsight instead of using "
            f"the prewarmed disk cache: misses={len(miss_events)} "
            f"new_keys={sorted(new_keys)}"
        )
    if used_keys != prewarm["used_cache_keys"]:
        raise CorpusError(
            f"case {case_id}: prewarm/profile specialization sets differ: "
            f"prewarm={prewarm['used_cache_keys']} profile={used_keys}"
        )
    expected_package_fingerprint = str(source_snapshot["sparkinfer_package"]["fingerprint"])
    manifests = {
        key: _validate_manifest(
            after[key],
            key,
            expected_package_fingerprint=expected_package_fingerprint,
        )
        for key in used_keys
    }
    for pattern in case["kernel_id_patterns"]:
        if not any(
            fnmatch.fnmatchcase(str(manifest["kernel_id"]), pattern)
            for manifest in manifests.values()
        ):
            raise CorpusError(
                f"case {case_id}: no used manifest matches kernel id {pattern!r}"
            )
    shape_branch_proof = _validate_shape_branch_evidence(case, reports, manifests)
    trace = _nsys_trace(sqlite_path, case_id, test_nvtx_ranges)
    result = {
        "id": case_id,
        "status": case["status"],
        "family": case["family"],
        "pytest_nodeids": case["pytest_nodeids"],
        "correctness": {
            "status": "passed",
            "oracle": case["correctness"].get("oracle"),
            "evidence_nodeids": case["correctness"].get("evidence_nodeids", []),
        },
        "graph": {
            "status": "passed" if case["graph"].get("required") else "not-applicable",
            "evidence_nodeids": case["graph"].get("evidence_nodeids", []),
        },
        "shape_branch_evidence": shape_branch_proof,
        "source_snapshot": _source_snapshot_binding(
            source_snapshot_path, source_snapshot
        ),
        "prewarm": prewarm,
        "compile_cache": {
            "counters": cache_info,
            "miss_event_count": len(miss_events),
            "disk_hit_event_count": len(disk_events),
            "used_cache_keys": used_keys,
            "new_cache_keys": sorted(new_keys),
        },
        "manifests": manifests,
        "nsys": {
            **trace,
            "artifacts": {
                "report": {"path": str(rep_path), "sha256": _sha256(rep_path)},
                "sqlite": {"path": str(sqlite_path), "sha256": _sha256(sqlite_path)},
                "export_log": {
                    "path": str(export_log),
                    "sha256": _sha256(export_log),
                },
            },
        },
        "artifacts": {
            "pytest_log": {"path": str(log_path), "sha256": _sha256(log_path)},
            "pytest_telemetry": {
                "path": str(telemetry_path),
                "sha256": _sha256(telemetry_path),
            },
            "launcher_evidence": {
                "path": str(launcher_evidence_path),
                "sha256": _sha256(launcher_evidence_path),
                "evidence": launcher_evidence,
            },
        },
    }
    _assert_source_snapshot(
        source_snapshot, matrix_path, stage=f"case {case_id} after case"
    )
    _assert_source_snapshot_environment(
        base_env,
        source_snapshot_path,
        source_snapshot,
        stage=f"case {case_id} after case",
    )
    return result


def _cutlass_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _CUTLASS_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _audit_command(
    cache_dir: Path,
    report: Path,
    *,
    gpu_uuid: str,
    versions: dict[str, str],
    full: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        _AUDITOR_MODULE,
        str(cache_dir),
        "--format",
        "csv",
        "--output",
        str(report),
        "--require-semantic-manifest",
        "--require-launch-dynamic-smem",
        "--occupancy-device",
        "0",
        "--require-driver-occupancy",
        "--require-driver-resource-validation",
        "--require-occupancy-gpu-uuid",
        gpu_uuid,
        "--require-architecture",
        "sm_120a",
    ]
    for package, version in versions.items():
        command.extend(["--require-cutlass-package", f"{package}={version}"])
    if full:
        command.extend(
            [
                "--require-kernel-id-pattern-file",
                str(_FAMILY_PATTERNS),
                "--require-kernel-symbol-pattern-file",
                str(_SYMBOL_PATTERNS),
            ]
        )
    return command


def _read_resource_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required_fields = {
            "cache_key",
            "semantic_key",
            "comparison_semantic_key",
            "kernel_id",
            "compile_spec_version",
            "compile_spec_hash",
            "compile_spec_json",
            "kernel",
            "registers",
            "frame_bytes",
            "min_stack_bytes",
            "local_load_instructions",
            "local_store_instructions",
            "threads_per_cta",
            "launch_dynamic_smem_bytes",
            "occupancy_active_ctas_per_sm",
        }
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise CorpusError(
                f"resource report schema is missing fields {sorted(missing_fields)}: "
                f"{path}"
            )
        rows = list(reader)
    if not rows:
        raise CorpusError(f"resource report is empty: {path}")
    return rows


def _bind_launches(
    cases: list[dict[str, Any]],
    resource_rows: list[dict[str, str]],
    matrix_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_cache: dict[str, list[dict[str, str]]] = defaultdict(list)
    resource_identities: set[tuple[str, str]] = set()
    for row in resource_rows:
        cache_key = row.get("cache_key", "")
        kernel = row.get("kernel", "")
        identity = (cache_key, kernel)
        if not cache_key or not kernel or identity in resource_identities:
            raise CorpusError(
                "resource report has an empty or duplicate cache-key/kernel row: "
                f"{identity!r}"
            )
        resource_identities.add(identity)
        rows_by_cache[cache_key].append(row)

    case_ids = [str(case.get("id", "")) for case in cases]
    matrix_ids = [str(case.get("id", "")) for case in matrix_cases]
    if (
        not case_ids
        or len(set(case_ids)) != len(case_ids)
        or len(set(matrix_ids)) != len(matrix_ids)
        or set(case_ids) != set(matrix_ids)
    ):
        raise CorpusError(
            "case traces and matrix cases must have the same unique ids: "
            f"traces={case_ids!r} matrix={matrix_ids!r}"
        )
    matrix_by_id = {str(case["id"]): case for case in matrix_cases}

    cache_case_owners: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        used_keys = case.get("compile_cache", {}).get("used_cache_keys")
        if (
            not isinstance(used_keys, list)
            or not used_keys
            or any(not isinstance(key, str) or not key for key in used_keys)
            or len(set(used_keys)) != len(used_keys)
        ):
            raise CorpusError(f"case {case['id']}: invalid used compile-cache key set")
        for cache_key in used_keys:
            cache_case_owners[cache_key].append(str(case["id"]))
    audited_keys = set(rows_by_cache)
    used_keys = set(cache_case_owners)
    if audited_keys != used_keys:
        raise CorpusError(
            "audited resource rows and used cache keys differ: "
            f"unbound_audited={sorted(audited_keys - used_keys)} "
            f"unaudited_used={sorted(used_keys - audited_keys)}"
        )
    multiply_bound_cases = {
        cache_key: owners
        for cache_key, owners in cache_case_owners.items()
        if len(owners) != 1
    }
    if multiply_bound_cases:
        raise CorpusError(
            "audited cache keys are multiply bound across cases: "
            f"{multiply_bound_cases}"
        )

    bindings: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        matrix_case = matrix_by_id[case_id]
        case_used_keys = set(case["compile_cache"]["used_cache_keys"])
        manifests = case.get("manifests")
        if not isinstance(manifests, dict) or set(manifests) != case_used_keys:
            raise CorpusError(
                f"case {case_id}: used cache keys and validated manifests differ"
            )
        claimed_rows = [
            row
            for cache_key in sorted(case_used_keys)
            for row in rows_by_cache[cache_key]
        ]
        for row in claimed_rows:
            manifest = manifests[row["cache_key"]]
            for field in (
                "semantic_key",
                "kernel_id",
                "compile_spec_version",
                "compile_spec_hash",
                "compile_spec_json",
            ):
                if str(row[field]) != str(manifest.get(field, "")):
                    raise CorpusError(
                        f"case {case_id}: audited row/manifest mismatch for "
                        f"{row['cache_key']} field {field}"
                    )

        for pattern in matrix_case["kernel_symbol_patterns"]:
            if not any(
                fnmatch.fnmatchcase(row["kernel"], pattern) for row in claimed_rows
            ):
                raise CorpusError(
                    f"case {case['id']}: no claimed row matches symbol {pattern!r}"
                )

        matrix_branches = list(matrix_case["shape_branches"])
        branch_proofs = case.get("shape_branch_evidence")
        if not isinstance(branch_proofs, dict) or set(branch_proofs) != set(
            matrix_branches
        ):
            raise CorpusError(f"case {case_id}: shape branch proof set differs")
        cache_branch_owners: dict[str, list[str]] = defaultdict(list)
        selected_rows_by_branch: dict[str, list[dict[str, str]]] = {}
        for branch in matrix_branches:
            proof = branch_proofs[branch]
            selected_keys = proof.get("selected_cache_keys")
            if (
                not isinstance(selected_keys, list)
                or not selected_keys
                or any(
                    not isinstance(cache_key, str) or cache_key not in case_used_keys
                    for cache_key in selected_keys
                )
                or len(set(selected_keys)) != len(selected_keys)
            ):
                raise CorpusError(
                    f"case {case_id}: branch {branch} has invalid selected cache keys"
                )
            for cache_key in selected_keys:
                cache_branch_owners[cache_key].append(branch)
            selected_rows_by_branch[branch] = [
                row for cache_key in selected_keys for row in rows_by_cache[cache_key]
            ]
        invalid_branch_owners = {
            cache_key: cache_branch_owners.get(cache_key, [])
            for cache_key in sorted(case_used_keys)
            if not cache_branch_owners.get(cache_key)
        }
        if invalid_branch_owners:
            raise CorpusError(
                f"case {case_id}: every audited cache key must bind to at least one "
                f"shape branch: {invalid_branch_owners}"
            )
        shared_keys_by_branch: dict[str, set[str]] = defaultdict(set)
        for cache_key, owners in cache_branch_owners.items():
            if len(owners) <= 1:
                continue
            groups = {
                branch_proofs[branch].get("shared_manifest_group") for branch in owners
            }
            if len(groups) != 1 or None in groups:
                raise CorpusError(
                    f"case {case_id}: shared manifest {cache_key} requires one "
                    f"explicit shared_manifest_group across branches {owners}"
                )
            for branch in owners:
                shared_keys_by_branch[branch].add(cache_key)
        for branch, proof in branch_proofs.items():
            group = proof.get("shared_manifest_group")
            if group is not None and not shared_keys_by_branch.get(branch):
                raise CorpusError(
                    f"case {case_id}: branch {branch} declares "
                    f"shared_manifest_group={group!r} but selects no shared manifest"
                )

        test_ranges = case.get("nsys", {}).get("test_ranges")
        if not isinstance(test_ranges, dict) or set(test_ranges) != set(
            matrix_case["pytest_nodeids"]
        ):
            raise CorpusError(f"case {case_id}: missing per-test Nsight ranges")
        node_symbol_owners: dict[tuple[str, str], tuple[str, str, str]] = {}
        for branch in matrix_branches:
            proof = branch_proofs[branch]
            selected_rows = selected_rows_by_branch[branch]
            rows_by_symbol: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in selected_rows:
                rows_by_symbol[row["kernel"]].append(row)
            ambiguous_symbols = {
                symbol: [(row["cache_key"], row["semantic_key"]) for row in rows]
                for symbol, rows in rows_by_symbol.items()
                if len(rows) != 1
            }
            if ambiguous_symbols:
                raise CorpusError(
                    f"case {case_id}: branch {branch} selects multiple audited "
                    f"rows with the same exact symbol: {ambiguous_symbols}"
                )
            evidence_sets = (
                ("correctness", proof.get("passed_correctness_nodeids")),
                ("graph", proof.get("passed_graph_nodeids")),
            )
            for evidence_kind, evidence_nodeids in evidence_sets:
                if (
                    not isinstance(evidence_nodeids, list)
                    or not evidence_nodeids
                    or any(
                        not isinstance(nodeid, str) or nodeid not in test_ranges
                        for nodeid in evidence_nodeids
                    )
                    or len(set(evidence_nodeids)) != len(evidence_nodeids)
                ):
                    raise CorpusError(
                        f"case {case_id}: branch {branch} has invalid "
                        f"{evidence_kind} test-range evidence"
                    )
                covered_rows: set[tuple[str, str]] = set()
                for nodeid in evidence_nodeids:
                    test_range = test_ranges[nodeid]
                    if (
                        not isinstance(test_range, dict)
                        or test_range.get("nvtx_label")
                        != _test_nvtx_label(case_id, nodeid)
                        or not isinstance(test_range.get("cuda_events"), list)
                    ):
                        raise CorpusError(
                            f"case {case_id}: invalid test range for {nodeid}"
                        )
                    trace_names = Counter(
                        event.get("shortName")
                        for event in test_range.get("cuda_events", [])
                        if isinstance(event, dict) and event.get("shortName")
                    )
                    launched_rows = [
                        row for row in selected_rows if trace_names[row["kernel"]] > 0
                    ]
                    if not launched_rows:
                        raise CorpusError(
                            f"case {case_id}: branch {branch} has no selected "
                            f"resource row in {evidence_kind} node {nodeid!r}"
                        )
                    for row in launched_rows:
                        launch_count = trace_names[row["kernel"]]
                        covered_rows.add((row["cache_key"], row["kernel"]))
                        ownership_key = (nodeid, row["kernel"])
                        row_owner = (
                            row["cache_key"],
                            row["semantic_key"],
                            branch,
                        )
                        prior_owner = node_symbol_owners.setdefault(
                            ownership_key, row_owner
                        )
                        if prior_owner != row_owner:
                            raise CorpusError(
                                f"case {case_id}: test/symbol launch is multiply "
                                f"bound: {ownership_key!r} owners="
                                f"{(prior_owner, row_owner)!r}"
                            )
                        bindings.append(
                            {
                                "case_id": case_id,
                                "branch": branch,
                                "shared_manifest_group": proof.get(
                                    "shared_manifest_group"
                                )
                                or "",
                                "evidence_kind": evidence_kind,
                                "nodeid": nodeid,
                                "test_nvtx_label": test_range.get("nvtx_label", ""),
                                "cache_key": row["cache_key"],
                                "semantic_key": row["semantic_key"],
                                "comparison_semantic_key": row[
                                    "comparison_semantic_key"
                                ],
                                "kernel_id": row["kernel_id"],
                                "compile_spec_version": row["compile_spec_version"],
                                "compile_spec_hash": row["compile_spec_hash"],
                                "compile_spec_json": row["compile_spec_json"],
                                "kernel": row["kernel"],
                                "launch_count": launch_count,
                                "registers": row["registers"],
                                "frame_bytes": row["frame_bytes"],
                                "min_stack_bytes": row["min_stack_bytes"],
                                "local_load_instructions": row[
                                    "local_load_instructions"
                                ],
                                "local_store_instructions": row[
                                    "local_store_instructions"
                                ],
                                "threads_per_cta": row["threads_per_cta"],
                                "launch_dynamic_smem_bytes": row[
                                    "launch_dynamic_smem_bytes"
                                ],
                                "occupancy_active_ctas_per_sm": row[
                                    "occupancy_active_ctas_per_sm"
                                ],
                            }
                        )
                expected_rows = {
                    (row["cache_key"], row["kernel"]) for row in selected_rows
                }
                missing_rows = sorted(expected_rows - covered_rows)
                if missing_rows:
                    raise CorpusError(
                        f"case {case_id}: branch {branch} has selected rows with "
                        f"no {evidence_kind} launch evidence: {missing_rows}"
                    )
    return bindings


def _write_binding_tsv(path: Path, bindings: list[dict[str, Any]]) -> None:
    if not bindings:
        raise CorpusError("cannot write an empty case/resource binding")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(bindings[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(bindings)


def _git_provenance() -> dict[str, Any]:
    commit = _run_checked(["git", "rev-parse", "HEAD"]).stdout.strip()
    status = _run_checked(["git", "status", "--short"]).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=_MATRIX)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--physical-gpu", type=int, choices=(4, 5))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--run-ready-subset", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--toolchain-label")
    parser.add_argument("--expected-cutlass-dsl-version")
    parser.add_argument("--cuda-path", type=Path)
    parser.add_argument("--smoke-case", default="quant_bf16_to_fp4")
    args = parser.parse_args()
    try:
        matrix_path = args.matrix.resolve()
        source_snapshot = _compute_source_snapshot(matrix_path)
        matrix, validation = validate_matrix(matrix_path)
        _assert_validation_matches_snapshot(validation, source_snapshot)
        smem_contract_static, smem_contract_static_text = (
            _run_smem_contract_audit()
        )
        validation = {
            **validation,
            "smem_contracts": smem_contract_static,
        }
        _assert_source_snapshot(
            source_snapshot, matrix_path, stage="runner start after static validation"
        )
        validation_artifact = validation
        if args.validate_only:
            validation_artifact = {
                **validation,
                "gpu_executed": False,
                "acceptance": False,
            }
        if args.validation_output is not None:
            args.validation_output.write_text(
                json.dumps(validation_artifact, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.validate_only:
            _assert_source_snapshot(
                source_snapshot, matrix_path, stage="validate-only runner end"
            )
            print(json.dumps(validation_artifact, indent=2, sort_keys=True))
            return 0
        if (
            args.physical_gpu is None
            or args.output_dir is None
            or args.cache_dir is None
            or not args.toolchain_label
            or not args.expected_cutlass_dsl_version
            or args.cuda_path is None
        ):
            raise CorpusError(
                "runtime requires --physical-gpu, --output-dir, --cache-dir, "
                "--toolchain-label, --expected-cutlass-dsl-version, and --cuda-path"
            )
        gaps = validation["gaps"]
        if gaps and not args.run_ready_subset:
            raise CorpusError(
                "full corpus refuses uncovered cases; resolve: "
                + "; ".join(f"{gap['id']}: {gap['reason']}" for gap in gaps)
            )
        selected = [
            case
            for case in matrix["cases"]
            if case["coverage_state"] == "ready"
            and (not args.case or case["id"] in set(args.case))
        ]
        unknown = set(args.case) - {case["id"] for case in matrix["cases"]}
        if unknown:
            raise CorpusError(f"unknown case ids: {sorted(unknown)}")
        if not selected:
            raise CorpusError("no ready cases selected")
        full = not gaps and not args.case and len(selected) == len(matrix["cases"])
        smoke_matches = [case for case in selected if case["id"] == args.smoke_case]
        if not smoke_matches:
            if full:
                raise CorpusError(
                    f"full corpus requires selected Nsight smoke case {args.smoke_case!r}"
                )
            smoke_case = selected[0]
        else:
            smoke_case = smoke_matches[0]
        selected = [smoke_case, *(case for case in selected if case is not smoke_case)]
        output_dir = args.output_dir.resolve()
        cache_dir = args.cache_dir.resolve()
        if output_dir == cache_dir:
            raise CorpusError("output and cache directories must be distinct")
        frozen_roots = tuple(
            (_ROOT / name).resolve()
            for name in ("sparkinfer", "benchmarks", "tests", "validation")
        )
        for label, path in (
            ("output directory", output_dir),
            ("compile cache", cache_dir),
        ):
            if any(path == root or root in path.parents for root in frozen_roots):
                raise CorpusError(
                    f"{label} must be outside frozen source trees: {path}"
                )
        _ensure_empty(output_dir, "output directory")
        _ensure_empty(cache_dir, "compile cache")
        source_snapshot_path = output_dir / "frozen-source-manifest.json"
        source_snapshot_path.write_text(
            json.dumps(source_snapshot, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        _assert_source_snapshot_artifact(
            source_snapshot_path, source_snapshot, stage="runner start"
        )
        gpu = _gpu_record(args.physical_gpu)
        versions = _cutlass_versions()
        if versions["nvidia-cutlass-dsl"] != args.expected_cutlass_dsl_version:
            raise CorpusError(
                "interpreter CUTLASS DSL version is "
                f"{versions['nvidia-cutlass-dsl']!r}, expected "
                f"{args.expected_cutlass_dsl_version!r}"
            )
        cuda_path = args.cuda_path.resolve()
        if not cuda_path.is_dir():
            raise CorpusError(f"canonical CUDA_PATH is not a directory: {cuda_path}")
        for tool in (cuda_path / "bin" / "ptxas", cuda_path / "bin" / "nvdisasm"):
            if not tool.is_file():
                raise CorpusError(f"required CUDA tool is missing: {tool}")
        ptx_dump_dir = output_dir / "cutlass-ptx-staging"
        ptx_dump_dir.mkdir()
        base_env, environment_record = _canonical_subprocess_environment(
            dict(os.environ), cuda_path=cuda_path, ptx_dump_dir=ptx_dump_dir
        )
        base_env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
                "CORPUS_PHYSICAL_GPU": str(args.physical_gpu),
                # Query identity once, outside Nsight.  Running nvidia-smi from
                # the injected pytest process can strand its child at exit and
                # leave subprocess.communicate() blocked before any test runs.
                "CORPUS_EXPECTED_GPU_UUID": str(gpu["uuid"]),
                "CORPUS_EXPECTED_GPU_NAME": str(gpu["name"]),
                "SPARKINFER_CUTE_COMPILE_CACHE_DIR": str(cache_dir),
                "SPARKINFER_CUTE_COMPILE_DISK_CACHE": "1",
                "SPARKINFER_LOG_CUTE_COMPILES": "1",
                "SPARKINFER_VLLM_ENGINE_STARTED": "1",
                "SPARKINFER_LOG_CUTE_COMPILES_AFTER_ENGINE_START": "1",
            }
        )
        base_env.update(
            _source_snapshot_environment(source_snapshot_path, source_snapshot)
        )
        _assert_source_snapshot_environment(
            base_env,
            source_snapshot_path,
            source_snapshot,
            stage="runner start before first case",
        )
        case_traces = [
            _run_case(
                smoke_case,
                cache_dir=cache_dir,
                output_dir=output_dir,
                base_env=base_env,
                expected_gpu=gpu,
                matrix_path=matrix_path,
                source_snapshot_path=source_snapshot_path,
                source_snapshot=source_snapshot,
                timeout=args.timeout,
            )
        ]
        smoke_report = output_dir / "nsys-shortname-smoke.csv"
        _run_checked(
            _audit_command(
                cache_dir,
                smoke_report,
                gpu_uuid=str(gpu["uuid"]),
                versions=versions,
                full=False,
            ),
            env=base_env,
            timeout=args.timeout,
            log_path=output_dir / "nsys-shortname-smoke.log",
        )
        smoke_bindings = _bind_launches(
            case_traces, _read_resource_rows(smoke_report), [smoke_case]
        )
        for case in selected[1:]:
            case_traces.append(
                _run_case(
                    case,
                    cache_dir=cache_dir,
                    output_dir=output_dir,
                    base_env=base_env,
                    expected_gpu=gpu,
                    matrix_path=matrix_path,
                    source_snapshot_path=source_snapshot_path,
                    source_snapshot=source_snapshot,
                    timeout=args.timeout,
                )
            )
        _assert_case_source_bindings(case_traces, source_snapshot_path, source_snapshot)

        initial_report = output_dir / "kernel-resources.initial.csv"
        audit_log = output_dir / "kernel-resources.initial.log"
        _run_checked(
            _audit_command(
                cache_dir,
                initial_report,
                gpu_uuid=str(gpu["uuid"]),
                versions=versions,
                full=full,
            ),
            env=base_env,
            timeout=args.timeout,
            log_path=audit_log,
        )
        resource_rows = _read_resource_rows(initial_report)
        bindings = _bind_launches(case_traces, resource_rows, selected)
        binding_path = output_dir / "case-resource-binding.tsv"
        _write_binding_tsv(binding_path, bindings)

        contract_path: Path | None = None
        contract_metadata_path: Path | None = None
        final_report = initial_report
        if full:
            source_log = output_dir / "source-inventory-audit.log"
            _run_checked(
                [
                    sys.executable,
                    "-m",
                    _SOURCE_AUDITOR_MODULE,
                    str(initial_report),
                    "--root",
                    str(_ROOT),
                    "--inventory",
                    str(_INVENTORY),
                ],
                log_path=source_log,
            )
            contract_path = output_dir / "exact-specialization-contract.tsv"
            contract_metadata_path = output_dir / "exact-specialization-contract.json"
            contract_log = output_dir / "exact-specialization-contract.log"
            _run_checked(
                [
                    sys.executable,
                    "-m",
                    _CONTRACT_BUILDER_MODULE,
                    str(initial_report),
                    "--output",
                    str(contract_path),
                    "--metadata-output",
                    str(contract_metadata_path),
                    "--corpus-id",
                    matrix["corpus_id"],
                    "--corpus-version",
                    matrix["version"],
                    "--corpus-driver",
                    str(_DRIVER_PATH),
                    "--shape-matrix",
                    str(matrix_path),
                    "--source-inventory",
                    str(_INVENTORY),
                ],
                log_path=contract_log,
            )
            final_report = output_dir / "kernel-resources.final.csv"
            exact_command = _audit_command(
                cache_dir,
                final_report,
                gpu_uuid=str(gpu["uuid"]),
                versions=versions,
                full=True,
            )
            exact_command.extend(
                [
                    "--require-exact-specialization-contract",
                    str(contract_path),
                    "--require-exact-specialization-contract-metadata",
                    str(contract_metadata_path),
                    "--require-corpus-driver",
                    str(_DRIVER_PATH),
                    "--require-shape-matrix",
                    str(matrix_path),
                    "--require-source-inventory",
                    str(_INVENTORY),
                    "--require-corpus-id",
                    matrix["corpus_id"],
                    "--require-corpus-version",
                    matrix["version"],
                ]
            )
            _run_checked(
                exact_command,
                env=base_env,
                timeout=args.timeout,
                log_path=output_dir / "kernel-resources.final.log",
            )

        _assert_source_snapshot(
            source_snapshot, matrix_path, stage="runner end before final evidence"
        )
        _assert_source_snapshot_environment(
            base_env,
            source_snapshot_path,
            source_snapshot,
            stage="runner end before final evidence",
        )
        _assert_case_source_bindings(case_traces, source_snapshot_path, source_snapshot)

        from validation.cutlass_migration.acceptance.corpus.ptx_capture import (
            validate_cache,
        )

        frontend_ptx_capture = validate_cache(cache_dir, required=True)
        if (
            frontend_ptx_capture.get("status") != "ok"
            or frontend_ptx_capture.get("redisassembled") is not True
        ):
            raise CorpusError(
                "frontend PTX capture is incomplete: "
                f"{frontend_ptx_capture.get('errors', [])}"
            )

        smem_contract_final, smem_contract_final_text = _run_smem_contract_audit()
        if (
            smem_contract_final != smem_contract_static
            or smem_contract_final_text != smem_contract_static_text
        ):
            raise CorpusError(
                "SMEM contract report changed between static validation and "
                "corpus finalization"
            )
        smem_contract_path = output_dir / "smem-contracts.final.json"
        smem_contract_path.write_text(smem_contract_final_text, encoding="utf-8")
        if _sha256(smem_contract_path) != smem_contract_final["report_sha256"]:
            raise CorpusError("SMEM contract artifact SHA-256 changed on write")
        smem_contract_finalization = {
            "schema": "sparkinfer.cute.migration.smem_contract_finalization.v1",
            "passed": True,
            "static_final_reports_equal": True,
            "gate": smem_contract_final,
            "artifact": {
                "path": str(smem_contract_path),
                "sha256": _sha256(smem_contract_path),
                "schema": _SMEM_CONTRACT_SCHEMA,
            },
        }

        trace = {
            "schema": _TRACE_SCHEMA,
            "corpus_id": matrix["corpus_id"],
            "corpus_version": matrix["version"],
            "complete": full,
            "toolchain_label": args.toolchain_label,
            "source_root": str(_ROOT),
            "gpu": gpu,
            "cutlass_packages": versions,
            "compile_environment": environment_record,
            "source_snapshot": {
                "binding": _source_snapshot_binding(
                    source_snapshot_path, source_snapshot
                ),
                "manifest": source_snapshot,
                "artifact": {
                    "path": str(source_snapshot_path),
                    "sha256": _sha256(source_snapshot_path),
                },
                "verification": {
                    "runner_pre_post": "matched",
                    "child_launcher_pre_runtime": "verified",
                    "child_pytest_pre_collection": "verified",
                    "child_pytest_session_finish": "verified",
                    "child_launcher_post_pytest": "verified",
                    "compile_manifest_package_fingerprint": "matched",
                },
            },
            "frontend_ptx_capture": frontend_ptx_capture,
            "python": {"executable": sys.executable, "prefix": sys.prefix},
            "git": _git_provenance(),
            "static_validation": validation,
            "smem_contracts": smem_contract_finalization,
            "selected_cases": [case["id"] for case in selected],
            "nsys_shortname_smoke": {
                "case_id": smoke_case["id"],
                "status": "passed",
                "bindings": smoke_bindings,
                "resource_report": {
                    "path": str(smoke_report),
                    "sha256": _sha256(smoke_report),
                },
            },
            "cases": case_traces,
            "case_resource_bindings": bindings,
            "artifacts": {
                "resource_report": {
                    "path": str(final_report),
                    "sha256": _sha256(final_report),
                },
                "case_resource_binding": {
                    "path": str(binding_path),
                    "sha256": _sha256(binding_path),
                },
                "smem_contracts": smem_contract_finalization["artifact"],
                "exact_specialization_contract": (
                    None
                    if contract_path is None
                    else {"path": str(contract_path), "sha256": _sha256(contract_path)}
                ),
                "exact_specialization_contract_metadata": (
                    None
                    if contract_metadata_path is None
                    else {
                        "path": str(contract_metadata_path),
                        "sha256": _sha256(contract_metadata_path),
                    }
                ),
            },
        }
        trace_path = output_dir / "case-trace.json"
        trace_path.write_text(
            json.dumps(trace, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _assert_source_snapshot(
            source_snapshot, matrix_path, stage="runner end after trace write"
        )
        _assert_source_snapshot_environment(
            base_env,
            source_snapshot_path,
            source_snapshot,
            stage="runner end after trace write",
        )
        _assert_case_source_bindings(case_traces, source_snapshot_path, source_snapshot)
        print(json.dumps({"complete": full, "trace": str(trace_path)}, indent=2))
        return 0
    except (CorpusError, OSError, SyntaxError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
