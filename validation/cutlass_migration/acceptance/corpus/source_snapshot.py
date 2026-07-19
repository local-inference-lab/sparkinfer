"""Child-process verification for the frozen migration-corpus source tree.

This module intentionally imports neither CUTLASS, torch, nor b12x.  The
launcher calls it before those runtimes are imported, and the pytest plugin
calls it before test collection and again at session shutdown.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SOURCE_SNAPSHOT_SCHEMA = "b12x.cute.migration.source_snapshot.v2"
SOURCE_SNAPSHOT_ENV = "CORPUS_FROZEN_SOURCE_MANIFEST"
SOURCE_SNAPSHOT_SHA256_ENV = "CORPUS_FROZEN_SOURCE_MANIFEST_SHA256"
SOURCE_SNAPSHOT_FINGERPRINT_ENV = (
    "CORPUS_FROZEN_SOURCE_SNAPSHOT_FINGERPRINT"
)
B12X_FINGERPRINT_ENV = "CORPUS_EXPECTED_B12X_PACKAGE_FINGERPRINT"


class SourceSnapshotError(RuntimeError):
    """The child no longer matches the runner's frozen source snapshot."""


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


def _source_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _source_file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or path.is_symlink():
        raise SourceSnapshotError(f"source input is not a regular file: {path}")
    content = resolved.read_bytes()
    return {
        "path": _source_path(resolved, repo_root),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _tree_snapshot(
    root: Path,
    repo_root: Path,
    *,
    root_label: str,
) -> dict[str, Any]:
    resolved_root = root.resolve()
    if not resolved_root.is_dir() or root.is_symlink():
        raise SourceSnapshotError(f"source tree is not a directory: {root}")
    paths: list[Path] = []
    for path in resolved_root.rglob("*"):
        if path.is_symlink():
            raise SourceSnapshotError(f"source tree contains a symlink: {path}")
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


def _resolve_source_path(raw_path: Any, repo_root: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise SourceSnapshotError(f"invalid source path in manifest: {raw_path!r}")
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceSnapshotError(f"cannot read frozen source manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceSnapshotError("frozen source manifest must be a JSON object")
    return value


def verify_frozen_source_from_environment(
    *,
    repo_root: Path,
    stage: str,
) -> dict[str, Any]:
    """Recompute every frozen source input and return hashable attestation."""

    resolved_repo = repo_root.resolve()
    raw_manifest_path = os.environ.get(SOURCE_SNAPSHOT_ENV, "").strip()
    expected_artifact_sha = os.environ.get(
        SOURCE_SNAPSHOT_SHA256_ENV, ""
    ).strip()
    expected_manifest_sha = os.environ.get(
        SOURCE_SNAPSHOT_FINGERPRINT_ENV, ""
    ).strip()
    expected_b12x_fingerprint = os.environ.get(B12X_FINGERPRINT_ENV, "").strip()
    if not all(
        (
            raw_manifest_path,
            expected_artifact_sha,
            expected_manifest_sha,
            expected_b12x_fingerprint,
        )
    ):
        raise SourceSnapshotError("frozen source environment is incomplete")

    manifest_path = Path(raw_manifest_path)
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise SourceSnapshotError(
            f"frozen source manifest must be an absolute file: {manifest_path}"
        )
    artifact_sha = _sha256(manifest_path)
    if artifact_sha != expected_artifact_sha:
        raise SourceSnapshotError(
            "frozen source manifest artifact hash changed: "
            f"{expected_artifact_sha}->{artifact_sha}"
        )

    manifest = _load_manifest(manifest_path)
    if manifest.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        raise SourceSnapshotError(
            f"unexpected frozen source schema: {manifest.get('schema')!r}"
        )
    recorded_manifest_sha = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    computed_manifest_sha = _canonical_sha256(payload)
    if (
        recorded_manifest_sha != computed_manifest_sha
        or computed_manifest_sha != expected_manifest_sha
    ):
        raise SourceSnapshotError(
            "frozen source canonical hash mismatch: "
            f"recorded={recorded_manifest_sha!r} "
            f"computed={computed_manifest_sha!r} "
            f"expected={expected_manifest_sha!r}"
        )

    expected_package = manifest.get("b12x_package")
    if not isinstance(expected_package, dict):
        raise SourceSnapshotError("frozen source manifest lacks b12x_package")
    observed_package = _tree_snapshot(
        resolved_repo / "b12x",
        resolved_repo,
        root_label="b12x",
    )
    if observed_package != expected_package:
        raise SourceSnapshotError("b12x package differs from frozen source manifest")
    if observed_package["fingerprint"] != expected_b12x_fingerprint:
        raise SourceSnapshotError(
            "b12x fingerprint differs from frozen source environment"
        )

    expected_trees = manifest.get("source_trees")
    if not isinstance(expected_trees, dict) or set(expected_trees) != {
        "benchmarks",
        "tests",
        "validation",
    }:
        raise SourceSnapshotError("frozen source manifest has invalid source_trees")
    verified_file_count = int(observed_package["file_count"])
    for name in ("benchmarks", "tests", "validation"):
        expected_tree = expected_trees[name]
        if not isinstance(expected_tree, dict):
            raise SourceSnapshotError(f"invalid frozen {name} tree record")
        tree_root = _resolve_source_path(expected_tree.get("root"), resolved_repo)
        observed_tree = _tree_snapshot(
            tree_root,
            resolved_repo,
            root_label=_source_path(tree_root, resolved_repo),
        )
        if observed_tree != expected_tree:
            raise SourceSnapshotError(
                f"{name} tree differs from frozen source manifest"
            )
        verified_file_count += int(observed_tree["file_count"])

    expected_inputs = manifest.get("inputs")
    if not isinstance(expected_inputs, dict) or not expected_inputs:
        raise SourceSnapshotError("frozen source manifest has no inputs")
    for name, expected_record in sorted(expected_inputs.items()):
        if not isinstance(expected_record, dict):
            raise SourceSnapshotError(f"invalid frozen input record {name!r}")
        path = _resolve_source_path(expected_record.get("path"), resolved_repo)
        observed_record = _source_file_record(path, resolved_repo)
        if observed_record != expected_record:
            raise SourceSnapshotError(
                f"source input {name!r} differs from frozen manifest"
            )

    return {
        "status": "verified",
        "stage": stage,
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "repo_root": str(resolved_repo),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_artifact_sha256": artifact_sha,
        "manifest_sha256": computed_manifest_sha,
        "b12x_package_fingerprint": observed_package["fingerprint"],
        "verified_file_count": verified_file_count,
        "verified_input_count": len(expected_inputs),
    }
