#!/usr/bin/env python3
"""Freeze production and measured-runtime sparkinfer trees for the E2E gate.

The baseline measurement tree contains resource-capture instrumentation that
is not part of the pre-migration production source.  This builder snapshots a
pristine production root and the actual runtime root independently, computes
their complete sparkinfer package fingerprints, and fails unless their delta is the
closed instrumentation allowlist.  The current arm must have no overlay.

Example:

  python -m validation.cutlass_migration acceptance source-manifest \
    --side baseline --source-id pre-migration-e71 \
    --production-root /tmp/sparkinfer-cutlass45-source \
    --runtime-root ~/projects/sparkinfer-research/rs-18-cutlass45-baseline \
    --output /tmp/sparkinfer-e2e-baseline-source.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from validation.cutlass_migration.acceptance.e2e.index import (
    BASELINE_RUNTIME_OVERLAY_PATHS,
    PRODUCTION_SOURCE_SCHEMA,
    EndToEndValidationError,
    _canonical_sha256,
    _validate_source_manifest,
)


class SourceManifestError(RuntimeError):
    """A production/runtime source snapshot is not admissible."""


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise SourceManifestError(
            f"git {' '.join(args)} failed for {repo_root}: {detail.strip()}"
        ) from exc
    return completed.stdout.strip()


def _git_provenance(repo_root: Path) -> dict[str, object]:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    status_text = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "sparkinfer",
    )
    return {
        "commit": commit,
        "status": sorted(set(status_text.splitlines())) if status_text else [],
    }


def _tree_snapshot(repo_root: Path) -> dict[str, Any]:
    package_root = repo_root / "sparkinfer"
    if not package_root.is_dir() or package_root.is_symlink():
        raise SourceManifestError(f"not a regular sparkinfer package tree: {package_root}")
    paths: list[Path] = []
    for path in package_root.rglob("*"):
        if path.is_symlink():
            raise SourceManifestError(f"sparkinfer package contains a symlink: {path}")
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        paths.append(path)
    paths.sort(key=lambda path: path.relative_to(package_root).as_posix())
    if not paths:
        raise SourceManifestError(f"empty sparkinfer package tree: {package_root}")

    content_digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(package_root).as_posix()
        content = path.read_bytes()
        content_digest.update(relative.encode("utf-8"))
        content_digest.update(b"\0")
        content_digest.update(content)
        content_digest.update(b"\0")
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    return {
        "root": "sparkinfer",
        "fingerprint": content_digest.hexdigest(),
        "records_sha256": _canonical_sha256(files),
        "file_count": len(files),
        "files": files,
    }


def _endpoint(repo_root: Path) -> dict[str, object]:
    return {
        "repo_root": str(repo_root),
        "git": _git_provenance(repo_root),
        "sparkinfer_package": _tree_snapshot(repo_root),
    }


def _overlay(
    production: dict[str, object],
    runtime: dict[str, object],
    *,
    side: str,
) -> dict[str, object]:
    production_package = production["sparkinfer_package"]
    runtime_package = runtime["sparkinfer_package"]
    assert isinstance(production_package, dict)
    assert isinstance(runtime_package, dict)
    production_files = {
        f"sparkinfer/{record['path']}": record for record in production_package["files"]
    }
    runtime_files = {
        f"sparkinfer/{record['path']}": record for record in runtime_package["files"]
    }
    changed_paths = sorted(
        path
        for path in set(production_files) | set(runtime_files)
        if production_files.get(path) != runtime_files.get(path)
    )
    details = {
        path: {
            "production": production_files.get(path),
            "runtime": runtime_files.get(path),
        }
        for path in changed_paths
    }
    if side == "baseline":
        allowed_paths = list(BASELINE_RUNTIME_OVERLAY_PATHS)
        policy = "instrumentation-only"
    else:
        allowed_paths = []
        policy = "none"
    if changed_paths != allowed_paths:
        raise SourceManifestError(
            f"{side} production/runtime delta is not reviewed: "
            f"expected={allowed_paths!r}, observed={changed_paths!r}"
        )
    return {
        "policy": policy,
        "allowed_paths": allowed_paths,
        "changed_paths": changed_paths,
        "details_sha256": _canonical_sha256(details),
    }


def build_manifest(
    *,
    side: str,
    source_id: str,
    production_root: Path,
    runtime_root: Path,
) -> dict[str, object]:
    production_root = production_root.resolve()
    runtime_root = runtime_root.resolve()
    production = _endpoint(production_root)
    runtime = _endpoint(runtime_root)
    production_git = production["git"]
    runtime_git = runtime["git"]
    assert isinstance(production_git, dict)
    assert isinstance(runtime_git, dict)
    if production_git["commit"] != runtime_git["commit"]:
        raise SourceManifestError(
            "production and runtime roots are not based on the same git commit: "
            f"{production_git['commit']} != {runtime_git['commit']}"
        )
    if side == "baseline" and production_git["status"]:
        raise SourceManifestError(
            "baseline production root is not pristine under sparkinfer: "
            f"{production_git['status']!r}"
        )
    payload = {
        "schema": PRODUCTION_SOURCE_SCHEMA,
        "side": side,
        "source_id": source_id,
        "production": production,
        "runtime": runtime,
        "runtime_overlay": _overlay(production, runtime, side=side),
    }
    return {**payload, "manifest_sha256": _canonical_sha256(payload)}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("baseline", "current"), required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.source_id.strip():
        parser.error("--source-id must be nonempty")
    output = args.output.resolve()
    for root in (args.production_root.resolve(), args.runtime_root.resolve()):
        package_root = root / "sparkinfer"
        if output == package_root or package_root in output.parents:
            parser.error("--output must not mutate a snapshotted sparkinfer package")
    return args


def main() -> int:
    args = _args()
    try:
        manifest = build_manifest(
            side=args.side,
            source_id=args.source_id,
            production_root=args.production_root,
            runtime_root=args.runtime_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        _validate_source_manifest(args.output.resolve(), args.side)
    except (OSError, SourceManifestError, EndToEndValidationError) as exc:
        print(f"production-source manifest failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"status=pass side={args.side} source_id={args.source_id} "
        f"manifest_sha256={manifest['manifest_sha256']} output={args.output.resolve()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
