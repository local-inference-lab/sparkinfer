"""Measurement-only pytest launcher for the CUTLASS migration corpus.

CUTLASS DSL 4.5 keeps ``_mlir_helpers`` below ``cutlass.base_dsl`` while
sparkinfer's 4.6-only runtime patch imports the promoted top-level module.  The
corpus needs to measure the 4.5 baseline without adding that compatibility
alias to production.  This launcher verifies PTX retention, installs the alias,
installs the cache-key-bound PTX hook before loading the remaining runtime,
verifies every runtime patch, records hashable evidence, and then hands off to
pytest.  It is not imported by the sparkinfer package.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from validation.cutlass_migration.paths import REPO_ROOT


_EVIDENCE_SCHEMA = "sparkinfer.cute.migration.launcher_evidence.v2"
_NSYS_PLATFORM_ARCHITECTURE_ENV = "CORPUS_NSYS_PLATFORM_ARCHITECTURE"
_PTX_CAPTURE_ENV = "CORPUS_RETAIN_FRONTEND_PTX"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_record(module: ModuleType) -> dict[str, str]:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise RuntimeError(f"module {module.__name__!r} has no source path")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"module path is not a file: {path}")
    return {"module": module.__name__, "path": str(path), "sha256": _sha256(path)}


def _evidence_path() -> Path:
    raw = os.environ.get("CORPUS_LAUNCHER_EVIDENCE")
    if not raw:
        raise RuntimeError("CORPUS_LAUNCHER_EVIDENCE is required")
    return Path(raw)


def _attest_frozen_source(stage: str) -> dict[str, Any]:
    from validation.cutlass_migration.acceptance.corpus.source_snapshot import (
        verify_frozen_source_from_environment,
    )

    return verify_frozen_source_from_environment(
        repo_root=REPO_ROOT,
        stage=stage,
    )


def _write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _frontend_ptx_capture_environment() -> dict[str, Any]:
    """Verify retention controls before importing CUTLASS or sparkinfer."""

    raw_enabled = os.environ.get(_PTX_CAPTURE_ENV, "")
    enabled = raw_enabled.strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return {"enabled": False, "verified_before_cutlass_import": True}
    keep = os.environ.get("CUTE_DSL_KEEP", "")
    keep_tokens = sorted(
        {token.strip().lower() for token in keep.split(",") if token.strip()}
    )
    if "ptx" not in keep_tokens:
        raise RuntimeError(
            f"{_PTX_CAPTURE_ENV}=1 requires CUTE_DSL_KEEP to include ptx"
        )
    dump_dir_raw = os.environ.get("CUTE_DSL_DUMP_DIR", "").strip()
    if not dump_dir_raw:
        raise RuntimeError(f"{_PTX_CAPTURE_ENV}=1 requires CUTE_DSL_DUMP_DIR")
    dump_dir = Path(dump_dir_raw)
    if not dump_dir.is_absolute():
        raise RuntimeError("CUTE_DSL_DUMP_DIR must be absolute")
    return {
        "enabled": True,
        "verified_before_cutlass_import": True,
        "cute_dsl_keep_tokens": keep_tokens,
        "cute_dsl_dump_dir": str(dump_dir),
    }


def _install_nsys_platform_architecture_override() -> dict[str, Any]:
    """Avoid spawning ``file(1)`` after Nsight has injected helper threads.

    ``triton.runtime.build.platform_key()`` calls ``platform.architecture()``,
    which shells out to ``file`` on Linux.  Under Nsight Systems the helper can
    exit but remain unreaped, leaving Python blocked in ``communicate()``.  The
    coordinator records the native result before starting Nsight and passes it
    only to the profiled subprocess; this launcher installs that exact value
    before importing sparkinfer, torch, or Triton.
    """

    raw = os.environ.get(_NSYS_PLATFORM_ARCHITECTURE_ENV)
    if raw is None:
        return {"installed": False, "value": None}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"invalid {_NSYS_PLATFORM_ARCHITECTURE_ENV}: {raw!r}"
        ) from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != 2
        or any(not isinstance(value, str) or not value for value in parsed)
    ):
        raise RuntimeError(
            f"invalid {_NSYS_PLATFORM_ARCHITECTURE_ENV} value: {parsed!r}"
        )
    value = (parsed[0], parsed[1])
    platform.architecture = lambda *args, **kwargs: value
    return {"installed": True, "value": list(value)}


def _install_cutlass_45_alias(cutlass_version: str) -> bool:
    alias_installed = False
    if cutlass_version.startswith("4.5."):
        from cutlass.base_dsl import _mlir_helpers as base_helpers

        existing = sys.modules.get("cutlass._mlir_helpers")
        if existing is not None and existing is not base_helpers:
            raise RuntimeError(
                "cutlass._mlir_helpers was imported before the 4.5 corpus alias"
            )
        sys.modules["cutlass._mlir_helpers"] = base_helpers
        alias_installed = True
    return alias_installed


def _install_frontend_ptx_capture() -> dict[str, Any]:
    from validation.cutlass_migration.acceptance.corpus.ptx_capture import (
        install,
        installation_status,
    )

    install()
    status = installation_status()
    if status.get("enabled") is True and status.get("installed") is not True:
        raise RuntimeError("frontend PTX capture hook was not installed")
    return status


def _prepare_runtime(cutlass_version: str, alias_installed: bool) -> dict[str, Any]:
    import cutlass._mlir_helpers as public_helpers
    from cutlass._mlir_helpers import op as op_helpers

    if cutlass_version.startswith("4.5."):
        from cutlass.base_dsl import _mlir_helpers as base_helpers

        if public_helpers is not base_helpers:
            raise RuntimeError("4.5 helper alias does not preserve module identity")

    from sparkinfer.cute.runtime_patches import (
        apply_cutlass_runtime_patches,
        cutlass_runtime_patch_status,
    )

    apply_cutlass_runtime_patches()
    patch_status = dict(cutlass_runtime_patch_status())
    if not patch_status or not all(patch_status.values()):
        raise RuntimeError(f"CUTLASS runtime patches are incomplete: {patch_status}")

    inspect_proxy = op_helpers.inspect
    proxy_marker = bool(getattr(inspect_proxy, "_sparkinfer_direct_frameinfo", False))
    proxy_type = f"{type(inspect_proxy).__module__}.{type(inspect_proxy).__qualname__}"
    expected_proxy_type = "sparkinfer.cute.runtime_patches._DirectFrameInfoInspectProxy"
    if not proxy_marker or proxy_type != expected_proxy_type:
        raise RuntimeError(
            "direct-frameinfo proxy assertion failed: "
            f"marker={proxy_marker} type={proxy_type!r}"
        )

    import sparkinfer
    import sparkinfer.cute.runtime_patches as runtime_patches

    return {
        "schema": _EVIDENCE_SCHEMA,
        "cutlass_dsl_version": cutlass_version,
        "measurement_only_45_alias": {
            "installed": alias_installed,
            "identity_verified": (
                not cutlass_version.startswith("4.5.")
                or public_helpers is sys.modules["cutlass._mlir_helpers"]
            ),
            "public_module_name": public_helpers.__name__,
        },
        "runtime_patch_status": patch_status,
        "direct_frameinfo_proxy": {
            "marker": proxy_marker,
            "type": proxy_type,
        },
        "artifacts": {
            "launcher": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "mlir_helpers": _module_record(public_helpers),
            "mlir_op_helpers": _module_record(op_helpers),
            "runtime_patches": _module_record(runtime_patches),
            "sparkinfer_package": _module_record(sparkinfer),
        },
    }


def main() -> int:
    # This is deliberately the first action: attest the complete source and
    # evidence-tool closure before importing CUTLASS, sparkinfer, torch, or pytest.
    source_pre_runtime = _attest_frozen_source("launcher_pre_runtime")
    capture_environment = _frontend_ptx_capture_environment()
    architecture_override = _install_nsys_platform_architecture_override()
    cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    alias_installed = _install_cutlass_45_alias(cutlass_version)
    capture_installation = _install_frontend_ptx_capture()
    evidence = _prepare_runtime(cutlass_version, alias_installed)
    evidence["frontend_ptx_capture_environment"] = capture_environment
    evidence["frontend_ptx_capture_installation"] = capture_installation
    evidence["nsys_platform_architecture_override"] = architecture_override
    evidence["source_attestation"] = {"pre_runtime": source_pre_runtime}
    path = _evidence_path()
    _write_evidence(path, evidence)

    import pytest

    exitstatus = int(pytest.main(sys.argv[1:]))
    evidence["source_attestation"]["post_pytest"] = _attest_frozen_source(
        "launcher_post_pytest"
    )
    _write_evidence(path, evidence)
    return exitstatus


if __name__ == "__main__":
    raise SystemExit(main())
