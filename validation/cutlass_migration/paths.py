"""Stable repository paths for the nested migration qualification package."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = VALIDATION_ROOT.parent
CORE_ROOT = PACKAGE_ROOT / "core"
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence"
ACCEPTANCE_ROOT = PACKAGE_ROOT / "acceptance"
CORPUS_ROOT = ACCEPTANCE_ROOT / "corpus"
E2E_ROOT = ACCEPTANCE_ROOT / "e2e"
SINGLE_ARM_ROOT = ACCEPTANCE_ROOT / "single_arm"
DIAGNOSTICS_ROOT = PACKAGE_ROOT / "diagnostics"
PAIRED_ROOT = DIAGNOSTICS_ROOT / "paired"
INTEGRITY_CHECKS_ROOT = PACKAGE_ROOT / "integrity_checks"
DATA_ROOT = PACKAGE_ROOT / "data"


def repo_relative(path: Path) -> str:
    """Return a canonical POSIX repository-relative path."""

    return path.resolve().relative_to(REPO_ROOT).as_posix()
