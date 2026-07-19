"""Explicit final/diagnostic classification for migration evidence producers."""

from __future__ import annotations

import argparse


EVIDENCE_STATUSES = ("final-source", "diagnostic-non-final")


def add_evidence_status_argument(parser: argparse.ArgumentParser) -> None:
    """Require callers to classify every emitted benchmark artifact."""

    parser.add_argument(
        "--evidence-status",
        choices=EVIDENCE_STATUSES,
        required=True,
        help=(
            "explicit source-finality classification; release indexing accepts "
            "only final-source"
        ),
    )


def validate_evidence_status(value: object) -> str:
    if value not in EVIDENCE_STATUSES:
        raise ValueError(
            "evidence status must be explicitly final-source or diagnostic-non-final"
        )
    return str(value)
