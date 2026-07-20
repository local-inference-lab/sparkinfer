#!/usr/bin/env python3
"""Build a reviewable external case/config contract from one complete GPU log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validation.cutlass_migration.diagnostics.graph_replay_abba import (
    _external_case_contract_payload,
    _read_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_log", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--backend", default="sparkinfer")
    parser.add_argument(
        "--allow-non-reference-oracle",
        action="store_true",
        help="allow a producer whose correctness schema is not torch-reference",
    )
    args = parser.parse_args()

    try:
        run = _read_run(
            args.sample_log,
            args.backend,
            minimum_warmup=1,
            minimum_replays=1,
            require_reference_oracle=not args.allow_non_reference_oracle,
        )
        contract = _external_case_contract_payload(args.backend, run.provenance)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(run.samples)} exact cases to {args.output}; review and check "
        "in this contract before using it as an acceptance gate",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
