#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run_json(cmd: list[str], env: dict[str, str]) -> tuple[str, dict[str, object] | None, str]:
    proc = subprocess.run(
        cmd,
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-20:])
        return "crash", None, tail
    try:
        return "ok", json.loads(proc.stdout), ""
    except json.JSONDecodeError:
        tail = "\n".join(proc.stdout.strip().splitlines()[-20:])
        return "parse_error", None, tail


def _probe_swizzle(swizzle: str, *, run_pytest: bool) -> dict[str, object]:
    env = os.environ.copy()
    env["SPARKINFER_PAGED_KV_TMA"] = "1"
    env["SPARKINFER_PAGED_KV_TMA_SWIZZLE"] = swizzle

    preg_cmd = [
        sys.executable,
        "benchmarks/probe_paged_bf16_p_regs.py",
        "--lhs-backend",
        "cpasync",
        "--rhs-backend",
        "tma",
    ]
    pvreg_cmd = [
        sys.executable,
        "benchmarks/probe_paged_bf16_pv_regs.py",
        "--lhs-backend",
        "cpasync",
        "--rhs-backend",
        "tma",
        "--lhs-consumer",
        "manual",
        "--rhs-consumer",
        "manual",
    ]

    preg_status, preg_report, preg_tail = _run_json(preg_cmd, env)
    pvreg_status, pvreg_report, pvreg_tail = _run_json(pvreg_cmd, env)

    result: dict[str, object] = {
        "swizzle": swizzle,
        "preg_status": preg_status,
        "pvreg_status": pvreg_status,
        "preg_mismatch_count": None if preg_report is None else preg_report["mismatch_count"],
        "pvreg_mismatch_count": None if pvreg_report is None else pvreg_report["mismatch_count"],
    }
    if preg_tail:
        result["preg_tail"] = preg_tail
    if pvreg_tail:
        result["pvreg_tail"] = pvreg_tail

    if (
        run_pytest
        and preg_report is not None
        and pvreg_report is not None
        and int(preg_report["mismatch_count"]) == 0
        and int(pvreg_report["mismatch_count"]) == 0
    ):
        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_attention_paged_forward.py::test_paged_forward_matches_reference_without_split",
            "-vv",
        ]
        pytest_proc = subprocess.run(
            pytest_cmd,
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        result["pytest_status"] = "ok" if pytest_proc.returncode == 0 else "fail"
        if pytest_proc.returncode != 0:
            result["pytest_tail"] = "\n".join(
                (pytest_proc.stdout + "\n" + pytest_proc.stderr).strip().splitlines()[-20:]
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep BF16 paged-KV TMA swizzle candidates against the exact cp.async PREGS/PVREGS contracts."
    )
    parser.add_argument(
        "--swizzle",
        action="append",
        default=[],
        help="candidate swizzle triple formatted as b,m,s; may be repeated",
    )
    parser.add_argument(
        "--run-pytest",
        action="store_true",
        help="run the main decode exactness test for candidates that hit zero PREGS/PVREGS mismatches",
    )
    args = parser.parse_args()

    candidates = args.swizzle or [
        "3,1,4",
        "3,1,5",
        "3,2,4",
        "3,2,5",
        "3,3,4",
        "3,3,5",
        "3,4,4",
        "3,4,5",
    ]

    reports = [_probe_swizzle(swizzle, run_pytest=args.run_pytest) for swizzle in candidates]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
