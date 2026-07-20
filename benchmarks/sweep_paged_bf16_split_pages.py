#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LINE_RE = re.compile(
    r"^(decode|extend)\s+bs=\s*(?P<batch>\d+)\s+q=\s*(?P<q>\d+)\s+k=\s*(?P<k>\d+)\s+"
    r"(?P<plan>[^|]+)\|\s+sparkinfer median=\s*(?P<sparkinfer>[0-9.]+)\s+us.*\|\s+fa2 median=\s*(?P<fa2>[0-9.]+)\s+us.*\|\s+fa2/sparkinfer=\s*(?P<speedup>[0-9.]+)x"
)


def _parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def _run_once(*, cache_len: int, fixed_split_pages: int, env_overrides: dict[str, str]) -> dict[str, object]:
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = [
        sys.executable,
        "benchmarks/benchmark_paged_attention.py",
        "--cache-seqlens",
        str(cache_len),
        "--kv-dtype",
        "bf16",
        "--warmup",
        "5",
        "--replays",
        "200",
        "--check",
        "--fixed-split-pages",
        str(fixed_split_pages),
    ]
    proc = subprocess.run(
        cmd,
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    result: dict[str, object] = {
        "cache_len": cache_len,
        "fixed_split_pages": fixed_split_pages,
        "returncode": proc.returncode,
    }
    if proc.returncode != 0:
        result["status"] = "crash"
        result["tail"] = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-40:])
        return result

    lines = {}
    for line in proc.stdout.splitlines():
        m = _LINE_RE.match(line.strip())
        if m is None:
            continue
        phase = m.group(1)
        lines[phase] = {
            "plan": m.group("plan").strip(),
            "sparkinfer_us": float(m.group("sparkinfer")),
            "fa2_us": float(m.group("fa2")),
            "fa2_over_sparkinfer": float(m.group("speedup")),
        }
    result["status"] = "ok"
    result["decode"] = lines.get("decode")
    result["extend"] = lines.get("extend")
    if "decode" in lines and "extend" in lines:
        result["geo"] = statistics.geometric_mean(
            [lines["decode"]["fa2_over_sparkinfer"], lines["extend"]["fa2_over_sparkinfer"]]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep BF16 fixed split pages for paged attention and emit parsed JSON."
    )
    parser.add_argument(
        "--cache-lens",
        required=True,
        help="comma-separated cache lengths to benchmark",
    )
    parser.add_argument(
        "--split-pages",
        required=True,
        help="comma-separated fixed_split_pages values to benchmark",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="env override formatted as KEY=VALUE; may be repeated",
    )
    args = parser.parse_args()

    env_overrides: dict[str, str] = {}
    for item in args.env:
        key, value = item.split("=", 1)
        env_overrides[key] = value

    reports = []
    for cache_len in _parse_csv_ints(args.cache_lens):
        for split_pages in _parse_csv_ints(args.split_pages):
            reports.append(
                _run_once(
                    cache_len=cache_len,
                    fixed_split_pages=split_pages,
                    env_overrides=env_overrides,
                )
            )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
