#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REMOTE = "lukea@orion.local:~/Desktop"
TRACE_SUFFIXES = (
    ".pt.trace.json.gz",
    ".pt.trace.json",
    ".trace.json.gz",
    ".trace.json",
    ".json.gz",
    ".json",
    ".nsys-rep",
)
RANK0_RE = re.compile(
    r"(^|[^0-9a-z])(?:rank|global_rank|tp_rank|local_rank)[_=\-.]?0([^0-9a-z]|$)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a live vLLM profile and scp the newest rank-0 trace."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--profile-dir", action="append", default=[])
    parser.add_argument("--wait-timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def post_start_profile(base_url: str, steps: int, prefix: str | None) -> None:
    payload = {
        "profiler": "torch",
        "profile": "torch",
        "activities": ["CPU", "CUDA"],
        "profile_activities": ["CPU", "CUDA"],
        "with_cpu": True,
        "with_cuda": True,
        "max_iterations": steps,
        "num_steps": steps,
        "profile_steps": steps,
        "delay_iterations": 0,
    }
    if prefix:
        payload["profile_prefix"] = prefix

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/start_profile",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"/start_profile returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"/start_profile returned HTTP {exc.code}: {body}") from exc


def proc_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def iter_vllm_processes() -> Iterable[tuple[Path, list[str], dict[str, str]]]:
    for proc_dir in Path("/proc").glob("[0-9]*"):
        raw_cmdline = proc_bytes(proc_dir / "cmdline")
        if not raw_cmdline:
            continue
        cmdline = [
            item.decode("utf-8", errors="replace")
            for item in raw_cmdline.split(b"\0")
            if item
        ]
        joined = " ".join(cmdline).lower()
        if "vllm" not in joined:
            continue

        env: dict[str, str] = {}
        for raw_item in proc_bytes(proc_dir / "environ").split(b"\0"):
            if b"=" not in raw_item:
                continue
            key, value = raw_item.split(b"=", 1)
            env[key.decode("utf-8", errors="replace")] = value.decode(
                "utf-8", errors="replace"
            )
        yield proc_dir, cmdline, env


def cwd_for_proc(proc_dir: Path) -> Path | None:
    try:
        return (proc_dir / "cwd").resolve()
    except OSError:
        return None


def normalize_profile_dir(path_value: str, base: Path | None) -> Path | None:
    if not path_value:
        return None
    if path_value.startswith("gs://"):
        return None
    path = Path(os.path.expandvars(os.path.expanduser(path_value)))
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def profiler_config_dirs(value: str, base: Path | None) -> list[Path]:
    dirs: list[Path] = []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("torch_profiler_dir", "torch-profiler-dir", "profile_dir"):
            maybe = normalize_profile_dir(str(parsed.get(key, "")), base)
            if maybe is not None:
                dirs.append(maybe)
    return dirs


def dirs_from_cmdline(cmdline: list[str], base: Path | None) -> list[Path]:
    dirs: list[Path] = []
    key_names = {
        "--profiler-config.torch_profiler_dir",
        "--profiler-config.torch-profiler-dir",
        "--torch-profiler-dir",
        "--torch_profiler_dir",
        "--profile-dir",
        "--profile_dir",
    }
    for idx, token in enumerate(cmdline):
        if token == "--profiler-config" and idx + 1 < len(cmdline):
            dirs.extend(profiler_config_dirs(cmdline[idx + 1], base))
            continue
        if token.startswith("--profiler-config="):
            dirs.extend(profiler_config_dirs(token.split("=", 1)[1], base))
            continue
        for key in key_names:
            if token == key and idx + 1 < len(cmdline):
                maybe = normalize_profile_dir(cmdline[idx + 1], base)
                if maybe is not None:
                    dirs.append(maybe)
            elif token.startswith(f"{key}="):
                maybe = normalize_profile_dir(token.split("=", 1)[1], base)
                if maybe is not None:
                    dirs.append(maybe)
    return dirs


def discover_profile_dirs(cli_dirs: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for value in cli_dirs:
        maybe = normalize_profile_dir(value, Path.cwd())
        if maybe is not None:
            dirs.append(maybe)

    for key in ("VLLM_TORCH_PROFILER_DIR", "VLLM_PROFILE_DIR", "PROFILE_DIR"):
        maybe = normalize_profile_dir(os.environ.get(key, ""), Path.cwd())
        if maybe is not None:
            dirs.append(maybe)

    for proc_dir, cmdline, env in iter_vllm_processes():
        base = cwd_for_proc(proc_dir)
        for key in ("VLLM_TORCH_PROFILER_DIR", "VLLM_PROFILE_DIR", "PROFILE_DIR"):
            maybe = normalize_profile_dir(env.get(key, ""), base)
            if maybe is not None:
                dirs.append(maybe)
        dirs.extend(dirs_from_cmdline(cmdline, base))

    cwd_default = Path.cwd() / "vllm_profile"
    if cwd_default.exists():
        dirs.append(cwd_default)

    unique: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def is_trace_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in TRACE_SUFFIXES)


def iter_recent_traces(profile_dirs: Iterable[Path], started_at: float) -> Iterable[Path]:
    for directory in profile_dirs:
        if not directory.exists():
            continue
        for root, subdirs, files in os.walk(directory):
            root_path = Path(root)
            if "done" in files:
                try:
                    if (root_path / "done").stat().st_mtime >= started_at:
                        yield root_path
                except OSError:
                    pass
            for filename in files:
                path = root_path / filename
                if not is_trace_file(path):
                    continue
                try:
                    if path.stat().st_mtime >= started_at:
                        yield path
                except OSError:
                    continue
            if len(root_path.relative_to(directory).parts) >= 8:
                subdirs[:] = []


def rank0_score(path: Path) -> int:
    text = str(path).lower()
    if RANK0_RE.search(text):
        return 3
    if "rank" not in text:
        return 1
    return 0


def choose_rank0_trace(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    scored = []
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        if rank0_score(path) == 0:
            continue
        scored.append((rank0_score(path), stat.st_mtime, stat.st_size, path))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][3]


def wait_for_rank0_trace(
    profile_dirs: list[Path], started_at: float, timeout: float, poll_interval: float
) -> Path:
    deadline = time.time() + timeout
    last_choice: Path | None = None
    last_size: int | None = None

    while time.time() < deadline:
        candidates = list(iter_recent_traces(profile_dirs, started_at))
        choice = choose_rank0_trace(candidates)
        if choice is not None:
            try:
                size = choice.stat().st_size
            except OSError:
                size = None
            if choice == last_choice and size is not None and size == last_size:
                return choice
            last_choice = choice
            last_size = size
        time.sleep(poll_interval)

    searched = ", ".join(str(path) for path in profile_dirs) or "<none>"
    raise TimeoutError(f"no rank-0 trace appeared before timeout; searched: {searched}")


def scp_trace(trace_path: Path, remote: str) -> None:
    subprocess.run(["scp", "-r", str(trace_path), remote], check=True)


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    profile_dirs = discover_profile_dirs(args.profile_dir)
    if not profile_dirs:
        raise RuntimeError(
            "no profile directory discovered; rerun with --profile-dir /absolute/profile/dir"
        )
    started_at = time.time() - 1.0

    print(f"posting {args.base_url.rstrip('/')}/start_profile for {args.steps} steps")
    post_start_profile(args.base_url, args.steps, args.prefix)

    print("profile dirs: " + ", ".join(str(path) for path in profile_dirs))

    trace = wait_for_rank0_trace(
        profile_dirs, started_at, args.wait_timeout, args.poll_interval
    )
    print(f"rank-0 trace: {trace}")
    print(f"copying to {args.remote}")
    scp_trace(trace, args.remote)
    print("done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
