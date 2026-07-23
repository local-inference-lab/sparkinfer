#!/usr/bin/env python3
"""Benchmark fused MLA query projection against the existing staged path.

The benchmark uses CUDA-graph replay for both arms, checks bitwise equality
before timing, records raw samples, and emits enough environment metadata to
reproduce a performance claim.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from sparkinfer import gemm
from sparkinfer.gemm import mla_query_projection


PACK_ROWS = 448
NOPE_DIM = 192
LATENT_DIM = 512
ROPE_DIM = 64
BMM_SPEC = {
    "a_dtype": "bfloat16",
    "b_dtype": "float8_e4m3fn",
    "sf_dtype": "float8_e8m0fnu",
    "c_dtype": "bfloat16",
    "sf_vec_size": 32,
    "b_major": "n",
    "sf_axis": "n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, choices=(8, 16), default=8)
    parser.add_argument("--m", type=int, choices=range(1, 33), default=1)
    parser.add_argument("--output-dtype", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def make_weight(
    *, heads: int, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    values = (
        torch.randn(
            heads * PACK_ROWS,
            LATENT_DIM,
            device=device,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    scales = torch.randint(
        118,
        132,
        (heads * PACK_ROWS, LATENT_DIM // 32),
        device=device,
        generator=generator,
        dtype=torch.uint8,
    )
    return (
        values.view(heads, PACK_ROWS, LATENT_DIM)[:, :NOPE_DIM, :],
        scales.view(heads, PACK_ROWS, LATENT_DIM // 32)[:, :NOPE_DIM, :],
    )


def capture_graph(fn: Callable[[], None]) -> Callable[[], None]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay() -> None:
        graph.replay()

    replay()
    torch.cuda.synchronize()
    return replay


def balanced_samples_us(
    baseline: Callable[[], None],
    fused: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> tuple[list[float], list[float]]:
    for index in range(warmup):
        (baseline if index % 2 == 0 else fused)()
        (fused if index % 2 == 0 else baseline)()
    torch.cuda.synchronize()

    baseline_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    fused_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def record(
        fn: Callable[[], None],
        destination: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        destination.append((start, end))

    for index in range(iters):
        if index % 2 == 0:
            record(baseline, baseline_events)
            record(fused, fused_events)
        else:
            record(fused, fused_events)
            record(baseline, baseline_events)
    torch.cuda.synchronize()
    return (
        [start.elapsed_time(end) * 1000.0 for start, end in baseline_events],
        [start.elapsed_time(end) * 1000.0 for start, end in fused_events],
    )


def git_revision(repo: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def git_is_dirty(repo: pathlib.Path) -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
    )


def gpu_snapshot() -> list[str]:
    query = (
        "index,name,uuid,pstate,clocks.current.sm,clocks.current.memory,"
        "power.limit,clocks_throttle_reasons.active"
    )
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return output.strip().splitlines()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", args.device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise RuntimeError("an SM120/SM121 GPU is required")
    torch.cuda.set_device(device)

    output_dtype = (
        torch.bfloat16 if args.output_dtype == "bf16" else torch.float8_e4m3fn
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    weight = make_weight(heads=args.heads, device=device, generator=generator)
    q_nope = torch.randn(
        args.heads,
        args.m,
        NOPE_DIM,
        device=device,
        generator=generator,
        dtype=torch.bfloat16,
    )
    q_full = torch.randn(
        args.m,
        args.heads,
        LATENT_DIM + ROPE_DIM,
        device=device,
        generator=generator,
        dtype=torch.bfloat16,
    )
    q_pe = q_full[..., LATENT_DIM:]
    q_scale = torch.tensor([0.037], device=device, dtype=torch.float32)

    gemm.prewarm_bmm(weight, [args.m], **BMM_SPEC)
    mla_query_projection.prewarm(
        weight, [args.m], output_dtype=output_dtype
    )

    projected = torch.empty(
        args.heads,
        args.m,
        LATENT_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    assembled_bf16 = torch.empty(
        args.m,
        args.heads,
        LATENT_DIM + ROPE_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    scaled_fp32 = torch.empty_like(assembled_bf16, dtype=torch.float32)
    baseline_out = (
        assembled_bf16
        if output_dtype == torch.bfloat16
        else torch.empty_like(assembled_bf16, dtype=output_dtype)
    )
    fused_out = torch.empty_like(assembled_bf16, dtype=output_dtype)
    inv_scale = torch.reciprocal(q_scale)

    def baseline() -> None:
        gemm.bmm(q_nope, weight, projected, **BMM_SPEC)
        torch.cat((projected.transpose(0, 1), q_pe), dim=-1, out=assembled_bf16)
        if output_dtype == torch.float8_e4m3fn:
            scaled_fp32.copy_(assembled_bf16)
            scaled_fp32.mul_(inv_scale)
            scaled_fp32.clamp_(-448.0, 448.0)
            baseline_out.copy_(scaled_fp32)

    def fused() -> None:
        mla_query_projection.run(
            q_nope,
            weight,
            q_pe,
            fused_out,
            q_scale=q_scale if output_dtype == torch.float8_e4m3fn else None,
        )

    baseline()
    fused()
    torch.cuda.synchronize()
    bitwise_equal = torch.equal(
        baseline_out.view(torch.uint8), fused_out.view(torch.uint8)
    )
    finite = bool(torch.isfinite(fused_out.float()).all().item())
    nonzero = bool(torch.count_nonzero(fused_out).item())
    if not bitwise_equal or not finite or not nonzero:
        raise RuntimeError(
            "correctness gate failed: "
            f"bitwise_equal={bitwise_equal}, finite={finite}, nonzero={nonzero}"
        )

    baseline_graph = capture_graph(baseline)
    fused_graph = capture_graph(fused)
    baseline_us, fused_us = balanced_samples_us(
        baseline_graph,
        fused_graph,
        warmup=args.warmup,
        iters=args.iters,
    )
    baseline_median = statistics.median(baseline_us)
    fused_median = statistics.median(fused_us)
    repo = pathlib.Path(__file__).resolve().parents[1]
    result = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "commit": git_revision(repo),
        "worktree": str(repo),
        "git_dirty": git_is_dirty(repo),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "gpu_snapshot": gpu_snapshot(),
        "correctness": {
            "bitwise_equal": bitwise_equal,
            "finite": finite,
            "nonzero": nonzero,
            "cuda_graph_replay": True,
        },
        "shape": {
            "heads": args.heads,
            "m": args.m,
            "nope_dim": NOPE_DIM,
            "latent_dim": LATENT_DIM,
            "rope_dim": ROPE_DIM,
            "output_dtype": args.output_dtype,
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "ratio_direction": "fused_median_us / staged_median_us; lower is better",
        "staged_median_us": baseline_median,
        "fused_median_us": fused_median,
        "fused_over_staged": fused_median / baseline_median,
        "staged_raw_us": baseline_us,
        "fused_raw_us": fused_us,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
