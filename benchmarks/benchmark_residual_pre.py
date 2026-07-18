#!/usr/bin/env python3
"""Benchmark the first-layer B12X mHC broadcast pre kernel."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
)
from b12x.integration import b12x_mhc_pre


def _reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_fp32 = residual.float()
    mixes = F.linear(residual_fp32, fn) * torch.rsqrt(
        residual_fp32.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    y_raw = pre.sum(dim=-1, keepdim=True) * residual_fp32
    norm_scale = torch.rsqrt(y_raw.square().mean(dim=-1, keepdim=True) + norm_eps)
    y = (y_raw.to(torch.bfloat16).float() * norm_scale * norm_weight.float()).to(
        torch.bfloat16
    )
    residual_out = residual.unsqueeze(1).expand(-1, 4, -1)
    return residual_out, post, comb, y


def _error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=91_400)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    args = parser.parse_args()

    device = require_sm120()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    residual = (
        torch.randn(
            (args.tokens, args.hidden_size),
            generator=generator,
            dtype=torch.float32,
        )
        .to(device)
        .to(torch.bfloat16)
        .contiguous()
    )
    fn = (
        torch.randn(
            (24, args.hidden_size),
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        / 64
    ).contiguous()
    scale = (
        torch.randn((3,), generator=generator, dtype=torch.float32).to(device) / 3
    ).contiguous()
    bias = (
        torch.randn((24,), generator=generator, dtype=torch.float32).to(device) / 5
    ).contiguous()
    norm_weight = (
        torch.randn((args.hidden_size,), generator=generator, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16)
        .contiguous()
    )
    output: tuple[torch.Tensor, ...] | None = None

    def run() -> None:
        nonlocal output
        output = b12x_mhc_pre(
            residual,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=args.norm_eps,
        )

    run()
    torch.cuda.synchronize(device)
    assert output is not None
    if not args.skip_check:
        expected = _reference(
            residual,
            fn,
            scale,
            bias,
            norm_weight,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            norm_eps=args.norm_eps,
        )
        errors = tuple(
            _error(actual, reference)
            for actual, reference in zip(output, expected, strict=True)
        )
    else:
        errors = (float("nan"),) * 4

    graph = capture_cuda_graph(run, warmup=args.warmup)
    stats = bench_cuda_graph(
        graph,
        replays=args.iters,
        l2_flush=make_l2_flush_fn(args.l2_flush, args.l2_flush_bytes),
    )
    samples = stats["replay_us"]
    print(
        "residual_mhc_pre "
        f"mode=graph tokens={args.tokens} hidden={args.hidden_size} "
        f"pre_us={statistics.median(samples):.2f}/{min(samples):.2f} "
        f"out_max={errors[0]:.3g} post_max={errors[1]:.3g} "
        f"comb_max={errors[2]:.3g} y_max={errors[3]:.3g}"
    )


if __name__ == "__main__":
    main()
