#!/usr/bin/env python3
"""Benchmark fused b12x mHC residual pre/post kernels."""

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
    bench_gpu_ms,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
)
from b12x.integration import (
    b12x_mhc_post,
    b12x_mhc_pre,
    empty_mhc_workspace,
)


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)
    return y, post, comb


def _mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(x.dtype)


def _make_inputs(
    *,
    tokens: int,
    hidden_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    residual = (
        torch.randn((tokens, 4, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 3
    ).to(torch.bfloat16)
    x = (
        torch.randn((tokens, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 4
    ).to(torch.bfloat16)
    fn = torch.randn((24, 4 * hidden_size), generator=gen, dtype=torch.float32).to(device) / 64
    scale = torch.randn((3,), generator=gen, dtype=torch.float32).to(device) / 3
    bias = torch.randn((24,), generator=gen, dtype=torch.float32).to(device) / 5
    return residual.contiguous(), x.contiguous(), fn.contiguous(), scale.contiguous(), bias.contiguous()


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = actual.float() - expected.float()
    return float(diff.abs().max().item()), float(torch.sqrt(torch.mean(diff * diff)).item())


def _bench_graph(fn, *, warmup: int, iters: int, l2_flush) -> tuple[float, float]:
    graph = capture_cuda_graph(fn, warmup=warmup)
    stats = bench_cuda_graph(graph, replays=iters, l2_flush=l2_flush)
    samples = stats["replay_us"]
    return statistics.median(samples), min(samples)


def _bench_eager(fn, *, warmup: int, iters: int, l2_flush) -> tuple[float, float]:
    samples = []
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        samples.append(bench_gpu_ms(fn, warmup=0, iters=1, l2_flush=l2_flush) * 1000.0)
    return statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--split-k", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--block-h", type=int, default=512)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=91_500)
    args = parser.parse_args()

    device = require_sm120()
    residual, x, fn, scale, bias = _make_inputs(
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        seed=args.seed,
        device=device,
    )
    workspace = empty_mhc_workspace(
        num_tokens=args.tokens,
        hidden_size=args.hidden_size,
        split_k=args.split_k,
        device=device,
    )
    y = workspace.y
    post = workspace.post
    comb = workspace.comb
    out = workspace.out

    def run_pre() -> None:
        b12x_mhc_pre(
            residual,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            workspace=workspace,
            y_out=y,
            post_out=post,
            comb_out=comb,
            split_k=args.split_k,
            block_k=args.block_k,
            block_h=args.block_h,
        )

    def run_post() -> None:
        b12x_mhc_post(
            x,
            residual,
            post,
            comb,
            workspace=workspace,
            out=out,
            block_h=args.block_h,
        )

    def run_pair() -> None:
        run_pre()
        run_post()

    run_pair()
    torch.cuda.synchronize()

    if not args.skip_check:
        y_ref, post_ref, comb_ref = _mhc_pre_reference(
            residual,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
        )
        out_ref = _mhc_post_reference(x, residual, post, comb)
        y_max, y_rmse = _error_stats(y, y_ref)
        post_max, _ = _error_stats(post, post_ref)
        comb_max, _ = _error_stats(comb, comb_ref)
        out_max, out_rmse = _error_stats(out, out_ref)
    else:
        y_max = y_rmse = post_max = comb_max = out_max = out_rmse = float("nan")

    l2_flush = make_l2_flush_fn(args.l2_flush, args.l2_flush_bytes)
    if args.eager:
        pre_median, pre_min = _bench_eager(
            run_pre,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        post_median, post_min = _bench_eager(
            run_post,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        pair_median, pair_min = _bench_eager(
            run_pair,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        mode = "eager"
    else:
        pre_median, pre_min = _bench_graph(
            run_pre,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        post_median, post_min = _bench_graph(
            run_post,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        pair_median, pair_min = _bench_graph(
            run_pair,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        mode = "graph"

    print(
        "residual_mhc "
        f"mode={mode} tokens={args.tokens} hidden={args.hidden_size} "
        f"split_k={args.split_k} block_k={args.block_k} block_h={args.block_h} "
        f"pre_us={pre_median:.2f}/{pre_min:.2f} "
        f"post_us={post_median:.2f}/{post_min:.2f} "
        f"pair_us={pair_median:.2f}/{pair_min:.2f} "
        f"y_max={y_max:.3g} y_rmse={y_rmse:.3g} "
        f"post_max={post_max:.3g} comb_max={comb_max:.3g} "
        f"out_max={out_max:.3g} out_rmse={out_rmse:.3g}"
    )


if __name__ == "__main__":
    main()
