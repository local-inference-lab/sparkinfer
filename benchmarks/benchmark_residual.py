#!/usr/bin/env python3
"""Benchmark the fused b12x mHC residual post_pre (Gram) kernel vs vLLM."""

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
    b12x_mhc_post_pre,
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
    y_dtype: torch.dtype | None = None,
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
    y = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    y = y.to(residual.dtype if y_dtype is None else y_dtype)
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


def _post_pre_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the fused post_pre. RMSNorm variance is taken in fp32 from
    the collapsed activation (matching the Gram kernel and vLLM, which both
    compute the norm in fp32 rather than from the bf16-rounded activation)."""
    residual_out = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_raw_fp32, post, comb = _mhc_pre_reference(
        residual_out,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        y_dtype=torch.float32,
    )
    if norm_weight is not None:
        rms = torch.rsqrt(y_raw_fp32.square().mean(dim=-1, keepdim=True) + norm_eps)
        y = (
            y_raw_fp32.to(torch.bfloat16).float() * rms * norm_weight.float()
        ).to(torch.bfloat16)
    else:
        y = y_raw_fp32.to(torch.bfloat16)
    return residual_out, y, post, comb


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


def _register_vllm_mhc_tilelang(vllm_path: pathlib.Path) -> None:
    vllm_root = str(vllm_path.expanduser().resolve())
    if vllm_root not in sys.path:
        sys.path.insert(1, vllm_root)
    import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401

    if not hasattr(torch.ops.vllm, "mhc_fused_post_pre_tilelang"):
        raise RuntimeError("vLLM mhc_fused_post_pre_tilelang custom op was not registered")


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
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=91_500)
    parser.add_argument(
        "--compare-vllm",
        action="store_true",
        help="Also benchmark vLLM's DeepSeek TileLang fused post_pre op.",
    )
    parser.add_argument(
        "--vllm-path",
        type=pathlib.Path,
        default=pathlib.Path("~/projects/vllm-other"),
        help="Path to the vLLM checkout used for --compare-vllm.",
    )
    args = parser.parse_args()

    device = require_sm120()
    if args.compare_vllm:
        _register_vllm_mhc_tilelang(args.vllm_path)

    residual, x, fn, scale, bias = _make_inputs(
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        seed=args.seed,
        device=device,
    )
    fused_workspace = empty_mhc_workspace(
        num_tokens=args.tokens,
        hidden_size=args.hidden_size,
        split_k=args.split_k,
        device=device,
    )
    fused_y = fused_workspace.y
    fused_post = fused_workspace.post
    fused_comb = fused_workspace.comb
    fused_out = fused_workspace.out
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=args.rms_eps,
        hc_eps=args.hc_eps,
        sinkhorn_iters=args.sinkhorn_iters,
    )
    prev_post = prev_post.contiguous()
    prev_comb = prev_comb.contiguous()
    norm_weight = None
    if args.fuse_rmsnorm:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.seed + 17)
        norm_weight = (
            torch.randn((args.hidden_size,), generator=gen, dtype=torch.float32)
            .to(device)
            .to(torch.bfloat16)
            .contiguous()
        )

    def run_fused() -> None:
        b12x_mhc_post_pre(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            workspace=fused_workspace,
            residual_out=fused_out,
            y_out=fused_y,
            post_out=fused_post,
            comb_out=fused_comb,
            norm_weight=norm_weight,
            norm_eps=args.norm_eps,
            split_k=args.split_k,
            block_k=args.block_k,
            block_h=args.block_h,
        )

    vllm_out = vllm_post = vllm_comb = vllm_y = None

    def run_vllm_fused() -> None:
        nonlocal vllm_out, vllm_post, vllm_comb, vllm_y
        vllm_out, vllm_post, vllm_comb, vllm_y = torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            args.rms_eps,
            args.hc_eps,
            args.hc_eps,
            2.0,
            args.sinkhorn_iters,
            1,
            1,
            norm_weight,
            args.norm_eps if norm_weight is not None else 0.0,
        )

    run_fused()
    if args.compare_vllm:
        run_vllm_fused()
    torch.cuda.synchronize()

    if not args.skip_check:
        out_ref, y_ref, post_ref, comb_ref = _post_pre_reference(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=args.norm_eps,
        )
        fused_y_max, fused_y_rmse = _error_stats(fused_y, y_ref)
        fused_post_max, _ = _error_stats(fused_post, post_ref)
        fused_comb_max, _ = _error_stats(fused_comb, comb_ref)
        fused_out_max, fused_out_rmse = _error_stats(fused_out, out_ref)
        if args.compare_vllm:
            assert vllm_out is not None and vllm_post is not None
            assert vllm_comb is not None and vllm_y is not None
            vllm_y_max, vllm_y_rmse = _error_stats(vllm_y, y_ref)
            vllm_post_max, _ = _error_stats(vllm_post.squeeze(-1), post_ref)
            vllm_comb_max, _ = _error_stats(vllm_comb, comb_ref)
            vllm_out_max, vllm_out_rmse = _error_stats(vllm_out, out_ref)
        else:
            vllm_y_max = vllm_y_rmse = vllm_post_max = vllm_comb_max = float("nan")
            vllm_out_max = vllm_out_rmse = float("nan")
    else:
        fused_y_max = fused_y_rmse = fused_post_max = fused_comb_max = float("nan")
        fused_out_max = fused_out_rmse = float("nan")
        vllm_y_max = vllm_y_rmse = vllm_post_max = vllm_comb_max = float("nan")
        vllm_out_max = vllm_out_rmse = float("nan")

    l2_flush = make_l2_flush_fn(args.l2_flush, args.l2_flush_bytes)
    bench = _bench_eager if args.eager else _bench_graph
    fused_median, fused_min = bench(
        run_fused, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush
    )
    if args.compare_vllm:
        vllm_median, vllm_min = bench(
            run_vllm_fused, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush
        )
    else:
        vllm_median = vllm_min = float("nan")
    mode = "eager" if args.eager else "graph"

    line = (
        "residual_mhc "
        f"mode={mode} tokens={args.tokens} hidden={args.hidden_size} "
        f"split_k={args.split_k} block_k={args.block_k} block_h={args.block_h} "
        f"post_pre_us={fused_median:.2f}/{fused_min:.2f} "
        f"fused_y_max={fused_y_max:.3g} fused_y_rmse={fused_y_rmse:.3g} "
        f"fused_post_max={fused_post_max:.3g} fused_comb_max={fused_comb_max:.3g} "
        f"fused_out_max={fused_out_max:.3g} fused_out_rmse={fused_out_rmse:.3g}"
    )
    if args.compare_vllm:
        line += (
            f" vllm_post_pre_us={vllm_median:.2f}/{vllm_min:.2f} "
            f"b12x_vs_vllm={vllm_median / fused_median:.3f}x "
            f"vllm_y_max={vllm_y_max:.3g} vllm_y_rmse={vllm_y_rmse:.3g} "
            f"vllm_post_max={vllm_post_max:.3g} vllm_comb_max={vllm_comb_max:.3g} "
            f"vllm_out_max={vllm_out_max:.3g} vllm_out_rmse={vllm_out_rmse:.3g}"
        )
    print(line)


if __name__ == "__main__":
    main()
