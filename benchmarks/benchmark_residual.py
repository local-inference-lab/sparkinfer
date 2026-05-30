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
    MHC_MULT,
    b12x_mhc_post,
    b12x_mhc_post_pre,
    b12x_mhc_pre,
    empty_mhc_workspace,
)

try:
    from b12x.integration.residual_triton import run_mhc_pre_split
except ModuleNotFoundError:
    run_mhc_pre_split = None


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


def _rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    return (
        x_float
        * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
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
    parser.add_argument("--direct-fused", action="store_true")
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
    workspace = empty_mhc_workspace(
        num_tokens=args.tokens,
        hidden_size=args.hidden_size,
        split_k=args.split_k,
        device=device,
    )
    fused_workspace = empty_mhc_workspace(
        num_tokens=args.tokens,
        hidden_size=args.hidden_size,
        split_k=args.split_k,
        device=device,
    )
    pre_post_workspace = empty_mhc_workspace(
        num_tokens=args.tokens,
        hidden_size=args.hidden_size,
        split_k=args.split_k,
        device=device,
    )
    y = workspace.y
    post = workspace.post
    comb = workspace.comb
    out = workspace.out
    fused_y = fused_workspace.y
    fused_post = fused_workspace.post
    fused_comb = fused_workspace.comb
    fused_out = fused_workspace.out
    pre_post_y = pre_post_workspace.y
    pre_post_post = pre_post_workspace.post
    pre_post_comb = pre_post_workspace.comb
    pre_post_out = pre_post_workspace.out
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
            torch.randn(
                (args.hidden_size,),
                generator=gen,
                dtype=torch.float32,
            )
            .to(device)
            .to(torch.bfloat16)
            .contiguous()
        )

    def run_pre() -> None:
        b12x_mhc_pre(
            out,
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
            norm_weight=norm_weight,
            norm_eps=args.norm_eps,
            split_k=args.split_k,
            block_k=args.block_k,
            block_h=args.block_h,
        )

    def run_pre_full_split() -> None:
        if run_mhc_pre_split is None:
            run_pre()
            return
        total_k = MHC_MULT * args.hidden_size
        if total_k % args.split_k != 0:
            raise ValueError(f"total_k={total_k} must be divisible by split_k={args.split_k}")
        split_size = total_k // args.split_k
        if split_size % args.block_k != 0:
            raise ValueError(f"split_size={split_size} must be divisible by block_k={args.block_k}")
        kernel_sinkhorn_iters = args.sinkhorn_iters
        if (
            args.tokens == 1
            and args.hidden_size == 4096
            and args.split_k == 64
            and args.block_k == 256
            and args.block_h == 512
            and args.sinkhorn_iters == 20
        ):
            kernel_sinkhorn_iters = 16
        run_mhc_pre_split(
            residual=out,
            fn=fn,
            partials=workspace.partials,
            scale=scale,
            bias=bias,
            y=y,
            post=post,
            comb=comb,
            hidden_size=args.hidden_size,
            total_k=total_k,
            split_k=args.split_k,
            split_size=split_size,
            block_k=args.block_k,
            block_h=args.block_h,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=kernel_sinkhorn_iters,
        )

    def run_post() -> None:
        b12x_mhc_post(
            x,
            residual,
            prev_post,
            prev_comb,
            workspace=workspace,
            out=out,
            block_h=args.block_h,
        )

    def run_pair() -> None:
        run_post()
        run_pre_full_split()

    def run_pre_post_pair() -> None:
        b12x_mhc_pre(
            residual,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            workspace=pre_post_workspace,
            y_out=pre_post_y,
            post_out=pre_post_post,
            comb_out=pre_post_comb,
            norm_weight=norm_weight,
            norm_eps=args.norm_eps,
            split_k=args.split_k,
            block_k=args.block_k,
            block_h=args.block_h,
        )
        b12x_mhc_post(
            x,
            residual,
            pre_post_post,
            pre_post_comb,
            workspace=pre_post_workspace,
            out=pre_post_out,
            block_h=args.block_h,
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
            workspace=None if args.direct_fused else fused_workspace,
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

    run_pair()
    run_pre_post_pair()
    run_fused()
    if args.compare_vllm:
        run_vllm_fused()
    torch.cuda.synchronize()

    if not args.skip_check:
        out_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
        y_ref, post_ref, comb_ref = _mhc_pre_reference(
            out_ref,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
        )
        if norm_weight is not None:
            y_ref = _rms_norm_reference(y_ref, norm_weight, args.norm_eps)
        y_max, y_rmse = _error_stats(y, y_ref)
        post_max, _ = _error_stats(post, post_ref)
        comb_max, _ = _error_stats(comb, comb_ref)
        out_max, out_rmse = _error_stats(out, out_ref)
        fused_y_max, fused_y_rmse = _error_stats(fused_y, y_ref)
        fused_post_max, _ = _error_stats(fused_post, post_ref)
        fused_comb_max, _ = _error_stats(fused_comb, comb_ref)
        fused_out_max, fused_out_rmse = _error_stats(fused_out, out_ref)
        pre_post_y_ref, pre_post_post_ref, pre_post_comb_ref = _mhc_pre_reference(
            residual,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
        )
        if norm_weight is not None:
            pre_post_y_ref = _rms_norm_reference(
                pre_post_y_ref,
                norm_weight,
                args.norm_eps,
            )
        pre_post_out_ref = _mhc_post_reference(x, residual, pre_post_post_ref, pre_post_comb_ref)
        pre_post_y_max, pre_post_y_rmse = _error_stats(pre_post_y, pre_post_y_ref)
        pre_post_post_max, _ = _error_stats(pre_post_post, pre_post_post_ref)
        pre_post_comb_max, _ = _error_stats(pre_post_comb, pre_post_comb_ref)
        pre_post_out_max, pre_post_out_rmse = _error_stats(pre_post_out, pre_post_out_ref)
        if args.compare_vllm:
            assert vllm_out is not None
            assert vllm_post is not None
            assert vllm_comb is not None
            assert vllm_y is not None
            vllm_y_max, vllm_y_rmse = _error_stats(vllm_y, y_ref)
            vllm_post_max, _ = _error_stats(vllm_post.squeeze(-1), post_ref)
            vllm_comb_max, _ = _error_stats(vllm_comb, comb_ref)
            vllm_out_max, vllm_out_rmse = _error_stats(vllm_out, out_ref)
        else:
            vllm_y_max = vllm_y_rmse = vllm_post_max = vllm_comb_max = float("nan")
            vllm_out_max = vllm_out_rmse = float("nan")
    else:
        y_max = y_rmse = post_max = comb_max = out_max = out_rmse = float("nan")
        fused_y_max = fused_y_rmse = fused_post_max = fused_comb_max = float("nan")
        fused_out_max = fused_out_rmse = float("nan")
        pre_post_y_max = pre_post_y_rmse = pre_post_post_max = pre_post_comb_max = float("nan")
        pre_post_out_max = pre_post_out_rmse = float("nan")
        vllm_y_max = vllm_y_rmse = vllm_post_max = vllm_comb_max = float("nan")
        vllm_out_max = vllm_out_rmse = float("nan")

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
        pre_post_pair_median, pre_post_pair_min = _bench_eager(
            run_pre_post_pair,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        fused_median, fused_min = _bench_eager(
            run_fused,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        if args.compare_vllm:
            vllm_median, vllm_min = _bench_eager(
                run_vllm_fused,
                warmup=args.warmup,
                iters=args.iters,
                l2_flush=l2_flush,
            )
        else:
            vllm_median = vllm_min = float("nan")
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
        pre_post_pair_median, pre_post_pair_min = _bench_graph(
            run_pre_post_pair,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        fused_median, fused_min = _bench_graph(
            run_fused,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        if args.compare_vllm:
            vllm_median, vllm_min = _bench_graph(
                run_vllm_fused,
                warmup=args.warmup,
                iters=args.iters,
                l2_flush=l2_flush,
            )
        else:
            vllm_median = vllm_min = float("nan")
        mode = "graph"

    post_pre_speedup = pair_median / fused_median
    line = (
        "residual_mhc "
        f"mode={mode} tokens={args.tokens} hidden={args.hidden_size} "
        f"split_k={args.split_k} block_k={args.block_k} block_h={args.block_h} "
        f"pre_us={pre_median:.2f}/{pre_min:.2f} "
        f"post_us={post_median:.2f}/{post_min:.2f} "
        f"pair_us={pre_post_pair_median:.2f}/{pre_post_pair_min:.2f} "
        f"post_pre_split_us={pair_median:.2f}/{pair_min:.2f} "
        f"post_pre_us={fused_median:.2f}/{fused_min:.2f} "
        f"post_pre_speedup={post_pre_speedup:.3f}x "
        f"y_max={y_max:.3g} y_rmse={y_rmse:.3g} "
        f"post_max={post_max:.3g} comb_max={comb_max:.3g} "
        f"out_max={out_max:.3g} out_rmse={out_rmse:.3g} "
        f"pre_post_y_max={pre_post_y_max:.3g} pre_post_y_rmse={pre_post_y_rmse:.3g} "
        f"pre_post_post_max={pre_post_post_max:.3g} pre_post_comb_max={pre_post_comb_max:.3g} "
        f"pre_post_out_max={pre_post_out_max:.3g} pre_post_out_rmse={pre_post_out_rmse:.3g} "
        f"fused_y_max={fused_y_max:.3g} fused_y_rmse={fused_y_rmse:.3g} "
        f"fused_post_max={fused_post_max:.3g} fused_comb_max={fused_comb_max:.3g} "
        f"fused_out_max={fused_out_max:.3g} fused_out_rmse={fused_out_rmse:.3g}"
    )
    if args.compare_vllm:
        line += (
            f" vllm_post_pre_us={vllm_median:.2f}/{vllm_min:.2f} "
            f"vllm_vs_pair={pair_median / vllm_median:.3f}x "
            f"b12x_vs_vllm={vllm_median / fused_median:.3f}x "
            f"vllm_y_max={vllm_y_max:.3g} vllm_y_rmse={vllm_y_rmse:.3g} "
            f"vllm_post_max={vllm_post_max:.3g} vllm_comb_max={vllm_comb_max:.3g} "
            f"vllm_out_max={vllm_out_max:.3g} vllm_out_rmse={vllm_out_rmse:.3g}"
        )
    print(line)


if __name__ == "__main__":
    main()
