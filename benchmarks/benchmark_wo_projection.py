#!/usr/bin/env python3
"""Benchmark DeepSeek-style WO-A/WO-B projection candidates.

This benchmark times the explicit native MXFP8 two-GEMM skeleton:

    WO-A: [tokens, group_width, groups] x [rank, group_width, groups]
    tmp:  [tokens, rank, groups] -> group-major [tokens, groups * rank]
    WO-B: [tokens, groups * rank] x [hidden, groups * rank]

The b12x path uses owned GPU quant/packing kernels for the activation operands
around the two native MXFP8 dense GEMMs. Weight quantization is still setup
work, matching model-load behavior rather than the per-token serving path.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from b12x.cute.utils import get_hardware_info
from b12x.gemm.wo_projection import (
    WOProjectionMXFP8Weights,
    dequantize_mxfp8_rows_torch,
    empty_wo_projection_workspace,
    quantize_mxfp8_rows_torch,
    wo_projection_mxfp8,
)


REFERENCE_LABEL = "PyTorch graph BF16 einsum+matmul"
COSINE_THRESHOLD = 0.995
_L2_FLUSH_BUFFER_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_AUTO_L2_FLUSH_MULTIPLIER = 2
_FALLBACK_L2_FLUSH_BYTES = 32 << 20


class BenchmarkAbort(RuntimeError):
    """Fatal benchmark failure that should stop the run without a summary."""


class CorrectnessError(BenchmarkAbort):
    """Raised when replay outputs fail the correctness gate."""


def resolve_l2_flush_bytes(bytes_hint: int) -> int:
    if bytes_hint < 0:
        raise ValueError(f"l2 flush bytes must be non-negative, got {bytes_hint}")
    if bytes_hint > 0:
        return int(bytes_hint)
    try:
        l2_bytes = int(get_hardware_info().get_l2_cache_size_in_bytes())
    except Exception:
        l2_bytes = 0
    if l2_bytes > 0:
        return l2_bytes * _AUTO_L2_FLUSH_MULTIPLIER
    return _FALLBACK_L2_FLUSH_BYTES


def make_l2_flush_fn(*, bytes_hint: int = 0) -> Callable[[], None]:
    flush_bytes = resolve_l2_flush_bytes(bytes_hint)
    device_idx = torch.cuda.current_device()
    key = (device_idx, flush_bytes)
    buffer = _L2_FLUSH_BUFFER_CACHE.get(key)
    if buffer is None:
        buffer = torch.empty(flush_bytes, dtype=torch.uint8, device=f"cuda:{device_idx}")
        _L2_FLUSH_BUFFER_CACHE[key] = buffer

    def flush(cache_buffer: torch.Tensor = buffer) -> None:
        cache_buffer.bitwise_not_()

    return flush


def require_sm120() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")


def bench_events(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    l2_flush: Callable[[], None],
) -> list[float]:
    for _ in range(warmup):
        l2_flush()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        l2_flush()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def fmt_us(times_ms: list[float]) -> str:
    med = statistics.median(times_ms) * 1000.0
    mn = min(times_ms) * 1000.0
    return f"{med:8.1f} us (min {mn:.1f})"


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return F.cosine_similarity(a_f, b_f, dim=0).item()


def check_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    label: str,
) -> None:
    cand_finite = bool(torch.isfinite(candidate).all().item())
    ref_finite = bool(torch.isfinite(reference).all().item())
    if not cand_finite or not ref_finite:
        raise CorrectnessError(
            f"non-finite output detected vs {label}: "
            f"candidate_finite={cand_finite}, reference_finite={ref_finite}"
        )
    diff = (candidate.float() - reference.float()).abs()
    max_abs = diff.max().item()
    rmse = diff.square().mean().sqrt().item()
    cos = cosine_similarity(candidate, reference)
    print(
        f"    check vs {label}: max_abs={max_abs:.8f} "
        f"rmse={rmse:.8f} cos={cos:.10f}"
    )
    if not math.isfinite(cos) or cos < COSINE_THRESHOLD:
        raise CorrectnessError(
            f"cosine similarity vs {label} fell below {COSINE_THRESHOLD:.6f}: "
            f"max_abs={max_abs:.8f}, rmse={rmse:.8f}, cos={cos}"
        )


def capture_graph_replay(fn: Callable[[], torch.Tensor | None]) -> tuple[Callable[[], None], torch.Tensor | None]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    replay()
    torch.cuda.synchronize()
    return replay, graph_output


def make_case(
    *,
    tokens: int,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
    seed: int,
) -> dict[str, torch.Tensor | object]:
    torch.manual_seed(seed)
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    ).contiguous()
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    ).contiguous()
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    ).contiguous()

    x_tdg_q = quantize_mxfp8_rows_torch(x_tgd.permute(0, 2, 1).contiguous())
    wo_a_rdg_q = quantize_mxfp8_rows_torch(wo_a_grd.permute(1, 2, 0).contiguous())
    wo_b_hrg_q = quantize_mxfp8_rows_torch(wo_b_hgr)
    weights = WOProjectionMXFP8Weights(
        wo_a=wo_a_rdg_q,
        wo_b=wo_b_hrg_q,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )

    x_deq_tgd = dequantize_mxfp8_rows_torch(
        x_tdg_q.values,
        x_tdg_q.scale_rows,
    ).permute(0, 2, 1).to(torch.bfloat16)
    wo_a_deq_grd = dequantize_mxfp8_rows_torch(
        wo_a_rdg_q.values,
        wo_a_rdg_q.scale_rows,
    ).permute(2, 0, 1).to(torch.bfloat16)
    wo_b_deq_hgr = dequantize_mxfp8_rows_torch(
        wo_b_hrg_q.values,
        wo_b_hrg_q.scale_rows,
    ).to(torch.bfloat16)

    return {
        "x_tgd": x_tgd,
        "wo_a_grd": wo_a_grd,
        "wo_b_hgr": wo_b_hgr,
        "x_tdg_q": x_tdg_q,
        "wo_a_rdg_q": wo_a_rdg_q,
        "wo_b_hrg_q": wo_b_hrg_q,
        "weights": weights,
        "x_deq_tgd": x_deq_tgd,
        "wo_a_deq_grd": wo_a_deq_grd,
        "wo_b_deq_hgr": wo_b_deq_hgr,
    }


def bench_one(
    tokens: int,
    *,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None],
    seed: int,
) -> dict[str, object]:
    case = make_case(
        tokens=tokens,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
        seed=seed,
    )

    results: dict[str, object] = {}

    try:
        workspace = empty_wo_projection_workspace(
            tokens,
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
            device="cuda",
        )

        def b12x_launch() -> torch.Tensor:
            return wo_projection_mxfp8(
                case["x_tgd"],
                case["weights"],
                workspace,
                return_3d=True,
            )

        b12x_replay, _ = capture_graph_replay(b12x_launch)
        results["b12x_replay"] = b12x_replay
        results["b12x_out"] = workspace.output
        results["b12x"] = bench_events(
            b12x_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results["b12x"] = None
        print(f"      b12x two-GEMM FAILED: {exc}")

    try:
        torch_outputs: list[torch.Tensor | None] = [None]

        def torch_launch() -> torch.Tensor:
            tmp_ref = torch.einsum(
                "tgd,grd->tgr",
                case["x_deq_tgd"],
                case["wo_a_deq_grd"],
            )
            out = tmp_ref.reshape(tokens, groups * rank) @ case["wo_b_deq_hgr"].T
            torch_outputs[0] = out
            return out

        torch_replay, torch_graph_out = capture_graph_replay(torch_launch)
        if torch_graph_out is None:
            torch_graph_out = torch_outputs[0]
        results["torch_replay"] = torch_replay
        results["torch_out"] = torch_graph_out
        results[REFERENCE_LABEL] = bench_events(
            torch_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results[REFERENCE_LABEL] = None
        print(f"      {REFERENCE_LABEL} FAILED: {exc}")

    if check:
        if results.get("b12x_replay") is None or results.get("torch_replay") is None:
            raise BenchmarkAbort("correctness check requires both b12x and torch replays")
        results["b12x_replay"]()
        results["torch_replay"]()
        torch.cuda.synchronize()
        check_outputs(
            results["b12x_out"][:, :, 0],
            results["torch_out"],
            label=REFERENCE_LABEL,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--group-width", type=int, default=512)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="Bytes to touch when evicting L2; 0 uses 2x the reported L2 size.",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    parser.set_defaults(check=True)
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Run correctness checks and fail hard when cosine similarity falls below the threshold (default: enabled).",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Disable correctness checks before timing.",
    )
    args = parser.parse_args()

    require_sm120()
    torch.empty(1, device="cuda")
    l2_flush = make_l2_flush_fn(bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)

    print(f"WO projection: b12x native MXFP8 two-GEMM vs {REFERENCE_LABEL}")
    print("Timing mode: CUDA graph replay")
    print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    if args.check:
        print(f"Correctness check: on (cos >= {COSINE_THRESHOLD:.6f})")
    else:
        print("Correctness check: off")
    print(
        "Shape: "
        f"groups={args.groups}, group_width={args.group_width}, "
        f"rank={args.rank}, hidden={args.hidden}"
    )
    print("b12x note: activation quant/scale packing is included in the graph replay path.")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print()

    all_results = []
    for tokens in args.token_counts:
        try:
            results = bench_one(
                tokens,
                groups=args.groups,
                group_width=args.group_width,
                rank=args.rank,
                hidden=args.hidden,
                warmup=args.warmup,
                iters=args.iters,
                check=args.check,
                l2_flush=l2_flush,
                seed=args.seed + tokens,
            )
        except BenchmarkAbort as exc:
            print(f"ERROR: benchmark aborted for tokens={tokens}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        b12x_times = results.get("b12x")
        torch_times = results.get(REFERENCE_LABEL)
        b12x_med = statistics.median(b12x_times) * 1000.0 if b12x_times else None
        torch_med = statistics.median(torch_times) * 1000.0 if torch_times else None

        parts = [f"  tokens={tokens:<4}"]
        if b12x_times is not None:
            parts.append(f"b12x={fmt_us(b12x_times)}")
        if torch_times is not None:
            parts.append(f"torch={fmt_us(torch_times)}")
        if b12x_med is not None and torch_med is not None:
            parts.append(f"b12x/torch={b12x_med / torch_med:.2f}x")
        print("  ".join(parts) + "  (graph replay)")
        all_results.append((tokens, b12x_med, torch_med))

    print(f"\n{'=' * 75}")
    print(f"  SUMMARY: b12x / {REFERENCE_LABEL} (CUDA graph replay, lower = b12x faster)")
    print(f"{'=' * 75}")
    print(f"  {'tokens':<10}  {'ratio':>10}")
    print("  " + "-" * 24)

    ratios = []
    for tokens, b12x_med, torch_med in all_results:
        if b12x_med is not None and torch_med is not None:
            ratio = b12x_med / torch_med
            ratios.append(ratio)
            print(f"  {tokens:<10}  {ratio:>9.2f}x")
        else:
            print(f"  {tokens:<10}  {'n/a':>10}")

    if ratios:
        geo = 1.0
        for ratio in ratios:
            geo *= ratio
        geo **= 1.0 / len(ratios)
        print(f"\n  geo mean: {geo:.2f}x")


if __name__ == "__main__":
    main()
