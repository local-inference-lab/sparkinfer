#!/usr/bin/env python3
"""Benchmark: b12x dense_gemm vs FlashInfer-CUTLASS with CUDA graph replay.

Compares block-scaled FP4 and MXFP8 dense GEMM performance on the Nemotron 3
Super shared-expert down-projection shape `[M, 5376] x [5376, 4096]` across
small decode-style batch sizes.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys
from typing import Callable, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from b12x.cute.fp4 import quantize_grouped_nvfp4_torch
from b12x.cute.utils import (
    convert_sf_from_mma_layout,
    convert_sf_to_mma_layout,
    get_hardware_info,
)
from b12x.gemm.dense import dense_gemm

from flashinfer import mxfp8_quantize
from flashinfer.gemm import mm_fp4, mm_mxfp8
from flashinfer.tllm_enums import SfLayout


# Nemotron 3 Super shared expert down projection from the released NVFP4
# checkpoint:
#   down: [M, 5376] x [5376, 4096]
NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE = 5376
NEMOTRON_HIDDEN_SIZE = 4096

GEMM_SPECS = [
    # (name, K, N, note)
    (
        "Nemotron shared expert down",
        NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE,
        NEMOTRON_HIDDEN_SIZE,
        "NVIDIA Nemotron 3 Super shared_experts.down_proj",
    ),
]

FP4_BATCH_SIZES = [2, 4, 8]
FP8_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
REFERENCE_BACKEND = "cutlass"
FP4_REFERENCE_LABEL = "FlashInfer CUTLASS FP4"
FP8_REFERENCE_LABEL = "FlashInfer CUTLASS MXFP8"
COSINE_THRESHOLD = 0.999999
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


def make_l2_flush_fn(
    *,
    enabled: bool,
    bytes_hint: int = 0,
) -> Callable[[], None] | None:
    if not enabled:
        return None
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


def bench_events(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    l2_flush: Callable[[], None] | None = None,
) -> List[float]:
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if l2_flush is not None:
            l2_flush()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def fmt_us(times_ms: List[float]) -> str:
    med = statistics.median(times_ms) * 1000
    mn = min(times_ms) * 1000
    return f"{med:7.1f} us (min {mn:.1f})"


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return F.cosine_similarity(a_f, b_f, dim=0).item()


def check_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    label: str,
    cosine_threshold: float,
) -> None:
    cand_finite = bool(torch.isfinite(candidate).all().item())
    ref_finite = bool(torch.isfinite(reference).all().item())
    if not cand_finite or not ref_finite:
        raise CorrectnessError(
            f"non-finite output detected during correctness check vs {label}: "
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
    if not math.isfinite(cos):
        raise CorrectnessError(
            f"cosine similarity vs {label} is non-finite: "
            f"max_abs={max_abs:.8f}, rmse={rmse:.8f}, cos={cos}"
        )
    if cos < cosine_threshold:
        raise CorrectnessError(
            f"cosine similarity vs {label} fell below threshold "
            f"{cosine_threshold:.6f}: got {cos:.10f}"
        )


def capture_graph_replay(fn: Callable[[], None]) -> Callable[[], None]:
    # Warm eager launch state before capture so compile/cache work does not leak
    # into the replay measurement.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    return replay


def make_quantized_operand(M: int, K: int):
    source = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16) / 4
    row_counts = torch.full((1,), M, dtype=torch.int32, device="cuda")
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32, device="cuda",
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return packed, scales, global_scale


def quantize_mxfp8_source(source: torch.Tensor):
    M, K = source.shape
    quantized, scale = mxfp8_quantize(
        input=source,
        is_sf_swizzled_layout=True,
        alignment=32,
        sf_swizzle_layout=SfLayout.layout_128x4,
    )
    scale_mma = convert_sf_to_mma_layout(
        scale.view(torch.float8_e8m0fnu),
        m=M,
        k=K,
        num_groups=1,
        sf_vec_size=32,
    )
    return quantized.contiguous(), scale.contiguous(), scale_mma


def make_mxfp8_operand(M: int, K: int):
    source = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 4).contiguous()
    return (*quantize_mxfp8_source(source), source)


def bench_one_fp4(
    M: int,
    N: int,
    K: int,
    *,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None] | None,
):
    """Benchmark one (M,N,K) problem with CUDA graph replay timing."""
    torch.manual_seed(42)
    a_packed, a_sf, a_gs = make_quantized_operand(M, K)
    b_packed, b_sf, b_gs = make_quantized_operand(N, K)
    alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)

    a_fp4_2d = a_packed[:, :, 0].contiguous()
    b_fp4_2d = b_packed[:, :, 0].contiguous()
    a_sf_2d = convert_sf_from_mma_layout(a_sf, m=M, k=K, num_groups=1)
    b_sf_2d = convert_sf_from_mma_layout(b_sf, m=N, k=K, num_groups=1)

    results = {}

    # b12x FP4.
    try:
        b12x_out = torch.empty((M, N, 1), device="cuda", dtype=torch.bfloat16)

        def b12x_launch():
            dense_gemm(
                (a_packed, a_sf), (b_packed, b_sf), alpha=alpha,
                ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16", sf_vec_size=16, out=b12x_out,
            )
        b12x_replay = capture_graph_replay(b12x_launch)
        results["b12x_replay"] = b12x_replay
        results["b12x_out"] = b12x_out
        results["b12x"] = bench_events(
            b12x_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results["b12x"] = None
        print(f"      b12x FAILED: {exc}")

    # FlashInfer CUTLASS FP4 reference.
    try:
        ref_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

        def cutlass_launch():
            mm_fp4(
                a_fp4_2d, b_fp4_2d.T, a_sf_2d, b_sf_2d.T,
                alpha, torch.bfloat16, ref_out, block_size=16,
                use_8x4_sf_layout=False, backend=REFERENCE_BACKEND, use_nvfp4=True,
            )
        ref_replay = capture_graph_replay(cutlass_launch)
        results["ref_replay"] = ref_replay
        results["ref_out"] = ref_out
        results[FP4_REFERENCE_LABEL] = bench_events(
            ref_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results[FP4_REFERENCE_LABEL] = None
        print(f"      {FP4_REFERENCE_LABEL} FAILED: {exc}")

    if check:
        if results.get("b12x_replay") is None or results.get("ref_replay") is None:
            raise BenchmarkAbort(
                "correctness check requires both b12x and reference replays"
            )
        results["b12x_replay"]()
        results["ref_replay"]()
        torch.cuda.synchronize()
        check_outputs(
            results["b12x_out"][:, :, 0],
            results["ref_out"],
            label=FP4_REFERENCE_LABEL,
            cosine_threshold=COSINE_THRESHOLD,
        )

    return results


def bench_one_fp8(
    M: int,
    N: int,
    K: int,
    *,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None] | None,
):
    """Benchmark one MXFP8 (M,N,K) problem with CUDA graph replay timing."""
    torch.manual_seed(42)
    a_quantized, a_scale, a_scale_mma, a_source = make_mxfp8_operand(M, K)
    b_quantized, b_scale, b_scale_mma, _ = make_mxfp8_operand(N, K)

    results = {}

    # b12x MXFP8.
    try:
        b12x_out = torch.empty((M, N, 1), device="cuda", dtype=torch.bfloat16)

        def b12x_launch():
            dense_gemm(
                (a_quantized.view(M, K, 1), a_scale_mma),
                (b_quantized.view(N, K, 1), b_scale_mma),
                ab_dtype="float8_e4m3fn",
                sf_dtype="float8_e8m0fnu",
                c_dtype="bfloat16",
                sf_vec_size=32,
                out=b12x_out,
            )

        b12x_replay = capture_graph_replay(b12x_launch)
        results["b12x_replay"] = b12x_replay
        results["b12x_out"] = b12x_out
        results["b12x"] = bench_events(
            b12x_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results["b12x"] = None
        print(f"      b12x FAILED: {exc}")

    # FlashInfer CUTLASS MXFP8 reference. FlashInfer currently rejects the
    # direct M=1 SM120 MXFP8 case, so keep it out of timed comparisons for that
    # shape. A padded M=2 launch still gives a graph-safe GPU correctness
    # reference for the first row without benchmarking a fake M=1 FlashInfer
    # path.
    if M == 1:
        try:
            ref_out_padded = torch.empty((2, N), device="cuda", dtype=torch.bfloat16)
            a_source_padded = torch.cat([a_source, torch.zeros_like(a_source)], dim=0)
            a_quantized_padded, a_scale_padded, _ = quantize_mxfp8_source(
                a_source_padded.contiguous()
            )

            def cutlass_launch():
                mm_mxfp8(
                    a_quantized_padded,
                    b_quantized.t(),
                    a_scale_padded,
                    b_scale,
                    out=ref_out_padded,
                    out_dtype=torch.bfloat16,
                    backend=REFERENCE_BACKEND,
                )

            ref_replay = capture_graph_replay(cutlass_launch)
            results["ref_replay"] = ref_replay
            results["ref_out"] = ref_out_padded[:1]
            results[FP8_REFERENCE_LABEL] = None
            print(f"      {FP8_REFERENCE_LABEL} skipped for direct M=1")
        except Exception as exc:
            results[FP8_REFERENCE_LABEL] = None
            print(f"      padded {FP8_REFERENCE_LABEL} correctness reference FAILED: {exc}")
    else:
        try:
            ref_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

            def cutlass_launch():
                mm_mxfp8(
                    a_quantized,
                    b_quantized.t(),
                    a_scale,
                    b_scale,
                    out=ref_out,
                    out_dtype=torch.bfloat16,
                    backend=REFERENCE_BACKEND,
                )

            ref_replay = capture_graph_replay(cutlass_launch)
            results["ref_replay"] = ref_replay
            results["ref_out"] = ref_out
            results[FP8_REFERENCE_LABEL] = bench_events(
                ref_replay,
                warmup=warmup,
                iters=iters,
                l2_flush=l2_flush,
            )
        except Exception as exc:
            results[FP8_REFERENCE_LABEL] = None
            print(f"      {FP8_REFERENCE_LABEL} FAILED: {exc}")

    if check:
        if results.get("b12x_replay") is None or results.get("ref_replay") is None:
            raise BenchmarkAbort(
                "correctness check requires both b12x and reference replays"
            )
        results["b12x_replay"]()
        results["ref_replay"]()
        torch.cuda.synchronize()
        check_outputs(
            results["b12x_out"][:, :, 0],
            results["ref_out"],
            label=FP8_REFERENCE_LABEL,
            cosine_threshold=COSINE_THRESHOLD,
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "M values to benchmark. Defaults to 2/4/8 for FP4 and "
            "1/2/4/8/16/32/64/128/256 for FP8."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("fp4", "fp8", "all"),
        default="fp4",
        help="Benchmark the existing NVFP4 path, the MXFP8 path, or both.",
    )
    parser.add_argument(
        "--flush-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evict GPU L2 before each warmup and timed launch (default: enabled).",
    )
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="Bytes to touch when evicting L2; 0 uses 2x the reported L2 size.",
    )
    parser.set_defaults(check=True)
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Run correctness checks against FlashInfer CUTLASS and fail hard when cosine similarity falls below the threshold (default: enabled).",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Disable correctness checks before timing.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.empty(1, device="cuda")
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0

    if args.dtype == "all":
        benchmark_modes = (
            ("fp4", FP4_REFERENCE_LABEL, bench_one_fp4),
            ("fp8", FP8_REFERENCE_LABEL, bench_one_fp8),
        )
    elif args.dtype == "fp4":
        benchmark_modes = (("fp4", FP4_REFERENCE_LABEL, bench_one_fp4),)
    else:
        benchmark_modes = (("fp8", FP8_REFERENCE_LABEL, bench_one_fp8),)
    if args.batch_sizes is not None:
        batch_sizes = args.batch_sizes
    elif args.dtype == "fp4":
        batch_sizes = FP4_BATCH_SIZES
    else:
        batch_sizes = FP8_BATCH_SIZES

    mode_desc = ", ".join(mode.upper() for mode, _, _ in benchmark_modes)
    print(f"Dense GEMM ({mode_desc}): b12x vs FlashInfer CUTLASS")
    print("NVIDIA Nemotron 3 Super shared-expert down-proj")
    print("Timing mode: CUDA graph replay")
    if args.flush_l2:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")
    if args.check:
        print(f"Correctness check: on (cos >= {COSINE_THRESHOLD:.6f})")
    else:
        print("Correctness check: off")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print(f"M values: {batch_sizes}")
    print()

    # Collect all results for summary
    all_results = []  # (mode, name, bs, M, N, K, b12x_med, ref_med)

    for mode, reference_label, bench_fn in benchmark_modes:
        print(f"{'=' * 75}")
        print(f"  {mode.upper()} dense GEMM vs {reference_label}")
        print(f"{'=' * 75}")

        for name, K, N, note in GEMM_SPECS:
            print(f"  {name}  K={K} N={N}  [{note}]")

            for bs in batch_sizes:
                M = bs
                try:
                    results = bench_fn(
                        M,
                        N,
                        K,
                        warmup=args.warmup,
                        iters=args.iters,
                        check=args.check,
                        l2_flush=l2_flush,
                    )
                except BenchmarkAbort as exc:
                    print(
                        f"ERROR: benchmark aborted for {mode} {name} "
                        f"(bs={bs}, M={M}, N={N}, K={K}): {exc}",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)

                b12x_med = (
                    statistics.median(results["b12x"]) * 1000
                    if results.get("b12x")
                    else None
                )
                ref_med = (
                    statistics.median(results[reference_label]) * 1000
                    if results.get(reference_label)
                    else None
                )

                parts = [f"  {mode:<3} bs={bs:<3} (M={M:>2})"]
                if b12x_med is not None:
                    parts.append(f"b12x={b12x_med:6.1f}")
                if ref_med is not None:
                    parts.append(f"FlashInfer={ref_med:6.1f}")

                ratios = []
                if b12x_med and ref_med:
                    r = b12x_med / ref_med
                    ratios.append(f"b12x/flashinfer-cutlass={r:.2f}x")

                print("  ".join(parts) + "  " + "  ".join(ratios) + "  (graph us)")

                all_results.append((mode, name, bs, M, N, K, b12x_med, ref_med))

            print()

        print()

    print(f"\n{'=' * 75}")
    print("  SUMMARY: b12x / FlashInfer CUTLASS (CUDA graph replay, lower = b12x faster)")
    print(f"{'=' * 75}")
    header = f"  {'MODE':<5} {'GEMM':<30}"
    for bs in batch_sizes:
        header += f"  M={bs:<5}"
    print(header)
    print("  " + "-" * 70)

    ref_ratios = []
    for mode, _, _ in benchmark_modes:
        for name, K, N, note in GEMM_SPECS:
            row = f"  {mode:<5} {name:<30}"
            for bs in batch_sizes:
                match = [r for r in all_results if r[0] == mode and r[1] == name and r[2] == bs]
                if match and match[0][6] and match[0][7]:
                    ratio = match[0][6] / match[0][7]
                    row += f"  {ratio:.2f}x "
                    ref_ratios.append(ratio)
                else:
                    row += f"  {'n/a':>6}"
            print(row)

    if ref_ratios:
        geo = 1.0
        for r in ref_ratios:
            geo *= r
        geo **= 1.0 / len(ref_ratios)
        print(f"\n  geo mean: {geo:.2f}x")


if __name__ == "__main__":
    main()
