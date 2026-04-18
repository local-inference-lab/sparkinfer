#!/usr/bin/env python3
"""Benchmark realistic dense FP4 paths against FlashInfer and sglang baselines.

Compares the two relevant b12x production-style two-step paths:

  1. ``flashinfer.fp4_quantize`` + direct ``b12x`` prequantized dense GEMM
  2. ``flashinfer.fp4_quantize`` + current sglang ``_b12x_fp4_gemm`` wrapper
  3. ``flashinfer.fp4_quantize`` + ``flashinfer.mm_fp4``

Both sides are timed with CUDA graph replay and optional L2 eviction before
every replay.
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

from benchmarks.common import compute_global_scale, make_l2_flush_fn, resolve_l2_flush_bytes
from b12x.gemm import dense_gemm_packed_fp4
from flashinfer import fp4_quantize, mm_fp4

try:
    from sglang.srt.layers.quantization.modelopt_quant import (
        _b12x_fp4_gemm as sglang_b12x_fp4_gemm,
    )
except Exception:
    sglang_b12x_fp4_gemm = None


NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE = 5376
NEMOTRON_HIDDEN_SIZE = 4096

GEMM_SPECS = [
    (
        "Nemotron shared expert down",
        NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE,
        NEMOTRON_HIDDEN_SIZE,
        "logical BF16/FP16 x FP4 -> BF16/FP16",
    ),
]

# Defaults focus on decode-ish sizes. Pass ``--batch-sizes 1 ...`` to include
# the true physical ``M=1`` singleton path.
BATCH_SIZES = [2, 4, 8, 16]
REFERENCE_BACKEND = "cutlass"
REFERENCE_LABEL = "FlashInfer fp4_quantize + CUTLASS mm_fp4"
CURRENT_B12X_LABEL = "SGLang fp4_quantize + b12x dense_gemm"
DIRECT_B12X_LABEL = "FlashInfer fp4_quantize + direct b12x dense_gemm"
COSINE_THRESHOLD = 0.9995


class BenchmarkAbort(RuntimeError):
    """Fatal benchmark failure that should stop the run without a summary."""


class CorrectnessError(BenchmarkAbort):
    """Raised when replay outputs fail the correctness gate."""


def parse_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype {name}")


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
    for idx in range(iters):
        if l2_flush is not None:
            l2_flush()
        starts[idx].record()
        fn()
        ends[idx].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


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


def capture_graph_replay(fn: Callable[[], torch.Tensor | None]) -> tuple[Callable[[], None], torch.Tensor | None]:
    captured_output = None
    for _ in range(3):
        captured_output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    return replay, captured_output


def make_activation_operand(
    m: int,
    k: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.randn((m, k), device="cuda", dtype=dtype) / 4
    quant_scale = compute_global_scale(source)
    input_scale_inv = (1.0 / quant_scale).reshape(1)
    return source, quant_scale, input_scale_inv


def make_weight_operand(
    n: int,
    k: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.randn((n, k), device="cuda", dtype=dtype) / 4
    quant_scale = compute_global_scale(source)
    packed, scales = fp4_quantize(source, quant_scale)
    return (packed, scales), quant_scale


def bench_one(
    m: int,
    n: int,
    k: int,
    *,
    act_dtype: torch.dtype,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None] | None,
    weight: tuple[torch.Tensor, torch.Tensor],
    weight_quant_scale: torch.Tensor,
):
    activation, input_quant_scale, input_scale_inv = make_activation_operand(
        m,
        k,
        dtype=act_dtype,
    )
    alpha = (1.0 / (input_quant_scale[0] * weight_quant_scale[0])).view(1)
    packed_b, sfb = weight
    out_dtype_name = "bfloat16" if act_dtype == torch.bfloat16 else "float16"

    results = {}

    try:
        ref_out = torch.empty((m, n), device="cuda", dtype=act_dtype)

        def flashinfer_launch() -> None:
            packed_a, sfa = fp4_quantize(activation, input_quant_scale)
            mm_fp4(
                packed_a,
                packed_b.T,
                sfa,
                sfb.T,
                alpha,
                act_dtype,
                ref_out,
                block_size=16,
                use_8x4_sf_layout=False,
                backend=REFERENCE_BACKEND,
            )

        ref_replay, _ = capture_graph_replay(flashinfer_launch)
        results["ref_replay"] = ref_replay
        results["ref_out"] = ref_out
        results[REFERENCE_LABEL] = bench_events(
            ref_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results[REFERENCE_LABEL] = None
        print(f"      {REFERENCE_LABEL} FAILED: {exc}")

    if sglang_b12x_fp4_gemm is not None:
        try:
            def current_b12x_launch() -> torch.Tensor:
                packed_a, sfa = fp4_quantize(activation, input_quant_scale)
                return sglang_b12x_fp4_gemm(
                    packed_a,
                    packed_b,
                    sfa,
                    sfb,
                    alpha,
                    act_dtype,
                )

            current_b12x_replay, current_b12x_out = capture_graph_replay(current_b12x_launch)
            results["current_b12x_replay"] = current_b12x_replay
            results["current_b12x_out"] = current_b12x_out
            results[CURRENT_B12X_LABEL] = bench_events(
                current_b12x_replay,
                warmup=warmup,
                iters=iters,
                l2_flush=l2_flush,
            )
        except Exception as exc:
            results[CURRENT_B12X_LABEL] = None
            print(f"      {CURRENT_B12X_LABEL} FAILED: {exc}")

    try:
        direct_b12x_out = torch.empty((m, n), device="cuda", dtype=act_dtype)

        def direct_b12x_launch() -> None:
            packed_a, sfa = fp4_quantize(activation, input_quant_scale)
            dense_gemm_packed_fp4(
                (packed_a, sfa),
                (packed_b, sfb),
                out=direct_b12x_out,
                alpha=alpha,
                sf_dtype="float8_e4m3fn",
                c_dtype=out_dtype_name,
                sf_vec_size=16,
            )

        direct_b12x_replay, _ = capture_graph_replay(direct_b12x_launch)
        results["direct_b12x_replay"] = direct_b12x_replay
        results["direct_b12x_out"] = direct_b12x_out
        results[DIRECT_B12X_LABEL] = bench_events(
            direct_b12x_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results[DIRECT_B12X_LABEL] = None
        print(f"      {DIRECT_B12X_LABEL} FAILED: {exc}")

    if check:
        if results.get("ref_replay") is None:
            raise BenchmarkAbort("correctness check requires the FlashInfer replay")
        results["ref_replay"]()
        torch.cuda.synchronize()
        if results.get("current_b12x_replay") is not None:
            results["current_b12x_replay"]()
            torch.cuda.synchronize()
            check_outputs(
                results["current_b12x_out"],
                results["ref_out"],
                label=REFERENCE_LABEL,
                cosine_threshold=COSINE_THRESHOLD,
            )
        if results.get("direct_b12x_replay") is not None:
            results["direct_b12x_replay"]()
            torch.cuda.synchronize()
            check_outputs(
                results["direct_b12x_out"],
                results["ref_out"],
                label=REFERENCE_LABEL,
                cosine_threshold=COSINE_THRESHOLD,
            )
            if results.get("current_b12x_replay") is not None:
                check_outputs(
                    results["direct_b12x_out"],
                    results["current_b12x_out"],
                    label=CURRENT_B12X_LABEL,
                    cosine_threshold=COSINE_THRESHOLD,
                )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Dense activation and output dtype.",
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
        help="Run correctness checks against FlashInfer and fail hard when cosine similarity falls below the threshold (default: enabled).",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Disable correctness checks before timing.",
    )
    args = parser.parse_args()

    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise RuntimeError(f"Requires sm_120 or sm_121, got sm_{major}{minor}")
    torch.empty(1, device="cuda")
    act_dtype = parse_dtype(args.dtype)
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0

    print(f"Dense FP4 GEMM Paths: direct/current b12x vs {REFERENCE_LABEL}")
    print("NVIDIA Nemotron 3 Super shared-expert down-proj")
    print("Timing mode: CUDA graph replay")
    print(f"Activation/output dtype: {args.dtype}")
    if args.flush_l2:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")
    if args.check:
        print(f"Correctness check: on (cos >= {COSINE_THRESHOLD:.6f})")
    else:
        print("Correctness check: off")
    if sglang_b12x_fp4_gemm is not None:
        print(f"Current b12x baseline: on ({CURRENT_B12X_LABEL})")
    else:
        print("Current b12x baseline: unavailable (sglang import failed)")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print()

    all_results = []

    for name, k, n, note in GEMM_SPECS:
        torch.manual_seed(1234)
        weight, weight_quant_scale = make_weight_operand(n, k, dtype=act_dtype)

        print(f"{'=' * 75}")
        print(f"  {name}  K={k} N={n}  [{note}]")
        print(f"{'=' * 75}")

        for bs in args.batch_sizes:
            m = bs
            torch.manual_seed(42 + bs)
            try:
                results = bench_one(
                    m,
                    n,
                    k,
                    act_dtype=act_dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                    check=args.check,
                    l2_flush=l2_flush,
                    weight=weight,
                    weight_quant_scale=weight_quant_scale,
                )
            except BenchmarkAbort as exc:
                print(
                    f"ERROR: benchmark aborted for {name} "
                    f"(bs={bs}, M={m}, N={n}, K={k}): {exc}",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            direct_b12x_med = (
                statistics.median(results[DIRECT_B12X_LABEL]) * 1000
                if results.get(DIRECT_B12X_LABEL)
                else None
            )
            current_b12x_med = (
                statistics.median(results[CURRENT_B12X_LABEL]) * 1000
                if results.get(CURRENT_B12X_LABEL)
                else None
            )
            ref_med = (
                statistics.median(results[REFERENCE_LABEL]) * 1000
                if results.get(REFERENCE_LABEL)
                else None
            )

            parts = [f"  bs={bs:<3} (M={m:>3})"]
            if direct_b12x_med is not None:
                parts.append(f"direct_b12x={direct_b12x_med:6.1f}")
            if current_b12x_med is not None:
                parts.append(f"sglang_b12x={current_b12x_med:6.1f}")
            if ref_med is not None:
                parts.append(f"flashinfer={ref_med:6.1f}")

            ratios = []
            if direct_b12x_med and current_b12x_med:
                ratios.append(
                    f"direct_b12x/sglang_b12x={direct_b12x_med / current_b12x_med:.2f}x"
                )
            if direct_b12x_med and ref_med:
                ratios.append(f"direct_b12x/flashinfer={direct_b12x_med / ref_med:.2f}x")

            print("  ".join(parts) + "  " + "  ".join(ratios) + "  (graph us)")
            all_results.append(
                (name, bs, m, n, k, direct_b12x_med, current_b12x_med, ref_med)
            )

        print()

    def print_summary_table(
        title: str,
        numerator_index: int,
        denominator_index: int,
        *,
        show_when_any: bool,
    ) -> None:
        if not show_when_any:
            return
        print(f"\n{'=' * 75}")
        print(f"  SUMMARY: {title} (CUDA graph replay, lower = numerator faster)")
        print(f"{'=' * 75}")
        header = f"  {'GEMM':<30}"
        for bs in args.batch_sizes:
            header += f"  M={bs:<5}"
        print(header)
        print("  " + "-" * 70)

        ratios = []
        for name, _, _, _ in GEMM_SPECS:
            row = f"  {name:<30}"
            for bs in args.batch_sizes:
                match = [result for result in all_results if result[0] == name and result[1] == bs]
                if match and match[0][numerator_index] and match[0][denominator_index]:
                    ratio = match[0][numerator_index] / match[0][denominator_index]
                    row += f"  {ratio:.2f}x "
                    ratios.append(ratio)
                else:
                    row += f"  {'n/a':>6}"
            print(row)

        if ratios:
            geo = 1.0
            for ratio in ratios:
                geo *= ratio
            geo **= 1.0 / len(ratios)
            print(f"\n  geo mean: {geo:.2f}x")

    have_direct_b12x = any(result[5] is not None for result in all_results)
    have_current_b12x = any(result[6] is not None for result in all_results)
    print_summary_table(
        f"{DIRECT_B12X_LABEL} / {REFERENCE_LABEL}",
        5,
        7,
        show_when_any=have_direct_b12x,
    )
    print_summary_table(
        f"{CURRENT_B12X_LABEL} / {REFERENCE_LABEL}",
        6,
        7,
        show_when_any=have_current_b12x,
    )
    print_summary_table(
        f"{DIRECT_B12X_LABEL} / {CURRENT_B12X_LABEL}",
        5,
        6,
        show_when_any=have_direct_b12x and have_current_b12x,
    )


if __name__ == "__main__":
    main()
