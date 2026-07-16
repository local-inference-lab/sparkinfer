#!/usr/bin/env python3
"""Benchmark b12x dense_gemm against reference backends with graph replay.

The FP4 track uses the Nemotron 3 Super shared-expert down projection. The
MXFP8 tracks use the per-rank dense-linear shapes from the cached DeepSeek V4
Flash DSpark checkpoint at TP=2, excluding routed experts. End-to-end MXFP8
includes activation quantization and compares only b12x with DeepGEMM; weight
quantization remains setup work.
"""

from __future__ import annotations

import argparse
import importlib
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
)
from b12x.gemm.block_fp8_linear import quantize_block_fp8_linear_input_mxfp8
from b12x.gemm.dense import dense_gemm
from b12x.gemm.wo_projection import empty_mxfp8_rows_for_dense_gemm
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes

from flashinfer import mxfp8_quantize
from flashinfer.gemm import mm_fp4, mm_mxfp8
from flashinfer.tllm_enums import SfLayout


# Nemotron 3 Super shared expert down projection from the released NVFP4
# checkpoint:
#   down: [M, 5376] x [5376, 4096]
NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE = 5376
NEMOTRON_HIDDEN_SIZE = 4096

FP4_GEMM_SPECS = [
    # (name, K, N, note)
    (
        "Nemotron shared expert down",
        NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE,
        NEMOTRON_HIDDEN_SIZE,
        "NVIDIA Nemotron 3 Super shared_experts.down_proj",
    ),
]

# DeepSeek-V4-Flash-DSpark, TP=2. The checkpoint stores wq_b as
# [32768, 1024]; vLLM's ColumnParallelLinear shards its output dimension to
# [16384, 1024] per rank. This is the representative generic FP8 dense linear.
# Routed experts are intentionally excluded because they execute through fused
# MoE, while WO is primarily covered by its specialized fused projection path.
FP8_GEMM_SPECS = [
    # (name, K, N, note)
    (
        "DSV4-DSpark TP2 q_b",
        1024,
        16384,
        "column-parallel half of checkpoint wq_b[32768,1024]",
    ),
]


def gemm_specs_for_mode(mode: str):
    return FP4_GEMM_SPECS if mode == "fp4" else FP8_GEMM_SPECS

FP4_BATCH_SIZES = [2, 4, 8]
FP8_BATCH_SIZES = [1, 2, 4, 8, 4096]
REFERENCE_BACKEND = "cutlass"
FP4_REFERENCE_LABEL = "FlashInfer CUTLASS FP4"
FP8_REFERENCE_LABEL = "FlashInfer CUTLASS MXFP8"
DEEPGEMM_E2E_REFERENCE_LABEL = "DeepGEMM fp8_gemm_nt e2e"
COSINE_THRESHOLD = 0.999999
DEEPGEMM_COSINE_THRESHOLD = 0.999


class BenchmarkAbort(RuntimeError):
    """Fatal benchmark failure that should stop the run without a summary."""


class CorrectnessError(BenchmarkAbort):
    """Raised when replay outputs fail the correctness gate."""


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
    return [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]


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
    print(f"    check vs {label}: max_abs={max_abs:.8f} rmse={rmse:.8f} cos={cos:.10f}")
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
        dtype=torch.float32,
        device="cuda",
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return packed, scales, global_scale


def quantize_mxfp8_source(source: torch.Tensor):
    M, K = source.shape
    quantized, scale = quantize_mxfp8_source_flashinfer(source)
    scale_mma = convert_sf_to_mma_layout(
        scale.view(torch.float8_e8m0fnu),
        m=M,
        k=K,
        num_groups=1,
        sf_vec_size=32,
    )
    return quantized.contiguous(), scale.contiguous(), scale_mma


def quantize_mxfp8_source_flashinfer(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp8_quantize(
        input=source,
        is_sf_swizzled_layout=True,
        alignment=32,
        sf_swizzle_layout=SfLayout.layout_128x4,
    )


def make_mxfp8_operand(M: int, K: int):
    source = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 4).contiguous()
    return (*quantize_mxfp8_source(source), source)


def load_deepgemm():
    try:
        deep_gemm = importlib.import_module("deep_gemm")
    except ImportError as exc:
        raise BenchmarkAbort(
            "fp8-e2e requires DeepGEMM with per_token_cast_to_fp8, "
            "per_block_cast_to_fp8, and fp8_gemm_nt"
        ) from exc
    required = (
        "per_token_cast_to_fp8",
        "per_block_cast_to_fp8",
        "fp8_gemm_nt",
    )
    missing = [name for name in required if not hasattr(deep_gemm, name)]
    if missing:
        raise BenchmarkAbort(
            f"installed DeepGEMM is missing required APIs: {', '.join(missing)}"
        )
    return deep_gemm


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
                (a_packed, a_sf),
                (b_packed, b_sf),
                alpha=alpha,
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16",
                sf_vec_size=16,
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

    # FlashInfer CUTLASS FP4 reference.
    try:
        ref_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

        def cutlass_launch():
            mm_fp4(
                a_fp4_2d,
                b_fp4_2d.T,
                a_sf_2d,
                b_sf_2d.T,
                alpha,
                torch.bfloat16,
                ref_out,
                block_size=16,
                use_8x4_sf_layout=False,
                backend=REFERENCE_BACKEND,
                use_nvfp4=True,
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
    include_input_quant: bool = False,
):
    """Benchmark one MXFP8 (M,N,K) problem with CUDA graph replay timing.

    When ``include_input_quant`` is true, each replay starts from the BF16 A
    operand. The b12x launch uses caller-owned MXFP8 storage and the production
    ``quantize_block_fp8_linear_input_mxfp8(..., out=...)`` path, which launches
    ``_quantize_dense_tk_to_tk_kernel`` before ``dense_gemm``. B remains a
    prequantized model weight for both backends.
    """
    torch.manual_seed(42)
    a_quantized, a_scale, a_scale_mma, a_source = make_mxfp8_operand(M, K)
    b_quantized, b_scale, b_scale_mma, b_source = make_mxfp8_operand(N, K)

    results = {}

    # b12x MXFP8. Keep quantizer output allocation outside capture so the e2e
    # replay matches an allocation-stable serving path.
    try:
        b12x_out = torch.empty((M, N, 1), device="cuda", dtype=torch.bfloat16)
        a_quantized_b12x = None
        if include_input_quant:
            a_quantized_b12x = empty_mxfp8_rows_for_dense_gemm(
                M,
                K,
                device=a_source.device,
            )

        def b12x_launch():
            if a_quantized_b12x is not None:
                quantize_block_fp8_linear_input_mxfp8(
                    a_source,
                    out=a_quantized_b12x,
                )
                a_values = a_quantized_b12x.values
                a_scale_for_gemm = a_quantized_b12x.scale_mma
            else:
                a_values = a_quantized
                a_scale_for_gemm = a_scale_mma
            dense_gemm(
                (a_values.view(M, K, 1), a_scale_for_gemm),
                (b_quantized.view(N, K, 1), b_scale_mma),
                ab_dtype="float8_e4m3fn",
                sf_dtype="float8_e8m0fnu",
                c_dtype="bfloat16",
                sf_vec_size=32,
                out=b12x_out,
                # Match the production scaled-mm route: the graph shape is the
                # regime hint, so 1024 stays on BK128 while 2048+ may select the
                # separately keyed BK64 specialization.
                expected_m=M,
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

    # FlashInfer is a reference only for the prequantized MXFP8 diagnostic.
    # The end-to-end mode intentionally compares only b12x with DeepGEMM.
    if not include_input_quant:
        reference_label = FP8_REFERENCE_LABEL
        # FlashInfer currently rejects direct M=1 SM120 MXFP8. Use padded M=2
        # only as a correctness reference for the first row, without timing it.
        if M == 1:
            try:
                ref_out_padded = torch.empty(
                    (2, N), device="cuda", dtype=torch.bfloat16
                )
                a_source_padded = torch.cat(
                    [a_source, torch.zeros_like(a_source)], dim=0
                )
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
                results[reference_label] = None
                print(f"      {reference_label} skipped for direct M=1")
            except Exception as exc:
                results[reference_label] = None
                print(
                    f"      padded {reference_label} correctness reference "
                    f"FAILED: {exc}"
                )
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
                results[reference_label] = bench_events(
                    ref_replay,
                    warmup=warmup,
                    iters=iters,
                    l2_flush=l2_flush,
                )
            except Exception as exc:
                results[reference_label] = None
                print(f"      {reference_label} FAILED: {exc}")

        if check:
            if (
                results.get("b12x_replay") is None
                or results.get("ref_replay") is None
            ):
                raise BenchmarkAbort(
                    "correctness check requires both b12x and reference replays"
                )
            results["b12x_replay"]()
            results["ref_replay"]()
            torch.cuda.synchronize()
            check_outputs(
                results["b12x_out"][:, :, 0],
                results["ref_out"],
                label=reference_label,
                cosine_threshold=COSINE_THRESHOLD,
            )
    elif check:
        if results.get("b12x_replay") is None:
            raise BenchmarkAbort("correctness check requires the b12x e2e replay")
        results["b12x_replay"]()
        b12x_oracle = (a_source.float() @ b_source.float().T).to(torch.bfloat16)
        torch.cuda.synchronize()
        check_outputs(
            results["b12x_out"][:, :, 0],
            b12x_oracle,
            label="BF16 source matmul oracle for b12x",
            cosine_threshold=DEEPGEMM_COSINE_THRESHOLD,
        )

    if include_input_quant:
        # DeepGEMM's dense-linear contract dynamically quantizes A per token and
        # consumes a model-load-time, per-128x128-block quantized B. Keep both
        # native quantization granularities rather than adapting b12x operands.
        try:
            deep_gemm = load_deepgemm()
            b_deepgemm = deep_gemm.per_block_cast_to_fp8(b_source, True)
            deepgemm_out = torch.empty(
                (M, N),
                device=a_source.device,
                dtype=torch.bfloat16,
            )

            def deepgemm_launch():
                a_deepgemm = deep_gemm.per_token_cast_to_fp8(a_source, True)
                deep_gemm.fp8_gemm_nt(a_deepgemm, b_deepgemm, deepgemm_out)

            deepgemm_replay = capture_graph_replay(deepgemm_launch)
            results["deepgemm_replay"] = deepgemm_replay
            results["deepgemm_out"] = deepgemm_out
            results[DEEPGEMM_E2E_REFERENCE_LABEL] = bench_events(
                deepgemm_replay,
                warmup=warmup,
                iters=iters,
                l2_flush=l2_flush,
            )
        except Exception as exc:
            results[DEEPGEMM_E2E_REFERENCE_LABEL] = None
            print(f"      {DEEPGEMM_E2E_REFERENCE_LABEL} FAILED: {exc}")

        if check:
            if results.get("deepgemm_replay") is None:
                raise BenchmarkAbort(
                    "correctness check requires the DeepGEMM e2e replay"
                )
            results["deepgemm_replay"]()
            deepgemm_oracle = (a_source.float() @ b_source.float().T).to(torch.bfloat16)
            torch.cuda.synchronize()
            check_outputs(
                results["deepgemm_out"],
                deepgemm_oracle,
                label="BF16 source matmul oracle for DeepGEMM",
                cosine_threshold=DEEPGEMM_COSINE_THRESHOLD,
            )

    return results


def bench_one_fp8_e2e(
    M: int,
    N: int,
    K: int,
    *,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None] | None,
):
    return bench_one_fp8(
        M,
        N,
        K,
        warmup=warmup,
        iters=iters,
        check=check,
        l2_flush=l2_flush,
        include_input_quant=True,
    )


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
            "1/2/4/8/4096 for FP8."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("fp4", "fp8", "fp8-e2e", "all"),
        default="fp4",
        help=(
            "Benchmark NVFP4, prequantized MXFP8, end-to-end MXFP8 including "
            "BF16 input quantization, or all three."
        ),
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
        help="Run correctness checks against reference backends and fail hard when cosine similarity falls below the backend threshold (default: enabled).",
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
            ("fp8-e2e", None, bench_one_fp8_e2e),
        )
    elif args.dtype == "fp4":
        benchmark_modes = (("fp4", FP4_REFERENCE_LABEL, bench_one_fp4),)
    elif args.dtype == "fp8":
        benchmark_modes = (("fp8", FP8_REFERENCE_LABEL, bench_one_fp8),)
    else:
        benchmark_modes = (("fp8-e2e", None, bench_one_fp8_e2e),)
    if args.batch_sizes is not None:
        batch_sizes = args.batch_sizes
    elif args.dtype == "fp4":
        batch_sizes = FP4_BATCH_SIZES
    else:
        batch_sizes = FP8_BATCH_SIZES

    mode_desc = ", ".join(mode.upper() for mode, _, _ in benchmark_modes)
    print(f"Dense GEMM ({mode_desc}): b12x vs reference backends")
    if args.dtype == "fp4":
        print("NVIDIA Nemotron 3 Super shared-expert down-proj")
    elif args.dtype in ("fp8", "fp8-e2e"):
        print("DeepSeek V4 Flash DSpark TP=2 q_b projection")
    else:
        print("FP4: Nemotron shared down; MXFP8: DSV4-DSpark TP=2 q_b")
    print("Timing mode: CUDA graph replay")
    if args.flush_l2:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")
    if args.check:
        if args.dtype != "fp8-e2e":
            print(
                "FlashInfer correctness check: on "
                f"(cos >= {COSINE_THRESHOLD:.6f})"
            )
        if args.dtype in ("fp8-e2e", "all"):
            print(
                "E2E BF16-oracle checks for b12x and DeepGEMM: on "
                f"(cos >= {DEEPGEMM_COSINE_THRESHOLD:.6f} vs BF16 oracle)"
            )
    else:
        print("Correctness check: off")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print(f"M values: {batch_sizes}")
    print()

    # Collect all results for summary.
    # (mode, name, bs, M, N, K, b12x_med, flashinfer_med, deepgemm_med)
    all_results = []

    for mode, reference_label, bench_fn in benchmark_modes:
        print(f"{'=' * 75}")
        section_references = (
            DEEPGEMM_E2E_REFERENCE_LABEL
            if mode == "fp8-e2e"
            else reference_label
        )
        print(f"  {mode.upper()} dense GEMM vs {section_references}")
        print(f"{'=' * 75}")

        for name, K, N, note in gemm_specs_for_mode(mode):
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
                    raise SystemExit(1) from None

                b12x_med = (
                    statistics.median(results["b12x"]) * 1000
                    if results.get("b12x")
                    else None
                )
                ref_med = (
                    statistics.median(results[reference_label]) * 1000
                    if reference_label is not None and results.get(reference_label)
                    else None
                )
                deepgemm_med = (
                    statistics.median(results[DEEPGEMM_E2E_REFERENCE_LABEL]) * 1000
                    if results.get(DEEPGEMM_E2E_REFERENCE_LABEL)
                    else None
                )

                parts = [f"  {mode:<8} bs={bs:<3} (M={M:>3})"]
                if b12x_med is not None:
                    parts.append(f"b12x={b12x_med:6.1f}")
                if ref_med is not None:
                    parts.append(f"FlashInfer={ref_med:6.1f}")
                if deepgemm_med is not None:
                    parts.append(f"DeepGEMM={deepgemm_med:6.1f}")

                ratios = []
                if b12x_med and ref_med:
                    r = b12x_med / ref_med
                    ratios.append(f"b12x/flashinfer-cutlass={r:.2f}x")
                if b12x_med and deepgemm_med:
                    r = b12x_med / deepgemm_med
                    ratios.append(f"b12x/deepgemm={r:.2f}x")

                print("  ".join(parts) + "  " + "  ".join(ratios) + "  (graph us)")

                all_results.append(
                    (
                        mode,
                        name,
                        bs,
                        M,
                        N,
                        K,
                        b12x_med,
                        ref_med,
                        deepgemm_med,
                    )
                )

            print()

        print()

    print(f"\n{'=' * 75}")
    print("  SUMMARY: b12x/reference (CUDA graph replay, lower = b12x faster)")
    print(f"{'=' * 75}")
    header = f"  {'MODE':<9} {'GEMM':<30}"
    for bs in batch_sizes:
        header += f"  M={bs:<5}"
    print(header)
    print("  " + "-" * 70)

    summary_references = (
        ("FlashInfer CUTLASS", 7, None),
        ("DeepGEMM fp8_gemm_nt", 8, "fp8-e2e"),
    )
    for summary_label, result_idx, required_mode in summary_references:
        if required_mode is not None and not any(
            mode == required_mode for mode, _, _ in benchmark_modes
        ):
            continue
        if summary_label == "FlashInfer CUTLASS" and not any(
            mode != "fp8-e2e" for mode, _, _ in benchmark_modes
        ):
            continue
        ref_ratios = []
        print(f"\n  vs {summary_label}")
        for mode, _, _ in benchmark_modes:
            if required_mode is not None and mode != required_mode:
                continue
            if summary_label == "FlashInfer CUTLASS" and mode == "fp8-e2e":
                continue
            for name, _K, _N, _note in gemm_specs_for_mode(mode):
                row = f"  {mode:<9} {name:<30}"
                for bs in batch_sizes:
                    match = [
                        r
                        for r in all_results
                        if r[0] == mode and r[1] == name and r[2] == bs
                    ]
                    if match and match[0][6] and match[0][result_idx]:
                        ratio = match[0][6] / match[0][result_idx]
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
            print(
                f"  geo mean: {geo:.2f}x over {len(ref_ratios)} ratios "
                "(minimize)"
            )


if __name__ == "__main__":
    main()
