#!/usr/bin/env python3
"""Probe dense FP4 tile/load variants for small-N experiments.

This is intentionally an offline probe rather than runtime autotuning. It
sweeps narrow tile-N + swap_ab variants and the cp.async load specialization,
then reports graph-replay timings and exactness vs FlashInfer cuDNN FP4.

Run:
  CUTE_DSL_ARCH=sm_120a /home/luke/projects/vllm-other/.venv/bin/python \
    benchmarks/probe_dense_fp4_tile_load_sweep.py
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cutlass
import torch

from benchmarks.benchmark_dense_gemm import (
    bench_events,
    capture_graph_replay,
    cosine_similarity,
    make_quantized_operand,
)
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from sparkinfer._lib.utils import convert_sf_from_mma_layout
from sparkinfer._lib.dense_gemm import DenseGemmKernel, dense_gemm
from flashinfer.gemm import mm_fp4


@dataclass(frozen=True)
class Candidate:
    tile: tuple[int, int]
    load_path: str
    swap_ab: bool

    def label(self) -> str:
        swap = "swap" if self.swap_ab else "normal"
        return f"{self.tile}/{self.load_path}/{swap}"


@dataclass(frozen=True)
class ShapeCase:
    name: str
    n: int
    k: int
    note: str

    def label(self) -> str:
        return f"{self.name}: N={self.n} K={self.k}"


DEFAULT_CANDIDATES = (
    Candidate((128, 128), "tma", False),
    Candidate((128, 128), "cpasync", False),
    Candidate((128, 64), "tma", False),
    Candidate((128, 64), "cpasync", False),
    Candidate((64, 128), "tma", False),
    Candidate((64, 128), "cpasync", False),
    Candidate((64, 64), "tma", False),
    Candidate((64, 64), "cpasync", False),
    Candidate((128, 32), "tma", True),
    Candidate((128, 32), "cpasync", True),
    Candidate((128, 16), "tma", True),
    Candidate((128, 16), "cpasync", True),
    Candidate((64, 32), "tma", True),
    Candidate((64, 32), "cpasync", True),
    Candidate((64, 16), "tma", True),
    Candidate((64, 16), "cpasync", True),
)


COMMON_SHAPES = (
    ShapeCase(
        "nemotron_shared_down_full",
        4096,
        5376,
        "benchmark_dense_gemm shared expert down",
    ),
    ShapeCase(
        "nemotron_shared_down_n2048",
        2048,
        5376,
        "shared-down output shard proxy",
    ),
    ShapeCase(
        "nemotron_shared_down_n1024",
        1024,
        5376,
        "shared-down output shard proxy",
    ),
    ShapeCase(
        "nemotron_shared_down_n512",
        512,
        5376,
        "shared-down output shard proxy",
    ),
    ShapeCase(
        "qwen35_moe_fc1_fused",
        2048,
        4096,
        "benchmark_sparse_moe_api K=4096 I_tp=1024, silu gate/up",
    ),
    ShapeCase(
        "qwen35_moe_fc2_down",
        4096,
        1024,
        "benchmark_sparse_moe_api K=4096 I_tp=1024, down",
    ),
    ShapeCase(
        "nemotron_backbone_relu2_fc1",
        2688,
        1024,
        "tp_moe exact relu2 bs1 plan, up",
    ),
    ShapeCase(
        "nemotron_backbone_relu2_fc2",
        1024,
        2688,
        "tp_moe exact relu2 bs1 plan, down",
    ),
    ShapeCase(
        "nano35_relu2_fc1",
        1856,
        2688,
        "benchmark_moe nano35 shape profile, up",
    ),
    ShapeCase(
        "nano35_relu2_fc2_unaligned_k",
        2688,
        1856,
        "benchmark_moe nano35 shape profile, down; K is not tile_k aligned",
    ),
    ShapeCase(
        "dsv4f_silu_fc1_fused",
        2048,
        6144,
        "benchmark_moe dsv4f shape profile, silu gate/up",
    ),
    ShapeCase(
        "dsv4f_silu_fc2_down",
        6144,
        1024,
        "benchmark_moe dsv4f shape profile, down",
    ),
    ShapeCase(
        "deepseek_qkv_a_down",
        1536,
        4096,
        "benchmark_dense_fp8_vs_deepgemm projection shape",
    ),
    ShapeCase(
        "deepseek_q_b_up",
        16384,
        1024,
        "benchmark_dense_fp8_vs_deepgemm projection shape",
    ),
    ShapeCase(
        "deepseek_wo_a",
        1024,
        4096,
        "benchmark_dense_fp8_vs_deepgemm projection shape",
    ),
    ShapeCase(
        "deepseek_wo_b",
        4096,
        4096,
        "benchmark_dense_fp8_vs_deepgemm projection shape",
    ),
)


def _supported(candidate: Candidate, *, n: int, k: int) -> bool:
    return DenseGemmKernel.can_implement(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.BFloat16,
        candidate.tile,
        (1, 1),
        n,
        k,
        1,
        "k",
        "k",
        "n",
        load_path=candidate.load_path,
        swap_ab=candidate.swap_ab,
    )


def _make_reference(
    a_packed: torch.Tensor,
    a_sf: torch.Tensor,
    b_packed: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    *,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    return mm_fp4(
        a_packed[:, :, 0].contiguous(),
        b_packed[:, :, 0].contiguous().T,
        convert_sf_from_mma_layout(a_sf, m=m, k=k, num_groups=1),
        convert_sf_from_mma_layout(b_sf, m=n, k=k, num_groups=1).T,
        alpha,
        torch.bfloat16,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cudnn",
        use_nvfp4=True,
    )


def _parse_m_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--m-list must contain at least one value")
    return values


def _shape_cases(args: argparse.Namespace) -> list[ShapeCase]:
    if args.shape_set == "single":
        return [ShapeCase("single", args.n, args.k, "CLI shape")]
    return list(COMMON_SHAPES)


def _shape_supported_by_any(shape: ShapeCase) -> bool:
    return any(_supported(candidate, n=shape.n, k=shape.k) for candidate in DEFAULT_CANDIDATES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=5376)
    parser.add_argument("--shape-set", choices=("single", "common"), default="single")
    parser.add_argument("--list-common-shapes", action="store_true")
    parser.add_argument(
        "--m-list",
        type=_parse_m_list,
        default=_parse_m_list("1,2,4,8,16,32,64,128"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-l2-flush", action="store_true")
    args = parser.parse_args()

    if args.list_common_shapes:
        for shape in COMMON_SHAPES:
            print(f"{shape.label()}  # {shape.note}")
        return

    l2_flush = make_l2_flush_fn(enabled=not args.no_l2_flush, bytes_hint=0)
    l2_flush_bytes = resolve_l2_flush_bytes(0) if l2_flush is not None else 0
    shapes = _shape_cases(args)
    print(f"shape_set={args.shape_set} shapes={len(shapes)} warmup={args.warmup} iters={args.iters}")
    if l2_flush is not None:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")

    per_candidate: dict[Candidate, list[float]] = {
        candidate: [] for candidate in DEFAULT_CANDIDATES
    }
    best_rows: list[tuple[ShapeCase, int, Candidate, float]] = []

    for shape in shapes:
        print(f"\n######## {shape.label()} ########")
        print(f"note: {shape.note}")
        if not _shape_supported_by_any(shape):
            print("  skipped: no candidate supports this shape")
            continue

        for m in args.m_list:
            torch.manual_seed(42 + m + shape.n + shape.k)
            a_packed, a_sf, a_gs = make_quantized_operand(m, shape.k)
            b_packed, b_sf, b_gs = make_quantized_operand(shape.n, shape.k)
            alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)
            ref = _make_reference(
                a_packed,
                a_sf,
                b_packed,
                b_sf,
                alpha,
                m=m,
                n=shape.n,
                k=shape.k,
            )
            torch.cuda.synchronize()

            print(f"\n=== {shape.name} M={m} ===")
            print(f"  {'candidate':>24}  {'median_us':>10}  {'min_us':>8}  {'cos':>11}  {'maxabs':>9}")
            rows: list[tuple[Candidate, float]] = []
            for candidate in DEFAULT_CANDIDATES:
                if not _supported(candidate, n=shape.n, k=shape.k):
                    print(f"  {candidate.label():>24}  {'unsupported':>10}")
                    continue
                try:
                    out = torch.empty((m, shape.n, 1), device="cuda", dtype=torch.bfloat16)

                    def launch(candidate: Candidate = candidate, out: torch.Tensor = out) -> None:
                        dense_gemm(
                            (a_packed, a_sf),
                            (b_packed, b_sf),
                            out=out,
                            alpha=alpha,
                            ab_dtype="float4_e2m1fn",
                            sf_dtype="float8_e4m3fn",
                            c_dtype="bfloat16",
                            sf_vec_size=16,
                            mma_tiler_mn=candidate.tile,
                            load_path=candidate.load_path,
                            swap_ab=candidate.swap_ab,
                        )

                    replay = capture_graph_replay(launch)
                    times = bench_events(
                        replay,
                        warmup=args.warmup,
                        iters=args.iters,
                        l2_flush=l2_flush,
                    )
                    replay()
                    torch.cuda.synchronize()
                    med = statistics.median(times) * 1000
                    mn = min(times) * 1000
                    maxabs = (out[:, :, 0].float() - ref.float()).abs().max().item()
                    cos = cosine_similarity(out[:, :, 0], ref)
                    rows.append((candidate, med))
                    per_candidate[candidate].append(med)
                    print(
                        f"  {candidate.label():>24}  {med:10.1f}  {mn:8.1f}  "
                        f"{cos:11.8f}  {maxabs:9.5f}"
                    )
                except Exception as exc:
                    msg = str(exc).splitlines()[0][:80]
                    print(f"  {candidate.label():>24}  FAIL: {msg}")
            if rows:
                best = min(rows, key=lambda row: row[1])
                best_rows.append((shape, m, best[0], best[1]))
                print(f"  -> best {shape.name} M={m}: {best[0].label()} {best[1]:.1f}us")

    print("\n=== geomean median_us across successful M values ===")
    scored: list[tuple[Candidate, float]] = []
    for candidate, values in per_candidate.items():
        if not values:
            continue
        product = 1.0
        for value in values:
            product *= value
        scored.append((candidate, product ** (1.0 / len(values))))
    for candidate, geomean in sorted(scored, key=lambda row: row[1]):
        print(f"  {candidate.label():>24}  geomean_us={geomean:8.1f}")

    if best_rows:
        print("\n=== winners ===")
        for shape, m, candidate, med in best_rows:
            print(f"  {shape.name:>32}  M={m:<4}  {candidate.label():>24}  {med:8.1f}us")


if __name__ == "__main__":
    main()
