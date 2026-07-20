#!/usr/bin/env python3
"""Probe: sweep mma_tiler_mn for the Nemotron FP8 dense GEMM across M.

Finds the per-M optimal tile (and the best single durable tile) for the
MXFP8 dense_gemm path, to inform _select_default_mma_tiler_mn. Reuses the
real benchmark's MXFP8 input construction + graph-replay timing so results
transfer directly to benchmark_dense_gemm.py.

Run:  CUDA_VISIBLE_DEVICES=0 .../vllm-other/.venv/bin/python \
        benchmarks/probe_dense_fp8_tile_sweep.py
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.benchmark_dense_gemm import (
    bench_events,
    capture_graph_replay,
    cosine_similarity,
    make_mxfp8_operand,
)
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from sparkinfer.gemm.dense import DenseGemmKernel, _select_default_mma_tiler_mn, dense_gemm

import cutlass

@dataclass(frozen=True)
class ShapeCase:
    name: str
    n: int
    k: int
    note: str

    def label(self) -> str:
        return f"{self.name}: N={self.n} K={self.k}"


@dataclass(frozen=True)
class Candidate:
    tile: tuple[int, int]
    swap_ab: bool = False

    def label(self) -> str:
        suffix = "/swap" if self.swap_ab else "/normal"
        return f"{self.tile}{suffix}"


# Extended to large M to find where the small-tile win ends and (128,128) takes
# back over. The common-shape path below can be restricted to M=1 for decode.
DEFAULT_M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048]

# Candidate tiles. can_implement allows tile_m%64==0 & tile_n%64==0, plus the
# FP8-only small-M tiles {16,32}x{64,128}. tile_n>128 does not compile on this
# kernel (BLOCK_N capped at 128), so only the compiling survivors are probed.
# Small-N (x64) tiles give 2x the column tiles -> more CTAs at tiny M.
TILES = [
    Candidate((128, 128)),  # current heuristic pin (baseline)
    Candidate((64, 128)),
    Candidate((64, 64)),
    Candidate((32, 128)),
    Candidate((32, 64)),
    Candidate((16, 128)),
    Candidate((16, 64)),
    Candidate((128, 32), True),
    Candidate((128, 16), True),
    Candidate((64, 32), True),
    Candidate((64, 16), True),
]

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
    ShapeCase(
        "glm5_dense_down",
        6144,
        1536,
        "shape called out by the original FP8 probe comment",
    ),
    ShapeCase(
        "wo_b_like_n7168_k512",
        7168,
        512,
        "shape called out by the original FP8 probe comment",
    ),
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


def _supported(candidate: Candidate, *, n: int, k: int):
    return DenseGemmKernel.can_implement(
        cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32, cutlass.BFloat16,
        candidate.tile, (1, 1), n, k, 1, "k", "k", "n",
        swap_ab=candidate.swap_ab,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_n", type=int, nargs="?")
    parser.add_argument("legacy_k", type=int, nargs="?")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=5376)
    parser.add_argument("--shape-set", choices=("single", "common"), default="single")
    parser.add_argument("--list-common-shapes", action="store_true")
    parser.add_argument("--m-list", type=_parse_m_list, default=DEFAULT_M_LIST)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-l2-flush", action="store_true")
    args = parser.parse_args()
    if args.legacy_n is not None:
        args.n = args.legacy_n
    if args.legacy_k is not None:
        args.k = args.legacy_k
    if args.list_common_shapes:
        for shape in COMMON_SHAPES:
            print(f"{shape.label()}  # {shape.note}")
        return

    l2_flush = make_l2_flush_fn(enabled=not args.no_l2_flush, bytes_hint=0)
    l2_flush_bytes = resolve_l2_flush_bytes(0) if l2_flush is not None else 0
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    shapes = _shape_cases(args)
    print(f"device SMs={sm}  shape_set={args.shape_set} shapes={len(shapes)}  warmup={args.warmup} iters={args.iters}")
    if l2_flush is not None:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")

    # best[(metric)] tracking for the single-tile-across-M summary
    per_tile_ratiosum = {t: [] for t in TILES}
    best_rows = []

    for shape in shapes:
        print(f"\n######## {shape.label()} ########")
        print(f"note: {shape.note}")
        for M in args.m_list:
            torch.manual_seed(42 + M + shape.n + shape.k)
            a_q, a_sf, a_sf_mma, _ = make_mxfp8_operand(M, shape.k)
            b_q, b_sf, b_sf_mma, _ = make_mxfp8_operand(shape.n, shape.k)

            # Known-good reference = (128,128) output.
            ref_out = torch.empty((M, shape.n, 1), device="cuda", dtype=torch.bfloat16)

            def make_launch(candidate, out):
                def launch():
                    dense_gemm(
                        (a_q.view(M, shape.k, 1), a_sf_mma),
                        (b_q.view(shape.n, shape.k, 1), b_sf_mma),
                        ab_dtype="float8_e4m3fn",
                        sf_dtype="float8_e8m0fnu",
                        c_dtype="bfloat16",
                        sf_vec_size=32,
                        out=out,
                        mma_tiler_mn=candidate.tile,
                        swap_ab=candidate.swap_ab,
                    )
                return launch

            # reference first
            make_launch(Candidate((128, 128)), ref_out)()
            torch.cuda.synchronize()

            default_tile = _select_default_mma_tiler_mn(
                M,
                shape.n,
                sm,
                is_mxfp8=True,
            )
            print(f"\n=== {shape.name} M={M} default={default_tile} ===")
            print(f"  {'candidate':>20}  {'median_us':>10}  {'min_us':>8}  {'cos_vs_128':>11}  {'maxabs':>9}")
            rows = []
            for candidate in TILES:
                if not _supported(candidate, n=shape.n, k=shape.k):
                    print(f"  {candidate.label():>20}  {'unsupported':>10}")
                    continue
                try:
                    out = torch.empty((M, shape.n, 1), device="cuda", dtype=torch.bfloat16)
                    replay = capture_graph_replay(make_launch(candidate, out))
                    t = bench_events(replay, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush)
                    med = statistics.median(t) * 1000
                    mn = min(t) * 1000
                    cos = cosine_similarity(out[:, :, 0], ref_out[:, :, 0])
                    maxabs = (out.float() - ref_out.float()).abs().max().item()
                    rows.append((candidate, med, cos))
                    per_tile_ratiosum[candidate].append(med)
                    print(f"  {candidate.label():>20}  {med:10.1f}  {mn:8.1f}  {cos:11.8f}  {maxabs:9.5f}")
                except Exception as exc:
                    msg = str(exc).splitlines()[0][:60]
                    print(f"  {candidate.label():>20}  FAIL: {msg}")
            if rows:
                best = min(rows, key=lambda r: r[1])
                base = next((r for r in rows if r[0] == Candidate((128, 128))), None)
                default = next((r for r in rows if r[0] == Candidate(default_tile)), None)
                spd = (base[1] / best[1]) if base else float("nan")
                default_ratio = (default[1] / best[1]) if default else float("nan")
                best_rows.append((shape, M, best[0], best[1], default_tile, default_ratio))
                print(f"  -> best {shape.name} M={M}: candidate={best[0].label()} {best[1]:.1f}us"
                      + (f"  ({spd:.2f}x vs (128,128) {base[1]:.1f}us)" if base else "")
                      + (f"  default/best={default_ratio:.2f}x" if default else ""))

    # Single durable tile: geomean of median_us across M (lower better)
    print("\n=== single-tile-across-M (geomean median_us, lower=better) ===")
    scored = []
    for candidate, vals in per_tile_ratiosum.items():
        good = [v for v in vals if v is not None]
        if not good:
            continue
        g = 1.0
        for v in good:
            g *= v
        g **= 1.0 / len(good)
        scored.append((candidate, g))
    for candidate, g in sorted(scored, key=lambda x: x[1]):
        print(f"  {candidate.label():>20}  geomean_us={g:8.1f}")
    if best_rows:
        print("\n=== winners ===")
        for shape, M, candidate, med, default_tile, default_ratio in best_rows:
            print(
                f"  {shape.name:>28}  M={M:<4}  best={candidate.label():>20}  "
                f"{med:8.1f}us  default={default_tile}  default/best={default_ratio:.2f}x"
            )


if __name__ == "__main__":
    main()
