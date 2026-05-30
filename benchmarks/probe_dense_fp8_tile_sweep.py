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

import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmark_dense_gemm import (
    bench_events,
    capture_graph_replay,
    cosine_similarity,
    make_l2_flush_fn,
    make_mxfp8_operand,
)
from b12x.gemm.dense import DenseGemmKernel, dense_gemm

import cutlass

# Nemotron 3 Super shared-expert down-proj (the benchmark's FP8 shape).
# Shape: default = Nemotron down-proj; override via argv "N K" to sweep other
# n>1536 shapes (e.g. GLM5 dense-down 6144 1536, a wo_b-like 7168 512).
if len(sys.argv) >= 3:
    N = int(sys.argv[1])
    K = int(sys.argv[2])
else:
    K = 5376
    N = 4096
# Extended to large M to find where the small-tile win ends and (128,128) (the
# compute-bound large-M regime) takes back over -- the benchmark only covers
# M<=256, but prefill goes higher and the heuristic must not regress there.
M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048]

# Candidate tiles. can_implement allows tile_m%64==0 & tile_n%64==0, plus the
# FP8-only small-M tiles {16,32}x{64,128}. tile_n>128 does not compile on this
# kernel (BLOCK_N capped at 128), so only the compiling survivors are probed.
# Small-N (x64) tiles give 2x the column tiles -> more CTAs at tiny M.
TILES = [
    (128, 128),  # current heuristic pin (baseline)
    (64, 128),
    (64, 64),
    (32, 128),
    (32, 64),
    (16, 128),
    (16, 64),
]

WARMUP = 10
ITERS = 50


def _supported(tile, m):
    return DenseGemmKernel.can_implement(
        cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32, cutlass.BFloat16,
        tile, (1, 1), N, K, 1, "k", "k", "n",
    )


def main():
    torch.manual_seed(42)
    l2_flush = make_l2_flush_fn(enabled=True, bytes_hint=0)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device SMs={sm}  shape N={N} K={K}  warmup={WARMUP} iters={ITERS}")

    # best[(metric)] tracking for the single-tile-across-M summary
    per_tile_ratiosum = {t: [] for t in TILES}

    for M in M_LIST:
        a_q, a_sf, a_sf_mma, _ = make_mxfp8_operand(M, K)
        b_q, b_sf, b_sf_mma, _ = make_mxfp8_operand(N, K)

        # Known-good reference = (128,128) output (baseline validated cos=1.0 vs CUTLASS).
        ref_out = torch.empty((M, N, 1), device="cuda", dtype=torch.bfloat16)

        def make_launch(tile, out):
            def launch():
                dense_gemm(
                    (a_q.view(M, K, 1), a_sf_mma),
                    (b_q.view(N, K, 1), b_sf_mma),
                    ab_dtype="float8_e4m3fn",
                    sf_dtype="float8_e8m0fnu",
                    c_dtype="bfloat16",
                    sf_vec_size=32,
                    out=out,
                    mma_tiler_mn=tile,
                )
            return launch

        # reference first
        make_launch((128, 128), ref_out)()
        torch.cuda.synchronize()

        print(f"\n=== M={M} ===")
        print(f"  {'tile':>10}  {'median_us':>10}  {'min_us':>8}  {'cos_vs_128':>11}  {'maxabs':>9}")
        rows = []
        for tile in TILES:
            if not _supported(tile, M):
                print(f"  {str(tile):>10}  {'unsupported':>10}")
                per_tile_ratiosum[tile].append(None)
                continue
            try:
                out = torch.empty((M, N, 1), device="cuda", dtype=torch.bfloat16)
                replay = capture_graph_replay(make_launch(tile, out))
                t = bench_events(replay, warmup=WARMUP, iters=ITERS, l2_flush=l2_flush)
                med = statistics.median(t) * 1000
                mn = min(t) * 1000
                cos = cosine_similarity(out[:, :, 0], ref_out[:, :, 0])
                maxabs = (out.float() - ref_out.float()).abs().max().item()
                rows.append((tile, med, cos))
                per_tile_ratiosum[tile].append(med)
                print(f"  {str(tile):>10}  {med:10.1f}  {mn:8.1f}  {cos:11.8f}  {maxabs:9.5f}")
            except Exception as exc:
                per_tile_ratiosum[tile].append(None)
                msg = str(exc).splitlines()[0][:60]
                print(f"  {str(tile):>10}  FAIL: {msg}")
        if rows:
            best = min(rows, key=lambda r: r[1])
            base = next((r for r in rows if r[0] == (128, 128)), None)
            spd = (base[1] / best[1]) if base else float("nan")
            print(f"  -> best M={M}: tile={best[0]} {best[1]:.1f}us"
                  + (f"  ({spd:.2f}x vs (128,128) {base[1]:.1f}us)" if base else ""))

    # Single durable tile: geomean of median_us across M (lower better)
    print("\n=== single-tile-across-M (geomean median_us, lower=better) ===")
    scored = []
    for tile, vals in per_tile_ratiosum.items():
        good = [v for v in vals if v is not None]
        if len(good) != len([m for m in M_LIST]):
            # only consider tiles valid for all M
            continue
        g = 1.0
        for v in good:
            g *= v
        g **= 1.0 / len(good)
        scored.append((tile, g))
    for tile, g in sorted(scored, key=lambda x: x[1]):
        print(f"  {str(tile):>10}  geomean_us={g:8.1f}")


if __name__ == "__main__":
    main()
