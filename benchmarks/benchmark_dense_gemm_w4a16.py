#!/usr/bin/env python3
"""Benchmark b12x W4A16 dense GEMM at the decode shapes vs Marlin.

Bench harness for the canonical W4A16 dense decode linears (attention
q/k/v/o, shared expert up/down, mamba in/out, lm_head).  Compares the
v4 forked CuTe-DSL kernel against the silicon **Marlin** per-call
times captured on Spark (NVIDIA GB10, SM121).

Usage:
  python benchmarks/benchmark_dense_gemm_w4a16.py
  python benchmarks/benchmark_dense_gemm_w4a16.py --m-list 1,8,16

On Spark (cutlass-dsl 4.3.4 default), prepend the 4.4.2 sidecar to
PYTHONPATH:
  pip3 install --target=/tmp/cutlass_4_4_2 'nvidia-cutlass-dsl==4.4.2'
  export PYTHONPATH=/tmp/cutlass_4_4_2/nvidia_cutlass_dsl/python_packages:${PYTHONPATH}
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Callable

# Make `b12x` importable when invoked as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.gemm.w4a16 import quantize_dense_weight_to_fp4, dense_gemm_w4a16


# Silicon Marlin per-call times captured on Spark (NVIDIA GB10, SM121).
# Source: vLLM serve, ISL≈32K, NVFP4 W4A16, ``marlin<128,1,4,8>`` +
# ``marlin<256,1,8,8>`` kernels.  Only present for M=1.
_DECODE_SHAPES = [
    # (name, K, N, marlin_us_at_M1)
    ("self_attn_qkv_linear", 2688,   4608,  35.8),
    ("self_attn_out_linear", 4096,   2688,  32.1),
    ("shared_fc1",           2688,   3712,  34.3),
    ("shared_fc2",           3712,   2688,  68.9),
    ("mamba_in_proj",        2688,  10304,  84.7),
    ("mamba_output_proj",    4096,   2688,  32.8),
    ("lm_head",              2688, 131072, 878.9),
]


def _bench(fn: Callable, warmup: int = 10, iters: int = 50) -> float:
    """Median per-call time in microseconds via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    S = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    E = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        S[i].record()
        fn()
        E[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(S, E))
    return times[iters // 2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--m-list", default="1,8,16",
        help="comma-separated M values to bench (decode-only: 1, 2, 4, 8, 10, 12, 16, 24, 32)",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(
        f"SM={props.major}{props.minor}  GPU={props.name}  torch={torch.__version__}",
        flush=True,
    )

    ms = [int(x) for x in args.m_list.split(",") if x]
    hdr = (
        f"{'shape':22s} {'M':>3s} {'K':>5s} {'N':>6s}  "
        f"{'marlin_us':>9s}  {'b12x_us':>8s}  {'b12x/marlin':>11s}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for name, k, n, marlin_us in _DECODE_SHAPES:
        for m in ms:
            torch.manual_seed(args.seed)
            x = (torch.randn(m, k, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
            w = (torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
            w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)

            def call():
                return dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha, out=out)

            try:
                us = _bench(call, warmup=args.warmup, iters=args.iters)
            except Exception as e:
                print(f"# {name} M={m}: failed: {e}", flush=True)
                us = None

            marl = f"{marlin_us:7.1f}us" if (marlin_us is not None and m == 1) else "    -"
            us_s = f"{us:6.1f}us" if us is not None else "    n/a"
            ratio = (
                f"{us / marlin_us:5.2f}x"
                if (us is not None and marlin_us is not None and m == 1)
                else "    -"
            )
            print(
                f"{name:22s} {m:3d} {k:5d} {n:6d}  {marl:>9s}  {us_s:>8s}  {ratio:>11s}",
                flush=True,
            )
        print("-" * len(hdr), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
