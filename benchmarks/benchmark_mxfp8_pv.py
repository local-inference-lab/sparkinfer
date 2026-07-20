#!/usr/bin/env python3
"""Benchmark and validate the proposed MXFP8 PV path.

This script keeps the current research line source-backed and reproducible inside
the repo. It does three things:

1. Verifies BF16->E4M3 conversion compiles on SM120.
2. Measures synthetic inner-loop throughput for:
   - current BF16 PV path (FP8 dequant + BF16 MMA)
   - proposed MXFP8 PV path (native block-scaled MMA)
   - proposed MXFP8 PV path including BF16->E4M3 conversion cost
3. Simulates decode-style attention accuracy when P is quantized to MXFP8 and
   V remains E4M3.

Usage:
    source ~/projects/sglang/.venv/bin/activate
    export CUTE_DSL_ARCH=sm_120a
    export CUDA_VISIBLE_DEVICES=2
    python benchmarks/benchmark_mxfp8_pv.py --mode all
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from sparkinfer._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    cvt_bf16x2_to_e4m3x2,
    fp8x4_e4m3_to_bfloat2x2,
    mxfp8_mma_m16n8k32_f32_e4m3,
)


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


@cute.jit
def cvt_smoke(m_out: cute.Tensor, stream: cuda.CUstream):
    cvt_smoke_kernel(m_out).launch(grid=(1, 1, 1), block=[32, 1, 1], stream=stream)


@cute.kernel
def cvt_smoke_kernel(m_out: cute.Tensor):
    tidx = cute.arch.thread_idx()[0]
    src = Uint32(0x3F803F80)  # bf16 1.0, 1.0
    result = cvt_bf16x2_to_e4m3x2(src)
    if tidx == Int32(0):
        m_out[0] = result.to(Float32)


@cute.jit
def bench_bf16_pv(m_out: cute.Tensor, stream: cuda.CUstream, iters: Int32):
    bench_bf16_pv_kernel(m_out, iters).launch(grid=(1, 1, 1), block=[32, 1, 1], stream=stream)


@cute.kernel
def bench_bf16_pv_kernel(m_out: cute.Tensor, iters: Int32):
    tidx = cute.arch.thread_idx()[0]
    d0 = Float32(0.0)
    d1 = Float32(0.0)
    d2 = Float32(0.0)
    d3 = Float32(0.0)

    raw = Uint32(0x3C3C3C3C)
    b0 = Uint32(0x3C3C3C3C)
    b1 = Uint32(0x3C3C3C3C)

    for _ in cutlass.range(iters, unroll=1):
        a0, a1 = fp8x4_e4m3_to_bfloat2x2(raw)
        a2, a3 = fp8x4_e4m3_to_bfloat2x2(raw)
        d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)

    if tidx == Int32(0):
        m_out[0] = d0


@cute.jit
def bench_mxfp8_pv(m_out: cute.Tensor, stream: cuda.CUstream, iters: Int32):
    bench_mxfp8_pv_kernel(m_out, iters).launch(
        grid=(1, 1, 1), block=[32, 1, 1], stream=stream
    )


@cute.kernel
def bench_mxfp8_pv_kernel(m_out: cute.Tensor, iters: Int32):
    tidx = cute.arch.thread_idx()[0]
    d0 = Float32(0.0)
    d1 = Float32(0.0)
    d2 = Float32(0.0)
    d3 = Float32(0.0)

    a0 = Uint32(0x3C3C3C3C)
    a1 = Uint32(0x3C3C3C3C)
    a2 = Uint32(0x3C3C3C3C)
    a3 = Uint32(0x3C3C3C3C)
    b0 = Uint32(0x3C3C3C3C)
    b1 = Uint32(0x3C3C3C3C)
    sfa = Uint32(0x7F7F7F7F)
    sfb = Uint32(0x7F7F7F7F)

    for _ in cutlass.range(iters, unroll=1):
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb
        )

    if tidx == Int32(0):
        m_out[0] = d0


@cute.jit
def bench_mxfp8_pv_with_cvt(m_out: cute.Tensor, stream: cuda.CUstream, iters: Int32):
    bench_mxfp8_pv_with_cvt_kernel(m_out, iters).launch(
        grid=(1, 1, 1), block=[32, 1, 1], stream=stream
    )


@cute.kernel
def bench_mxfp8_pv_with_cvt_kernel(m_out: cute.Tensor, iters: Int32):
    tidx = cute.arch.thread_idx()[0]
    d0 = Float32(0.0)
    d1 = Float32(0.0)
    d2 = Float32(0.0)
    d3 = Float32(0.0)

    p0 = Uint32(0x3C003C00)
    p1 = Uint32(0x3C003C00)
    p2 = Uint32(0x3C003C00)
    p3 = Uint32(0x3C003C00)
    p4 = Uint32(0x3C003C00)
    p5 = Uint32(0x3C003C00)
    p6 = Uint32(0x3C003C00)
    p7 = Uint32(0x3C003C00)
    b0 = Uint32(0x3C3C3C3C)
    b1 = Uint32(0x3C3C3C3C)
    sfa = Uint32(0x7F7F7F7F)
    sfb = Uint32(0x7F7F7F7F)

    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)

    for _ in cutlass.range(iters, unroll=1):
        q0 = cvt_bf16x2_to_e4m3x2(p0)
        q1 = cvt_bf16x2_to_e4m3x2(p1)
        q2 = cvt_bf16x2_to_e4m3x2(p2)
        q3 = cvt_bf16x2_to_e4m3x2(p3)
        q4 = cvt_bf16x2_to_e4m3x2(p4)
        q5 = cvt_bf16x2_to_e4m3x2(p5)
        q6 = cvt_bf16x2_to_e4m3x2(p6)
        q7 = cvt_bf16x2_to_e4m3x2(p7)
        a0 = (q0 & mask16) | ((q1 & mask16) << shift16)
        a1 = (q2 & mask16) | ((q3 & mask16) << shift16)
        a2 = (q4 & mask16) | ((q5 & mask16) << shift16)
        a3 = (q6 & mask16) | ((q7 & mask16) << shift16)
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb
        )

    if tidx == Int32(0):
        m_out[0] = d0


def _bench(
    compiled,
    args,
    out_tensor,
    stream,
    warmup: int,
    repeats: int,
    *,
    l2_flush=None,
) -> float:
    cute_out = _to_cute_tensor(out_tensor, cutlass.Float32)

    def launch() -> None:
        compiled(cute_out, stream, args)

    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        launch()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for idx in range(repeats):
        if l2_flush is not None:
            l2_flush()
        starts[idx].record()
        launch()
        ends[idx].record()
    torch.cuda.synchronize()
    return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) / repeats


def quantize_to_mxfp8(x: torch.Tensor, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 weights to E4M3 with per-block ue8m0-style power-of-two scale."""
    original_shape = x.shape
    assert x.shape[-1] % block_size == 0
    x_blocked = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    block_max = x_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    log2_scale = torch.ceil(torch.log2(block_max / 448.0)).clamp(min=-127, max=127)
    scale = 2.0 ** log2_scale
    x_scaled = torch.clamp(x_blocked / scale, min=-448.0, max=448.0)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    x_dequant = x_fp8.to(torch.float32) * scale
    return x_dequant.reshape(original_shape), scale.squeeze(-1)


def apply_serving_causal_mask(scores: torch.Tensor, cache_len: int) -> torch.Tensor:
    """Apply serving-style causal masking.

    `scores` has shape [..., q_len, kv_len], where `kv_len == cache_len + q_len`
    for extend and `kv_len == cache_len` for decode. For decode `q_len == 1`, all
    cache tokens are visible, so the mask is a no-op.
    """
    q_len = scores.shape[-2]
    kv_len = scores.shape[-1]
    q_pos = torch.arange(q_len, device=scores.device).unsqueeze(-1)
    k_pos = torch.arange(kv_len, device=scores.device).unsqueeze(0)
    visible_until = cache_len + q_pos
    mask = k_pos > visible_until
    scores.masked_fill_(mask, float("-inf"))
    return scores


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cache_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    apply_serving_causal_mask(scores, cache_len)
    probs = F.softmax(scores, dim=-1)
    probs_bf16 = probs.to(torch.bfloat16).float()
    v_fp8 = v.to(torch.float8_e4m3fn).to(torch.float32)
    return probs_bf16 @ v_fp8, probs


def mxfp8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    *,
    cache_len: int,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    apply_serving_causal_mask(scores, cache_len)
    probs = F.softmax(scores, dim=-1)
    seq_k = probs.shape[-1]
    pad = (block_size - seq_k % block_size) % block_size
    probs_padded = F.pad(probs, (0, pad), value=0.0) if pad else probs
    probs_q, _ = quantize_to_mxfp8(probs_padded.to(torch.bfloat16), block_size=block_size)
    if pad:
        probs_q = probs_q[..., :seq_k]
    v_fp8 = v.to(torch.float8_e4m3fn).to(torch.float32)
    return probs_q @ v_fp8


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def run_accuracy_suite(block_size: int, seed: int, cos_threshold: float) -> int:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    cases = []
    for kv_len in [64, 512, 2048, 8192, 32768]:
        cases.append(("decode", 8, 1, kv_len, 1.0, kv_len))
        cases.append(("decode_adversarial", 8, 1, kv_len, 6.0, kv_len))

    failures = 0
    print("Accuracy sweep")
    for label, batch, q_len, kv_len, q_scale, cache_len in cases:
        q = (torch.randn(batch, 1, q_len, 256, device=device, dtype=torch.bfloat16) * q_scale)
        k = (torch.randn(batch, 1, kv_len, 256, device=device, dtype=torch.bfloat16) * q_scale)
        v = torch.randn(batch, 1, kv_len, 256, device=device, dtype=torch.bfloat16)
        ref, _ = reference_attention(q, k, v, cache_len=cache_len)
        cand = mxfp8_attention(q, k, v, block_size=block_size, cache_len=cache_len)
        cos = cosine_similarity(ref, cand)
        max_err = (ref.float() - cand.float()).abs().max().item()
        tag = "PASS" if cos > cos_threshold else "FAIL"
        if tag == "FAIL":
            failures += 1
        print(
            f"  {label:16s} q={q_len:2d} k={kv_len:5d} | cos={cos:.8f} max_err={max_err:.6f} [{tag}]"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the MXFP8 PV proposal.")
    parser.add_argument(
        "--mode",
        choices=["all", "cvt", "throughput", "accuracy"],
        default="all",
    )
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cos-threshold", type=float, default=0.9995)
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    args = parser.parse_args()

    torch.cuda.init()
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    device = torch.device("cuda")
    out = torch.zeros(1, device=device, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cute_out = _to_cute_tensor(out, cutlass.Float32)
    flush_desc = f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)" if l2_flush else "off"
    print(f"L2 flush: {flush_desc}")

    if args.mode in {"all", "cvt"}:
        print("Compiling BF16->E4M3 conversion smoke kernel...")
        compiled_cvt = cute.compile(cvt_smoke, cute_out, stream)
        compiled_cvt(cute_out, stream)
        torch.cuda.synchronize()
        print("  SUCCESS")

    if args.mode in {"all", "throughput"}:
        iter_arg = Int32(args.iters)
        print("Compiling throughput kernels...")
        compiled_bf16 = cute.compile(bench_bf16_pv, cute_out, stream, iter_arg)
        compiled_mxfp8 = cute.compile(bench_mxfp8_pv, cute_out, stream, iter_arg)
        compiled_mxfp8_cvt = cute.compile(bench_mxfp8_pv_with_cvt, cute_out, stream, iter_arg)
        bf16_ms = _bench(
            compiled_bf16,
            iter_arg,
            out,
            stream,
            args.warmup,
            args.repeats,
            l2_flush=l2_flush,
        )
        mxfp8_ms = _bench(
            compiled_mxfp8,
            iter_arg,
            out,
            stream,
            args.warmup,
            args.repeats,
            l2_flush=l2_flush,
        )
        mxfp8_cvt_ms = _bench(
            compiled_mxfp8_cvt,
            iter_arg,
            out,
            stream,
            args.warmup,
            args.repeats,
            l2_flush=l2_flush,
        )
        bf16_k = 16 * args.iters / bf16_ms
        mxfp8_k = 32 * args.iters / mxfp8_ms
        mxfp8_cvt_k = 32 * args.iters / mxfp8_cvt_ms
        print("Throughput")
        print(f"  BF16 dequant+mma    : {bf16_ms:8.4f} ms | {bf16_k:10.0f} K-elems/ms")
        print(
            f"  MXFP8 pure mma      : {mxfp8_ms:8.4f} ms | {mxfp8_k:10.0f} K-elems/ms | {(mxfp8_k / bf16_k):5.2f}x K-throughput"
        )
        print(
            f"  MXFP8 cvt+mma       : {mxfp8_cvt_ms:8.4f} ms | {mxfp8_cvt_k:10.0f} K-elems/ms | {(mxfp8_cvt_k / bf16_k):5.2f}x K-throughput"
        )

    if args.mode in {"all", "accuracy"}:
        failures = run_accuracy_suite(args.block_size, args.seed, args.cos_threshold)
        if failures:
            raise SystemExit(f"accuracy sweep failed: {failures} cases below threshold")


if __name__ == "__main__":
    main()
