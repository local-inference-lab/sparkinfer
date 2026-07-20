#!/usr/bin/env python3
"""Probe the m16n8k32 E4M3 MXFP8 MMA fragment layout on SM120.

Determines how the A and B operand registers map to matrix positions
for the block-scaled MMA, by feeding known byte patterns and reading
back the accumulator to reverse-engineer the layout.

For a m16n8k32 E4M3 MMA:
  A: 4 u32 = 16 e4m3 bytes, covers 16 rows × 32 cols (shared across warp)
  B: 2 u32 = 8 e4m3 bytes, covers 32 rows × 8 cols (shared across warp)
  D: 4 f32 = 4 output values per thread

We feed A = identity-like patterns and B = constant, then check which
output positions light up, to determine the byte-to-position mapping.

Usage:
    cd ~/projects/sparkinfer-research/rs-4
    source ~/projects/sglang/.venv/bin/activate
    export CUTE_DSL_ARCH=sm_120a CUDA_VISIBLE_DEVICES=2
    PYTHONPATH=. python benchmarks/probe_mxfp8_fragment_layout.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from sparkinfer._lib.intrinsics import mxfp8_mma_m16n8k32_f32_e4m3


def _to_cute(x, dtype):
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


@cute.jit
def probe_mxfp8_layout(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mD: cute.Tensor,
    stream: cuda.CUstream,
):
    probe_kernel(mA, mB, mD).launch(
        grid=(1, 1, 1),
        block=[32, 1, 1],
        stream=stream,
    )


@cute.kernel
def probe_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mD: cute.Tensor,
):
    """Run one MXFP8 MMA with caller-provided A/B registers and dump D."""
    tidx = cute.arch.thread_idx()[0]

    # Load A registers (4 u32 per thread)
    a0 = mA[tidx, 0]
    a1 = mA[tidx, 1]
    a2 = mA[tidx, 2]
    a3 = mA[tidx, 3]

    # Load B registers (2 u32 per thread)
    b0 = mB[tidx, 0]
    b1 = mB[tidx, 1]

    # Scale = 1.0 (0x7F in ue8m0)
    sfa = Uint32(0x7F7F7F7F)
    sfb = Uint32(0x7F7F7F7F)

    # Zero accumulator
    d0 = Float32(0.0)
    d1 = Float32(0.0)
    d2 = Float32(0.0)
    d3 = Float32(0.0)

    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
        d0, d1, d2, d3,
        a0, a1, a2, a3,
        b0, b1,
        sfa, sfb,
    )

    # Store D (4 f32 per thread)
    mD[tidx, 0] = d0
    mD[tidx, 1] = d1
    mD[tidx, 2] = d2
    mD[tidx, 3] = d3


def e4m3_encode(val: float) -> int:
    """Encode a float to E4M3 byte using PyTorch."""
    t = torch.tensor([val], dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    return int(t.view(torch.uint8).item())


def run_probe():
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Allocate register buffers
    A = torch.zeros(32, 4, device=device, dtype=torch.int32)  # 4 u32 per thread
    B = torch.zeros(32, 2, device=device, dtype=torch.int32)  # 2 u32 per thread
    D = torch.zeros(32, 4, device=device, dtype=torch.float32)  # 4 f32 per thread

    print("Compiling MXFP8 MMA probe kernel...")
    compiled = cute.compile(
        probe_mxfp8_layout,
        _to_cute(A, cutlass.Int32),
        _to_cute(B, cutlass.Int32),
        _to_cute(D, cutlass.Float32),
        stream,
    )

    # E4M3 encoding: 0x3C = 1.0, 0x00 = 0.0
    one = e4m3_encode(1.0)
    zero = 0x00
    print(f"E4M3 encoding: 1.0 = 0x{one:02x}")

    results = {}

    # === Probe 1: Which A byte positions contribute to which D positions? ===
    # Set B to all 1.0, then set one A byte at a time to 1.0 (rest 0.0)
    # and see which D values become nonzero.
    print("\n=== Probing A register layout ===")
    b_ones = one | (one << 8) | (one << 16) | (one << 24)
    B.fill_(b_ones)

    a_layout = {}
    for reg_idx in range(4):  # 4 A registers
        for byte_idx in range(4):  # 4 bytes per register
            # Set all A to zero, then set one byte to 1.0
            A.fill_(0)
            val = one << (byte_idx * 8)
            # Set this byte for ALL threads (to see the pattern)
            A[:, reg_idx] = val

            compiled(
                _to_cute(A, cutlass.Int32),
                _to_cute(B, cutlass.Int32),
                _to_cute(D, cutlass.Float32),
                stream,
            )
            torch.cuda.synchronize()

            # Check which threads/D-positions got nonzero output
            d_cpu = D.cpu()
            nonzero_positions = []
            for tid in range(32):
                for didx in range(4):
                    v = d_cpu[tid, didx].item()
                    if abs(v) > 1e-6:
                        nonzero_positions.append({
                            "thread": tid,
                            "d_idx": didx,
                            "value": v,
                        })

            key = f"A_reg{reg_idx}_byte{byte_idx}"
            a_layout[key] = {
                "nonzero_count": len(nonzero_positions),
                "positions": nonzero_positions[:8],  # first 8 for brevity
            }
            nz_threads = set(p["thread"] for p in nonzero_positions)
            nz_didxs = set(p["d_idx"] for p in nonzero_positions)
            print(f"  {key}: {len(nonzero_positions)} nonzero, "
                  f"threads={sorted(nz_threads)[:8]}, d_idxs={sorted(nz_didxs)}")

    # === Probe 2: Which B byte positions contribute to which D positions? ===
    print("\n=== Probing B register layout ===")
    a_ones = one | (one << 8) | (one << 16) | (one << 24)
    A.fill_(a_ones)

    b_layout = {}
    for reg_idx in range(2):  # 2 B registers
        for byte_idx in range(4):  # 4 bytes per register
            B.fill_(0)
            val = one << (byte_idx * 8)
            B[:, reg_idx] = val

            compiled(
                _to_cute(A, cutlass.Int32),
                _to_cute(B, cutlass.Int32),
                _to_cute(D, cutlass.Float32),
                stream,
            )
            torch.cuda.synchronize()

            d_cpu = D.cpu()
            nonzero_positions = []
            for tid in range(32):
                for didx in range(4):
                    v = d_cpu[tid, didx].item()
                    if abs(v) > 1e-6:
                        nonzero_positions.append({
                            "thread": tid,
                            "d_idx": didx,
                            "value": v,
                        })

            key = f"B_reg{reg_idx}_byte{byte_idx}"
            b_layout[key] = {
                "nonzero_count": len(nonzero_positions),
                "positions": nonzero_positions[:8],
            }
            nz_threads = set(p["thread"] for p in nonzero_positions)
            nz_didxs = set(p["d_idx"] for p in nonzero_positions)
            print(f"  {key}: {len(nonzero_positions)} nonzero, "
                  f"threads={sorted(nz_threads)[:8]}, d_idxs={sorted(nz_didxs)}")

    # === Probe 3: Compare with BF16 m16n8k16 layout ===
    # The key question: how do the k=32 A registers map vs two k=16 A registers?
    print("\n=== Probing A k-element ordering ===")
    # Set B to all 1.0, set A to have distinct values per byte position
    B.fill_(b_ones)
    # Put unique values in A: byte i of register j gets value (j*4 + i + 1) in E4M3
    for tid in range(32):
        for reg_idx in range(4):
            packed = 0
            for byte_idx in range(4):
                val_f = float(reg_idx * 4 + byte_idx + 1)
                e = e4m3_encode(val_f)
                packed |= e << (byte_idx * 8)
            A[tid, reg_idx] = packed

    compiled(
        _to_cute(A, cutlass.Int32),
        _to_cute(B, cutlass.Int32),
        _to_cute(D, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize()

    d_cpu = D.cpu()
    print("  Thread 0 D values with A=[1..16], B=all_ones:")
    for didx in range(4):
        print(f"    D[0][{didx}] = {d_cpu[0, didx].item():.4f}")
    print("  (Each D value is a dot product over K=32 elements)")
    print(f"  Sum of 1..16 = {sum(range(1, 17))}")
    print(f"  Expected if all 16 A bytes are used: sum * 8_b_ones = {sum(range(1,17)) * 8}")

    # Dump full results
    payload = {
        "a_layout": a_layout,
        "b_layout": b_layout,
        "thread0_d_with_unique_a": [d_cpu[0, i].item() for i in range(4)],
        "all_threads_d_with_unique_a": {
            tid: [d_cpu[tid, i].item() for i in range(4)]
            for tid in range(32)
        },
    }
    out_path = pathlib.Path("probe.mxfp8_fragment_layout.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    torch.cuda.init()
    run_probe()
