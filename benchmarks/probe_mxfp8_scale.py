#!/usr/bin/env python3
"""Probe ue8m0 scale encoding for SM120 MXFP8 MMA."""

from __future__ import annotations
import sys, pathlib
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
def probe_scale(mA: cute.Tensor, mB: cute.Tensor, mSFA: cute.Tensor, mD: cute.Tensor, stream: cuda.CUstream):
    probe_scale_kernel(mA, mB, mSFA, mD).launch(grid=(1,1,1), block=[32,1,1], stream=stream)


@cute.kernel
def probe_scale_kernel(mA: cute.Tensor, mB: cute.Tensor, mSFA: cute.Tensor, mD: cute.Tensor):
    tidx = cute.arch.thread_idx()[0]
    lane = tidx  # single warp, so tidx == lane_id

    a0 = mA[tidx, 0]
    a1 = mA[tidx, 1]
    a2 = mA[tidx, 2]
    a3 = mA[tidx, 3]
    b0 = mB[tidx, 0]
    b1 = mB[tidx, 1]

    # Load scale factor the way gau-nernst does:
    # Each thread loads from a different smem address based on lane_id.
    # The scale factor covers 32 rows, with 4 bytes per u32 (one per k-step).
    # For our single k-step, all 4 bytes should be the same scale value.
    # Address pattern: (lane_id % 4) * 8 + (lane_id / 4)
    # This means the 32 threads load from 32 different positions within
    # a 32-element scale factor table stored in shared memory.
    #
    # For simplicity, we just use the global-memory value directly,
    # which means all threads get the same scale register value.
    sfa = mSFA[0]
    sfb = mSFA[0]

    d0 = Float32(0.0)
    d1 = Float32(0.0)
    d2 = Float32(0.0)
    d3 = Float32(0.0)
    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
        d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
    )
    mD[tidx, 0] = d0
    mD[tidx, 1] = d1
    mD[tidx, 2] = d2
    mD[tidx, 3] = d3


def main():
    torch.cuda.init()
    device = torch.device("cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # E4M3 1.0 = 0x38
    one_e4m3 = 0x38
    all_ones = one_e4m3 | (one_e4m3 << 8) | (one_e4m3 << 16) | (one_e4m3 << 24)

    A = torch.full((32, 4), all_ones, device=device, dtype=torch.int32)
    B = torch.full((32, 2), all_ones, device=device, dtype=torch.int32)
    SFA = torch.zeros(1, device=device, dtype=torch.int32)
    D = torch.zeros(32, 4, device=device, dtype=torch.float32)

    compiled = cute.compile(
        probe_scale,
        _to_cute(A, cutlass.Int32),
        _to_cute(B, cutlass.Int32),
        _to_cute(SFA, cutlass.Int32),
        _to_cute(D, cutlass.Float32),
        stream,
    )

    print("Probing ue8m0 scale encoding (A=1.0, B=1.0, sfb=0x7F)")
    print("Expected: D = sum(A*B over k=32) * scale_a * scale_b")
    print("With A=B=1.0 and k=32, base dot product = 32.0")
    print()

    for scale_byte in [0x00, 0x01, 0x3F, 0x7D, 0x7E, 0x7F]:
        scale_packed = scale_byte | (scale_byte << 8) | (scale_byte << 16) | (scale_byte << 24)
        SFA.fill_(scale_packed)

        compiled(
            _to_cute(A, cutlass.Int32),
            _to_cute(B, cutlass.Int32),
            _to_cute(SFA, cutlass.Int32),
            _to_cute(D, cutlass.Float32),
            stream,
        )
        torch.cuda.synchronize()

        d_val = D[0, 1].item()  # use d_idx=1 since that's where output lands
        print(f"  sfa=0x{scale_byte:02x} ({scale_byte:3d}) -> D[0][1] = {d_val:>20.6f}")


if __name__ == "__main__":
    main()
