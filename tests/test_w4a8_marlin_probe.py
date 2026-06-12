"""Staging probe for the W4A8 throughput-tier (Marlin-skeleton) QMMA mainloop.

The register-level QMMA contract (fragment k-permutation, e2m1->e4m3 B
expansion, SF selector mapping) is already pinned by
tests/test_w4a8_fragment_probe.py. This probe pins what the Marlin port adds
on top, bit-tight and before touching w4a16/kernel.py:

- the host-side B repack: ``B_rp[k_tile][n8][k32][lane] u32`` so each lane's
  per-(n8, k32) packed-FP4 word read is a coalesced u32;
- plain row-major e4m3 A staging (128B rows per k-tile) with the
  a0..a3 = {row q, row q+8} x {+0B, +4B} fragment addressing;
- per-lane scale words: A scales as one u32 per (row, k-tile) with the
  consumed byte extracted per k32 step (lane l supplies row (l>>2) + 8*(l&1);
  only quad-columns 0/1 are consumed), B scales as one u32 per (n8, col,
  k-tile) (lane l supplies col l>>2; only quad-column 0 is consumed);
- multi-n8 accumulation per warp and k-tile sweep accumulation in registers.

Dyadic payloads + power-of-two scales make every product and partial sum
exact in f32, so all assertions are atol=0.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import (
    e2m1x8_to_e4m3x8,
    ld_shared_u32,
    mxfp8_mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    st_shared_u32,
)
from tests.test_w4a8_fragment_probe import _e4m3_bytes, _pack_b

from .helpers import require_sm120

_M = 16          # one m-block
_N8 = 2          # n8 tiles per warp -> tile_n 16
_N = 8 * _N8
_TILE_K = 128    # 4 k32 steps per tile


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


class _MarlinProbeKernel:
    num_threads = 32

    def __init__(self, *, k_tiles: int):
        self.k_tiles = int(k_tiles)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,    # flat u32: [16, K/4] row-major e4m3 bytes
        mAsf: cute.Tensor,  # flat u32: [16, K/128] (4 scale bytes per k-tile)
        mB: cute.Tensor,    # flat u32: [k_tiles, _N8, 4, 32] repacked fp4 words
        mBsf: cute.Tensor,  # flat u32: [k_tiles, _N8, 8] (4 scale bytes per k-tile)
        mOut: cute.Tensor,  # [16, _N] f32
        stream: cuda.CUstream,
    ):
        self.kernel(mA, mAsf, mB, mBsf, mOut).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mAsf: cute.Tensor,
        mB: cute.Tensor,
        mBsf: cute.Tensor,
        mOut: cute.Tensor,
    ):
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, 16 * (_TILE_K // 4)], 16
            ]
            sAsf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, 16], 16]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, _N8 * 4 * 32], 16
            ]
            sBsf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, _N8 * 8], 16
            ]

        storage = smem.allocate(Storage)
        sa_base = shared_ptr_to_u32(storage.sA.data_ptr())
        sasf_base = shared_ptr_to_u32(storage.sAsf.data_ptr())
        sb_base = shared_ptr_to_u32(storage.sB.data_ptr())
        sbsf_base = shared_ptr_to_u32(storage.sBsf.data_ptr())

        lane = cute.arch.lane_idx()
        q = lane >> Int32(2)        # quad row: A rows q / q+8, B col q
        c = lane & Int32(3)         # quad column: 8-byte k group within k32

        # f32 accumulators: [n8 tile][4 fragment regs], carried across k-tiles.
        facc = tuple(Float32(0.0) for _ in range(_N8 * 4))

        a_u32_per_row = Int32(self.k_tiles * (_TILE_K // 4))

        for kt in cutlass.range_constexpr(self.k_tiles):
            # ---- stage this k-tile (plain ld/st; the Marlin integration will
            # reuse its own cp.async pipeline — staging mechanism is not what
            # this probe pins) ----
            for s in cutlass.range_constexpr(16):
                idx = Int32(s * 32) + lane          # [0, 512): row*32 + j
                row = idx >> Int32(5)
                j = idx & Int32(31)
                word = Uint32(mA[row * a_u32_per_row + Int32(kt * 32) + j])
                st_shared_u32(sa_base + (idx << Int32(2)), word)
            if lane < Int32(16):
                w = Uint32(mAsf[lane * Int32(self.k_tiles) + Int32(kt)])
                st_shared_u32(sasf_base + (lane << Int32(2)), w)
            for s in cutlass.range_constexpr(_N8 * 4):
                idx = Int32(s * 32) + lane          # [(n8*4 + kb)*32 + lane']
                word = Uint32(mB[Int32(kt * _N8 * 4 * 32) + idx])
                st_shared_u32(sb_base + (idx << Int32(2)), word)
            if lane < Int32(_N8 * 8):
                w = Uint32(mBsf[Int32(kt * _N8 * 8) + lane])
                st_shared_u32(sbsf_base + (lane << Int32(2)), w)
            cute.arch.sync_threads()

            # ---- consume: 4 k32 steps ----
            # Lane l supplies the SFA byte for row q + 8*(l&1) (quad-cols 2/3
            # are unconsumed; the &1 keeps their loads in range).
            asf_row = q + ((lane & Int32(1)) << Int32(3))
            asc_word = ld_shared_u32(sasf_base + (asf_row << Int32(2)))
            for kb in cutlass.range_constexpr(4):
                a_lo = sa_base + (q << Int32(7)) + Int32(kb * 32) + (c << Int32(3))
                a0 = ld_shared_u32(a_lo)
                a2 = ld_shared_u32(a_lo + Int32(4))
                a1 = ld_shared_u32(a_lo + Int32(8 * 128))
                a3 = ld_shared_u32(a_lo + Int32(8 * 128 + 4))
                sfa_b = (asc_word >> Uint32(kb * 8)) & Uint32(0xFF)
                for nt in cutlass.range_constexpr(_N8):
                    w_pk = ld_shared_u32(
                        sb_base + ((Int32((nt * 4 + kb) * 32) + lane) << Int32(2))
                    )
                    b0, b1 = e2m1x8_to_e4m3x8(w_pk)
                    sfb_word = ld_shared_u32(
                        sbsf_base + ((Int32(nt * 8) + q) << Int32(2))
                    )
                    sfb_b = (sfb_word >> Uint32(kb * 8)) & Uint32(0xFF)
                    _fi = nt * 4
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        facc[_fi],
                        facc[_fi + 1],
                        facc[_fi + 2],
                        facc[_fi + 3],
                        a0, a1, a2, a3,
                        b0, b1,
                        sfa_b, sfb_b,
                    )
                    facc = facc[:_fi] + (d0, d1, d2, d3) + facc[_fi + 4 :]
            cute.arch.sync_threads()

        col = Int32(2) * c
        for nt in cutlass.range_constexpr(_N8):
            _fi = nt * 4
            col_base = Int32(nt * 8) + col
            mOut[q, col_base] = facc[_fi]
            mOut[q, col_base + Int32(1)] = facc[_fi + 1]
            mOut[q + Int32(8), col_base] = facc[_fi + 2]
            mOut[q + Int32(8), col_base + Int32(1)] = facc[_fi + 3]


def _repack_b(b_packed: torch.Tensor, k_tiles: int) -> torch.Tensor:
    """[N, K/2] u8 row-major -> [k_tiles, _N8, 4, 32] u32 lane-coalesced.

    Word (kt, n8, kb, lane) = the 8 packed nibbles of B row ``n8*8 + lane>>2``
    covering original k in ``[kt*128 + kb*32 + (lane&3)*8, +8)``.
    """
    n, _ = b_packed.shape
    assert n == _N
    b_u32 = b_packed.view(torch.int32).reshape(n, -1)  # [N, K/8] u32
    out = torch.empty(k_tiles, _N8, 4, 32, dtype=torch.int32, device=b_packed.device)
    for lane in range(32):
        row_in_n8 = lane >> 2
        cgrp = lane & 3
        for n8 in range(_N8):
            for kt in range(k_tiles):
                for kb in range(4):
                    out[kt, n8, kb, lane] = b_u32[
                        n8 * 8 + row_in_n8, kt * 16 + kb * 4 + cgrp
                    ]
    return out.contiguous()


def _sf_words_a(sf_bytes: torch.Tensor) -> torch.Tensor:
    """[16, K/32] u8 -> flat u32 [16, k_tiles] (byte kb of word = k-block kb)."""
    m, kg = sf_bytes.shape
    assert kg % 4 == 0
    return sf_bytes.reshape(m, kg // 4, 4).contiguous().view(torch.int32).reshape(m, -1)


def _sf_words_b(sf_bytes: torch.Tensor, k_tiles: int) -> torch.Tensor:
    """[N, K/32] u8 -> [k_tiles, _N8, 8] u32 per-col words."""
    n, kg = sf_bytes.shape
    assert n == _N and kg == k_tiles * 4
    out = torch.empty(k_tiles, _N8, 8, 4, dtype=torch.uint8, device=sf_bytes.device)
    for kt in range(k_tiles):
        for n8 in range(_N8):
            for col in range(8):
                out[kt, n8, col] = sf_bytes[n8 * 8 + col, kt * 4 : kt * 4 + 4]
    return out.contiguous().view(torch.int32).reshape(-1)


def _run_probe(
    a_bytes: torch.Tensor,
    a_sf: torch.Tensor,
    b_packed: torch.Tensor,
    b_sf: torch.Tensor,
    k_tiles: int,
) -> torch.Tensor:
    device = a_bytes.device
    out = torch.zeros(_M, _N, device=device, dtype=torch.float32)
    args = (
        _to_cute_tensor(a_bytes.view(torch.int32).reshape(-1), cutlass.Uint32),
        _to_cute_tensor(_sf_words_a(a_sf).reshape(-1), cutlass.Uint32),
        _to_cute_tensor(_repack_b(b_packed, k_tiles).reshape(-1), cutlass.Uint32),
        _to_cute_tensor(_sf_words_b(b_sf, k_tiles), cutlass.Uint32),
        _to_cute_tensor(out, cutlass.Float32),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    kernel = _MarlinProbeKernel(k_tiles=k_tiles)
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return out


def _dyadic_inputs(k_tiles: int, seed: int):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    k = k_tiles * _TILE_K
    a_vals = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0, -0.5, 4.0, -2.0], device=device)
    a = a_vals[torch.randint(0, 8, (_M, k), device=device)]
    fp4_grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
    )
    b = fp4_grid[torch.randint(0, 15, (_N, k), device=device)]
    return a, b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("k_tiles", [1, 4])
def test_marlin_probe_unit_scales_exact(k_tiles: int) -> None:
    require_sm120()
    device = torch.device("cuda")
    a, b = _dyadic_inputs(k_tiles, seed=10 + k_tiles)
    kg = k_tiles * 4
    unit_a = torch.full((_M, kg), 127, dtype=torch.uint8, device=device)
    unit_b = torch.full((_N, kg), 127, dtype=torch.uint8, device=device)
    out = _run_probe(_e4m3_bytes(a), unit_a, _pack_b(b), unit_b, k_tiles)
    torch.testing.assert_close(out, a @ b.T, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("k_tiles", [1, 4])
def test_marlin_probe_block_scales_exact(k_tiles: int) -> None:
    require_sm120()
    device = torch.device("cuda")
    a, b = _dyadic_inputs(k_tiles, seed=20 + k_tiles)
    kg = k_tiles * 4
    sfa_exp = torch.randint(-3, 4, (_M, kg), device=device)
    sfb_exp = torch.randint(-3, 4, (_N, kg), device=device)
    out = _run_probe(
        _e4m3_bytes(a),
        (sfa_exp + 127).to(torch.uint8),
        _pack_b(b),
        (sfb_exp + 127).to(torch.uint8),
        k_tiles,
    )
    a_eff = (a.view(_M, kg, 32) * torch.exp2(sfa_exp.float()).unsqueeze(-1)).view(_M, -1)
    b_eff = (b.view(_N, kg, 32) * torch.exp2(sfb_exp.float()).unsqueeze(-1)).view(_N, -1)
    torch.testing.assert_close(out, a_eff @ b_eff.T, atol=0.0, rtol=0.0)
