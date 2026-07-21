"""Weight-only-quantized batched GEMM for MLA absorbed projections (sm120).

Implements AGENT_G_QBMM_DESIGN.md sections 5.1-5.5: decode-path MLA consumes the
absorbed projections directly from the kv_b mxfp8 pack (fp8 e4m3 values +
per-32-in-axis e8m0 scales), replacing the resident BF16 ``W_UK_T``/``W_UV``
pair.  One kernel template, two orientations:

  op "ukt" (site mla_attention.py:977):
      out[n, b, l] = sum_p A[n, b, p] * deq(pack[448*n + p,        l])
      GEMM per head: M=B, N=latent(512), K=p(192).  The weight tile is read in
      its native row-major orientation and fed to the MMA *transposed*
      (contraction axis p = pack rows != scale axis l = pack cols, so the
      dequant multiplier varies per K element and is constant along 32-wide
      windows of the GEMM N axis).

  op "uv" (site mla_attention.py:1521):
      out[n, b, v] = sum_l A[n, b, l] * deq(pack[448*n + 192 + v,  l])
      GEMM per head: M=B, N=v(256), K=latent(512).  Layout-native: contraction
      axis == scale axis, classic [N, K]-row-major weight with e8m0_k32 groups.

  deq(pack[r, c]) = bf16(e4m3 value) * bf16(2^(scale[r, c//32] - 127))
                  computed as one bf16 multiply -- bitwise identical to the
                  torch reference:
                  w.to(bf16) * s.view(float8_e8m0fnu).to(bf16)
                                .repeat_interleave(32, dim=1).
                  The e8m0 view is load-bearing: scale tensors are stored as
                  uint8, and a raw uint8 .to(bf16) converts the exponent BYTE
                  as an integer (127 -> 127.0, not 2^0) -- a silent x1e2-1e12
                  dequant catastrophe.

Numerics: fp8->bf16 conversion is exact (e4m3 is a subset of bf16, via the
hardware cvt e4m3x2->f16x2->f32->bf16 chain, all steps exact); e8m0->bf16 is
exact for every exponent byte except 255 (NaN, impossible for weights derived
from finite bf16); the value*scale product is a single RN bf16 multiply, the
same operation torch performs.  NOTE: the existing b12x fp4-path dequant
primitives (packed_dequant_e4m3x4_to_bfloat2x2 / packed_dequant_e8m0x4_to_
bfloat2x2 in b12x/cute/fp4.py) are *biased* variants (scale-only sign
embedding, 2^7 scale bias compensated in a global-scale epilogue) and cannot
satisfy the bitwise contract, so this module carries its own exact-value
conversions modelled on the same inline-PTX idiom.

MMA: bf16 HMMA m16n8k16, fp32 accumulators, K unsplit (deterministic, no
atomics, replay-stable).  Grid = (head, n_tile) -- one wave.  No workspace, no
allocation on the launch path, out= only.

Compile keys: (op, B, tile_cfg, sm_count) -- B is a compile-time constant per
captured graph size, tile_m = 16 (B <= 16) or 32 (B <= 32).

Torch custom ops (with fake impls for torch.compile tracing):
    torch.ops.b12x.qbmm_absorb_ukt(a, pack_values, pack_scales, out, stream_int)
    torch.ops.b12x.qbmm_absorb_uv (a, pack_values, pack_scales, out, stream_int)

Warmup (jit-cache value-bake law: all compiles must happen before capture --
covers the target's capture sizes AND the speculator's prefill-capture batch;
a compile miss during capture raises rather than corrupting the graph):
    warmup_qbmm_absorb(pack_values, pack_scales,
                       batch_sizes=(1, 2, 4, 8, 16, 25, 32))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass import Float32, Uint32

from b12x.cute.compiler import KernelCompileSpec, compile as b12x_compile
from b12x.cute.fp4 import (
    bfloat2_mul,
    bf16_mma_m16n8k16_f32,
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    ld_global_b16,
    ld_shared_u8_offset,
    ldmatrix_m8n8x4_b16,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_u16,
    st_shared_v4_u32,
)
from b12x.cute.utils import current_cuda_stream, get_num_sm, make_ptr

# ---------------------------------------------------------------------------
# Fixed v1 tile configuration (design 5.5): tiny-M tiles, 2-3 stage cp.async
# pipeline.  tile_m tracks B (16 for B<=16, 32 for B<=32); tile_n=128,
# tile_k=64 give whole tiles for every op dimension (192, 256, 512 are all
# multiples of 64/128) -- no tail handling anywhere.
# ---------------------------------------------------------------------------
_TILE_N = 128
_TILE_K = 64
_STAGES = 3
_CTA_THREADS = 128  # 4 warps; each warp owns a 32-wide N slice of the tile.
_MAX_BATCH = 32
# Every batch size a CUDA-graph capture can replay must compile BEFORE any
# capture starts (jit-cache value-bake law).  That is the target's capture
# list (1, 2, 4, 8, 32 on the serving config) PLUS the speculator's
# prefill-capture batch: num_tokens - num_reqs + 1 = 25 for 32-token graphs
# at 8 seqs / k=3.  25 was missing on 2026-07-21: the draft's kernels
# JIT-compiled inside the speculator capture and the captured graph replayed
# garbage -- MTP acceptance 0.000 flat while the (fully warmed) target
# stayed healthy.  16 = tile_m boundary insurance.  A compile miss during
# capture now raises instead of corrupting silently (see _compile_qbmm).
_DEFAULT_WARMUP_BATCH_SIZES = (1, 2, 4, 8, 16, 25, 32)

_QBMM_KERNEL_ID = "gemm.qbmm_absorb"
_QBMM_KERNEL_VERSION = 1


# ---------------------------------------------------------------------------
# Exact-value device conversions (this module's own primitives).
# ---------------------------------------------------------------------------


@dsl_user_op
def ld_shared_u16_zx(smem_addr: Int32, *, loc=None, ip=None) -> Uint32:
    """Load a shared halfword zero-extended into a 32-bit register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u16 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def packed_cvt_e4m3x4_to_bfloat2x2_exact(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Exact e4m3 -> bf16 for 4 packed bytes, sequential pairing.

    Returns (lo, hi): lo = bf16x2(byte0, byte1), hi = bf16x2(byte2, byte3),
    with byte N in the low half of its pair.  Chain: cvt.rn.f16x2.e4m3x2
    (exact: every e4m3 value incl. subnormals and +-0 is representable in
    f16) -> cvt.f32.f16 (exact) -> cvt.rn.bf16x2.f32 (exact for e4m3-derived
    values: 4-bit mantissa fits bf16's 8).  This is the true-value conversion
    torch's ``.to(torch.bfloat16)`` performs, unlike the biased fp4-path
    dequant helpers.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e01, e23, h0, h1, h2, h3;
            .reg .b32 p01, p23;
            .reg .f32 f0, f1, f2, f3;
            mov.b32 {e01, e23}, $2;
            cvt.rn.f16x2.e4m3x2 p01, e01;
            cvt.rn.f16x2.e4m3x2 p23, e23;
            mov.b32 {h0, h1}, p01;
            mov.b32 {h2, h3}, p23;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_cvt_e8m0x4_to_bfloat2x2_exact(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Exact e8m0 -> bf16 for 4 packed bytes, sequential pairing.

    byte e -> bf16(2^(e-127)): bit pattern e<<7 (exponent field = e, mantissa
    0) for e in [1, 254]; e == 0 -> the exact bf16 subnormal 2^-127 (0x0040).
    e == 255 (e8m0 NaN) maps to +Inf, not NaN -- impossible for weight scales
    derived from finite bf16 (design risk #8); asserted away in the unit
    harness.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p0, p1, p2, p3;
            .reg .b32 b0, b1, b2, b3, h0, h1, h2, h3, t;
            and.b32 b0, $2, 0x000000ff;
            shr.u32 b1, $2, 8;
            and.b32 b1, b1, 0x000000ff;
            shr.u32 b2, $2, 16;
            and.b32 b2, b2, 0x000000ff;
            shr.u32 b3, $2, 24;
            shl.b32 h0, b0, 7;
            shl.b32 h1, b1, 7;
            shl.b32 h2, b2, 7;
            shl.b32 h3, b3, 7;
            setp.eq.u32 p0, b0, 0;
            setp.eq.u32 p1, b1, 0;
            setp.eq.u32 p2, b2, 0;
            setp.eq.u32 p3, b3, 0;
            @p0 mov.b32 h0, 0x00000040;
            @p1 mov.b32 h1, 0x00000040;
            @p2 mov.b32 h2, 0x00000040;
            @p3 mov.b32 h3, 0x00000040;
            shl.b32 t, h1, 16;
            or.b32 $0, h0, t;
            shl.b32 t, h3, 16;
            or.b32 $1, h2, t;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def cvt_e8m0_byte_to_bfloat2_exact(byte: Uint32, *, loc=None, ip=None) -> Uint32:
    """One e8m0 byte -> bf16(2^(e-127)) broadcast into both bf16x2 lanes."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(byte).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred pz;
                .reg .b32 e, t, s;
                and.b32 e, $1, 0x000000ff;
                shl.b32 t, e, 7;
                setp.eq.u32 pz, e, 0;
                @pz mov.b32 t, 0x00000040;
                shl.b32 s, t, 16;
                or.b32 $0, s, t;
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_f32x2_to_bfloat2_rn(
    x0: Float32, x1: Float32, *, loc=None, ip=None
) -> Uint32:
    """Pack 2 f32 into bf16x2 with plain RN (no satfinite), x0 in the low half.

    Matches the cuBLAS/torch epilogue conversion semantics (overflow -> Inf),
    unlike the existing satfinite packer.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# The kernel template.
# ---------------------------------------------------------------------------


class QbmmAbsorbKernel:
    """One CTA per (head, n_tile); 128 threads; 3-stage cp.async pipeline.

    Shared memory per stage (16B-aligned regions, no dynamic smem):
      [B tile: 8 KiB fp8 bytes] [A tile: tile_m*128 B bf16] [scales: 256 B]
    B-tile staging keeps the pack's native row-major orientation; the "ukt"
    orientation differs from "uv" only in which pack axis is the smem row and
    in per-fragment scale indexing (design section 3: the scale-axis asymmetry
    collapses into an MMA-feed orientation flag, not a data transform).
    """

    def __init__(
        self,
        *,
        op: str,
        batch: int,
        num_heads: int = 16,
        head_stride: int = 448,
        p_dim: int = 192,
        v_dim: int = 256,
        latent_dim: int = 512,
        tile_n: int = _TILE_N,
        tile_k: int = _TILE_K,
        stages: int = _STAGES,
        sm_count: int = 0,
    ):
        if op not in ("ukt", "uv"):
            raise ValueError(f"op must be 'ukt' or 'uv', got {op!r}")
        if not (1 <= int(batch) <= _MAX_BATCH):
            raise ValueError(
                f"qbmm_absorb v1 supports 1 <= B <= {_MAX_BATCH}, got {batch}"
            )
        if head_stride != p_dim + v_dim:
            raise ValueError("head_stride must equal p_dim + v_dim")
        if tile_n != 128 or tile_k != 64:
            raise ValueError("v1 tile config is fixed at tile_n=128, tile_k=64")
        if stages not in (2, 3):
            raise ValueError("stages must be 2 or 3")
        if latent_dim % 32 != 0:
            raise ValueError("latent_dim must be a multiple of 32 (e8m0 groups)")

        self.op = str(op)
        self.is_ukt = op == "ukt"
        self.batch = int(batch)
        self.num_heads = int(num_heads)
        self.head_stride = int(head_stride)
        self.p_dim = int(p_dim)
        self.v_dim = int(v_dim)
        self.latent_dim = int(latent_dim)
        self.tile_n = int(tile_n)
        self.tile_k = int(tile_k)
        self.stages = int(stages)
        self.sm_count = int(sm_count)
        self.cta_threads = _CTA_THREADS

        # GEMM geometry per head.
        if self.is_ukt:
            self.gemm_n = self.latent_dim  # l
            self.gemm_k = self.p_dim  # p
            self.head_row_base = 0  # pack rows [448n, 448n+192) are k_nope
        else:
            self.gemm_n = self.v_dim  # v
            self.gemm_k = self.latent_dim  # l
            self.head_row_base = self.p_dim  # pack rows [448n+192, 448n+448)

        if self.gemm_n % self.tile_n != 0:
            raise ValueError(f"gemm_n {self.gemm_n} not divisible by tile_n")
        if self.gemm_k % self.tile_k != 0:
            raise ValueError(f"gemm_k {self.gemm_k} not divisible by tile_k")

        self.n_tiles = self.gemm_n // self.tile_n
        self.k_tiles = self.gemm_k // self.tile_k
        self.k_steps = self.tile_k // 16  # MMA k16 steps per k-tile
        if self.stages - 1 > self.k_tiles:
            raise ValueError("stages-1 must not exceed k_tiles")
        self.grid_x = self.num_heads * self.n_tiles

        self.tile_m = 16 if self.batch <= 16 else 32
        self.m_blocks = self.tile_m // 16
        self.n_warps = self.cta_threads // 32  # 4: each owns 32 N columns
        if self.n_warps * 32 != self.tile_n:
            raise ValueError("tile_n must equal 32 * n_warps")

        # Weight tile smem geometry (native pack orientation, bytes):
        #   uv : rows = tile_n (v rows),   row_bytes = tile_k (l window)
        #   ukt: rows = tile_k (p k-rows), row_bytes = tile_n (l window)
        self.b_rows = self.tile_n if not self.is_ukt else self.tile_k
        self.b_row_bytes = self.tile_k if not self.is_ukt else self.tile_n
        self.b_units_per_row = self.b_row_bytes // 16  # int4 units
        self.b_swz_mask = min(self.b_units_per_row, 8) - 1
        self.b_stage_bytes = self.b_rows * self.b_row_bytes  # 8192 either way
        self.b_units = self.b_stage_bytes // 16  # 512

        # A tile: tile_m rows x tile_k bf16 cols = tile_m x 128 bytes.
        self.a_units_per_row = (self.tile_k * 2) // 16  # 8
        self.a_swz_mask = min(self.a_units_per_row, 8) - 1
        self.a_stage_bytes = self.tile_m * self.tile_k * 2
        self.a_units = self.a_stage_bytes // 16

        # Scale region: uv = tile_n rows x (tile_k/32) bytes = 128*2;
        #               ukt = tile_k rows x (tile_n/32) bytes = 64*4.
        if not self.is_ukt:
            self.s_rows = self.tile_n
            self.s_row_bytes = self.tile_k // 32  # 2
        else:
            self.s_rows = self.tile_k
            self.s_row_bytes = self.tile_n // 32  # 4
        self.s_stage_bytes = self.s_rows * self.s_row_bytes  # 256 either way

        self.b_off = 0
        self.a_off = self.b_stage_bytes
        self.s_off = self.b_stage_bytes + self.a_stage_bytes
        stage_bytes = self.b_stage_bytes + self.a_stage_bytes + self.s_stage_bytes
        self.stage_bytes = (stage_bytes + 15) // 16 * 16
        self.smem_bytes = self.stages * self.stage_bytes
        if self.smem_bytes > 48 * 1024:
            raise ValueError(f"smem {self.smem_bytes} exceeds 48 KiB static limit")
        self.smem_words = self.smem_bytes // 4

        self.pack_rows = self.num_heads * self.head_stride
        self.scale_cols = self.latent_dim // 32

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.op,
            self.batch,
            self.num_heads,
            self.head_stride,
            self.p_dim,
            self.v_dim,
            self.latent_dim,
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.stages,
            self.cta_threads,
            self.sm_count,
        )

    # -- host-side launch entry (compiled once per cache key) ---------------

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,  # bf16 activations (num_heads, B, gemm_k) strided
        w_ptr: cute.Pointer,  # u8 view of pack values [pack_rows, latent_dim]
        s_ptr: cute.Pointer,  # u8 view of pack scales [pack_rows, latent_dim/32]
        c_ptr: cute.Pointer,  # bf16 out (num_heads, B, gemm_n) strided
        a_stride_head: cutlass.Int32,  # elements
        a_stride_b: cutlass.Int32,  # elements
        c_stride_head: cutlass.Int32,  # elements
        c_stride_b: cutlass.Int32,  # elements
        stream: cuda.CUstream,
    ):
        a_span = (
            Int32(self.num_heads - 1) * a_stride_head
            + Int32(self.batch - 1) * a_stride_b
            + Int32(self.gemm_k)
        )
        c_span = (
            Int32(self.num_heads - 1) * c_stride_head
            + Int32(self.batch - 1) * c_stride_b
            + Int32(self.gemm_n)
        )
        a_flat = cute.make_tensor(
            a_ptr, layout=cute.make_layout((a_span,), stride=(1,))
        )
        c_flat = cute.make_tensor(
            c_ptr, layout=cute.make_layout((c_span,), stride=(1,))
        )
        w_flat = cute.make_tensor(
            w_ptr,
            layout=cute.make_layout(
                (self.pack_rows * self.latent_dim,), stride=(1,)
            ),
        )
        s_flat = cute.make_tensor(
            s_ptr,
            layout=cute.make_layout(
                (self.pack_rows * self.scale_cols,), stride=(1,)
            ),
        )
        self.kernel(
            a_flat,
            w_flat,
            s_flat,
            c_flat,
            a_stride_head,
            a_stride_b,
            c_stride_head,
            c_stride_b,
        ).launch(
            grid=(self.grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    # -- device code --------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        a_flat: cute.Tensor,
        w_flat: cute.Tensor,
        s_flat: cute.Tensor,
        c_flat: cute.Tensor,
        a_stride_head: cutlass.Int32,
        a_stride_b: cutlass.Int32,
        c_stride_head: cutlass.Int32,
        c_stride_b: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.smem_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        head = cta // Int32(self.n_tiles)
        jt = cta - head * Int32(self.n_tiles)

        # Pack row base for this head's weight slab (rows are e4m3 bytes wide
        # latent_dim; scale rows are latent_dim/32 bytes wide).
        w_row0 = head * Int32(self.head_stride) + Int32(self.head_row_base)

        acc = cute.make_rmem_tensor((self.m_blocks, 4, 4), cutlass.Float32)
        acc.fill(0.0)
        a_regs = cute.make_rmem_tensor((self.m_blocks, 4), Uint32)

        # Pipeline prologue: fill stages-1 buffers (one commit per stage).
        for s in cutlass.range_constexpr(self.stages - 1):
            self._stage_tile(
                a_flat,
                w_flat,
                s_flat,
                smem_base,
                tid,
                head,
                jt,
                w_row0,
                Int32(s),
                Int32(s),
                a_stride_head,
                a_stride_b,
            )

        # Main loop: [wait][sync][consume][stage next or empty commit].
        # Exactly one commit per iteration keeps wait_group(stages-2) exact:
        # before the wait at iteration kt, stages-1+kt groups have been
        # committed and group kt must have landed once <= stages-2 remain.
        kt = Int32(0)
        while kt < Int32(self.k_tiles):
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
            slot = kt % Int32(self.stages)
            self._consume_tile(smem_base, tid, slot, acc, a_regs)
            nkt = kt + Int32(self.stages - 1)
            if nkt < Int32(self.k_tiles):
                self._stage_tile(
                    a_flat,
                    w_flat,
                    s_flat,
                    smem_base,
                    tid,
                    head,
                    jt,
                    w_row0,
                    nkt,
                    nkt % Int32(self.stages),
                    a_stride_head,
                    a_stride_b,
                )
            else:
                cute.arch.cp_async_commit_group()
            kt += Int32(1)

        self._store_output(
            c_flat, tid, head, jt, acc, c_stride_head, c_stride_b
        )

    @cute.jit
    def _stage_tile(
        self,
        a_flat: cute.Tensor,
        w_flat: cute.Tensor,
        s_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        head: Int32,
        jt: Int32,
        w_row0: Int32,
        kt: Int32,
        slot: Int32,
        a_stride_head: Int32,
        a_stride_b: Int32,
    ):
        """Async-copy one k-tile of B (native pack orientation), A, scales.

        Ends with cp_async_commit_group() -- exactly one group per call.
        XOR swizzles (unit_col ^= row & mask) are applied identically on the
        store side here and the read side in _consume_tile, so they are
        correctness-neutral by construction.
        """
        st_base = smem_base + slot * Int32(self.stage_bytes)

        # ---- B tile: 512 int4 units, 4 per thread. ----
        for i in cutlass.range_constexpr(self.b_units // self.cta_threads):
            u = Int32(i * self.cta_threads) + tid
            row = u // Int32(self.b_units_per_row)
            col = u - row * Int32(self.b_units_per_row)
            col_sw = col ^ (row & Int32(self.b_swz_mask))
            dst = (
                st_base
                + Int32(self.b_off)
                + (row * Int32(self.b_units_per_row) + col_sw) * Int32(16)
            )
            if cutlass.const_expr(not self.is_ukt):
                # uv: smem row = pack row (v), bytes = l window of this k-tile.
                src = (w_row0 + jt * Int32(self.tile_n) + row) * Int32(
                    self.latent_dim
                ) + kt * Int32(self.tile_k) + col * Int32(16)
            else:
                # ukt: smem row = pack row (p = GEMM k), bytes = l (GEMM n).
                src = (w_row0 + kt * Int32(self.tile_k) + row) * Int32(
                    self.latent_dim
                ) + jt * Int32(self.tile_n) + col * Int32(16)
            cp_async4_shared_global(dst, get_ptr_as_int64(w_flat, src))

        # ---- A tile: tile_m rows x tile_k bf16 = a_units int4 units. ----
        for i in cutlass.range_constexpr(self.a_units // self.cta_threads):
            u = Int32(i * self.cta_threads) + tid
            row = u // Int32(self.a_units_per_row)
            col = u - row * Int32(self.a_units_per_row)
            col_sw = col ^ (row & Int32(self.a_swz_mask))
            dst = (
                st_base
                + Int32(self.a_off)
                + (row * Int32(self.a_units_per_row) + col_sw) * Int32(16)
            )
            if cutlass.const_expr(self.batch < self.tile_m):
                if row < Int32(self.batch):
                    src = (
                        head * a_stride_head
                        + row * a_stride_b
                        + kt * Int32(self.tile_k)
                        + col * Int32(8)
                    )
                    cp_async4_shared_global(dst, get_ptr_as_int64(a_flat, src))
                else:
                    st_shared_v4_u32(
                        dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    )
            else:
                src = (
                    head * a_stride_head
                    + row * a_stride_b
                    + kt * Int32(self.tile_k)
                    + col * Int32(8)
                )
                cp_async4_shared_global(dst, get_ptr_as_int64(a_flat, src))

        # ---- Scales. ----
        if cutlass.const_expr(not self.is_ukt):
            # uv: one u16 per pack row = e8m0 groups (2*kt, 2*kt+1).
            # 2-byte granularity is below cp.async's 4B minimum, so this is a
            # synchronous ld.global + st.shared (visibility ordered by the
            # consumer-side syncthreads).
            if tid < Int32(self.s_rows):
                src = (
                    w_row0 + jt * Int32(self.tile_n) + tid
                ) * Int32(self.scale_cols) + kt * Int32(self.tile_k // 32)
                v = ld_global_b16(get_ptr_as_int64(s_flat, src))
                st_shared_u16(st_base + Int32(self.s_off) + tid * Int32(2), v)
        else:
            # ukt: one u32 per pack k-row = e8m0 groups jt*4 + [0, 4).
            if tid < Int32(self.s_rows):
                src = (
                    w_row0 + kt * Int32(self.tile_k) + tid
                ) * Int32(self.scale_cols) + jt * Int32(self.tile_n // 32)
                cp_async_u32_shared_global(
                    st_base + Int32(self.s_off) + tid * Int32(4),
                    get_ptr_as_int64(s_flat, src),
                )

        cute.arch.cp_async_commit_group()

    @cute.jit
    def _consume_tile(
        self,
        smem_base: Int32,
        tid: Int32,
        slot: Int32,
        acc: cute.Tensor,
        a_regs: cute.Tensor,
    ):
        """Dequant + MMA over one staged k-tile (4 k16 steps).

        m16n8k16 fragment mapping (canonical PTX, lane = 4*g + t, g = lane>>2,
        t = lane&3):
          A: a0=(row g, k 2t..2t+1) a1=(g+8, same) a2=(g, k+8) a3=(g+8, k+8)
          B: b0=(k 2t..2t+1, n g)   b1=(k 2t+8..2t+9, n g)
          C: c0=(g, 2t) c1=(g, 2t+1) c2=(g+8, 2t) c3=(g+8, 2t+1)
        Each warp owns N columns [warp*32, warp*32+32) of the CTA tile as 4
        jj-blocks of 8.
        """
        st_base = smem_base + slot * Int32(self.stage_bytes)
        lane = tid & Int32(31)
        warp = tid // Int32(32)
        g = lane // Int32(4)
        t = lane - g * Int32(4)
        t2 = t * Int32(2)

        a_base = st_base + Int32(self.a_off)
        b_base = st_base + Int32(self.b_off)
        s_base = st_base + Int32(self.s_off)

        # ldmatrix lane addressing: row = lane&15 (+16 per m-block), int4
        # column half = lane>>4, same xor swizzle as the store side.
        lm_row_in_blk = lane & Int32(15)
        lm_half = lane // Int32(16)

        for kk in cutlass.range_constexpr(self.k_steps):
            # A fragments for every m-block (shared across the 4 jj blocks).
            for mb in cutlass.range_constexpr(self.m_blocks):
                lm_row = Int32(16 * mb) + lm_row_in_blk
                lm_col = Int32(2 * kk) + lm_half
                lm_col_sw = lm_col ^ (lm_row & Int32(self.a_swz_mask))
                a_addr = a_base + (
                    lm_row * Int32(self.a_units_per_row) + lm_col_sw
                ) * Int32(16)
                r0, r1, r2, r3 = ldmatrix_m8n8x4_b16(a_addr)
                a_regs[mb, 0] = r0
                a_regs[mb, 1] = r1
                a_regs[mb, 2] = r2
                a_regs[mb, 3] = r3

            if cutlass.const_expr(self.is_ukt):
                # Per-K-row scales, shared by all jj (l-group = this warp's
                # 32-wide N window).  k rows: kk*16 + {2t, 2t+1, 2t+8, 2t+9}.
                kd0 = Int32(16 * kk) + t2
                s0 = ld_shared_u8_offset(
                    s_base + kd0 * Int32(self.s_row_bytes) + warp, 0
                )
                s1 = ld_shared_u8_offset(
                    s_base + (kd0 + Int32(1)) * Int32(self.s_row_bytes) + warp, 0
                )
                s2 = ld_shared_u8_offset(
                    s_base + (kd0 + Int32(8)) * Int32(self.s_row_bytes) + warp, 0
                )
                s3 = ld_shared_u8_offset(
                    s_base + (kd0 + Int32(9)) * Int32(self.s_row_bytes) + warp, 0
                )
                s_word = s0 + (s1 << Uint32(8)) + (s2 << Uint32(16)) + (
                    s3 << Uint32(24)
                )
                sc_lo, sc_hi = packed_cvt_e8m0x4_to_bfloat2x2_exact(s_word)

            for jj in cutlass.range_constexpr(4):
                n_local = warp * Int32(32) + Int32(8 * jj) + g

                if cutlass.const_expr(not self.is_ukt):
                    # uv values: pack row = n_local, k bytes contiguous.
                    unit = Int32(kk) ^ (n_local & Int32(self.b_swz_mask))
                    v_addr = b_base + (
                        n_local * Int32(self.b_units_per_row) + unit
                    ) * Int32(16) + t2
                    v_lo16 = ld_shared_u16_zx(v_addr)
                    v_hi16 = ld_shared_u16_zx(v_addr + Int32(8))
                    v_word = v_lo16 + (v_hi16 << Uint32(16))
                    v_lo, v_hi = packed_cvt_e4m3x4_to_bfloat2x2_exact(v_word)
                    # One e8m0 group covers the whole k16 step: byte kk>>1 of
                    # this row's staged u16.
                    s_byte = ld_shared_u8_offset(
                        s_base + n_local * Int32(2) + Int32(kk // 2), 0
                    )
                    sc = cvt_e8m0_byte_to_bfloat2_exact(s_byte)
                    b0 = bfloat2_mul(v_lo, sc)
                    b1 = bfloat2_mul(v_hi, sc)
                else:
                    # ukt values: k rows kk*16 + {2t,2t+1,2t+8,2t+9} (= kd0
                    # from the scale hoist above), byte column n_local;
                    # per-row xor swizzle on the int4 unit.
                    n_unit = n_local // Int32(16)
                    n_in_unit = n_local & Int32(15)
                    b0v = self._ukt_value_byte(b_base, kd0, n_unit, n_in_unit)
                    b1v = self._ukt_value_byte(
                        b_base, kd0 + Int32(1), n_unit, n_in_unit
                    )
                    b2v = self._ukt_value_byte(
                        b_base, kd0 + Int32(8), n_unit, n_in_unit
                    )
                    b3v = self._ukt_value_byte(
                        b_base, kd0 + Int32(9), n_unit, n_in_unit
                    )
                    v_word = b0v + (b1v << Uint32(8)) + (b2v << Uint32(16)) + (
                        b3v << Uint32(24)
                    )
                    v_lo, v_hi = packed_cvt_e4m3x4_to_bfloat2x2_exact(v_word)
                    b0 = bfloat2_mul(v_lo, sc_lo)
                    b1 = bfloat2_mul(v_hi, sc_hi)

                for mb in cutlass.range_constexpr(self.m_blocks):
                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                        acc[mb, jj, 0],
                        acc[mb, jj, 1],
                        acc[mb, jj, 2],
                        acc[mb, jj, 3],
                        a_regs[mb, 0],
                        a_regs[mb, 1],
                        a_regs[mb, 2],
                        a_regs[mb, 3],
                        b0,
                        b1,
                    )
                    acc[mb, jj, 0] = d0
                    acc[mb, jj, 1] = d1
                    acc[mb, jj, 2] = d2
                    acc[mb, jj, 3] = d3

    @cute.jit
    def _ukt_value_byte(
        self, b_base: Int32, kd: Int32, n_unit: Int32, n_in_unit: Int32
    ) -> Uint32:
        unit_sw = n_unit ^ (kd & Int32(self.b_swz_mask))
        addr = b_base + (
            kd * Int32(self.b_units_per_row) + unit_sw
        ) * Int32(16) + n_in_unit
        return ld_shared_u8_offset(addr, 0)

    @cute.jit
    def _store_output(
        self,
        c_flat: cute.Tensor,
        tid: Int32,
        head: Int32,
        jt: Int32,
        acc: cute.Tensor,
        c_stride_head: Int32,
        c_stride_b: Int32,
    ):
        """f32 acc -> bf16 pairs -> u32 stores into the strided C view.

        C rows (batch) beyond B are never written; with the compile-time B the
        predicates fold away for full tiles.  Column pairs are contiguous (C
        inner dim contiguous by contract) so every store is one aligned u32.
        """
        lane = tid & Int32(31)
        warp = tid // Int32(32)
        g = lane // Int32(4)
        t = lane - g * Int32(4)
        col0 = jt * Int32(self.tile_n) + warp * Int32(32) + t * Int32(2)
        c_head = head * c_stride_head

        for mb in cutlass.range_constexpr(self.m_blocks):
            row0 = Int32(16 * mb) + g
            row1 = row0 + Int32(8)
            for jj in cutlass.range_constexpr(4):
                col = col0 + Int32(8 * jj)
                if row0 < Int32(self.batch):
                    off = c_head + row0 * c_stride_b + col
                    st_global_u32(
                        get_ptr_as_int64(c_flat, off),
                        pack_f32x2_to_bfloat2_rn(acc[mb, jj, 0], acc[mb, jj, 1]),
                    )
                if row1 < Int32(self.batch):
                    off = c_head + row1 * c_stride_b + col
                    st_global_u32(
                        get_ptr_as_int64(c_flat, off),
                        pack_f32x2_to_bfloat2_rn(acc[mb, jj, 2], acc[mb, jj, 3]),
                    )


# ---------------------------------------------------------------------------
# Host-side compile cache + launch.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QbmmLaunch:
    compiled: object
    kernel: QbmmAbsorbKernel


_QBMM_CACHE: dict = {}


def _tile_m_for_batch(batch: int) -> int:
    return 16 if int(batch) <= 16 else 32


def _compile_qbmm(
    *,
    op: str,
    batch: int,
    num_heads: int,
    head_stride: int,
    p_dim: int,
    v_dim: int,
    latent_dim: int,
    device: torch.device,
) -> _QbmmLaunch:
    device_index = int(device.index if device.index is not None else 0)
    sm_count = get_num_sm(device)
    tile_m = _tile_m_for_batch(batch)
    cache_key = (
        _QBMM_KERNEL_ID,
        op,
        int(batch),
        device_index,
        (tile_m, _TILE_N, _TILE_K, _STAGES, _CTA_THREADS),
        int(sm_count),
        (num_heads, head_stride, p_dim, v_dim, latent_dim),
    )
    cached = _QBMM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"qbmm_absorb compile miss for op={op!r} batch={batch} during "
            "CUDA-graph capture: a graph captured around a JIT compile "
            "replays corrupt output (the 0.000-acceptance failure mode). "
            "Add this batch size to warmup_qbmm_absorb(batch_sizes=...) so "
            "it compiles before capture."
        )

    kernel = QbmmAbsorbKernel(
        op=op,
        batch=batch,
        num_heads=num_heads,
        head_stride=head_stride,
        p_dim=p_dim,
        v_dim=v_dim,
        latent_dim=latent_dim,
        tile_n=_TILE_N,
        tile_k=_TILE_K,
        stages=_STAGES,
        sm_count=sm_count,
    )

    def dummy(dt, align=16):
        return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=align)

    # b12x_compile guards frozen resolution on true cache misses itself
    # (memory/disk hits bypass the freeze -- warm boots stay legal).
    # Pointer assumed_align is part of the compiled signature: A/pack are
    # 16B (cp.async4 sources), C is 4B (u32 stores only; permits row-sliced
    # output views).
    compiled = b12x_compile(
        kernel,
        dummy(cutlass.BFloat16),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.BFloat16, align=4),
        Int32(kernel.gemm_k),
        Int32(kernel.gemm_k),
        Int32(kernel.gemm_n),
        Int32(kernel.gemm_n),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_facts(
            _QBMM_KERNEL_ID,
            _QBMM_KERNEL_VERSION,
            ("device_index", device_index),
            ("op", str(op)),
            ("batch", int(batch)),
            ("tile_m", int(tile_m)),
            ("tile_n", int(_TILE_N)),
            ("tile_k", int(_TILE_K)),
            ("stages", int(_STAGES)),
            ("cta_threads", int(_CTA_THREADS)),
            ("num_heads", int(num_heads)),
            ("head_stride", int(head_stride)),
            ("p_dim", int(p_dim)),
            ("v_dim", int(v_dim)),
            ("latent_dim", int(latent_dim)),
            ("sm_count", int(sm_count)),
            ("grid_x", int(kernel.grid_x)),
        ),
    )
    launch = _QbmmLaunch(compiled=compiled, kernel=kernel)
    _QBMM_CACHE[cache_key] = launch
    return launch


def _check_operand(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    stride_mod: int,
    ptr_align: int,
) -> None:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.stride(2) != 1:
        raise ValueError(f"{name} inner dim must be contiguous")
    if tensor.stride(0) % stride_mod or tensor.stride(1) % stride_mod:
        raise ValueError(
            f"{name} outer strides {tensor.stride(0)}/{tensor.stride(1)} must be "
            f"multiples of {stride_mod} elements"
        )
    if tensor.data_ptr() % ptr_align:
        raise ValueError(f"{name} base pointer must be {ptr_align}-byte aligned")


def qbmm_absorb(
    a: torch.Tensor,
    pack_values: torch.Tensor,
    pack_scales: torch.Tensor,
    out: torch.Tensor,
    *,
    op: Literal["ukt", "uv"],
    num_heads: int = 16,
    head_stride: int = 448,
    p_dim: int = 192,
    v_dim: int = 256,
    latent_dim: int = 512,
    stream: Optional[int] = None,
) -> None:
    """Shared low-level entry (design 5.1).  No allocations; writes ``out``.

    a           (num_heads, B, K) bf16, arbitrary outer strides (multiples of
                8 elements), inner dim contiguous.  K = p_dim for "ukt",
                latent_dim for "uv".
    pack_values kv_b_proj.weight        [num_heads*head_stride, latent_dim]
                float8_e4m3fn, contiguous (aliases the mxfp8 pack).
    pack_scales kv_b_proj.weight_scale  [num_heads*head_stride, latent_dim/32]
                e8m0 (or its uint8 view), contiguous.
    out         "ukt": (num_heads, B, latent_dim) dense or padded-head view;
                "uv":  (num_heads, B, v_dim) strided view of (B, num_heads,
                v_dim)-contiguous storage (``out.transpose(0, 1)`` at the call
                site).  Inner dim contiguous, outer strides even.
    """
    if op not in ("ukt", "uv"):
        raise ValueError(f"op must be 'ukt' or 'uv', got {op!r}")
    if a.ndim != 3:
        raise ValueError(f"a must be 3-D (num_heads, B, K), got {a.ndim}-D")
    batch = int(a.shape[1])
    if not (1 <= batch <= _MAX_BATCH):
        raise ValueError(
            f"qbmm_absorb v1 supports 1 <= B <= {_MAX_BATCH}, got {batch} "
            "(fall back to the absorb-fly path for larger eager batches)"
        )
    gemm_k = p_dim if op == "ukt" else latent_dim
    gemm_n = latent_dim if op == "ukt" else v_dim
    _check_operand(
        "a", a, shape=(num_heads, batch, gemm_k), stride_mod=8, ptr_align=16
    )
    _check_operand(
        "out", out, shape=(num_heads, batch, gemm_n), stride_mod=2, ptr_align=4
    )

    pack_rows = num_heads * head_stride
    if pack_values.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"pack_values must be float8_e4m3fn, got {pack_values.dtype}"
        )
    if tuple(pack_values.shape) != (pack_rows, latent_dim):
        raise ValueError(
            f"pack_values must be [{pack_rows}, {latent_dim}], "
            f"got {tuple(pack_values.shape)}"
        )
    if not pack_values.is_contiguous():
        raise ValueError("pack_values must be contiguous")
    if pack_scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError(
            f"pack_scales must be uint8/e8m0, got {pack_scales.dtype}"
        )
    if tuple(pack_scales.shape) != (pack_rows, latent_dim // 32):
        raise ValueError(
            f"pack_scales must be [{pack_rows}, {latent_dim // 32}], "
            f"got {tuple(pack_scales.shape)}"
        )
    if not pack_scales.is_contiguous():
        raise ValueError("pack_scales must be contiguous")
    if not (a.is_cuda and out.is_cuda and pack_values.is_cuda and pack_scales.is_cuda):
        raise ValueError("qbmm_absorb operands must be CUDA tensors")

    launch = _compile_qbmm(
        op=op,
        batch=batch,
        num_heads=num_heads,
        head_stride=head_stride,
        p_dim=p_dim,
        v_dim=v_dim,
        latent_dim=latent_dim,
        device=a.device,
    )

    stream_int = (
        int(stream)
        if stream is not None
        else torch.cuda.current_stream().cuda_stream
    )
    launch.compiled(
        make_ptr(
            cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            cutlass.Uint8,
            pack_values.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint8,
            pack_scales.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        ),
        Int32(int(a.stride(0))),
        Int32(int(a.stride(1))),
        Int32(int(out.stride(0))),
        Int32(int(out.stride(1))),
        cuda.CUstream(stream_int),
    )


# ---------------------------------------------------------------------------
# Torch custom ops (op + fake pattern, per b12x::dense_gemm_launch /
# vllm b12x_mxfp8_linear).  out-mutating, None-returning: safe for CUDA-graph
# capture and torch.compile tracing.
# ---------------------------------------------------------------------------


@torch.library.custom_op("b12x::qbmm_absorb_ukt", mutates_args=("out",))
def _qbmm_absorb_ukt_op(
    a: torch.Tensor,
    pack_values: torch.Tensor,
    pack_scales: torch.Tensor,
    out: torch.Tensor,
    stream_int: Optional[int] = None,
    num_heads: int = 16,
    head_stride: int = 448,
    p_dim: int = 192,
    v_dim: int = 256,
    latent_dim: int = 512,
) -> None:
    qbmm_absorb(
        a, pack_values, pack_scales, out, op="ukt", stream=stream_int,
        num_heads=num_heads, head_stride=head_stride, p_dim=p_dim,
        v_dim=v_dim, latent_dim=latent_dim,
    )


@_qbmm_absorb_ukt_op.register_fake
def _qbmm_absorb_ukt_fake(
    a: torch.Tensor,
    pack_values: torch.Tensor,
    pack_scales: torch.Tensor,
    out: torch.Tensor,
    stream_int: Optional[int] = None,
    num_heads: int = 16,
    head_stride: int = 448,
    p_dim: int = 192,
    v_dim: int = 256,
    latent_dim: int = 512,
) -> None:
    return None


@torch.library.custom_op("b12x::qbmm_absorb_uv", mutates_args=("out",))
def _qbmm_absorb_uv_op(
    a: torch.Tensor,
    pack_values: torch.Tensor,
    pack_scales: torch.Tensor,
    out: torch.Tensor,
    stream_int: Optional[int] = None,
    num_heads: int = 16,
    head_stride: int = 448,
    p_dim: int = 192,
    v_dim: int = 256,
    latent_dim: int = 512,
) -> None:
    qbmm_absorb(
        a, pack_values, pack_scales, out, op="uv", stream=stream_int,
        num_heads=num_heads, head_stride=head_stride, p_dim=p_dim,
        v_dim=v_dim, latent_dim=latent_dim,
    )


@_qbmm_absorb_uv_op.register_fake
def _qbmm_absorb_uv_fake(
    a: torch.Tensor,
    pack_values: torch.Tensor,
    pack_scales: torch.Tensor,
    out: torch.Tensor,
    stream_int: Optional[int] = None,
    num_heads: int = 16,
    head_stride: int = 448,
    p_dim: int = 192,
    v_dim: int = 256,
    latent_dim: int = 512,
) -> None:
    return None


def qbmm_absorb_ukt(
    a: torch.Tensor,
    pack_values: torch.Tensor,
    pack_scales: torch.Tensor,
    out: torch.Tensor,
    stream: Optional[object] = None,
) -> None:
    """out[n, b, :] = a[n, b, :] @ W_UK_T[n]  (dequantized on the fly)."""
    torch.ops.b12x.qbmm_absorb_ukt(
        a, pack_values, pack_scales, out, _stream_to_int(stream)
    )


def qbmm_absorb_uv(
    a: torch.Tensor,
    pack_values: torch.Tensor,
    pack_scales: torch.Tensor,
    out: torch.Tensor,
    stream: Optional[object] = None,
) -> None:
    """out[n, b, :] = a[n, b, :] @ W_UV[n]  (dequantized on the fly)."""
    torch.ops.b12x.qbmm_absorb_uv(
        a, pack_values, pack_scales, out, _stream_to_int(stream)
    )


def _stream_to_int(stream: Optional[object]) -> Optional[int]:
    if stream is None:
        return None
    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is not None:
        return int(cuda_stream)
    return int(stream)


# ---------------------------------------------------------------------------
# Warmup precompile (design 5.6.5): register both ops for every live batch
# size BEFORE any CUDA-graph capture (jit-cache value-bake law).  Mirrors
# warmup_b12x_mxfp8_linear / the aiter per-batch-size precompile loop.
# ---------------------------------------------------------------------------


def warmup_qbmm_absorb(
    pack_values: torch.Tensor,
    pack_scales: torch.Tensor,
    *,
    batch_sizes: tuple[int, ...] = _DEFAULT_WARMUP_BATCH_SIZES,
    num_heads: int = 16,
    head_stride: int = 448,
    p_dim: int = 192,
    v_dim: int = 256,
    latent_dim: int = 512,
    stream: Optional[object] = None,
    synchronize: bool = True,
) -> int:
    """Compile + first-launch both ops for each batch size.  Returns count.

    Allocates small scratch activations/outputs (warmup only -- the serving
    launch path never allocates).  Output layouts mirror the live call sites:
    "ukt" writes a dense (N, B, L) tensor, "uv" writes the transposed view of
    a (B, N, V)-contiguous tensor.
    """
    device = pack_values.device
    stream_int = _stream_to_int(stream)
    warmed = 0
    for batch in batch_sizes:
        b = int(batch)
        a_ukt = torch.zeros(
            (num_heads, b, p_dim), dtype=torch.bfloat16, device=device
        )
        out_ukt = torch.empty(
            (num_heads, b, latent_dim), dtype=torch.bfloat16, device=device
        )
        torch.ops.b12x.qbmm_absorb_ukt(
            a_ukt, pack_values, pack_scales, out_ukt, stream_int
        )
        warmed += 1

        a_uv = torch.zeros(
            (num_heads, b, latent_dim), dtype=torch.bfloat16, device=device
        )
        out_uv_backing = torch.empty(
            (b, num_heads, v_dim), dtype=torch.bfloat16, device=device
        )
        torch.ops.b12x.qbmm_absorb_uv(
            a_uv,
            pack_values,
            pack_scales,
            out_uv_backing.transpose(0, 1),
            stream_int,
        )
        warmed += 1
    if synchronize and device.type == "cuda":
        torch.cuda.synchronize(device)
    return warmed


def qbmm_absorb_supported(
    *,
    num_heads: int,
    head_stride: int,
    p_dim: int,
    v_dim: int,
    latent_dim: int,
) -> bool:
    """True when the v1 kernel envelope covers this MLA geometry.

    Cheap (no compile): constructs both orientations' launch descriptors and
    reports whether their tile constraints hold.  Integrations should call
    this before electing the qbmm path and fall back to the BF16 absorbed
    pair otherwise."""
    for op in ("ukt", "uv"):
        try:
            QbmmAbsorbKernel(
                op=op,
                batch=1,
                num_heads=num_heads,
                head_stride=head_stride,
                p_dim=p_dim,
                v_dim=v_dim,
                latent_dim=latent_dim,
            )
        except ValueError:
            return False
    return True


__all__ = [
    "QbmmAbsorbKernel",
    "qbmm_absorb",
    "qbmm_absorb_supported",
    "qbmm_absorb_ukt",
    "qbmm_absorb_uv",
    "warmup_qbmm_absorb",
]
