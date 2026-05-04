# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.
#
# This is a real SM120 forward kernel skeleton:
# - SM90-style outer structure (tile scheduler + TMA producer/consumer split)
# - SM80-style warp MMA math core
# - 160-thread CTA: 4 compute warps + 1 TMA producer warp
#
# The initial slice is intentionally narrow:
# - fixed-length and packed varlen Q/K/V
# - paged KV for the serving-style path
# - no seqused_q
# - no block sparsity

import math
from functools import partial
from typing import Callable, Literal, Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
import cutlass.utils as utils_basic

from b12x.attention.contiguous import copy_utils
from b12x.attention.contiguous import layout_utils
from b12x.attention.contiguous.cute_dsl_utils import assume_tensor_aligned
from b12x.attention.contiguous import pipeline
from b12x.attention import utils
from b12x.attention.contiguous.mask import AttentionMask
from b12x.attention.contiguous.softmax import Softmax
from b12x.attention.contiguous.seqlen_info import SeqlenInfoQK
from b12x.attention.contiguous.block_info import BlockInfo
from b12x.attention.contiguous.pack_gqa import PackGQA, pack_gqa_layout
from b12x.attention.contiguous.named_barrier import NamedBarrierFwd
from b12x.attention.contiguous.tile_scheduler import (
    SingleTileDecodeScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from b12x.cute.fp4 import (
    bfloat2_to_float2_scaled,
    cvt_bf16x2_to_e4m3x2,
    fp8x4_e4m3_to_bfloat2x2,
    mxfp8_mma_m16n8k32_f32_e4m3,
)

@cute.jit
def warp_mma_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    A_in_regs: cutlass.Constexpr = False,
    B_in_regs: cutlass.Constexpr = False,
):
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    if const_expr(not A_in_regs):
        cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    if const_expr(not B_in_regs):
        cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        if k < cute.size(tCsA.shape[2]) - 1:
            if const_expr(not A_in_regs):
                cute.copy(
                    smem_thr_copy_A,
                    tCsA[None, None, k + 1],
                    tCrA_copy_view[None, None, k + 1],
                )
            if const_expr(not B_in_regs):
                cute.copy(
                    smem_thr_copy_B,
                    tCsB[None, None, k + 1],
                    tCrB_copy_view[None, None, k + 1],
                )
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def warp_mma_gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
):
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def convert_fp8_fragment_to_bf16(
    dst: cute.Tensor,
    src: cute.Tensor,
    transpose: cutlass.Constexpr = False,
):
    del transpose
    src_u8 = cute.flatten(cute.recast_tensor(src, cutlass.Uint8))
    src_u32 = cute.recast_tensor(src_u8, cutlass.Uint32)
    dst_u32 = cute.recast_tensor(dst, cutlass.Uint32)
    num_packed = cute.size(src_u32.shape)
    for i in cutlass.range_constexpr(num_packed):
        packed = src_u32[i]
        bf2_lo, bf2_hi = fp8x4_e4m3_to_bfloat2x2(packed)
        dst_u32[2 * i + 0] = bf2_lo
        dst_u32[2 * i + 1] = bf2_hi


@cute.jit
def copy_flattened(src: cute.Tensor, dst: cute.Tensor):
    src_flat = cute.flatten(src)
    dst_flat = cute.flatten(dst)
    for i in cutlass.range_constexpr(cute.size(dst_flat.shape)):
        dst_flat[i] = src_flat[i]


@cute.jit
def warp_mma_gemm_fp8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCrBRaw: cute.Tensor,
    tCsA: cute.Tensor,
    tCsBRaw: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B_raw: cute.TiledCopy,
    A_in_regs: cutlass.Constexpr = False,
    transpose: cutlass.Constexpr = False,
):
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_raw_copy_view = smem_thr_copy_B_raw.retile(tCrBRaw)
    if const_expr(not A_in_regs):
        cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    copy_flattened(tCsBRaw[None, None, 0], tCrB_raw_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        if k < cute.size(tCsA.shape[2]) - 1:
            if const_expr(not A_in_regs):
                cute.copy(
                    smem_thr_copy_A,
                    tCsA[None, None, k + 1],
                    tCrA_copy_view[None, None, k + 1],
                )
            copy_flattened(tCsBRaw[None, None, k + 1], tCrB_raw_copy_view[None, None, k + 1])
        convert_fp8_fragment_to_bf16(tCrB[None, None, k], tCrBRaw[None, None, k], transpose)
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def warp_mma_gemm_rs_fp8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCrBRaw: cute.Tensor,
    tCsBRaw: cute.Tensor,
    smem_thr_copy_B_raw: cute.TiledCopy,
    transpose: cutlass.Constexpr = False,
):
    tCrB_raw_copy_view = smem_thr_copy_B_raw.retile(tCrBRaw)
    copy_flattened(tCsBRaw[None, None, 0], tCrB_raw_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if const_expr(k < cute.size(tCrA.shape[2]) - 1):
            copy_flattened(tCsBRaw[None, None, k + 1], tCrB_raw_copy_view[None, None, k + 1])
        convert_fp8_fragment_to_bf16(tCrB[None, None, k], tCrBRaw[None, None, k], transpose)
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def warp_mma_gemm_rs_mxfp8(
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrBRaw: cute.Tensor,
    tCsBRaw: cute.Tensor,
    smem_thr_copy_B_raw: cute.TiledCopy,
):
    """PV accumulation using MXFP8 block-scaled MMA m16n8k32.

    Instead of dequanting V (FP8->BF16) and using BF16 MMA k=16, this path:
    1. Quantizes P (A operand) from BF16 to E4M3 using cvt.rn.satfinite.e4m3x2.bf16x2
    2. Loads V (B operand) as raw E4M3 bytes (already FP8 in KV cache)
    3. Uses MXFP8 block-scaled MMA m16n8k32 with scale=1.0 for both operands

    The k=32 MMA processes 2x the K-elements per instruction at similar cycle cost,
    giving ~2x MMA throughput on the PV accumulation.

    Requires num_k_steps (= tile_n / 16) to be even, i.e. tile_n divisible by 32.
    """
    tCrB_raw_copy_view = smem_thr_copy_B_raw.retile(tCrBRaw)

    # Scale factors: ue8m0 = 0x7F = 2^0 = 1.0 for both P and V.
    sfa = Uint32(0x7F7F7F7F)
    sfb = Uint32(0x7F7F7F7F)
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)

    num_k_steps = cute.size(tCrA.shape[2])

    # Recast all fragments to u32 for direct register manipulation.
    # The BF16 tiled_mma_pv partitions the A/B fragments into per-thread slices.
    # For a warp-level m16n8k16 MMA, each thread's fragment per k-step is:
    #   A: 4 u32 (8 bf16 values)
    #   B: 2 u32 (4 bf16 values) — but for raw FP8, 4 u8 = 1 u32
    #   C: 4 f32 (4 accumulator values)
    # We flatten everything and work on the raw registers directly.

    acc_flat = cute.flatten(acc)

    # Prefetch first two k-steps of V raw bytes
    copy_flattened(tCsBRaw[None, None, 0], tCrB_raw_copy_view[None, None, 0])
    if const_expr(num_k_steps > 1):
        copy_flattened(tCsBRaw[None, None, 1], tCrB_raw_copy_view[None, None, 1])

    for k_pair in cutlass.range_constexpr(num_k_steps // 2):
        k0 = k_pair * 2
        k1 = k0 + 1

        # Prefetch next pair of V k-steps
        if const_expr(k1 + 1 < num_k_steps):
            copy_flattened(tCsBRaw[None, None, k1 + 1], tCrB_raw_copy_view[None, None, k1 + 1])
        if const_expr(k1 + 2 < num_k_steps):
            copy_flattened(tCsBRaw[None, None, k1 + 2], tCrB_raw_copy_view[None, None, k1 + 2])

        # Quantize A (P): flatten both k-steps to u32, convert bf16x2 -> e4m3x2, pack
        a_k0 = cute.flatten(cute.recast_tensor(tCrA[None, None, k0], cutlass.Uint32))
        a_k1 = cute.flatten(cute.recast_tensor(tCrA[None, None, k1], cutlass.Uint32))

        # Pack B (V): flatten both k-steps to u32
        b_k0 = cute.flatten(cute.recast_tensor(tCrBRaw[None, None, k0], cutlass.Uint32))
        b_k1 = cute.flatten(cute.recast_tensor(tCrBRaw[None, None, k1], cutlass.Uint32))

        # For the tiled MMA with num_compute_warps warps, cute.gemm would issue
        # multiple MMA instructions per k-step: one per (m_tile, n_tile) pair.
        # The flattened A fragment has 4 * num_m_tiles u32 values.
        # The flattened B fragment has 1 * num_n_tiles u32 values.
        # The flattened acc has 4 * num_m_tiles * num_n_tiles f32 values.
        #
        # Each MMA m16n8k32 consumes: A[4 u32], B[2 u32], C[4 f32]
        # We iterate over accumulator tiles in the same order cute.gemm would.

        num_a_words = cute.size(a_k0.shape)  # 4 * num_m_tiles
        num_b_words = cute.size(b_k0.shape)  # 1 * num_n_tiles
        num_m_tiles = num_a_words // 4
        num_n_tiles = num_b_words  # 1 u32 per n_tile per k-step

        for m in cutlass.range_constexpr(num_m_tiles):
            # Quantize this m-tile's A registers for k=32.
            #
            # BF16 m16n8k16 A-fragment layout per m-tile (4 u32):
            #   reg0: rows 0-1, k[0:4] as bf16x2
            #   reg1: rows 0-1, k[4:8] as bf16x2
            #   reg2: rows 2-3, k[0:4] as bf16x2
            #   reg3: rows 2-3, k[4:8] as bf16x2
            #
            # MXFP8 m16n8k32 A-fragment layout per m-tile (4 u32):
            #   reg0: rows 0-1, k[0:4] as e4m3x4   (first k-half, rows 0-1)
            #   reg1: rows 2-3, k[0:4] as e4m3x4   (first k-half, rows 2-3)
            #   reg2: rows 0-1, k[16:20] as e4m3x4 (second k-half, rows 0-1)
            #   reg3: rows 2-3, k[16:20] as e4m3x4 (second k-half, rows 2-3)
            #
            # Each cvt_bf16x2_to_e4m3x2 converts 2 bf16 → 2 e4m3 (in low 16 bits).
            # We pack two cvt results into one u32 (4 e4m3 bytes).
            a_base = m * 4
            # Pack two k=16 BF16 A-fragments into one k=32 MXFP8 A-fragment.
            # Each BF16 reg has 2 bf16 values; cvt produces 2 e4m3 in low 16 bits.
            # Pack two cvt results (4 e4m3 bytes) into one u32.
            # The k0/k1 interleaving within each register matches the MXFP8 layout:
            # each A register holds bytes from both k-halves for its owned rows.
            qa0 = (cvt_bf16x2_to_e4m3x2(a_k0[a_base + 0]) & mask16) | ((cvt_bf16x2_to_e4m3x2(a_k1[a_base + 0]) & mask16) << shift16)
            qa1 = (cvt_bf16x2_to_e4m3x2(a_k0[a_base + 1]) & mask16) | ((cvt_bf16x2_to_e4m3x2(a_k1[a_base + 1]) & mask16) << shift16)
            qa2 = (cvt_bf16x2_to_e4m3x2(a_k0[a_base + 2]) & mask16) | ((cvt_bf16x2_to_e4m3x2(a_k1[a_base + 2]) & mask16) << shift16)
            qa3 = (cvt_bf16x2_to_e4m3x2(a_k0[a_base + 3]) & mask16) | ((cvt_bf16x2_to_e4m3x2(a_k1[a_base + 3]) & mask16) << shift16)

            for n in cutlass.range_constexpr(num_n_tiles):
                # Pack this n-tile's B registers for k=32
                qb0 = b_k0[n]
                qb1 = b_k1[n]

                # Accumulator index: (m * num_n_tiles + n) * 4
                acc_base = (m * num_n_tiles + n) * 4
                d0 = acc_flat[acc_base + 0]
                d1 = acc_flat[acc_base + 1]
                d2 = acc_flat[acc_base + 2]
                d3 = acc_flat[acc_base + 3]

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    d0, d1, d2, d3,
                    qa0, qa1, qa2, qa3,
                    qb0, qb1,
                    sfa, sfb,
                )

                acc_flat[acc_base + 0] = d0
                acc_flat[acc_base + 1] = d1
                acc_flat[acc_base + 2] = d2
                acc_flat[acc_base + 3] = d3


class ContiguousAttentionForwardKernel:
    arch = 120

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        kv_dtype: Optional[Type[cutlass.Numeric]] = None,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_splits: int = 1,
        num_threads: int = 160,
        num_compute_warps: int = 4,
        Q_in_regs: bool = False,
        score_mod: Optional[cutlass.Constexpr] = None,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        paged_direct_q_seqlen: int = 0,
        use_native_fp8_pv_mma: Optional[bool] = None,
    ):
        self.dtype = dtype
        self.kv_dtype = dtype if kv_dtype is None else kv_dtype
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.num_splits = num_splits
        self.is_split_kv = num_splits > 1
        self.Q_in_regs = Q_in_regs
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.qk_acc_dtype = Float32
        assert self.score_mod is None, "score_mod is not part of the initial b12x transplant"
        assert self.mask_mod is None, "mask_mod is not part of the initial b12x transplant"
        self.mma_pv_is_rs = mma_pv_is_rs
        assert self.mma_pv_is_rs, "SM120 rewrite currently only supports register-sourced PV"
        self.buffer_align_bytes = 1024
        self.num_compute_warps = num_compute_warps
        assert self.num_compute_warps >= 1
        self.num_threads_per_warp = 32
        self.producer_warp_idx = self.num_compute_warps
        self.use_tma_KV = not paged_kv_non_tma
        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "SM120 paged KV cp.async path does not support irregular head dim"
        )
        self.kv_is_fp8 = self.kv_dtype == cutlass.Float8E4M3FN
        # Native MXFP8 block-scaled MMA for PV accumulation in FP8-weight regimes.
        # Default remains the BF16 dequant path because contiguous integration does
        # not yet carry projection-weight dtype metadata.
        if use_native_fp8_pv_mma is None:
            self.use_native_fp8_pv_mma = False
        elif use_native_fp8_pv_mma:
            assert self.kv_is_fp8, "native FP8 PV MMA requires FP8 KV cache"
            assert tile_n % 32 == 0, "native FP8 PV MMA requires tile_n divisible by 32"
            self.use_native_fp8_pv_mma = True
        else:
            self.use_native_fp8_pv_mma = False
        self.paged_direct_q_seqlen = paged_direct_q_seqlen

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric] | None,
        mCuSeqlensQ_type: Type[cutlass.Numeric] | None,
        mCuSeqlensK_type: Type[cutlass.Numeric] | None,
        mSeqUsedQ_type: Type[cutlass.Numeric] | None,
        mSeqUsedK_type: Type[cutlass.Numeric] | None,
        mKDescale_type: Type[cutlass.Numeric] | None,
        mVDescale_type: Type[cutlass.Numeric] | None,
        mFp8Lut_type: Type[cutlass.Numeric] | None,
        learnable_sink_type: Type[cutlass.Numeric] | None,
    ):
        if const_expr(not (mQ_type == mO_type)):
            raise TypeError("Q and O tensors must have the same data type")
        if const_expr(not (mK_type == mV_type)):
            raise TypeError("K and V tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Q/O tensors must be Float16 or BFloat16")
        if const_expr(mK_type not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN]):
            raise TypeError("K/V tensors must be Float16, BFloat16, or Float8E4M3FN")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        if const_expr(mSeqUsedQ_type not in [None, Int32]):
            raise TypeError("seqused_q tensor must be Int32")
        if const_expr(mSeqUsedK_type not in [None, Int32]):
            raise TypeError("seqused_k tensor must be Int32")
        if const_expr(mKDescale_type not in [None, Float32]):
            raise TypeError("k_descale tensor must be Float32")
        if const_expr(mVDescale_type not in [None, Float32]):
            raise TypeError("v_descale tensor must be Float32")
        if const_expr(mFp8Lut_type not in [None, Float32]):
            raise TypeError("fp8 LUT tensor must be Float32")
        if const_expr(learnable_sink_type not in [None, Float32]):
            raise TypeError("learnable sink tensor must be Float32")
        assert mQ_type == self.dtype
        assert mK_type == self.kv_dtype

    def _setup_attributes(self):
        sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom = (
            self._get_smem_layout_atom()
        )
        self.sQ_layout = cute.tile_to_shape(sQ_layout_atom, (self.tile_m, self.tile_hdim), (0, 1))
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom,
            (self.tile_n, self.tile_hdim, self.num_stages),
            (0, 1, 2),
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.tile_n, self.tile_hdimv, self.num_stages),
            (0, 1, 2),
        )
        self.sO_layout = cute.tile_to_shape(sO_layout_atom, (self.tile_m, self.tile_hdimv), (0, 1))
        self.sP_layout = (
            cute.tile_to_shape(sP_layout_atom, (self.tile_m, self.tile_n), (0, 1))
            if const_expr(sP_layout_atom is not None)
            else None
        )
        self.sK_raw_layout = (
            cute.make_layout(
                (self.tile_n, self.tile_hdim, self.num_stages),
                stride=(self.tile_hdim, 1, self.tile_n * self.tile_hdim),
            )
            if const_expr(self.kv_is_fp8)
            else None
        )
        self.sV_raw_layout = (
            cute.make_layout(
                (self.tile_n, self.tile_hdimv, self.num_stages),
                stride=(self.tile_hdimv, 1, self.tile_n * self.tile_hdimv),
            )
            if const_expr(self.kv_is_fp8)
            else None
        )

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_Q_load_threads % tQK_shape_dim_1 == 0
        assert self.num_producer_threads % tQK_shape_dim_1 == 0
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        assert self.tile_m % tQ_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        assert self.tile_m % tO_layout.shape[0] == 0
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout
        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, tQ_layout, vQKV_layout)
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async_copy, tK_layout, vQKV_layout)
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async_copy, tV_layout, vQKV_layout)
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
        num_compute_warps=4,
        kv_dtype=None,
    ) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if kv_dtype is None:
            kv_dtype = dtype
        if kv_dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN]:
            return False
        if kv_dtype == cutlass.Float8E4M3FN and dtype != cutlass.BFloat16:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if num_compute_warps < 1:
            return False
        if tile_m % (num_compute_warps * 16) != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads != (num_compute_warps + 1) * 32:
            return False
        q_elem_bytes = dtype.width // 8
        kv_elem_bytes = kv_dtype.width // 8
        smem_usage_Q = tile_m * head_dim * q_elem_bytes
        smem_usage_K = tile_n * head_dim * num_stages * q_elem_bytes
        smem_usage_V = tile_n * head_dim_v * num_stages * q_elem_bytes
        smem_usage_QV = max(smem_usage_Q, smem_usage_V) if Q_in_regs else (smem_usage_Q + smem_usage_V)
        smem_usage = smem_usage_QV + smem_usage_K
        if kv_dtype == cutlass.Float8E4M3FN:
            smem_usage += tile_n * head_dim * num_stages * kv_elem_bytes
            smem_usage += tile_n * head_dim_v * num_stages * kv_elem_bytes
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False
        return True

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct

        if const_expr(not self.kv_is_fp8 and not self.Q_in_regs):
            return SharedStorageQKV
        if const_expr(not self.kv_is_fp8):
            return SharedStorageSharedQV

        sK_raw_struct = cute.struct.Align[
            cute.struct.MemRange[self.kv_dtype, cute.cosize(self.sK_raw_layout)], self.buffer_align_bytes
        ]
        sV_raw_struct = cute.struct.Align[
            cute.struct.MemRange[self.kv_dtype, cute.cosize(self.sV_raw_layout)], self.buffer_align_bytes
        ]

        @cute.struct
        class SharedStorageSharedQVFp8:
            mbar_ptr: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sV_raw: sV_raw_struct
            sK_raw: sK_raw_struct

        return SharedStorageSharedQVFp8

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mVDescale: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32 = 0,
    ):
        del tma_atom_O
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        if const_expr(self.kv_is_fp8 and mVDescale is not None):
            head_idx_kv = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
            out_scale = mVDescale[batch_idx, head_idx_kv]
            rO_scaled = cute.make_fragment_like(acc_O, Float32)
            rO_scaled.store(rO.load().to(Float32) * out_scale)
            rO.store(rO_scaled.load().to(self.dtype))
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads
        )
        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=2 * self.dtype.width,
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead
        )
        if const_expr(mLSE is not None):
            if const_expr(not seqlen.has_cu_seqlens_q):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
            else:
                offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                mLSE_select = (
                    mLSE[None, head_idx, split_idx]
                    if const_expr(self.is_split_kv)
                    else mLSE[None, head_idx]
                )
                mLSE_cur = cute.domain_offset((offset,), mLSE_select)
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
                )
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
                taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
                if taccOcO[0][1] == 0:
                    for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                        if (
                            t0accOcO[m, 0][0]
                            < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                        ):
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)

        if const_expr(not seqlen.has_cu_seqlens_q):
            mO_cur = mO[None, None, head_idx, batch_idx]
        else:
            offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
            mO_select = (
                mO[None, None, head_idx, split_idx]
                if const_expr(self.is_split_kv)
                else mO[None, None, head_idx]
            )
            mO_cur = cute.domain_offset((offset, 0), mO_select)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, self.dtype)
        cute.autovec_copy(tOsO, tOrO)
        if const_expr(not self.pack_gqa):
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0]
                    < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        mKDescale: Optional[cute.Tensor] = None,
        mVDescale: Optional[cute.Tensor] = None,
        mFp8Lut: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        learnable_sink: Optional[cute.Tensor] = None,
        has_attention_sink_bias: cutlass.Constexpr = False,
        blocksparse_tensors=None,
        aux_tensors=None,
        logical_num_batch_static: Int32 = 1,
        logical_seqlen_q_static: Int32 = 0,
        logical_seqlen_k_static: Int32 = 0,
        stream: cuda.CUstream = None,
    ):
        assert mSeqUsedQ is None
        assert blocksparse_tensors is None
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                    mKDescale,
                    mVDescale,
                    mFp8Lut,
                    learnable_sink,
                )
            )
        )

        self.num_threads = (self.num_compute_warps + 1) * self.num_threads_per_warp
        self.num_mma_threads = self.num_compute_warps * self.num_threads_per_warp
        self.num_producer_threads = self.num_threads_per_warp
        self.num_Q_load_threads = self.num_mma_threads
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs = 248
        self.num_producer_regs = 80
        self.use_tma_Q = True
        self.use_tma_KV = mK.element_type in [cutlass.Float16, cutlass.BFloat16] or (
            self.kv_is_fp8 and const_expr(mPageTable is not None) and mPageTable.shape[1] > 0
        )
        self.use_tma_O = False
        if const_expr(not self.use_tma_KV and self.dtype != cutlass.BFloat16):
            assert mFp8Lut is not None, "FP8 KV path requires an FP8 lookup table"

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(cute.rank(mQ) == 4) else [0, 2, 1]
        O_layout_transpose = [1, 3, 2, 0] if const_expr(cute.rank(mO) == 4) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(cute.rank(mK) == 4) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(mLSE is not None):
            LSE_layout_transpose = [2, 1, 0] if const_expr(cute.rank(mLSE) == 3) else [1, 0]
            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))

        q_heads_unpacked = mQ.shape[2]
        kv_heads = mK.shape[2]
        logical_num_head = kv_heads if const_expr(self.pack_gqa) else q_heads_unpacked
        logical_q_rows_static = logical_seqlen_q_static * (
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
        )
        logical_num_block = cute.ceil_div(logical_q_rows_static, self.tile_m)
        logical_total_q = (
            mQ.shape[0] * (self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1)
            if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None)
            else logical_q_rows_static * logical_num_batch_static
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        sK_tma_layout = (
            cute.select(self.sK_raw_layout, mode=[0, 1])
            if const_expr(self.kv_is_fp8)
            else cute.select(self.sK_layout, mode=[0, 1])
        )
        sV_tma_layout = (
            cute.select(self.sV_raw_layout, mode=[0, 1])
            if const_expr(self.kv_is_fp8)
            else cute.select(self.sV_layout, mode=[0, 1])
        )
        self.tma_copy_bytes = {
            "Q": cute.size_in_bytes(mQ.element_type, self.sQ_layout),
            "K": cute.size_in_bytes(mK.element_type, sK_tma_layout),
            "V": cute.size_in_bytes(mV.element_type, sV_tma_layout),
        }
        if const_expr(mPageTable is not None):
            assert mK.shape[0] == self.tile_n, "paged TMA path requires page_size == tile_n"
            assert mV.shape[0] == self.tile_n, "paged TMA path requires page_size == tile_n"

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q,
            mQ,
            self.sQ_layout,
            (self.tile_m, self.tile_hdim),
        )
        TileScheduler = (
            SingleTileDecodeScheduler
            if const_expr(mPageTable is not None and self.paged_direct_q_seqlen > 0)
            else (
                SingleTileVarlenScheduler
                if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None)
                else SingleTileScheduler
            )
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=logical_num_block,
            num_head=logical_num_head,
            num_batch=(
                logical_num_batch_static
                if const_expr(mCuSeqlensQ is None)
                else mCuSeqlensQ.shape[0] - 1
            ),
            num_splits=self.num_splits,
            seqlen_k=logical_seqlen_k_static,
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=logical_total_q,
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            direct_q_rows_per_batch=(
                self.paged_direct_q_seqlen
                * (self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1)
            ),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            element_size=self.dtype.width // 8,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale)
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_O,
                mO,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),
            )
        tma_atom_K, tma_tensor_K = (None, None)
        tma_atom_V, tma_tensor_V = (None, None)
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                sK_tma_layout,
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                sV_tma_layout,
                (self.tile_n, self.tile_hdimv),
                1,
            )
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            mKDescale,
            mVDescale,
            mFp8Lut,
            learnable_sink,
            has_attention_sink_bias,
            tma_atom_Q,
            tma_atom_K if const_expr(self.use_tma_KV) else None,
            tma_atom_V if const_expr(self.use_tma_KV) else None,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            logical_seqlen_q_static,
            logical_seqlen_k_static,
            aux_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        mKDescale: Optional[cute.Tensor],
        mVDescale: Optional[cute.Tensor],
        mFp8Lut: Optional[cute.Tensor],
        mAttentionSinkBias: cute.Tensor,
        has_attention_sink_bias: cutlass.Constexpr,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr,
        SharedStorage: cutlass.Constexpr,
        logical_seqlen_q_static: Int32,
        logical_seqlen_k_static: Int32,
        aux_tensors=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            if const_expr(self.use_tma_KV):
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 0:
            cute.arch.mbarrier_init(mbar_ptr_Q, 1)
        cute.arch.sync_threads()

        if const_expr(self.use_tma_KV):
            pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, self.num_compute_warps
            )
            pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread
            )
            pipeline_k = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline_kv_producer_group,
                consumer_group=pipeline_kv_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline_kv_producer_group,
                consumer_group=pipeline_kv_consumer_group,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=False,
            )
        else:
            # PipelineAsync barriers are not warp-gated. Use actual thread counts or
            # the producer/consumer arrive counts diverge and the launch can fault.
            pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, self.num_mma_threads
            )
            pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, self.num_producer_threads
            )
            pipeline_k = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline_kv_producer_group,
                consumer_group=pipeline_kv_consumer_group,
                defer_sync=True,
            )
            pipeline_v = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline_kv_producer_group,
                consumer_group=pipeline_kv_consumer_group,
                defer_sync=False,
            )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = (
            storage.sQ.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=self.dtype)
            if const_expr(self.Q_in_regs)
            else storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        )
        sKRaw = (
            storage.sK_raw.get_tensor(
                cute.make_layout(
                    (self.tile_n, self.tile_hdim, self.num_stages),
                    stride=(self.tile_hdim, 1, self.tile_n * self.tile_hdim),
                )
            )
            if const_expr(self.kv_is_fp8)
            else None
        )
        sVRaw = (
            storage.sV_raw.get_tensor(
                cute.make_layout(
                    (self.tile_n, self.tile_hdimv, self.num_stages),
                    stride=(self.tile_hdimv, 1, self.tile_n * self.tile_hdimv),
                )
            )
            if const_expr(self.kv_is_fp8)
            else None
        )
        sVt = layout_utils.transpose_view(sV)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = (
            partial(
                SeqlenInfoQK.create_decode,
                seqlen_q_static=logical_seqlen_q_static,
                seqlen_k_static=logical_seqlen_k_static,
                mSeqUsedK=mSeqUsedK,
            )
            if const_expr(mPageTable is not None and self.paged_direct_q_seqlen == 1)
            else (
                partial(
                    SeqlenInfoQK.create_uniform_q,
                    seqlen_q_static=self.paged_direct_q_seqlen,
                    seqlen_k_static=logical_seqlen_k_static,
                    mSeqUsedK=mSeqUsedK,
                )
                if const_expr(mPageTable is not None and self.paged_direct_q_seqlen > 1)
                else partial(
                    SeqlenInfoQK.create,
                    seqlen_q_static=logical_seqlen_q_static,
                    seqlen_k_static=logical_seqlen_k_static,
                    mCuSeqlensQ=mCuSeqlensQ,
                    mCuSeqlensK=mCuSeqlensK,
                    mSeqUsedQ=mSeqUsedQ,
                    mSeqUsedK=mSeqUsedK,
                )
            )
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx == self.producer_warp_idx:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                sKRaw,
                sVRaw,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
        elif warp_idx < self.num_compute_warps:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx = cute.arch.thread_idx()[0]
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mQ,
                mO,
                mLSE,
                sQ,
                sK,
                sKRaw,
                sV,
                sVRaw,
                sVt,
                sO,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                mKDescale,
                mVDescale,
                softmax_scale,
                mFp8Lut,
                mAttentionSinkBias,
                has_attention_sink_bias,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                aux_tensors,
            )

    @cute.jit
    def load_paged_kv_stage_raw(
        self,
        mX: cute.Tensor,
        sX: cute.Tensor,
        batch_idx: Int32,
        head_idx_kv: Int32,
        src_idx: Int32,
        stage_idx: Int32,
        tile_hdim_x: cutlass.Constexpr,
    ):
        lane = cute.arch.lane_idx()
        del batch_idx
        mXu32 = cute.recast_tensor(mX, cutlass.Uint32)
        sXu32 = cute.recast_tensor(sX, cutlass.Uint32)
        words_per_row = tile_hdim_x // 4
        total_packed = self.tile_n * words_per_row
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_packed, cute.arch.WARP_SIZE)):
            packed_idx = lane + idx_iter * cute.arch.WARP_SIZE
            if packed_idx < total_packed:
                row = packed_idx // words_per_row
                col_word = packed_idx - row * words_per_row
                sXu32[row, col_word, stage_idx] = mXu32[row, col_word, head_idx_kv, src_idx]

    @cute.jit
    def dequant_fp8_stage_shared(
        self,
        sXRaw: cute.Tensor,
        sX: cute.Tensor,
        mFp8Lut: Optional[cute.Tensor],
        stage_idx: Int32,
        tile_hdim_x: cutlass.Constexpr,
        tidx: Int32,
    ):
        total_elems = self.tile_n * tile_hdim_x
        total_vec4 = total_elems // 4
        one = Float32(1.0)
        sXRaw_u8 = cute.recast_tensor(sXRaw, cutlass.Uint8)
        if const_expr(self.dtype == cutlass.BFloat16):
            for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_vec4, self.num_mma_threads)):
                vec_idx = tidx + idx_iter * self.num_mma_threads
                if vec_idx < total_vec4:
                    linear_idx = vec_idx * 4
                    row = linear_idx // tile_hdim_x
                    col = linear_idx - row * tile_hdim_x
                    packed = (
                        cutlass.Uint32(sXRaw_u8[row, col + 0, stage_idx])
                        | (cutlass.Uint32(sXRaw_u8[row, col + 1, stage_idx]) << cutlass.Uint32(8))
                        | (cutlass.Uint32(sXRaw_u8[row, col + 2, stage_idx]) << cutlass.Uint32(16))
                        | (cutlass.Uint32(sXRaw_u8[row, col + 3, stage_idx]) << cutlass.Uint32(24))
                    )
                    bf2_01, bf2_23 = fp8x4_e4m3_to_bfloat2x2(packed)
                    value0, value1 = bfloat2_to_float2_scaled(bf2_01, one)
                    value2, value3 = bfloat2_to_float2_scaled(bf2_23, one)
                    sX[row, col + 0, stage_idx] = value0.to(self.dtype)
                    sX[row, col + 1, stage_idx] = value1.to(self.dtype)
                    sX[row, col + 2, stage_idx] = value2.to(self.dtype)
                    sX[row, col + 3, stage_idx] = value3.to(self.dtype)
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.KVConvert),
                number_of_threads=self.num_mma_threads,
            )
            return
        assert mFp8Lut is not None
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_elems, self.num_mma_threads)):
            linear_idx = tidx + idx_iter * self.num_mma_threads
            if linear_idx < total_elems:
                row = linear_idx // tile_hdim_x
                col = linear_idx - row * tile_hdim_x
                value = mFp8Lut[sXRaw_u8[row, col, stage_idx].to(Int32)]
                sX[row, col, stage_idx] = value.to(self.dtype)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.KVConvert),
            number_of_threads=self.num_mma_threads,
        )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sKRaw: Optional[cute.Tensor],
        sVRaw: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        kv_producer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.num_stages
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        wait_for_q_consumed = False
        while work_tile.is_valid_tile:
            if const_expr(self.Q_in_regs) and wait_for_q_consumed:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PEmpty),
                    number_of_threads=self.num_threads,
                )
                wait_for_q_consumed = False
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(cute.rank(mQ) == 4):
                mQ_batch = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)
            elif const_expr(seqlen.has_cu_seqlens_q):
                mQ_batch = seqlen.offset_batch_Q(mQ, batch_idx, dim=2)
            else:
                mQ_batch = mQ
            mQ_cur = mQ_batch[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
            )
            head_idx_kv = (
                head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
            )
            if const_expr(mPageTable is not None):
                mK_cur = mK[None, None, head_idx_kv, None]
                mV_cur = mV[None, None, head_idx_kv, None]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None))
            else:
                if const_expr(cute.rank(mK) == 4):
                    mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                        None, None, head_idx_kv
                    ]
                    mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                        None, None, head_idx_kv
                    ]
                elif const_expr(seqlen.has_cu_seqlens_k):
                    mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=2)[
                        None, None, head_idx_kv
                    ]
                    mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=2)[
                        None, None, head_idx_kv
                    ]
                else:
                    mK_cur = mK[None, None, head_idx_kv]
                    mV_cur = mV[None, None, head_idx_kv]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
            if const_expr(self.use_tma_KV):
                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    gK,
                    sKRaw if const_expr(self.kv_is_fp8) else sK,
                )
                load_K = copy_utils.tma_producer_copy_fn(load_K, pipeline_k)
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    gV,
                    sVRaw if const_expr(self.kv_is_fp8) else sV,
                )
                load_V = copy_utils.tma_producer_copy_fn(load_V, pipeline_v)

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, self.num_splits
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr_Q, self.tma_copy_bytes["Q"])
            load_Q(tma_bar_ptr=mbar_ptr_Q)
            if const_expr(self.Q_in_regs):
                wait_for_q_consumed = True
            for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                n_block = n_block_max - 1 - n_tile
                src_idx = mPageTable[batch_idx, n_block] if const_expr(mPageTable is not None) else n_block
                pipeline_k.producer_acquire(kv_producer_state)
                if const_expr(self.use_tma_KV):
                    load_K(src_idx=src_idx, producer_state=kv_producer_state)
                else:
                    self.load_paged_kv_stage_raw(
                        mK,
                        sKRaw,
                        batch_idx,
                        head_idx_kv,
                        src_idx,
                        kv_producer_state.index,
                        self.tile_hdim,
                    )
                    pipeline_k.producer_commit(kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state)
                if const_expr(self.use_tma_KV):
                    load_V(src_idx=src_idx, producer_state=kv_producer_state)
                else:
                    self.load_paged_kv_stage_raw(
                        mV,
                        sVRaw,
                        batch_idx,
                        head_idx_kv,
                        src_idx,
                        kv_producer_state.index,
                        self.tile_hdimv,
                    )
                    pipeline_v.producer_commit(kv_producer_state)
                kv_producer_state.advance()

            if const_expr(not self.Q_in_regs):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_threads,
                )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(self.Q_in_regs) and wait_for_q_consumed:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PEmpty),
                number_of_threads=self.num_threads,
            )
        pipeline_k.producer_tail(kv_producer_state)
        pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def mma_one_n_block(
        self,
        n_block: Int32,
        kv_consumer_state,
        thr_mma_qk: cute.TiledMma,
        thr_mma_pv: cute.TiledMma,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tSrKRaw: Optional[cute.Tensor],
        tOrVt: cute.Tensor,
        tOrVtRaw: Optional[cute.Tensor],
        acc_O: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sKRaw: Optional[cute.Tensor],
        sVRaw: Optional[cute.Tensor],
        smem_thr_copy_Q: cute.TiledCopy,
        smem_thr_copy_K: cute.TiledCopy,
        smem_thr_copy_KRaw: Optional[cute.TiledCopy],
        smem_thr_copy_V: cute.TiledCopy,
        smem_thr_copy_VRaw: Optional[cute.TiledCopy],
        tSsQ: cute.Tensor,
        tSsK: cute.Tensor,
        tSsKRaw: Optional[cute.Tensor],
        tOsVt: cute.Tensor,
        tOsVtRaw: Optional[cute.Tensor],
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        softmax: Softmax,
        mFp8Lut: Optional[cute.Tensor],
        seqlen: SeqlenInfoQK,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        mask_fn: Callable,
        aux_tensors=None,
        fastdiv_mods=None,
        is_first_n_block: cutlass.Constexpr = False,
    ):
        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        acc_shape_S = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_fragment(acc_shape_S, Float32)
        acc_S.fill(0.0)
        if const_expr(self.kv_is_fp8):
            warp_mma_gemm_fp8(
                thr_mma_qk,
                acc_S,
                tSrQ,
                tSrK,
                tSrKRaw,
                tSsQ,
                tSsKRaw[
                    None, None, None, kv_consumer_state.index if const_expr(self.num_stages > 1) else 0
                ],
                smem_thr_copy_Q,
                smem_thr_copy_KRaw,
                A_in_regs=self.Q_in_regs,
                transpose=False,
            )
        else:
            warp_mma_gemm(
                thr_mma_qk,
                acc_S,
                tSrQ,
                tSrK,
                tSsQ,
                tSsK[
                    None, None, None, kv_consumer_state.index if const_expr(self.num_stages > 1) else 0
                ],
                smem_thr_copy_Q,
                smem_thr_copy_K,
                A_in_regs=self.Q_in_regs,
            )
        pipeline_k.consumer_release(kv_consumer_state)

        mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=True)
        softmax.rescale_O(acc_O, row_scale)

        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)

        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        if const_expr(self.use_native_fp8_pv_mma):
            warp_mma_gemm_rs_mxfp8(
                acc_O,
                tOrP,
                tOrVtRaw,
                tOsVtRaw[
                    None, None, None, kv_consumer_state.index if const_expr(self.num_stages > 1) else 0
                ],
                smem_thr_copy_VRaw,
            )
        elif const_expr(self.kv_is_fp8):
            warp_mma_gemm_rs_fp8(
                thr_mma_pv,
                acc_O,
                tOrP,
                tOrVt,
                tOrVtRaw,
                tOsVtRaw[
                    None, None, None, kv_consumer_state.index if const_expr(self.num_stages > 1) else 0
                ],
                smem_thr_copy_VRaw,
                transpose=True,
            )
        else:
            warp_mma_gemm_rs(
                thr_mma_pv,
                acc_O,
                tOrP,
                tOrVt,
                tOsVt[
                    None, None, None, kv_consumer_state.index if const_expr(self.num_stages > 1) else 0
                ],
                smem_thr_copy_V,
            )
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sKRaw: Optional[cute.Tensor],
        sV: cute.Tensor,
        sVRaw: Optional[cute.Tensor],
        sVt: cute.Tensor,
        sO: cute.Tensor,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        tidx: Int32,
        softmax_scale_log2: Float32,
        mKDescale: Optional[cute.Tensor],
        mVDescale: Optional[cute.Tensor],
        softmax_scale: Optional[Float32],
        mFp8Lut: Optional[cute.Tensor],
        mAttentionSinkBias: cute.Tensor,
        has_attention_sink_bias: cutlass.Constexpr,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors=None,
    ):
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        sVtRaw = layout_utils.transpose_view(sVRaw) if const_expr(self.kv_is_fp8) else None
        sKRawU8 = cute.recast_tensor(sKRaw, cutlass.Uint8) if const_expr(self.kv_is_fp8) else None
        sVtRawU8 = (
            cute.recast_tensor(sVtRaw, cutlass.Uint8) if const_expr(self.kv_is_fp8) else None
        )
        tSrKRaw = (
            cute.make_fragment_like(cute.recast_tensor(tSrK, cutlass.Uint8), cutlass.Uint8)
            if const_expr(self.kv_is_fp8)
            else None
        )
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        tOrVtRaw = (
            cute.make_fragment_like(cute.recast_tensor(tOrVt, cutlass.Uint8), cutlass.Uint8)
            if const_expr(self.kv_is_fp8)
            else None
        )
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_fragment(acc_shape_O, Float32)
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        taccOcO = thr_mma_pv.partition_C(cO)
        taccOcO_mn = layout_utils.reshape_acc_to_mn(taccOcO)

        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
        )
        smem_copy_atom_KRaw = (
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Uint8,
            )
            if const_expr(self.kv_is_fp8)
            else None
        )
        smem_copy_atom_VRaw = (
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Uint8,
            )
            if const_expr(self.kv_is_fp8)
            else None
        )
        smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)
        smem_thr_copy_KRaw = (
            utils.make_tiled_copy_B(smem_copy_atom_KRaw, tiled_mma_qk).get_slice(tidx)
            if const_expr(self.kv_is_fp8)
            else None
        )
        smem_thr_copy_VRaw = (
            utils.make_tiled_copy_B(smem_copy_atom_VRaw, tiled_mma_pv).get_slice(tidx)
            if const_expr(self.kv_is_fp8)
            else None
        )
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tSsKRaw = smem_thr_copy_KRaw.partition_S(sKRawU8) if const_expr(self.kv_is_fp8) else None
        tOsVtRaw = smem_thr_copy_VRaw.partition_S(sVtRawU8) if const_expr(self.kv_is_fp8) else None

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        base_softmax_scale_log2 = softmax_scale_log2
        softmax_num_rows = (
            cutlass.min(acc_O.shape[0][0] * acc_O.shape[1], self.qhead_per_kvhead)
            if const_expr(self.paged_direct_q_seqlen == 1 and self.pack_gqa)
            else acc_O.shape[0][0] * acc_O.shape[1]
        )
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=softmax_num_rows,
            softmax_scale=softmax_scale,
        )
        q_consumer_phase = Int32(0)
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
            if const_expr(self.kv_is_fp8 and mKDescale is not None):
                softmax.scale_log2 = base_softmax_scale_log2 * mKDescale[batch_idx, head_idx_kv]
            else:
                softmax.scale_log2 = base_softmax_scale_log2
            cute.arch.mbarrier_wait(mbar_ptr_Q, phase=q_consumer_phase)
            q_consumer_phase ^= 1
            if const_expr(self.Q_in_regs):
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.PEmpty),
                    number_of_threads=self.num_threads,
                )

            softmax.reset()
            acc_O.fill(0.0)
            kv_consumer_state = pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
            )

            mask = AttentionMask(
                self.tile_m,
                self.tile_n,
                seqlen,
                block_info.window_size_left,
                block_info.window_size_right,
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            )
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_tensors=aux_tensors,
                fastdiv_mods=None,
                mask_mod=self.mask_mod,
            )

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, self.num_splits
            )
            if n_block_max > n_block_min:
                kv_consumer_state = self.mma_one_n_block(
                    n_block_max - 1,
                    kv_consumer_state,
                    thr_mma_qk,
                    thr_mma_pv,
                    tSrQ,
                    tSrK,
                    tSrKRaw,
                    tOrVt,
                    tOrVtRaw,
                    acc_O,
                    sK,
                    sV,
                    sKRaw,
                    sVRaw,
                    smem_thr_copy_Q,
                    smem_thr_copy_K,
                    smem_thr_copy_KRaw,
                    smem_thr_copy_V,
                    smem_thr_copy_VRaw,
                    tSsQ,
                    tSsK,
                    tSsKRaw,
                    tOsVt,
                    tOsVtRaw,
                    pipeline_k,
                    pipeline_v,
                    softmax,
                    mFp8Lut,
                    seqlen,
                    batch_idx,
                    head_idx,
                    m_block,
                    partial(mask_fn, mask_seqlen=True),
                    aux_tensors=aux_tensors,
                    is_first_n_block=True,
                )
                n_block_max -= 1

                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        kv_consumer_state = self.mma_one_n_block(
                            n_block_max - 1 - n_tile,
                            kv_consumer_state,
                            thr_mma_qk,
                            thr_mma_pv,
                            tSrQ,
                            tSrK,
                            tSrKRaw,
                            tOrVt,
                            tOrVtRaw,
                            acc_O,
                            sK,
                            sV,
                            sKRaw,
                            sVRaw,
                            smem_thr_copy_Q,
                            smem_thr_copy_K,
                            smem_thr_copy_KRaw,
                            smem_thr_copy_V,
                            smem_thr_copy_VRaw,
                            tSsQ,
                            tSsK,
                            tSsKRaw,
                            tOsVt,
                            tOsVtRaw,
                            pipeline_k,
                            pipeline_v,
                            softmax,
                            mFp8Lut,
                            seqlen,
                            batch_idx,
                            head_idx,
                            m_block,
                            partial(mask_fn, mask_seqlen=False),
                            aux_tensors=aux_tensors,
                        )
                    n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)

                n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                )
                n_block_min_before_local_mask = cutlass.min(
                    n_block_min_before_local_mask, n_block_max
                )
                for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                    kv_consumer_state = self.mma_one_n_block(
                        n_block_max - 1 - n_tile,
                        kv_consumer_state,
                        thr_mma_qk,
                        thr_mma_pv,
                        tSrQ,
                        tSrK,
                        tSrKRaw,
                        tOrVt,
                        tOrVtRaw,
                        acc_O,
                        sK,
                        sV,
                        sKRaw,
                        sVRaw,
                        smem_thr_copy_Q,
                        smem_thr_copy_K,
                        smem_thr_copy_KRaw,
                        smem_thr_copy_V,
                        smem_thr_copy_VRaw,
                        tSsQ,
                        tSsK,
                        tSsKRaw,
                        tOsVt,
                        tOsVtRaw,
                        pipeline_k,
                        pipeline_v,
                        softmax,
                        mFp8Lut,
                        seqlen,
                        batch_idx,
                        head_idx,
                        m_block,
                        partial(mask_fn, mask_seqlen=False),
                        aux_tensors=aux_tensors,
                    )
                n_block_max = n_block_min_before_local_mask

                if const_expr(self.is_local and block_info.window_size_left is not None):
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        kv_consumer_state = self.mma_one_n_block(
                            n_block_max - 1 - n_tile,
                            kv_consumer_state,
                            thr_mma_qk,
                            thr_mma_pv,
                            tSrQ,
                            tSrK,
                            tSrKRaw,
                            tOrVt,
                            tOrVtRaw,
                            acc_O,
                            sK,
                            sV,
                            sKRaw,
                            sVRaw,
                            smem_thr_copy_Q,
                            smem_thr_copy_K,
                            smem_thr_copy_KRaw,
                            smem_thr_copy_V,
                            smem_thr_copy_VRaw,
                            tSsQ,
                            tSsK,
                            tSsKRaw,
                            tOsVt,
                            tOsVtRaw,
                            pipeline_k,
                            pipeline_v,
                            softmax,
                            mFp8Lut,
                            seqlen,
                            batch_idx,
                            head_idx,
                            m_block,
                            partial(mask_fn, mask_seqlen=False),
                            aux_tensors=aux_tensors,
                        )

            sink_val = None
            if const_expr(has_attention_sink_bias):
                sink_val = cute.make_fragment(softmax.num_rows, Float32)
                for r in cutlass.range_constexpr(softmax.num_rows):
                    if const_expr(self.pack_gqa):
                        row_idx = m_block * self.tile_m + taccOcO_mn[r, 0][0]
                        q_token_idx = row_idx // self.qhead_per_kvhead
                        q_head_in_group = row_idx - q_token_idx * self.qhead_per_kvhead
                        q_head_idx = head_idx * self.qhead_per_kvhead + q_head_in_group
                    else:
                        q_head_idx = head_idx
                    sink_val[r] = mAttentionSinkBias[q_head_idx]
            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                mVDescale,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                split_idx,
            )

            if const_expr(not self.Q_in_regs):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_threads,
                )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
