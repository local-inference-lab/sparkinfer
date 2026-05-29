"""Unified SM120 sparse-MLA *prefill* (P8) -- CORRECTNESS-FIRST, decode-reuse.

This is the P8/P8b prefill: a CORRECT single-pass DSV4 prefill built by REUSING
the proven, byte-identical decode pipeline (decode_math S0-S7 + io.py
io_issue_gather + smem) with the ABSOLUTE MINIMUM of new code. P8b SURGICALLY
scales the proven 1-IO/288 pipeline to the FlashInfer 4-IO/384 layout (8 math +
4 IO warps, io_threads=128, setmaxnreg dec/inc) for PREFILL PTX PARITY -- the
mbarrier full/empty protocol is KEPT BIT-IDENTICAL (the from-scratch 4-IO attempt
that built a NEW pipeline DEADLOCKED; this one changes ONLY the IO thread count +
register split, never the handshake).

KEY INSIGHT (why this works): the DSV4 decode CTA at ``num_splits=1`` ALREADY
processes ALL topk candidates for one query token in a single CTA, carrying the
online-softmax ``global_max``/``global_sum`` + ``acc_o`` across the chunk loop
(decode_math S4 does the per-chunk cross-warp reduce + cross-chunk acc rescale).
That IS a correct single-pass prefill. The ONLY differences vs decode are:

  (a) EPILOGUE: decode writes PER-SPLIT NORMALIZED partials to mid_out/mid_lse for
      the split.py merge; prefill wants the FINAL normalized BF16 O written
      directly to output[token, h, :] + a final base-2 LSE. This is exactly the
      ``epilogue_mode=FINAL_BF16`` branch added to ``decode_math.s7_epilogue``
      (the partial-writeback default keeps decode byte-identical).
  (b) attn_sink (optional) folded into the normalizer + LSE (FINAL_BF16 path).
  (c) PER-TOKEN variable ``topk_length`` -> the per-token ``section_len`` (the
      same runtime scalar the decode S3 mask + io gather already key off).
  (d) GRID is per query token: ``(num_tokens, H_BLOCKS, 1)`` (num_tokens may be
      >1; one CTA per (token, HPB head-group)).

So this prefill kernel is the decode CTA body (P8b: 384 threads = 8 math + 4 IO
warps, io_threads=128) with: a compile-time chunk loop of
``num_tiles = ceil(topk/BI)`` chunks, a PER-TOKEN ``section_len =
topk_length[token]`` (chunks past the token's length gather all -1 -> S3 masks
them -> they add nothing to the online softmax), and
``s7_epilogue(epilogue_mode=FINAL_BF16, attn_sink=...)``. The hot-op MMA PTX
(14 block-scaled + 14 plain e4m3 + 8 bf16) and the mbarrier handshake are
IDENTICAL to the decode kernel -- scaling 32->128 IO threads does NOT touch the
full/empty parity, so the proven deadlock-free pipeline stays deadlock-free.

SCOPE: DSV4, FP8 compute, main cache. DSV4 + GLM DECODE kernels stay byte-identical.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import from_dlpack

from b12x.attention._cute.ops import LOG2_E
from b12x.cute.compiler import (
    DimKey,
    KernelCompileSpec,
    key_field,
    launch as b12x_launch,
    tensor_key,
)
from b12x.cute.fp4 import shared_ptr_to_u32

from .decode_math import (
    EPILOGUE_FINAL_BF16,
    s0_quantize_q_to_smem,
    s0b_requant_k_glm,
    s1_qk_nope_block_scaled,
    s2_qk_rope_bf16,
    s3_mask_and_scale,
    s4_online_softmax,
    s5_fill_sm_p_full,
    s6_xv_nope,
    s6b_xv_rope,
    s7_epilogue,
)
from .io import io_issue_gather
from .smem import get_unified_shared_storage_cls, make_smem_layout
from .traits import infer_model_type, make_unified_traits


# BI=64 candidates per chunk (one full/empty KV buffer window). Same as decode.
_CAND_WINDOW = 64
# DSV4 compressed contract head dim (q_nope 448 + q_rope 64).
_DSV4_HEAD_DIM = 512

# P8b: 4-IO / 384-thread layout for FlashInfer prefill PTX parity. The decode
# traits pin block_threads=288 (1 IO warp); prefill overrides to 384 = 8 math
# warps (256, the CONSUMER) + 4 IO warps (128, the PRODUCER) so the gather is
# shared across 128 IO threads (io.py io_issue_gather io_threads=128). The
# mbarrier full/empty double-buffer protocol is KEPT BIT-IDENTICAL to the proven
# 1-IO pipeline; the ONLY semantic change is 4x more IO threads. setmaxnreg
# dec(32)/inc(232) matches FlashInfer prefill_kernel.cuh:126/161 (IO/math reg
# split). DECODE is untouched (its traits.block_threads stays 288).
_PREFILL_BLOCK_THREADS = 384  # 8 math (256) + 4 IO (128).
_PREFILL_IO_THREADS = 128     # 4 IO warps share one gather (io.py io_threads).
_IO_REGS = 32                 # setmaxnreg.dec on the IO warps.
_MATH_REGS = 232              # setmaxnreg.inc on the math warps.


class UnifiedPrefillKernel:
    """384-thread single-pass DSV4 prefill: ONE CTA per (token, HPB head-group).

    P8b: structurally the UnifiedDecodeKernel CTA body (8 math warps = CONSUMER +
    4 IO warps = PRODUCER, io_threads=128, the proven full/empty mbarrier
    double-buffer) run at the implicit ``num_splits=1``: the math loops over ALL
    ``num_tiles=ceil(topk/BI)`` chunks carrying
    ``global_max``/``global_sum``/``acc_o`` across the loop (the decode S4
    cross-chunk online softmax), then ``s7_epilogue`` writes the FINAL normalized
    BF16 O directly to output[token, head, :] + the final base-2 LSE (FINAL_BF16
    epilogue, with optional attn_sink fold).

    The 4-IO scale-up vs the proven 1-IO prefill is SURGICAL: block 288->384, the
    IO gather shared across 128 IO threads (io_threads=128, io_lane = tid-256),
    setmaxnreg dec(32)/inc(232) on the IO/math split. The mbarrier full/empty
    parity + arrive_expect_tx(leader) + try_wait(consumer) + math-only
    ``barrier(3, 256)`` are KEPT BIT-IDENTICAL to the 1-IO pipeline; the IO warps
    NEVER enter a 256-count named barrier (gap #8 deadlock guard). DECODE traits
    stay 288 -- the decode kernel is untouched.

    The PER-TOKEN ``topk_length[token]`` is the per-CTA ``section_len`` (a runtime
    scalar): the io gather clamps each chunk to ``g_end=min(g_start+BI, len)`` and
    stages -1 for out-of-range candidates, and S3 masks ``abs_cand >= section_len``
    -> chunks past the token's length contribute nothing. Grid = (num_tokens,
    h_blocks, 1).
    """

    def __init__(self, traits, layout, page_block_size, num_tiles,
                 num_tokens, h_blocks, num_heads, has_sink):
        self.traits = traits
        self.layout = layout
        self.page_block_size = int(page_block_size)
        self.num_tiles = int(num_tiles)  # ceil(topk / BI), compile-time chunk count.
        self.num_tokens = int(num_tokens)
        self.h_blocks = int(h_blocks)
        self.num_heads = int(num_heads)
        self.has_sink = bool(has_sink)
        self.math_threads = int(traits.math_threads)  # 256
        # P8b: prefill runs 384 threads (8 math + 4 IO), NOT the decode 288. The
        # decode traits.block_threads (288) is left untouched so the decode
        # kernel stays byte-identical; prefill pins 384 here.
        self.block_threads = _PREFILL_BLOCK_THREADS  # 384 (8 math + 4 IO)

    @cute.jit
    def __call__(
        self,
        q_all: cute.Tensor,          # (T, heads, D_QK) bf16
        kv_cache_u8: cute.Tensor,    # flat (pages*page_nbytes,) u8
        indices: cute.Tensor,        # (T, topk) int32
        topk_length: cute.Tensor,    # (T,) int32 per-token valid length
        attn_sink: cute.Tensor,      # (heads,) f32 (dummy 1-elem when no sink)
        output: cute.Tensor,         # (T, heads, D_V) bf16
        out_lse: cute.Tensor,        # (T, heads) f32 base-2 LSE
        sm_scale_log2: Float32,
        stride_kv_block: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_all, kv_cache_u8, indices, topk_length, attn_sink,
            output, out_lse, sm_scale_log2, stride_kv_block,
        ).launch(
            # Grid = (num_tokens, h_blocks, 1), one CTA per (token, HPB head-group).
            # These launchers trace with CONCRETE-shape tensors (compile_args ==
            # runtime_args), so the grid token dim is baked at trace time -- the
            # compile key MUST therefore distinguish ``num_tokens`` (keyed via
            # DimKey.exact on the q/output row dim + a num_tokens key_field) so a
            # cached T=1 kernel is NOT reused for a later T>1 call (which would
            # launch only token 0). h_blocks is keyed via num_heads.
            grid=(self.num_tokens, self.h_blocks, 1),
            block=[self.block_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        indices: cute.Tensor,
        topk_length: cute.Tensor,
        attn_sink: cute.Tensor,
        output: cute.Tensor,
        out_lse: cute.Tensor,
        sm_scale_log2: Float32,
        stride_kv_block: Int64,
    ):
        t = self.traits
        L = self.layout
        tid = Int32(cute.arch.thread_idx()[0])
        lane = cute.arch.lane_idx()
        warp_id = tid >> Int32(5)

        token_idx, head_block, _ = cute.arch.block_idx()
        token_idx = Int32(token_idx)
        head_block = Int32(head_block)
        head_base = head_block * Int32(t.hpb)

        smem = cutlass_utils.SmemAllocator()
        SharedStorage = get_unified_shared_storage_cls(t)
        st = smem.allocate(SharedStorage)

        q_fp8_addr = shared_ptr_to_u32(st.q_fp8.data_ptr())
        q_rope_addr = shared_ptr_to_u32(st.q_rope.data_ptr())
        kv_fp8_addr = shared_ptr_to_u32(st.kv_fp8.data_ptr())
        kv_sc_addr = shared_ptr_to_u32(st.kv_sc.data_ptr())
        kv_rope_addr = shared_ptr_to_u32(st.kv_rope.data_ptr())
        reduce_addr = shared_ptr_to_u32(st.reduce.data_ptr())
        reduce_max_addr = reduce_addr + Int32(L.reduce_warp_max_off - L.reduce_off)
        reduce_sum_addr = reduce_addr + Int32(L.reduce_warp_sum_off - L.reduce_off)
        w_fp8_addr = shared_ptr_to_u32(st.w_fp8.data_ptr())
        sm_p_full_addr = shared_ptr_to_u32(st.sm_p_full.data_ptr())

        q_sc_view = st.q_sc.get_tensor(cute.make_layout(int(L.q_sc_bytes // 4)))
        amax_view = st.reduce.get_tensor(cute.make_layout(int(L.reduce_bytes // 4)))
        token_idx_view = st.token_idx.get_tensor(
            cute.make_layout(int(L.token_idx_buf_bytes * L.token_idx_bufs // 4))
        )
        w_head_sc_view = st.w_head_sc.get_tensor(
            cute.make_layout(int(L.w_head_sc_bytes // 4))
        )

        # ── 384 threads = 8 math warps (CONSUMER, warps 0-7 = 256) + 4 IO warps
        #    (PRODUCER, warps 8-11 = 128). math_threads//32 == 8, so warp_id>=8
        #    selects exactly the 4 IO warps. The 256-count named barriers below
        #    EXCLUDE these IO warps (the gap #8 deadlock guard). ──
        is_io = warp_id >= Int32(self.math_threads // 32)

        kv_fp8_buf = Int32(L.kv_fp8_buf_bytes)
        kv_rope_buf = Int32(L.kv_rope_buf_bytes)
        kv_sc_buf = Int32(L.kv_sc_buf_bytes)
        tok_buf_elems = Int32(L.token_idx_buf_bytes // 4)

        # mbarrier array: full[0], full[1], empty[0], empty[1] (u64 each).
        mbar_base = st.mbar.data_ptr()
        n_buf = int(L.kv_bufs)

        if tid == Int32(0):
            for s in cutlass.range_constexpr(n_buf):
                cute.arch.mbarrier_init(mbar_base + s, Int32(1))           # full[s]
                cute.arch.mbarrier_init(mbar_base + n_buf + s, Int32(1))   # empty[s]
        cute.arch.barrier()  # full-CTA (288) structural fence.

        # PER-TOKEN valid length: this CTA's section_len (the decode mask + io
        # gather boundary). Clamped to [0, topk]. Chunks with g_start >= section_len
        # gather all -1 (io clamps g_end) and S3 masks them -> they add nothing.
        topk_total = Int32(indices.shape[1])
        section_len = Int32(topk_length[token_idx])
        if section_len < Int32(0):
            section_len = Int32(0)
        if section_len > topk_total:
            section_len = topk_total

        # indices for THIS token row (1-D (topk,) slice).
        topk_row = cute.make_tensor(
            indices.iterator + token_idx.to(Int64) * Int64(indices.stride[0]),
            cute.make_layout(indices.shape[1]),
        )
        # q for THIS token (2-D (heads, D_QK) view; s0 indexes [head_base+h, d]).
        q_token = cute.make_tensor(
            q_all.iterator + token_idx.to(Int64) * Int64(q_all.stride[0]),
            cute.make_layout(
                (q_all.shape[1], q_all.shape[2]),
                stride=(q_all.stride[1], q_all.stride[2]),
            ),
        )
        warp_first_cand = warp_id * Int32(8)

        # ════════════════════════════════════════════════════════════════════
        # IO WARP (PRODUCER) vs MATH WARPS (CONSUMER). EXACT decode protocol.
        # ════════════════════════════════════════════════════════════════════
        if is_io:
            # setmaxnreg.dec(32): release registers on the 4 IO warps so the math
            # warps can claim 232 (FlashInfer prefill_kernel.cuh:126). Perf/parity
            # only; the gather is register-light.
            cute.arch.setmaxregister_decrease(_IO_REGS)
            # io_lane is the lane within the WHOLE IO group [0, 128) (4 warps), NOT
            # the per-warp lane: io_issue_gather strides BI=64 entries across all
            # 128 IO threads (1 pass) and the leader is io_lane==0 (tid==256).
            io_lane = tid - Int32(self.math_threads)  # [0, 128)
            prod_phase = Int32(1)
            prod_idx = Int32(0)
            for lc in cutlass.range(self.num_tiles, unroll=1):
                ci = Int32(lc)
                buf = Int32(lc) & Int32(1)
                g_start = ci * Int32(_CAND_WINDOW)
                g_end = g_start + Int32(_CAND_WINDOW)
                if g_end > section_len:
                    g_end = section_len

                cute.arch.mbarrier_wait(mbar_base + n_buf + prod_idx, phase=prod_phase)

                tok_buf_view = cute.make_tensor(
                    token_idx_view.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )
                io_issue_gather(
                    kv_cache_u8, topk_row,
                    kv_fp8_addr + buf * kv_fp8_buf,
                    kv_rope_addr + buf * kv_rope_buf,
                    kv_sc_addr + buf * kv_sc_buf,
                    tok_buf_view,
                    mbar_base + buf,  # full[buf]
                    g_start, g_end,
                    Int32(self.page_block_size), stride_kv_block, io_lane,
                    bi=t.bi, kv_smem_stride=t.kv_smem_stride, rope_smem_stride=t.d_rope,
                    scale_bytes_per_token=8, bulk_tx_bytes=t.bulk_tx_bytes,
                    scale_format=t.scale_format, io_threads=_PREFILL_IO_THREADS,
                )
                prod_idx += Int32(1)
                if prod_idx == Int32(n_buf):
                    prod_idx = Int32(0)
                    prod_phase ^= Int32(1)

        else:
            # MATH WARPS (CONSUMER, warps 0-7 = 256 threads).
            # setmaxnreg.inc(232): claim the registers the IO warps released
            # (FlashInfer prefill_kernel.cuh:161). Perf/parity only.
            cute.arch.setmaxregister_increase(_MATH_REGS)
            n_acc_tiles = int(t.n_v_chunks) * int(t.nt_per_warp_xv)
            s0_quantize_q_to_smem(
                q_token, q_fp8_addr, q_sc_view, q_rope_addr, amax_view,
                head_base, Int32(t.hpb), tid,
                d_nope=t.d_nope, d_rope=t.d_rope, d_qk=t.d_nope + t.d_rope,
                quant_tile=t.quant_tile, num_scales=t.num_scales, hpb=t.hpb,
                q_nope_stride=t.q_nope_stride, num_threads=self.math_threads, barrier_id=2,
            )

            accn_frag = cute.make_rmem_tensor(n_acc_tiles * 4, Float32)
            accr_frag = cute.make_rmem_tensor(4, Float32)
            gmax_frag = cute.make_rmem_tensor(2, Float32)
            gsum_frag = cute.make_rmem_tensor(2, Float32)
            for k in cutlass.range_constexpr(n_acc_tiles * 4):
                accn_frag[k] = Float32(0.0)
            for k in cutlass.range_constexpr(4):
                accr_frag[k] = Float32(0.0)
            gmax_frag[0] = Float32(-1e30); gmax_frag[1] = Float32(-1e30)
            gsum_frag[0] = Float32(0.0); gsum_frag[1] = Float32(0.0)

            cons_phase = Int32(0)
            cons_idx = Int32(0)

            for lc in cutlass.range(self.num_tiles, unroll=1):
                ci = Int32(lc)
                split_cand_start = ci * Int32(_CAND_WINDOW)
                buf = Int32(lc) & Int32(1)

                kv_fp8_b = kv_fp8_addr + buf * kv_fp8_buf
                kv_rope_b = kv_rope_addr + buf * kv_rope_buf
                kv_sc_b = kv_sc_addr + buf * kv_sc_buf
                tok_buf_view = cute.make_tensor(
                    token_idx_view.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )

                acc_nope = [
                    [accn_frag[at * 4 + 0], accn_frag[at * 4 + 1],
                     accn_frag[at * 4 + 2], accn_frag[at * 4 + 3]]
                    for at in range(n_acc_tiles)
                ]
                acc_rope = [accr_frag[0], accr_frag[1], accr_frag[2], accr_frag[3]]
                global_max = [gmax_frag[0], gmax_frag[1]]
                global_sum = [gsum_frag[0], gsum_frag[1]]

                cute.arch.mbarrier_wait(mbar_base + cons_idx, phase=cons_phase)
                cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)

                # GLM-only S0b (const_expr-elided for DSV4). Kept for symmetry with
                # the decode kernel; DSV4 prefill never enters it.
                if cutlass.const_expr(t.scale_format == 1):
                    s0b_requant_k_glm(
                        kv_fp8_b, tid,
                        bi=t.bi, d_nope=t.d_nope, quant_tile=t.quant_tile,
                        kv_smem_stride=t.kv_smem_stride,
                        num_threads=self.math_threads, barrier_id=3,
                    )

                qk = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                qk = s1_qk_nope_block_scaled(
                    qk, q_fp8_addr, kv_fp8_b, q_sc_view, kv_sc_b,
                    warp_first_cand, lane,
                    num_scales=t.num_scales, quant_tile=t.quant_tile,
                    q_nope_stride=t.q_nope_stride, kv_smem_stride=t.kv_smem_stride,
                    scale_bytes_per_token=8, scale_format=t.scale_format,
                )
                qk = s2_qk_rope_bf16(
                    qk, q_rope_addr, kv_rope_b, warp_first_cand, lane, d_rope=t.d_rope,
                )

                # PER-TOKEN mask: invalid if idx<0 OR abs_cand >= section_len. The
                # single CTA owns the whole row so split_cand_end == section_len.
                split_cand_end = split_cand_start + Int32(_CAND_WINDOW)
                if split_cand_end > section_len:
                    split_cand_end = section_len
                qk = s3_mask_and_scale(
                    qk, tok_buf_view, warp_first_cand,
                    split_cand_start, split_cand_end, section_len,
                    sm_scale_log2, lane,
                )

                p = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                p, wr0, wr1 = s4_online_softmax(
                    qk, p, acc_nope, acc_rope, global_max, global_sum,
                    reduce_max_addr, reduce_sum_addr, False,
                    warp_id, lane, tid,
                    n_v_chunks=t.n_v_chunks, hpb=t.hpb, n_warps=8, valid_hpb=t.hpb,
                    num_threads=self.math_threads, barrier_id=3,
                    n_acc_tiles=n_acc_tiles,
                )
                w_pre = [p[0] * wr0, p[1] * wr0, p[2] * wr1, p[3] * wr1]

                s5_fill_sm_p_full(
                    w_pre, sm_p_full_addr, w_head_sc_view, warp_id, lane, tid,
                    bi=t.bi, n_v_chunks=t.n_v_chunks, hpb=t.hpb,
                    num_threads=self.math_threads, barrier_id=3,
                )
                cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)

                acc_nope = s6_xv_nope(
                    w_pre, acc_nope, kv_fp8_b, kv_sc_b, w_head_sc_view, w_fp8_addr,
                    warp_id, lane, tid,
                    n_v_chunks=t.n_v_chunks, v_chunk=t.quant_tile, hpb=t.hpb, bi=t.bi,
                    kv_smem_stride=t.kv_smem_stride, w_fp8_stride=t.bi + 16, n_warps=8,
                    scale_bytes_per_token=8, nt_per_warp_xv=t.nt_per_warp_xv,
                    scale_format=t.scale_format,
                    num_threads=self.math_threads, barrier_id=3,
                )

                if cutlass.const_expr(t.v_has_rope):
                    acc_rope = s6b_xv_rope(
                        acc_rope, sm_p_full_addr, kv_rope_b, warp_id, lane,
                        bi=t.bi, d_rope=t.d_rope, n_warps=8,
                    )

                for at in cutlass.range_constexpr(n_acc_tiles):
                    accn_frag[at * 4 + 0] = acc_nope[at][0]
                    accn_frag[at * 4 + 1] = acc_nope[at][1]
                    accn_frag[at * 4 + 2] = acc_nope[at][2]
                    accn_frag[at * 4 + 3] = acc_nope[at][3]
                accr_frag[0] = acc_rope[0]; accr_frag[1] = acc_rope[1]
                accr_frag[2] = acc_rope[2]; accr_frag[3] = acc_rope[3]
                gmax_frag[0] = global_max[0]; gmax_frag[1] = global_max[1]
                gsum_frag[0] = global_sum[0]; gsum_frag[1] = global_sum[1]

                cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)
                if tid == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + n_buf + cons_idx)
                cons_idx += Int32(1)
                if cons_idx == Int32(n_buf):
                    cons_idx = Int32(0)
                    cons_phase ^= Int32(1)

            # ── S7 FINAL_BF16: write the final normalized BF16 O directly into
            #    output[token, head_base + h, dim] + the final base-2 LSE (with
            #    optional attn_sink fold). NO merge. ──
            fin_acc_nope = [
                [accn_frag[at * 4 + 0], accn_frag[at * 4 + 1],
                 accn_frag[at * 4 + 2], accn_frag[at * 4 + 3]]
                for at in range(n_acc_tiles)
            ]
            fin_acc_rope = [accr_frag[0], accr_frag[1], accr_frag[2], accr_frag[3]]
            fin_gmax = [gmax_frag[0], gmax_frag[1]]
            fin_gsum = [gsum_frag[0], gsum_frag[1]]

            # output[token, head_base + h, dim]: (HPB, D_V) view for this
            # (token, head_block). output stride = (heads*Dv, Dv, 1).
            out_o = cute.make_tensor(
                output.iterator
                + token_idx.to(Int64) * Int64(output.stride[0])
                + head_base.to(Int64) * Int64(output.stride[1]),
                cute.make_layout(
                    (t.hpb, t.d_v),
                    stride=(output.stride[1], output.stride[2]),
                ),
            )
            # out_lse[token, head_base + h]: (HPB,) view.
            out_lse_v = cute.make_tensor(
                out_lse.iterator
                + token_idx.to(Int64) * Int64(out_lse.stride[0])
                + head_base.to(Int64) * Int64(out_lse.stride[1]),
                cute.make_layout((t.hpb,), stride=(out_lse.stride[1],)),
            )
            s7_epilogue(
                fin_acc_nope, fin_acc_rope, fin_gmax, fin_gsum, out_o, out_lse_v,
                warp_id, lane,
                n_v_chunks=t.n_v_chunks, v_chunk=t.quant_tile, d_nope=t.d_nope,
                d_rope=t.d_rope, n_warps=8, valid_hpb=t.hpb,
                nt_per_warp_xv=t.nt_per_warp_xv, v_has_rope=t.v_has_rope,
                epilogue_mode=EPILOGUE_FINAL_BF16,
                has_attn_sink=self.has_sink, attn_sink=attn_sink, head_base=head_base,
            )


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------
def _to_cute(x, dtype, align=16):
    c = from_dlpack(x, assumed_align=align)
    c.element_type = dtype
    return c


def _topk_bucket(topk: int) -> int:
    return 1 << (max(int(topk), 1) - 1).bit_length()


def run_unified_prefill(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    page_block_size: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    lse_out: torch.Tensor | None = None,
    stride_kv_block: int | None = None,
    workspace=None,
):
    """Unified SM120 sparse-MLA DSV4 prefill: single-pass kernel -> BF16 O + LSE.

    A THIN launcher over ``UnifiedPrefillKernel`` (the decode CTA body run at the
    implicit num_splits=1, with the FINAL_BF16 epilogue + per-token section_len +
    optional attn_sink). P8b: 4 IO warps, 384 threads (8 math + 4 IO), the proven
    1-IO mbarrier protocol scaled to 128 IO threads + setmaxnreg dec/inc.

    Args:
      q:            (T, heads, D_QK=512) bf16. T query tokens (prefill regime).
      kv_cache:     flat uint8 KV cache (reshaped to 1-D). For DSV4 the per-page
                    byte stride is ``stride_kv_block`` (compressed page nbytes).
      topk_indices: (T, topk) int32 flat slot ids (-1 = invalid sentinel).
      sm_scale:     softmax scale (typically D_QK**-0.5).
      page_block_size: tokens per KV block (64 for DSV4).
      topk_length:  optional (T,) int32 per-token valid length; entries past it
                    are masked. Defaults to full ``topk`` for every token.
      attn_sink:    optional (heads,) fp32 per-head natural-log sink, folded into
                    the normalizer + base-2 LSE (FlashMLA V4).
      output:       optional pre-allocated (T, heads, D_V) bf16 output (else made).
      lse_out:      optional pre-allocated (T, heads) f32 base-2 LSE (else made).
      stride_kv_block: per-block gmem byte stride. Derived from page_block_size
                    when omitted (compressed_mla_page_nbytes for DSV4).
      workspace:    unused (prefill is single-pass, no split/merge workspace);
                    accepted for launcher-signature symmetry.

    Returns (O[T, heads, D_V=512] bf16, lse[T, heads] f32 base-2).
    """
    from b12x.attention.mla.compressed_reference import compressed_mla_page_nbytes

    del workspace  # prefill is single-pass; no split/merge workspace needed.

    q_head_dim = int(q.shape[-1])
    if q_head_dim != _DSV4_HEAD_DIM:
        raise NotImplementedError(
            f"unified_sm120 prefill supports DSV4 (q_head_dim=512) only; got {q_head_dim}"
        )

    num_tokens, heads, _ = q.shape
    hpb = 16
    if heads % hpb != 0:
        raise NotImplementedError(
            f"unified_sm120 prefill requires heads divisible by HPB={hpb}, got {heads}"
        )
    h_blocks = heads // hpb

    model_type, compute_mode, scale_format = infer_model_type(q_head_dim, kv_cache.dtype)
    traits = make_unified_traits(model_type, compute_mode, scale_format)
    layout = make_smem_layout(traits)
    d_v = int(traits.d_v)

    topk = int(topk_indices.shape[1])
    num_tiles = (topk + _CAND_WINDOW - 1) // _CAND_WINDOW

    device = q.device
    if topk_length is None:
        topk_length = torch.full((num_tokens,), topk, dtype=torch.int32, device=device)
    else:
        topk_length = topk_length.to(device=device, dtype=torch.int32).contiguous()

    has_sink = attn_sink is not None
    if has_sink:
        attn_sink_t = attn_sink.to(device=device, dtype=torch.float32).contiguous()
    else:
        # dummy 1-elem tensor so the kernel arg exists; never read (const_expr gate).
        attn_sink_t = torch.zeros(1, dtype=torch.float32, device=device)

    if stride_kv_block is None:
        stride_kv_block = int(compressed_mla_page_nbytes(int(page_block_size)))

    q = q.contiguous()
    topk_indices = topk_indices.contiguous()
    if output is None:
        output = torch.empty((num_tokens, heads, d_v), dtype=torch.bfloat16, device=device)
    if lse_out is None:
        lse_out = torch.empty((num_tokens, heads), dtype=torch.float32, device=device)

    kernel = UnifiedPrefillKernel(
        traits, layout, int(page_block_size), num_tiles,
        num_tokens=num_tokens, h_blocks=h_blocks, num_heads=heads, has_sink=has_sink,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kv_flat = kv_cache.reshape(-1)
    args = (
        _to_cute(q, cutlass.BFloat16),
        _to_cute(kv_flat, cutlass.Uint8, align=16),
        _to_cute(topk_indices, cutlass.Int32, align=4),
        _to_cute(topk_length, cutlass.Int32, align=4),
        _to_cute(attn_sink_t, cutlass.Float32, align=4),
        _to_cute(output, cutlass.BFloat16, align=16),
        _to_cute(lse_out, cutlass.Float32, align=4),
        Float32(float(sm_scale) * LOG2_E),
        Int64(stride_kv_block),
        stream,
    )
    compile_spec = KernelCompileSpec.from_fields(
        "attention.mla.unified_sm120.prefill",
        # version 3: P8b 4-IO/384-thread scale-up (8 math + 4 IO, io_threads=128,
        # setmaxnreg dec/inc) over the proven v2 1-IO mbarrier protocol. The cache
        # key is kernel_id + version + fields only (NOT a source hash), so this
        # bump invalidates any stale v2 288-thread prefill objects on disk.
        3,
        key_field("model_type", traits.model_type),
        key_field("compute_mode", traits.compute_mode),
        key_field("scale_format", traits.scale_format),
        key_field("num_heads", int(heads)),
        key_field("hpb", int(hpb)),
        key_field("num_tiles", int(num_tiles)),
        key_field("page_block_size", int(page_block_size)),
        key_field("topk_bucket", _topk_bucket(topk)),
        key_field("has_sink", int(has_sink)),
        # num_tokens is baked into the launch grid (concrete-shape trace), so it
        # MUST be a compile key (DimKey.exact row dims below). A T-bucket here
        # would silently reuse a wrong-grid kernel; key the exact T.
        key_field("num_tokens", int(num_tokens)),
        tensor_key("q", q, dims=(DimKey.exact(num_tokens), DimKey.exact(heads), DimKey.exact(q_head_dim))),
        tensor_key("topk_indices", topk_indices, dims=(DimKey.exact(num_tokens), DimKey.bucket(topk))),
        tensor_key("output", output, dims=(DimKey.exact(num_tokens), DimKey.exact(heads), DimKey.exact(d_v))),
        tensor_key("out_lse", lse_out, dims=(DimKey.exact(num_tokens), DimKey.exact(heads))),
    )
    b12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )
    return output, lse_out
