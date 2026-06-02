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
    tensor_key,
)
from b12x.cute.compiler import (
    launch as b12x_launch,
)
from b12x.cute.fp4 import shared_ptr_to_u32

from .decode_math import (
    EPILOGUE_FINAL_BF16,
    s0_quantize_q_to_smem,
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
from .traits import ModelType, infer_model_type, make_unified_traits

# BI=64 candidates per chunk (one full/empty KV buffer window). Same as decode.
_CAND_WINDOW = 64
# DSV4 compressed contract head dim (q_nope 448 + q_rope 64).
_DSV4_HEAD_DIM = 512
# GLM_NSA uncompressed contract head dim (q_nope 512 + q_rope 64).
_GLM_HEAD_DIM = 576
# GLM per-token packed cache record (reference.pack_mla_kv_cache_reference).
_GLM_KV_GMEM_STRIDE = 656

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
                 h_blocks, num_heads, q_head_dim, topk, extra_topk, has_sink,
                 q_stride, indices_stride0, extra_indices_stride0,
                 output_stride, out_lse_stride,
                 has_extra=False, pbs_extra=1, num_main_tiles=0):
        self.traits = traits
        self.layout = layout
        self.page_block_size = int(page_block_size)
        self.num_tiles = int(num_tiles)  # ceil(topk / BI), compile-time chunk count.
        self.h_blocks = int(h_blocks)
        self.num_heads = int(num_heads)
        self.q_head_dim = int(q_head_dim)
        self.topk = int(topk)
        self.extra_topk = int(extra_topk)
        self.q_stride_row = int(q_stride[0])
        self.q_stride_head = int(q_stride[1])
        self.q_stride_dim = int(q_stride[2])
        self.indices_stride_row = int(indices_stride0)
        self.extra_indices_stride_row = int(extra_indices_stride0)
        self.output_stride_row = int(output_stride[0])
        self.output_stride_head = int(output_stride[1])
        self.output_stride_dim = int(output_stride[2])
        self.out_lse_stride_row = int(out_lse_stride[0])
        self.out_lse_stride_head = int(out_lse_stride[1])
        self.has_sink = bool(has_sink)
        # DSV4 dual-cache prefill (P10 3c). When False the extra-section code is
        # const_expr-elided -> no-extra DSV4 / GLM prefill PTX byte-identical. The
        # union spans num_main_tiles main chunks (gathered from the main cache) then
        # the remaining (num_tiles - num_main_tiles) extra chunks (gathered from the
        # extra cache), exactly like the decode has_extra union.
        self.has_extra = bool(has_extra)
        self.pbs_extra = int(pbs_extra)
        self.num_main_tiles = int(num_main_tiles)
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
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        # SINGLE-CACHE entry (DSV4 main OR GLM): EXACTLY the original traced
        # signature (9 data args + stream). The dispatcher selects this (func=kernel
        # -> __call__) when has_extra=False so the no-extra DSV4 / GLM trace +
        # mangled name + launched @cute.kernel stay byte-identical: the extra-section
        # args never enter the device entry.
        self.kernel(
            q_all, kv_cache_u8, indices, topk_length, attn_sink,
            output, out_lse, sm_scale_log2, stride_kv_block,
        ).launch(
            # Grid = (num_tokens, h_blocks, 1), one CTA per token/head block.
            # The token count is a live launch-grid value, not kernel binary
            # identity. The compile key below keeps row dims dynamic and only
            # specializes model/tile/cache settings.
            grid=(num_tokens, self.h_blocks, 1),
            block=[self.block_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def call_extra(
        self,
        q_all: cute.Tensor,          # (T, heads, D_QK) bf16
        kv_cache_u8: cute.Tensor,    # flat u8 MAIN cache
        indices: cute.Tensor,        # (T, topk) int32 MAIN indices
        topk_length: cute.Tensor,    # (T,) int32 per-token MAIN valid length
        attn_sink: cute.Tensor,      # (heads,) f32 (dummy 1-elem when no sink)
        output: cute.Tensor,         # (T, heads, D_V) bf16
        out_lse: cute.Tensor,        # (T, heads) f32 base-2 LSE
        sm_scale_log2: Float32,
        stride_kv_block: Int64,      # MAIN per-block byte stride
        extra_kv_cache_u8: cute.Tensor,  # flat u8 EXTRA cache (DSV4 dual-cache)
        extra_indices: cute.Tensor,      # (T, extra_topk) int32
        extra_topk_length: cute.Tensor,  # (T,) int32 per-token EXTRA valid length
        stride_extra_kv_block: Int64,    # EXTRA per-block byte stride
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        # DUAL-CACHE entry (DSV4 prefill P10 3c): the dispatcher selects this
        # (func=kernel.call_extra) only when has_extra=True, so its DISTINCT mangled
        # name never collides with the byte-identical single-cache __call__. It
        # launches the 13-param @cute.kernel (self.kernel_extra), which shares the
        # body via _prefill_body(has_extra=True). num_main_tiles (the uniform
        # main/extra chunk split) is a compile-time self.num_main_tiles.
        self.kernel_extra(
            q_all, kv_cache_u8, indices, topk_length, attn_sink,
            output, out_lse, sm_scale_log2, stride_kv_block,
            extra_kv_cache_u8, extra_indices, extra_topk_length,
            stride_extra_kv_block,
        ).launch(
            grid=(num_tokens, self.h_blocks, 1),
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
        # SINGLE-CACHE @cute.kernel (9 device params): the byte-identical DSV4-main /
        # GLM prefill path. Threads dummy extra args into the shared body with
        # has_extra=False so the extra-section code is fully const_expr-elided.
        self._prefill_body(
            q_all, kv_cache_u8, indices, topk_length, attn_sink,
            output, out_lse, sm_scale_log2, stride_kv_block,
            kv_cache_u8, indices, topk_length, stride_kv_block,
            has_extra=False,
        )

    @cute.kernel
    def kernel_extra(
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
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_topk_length: cute.Tensor,
        stride_extra_kv_block: Int64,
    ):
        # DUAL-CACHE @cute.kernel (13 device params): threads the real extra-section
        # args into the shared body (has_extra=True).
        self._prefill_body(
            q_all, kv_cache_u8, indices, topk_length, attn_sink,
            output, out_lse, sm_scale_log2, stride_kv_block,
            extra_kv_cache_u8, extra_indices, extra_topk_length,
            stride_extra_kv_block,
            has_extra=True,
        )

    @cute.jit
    def _prefill_body(
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
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_topk_length: cute.Tensor,
        stride_extra_kv_block: Int64,
        *,
        has_extra: cutlass.Constexpr,
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
        topk_total = Int32(self.topk)
        section_len = Int32(topk_length[token_idx])
        if section_len < Int32(0):
            section_len = Int32(0)
        if section_len > topk_total:
            section_len = topk_total

        # DSV4 dual-cache: PER-TOKEN EXTRA section length (the union's second pool).
        # const_expr-elided when has_extra=False so the no-extra trace is unchanged.
        num_main_tiles = Int32(self.num_main_tiles)
        if cutlass.const_expr(has_extra):
            extra_total = Int32(self.extra_topk)
            extra_section_len = Int32(extra_topk_length[token_idx])
            if extra_section_len < Int32(0):
                extra_section_len = Int32(0)
            if extra_section_len > extra_total:
                extra_section_len = extra_total
        else:
            extra_section_len = section_len

        # EMPTY-ROW flag: a (token, head-block) with NO valid candidates -- the
        # per-token section_len (main + any extra) is 0. Used by the FINAL_BF16
        # epilogue to force O=0 + LSE=-inf (a row with no keys has no attention
        # output). Detect it HERE from the section length(s), NOT from a magic
        # global_max threshold: the all-masked online softmax leaves global_max at
        # the FINITE _QK_MASK sentinel (-1e30 * sm_scale_log2), whose magnitude
        # scales with sm_scale, so no fixed cutoff cleanly separates it from a real
        # (small) qk. The section length IS the exact contract.
        is_empty_row = section_len == Int32(0)
        if cutlass.const_expr(has_extra):
            is_empty_row = is_empty_row and (extra_section_len == Int32(0))

        # indices for THIS token row (1-D (topk,) slice).
        if cutlass.const_expr(self.topk == 0):
            topk_row = cute.make_tensor(
                indices.iterator
                + token_idx.to(Int64) * Int64(self.indices_stride_row),
                cute.make_layout(1),
            )
        else:
            topk_row = cute.make_tensor(
                indices.iterator
                + token_idx.to(Int64) * Int64(self.indices_stride_row),
                cute.make_layout(self.topk),
            )
        # extra_indices for THIS token row (DSV4 dual-cache). Built ONLY when
        # has_extra; const_expr-elided so the no-extra trace never references it.
        if cutlass.const_expr(has_extra):
            extra_row = cute.make_tensor(
                extra_indices.iterator
                + token_idx.to(Int64) * Int64(self.extra_indices_stride_row),
                cute.make_layout(self.extra_topk),
            )
        else:
            extra_row = topk_row
        # q for THIS token (2-D (heads, D_QK) view; s0 indexes [head_base+h, d]).
        q_token = cute.make_tensor(
            q_all.iterator + token_idx.to(Int64) * Int64(self.q_stride_row),
            cute.make_layout(
                (self.num_heads, self.q_head_dim),
                stride=(self.q_stride_head, self.q_stride_dim),
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
            io_kw = dict(
                bi=t.bi, kv_smem_stride=t.kv_smem_stride, rope_smem_stride=t.d_rope,
                scale_bytes_per_token=8, bulk_tx_bytes=t.bulk_tx_bytes,
                scale_format=t.scale_format, io_threads=_PREFILL_IO_THREADS,
            )
            for lc in cutlass.range(self.num_tiles, unroll=1):
                ci = Int32(lc)
                buf = Int32(lc) & Int32(1)

                cute.arch.mbarrier_wait(mbar_base + n_buf + prod_idx, phase=prod_phase)

                tok_buf_view = cute.make_tensor(
                    token_idx_view.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )
                # Per-chunk section dispatch (DSV4 dual-cache; mirrors decode
                # _kernel_body). chunks [0, num_main_tiles) gather from the MAIN
                # cache; chunks >= num_main_tiles re-base their offset and gather from
                # the EXTRA cache (its own base ptr / page size / indices / stride).
                # const_expr-pinned to the main gather when has_extra=False -> the
                # no-extra trace + PTX are byte-identical.
                if cutlass.const_expr(has_extra):
                    if ci >= num_main_tiles:
                        cis = ci - num_main_tiles
                        g_start = cis * Int32(_CAND_WINDOW)
                        g_end = g_start + Int32(_CAND_WINDOW)
                        if g_end > extra_section_len:
                            g_end = extra_section_len
                        io_issue_gather(
                            extra_kv_cache_u8, extra_row,
                            kv_fp8_addr + buf * kv_fp8_buf,
                            kv_rope_addr + buf * kv_rope_buf,
                            kv_sc_addr + buf * kv_sc_buf,
                            tok_buf_view,
                            mbar_base + buf,
                            g_start, g_end,
                            Int32(self.pbs_extra), stride_extra_kv_block, io_lane,
                            **io_kw,
                        )
                    else:
                        g_start = ci * Int32(_CAND_WINDOW)
                        g_end = g_start + Int32(_CAND_WINDOW)
                        if g_end > section_len:
                            g_end = section_len
                        io_issue_gather(
                            kv_cache_u8, topk_row,
                            kv_fp8_addr + buf * kv_fp8_buf,
                            kv_rope_addr + buf * kv_rope_buf,
                            kv_sc_addr + buf * kv_sc_buf,
                            tok_buf_view,
                            mbar_base + buf,
                            g_start, g_end,
                            Int32(self.page_block_size), stride_kv_block, io_lane,
                            **io_kw,
                        )
                else:
                    g_start = ci * Int32(_CAND_WINDOW)
                    g_end = g_start + Int32(_CAND_WINDOW)
                    if g_end > section_len:
                        g_end = section_len
                    io_issue_gather(
                        kv_cache_u8, topk_row,
                        kv_fp8_addr + buf * kv_fp8_buf,
                        kv_rope_addr + buf * kv_rope_buf,
                        kv_sc_addr + buf * kv_sc_buf,
                        tok_buf_view,
                        mbar_base + buf,  # full[buf]
                        g_start, g_end,
                        Int32(self.page_block_size), stride_kv_block, io_lane,
                        **io_kw,
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

                # P10f: GLM keeps RAW e4m3 K/V (no S0b dequant+requant); the
                # arbitrary fp32 group scale is applied POST-MMA in S1 (QK) / inline
                # in S6 (V). DSV4 prefill never ran S0b (scale_format==0) so its
                # trace/PTX stay byte-identical.

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
                # DSV4 dual-cache: an extra chunk (ci >= num_main_tiles) re-bases the
                # candidate offset WITHIN ITS SECTION and swaps in the extra section
                # length; the MATH (S0-S6b) reads only the buffered smem, so it is
                # section-agnostic. const_expr-pinned to the main expressions when
                # has_extra=False -> the no-extra trace + PTX are byte-identical.
                if cutlass.const_expr(has_extra):
                    sc_start = split_cand_start
                    sec_len = section_len
                    if ci >= num_main_tiles:
                        sc_start = (ci - num_main_tiles) * Int32(_CAND_WINDOW)
                        sec_len = extra_section_len
                    sc_end = sc_start + Int32(_CAND_WINDOW)
                    if sc_end > sec_len:
                        sc_end = sec_len
                    qk = s3_mask_and_scale(
                        qk, tok_buf_view, warp_first_cand,
                        sc_start, sc_end, sec_len, sm_scale_log2, lane,
                    )
                else:
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

            # EMPTY-ROW guard. A row with no valid candidates (section_len==0, and
            # for dual-cache extra_section_len==0 too) leaves EVERY qk at the FINITE
            # _QK_MASK sentinel (-1e30 * sm_scale_log2). s4 then computes
            # p = exp2(qk - local_max) = exp2(0) = 1 for those masked candidates, so
            # global_sum becomes a SPURIOUS positive (== masked-candidate count) and
            # global_max sits at the sentinel level. The warp_rescale trick only
            # cancels those spurious 1s when SOME warp holds a real (higher) max; for
            # a fully-empty row no warp is valid, so nothing cancels it and S7's
            # `global_sum > 0` empty-guard never fires -> garbage normalized output
            # (the zero-length extend row that lands cos~0). Force global_sum=0 so S7
            # takes its inv_g=0 path -> O=0 + LSE=-inf, the correct no-valid-keys
            # result. PREFILL-ONLY: s4/s7 and the decode path are untouched (decode
            # merges per-split via the -inf LSE sentinel; this single-pass prefill
            # has no merge to consume it). NON-empty rows are bit-unchanged.
            if is_empty_row:
                fin_gsum[0] = Float32(0.0)
                fin_gsum[1] = Float32(0.0)

            # output[token, head_base + h, dim]: (HPB, D_V) view for this
            # (token, head_block). output stride = (heads*Dv, Dv, 1).
            out_o = cute.make_tensor(
                output.iterator
                + token_idx.to(Int64) * Int64(self.output_stride_row)
                + head_base.to(Int64) * Int64(self.output_stride_head),
                cute.make_layout(
                    (t.hpb, t.d_v),
                    stride=(self.output_stride_head, self.output_stride_dim),
                ),
            )
            # out_lse[token, head_base + h]: (HPB,) view.
            out_lse_v = cute.make_tensor(
                out_lse.iterator
                + token_idx.to(Int64) * Int64(self.out_lse_stride_row)
                + head_base.to(Int64) * Int64(self.out_lse_stride_head),
                cute.make_layout((t.hpb,), stride=(self.out_lse_stride_head,)),
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
def _to_cute(x, dtype, align=16, dynamic_layout=False):
    c = from_dlpack(x, assumed_align=align)
    c.element_type = dtype
    if dynamic_layout and x.ndim >= 1:
        leading_dim = next(
            (idx for idx, stride in enumerate(x.stride()) if stride == 1), None
        )
        if leading_dim is not None:
            c = c.mark_layout_dynamic(leading_dim=leading_dim)
    return c


def _topk_bucket(topk: int) -> int:
    return 1 << (max(int(topk), 1) - 1).bit_length()


def _unified_sm120_prefill_flat_launch(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink_t: torch.Tensor,
    output: torch.Tensor,
    lse_out: torch.Tensor,
    extra_kv_flat: torch.Tensor,
    extra_indices_t: torch.Tensor,
    extra_len_t: torch.Tensor,
    sm_scale: float,
    model_type: int,
    compute_mode: int,
    scale_format: int,
    page_block_size: int,
    topk: int,
    extra_topk: int,
    num_main_tiles: int,
    num_tiles: int,
    stride_kv_block: int,
    pbs_extra: int,
    stride_extra_kv_block: int,
    has_sink: bool,
    has_extra: bool,
) -> None:
    traits = make_unified_traits(
        int(model_type),
        int(compute_mode),
        int(scale_format),
    )
    layout = make_smem_layout(traits)
    q_head_dim = int(q.shape[-1])
    num_tokens = int(q.shape[0])
    heads = int(q.shape[1])
    hpb = int(traits.hpb)
    h_blocks = heads // hpb
    d_v = int(traits.d_v)
    kernel = UnifiedPrefillKernel(
        traits,
        layout,
        int(page_block_size),
        int(num_tiles),
        h_blocks=h_blocks,
        num_heads=heads,
        q_head_dim=q_head_dim,
        topk=topk,
        extra_topk=extra_topk,
        q_stride=tuple(q.stride()),
        indices_stride0=int(topk_indices.stride(0)),
        extra_indices_stride0=int(extra_indices_t.stride(0)),
        output_stride=tuple(output.stride()),
        out_lse_stride=tuple(lse_out.stride()),
        has_sink=bool(has_sink),
        has_extra=bool(has_extra),
        pbs_extra=int(pbs_extra),
        num_main_tiles=int(num_main_tiles),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    base_args = (
        _to_cute(q, cutlass.BFloat16, dynamic_layout=True),
        _to_cute(kv_flat, cutlass.Uint8, align=16),
        _to_cute(topk_indices, cutlass.Int32, align=4, dynamic_layout=True),
        _to_cute(topk_length, cutlass.Int32, align=4, dynamic_layout=True),
        _to_cute(attn_sink_t, cutlass.Float32, align=4),
        _to_cute(output, cutlass.BFloat16, align=16, dynamic_layout=True),
        _to_cute(lse_out, cutlass.Float32, align=4, dynamic_layout=True),
        Float32(float(sm_scale) * LOG2_E),
        Int64(stride_kv_block),
    )
    if has_extra:
        args = base_args + (
            _to_cute(extra_kv_flat, cutlass.Uint8, align=16),
            _to_cute(extra_indices_t, cutlass.Int32, align=4, dynamic_layout=True),
            _to_cute(extra_len_t, cutlass.Int32, align=4, dynamic_layout=True),
            Int64(stride_extra_kv_block),
            Int32(num_tokens),
            stream,
        )
    else:
        args = base_args + (Int32(num_tokens), stream)

    spec_fields = [
        key_field("model_type", traits.model_type),
        key_field("compute_mode", traits.compute_mode),
        key_field("scale_format", traits.scale_format),
        key_field("num_heads", heads),
        key_field("hpb", hpb),
        key_field("num_tiles", int(num_tiles)),
        key_field("num_main_tiles", int(num_main_tiles)),
        key_field("page_block_size", int(page_block_size)),
        key_field("topk_bucket", _topk_bucket(topk)),
        key_field("has_sink", int(has_sink)),
        key_field("has_extra", int(has_extra)),
        key_field("pbs_extra", int(pbs_extra)),
        key_field("extra_topk_bucket", _topk_bucket(extra_topk) if has_extra else 0),
        tensor_key(
            "q",
            q,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(heads),
                DimKey.exact(q_head_dim),
            ),
        ),
        tensor_key(
            "topk_indices",
            topk_indices,
            dims=(DimKey.dynamic(), DimKey.bucket(topk)),
        ),
        tensor_key(
            "output",
            output,
            dims=(DimKey.dynamic(), DimKey.exact(heads), DimKey.exact(d_v)),
        ),
        tensor_key(
            "out_lse",
            lse_out,
            dims=(DimKey.dynamic(), DimKey.exact(heads)),
        ),
    ]
    if has_extra:
        spec_fields.append(
            tensor_key(
                "extra_indices",
                extra_indices_t,
                dims=(DimKey.dynamic(), DimKey.bucket(max(extra_topk, 1))),
            )
        )
    compile_spec = KernelCompileSpec.from_fields(
        "attention.mla.unified_sm120.prefill",
        8,
        *spec_fields,
    )
    entry = kernel.call_extra if has_extra else kernel
    b12x_launch(
        entry,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "b12x::unified_sm120_prefill",
    mutates_args=("output", "lse_out"),
)
def _unified_sm120_prefill_op(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink_t: torch.Tensor,
    output: torch.Tensor,
    lse_out: torch.Tensor,
    extra_kv_flat: torch.Tensor,
    extra_indices_t: torch.Tensor,
    extra_len_t: torch.Tensor,
    sm_scale: float,
    model_type: int,
    compute_mode: int,
    scale_format: int,
    page_block_size: int,
    topk: int,
    extra_topk: int,
    num_main_tiles: int,
    num_tiles: int,
    stride_kv_block: int,
    pbs_extra: int,
    stride_extra_kv_block: int,
    has_sink: bool,
    has_extra: bool,
) -> None:
    _unified_sm120_prefill_flat_launch(
        q,
        kv_flat,
        topk_indices,
        topk_length,
        attn_sink_t,
        output,
        lse_out,
        extra_kv_flat,
        extra_indices_t,
        extra_len_t,
        sm_scale,
        model_type,
        compute_mode,
        scale_format,
        page_block_size,
        topk,
        extra_topk,
        num_main_tiles,
        num_tiles,
        stride_kv_block,
        pbs_extra,
        stride_extra_kv_block,
        has_sink,
        has_extra,
    )


@_unified_sm120_prefill_op.register_fake
def _unified_sm120_prefill_fake(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink_t: torch.Tensor,
    output: torch.Tensor,
    lse_out: torch.Tensor,
    extra_kv_flat: torch.Tensor,
    extra_indices_t: torch.Tensor,
    extra_len_t: torch.Tensor,
    sm_scale: float,
    model_type: int,
    compute_mode: int,
    scale_format: int,
    page_block_size: int,
    topk: int,
    extra_topk: int,
    num_main_tiles: int,
    num_tiles: int,
    stride_kv_block: int,
    pbs_extra: int,
    stride_extra_kv_block: int,
    has_sink: bool,
    has_extra: bool,
) -> None:
    return None


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
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    extra_page_block_size: int | None = None,
    stride_extra_kv_block: int | None = None,
    workspace=None,
):
    """Unified SM120 sparse-MLA single-pass prefill -> BF16 O + base-2 LSE.

    A THIN launcher over ``UnifiedPrefillKernel`` (the decode CTA body run at the
    implicit num_splits=1, with the FINAL_BF16 epilogue + per-token section_len +
    optional attn_sink). P8b: 4 IO warps, 384 threads (8 math + 4 IO), the proven
    1-IO mbarrier protocol scaled to 128 IO threads + setmaxnreg dec/inc.

    Routes DSV4 (q_head_dim==512, UE8M0 footer, V_HAS_ROPE) AND GLM_NSA
    (q_head_dim==576, ARBITRARY_FP32 inline scales, V==nope) through the SAME kernel
    via the traits const_expr branches (model_type/scale_format/v_has_rope/
    nt_per_warp_xv), exactly like the decode launcher. DSV4 additionally supports a
    DUAL-CACHE union (extra_kv_cache / extra_indices / extra_topk_length /
    extra_page_block_size): the CTA attends over the UNION of the MAIN topk cache and
    the EXTRA cache in ONE online softmax (num_main_tiles main chunks then the extra
    chunks). The extra cache is DSV4-only (GLM has no extra section -> RAISE).

    Args:
      q:            (T, heads, D_QK) bf16. D_QK 512 (DSV4) or 576 (GLM_NSA).
      kv_cache:     flat uint8 MAIN KV cache (reshaped to 1-D).
      topk_indices: (T, topk) int32 flat slot ids (-1 = invalid sentinel).
      sm_scale:     softmax scale (typically D_QK**-0.5).
      page_block_size: tokens per MAIN KV block (64 for DSV4/GLM).
      topk_length:  optional (T,) int32 per-token MAIN valid length; entries past it
                    are masked. Defaults to full ``topk`` for every token.
      attn_sink:    optional (heads,) fp32 per-head natural-log sink, folded into
                    the normalizer + base-2 LSE (FlashMLA V4).
      output:       optional pre-allocated (T, heads, D_V) bf16 output (else made).
      lse_out:      optional pre-allocated (T, heads) f32 base-2 LSE (else made).
      stride_kv_block: per-block gmem byte stride for the MAIN cache. Derived from
                    page_block_size + model_type when omitted.
      extra_kv_cache / extra_indices / extra_topk_length / extra_page_block_size:
                    DSV4 dual-cache EXTRA pool (all-or-none; partial trio RAISEs).
      stride_extra_kv_block: EXTRA per-block byte stride (derived when omitted).
      workspace:    unused (prefill is single-pass, no split/merge workspace);
                    accepted for launcher-signature symmetry.

    Returns (O[T, heads, D_V=512] bf16, lse[T, heads] f32 base-2).
    """
    from b12x.attention.mla.compressed_reference import compressed_mla_page_nbytes

    del workspace  # prefill is single-pass; no split/merge workspace needed.

    q_head_dim = int(q.shape[-1])
    if q_head_dim not in (_DSV4_HEAD_DIM, _GLM_HEAD_DIM):
        # Genuinely-unsupported contract -> error like upstream (infer_model_type
        # ICHECKs d_qk in {512, 576}). NOT a legacy fallback.
        raise ValueError(
            f"unified_sm120 prefill supports DSV4 (q_head_dim=512) or GLM_NSA "
            f"(q_head_dim=576); got q_head_dim={q_head_dim}"
        )

    num_tokens, heads, _ = q.shape
    hpb = 16
    if heads % hpb != 0:
        # VALID_HPB<16 small-TP shards are a separate (decode-landed) feature; until
        # ported in prefill this is an unsupported shape -> RAISE (not legacy).
        raise ValueError(
            f"unified_sm120 prefill requires heads divisible by HPB={hpb}, got {heads}"
        )

    model_type, compute_mode, scale_format = infer_model_type(q_head_dim, kv_cache.dtype)
    traits = make_unified_traits(model_type, compute_mode, scale_format)
    d_v = int(traits.d_v)

    # ── DSV4 dual-cache: validate the extra trio (all-or-none) and that it is DSV4. ──
    has_extra = (
        extra_kv_cache is not None
        or extra_indices is not None
        or extra_topk_length is not None
    )
    if has_extra:
        if (
            extra_kv_cache is None
            or extra_indices is None
            or extra_page_block_size is None
        ):
            raise ValueError(
                "unified_sm120 prefill dual-cache requires extra_kv_cache, "
                "extra_indices, and extra_page_block_size together (partial extra "
                "trio is unsupported, matching upstream sparse_mla_sm120.cu:171-174)"
            )
        if model_type != ModelType.DSV4:
            raise ValueError(
                "unified_sm120 prefill dual-cache (extra tokens) is DSV4-only "
                "(q_head_dim==512); GLM/DSV3.2 has no extra cache"
            )

    topk = int(topk_indices.shape[1])
    num_main_tiles = (topk + _CAND_WINDOW - 1) // _CAND_WINDOW
    if has_extra:
        extra_topk = int(extra_indices.shape[1])
        num_extra_tiles = (extra_topk + _CAND_WINDOW - 1) // _CAND_WINDOW
    else:
        extra_topk = 0
        num_extra_tiles = 0
    num_tiles = num_main_tiles + num_extra_tiles

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
        if model_type == ModelType.GLM_NSA:
            # GLM cache: per-token 656B contiguous record; a paged "block" holds
            # page_block_size tokens, so the per-block byte stride is pbs*656.
            stride_kv_block = int(page_block_size) * _GLM_KV_GMEM_STRIDE
        else:
            stride_kv_block = int(compressed_mla_page_nbytes(int(page_block_size)))

    q = q.contiguous()
    topk_indices = topk_indices.contiguous()
    if output is None:
        output = torch.empty((num_tokens, heads, d_v), dtype=torch.bfloat16, device=device)
    if lse_out is None:
        lse_out = torch.empty((num_tokens, heads), dtype=torch.float32, device=device)

    # ── EXTRA (dual-cache) tensors / stride. When no extra cache they alias the main
    #    cache / indices and the has_extra=False const_expr elides the extra reads. ──
    if has_extra:
        pbs_extra = int(extra_page_block_size)
        if stride_extra_kv_block is None:
            stride_extra_kv_block = int(compressed_mla_page_nbytes(pbs_extra))
        extra_kv_flat = extra_kv_cache.reshape(-1)
        extra_indices_t = extra_indices.contiguous()
        if extra_topk_length is None:
            extra_len_t = torch.full(
                (num_tokens,), extra_topk, dtype=torch.int32, device=device
            )
        else:
            extra_len_t = extra_topk_length.to(
                device=device, dtype=torch.int32
            ).contiguous()
    else:
        pbs_extra = 1
        stride_extra_kv_block = 0
        extra_kv_flat = kv_cache.reshape(-1)  # alias (never read when has_extra=False)
        extra_indices_t = topk_indices        # alias (never read)
        extra_len_t = topk_length             # alias (never read)

    kv_flat = kv_cache.reshape(-1)
    torch.ops.b12x.unified_sm120_prefill(
        q,
        kv_flat,
        topk_indices,
        topk_length,
        attn_sink_t,
        output,
        lse_out,
        extra_kv_flat,
        extra_indices_t,
        extra_len_t,
        float(sm_scale),
        int(model_type),
        int(compute_mode),
        int(scale_format),
        int(page_block_size),
        int(topk),
        int(extra_topk),
        int(num_main_tiles),
        int(num_tiles),
        int(stride_kv_block),
        int(pbs_extra),
        int(stride_extra_kv_block),
        bool(has_sink),
        bool(has_extra),
    )
    return output, lse_out
