"""Launch / dispatch entrypoints for the unified SM120 sparse-MLA backend.

This is the opt-in, parallel backend (existing kernels remain the default and
untouched). Selection is gated by the ``B12X_MLA_SM120_UNIFIED`` env flag (or a
``backend="sm120_unified"`` kwarg wired in by the API agent). The flag is parsed
ONCE into a module-level bool using the SAME helper pattern as
``b12x/attention/mla/api.py``'s ``_env_flag`` (api.py:85-86):

    os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")

The ``run_unified_decode`` entrypoint (P7) is a REAL launcher: it builds views,
plans the split-K chunk ranges, launches the warp-specialized 288-thread DSV4
decode kernel over grid ``(num_tokens, H_BLOCKS, num_splits)`` writing PER-SPLIT
NORMALIZED partials into the workspace ``mid_out`` / ``mid_lse`` (exactly
split.py's ``SparseMLASplitDecodeMergeKernel`` convention), then runs the REUSED
base-2 merge (``run_sparse_mla_split_decode_merge``) to produce the final O.
``num_splits`` is chosen by a FlashInfer-ported wave-balance heuristic (P9b; see
``plan_unified_decode_splits`` / ``_wave_balanced_num_splits``) -- this fills the
SMs at batch=1 instead of the prior serial ``num_splits=1`` (the #1 decode-perf
lever from ``.sm120port/P9_benchmark_findings.md``). ``num_splits=1`` remains the
trivial 1-split merge (partial == final O).

SCOPE: DSV4 main-cache ONLY. GLM (q=576) and has_extra_cache (extra-tokens) are
NOT supported here -- the compressed_api.py dispatch gate must fall back to the
legacy backend for those (and for non-SM120 devices); see compressed_api.py.

``run_unified_prefill`` / ``run_unified_merge`` remain STUBS (out of P7 scope).
"""

from __future__ import annotations

import math
import os

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
from .traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
    infer_model_type,
    make_unified_traits,
)


_MLA_SM120_UNIFIED_ENV = "B12X_MLA_SM120_UNIFIED"

# BI=64 candidates per chunk (one full/empty KV buffer window). The split-K
# planner cuts the topk into chunk-aligned ranges so split partials are disjoint
# and the merge is exact (split boundary == chunk boundary).
_CAND_WINDOW = 64

# DSV4 compressed contract head dim (q_nope 448 + q_rope 64).
_DSV4_HEAD_DIM = 512
# GLM_NSA uncompressed contract head dim (q_nope 512 + q_rope 64).
_GLM_HEAD_DIM = 576
# GLM per-token packed cache record (reference.pack_mla_kv_cache_reference).
_GLM_KV_GMEM_STRIDE = 656


def _env_flag(name: str) -> bool:
    """Match api.py:86 exactly so flag parsing is identical across backends."""
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


# Parsed ONCE at import time (module-level bool). The dispatcher consults this
# (OR the backend kwarg) together with get_sm_version() >= 120 before routing to
# the unified backend.
B12X_MLA_SM120_UNIFIED: bool = _env_flag(_MLA_SM120_UNIFIED_ENV)


# Optional decode num_splits override (P9b AutoTuner sweep hook). ``<= 0`` (or an
# unparseable value, or unset) means "use the FlashInfer-ported wave-balanced
# heuristic"; ``>= 1`` pins num_splits to that value (still clamped to
# [1, num_chunks] and to the workspace split capacity). Read PER CALL (not cached
# at import) so a sweep / AutoTuner can flip it between launches.
_MLA_SM120_NUM_SPLITS_ENV = "B12X_MLA_SM120_NUM_SPLITS"

# FlashInfer's decode-dsv4 chunks_per_block wave-balance cap
# (csrc/sparse_mla_sm120_decode_dsv4.cu:85). cpb candidates whose last-wave tail
# gap looks small but require more than this many integer waves are rejected.
_CEIL_WAVES_MAX = 3

# Observability: the most recent decode split plan (read by benchmarks / the
# P9c AutoTuner sweep to report num_splits_used). Pure side-channel -- does NOT
# affect numerics or the launch. Keys are informational.
LAST_DECODE_PLAN: dict = {}


def _env_num_splits_override() -> int:
    """Read ``B12X_MLA_SM120_NUM_SPLITS`` per call. ``<= 0`` / unset / bad -> 0
    (heuristic). Mirrors FlashInfer's ``chunks_per_block_override <= 0 -> C++
    heuristic`` convention, but for OUR num_splits (== FlashInfer's active block
    count num_splits_eff)."""
    raw = os.environ.get(_MLA_SM120_NUM_SPLITS_ENV)
    if raw is None:
        return 0
    try:
        v = int(raw.strip())
    except (TypeError, ValueError):
        return 0
    return v if v > 0 else 0


def _wave_balanced_num_splits(
    *, num_chunks: int, per_token_head: int, sm_count: int
) -> int:
    """Replicate FlashInfer's decode-dsv4 occupancy decision in OUR chunk-range
    parameterization, returning OUR ``num_splits`` (== FlashInfer's active block
    count ``num_splits_eff``).

    FlashInfer splits MAXIMALLY: its ``num_splits`` == ``num_chunks`` (one block
    per KV chunk; sparse_mla_sm120.py:259-260,286), then a C++ heuristic
    (csrc/sparse_mla_sm120_decode_dsv4.cu:69-102) picks ``chunks_per_block`` to
    wave-balance the launch and computes ``num_splits_eff = ceil(num_splits /
    cpb)`` ACTIVE blocks. Our UnifiedDecodeKernel processes a contiguous
    chunk-RANGE per CTA, so OUR ``num_splits`` directly IS that active count: we
    port the cpb tail-gap search VERBATIM over ``num_chunks`` chunks, then return
    ``ceil(num_chunks / cpb*)``.

    Tail-gap formula (VERBATIM from the .cu):
        per_token_head = num_tokens * H_BLOCKS
        for cpb in 1..num_chunks:                  # FlashInfer: 1..num_splits
            eff    = ceil(num_chunks / cpb)
            active = per_token_head * eff
            ceil_w = ceil(active / sm_count)
            if ceil_w > CEIL_WAVES_MAX(=3): continue
            waves  = active / sm_count
            gap    = ceil_w - waves
            pick cpb minimizing gap (tie -> larger cpb, fewer launched blocks)
        num_splits = ceil(num_chunks / cpb*)       # == FlashInfer num_splits_eff
    """
    num_chunks = max(int(num_chunks), 1)
    per_token_head = max(int(per_token_head), 1)
    sm_count = max(int(sm_count), 1)

    chunks_per_block = 1
    best_gap = 2.0
    for cpb in range(1, num_chunks + 1):
        eff = (num_chunks + cpb - 1) // cpb
        active = per_token_head * eff
        ceil_w = (active + sm_count - 1) // sm_count
        if ceil_w > _CEIL_WAVES_MAX:
            continue
        waves = active / sm_count
        gap = ceil_w - waves
        if gap < best_gap - 1e-6 or (gap < best_gap + 1e-6 and cpb > chunks_per_block):
            best_gap = gap
            chunks_per_block = cpb
    # OUR num_splits == FlashInfer's num_splits_eff = ceil(num_splits / cpb*),
    # where FlashInfer's num_splits == num_chunks (maximal split).
    return (num_chunks + chunks_per_block - 1) // chunks_per_block


# ---------------------------------------------------------------------------
# Split-K planning. Reuse the compressed planner's chunk-count idiom but pin the
# per-split chunk granularity to the kernel's BI=64 window so split boundaries
# land on chunk boundaries (a candidate is processed by exactly one split ->
# multi-split is numerically identical to single-split).
# ---------------------------------------------------------------------------
def plan_unified_decode_splits(
    *,
    topk: int,
    max_chunks: int,
    forced_num_splits: int | None = None,
    num_tokens: int = 1,
    h_blocks: int = 1,
    sm_count: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(num_chunks, num_splits, chunks_per_split)``.

    ``num_chunks = ceil(topk / BI)`` is the number of BI=64 candidate windows.
    ``num_splits`` is chosen by replicating FlashInfer's decode launch tuning:
    FlashInfer splits MAXIMALLY (one block per KV chunk) then wave-balances via a
    chunks_per_block heuristic to an ACTIVE block count
    ``num_splits_eff = ceil(num_chunks / cpb*)``. Our CTA owns a chunk-RANGE, so
    OUR ``num_splits`` directly IS that active count -- we port the same
    CEIL_WAVES_MAX=3 tail-gap search (see ``_wave_balanced_num_splits``).

    Override precedence (highest first):
      1. ``forced_num_splits`` (explicit caller arg -- multi-split numeric checks).
      2. ``B12X_MLA_SM120_NUM_SPLITS`` env (>=1 pins; <=0/unset -> heuristic).
      3. The FlashInfer-ported wave-balanced heuristic (needs ``sm_count``;
         falls back to 1 if ``sm_count`` is unavailable).

    ``num_splits`` is clamped to ``[1, num_chunks]`` and to ``max_chunks`` (the
    workspace mid_out/mid_lse split capacity).
    """
    topk = max(int(topk), 1)
    num_chunks = (topk + _CAND_WINDOW - 1) // _CAND_WINDOW

    if forced_num_splits is not None:
        num_splits = max(1, int(forced_num_splits))
    else:
        env_override = _env_num_splits_override()
        if env_override > 0:
            num_splits = env_override
        elif sm_count and sm_count > 0:
            num_splits = _wave_balanced_num_splits(
                num_chunks=num_chunks,
                per_token_head=max(1, int(num_tokens)) * max(1, int(h_blocks)),
                sm_count=int(sm_count),
            )
        else:
            num_splits = 1

    num_splits = min(num_splits, num_chunks, max(1, int(max_chunks)))
    chunks_per_split = (num_chunks + num_splits - 1) // num_splits
    return num_chunks, num_splits, chunks_per_split


class UnifiedDecodeKernel:
    """288-thread warp-specialized DSV4 decode with split-K partial writeback.

    Grid = (num_tokens, H_BLOCKS, num_splits). Each CTA owns one query token, one
    HPB=16-head block, and one chunk-range slice (split). 8 math warps consume
    the double-buffered KV gathered by the 9th IO warp (cp.async.bulk + mbarrier,
    io.py); the math runs S0-S6b over the split's chunks then S7 writes this
    split's NORMALIZED partial O + base-2 LSE into mid_out / mid_lse in the exact
    split.py merge convention. The hot-op MMA PTX (14 block-scaled + 14 plain
    e4m3 + 8 bf16) is identical to the single-CTA P6 kernel: the split slicing
    only changes the chunk-loop BOUNDS and the epilogue DESTINATION.
    """

    def __init__(self, traits, layout, page_block_size, chunks_per_split,
                 num_tokens, h_blocks, num_splits):
        self.traits = traits
        self.layout = layout
        self.page_block_size = int(page_block_size)
        self.chunks_per_split = int(chunks_per_split)
        self.num_tokens = int(num_tokens)
        self.h_blocks = int(h_blocks)
        self.num_splits = int(num_splits)
        self.math_threads = int(traits.math_threads)
        self.block_threads = int(traits.block_threads)

    @cute.jit
    def __call__(
        self,
        q_all: cute.Tensor,          # (rows, heads, D_QK) bf16
        kv_cache_u8: cute.Tensor,    # flat (pages*page_nbytes,) u8
        swa_indices: cute.Tensor,    # (rows, topk) int32
        mid_out: cute.Tensor,        # (rows, heads, splits, D_V) bf16 partials
        mid_lse: cute.Tensor,        # (rows, heads, splits) f32 base-2 LSE
        sm_scale_log2: Float32,
        section_len: Int32,
        stride_kv_block: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_all,
            kv_cache_u8,
            swa_indices,
            mid_out,
            mid_lse,
            sm_scale_log2,
            section_len,
            stride_kv_block,
        ).launch(
            grid=(self.num_tokens, self.h_blocks, self.num_splits),
            block=[self.block_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        section_len: Int32,
        stride_kv_block: Int64,
    ):
        t = self.traits
        L = self.layout
        tid = Int32(cute.arch.thread_idx()[0])
        lane = cute.arch.lane_idx()
        warp_id = tid >> Int32(5)

        token_idx, head_block, split_idx = cute.arch.block_idx()
        token_idx = Int32(token_idx)
        head_block = Int32(head_block)
        split_idx = Int32(split_idx)
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

        # ── 288 threads = 8 math warps (CONSUMER, warps 0-7) + 1 IO warp (warp 8). ──
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

        # Per-split chunk range [split_first_chunk, split_last_chunk) over BI windows.
        cps = Int32(self.chunks_per_split)
        split_first_chunk = split_idx * cps

        # swa_indices for THIS token row: a 1-D (topk,) slice.
        topk_row = cute.make_tensor(
            swa_indices.iterator + token_idx.to(Int64) * Int64(swa_indices.stride[0]),
            cute.make_layout(swa_indices.shape[1]),
        )
        # q for THIS token row: a 2-D (heads, D_QK) view (s0 indexes [head_base+h, d]).
        q_token = cute.make_tensor(
            q_all.iterator + token_idx.to(Int64) * Int64(q_all.stride[0]),
            cute.make_layout(
                (q_all.shape[1], q_all.shape[2]),
                stride=(q_all.stride[1], q_all.stride[2]),
            ),
        )
        warp_first_cand = warp_id * Int32(8)

        # ════════════════════════════════════════════════════════════════════
        # IO WARP (PRODUCER) vs MATH WARPS (CONSUMER).
        # ════════════════════════════════════════════════════════════════════
        if is_io:
            io_lane = lane
            prod_phase = Int32(1)
            prod_idx = Int32(0)
            for lc in cutlass.range(self.chunks_per_split, unroll=1):
                ci = split_first_chunk + Int32(lc)
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
                    scale_format=t.scale_format,
                )
                prod_idx += Int32(1)
                if prod_idx == Int32(n_buf):
                    prod_idx = Int32(0)
                    prod_phase ^= Int32(1)

        else:
            # MATH WARPS (CONSUMER, warps 0-7 = 256 threads).
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

            for lc in cutlass.range(self.chunks_per_split, unroll=1):
                ci = split_first_chunk + Int32(lc)
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

                # GLM-only S0b: in-place K dequant+requant (ARBITRARY_FP32 -> true
                # value e4m3, unit sfb in S1). const_expr-elided for DSV4.
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

                # S6b (XV-RoPE) is DSV4-only (V_HAS_ROPE). const_expr-elided for GLM.
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

            # ── S7: write this split's NORMALIZED partial + base-2 LSE into
            #    mid_out[token, :, split, :] / mid_lse[token, :, split]. ──
            fin_acc_nope = [
                [accn_frag[at * 4 + 0], accn_frag[at * 4 + 1],
                 accn_frag[at * 4 + 2], accn_frag[at * 4 + 3]]
                for at in range(n_acc_tiles)
            ]
            fin_acc_rope = [accr_frag[0], accr_frag[1], accr_frag[2], accr_frag[3]]
            fin_gmax = [gmax_frag[0], gmax_frag[1]]
            fin_gsum = [gsum_frag[0], gsum_frag[1]]

            # mid_out[token, head_base + h, split, dim]: (HPB, D_V) view for this
            # (token, head_block, split). mid_out stride = (h*S*Dv, S*Dv, Dv, 1).
            out_o = cute.make_tensor(
                mid_out.iterator
                + token_idx.to(Int64) * Int64(mid_out.stride[0])
                + head_base.to(Int64) * Int64(mid_out.stride[1])
                + split_idx.to(Int64) * Int64(mid_out.stride[2]),
                cute.make_layout(
                    (t.hpb, t.d_v),
                    stride=(mid_out.stride[1], mid_out.stride[3]),
                ),
            )
            # mid_lse[token, head_base + h, split]: (HPB,) view.
            out_lse = cute.make_tensor(
                mid_lse.iterator
                + token_idx.to(Int64) * Int64(mid_lse.stride[0])
                + head_base.to(Int64) * Int64(mid_lse.stride[1])
                + split_idx.to(Int64) * Int64(mid_lse.stride[2]),
                cute.make_layout((t.hpb,), stride=(mid_lse.stride[1],)),
            )
            s7_epilogue(
                fin_acc_nope, fin_acc_rope, fin_gmax, fin_gsum, out_o, out_lse,
                warp_id, lane,
                n_v_chunks=t.n_v_chunks, v_chunk=t.quant_tile, d_nope=t.d_nope,
                d_rope=t.d_rope, n_warps=8, valid_hpb=t.hpb,
                nt_per_warp_xv=t.nt_per_warp_xv, v_has_rope=t.v_has_rope,
            )


def _to_cute(x, dtype, align=16):
    c = from_dlpack(x, assumed_align=align)
    c.element_type = dtype
    return c


def _topk_bucket(topk: int) -> int:
    """Coarse topk bucket for the compile key (chunks_per_split is the real
    specialization driver; the bucket just keeps the key compact)."""
    return 1 << (max(int(topk), 1) - 1).bit_length()


def run_unified_decode(
    *,
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    workspace,
    sm_scale: float,
    swa_page_size: int,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_topk_lengths: torch.Tensor | None = None,
    indexed_page_size: int | None = None,
    indexed_page_table: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    return_lse: bool = False,
    lse_scale: str = "base2",
    forced_num_splits: int | None = None,
):
    """Unified SM120 sparse-MLA decode: kernel (split-K partials) + merge.

    Routes DSV4 (q_head_dim==512, UE8M0 footer) AND GLM_NSA (q_head_dim==576,
    ARBITRARY_FP32 inline scales) to the SAME warp-specialized kernel via the
    cute.constexpr traits branches (model_type/scale_format/v_has_rope). The
    dispatch gate guarantees SM120; this entrypoint rejects unsupported features
    so an accidental route never silently mis-computes.
    """
    from b12x.attention.mla.compressed_reference import compressed_mla_page_nbytes

    if indexed_k_cache is not None or indexed_indices is not None or indexed_topk_lengths is not None:
        raise NotImplementedError(
            "unified_sm120 decode: has_extra_cache / indexed (extra-tokens) is P7c; "
            "the dispatch gate must fall back to legacy"
        )
    if attn_sink is not None:
        raise NotImplementedError(
            "unified_sm120 decode: attn_sink fold is not yet supported; fall back to legacy"
        )
    if return_lse:
        raise NotImplementedError(
            "unified_sm120 decode: return_lse is not yet supported; fall back to legacy"
        )

    q_head_dim = int(q_all.shape[-1])
    if q_head_dim not in (_DSV4_HEAD_DIM, _GLM_HEAD_DIM):
        raise NotImplementedError(
            f"unified_sm120 decode supports q_head_dim 512 (DSV4) or 576 (GLM); "
            f"got {q_head_dim} -- fall back to legacy"
        )

    rows, heads, _ = q_all.shape
    hpb = 16
    if heads % hpb != 0:
        raise NotImplementedError(
            f"unified_sm120 decode requires heads divisible by HPB={hpb}, got {heads}"
        )
    h_blocks = heads // hpb

    model_type, compute_mode, scale_format = infer_model_type(q_head_dim, swa_k_cache.dtype)
    traits = make_unified_traits(model_type, compute_mode, scale_format)
    layout = make_smem_layout(traits)
    d_v = int(traits.d_v)  # output O dim (512 for both; V == nope for GLM)

    topk = int(swa_indices.shape[1])
    max_chunks = int(workspace.max_chunks_per_row)
    # SM count read at RUNTIME (RTX PRO 6000 Blackwell sm_120 et al.) -- feeds the
    # FlashInfer-ported wave-balance tail-gap search. None if no CUDA device.
    sm_count = None
    if q_all.is_cuda:
        sm_count = int(
            torch.cuda.get_device_properties(q_all.device).multi_processor_count
        )
    num_chunks, num_splits, chunks_per_split = plan_unified_decode_splits(
        topk=topk,
        max_chunks=max_chunks,
        forced_num_splits=forced_num_splits,
        num_tokens=rows,
        h_blocks=h_blocks,
        sm_count=sm_count,
    )
    # Side-channel record of the chosen split plan (benchmarks / AutoTuner read
    # LAST_DECODE_PLAN["num_splits"]). Informational only.
    LAST_DECODE_PLAN.clear()
    LAST_DECODE_PLAN.update(
        model_type=str(model_type),
        topk=int(topk),
        num_chunks=int(num_chunks),
        num_splits=int(num_splits),
        chunks_per_split=int(chunks_per_split),
        num_tokens=int(rows),
        h_blocks=int(h_blocks),
        sm_count=(int(sm_count) if sm_count else None),
    )
    # Workspace mid_out/mid_lse must hold num_splits partials per (token, head).
    if num_splits > max_chunks:
        raise ValueError(
            f"unified_sm120 decode num_splits {num_splits} exceeds workspace "
            f"max_chunks_per_row {max_chunks}"
        )

    if workspace.tmp_output is None or workspace.tmp_lse is None:
        raise RuntimeError("unified_sm120 decode workspace is missing mid_out/mid_lse")

    # mid_out / mid_lse views over the workspace split buffers (the merge's
    # exact tmp_output[rows,heads,chunks,dim] / tmp_lse[rows,heads,chunks]). The
    # partial O dim is d_v (512) for both models.
    mid_out = workspace.tmp_output[:rows, :heads, :num_splits, :d_v]
    mid_lse = workspace.tmp_lse[:rows, :heads, :num_splits]
    # Seed empty-split LSE = -inf so the merge skips splits with no chunks (and
    # so partially-written rows are well-defined). The kernel overwrites every
    # (token, head, split) it actually runs.
    mid_lse.fill_(float("-inf"))

    if model_type == ModelType.GLM_NSA:
        # GLM cache: per-token 656B contiguous record; one paged "block" holds
        # page_block_size tokens, so the per-block byte stride is pbs*656.
        stride_kv_block = int(swa_page_size) * _GLM_KV_GMEM_STRIDE
    else:
        stride_kv_block = int(compressed_mla_page_nbytes(int(swa_page_size)))

    output = workspace.output_buffer[:rows, :heads, :d_v]

    kernel = UnifiedDecodeKernel(
        traits, layout, int(swa_page_size), chunks_per_split,
        num_tokens=rows, h_blocks=h_blocks, num_splits=num_splits,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kv_flat = swa_k_cache.reshape(-1)
    args = (
        _to_cute(q_all, cutlass.BFloat16),
        _to_cute(kv_flat, cutlass.Uint8, align=16),
        _to_cute(swa_indices, cutlass.Int32, align=4),
        _to_cute(mid_out, cutlass.BFloat16, align=16),
        _to_cute(mid_lse, cutlass.Float32, align=4),
        Float32(float(sm_scale) * LOG2_E),
        Int32(topk),
        Int64(stride_kv_block),
        stream,
    )
    compile_spec = KernelCompileSpec.from_fields(
        "attention.mla.unified_sm120.decode",
        1,
        key_field("model_type", traits.model_type),
        key_field("compute_mode", traits.compute_mode),
        key_field("scale_format", traits.scale_format),
        key_field("num_heads", int(heads)),
        key_field("hpb", int(hpb)),
        key_field("chunks_per_split", int(chunks_per_split)),
        key_field("page_block_size", int(swa_page_size)),
        key_field("topk_bucket", _topk_bucket(topk)),
        # rows (== num_tokens) is BAKED into the launch grid
        # (grid=(num_tokens, h_blocks, num_splits), concrete-shape trace), so it
        # MUST be a compile key -- DimKey.exact on every row dim + a num_tokens
        # key_field. A DimKey.dynamic() row dim would silently REUSE a kernel
        # traced for a different num_tokens with the wrong grid (latent rows>1
        # bug); key the exact T, exactly as prefill.py does.
        key_field("num_tokens", int(rows)),
        tensor_key("q_all", q_all, dims=(DimKey.exact(rows), DimKey.exact(heads), DimKey.exact(q_head_dim))),
        tensor_key("swa_indices", swa_indices, dims=(DimKey.exact(rows), DimKey.bucket(topk))),
        tensor_key("mid_out", mid_out, dims=(DimKey.exact(rows), DimKey.exact(heads), DimKey.bucket(num_splits), DimKey.exact(d_v))),
        tensor_key("mid_lse", mid_lse, dims=(DimKey.exact(rows), DimKey.exact(heads), DimKey.bucket(num_splits))),
    )
    b12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )

    # ── REUSED base-2 merge over the split axis -> final O. num_splits=1 is the
    #    trivial 1-split merge (partial == final O). ──
    from b12x.attention.mla.split import (
        build_sparse_mla_split_decode_merge_binding,
        run_sparse_mla_split_decode_merge,
    )

    if int(workspace.num_chunks_value or -1) != num_splits:
        workspace.set_split_chunk_config(kv_chunk_size=_CAND_WINDOW, num_chunks=num_splits)

    merge_binding = build_sparse_mla_split_decode_merge_binding(
        tmp_output=mid_out,
        tmp_lse=mid_lse,
        num_chunks_ptr=workspace.num_chunks_ptr,
        output=output,
        attn_sink=None,
        workspace=workspace,
    )
    run_sparse_mla_split_decode_merge(binding=merge_binding)
    return output


def run_unified_prefill(*args, **kwargs):
    """Unified SM120 sparse-MLA DSV4 prefill (P8).

    Thin re-export of ``prefill.run_unified_prefill`` (the correctness-first
    single-pass DSV4 prefill that REUSES the proven 288-thread / 1-IO-warp decode
    pipeline with a FINAL_BF16 epilogue). Imported lazily so launch.py has no hard
    dependency on the prefill module's CuTe symbols at import time."""
    from .prefill import run_unified_prefill as _impl

    return _impl(*args, **kwargs)


def run_unified_merge(*args, **kwargs):
    """Unified SM120 sparse-MLA partial merge.

    The unified decode REUSES split.py's base-2 SparseMLASplitDecodeMergeKernel
    directly (see run_unified_decode), so a separate merge entrypoint is not
    needed; kept as a STUB for API symmetry."""
    raise NotImplementedError(
        "unified_sm120 run_unified_merge: the decode reuses split.py's merge "
        "(run_sparse_mla_split_decode_merge) directly"
    )
