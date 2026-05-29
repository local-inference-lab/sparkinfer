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

SCOPE: DSV4 (main cache + the P7c DUAL-CACHE extra-tokens second KV pool) AND
GLM_NSA (q=576, main cache; no extra). The DSV4 dual-cache uses TWO @cute.kernel
entries that share ``_kernel_body(has_extra)``: ``kernel`` (8 device params, the
byte-identical-to-pre-P7c single-cache path, used when has_extra=False) and
``kernel_extra`` (13 params, the dual-cache path). The single-cache __call__ ->
kernel trace + PTX are UNCHANGED from the pre-P7c kernel (extra args never enter
that device entry). Non-SM120 / unsupported features fall back to legacy via the
compressed_api.py dispatch gate.

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

# natural-log of 2 (base2 <-> natural LSE conversion).
_LN2 = math.log(2.0)

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
    extra_topk: int = 0,
) -> tuple[int, int, int]:
    """Return ``(num_chunks, num_splits, chunks_per_split)``.

    ``num_chunks = ceil(topk / BI) + ceil(extra_topk / BI)`` is the number of
    BI=64 candidate windows spanning BOTH the main and the EXTRA cache sections
    (DSV4 dual-cache; ``extra_topk=0`` reduces to the single-cache main count).
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
    extra_topk = max(int(extra_topk), 0)
    # num_chunks spans main + extra sections (FlashInfer num_splits = ceil(topk/BI)
    # + ceil(extra_topk/BI)); the wave-balance heuristic then picks the active
    # block count over the COMBINED chunk count.
    num_chunks = (topk + _CAND_WINDOW - 1) // _CAND_WINDOW + (
        (extra_topk + _CAND_WINDOW - 1) // _CAND_WINDOW
    )

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
                 num_tokens, h_blocks, num_splits, has_extra=False,
                 pbs_extra=1, valid_hpb=None, head_block_offset=0,
                 per_token_len=False):
        self.traits = traits
        self.layout = layout
        self.page_block_size = int(page_block_size)
        self.chunks_per_split = int(chunks_per_split)
        self.num_tokens = int(num_tokens)
        self.h_blocks = int(h_blocks)
        self.num_splits = int(num_splits)
        # DSV4 dual-cache (P7c). When False the extra-section code is const_expr-
        # elided -> no-extra DSV4 / GLM PTX byte-identical.
        self.has_extra = bool(has_extra)
        self.pbs_extra = int(pbs_extra)
        # P10b multi-token per-token topk_length. When False the section_len /
        # extra_section_len are uniform Int32 scalars (the byte-identical base /
        # uniform-length path; the per-token length tensors never enter the device
        # entry). When True a per-token int32 length tensor is read in-kernel at
        # t=blockIdx.x and clamped to [0, topk] -> the per-CTA section bound. The
        # launcher routes here ONLY for a genuinely-mixed-length multi-token batch
        # (a uniform batch collapses to the scalar path -> PTX byte-identical).
        self.per_token_len = bool(per_token_len)
        # VALID_HPB (small-TP / non-multiple-of-16 head shards). Upstream
        # VALID_HPB=min(NUM_HEADS,HPB) (decode_dsv4_kernel.cuh:152): the kernel
        # computes a FULL HPB=16 tile with zero-Q padding then gates output/LSE
        # writes to valid_hpb rows. ``valid_hpb`` is a const_expr (s0/s4/s7 gate on
        # it). When valid_hpb == t.hpb (the FULL-block default) the s0/s4/s7 calls
        # pass the IDENTICAL constexpr value as the pre-P10 kernel, so the
        # full-block trace + PTX stay byte-identical. A REMAINDER block (heads not
        # a multiple of 16) is launched as a SEPARATE 1-block grid with
        # valid_hpb=remainder and head_block_offset shifting head_base to the
        # tail head range.
        self.valid_hpb = int(valid_hpb) if valid_hpb is not None else int(traits.hpb)
        # head_block_offset shifts head_base = (head_block + offset) * hpb so a
        # remainder-only 1-block grid writes the correct (tail) head range. When 0
        # (the full-block / base path) the const_expr branch is elided -> the
        # head_base computation is byte-identical to the pre-P10 kernel.
        self.head_block_offset = int(head_block_offset)
        self.math_threads = int(traits.math_threads)
        self.block_threads = int(traits.block_threads)

    @cute.jit
    def __call__(
        self,
        q_all: cute.Tensor,          # (rows, heads, D_QK) bf16
        kv_cache_u8: cute.Tensor,    # flat (pages*page_nbytes,) u8 (MAIN cache)
        swa_indices: cute.Tensor,    # (rows, topk) int32 (MAIN indices)
        mid_out: cute.Tensor,        # (rows, heads, splits, D_V) bf16 partials
        mid_lse: cute.Tensor,        # (rows, heads, splits) f32 base-2 LSE
        sm_scale_log2: Float32,
        section_len: Int32,          # MAIN per-row valid topk length
        stride_kv_block: Int64,      # MAIN per-block byte stride
        stream: cuda.CUstream,
    ):
        # SINGLE-CACHE entry: EXACTLY the v1 traced signature (8 data args +
        # stream). The dispatcher selects this (func=kernel -> __call__) when
        # has_extra=False so the no-extra DSV4 / GLM trace, mangled name, and the
        # launched @cute.kernel (self.kernel, 8 params) stay byte-identical to the
        # pre-P7c kernel: the extra-section args never enter the device entry.
        self.kernel(
            q_all, kv_cache_u8, swa_indices, mid_out, mid_lse,
            sm_scale_log2, section_len, stride_kv_block,
        ).launch(
            grid=(self.num_tokens, self.h_blocks, self.num_splits),
            block=[self.block_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def call_extra(
        self,
        q_all: cute.Tensor,          # (rows, heads, D_QK) bf16
        kv_cache_u8: cute.Tensor,    # flat (pages*page_nbytes,) u8 (MAIN cache)
        swa_indices: cute.Tensor,    # (rows, topk) int32 (MAIN indices)
        mid_out: cute.Tensor,        # (rows, heads, splits, D_V) bf16 partials
        mid_lse: cute.Tensor,        # (rows, heads, splits) f32 base-2 LSE
        sm_scale_log2: Float32,
        section_len: Int32,          # MAIN per-row valid topk length
        stride_kv_block: Int64,      # MAIN per-block byte stride
        extra_kv_cache_u8: cute.Tensor,  # flat u8 EXTRA cache (DSV4 dual-cache)
        extra_indices: cute.Tensor,      # (rows, extra_topk) int32
        extra_section_len: Int32,        # EXTRA per-row valid length
        num_main_chunks: Int32,          # ceil(main_len/BI); chunks >= this read the EXTRA cache
        stride_extra_kv_block: Int64,    # EXTRA per-block byte stride
        stream: cuda.CUstream,
    ):
        # DUAL-CACHE entry (DSV4 P7c): the dispatcher selects this (func=
        # kernel.call_extra) only when has_extra=True, so its DISTINCT mangled name
        # never collides with the byte-identical single-cache __call__. It launches
        # the 13-param @cute.kernel (self.kernel_extra), which shares the body via
        # _kernel_body(has_extra=True).
        self.kernel_extra(
            q_all, kv_cache_u8, swa_indices, mid_out, mid_lse,
            sm_scale_log2, section_len, stride_kv_block,
            extra_kv_cache_u8, extra_indices, extra_section_len,
            num_main_chunks, stride_extra_kv_block,
        ).launch(
            grid=(self.num_tokens, self.h_blocks, self.num_splits),
            block=[self.block_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def call_pertok(
        self,
        q_all: cute.Tensor,          # (rows, heads, D_QK) bf16
        kv_cache_u8: cute.Tensor,    # flat (pages*page_nbytes,) u8 (MAIN cache)
        swa_indices: cute.Tensor,    # (rows, topk) int32 (MAIN indices)
        mid_out: cute.Tensor,        # (rows, heads, splits, D_V) bf16 partials
        mid_lse: cute.Tensor,        # (rows, heads, splits) f32 base-2 LSE
        sm_scale_log2: Float32,
        topk_length: cute.Tensor,    # (rows,) int32 per-token MAIN valid length
        stride_kv_block: Int64,      # MAIN per-block byte stride
        stream: cuda.CUstream,
    ):
        # SINGLE-CACHE PER-TOKEN entry (P10b multi-token mixed-length): a DISTINCT
        # mangled name (the topk_length tensor replaces the section_len scalar) so
        # it never collides with the byte-identical uniform-length __call__. Each
        # CTA reads section_len = clamp(topk_length[blockIdx.x], 0, topk) -> the
        # per-token section bound; over-allocated chunks for short tokens are fully
        # masked (idx<0 + section bound) so their partials are -inf and the merge
        # ignores them.
        self.kernel_pertok(
            q_all, kv_cache_u8, swa_indices, mid_out, mid_lse,
            sm_scale_log2, topk_length, stride_kv_block,
        ).launch(
            grid=(self.num_tokens, self.h_blocks, self.num_splits),
            block=[self.block_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def call_extra_pertok(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        topk_length: cute.Tensor,        # (rows,) int32 per-token MAIN valid length
        stride_kv_block: Int64,
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_topk_length: cute.Tensor,  # (rows,) int32 per-token EXTRA valid length
        num_main_chunks: Int32,          # ceil(MAX main_len/BI); chunks >= this read EXTRA
        stride_extra_kv_block: Int64,
        stream: cuda.CUstream,
    ):
        # DUAL-CACHE PER-TOKEN entry (P10b): both the MAIN and EXTRA section lengths
        # are read per-token (topk_length / extra_topk_length at t=blockIdx.x). The
        # main/extra chunk split (num_main_chunks) stays UNIFORM (workspace geometry
        # over the MAX topk); per-token clamping zeroes the over-allocated chunks.
        self.kernel_extra_pertok(
            q_all, kv_cache_u8, swa_indices, mid_out, mid_lse,
            sm_scale_log2, topk_length, stride_kv_block,
            extra_kv_cache_u8, extra_indices, extra_topk_length,
            num_main_chunks, stride_extra_kv_block,
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
        # REMAINDER-block grids shift head_base to the tail head range; the offset
        # const_expr is elided (==0) for the full-block / base path -> byte-identical.
        if cutlass.const_expr(self.head_block_offset != 0):
            head_block = head_block + Int32(self.head_block_offset)
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
                head_base, Int32(self.valid_hpb), tid,
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
                    n_v_chunks=t.n_v_chunks, hpb=t.hpb, n_warps=8, valid_hpb=self.valid_hpb,
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
                d_rope=t.d_rope, n_warps=8, valid_hpb=self.valid_hpb,
                nt_per_warp_xv=t.nt_per_warp_xv, v_has_rope=t.v_has_rope,
            )

    @cute.kernel
    def kernel_extra(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        section_len: Int32,
        stride_kv_block: Int64,
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_section_len: Int32,
        num_main_chunks: Int32,
        stride_extra_kv_block: Int64,
    ):
        # DUAL-CACHE @cute.kernel: 13 device params; threads the real extra-section
        # args into the shared body (has_extra=True).
        self._kernel_body(
            q_all, kv_cache_u8, swa_indices, mid_out, mid_lse,
            sm_scale_log2, section_len, stride_kv_block,
            extra_kv_cache_u8, extra_indices, extra_section_len,
            num_main_chunks, stride_extra_kv_block,
            swa_indices, extra_indices,  # length tensors unused (per_token_len=False)
            has_extra=True, per_token_len=False,
        )

    @cute.kernel
    def kernel_pertok(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        topk_length: cute.Tensor,
        stride_kv_block: Int64,
    ):
        # SINGLE-CACHE PER-TOKEN @cute.kernel (P10b): the per-token MAIN length
        # tensor replaces the scalar section_len; the body reads
        # section_len = clamp(topk_length[blockIdx.x], 0, topk) per CTA. The scalar
        # section_len/extra_section_len/num_main_chunks args are dummies (elided by
        # per_token_len/has_extra const_expr); the extra tensor slots alias
        # swa_indices (never read when has_extra=False).
        self._kernel_body(
            q_all, kv_cache_u8, swa_indices, mid_out, mid_lse,
            sm_scale_log2, Int32(0), stride_kv_block,
            kv_cache_u8, swa_indices, Int32(0),
            Int32(0), stride_kv_block,
            topk_length, swa_indices,
            has_extra=False, per_token_len=True,
        )

    @cute.kernel
    def kernel_extra_pertok(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        topk_length: cute.Tensor,
        stride_kv_block: Int64,
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_topk_length: cute.Tensor,
        num_main_chunks: Int32,
        stride_extra_kv_block: Int64,
    ):
        # DUAL-CACHE PER-TOKEN @cute.kernel (P10b): both MAIN and EXTRA section
        # lengths are read per-token (topk_length / extra_topk_length at
        # t=blockIdx.x). num_main_chunks (the uniform main/extra chunk split) stays
        # a scalar; per-token clamping masks the over-allocated chunks.
        self._kernel_body(
            q_all, kv_cache_u8, swa_indices, mid_out, mid_lse,
            sm_scale_log2, Int32(0), stride_kv_block,
            extra_kv_cache_u8, extra_indices, Int32(0),
            num_main_chunks, stride_extra_kv_block,
            topk_length, extra_topk_length,
            has_extra=True, per_token_len=True,
        )

    @cute.jit
    def _kernel_body(
        self,
        q_all: cute.Tensor,
        kv_cache_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        mid_out: cute.Tensor,
        mid_lse: cute.Tensor,
        sm_scale_log2: Float32,
        section_len: Int32,
        stride_kv_block: Int64,
        extra_kv_cache_u8: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_section_len: Int32,
        num_main_chunks: Int32,
        stride_extra_kv_block: Int64,
        topk_length: cute.Tensor,
        extra_topk_length: cute.Tensor,
        *,
        has_extra: cutlass.Constexpr,
        per_token_len: cutlass.Constexpr,
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
        # REMAINDER-block grids shift head_base to the tail head range; the offset
        # const_expr is elided (==0) for the full-block / base path -> byte-identical.
        if cutlass.const_expr(self.head_block_offset != 0):
            head_block = head_block + Int32(self.head_block_offset)
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

        # P10b PER-TOKEN section length: this CTA's section_len = clamp(
        # topk_length[token_idx], 0, topk). Elided (per_token_len=False) on the
        # uniform / byte-identical path -> the scalar section_len passed in is used
        # unchanged. The main/extra chunk geometry (num_main_chunks, chunks_per_split)
        # stays UNIFORM over the MAX topk; this per-token clamp is what masks the
        # over-allocated chunks (their candidates fall past section_len -> S3 -inf).
        if cutlass.const_expr(per_token_len):
            topk_total = Int32(swa_indices.shape[1])
            section_len = Int32(topk_length[token_idx])
            if section_len < Int32(0):
                section_len = Int32(0)
            if section_len > topk_total:
                section_len = topk_total
            if cutlass.const_expr(has_extra):
                extra_total = Int32(extra_indices.shape[1])
                extra_section_len = Int32(extra_topk_length[token_idx])
                if extra_section_len < Int32(0):
                    extra_section_len = Int32(0)
                if extra_section_len > extra_total:
                    extra_section_len = extra_total

        # swa_indices for THIS token row: a 1-D (topk,) slice.
        topk_row = cute.make_tensor(
            swa_indices.iterator + token_idx.to(Int64) * Int64(swa_indices.stride[0]),
            cute.make_layout(swa_indices.shape[1]),
        )
        # extra_indices for THIS token row (DSV4 dual-cache). Built ONLY when
        # has_extra (extra_indices is None otherwise); const_expr-elided so the
        # no-extra trace never references the extra tensor.
        if cutlass.const_expr(has_extra):
            extra_row = cute.make_tensor(
                extra_indices.iterator + token_idx.to(Int64) * Int64(extra_indices.stride[0]),
                cute.make_layout(extra_indices.shape[1]),
            )
        else:
            extra_row = topk_row
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

                cute.arch.mbarrier_wait(mbar_base + n_buf + prod_idx, phase=prod_phase)

                tok_buf_view = cute.make_tensor(
                    token_idx_view.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )
                io_kw = dict(
                    bi=t.bi, kv_smem_stride=t.kv_smem_stride, rope_smem_stride=t.d_rope,
                    scale_bytes_per_token=8, bulk_tx_bytes=t.bulk_tx_bytes,
                    scale_format=t.scale_format,
                )
                # Per-chunk section dispatch (DSV4 dual-cache; FlashInfer
                # decode_dsv4 :243-322). chunks [0, num_main_chunks) gather from the
                # MAIN cache; chunks >= num_main_chunks gather from the EXTRA cache
                # (different base ptr / page size / indices / stride). is_extra is
                # uniform across the IO warp (derived from the chunk index) so the
                # runtime branch is divergence-free. When has_extra=False this is
                # const_expr-pinned to the main gather -> byte-identical PTX.
                if cutlass.const_expr(has_extra):
                    if ci >= num_main_chunks:
                        cis = ci - num_main_chunks
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
                        mbar_base + buf,
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
            n_acc_tiles = int(t.n_v_chunks) * int(t.nt_per_warp_xv)
            s0_quantize_q_to_smem(
                q_token, q_fp8_addr, q_sc_view, q_rope_addr, amax_view,
                head_base, Int32(self.valid_hpb), tid,
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

                # Per-chunk section dispatch for the S3 mask: compare the
                # candidate's offset WITHIN ITS SECTION against that section's
                # length. has_extra=False uses the EXACT pre-P7c expressions
                # (split_cand_start, section_len) verbatim -> byte-identical PTX.
                # An extra chunk (ci >= num_main_chunks) re-bases the offset and
                # swaps in the extra section length; the MATH (S0-S6b) above reads
                # only the buffered smem, so it is section-agnostic.
                if cutlass.const_expr(has_extra):
                    sc_start = split_cand_start
                    sec_len = section_len
                    if ci >= num_main_chunks:
                        sc_start = (ci - num_main_chunks) * Int32(_CAND_WINDOW)
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
                    n_v_chunks=t.n_v_chunks, hpb=t.hpb, n_warps=8, valid_hpb=self.valid_hpb,
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
                d_rope=t.d_rope, n_warps=8, valid_hpb=self.valid_hpb,
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

    DSV4 DUAL-CACHE (P7c): when ``indexed_k_cache`` / ``indexed_indices`` /
    ``indexed_topk_lengths`` are supplied (the "extra"-tokens second KV pool), the
    decode attends over the UNION of the MAIN paged topk cache and the EXTRA cache
    in ONE online softmax. The chunk loop processes ``num_main_chunks =
    ceil(topk/BI)`` main chunks (gathering from the main cache) then
    ``num_extra_chunks = ceil(extra_topk/BI)`` extra chunks (gathering from
    ``indexed_k_cache`` with ``indexed_page_size`` / its own per-block stride);
    ``num_splits`` spans both. The extra cache is DSV4-only (GLM has no extra
    section). With no extra cache the kernel is compiled with ``has_extra=False``
    so the extra-section code is const_expr-elided (PTX byte-identical).
    """
    from b12x.attention.mla.compressed_reference import compressed_mla_page_nbytes

    has_extra = (
        indexed_k_cache is not None
        or indexed_indices is not None
        or indexed_topk_lengths is not None
    )
    # Mapped extra page table is GENUINELY-UPSTREAM-UNSUPPORTED (upstream is
    # raw-slot-id only; no page-table indirection). RAISE, not fallback. Checked
    # BEFORE the has_extra branch so a mapped page table passed WITHOUT the extra
    # trio is still a hard error (never silently ignored / silently routed).
    if indexed_page_table is not None:
        raise ValueError(
            "unified_sm120 decode: indexed_page_table (mapped extra pages) is "
            "unsupported on SM120 sparse-MLA; upstream addresses the extra cache "
            "by raw slot id only"
        )
    if has_extra:
        # Partial dual-cache trio is a HARD ERROR (upstream ICHECKs extra_indices
        # requires extra_kv_cache; sparse_mla_sm120.cu:171-174). NOT a fallback.
        if (
            indexed_k_cache is None
            or indexed_indices is None
            or indexed_page_size is None
        ):
            raise ValueError(
                "unified_sm120 decode dual-cache requires indexed_k_cache, "
                "indexed_indices, and indexed_page_size together (partial extra "
                "trio is unsupported, matching upstream sparse_mla_sm120.cu:171-174)"
            )
        if int(q_all.shape[-1]) != _DSV4_HEAD_DIM:
            raise ValueError(
                "unified_sm120 decode dual-cache (extra tokens) is DSV4-only "
                "(q_head_dim==512); GLM/DSV3.2 has no extra cache"
            )

    q_head_dim = int(q_all.shape[-1])
    if q_head_dim not in (_DSV4_HEAD_DIM, _GLM_HEAD_DIM):
        raise NotImplementedError(
            f"unified_sm120 decode supports q_head_dim 512 (DSV4) or 576 (GLM); "
            f"got {q_head_dim} -- fall back to legacy"
        )

    rows, heads, _ = q_all.shape
    hpb = 16
    if heads <= 0:
        raise ValueError(f"unified_sm120 decode requires heads > 0, got {heads}")

    # attn_sink [num_heads] f32: upstream applies it in the DECODE MERGE
    # (sparse_mla_sm120_decode_dsv4.cu:128-129). Validate shape/dtype/device; the
    # fold itself is wired into the split.py sink-merge below (no kernel change).
    if attn_sink is not None:
        attn_sink = attn_sink.detach()
        if attn_sink.shape != (heads,):
            raise ValueError(
                f"unified_sm120 decode attn_sink must have shape [{heads}], "
                f"got {tuple(attn_sink.shape)}"
            )
        if attn_sink.dtype != torch.float32:
            raise TypeError(
                f"unified_sm120 decode attn_sink must be float32, got {attn_sink.dtype}"
            )
        if attn_sink.device != q_all.device:
            raise ValueError(
                "unified_sm120 decode attn_sink must be on the same device as q_all"
            )
        if not attn_sink.is_contiguous():
            raise ValueError("unified_sm120 decode attn_sink must be contiguous")
    if lse_scale not in ("base2", "natural"):
        raise ValueError(
            f"unified_sm120 decode lse_scale must be 'base2' or 'natural', got {lse_scale!r}"
        )
    # VALID_HPB<16 / non-multiple-of-16 heads (small-TP shards): upstream
    # VALID_HPB=min(NUM_HEADS,HPB) (decode_dsv4_kernel.cuh:152) computes a full
    # HPB=16 tile with zero-Q padding and gates writes to valid_hpb rows. We
    # realise this with up to TWO grid launches: ``h_blocks_full`` full blocks
    # (valid_hpb=16) plus one REMAINDER block (valid_hpb=heads%16) when heads is
    # not a multiple of 16. heads in {8} -> 0 full blocks + 1 remainder block of
    # valid_hpb=8. The base case (heads multiple of 16, e.g. 128) is a single
    # full-block grid -> byte-identical to the pre-P10 kernel.
    h_blocks_full = heads // hpb
    rem_heads = heads % hpb
    h_blocks = h_blocks_full + (1 if rem_heads else 0)

    model_type, compute_mode, scale_format = infer_model_type(q_head_dim, swa_k_cache.dtype)
    traits = make_unified_traits(model_type, compute_mode, scale_format)
    layout = make_smem_layout(traits)
    d_v = int(traits.d_v)  # output O dim (512 for both; V == nope for GLM)

    topk = int(swa_indices.shape[1])
    extra_topk = int(indexed_indices.shape[1]) if has_extra else 0
    num_main_chunks = (topk + _CAND_WINDOW - 1) // _CAND_WINDOW
    max_chunks = int(workspace.max_chunks_per_row)

    # ── P10b PER-TOKEN topk_length threading ──────────────────────────────────
    # Decide whether to route to the per-token kernel (section_len read per CTA
    # from a (rows,) int32 length tensor) or the byte-identical UNIFORM scalar
    # path. The scalar path is taken when every token's length equals the full
    # topk (the common decode contract: swa_topk_lengths[t] == topk for all t, OR
    # the caller -1-pads indices past the length so the uniform full-topk section
    # bound + the S3 idx<0 mask already realise the per-token length). For a
    # GENUINELY-mixed-length batch (some topk_length[t] < topk) the per-token
    # kernel reads each token's clamped length, so over-allocated chunks for short
    # tokens are fully masked (-> mid_lse=-inf -> merge ignores). A batch is uniform
    # ONLY when EVERY row's length >= the full section width (so the scalar bound
    # already equals each clamped length); a single SHORT row (lt[0] < cap) is NOT
    # uniform and must take the per-token clamp path.
    # CUDA-graph capture safety: the uniform-vs-per-token decision below reads a
    # data-dependent reduction off the device (torch.all(...).item()), which is a
    # device->host SYNC and is ILLEGAL during stream capture (cudaErrorStreamCapture
    # Unsupported). It is also fundamentally graph-unsafe: the length tensor values
    # can change between graph replays, so a length-dependent kernel SELECTION baked
    # at capture time would be wrong. Under capture we therefore SKIP the sync and
    # conservatively take the PER-TOKEN path (the per-token kernel reads each token's
    # clamped length, so it is correct for uniform batches too -- uniform is a subset
    # of per-token). Outside capture we keep the host-sync fast-path so the common
    # uniform decode stays on the byte-identical scalar kernel.
    capturing = q_all.is_cuda and torch.cuda.is_current_stream_capturing()

    def _length_tensor(lengths, name, cap):
        if lengths is None:
            return None, True
        if not isinstance(lengths, torch.Tensor):
            raise TypeError(f"unified_sm120 decode {name} must be a torch.Tensor")
        if lengths.shape != (rows,):
            raise ValueError(
                f"unified_sm120 decode {name} must have shape [{rows}], "
                f"got {tuple(lengths.shape)}"
            )
        lt = lengths.to(device=q_all.device, dtype=torch.int32).contiguous()
        # Under graph capture: no host sync allowed -> conservatively take the
        # PER-TOKEN path (uniform=False). The per-token kernel reads each token's
        # CLAMPED length, so it is correct for every case (uniform full-width AND a
        # genuinely-short single row); only the scalar fast-path is skipped. NOTE:
        # rows==1 is NOT automatically uniform here -- a single row whose length is
        # SHORTER than the section width (lt[0] < cap) must still be clamped, so it
        # is per-token, not scalar. (The earlier "rows==1 is always uniform" claim
        # conflated cross-token mixing with the per-token clamp and was wrong for a
        # short single row.)
        if capturing:
            return lt, False
        # Uniform iff every token's CLAMPED length is the full section width: then
        # the scalar section bound (Int32(cap)) already equals every token's length
        # -> byte-identical scalar path. A length >= cap clamps to cap in-kernel, so
        # >= (not ==) is the correct full-section test (seqlen can exceed topk).
        is_uniform = bool(torch.all(lt >= int(cap)).item())
        return lt, is_uniform

    swa_len_t, main_uniform = _length_tensor(swa_topk_lengths, "swa_topk_lengths", topk)
    if has_extra:
        extra_len_t, extra_uniform = _length_tensor(
            indexed_topk_lengths, "indexed_topk_lengths", extra_topk
        )
    else:
        extra_len_t, extra_uniform = None, True
    # Per-token kernel only when there IS a length tensor that is not uniform.
    per_token_len = (swa_len_t is not None and not main_uniform) or (
        has_extra and extra_len_t is not None and not extra_uniform
    )
    if per_token_len:
        # Both length tensors must exist for the per-token entries (the kernel reads
        # topk_length[t] / extra_topk_length[t]). Synthesize a full-length tensor
        # for whichever section is uniform / unset so the read collapses to topk.
        if swa_len_t is None:
            swa_len_t = torch.full((rows,), topk, dtype=torch.int32, device=q_all.device)
        if has_extra and extra_len_t is None:
            extra_len_t = torch.full(
                (rows,), extra_topk, dtype=torch.int32, device=q_all.device
            )
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
        extra_topk=extra_topk,
    )
    # Side-channel record of the chosen split plan (benchmarks / AutoTuner read
    # LAST_DECODE_PLAN["num_splits"]). Informational only.
    LAST_DECODE_PLAN.clear()
    LAST_DECODE_PLAN.update(
        model_type=str(model_type),
        topk=int(topk),
        extra_topk=int(extra_topk),
        has_extra=bool(has_extra),
        num_main_chunks=int(num_main_chunks),
        num_chunks=int(num_chunks),
        num_splits=int(num_splits),
        chunks_per_split=int(chunks_per_split),
        num_tokens=int(rows),
        h_blocks=int(h_blocks),
        sm_count=(int(sm_count) if sm_count else None),
        per_token_len=bool(per_token_len),
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

    # ── EXTRA (indexed) cache views. When there is no extra cache they alias the
    #    main cache / main indices and the kernel's has_extra=False const_expr
    #    elides the extra-section reads -> PTX byte-identical. ──
    if has_extra:
        pbs_extra = int(indexed_page_size)
        # The extra cache uses the IDENTICAL DSV4 compressed packed byte layout, so
        # its per-block stride is compressed_mla_page_nbytes(pbs_extra) (same as
        # the main cache derives from swa_page_size). DSV4-only here.
        stride_extra_kv_block = int(compressed_mla_page_nbytes(pbs_extra))
        extra_kv_flat = indexed_k_cache.reshape(-1)
        extra_indices_t = indexed_indices.contiguous()
    else:
        pbs_extra = 1
        stride_extra_kv_block = 0
        extra_kv_flat = swa_k_cache.reshape(-1)  # alias (never read when has_extra=False)
        extra_indices_t = swa_indices            # alias (never read)

    output = workspace.output_buffer[:rows, :heads, :d_v]

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kv_flat = swa_k_cache.reshape(-1)
    if per_token_len:
        # PER-TOKEN entry args: the (rows,) int32 length tensor(s) replace the
        # scalar section_len / extra_section_len (section_len read per CTA in-kernel
        # at t=blockIdx.x). num_main_chunks stays a UNIFORM scalar (main/extra chunk
        # split over the MAX topk); per-token clamping masks short-token chunks.
        pertok_base = (
            _to_cute(q_all, cutlass.BFloat16),
            _to_cute(kv_flat, cutlass.Uint8, align=16),
            _to_cute(swa_indices, cutlass.Int32, align=4),
            _to_cute(mid_out, cutlass.BFloat16, align=16),
            _to_cute(mid_lse, cutlass.Float32, align=4),
            Float32(float(sm_scale) * LOG2_E),
            _to_cute(swa_len_t, cutlass.Int32, align=4),
            Int64(stride_kv_block),
        )
        if has_extra:
            args = pertok_base + (
                _to_cute(extra_kv_flat, cutlass.Uint8, align=16),
                _to_cute(extra_indices_t, cutlass.Int32, align=4),
                _to_cute(extra_len_t, cutlass.Int32, align=4),
                Int32(num_main_chunks),
                Int64(stride_extra_kv_block),
                stream,
            )
        else:
            args = pertok_base + (stream,)
    else:
        # Base (single-cache) args -- EXACTLY the v1 traced signature. The no-extra
        # path passes ONLY these, so its trace + PTX stay byte-identical.
        base_args = (
            _to_cute(q_all, cutlass.BFloat16),
            _to_cute(kv_flat, cutlass.Uint8, align=16),
            _to_cute(swa_indices, cutlass.Int32, align=4),
            _to_cute(mid_out, cutlass.BFloat16, align=16),
            _to_cute(mid_lse, cutlass.Float32, align=4),
            Float32(float(sm_scale) * LOG2_E),
            Int32(topk),
            Int64(stride_kv_block),
        )
        if has_extra:
            args = base_args + (
                _to_cute(extra_kv_flat, cutlass.Uint8, align=16),
                _to_cute(extra_indices_t, cutlass.Int32, align=4),
                Int32(extra_topk),
                Int32(num_main_chunks),
                Int64(stride_extra_kv_block),
                stream,
            )
        else:
            args = base_args + (stream,)

    def _launch_grid(grid_h_blocks: int, valid_hpb: int, head_block_offset: int):
        # Build + launch ONE grid of ``grid_h_blocks`` head-blocks with a fixed
        # ``valid_hpb`` const_expr and a ``head_block_offset`` that shifts head_base
        # to the right head range. The full-block grid (valid_hpb=16, offset=0,
        # heads a multiple of 16) is the byte-identical base path; the remainder
        # grid (valid_hpb<16, offset=h_blocks_full) handles the tail head-block.
        kernel = UnifiedDecodeKernel(
            traits, layout, int(swa_page_size), chunks_per_split,
            num_tokens=rows, h_blocks=int(grid_h_blocks), num_splits=num_splits,
            has_extra=has_extra, pbs_extra=pbs_extra,
            valid_hpb=int(valid_hpb), head_block_offset=int(head_block_offset),
            per_token_len=per_token_len,
        )
        spec_fields = [
            key_field("model_type", traits.model_type),
            key_field("compute_mode", traits.compute_mode),
            key_field("scale_format", traits.scale_format),
            key_field("num_heads", int(heads)),
            key_field("hpb", int(hpb)),
            # valid_hpb + head_block_offset specialize the VALID_HPB<16 / tail
            # head-block const_expr (full-block grid keeps valid_hpb=16, offset=0
            # -> PTX byte-identical to the pre-P10 kernel).
            key_field("valid_hpb", int(valid_hpb)),
            key_field("head_block_offset", int(head_block_offset)),
            key_field("grid_h_blocks", int(grid_h_blocks)),
            key_field("chunks_per_split", int(chunks_per_split)),
            key_field("page_block_size", int(swa_page_size)),
            key_field("topk_bucket", _topk_bucket(topk)),
            # has_extra + pbs_extra + extra_topk_bucket specialize the dual-cache
            # kernel: has_extra gates the extra-section const_expr (no-extra DSV4 PTX
            # byte-identical), pbs_extra is the runtime extra page block size, and the
            # extra topk bucket keeps the key compact.
            key_field("has_extra", int(has_extra)),
            key_field("pbs_extra", int(pbs_extra)),
            key_field("extra_topk_bucket", _topk_bucket(extra_topk) if has_extra else 0),
            # per_token_len gates the P10b per-token section_len const_expr: the
            # uniform (per_token_len=0) trace + PTX are byte-identical to the
            # pre-P10b kernel (the length tensor never enters that device entry).
            key_field("per_token_len", int(per_token_len)),
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
        ]
        if has_extra:
            spec_fields.append(
                tensor_key("extra_indices", extra_indices_t, dims=(DimKey.exact(rows), DimKey.bucket(max(extra_topk, 1))))
            )
        if per_token_len:
            # The (rows,) int32 length tensor(s) enter the per-token device entry; key
            # their row dim (exact rows, baked into the grid).
            spec_fields.append(
                tensor_key("topk_length", swa_len_t, dims=(DimKey.exact(rows),))
            )
            if has_extra:
                spec_fields.append(
                    tensor_key("extra_topk_length", extra_len_t, dims=(DimKey.exact(rows),))
                )
        compile_spec = KernelCompileSpec.from_fields(
            "attention.mla.unified_sm120.decode",
            # version 3: P10b per-token topk_length entries (kernel_pertok /
            # kernel_extra_pertok + per_token_len const_expr). The cache key is
            # kernel_id + version + fields; this bump invalidates stale v2 objects.
            # The uniform (per_token_len=0) trace + PTX are unchanged from v2 (the
            # length tensor never enters the uniform device entry), and the
            # per_token_len key_field keeps the per-token entries on distinct keys.
            3,
            *spec_fields,
        )
        # Select the entry method: each (single/dual cache) x (uniform/per-token)
        # combination has a DISTINCT mangled name. The uniform single-cache path
        # uses the kernel object (-> __call__) whose name + PTX stay byte-identical
        # to the pre-P7c/P10b kernel; the others use their named entries.
        if per_token_len:
            entry = kernel.call_extra_pertok if has_extra else kernel.call_pertok
        else:
            entry = kernel.call_extra if has_extra else kernel
        b12x_launch(
            entry,
            compile_spec=compile_spec,
            compile_args=args,
            runtime_args=args,
        )

    if h_blocks_full > 0:
        # FULL HPB=16 head-blocks (the base path when heads is a multiple of 16).
        _launch_grid(h_blocks_full, hpb, 0)
    if rem_heads:
        # REMAINDER tail head-block: a 1-block grid with valid_hpb=rem_heads at
        # head_block offset h_blocks_full (so head_base = h_blocks_full*16).
        _launch_grid(1, rem_heads, h_blocks_full)

    # ── REUSED base-2 merge over the split axis -> final O. num_splits=1 is the
    #    trivial 1-split merge (partial == final O). ──
    from b12x.attention.mla.split import (
        build_sparse_mla_split_decode_merge_binding,
        run_sparse_mla_split_decode_merge,
    )

    if int(workspace.num_chunks_value or -1) != num_splits:
        workspace.set_split_chunk_config(kv_chunk_size=_CAND_WINDOW, num_chunks=num_splits)

    # When attn_sink is supplied, the merge SELECTS the sink-folding merge kernel
    # (split.py SparseMLASplitDecodeSinkMergeKernel) which applies the FlashMLA V4
    # fold output *= sigmoid(lse_e - sink) directly into O (exactly upstream's
    # sink-in-merge design, sparse_mla_sm120_decode_dsv4.cu:128-129). With no sink
    # it is the plain base-2 merge -> PTX/numerics byte-identical to the base path.
    merge_binding = build_sparse_mla_split_decode_merge_binding(
        tmp_output=mid_out,
        tmp_lse=mid_lse,
        num_chunks_ptr=workspace.num_chunks_ptr,
        output=output,
        attn_sink=attn_sink,
        workspace=workspace,
    )
    run_sparse_mla_split_decode_merge(binding=merge_binding)
    if not return_lse:
        return output

    # return_lse: reconstruct the FINAL LSE from the per-split base-2 mid_lse
    # (logsumexp over the split axis, base2->natural). mid_lse aliases
    # workspace.tmp_lse[:rows,:heads,:num_splits], so reuse the shared helper.
    from b12x.attention.mla.api import _final_lse_from_split_workspace

    lse_natural = _final_lse_from_split_workspace(
        workspace=workspace,
        q_rows=rows,
        num_heads=heads,
        launch_num_chunks=num_splits,
        scale="natural",
    )
    if attn_sink is not None:
        # Fold the per-head sink into the LSE in the natural-log domain (the merge
        # already folded it into O): lse' = log(exp(lse) + exp(sink)).
        sink = attn_sink.float().view(1, heads)
        lse_natural = torch.logaddexp(lse_natural.float(), sink)
    if lse_scale == "base2":
        return output, (lse_natural / _LN2)
    return output, lse_natural


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
