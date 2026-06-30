"""FlashInfer-shaped SM120 DSV4/GLM MG prefill path.

One CTA handles up to two HPB=16 head groups and reuses a single NoPE KV gather
across them. GLM TP8 uses a half-full head group and packs the accurate PV
HIGH/LOW residual pair into the lower/upper rows of one m16 MMA. The DSV4 RoPE
payload is intentionally not bulk-staged into smem; QK-RoPE and XV-RoPE read it
from global/L2.
"""

from __future__ import annotations

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.attention._cute.ops import LOG2_E
from b12x.cute.compiler import DimKey, KernelCompileSpec, key_field, tensor_key
from b12x.cute.compiler import launch as b12x_launch
from b12x.cute.fp4 import (
    atomic_max_shared_f32,
    dequant_kv_e4m3_pair_to_bf16x2,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    st_shared_bf16_from_f32,
    st_shared_u8,
)

from .decode_math import (
    EPILOGUE_FINAL_BF16,
    _d2_load_b_fp8,
    _exp2_approx_ftz_f32,
    _ld_u16_zext,
    _ld_u8_zext,
    _quant_e4m3_byte,
    _ue8m0_byte_to_fp32,
    ld_shared_f32,
    s0_quantize_q_to_smem,
    s1_qk_nope_block_scaled,
    s3_mask_and_scale,
    s6_xv_nope,
    st_shared_f32,
    s7_epilogue,
)
from .io_mg import io_issue_gather_dsv4_nope, io_issue_gather_glm_mg
from .smem_mg import get_prefill_mg_shared_storage_cls, make_smem_layout_mg
from .traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits


_CAND_WINDOW = 64
_DSV4_HEAD_DIM = 512
_GLM_HEAD_DIM = 576
_PREFILL_BLOCK_THREADS = 384
_PREFILL_IO_THREADS = 128
_IO_REGS = 32
_MATH_REGS = 232
_DSV4_IO_STRIDE = 576
_DSV4_ROPE_GMEM_OFFSET = 448
# GLM (ARBITRARY_FP32) per-token record: 512 e4m3 nope + 16 inline fp32 scales +
# 128 bf16 rope == 656B. RoPE byte offset within the record == 528 (D_NOPE +
# 4*4 inline-scale bytes). MG reads it from global/L2 (no smem staging).
_GLM_IO_STRIDE = 656
_GLM_ROPE_GMEM_OFFSET = 528


@cute.jit
def _wfp8_row_xor(row: Int32) -> Int32:
    """FlashInfer DSV4-dual W (A-operand) smem-bank-conflict swizzle.

    Matches ``wfp8_row_xor`` in ldmatrix_sm120.cuh:76 (``row ^ (row >> 3)``).
    A self-inverse bijection on [0, 15] (it flips bit 0 with bit 3), so applying
    it symmetrically at the W store AND the ldmatrix load is numerically identity
    -- it is a pure smem-bank-conflict swizzle. Only the DSV4 dual ``pbs_extra==2``
    XV path enables it (USE_WFP8_ROW_XOR = DUAL_CACHE && pbs_extra==2,
    prefill_kernel.cuh:624); every other caller passes ``row_xor=False`` and stays
    byte-identical (the const_expr gate elides this call so the address arithmetic
    is textually unchanged).
    """
    return row ^ (row >> Int32(3))


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


def _cache_base_tensor(cache: torch.Tensor) -> torch.Tensor:
    if cache.is_contiguous():
        return cache.reshape(-1)
    if cache.ndim < 2:
        return cache

    # Packed vLLM cache views retain page subdimensions and a storage_offset
    # into a larger per-block allocation. The MG kernels do raw pointer-offset
    # indexing, so expose this layer's full physical span as a 1-D view and keep
    # the packed block stride in the explicit stride argument.
    span = 1
    for size, stride in zip(cache.shape, cache.stride(), strict=True):
        span += (int(size) - 1) * int(stride)
    return torch.as_strided(cache, size=(span,), stride=(1,))


def _cache_block_stride_bytes(
    cache: torch.Tensor,
    *,
    page_size: int,
    is_glm: bool,
) -> int:
    from b12x.attention.mla.compressed_reference import compressed_mla_page_nbytes

    if is_glm:
        expected = int(page_size) * _GLM_IO_STRIDE
    else:
        expected = int(compressed_mla_page_nbytes(int(page_size)))
    # Contiguous inputs are flattened before launch, so their original rank is
    # not a physical-layout contract and the standard page stride applies.
    # Packed vLLM page views are non-contiguous and carry the physical
    # per-block stride in dimension 0.
    if not cache.is_contiguous() and cache.ndim >= 2:
        stride = int(cache.stride(0)) * int(cache.element_size())
        if stride < expected:
            raise ValueError(
                f"SM120 sparse MLA prefill cache block stride {stride} is "
                f"smaller than page payload {expected}"
            )
        return stride
    return expected


def _topk_bucket(topk: int) -> int:
    return 1 << (max(int(topk), 1) - 1).bit_length()


@cute.jit
def _smem_byte(base_addr: Int32, byte_off) -> Int32:
    return base_addr + Int32(byte_off)


@cute.jit
def _dsv4_rope_base_off(idx: Int32, page_block_size: Int32, stride_kv_block: Int64) -> Int64:
    if idx < Int32(0):
        idx = Int32(0)
    block_idx = idx // page_block_size
    local_idx = idx - block_idx * page_block_size
    return (
        Int64(block_idx) * stride_kv_block
        + Int64(local_idx) * Int64(_DSV4_IO_STRIDE)
        + Int64(_DSV4_ROPE_GMEM_OFFSET)
    )


@cute.jit
def _ld_global_dsv4_rope_b16(
    kv_cache_u8: cute.Tensor,
    idx: Int32,
    dim: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
) -> Uint32:
    out = Uint32(0)
    if idx >= Int32(0):
        dim_even = dim & ~Int32(1)
        byte_off = (
            _dsv4_rope_base_off(idx, page_block_size, stride_kv_block)
            + Int64(dim_even) * Int64(2)
        )
        word = ld_global_nc_u32(get_ptr_as_int64(kv_cache_u8, byte_off))
        if (dim & Int32(1)) != Int32(0):
            out = (word >> Uint32(16)) & Uint32(0xFFFF)
        else:
            out = word & Uint32(0xFFFF)
    return out


@cute.jit
def s2_qk_rope_global_dsv4(
    qk,
    q_rope_base_addr: Int32,
    kv_cache_u8: cute.Tensor,
    token_idx_view: cute.Tensor,
    warp_first_cand: Int32,
    lane: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
    *,
    d_rope: cutlass.Constexpr,
):
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    entry = warp_first_cand + gid
    idx = Int32(token_idx_view[entry])
    rope_base = _dsv4_rope_base_off(idx, page_block_size, stride_kv_block)

    for ks in cutlass.range_constexpr(d_rope // 16):
        ko = Int32(ks) * Int32(16)
        a_byte = a_row * Int32(d_rope * 2) + (ko + a_col) * Int32(2)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_byte(q_rope_base_addr, a_byte))
        b0 = ld_global_nc_u32(
            get_ptr_as_int64(kv_cache_u8, rope_base + Int64(ko + tid * Int32(2)) * Int64(2))
        )
        b1 = ld_global_nc_u32(
            get_ptr_as_int64(
                kv_cache_u8,
                rope_base + Int64(ko + Int32(8) + tid * Int32(2)) * Int64(2),
            )
        )
        qk[0], qk[1], qk[2], qk[3] = mma_m16n8k16_f32_bf16(
            qk[0], qk[1], qk[2], qk[3], a0, a1, a2, a3, b0, b1
        )
    return qk


@cute.jit
def s2_qk_rope_global_mg_dsv4(
    qk0,
    qk1,
    q_rope_g0_addr: Int32,
    q_rope_g1_addr: Int32,
    kv_cache_u8: cute.Tensor,
    token_idx_view: cute.Tensor,
    warp_first_cand: Int32,
    lane: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
    *,
    d_rope: cutlass.Constexpr,
    n_hg: cutlass.Constexpr = 2,
    valid_hpb: cutlass.Constexpr = 16,
):
    """Fused DSV4 QK-RoPE.

    The KV-RoPE B operand (b0/b1) depends only on the candidate token and the
    rope chunk -- NOT on the head group. FlashInfer's prefill prefetches it once
    per tile (``prefetch_kv_rope``) and reuses it for both groups
    (``compute_qk_rope`` x MG_N_HG). This fuses the per-group
    ``s2_qk_rope_global_dsv4`` calls so the vectorized (b16-pair = nc.u32) KV-RoPE
    gather runs ONCE per CTA tile instead of once per head group, halving the
    QK-RoPE global loads. Numerically identical: same B operands, same MMAs. When
    ``n_hg==1`` the group-1 ldmatrix+MMA const_expr-elides.
    """
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    entry = warp_first_cand + gid
    idx = Int32(token_idx_view[entry])
    rope_base = _dsv4_rope_base_off(idx, page_block_size, stride_kv_block)

    for ks in cutlass.range_constexpr(d_rope // 16):
        ko = Int32(ks) * Int32(16)
        # KV-RoPE B operand: loaded once, shared across both head groups.
        b0 = ld_global_nc_u32(
            get_ptr_as_int64(kv_cache_u8, rope_base + Int64(ko + tid * Int32(2)) * Int64(2))
        )
        b1 = ld_global_nc_u32(
            get_ptr_as_int64(
                kv_cache_u8,
                rope_base + Int64(ko + Int32(8) + tid * Int32(2)) * Int64(2),
            )
        )
        a_byte = a_row * Int32(d_rope * 2) + (ko + a_col) * Int32(2)
        a00, a01, a02, a03 = ldmatrix_m8n8x4_b16(_smem_byte(q_rope_g0_addr, a_byte))
        qk0[0], qk0[1], qk0[2], qk0[3] = mma_m16n8k16_f32_bf16(
            qk0[0], qk0[1], qk0[2], qk0[3], a00, a01, a02, a03, b0, b1
        )
        if cutlass.const_expr(n_hg == 2):
            a10, a11, a12, a13 = ldmatrix_m8n8x4_b16(_smem_byte(q_rope_g1_addr, a_byte))
            qk1[0], qk1[1], qk1[2], qk1[3] = mma_m16n8k16_f32_bf16(
                qk1[0], qk1[1], qk1[2], qk1[3], a10, a11, a12, a13, b0, b1
            )
    return qk0, qk1


# =============================================================================
# GLM (ARBITRARY_FP32) MG QK-RoPE from global/L2.
#
# Like the DSV4 MG path, GLM MG reads the KV-RoPE B operand from global/L2 (no
# smem staging), so the 528-stride GLM KV fits the carveout for mg_n_hg==2. The
# GLM record is 656B (512 nope + 16 inline fp32 scales + 128 bf16 rope), so the
# rope byte offset within a token record is 528. The numerics are identical to
# the validated GLM decode path (decode_math.s2_qk_rope_bf16): same bf16 rope
# values, same per-lane scalar packing into the bf16 m16n8k16 MMA B operand.
# =============================================================================
@cute.jit
def _glm_rope_base_off(idx: Int32, page_block_size: Int32, stride_kv_block: Int64) -> Int64:
    if idx < Int32(0):
        idx = Int32(0)
    block_idx = idx // page_block_size
    local_idx = idx - block_idx * page_block_size
    return (
        Int64(block_idx) * stride_kv_block
        + Int64(local_idx) * Int64(_GLM_IO_STRIDE)
        + Int64(_GLM_ROPE_GMEM_OFFSET)
    )


@cute.jit
def _ld_global_glm_rope_u32(
    kv_cache_u8: cute.Tensor,
    rope_base: Int64,
    elem: Int32,
) -> Uint32:
    """Load two consecutive bf16 rope elems (one u32) from global at rope_base +
    elem*2 bytes. ``elem`` is even (the lane reads bf16 pairs)."""
    return ld_global_nc_u32(get_ptr_as_int64(kv_cache_u8, rope_base + Int64(elem) * Int64(2)))


@cute.jit
def s2_qk_rope_regs_mg_glm(
    qk0,
    qk1,
    q_rope_regs0,
    q_rope_regs1,
    kv_cache_u8: cute.Tensor,
    token_idx_view: cute.Tensor,
    warp_first_cand: Int32,
    lane: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
    *,
    d_rope: cutlass.Constexpr,
    n_hg: cutlass.Constexpr = 2,
    valid_hpb: cutlass.Constexpr = 16,
):
    """GLM MG QK-RoPE. Mirrors ``s2_qk_rope_regs_mg_dsv4`` (registerized Q-rope A,
    KV-rope B from global) but with the GLM record geometry and the GLM B-operand
    packing (decode_math.s2_qk_rope_bf16): each lane's K-rope entry is
    ``warp_first_cand + gid`` (NOT the DSV4 nt/tid layout), and b0/b1 are two
    consecutive bf16 rope-pairs of THAT one entry's rope row. The B operand
    depends only on the candidate token + rope chunk, so it is gathered once and
    reused across both head groups (n_hg==2)."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    entry = warp_first_cand + gid
    idx = Int32(token_idx_view[entry])
    rope_base = _glm_rope_base_off(idx, page_block_size, stride_kv_block)
    hi0 = cutlass.const_expr(valid_hpb > 8)

    for ks in cutlass.range_constexpr(d_rope // 16):
        ko = Int32(ks) * Int32(16)
        # decode_math.s2_qk_rope_bf16 B packing: b0 = rope[ko + tid*2 .. +1],
        # b1 = rope[ko + tid*2 + 8 .. +9] of this entry's rope row.
        b0 = _ld_global_glm_rope_u32(kv_cache_u8, rope_base, ko + tid * Int32(2))
        b1 = _ld_global_glm_rope_u32(kv_cache_u8, rope_base, ko + tid * Int32(2) + Int32(8))
        base = Int32(ks) * Int32(4)
        d0, d1, d2, d3 = mma_m16n8k16_f32_bf16(
            qk0[0], qk0[1], qk0[2], qk0[3],
            q_rope_regs0[base + 0], q_rope_regs0[base + 1],
            q_rope_regs0[base + 2], q_rope_regs0[base + 3],
            b0, b1,
        )
        qk0[0] = d0
        qk0[1] = d1
        if cutlass.const_expr(hi0):
            qk0[2] = d2
            qk0[3] = d3
        if cutlass.const_expr(n_hg == 2):
            qk1[0], qk1[1], qk1[2], qk1[3] = mma_m16n8k16_f32_bf16(
                qk1[0], qk1[1], qk1[2], qk1[3],
                q_rope_regs1[base + 0], q_rope_regs1[base + 1],
                q_rope_regs1[base + 2], q_rope_regs1[base + 3],
                b0, b1,
            )
    return qk0, qk1


# =============================================================================
# BF16-QK (ComputeMode.BF16) MG specialization
#
# Mirrors FlashInfer's sparse_mla prefill BF16-QK path (prefill_kernel.cuh
# :788-931 / common/q_rope.cuh): S0 loads Q-NoPE straight to BF16 smem (NO FP8
# Q-quant prologue); S1 QK-NoPE does per-thread inline FP8->BF16 K dequant
# (cvt.rn.f16x2.e4m3x2 * UE8M0 fp32 scale -> cvt.rn.bf16x2.f32) and a bf16
# m16n8k16 MMA, fusing the two head groups so the K dequant + B operand run ONCE
# per K-step. XV (S5/S6) stays FP8 -- unchanged from the FP8 path.
# =============================================================================
@cute.jit
def s0_load_q_bf16_to_smem_mg(
    q_token: cute.Tensor,            # (NUM_HEADS, D_QK) bf16 view for this token
    q_nope_bf16_g0_addr: Int32,      # smem addr of group-0 Q-NoPE bf16 buffer
    q_nope_bf16_g1_addr: Int32,      # smem addr of group-1 Q-NoPE bf16 buffer
    q_rope_g0_addr: Int32,           # smem addr of group-0 Q-rope bf16 scratch
    q_rope_g1_addr: Int32,           # smem addr of group-1 Q-rope bf16 scratch
    head_base: Int32,                # first head index of this CTA (h_start)
    tid: Int32,
    *,
    d_nope: cutlass.Constexpr,       # 448
    d_rope: cutlass.Constexpr,       # 64
    hpb: cutlass.Constexpr,          # 16
    q_nope_bf16_stride: cutlass.Constexpr,  # D_NOPE + 8
    num_threads: cutlass.Constexpr,  # 256
    barrier_id: cutlass.Constexpr,
    n_hg: cutlass.Constexpr = 2,
    valid_hpb: cutlass.Constexpr = 16,
):
    """S0 (BF16): cooperative gmem->smem BF16 copy of Q-NoPE + Q-rope for the
    ``n_hg`` head groups. Counterpart to ``s0_quantize_q_to_smem`` -- NO FP8
    quant, no amax/scale (FlashInfer ``load_q_bf16_to_smem``). All HPB heads of
    each group are valid (heads % HPB == 0). The cooperative loops span
    ``n_hg * hpb * d_*`` (one group for n_hg==1); the per-group g==1 store target
    selection const_expr-elides for n_hg==1 since the loop never reaches g==1.

    ``valid_hpb`` is smaller than ``hpb`` only for the heads==8 small-TP shard.
    Those inactive rows are zero-filled so the M=16 QK MMA remains well-defined
    while the epilogue gates stores to the real rows only."""
    bar_kw = dict(barrier_id=barrier_id, number_of_threads=num_threads)

    # Q-NoPE -> bf16 smem (all groups). group g head h dim d.
    i = tid
    while i < Int32(n_hg * hpb * d_nope):
        g = i // Int32(hpb * d_nope)
        rem = i - g * Int32(hpb * d_nope)
        h = rem // Int32(d_nope)
        d = rem - h * Int32(d_nope)
        dst = q_nope_bf16_g0_addr
        if cutlass.const_expr(n_hg == 2):
            if g != Int32(0):
                dst = q_nope_bf16_g1_addr
        if cutlass.const_expr(valid_hpb == hpb):
            val = Float32(q_token[head_base + g * Int32(hpb) + h, d])
        else:
            val = Float32(0.0)
            if h < valid_hpb:
                val = Float32(q_token[head_base + g * Int32(hpb) + h, d])
        st_shared_bf16_from_f32(dst + (h * Int32(q_nope_bf16_stride) + d) * Int32(2), val)
        i += Int32(num_threads)

    # Q-rope -> bf16 smem scratch (all groups). Stride D_ROPE (no pad).
    i = tid
    while i < Int32(n_hg * hpb * d_rope):
        g = i // Int32(hpb * d_rope)
        rem = i - g * Int32(hpb * d_rope)
        h = rem // Int32(d_rope)
        d = rem - h * Int32(d_rope)
        dst = q_rope_g0_addr
        if cutlass.const_expr(n_hg == 2):
            if g != Int32(0):
                dst = q_rope_g1_addr
        if cutlass.const_expr(valid_hpb == hpb):
            val = Float32(q_token[head_base + g * Int32(hpb) + h, Int32(d_nope) + d])
        else:
            val = Float32(0.0)
            if h < valid_hpb:
                val = Float32(q_token[head_base + g * Int32(hpb) + h, Int32(d_nope) + d])
        st_shared_bf16_from_f32(dst + (h * Int32(d_rope) + d) * Int32(2), val)
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)


@cute.jit
def preload_q_rope_regs_mg(
    q_rope_base_addr: Int32,
    lane: Int32,
    *,
    d_rope: cutlass.Constexpr,        # 64
):
    """Preload one head group's Q-rope (bf16) A operands into registers, ONCE
    before the main loop (FlashInfer ``preload_q_rope_regs``). Returns a flat
    python list of (d_rope//16)*4 Uint32 ldmatrix.x4 fragments. The Q-rope smem
    scratch then aliases the W_FP8 region (only used in S6)."""
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    regs = []
    for ks in cutlass.range_constexpr(d_rope // 16):
        ko = Int32(ks) * Int32(16)
        a_byte = a_row * Int32(d_rope * 2) + (ko + a_col) * Int32(2)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_byte(q_rope_base_addr, a_byte))
        regs.append(a0)
        regs.append(a1)
        regs.append(a2)
        regs.append(a3)
    return regs


@cute.jit
def s1_qk_nope_bf16_mg2(
    qk0,
    qk1,
    q_nope_bf16_g0_addr: Int32,
    q_nope_bf16_g1_addr: Int32,
    kv_fp8_base_addr: Int32,
    kv_sc_base_addr: Int32,
    warp_first_cand: Int32,
    lane: Int32,
    *,
    num_scales: cutlass.Constexpr,    # 7
    quant_tile: cutlass.Constexpr,    # 64
    q_nope_bf16_stride: cutlass.Constexpr,  # D_NOPE + 8
    kv_smem_stride: cutlass.Constexpr,  # 448 (BF16 layout)
    scale_bytes_per_token: cutlass.Constexpr,  # 8
    n_hg: cutlass.Constexpr = 2,
):
    """S1 (BF16): fused QK-NoPE bf16 m16n8k16 MMA with per-thread inline
    FP8->BF16 K dequant.

    Byte-for-byte mirror of FlashInfer prefill_kernel.cuh:883-932. The K B
    operand (b0/b1) and the per-(token,blk) UE8M0 scale depend only on the
    candidate token, NOT on the head group, so the dequant runs ONCE per K-step
    and feeds both groups' bf16 MMAs. NUM_SCALES blk x QUANT_TILE/16 ks = 7*4=28
    bf16 m16n8k16 MMAs per group. NO block-scaled FP8 MMA. When ``n_hg==1`` the
    group-1 ldmatrix+MMA const_expr-elides (single-group MG, heads==16).
    """
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    # ldmatrix A (Q bf16 16x16): row/col -- ldmatrix_load_A_bf16.
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    # The K row this lane group reads for the dequant B operand.
    kv_gid_row = warp_first_cand + gid

    for blk in cutlass.range_constexpr(num_scales):
        # DSV4 UE8M0_BYTE: per-(token,blk) fp32 scale from the K footer scale buf.
        sc_byte_off = kv_gid_row * Int32(scale_bytes_per_token) + Int32(blk)
        scale_f = _ue8m0_byte_to_fp32(_ld_u8_zext(kv_sc_base_addr, sc_byte_off))
        for ks in cutlass.range_constexpr(quant_tile // 16):
            ko = Int32(blk) * Int32(quant_tile) + Int32(ks) * Int32(16)
            # K dequant B operand: two e4m3 byte-pairs (u16) -> bf16x2 b0/b1.
            # _ld_u16_zext handles the 2-byte (non-4-aligned) offsets safely.
            kv_row_base = kv_gid_row * Int32(kv_smem_stride) + ko
            p0 = _ld_u16_zext(kv_fp8_base_addr, kv_row_base + tid * Int32(2))
            p1 = _ld_u16_zext(kv_fp8_base_addr, kv_row_base + tid * Int32(2) + Int32(8))
            b0, b1 = dequant_kv_e4m3_pair_to_bf16x2(p0, p1, scale_f)
            # Group 0 A operand + MMA.
            a_byte = a_row * Int32(q_nope_bf16_stride * 2) + (ko + a_col) * Int32(2)
            a00, a01, a02, a03 = ldmatrix_m8n8x4_b16(
                _smem_byte(q_nope_bf16_g0_addr, a_byte)
            )
            qk0[0], qk0[1], qk0[2], qk0[3] = mma_m16n8k16_f32_bf16(
                qk0[0], qk0[1], qk0[2], qk0[3], a00, a01, a02, a03, b0, b1
            )
            # Group 1 A operand + MMA (same b0/b1). Elided when n_hg==1.
            if cutlass.const_expr(n_hg == 2):
                a10, a11, a12, a13 = ldmatrix_m8n8x4_b16(
                    _smem_byte(q_nope_bf16_g1_addr, a_byte)
                )
                qk1[0], qk1[1], qk1[2], qk1[3] = mma_m16n8k16_f32_bf16(
                    qk1[0], qk1[1], qk1[2], qk1[3], a10, a11, a12, a13, b0, b1
                )
    return qk0, qk1


@cute.jit
def s2_qk_rope_regs_mg_dsv4(
    qk0,
    qk1,
    q_rope_regs0,
    q_rope_regs1,
    kv_cache_u8: cute.Tensor,
    token_idx_view: cute.Tensor,
    warp_first_cand: Int32,
    lane: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
    *,
    d_rope: cutlass.Constexpr,
    n_hg: cutlass.Constexpr = 2,
):
    """Fused DSV4 QK-RoPE for the BF16 path. Identical to
    ``s2_qk_rope_global_mg_dsv4`` except the Q-rope A operands come from the
    preloaded registers (the Q-rope smem aliases W_FP8). KV-RoPE B operand is
    gathered ONCE per CTA tile and reused across both groups. When ``n_hg==1``
    the group-1 MMA const_expr-elides."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    entry = warp_first_cand + gid
    idx = Int32(token_idx_view[entry])
    rope_base = _dsv4_rope_base_off(idx, page_block_size, stride_kv_block)

    for ks in cutlass.range_constexpr(d_rope // 16):
        ko = Int32(ks) * Int32(16)
        b0 = ld_global_nc_u32(
            get_ptr_as_int64(kv_cache_u8, rope_base + Int64(ko + tid * Int32(2)) * Int64(2))
        )
        b1 = ld_global_nc_u32(
            get_ptr_as_int64(
                kv_cache_u8,
                rope_base + Int64(ko + Int32(8) + tid * Int32(2)) * Int64(2),
            )
        )
        base = Int32(ks) * Int32(4)
        qk0[0], qk0[1], qk0[2], qk0[3] = mma_m16n8k16_f32_bf16(
            qk0[0],
            qk0[1],
            qk0[2],
            qk0[3],
            q_rope_regs0[base + 0],
            q_rope_regs0[base + 1],
            q_rope_regs0[base + 2],
            q_rope_regs0[base + 3],
            b0,
            b1,
        )
        if cutlass.const_expr(n_hg == 2):
            qk1[0], qk1[1], qk1[2], qk1[3] = mma_m16n8k16_f32_bf16(
                qk1[0],
                qk1[1],
                qk1[2],
                qk1[3],
                q_rope_regs1[base + 0],
                q_rope_regs1[base + 1],
                q_rope_regs1[base + 2],
                q_rope_regs1[base + 3],
                b0,
                b1,
            )
    return qk0, qk1


@cute.jit
def s6b_xv_rope_global_dsv4(
    acc_rope,
    sm_p_full_addr: Int32,
    kv_cache_u8: cute.Tensor,
    token_idx_view: cute.Tensor,
    warp_id: Int32,
    lane: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
    *,
    bi: cutlass.Constexpr,
    d_rope: cutlass.Constexpr,
    n_warps: cutlass.Constexpr,
):
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    rope_dim_base = warp_id * Int32(d_rope // n_warps)
    dim_n = rope_dim_base + gid

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)

    for ks in cutlass.range_constexpr(bi // 16):
        k_base = Int32(ks) * Int32(16)
        a_byte = (a_row * Int32(bi) + (k_base + a_col)) * Int32(2)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(sm_p_full_addr + a_byte)

        ent0 = k_base + tid * Int32(2)
        idx0 = Int32(token_idx_view[ent0])
        idx1 = Int32(token_idx_view[ent0 + Int32(1)])
        idx8 = Int32(token_idx_view[ent0 + Int32(8)])
        idx9 = Int32(token_idx_view[ent0 + Int32(9)])
        v0 = _ld_global_dsv4_rope_b16(
            kv_cache_u8, idx0, dim_n, page_block_size, stride_kv_block
        )
        v1 = _ld_global_dsv4_rope_b16(
            kv_cache_u8, idx1, dim_n, page_block_size, stride_kv_block
        )
        v8 = _ld_global_dsv4_rope_b16(
            kv_cache_u8, idx8, dim_n, page_block_size, stride_kv_block
        )
        v9 = _ld_global_dsv4_rope_b16(
            kv_cache_u8, idx9, dim_n, page_block_size, stride_kv_block
        )
        b0 = v0 | (v1 << Uint32(16))
        b1 = v8 | (v9 << Uint32(16))
        acc_rope[0], acc_rope[1], acc_rope[2], acc_rope[3] = mma_m16n8k16_f32_bf16(
            acc_rope[0], acc_rope[1], acc_rope[2], acc_rope[3], a0, a1, a2, a3, b0, b1
        )
    return acc_rope


@cute.jit
def s6b_xv_rope_global_mg_dsv4(
    acc_rope0,
    acc_rope1,
    sm_p_g0_addr: Int32,
    sm_p_g1_addr: Int32,
    kv_cache_u8: cute.Tensor,
    token_idx_view: cute.Tensor,
    warp_id: Int32,
    lane: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
    *,
    bi: cutlass.Constexpr,
    d_rope: cutlass.Constexpr,
    n_warps: cutlass.Constexpr,
    n_hg: cutlass.Constexpr = 2,
):
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    rope_dim_base = warp_id * Int32(d_rope // n_warps)
    dim_n = rope_dim_base + gid

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)

    for ks in cutlass.range_constexpr(bi // 16):
        k_base = Int32(ks) * Int32(16)
        ent0 = k_base + tid * Int32(2)
        idx0 = Int32(token_idx_view[ent0])
        idx1 = Int32(token_idx_view[ent0 + Int32(1)])
        idx8 = Int32(token_idx_view[ent0 + Int32(8)])
        idx9 = Int32(token_idx_view[ent0 + Int32(9)])
        v0 = _ld_global_dsv4_rope_b16(
            kv_cache_u8, idx0, dim_n, page_block_size, stride_kv_block
        )
        v1 = _ld_global_dsv4_rope_b16(
            kv_cache_u8, idx1, dim_n, page_block_size, stride_kv_block
        )
        v8 = _ld_global_dsv4_rope_b16(
            kv_cache_u8, idx8, dim_n, page_block_size, stride_kv_block
        )
        v9 = _ld_global_dsv4_rope_b16(
            kv_cache_u8, idx9, dim_n, page_block_size, stride_kv_block
        )
        b0 = v0 | (v1 << Uint32(16))
        b1 = v8 | (v9 << Uint32(16))

        a_byte = (a_row * Int32(bi) + (k_base + a_col)) * Int32(2)
        a00, a01, a02, a03 = ldmatrix_m8n8x4_b16(sm_p_g0_addr + a_byte)
        acc_rope0[0], acc_rope0[1], acc_rope0[2], acc_rope0[3] = (
            mma_m16n8k16_f32_bf16(
                acc_rope0[0],
                acc_rope0[1],
                acc_rope0[2],
                acc_rope0[3],
                a00,
                a01,
                a02,
                a03,
                b0,
                b1,
            )
        )
        if cutlass.const_expr(n_hg == 2):
            a10, a11, a12, a13 = ldmatrix_m8n8x4_b16(sm_p_g1_addr + a_byte)
            acc_rope1[0], acc_rope1[1], acc_rope1[2], acc_rope1[3] = (
                mma_m16n8k16_f32_bf16(
                    acc_rope1[0],
                    acc_rope1[1],
                    acc_rope1[2],
                    acc_rope1[3],
                    a10,
                    a11,
                    a12,
                    a13,
                    b0,
                    b1,
                )
            )
    return acc_rope0, acc_rope1


@cute.jit
def s4_online_softmax_mg2(
    qk0,
    qk1,
    p0,
    p1,
    acc0,
    acc1,
    rope0,
    rope1,
    gm0,
    gm1,
    gs0,
    gs1,
    reduce0_max_addr: Int32,
    reduce0_sum_addr: Int32,
    reduce1_max_addr: Int32,
    reduce1_sum_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    tid_flat: Int32,
    *,
    n_v_chunks: cutlass.Constexpr,
    hpb: cutlass.Constexpr,
    n_warps: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    n_acc_tiles: cutlass.Constexpr,
    n_hg: cutlass.Constexpr = 2,
    valid_hpb: cutlass.Constexpr = 16,
):
    """S4 online softmax with FlashInfer-style deferred row sums. Every group-1
    statement is gated inline behind ``const_expr(n_hg == 2)`` so the n_hg==2
    trace is the original interleaved instruction stream (byte-identical) and
    n_hg==1 cleanly elides all qk1/acc1/rope1/gs1/reduce1 work (single HPB head
    group; the cross-warp reduce spans tid_flat < hpb, group 0 only)."""
    two = cutlass.const_expr(n_hg == 2)
    hi0 = cutlass.const_expr(valid_hpb > 8)
    reduce_heads = cutlass.const_expr(n_hg * hpb if n_hg == 2 else valid_hpb)
    bar_kw = dict(barrier_id=barrier_id, number_of_threads=num_threads)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)

    lm00 = fmax_f32(qk0[0], qk0[1])
    if cutlass.const_expr(hi0):
        lm01 = fmax_f32(qk0[2], qk0[3])
    if cutlass.const_expr(two):
        lm10 = fmax_f32(qk1[0], qk1[1])
        lm11 = fmax_f32(qk1[2], qk1[3])
    lm00 = fmax_f32(lm00, cute.arch.shuffle_sync_bfly(lm00, offset=2))
    if cutlass.const_expr(hi0):
        lm01 = fmax_f32(lm01, cute.arch.shuffle_sync_bfly(lm01, offset=2))
    if cutlass.const_expr(two):
        lm10 = fmax_f32(lm10, cute.arch.shuffle_sync_bfly(lm10, offset=2))
        lm11 = fmax_f32(lm11, cute.arch.shuffle_sync_bfly(lm11, offset=2))
    lm00 = fmax_f32(lm00, cute.arch.shuffle_sync_bfly(lm00, offset=1))
    if cutlass.const_expr(hi0):
        lm01 = fmax_f32(lm01, cute.arch.shuffle_sync_bfly(lm01, offset=1))
    if cutlass.const_expr(two):
        lm10 = fmax_f32(lm10, cute.arch.shuffle_sync_bfly(lm10, offset=1))
        lm11 = fmax_f32(lm11, cute.arch.shuffle_sync_bfly(lm11, offset=1))

    if tid == Int32(0):
        st_shared_f32(reduce0_max_addr + (warp_id * Int32(hpb) + gid) * Int32(4), lm00)
        if cutlass.const_expr(hi0):
            st_shared_f32(
                reduce0_max_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
                lm01,
            )
        if cutlass.const_expr(two):
            st_shared_f32(reduce1_max_addr + (warp_id * Int32(hpb) + gid) * Int32(4), lm10)
            st_shared_f32(
                reduce1_max_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
                lm11,
            )
    cute.arch.barrier(**bar_kw)

    if tid_flat < Int32(reduce_heads):
        if cutlass.const_expr(two):
            group = tid_flat // Int32(hpb)
            h = tid_flat - group * Int32(hpb)
        else:
            group = Int32(0)
            h = tid_flat
        rmax = reduce0_max_addr
        if cutlass.const_expr(two):
            if group != Int32(0):
                rmax = reduce1_max_addr
        bmax = Float32(-1e30)
        for w in cutlass.range_constexpr(n_warps):
            wm = ld_shared_f32(rmax + (Int32(w) * Int32(hpb) + h) * Int32(4))
            bmax = fmax_f32(bmax, wm)
        st_shared_f32(rmax + h * Int32(4), bmax)
    cute.arch.barrier(**bar_kw)

    blm00 = ld_shared_f32(reduce0_max_addr + gid * Int32(4))
    if cutlass.const_expr(hi0):
        blm01 = ld_shared_f32(reduce0_max_addr + (gid + Int32(8)) * Int32(4))
    if cutlass.const_expr(two):
        blm10 = ld_shared_f32(reduce1_max_addr + gid * Int32(4))
        blm11 = ld_shared_f32(reduce1_max_addr + (gid + Int32(8)) * Int32(4))

    ngm00 = fmax_f32(gm0[0], blm00)
    if cutlass.const_expr(hi0):
        ngm01 = fmax_f32(gm0[1], blm01)
    if cutlass.const_expr(two):
        ngm10 = fmax_f32(gm1[0], blm10)
        ngm11 = fmax_f32(gm1[1], blm11)
    alpha00 = _exp2_approx_ftz_f32(gm0[0] - ngm00)
    if cutlass.const_expr(hi0):
        alpha01 = _exp2_approx_ftz_f32(gm0[1] - ngm01)
    if cutlass.const_expr(two):
        alpha10 = _exp2_approx_ftz_f32(gm1[0] - ngm10)
        alpha11 = _exp2_approx_ftz_f32(gm1[1] - ngm11)

    for vc in cutlass.range_constexpr(n_acc_tiles):
        acc0[vc][0] = acc0[vc][0] * alpha00
        acc0[vc][1] = acc0[vc][1] * alpha00
        if cutlass.const_expr(hi0):
            acc0[vc][2] = acc0[vc][2] * alpha01
            acc0[vc][3] = acc0[vc][3] * alpha01
        if cutlass.const_expr(two):
            acc1[vc][0] = acc1[vc][0] * alpha10
            acc1[vc][1] = acc1[vc][1] * alpha10
            acc1[vc][2] = acc1[vc][2] * alpha11
            acc1[vc][3] = acc1[vc][3] * alpha11
    rope0[0] = rope0[0] * alpha00
    rope0[1] = rope0[1] * alpha00
    if cutlass.const_expr(hi0):
        rope0[2] = rope0[2] * alpha01
        rope0[3] = rope0[3] * alpha01
    if cutlass.const_expr(two):
        rope1[0] = rope1[0] * alpha10
        rope1[1] = rope1[1] * alpha10
        rope1[2] = rope1[2] * alpha11
        rope1[3] = rope1[3] * alpha11

    gs0[0] = gs0[0] * alpha00
    if cutlass.const_expr(hi0):
        gs0[1] = gs0[1] * alpha01
    if cutlass.const_expr(two):
        gs1[0] = gs1[0] * alpha10
        gs1[1] = gs1[1] * alpha11

    p0[0] = _exp2_approx_ftz_f32(qk0[0] - ngm00)
    p0[1] = _exp2_approx_ftz_f32(qk0[1] - ngm00)
    if cutlass.const_expr(hi0):
        p0[2] = _exp2_approx_ftz_f32(qk0[2] - ngm01)
        p0[3] = _exp2_approx_ftz_f32(qk0[3] - ngm01)
    else:
        p0[2] = Float32(0.0)
        p0[3] = Float32(0.0)
    if cutlass.const_expr(two):
        p1[0] = _exp2_approx_ftz_f32(qk1[0] - ngm10)
        p1[1] = _exp2_approx_ftz_f32(qk1[1] - ngm10)
        p1[2] = _exp2_approx_ftz_f32(qk1[2] - ngm11)
        p1[3] = _exp2_approx_ftz_f32(qk1[3] - ngm11)

    ls00 = p0[0] + p0[1]
    if cutlass.const_expr(hi0):
        ls01 = p0[2] + p0[3]
    if cutlass.const_expr(two):
        ls10 = p1[0] + p1[1]
        ls11 = p1[2] + p1[3]
    ls00 = ls00 + cute.arch.shuffle_sync_bfly(ls00, offset=2)
    if cutlass.const_expr(hi0):
        ls01 = ls01 + cute.arch.shuffle_sync_bfly(ls01, offset=2)
    if cutlass.const_expr(two):
        ls10 = ls10 + cute.arch.shuffle_sync_bfly(ls10, offset=2)
        ls11 = ls11 + cute.arch.shuffle_sync_bfly(ls11, offset=2)
    ls00 = ls00 + cute.arch.shuffle_sync_bfly(ls00, offset=1)
    if cutlass.const_expr(hi0):
        ls01 = ls01 + cute.arch.shuffle_sync_bfly(ls01, offset=1)
    if cutlass.const_expr(two):
        ls10 = ls10 + cute.arch.shuffle_sync_bfly(ls10, offset=1)
        ls11 = ls11 + cute.arch.shuffle_sync_bfly(ls11, offset=1)

    gs0[0] = gs0[0] + ls00
    if cutlass.const_expr(hi0):
        gs0[1] = gs0[1] + ls01
    if cutlass.const_expr(two):
        gs1[0] = gs1[0] + ls10
        gs1[1] = gs1[1] + ls11
    gm0[0] = ngm00
    if cutlass.const_expr(hi0):
        gm0[1] = ngm01
    if cutlass.const_expr(two):
        gm1[0] = ngm10
        gm1[1] = ngm11
    return (
        p0,
        p1,
        Float32(1.0),
        Float32(1.0),
        Float32(1.0),
        Float32(1.0),
    )


@cute.jit
def s4_finalize_row_sum_mg2(
    gs0,
    gs1,
    reduce0_sum_addr: Int32,
    reduce1_sum_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    tid_flat: Int32,
    *,
    hpb: cutlass.Constexpr,
    n_warps: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    n_hg: cutlass.Constexpr = 2,
    valid_hpb: cutlass.Constexpr = 16,
):
    """Final cross-warp row-sum reduction for deferred MG softmax. Group-1
    statements gate inline behind ``const_expr(n_hg == 2)`` (n_hg==2 trace
    unchanged); n_hg==1 reduces only group 0 (tid_flat < hpb)."""
    two = cutlass.const_expr(n_hg == 2)
    hi0 = cutlass.const_expr(valid_hpb > 8)
    reduce_heads = cutlass.const_expr(n_hg * hpb if n_hg == 2 else valid_hpb)
    bar_kw = dict(barrier_id=barrier_id, number_of_threads=num_threads)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)

    if tid == Int32(0):
        st_shared_f32(reduce0_sum_addr + (warp_id * Int32(hpb) + gid) * Int32(4), gs0[0])
        if cutlass.const_expr(hi0):
            st_shared_f32(
                reduce0_sum_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
                gs0[1],
            )
        if cutlass.const_expr(two):
            st_shared_f32(reduce1_sum_addr + (warp_id * Int32(hpb) + gid) * Int32(4), gs1[0])
            st_shared_f32(
                reduce1_sum_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
                gs1[1],
            )
    cute.arch.barrier(**bar_kw)

    if tid_flat < Int32(reduce_heads):
        if cutlass.const_expr(two):
            group = tid_flat // Int32(hpb)
            h = tid_flat - group * Int32(hpb)
        else:
            group = Int32(0)
            h = tid_flat
        rsum = reduce0_sum_addr
        if cutlass.const_expr(two):
            if group != Int32(0):
                rsum = reduce1_sum_addr
        total = Float32(0.0)
        for w in cutlass.range_constexpr(n_warps):
            total = total + ld_shared_f32(
                rsum + (Int32(w) * Int32(hpb) + h) * Int32(4)
            )
        st_shared_f32(rsum + h * Int32(4), total)
    cute.arch.barrier(**bar_kw)

    gs0[0] = ld_shared_f32(reduce0_sum_addr + gid * Int32(4))
    if cutlass.const_expr(hi0):
        gs0[1] = ld_shared_f32(reduce0_sum_addr + (gid + Int32(8)) * Int32(4))
    if cutlass.const_expr(two):
        gs1[0] = ld_shared_f32(reduce1_sum_addr + gid * Int32(4))
        gs1[1] = ld_shared_f32(reduce1_sum_addr + (gid + Int32(8)) * Int32(4))
    return gs0, gs1


@cute.jit
def s6_xv_nope_mg_dsv4(
    w_pre0,
    w_pre1,
    acc0,
    acc1,
    kv_fp8_base_addr: Int32,
    kv_sc_base_addr: Int32,
    w_head_sc_view: cute.Tensor,
    w_fp8_base_addr: Int32,
    sm_p_g0_addr: Int32,
    sm_p_g1_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    tid_flat: Int32,
    *,
    n_v_chunks: cutlass.Constexpr,
    v_chunk: cutlass.Constexpr,
    hpb: cutlass.Constexpr,
    bi: cutlass.Constexpr,
    kv_smem_stride: cutlass.Constexpr,
    w_fp8_stride: cutlass.Constexpr,
    w_fp8_group_stride: cutlass.Constexpr,
    w_fp8_parity_stride: cutlass.Constexpr,
    n_warps: cutlass.Constexpr,
    scale_bytes_per_token: cutlass.Constexpr,
    nt_per_warp_xv: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    n_hg: cutlass.Constexpr = 2,
    row_xor: cutlass.Constexpr = False,
):
    """FlashInfer-shaped DSV4 MG XV-NoPE for ``n_hg`` HPB head groups.

    V scales and V B operands are shared across the head groups. This fuses the
    decode-style S5/S6 calls used by the first correctness port while keeping the
    same DSV4 FP8 W-quantization math. Every group-1 statement gates inline behind
    ``const_expr(n_hg == 2)`` (n_hg==2 trace byte-identical); n_hg==1 elides the
    group-1 sm_p / W-quant stores + the second per-k XV MMA, so the per-CTA XV
    ldmatrix/mma count halves. The w_head_sc zero/scale loops span
    ``n_hg * n_v_chunks * hpb`` (one group's worth for n_hg==1).
    """
    two = cutlass.const_expr(n_hg == 2)
    bar_kw = dict(barrier_id=barrier_id, number_of_threads=num_threads)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    warp_first_cand = warp_id * Int32(8)
    cand_e0 = warp_first_cand + tid * Int32(2)
    cand_e1 = cand_e0 + Int32(1)
    group_sc_elems = Int32(n_v_chunks * hpb)

    st_shared_bf16_from_f32(
        sm_p_g0_addr + (gid * Int32(bi) + cand_e0) * Int32(2), w_pre0[0]
    )
    st_shared_bf16_from_f32(
        sm_p_g0_addr + (gid * Int32(bi) + cand_e1) * Int32(2), w_pre0[1]
    )
    st_shared_bf16_from_f32(
        sm_p_g0_addr + ((gid + Int32(8)) * Int32(bi) + cand_e0) * Int32(2),
        w_pre0[2],
    )
    st_shared_bf16_from_f32(
        sm_p_g0_addr + ((gid + Int32(8)) * Int32(bi) + cand_e1) * Int32(2),
        w_pre0[3],
    )
    if cutlass.const_expr(two):
        st_shared_bf16_from_f32(
            sm_p_g1_addr + (gid * Int32(bi) + cand_e0) * Int32(2), w_pre1[0]
        )
        st_shared_bf16_from_f32(
            sm_p_g1_addr + (gid * Int32(bi) + cand_e1) * Int32(2), w_pre1[1]
        )
        st_shared_bf16_from_f32(
            sm_p_g1_addr + ((gid + Int32(8)) * Int32(bi) + cand_e0) * Int32(2),
            w_pre1[2],
        )
        st_shared_bf16_from_f32(
            sm_p_g1_addr + ((gid + Int32(8)) * Int32(bi) + cand_e1) * Int32(2),
            w_pre1[3],
        )

    i = tid_flat
    while i < Int32(n_hg * n_v_chunks * hpb):
        w_head_sc_view[i] = Float32(0.0)
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    w_head_sc_base = shared_ptr_to_u32(w_head_sc_view.iterator)
    for vc in cutlass.range_constexpr(n_v_chunks):
        vsc0 = _ue8m0_byte_to_fp32(
            _ld_u8_zext(kv_sc_base_addr, cand_e0 * Int32(scale_bytes_per_token) + Int32(vc))
        )
        vsc1 = _ue8m0_byte_to_fp32(
            _ld_u8_zext(kv_sc_base_addr, cand_e1 * Int32(scale_bytes_per_token) + Int32(vc))
        )
        m00 = fmax_f32(fabs_f32(w_pre0[0] * vsc0), fabs_f32(w_pre0[1] * vsc1))
        m01 = fmax_f32(fabs_f32(w_pre0[2] * vsc0), fabs_f32(w_pre0[3] * vsc1))
        if cutlass.const_expr(two):
            m10 = fmax_f32(fabs_f32(w_pre1[0] * vsc0), fabs_f32(w_pre1[1] * vsc1))
            m11 = fmax_f32(fabs_f32(w_pre1[2] * vsc0), fabs_f32(w_pre1[3] * vsc1))
        vc_base = Int32(vc) * Int32(hpb)
        atomic_max_shared_f32(w_head_sc_base + (vc_base + gid) * Int32(4), m00)
        atomic_max_shared_f32(
            w_head_sc_base + (vc_base + gid + Int32(8)) * Int32(4), m01
        )
        if cutlass.const_expr(two):
            atomic_max_shared_f32(
                w_head_sc_base + (group_sc_elems + vc_base + gid) * Int32(4), m10
            )
            atomic_max_shared_f32(
                w_head_sc_base + (group_sc_elems + vc_base + gid + Int32(8)) * Int32(4),
                m11,
            )
    cute.arch.barrier(**bar_kw)

    i = tid_flat
    while i < Int32(n_hg * n_v_chunks * hpb):
        w_head_sc_view[i] = fmax_f32(w_head_sc_view[i], Float32(1e-10)) * Float32(
            1.0 / 448.0
        )
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    # ldmatrix A-operand load row. FI applies the SAME row^(row>>3) swizzle on the
    # load (ldmatrix_load_A_fp8_layout<ROW_XOR>, ldmatrix_sm120.cuh:79-86) so it is
    # symmetric with the store -> numerically identity. const_expr(row_xor=False)
    # leaves a_row_eff == a_row (load address arithmetic textually unchanged).
    if cutlass.const_expr(row_xor):
        a_row_eff = _wfp8_row_xor(a_row)
    else:
        a_row_eff = a_row
    a_col = (lane >> Int32(4)) * Int32(16)
    for vc in cutlass.range_constexpr(n_v_chunks):
        vc_base = Int32(vc) * Int32(hpb)
        w_fp8_parity_addr = (
            w_fp8_base_addr + Int32(vc & 1) * Int32(w_fp8_parity_stride)
        )
        w_fp8_g0 = w_fp8_parity_addr
        w_fp8_g1 = w_fp8_parity_addr + Int32(w_fp8_group_stride)

        sc00 = w_head_sc_view[vc_base + gid]
        sc01 = w_head_sc_view[vc_base + gid + Int32(8)]
        if cutlass.const_expr(two):
            sc10 = w_head_sc_view[group_sc_elems + vc_base + gid]
            sc11 = w_head_sc_view[group_sc_elems + vc_base + gid + Int32(8)]
        si00 = Float32(1.0) / sc00
        si01 = Float32(1.0) / sc01
        if cutlass.const_expr(two):
            si10 = Float32(1.0) / sc10
            si11 = Float32(1.0) / sc11

        vsc0 = _ue8m0_byte_to_fp32(
            _ld_u8_zext(kv_sc_base_addr, cand_e0 * Int32(scale_bytes_per_token) + Int32(vc))
        )
        vsc1 = _ue8m0_byte_to_fp32(
            _ld_u8_zext(kv_sc_base_addr, cand_e1 * Int32(scale_bytes_per_token) + Int32(vc))
        )

        # W (A-operand) store rows. FI stores at wrow0=gid, wrow1=gid+8 and, for
        # the dual pbs_extra==2 path, applies the bank-conflict swizzle row^(row>>3)
        # (prefill_kernel.cuh:1258-1262). const_expr(row_xor=False) -> r0/r8 are the
        # literal gid / gid+8, so the no-rowxor store address arithmetic is
        # textually identical to today (byte-identical PTX). Columns (cand_e0/e1)
        # are NEVER swizzled.
        if cutlass.const_expr(row_xor):
            r0 = _wfp8_row_xor(gid)
            r8 = _wfp8_row_xor(gid + Int32(8))
        else:
            r0 = gid
            r8 = gid + Int32(8)

        st_shared_u8(
            w_fp8_g0 + r0 * Int32(w_fp8_stride) + cand_e0,
            _quant_e4m3_byte(w_pre0[0] * vsc0 * si00).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g0 + r0 * Int32(w_fp8_stride) + cand_e1,
            _quant_e4m3_byte(w_pre0[1] * vsc1 * si00).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g0 + r8 * Int32(w_fp8_stride) + cand_e0,
            _quant_e4m3_byte(w_pre0[2] * vsc0 * si01).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g0 + r8 * Int32(w_fp8_stride) + cand_e1,
            _quant_e4m3_byte(w_pre0[3] * vsc1 * si01).to(cutlass.Uint8),
        )
        if cutlass.const_expr(two):
            st_shared_u8(
                w_fp8_g1 + r0 * Int32(w_fp8_stride) + cand_e0,
                _quant_e4m3_byte(w_pre1[0] * vsc0 * si10).to(cutlass.Uint8),
            )
            st_shared_u8(
                w_fp8_g1 + r0 * Int32(w_fp8_stride) + cand_e1,
                _quant_e4m3_byte(w_pre1[1] * vsc1 * si10).to(cutlass.Uint8),
            )
            st_shared_u8(
                w_fp8_g1 + r8 * Int32(w_fp8_stride) + cand_e0,
                _quant_e4m3_byte(w_pre1[2] * vsc0 * si11).to(cutlass.Uint8),
            )
            st_shared_u8(
                w_fp8_g1 + r8 * Int32(w_fp8_stride) + cand_e1,
                _quant_e4m3_byte(w_pre1[3] * vsc1 * si11).to(cutlass.Uint8),
            )
        cute.arch.barrier(**bar_kw)

        for nt in cutlass.range_constexpr(nt_per_warp_xv):
            at = vc * nt_per_warp_xv + nt
            dim = (
                Int32(vc) * Int32(v_chunk)
                + (Int32(nt) * Int32(n_warps) + warp_id) * Int32(8)
            )
            xv00 = Float32(0.0)
            xv01 = Float32(0.0)
            xv02 = Float32(0.0)
            xv03 = Float32(0.0)
            if cutlass.const_expr(two):
                xv10 = Float32(0.0)
                xv11 = Float32(0.0)
                xv12 = Float32(0.0)
                xv13 = Float32(0.0)
            for kstep in cutlass.range_constexpr(bi // 32):
                ko = Int32(kstep) * Int32(32)
                b0, b1 = _d2_load_b_fp8(
                    kv_fp8_base_addr, ko, dim, lane, kv_smem_stride=kv_smem_stride
                )
                a_addr0 = w_fp8_g0 + a_row_eff * Int32(w_fp8_stride) + ko + a_col
                a00, a01, a02, a03 = ldmatrix_m8n8x4_b16(a_addr0)
                xv00, xv01, xv02, xv03 = mma_m16n8k32_f32_e4m3(
                    xv00, xv01, xv02, xv03, a00, a01, a02, a03, b0, b1
                )
                if cutlass.const_expr(two):
                    a_addr1 = w_fp8_g1 + a_row_eff * Int32(w_fp8_stride) + ko + a_col
                    a10, a11, a12, a13 = ldmatrix_m8n8x4_b16(a_addr1)
                    xv10, xv11, xv12, xv13 = mma_m16n8k32_f32_e4m3(
                        xv10, xv11, xv12, xv13, a10, a11, a12, a13, b0, b1
                    )

            acc0[at][0] = acc0[at][0] + xv00 * sc00
            acc0[at][1] = acc0[at][1] + xv01 * sc00
            acc0[at][2] = acc0[at][2] + xv02 * sc01
            acc0[at][3] = acc0[at][3] + xv03 * sc01
            if cutlass.const_expr(two):
                acc1[at][0] = acc1[at][0] + xv10 * sc10
                acc1[at][1] = acc1[at][1] + xv11 * sc10
                acc1[at][2] = acc1[at][2] + xv12 * sc11
                acc1[at][3] = acc1[at][3] + xv13 * sc11
    return acc0, acc1


class UnifiedPrefillMGKernel:
    def __init__(
        self,
        traits,
        layout,
        page_block_size,
        num_tiles,
        replicate_h,
        num_heads,
        q_stride,
        indices_stride0,
        output_stride,
        out_lse_stride,
        has_sink,
        topk,
        has_extra=False,
        pbs_extra=1,
        num_main_tiles=0,
        extra_topk=0,
        extra_indices_stride0=0,
        row_xor=False,
        head_offset=0,
        valid_hpb=None,
        pack_hilo_rows=False,
    ):
        self.traits = traits
        self.layout = layout
        self.page_block_size = int(page_block_size)
        self.num_tiles = int(num_tiles)
        self.replicate_h = int(replicate_h)
        self.num_heads = int(num_heads)
        self.q_stride_row = int(q_stride[0])
        self.q_stride_head = int(q_stride[1])
        self.q_stride_dim = int(q_stride[2])
        self.indices_stride_row = int(indices_stride0)
        self.output_stride_row = int(output_stride[0])
        self.output_stride_head = int(output_stride[1])
        self.output_stride_dim = int(output_stride[2])
        self.out_lse_stride_row = int(out_lse_stride[0])
        self.out_lse_stride_head = int(out_lse_stride[1])
        self.has_sink = bool(has_sink)
        self.topk = int(topk)
        # DSV4 dual-cache (has_extra) prefill onto MG. All default to current
        # behavior so the single-cache __call__/kernel trace is byte-identical:
        # has_extra=False const_expr-elides every extra-section arm in _body,
        # row_xor=False keeps the s6_xv_nope_mg_dsv4 address arithmetic textually
        # unchanged. The union spans num_main_tiles main chunks (gathered from the
        # main cache) then ceil(extra_section_len/64) extra chunks (gathered from
        # the extra cache, page_block_size=pbs_extra).
        self.has_extra = bool(has_extra)
        self.pbs_extra = int(pbs_extra)
        self.num_main_tiles = int(num_main_tiles)
        self.extra_topk = int(extra_topk)
        self.extra_indices_stride_row = int(extra_indices_stride0)
        self.row_xor = bool(row_xor)
        self.head_offset = int(head_offset)
        self.valid_hpb = int(traits.hpb if valid_hpb is None else valid_hpb)
        self.pack_hilo_rows = bool(pack_hilo_rows)
        # MG head-group count (1 for heads==16, 2 for heads % 32 == 0). Derived
        # from the layout (heads_per_cta // hpb) and threaded as a compile-time
        # const so every group-1 op const_expr-elides for n_hg==1.
        self.mg_n_hg = int(layout.heads_per_cta // traits.hpb)
        self.math_threads = int(traits.math_threads)
        self.block_threads = _PREFILL_BLOCK_THREADS

    @cute.jit
    def __call__(
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
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_all,
            kv_cache_u8,
            indices,
            topk_length,
            attn_sink,
            output,
            out_lse,
            sm_scale_log2,
            stride_kv_block,
        ).launch(
            grid=(num_tokens * Int32(self.replicate_h), 1, 1),
            block=[self.block_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def call_dual(
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
        # DUAL-CACHE entry (DSV4 dual-cache prefill onto MG): the dispatcher selects
        # this (entry=kernel.call_dual) only when has_extra=True, so its DISTINCT
        # mangled name never collides with the byte-identical single-cache __call__.
        # Launches the 13-param @cute.kernel (self.kernel_dual), which shares the
        # body via _body(has_extra=True, row_xor=self.row_xor). The extra device
        # args are appended AFTER stride_kv_block in the order:
        # extra_kv_cache_u8, extra_indices, extra_topk_length, stride_extra_kv_block.
        self.kernel_dual(
            q_all,
            kv_cache_u8,
            indices,
            topk_length,
            attn_sink,
            output,
            out_lse,
            sm_scale_log2,
            stride_kv_block,
            extra_kv_cache_u8,
            extra_indices,
            extra_topk_length,
            stride_extra_kv_block,
        ).launch(
            grid=(num_tokens * Int32(self.replicate_h), 1, 1),
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
        # DSV4-BF16 / GLM MG prefill path. Threads dummy extra args (aliased to the
        # main tensors) into the shared body with has_extra=False so the
        # extra-section code is fully const_expr-elided.
        self._body(
            q_all,
            kv_cache_u8,
            indices,
            topk_length,
            attn_sink,
            output,
            out_lse,
            sm_scale_log2,
            stride_kv_block,
            kv_cache_u8,
            indices,
            topk_length,
            stride_kv_block,
            has_extra=False,
            row_xor=False,
        )

    @cute.kernel
    def kernel_dual(
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
        # args into the shared body (has_extra=True, row_xor=self.row_xor).
        self._body(
            q_all,
            kv_cache_u8,
            indices,
            topk_length,
            attn_sink,
            output,
            out_lse,
            sm_scale_log2,
            stride_kv_block,
            extra_kv_cache_u8,
            extra_indices,
            extra_topk_length,
            stride_extra_kv_block,
            has_extra=True,
            row_xor=cutlass.const_expr(self.row_xor),
        )

    @cute.jit
    def _body(
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
        has_extra: cutlass.Constexpr = False,
        row_xor: cutlass.Constexpr = False,
    ):
        t = self.traits
        L = self.layout
        tid = Int32(cute.arch.thread_idx()[0])
        lane = cute.arch.lane_idx()
        warp_id = tid >> Int32(5)
        block_x, _, _ = cute.arch.block_idx()
        block_x = Int32(block_x)
        rep_h = Int32(self.replicate_h)
        token_idx = block_x // rep_h
        h_tile = block_x - token_idx * rep_h
        head_base = Int32(self.head_offset) + h_tile * Int32(L.heads_per_cta)

        bf16_qk = cutlass.const_expr(t.compute_mode == ComputeMode.BF16)
        # GLM (ARBITRARY_FP32) const_expr arm: post-MMA fp32 QK scale, GLM 2-pass
        # W XV, V==nope (no XV-rope), 656/528 KV geometry, KV-rope from global/L2,
        # Q-rope registerized (aliased onto W_FP8). DSV4 is the scale_format==0 /
        # has_extra arm and is byte-identical.
        is_glm = cutlass.const_expr(t.model_type == ModelType.GLM_NSA)
        q_head_dim = cutlass.const_expr(_GLM_HEAD_DIM if is_glm else _DSV4_HEAD_DIM)
        # Compile-time head-group count: 1 (heads==16) or 2 (heads % 32 == 0). All
        # group-1 work below const_expr-elides when n_hg==1 (single-group MG).
        n_hg = cutlass.const_expr(self.mg_n_hg)

        smem = cutlass_utils.SmemAllocator()
        SharedStorage = get_prefill_mg_shared_storage_cls(t, n_hg)
        st = smem.allocate(SharedStorage)

        kv_fp8_addr = shared_ptr_to_u32(st.kv_fp8.data_ptr())
        reduce_addr = shared_ptr_to_u32(st.reduce.data_ptr())
        w_fp8_addr = shared_ptr_to_u32(st.w_fp8.data_ptr())
        # GLM reads KV-rope from global/L2 (no sm_p_full XV-rope), so sm_p_full is
        # DSV4-only; GLM has no kv_sc footer (inline scales). Both have a 0 addr.
        if cutlass.const_expr(is_glm):
            kv_sc_addr = Int32(0)
            sm_p_full_addr = Int32(0)
        else:
            kv_sc_addr = shared_ptr_to_u32(st.kv_sc.data_ptr())
            sm_p_full_addr = shared_ptr_to_u32(st.sm_p_full.data_ptr())

        if cutlass.const_expr(bf16_qk):
            # BF16: Q-NoPE is a single bf16 buffer; the FP8 q_fp8/q_sc do not
            # exist. Q-rope scratch ALIASES the W_FP8 region (S0-only).
            q_nope_bf16_addr = shared_ptr_to_u32(st.q_nope_bf16.data_ptr())
            q_fp8_addr = Int32(0)
            q_rope_addr = w_fp8_addr
            q_sc_view_all = None
        else:
            q_nope_bf16_addr = Int32(0)
            q_fp8_addr = shared_ptr_to_u32(st.q_fp8.data_ptr())
            # GLM FP8 registerizes Q-rope: its scratch ALIASES W_FP8 (S0-only),
            # exactly like the BF16 path. DSV4 FP8 keeps a dedicated q_rope field.
            if cutlass.const_expr(is_glm):
                q_rope_addr = w_fp8_addr
            else:
                q_rope_addr = shared_ptr_to_u32(st.q_rope.data_ptr())
            q_sc_view_all = st.q_sc.get_tensor(cute.make_layout(int(L.q_sc_bytes // 4)))

        amax_view = st.reduce.get_tensor(cute.make_layout(int(L.reduce_group_bytes // 4)))
        token_idx_view_all = st.token_idx.get_tensor(
            cute.make_layout(int(L.token_idx_buf_bytes * L.token_idx_bufs // 4))
        )
        w_head_sc_view_all = st.w_head_sc.get_tensor(
            cute.make_layout(int(L.w_head_sc_bytes // 4))
        )

        is_io = warp_id >= Int32(self.math_threads // 32)

        kv_fp8_buf = Int32(L.kv_fp8_buf_bytes)
        kv_sc_buf = Int32(L.kv_sc_buf_bytes)
        tok_buf_elems = Int32(L.token_idx_buf_bytes // 4)

        q_fp8_g0 = q_fp8_addr
        q_fp8_g1 = q_fp8_addr + Int32(L.q_fp8_group_bytes)
        q_rope_g0 = q_rope_addr
        q_rope_g1 = q_rope_addr + Int32(L.q_rope_group_bytes)
        q_nope_bf16_g0 = q_nope_bf16_addr
        q_nope_bf16_g1 = q_nope_bf16_addr + Int32(L.q_nope_bf16_group_bytes)
        reduce_g0 = reduce_addr
        reduce_g1 = reduce_addr + Int32(L.reduce_group_bytes)
        sm_p_g0 = sm_p_full_addr
        sm_p_g1 = sm_p_full_addr + Int32(L.sm_p_full_group_bytes)
        # GLM calls the per-group decode s6_xv_nope, which expects a CONTIGUOUS
        # per-group W_FP8 double-buffer [parity0(hpb)][parity1(hpb)] addressed by
        # ``(vc&1)*hpb*w_fp8_stride``. So each GLM group gets its own
        # 2*w_fp8_group_bytes slab (the total is identical to the DSV4 MG
        # [par0:g0,g1][par1:g0,g1] interleave; only the per-group base differs).
        glm_w_fp8_g0 = w_fp8_addr
        glm_w_fp8_g1 = w_fp8_addr + Int32(2 * L.w_fp8_group_bytes)

        if cutlass.const_expr(not bf16_qk):
            q_sc_g0 = cute.make_tensor(
                q_sc_view_all.iterator, cute.make_layout(int(L.q_sc_group_bytes // 4))
            )
            q_sc_g1 = cute.make_tensor(
                q_sc_view_all.iterator + Int32(L.q_sc_group_bytes // 4),
                cute.make_layout(int(L.q_sc_group_bytes // 4)),
            )
        w_head_sc_g0 = cute.make_tensor(
            w_head_sc_view_all.iterator, cute.make_layout(int(L.w_head_sc_group_bytes // 4))
        )
        w_head_sc_g1 = cute.make_tensor(
            w_head_sc_view_all.iterator + Int32(L.w_head_sc_group_bytes // 4),
            cute.make_layout(int(L.w_head_sc_group_bytes // 4)),
        )

        mbar_base = st.mbar.data_ptr()
        n_buf = int(L.kv_bufs)
        if tid == Int32(0):
            for s in cutlass.range_constexpr(n_buf):
                cute.arch.mbarrier_init(mbar_base + s, Int32(1))
        cute.arch.barrier()

        section_len = Int32(topk_length[token_idx])
        if section_len < Int32(0):
            section_len = Int32(0)
        if section_len > Int32(self.topk):
            section_len = Int32(self.topk)
        is_empty_row = section_len == Int32(0)
        actual_tiles = (
            section_len + Int32(_CAND_WINDOW - 1)
        ) // Int32(_CAND_WINDOW)

        # DSV4 dual-cache (has_extra) union. main occupies COMPILE-TIME
        # num_main_tiles slots (= ceil(topk/64)); extra runs ceil(extra_section_len
        # /64) chunks. loop_tiles drives BOTH the IO prefetch loop and the math
        # loop. const_expr(not has_extra) pins loop_tiles == actual_tiles (the exact
        # current runtime value), so the no-extra trace + PTX are byte-identical.
        num_main_tiles = Int32(self.num_main_tiles)
        if cutlass.const_expr(has_extra):
            extra_total = Int32(self.extra_topk)
            extra_section_len = Int32(extra_topk_length[token_idx])
            if extra_section_len < Int32(0):
                extra_section_len = Int32(0)
            if extra_section_len > extra_total:
                extra_section_len = extra_total
            is_empty_row = is_empty_row and (extra_section_len == Int32(0))
            num_extra_tiles = (
                extra_section_len + Int32(_CAND_WINDOW - 1)
            ) // Int32(_CAND_WINDOW)
            loop_tiles = num_main_tiles + num_extra_tiles
        else:
            extra_section_len = section_len
            loop_tiles = actual_tiles

        topk_row = cute.make_tensor(
            indices.iterator + token_idx.to(Int64) * Int64(self.indices_stride_row),
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
        q_token = cute.make_tensor(
            q_all.iterator + token_idx.to(Int64) * Int64(self.q_stride_row),
            cute.make_layout(
                (self.num_heads, q_head_dim),
                stride=(self.q_stride_head, self.q_stride_dim),
            ),
        )
        warp_first_cand = warp_id * Int32(8)

        if is_io:
            cute.arch.setmaxregister_decrease(_IO_REGS)
            io_lane = tid - Int32(self.math_threads)

            # PROLOGUE: gather tile 0. For has_extra the launcher ASSERTs
            # num_main_tiles>=1 (topk==128 => 2), so tile 0 is ALWAYS the MAIN
            # section here -- no extra-prologue arm is needed. The guard becomes
            # loop_tiles>0 (== actual_tiles when not has_extra -> byte-identical).
            if loop_tiles > Int32(0):
                g_end0 = Int32(_CAND_WINDOW)
                if g_end0 > section_len:
                    g_end0 = section_len
                tok0 = cute.make_tensor(
                    token_idx_view_all.iterator, cute.make_layout(int(L.token_idx_buf_bytes // 4))
                )
                if cutlass.const_expr(is_glm):
                    io_issue_gather_glm_mg(
                        kv_cache_u8,
                        topk_row,
                        kv_fp8_addr,
                        tok0,
                        mbar_base,
                        Int32(0),
                        g_end0,
                        Int32(self.page_block_size),
                        stride_kv_block,
                        io_lane,
                        bi=t.bi,
                        kv_smem_stride=L.kv_smem_stride,
                        io_threads=_PREFILL_IO_THREADS,
                    )
                else:
                    io_issue_gather_dsv4_nope(
                        kv_cache_u8,
                        topk_row,
                        kv_fp8_addr,
                        kv_sc_addr,
                        tok0,
                        mbar_base,
                        Int32(0),
                        g_end0,
                        Int32(self.page_block_size),
                        stride_kv_block,
                        io_lane,
                        bi=t.bi,
                        kv_smem_stride=L.kv_smem_stride,
                        io_threads=_PREFILL_IO_THREADS,
                    )

            for lc in cutlass.range(loop_tiles, unroll=1):
                next_lc = Int32(lc) + Int32(1)
                if next_lc < loop_tiles:
                    buf = next_lc & Int32(1)
                    tok_buf_view = cute.make_tensor(
                        token_idx_view_all.iterator + buf * tok_buf_elems,
                        cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                    )
                    # Per-tile section dispatch (DSV4 dual-cache). The OUTER guard
                    # is const_expr(has_extra) so the whole dual arm is compile-time
                    # elided when not has_extra -> the no-extra `else` branch traces
                    # textually identical to today (byte-identical). has_extra is
                    # DSV4-only (GLM dual RAISEs), so the dual arm is DSV4-only:
                    # tiles >= num_main_tiles re-base into the EXTRA section and
                    # gather from the EXTRA cache (its own base ptr / page size /
                    # indices / stride); earlier tiles gather MAIN. The gather helper
                    # is section-agnostic so only the tensors/pbs/stride differ.
                    if cutlass.const_expr(has_extra):
                        if next_lc >= num_main_tiles:
                            cis = next_lc - num_main_tiles
                            g_start = cis * Int32(_CAND_WINDOW)
                            g_end = g_start + Int32(_CAND_WINDOW)
                            if g_end > extra_section_len:
                                g_end = extra_section_len
                            io_issue_gather_dsv4_nope(
                                extra_kv_cache_u8,
                                extra_row,
                                kv_fp8_addr + buf * kv_fp8_buf,
                                kv_sc_addr + buf * kv_sc_buf,
                                tok_buf_view,
                                mbar_base + buf,
                                g_start,
                                g_end,
                                Int32(self.pbs_extra),
                                stride_extra_kv_block,
                                io_lane,
                                bi=t.bi,
                                kv_smem_stride=L.kv_smem_stride,
                                io_threads=_PREFILL_IO_THREADS,
                            )
                        else:
                            g_start = next_lc * Int32(_CAND_WINDOW)
                            g_end = g_start + Int32(_CAND_WINDOW)
                            if g_end > section_len:
                                g_end = section_len
                            io_issue_gather_dsv4_nope(
                                kv_cache_u8,
                                topk_row,
                                kv_fp8_addr + buf * kv_fp8_buf,
                                kv_sc_addr + buf * kv_sc_buf,
                                tok_buf_view,
                                mbar_base + buf,
                                g_start,
                                g_end,
                                Int32(self.page_block_size),
                                stride_kv_block,
                                io_lane,
                                bi=t.bi,
                                kv_smem_stride=L.kv_smem_stride,
                                io_threads=_PREFILL_IO_THREADS,
                            )
                    else:
                        g_start = next_lc * Int32(_CAND_WINDOW)
                        g_end = g_start + Int32(_CAND_WINDOW)
                        if g_end > section_len:
                            g_end = section_len
                        if cutlass.const_expr(is_glm):
                            io_issue_gather_glm_mg(
                                kv_cache_u8,
                                topk_row,
                                kv_fp8_addr + buf * kv_fp8_buf,
                                tok_buf_view,
                                mbar_base + buf,
                                g_start,
                                g_end,
                                Int32(self.page_block_size),
                                stride_kv_block,
                                io_lane,
                                bi=t.bi,
                                kv_smem_stride=L.kv_smem_stride,
                                io_threads=_PREFILL_IO_THREADS,
                            )
                        else:
                            io_issue_gather_dsv4_nope(
                                kv_cache_u8,
                                topk_row,
                                kv_fp8_addr + buf * kv_fp8_buf,
                                kv_sc_addr + buf * kv_sc_buf,
                                tok_buf_view,
                                mbar_base + buf,
                                g_start,
                                g_end,
                                Int32(self.page_block_size),
                                stride_kv_block,
                                io_lane,
                                bi=t.bi,
                                kv_smem_stride=L.kv_smem_stride,
                                io_threads=_PREFILL_IO_THREADS,
                            )
                cute.arch.barrier(barrier_id=1, number_of_threads=self.block_threads)

        else:
            cute.arch.setmaxregister_increase(_MATH_REGS)
            n_acc_tiles = int(t.n_v_chunks) * int(t.nt_per_warp_xv)
            if cutlass.const_expr(bf16_qk):
                # S0 (BF16): load Q-NoPE straight to bf16 smem + Q-rope to the
                # aliased scratch, then preload Q-rope A operands to registers so
                # the scratch can be overwritten by W_FP8 in S6.
                s0_load_q_bf16_to_smem_mg(
                    q_token,
                    q_nope_bf16_g0,
                    q_nope_bf16_g1,
                    q_rope_g0,
                    q_rope_g1,
                    head_base,
                    tid,
                    d_nope=t.d_nope,
                    d_rope=t.d_rope,
                    hpb=t.hpb,
                    q_nope_bf16_stride=L.q_nope_bf16_stride,
                    num_threads=self.math_threads,
                    barrier_id=2,
                    n_hg=n_hg,
                    valid_hpb=self.valid_hpb,
                )
                q_rope_regs0 = preload_q_rope_regs_mg(q_rope_g0, lane, d_rope=t.d_rope)
                if cutlass.const_expr(n_hg == 2):
                    q_rope_regs1 = preload_q_rope_regs_mg(q_rope_g1, lane, d_rope=t.d_rope)
                else:
                    q_rope_regs1 = q_rope_regs0  # never read when n_hg==1.
                # Math/IO sync so the W_FP8 region (aliased Q-rope scratch) is free
                # for S6 once every math lane has its Q-rope regs.
                cute.arch.barrier(barrier_id=2, number_of_threads=self.math_threads)
            else:
                s0_quantize_q_to_smem(
                    q_token,
                    q_fp8_g0,
                    q_sc_g0,
                    q_rope_g0,
                    amax_view,
                    head_base,
                    Int32(self.valid_hpb),
                    tid,
                    d_nope=t.d_nope,
                    d_rope=t.d_rope,
                    d_qk=t.d_nope + t.d_rope,
                    quant_tile=t.quant_tile,
                    num_scales=t.num_scales,
                    hpb=t.hpb,
                    q_nope_stride=t.q_nope_stride,
                    num_threads=self.math_threads,
                    barrier_id=2,
                )
                if cutlass.const_expr(n_hg == 2):
                    s0_quantize_q_to_smem(
                        q_token,
                        q_fp8_g1,
                        q_sc_g1,
                        q_rope_g1,
                        amax_view,
                        head_base + Int32(t.hpb),
                        Int32(t.hpb),
                        tid,
                        d_nope=t.d_nope,
                        d_rope=t.d_rope,
                        d_qk=t.d_nope + t.d_rope,
                        quant_tile=t.quant_tile,
                        num_scales=t.num_scales,
                        hpb=t.hpb,
                        q_nope_stride=t.q_nope_stride,
                        num_threads=self.math_threads,
                        barrier_id=2,
                    )
                if cutlass.const_expr(is_glm):
                    # GLM FP8 registerizes Q-rope (the smem scratch aliases W_FP8,
                    # only live in S0/XV-free): preload the bf16 Q-rope A operands
                    # to registers, then sync so W_FP8 is free for S6 -- exactly
                    # like the BF16 path. DSV4 FP8 keeps Q-rope in smem (no preload).
                    q_rope_regs0 = preload_q_rope_regs_mg(q_rope_g0, lane, d_rope=t.d_rope)
                    if cutlass.const_expr(n_hg == 2):
                        q_rope_regs1 = preload_q_rope_regs_mg(q_rope_g1, lane, d_rope=t.d_rope)
                    else:
                        q_rope_regs1 = q_rope_regs0  # never read when n_hg==1.
                    cute.arch.barrier(barrier_id=2, number_of_threads=self.math_threads)

            acc0_frag = cute.make_rmem_tensor(n_acc_tiles * 4, Float32)
            acc1_frag = cute.make_rmem_tensor(n_acc_tiles * 4, Float32)
            rope0_frag = cute.make_rmem_tensor(4, Float32)
            rope1_frag = cute.make_rmem_tensor(4, Float32)
            gmax0_frag = cute.make_rmem_tensor(2, Float32)
            gmax1_frag = cute.make_rmem_tensor(2, Float32)
            gsum0_frag = cute.make_rmem_tensor(2, Float32)
            gsum1_frag = cute.make_rmem_tensor(2, Float32)
            for k in cutlass.range_constexpr(n_acc_tiles * 4):
                acc0_frag[k] = Float32(0.0)
                acc1_frag[k] = Float32(0.0)
            for k in cutlass.range_constexpr(4):
                rope0_frag[k] = Float32(0.0)
                rope1_frag[k] = Float32(0.0)
            gmax0_frag[0] = Float32(-1e30)
            gmax0_frag[1] = Float32(-1e30)
            gmax1_frag[0] = Float32(-1e30)
            gmax1_frag[1] = Float32(-1e30)
            gsum0_frag[0] = Float32(0.0)
            gsum0_frag[1] = Float32(0.0)
            gsum1_frag[0] = Float32(0.0)
            gsum1_frag[1] = Float32(0.0)

            if loop_tiles > Int32(0):
                cute.arch.mbarrier_wait(mbar_base, phase=0)

            for lc in cutlass.range(loop_tiles, unroll=1):
                ci = Int32(lc)
                # Per-tile section geometry (DSV4 dual-cache). An EXTRA tile (ci >=
                # num_main_tiles) re-bases the candidate offset WITHIN its section
                # and swaps in the extra section length; the s3 mask is
                # section-agnostic (masks abs_cand >= sec_len_now). The math reads
                # only the buffered smem the IO gathered, so it is correct as long
                # as the rope reads (below) use the matching cache geometry.
                # const_expr(not has_extra) pins the main expressions (the literal
                # section_len) -> byte-identical.
                # Default to the MAIN section. Under const_expr(has_extra) an EXTRA
                # tile (ci >= num_main_tiles) re-bases the candidate offset WITHIN its
                # section and swaps in the extra section length. The OUTER guard is
                # const_expr so the whole rebase is compile-time elided when not
                # has_extra -> the no-extra trace is byte-identical (split_cand_start
                # / sec_len_now are the literal main expressions, exactly as today).
                split_cand_start = ci * Int32(_CAND_WINDOW)
                sec_len_now = section_len
                if cutlass.const_expr(has_extra):
                    if ci >= num_main_tiles:
                        cis = ci - num_main_tiles
                        split_cand_start = cis * Int32(_CAND_WINDOW)
                        sec_len_now = extra_section_len
                split_cand_end = split_cand_start + Int32(_CAND_WINDOW)
                if split_cand_end > sec_len_now:
                    split_cand_end = sec_len_now
                buf = ci & Int32(1)
                kv_fp8_b = kv_fp8_addr + buf * kv_fp8_buf
                kv_sc_b = kv_sc_addr + buf * kv_sc_buf
                tok_buf_view = cute.make_tensor(
                    token_idx_view_all.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )
                # MAIN rope geometry (used by the FP8 / GLM QK-rope arms, which are
                # never on the dual path). The DSV4 BF16 QK-rope (s2) + XV-rope (s6b)
                # do their OWN per-tile section dispatch below (a const_expr(has_extra)
                # dynamic if/else that calls the rope helper with the EXTRA tensor in
                # one arm and the MAIN tensor in the other) -- the cute.Tensor cannot
                # be re-bound inside a dynamic `if`, so the section switch lives at
                # the call sites. const_expr(not has_extra) elides the dual arms ->
                # byte-identical.
                rope_cache = kv_cache_u8
                rope_pbs = Int32(self.page_block_size)
                rope_stride = stride_kv_block

                acc0 = [
                    [
                        acc0_frag[at * 4 + 0],
                        acc0_frag[at * 4 + 1],
                        acc0_frag[at * 4 + 2],
                        acc0_frag[at * 4 + 3],
                    ]
                    for at in range(n_acc_tiles)
                ]
                rope0 = [rope0_frag[0], rope0_frag[1], rope0_frag[2], rope0_frag[3]]
                gm0 = [gmax0_frag[0], gmax0_frag[1]]
                gs0 = [gsum0_frag[0], gsum0_frag[1]]
                qk0 = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                acc1 = [
                    [
                        acc1_frag[at * 4 + 0],
                        acc1_frag[at * 4 + 1],
                        acc1_frag[at * 4 + 2],
                        acc1_frag[at * 4 + 3],
                    ]
                    for at in range(n_acc_tiles)
                ]
                rope1 = [rope1_frag[0], rope1_frag[1], rope1_frag[2], rope1_frag[3]]
                gm1 = [gmax1_frag[0], gmax1_frag[1]]
                gs1 = [gsum1_frag[0], gsum1_frag[1]]
                qk1 = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]

                if cutlass.const_expr(bf16_qk):
                    # BF16-QK: fused two-group bf16 m16n8k16 QK-NoPE with inline
                    # FP8->BF16 K dequant (NO block-scaled FP8 MMA, NO Q-quant).
                    qk0, qk1 = s1_qk_nope_bf16_mg2(
                        qk0,
                        qk1,
                        q_nope_bf16_g0,
                        q_nope_bf16_g1,
                        kv_fp8_b,
                        kv_sc_b,
                        warp_first_cand,
                        lane,
                        num_scales=t.num_scales,
                        quant_tile=t.quant_tile,
                        q_nope_bf16_stride=L.q_nope_bf16_stride,
                        kv_smem_stride=L.kv_smem_stride,
                        scale_bytes_per_token=8,
                        n_hg=n_hg,
                    )
                    # QK-RoPE: Q-rope A from preloaded registers, KV-RoPE B from
                    # global (gathered once per tile, reused across groups). For the
                    # DSV4 dual path an EXTRA tile reads KV-rope from the EXTRA cache
                    # (its pbs/stride); the cute.Tensor cannot be re-bound in a dynamic
                    # if, so the section switch is a const_expr(has_extra) dynamic
                    # if/else around the helper call. const_expr(not has_extra) elides
                    # the dual arm -> a single MAIN call, byte-identical to today.
                    if cutlass.const_expr(has_extra):
                        if ci >= num_main_tiles:
                            qk0, qk1 = s2_qk_rope_regs_mg_dsv4(
                                qk0,
                                qk1,
                                q_rope_regs0,
                                q_rope_regs1,
                                extra_kv_cache_u8,
                                tok_buf_view,
                                warp_first_cand,
                                lane,
                                Int32(self.pbs_extra),
                                stride_extra_kv_block,
                                d_rope=t.d_rope,
                                n_hg=n_hg,
                            )
                        else:
                            qk0, qk1 = s2_qk_rope_regs_mg_dsv4(
                                qk0,
                                qk1,
                                q_rope_regs0,
                                q_rope_regs1,
                                kv_cache_u8,
                                tok_buf_view,
                                warp_first_cand,
                                lane,
                                Int32(self.page_block_size),
                                stride_kv_block,
                                d_rope=t.d_rope,
                                n_hg=n_hg,
                            )
                    else:
                        qk0, qk1 = s2_qk_rope_regs_mg_dsv4(
                            qk0,
                            qk1,
                            q_rope_regs0,
                            q_rope_regs1,
                            rope_cache,
                            tok_buf_view,
                            warp_first_cand,
                            lane,
                            rope_pbs,
                            rope_stride,
                            d_rope=t.d_rope,
                            n_hg=n_hg,
                        )
                else:
                    qk0 = s1_qk_nope_block_scaled(
                        qk0,
                        q_fp8_g0,
                        kv_fp8_b,
                        q_sc_g0,
                        kv_sc_b,
                        warp_first_cand,
                        lane,
                        num_scales=t.num_scales,
                        quant_tile=t.quant_tile,
                        q_nope_stride=t.q_nope_stride,
                        kv_smem_stride=L.kv_smem_stride,
                        scale_bytes_per_token=8,
                        scale_format=t.scale_format,
                        valid_hpb=self.valid_hpb,
                    )
                    if cutlass.const_expr(n_hg == 2):
                        qk1 = s1_qk_nope_block_scaled(
                            qk1,
                            q_fp8_g1,
                            kv_fp8_b,
                            q_sc_g1,
                            kv_sc_b,
                            warp_first_cand,
                            lane,
                            num_scales=t.num_scales,
                            quant_tile=t.quant_tile,
                            q_nope_stride=t.q_nope_stride,
                            kv_smem_stride=L.kv_smem_stride,
                            scale_bytes_per_token=8,
                            scale_format=t.scale_format,
                            valid_hpb=t.hpb,
                        )
                    if cutlass.const_expr(is_glm):
                        # GLM QK-RoPE: Q-rope A from preloaded registers, KV-rope B
                        # from global/L2 (GLM record packing), once per tile reused
                        # across head groups. v_has_rope=False so there is no XV-rope.
                        qk0, qk1 = s2_qk_rope_regs_mg_glm(
                            qk0,
                            qk1,
                            q_rope_regs0,
                            q_rope_regs1,
                            rope_cache,
                            tok_buf_view,
                            warp_first_cand,
                            lane,
                            rope_pbs,
                            rope_stride,
                            d_rope=t.d_rope,
                            n_hg=n_hg,
                            valid_hpb=self.valid_hpb,
                        )
                    else:
                        # Fused QK-RoPE: KV-RoPE B operand gathered ONCE per CTA tile
                        # (vectorized nc.u32 b16-pair), reused across head groups --
                        # matches FlashInfer's prefetch_kv_rope reuse.
                        qk0, qk1 = s2_qk_rope_global_mg_dsv4(
                            qk0,
                            qk1,
                            q_rope_g0,
                            q_rope_g1,
                            rope_cache,
                            tok_buf_view,
                            warp_first_cand,
                            lane,
                            rope_pbs,
                            rope_stride,
                            d_rope=t.d_rope,
                            n_hg=n_hg,
                            valid_hpb=self.valid_hpb,
                        )
                qk0 = s3_mask_and_scale(
                    qk0,
                    tok_buf_view,
                    warp_first_cand,
                    split_cand_start,
                    split_cand_end,
                    sec_len_now,
                    sm_scale_log2,
                    lane,
                )
                if cutlass.const_expr(n_hg == 2):
                    qk1 = s3_mask_and_scale(
                        qk1,
                        tok_buf_view,
                        warp_first_cand,
                        split_cand_start,
                        split_cand_end,
                        sec_len_now,
                        sm_scale_log2,
                        lane,
                    )
                p0 = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                p1 = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                p0, p1, wr00, wr01, wr10, wr11 = s4_online_softmax_mg2(
                    qk0,
                    qk1,
                    p0,
                    p1,
                    acc0,
                    acc1,
                    rope0,
                    rope1,
                    gm0,
                    gm1,
                    gs0,
                    gs1,
                    reduce_g0 + Int32(L.reduce_warp_max_group_off),
                    reduce_g0 + Int32(L.reduce_warp_sum_group_off),
                    reduce_g1 + Int32(L.reduce_warp_max_group_off),
                    reduce_g1 + Int32(L.reduce_warp_sum_group_off),
                    warp_id,
                    lane,
                    tid,
                    n_v_chunks=t.n_v_chunks,
                    hpb=t.hpb,
                    n_warps=8,
                    num_threads=self.math_threads,
                    barrier_id=3,
                    n_acc_tiles=n_acc_tiles,
                    n_hg=n_hg,
                    valid_hpb=self.valid_hpb,
                )
                w_pre0 = [p0[0] * wr00, p0[1] * wr00, p0[2] * wr01, p0[3] * wr01]
                w_pre1 = [p1[0] * wr10, p1[1] * wr10, p1[2] * wr11, p1[3] * wr11]

                if cutlass.const_expr(is_glm):
                    # GLM XV-NoPE: the per-group decode s6_xv_nope (raw-e4m3 V +
                    # per-(cand,vc) inline fp32 group scale + 2-pass W HIGH+LOW
                    # residual). V scales + V B operands come from the SHARED kv_fp8
                    # smem the IO gathered once per tile, so the two groups still
                    # share the KV gather (the MG win). v_has_rope=False -> NO S6b
                    # XV-rope.
                    #
                    # w_head_sc ZERO-INIT: the decode s6_xv_nope's per-(vc,head)
                    # absmax is an atomicMax-on-fp32 that ASSUMES w_head_sc was
                    # pre-zeroed (decode does it in s5_fill_sm_p_full; the DSV4 MG
                    # fused s6_xv_nope_mg_dsv4 does it internally). GLM calls the bare
                    # decode s6 with NO s5/sm_p_full, so the prefill arm MUST zero
                    # w_head_sc itself -- otherwise the FIRST tile of the FIRST launch
                    # atomicMaxes against uninitialized smem (catastrophic on cold
                    # launch) and later tiles never reset it (stale max bias).
                    i_hs = tid
                    while i_hs < Int32(n_hg * t.n_v_chunks * self.valid_hpb):
                        group_span = Int32(t.n_v_chunks * self.valid_hpb)
                        g_hs = i_hs // group_span
                        rem_hs = i_hs - g_hs * group_span
                        vc_hs = rem_hs // Int32(self.valid_hpb)
                        h_hs = rem_hs - vc_hs * Int32(self.valid_hpb)
                        slot_hs = (
                            g_hs * Int32(t.n_v_chunks * t.hpb)
                            + vc_hs * Int32(t.hpb)
                            + h_hs
                        )
                        w_head_sc_view_all[slot_hs] = Float32(0.0)
                        i_hs += Int32(self.math_threads)
                    cute.arch.barrier(
                        barrier_id=3, number_of_threads=self.math_threads
                    )
                    acc0 = s6_xv_nope(
                        w_pre0,
                        acc0,
                        kv_fp8_b,
                        kv_sc_b,
                        w_head_sc_g0,
                        glm_w_fp8_g0,
                        warp_id,
                        lane,
                        tid,
                        n_v_chunks=t.n_v_chunks,
                        v_chunk=t.quant_tile,
                        hpb=t.hpb,
                        bi=t.bi,
                        kv_smem_stride=L.kv_smem_stride,
                        w_fp8_stride=t.bi + 16,
                        n_warps=8,
                        scale_bytes_per_token=8,
                        nt_per_warp_xv=t.nt_per_warp_xv,
                        scale_format=t.scale_format,
                        num_threads=self.math_threads,
                        barrier_id=3,
                        valid_hpb=self.valid_hpb,
                        pack_hilo_rows=self.pack_hilo_rows,
                    )
                    if cutlass.const_expr(n_hg == 2):
                        acc1 = s6_xv_nope(
                            w_pre1,
                            acc1,
                            kv_fp8_b,
                            kv_sc_b,
                            w_head_sc_g1,
                            glm_w_fp8_g1,
                            warp_id,
                            lane,
                            tid,
                            n_v_chunks=t.n_v_chunks,
                            v_chunk=t.quant_tile,
                            hpb=t.hpb,
                            bi=t.bi,
                            kv_smem_stride=L.kv_smem_stride,
                            w_fp8_stride=t.bi + 16,
                            n_warps=8,
                            scale_bytes_per_token=8,
                            nt_per_warp_xv=t.nt_per_warp_xv,
                            scale_format=t.scale_format,
                            num_threads=self.math_threads,
                            barrier_id=3,
                            valid_hpb=t.hpb,
                        )
                else:
                    acc0, acc1 = s6_xv_nope_mg_dsv4(
                        w_pre0,
                        w_pre1,
                        acc0,
                        acc1,
                        kv_fp8_b,
                        kv_sc_b,
                        w_head_sc_view_all,
                        w_fp8_addr,
                        sm_p_g0,
                        sm_p_g1,
                        warp_id,
                        lane,
                        tid,
                        n_v_chunks=t.n_v_chunks,
                        v_chunk=t.quant_tile,
                        hpb=t.hpb,
                        bi=t.bi,
                        kv_smem_stride=L.kv_smem_stride,
                        w_fp8_stride=t.bi + 16,
                        w_fp8_group_stride=L.w_fp8_group_bytes,
                        w_fp8_parity_stride=L.w_fp8_parity_bytes,
                        n_warps=8,
                        scale_bytes_per_token=8,
                        nt_per_warp_xv=t.nt_per_warp_xv,
                        num_threads=self.math_threads,
                        barrier_id=3,
                        n_hg=n_hg,
                        row_xor=cutlass.const_expr(row_xor),
                    )
                    # XV-RoPE: per-tile section switch mirrors the QK-rope above (the
                    # cute.Tensor cannot be re-bound in a dynamic if). const_expr(not
                    # has_extra) elides the dual arm -> a single MAIN call,
                    # byte-identical.
                    if cutlass.const_expr(has_extra):
                        if ci >= num_main_tiles:
                            rope0, rope1 = s6b_xv_rope_global_mg_dsv4(
                                rope0,
                                rope1,
                                sm_p_g0,
                                sm_p_g1,
                                extra_kv_cache_u8,
                                tok_buf_view,
                                warp_id,
                                lane,
                                Int32(self.pbs_extra),
                                stride_extra_kv_block,
                                bi=t.bi,
                                d_rope=t.d_rope,
                                n_warps=8,
                                n_hg=n_hg,
                            )
                        else:
                            rope0, rope1 = s6b_xv_rope_global_mg_dsv4(
                                rope0,
                                rope1,
                                sm_p_g0,
                                sm_p_g1,
                                kv_cache_u8,
                                tok_buf_view,
                                warp_id,
                                lane,
                                Int32(self.page_block_size),
                                stride_kv_block,
                                bi=t.bi,
                                d_rope=t.d_rope,
                                n_warps=8,
                                n_hg=n_hg,
                            )
                    else:
                        rope0, rope1 = s6b_xv_rope_global_mg_dsv4(
                            rope0,
                            rope1,
                            sm_p_g0,
                            sm_p_g1,
                            rope_cache,
                            tok_buf_view,
                            warp_id,
                            lane,
                            rope_pbs,
                            rope_stride,
                            bi=t.bi,
                            d_rope=t.d_rope,
                            n_warps=8,
                            n_hg=n_hg,
                        )
                for at in cutlass.range_constexpr(n_acc_tiles):
                    acc0_frag[at * 4 + 0] = acc0[at][0]
                    acc0_frag[at * 4 + 1] = acc0[at][1]
                    acc0_frag[at * 4 + 2] = acc0[at][2]
                    acc0_frag[at * 4 + 3] = acc0[at][3]
                rope0_frag[0] = rope0[0]
                rope0_frag[1] = rope0[1]
                rope0_frag[2] = rope0[2]
                rope0_frag[3] = rope0[3]
                gmax0_frag[0] = gm0[0]
                gmax0_frag[1] = gm0[1]
                gsum0_frag[0] = gs0[0]
                gsum0_frag[1] = gs0[1]
                if cutlass.const_expr(n_hg == 2):
                    for at in cutlass.range_constexpr(n_acc_tiles):
                        acc1_frag[at * 4 + 0] = acc1[at][0]
                        acc1_frag[at * 4 + 1] = acc1[at][1]
                        acc1_frag[at * 4 + 2] = acc1[at][2]
                        acc1_frag[at * 4 + 3] = acc1[at][3]
                    rope1_frag[0] = rope1[0]
                    rope1_frag[1] = rope1[1]
                    rope1_frag[2] = rope1[2]
                    rope1_frag[3] = rope1[3]
                    gmax1_frag[0] = gm1[0]
                    gmax1_frag[1] = gm1[1]
                    gsum1_frag[0] = gs1[0]
                    gsum1_frag[1] = gs1[1]

                cute.arch.barrier(barrier_id=1, number_of_threads=self.block_threads)
                next_lc = ci + Int32(1)
                if next_lc < loop_tiles:
                    next_phase = (next_lc >> Int32(1)) & Int32(1)
                    cute.arch.mbarrier_wait(mbar_base + (next_lc & Int32(1)), phase=next_phase)

            if is_empty_row:
                gsum0_frag[0] = Float32(0.0)
                gsum0_frag[1] = Float32(0.0)
                if cutlass.const_expr(n_hg == 2):
                    gsum1_frag[0] = Float32(0.0)
                    gsum1_frag[1] = Float32(0.0)

            final_sum0 = [gsum0_frag[0], gsum0_frag[1]]
            final_sum1 = [gsum1_frag[0], gsum1_frag[1]]
            final_sum0, final_sum1 = s4_finalize_row_sum_mg2(
                final_sum0,
                final_sum1,
                reduce_g0 + Int32(L.reduce_warp_sum_group_off),
                reduce_g1 + Int32(L.reduce_warp_sum_group_off),
                warp_id,
                lane,
                tid,
                hpb=t.hpb,
                n_warps=8,
                num_threads=self.math_threads,
                barrier_id=3,
                n_hg=n_hg,
                valid_hpb=self.valid_hpb,
            )
            gsum0_frag[0] = final_sum0[0]
            gsum0_frag[1] = final_sum0[1]
            if cutlass.const_expr(n_hg == 2):
                gsum1_frag[0] = final_sum1[0]
                gsum1_frag[1] = final_sum1[1]

            fin0 = [
                [
                    acc0_frag[at * 4 + 0],
                    acc0_frag[at * 4 + 1],
                    acc0_frag[at * 4 + 2],
                    acc0_frag[at * 4 + 3],
                ]
                for at in range(n_acc_tiles)
            ]
            rope0 = [rope0_frag[0], rope0_frag[1], rope0_frag[2], rope0_frag[3]]
            out_o0 = cute.make_tensor(
                output.iterator
                + token_idx.to(Int64) * Int64(self.output_stride_row)
                + head_base.to(Int64) * Int64(self.output_stride_head),
                cute.make_layout(
                    (self.valid_hpb, t.d_v),
                    stride=(self.output_stride_head, self.output_stride_dim),
                ),
            )
            out_lse0 = cute.make_tensor(
                out_lse.iterator
                + token_idx.to(Int64) * Int64(self.out_lse_stride_row)
                + head_base.to(Int64) * Int64(self.out_lse_stride_head),
                cute.make_layout((self.valid_hpb,), stride=(self.out_lse_stride_head,)),
            )
            s7_epilogue(
                fin0,
                rope0,
                [gmax0_frag[0], gmax0_frag[1]],
                [gsum0_frag[0], gsum0_frag[1]],
                out_o0,
                out_lse0,
                warp_id,
                lane,
                n_v_chunks=t.n_v_chunks,
                v_chunk=t.quant_tile,
                d_nope=t.d_nope,
                d_rope=t.d_rope,
                n_warps=8,
                valid_hpb=self.valid_hpb,
                nt_per_warp_xv=t.nt_per_warp_xv,
                v_has_rope=t.v_has_rope,
                epilogue_mode=EPILOGUE_FINAL_BF16,
                has_attn_sink=self.has_sink,
                attn_sink=attn_sink,
                head_base=head_base,
            )

            # Group-1 epilogue: const_expr-elided for n_hg==1 (single-group MG ->
            # only group-0 output). The inter-group barrier (group-0 epilogue reads
            # smem the group-1 path reuses) is part of the gated block, so n_hg==1
            # emits neither it nor any group-1 instruction.
            if cutlass.const_expr(n_hg == 2):
                cute.arch.barrier(barrier_id=3, number_of_threads=self.math_threads)

                fin1 = [
                    [
                        acc1_frag[at * 4 + 0],
                        acc1_frag[at * 4 + 1],
                        acc1_frag[at * 4 + 2],
                        acc1_frag[at * 4 + 3],
                    ]
                    for at in range(n_acc_tiles)
                ]
                rope1 = [rope1_frag[0], rope1_frag[1], rope1_frag[2], rope1_frag[3]]
                head_base1 = head_base + Int32(t.hpb)
                out_o1 = cute.make_tensor(
                    output.iterator
                    + token_idx.to(Int64) * Int64(self.output_stride_row)
                    + head_base1.to(Int64) * Int64(self.output_stride_head),
                    cute.make_layout(
                        (t.hpb, t.d_v),
                        stride=(self.output_stride_head, self.output_stride_dim),
                    ),
                )
                out_lse1 = cute.make_tensor(
                    out_lse.iterator
                    + token_idx.to(Int64) * Int64(self.out_lse_stride_row)
                    + head_base1.to(Int64) * Int64(self.out_lse_stride_head),
                    cute.make_layout((t.hpb,), stride=(self.out_lse_stride_head,)),
                )
                s7_epilogue(
                    fin1,
                    rope1,
                    [gmax1_frag[0], gmax1_frag[1]],
                    [gsum1_frag[0], gsum1_frag[1]],
                    out_o1,
                    out_lse1,
                    warp_id,
                    lane,
                    n_v_chunks=t.n_v_chunks,
                    v_chunk=t.quant_tile,
                    d_nope=t.d_nope,
                    d_rope=t.d_rope,
                    n_warps=8,
                    valid_hpb=t.hpb,
                    nt_per_warp_xv=t.nt_per_warp_xv,
                    v_has_rope=t.v_has_rope,
                    epilogue_mode=EPILOGUE_FINAL_BF16,
                    has_attn_sink=self.has_sink,
                    attn_sink=attn_sink,
                    head_base=head_base1,
                )


def _sparse_mla_prefill_mg_flat_launch(
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
    page_block_size: int,
    topk: int,
    num_tiles: int,
    stride_kv_block: int,
    has_sink: bool,
    compute_mode: int,
    mg_n_hg: int,
    model_type: int,
    scale_format: int,
    has_extra: bool,
    extra_topk: int,
    num_main_tiles: int,
    pbs_extra: int,
    stride_extra_kv_block: int,
    row_xor: bool,
    *,
    active_heads: int | None = None,
    head_offset: int = 0,
) -> None:
    traits = make_unified_traits(int(model_type), compute_mode, int(scale_format))
    layout = make_smem_layout_mg(traits, int(mg_n_hg))
    q_head_dim = int(q.shape[2])
    total_heads = int(q.shape[1])
    if active_heads is None:
        active_heads = total_heads
    active_heads = int(active_heads)
    head_offset = int(head_offset)
    heads_per_cta = int(layout.heads_per_cta)
    if active_heads <= 0:
        raise ValueError(f"SM120 sparse MLA MG prefill requires active_heads>0, got {active_heads}")
    if head_offset < 0 or head_offset + active_heads > total_heads:
        raise ValueError(
            "SM120 sparse MLA MG prefill head range out of bounds: "
            f"head_offset={head_offset}, active_heads={active_heads}, total_heads={total_heads}"
        )
    if active_heads % heads_per_cta == 0:
        valid_hpb = int(traits.hpb)
        replicate_h = active_heads // heads_per_cta
    elif int(mg_n_hg) == 1 and active_heads == int(traits.hpb) // 2:
        valid_hpb = active_heads
        replicate_h = 1
    else:
        raise ValueError(
            "SM120 sparse MLA MG prefill active head range must be divisible by "
            f"heads_per_cta={heads_per_cta} (or active_heads==hpb//2 with "
            f"mg_n_hg==1 for the 8-head shard); got active_heads={active_heads}"
        )
    pack_hilo_rows = (
        int(model_type) == int(ModelType.GLM_NSA)
        and int(scale_format) == int(ScaleFormat.ARBITRARY_FP32)
        and int(mg_n_hg) == 1
        and valid_hpb == 8
        and os.environ.get("B12X_MLA_SM120_PREFILL_PACK_HILO_ROWS", "1")
        not in ("0", "false", "False", "off")
    )
    # DSV4 dual-cache MG: the union puts tile 0 in the MAIN section, so
    # num_main_tiles MUST be >= 1 (topk==128 => 2). All-extra (num_main_tiles==0)
    # has no extra-prologue arm and is forbidden.
    if bool(has_extra) and int(num_main_tiles) < 1:
        raise ValueError(
            "SM120 sparse MLA MG dual-cache prefill requires num_main_tiles>=1 "
            f"(topk>=1); got num_main_tiles={num_main_tiles}"
        )
    kernel = UnifiedPrefillMGKernel(
        traits,
        layout,
        int(page_block_size),
        int(num_tiles),
        replicate_h=replicate_h,
        num_heads=total_heads,
        q_stride=tuple(q.stride()),
        indices_stride0=int(topk_indices.stride(0)),
        output_stride=tuple(output.stride()),
        out_lse_stride=tuple(lse_out.stride()),
        has_sink=bool(has_sink),
        topk=int(topk),
        has_extra=bool(has_extra),
        pbs_extra=int(pbs_extra),
        num_main_tiles=int(num_main_tiles),
        extra_topk=int(extra_topk),
        extra_indices_stride0=int(extra_indices_t.stride(0)),
        row_xor=bool(row_xor),
        head_offset=head_offset,
        valid_hpb=valid_hpb,
        pack_hilo_rows=pack_hilo_rows,
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
        # call_dual signature: ... stride_kv_block, extra_kv, extra_indices,
        # extra_topk_length, stride_extra_kv_block, num_tokens, stream.
        args = base_args + (
            _to_cute(extra_kv_flat, cutlass.Uint8, align=16),
            _to_cute(extra_indices_t, cutlass.Int32, align=4, dynamic_layout=True),
            _to_cute(extra_len_t, cutlass.Int32, align=4, dynamic_layout=True),
            Int64(stride_extra_kv_block),
            Int32(int(q.shape[0])),
            stream,
        )
    else:
        args = base_args + (Int32(int(q.shape[0])), stream)
    spec_fields = [
        key_field("model_type", int(model_type)),
        key_field("compute_mode", int(compute_mode)),
        key_field("scale_format", int(scale_format)),
        key_field("num_heads", total_heads),
        key_field("heads_per_cta", heads_per_cta),
        key_field("mg_n_hg", int(mg_n_hg)),
        key_field("valid_hpb", int(valid_hpb)),
        key_field("pack_hilo_rows", int(pack_hilo_rows)),
        key_field("num_tiles", int(num_tiles)),
        key_field("page_block_size", int(page_block_size)),
        key_field("topk_bucket", _topk_bucket(topk)),
        key_field("has_sink", int(has_sink)),
        tensor_key(
            "q",
            q,
            dims=(DimKey.dynamic(), DimKey.exact(total_heads), DimKey.exact(q_head_dim)),
        ),
        tensor_key("topk_indices", topk_indices, dims=(DimKey.dynamic(), DimKey.bucket(topk))),
        tensor_key(
            "output",
            output,
            dims=(DimKey.dynamic(), DimKey.exact(total_heads), DimKey.exact(512)),
        ),
        tensor_key("out_lse", lse_out, dims=(DimKey.dynamic(), DimKey.exact(total_heads))),
    ]
    if active_heads != total_heads or head_offset != 0:
        spec_fields.extend(
            [
                key_field("active_heads", active_heads),
                key_field("head_offset", head_offset),
            ]
        )
    if has_extra:
        # The dual key_fields are appended ONLY for has_extra so the single-cache
        # compile-spec hash is byte-unchanged (decode/no-extra cache key identical).
        spec_fields.extend(
            [
                key_field("has_extra", int(has_extra)),
                key_field("num_main_tiles", int(num_main_tiles)),
                key_field("pbs_extra", int(pbs_extra)),
                key_field("extra_topk_bucket", _topk_bucket(extra_topk)),
                key_field("row_xor", int(row_xor)),
                tensor_key(
                    "extra_indices",
                    extra_indices_t,
                    dims=(DimKey.dynamic(), DimKey.bucket(max(extra_topk, 1))),
                ),
            ]
        )
    compile_spec = KernelCompileSpec.from_fields(
        "attention.mla.sm120.prefill_mg",
        1,
        *spec_fields,
    )
    entry = kernel.call_dual if has_extra else kernel
    b12x_launch(entry, compile_spec=compile_spec, compile_args=args, runtime_args=args)


@torch.library.custom_op(
    "b12x::sparse_mla_sm120_prefill_mg",
    mutates_args=("output", "lse_out"),
)
def _sparse_mla_prefill_mg_op(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink_t: torch.Tensor,
    output: torch.Tensor,
    lse_out: torch.Tensor,
    sm_scale: float,
    page_block_size: int,
    topk: int,
    num_tiles: int,
    stride_kv_block: int,
    has_sink: bool,
    compute_mode: int,
    mg_n_hg: int,
    model_type: int,
    scale_format: int,
) -> None:
    # SINGLE-CACHE op (DSV4 main / DSV4-BF16 / GLM MG). Byte-identical: passes
    # has_extra=False with the extra device args aliased to the main tensors (never
    # read under const_expr(has_extra=False)). Op signature is UNCHANGED so the
    # no-extra cache key and all existing callers are untouched.
    _sparse_mla_prefill_mg_flat_launch(
        q,
        kv_flat,
        topk_indices,
        topk_length,
        attn_sink_t,
        output,
        lse_out,
        kv_flat,        # extra_kv_flat alias (never read)
        topk_indices,   # extra_indices alias (never read)
        topk_length,    # extra_len alias (never read)
        sm_scale,
        page_block_size,
        topk,
        num_tiles,
        stride_kv_block,
        has_sink,
        compute_mode,
        mg_n_hg,
        model_type,
        scale_format,
        False,  # has_extra
        0,      # extra_topk
        0,      # num_main_tiles
        1,      # pbs_extra
        0,      # stride_extra_kv_block
        False,  # row_xor
    )


@_sparse_mla_prefill_mg_op.register_fake
def _sparse_mla_prefill_mg_fake(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink_t: torch.Tensor,
    output: torch.Tensor,
    lse_out: torch.Tensor,
    sm_scale: float,
    page_block_size: int,
    topk: int,
    num_tiles: int,
    stride_kv_block: int,
    has_sink: bool,
    compute_mode: int,
    mg_n_hg: int,
    model_type: int,
    scale_format: int,
) -> None:
    return None


@torch.library.custom_op(
    "b12x::sparse_mla_sm120_prefill_mg_dual",
    mutates_args=("output", "lse_out"),
)
def _sparse_mla_prefill_mg_dual_op(
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
    page_block_size: int,
    topk: int,
    num_tiles: int,
    stride_kv_block: int,
    has_sink: bool,
    compute_mode: int,
    mg_n_hg: int,
    model_type: int,
    scale_format: int,
    extra_topk: int,
    num_main_tiles: int,
    pbs_extra: int,
    stride_extra_kv_block: int,
    row_xor: bool,
) -> None:
    # DUAL-CACHE op (DSV4 has_extra union -> MG). SEPARATE op from the single-cache
    # one so the no-extra op signature / cache key stays byte-identical. has_extra
    # is implicitly True.
    _sparse_mla_prefill_mg_flat_launch(
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
        page_block_size,
        topk,
        num_tiles,
        stride_kv_block,
        has_sink,
        compute_mode,
        mg_n_hg,
        model_type,
        scale_format,
        True,  # has_extra
        extra_topk,
        num_main_tiles,
        pbs_extra,
        stride_extra_kv_block,
        row_xor,
    )


@_sparse_mla_prefill_mg_dual_op.register_fake
def _sparse_mla_prefill_mg_dual_fake(
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
    page_block_size: int,
    topk: int,
    num_tiles: int,
    stride_kv_block: int,
    has_sink: bool,
    compute_mode: int,
    mg_n_hg: int,
    model_type: int,
    scale_format: int,
    extra_topk: int,
    num_main_tiles: int,
    pbs_extra: int,
    stride_extra_kv_block: int,
    row_xor: bool,
) -> None:
    return None


def run_unified_prefill_mg(
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
    compute_mode: int = ComputeMode.FP8,
    mg_n_hg: int = 2,
    model_type: int = ModelType.DSV4,
    scale_format: int | None = None,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    extra_page_block_size: int | None = None,
    stride_extra_kv_block: int | None = None,
    active_heads: int | None = None,
    head_offset: int = 0,
):
    model_type = int(model_type)
    is_glm = model_type == ModelType.GLM_NSA
    # DSV4 dual-cache (has_extra) union. all-or-none + DSV4-only (GLM has no extra
    # section). Forced BF16-QK by the caller (FI ships dual-cache as BF16 only).
    has_extra = extra_kv_cache is not None
    if has_extra:
        if extra_indices is None or extra_page_block_size is None:
            raise ValueError(
                "SM120 sparse MLA MG dual-cache prefill requires extra_kv_cache, "
                "extra_indices, and extra_page_block_size together"
            )
        if is_glm:
            raise ValueError(
                "SM120 sparse MLA MG dual-cache prefill is DSV4-only "
                "(q_head_dim==512); GLM/DSV3.2 has no extra cache"
            )
    if scale_format is None:
        scale_format = ScaleFormat.ARBITRARY_FP32 if is_glm else ScaleFormat.UE8M0_BYTE
    scale_format = int(scale_format)
    expected_qdim = _GLM_HEAD_DIM if is_glm else _DSV4_HEAD_DIM
    if int(q.shape[-1]) != expected_qdim:
        raise ValueError(
            f"SM120 sparse MLA MG prefill ({'GLM' if is_glm else 'DSV4'}) expects "
            f"q_head_dim={expected_qdim}, got {int(q.shape[-1])}"
        )
    if int(mg_n_hg) not in (1, 2):
        raise ValueError(f"SM120 sparse MLA MG prefill supports mg_n_hg in {{1, 2}}, got {mg_n_hg}")
    num_tokens, heads, _ = q.shape
    total_heads = int(heads)
    if active_heads is None:
        active_heads = total_heads
    active_heads = int(active_heads)
    head_offset = int(head_offset)
    if active_heads <= 0:
        raise ValueError(f"SM120 sparse MLA MG prefill requires active_heads>0, got {active_heads}")
    if head_offset < 0 or head_offset + active_heads > total_heads:
        raise ValueError(
            "SM120 sparse MLA MG prefill head range out of bounds: "
            f"head_offset={head_offset}, active_heads={active_heads}, total_heads={total_heads}"
        )
    traits = make_unified_traits(model_type, int(compute_mode), scale_format)
    # heads_per_cta = mg_n_hg * HPB. mg_n_hg==2 covers paired head groups; mg_n_hg==1
    # covers a single-group launch, including 16-head tails and the heads==8
    # valid_hpb shard. The caller picks mg_n_hg and active head range.
    heads_per_cta = int(mg_n_hg) * int(traits.hpb)
    full_head_tile = active_heads % heads_per_cta == 0
    valid_hpb_shard = int(mg_n_hg) == 1 and active_heads == int(traits.hpb) // 2
    if not full_head_tile and not valid_hpb_shard:
        raise ValueError(
            f"SM120 sparse MLA MG prefill (mg_n_hg={mg_n_hg}) requires heads divisible "
            f"by {heads_per_cta} (or active_heads==hpb//2 with mg_n_hg==1 for the "
            f"8-head shard), got active_heads={active_heads}"
        )

    topk = int(topk_indices.shape[1])
    num_main_tiles = (topk + _CAND_WINDOW - 1) // _CAND_WINDOW
    device = q.device
    if topk_length is None:
        topk_length = torch.full((num_tokens,), topk, dtype=torch.int32, device=device)
    else:
        topk_length = topk_length.to(device=device, dtype=torch.int32).contiguous()

    has_sink = attn_sink is not None
    if has_sink:
        attn_sink_t = attn_sink.to(device=device, dtype=torch.float32).contiguous()
    else:
        attn_sink_t = torch.zeros(1, dtype=torch.float32, device=device)

    if stride_kv_block is None:
        stride_kv_block = _cache_block_stride_bytes(
            kv_cache,
            page_size=int(page_block_size),
            is_glm=bool(is_glm),
        )

    q = q.contiguous()
    topk_indices = topk_indices.contiguous()
    if output is None:
        output = torch.empty((num_tokens, heads, int(traits.d_v)), dtype=torch.bfloat16, device=device)
    if lse_out is None:
        lse_out = torch.empty((num_tokens, heads), dtype=torch.float32, device=device)

    if has_extra:
        # DSV4 dual-cache union -> the separate dual op. num_main_tiles main chunks
        # then ceil(extra_topk/64) extra chunks; the kernel masks each section by
        # its per-token length. row_xor = (pbs_extra == 2) (FI USE_WFP8_ROW_XOR).
        pbs_extra = int(extra_page_block_size)
        if stride_extra_kv_block is None:
            stride_extra_kv_block = _cache_block_stride_bytes(
                extra_kv_cache,
                page_size=pbs_extra,
                is_glm=False,
            )
        extra_topk = int(extra_indices.shape[1])
        num_extra_tiles = (extra_topk + _CAND_WINDOW - 1) // _CAND_WINDOW
        num_tiles = num_main_tiles + num_extra_tiles
        row_xor = pbs_extra == 2
        extra_indices_t = extra_indices.contiguous()
        if extra_topk_length is None:
            extra_len_t = torch.full(
                (num_tokens,), extra_topk, dtype=torch.int32, device=device
            )
        else:
            extra_len_t = extra_topk_length.to(
                device=device, dtype=torch.int32
            ).contiguous()
        if full_head_tile and active_heads == total_heads and head_offset == 0:
            torch.ops.b12x.sparse_mla_sm120_prefill_mg_dual(
                q,
                _cache_base_tensor(kv_cache),
                topk_indices,
                topk_length,
                attn_sink_t,
                output,
                lse_out,
                _cache_base_tensor(extra_kv_cache),
                extra_indices_t,
                extra_len_t,
                float(sm_scale),
                int(page_block_size),
                int(topk),
                int(num_tiles),
                int(stride_kv_block),
                bool(has_sink),
                int(compute_mode),
                int(mg_n_hg),
                model_type,
                scale_format,
                int(extra_topk),
                int(num_main_tiles),
                int(pbs_extra),
                int(stride_extra_kv_block),
                bool(row_xor),
            )
        else:
            _sparse_mla_prefill_mg_flat_launch(
                q,
                _cache_base_tensor(kv_cache),
                topk_indices,
                topk_length,
                attn_sink_t,
                output,
                lse_out,
                _cache_base_tensor(extra_kv_cache),
                extra_indices_t,
                extra_len_t,
                float(sm_scale),
                int(page_block_size),
                int(topk),
                int(num_tiles),
                int(stride_kv_block),
                bool(has_sink),
                int(compute_mode),
                int(mg_n_hg),
                model_type,
                scale_format,
                True,  # has_extra
                int(extra_topk),
                int(num_main_tiles),
                int(pbs_extra),
                int(stride_extra_kv_block),
                bool(row_xor),
                active_heads=active_heads,
                head_offset=head_offset,
            )
        return output, lse_out

    if full_head_tile and active_heads == total_heads and head_offset == 0:
        torch.ops.b12x.sparse_mla_sm120_prefill_mg(
            q,
            _cache_base_tensor(kv_cache),
            topk_indices,
            topk_length,
            attn_sink_t,
            output,
            lse_out,
            float(sm_scale),
            int(page_block_size),
            int(topk),
            int(num_main_tiles),
            int(stride_kv_block),
            bool(has_sink),
            int(compute_mode),
            int(mg_n_hg),
            model_type,
            scale_format,
        )
    else:
        _sparse_mla_prefill_mg_flat_launch(
            q,
            _cache_base_tensor(kv_cache),
            topk_indices,
            topk_length,
            attn_sink_t,
            output,
            lse_out,
            _cache_base_tensor(kv_cache),  # extra_kv_flat alias (never read)
            topk_indices,          # extra_indices alias (never read)
            topk_length,           # extra_len alias (never read)
            float(sm_scale),
            int(page_block_size),
            int(topk),
            int(num_main_tiles),
            int(stride_kv_block),
            bool(has_sink),
            int(compute_mode),
            int(mg_n_hg),
            model_type,
            scale_format,
            False,  # has_extra
            0,      # extra_topk
            0,      # num_main_tiles
            1,      # pbs_extra
            0,      # stride_extra_kv_block
            False,  # row_xor
            active_heads=active_heads,
            head_offset=head_offset,
        )
    return output, lse_out
