"""FlashInfer-shaped SM120 DSV4 MG prefill path.

This is a dedicated DSV4/FP8/main-cache prefill kernel matching FlashInfer's
multi-head-group launch structure for NUM_HEADS in {32, 64, 128}: one CTA handles
two HPB=16 head groups and reuses a single NoPE KV gather across both groups.
The DSV4 RoPE payload is intentionally not bulk-staged into smem; QK-RoPE and
XV-RoPE read it from global/L2.
"""

from __future__ import annotations

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
    _ld_u8_zext,
    _quant_e4m3_byte,
    _ue8m0_byte_to_fp32,
    ld_shared_f32,
    s0_quantize_q_to_smem,
    s1_qk_nope_block_scaled,
    s3_mask_and_scale,
    st_shared_f32,
    s7_epilogue,
)
from .io_mg import io_issue_gather_dsv4_nope
from .smem_mg import get_prefill_mg_shared_storage_cls, make_smem_layout_mg
from .traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits


_CAND_WINDOW = 64
_DSV4_HEAD_DIM = 512
_PREFILL_BLOCK_THREADS = 384
_PREFILL_IO_THREADS = 128
_IO_REGS = 32
_MATH_REGS = 232
_DSV4_IO_STRIDE = 576
_DSV4_ROPE_GMEM_OFFSET = 448


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
):
    """Two-group S4 online softmax with FlashInfer-style deferred row sums."""
    bar_kw = dict(barrier_id=barrier_id, number_of_threads=num_threads)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)

    lm00 = fmax_f32(qk0[0], qk0[1])
    lm01 = fmax_f32(qk0[2], qk0[3])
    lm10 = fmax_f32(qk1[0], qk1[1])
    lm11 = fmax_f32(qk1[2], qk1[3])
    lm00 = fmax_f32(lm00, cute.arch.shuffle_sync_bfly(lm00, offset=2))
    lm01 = fmax_f32(lm01, cute.arch.shuffle_sync_bfly(lm01, offset=2))
    lm10 = fmax_f32(lm10, cute.arch.shuffle_sync_bfly(lm10, offset=2))
    lm11 = fmax_f32(lm11, cute.arch.shuffle_sync_bfly(lm11, offset=2))
    lm00 = fmax_f32(lm00, cute.arch.shuffle_sync_bfly(lm00, offset=1))
    lm01 = fmax_f32(lm01, cute.arch.shuffle_sync_bfly(lm01, offset=1))
    lm10 = fmax_f32(lm10, cute.arch.shuffle_sync_bfly(lm10, offset=1))
    lm11 = fmax_f32(lm11, cute.arch.shuffle_sync_bfly(lm11, offset=1))

    if tid == Int32(0):
        st_shared_f32(reduce0_max_addr + (warp_id * Int32(hpb) + gid) * Int32(4), lm00)
        st_shared_f32(
            reduce0_max_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
            lm01,
        )
        st_shared_f32(reduce1_max_addr + (warp_id * Int32(hpb) + gid) * Int32(4), lm10)
        st_shared_f32(
            reduce1_max_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
            lm11,
        )
    cute.arch.barrier(**bar_kw)

    if tid_flat < Int32(2 * hpb):
        group = tid_flat // Int32(hpb)
        h = tid_flat - group * Int32(hpb)
        rmax = reduce0_max_addr
        if group != Int32(0):
            rmax = reduce1_max_addr
        bmax = Float32(-1e30)
        for w in cutlass.range_constexpr(n_warps):
            wm = ld_shared_f32(rmax + (Int32(w) * Int32(hpb) + h) * Int32(4))
            bmax = fmax_f32(bmax, wm)
        st_shared_f32(rmax + h * Int32(4), bmax)
    cute.arch.barrier(**bar_kw)

    blm00 = ld_shared_f32(reduce0_max_addr + gid * Int32(4))
    blm01 = ld_shared_f32(reduce0_max_addr + (gid + Int32(8)) * Int32(4))
    blm10 = ld_shared_f32(reduce1_max_addr + gid * Int32(4))
    blm11 = ld_shared_f32(reduce1_max_addr + (gid + Int32(8)) * Int32(4))

    ngm00 = fmax_f32(gm0[0], blm00)
    ngm01 = fmax_f32(gm0[1], blm01)
    ngm10 = fmax_f32(gm1[0], blm10)
    ngm11 = fmax_f32(gm1[1], blm11)
    alpha00 = _exp2_approx_ftz_f32(gm0[0] - ngm00)
    alpha01 = _exp2_approx_ftz_f32(gm0[1] - ngm01)
    alpha10 = _exp2_approx_ftz_f32(gm1[0] - ngm10)
    alpha11 = _exp2_approx_ftz_f32(gm1[1] - ngm11)

    for vc in cutlass.range_constexpr(n_acc_tiles):
        acc0[vc][0] = acc0[vc][0] * alpha00
        acc0[vc][1] = acc0[vc][1] * alpha00
        acc0[vc][2] = acc0[vc][2] * alpha01
        acc0[vc][3] = acc0[vc][3] * alpha01
        acc1[vc][0] = acc1[vc][0] * alpha10
        acc1[vc][1] = acc1[vc][1] * alpha10
        acc1[vc][2] = acc1[vc][2] * alpha11
        acc1[vc][3] = acc1[vc][3] * alpha11
    rope0[0] = rope0[0] * alpha00
    rope0[1] = rope0[1] * alpha00
    rope0[2] = rope0[2] * alpha01
    rope0[3] = rope0[3] * alpha01
    rope1[0] = rope1[0] * alpha10
    rope1[1] = rope1[1] * alpha10
    rope1[2] = rope1[2] * alpha11
    rope1[3] = rope1[3] * alpha11

    gs0[0] = gs0[0] * alpha00
    gs0[1] = gs0[1] * alpha01
    gs1[0] = gs1[0] * alpha10
    gs1[1] = gs1[1] * alpha11

    p0[0] = _exp2_approx_ftz_f32(qk0[0] - ngm00)
    p0[1] = _exp2_approx_ftz_f32(qk0[1] - ngm00)
    p0[2] = _exp2_approx_ftz_f32(qk0[2] - ngm01)
    p0[3] = _exp2_approx_ftz_f32(qk0[3] - ngm01)
    p1[0] = _exp2_approx_ftz_f32(qk1[0] - ngm10)
    p1[1] = _exp2_approx_ftz_f32(qk1[1] - ngm10)
    p1[2] = _exp2_approx_ftz_f32(qk1[2] - ngm11)
    p1[3] = _exp2_approx_ftz_f32(qk1[3] - ngm11)

    ls00 = p0[0] + p0[1]
    ls01 = p0[2] + p0[3]
    ls10 = p1[0] + p1[1]
    ls11 = p1[2] + p1[3]
    ls00 = ls00 + cute.arch.shuffle_sync_bfly(ls00, offset=2)
    ls01 = ls01 + cute.arch.shuffle_sync_bfly(ls01, offset=2)
    ls10 = ls10 + cute.arch.shuffle_sync_bfly(ls10, offset=2)
    ls11 = ls11 + cute.arch.shuffle_sync_bfly(ls11, offset=2)
    ls00 = ls00 + cute.arch.shuffle_sync_bfly(ls00, offset=1)
    ls01 = ls01 + cute.arch.shuffle_sync_bfly(ls01, offset=1)
    ls10 = ls10 + cute.arch.shuffle_sync_bfly(ls10, offset=1)
    ls11 = ls11 + cute.arch.shuffle_sync_bfly(ls11, offset=1)

    gs0[0] = gs0[0] + ls00
    gs0[1] = gs0[1] + ls01
    gs1[0] = gs1[0] + ls10
    gs1[1] = gs1[1] + ls11
    gm0[0] = ngm00
    gm0[1] = ngm01
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
):
    """Final cross-warp row-sum reduction for deferred MG softmax."""
    bar_kw = dict(barrier_id=barrier_id, number_of_threads=num_threads)
    gid = lane >> Int32(2)
    tid = lane & Int32(3)

    if tid == Int32(0):
        st_shared_f32(reduce0_sum_addr + (warp_id * Int32(hpb) + gid) * Int32(4), gs0[0])
        st_shared_f32(
            reduce0_sum_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
            gs0[1],
        )
        st_shared_f32(reduce1_sum_addr + (warp_id * Int32(hpb) + gid) * Int32(4), gs1[0])
        st_shared_f32(
            reduce1_sum_addr + (warp_id * Int32(hpb) + gid + Int32(8)) * Int32(4),
            gs1[1],
        )
    cute.arch.barrier(**bar_kw)

    if tid_flat < Int32(2 * hpb):
        group = tid_flat // Int32(hpb)
        h = tid_flat - group * Int32(hpb)
        rsum = reduce0_sum_addr
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
    gs0[1] = ld_shared_f32(reduce0_sum_addr + (gid + Int32(8)) * Int32(4))
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
):
    """FlashInfer-shaped DSV4 MG XV-NoPE for two HPB head groups.

    V scales and V B operands are shared across the two head groups. This fuses
    the two decode-style S5/S6 calls used by the first correctness port while
    keeping the same DSV4 FP8 W-quantization math.
    """
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
    while i < Int32(2 * n_v_chunks * hpb):
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
        m10 = fmax_f32(fabs_f32(w_pre1[0] * vsc0), fabs_f32(w_pre1[1] * vsc1))
        m11 = fmax_f32(fabs_f32(w_pre1[2] * vsc0), fabs_f32(w_pre1[3] * vsc1))
        vc_base = Int32(vc) * Int32(hpb)
        atomic_max_shared_f32(w_head_sc_base + (vc_base + gid) * Int32(4), m00)
        atomic_max_shared_f32(
            w_head_sc_base + (vc_base + gid + Int32(8)) * Int32(4), m01
        )
        atomic_max_shared_f32(
            w_head_sc_base + (group_sc_elems + vc_base + gid) * Int32(4), m10
        )
        atomic_max_shared_f32(
            w_head_sc_base + (group_sc_elems + vc_base + gid + Int32(8)) * Int32(4),
            m11,
        )
    cute.arch.barrier(**bar_kw)

    i = tid_flat
    while i < Int32(2 * n_v_chunks * hpb):
        w_head_sc_view[i] = fmax_f32(w_head_sc_view[i], Float32(1e-10)) * Float32(
            1.0 / 448.0
        )
        i += Int32(num_threads)
    cute.arch.barrier(**bar_kw)

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
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
        sc10 = w_head_sc_view[group_sc_elems + vc_base + gid]
        sc11 = w_head_sc_view[group_sc_elems + vc_base + gid + Int32(8)]
        si00 = Float32(1.0) / sc00
        si01 = Float32(1.0) / sc01
        si10 = Float32(1.0) / sc10
        si11 = Float32(1.0) / sc11

        vsc0 = _ue8m0_byte_to_fp32(
            _ld_u8_zext(kv_sc_base_addr, cand_e0 * Int32(scale_bytes_per_token) + Int32(vc))
        )
        vsc1 = _ue8m0_byte_to_fp32(
            _ld_u8_zext(kv_sc_base_addr, cand_e1 * Int32(scale_bytes_per_token) + Int32(vc))
        )

        st_shared_u8(
            w_fp8_g0 + gid * Int32(w_fp8_stride) + cand_e0,
            _quant_e4m3_byte(w_pre0[0] * vsc0 * si00).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g0 + gid * Int32(w_fp8_stride) + cand_e1,
            _quant_e4m3_byte(w_pre0[1] * vsc1 * si00).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g0 + (gid + Int32(8)) * Int32(w_fp8_stride) + cand_e0,
            _quant_e4m3_byte(w_pre0[2] * vsc0 * si01).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g0 + (gid + Int32(8)) * Int32(w_fp8_stride) + cand_e1,
            _quant_e4m3_byte(w_pre0[3] * vsc1 * si01).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g1 + gid * Int32(w_fp8_stride) + cand_e0,
            _quant_e4m3_byte(w_pre1[0] * vsc0 * si10).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g1 + gid * Int32(w_fp8_stride) + cand_e1,
            _quant_e4m3_byte(w_pre1[1] * vsc1 * si10).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g1 + (gid + Int32(8)) * Int32(w_fp8_stride) + cand_e0,
            _quant_e4m3_byte(w_pre1[2] * vsc0 * si11).to(cutlass.Uint8),
        )
        st_shared_u8(
            w_fp8_g1 + (gid + Int32(8)) * Int32(w_fp8_stride) + cand_e1,
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
            xv10 = Float32(0.0)
            xv11 = Float32(0.0)
            xv12 = Float32(0.0)
            xv13 = Float32(0.0)
            for kstep in cutlass.range_constexpr(bi // 32):
                ko = Int32(kstep) * Int32(32)
                b0, b1 = _d2_load_b_fp8(
                    kv_fp8_base_addr, ko, dim, lane, kv_smem_stride=kv_smem_stride
                )
                a_addr0 = w_fp8_g0 + a_row * Int32(w_fp8_stride) + ko + a_col
                a00, a01, a02, a03 = ldmatrix_m8n8x4_b16(a_addr0)
                xv00, xv01, xv02, xv03 = mma_m16n8k32_f32_e4m3(
                    xv00, xv01, xv02, xv03, a00, a01, a02, a03, b0, b1
                )
                a_addr1 = w_fp8_g1 + a_row * Int32(w_fp8_stride) + ko + a_col
                a10, a11, a12, a13 = ldmatrix_m8n8x4_b16(a_addr1)
                xv10, xv11, xv12, xv13 = mma_m16n8k32_f32_e4m3(
                    xv10, xv11, xv12, xv13, a10, a11, a12, a13, b0, b1
                )

            acc0[at][0] = acc0[at][0] + xv00 * sc00
            acc0[at][1] = acc0[at][1] + xv01 * sc00
            acc0[at][2] = acc0[at][2] + xv02 * sc01
            acc0[at][3] = acc0[at][3] + xv03 * sc01
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
        block_x, _, _ = cute.arch.block_idx()
        block_x = Int32(block_x)
        rep_h = Int32(self.replicate_h)
        token_idx = block_x // rep_h
        h_tile = block_x - token_idx * rep_h
        head_base = h_tile * Int32(L.heads_per_cta)

        smem = cutlass_utils.SmemAllocator()
        SharedStorage = get_prefill_mg_shared_storage_cls(t)
        st = smem.allocate(SharedStorage)

        q_fp8_addr = shared_ptr_to_u32(st.q_fp8.data_ptr())
        q_rope_addr = shared_ptr_to_u32(st.q_rope.data_ptr())
        kv_fp8_addr = shared_ptr_to_u32(st.kv_fp8.data_ptr())
        kv_sc_addr = shared_ptr_to_u32(st.kv_sc.data_ptr())
        reduce_addr = shared_ptr_to_u32(st.reduce.data_ptr())
        w_fp8_addr = shared_ptr_to_u32(st.w_fp8.data_ptr())
        sm_p_full_addr = shared_ptr_to_u32(st.sm_p_full.data_ptr())

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
        reduce_g0 = reduce_addr
        reduce_g1 = reduce_addr + Int32(L.reduce_group_bytes)
        w_fp8_g0 = w_fp8_addr
        w_fp8_g1 = w_fp8_addr + Int32(L.w_fp8_group_bytes)
        sm_p_g0 = sm_p_full_addr
        sm_p_g1 = sm_p_full_addr + Int32(L.sm_p_full_group_bytes)

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

        topk_row = cute.make_tensor(
            indices.iterator + token_idx.to(Int64) * Int64(self.indices_stride_row),
            cute.make_layout(self.topk),
        )
        q_token = cute.make_tensor(
            q_all.iterator + token_idx.to(Int64) * Int64(self.q_stride_row),
            cute.make_layout(
                (self.num_heads, _DSV4_HEAD_DIM),
                stride=(self.q_stride_head, self.q_stride_dim),
            ),
        )
        warp_first_cand = warp_id * Int32(8)

        if is_io:
            cute.arch.setmaxregister_decrease(_IO_REGS)
            io_lane = tid - Int32(self.math_threads)

            if actual_tiles > Int32(0):
                g_end0 = Int32(_CAND_WINDOW)
                if g_end0 > section_len:
                    g_end0 = section_len
                tok0 = cute.make_tensor(
                    token_idx_view_all.iterator, cute.make_layout(int(L.token_idx_buf_bytes // 4))
                )
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
                    kv_smem_stride=t.kv_smem_stride,
                    io_threads=_PREFILL_IO_THREADS,
                )

            for lc in cutlass.range(actual_tiles, unroll=1):
                next_lc = Int32(lc) + Int32(1)
                if next_lc < actual_tiles:
                    buf = next_lc & Int32(1)
                    g_start = next_lc * Int32(_CAND_WINDOW)
                    g_end = g_start + Int32(_CAND_WINDOW)
                    if g_end > section_len:
                        g_end = section_len
                    tok_buf_view = cute.make_tensor(
                        token_idx_view_all.iterator + buf * tok_buf_elems,
                        cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                    )
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
                        kv_smem_stride=t.kv_smem_stride,
                        io_threads=_PREFILL_IO_THREADS,
                    )
                cute.arch.barrier(barrier_id=1, number_of_threads=self.block_threads)

        else:
            cute.arch.setmaxregister_increase(_MATH_REGS)
            n_acc_tiles = int(t.n_v_chunks) * int(t.nt_per_warp_xv)
            s0_quantize_q_to_smem(
                q_token,
                q_fp8_g0,
                q_sc_g0,
                q_rope_g0,
                amax_view,
                head_base,
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

            if actual_tiles > Int32(0):
                cute.arch.mbarrier_wait(mbar_base, phase=0)

            for lc in cutlass.range(actual_tiles, unroll=1):
                ci = Int32(lc)
                split_cand_start = ci * Int32(_CAND_WINDOW)
                split_cand_end = split_cand_start + Int32(_CAND_WINDOW)
                if split_cand_end > section_len:
                    split_cand_end = section_len
                buf = ci & Int32(1)
                kv_fp8_b = kv_fp8_addr + buf * kv_fp8_buf
                kv_sc_b = kv_sc_addr + buf * kv_sc_buf
                tok_buf_view = cute.make_tensor(
                    token_idx_view_all.iterator + buf * tok_buf_elems,
                    cute.make_layout(int(L.token_idx_buf_bytes // 4)),
                )

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
                    kv_smem_stride=t.kv_smem_stride,
                    scale_bytes_per_token=8,
                    scale_format=t.scale_format,
                )
                qk0 = s2_qk_rope_global_dsv4(
                    qk0,
                    q_rope_g0,
                    kv_cache_u8,
                    tok_buf_view,
                    warp_first_cand,
                    lane,
                    Int32(self.page_block_size),
                    stride_kv_block,
                    d_rope=t.d_rope,
                )
                qk0 = s3_mask_and_scale(
                    qk0,
                    tok_buf_view,
                    warp_first_cand,
                    split_cand_start,
                    split_cand_end,
                    section_len,
                    sm_scale_log2,
                    lane,
                )
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
                    kv_smem_stride=t.kv_smem_stride,
                    scale_bytes_per_token=8,
                    scale_format=t.scale_format,
                )
                qk1 = s2_qk_rope_global_dsv4(
                    qk1,
                    q_rope_g1,
                    kv_cache_u8,
                    tok_buf_view,
                    warp_first_cand,
                    lane,
                    Int32(self.page_block_size),
                    stride_kv_block,
                    d_rope=t.d_rope,
                )
                qk1 = s3_mask_and_scale(
                    qk1,
                    tok_buf_view,
                    warp_first_cand,
                    split_cand_start,
                    split_cand_end,
                    section_len,
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
                )
                w_pre0 = [p0[0] * wr00, p0[1] * wr00, p0[2] * wr01, p0[3] * wr01]
                w_pre1 = [p1[0] * wr10, p1[1] * wr10, p1[2] * wr11, p1[3] * wr11]

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
                    kv_smem_stride=t.kv_smem_stride,
                    w_fp8_stride=t.bi + 16,
                    w_fp8_group_stride=L.w_fp8_group_bytes,
                    w_fp8_parity_stride=L.w_fp8_parity_bytes,
                    n_warps=8,
                    scale_bytes_per_token=8,
                    nt_per_warp_xv=t.nt_per_warp_xv,
                    num_threads=self.math_threads,
                    barrier_id=3,
                )
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
                if next_lc < actual_tiles:
                    next_phase = (next_lc >> Int32(1)) & Int32(1)
                    cute.arch.mbarrier_wait(mbar_base + (next_lc & Int32(1)), phase=next_phase)

            if is_empty_row:
                gsum0_frag[0] = Float32(0.0)
                gsum0_frag[1] = Float32(0.0)
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
            )
            gsum0_frag[0] = final_sum0[0]
            gsum0_frag[1] = final_sum0[1]
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
                    (t.hpb, t.d_v),
                    stride=(self.output_stride_head, self.output_stride_dim),
                ),
            )
            out_lse0 = cute.make_tensor(
                out_lse.iterator
                + token_idx.to(Int64) * Int64(self.out_lse_stride_row)
                + head_base.to(Int64) * Int64(self.out_lse_stride_head),
                cute.make_layout((t.hpb,), stride=(self.out_lse_stride_head,)),
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
                valid_hpb=t.hpb,
                nt_per_warp_xv=t.nt_per_warp_xv,
                v_has_rope=t.v_has_rope,
                epilogue_mode=EPILOGUE_FINAL_BF16,
                has_attn_sink=self.has_sink,
                attn_sink=attn_sink,
                head_base=head_base,
            )
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


def _unified_sm120_prefill_mg_flat_launch(
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
) -> None:
    traits = make_unified_traits(ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE)
    layout = make_smem_layout_mg(traits)
    heads = int(q.shape[1])
    replicate_h = heads // int(layout.heads_per_cta)
    kernel = UnifiedPrefillMGKernel(
        traits,
        layout,
        int(page_block_size),
        int(num_tiles),
        replicate_h=replicate_h,
        num_heads=heads,
        q_stride=tuple(q.stride()),
        indices_stride0=int(topk_indices.stride(0)),
        output_stride=tuple(output.stride()),
        out_lse_stride=tuple(lse_out.stride()),
        has_sink=bool(has_sink),
        topk=int(topk),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _to_cute(q, cutlass.BFloat16, dynamic_layout=True),
        _to_cute(kv_flat, cutlass.Uint8, align=16),
        _to_cute(topk_indices, cutlass.Int32, align=4, dynamic_layout=True),
        _to_cute(topk_length, cutlass.Int32, align=4, dynamic_layout=True),
        _to_cute(attn_sink_t, cutlass.Float32, align=4),
        _to_cute(output, cutlass.BFloat16, align=16, dynamic_layout=True),
        _to_cute(lse_out, cutlass.Float32, align=4, dynamic_layout=True),
        Float32(float(sm_scale) * LOG2_E),
        Int64(stride_kv_block),
        Int32(int(q.shape[0])),
        stream,
    )
    compile_spec = KernelCompileSpec.from_fields(
        "attention.mla.unified_sm120.prefill_mg",
        1,
        key_field("model_type", ModelType.DSV4),
        key_field("compute_mode", ComputeMode.FP8),
        key_field("scale_format", ScaleFormat.UE8M0_BYTE),
        key_field("num_heads", heads),
        key_field("heads_per_cta", int(layout.heads_per_cta)),
        key_field("num_tiles", int(num_tiles)),
        key_field("page_block_size", int(page_block_size)),
        key_field("topk_bucket", _topk_bucket(topk)),
        key_field("has_sink", int(has_sink)),
        tensor_key(
            "q",
            q,
            dims=(DimKey.dynamic(), DimKey.exact(heads), DimKey.exact(_DSV4_HEAD_DIM)),
        ),
        tensor_key("topk_indices", topk_indices, dims=(DimKey.dynamic(), DimKey.bucket(topk))),
        tensor_key(
            "output",
            output,
            dims=(DimKey.dynamic(), DimKey.exact(heads), DimKey.exact(512)),
        ),
        tensor_key("out_lse", lse_out, dims=(DimKey.dynamic(), DimKey.exact(heads))),
    )
    b12x_launch(kernel, compile_spec=compile_spec, compile_args=args, runtime_args=args)


@torch.library.custom_op(
    "b12x::unified_sm120_prefill_mg",
    mutates_args=("output", "lse_out"),
)
def _unified_sm120_prefill_mg_op(
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
) -> None:
    _unified_sm120_prefill_mg_flat_launch(
        q,
        kv_flat,
        topk_indices,
        topk_length,
        attn_sink_t,
        output,
        lse_out,
        sm_scale,
        page_block_size,
        topk,
        num_tiles,
        stride_kv_block,
        has_sink,
    )


@_unified_sm120_prefill_mg_op.register_fake
def _unified_sm120_prefill_mg_fake(
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
):
    from b12x.attention.mla.compressed_reference import compressed_mla_page_nbytes

    if int(q.shape[-1]) != _DSV4_HEAD_DIM:
        raise ValueError("unified_sm120 MG prefill is DSV4-only")
    num_tokens, heads, _ = q.shape
    traits = make_unified_traits(ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE)
    heads_per_cta = 2 * int(traits.hpb)
    if heads % heads_per_cta != 0:
        raise ValueError(
            f"unified_sm120 MG prefill requires heads divisible by {heads_per_cta}, got {heads}"
        )

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
        attn_sink_t = torch.zeros(1, dtype=torch.float32, device=device)

    if stride_kv_block is None:
        stride_kv_block = int(compressed_mla_page_nbytes(int(page_block_size)))

    q = q.contiguous()
    topk_indices = topk_indices.contiguous()
    if output is None:
        output = torch.empty((num_tokens, heads, int(traits.d_v)), dtype=torch.bfloat16, device=device)
    if lse_out is None:
        lse_out = torch.empty((num_tokens, heads), dtype=torch.float32, device=device)

    torch.ops.b12x.unified_sm120_prefill_mg(
        q,
        kv_cache.reshape(-1),
        topk_indices,
        topk_length,
        attn_sink_t,
        output,
        lse_out,
        float(sm_scale),
        int(page_block_size),
        int(topk),
        int(num_tiles),
        int(stride_kv_block),
        bool(has_sink),
    )
    return output, lse_out
