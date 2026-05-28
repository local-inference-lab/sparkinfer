from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cute.runtime import from_dlpack

from b12x.attention._cute import ops as attention_ops
from b12x.attention.mla.reference import _MLA_GROUP_SIZE, dense_mla_reference, pack_mla_kv_cache_reference
from b12x.attention.mla.kernel import (
    _MLA_HEADS_PER_TILE,
    _MLA_NUM_MMA_KV,
    _MLA_NOPE_GROUP_KV_VECS,
    _MLA_NOPE_GROUP_KV_BF16_VECS,
    _MLA_NOPE_GROUP_Q_VECS,
    _MLA_NOPE_QK_NUM_MMA_D,
    _MLA_TOKEN_TILE,
    _MLA_VO_NUM_MMA_D,
    _compute_score_tile_scaled,
    _fill_normalized_p_frag_from_scores,
    _advance_offset_by_column_128b_2,
    _advance_offset_by_row_128b,
    _literal_qk_mma_into_sfrag_mxfp8_raw,
    _literal_pv_mma_into_ofrag_fp8_raw_scaled,
    _literal_pv_mma_into_ofrag_mxfp8_scaled,
    _permuted_offset_128b,
    _smem_addr_from_b128_offset,
    _store_output_group,
    _stage_kv_u32_block,
    _stage_token_scales,
    _stage_q_u32_block,
    _literal_qk_mma_into_sfrag_bf16,
    _zero_score_frag,
    _update_softmax_stats_b2,
    _view_last_dim_as_u32,
    _zero_output_frag,
    bfloat2_mul,
    bfloat2_habs2,
    bfloat2_hmax2,
    bfloat2_hmax_to_f32,
    broadcast_f32_to_bfloat2,
    cvt_f32_to_ue8m0,
    cvt_bf16x2_to_e4m3x2,
    frag_layout_swizzle_16b_to_8b,
    frag_layout_swizzle_16b_to_8b_trans,
    get_sparse_mla_shared_storage_cls,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_trans_left_half_b16,
    ldmatrix_m8n8x4_trans_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_shared_v4_u32,
    ue8m0_to_output_scale,
)
from b12x.cute.fp4 import byte_perm, ldmatrix_m8n8x4_left_half_b16, ldmatrix_m8n8x4_right_half_b16

from .helpers import require_sm120
from .test_attention_mla_reference import _make_glm_case


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


@cute.jit
def _fill_probe_p_frag(
    mP: cute.Tensor,
    p_frag: cute.Tensor,
    lane: Int32,
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    row0 = lane_group
    row1 = lane_group + Int32(8)
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        k_base = Int32(mma_kv * 16) + lane_pair_base
        p_frag[0, mma_kv, 0] = pack_f32x2_to_bfloat2(
            Float32(mP[row0, k_base + Int32(0)]),
            Float32(mP[row0, k_base + Int32(1)]),
        )
        p_frag[0, mma_kv, 1] = pack_f32x2_to_bfloat2(
            Float32(mP[row1, k_base + Int32(0)]),
            Float32(mP[row1, k_base + Int32(1)]),
        )
        p_frag[0, mma_kv, 2] = pack_f32x2_to_bfloat2(
            Float32(mP[row0, k_base + Int32(8)]),
            Float32(mP[row0, k_base + Int32(9)]),
        )
        p_frag[0, mma_kv, 3] = pack_f32x2_to_bfloat2(
            Float32(mP[row1, k_base + Int32(8)]),
            Float32(mP[row1, k_base + Int32(9)]),
        )


@cute.jit
def _load_u8_from_u32_row(src_u32: cute.Tensor, row: Int32, col: Int32) -> Uint32:
    word = Uint32(src_u32[row, col // Int32(4)])
    shift = (col % Int32(4)) * Int32(8)
    return (word >> shift) & Uint32(0xFF)


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp8_probe_swap_k_halves(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    sScale: cute.Tensor,
    lane: Int32,
):
    p_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 4), stride=(8, 4, 1))
    p_swapped = cute.make_rmem_tensor(p_layout, Uint32)
    for reg_id in cutlass.range_constexpr(4):
        p_swapped[0, 0, reg_id] = p_frag[0, 1, reg_id]
        p_swapped[0, 1, reg_id] = p_frag[0, 0, reg_id]
    _literal_pv_mma_into_ofrag_mxfp8_scaled(
        o_frag,
        p_swapped,
        v_base_addr,
        sScale,
        Float32(1.0),
        lane,
    )


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp8_probe_swizzle_a(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    lane: Int32,
):
    unit_scale = Uint32(0x7F7F7F7F)
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    a_regs = cute.make_rmem_tensor(
        cute.make_layout((1, 4), stride=(4, 1)),
        Uint32,
    )
    a_regs[0, 0] = (cvt_bf16x2_to_e4m3x2(p_frag[0, 0, 0]) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(p_frag[0, 1, 0]) & mask16) << shift16
    )
    a_regs[0, 1] = (cvt_bf16x2_to_e4m3x2(p_frag[0, 0, 1]) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(p_frag[0, 1, 1]) & mask16) << shift16
    )
    a_regs[0, 2] = (cvt_bf16x2_to_e4m3x2(p_frag[0, 0, 2]) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(p_frag[0, 1, 2]) & mask16) << shift16
    )
    a_regs[0, 3] = (cvt_bf16x2_to_e4m3x2(p_frag[0, 0, 3]) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(p_frag[0, 1, 3]) & mask16) << shift16
    )
    for reg_id in cutlass.range_constexpr(4):
        a_regs[0, reg_id] = frag_layout_swizzle_16b_to_8b(a_regs[0, reg_id])

    v_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        Int32(_MLA_NOPE_GROUP_KV_VECS),
    )
    v_offset_k0 = v_offset
    v_offset_k1 = _advance_offset_by_row_128b(v_offset, 16, Int32(_MLA_NOPE_GROUP_KV_VECS))
    for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
        b0_k0 = Uint32(0)
        b1_k0 = Uint32(0)
        b0_k1 = Uint32(0)
        b1_k1 = Uint32(0)
        if mma_d % 2 == 0:
            b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_left_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_left_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
            )
        else:
            b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_right_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_right_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
            )
        b0_k0 = frag_layout_swizzle_16b_to_8b_trans(b0_k0)
        b1_k0 = frag_layout_swizzle_16b_to_8b_trans(b1_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b_trans(b0_k1)
        b1_k1 = frag_layout_swizzle_16b_to_8b_trans(b1_k1)

        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            o_frag[0, mma_d, 0],
            o_frag[0, mma_d, 1],
            o_frag[0, mma_d, 2],
            o_frag[0, mma_d, 3],
            a_regs[0, 0],
            a_regs[0, 1],
            a_regs[0, 2],
            a_regs[0, 3],
            b0_k0,
            b0_k1,
            unit_scale,
            unit_scale,
        )
        d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
            o_frag[0, mma_d, 4],
            o_frag[0, mma_d, 5],
            o_frag[0, mma_d, 6],
            o_frag[0, mma_d, 7],
            a_regs[0, 0],
            a_regs[0, 1],
            a_regs[0, 2],
            a_regs[0, 3],
            b1_k0,
            b1_k1,
            unit_scale,
            unit_scale,
        )
        o_frag[0, mma_d, 0] = d0
        o_frag[0, mma_d, 1] = d1
        o_frag[0, mma_d, 2] = d2
        o_frag[0, mma_d, 3] = d3
        o_frag[0, mma_d, 4] = d4
        o_frag[0, mma_d, 5] = d5
        o_frag[0, mma_d, 6] = d6
        o_frag[0, mma_d, 7] = d7
        if mma_d % 2 == 1:
            v_offset_k0 = _advance_offset_by_column_128b_2(v_offset_k0, mma_d // 2)
            v_offset_k1 = _advance_offset_by_column_128b_2(v_offset_k1, mma_d // 2)


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp8_probe_interleave_b(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    sScale: cute.Tensor,
    lane: Int32,
):
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    lane_pair_base = Int32(2) * (lane % Int32(4))

    scale01_k0 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(0)]), Float32(sScale[lane_pair_base + Int32(1)]))
    scale89_k0 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(8)]), Float32(sScale[lane_pair_base + Int32(9)]))
    scale01_k1 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(16)]), Float32(sScale[lane_pair_base + Int32(17)]))
    scale89_k1 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(24)]), Float32(sScale[lane_pair_base + Int32(25)]))
    sfa01_k0 = cvt_f32_to_ue8m0(
        attention_ops.fmax(Float32(sScale[lane_pair_base + Int32(0)]), Float32(sScale[lane_pair_base + Int32(1)]))
    )
    sfa89_k0 = cvt_f32_to_ue8m0(
        attention_ops.fmax(Float32(sScale[lane_pair_base + Int32(8)]), Float32(sScale[lane_pair_base + Int32(9)]))
    )
    sfa01_k1 = cvt_f32_to_ue8m0(
        attention_ops.fmax(Float32(sScale[lane_pair_base + Int32(16)]), Float32(sScale[lane_pair_base + Int32(17)]))
    )
    sfa89_k1 = cvt_f32_to_ue8m0(
        attention_ops.fmax(Float32(sScale[lane_pair_base + Int32(24)]), Float32(sScale[lane_pair_base + Int32(25)]))
    )
    inv_sfa01_k0 = broadcast_f32_to_bfloat2(ue8m0_to_output_scale(sfa01_k0))
    inv_sfa89_k0 = broadcast_f32_to_bfloat2(ue8m0_to_output_scale(sfa89_k0))
    inv_sfa01_k1 = broadcast_f32_to_bfloat2(ue8m0_to_output_scale(sfa01_k1))
    inv_sfa89_k1 = broadcast_f32_to_bfloat2(ue8m0_to_output_scale(sfa89_k1))
    scale01_k0 = bfloat2_mul(scale01_k0, inv_sfa01_k0)
    scale89_k0 = bfloat2_mul(scale89_k0, inv_sfa89_k0)
    scale01_k1 = bfloat2_mul(scale01_k1, inv_sfa01_k1)
    scale89_k1 = bfloat2_mul(scale89_k1, inv_sfa89_k1)
    sfa = Uint32(
        sfa01_k0
        | (sfa89_k0 << Uint32(8))
        | (sfa01_k1 << Uint32(16))
        | (sfa89_k1 << Uint32(24))
    )
    unit_scale = Uint32(0x7F7F7F7F)

    a_regs = cute.make_rmem_tensor(
        cute.make_layout((1, 4), stride=(4, 1)),
        Uint32,
    )
    a_regs[0, 0] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 0, 0], scale01_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 1, 0], scale01_k1)) & mask16) << shift16
    )
    a_regs[0, 1] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 0, 1], scale01_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 1, 1], scale01_k1)) & mask16) << shift16
    )
    a_regs[0, 2] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 0, 2], scale89_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 1, 2], scale89_k1)) & mask16) << shift16
    )
    a_regs[0, 3] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 0, 3], scale89_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 1, 3], scale89_k1)) & mask16) << shift16
    )

    v_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        Int32(_MLA_NOPE_GROUP_KV_VECS),
    )
    v_offset_k0 = v_offset
    v_offset_k1 = _advance_offset_by_row_128b(v_offset, 16, Int32(_MLA_NOPE_GROUP_KV_VECS))
    for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
        b0_k0 = Uint32(0)
        b1_k0 = Uint32(0)
        b0_k1 = Uint32(0)
        b1_k1 = Uint32(0)
        if mma_d % 2 == 0:
            b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_left_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_left_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
            )
        else:
            b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_right_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_right_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
            )
        b0_k0 = frag_layout_swizzle_16b_to_8b_trans(b0_k0)
        b1_k0 = frag_layout_swizzle_16b_to_8b_trans(b1_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b_trans(b0_k1)
        b1_k1 = frag_layout_swizzle_16b_to_8b_trans(b1_k1)
        b01_k0 = byte_perm(b0_k0, b1_k0, Int32(0x5410))
        b23_k0 = byte_perm(b0_k0, b1_k0, Int32(0x7632))
        b01_k1 = byte_perm(b0_k1, b1_k1, Int32(0x5410))
        b23_k1 = byte_perm(b0_k1, b1_k1, Int32(0x7632))

        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            o_frag[0, mma_d, 0],
            o_frag[0, mma_d, 1],
            o_frag[0, mma_d, 2],
            o_frag[0, mma_d, 3],
            a_regs[0, 0],
            a_regs[0, 1],
            a_regs[0, 2],
            a_regs[0, 3],
            b01_k0,
            b01_k1,
            sfa,
            unit_scale,
        )
        d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
            o_frag[0, mma_d, 4],
            o_frag[0, mma_d, 5],
            o_frag[0, mma_d, 6],
            o_frag[0, mma_d, 7],
            a_regs[0, 0],
            a_regs[0, 1],
            a_regs[0, 2],
            a_regs[0, 3],
            b23_k0,
            b23_k1,
            sfa,
            unit_scale,
        )
        o_frag[0, mma_d, 0] = d0
        o_frag[0, mma_d, 1] = d1
        o_frag[0, mma_d, 2] = d2
        o_frag[0, mma_d, 3] = d3
        o_frag[0, mma_d, 4] = d4
        o_frag[0, mma_d, 5] = d5
        o_frag[0, mma_d, 6] = d6
        o_frag[0, mma_d, 7] = d7
        if mma_d % 2 == 1:
            v_offset_k0 = _advance_offset_by_column_128b_2(v_offset_k0, mma_d // 2)
            v_offset_k1 = _advance_offset_by_column_128b_2(v_offset_k1, mma_d // 2)


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp8_probe_repack_a_interleave_b(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    sScale: cute.Tensor,
    lane: Int32,
):
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    lane_pair_base = Int32(2) * (lane % Int32(4))

    scale01_k0 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(0)]), Float32(sScale[lane_pair_base + Int32(1)]))
    scale89_k0 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(8)]), Float32(sScale[lane_pair_base + Int32(9)]))
    scale01_k1 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(16)]), Float32(sScale[lane_pair_base + Int32(17)]))
    scale89_k1 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(24)]), Float32(sScale[lane_pair_base + Int32(25)]))
    a00 = bfloat2_mul(p_frag[0, 0, 0], scale01_k0)
    a10 = bfloat2_mul(p_frag[0, 0, 1], scale01_k0)
    a80 = bfloat2_mul(p_frag[0, 0, 2], scale89_k0)
    a90 = bfloat2_mul(p_frag[0, 0, 3], scale89_k0)
    a16 = bfloat2_mul(p_frag[0, 1, 0], scale01_k1)
    a17 = bfloat2_mul(p_frag[0, 1, 1], scale01_k1)
    a24 = bfloat2_mul(p_frag[0, 1, 2], scale89_k1)
    a25 = bfloat2_mul(p_frag[0, 1, 3], scale89_k1)
    sfa_k0 = cvt_f32_to_ue8m0(
        bfloat2_hmax_to_f32(
            bfloat2_hmax2(
                bfloat2_hmax2(
                    bfloat2_habs2(a00),
                    bfloat2_habs2(a10),
                ),
                bfloat2_hmax2(
                    bfloat2_habs2(a80),
                    bfloat2_habs2(a90),
                ),
            )
        )
    )
    sfa_k1 = cvt_f32_to_ue8m0(
        bfloat2_hmax_to_f32(
            bfloat2_hmax2(
                bfloat2_hmax2(
                    bfloat2_habs2(a16),
                    bfloat2_habs2(a17),
                ),
                bfloat2_hmax2(
                    bfloat2_habs2(a24),
                    bfloat2_habs2(a25),
                ),
            )
        )
    )
    inv_sfa_k0 = broadcast_f32_to_bfloat2(ue8m0_to_output_scale(sfa_k0))
    inv_sfa_k1 = broadcast_f32_to_bfloat2(ue8m0_to_output_scale(sfa_k1))
    sfa = Uint32(sfa_k0 | (sfa_k0 << Uint32(8)) | (sfa_k1 << Uint32(16)) | (sfa_k1 << Uint32(24)))
    unit_scale = Uint32(0x7F7F7F7F)

    a_regs = cute.make_rmem_tensor(
        cute.make_layout((1, 4), stride=(4, 1)),
        Uint32,
    )
    a_regs[0, 0] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a00, inv_sfa_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a80, inv_sfa_k0)) & mask16) << shift16
    )
    a_regs[0, 1] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a10, inv_sfa_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a90, inv_sfa_k0)) & mask16) << shift16
    )
    a_regs[0, 2] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a16, inv_sfa_k1)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a24, inv_sfa_k1)) & mask16) << shift16
    )
    a_regs[0, 3] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a17, inv_sfa_k1)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a25, inv_sfa_k1)) & mask16) << shift16
    )

    v_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        Int32(_MLA_NOPE_GROUP_KV_VECS),
    )
    v_offset_k0 = v_offset
    v_offset_k1 = _advance_offset_by_row_128b(v_offset, 16, Int32(_MLA_NOPE_GROUP_KV_VECS))
    for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
        b0_k0 = Uint32(0)
        b1_k0 = Uint32(0)
        b0_k1 = Uint32(0)
        b1_k1 = Uint32(0)
        if mma_d % 2 == 0:
            b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_left_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_left_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
            )
        else:
            b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_right_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_right_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
            )
        b0_k0 = frag_layout_swizzle_16b_to_8b_trans(b0_k0)
        b1_k0 = frag_layout_swizzle_16b_to_8b_trans(b1_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b_trans(b0_k1)
        b1_k1 = frag_layout_swizzle_16b_to_8b_trans(b1_k1)
        b01_k0 = byte_perm(b0_k0, b1_k0, Int32(0x5410))
        b23_k0 = byte_perm(b0_k0, b1_k0, Int32(0x7632))
        b01_k1 = byte_perm(b0_k1, b1_k1, Int32(0x5410))
        b23_k1 = byte_perm(b0_k1, b1_k1, Int32(0x7632))

        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            o_frag[0, mma_d, 0],
            o_frag[0, mma_d, 1],
            o_frag[0, mma_d, 2],
            o_frag[0, mma_d, 3],
            a_regs[0, 0],
            a_regs[0, 1],
            a_regs[0, 2],
            a_regs[0, 3],
            b01_k0,
            b01_k1,
            sfa,
            unit_scale,
        )
        d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
            o_frag[0, mma_d, 4],
            o_frag[0, mma_d, 5],
            o_frag[0, mma_d, 6],
            o_frag[0, mma_d, 7],
            a_regs[0, 0],
            a_regs[0, 1],
            a_regs[0, 2],
            a_regs[0, 3],
            b23_k0,
            b23_k1,
            sfa,
            unit_scale,
        )
        o_frag[0, mma_d, 0] = d0
        o_frag[0, mma_d, 1] = d1
        o_frag[0, mma_d, 2] = d2
        o_frag[0, mma_d, 3] = d3
        o_frag[0, mma_d, 4] = d4
        o_frag[0, mma_d, 5] = d5
        o_frag[0, mma_d, 6] = d6
        o_frag[0, mma_d, 7] = d7
        if mma_d % 2 == 1:
            v_offset_k0 = _advance_offset_by_column_128b_2(v_offset_k0, mma_d // 2)
            v_offset_k1 = _advance_offset_by_column_128b_2(v_offset_k1, mma_d // 2)


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp8_probe_bscale(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    sScale: cute.Tensor,
    lane: Int32,
    *,
    bid_b: int = 0,
    tid_b: int = 0,
):
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    unit_scale = Uint32(0x7F)

    a_regs = cute.make_rmem_tensor(cute.make_layout((1, 4), stride=(4, 1)), Uint32)
    a_regs[0, 0] = (cvt_bf16x2_to_e4m3x2(p_frag[0, 0, 0]) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(p_frag[0, 0, 2]) & mask16) << shift16
    )
    a_regs[0, 1] = (cvt_bf16x2_to_e4m3x2(p_frag[0, 0, 1]) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(p_frag[0, 0, 3]) & mask16) << shift16
    )
    a_regs[0, 2] = (cvt_bf16x2_to_e4m3x2(p_frag[0, 1, 0]) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(p_frag[0, 1, 2]) & mask16) << shift16
    )
    a_regs[0, 3] = (cvt_bf16x2_to_e4m3x2(p_frag[0, 1, 1]) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(p_frag[0, 1, 3]) & mask16) << shift16
    )

    quad = lane // Int32(4)
    sfb = Uint32(0)
    if lane % Int32(4) == Int32(tid_b):
        base = quad * Int32(4)
        sb0 = Uint32(cvt_f32_to_ue8m0(Float32(sScale[base + Int32(0)])))
        sb1 = Uint32(cvt_f32_to_ue8m0(Float32(sScale[base + Int32(1)])))
        sb2 = Uint32(cvt_f32_to_ue8m0(Float32(sScale[base + Int32(2)])))
        sb3 = Uint32(cvt_f32_to_ue8m0(Float32(sScale[base + Int32(3)])))
        sfb = sb0 | (sb1 << Uint32(8)) | (sb2 << Uint32(16)) | (sb3 << Uint32(24))

    v_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        Int32(_MLA_NOPE_GROUP_KV_VECS),
    )
    v_offset_k0 = v_offset
    v_offset_k1 = _advance_offset_by_row_128b(v_offset, 16, Int32(_MLA_NOPE_GROUP_KV_VECS))
    for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
        b0_k0 = Uint32(0)
        b1_k0 = Uint32(0)
        b0_k1 = Uint32(0)
        b1_k1 = Uint32(0)
        if mma_d % 2 == 0:
            b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_left_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_left_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
            )
        else:
            b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_right_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_right_half_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
            )
        b0_k0 = frag_layout_swizzle_16b_to_8b_trans(b0_k0)
        b1_k0 = frag_layout_swizzle_16b_to_8b_trans(b1_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b_trans(b0_k1)
        b1_k1 = frag_layout_swizzle_16b_to_8b_trans(b1_k1)
        b01_k0 = byte_perm(b0_k0, b1_k0, Int32(0x5410))
        b23_k0 = byte_perm(b0_k0, b1_k0, Int32(0x7632))
        b01_k1 = byte_perm(b0_k1, b1_k1, Int32(0x5410))
        b23_k1 = byte_perm(b0_k1, b1_k1, Int32(0x7632))

        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            o_frag[0, mma_d, 0],
            o_frag[0, mma_d, 1],
            o_frag[0, mma_d, 2],
            o_frag[0, mma_d, 3],
            a_regs[0, 0],
            a_regs[0, 1],
            a_regs[0, 2],
            a_regs[0, 3],
            b01_k0,
            b01_k1,
            unit_scale,
            sfb,
            bid_b=bid_b,
            tid_b=tid_b,
        )
        d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
            o_frag[0, mma_d, 4],
            o_frag[0, mma_d, 5],
            o_frag[0, mma_d, 6],
            o_frag[0, mma_d, 7],
            a_regs[0, 0],
            a_regs[0, 1],
            a_regs[0, 2],
            a_regs[0, 3],
            b23_k0,
            b23_k1,
            unit_scale,
            sfb,
            bid_b=bid_b,
            tid_b=tid_b,
        )
        o_frag[0, mma_d, 0] = d0
        o_frag[0, mma_d, 1] = d1
        o_frag[0, mma_d, 2] = d2
        o_frag[0, mma_d, 3] = d3
        o_frag[0, mma_d, 4] = d4
        o_frag[0, mma_d, 5] = d5
        o_frag[0, mma_d, 6] = d6
        o_frag[0, mma_d, 7] = d7
        if mma_d % 2 == 1:
            v_offset_k0 = _advance_offset_by_column_128b_2(v_offset_k0, mma_d // 2)
            v_offset_k1 = _advance_offset_by_column_128b_2(v_offset_k1, mma_d // 2)


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp8_probe_grouped_a4(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    sScale: cute.Tensor,
    lane: Int32,
):
    p_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 4), stride=(8, 4, 1))
    for group_idx in cutlass.range_constexpr(4):
        p_group = cute.make_rmem_tensor(p_layout, Uint32)
        for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
            for reg_id in cutlass.range_constexpr(4):
                p_group[0, mma_kv, reg_id] = Uint32(0)
        if group_idx == 0:
            p_group[0, 0, 0] = p_frag[0, 0, 0]
            p_group[0, 0, 1] = p_frag[0, 0, 1]
        elif group_idx == 1:
            p_group[0, 0, 2] = p_frag[0, 0, 2]
            p_group[0, 0, 3] = p_frag[0, 0, 3]
        elif group_idx == 2:
            p_group[0, 1, 0] = p_frag[0, 1, 0]
            p_group[0, 1, 1] = p_frag[0, 1, 1]
        else:
            p_group[0, 1, 2] = p_frag[0, 1, 2]
            p_group[0, 1, 3] = p_frag[0, 1, 3]
        _literal_pv_mma_into_ofrag_mxfp8_probe_repack_a_interleave_b(
            o_frag,
            p_group,
            v_base_addr,
            sScale,
            lane,
        )


class TinyMlaPvProbeKernel:
    num_threads = 32

    def __init__(
        self,
        *,
        use_mxfp8: bool,
        swap_k_halves: bool = False,
        swizzle_a: bool = False,
        interleave_b: bool = False,
        repack_a_interleave_b: bool = False,
        bscale_metadata: bool = False,
        bid_b: int = 0,
        tid_b: int = 0,
        grouped_a4: bool = False,
    ):
        self.use_mxfp8 = bool(use_mxfp8)
        self.swap_k_halves = bool(swap_k_halves)
        self.swizzle_a = bool(swizzle_a)
        self.interleave_b = bool(interleave_b)
        self.repack_a_interleave_b = bool(repack_a_interleave_b)
        self.bscale_metadata = bool(bscale_metadata)
        self.bid_b = int(bid_b)
        self.tid_b = int(tid_b)
        self.grouped_a4 = bool(grouped_a4)

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mVWords: cute.Tensor,
        mScale: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(mP.element_type != cutlass.BFloat16):
            raise TypeError("P must be BFloat16")
        if const_expr(mVWords.element_type != cutlass.Uint32):
            raise TypeError("V words must be Uint32")
        if const_expr(mScale.element_type != cutlass.Float32):
            raise TypeError("scale must be Float32")
        if const_expr(mOut.element_type != cutlass.Float32):
            raise TypeError("out must be Float32")
        if const_expr(mP.shape != (_MLA_HEADS_PER_TILE, _MLA_TOKEN_TILE)):
            raise ValueError(f"P must have shape ({_MLA_HEADS_PER_TILE}, {_MLA_TOKEN_TILE})")
        if const_expr(mVWords.shape != (_MLA_TOKEN_TILE, _MLA_NOPE_GROUP_KV_VECS * 4)):
            raise ValueError("V words must have shape (32, 32)")
        if const_expr(mScale.shape != (_MLA_TOKEN_TILE,)):
            raise ValueError("scale must have shape (32,)")
        if const_expr(mOut.shape != (1, _MLA_HEADS_PER_TILE, _MLA_VO_NUM_MMA_D * 16)):
            raise ValueError("out must have shape (1, 16, 128)")

        self.kernel(mP, mVWords, mScale, mOut).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mVWords: cute.Tensor,
        mScale: cute.Tensor,
        mOut: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = get_sparse_mla_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        sTokenIdx = storage.token_idx.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        sScale = storage.token_scale_a.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        kv_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

        token_local = lane
        while token_local < Int32(_MLA_TOKEN_TILE):
            sTokenIdx[token_local] = token_local
            sScale[token_local] = Float32(mScale[token_local])
            token_local += Int32(self.num_threads)
        cute.arch.sync_threads()

        _stage_kv_u32_block(
            mVWords,
            sTokenIdx,
            Int32(0),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            kv_base_addr,
            Int32(mVWords.shape[0]),
            lane,
        )
        cute.arch.sync_threads()

        p_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 4), stride=(8, 4, 1))
        o_layout = cute.make_layout((1, _MLA_VO_NUM_MMA_D, 8), stride=(_MLA_VO_NUM_MMA_D * 8, 8, 1))
        md_layout = cute.make_layout((1, 2), stride=(2, 1))

        p_frag = cute.make_rmem_tensor(p_layout, Uint32)
        _fill_probe_p_frag(mP, p_frag, lane)

        o_frag = cute.make_rmem_tensor(o_layout, Float32)
        for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
            for reg_id in cutlass.range_constexpr(8):
                o_frag[0, mma_d, reg_id] = Float32(0.0)

        if const_expr(self.use_mxfp8):
            if const_expr(self.swizzle_a):
                _literal_pv_mma_into_ofrag_mxfp8_probe_swizzle_a(
                    o_frag,
                    p_frag,
                    kv_base_addr,
                    lane,
                )
            elif const_expr(self.bscale_metadata):
                _literal_pv_mma_into_ofrag_mxfp8_probe_bscale(
                    o_frag,
                    p_frag,
                    kv_base_addr,
                    sScale,
                    lane,
                    bid_b=self.bid_b,
                    tid_b=self.tid_b,
                )
            elif const_expr(self.grouped_a4):
                _literal_pv_mma_into_ofrag_mxfp8_probe_grouped_a4(
                    o_frag,
                    p_frag,
                    kv_base_addr,
                    sScale,
                    lane,
                )
            elif const_expr(self.repack_a_interleave_b):
                _literal_pv_mma_into_ofrag_mxfp8_probe_repack_a_interleave_b(
                    o_frag,
                    p_frag,
                    kv_base_addr,
                    sScale,
                    lane,
                )
            elif const_expr(self.interleave_b):
                _literal_pv_mma_into_ofrag_mxfp8_probe_interleave_b(
                    o_frag,
                    p_frag,
                    kv_base_addr,
                    sScale,
                    lane,
                )
            elif const_expr(self.swap_k_halves):
                _literal_pv_mma_into_ofrag_mxfp8_probe_swap_k_halves(
                    o_frag,
                    p_frag,
                    kv_base_addr,
                    sScale,
                    lane,
                )
            else:
                _literal_pv_mma_into_ofrag_mxfp8_scaled(
                    o_frag,
                    p_frag,
                    kv_base_addr,
                    sScale,
                    Float32(1.0),
                    lane,
                )
        else:
            _literal_pv_mma_into_ofrag_fp8_raw_scaled(
                o_frag,
                p_frag,
                kv_base_addr,
                sScale,
                Int32(0),
                lane,
            )

        d_frag = cute.make_rmem_tensor(md_layout, Float32)
        d_frag[0, 0] = Float32(1.0)
        d_frag[0, 1] = Float32(1.0)
        _store_output_group(
            mOut,
            o_frag,
            d_frag,
            Int32(0),
            Int32(0),
            Int32(0),
            lane,
        )


def _run_probe(
    *,
    p: torch.Tensor,
    v_src: torch.Tensor,
    scales: torch.Tensor,
    use_mxfp8: bool,
    swap_k_halves: bool = False,
    swizzle_a: bool = False,
    interleave_b: bool = False,
    repack_a_interleave_b: bool = False,
    bscale_metadata: bool = False,
    bid_b: int = 0,
    tid_b: int = 0,
    grouped_a4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    require_sm120()
    device = p.device
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    v_fp8 = (v_src / scales.unsqueeze(1)).clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
    v_words = v_fp8.contiguous().view(torch.uint8).view(torch.uint32)
    out = torch.empty((1, 16, 128), device=device, dtype=torch.float32)

    kernel = TinyMlaPvProbeKernel(
        use_mxfp8=use_mxfp8,
        swap_k_halves=swap_k_halves,
        swizzle_a=swizzle_a,
        interleave_b=interleave_b,
        repack_a_interleave_b=repack_a_interleave_b,
        bscale_metadata=bscale_metadata,
        bid_b=bid_b,
        tid_b=tid_b,
        grouped_a4=grouped_a4,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(p, cutlass.BFloat16),
        _to_cute_tensor(v_words, cutlass.Uint32),
        _to_cute_tensor(scales, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled(
        _to_cute_tensor(p, cutlass.BFloat16),
        _to_cute_tensor(v_words, cutlass.Uint32),
        _to_cute_tensor(scales, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize(device)
    ref = torch.matmul(p.to(torch.float32), v_fp8.to(torch.float32) * scales.unsqueeze(1))
    return out.squeeze(0), ref


class LiveMlaPvProbeKernel:
    num_threads = 32

    def __init__(self, *, use_mxfp8: bool):
        self.use_mxfp8 = bool(use_mxfp8)

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        sm_scale: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_u32,
            kv_rows_u32,
            kv_scales,
            page_table_1,
            sm_scale,
            mOut,
        ).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        sm_scale: cute.Tensor,
        mOut: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        SharedStorage = get_sparse_mla_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        sTokenIdx = storage.token_idx.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        sScale = storage.token_scale_a.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())
        kv_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

        frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))
        p_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 4), stride=(8, 4, 1))
        o_layout = cute.make_layout((1, _MLA_VO_NUM_MMA_D, 8), stride=(_MLA_VO_NUM_MMA_D * 8, 8, 1))
        md_layout = cute.make_layout((1, 2), stride=(2, 1))

        score_frag = cute.make_rmem_tensor(frag_layout, Float32)
        _compute_score_tile_scaled(
            score_frag,
            q_u32,
            kv_rows_u32,
            kv_scales,
            page_table_1,
            sTokenIdx,
            sScale,
            q_base_addr,
            kv_base_addr,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(page_table_1.shape[1]),
            Float32(sm_scale[Int32(0)] * attention_ops.LOG2_E),
            lane,
            False,
        )

        m_frag = cute.make_rmem_tensor(md_layout, Float32)
        d_frag = cute.make_rmem_tensor(md_layout, Float32)
        o_rescale_frag = cute.make_rmem_tensor(md_layout, Float32)
        for row_slot in cutlass.range_constexpr(2):
            m_frag[0, row_slot] = Float32(-Float32.inf)
            d_frag[0, row_slot] = Float32(0.0)
            o_rescale_frag[0, row_slot] = Float32(1.0)
        _update_softmax_stats_b2(score_frag, m_frag, d_frag, o_rescale_frag)

        p_frag = cute.make_rmem_tensor(p_layout, Uint32)
        _fill_normalized_p_frag_from_scores(p_frag, score_frag, m_frag, d_frag)

        _stage_token_scales(kv_scales, sTokenIdx, sScale, Int32(0), Int32(kv_rows_u32.shape[0]), lane)
        _stage_kv_u32_block(
            kv_rows_u32,
            sTokenIdx,
            Int32(0),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            kv_base_addr,
            Int32(kv_rows_u32.shape[0]),
            lane,
        )
        cute.arch.sync_threads()

        o_frag = cute.make_rmem_tensor(o_layout, Float32)
        _zero_output_frag(o_frag)
        if const_expr(self.use_mxfp8):
            _literal_pv_mma_into_ofrag_mxfp8_scaled(
                o_frag,
                p_frag,
                kv_base_addr,
                sScale,
                Float32(1.0),
                lane,
            )
        else:
            _literal_pv_mma_into_ofrag_fp8_raw_scaled(
                o_frag,
                p_frag,
                kv_base_addr,
                sScale,
                Int32(0),
                lane,
            )
        _store_output_group(
            mOut,
            o_frag,
            d_frag,
            Int32(0),
            Int32(0),
            Int32(0),
            lane,
        )


def _run_live_fragment_probe(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    sm_scale: float,
    use_mxfp8: bool,
) -> torch.Tensor:
    kv_rows_bytes = kv_cache[:, 0, :].view(torch.uint8)
    kv_scales = kv_rows_bytes[:, 512 : 512 + 16].view(torch.float32)
    q_u32 = _view_last_dim_as_u32(q_all)
    kv_rows_u32 = _view_last_dim_as_u32(kv_rows_bytes)
    sm_scale_tensor = torch.tensor([sm_scale], device=q_all.device, dtype=torch.float32)
    out = torch.empty((1, 16, _MLA_GROUP_SIZE), device=q_all.device, dtype=torch.float32)

    kernel = LiveMlaPvProbeKernel(use_mxfp8=use_mxfp8)
    stream = cuda.CUstream(torch.cuda.current_stream(device=q_all.device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(kv_rows_u32, cutlass.Uint32),
        _to_cute_tensor(kv_scales, cutlass.Float32),
        _to_cute_tensor(page_table_1, cutlass.Int32),
        _to_cute_tensor(sm_scale_tensor, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled(
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(kv_rows_u32, cutlass.Uint32),
        _to_cute_tensor(kv_scales, cutlass.Float32),
        _to_cute_tensor(page_table_1, cutlass.Int32),
        _to_cute_tensor(sm_scale_tensor, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize(q_all.device)
    return out


@cute.jit
def _pack_live_mxfp8_a_regs_for_dump(
    a_dump: cute.Tensor,
    sfa_dump: cute.Tensor,
    p_frag: cute.Tensor,
    sScale: cute.Tensor,
    lane: Int32,
):
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    lane_pair_base = Int32(2) * (lane % Int32(4))

    scale01_k0 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(0)]), Float32(sScale[lane_pair_base + Int32(1)]))
    scale89_k0 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(8)]), Float32(sScale[lane_pair_base + Int32(9)]))
    scale01_k1 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(16)]), Float32(sScale[lane_pair_base + Int32(17)]))
    scale89_k1 = pack_f32x2_to_bfloat2(Float32(sScale[lane_pair_base + Int32(24)]), Float32(sScale[lane_pair_base + Int32(25)]))
    scale01_max = attention_ops.fmax(
        attention_ops.fmax(Float32(sScale[lane_pair_base + Int32(0)]), Float32(sScale[lane_pair_base + Int32(1)])),
        attention_ops.fmax(Float32(sScale[lane_pair_base + Int32(16)]), Float32(sScale[lane_pair_base + Int32(17)])),
    )
    scale89_max = attention_ops.fmax(
        attention_ops.fmax(Float32(sScale[lane_pair_base + Int32(8)]), Float32(sScale[lane_pair_base + Int32(9)])),
        attention_ops.fmax(Float32(sScale[lane_pair_base + Int32(24)]), Float32(sScale[lane_pair_base + Int32(25)])),
    )
    sfa01 = cvt_f32_to_ue8m0(scale01_max)
    sfa89 = cvt_f32_to_ue8m0(scale89_max)
    inv_sfa01 = broadcast_f32_to_bfloat2(ue8m0_to_output_scale(sfa01))
    inv_sfa89 = broadcast_f32_to_bfloat2(ue8m0_to_output_scale(sfa89))
    scale01_k0 = bfloat2_mul(scale01_k0, inv_sfa01)
    scale89_k0 = bfloat2_mul(scale89_k0, inv_sfa89)
    scale01_k1 = bfloat2_mul(scale01_k1, inv_sfa01)
    scale89_k1 = bfloat2_mul(scale89_k1, inv_sfa89)
    sfa = Uint32(sfa01 | (sfa01 << Uint32(8)) | (sfa89 << Uint32(16)) | (sfa89 << Uint32(24)))

    a_dump[lane, 0] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 0, 0], scale01_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 1, 0], scale01_k1)) & mask16) << shift16
    )
    a_dump[lane, 1] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 0, 1], scale01_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 1, 1], scale01_k1)) & mask16) << shift16
    )
    a_dump[lane, 2] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 0, 2], scale89_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 1, 2], scale89_k1)) & mask16) << shift16
    )
    a_dump[lane, 3] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 0, 3], scale89_k0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[0, 1, 3], scale89_k1)) & mask16) << shift16
    )
    sfa_dump[lane] = sfa


@cute.jit
def _dump_live_mxfp8_b_regs_mma_d0(
    b_dump: cute.Tensor,
    v_base_addr: Int32,
    lane: Int32,
):
    v_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        Int32(_MLA_NOPE_GROUP_KV_VECS),
    )
    v_offset_k0 = v_offset
    v_offset_k1 = _advance_offset_by_row_128b(v_offset, 16, Int32(_MLA_NOPE_GROUP_KV_VECS))
    b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_left_half_b16(
        _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
    )
    b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_left_half_b16(
        _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
    )
    b_dump[lane, 0] = frag_layout_swizzle_16b_to_8b_trans(b0_k0)
    b_dump[lane, 1] = frag_layout_swizzle_16b_to_8b_trans(b1_k0)
    b_dump[lane, 2] = frag_layout_swizzle_16b_to_8b_trans(b0_k1)
    b_dump[lane, 3] = frag_layout_swizzle_16b_to_8b_trans(b1_k1)


class LiveMlaPvRegisterDumpKernel:
    num_threads = 32

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        sm_scale: cute.Tensor,
        p_dump: cute.Tensor,
        a_dump: cute.Tensor,
        sfa_dump: cute.Tensor,
        b_dump: cute.Tensor,
        scale_dump: cute.Tensor,
        token_idx_dump: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_u32,
            kv_rows_u32,
            kv_scales,
            page_table_1,
            sm_scale,
            p_dump,
            a_dump,
            sfa_dump,
            b_dump,
            scale_dump,
            token_idx_dump,
        ).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        sm_scale: cute.Tensor,
        p_dump: cute.Tensor,
        a_dump: cute.Tensor,
        sfa_dump: cute.Tensor,
        b_dump: cute.Tensor,
        scale_dump: cute.Tensor,
        token_idx_dump: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        SharedStorage = get_sparse_mla_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        sTokenIdx = storage.token_idx.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        sScale = storage.token_scale_a.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())
        kv_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

        frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))
        p_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 4), stride=(8, 4, 1))
        md_layout = cute.make_layout((1, 2), stride=(2, 1))

        score_frag = cute.make_rmem_tensor(frag_layout, Float32)
        _compute_score_tile_scaled(
            score_frag,
            q_u32,
            kv_rows_u32,
            kv_scales,
            page_table_1,
            sTokenIdx,
            sScale,
            q_base_addr,
            kv_base_addr,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(page_table_1.shape[1]),
            Float32(sm_scale[Int32(0)] * attention_ops.LOG2_E),
            lane,
            False,
        )

        m_frag = cute.make_rmem_tensor(md_layout, Float32)
        d_frag = cute.make_rmem_tensor(md_layout, Float32)
        o_rescale_frag = cute.make_rmem_tensor(md_layout, Float32)
        for row_slot in cutlass.range_constexpr(2):
            m_frag[0, row_slot] = Float32(-Float32.inf)
            d_frag[0, row_slot] = Float32(0.0)
            o_rescale_frag[0, row_slot] = Float32(1.0)
        _update_softmax_stats_b2(score_frag, m_frag, d_frag, o_rescale_frag)

        p_frag = cute.make_rmem_tensor(p_layout, Uint32)
        _fill_normalized_p_frag_from_scores(p_frag, score_frag, m_frag, d_frag)
        for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
            for reg_id in cutlass.range_constexpr(4):
                p_dump[lane, mma_kv, reg_id] = p_frag[0, mma_kv, reg_id]

        _stage_token_scales(kv_scales, sTokenIdx, sScale, Int32(0), Int32(kv_rows_u32.shape[0]), lane)
        token_local = lane
        while token_local < Int32(_MLA_TOKEN_TILE):
            scale_dump[token_local] = Float32(sScale[token_local])
            token_idx_dump[token_local] = Int32(sTokenIdx[token_local])
            token_local += Int32(self.num_threads)

        _stage_kv_u32_block(
            kv_rows_u32,
            sTokenIdx,
            Int32(0),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            kv_base_addr,
            Int32(kv_rows_u32.shape[0]),
            lane,
        )
        cute.arch.sync_threads()
        _pack_live_mxfp8_a_regs_for_dump(a_dump, sfa_dump, p_frag, sScale, lane)
        _dump_live_mxfp8_b_regs_mma_d0(b_dump, kv_base_addr, lane)


def _run_live_fragment_register_dump(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    sm_scale: float,
) -> dict[str, torch.Tensor]:
    kv_rows_bytes = kv_cache[:, 0, :].view(torch.uint8)
    kv_scales = kv_rows_bytes[:, 512 : 512 + 16].view(torch.float32)
    q_u32 = _view_last_dim_as_u32(q_all)
    kv_rows_u32 = _view_last_dim_as_u32(kv_rows_bytes)
    sm_scale_tensor = torch.tensor([sm_scale], device=q_all.device, dtype=torch.float32)
    p_dump = torch.empty((32, _MLA_NUM_MMA_KV, 4), device=q_all.device, dtype=torch.uint32)
    a_dump = torch.empty((32, 4), device=q_all.device, dtype=torch.uint32)
    sfa_dump = torch.empty((32,), device=q_all.device, dtype=torch.uint32)
    b_dump = torch.empty((32, 4), device=q_all.device, dtype=torch.uint32)
    scale_dump = torch.empty((32,), device=q_all.device, dtype=torch.float32)
    token_idx_dump = torch.empty((32,), device=q_all.device, dtype=torch.int32)

    kernel = LiveMlaPvRegisterDumpKernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=q_all.device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(kv_rows_u32, cutlass.Uint32),
        _to_cute_tensor(kv_scales, cutlass.Float32),
        _to_cute_tensor(page_table_1, cutlass.Int32),
        _to_cute_tensor(sm_scale_tensor, cutlass.Float32),
        _to_cute_tensor(p_dump, cutlass.Uint32),
        _to_cute_tensor(a_dump, cutlass.Uint32),
        _to_cute_tensor(sfa_dump, cutlass.Uint32),
        _to_cute_tensor(b_dump, cutlass.Uint32),
        _to_cute_tensor(scale_dump, cutlass.Float32),
        _to_cute_tensor(token_idx_dump, cutlass.Int32),
        stream,
    )
    compiled(
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(kv_rows_u32, cutlass.Uint32),
        _to_cute_tensor(kv_scales, cutlass.Float32),
        _to_cute_tensor(page_table_1, cutlass.Int32),
        _to_cute_tensor(sm_scale_tensor, cutlass.Float32),
        _to_cute_tensor(p_dump, cutlass.Uint32),
        _to_cute_tensor(a_dump, cutlass.Uint32),
        _to_cute_tensor(sfa_dump, cutlass.Uint32),
        _to_cute_tensor(b_dump, cutlass.Uint32),
        _to_cute_tensor(scale_dump, cutlass.Float32),
        _to_cute_tensor(token_idx_dump, cutlass.Int32),
        stream,
    )
    torch.cuda.synchronize(q_all.device)
    return {
        "p_dump": p_dump.cpu(),
        "a_dump": a_dump.cpu(),
        "sfa_dump": sfa_dump.cpu(),
        "b_dump": b_dump.cpu(),
        "scale_dump": scale_dump.cpu(),
        "token_idx_dump": token_idx_dump.cpu(),
    }


def _u32_bytes_le(value: int) -> list[int]:
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


def _decode_e4m3x4_u32(value: int) -> torch.Tensor:
    raw = torch.tensor(_u32_bytes_le(value), dtype=torch.uint8)
    return raw.view(torch.float8_e4m3fn).to(torch.float32)


def _decode_bf16x2_u32(value: int) -> torch.Tensor:
    raw = torch.tensor(
        [value & 0xFFFF, (value >> 16) & 0xFFFF],
        dtype=torch.uint16,
    )
    return raw.view(torch.bfloat16).to(torch.float32)


def _ue8m0_scale_from_byte(byte: int) -> float:
    if byte == 0:
        return 0.0
    return float(2.0 ** (byte - 127))


class TinyMxfp8RawQkProbeKernelV2:
    num_threads = 32

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        k_u32: cute.Tensor,
        a_lo_bf16_dump: cute.Tensor,
        a_hi_bf16_dump: cute.Tensor,
        a_dump: cute.Tensor,
        b_dump: cute.Tensor,
        out_dump: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(q_u32, k_u32, a_lo_bf16_dump, a_hi_bf16_dump, a_dump, b_dump, out_dump).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        k_u32: cute.Tensor,
        a_lo_bf16_dump: cute.Tensor,
        a_hi_bf16_dump: cute.Tensor,
        a_dump: cute.Tensor,
        b_dump: cute.Tensor,
        out_dump: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        SharedStorage = get_sparse_mla_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())
        k_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

        _stage_q_u32_block(
            q_u32,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            q_base_addr,
            lane,
        )

        linear = lane
        total = Int32(_MLA_TOKEN_TILE) * Int32(_MLA_NOPE_GROUP_KV_VECS)
        while linear < total:
            row = linear // Int32(_MLA_NOPE_GROUP_KV_VECS)
            vec_idx = linear - row * Int32(_MLA_NOPE_GROUP_KV_VECS)
            dst_addr = _smem_addr_from_b128_offset(
                k_base_addr,
                _permuted_offset_128b(row, vec_idx, Int32(_MLA_NOPE_GROUP_KV_VECS)),
            )
            src_u32 = vec_idx * Int32(4)
            st_shared_v4_u32(
                dst_addr,
                Uint32(k_u32[row, src_u32 + Int32(0)]),
                Uint32(k_u32[row, src_u32 + Int32(1)]),
                Uint32(k_u32[row, src_u32 + Int32(2)]),
                Uint32(k_u32[row, src_u32 + Int32(3)]),
            )
            linear += Int32(self.num_threads)
        cute.arch.sync_threads()

        q_offset = _permuted_offset_128b(
            lane % Int32(16),
            lane // Int32(16),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
        )
        mask16 = Uint32(0xFFFF)
        shift16 = Uint32(16)
        a_regs_k0 = cute.make_rmem_tensor(cute.make_layout((1, 4), stride=(4, 1)), Uint32)
        q_regs = cute.make_rmem_tensor(cute.make_layout((1, 4), stride=(4, 1)), Uint32)
        q_offset_cur = q_offset
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset_cur))
        a_regs_k0[0, 0] = a0
        a_regs_k0[0, 1] = a1
        a_regs_k0[0, 2] = a2
        a_regs_k0[0, 3] = a3
        a_lo_bf16_dump[lane, 0] = a0
        a_lo_bf16_dump[lane, 1] = a1
        a_lo_bf16_dump[lane, 2] = a2
        a_lo_bf16_dump[lane, 3] = a3
        q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, Int32(_MLA_NOPE_GROUP_Q_VECS))
        q_offset_mid = _advance_offset_by_column_128b_2(q_offset_cur, 0) - Int32(16 * _MLA_NOPE_GROUP_Q_VECS)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset_mid))
        a_hi_bf16_dump[lane, 0] = a0
        a_hi_bf16_dump[lane, 1] = a1
        a_hi_bf16_dump[lane, 2] = a2
        a_hi_bf16_dump[lane, 3] = a3
        q_regs[0, 0] = (cvt_bf16x2_to_e4m3x2(a_regs_k0[0, 0]) & mask16) | (
            (cvt_bf16x2_to_e4m3x2(a_regs_k0[0, 2]) & mask16) << shift16
        )
        q_regs[0, 1] = (cvt_bf16x2_to_e4m3x2(a_regs_k0[0, 1]) & mask16) | (
            (cvt_bf16x2_to_e4m3x2(a_regs_k0[0, 3]) & mask16) << shift16
        )
        q_regs[0, 2] = (cvt_bf16x2_to_e4m3x2(a0) & mask16) | (
            (cvt_bf16x2_to_e4m3x2(a2) & mask16) << shift16
        )
        q_regs[0, 3] = (cvt_bf16x2_to_e4m3x2(a1) & mask16) | (
            (cvt_bf16x2_to_e4m3x2(a3) & mask16) << shift16
        )

        k_offset = _permuted_offset_128b(
            lane % Int32(8),
            (lane % Int32(16)) // Int32(8),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
        ) + Int32(8) * (lane // Int32(16)) * Int32(_MLA_NOPE_GROUP_KV_VECS)
        b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(
            _smem_addr_from_b128_offset(k_base_addr, k_offset)
        )
        b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(
            _smem_addr_from_b128_offset(k_base_addr, k_offset)
        )
        b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
        b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
        b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)

        for reg_id in cutlass.range_constexpr(4):
            a_dump[lane, reg_id] = q_regs[0, reg_id]
        b_dump[lane, 0] = b0_k0
        b_dump[lane, 1] = b0_k1
        b_dump[lane, 2] = b1_k0
        b_dump[lane, 3] = b1_k1

        unit_scale = Uint32(0x7F7F7F7F)
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            q_regs[0, 0],
            q_regs[0, 1],
            q_regs[0, 2],
            q_regs[0, 3],
            b0_k0,
            b0_k1,
            unit_scale,
            unit_scale,
        )
        d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            q_regs[0, 0],
            q_regs[0, 1],
            q_regs[0, 2],
            q_regs[0, 3],
            b1_k0,
            b1_k1,
            unit_scale,
            unit_scale,
        )
        out_dump[lane, 0] = Float32(d0)
        out_dump[lane, 1] = Float32(d1)
        out_dump[lane, 2] = Float32(d2)
        out_dump[lane, 3] = Float32(d3)
        out_dump[lane, 4] = Float32(d4)
        out_dump[lane, 5] = Float32(d5)
        out_dump[lane, 6] = Float32(d6)
        out_dump[lane, 7] = Float32(d7)


class TinyBf16RawQkProbeKernel:
    num_threads = 32

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        k_bf16_u32: cute.Tensor,
        out_dump: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(q_u32, k_bf16_u32, out_dump).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        k_bf16_u32: cute.Tensor,
        out_dump: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        SharedStorage = get_sparse_mla_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())
        k_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

        _stage_q_u32_block(
            q_u32,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            q_base_addr,
            lane,
        )

        linear = lane
        total = Int32(_MLA_HEADS_PER_TILE) * Int32(_MLA_NOPE_GROUP_KV_BF16_VECS)
        while linear < total:
            row = linear // Int32(_MLA_NOPE_GROUP_KV_BF16_VECS)
            vec_idx = linear - row * Int32(_MLA_NOPE_GROUP_KV_BF16_VECS)
            dst_addr = _smem_addr_from_b128_offset(
                k_base_addr,
                _permuted_offset_128b(row, vec_idx, Int32(_MLA_NOPE_GROUP_KV_BF16_VECS)),
            )
            src_u32 = vec_idx * Int32(4)
            st_shared_v4_u32(
                dst_addr,
                Uint32(k_bf16_u32[row, src_u32 + Int32(0)]),
                Uint32(k_bf16_u32[row, src_u32 + Int32(1)]),
                Uint32(k_bf16_u32[row, src_u32 + Int32(2)]),
                Uint32(k_bf16_u32[row, src_u32 + Int32(3)]),
            )
            linear += Int32(self.num_threads)
        cute.arch.sync_threads()

        frag_layout = cute.make_layout((1, 1, 8), stride=(8, 8, 1))
        score_frag = cute.make_rmem_tensor(frag_layout, Float32)
        for reg_id in cutlass.range_constexpr(8):
            score_frag[0, 0, reg_id] = Float32(0.0)

        _literal_qk_mma_into_sfrag_bf16(
            score_frag,
            q_base_addr,
            k_base_addr,
            lane,
            Int32(0),
            Int32(1),
            Int32(1),
            Int32(_MLA_NOPE_QK_NUM_MMA_D),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            Int32(_MLA_NOPE_GROUP_KV_BF16_VECS),
        )
        for reg_id in cutlass.range_constexpr(8):
            out_dump[lane, reg_id] = Float32(score_frag[0, 0, reg_id])


class TinyRawFp8QLoadProbeKernel:
    num_threads = 32

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        a_lo_raw_dump: cute.Tensor,
        a_hi_raw_dump: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(q_u32, a_lo_raw_dump, a_hi_raw_dump).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        a_lo_raw_dump: cute.Tensor,
        a_hi_raw_dump: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            q_group_stage: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _MLA_HEADS_PER_TILE * 128],
                16,
            ]

        storage = smem.allocate(SharedStorage)
        q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())

        linear = lane
        total = Int32(_MLA_HEADS_PER_TILE) * Int32(128 // 16)
        while linear < total:
            row = linear // Int32(128 // 16)
            vec_idx = linear - row * Int32(128 // 16)
            dst_addr = _smem_addr_from_b128_offset(
                q_base_addr,
                _permuted_offset_128b(row, vec_idx, Int32(128 // 16)),
            )
            src_u32 = vec_idx * Int32(4)
            st_shared_v4_u32(
                dst_addr,
                Uint32(q_u32[Int32(0), row, src_u32 + Int32(0)]),
                Uint32(q_u32[Int32(0), row, src_u32 + Int32(1)]),
                Uint32(q_u32[Int32(0), row, src_u32 + Int32(2)]),
                Uint32(q_u32[Int32(0), row, src_u32 + Int32(3)]),
            )
            linear += Int32(self.num_threads)
        cute.arch.sync_threads()

        q_offset = _permuted_offset_128b(
            lane % Int32(16),
            lane // Int32(16),
            Int32(128 // 16),
        )
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset))
        a_lo_raw_dump[lane, 0] = a0
        a_lo_raw_dump[lane, 1] = a1
        a_lo_raw_dump[lane, 2] = a2
        a_lo_raw_dump[lane, 3] = a3

        q_offset_cur = _advance_offset_by_row_128b(q_offset, 16, Int32(128 // 16))
        q_offset_mid = _advance_offset_by_column_128b_2(q_offset_cur, 0) - Int32(
            16 * (128 // 16)
        )
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset_mid))
        a_hi_raw_dump[lane, 0] = a0
        a_hi_raw_dump[lane, 1] = a1
        a_hi_raw_dump[lane, 2] = a2
        a_hi_raw_dump[lane, 3] = a3


class TinyRawFp8QkProbeKernel:
    num_threads = 32

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        k_u32: cute.Tensor,
        a_dump: cute.Tensor,
        out_dump: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(q_u32, k_u32, a_dump, out_dump).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        k_u32: cute.Tensor,
        a_dump: cute.Tensor,
        out_dump: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        SharedStorage = get_sparse_mla_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())
        k_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())
        s_q_bytes = storage.q_group_stage.get_tensor(
            cute.make_layout((_MLA_HEADS_PER_TILE, 128), stride=(128, 1))
        )

        linear = lane
        total_q = Int32(_MLA_HEADS_PER_TILE) * Int32(128 // 16)
        while linear < total_q:
            row = linear // Int32(128 // 16)
            vec_idx = linear - row * Int32(128 // 16)
            dst_addr = q_base_addr + row * Int32(128) + vec_idx * Int32(16)
            src_u32 = vec_idx * Int32(4)
            st_shared_v4_u32(
                dst_addr,
                Uint32(q_u32[Int32(0), row, src_u32 + Int32(0)]),
                Uint32(q_u32[Int32(0), row, src_u32 + Int32(1)]),
                Uint32(q_u32[Int32(0), row, src_u32 + Int32(2)]),
                Uint32(q_u32[Int32(0), row, src_u32 + Int32(3)]),
            )
            linear += Int32(self.num_threads)

        linear = lane
        total_k = Int32(_MLA_TOKEN_TILE) * Int32(_MLA_NOPE_GROUP_KV_VECS)
        while linear < total_k:
            row = linear // Int32(_MLA_NOPE_GROUP_KV_VECS)
            vec_idx = linear - row * Int32(_MLA_NOPE_GROUP_KV_VECS)
            dst_addr = _smem_addr_from_b128_offset(
                k_base_addr,
                _permuted_offset_128b(row, vec_idx, Int32(_MLA_NOPE_GROUP_KV_VECS)),
            )
            src_u32 = vec_idx * Int32(4)
            st_shared_v4_u32(
                dst_addr,
                Uint32(k_u32[row, src_u32 + Int32(0)]),
                Uint32(k_u32[row, src_u32 + Int32(1)]),
                Uint32(k_u32[row, src_u32 + Int32(2)]),
                Uint32(k_u32[row, src_u32 + Int32(3)]),
            )
            linear += Int32(self.num_threads)
        cute.arch.sync_threads()

        group = lane // Int32(4)
        sub = lane % Int32(4)
        base = sub * Int32(2)
        q_regs = cute.make_rmem_tensor(cute.make_layout((1, 4), stride=(4, 1)), Uint32)
        q_regs[0, 0] = (
            Uint32(s_q_bytes[group, base + Int32(0)])
            | (Uint32(s_q_bytes[group, base + Int32(1)]) << Int32(8))
            | (Uint32(s_q_bytes[group, base + Int32(8)]) << Int32(16))
            | (Uint32(s_q_bytes[group, base + Int32(9)]) << Int32(24))
        )
        q_regs[0, 1] = (
            Uint32(s_q_bytes[group + Int32(8), base + Int32(0)])
            | (Uint32(s_q_bytes[group + Int32(8), base + Int32(1)]) << Int32(8))
            | (Uint32(s_q_bytes[group + Int32(8), base + Int32(8)]) << Int32(16))
            | (Uint32(s_q_bytes[group + Int32(8), base + Int32(9)]) << Int32(24))
        )
        q_regs[0, 2] = (
            Uint32(s_q_bytes[group, base + Int32(16)])
            | (Uint32(s_q_bytes[group, base + Int32(17)]) << Int32(8))
            | (Uint32(s_q_bytes[group, base + Int32(24)]) << Int32(16))
            | (Uint32(s_q_bytes[group, base + Int32(25)]) << Int32(24))
        )
        q_regs[0, 3] = (
            Uint32(s_q_bytes[group + Int32(8), base + Int32(16)])
            | (Uint32(s_q_bytes[group + Int32(8), base + Int32(17)]) << Int32(8))
            | (Uint32(s_q_bytes[group + Int32(8), base + Int32(24)]) << Int32(16))
            | (Uint32(s_q_bytes[group + Int32(8), base + Int32(25)]) << Int32(24))
        )
        for reg_id in cutlass.range_constexpr(4):
            a_dump[lane, reg_id] = q_regs[0, reg_id]

        k_offset = _permuted_offset_128b(
            lane % Int32(8),
            (lane % Int32(16)) // Int32(8),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
        ) + Int32(8) * (lane // Int32(16)) * Int32(_MLA_NOPE_GROUP_KV_VECS)
        b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(
            _smem_addr_from_b128_offset(k_base_addr, k_offset)
        )
        b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(
            _smem_addr_from_b128_offset(k_base_addr, k_offset)
        )
        b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
        b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
        b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)

        unit_scale = Uint32(0x7F7F7F7F)
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            q_regs[0, 0],
            q_regs[0, 1],
            q_regs[0, 2],
            q_regs[0, 3],
            b0_k0,
            b0_k1,
            unit_scale,
            unit_scale,
        )
        d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            q_regs[0, 0],
            q_regs[0, 1],
            q_regs[0, 2],
            q_regs[0, 3],
            b1_k0,
            b1_k1,
            unit_scale,
            unit_scale,
        )
        out_dump[lane, 0] = Float32(d0)
        out_dump[lane, 1] = Float32(d1)
        out_dump[lane, 2] = Float32(d2)
        out_dump[lane, 3] = Float32(d3)
        out_dump[lane, 4] = Float32(d4)
        out_dump[lane, 5] = Float32(d5)
        out_dump[lane, 6] = Float32(d6)
        out_dump[lane, 7] = Float32(d7)


def _run_tiny_raw_qk_probe(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
) -> dict[str, torch.Tensor]:
    q_u32 = _view_last_dim_as_u32(q)
    k_u32 = _view_last_dim_as_u32(k.view(torch.uint8))
    a_lo_bf16_dump = torch.empty((32, 4), device=q.device, dtype=torch.uint32)
    a_hi_bf16_dump = torch.empty((32, 4), device=q.device, dtype=torch.uint32)
    a_dump = torch.empty((32, 4), device=q.device, dtype=torch.uint32)
    b_dump = torch.empty((32, 4), device=q.device, dtype=torch.uint32)
    out_dump = torch.empty((32, 8), device=q.device, dtype=torch.float32)

    kernel = TinyMxfp8RawQkProbeKernelV2()
    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(k_u32, cutlass.Uint32),
        _to_cute_tensor(a_lo_bf16_dump, cutlass.Uint32),
        _to_cute_tensor(a_hi_bf16_dump, cutlass.Uint32),
        _to_cute_tensor(a_dump, cutlass.Uint32),
        _to_cute_tensor(b_dump, cutlass.Uint32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    compiled(
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(k_u32, cutlass.Uint32),
        _to_cute_tensor(a_lo_bf16_dump, cutlass.Uint32),
        _to_cute_tensor(a_hi_bf16_dump, cutlass.Uint32),
        _to_cute_tensor(a_dump, cutlass.Uint32),
        _to_cute_tensor(b_dump, cutlass.Uint32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize(q.device)
    return {
        "a_lo_bf16_dump": a_lo_bf16_dump.cpu(),
        "a_hi_bf16_dump": a_hi_bf16_dump.cpu(),
        "a_dump": a_dump.cpu(),
        "b_dump": b_dump.cpu(),
        "out_dump": out_dump.cpu(),
    }


def _run_tiny_raw_fp8_q_load_probe(
    *,
    q_fp8: torch.Tensor,
) -> dict[str, torch.Tensor]:
    q_u32 = _view_last_dim_as_u32(q_fp8.view(torch.uint8))
    a_lo_raw_dump = torch.empty((32, 4), device=q_fp8.device, dtype=torch.uint32)
    a_hi_raw_dump = torch.empty((32, 4), device=q_fp8.device, dtype=torch.uint32)

    kernel = TinyRawFp8QLoadProbeKernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=q_fp8.device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(a_lo_raw_dump, cutlass.Uint32),
        _to_cute_tensor(a_hi_raw_dump, cutlass.Uint32),
        stream,
    )
    compiled(
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(a_lo_raw_dump, cutlass.Uint32),
        _to_cute_tensor(a_hi_raw_dump, cutlass.Uint32),
        stream,
    )
    torch.cuda.synchronize(q_fp8.device)
    return {
        "a_lo_raw_dump": a_lo_raw_dump.cpu(),
        "a_hi_raw_dump": a_hi_raw_dump.cpu(),
    }


def _run_tiny_raw_fp8_qk_probe(
    *,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
) -> dict[str, torch.Tensor]:
    q_u32 = _view_last_dim_as_u32(q_fp8.view(torch.uint8))
    k_u32 = _view_last_dim_as_u32(k.view(torch.uint8))
    a_dump = torch.empty((32, 4), device=q_fp8.device, dtype=torch.uint32)
    out_dump = torch.empty((32, 8), device=q_fp8.device, dtype=torch.float32)

    kernel = TinyRawFp8QkProbeKernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=q_fp8.device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(k_u32, cutlass.Uint32),
        _to_cute_tensor(a_dump, cutlass.Uint32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    compiled(
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(k_u32, cutlass.Uint32),
        _to_cute_tensor(a_dump, cutlass.Uint32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize(q_fp8.device)
    return {
        "a_dump": a_dump.cpu(),
        "out_dump": out_dump.cpu(),
    }


def _run_tiny_raw_qk_bf16_probe(
    *,
    q: torch.Tensor,
    k_bf16: torch.Tensor,
) -> torch.Tensor:
    q_u32 = _view_last_dim_as_u32(q)
    k_bf16_u32 = _view_last_dim_as_u32(k_bf16)
    out_dump = torch.empty((32, 8), device=q.device, dtype=torch.float32)

    kernel = TinyBf16RawQkProbeKernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(k_bf16_u32, cutlass.Uint32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    compiled(
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(k_bf16_u32, cutlass.Uint32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize(q.device)
    return out_dump.cpu()


class FullMxfp8ScoreProbeKernel:
    num_threads = 32

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        k_u32: cute.Tensor,
        out_dump: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(q_u32, k_u32, out_dump).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        k_u32: cute.Tensor,
        out_dump: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        SharedStorage = get_sparse_mla_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())
        k_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

        _stage_q_u32_block(
            q_u32,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            q_base_addr,
            lane,
        )

        linear = lane
        total = Int32(_MLA_TOKEN_TILE) * Int32(_MLA_NOPE_GROUP_KV_VECS)
        while linear < total:
            row = linear // Int32(_MLA_NOPE_GROUP_KV_VECS)
            vec_idx = linear - row * Int32(_MLA_NOPE_GROUP_KV_VECS)
            dst_addr = _smem_addr_from_b128_offset(
                k_base_addr,
                _permuted_offset_128b(row, vec_idx, Int32(_MLA_NOPE_GROUP_KV_VECS)),
            )
            src_u32 = vec_idx * Int32(4)
            st_shared_v4_u32(
                dst_addr,
                Uint32(k_u32[row, src_u32 + Int32(0)]),
                Uint32(k_u32[row, src_u32 + Int32(1)]),
                Uint32(k_u32[row, src_u32 + Int32(2)]),
                Uint32(k_u32[row, src_u32 + Int32(3)]),
            )
            linear += Int32(self.num_threads)
        cute.arch.sync_threads()

        frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))
        score_frag = cute.make_rmem_tensor(frag_layout, Float32)
        _zero_score_frag(score_frag)
        _literal_qk_mma_into_sfrag_mxfp8_raw(
            score_frag,
            q_base_addr,
            k_base_addr,
            lane,
            Int32(0),
            Int32(1),
            Int32(_MLA_NUM_MMA_KV),
            Int32(_MLA_NOPE_QK_NUM_MMA_D),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
        )
        for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
            for reg_id in cutlass.range_constexpr(8):
                out_dump[lane, mma_kv, reg_id] = Float32(score_frag[0, mma_kv, reg_id])


def _run_full_mxfp8_score_probe(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    q_u32 = _view_last_dim_as_u32(q)
    k_u32 = _view_last_dim_as_u32(k.view(torch.uint8))
    out_dump = torch.empty((32, _MLA_NUM_MMA_KV, 8), device=q.device, dtype=torch.float32)

    kernel = FullMxfp8ScoreProbeKernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(k_u32, cutlass.Uint32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    compiled(
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(k_u32, cutlass.Uint32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize(q.device)
    return out_dump.cpu()


class ScoreTileProbeKernel:
    num_threads = 32

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        sm_scale: cute.Tensor,
        token_base: cute.Tensor,
        token_end: cute.Tensor,
        out_dump: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_u32,
            kv_rows_u32,
            kv_scales,
            page_table_1,
            sm_scale,
            token_base,
            token_end,
            out_dump,
        ).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        sm_scale: cute.Tensor,
        token_base: cute.Tensor,
        token_end: cute.Tensor,
        out_dump: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        SharedStorage = get_sparse_mla_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        sTokenIdx = storage.token_idx.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        sScale = storage.token_scale_a.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())
        kv_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

        frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))
        score_frag = cute.make_rmem_tensor(frag_layout, Float32)
        _compute_score_tile_scaled(
            score_frag,
            q_u32,
            kv_rows_u32,
            kv_scales,
            page_table_1,
            sTokenIdx,
            sScale,
            q_base_addr,
            kv_base_addr,
            Int32(0),
            Int32(0),
            Int32(token_base[Int32(0)]),
            Int32(token_end[Int32(0)]),
            Float32(sm_scale[Int32(0)] * attention_ops.LOG2_E),
            lane,
            False,
        )
        for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
            for reg_id in cutlass.range_constexpr(8):
                out_dump[lane, mma_kv, reg_id] = Float32(score_frag[0, mma_kv, reg_id])


def _run_score_tile_probe(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    sm_scale: float,
    token_base: int,
    token_end: int,
) -> torch.Tensor:
    kv_rows_bytes = kv_cache[:, 0, :].view(torch.uint8)
    kv_scales = kv_rows_bytes[:, 512 : 512 + 16].view(torch.float32)
    q_u32 = _view_last_dim_as_u32(q_all)
    kv_rows_u32 = _view_last_dim_as_u32(kv_rows_bytes)
    sm_scale_tensor = torch.tensor([sm_scale], device=q_all.device, dtype=torch.float32)
    token_base_tensor = torch.tensor([token_base], device=q_all.device, dtype=torch.int32)
    token_end_tensor = torch.tensor([token_end], device=q_all.device, dtype=torch.int32)
    out_dump = torch.empty((32, _MLA_NUM_MMA_KV, 8), device=q_all.device, dtype=torch.float32)

    kernel = ScoreTileProbeKernel()
    stream = cuda.CUstream(torch.cuda.current_stream(device=q_all.device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(kv_rows_u32, cutlass.Uint32),
        _to_cute_tensor(kv_scales, cutlass.Float32),
        _to_cute_tensor(page_table_1, cutlass.Int32),
        _to_cute_tensor(sm_scale_tensor, cutlass.Float32),
        _to_cute_tensor(token_base_tensor, cutlass.Int32),
        _to_cute_tensor(token_end_tensor, cutlass.Int32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    compiled(
        _to_cute_tensor(q_u32, cutlass.Uint32),
        _to_cute_tensor(kv_rows_u32, cutlass.Uint32),
        _to_cute_tensor(kv_scales, cutlass.Float32),
        _to_cute_tensor(page_table_1, cutlass.Int32),
        _to_cute_tensor(sm_scale_tensor, cutlass.Float32),
        _to_cute_tensor(token_base_tensor, cutlass.Int32),
        _to_cute_tensor(token_end_tensor, cutlass.Int32),
        _to_cute_tensor(out_dump, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize(q_all.device)
    return out_dump.cpu()


def _reconstruct_qk_output_tile_from_dump(out_dump: torch.Tensor) -> torch.Tensor:
    tile = torch.empty((16, 16), dtype=torch.float32)
    for lane in range(32):
        lane_group = lane // 4
        lane_pair_base = 2 * (lane % 4)
        for reg_id in range(8):
            row_slot = (reg_id % 4) // 2
            row = lane_group + 8 * row_slot
            col = lane_pair_base + 8 * (reg_id // 4) + (reg_id % 2)
            tile[row, col] = float(out_dump[lane, reg_id])
    return tile


def _reconstruct_score_tile32_from_dump(out_dump: torch.Tensor) -> torch.Tensor:
    tile = torch.empty((16, 32), dtype=torch.float32)
    for lane in range(32):
        lane_group = lane // 4
        lane_pair_base = 2 * (lane % 4)
        for mma_kv in range(_MLA_NUM_MMA_KV):
            for reg_id in range(8):
                row_slot = (reg_id % 4) // 2
                row = lane_group + 8 * row_slot
                col = mma_kv * 16 + lane_pair_base + 8 * (reg_id // 4) + (reg_id % 2)
                tile[row, col] = float(out_dump[lane, mma_kv, reg_id])
    return tile


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiny_mla_pv_probe_bf16_path_matches_reference() -> None:
    device = require_sm120()
    torch.manual_seed(0)

    p = torch.randn((16, _MLA_TOKEN_TILE), device=device, dtype=torch.float32).to(torch.bfloat16) / 4
    scales = torch.linspace(0.125, 1.5, _MLA_TOKEN_TILE, device=device, dtype=torch.float32)
    v_src = torch.randn((_MLA_TOKEN_TILE, 128), device=device, dtype=torch.float32) / 4

    actual, ref = _run_probe(p=p, v_src=v_src, scales=scales, use_mxfp8=False)
    torch.testing.assert_close(actual, ref, atol=5e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_raw_qk_bf16_probe_matches_reference() -> None:
    device = require_sm120()
    torch.manual_seed(60_101)

    q = (torch.randn((1, 16, 128), device=device, dtype=torch.float32) / 4).to(torch.bfloat16)
    k_bf16 = (torch.randn((16, 128), device=device, dtype=torch.float32) / 4).to(torch.bfloat16)

    actual = _reconstruct_qk_output_tile_from_dump(_run_tiny_raw_qk_bf16_probe(q=q, k_bf16=k_bf16))
    expected = torch.matmul(q[0].to(torch.float32), k_bf16.to(torch.float32).transpose(0, 1)).cpu()

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_raw_qk_mxfp8_probe_matches_k32_quantized_reference() -> None:
    device = require_sm120()
    torch.manual_seed(60_201)

    q = (torch.randn((1, 16, 128), device=device, dtype=torch.float32) / 4).to(torch.bfloat16)
    k = (torch.randn((32, 128), device=device, dtype=torch.float32) / 4).to(torch.float8_e4m3fn)

    dump = _run_tiny_raw_qk_probe(q=q, k=k)
    actual = _reconstruct_qk_output_tile_from_dump(dump["out_dump"])
    expected = torch.matmul(
        q[0, :, :32].to(torch.float8_e4m3fn).to(torch.float32),
        k[:16, :32].to(torch.float32).transpose(0, 1),
    ).cpu()

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.0)


@pytest.mark.skip(reason="Manual diagnostic probe while MXFP8 PV fragment mapping is under investigation")
def test_tiny_mla_pv_probe_mxfp8_unit_scale_matches_reference() -> None:
    device = require_sm120()
    torch.manual_seed(1)

    p = torch.randn((16, _MLA_TOKEN_TILE), device=device, dtype=torch.float32).to(torch.bfloat16) / 2
    scales = torch.ones((_MLA_TOKEN_TILE,), device=device, dtype=torch.float32)
    v_src = torch.randn((_MLA_TOKEN_TILE, 128), device=device, dtype=torch.float32) / 2

    actual, ref = _run_probe(p=p, v_src=v_src, scales=scales, use_mxfp8=True)
    diff = (actual - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        actual.reshape(-1).to(torch.float32),
        ref.reshape(-1).to(torch.float32),
        dim=0,
    ).item()
    assert diff <= 0.15, f"max_abs={diff:.6f}"
    assert cos >= 0.999, f"cos={cos:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_live_fragment_probe_bf16_matches_dense_reference_group0() -> None:
    device = require_sm120()
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=2050,
        q_len=1,
        seed=52_001,
        device=device,
    )
    kv_cache = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = torch.arange(_MLA_TOKEN_TILE, device=device, dtype=torch.int32).unsqueeze(0)

    actual = _run_live_fragment_probe(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        use_mxfp8=False,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=_MLA_GROUP_SIZE,
    )[:, :_MLA_HEADS_PER_TILE, :_MLA_GROUP_SIZE].to(torch.float32)

    max_abs = (actual - expected).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        actual.reshape(-1).to(torch.float32),
        expected.reshape(-1).to(torch.float32),
        dim=0,
    ).item()
    assert max_abs <= 0.08, f"max_abs={max_abs:.6f}"
    assert cos >= 0.999, f"cos={cos:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_live_fragment_probe_a_pack_matches_host_reconstruction() -> None:
    device = require_sm120()
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=2050,
        q_len=1,
        seed=52_101,
        device=device,
    )
    kv_cache = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = torch.arange(_MLA_TOKEN_TILE, device=device, dtype=torch.int32).unsqueeze(0)
    dump = _run_live_fragment_register_dump(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
    )

    p_dump = dump["p_dump"]
    a_dump = dump["a_dump"]
    sfa_dump = dump["sfa_dump"]
    scales = dump["scale_dump"].to(torch.float32)

    max_abs = 0.0
    for lane in range(32):
        lane_pair_base = 2 * (lane % 4)
        scale01 = torch.tensor(
            [
                float(scales[lane_pair_base + 0]),
                float(scales[lane_pair_base + 1]),
                float(scales[lane_pair_base + 16]),
                float(scales[lane_pair_base + 17]),
            ],
            dtype=torch.float32,
        )
        scale89 = torch.tensor(
            [
                float(scales[lane_pair_base + 8]),
                float(scales[lane_pair_base + 9]),
                float(scales[lane_pair_base + 24]),
                float(scales[lane_pair_base + 25]),
            ],
            dtype=torch.float32,
        )
        p_targets = [
            torch.cat(
                [
                    _decode_bf16x2_u32(int(p_dump[lane, 0, 0])),
                    _decode_bf16x2_u32(int(p_dump[lane, 1, 0])),
                ]
            )
            * scale01,
            torch.cat(
                [
                    _decode_bf16x2_u32(int(p_dump[lane, 0, 1])),
                    _decode_bf16x2_u32(int(p_dump[lane, 1, 1])),
                ]
            )
            * scale01,
            torch.cat(
                [
                    _decode_bf16x2_u32(int(p_dump[lane, 0, 2])),
                    _decode_bf16x2_u32(int(p_dump[lane, 1, 2])),
                ]
            )
            * scale89,
            torch.cat(
                [
                    _decode_bf16x2_u32(int(p_dump[lane, 0, 3])),
                    _decode_bf16x2_u32(int(p_dump[lane, 1, 3])),
                ]
            )
            * scale89,
        ]
        sfa_bytes = _u32_bytes_le(int(sfa_dump[lane]))
        scale01_sfa = _ue8m0_scale_from_byte(sfa_bytes[0])
        scale89_sfa = _ue8m0_scale_from_byte(sfa_bytes[2])
        reg_scales = [scale01_sfa, scale01_sfa, scale89_sfa, scale89_sfa]

        for reg_id in range(4):
            represented = _decode_e4m3x4_u32(int(a_dump[lane, reg_id])) * reg_scales[reg_id]
            reg_err = float((represented - p_targets[reg_id]).abs().max().item())
            max_abs = max(max_abs, reg_err)

    assert max_abs <= 0.03, f"max_abs={max_abs:.6f}"
