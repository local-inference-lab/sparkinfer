"""MXFP8/BF16 sparse MLA kernels for the exact GLM-5.1 NSA packed-cache contract."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from b12x.attention._cute import ops as attention_ops
from b12x.cute.compiler import KernelCompileSpec, launch as b12x_launch
from b12x.cute.fp4 import (
    bf16_mma_m16n16k16_f32,
    bfloat2_habs2,
    bfloat2_hmax2,
    bfloat2_hmax_to_f32,
    bfloat2_mul,
    broadcast_f32_to_bfloat2,
    byte_perm,
    cvt_f32_to_ue8m0,
    cvt_bf16x2_to_e4m3x2,
    fp8x4_e4m3_to_bfloat2x2,
    frag_layout_swizzle_16b_to_8b,
    frag_layout_swizzle_16b_to_8b_trans,
    get_ptr_as_int64,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    ldmatrix_m8n8x4_trans_left_half_b16,
    ldmatrix_m8n8x4_trans_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_shared_v4_u32,
    ue8m0_to_output_scale,
)
from b12x.cute.utils import current_cuda_stream

from .reference import _MLA_GROUP_SIZE, _MLA_NOPE_DIM, _MLA_PACKED_DIM, _MLA_ROPE_DIM
from .traits import SparseMLATraits, select_sparse_mla_traits


_MLA_HEAD_DIM = _MLA_NOPE_DIM + _MLA_ROPE_DIM
_MLA_SCALE_GROUPS = _MLA_NOPE_DIM // _MLA_GROUP_SIZE
_MLA_OUTPUT_DIM = _MLA_NOPE_DIM
_MLA_WARP_THREADS = 32
_MLA_OUTPUT_FRAGMENTS_PER_LANE = _MLA_OUTPUT_DIM // _MLA_WARP_THREADS
_MLA_HEADS_PER_TILE = 16
_MLA_TOKEN_TILE = 32
_MLA_NOPE_GROUP_ELEMS = 128
_MLA_NOPE_GROUP_Q_U32 = _MLA_NOPE_GROUP_ELEMS // 2
_MLA_NOPE_GROUP_KV_U32 = _MLA_NOPE_GROUP_ELEMS // 4
_MLA_NOPE_GROUP_Q_VECS = (_MLA_NOPE_GROUP_ELEMS * 2) // 16
_MLA_NOPE_GROUP_KV_VECS = _MLA_NOPE_GROUP_ELEMS // 16
_MLA_NOPE_GROUP_KV_BF16_VECS = (_MLA_NOPE_GROUP_ELEMS * 2) // 16
_MLA_NOPE_QK_NUM_MMA_D = _MLA_NOPE_GROUP_ELEMS // 16
_MLA_ROPE_Q_U32 = _MLA_ROPE_DIM // 2
_MLA_ROPE_VECS = (_MLA_ROPE_DIM * 2) // 16
_MLA_Q_NOPE_STAGE_BYTES = _MLA_HEADS_PER_TILE * _MLA_NOPE_GROUP_ELEMS * 2
_MLA_Q_ROPE_STAGE_BYTES = _MLA_HEADS_PER_TILE * _MLA_ROPE_DIM * 2
_MLA_KV_NOPE_STAGE_BYTES = _MLA_TOKEN_TILE * _MLA_NOPE_GROUP_ELEMS
_MLA_KV_NOPE_QK_STAGE_BYTES = _MLA_TOKEN_TILE * _MLA_NOPE_GROUP_ELEMS * 2
_MLA_KV_ROPE_STAGE_BYTES = _MLA_TOKEN_TILE * _MLA_ROPE_DIM * 2
_MLA_Q_STAGE_BYTES = _MLA_SCALE_GROUPS * _MLA_Q_NOPE_STAGE_BYTES + _MLA_Q_ROPE_STAGE_BYTES
_MLA_KV_STAGE_BYTES = max(
    _MLA_KV_NOPE_QK_STAGE_BYTES,
    _MLA_SCALE_GROUPS * _MLA_KV_NOPE_STAGE_BYTES + _MLA_KV_ROPE_STAGE_BYTES,
)
_MLA_SCALE_STAGE_ELEMS = _MLA_TOKEN_TILE * _MLA_SCALE_GROUPS
_MLA_SCALE_BYTES = _MLA_SCALE_GROUPS * 4
_MLA_NOPE_U32_OFFSET = 0
_MLA_SCALE_U32_OFFSET = _MLA_NOPE_DIM // 4
_MLA_ROPE_U32_OFFSET = _MLA_SCALE_U32_OFFSET + _MLA_SCALE_GROUPS
_MLA_NUM_MMA_KV = 2
_MLA_QK_NUM_MMA_D = 4
_MLA_VO_NUM_MMA_D = _MLA_NOPE_GROUP_ELEMS // 16
def _raise_binding_extras(api_name: str, extras: list[str]) -> None:
    raise ValueError(
        f"{api_name} binding owns runtime tensors, workspace, and kernel options; "
        f"do not also pass {', '.join(extras)}"
    )


def _require_bound_arg(value, *, api_name: str, name: str):
    if value is None:
        raise TypeError(f"{api_name} requires {name} or binding")
    return value


@dataclass(frozen=True, kw_only=True)
class SparseMLAOnePassKernelBinding:
    q_all: torch.Tensor
    kv_cache: torch.Tensor
    page_table_1: torch.Tensor
    active_token_counts: torch.Tensor
    sm_scale: float | torch.Tensor
    output: torch.Tensor
    workspace: object | None = None

    def run(self) -> None:
        run_sparse_mla_kernel(binding=self)


def build_sparse_mla_onepass_kernel_binding(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    active_token_counts: torch.Tensor,
    sm_scale: float | torch.Tensor,
    output: torch.Tensor,
    workspace: object | None = None,
) -> SparseMLAOnePassKernelBinding:
    return SparseMLAOnePassKernelBinding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        output=output,
        workspace=workspace,
    )


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.int32:
        return cutlass.Int32
    if dtype == torch.uint8:
        return cutlass.Uint8
    if dtype == torch.uint32:
        return cutlass.Uint32
    raise TypeError(f"unsupported dtype {dtype}")


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> cutlass.cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    leading_dim = next((idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None)
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_meta_key(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _workspace_contract_kv_tensors(
    workspace: object | None,
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if workspace is None:
        return None, None
    contract_selector = getattr(workspace, "contract_kv_tensors_for", None)
    if callable(contract_selector):
        return contract_selector(kv_cache)
    return (
        getattr(workspace, "_contract_kv_rows", None),
        getattr(workspace, "_contract_kv_scales", None),
    )


@cute.jit
def _warp_allreduce_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


@cute.jit
def _warp_allreduce_max(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = attention_ops.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


@dsl_user_op
def _cp_async_load_128b_pred(
    smem_addr: Int32,
    gmem_addr: Int64,
    predicate: Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(predicate).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, $0, 0;\n"
        " @p cp.async.cg.shared.global.L2::128B [$1], [$2], 16;\n"
        "}",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _exp2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _log2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _permuted_offset_128b(row_idx, vec_idx, stride_128b):
    return row_idx * stride_128b + (vec_idx ^ (row_idx % 8))


@cute.jit
def _smem_addr_from_b128_offset(base_addr: Int32, offset_128b):
    return base_addr + Int32(offset_128b * 16)


@cute.jit
def _advance_offset_by_row_128b(offset_128b, step_size, row_stride_128b):
    return offset_128b + step_size * row_stride_128b


@cute.jit
def _advance_offset_by_column_128b_2(offset_128b, step_idx):
    xor_term = Int32(0x2) + (Int32(0x4) if step_idx % 2 == 1 else Int32(0))
    extra = Int32(8) if step_idx % 4 == 3 else Int32(0)
    return (offset_128b ^ xor_term) + extra


@cute.jit
def _stage_token_indices(
    page_table_1: cute.Tensor,
    sTokenIdx: cute.Tensor,
    q_idx: Int32,
    token_base: Int32,
    token_end: Int32,
    lane: Int32,
):
    token_local = lane
    while token_local < Int32(_MLA_TOKEN_TILE):
        token_pos = token_base + token_local
        sTokenIdx[token_local] = (
            Int32(page_table_1[q_idx, token_pos]) if token_pos < token_end else Int32(-1)
        )
        token_local += Int32(_MLA_WARP_THREADS)


@cute.jit
def _stage_token_scales(
    kv_scales: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    group_idx: Int32,
    num_kv: Int32,
    lane: Int32,
):
    token_local = lane
    while token_local < Int32(_MLA_TOKEN_TILE):
        token_idx = Int32(sTokenIdx[token_local])
        sScale[token_local] = (
            Float32(kv_scales[token_idx, group_idx])
            if token_idx >= Int32(0) and token_idx < num_kv
            else Float32(0.0)
        )
        token_local += Int32(_MLA_WARP_THREADS)


@cute.jit
def _stage_all_token_scales(
    kv_scales: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    num_kv: Int32,
    lane: Int32,
):
    linear = lane
    total = Int32(_MLA_SCALE_STAGE_ELEMS)
    while linear < total:
        group_idx = linear // Int32(_MLA_TOKEN_TILE)
        token_local = linear - group_idx * Int32(_MLA_TOKEN_TILE)
        token_idx = Int32(sTokenIdx[token_local])
        sScale[linear] = (
            Float32(kv_scales[token_idx, group_idx])
            if token_idx >= Int32(0) and token_idx < num_kv
            else Float32(0.0)
        )
        linear += Int32(_MLA_WARP_THREADS)


@cute.jit
def _clamp_active_token_count(
    active_token_counts: cute.Tensor,
    q_idx: Int32,
    max_width: Int32,
) -> Int32:
    token_end = Int32(active_token_counts[q_idx])
    if token_end < Int32(0):
        token_end = Int32(0)
    if token_end > max_width:
        token_end = max_width
    return token_end


def _extract_packed_kv_runtime_views(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_rows_bytes = kv_cache[:, 0, :].view(torch.uint8)
    kv_rows_u32 = _view_last_dim_as_u32(kv_rows_bytes)
    kv_scales = kv_rows_bytes[:, _MLA_NOPE_DIM : _MLA_NOPE_DIM + _MLA_SCALE_BYTES].view(
        torch.float32
    )
    return kv_rows_u32, kv_scales


@cute.jit
def _stage_q_u32_block(
    q_u32: cute.Tensor,
    q_idx: Int32,
    head_tile_start: Int32,
    base_u32: Int32,
    vecs_per_row: Int32,
    row_stride_128b: Int32,
    q_base_addr: Int32,
    lane: Int32,
):
    linear = lane
    total = Int32(_MLA_HEADS_PER_TILE) * vecs_per_row
    num_heads = Int32(q_u32.shape[1])
    while linear < total:
        row = linear // vecs_per_row
        vec_idx = linear - row * vecs_per_row
        head_idx = head_tile_start + row
        dst_addr = _smem_addr_from_b128_offset(
            q_base_addr,
            _permuted_offset_128b(row, vec_idx, row_stride_128b),
        )
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
        if head_idx < num_heads:
            src_u32 = base_u32 + vec_idx * Int32(4)
            v0 = Uint32(q_u32[q_idx, head_idx, src_u32 + Int32(0)])
            v1 = Uint32(q_u32[q_idx, head_idx, src_u32 + Int32(1)])
            v2 = Uint32(q_u32[q_idx, head_idx, src_u32 + Int32(2)])
            v3 = Uint32(q_u32[q_idx, head_idx, src_u32 + Int32(3)])
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(_MLA_WARP_THREADS)


@cute.jit
def _stage_q_u32_block_async(
    q_u32: cute.Tensor,
    q_idx: Int32,
    head_tile_start: Int32,
    base_u32: Int32,
    vecs_per_row: Int32,
    row_stride_128b: Int32,
    q_base_addr: Int32,
    lane: Int32,
):
    linear = lane
    total = Int32(_MLA_HEADS_PER_TILE) * vecs_per_row
    num_heads = Int32(q_u32.shape[1])
    row_stride_u32 = Int32(q_u32.shape[2])
    q_row_stride_u32 = Int32(q_u32.shape[1]) * row_stride_u32
    while linear < total:
        row = linear // vecs_per_row
        vec_idx = linear - row * vecs_per_row
        head_idx = head_tile_start + row
        dst_addr = _smem_addr_from_b128_offset(
            q_base_addr,
            _permuted_offset_128b(row, vec_idx, row_stride_128b),
        )
        valid = Int32(0)
        safe_head_idx = Int32(0)
        if head_idx < num_heads:
            valid = Int32(1)
            safe_head_idx = head_idx
        src_u32 = base_u32 + vec_idx * Int32(4)
        _cp_async_load_128b_pred(
            dst_addr,
            get_ptr_as_int64(
                q_u32,
                q_idx * q_row_stride_u32 + safe_head_idx * row_stride_u32 + src_u32,
            ),
            valid,
        )
        linear += Int32(_MLA_WARP_THREADS)


@cute.jit
def _stage_kv_u32_block(
    kv_u32: cute.Tensor,
    sTokenIdx: cute.Tensor,
    base_u32: Int32,
    vecs_per_row: Int32,
    row_stride_128b: Int32,
    kv_base_addr: Int32,
    num_kv: Int32,
    lane: Int32,
):
    linear = lane
    total = Int32(_MLA_TOKEN_TILE) * vecs_per_row
    while linear < total:
        row = linear // vecs_per_row
        vec_idx = linear - row * vecs_per_row
        token_idx = Int32(sTokenIdx[row])
        dst_addr = _smem_addr_from_b128_offset(
            kv_base_addr,
            _permuted_offset_128b(row, vec_idx, row_stride_128b),
        )
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
        if token_idx >= Int32(0) and token_idx < num_kv:
            src_u32 = base_u32 + vec_idx * Int32(4)
            v0 = Uint32(kv_u32[token_idx, src_u32 + Int32(0)])
            v1 = Uint32(kv_u32[token_idx, src_u32 + Int32(1)])
            v2 = Uint32(kv_u32[token_idx, src_u32 + Int32(2)])
            v3 = Uint32(kv_u32[token_idx, src_u32 + Int32(3)])
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(_MLA_WARP_THREADS)


@cute.jit
def _stage_kv_u32_block_async(
    kv_u32: cute.Tensor,
    sTokenIdx: cute.Tensor,
    base_u32: Int32,
    vecs_per_row: Int32,
    row_stride_128b: Int32,
    kv_base_addr: Int32,
    num_kv: Int32,
    lane: Int32,
):
    linear = lane
    total = Int32(_MLA_TOKEN_TILE) * vecs_per_row
    row_stride_u32 = Int32(kv_u32.shape[1])
    while linear < total:
        row = linear // vecs_per_row
        vec_idx = linear - row * vecs_per_row
        token_idx = Int32(sTokenIdx[row])
        dst_addr = _smem_addr_from_b128_offset(
            kv_base_addr,
            _permuted_offset_128b(row, vec_idx, row_stride_128b),
        )
        valid = Int32(0)
        safe_token_idx = Int32(0)
        if token_idx >= Int32(0):
            if token_idx < num_kv:
                valid = Int32(1)
                safe_token_idx = token_idx
        src_u32 = base_u32 + vec_idx * Int32(4)
        _cp_async_load_128b_pred(
            dst_addr,
            get_ptr_as_int64(kv_u32, safe_token_idx * row_stride_u32 + src_u32),
            valid,
        )
        linear += Int32(_MLA_WARP_THREADS)


@cute.jit
def _stage_kv_bf16_block(
    kv_u32: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    base_u32: Int32,
    vecs_per_row: Int32,
    row_stride_128b: Int32,
    kv_base_addr: Int32,
    num_kv: Int32,
    lane: Int32,
):
    linear = lane
    total = Int32(_MLA_TOKEN_TILE) * vecs_per_row
    while linear < total:
        row = linear // vecs_per_row
        vec_idx = linear - row * vecs_per_row
        token_idx = Int32(sTokenIdx[row])
        dst_addr = _smem_addr_from_b128_offset(
            kv_base_addr,
            _permuted_offset_128b(row, vec_idx, row_stride_128b),
        )
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
        if token_idx >= Int32(0) and token_idx < num_kv:
            scale_bf2 = broadcast_f32_to_bfloat2(Float32(sScale[row]))
            src_u32 = base_u32 + vec_idx * Int32(2)
            raw0 = Uint32(kv_u32[token_idx, src_u32 + Int32(0)])
            raw1 = Uint32(kv_u32[token_idx, src_u32 + Int32(1)])
            v0, v1 = fp8x4_e4m3_to_bfloat2x2(raw0)
            v2, v3 = fp8x4_e4m3_to_bfloat2x2(raw1)
            v0 = bfloat2_mul(v0, scale_bf2)
            v1 = bfloat2_mul(v1, scale_bf2)
            v2 = bfloat2_mul(v2, scale_bf2)
            v3 = bfloat2_mul(v3, scale_bf2)
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(_MLA_WARP_THREADS)


@cute.jit
def _literal_qk_mma_into_sfrag_mxfp8_raw(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane: Int32,
    row_base: Int32,
    num_mma_q: Int32,
    num_mma_kv: Int32,
    num_mma_d_qk: Int32,
    upcast_stride_q: Int32,
    upcast_stride_k: Int32,
):
    unit_scale = Uint32(0x7F7F7F7F)
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    q_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        upcast_stride_q,
    )
    k_offset = _permuted_offset_128b(
        row_base + Int32(8) * (lane // Int32(16)) + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        upcast_stride_k,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        a_regs_k0 = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        q_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )

        q_offset_cur = q_offset
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset_cur))
            a_regs_k0[mma_q, 0] = a0
            a_regs_k0[mma_q, 1] = a1
            a_regs_k0[mma_q, 2] = a2
            a_regs_k0[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, upcast_stride_q)

        mma_d0 = mma_pair * 2
        q_offset_mid = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )
        q_offset_cur = q_offset_mid
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset_cur))
            q_regs[mma_q, 0] = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 0]) & mask16) | (
                (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 2]) & mask16) << shift16
            )
            q_regs[mma_q, 1] = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 1]) & mask16) | (
                (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 3]) & mask16) << shift16
            )
            q_regs[mma_q, 2] = (cvt_bf16x2_to_e4m3x2(a0) & mask16) | (
                (cvt_bf16x2_to_e4m3x2(a2) & mask16) << shift16
            )
            q_regs[mma_q, 3] = (cvt_bf16x2_to_e4m3x2(a1) & mask16) | (
                (cvt_bf16x2_to_e4m3x2(a3) & mask16) << shift16
            )
            q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, upcast_stride_q)
        q_offset = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0 + 1) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)
            k_offset_cur = _advance_offset_by_row_128b(k_offset_cur, 16, upcast_stride_k)

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    q_regs[mma_q, 0],
                    q_regs[mma_q, 1],
                    q_regs[mma_q, 2],
                    q_regs[mma_q, 3],
                    b0_k0,
                    b0_k1,
                    unit_scale,
                    unit_scale,
                )
                d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    q_regs[mma_q, 0],
                    q_regs[mma_q, 1],
                    q_regs[mma_q, 2],
                    q_regs[mma_q, 3],
                    b1_k0,
                    b1_k1,
                    unit_scale,
                    unit_scale,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            num_mma_kv * 16 * upcast_stride_k
        )


@cute.jit
def _literal_qk_mma_into_sfrag_bf16(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane: Int32,
    row_base: Int32,
    num_mma_q: Int32,
    num_mma_kv: Int32,
    num_mma_d_qk: Int32,
    upcast_stride_q: Int32,
    upcast_stride_k: Int32,
):
    for mma_d in cutlass.range_constexpr(num_mma_d_qk):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            q_row = lane % Int32(16)
            q_col = mma_d * 2 + lane // Int32(16)
            q_offset = _permuted_offset_128b(q_row, q_col, upcast_stride_q)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset))
            a_regs[mma_q, 0] = a0
            a_regs[mma_q, 1] = a1
            a_regs[mma_q, 2] = a2
            a_regs[mma_q, 3] = a3

        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            k_row = row_base + mma_kv * 16 + Int32(8) * (lane // Int32(16)) + lane % Int32(8)
            k_col = mma_d * 2 + (lane % Int32(16)) // Int32(8)
            k_offset = _permuted_offset_128b(k_row, k_col, upcast_stride_k)
            b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(k_base_addr, k_offset))

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7


@cute.jit
def _zero_score_frag(score_frag: cute.Tensor):
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        for reg_id in cutlass.range_constexpr(8):
            score_frag[0, mma_kv, reg_id] = Float32(0.0)


@cute.jit
def _zero_output_frag(o_frag: cute.Tensor):
    for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
        for reg_id in cutlass.range_constexpr(8):
            o_frag[0, mma_d, reg_id] = Float32(0.0)


@cute.jit
def _accumulate_scaled_score_frag(
    dst_frag: cute.Tensor,
    src_frag: cute.Tensor,
    sScale: cute.Tensor,
    scale_base: Int32,
    lane: Int32,
):
    lane_pair_base = Int32(2) * (lane % Int32(4))
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        for reg_id in cutlass.range_constexpr(8):
            token_local = (
                mma_kv * 16
                + lane_pair_base
                + Int32(8) * (reg_id // 4)
                + Int32(reg_id % 2)
            )
            dst_frag[0, mma_kv, reg_id] = Float32(
                dst_frag[0, mma_kv, reg_id]
                + src_frag[0, mma_kv, reg_id] * Float32(sScale[scale_base + token_local])
            )


@cute.jit
def _compute_score_tile_scaled(
    score_frag: cute.Tensor,
    q_u32: cute.Tensor,
    kv_rows_u32: cute.Tensor,
    kv_scales: cute.Tensor,
    page_table_1: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    q_base_addr: Int32,
    kv_base_addr: Int32,
    q_idx: Int32,
    head_tile_start: Int32,
    token_base: Int32,
    token_end: Int32,
    sm_scale_log2: Float32,
    lane: Int32,
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    num_heads = Int32(q_u32.shape[1])
    num_kv = Int32(kv_rows_u32.shape[0])
    tile_tokens = token_end - token_base

    _stage_token_indices(page_table_1, sTokenIdx, q_idx, token_base, token_end, lane)
    cute.arch.sync_threads()

    _zero_score_frag(score_frag)
    frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))

    for group_idx in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
        _stage_q_u32_block(
            q_u32,
            q_idx,
            head_tile_start,
            Int32(group_idx * _MLA_NOPE_GROUP_Q_U32),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            q_base_addr,
            lane,
        )
        _stage_token_scales(kv_scales, sTokenIdx, sScale, Int32(group_idx), num_kv, lane)
        _stage_kv_u32_block(
            kv_rows_u32,
            sTokenIdx,
            Int32(_MLA_NOPE_U32_OFFSET + group_idx * _MLA_NOPE_GROUP_KV_U32),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            kv_base_addr,
            num_kv,
            lane,
        )
        cute.arch.sync_threads()

        frag_tmp = cute.make_rmem_tensor(frag_layout, Float32)
        _zero_score_frag(frag_tmp)
        if cutlass.const_expr(os.environ.get("B12X_MLA_DEBUG_QK_BF16", "0") == "1"):
            _stage_kv_bf16_block(
                kv_rows_u32,
                sTokenIdx,
                sScale,
                Int32(_MLA_NOPE_U32_OFFSET + group_idx * _MLA_NOPE_GROUP_KV_U32),
                Int32(_MLA_NOPE_GROUP_KV_BF16_VECS),
                Int32(_MLA_NOPE_GROUP_KV_BF16_VECS),
                kv_base_addr,
                num_kv,
                lane,
            )
            cute.arch.sync_threads()
            _literal_qk_mma_into_sfrag_bf16(
                frag_tmp,
                q_base_addr,
                kv_base_addr,
                lane,
                Int32(0),
                Int32(1),
                Int32(_MLA_NUM_MMA_KV),
                Int32(_MLA_NOPE_QK_NUM_MMA_D),
                Int32(_MLA_NOPE_GROUP_Q_VECS),
                Int32(_MLA_NOPE_GROUP_KV_BF16_VECS),
            )
            for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
                for reg_id in cutlass.range_constexpr(8):
                    score_frag[0, mma_kv, reg_id] = Float32(
                        score_frag[0, mma_kv, reg_id] + frag_tmp[0, mma_kv, reg_id]
                    )
        else:
            _literal_qk_mma_into_sfrag_mxfp8_raw(
                frag_tmp,
                q_base_addr,
                kv_base_addr,
                lane,
                Int32(0),
                Int32(1),
                Int32(_MLA_NUM_MMA_KV),
                Int32(_MLA_NOPE_QK_NUM_MMA_D),
                Int32(_MLA_NOPE_GROUP_Q_VECS),
                Int32(_MLA_NOPE_GROUP_KV_VECS),
            )
            _accumulate_scaled_score_frag(score_frag, frag_tmp, sScale, Int32(0), lane)
        cute.arch.sync_threads()

    _stage_q_u32_block(
        q_u32,
        q_idx,
        head_tile_start,
        Int32(_MLA_NOPE_DIM // 2),
        Int32(_MLA_ROPE_VECS),
        Int32(_MLA_ROPE_VECS),
        q_base_addr,
        lane,
    )
    _stage_kv_u32_block(
        kv_rows_u32,
        sTokenIdx,
        Int32(_MLA_ROPE_U32_OFFSET),
        Int32(_MLA_ROPE_VECS),
        Int32(_MLA_ROPE_VECS),
        kv_base_addr,
        num_kv,
        lane,
    )
    cute.arch.sync_threads()

    frag_rope = cute.make_rmem_tensor(frag_layout, Float32)
    _zero_score_frag(frag_rope)
    _literal_qk_mma_into_sfrag_bf16(
        frag_rope,
        q_base_addr,
        kv_base_addr,
        lane,
        Int32(0),
        Int32(1),
        Int32(_MLA_NUM_MMA_KV),
        Int32(_MLA_QK_NUM_MMA_D),
        Int32(_MLA_ROPE_VECS),
        Int32(_MLA_ROPE_VECS),
    )
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        for reg_id in cutlass.range_constexpr(8):
            score_frag[0, mma_kv, reg_id] = Float32(score_frag[0, mma_kv, reg_id] + frag_rope[0, mma_kv, reg_id])

    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        for reg_id in cutlass.range_constexpr(8):
            row_slot = (reg_id % 4) // 2
            head_local = lane_group + Int32(8) * row_slot
            head_idx = head_tile_start + head_local
            token_local = (
                mma_kv * 16
                + lane_pair_base
                + Int32(8) * (reg_id // 4)
                + Int32(reg_id % 2)
            )
            token_idx = Int32(sTokenIdx[token_local])
            valid = token_local < tile_tokens
            if valid:
                valid = valid and token_idx >= Int32(0)
            if valid:
                valid = valid and token_idx < num_kv
            if valid:
                valid = valid and head_idx < num_heads
            score_frag[0, mma_kv, reg_id] = (
                Float32(score_frag[0, mma_kv, reg_id] * sm_scale_log2)
                if valid
                else Float32(-Float32.inf)
            )
    cute.arch.sync_threads()


@cute.jit
def _update_softmax_stats_b2(
    score_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    o_rescale_frag: cute.Tensor,
):
    for row_slot in cutlass.range_constexpr(2):
        m_prev = Float32(m_frag[0, row_slot])
        m_new = Float32(m_prev)
        for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
            m_local = attention_ops.fmax(
                attention_ops.fmax(
                    score_frag[0, mma_kv, row_slot * 2 + 0],
                    score_frag[0, mma_kv, row_slot * 2 + 1],
                ),
                attention_ops.fmax(
                    score_frag[0, mma_kv, row_slot * 2 + 4],
                    score_frag[0, mma_kv, row_slot * 2 + 5],
                ),
            )
            m_new = attention_ops.fmax(m_new, m_local)
        m_new = attention_ops.fmax(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=2))
        m_new = attention_ops.fmax(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=1))

        scale_term = (
            Float32(1.0)
            if m_prev == -Float32.inf
            else _exp2_approx_ftz_f32(m_prev - m_new)
        )
        o_rescale_frag[0, row_slot] = scale_term
        # d_frag is stored as the 4-lane reduced total on every lane in the row group.
        # Divide by 4 before the next reduction so we do not re-count the prior state.
        d_acc = Float32(d_frag[0, row_slot] * scale_term * Float32(0.25))
        for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
            p0 = (
                Float32(0.0)
                if m_new == -Float32.inf
                else _exp2_approx_ftz_f32(score_frag[0, mma_kv, row_slot * 2 + 0] - m_new)
            )
            p1 = (
                Float32(0.0)
                if m_new == -Float32.inf
                else _exp2_approx_ftz_f32(score_frag[0, mma_kv, row_slot * 2 + 1] - m_new)
            )
            p2 = (
                Float32(0.0)
                if m_new == -Float32.inf
                else _exp2_approx_ftz_f32(score_frag[0, mma_kv, row_slot * 2 + 4] - m_new)
            )
            p3 = (
                Float32(0.0)
                if m_new == -Float32.inf
                else _exp2_approx_ftz_f32(score_frag[0, mma_kv, row_slot * 2 + 5] - m_new)
            )
            d_acc = Float32(d_acc + p0 + p1 + p2 + p3)
        d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=2))
        d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=1))
        m_frag[0, row_slot] = Float32(m_new)
        d_frag[0, row_slot] = Float32(d_acc)


@cute.jit
def _fill_normalized_p_frag_from_scores(
    p_frag: cute.Tensor,
    score_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
):
    del d_frag
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        for row_slot in cutlass.range_constexpr(2):
            m_scaled = Float32(m_frag[0, row_slot])
            p0 = (
                Float32(0.0)
                if m_scaled == -Float32.inf
                else Float32(_exp2_approx_ftz_f32(score_frag[0, mma_kv, row_slot * 2 + 0] - m_scaled))
            )
            p1 = (
                Float32(0.0)
                if m_scaled == -Float32.inf
                else Float32(_exp2_approx_ftz_f32(score_frag[0, mma_kv, row_slot * 2 + 1] - m_scaled))
            )
            p2 = (
                Float32(0.0)
                if m_scaled == -Float32.inf
                else Float32(_exp2_approx_ftz_f32(score_frag[0, mma_kv, row_slot * 2 + 4] - m_scaled))
            )
            p3 = (
                Float32(0.0)
                if m_scaled == -Float32.inf
                else Float32(_exp2_approx_ftz_f32(score_frag[0, mma_kv, row_slot * 2 + 5] - m_scaled))
            )
            p_frag[0, mma_kv, row_slot + 0] = pack_f32x2_to_bfloat2(p0, p1)
            p_frag[0, mma_kv, row_slot + 2] = pack_f32x2_to_bfloat2(p2, p3)


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp8_scaled(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    sScale: cute.Tensor,
    scale_base: Int32,
    pv_scale: Float32,
    lane: Int32,
):
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    del pv_scale

    scale01_k0 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(0)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(1)]),
    )
    scale89_k0 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(8)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(9)]),
    )
    scale01_k1 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(16)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(17)]),
    )
    scale89_k1 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(24)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(25)]),
    )

    a00 = bfloat2_mul(p_frag[0, 0, 0], scale01_k0)
    a10 = bfloat2_mul(p_frag[0, 0, 1], scale01_k0)
    a80 = bfloat2_mul(p_frag[0, 0, 2], scale89_k0)
    a90 = bfloat2_mul(p_frag[0, 0, 3], scale89_k0)
    a16 = bfloat2_mul(p_frag[0, 1, 0], scale01_k1)
    a17 = bfloat2_mul(p_frag[0, 1, 1], scale01_k1)
    a24 = bfloat2_mul(p_frag[0, 1, 2], scale89_k1)
    a25 = bfloat2_mul(p_frag[0, 1, 3], scale89_k1)

    sfa0 = cvt_f32_to_ue8m0(
        bfloat2_hmax_to_f32(
            bfloat2_hmax2(
                bfloat2_hmax2(
                    bfloat2_hmax2(
                        bfloat2_habs2(a00),
                        bfloat2_habs2(a10),
                    ),
                    bfloat2_hmax2(
                        bfloat2_habs2(a80),
                        bfloat2_habs2(a90),
                    ),
                ),
                bfloat2_hmax2(
                    bfloat2_hmax2(
                        bfloat2_habs2(a16),
                        bfloat2_habs2(a17),
                    ),
                    bfloat2_hmax2(
                        bfloat2_habs2(a24),
                        bfloat2_habs2(a25),
                    ),
                ),
            )
        )
    )
    inv_sfa0 = broadcast_f32_to_bfloat2(ue8m0_to_output_scale(sfa0))
    sfa = Uint32(sfa0)
    unit_scale = Uint32(0x7F)

    a_regs = cute.make_rmem_tensor(
        cute.make_layout((1, 4), stride=(4, 1)),
        Uint32,
    )
    a_regs[0, 0] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a00, inv_sfa0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a80, inv_sfa0)) & mask16) << shift16
    )
    a_regs[0, 1] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a10, inv_sfa0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a90, inv_sfa0)) & mask16) << shift16
    )
    a_regs[0, 2] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a16, inv_sfa0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a24, inv_sfa0)) & mask16) << shift16
    )
    a_regs[0, 3] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a17, inv_sfa0)) & mask16) | (
        (cvt_bf16x2_to_e4m3x2(bfloat2_mul(a25, inv_sfa0)) & mask16) << shift16
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
def _literal_pv_mma_into_ofrag_fp8_raw_scaled(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    sScale: cute.Tensor,
    scale_base: Int32,
    lane: Int32,
):
    lane_pair_base = Int32(2) * (lane % Int32(4))

    scale01_k0 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(0)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(1)]),
    )
    scale89_k0 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(8)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(9)]),
    )
    scale01_k1 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(16)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(17)]),
    )
    scale89_k1 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(24)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(25)]),
    )

    v_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        Int32(_MLA_NOPE_GROUP_KV_VECS),
    )
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((1, 4), stride=(4, 1)),
            Uint32,
        )
        scale01 = scale01_k0 if mma_kv == 0 else scale01_k1
        scale89 = scale89_k0 if mma_kv == 0 else scale89_k1
        a_regs[0, 0] = bfloat2_mul(p_frag[0, mma_kv, 0], scale01)
        a_regs[0, 1] = bfloat2_mul(p_frag[0, mma_kv, 1], scale01)
        a_regs[0, 2] = bfloat2_mul(p_frag[0, mma_kv, 2], scale89)
        a_regs[0, 3] = bfloat2_mul(p_frag[0, mma_kv, 3], scale89)

        v_offset_cur = v_offset
        for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
            b_f8_0 = Uint32(0)
            b_f8_1 = Uint32(0)
            if mma_d % 2 == 0:
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_trans_left_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_cur)
                )
            else:
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_trans_right_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_cur)
                )
            b_f8_0 = frag_layout_swizzle_16b_to_8b_trans(b_f8_0)
            b_f8_1 = frag_layout_swizzle_16b_to_8b_trans(b_f8_1)
            b0, b1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1)
            tmp = b1
            b1 = b2
            b2 = tmp
            if mma_d % 2 == 1:
                v_offset_cur = _advance_offset_by_column_128b_2(v_offset_cur, mma_d // 2)
            d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                o_frag[0, mma_d, 0],
                o_frag[0, mma_d, 1],
                o_frag[0, mma_d, 2],
                o_frag[0, mma_d, 3],
                o_frag[0, mma_d, 4],
                o_frag[0, mma_d, 5],
                o_frag[0, mma_d, 6],
                o_frag[0, mma_d, 7],
                a_regs[0, 0],
                a_regs[0, 1],
                a_regs[0, 2],
                a_regs[0, 3],
                b0,
                b1,
                b2,
                b3,
            )
            o_frag[0, mma_d, 0] = d0
            o_frag[0, mma_d, 1] = d1
            o_frag[0, mma_d, 2] = d2
            o_frag[0, mma_d, 3] = d3
            o_frag[0, mma_d, 4] = d4
            o_frag[0, mma_d, 5] = d5
            o_frag[0, mma_d, 6] = d6
            o_frag[0, mma_d, 7] = d7
        v_offset = _advance_offset_by_row_128b(v_offset_cur, 16, Int32(_MLA_NOPE_GROUP_KV_VECS)) - Int32(_MLA_VO_NUM_MMA_D)
    v_offset -= Int32(16 * _MLA_NUM_MMA_KV * _MLA_NOPE_GROUP_KV_VECS)


@cute.jit
def _store_output_group(
    out_tensor: cute.Tensor,
    o_frag: cute.Tensor,
    d_frag: cute.Tensor,
    out_row_idx: Int32,
    head_tile_start: Int32,
    group_idx: Int32,
    lane: Int32,
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    for row_slot in cutlass.range_constexpr(2):
        head_local = lane_group + Int32(8) * row_slot
        head_idx = head_tile_start + head_local
        if head_idx < Int32(out_tensor.shape[1]):
            reg_base = row_slot * 2
            inv_d = (
                Float32(0.0)
                if d_frag[0, row_slot] == Float32(0.0)
                else Float32(1.0) / Float32(d_frag[0, row_slot])
            )
            for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
                dim_base = group_idx * Int32(_MLA_GROUP_SIZE) + mma_d * Int32(16) + lane_pair_base
                out_tensor[out_row_idx, head_idx, dim_base + Int32(0)] = Float32(
                    o_frag[0, mma_d, reg_base + 0] * inv_d
                ).to(out_tensor.element_type)
                out_tensor[out_row_idx, head_idx, dim_base + Int32(1)] = Float32(
                    o_frag[0, mma_d, reg_base + 1] * inv_d
                ).to(out_tensor.element_type)
                out_tensor[out_row_idx, head_idx, dim_base + Int32(8)] = Float32(
                    o_frag[0, mma_d, reg_base + 4] * inv_d
                ).to(out_tensor.element_type)
                out_tensor[out_row_idx, head_idx, dim_base + Int32(9)] = Float32(
                    o_frag[0, mma_d, reg_base + 5] * inv_d
                ).to(out_tensor.element_type)


@cute.jit
def _store_output_group_chunked(
    out_tensor: cute.Tensor,
    o_frag: cute.Tensor,
    d_frag: cute.Tensor,
    out_row_idx: Int32,
    out_chunk_idx: Int32,
    head_tile_start: Int32,
    group_idx: Int32,
    lane: Int32,
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    for row_slot in cutlass.range_constexpr(2):
        head_local = lane_group + Int32(8) * row_slot
        head_idx = head_tile_start + head_local
        if head_idx < Int32(out_tensor.shape[1]):
            reg_base = row_slot * 2
            inv_d = (
                Float32(0.0)
                if d_frag[0, row_slot] == Float32(0.0)
                else Float32(1.0) / Float32(d_frag[0, row_slot])
            )
            for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
                dim_base = group_idx * Int32(_MLA_GROUP_SIZE) + mma_d * Int32(16) + lane_pair_base
                out_tensor[out_row_idx, head_idx, out_chunk_idx, dim_base + Int32(0)] = Float32(
                    o_frag[0, mma_d, reg_base + 0] * inv_d
                ).to(out_tensor.element_type)
                out_tensor[out_row_idx, head_idx, out_chunk_idx, dim_base + Int32(1)] = Float32(
                    o_frag[0, mma_d, reg_base + 1] * inv_d
                ).to(out_tensor.element_type)
                out_tensor[out_row_idx, head_idx, out_chunk_idx, dim_base + Int32(8)] = Float32(
                    o_frag[0, mma_d, reg_base + 4] * inv_d
                ).to(out_tensor.element_type)
                out_tensor[out_row_idx, head_idx, out_chunk_idx, dim_base + Int32(9)] = Float32(
                    o_frag[0, mma_d, reg_base + 5] * inv_d
                ).to(out_tensor.element_type)


@cute.jit
def _store_partial_lse(
    tmp_lse: cute.Tensor,
    partial_idx: Int32,
    head_tile_start: Int32,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    lane: Int32,
):
    if lane % Int32(4) == Int32(0):
        lane_group = lane // Int32(4)
        for row_slot in cutlass.range_constexpr(2):
            head_local = lane_group + Int32(8) * row_slot
            head_idx = head_tile_start + head_local
            if head_idx < Int32(tmp_lse.shape[1]):
                row_lse = Float32(-Float32.inf)
                if m_frag[0, row_slot] != -Float32.inf:
                    row_lse = Float32(m_frag[0, row_slot] + _log2_approx_ftz_f32(d_frag[0, row_slot]))
                tmp_lse[partial_idx, head_idx] = row_lse


@cute.jit
def _store_partial_lse_chunked(
    tmp_lse: cute.Tensor,
    out_row_idx: Int32,
    out_chunk_idx: Int32,
    head_tile_start: Int32,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    lane: Int32,
):
    if lane % Int32(4) == Int32(0):
        lane_group = lane // Int32(4)
        for row_slot in cutlass.range_constexpr(2):
            head_local = lane_group + Int32(8) * row_slot
            head_idx = head_tile_start + head_local
            if head_idx < Int32(tmp_lse.shape[1]):
                row_lse = Float32(-Float32.inf)
                if m_frag[0, row_slot] != -Float32.inf:
                    row_lse = Float32(
                        m_frag[0, row_slot] + _log2_approx_ftz_f32(d_frag[0, row_slot])
                    )
                tmp_lse[out_row_idx, head_idx, out_chunk_idx] = row_lse


@cute.jit
def _accumulate_scaled_output_frag(
    dst_frag: cute.Tensor,
    src_frag: cute.Tensor,
    scale: Float32,
):
    for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
        for reg_id in cutlass.range_constexpr(8):
            dst_frag[0, mma_d, reg_id] = Float32(dst_frag[0, mma_d, reg_id] + src_frag[0, mma_d, reg_id] * scale)


@cute.jit
def _run_staged_pv_group_into_target(
    target_frag: cute.Tensor,
    p_frag: cute.Tensor,
    sScale: cute.Tensor,
    group_base_addr: Int32,
    scale_base: Int32,
    tile_pv_scale: Float32,
    lane: Int32,
):
    if cutlass.const_expr(
        os.environ.get("B12X_MLA_ENABLE_MXFP8_PV", "0") != "1"
        or os.environ.get("B12X_MLA_DEBUG_PV_BF16", "0") == "1"
    ):
        _literal_pv_mma_into_ofrag_fp8_raw_scaled(
            target_frag,
            p_frag,
            group_base_addr,
            sScale,
            scale_base,
            lane,
        )
    else:
        _literal_pv_mma_into_ofrag_mxfp8_scaled(
            target_frag,
            p_frag,
            group_base_addr,
            sScale,
            scale_base,
            tile_pv_scale,
            lane,
        )


@cute.jit
def _accumulate_pv_groups_from_p_frag(
    o_frag0: cute.Tensor,
    o_frag1: cute.Tensor,
    o_frag2: cute.Tensor,
    o_frag3: cute.Tensor,
    p_frag: cute.Tensor,
    kv_rows_u32: cute.Tensor,
    kv_scales: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    kv_base_addr: Int32,
    lane: Int32,
):
    for block_offset in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
        group_idx = Int32(block_offset)
        _stage_token_scales(
            kv_scales,
            sTokenIdx,
            sScale,
            group_idx,
            Int32(kv_rows_u32.shape[0]),
            lane,
        )
        tile_output_scale = _warp_allreduce_max(Float32(sScale[lane]))
        tile_pv_scale = (
            Float32(0.0)
            if tile_output_scale == Float32(0.0)
            else cute.arch.rcp_approx(tile_output_scale)
        )
        _stage_kv_u32_block(
            kv_rows_u32,
            sTokenIdx,
            Int32(_MLA_NOPE_U32_OFFSET) + group_idx * Int32(_MLA_NOPE_GROUP_KV_U32),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            kv_base_addr,
            Int32(kv_rows_u32.shape[0]),
            lane,
        )
        cute.arch.sync_threads()
        tile_o_frag = cute.make_rmem_tensor(
            cute.make_layout((1, _MLA_VO_NUM_MMA_D, 8), stride=(_MLA_VO_NUM_MMA_D * 8, 8, 1)),
            Float32,
        )
        _zero_output_frag(tile_o_frag)
        if cutlass.const_expr(
            os.environ.get("B12X_MLA_ENABLE_MXFP8_PV", "0") != "1"
            or os.environ.get("B12X_MLA_DEBUG_PV_BF16", "0") == "1"
        ):
            _literal_pv_mma_into_ofrag_fp8_raw_scaled(
                tile_o_frag,
                p_frag,
                kv_base_addr,
                sScale,
                Int32(0),
                lane,
            )
            accum_scale = Float32(1.0)
        else:
            _literal_pv_mma_into_ofrag_mxfp8_scaled(
                tile_o_frag,
                p_frag,
                kv_base_addr,
                sScale,
                Int32(0),
                tile_pv_scale,
                lane,
            )
            accum_scale = Float32(1.0)
        if cutlass.const_expr(block_offset == 0):
            _accumulate_scaled_output_frag(o_frag0, tile_o_frag, accum_scale)
        elif cutlass.const_expr(block_offset == 1):
            _accumulate_scaled_output_frag(o_frag1, tile_o_frag, accum_scale)
        elif cutlass.const_expr(block_offset == 2):
            _accumulate_scaled_output_frag(o_frag2, tile_o_frag, accum_scale)
        else:
            _accumulate_scaled_output_frag(o_frag3, tile_o_frag, accum_scale)
        cute.arch.sync_threads()


@cute.jit
def _compute_score_tile_scaled_from_staged_nope(
    score_frag: cute.Tensor,
    q_u32: cute.Tensor,
    kv_rows_u32: cute.Tensor,
    kv_scales: cute.Tensor,
    page_table_1: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    q_base_addr: Int32,
    kv_nope_base_addr: Int32,
    kv_rope_base_addr: Int32,
    q_idx: Int32,
    head_tile_start: Int32,
    token_base: Int32,
    token_end: Int32,
    sm_scale_log2: Float32,
    lane: Int32,
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    num_heads = Int32(q_u32.shape[1])
    num_kv = Int32(kv_rows_u32.shape[0])
    tile_tokens = token_end - token_base

    _stage_token_indices(page_table_1, sTokenIdx, q_idx, token_base, token_end, lane)
    cute.arch.sync_threads()
    _stage_all_token_scales(kv_scales, sTokenIdx, sScale, num_kv, lane)
    for block_offset in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
        group_idx = Int32(block_offset)
        _stage_q_u32_block_async(
            q_u32,
            q_idx,
            head_tile_start,
            group_idx * Int32(_MLA_NOPE_GROUP_Q_U32),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            q_base_addr + group_idx * Int32(_MLA_Q_NOPE_STAGE_BYTES),
            lane,
        )
    _stage_q_u32_block_async(
        q_u32,
        q_idx,
        head_tile_start,
        Int32(_MLA_NOPE_DIM // 2),
        Int32(_MLA_ROPE_VECS),
        Int32(_MLA_ROPE_VECS),
        q_base_addr + Int32(_MLA_SCALE_GROUPS * _MLA_Q_NOPE_STAGE_BYTES),
        lane,
    )
    cute.arch.cp_async_commit_group()
    for block_offset in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
        group_idx = Int32(block_offset)
        _stage_kv_u32_block_async(
            kv_rows_u32,
            sTokenIdx,
            Int32(_MLA_NOPE_U32_OFFSET) + group_idx * Int32(_MLA_NOPE_GROUP_KV_U32),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            kv_nope_base_addr + group_idx * Int32(_MLA_KV_NOPE_STAGE_BYTES),
            num_kv,
            lane,
        )
        cute.arch.cp_async_commit_group()
    _stage_kv_u32_block_async(
        kv_rows_u32,
        sTokenIdx,
        Int32(_MLA_ROPE_U32_OFFSET),
        Int32(_MLA_ROPE_VECS),
        Int32(_MLA_ROPE_VECS),
        kv_rope_base_addr,
        num_kv,
        lane,
    )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    _zero_score_frag(score_frag)
    frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))
    for block_offset in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
        group_idx = Int32(block_offset)
        frag_tmp = cute.make_rmem_tensor(frag_layout, Float32)
        _zero_score_frag(frag_tmp)
        _literal_qk_mma_into_sfrag_mxfp8_raw(
            frag_tmp,
            q_base_addr + group_idx * Int32(_MLA_Q_NOPE_STAGE_BYTES),
            kv_nope_base_addr + group_idx * Int32(_MLA_KV_NOPE_STAGE_BYTES),
            lane,
            Int32(0),
            Int32(1),
            Int32(_MLA_NUM_MMA_KV),
            Int32(_MLA_NOPE_QK_NUM_MMA_D),
            Int32(_MLA_NOPE_GROUP_Q_VECS),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
        )
        _accumulate_scaled_score_frag(
            score_frag,
            frag_tmp,
            sScale,
            group_idx * Int32(_MLA_TOKEN_TILE),
            lane,
        )

    frag_rope = cute.make_rmem_tensor(frag_layout, Float32)
    _zero_score_frag(frag_rope)
    _literal_qk_mma_into_sfrag_bf16(
        frag_rope,
        q_base_addr + Int32(_MLA_SCALE_GROUPS * _MLA_Q_NOPE_STAGE_BYTES),
        kv_rope_base_addr,
        lane,
        Int32(0),
        Int32(1),
        Int32(_MLA_NUM_MMA_KV),
        Int32(_MLA_QK_NUM_MMA_D),
        Int32(_MLA_ROPE_VECS),
        Int32(_MLA_ROPE_VECS),
    )
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        for reg_id in cutlass.range_constexpr(8):
            score_frag[0, mma_kv, reg_id] = Float32(score_frag[0, mma_kv, reg_id] + frag_rope[0, mma_kv, reg_id])

    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        for reg_id in cutlass.range_constexpr(8):
            row_slot = (reg_id % 4) // 2
            head_local = lane_group + Int32(8) * row_slot
            head_idx = head_tile_start + head_local
            token_local = (
                mma_kv * 16
                + lane_pair_base
                + Int32(8) * (reg_id // 4)
                + Int32(reg_id % 2)
            )
            token_idx = Int32(sTokenIdx[token_local])
            valid = token_local < tile_tokens
            if valid:
                valid = valid and token_idx >= Int32(0)
            if valid:
                valid = valid and token_idx < num_kv
            if valid:
                valid = valid and head_idx < num_heads
            score_frag[0, mma_kv, reg_id] = (
                Float32(score_frag[0, mma_kv, reg_id] * sm_scale_log2)
                if valid
                else Float32(-Float32.inf)
            )
    cute.arch.sync_threads()


@cute.jit
def _accumulate_pv_groups_from_p_frag_staged(
    o_frag0: cute.Tensor,
    o_frag1: cute.Tensor,
    o_frag2: cute.Tensor,
    o_frag3: cute.Tensor,
    p_frag: cute.Tensor,
    sScale: cute.Tensor,
    kv_nope_base_addr: Int32,
    lane: Int32,
):
    for block_offset in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
        group_idx = Int32(block_offset)
        scale_base = group_idx * Int32(_MLA_TOKEN_TILE)
        tile_output_scale = _warp_allreduce_max(Float32(sScale[scale_base + lane]))
        tile_pv_scale = (
            Float32(0.0)
            if tile_output_scale == Float32(0.0)
            else cute.arch.rcp_approx(tile_output_scale)
        )
        group_base_addr = kv_nope_base_addr + group_idx * Int32(_MLA_KV_NOPE_STAGE_BYTES)
        if cutlass.const_expr(block_offset == 0):
            _run_staged_pv_group_into_target(
                o_frag0, p_frag, sScale, group_base_addr, scale_base, tile_pv_scale, lane
            )
        elif cutlass.const_expr(block_offset == 1):
            _run_staged_pv_group_into_target(
                o_frag1, p_frag, sScale, group_base_addr, scale_base, tile_pv_scale, lane
            )
        elif cutlass.const_expr(block_offset == 2):
            _run_staged_pv_group_into_target(
                o_frag2, p_frag, sScale, group_base_addr, scale_base, tile_pv_scale, lane
            )
        else:
            _run_staged_pv_group_into_target(
                o_frag3, p_frag, sScale, group_base_addr, scale_base, tile_pv_scale, lane
            )


@cute.jit
def _store_output_groups(
    out_tensor: cute.Tensor,
    o_frag0: cute.Tensor,
    o_frag1: cute.Tensor,
    o_frag2: cute.Tensor,
    o_frag3: cute.Tensor,
    d_frag: cute.Tensor,
    out_row_idx: Int32,
    head_tile_start: Int32,
    lane: Int32,
):
    _store_output_group(
        out_tensor,
        o_frag0,
        d_frag,
        out_row_idx,
        head_tile_start,
        Int32(0),
        lane,
    )
    _store_output_group(
        out_tensor,
        o_frag1,
        d_frag,
        out_row_idx,
        head_tile_start,
        Int32(1),
        lane,
    )
    _store_output_group(
        out_tensor,
        o_frag2,
        d_frag,
        out_row_idx,
        head_tile_start,
        Int32(2),
        lane,
    )
    _store_output_group(
        out_tensor,
        o_frag3,
        d_frag,
        out_row_idx,
        head_tile_start,
        Int32(3),
        lane,
    )


@cute.jit
def _store_output_groups_chunked(
    out_tensor: cute.Tensor,
    o_frag0: cute.Tensor,
    o_frag1: cute.Tensor,
    o_frag2: cute.Tensor,
    o_frag3: cute.Tensor,
    d_frag: cute.Tensor,
    out_row_idx: Int32,
    out_chunk_idx: Int32,
    head_tile_start: Int32,
    lane: Int32,
):
    _store_output_group_chunked(
        out_tensor,
        o_frag0,
        d_frag,
        out_row_idx,
        out_chunk_idx,
        head_tile_start,
        Int32(0),
        lane,
    )
    _store_output_group_chunked(
        out_tensor,
        o_frag1,
        d_frag,
        out_row_idx,
        out_chunk_idx,
        head_tile_start,
        Int32(1),
        lane,
    )
    _store_output_group_chunked(
        out_tensor,
        o_frag2,
        d_frag,
        out_row_idx,
        out_chunk_idx,
        head_tile_start,
        Int32(2),
        lane,
    )
    _store_output_group_chunked(
        out_tensor,
        o_frag3,
        d_frag,
        out_row_idx,
        out_chunk_idx,
        head_tile_start,
        Int32(3),
        lane,
    )


@cute.jit
def _run_one_pass_sparse_mla_tile(
    q_u32: cute.Tensor,
    kv_rows_u32: cute.Tensor,
    kv_scales: cute.Tensor,
    page_table_1: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    q_base_addr: Int32,
    kv_base_addr: Int32,
    q_idx: Int32,
    head_tile_start: Int32,
    token_start: Int32,
    token_end: Int32,
    sm_scale_log2: Float32,
    lane: Int32,
    out_tensor: cute.Tensor,
    out_row_idx: Int32,
    out_chunk_idx: Int32,
    lse_tensor: cute.Tensor | None,
):
    md_layout = cute.make_layout((1, 2), stride=(2, 1))
    frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))
    p_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 4), stride=(8, 4, 1))
    o_layout = cute.make_layout((1, _MLA_VO_NUM_MMA_D, 8), stride=(_MLA_VO_NUM_MMA_D * 8, 8, 1))

    m_frag = cute.make_rmem_tensor(md_layout, Float32)
    d_frag = cute.make_rmem_tensor(md_layout, Float32)
    o_rescale_frag = cute.make_rmem_tensor(md_layout, Float32)
    for row_slot in cutlass.range_constexpr(2):
        m_frag[0, row_slot] = Float32(-Float32.inf)
        d_frag[0, row_slot] = Float32(0.0)
        o_rescale_frag[0, row_slot] = Float32(1.0)

    o_frag0 = cute.make_rmem_tensor(o_layout, Float32)
    o_frag1 = cute.make_rmem_tensor(o_layout, Float32)
    o_frag2 = cute.make_rmem_tensor(o_layout, Float32)
    o_frag3 = cute.make_rmem_tensor(o_layout, Float32)
    _zero_output_frag(o_frag0)
    _zero_output_frag(o_frag1)
    _zero_output_frag(o_frag2)
    _zero_output_frag(o_frag3)
    if token_end - token_start <= Int32(_MLA_TOKEN_TILE):
        # Single-tile path: already single-pass, unchanged
        score_frag = cute.make_rmem_tensor(frag_layout, Float32)
        if cutlass.const_expr(os.environ.get("B12X_MLA_DEBUG_QK_BF16", "0") == "1"):
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
                q_idx,
                head_tile_start,
                token_start,
                token_end,
                sm_scale_log2,
                lane,
            )
        else:
            _compute_score_tile_scaled_from_staged_nope(
                score_frag,
                q_u32,
                kv_rows_u32,
                kv_scales,
                page_table_1,
                sTokenIdx,
                sScale,
                q_base_addr,
                kv_base_addr,
                kv_base_addr + Int32(_MLA_SCALE_GROUPS * _MLA_KV_NOPE_STAGE_BYTES),
                q_idx,
                head_tile_start,
                token_start,
                token_end,
                sm_scale_log2,
                lane,
            )
        _update_softmax_stats_b2(score_frag, m_frag, d_frag, o_rescale_frag)
        p_frag = cute.make_rmem_tensor(p_layout, Uint32)
        _fill_normalized_p_frag_from_scores(p_frag, score_frag, m_frag, d_frag)
        if cutlass.const_expr(os.environ.get("B12X_MLA_DEBUG_QK_BF16", "0") == "1"):
            _accumulate_pv_groups_from_p_frag(
                o_frag0,
                o_frag1,
                o_frag2,
                o_frag3,
                p_frag,
                kv_rows_u32,
                kv_scales,
                sTokenIdx,
                sScale,
                kv_base_addr,
                lane,
            )
        else:
            _accumulate_pv_groups_from_p_frag_staged(
                o_frag0,
                o_frag1,
                o_frag2,
                o_frag3,
                p_frag,
                sScale,
                kv_base_addr,
                lane,
            )
    else:
        # Multi-tile: single-pass with online softmax O rescaling (Flash Attention)
        token_base = token_start
        while token_base < token_end:
            tile_end = cutlass.select_(
                token_base + Int32(_MLA_TOKEN_TILE) < token_end,
                token_base + Int32(_MLA_TOKEN_TILE),
                token_end,
            )
            score_frag = cute.make_rmem_tensor(frag_layout, Float32)
            _compute_score_tile_scaled_from_staged_nope(
                score_frag,
                q_u32,
                kv_rows_u32,
                kv_scales,
                page_table_1,
                sTokenIdx,
                sScale,
                q_base_addr,
                kv_base_addr,
                kv_base_addr + Int32(_MLA_SCALE_GROUPS * _MLA_KV_NOPE_STAGE_BYTES),
                q_idx,
                head_tile_start,
                token_base,
                tile_end,
                sm_scale_log2,
                lane,
            )
            # Update softmax stats and get O rescaling factor
            _update_softmax_stats_b2(score_frag, m_frag, d_frag, o_rescale_frag)

            # Rescale O accumulator: O *= exp2(m_prev - m_new) for each row_slot
            for row_slot in cutlass.range_constexpr(2):
                rs = Float32(o_rescale_frag[0, row_slot])
                for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
                    reg_base = row_slot * 2
                    o_frag0[0, mma_d, reg_base + 0] = Float32(o_frag0[0, mma_d, reg_base + 0] * rs)
                    o_frag0[0, mma_d, reg_base + 1] = Float32(o_frag0[0, mma_d, reg_base + 1] * rs)
                    o_frag0[0, mma_d, reg_base + 4] = Float32(o_frag0[0, mma_d, reg_base + 4] * rs)
                    o_frag0[0, mma_d, reg_base + 5] = Float32(o_frag0[0, mma_d, reg_base + 5] * rs)
                    o_frag1[0, mma_d, reg_base + 0] = Float32(o_frag1[0, mma_d, reg_base + 0] * rs)
                    o_frag1[0, mma_d, reg_base + 1] = Float32(o_frag1[0, mma_d, reg_base + 1] * rs)
                    o_frag1[0, mma_d, reg_base + 4] = Float32(o_frag1[0, mma_d, reg_base + 4] * rs)
                    o_frag1[0, mma_d, reg_base + 5] = Float32(o_frag1[0, mma_d, reg_base + 5] * rs)
                    o_frag2[0, mma_d, reg_base + 0] = Float32(o_frag2[0, mma_d, reg_base + 0] * rs)
                    o_frag2[0, mma_d, reg_base + 1] = Float32(o_frag2[0, mma_d, reg_base + 1] * rs)
                    o_frag2[0, mma_d, reg_base + 4] = Float32(o_frag2[0, mma_d, reg_base + 4] * rs)
                    o_frag2[0, mma_d, reg_base + 5] = Float32(o_frag2[0, mma_d, reg_base + 5] * rs)
                    o_frag3[0, mma_d, reg_base + 0] = Float32(o_frag3[0, mma_d, reg_base + 0] * rs)
                    o_frag3[0, mma_d, reg_base + 1] = Float32(o_frag3[0, mma_d, reg_base + 1] * rs)
                    o_frag3[0, mma_d, reg_base + 4] = Float32(o_frag3[0, mma_d, reg_base + 4] * rs)
                    o_frag3[0, mma_d, reg_base + 5] = Float32(o_frag3[0, mma_d, reg_base + 5] * rs)

            # Normalize P from this tile's scores and do PV accumulation
            # V is already staged in smem from the QK async copy, so use staged PV
            p_frag = cute.make_rmem_tensor(p_layout, Uint32)
            _fill_normalized_p_frag_from_scores(p_frag, score_frag, m_frag, d_frag)
            _accumulate_pv_groups_from_p_frag_staged(
                o_frag0,
                o_frag1,
                o_frag2,
                o_frag3,
                p_frag,
                sScale,
                kv_base_addr,
                lane,
            )
            token_base = tile_end

    if cutlass.const_expr(lse_tensor is None):
        _store_output_groups(
            out_tensor,
            o_frag0,
            o_frag1,
            o_frag2,
            o_frag3,
            d_frag,
            out_row_idx,
            head_tile_start,
            lane,
        )
    else:
        _store_output_groups_chunked(
            out_tensor,
            o_frag0,
            o_frag1,
            o_frag2,
            o_frag3,
            d_frag,
            out_row_idx,
            out_chunk_idx,
            head_tile_start,
            lane,
        )
        _store_partial_lse_chunked(
            lse_tensor,
            out_row_idx,
            out_chunk_idx,
            head_tile_start,
            m_frag,
            d_frag,
            lane,
        )

def get_sparse_mla_shared_storage_cls():
    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "q_stage": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(_MLA_Q_STAGE_BYTES)],
            128,
        ],
        "kv_stage": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(_MLA_KV_STAGE_BYTES)],
            128,
        ],
        "token_idx": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, _MLA_TOKEN_TILE],
            16,
        ],
        "token_scale": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MLA_SCALE_STAGE_ELEMS],
            16,
        ],
    }
    return cute.struct(SharedStorage)


class SparseMLAKernel:
    """Single-pass sparse MLA kernel using MXFP8 MMA for nope and BF16 MMA for rope."""

    def __init__(self, head_tiles: int):
        self.head_tiles = int(head_tiles)

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        active_token_counts: cute.Tensor,
        sm_scale: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_u32,
            kv_rows_u32,
            kv_scales,
            page_table_1,
            active_token_counts,
            sm_scale,
            output,
        ).launch(
            grid=(output.shape[0], self.head_tiles, 1),
            block=[_MLA_WARP_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        active_token_counts: cute.Tensor,
        sm_scale: cute.Tensor,
        output: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        q_idx, head_tile_idx, _ = cute.arch.block_idx()
        q_idx = Int32(q_idx)
        head_tile_start = Int32(head_tile_idx * _MLA_HEADS_PER_TILE)
        token_end = _clamp_active_token_count(active_token_counts, q_idx, Int32(page_table_1.shape[1]))

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = get_sparse_mla_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        sTokenIdx = storage.token_idx.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        sScale = storage.token_scale.get_tensor(
            cute.make_layout((_MLA_SCALE_STAGE_ELEMS,), stride=(1,))
        )
        q_base_addr = shared_ptr_to_u32(storage.q_stage.data_ptr())
        kv_base_addr = shared_ptr_to_u32(storage.kv_stage.data_ptr())

        _run_one_pass_sparse_mla_tile(
            q_u32,
            kv_rows_u32,
            kv_scales,
            page_table_1,
            sTokenIdx,
            sScale,
            q_base_addr,
            kv_base_addr,
            q_idx,
            head_tile_start,
            Int32(0),
            token_end,
            Float32(sm_scale[Int32(0)] * attention_ops.LOG2_E),
            lane,
            output,
            q_idx,
            Int32(0),
            None,
        )

@lru_cache(maxsize=16)
def _build_sparse_mla_kernel_for_shape(
    traits: SparseMLATraits,
    head_tiles: int,
) -> SparseMLAKernel:
    del traits
    return SparseMLAKernel(head_tiles)


def clear_sparse_mla_kernel_cache() -> None:
    _build_sparse_mla_kernel_for_shape.cache_clear()


def _view_last_dim_as_u32(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    byte_width = tensor.shape[-1] * tensor.element_size()
    if byte_width % 4 != 0:
        raise ValueError(f"last dimension byte-width must be divisible by 4, got {byte_width}")
    byte_view = tensor.view(torch.uint8).reshape(*tensor.shape[:-1], byte_width)
    return byte_view.view(torch.uint32).reshape(*tensor.shape[:-1], byte_width // 4)


def supports_sparse_mla_kernel(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    v_head_dim: int,
) -> bool:
    return (
        select_sparse_mla_traits(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=page_table_1,
            output_dtype=q_all.dtype,
            v_head_dim=v_head_dim,
        )
        is not None
    )


def run_sparse_mla_kernel(
    *,
    q_all: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    page_table_1: torch.Tensor | None = None,
    active_token_counts: torch.Tensor | None = None,
    sm_scale: float | torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    workspace: object | None = None,
    binding: SparseMLAOnePassKernelBinding | None = None,
) -> None:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q_all", q_all),
                ("kv_cache", kv_cache),
                ("page_table_1", page_table_1),
                ("active_token_counts", active_token_counts),
                ("sm_scale", sm_scale),
                ("output", output),
                ("workspace", workspace),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_sparse_mla_kernel", extras)
        q_all = binding.q_all
        kv_cache = binding.kv_cache
        page_table_1 = binding.page_table_1
        active_token_counts = binding.active_token_counts
        sm_scale = binding.sm_scale
        output = binding.output
        workspace = binding.workspace

    q_all = _require_bound_arg(q_all, api_name="run_sparse_mla_kernel", name="q_all")
    kv_cache = _require_bound_arg(kv_cache, api_name="run_sparse_mla_kernel", name="kv_cache")
    page_table_1 = _require_bound_arg(
        page_table_1,
        api_name="run_sparse_mla_kernel",
        name="page_table_1",
    )
    active_token_counts = _require_bound_arg(
        active_token_counts,
        api_name="run_sparse_mla_kernel",
        name="active_token_counts",
    )
    sm_scale = _require_bound_arg(sm_scale, api_name="run_sparse_mla_kernel", name="sm_scale")
    output = _require_bound_arg(output, api_name="run_sparse_mla_kernel", name="output")

    traits = select_sparse_mla_traits(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        output_dtype=output.dtype,
        v_head_dim=output.shape[-1],
    )
    if traits is None:
        raise ValueError("sparse MLA kernel only supports the exact CUDA GLM-5.1 contract")
    if active_token_counts.dtype != torch.int32:
        raise ValueError(
            f"active_token_counts must have dtype torch.int32, got {active_token_counts.dtype}"
        )
    if active_token_counts.device != q_all.device:
        raise ValueError("active_token_counts must be on the same device as q_all")
    if active_token_counts.ndim != 1 or active_token_counts.shape[0] != q_all.shape[0]:
        raise ValueError(
            "active_token_counts must be rank-1 with one entry per query row, "
            f"got {tuple(active_token_counts.shape)} for q rows {q_all.shape[0]}"
        )

    kv_rows_u32, kv_scales = _extract_packed_kv_runtime_views(kv_cache)
    q_u32 = _view_last_dim_as_u32(q_all)
    if isinstance(sm_scale, torch.Tensor):
        sm_scale_tensor = sm_scale
    else:
        sm_scale_tensor = torch.tensor([sm_scale], dtype=torch.float32, device=q_all.device)
    if sm_scale_tensor.shape != (1,) or sm_scale_tensor.dtype != torch.float32:
        raise ValueError("sm_scale tensor must have shape (1,) and dtype float32")
    if sm_scale_tensor.device != q_all.device:
        raise ValueError("sm_scale tensor must be on the same device as q_all")

    head_tiles = (int(output.shape[1]) + _MLA_HEADS_PER_TILE - 1) // _MLA_HEADS_PER_TILE
    kernel = _build_sparse_mla_kernel_for_shape(traits, head_tiles)
    args = (
        _to_kernel_tensor(q_u32, cutlass.Uint32, assumed_align=16),
        _to_kernel_tensor(kv_rows_u32, cutlass.Uint32, assumed_align=16),
        _to_kernel_tensor(kv_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(page_table_1, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(active_token_counts, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(sm_scale_tensor, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
        current_cuda_stream(),
    )
    # Use phantom tensors from workspace for stable cache keys when available.
    _cq = getattr(workspace, "_contract_q", None)
    _ckv, _cks = _workspace_contract_kv_tensors(workspace, kv_cache)
    _cpt = getattr(workspace, "_contract_page_table", None)
    _cnt = getattr(workspace, "_contract_indexer_cache_seqlens", None)
    _co = getattr(workspace, "_contract_output", None)
    cache_key = (
        _tensor_meta_key(_cq if _cq is not None else q_u32),
        _tensor_meta_key(_ckv if _ckv is not None else kv_rows_u32),
        _tensor_meta_key(_cks if _cks is not None else kv_scales),
        _tensor_meta_key(_cpt if _cpt is not None else page_table_1),
        _tensor_meta_key(_cnt if _cnt is not None else active_token_counts),
        _tensor_meta_key(_co if _co is not None else output),
        traits,
        head_tiles,
        str(output.dtype),
    )
    compile_spec = KernelCompileSpec.from_key(
        "attention.mla.onepass_sparse",
        1,
        cache_key,
        labels=(
            "q",
            "kv_rows",
            "kv_scales",
            "page_table",
            "active_token_counts",
            "output",
            "traits",
            "head_tiles",
            "output_dtype",
        ),
    )
    b12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )
