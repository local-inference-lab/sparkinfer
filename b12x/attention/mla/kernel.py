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
    ld_global_nc_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    ldmatrix_m8n8x4_trans_b16,
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
_MLA_TOKEN_TILE = 64
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
# Per-group Q stage: one nope group (4KB) or rope (2KB) at a time.
# Reduces smem ~23KB -> ~9KB, enabling ~11 CTAs/SM (vs 4).
_MLA_Q_GROUP_STAGE_BYTES = _MLA_Q_NOPE_STAGE_BYTES
# Per-group streaming: KV stage holds one nope group OR rope at a time.
# Reduces smem ~38.6KB -> ~22.6KB, enabling 4 CTAs/SM (vs 2).
_MLA_KV_STAGE_BYTES = _MLA_KV_NOPE_STAGE_BYTES
_MLA_SCALE_STAGE_ELEMS = _MLA_TOKEN_TILE * _MLA_SCALE_GROUPS
_MLA_SCALE_BYTES = _MLA_SCALE_GROUPS * 4
_MLA_NOPE_U32_OFFSET = 0
_MLA_SCALE_U32_OFFSET = _MLA_NOPE_DIM // 4
_MLA_ROPE_U32_OFFSET = _MLA_SCALE_U32_OFFSET + _MLA_SCALE_GROUPS
_MLA_NUM_MMA_KV = _MLA_TOKEN_TILE // 16
_MLA_QK_NUM_MMA_D = 4
_MLA_VO_NUM_MMA_D = _MLA_NOPE_GROUP_ELEMS // 16

_COMPRESSED_MLA_NOPE_DIM = 448
_COMPRESSED_MLA_ROPE_DIM = 64
_COMPRESSED_MLA_HEAD_DIM = _COMPRESSED_MLA_NOPE_DIM + _COMPRESSED_MLA_ROPE_DIM
_COMPRESSED_MLA_GROUP_SIZE = 64
_COMPRESSED_MLA_SCALE_GROUPS = _COMPRESSED_MLA_NOPE_DIM // _COMPRESSED_MLA_GROUP_SIZE
_COMPRESSED_MLA_PAYLOAD_BYTES = 576
_COMPRESSED_MLA_SCALE_BYTES = 8
_COMPRESSED_MLA_GROUP_Q_U32 = _COMPRESSED_MLA_GROUP_SIZE // 2
_COMPRESSED_MLA_GROUP_Q_VECS = (_COMPRESSED_MLA_GROUP_SIZE * 2) // 16
_COMPRESSED_MLA_GROUP_KV_VECS = _COMPRESSED_MLA_GROUP_SIZE // 16
_COMPRESSED_MLA_GROUP_KV_STAGE_VECS = _MLA_NOPE_GROUP_KV_VECS
_COMPRESSED_MLA_GROUP_KV_BF16_VECS = (_COMPRESSED_MLA_GROUP_SIZE * 2) // 16
_COMPRESSED_MLA_QK_NUM_MMA_D = _COMPRESSED_MLA_GROUP_SIZE // 16
_COMPRESSED_MLA_SCALE_STAGE_STRIDE = _MLA_TOKEN_TILE + _MLA_TOKEN_TILE // 8
_COMPRESSED_MLA_SCALE_STAGE_ELEMS = (
    _COMPRESSED_MLA_SCALE_STAGE_STRIDE * _COMPRESSED_MLA_SCALE_GROUPS
)
_MLA_SHARED_SCALE_STAGE_ELEMS = max(
    _MLA_SCALE_STAGE_ELEMS,
    _COMPRESSED_MLA_SCALE_STAGE_ELEMS,
)
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
class SparseMLAKernelBinding:
    q_all: torch.Tensor
    kv_cache: torch.Tensor
    page_table_1: torch.Tensor
    active_token_counts: torch.Tensor
    sm_scale: float | torch.Tensor
    output: torch.Tensor
    workspace: object | None = None
    identity_page_table: bool = False

    def run(self) -> None:
        run_sparse_mla_kernel(binding=self)


def build_sparse_mla_kernel_binding(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    active_token_counts: torch.Tensor,
    sm_scale: float | torch.Tensor,
    output: torch.Tensor,
    workspace: object | None = None,
    identity_page_table: bool = False,
) -> SparseMLAKernelBinding:
    return SparseMLAKernelBinding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        output=output,
        workspace=workspace,
        identity_page_table=bool(identity_page_table),
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
        " @p cp.async.ca.shared.global.L2::64B [$1], [$2], 16;\n"
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


@dsl_user_op
def _ld_global_u8(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ue8m0_to_input_scale(scale_u8: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(scale_u8).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred is_zero;
                .reg .b32 bits, subnormal;
                setp.eq.u32 is_zero, $1, 0;
                shl.b32 bits, $1, 23;
                mov.u32 subnormal, 0x00400000;
                selp.b32 bits, subnormal, bits, is_zero;
                mov.b32 $0, bits;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ue8m0x4_to_input_scales(
    scale_u32: Uint32, *, loc=None, ip=None
) -> tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(scale_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p0, p1, p2, p3;
            .reg .b32 b0, b1, b2, b3;
            .reg .b32 s0, s1, s2, s3;
            and.b32 b0, $4, 0x000000ff;
            shr.u32 b1, $4, 8;
            and.b32 b1, b1, 0x000000ff;
            shr.u32 b2, $4, 16;
            and.b32 b2, b2, 0x000000ff;
            shr.u32 b3, $4, 24;
            setp.eq.u32 p0, b0, 0;
            setp.eq.u32 p1, b1, 0;
            setp.eq.u32 p2, b2, 0;
            setp.eq.u32 p3, b3, 0;
            shl.b32 s0, b0, 23;
            shl.b32 s1, b1, 23;
            shl.b32 s2, b2, 23;
            shl.b32 s3, b3, 23;
            selp.b32 s0, 0x00400000, s0, p0;
            selp.b32 s1, 0x00400000, s1, p1;
            selp.b32 s2, 0x00400000, s2, p2;
            selp.b32 s3, 0x00400000, s3, p3;
            mov.b32 $0, s0;
            mov.b32 $1, s1;
            mov.b32 $2, s2;
            mov.b32 $3, s3;
        }
        """,
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


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
    identity_page_table: cutlass.Constexpr[bool],
):
    token_local = lane
    while token_local < Int32(_MLA_TOKEN_TILE):
        token_pos = token_base + token_local
        if cutlass.const_expr(identity_page_table):
            sTokenIdx[token_local] = (
                q_idx * Int32(page_table_1.shape[1]) + token_pos
                if token_pos < token_end
                else Int32(-1)
            )
        else:
            sTokenIdx[token_local] = (
                Int32(page_table_1[q_idx, token_pos]) if token_pos < token_end else Int32(-1)
            )
        token_local += Int32(_MLA_WARP_THREADS)


@cute.jit
def _stage_compressed_token_indices(
    swa_indices: cute.Tensor,
    swa_lengths: cute.Tensor,
    indexed_indices: cute.Tensor,
    indexed_lengths: cute.Tensor,
    indexed_page_table: cute.Tensor,
    sTokenIdx: cute.Tensor,
    q_idx: Int32,
    token_base: Int32,
    token_end: Int32,
    lane: Int32,
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    map_indexed_page_table: cutlass.Constexpr[bool],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_table_width: Int32,
):
    if cutlass.const_expr(not has_swa):
        indexed_len = Int32(0)
        if cutlass.const_expr(has_indexed):
            indexed_len = Int32(indexed_lengths[q_idx])
            if indexed_len < Int32(0):
                indexed_len = Int32(0)
            if indexed_len > Int32(indexed_indices.shape[1]):
                indexed_len = Int32(indexed_indices.shape[1])

        token_local = lane
        while token_local < Int32(_MLA_TOKEN_TILE):
            token_pos = token_base + token_local
            encoded = Int32(-1)
            if token_pos < token_end:
                if cutlass.const_expr(has_indexed):
                    if token_pos < indexed_len:
                        raw_indexed = Int32(indexed_indices[q_idx, token_pos])
                        if cutlass.const_expr(map_indexed_page_table):
                            page_col = raw_indexed // Int32(indexed_page_size)
                            page_off = raw_indexed - page_col * Int32(indexed_page_size)
                            valid_page_col = raw_indexed >= Int32(0)
                            if valid_page_col:
                                valid_page_col = page_col >= Int32(0)
                            if valid_page_col:
                                valid_page_col = page_col < Int32(indexed_page_table_width)
                            page_id = Int32(-1)
                            if valid_page_col:
                                page_id = Int32(indexed_page_table[q_idx, page_col])
                            if page_id >= Int32(0):
                                token_idx = page_id * Int32(indexed_page_size) + page_off
                                encoded = Int32(0) - token_idx - Int32(2)
                        else:
                            if raw_indexed >= Int32(0):
                                encoded = Int32(0) - raw_indexed - Int32(2)
            sTokenIdx[token_local] = encoded
            token_local += Int32(_MLA_WARP_THREADS)
    else:
        swa_len = Int32(swa_lengths[q_idx])
        if swa_len < Int32(0):
            swa_len = Int32(0)
        if swa_len > Int32(swa_indices.shape[1]):
            swa_len = Int32(swa_indices.shape[1])

        indexed_len = Int32(0)
        if cutlass.const_expr(has_indexed):
            indexed_len = Int32(indexed_lengths[q_idx])
            if indexed_len < Int32(0):
                indexed_len = Int32(0)
            if indexed_len > Int32(indexed_indices.shape[1]):
                indexed_len = Int32(indexed_indices.shape[1])

        token_local = lane
        while token_local < Int32(_MLA_TOKEN_TILE):
            token_pos = token_base + token_local
            encoded = Int32(-1)
            if token_pos < token_end:
                if token_pos < swa_len:
                    raw_swa = Int32(swa_indices[q_idx, token_pos])
                    if raw_swa >= Int32(0):
                        encoded = raw_swa
                else:
                    extra_slot = token_pos - swa_len
                    if cutlass.const_expr(has_indexed):
                        if extra_slot < indexed_len:
                            raw_indexed = Int32(indexed_indices[q_idx, extra_slot])
                            if cutlass.const_expr(map_indexed_page_table):
                                page_col = raw_indexed // Int32(indexed_page_size)
                                page_off = raw_indexed - page_col * Int32(indexed_page_size)
                                valid_page_col = raw_indexed >= Int32(0)
                                if valid_page_col:
                                    valid_page_col = page_col >= Int32(0)
                                if valid_page_col:
                                    valid_page_col = page_col < Int32(indexed_page_table_width)
                                page_id = Int32(-1)
                                if valid_page_col:
                                    page_id = Int32(indexed_page_table[q_idx, page_col])
                                if page_id >= Int32(0):
                                    token_idx = page_id * Int32(indexed_page_size) + page_off
                                    encoded = Int32(0) - token_idx - Int32(2)
                            else:
                                if raw_indexed >= Int32(0):
                                    encoded = Int32(0) - raw_indexed - Int32(2)
            sTokenIdx[token_local] = encoded
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
def _compressed_token_base(
    swa_u8: cute.Tensor,
    indexed_u8: cute.Tensor,
    encoded_token: Int32,
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
) -> tuple[Int64, Int64, Int64, Int64]:
    if cutlass.const_expr(not has_swa):
        src_u8 = get_ptr_as_int64(indexed_u8, Int64(0))
        page_size = Int64(indexed_page_size)
        page_nbytes = Int64(indexed_page_nbytes)
        token_idx = Int64(0)
        valid = Int64(0)
        if encoded_token <= Int32(-2):
            token_idx = Int64(Int32(0) - encoded_token - Int32(2))
            valid = Int64(1)
    elif cutlass.const_expr(not has_indexed):
        src_u8 = get_ptr_as_int64(swa_u8, Int64(0))
        page_size = Int64(swa_page_size)
        page_nbytes = Int64(swa_page_nbytes)
        token_idx = Int64(0)
        valid = Int64(0)
        if encoded_token >= Int32(0):
            token_idx = Int64(encoded_token)
            valid = Int64(1)
    else:
        src_u8 = get_ptr_as_int64(swa_u8, Int64(0))
        page_size = Int64(swa_page_size)
        page_nbytes = Int64(swa_page_nbytes)
        token_idx = Int64(0)
        valid = Int64(0)
        if encoded_token >= Int32(0):
            token_idx = Int64(encoded_token)
            valid = Int64(1)
        elif encoded_token <= Int32(-2):
            token_idx = Int64(Int32(0) - encoded_token - Int32(2))
            src_u8 = get_ptr_as_int64(indexed_u8, Int64(0))
            page_size = Int64(indexed_page_size)
            page_nbytes = Int64(indexed_page_nbytes)
            valid = Int64(1)
    page = token_idx // page_size
    token_offset = token_idx - page * page_size
    payload_base = page * page_nbytes + token_offset * Int64(_COMPRESSED_MLA_PAYLOAD_BYTES)
    scale_base = (
        page * page_nbytes
        + page_size * Int64(_COMPRESSED_MLA_PAYLOAD_BYTES)
        + token_offset * Int64(_COMPRESSED_MLA_SCALE_BYTES)
    )
    return src_u8, payload_base, scale_base, valid


@cute.jit
def _compressed_scale_smem_idx(token_local: Int32, group_idx: Int32):
    return (
        group_idx * Int32(_COMPRESSED_MLA_SCALE_STAGE_STRIDE)
        + token_local
        + token_local // Int32(8)
    )


@cute.jit
def _stage_compressed_token_scales(
    swa_u8: cute.Tensor,
    indexed_u8: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    group_idx: Int32,
    lane: Int32,
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
):
    token_local = lane
    while token_local < Int32(_MLA_TOKEN_TILE):
        encoded = Int32(sTokenIdx[token_local])
        src_u8, _, scale_base, valid = _compressed_token_base(
            swa_u8,
            indexed_u8,
            encoded,
            has_swa,
            has_indexed,
            swa_page_size,
            swa_page_nbytes,
            indexed_page_size,
            indexed_page_nbytes,
        )
        scale = Float32(0.0)
        if valid != Int64(0):
            scale = _ue8m0_to_input_scale(_ld_global_u8(src_u8 + scale_base + Int64(group_idx)))
        sScale[_compressed_scale_smem_idx(token_local, group_idx)] = scale
        token_local += Int32(_MLA_WARP_THREADS)


@cute.jit
def _stage_all_compressed_token_scales(
    swa_u8: cute.Tensor,
    indexed_u8: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    tile_tokens: Int32,
    lane: Int32,
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
):
    token_local = lane
    while token_local < Int32(_MLA_TOKEN_TILE):
        encoded = Int32(-1)
        if token_local < tile_tokens:
            encoded = Int32(sTokenIdx[token_local])
        src_u8, _, scale_base, valid = _compressed_token_base(
            swa_u8,
            indexed_u8,
            encoded,
            has_swa,
            has_indexed,
            swa_page_size,
            swa_page_nbytes,
            indexed_page_size,
            indexed_page_nbytes,
        )
        scale0 = Float32(0.0)
        scale1 = Float32(0.0)
        scale2 = Float32(0.0)
        scale3 = Float32(0.0)
        scale4 = Float32(0.0)
        scale5 = Float32(0.0)
        scale6 = Float32(0.0)
        if valid != Int64(0):
            word0 = ld_global_nc_u32(src_u8 + scale_base)
            scale0, scale1, scale2, scale3 = _ue8m0x4_to_input_scales(word0)
            word1 = ld_global_nc_u32(src_u8 + scale_base + Int64(4))
            scale4, scale5, scale6, _ = _ue8m0x4_to_input_scales(word1)
        sScale[_compressed_scale_smem_idx(token_local, Int32(0))] = scale0
        sScale[_compressed_scale_smem_idx(token_local, Int32(1))] = scale1
        sScale[_compressed_scale_smem_idx(token_local, Int32(2))] = scale2
        sScale[_compressed_scale_smem_idx(token_local, Int32(3))] = scale3
        sScale[_compressed_scale_smem_idx(token_local, Int32(4))] = scale4
        sScale[_compressed_scale_smem_idx(token_local, Int32(5))] = scale5
        sScale[_compressed_scale_smem_idx(token_local, Int32(6))] = scale6
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
def _stage_compressed_kv_u32_block_active_only(
    swa_u8: cute.Tensor,
    indexed_u8: cute.Tensor,
    sTokenIdx: cute.Tensor,
    base_byte: Int32,
    vecs_per_row: Int32,
    row_stride_128b: Int32,
    kv_base_addr: Int32,
    tile_tokens: Int32,
    lane: Int32,
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
):
    total = Int32(_MLA_TOKEN_TILE) * vecs_per_row
    active_total = tile_tokens * vecs_per_row
    if active_total > total:
        active_total = total

    linear = lane
    while linear < total:
        row = linear // vecs_per_row
        vec_idx = linear - row * vecs_per_row
        dst_addr = _smem_addr_from_b128_offset(
            kv_base_addr,
            _permuted_offset_128b(row, vec_idx, row_stride_128b),
        )
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
        if linear < active_total:
            encoded = Int32(sTokenIdx[row])
            src_u8, payload_base, _, valid = _compressed_token_base(
                swa_u8,
                indexed_u8,
                encoded,
                has_swa,
                has_indexed,
                swa_page_size,
                swa_page_nbytes,
                indexed_page_size,
                indexed_page_nbytes,
            )
            if valid != Int64(0):
                src_byte = payload_base + Int64(base_byte) + Int64(vec_idx) * Int64(16)
                v0 = ld_global_nc_u32(src_u8 + src_byte + Int64(0))
                v1 = ld_global_nc_u32(src_u8 + src_byte + Int64(4))
                v2 = ld_global_nc_u32(src_u8 + src_byte + Int64(8))
                v3 = ld_global_nc_u32(src_u8 + src_byte + Int64(12))
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(_MLA_WARP_THREADS)


@cute.jit
def _stage_compressed_kv_u32_block(
    swa_u8: cute.Tensor,
    indexed_u8: cute.Tensor,
    sTokenIdx: cute.Tensor,
    base_byte: Int32,
    vecs_per_row: Int32,
    row_stride_128b: Int32,
    kv_base_addr: Int32,
    tile_tokens: Int32,
    lane: Int32,
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
):
    total = Int32(_MLA_TOKEN_TILE) * vecs_per_row
    active_total = tile_tokens * vecs_per_row
    if active_total > total:
        active_total = total

    linear = lane
    while linear < total:
        row = linear // vecs_per_row
        vec_idx = linear - row * vecs_per_row
        dst_addr = _smem_addr_from_b128_offset(
            kv_base_addr,
            _permuted_offset_128b(row, vec_idx, row_stride_128b),
        )
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
        if linear < active_total:
            encoded = Int32(sTokenIdx[row])
            src_u8, payload_base, _, valid = _compressed_token_base(
                swa_u8,
                indexed_u8,
                encoded,
                has_swa,
                has_indexed,
                swa_page_size,
                swa_page_nbytes,
                indexed_page_size,
                indexed_page_nbytes,
            )
            if valid != Int64(0):
                src_byte = payload_base + Int64(base_byte) + Int64(vec_idx) * Int64(16)
                v0 = ld_global_nc_u32(src_u8 + src_byte + Int64(0))
                v1 = ld_global_nc_u32(src_u8 + src_byte + Int64(4))
                v2 = ld_global_nc_u32(src_u8 + src_byte + Int64(8))
                v3 = ld_global_nc_u32(src_u8 + src_byte + Int64(12))
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(_MLA_WARP_THREADS)

    linear = active_total + lane
    while linear < total:
        row = linear // vecs_per_row
        vec_idx = linear - row * vecs_per_row
        dst_addr = _smem_addr_from_b128_offset(
            kv_base_addr,
            _permuted_offset_128b(row, vec_idx, row_stride_128b),
        )
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
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
def _zero_output_frag_b1(o_frag: cute.Tensor):
    for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
        o_frag[0, mma_d, 0] = Float32(0.0)
        o_frag[0, mma_d, 1] = Float32(0.0)
        o_frag[0, mma_d, 4] = Float32(0.0)
        o_frag[0, mma_d, 5] = Float32(0.0)


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
def _accumulate_compressed_scaled_score_frag(
    dst_frag: cute.Tensor,
    src_frag: cute.Tensor,
    sScale: cute.Tensor,
    group_idx: Int32,
    lane: Int32,
):
    lane_pair_base = Int32(2) * (lane % Int32(4))
    scale_base = group_idx * Int32(_COMPRESSED_MLA_SCALE_STAGE_STRIDE)
    scale01_k0 = Float32(sScale[scale_base + lane_pair_base + Int32(0)])
    scale02_k0 = Float32(sScale[scale_base + lane_pair_base + Int32(1)])
    scale89_k0 = Float32(sScale[scale_base + lane_pair_base + Int32(9)])
    scale90_k0 = Float32(sScale[scale_base + lane_pair_base + Int32(10)])
    scale01_k1 = Float32(sScale[scale_base + lane_pair_base + Int32(18)])
    scale02_k1 = Float32(sScale[scale_base + lane_pair_base + Int32(19)])
    scale89_k1 = Float32(sScale[scale_base + lane_pair_base + Int32(27)])
    scale90_k1 = Float32(sScale[scale_base + lane_pair_base + Int32(28)])
    scale01_k2 = Float32(sScale[scale_base + lane_pair_base + Int32(36)])
    scale02_k2 = Float32(sScale[scale_base + lane_pair_base + Int32(37)])
    scale89_k2 = Float32(sScale[scale_base + lane_pair_base + Int32(45)])
    scale90_k2 = Float32(sScale[scale_base + lane_pair_base + Int32(46)])
    scale01_k3 = Float32(sScale[scale_base + lane_pair_base + Int32(54)])
    scale02_k3 = Float32(sScale[scale_base + lane_pair_base + Int32(55)])
    scale89_k3 = Float32(sScale[scale_base + lane_pair_base + Int32(63)])
    scale90_k3 = Float32(sScale[scale_base + lane_pair_base + Int32(64)])

    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        scale01 = scale01_k0
        scale02 = scale02_k0
        scale89 = scale89_k0
        scale90 = scale90_k0
        if mma_kv == 1:
            scale01 = scale01_k1
            scale02 = scale02_k1
            scale89 = scale89_k1
            scale90 = scale90_k1
        elif mma_kv == 2:
            scale01 = scale01_k2
            scale02 = scale02_k2
            scale89 = scale89_k2
            scale90 = scale90_k2
        elif mma_kv == 3:
            scale01 = scale01_k3
            scale02 = scale02_k3
            scale89 = scale89_k3
            scale90 = scale90_k3

        dst_frag[0, mma_kv, 0] = Float32(
            dst_frag[0, mma_kv, 0] + src_frag[0, mma_kv, 0] * scale01
        )
        dst_frag[0, mma_kv, 1] = Float32(
            dst_frag[0, mma_kv, 1] + src_frag[0, mma_kv, 1] * scale02
        )
        dst_frag[0, mma_kv, 2] = Float32(
            dst_frag[0, mma_kv, 2] + src_frag[0, mma_kv, 2] * scale01
        )
        dst_frag[0, mma_kv, 3] = Float32(
            dst_frag[0, mma_kv, 3] + src_frag[0, mma_kv, 3] * scale02
        )
        dst_frag[0, mma_kv, 4] = Float32(
            dst_frag[0, mma_kv, 4] + src_frag[0, mma_kv, 4] * scale89
        )
        dst_frag[0, mma_kv, 5] = Float32(
            dst_frag[0, mma_kv, 5] + src_frag[0, mma_kv, 5] * scale90
        )
        dst_frag[0, mma_kv, 6] = Float32(
            dst_frag[0, mma_kv, 6] + src_frag[0, mma_kv, 6] * scale89
        )
        dst_frag[0, mma_kv, 7] = Float32(
            dst_frag[0, mma_kv, 7] + src_frag[0, mma_kv, 7] * scale90
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
    identity_page_table: cutlass.Constexpr[bool],
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    num_heads = Int32(q_u32.shape[1])
    num_kv = Int32(kv_rows_u32.shape[0])
    tile_tokens = token_end - token_base

    _stage_token_indices(
        page_table_1,
        sTokenIdx,
        q_idx,
        token_base,
        token_end,
        lane,
        identity_page_table,
    )
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
def _update_softmax_stats_b1(
    score_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    o_rescale_frag: cute.Tensor,
):
    m_prev = Float32(m_frag[0, 0])
    m_new = Float32(m_prev)
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        m_local = attention_ops.fmax(
            attention_ops.fmax(
                score_frag[0, mma_kv, 0],
                score_frag[0, mma_kv, 1],
            ),
            attention_ops.fmax(
                score_frag[0, mma_kv, 4],
                score_frag[0, mma_kv, 5],
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
    o_rescale_frag[0, 0] = scale_term
    d_acc = Float32(d_frag[0, 0] * scale_term * Float32(0.25))
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        p0 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 0] - m_new)
        )
        p1 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 1] - m_new)
        )
        p2 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 4] - m_new)
        )
        p3 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 5] - m_new)
        )
        d_acc = Float32(d_acc + p0 + p1 + p2 + p3)
    d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=2))
    d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=1))
    m_frag[0, 0] = Float32(m_new)
    d_frag[0, 0] = Float32(d_acc)


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
def _fill_normalized_p_frag_from_scores_b1(
    p_frag: cute.Tensor,
    score_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
):
    del d_frag
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        m_scaled = Float32(m_frag[0, 0])
        p0 = (
            Float32(0.0)
            if m_scaled == -Float32.inf
            else Float32(_exp2_approx_ftz_f32(score_frag[0, mma_kv, 0] - m_scaled))
        )
        p1 = (
            Float32(0.0)
            if m_scaled == -Float32.inf
            else Float32(_exp2_approx_ftz_f32(score_frag[0, mma_kv, 1] - m_scaled))
        )
        p2 = (
            Float32(0.0)
            if m_scaled == -Float32.inf
            else Float32(_exp2_approx_ftz_f32(score_frag[0, mma_kv, 4] - m_scaled))
        )
        p3 = (
            Float32(0.0)
            if m_scaled == -Float32.inf
            else Float32(_exp2_approx_ftz_f32(score_frag[0, mma_kv, 5] - m_scaled))
        )
        p_frag[0, mma_kv, 0] = pack_f32x2_to_bfloat2(p0, p1)
        p_frag[0, mma_kv, 1] = Uint32(0)
        p_frag[0, mma_kv, 2] = pack_f32x2_to_bfloat2(p2, p3)
        p_frag[0, mma_kv, 3] = Uint32(0)



@cute.jit
def _update_softmax_rescale_and_p_b1(
    score_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
    o_frag0: cute.Tensor,
    o_frag1: cute.Tensor,
    o_frag2: cute.Tensor,
    o_frag3: cute.Tensor,
):
    """Fused softmax-stats + O-rescale + P-norm for single-head-slot (b1) path."""
    m_prev = Float32(m_frag[0, 0])
    m_new = Float32(m_prev)
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        m_local = attention_ops.fmax(
            attention_ops.fmax(
                score_frag[0, mma_kv, 0],
                score_frag[0, mma_kv, 1],
            ),
            attention_ops.fmax(
                score_frag[0, mma_kv, 4],
                score_frag[0, mma_kv, 5],
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

    # O-rescale: apply scale_term to all slot-0 elements of o_frag0-3
    for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
        o_frag0[0, mma_d, 0] = Float32(o_frag0[0, mma_d, 0] * scale_term)
        o_frag0[0, mma_d, 1] = Float32(o_frag0[0, mma_d, 1] * scale_term)
        o_frag0[0, mma_d, 4] = Float32(o_frag0[0, mma_d, 4] * scale_term)
        o_frag0[0, mma_d, 5] = Float32(o_frag0[0, mma_d, 5] * scale_term)
        o_frag1[0, mma_d, 0] = Float32(o_frag1[0, mma_d, 0] * scale_term)
        o_frag1[0, mma_d, 1] = Float32(o_frag1[0, mma_d, 1] * scale_term)
        o_frag1[0, mma_d, 4] = Float32(o_frag1[0, mma_d, 4] * scale_term)
        o_frag1[0, mma_d, 5] = Float32(o_frag1[0, mma_d, 5] * scale_term)
        o_frag2[0, mma_d, 0] = Float32(o_frag2[0, mma_d, 0] * scale_term)
        o_frag2[0, mma_d, 1] = Float32(o_frag2[0, mma_d, 1] * scale_term)
        o_frag2[0, mma_d, 4] = Float32(o_frag2[0, mma_d, 4] * scale_term)
        o_frag2[0, mma_d, 5] = Float32(o_frag2[0, mma_d, 5] * scale_term)
        o_frag3[0, mma_d, 0] = Float32(o_frag3[0, mma_d, 0] * scale_term)
        o_frag3[0, mma_d, 1] = Float32(o_frag3[0, mma_d, 1] * scale_term)
        o_frag3[0, mma_d, 4] = Float32(o_frag3[0, mma_d, 4] * scale_term)
        o_frag3[0, mma_d, 5] = Float32(o_frag3[0, mma_d, 5] * scale_term)

    # Combined d accumulation + p_frag fill: compute exp2(score - m_new) once
    d_acc = Float32(d_frag[0, 0] * scale_term * Float32(0.25))
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        p0 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 0] - m_new)
        )
        p1 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 1] - m_new)
        )
        p2 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 4] - m_new)
        )
        p3 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 5] - m_new)
        )
        d_acc = Float32(d_acc + p0 + p1 + p2 + p3)
        p_frag[0, mma_kv, 0] = pack_f32x2_to_bfloat2(p0, p1)
        p_frag[0, mma_kv, 1] = Uint32(0)
        p_frag[0, mma_kv, 2] = pack_f32x2_to_bfloat2(p2, p3)
        p_frag[0, mma_kv, 3] = Uint32(0)
    d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=2))
    d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=1))
    m_frag[0, 0] = Float32(m_new)
    d_frag[0, 0] = Float32(d_acc)


@cute.jit
def _update_softmax_rescale_and_p_b2(
    score_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
    o_frag0: cute.Tensor,
    o_frag1: cute.Tensor,
    o_frag2: cute.Tensor,
    o_frag3: cute.Tensor,
):
    """Fused softmax-stats + O-rescale + P-norm for dual-head-slot (b2) path."""
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

        # O-rescale for this row_slot
        if cutlass.const_expr(row_slot == 0):
            rs = scale_term
            for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
                o_frag0[0, mma_d, 0] = Float32(o_frag0[0, mma_d, 0] * rs)
                o_frag0[0, mma_d, 1] = Float32(o_frag0[0, mma_d, 1] * rs)
                o_frag0[0, mma_d, 4] = Float32(o_frag0[0, mma_d, 4] * rs)
                o_frag0[0, mma_d, 5] = Float32(o_frag0[0, mma_d, 5] * rs)
                o_frag1[0, mma_d, 0] = Float32(o_frag1[0, mma_d, 0] * rs)
                o_frag1[0, mma_d, 1] = Float32(o_frag1[0, mma_d, 1] * rs)
                o_frag1[0, mma_d, 4] = Float32(o_frag1[0, mma_d, 4] * rs)
                o_frag1[0, mma_d, 5] = Float32(o_frag1[0, mma_d, 5] * rs)
                o_frag2[0, mma_d, 0] = Float32(o_frag2[0, mma_d, 0] * rs)
                o_frag2[0, mma_d, 1] = Float32(o_frag2[0, mma_d, 1] * rs)
                o_frag2[0, mma_d, 4] = Float32(o_frag2[0, mma_d, 4] * rs)
                o_frag2[0, mma_d, 5] = Float32(o_frag2[0, mma_d, 5] * rs)
                o_frag3[0, mma_d, 0] = Float32(o_frag3[0, mma_d, 0] * rs)
                o_frag3[0, mma_d, 1] = Float32(o_frag3[0, mma_d, 1] * rs)
                o_frag3[0, mma_d, 4] = Float32(o_frag3[0, mma_d, 4] * rs)
                o_frag3[0, mma_d, 5] = Float32(o_frag3[0, mma_d, 5] * rs)
        else:
            rs1 = scale_term
            for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
                o_frag0[0, mma_d, 2] = Float32(o_frag0[0, mma_d, 2] * rs1)
                o_frag0[0, mma_d, 3] = Float32(o_frag0[0, mma_d, 3] * rs1)
                o_frag0[0, mma_d, 6] = Float32(o_frag0[0, mma_d, 6] * rs1)
                o_frag0[0, mma_d, 7] = Float32(o_frag0[0, mma_d, 7] * rs1)
                o_frag1[0, mma_d, 2] = Float32(o_frag1[0, mma_d, 2] * rs1)
                o_frag1[0, mma_d, 3] = Float32(o_frag1[0, mma_d, 3] * rs1)
                o_frag1[0, mma_d, 6] = Float32(o_frag1[0, mma_d, 6] * rs1)
                o_frag1[0, mma_d, 7] = Float32(o_frag1[0, mma_d, 7] * rs1)
                o_frag2[0, mma_d, 2] = Float32(o_frag2[0, mma_d, 2] * rs1)
                o_frag2[0, mma_d, 3] = Float32(o_frag2[0, mma_d, 3] * rs1)
                o_frag2[0, mma_d, 6] = Float32(o_frag2[0, mma_d, 6] * rs1)
                o_frag2[0, mma_d, 7] = Float32(o_frag2[0, mma_d, 7] * rs1)
                o_frag3[0, mma_d, 2] = Float32(o_frag3[0, mma_d, 2] * rs1)
                o_frag3[0, mma_d, 3] = Float32(o_frag3[0, mma_d, 3] * rs1)
                o_frag3[0, mma_d, 6] = Float32(o_frag3[0, mma_d, 6] * rs1)
                o_frag3[0, mma_d, 7] = Float32(o_frag3[0, mma_d, 7] * rs1)

        # Combined d accumulation + p_frag fill
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
            p_frag[0, mma_kv, row_slot + 0] = pack_f32x2_to_bfloat2(p0, p1)
            p_frag[0, mma_kv, row_slot + 2] = pack_f32x2_to_bfloat2(p2, p3)
        d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=2))
        d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=1))
        m_frag[0, row_slot] = Float32(m_new)
        d_frag[0, row_slot] = Float32(d_acc)


@cute.jit
def _update_softmax_and_p_b1(
    score_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
):
    m_prev = Float32(m_frag[0, 0])
    m_new = Float32(m_prev)
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        m_local = attention_ops.fmax(
            attention_ops.fmax(
                score_frag[0, mma_kv, 0],
                score_frag[0, mma_kv, 1],
            ),
            attention_ops.fmax(
                score_frag[0, mma_kv, 4],
                score_frag[0, mma_kv, 5],
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

    d_acc = Float32(d_frag[0, 0] * scale_term * Float32(0.25))
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        p0 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 0] - m_new)
        )
        p1 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 1] - m_new)
        )
        p2 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 4] - m_new)
        )
        p3 = (
            Float32(0.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32(score_frag[0, mma_kv, 5] - m_new)
        )
        d_acc = Float32(d_acc + p0 + p1 + p2 + p3)
        p_frag[0, mma_kv, 0] = pack_f32x2_to_bfloat2(p0, p1)
        p_frag[0, mma_kv, 1] = Uint32(0)
        p_frag[0, mma_kv, 2] = pack_f32x2_to_bfloat2(p2, p3)
        p_frag[0, mma_kv, 3] = Uint32(0)
    d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=2))
    d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=1))
    m_frag[0, 0] = Float32(m_new)
    d_frag[0, 0] = Float32(d_acc)


@cute.jit
def _update_softmax_and_p_b2(
    score_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
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
            p_frag[0, mma_kv, row_slot + 0] = pack_f32x2_to_bfloat2(p0, p1)
            p_frag[0, mma_kv, row_slot + 2] = pack_f32x2_to_bfloat2(p2, p3)
        d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=2))
        d_acc = Float32(d_acc + cute.arch.shuffle_sync_bfly(d_acc, offset=1))
        m_frag[0, row_slot] = Float32(m_new)
        d_frag[0, row_slot] = Float32(d_acc)


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
    scale01_k2 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(32)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(33)]),
    )
    scale89_k2 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(40)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(41)]),
    )
    scale01_k3 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(48)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(49)]),
    )
    scale89_k3 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(56)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(57)]),
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
    scale01_k2 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(32)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(33)]),
    )
    scale89_k2 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(40)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(41)]),
    )
    scale01_k3 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(48)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(49)]),
    )
    scale89_k3 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(56)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(57)]),
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
        scale01 = scale01_k0
        scale89 = scale89_k0
        if mma_kv == 1:
            scale01 = scale01_k1
            scale89 = scale89_k1
        elif mma_kv == 2:
            scale01 = scale01_k2
            scale89 = scale89_k2
        elif mma_kv == 3:
            scale01 = scale01_k3
            scale89 = scale89_k3
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
def _literal_pv_mma_into_ofrag_fp8_raw_scaled_64(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    sScale: cute.Tensor,
    scale_base: Int32,
    lane: Int32,
    mma_d_offset: cutlass.Constexpr[int],
):
    lane_pair_base = Int32(2) * (lane % Int32(4))

    scale01_k0 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(0)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(1)]),
    )
    scale89_k0 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(9)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(10)]),
    )
    scale01_k1 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(18)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(19)]),
    )
    scale89_k1 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(27)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(28)]),
    )
    scale01_k2 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(36)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(37)]),
    )
    scale89_k2 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(45)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(46)]),
    )
    scale01_k3 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(54)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(55)]),
    )
    scale89_k3 = pack_f32x2_to_bfloat2(
        Float32(sScale[scale_base + lane_pair_base + Int32(63)]),
        Float32(sScale[scale_base + lane_pair_base + Int32(64)]),
    )

    v_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        Int32(_COMPRESSED_MLA_GROUP_KV_STAGE_VECS),
    )
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((1, 4), stride=(4, 1)),
            Uint32,
        )
        scale01 = scale01_k0
        scale89 = scale89_k0
        if mma_kv == 1:
            scale01 = scale01_k1
            scale89 = scale89_k1
        elif mma_kv == 2:
            scale01 = scale01_k2
            scale89 = scale89_k2
        elif mma_kv == 3:
            scale01 = scale01_k3
            scale89 = scale89_k3
        a_regs[0, 0] = bfloat2_mul(p_frag[0, mma_kv, 0], scale01)
        a_regs[0, 1] = bfloat2_mul(p_frag[0, mma_kv, 1], scale01)
        a_regs[0, 2] = bfloat2_mul(p_frag[0, mma_kv, 2], scale89)
        a_regs[0, 3] = bfloat2_mul(p_frag[0, mma_kv, 3], scale89)

        v_offset_cur = v_offset
        for local_mma_d in cutlass.range_constexpr(_COMPRESSED_MLA_GROUP_SIZE // 16):
            b_f8_0 = Uint32(0)
            b_f8_1 = Uint32(0)
            if local_mma_d % 2 == 0:
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
            if local_mma_d % 2 == 1:
                v_offset_cur = _advance_offset_by_column_128b_2(v_offset_cur, local_mma_d // 2)
            mma_d = mma_d_offset + local_mma_d
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
        v_offset = _advance_offset_by_row_128b(
            v_offset_cur,
            16,
            Int32(_COMPRESSED_MLA_GROUP_KV_STAGE_VECS),
        ) - Int32(_COMPRESSED_MLA_GROUP_SIZE // 16)
        # The 64-wide compressed group is staged in an 8-vector swizzled row.
        # Rows whose swizzle sets bit 2 need to advance back to logical vec0.
        if lane % Int32(8) >= Int32(4):
            v_offset += Int32(8)


@cute.jit
def _literal_pv_mma_into_ofrag_bf16_64(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    lane: Int32,
    mma_d_offset: cutlass.Constexpr[int],
):
    v_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        Int32(_COMPRESSED_MLA_GROUP_KV_BF16_VECS),
    )
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((1, 4), stride=(4, 1)),
            Uint32,
        )
        a_regs[0, 0] = p_frag[0, mma_kv, 0]
        a_regs[0, 1] = p_frag[0, mma_kv, 1]
        a_regs[0, 2] = p_frag[0, mma_kv, 2]
        a_regs[0, 3] = p_frag[0, mma_kv, 3]

        v_offset_cur = v_offset
        for local_mma_d in cutlass.range_constexpr(_COMPRESSED_MLA_ROPE_DIM // 16):
            mma_d = mma_d_offset + local_mma_d
            v_row = mma_kv * Int32(16) + lane % Int32(16)
            v_col = Int32(local_mma_d) * Int32(2) + lane // Int32(16)
            v_offset_cur = _permuted_offset_128b(
                v_row,
                v_col,
                Int32(_COMPRESSED_MLA_GROUP_KV_BF16_VECS),
            )
            b0, b1, b2, b3 = ldmatrix_m8n8x4_trans_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_cur)
            )
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
def _store_output_group_with_sink(
    out_tensor: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    attn_sink: cute.Tensor,
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
            row_m = Float32(m_frag[0, row_slot])
            sink_m = Float32(attn_sink[head_idx] * attention_ops.LOG2_E)
            new_m = attention_ops.fmax(row_m, sink_m)
            prev_scale = (
                Float32(0.0)
                if row_m == -Float32.inf
                else _exp2_approx_ftz_f32(row_m - new_m)
            )
            sink_scale = _exp2_approx_ftz_f32(sink_m - new_m)
            denom = Float32(d_frag[0, row_slot] * prev_scale + sink_scale)
            out_scale = (
                Float32(0.0)
                if denom == Float32(0.0)
                else Float32(prev_scale) / denom
            )
            for mma_d in cutlass.range_constexpr(_MLA_VO_NUM_MMA_D):
                dim_base = group_idx * Int32(_MLA_GROUP_SIZE) + mma_d * Int32(16) + lane_pair_base
                out_tensor[out_row_idx, head_idx, dim_base + Int32(0)] = Float32(
                    o_frag[0, mma_d, reg_base + 0] * out_scale
                ).to(out_tensor.element_type)
                out_tensor[out_row_idx, head_idx, dim_base + Int32(1)] = Float32(
                    o_frag[0, mma_d, reg_base + 1] * out_scale
                ).to(out_tensor.element_type)
                out_tensor[out_row_idx, head_idx, dim_base + Int32(8)] = Float32(
                    o_frag[0, mma_d, reg_base + 4] * out_scale
                ).to(out_tensor.element_type)
                out_tensor[out_row_idx, head_idx, dim_base + Int32(9)] = Float32(
                    o_frag[0, mma_d, reg_base + 5] * out_scale
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
    kv_base_addr: Int32,
    q_idx: Int32,
    head_tile_start: Int32,
    token_base: Int32,
    token_end: Int32,
    sm_scale_log2: Float32,
    lane: Int32,
    identity_page_table: cutlass.Constexpr[bool],
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    num_heads = Int32(q_u32.shape[1])
    num_kv = Int32(kv_rows_u32.shape[0])
    tile_tokens = token_end - token_base

    _stage_token_indices(
        page_table_1,
        sTokenIdx,
        q_idx,
        token_base,
        token_end,
        lane,
        identity_page_table,
    )
    cute.arch.sync_threads()
    _stage_all_token_scales(kv_scales, sTokenIdx, sScale, num_kv, lane)

    _zero_score_frag(score_frag)
    frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))

    # Pipelined per-group Q+KV co-streaming: overlap async staging with compute
    # Prologue: stage nope group 0
    _stage_q_u32_block_async(
        q_u32, q_idx, head_tile_start,
        Int32(0), Int32(_MLA_NOPE_GROUP_Q_VECS), Int32(_MLA_NOPE_GROUP_Q_VECS),
        q_base_addr, lane,
    )
    _stage_kv_u32_block_async(
        kv_rows_u32, sTokenIdx,
        Int32(_MLA_NOPE_U32_OFFSET),
        Int32(_MLA_NOPE_GROUP_KV_VECS), Int32(_MLA_NOPE_GROUP_KV_VECS),
        kv_base_addr, num_kv, lane,
    )
    cute.arch.cp_async_commit_group()

    for block_offset in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
        group_idx = Int32(block_offset)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Compute current nope group
        frag_tmp = cute.make_rmem_tensor(frag_layout, Float32)
        _zero_score_frag(frag_tmp)
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
        _accumulate_scaled_score_frag(
            score_frag,
            frag_tmp,
            sScale,
            group_idx * Int32(_MLA_TOKEN_TILE),
            lane,
        )
        cute.arch.sync_threads()

        # Issue async staging for next nope group (overlapped with next iteration)
        if cutlass.const_expr(block_offset < _MLA_SCALE_GROUPS - 1):
            _stage_q_u32_block_async(
                q_u32, q_idx, head_tile_start,
                (group_idx + Int32(1)) * Int32(_MLA_NOPE_GROUP_Q_U32),
                Int32(_MLA_NOPE_GROUP_Q_VECS), Int32(_MLA_NOPE_GROUP_Q_VECS),
                q_base_addr, lane,
            )
            _stage_kv_u32_block_async(
                kv_rows_u32, sTokenIdx,
                Int32(_MLA_NOPE_U32_OFFSET) + (group_idx + Int32(1)) * Int32(_MLA_NOPE_GROUP_KV_U32),
                Int32(_MLA_NOPE_GROUP_KV_VECS), Int32(_MLA_NOPE_GROUP_KV_VECS),
                kv_base_addr, num_kv, lane,
            )
            cute.arch.cp_async_commit_group()

    # Rope QK: co-stream Q rope + KV rope
    _stage_q_u32_block_async(
        q_u32,
        q_idx,
        head_tile_start,
        Int32(_MLA_NOPE_DIM // 2),
        Int32(_MLA_ROPE_VECS),
        Int32(_MLA_ROPE_VECS),
        q_base_addr,
        lane,
    )
    _stage_kv_u32_block_async(
        kv_rows_u32,
        sTokenIdx,
        Int32(_MLA_ROPE_U32_OFFSET),
        Int32(_MLA_ROPE_VECS),
        Int32(_MLA_ROPE_VECS),
        kv_base_addr,
        num_kv,
        lane,
    )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
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
def _compute_compressed_score_tile_scaled(
    score_frag: cute.Tensor,
    q_u32: cute.Tensor,
    swa_u8: cute.Tensor,
    swa_indices: cute.Tensor,
    swa_lengths: cute.Tensor,
    indexed_u8: cute.Tensor,
    indexed_indices: cute.Tensor,
    indexed_lengths: cute.Tensor,
    indexed_page_table: cute.Tensor,
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
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    map_indexed_page_table: cutlass.Constexpr[bool],
    indexed_page_table_width: Int32,
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    num_heads = Int32(q_u32.shape[1])
    tile_tokens = token_end - token_base

    _stage_compressed_token_indices(
        swa_indices,
        swa_lengths,
        indexed_indices,
        indexed_lengths,
        indexed_page_table,
        sTokenIdx,
        q_idx,
        token_base,
        token_end,
        lane,
        has_swa,
        has_indexed,
        map_indexed_page_table,
        indexed_page_size,
        indexed_page_table_width,
    )
    cute.arch.sync_threads()

    _zero_score_frag(score_frag)
    frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))
    _stage_all_compressed_token_scales(
        swa_u8,
        indexed_u8,
        sTokenIdx,
        sScale,
        tile_tokens,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
    )

    for block_offset in cutlass.range_constexpr(_COMPRESSED_MLA_SCALE_GROUPS):
        group_idx = Int32(block_offset)
        _stage_q_u32_block(
            q_u32,
            q_idx,
            head_tile_start,
            group_idx * Int32(_COMPRESSED_MLA_GROUP_Q_U32),
            Int32(_COMPRESSED_MLA_GROUP_Q_VECS),
            Int32(_COMPRESSED_MLA_GROUP_Q_VECS),
            q_base_addr,
            lane,
        )
        _stage_compressed_kv_u32_block_active_only(
            swa_u8,
            indexed_u8,
            sTokenIdx,
            group_idx * Int32(_COMPRESSED_MLA_GROUP_SIZE),
            Int32(_COMPRESSED_MLA_GROUP_KV_VECS),
            Int32(_COMPRESSED_MLA_GROUP_KV_STAGE_VECS),
            kv_base_addr,
            tile_tokens,
            lane,
            has_swa,
            has_indexed,
            swa_page_size,
            swa_page_nbytes,
            indexed_page_size,
            indexed_page_nbytes,
        )
        cute.arch.sync_threads()

        frag_tmp = cute.make_rmem_tensor(frag_layout, Float32)
        _zero_score_frag(frag_tmp)
        _literal_qk_mma_into_sfrag_mxfp8_raw(
            frag_tmp,
            q_base_addr,
            kv_base_addr,
            lane,
            Int32(0),
            Int32(1),
            Int32(_MLA_NUM_MMA_KV),
            Int32(_COMPRESSED_MLA_QK_NUM_MMA_D),
            Int32(_COMPRESSED_MLA_GROUP_Q_VECS),
            Int32(_COMPRESSED_MLA_GROUP_KV_STAGE_VECS),
        )
        _accumulate_compressed_scaled_score_frag(
            score_frag,
            frag_tmp,
            sScale,
            group_idx,
            lane,
        )
        cute.arch.sync_threads()

    _stage_q_u32_block(
        q_u32,
        q_idx,
        head_tile_start,
        Int32(_COMPRESSED_MLA_NOPE_DIM // 2),
        Int32(_COMPRESSED_MLA_GROUP_Q_VECS),
        Int32(_COMPRESSED_MLA_GROUP_Q_VECS),
        q_base_addr,
        lane,
    )
    _stage_compressed_kv_u32_block_active_only(
        swa_u8,
        indexed_u8,
        sTokenIdx,
        Int32(_COMPRESSED_MLA_NOPE_DIM),
        Int32(_COMPRESSED_MLA_GROUP_KV_BF16_VECS),
        Int32(_COMPRESSED_MLA_GROUP_KV_BF16_VECS),
        kv_base_addr,
        tile_tokens,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
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
        Int32(_COMPRESSED_MLA_QK_NUM_MMA_D),
        Int32(_COMPRESSED_MLA_GROUP_Q_VECS),
        Int32(_COMPRESSED_MLA_GROUP_KV_BF16_VECS),
    )
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        for reg_id in cutlass.range_constexpr(8):
            score_frag[0, mma_kv, reg_id] = Float32(
                score_frag[0, mma_kv, reg_id] + frag_rope[0, mma_kv, reg_id]
            )

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
            encoded = Int32(sTokenIdx[token_local])
            valid = token_local < tile_tokens
            if valid:
                valid = valid and encoded != Int32(-1)
            if valid:
                valid = valid and head_idx < num_heads
            score_frag[0, mma_kv, reg_id] = (
                Float32(score_frag[0, mma_kv, reg_id] * sm_scale_log2)
                if valid
                else Float32(-Float32.inf)
            )
    cute.arch.sync_threads()




@cute.jit
def _pipeline_stage_q_async(
    q_u32: cute.Tensor,
    q_base_addr: Int32,
    q_idx: Int32,
    head_tile_start: Int32,
    lane: Int32,
):
    """Stage Q into smem (async). Call AFTER QK compute frees q_stage."""
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






@cute.jit
def _accumulate_pv_groups_from_p_frag_staged(
    o_frag0: cute.Tensor,
    o_frag1: cute.Tensor,
    o_frag2: cute.Tensor,
    o_frag3: cute.Tensor,
    p_frag: cute.Tensor,
    kv_rows_u32: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    kv_base_addr: Int32,
    num_kv: Int32,
    lane: Int32,
):
    # Pipelined KV streaming: overlap stage of next group with compute of current
    # Prologue: stage nope group 0
    _stage_kv_u32_block_async(
        kv_rows_u32,
        sTokenIdx,
        Int32(_MLA_NOPE_U32_OFFSET),
        Int32(_MLA_NOPE_GROUP_KV_VECS),
        Int32(_MLA_NOPE_GROUP_KV_VECS),
        kv_base_addr,
        num_kv,
        lane,
    )
    cute.arch.cp_async_commit_group()

    for block_offset in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
        group_idx = Int32(block_offset)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        scale_base = group_idx * Int32(_MLA_TOKEN_TILE)
        tile_output_scale = _warp_allreduce_max(Float32(sScale[scale_base + lane]))
        tile_pv_scale = (
            Float32(0.0)
            if tile_output_scale == Float32(0.0)
            else cute.arch.rcp_approx(tile_output_scale)
        )
        # Compute PV from current buffer
        if cutlass.const_expr(block_offset == 0):
            _run_staged_pv_group_into_target(
                o_frag0, p_frag, sScale, kv_base_addr, scale_base, tile_pv_scale, lane
            )
        elif cutlass.const_expr(block_offset == 1):
            _run_staged_pv_group_into_target(
                o_frag1, p_frag, sScale, kv_base_addr, scale_base, tile_pv_scale, lane
            )
        elif cutlass.const_expr(block_offset == 2):
            _run_staged_pv_group_into_target(
                o_frag2, p_frag, sScale, kv_base_addr, scale_base, tile_pv_scale, lane
            )
        else:
            _run_staged_pv_group_into_target(
                o_frag3, p_frag, sScale, kv_base_addr, scale_base, tile_pv_scale, lane
            )
        cute.arch.sync_threads()

        # Issue async staging for next nope group
        if cutlass.const_expr(block_offset < _MLA_SCALE_GROUPS - 1):
            _stage_kv_u32_block_async(
                kv_rows_u32,
                sTokenIdx,
                Int32(_MLA_NOPE_U32_OFFSET) + (group_idx + Int32(1)) * Int32(_MLA_NOPE_GROUP_KV_U32),
                Int32(_MLA_NOPE_GROUP_KV_VECS),
                Int32(_MLA_NOPE_GROUP_KV_VECS),
                kv_base_addr,
                num_kv,
                lane,
            )
            cute.arch.cp_async_commit_group()


@cute.jit
def _accumulate_compressed_pv_groups_from_p_frag_staged(
    o_frag0: cute.Tensor,
    o_frag1: cute.Tensor,
    o_frag2: cute.Tensor,
    o_frag3: cute.Tensor,
    p_frag: cute.Tensor,
    swa_u8: cute.Tensor,
    indexed_u8: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    kv_base_addr: Int32,
    tile_tokens: Int32,
    lane: Int32,
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
):
    for block_offset in cutlass.range_constexpr(_COMPRESSED_MLA_SCALE_GROUPS):
        group_idx = Int32(block_offset)
        _stage_compressed_kv_u32_block(
            swa_u8,
            indexed_u8,
            sTokenIdx,
            group_idx * Int32(_COMPRESSED_MLA_GROUP_SIZE),
            Int32(_COMPRESSED_MLA_GROUP_KV_VECS),
            Int32(_COMPRESSED_MLA_GROUP_KV_STAGE_VECS),
            kv_base_addr,
            tile_tokens,
            lane,
            has_swa,
            has_indexed,
            swa_page_size,
            swa_page_nbytes,
            indexed_page_size,
            indexed_page_nbytes,
        )
        cute.arch.sync_threads()

        scale_base = group_idx * Int32(_COMPRESSED_MLA_SCALE_STAGE_STRIDE)
        if cutlass.const_expr(block_offset == 0):
            _literal_pv_mma_into_ofrag_fp8_raw_scaled_64(o_frag0, p_frag, kv_base_addr, sScale, scale_base, lane, 0)
        elif cutlass.const_expr(block_offset == 1):
            _literal_pv_mma_into_ofrag_fp8_raw_scaled_64(o_frag0, p_frag, kv_base_addr, sScale, scale_base, lane, 4)
        elif cutlass.const_expr(block_offset == 2):
            _literal_pv_mma_into_ofrag_fp8_raw_scaled_64(o_frag1, p_frag, kv_base_addr, sScale, scale_base, lane, 0)
        elif cutlass.const_expr(block_offset == 3):
            _literal_pv_mma_into_ofrag_fp8_raw_scaled_64(o_frag1, p_frag, kv_base_addr, sScale, scale_base, lane, 4)
        elif cutlass.const_expr(block_offset == 4):
            _literal_pv_mma_into_ofrag_fp8_raw_scaled_64(o_frag2, p_frag, kv_base_addr, sScale, scale_base, lane, 0)
        elif cutlass.const_expr(block_offset == 5):
            _literal_pv_mma_into_ofrag_fp8_raw_scaled_64(o_frag2, p_frag, kv_base_addr, sScale, scale_base, lane, 4)
        else:
            _literal_pv_mma_into_ofrag_fp8_raw_scaled_64(o_frag3, p_frag, kv_base_addr, sScale, scale_base, lane, 0)
        cute.arch.sync_threads()

    _stage_compressed_kv_u32_block(
        swa_u8,
        indexed_u8,
        sTokenIdx,
        Int32(_COMPRESSED_MLA_NOPE_DIM),
        Int32(_COMPRESSED_MLA_GROUP_KV_BF16_VECS),
        Int32(_COMPRESSED_MLA_GROUP_KV_BF16_VECS),
        kv_base_addr,
        tile_tokens,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
    )
    cute.arch.sync_threads()
    _literal_pv_mma_into_ofrag_bf16_64(o_frag3, p_frag, kv_base_addr, lane, 4)
    cute.arch.sync_threads()


@cute.jit
def _accumulate_compressed_pv_fp8_group_into_frag(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    swa_u8: cute.Tensor,
    indexed_u8: cute.Tensor,
    sTokenIdx: cute.Tensor,
    sScale: cute.Tensor,
    kv_base_addr: Int32,
    tile_tokens: Int32,
    lane: Int32,
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
    scale_group: cutlass.Constexpr[int],
    mma_d_offset: cutlass.Constexpr[int],
):
    _stage_compressed_kv_u32_block(
        swa_u8,
        indexed_u8,
        sTokenIdx,
        Int32(scale_group * _COMPRESSED_MLA_GROUP_SIZE),
        Int32(_COMPRESSED_MLA_GROUP_KV_VECS),
        Int32(_COMPRESSED_MLA_GROUP_KV_STAGE_VECS),
        kv_base_addr,
        tile_tokens,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
    )
    cute.arch.sync_threads()
    _literal_pv_mma_into_ofrag_fp8_raw_scaled_64(
        o_frag,
        p_frag,
        kv_base_addr,
        sScale,
        Int32(scale_group * _COMPRESSED_MLA_SCALE_STAGE_STRIDE),
        lane,
        mma_d_offset,
    )
    cute.arch.sync_threads()


@cute.jit
def _accumulate_compressed_pv_rope_group_into_frag(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    swa_u8: cute.Tensor,
    indexed_u8: cute.Tensor,
    sTokenIdx: cute.Tensor,
    kv_base_addr: Int32,
    tile_tokens: Int32,
    lane: Int32,
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
):
    _stage_compressed_kv_u32_block(
        swa_u8,
        indexed_u8,
        sTokenIdx,
        Int32(_COMPRESSED_MLA_NOPE_DIM),
        Int32(_COMPRESSED_MLA_GROUP_KV_BF16_VECS),
        Int32(_COMPRESSED_MLA_GROUP_KV_BF16_VECS),
        kv_base_addr,
        tile_tokens,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
    )
    cute.arch.sync_threads()
    _literal_pv_mma_into_ofrag_bf16_64(o_frag, p_frag, kv_base_addr, lane, 4)
    cute.arch.sync_threads()


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
def _store_output_groups_with_sink(
    out_tensor: cute.Tensor,
    o_frag0: cute.Tensor,
    o_frag1: cute.Tensor,
    o_frag2: cute.Tensor,
    o_frag3: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    attn_sink: cute.Tensor,
    out_row_idx: Int32,
    head_tile_start: Int32,
    lane: Int32,
):
    _store_output_group_with_sink(
        out_tensor,
        o_frag0,
        m_frag,
        d_frag,
        attn_sink,
        out_row_idx,
        head_tile_start,
        Int32(0),
        lane,
    )
    _store_output_group_with_sink(
        out_tensor,
        o_frag1,
        m_frag,
        d_frag,
        attn_sink,
        out_row_idx,
        head_tile_start,
        Int32(1),
        lane,
    )
    _store_output_group_with_sink(
        out_tensor,
        o_frag2,
        m_frag,
        d_frag,
        attn_sink,
        out_row_idx,
        head_tile_start,
        Int32(2),
        lane,
    )
    _store_output_group_with_sink(
        out_tensor,
        o_frag3,
        m_frag,
        d_frag,
        attn_sink,
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
def _run_single_tile_compressed_mla_tile(
    q_u32: cute.Tensor,
    swa_u8: cute.Tensor,
    swa_indices: cute.Tensor,
    swa_lengths: cute.Tensor,
    indexed_u8: cute.Tensor,
    indexed_indices: cute.Tensor,
    indexed_lengths: cute.Tensor,
    indexed_page_table: cute.Tensor,
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
    attn_sink: cute.Tensor,
    apply_attn_sink: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    map_indexed_page_table: cutlass.Constexpr[bool],
    indexed_page_table_width: Int32,
):
    md_layout = cute.make_layout((1, 2), stride=(2, 1))
    frag_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 8), stride=(16, 8, 1))
    p_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 4), stride=(8, 4, 1))
    o_layout = cute.make_layout((1, _MLA_VO_NUM_MMA_D, 8), stride=(_MLA_VO_NUM_MMA_D * 8, 8, 1))

    m_frag = cute.make_rmem_tensor(md_layout, Float32)
    d_frag = cute.make_rmem_tensor(md_layout, Float32)
    for row_slot in cutlass.range_constexpr(2):
        m_frag[0, row_slot] = Float32(-Float32.inf)
        d_frag[0, row_slot] = Float32(0.0)

    score_frag = cute.make_rmem_tensor(frag_layout, Float32)
    _compute_compressed_score_tile_scaled(
        score_frag,
        q_u32,
        swa_u8,
        swa_indices,
        swa_lengths,
        indexed_u8,
        indexed_indices,
        indexed_lengths,
        indexed_page_table,
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
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        has_swa,
        has_indexed,
        map_indexed_page_table,
        indexed_page_table_width,
    )

    p_frag = cute.make_rmem_tensor(p_layout, Uint32)
    num_heads = Int32(q_u32.shape[1])
    has_second_head_slot = head_tile_start + Int32(8) < num_heads
    if has_second_head_slot:
        _update_softmax_and_p_b2(score_frag, m_frag, d_frag, p_frag)
    else:
        _update_softmax_and_p_b1(score_frag, m_frag, d_frag, p_frag)

    o_frag = cute.make_rmem_tensor(o_layout, Float32)
    if has_second_head_slot:
        _zero_output_frag(o_frag)
    else:
        _zero_output_frag_b1(o_frag)
    _accumulate_compressed_pv_fp8_group_into_frag(
        o_frag,
        p_frag,
        swa_u8,
        indexed_u8,
        sTokenIdx,
        sScale,
        kv_base_addr,
        token_end - token_start,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        0,
        0,
    )
    _accumulate_compressed_pv_fp8_group_into_frag(
        o_frag,
        p_frag,
        swa_u8,
        indexed_u8,
        sTokenIdx,
        sScale,
        kv_base_addr,
        token_end - token_start,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        1,
        4,
    )
    if cutlass.const_expr(lse_tensor is None):
        if cutlass.const_expr(apply_attn_sink):
            _store_output_group_with_sink(
                out_tensor,
                o_frag,
                m_frag,
                d_frag,
                attn_sink,
                out_row_idx,
                head_tile_start,
                Int32(0),
                lane,
            )
        else:
            _store_output_group(out_tensor, o_frag, d_frag, out_row_idx, head_tile_start, Int32(0), lane)
    else:
        _store_output_group_chunked(out_tensor, o_frag, d_frag, out_row_idx, out_chunk_idx, head_tile_start, Int32(0), lane)

    if has_second_head_slot:
        _zero_output_frag(o_frag)
    else:
        _zero_output_frag_b1(o_frag)
    _accumulate_compressed_pv_fp8_group_into_frag(
        o_frag,
        p_frag,
        swa_u8,
        indexed_u8,
        sTokenIdx,
        sScale,
        kv_base_addr,
        token_end - token_start,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        2,
        0,
    )
    _accumulate_compressed_pv_fp8_group_into_frag(
        o_frag,
        p_frag,
        swa_u8,
        indexed_u8,
        sTokenIdx,
        sScale,
        kv_base_addr,
        token_end - token_start,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        3,
        4,
    )
    if cutlass.const_expr(lse_tensor is None):
        if cutlass.const_expr(apply_attn_sink):
            _store_output_group_with_sink(
                out_tensor,
                o_frag,
                m_frag,
                d_frag,
                attn_sink,
                out_row_idx,
                head_tile_start,
                Int32(1),
                lane,
            )
        else:
            _store_output_group(out_tensor, o_frag, d_frag, out_row_idx, head_tile_start, Int32(1), lane)
    else:
        _store_output_group_chunked(out_tensor, o_frag, d_frag, out_row_idx, out_chunk_idx, head_tile_start, Int32(1), lane)

    if has_second_head_slot:
        _zero_output_frag(o_frag)
    else:
        _zero_output_frag_b1(o_frag)
    _accumulate_compressed_pv_fp8_group_into_frag(
        o_frag,
        p_frag,
        swa_u8,
        indexed_u8,
        sTokenIdx,
        sScale,
        kv_base_addr,
        token_end - token_start,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        4,
        0,
    )
    _accumulate_compressed_pv_fp8_group_into_frag(
        o_frag,
        p_frag,
        swa_u8,
        indexed_u8,
        sTokenIdx,
        sScale,
        kv_base_addr,
        token_end - token_start,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        5,
        4,
    )
    if cutlass.const_expr(lse_tensor is None):
        if cutlass.const_expr(apply_attn_sink):
            _store_output_group_with_sink(
                out_tensor,
                o_frag,
                m_frag,
                d_frag,
                attn_sink,
                out_row_idx,
                head_tile_start,
                Int32(2),
                lane,
            )
        else:
            _store_output_group(out_tensor, o_frag, d_frag, out_row_idx, head_tile_start, Int32(2), lane)
    else:
        _store_output_group_chunked(out_tensor, o_frag, d_frag, out_row_idx, out_chunk_idx, head_tile_start, Int32(2), lane)

    if has_second_head_slot:
        _zero_output_frag(o_frag)
    else:
        _zero_output_frag_b1(o_frag)
    _accumulate_compressed_pv_fp8_group_into_frag(
        o_frag,
        p_frag,
        swa_u8,
        indexed_u8,
        sTokenIdx,
        sScale,
        kv_base_addr,
        token_end - token_start,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        6,
        0,
    )
    _accumulate_compressed_pv_rope_group_into_frag(
        o_frag,
        p_frag,
        swa_u8,
        indexed_u8,
        sTokenIdx,
        kv_base_addr,
        token_end - token_start,
        lane,
        has_swa,
        has_indexed,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
    )
    if cutlass.const_expr(lse_tensor is None):
        if cutlass.const_expr(apply_attn_sink):
            _store_output_group_with_sink(
                out_tensor,
                o_frag,
                m_frag,
                d_frag,
                attn_sink,
                out_row_idx,
                head_tile_start,
                Int32(3),
                lane,
            )
        else:
            _store_output_group(out_tensor, o_frag, d_frag, out_row_idx, head_tile_start, Int32(3), lane)
    else:
        _store_output_group_chunked(out_tensor, o_frag, d_frag, out_row_idx, out_chunk_idx, head_tile_start, Int32(3), lane)
        _store_partial_lse_chunked(
            lse_tensor,
            out_row_idx,
            out_chunk_idx,
            head_tile_start,
            m_frag,
            d_frag,
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
    identity_page_table: cutlass.Constexpr[bool],
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
    num_heads = Int32(q_u32.shape[1])
    has_second_head_slot = head_tile_start + Int32(8) < num_heads
    if has_second_head_slot:
        _zero_output_frag(o_frag0)
        _zero_output_frag(o_frag1)
        _zero_output_frag(o_frag2)
        _zero_output_frag(o_frag3)
    else:
        _zero_output_frag_b1(o_frag0)
        _zero_output_frag_b1(o_frag1)
        _zero_output_frag_b1(o_frag2)
        _zero_output_frag_b1(o_frag3)
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
                identity_page_table,
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
                q_idx,
                head_tile_start,
                token_start,
                token_end,
                sm_scale_log2,
                lane,
                identity_page_table,
            )
        if has_second_head_slot:
            _update_softmax_stats_b2(score_frag, m_frag, d_frag, o_rescale_frag)
        else:
            _update_softmax_stats_b1(score_frag, m_frag, d_frag, o_rescale_frag)
        p_frag = cute.make_rmem_tensor(p_layout, Uint32)
        if has_second_head_slot:
            _fill_normalized_p_frag_from_scores(p_frag, score_frag, m_frag, d_frag)
        else:
            _fill_normalized_p_frag_from_scores_b1(p_frag, score_frag, m_frag, d_frag)
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
                kv_rows_u32,
                sTokenIdx,
                sScale,
                kv_base_addr,
                Int32(kv_rows_u32.shape[0]),
                lane,
            )
    else:
        # Multi-tile: sequential per-group QK+PV (~10KB smem, ~9 CTAs/SM)
        num_kv = Int32(kv_rows_u32.shape[0])

        token_base = token_start

        while token_base < token_end:
            tile_end = cutlass.select_(
                token_base + Int32(_MLA_TOKEN_TILE) < token_end,
                token_base + Int32(_MLA_TOKEN_TILE),
                token_end,
            )

            # QK: sequential per-group (no KV double-buffering)
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
                    token_base,
                    tile_end,
                    sm_scale_log2,
                    lane,
                    identity_page_table,
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
                    q_idx,
                    head_tile_start,
                    token_base,
                    tile_end,
                    sm_scale_log2,
                    lane,
                    identity_page_table,
                )

            # Fused softmax-stats + O-rescale + P-norm
            p_frag = cute.make_rmem_tensor(p_layout, Uint32)
            if has_second_head_slot:
                _update_softmax_rescale_and_p_b2(
                    score_frag, m_frag, d_frag, p_frag,
                    o_frag0, o_frag1, o_frag2, o_frag3,
                )
            else:
                _update_softmax_rescale_and_p_b1(
                    score_frag, m_frag, d_frag, p_frag,
                    o_frag0, o_frag1, o_frag2, o_frag3,
                )


            # PV: sequential per-group (no KV double-buffering)
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
                    kv_rows_u32,
                    sTokenIdx,
                    sScale,
                    kv_base_addr,
                    num_kv,
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


@cute.jit
def _run_one_pass_compressed_mla_tile(
    q_u32: cute.Tensor,
    swa_u8: cute.Tensor,
    swa_indices: cute.Tensor,
    swa_lengths: cute.Tensor,
    indexed_u8: cute.Tensor,
    indexed_indices: cute.Tensor,
    indexed_lengths: cute.Tensor,
    indexed_page_table: cute.Tensor,
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
    attn_sink: cute.Tensor,
    apply_attn_sink: cutlass.Constexpr[bool],
    swa_page_size: cutlass.Constexpr[int],
    swa_page_nbytes: cutlass.Constexpr[int],
    indexed_page_size: cutlass.Constexpr[int],
    indexed_page_nbytes: cutlass.Constexpr[int],
    has_swa: cutlass.Constexpr[bool],
    has_indexed: cutlass.Constexpr[bool],
    map_indexed_page_table: cutlass.Constexpr[bool],
    indexed_page_table_width: Int32,
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
    num_heads = Int32(q_u32.shape[1])
    has_second_head_slot = head_tile_start + Int32(8) < num_heads
    if has_second_head_slot:
        _zero_output_frag(o_frag0)
        _zero_output_frag(o_frag1)
        _zero_output_frag(o_frag2)
        _zero_output_frag(o_frag3)
    else:
        _zero_output_frag_b1(o_frag0)
        _zero_output_frag_b1(o_frag1)
        _zero_output_frag_b1(o_frag2)
        _zero_output_frag_b1(o_frag3)

    token_base = token_start
    while token_base < token_end:
        tile_end = cutlass.select_(
            token_base + Int32(_MLA_TOKEN_TILE) < token_end,
            token_base + Int32(_MLA_TOKEN_TILE),
            token_end,
        )

        score_frag = cute.make_rmem_tensor(frag_layout, Float32)
        _compute_compressed_score_tile_scaled(
            score_frag,
            q_u32,
            swa_u8,
            swa_indices,
            swa_lengths,
            indexed_u8,
            indexed_indices,
            indexed_lengths,
            indexed_page_table,
            sTokenIdx,
            sScale,
            q_base_addr,
            kv_base_addr,
            q_idx,
            head_tile_start,
            token_base,
            tile_end,
            sm_scale_log2,
            lane,
            swa_page_size,
            swa_page_nbytes,
            indexed_page_size,
            indexed_page_nbytes,
            has_swa,
            has_indexed,
            map_indexed_page_table,
            indexed_page_table_width,
        )

        p_frag = cute.make_rmem_tensor(p_layout, Uint32)
        if has_second_head_slot:
            _update_softmax_rescale_and_p_b2(
                score_frag,
                m_frag,
                d_frag,
                p_frag,
                o_frag0,
                o_frag1,
                o_frag2,
                o_frag3,
            )
        else:
            _update_softmax_rescale_and_p_b1(
                score_frag,
                m_frag,
                d_frag,
                p_frag,
                o_frag0,
                o_frag1,
                o_frag2,
                o_frag3,
            )

        _accumulate_compressed_pv_groups_from_p_frag_staged(
            o_frag0,
            o_frag1,
            o_frag2,
            o_frag3,
            p_frag,
            swa_u8,
            indexed_u8,
            sTokenIdx,
            sScale,
            kv_base_addr,
            tile_end - token_base,
            lane,
            has_swa,
            has_indexed,
            swa_page_size,
            swa_page_nbytes,
            indexed_page_size,
            indexed_page_nbytes,
        )
        token_base = tile_end

    if cutlass.const_expr(lse_tensor is None):
        if cutlass.const_expr(apply_attn_sink):
            _store_output_groups_with_sink(
                out_tensor,
                o_frag0,
                o_frag1,
                o_frag2,
                o_frag3,
                m_frag,
                d_frag,
                attn_sink,
                out_row_idx,
                head_tile_start,
                lane,
            )
        else:
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
        "q_group_stage": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(_MLA_Q_GROUP_STAGE_BYTES)],
            128,
        ],
        "kv_stage_a": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(_MLA_KV_STAGE_BYTES)],
            128,
        ],
        "token_idx": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, _MLA_TOKEN_TILE],
            16,
        ],
        "token_scale_a": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MLA_SHARED_SCALE_STAGE_ELEMS],
            16,
        ],
    }
    return cute.struct(SharedStorage)


class SparseMLAKernel:
    """Single-pass sparse MLA kernel using MXFP8 MMA for nope and BF16 MMA for rope."""

    def __init__(self, head_tiles: int, identity_page_table: bool = False):
        self.head_tiles = int(head_tiles)
        self.identity_page_table = bool(identity_page_table)

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
        sScale = storage.token_scale_a.get_tensor(
            cute.make_layout((_MLA_SHARED_SCALE_STAGE_ELEMS,), stride=(1,)))
        q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())
        kv_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

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
            self.identity_page_table,
        )

@lru_cache(maxsize=16)
def _build_sparse_mla_kernel_for_shape(
    traits: SparseMLATraits,
    head_tiles: int,
    identity_page_table: bool,
) -> SparseMLAKernel:
    del traits
    return SparseMLAKernel(head_tiles, identity_page_table)


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
    identity_page_table: bool | None = None,
    binding: SparseMLAKernelBinding | None = None,
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
                ("identity_page_table", identity_page_table),
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
        identity_page_table = binding.identity_page_table

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
    identity_page_table = False if identity_page_table is None else bool(identity_page_table)

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
    kernel = _build_sparse_mla_kernel_for_shape(traits, head_tiles, bool(identity_page_table))
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
        bool(identity_page_table),
    )
    compile_spec = KernelCompileSpec.from_key(
        "attention.mla.sparse",
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
            "identity_page_table",
        ),
    )
    b12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )
