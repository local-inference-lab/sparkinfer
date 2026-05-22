"""CuTe DSL compressed-MLA KV prep kernels."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import os
import warnings

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import (
    cvt_e4m3_to_f32_via_f16,
    cvt_f32_to_e4m3,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_nc_u32,
    st_global_f32,
    st_global_u8,
)
from b12x.cute.utils import current_cuda_stream as current_stream
from b12x.runtime_control import raise_if_kernel_resolution_frozen


_EAGER_HOST_LAUNCHER_CACHE_SIZE = int(os.getenv("B12X_EAGER_HOST_LAUNCHER_CACHE_SIZE", "512"))
_THREADS = 64
_MLA_PACKED_DIM = 656
_MLA_SCALE_OFFSET_BYTES = 512
_MLA_ROPE_OFFSET_BYTES = 528
_COMPRESSED_PAYLOAD_BYTES = 576
_COMPRESSED_NOPE_BYTES = 448
_FP8_MAX = 448.0
_FP8_MAX_RECIP = 1.0 / _FP8_MAX


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
def _ld_global_bf16_to_f32(base_ptr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 tmp; ld.global.b16 tmp, [$1]; cvt.f32.bf16 $0, tmp; }",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_u32(base_ptr: Int64, value: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
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


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    if tensor.ndim >= 2:
        leading_dim = next((idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None)
        if leading_dim is not None:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_meta_key(tensor: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _launcher_cache_lookup(kernel: object, cache_key: tuple[object, ...]):
    cache = getattr(kernel, "_eager_host_launchers", None)
    if cache is None:
        cache = OrderedDict()
        setattr(kernel, "_eager_host_launchers", cache)
        return cache, None
    compiled = cache.get(cache_key)
    if compiled is not None:
        cache.move_to_end(cache_key)
    return cache, compiled


def _run_cached_host_launcher(kernel: object, cache_key: tuple[object, ...], args: tuple[object, ...]) -> None:
    cache, compiled = _launcher_cache_lookup(kernel, cache_key)
    if compiled is None:
        raise_if_kernel_resolution_frozen(
            "eager host launcher compile",
            target=kernel,
            cache_key=cache_key,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cache is disabled as user wants to compile only.",
                category=UserWarning,
            )
            compiled = kernel(*args, compile_only=True)
        cache[cache_key] = compiled
        if len(cache) > _EAGER_HOST_LAUNCHER_CACHE_SIZE:
            cache.popitem(last=False)
    exe_args, _ = compiled.generate_execution_args(*args)
    compiled.run_compiled_program(exe_args)


class CompressedMLAPrepKVKernel:
    def __init__(
        self,
        *,
        rows: int,
        total_width: int,
        swa_width: int,
        indexed_width: int,
        swa_page_size: int,
        swa_page_nbytes: int,
        indexed_page_size: int,
        indexed_page_nbytes: int,
        has_indexed: bool,
        map_indexed_page_table: bool,
        indexed_page_table_width: int,
    ):
        self.rows = int(rows)
        self.total_width = int(total_width)
        self.swa_width = int(swa_width)
        self.indexed_width = int(indexed_width)
        self.swa_page_size = int(swa_page_size)
        self.swa_page_nbytes = int(swa_page_nbytes)
        self.indexed_page_size = int(indexed_page_size)
        self.indexed_page_nbytes = int(indexed_page_nbytes)
        self.has_indexed = bool(has_indexed)
        self.map_indexed_page_table = bool(map_indexed_page_table)
        self.indexed_page_table_width = int(indexed_page_table_width)

    @cute.jit
    def __call__(
        self,
        swa_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        swa_lengths: cute.Tensor,
        indexed_u8: cute.Tensor,
        indexed_indices: cute.Tensor,
        indexed_lengths: cute.Tensor,
        indexed_page_table: cute.Tensor,
        kv_u8: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            swa_u8,
            swa_indices,
            swa_lengths,
            indexed_u8,
            indexed_indices,
            indexed_lengths,
            indexed_page_table,
            kv_u8,
        ).launch(
            grid=(self.rows, self.total_width, 4),
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        swa_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        swa_lengths: cute.Tensor,
        indexed_u8: cute.Tensor,
        indexed_indices: cute.Tensor,
        indexed_lengths: cute.Tensor,
        indexed_page_table: cute.Tensor,
        kv_u8: cute.Tensor,
    ):
        tx, _, _ = cute.arch.thread_idx()
        row, slot, group = cute.arch.block_idx()
        tx = Int32(tx)
        row = Int32(row)
        slot = Int32(slot)
        group = Int32(group)

        @cute.struct
        class SharedStorage:
            warp_maxes: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 2], 128]
            scale: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 1], 128]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_warp_maxes = storage.warp_maxes.get_tensor(cute.make_layout((4,), stride=(1,)))
        s_scale = storage.scale.get_tensor(cute.make_layout((1,), stride=(1,)))

        swa_len = Int32(swa_lengths[row])
        indexed_len = Int32(0)
        if cutlass.const_expr(self.has_indexed):
            indexed_len = Int32(indexed_lengths[row])

        active_swa = slot < swa_len
        extra_slot = slot - swa_len
        active_indexed = False
        if cutlass.const_expr(self.has_indexed):
            active_indexed = (not active_swa) and (extra_slot < indexed_len)
        active = active_swa or active_indexed

        token_index = Int64(0)
        if active_swa:
            token_index = Int64(swa_indices[row, slot])
        elif active_indexed:
            indexed_index_i32 = Int32(indexed_indices[row, extra_slot])
            if cutlass.const_expr(self.map_indexed_page_table):
                indexed_page_col = indexed_index_i32 // Int32(self.indexed_page_size)
                indexed_page_off = indexed_index_i32 - indexed_page_col * Int32(self.indexed_page_size)
                valid_page_col = (
                    indexed_index_i32 >= Int32(0)
                    and indexed_page_col >= Int32(0)
                    and indexed_page_col < Int32(self.indexed_page_table_width)
                )
                page_id = Int32(-1)
                if valid_page_col:
                    page_id = Int32(indexed_page_table[row, indexed_page_col])
                active_indexed = valid_page_col and page_id >= Int32(0)
                active = active_indexed
                token_index = Int64(page_id * Int32(self.indexed_page_size) + indexed_page_off)
            else:
                token_index = Int64(indexed_index_i32)

        page_size = Int64(self.swa_page_size)
        page_nbytes = Int64(self.swa_page_nbytes)
        src_u8 = get_ptr_as_int64(swa_u8, Int64(0))
        if not active_swa:
            page_size = Int64(self.indexed_page_size)
            page_nbytes = Int64(self.indexed_page_nbytes)
            src_u8 = get_ptr_as_int64(indexed_u8, Int64(0))

        page = token_index // page_size
        token_offset = token_index - page * page_size
        payload_base = page * page_nbytes + token_offset * Int64(_COMPRESSED_PAYLOAD_BYTES)
        scale_base = page * page_nbytes + page_size * Int64(_COMPRESSED_PAYLOAD_BYTES) + token_offset * Int64(8)

        dim0 = group * Int32(128) + tx
        dim1 = dim0 + Int32(64)
        val0 = Float32(0.0)
        val1 = Float32(0.0)
        if active:
            if group < Int32(3):
                raw_byte0 = _ld_global_u8(src_u8 + payload_base + Int64(dim0))
                scale_id0 = dim0 // Int32(64)
                scale_byte0 = _ld_global_u8(src_u8 + scale_base + Int64(scale_id0))
                val0 = cvt_e4m3_to_f32_via_f16(raw_byte0) * _ue8m0_to_input_scale(scale_byte0)
                raw_byte1 = _ld_global_u8(src_u8 + payload_base + Int64(dim1))
                scale_id1 = dim1 // Int32(64)
                scale_byte1 = _ld_global_u8(src_u8 + scale_base + Int64(scale_id1))
                val1 = cvt_e4m3_to_f32_via_f16(raw_byte1) * _ue8m0_to_input_scale(scale_byte1)
            else:
                raw_byte0 = _ld_global_u8(src_u8 + payload_base + Int64(384) + Int64(tx))
                scale_byte0 = _ld_global_u8(src_u8 + scale_base + Int64(6))
                val0 = cvt_e4m3_to_f32_via_f16(raw_byte0) * _ue8m0_to_input_scale(scale_byte0)
                val1 = _ld_global_bf16_to_f32(
                    src_u8 + payload_base + Int64(_COMPRESSED_NOPE_BYTES) + Int64(tx) * Int64(2)
                )

                if tx < Int32(32):
                    rope_word = ld_global_nc_u32(src_u8 + payload_base + Int64(_COMPRESSED_NOPE_BYTES) + Int64(tx) * Int64(4))
                    core_row = Int64(row) * Int64(self.total_width) + Int64(slot)
                    kv_base = get_ptr_as_int64(kv_u8, Int64(0))
                    _st_global_u32(kv_base + core_row * Int64(_MLA_PACKED_DIM) + Int64(_MLA_ROPE_OFFSET_BYTES) + Int64(tx) * Int64(4), rope_word)

        lane = tx & Int32(31)
        warp = tx >> Int32(5)
        local_max = fmax_f32(fabs_f32(val0), fabs_f32(val1))
        for step in cutlass.range_constexpr(5):
            local_max = fmax_f32(local_max, cute.arch.shuffle_sync_bfly(local_max, offset=1 << step))
        if lane == Int32(0):
            s_warp_maxes[warp] = local_max
        cute.arch.sync_threads()

        scale = Float32(1.0)
        if warp == Int32(0):
            max_abs = Float32(0.0)
            if lane < Int32(2):
                max_abs = Float32(s_warp_maxes[lane])
            max_abs = fmax_f32(max_abs, cute.arch.shuffle_sync_bfly(max_abs, offset=1))
            if lane == Int32(0):
                if max_abs > Float32(0.0):
                    scale = max_abs * Float32(_FP8_MAX_RECIP)
                s_scale[Int32(0)] = scale
                core_row = Int64(row) * Int64(self.total_width) + Int64(slot)
                kv_base = get_ptr_as_int64(kv_u8, Int64(0))
                st_global_f32(kv_base + core_row * Int64(_MLA_PACKED_DIM) + Int64(_MLA_SCALE_OFFSET_BYTES) + Int64(group) * Int64(4), scale)
        cute.arch.sync_threads()

        out_scale = Float32(s_scale[Int32(0)])
        quant0 = cvt_f32_to_e4m3(val0 / out_scale)
        quant1 = cvt_f32_to_e4m3(val1 / out_scale)
        core_row = Int64(row) * Int64(self.total_width) + Int64(slot)
        kv_base = get_ptr_as_int64(kv_u8, Int64(0))
        st_global_u8(
            kv_base + core_row * Int64(_MLA_PACKED_DIM) + Int64(group) * Int64(128) + Int64(tx),
            Uint8(quant0 & Uint32(0xFF)),
        )
        st_global_u8(
            kv_base + core_row * Int64(_MLA_PACKED_DIM) + Int64(group) * Int64(128) + Int64(64) + Int64(tx),
            Uint8(quant1 & Uint32(0xFF)),
        )


@lru_cache(maxsize=256)
def _get_prep_kv_kernel(
    rows: int,
    total_width: int,
    swa_width: int,
    indexed_width: int,
    swa_page_size: int,
    swa_page_nbytes: int,
    indexed_page_size: int,
    indexed_page_nbytes: int,
    has_indexed: bool,
    map_indexed_page_table: bool,
    indexed_page_table_width: int,
) -> CompressedMLAPrepKVKernel:
    return CompressedMLAPrepKVKernel(
        rows=rows,
        total_width=total_width,
        swa_width=swa_width,
        indexed_width=indexed_width,
        swa_page_size=swa_page_size,
        swa_page_nbytes=swa_page_nbytes,
        indexed_page_size=indexed_page_size,
        indexed_page_nbytes=indexed_page_nbytes,
        has_indexed=has_indexed,
        map_indexed_page_table=map_indexed_page_table,
        indexed_page_table_width=indexed_page_table_width,
    )


def run_prepare_compressed_mla_kv_cute(
    *,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_valid_lengths: torch.Tensor,
    indexed_k_cache: torch.Tensor,
    indexed_indices: torch.Tensor,
    indexed_valid_lengths: torch.Tensor,
    indexed_page_table: torch.Tensor,
    kv_cache: torch.Tensor,
    rows: int,
    swa_width: int,
    indexed_width: int,
    total_width: int,
    swa_page_size: int,
    swa_page_nbytes: int,
    indexed_page_size: int,
    indexed_page_nbytes: int,
    has_indexed: bool,
    map_indexed_page_table: bool,
    indexed_page_table_width: int,
) -> None:
    """Run the CuTe DSL compressed KV gather/dequant/repack kernel."""

    if rows <= 0 or total_width <= 0:
        raise ValueError("rows and total_width must be positive")
    kernel = _get_prep_kv_kernel(
        rows,
        total_width,
        swa_width,
        indexed_width,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        has_indexed,
        map_indexed_page_table,
        indexed_page_table_width,
    )
    swa_u8 = swa_k_cache.reshape(-1)
    indexed_u8 = indexed_k_cache.reshape(-1)
    kv_u8 = kv_cache.reshape(-1)
    args = (
        _to_kernel_tensor(swa_u8, cutlass.Uint8),
        _to_kernel_tensor(swa_indices, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(swa_valid_lengths, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(indexed_u8, cutlass.Uint8),
        _to_kernel_tensor(indexed_indices, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(indexed_valid_lengths, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(indexed_page_table, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(kv_u8, cutlass.Uint8),
        current_stream(),
    )
    cache_key = (
        "compressed_mla_prep_kv_cute",
        rows,
        total_width,
        swa_width,
        indexed_width,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        has_indexed,
        map_indexed_page_table,
        indexed_page_table_width,
        _tensor_meta_key(swa_u8),
        _tensor_meta_key(swa_indices),
        _tensor_meta_key(indexed_u8),
        _tensor_meta_key(indexed_indices),
        _tensor_meta_key(indexed_page_table),
        _tensor_meta_key(kv_u8),
    )
    _run_cached_host_launcher(kernel, cache_key, args)
