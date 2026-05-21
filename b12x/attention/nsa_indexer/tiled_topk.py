"""Faithful CuTe DSL port of the sglang radix-select topk kernel.

Ported from: sgl-kernel/csrc/elementwise/topk.cu :: fast_topk_cuda_tl

Uses raw PTX shared memory loads/stores/atomics for the double-buffered
histogram prefix sum, since CuTe DSL does not support dynamic selection
between tensor objects. Variables in dynamic control flow require SSA-style
initialization before use.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import os
import warnings

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import (
    atomic_add_shared_i32,
    ld_shared_i32,
    ld_shared_i32_relaxed,
    red_add_shared_i32,
    shared_ptr_to_u32,
    st_shared_i32,
)
from b12x.cute.utils import current_cuda_stream
from b12x.runtime_control import raise_if_kernel_resolution_frozen

_EAGER_HOST_LAUNCHER_CACHE_SIZE = int(
    os.getenv("B12X_EAGER_HOST_LAUNCHER_CACHE_SIZE", "512")
)
_THREADS_PER_CTA = 1024
_DEFAULT_TOPK = 2048
_SUPPORTED_TOPK = (512, 2048)
_RADIX = 256
_SMEM_CANDS = 4096
_SCAN_UNROLL = 4
_SUPERTILE_K_ENV = "B12X_NSA_EXTEND_TOPK_SUPERTILE_K"
_SUPERTILE_K_DEFAULT = 32768


@dsl_user_op
def _cvt_rn_f16_f32(val: Float32, *, loc=None, ip=None) -> Uint32:
    result = llvm.inline_asm(
        T.i32(),
        [Float32(val).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16.f32 $0, $1;",
        "=h,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _float_as_uint32(val: Float32, *, loc=None, ip=None) -> Uint32:
    result = llvm.inline_asm(
        T.i32(),
        [Float32(val).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@cute.jit
def _smem_ld(base: Int32, idx: Int32) -> Int32:
    return ld_shared_i32(base + idx * Int32(4))


@cute.jit
def _smem_ld_relaxed(base: Int32, idx: Int32) -> Int32:
    return ld_shared_i32_relaxed(base + idx * Int32(4))


@cute.jit
def _smem_st(base: Int32, idx: Int32, val: Int32):
    st_shared_i32(base + idx * Int32(4), val)


@cute.jit
def _smem_xadd(base: Int32, idx: Int32, val: Int32) -> Int32:
    return atomic_add_shared_i32(base + idx * Int32(4), val)


@cute.jit
def _smem_red_add(base: Int32, idx: Int32, val: Int32):
    red_add_shared_i32(base + idx * Int32(4), val)


@cute.jit
def _load_topk_input_from_row_base(
    input_tensor,
    row_base: Int32,
    logical_k: Int32,
    block_q: cutlass.Constexpr[int],
    block_k: cutlass.Constexpr[int],
    is_tiled: cutlass.Constexpr[bool],
) -> Float32:
    value = Float32(0.0)
    if cutlass.const_expr(is_tiled):
        tile_size = Int32(block_q * block_k)
        k_tile_idx = Int32(0)
        k_local = Int32(0)
        if cutlass.const_expr(block_k == 256):
            k_tile_idx = logical_k >> Int32(8)
            k_local = logical_k & Int32(255)
        else:
            k_tile_idx = logical_k >> Int32(9)
            k_local = logical_k & Int32(511)
        value = Float32(input_tensor[row_base + k_tile_idx * tile_size + k_local])
    else:
        value = Float32(input_tensor[row_base + logical_k])
    return value


@cute.jit
def _convert_to_uint8(x: Float32) -> Uint32:
    h_bits = _cvt_rn_f16_f32(x)
    bits16 = h_bits & Uint32(0xFFFF)
    sign = bits16 & Uint32(0x8000)
    key16 = Uint32(0)
    if sign != Uint32(0):
        key16 = Uint32(0xFFFF) ^ bits16
    else:
        key16 = bits16 | Uint32(0x8000)
    return (key16 >> Uint32(8)) & Uint32(0xFF)


@cute.jit
def _convert_to_uint32(x: Float32) -> Uint32:
    bits = _float_as_uint32(x)
    sign = bits & Uint32(0x80000000)
    result = Uint32(0)
    if sign != Uint32(0):
        result = ~bits
    else:
        result = bits | Uint32(0x80000000)
    return result


def _to_kernel_tensor(tensor, dtype, *, assumed_align=16):
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    if tensor.ndim >= 2:
        leading_dim = next((idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None)
        if leading_dim is not None:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_meta_key(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _launcher_cache_lookup(kernel, cache_key):
    cache = getattr(kernel, "_eager_host_launchers", None)
    if cache is None:
        cache = OrderedDict()
        setattr(kernel, "_eager_host_launchers", cache)
        return cache, None
    compiled = cache.get(cache_key)
    if compiled is not None:
        cache.move_to_end(cache_key)
    return cache, compiled


def _run_cached_host_launcher(kernel, cache_key, args):
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


class SparseNSATiledTopkKernel:
    def __init__(
        self,
        *,
        is_tiled: bool = False,
        block_q: int = 1,
        block_k: int = 1,
        topk: int = _DEFAULT_TOPK,
        zero_row_start: bool = False,
    ):
        self.is_tiled = is_tiled
        self.block_q = int(block_q)
        self.block_k = int(block_k)
        self.topk = int(topk)
        self.zero_row_start = bool(zero_row_start)

    @cute.jit
    def __call__(
        self,
        input_tensor, row_starts, lengths, values, indices,
        batch_size, input_stride, num_k_tiles, tile_k_offset, block_q, block_k, topk_val,
        input_index_offset, input_extent, output_index_offset, stream,
    ):
        self.kernel(
            input_tensor, row_starts, lengths, values, indices,
            batch_size, input_stride, num_k_tiles, tile_k_offset, block_q, block_k, topk_val,
            input_index_offset, input_extent, output_index_offset,
        ).launch(
            grid=(batch_size, 1, 1),
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_tensor: cute.Tensor,
        row_starts: cute.Tensor,
        lengths: cute.Tensor,
        values: cute.Tensor,
        indices: cute.Tensor,
        batch_size: Int32,
        input_stride: Int32,
        num_k_tiles: Int32,
        tile_k_offset: Int32,
        block_q: Int32,
        block_k: Int32,
        topk_val: Int32,
        input_index_offset: Int32,
        input_extent: Int32,
        output_index_offset: Int32,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        bid = Int32(bid)

        row_start = Int32(0)
        if not cutlass.const_expr(self.zero_row_start):
            row_start = Int32(row_starts[bid])
        length = Int32(lengths[bid])
        if input_extent > Int32(0):
            row_end = row_start + length
            clipped_start = row_start
            if clipped_start < input_index_offset:
                clipped_start = input_index_offset
            clipped_end = row_end
            chunk_end = input_index_offset + input_extent
            if clipped_end > chunk_end:
                clipped_end = chunk_end
            if clipped_end > clipped_start:
                row_start = clipped_start - input_index_offset
                length = clipped_end - clipped_start
            else:
                row_start = Int32(0)
                length = Int32(0)
        topk_capacity = self.topk
        topk_static = Int32(self.topk)
        out_base = bid * topk_static
        row_base = bid * input_stride
        if cutlass.const_expr(self.is_tiled):
            block_q_i = Int32(self.block_q)
            block_k_i = Int32(self.block_k)
            tile_size = Int32(self.block_q * self.block_k)
            q_tile_idx = bid // block_q_i
            q_local = bid - q_tile_idx * block_q_i
            row_base = (
                q_tile_idx * num_k_tiles * tile_size
                + tile_k_offset * tile_size
                + q_local * block_k_i
            )

        smem_alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            hist0: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 384], 128]
            hist1: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 384], 128]
            out_idx: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, topk_capacity], 128]
            counter: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            thr_id: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            ni0: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            ni1: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            last_rem: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            cand0: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, _SMEM_CANDS], 128]
            cand1: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, _SMEM_CANDS], 128]

        storage = smem_alloc.allocate(SharedStorage)

        s_hist0 = storage.hist0.get_tensor(cute.make_layout((384,), stride=(1,)))
        s_hist1 = storage.hist1.get_tensor(cute.make_layout((384,), stride=(1,)))
        s_out = storage.out_idx.get_tensor(cute.make_layout((topk_capacity,), stride=(1,)))
        s_cand0 = storage.cand0.get_tensor(cute.make_layout((_SMEM_CANDS,), stride=(1,)))
        s_cand1 = storage.cand1.get_tensor(cute.make_layout((_SMEM_CANDS,), stride=(1,)))

        h0 = shared_ptr_to_u32(storage.hist0.data_ptr())
        h1 = shared_ptr_to_u32(storage.hist1.data_ptr())
        ctr = shared_ptr_to_u32(storage.counter.data_ptr())
        thr = shared_ptr_to_u32(storage.thr_id.data_ptr())
        ni0 = shared_ptr_to_u32(storage.ni0.data_ptr())
        ni1 = shared_ptr_to_u32(storage.ni1.data_ptr())
        lr = shared_ptr_to_u32(storage.last_rem.data_ptr())

        need_radix = length > topk_static

        if not need_radix:
            i = Int32(tx)
            while i < topk_static:
                is_valid = i < length
                values[out_base + i] = (
                    _load_topk_input_from_row_base(
                        input_tensor,
                        row_base,
                        row_start + i,
                        self.block_q,
                        self.block_k,
                        self.is_tiled,
                    )
                    if is_valid
                    else Float32(float("-inf"))
                )
                indices[out_base + i] = row_start + i + output_index_offset if is_valid else Int32(-1)
                i = i + Int32(_THREADS_PER_CTA)

        if need_radix:
            topk = topk_static

            if tx < Int32(257):
                s_hist0[tx] = Int32(0)
            cute.arch.sync_threads()

            idx_base = Int32(tx)
            full_scan_limit = length - Int32((_SCAN_UNROLL - 1) * _THREADS_PER_CTA)
            while idx_base < full_scan_limit:
                for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
                    idx = idx_base + Int32(scan_u * _THREADS_PER_CTA)
                    val = _load_topk_input_from_row_base(
                        input_tensor,
                        row_base,
                        row_start + idx,
                        self.block_q,
                        self.block_k,
                        self.is_tiled,
                    )
                    bin8 = _convert_to_uint8(val)
                    _smem_red_add(h0, Int32(bin8), Int32(1))
                idx_base = idx_base + Int32(_THREADS_PER_CTA * _SCAN_UNROLL)
            while idx_base < length:
                val = _load_topk_input_from_row_base(
                    input_tensor,
                    row_base,
                    row_start + idx_base,
                    self.block_q,
                    self.block_k,
                    self.is_tiled,
                )
                bin8 = _convert_to_uint8(val)
                _smem_red_add(h0, Int32(bin8), Int32(1))
                idx_base = idx_base + Int32(_THREADS_PER_CTA)

            cute.arch.sync_threads()

            # Parallel prefix sum: 8 stages, double-buffer h0/h1
            for stage in cutlass.range_constexpr(8):
                j = Int32(1 << stage)
                if tx < Int32(256):
                    if (stage & 1) == 0:
                        value = Int32(s_hist0[tx])
                        if tx < Int32(256) - j:
                            value = value + Int32(s_hist0[tx + j])
                        s_hist1[tx] = value
                    else:
                        value = Int32(s_hist1[tx])
                        if tx < Int32(256) - j:
                            value = value + Int32(s_hist1[tx + j])
                        s_hist0[tx] = value
                cute.arch.sync_threads()

            # Find threshold bin
            if tx < Int32(256):
                val_tx = Int32(s_hist0[tx])
                val_tx1 = Int32(s_hist0[tx + Int32(1)])
                if val_tx > topk:
                    if val_tx1 <= topk:
                        _smem_st(thr, Int32(0), Int32(tx))
                        _smem_st(ni0, Int32(0), Int32(0))
                        _smem_st(ctr, Int32(0), Int32(0))

            cute.arch.sync_threads()
            threshold_bin = _smem_ld(thr, Int32(0))
            topk = topk - Int32(s_hist0[threshold_bin + Int32(1)])

            if topk == Int32(0):
                idx_base = Int32(tx)
                full_scan_limit = length - Int32((_SCAN_UNROLL - 1) * _THREADS_PER_CTA)
                while idx_base < full_scan_limit:
                    for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
                        idx = idx_base + Int32(scan_u * _THREADS_PER_CTA)
                        val = _load_topk_input_from_row_base(
                            input_tensor,
                            row_base,
                            row_start + idx,
                            self.block_q,
                            self.block_k,
                            self.is_tiled,
                        )
                        bin8 = _convert_to_uint8(val)
                        if Int32(bin8) > threshold_bin:
                            pos = _smem_xadd(ctr, Int32(0), Int32(1))
                            s_out[pos] = idx
                    idx_base = idx_base + Int32(_THREADS_PER_CTA * _SCAN_UNROLL)
                while idx_base < length:
                    val = _load_topk_input_from_row_base(
                        input_tensor,
                        row_base,
                        row_start + idx_base,
                        self.block_q,
                        self.block_k,
                        self.is_tiled,
                    )
                    bin8 = _convert_to_uint8(val)
                    if Int32(bin8) > threshold_bin:
                        pos = _smem_xadd(ctr, Int32(0), Int32(1))
                        s_out[pos] = idx_base
                    idx_base = idx_base + Int32(_THREADS_PER_CTA)

            if topk != Int32(0):
                cute.arch.sync_threads()

                if tx < Int32(257):
                    s_hist0[tx] = Int32(0)
                cute.arch.sync_threads()

                idx_base = Int32(tx)
                full_scan_limit = length - Int32((_SCAN_UNROLL - 1) * _THREADS_PER_CTA)
                while idx_base < full_scan_limit:
                    for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
                        idx = idx_base + Int32(scan_u * _THREADS_PER_CTA)
                        raw_input = _load_topk_input_from_row_base(
                            input_tensor,
                            row_base,
                            row_start + idx,
                            self.block_q,
                            self.block_k,
                            self.is_tiled,
                        )
                        bin8 = _convert_to_uint8(raw_input)
                        if Int32(bin8) > threshold_bin:
                            pos = _smem_xadd(ctr, Int32(0), Int32(1))
                            s_out[pos] = idx
                        else:
                            if Int32(bin8) == threshold_bin:
                                cand_pos = _smem_xadd(ni0, Int32(0), Int32(1))
                                if cand_pos < Int32(_SMEM_CANDS):
                                    s_cand0[cand_pos] = idx
                                    key32 = _convert_to_uint32(raw_input)
                                    sub_bin = (key32 >> Uint32(24)) & Uint32(0xFF)
                                    _smem_red_add(h0, Int32(sub_bin), Int32(1))
                    idx_base = idx_base + Int32(_THREADS_PER_CTA * _SCAN_UNROLL)
                while idx_base < length:
                    raw_input = _load_topk_input_from_row_base(
                        input_tensor,
                        row_base,
                        row_start + idx_base,
                        self.block_q,
                        self.block_k,
                        self.is_tiled,
                    )
                    bin8 = _convert_to_uint8(raw_input)
                    if Int32(bin8) > threshold_bin:
                        pos = _smem_xadd(ctr, Int32(0), Int32(1))
                        s_out[pos] = idx_base
                    else:
                        if Int32(bin8) == threshold_bin:
                            cand_pos = _smem_xadd(ni0, Int32(0), Int32(1))
                            if cand_pos < Int32(_SMEM_CANDS):
                                s_cand0[cand_pos] = idx_base
                                key32 = _convert_to_uint32(raw_input)
                                sub_bin = (key32 >> Uint32(24)) & Uint32(0xFF)
                                _smem_red_add(h0, Int32(sub_bin), Int32(1))
                    idx_base = idx_base + Int32(_THREADS_PER_CTA)

                cute.arch.sync_threads()

                # Stage 2: refine with 8-bit radix passes
                for round_idx in cutlass.range_constexpr(4):
                    if topk != Int32(-1):
                        r_idx_is_0 = (round_idx % 2) == 0
                        r_idx_next_is_0 = not r_idx_is_0

                        raw_num_input = _smem_ld(ni0, Int32(0)) if cutlass.const_expr(r_idx_is_0) else _smem_ld(ni1, Int32(0))
                        num_input = raw_num_input if raw_num_input < Int32(_SMEM_CANDS) else Int32(_SMEM_CANDS)

                        # Prefix sum
                        for stage in cutlass.range_constexpr(8):
                            j = Int32(1 << stage)
                            if tx < Int32(256):
                                if (stage & 1) == 0:
                                    value = Int32(s_hist0[tx])
                                    if tx < Int32(256) - j:
                                        value = value + Int32(s_hist0[tx + j])
                                    s_hist1[tx] = value
                                else:
                                    value = Int32(s_hist1[tx])
                                    if tx < Int32(256) - j:
                                        value = value + Int32(s_hist1[tx + j])
                                    s_hist0[tx] = value
                            cute.arch.sync_threads()

                        if tx < Int32(256):
                            val_tx = Int32(s_hist0[tx])
                            val_tx1 = Int32(s_hist0[tx + Int32(1)])
                            if val_tx > topk:
                                if val_tx1 <= topk:
                                    _smem_st(thr, Int32(0), Int32(tx))
                                    if cutlass.const_expr(r_idx_next_is_0):
                                        _smem_st(ni0, Int32(0), Int32(0))
                                    else:
                                        _smem_st(ni1, Int32(0), Int32(0))
                                    _smem_st(lr, Int32(0), topk - val_tx1)

                        cute.arch.sync_threads()

                        sub_threshold = _smem_ld(thr, Int32(0))
                        topk = topk - Int32(s_hist0[sub_threshold + Int32(1)])

                        # Quick exit
                        if topk == Int32(0):
                            i = Int32(tx)
                            while i < num_input:
                                c_idx = Int32(s_cand0[i]) if cutlass.const_expr(r_idx_is_0) else Int32(s_cand1[i])
                                offset = Int32(24 - round_idx * 8)
                                raw_val = _load_topk_input_from_row_base(
                                    input_tensor,
                                    row_base,
                                    row_start + c_idx,
                                    self.block_q,
                                    self.block_k,
                                    self.is_tiled,
                                )
                                key32 = _convert_to_uint32(raw_val)
                                bin = (key32 >> Uint32(offset)) & Uint32(0xFF)
                                if Int32(bin) > sub_threshold:
                                    pos = _smem_xadd(ctr, Int32(0), Int32(1))
                                    s_out[pos] = c_idx
                                i = i + Int32(_THREADS_PER_CTA)
                            topk = Int32(-1)

                        # Continue refinement
                        if topk != Int32(-1):
                            cute.arch.sync_threads()

                            if tx < Int32(257):
                                s_hist0[tx] = Int32(0)
                            cute.arch.sync_threads()

                            i = Int32(tx)
                            while i < num_input:
                                c_idx = Int32(s_cand0[i]) if cutlass.const_expr(r_idx_is_0) else Int32(s_cand1[i])
                                raw_val = _load_topk_input_from_row_base(
                                    input_tensor,
                                    row_base,
                                    row_start + c_idx,
                                    self.block_q,
                                    self.block_k,
                                    self.is_tiled,
                                )
                                offset = Int32(24 - round_idx * 8)
                                key32 = _convert_to_uint32(raw_val)
                                bin = (key32 >> Uint32(offset)) & Uint32(0xFF)

                                if Int32(bin) > sub_threshold:
                                    pos = _smem_xadd(ctr, Int32(0), Int32(1))
                                    s_out[pos] = c_idx
                                else:
                                    if Int32(bin) == sub_threshold:
                                        if cutlass.const_expr(round_idx == 3):
                                            old_rem = _smem_xadd(lr, Int32(0), Int32(-1))
                                            if old_rem > Int32(0):
                                                s_out[topk_static - old_rem] = c_idx
                                        else:
                                            cand_pos = _smem_xadd(ni0, Int32(0), Int32(1)) if cutlass.const_expr(r_idx_next_is_0) else _smem_xadd(ni1, Int32(0), Int32(1))
                                            if cand_pos < Int32(_SMEM_CANDS):
                                                if cutlass.const_expr(r_idx_next_is_0):
                                                    s_cand0[cand_pos] = c_idx
                                                else:
                                                    s_cand1[cand_pos] = c_idx
                                                sub_bin = (key32 >> Uint32(24 - (round_idx + 1) * 8)) & Uint32(0xFF)
                                                _smem_red_add(h0, Int32(sub_bin), Int32(1))

                                i = i + Int32(_THREADS_PER_CTA)

                            cute.arch.sync_threads()

            cute.arch.sync_threads()
            idx0 = Int32(tx)
            if idx0 < topk_static:
                selected0 = Int32(s_out[idx0])
                values[out_base + idx0] = _load_topk_input_from_row_base(
                    input_tensor,
                    row_base,
                    row_start + selected0,
                    self.block_q,
                    self.block_k,
                    self.is_tiled,
                )
                indices[out_base + idx0] = row_start + selected0 + output_index_offset
            idx1 = idx0 + Int32(_THREADS_PER_CTA)
            if idx1 < topk_static:
                selected1 = Int32(s_out[idx1])
                values[out_base + idx1] = _load_topk_input_from_row_base(
                    input_tensor,
                    row_base,
                    row_start + selected1,
                    self.block_q,
                    self.block_k,
                    self.is_tiled,
                )
                indices[out_base + idx1] = row_start + selected1 + output_index_offset


@lru_cache(maxsize=32)
def _build_tiled_topk_kernel(block_q: int, block_k: int, topk: int, zero_row_start: bool = False):
    return SparseNSATiledTopkKernel(
        is_tiled=True,
        block_q=block_q,
        block_k=block_k,
        topk=topk,
        zero_row_start=zero_row_start,
    )


@lru_cache(maxsize=8)
def _build_row_topk_kernel(topk: int):
    return SparseNSATiledTopkKernel(
        is_tiled=False,
        block_q=1,
        block_k=1,
        topk=topk,
        zero_row_start=True,
    )


def clear_tiled_topk_kernel_cache() -> None:
    _build_tiled_topk_kernel.cache_clear()
    _build_row_topk_kernel.cache_clear()


def _validate_supported_topk(topk: int, *, caller: str) -> int:
    topk = int(topk)
    if topk not in _SUPPORTED_TOPK:
        raise ValueError(
            f"{caller} supports topk values {_SUPPORTED_TOPK}, got topk={topk}"
        )
    return topk


def run_tiled_topk(
    *,
    tile_logits: torch.Tensor,
    k_start: torch.Tensor | None,
    k_end: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    topk: int,
    block_q: int,
    block_k: int,
    output_values: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
    num_k_tiles: int | None = None,
    tile_k_offset: int = 0,
    input_index_offset: int = 0,
    input_extent: int = 0,
    output_index_offset: int = 0,
    zero_row_start: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk = _validate_supported_topk(topk, caller="run_tiled_topk")
    if k_end is None and lengths is None:
        raise ValueError("run_tiled_topk requires either k_end or lengths")
    if not tile_logits.is_contiguous():
        raise ValueError("tile_logits must be contiguous")
    if k_start is None:
        if not zero_row_start:
            raise ValueError("run_tiled_topk requires k_start unless zero_row_start is true")
        if lengths is None:
            raise ValueError("run_tiled_topk zero_row_start requires explicit lengths")
        k_start = lengths
    if not k_start.is_contiguous():
        raise ValueError("k_start must be contiguous")
    if lengths is None:
        if k_end is None:
            raise AssertionError("unreachable")
        lengths = k_end - k_start
    elif not lengths.is_contiguous():
        raise ValueError("lengths must be contiguous")

    num_q_rows = int(k_start.shape[0])
    num_q_tiles = (num_q_rows + block_q - 1) // block_q
    tile_size = block_q * block_k
    total_elements = int(tile_logits.shape[0])
    if num_k_tiles is None:
        num_k_tiles = total_elements // (num_q_tiles * tile_size)
        if num_k_tiles == 0:
            num_k_tiles = getattr(tile_logits, '_b12x_num_k_tiles', None)
            if num_k_tiles is None:
                raise ValueError("Cannot determine num_k_tiles")
    if int(num_k_tiles) <= 0:
        raise ValueError("Cannot determine num_k_tiles")

    if output_indices is None:
        topk_indices = torch.empty(
            (num_q_rows, topk), dtype=torch.int32, device=tile_logits.device,
        )
    else:
        if output_indices.shape != (num_q_rows, topk):
            raise ValueError(
                f"output_indices must have shape {(num_q_rows, topk)}, got {tuple(output_indices.shape)}"
            )
        if not output_indices.is_contiguous():
            raise ValueError("output_indices must be contiguous")
        topk_indices = output_indices
    if output_values is None:
        topk_values = torch.empty(
            (num_q_rows, topk), dtype=torch.float32, device=tile_logits.device,
        )
    else:
        if output_values.shape != (num_q_rows, topk):
            raise ValueError(
                f"output_values must have shape {(num_q_rows, topk)}, got {tuple(output_values.shape)}"
            )
        if not output_values.is_contiguous():
            raise ValueError("output_values must be contiguous")
        topk_values = output_values

    input_stride = Int32(0)
    flat_input = tile_logits.reshape(-1)
    flat_values = topk_values.reshape(-1).contiguous()
    flat_indices = topk_indices.reshape(-1).contiguous()

    kernel = _build_tiled_topk_kernel(block_q, block_k, topk, bool(zero_row_start))
    args = (
        _to_kernel_tensor(flat_input, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_start, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(lengths, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(flat_values, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(flat_indices, cutlass.Int32, assumed_align=4),
        Int32(num_q_rows),
        input_stride,
        Int32(num_k_tiles),
        Int32(tile_k_offset),
        Int32(block_q),
        Int32(block_k),
        Int32(topk),
        Int32(input_index_offset),
        Int32(input_extent),
        Int32(output_index_offset),
        current_cuda_stream(),
    )
    cache_key = (
        _tensor_meta_key(flat_input),
        _tensor_meta_key(k_start),
        _tensor_meta_key(lengths),
        _tensor_meta_key(flat_values),
        _tensor_meta_key(flat_indices),
        (
            "tiled_topk_v17",
            topk,
            block_q,
            block_k,
            bool(zero_row_start),
        ),
    )
    _run_cached_host_launcher(kernel, cache_key, args)
    return topk_values, topk_indices


def run_row_topk(
    *,
    row_logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    output_values: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
    output_index_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact row-wise topk over a dense row-major logits tile."""
    topk = _validate_supported_topk(topk, caller="run_row_topk")
    if row_logits.ndim != 2:
        raise ValueError(f"row_logits must be rank-2, got {tuple(row_logits.shape)}")
    if not row_logits.is_contiguous():
        raise ValueError("row_logits must be contiguous")
    if row_logits.dtype != torch.float32:
        raise ValueError(f"row_logits must have dtype torch.float32, got {row_logits.dtype}")
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be rank-1, got {tuple(lengths.shape)}")
    if lengths.dtype != torch.int32:
        raise ValueError(f"lengths must have dtype torch.int32, got {lengths.dtype}")
    if not lengths.is_contiguous():
        raise ValueError("lengths must be contiguous")
    num_q_rows = int(row_logits.shape[0])
    width = int(row_logits.shape[1])
    if lengths.shape[0] != num_q_rows:
        raise ValueError(
            f"lengths rows {lengths.shape[0]} do not match row_logits rows {num_q_rows}"
        )

    if output_indices is None:
        topk_indices = torch.empty(
            (num_q_rows, topk), dtype=torch.int32, device=row_logits.device,
        )
    else:
        if output_indices.shape != (num_q_rows, topk):
            raise ValueError(
                f"output_indices must have shape {(num_q_rows, topk)}, got {tuple(output_indices.shape)}"
            )
        if not output_indices.is_contiguous():
            raise ValueError("output_indices must be contiguous")
        topk_indices = output_indices
    if output_values is None:
        topk_values = torch.empty(
            (num_q_rows, topk), dtype=torch.float32, device=row_logits.device,
        )
    else:
        if output_values.shape != (num_q_rows, topk):
            raise ValueError(
                f"output_values must have shape {(num_q_rows, topk)}, got {tuple(output_values.shape)}"
            )
        if not output_values.is_contiguous():
            raise ValueError("output_values must be contiguous")
        topk_values = output_values

    flat_input = row_logits.reshape(-1)
    flat_values = topk_values.reshape(-1).contiguous()
    flat_indices = topk_indices.reshape(-1).contiguous()
    kernel = _build_row_topk_kernel(topk)
    args = (
        _to_kernel_tensor(flat_input, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(lengths, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(lengths, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(flat_values, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(flat_indices, cutlass.Int32, assumed_align=4),
        Int32(num_q_rows),
        Int32(width),
        Int32(0),
        Int32(0),
        Int32(1),
        Int32(1),
        Int32(topk),
        Int32(0),
        Int32(width),
        Int32(output_index_offset),
        current_cuda_stream(),
    )
    cache_key = (
        _tensor_meta_key(flat_input),
        _tensor_meta_key(lengths),
        _tensor_meta_key(flat_values),
        _tensor_meta_key(flat_indices),
        (
            "row_topk_v1",
            topk,
        ),
    )
    _run_cached_host_launcher(kernel, cache_key, args)
    return topk_values, topk_indices


def _resolve_supertile_k(supertile_k: int | None, *, block_k: int) -> int:
    if supertile_k is None:
        raw = os.environ.get(_SUPERTILE_K_ENV)
        if raw is None:
            supertile_k = _SUPERTILE_K_DEFAULT
        else:
            try:
                supertile_k = int(raw)
            except ValueError as exc:
                raise ValueError(f"{_SUPERTILE_K_ENV} must be an integer, got {raw!r}") from exc
    supertile_k = max(int(supertile_k), int(block_k))
    return ((supertile_k + block_k - 1) // block_k) * block_k


def merge_tiled_topk_candidates(
    *,
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge per-supertile exact topk candidates into one row-wise topk."""
    if candidate_values.shape != candidate_indices.shape:
        raise ValueError(
            "candidate_values and candidate_indices must have the same shape, got "
            f"{tuple(candidate_values.shape)} vs {tuple(candidate_indices.shape)}"
        )
    if candidate_values.ndim != 3:
        raise ValueError(f"candidates must have shape (chunks, rows, topk), got {tuple(candidate_values.shape)}")
    num_chunks, num_q_rows, local_topk = candidate_values.shape
    if int(local_topk) != int(topk):
        raise ValueError(f"candidate local topk {local_topk} does not match requested topk {topk}")
    candidate_cols = int(num_chunks) * int(topk)
    candidate_values_2d = candidate_values.permute(1, 0, 2).reshape(num_q_rows, candidate_cols)
    candidate_indices_2d = candidate_indices.permute(1, 0, 2).reshape(num_q_rows, candidate_cols)
    merge_pos = torch.topk(candidate_values_2d, k=topk, dim=1, largest=True, sorted=False).indices
    topk_indices = torch.gather(candidate_indices_2d, 1, merge_pos).contiguous()
    topk_values = torch.gather(candidate_values_2d, 1, merge_pos).contiguous()
    return topk_values, topk_indices


def run_tiled_supertile_topk(
    *,
    tile_logits: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    topk: int,
    block_q: int,
    block_k: int,
    supertile_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact topk over tiled logits by local-selecting K supertiles then merging candidates."""
    topk = _validate_supported_topk(topk, caller="run_tiled_supertile_topk")
    if not tile_logits.is_contiguous():
        raise ValueError("tile_logits must be contiguous")
    if not k_start.is_contiguous() or not k_end.is_contiguous():
        raise ValueError("k_start and k_end must be contiguous")

    num_q_rows = int(k_start.shape[0])
    num_q_tiles = (num_q_rows + block_q - 1) // block_q
    tile_size = block_q * block_k
    total_elements = int(tile_logits.shape[0])
    num_k_tiles = total_elements // (num_q_tiles * tile_size)
    if num_k_tiles == 0:
        num_k_tiles = getattr(tile_logits, "_b12x_num_k_tiles", None)
        if num_k_tiles is None:
            raise ValueError("Cannot determine num_k_tiles")
    resolved_supertile_k = _resolve_supertile_k(supertile_k, block_k=block_k)
    supertile_tiles = max(1, resolved_supertile_k // block_k)
    num_chunks = (int(num_k_tiles) + supertile_tiles - 1) // supertile_tiles
    if num_chunks <= 1:
        return run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            k_end=k_end,
            topk=topk,
            block_q=block_q,
            block_k=block_k,
        )

    candidate_values = torch.empty((num_chunks, num_q_rows, topk), dtype=torch.float32, device=tile_logits.device)
    candidate_indices = torch.empty((num_chunks, num_q_rows, topk), dtype=torch.int32, device=tile_logits.device)
    global_lengths = (k_end - k_start).contiguous()

    for chunk_idx in range(num_chunks):
        chunk_tile_begin = chunk_idx * supertile_tiles
        chunk_tile_end = min(chunk_tile_begin + supertile_tiles, int(num_k_tiles))
        chunk_start = chunk_tile_begin * block_k
        chunk_rows = (chunk_tile_end - chunk_tile_begin) * block_k
        run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            lengths=global_lengths,
            topk=topk,
            block_q=block_q,
            block_k=block_k,
            output_values=candidate_values[chunk_idx],
            output_indices=candidate_indices[chunk_idx],
            num_k_tiles=int(num_k_tiles),
            tile_k_offset=chunk_tile_begin,
            input_index_offset=chunk_start,
            input_extent=chunk_rows,
            output_index_offset=chunk_start,
        )

    return merge_tiled_topk_candidates(
        candidate_values=candidate_values,
        candidate_indices=candidate_indices,
        topk=topk,
    )
