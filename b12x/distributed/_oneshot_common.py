"""Shared low-level helpers for PCIe oneshot-style kernels."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

from b12x.cute.fp4 import warp_reduce

SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
MAX_BLOCKS = 36
THREADS_PER_BLOCK = 32
FLAG_STRIDE = 32
SELF_COUNTER_WORDS = MAX_BLOCKS * 8
PEER_BLOCK_WORDS = 16 * FLAG_STRIDE
PEER_PHASE_WORDS = MAX_BLOCKS * PEER_BLOCK_WORDS
PEER_COUNTER_BASE_WORDS = SELF_COUNTER_WORDS
SIGNAL_BYTES = (SELF_COUNTER_WORDS + 2 * PEER_PHASE_WORDS) * 4


def cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"unsupported dtype {dtype}")


def align_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


@dsl_user_op
def ptr_to_int64(ptr: cute.Pointer, *, loc=None, ip=None) -> Int64:
    return Int64(llvm.ptrtoint(T.i64(), ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def threadfence_system(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "membar.sys;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def sqrt_f32(x, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "sqrt.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ld_global_relaxed_sys_i32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def st_global_relaxed_sys_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.relaxed.sys.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def add_f32(a, b):
    return cutlass.Float32(a) + cutlass.Float32(b)


@cute.jit
def reduce_peer_row_sum(
    *,
    inputs: list[cute.Tensor],
    world_size: int,
    bidx: Int32,
    col: Int32,
    element_dtype,
):
    acc = element_dtype(inputs[0][bidx, col])
    if cutlass.const_expr(world_size > 1):
        acc = element_dtype(acc + element_dtype(inputs[1][bidx, col]))
    if cutlass.const_expr(world_size > 2):
        acc = element_dtype(acc + element_dtype(inputs[2][bidx, col]))
    if cutlass.const_expr(world_size > 3):
        acc = element_dtype(acc + element_dtype(inputs[3][bidx, col]))
    if cutlass.const_expr(world_size > 4):
        acc = element_dtype(acc + element_dtype(inputs[4][bidx, col]))
    if cutlass.const_expr(world_size > 5):
        acc = element_dtype(acc + element_dtype(inputs[5][bidx, col]))
    if cutlass.const_expr(world_size > 6):
        acc = element_dtype(acc + element_dtype(inputs[6][bidx, col]))
    if cutlass.const_expr(world_size > 7):
        acc = element_dtype(acc + element_dtype(inputs[7][bidx, col]))
    return acc


@cute.jit
def wait_for_peer_signals(
    *,
    signal_ptrs: list[cute.Pointer],
    self_signal: cute.Pointer,
    rank: Int32,
    world_size: int,
    bidx: Int32,
    tidx: Int32,
):
    signal_bases = [ptr_to_int64(ptr) for ptr in signal_ptrs]
    self_signal_base = ptr_to_int64(self_signal)
    if tidx < Int32(world_size):
        threadfence_system()
        self_counter_idx = bidx * Int32(8) + tidx
        self_counter_addr = self_signal_base + Int64(self_counter_idx) * Int64(4)
        counter = ld_global_relaxed_sys_i32(self_counter_addr) + Int32(1)
        st_global_relaxed_sys_i32(self_counter_addr, counter)

        phase = counter % Int32(2)
        peer_block_base = (
            Int32(PEER_COUNTER_BASE_WORDS)
            + phase * Int32(PEER_PHASE_WORDS)
            + bidx * Int32(PEER_BLOCK_WORDS)
        )
        remote_signal_base = signal_bases[0]
        if cutlass.const_expr(world_size > 1):
            if tidx == Int32(1):
                remote_signal_base = signal_bases[1]
        if cutlass.const_expr(world_size > 2):
            if tidx == Int32(2):
                remote_signal_base = signal_bases[2]
        if cutlass.const_expr(world_size > 3):
            if tidx == Int32(3):
                remote_signal_base = signal_bases[3]
        if cutlass.const_expr(world_size > 4):
            if tidx == Int32(4):
                remote_signal_base = signal_bases[4]
        if cutlass.const_expr(world_size > 5):
            if tidx == Int32(5):
                remote_signal_base = signal_bases[5]
        if cutlass.const_expr(world_size > 6):
            if tidx == Int32(6):
                remote_signal_base = signal_bases[6]
        if cutlass.const_expr(world_size > 7):
            if tidx == Int32(7):
                remote_signal_base = signal_bases[7]
        remote_addr = remote_signal_base + Int64(peer_block_base + rank * Int32(FLAG_STRIDE)) * Int64(4)
        local_wait_addr = self_signal_base + Int64(peer_block_base + tidx * Int32(FLAG_STRIDE)) * Int64(4)
        st_global_relaxed_sys_i32(remote_addr, counter)
        while ld_global_relaxed_sys_i32(local_wait_addr) != counter:
            pass
    cute.arch.barrier()


__all__ = [
    "SUPPORTED_WORLD_SIZES",
    "MAX_BLOCKS",
    "THREADS_PER_BLOCK",
    "SIGNAL_BYTES",
    "align_bytes",
    "cutlass_dtype",
    "add_f32",
    "reduce_peer_row_sum",
    "wait_for_peer_signals",
    "sqrt_f32",
    "warp_reduce",
]
