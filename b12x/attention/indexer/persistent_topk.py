"""Persistent large-N TopK=2048 selector for NSA decode logits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32

from b12x.cute.fp4 import (
    atomic_add_global_i32,
    get_ptr_as_int64,
    red_add_global_release_i32,
    shared_ptr_to_u32,
    spin_wait_global_ge_i32,
    st_global_release_i32,
)
from b12x.cute.utils import current_cuda_stream

from .tiled_topk import (
    _convert_to_uint32,
    _run_cached_host_launcher,
    _tensor_meta_key,
    _to_kernel_tensor,
    _smem_red_add,
    _smem_xadd,
)

_TOPK = 2048
_RADIX = 256
_THREADS_PER_CTA = 1024
_RADIX_THRESHOLD = 32768
_MAX_CHUNK_ELEMENTS = 8192
_STATE_WORDS = 3 * _RADIX + 4
_STATE_HIST0 = 0
_STATE_HIST1 = _RADIX
_STATE_HIST2 = 2 * _RADIX
_STATE_REMAINING_K = 3 * _RADIX
_STATE_PREFIX = _STATE_REMAINING_K + 1
_STATE_ARRIVAL_COUNTER = _STATE_REMAINING_K + 2
_STATE_OUTPUT_COUNTER = _STATE_REMAINING_K + 3


@dataclass(frozen=True)
class _LaunchConfig:
    chunk_size: int
    ctas_per_group: int
    num_groups: int
    total_ctas: int


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _resolve_vec_size(stride: int) -> int:
    if stride % 4 == 0:
        return 4
    if stride % 2 == 0:
        return 2
    return 1


def _resolve_launch_config(num_rows: int, stride: int, device: torch.device) -> _LaunchConfig:
    vec_size = _resolve_vec_size(stride)
    max_chunk = (_MAX_CHUNK_ELEMENTS // vec_size) * vec_size
    max_chunk = max(max_chunk, vec_size * _THREADS_PER_CTA)
    ctas_per_group = max(1, (stride + max_chunk - 1) // max_chunk)
    chunk_size = _align_up((stride + ctas_per_group - 1) // ctas_per_group, vec_size)
    chunk_size = min(chunk_size, max_chunk)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_groups = min(num_rows, max(1, num_sms // ctas_per_group))
    total_ctas = max(1, num_groups * ctas_per_group)
    return _LaunchConfig(
        chunk_size=chunk_size,
        ctas_per_group=ctas_per_group,
        num_groups=num_groups,
        total_ctas=total_ctas,
    )


def persistent_topk2048_workspace_nbytes(
    num_rows: int,
    stride: int,
    *,
    device: torch.device | str | None = None,
) -> int:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    launch = _resolve_launch_config(int(num_rows), int(stride), device)
    return launch.num_groups * _STATE_WORDS * torch.empty((), dtype=torch.int32).element_size()


@cute.jit
def _state_offset(group_id: Int32, word: Int32) -> Int32:
    return group_id * Int32(_STATE_WORDS) + word


@cute.jit
def _global_state_ptr(state: cute.Tensor, group_id: Int32, word: Int32):
    return get_ptr_as_int64(state, _state_offset(group_id, word))


@cute.jit
def _group_barrier(state: cute.Tensor, group_id: Int32, phase: Int32, ctas_per_group: Int32, tx: Int32) -> Int32:
    arrival_ptr = _global_state_ptr(state, group_id, Int32(_STATE_ARRIVAL_COUNTER))
    if tx == Int32(0):
        red_add_global_release_i32(arrival_ptr, Int32(1))
        spin_wait_global_ge_i32(arrival_ptr, (phase + Int32(1)) * ctas_per_group)
    cute.arch.sync_threads()
    return phase + Int32(1)


@cute.jit
def _store_output_index(
    output: cute.Tensor,
    page_table: cute.Tensor,
    row_idx: Int32,
    output_pos: Int32,
    logical_idx: Int32,
    page_table_stride: Int32,
    paged_output: cutlass.Constexpr[bool],
):
    out_val = logical_idx
    if cutlass.const_expr(paged_output):
        out_val = Int32(page_table[row_idx * page_table_stride + logical_idx])
    output[row_idx * Int32(_TOPK) + output_pos] = out_val


class SparseNSAPersistentTopK2048Kernel:
    def __init__(self, *, paged_output: bool):
        self.paged_output = bool(paged_output)

    @cute.jit
    def __call__(
        self,
        logits,
        lengths,
        page_table,
        output,
        state,
        num_rows,
        stride,
        chunk_size,
        ctas_per_group,
        num_groups,
        page_table_stride,
        stream,
    ):
        self.kernel(
            logits,
            lengths,
            page_table,
            output,
            state,
            num_rows,
            stride,
            chunk_size,
            ctas_per_group,
            num_groups,
            page_table_stride,
        ).launch(
            grid=(num_groups * ctas_per_group, 1, 1),
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        lengths: cute.Tensor,
        page_table: cute.Tensor,
        output: cute.Tensor,
        state: cute.Tensor,
        num_rows: Int32,
        stride: Int32,
        chunk_size: Int32,
        ctas_per_group: Int32,
        num_groups: Int32,
        page_table_stride: Int32,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        tx = Int32(tx)
        bid = Int32(bid)
        group_id = bid // ctas_per_group
        cta_in_group = bid - group_id * ctas_per_group

        smem_alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            local_histogram: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, _RADIX], 128]
            suffix_sum: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, _RADIX], 128]
            scalars: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, 4], 128]
            counters: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2], 128]
            ordered: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, _MAX_CHUNK_ELEMENTS], 128]

        storage = smem_alloc.allocate(SharedStorage)
        s_hist = storage.local_histogram.get_tensor(cute.make_layout((_RADIX,), stride=(1,)))
        s_suffix = storage.suffix_sum.get_tensor(cute.make_layout((_RADIX,), stride=(1,)))
        s_scalars = storage.scalars.get_tensor(cute.make_layout((4,), stride=(1,)))
        s_counters = storage.counters.get_tensor(cute.make_layout((2,), stride=(1,)))
        s_ordered = storage.ordered.get_tensor(cute.make_layout((_MAX_CHUNK_ELEMENTS,), stride=(1,)))
        hist_ptr = shared_ptr_to_u32(storage.local_histogram.data_ptr())
        counter_ptr = shared_ptr_to_u32(storage.counters.data_ptr())

        if cta_in_group == Int32(0):
            i = tx
            while i < Int32(_STATE_WORDS):
                state[_state_offset(group_id, i)] = Int32(0)
                i = i + Int32(_THREADS_PER_CTA)
        cute.arch.sync_threads()

        barrier_phase = Int32(0)
        total_iters = (num_rows + num_groups - Int32(1)) // num_groups
        iter_idx = Int32(0)
        row_idx = Int32(0)
        seq_len = Int32(0)
        row_base = Int32(0)
        row_out_base = Int32(0)
        chunk_start = Int32(0)
        chunk_end = Int32(0)
        actual_chunk_size = Int32(0)
        val = Float32(0.0)
        ordered = Uint32(0)
        mask = Uint32(0)
        bucket = Int32(0)
        local_count = Int32(0)
        global_round = Int32(0)
        hist_buf = Int32(0)
        next_hist_buf = Int32(0)
        current_hist_word = Int32(0)
        next_hist_word = Int32(0)
        shift = Uint32(0)
        prefix = Uint32(0)
        remaining_k = Int32(0)
        stride_scan = Int32(0)
        scan_val = Int32(0)
        count_ge = Int32(0)
        count_gt = Int32(0)
        bucket_u32 = Uint32(0)
        ordered_pivot = Uint32(0)
        local_gt_count = Int32(0)
        base = Int32(0)
        local_pos = Int32(0)
        pos = Int32(0)
        while iter_idx < total_iters:
            row_idx = group_id + iter_idx * num_groups
            seq_len = Int32(0)
            row_base = Int32(0)
            row_out_base = Int32(0)
            chunk_start = Int32(0)
            chunk_end = Int32(0)
            actual_chunk_size = Int32(0)
            val = Float32(0.0)
            ordered = Uint32(0)
            mask = Uint32(0)
            bucket = Int32(0)
            local_count = Int32(0)
            global_round = Int32(0)
            hist_buf = Int32(0)
            next_hist_buf = Int32(0)
            current_hist_word = Int32(0)
            next_hist_word = Int32(0)
            shift = Uint32(0)
            prefix = Uint32(0)
            remaining_k = Int32(0)
            stride_scan = Int32(0)
            scan_val = Int32(0)
            count_ge = Int32(0)
            count_gt = Int32(0)
            bucket_u32 = Uint32(0)
            ordered_pivot = Uint32(0)
            local_gt_count = Int32(0)
            base = Int32(0)
            local_pos = Int32(0)
            pos = Int32(0)
            if row_idx < num_rows:
                seq_len = Int32(lengths[row_idx])
                row_base = row_idx * stride
                row_out_base = row_idx * Int32(_TOPK)

                chunk_start = Int32(0)
                chunk_end = Int32(0)
                actual_chunk_size = Int32(0)
                val = Float32(0.0)
                ordered = Uint32(0)
                mask = Uint32(0)
                bucket = Int32(0)
                local_count = Int32(0)
                global_round = Int32(0)
                hist_buf = Int32(0)
                next_hist_buf = Int32(0)
                current_hist_word = Int32(0)
                next_hist_word = Int32(0)
                shift = Uint32(0)
                prefix = Uint32(0)
                remaining_k = Int32(0)
                stride_scan = Int32(0)
                scan_val = Int32(0)
                count_ge = Int32(0)
                count_gt = Int32(0)
                bucket_u32 = Uint32(0)
                ordered_pivot = Uint32(0)
                local_gt_count = Int32(0)
                base = Int32(0)
                local_pos = Int32(0)
                pos = Int32(0)

                if seq_len <= Int32(_TOPK):
                    if cta_in_group == Int32(0):
                        i = tx
                        while i < Int32(_TOPK):
                            if i < seq_len:
                                _store_output_index(
                                    output,
                                    page_table,
                                    row_idx,
                                    i,
                                    i,
                                    page_table_stride,
                                    self.paged_output,
                                )
                            else:
                                output[row_out_base + i] = Int32(-1)
                            i = i + Int32(_THREADS_PER_CTA)

                if seq_len > Int32(_TOPK):
                    chunk_start = cta_in_group * chunk_size
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > seq_len:
                        chunk_end = seq_len
                    actual_chunk_size = Int32(0)
                    if chunk_start < seq_len:
                        actual_chunk_size = chunk_end - chunk_start

                    i = tx
                    while i < actual_chunk_size:
                        val = Float32(logits[row_base + chunk_start + i])
                        s_ordered[i] = _convert_to_uint32(val)
                        i = i + Int32(_THREADS_PER_CTA)
                    cute.arch.sync_threads()

                    if tx == Int32(0):
                        s_scalars[0] = Uint32(0)
                        s_scalars[1] = Uint32(_TOPK)
                    cute.arch.sync_threads()

                    barrier_phase = _group_barrier(
                        state, group_id, barrier_phase, ctas_per_group, tx
                    )

                    if cta_in_group == Int32(0) and tx == Int32(0):
                        st_global_release_i32(
                            _global_state_ptr(state, group_id, Int32(_STATE_OUTPUT_COUNTER)),
                            Int32(0),
                        )
                    cute.arch.sync_threads()

                    for round_idx in cutlass.range_constexpr(4):
                        global_round = iter_idx * Int32(4) + Int32(round_idx)
                        hist_buf = global_round % Int32(3)
                        next_hist_buf = (global_round + Int32(1)) % Int32(3)
                        current_hist_word = hist_buf * Int32(_RADIX)
                        next_hist_word = next_hist_buf * Int32(_RADIX)
                        shift = Uint32(24 - round_idx * 8)
                        prefix = Uint32(s_scalars[0])
                        remaining_k = Int32(s_scalars[1])

                        i = tx
                        while i < Int32(_RADIX):
                            s_hist[i] = Int32(0)
                            i = i + Int32(_THREADS_PER_CTA)
                        cute.arch.sync_threads()

                        i = tx
                        ordered = Uint32(0)
                        mask = Uint32(0)
                        bucket = Int32(0)
                        while i < actual_chunk_size:
                            ordered = Uint32(s_ordered[i])
                            mask = Uint32(0)
                            if cutlass.const_expr(round_idx != 0):
                                mask = Uint32(0xFFFFFFFF) << Uint32(32 - round_idx * 8)
                            if (ordered & mask) == prefix:
                                bucket = Int32((ordered >> shift) & Uint32(0xFF))
                                _smem_red_add(hist_ptr, bucket, Int32(1))
                            i = i + Int32(_THREADS_PER_CTA)
                        cute.arch.sync_threads()

                        i = tx
                        local_count = Int32(0)
                        while i < Int32(_RADIX):
                            local_count = Int32(s_hist[i])
                            if local_count > Int32(0):
                                atomic_add_global_i32(
                                    _global_state_ptr(state, group_id, current_hist_word + i),
                                    local_count,
                                )
                            i = i + Int32(_THREADS_PER_CTA)

                        if cta_in_group == Int32(0):
                            i = tx
                            while i < Int32(_RADIX):
                                state[_state_offset(group_id, next_hist_word + i)] = Int32(0)
                                i = i + Int32(_THREADS_PER_CTA)

                        barrier_phase = _group_barrier(
                            state, group_id, barrier_phase, ctas_per_group, tx
                        )

                        i = tx
                        while i < Int32(_RADIX):
                            s_suffix[i] = Int32(state[_state_offset(group_id, current_hist_word + i)])
                            i = i + Int32(_THREADS_PER_CTA)
                        cute.arch.sync_threads()

                        for stage in cutlass.range_constexpr(8):
                            stride_scan = Int32(1 << stage)
                            scan_val = Int32(0)
                            if tx < Int32(_RADIX):
                                scan_val = Int32(s_suffix[tx])
                                if tx < Int32(_RADIX) - stride_scan:
                                    scan_val = scan_val + Int32(s_suffix[tx + stride_scan])
                            cute.arch.sync_threads()
                            if tx < Int32(_RADIX):
                                s_suffix[tx] = scan_val
                            cute.arch.sync_threads()

                        if tx == Int32(0):
                            s_scalars[2] = Uint32(0)
                            s_scalars[3] = Uint32(remaining_k)
                        cute.arch.sync_threads()

                        if tx < Int32(_RADIX):
                            count_ge = Int32(s_suffix[tx])
                            count_gt = Int32(0)
                            if tx + Int32(1) < Int32(_RADIX):
                                count_gt = Int32(s_suffix[tx + Int32(1)])
                            if count_ge >= remaining_k and count_gt < remaining_k:
                                s_scalars[2] = Uint32(tx)
                                s_scalars[3] = Uint32(remaining_k - count_gt)
                        cute.arch.sync_threads()

                        if tx == Int32(0):
                            bucket_u32 = Uint32(s_scalars[2])
                            s_scalars[0] = prefix | (bucket_u32 << shift)
                            s_scalars[1] = s_scalars[3]
                        cute.arch.sync_threads()

                    ordered_pivot = Uint32(s_scalars[0])

                    if tx == Int32(0):
                        s_counters[0] = Int32(0)
                    cute.arch.sync_threads()

                    i = tx
                    while i < actual_chunk_size:
                        if Uint32(s_ordered[i]) > ordered_pivot:
                            _smem_red_add(counter_ptr, Int32(0), Int32(1))
                        i = i + Int32(_THREADS_PER_CTA)
                    cute.arch.sync_threads()

                    local_gt_count = Int32(s_counters[0])
                    if tx == Int32(0):
                        s_counters[0] = Int32(0)
                        base = Int32(0)
                        if local_gt_count > Int32(0):
                            base = atomic_add_global_i32(
                                _global_state_ptr(state, group_id, Int32(_STATE_OUTPUT_COUNTER)),
                                local_gt_count,
                            )
                        s_counters[1] = base
                    cute.arch.sync_threads()

                    i = tx
                    local_pos = Int32(0)
                    pos = Int32(0)
                    while i < actual_chunk_size:
                        if Uint32(s_ordered[i]) > ordered_pivot:
                            local_pos = _smem_xadd(counter_ptr, Int32(0), Int32(1))
                            pos = Int32(s_counters[1]) + local_pos
                            _store_output_index(
                                output,
                                page_table,
                                row_idx,
                                pos,
                                chunk_start + i,
                                page_table_stride,
                                self.paged_output,
                            )
                        i = i + Int32(_THREADS_PER_CTA)

                    barrier_phase = _group_barrier(
                        state, group_id, barrier_phase, ctas_per_group, tx
                    )

                    i = tx
                    pos = Int32(0)
                    while i < actual_chunk_size:
                        if Uint32(s_ordered[i]) == ordered_pivot:
                            pos = atomic_add_global_i32(
                                _global_state_ptr(state, group_id, Int32(_STATE_OUTPUT_COUNTER)),
                                Int32(1),
                            )
                            if pos < Int32(_TOPK):
                                _store_output_index(
                                    output,
                                    page_table,
                                    row_idx,
                                    pos,
                                    chunk_start + i,
                                    page_table_stride,
                                    self.paged_output,
                                )
                        i = i + Int32(_THREADS_PER_CTA)

            iter_idx = iter_idx + Int32(1)


@lru_cache(maxsize=16)
def _build_persistent_topk_kernel(*, paged_output: bool):
    return SparseNSAPersistentTopK2048Kernel(paged_output=paged_output)


def clear_persistent_topk2048_kernel_cache() -> None:
    _build_persistent_topk_kernel.cache_clear()


def supports_persistent_topk2048(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    topk: int = _TOPK,
    page_table_1: torch.Tensor | None = None,
) -> bool:
    if topk != _TOPK:
        return False
    if not logits.is_cuda or logits.dtype != torch.float32 or logits.ndim != 2:
        return False
    if not logits.is_contiguous():
        return False
    if lengths.device != logits.device or lengths.dtype != torch.int32:
        return False
    if lengths.numel() != logits.shape[0] or not lengths.is_contiguous():
        return False
    if logits.shape[1] <= _RADIX_THRESHOLD:
        return False
    if page_table_1 is not None:
        if page_table_1.device != logits.device or page_table_1.dtype != torch.int32:
            return False
        if page_table_1.ndim != 2 or page_table_1.shape[0] < logits.shape[0]:
            return False
        if page_table_1.shape[1] < logits.shape[1] or not page_table_1.is_contiguous():
            return False
    return True


def _fallback_topk2048(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    page_table_1: torch.Tensor | None,
    output_indices: torch.Tensor | None,
) -> torch.Tensor:
    rows, cols = logits.shape
    if output_indices is None:
        output = torch.full((rows, _TOPK), -1, dtype=torch.int32, device=logits.device)
    else:
        if output_indices.shape != (rows, _TOPK) or output_indices.dtype != torch.int32:
            raise ValueError(
                f"output_indices must have shape {(rows, _TOPK)} and dtype int32, "
                f"got {tuple(output_indices.shape)} {output_indices.dtype}"
            )
        if output_indices.device != logits.device:
            raise ValueError("output_indices must be on the logits device")
        if not output_indices.is_contiguous():
            raise ValueError("output_indices must be contiguous")
        output = output_indices
        output.fill_(-1)
    gather_k = min(_TOPK, cols)
    if gather_k == 0:
        return output
    mask = torch.arange(cols, device=logits.device).unsqueeze(0) >= lengths.reshape(-1, 1)
    masked_logits = torch.where(mask, torch.full_like(logits, float("-inf")), logits)
    topk_values, topk_pos = torch.topk(masked_logits, k=gather_k, dim=1, largest=True, sorted=False)
    if page_table_1 is None:
        gathered = topk_pos.to(torch.int32)
    else:
        gathered = torch.gather(page_table_1, 1, topk_pos.to(torch.long))
    output[:, :gather_k] = torch.where(
        torch.isfinite(topk_values),
        gathered,
        torch.full_like(gathered, -1),
    )
    return output


def run_persistent_topk2048(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    page_table_1: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    if lengths.ndim != 1:
        lengths = lengths.reshape(-1)
    if max_seq_len is None:
        max_seq_len = int(logits.shape[1])
    if not supports_persistent_topk2048(logits, lengths, page_table_1=page_table_1):
        return _fallback_topk2048(
            logits,
            lengths,
            page_table_1=page_table_1,
            output_indices=output_indices,
        )
    if int(max_seq_len) <= _RADIX_THRESHOLD:
        return _fallback_topk2048(
            logits,
            lengths,
            page_table_1=page_table_1,
            output_indices=output_indices,
        )

    rows, stride = logits.shape
    launch = _resolve_launch_config(rows, stride, logits.device)
    if launch.chunk_size > _MAX_CHUNK_ELEMENTS:
        return _fallback_topk2048(
            logits,
            lengths,
            page_table_1=page_table_1,
            output_indices=output_indices,
        )

    if output_indices is None:
        output_indices = torch.empty((rows, _TOPK), dtype=torch.int32, device=logits.device)
    elif output_indices.shape != (rows, _TOPK) or output_indices.dtype != torch.int32:
        raise ValueError(
            f"output_indices must have shape {(rows, _TOPK)} and dtype int32, "
            f"got {tuple(output_indices.shape)} {output_indices.dtype}"
        )
    elif output_indices.device != logits.device:
        raise ValueError("output_indices must be on the logits device")
    elif not output_indices.is_contiguous():
        raise ValueError("output_indices must be contiguous")

    state_words = launch.num_groups * _STATE_WORDS
    if workspace is None:
        state = torch.empty((state_words,), dtype=torch.int32, device=logits.device)
    else:
        if workspace.dtype != torch.int32 or workspace.device != logits.device:
            raise ValueError("workspace must be an int32 tensor on the logits device")
        if workspace.numel() < state_words:
            raise ValueError(
                f"workspace too small: need {state_words * 4} bytes, "
                f"got {workspace.numel() * workspace.element_size()} bytes"
            )
        state = workspace.reshape(-1)[:state_words]
        if not state.is_contiguous():
            raise ValueError("workspace must be contiguous")

    paged_output = page_table_1 is not None
    if page_table_1 is None:
        page_table = output_indices
        page_table_stride = _TOPK
    else:
        page_table = page_table_1
        page_table_stride = int(page_table_1.shape[1])

    kernel = _build_persistent_topk_kernel(paged_output=paged_output)
    args = (
        _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(lengths, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(page_table, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(output_indices, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(state, cutlass.Int32, assumed_align=4),
        Int32(rows),
        Int32(stride),
        Int32(launch.chunk_size),
        Int32(launch.ctas_per_group),
        Int32(launch.num_groups),
        Int32(page_table_stride),
        current_cuda_stream(),
    )
    cache_key = (
        _tensor_meta_key(logits),
        _tensor_meta_key(lengths),
        _tensor_meta_key(page_table),
        _tensor_meta_key(output_indices),
        _tensor_meta_key(state),
        (
            "persistent_topk2048_v1",
            launch.chunk_size,
            launch.ctas_per_group,
            launch.num_groups,
            paged_output,
        ),
    )
    _run_cached_host_launcher(kernel, cache_key, args)
    return output_indices
