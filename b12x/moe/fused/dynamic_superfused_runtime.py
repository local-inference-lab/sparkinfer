from __future__ import annotations

import ctypes
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from torch.utils import dlpack as torch_dlpack

from b12x.cute.fp4 import align_up
from b12x.cute.utils import current_cuda_stream, make_ptr
from b12x.integration.tp_moe import _WeightViews, TPDynamicSuperfusedWorkspace
from b12x.moe.fused.dynamic_superfused import MoEDynamicSuperfusedKernel
from b12x.moe.fused.pre_mlp_static import UnifiedPreMLPIPC


_KDLGPU = 2
_KDLBFLOAT = 4
_DYNAMIC_SUPERFUSED_MAX_ACTIVE_CLUSTERS = 188
_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_RAW_TENSOR_REGISTRY: dict[int, "_RawCudaTensorView"] = {}
_DLMTensorPtr = ctypes.POINTER(_DLManagedTensor)
_DLDeleter = ctypes.CFUNCTYPE(None, _DLMTensorPtr)


@_DLDeleter
def _raw_tensor_deleter(ptr):
    _RAW_TENSOR_REGISTRY.pop(ctypes.addressof(ptr.contents), None)


_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLDeleter),
]


class _RawCudaTensorView:
    def __init__(
        self,
        *,
        ptr: int,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        device: int,
        dtype: torch.dtype,
    ):
        if dtype is not torch.bfloat16:
            raise NotImplementedError(f"unsupported raw tensor dtype: {dtype}")
        self._shape = (ctypes.c_int64 * len(shape))(*shape)
        self._strides = (ctypes.c_int64 * len(strides))(*strides)
        self._managed = _DLManagedTensor()
        self._managed.dl_tensor.data = int(ptr)
        self._managed.dl_tensor.device = _DLDevice(_KDLGPU, int(device))
        self._managed.dl_tensor.ndim = len(shape)
        self._managed.dl_tensor.dtype = _DLDataType(_KDLBFLOAT, 16, 1)
        self._managed.dl_tensor.shape = self._shape
        self._managed.dl_tensor.strides = self._strides
        self._managed.dl_tensor.byte_offset = 0
        self._managed.manager_ctx = 0
        self._managed.deleter = _raw_tensor_deleter

    def __dlpack__(self, stream=None):
        del stream
        addr = ctypes.addressof(self._managed)
        _RAW_TENSOR_REGISTRY[addr] = self
        return _PyCapsule_New(addr, b"dltensor", None)

    def __dlpack_device__(self):
        return (_KDLGPU, int(self._managed.dl_tensor.device.device_id))


def _tensor_from_cuda_pointer(
    *,
    ptr: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    device: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch_dlpack.from_dlpack(
        _RawCudaTensorView(ptr=ptr, shape=shape, strides=strides, device=device, dtype=dtype)
    )


def _get_peer_input_tensors(
    *,
    peer_input_ptrs: tuple[int, ...],
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    device: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    tensors = tuple(
        _tensor_from_cuda_pointer(
            ptr=ptr,
            shape=shape,
            strides=strides,
            device=device,
            dtype=dtype,
        )
        for ptr in peer_input_ptrs
    )
    if not tensors:
        raise ValueError("peer_input_ptrs must not be empty")
    padded = list(tensors)
    while len(padded) < 8:
        padded.append(padded[0])
    return tuple(padded[:8])


@lru_cache(maxsize=64)
def _get_signal_ptr_args(signal_ptrs: tuple[int, ...]) -> tuple[cute.Pointer, ...]:
    args = [
        make_ptr(cutlass.Int32, ptr, cute.AddressSpace.gmem, assumed_align=128)
        for ptr in signal_ptrs
    ]
    if not args:
        raise ValueError("signal_ptrs must not be empty")
    while len(args) < 8:
        args.append(args[0])
    return tuple(args[:8])


class _DynamicSuperfusedLaunch:
    """Pointer-based launch wrapper for the dynamic superfused kernel."""

    def __init__(self, kernel, k: int):
        self._kernel = kernel
        self._k = k
        self._half_k = k // 2
        self._cols_pad_k = align_up(k // 16, 4)

    @cute.jit
    def __call__(
        self,
        inp0: cute.Tensor,
        inp1: cute.Tensor,
        inp2: cute.Tensor,
        inp3: cute.Tensor,
        inp4: cute.Tensor,
        inp5: cute.Tensor,
        inp6: cute.Tensor,
        inp7: cute.Tensor,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        self_signal: cute.Pointer,
        rank: cutlass.Int32,
        residual_in: cute.Tensor,
        residual_out: cute.Tensor,
        norm_weight: cute.Tensor,
        sparse_gate_weight: cute.Tensor,
        shared_gate_weight: cute.Tensor,
        packed_a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        packed_a_storage_ptr: cute.Pointer,
        scale_storage_ptr: cute.Pointer,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        token_head: cute.Tensor,
        next_tile_alloc: cute.Tensor,
        producers_done_count: cute.Tensor,
        all_work_published: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_ready_ptr: cute.Pointer,
        task_expert_ptr: cute.Pointer,
        task_m_tile_ptr: cute.Pointer,
        task_slice_begin_ptr: cute.Pointer,
        task_slice_count_ptr: cute.Pointer,
        task_valid_rows_ptr: cute.Pointer,
        expert_tile_ptr_ptr: cute.Pointer,
        tile_write_count_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        max_tiles_per_expert: cutlass.Int32,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_ptr: cute.Pointer,
        token_map_ptr: cute.Pointer,
        token_weights_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        rows_padded: cutlass.Int32,
        task_capacity: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        packed_a_u8 = cute.make_tensor(
            packed_a_ptr,
            layout=cute.make_layout(
                (rows_padded, self._half_k, 1),
                stride=(self._half_k, 1, rows_padded * self._half_k),
            ),
        )
        packed_a_storage = cute.make_tensor(
            packed_a_storage_ptr,
            layout=cute.make_layout((rows_padded * self._half_k,), stride=(1,)),
        )
        scale_storage = cute.make_tensor(
            scale_storage_ptr,
            layout=cute.make_layout((rows_padded * self._cols_pad_k,), stride=(1,)),
        )
        task_ready = cute.make_tensor(task_ready_ptr, layout=cute.make_layout((task_capacity,), stride=(1,)))
        task_expert = cute.make_tensor(task_expert_ptr, layout=cute.make_layout((task_capacity,), stride=(1,)))
        task_m_tile = cute.make_tensor(task_m_tile_ptr, layout=cute.make_layout((task_capacity,), stride=(1,)))
        task_slice_begin = cute.make_tensor(task_slice_begin_ptr, layout=cute.make_layout((task_capacity,), stride=(1,)))
        task_slice_count = cute.make_tensor(task_slice_count_ptr, layout=cute.make_layout((task_capacity,), stride=(1,)))
        task_valid_rows = cute.make_tensor(task_valid_rows_ptr, layout=cute.make_layout((task_capacity,), stride=(1,)))
        expert_tile_ptr_flat = cute.make_tensor(
            expert_tile_ptr_ptr,
            layout=cute.make_layout((row_counts.shape[0] * max_tiles_per_expert,), stride=(1,)),
        )
        tile_write_count = cute.make_tensor(
            tile_write_count_ptr,
            layout=cute.make_layout((rows_padded // 128,), stride=(1,)),
        )
        scatter_output = cute.make_tensor(
            scatter_ptr,
            layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1)),
        )
        token_map = cute.make_tensor(token_map_ptr, layout=cute.make_layout((rows_padded,), stride=(1,)))
        token_weights = cute.make_tensor(token_weights_ptr, layout=cute.make_layout((rows_padded,), stride=(1,)))
        self._kernel(
            inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7,
            signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7,
            self_signal, rank,
            residual_in, residual_out,
            norm_weight, sparse_gate_weight, shared_gate_weight,
            packed_a_u8, sfa_ptr, packed_a_storage, scale_storage,
            barrier_count, barrier_epoch,
            token_head, next_tile_alloc, producers_done_count, all_work_published,
            task_head, task_tail, task_ready,
            task_expert, task_m_tile, task_slice_begin, task_slice_count, task_valid_rows,
            expert_tile_ptr_flat, tile_write_count, row_counts, max_tiles_per_expert,
            b_w13, sfb_w13_ptr, b_down, sfb_down_ptr,
            input_global_scale, alpha, down_alpha, global_scale,
            scatter_output, token_map, token_weights,
            max_active_clusters=max_active_clusters,
            eps=eps,
            stream=stream,
        )


@lru_cache(maxsize=32)
def _get_dynamic_superfused_launch(
    *,
    world_size: int,
    num_tokens: int,
    hidden_size: int,
    output_n: int,
    num_sparse_experts: int,
    top_k: int,
    weight_E: int,
    rows_padded: int,
    task_capacity: int,
    max_tiles_per_expert: int,
    intermediate_size: int,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    renormalize_topk: bool,
):
    kernel = MoEDynamicSuperfusedKernel(
        world_size=world_size,
        num_sparse_experts=num_sparse_experts,
        top_k=top_k,
        sf_vec_size=16,
        mma_tiler_mn=(128, 128),
        output_tile_count_n=max(1, (output_n + 128 - 1) // 128),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        renormalize_topk=renormalize_topk,
    )
    launch = _DynamicSuperfusedLaunch(kernel, k=hidden_size)
    signal_ptr = make_ptr(cutlass.Int32, 128, cute.AddressSpace.gmem, assumed_align=128)
    row_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (num_tokens, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    norm_weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (hidden_size,), assumed_align=16
    )
    sparse_gate_weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (num_sparse_experts, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    shared_gate_weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (1, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    packed_a_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    scale_storage_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    i32_one_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (weight_E,), assumed_align=4
    )
    task_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    expert_tile_ptr_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (weight_E,), assumed_align=4
    )
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4)
    w13_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (2 * intermediate_size, hidden_size // 2, weight_E), stride_order=(1, 0, 2), assumed_align=16
    )
    down_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (hidden_size, intermediate_size // 2, weight_E), stride_order=(1, 0, 2), assumed_align=16
    )
    f8_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(
        launch,
        row_fake, row_fake, row_fake, row_fake, row_fake, row_fake, row_fake, row_fake,
        signal_ptr, signal_ptr, signal_ptr, signal_ptr, signal_ptr, signal_ptr, signal_ptr, signal_ptr,
        signal_ptr,
        0,
        row_fake,
        row_fake,
        norm_weight_fake,
        sparse_gate_weight_fake,
        shared_gate_weight_fake,
        packed_a_fake,
        f8_ptr,
        packed_a_storage_fake,
        scale_storage_fake,
        i32_one_fake,
        i32_one_fake,
        i32_one_fake,
        i32_one_fake,
        i32_one_fake,
        i32_one_fake,
        i32_one_fake,
        i32_one_fake,
        task_fake,
        task_fake,
        task_fake,
        task_fake,
        task_fake,
        task_fake,
        expert_tile_ptr_fake,
        task_fake,
        row_counts_fake,
        max_tiles_per_expert,
        w13_fake,
        f8_ptr,
        down_fake,
        f8_ptr,
        alpha_fake,
        alpha_fake,
        alpha_fake,
        alpha_fake,
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        token_map_fake,
        token_weights_fake,
        num_tokens,
        rows_padded,
        task_capacity,
        _DYNAMIC_SUPERFUSED_MAX_ACTIVE_CLUSTERS,
        1.0,
        current_cuda_stream(),
    )


def launch_dynamic_superfused(
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    ipc: UnifiedPreMLPIPC,
    workspace: TPDynamicSuperfusedWorkspace,
    weights: _WeightViews,
    input_global_scale: torch.Tensor,
    expert_alpha: torch.Tensor,
    down_alpha: torch.Tensor,
    global_scale: torch.Tensor,
    num_sparse_experts: int,
    top_k: int,
    output: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    renormalize_topk: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ipc.validate()
    if output is None:
        output = torch.empty_like(hidden_states)
    if residual_out is None:
        residual_out = torch.empty_like(hidden_states)

    compiled = _get_dynamic_superfused_launch(
        world_size=ipc.world_size,
        num_tokens=int(hidden_states.shape[0]),
        hidden_size=int(hidden_states.shape[1]),
        output_n=int(weights.down.shape[1] * 2),
        num_sparse_experts=num_sparse_experts,
        top_k=top_k,
        weight_E=int(workspace.weight_E),
        rows_padded=int(workspace.physical_tiles_capacity * 128),
        task_capacity=int(workspace.task_capacity),
        max_tiles_per_expert=int(workspace.max_tiles_per_expert),
        intermediate_size=int(weights.w13.shape[0] // 2),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        renormalize_topk=renormalize_topk,
    )

    peer_input_tensors = _get_peer_input_tensors(
        peer_input_ptrs=tuple(int(ptr) for ptr in ipc.peer_input_ptrs),
        shape=tuple(hidden_states.shape),
        strides=tuple(hidden_states.stride()),
        device=hidden_states.device.index or 0,
        dtype=hidden_states.dtype,
    )
    signal_ptr_args = _get_signal_ptr_args(tuple(int(ptr) for ptr in ipc.signal_ptrs))
    _gptr = lambda dtype, t, align=16: make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

    compiled(
        *peer_input_tensors[:8],
        *signal_ptr_args[:8],
        make_ptr(cutlass.Int32, ipc.signal_ptrs[ipc.rank], cute.AddressSpace.gmem, assumed_align=128),
        ipc.rank,
        residual,
        residual_out,
        norm_weight,
        gate_weight,
        shared_gate_weight,
        _gptr(cutlass.Uint8, workspace.packed_input),
        workspace.sfa_ptr,
        _gptr(cutlass.Uint8, workspace.packed_a_flat),
        _gptr(cutlass.Uint8, workspace.scale_flat),
        workspace.barrier_count,
        workspace.barrier_epoch,
        workspace.token_head,
        workspace.next_tile_alloc,
        workspace.producers_done_count,
        workspace.all_work_published,
        workspace.task_head,
        workspace.task_tail,
        _gptr(cutlass.Int32, workspace.task_ready, 4),
        _gptr(cutlass.Int32, workspace.task_expert, 4),
        _gptr(cutlass.Int32, workspace.task_m_tile, 4),
        _gptr(cutlass.Int32, workspace.task_slice_begin, 4),
        _gptr(cutlass.Int32, workspace.task_slice_count, 4),
        _gptr(cutlass.Int32, workspace.task_valid_rows, 4),
        _gptr(cutlass.Int32, workspace.expert_tile_ptr, 4),
        _gptr(cutlass.Int32, workspace.tile_write_count, 4),
        workspace.row_counts,
        workspace.max_tiles_per_expert,
        weights.w13,
        weights.sfb_w13_ptr,
        weights.down,
        weights.sfb_down_ptr,
        input_global_scale,
        expert_alpha,
        down_alpha,
        global_scale,
        _gptr(cutlass.BFloat16, output),
        _gptr(cutlass.Int32, workspace.token_map, 4),
        _gptr(cutlass.Float32, workspace.token_weights, 4),
        hidden_states.shape[0],
        workspace.physical_tiles_capacity * 128,
        workspace.task_capacity,
        _DYNAMIC_SUPERFUSED_MAX_ACTIVE_CLUSTERS,
        float(eps),
        current_cuda_stream(),
    )
    return output, residual_out
