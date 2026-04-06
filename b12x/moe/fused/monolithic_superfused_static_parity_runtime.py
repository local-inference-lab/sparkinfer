from __future__ import annotations

import ctypes
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from torch.utils import dlpack as torch_dlpack
from cutlass.cutlass_dsl import Int32

from b12x.cute.fp4 import align_up
from b12x.cute.utils import current_cuda_stream, make_ptr
from b12x.distributed._oneshot_common import align_bytes, cutlass_dtype
from b12x.integration.tp_moe import _WeightViews, TPCompactStaticWorkspace
from b12x.moe.fused.monolithic_superfused_static_parity import MoESuperfusedStaticKernel
from b12x.moe.fused.pre_mlp_static import UnifiedPreMLPIPC


_KDLGPU = 2
_KDLBFLOAT = 4
_SUPERFUSED_STATIC_MAX_ACTIVE_CLUSTERS = 188
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


def _get_workspace_scratch(
    workspace: TPCompactStaticWorkspace,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = getattr(workspace, name, None)
    if (
        tensor is None
        or tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
    ):
        tensor = torch.empty(shape, dtype=dtype, device=device)
        setattr(workspace, name, tensor)
    return tensor


@lru_cache(maxsize=32)
def _get_superfused_static_parity_launch(
    *,
    world_size: int,
    num_tokens: int,
    hidden_size: int,
    output_n: int,
    num_sparse_experts: int,
    top_k: int,
    state_E: int,
    weight_E: int,
    max_rows: int,
    intermediate_size: int,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    fc2_tile_amax: bool,
    emit_normalized: bool,
    emit_routing_output: bool,
    renormalize_topk: bool,
    prequantized_input: bool,
):
    kernel = MoESuperfusedStaticKernel(
        world_size=world_size,
        num_sparse_experts=num_sparse_experts,
        top_k=top_k,
        sf_vec_size=16,
        mma_tiler_mn=(128, 128),
        output_tile_count_n=max(1, (output_n + 128 - 1) // 128),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        fc2_tile_amax=fc2_tile_amax,
        emit_normalized=emit_normalized,
        emit_routing_output=emit_routing_output,
        renormalize_topk=renormalize_topk,
        prequantized_input=prequantized_input,
    )
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
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_tokens * (top_k + 1),), assumed_align=4
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (num_tokens * (top_k + 1),), assumed_align=4
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (max_rows, hidden_size // 2, state_E), stride_order=(1, 0, 2), assumed_align=16
    )
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * max_rows * (hidden_size // 2),), assumed_align=16
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * align_up(max_rows, 128) * align_up(hidden_size // 16, 4),), assumed_align=16
    )
    one_i32_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4
    )
    global_to_local_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (weight_E,), assumed_align=4
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (weight_E,), assumed_align=4
    )
    tile_alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (state_E * (align_up(max_rows, 128) // 128),), assumed_align=4
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E, max_rows), stride_order=(1, 0), assumed_align=4
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (state_E, max_rows), stride_order=(1, 0), assumed_align=4
    )
    w13_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (2 * intermediate_size, hidden_size // 2, weight_E), stride_order=(1, 0, 2), assumed_align=16
    )
    down_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (hidden_size, intermediate_size // 2, weight_E), stride_order=(1, 0, 2), assumed_align=16
    )
    f8_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(
        kernel,
        row_fake, row_fake, row_fake, row_fake, row_fake, row_fake, row_fake, row_fake,
        signal_ptr, signal_ptr, signal_ptr, signal_ptr, signal_ptr, signal_ptr, signal_ptr, signal_ptr,
        signal_ptr,
        0,
        row_fake,
        row_fake,
        row_fake,
        norm_weight_fake,
        sparse_gate_weight_fake,
        shared_gate_weight_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        f8_ptr,
        packed_a_storage_fake,
        scale_storage_fake,
        one_i32_fake,
        one_i32_fake,
        w13_fake,
        f8_ptr,
        down_fake,
        f8_ptr,
        row_counts_fake,
        one_i32_fake,
        weight_expert_ids_fake,
        global_to_local_fake,
        alpha_fake,
        alpha_fake,
        alpha_fake,
        alpha_fake,
        tile_alpha_fake,
        tile_alpha_fake,
        row_fake,
        token_map_fake,
        token_weights_fake,
        1,
        1.0,
        current_cuda_stream(),
    )


def launch_superfused_static_parity(
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    ipc: UnifiedPreMLPIPC,
    workspace: TPCompactStaticWorkspace,
    weights: _WeightViews,
    input_global_scale: torch.Tensor,
    expert_alpha: torch.Tensor,
    down_alpha: torch.Tensor,
    global_scale: torch.Tensor,
    num_sparse_experts: int,
    top_k: int,
    output: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    normalized_out: torch.Tensor | None = None,
    topk_ids_out: torch.Tensor | None = None,
    topk_weights_out: torch.Tensor | None = None,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    fc2_tile_amax: bool,
    renormalize_topk: bool,
    eps: float,
    emit_routing_output: bool = False,
    prequantized_input: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ipc.validate()
    if output is None:
        output = torch.empty_like(hidden_states)
    if residual_out is None:
        residual_out = torch.empty_like(hidden_states)
    emit_normalized = normalized_out is not None
    normalized_arg = normalized_out if normalized_out is not None else hidden_states
    if emit_routing_output:
        if topk_ids_out is None:
            topk_ids_out = _get_workspace_scratch(
                workspace,
                name="_superfused_topk_ids_scratch",
                shape=(hidden_states.shape[0] * (top_k + 1),),
                dtype=torch.int32,
                device=hidden_states.device,
            )
        if topk_weights_out is None:
            topk_weights_out = _get_workspace_scratch(
                workspace,
                name="_superfused_topk_weights_scratch",
                shape=(hidden_states.shape[0] * (top_k + 1),),
                dtype=torch.float32,
                device=hidden_states.device,
            )
    else:
        topk_ids_out = _get_workspace_scratch(
            workspace,
            name="_superfused_topk_ids_dummy",
            shape=(1,),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        topk_weights_out = _get_workspace_scratch(
            workspace,
            name="_superfused_topk_weights_dummy",
            shape=(1,),
            dtype=torch.float32,
            device=hidden_states.device,
        )

    compiled = _get_superfused_static_parity_launch(
        world_size=ipc.world_size,
        num_tokens=int(hidden_states.shape[0]),
        hidden_size=int(hidden_states.shape[1]),
        output_n=int(weights.down.shape[1] * 2),
        num_sparse_experts=num_sparse_experts,
        top_k=top_k,
        state_E=int(workspace.state_E),
        weight_E=int(workspace.weight_E),
        max_rows=int(workspace.max_rows),
        intermediate_size=int(weights.w13.shape[0] // 2),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        fc2_tile_amax=fc2_tile_amax,
        emit_normalized=emit_normalized,
        emit_routing_output=emit_routing_output,
        renormalize_topk=renormalize_topk,
        prequantized_input=prequantized_input,
    )

    peer_input_tensors = _get_peer_input_tensors(
        peer_input_ptrs=tuple(int(ptr) for ptr in ipc.peer_input_ptrs),
        shape=tuple(hidden_states.shape),
        strides=tuple(hidden_states.stride()),
        device=hidden_states.device.index or 0,
        dtype=hidden_states.dtype,
    )
    signal_ptr_args = _get_signal_ptr_args(tuple(int(ptr) for ptr in ipc.signal_ptrs))

    packed_a_u8 = workspace.packed_input.permute(1, 2, 0)
    compiled(
        *peer_input_tensors[:8],
        *signal_ptr_args[:8],
        make_ptr(cutlass.Int32, ipc.signal_ptrs[ipc.rank], cute.AddressSpace.gmem, assumed_align=128),
        ipc.rank,
        residual,
        normalized_arg,
        residual_out,
        norm_weight,
        gate_weight,
        shared_gate_weight,
        topk_ids_out,
        topk_weights_out,
        packed_a_u8,
        workspace.sfa_ptr,
        workspace.packed_a_flat,
        workspace.scale_flat,
        workspace.barrier_count,
        workspace.barrier_epoch,
        weights.w13,
        weights.sfb_w13_ptr,
        weights.down,
        weights.sfb_down_ptr,
        workspace.row_counts,
        workspace.active_expert_count,
        workspace.weight_expert_ids,
        workspace.global_to_local_expert,
        input_global_scale,
        expert_alpha,
        down_alpha,
        global_scale,
        workspace.fc1_tile_scale.view(-1),
        workspace.fc1_tile_alpha.view(-1),
        output,
        workspace.token_map,
        workspace.token_weights,
        _SUPERFUSED_STATIC_MAX_ACTIVE_CLUSTERS,
        float(eps),
        current_cuda_stream(),
    )
    return output, residual_out, normalized_out, topk_ids_out, topk_weights_out
