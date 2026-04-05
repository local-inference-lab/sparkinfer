"""Internal PCIe oneshot allreduce + GemmaRMSNorm component.

This is intentionally generic: callers can either let this module allocate and
exchange IPC buffers through a `torch.distributed` process group, or construct
it from already-shared signal/data pointers supplied by an external runtime.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Int32
from b12x.cute.utils import current_cuda_stream, make_ptr

from ._cuda_ipc import CudaRTLibrary
from ._oneshot_common import (
    MAX_BLOCKS as _MAX_BLOCKS,
    SIGNAL_BYTES as _SIGNAL_BYTES,
    SUPPORTED_WORLD_SIZES as _SUPPORTED_WORLD_SIZES,
    THREADS_PER_BLOCK as _THREADS_PER_BLOCK,
    add_f32 as _add_f32,
    align_bytes as _align_bytes,
    cutlass_dtype as _cutlass_dtype,
    reduce_peer_row_sum as _reduce_peer_row_sum,
    sqrt_f32 as _sqrt_f32,
    wait_for_peer_signals as _wait_for_peer_signals,
    warp_reduce,
)


class _PCIeGemmaRMSNormKernel:
    def __init__(self, *, world_size: int, num_tokens: int, hidden_size: int, element_dtype):
        self.world_size = world_size
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.element_dtype = element_dtype

    def __call__(self, *args):
        stream = args[-1]
        self.kernel(*args[:-1]).launch(
            grid=[self.num_tokens, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        inp0: cute.Tensor,
        inp1: cute.Tensor,
        inp2: cute.Tensor,
        inp3: cute.Tensor,
        inp4: cute.Tensor,
        inp5: cute.Tensor,
        inp6: cute.Tensor,
        inp7: cute.Tensor,
        residual_in: cute.Tensor,
        output: cute.Tensor,
        residual_out: cute.Tensor,
        weight: cute.Tensor,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        self_signal: cute.Pointer,
        rank: Int32,
        eps: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tidx = Int32(tidx)
        bidx = Int32(bidx)

        signal_ptrs = [signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7]
        inputs = [inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7]
        _wait_for_peer_signals(
            signal_ptrs=signal_ptrs,
            self_signal=self_signal,
            rank=rank,
            world_size=self.world_size,
            bidx=bidx,
            tidx=tidx,
        )

        local_sum_sq = cutlass.Float32(0.0)
        col = tidx
        while col < Int32(self.hidden_size):
            acc = _reduce_peer_row_sum(
                inputs=inputs,
                world_size=self.world_size,
                bidx=bidx,
                col=col,
                element_dtype=self.element_dtype,
            )

            residual_val = self.element_dtype(acc + self.element_dtype(residual_in[bidx, col]))
            residual_out[bidx, col] = residual_val
            residual_f32 = cutlass.Float32(residual_val)
            local_sum_sq += residual_f32 * residual_f32
            col += Int32(_THREADS_PER_BLOCK)

        sum_sq = warp_reduce(local_sum_sq, _add_f32)
        inv_scale = cutlass.Float32(1.0) / _sqrt_f32(sum_sq / cutlass.Float32(self.hidden_size) + eps)

        col = tidx
        while col < Int32(self.hidden_size):
            gamma = cutlass.Float32(1.0) + cutlass.Float32(weight[col])
            out = cutlass.Float32(residual_out[bidx, col]) * inv_scale * gamma
            output[bidx, col] = self.element_dtype(out)
            col += Int32(_THREADS_PER_BLOCK)


class _GemmaLaunch:
    def __init__(self, *, world_size: int, num_tokens: int, hidden_size: int, input_dtype: torch.dtype, weight_dtype: torch.dtype):
        self.world_size = world_size
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.input_cutlass_dtype = _cutlass_dtype(input_dtype)
        self.weight_cutlass_dtype = _cutlass_dtype(weight_dtype)
        self.kernel = _PCIeGemmaRMSNormKernel(
            world_size=world_size,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            element_dtype=self.input_cutlass_dtype,
        )

    @cute.jit
    def __call__(
        self,
        inp0_ptr: cute.Pointer,
        inp1_ptr: cute.Pointer,
        inp2_ptr: cute.Pointer,
        inp3_ptr: cute.Pointer,
        inp4_ptr: cute.Pointer,
        inp5_ptr: cute.Pointer,
        inp6_ptr: cute.Pointer,
        inp7_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        residual_out_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        signal0_ptr: cute.Pointer,
        signal1_ptr: cute.Pointer,
        signal2_ptr: cute.Pointer,
        signal3_ptr: cute.Pointer,
        signal4_ptr: cute.Pointer,
        signal5_ptr: cute.Pointer,
        signal6_ptr: cute.Pointer,
        signal7_ptr: cute.Pointer,
        self_signal_ptr: cute.Pointer,
        rank: Int32,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        row_layout = cute.make_layout((self.num_tokens, self.hidden_size), stride=(self.hidden_size, 1))
        weight_layout = cute.make_layout((self.hidden_size,), stride=(1,))
        inputs = [
            cute.make_tensor(inp0_ptr, layout=row_layout),
            cute.make_tensor(inp1_ptr, layout=row_layout),
            cute.make_tensor(inp2_ptr, layout=row_layout),
            cute.make_tensor(inp3_ptr, layout=row_layout),
            cute.make_tensor(inp4_ptr, layout=row_layout),
            cute.make_tensor(inp5_ptr, layout=row_layout),
            cute.make_tensor(inp6_ptr, layout=row_layout),
            cute.make_tensor(inp7_ptr, layout=row_layout),
        ]
        residual = cute.make_tensor(residual_ptr, layout=row_layout)
        output = cute.make_tensor(output_ptr, layout=row_layout)
        residual_out = cute.make_tensor(residual_out_ptr, layout=row_layout)
        weight = cute.make_tensor(weight_ptr, layout=weight_layout)
        self.kernel(
            *inputs,
            residual,
            output,
            residual_out,
            weight,
            signal0_ptr,
            signal1_ptr,
            signal2_ptr,
            signal3_ptr,
            signal4_ptr,
            signal5_ptr,
            signal6_ptr,
            signal7_ptr,
            self_signal_ptr,
            rank,
            eps,
            stream,
        )


@lru_cache(maxsize=128)
def _get_gemma_kernel(
    world_size: int,
    num_tokens: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
):
    launch = _GemmaLaunch(
        world_size=world_size,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
    )
    ptr_align = _align_bytes(input_dtype)
    weight_align = _align_bytes(weight_dtype)
    input_fake = make_ptr(_cutlass_dtype(input_dtype), max(16, ptr_align), cute.AddressSpace.gmem, assumed_align=ptr_align)
    residual_fake = make_ptr(_cutlass_dtype(input_dtype), max(16, ptr_align), cute.AddressSpace.gmem, assumed_align=ptr_align)
    output_fake = make_ptr(_cutlass_dtype(input_dtype), max(16, ptr_align), cute.AddressSpace.gmem, assumed_align=ptr_align)
    residual_out_fake = make_ptr(_cutlass_dtype(input_dtype), max(16, ptr_align), cute.AddressSpace.gmem, assumed_align=ptr_align)
    weight_fake = make_ptr(_cutlass_dtype(weight_dtype), max(16, weight_align), cute.AddressSpace.gmem, assumed_align=weight_align)
    signal_fake = make_ptr(cutlass.Int32, 128, cute.AddressSpace.gmem, assumed_align=128)
    return cute.compile(
        launch,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        residual_fake,
        output_fake,
        residual_out_fake,
        weight_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        0,
        1.0,
        current_cuda_stream(),
    )


@dataclass
class _OwnedSharedBuffer:
    local_ptr: int
    peer_ptrs: tuple[int, ...]
    remote_ptrs: tuple[int, ...]


class PCIeOneshotAllReduce:
    """Generic runtime wrapper for the PCIe oneshot fused layernorm kernel."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]] = None,
        eager_buffer_ptrs1: Optional[Sequence[int]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]] = None,
    ):
        if world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if len(signal_ptrs) != world_size:
            raise ValueError("signal_ptrs must match world size")
        if (eager_buffer_ptrs0 is None) != (eager_buffer_ptrs1 is None):
            raise ValueError("eager buffers must be provided as a pair")
        if eager_buffer_ptrs0 is not None and len(eager_buffer_ptrs0) != world_size:
            raise ValueError("eager_buffer_ptrs0 must match world size")
        if eager_buffer_ptrs1 is not None and len(eager_buffer_ptrs1) != world_size:
            raise ValueError("eager_buffer_ptrs1 must match world size")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = torch.device(device)
        self.process_group = process_group
        self._ipc = ipc or CudaRTLibrary()
        self._signal_ptrs = tuple(int(ptr) for ptr in signal_ptrs)
        self._eager_ptrs = None
        if eager_buffer_ptrs0 is not None and eager_buffer_ptrs1 is not None:
            self._eager_ptrs = (
                tuple(int(ptr) for ptr in eager_buffer_ptrs0),
                tuple(int(ptr) for ptr in eager_buffer_ptrs1),
            )
        self._slot = 0
        self._owned_buffers = list(owned_buffers or [])

    @classmethod
    def from_ipc(
        cls,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]] = None,
        eager_buffer_ptrs1: Optional[Sequence[int]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> "PCIeOneshotAllReduce":
        return cls(
            rank=rank,
            world_size=world_size,
            device=device,
            signal_ptrs=signal_ptrs,
            eager_buffer_ptrs0=eager_buffer_ptrs0,
            eager_buffer_ptrs1=eager_buffer_ptrs1,
            process_group=process_group,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: dist.ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: Optional[int] = None,
    ) -> "PCIeOneshotAllReduce":
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        if world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        ipc = CudaRTLibrary()
        device_obj = torch.device(device)
        if device_obj.type != "cuda":
            raise ValueError("PCIe oneshot requires a CUDA device")
        ipc.cudaSetDevice(device_obj.index or 0)

        owned_buffers: list[_OwnedSharedBuffer] = []
        signal_buf = cls._allocate_shared_buffer(process_group, _SIGNAL_BYTES, zero_fill=True, ipc=ipc)
        owned_buffers.append(signal_buf)
        eager0 = None
        eager1 = None
        if max_input_bytes is not None:
            eager0 = cls._allocate_shared_buffer(process_group, max_input_bytes, zero_fill=False, ipc=ipc)
            eager1 = cls._allocate_shared_buffer(process_group, max_input_bytes, zero_fill=False, ipc=ipc)
            owned_buffers.extend([eager0, eager1])

        return cls(
            rank=rank,
            world_size=world_size,
            device=device_obj,
            signal_ptrs=signal_buf.peer_ptrs,
            eager_buffer_ptrs0=None if eager0 is None else eager0.peer_ptrs,
            eager_buffer_ptrs1=None if eager1 is None else eager1.peer_ptrs,
            process_group=process_group,
            ipc=ipc,
            owned_buffers=owned_buffers,
        )

    @staticmethod
    def _allocate_shared_buffer(
        process_group: dist.ProcessGroup,
        size_in_bytes: int,
        *,
        zero_fill: bool,
        ipc: CudaRTLibrary,
    ) -> _OwnedSharedBuffer:
        local_ptr = ipc.cudaMalloc(size_in_bytes)
        if zero_fill:
            ipc.cudaMemset(local_ptr, 0, size_in_bytes)
        local_handle = ipc.cudaIpcGetMemHandleBytes(local_ptr)
        world_size = dist.get_world_size(group=process_group)
        rank = dist.get_rank(group=process_group)
        handles: list[bytes] = [b""] * world_size
        dist.all_gather_object(handles, local_handle, group=process_group)
        peer_ptrs: list[int] = []
        remote_ptrs: list[int] = []
        for idx, handle in enumerate(handles):
            if idx == rank:
                peer_ptrs.append(local_ptr)
            else:
                remote_ptr = ipc.cudaIpcOpenMemHandleBytes(handle)
                peer_ptrs.append(remote_ptr)
                remote_ptrs.append(remote_ptr)
        return _OwnedSharedBuffer(
            local_ptr=local_ptr,
            peer_ptrs=tuple(peer_ptrs),
            remote_ptrs=tuple(remote_ptrs),
        )

    @property
    def signal_ptrs(self) -> tuple[int, ...]:
        return self._signal_ptrs

    def close(self) -> None:
        for shared in self._owned_buffers:
            for ptr in shared.remote_ptrs:
                self._ipc.cudaIpcCloseMemHandle(ptr)
            self._ipc.cudaFree(shared.local_ptr)
        self._owned_buffers.clear()

    def _select_peer_input_ptrs(self, inp: torch.Tensor, peer_input_ptrs: Optional[Sequence[int]]) -> tuple[int, ...]:
        if peer_input_ptrs is not None:
            if len(peer_input_ptrs) != self.world_size:
                raise ValueError("peer_input_ptrs must match world size")
            return tuple(int(ptr) for ptr in peer_input_ptrs)

        if self._eager_ptrs is None:
            raise ValueError("peer_input_ptrs are required when eager buffers are not configured")

        byte_count = inp.numel() * inp.element_size()
        slot = self._slot % 2
        self._slot += 1
        local_dst = self._eager_ptrs[slot][self.rank]
        self._ipc.cudaMemcpyAsync(
            dst=local_dst,
            src=int(inp.data_ptr()),
            count=byte_count,
            stream=int(torch.cuda.current_stream(device=self.device).cuda_stream),
        )
        return self._eager_ptrs[slot]

    def allreduce_gemma_rmsnorm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        out: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inp.device != self.device:
            raise ValueError(f"input device {inp.device} does not match runtime device {self.device}")
        if residual.device != self.device or weight.device != self.device:
            raise ValueError("all tensors must be on the runtime device")
        if inp.shape != residual.shape:
            raise ValueError("input and residual must have the same shape")
        if inp.dim() != 2:
            raise ValueError("input must be [num_tokens, hidden_size]")
        if weight.dim() != 1 or weight.numel() != inp.shape[1]:
            raise ValueError("weight must be [hidden_size]")
        if inp.shape[0] > _MAX_BLOCKS:
            raise ValueError(f"num_tokens {inp.shape[0]} exceeds {_MAX_BLOCKS}")
        if not inp.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
            raise ValueError("input, residual, and weight must be contiguous")

        if out is None:
            out = torch.empty_like(inp)
        if residual_out is None:
            residual_out = torch.empty_like(inp)
        if out.shape != inp.shape or residual_out.shape != inp.shape:
            raise ValueError("output tensors must match input shape")
        if out.dtype != inp.dtype or residual_out.dtype != inp.dtype:
            raise ValueError("output tensors must match input dtype")
        if not out.is_contiguous() or not residual_out.is_contiguous():
            raise ValueError("output tensors must be contiguous")

        peer_ptrs = self._select_peer_input_ptrs(inp, peer_input_ptrs)
        compiled = _get_gemma_kernel(
            self.world_size,
            int(inp.shape[0]),
            int(inp.shape[1]),
            inp.dtype,
            weight.dtype,
        )

        input_ptr_args = [make_ptr(_cutlass_dtype(inp.dtype), ptr, cute.AddressSpace.gmem, assumed_align=_align_bytes(inp.dtype)) for ptr in peer_ptrs]
        while len(input_ptr_args) < 8:
            input_ptr_args.append(input_ptr_args[0])
        signal_ptr_args = [make_ptr(cutlass.Int32, ptr, cute.AddressSpace.gmem, assumed_align=128) for ptr in self._signal_ptrs]
        while len(signal_ptr_args) < 8:
            signal_ptr_args.append(signal_ptr_args[0])

        compiled(
            *input_ptr_args[:8],
            make_ptr(_cutlass_dtype(inp.dtype), int(residual.data_ptr()), cute.AddressSpace.gmem, assumed_align=_align_bytes(inp.dtype)),
            make_ptr(_cutlass_dtype(inp.dtype), int(out.data_ptr()), cute.AddressSpace.gmem, assumed_align=_align_bytes(inp.dtype)),
            make_ptr(_cutlass_dtype(inp.dtype), int(residual_out.data_ptr()), cute.AddressSpace.gmem, assumed_align=_align_bytes(inp.dtype)),
            make_ptr(_cutlass_dtype(weight.dtype), int(weight.data_ptr()), cute.AddressSpace.gmem, assumed_align=_align_bytes(weight.dtype)),
            *signal_ptr_args[:8],
            make_ptr(cutlass.Int32, self._signal_ptrs[self.rank], cute.AddressSpace.gmem, assumed_align=128),
            self.rank,
            float(eps),
            current_cuda_stream(),
        )
        return out, residual_out

    def capture(self):
        """Placeholder capture helper for future graph-aware pointer exchange."""
        return nullcontext()
