"""Direct PCIe exchange for destination-selected fixed-width records."""

from __future__ import annotations

import ctypes
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from ._cuda_ipc import CudaRTLibrary, cudaIpcMemHandle_t
from .pcie_dma import FLAG_STRIDE
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    _current_stream_key,
    _is_current_stream_capturing,
    _normalize_device,
    _OwnedSharedBuffer,
    _align_up,
)


MAX_WORLD_SIZE = 32
DEFAULT_BARRIER_TIMEOUT_CYCLES = 8_000_000_000
_MAX_INT64 = (1 << 63) - 1
_IPC_HANDLE_BYTES = ctypes.sizeof(cudaIpcMemHandle_t)
_CONFIG_VERSION = 1
_CONFIG_FIELDS = (
    "version",
    "world_size",
    "max_records",
    "record_bytes",
    "flags_bytes",
    "payload_offset",
    "payload_bytes",
    "slab_bytes",
    "flag_stride",
    "slab_alignment",
)


class PCIeSelectedRecordExchangeInitializationError(RuntimeError):
    """The selected-record channel is unavailable and callers should fall back."""


@dataclass(frozen=True)
class _SelectedRecordLayout:
    flags_bytes: int
    payload_offset: int
    payload_bytes: int
    slab_bytes: int


def _selected_record_layout(
    *, world_size: int, max_records: int, record_bytes: int
) -> _SelectedRecordLayout:
    world_size = int(world_size)
    max_records = int(max_records)
    record_bytes = int(record_bytes)
    if not 1 <= world_size <= MAX_WORLD_SIZE:
        raise ValueError(
            f"world_size must be in [1, {MAX_WORLD_SIZE}], got {world_size}"
        )
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if record_bytes <= 0:
        raise ValueError("record_bytes must be positive")
    if max_records > _MAX_INT64 // record_bytes:
        raise ValueError("max_records * record_bytes exceeds int64 capacity")

    flags_bytes = _align_up(
        2 * world_size * FLAG_STRIDE,
        IPC_SLAB_ALIGNMENT,
    )
    payload_offset = flags_bytes
    payload_bytes = max_records * record_bytes
    slab_bytes = payload_offset + payload_bytes
    if slab_bytes > _MAX_INT64:
        raise ValueError("selected-record IPC slab exceeds int64 capacity")
    return _SelectedRecordLayout(
        flags_bytes=flags_bytes,
        payload_offset=payload_offset,
        payload_bytes=payload_bytes,
        slab_bytes=slab_bytes,
    )


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_selected_records.cu")
    return load(
        name="sparkinfer_pcie_selected_records_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def _all_ranks_succeeded(
    status: torch.Tensor,
    local_success: bool,
    process_group: ProcessGroup,
) -> bool:
    status.fill_(1 if local_success else 0)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=process_group)
    return bool(status.item())


def _configuration_values(
    *,
    world_size: int,
    max_records: int,
    record_bytes: int,
    layout: _SelectedRecordLayout,
) -> tuple[int, ...]:
    return (
        _CONFIG_VERSION,
        int(world_size),
        int(max_records),
        int(record_bytes),
        layout.flags_bytes,
        layout.payload_offset,
        layout.payload_bytes,
        layout.slab_bytes,
        FLAG_STRIDE,
        IPC_SLAB_ALIGNMENT,
    )


def _validate_rank_configuration(
    *,
    process_group: ProcessGroup,
    device: torch.device,
    status: torch.Tensor,
    world_size: int,
    max_records: int,
    record_bytes: int,
    layout: _SelectedRecordLayout,
) -> None:
    local_config: Optional[torch.Tensor] = None
    gathered_configs: Optional[torch.Tensor] = None
    local_error: Optional[Exception] = None
    try:
        local_config = torch.tensor(
            _configuration_values(
                world_size=world_size,
                max_records=max_records,
                record_bytes=record_bytes,
                layout=layout,
            ),
            dtype=torch.int64,
            device=device,
        )
        gathered_configs = torch.empty(
            world_size * len(_CONFIG_FIELDS),
            dtype=torch.int64,
            device=device,
        )
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record configuration handshake allocation failed on at least "
            "one rank"
        ) from local_error

    assert local_config is not None
    assert gathered_configs is not None
    try:
        dist.all_gather_into_tensor(
            gathered_configs,
            local_config,
            group=process_group,
        )
    except Exception as exc:
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record configuration handshake collective failed"
        ) from exc

    rows = gathered_configs.view(world_size, len(_CONFIG_FIELDS))
    local_matches = False
    local_error = None
    try:
        local_matches = bool(torch.all(rows == local_config.unsqueeze(0)).item())
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record configuration handshake comparison failed on at least "
            "one rank"
        ) from local_error
    if not _all_ranks_succeeded(status, local_matches, process_group):
        try:
            rank_configs = rows.cpu().tolist()
        except Exception:
            rank_configs = "unavailable"
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record configuration mismatch across ranks "
            f"(fields={_CONFIG_FIELDS}, values={rank_configs})"
        )


def _gather_ipc_handles(
    *,
    process_group: ProcessGroup,
    device: torch.device,
    world_size: int,
    local_ptr: int,
    ipc: CudaRTLibrary,
    status: torch.Tensor,
) -> tuple[bytes, ...]:
    local_handle_tensor: Optional[torch.Tensor] = None
    gathered_handle_tensor: Optional[torch.Tensor] = None
    local_error: Optional[Exception] = None
    try:
        local_handle_tensor = torch.empty(
            _IPC_HANDLE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        gathered_handle_tensor = torch.empty(
            world_size * _IPC_HANDLE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        local_handle = ipc.cudaIpcGetMemHandleBytes(local_ptr)
        if len(local_handle) != _IPC_HANDLE_BYTES:
            raise ValueError(
                f"CUDA IPC handle has {len(local_handle)} bytes, "
                f"expected {_IPC_HANDLE_BYTES}"
            )
        local_handle_tensor.copy_(
            torch.frombuffer(bytearray(local_handle), dtype=torch.uint8)
        )
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record IPC handle preparation failed on at least one rank"
        ) from local_error

    assert local_handle_tensor is not None
    assert gathered_handle_tensor is not None
    try:
        dist.all_gather_into_tensor(
            gathered_handle_tensor,
            local_handle_tensor,
            group=process_group,
        )
    except Exception as exc:
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record IPC handle tensor exchange failed"
        ) from exc

    handles: Optional[tuple[bytes, ...]] = None
    local_error = None
    try:
        gathered_cpu = gathered_handle_tensor.view(world_size, _IPC_HANDLE_BYTES).cpu()
        handles = tuple(bytes(row.tolist()) for row in gathered_cpu)
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record IPC handle materialization failed on at least one rank"
        ) from local_error
    assert handles is not None
    return handles


def _free_shared_buffer(ipc: CudaRTLibrary, shared: _OwnedSharedBuffer) -> None:
    for ptr in shared.remote_ptrs:
        with suppress(Exception):
            ipc.cudaIpcCloseMemHandle(ptr)
    with suppress(Exception):
        ipc.cudaFree(shared.local_ptr)


def _allocate_shared_buffer_rank_consistent(
    *,
    process_group: ProcessGroup,
    rank: int,
    world_size: int,
    device: torch.device,
    size_in_bytes: int,
    ipc: CudaRTLibrary,
    status: torch.Tensor,
) -> _OwnedSharedBuffer:
    local_ptr: Optional[int] = None
    local_error: Optional[Exception] = None
    try:
        local_ptr = ipc.cudaMalloc(size_in_bytes)
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        if local_ptr is not None:
            with suppress(Exception):
                ipc.cudaFree(local_ptr)
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record IPC slab allocation failed on at least one rank"
        ) from local_error

    assert local_ptr is not None
    try:
        ipc.cudaMemset(local_ptr, 0, size_in_bytes)
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        with suppress(Exception):
            ipc.cudaFree(local_ptr)
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record IPC slab initialization failed on at least one rank"
        ) from local_error

    try:
        handles = _gather_ipc_handles(
            process_group=process_group,
            device=device,
            world_size=world_size,
            local_ptr=local_ptr,
            ipc=ipc,
            status=status,
        )
    except Exception:
        with suppress(Exception):
            ipc.cudaFree(local_ptr)
        raise

    peer_ptrs: list[int] = []
    remote_ptrs: list[int] = []
    local_error = None
    try:
        for peer_rank, handle in enumerate(handles):
            if peer_rank == rank:
                peer_ptrs.append(local_ptr)
            else:
                remote_ptr = ipc.cudaIpcOpenMemHandleBytes(handle)
                peer_ptrs.append(remote_ptr)
                remote_ptrs.append(remote_ptr)
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        shared = _OwnedSharedBuffer(
            local_ptr=local_ptr,
            peer_ptrs=tuple(peer_ptrs),
            remote_ptrs=tuple(remote_ptrs),
        )
        _free_shared_buffer(ipc, shared)
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record IPC handle open failed on at least one rank"
        ) from local_error
    opened_every_rank = len(peer_ptrs) == world_size
    if not _all_ranks_succeeded(status, opened_every_rank, process_group):
        shared = _OwnedSharedBuffer(
            local_ptr=local_ptr,
            peer_ptrs=tuple(peer_ptrs),
            remote_ptrs=tuple(remote_ptrs),
        )
        _free_shared_buffer(ipc, shared)
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record IPC setup did not open every rank on all processes"
        )
    shared = _OwnedSharedBuffer(
        local_ptr=local_ptr,
        peer_ptrs=tuple(peer_ptrs),
        remote_ptrs=tuple(remote_ptrs),
    )

    local_error = None
    try:
        stream = torch.cuda.current_stream(device)
        stream_ptr = int(stream.cuda_stream)
        for remote_ptr in shared.remote_ptrs:
            ipc.cudaMemcpyAsync(local_ptr, remote_ptr, 1, stream_ptr)
            ipc.cudaMemcpyAsync(remote_ptr, local_ptr, 1, stream_ptr)
        stream.synchronize()
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(status, local_error is None, process_group):
        _free_shared_buffer(ipc, shared)
        raise PCIeSelectedRecordExchangeInitializationError(
            "selected-record CUDA IPC peer open/copy probe failed on at least one rank"
        ) from local_error
    return shared


class PCIeSelectedRecordExchange:
    """One ordered direct-scatter channel for selected byte records.

    ``local_indices_by_destination`` is destination-major. Its trailing
    dimensions flatten to the output record order. Each entry is either a
    non-negative index into ``records`` or negative when this source does not
    own that selected record. Across all ranks, exactly one source must own
    each destination/output position. Every rank must invoke the channel in
    the same order with the same selected-record count.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        peer_slab_ptrs: Sequence[int],
        payload_offset: int,
        max_records: int,
        record_bytes: int,
        process_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffer: Optional[_OwnedSharedBuffer] = None,
        ext_module=None,
        barrier_timeout_cycles: int = DEFAULT_BARRIER_TIMEOUT_CYCLES,
        stream_affine: bool = True,
    ) -> None:
        layout = _selected_record_layout(
            world_size=world_size,
            max_records=max_records,
            record_bytes=record_bytes,
        )
        if not 0 <= int(rank) < int(world_size):
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if len(peer_slab_ptrs) != int(world_size):
            raise ValueError("peer_slab_ptrs must match world_size")
        if int(payload_offset) < layout.flags_bytes:
            raise ValueError("payload_offset overlaps selected-record barrier flags")
        if int(barrier_timeout_cycles) <= 0:
            raise ValueError("barrier_timeout_cycles must be positive")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = _normalize_device(device)
        self.process_group = process_group
        self.max_records = int(max_records)
        self.record_bytes = int(record_bytes)
        self.payload_offset = int(payload_offset)
        self.barrier_timeout_cycles = int(barrier_timeout_cycles)
        self._ipc = ipc
        self._owned_buffer = owned_buffer
        self._ext = ext_module or _load_extension()
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False

        slab_ptrs = tuple(int(ptr) for ptr in peer_slab_ptrs)
        self._local_payload_ptr = slab_ptrs[self.rank] + self.payload_offset
        self._peer_payload_ptrs = torch.tensor(
            [ptr + self.payload_offset for ptr in slab_ptrs],
            dtype=torch.int64,
            device=self.device,
        )
        self._barrier_publish_ptrs = torch.tensor(
            [
                [
                    slab_ptrs[destination]
                    + (phase * self.world_size + self.rank) * FLAG_STRIDE
                    for destination in range(self.world_size)
                ]
                for phase in range(2)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._barrier_wait_ptrs = torch.tensor(
            [
                [
                    slab_ptrs[self.rank]
                    + (phase * self.world_size + source) * FLAG_STRIDE
                    for source in range(self.world_size)
                ]
                for phase in range(2)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._send_counters = torch.zeros(
            (2, self.world_size),
            dtype=torch.int32,
            device=self.device,
        )
        self._wait_counters = torch.zeros_like(self._send_counters)

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_records: int,
        record_bytes: int,
        barrier_timeout_cycles: int = DEFAULT_BARRIER_TIMEOUT_CYCLES,
        ext_module=None,
        stream_affine: bool = True,
    ) -> "PCIeSelectedRecordExchange":
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        device_obj = _normalize_device(device)
        if device_obj.type != "cuda":
            raise PCIeSelectedRecordExchangeInitializationError(
                "selected-record exchange requires a CUDA device"
            )
        device_index = (
            torch.cuda.current_device()
            if device_obj.index is None
            else int(device_obj.index)
        )
        device_obj = torch.device("cuda", device_index)

        try:
            backend = dist.get_backend(group=process_group)
        except TypeError:
            backend = dist.get_backend(process_group)
        if "nccl" not in str(backend).lower():
            raise PCIeSelectedRecordExchangeInitializationError(
                f"selected-record exchange requires an NCCL process group, got {backend}"
            )

        status = torch.empty(1, dtype=torch.int32, device=device_obj)

        layout: Optional[_SelectedRecordLayout] = None
        local_error: Optional[Exception] = None
        try:
            layout = _selected_record_layout(
                world_size=world_size,
                max_records=max_records,
                record_bytes=record_bytes,
            )
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            raise PCIeSelectedRecordExchangeInitializationError(
                "selected-record configuration is invalid on at least one rank"
            ) from local_error
        assert layout is not None
        _validate_rank_configuration(
            process_group=process_group,
            device=device_obj,
            status=status,
            world_size=world_size,
            max_records=max_records,
            record_bytes=record_bytes,
            layout=layout,
        )

        ext = None
        local_error = None
        try:
            ext = ext_module or _load_extension()
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            raise PCIeSelectedRecordExchangeInitializationError(
                "selected-record CUDA extension failed to load on at least one rank"
            ) from local_error
        assert ext is not None

        ipc: Optional[CudaRTLibrary] = None
        local_error = None
        try:
            ipc = CudaRTLibrary()
            ipc.cudaSetDevice(device_index)
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            raise PCIeSelectedRecordExchangeInitializationError(
                "selected-record CUDA IPC setup failed on at least one rank"
            ) from local_error
        assert ipc is not None

        shared = _allocate_shared_buffer_rank_consistent(
            process_group=process_group,
            rank=rank,
            world_size=world_size,
            device=device_obj,
            size_in_bytes=layout.slab_bytes,
            ipc=ipc,
            status=status,
        )
        exchange: Optional[PCIeSelectedRecordExchange] = None
        local_error = None
        try:
            exchange = cls(
                rank=rank,
                world_size=world_size,
                device=device_obj,
                peer_slab_ptrs=shared.peer_ptrs,
                payload_offset=layout.payload_offset,
                max_records=max_records,
                record_bytes=record_bytes,
                process_group=process_group,
                ipc=ipc,
                owned_buffer=None,
                ext_module=ext,
                barrier_timeout_cycles=barrier_timeout_cycles,
                stream_affine=stream_affine,
            )
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            if exchange is not None:
                exchange._closed = True
            _free_shared_buffer(ipc, shared)
            raise PCIeSelectedRecordExchangeInitializationError(
                "selected-record runtime initialization failed on at least one rank"
            ) from local_error
        assert exchange is not None
        exchange._owned_buffer = shared
        return exchange

    def _bind_stream_key(self, stream_key: Optional[int]) -> None:
        if not self._stream_affine or stream_key is None:
            return
        if self._owner_stream_key is None:
            self._owner_stream_key = int(stream_key)
            return
        if self._owner_stream_key != int(stream_key):
            raise RuntimeError(
                "PCIe selected-record channels are stream-affine; create a "
                "separate channel for each CUDA stream"
            )

    def _check_stream(self) -> None:
        if self.device.type != "cuda":
            return
        stream_key = _current_stream_key(self.device)
        if (
            _is_current_stream_capturing(self.device)
            and self._stream_affine
            and self._owner_stream_key is None
        ):
            raise RuntimeError(
                "PCIe selected-record channels must be bound to their CUDA stream "
                "before first-use graph capture"
            )
        self._bind_stream_key(stream_key)

    def _validate(
        self,
        records: torch.Tensor,
        local_indices_by_destination: torch.Tensor,
        out: torch.Tensor,
    ) -> int:
        if self._closed:
            raise RuntimeError("PCIeSelectedRecordExchange is closed")
        if records.device != self.device:
            raise ValueError("records must be on the exchange device")
        if local_indices_by_destination.device != self.device:
            raise ValueError("local indices must be on the exchange device")
        if out.device != self.device:
            raise ValueError("output must be on the exchange device")
        if records.dtype != torch.uint8 or not records.is_contiguous():
            raise ValueError("records must be contiguous uint8")
        if records.ndim < 2 or records.shape[-1] != self.record_bytes:
            raise ValueError(
                f"records must end in the configured {self.record_bytes}-byte width"
            )
        if local_indices_by_destination.dtype not in (torch.int32, torch.int64):
            raise ValueError("local indices must be int32 or int64")
        if not local_indices_by_destination.is_contiguous():
            raise ValueError("local indices must be contiguous")
        if (
            local_indices_by_destination.ndim < 2
            or local_indices_by_destination.shape[0] != self.world_size
        ):
            raise ValueError(
                "local indices must have shape [world_size, ...selected records]"
            )
        active_records = local_indices_by_destination.numel() // self.world_size
        if active_records > self.max_records:
            raise ValueError(
                f"selected record count {active_records} exceeds capacity "
                f"{self.max_records}"
            )
        if out.dtype != torch.uint8 or not out.is_contiguous():
            raise ValueError("output must be contiguous uint8")
        if out.ndim < 2 or out.shape[-1] != self.record_bytes:
            raise ValueError(
                f"output must end in the configured {self.record_bytes}-byte width"
            )
        if out.numel() != active_records * self.record_bytes:
            raise ValueError(
                "output must contain exactly one record per selected position"
            )
        return active_records

    def exchange(
        self,
        records: torch.Tensor,
        local_indices_by_destination: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter owned records to each destination and fill ``out`` exactly."""
        self._check_stream()
        self._validate(records, local_indices_by_destination, out)
        self._ext.exchange(
            records,
            local_indices_by_destination,
            self._peer_payload_ptrs,
            self._local_payload_ptr,
            self._barrier_publish_ptrs,
            self._barrier_wait_ptrs,
            self._send_counters,
            self._wait_counters,
            out,
            self.record_bytes,
            self.barrier_timeout_cycles,
        )
        return out

    def _release_owned_buffer(self, *, synchronize: bool) -> None:
        if self._owned_buffer is None or self._ipc is None:
            return
        if self.device.type == "cuda":
            device_index = (
                torch.cuda.current_device()
                if self.device.index is None
                else int(self.device.index)
            )
            self._ipc.cudaSetDevice(device_index)
            if synchronize:
                with suppress(Exception):
                    torch.cuda.synchronize(self.device)
        _free_shared_buffer(self._ipc, self._owned_buffer)
        self._owned_buffer = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release_owned_buffer(synchronize=True)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = [
    "PCIeSelectedRecordExchange",
    "PCIeSelectedRecordExchangeInitializationError",
]
