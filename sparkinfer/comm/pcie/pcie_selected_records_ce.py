"""Copy-engine PCIe exchange for destination-selected fixed-width records."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from . import pcie_dma as _pcie_dma
from ._cuda_ipc import CudaRTLibrary
from .pcie_dma import FLAG_STRIDE
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    _current_stream_key,
    _is_current_stream_capturing,
    _normalize_device,
    _OwnedSharedBuffer,
    _align_up,
)
from .pcie_selected_records import (
    DEFAULT_BARRIER_TIMEOUT_CYCLES,
    MAX_WORLD_SIZE,
    PCIeSelectedRecordExchangeInitializationError,
    _all_ranks_succeeded,
    _allocate_shared_buffer_rank_consistent,
    _free_shared_buffer,
)


_MAX_INT64 = (1 << 63) - 1
_CONFIG_VERSION = 2
_COUNT_BYTES = 8
_POSITION_BYTES = 8
_PACKET_HEADER_BYTES = 2 * _COUNT_BYTES
_LAYER_PLANES = 3
_CONFIG_FIELDS = (
    "version",
    "world_size",
    "max_records",
    "record_bytes",
    "layered_max_records",
    "layered_record_bytes",
    "layered_slab_bytes",
    "primary_capacity",
    "overflow_capacity",
    "flags_bytes",
    "receive_primary_base",
    "staging_primary_base",
    "staging_overflow_base",
    "primary_positions_offset",
    "primary_records_offset",
    "primary_stride",
    "overflow_positions_offset",
    "overflow_records_offset",
    "overflow_stride",
    "slab_bytes",
    "flag_stride",
    "slab_alignment",
)


@dataclass(frozen=True)
class _CopyExchangeLayout:
    flags_bytes: int
    receive_primary_base: int
    staging_primary_base: int
    staging_overflow_base: int
    primary_capacity: int
    overflow_capacity: int
    primary_positions_offset: int
    primary_records_offset: int
    primary_stride: int
    overflow_positions_offset: int
    overflow_records_offset: int
    overflow_stride: int
    slab_bytes: int


def _checked_region_end(base: int, count: int, stride: int, name: str) -> int:
    if count < 0 or stride < 0 or base < 0:
        raise ValueError(f"{name} has a negative layout component")
    if count and stride > (_MAX_INT64 - base) // count:
        raise ValueError(f"{name} exceeds int64 capacity")
    return base + count * stride


def _copy_exchange_layout(
    *,
    world_size: int,
    max_records: int,
    record_bytes: int,
    primary_capacity: Optional[int] = None,
) -> _CopyExchangeLayout:
    world_size = int(world_size)
    max_records = int(max_records)
    record_bytes = int(record_bytes)
    if not 2 <= world_size <= MAX_WORLD_SIZE:
        raise ValueError(
            f"world_size must be in [2, {MAX_WORLD_SIZE}], got {world_size}"
        )
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if record_bytes <= 0:
        raise ValueError("record_bytes must be positive")
    if max_records > _MAX_INT64 // record_bytes:
        raise ValueError("max_records * record_bytes exceeds int64 capacity")

    if primary_capacity is None:
        primary_capacity = (max_records + world_size - 1) // world_size
    primary_capacity = int(primary_capacity)
    if not 1 <= primary_capacity <= max_records:
        raise ValueError("primary_capacity must be in [1, max_records]")
    overflow_capacity = max_records - primary_capacity

    flags_bytes = _align_up(
        2 * world_size * FLAG_STRIDE,
        IPC_SLAB_ALIGNMENT,
    )
    primary_positions_offset = _PACKET_HEADER_BYTES
    primary_records_offset = _align_up(
        primary_positions_offset + primary_capacity * _POSITION_BYTES,
        16,
    )
    primary_stride = _align_up(
        _checked_region_end(
            primary_records_offset,
            primary_capacity,
            record_bytes,
            "primary packet",
        ),
        IPC_SLAB_ALIGNMENT,
    )

    # Keep a valid, distinct overflow pointer even when primary_capacity covers
    # the complete pool. No overflow bytes are consumed in that configuration.
    overflow_storage_records = max(1, overflow_capacity)
    overflow_positions_offset = 0
    overflow_records_offset = _align_up(
        overflow_storage_records * _POSITION_BYTES,
        16,
    )
    overflow_stride = _align_up(
        _checked_region_end(
            overflow_records_offset,
            overflow_storage_records,
            record_bytes,
            "overflow packet",
        ),
        IPC_SLAB_ALIGNMENT,
    )

    receive_primary_base = flags_bytes
    staging_primary_base = _align_up(
        _checked_region_end(
            receive_primary_base,
            world_size,
            primary_stride,
            "primary receive area",
        ),
        IPC_SLAB_ALIGNMENT,
    )
    staging_overflow_base = _align_up(
        _checked_region_end(
            staging_primary_base,
            world_size,
            primary_stride,
            "primary staging area",
        ),
        IPC_SLAB_ALIGNMENT,
    )
    slab_bytes = _checked_region_end(
        staging_overflow_base,
        world_size,
        overflow_stride,
        "overflow staging area",
    )
    if slab_bytes > _MAX_INT64:
        raise ValueError("copy-engine selected-record slab exceeds int64 capacity")

    return _CopyExchangeLayout(
        flags_bytes=flags_bytes,
        receive_primary_base=receive_primary_base,
        staging_primary_base=staging_primary_base,
        staging_overflow_base=staging_overflow_base,
        primary_capacity=primary_capacity,
        overflow_capacity=overflow_capacity,
        primary_positions_offset=primary_positions_offset,
        primary_records_offset=primary_records_offset,
        primary_stride=primary_stride,
        overflow_positions_offset=overflow_positions_offset,
        overflow_records_offset=overflow_records_offset,
        overflow_stride=overflow_stride,
        slab_bytes=slab_bytes,
    )


def _largest_layered_capacity_for_slab(
    *,
    world_size: int,
    max_records: int,
    record_bytes: int,
    slab_bytes: int,
) -> tuple[int, _CopyExchangeLayout]:
    max_records = int(max_records)
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if record_bytes > _MAX_INT64 // _LAYER_PLANES:
        raise ValueError("layered record width exceeds int64 capacity")
    layered_record_bytes = record_bytes * _LAYER_PLANES
    low = 1
    high = max_records
    first_layout = _copy_exchange_layout(
        world_size=world_size,
        max_records=1,
        record_bytes=layered_record_bytes,
    )
    if first_layout.slab_bytes > int(slab_bytes):
        raise ValueError("base slab cannot hold one three-layer record")
    while low < high:
        candidate = (low + high + 1) // 2
        candidate_layout = _copy_exchange_layout(
            world_size=world_size,
            max_records=candidate,
            record_bytes=layered_record_bytes,
        )
        if candidate_layout.slab_bytes <= int(slab_bytes):
            low = candidate
        else:
            high = candidate - 1
    layered_layout = _copy_exchange_layout(
        world_size=world_size,
        max_records=low,
        record_bytes=layered_record_bytes,
    )
    return low, layered_layout


def _layered_layout_for_slab(
    *,
    world_size: int,
    max_records: int,
    record_bytes: int,
    slab_bytes: int,
    layered_max_records: Optional[int],
    fallback_layout: _CopyExchangeLayout,
) -> tuple[int, _CopyExchangeLayout]:
    if layered_max_records is None:
        try:
            return _largest_layered_capacity_for_slab(
                world_size=world_size,
                max_records=max_records,
                record_bytes=record_bytes,
                slab_bytes=slab_bytes,
            )
        except ValueError as exc:
            if str(exc) != "base slab cannot hold one three-layer record":
                raise
            return 0, fallback_layout

    layered_max_records = int(layered_max_records)
    if layered_max_records == 0:
        return 0, fallback_layout
    if not 1 <= layered_max_records <= int(max_records):
        raise ValueError("layered_max_records must be in [0, max_records]")
    if record_bytes > _MAX_INT64 // _LAYER_PLANES:
        raise ValueError("layered record width exceeds int64 capacity")
    layered_layout = _copy_exchange_layout(
        world_size=world_size,
        max_records=layered_max_records,
        record_bytes=record_bytes * _LAYER_PLANES,
    )
    if layered_layout.slab_bytes > int(slab_bytes):
        raise ValueError(
            "three-layer layout exceeds the existing selected-record slab "
            f"({layered_layout.slab_bytes} > {slab_bytes} bytes)"
        )
    return layered_max_records, layered_layout


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_selected_records_ce.cu")
    return load(
        name="sparkinfer_pcie_selected_records_ce_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def _configuration_values(
    *,
    world_size: int,
    max_records: int,
    record_bytes: int,
    layout: _CopyExchangeLayout,
    layered_max_records: int,
    layered_layout: _CopyExchangeLayout,
) -> tuple[int, ...]:
    return (
        _CONFIG_VERSION,
        int(world_size),
        int(max_records),
        int(record_bytes),
        int(layered_max_records),
        int(record_bytes) * _LAYER_PLANES,
        layered_layout.slab_bytes,
        layout.primary_capacity,
        layout.overflow_capacity,
        layout.flags_bytes,
        layout.receive_primary_base,
        layout.staging_primary_base,
        layout.staging_overflow_base,
        layout.primary_positions_offset,
        layout.primary_records_offset,
        layout.primary_stride,
        layout.overflow_positions_offset,
        layout.overflow_records_offset,
        layout.overflow_stride,
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
    layout: _CopyExchangeLayout,
    layered_max_records: int,
    layered_layout: _CopyExchangeLayout,
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
                layered_max_records=layered_max_records,
                layered_layout=layered_layout,
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
            "copy-engine selected-record configuration allocation failed on at "
            "least one rank"
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
            "copy-engine selected-record configuration collective failed"
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
            "copy-engine selected-record configuration comparison failed on at "
            "least one rank"
        ) from local_error
    if not _all_ranks_succeeded(status, local_matches, process_group):
        try:
            rank_configs = rows.cpu().tolist()
        except Exception:
            rank_configs = "unavailable"
        raise PCIeSelectedRecordExchangeInitializationError(
            "copy-engine selected-record configuration mismatch across ranks "
            f"(fields={_CONFIG_FIELDS}, values={rank_configs})"
        )


class PCIeSelectedRecordCopyExchange:
    """Ordered pack/copy-engine/unpack exchange for selected byte records.

    ``local_indices_by_destination`` is destination-major. Its trailing
    dimensions flatten to the output record order. Each entry is either a
    non-negative index into ``records`` or negative when this source does not
    own that selected record. Across all ranks, exactly one source must own
    each destination/output position. Every rank must invoke the channel in
    the same order with the same selected-record count.

    Records are compacted per destination. A balanced primary packet is moved
    with the CUDA copy engine; owner skew beyond that packet remains in the
    source IPC slab and is read during unpack. Both exchange modes publish a
    release after unpack and defer the matching wait until the next staging
    reuse. Keeping that protocol identical makes separately captured normal and
    layered CUDA graphs safe to alternate. ``exchange_layers`` packs exactly
    three record planes into one packet. The layered packet layout overlays the
    same IPC slab with a smaller ``layered_max_records`` capacity; callers must
    use the single-layer fallback above that bound. Configurations whose slab
    cannot hold even one layered record expose a zero layered capacity while
    retaining the normal exchange.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        peer_slab_ptrs: Sequence[int],
        max_records: int,
        record_bytes: int,
        primary_capacity: Optional[int] = None,
        layered_max_records: Optional[int] = None,
        process_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffer: Optional[_OwnedSharedBuffer] = None,
        ext_module=None,
        dma_ext_module=None,
        barrier_timeout_cycles: int = DEFAULT_BARRIER_TIMEOUT_CYCLES,
        stream_affine: bool = True,
    ) -> None:
        layout = _copy_exchange_layout(
            world_size=world_size,
            max_records=max_records,
            record_bytes=record_bytes,
            primary_capacity=primary_capacity,
        )
        layered_max_records, layered_layout = _layered_layout_for_slab(
            world_size=world_size,
            max_records=max_records,
            record_bytes=record_bytes,
            slab_bytes=layout.slab_bytes,
            layered_max_records=layered_max_records,
            fallback_layout=layout,
        )
        if not 0 <= int(rank) < int(world_size):
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if len(peer_slab_ptrs) != int(world_size):
            raise ValueError("peer_slab_ptrs must match world_size")
        if int(barrier_timeout_cycles) <= 0:
            raise ValueError("barrier_timeout_cycles must be positive")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = _normalize_device(device)
        self.process_group = process_group
        self.max_records = int(max_records)
        self.record_bytes = int(record_bytes)
        self.layered_max_records = layered_max_records
        self.primary_capacity = layout.primary_capacity
        self.overflow_capacity = layout.overflow_capacity
        self.barrier_timeout_cycles = int(barrier_timeout_cycles)
        self._layout = layout
        self._layer_layout = layered_layout
        self._ipc = ipc
        self._owned_buffer = owned_buffer
        self._ext = ext_module or _load_extension()
        self._dma_ext = dma_ext_module or _pcie_dma._load_extension()
        self._stream_affine = bool(stream_affine)
        self._owner_stream_key: Optional[int] = None
        self._closed = False
        self._deferred_release_pending = False

        slab_ptrs = tuple(int(pointer) for pointer in peer_slab_ptrs)
        self._slab_ptrs = slab_ptrs
        self._local_slab_ptr = slab_ptrs[self.rank]
        self._peer_overflow_ptrs = torch.tensor(
            [
                [
                    slab_ptrs[source]
                    + layout.staging_overflow_base
                    + destination * layout.overflow_stride
                    for source in range(self.world_size)
                ]
                for destination in range(self.world_size)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._peer_layer_overflow_ptrs = torch.tensor(
            [
                [
                    slab_ptrs[source]
                    + layered_layout.staging_overflow_base
                    + destination * layered_layout.overflow_stride
                    for source in range(self.world_size)
                ]
                for destination in range(self.world_size)
            ],
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
        dma_ext_module=None,
        stream_affine: bool = True,
        primary_capacity: Optional[int] = None,
        layered_max_records: Optional[int] = None,
    ) -> "PCIeSelectedRecordCopyExchange":
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        device_obj = _normalize_device(device)
        if device_obj.type != "cuda":
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record exchange requires a CUDA device"
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
                "copy-engine selected-record exchange requires an NCCL process "
                f"group, got {backend}"
            )

        status = torch.empty(1, dtype=torch.int32, device=device_obj)
        layout: Optional[_CopyExchangeLayout] = None
        layered_layout: Optional[_CopyExchangeLayout] = None
        resolved_layered_max_records: Optional[int] = None
        local_error: Optional[Exception] = None
        try:
            layout = _copy_exchange_layout(
                world_size=world_size,
                max_records=max_records,
                record_bytes=record_bytes,
                primary_capacity=primary_capacity,
            )
            resolved_layered_max_records, layered_layout = _layered_layout_for_slab(
                world_size=world_size,
                max_records=max_records,
                record_bytes=record_bytes,
                slab_bytes=layout.slab_bytes,
                layered_max_records=layered_max_records,
                fallback_layout=layout,
            )
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record configuration is invalid on at "
                "least one rank"
            ) from local_error
        assert layout is not None
        assert layered_layout is not None
        assert resolved_layered_max_records is not None
        _validate_rank_configuration(
            process_group=process_group,
            device=device_obj,
            status=status,
            world_size=world_size,
            max_records=max_records,
            record_bytes=record_bytes,
            layout=layout,
            layered_max_records=resolved_layered_max_records,
            layered_layout=layered_layout,
        )

        ext = None
        dma_ext = None
        local_error = None
        try:
            ext = ext_module or _load_extension()
            dma_ext = dma_ext_module or _pcie_dma._load_extension()
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record CUDA extensions failed to load on "
                "at least one rank"
            ) from local_error
        assert ext is not None
        assert dma_ext is not None

        ipc: Optional[CudaRTLibrary] = None
        local_error = None
        try:
            ipc = CudaRTLibrary()
            ipc.cudaSetDevice(device_index)
        except Exception as exc:
            local_error = exc
        if not _all_ranks_succeeded(status, local_error is None, process_group):
            raise PCIeSelectedRecordExchangeInitializationError(
                "copy-engine selected-record CUDA IPC setup failed on at least one rank"
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
        exchange: Optional[PCIeSelectedRecordCopyExchange] = None
        local_error = None
        try:
            exchange = cls(
                rank=rank,
                world_size=world_size,
                device=device_obj,
                peer_slab_ptrs=shared.peer_ptrs,
                max_records=max_records,
                record_bytes=record_bytes,
                primary_capacity=layout.primary_capacity,
                layered_max_records=resolved_layered_max_records,
                process_group=process_group,
                ipc=ipc,
                owned_buffer=None,
                ext_module=ext,
                dma_ext_module=dma_ext,
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
                "copy-engine selected-record runtime initialization failed on at "
                "least one rank"
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
                "PCIe selected-record copy channels are stream-affine; create a "
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
                "PCIe selected-record copy channels must be bound to their CUDA "
                "stream before first-use graph capture"
            )
        self._bind_stream_key(stream_key)

    def _validate(
        self,
        records: torch.Tensor,
        local_indices_by_destination: torch.Tensor,
        out: torch.Tensor,
    ) -> int:
        if self._closed:
            raise RuntimeError("PCIeSelectedRecordCopyExchange is closed")
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
        if active_records > _MAX_INT64 // self.record_bytes:
            raise ValueError("active selected-record payload exceeds int64 capacity")
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

    def _active_primary_capacity(self, active_records: int) -> int:
        if active_records == 0:
            return 0
        scaled = (
            self.primary_capacity * active_records + self.max_records - 1
        ) // self.max_records
        active_primary = min(scaled, self.primary_capacity, active_records)
        active_overflow = active_records - active_primary
        if active_overflow > self.overflow_capacity:
            raise ValueError("active selected-record overflow exceeds channel capacity")
        return active_primary

    def _active_layer_primary_capacity(self, active_records: int) -> int:
        if active_records == 0:
            return 0
        layout = self._layer_layout
        scaled = (
            layout.primary_capacity * active_records + self.layered_max_records - 1
        ) // self.layered_max_records
        active_primary = min(
            scaled,
            layout.primary_capacity,
            active_records,
        )
        active_overflow = active_records - active_primary
        if active_overflow > layout.overflow_capacity:
            raise ValueError(
                "active three-layer selected-record overflow exceeds channel capacity"
            )
        return active_primary

    def _receive_primary_ptr(self, destination: int, source: int) -> int:
        return (
            self._slab_ptrs[destination]
            + self._layout.receive_primary_base
            + source * self._layout.primary_stride
        )

    def _local_staging_primary_ptr(self, destination: int) -> int:
        return (
            self._local_slab_ptr
            + self._layout.staging_primary_base
            + destination * self._layout.primary_stride
        )

    def _local_staging_overflow_ptr(self, destination: int) -> int:
        return (
            self._local_slab_ptr
            + self._layout.staging_overflow_base
            + destination * self._layout.overflow_stride
        )

    def _layer_receive_primary_ptr(self, destination: int, source: int) -> int:
        return (
            self._slab_ptrs[destination]
            + self._layer_layout.receive_primary_base
            + source * self._layer_layout.primary_stride
        )

    def _local_layer_staging_primary_ptr(self, destination: int) -> int:
        return (
            self._local_slab_ptr
            + self._layer_layout.staging_primary_base
            + destination * self._layer_layout.primary_stride
        )

    def _local_layer_staging_overflow_ptr(self, destination: int) -> int:
        return (
            self._local_slab_ptr
            + self._layer_layout.staging_overflow_base
            + destination * self._layer_layout.overflow_stride
        )

    def _barrier(self, phase: int) -> None:
        self._ext.barrier_all_peers(
            self._barrier_publish_ptrs,
            self._barrier_wait_ptrs,
            self._send_counters,
            self._wait_counters,
            int(phase),
            self.barrier_timeout_cycles,
        )

    def _publish_deferred_release(self) -> None:
        self._ext.publish_all_peers(
            self._barrier_publish_ptrs,
            self._send_counters,
            1,
        )
        self._deferred_release_pending = True

    def _wait_for_deferred_release(self) -> None:
        if not self._deferred_release_pending:
            return
        self._ext.wait_all_peers(
            self._barrier_wait_ptrs,
            self._wait_counters,
            1,
            self.barrier_timeout_cycles,
        )
        self._deferred_release_pending = False

    def _require_eager_warmup_for_capture(self) -> None:
        if (
            self.device.type == "cuda"
            and _is_current_stream_capturing(self.device)
            and not self._deferred_release_pending
        ):
            raise RuntimeError(
                "copy-engine selected-record exchange requires one eager "
                "warmup before CUDA graph capture"
            )

    def _validate_layers(
        self,
        records_by_layer: Sequence[torch.Tensor],
        local_indices_by_destination: torch.Tensor,
        outputs_by_layer: Sequence[torch.Tensor],
    ) -> int:
        if len(records_by_layer) != _LAYER_PLANES:
            raise ValueError(f"records_by_layer must contain {_LAYER_PLANES} tensors")
        if len(outputs_by_layer) != _LAYER_PLANES:
            raise ValueError(f"outputs_by_layer must contain {_LAYER_PLANES} tensors")

        active_records: Optional[int] = None
        for records, out in zip(records_by_layer, outputs_by_layer, strict=True):
            current = self._validate(records, local_indices_by_destination, out)
            if active_records is None:
                active_records = current
            elif current != active_records:
                raise ValueError("all layer outputs must have the same record count")

        assert active_records is not None
        if active_records:
            output_ranges = sorted(
                (
                    int(output.data_ptr()),
                    int(output.data_ptr()) + output.numel(),
                )
                for output in outputs_by_layer
            )
            if any(
                output_ranges[index][1] > output_ranges[index + 1][0]
                for index in range(len(output_ranges) - 1)
            ):
                raise ValueError("outputs_by_layer must use non-overlapping storage")
        if active_records > self.layered_max_records:
            raise ValueError(
                f"three-layer selected record count {active_records} exceeds "
                f"bounded capacity {self.layered_max_records}; use the "
                "single-layer exchange fallback"
            )
        record_shapes = {tuple(records.shape) for records in records_by_layer}
        if len(record_shapes) != 1:
            raise ValueError("all layer record tensors must have the same shape")
        return active_records

    def exchange(
        self,
        records: torch.Tensor,
        local_indices_by_destination: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Pack owned records, copy them to peers, and reconstruct ``out``."""
        self._check_stream()
        self._require_eager_warmup_for_capture()
        active_records = self._validate(
            records,
            local_indices_by_destination,
            out,
        )
        self._wait_for_deferred_release()
        if active_records == 0:
            self._barrier(0)
            out.zero_()
            self._publish_deferred_release()
            return out

        active_primary = self._active_primary_capacity(active_records)
        primary_copy_bytes = _align_up(
            self._layout.primary_records_offset + active_primary * self.record_bytes,
            16,
        )

        # Rotate both pack and CE issue order by source rank. At phase zero,
        # rank R targets destination R instead of every rank targeting rank 0.
        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._ext.pack_compact_records(
                records,
                local_indices_by_destination[destination],
                self._local_staging_primary_ptr(destination),
                self._local_staging_overflow_ptr(destination),
                self.record_bytes,
                active_primary,
                self._layout.primary_positions_offset,
                self._layout.primary_records_offset,
                self._layout.overflow_positions_offset,
                self._layout.overflow_records_offset,
            )

        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._dma_ext.dma_copy(
                self._receive_primary_ptr(destination, self.rank),
                self._local_staging_primary_ptr(destination),
                primary_copy_bytes,
            )

        self._barrier(0)
        out.zero_()
        self._ext.unpack_compact_records(
            self._local_slab_ptr + self._layout.receive_primary_base,
            self._layout.primary_stride,
            self._peer_overflow_ptrs[self.rank],
            out,
            active_records,
            self.record_bytes,
            active_primary,
            self._layout.primary_positions_offset,
            self._layout.primary_records_offset,
            self._layout.overflow_positions_offset,
            self._layout.overflow_records_offset,
        )
        self._publish_deferred_release()
        return out

    def exchange_layers(
        self,
        records_by_layer: Sequence[torch.Tensor],
        local_indices_by_destination: torch.Tensor,
        outputs_by_layer: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exchange three layer planes with one shared destination-position table.

        Each selected slot carries three native records in one packet. The
        arrival barrier protects packet visibility. Unpack then publishes a
        release generation without waiting; the matching wait is issued only
        before this channel reuses staging for its next exchange.
        """
        self._check_stream()
        self._require_eager_warmup_for_capture()
        active_records = self._validate_layers(
            records_by_layer,
            local_indices_by_destination,
            outputs_by_layer,
        )
        outputs = (
            outputs_by_layer[0],
            outputs_by_layer[1],
            outputs_by_layer[2],
        )
        self._wait_for_deferred_release()

        if active_records == 0:
            self._barrier(0)
            for output in outputs:
                output.zero_()
            self._publish_deferred_release()
            return outputs

        active_primary = self._active_layer_primary_capacity(active_records)
        layout = self._layer_layout
        packet_record_bytes = self.record_bytes * _LAYER_PLANES
        primary_copy_bytes = _align_up(
            layout.primary_records_offset + active_primary * packet_record_bytes,
            16,
        )

        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._ext.pack_compact_record_layers(
                records_by_layer[0],
                records_by_layer[1],
                records_by_layer[2],
                local_indices_by_destination[destination],
                self._local_layer_staging_primary_ptr(destination),
                self._local_layer_staging_overflow_ptr(destination),
                self.record_bytes,
                active_primary,
                layout.primary_positions_offset,
                layout.primary_records_offset,
                layout.overflow_positions_offset,
                layout.overflow_records_offset,
            )

        for destination_phase in range(self.world_size):
            destination = (self.rank + destination_phase) % self.world_size
            self._dma_ext.dma_copy(
                self._layer_receive_primary_ptr(destination, self.rank),
                self._local_layer_staging_primary_ptr(destination),
                primary_copy_bytes,
            )

        self._barrier(0)
        for output in outputs:
            output.zero_()
        self._ext.unpack_compact_record_layers(
            self._local_slab_ptr + layout.receive_primary_base,
            layout.primary_stride,
            self._peer_layer_overflow_ptrs[self.rank],
            outputs[0],
            outputs[1],
            outputs[2],
            active_records,
            self.record_bytes,
            active_primary,
            layout.primary_positions_offset,
            layout.primary_records_offset,
            layout.overflow_positions_offset,
            layout.overflow_records_offset,
        )
        self._publish_deferred_release()
        return outputs

    def _finish_deferred_release_for_close(self) -> None:
        if not self._deferred_release_pending:
            return
        if self.device.type == "cuda":
            device_index = (
                torch.cuda.current_device()
                if self.device.index is None
                else int(self.device.index)
            )
            if self._ipc is not None:
                self._ipc.cudaSetDevice(device_index)
            torch.cuda.synchronize(self.device)
        self._wait_for_deferred_release()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

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
        self._finish_deferred_release_for_close()
        self._closed = True
        self._release_owned_buffer(synchronize=True)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = [
    "PCIeSelectedRecordCopyExchange",
    "PCIeSelectedRecordExchangeInitializationError",
]
