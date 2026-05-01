"""PCIe oneshot allreduce runtime with optional crossover autotuning."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from ._cuda_ipc import CudaRTLibrary


logger = logging.getLogger(__name__)

SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DEFAULT_MAX_SIZE = 8 * 1024 * 1024
DEFAULT_RANK_DATA_BYTES = 8 * 1024 * 1024
AUTOTUNE_CEILING = 1 * 1024 * 1024
AUTOTUNE_FINE_STEP = 8 * 1024


def parse_pcie_oneshot_max_size(value: str | int | None) -> Optional[int]:
    """Parse a byte-size string, or return ``None`` for ``auto``."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if value.lower() == "auto":
        return None
    normalized = value.upper().strip()
    suffixes = {
        "KB": 1024,
        "K": 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
    }
    for suffix, multiplier in sorted(suffixes.items(), key=lambda item: -len(item[0])):
        if normalized.endswith(suffix):
            return int(normalized[: -len(suffix)]) * multiplier
    return int(value)


def _normalize_device(device: torch.device | int | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    if inp.is_contiguous():
        return True
    storage = inp.untyped_storage()
    return storage.nbytes() - inp.storage_offset() * inp.element_size() == inp.numel() * inp.element_size()


def _resolve_exchange_group(
    exchange_group: Optional[ProcessGroup],
    process_group: Optional[ProcessGroup],
) -> Optional[ProcessGroup]:
    if exchange_group is not None and process_group is not None and exchange_group is not process_group:
        raise ValueError("pass only one of exchange_group or process_group")
    return exchange_group if exchange_group is not None else process_group


def _group_ranks(group: ProcessGroup) -> list[int]:
    world_size = dist.get_world_size(group=group)
    if hasattr(dist, "get_process_group_ranks"):
        ranks = sorted(dist.get_process_group_ranks(group=group))
        if len(ranks) != world_size:
            raise RuntimeError("process-group rank list does not match world size")
        return ranks
    return list(range(world_size))


def _broadcast_gather_object(local_object: object, group: ProcessGroup) -> list[object]:
    # `broadcast_object_list` is more robust than `all_gather_object` for
    # the CPU-side exchange that happens around IPC setup and graph capture.
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    all_objects: list[list[object | None]] = [[None] for _ in range(world_size)]
    all_objects[rank][0] = local_object
    for index, src_rank in enumerate(_group_ranks(group)):
        try:
            dist.broadcast_object_list(all_objects[index], src=src_rank, group=group, device="cpu")
        except RuntimeError as exc:
            if "No backend type associated with device type cpu" not in str(exc):
                raise
            cuda_index = torch.cuda.current_device()
            dist.broadcast_object_list(
                all_objects[index],
                src=src_rank,
                group=group,
                device=torch.device("cuda", cuda_index),
            )
    return [entry[0] for entry in all_objects]


@dataclass(frozen=True)
class _OwnedSharedBuffer:
    local_ptr: int
    peer_ptrs: tuple[int, ...]
    remote_ptrs: tuple[int, ...]


@dataclass(frozen=True)
class _BenchmarkResult:
    size_bytes: int
    custom_us: float
    nccl_us: float
    winner: str


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_oneshot.cu")
    verbose = os.getenv("B12X_PCIE_ONESHOT_VERBOSE_BUILD", "0") == "1"
    return load(
        name="b12x_pcie_oneshot_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O2", "--expt-relaxed-constexpr"],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def _compute_crossover_size(
    benchmark: Callable[[int], tuple[float, float]],
    *,
    ceiling_bytes: int = AUTOTUNE_CEILING,
    fine_step_bytes: int = AUTOTUNE_FINE_STEP,
) -> tuple[int, list[_BenchmarkResult]]:
    coarse_sizes = []
    current = 1024
    while current <= ceiling_bytes:
        coarse_sizes.append(current)
        current *= 2

    results: list[_BenchmarkResult] = []
    seen_sizes: set[int] = set()
    first_nccl_win: Optional[int] = None
    last_custom_win = 0

    def record(size_bytes: int) -> None:
        nonlocal first_nccl_win, last_custom_win
        custom_us, nccl_us = benchmark(size_bytes)
        winner = "custom" if custom_us < nccl_us else "NCCL"
        results.append(_BenchmarkResult(size_bytes, custom_us, nccl_us, winner))
        seen_sizes.add(size_bytes)
        if winner == "custom":
            last_custom_win = max(last_custom_win, size_bytes)
        elif first_nccl_win is None:
            first_nccl_win = size_bytes

    for size_bytes in coarse_sizes:
        record(size_bytes)

    if last_custom_win > 0 and first_nccl_win is not None:
        fine_start = last_custom_win
        fine_end = min(first_nccl_win, last_custom_win * 4)
        fine_size = fine_start + fine_step_bytes
        while fine_size < fine_end:
            aligned = (fine_size // 16) * 16
            if aligned not in seen_sizes:
                record(aligned)
            fine_size += fine_step_bytes

    results.sort(key=lambda item: item.size_bytes)
    crossover = 1024
    for result in results:
        if result.winner == "custom":
            crossover = result.size_bytes
    return crossover, results


class PCIeOneshotAllReduce:
    """Standalone unfused PCIe oneshot allreduce runtime."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device | int | str,
        signal_ptrs: Sequence[int],
        eager_buffer_ptrs0: Optional[Sequence[int]] = None,
        eager_buffer_ptrs1: Optional[Sequence[int]] = None,
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        ipc: Optional[CudaRTLibrary] = None,
        owned_buffers: Optional[Sequence[_OwnedSharedBuffer]] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
    ):
        if world_size not in SUPPORTED_WORLD_SIZES:
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
        self.device = _normalize_device(device)
        self.exchange_group = _resolve_exchange_group(exchange_group, process_group)
        # Compatibility alias for older callers that still refer to this as
        # `process_group`.
        self.process_group = self.exchange_group
        self.max_size = int(max_size)
        self._ipc = ipc
        if self._ipc is None and (ext_module is None or owned_buffers):
            self._ipc = CudaRTLibrary()
        self._signal_ptrs = tuple(int(ptr) for ptr in signal_ptrs)
        self._owned_buffers = list(owned_buffers or [])
        self._registered_input_ptrs: dict[int, tuple[int, ...]] = {}
        self._closed = False
        self._ext = ext_module or _load_extension()

        if ext_module is None and self.device.type != "cuda":
            raise ValueError("PCIe oneshot allreduce requires a CUDA device")

        self.rank_data = torch.empty(rank_data_bytes, dtype=torch.uint8, device=self.device)
        self._ptr = self._ext.init_custom_ar(list(self._signal_ptrs), self.rank_data, self.rank)

        self._eager_ptrs: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None
        if eager_buffer_ptrs0 is not None and eager_buffer_ptrs1 is not None:
            self._eager_ptrs = (
                tuple(int(ptr) for ptr in eager_buffer_ptrs0),
                tuple(int(ptr) for ptr in eager_buffer_ptrs1),
            )
            self._ext.register_pcie_buffers(
                self._ptr,
                list(self._eager_ptrs[0]),
                list(self._eager_ptrs[1]),
            )

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
        exchange_group: Optional[ProcessGroup] = None,
        process_group: Optional[ProcessGroup] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
    ) -> "PCIeOneshotAllReduce":
        return cls(
            rank=rank,
            world_size=world_size,
            device=device,
            signal_ptrs=signal_ptrs,
            eager_buffer_ptrs0=eager_buffer_ptrs0,
            eager_buffer_ptrs1=eager_buffer_ptrs1,
            exchange_group=exchange_group,
            process_group=process_group,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
        )

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
    ) -> "PCIeOneshotAllReduce":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")

        device_obj = _normalize_device(device)
        if device_obj.type != "cuda":
            raise ValueError("PCIe oneshot requires a CUDA device")

        ipc = CudaRTLibrary()
        ipc.cudaSetDevice(device_obj.index or 0)
        ext = ext_module or _load_extension()

        owned_buffers: list[_OwnedSharedBuffer] = []
        signal_buf = cls._allocate_shared_buffer(
            exchange_group,
            ext.meta_size(),
            zero_fill=True,
            ipc=ipc,
        )
        owned_buffers.append(signal_buf)
        eager0 = cls._allocate_shared_buffer(
            exchange_group,
            eager_buffer_bytes,
            zero_fill=False,
            ipc=ipc,
        )
        eager1 = cls._allocate_shared_buffer(
            exchange_group,
            eager_buffer_bytes,
            zero_fill=False,
            ipc=ipc,
        )
        owned_buffers.extend([eager0, eager1])

        return cls(
            rank=rank,
            world_size=world_size,
            device=device_obj,
            signal_ptrs=signal_buf.peer_ptrs,
            eager_buffer_ptrs0=eager0.peer_ptrs,
            eager_buffer_ptrs1=eager1.peer_ptrs,
            exchange_group=exchange_group,
            ipc=ipc,
            owned_buffers=owned_buffers,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
    ) -> "PCIeOneshotAllReduce":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=max_input_bytes if eager_buffer_bytes is None else eager_buffer_bytes,
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
        )

    @staticmethod
    def _allocate_shared_buffer(
        exchange_group: ProcessGroup,
        size_in_bytes: int,
        *,
        zero_fill: bool,
        ipc: CudaRTLibrary,
    ) -> _OwnedSharedBuffer:
        local_ptr = ipc.cudaMalloc(size_in_bytes)
        if zero_fill:
            ipc.cudaMemset(local_ptr, 0, size_in_bytes)
        local_handle = ipc.cudaIpcGetMemHandleBytes(local_ptr)
        world_size = dist.get_world_size(group=exchange_group)
        rank = dist.get_rank(group=exchange_group)
        handles = _broadcast_gather_object(local_handle, exchange_group)

        peer_ptrs: list[int] = []
        remote_ptrs: list[int] = []
        for idx, handle in enumerate(handles):
            if idx == rank:
                peer_ptrs.append(local_ptr)
            else:
                remote_ptr = ipc.cudaIpcOpenMemHandleBytes(handle)
                peer_ptrs.append(remote_ptr)
                remote_ptrs.append(remote_ptr)
        if len(peer_ptrs) != world_size:
            raise RuntimeError("failed to gather IPC handles for all ranks")
        return _OwnedSharedBuffer(
            local_ptr=local_ptr,
            peer_ptrs=tuple(peer_ptrs),
            remote_ptrs=tuple(remote_ptrs),
        )

    @property
    def signal_ptrs(self) -> tuple[int, ...]:
        return self._signal_ptrs

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._closed:
            return False
        if inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        inp_bytes = inp.numel() * inp.element_size()
        if inp_bytes > self.max_size:
            return False
        if inp_bytes % 16 != 0:
            return False
        if not _is_weak_contiguous(inp):
            return False
        return True

    def register_buffer(self, peer_input_ptrs: Sequence[int]) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if len(peer_input_ptrs) != self.world_size:
            raise ValueError("peer_input_ptrs must match world size")
        ptrs = tuple(int(ptr) for ptr in peer_input_ptrs)
        local_ptr = ptrs[self.rank]
        existing = self._registered_input_ptrs.get(local_ptr)
        if existing is not None:
            if existing != ptrs:
                raise ValueError("input pointer is already registered with different peer_input_ptrs")
            return
        self._ext.register_buffer(self._ptr, list(ptrs))
        self._registered_input_ptrs[local_ptr] = ptrs

    def get_graph_buffer_ipc_meta(self) -> tuple[list[int], list[int]]:
        if self._closed:
            raise RuntimeError("runtime is closed")
        handle, offsets = self._ext.get_graph_buffer_ipc_meta(self._ptr)
        return list(handle), list(offsets)

    def register_graph_buffers_from_ranks(
        self,
        handles: Sequence[Sequence[int]],
        offsets: Sequence[Sequence[int]],
    ) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if len(handles) != self.world_size:
            raise ValueError("handles must match world size")
        if len(offsets) != self.world_size:
            raise ValueError("offsets must match world size")
        self._ext.register_graph_buffers(
            self._ptr,
            [list(map(int, handle)) for handle in handles],
            [list(map(int, rank_offsets)) for rank_offsets in offsets],
        )

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if inp.device != self.device:
            raise ValueError(f"input device {inp.device} does not match runtime device {self.device}")
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy device/dtype/size/alignment/contiguity requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )

        if out is None:
            out = torch.empty_like(inp)
        if out.device != inp.device:
            raise ValueError("output tensor must be on the same device as the input")
        if out.shape != inp.shape or out.dtype != inp.dtype:
            raise ValueError("output tensor must match input shape and dtype")
        if not _is_weak_contiguous(out):
            raise ValueError("output tensor must be weak-contiguous")

        local_ptr = int(inp.data_ptr())
        if peer_input_ptrs is not None:
            if len(peer_input_ptrs) != self.world_size:
                raise ValueError("peer_input_ptrs must match world size")
            ptrs = tuple(int(ptr) for ptr in peer_input_ptrs)
            if ptrs[self.rank] != local_ptr:
                raise ValueError("peer_input_ptrs[self.rank] must match inp.data_ptr()")
            self.register_buffer(ptrs)
        elif self._eager_ptrs is None and local_ptr not in self._registered_input_ptrs:
            raise ValueError(
                "peer_input_ptrs are required unless eager IPC buffers are configured "
                "or this input was already registered"
            )

        self._ext.all_reduce(self._ptr, inp, out, 0, 0)
        return out

    @contextmanager
    def capture(self):
        if self.exchange_group is None:
            raise ValueError("exchange_group is required for CUDA graph capture registration")
        try:
            yield
        finally:
            self.register_graph_buffers()

    def register_graph_buffers(self) -> None:
        if self.exchange_group is None:
            raise ValueError("exchange_group is required to register graph buffers")
        local_meta = self.get_graph_buffer_ipc_meta()
        all_meta = _broadcast_gather_object(local_meta, self.exchange_group)
        self.register_graph_buffers_from_ranks(
            [entry[0] for entry in all_meta],
            [entry[1] for entry in all_meta],
        )

    def _bench_graph_latency(
        self,
        size_bytes: int,
        nccl_group: ProcessGroup,
        stream: torch.cuda.Stream,
        warmup: int,
        iters: int,
    ) -> tuple[float, float]:
        if self.exchange_group is None:
            raise ValueError("exchange_group is required for graph-based autotuning")

        numel = size_bytes // torch.tensor([], dtype=torch.bfloat16).element_size()
        device = self.device

        def run_custom() -> float:
            with torch.cuda.stream(stream):
                graph_inp = torch.ones(numel, dtype=torch.bfloat16, device=device)
                graph_out = torch.zeros_like(graph_inp)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._ext.all_reduce(self._ptr, graph_inp, graph_out, 0, 0)
            self.register_graph_buffers()
            dist.barrier(group=nccl_group)
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    graph.replay()
            stream.synchronize()
            start = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(iters):
                    graph.replay()
            stream.synchronize()
            return (time.perf_counter() - start) / iters * 1e6

        def run_nccl() -> float:
            with torch.cuda.stream(stream):
                graph_inp = torch.ones(numel, dtype=torch.bfloat16, device=device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                dist.all_reduce(graph_inp, group=nccl_group)
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    graph.replay()
            stream.synchronize()
            start = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(iters):
                    graph.replay()
            stream.synchronize()
            return (time.perf_counter() - start) / iters * 1e6

        custom_runs = sorted(run_custom() for _ in range(3))
        nccl_runs = sorted(run_nccl() for _ in range(3))
        return custom_runs[1], nccl_runs[1]

    def find_crossover_size(
        self,
        nccl_group: ProcessGroup,
        *,
        ceiling_bytes: int = AUTOTUNE_CEILING,
        fine_step_bytes: int = AUTOTUNE_FINE_STEP,
        warmup: int = 100,
        iters: int = 1000,
    ) -> int:
        if self.device.type != "cuda":
            raise ValueError("autotune requires a CUDA device")
        bench_stream = torch.cuda.Stream(device=self.device)
        crossover, results = _compute_crossover_size(
            lambda size_bytes: self._bench_graph_latency(
                size_bytes,
                nccl_group,
                bench_stream,
                warmup,
                iters,
            ),
            ceiling_bytes=ceiling_bytes,
            fine_step_bytes=fine_step_bytes,
        )
        self.max_size = crossover

        if self.rank == 0:
            def fmt_size(size_bytes: int) -> str:
                if size_bytes >= 1024 * 1024:
                    return f"{size_bytes // (1024 * 1024)}MB"
                if size_bytes >= 1024:
                    return f"{size_bytes // 1024}KB"
                return f"{size_bytes}B"

            lines = [f"[PCIe oneshot allreduce] Crossover benchmark ({self.world_size} GPUs, bf16):"]
            for result in results:
                lines.append(
                    f"  {fmt_size(result.size_bytes):>6s}:  custom {result.custom_us:6.1f} us  "
                    f"vs  NCCL {result.nccl_us:6.1f} us  -> {result.winner} wins"
                )
            lines.append(
                f"  Setting max_size = {fmt_size(crossover)} (last size where custom AR wins)"
            )
            logger.info("\n".join(lines))
        return crossover

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if getattr(self, "_ptr", 0):
            self._ext.dispose(self._ptr)
            self._ptr = 0
        for shared in self._owned_buffers:
            for ptr in shared.remote_ptrs:
                if self._ipc is not None:
                    self._ipc.cudaIpcCloseMemHandle(ptr)
            if self._ipc is not None:
                self._ipc.cudaFree(shared.local_ptr)
        self._owned_buffers.clear()
        self._registered_input_ptrs.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "PCIeOneshotAllReduce",
    "SUPPORTED_WORLD_SIZES",
    "parse_pcie_oneshot_max_size",
]
