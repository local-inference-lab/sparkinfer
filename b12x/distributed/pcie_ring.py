"""CE-driven PCIe ring allreduce for prefill-size tensors.

NCCL's SM-copy transport sustains ~34 GB/s bus bandwidth on this fabric
while CE peer copies run at ~56 GB/s on every ring hop concurrently
(including the two root-complex crossings, which each own a partition
uplink per direction). This runtime drives a classic reduce-scatter +
all-gather ring where the data plane is CE copies and the SM only
synchronizes (monotonic flag kernels) and reduces, so captured graphs
replay without host patching.
"""

from __future__ import annotations

import os
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import (
    PCIeOneshotAllReduce,
    _broadcast_gather_object,
    _OwnedSharedBuffer,
)

SUPPORTED_DTYPES = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}
FLAG_STRIDE = 128
FLAG_SLOTS = 40
SCRATCH_ALIGN = 256


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_ring.cu")
    verbose = os.getenv("B12X_PCIE_RING_VERBOSE_BUILD", "0") == "1"
    return load(
        name="b12x_pcie_ring_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O2"],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class PCIeRingAllReduce:
    """Single-channel ring allreduce over IPC scratch buffers.

    A channel is a single ordered stream context; concurrent use from
    multiple CUDA streams needs separate channels (same contract as the
    oneshot runtime).
    """

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_bytes: int,
        ext_module=None,
    ) -> None:
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = (
            device
            if isinstance(device, torch.device)
            else torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        )
        if self.device.type != "cuda":
            raise ValueError("PCIe ring allreduce requires a CUDA device")
        if self.world_size < 2:
            raise ValueError("ring allreduce needs at least 2 ranks")
        self.max_bytes = int(max_bytes)
        self._ext = ext_module or _load_extension()
        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._closed = False

        self.shard_capacity = _align_up(
            (self.max_bytes + self.world_size - 1) // self.world_size, SCRATCH_ALIGN
        )
        steps = 2 * (self.world_size - 1)
        flags_bytes = FLAG_SLOTS * FLAG_STRIDE
        slab_bytes = flags_bytes + steps * self.shard_capacity
        self._slab = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group, slab_bytes, zero_fill=True, ipc=self._ipc
        )
        self._flags_base = list(self._slab.peer_ptrs)
        self._scratch_base = [ptr + flags_bytes for ptr in self._slab.peer_ptrs]
        # Device-resident monotonic counters: one per flag slot for the
        # publisher role and one for the waiter role.
        self._send_counters = torch.zeros(
            FLAG_SLOTS, dtype=torch.int32, device=self.device
        )
        self._wait_counters = torch.zeros(
            FLAG_SLOTS, dtype=torch.int32, device=self.device
        )
        self._copy_stream = torch.cuda.Stream(device=self.device)

    def _flag_ptr(self, rank: int, slot: int) -> int:
        return self._flags_base[rank] + slot * FLAG_STRIDE

    def _counter_ptr(self, counters: torch.Tensor, slot: int) -> int:
        return counters.data_ptr() + slot * 4

    def _scratch_ptr(self, rank: int, step: int) -> int:
        return self._scratch_base[rank] + step * self.shard_capacity

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._closed or inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        numel = inp.numel()
        if numel <= 0 or numel % (self.world_size * 8) != 0:
            return False
        return (
            inp.is_contiguous()
            and inp.numel() * inp.element_size() <= self.max_bytes
        )

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy ring allreduce requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if out is None:
            out = torch.empty_like(inp)
        elif out.shape != inp.shape or out.dtype != inp.dtype or not out.is_contiguous():
            raise ValueError("output must match input shape/dtype and be contiguous")
        ext = self._ext
        world = self.world_size
        rank = self.rank
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        dtype_code = SUPPORTED_DTYPES[inp.dtype]
        elem = inp.element_size()
        shard_elems = inp.numel() // world
        shard_bytes = shard_elems * elem
        base = out.data_ptr()

        # Sub-chunking with a dedicated copy stream keeps the copy engine
        # busy: the CE never waits for a flag round trip or an add because
        # sub-chunk c+1's copy overlaps sub-chunk c's wait+reduce.
        pieces = 2 if shard_elems % (2 * 8) == 0 and shard_bytes >= 2 << 20 else 1
        piece_elems = shard_elems // pieces
        piece_bytes = piece_elems * elem
        steps = 2 * (world - 1)

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        copy_stream.wait_stream(main)

        out.copy_(inp)
        ready = torch.cuda.Event()
        ready.record(main)
        copy_stream.wait_event(ready)

        def piece_ptr(chunk: int, piece: int) -> int:
            return base + chunk * shard_bytes + piece * piece_bytes

        def scratch_piece(owner: int, step: int, piece: int) -> int:
            return self._scratch_ptr(owner, step) + piece * piece_bytes

        def slot(step: int, piece: int) -> int:
            return step * pieces + piece

        # Events gating each step's send on the previous step's reduce of
        # the same payload piece.
        add_done: dict[int, torch.cuda.Event] = {}

        for k in range(steps):
            reduce_phase = k < world - 1
            if reduce_phase:
                send_chunk = (rank - k) % world
                recv_chunk = (rank - k - 1) % world
            else:
                send_chunk = (rank + 1 - (k - (world - 1))) % world
                recv_chunk = (rank - (k - (world - 1))) % world
            for p in range(pieces):
                with torch.cuda.stream(copy_stream):
                    if k > 0:
                        copy_stream.wait_event(add_done[p])
                    ext.ring_copy(
                        scratch_piece(nxt, k, p),
                        piece_ptr(send_chunk, p),
                        piece_bytes,
                    )
                    ext.ring_set_flag(
                        self._flag_ptr(nxt, slot(k, p)),
                        self._counter_ptr(self._send_counters, slot(k, p)),
                    )
                ext.ring_wait_flag(
                    self._flag_ptr(rank, slot(k, p)),
                    self._counter_ptr(self._wait_counters, slot(k, p)),
                )
                if reduce_phase:
                    ext.ring_add(
                        piece_ptr(recv_chunk, p),
                        scratch_piece(rank, k, p),
                        piece_elems,
                        dtype_code,
                    )
                else:
                    ext.ring_copy(
                        piece_ptr(recv_chunk, p),
                        scratch_piece(rank, k, p),
                        piece_bytes,
                    )
                event = torch.cuda.Event()
                event.record(main)
                add_done[p] = event

        # Neighbor handshake so the next call (or graph replay) cannot
        # overwrite scratch a lagging neighbor still reads. The main stream
        # must also drain the copy stream before the op is considered done.
        main.wait_stream(copy_stream)
        done = steps * pieces
        ext.ring_set_flag(
            self._flag_ptr(prv, done), self._counter_ptr(self._send_counters, done)
        )
        ext.ring_wait_flag(
            self._flag_ptr(rank, done), self._counter_ptr(self._wait_counters, done)
        )
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for ptr in self._slab.remote_ptrs:
            with suppress(Exception):
                self._ipc.cudaIpcCloseMemHandle(ptr)
        with suppress(Exception):
            self._ipc.cudaFree(self._slab.local_ptr)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = ["PCIeRingAllReduce"]
