from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_dma import PCIeDmaAllReduce


pytestmark = pytest.mark.skipif(
    os.getenv("SPARKINFER_RUN_PCIE_DMA_TEST") != "1",
    reason="set SPARKINFER_RUN_PCIE_DMA_TEST=1 to run PCIe ring allreduce GPU tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_input(
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    iteration: int,
) -> torch.Tensor:
    values = torch.arange(rows * hidden, device=device, dtype=torch.float32)
    values = values.reshape(rows, hidden)
    return (torch.sin(values * 0.001 + rank * 0.31 + iteration * 0.17) * 0.5).to(dtype)


def _reference(inp: torch.Tensor) -> torch.Tensor:
    # clone: .float() aliases the input for fp32 and all_reduce is in-place.
    ref = inp.detach().clone().float()
    dist.all_reduce(ref)
    return ref


def _assert_close(actual: torch.Tensor, ref: torch.Tensor, world_size: int) -> None:
    # Stepwise low-precision ring adds; allow world_size half-ulps around the
    # fp32 reference. Compressed wire modes need a wider band because the ring
    # can requantize partial sums at each reduce-scatter hop.
    if os.getenv("SPARKINFER_PCIE_DMA_FP8", "0") not in ("", "0"):
        torch.testing.assert_close(
            actual.float(), ref, rtol=1.5e-1, atol=6e-2 * world_size
        )
    elif actual.dtype == torch.float32:
        torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-5 * world_size)
    else:
        torch.testing.assert_close(
            actual.float(), ref, rtol=3e-2, atol=3e-2 * world_size
        )


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    hidden = 6144
    max_rows = 512
    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_bytes=max_rows * hidden * 4,
    )
    try:
        for dtype in (torch.bfloat16,):
            for rows in (8, 64, 256):
                inp = _make_input(rows, hidden, dtype, device, rank, 0)
                ref = _reference(inp)
                out = ring.all_reduce(inp)
                torch.cuda.synchronize(device)
                _assert_close(out, ref, world_size)

        # Graph capture and replay with changing inputs.
        rows = 256
        dtype = torch.bfloat16
        inp = _make_input(rows, hidden, dtype, device, rank, 0)
        out = torch.empty_like(inp)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ring.all_reduce(inp, out=out)
        for iteration in range(1, 4):
            inp.copy_(_make_input(rows, hidden, dtype, device, rank, iteration))
            ref = _reference(inp)
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_close(out, ref, world_size)

        dist.barrier()
    finally:
        ring.close()
        dist.destroy_process_group()


def test_pcie_dma_all_reduce_eager_and_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("SPARKINFER_PCIE_DMA_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


def _fp8_worker(rank: int, world_size: int, port: int, mode: str) -> None:
    os.environ["SPARKINFER_PCIE_DMA_FP8"] = mode
    _worker(rank, world_size, port)


@pytest.mark.parametrize(
    "mode", ["ag", "a2a", "ring", "i8", "i8_a2a", "i8_ring"]
)
def test_pcie_dma_all_reduce_compressed_wire(mode: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("SPARKINFER_PCIE_DMA_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _fp8_worker,
        args=(world_size, _free_port(), mode),
        nprocs=world_size,
        join=True,
    )
