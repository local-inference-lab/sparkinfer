from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool


TOTAL_HEADS = 32
HEAD_DIM = 512
QUERY_HEAD_DIM = 576
MAX_BATCH = 8


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _capture(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _measure(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
    *,
    warmup: int = 200,
    iterations: int = 2_000,
) -> float:
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize(device)
    latency = (time.perf_counter() - start) * 1e6 / iterations
    rank_latency = torch.tensor(latency, dtype=torch.float64, device=device)
    dist.all_reduce(rank_latency, op=dist.ReduceOp.MAX)
    return float(rank_latency.item())


def _median_latency(graph: torch.cuda.CUDAGraph, device: torch.device) -> float:
    return sorted(_measure(graph, device) for _ in range(3))[1]


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    dtype = torch.bfloat16
    heads_per_rank = TOTAL_HEADS // world_size
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=MAX_BATCH,
        total_heads=TOTAL_HEADS,
        head_dim=HEAD_DIM,
        query_head_dim=QUERY_HEAD_DIM,
    )
    pool.for_stream()
    try:
        if rank == 0:
            print("batch,reduce_in_bytes,gather_in_bytes,reduce_us,gather_us")
        for batch in (1, 2, 4, 8):
            reduce_out = torch.randn(
                batch, TOTAL_HEADS, HEAD_DIM, dtype=dtype, device=device
            )
            reduce_lse = torch.randn(
                batch, TOTAL_HEADS, dtype=torch.float32, device=device
            )
            reduce_result = torch.empty(
                batch, heads_per_rank, HEAD_DIM, dtype=dtype, device=device
            )
            reduce_graph = _capture(
                lambda: pool.lse_reduce_scatter(
                    reduce_out, reduce_lse, reduce_result
                )
            )

            gather_in = torch.randn(
                batch, heads_per_rank, QUERY_HEAD_DIM, dtype=dtype, device=device
            )
            gather_out = torch.empty(
                batch, TOTAL_HEADS, QUERY_HEAD_DIM, dtype=dtype, device=device
            )
            gather_graph = _capture(
                lambda: pool.all_gather_heads(gather_in, gather_out)
            )

            reduce_us = _median_latency(reduce_graph, device)
            gather_us = _median_latency(gather_graph, device)
            if rank == 0:
                reduce_bytes = reduce_out.numel() * dtype.itemsize
                gather_bytes = gather_in.numel() * dtype.itemsize
                print(
                    f"{batch},{reduce_bytes},{gather_bytes},"
                    f"{reduce_us:.3f},{gather_us:.3f}",
                    flush=True,
                )
            del reduce_graph, gather_graph
            torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    world_size = int(os.getenv("SPARKINFER_PCIE_DCP_A2A_WORLD_SIZE", "8"))
    if torch.cuda.device_count() < world_size:
        raise SystemExit(
            f"need {world_size} GPUs, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
