from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.distributed.pcie_oneshot import PCIeOneshotAllReducePool
from vllm import _custom_ops as ops


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _capture(fn: Callable[[], None], pool=None) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    if pool is None:
        with torch.cuda.graph(graph):
            fn()
    else:
        with pool.capture(), torch.cuda.graph(graph):
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


def _median_latency(
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
) -> float:
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
    hidden_size = 6144
    dtype = torch.bfloat16
    epsilon = 1e-6
    max_bytes = 128 * 1024
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=max_bytes,
        max_size=max_bytes,
    )
    pool.for_stream()
    try:
        if rank == 0:
            print(
                "rows,bytes,b12x_fused_us,b12x_plus_rms_us,"
                "nccl_plus_rms_us"
            )
        for rows in (1, 2, 3, 4, 5, 6, 7, 8):
            shape = (rows, hidden_size)
            weight = torch.ones(hidden_size, dtype=dtype, device=device)

            fused_in = torch.randn(shape, dtype=dtype, device=device) * 0.01
            fused_residual = torch.randn(shape, dtype=dtype, device=device)
            fused_out = torch.empty_like(fused_in)
            fused_residual_out = torch.empty_like(fused_in)
            fused_graph = _capture(
                lambda: pool.all_reduce_fused_add_rms_norm(
                    fused_in,
                    fused_residual,
                    weight,
                    epsilon,
                    out=fused_out,
                    residual_out=fused_residual_out,
                ),
                pool,
            )

            bare_in = torch.randn(shape, dtype=dtype, device=device) * 0.01
            bare_residual = torch.randn(shape, dtype=dtype, device=device)
            bare_out = torch.empty_like(bare_in)

            def bare_then_rms() -> None:
                pool.all_reduce(bare_in, out=bare_out)
                ops.fused_add_rms_norm(
                    bare_out,
                    bare_residual,
                    weight,
                    epsilon,
                )

            bare_graph = _capture(bare_then_rms, pool)

            nccl_in = torch.randn(shape, dtype=dtype, device=device) * 0.01
            nccl_residual = torch.randn(shape, dtype=dtype, device=device)

            def nccl_then_rms() -> None:
                dist.all_reduce(nccl_in)
                ops.fused_add_rms_norm(
                    nccl_in,
                    nccl_residual,
                    weight,
                    epsilon,
                )

            nccl_graph = _capture(nccl_then_rms)
            fused_us = _median_latency(fused_graph, device)
            bare_us = _median_latency(bare_graph, device)
            nccl_us = _median_latency(nccl_graph, device)
            if rank == 0:
                size_bytes = rows * hidden_size * dtype.itemsize
                print(
                    f"{rows},{size_bytes},{fused_us:.3f},"
                    f"{bare_us:.3f},{nccl_us:.3f}",
                    flush=True,
                )
            del fused_graph, bare_graph, nccl_graph
            torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    world_size = int(os.getenv("B12X_PCIE_ONESHOT_WORLD_SIZE", "8"))
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
