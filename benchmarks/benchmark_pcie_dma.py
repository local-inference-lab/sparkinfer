"""CE ring allreduce vs NCCL at prefill sizes: parity + graph latency."""
from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.distributed.pcie_dma import PCIeDmaAllReduce


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _time_graph(graph, device, warmup=5, iters=30) -> float:
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize(device)
    us = (time.perf_counter() - start) * 1e6 / iters
    lat = torch.tensor(us, dtype=torch.float64, device=device)
    dist.all_reduce(lat, op=dist.ReduceOp.MAX)
    return float(lat.item())


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    max_bytes = 8192 * 6144 * 2
    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD, device=device, max_bytes=max_bytes
    )
    try:
        if rank == 0:
            print(
                "rows,bytes,nccl_us,dma_us,nccl_bus_GBps,ring_bus_GBps,"
                "maxerr_ratio",
                flush=True,
            )
        for rows in (256, 1024, 4096, 8192):
            gen = torch.Generator(device=device).manual_seed(1234 + rank + rows)
            inp = torch.randn(
                rows, 6144, dtype=torch.float32, device=device, generator=gen
            ).to(torch.bfloat16)

            ref = inp.float()
            dist.all_reduce(ref)

            nccl_in = inp.clone()
            dist.all_reduce(nccl_in)
            nccl_err = (nccl_in.float() - ref).abs().max()

            ring_out = ring.all_reduce(inp)
            torch.cuda.synchronize(device)
            ring_err = (ring_out.float() - ref).abs().max()
            err_ratio = float((ring_err / nccl_err.clamp_min(1e-9)).item())

            nccl_buf = inp.clone()
            nccl_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(nccl_graph):
                dist.all_reduce(nccl_buf)

            ring_in = inp.clone()
            ring_res = torch.empty_like(ring_in)
            ring_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(ring_graph):
                ring.all_reduce(ring_in, out=ring_res)

            nccl_us = _time_graph(nccl_graph, device)
            dma_us = _time_graph(ring_graph, device)
            size = inp.numel() * 2
            bus = 2 * (world_size - 1) / world_size * size
            if rank == 0:
                print(
                    f"{rows},{size},{nccl_us:.1f},{dma_us:.1f},"
                    f"{bus / (nccl_us * 1e-6) / 1e9:.1f},"
                    f"{bus / (dma_us * 1e-6) / 1e9:.1f},{err_ratio:.2f}",
                    flush=True,
                )
            del nccl_graph, ring_graph
            torch.cuda.synchronize(device)
    finally:
        ring.close()
        dist.destroy_process_group()


def main() -> None:
    world_size = 8
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
