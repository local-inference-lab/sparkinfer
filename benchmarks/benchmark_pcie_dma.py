"""CE ring allreduce vs NCCL at prefill sizes: parity + graph latency."""
from __future__ import annotations

import os
import socket
from statistics import median

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_PROTO", "LL,LL128,Simple")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.distributed.pcie_dma import PCIeDmaAllReduce


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _time_graphs(
    graphs: dict[str, torch.cuda.CUDAGraph],
    device: torch.device,
    *,
    warmup: int = 10,
    iters: int = 100,
    samples: int = 9,
) -> dict[str, list[float]]:
    for graph in graphs.values():
        for _ in range(warmup):
            graph.replay()
    torch.cuda.synchronize(device)
    timings = {name: [] for name in graphs}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    rank_max = torch.empty((), dtype=torch.float64, device=device)
    names = list(graphs)
    for sample in range(samples):
        order = names if sample % 2 == 0 else list(reversed(names))
        for name in order:
            dist.barrier(device_ids=[device.index])
            start.record()
            for _ in range(iters):
                graphs[name].replay()
            end.record()
            end.synchronize()
            rank_max.fill_(start.elapsed_time(end) * 1e3 / iters)
            dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
            timings[name].append(float(rank_max.item()))
    return timings


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    max_bytes = 8192 * 6144 * 2
    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD, device=device, max_bytes=max_bytes
    )
    try:
        if rank == 0:
            print(
                "rows,bytes,nccl_median_us,dma_median_us,nccl_min_us,"
                "nccl_max_us,dma_min_us,dma_max_us,nccl_over_dma,"
                "maxerr_ratio",
                flush=True,
            )
        detailed_rows = (256, 288) + tuple(range(320, 1056, 32)) + (1152, 1280)
        for rows in detailed_rows:
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

            timings = _time_graphs(
                {"nccl": nccl_graph, "dma": ring_graph},
                device,
            )
            nccl_us = float(median(timings["nccl"]))
            dma_us = float(median(timings["dma"]))
            size = inp.numel() * 2
            if rank == 0:
                print(
                    f"{rows},{size},{nccl_us:.1f},{dma_us:.1f},"
                    f"{min(timings['nccl']):.1f},{max(timings['nccl']):.1f},"
                    f"{min(timings['dma']):.1f},{max(timings['dma']):.1f},"
                    f"{nccl_us / dma_us:.4f},{err_ratio:.2f}",
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
