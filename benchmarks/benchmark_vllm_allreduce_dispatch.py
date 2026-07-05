"""Validate vLLM CustomAllreduce b12x wiring: autotuned crossovers + routing."""
from __future__ import annotations

import os
import socket
import time

# Match the serving environment's NCCL configuration: on this PCIe-only
# topology, default NCCL refuses P2P across the root complex and falls
# back to shared-memory transport, which understates NCCL badly.
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_PROTO", "LL,LL128,Simple")
os.environ.setdefault("VLLM_ENABLE_PCIE_ALLREDUCE", "1")
os.environ.setdefault("VLLM_PCIE_ALLREDUCE_BACKEND", "b12x")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    gloo = dist.new_group(backend="gloo")

    import logging

    if rank == 0:
        logging.basicConfig(level=logging.INFO)

    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    start = time.perf_counter()
    ca = CustomAllreduce(group=gloo, device=device, nccl_group=dist.group.WORLD)
    init_s = time.perf_counter() - start
    assert not ca.disabled
    oneshot_max = ca._pcie_allreduce_max_size
    dma = ca._pcie_dma
    dma_min = dma.min_bytes if dma is not None else None
    if rank == 0:
        print(
            f"[e2e] backend={ca.backend_name()} init={init_s:.1f}s "
            f"oneshot_max={oneshot_max} dma_min={dma_min}",
            flush=True,
        )

    hidden = 6144
    for rows in (1, 2, 3, 4, 5, 6, 7, 8, 12, 32, 128, 512, 2048, 4096, 8192):
        inp = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
        size = inp.numel() * inp.element_size()
        routed = ca.should_custom_ar(inp)
        expect = size <= oneshot_max or (
            dma is not None and dma.should_allreduce(inp)
        )
        assert routed == expect, f"rows={rows}: routed={routed} expect={expect}"
        which = (
            "oneshot" if size <= oneshot_max
            else "dma" if routed
            else "nccl"
        )
        if routed:
            ca.custom_all_reduce(inp)
        if rank == 0:
            print(f"[e2e] rows={rows:5d} ({size >> 10}KB) -> {which}", flush=True)

    # Real decode path: fused AR+RMSNorm.
    inp = torch.randn(4, hidden, dtype=torch.bfloat16, device=device) * 0.01
    residual = torch.randn(4, hidden, dtype=torch.bfloat16, device=device)
    weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    fused = ca.try_fused_add_rms_norm(inp, residual, weight, 1e-6)
    assert fused == ((4 * hidden * 2) <= ca._pcie_fused_add_rms_norm_max_size)
    if rank == 0:
        print(f"[e2e] fused_add_rms_norm dispatched={fused}", flush=True)

    # Real prefill path under graph capture.
    inp = torch.randn(4096, hidden, dtype=torch.bfloat16, device=device)
    graph = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(graph):
            out = ca.custom_all_reduce(inp)
    assert out is not None
    dist.barrier()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(30):
        graph.replay()
    torch.cuda.synchronize(device)
    us = (time.perf_counter() - t0) * 1e6 / 30
    lat = torch.tensor(us, dtype=torch.float64, device=device)
    dist.all_reduce(lat, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"[e2e] 4096x6144 captured replay: {float(lat.item()):.1f} us", flush=True)
        print("[e2e] PASS", flush=True)
    ca.close()
    dist.destroy_process_group()


def main() -> None:
    mp.spawn(_worker, args=(8, _free_port()), nprocs=8, join=True)


if __name__ == "__main__":
    main()
