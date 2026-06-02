from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.distributed.pcie_oneshot import PCIeOneshotAllReducePool


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_ONESHOT_TORTURE") != "1",
    reason="set B12X_RUN_PCIE_ONESHOT_TORTURE=1 to run PCIe oneshot CUDA torture tests",
)

TORTURE_EAGER_ITERS = int(os.getenv("B12X_PCIE_ONESHOT_TORTURE_EAGER_ITERS", "256"))
TORTURE_GRAPH_REPLAYS = int(os.getenv("B12X_PCIE_ONESHOT_TORTURE_GRAPH_REPLAYS", "256"))
TORTURE_MULTISTREAM_ITERS = int(os.getenv("B12X_PCIE_ONESHOT_TORTURE_MULTISTREAM_ITERS", "256"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_constant(tensor: torch.Tensor, value: float) -> None:
    expected = torch.full_like(tensor, value)
    torch.testing.assert_close(tensor, expected, rtol=1e-2, atol=1e-2)


def _rank_sum(world_size: int) -> int:
    return world_size * (world_size - 1) // 2


def _run_eager(pool: PCIeOneshotAllReducePool, device: torch.device, rank: int, world_size: int) -> None:
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    numels = (8, 256, 4096, 32768)
    rank_sum = _rank_sum(world_size)

    for dtype in dtypes:
        for numel in numels:
            inp = torch.empty(numel, device=device, dtype=dtype)
            out = torch.empty_like(inp)
            for iteration in range(TORTURE_EAGER_ITERS):
                base = float((iteration % 64) * 3)
                inp.fill_(base + rank)
                pool.all_reduce(inp, out=out)
                torch.cuda.synchronize(device)
                _assert_constant(out, world_size * base + rank_sum)


def _run_graph_scratch_reuse(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    stream = torch.cuda.Stream(device=device)
    channel = pool.for_stream(stream)
    rank_sum = _rank_sum(world_size)
    layers = 17
    numel = 4096
    dtype = torch.bfloat16
    sources = [torch.empty(numel, device=device, dtype=dtype) for _ in range(layers)]
    scratch = torch.empty(numel, device=device, dtype=dtype)
    outs = [torch.empty(numel, device=device, dtype=dtype) for _ in range(layers)]

    def fill_sources(iteration: int) -> None:
        for layer, source in enumerate(sources):
            source.fill_(float((iteration % 32) * 5 + layer) + rank)

    fill_sources(0)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with pool.capture(stream), torch.cuda.graph(graph, stream=stream):
        for layer in range(layers):
            scratch.copy_(sources[layer])
            channel.all_reduce(scratch, out=outs[layer])
    stream.synchronize()

    for iteration in range(TORTURE_GRAPH_REPLAYS):
        fill_sources(iteration)
        graph.replay()
        stream.synchronize()
        for layer, out in enumerate(outs):
            base = float((iteration % 32) * 5 + layer)
            _assert_constant(out, world_size * base + rank_sum)


def _run_multistream(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    pool.for_stream(stream_a)
    pool.for_stream(stream_b)
    rank_sum = _rank_sum(world_size)

    inp_a = torch.empty(2048, device=device, dtype=torch.float16)
    out_a = torch.empty_like(inp_a)
    inp_b = torch.empty(2048, device=device, dtype=torch.bfloat16)
    out_b = torch.empty_like(inp_b)

    for iteration in range(TORTURE_MULTISTREAM_ITERS):
        base_a = float(iteration % 64)
        base_b = float(100 + (iteration % 64) * 2)
        with torch.cuda.stream(stream_a):
            inp_a.fill_(base_a + rank)
            pool.all_reduce(inp_a, out=out_a)
        with torch.cuda.stream(stream_b):
            inp_b.fill_(base_b + rank)
            pool.all_reduce(inp_b, out=out_b)
        stream_a.synchronize()
        stream_b.synchronize()
        _assert_constant(out_a, world_size * base_a + rank_sum)
        _assert_constant(out_b, world_size * base_b + rank_sum)


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=1 << 20,
    )
    try:
        _run_eager(pool, device, rank, world_size)
        dist.barrier()
        _run_graph_scratch_reuse(pool, device, rank, world_size)
        dist.barrier()
        _run_multistream(pool, device, rank, world_size)
        torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def test_pcie_oneshot_eager_graph_and_multistream_torture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    available = torch.cuda.device_count()
    requested = int(os.getenv("B12X_PCIE_ONESHOT_TORTURE_WORLD_SIZE", "2"))
    if requested not in (2, 4, 6, 8, 10):
        pytest.skip("PCIe oneshot only supports world sizes 2, 4, 6, 8, and 10")
    if available < requested:
        pytest.skip(f"need {requested} CUDA devices, found {available}")
    mp.spawn(_worker, args=(requested, _free_port()), nprocs=requested, join=True)
