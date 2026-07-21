from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_dcp_a2a import (
    PCIeDCPA2APool,
    _load_extension,
    lse_reduce_scatter_reference,
)


pytestmark = pytest.mark.skipif(
    os.getenv("SPARKINFER_RUN_PCIE_DCP_A2A_TEST") != "1",
    reason="set SPARKINFER_RUN_PCIE_DCP_A2A_TEST=1 to run PCIe DCP A2A GPU tests",
)

TOTAL_HEADS = 16
HEAD_DIM = 512
QUERY_HEAD_DIM = 576
MAX_BATCH = 64
TEST_BATCHES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_inputs(
    step: int,
    source_rank: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(10000 * step + source_rank)
    output = torch.randn(
        batch,
        TOTAL_HEADS,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    lse = torch.randn(
        batch,
        TOTAL_HEADS,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    if batch > 0:
        lse[0, 0] = -torch.inf
        lse[0, 1] = torch.nan
        if source_rank == 0:
            lse[0, 2] = -torch.inf
    return output, lse


def _rank_query(
    step: int,
    source_rank: int,
    world_size: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20000 * step + source_rank)
    return torch.randn(
        batch,
        TOTAL_HEADS // world_size,
        QUERY_HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)


def _reference(
    step: int,
    rank: int,
    world_size: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    inputs = [
        _rank_inputs(step, source_rank, batch, dtype, device)
        for source_rank in range(world_size)
    ]
    return lse_reduce_scatter_reference(
        torch.stack([item[0] for item in inputs]),
        torch.stack([item[1] for item in inputs]),
        rank,
    )


def _check_eager(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    for dtype in (torch.bfloat16, torch.float16):
        for step, batch in enumerate(TEST_BATCHES, start=1):
            local_q = _rank_query(
                step + 100,
                rank,
                world_size,
                batch,
                dtype,
                device,
            )
            gathered_q = pool.all_gather_heads(local_q)
            expected_q = torch.cat(
                [
                    _rank_query(
                        step + 100,
                        source,
                        world_size,
                        batch,
                        dtype,
                        device,
                    )
                    for source in range(world_size)
                ],
                dim=1,
            )
            torch.testing.assert_close(gathered_q, expected_q, rtol=0, atol=0)

            partial_output, partial_lse = _rank_inputs(step, rank, batch, dtype, device)
            out = pool.lse_reduce_scatter(partial_output, partial_lse)
            torch.cuda.synchronize(device)
            expected = _reference(
                step,
                rank,
                world_size,
                batch,
                dtype,
                device,
            )
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

            input_storage = torch.empty(
                TOTAL_HEADS,
                MAX_BATCH,
                HEAD_DIM,
                dtype=dtype,
                device=device,
            )
            head_major_input = input_storage.transpose(0, 1)[:batch]
            head_major_input.copy_(partial_output)
            output_storage = torch.empty(
                TOTAL_HEADS // world_size,
                MAX_BATCH,
                HEAD_DIM,
                dtype=dtype,
                device=device,
            )
            head_major_output = output_storage.transpose(0, 1)[:batch]
            actual = pool.lse_reduce_scatter(
                head_major_input,
                partial_lse,
                out=head_major_output,
            )
            torch.cuda.synchronize(device)
            assert actual is head_major_output
            assert actual.movedim(0, 1).stride(0) == MAX_BATCH * HEAD_DIM
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _check_graph(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    stream = torch.cuda.Stream(device=device)
    channel = pool.for_stream(stream)
    layers = 7
    input_storages = [
        torch.empty(
            TOTAL_HEADS,
            MAX_BATCH,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(layers)
    ]
    inputs = [storage.transpose(0, 1) for storage in input_storages]
    lses = [
        torch.empty(MAX_BATCH, TOTAL_HEADS, dtype=torch.float32, device=device)
        for _ in range(layers)
    ]
    output_storages = [
        torch.empty(
            TOTAL_HEADS // world_size,
            MAX_BATCH,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(layers)
    ]
    outputs = [storage.transpose(0, 1) for storage in output_storages]
    assert all(tensor.stride(1) == MAX_BATCH * HEAD_DIM for tensor in inputs)
    assert all(tensor.stride(1) == MAX_BATCH * HEAD_DIM for tensor in outputs)
    local_queries = [
        torch.empty(
            1,
            TOTAL_HEADS // world_size,
            QUERY_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(layers)
    ]
    gathered_queries = [
        torch.empty(
            1,
            TOTAL_HEADS,
            QUERY_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(layers)
    ]

    with torch.cuda.stream(stream):
        channel.all_gather_heads(local_queries[0], gathered_queries[0])
        channel.lse_reduce_scatter(inputs[0], lses[0], outputs[0])
    stream.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for layer in range(layers):
            channel.all_gather_heads(local_queries[layer], gathered_queries[layer])
            channel.lse_reduce_scatter(inputs[layer], lses[layer], outputs[layer])
    stream.synchronize()

    for replay in range(8):
        expected = []
        expected_queries = []
        for layer in range(layers):
            step = 1000 * replay + layer + 100
            partial_output, partial_lse = _rank_inputs(
                step,
                rank,
                MAX_BATCH,
                torch.bfloat16,
                device,
            )
            inputs[layer].copy_(partial_output)
            lses[layer].copy_(partial_lse)
            local_queries[layer].copy_(
                _rank_query(
                    step,
                    rank,
                    world_size,
                    1,
                    torch.bfloat16,
                    device,
                )
            )
            expected_queries.append(
                torch.cat(
                    [
                        _rank_query(
                            step,
                            source,
                            world_size,
                            1,
                            torch.bfloat16,
                            device,
                        )
                        for source in range(world_size)
                    ],
                    dim=1,
                )
            )
            expected.append(
                _reference(
                    step,
                    rank,
                    world_size,
                    MAX_BATCH,
                    torch.bfloat16,
                    device,
                )
            )
        stream.wait_stream(torch.cuda.current_stream(device))
        graph.replay()
        stream.synchronize()
        for out, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(out, reference, rtol=2e-2, atol=2e-2)
        for out, reference in zip(gathered_queries, expected_queries, strict=True):
            torch.testing.assert_close(out, reference, rtol=0, atol=0)


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=MAX_BATCH,
        total_heads=TOTAL_HEADS,
        head_dim=HEAD_DIM,
        query_head_dim=QUERY_HEAD_DIM,
    )
    try:
        _check_eager(pool, rank, world_size, device)
        dist.barrier()
        _check_graph(pool, rank, world_size, device)
        torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def test_pcie_dcp_a2a_eager_and_cuda_graph_correctness():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    world_size = int(os.getenv("SPARKINFER_PCIE_DCP_A2A_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 8):
        pytest.skip("PCIe DCP A2A supports world sizes 2, 4, and 8")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    _load_extension()
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
