from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cuda.bindings import runtime as cudart

from sparkinfer.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


pytestmark = pytest.mark.skipif(
    os.getenv("SPARKINFER_RUN_PCIE_ONESHOT_RMS_TEST") != "1",
    reason=(
        "set SPARKINFER_RUN_PCIE_ONESHOT_RMS_TEST=1 to run PCIe oneshot fused "
        "RMSNorm GPU tests"
    ),
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reference(
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = inp.clone()
    dist.all_reduce(reduced)
    residual_out = reduced + residual
    variance = residual_out.float().square().mean(dim=-1, keepdim=True)
    out = residual_out.float() * torch.rsqrt(variance + epsilon)
    out = (out * weight.float()).to(inp.dtype)
    return out, residual_out


def _make_inputs(
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    iteration: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.arange(rows * hidden_size, device=device, dtype=torch.float32)
    values = values.reshape(rows, hidden_size)
    inp = (torch.sin(values * 0.013 + rank * 0.17 + iteration * 0.11) * 0.25).to(dtype)
    residual = (torch.cos(values * 0.007 + rank * 0.03 + iteration * 0.19) * 0.5).to(
        dtype
    )
    weight = torch.linspace(0.5, 1.5, hidden_size, device=device, dtype=dtype)
    return inp, residual, weight


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _cuda_graph_kernel_count(graph: torch.cuda.CUDAGraph) -> int:
    graph_handle = graph.raw_cuda_graph()
    result, _, num_nodes = cudart.cudaGraphGetNodes(graph_handle)
    assert result == cudart.cudaError_t.cudaSuccess
    result, nodes, returned_nodes = cudart.cudaGraphGetNodes(
        graph_handle,
        num_nodes,
    )
    assert result == cudart.cudaError_t.cudaSuccess
    assert returned_nodes == num_nodes
    kernel_type = cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    kernel_count = 0
    for node in nodes[:num_nodes]:
        result, node_type = cudart.cudaGraphNodeGetType(node)
        assert result == cudart.cudaError_t.cudaSuccess
        kernel_count += node_type == kernel_type
    return kernel_count


def _run_eager(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    epsilon = 1e-6
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for rows, hidden_size in (
            (1, 6144),
            (1, 8192),
            (2, 6144),
            (3, 6144),
            (4, 6144),
            (5, 6144),
            (6, 6144),
            (8, 6144),
            (3, 128),
        ):
            if rows * hidden_size * dtype.itemsize > 128 * 1024:
                continue
            inp, residual, weight = _make_inputs(
                rows,
                hidden_size,
                dtype,
                device,
                rank,
            )
            expected_out, expected_residual = _reference(
                inp,
                residual,
                weight,
                epsilon,
            )
            torch.cuda.synchronize(device)
            out, residual_out = pool.all_reduce_fused_add_rms_norm(
                inp,
                residual,
                weight,
                epsilon,
            )
            torch.cuda.synchronize(device)
            _assert_close(out, expected_out, dtype)
            _assert_close(residual_out, expected_residual, dtype)


def _run_graph(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
) -> None:
    rows = 4
    hidden_size = 6144
    epsilon = 1e-6
    dtype = torch.bfloat16
    inp, residual, weight = _make_inputs(
        rows,
        hidden_size,
        dtype,
        device,
        rank,
    )
    out = torch.empty_like(inp)
    pool.for_stream()

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with pool.capture(), torch.cuda.graph(graph):
        pool.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            out=out,
            residual_out=residual,
        )
    assert _cuda_graph_kernel_count(graph) == 1

    for iteration in range(3):
        next_inp, next_residual, _ = _make_inputs(
            rows,
            hidden_size,
            dtype,
            device,
            rank,
            iteration=iteration + 1,
        )
        inp.copy_(next_inp)
        residual.copy_(next_residual)
        expected_out, expected_residual = _reference(
            inp,
            residual,
            weight,
            epsilon,
        )
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_close(out, expected_out, dtype)
        _assert_close(residual, expected_residual, dtype)


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
        max_input_bytes=128 * 1024,
        max_size=128 * 1024,
    )
    try:
        _run_eager(pool, device, rank)
        dist.barrier()
        _run_graph(pool, device, rank)
        torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def test_pcie_oneshot_fused_add_rms_norm_eager_and_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("SPARKINFER_PCIE_ONESHOT_RMS_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 6, 8, 10):
        pytest.skip("PCIe oneshot only supports world sizes 2, 4, 6, 8, and 10")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
