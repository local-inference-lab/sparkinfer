from __future__ import annotations

import os
import socket
from contextlib import suppress

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie.pcie_selected_records import (
    PCIeSelectedRecordExchange,
    PCIeSelectedRecordExchangeInitializationError,
    _load_extension,
)


pytestmark = pytest.mark.skipif(
    os.getenv("SPARKINFER_RUN_PCIE_SELECTED_RECORDS_TEST") != "1",
    reason=(
        "set SPARKINFER_RUN_PCIE_SELECTED_RECORDS_TEST=1 to run selected-record GPU tests"
    ),
)

MAX_RECORDS = 31
RECORD_WIDTHS = (16, 37, 368, 432)
ODD_RECORD_BYTES = 37
POOL_RECORDS_PER_RANK = 67


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _record_values(
    global_indices: torch.Tensor,
    record_bytes: int,
    iteration: int,
) -> torch.Tensor:
    byte = torch.arange(record_bytes, device=global_indices.device, dtype=torch.int64)
    values = (
        global_indices.to(torch.int64).unsqueeze(1) * 37
        + byte.unsqueeze(0) * 13
        + iteration * 7
    )
    return values.remainder(256).to(torch.uint8)


def _local_records(
    rank: int,
    world_size: int,
    device: torch.device,
    iteration: int,
    record_bytes: int,
) -> torch.Tensor:
    global_indices = (
        torch.arange(POOL_RECORDS_PER_RANK, device=device, dtype=torch.int64)
        * world_size
        + rank
    )
    return _record_values(global_indices, record_bytes, iteration)


def _selection(destination: int, total_records: int, count: int, iteration: int):
    selected = torch.arange(count, dtype=torch.int64)
    return (selected * 17 + destination * 11 + iteration * 23).remainder(total_records)


def _local_maps(
    rank: int,
    world_size: int,
    count: int,
    iteration: int,
    device: torch.device,
) -> torch.Tensor:
    total_records = POOL_RECORDS_PER_RANK * world_size
    maps = torch.full((world_size, count), -1, dtype=torch.int32)
    for destination in range(world_size):
        selected = _selection(destination, total_records, count, iteration)
        owned = selected.remainder(world_size) == rank
        maps[destination, owned] = (
            selected[owned].div(world_size, rounding_mode="floor").to(torch.int32)
        )
    return maps.to(device)


def _expected(
    rank: int,
    world_size: int,
    count: int,
    iteration: int,
    device: torch.device,
    record_bytes: int,
) -> torch.Tensor:
    selected = _selection(
        rank,
        POOL_RECORDS_PER_RANK * world_size,
        count,
        iteration,
    ).to(device)
    return _record_values(selected, record_bytes, iteration)


def _internal_addresses(exchange: PCIeSelectedRecordExchange) -> tuple[object, ...]:
    shared = exchange._owned_buffer
    assert shared is not None
    return (
        shared.local_ptr,
        shared.peer_ptrs,
        shared.remote_ptrs,
        exchange._local_payload_ptr,
        exchange._peer_payload_ptrs.data_ptr(),
        exchange._barrier_publish_ptrs.data_ptr(),
        exchange._barrier_wait_ptrs.data_ptr(),
        exchange._send_counters.data_ptr(),
        exchange._wait_counters.data_ptr(),
    )


def _check_eager(
    exchange: PCIeSelectedRecordExchange,
    rank: int,
    world_size: int,
    device: torch.device,
    record_bytes: int,
) -> None:
    addresses = _internal_addresses(exchange)
    for iteration, count in enumerate((0, 1, 13, MAX_RECORDS), start=1):
        records = _local_records(rank, world_size, device, iteration, record_bytes)
        maps = _local_maps(rank, world_size, count, iteration, device)
        out = torch.empty((count, record_bytes), dtype=torch.uint8, device=device)
        returned = exchange.exchange(records, maps, out)
        torch.cuda.synchronize(device)
        assert returned is out
        assert torch.equal(
            out,
            _expected(rank, world_size, count, iteration, device, record_bytes),
        )
        assert _internal_addresses(exchange) == addresses

    iteration = 50
    records = _local_records(rank, world_size, device, iteration, record_bytes)
    maps = _local_maps(rank, world_size, MAX_RECORDS, iteration, device)
    out = torch.empty((MAX_RECORDS, record_bytes), dtype=torch.uint8, device=device)
    exchange.exchange(records, maps, out)
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(8):
        exchange.exchange(records, maps, out)
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated_before
    assert torch.cuda.max_memory_allocated(device) == allocated_before
    assert _internal_addresses(exchange) == addresses
    assert torch.equal(
        out,
        _expected(
            rank,
            world_size,
            MAX_RECORDS,
            iteration,
            device,
            record_bytes,
        ),
    )


def _check_graph(
    exchange: PCIeSelectedRecordExchange,
    rank: int,
    world_size: int,
    device: torch.device,
    record_bytes: int,
) -> None:
    records = _local_records(rank, world_size, device, 100, record_bytes)
    maps = _local_maps(rank, world_size, MAX_RECORDS, 100, device)
    out = torch.empty((MAX_RECORDS, record_bytes), dtype=torch.uint8, device=device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        exchange.exchange(records, maps, out)
    stream.synchronize()
    dist.barrier()
    addresses = _internal_addresses(exchange)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        exchange.exchange(records, maps, out)
    graph.replay()
    stream.synchronize()
    assert _internal_addresses(exchange) == addresses

    iterations = tuple(range(101, 109))
    record_updates = [
        _local_records(rank, world_size, device, iteration, record_bytes)
        for iteration in iterations
    ]
    map_updates = [
        _local_maps(rank, world_size, MAX_RECORDS, iteration, device)
        for iteration in iterations
    ]
    expected = torch.stack(
        [
            _expected(
                rank,
                world_size,
                MAX_RECORDS,
                iteration,
                device,
                record_bytes,
            )
            for iteration in iterations
        ]
    )
    observed = torch.empty_like(expected)
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    for replay in range(len(iterations)):
        records.copy_(record_updates[replay])
        maps.copy_(map_updates[replay])
        stream.wait_stream(torch.cuda.current_stream(device))
        graph.replay()
        stream.synchronize()
        observed[replay].copy_(out)
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated_before
    assert torch.cuda.max_memory_allocated(device) == allocated_before
    assert _internal_addresses(exchange) == addresses
    assert torch.equal(observed, expected)


def _check_big_pool_offset(
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    record_bytes = 65_536
    high_record = 2**31 // record_bytes + 1
    required_bytes = (high_record + 1) * record_bytes
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    available = torch.tensor(
        [1 if rank != 0 or free_bytes >= required_bytes + 512 * 1024**2 else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(available, op=dist.ReduceOp.MIN)
    if not bool(available.item()):
        raise RuntimeError(
            "selected-record big-offset gate requires at least "
            f"{required_bytes + 512 * 1024**2} free bytes on rank 0"
        )

    exchange = PCIeSelectedRecordExchange.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_records=1,
        record_bytes=record_bytes,
    )
    try:
        rows = high_record + 1 if rank == 0 else 1
        records = torch.empty((rows, record_bytes), dtype=torch.uint8, device=device)
        expected = torch.arange(record_bytes, dtype=torch.int64, device=device)
        expected = expected.remainder(251).to(torch.uint8).reshape(1, record_bytes)
        if rank == 0:
            records[high_record].copy_(expected[0])
        maps = torch.full((world_size, 1), -1, dtype=torch.int32, device=device)
        if rank == 0:
            maps.fill_(high_record)
        out = torch.empty_like(expected)
        exchange.exchange(records, maps, out)
        torch.cuda.synchronize(device)
        assert high_record * record_bytes > 2**31
        assert torch.equal(out, expected)
    finally:
        exchange.close()
        with suppress(Exception):
            del records
        torch.cuda.empty_cache()
    dist.barrier()


def _check_large_offset_arithmetic() -> None:
    extension = _load_extension()
    high_source_record = 2**31 // 65_536 + 1
    high_destination_record = 2**31 // 432 + 1
    assert extension.record_byte_offset_for_test(high_source_record, 65_536) > 2**31
    assert extension.record_byte_offset_for_test(high_destination_record, 432) > 2**31


def _check_rank_configuration_mismatch(
    rank: int,
    device: torch.device,
) -> None:
    from sparkinfer.comm.pcie import pcie_selected_records as module

    malloc_calls = []
    original_malloc = module.CudaRTLibrary.cudaMalloc

    def unexpected_malloc(_self, size):
        malloc_calls.append(size)
        raise AssertionError("configuration mismatch reached IPC allocation")

    module.CudaRTLibrary.cudaMalloc = unexpected_malloc
    try:
        with pytest.raises(
            PCIeSelectedRecordExchangeInitializationError,
            match="configuration mismatch",
        ):
            PCIeSelectedRecordExchange.from_process_group(
                process_group=dist.group.WORLD,
                device=device,
                max_records=MAX_RECORDS + rank % 2,
                record_bytes=ODD_RECORD_BYTES,
            )
        assert malloc_calls == []
    finally:
        module.CudaRTLibrary.cudaMalloc = original_malloc
    dist.barrier()


def _check_rank_consistent_handle_serialization_failure(
    rank: int,
    device: torch.device,
) -> None:
    from sparkinfer.comm.pcie import pcie_selected_records as module

    original_get_handle = module.CudaRTLibrary.cudaIpcGetMemHandleBytes
    if rank == 0:

        def fail_get_handle(_self, _pointer):
            raise RuntimeError("intentional IPC handle serialization failure")

        module.CudaRTLibrary.cudaIpcGetMemHandleBytes = fail_get_handle
    try:
        with pytest.raises(
            PCIeSelectedRecordExchangeInitializationError,
            match="handle preparation failed",
        ):
            PCIeSelectedRecordExchange.from_process_group(
                process_group=dist.group.WORLD,
                device=device,
                max_records=1,
                record_bytes=ODD_RECORD_BYTES,
            )
    finally:
        module.CudaRTLibrary.cudaIpcGetMemHandleBytes = original_get_handle
    dist.barrier()


def _check_rank_consistent_initialization_failure(
    rank: int,
    device: torch.device,
) -> None:
    from sparkinfer.comm.pcie import pcie_selected_records as module

    original_malloc = module.CudaRTLibrary.cudaMalloc
    if rank == 0:

        def fail_malloc(_self, _size):
            raise RuntimeError("intentional selected-record allocation failure")

        module.CudaRTLibrary.cudaMalloc = fail_malloc
    try:
        with pytest.raises(PCIeSelectedRecordExchangeInitializationError):
            PCIeSelectedRecordExchange.from_process_group(
                process_group=dist.group.WORLD,
                device=device,
                max_records=1,
                record_bytes=ODD_RECORD_BYTES,
            )
    finally:
        module.CudaRTLibrary.cudaMalloc = original_malloc
    dist.barrier()


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    _check_large_offset_arithmetic()
    _check_rank_configuration_mismatch(rank, device)
    for record_bytes in RECORD_WIDTHS:
        exchange = PCIeSelectedRecordExchange.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_records=MAX_RECORDS,
            record_bytes=record_bytes,
        )
        try:
            _check_eager(exchange, rank, world_size, device, record_bytes)
            torch.cuda.synchronize(device)
        finally:
            exchange.close()
        dist.barrier()

        graph_exchange = PCIeSelectedRecordExchange.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_records=MAX_RECORDS,
            record_bytes=record_bytes,
        )
        try:
            _check_graph(
                graph_exchange,
                rank,
                world_size,
                device,
                record_bytes,
            )
            torch.cuda.synchronize(device)
        finally:
            graph_exchange.close()
        dist.barrier()

    _check_big_pool_offset(rank, world_size, device)
    _check_rank_consistent_handle_serialization_failure(rank, device)
    _check_rank_consistent_initialization_failure(rank, device)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4), ids=("world2", "world4"))
def test_pcie_selected_records_eager_graph_and_big_offset_correctness(
    world_size: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if (
        world_size == 4
        and os.getenv("SPARKINFER_RUN_PCIE_SELECTED_RECORDS_WORLD4_TEST") != "1"
    ):
        pytest.skip(
            "set SPARKINFER_RUN_PCIE_SELECTED_RECORDS_WORLD4_TEST=1 for the 4-GPU gate"
        )
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    _load_extension()
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
