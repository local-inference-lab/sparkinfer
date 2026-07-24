from __future__ import annotations

import os
import socket
from contextlib import suppress

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparkinfer.comm.pcie import pcie_dma as _pcie_dma
from sparkinfer.comm.pcie.pcie_selected_records_ce import (
    PCIeSelectedRecordCopyExchange,
    _load_extension,
)


pytestmark = pytest.mark.skipif(
    os.getenv("SPARKINFER_RUN_PCIE_SELECTED_RECORDS_CE_TEST") != "1",
    reason=(
        "set SPARKINFER_RUN_PCIE_SELECTED_RECORDS_CE_TEST=1 to run copy-engine "
        "selected-record GPU tests"
    ),
)

MAX_RECORDS = 31
PRIMARY_CAPACITY = 1
POOL_RECORDS_PER_RANK = 67
BIG_RECORD_BYTES = 65_536


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _record_values(
    global_indices: torch.Tensor,
    record_bytes: int,
    iteration: int,
) -> torch.Tensor:
    byte = torch.arange(
        record_bytes,
        device=global_indices.device,
        dtype=torch.int64,
    )
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
        torch.arange(
            POOL_RECORDS_PER_RANK,
            device=device,
            dtype=torch.int64,
        )
        * world_size
        + rank
    )
    return _record_values(global_indices, record_bytes, iteration)


def _selection(
    destination: int,
    total_records: int,
    count: int,
    iteration: int,
) -> torch.Tensor:
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


def _layer_records(
    rank: int,
    world_size: int,
    device: torch.device,
    iteration: int,
    record_bytes: int,
    layer: int,
) -> torch.Tensor:
    records = _local_records(
        rank,
        world_size,
        device,
        iteration,
        record_bytes,
    )
    return (records.to(torch.int16) + layer * 53).remainder(256).to(torch.uint8)


def _skewed_local_maps(
    rank: int,
    world_size: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    maps = torch.full(
        (world_size, count),
        -1,
        dtype=torch.int32,
        device=device,
    )
    if rank == 0:
        maps.copy_(
            torch.arange(count, dtype=torch.int32, device=device).expand(
                world_size,
                count,
            )
        )
    return maps


def _layer_expected(
    world_size: int,
    count: int,
    iteration: int,
    device: torch.device,
    record_bytes: int,
    layer: int,
) -> torch.Tensor:
    selected = torch.arange(count, dtype=torch.int64, device=device) * world_size
    expected = _record_values(selected, record_bytes, iteration)
    return (expected.to(torch.int16) + layer * 53).remainder(256).to(torch.uint8)


def _internal_addresses(
    exchange: PCIeSelectedRecordCopyExchange,
) -> tuple[object, ...]:
    shared = exchange._owned_buffer
    assert shared is not None
    return (
        shared.local_ptr,
        shared.peer_ptrs,
        shared.remote_ptrs,
        exchange._local_slab_ptr,
        exchange._peer_overflow_ptrs.data_ptr(),
        exchange._peer_layer_overflow_ptrs.data_ptr(),
        exchange._barrier_publish_ptrs.data_ptr(),
        exchange._barrier_wait_ptrs.data_ptr(),
        exchange._send_counters.data_ptr(),
        exchange._wait_counters.data_ptr(),
    )


def _assert_forced_overflow(
    exchange: PCIeSelectedRecordCopyExchange,
    maps: torch.Tensor,
) -> None:
    active_records = maps.numel() // exchange.world_size
    active_primary = exchange._active_primary_capacity(active_records)
    valid_by_destination = maps.ge(0).sum(dim=1)
    assert bool(torch.all(valid_by_destination > active_primary).item())


def _check_eager(
    exchange: PCIeSelectedRecordCopyExchange,
    rank: int,
    world_size: int,
    device: torch.device,
    record_bytes: int,
) -> None:
    addresses = _internal_addresses(exchange)
    for iteration, count in enumerate((0, 1, 13, MAX_RECORDS), start=1):
        records = _local_records(
            rank,
            world_size,
            device,
            iteration,
            record_bytes,
        )
        maps = _local_maps(rank, world_size, count, iteration, device)
        if count == MAX_RECORDS:
            _assert_forced_overflow(exchange, maps)
        out = torch.empty(
            (count, record_bytes),
            dtype=torch.uint8,
            device=device,
        )
        returned = exchange.exchange(records, maps, out)
        torch.cuda.synchronize(device)
        assert returned is out
        assert torch.equal(
            out,
            _expected(
                rank,
                world_size,
                count,
                iteration,
                device,
                record_bytes,
            ),
        )
        assert _internal_addresses(exchange) == addresses

    iteration = 50
    records = _local_records(
        rank,
        world_size,
        device,
        iteration,
        record_bytes,
    )
    maps = _local_maps(rank, world_size, MAX_RECORDS, iteration, device)
    _assert_forced_overflow(exchange, maps)
    out = torch.empty(
        (MAX_RECORDS, record_bytes),
        dtype=torch.uint8,
        device=device,
    )
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
    exchange: PCIeSelectedRecordCopyExchange,
    rank: int,
    world_size: int,
    device: torch.device,
    record_bytes: int,
) -> None:
    records = _local_records(rank, world_size, device, 100, record_bytes)
    maps = _local_maps(rank, world_size, MAX_RECORDS, 100, device)
    _assert_forced_overflow(exchange, maps)
    out = torch.empty(
        (MAX_RECORDS, record_bytes),
        dtype=torch.uint8,
        device=device,
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        exchange.exchange(records, maps, out)
    stream.synchronize()
    dist.barrier()
    addresses = _internal_addresses(exchange)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        exchange.exchange(records, maps, out)
    dist.barrier()
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


def _check_layers_eager(
    exchange: PCIeSelectedRecordCopyExchange,
    rank: int,
    world_size: int,
    device: torch.device,
    record_bytes: int,
) -> None:
    count = min(8, exchange.layered_max_records)
    assert count == 8
    maps = _skewed_local_maps(rank, world_size, count, device)
    active_primary = exchange._active_layer_primary_capacity(count)
    if rank == 0:
        assert bool(torch.all(maps.ge(0).sum(dim=1) > active_primary).item())
    addresses = _internal_addresses(exchange)

    for iteration in (200, 201):
        records = tuple(
            _layer_records(
                rank,
                world_size,
                device,
                iteration,
                record_bytes,
                layer,
            )
            for layer in range(3)
        )
        outputs = tuple(
            torch.empty((count, record_bytes), dtype=torch.uint8, device=device)
            for _ in range(3)
        )
        returned = exchange.exchange_layers(records, maps, outputs)
        torch.cuda.synchronize(device)
        for layer, (actual, output) in enumerate(zip(returned, outputs, strict=True)):
            assert actual is output
            assert torch.equal(
                output,
                _layer_expected(
                    world_size,
                    count,
                    iteration,
                    device,
                    record_bytes,
                    layer,
                ),
            )
        assert _internal_addresses(exchange) == addresses


def _check_layers_graph(
    exchange: PCIeSelectedRecordCopyExchange,
    rank: int,
    world_size: int,
    device: torch.device,
    record_bytes: int,
) -> None:
    count = min(8, exchange.layered_max_records)
    assert count == 8
    iteration = 300
    records = tuple(
        _layer_records(
            rank,
            world_size,
            device,
            iteration,
            record_bytes,
            layer,
        )
        for layer in range(3)
    )
    maps = _skewed_local_maps(rank, world_size, count, device)
    outputs = tuple(
        torch.empty((count, record_bytes), dtype=torch.uint8, device=device)
        for _ in range(3)
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        exchange.exchange_layers(records, maps, outputs)
    stream.synchronize()
    dist.barrier()
    addresses = _internal_addresses(exchange)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        exchange.exchange_layers(records, maps, outputs)
    dist.barrier()
    graph.replay()
    stream.synchronize()
    assert _internal_addresses(exchange) == addresses

    iterations = tuple(range(301, 305))
    updates = [
        tuple(
            _layer_records(
                rank,
                world_size,
                device,
                current_iteration,
                record_bytes,
                layer,
            )
            for layer in range(3)
        )
        for current_iteration in iterations
    ]
    observed = tuple(
        torch.empty(
            (len(iterations), count, record_bytes),
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(3)
    )
    expected = tuple(
        torch.stack(
            [
                _layer_expected(
                    world_size,
                    count,
                    current_iteration,
                    device,
                    record_bytes,
                    layer,
                )
                for current_iteration in iterations
            ]
        )
        for layer in range(3)
    )
    stream.wait_stream(torch.cuda.current_stream(device))
    stream.synchronize()
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    for replay, _ in enumerate(iterations):
        for records_plane, update in zip(records, updates[replay], strict=True):
            records_plane.copy_(update)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph.replay()
        stream.synchronize()
        for layer, output in enumerate(outputs):
            observed[layer][replay].copy_(output)
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated_before
    assert torch.cuda.max_memory_allocated(device) == allocated_before
    assert _internal_addresses(exchange) == addresses
    assert all(
        torch.equal(actual, wanted)
        for actual, wanted in zip(observed, expected, strict=True)
    )


def _check_mixed_graph_modes(
    exchange: PCIeSelectedRecordCopyExchange,
    rank: int,
    world_size: int,
    device: torch.device,
    record_bytes: int,
) -> None:
    layer_count = min(8, exchange.layered_max_records)
    layer_maps = _skewed_local_maps(rank, world_size, layer_count, device)
    single_maps = _local_maps(rank, world_size, MAX_RECORDS, 400, device)
    single_records = _local_records(rank, world_size, device, 400, record_bytes)
    single_out = torch.empty(
        (MAX_RECORDS, record_bytes), dtype=torch.uint8, device=device
    )
    layer_records = tuple(
        _layer_records(
            rank,
            world_size,
            device,
            500,
            record_bytes,
            layer,
        )
        for layer in range(3)
    )
    layer_outputs = tuple(
        torch.empty((layer_count, record_bytes), dtype=torch.uint8, device=device)
        for _ in range(3)
    )
    empty_records = torch.empty((0, record_bytes), dtype=torch.uint8, device=device)
    empty_maps = torch.empty((world_size, 0), dtype=torch.int32, device=device)
    empty_out = torch.empty((0, record_bytes), dtype=torch.uint8, device=device)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        exchange.exchange(single_records, single_maps, single_out)
        exchange.exchange_layers(layer_records, layer_maps, layer_outputs)
    stream.synchronize()
    dist.barrier()

    single_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single_graph, stream=stream):
        exchange.exchange(single_records, single_maps, single_out)
    dist.barrier()
    layer_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(layer_graph, stream=stream):
        exchange.exchange_layers(layer_records, layer_maps, layer_outputs)
    dist.barrier()
    empty_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(empty_graph, stream=stream):
        exchange.exchange(empty_records, empty_maps, empty_out)
    dist.barrier()

    # Reproduce production switching and preserve the graph-balanced release
    # chain even when one rank-consistent exchange has no selected records.
    replay_modes = ("layer", "empty", "empty", "single", "single", "layer")
    for replay, mode in enumerate(replay_modes):
        iteration = 600 + replay
        if mode == "single":
            single_records.copy_(
                _local_records(rank, world_size, device, iteration, record_bytes)
            )
            single_maps.copy_(
                _local_maps(rank, world_size, MAX_RECORDS, iteration, device)
            )
        elif mode == "layer":
            for layer, records in enumerate(layer_records):
                records.copy_(
                    _layer_records(
                        rank,
                        world_size,
                        device,
                        iteration,
                        record_bytes,
                        layer,
                    )
                )
        stream.wait_stream(torch.cuda.current_stream(device))
        dist.barrier()
        if mode == "single":
            single_graph.replay()
        elif mode == "empty":
            empty_graph.replay()
        else:
            layer_graph.replay()
        stream.synchronize()

        if mode == "single":
            assert torch.equal(
                single_out,
                _expected(
                    rank,
                    world_size,
                    MAX_RECORDS,
                    iteration,
                    device,
                    record_bytes,
                ),
            )
        elif mode == "layer":
            for layer, output in enumerate(layer_outputs):
                assert torch.equal(
                    output,
                    _layer_expected(
                        world_size,
                        layer_count,
                        iteration,
                        device,
                        record_bytes,
                        layer,
                    ),
                )


def _check_big_pool_offset(
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    high_record = 2**31 // BIG_RECORD_BYTES + 1
    required_bytes = (high_record + 1) * BIG_RECORD_BYTES
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
            "copy-engine selected-record big-offset gate requires at least "
            f"{required_bytes + 512 * 1024**2} free bytes on rank 0"
        )

    exchange = PCIeSelectedRecordCopyExchange.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_records=1,
        record_bytes=BIG_RECORD_BYTES,
        primary_capacity=1,
    )
    records = None
    try:
        rows = high_record + 1 if rank == 0 else 1
        records = torch.empty(
            (rows, BIG_RECORD_BYTES),
            dtype=torch.uint8,
            device=device,
        )
        expected = torch.arange(
            BIG_RECORD_BYTES,
            dtype=torch.int64,
            device=device,
        )
        expected = (
            expected.remainder(251)
            .to(torch.uint8)
            .reshape(
                1,
                BIG_RECORD_BYTES,
            )
        )
        if rank == 0:
            records[high_record].copy_(expected[0])
        maps = torch.full(
            (world_size, 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        if rank == 0:
            maps.fill_(high_record)
        out = torch.empty_like(expected)
        exchange.exchange(records, maps, out)
        torch.cuda.synchronize(device)
        assert high_record * BIG_RECORD_BYTES > 2**31
        assert torch.equal(out, expected)
    finally:
        exchange.close()
        with suppress(Exception):
            del records
        torch.cuda.empty_cache()
    dist.barrier()


def _worker(
    rank: int,
    world_size: int,
    port: int,
    record_bytes: int,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        eager_exchange = PCIeSelectedRecordCopyExchange.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_records=MAX_RECORDS,
            record_bytes=record_bytes,
            primary_capacity=PRIMARY_CAPACITY,
        )
        try:
            _check_eager(
                eager_exchange,
                rank,
                world_size,
                device,
                record_bytes,
            )
            _check_layers_eager(
                eager_exchange,
                rank,
                world_size,
                device,
                record_bytes,
            )
            torch.cuda.synchronize(device)
        finally:
            eager_exchange.close()
        dist.barrier()

        graph_exchange = PCIeSelectedRecordCopyExchange.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_records=MAX_RECORDS,
            record_bytes=record_bytes,
            primary_capacity=PRIMARY_CAPACITY,
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

        layer_graph_exchange = PCIeSelectedRecordCopyExchange.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_records=MAX_RECORDS,
            record_bytes=record_bytes,
            primary_capacity=PRIMARY_CAPACITY,
        )
        try:
            _check_layers_graph(
                layer_graph_exchange,
                rank,
                world_size,
                device,
                record_bytes,
            )
            torch.cuda.synchronize(device)
        finally:
            layer_graph_exchange.close()
        dist.barrier()

        mixed_graph_exchange = PCIeSelectedRecordCopyExchange.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_records=MAX_RECORDS,
            record_bytes=record_bytes,
            primary_capacity=PRIMARY_CAPACITY,
        )
        try:
            _check_mixed_graph_modes(
                mixed_graph_exchange,
                rank,
                world_size,
                device,
                record_bytes,
            )
            torch.cuda.synchronize(device)
        finally:
            mixed_graph_exchange.close()
        dist.barrier()

        _check_big_pool_offset(rank, world_size, device)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world_size", "record_bytes"),
    ((4, 368), (4, 432), (8, 368), (8, 656)),
    ids=("dcp4-record368", "dcp4-record432", "dcp8-record368", "dcp8-record656"),
)
def test_pcie_selected_record_ce_eager_graph_overflow_and_big_offset(
    world_size: int,
    record_bytes: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if (
        world_size == 8
        and os.getenv("SPARKINFER_RUN_PCIE_SELECTED_RECORDS_CE_WORLD8_TEST") != "1"
    ):
        pytest.skip(
            "set SPARKINFER_RUN_PCIE_SELECTED_RECORDS_CE_WORLD8_TEST=1 for "
            "the DCP8 selected-record gate"
        )
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )

    _load_extension()
    _pcie_dma._load_extension()
    mp.spawn(
        _worker,
        args=(world_size, _free_port(), record_bytes),
        nprocs=world_size,
        join=True,
    )
