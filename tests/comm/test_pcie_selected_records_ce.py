from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sparkinfer.comm.pcie.pcie_dma import FLAG_STRIDE
from sparkinfer.comm.pcie.pcie_oneshot import _OwnedSharedBuffer
from sparkinfer.comm.pcie.pcie_selected_records import (
    PCIeSelectedRecordExchangeInitializationError,
)
from sparkinfer.comm.pcie.pcie_selected_records_ce import (
    _CONFIG_FIELDS,
    PCIeSelectedRecordCopyExchange,
    _copy_exchange_layout,
    _layered_layout_for_slab,
    _largest_layered_capacity_for_slab,
    _validate_rank_configuration,
)


class _FakeExt:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.pack_calls = []
        self.layer_pack_calls = []
        self.barrier_calls = []
        self.unpack_calls = []
        self.layer_unpack_calls = []
        self.publish_calls = []
        self.wait_calls = []
        self.operations = []
        self._packets = {}

    def pack_compact_records(
        self,
        records,
        local_indices,
        primary_ptr,
        overflow_ptr,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
    ):
        call = {
            "records": records.clone(),
            "indices": local_indices.clone(),
            "primary_ptr": primary_ptr,
            "overflow_ptr": overflow_ptr,
            "record_bytes": record_bytes,
            "primary_capacity": primary_capacity,
            "primary_positions_offset": primary_positions_offset,
            "primary_records_offset": primary_records_offset,
            "overflow_positions_offset": overflow_positions_offset,
            "overflow_records_offset": overflow_records_offset,
        }
        self.pack_calls.append(call)
        self.operations.append(("pack", primary_ptr))
        self._packets[primary_ptr] = call

    def pack_compact_record_layers(
        self,
        records0,
        records1,
        records2,
        local_indices,
        primary_ptr,
        overflow_ptr,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
    ):
        call = {
            "records": (records0.clone(), records1.clone(), records2.clone()),
            "indices": local_indices.clone(),
            "primary_ptr": primary_ptr,
            "overflow_ptr": overflow_ptr,
            "record_bytes": record_bytes,
            "primary_capacity": primary_capacity,
            "primary_positions_offset": primary_positions_offset,
            "primary_records_offset": primary_records_offset,
            "overflow_positions_offset": overflow_positions_offset,
            "overflow_records_offset": overflow_records_offset,
        }
        self.layer_pack_calls.append(call)
        self.operations.append(("pack_layers", primary_ptr))
        self._packets[primary_ptr] = call

    def barrier_all_peers(
        self,
        publish_ptrs,
        wait_ptrs,
        send_counters,
        wait_counters,
        phase,
        timeout_cycles,
    ):
        self.barrier_calls.append(
            {
                "phase": phase,
                "timeout_cycles": timeout_cycles,
                "publish_ptrs": publish_ptrs.clone(),
                "wait_ptrs": wait_ptrs.clone(),
            }
        )
        self.operations.append(("barrier", phase))
        send_counters[phase].add_(1)
        wait_counters[phase].add_(1)

    def publish_all_peers(self, publish_ptrs, send_counters, phase):
        self.publish_calls.append(
            {"phase": phase, "publish_ptrs": publish_ptrs.clone()}
        )
        self.operations.append(("publish", phase))
        send_counters[phase].add_(1)

    def wait_all_peers(
        self,
        wait_ptrs,
        wait_counters,
        phase,
        timeout_cycles,
    ):
        self.wait_calls.append(
            {
                "phase": phase,
                "timeout_cycles": timeout_cycles,
                "wait_ptrs": wait_ptrs.clone(),
            }
        )
        self.operations.append(("wait", phase))
        wait_counters[phase].add_(1)

    def unpack_compact_records(
        self,
        primary_base_ptr,
        primary_stride,
        peer_overflow_ptrs,
        out,
        selected_records,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
    ):
        self.unpack_calls.append(
            {
                "primary_base_ptr": primary_base_ptr,
                "primary_stride": primary_stride,
                "peer_overflow_ptrs": peer_overflow_ptrs.clone(),
                "selected_records": selected_records,
                "record_bytes": record_bytes,
                "primary_capacity": primary_capacity,
            }
        )
        self.operations.append(("unpack", selected_records))
        # A single-process fake can reconstruct the records this source owns
        # for its own destination. Distributed GPU tests cover peer packets.
        # Rank rotation always packs this rank's own destination at phase 0.
        local_packet = self.pack_calls[0]
        flat_records = local_packet["records"].reshape(-1, record_bytes)
        flat_indices = local_packet["indices"].reshape(-1)
        flat_out = out.reshape(-1, record_bytes)
        for selected, local_index in enumerate(flat_indices.tolist()):
            if local_index >= 0:
                flat_out[selected].copy_(flat_records[local_index])

    def unpack_compact_record_layers(
        self,
        primary_base_ptr,
        primary_stride,
        peer_overflow_ptrs,
        out0,
        out1,
        out2,
        selected_records,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
    ):
        self.layer_unpack_calls.append(
            {
                "primary_base_ptr": primary_base_ptr,
                "primary_stride": primary_stride,
                "peer_overflow_ptrs": peer_overflow_ptrs.clone(),
                "selected_records": selected_records,
                "record_bytes": record_bytes,
                "primary_capacity": primary_capacity,
            }
        )
        self.operations.append(("unpack_layers", selected_records))
        local_packet = self.layer_pack_calls[0]
        flat_indices = local_packet["indices"].reshape(-1)
        for records, out in zip(
            local_packet["records"],
            (out0, out1, out2),
            strict=True,
        ):
            flat_records = records.reshape(-1, record_bytes)
            flat_out = out.reshape(-1, record_bytes)
            for selected, local_index in enumerate(flat_indices.tolist()):
                if local_index >= 0:
                    flat_out[selected].copy_(flat_records[local_index])


class _FakeDma:
    def __init__(self) -> None:
        self.calls = []

    def dma_copy(self, destination, source, size):
        self.calls.append((destination, source, size))


def _make_runtime(
    *,
    rank: int = 1,
    world_size: int = 3,
    max_records: int = 5,
    record_bytes: int = 37,
    primary_capacity: int | None = None,
    layered_max_records: int | None = None,
):
    ext = _FakeExt(rank)
    dma = _FakeDma()
    slab_ptrs = tuple(1_000_000 * (peer + 1) for peer in range(world_size))
    runtime = PCIeSelectedRecordCopyExchange(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        peer_slab_ptrs=slab_ptrs,
        max_records=max_records,
        record_bytes=record_bytes,
        primary_capacity=primary_capacity,
        layered_max_records=layered_max_records,
        ext_module=ext,
        dma_ext_module=dma,
    )
    return runtime, ext, dma, slab_ptrs


def test_layout_supports_runtime_world_sizes_balanced_packets_and_int64_offsets():
    layout = _copy_exchange_layout(
        world_size=7,
        max_records=32_769,
        record_bytes=65_537,
    )

    assert layout.primary_capacity == (32_769 + 6) // 7
    assert layout.overflow_capacity == 32_769 - layout.primary_capacity
    assert layout.flags_bytes >= 2 * 7 * FLAG_STRIDE
    assert layout.flags_bytes % 256 == 0
    assert layout.primary_records_offset % 16 == 0
    assert layout.overflow_records_offset % 16 == 0
    assert layout.primary_stride % 256 == 0
    assert layout.overflow_stride % 256 == 0
    assert layout.slab_bytes > 2**31

    base = _copy_exchange_layout(
        world_size=4,
        max_records=65_536,
        record_bytes=432,
    )
    naive_layered = _copy_exchange_layout(
        world_size=4,
        max_records=65_536,
        record_bytes=1_296,
    )
    layered_max_records, layered = _largest_layered_capacity_for_slab(
        world_size=4,
        max_records=65_536,
        record_bytes=432,
        slab_bytes=base.slab_bytes,
    )
    assert base.slab_bytes == 144_182_272
    assert naive_layered.slab_bytes == base.slab_bytes + 270 * 1024**2
    assert naive_layered.slab_bytes - base.flags_bytes == 427_296_768
    assert layered_max_records == 22_112
    assert layered_max_records >= 2 * 8_192
    assert layered_max_records < 3 * 8_192
    assert layered.slab_bytes == 144_173_056
    assert layered.slab_bytes <= base.slab_bytes


@pytest.mark.parametrize("world_size", range(2, 9))
def test_layered_layout_is_deterministic_and_bounded_for_dcp2_through_dcp8(
    world_size,
):
    base = _copy_exchange_layout(
        world_size=world_size,
        max_records=65_536,
        record_bytes=432,
    )
    layered_max_records, layered = _largest_layered_capacity_for_slab(
        world_size=world_size,
        max_records=65_536,
        record_bytes=432,
        slab_bytes=base.slab_bytes,
    )

    assert 1 <= layered_max_records <= 65_536
    assert layered.slab_bytes <= base.slab_bytes


def test_layered_exchange_reuses_the_exact_production_slab_allocation():
    runtime, _, _, _ = _make_runtime(
        rank=0,
        world_size=4,
        max_records=65_536,
        record_bytes=432,
    )

    legacy_layout = _copy_exchange_layout(
        world_size=4,
        max_records=65_536,
        record_bytes=432,
    )
    assert runtime._layout.slab_bytes == legacy_layout.slab_bytes == 144_182_272
    assert runtime._layout.slab_bytes < 150 * 1024**2
    assert runtime.layered_max_records == 22_112
    assert runtime._layer_layout.slab_bytes == 144_173_056
    assert runtime._layer_layout.slab_bytes <= runtime._layout.slab_bytes


def test_single_layer_exchange_remains_available_when_layered_packet_cannot_fit():
    base = _copy_exchange_layout(
        world_size=4,
        max_records=1,
        record_bytes=65_536,
        primary_capacity=1,
    )

    layered_max_records, layered = _layered_layout_for_slab(
        world_size=4,
        max_records=1,
        record_bytes=65_536,
        slab_bytes=base.slab_bytes,
        layered_max_records=None,
        fallback_layout=base,
    )

    assert layered_max_records == 0
    assert layered is base


@pytest.mark.parametrize("world_size", [1, 33])
def test_layout_rejects_unsupported_world_sizes(world_size):
    with pytest.raises(ValueError, match="world_size"):
        _copy_exchange_layout(
            world_size=world_size,
            max_records=4,
            record_bytes=37,
        )


def test_layout_rejects_invalid_and_int64_overflowing_capacities():
    with pytest.raises(ValueError, match="max_records"):
        _copy_exchange_layout(world_size=2, max_records=0, record_bytes=1)
    with pytest.raises(ValueError, match="record_bytes"):
        _copy_exchange_layout(world_size=2, max_records=1, record_bytes=0)
    with pytest.raises(ValueError, match="primary_capacity"):
        _copy_exchange_layout(
            world_size=2,
            max_records=4,
            record_bytes=1,
            primary_capacity=5,
        )
    with pytest.raises(ValueError, match="int64"):
        _copy_exchange_layout(
            world_size=2,
            max_records=2**62,
            record_bytes=4,
        )
    with pytest.raises(ValueError, match="max_records"):
        _largest_layered_capacity_for_slab(
            world_size=2,
            max_records=0,
            record_bytes=1,
            slab_bytes=1,
        )


def test_exchange_uses_rank_rotated_pack_and_dma_order_with_odd_width_records():
    runtime, ext, dma, slab_ptrs = _make_runtime()
    records = torch.arange(7 * 37, dtype=torch.uint8).reshape(7, 37)
    local_indices = torch.full((3, 1, 4), -1, dtype=torch.int64)
    local_indices[1, 0, :3] = torch.tensor([6, 2, 4], dtype=torch.int64)
    out = torch.full((1, 4, 37), 255, dtype=torch.uint8)

    returned = runtime.exchange(records, local_indices, out)

    assert returned is out
    assert torch.equal(out[0, 0], records[6])
    assert torch.equal(out[0, 1], records[2])
    assert torch.equal(out[0, 2], records[4])
    assert torch.equal(out[0, 3], torch.zeros(37, dtype=torch.uint8))
    layout = runtime._layout
    expected_destinations = [1, 2, 0]
    assert [
        (call["primary_ptr"] - layout.staging_primary_base - slab_ptrs[1])
        // layout.primary_stride
        for call in ext.pack_calls
    ] == expected_destinations
    assert [
        (destination - layout.receive_primary_base - slab_ptrs[dest])
        // layout.primary_stride
        for (destination, _, _), dest in zip(
            dma.calls, expected_destinations, strict=True
        )
    ] == [1, 1, 1]
    assert [
        (source - layout.staging_primary_base - slab_ptrs[1]) // layout.primary_stride
        for _, source, _ in dma.calls
    ] == expected_destinations
    assert [call["phase"] for call in ext.barrier_calls] == [0]
    assert [call["phase"] for call in ext.publish_calls] == [1]
    assert ext.wait_calls == []
    assert torch.equal(
        runtime._send_counters,
        torch.ones((2, 3), dtype=torch.int32),
    )
    assert torch.equal(
        runtime._wait_counters,
        torch.stack(
            (
                torch.ones(3, dtype=torch.int32),
                torch.zeros(3, dtype=torch.int32),
            )
        ),
    )
    runtime.close()
    assert [call["phase"] for call in ext.wait_calls] == [1]


def test_exchange_preserves_persistent_state_and_accepts_int32_indices():
    runtime, ext, dma, _ = _make_runtime(record_bytes=32)
    records = torch.arange(6 * 32, dtype=torch.uint8).reshape(6, 32)
    local_indices = torch.full((3, 2), -1, dtype=torch.int32)
    local_indices[1] = torch.tensor([5, 1], dtype=torch.int32)
    out = torch.zeros(2, 32, dtype=torch.uint8)

    runtime.exchange(records, local_indices, out)
    runtime.exchange(records.flip(0).contiguous(), local_indices, out)

    assert len(ext.pack_calls) == 6
    assert len(dma.calls) == 6
    assert [call["phase"] for call in ext.barrier_calls] == [0, 0]
    assert [call["phase"] for call in ext.publish_calls] == [1, 1]
    assert [call["phase"] for call in ext.wait_calls] == [1]
    assert torch.equal(
        runtime._send_counters,
        torch.full((2, 3), 2, dtype=torch.int32),
    )
    runtime.close()
    assert [call["phase"] for call in ext.wait_calls] == [1, 1]


def test_exchange_layers_uses_one_packet_per_destination_and_deferred_release():
    runtime, ext, dma, _ = _make_runtime(record_bytes=32)
    base = torch.arange(7 * 32, dtype=torch.int64).remainder(256).to(torch.uint8)
    records = tuple(
        (base.reshape(7, 32) + layer * 41).to(torch.uint8) for layer in range(3)
    )
    local_indices = torch.full((3, 1, 4), -1, dtype=torch.int64)
    local_indices[1, 0, :3] = torch.tensor([6, 2, 4], dtype=torch.int64)
    outputs = tuple(torch.full((1, 4, 32), 255, dtype=torch.uint8) for _ in range(3))

    returned = runtime.exchange_layers(records, local_indices, outputs)

    assert all(
        actual is expected for actual, expected in zip(returned, outputs, strict=True)
    )
    for layer, output in enumerate(outputs):
        assert torch.equal(output[0, 0], records[layer][6])
        assert torch.equal(output[0, 1], records[layer][2])
        assert torch.equal(output[0, 2], records[layer][4])
        assert torch.equal(output[0, 3], torch.zeros(32, dtype=torch.uint8))
    assert len(ext.layer_pack_calls) == runtime.world_size
    assert len(ext.layer_unpack_calls) == 1
    assert len(dma.calls) == runtime.world_size
    assert [call["phase"] for call in ext.barrier_calls] == [0]
    assert [call["phase"] for call in ext.publish_calls] == [1]
    assert ext.wait_calls == []
    expected_copy_bytes = (
        (
            runtime._layer_layout.primary_records_offset
            + runtime._active_layer_primary_capacity(4) * 3 * runtime.record_bytes
            + 15
        )
        // 16
        * 16
    )
    assert {size for _, _, size in dma.calls} == {expected_copy_bytes}
    assert runtime._deferred_release_pending

    operation_count = len(ext.operations)
    runtime.exchange_layers(records, local_indices, outputs)
    second_operations = ext.operations[operation_count:]
    assert second_operations[0] == ("wait", 1)
    assert second_operations[-1] == ("publish", 1)
    assert [call["phase"] for call in ext.barrier_calls] == [0, 0]
    assert [call["phase"] for call in ext.wait_calls] == [1]
    assert [call["phase"] for call in ext.publish_calls] == [1, 1]

    runtime.close()
    assert [call["phase"] for call in ext.wait_calls] == [1, 1]
    assert not runtime._deferred_release_pending


def test_exchange_layers_empty_selection_keeps_order_and_defers_release():
    runtime, ext, dma, _ = _make_runtime(max_records=2)
    records = tuple(torch.empty((0, 37), dtype=torch.uint8) for _ in range(3))
    indices = torch.empty((3, 0), dtype=torch.int64)
    outputs = tuple(torch.empty((0, 37), dtype=torch.uint8) for _ in range(3))

    returned = runtime.exchange_layers(records, indices, outputs)

    assert all(
        actual is expected for actual, expected in zip(returned, outputs, strict=True)
    )
    assert ext.layer_pack_calls == []
    assert dma.calls == []
    assert [call["phase"] for call in ext.barrier_calls] == [0]
    assert [call["phase"] for call in ext.publish_calls] == [1]
    runtime.close()
    assert [call["phase"] for call in ext.wait_calls] == [1]


def test_single_layer_exchange_empty_selection_keeps_order_and_defers_release():
    runtime, ext, dma, _ = _make_runtime(max_records=2)
    records = torch.empty((0, 37), dtype=torch.uint8)
    indices = torch.empty((3, 0), dtype=torch.int64)
    output = torch.empty((0, 37), dtype=torch.uint8)

    returned = runtime.exchange(records, indices, output)

    assert returned is output
    assert ext.pack_calls == []
    assert dma.calls == []
    assert [call["phase"] for call in ext.barrier_calls] == [0]
    assert [call["phase"] for call in ext.publish_calls] == [1]
    assert runtime._deferred_release_pending
    runtime.close()
    assert [call["phase"] for call in ext.wait_calls] == [1]


def test_single_layer_exchange_drains_layered_release_before_reusing_slab():
    runtime, ext, _, _ = _make_runtime(record_bytes=32)
    records = tuple(torch.zeros(4, 32, dtype=torch.uint8) for _ in range(3))
    indices = torch.full((3, 2), -1, dtype=torch.int32)
    outputs = tuple(torch.zeros(2, 32, dtype=torch.uint8) for _ in range(3))
    runtime.exchange_layers(records, indices, outputs)

    operation_count = len(ext.operations)
    runtime.exchange(records[0], indices, outputs[0])

    operations = ext.operations[operation_count:]
    assert operations[0] == ("wait", 1)
    assert [operation for operation in operations if operation[0] == "barrier"] == [
        ("barrier", 0),
    ]
    assert operations[-1] == ("publish", 1)
    assert runtime._deferred_release_pending
    runtime.close()
    assert not runtime._deferred_release_pending


def test_layered_and_single_exchanges_keep_one_balanced_release_chain():
    runtime, ext, _, _ = _make_runtime(record_bytes=32)
    records = tuple(torch.zeros(4, 32, dtype=torch.uint8) for _ in range(3))
    indices = torch.full((3, 2), -1, dtype=torch.int32)
    outputs = tuple(torch.zeros(2, 32, dtype=torch.uint8) for _ in range(3))

    runtime.exchange_layers(records, indices, outputs)
    operation_count = len(ext.operations)
    calls = (
        lambda: runtime.exchange(records[0], indices, outputs[0]),
        lambda: runtime.exchange(records[0], indices, outputs[0]),
        lambda: runtime.exchange_layers(records, indices, outputs),
    )
    for call in calls:
        call()
        operations = ext.operations[operation_count:]
        assert operations[0] == ("wait", 1)
        assert operations[-1] == ("publish", 1)
        assert [op for op in operations if op[0] == "barrier"] == [("barrier", 0)]
        operation_count = len(ext.operations)

    assert len(ext.publish_calls) == 4
    assert len(ext.wait_calls) == 3
    runtime.close()
    assert len(ext.wait_calls) == 4


def test_exchange_layers_rejects_capacity_invalid_or_aliased_planes():
    records = tuple(torch.zeros(3, 37, dtype=torch.uint8) for _ in range(3))
    indices = torch.full((3, 2), -1, dtype=torch.int32)
    outputs = tuple(torch.zeros(2, 37, dtype=torch.uint8) for _ in range(3))

    layered, _, _, _ = _make_runtime(max_records=2)
    with pytest.raises(ValueError, match="records_by_layer"):
        layered.exchange_layers(records[:2], indices, outputs)
    with pytest.raises(ValueError, match="outputs_by_layer"):
        layered.exchange_layers(records, indices, outputs[:2])
    with pytest.raises(ValueError, match="non-overlapping storage"):
        layered.exchange_layers(records, indices, (outputs[0], outputs[0], outputs[2]))
    overlapping = torch.zeros(3, 37, dtype=torch.uint8)
    with pytest.raises(ValueError, match="non-overlapping storage"):
        layered.exchange_layers(
            records,
            indices,
            (overlapping[:2], overlapping[1:], outputs[2]),
        )


def test_exchange_layers_rejects_counts_above_same_slab_capacity():
    runtime, _, _, _ = _make_runtime(
        max_records=5,
        layered_max_records=2,
    )
    count = runtime.layered_max_records + 1
    records = tuple(torch.zeros(count, 37, dtype=torch.uint8) for _ in range(3))
    indices = torch.full((3, count), -1, dtype=torch.int32)
    outputs = tuple(torch.zeros(count, 37, dtype=torch.uint8) for _ in range(3))

    with pytest.raises(ValueError, match="single-layer exchange fallback"):
        runtime.exchange_layers(records, indices, outputs)


def test_explicit_layered_capacity_cannot_grow_the_base_slab():
    with pytest.raises(ValueError, match="exceeds the existing selected-record slab"):
        _make_runtime(
            world_size=4,
            max_records=65_536,
            record_bytes=432,
            layered_max_records=22_113,
        )


def test_explicit_full_primary_capacity_scales_without_overflow():
    runtime, _, _, _ = _make_runtime(
        world_size=2,
        max_records=8,
        primary_capacity=8,
    )

    assert runtime._active_primary_capacity(1) == 1
    assert runtime._active_primary_capacity(5) == 5
    assert runtime._active_primary_capacity(8) == 8


def test_exchange_accepts_empty_selection_but_keeps_collective_order():
    runtime, ext, dma, _ = _make_runtime(max_records=2)
    records = torch.empty((0, 37), dtype=torch.uint8)
    local_indices = torch.empty((3, 0), dtype=torch.int64)
    out = torch.empty((0, 37), dtype=torch.uint8)

    assert runtime.exchange(records, local_indices, out) is out
    assert ext.pack_calls == []
    assert dma.calls == []
    assert [call["phase"] for call in ext.barrier_calls] == [0]
    assert [call["phase"] for call in ext.publish_calls] == [1]
    assert runtime._deferred_release_pending


def test_exchange_rejects_invalid_shapes_dtypes_capacity_and_closed_state():
    runtime, _, _, _ = _make_runtime(max_records=2)
    records = torch.zeros(3, 37, dtype=torch.uint8)
    indices = torch.full((3, 2), -1, dtype=torch.int32)
    out = torch.zeros(2, 37, dtype=torch.uint8)

    with pytest.raises(ValueError, match="configured 37-byte width"):
        runtime.exchange(records[:, :-1].contiguous(), indices, out)
    with pytest.raises(ValueError, match="int32 or int64"):
        runtime.exchange(records, indices.to(torch.int16), out)
    with pytest.raises(ValueError, match="world_size"):
        runtime.exchange(records, indices[:2], out)
    with pytest.raises(ValueError, match="exceeds capacity"):
        runtime.exchange(
            records,
            torch.full((3, 3), -1, dtype=torch.int32),
            torch.zeros(3, 37, dtype=torch.uint8),
        )
    with pytest.raises(ValueError, match="exactly one record"):
        runtime.exchange(records, indices, torch.zeros(3, 37, dtype=torch.uint8))

    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.exchange(records, indices, out)


def test_cpu_process_group_initialization_raises_fallback_exception(monkeypatch):
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records_ce.dist.get_rank",
        lambda group=None: 0,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records_ce.dist.get_world_size",
        lambda group=None: 2,
    )

    with pytest.raises(
        PCIeSelectedRecordExchangeInitializationError,
        match="requires a CUDA device",
    ):
        PCIeSelectedRecordCopyExchange.from_process_group(
            process_group=object(),
            device="cpu",
            max_records=2,
            record_bytes=37,
        )


@pytest.mark.parametrize(
    "mismatched_field",
    ("record_bytes", "layered_max_records", "layered_slab_bytes"),
)
def test_rank_configuration_handshake_rejects_mismatches(
    monkeypatch,
    mismatched_field,
):
    layout = _copy_exchange_layout(
        world_size=2,
        max_records=4,
        record_bytes=37,
    )
    layered_max_records, layered_layout = _largest_layered_capacity_for_slab(
        world_size=2,
        max_records=4,
        record_bytes=37,
        slab_bytes=layout.slab_bytes,
    )
    field_index = _CONFIG_FIELDS.index(mismatched_field)

    def fake_all_gather_into_tensor(output, local, group=None):
        rows = local.repeat(2, 1)
        rows[1, field_index].add_(1)
        output.copy_(rows.reshape(-1))

    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records_ce.dist.all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records_ce._all_ranks_succeeded",
        lambda status, local_success, process_group: local_success,
    )

    with pytest.raises(
        PCIeSelectedRecordExchangeInitializationError,
        match="configuration mismatch",
    ):
        _validate_rank_configuration(
            process_group=object(),
            device=torch.device("cpu"),
            status=torch.empty(1, dtype=torch.int32),
            world_size=2,
            max_records=4,
            record_bytes=37,
            layout=layout,
            layered_max_records=layered_max_records,
            layered_layout=layered_layout,
        )


def test_first_use_graph_capture_requires_prebound_matching_stream(monkeypatch):
    runtime, _, _, _ = _make_runtime()
    runtime.device = torch.device("cuda", 0)
    stream_key = [77]
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records_ce._is_current_stream_capturing",
        lambda device: True,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records_ce._current_stream_key",
        lambda device: stream_key[0],
    )

    with pytest.raises(RuntimeError, match="before first-use graph capture"):
        runtime._check_stream()
    runtime._bind_stream_key(77)
    runtime._check_stream()
    stream_key[0] = 78
    with pytest.raises(RuntimeError, match="stream-affine"):
        runtime._check_stream()


def test_cleanup_rebinds_configured_cuda_device(monkeypatch):
    operations = []

    class _CleanupIPC:
        def cudaSetDevice(self, device):
            operations.append(("set_device", device))

        def cudaIpcCloseMemHandle(self, pointer):
            operations.append(("close", pointer))

        def cudaFree(self, pointer):
            operations.append(("free", pointer))

    runtime, _, _, _ = _make_runtime()
    runtime.device = torch.device("cuda", 3)
    runtime._ipc = _CleanupIPC()
    runtime._owned_buffer = _OwnedSharedBuffer(
        local_ptr=100,
        peer_ptrs=(100, 200),
        remote_ptrs=(200,),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: operations.append(("synchronize", device.index)),
    )

    runtime.close()

    assert operations == [
        ("set_device", 3),
        ("synchronize", 3),
        ("close", 200),
        ("free", 100),
    ]


def test_cuda_source_uses_ce_pack_unpack_rotation_and_generic_contract():
    module_path = (
        Path(__file__).parents[2]
        / "sparkinfer"
        / "comm"
        / "pcie"
        / "pcie_selected_records_ce.cu"
    )
    source = module_path.read_text(encoding="utf-8")
    python_source = module_path.with_suffix(".py").read_text(encoding="utf-8")

    assert "pack_compact_records_kernel" in source
    assert "unpack_compact_records_kernel" in source
    assert "pack_compact_record_layers_kernel" in source
    assert "unpack_compact_record_layers_kernel" in source
    assert "publish_all_peers_kernel" in source
    assert "wait_all_peers_kernel" in source
    assert "launch_pack<int32_t, uint4>" in source
    assert "launch_pack<int32_t, uint8_t>" in source
    assert "launch_pack<int64_t, uint4>" in source
    assert "launch_pack<int64_t, uint8_t>" in source
    assert "const int64_t local_record" in source
    assert "record_byte_offset(local_record, record_bytes)" in source
    assert "barrier_all_peers_kernel" in source
    assert "cudaMemcpy" not in source
    assert "cudaMalloc" not in source
    assert "_pcie_dma._load_extension()" in python_source
    assert "self._dma_ext.dma_copy(" in python_source
    assert "(self.rank + destination_phase) % self.world_size" in python_source
    assert "def exchange_layers(" in python_source
    assert "self._publish_deferred_release()" in python_source
    assert "self._wait_for_deferred_release()" in python_source
    assert "torch.cat" not in python_source
    lowered = (source + python_source).lower()
    assert "topk" not in lowered
    assert "glm" not in lowered
    assert "mtp" not in lowered
