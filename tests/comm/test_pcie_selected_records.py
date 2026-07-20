from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sparkinfer.comm.pcie import (
    SelectedRecordExchange as PCIeSelectedRecordExchange,
    SelectedRecordExchangeInitializationError as PCIeSelectedRecordExchangeInitializationError,
)
from sparkinfer.comm.pcie.pcie_dma import FLAG_STRIDE
from sparkinfer.comm.pcie.pcie_oneshot import _OwnedSharedBuffer
from sparkinfer.comm.pcie.pcie_selected_records import (
    _CONFIG_FIELDS,
    _IPC_HANDLE_BYTES,
    _gather_ipc_handles,
    _selected_record_layout,
    _validate_rank_configuration,
)


class _FakeExt:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.calls = []

    def exchange(
        self,
        records,
        local_indices_by_destination,
        peer_payload_ptrs,
        local_payload_ptr,
        barrier_publish_ptrs,
        barrier_wait_ptrs,
        send_counters,
        wait_counters,
        out,
        record_bytes,
        timeout_cycles,
    ):
        self.calls.append(
            {
                "peer_payload_ptrs": peer_payload_ptrs.clone(),
                "local_payload_ptr": local_payload_ptr,
                "publish_ptrs": barrier_publish_ptrs.clone(),
                "wait_ptrs": barrier_wait_ptrs.clone(),
                "send_counters_ptr": send_counters.data_ptr(),
                "wait_counters_ptr": wait_counters.data_ptr(),
                "record_bytes": record_bytes,
                "timeout_cycles": timeout_cycles,
            }
        )
        flat_records = records.reshape(-1, record_bytes)
        flat_indices = local_indices_by_destination[self.rank].reshape(-1)
        flat_out = out.reshape(-1, record_bytes)
        for selected, local_index in enumerate(flat_indices.tolist()):
            if local_index >= 0:
                flat_out[selected].copy_(flat_records[local_index])
        send_counters.add_(1)
        wait_counters.add_(1)


def _make_runtime(
    *,
    rank: int = 1,
    world_size: int = 3,
    max_records: int = 4,
    record_bytes: int = 37,
) -> tuple[PCIeSelectedRecordExchange, _FakeExt, tuple[int, ...]]:
    ext = _FakeExt(rank)
    layout = _selected_record_layout(
        world_size=world_size,
        max_records=max_records,
        record_bytes=record_bytes,
    )
    slab_ptrs = tuple(10_000 * (peer + 1) for peer in range(world_size))
    runtime = PCIeSelectedRecordExchange(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        peer_slab_ptrs=slab_ptrs,
        payload_offset=layout.payload_offset,
        max_records=max_records,
        record_bytes=record_bytes,
        ext_module=ext,
    )
    return runtime, ext, slab_ptrs


def test_layout_uses_runtime_world_size_and_pool_scaled_int64_capacity():
    layout = _selected_record_layout(
        world_size=7,
        max_records=32_769,
        record_bytes=65_536,
    )

    assert layout.flags_bytes >= 2 * 7 * FLAG_STRIDE
    assert layout.flags_bytes % 256 == 0
    assert layout.payload_offset == layout.flags_bytes
    assert layout.payload_bytes == 32_769 * 65_536
    assert layout.payload_bytes > 2**31
    assert layout.slab_bytes == layout.payload_offset + layout.payload_bytes


def test_layout_rejects_invalid_and_int64_overflowing_capacities():
    with pytest.raises(ValueError, match="world_size"):
        _selected_record_layout(world_size=33, max_records=1, record_bytes=1)
    with pytest.raises(ValueError, match="max_records"):
        _selected_record_layout(world_size=2, max_records=0, record_bytes=1)
    with pytest.raises(ValueError, match="record_bytes"):
        _selected_record_layout(world_size=2, max_records=1, record_bytes=0)
    with pytest.raises(ValueError, match="int64"):
        _selected_record_layout(
            world_size=2,
            max_records=2**62,
            record_bytes=4,
        )


def test_exchange_dispatches_exact_odd_width_records_with_persistent_state():
    runtime, ext, slab_ptrs = _make_runtime()
    records = torch.arange(5 * 37, dtype=torch.uint8).reshape(5, 37)
    local_indices = torch.full((3, 1, 2), -1, dtype=torch.int32)
    local_indices[1, 0] = torch.tensor([4, 1], dtype=torch.int32)
    out = torch.zeros(1, 2, 37, dtype=torch.uint8)

    returned = runtime.exchange(records, local_indices, out)
    runtime.exchange(records.flip(0).contiguous(), local_indices, out)

    assert returned is out
    assert torch.equal(out[0, 0], records[0])
    assert torch.equal(out[0, 1], records[3])
    assert len(ext.calls) == 2
    assert ext.calls[0]["record_bytes"] == 37
    assert ext.calls[0]["send_counters_ptr"] == ext.calls[1]["send_counters_ptr"]
    assert ext.calls[0]["wait_counters_ptr"] == ext.calls[1]["wait_counters_ptr"]

    layout = _selected_record_layout(
        world_size=3,
        max_records=4,
        record_bytes=37,
    )
    assert ext.calls[0]["peer_payload_ptrs"].tolist() == [
        ptr + layout.payload_offset for ptr in slab_ptrs
    ]
    assert ext.calls[0]["local_payload_ptr"] == slab_ptrs[1] + layout.payload_offset
    assert ext.calls[0]["publish_ptrs"].tolist() == [
        [slab_ptrs[destination] + FLAG_STRIDE for destination in range(3)],
        [slab_ptrs[destination] + 4 * FLAG_STRIDE for destination in range(3)],
    ]
    assert ext.calls[0]["wait_ptrs"].tolist() == [
        [slab_ptrs[1] + source * FLAG_STRIDE for source in range(3)],
        [slab_ptrs[1] + (3 + source) * FLAG_STRIDE for source in range(3)],
    ]
    assert torch.equal(runtime._send_counters, torch.full((2, 3), 2, dtype=torch.int32))
    assert torch.equal(runtime._wait_counters, torch.full((2, 3), 2, dtype=torch.int32))


def test_exchange_accepts_int64_indices_and_an_empty_selection():
    runtime, ext, _ = _make_runtime(max_records=2)
    records = torch.empty((0, 37), dtype=torch.uint8)
    local_indices = torch.empty((3, 0), dtype=torch.int64)
    out = torch.empty((0, 37), dtype=torch.uint8)

    assert runtime.exchange(records, local_indices, out) is out
    assert len(ext.calls) == 1


def test_exchange_rejects_invalid_shapes_dtypes_capacity_and_closed_state():
    runtime, _, _ = _make_runtime(max_records=2)
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


def test_cpu_process_group_initialization_raises_the_fallback_exception(monkeypatch):
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records.dist.get_rank",
        lambda group=None: 0,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records.dist.get_world_size",
        lambda group=None: 2,
    )

    with pytest.raises(
        PCIeSelectedRecordExchangeInitializationError,
        match="requires a CUDA device",
    ):
        PCIeSelectedRecordExchange.from_process_group(
            process_group=object(),
            device="cpu",
            max_records=2,
            record_bytes=37,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("world_size", 3),
        ("max_records", 5),
        ("record_bytes", 38),
        ("payload_offset", 4096),
    ),
)
def test_rank_configuration_handshake_rejects_mismatches(
    monkeypatch,
    field: str,
    replacement: int,
):
    layout = _selected_record_layout(
        world_size=2,
        max_records=4,
        record_bytes=37,
    )
    field_index = _CONFIG_FIELDS.index(field)

    def fake_all_gather_into_tensor(output, local, group=None):
        rows = local.repeat(2, 1)
        rows[1, field_index] = replacement
        output.copy_(rows.reshape(-1))

    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records.dist.all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records._all_ranks_succeeded",
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
        )


def test_ipc_handle_preflight_stops_before_tensor_collective(monkeypatch):
    collective_calls = []

    class _SerializationFailureIPC:
        def cudaIpcGetMemHandleBytes(self, pointer):
            raise RuntimeError(f"cannot serialize pointer {pointer}")

    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records.dist.all_gather_into_tensor",
        lambda *args, **kwargs: collective_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records._all_ranks_succeeded",
        lambda status, local_success, process_group: local_success,
    )

    with pytest.raises(
        PCIeSelectedRecordExchangeInitializationError,
        match="handle preparation failed",
    ):
        _gather_ipc_handles(
            process_group=object(),
            device=torch.device("cpu"),
            world_size=2,
            local_ptr=1234,
            ipc=_SerializationFailureIPC(),
            status=torch.empty(1, dtype=torch.int32),
        )
    assert collective_calls == []


def test_ipc_handle_allocation_preflight_stops_before_tensor_collective(monkeypatch):
    collective_calls = []
    status = torch.empty(1, dtype=torch.int32)

    def fail_empty(*args, **kwargs):
        raise RuntimeError("intentional handle tensor allocation failure")

    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records.torch.empty",
        fail_empty,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records.dist.all_gather_into_tensor",
        lambda *args, **kwargs: collective_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records._all_ranks_succeeded",
        lambda status, local_success, process_group: local_success,
    )

    with pytest.raises(
        PCIeSelectedRecordExchangeInitializationError,
        match="handle preparation failed",
    ):
        _gather_ipc_handles(
            process_group=object(),
            device=torch.device("cpu"),
            world_size=2,
            local_ptr=1234,
            ipc=object(),
            status=status,
        )
    assert collective_calls == []


def test_ipc_handles_use_one_fixed_uint8_tensor_collective(monkeypatch):
    local_handle = bytes(range(_IPC_HANDLE_BYTES))
    peer_handle = bytes(reversed(range(_IPC_HANDLE_BYTES)))
    collective_shapes = []

    class _IPC:
        def cudaIpcGetMemHandleBytes(self, pointer):
            assert pointer == 1234
            return local_handle

    def fake_all_gather_into_tensor(output, local, group=None):
        collective_shapes.append((tuple(output.shape), tuple(local.shape), local.dtype))
        output.view(2, _IPC_HANDLE_BYTES)[0].copy_(local)
        output.view(2, _IPC_HANDLE_BYTES)[1].copy_(
            torch.tensor(tuple(peer_handle), dtype=torch.uint8)
        )

    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records.dist.all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records._all_ranks_succeeded",
        lambda status, local_success, process_group: local_success,
    )

    handles = _gather_ipc_handles(
        process_group=object(),
        device=torch.device("cpu"),
        world_size=2,
        local_ptr=1234,
        ipc=_IPC(),
        status=torch.empty(1, dtype=torch.int32),
    )

    assert handles == (local_handle, peer_handle)
    assert collective_shapes == [
        ((2 * _IPC_HANDLE_BYTES,), (_IPC_HANDLE_BYTES,), torch.uint8)
    ]


def test_first_use_graph_capture_requires_a_prebound_matching_stream(monkeypatch):
    runtime, _, _ = _make_runtime()
    runtime.device = torch.device("cuda", 0)
    stream_key = [77]
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records._is_current_stream_capturing",
        lambda device: True,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_selected_records._current_stream_key",
        lambda device: stream_key[0],
    )

    with pytest.raises(RuntimeError, match="before first-use graph capture"):
        runtime._check_stream()

    runtime._bind_stream_key(77)
    runtime._check_stream()
    stream_key[0] = 78
    with pytest.raises(RuntimeError, match="stream-affine"):
        runtime._check_stream()


def test_cleanup_rebinds_the_configured_cuda_device(monkeypatch):
    operations = []

    class _CleanupIPC:
        def cudaSetDevice(self, device):
            operations.append(("set_device", device))

        def cudaIpcCloseMemHandle(self, pointer):
            operations.append(("close", pointer))

        def cudaFree(self, pointer):
            operations.append(("free", pointer))

    runtime, _, _ = _make_runtime()
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


def test_cuda_source_uses_int64_pool_scaled_offsets_and_no_domain_policy():
    module_path = (
        Path(__file__).parents[2]
        / "sparkinfer"
        / "comm"
        / "pcie"
        / "pcie_selected_records.cu"
    )
    source = module_path.read_text(encoding="utf-8")
    python_source = module_path.with_suffix(".py").read_text(encoding="utf-8")

    assert "const int64_t local_record" in source
    assert "record_byte_offset(local_record, record_bytes)" in source
    assert "record_byte_offset(selected, record_bytes)" in source
    assert "const int64_t destination_offset" in source
    assert "destination_ptr" in source
    assert "scatter_records_kernel<int32_t, uint4>" in source
    assert "scatter_records_kernel<int32_t, uint8_t>" in source
    assert "barrier_all_peers_kernel" in source
    assert source.count("barrier_all_peers_kernel<<<") == 2
    assert "cudaMalloc" not in source
    assert "compact" not in source.lower()
    assert "remap" not in source.lower()
    assert "_broadcast_gather_object" not in python_source
    assert python_source.count("dist.all_gather_into_tensor") == 2
