from __future__ import annotations

import pytest
import torch

from b12x.distributed.pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    PCIeOneshotAllReducePool,
    _compute_crossover_size,
    parse_pcie_oneshot_max_size,
)


class _FakeExt:
    def __init__(self):
        self.init_calls = []
        self.register_pcie_buffers_calls = []
        self.register_buffer_calls = []
        self.all_reduce_calls = []
        self.all_reduce_fused_add_rms_norm_calls = []
        self.dispose_calls = []
        self.register_graph_buffers_calls = []
        self.handle_bytes = [1, 2, 3]
        self.offsets = [0, 64]

    def init_custom_ar(self, signal_ptrs, rank_data, rank):
        self.init_calls.append((tuple(signal_ptrs), rank_data.device.type, rank))
        return 12345

    def register_pcie_buffers(self, ptr, ptrs0, ptrs1):
        self.register_pcie_buffers_calls.append((ptr, tuple(ptrs0), tuple(ptrs1)))

    def register_buffer(self, ptr, peer_input_ptrs):
        self.register_buffer_calls.append((ptr, tuple(peer_input_ptrs)))

    def all_reduce(self, ptr, inp, out, reg_buffer, reg_buffer_bytes):
        self.all_reduce_calls.append(
            (
                ptr,
                int(inp.data_ptr()),
                int(out.data_ptr()),
                reg_buffer,
                reg_buffer_bytes,
            )
        )
        out.copy_(inp)

    def all_reduce_fused_add_rms_norm(
        self,
        ptr,
        inp,
        residual,
        weight,
        out,
        residual_out,
        epsilon,
        reg_buffer,
        reg_buffer_bytes,
    ):
        self.all_reduce_fused_add_rms_norm_calls.append(
            (
                ptr,
                int(inp.data_ptr()),
                int(residual.data_ptr()),
                int(weight.data_ptr()),
                int(out.data_ptr()),
                int(residual_out.data_ptr()),
                epsilon,
                reg_buffer,
                reg_buffer_bytes,
            )
        )
        residual_value = residual.clone()
        residual_out.copy_(inp)
        residual_out.add_(residual_value)
        variance = residual_out.float().square().mean(dim=-1, keepdim=True)
        normalized = residual_out.float() * torch.rsqrt(variance + epsilon)
        out.copy_((normalized * weight.float()).to(out.dtype))

    def dispose(self, ptr):
        self.dispose_calls.append(ptr)

    def meta_size(self):
        return 256

    def get_graph_buffer_ipc_meta(self, ptr):
        return list(self.handle_bytes), list(self.offsets)

    def register_graph_buffers(self, ptr, handles, offsets):
        self.register_graph_buffers_calls.append((ptr, handles, offsets))


def _make_runtime(
    *,
    rank=0,
    world_size=2,
    exchange_group=None,
    max_size=8 * 1024 * 1024,
    eager=False,
    ext=None,
    stream_affine=True,
):
    ext = ext or _FakeExt()
    kwargs = {}
    if eager:
        kwargs["eager_buffer_ptrs0"] = tuple(range(200, 200 + world_size))
        kwargs["eager_buffer_ptrs1"] = tuple(range(300, 300 + world_size))
    return PCIeOneshotAllReduce(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        signal_ptrs=tuple(range(100, 100 + world_size)),
        exchange_group=exchange_group,
        max_size=max_size,
        ext_module=ext,
        stream_affine=stream_affine,
        **kwargs,
    )


def test_parse_pcie_oneshot_max_size_accepts_auto_and_suffixes():
    assert parse_pcie_oneshot_max_size(None) is None
    assert parse_pcie_oneshot_max_size("auto") is None
    assert parse_pcie_oneshot_max_size("64KB") == 64 * 1024
    assert parse_pcie_oneshot_max_size("2m") == 2 * 1024 * 1024
    assert parse_pcie_oneshot_max_size(4096) == 4096


def test_compute_crossover_size_runs_fine_sweep():
    seen_sizes = []

    def benchmark(size_bytes: int) -> tuple[float, float]:
        seen_sizes.append(size_bytes)
        if size_bytes <= 48 * 1024:
            return 1.0, 2.0
        return 3.0, 2.0

    crossover, results = _compute_crossover_size(
        benchmark,
        ceiling_bytes=64 * 1024,
        fine_step_bytes=8 * 1024,
    )

    assert crossover == 48 * 1024
    assert 40 * 1024 in seen_sizes
    assert 48 * 1024 in seen_sizes
    assert 56 * 1024 in seen_sizes
    assert results[-1].size_bytes == 64 * 1024


def test_register_buffer_is_idempotent_for_same_mapping():
    runtime = _make_runtime()
    ext = runtime._ext

    runtime.register_buffer((111, 222))
    runtime.register_buffer((111, 222))

    assert ext.register_buffer_calls == [(12345, (111, 222))]


def test_register_buffer_rejects_mismatched_mapping_for_same_local_ptr():
    runtime = _make_runtime()

    runtime.register_buffer((111, 222))

    with pytest.raises(ValueError, match="already registered"):
        runtime.register_buffer((111, 333))


def test_all_reduce_registers_explicit_peer_ptrs_once():
    runtime = _make_runtime()
    ext = runtime._ext
    inp = torch.arange(8, dtype=torch.bfloat16)

    out0 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))
    out1 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))

    assert torch.equal(out0, inp)
    assert torch.equal(out1, inp)
    assert ext.register_buffer_calls == [(12345, (inp.data_ptr(), 222))]
    assert len(ext.all_reduce_calls) == 2


def test_all_reduce_requires_registration_without_eager_buffers():
    runtime = _make_runtime()
    inp = torch.arange(8, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="peer_input_ptrs are required"):
        runtime.all_reduce(inp)


def test_eager_buffers_allow_all_reduce_without_peer_ptrs():
    runtime = _make_runtime(eager=True)
    ext = runtime._ext
    inp = torch.arange(8, dtype=torch.bfloat16)

    out = runtime.all_reduce(inp)

    assert torch.equal(out, inp)
    assert ext.register_pcie_buffers_calls == [(12345, (200, 201), (300, 301))]
    assert ext.register_buffer_calls == []
    assert len(ext.all_reduce_calls) == 1


def test_fused_add_rms_norm_returns_norm_and_residual_outputs():
    runtime = _make_runtime(eager=True)
    ext = runtime._ext
    inp = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 8
    residual = torch.linspace(-0.5, 0.5, 16, dtype=torch.bfloat16).reshape(2, 8)
    weight = torch.linspace(0.75, 1.25, 8, dtype=torch.bfloat16)

    out, residual_out = runtime.all_reduce_fused_add_rms_norm(
        inp,
        residual,
        weight,
        1e-6,
    )

    expected_residual = inp + residual
    variance = expected_residual.float().square().mean(dim=-1, keepdim=True)
    expected_out = (
        expected_residual.float() * torch.rsqrt(variance + 1e-6) * weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual_out, expected_residual)
    torch.testing.assert_close(out, expected_out)
    assert len(ext.all_reduce_fused_add_rms_norm_calls) == 1


def test_fused_add_rms_norm_supports_inplace_residual_output():
    runtime = _make_runtime(eager=True)
    inp = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)
    residual = torch.ones_like(inp)
    original_residual_ptr = residual.data_ptr()

    _, residual_out = runtime.all_reduce_fused_add_rms_norm(
        inp,
        residual,
        torch.ones(8, dtype=torch.bfloat16),
        1e-6,
        residual_out=residual,
    )

    assert residual_out.data_ptr() == original_residual_ptr
    torch.testing.assert_close(residual_out, inp + 1)


def test_fused_add_rms_norm_requires_pack_aligned_rows():
    runtime = _make_runtime(eager=True)
    inp = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)

    with pytest.raises(ValueError, match="last input dimension"):
        runtime.all_reduce_fused_add_rms_norm(
            inp,
            torch.zeros_like(inp),
            torch.ones(4, dtype=torch.bfloat16),
            1e-6,
        )


def test_fused_add_rms_norm_validates_weight():
    runtime = _make_runtime(eager=True)
    inp = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)

    with pytest.raises(ValueError, match="weight tensor"):
        runtime.all_reduce_fused_add_rms_norm(
            inp,
            torch.zeros_like(inp),
            torch.ones(8, dtype=torch.float32),
            1e-6,
        )


def test_world_size_10_is_supported_by_eager_pool():
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(rank=3, world_size=10, eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=3,
        world_size=10,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )

    runtime = pool.for_stream()

    assert runtime.world_size == 10
    assert runtime._ext.init_calls == [
        ((100, 101, 102, 103, 104, 105, 106, 107, 108, 109), "cpu", 3)
    ]
    assert runtime._ext.register_pcie_buffers_calls == [
        (
            12345,
            (200, 201, 202, 203, 204, 205, 206, 207, 208, 209),
            (300, 301, 302, 303, 304, 305, 306, 307, 308, 309),
        )
    ]
    assert created == [(None, runtime)]


def test_runtime_rejects_reuse_from_another_stream_key(monkeypatch):
    runtime = _make_runtime(eager=True)
    inp = torch.arange(8, dtype=torch.bfloat16)
    state = {"stream_key": 11}

    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._current_stream_key",
        lambda device, stream=None: state["stream_key"],
    )

    runtime.all_reduce(inp)
    state["stream_key"] = 22

    with pytest.raises(RuntimeError, match="stream-affine"):
        runtime.all_reduce(inp)


def test_runtime_can_disable_stream_affinity(monkeypatch):
    runtime = _make_runtime(eager=True, stream_affine=False)
    inp = torch.arange(8, dtype=torch.bfloat16)
    state = {"stream_key": 11}

    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._current_stream_key",
        lambda device, stream=None: state["stream_key"],
    )

    runtime.all_reduce(inp)
    state["stream_key"] = 22
    runtime.all_reduce(inp)


def test_should_allreduce_checks_device_dtype_size_alignment_and_contiguity():
    runtime = _make_runtime(max_size=16)

    good = torch.arange(8, dtype=torch.bfloat16)
    assert runtime.should_allreduce(good) is True
    assert runtime.should_allreduce(torch.arange(4, dtype=torch.int32)) is False
    assert runtime.should_allreduce(torch.arange(16, dtype=torch.bfloat16)) is False
    assert runtime.should_allreduce(torch.arange(7, dtype=torch.bfloat16)) is False
    assert runtime.should_allreduce(torch.arange(16, dtype=torch.bfloat16)[::2]) is False


def test_graph_buffer_api_exposes_explicit_registration_hooks():
    runtime = _make_runtime()
    ext = runtime._ext

    assert runtime.get_graph_buffer_ipc_meta() == ([1, 2, 3], [0, 64])

    runtime.register_graph_buffers_from_ranks(
        ([1, 2, 3], [4, 5, 6]),
        ([0, 64], [8, 72]),
    )

    assert ext.register_graph_buffers_calls == [
        (12345, [[1, 2, 3], [4, 5, 6]], [[0, 64], [8, 72]])
    ]


def test_allocate_shared_buffer_cleans_up_on_failed_peer_open(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.closed = []
            self.freed = []

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            pass

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            if handle == b"remote1":
                raise RuntimeError("open failed")
            return 2000

        def cudaIpcCloseMemHandle(self, ptr):
            self.closed.append(ptr)

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 3)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._broadcast_gather_object",
        lambda local_object, group: [local_object, b"remote0", b"remote1"],
    )

    with pytest.raises(RuntimeError, match="peer rank 2"):
        PCIeOneshotAllReduce._allocate_shared_buffer(
            object(), 256, zero_fill=True, ipc=ipc
        )

    assert ipc.closed == [2000]
    assert ipc.freed == [1000]


def test_eager_channel_buffers_use_single_ipc_slab(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.malloc_sizes = []
            self.memsets = []
            self.opened = []

        def cudaMalloc(self, size):
            self.malloc_sizes.append(size)
            return 1000

        def cudaMemset(self, ptr, value, size):
            self.memsets.append((ptr, value, size))

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            ptr = {b"remote0": 2000, b"remote1": 3000}[handle]
            self.opened.append(ptr)
            return ptr

        def cudaIpcCloseMemHandle(self, ptr):
            raise AssertionError("success path should not close remote ptrs")

        def cudaFree(self, ptr):
            raise AssertionError("success path should not free local ptr")

    ipc = FakeIPC()
    exchange_group = object()
    signal_bytes = 300
    eager_bytes = 128
    eager0_offset = IPC_SLAB_ALIGNMENT * 2
    eager1_offset = eager0_offset + IPC_SLAB_ALIGNMENT

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 3)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._broadcast_gather_object",
        lambda local_object, group: [local_object, b"remote0", b"remote1"],
    )

    buffers = PCIeOneshotAllReduce._allocate_eager_channel_buffers(
        exchange_group,
        signal_bytes=signal_bytes,
        eager_buffer_bytes=eager_bytes,
        ipc=ipc,
    )

    assert ipc.malloc_sizes == [eager1_offset + eager_bytes]
    assert ipc.memsets == [(1000, 0, eager1_offset + eager_bytes)]
    assert ipc.opened == [2000, 3000]
    assert buffers.owned_buffer.local_ptr == 1000
    assert buffers.owned_buffer.remote_ptrs == (2000, 3000)
    assert buffers.signal_ptrs == (1000, 2000, 3000)
    assert buffers.eager0_ptrs == (
        1000 + eager0_offset,
        2000 + eager0_offset,
        3000 + eager0_offset,
    )
    assert buffers.eager1_ptrs == (
        1000 + eager1_offset,
        2000 + eager1_offset,
        3000 + eager1_offset,
    )


def test_eager_channel_buffers_cleanup_when_slab_zero_fails(monkeypatch):
    class FakeIPC:
        def __init__(self):
            self.closed = []
            self.freed = []

        def cudaMalloc(self, size):
            return 1000

        def cudaMemset(self, ptr, value, size):
            raise RuntimeError("memset failed")

        def cudaIpcGetMemHandleBytes(self, ptr):
            return b"local"

        def cudaIpcOpenMemHandleBytes(self, handle):
            return {b"remote0": 2000, b"remote1": 3000}[handle]

        def cudaIpcCloseMemHandle(self, ptr):
            self.closed.append(ptr)

        def cudaFree(self, ptr):
            self.freed.append(ptr)

    ipc = FakeIPC()

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 3)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._broadcast_gather_object",
        lambda local_object, group: [local_object, b"remote0", b"remote1"],
    )

    with pytest.raises(RuntimeError, match="memset failed"):
        PCIeOneshotAllReduce._allocate_eager_channel_buffers(
            object(),
            signal_bytes=300,
            eager_buffer_bytes=128,
            ipc=ipc,
        )

    assert ipc.closed == []
    assert ipc.freed == [1000]


def test_register_graph_buffers_uses_exchange_group_broadcast(monkeypatch):
    remote_meta = {
        0: ([1, 2, 3], [0, 64]),
        1: ([9, 8, 7], [16, 80]),
    }

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr("torch.distributed.get_process_group_ranks", lambda group=None: [0, 1])
    monkeypatch.setattr("b12x.distributed.pcie_oneshot._object_broadcast_device", lambda group: "cpu")

    def fake_broadcast(object_list, src, group=None, device=None):
        object_list[0] = remote_meta[src]

    monkeypatch.setattr("torch.distributed.broadcast_object_list", fake_broadcast)

    runtime = _make_runtime(exchange_group=object())
    ext = runtime._ext
    runtime.register_graph_buffers()

    assert ext.register_graph_buffers_calls == [
        (12345, [[1, 2, 3], [9, 8, 7]], [[0, 64], [16, 80]])
    ]


def test_register_graph_buffers_noops_when_no_rank_registered_buffers(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr("torch.distributed.get_process_group_ranks", lambda group=None: [0, 1])
    monkeypatch.setattr("b12x.distributed.pcie_oneshot._object_broadcast_device", lambda group: "cpu")
    monkeypatch.setattr(
        "torch.distributed.broadcast_object_list",
        lambda object_list, src, group=None, device=None: object_list.__setitem__(0, ([], [])),
    )

    runtime = _make_runtime(exchange_group=object())
    ext = runtime._ext
    ext.handle_bytes = []
    ext.offsets = []

    runtime.register_graph_buffers()

    assert ext.register_graph_buffers_calls == []


def test_capture_registers_graph_buffers_after_context(monkeypatch):
    runtime = _make_runtime(exchange_group=object())
    calls = []

    monkeypatch.setattr(runtime, "register_graph_buffers", lambda: calls.append("registered"))

    with runtime.capture():
        pass

    assert calls == ["registered"]


def test_eager_capture_skips_graph_buffer_registration(monkeypatch):
    runtime = _make_runtime(eager=True, exchange_group=object())
    calls = []

    monkeypatch.setattr(runtime, "register_graph_buffers", lambda: calls.append("registered"))

    with runtime.capture():
        pass

    assert calls == []


def test_pool_creates_distinct_channels_per_stream_key(monkeypatch):
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )

    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._current_stream_key",
        lambda device, stream=None: 7 if stream is None else int(stream),
    )

    ch7 = pool.for_stream()
    ch8 = pool.for_stream(8)

    assert pool.for_stream() is ch7
    assert pool.for_stream(8) is ch8
    assert ch7 is not ch8
    assert [entry[0] for entry in created] == [7, 8]


def test_pool_reuses_single_channel_across_stream_keys(monkeypatch):
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        single_channel=True,
        channel_factory=make_channel,
    )

    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._current_stream_key",
        lambda device, stream=None: 7 if stream is None else int(stream),
    )

    ch7 = pool.for_stream()
    ch8 = pool.for_stream(8)

    assert ch7 is ch8
    assert [entry[0] for entry in created] == [None]


def test_pool_requires_precreated_channel_during_capture(monkeypatch):
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )

    monkeypatch.setattr("b12x.distributed.pcie_oneshot._current_stream_key", lambda device, stream=None: 7)
    monkeypatch.setattr("b12x.distributed.pcie_oneshot._is_current_stream_capturing", lambda device: True)

    with pytest.raises(RuntimeError, match="before capture starts"):
        pool.for_stream()

    pool._channels[7] = _make_runtime(eager=True)

    assert pool.for_stream() is pool._channels[7]


def test_nested_capture_reuses_its_outer_channel(monkeypatch):
    created = []
    current_stream = [7]
    capturing = [False]

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    with pool.capture(7) as target_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is target_channel
        capturing[0] = False

    with pool.capture(8) as draft_channel:
        capturing[0] = True
        current_stream[0] = 80
        assert pool.for_stream() is draft_channel
        capturing[0] = False

    assert target_channel is not draft_channel
    assert pool._channels[70] is target_channel
    assert pool._channels[80] is draft_channel
    assert [entry[0] for entry in created] == [7, 8]


def test_reused_capture_stream_keys_get_distinct_channels(monkeypatch):
    created = []
    current_stream = [7]
    capturing = [False]

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    with pool.capture(7) as target_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is target_channel
        capturing[0] = False

    # CUDA may recycle both handles for the next graph manager. Neither stale
    # mapping may make the draft graph retain the target graph's IPC channel.
    with pool.capture(7) as draft_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is draft_channel
        capturing[0] = False

    assert target_channel is not draft_channel
    assert pool._channels[7] is draft_channel
    assert pool._channels[70] is draft_channel
    assert target_channel in pool._all_channels
    assert draft_channel in pool._all_channels
    assert [entry[0] for entry in created] == [7, 7]

    pool.close()
    assert target_channel._ext.dispose_calls == [12345]
    assert draft_channel._ext.dispose_calls == [12345]


def test_pool_rolls_back_throwaway_capture_channels(monkeypatch):
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._current_stream_key",
        lambda device, stream=None: 3 if stream is None else int(stream),
    )

    eager_channel = pool.for_stream()
    checkpoint = pool.checkpoint_channels()
    with pool.capture(7) as profile_channel:
        pass

    pool.rollback_channels(checkpoint)

    assert pool._all_channels == [eager_channel]
    assert pool._channels == {3: eager_channel}
    assert profile_channel._ext.dispose_calls == [12345]
    assert eager_channel._ext.dispose_calls == []


def test_pool_rejects_channel_rollback_during_capture():
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )
    checkpoint = pool.checkpoint_channels()

    with pool.capture(7), pytest.raises(RuntimeError, match="during capture"):
        pool.rollback_channels(checkpoint)
