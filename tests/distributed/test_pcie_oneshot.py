from __future__ import annotations

import types

import torch

from b12x.distributed.pcie_oneshot import PCIeOneshotAllReduce


class _FakeIPC:
    def __init__(self):
        self.memcpy_calls = []

    def cudaMemcpyAsync(self, *, dst, src, count, stream):
        self.memcpy_calls.append((dst, src, count, stream))


def test_from_ipc_selects_eager_buffers_and_copies(monkeypatch):
    runtime = PCIeOneshotAllReduce(
        rank=1,
        world_size=4,
        device=torch.device("cpu"),
        signal_ptrs=(100, 200, 300, 400),
        eager_buffer_ptrs0=(10, 20, 30, 40),
        eager_buffer_ptrs1=(11, 21, 31, 41),
        ipc=_FakeIPC(),
    )

    compiled_calls = []

    def fake_compiled(*args):
        compiled_calls.append(args)

    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._get_gemma_kernel",
        lambda *args, **kwargs: fake_compiled,
    )
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot.current_cuda_stream",
        lambda: 1234,
    )
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot.make_ptr",
        lambda dtype, ptr, mem_space, assumed_align=None: ("ptr", int(ptr), assumed_align),
    )
    monkeypatch.setattr(
        "torch.cuda.current_stream",
        lambda device=None: types.SimpleNamespace(cuda_stream=77),
    )

    inp = torch.empty((2, 16), dtype=torch.bfloat16)
    residual = torch.empty_like(inp)
    weight = torch.empty((16,), dtype=torch.bfloat16)

    out, residual_out = runtime.allreduce_gemma_rmsnorm(inp, residual, weight, 1e-6)

    assert out.shape == inp.shape
    assert residual_out.shape == inp.shape
    assert runtime._ipc.memcpy_calls == [(20, int(inp.data_ptr()), inp.numel() * inp.element_size(), 77)]
    assert len(compiled_calls) == 1


def test_from_ipc_uses_explicit_peer_input_ptrs(monkeypatch):
    runtime = PCIeOneshotAllReduce(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        signal_ptrs=(100, 200),
        ipc=_FakeIPC(),
    )

    compiled_calls = []

    def fake_compiled(*args):
        compiled_calls.append(args)

    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._get_gemma_kernel",
        lambda *args, **kwargs: fake_compiled,
    )
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot.current_cuda_stream",
        lambda: 1234,
    )
    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot.make_ptr",
        lambda dtype, ptr, mem_space, assumed_align=None: ("ptr", int(ptr), assumed_align),
    )

    inp = torch.empty((1, 8), dtype=torch.bfloat16)
    residual = torch.empty_like(inp)
    weight = torch.empty((8,), dtype=torch.bfloat16)

    runtime.allreduce_gemma_rmsnorm(
        inp,
        residual,
        weight,
        1e-6,
        peer_input_ptrs=(111, 222),
    )

    assert runtime._ipc.memcpy_calls == []
    assert len(compiled_calls) == 1
