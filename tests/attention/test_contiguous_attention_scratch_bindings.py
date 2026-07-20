from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import sparkinfer.attention._shared.contiguous.api as contig
from sparkinfer.attention._shared.contiguous import (
    AttentionBinding,
    VarlenAttentionBinding,
    plan_attention_scratch,
    plan_varlen_attention_scratch,
)


class _Compiled:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


def _attention_plan(*, compiled: _Compiled | None = None):
    return SimpleNamespace(
        q_shape=(2, 3, 4, 64),
        k_shape=(2, 5, 2, 64),
        v_shape=(2, 5, 2, 64),
        device=torch.device("cpu"),
        device_index=0,
        dtype=torch.bfloat16,
        causal=True,
        window_size_left=-1,
        window_size_right=-1,
        has_attention_sink_bias=True,
        tile_m=128,
        tile_n=64,
        key=("attention",),
        compiled=compiled or _Compiled(),
        cutlass_dtype=object,
    )


def _varlen_plan(*, compiled: _Compiled | None = None):
    return SimpleNamespace(
        q_shape=(5, 4, 64),
        k_shape=(7, 2, 64),
        v_shape=(7, 2, 64),
        cu_seqlens_q_shape=(3,),
        cu_seqlens_k_shape=(3,),
        device=torch.device("cpu"),
        device_index=0,
        dtype=torch.bfloat16,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        has_attention_sink_bias=True,
        tile_m=128,
        tile_n=64,
        max_seqlen_q=3,
        max_seqlen_k=4,
        key=("varlen",),
        compiled=compiled or _Compiled(),
        cutlass_dtype=object,
    )


def _patch_attention_validation(monkeypatch) -> None:
    def fake_validate(q, k, v):
        return tuple(q.shape), tuple(k.shape), tuple(v.shape), q.device, q.dtype

    monkeypatch.setattr(contig, "_validate_forward_inputs", fake_validate)


def _patch_varlen_validation(monkeypatch) -> None:
    def fake_validate(q, k, v, cu_seqlens_q, cu_seqlens_k):
        return (
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            tuple(cu_seqlens_q.shape),
            tuple(cu_seqlens_k.shape),
            q.device,
            q.dtype,
        )

    monkeypatch.setattr(contig, "_validate_varlen_inputs", fake_validate)


def _patch_launch(monkeypatch) -> None:
    monkeypatch.setattr(contig, "make_ptr", lambda dtype, ptr, *args, **kwargs: ptr)
    monkeypatch.setattr(contig, "current_cuda_stream", lambda: 0)


def test_attention_scratch_plan_exposes_one_opaque_scratch_spec() -> None:
    plan = plan_attention_scratch(_attention_plan())

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "contiguous_attention.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]


def test_attention_scratch_plan_binds_live_tensors(monkeypatch) -> None:
    _patch_attention_validation(monkeypatch)
    plan = plan_attention_scratch(_attention_plan())
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty(plan.plan.q_shape, dtype=torch.bfloat16)
    k = torch.empty(plan.plan.k_shape, dtype=torch.bfloat16)
    v = torch.empty(plan.plan.v_shape, dtype=torch.bfloat16)
    sink = torch.empty((4,), dtype=torch.float32)

    binding = plan.bind(scratch=scratch, q=q, k=k, v=v, attention_sink_bias=sink)

    assert isinstance(binding, AttentionBinding)
    assert binding.q is q
    assert binding.k is k
    assert binding.v is v
    assert binding.output.shape == plan.plan.q_shape
    assert binding.lse.shape == (2, 4, 3)


def test_attention_scratch_plan_binding_maps_caller_owned_scratch(monkeypatch) -> None:
    _patch_attention_validation(monkeypatch)
    plan = plan_attention_scratch(_attention_plan())
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty(plan.plan.q_shape, dtype=torch.bfloat16)
    k = torch.empty(plan.plan.k_shape, dtype=torch.bfloat16)
    v = torch.empty(plan.plan.v_shape, dtype=torch.bfloat16)
    sink = torch.empty((4,), dtype=torch.float32)

    binding = plan.bind(scratch=scratch, q=q, k=k, v=v, attention_sink_bias=sink)

    assert isinstance(binding, AttentionBinding)
    assert binding.output.shape == plan.plan.q_shape
    assert binding.output.untyped_storage().data_ptr() == scratch.untyped_storage().data_ptr()


def test_attention_scratch_plan_bind_returns_common_binding_type(monkeypatch) -> None:
    _patch_attention_validation(monkeypatch)
    plan = plan_attention_scratch(_attention_plan())
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty(plan.plan.q_shape, dtype=torch.bfloat16)
    k = torch.empty(plan.plan.k_shape, dtype=torch.bfloat16)
    v = torch.empty(plan.plan.v_shape, dtype=torch.bfloat16)
    sink = torch.empty((4,), dtype=torch.float32)

    binding = plan.bind(scratch=scratch, q=q, k=k, v=v, attention_sink_bias=sink)

    assert isinstance(binding, AttentionBinding)
    assert binding.output.shape == plan.plan.q_shape
    assert binding.lse.shape == (2, 4, 3)
    assert binding.plan is plan.plan


def test_attention_binding_supplies_runtime_tensors(monkeypatch) -> None:
    _patch_attention_validation(monkeypatch)
    _patch_launch(monkeypatch)
    compiled = _Compiled()
    plan = plan_attention_scratch(_attention_plan(compiled=compiled))
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty(plan.plan.q_shape, dtype=torch.bfloat16)
    k = torch.empty(plan.plan.k_shape, dtype=torch.bfloat16)
    v = torch.empty(plan.plan.v_shape, dtype=torch.bfloat16)
    sink = torch.empty((4,), dtype=torch.float32)
    binding = plan.bind(scratch=scratch, q=q, k=k, v=v, attention_sink_bias=sink)

    out, lse = contig.sparkinfer_attention_forward(binding=binding)

    assert out is binding.output
    assert lse is binding.lse
    assert len(compiled.calls) == 1
    call = compiled.calls[0]
    assert call[0] == q.data_ptr()
    assert call[1] == k.data_ptr()
    assert call[2] == v.data_ptr()
    assert call[3] == binding.output.data_ptr()
    assert call[4] == binding.lse.data_ptr()
    assert call[5] == sink.data_ptr()


def test_attention_binding_owns_runtime_tensors(monkeypatch) -> None:
    _patch_attention_validation(monkeypatch)
    plan = plan_attention_scratch(_attention_plan())
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty(plan.plan.q_shape, dtype=torch.bfloat16)
    k = torch.empty(plan.plan.k_shape, dtype=torch.bfloat16)
    v = torch.empty(plan.plan.v_shape, dtype=torch.bfloat16)
    sink = torch.empty((4,), dtype=torch.float32)
    binding = plan.bind(scratch=scratch, q=q, k=k, v=v, attention_sink_bias=sink)

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        contig.sparkinfer_attention_forward(q, k, v, binding=binding)


def test_varlen_attention_scratch_plan_binds_live_tensors(monkeypatch) -> None:
    _patch_varlen_validation(monkeypatch)
    plan = plan_varlen_attention_scratch(_varlen_plan())
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty(plan.plan.q_shape, dtype=torch.bfloat16)
    k = torch.empty(plan.plan.k_shape, dtype=torch.bfloat16)
    v = torch.empty(plan.plan.v_shape, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    sink = torch.empty((4,), dtype=torch.float32)

    binding = plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        attention_sink_bias=sink,
    )

    assert isinstance(binding, VarlenAttentionBinding)
    assert binding.cu_seqlens_q is cu_q
    assert binding.cu_seqlens_k is cu_q
    assert binding.output.shape == plan.plan.q_shape
    assert binding.lse.shape == (4, 5)


def test_varlen_attention_scratch_plan_binding_maps_caller_owned_scratch(monkeypatch) -> None:
    _patch_varlen_validation(monkeypatch)
    plan = plan_varlen_attention_scratch(_varlen_plan())
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty(plan.plan.q_shape, dtype=torch.bfloat16)
    k = torch.empty(plan.plan.k_shape, dtype=torch.bfloat16)
    v = torch.empty(plan.plan.v_shape, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    sink = torch.empty((4,), dtype=torch.float32)

    binding = plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        attention_sink_bias=sink,
    )

    assert isinstance(binding, VarlenAttentionBinding)
    assert binding.output.shape == plan.plan.q_shape
    assert binding.output.untyped_storage().data_ptr() == scratch.untyped_storage().data_ptr()


def test_varlen_attention_binding_supplies_runtime_tensors(monkeypatch) -> None:
    _patch_varlen_validation(monkeypatch)
    _patch_launch(monkeypatch)
    compiled = _Compiled()
    plan = plan_varlen_attention_scratch(_varlen_plan(compiled=compiled))
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty(plan.plan.q_shape, dtype=torch.bfloat16)
    k = torch.empty(plan.plan.k_shape, dtype=torch.bfloat16)
    v = torch.empty(plan.plan.v_shape, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)
    sink = torch.empty((4,), dtype=torch.float32)
    binding = plan.bind(
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        attention_sink_bias=sink,
    )

    out, lse = contig.sparkinfer_varlen_attention_forward(binding=binding)

    assert out is binding.output
    assert lse is binding.lse
    assert len(compiled.calls) == 1
    call = compiled.calls[0]
    assert call[0] == q.data_ptr()
    assert call[1] == k.data_ptr()
    assert call[2] == v.data_ptr()
    assert call[3] == binding.output.data_ptr()
    assert call[4] == binding.lse.data_ptr()
    assert call[5] == cu_q.data_ptr()
    assert call[6] == cu_k.data_ptr()
    assert call[7] == sink.data_ptr()
