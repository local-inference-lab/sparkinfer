from __future__ import annotations

import pytest
import torch

import b12x.gemm.wo_projection as wo_impl
from b12x.gemm import (
    WOProjectionBinding,
    WOProjectionInvRopeBinding,
    WOProjectionScratchCaps,
    empty_mxfp8_rows_for_dense_gemm,
    empty_wo_projection_workspace,
    plan_wo_projection_scratch,
)
from b12x.gemm.wo_projection import WOProjectionMXFP8Weights


def _weights(
    *,
    groups: int = 2,
    group_width: int = 128,
    rank: int = 64,
    hidden: int = 256,
) -> WOProjectionMXFP8Weights:
    return WOProjectionMXFP8Weights(
        wo_a=empty_mxfp8_rows_for_dense_gemm(
            rank,
            group_width,
            num_groups=groups,
            device="cpu",
        ),
        wo_b=empty_mxfp8_rows_for_dense_gemm(
            hidden,
            rank * groups,
            num_groups=1,
            device="cpu",
        ),
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )


def _plan():
    return plan_wo_projection_scratch(
        WOProjectionScratchCaps(
            device="cpu",
            max_tokens=4,
            groups=2,
            group_width=128,
            rank=64,
            hidden=256,
        )
    )


def test_wo_projection_scratch_plan_exposes_one_component_scratch_spec() -> None:
    plan = _plan()

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "wo_projection.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert specs[0].nbytes == plan.layout.nbytes


def test_wo_projection_scratch_plan_binds_live_shape(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()

    binding = plan.bind(scratch=scratch, source_tgd=source, weights=weights)

    assert isinstance(binding, WOProjectionBinding)
    assert binding.source_tgd is source
    assert binding.weights is weights
    assert not hasattr(binding, "workspace")
    assert binding.x_q.values.shape == (3, 128, 2)
    assert binding.tmp.shape == (3, 64, 2)
    assert binding.output.shape == (3, 256, 1)


def test_wo_projection_workspace_bind_returns_common_binding_type(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    workspace = empty_wo_projection_workspace(
        3,
        groups=2,
        group_width=128,
        rank=64,
        hidden=256,
        device="cpu",
    )
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()

    binding = workspace.bind(source_tgd=source, weights=weights)

    assert isinstance(binding, WOProjectionBinding)
    assert not hasattr(binding, "workspace")
    assert binding.x_q is workspace.x_q
    assert binding.tmp is workspace.tmp
    assert binding.tmp_q is workspace.tmp_q
    assert binding.output is workspace.output
    assert binding.source_tgd is source
    assert binding.weights is weights


def test_wo_projection_binding_supplies_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    workspace = empty_wo_projection_workspace(
        3,
        groups=2,
        group_width=128,
        rank=64,
        hidden=256,
        device="cpu",
    )
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()
    binding = workspace.bind(source_tgd=source, weights=weights)
    calls = {}

    def fake_quantize_a(source_tgd, *, out):
        calls["source_tgd"] = source_tgd
        calls["x_q_out"] = out
        return out

    def fake_wo_a(x_q, wo_a, *, out):
        calls["x_q"] = x_q
        calls["wo_a"] = wo_a
        calls["tmp_out"] = out
        return out

    def fake_quantize_b(tmp, *, out):
        calls["tmp"] = tmp
        calls["tmp_q_out"] = out
        return out

    def fake_wo_b(tmp_q, wo_b, *, out):
        calls["tmp_q"] = tmp_q
        calls["wo_b"] = wo_b
        calls["output_out"] = out
        out.zero_()
        return out

    monkeypatch.setattr(wo_impl, "quantize_wo_a_input_mxfp8", fake_quantize_a)
    monkeypatch.setattr(wo_impl, "wo_a_dense_gemm_mxfp8", fake_wo_a)
    monkeypatch.setattr(wo_impl, "quantize_wo_b_input_mxfp8", fake_quantize_b)
    monkeypatch.setattr(wo_impl, "wo_b_dense_gemm_mxfp8", fake_wo_b)

    out = wo_impl.wo_projection_mxfp8(binding=binding)

    assert calls["source_tgd"] is source
    assert calls["x_q_out"] is workspace.x_q
    assert calls["tmp_out"] is workspace.tmp
    assert calls["tmp_q_out"] is workspace.tmp_q
    assert calls["output_out"] is workspace.output
    assert out.shape == (3, 256)


def test_wo_projection_inv_rope_binding_supplies_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    o = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    positions = torch.empty((3,), dtype=torch.int64)
    cos_sin_cache = torch.empty((16, 32), dtype=torch.bfloat16)
    weights = _weights()
    binding = plan.bind_inv_rope(
        scratch=scratch,
        o=o,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        heads_per_group=1,
        nope_dim=96,
        rope_dim=32,
        return_3d=True,
    )
    calls = {}

    def fake_quantize_a(o_arg, positions_arg, cos_sin_cache_arg, **kwargs):
        calls["o"] = o_arg
        calls["positions"] = positions_arg
        calls["cos_sin_cache"] = cos_sin_cache_arg
        calls["x_q_out"] = kwargs["out"]
        return kwargs["out"]

    monkeypatch.setattr(wo_impl, "quantize_wo_a_input_inv_rope_mxfp8", fake_quantize_a)
    monkeypatch.setattr(wo_impl, "wo_a_dense_gemm_mxfp8", lambda x_q, wo_a, *, out: out)
    monkeypatch.setattr(wo_impl, "quantize_wo_b_input_mxfp8", lambda tmp, *, out: out)

    def fake_wo_b(tmp_q, wo_b, *, out):
        out.zero_()
        return out

    monkeypatch.setattr(wo_impl, "wo_b_dense_gemm_mxfp8", fake_wo_b)

    out = wo_impl.wo_projection_inv_rope_mxfp8(binding=binding)

    assert isinstance(binding, WOProjectionInvRopeBinding)
    assert not hasattr(binding, "workspace")
    assert calls["o"] is o
    assert calls["positions"] is positions
    assert calls["cos_sin_cache"] is cos_sin_cache
    assert calls["x_q_out"] is binding.x_q
    assert out.shape == (3, 256, 1)


def test_wo_projection_binding_owns_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    workspace = empty_wo_projection_workspace(
        3,
        groups=2,
        group_width=128,
        rank=64,
        hidden=256,
        device="cpu",
    )
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()
    binding = workspace.bind(source_tgd=source, weights=weights)

    with pytest.raises(ValueError, match="binding owns source_tgd"):
        wo_impl.wo_projection_mxfp8(
            source,
            weights,
            workspace,
            binding=binding,
        )
