from __future__ import annotations

import pytest
import torch

import b12x.gemm.wo_projection as wo_impl
from b12x.gemm import (
    WOProjectionBinding,
    WOProjectionInvRopeBinding,
    WOProjectionScratchCaps,
    empty_mxfp8_rows_for_dense_gemm,
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


def test_wo_projection_plan_binding_maps_scratch_views(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()

    binding = plan.bind(scratch=scratch, source_tgd=source, weights=weights)

    assert isinstance(binding, WOProjectionBinding)
    assert not hasattr(binding, "workspace")
    assert binding.x_q.values.data_ptr() == scratch.data_ptr()
    assert binding.tmp.shape == (3, 64, 2)
    assert binding.tmp_q.values.shape == (3, 128)
    assert binding.output.shape == (3, 256, 1)
    assert binding.source_tgd is source
    assert binding.weights is weights


def test_wo_projection_binding_supplies_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()
    binding = plan.bind(scratch=scratch, source_tgd=source, weights=weights)
    calls = {}

    def fake_quantize_a(source_tgd, *, out):
        calls["source_tgd"] = source_tgd
        calls["x_q_out"] = out
        return out

    def fake_wo_a(x_q, wo_a, *, out, expected_m=None, **kwargs):
        calls["x_q"] = x_q
        calls["wo_a"] = wo_a
        calls["tmp_out"] = out
        return out

    def fake_quantize_b(tmp, *, out):
        calls["tmp"] = tmp
        calls["tmp_q_out"] = out
        return out

    def fake_wo_b(tmp_q, wo_b, *, out, expected_m=None, **kwargs):
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
    assert calls["x_q_out"] is binding.x_q
    assert calls["tmp_out"] is binding.tmp
    assert calls["tmp_q_out"] is binding.tmp_q
    assert calls["output_out"] is binding.output
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

    def fake_fused(
        o_arg,
        positions_arg,
        cos_sin_cache_arg,
        wo_a_values,
        wo_a_scale_rows,
        wo_a_scale_mma,
        wo_b_values,
        wo_b_scale_rows,
        wo_b_scale_mma,
        groups,
        group_width,
        rank,
        hidden,
        heads_per_group,
        nope_dim,
        rope_dim,
        expected_m,
        stream_int,
    ):
        calls["o"] = o_arg
        calls["positions"] = positions_arg
        calls["cos_sin_cache"] = cos_sin_cache_arg
        calls["wo_a_values"] = wo_a_values
        calls["wo_a_scale_rows"] = wo_a_scale_rows
        calls["wo_a_scale_mma"] = wo_a_scale_mma
        calls["wo_b_values"] = wo_b_values
        calls["wo_b_scale_rows"] = wo_b_scale_rows
        calls["wo_b_scale_mma"] = wo_b_scale_mma
        calls["groups"] = groups
        calls["group_width"] = group_width
        calls["rank"] = rank
        calls["hidden"] = hidden
        calls["heads_per_group"] = heads_per_group
        calls["nope_dim"] = nope_dim
        calls["rope_dim"] = rope_dim
        calls["expected_m"] = expected_m
        calls["stream_int"] = stream_int
        return torch.empty((o_arg.shape[0], hidden, 1), dtype=o_arg.dtype)

    monkeypatch.setattr(
        wo_impl.torch.ops.b12x,
        "wo_projection_inv_rope_mxfp8_fused",
        fake_fused,
    )

    out = wo_impl.wo_projection_inv_rope_mxfp8(binding=binding, stream=123)

    assert isinstance(binding, WOProjectionInvRopeBinding)
    assert not hasattr(binding, "workspace")
    assert calls["o"] is o
    assert calls["positions"] is positions
    assert calls["cos_sin_cache"] is cos_sin_cache
    assert calls["wo_a_values"] is weights.wo_a.values
    assert calls["wo_a_scale_rows"] is weights.wo_a.scale_rows
    assert calls["wo_a_scale_mma"] is weights.wo_a.scale_mma
    assert calls["wo_b_values"] is weights.wo_b.values
    assert calls["wo_b_scale_rows"] is weights.wo_b.scale_rows
    assert calls["wo_b_scale_mma"] is weights.wo_b.scale_mma
    assert calls["groups"] == 2
    assert calls["group_width"] == 128
    assert calls["rank"] == 64
    assert calls["hidden"] == 256
    assert calls["heads_per_group"] == 1
    assert calls["nope_dim"] == 96
    assert calls["rope_dim"] == 32
    assert calls["expected_m"] == 3
    assert calls["stream_int"] == 123
    assert out.shape == (3, 256, 1)


def test_wo_projection_binding_owns_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()
    binding = plan.bind(scratch=scratch, source_tgd=source, weights=weights)

    with pytest.raises(ValueError, match="binding owns source_tgd"):
        wo_impl.wo_projection_mxfp8(
            source,
            weights,
            binding=binding,
        )
