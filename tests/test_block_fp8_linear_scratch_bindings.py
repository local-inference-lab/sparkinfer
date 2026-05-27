from __future__ import annotations

import math

import pytest
import torch

import b12x.gemm.block_fp8_linear as block_impl
from b12x.gemm import (
    BlockFP8LinearBinding,
    BlockFP8LinearScratchCaps,
    BlockFP8LinearWeight,
    empty_block_fp8_linear_workspace,
    plan_block_fp8_linear_scratch,
)
from b12x.gemm.wo_projection import MXFP8Rows


def _packed_weight(*, in_features: int = 128, out_features: int = 256) -> BlockFP8LinearWeight:
    scale_rows = torch.empty(
        (1, out_features, in_features // 32),
        dtype=torch.float8_e8m0fnu,
    )
    scale_mma = torch.empty(
        (32, 4, math.ceil(out_features / 128), 4, math.ceil((in_features // 32) / 4), 1),
        dtype=torch.float8_e8m0fnu,
    )
    weight = MXFP8Rows(
        values=torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn),
        scale_rows=scale_rows,
        scale_mma=scale_mma,
    )
    return BlockFP8LinearWeight(
        weight=weight,
        in_features=in_features,
        out_features=out_features,
        block_size=(128, 128),
    )


def test_block_fp8_linear_scratch_plan_exposes_one_opaque_scratch_spec() -> None:
    plan = plan_block_fp8_linear_scratch(
        BlockFP8LinearScratchCaps(
            device="cpu",
            max_tokens=4,
            in_features=128,
            out_features=256,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "block_fp8_linear.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]


def test_block_fp8_linear_scratch_plan_binds_live_shape(monkeypatch) -> None:
    monkeypatch.setattr(block_impl, "_check_mxfp8_rows_storage", lambda *args, **kwargs: None)
    plan = plan_block_fp8_linear_scratch(
        BlockFP8LinearScratchCaps(
            device="cpu",
            max_tokens=4,
            in_features=128,
            out_features=256,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    source = torch.empty((3, 128), dtype=torch.bfloat16)
    output = torch.empty((3, 256, 1), dtype=torch.bfloat16)
    packed = _packed_weight()

    binding = plan.bind(
        scratch=scratch,
        source=source,
        packed_weight=packed,
        output=output,
    )

    assert isinstance(binding, BlockFP8LinearBinding)
    assert binding.source is source
    assert binding.packed_weight is packed
    assert not hasattr(binding, "workspace")
    assert binding.x_q.values.shape == (3, 128)
    assert binding.output is output


def test_block_fp8_linear_workspace_bind_returns_common_binding_type(monkeypatch) -> None:
    monkeypatch.setattr(block_impl, "_check_mxfp8_rows_storage", lambda *args, **kwargs: None)
    workspace = empty_block_fp8_linear_workspace(
        3,
        128,
        256,
        device="cpu",
        output_dtype=torch.bfloat16,
    )
    source = torch.empty((3, 128), dtype=torch.bfloat16)
    packed = _packed_weight()

    binding = workspace.bind(source=source, packed_weight=packed)

    assert isinstance(binding, BlockFP8LinearBinding)
    assert not hasattr(binding, "workspace")
    assert binding.x_q is workspace.x_q
    assert binding.output is workspace.output
    assert binding.source is source
    assert binding.packed_weight is packed


def test_block_fp8_linear_binding_supplies_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(block_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(block_impl, "_check_mxfp8_rows_storage", lambda *args, **kwargs: None)
    workspace = empty_block_fp8_linear_workspace(
        3,
        128,
        256,
        device="cpu",
        output_dtype=torch.bfloat16,
    )
    source = torch.empty((3, 128), dtype=torch.bfloat16)
    packed = _packed_weight()
    binding = workspace.bind(source=source, packed_weight=packed)
    calls = {}

    def fake_quantize(source_tk, *, out=None):
        calls["source_tk"] = source_tk
        calls["x_q_out"] = out
        return out

    def fake_dense_gemm(a, b, **kwargs):
        calls["a"] = a
        calls["b"] = b
        calls["dense_out"] = kwargs["out"]
        kwargs["out"].zero_()
        return kwargs["out"]

    monkeypatch.setattr(block_impl, "quantize_block_fp8_linear_input_mxfp8", fake_quantize)
    monkeypatch.setattr(block_impl, "dense_gemm", fake_dense_gemm)

    out = block_impl.block_fp8_linear_mxfp8(binding=binding)

    assert calls["source_tk"].data_ptr() == source.data_ptr()
    assert calls["x_q_out"] is workspace.x_q
    assert calls["dense_out"] is workspace.output
    assert out.shape == (3, 256)


def test_block_fp8_linear_binding_owns_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(block_impl, "_check_mxfp8_rows_storage", lambda *args, **kwargs: None)
    workspace = empty_block_fp8_linear_workspace(
        3,
        128,
        256,
        device="cpu",
        output_dtype=torch.bfloat16,
    )
    source = torch.empty((3, 128), dtype=torch.bfloat16)
    packed = _packed_weight()
    binding = workspace.bind(source=source, packed_weight=packed)

    with pytest.raises(ValueError, match="binding owns source"):
        block_impl.block_fp8_linear_mxfp8(
            source,
            packed,
            binding=binding,
        )
