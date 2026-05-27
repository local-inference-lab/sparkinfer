from __future__ import annotations

import pytest
import torch

import b12x.integration.residual as residual_impl
from b12x.integration import (
    B12XMHCBinding,
    B12XMHCScratchCaps,
    empty_mhc_workspace,
    plan_mhc_scratch,
)
from b12x.integration.residual import MHC_DEFAULT_BLOCK_K, MHC_DEFAULT_SPLIT_K


def test_mhc_scratch_plan_exposes_one_component_scratch_spec() -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=8,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "mhc.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert specs[0].nbytes == plan.layout.nbytes
    assert plan.caps.max_tokens == 4
    assert plan.caps.hidden_size == 16
    assert plan.caps.split_k == 8


def test_mhc_scratch_plan_binds_caller_owned_scratch() -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=8,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    binding = plan.bind(scratch=scratch)

    assert isinstance(binding, B12XMHCBinding)
    assert not hasattr(binding, "workspace")
    assert binding.partials.shape == (4, 8, residual_impl.MHC_PARTIALS)
    assert binding.y is None
    assert binding.post_buffer is None
    assert binding.comb_buffer is None
    assert binding.out is None
    assert binding.split_k == 8
    assert binding.partials.device == scratch.device


def test_mhc_scratch_plan_binds_live_token_shape() -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=8,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    y = torch.empty((2, 16), dtype=torch.bfloat16)
    post = torch.empty((2, 4), dtype=torch.float32)
    comb = torch.empty((2, 4, 4), dtype=torch.float32)
    out = torch.empty((2, 4, 16), dtype=torch.bfloat16)

    binding = plan.bind(
        scratch=scratch,
        tokens=2,
        y=y,
        post=post,
        comb=comb,
        out=out,
    )

    assert binding.partials.shape == (2, 8, residual_impl.MHC_PARTIALS)
    assert binding.y is y
    assert binding.post_buffer is post
    assert binding.comb_buffer is comb
    assert binding.out is out


def test_mhc_workspace_bind_returns_common_binding_type() -> None:
    workspace = empty_mhc_workspace(
        num_tokens=4,
        hidden_size=16,
        split_k=8,
        device="cpu",
    )

    binding = workspace.bind()

    assert isinstance(binding, B12XMHCBinding)
    assert not hasattr(binding, "workspace")
    assert binding.partials is workspace.partials
    assert binding.y is workspace.y
    assert binding.post_buffer is workspace.post
    assert binding.comb_buffer is workspace.comb
    assert binding.out is workspace.out


def test_mhc_pre_binding_supplies_workspace_outputs(monkeypatch) -> None:
    workspace = empty_mhc_workspace(
        num_tokens=4,
        hidden_size=16,
        split_k=MHC_DEFAULT_SPLIT_K,
        device="cpu",
    )
    binding = workspace.bind()
    residual = torch.empty((0, 4, 16), dtype=torch.bfloat16)
    fn = torch.empty((24, 64), dtype=torch.float32)
    hc_scale = torch.empty((3,), dtype=torch.float32)
    hc_base = torch.empty((24,), dtype=torch.float32)

    def fake_validate_pre_inputs(*args):
        return 0, 16, MHC_DEFAULT_SPLIT_K * MHC_DEFAULT_BLOCK_K

    monkeypatch.setattr(residual_impl, "_validate_pre_inputs", fake_validate_pre_inputs)

    y, post, comb = residual_impl.b12x_mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=2,
        binding=binding,
    )

    assert y.shape == (0, 16)
    assert post.shape == (0, 4)
    assert comb.shape == (0, 4, 4)
    assert y.untyped_storage().data_ptr() == workspace.y.untyped_storage().data_ptr()
    assert post.untyped_storage().data_ptr() == workspace.post.untyped_storage().data_ptr()
    assert comb.untyped_storage().data_ptr() == workspace.comb.untyped_storage().data_ptr()


def test_mhc_pre_binding_owns_outputs() -> None:
    workspace = empty_mhc_workspace(
        num_tokens=4,
        hidden_size=16,
        split_k=8,
        device="cpu",
    )
    y_out = torch.empty((0, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="binding owns workspace and output buffers"):
        residual_impl.b12x_mhc_pre(
            torch.empty((0, 4, 16), dtype=torch.bfloat16),
            torch.empty((24, 64), dtype=torch.float32),
            torch.empty((3,), dtype=torch.float32),
            torch.empty((24,), dtype=torch.float32),
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=2,
            binding=workspace.bind(),
            y_out=y_out,
        )
