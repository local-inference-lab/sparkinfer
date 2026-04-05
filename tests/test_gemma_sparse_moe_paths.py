from __future__ import annotations

import functools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_moe import (
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    ModelSpec,
    load_expert_weights,
    load_gate_weight,
    make_input_activations,
)
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    _b12x_gemma_sparse_moe_fp4_static,
    allocate_tp_moe_workspace_pool,
    b12x_sparse_moe_fp4,
    clear_tp_moe_caches,
)
from b12x.moe.fused.reference import compare_to_reference


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        pytest.skip(f"Requires SM120, got sm_{major}{minor}")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


def _pack_experts(weights) -> B12XFP4ExpertWeights:
    return B12XFP4ExpertWeights(
        a1_gscale=weights.w13_input_scale_per_expert,
        w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled,
        w1_alphas=weights.g1_alphas_per_expert,
        a2_gscale=weights.w2_input_scale_per_expert,
        w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled,
        w2_alphas=weights.g2_alphas_per_expert,
    )


def _gemma_rmsnorm_after_allreduce(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = hidden_states + residual
    x_fp32 = residual_out.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    normed = (x_fp32 * inv_rms * (1.0 + norm_weight.float())).to(dtype=hidden_states.dtype)
    return normed, residual_out


@functools.lru_cache(maxsize=1)
def _load_norm_weight() -> torch.Tensor:
    from benchmarks.checkpoint_loader import IndexedSafetensorLoader

    return IndexedSafetensorLoader(MODEL_PATH).get_tensor(
        "model.language_model.layers.0.post_attention_layernorm.weight"
    ).to(device=torch.device("cuda"), dtype=torch.bfloat16).contiguous()


def test_static_full_fused_path_matches_semi_fused_baseline() -> None:
    _skip_if_unavailable()
    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=0)
    experts = _pack_experts(weights)
    norm_weight = _load_norm_weight()
    hidden_states = make_input_activations(spec, 4, seed=4100, device=device)
    residual = make_input_activations(spec, 4, seed=4200, device=device)
    normed_hidden_states, residual_expected = _gemma_rmsnorm_after_allreduce(
        hidden_states,
        residual,
        norm_weight,
        1e-6,
    )

    class _FakePreMLPRuntime:
        def allreduce_gemma_rmsnorm(
            self,
            inp: torch.Tensor,
            residual_in: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            *,
            peer_input_ptrs=None,
            out=None,
            residual_out=None,
        ):
            del peer_input_ptrs
            assert inp is hidden_states
            assert residual_in is residual
            assert weight is norm_weight
            assert eps == 1e-6
            if out is None:
                out = torch.empty_like(normed_hidden_states)
            out.copy_(normed_hidden_states)
            if residual_out is None:
                residual_out = torch.empty_like(residual_expected)
            residual_out.copy_(residual_expected)
            return out, residual_out

    fused_pool = allocate_tp_moe_workspace_pool()
    semi_pool = allocate_tp_moe_workspace_pool()
    fused_out, fused_residual_out = _b12x_gemma_sparse_moe_fp4_static(
        hidden_states,
        residual,
        pre_mlp_runtime=_FakePreMLPRuntime(),
        norm_weight=norm_weight,
        norm_eps=1e-6,
        experts=experts,
        workspace=fused_pool,
        top_k=spec.top_k,
        gate_weight=gate_weight,
        input_scales_static=True,
        # Compare fusion boundaries, not a different FC1 quantization contract.
        fc1_tile_amax=False,
    )
    semi_out = b12x_sparse_moe_fp4(
        normed_hidden_states,
        experts=experts,
        workspace=semi_pool,
        top_k=spec.top_k,
        gate_weight=gate_weight,
        input_scales_static=True,
    )
    torch.cuda.synchronize()

    output_metrics = compare_to_reference(fused_out, semi_out)
    residual_metrics = compare_to_reference(fused_residual_out, residual_expected)
    assert output_metrics.max_abs <= 1e-4
    assert output_metrics.rmse <= 1e-5
    assert output_metrics.mean_abs <= 1e-5
    assert output_metrics.cos > 0.999
    assert residual_metrics.max_abs == 0.0
    assert residual_metrics.cos == 1.0
