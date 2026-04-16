from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_moe import (
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    ModelSpec,
    get_scale_contract_params,
    load_expert_weights,
    make_routed_inputs,
)
from b12x.integration.tp_moe import (
    allocate_tp_moe_workspace,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    clear_tp_moe_caches,
)
from b12x.moe.fused.nvfp4.reference import compare_to_reference, moe_reference_nvfp4

from .helpers import require_sm120


def _require_model_weights() -> None:
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


def test_moe_eager_prefill_matches_oracle_across_shapes() -> None:
    device = require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    scale_params = get_scale_contract_params(weights, "shared")
    workspace = allocate_tp_moe_workspace_pool()

    for m, seed in ((23, 2300), (80, 8000)):
        x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=seed, device=device)
        exact_workspace = allocate_tp_moe_workspace(
            x,
            scale_params.a1_gscale,
            weights.w13_weight,
            scale_params.a2_gscale,
            weights.w2_weight,
            topk_ids,
            input_scales_static=True,
        )
        expected = b12x_moe_fp4(
            x,
            scale_params.a1_gscale,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            scale_params.g1_alphas,
            scale_params.a2_gscale,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            scale_params.g2_alphas,
            topk_weights,
            topk_ids,
            workspace=exact_workspace,
            input_scales_static=True,
        ).clone()
        actual = b12x_moe_fp4(
            x,
            scale_params.a1_gscale,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            scale_params.g1_alphas,
            scale_params.a2_gscale,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            scale_params.g2_alphas,
            topk_weights,
            topk_ids,
            workspace=workspace,
            input_scales_static=True,
        ).clone()
        reference = moe_reference_nvfp4(
            x,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            scale_params.g1_alphas,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            scale_params.g2_alphas,
            scale_params.a1_gscale,
            scale_params.a2_gscale,
            topk_ids,
            topk_weights,
            spec.num_experts,
            spec.hidden_size,
            spec.I_tp,
        )
        torch.cuda.synchronize(device)

        expected_metrics = compare_to_reference(actual, expected)
        assert expected_metrics.max_abs <= 1e-3, f"m={m}: pooled-vs-exact max_abs={expected_metrics.max_abs:.6f}"
        assert expected_metrics.cos > 0.9999, f"m={m}: pooled-vs-exact cos={expected_metrics.cos:.6f}"

        metrics = compare_to_reference(actual, reference)
        assert metrics.max_abs <= 8e-4, f"m={m}: max_abs={metrics.max_abs:.6f}"
        assert metrics.rmse <= 5e-5, f"m={m}: rmse={metrics.rmse:.6f}"
        assert metrics.cos > 0.9999, f"m={m}: cos={metrics.cos:.6f}"
