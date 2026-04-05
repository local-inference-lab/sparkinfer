from __future__ import annotations

from unittest.mock import patch
import os

import torch

import b12x.integration.tp_moe as tp_moe
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    B12XTopKRouting,
    b12x_route_experts_fast,
    b12x_sparse_moe_fp4,
)


def _make_experts(hidden_size: int, num_experts: int = 3) -> B12XFP4ExpertWeights:
    return B12XFP4ExpertWeights(
        a1_gscale=torch.ones(num_experts, dtype=torch.float32),
        w1_fp4=torch.zeros(num_experts, 4, max(1, hidden_size // 2), dtype=torch.uint8),
        w1_blockscale=torch.zeros(num_experts, 1, dtype=torch.float32),
        w1_alphas=torch.ones(num_experts, dtype=torch.float32),
        a2_gscale=torch.ones(num_experts, dtype=torch.float32),
        w2_fp4=torch.zeros(num_experts, hidden_size, 1, dtype=torch.uint8),
        w2_blockscale=torch.zeros(num_experts, 1, dtype=torch.float32),
        w2_alphas=torch.ones(num_experts, dtype=torch.float32),
    )


def test_route_experts_fast_from_gate_weight_renormalizes() -> None:
    hidden_states = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    gate_weight = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [-1.0, -1.0],
        ],
        dtype=torch.float32,
    )

    routing = b12x_route_experts_fast(hidden_states, top_k=2, gate_weight=gate_weight)

    assert routing.router_logits is not None
    assert routing.topk_ids.dtype == torch.int32
    assert routing.flat_ids is not None
    assert routing.flat_weights is not None
    assert routing.topk_ids.tolist() == [[0, 1], [1, 0]]
    expected = torch.softmax(
        torch.tensor(
            [
                [10.0, 0.0],
                [10.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    torch.testing.assert_close(routing.topk_weights, expected)
    torch.testing.assert_close(routing.flat_ids, routing.topk_ids.view(-1))
    torch.testing.assert_close(routing.flat_weights, routing.topk_weights.view(-1))


def test_route_experts_fast_without_renormalize_returns_topk_logits() -> None:
    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    router_logits = torch.tensor([[0.5, 3.0, -4.0]], dtype=torch.float32)

    routing = b12x_route_experts_fast(
        hidden_states,
        top_k=2,
        router_logits=router_logits,
        renormalize=False,
    )

    assert routing.topk_ids.tolist() == [[1, 0]]
    torch.testing.assert_close(
        routing.topk_weights,
        torch.tensor([[3.0, 0.5]], dtype=torch.float32),
    )


def test_route_experts_fast_applies_gate_bias() -> None:
    hidden_states = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    gate_weight = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gate_bias = torch.tensor([5.0, 0.0], dtype=torch.float32)

    routing = b12x_route_experts_fast(
        hidden_states,
        top_k=1,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
    )

    assert routing.topk_ids.tolist() == [[0]]
    torch.testing.assert_close(routing.router_logits, torch.tensor([[5.0, 1.0]]))


def test_sparse_moe_fp4_accepts_precomputed_router_logits() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    workspace = object()
    captured: dict[str, torch.Tensor | object] = {}

    def fake_b12x_moe_fp4(
        a,
        a1_gscale,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        a2_gscale,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        topk_weights,
        topk_ids,
        *,
        workspace,
        output=None,
        input_scales_are_reciprocal=False,
        input_scales_static=False,
        fast_math=None,
        fc2_tile_amax=False,
    ):
        del a1_gscale, w1_fp4, w1_blockscale, w1_alphas
        del a2_gscale, w2_fp4, w2_blockscale, w2_alphas
        del input_scales_are_reciprocal, input_scales_static, fast_math, fc2_tile_amax
        captured["a"] = a
        captured["topk_weights"] = topk_weights
        captured["topk_ids"] = topk_ids
        captured["workspace"] = workspace
        if output is None:
            return torch.full_like(a, 7.0)
        output.fill_(7.0)
        return output

    router_logits = torch.tensor(
        [
            [0.5, 3.0, -1.0],
            [2.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    with patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4):
        out, routing = b12x_sparse_moe_fp4(
            hidden_states,
            experts=experts,
            workspace=workspace,
            top_k=2,
            router_logits=router_logits,
            return_routing=True,
        )

    assert captured["workspace"] is workspace
    assert captured["a"] is hidden_states
    assert out.shape == hidden_states.shape
    assert routing.topk_ids.tolist() == [[1, 0], [0, 2]]
    torch.testing.assert_close(captured["topk_ids"], routing.topk_ids)
    torch.testing.assert_close(captured["topk_weights"], routing.topk_weights)


def test_sparse_moe_fp4_forwards_low_level_flags() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    workspace = object()
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 2, dtype=torch.float32),
        topk_ids=torch.zeros(2, 2, dtype=torch.int64),
    )
    captured: dict[str, object] = {}

    def fake_b12x_moe_fp4(
        *args,
        workspace,
        output=None,
        input_scales_are_reciprocal=False,
        input_scales_static=False,
        fast_math=None,
        fc2_tile_amax=False,
    ):
        del args
        captured["workspace"] = workspace
        captured["output"] = output
        captured["input_scales_are_reciprocal"] = input_scales_are_reciprocal
        captured["input_scales_static"] = input_scales_static
        captured["fast_math"] = fast_math
        captured["fc2_tile_amax"] = fc2_tile_amax
        if output is None:
            return torch.ones_like(hidden_states)
        output.fill_(1.0)
        return output

    output = torch.empty_like(hidden_states)
    with patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4):
        actual = b12x_sparse_moe_fp4(
            hidden_states,
            experts=experts,
            workspace=workspace,
            routing=routing,
            output=output,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
            fast_math=False,
        )

    assert actual is output
    assert captured == {
        "workspace": workspace,
        "output": output,
        "input_scales_are_reciprocal": True,
        "input_scales_static": True,
        "fast_math": False,
        "fc2_tile_amax": False,
    }


def test_sparse_moe_fp4_scales_output_in_place() -> None:
    hidden_states = torch.randn(3, 4)
    experts = _make_experts(hidden_size=4)
    workspace = object()
    output = torch.empty_like(hidden_states)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(3, 2, dtype=torch.float32),
        topk_ids=torch.zeros(3, 2, dtype=torch.int64),
    )

    def fake_b12x_moe_fp4(*args, output=None, **kwargs):
        del args, kwargs
        assert output is not None
        output.fill_(2.0)
        return output

    with patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4):
        actual = b12x_sparse_moe_fp4(
            hidden_states,
            experts=experts,
            workspace=workspace,
            routing=routing,
            output=output,
            routed_scaling_factor=0.25,
        )

    assert actual is output
    torch.testing.assert_close(actual, torch.full_like(hidden_states, 0.5))


def test_b12x_moe_fp4_env_forces_fc2_tile_amax() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    plan = tp_moe.TPMoEPlan(
        implementation="static",
        state_E=2,
        weight_E=experts.w1_fp4.shape[0],
        routed_rows=2,
        max_rows=2,
        k=hidden_states.shape[1],
        n=experts.w2_fp4.shape[2] * 2,
        num_topk=1,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
        max_tokens_per_launch=hidden_states.shape[0],
    )
    workspace = tp_moe.TPCompactStaticWorkspace(
        implementation="static",
        state_E=2,
        weight_E=experts.w1_fp4.shape[0],
        max_rows=2,
        k=hidden_states.shape[1],
        n=experts.w2_fp4.shape[2] * 2,
        num_topk=1,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
        row_counts=torch.zeros(2, dtype=torch.int32),
        token_map=torch.zeros(2, 2, dtype=torch.int32),
        token_weights=torch.zeros(2, 2, dtype=torch.float32),
        packed_input=torch.zeros(2, 2, hidden_states.shape[1] // 2, dtype=torch.uint8),
        packed_input_scale=torch.zeros(2, 128, 4, dtype=torch.uint8),
        barrier_count=torch.zeros(1, dtype=torch.int32),
        barrier_epoch=torch.zeros(1, dtype=torch.int32),
        routed_rows_capacity=2,
        active_expert_count=torch.zeros(1, dtype=torch.int32),
        weight_expert_ids=torch.arange(2, dtype=torch.int32),
        global_to_local_expert=torch.zeros(experts.w1_fp4.shape[0], dtype=torch.int32),
        compact_topk_ids=torch.zeros(2, dtype=torch.int32),
    )
    captured: dict[str, object] = {}

    def fake_launch_compact_static(**kwargs):
        captured["fc2_tile_amax"] = kwargs["fc2_tile_amax"]
        return None

    with (
        patch.dict(os.environ, {"B12X_MOE_FC2_TILE_AMAX": "1"}, clear=False),
        patch.object(tp_moe, "_make_workspace_plan", return_value=plan),
        patch.object(tp_moe, "_resolve_workspace", return_value=workspace),
        patch.object(tp_moe, "_get_weight_views", return_value=object()),
        patch.object(tp_moe, "current_cuda_stream", return_value=object()),
        patch.object(tp_moe, "_launch_compact_static", side_effect=fake_launch_compact_static),
    ):
        out = tp_moe.b12x_moe_fp4(
            hidden_states,
            experts.a1_gscale,
            experts.w1_fp4,
            experts.w1_blockscale,
            experts.w1_alphas,
            experts.a2_gscale,
            experts.w2_fp4,
            experts.w2_blockscale,
            experts.w2_alphas,
            topk_weights,
            topk_ids,
            workspace=workspace,
            fc2_tile_amax=False,
        )

    assert captured["fc2_tile_amax"] is True
    assert out.shape == hidden_states.shape


def test_sparse_moe_fp4_requires_topk_or_routing() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)

    try:
        b12x_sparse_moe_fp4(
            hidden_states,
            experts=experts,
            workspace=object(),
        )
    except ValueError as exc:
        assert "top_k is required" in str(exc)
    else:
        raise AssertionError("expected missing top_k validation to fire")


def test_sparse_moe_fp4_keeps_routing_path_explicit() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )

    try:
        b12x_sparse_moe_fp4(
            hidden_states,
            experts=experts,
            workspace=object(),
            routing=routing,
            top_k=1,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected routing/top_k exclusivity check to fire")


def test_sparse_moe_fp4_rejects_routing_batch_mismatch() -> None:
    hidden_states = torch.randn(3, 4)
    experts = _make_experts(hidden_size=4)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )

    try:
        b12x_sparse_moe_fp4(
            hidden_states,
            experts=experts,
            workspace=object(),
            routing=routing,
        )
    except ValueError as exc:
        assert "routing batch mismatch" in str(exc)
    else:
        raise AssertionError("expected routing batch validation to fire")
