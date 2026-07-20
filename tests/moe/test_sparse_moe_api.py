from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import sparkinfer.moe.fused_moe._impl as tp_moe
from sparkinfer.moe.fused_moe._impl import (
    SPARKINFERFP4ExpertWeights,
    SPARKINFERTopKRouting,
    TPMoEFP4Binding,
    sparkinfer_moe_fp4,
    sparkinfer_route_experts_fast,
    sparkinfer_sparse_moe_fp4,
    build_tp_moe_route_binding,
    build_tp_moe_sparse_fp4_binding,
    plan_sparkinfer_fp4_moe_weights,
    _PreparedWeightRepresentation,
)


def _make_experts(
    hidden_size: int,
    num_experts: int = 3,
    *,
    source_format: str = "modelopt_nvfp4",
    activation: str = "silu",
    quant_mode: str | None = None,
) -> SPARKINFERFP4ExpertWeights:
    from types import SimpleNamespace

    quant_mode = quant_mode or (
        "w4a16" if source_format == "compressed_tensors" else "nvfp4"
    )
    w1_fp4 = torch.zeros(
        num_experts, 4, max(1, hidden_size // 2), dtype=torch.uint8
    )
    w2_fp4 = torch.zeros(num_experts, hidden_size, 1, dtype=torch.uint8)
    w1_alphas = torch.ones(num_experts, dtype=torch.float32)
    w2_alphas = torch.ones(num_experts, dtype=torch.float32)
    plan = plan_sparkinfer_fp4_moe_weights(
        quant_modes=quant_mode,
        source_format=source_format,
        activation=activation,
        params_dtype=torch.float32,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=2,
    )
    layout = plan.required_weight_layout(quant_mode)
    w1_blockscale = torch.zeros(num_experts, 1, dtype=torch.float32)
    w2_blockscale = torch.zeros(num_experts, 1, dtype=torch.float32)
    representation = None
    if layout is not None:
        payload = SimpleNamespace(
            w13=w1_fp4,
            w13_scale=w1_blockscale,
            w13_global_scale=w1_alphas,
            w2=w2_fp4,
            w2_scale=w2_blockscale,
            w2_global_scale=w2_alphas,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=2,
        )
        representation = _PreparedWeightRepresentation(
            quant_mode=quant_mode,
            layout=layout,
            value=payload,
        )
    return SPARKINFERFP4ExpertWeights(
        plan=plan,
        a1_gscale=torch.ones(num_experts, dtype=torch.float32),
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=w1_alphas,
        a2_gscale=torch.ones(num_experts, dtype=torch.float32),
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_alphas,
        representation=representation,
    )


def _make_scratch():
    return tp_moe.TPMoEWorkspacePool()


def _make_fp4_binding_from_kwargs(**kwargs) -> TPMoEFP4Binding:
    a = kwargs["a"]
    experts = kwargs["experts"]
    topk_ids = kwargs["topk_ids"]
    mode = kwargs.get("quant_mode") or next(iter(experts.plan.quant_modes))
    return TPMoEFP4Binding(
        a=a,
        experts=experts,
        topk_weights=kwargs["topk_weights"],
        topk_ids=topk_ids,
        implementation="test",
        state_E=experts.num_experts,
        weight_E=experts.num_experts,
        max_rows=int(a.shape[0]),
        k=int(a.shape[1]),
        n=experts.intermediate_size,
        num_topk=int(topk_ids.shape[1]),
        device=a.device,
        dtype=a.dtype,
        apply_router_weight_on_input=bool(
            kwargs.get("apply_router_weight_on_input", False)
        ),
        output=kwargs.get("output"),
        input_scales_static=bool(kwargs.get("input_scales_static", False)),
        fast_math=kwargs.get("fast_math"),
        quant_mode=mode,
        unit_scale_contract=bool(kwargs.get("unit_scale_contract", False)),
        swiglu_limit=kwargs.get("swiglu_limit"),
        swiglu_alpha=kwargs.get("swiglu_alpha"),
        swiglu_beta=kwargs.get("swiglu_beta"),
    )


def _make_fp4_binding(
    hidden_states: torch.Tensor,
    experts: SPARKINFERFP4ExpertWeights,
    routing: SPARKINFERTopKRouting,
    **kwargs,
) -> TPMoEFP4Binding:
    binding_kwargs = {
        "a": hidden_states,
        "experts": experts,
        "topk_weights": routing.topk_weights,
        "topk_ids": routing.topk_ids,
    }
    binding_kwargs.update(kwargs)
    return _make_fp4_binding_from_kwargs(**binding_kwargs)


def _make_sparse_binding(
    hidden_states: torch.Tensor,
    experts: SPARKINFERFP4ExpertWeights,
    **kwargs,
):
    return build_tp_moe_sparse_fp4_binding(
        scratch=_make_scratch(),
        hidden_states=hidden_states,
        experts=experts,
        **kwargs,
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

    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )
    routing = sparkinfer_route_experts_fast(binding=binding)

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

    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        router_logits=router_logits,
        renormalize=False,
    )
    routing = sparkinfer_route_experts_fast(binding=binding)

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

    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=1,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
    )
    routing = sparkinfer_route_experts_fast(binding=binding)

    assert routing.topk_ids.tolist() == [[0]]
    torch.testing.assert_close(routing.router_logits, torch.tensor([[5.0, 1.0]]))


def test_sparse_moe_fp4_accepts_precomputed_router_logits() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    captured: dict[str, torch.Tensor | object] = {}

    def fake_build_tp_moe_fp4_binding(**kwargs):
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_sparkinfer_moe_fp4(*, binding):
        captured["a"] = binding.a
        captured["topk_weights"] = binding.topk_weights
        captured["topk_ids"] = binding.topk_ids
        if binding.output is None:
            return torch.full_like(binding.a, 7.0)
        binding.output.fill_(7.0)
        return binding.output

    router_logits = torch.tensor(
        [
            [0.5, 3.0, -1.0],
            [2.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    binding = _make_sparse_binding(
        hidden_states,
        experts,
        top_k=2,
        router_logits=router_logits,
        return_routing=True,
    )
    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "sparkinfer_moe_fp4", fake_sparkinfer_moe_fp4),
    ):
        out, routing = sparkinfer_sparse_moe_fp4(binding=binding)

    assert captured["a"] is hidden_states
    assert out.shape == hidden_states.shape
    assert routing.topk_ids.tolist() == [[1, 0], [0, 2]]
    torch.testing.assert_close(captured["topk_ids"], routing.topk_ids)
    torch.testing.assert_close(captured["topk_weights"], routing.topk_weights)


def test_sparse_moe_fp4_forwards_prepared_contract_and_launch_options() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(
        hidden_size=4,
        source_format="compressed_tensors",
        activation="swigluoai_uninterleave",
        quant_mode="w4a16",
    )
    routing = SPARKINFERTopKRouting(
        topk_weights=torch.ones(2, 2, dtype=torch.float32),
        topk_ids=torch.zeros(2, 2, dtype=torch.int64),
    )
    captured: dict[str, object] = {}

    def fake_build_tp_moe_fp4_binding(**kwargs):
        captured["output"] = kwargs.get("output")
        captured["input_scales_static"] = kwargs.get("input_scales_static")
        captured["fast_math"] = kwargs.get("fast_math")
        captured["experts"] = kwargs.get("experts")
        captured["quant_mode"] = kwargs.get("quant_mode")
        captured["swiglu_limit"] = kwargs.get("swiglu_limit")
        captured["swiglu_alpha"] = kwargs.get("swiglu_alpha")
        captured["swiglu_beta"] = kwargs.get("swiglu_beta")
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_sparkinfer_moe_fp4(*, binding):
        if binding.output is None:
            return torch.ones_like(hidden_states)
        binding.output.fill_(1.0)
        return binding.output

    output = torch.empty_like(hidden_states)
    binding = _make_sparse_binding(
        hidden_states,
        experts,
        routing=routing,
        output=output,
        input_scales_static=True,
        fast_math=False,
        quant_mode="w4a16",
        swiglu_limit=5.0,
        swiglu_alpha=1.5,
        swiglu_beta=0.25,
    )
    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "sparkinfer_moe_fp4", fake_sparkinfer_moe_fp4),
    ):
        actual = sparkinfer_sparse_moe_fp4(binding=binding)

    assert actual is output
    assert captured["output"] is output
    assert captured["experts"] is experts
    assert captured["input_scales_static"] is True
    assert captured["fast_math"] is False
    assert captured["quant_mode"] == "w4a16"
    assert captured["swiglu_limit"] == 5.0
    assert captured["swiglu_alpha"] == 1.5
    assert captured["swiglu_beta"] == 0.25


def test_fp4_expert_weights_default_to_modelopt_nvfp4_source_format() -> None:
    experts = _make_experts(hidden_size=4)

    assert experts.source_format == "modelopt_nvfp4"


def test_moe_fp4_rejects_compressed_tensors_with_nvfp4() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4, source_format="compressed_tensors")
    routing = SPARKINFERTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    binding = _make_fp4_binding(
        hidden_states,
        experts,
        routing,
        quant_mode="nvfp4",
    )
    with pytest.raises(ValueError) as exc_info:
        sparkinfer_moe_fp4(binding=binding)

    message = str(exc_info.value)
    assert "source_format='compressed_tensors'" in message
    assert "quant_mode='nvfp4'" in message


def test_sparse_moe_fp4_rejects_compressed_tensors_with_nvfp4() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4, source_format="compressed_tensors")
    routing = SPARKINFERTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    with pytest.raises(ValueError) as exc_info:
        _make_sparse_binding(
            hidden_states,
            experts,
            routing=routing,
            quant_mode="nvfp4",
        )

    message = str(exc_info.value)
    assert "quant_mode='nvfp4'" in message
    assert "prepared-weight plan ['w4a16']" in message


def test_sparse_moe_fp4_prepared_plan_ignores_runtime_force_env(monkeypatch) -> None:
    monkeypatch.setenv("SPARKINFER_MOE_FORCE_A16", "1")
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    routing = SPARKINFERTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    captured: list[object] = []

    def fake_build_tp_moe_fp4_binding(**kwargs):
        captured.append(kwargs.get("quant_mode"))
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_sparkinfer_moe_fp4(*, binding):
        del binding
        return torch.ones_like(hidden_states)

    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "sparkinfer_moe_fp4", fake_sparkinfer_moe_fp4),
    ):
        sparkinfer_sparse_moe_fp4(
            binding=_make_sparse_binding(hidden_states, experts, routing=routing)
        )
        sparkinfer_sparse_moe_fp4(
            binding=_make_sparse_binding(
                hidden_states,
                experts,
                routing=routing,
                quant_mode="nvfp4",
            )
        )

    assert captured == ["nvfp4", "nvfp4"]


def test_sparse_moe_fp4_scales_output_in_place() -> None:
    hidden_states = torch.randn(3, 4)
    experts = _make_experts(hidden_size=4)
    output = torch.empty_like(hidden_states)
    routing = SPARKINFERTopKRouting(
        topk_weights=torch.ones(3, 2, dtype=torch.float32),
        topk_ids=torch.zeros(3, 2, dtype=torch.int64),
    )

    def fake_build_tp_moe_fp4_binding(**kwargs):
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_sparkinfer_moe_fp4(*, binding):
        assert binding.output is not None
        binding.output.fill_(2.0)
        return binding.output

    binding = _make_sparse_binding(
        hidden_states,
        experts,
        routing=routing,
        output=output,
        routed_scaling_factor=0.25,
    )
    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "sparkinfer_moe_fp4", fake_sparkinfer_moe_fp4),
    ):
        actual = sparkinfer_sparse_moe_fp4(binding=binding)

    assert actual is output
    torch.testing.assert_close(actual, torch.full_like(hidden_states, 0.5))


def test_sparse_moe_fp4_requires_topk_or_routing() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)

    try:
        _make_sparse_binding(hidden_states, experts)
    except ValueError as exc:
        assert "top_k is required" in str(exc)
    else:
        raise AssertionError("expected missing top_k validation to fire")


def test_sparse_moe_fp4_keeps_routing_path_explicit() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    routing = SPARKINFERTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )

    try:
        _make_sparse_binding(hidden_states, experts, routing=routing, top_k=1)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected routing/top_k exclusivity check to fire")


def test_sparse_moe_fp4_rejects_routing_batch_mismatch() -> None:
    hidden_states = torch.randn(3, 4)
    experts = _make_experts(hidden_size=4)
    routing = SPARKINFERTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    binding = _make_sparse_binding(hidden_states, experts, routing=routing)

    try:
        sparkinfer_sparse_moe_fp4(binding=binding)
    except ValueError as exc:
        assert "routing batch mismatch" in str(exc)
    else:
        raise AssertionError("expected routing batch validation to fire")
