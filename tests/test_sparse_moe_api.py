from __future__ import annotations

from unittest.mock import patch

import torch

import b12x.integration.tp_moe as tp_moe
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    B12XTopKRouting,
    TPMoEFP4Binding,
    b12x_moe_fp4,
    b12x_route_experts_fast,
    b12x_sparse_moe_fp4,
    build_tp_moe_route_binding,
    build_tp_moe_sparse_fp4_binding,
)


def _make_experts(
    hidden_size: int,
    num_experts: int = 3,
    *,
    source_format: str = "modelopt_nvfp4",
) -> B12XFP4ExpertWeights:
    return B12XFP4ExpertWeights(
        a1_gscale=torch.ones(num_experts, dtype=torch.float32),
        w1_fp4=torch.zeros(num_experts, 4, max(1, hidden_size // 2), dtype=torch.uint8),
        w1_blockscale=torch.zeros(num_experts, 1, dtype=torch.float32),
        w1_alphas=torch.ones(num_experts, dtype=torch.float32),
        a2_gscale=torch.ones(num_experts, dtype=torch.float32),
        w2_fp4=torch.zeros(num_experts, hidden_size, 1, dtype=torch.uint8),
        w2_blockscale=torch.zeros(num_experts, 1, dtype=torch.float32),
        w2_alphas=torch.ones(num_experts, dtype=torch.float32),
        source_format=source_format,
    )


def _make_scratch():
    return tp_moe.TPMoEWorkspacePool()


def _make_fp4_binding_from_kwargs(**kwargs) -> TPMoEFP4Binding:
    a = kwargs["a"]
    w1_fp4 = kwargs["w1_fp4"]
    w2_fp4 = kwargs["w2_fp4"]
    topk_ids = kwargs["topk_ids"]
    return TPMoEFP4Binding(
        a=a,
        a1_gscale=kwargs["a1_gscale"],
        w1_fp4=w1_fp4,
        w1_blockscale=kwargs["w1_blockscale"],
        w1_alphas=kwargs["w1_alphas"],
        a2_gscale=kwargs["a2_gscale"],
        w2_fp4=w2_fp4,
        w2_blockscale=kwargs["w2_blockscale"],
        w2_alphas=kwargs["w2_alphas"],
        topk_weights=kwargs["topk_weights"],
        topk_ids=topk_ids,
        implementation="test",
        state_E=int(w1_fp4.shape[0]),
        weight_E=int(w1_fp4.shape[0]),
        max_rows=int(a.shape[0]),
        k=int(a.shape[1]),
        n=int(w2_fp4.shape[2]) * 2,
        num_topk=int(topk_ids.shape[1]),
        device=a.device,
        dtype=a.dtype,
        apply_router_weight_on_input=bool(
            kwargs.get("apply_router_weight_on_input", False)
        ),
        output=kwargs.get("output"),
        input_scales_are_reciprocal=kwargs.get("input_scales_are_reciprocal"),
        input_scales_static=bool(kwargs.get("input_scales_static", False)),
        fast_math=kwargs.get("fast_math"),
        activation=kwargs.get("activation", "silu"),
        quant_mode=kwargs.get("quant_mode"),
        unit_scale_contract=bool(kwargs.get("unit_scale_contract", False)),
        source_format=kwargs.get("source_format", "modelopt_nvfp4"),
        w13_layout=kwargs.get("w13_layout", "w13"),
        prepared_w4a16=kwargs.get("prepared_w4a16"),
        swiglu_limit=kwargs.get("swiglu_limit"),
        swiglu_alpha=kwargs.get("swiglu_alpha"),
        swiglu_beta=kwargs.get("swiglu_beta"),
    )


def _make_fp4_binding(
    hidden_states: torch.Tensor,
    experts: B12XFP4ExpertWeights,
    routing: B12XTopKRouting,
    **kwargs,
) -> TPMoEFP4Binding:
    binding_kwargs = {
        "a": hidden_states,
        "a1_gscale": experts.a1_gscale,
        "w1_fp4": experts.w1_fp4,
        "w1_blockscale": experts.w1_blockscale,
        "w1_alphas": experts.w1_alphas,
        "a2_gscale": experts.a2_gscale,
        "w2_fp4": experts.w2_fp4,
        "w2_blockscale": experts.w2_blockscale,
        "w2_alphas": experts.w2_alphas,
        "topk_weights": routing.topk_weights,
        "topk_ids": routing.topk_ids,
        "source_format": experts.source_format,
        "w13_layout": experts.w13_layout,
    }
    binding_kwargs.update(kwargs)
    return _make_fp4_binding_from_kwargs(**binding_kwargs)


def _make_sparse_binding(
    hidden_states: torch.Tensor,
    experts: B12XFP4ExpertWeights,
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
    routing = b12x_route_experts_fast(binding=binding)

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
    routing = b12x_route_experts_fast(binding=binding)

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
    routing = b12x_route_experts_fast(binding=binding)

    assert routing.topk_ids.tolist() == [[0]]
    torch.testing.assert_close(routing.router_logits, torch.tensor([[5.0, 1.0]]))


def test_sparse_moe_fp4_accepts_precomputed_router_logits() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    captured: dict[str, torch.Tensor | object] = {}

    def fake_build_tp_moe_fp4_binding(**kwargs):
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_b12x_moe_fp4(*, binding):
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
        patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4),
    ):
        out, routing = b12x_sparse_moe_fp4(binding=binding)

    assert captured["a"] is hidden_states
    assert out.shape == hidden_states.shape
    assert routing.topk_ids.tolist() == [[1, 0], [0, 2]]
    torch.testing.assert_close(captured["topk_ids"], routing.topk_ids)
    torch.testing.assert_close(captured["topk_weights"], routing.topk_weights)


def test_sparse_moe_fp4_forwards_low_level_flags() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4, source_format="compressed_tensors")
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 2, dtype=torch.float32),
        topk_ids=torch.zeros(2, 2, dtype=torch.int64),
    )
    captured: dict[str, object] = {}

    def fake_build_tp_moe_fp4_binding(**kwargs):
        captured["output"] = kwargs.get("output")
        captured["input_scales_static"] = kwargs.get("input_scales_static")
        captured["fast_math"] = kwargs.get("fast_math")
        captured["activation"] = kwargs.get("activation")
        captured["quant_mode"] = kwargs.get("quant_mode")
        captured["source_format"] = kwargs.get("source_format")
        captured["w13_layout"] = kwargs.get("w13_layout")
        captured["swiglu_limit"] = kwargs.get("swiglu_limit")
        captured["swiglu_alpha"] = kwargs.get("swiglu_alpha")
        captured["swiglu_beta"] = kwargs.get("swiglu_beta")
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_b12x_moe_fp4(*, binding):
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
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        fast_math=False,
        activation="swigluoai_uninterleave",
        quant_mode="w4a16",
        swiglu_limit=5.0,
        swiglu_alpha=1.5,
        swiglu_beta=0.25,
    )
    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4),
    ):
        actual = b12x_sparse_moe_fp4(binding=binding)

    assert actual is output
    assert captured == {
        "output": output,
        "input_scales_static": True,
        "fast_math": False,
        "activation": "swigluoai_uninterleave",
        "quant_mode": "w4a16",
        "source_format": "compressed_tensors",
        "w13_layout": "w13",
        "swiglu_limit": 5.0,
        "swiglu_alpha": 1.5,
        "swiglu_beta": 0.25,
    }


def test_fp4_expert_weights_default_to_modelopt_nvfp4_source_format() -> None:
    experts = _make_experts(hidden_size=4)

    assert experts.source_format == "modelopt_nvfp4"


def test_moe_fp4_rejects_compressed_tensors_with_nvfp4() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    binding = _make_fp4_binding(
        hidden_states,
        experts,
        routing,
        quant_mode="nvfp4",
        source_format="compressed_tensors",
    )

    try:
        b12x_moe_fp4(binding=binding)
    except ValueError as exc:
        message = str(exc)
        assert "source_format='compressed_tensors'" in message
        assert "quant_mode='w4a16'" in message
        assert "source_format='modelopt_nvfp4'" in message
    else:
        raise AssertionError("expected compressed_tensors NVFP4 validation to fire")


def test_sparse_moe_fp4_rejects_compressed_tensors_with_nvfp4() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4, source_format="compressed_tensors")
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    binding = _make_sparse_binding(
        hidden_states,
        experts,
        routing=routing,
        quant_mode="nvfp4",
    )

    try:
        b12x_sparse_moe_fp4(binding=binding)
    except ValueError as exc:
        message = str(exc)
        assert "source_format='compressed_tensors'" in message
        assert "quant_mode='w4a16'" in message
        assert "source_format='modelopt_nvfp4'" in message
    else:
        raise AssertionError("expected compressed_tensors NVFP4 validation to fire")


def test_moe_fp4_rejects_false_deprecated_reciprocal_flag() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    binding = _make_fp4_binding(
        hidden_states,
        experts,
        routing,
        input_scales_are_reciprocal=False,
    )

    try:
        b12x_moe_fp4(binding=binding)
    except AssertionError as exc:
        assert "input_scales_are_reciprocal is deprecated" in str(exc)
    else:
        raise AssertionError("expected deprecated reciprocal flag validation to fire")


def test_sparse_moe_fp4_rejects_false_deprecated_reciprocal_flag() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    binding = _make_sparse_binding(
        hidden_states,
        experts,
        routing=routing,
        input_scales_are_reciprocal=False,
    )

    try:
        b12x_sparse_moe_fp4(binding=binding)
    except AssertionError as exc:
        assert "input_scales_are_reciprocal is deprecated" in str(exc)
    else:
        raise AssertionError("expected deprecated reciprocal flag validation to fire")


def test_sparse_moe_fp4_env_defaults_to_w4a16(monkeypatch) -> None:
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    captured: list[object] = []

    def fake_build_tp_moe_fp4_binding(**kwargs):
        captured.append(kwargs.get("quant_mode"))
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_b12x_moe_fp4(*, binding):
        del binding
        return torch.ones_like(hidden_states)

    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4),
    ):
        b12x_sparse_moe_fp4(
            binding=_make_sparse_binding(hidden_states, experts, routing=routing)
        )
        b12x_sparse_moe_fp4(
            binding=_make_sparse_binding(
                hidden_states,
                experts,
                routing=routing,
                quant_mode="nvfp4",
            )
        )

    assert captured == [None, "nvfp4"]


def test_sparse_moe_fp4_scales_output_in_place() -> None:
    hidden_states = torch.randn(3, 4)
    experts = _make_experts(hidden_size=4)
    output = torch.empty_like(hidden_states)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(3, 2, dtype=torch.float32),
        topk_ids=torch.zeros(3, 2, dtype=torch.int64),
    )

    def fake_build_tp_moe_fp4_binding(**kwargs):
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_b12x_moe_fp4(*, binding):
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
        patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4),
    ):
        actual = b12x_sparse_moe_fp4(binding=binding)

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
    routing = B12XTopKRouting(
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
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    binding = _make_sparse_binding(hidden_states, experts, routing=routing)

    try:
        b12x_sparse_moe_fp4(binding=binding)
    except ValueError as exc:
        assert "routing batch mismatch" in str(exc)
    else:
        raise AssertionError("expected routing batch validation to fire")
