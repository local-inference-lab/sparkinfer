from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import b12x.integration.tp_moe as tp_moe_impl
from b12x.integration import (
    B12XFP4ExpertWeights,
    TPMoEFP4Binding,
    TPMoERouteBinding,
    TPMoEScratchCaps,
    TPMoESparseFP4Binding,
    build_tp_moe_route_binding,
    build_tp_moe_sparse_fp4_binding,
    plan_b12x_fp4_moe_weights,
    plan_tp_moe_scratch,
    prepare_b12x_fp4_moe_weights,
)
from b12x.moe.execution import PreparedWeightLayout
from b12x.moe.fused.w4a8.weights import repack_w4a8_weights


def _weight_plan(
    quant_mode: str = "nvfp4",
    *,
    source_format: str = "modelopt_nvfp4",
    experts: int = 8,
    k: int = 128,
    n: int = 64,
    activation: str = "silu",
    w4a16_layout: PreparedWeightLayout | None = None,
):
    return plan_b12x_fp4_moe_weights(
        quant_modes=quant_mode,
        source_format=source_format,
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=k,
        intermediate_size=n,
        w4a16_layout=w4a16_layout,
    )


def _caps(**overrides) -> TPMoEScratchCaps:
    weight_plan = overrides.pop("weight_plan", None) or _weight_plan()
    values = dict(
        device="cpu",
        max_tokens=4,
        num_topk=2,
        weight_plan=weight_plan,
        quant_mode=next(iter(weight_plan.quant_modes)),
    )
    values.update(overrides)
    return TPMoEScratchCaps(**values)


def _clear_moe_force_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "B12X_MOE_FORCE_A8",
        "B12X_FORCE_MOE_A8",
        "B12X_MOE_FORCE_A16",
    ):
        monkeypatch.delenv(name, raising=False)


def test_dynamic_deterministic_output_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", raising=False)

    assert not tp_moe_impl._dynamic_deterministic_output_enabled(
        quant_mode="nvfp4",
        device=torch.device("cuda"),
    )

    monkeypatch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")

    assert tp_moe_impl._dynamic_deterministic_output_enabled(
        quant_mode="nvfp4",
        device=torch.device("cuda"),
    )
    assert not tp_moe_impl._dynamic_deterministic_output_enabled(
        quant_mode="w4a16",
        device=torch.device("cuda"),
    )
    assert not tp_moe_impl._dynamic_deterministic_output_enabled(
        quant_mode="nvfp4",
        device=torch.device("cpu"),
    )


def test_moe_force_envs_do_not_override_explicit_quant_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_moe_force_env(monkeypatch)
    monkeypatch.setenv("B12X_MOE_FORCE_A8", "1")
    monkeypatch.setenv("B12X_FORCE_MOE_A8", "1")
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")

    assert tp_moe_impl.default_moe_quant_mode() == "nvfp4"
    assert tp_moe_impl._normalize_quant_mode(None) == "nvfp4"
    assert tp_moe_impl._normalize_quant_mode("nvfp4") == "nvfp4"
    assert tp_moe_impl._normalize_quant_mode("w4a16") == "w4a16"
    assert (
        tp_moe_impl._normalize_quant_mode_for_source(
            "w4a16",
            "fp4_e8m0_k32",
        )
        == "w4a16"
    )
    caps = _caps(
        weight_plan=_weight_plan(
            "w4a16",
            source_format="fp4_e8m0_k32",
        )
    )

    assert caps.quant_mode == "w4a16"


def test_explicit_w4a8_mx_binds_prepared_metadata() -> None:
    weight_plan = _weight_plan(
        "w4a8_mx",
        source_format="fp4_e8m0_k32",
        k=256,
        n=128,
    )
    plan = plan_tp_moe_scratch(
        _caps(
            weight_plan=weight_plan,
            route_num_experts=0,
            swiglu_limit=7.0,
        )
    )
    scratch = _scratch_for_plan(plan)
    prepared_w4a8 = SimpleNamespace(
        num_experts=8,
        hidden_size=256,
        intermediate_size=128,
        params_dtype=torch.bfloat16,
        w13_rp=torch.empty((1,), dtype=torch.int32),
        w13_sfb=torch.empty((1,), dtype=torch.int32),
        w2_rp=torch.empty((1,), dtype=torch.int32),
        w2_sfb=torch.empty((1,), dtype=torch.int32),
    )
    runtime_tensors = _runtime_tensors(k=256, n=128)
    experts = _experts(runtime_tensors, weight_plan, prepared_w4a8)

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(runtime_tensors, experts),
    )

    assert binding.quant_mode == "w4a8_mx"
    assert binding.weight_E == 8
    assert binding.n == 128
    assert binding.experts is experts
    assert experts.representation_for("w4a8_mx") is prepared_w4a8
    assert binding.swiglu_limit == 7.0


def test_explicit_w4a8_mx_prepares_native_e8m0_source_in_place() -> None:
    experts, k, n = 8, 256, 128
    w1_rows = 2 * n
    w1_source = (
        torch.arange(experts * w1_rows * (k // 2), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(experts, w1_rows, k // 2)
    )
    w2_source = (
        torch.arange(experts * k * (n // 2), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(experts, k, n // 2)
    )
    w1_scale = (
        torch.arange(experts * w1_rows * (k // 32), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(experts, w1_rows, k // 32)
    )
    w2_scale = (
        torch.arange(experts * k * (n // 32), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(experts, k, n // 32)
    )
    w1_original = w1_source.clone()
    w2_original = w2_source.clone()
    w1_scale_original = w1_scale.clone()
    w2_scale_original = w2_scale.clone()

    w1_expected = torch.cat([w1_original[:, n:], w1_original[:, :n]], dim=1)
    w1_scale_expected = torch.cat(
        [w1_scale_original[:, n:], w1_scale_original[:, :n]], dim=1
    )
    expected_w13_rp, expected_w13_sfb = repack_w4a8_weights(
        w1_expected.contiguous(),
        w1_scale_expected.clamp(max=247).contiguous(),
    )
    expected_w2_rp, expected_w2_sfb = repack_w4a8_weights(
        w2_original.contiguous(),
        w2_scale_original.clamp(max=247).contiguous(),
    )

    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=k,
        intermediate_size=n,
        w13_layout="w31",
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=weight_plan,
        w1_fp4=w1_source,
        w1_blockscale=w1_scale,
        w1_global_scale=torch.ones((experts,), dtype=torch.float32),
        a1_gscale=torch.ones((experts,), dtype=torch.float32),
        w2_fp4=w2_source,
        w2_blockscale=w2_scale,
        w2_global_scale=torch.ones((experts,), dtype=torch.float32),
        a2_gscale=torch.ones((experts,), dtype=torch.float32),
        params_dtype=torch.bfloat16,
    )
    w4a8 = prepared.representation_for("w4a8_mx")

    assert w4a8 is not None
    assert w4a8.num_experts == experts
    assert w4a8.hidden_size == k
    assert w4a8.intermediate_size == n
    assert (
        w4a8.w13_rp.untyped_storage().data_ptr()
        == w1_source.untyped_storage().data_ptr()
    )
    assert (
        w4a8.w2_rp.untyped_storage().data_ptr()
        == w2_source.untyped_storage().data_ptr()
    )
    assert (
        w4a8.w13_sfb.untyped_storage().data_ptr()
        == w1_scale.untyped_storage().data_ptr()
    )
    assert (
        w4a8.w2_sfb.untyped_storage().data_ptr()
        == w2_scale.untyped_storage().data_ptr()
    )
    assert torch.equal(w4a8.w13_rp, expected_w13_rp)
    assert torch.equal(w4a8.w13_sfb, expected_w13_sfb)
    assert torch.equal(w4a8.w2_rp, expected_w2_rp)
    assert torch.equal(w4a8.w2_sfb, expected_w2_sfb)


def _runtime_tensors(
    m: int = 3,
    topk: int = 2,
    *,
    experts: int = 8,
    k: int = 128,
    n: int = 64,
):
    a = torch.empty((m, k), dtype=torch.bfloat16)
    a1_gscale = torch.ones((experts,), dtype=torch.float32)
    w1_fp4 = torch.empty((experts, 2 * n, k // 2), dtype=torch.uint8)
    w1_blockscale = torch.empty((experts, 1, 1), dtype=torch.uint8)
    w1_alphas = torch.ones((experts,), dtype=torch.float32)
    a2_gscale = torch.ones((experts,), dtype=torch.float32)
    w2_fp4 = torch.empty((experts, k, n // 2), dtype=torch.uint8)
    w2_blockscale = torch.empty((experts, 1, 1), dtype=torch.uint8)
    w2_alphas = torch.ones((experts,), dtype=torch.float32)
    topk_weights = torch.empty((m, topk), dtype=torch.float32)
    topk_ids = torch.empty((m, topk), dtype=torch.int32)
    return {
        "a": a,
        "a1_gscale": a1_gscale,
        "w1_fp4": w1_fp4,
        "w1_blockscale": w1_blockscale,
        "w1_alphas": w1_alphas,
        "a2_gscale": a2_gscale,
        "w2_fp4": w2_fp4,
        "w2_blockscale": w2_blockscale,
        "w2_alphas": w2_alphas,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def _experts(
    tensors: dict[str, torch.Tensor],
    weight_plan=None,
    payload: object | None = None,
) -> B12XFP4ExpertWeights:
    if weight_plan is None:
        weight_plan = _weight_plan(
            experts=int(tensors["w1_fp4"].shape[0]),
            k=int(tensors["w2_fp4"].shape[1]),
            n=int(tensors["w2_fp4"].shape[2]) * 2,
        )
    representation = None
    for mode in weight_plan.quant_modes:
        layout = weight_plan.required_weight_layout(mode)
        if layout is None:
            continue
        if payload is None:
            payload = SimpleNamespace()
        if mode == "w4a8_mx":
            defaults = {
                "w13_rp": tensors["w1_fp4"],
                "w13_sfb": tensors["w1_blockscale"],
                "w2_rp": tensors["w2_fp4"],
                "w2_sfb": tensors["w2_blockscale"],
            }
        else:
            defaults = {
                "w13": tensors["w1_fp4"],
                "w13_scale": tensors["w1_blockscale"],
                "w13_global_scale": tensors["w1_alphas"],
                "w2": tensors["w2_fp4"],
                "w2_scale": tensors["w2_blockscale"],
                "w2_global_scale": tensors["w2_alphas"],
                "weight_layout": (
                    "modelopt"
                    if layout is PreparedWeightLayout.SOURCE_NATIVE
                    else "packed"
                ),
                "scale_format": (
                    "e8m0_k32"
                    if weight_plan.source_format == "fp4_e8m0_k32"
                    else "e4m3_k16"
                ),
            }
        defaults.update(
            num_experts=weight_plan.num_experts,
            hidden_size=weight_plan.hidden_size,
            intermediate_size=weight_plan.intermediate_size,
            params_dtype=torch.bfloat16,
            is_gated=True,
        )
        for name, value in defaults.items():
            if not hasattr(payload, name):
                setattr(payload, name, value)
        assert representation is None
        representation = tp_moe_impl._PreparedWeightRepresentation(
            quant_mode=mode,
            layout=layout,
            value=payload,
        )
    canonical_w1 = tensors["w1_fp4"]
    canonical_s1 = tensors["w1_blockscale"]
    canonical_w2 = tensors["w2_fp4"]
    canonical_s2 = tensors["w2_blockscale"]
    if representation is not None and weight_plan.discards_source_parameters:
        value = representation.value
        canonical_w1 = getattr(value, "w13_rp", getattr(value, "w13", None))
        canonical_s1 = getattr(
            value, "w13_sfb", getattr(value, "w13_scale", None)
        )
        canonical_w2 = getattr(value, "w2_rp", getattr(value, "w2", None))
        canonical_s2 = getattr(value, "w2_sfb", getattr(value, "w2_scale", None))
    return B12XFP4ExpertWeights(
        plan=weight_plan,
        a1_gscale=tensors["a1_gscale"],
        w1_fp4=canonical_w1,
        w1_blockscale=canonical_s1,
        w1_alphas=tensors["w1_alphas"],
        a2_gscale=tensors["a2_gscale"],
        w2_fp4=canonical_w2,
        w2_blockscale=canonical_s2,
        w2_alphas=tensors["w2_alphas"],
        representation=representation,
    )


def _binding_args(
    tensors: dict[str, torch.Tensor],
    experts: B12XFP4ExpertWeights,
) -> dict[str, object]:
    return {
        "a": tensors["a"],
        "experts": experts,
        "topk_weights": tensors["topk_weights"],
        "topk_ids": tensors["topk_ids"],
    }


def _scratch_for_plan(plan):
    return tuple(
        torch.empty(shape, dtype=dtype, device=plan.scratch_specs()[idx].device)
        for idx, (shape, dtype) in enumerate(plan.shapes_and_dtypes())
    )


def test_tp_moe_scratch_plan_exposes_one_opaque_scratch_spec() -> None:
    plan = plan_tp_moe_scratch(_caps())

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "tp_moe.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.layout.route_workspace_nbytes > 0
    assert plan.layout.core_workspace_nbytes > 0
    assert plan.layout.total_nbytes == specs[0].nbytes


def test_tp_moe_scratch_plan_can_skip_route_scratch() -> None:
    caps = _caps(route_num_experts=0)
    plan = plan_tp_moe_scratch(caps)

    assert plan.layout.route_workspace_nbytes == 0
    assert plan.layout.core_workspace_nbytes > 0
    assert plan.scratch_specs()[0].name == "tp_moe.scratch"


def test_w4a16_scratch_plan_uses_route_pack_capacity_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)

    weight_plan = _weight_plan(
        "w4a16",
        experts=256,
        k=4096,
        n=7168,
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    base_caps = dict(
        device="cpu",
        weight_plan=weight_plan,
        num_topk=8,
        route_num_experts=0,
        quant_mode="w4a16",
    )
    plan_4080 = plan_tp_moe_scratch(
        TPMoEScratchCaps(max_tokens=4080, core_token_counts=(4080,), **base_caps)
    )
    plan_4096 = plan_tp_moe_scratch(
        TPMoEScratchCaps(max_tokens=4096, core_token_counts=(4096,), **base_caps)
    )
    plan_topk6 = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=4080,
            core_token_counts=(4080,),
            **{**base_caps, "num_topk": 6},
        )
    )

    assert plan_4080.layout.core_token_counts[0] == 4096
    assert plan_4096.layout.core_token_counts[0] == 4096
    assert plan_topk6.layout.core_token_counts[0] == 4096
    assert 4080 not in plan_4080.layout.core_token_counts
    assert 4080 not in plan_topk6.layout.core_token_counts
    assert plan_4080.shapes_and_dtypes() == plan_4096.shapes_and_dtypes()


def test_w4a16_topk6_bucket_binds_with_planned_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)
    weight_plan = _weight_plan(
        "w4a16",
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    plan = plan_tp_moe_scratch(
        _caps(
            weight_plan=weight_plan,
            max_tokens=15,
            num_topk=6,
            core_token_counts=(15,),
            route_num_experts=0,
        )
    )
    scratch = _scratch_for_plan(plan)

    tensors = _runtime_tensors(m=15, topk=6)
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, weight_plan)),
    )

    assert binding.implementation == "w4a16"
    assert binding.routed_rows_capacity is not None
    assert binding.routed_rows_capacity >= 15 * 6


def test_w4a16_materialize_can_prewarm_activation_amax_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    fused = object()

    def _fake_w4a16_prewarm(
        workspace, *, token_counts, collect_activation_amax=False, **_kwargs
    ) -> None:
        captured["collect_activation_amax"] = bool(collect_activation_amax)
        workspace.planned_fused_moe_launches = {
            ("packed", "e4m3_k16", int(token_count), bool(collect_activation_amax)): fused
            for token_count in token_counts
        }
        workspace.planned_topk_sum_launches = {
            int(token_count): object() for token_count in token_counts
        }
        workspace.planned_collect_activation_amax = bool(collect_activation_amax)

    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)
    monkeypatch.setattr(
        tp_moe_impl,
        "_prewarm_w4a16_planned_launches",
        _fake_w4a16_prewarm,
    )
    pool = tp_moe_impl.allocate_tp_moe_workspace_pool(frozen=True)
    weight_plan = _weight_plan(
        "w4a16",
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )

    tp_moe_impl.materialize_tp_moe_arena_workspaces(
        pool,
        caps=_caps(
            max_tokens=4,
            weight_plan=weight_plan,
            core_token_counts=(4,),
            route_num_experts=0,
            collect_activation_amax=True,
        ),
    )

    workspace = next(iter(pool.workspaces.values()))
    selected, topk_sum = tp_moe_impl._w4a16_preplanned_launches(
        workspace,
        token_count=4,
        weight_layout="packed",
        scale_format="e4m3_k16",
        collect_activation_amax=True,
    )

    assert captured["collect_activation_amax"] is True
    assert workspace.planned_collect_activation_amax is True
    assert selected is fused
    assert topk_sum is not None


def test_w4a16_scratch_binding_carries_activation_amax_to_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight_plan = _weight_plan(
        "w4a16",
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    plan = plan_tp_moe_scratch(
        _caps(weight_plan=weight_plan, route_num_experts=0)
    )
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    activation_amax = torch.zeros((3, 8, 2), dtype=torch.float32)
    output = torch.empty_like(tensors["a"])
    payload = SimpleNamespace(
        num_experts=8,
        hidden_size=128,
        intermediate_size=64,
        params_dtype=torch.bfloat16,
        is_gated=True,
        weight_layout="packed",
        scale_format="e4m3_k16",
    )
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, weight_plan, payload)),
        output=output,
        activation_amax=activation_amax,
        layer_idx=2,
    )
    calls = {}

    import b12x.moe.fused.w4a16.kernel as w4a16_kernel

    def _fake_run_w4a16(*args, **kwargs):
        calls.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(w4a16_kernel, "run_w4a16_moe", _fake_run_w4a16)

    result = tp_moe_impl.b12x_moe_fp4(binding=binding)

    assert result is output
    assert calls["activation_amax"] is activation_amax
    assert calls["layer_idx"] == 2


def test_activation_amax_is_w4a16_only() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    activation_amax = torch.zeros((1, 8, 2), dtype=torch.float32)
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
        activation_amax=activation_amax,
        layer_idx=0,
    )

    with pytest.raises(NotImplementedError, match="only supported for W4A16"):
        tp_moe_impl.b12x_moe_fp4(binding=binding)


def test_tp_moe_scratch_plan_binding_maps_caller_owned_scratch() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )

    assert isinstance(binding, TPMoEFP4Binding)
    assert binding.row_counts is not None
    assert binding.row_counts.untyped_storage().data_ptr() == scratch[0].untyped_storage().data_ptr()
    assert binding.a is tensors["a"]
    assert binding.topk_ids is tensors["topk_ids"]


def test_tp_moe_scratch_plan_binds_caller_owned_scratch() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )

    assert isinstance(binding, TPMoEFP4Binding)
    assert not hasattr(binding, "workspace")
    assert not hasattr(binding, "scratch")
    assert binding.row_counts is not None
    assert binding.token_map is not None
    assert binding.packed_input is not None
    assert binding.a is tensors["a"]
    assert binding.topk_ids is tensors["topk_ids"]


def test_tp_moe_fp4_binding_rehydrates_micro_workspace_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    output = torch.empty_like(tensors["a"])
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
        output=output,
    )
    calls = {}

    monkeypatch.setattr(tp_moe_impl, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(tp_moe_impl, "_get_weight_views", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tp_moe_impl,
        "_launch_micro",
        lambda **kwargs: calls.update(kwargs),
    )

    result = tp_moe_impl.b12x_moe_fp4(binding=binding)

    assert result is output
    assert isinstance(calls["workspace"], tp_moe_impl.TPMicroWorkspace)
    assert calls["workspace"].active_expert_count is binding.active_expert_count
    assert calls["workspace"].micro_intermediate is binding.micro_intermediate


def test_tp_moe_fp4_binding_rehydrates_dynamic_workspace_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps = _caps(max_tokens=400, route_num_experts=0)
    plan = plan_tp_moe_scratch(caps)
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors(m=400)
    output = torch.empty_like(tensors["a"])
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
        output=output,
    )
    calls = {}

    monkeypatch.setattr(tp_moe_impl, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(tp_moe_impl, "_get_weight_views", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tp_moe_impl,
        "_launch_dynamic",
        lambda **kwargs: calls.update(kwargs),
    )

    result = tp_moe_impl.b12x_moe_fp4(binding=binding)

    assert result is output
    assert binding.route_output is not None
    assert isinstance(calls["workspace"], tp_moe_impl.TPDynamicWorkspace)
    assert calls["workspace"].route_output is binding.route_output
    assert calls["workspace"].input_gs is binding.input_gs
    assert calls["workspace"].task_ready is binding.task_ready


def test_tp_moe_scratch_plan_bind_does_not_materialize_workspace_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_materialize(*_args, **_kwargs) -> None:
        raise AssertionError("bind must not materialize or prewarm workspaces")

    monkeypatch.setattr(
        tp_moe_impl,
        "materialize_tp_moe_arena_workspaces",
        _fail_materialize,
    )
    monkeypatch.setattr(
        tp_moe_impl,
        "_prewarm_w4a16_planned_launches",
        _fail_materialize,
    )
    monkeypatch.setattr(
        tp_moe_impl,
        "_materialize_workspace_from_core_arena",
        _fail_materialize,
    )
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)
    weight_plan = _weight_plan(
        "w4a16",
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    plan = plan_tp_moe_scratch(
        _caps(weight_plan=weight_plan, route_num_experts=0)
    )
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, weight_plan)),
    )

    assert isinstance(binding, TPMoEFP4Binding)
    assert not hasattr(binding, "workspace")
    assert not hasattr(binding, "scratch")
    assert binding.intermediate_cache13 is not None
    assert binding.intermediate_cache2 is not None
    assert binding.packed_route_indices is not None


def test_tp_moe_plan_bind_fp4_returns_common_binding_type() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )

    assert isinstance(binding, TPMoEFP4Binding)
    assert not hasattr(binding, "workspace")
    assert not hasattr(binding, "scratch")
    assert binding.row_counts is not None
    assert binding.token_map is not None
    assert binding.a is tensors["a"]
    assert binding.topk_ids is tensors["topk_ids"]


def test_tp_moe_route_builder_returns_common_binding_type() -> None:
    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 128), dtype=torch.bfloat16)

    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )

    assert isinstance(binding, TPMoERouteBinding)
    assert not hasattr(binding, "workspace")
    assert binding.scratch is None
    assert binding.hidden_states is hidden_states
    assert binding.gate_weight is gate_weight


def test_tp_moe_sparse_fp4_builder_returns_common_binding_type() -> None:
    scratch = tp_moe_impl.TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    experts = _experts(tensors)

    binding = build_tp_moe_sparse_fp4_binding(
        scratch=scratch,
        hidden_states=tensors["a"],
        experts=experts,
        routing=tp_moe_impl.B12XTopKRouting(
            topk_weights=tensors["topk_weights"],
            topk_ids=tensors["topk_ids"],
        ),
    )

    assert isinstance(binding, TPMoESparseFP4Binding)
    assert not hasattr(binding, "workspace")
    assert binding.scratch is scratch
    assert binding.hidden_states is tensors["a"]
    assert binding.experts is experts


def test_tp_moe_fp4_binding_run_uses_function_binding_argument(monkeypatch) -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )
    calls = {}
    sentinel = object()

    def fake_moe_fp4(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(tp_moe_impl, "b12x_moe_fp4", fake_moe_fp4)

    assert binding.run() is sentinel
    assert calls["binding"] is binding


def test_tp_moe_route_binding_run_uses_function_binding_argument(monkeypatch) -> None:
    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 128), dtype=torch.bfloat16)
    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )
    calls = {}
    sentinel = object()

    def fake_route(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(tp_moe_impl, "b12x_route_experts_fast", fake_route)

    assert binding.run() is sentinel
    assert calls["binding"] is binding


def test_tp_moe_sparse_fp4_binding_run_uses_function_binding_argument(monkeypatch) -> None:
    scratch = tp_moe_impl.TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    binding = build_tp_moe_sparse_fp4_binding(
        scratch=scratch,
        hidden_states=tensors["a"],
        experts=_experts(tensors),
        routing=tp_moe_impl.B12XTopKRouting(
            topk_weights=tensors["topk_weights"],
            topk_ids=tensors["topk_ids"],
        ),
    )
    calls = {}
    sentinel = object()

    def fake_sparse(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(tp_moe_impl, "b12x_sparse_moe_fp4", fake_sparse)

    assert binding.run() is sentinel
    assert calls["binding"] is binding


def test_tp_moe_fp4_binding_owns_runtime_tensors() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()
    binding = plan.bind(
        scratch=scratch,
        **_binding_args(tensors, _experts(tensors, plan.caps.weight_plan)),
    )

    with pytest.raises(TypeError):
        tp_moe_impl.b12x_moe_fp4(tensors["a"], binding=binding)


def test_tp_moe_route_binding_owns_runtime_tensors() -> None:
    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 128), dtype=torch.bfloat16)
    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )

    with pytest.raises(TypeError):
        tp_moe_impl.b12x_route_experts_fast(hidden_states, binding=binding)


def test_tp_moe_sparse_fp4_binding_owns_runtime_tensors() -> None:
    scratch = tp_moe_impl.TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    experts = _experts(tensors)
    binding = build_tp_moe_sparse_fp4_binding(
        scratch=scratch,
        hidden_states=tensors["a"],
        experts=experts,
        routing=tp_moe_impl.B12XTopKRouting(
            topk_weights=tensors["topk_weights"],
            topk_ids=tensors["topk_ids"],
        ),
    )

    with pytest.raises(TypeError):
        tp_moe_impl.b12x_sparse_moe_fp4(
            tensors["a"],
            experts=experts,
            binding=binding,
        )


def test_tp_moe_fp4_entrypoint_requires_tensors_or_binding() -> None:
    with pytest.raises(TypeError):
        tp_moe_impl.b12x_moe_fp4()


def test_tp_moe_route_entrypoint_requires_inputs_or_binding() -> None:
    with pytest.raises(TypeError):
        tp_moe_impl.b12x_route_experts_fast()


def test_tp_moe_sparse_fp4_entrypoint_requires_inputs_or_binding() -> None:
    with pytest.raises(TypeError):
        tp_moe_impl.b12x_sparse_moe_fp4()
