from __future__ import annotations

import pytest
import torch

import b12x.integration.tp_moe as tp_moe_impl
from b12x.integration import (
    B12XFP4ExpertWeights,
    TPMoEFP4Binding,
    TPMoERouteBinding,
    TPMoEScratchCaps,
    TPMoESparseFP4Binding,
    TPMoEWorkspacePool,
    plan_tp_moe_scratch,
)


def _caps() -> TPMoEScratchCaps:
    return TPMoEScratchCaps(
        device="cpu",
        max_tokens=4,
        weight_E=8,
        k=128,
        n=64,
        num_topk=2,
        dtype=torch.bfloat16,
    )


def _runtime_tensors(m: int = 3):
    a = torch.empty((m, 128), dtype=torch.bfloat16)
    a1_gscale = torch.ones((8,), dtype=torch.float32)
    w1_fp4 = torch.empty((8, 128, 64), dtype=torch.uint8)
    w1_blockscale = torch.empty((8, 1, 1), dtype=torch.uint8)
    w1_alphas = torch.ones((8,), dtype=torch.float32)
    a2_gscale = torch.ones((8,), dtype=torch.float32)
    w2_fp4 = torch.empty((8, 128, 32), dtype=torch.uint8)
    w2_blockscale = torch.empty((8, 1, 1), dtype=torch.uint8)
    w2_alphas = torch.ones((8,), dtype=torch.float32)
    topk_weights = torch.empty((m, 2), dtype=torch.float32)
    topk_ids = torch.empty((m, 2), dtype=torch.int32)
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


def _experts(tensors: dict[str, torch.Tensor]) -> B12XFP4ExpertWeights:
    return B12XFP4ExpertWeights(
        a1_gscale=tensors["a1_gscale"],
        w1_fp4=tensors["w1_fp4"],
        w1_blockscale=tensors["w1_blockscale"],
        w1_alphas=tensors["w1_alphas"],
        a2_gscale=tensors["a2_gscale"],
        w2_fp4=tensors["w2_fp4"],
        w2_blockscale=tensors["w2_blockscale"],
        w2_alphas=tensors["w2_alphas"],
    )


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
    caps = TPMoEScratchCaps(
        device="cpu",
        max_tokens=4,
        weight_E=8,
        k=128,
        n=64,
        num_topk=2,
        dtype=torch.bfloat16,
        route_num_experts=0,
    )
    plan = plan_tp_moe_scratch(caps)

    assert plan.layout.route_workspace_nbytes == 0
    assert plan.layout.core_workspace_nbytes > 0
    assert plan.scratch_specs()[0].name == "tp_moe.scratch"


def test_w4a16_scratch_plan_uses_route_pack_capacity_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)

    base_caps = dict(
        device="cpu",
        weight_E=256,
        k=4096,
        n=7168,
        num_topk=8,
        dtype=torch.bfloat16,
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


def test_w4a16_topk6_bucket_materializes_with_planned_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_w4a16_prewarm(workspace, *, token_counts, **_kwargs) -> None:
        workspace.planned_fused_moe_launches = {
            ("packed", "e4m3_k16", int(token_count)): object()
            for token_count in token_counts
        }
        workspace.planned_topk_sum_launches = {
            int(token_count): object() for token_count in token_counts
        }

    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)
    monkeypatch.setattr(
        tp_moe_impl,
        "_prewarm_w4a16_planned_launches",
        _fake_w4a16_prewarm,
    )
    plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            device="cpu",
            max_tokens=15,
            weight_E=8,
            k=128,
            n=64,
            num_topk=6,
            dtype=torch.bfloat16,
            core_token_counts=(15,),
            route_num_experts=0,
            quant_mode="w4a16",
        )
    )
    scratch = _scratch_for_plan(plan)

    pool = plan.make_workspace_pool(scratch=scratch)

    assert pool.workspaces


def test_tp_moe_scratch_plan_materializes_caller_owned_scratch_pool() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)

    pool = plan.make_workspace_pool(scratch=scratch)

    assert isinstance(pool, TPMoEWorkspacePool)
    assert pool.shared_arena is not None
    assert pool.shared_arena.data_ptr() == scratch[0].data_ptr()
    assert pool.route_workspace_nbytes == plan.layout.route_workspace_nbytes
    assert pool.core_arena_nbytes == plan.layout.core_workspace_nbytes
    assert pool.frozen is True
    assert pool.workspaces


def test_tp_moe_scratch_plan_binds_caller_owned_scratch() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(scratch=scratch, **tensors)

    assert isinstance(binding, TPMoEFP4Binding)
    assert not hasattr(binding, "workspace")
    assert not hasattr(binding, "scratch")
    assert binding.row_counts is not None
    assert binding.token_map is not None
    assert binding.packed_input is not None
    assert binding.a is tensors["a"]
    assert binding.topk_ids is tensors["topk_ids"]


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
    monkeypatch.setattr(tp_moe_impl, "get_num_sm", lambda _device: 120)
    plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            device="cpu",
            max_tokens=4,
            weight_E=8,
            k=128,
            n=64,
            num_topk=2,
            dtype=torch.bfloat16,
            route_num_experts=0,
            quant_mode="w4a16",
        )
    )
    scratch = _scratch_for_plan(plan)
    tensors = _runtime_tensors()

    binding = plan.bind(scratch=scratch, **tensors, quant_mode="w4a16")

    assert isinstance(binding, TPMoEFP4Binding)
    assert not hasattr(binding, "workspace")
    assert not hasattr(binding, "scratch")
    assert binding.intermediate_cache13 is not None
    assert binding.intermediate_cache2 is not None
    assert binding.packed_route_indices is not None


def test_tp_moe_workspace_pool_bind_fp4_returns_common_binding_type() -> None:
    plan = plan_tp_moe_scratch(_caps())
    scratch = _scratch_for_plan(plan)
    pool = plan.make_workspace_pool(scratch=scratch)
    tensors = _runtime_tensors()

    binding = pool.bind_fp4(**tensors)

    assert isinstance(binding, TPMoEFP4Binding)
    assert not hasattr(binding, "workspace")
    assert not hasattr(binding, "scratch")
    assert binding.row_counts is not None
    assert binding.token_map is not None
    assert binding.a is tensors["a"]
    assert binding.topk_ids is tensors["topk_ids"]


def test_tp_moe_workspace_pool_bind_route_returns_common_binding_type() -> None:
    pool = TPMoEWorkspacePool()
    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 128), dtype=torch.bfloat16)

    binding = pool.bind_route(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )

    assert isinstance(binding, TPMoERouteBinding)
    assert not hasattr(binding, "workspace")
    assert binding.scratch is pool
    assert binding.hidden_states is hidden_states
    assert binding.gate_weight is gate_weight


def test_tp_moe_workspace_pool_bind_sparse_fp4_returns_common_binding_type() -> None:
    pool = TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    experts = _experts(tensors)

    binding = pool.bind_sparse_fp4(
        hidden_states=tensors["a"],
        experts=experts,
        routing=tp_moe_impl.B12XTopKRouting(
            topk_weights=tensors["topk_weights"],
            topk_ids=tensors["topk_ids"],
        ),
    )

    assert isinstance(binding, TPMoESparseFP4Binding)
    assert not hasattr(binding, "workspace")
    assert binding.scratch is pool
    assert binding.hidden_states is tensors["a"]
    assert binding.experts is experts


def test_tp_moe_fp4_binding_run_uses_function_binding_argument(monkeypatch) -> None:
    pool = TPMoEWorkspacePool()
    binding = pool.bind_fp4(**_runtime_tensors())
    calls = {}
    sentinel = object()

    def fake_moe_fp4(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(tp_moe_impl, "b12x_moe_fp4", fake_moe_fp4)

    assert binding.run() is sentinel
    assert calls["binding"] is binding


def test_tp_moe_route_binding_run_uses_function_binding_argument(monkeypatch) -> None:
    pool = TPMoEWorkspacePool()
    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 128), dtype=torch.bfloat16)
    binding = pool.bind_route(
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
    pool = TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    binding = pool.bind_sparse_fp4(
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
    pool = TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    binding = pool.bind_fp4(**tensors)

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        tp_moe_impl.b12x_moe_fp4(tensors["a"], binding=binding)


def test_tp_moe_route_binding_owns_runtime_tensors() -> None:
    pool = TPMoEWorkspacePool()
    hidden_states = torch.empty((3, 128), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 128), dtype=torch.bfloat16)
    binding = pool.bind_route(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )

    with pytest.raises(ValueError, match="route binding owns runtime tensors"):
        tp_moe_impl.b12x_route_experts_fast(hidden_states, binding=binding)


def test_tp_moe_sparse_fp4_binding_owns_runtime_tensors() -> None:
    pool = TPMoEWorkspacePool()
    tensors = _runtime_tensors()
    experts = _experts(tensors)
    binding = pool.bind_sparse_fp4(
        hidden_states=tensors["a"],
        experts=experts,
        routing=tp_moe_impl.B12XTopKRouting(
            topk_weights=tensors["topk_weights"],
            topk_ids=tensors["topk_ids"],
        ),
    )

    with pytest.raises(ValueError, match="sparse FP4 binding owns runtime tensors"):
        tp_moe_impl.b12x_sparse_moe_fp4(
            tensors["a"],
            experts=experts,
            workspace=pool,
            binding=binding,
        )


def test_tp_moe_fp4_entrypoint_requires_tensors_or_binding() -> None:
    with pytest.raises(TypeError, match="requires all FP4 tensors"):
        tp_moe_impl.b12x_moe_fp4()


def test_tp_moe_route_entrypoint_requires_inputs_or_binding() -> None:
    with pytest.raises(TypeError, match="requires hidden_states/top_k"):
        tp_moe_impl.b12x_route_experts_fast()


def test_tp_moe_sparse_fp4_entrypoint_requires_inputs_or_binding() -> None:
    with pytest.raises(TypeError, match="requires hidden_states, experts, workspace"):
        tp_moe_impl.b12x_sparse_moe_fp4()
