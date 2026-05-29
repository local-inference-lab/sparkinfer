from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import b12x.integration.tp_moe as tp_moe
from b12x.attention.workspace import (
    B12XAttentionArena,
    B12XAttentionArenaCaps,
    B12XAttentionWorkspaceContract,
)
from b12x.attention.paged.workspace import (
    PagedAttentionArena,
    PagedAttentionArenaCaps,
    PagedAttentionWorkspaceContract,
)
from b12x.integration.arena import (
    B12XExecutionLaneArena,
    B12XJointArenaSpec,
    B12XMoEArenaCaps,
    _EXECUTION_LANES,
    ensure_b12x_execution_lane_arena,
)


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("B12X execution lane arenas require CUDA")
    return torch.device("cuda", torch.cuda.current_device())


def _attention_caps(device: torch.device) -> B12XAttentionArenaCaps:
    return B12XAttentionArenaCaps(
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=2,
        indexer_num_q_heads=2,
        head_dim=128,
        max_v_head_dim=128,
        topk=4,
        max_page_table_width=4,
        extend_max_total_q=8,
        extend_max_batch=2,
        extend_max_kv_rows=64,
        paged_max_q_rows=2,
        paged_max_batch=2,
    )


def _moe_caps(device: torch.device) -> B12XMoEArenaCaps:
    return B12XMoEArenaCaps(
        device=device,
        dtype=torch.bfloat16,
        weight_E=8,
        route_num_experts=8,
        k=16,
        n=16,
        num_topk=2,
        max_tokens=4,
    )


def _paged_attention_caps(device: torch.device) -> PagedAttentionArenaCaps:
    return PagedAttentionArenaCaps(
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=4,
        num_kv_heads=1,
        head_dim_qk=128,
        max_head_dim_vo=128,
        page_size=64,
        max_total_q=8,
        max_batch=2,
        max_page_table_width=4,
        max_work_items=16,
        max_partial_rows=32,
    )


def test_joint_arena_size_is_max_of_attention_and_moe_phases() -> None:
    device = _cuda_device()
    attn_caps = _attention_caps(device)
    paged_caps = _paged_attention_caps(device)
    moe_caps = _moe_caps(device)
    lane = B12XExecutionLaneArena.allocate(
        B12XJointArenaSpec(
            device=device,
            attention_caps=attn_caps,
            paged_attention_caps=paged_caps,
            moe_caps=moe_caps,
        )
    )

    expected_attn = B12XAttentionArena.required_nbytes(attn_caps)
    expected_paged_attn = PagedAttentionArena.required_nbytes(paged_caps)
    expected_moe = moe_caps.layout().total_nbytes
    assert lane.attention_nbytes == expected_attn
    assert lane.paged_attention_nbytes == expected_paged_attn
    assert lane.moe_nbytes == expected_moe
    assert lane.shared_arena_nbytes == max(
        expected_attn, expected_paged_attn, expected_moe, 1
    )


def test_joint_arena_views_share_one_backing_allocation() -> None:
    device = _cuda_device()
    lane = B12XExecutionLaneArena.allocate(
        B12XJointArenaSpec(
            device=device,
            attention_caps=_attention_caps(device),
            paged_attention_caps=_paged_attention_caps(device),
            moe_caps=_moe_caps(device),
        )
    )
    base_ptr = _storage_ptr(lane.shared_arena)

    attn_ws = lane.make_attention_workspace(
        B12XAttentionWorkspaceContract(
            mode="decode",
            max_total_q=2,
            max_batch=2,
            max_paged_q_rows=2,
            max_kv_rows=64,
            v_head_dim=128,
            indexer_num_q_heads=2,
            max_page_table_width=4,
        )
    )
    assert attn_ws.ragged_kv_cache is not None
    assert _storage_ptr(attn_ws.ragged_kv_cache) == base_ptr

    paged_ws = lane.make_paged_attention_workspace(
        PagedAttentionWorkspaceContract(
            mode="extend",
            max_total_q=8,
            max_batch=2,
            max_page_table_width=4,
            max_work_items=16,
            max_partial_rows=32,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim_qk=128,
            head_dim_vo=128,
            num_cache_pages=8,
        )
    )
    assert paged_ws.request_indices is not None
    assert _storage_ptr(paged_ws.request_indices) == base_ptr
    assert paged_ws.shared_arena is lane.shared_arena

    pool = lane.get_moe_workspace_pool()
    route_ws = tp_moe._get_route_workspace(
        torch.empty(2, 16, dtype=torch.bfloat16, device=device),
        num_experts=8,
        top_k=2,
        logits_dtype=torch.bfloat16,
        workspace=pool,
    )
    assert route_ws is not None
    assert _storage_ptr(route_ws.router_logits) == base_ptr

    a1_gscale = torch.ones(8, dtype=torch.float32, device=device)
    a2_gscale = torch.ones(8, dtype=torch.float32, device=device)
    moe_ws = tp_moe._alloc_workspace(
        "static",
        "nvfp4",
        8,
        8,
        16,
        16,
        2,
        device,
        torch.bfloat16,
        a1_gscale,
        a2_gscale,
        routed_rows=8,
        max_rows=8,
        input_scales_static=True,
        pool=pool,
        storage_key=("static",),
    )
    assert _storage_ptr(moe_ws.packed_input) == base_ptr
    assert (
        moe_ws.packed_input.data_ptr()
        >= lane.shared_arena.data_ptr() + pool.core_arena_offset_bytes
    )
    assert (
        route_ws.router_logits.data_ptr()
        < lane.shared_arena.data_ptr() + pool.core_arena_offset_bytes
    )
    assert pool.shared_arena is lane.shared_arena


def test_moe_pool_keys_are_not_stream_partitioned() -> None:
    key = tp_moe._workspace_pool_key(
        "static",
        quant_mode="nvfp4",
        activation="silu",
        state_E=4,
        weight_E=8,
        max_rows=4,
        k=16,
        n=16,
        num_topk=2,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )
    assert all("stream" not in repr(part).lower() for part in key)


def test_moe_arena_caps_env_defaults_to_w4a16(monkeypatch) -> None:
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    caps = B12XMoEArenaCaps(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        weight_E=8,
        k=16,
        n=16,
        num_topk=2,
        max_tokens=4,
    )

    assert caps.quant_mode == "w4a16"


def test_w4a16_joint_arena_materializes_planned_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_w4a16_prewarm(workspace, *, token_counts, **_kwargs) -> None:
        counts = tuple(int(token_count) for token_count in token_counts)
        workspace.planned_fused_moe_launches = {
            token_count: object() for token_count in counts
        }
        workspace.planned_topk_sum_launches = {
            token_count: object() for token_count in counts
        }

    monkeypatch.setattr(tp_moe, "get_num_sm", lambda _device: 48)
    monkeypatch.setattr(
        tp_moe,
        "_prewarm_w4a16_planned_launches",
        _fake_w4a16_prewarm,
    )
    device = _cuda_device()
    caps = B12XMoEArenaCaps(
        device=device,
        dtype=torch.bfloat16,
        quant_mode="w4a16",
        weight_E=8,
        route_num_experts=8,
        k=16,
        n=16,
        num_topk=2,
        max_tokens=4,
        core_token_counts=(1, 4),
    )
    lane = B12XExecutionLaneArena.allocate(
        B12XJointArenaSpec(device=device, moe_caps=caps)
    )
    pool = lane.get_moe_workspace_pool()
    plan = tp_moe._make_workspace_plan(
        num_tokens=4,
        weight_E=8,
        k=16,
        n=16,
        num_topk=2,
        device=device,
        dtype=torch.bfloat16,
        quant_mode="w4a16",
    )
    key = tp_moe._workspace_pool_key(
        plan.implementation,
        quant_mode=plan.quant_mode,
        activation=plan.activation,
        state_E=plan.state_E,
        weight_E=plan.weight_E,
        max_rows=plan.max_rows,
        k=plan.k,
        n=plan.n,
        num_topk=plan.num_topk,
        device=plan.device,
        dtype=plan.dtype,
    )

    workspace = pool.workspaces.get(key)
    assert isinstance(workspace, tp_moe.TPW4A16Workspace)
    assert workspace.planned_token_counts == frozenset({1, 4})
    assert workspace.planned_apply_router_weight_on_input is False
    assert workspace.planned_swiglu_limit is None
    assert workspace.routed_rows_capacity >= plan.routed_rows
    assert _storage_ptr(workspace.intermediate_cache13) == _storage_ptr(
        lane.shared_arena
    )
    assert (
        workspace.intermediate_cache13.data_ptr()
        >= lane.shared_arena.data_ptr() + pool.core_arena_offset_bytes
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    resolved = tp_moe._resolve_workspace(
        pool,
        plan=plan,
        a1_gscale=torch.empty(0, dtype=torch.float32, device=device),
        a2_gscale=torch.empty(0, dtype=torch.float32, device=device),
        input_scales_static=True,
    )
    assert resolved is workspace

    tail_plan = tp_moe._make_workspace_plan(
        num_tokens=2,
        weight_E=8,
        k=16,
        n=16,
        num_topk=2,
        device=device,
        dtype=torch.bfloat16,
        quant_mode="w4a16",
    )
    resolved_tail = tp_moe._resolve_workspace(
        pool,
        plan=tail_plan,
        a1_gscale=torch.empty(0, dtype=torch.float32, device=device),
        a2_gscale=torch.empty(0, dtype=torch.float32, device=device),
        input_scales_static=True,
    )
    assert resolved_tail is workspace

    oversized_plan = tp_moe._make_workspace_plan(
        num_tokens=8,
        weight_E=8,
        k=16,
        n=16,
        num_topk=2,
        device=device,
        dtype=torch.bfloat16,
        quant_mode="w4a16",
    )
    with pytest.raises(RuntimeError, match="capacity is too small|unplanned token count"):
        tp_moe._resolve_workspace(
            pool,
            plan=oversized_plan,
            a1_gscale=torch.empty(0, dtype=torch.float32, device=device),
            a2_gscale=torch.empty(0, dtype=torch.float32, device=device),
            input_scales_static=True,
        )


def test_w4a16_preplanned_launches_use_packed_layout() -> None:
    empty = torch.empty(0)
    workspace = tp_moe.TPW4A16Workspace(
        implementation="w4a16",
        quant_mode="w4a16",
        activation="silu",
        state_E=8,
        weight_E=8,
        max_rows=8,
        k=16,
        n=16,
        num_topk=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        routed_rows_capacity=8,
        intermediate_cache13=empty,
        intermediate_cache2=empty,
        fc1_c_tmp=empty,
        fc2_c_tmp=empty,
        packed_route_indices=empty,
        block_expert_ids=empty,
        packed_route_count=empty,
        expert_offsets=empty,
        planned_token_counts=frozenset({4}),
        planned_fused_moe_launches={
            ("packed", 4): "packed-launch",
        },
        planned_topk_sum_launches={4: "topk-sum"},
    )

    assert tp_moe._w4a16_preplanned_launches(
        workspace,
        token_count=4,
        weight_layout="packed",
    ) == ("packed-launch", "topk-sum")


def _tc_decode_selection_workspace() -> "tp_moe.TPW4A16Workspace":
    empty = torch.empty(0)
    return tp_moe.TPW4A16Workspace(
        implementation="w4a16",
        quant_mode="w4a16",
        activation="silu",
        state_E=8,
        weight_E=8,
        max_rows=8,
        k=16,
        n=16,
        num_topk=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        routed_rows_capacity=16,
        intermediate_cache13=empty,
        intermediate_cache2=empty,
        fc1_c_tmp=empty,
        fc2_c_tmp=empty,
        packed_route_indices=empty,
        block_expert_ids=empty,
        packed_route_count=empty,
        expert_offsets=empty,
        planned_token_counts=frozenset({1, 2, 3, 4, 6, 8}),
        planned_fused_moe_launches={
            ("packed", "e4m3_k16", tc): f"packed-launch-{tc}"
            for tc in (1, 2, 3, 4, 6, 8)
        },
        planned_topk_sum_launches={tc: f"topk-sum-{tc}" for tc in (1, 2, 3, 4, 6, 8)},
        # Built for the whole small-M range, including the MTP verify sizes 3, 6.
        planned_tc_decode_launches={
            tc: f"tc-decode-{tc}" for tc in (1, 2, 3, 4, 6, 8)
        },
    )


def test_w4a16_preplanned_launches_select_tc_decode_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_W4A16_TC_DECODE", "1")
    workspace = _tc_decode_selection_workspace()

    # Every small-M decode size (m <= 8), including the MTP verify sizes 3 and 6,
    # routes to the TC-decode fused-sum launch when one was preplanned.
    for tc in (1, 2, 3, 4, 6, 8):
        fused, _ = tp_moe._w4a16_preplanned_launches(
            workspace, token_count=tc, weight_layout="packed"
        )
        assert fused == f"tc-decode-{tc}", (tc, fused)

    # A small-M size with no preplanned TC-decode launch falls back to the
    # general packed fused launch rather than erroring.
    del workspace.planned_tc_decode_launches[3]
    fused, _ = tp_moe._w4a16_preplanned_launches(
        workspace, token_count=3, weight_layout="packed"
    )
    assert fused == "packed-launch-3", fused


def test_w4a16_preplanned_launches_ignore_tc_decode_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_W4A16_TC_DECODE", "0")
    workspace = _tc_decode_selection_workspace()

    # With the flag off the general packed launch is selected for every size,
    # even though TC-decode launches were preplanned.
    for tc in (1, 2, 4, 8):
        fused, _ = tp_moe._w4a16_preplanned_launches(
            workspace, token_count=tc, weight_layout="packed"
        )
        assert fused == f"packed-launch-{tc}", (tc, fused)


def test_nvfp4_joint_arena_materializes_planned_workspaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    device = torch.device("cpu")
    num_topk = 8
    static_tokens = tp_moe._get_static_compact_cutover_pairs("nvfp4") // num_topk
    dynamic_tokens = static_tokens + 1
    caps = B12XMoEArenaCaps(
        device=device,
        dtype=torch.bfloat16,
        quant_mode="nvfp4",
        weight_E=32,
        route_num_experts=32,
        k=128,
        n=128,
        num_topk=num_topk,
        max_tokens=dynamic_tokens,
        core_token_counts=(1, dynamic_tokens),
    )
    lane = B12XExecutionLaneArena.allocate(
        B12XJointArenaSpec(device=device, moe_caps=caps)
    )
    pool = lane.get_moe_workspace_pool()

    def _key(plan: tp_moe.TPMoEPlan) -> tuple:
        return tp_moe._workspace_pool_key(
            plan.implementation,
            quant_mode=plan.quant_mode,
            activation=plan.activation,
            state_E=plan.state_E,
            weight_E=plan.weight_E,
            max_rows=plan.max_rows,
            k=plan.k,
            n=plan.n,
            num_topk=plan.num_topk,
            device=plan.device,
            dtype=plan.dtype,
        )

    static_plan = tp_moe._make_workspace_plan(
        num_tokens=static_tokens,
        weight_E=32,
        k=128,
        n=128,
        num_topk=num_topk,
        device=device,
        dtype=torch.bfloat16,
        quant_mode="nvfp4",
    )
    dynamic_plan = tp_moe._make_workspace_plan(
        num_tokens=dynamic_tokens,
        weight_E=32,
        k=128,
        n=128,
        num_topk=num_topk,
        device=device,
        dtype=torch.bfloat16,
        quant_mode="nvfp4",
    )
    assert static_plan.implementation == "static"
    assert dynamic_plan.implementation == "dynamic"

    static_workspace = pool.workspaces.get(_key(static_plan))
    dynamic_workspace = pool.workspaces.get(_key(dynamic_plan))
    assert isinstance(static_workspace, tp_moe.TPCompactStaticWorkspace)
    assert isinstance(dynamic_workspace, tp_moe.TPDynamicWorkspace)
    assert (
        static_workspace.row_counts.data_ptr()
        >= lane.shared_arena.data_ptr() + pool.core_arena_offset_bytes
    )
    assert (
        dynamic_workspace.row_counts.data_ptr()
        >= lane.shared_arena.data_ptr() + pool.core_arena_offset_bytes
    )

    a1_gscale = torch.ones(1, dtype=torch.float32, device=device)
    a2_gscale = torch.ones(1, dtype=torch.float32, device=device)
    assert (
        tp_moe._resolve_workspace(
            pool,
            plan=static_plan,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            input_scales_static=True,
        )
        is static_workspace
    )
    assert (
        tp_moe._resolve_workspace(
            pool,
            plan=dynamic_plan,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            input_scales_static=True,
        )
        is dynamic_workspace
    )


def test_joint_arena_materializes_mixed_moe_quant_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_w4a16_prewarm(workspace, *, token_counts, **_kwargs) -> None:
        counts = tuple(int(token_count) for token_count in token_counts)
        workspace.planned_fused_moe_launches = {
            ("packed", token_count): object() for token_count in counts
        }
        workspace.planned_topk_sum_launches = {
            token_count: object() for token_count in counts
        }

    monkeypatch.setattr(
        tp_moe,
        "_prewarm_w4a16_planned_launches",
        _fake_w4a16_prewarm,
    )
    monkeypatch.setattr(tp_moe, "get_num_sm", lambda _device: 48)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    device = torch.device("cpu")
    num_topk = 8
    dynamic_tokens = tp_moe._get_static_compact_cutover_pairs("nvfp4") // num_topk + 1
    nvfp4_caps = B12XMoEArenaCaps(
        device=device,
        dtype=torch.bfloat16,
        quant_mode="nvfp4",
        weight_E=32,
        route_num_experts=32,
        k=128,
        n=128,
        num_topk=num_topk,
        max_tokens=dynamic_tokens,
        core_token_counts=(1, dynamic_tokens),
    )
    w4a16_caps = replace(nvfp4_caps, quant_mode="w4a16")
    lane = B12XExecutionLaneArena.allocate(
        B12XJointArenaSpec(device=device, moe_caps=(nvfp4_caps, w4a16_caps))
    )
    pool = lane.get_moe_workspace_pool()
    nvfp4_layout = nvfp4_caps.layout()
    w4a16_layout = w4a16_caps.layout()

    assert lane.moe_nbytes == max(
        nvfp4_layout.route_workspace_nbytes,
        w4a16_layout.route_workspace_nbytes,
    ) + max(
        nvfp4_layout.core_workspace_nbytes,
        w4a16_layout.core_workspace_nbytes,
    )

    def _key(plan: tp_moe.TPMoEPlan) -> tuple:
        return tp_moe._workspace_pool_key(
            plan.implementation,
            quant_mode=plan.quant_mode,
            activation=plan.activation,
            state_E=plan.state_E,
            weight_E=plan.weight_E,
            max_rows=plan.max_rows,
            k=plan.k,
            n=plan.n,
            num_topk=plan.num_topk,
            device=plan.device,
            dtype=plan.dtype,
        )

    nvfp4_plan = tp_moe._make_workspace_plan(
        num_tokens=dynamic_tokens,
        weight_E=32,
        k=128,
        n=128,
        num_topk=num_topk,
        device=device,
        dtype=torch.bfloat16,
        quant_mode="nvfp4",
    )
    w4a16_plan = tp_moe._make_workspace_plan(
        num_tokens=dynamic_tokens,
        weight_E=32,
        k=128,
        n=128,
        num_topk=num_topk,
        device=device,
        dtype=torch.bfloat16,
        quant_mode="w4a16",
    )

    assert isinstance(pool.workspaces.get(_key(nvfp4_plan)), tp_moe.TPDynamicWorkspace)
    w4a16_workspace = pool.workspaces.get(_key(w4a16_plan))
    assert isinstance(w4a16_workspace, tp_moe.TPW4A16Workspace)
    assert w4a16_workspace.planned_token_counts == frozenset(
        w4a16_layout.core_token_counts
    )


def test_moe_arena_caps_cover_static_boundary_shapes() -> None:
    device = torch.device("cpu")
    num_topk = 8
    static_tokens = tp_moe._get_static_compact_cutover_pairs() // num_topk
    assert static_tokens >= 1

    caps = B12XMoEArenaCaps(
        device=device,
        dtype=torch.bfloat16,
        weight_E=256,
        route_num_experts=256,
        k=6144,
        n=256,
        num_topk=num_topk,
        max_tokens=4096,
        core_token_counts=(4096, 32),
    )
    layout = caps.layout()

    plan = tp_moe._make_workspace_plan(
        num_tokens=static_tokens,
        weight_E=256,
        k=6144,
        n=256,
        num_topk=num_topk,
        device=device,
        dtype=torch.bfloat16,
    )
    core_plan = tp_moe._plan_core_workspace(
        plan.implementation,
        plan.quant_mode,
        plan.state_E,
        plan.weight_E,
        plan.k,
        plan.n,
        plan.num_topk,
        plan.device,
        plan.dtype,
        routed_rows=plan.routed_rows,
        max_rows=plan.max_rows,
        dynamic_physical_tiles=plan.dynamic_physical_tiles,
        dynamic_task_capacity=plan.dynamic_task_capacity,
    )

    assert plan.implementation == "static"
    assert layout.core_workspace_nbytes >= tp_moe._core_workspace_nbytes(core_plan)


def test_shared_moe_workspace_resets_overlaid_barrier_state() -> None:
    device = _cuda_device()
    lane = B12XExecutionLaneArena.allocate(
        B12XJointArenaSpec(
            device=device,
            attention_caps=_attention_caps(device),
            moe_caps=_moe_caps(device),
        )
    )
    pool = lane.get_moe_workspace_pool()

    a1_gscale = torch.ones(8, dtype=torch.float32, device=device)
    a2_gscale = torch.ones(8, dtype=torch.float32, device=device)
    moe_ws = tp_moe._alloc_workspace(
        "static",
        "nvfp4",
        8,
        8,
        16,
        16,
        2,
        device,
        torch.bfloat16,
        a1_gscale,
        a2_gscale,
        routed_rows=8,
        max_rows=8,
        input_scales_static=True,
        pool=pool,
        storage_key=("static", "volatile"),
    )

    assert moe_ws.volatile_launch_state
    moe_ws.barrier_count.fill_(123)
    moe_ws.barrier_epoch.fill_(456)
    tp_moe._reset_volatile_launch_state(moe_ws)
    assert int(moe_ws.barrier_count[0].item()) == 0
    assert int(moe_ws.barrier_epoch[0].item()) == 0


def test_execution_lane_arena_reuses_target_for_smaller_draft_caps() -> None:
    device = _cuda_device()
    _EXECUTION_LANES.clear()
    try:
        target_spec = B12XJointArenaSpec(
            device=device,
            paged_attention_caps=_paged_attention_caps(device),
            moe_caps=_moe_caps(device),
        )
        target_lane = ensure_b12x_execution_lane_arena(target_spec)

        draft_spec = B12XJointArenaSpec(
            device=device,
            paged_attention_caps=replace(
                _paged_attention_caps(device),
                num_q_heads=2,
                max_total_q=4,
                max_batch=1,
                max_work_items=8,
                max_partial_rows=16,
            ),
            moe_caps=replace(_moe_caps(device), max_tokens=2),
        )
        draft_lane = ensure_b12x_execution_lane_arena(draft_spec)

        assert draft_lane is target_lane
        assert draft_lane.arena is target_lane.arena
    finally:
        _EXECUTION_LANES.clear()


def test_execution_lane_arena_rejects_incompatible_draft_geometry() -> None:
    device = _cuda_device()
    _EXECUTION_LANES.clear()
    try:
        target_spec = B12XJointArenaSpec(
            device=device,
            paged_attention_caps=_paged_attention_caps(device),
            moe_caps=_moe_caps(device),
        )
        ensure_b12x_execution_lane_arena(target_spec)

        draft_spec = B12XJointArenaSpec(
            device=device,
            paged_attention_caps=replace(
                _paged_attention_caps(device),
                kv_dtype=torch.float32,
            ),
            moe_caps=_moe_caps(device),
        )
        with pytest.raises(RuntimeError, match="incompatible sizing caps"):
            ensure_b12x_execution_lane_arena(draft_spec)
    finally:
        _EXECUTION_LANES.clear()
