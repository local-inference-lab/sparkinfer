from __future__ import annotations

import torch

import b12x.integration.tp_moe as tp_moe
from b12x.attention.mla.workspace import (
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
)


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


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
    device = torch.device("cpu")
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
    assert lane.shared_arena_nbytes == max(expected_attn, expected_paged_attn, expected_moe, 1)


def test_joint_arena_views_share_one_backing_allocation() -> None:
    device = torch.device("cpu")
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
    assert moe_ws.packed_input.data_ptr() >= lane.shared_arena.data_ptr() + pool.core_arena_offset_bytes
    assert route_ws.router_logits.data_ptr() < lane.shared_arena.data_ptr() + pool.core_arena_offset_bytes
    assert pool.shared_arena is lane.shared_arena


def test_moe_pool_keys_are_not_stream_partitioned() -> None:
    key = tp_moe._workspace_pool_key(
        "static",
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
    device = torch.device("cpu")
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
    tp_moe._prepare_workspace_for_launch(moe_ws)
    assert int(moe_ws.barrier_count.item()) == 0
    assert int(moe_ws.barrier_epoch.item()) == 0
