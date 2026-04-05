from __future__ import annotations

import pytest
import torch

import b12x.integration.tp_moe as tp_moe
from benchmarks.benchmark_moe import MODEL_PATH, TP_RANK, TP_SIZE, ModelSpec, load_expert_weights, make_routed_inputs
from b12x.integration.tp_moe import (
    allocate_tp_moe_workspace,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    clear_tp_moe_caches,
)
from b12x.moe.fused.reference import compare_to_reference

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


def _dynamic_token_count(spec: ModelSpec) -> int:
    return (tp_moe._get_static_compact_cutover_pairs() // spec.top_k) + 1


def test_workspace_pool_handles_chunked_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x, topk_ids, topk_weights = make_routed_inputs(
        spec,
        _dynamic_token_count(spec),
        seed=321,
        device=device,
    )

    exact_workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )
    assert isinstance(exact_workspace, tp_moe.TPDynamicWorkspace)
    expected = b12x_moe_fp4(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        workspace=exact_workspace,
        input_scales_static=True,
    ).clone()
    torch.cuda.synchronize(device)

    pool = allocate_tp_moe_workspace_pool()
    monkeypatch.setattr(tp_moe, "_eager_dynamic_token_chunk_limit", lambda *args, **kwargs: 13)
    monkeypatch.setattr(tp_moe, "_dynamic_token_chunk_limit", lambda *_args: 13)
    actual = b12x_moe_fp4(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        workspace=pool,
        input_scales_static=True,
    ).clone()
    torch.cuda.synchronize(device)

    with pytest.raises(ValueError, match="chunked requests require a TPMoEWorkspacePool"):
        b12x_moe_fp4(
            x,
            weights.w13_input_scale_per_expert,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            weights.g1_alphas_per_expert,
            weights.w2_input_scale_per_expert,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            weights.g2_alphas_per_expert,
            topk_weights,
            topk_ids,
            workspace=exact_workspace,
            input_scales_static=True,
        )

    assert len(pool.workspaces) == 1
    metrics = compare_to_reference(actual, expected)
    assert metrics.max_abs <= 1e-3
    assert metrics.cos > 0.9998


def test_cuda_graph_capture_requires_output_buffer() -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x, topk_ids, topk_weights = make_routed_inputs(spec, 1, seed=654, device=device)
    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(ValueError, match="caller-owned output buffer"):
        with torch.cuda.graph(graph):
            b12x_moe_fp4(
                x,
                weights.w13_input_scale_per_expert,
                weights.w13_weight,
                weights.w13_blockscale_swizzled,
                weights.g1_alphas_per_expert,
                weights.w2_input_scale_per_expert,
                weights.w2_weight,
                weights.w2_blockscale_swizzled,
                weights.g2_alphas_per_expert,
                topk_weights,
                topk_ids,
                workspace=workspace,
                input_scales_static=True,
            )


def test_static_workspace_accepts_smaller_logical_requests() -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x_large, topk_ids_large, topk_weights_large = make_routed_inputs(spec, 12, seed=900, device=device)
    x_small, topk_ids_small, topk_weights_small = make_routed_inputs(spec, 1, seed=901, device=device)

    large_workspace = allocate_tp_moe_workspace(
        x_large,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids_large,
        input_scales_static=True,
    )
    assert isinstance(large_workspace, tp_moe.TPCompactStaticWorkspace)
    assert large_workspace.routed_rows_capacity == x_large.shape[0] * spec.top_k

    expected = b12x_moe_fp4(
        x_small,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights_small,
        topk_ids_small,
        workspace=allocate_tp_moe_workspace(
            x_small,
            weights.w13_input_scale_per_expert,
            weights.w13_weight,
            weights.w2_input_scale_per_expert,
            weights.w2_weight,
            topk_ids_small,
            input_scales_static=True,
        ),
        input_scales_static=True,
    ).clone()
    actual = b12x_moe_fp4(
        x_small,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights_small,
        topk_ids_small,
        workspace=large_workspace,
        input_scales_static=True,
    ).clone()
    torch.cuda.synchronize(device)

    metrics = compare_to_reference(actual, expected)
    assert metrics.max_abs <= 1e-3
    assert metrics.cos > 0.9998


def test_static_workspace_pool_reuses_largest_capacity() -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x_large, topk_ids_large, topk_weights_large = make_routed_inputs(spec, 12, seed=910, device=device)
    x_small, topk_ids_small, topk_weights_small = make_routed_inputs(spec, 2, seed=911, device=device)
    pool = allocate_tp_moe_workspace_pool()

    expected_large = b12x_moe_fp4(
        x_large,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights_large,
        topk_ids_large,
        workspace=allocate_tp_moe_workspace(
            x_large,
            weights.w13_input_scale_per_expert,
            weights.w13_weight,
            weights.w2_input_scale_per_expert,
            weights.w2_weight,
            topk_ids_large,
            input_scales_static=True,
        ),
        input_scales_static=True,
    ).clone()
    expected_small = b12x_moe_fp4(
        x_small,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights_small,
        topk_ids_small,
        workspace=allocate_tp_moe_workspace(
            x_small,
            weights.w13_input_scale_per_expert,
            weights.w13_weight,
            weights.w2_input_scale_per_expert,
            weights.w2_weight,
            topk_ids_small,
            input_scales_static=True,
        ),
        input_scales_static=True,
    ).clone()

    actual_large = b12x_moe_fp4(
        x_large,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights_large,
        topk_ids_large,
        workspace=pool,
        input_scales_static=True,
    ).clone()
    actual_small = b12x_moe_fp4(
        x_small,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights_small,
        topk_ids_small,
        workspace=pool,
        input_scales_static=True,
    ).clone()
    torch.cuda.synchronize(device)

    assert len(pool.workspaces) == 1
    pooled_workspace = next(iter(pool.workspaces.values()))
    assert isinstance(pooled_workspace, tp_moe.TPCompactStaticWorkspace)
    assert pooled_workspace.routed_rows_capacity == x_large.shape[0] * spec.top_k
    assert pooled_workspace.max_rows == x_large.shape[0] * spec.top_k
    assert pooled_workspace.state_E == x_large.shape[0] * spec.top_k

    large_metrics = compare_to_reference(actual_large, expected_large)
    assert large_metrics.max_abs <= 1e-3
    assert large_metrics.cos > 0.9998
    small_metrics = compare_to_reference(actual_small, expected_small)
    assert small_metrics.max_abs <= 1e-3
    assert small_metrics.cos > 0.9998


def test_static_prequantized_device_populator_matches_host_reference() -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x, topk_ids, topk_weights = make_routed_inputs(spec, 12, seed=1777, device=device)

    host_workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )
    device_workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )
    assert isinstance(host_workspace, tp_moe.TPCompactStaticWorkspace)
    assert isinstance(device_workspace, tp_moe.TPCompactStaticWorkspace)

    effective_input_scale = tp_moe._effective_input_scales(
        weights.w13_input_scale_per_expert,
        spec.num_experts,
        input_scales_are_reciprocal=False,
    )
    tp_moe._populate_static_prequantized_workspace_host(
        host_workspace,
        a=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        expert_input_scale=effective_input_scale,
        expert_alpha=weights.g1_alphas_per_expert,
        fc1_tile_amax=True,
    )
    tp_moe._populate_static_prequantized_workspace_device(
        device_workspace,
        a=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        expert_input_scale=effective_input_scale,
        expert_alpha=weights.g1_alphas_per_expert,
        fc1_tile_amax=True,
    )

    assert torch.equal(device_workspace.active_expert_count, host_workspace.active_expert_count)
    active_experts = int(host_workspace.active_expert_count.item())
    assert torch.equal(
        device_workspace.weight_expert_ids[:active_experts],
        host_workspace.weight_expert_ids[:active_experts],
    )
    assert torch.equal(device_workspace.global_to_local_expert, host_workspace.global_to_local_expert)
    assert torch.equal(device_workspace.row_counts, host_workspace.row_counts)
    assert torch.equal(device_workspace.token_map, host_workspace.token_map)
    assert torch.equal(device_workspace.token_weights, host_workspace.token_weights)
    assert torch.equal(device_workspace.packed_input, host_workspace.packed_input)
    assert torch.equal(device_workspace.packed_input_scale, host_workspace.packed_input_scale)
    assert torch.equal(device_workspace.fc1_tile_scale, host_workspace.fc1_tile_scale)
    assert torch.equal(device_workspace.fc1_tile_alpha, host_workspace.fc1_tile_alpha)


def test_dynamic_chunk_limit_uses_compact_layout() -> None:
    old_limit = tp_moe._safe_token_chunk(512, 4096, 256, 10)
    compact_limit = tp_moe._safe_dynamic_token_chunk(512, 4096, 256, 10)

    assert old_limit == 192
    assert compact_limit == 98304
    assert compact_limit > old_limit


def test_eager_dynamic_chunk_limit_uses_exact_routing_tiles() -> None:
    m = 98_305
    topk_ids = torch.arange(10, dtype=torch.int32).expand(m, -1).contiguous()

    eager_limit = tp_moe._eager_dynamic_token_chunk_limit(
        topk_ids,
        weight_E=512,
        k=4096,
        n=256,
        num_topk=10,
    )
    compact_limit = tp_moe._dynamic_token_chunk_limit(512, 4096, 256, 10)

    assert compact_limit == 98_304
    assert eager_limit == m
    assert eager_limit > compact_limit


def test_dynamic_task_geometry_caps_active_experts_by_routed_rows() -> None:
    max_phys_tiles, gate_tile_cnt, max_tasks = tp_moe._dynamic_task_geometry(512, 1024, 10)

    assert max_phys_tiles == 10
    assert gate_tile_cnt == 8
    assert max_tasks == 40


def test_dynamic_workspace_uses_compact_storage() -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x, topk_ids, _topk_weights = make_routed_inputs(
        spec,
        _dynamic_token_count(spec),
        seed=777,
        device=device,
    )

    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )
    assert isinstance(workspace, tp_moe.TPDynamicWorkspace)

    n = spec.intermediate_size // spec.tp_size
    max_phys_tiles, _, max_tasks = tp_moe._dynamic_task_geometry(
        spec.num_experts,
        n,
        x.shape[0] * spec.top_k,
    )
    rows_padded = max_phys_tiles * tp_moe._LEVEL_TILE_M
    cols_pad_k = tp_moe.align_up(spec.hidden_size // tp_moe._NVFP4_BLOCK_SIZE, 4)

    assert tuple(workspace.token_map.shape) == (rows_padded,)
    assert tuple(workspace.token_weights.shape) == (rows_padded,)
    assert tuple(workspace.packed_input.shape) == (1, rows_padded, spec.hidden_size // 2)
    assert tuple(workspace.packed_input_scale.shape) == (rows_padded, cols_pad_k)
    assert tuple(workspace.expert_write_rows.shape) == (spec.num_experts,)
    assert tuple(workspace.expert_tile_base.shape) == (spec.num_experts + 1,)
    assert workspace.routed_rows_capacity == x.shape[0] * spec.top_k
    assert workspace.physical_tiles_capacity == max_phys_tiles
    assert workspace.task_capacity == max_tasks
    assert tuple(workspace.tile_write_count.shape) == (max_phys_tiles,)
    assert tuple(workspace.task_ready.shape) == (max_tasks,)
    assert tp_moe.select_tp_moe_backend(num_tokens=x.shape[0], num_topk=spec.top_k) == "dynamic"


def test_dynamic_workspace_pool_uses_eager_routing_geometry() -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x, _topk_ids, _topk_weights = make_routed_inputs(
        spec,
        _dynamic_token_count(spec),
        seed=1701,
        device=device,
    )
    topk_ids = torch.arange(spec.top_k, dtype=torch.int32, device=device).expand(x.shape[0], -1).contiguous()
    topk_weights = torch.full(
        (x.shape[0], spec.top_k),
        1.0 / spec.top_k,
        dtype=torch.float32,
        device=device,
    )

    exact_workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )
    expected = b12x_moe_fp4(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        workspace=exact_workspace,
        input_scales_static=True,
    ).clone()

    pool = allocate_tp_moe_workspace_pool()
    actual = b12x_moe_fp4(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        workspace=pool,
        input_scales_static=True,
    ).clone()
    torch.cuda.synchronize(device)

    assert len(pool.workspaces) == 1
    pooled_workspace = next(iter(pool.workspaces.values()))
    assert isinstance(pooled_workspace, tp_moe.TPDynamicWorkspace)
    exact_tiles, _, exact_tasks = tp_moe._dynamic_task_geometry_from_routing(
        topk_ids,
        weight_E=spec.num_experts,
        n=spec.intermediate_size // spec.tp_size,
    )
    assert pooled_workspace.routed_rows_capacity == x.shape[0] * spec.top_k
    assert pooled_workspace.max_rows == tp_moe.align_up(x.shape[0] * spec.top_k, tp_moe._LEVEL_TILE_M)
    assert pooled_workspace.physical_tiles_capacity == exact_tiles
    assert pooled_workspace.task_capacity == exact_tasks
    assert tuple(pooled_workspace.packed_input.shape) == (
        1,
        exact_tiles * tp_moe._LEVEL_TILE_M,
        spec.hidden_size // 2,
    )

    metrics = compare_to_reference(actual, expected)
    assert metrics.max_abs <= 1e-3
    assert metrics.cos > 0.9998
