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
    load_shared_expert_weights,
    load_shared_gate_weight,
    make_input_activations,
)
from benchmarks.checkpoint_loader import IndexedSafetensorLoader
from b12x.integration import tp_moe
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    TPCompactStaticWorkspace,
    _append_expert_bank,
    _append_shared_expert_routing,
    _b12x_gemma_moe_block_fp4_static,
    _expand_expert_vector,
    _effective_input_scales,
    _populate_static_prequantized_workspace_device,
    _shared_expert_gate_weights,
    allocate_tp_moe_workspace,
    b12x_route_experts_fast,
)
from b12x.moe.fused.pre_mlp_static import slice_c_compact_route_assignment
from b12x.moe.fused.pre_mlp_static_hot import (
    stage2_compact_route_metadata,
    stage2_quantize_fc1_inputs,
)


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


def _pack_sparse_experts_per_expert(weights) -> B12XFP4ExpertWeights:
    return B12XFP4ExpertWeights(
        a1_gscale=weights.w13_input_scale_per_expert.reciprocal().to(torch.float32),
        w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled,
        w1_alphas=weights.g1_alphas_per_expert,
        a2_gscale=weights.w2_input_scale_per_expert.reciprocal().to(torch.float32),
        w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled,
        w2_alphas=weights.g2_alphas_per_expert,
    )


def _pack_shared_expert(weights) -> B12XFP4ExpertWeights:
    return B12XFP4ExpertWeights(
        a1_gscale=weights.w13_input_scale_quant,
        w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled,
        w1_alphas=weights.g1_alphas,
        a2_gscale=weights.w2_input_scale_quant,
        w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled,
        w2_alphas=weights.g2_alphas,
    )


@functools.lru_cache(maxsize=1)
def _load_combined_experts() -> B12XFP4ExpertWeights:
    spec = _make_spec()
    sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=0)
    return _append_expert_bank(
        _pack_sparse_experts_per_expert(sparse_weights),
        _pack_shared_expert(shared_weights),
    )


@functools.lru_cache(maxsize=1)
def _load_expert_banks() -> tuple[B12XFP4ExpertWeights, B12XFP4ExpertWeights, B12XFP4ExpertWeights]:
    spec = _make_spec()
    sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=0)
    sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
    shared_expert = _pack_shared_expert(shared_weights)
    combined_experts = _append_expert_bank(sparse_experts, shared_expert)
    return sparse_experts, shared_expert, combined_experts


@functools.lru_cache(maxsize=1)
def _load_gate_weights():
    spec = _make_spec()
    return (
        load_gate_weight(MODEL_PATH, spec, layer_idx=0),
        load_shared_gate_weight(MODEL_PATH, layer_idx=0),
    )


@functools.lru_cache(maxsize=1)
def _load_norm_weight() -> torch.Tensor:
    return IndexedSafetensorLoader(MODEL_PATH).get_tensor(
        "model.language_model.layers.0.post_attention_layernorm.weight"
    ).to(device=torch.device("cuda"), dtype=torch.bfloat16).contiguous()


@pytest.mark.parametrize("m", [1, 2, 4])
def test_slice_c_matches_static_route_pack_metadata(m: int) -> None:
    _skip_if_unavailable()
    spec = _make_spec()
    combined_experts = _load_combined_experts()
    sparse_gate_weight, shared_gate_weight = _load_gate_weights()
    hidden_states = make_input_activations(spec, m, seed=7000 + m, device=torch.device("cuda"))

    sparse_routing = b12x_route_experts_fast(
        hidden_states,
        top_k=spec.top_k,
        gate_weight=sparse_gate_weight,
    )
    combined_routing = _append_shared_expert_routing(
        sparse_routing,
        shared_gate_weights=_shared_expert_gate_weights(
            hidden_states,
            gate_weight=shared_gate_weight,
        ),
        shared_expert_id=spec.num_experts,
    )

    slice_workspace = allocate_tp_moe_workspace(
        hidden_states,
        combined_experts.a1_gscale,
        combined_experts.w1_fp4,
        combined_experts.a2_gscale,
        combined_experts.w2_fp4,
        combined_routing.topk_ids,
        input_scales_static=True,
    )
    ref_workspace = allocate_tp_moe_workspace(
        hidden_states,
        combined_experts.a1_gscale,
        combined_experts.w1_fp4,
        combined_experts.a2_gscale,
        combined_experts.w2_fp4,
        combined_routing.topk_ids,
        input_scales_static=True,
    )
    assert isinstance(slice_workspace, TPCompactStaticWorkspace)
    assert isinstance(ref_workspace, TPCompactStaticWorkspace)

    slice_c = slice_c_compact_route_assignment(
        slice_workspace,
        topk_ids=combined_routing.topk_ids,
        topk_weights=combined_routing.topk_weights,
    )
    effective_input_scale = _effective_input_scales(
        combined_experts.a1_gscale,
        combined_experts.w1_fp4.shape[0],
        input_scales_are_reciprocal=True,
    )
    _populate_static_prequantized_workspace_device(
        ref_workspace,
        a=hidden_states,
        topk_ids=combined_routing.topk_ids,
        topk_weights=combined_routing.topk_weights,
        expert_input_scale=effective_input_scale,
        expert_alpha=_expand_expert_vector(
            combined_experts.w1_alphas,
            combined_experts.w1_fp4.shape[0],
            name="combined_experts.w1_alphas",
        ),
        fc1_tile_amax=False,
    )

    active_experts = int(ref_workspace.active_expert_count.item())
    assert int(slice_c.active_expert_count.item()) == active_experts
    assert torch.equal(
        slice_c.weight_expert_ids[:active_experts],
        ref_workspace.weight_expert_ids[:active_experts],
    )
    assert torch.equal(slice_c.global_to_local_expert, ref_workspace.global_to_local_expert)
    assert torch.equal(slice_c.row_counts, ref_workspace.row_counts)
    assert torch.equal(slice_c.token_map, ref_workspace.token_map)
    assert torch.equal(slice_c.token_weights, ref_workspace.token_weights)


@pytest.mark.parametrize("m", [1, 2, 4])
def test_stage2_compact_route_metadata_matches_slice_c(m: int) -> None:
    _skip_if_unavailable()
    spec = _make_spec()
    combined_experts = _load_combined_experts()
    sparse_gate_weight, shared_gate_weight = _load_gate_weights()
    hidden_states = make_input_activations(spec, m, seed=7200 + m, device=torch.device("cuda"))

    sparse_routing = b12x_route_experts_fast(
        hidden_states,
        top_k=spec.top_k,
        gate_weight=sparse_gate_weight,
    )
    combined_routing = _append_shared_expert_routing(
        sparse_routing,
        shared_gate_weights=_shared_expert_gate_weights(
            hidden_states,
            gate_weight=shared_gate_weight,
        ),
        shared_expert_id=spec.num_experts,
    )

    stage2_workspace = allocate_tp_moe_workspace(
        hidden_states,
        combined_experts.a1_gscale,
        combined_experts.w1_fp4,
        combined_experts.a2_gscale,
        combined_experts.w2_fp4,
        combined_routing.topk_ids,
        input_scales_static=True,
    )
    slice_workspace = allocate_tp_moe_workspace(
        hidden_states,
        combined_experts.a1_gscale,
        combined_experts.w1_fp4,
        combined_experts.a2_gscale,
        combined_experts.w2_fp4,
        combined_routing.topk_ids,
        input_scales_static=True,
    )
    assert isinstance(stage2_workspace, TPCompactStaticWorkspace)
    assert isinstance(slice_workspace, TPCompactStaticWorkspace)

    stage2_compact_route_metadata(
        stage2_workspace,
        topk_ids=combined_routing.topk_ids,
        topk_weights=combined_routing.topk_weights,
    )
    slice_c = slice_c_compact_route_assignment(
        slice_workspace,
        topk_ids=combined_routing.topk_ids,
        topk_weights=combined_routing.topk_weights,
    )

    active_experts = int(slice_c.active_expert_count.item())
    assert int(stage2_workspace.active_expert_count.item()) == active_experts
    assert torch.equal(
        stage2_workspace.weight_expert_ids[:active_experts],
        slice_c.weight_expert_ids[:active_experts],
    )
    assert torch.equal(stage2_workspace.global_to_local_expert, slice_c.global_to_local_expert)
    assert torch.equal(stage2_workspace.row_counts, slice_c.row_counts)
    assert torch.equal(stage2_workspace.token_map, slice_c.token_map)
    assert torch.equal(stage2_workspace.token_weights, slice_c.token_weights)


@pytest.mark.parametrize("m", [1, 2, 4])
def test_stage2_quantize_fc1_inputs_matches_slice_d(m: int) -> None:
    _skip_if_unavailable()
    spec = _make_spec()
    combined_experts = _load_combined_experts()
    sparse_gate_weight, shared_gate_weight = _load_gate_weights()
    hidden_states = make_input_activations(spec, m, seed=7300 + m, device=torch.device("cuda"))

    sparse_routing = b12x_route_experts_fast(
        hidden_states,
        top_k=spec.top_k,
        gate_weight=sparse_gate_weight,
    )
    combined_routing = _append_shared_expert_routing(
        sparse_routing,
        shared_gate_weights=_shared_expert_gate_weights(
            hidden_states,
            gate_weight=shared_gate_weight,
        ),
        shared_expert_id=spec.num_experts,
    )

    stage2_workspace = allocate_tp_moe_workspace(
        hidden_states,
        combined_experts.a1_gscale,
        combined_experts.w1_fp4,
        combined_experts.a2_gscale,
        combined_experts.w2_fp4,
        combined_routing.topk_ids,
        input_scales_static=True,
    )
    slice_workspace = allocate_tp_moe_workspace(
        hidden_states,
        combined_experts.a1_gscale,
        combined_experts.w1_fp4,
        combined_experts.a2_gscale,
        combined_experts.w2_fp4,
        combined_routing.topk_ids,
        input_scales_static=True,
    )
    assert isinstance(stage2_workspace, TPCompactStaticWorkspace)
    assert isinstance(slice_workspace, TPCompactStaticWorkspace)

    effective_input_scale = _effective_input_scales(
        combined_experts.a1_gscale,
        combined_experts.w1_fp4.shape[0],
        input_scales_are_reciprocal=True,
    )
    expert_alpha = _expand_expert_vector(
        combined_experts.w1_alphas,
        combined_experts.w1_fp4.shape[0],
        name="combined_experts.w1_alphas",
    )

    stage2_compact_route_metadata(
        stage2_workspace,
        topk_ids=combined_routing.topk_ids,
        topk_weights=combined_routing.topk_weights,
    )
    slice_c_compact_route_assignment(
        slice_workspace,
        topk_ids=combined_routing.topk_ids,
        topk_weights=combined_routing.topk_weights,
    )
    stage2_quantize_fc1_inputs(
        stage2_workspace,
        normalized_hidden_states=hidden_states,
        expert_input_scale=effective_input_scale,
        expert_alpha=expert_alpha,
        fc1_tile_amax=False,
    )
    from b12x.moe.fused.pre_mlp_static import slice_d_quantize_fc1_inputs

    slice_d_quantize_fc1_inputs(
        slice_workspace,
        normalized_hidden_states=hidden_states,
        expert_input_scale=effective_input_scale,
        expert_alpha=expert_alpha,
        fc1_tile_amax=False,
    )

    assert torch.equal(stage2_workspace.packed_input, slice_workspace.packed_input)
    assert torch.equal(stage2_workspace.packed_input_scale, slice_workspace.packed_input_scale)
    assert torch.equal(stage2_workspace.fc1_tile_scale, slice_workspace.fc1_tile_scale)
    assert torch.equal(stage2_workspace.fc1_tile_alpha, slice_workspace.fc1_tile_alpha)


@pytest.mark.parametrize("m", [1, 2, 4])
def test_slice_c_and_d_match_static_full_block_route_pack(m: int) -> None:
    _skip_if_unavailable()
    spec = _make_spec()
    sparse_experts, shared_expert, combined_experts = _load_expert_banks()
    sparse_gate_weight, shared_gate_weight = _load_gate_weights()
    norm_weight = _load_norm_weight()
    normalized_hidden_states = make_input_activations(
        spec,
        m,
        seed=7100 + m,
        device=torch.device("cuda"),
    )
    residual_expected = torch.zeros_like(normalized_hidden_states)

    sparse_routing = b12x_route_experts_fast(
        normalized_hidden_states,
        top_k=spec.top_k,
        gate_weight=sparse_gate_weight,
    )
    combined_routing = _append_shared_expert_routing(
        sparse_routing,
        shared_gate_weights=_shared_expert_gate_weights(
            normalized_hidden_states,
            gate_weight=shared_gate_weight,
        ),
        shared_expert_id=spec.num_experts,
    )

    slice_workspace = allocate_tp_moe_workspace(
        normalized_hidden_states,
        combined_experts.a1_gscale,
        combined_experts.w1_fp4,
        combined_experts.a2_gscale,
        combined_experts.w2_fp4,
        combined_routing.topk_ids,
        input_scales_static=True,
    )
    wrapper_workspace = allocate_tp_moe_workspace(
        normalized_hidden_states,
        combined_experts.a1_gscale,
        combined_experts.w1_fp4,
        combined_experts.a2_gscale,
        combined_experts.w2_fp4,
        combined_routing.topk_ids,
        input_scales_static=True,
    )
    assert isinstance(slice_workspace, TPCompactStaticWorkspace)
    assert isinstance(wrapper_workspace, TPCompactStaticWorkspace)

    effective_input_scale = _effective_input_scales(
        combined_experts.a1_gscale,
        combined_experts.w1_fp4.shape[0],
        input_scales_are_reciprocal=True,
    )
    expert_alpha = _expand_expert_vector(
        combined_experts.w1_alphas,
        combined_experts.w1_fp4.shape[0],
        name="combined_experts.w1_alphas",
    )
    slice_c_compact_route_assignment(
        slice_workspace,
        topk_ids=combined_routing.topk_ids,
        topk_weights=combined_routing.topk_weights,
    )
    from b12x.moe.fused.pre_mlp_static import slice_d_quantize_fc1_inputs

    slice_d_quantize_fc1_inputs(
        slice_workspace,
        normalized_hidden_states=normalized_hidden_states,
        expert_input_scale=effective_input_scale,
        expert_alpha=expert_alpha,
        fc1_tile_amax=False,
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
            del inp, residual_in, weight, eps, peer_input_ptrs
            if out is None:
                out = torch.empty_like(normalized_hidden_states)
            out.copy_(normalized_hidden_states)
            if residual_out is None:
                residual_out = torch.empty_like(residual_expected)
            residual_out.copy_(residual_expected)
            return out, residual_out

    dummy_hidden_states = torch.empty_like(normalized_hidden_states)
    dummy_residual = torch.empty_like(normalized_hidden_states)
    _b12x_gemma_moe_block_fp4_static(
        dummy_hidden_states,
        dummy_residual,
        pre_mlp_runtime=_FakePreMLPRuntime(),
        norm_weight=norm_weight,
        norm_eps=1e-6,
        sparse_experts=sparse_experts,
        shared_expert=shared_expert,
        shared_gate_weight=shared_gate_weight,
        combined_experts=combined_experts,
        workspace=wrapper_workspace,
        top_k=spec.top_k,
        gate_weight=sparse_gate_weight,
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        fc1_tile_amax=False,
    )
    torch.cuda.synchronize()

    active_experts = int(wrapper_workspace.active_expert_count.item())
    assert int(slice_workspace.active_expert_count.item()) == active_experts
    assert torch.equal(
        slice_workspace.weight_expert_ids[:active_experts],
        wrapper_workspace.weight_expert_ids[:active_experts],
    )
    assert torch.equal(slice_workspace.global_to_local_expert, wrapper_workspace.global_to_local_expert)
    assert torch.equal(slice_workspace.row_counts, wrapper_workspace.row_counts)
    assert torch.equal(slice_workspace.token_map, wrapper_workspace.token_map)
    assert torch.equal(slice_workspace.token_weights, wrapper_workspace.token_weights)
    assert torch.equal(slice_workspace.packed_input, wrapper_workspace.packed_input)
    assert torch.equal(slice_workspace.packed_input_scale, wrapper_workspace.packed_input_scale)
    assert torch.equal(slice_workspace.fc1_tile_scale, wrapper_workspace.fc1_tile_scale)
    assert torch.equal(slice_workspace.fc1_tile_alpha, wrapper_workspace.fc1_tile_alpha)
