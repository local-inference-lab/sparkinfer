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
from b12x.distributed._oneshot_common import SIGNAL_BYTES
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    _append_expert_bank,
    _append_shared_expert_routing,
    _b12x_gemma_moe_block_fp4_static,
    _b12x_gemma_moe_block_fp4_static_monolithic,
    _b12x_gemma_moe_block_fp4_static_producer,
    _shared_expert_gate_weights,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    b12x_route_experts_fast,
    clear_tp_moe_caches,
)
from b12x.moe.fused.pre_mlp_static import UnifiedPreMLPIPC
from b12x.moe.fused.reference import compare_to_reference


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


def _gemma_rmsnorm_after_allreduce(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = hidden_states + residual
    x_fp32 = residual_out.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    normed = (x_fp32 * inv_rms * (1.0 + norm_weight.float())).to(dtype=hidden_states.dtype)
    return normed, residual_out


@functools.lru_cache(maxsize=1)
def _load_norm_weight() -> torch.Tensor:
    return IndexedSafetensorLoader(MODEL_PATH).get_tensor(
        "model.language_model.layers.0.post_attention_layernorm.weight"
    ).to(device=torch.device("cuda"), dtype=torch.bfloat16).contiguous()


def test_static_full_block_path_matches_semi_fused_baseline() -> None:
    _skip_if_unavailable()
    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=0)
    gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=0)
    shared_gate_weight = load_shared_gate_weight(MODEL_PATH, layer_idx=0)
    sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
    shared_expert = _pack_shared_expert(shared_weights)
    combined_experts = _append_expert_bank(sparse_experts, shared_expert)
    norm_weight = _load_norm_weight()
    hidden_states = make_input_activations(spec, 4, seed=5100, device=device)
    residual = make_input_activations(spec, 4, seed=5200, device=device)
    normed_hidden_states, residual_expected = _gemma_rmsnorm_after_allreduce(
        hidden_states,
        residual,
        norm_weight,
        1e-6,
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
            del peer_input_ptrs
            assert inp is hidden_states
            assert residual_in is residual
            assert weight is norm_weight
            assert eps == 1e-6
            if out is None:
                out = torch.empty_like(normed_hidden_states)
            out.copy_(normed_hidden_states)
            if residual_out is None:
                residual_out = torch.empty_like(residual_expected)
            residual_out.copy_(residual_expected)
            return out, residual_out

    fused_pool = allocate_tp_moe_workspace_pool()
    semi_pool = allocate_tp_moe_workspace_pool()
    fused_out, fused_residual_out, fused_routing = _b12x_gemma_moe_block_fp4_static(
        hidden_states,
        residual,
        pre_mlp_runtime=_FakePreMLPRuntime(),
        norm_weight=norm_weight,
        norm_eps=1e-6,
        sparse_experts=sparse_experts,
        shared_expert=shared_expert,
        shared_gate_weight=shared_gate_weight,
        combined_experts=combined_experts,
        workspace=fused_pool,
        top_k=spec.top_k,
        gate_weight=gate_weight,
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        fc1_tile_amax=False,
        return_routing=True,
    )

    sparse_routing = b12x_route_experts_fast(
        normed_hidden_states,
        top_k=spec.top_k,
        gate_weight=gate_weight,
        workspace=semi_pool,
    )
    semi_routing = _append_shared_expert_routing(
        sparse_routing,
        shared_gate_weights=_shared_expert_gate_weights(
            normed_hidden_states,
            gate_weight=shared_gate_weight,
        ),
        shared_expert_id=sparse_experts.w1_fp4.shape[0],
    )
    semi_out = b12x_moe_fp4(
        normed_hidden_states,
        combined_experts.a1_gscale,
        combined_experts.w1_fp4,
        combined_experts.w1_blockscale,
        combined_experts.w1_alphas,
        combined_experts.a2_gscale,
        combined_experts.w2_fp4,
        combined_experts.w2_blockscale,
        combined_experts.w2_alphas,
        semi_routing.topk_weights,
        semi_routing.topk_ids,
        workspace=semi_pool,
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        fc2_tile_amax=False,
    )
    torch.cuda.synchronize()

    assert torch.equal(fused_routing.topk_ids, semi_routing.topk_ids)
    routing_weight_metrics = compare_to_reference(
        fused_routing.topk_weights,
        semi_routing.topk_weights,
    )
    assert routing_weight_metrics.max_abs == 0.0
    assert routing_weight_metrics.cos == 1.0

    output_metrics = compare_to_reference(fused_out, semi_out)
    residual_metrics = compare_to_reference(fused_residual_out, residual_expected)
    # Shared+sparse accumulation order differs slightly between the fully fused
    # prequantized path and the BF16 semi-fused baseline. Keep cosine strict
    # and allow a small number of output-level BF16 ulps.
    assert output_metrics.max_abs <= 5e-4
    assert output_metrics.rmse <= 3e-5
    assert output_metrics.mean_abs <= 1.2e-5
    assert output_metrics.cos > 0.999
    assert residual_metrics.max_abs == 0.0
    assert residual_metrics.cos == 1.0


def test_static_producer_full_block_path_matches_existing_wrapper() -> None:
    _skip_if_unavailable()
    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=0)
    gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=0)
    shared_gate_weight = load_shared_gate_weight(MODEL_PATH, layer_idx=0)
    sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
    shared_expert = _pack_shared_expert(shared_weights)
    combined_experts = _append_expert_bank(sparse_experts, shared_expert)
    norm_weight = _load_norm_weight()
    hidden_states = make_input_activations(spec, 4, seed=5300, device=device)
    residual = make_input_activations(spec, 4, seed=5400, device=device)
    normed_hidden_states, residual_expected = _gemma_rmsnorm_after_allreduce(
        hidden_states,
        residual,
        norm_weight,
        1e-6,
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
            del peer_input_ptrs
            assert inp is hidden_states
            assert residual_in is residual
            assert weight is norm_weight
            assert eps == 1e-6
            if out is None:
                out = torch.empty_like(normed_hidden_states)
            out.copy_(normed_hidden_states)
            if residual_out is None:
                residual_out = torch.empty_like(residual_expected)
            residual_out.copy_(residual_expected)
            return out, residual_out

    ref_pool = allocate_tp_moe_workspace_pool()
    producer_pool = allocate_tp_moe_workspace_pool()
    ref_out, ref_residual_out, ref_routing = _b12x_gemma_moe_block_fp4_static(
        hidden_states,
        residual,
        pre_mlp_runtime=_FakePreMLPRuntime(),
        norm_weight=norm_weight,
        norm_eps=1e-6,
        sparse_experts=sparse_experts,
        shared_expert=shared_expert,
        shared_gate_weight=shared_gate_weight,
        combined_experts=combined_experts,
        workspace=ref_pool,
        top_k=spec.top_k,
        gate_weight=gate_weight,
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        fc1_tile_amax=False,
        return_routing=True,
    )
    producer_out, producer_residual_out, producer_routing = _b12x_gemma_moe_block_fp4_static_producer(
        hidden_states,
        residual,
        pre_mlp_runtime=_FakePreMLPRuntime(),
        norm_weight=norm_weight,
        norm_eps=1e-6,
        sparse_experts=sparse_experts,
        shared_expert=shared_expert,
        shared_gate_weight=shared_gate_weight,
        combined_experts=combined_experts,
        workspace=producer_pool,
        top_k=spec.top_k,
        gate_weight=gate_weight,
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        fc1_tile_amax=False,
        return_routing=True,
    )
    torch.cuda.synchronize()

    assert torch.equal(producer_routing.topk_ids, ref_routing.topk_ids)
    routing_metrics = compare_to_reference(
        producer_routing.topk_weights,
        ref_routing.topk_weights,
    )
    output_metrics = compare_to_reference(producer_out, ref_out)
    residual_metrics = compare_to_reference(producer_residual_out, ref_residual_out)
    assert routing_metrics.cos > 0.999
    assert output_metrics.cos > 0.999
    assert residual_metrics.cos == 1.0


def test_static_monolithic_full_block_path_matches_existing_wrapper() -> None:
    _skip_if_unavailable()
    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=0)
    gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=0)
    shared_gate_weight = load_shared_gate_weight(MODEL_PATH, layer_idx=0)
    sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
    shared_expert = _pack_shared_expert(shared_weights)
    combined_experts = _append_expert_bank(sparse_experts, shared_expert)
    norm_weight = _load_norm_weight()
    hidden_states = make_input_activations(spec, 4, seed=5500, device=device)
    residual = make_input_activations(spec, 4, seed=5600, device=device)
    normed_hidden_states, residual_expected = _gemma_rmsnorm_after_allreduce(
        hidden_states,
        residual,
        norm_weight,
        1e-6,
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
            del peer_input_ptrs
            assert inp is hidden_states
            assert residual_in is residual
            assert weight is norm_weight
            assert eps == 1e-6
            if out is None:
                out = torch.empty_like(normed_hidden_states)
            out.copy_(normed_hidden_states)
            if residual_out is None:
                residual_out = torch.empty_like(residual_expected)
            residual_out.copy_(residual_expected)
            return out, residual_out

    ref_pool = allocate_tp_moe_workspace_pool()
    mono_pool = allocate_tp_moe_workspace_pool()
    signal_storage = torch.zeros(SIGNAL_BYTES // 4, dtype=torch.int32, device=device)
    ipc = UnifiedPreMLPIPC(
        rank=0,
        world_size=1,
        signal_ptrs=(int(signal_storage.data_ptr()),),
        peer_input_ptrs=(int(hidden_states.data_ptr()),),
    )

    ref_out, ref_residual_out, ref_routing = _b12x_gemma_moe_block_fp4_static_producer(
        hidden_states,
        residual,
        pre_mlp_runtime=_FakePreMLPRuntime(),
        norm_weight=norm_weight,
        norm_eps=1e-6,
        sparse_experts=sparse_experts,
        shared_expert=shared_expert,
        shared_gate_weight=shared_gate_weight,
        combined_experts=combined_experts,
        workspace=ref_pool,
        top_k=spec.top_k,
        gate_weight=gate_weight,
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        fc1_tile_amax=False,
        return_routing=True,
    )
    mono_out, mono_residual_out, mono_routing = _b12x_gemma_moe_block_fp4_static_monolithic(
        hidden_states,
        residual,
        pre_mlp_ipc=ipc,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        sparse_experts=sparse_experts,
        shared_expert=shared_expert,
        shared_gate_weight=shared_gate_weight,
        combined_experts=combined_experts,
        workspace=mono_pool,
        top_k=spec.top_k,
        gate_weight=gate_weight,
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        fc1_tile_amax=False,
        return_routing=True,
    )
    torch.cuda.synchronize()

    assert torch.equal(mono_routing.topk_ids, ref_routing.topk_ids)
    routing_metrics = compare_to_reference(
        mono_routing.topk_weights,
        ref_routing.topk_weights,
    )
    output_metrics = compare_to_reference(mono_out, ref_out)
    residual_metrics = compare_to_reference(mono_residual_out, ref_residual_out)
    assert routing_metrics.cos > 0.999
    assert output_metrics.cos > 0.999
    assert residual_metrics.cos == 1.0
