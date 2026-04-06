from __future__ import annotations

import os
import pathlib
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

from benchmarks.benchmark_moe import (  # noqa: E402
    MODEL_PATH,
    ModelSpec,
    load_expert_weights,
    load_gate_weight,
    load_shared_expert_weights,
    load_shared_gate_weight,
    make_input_activations,
)
from benchmarks.benchmark_gemma_moe_block_paths import (  # noqa: E402
    _gemma_rmsnorm_after_allreduce,
    _load_post_attention_layernorm_weight,
    _pack_shared_expert,
    _pack_sparse_experts_per_expert,
)
from b12x.distributed.pcie_oneshot import PCIeOneshotAllReduce  # noqa: E402
from b12x.integration.tp_moe import (  # noqa: E402
    _append_expert_bank,
    _append_shared_expert_routing,
    _b12x_gemma_moe_block_fp4_static_superfused_parity,
    _effective_input_scales,
    _get_weight_views,
    _launch_prequantized_moe_consumer,
    _prepare_expert_scale,
    _shared_expert_gate_weights,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    b12x_route_experts_fast,
    clear_tp_moe_caches,
)
from b12x.moe.fused.monolithic_superfused_static_parity_runtime import (  # noqa: E402
    launch_superfused_static_parity,
)
from b12x.moe.fused.pre_mlp_static import UnifiedPreMLPIPC  # noqa: E402
from b12x.moe.fused.reference import compare_to_reference  # noqa: E402


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank0_print(msg: str) -> None:
    if _rank() == 0:
        print(msg, flush=True)


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=_world_size(),
        tp_rank=_rank(),
    )


def main() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", _local_rank()))
    torch.cuda.set_device(_local_rank())
    device = torch.device("cuda", _local_rank())
    torch.set_grad_enabled(False)

    clear_tp_moe_caches()
    spec = _make_spec()
    m = 2
    hidden_local = make_input_activations(spec, m, seed=20100 + _rank(), device=device)
    residual = make_input_activations(spec, m, seed=21100 + _rank(), device=device)
    max_input_bytes = m * spec.hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()
    runtime = PCIeOneshotAllReduce.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=max_input_bytes,
    )
    try:
        sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
        shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=0)
        gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=0)
        shared_gate_weight = load_shared_gate_weight(MODEL_PATH, layer_idx=0)
        norm_weight = _load_post_attention_layernorm_weight(MODEL_PATH, layer_idx=0, device=device)
        sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
        shared_expert = _pack_shared_expert(shared_weights)
        combined_experts = _append_expert_bank(sparse_experts, shared_expert)

        fused_pool = allocate_tp_moe_workspace_pool()
        semi_pool = allocate_tp_moe_workspace_pool()

        fused_out, fused_residual_out, fused_routing = _b12x_gemma_moe_block_fp4_static_superfused_parity(
            hidden_local,
            residual,
            pre_mlp_runtime=runtime,
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
        fused_workspace = fused_pool.workspaces[next(iter(fused_pool.workspaces))]

        reduced = hidden_local.clone()
        dist.all_reduce(reduced)
        normed_hidden_states, semi_residual_out = _gemma_rmsnorm_after_allreduce(
            reduced,
            residual,
            norm_weight,
            1e-6,
        )
        sparse_routing = b12x_route_experts_fast(
            normed_hidden_states,
            top_k=spec.top_k,
            gate_weight=gate_weight,
            workspace=semi_pool,
        )
        combined_routing = _append_shared_expert_routing(
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
            combined_routing.topk_weights,
            combined_routing.topk_ids,
            workspace=semi_pool,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
            fc2_tile_amax=False,
        )
        parity_static_consumer = _launch_prequantized_moe_consumer(
            normed_hidden_states,
            experts=combined_experts,
            workspace=fused_workspace,
            routing=fused_routing,
            input_scales_are_reciprocal=True,
            fc1_tile_amax=False,
            fc2_tile_amax=False,
        )
        ipc = UnifiedPreMLPIPC.from_oneshot_runtime(runtime, inp=hidden_local)
        weights = _get_weight_views(
            combined_experts.w1_fp4,
            combined_experts.w1_blockscale,
            combined_experts.w2_fp4,
            combined_experts.w2_blockscale,
            combined_experts.w1_alphas,
            combined_experts.w2_alphas,
            combined_experts.w2_fp4.shape[2] * 2,
            spec.hidden_size,
        )
        prequantized_kernel_out, _, _, _, _ = launch_superfused_static_parity(
            hidden_states=hidden_local,
            residual=residual,
            norm_weight=norm_weight,
            gate_weight=gate_weight,
            shared_gate_weight=shared_gate_weight,
            ipc=ipc,
            workspace=fused_workspace,
            weights=weights,
            input_global_scale=_effective_input_scales(
                combined_experts.a1_gscale,
                combined_experts.w1_fp4.shape[0],
                input_scales_are_reciprocal=True,
            ),
            expert_alpha=combined_experts.w1_alphas,
            down_alpha=combined_experts.w2_alphas,
            global_scale=_prepare_expert_scale(
                combined_experts.a2_gscale,
                combined_experts.w2_fp4.shape[0],
            ),
            num_sparse_experts=sparse_experts.w1_fp4.shape[0],
            top_k=spec.top_k,
            output=torch.empty_like(hidden_local),
            residual_out=torch.empty_like(hidden_local),
            input_scales_are_reciprocal=True,
            fast_math=False,
            fc2_tile_amax=False,
            renormalize_topk=True,
            eps=1e-6,
            prequantized_input=True,
        )
        torch.cuda.synchronize(device)

        fused_metrics = compare_to_reference(fused_out, semi_out)
        parity_static_metrics = compare_to_reference(parity_static_consumer, semi_out)
        prequantized_kernel_metrics = compare_to_reference(prequantized_kernel_out, semi_out)
        residual_metrics = compare_to_reference(fused_residual_out, semi_residual_out)
        _rank0_print(
            "superfused parity discriminator: "
            f"fused(max_abs={fused_metrics.max_abs:.3e}, cos={fused_metrics.cos:.6f}) "
            f"self_prequantized(max_abs={prequantized_kernel_metrics.max_abs:.3e}, cos={prequantized_kernel_metrics.cos:.6f}) "
            f"static_consumer_from_parity_ws(max_abs={parity_static_metrics.max_abs:.3e}, cos={parity_static_metrics.cos:.6f}) "
            f"residual(max_abs={residual_metrics.max_abs:.3e}, cos={residual_metrics.cos:.6f})"
        )
    finally:
        runtime.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
