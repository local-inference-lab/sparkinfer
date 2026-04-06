#!/usr/bin/env python3
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
import torch.distributed as dist

from benchmarks.benchmark_gemma_moe_block_paths import (
    _load_post_attention_layernorm_weight,
    _pack_shared_expert,
    _pack_sparse_experts_per_expert,
)
from benchmarks.benchmark_moe import (
    MODEL_PATH,
    ModelSpec,
    load_expert_weights,
    load_gate_weight,
    load_shared_expert_weights,
    load_shared_gate_weight,
    make_input_activations,
    require_sm120,
)
from b12x.distributed.pcie_oneshot import PCIeOneshotAllReduce
from b12x.integration.tp_moe import (
    _append_expert_bank,
    _b12x_gemma_moe_block_fp4_dynamic_superfused,
    allocate_tp_moe_workspace_pool,
    clear_tp_moe_caches,
)


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


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
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    require_sm120()
    torch.set_grad_enabled(False)
    clear_tp_moe_caches()

    spec = _make_spec()
    num_tokens = 1
    max_input_bytes = num_tokens * spec.hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()
    runtime = PCIeOneshotAllReduce.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=max_input_bytes,
    )
    try:
        with torch.no_grad():
            sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
            shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=0)
            gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=0)
            shared_gate_weight = load_shared_gate_weight(MODEL_PATH, layer_idx=0)
            norm_weight = _load_post_attention_layernorm_weight(
                MODEL_PATH,
                layer_idx=0,
                device=device,
            )
            sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
            shared_expert = _pack_shared_expert(shared_weights)
            combined_experts = _append_expert_bank(sparse_experts, shared_expert)

            hidden_local = make_input_activations(
                spec,
                num_tokens,
                seed=123 + 17 * _rank() + num_tokens,
                device=device,
            )
            residual = make_input_activations(
                spec,
                num_tokens,
                seed=10_123 + num_tokens,
                device=device,
            )

            fused_pool = allocate_tp_moe_workspace_pool()
            fused_output = torch.empty_like(hidden_local)
            fused_residual_out = torch.empty_like(hidden_local)

            def superfused_path() -> None:
                _b12x_gemma_moe_block_fp4_dynamic_superfused(
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
                    output=fused_output,
                    residual_out=fused_residual_out,
                    input_scales_are_reciprocal=True,
                    input_scales_static=True,
                )

            superfused_path()
            torch.cuda.synchronize(device)

            cudart = torch.cuda.cudart()
            cudart.cudaProfilerStart()
            superfused_path()
            torch.cuda.synchronize(device)
            cudart.cudaProfilerStop()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
