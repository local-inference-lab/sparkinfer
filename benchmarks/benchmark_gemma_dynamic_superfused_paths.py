#!/usr/bin/env python3
"""Benchmark dynamic superfused Gemma/Qwen block paths against graph-safe baselines.

Run under torchrun, for example:

  torchrun --nproc-per-node=2 benchmarks/benchmark_gemma_dynamic_superfused_paths.py \
      --batch-sizes 1 2 4 8
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from benchmarks.benchmark_gemma_moe_block_paths import (
    _gemma_rmsnorm_after_allreduce,
    _load_post_attention_layernorm_weight,
    _pack_shared_expert,
    _pack_sparse_experts_per_expert,
)
from benchmarks.benchmark_moe import (
    BATCH_SIZE_PROFILES,
    MODEL_PATH,
    ModelSpec,
    bench_events,
    load_expert_weights,
    load_gate_weight,
    load_shared_expert_weights,
    load_shared_gate_weight,
    make_input_activations,
    require_sm120,
)
from b12x.distributed.pcie_oneshot import PCIeOneshotAllReduce
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    _append_expert_bank,
    _append_shared_expert_routing,
    _b12x_gemma_moe_block_fp4_dynamic_superfused,
    _effective_input_scales,
    _launch_prequantized_moe_consumer,
    _prepare_gemma_moe_block_fp4_static_monolithic,
    _shared_expert_gate_weights,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    b12x_route_experts_fast,
    clear_tp_moe_caches,
)
from b12x.moe.fused.reference import compare_to_reference


def _rank0_print(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


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


def _fmt_us(times_ms: list[float]) -> str:
    median_us = statistics.median(times_ms) * 1000.0
    min_us = min(times_ms) * 1000.0
    return f"{median_us:8.1f} us (min {min_us:.1f})"


def _gather_rank_medians(times_ms: list[float]) -> list[float]:
    local = statistics.median(times_ms)
    gathered: list[float] = [0.0 for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered


def _bench_graph_replay(
    fn,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> list[float]:
    # Warm eager launch state so compile/cache work does not leak into capture
    # or replay timing.
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    return bench_events(replay, warmup=warmup, iters=iters)


def _pick_batch_sizes(args: argparse.Namespace) -> list[int]:
    if args.batch_sizes:
        return list(args.batch_sizes)
    return list(BATCH_SIZE_PROFILES[args.batch_size_profile])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size-profile", choices=sorted(BATCH_SIZE_PROFILES), default="micro")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip output validation between the dynamic superfused and semi-fused paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", _local_rank()))
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    require_sm120()
    torch.set_grad_enabled(False)
    clear_tp_moe_caches()

    spec = _make_spec()
    batch_sizes = _pick_batch_sizes(args)
    max_tokens = max(batch_sizes)
    max_input_bytes = max_tokens * spec.hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()

    _rank0_print(
        "Gemma dynamic superfused benchmark | "
        f"world_size={dist.get_world_size()} K={spec.hidden_size} I_tp={spec.I_tp} "
        f"E={spec.num_experts}+1 top_k={spec.top_k}+1 | timings use CUDA graph replay"
    )

    runtime = PCIeOneshotAllReduce.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=max_input_bytes,
    )
    try:
        with torch.no_grad():
            sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=args.layer_idx)
            shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=args.layer_idx)
            gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=args.layer_idx)
            shared_gate_weight = load_shared_gate_weight(MODEL_PATH, layer_idx=args.layer_idx)
            norm_weight = _load_post_attention_layernorm_weight(
                MODEL_PATH,
                layer_idx=args.layer_idx,
                device=device,
            )
            sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
            shared_expert = _pack_shared_expert(shared_weights)
            combined_experts = _append_expert_bank(sparse_experts, shared_expert)
            split_graph_experts = B12XFP4ExpertWeights(
                a1_gscale=_effective_input_scales(
                    combined_experts.a1_gscale,
                    combined_experts.w1_fp4.shape[0],
                    input_scales_are_reciprocal=True,
                ),
                w1_fp4=combined_experts.w1_fp4,
                w1_blockscale=combined_experts.w1_blockscale,
                w1_alphas=combined_experts.w1_alphas,
                a2_gscale=_effective_input_scales(
                    combined_experts.a2_gscale,
                    combined_experts.w2_fp4.shape[0],
                    input_scales_are_reciprocal=True,
                ),
                w2_fp4=combined_experts.w2_fp4,
                w2_blockscale=combined_experts.w2_blockscale,
                w2_alphas=combined_experts.w2_alphas,
            )
            shared_expert_id = sparse_experts.w1_fp4.shape[0]

            for m in batch_sizes:
                hidden_local = make_input_activations(
                    spec,
                    m,
                    seed=args.seed + 17 * _rank() + m,
                    device=device,
                )
                residual = make_input_activations(
                    spec,
                    m,
                    seed=args.seed + 10_000 + m,
                    device=device,
                )

                fused_pool = allocate_tp_moe_workspace_pool()
                split_pool = allocate_tp_moe_workspace_pool()
                semi_pool = allocate_tp_moe_workspace_pool()
                fused_output = torch.empty_like(hidden_local)
                fused_residual_out = torch.empty_like(hidden_local)
                split_output = torch.empty_like(hidden_local)
                split_residual_out = torch.empty_like(hidden_local)
                semi_output = torch.empty_like(hidden_local)

                def superfused_path() -> None:
                    _b12x_gemma_moe_block_fp4_dynamic_superfused(
                        hidden_local,
                        residual,
                        pre_mlp_runtime=runtime,
                        norm_weight=norm_weight,
                        norm_eps=args.norm_eps,
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

                def split_prequantized_path() -> None:
                    producer, resolved_workspace, prepared_experts, combined_routing = (
                        _prepare_gemma_moe_block_fp4_static_monolithic(
                            hidden_local,
                            residual,
                            pre_mlp_runtime=runtime,
                            norm_weight=norm_weight,
                            norm_eps=args.norm_eps,
                            sparse_experts=sparse_experts,
                            shared_expert=shared_expert,
                            shared_gate_weight=shared_gate_weight,
                            combined_experts=split_graph_experts,
                            workspace=split_pool,
                            top_k=spec.top_k,
                            gate_weight=gate_weight,
                            residual_out=split_residual_out,
                            input_scales_are_reciprocal=False,
                            input_scales_static=True,
                            fc1_tile_amax=False,
                        )
                    )
                    _launch_prequantized_moe_consumer(
                        producer.normalized,
                        experts=prepared_experts,
                        workspace=resolved_workspace,
                        routing=combined_routing,
                        output=split_output,
                        input_scales_are_reciprocal=False,
                        fast_math=None,
                        fc1_tile_amax=False,
                        fc2_tile_amax=False,
                    )

                def semi_fused_path() -> tuple[torch.Tensor, torch.Tensor]:
                    reduced = hidden_local.clone()
                    dist.all_reduce(reduced)
                    normed_hidden_states, semi_residual_out = _gemma_rmsnorm_after_allreduce(
                        reduced,
                        residual,
                        norm_weight,
                        args.norm_eps,
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
                        shared_expert_id=shared_expert_id,
                    )
                    out = b12x_moe_fp4(
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
                        output=semi_output,
                        input_scales_are_reciprocal=True,
                        input_scales_static=True,
                        fc2_tile_amax=False,
                    )
                    return out, semi_residual_out

                superfused_path()
                split_prequantized_path()
                semi_out, semi_residual_out = semi_fused_path()
                torch.cuda.synchronize(device)

                if not args.skip_validate:
                    output_metrics = compare_to_reference(fused_output, semi_out)
                    residual_metrics = compare_to_reference(fused_residual_out, semi_residual_out)
                    split_output_metrics = compare_to_reference(split_output, semi_out)
                    split_residual_metrics = compare_to_reference(split_residual_out, semi_residual_out)
                    if output_metrics.max_abs > 2e-2 or output_metrics.cos <= 0.999:
                        raise RuntimeError(
                            f"m={m} output mismatch on rank {_rank()}: "
                            f"max_abs={output_metrics.max_abs:.3e} cos={output_metrics.cos:.6f}"
                        )
                    if residual_metrics.max_abs > 0.0 or residual_metrics.cos <= 0.999999:
                        raise RuntimeError(
                            f"m={m} residual mismatch on rank {_rank()}: "
                            f"max_abs={residual_metrics.max_abs:.3e} cos={residual_metrics.cos:.6f}"
                        )
                    if split_output_metrics.max_abs > 2e-2 or split_output_metrics.cos <= 0.999:
                        raise RuntimeError(
                            f"m={m} split output mismatch on rank {_rank()}: "
                            f"max_abs={split_output_metrics.max_abs:.3e} cos={split_output_metrics.cos:.6f}"
                        )
                    if split_residual_metrics.max_abs > 0.0 or split_residual_metrics.cos <= 0.999999:
                        raise RuntimeError(
                            f"m={m} split residual mismatch on rank {_rank()}: "
                            f"max_abs={split_residual_metrics.max_abs:.3e} cos={split_residual_metrics.cos:.6f}"
                        )
                    _rank0_print(
                        f"m={m} validate: out(max_abs={output_metrics.max_abs:.3e}, cos={output_metrics.cos:.6f}) "
                        f"residual(max_abs={residual_metrics.max_abs:.3e}, cos={residual_metrics.cos:.6f}) "
                        f"split_out(max_abs={split_output_metrics.max_abs:.3e}, cos={split_output_metrics.cos:.6f})"
                    )

                superfused_times = _bench_graph_replay(
                    superfused_path,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                split_times = _bench_graph_replay(
                    split_prequantized_path,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                semi_times = _bench_graph_replay(
                    lambda: semi_fused_path()[0],
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                superfused_rank_medians = _gather_rank_medians(superfused_times)
                split_rank_medians = _gather_rank_medians(split_times)
                semi_rank_medians = _gather_rank_medians(semi_times)
                if _rank() == 0:
                    superfused_med_us = max(superfused_rank_medians) * 1000.0
                    split_med_us = max(split_rank_medians) * 1000.0
                    semi_med_us = max(semi_rank_medians) * 1000.0
                    superfused_ratio = superfused_med_us / semi_med_us
                    split_ratio = split_med_us / semi_med_us
                    print(
                        f"m={m:5d} | dynamic_superfused { _fmt_us(superfused_times) } | "
                        f"split_prequantized { _fmt_us(split_times) } | "
                        f"semi_fused { _fmt_us(semi_times) } | "
                        f"ratios dyn {superfused_ratio:.3f}x split {split_ratio:.3f}x",
                        flush=True,
                    )
    finally:
        runtime.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
