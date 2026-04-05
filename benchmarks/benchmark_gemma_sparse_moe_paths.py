#!/usr/bin/env python3
"""Benchmark static Qwen3.5 sparse-MoE paths with and without full pre-MLP fusion.

Run under torchrun, for example:

  torchrun --nproc-per-node=2 benchmarks/benchmark_gemma_sparse_moe_paths.py \
      --batch-sizes 1 2 4 8

Path 1: fully fused static path
  allreduce + residual + GemmaRMSNorm + routing + prequantized compact MoE

Path 2: semi-fused baseline
  normal allreduce + GemmaRMSNorm + BF16 sparse MoE wrapper
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

from benchmarks.benchmark_moe import (
    BATCH_SIZE_PROFILES,
    MODEL_PATH,
    ModelSpec,
    bench_events,
    load_expert_weights,
    load_gate_weight,
    make_input_activations,
    require_sm120,
)
from benchmarks.checkpoint_loader import IndexedSafetensorLoader
from b12x.distributed.pcie_oneshot import PCIeOneshotAllReduce
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    _b12x_gemma_sparse_moe_fp4_static,
    allocate_tp_moe_workspace_pool,
    b12x_sparse_moe_fp4,
    clear_tp_moe_caches,
)
from b12x.moe.fused.reference import compare_to_reference


def _rank0_print(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


def _barrier() -> None:
    dist.barrier(device_ids=[_local_rank()])


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


def _pack_experts(weights) -> B12XFP4ExpertWeights:
    return B12XFP4ExpertWeights(
        a1_gscale=weights.w13_input_scale_per_expert,
        w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled,
        w1_alphas=weights.g1_alphas_per_expert,
        a2_gscale=weights.w2_input_scale_per_expert,
        w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled,
        w2_alphas=weights.g2_alphas_per_expert,
    )


def _load_post_attention_layernorm_weight(
    model_path: pathlib.Path,
    *,
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    weight = IndexedSafetensorLoader(model_path).get_tensor(
        f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight"
    )
    return weight.to(device=device, dtype=torch.bfloat16).contiguous()


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


def _fmt_us(times_ms: list[float]) -> str:
    median_us = statistics.median(times_ms) * 1000.0
    min_us = min(times_ms) * 1000.0
    return f"{median_us:8.1f} us (min {min_us:.1f})"


def _gather_rank_medians(times_ms: list[float]) -> list[float]:
    local = statistics.median(times_ms)
    gathered: list[float] = [0.0 for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered


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
        help="Skip output validation between the fused and semi-fused paths.",
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
        "Gemma sparse MoE path benchmark | "
        f"world_size={dist.get_world_size()} K={spec.hidden_size} I_tp={spec.I_tp} "
        f"E={spec.num_experts} top_k={spec.top_k}"
    )

    runtime = PCIeOneshotAllReduce.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=max_input_bytes,
    )
    try:
        with torch.no_grad():
            weights = load_expert_weights(MODEL_PATH, spec, layer_idx=args.layer_idx)
            gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=args.layer_idx)
            norm_weight = _load_post_attention_layernorm_weight(
                MODEL_PATH,
                layer_idx=args.layer_idx,
                device=device,
            )
            experts = _pack_experts(weights)

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
                semi_pool = allocate_tp_moe_workspace_pool()
                fused_output = torch.empty_like(hidden_local)
                fused_residual_out = torch.empty_like(hidden_local)
                semi_output = torch.empty_like(hidden_local)

                def fused_path() -> None:
                    _b12x_gemma_sparse_moe_fp4_static(
                        hidden_local,
                        residual,
                        pre_mlp_runtime=runtime,
                        norm_weight=norm_weight,
                        norm_eps=args.norm_eps,
                        experts=experts,
                        workspace=fused_pool,
                        top_k=spec.top_k,
                        gate_weight=gate_weight,
                        output=fused_output,
                        residual_out=fused_residual_out,
                        input_scales_static=True,
                        # Compare fusion boundaries under the same FC1 contract.
                        fc1_tile_amax=False,
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
                    out = b12x_sparse_moe_fp4(
                        normed_hidden_states,
                        experts=experts,
                        workspace=semi_pool,
                        top_k=spec.top_k,
                        gate_weight=gate_weight,
                        output=semi_output,
                        input_scales_static=True,
                    )
                    return out, semi_residual_out

                fused_path()
                semi_out, semi_residual_out = semi_fused_path()
                torch.cuda.synchronize(device)

                if not args.skip_validate:
                    output_metrics = compare_to_reference(fused_output, semi_out)
                    residual_metrics = compare_to_reference(fused_residual_out, semi_residual_out)
                    if output_metrics.max_abs > 1e-4 or output_metrics.cos <= 0.999:
                        raise AssertionError(
                            f"output mismatch at m={m}: max_abs={output_metrics.max_abs:.6f} cos={output_metrics.cos:.6f}"
                        )
                    if residual_metrics.max_abs > 5e-2 or residual_metrics.cos <= 0.9999:
                        raise AssertionError(
                            f"residual mismatch at m={m}: max_abs={residual_metrics.max_abs:.6f} cos={residual_metrics.cos:.6f}"
                        )
                    if dist.get_rank() == 0:
                        print(
                            f"\nm={m} validate: "
                            f"out(max_abs={output_metrics.max_abs:.3e}, cos={output_metrics.cos:.6f}) "
                            f"residual(max_abs={residual_metrics.max_abs:.3e}, cos={residual_metrics.cos:.6f})",
                            flush=True,
                        )

                _barrier()
                fused_times = bench_events(fused_path, warmup=args.warmup, iters=args.iters)
                _barrier()
                semi_times = bench_events(lambda: semi_fused_path()[0], warmup=args.warmup, iters=args.iters)
                _barrier()

                fused_rank_medians = _gather_rank_medians(fused_times)
                semi_rank_medians = _gather_rank_medians(semi_times)
                if dist.get_rank() == 0:
                    fused_median_us = max(fused_rank_medians) * 1000.0
                    semi_median_us = max(semi_rank_medians) * 1000.0
                    print(f"\nm={m}  (tokens*top_k = {m * spec.top_k})", flush=True)
                    print(f"  fully fused : {_fmt_us(fused_times)} | max-rank median {fused_median_us:.1f} us", flush=True)
                    print(f"  semi-fused  : {_fmt_us(semi_times)} | max-rank median {semi_median_us:.1f} us", flush=True)
                    print(
                        f"  delta       : {fused_median_us - semi_median_us:8.1f} us | ratio "
                        f"{(fused_median_us / semi_median_us) if semi_median_us else float('inf'):.3f}x",
                        flush=True,
                    )
    finally:
        runtime.close()
        _barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
