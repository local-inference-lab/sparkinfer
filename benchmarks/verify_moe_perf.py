#!/usr/bin/env python3
"""Measure fixed-size MoE kernel latency for migration perf gating."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.benchmark_moe import (
    MODEL_PATH,
    ModelSpec,
    bench_repeated,
    get_scale_contract_params,
    load_expert_weights,
    make_l2_flush_fn,
    prepare_b12x_benchmark_weights,
    require_sm120,
    resolve_l2_flush_bytes,
)
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    build_tp_moe_fp4_binding,
    clear_tp_moe_caches,
)


DEFAULT_BATCH_SIZES = [1, 4, 32, 80]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--scale-contract", choices=["shared", "per-expert"], default="shared")
    parser.add_argument("--activation", choices=["silu", "relu2"], default="silu")
    parser.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--flush-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evict GPU L2 before each warmup and timed launch (default: enabled).",
    )
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="Bytes to touch when evicting L2; 0 uses 2x the reported L2 size.",
    )
    return parser


def _measure_batch(
    *,
    batch_size: int,
    spec: ModelSpec,
    experts: B12XFP4ExpertWeights,
    warmup: int,
    iters: int,
    repeats: int,
    fast_math: bool,
    device: torch.device,
    l2_flush,
) -> dict[str, float]:
    torch.manual_seed(42 + batch_size)
    x = torch.randn(batch_size, spec.hidden_size, dtype=torch.bfloat16, device=device)
    routing_logits = torch.randn(batch_size, spec.num_experts, dtype=torch.float32, device=device)
    topk_logits, topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    output = torch.empty_like(x)
    workspace = allocate_tp_moe_workspace_pool()
    binding = build_tp_moe_fp4_binding(
        scratch=workspace,
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        fast_math=fast_math,
        output=output,
        input_scales_static=True,
        quant_mode="nvfp4",
    )

    def launch() -> torch.Tensor:
        return b12x_moe_fp4(binding=binding)

    eager_stats = bench_repeated(
        launch,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        l2_flush=l2_flush,
    )

    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    def replay() -> None:
        graph.replay()

    graph_stats = bench_repeated(
        replay,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        l2_flush=l2_flush,
    )

    return {
        "eager_median_us": eager_stats.median_us,
        "eager_min_us": eager_stats.min_us,
        "graph_median_us": graph_stats.median_us,
        "graph_min_us": graph_stats.min_us,
    }


def main() -> None:
    args = _build_parser().parse_args()
    require_sm120()
    torch.empty(1, device="cuda")
    device = torch.device("cuda")
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0

    spec = ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=4,
        tp_rank=0,
    )
    clear_tp_moe_caches()
    weights = load_expert_weights(
        MODEL_PATH,
        spec,
        layer_idx=args.layer_idx,
        activation=args.activation,
    )
    params = get_scale_contract_params(weights, args.scale_contract)
    experts, _ = prepare_b12x_benchmark_weights(
        weights,
        params,
        quant_mode="nvfp4",
        activation=args.activation,
    )

    results = {
        "activation": args.activation,
        "batch_sizes": {},
        "fast_math": args.fast_math,
        "iters": args.iters,
        "layer_idx": args.layer_idx,
        "l2_flush_bytes": l2_flush_bytes,
        "l2_flush_enabled": args.flush_l2,
        "model_path": str(MODEL_PATH),
        "repeats": args.repeats,
        "scale_contract": args.scale_contract,
        "warmup": args.warmup,
    }
    for batch_size in args.batch_sizes:
        results["batch_sizes"][str(batch_size)] = _measure_batch(
            batch_size=batch_size,
            spec=spec,
            experts=experts,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            fast_math=args.fast_math,
            device=device,
            l2_flush=l2_flush,
        )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
