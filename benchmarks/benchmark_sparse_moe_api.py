#!/usr/bin/env python3
"""Benchmark the sparse-block sparkinfer API with Qwen-style hidden-state inputs.

This benchmark is graph-first: timings are CUDA graph replay times, not eager
Python dispatch.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from benchmarks.benchmark_moe import (
    BATCH_SIZE_PROFILES,
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    ModelSpec,
    bench_events,
    load_expert_weights,
    load_gate_weight,
    make_input_activations,
    require_sm120,
)
from sparkinfer.moe.fused_moe._impl import (
    SPARKINFERFP4ExpertWeights,
    allocate_tp_moe_workspace_pool,
    build_tp_moe_fp4_binding,
    build_tp_moe_route_binding,
    build_tp_moe_sparse_fp4_binding,
    clear_tp_moe_caches,
    plan_sparkinfer_fp4_moe_weights,
    prepare_sparkinfer_fp4_moe_weights,
)


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


def _pack_experts(weights) -> SPARKINFERFP4ExpertWeights:
    plan = plan_sparkinfer_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format=weights.source_format,
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=weights.spec.num_experts,
        hidden_size=weights.spec.hidden_size,
        intermediate_size=weights.spec.I_tp,
        w13_layout=weights.w13_layout,
    )
    prepared = prepare_sparkinfer_fp4_moe_weights(
        plan=plan,
        w1_global_scale=(
            weights.g1_alphas_per_expert
            * weights.w13_input_scale_quant_per_expert
        ),
        w2_global_scale=(
            weights.g2_alphas_per_expert
            * weights.w2_input_scale_quant_per_expert
        ),
        w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled,
        w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled,
        a1_gscale=weights.w13_input_scale_quant_per_expert,
        a2_gscale=weights.w2_input_scale_quant_per_expert,
        params_dtype=torch.bfloat16,
    )
    return prepared


def _manual_route(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_logits = F.linear(hidden_states, gate_weight)
    topk_logits, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits.to(torch.float32), dim=-1)
    return router_logits, topk_ids, topk_weights


def _selected_logits(router_logits: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(router_logits, 1, topk_ids.to(torch.int64))


def _fmt_us(times_ms: list[float]) -> str:
    median_us = statistics.median(times_ms) * 1000.0
    min_us = min(times_ms) * 1000.0
    return f"{median_us:8.1f} us (min {min_us:.1f})"


def _capture_graph(fn) -> torch.cuda.CUDAGraph:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size-profile", choices=sorted(BATCH_SIZE_PROFILES), default="sglang-single-request")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip output equality checks between manual routing and the sparse wrapper.",
    )
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    return parser.parse_args()


def _pick_batch_sizes(args: argparse.Namespace) -> list[int]:
    if args.batch_sizes:
        return list(args.batch_sizes)
    return list(BATCH_SIZE_PROFILES[args.batch_size_profile])


def main() -> None:
    args = _parse_args()
    require_sm120()
    torch.set_grad_enabled(False)
    clear_tp_moe_caches()

    device = torch.device("cuda")
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    spec = _make_spec()
    print(
        "Sparse API benchmark (graph replay) | "
        f"Qwen3.5 TP={spec.tp_size} K={spec.hidden_size} I_tp={spec.I_tp} "
        f"E={spec.num_experts} top_k={spec.top_k}"
    )
    flush_desc = f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)" if l2_flush else "off"
    print(f"L2 flush: {flush_desc}")

    with torch.no_grad():
        weights = load_expert_weights(MODEL_PATH, spec, layer_idx=args.layer_idx)
        gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=args.layer_idx)
        experts = _pack_experts(weights)

        for m in _pick_batch_sizes(args):
            hidden_states = make_input_activations(spec, m, seed=args.seed + m, device=device)
            _, topk_ids, topk_weights = _manual_route(hidden_states, gate_weight, spec.top_k)
            workspace = allocate_tp_moe_workspace_pool()

            routed_output = torch.empty_like(hidden_states)
            manual_output = torch.empty_like(hidden_states)
            sparse_output = torch.empty_like(hidden_states)
            manual_router_logits = torch.empty(
                m,
                spec.num_experts,
                dtype=torch.result_type(hidden_states, gate_weight),
                device=device,
            )
            manual_topk_logits = torch.empty(
                m,
                spec.top_k,
                dtype=manual_router_logits.dtype,
                device=device,
            )
            manual_topk_logits_f32 = torch.empty(
                m,
                spec.top_k,
                dtype=torch.float32,
                device=device,
            )
            manual_topk_ids = torch.empty(m, spec.top_k, dtype=torch.int64, device=device)
            manual_topk_weights = torch.empty(m, spec.top_k, dtype=torch.float32, device=device)
            route_binding = build_tp_moe_route_binding(
                scratch=workspace,
                hidden_states=hidden_states,
                top_k=spec.top_k,
                gate_weight=gate_weight,
            )
            routed_binding = build_tp_moe_fp4_binding(
                scratch=workspace,
                a=hidden_states,
                experts=experts,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=routed_output,
                input_scales_static=True,
            )
            manual_binding = build_tp_moe_fp4_binding(
                scratch=workspace,
                a=hidden_states,
                experts=experts,
                topk_weights=manual_topk_weights,
                topk_ids=manual_topk_ids,
                output=manual_output,
                input_scales_static=True,
            )
            sparse_binding = build_tp_moe_sparse_fp4_binding(
                scratch=workspace,
                hidden_states=hidden_states,
                experts=experts,
                top_k=spec.top_k,
                gate_weight=gate_weight,
                output=sparse_output,
                input_scales_static=True,
            )

            def manual_route_only() -> None:
                torch.mm(hidden_states, gate_weight.t(), out=manual_router_logits)
                torch.topk(
                    manual_router_logits,
                    k=spec.top_k,
                    dim=-1,
                    out=(manual_topk_logits, manual_topk_ids),
                )
                manual_topk_logits_f32.copy_(manual_topk_logits)
                torch.softmax(manual_topk_logits_f32, dim=-1, out=manual_topk_weights)

            def route_api_only() -> None:
                route_binding.run()

            def tp_only() -> torch.Tensor:
                return routed_binding.run()

            def manual_e2e() -> torch.Tensor:
                manual_route_only()
                return manual_binding.run()

            def sparse_api() -> None:
                sparse_binding.run()

            manual_route_only()
            route_api = route_binding.run()
            manual_e2e()
            sparse_api()
            torch.cuda.synchronize()

            if not args.skip_validate:
                manual_router_logits_ref, _, _ = _manual_route(hidden_states, gate_weight, spec.top_k)
                torch.testing.assert_close(route_api.router_logits, manual_router_logits_ref)
                torch.testing.assert_close(manual_router_logits, manual_router_logits_ref)
                torch.testing.assert_close(
                    _selected_logits(manual_router_logits_ref, route_api.topk_ids),
                    _selected_logits(manual_router_logits_ref, topk_ids),
                )
                torch.testing.assert_close(route_api.topk_weights, topk_weights)
                assert route_api.flat_ids is not None
                assert route_api.flat_weights is not None
                torch.testing.assert_close(route_api.flat_ids, route_api.topk_ids.view(-1))
                torch.testing.assert_close(route_api.flat_weights, route_api.topk_weights.view(-1))
                torch.testing.assert_close(manual_topk_ids, topk_ids)
                torch.testing.assert_close(manual_topk_weights, topk_weights)
                torch.testing.assert_close(sparse_output, manual_output, atol=5e-4, rtol=1e-2)

            manual_route_graph = _capture_graph(manual_route_only)
            route_api_graph = _capture_graph(route_api_only)
            tp_graph = _capture_graph(tp_only)
            manual_e2e_graph = _capture_graph(manual_e2e)
            sparse_graph = _capture_graph(sparse_api)

            routing_times = bench_events(
                manual_route_graph.replay,
                warmup=args.warmup,
                iters=args.iters,
                l2_flush=l2_flush,
            )
            route_api_times = bench_events(
                route_api_graph.replay,
                warmup=args.warmup,
                iters=args.iters,
                l2_flush=l2_flush,
            )
            tp_times = bench_events(
                tp_graph.replay,
                warmup=args.warmup,
                iters=args.iters,
                l2_flush=l2_flush,
            )
            manual_times = bench_events(
                manual_e2e_graph.replay,
                warmup=args.warmup,
                iters=args.iters,
                l2_flush=l2_flush,
            )
            sparse_times = bench_events(
                sparse_graph.replay,
                warmup=args.warmup,
                iters=args.iters,
                l2_flush=l2_flush,
            )

            route_manual_us = statistics.median(routing_times) * 1000.0
            route_api_us = statistics.median(route_api_times) * 1000.0
            route_delta_us = route_api_us - route_manual_us
            route_ratio = route_api_us / route_manual_us if route_manual_us else float("inf")
            manual_us = statistics.median(manual_times) * 1000.0
            sparse_us = statistics.median(sparse_times) * 1000.0
            delta_us = sparse_us - manual_us
            ratio = sparse_us / manual_us if manual_us else float("inf")

            print(f"\nm={m}  (tokens*top_k = {m * spec.top_k})")
            print(f"  route manual graph : {_fmt_us(routing_times)}")
            print(f"  route fast graph   : {_fmt_us(route_api_times)}")
            print(f"  route delta  : {route_delta_us:8.1f} us | ratio {route_ratio:.3f}x")
            print(f"  tp routed graph    : {_fmt_us(tp_times)}")
            print(f"  manual e2e graph   : {_fmt_us(manual_times)}")
            print(f"  sparse api graph   : {_fmt_us(sparse_times)}")
            print(f"  wrapper delta: {delta_us:8.1f} us | ratio {ratio:.3f}x")


if __name__ == "__main__":
    main()
