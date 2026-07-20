#!/usr/bin/env python3
"""GPU-only correctness and graph-replay probe for standalone W4A16 kernels.

This intentionally exercises the legacy standalone GEMM and activation
classes directly.  The fused MoE path embeds GEMM objects but does not emit
the standalone ``W4A16GemmKernel`` or ``W4A16ActivationKernel`` symbols.
Point ``B12X_CUTE_COMPILE_CACHE_DIR`` at a fresh directory when using this
probe to populate a resource-audit corpus.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from typing import Callable

import torch

from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument,
    require_target_gpu,
)
from b12x.cute.intrinsics import swizzle_block_scale
from b12x.cute.utils import current_cuda_stream
from b12x.moe.fused.w4a16.host import (
    packed_gemm_scratch_elements,
    unswizzle_expert_scales,
)
from b12x.moe.fused.w4a16.kernel import (
    compile_w4a16_activation,
    compile_w4a16_gemm,
    pack_topk_routes_by_expert,
)
from b12x.moe.fused.w4a16.prepare import (
    prepare_w4a16_modelopt_nvfp4_weights,
)


_FP4_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _positive_fp8(shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    return (torch.rand(shape, device=device) * 0.25 + 0.03125).to(torch.float8_e4m3fn)


def _make_source_weights(
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    gated = activation == "silu"
    w13_rows = intermediate_size * (2 if gated else 1)
    w13 = torch.randint(
        0,
        256,
        (experts, w13_rows, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = swizzle_block_scale(
        _positive_fp8(
            (experts, w13_rows, hidden_size // 16),
            device=device,
        )
    )
    w2_scale = swizzle_block_scale(
        _positive_fp8(
            (experts, hidden_size, intermediate_size // 16),
            device=device,
        )
    )
    w13_global = (torch.rand(experts, device=device) * 0.1 + 0.05).float()
    w2_global = (torch.rand(experts, device=device) * 0.1 + 0.05).float()
    return w13, w13_scale, w13_global, w2, w2_scale, w2_global


def _dequant_source_w13(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    activation: str,
) -> torch.Tensor:
    """Return the BF16 matrix values consumed by the packed GEMM.

    ModelOpt ``w13`` source order is up/gate.  Packed W4A16 rotates it to the
    activation kernel's gate/up order, so the oracle performs the same row
    rotation after source-layout dequantization.
    """
    experts, rows, _ = w13.shape
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=w13.device)
    low = (w13 & 0x0F).long()
    high = ((w13 >> 4) & 0x0F).long()
    raw = torch.stack((lut[low], lut[high]), dim=-1).reshape(experts, rows, hidden_size)
    scales = unswizzle_expert_scales(
        w13_scale,
        rows=rows,
        cols=hidden_size,
    ).float()
    dequant = (
        raw.reshape(experts, rows, hidden_size // 16, 16) * scales.unsqueeze(-1)
    ).reshape(experts, rows, hidden_size)
    # Device dequant produces BF16 MMA operands before FP32 accumulation.
    dequant = dequant.to(torch.bfloat16).float()
    if activation == "silu":
        dequant = torch.cat(
            (
                dequant[:, intermediate_size:],
                dequant[:, :intermediate_size],
            ),
            dim=1,
        )
    return dequant


def _gemm_reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    source_w13: torch.Tensor,
    source_w13_scale: torch.Tensor,
    source_w13_global: torch.Tensor,
    *,
    intermediate_size: int,
    activation: str,
) -> torch.Tensor:
    m, hidden_size = x.shape
    topk = int(topk_ids.shape[1])
    route_ids = topk_ids.reshape(-1).long()
    route_x = x[:, None, :].expand(m, topk, hidden_size).reshape(-1, hidden_size)
    dense = _dequant_source_w13(
        source_w13,
        source_w13_scale,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    output = torch.empty(
        (m * topk, int(dense.shape[1])),
        dtype=torch.float32,
        device=x.device,
    )
    for expert in range(int(dense.shape[0])):
        selected = route_ids == expert
        if bool(selected.any().item()):
            output[selected] = (
                route_x[selected].float() @ dense[expert].T
            ) * source_w13_global[expert]
    return output.to(torch.bfloat16)


def _activation_reference(fc1: torch.Tensor, *, activation: str) -> torch.Tensor:
    if activation == "relu2":
        value = torch.clamp_min(fc1.float(), 0.0)
        return (value * value).to(torch.bfloat16)
    gate, up = fc1.chunk(2, dim=1)
    # Match W4A16ActivationKernel's BF16 silu and up intermediates.
    silu = (gate.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    return (silu * up).to(torch.bfloat16)


def _check_against_oracle(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    min_cosine: float,
) -> dict[str, float | bool]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    finite = bool(torch.isfinite(actual_f32).all().item())
    nonzero = bool((actual != 0).any().item())
    diff = actual_f32 - expected_f32
    actual_flat = actual_f32.reshape(-1)
    expected_flat = expected_f32.reshape(-1)
    denom = actual_flat.norm() * expected_flat.norm()
    cosine = float(
        ((actual_flat * expected_flat).sum() / denom.clamp_min(1.0e-24)).item()
    )
    max_abs = float(diff.abs().max().item())
    rmse = float(diff.square().mean().sqrt().item())
    reference_max_abs = float(expected_f32.abs().max().item())
    max_abs_limit = max(0.125, 0.05 * reference_max_abs)
    if not finite:
        raise AssertionError(f"{name} produced non-finite values")
    if not nonzero:
        raise AssertionError(f"{name} produced only zeros")
    if cosine < min_cosine or max_abs > max_abs_limit:
        raise AssertionError(
            f"{name} failed oracle: cosine={cosine:.8f} "
            f"(minimum {min_cosine:.8f}), max_abs={max_abs:.8f} "
            f"(limit {max_abs_limit:.8f}), rmse={rmse:.8f}"
        )
    return {
        "finite": finite,
        "nonzero": nonzero,
        "cosine": cosine,
        "max_abs": max_abs,
        "rmse": rmse,
        "max_abs_limit": max_abs_limit,
    }


def _capture_and_check(
    name: str,
    launch: Callable[[], None],
    output: torch.Tensor,
) -> tuple[torch.cuda.CUDAGraph, float]:
    launch()
    torch.cuda.synchronize()
    eager = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    replay_max_abs = float((output.float() - eager.float()).abs().max().item())
    if replay_max_abs != 0.0:
        raise AssertionError(
            f"{name} graph replay differs from eager output: {replay_max_abs}"
        )
    return graph, replay_max_abs


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000.0 / iterations)
    return {
        "runs_us": samples,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_gpu_argument(parser)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--activation", choices=("silu", "relu2"), default="silu")
    parser.add_argument("--moe-block-size", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-cosine", type=float, default=0.9975)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    require_target_gpu(args.expected_physical_gpu)
    if args.device != 0:
        raise ValueError("--device must be logical CUDA device 0")
    if min(args.m, args.hidden_size, args.intermediate_size) <= 0:
        raise ValueError("M, hidden size, and intermediate size must be positive")
    if args.hidden_size % 16 or args.intermediate_size % 16:
        raise ValueError("hidden and intermediate sizes must be divisible by 16")
    if args.topk <= 0 or args.topk > args.experts:
        raise ValueError("topk must be in [1, experts]")
    if min(args.warmup, args.iterations, args.repeats) <= 0:
        raise ValueError("warmup, iterations, and repeats must be positive")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    source_weights = _make_source_weights(
        experts=args.experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        activation=args.activation,
        device=device,
    )
    prepared = prepare_w4a16_modelopt_nvfp4_weights(
        *source_weights,
        activation=args.activation,
        params_dtype=torch.bfloat16,
    )
    x = (torch.randn(args.m, args.hidden_size, device=device) * 0.25).to(torch.bfloat16)
    token = torch.arange(args.m, device=device, dtype=torch.int32)[:, None]
    rank = torch.arange(args.topk, device=device, dtype=torch.int32)[None, :]
    topk_ids = ((token + rank * 3) % args.experts).contiguous()
    topk_weights = torch.full(
        (args.m, args.topk),
        1.0 / args.topk,
        dtype=torch.float32,
        device=device,
    )
    packed_routes, block_experts, packed_route_count = pack_topk_routes_by_expert(
        topk_ids,
        args.moe_block_size,
        args.experts,
    )
    torch.cuda.synchronize()
    active_route_slots = int(packed_route_count.item())
    active_route_blocks = active_route_slots // args.moe_block_size
    if active_route_slots <= 0 or active_route_slots % args.moe_block_size:
        raise AssertionError(f"invalid packed route count {active_route_slots}")

    w13_rows = args.intermediate_size * (2 if args.activation == "silu" else 1)
    if w13_rows % args.tile_n or args.hidden_size % args.tile_k:
        raise ValueError("standalone GEMM dimensions must be divisible by tile N/K")
    gemm = compile_w4a16_gemm(
        size_m=args.m,
        size_n=w13_rows,
        size_k=args.hidden_size,
        num_experts=args.experts,
        top_k=args.topk,
        mul_topk_weights=False,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        moe_block_size=args.moe_block_size,
        max_m_blocks=int(block_experts.numel()),
        element_dtype="bf16",
        scale_format="e4m3_k16",
    )
    activation = compile_w4a16_activation(
        rows=args.m * args.topk,
        intermediate_size=args.intermediate_size,
        activation=args.activation,
        element_dtype="bf16",
        fast_math=True,
    )

    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    n_tiles = w13_rows // args.tile_n
    grid_cap = sms * int(gemm.blocks_per_sm)
    grid_x = max(min(grid_cap, active_route_blocks * n_tiles), 1)
    fc1 = torch.empty(
        (args.m * args.topk, w13_rows),
        dtype=torch.bfloat16,
        device=device,
    )
    activated = torch.empty(
        (args.m * args.topk, args.intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    c_tmp = torch.empty(
        packed_gemm_scratch_elements(
            size_n=w13_rows,
            route_slots=int(packed_routes.numel()),
            moe_block_size=args.moe_block_size,
            sms=sms,
        ),
        dtype=torch.float32,
        device=device,
    )
    locks = torch.zeros(sms * 4, dtype=torch.int32, device=device)

    def launch_gemm() -> None:
        gemm.compiled(
            x.reshape(-1),
            prepared.w13.reshape(-1),
            fc1.reshape(-1),
            prepared.w13_scale.view(torch.uint8).view(torch.int32).reshape(-1),
            prepared.w13_global_scale.reshape(-1),
            packed_routes.reshape(-1),
            block_experts.reshape(-1),
            packed_route_count.reshape(-1),
            topk_weights.reshape(-1),
            c_tmp.reshape(-1),
            locks.reshape(-1),
            args.m,
            grid_x,
            current_cuda_stream(),
        )

    def launch_activation() -> None:
        activation.compiled(
            fc1.reshape(-1),
            activated.reshape(-1),
            args.m * args.topk,
            current_cuda_stream(),
        )

    launch_gemm()
    torch.cuda.synchronize()
    gemm_expected = _gemm_reference(
        x,
        topk_ids,
        source_weights[0],
        source_weights[1],
        source_weights[2],
        intermediate_size=args.intermediate_size,
        activation=args.activation,
    )
    gemm_correctness = _check_against_oracle(
        "W4A16GemmKernel",
        fc1,
        gemm_expected,
        min_cosine=args.min_cosine,
    )
    launch_activation()
    torch.cuda.synchronize()
    activation_expected = _activation_reference(fc1, activation=args.activation)
    activation_correctness = _check_against_oracle(
        "W4A16ActivationKernel",
        activated,
        activation_expected,
        min_cosine=args.min_cosine,
    )

    gemm_graph, gemm_graph_max_abs = _capture_and_check(
        "W4A16GemmKernel", launch_gemm, fc1
    )
    activation_graph, activation_graph_max_abs = _capture_and_check(
        "W4A16ActivationKernel", launch_activation, activated
    )
    gemm_timing = _time_graph(
        gemm_graph,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    activation_timing = _time_graph(
        activation_graph,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )

    print(
        json.dumps(
            {
                "activation": args.activation,
                "activation_correctness": activation_correctness,
                "activation_graph_max_abs": activation_graph_max_abs,
                "activation_graph_replay": activation_timing,
                "active_route_blocks": active_route_blocks,
                "active_route_slots": active_route_slots,
                "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
                "device": props.name,
                "experts": args.experts,
                "gemm_blocks_per_sm": int(gemm.blocks_per_sm),
                "gemm_correctness": gemm_correctness,
                "gemm_graph_max_abs": gemm_graph_max_abs,
                "gemm_graph_replay": gemm_timing,
                "grid_x": grid_x,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "label": args.label,
                "m": args.m,
                "moe_block_size": args.moe_block_size,
                "route_capacity_slots": int(packed_routes.numel()),
                "topk": args.topk,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
