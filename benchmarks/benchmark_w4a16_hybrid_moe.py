#!/usr/bin/env python3
"""Benchmark the hybrid two-tier W4A16 fused MoE decode against the serial
two-launch baseline at the GLM-5.2 TP4 shard geometry (64 NVFP4 + 192 NF3
experts, hidden 6144, intermediate 512, top-k 8).

Graph-replayed decode timing per CLAUDE.md norms: correctness gates run
before any timing, every variant is captured into a CUDA graph, and raw
replay timings plus ratios are reported.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from benchmarks.common import require_sm120
from b12x.cute.fp4 import swizzle_block_scale
from b12x.cute.utils import make_ptr
from b12x.moe.fused.w4a16.host import make_w4a16_packed_buffers
from b12x.moe.fused.w4a16.kernel import (
    _DEFAULT_MAX_SHARED_MEM,
    _cutlass_element_dtype,
    _w4a16_fused_persistent_grid_x,
    build_w4a16_tier_local_map,
    compile_w4a16_fused_moe,
    compile_w4a16_fused_moe_hybrid,
    run_w4a16_moe,
)
from b12x.moe.fused.w4a16.prepare import (
    prepare_nf3_moe_weights,
    prepare_w4a16_modelopt_nvfp4_weights,
)

_HIDDEN = 6144
_INTERMEDIATE = 512
_TOPK = 8
_T0_EXPERTS = 64
_T1_EXPERTS = 192
_MAP_SLOTS = _T0_EXPERTS + _T1_EXPERTS
_TILE_CONFIG = (64, 256, 64, 256)
_DTYPE = torch.bfloat16


def _git_describe() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[1],
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[1],
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _device_limits() -> tuple[int, int]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(props.multi_processor_count), int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )


def _build_tier0() -> object:
    w13_rows = 2 * _INTERMEDIATE
    device = torch.device("cuda")
    w13 = torch.randint(
        0, 256, (_T0_EXPERTS, w13_rows, _HIDDEN // 2), dtype=torch.uint8, device=device
    )
    w2 = torch.randint(
        0,
        256,
        (_T0_EXPERTS, _HIDDEN, _INTERMEDIATE // 2),
        dtype=torch.uint8,
        device=device,
    )

    def _fp8(shape):
        return (0.05 + 0.2 * torch.rand(shape, device=device)).to(torch.float8_e4m3fn)

    prepared = prepare_w4a16_modelopt_nvfp4_weights(
        w13,
        swizzle_block_scale(_fp8((_T0_EXPERTS, w13_rows, _HIDDEN // 16))),
        (torch.rand(_T0_EXPERTS, device=device) * 0.1 + 0.05).float(),
        w2,
        swizzle_block_scale(_fp8((_T0_EXPERTS, _HIDDEN, _INTERMEDIATE // 16))),
        (torch.rand(_T0_EXPERTS, device=device) * 0.1 + 0.05).float(),
        activation="silu",
        params_dtype=_DTYPE,
    )
    del w13, w2
    torch.cuda.empty_cache()
    return prepared


def _build_tier1() -> object:
    w13_rows = 2 * _INTERMEDIATE
    device = torch.device("cuda")
    w13_codes = torch.randint(
        0, 8, (_T1_EXPERTS, w13_rows, _HIDDEN), dtype=torch.int32, device=device
    )
    w2_codes = torch.randint(
        0, 8, (_T1_EXPERTS, _HIDDEN, _INTERMEDIATE), dtype=torch.int32, device=device
    )

    def _e4m3_scale(shape):
        t_s = (0.01 + 0.24 * torch.rand(shape, device=device)).clamp(min=2.0**-7)
        e = torch.floor(torch.log2(t_s))
        step = torch.pow(2.0, e - 3)
        return torch.round(t_s / step) * step

    prepared = prepare_nf3_moe_weights(
        w13_codes,
        _e4m3_scale((_T1_EXPERTS, w13_rows, _HIDDEN // 32)),
        w2_codes,
        _e4m3_scale((_T1_EXPERTS, _HIDDEN, _INTERMEDIATE // 32)),
        activation="silu",
        fc1_tile_n=_TILE_CONFIG[1],
        fc2_tile_n=_TILE_CONFIG[3],
        params_dtype=_DTYPE,
    )
    del w13_codes, w2_codes
    torch.cuda.empty_cache()
    return prepared


def _routing(m: int, kind: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for _ in range(m):
        if kind == "mixed":
            rows.append(torch.randperm(_MAP_SLOTS, generator=generator)[:_TOPK])
        elif kind == "tier1-heavy":
            perm = _T0_EXPERTS + torch.randperm(_T1_EXPERTS, generator=generator)
            rows.append(perm[:_TOPK])
        else:
            raise ValueError(kind)
    return torch.stack(rows).to(dtype=torch.int32, device="cuda")


def _tier_local_ids(global_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t0 = torch.where(global_ids < _T0_EXPERTS, global_ids, global_ids.new_tensor(-1))
    t1 = torch.where(
        global_ids >= _T0_EXPERTS, global_ids - _T0_EXPERTS, global_ids.new_tensor(-1)
    )
    return t0.contiguous(), t1.contiguous()


def _capture_graph(fn, *, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _bench_graph(graph: torch.cuda.CUDAGraph, *, replays: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for idx in range(replays):
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--routing", choices=("mixed", "tier1-heavy"), default="mixed")
    args = parser.parse_args()

    require_sm120()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    sms, max_shared_mem = _device_limits()
    props = torch.cuda.get_device_properties(device)
    m = int(args.m)
    print(
        f"device={props.name} sms={sms} torch={torch.__version__} "
        f"commit={_git_describe()}"
    )
    print(
        f"config: m={m} hidden={_HIDDEN} intermediate={_INTERMEDIATE} "
        f"topk={_TOPK} tiers={_T0_EXPERTS}xNVFP4+{_T1_EXPERTS}xNF3 "
        f"tiles={_TILE_CONFIG} routing={args.routing}"
    )

    prepared_t0 = _build_tier0()
    prepared_t1 = _build_tier1()
    x = torch.randn(m, _HIDDEN, dtype=_DTYPE, device=device) * 0.1
    topk_weights = torch.softmax(
        torch.randn(m, _TOPK, device=device, dtype=torch.float32), dim=-1
    )
    global_ids = _routing(m, args.routing, args.seed)
    t0_local_ids, t1_local_ids = _tier_local_ids(global_ids)
    tier_local_map = build_w4a16_tier_local_map(
        tuple(range(_T0_EXPERTS)),
        tuple(range(_T0_EXPERTS, _MAP_SLOTS)),
        map_slots=_MAP_SLOTS,
        device=device,
    )

    # --- serial two-launch baseline (production fallback path) ---
    serial_state = []
    for prepared, local_ids in (
        (prepared_t0, t0_local_ids),
        (prepared_t1, t1_local_ids),
    ):
        fused = compile_w4a16_fused_moe(
            size_m=m,
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            num_experts=int(prepared.num_experts),
            top_k=_TOPK,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            max_m_blocks=m * _TOPK,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout=prepared.weight_layout,
            scale_format=prepared.scale_format,
            w13_layout=getattr(prepared, "w13_layout", "packed"),
            direct_topk_routes=True,
            tc_decode_fused_sum=True,
            force_tile_config=_TILE_CONFIG,
        )
        buffers = make_w4a16_packed_buffers(
            prepared, m=m, topk=_TOPK, dtype=_DTYPE, device=device
        )
        serial_state.append((prepared, fused, buffers, local_ids))
    serial_sum = torch.zeros(m, _HIDDEN, dtype=_DTYPE, device=device)

    def run_serial() -> None:
        outs = []
        for prepared, fused, buffers, local_ids in serial_state:
            out = run_w4a16_moe(
                x,
                prepared,
                topk_weights,
                local_ids,
                activation="silu",
                intermediate_cache13=buffers.intermediate_cache13,
                intermediate_cache2=buffers.intermediate_cache2,
                output=buffers.output[:m],
                fc1_c_tmp=buffers.fc1_c_tmp,
                fc2_c_tmp=buffers.fc2_c_tmp,
                packed_route_indices=buffers.packed_route_indices,
                block_expert_ids=buffers.block_expert_ids,
                packed_route_count=buffers.packed_route_count,
                fused_launch=fused,
            )
            outs.append(out)
        torch.add(outs[0], outs[1], out=serial_sum)

    # --- hybrid one-grid variants ---
    hybrid_buffers = make_w4a16_packed_buffers(
        prepared_t0, m=m, topk=_TOPK, dtype=_DTYPE, device=device
    )
    hybrid_out = torch.zeros(m, _HIDDEN, dtype=_DTYPE, device=device)
    marshal = dict(
        t0_w13=prepared_t0.w13.view(torch.int32).view(-1),
        t0_w2=prepared_t0.w2.view(torch.int32).view(-1),
        t0_w13_scale=prepared_t0.w13_scale.view(torch.uint8)
        .view(torch.int32)
        .view(-1),
        t0_w2_scale=prepared_t0.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        t1_w13=prepared_t1.w13.view(torch.int32).view(-1),
        t1_w2=prepared_t1.w2.view(torch.int32).view(-1),
        t1_w13_scale=prepared_t1.w13_scale.view(torch.uint8)
        .view(torch.int32)
        .view(-1),
        t1_w2_scale=prepared_t1.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
    )
    fc1_cols = 2 * _INTERMEDIATE
    routed_rows = m * _TOPK
    fc1_out = hybrid_buffers.intermediate_cache13.view(-1)[: routed_rows * fc1_cols]
    activated = hybrid_buffers.intermediate_cache2.view(-1)[
        : routed_rows * _INTERMEDIATE
    ]

    def make_hybrid_fn(schedule_whole_tiles: bool, grid_x: int):
        hybrid = compile_w4a16_fused_moe_hybrid(
            size_m=m,
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            tier0_num_experts=_T0_EXPERTS,
            tier1_num_experts=_T1_EXPERTS,
            top_k=_TOPK,
            activation="silu",
            map_slots=_MAP_SLOTS,
            sms=sms,
            max_shared_mem=max_shared_mem,
            force_tile_config=_TILE_CONFIG,
            schedule_whole_tiles=schedule_whole_tiles,
        )

        def fn() -> None:
            hybrid.compiled(
                make_ptr(
                    _cutlass_element_dtype("bf16"),
                    x.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                marshal["t0_w13"],
                marshal["t0_w2"],
                marshal["t0_w13_scale"],
                marshal["t0_w2_scale"],
                prepared_t0.w13_global_scale,
                prepared_t0.w2_global_scale,
                marshal["t1_w13"],
                marshal["t1_w2"],
                marshal["t1_w13_scale"],
                marshal["t1_w2_scale"],
                prepared_t1.w13_global_scale,
                prepared_t1.w2_global_scale,
                global_ids.view(-1),
                tier_local_map,
                fc1_out,
                activated,
                hybrid_out.view(-1),
                make_ptr(
                    cutlass.Float32,
                    topk_weights.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                hybrid_buffers.fc1_c_tmp,
                hybrid_buffers.fc2_c_tmp,
                prepared_t0.workspace,
                m,
                int(grid_x),
                cuda.CUstream(int(torch.cuda.current_stream().cuda_stream)),
            )

        return fn, hybrid

    policy_grid = None
    cap_grid = sms
    variants: dict[str, object] = {}
    for schedule_whole_tiles in (True, False):
        fn, hybrid = make_hybrid_fn(schedule_whole_tiles, sms)
        if policy_grid is None:
            policy_grid = _w4a16_fused_persistent_grid_x(
                fused=hybrid,
                m=m,
                topk=_TOPK,
                intermediate_size=_INTERMEDIATE,
                activation="silu",
                direct_topk_routes=True,
                sms=sms,
            )
        schedule_name = "whole" if schedule_whole_tiles else "splitk"
        for grid_x in sorted({int(policy_grid), int(cap_grid)}):
            fn_g, _ = make_hybrid_fn(schedule_whole_tiles, grid_x)
            variants[f"hybrid_{schedule_name}_g{grid_x}"] = fn_g
    print(f"grids: policy={policy_grid} cap={cap_grid}")

    # --- correctness gates before timing ---
    run_serial()
    torch.cuda.synchronize()
    reference = serial_sum.float().clone()
    if not torch.isfinite(reference).all():
        raise SystemExit("serial reference produced nonfinite values")
    for name, fn in variants.items():
        hybrid_out.zero_()
        fn()
        torch.cuda.synchronize()
        got = hybrid_out.float()
        rel = ((got - reference).abs().amax() / reference.abs().amax().clamp(min=1e-6))
        if not torch.isfinite(got).all() or float(rel) >= 2e-2:
            raise SystemExit(f"correctness gate failed for {name}: rel={float(rel)}")
        print(f"gate {name}: rel_max_err={float(rel):.2e} OK")

    # --- timing ---
    results: dict[str, list[float]] = {}
    graph = _capture_graph(run_serial, warmup=args.warmup)
    results["serial_2launch"] = _bench_graph(graph, replays=args.replays)
    for name, fn in variants.items():
        graph = _capture_graph(fn, warmup=args.warmup)
        results[name] = _bench_graph(graph, replays=args.replays)

    base_median = statistics.median(results["serial_2launch"])
    print(f"\n{'variant':<24}{'median_us':>12}{'mean_us':>12}{'p90_us':>12}{'vs_serial':>12}")
    for name, times in results.items():
        times_us = [t * 1000.0 for t in times]
        median = statistics.median(times_us)
        mean = statistics.fmean(times_us)
        p90 = sorted(times_us)[int(0.9 * len(times_us))]
        ratio = (base_median * 1000.0) / median if median else float("inf")
        print(f"{name:<24}{median:>12.2f}{mean:>12.2f}{p90:>12.2f}{ratio:>11.3f}x")


if __name__ == "__main__":
    main()
