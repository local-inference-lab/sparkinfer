#!/usr/bin/env python3
"""Benchmark DeepSeek-style WO-A/WO-B projection candidates.

This benchmark times the explicit native MXFP8 two-GEMM skeleton:

    WO-A: [tokens, group_width, groups] x [rank, group_width, groups]
    tmp:  [tokens, rank, groups] -> group-major [tokens, groups * rank]
    WO-B: [tokens, groups * rank] x [hidden, groups * rank]

The sparkinfer path uses owned GPU quant/packing kernels for the activation operands
around the two native MXFP8 dense GEMMs. Weight quantization is still setup
work, matching model-load behavior rather than the per-token serving path.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import sparkinfer.gemm.wo_projection as wo_projection_impl
from sparkinfer.gemm.wo_projection import (
    WOProjectionScratchCaps,
    dequantize_mxfp8_rows_torch,
    pack_wo_projection_fp8_block_scaled_weights_mxfp8,
    plan_wo_projection_scratch,
    quantize_wo_a_input_inv_rope_mxfp8,
    quantize_wo_a_input_mxfp8,
    quantize_wo_projection_weights_mxfp8_torch,
    wo_projection_inv_rope_mxfp8,
    wo_projection_mxfp8,
)
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes


REFERENCE_LABEL = "PyTorch graph BF16 einsum+matmul"
DEEPGEMM_LABEL = "deepgemm"
COSINE_THRESHOLD = 0.995


class BenchmarkAbort(RuntimeError):
    """Fatal benchmark failure that should stop the run without a summary."""


class CorrectnessError(BenchmarkAbort):
    """Raised when replay outputs fail the correctness gate."""


def require_sm120() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")


def bench_events(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    l2_flush: Callable[[], None],
) -> list[float]:
    for _ in range(warmup):
        l2_flush()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        l2_flush()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]


def fmt_us(times_ms: list[float]) -> str:
    med = statistics.median(times_ms) * 1000.0
    mn = min(times_ms) * 1000.0
    return f"{med:8.1f} us (min {mn:.1f})"


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return F.cosine_similarity(a_f, b_f, dim=0).item()


def check_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    label: str,
) -> None:
    cand_finite = bool(torch.isfinite(candidate).all().item())
    ref_finite = bool(torch.isfinite(reference).all().item())
    if not cand_finite or not ref_finite:
        raise CorrectnessError(
            f"non-finite output detected vs {label}: "
            f"candidate_finite={cand_finite}, reference_finite={ref_finite}"
        )
    diff = (candidate.float() - reference.float()).abs()
    max_abs = diff.max().item()
    rmse = diff.square().mean().sqrt().item()
    cos = cosine_similarity(candidate, reference)
    print(f"    check vs {label}: max_abs={max_abs:.8f} rmse={rmse:.8f} cos={cos:.10f}")
    if not math.isfinite(cos) or cos < COSINE_THRESHOLD:
        raise CorrectnessError(
            f"cosine similarity vs {label} fell below {COSINE_THRESHOLD:.6f}: "
            f"max_abs={max_abs:.8f}, rmse={rmse:.8f}, cos={cos}"
        )


def capture_graph_replay(
    fn: Callable[[], torch.Tensor | None],
) -> tuple[Callable[[], None], torch.Tensor | None]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    replay()
    torch.cuda.synchronize()
    return replay, graph_output


def _block_fp8_checkpoint(
    weight_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 `[N,K]` weight to checkpoint-style FP8 with exact
    power-of-two 128x128 block scales, so sparkinfer and DeepGEMM consume identical
    FP8 bytes."""

    n, k = map(int, weight_bf16.shape)
    if n % 128 or k % 128:
        raise ValueError(f"block-scaled checkpoint needs 128-divisible dims, got {n}x{k}")
    blocked = weight_bf16.reshape(n // 128, 128, k // 128, 128)
    amax = blocked.abs().amax(dim=(1, 3), keepdim=True).float()
    scale = torch.pow(2.0, torch.ceil(torch.log2((amax / 448.0).clamp(min=1e-12))))
    w_fp8 = (blocked.float() / scale).to(torch.float8_e4m3fn).reshape(n, k)
    return w_fp8, scale[:, 0, :, 0].contiguous()


def build_deepgemm_launch(
    case: dict[str, torch.Tensor | object],
    *,
    tokens: int,
    groups: int,
    rank: int,
    hidden: int,
    nope_dim: int,
    rope_dim: int,
) -> Callable[[], torch.Tensor]:
    """Build the serving DeepGEMM WO chain (vLLM deep_gemm_fp8_o_proj):
    fused inverse-RoPE FP8 quant -> fp8_einsum (WO-A) -> QuantFP8 ->
    fp8_gemm_nt (WO-B), from the same FP8 checkpoint tensors as sparkinfer."""

    import vllm.envs as envs
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
    from vllm.utils.deep_gemm import fp8_einsum, fp8_gemm_nt, is_deep_gemm_e8m0_used

    use_e8m0 = is_deep_gemm_e8m0_used()
    recipe, tma_aligned = compute_fp8_einsum_recipe()
    wa_dg, wa_dg_scale = deepgemm_post_process_fp8_weight_block(
        wq=case["ckpt_wo_a_fp8"].clone(),
        ws=case["ckpt_wo_a_scale"].clone(),
        quant_block_shape=(128, 128),
        use_e8m0=use_e8m0,
        is_bmm=True,
        bmm_batch_size=groups,
    )
    wb_dg, wb_dg_scale = deepgemm_post_process_fp8_weight_block(
        wq=case["ckpt_wo_b_fp8"].clone(),
        ws=case["ckpt_wo_b_scale"].clone(),
        quant_block_shape=(128, 128),
        use_e8m0=use_e8m0,
    )
    with set_current_vllm_config(VllmConfig()):
        quant_fp8 = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            use_ue8m0=use_e8m0,
            tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES,
            column_major_scales=True,
        )
    o = case["o"]
    positions = case["positions"]
    cos_sin_cache = case["cos_sin_cache"]
    heads_per_group = int(case["heads_per_group"])

    def launch() -> torch.Tensor:
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            tma_aligned_scales=tma_aligned,
        )
        z = torch.empty((tokens, groups, rank), device=o.device, dtype=torch.bfloat16)
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (wa_dg, wa_dg_scale),
            z,
            recipe=recipe,
        )
        q, s = quant_fp8(z.flatten(1))
        out = torch.empty((tokens, hidden), device=o.device, dtype=torch.bfloat16)
        fp8_gemm_nt((q, s), (wb_dg, wb_dg_scale), out, is_deep_gemm_e8m0_used=use_e8m0)
        return out

    return launch


def make_case(
    *,
    tokens: int,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
    seed: int,
    inv_rope: bool,
    context_length: int,
    nope_dim: int,
    rope_dim: int,
    block_scaled_weights: bool = False,
) -> dict[str, torch.Tensor | object]:
    torch.manual_seed(seed)
    if inv_rope:
        head_dim = nope_dim + rope_dim
        if group_width % head_dim:
            raise ValueError(
                f"group_width={group_width} must be divisible by head_dim={head_dim}"
            )
        if tokens > context_length:
            raise ValueError(f"tokens={tokens} exceeds context_length={context_length}")
        heads_per_group = group_width // head_dim
        o = (
            torch.randn(
                (tokens, groups * heads_per_group, head_dim),
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 4
        ).contiguous()
        positions = torch.arange(
            context_length - tokens,
            context_length,
            device="cuda",
            dtype=torch.long,
        )
        angles = torch.randn(
            (context_length, rope_dim // 2),
            device="cuda",
            dtype=torch.float32,
        )
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=1)
        x_tgd = None
        x_tdg_q = quantize_wo_a_input_inv_rope_mxfp8(
            o,
            positions,
            cos_sin_cache,
            groups=groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
    else:
        heads_per_group = 0
        o = None
        positions = None
        cos_sin_cache = None
        x_tgd = (
            torch.randn(
                (tokens, groups, group_width),
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 4
        ).contiguous()
        x_tdg_q = quantize_wo_a_input_mxfp8(x_tgd)
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    ).contiguous()
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    ).contiguous()

    ckpt_wo_a_fp8 = ckpt_wo_a_scale = ckpt_wo_b_fp8 = ckpt_wo_b_scale = None
    if block_scaled_weights:
        # Serving checkpoints carry FP8 weights with 128x128 block scales;
        # both sparkinfer and DeepGEMM pack from the same FP8 bytes.
        ckpt_wo_a_fp8, ckpt_wo_a_scale = _block_fp8_checkpoint(
            wo_a_grd.reshape(groups * rank, group_width)
        )
        ckpt_wo_b_fp8, ckpt_wo_b_scale = _block_fp8_checkpoint(wo_b_hgr)
        weights = pack_wo_projection_fp8_block_scaled_weights_mxfp8(
            ckpt_wo_a_fp8,
            ckpt_wo_a_scale,
            ckpt_wo_b_fp8,
            ckpt_wo_b_scale,
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
        )
    else:
        weights = quantize_wo_projection_weights_mxfp8_torch(wo_a_grd, wo_b_hgr)

    x_deq_tgd = dequantize_mxfp8_rows_torch(
        x_tdg_q.values,
        x_tdg_q.scale_rows,
    )
    if groups == 1:
        x_deq_tgd = x_deq_tgd.unsqueeze(1)
    else:
        x_deq_tgd = x_deq_tgd.permute(0, 2, 1)
    x_deq_tgd = x_deq_tgd.to(torch.bfloat16)
    wo_a_deq_grd = dequantize_mxfp8_rows_torch(
        weights.wo_a.values,
        weights.wo_a.scale_rows,
    )
    if groups == 1:
        wo_a_deq_grd = wo_a_deq_grd.unsqueeze(0)
    else:
        wo_a_deq_grd = wo_a_deq_grd.permute(2, 0, 1)
    wo_a_deq_grd = wo_a_deq_grd.to(torch.bfloat16)
    wo_b_deq_hgr = dequantize_mxfp8_rows_torch(
        weights.wo_b.values,
        weights.wo_b.scale_rows,
    ).to(torch.bfloat16)

    return {
        "x_tgd": x_tgd,
        "o": o,
        "positions": positions,
        "cos_sin_cache": cos_sin_cache,
        "heads_per_group": heads_per_group,
        "wo_a_grd": wo_a_grd,
        "wo_b_hgr": wo_b_hgr,
        "x_tdg_q": x_tdg_q,
        "weights": weights,
        "x_deq_tgd": x_deq_tgd,
        "wo_a_deq_grd": wo_a_deq_grd,
        "wo_b_deq_hgr": wo_b_deq_hgr,
        "ckpt_wo_a_fp8": ckpt_wo_a_fp8,
        "ckpt_wo_a_scale": ckpt_wo_a_scale,
        "ckpt_wo_b_fp8": ckpt_wo_b_fp8,
        "ckpt_wo_b_scale": ckpt_wo_b_scale,
    }


def bench_one(
    tokens: int,
    *,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None],
    seed: int,
    inv_rope: bool,
    context_length: int,
    nope_dim: int,
    rope_dim: int,
    compare_deepgemm: bool = False,
) -> dict[str, object]:
    case = make_case(
        tokens=tokens,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
        seed=seed,
        inv_rope=inv_rope,
        context_length=context_length,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        block_scaled_weights=compare_deepgemm,
    )

    results: dict[str, object] = {}

    try:
        plan = plan_wo_projection_scratch(
            WOProjectionScratchCaps(
                device="cuda",
                max_tokens=tokens,
                groups=groups,
                group_width=group_width,
                rank=rank,
                hidden=hidden,
            )
        )
        scratch = tuple(
            torch.empty(shape, dtype=dtype, device="cuda")
            for shape, dtype in plan.shapes_and_dtypes()
        )
        if inv_rope:
            binding = plan.bind_inv_rope(
                scratch=scratch,
                o=case["o"],
                positions=case["positions"],
                cos_sin_cache=case["cos_sin_cache"],
                weights=case["weights"],
                heads_per_group=case["heads_per_group"],
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                return_3d=True,
                expected_m=tokens,
            )

            def sparkinfer_launch() -> torch.Tensor:
                return wo_projection_inv_rope_mxfp8(binding=binding)

        else:
            binding = plan.bind(
                scratch=scratch,
                source_tgd=case["x_tgd"],
                weights=case["weights"],
                return_3d=True,
                expected_m=tokens,
            )

            def sparkinfer_launch() -> torch.Tensor:
                return wo_projection_mxfp8(binding=binding)

        sparkinfer_replay, sparkinfer_graph_out = capture_graph_replay(sparkinfer_launch)
        results["sparkinfer_replay"] = sparkinfer_replay
        results["sparkinfer_out"] = sparkinfer_graph_out if inv_rope else binding.output
        results["sparkinfer"] = bench_events(
            sparkinfer_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results["sparkinfer"] = None
        print(f"      sparkinfer two-GEMM FAILED: {exc}")

    try:
        torch_outputs: list[torch.Tensor | None] = [None]

        def torch_launch() -> torch.Tensor:
            tmp_ref = torch.einsum(
                "tgd,grd->tgr",
                case["x_deq_tgd"],
                case["wo_a_deq_grd"],
            )
            out = tmp_ref.reshape(tokens, groups * rank) @ case["wo_b_deq_hgr"].T
            torch_outputs[0] = out
            return out

        torch_replay, torch_graph_out = capture_graph_replay(torch_launch)
        if torch_graph_out is None:
            torch_graph_out = torch_outputs[0]
        results["torch_replay"] = torch_replay
        results["torch_out"] = torch_graph_out
        results[REFERENCE_LABEL] = bench_events(
            torch_replay,
            warmup=warmup,
            iters=iters,
            l2_flush=l2_flush,
        )
    except Exception as exc:
        results[REFERENCE_LABEL] = None
        print(f"      {REFERENCE_LABEL} FAILED: {exc}")

    if compare_deepgemm:
        try:
            dg_launch = build_deepgemm_launch(
                case,
                tokens=tokens,
                groups=groups,
                rank=rank,
                hidden=hidden,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
            )
            dg_replay, dg_graph_out = capture_graph_replay(dg_launch)
            results["deepgemm_replay"] = dg_replay
            results["deepgemm_out"] = dg_graph_out
            results[DEEPGEMM_LABEL] = bench_events(
                dg_replay,
                warmup=warmup,
                iters=iters,
                l2_flush=l2_flush,
            )
        except Exception as exc:
            results[DEEPGEMM_LABEL] = None
            print(f"      deepgemm WO chain FAILED: {exc}")

    if check:
        if results.get("sparkinfer_replay") is None or results.get("torch_replay") is None:
            raise BenchmarkAbort(
                "correctness check requires both sparkinfer and torch replays"
            )
        results["sparkinfer_replay"]()
        results["torch_replay"]()
        torch.cuda.synchronize()
        check_outputs(
            results["sparkinfer_out"][:, :, 0],
            results["torch_out"],
            label=REFERENCE_LABEL,
        )
        if compare_deepgemm and results.get("deepgemm_replay") is not None:
            results["deepgemm_replay"]()
            torch.cuda.synchronize()
            check_outputs(
                results["deepgemm_out"],
                results["torch_out"],
                label=f"{REFERENCE_LABEL} (deepgemm route)",
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--group-width", type=int, default=512)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument(
        "--inv-rope",
        action="store_true",
        help="Benchmark the opaque inverse-RoPE serving route.",
    )
    parser.add_argument(
        "--compare-deepgemm",
        action="store_true",
        help=(
            "Also run the serving DeepGEMM WO chain (vLLM deep_gemm_fp8_o_proj: "
            "fused inv-RoPE FP8 quant + fp8_einsum + QuantFP8 + fp8_gemm_nt) from "
            "the same FP8 block-scaled checkpoint weights. Implies --inv-rope and "
            "requires a vLLM+deep_gemm environment."
        ),
    )
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--nope-dim", type=int, default=448)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="Bytes to touch when evicting L2; 0 uses 2x the reported L2 size.",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    parser.set_defaults(check=True)
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Run correctness checks and fail hard when cosine similarity falls below the threshold (default: enabled).",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Disable correctness checks before timing.",
    )
    args = parser.parse_args()
    if args.compare_deepgemm:
        args.inv_rope = True

    require_sm120()
    torch.empty(1, device="cuda")
    l2_flush = make_l2_flush_fn(
        enabled=True,
        bytes_hint=args.l2_flush_bytes,
    )
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)

    print(f"WO projection: sparkinfer native MXFP8 two-GEMM vs {REFERENCE_LABEL}")
    route = "inverse-RoPE serving op" if args.inv_rope else "plain WO binding"
    print(f"Route: {route}")
    if args.compare_deepgemm:
        print(
            "DeepGEMM comparison: serving deep_gemm_fp8_o_proj chain, both "
            "routes packed from the same FP8 128x128 block-scaled checkpoint"
        )
    print("Timing mode: CUDA graph replay")
    print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    if args.check:
        print(f"Correctness check: on (cos >= {COSINE_THRESHOLD:.6f})")
    else:
        print("Correctness check: off")
    print(
        "Shape: "
        f"groups={args.groups}, group_width={args.group_width}, "
        f"rank={args.rank}, hidden={args.hidden}"
    )
    if args.inv_rope:
        print(
            f"RoPE: context={args.context_length}, nope_dim={args.nope_dim}, "
            f"rope_dim={args.rope_dim}"
        )
    print(
        "WO quant tile: "
        f"{wo_projection_impl._WO_QUANT_CHUNKS_PER_PROGRAM}x32 values/program, "
        "4 warps"
    )
    print(
        "sparkinfer note: activation quant/scale packing is included in the graph replay path."
    )
    print(f"warmup={args.warmup}, iters={args.iters}")
    print()

    all_results = []
    for tokens in args.token_counts:
        try:
            results = bench_one(
                tokens,
                groups=args.groups,
                group_width=args.group_width,
                rank=args.rank,
                hidden=args.hidden,
                warmup=args.warmup,
                iters=args.iters,
                check=args.check,
                l2_flush=l2_flush,
                seed=args.seed + tokens,
                inv_rope=args.inv_rope,
                context_length=args.context_length,
                nope_dim=args.nope_dim,
                rope_dim=args.rope_dim,
                compare_deepgemm=args.compare_deepgemm,
            )
        except BenchmarkAbort as exc:
            print(
                f"ERROR: benchmark aborted for tokens={tokens}: {exc}", file=sys.stderr
            )
            raise SystemExit(1) from exc

        sparkinfer_times = results.get("sparkinfer")
        torch_times = results.get(REFERENCE_LABEL)
        dg_times = results.get(DEEPGEMM_LABEL)
        sparkinfer_med = statistics.median(sparkinfer_times) * 1000.0 if sparkinfer_times else None
        torch_med = statistics.median(torch_times) * 1000.0 if torch_times else None
        dg_med = statistics.median(dg_times) * 1000.0 if dg_times else None

        parts = [f"  tokens={tokens:<4}"]
        if sparkinfer_times is not None:
            parts.append(f"sparkinfer={fmt_us(sparkinfer_times)}")
        if dg_times is not None:
            parts.append(f"deepgemm={fmt_us(dg_times)}")
        if torch_times is not None:
            parts.append(f"torch={fmt_us(torch_times)}")
        if sparkinfer_med is not None and dg_med is not None:
            parts.append(f"sparkinfer/dg={sparkinfer_med / dg_med:.2f}x")
        if sparkinfer_med is not None and torch_med is not None:
            parts.append(f"sparkinfer/torch={sparkinfer_med / torch_med:.2f}x")
        print("  ".join(parts) + "  (graph replay)")
        all_results.append((tokens, sparkinfer_med, torch_med, dg_med))

    def print_summary(label: str, pairs: list[tuple[int, float | None, float | None]]) -> None:
        print(f"\n{'=' * 75}")
        print(f"  SUMMARY: sparkinfer / {label} (CUDA graph replay, lower = sparkinfer faster)")
        print(f"{'=' * 75}")
        print(f"  {'tokens':<10}  {'ratio':>10}")
        print("  " + "-" * 24)
        ratios = []
        for tokens, sparkinfer_med, other_med in pairs:
            if sparkinfer_med is not None and other_med is not None:
                ratio = sparkinfer_med / other_med
                ratios.append(ratio)
                print(f"  {tokens:<10}  {ratio:>9.2f}x")
            else:
                print(f"  {tokens:<10}  {'n/a':>10}")
        if ratios:
            geo = 1.0
            for ratio in ratios:
                geo *= ratio
            geo **= 1.0 / len(ratios)
            print(f"\n  geo mean: {geo:.2f}x")

    if args.compare_deepgemm:
        print_summary(
            f"{DEEPGEMM_LABEL} serving WO chain",
            [(t, b, d) for t, b, _, d in all_results],
        )
    print_summary(REFERENCE_LABEL, [(t, b, r) for t, b, r, _ in all_results])


if __name__ == "__main__":
    main()
