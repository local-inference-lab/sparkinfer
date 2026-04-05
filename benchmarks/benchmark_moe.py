#!/usr/bin/env python3
"""Static MoE benchmark and shared weight-loading utilities.

By default this is a pre-routed benchmark: model loading, routing-logit
generation, top-k selection, compilation, and oracle/reference checks all
happen outside the timed region. Use ``--include-routing`` to include the
deterministic top-k + softmax routing step in the measured closure.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.checkpoint_loader import IndexedSafetensorLoader
from b12x.moe.fused.reference import (
    OracleMetrics,
    compare_to_reference,
    moe_reference_f32,
    moe_reference_nvfp4,
)
from b12x.cute.fp4 import as_grouped_scale_view, swizzle_block_scale


MODEL_PATH = pathlib.Path(os.environ.get("B12X_MODEL_PATH", "/data/models/Qwen3.5-397B-A17B-NVFP4"))

LEGACY_BATCH_SIZES = [1, 2, 4, 8]
# Observed in the live single-request sglang probe:
# - prefill m=23 for the prompt itself
# - larger prefill chunk m=80 during the same request path
# - decode remains effectively m=1 for a single running request
RECORDED_SGLANG_SINGLE_REQUEST_BATCH_SIZES = [1, 23, 80]
# Representative eager-prefill forwards without CUDA graph replay.
EAGER_PREFILL_BATCH_SIZES = [16384, 32768]
# Representative total-token sizes for packed chunked-prefill forwards.
# The first point is one full server-side prefill chunk, then we scale to
# larger packed forwards up to four chunks' worth of tokens.
CHUNKED_PREFILL_BATCH_SIZES = [8192, 16384, 24576, 32768]
BATCH_SIZE_PROFILES = {
    "eager-prefill": EAGER_PREFILL_BATCH_SIZES,
    "micro": LEGACY_BATCH_SIZES,
    "sglang-single-request": RECORDED_SGLANG_SINGLE_REQUEST_BATCH_SIZES,
    "chunked-prefill": CHUNKED_PREFILL_BATCH_SIZES,
}
TP_SIZE = 4
TP_RANK = 0
EP_SIZE = 1
EP_RANK = 0


def require_sm120() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(f"Requires sm_120, got sm_{major}{minor}")


def bench_events(fn: Callable[[], None], *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def fmt_us(times_ms: list[float]) -> str:
    median_us = statistics.median(times_ms) * 1000.0
    min_us = min(times_ms) * 1000.0
    return f"{median_us:8.1f} us (min {min_us:.1f})"


@dataclass(frozen=True)
class TimingStats:
    per_repeat_median_ms: list[float]
    per_repeat_min_ms: list[float]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.per_repeat_median_ms)

    @property
    def min_ms(self) -> float:
        return min(self.per_repeat_min_ms)

    @property
    def median_us(self) -> float:
        return self.median_ms * 1000.0

    @property
    def min_us(self) -> float:
        return self.min_ms * 1000.0

    @property
    def repeat_count(self) -> int:
        return len(self.per_repeat_median_ms)

    @property
    def median_range_us(self) -> tuple[float, float]:
        return (
            min(self.per_repeat_median_ms) * 1000.0,
            max(self.per_repeat_median_ms) * 1000.0,
        )


@dataclass(frozen=True)
class RatioStats:
    per_repeat_ratio: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.per_repeat_ratio)

    @property
    def min(self) -> float:
        return min(self.per_repeat_ratio)

    @property
    def max(self) -> float:
        return max(self.per_repeat_ratio)

    @property
    def repeat_count(self) -> int:
        return len(self.per_repeat_ratio)


@dataclass(frozen=True)
class BatchResult:
    backend_stats: TimingStats
    ref_stats: TimingStats | None
    ratio_stats: RatioStats | None


def summarize_timing_runs(runs_ms: list[list[float]]) -> TimingStats:
    return TimingStats(
        per_repeat_median_ms=[statistics.median(run) for run in runs_ms],
        per_repeat_min_ms=[min(run) for run in runs_ms],
    )


def fmt_timing_stats(stats: TimingStats) -> str:
    if stats.repeat_count == 1:
        return f"{stats.median_us:8.1f} us (min {stats.min_us:.1f})"
    low_us, high_us = stats.median_range_us
    return (
        f"{stats.median_us:8.1f} us "
        f"(repeat medians {low_us:.1f}-{high_us:.1f}, sample min {stats.min_us:.1f})"
    )


def fmt_ratio_stats(stats: RatioStats) -> str:
    if stats.repeat_count == 1:
        return f"{stats.median:.2f}x"
    return f"{stats.median:.2f}x (repeat range {stats.min:.2f}-{stats.max:.2f})"


@dataclass(frozen=True)
class ScaleContractParams:
    a1_gscale: torch.Tensor
    a2_gscale: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor


@dataclass
class ModelSpec:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    tp_size: int
    tp_rank: int

    @property
    def I_tp(self) -> int:
        return self.intermediate_size // self.tp_size


@dataclass
class ExpertWeights:
    layer_idx: int
    spec: ModelSpec
    w13_permuted: torch.Tensor
    w13_scale: torch.Tensor
    down_permuted: torch.Tensor
    down_scale: torch.Tensor
    w13_weight: torch.Tensor
    w13_blockscale_swizzled: torch.Tensor
    w2_weight: torch.Tensor
    w2_blockscale_swizzled: torch.Tensor
    w13_input_scale: torch.Tensor
    w2_input_scale: torch.Tensor
    w13_input_scale_quant: torch.Tensor
    w2_input_scale_quant: torch.Tensor
    w13_input_scale_per_expert: torch.Tensor
    w2_input_scale_per_expert: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor
    g1_alphas_per_expert: torch.Tensor
    g2_alphas_per_expert: torch.Tensor


@dataclass
class SharedExpertWeights:
    layer_idx: int
    spec: ModelSpec
    w13_weight: torch.Tensor
    w13_blockscale_swizzled: torch.Tensor
    w2_weight: torch.Tensor
    w2_blockscale_swizzled: torch.Tensor
    w13_input_scale_quant: torch.Tensor
    w2_input_scale_quant: torch.Tensor
    w13_input_scale: torch.Tensor
    w2_input_scale: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor


def load_expert_weights(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_idx: int = 0,
) -> ExpertWeights:
    cfg = json.loads((model_path / "config.json").read_text())["text_config"]
    assert cfg["num_experts"] == spec.num_experts
    assert cfg["moe_intermediate_size"] == spec.intermediate_size
    assert cfg["hidden_size"] == spec.hidden_size

    device = torch.device("cuda")
    E = spec.num_experts
    K = spec.hidden_size
    I_tp = spec.I_tp
    prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
    loader = IndexedSafetensorLoader(model_path)

    gate_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
    up_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
    down_w = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)

    gate_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
    up_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
    down_sf = torch.empty(E, K, I_tp // 16, dtype=torch.float8_e4m3fn, device=device)

    gate_gs = torch.empty(E, dtype=torch.float32, device=device)
    down_gs = torch.empty(E, dtype=torch.float32, device=device)
    gate_is = torch.empty(E, dtype=torch.float32, device=device)
    down_is = torch.empty(E, dtype=torch.float32, device=device)

    print(f"  Loading {E} experts...", end="", flush=True)
    for eid in range(E):
        ep = f"{prefix}.{eid}"
        tp_off = TP_RANK * I_tp
        tp_off_packed = TP_RANK * (I_tp // 2)
        tp_sf_cols = I_tp // 16
        tp_sf_off = TP_RANK * tp_sf_cols

        gate_w[eid] = loader.get_tensor(f"{ep}.gate_proj.weight").narrow(0, tp_off, I_tp).to(device)
        gate_sf[eid] = loader.get_tensor(f"{ep}.gate_proj.weight_scale").narrow(0, tp_off, I_tp).to(device)
        gate_gs[eid] = loader.get_tensor(f"{ep}.gate_proj.weight_scale_2").to(device)
        gate_is[eid] = loader.get_tensor(f"{ep}.gate_proj.input_scale").to(device)

        up_w[eid] = loader.get_tensor(f"{ep}.up_proj.weight").narrow(0, tp_off, I_tp).to(device)
        up_sf[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale").narrow(0, tp_off, I_tp).to(device)

        down_w[eid] = loader.get_tensor(f"{ep}.down_proj.weight").narrow(1, tp_off_packed, I_tp // 2).to(device)
        down_sf[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale").narrow(1, tp_sf_off, tp_sf_cols).to(device)
        down_gs[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale_2").to(device)
        down_is[eid] = loader.get_tensor(f"{ep}.down_proj.input_scale").to(device)
    print(" done.")

    w13_weight = torch.cat([up_w, gate_w], dim=1).contiguous()
    w13_sf = torch.cat([up_sf, gate_sf], dim=1).contiguous()
    w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
    w2_weight = down_w.contiguous()
    w2_blockscale_swizzled = swizzle_block_scale(down_sf)

    w13_permuted = w13_weight.permute(1, 2, 0)
    w13_scale = as_grouped_scale_view(w13_blockscale_swizzled.view(torch.uint8), 2 * I_tp, K)
    down_permuted = w2_weight.permute(1, 2, 0)
    down_scale = as_grouped_scale_view(w2_blockscale_swizzled.view(torch.uint8), K, I_tp)

    w13_input_scale = gate_is.max()
    w2_input_scale = down_is.max()
    g1_alphas = (w13_input_scale * gate_gs).to(torch.float32)
    g2_alphas = (w2_input_scale * down_gs).to(torch.float32)
    g1_alphas_per_expert = (gate_is * gate_gs).to(torch.float32)
    g2_alphas_per_expert = (down_is * down_gs).to(torch.float32)
    w13_input_scale_quant = (1.0 / w13_input_scale).to(torch.float32)
    w2_input_scale_quant = (1.0 / w2_input_scale).to(torch.float32)

    return ExpertWeights(
        layer_idx=layer_idx,
        spec=spec,
        w13_permuted=w13_permuted,
        w13_scale=w13_scale,
        down_permuted=down_permuted,
        down_scale=down_scale,
        w13_weight=w13_weight,
        w13_blockscale_swizzled=w13_blockscale_swizzled,
        w2_weight=w2_weight,
        w2_blockscale_swizzled=w2_blockscale_swizzled,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
        w13_input_scale_quant=w13_input_scale_quant,
        w2_input_scale_quant=w2_input_scale_quant,
        w13_input_scale_per_expert=gate_is,
        w2_input_scale_per_expert=down_is,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        g1_alphas_per_expert=g1_alphas_per_expert,
        g2_alphas_per_expert=g2_alphas_per_expert,
    )


def load_shared_expert_weights(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_idx: int = 0,
) -> SharedExpertWeights:
    cfg = json.loads((model_path / "config.json").read_text())["text_config"]
    assert cfg["shared_expert_intermediate_size"] == spec.intermediate_size
    assert cfg["hidden_size"] == spec.hidden_size

    device = torch.device("cuda")
    K = spec.hidden_size
    I_tp = spec.I_tp
    prefix = f"model.language_model.layers.{layer_idx}.mlp.shared_expert"
    loader = IndexedSafetensorLoader(model_path)

    tp_off = spec.tp_rank * I_tp
    tp_off_packed = spec.tp_rank * (I_tp // 2)
    tp_sf_cols = I_tp // 16
    tp_sf_off = spec.tp_rank * tp_sf_cols

    gate_w = loader.get_tensor(f"{prefix}.gate_proj.weight").narrow(0, tp_off, I_tp).to(device)
    gate_sf = loader.get_tensor(f"{prefix}.gate_proj.weight_scale").narrow(0, tp_off, I_tp).to(device)
    gate_gs = loader.get_tensor(f"{prefix}.gate_proj.weight_scale_2").to(device)
    gate_is = loader.get_tensor(f"{prefix}.gate_proj.input_scale").to(device)

    up_w = loader.get_tensor(f"{prefix}.up_proj.weight").narrow(0, tp_off, I_tp).to(device)
    up_sf = loader.get_tensor(f"{prefix}.up_proj.weight_scale").narrow(0, tp_off, I_tp).to(device)

    down_w = loader.get_tensor(f"{prefix}.down_proj.weight").narrow(1, tp_off_packed, I_tp // 2).to(device)
    down_sf = loader.get_tensor(f"{prefix}.down_proj.weight_scale").narrow(1, tp_sf_off, tp_sf_cols).to(device)
    down_gs = loader.get_tensor(f"{prefix}.down_proj.weight_scale_2").to(device)
    down_is = loader.get_tensor(f"{prefix}.down_proj.input_scale").to(device)

    w13_weight = torch.cat([up_w, gate_w], dim=0).unsqueeze(0).contiguous()
    w13_sf = torch.cat([up_sf, gate_sf], dim=0).unsqueeze(0).contiguous()
    w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
    w2_weight = down_w.unsqueeze(0).contiguous()
    w2_blockscale_swizzled = swizzle_block_scale(down_sf.unsqueeze(0))

    w13_input_scale = gate_is.to(torch.float32).reshape(1)
    w2_input_scale = down_is.to(torch.float32).reshape(1)
    w13_input_scale_quant = (1.0 / w13_input_scale).to(torch.float32)
    w2_input_scale_quant = (1.0 / w2_input_scale).to(torch.float32)
    g1_alphas = (w13_input_scale * gate_gs.to(torch.float32)).to(torch.float32)
    g2_alphas = (w2_input_scale * down_gs.to(torch.float32)).to(torch.float32)

    return SharedExpertWeights(
        layer_idx=layer_idx,
        spec=spec,
        w13_weight=w13_weight,
        w13_blockscale_swizzled=w13_blockscale_swizzled,
        w2_weight=w2_weight,
        w2_blockscale_swizzled=w2_blockscale_swizzled,
        w13_input_scale_quant=w13_input_scale_quant,
        w2_input_scale_quant=w2_input_scale_quant,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
    )


def load_expert_weight_stack(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_start: int,
    num_layers: int,
) -> list[ExpertWeights]:
    return [
        load_expert_weights(
            model_path,
            spec,
            layer_idx=layer_start + layer_offset,
        )
        for layer_offset in range(num_layers)
    ]


def load_gate_weight(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Load the replicated sparse-gate projection for a Qwen-style MoE block."""
    cfg = json.loads((model_path / "config.json").read_text())["text_config"]
    assert cfg["num_experts"] == spec.num_experts
    assert cfg["hidden_size"] == spec.hidden_size

    gate_weight = IndexedSafetensorLoader(model_path).get_tensor(
        f"model.language_model.layers.{layer_idx}.mlp.gate.weight"
    )
    expected_shape = (spec.num_experts, spec.hidden_size)
    if tuple(gate_weight.shape) != expected_shape:
        raise ValueError(
            f"expected gate.weight shape {expected_shape}, got {tuple(gate_weight.shape)}"
        )
    return gate_weight.to(device=torch.device("cuda")).contiguous()


def load_shared_gate_weight(
    model_path: pathlib.Path,
    *,
    layer_idx: int = 0,
) -> torch.Tensor:
    weight = IndexedSafetensorLoader(model_path).get_tensor(
        f"model.language_model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
    )
    return weight.to(device=torch.device("cuda"), dtype=torch.bfloat16).contiguous()


def make_input_activations(
    spec: ModelSpec,
    m: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_heads = 16
    if spec.hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size={spec.hidden_size} must be divisible by {num_heads}")
    head_dim = spec.hidden_size // num_heads

    q = torch.randn(m, num_heads, head_dim, generator=generator, dtype=torch.float32)
    k = torch.randn(m, num_heads, head_dim, generator=generator, dtype=torch.float32)
    v = torch.randn(m, num_heads, head_dim, generator=generator, dtype=torch.float32)

    logits = torch.einsum("thd,shd->hts", q, k) / (head_dim ** 0.5)

    # Add realistic pre-softmax structure:
    # - locality bias
    # - per-head bias offsets
    # - occasional large positive logit spikes that create concentrated attention
    pos = torch.arange(m, dtype=torch.float32)
    locality_bias = -0.18 * (pos[:, None] - pos[None, :]).abs()
    head_bias = 0.35 * torch.randn(num_heads, 1, 1, generator=generator, dtype=torch.float32)
    logits = logits + locality_bias.unsqueeze(0) + head_bias

    spike_mask = torch.rand(num_heads, m, m, generator=generator, dtype=torch.float32) < 0.01
    if spike_mask.any():
        spike_mag = torch.exp(1.0 + 0.8 * torch.randn(num_heads, m, m, generator=generator, dtype=torch.float32))
        logits = logits + spike_mask.to(torch.float32) * spike_mag

    attn = torch.softmax(logits, dim=-1)
    x = torch.einsum("hts,shd->thd", attn, v).reshape(m, spec.hidden_size)

    # Post-attention RMSNorm-like shaping, without injecting new variance after normalization.
    row_rms = x.square().mean(dim=-1, keepdim=True).add_(1e-6).sqrt_()
    x = x / row_rms
    return x.to(device=device, dtype=torch.bfloat16)


def make_global_scale_overflow_activations(
    spec: ModelSpec,
    m: int,
    seed: int,
    device: torch.device,
    *,
    global_scale: float,
    spike_multiple: float = 8.0,
    background_scale: float = 0.02,
) -> torch.Tensor:
    """Construct RMSNorm-like activations with one block-local overflow spike per row.

    The planted spike exceeds the largest value representable by the FP4+FP8
    activation quantizer under a fixed outer input scale:

        max_representable = 6 * max_fp8_e4m3 * global_scale

    The rest of the 16-value block is attenuated so the row is dominated by
    that single outlier, which is the adversarial case for a coarse global
    input scale.
    """
    x = make_input_activations(spec, m, seed, device).float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 97)

    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)
    spike_value = spike_multiple * 6.0 * fp8_e4m3_max * float(global_scale)
    num_blocks = spec.hidden_size // 16

    for row in range(m):
        block_idx = int(torch.randint(num_blocks, (1,), generator=generator).item())
        block_start = block_idx * 16
        x[row, block_start:block_start + 16] *= background_scale
        sign = -1.0 if torch.rand((), generator=generator).item() < 0.5 else 1.0
        x[row, block_start] = sign * spike_value

    return x.to(device=device, dtype=torch.bfloat16)


def make_routed_inputs(
    spec: ModelSpec,
    m: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = make_input_activations(spec, m, seed, device)
    routing_generator = torch.Generator(device="cpu")
    routing_generator.manual_seed(seed + 1)
    routing_logits = torch.randn(
        m,
        spec.num_experts,
        generator=routing_generator,
        dtype=torch.float32,
    ).to(device=device)
    topk_logits, topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    return x, topk_ids, topk_weights


def _make_structured_routing_ids(
    spec: ModelSpec,
    m: int,
    *,
    layer_idx: int,
    pattern: str,
    seed: int,
) -> torch.Tensor:
    if pattern not in {"disjoint", "overlap", "random"}:
        raise ValueError(f"unsupported routing pattern {pattern!r}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer_idx * 101)
    top_k = spec.top_k
    expert_count = spec.num_experts

    if pattern == "random":
        ids = torch.empty(m, top_k, dtype=torch.int64)
        for token_idx in range(m):
            ids[token_idx] = torch.randperm(expert_count, generator=generator)[:top_k]
        return ids

    pool_size = min(expert_count, max(top_k, m * top_k))
    layer_stride = pool_size if pattern == "disjoint" else max(top_k, pool_size // 2)
    pool_start = (seed * 17 + layer_idx * layer_stride) % expert_count
    pool = (torch.arange(pool_size, dtype=torch.int64) + pool_start) % expert_count
    pool = pool[torch.randperm(pool_size, generator=generator)]

    ids = torch.empty(m, top_k, dtype=torch.int64)
    cursor = 0
    for token_idx in range(m):
        if cursor + top_k > pool_size:
            pool = pool[torch.randperm(pool_size, generator=generator)]
            cursor = 0
        ids[token_idx] = pool[cursor:cursor + top_k]
        cursor += top_k
    return ids


def make_multilayer_routing_case(
    spec: ModelSpec,
    m: int,
    num_layers: int,
    device: torch.device,
    *,
    pattern: str,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    routing_case: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(num_layers):
        topk_ids = _make_structured_routing_ids(
            spec,
            m,
            layer_idx=layer_idx,
            pattern=pattern,
            seed=seed,
        ).to(device=device)
        weight_generator = torch.Generator(device="cpu")
        weight_generator.manual_seed(seed + layer_idx * 1009 + 7)
        topk_logits = torch.randn(m, spec.top_k, generator=weight_generator, dtype=torch.float32)
        topk_weights = torch.softmax(topk_logits, dim=-1).to(device=device)
        routing_case.append((topk_ids, topk_weights))
    return routing_case


def get_scale_contract_params(weights: ExpertWeights, scale_contract: str) -> ScaleContractParams:
    if scale_contract == "per-expert":
        return ScaleContractParams(
            a1_gscale=weights.w13_input_scale_per_expert,
            a2_gscale=weights.w2_input_scale_per_expert,
            g1_alphas=weights.g1_alphas_per_expert,
            g2_alphas=weights.g2_alphas_per_expert,
        )
    if scale_contract == "shared":
        return ScaleContractParams(
            a1_gscale=weights.w13_input_scale,
            a2_gscale=weights.w2_input_scale,
            g1_alphas=weights.g1_alphas,
            g2_alphas=weights.g2_alphas,
        )
    raise ValueError(f"Unsupported scale contract: {scale_contract}")


def bench_flashinfer(
    weights: ExpertWeights,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[Callable[[], None], torch.Tensor]:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe

    output = torch.empty(x.shape[0], weights.spec.hidden_size, dtype=torch.bfloat16, device=x.device)
    quant_scales = [
        weights.w13_input_scale_quant,
        weights.w13_blockscale_swizzled.view(torch.int32),
        weights.g1_alphas,
        weights.w2_input_scale_quant,
        weights.w2_blockscale_swizzled.view(torch.int32),
        weights.g2_alphas,
    ]

    def launch() -> None:
        flashinfer_cutlass_fused_moe(
            output=output,
            input=x,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=weights.w13_weight.view(torch.long),
            fc2_expert_weights=weights.w2_weight.view(torch.long),
            output_dtype=torch.bfloat16,
            quant_scales=quant_scales,
            input_sf=None,
            tp_size=TP_SIZE,
            tp_rank=TP_RANK,
            ep_size=EP_SIZE,
            ep_rank=EP_RANK,
            tune_max_num_tokens=max(16, x.shape[0]),
        )

    return launch, output


def make_oracle_reference(
    oracle_mode: str,
    x: torch.Tensor,
    weights: ExpertWeights,
    params: ScaleContractParams,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    oracle_fn = moe_reference_nvfp4 if oracle_mode == "nvfp4" else moe_reference_f32
    spec = weights.spec
    return oracle_fn(
        x,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        params.g1_alphas,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        params.g2_alphas,
        params.a1_gscale,
        params.a2_gscale,
        topk_ids,
        topk_weights,
        spec.num_experts,
        spec.hidden_size,
        spec.I_tp,
    )


ORACLE_TOLERANCES = {
    "max_abs": 0.0005,
    "rmse": 0.0001,
    "mean_abs": 0.0001,
    "cos_min": 0.99925,
}


def format_oracle_metrics(name: str, metrics: OracleMetrics) -> str:
    return (
        f"{name}: max_abs={metrics.max_abs:.5f} "
        f"rmse={metrics.rmse:.5f} "
        f"mean_abs={metrics.mean_abs:.5f} "
        f"cos={metrics.cos:.6f}"
    )


def check_oracle_metrics(
    label: str, metrics: OracleMetrics, batch_size: int
) -> list[str]:
    failures = []
    tol = ORACLE_TOLERANCES
    if metrics.max_abs > tol["max_abs"]:
        failures.append(f"  bs={batch_size} {label}: max_abs={metrics.max_abs:.5f} > {tol['max_abs']}")
    if metrics.rmse > tol["rmse"]:
        failures.append(f"  bs={batch_size} {label}: rmse={metrics.rmse:.5f} > {tol['rmse']}")
    if metrics.mean_abs > tol["mean_abs"]:
        failures.append(f"  bs={batch_size} {label}: mean_abs={metrics.mean_abs:.5f} > {tol['mean_abs']}")
    if metrics.cos < tol["cos_min"]:
        failures.append(f"  bs={batch_size} {label}: cos={metrics.cos:.6f} < {tol['cos_min']}")
    return failures


def _clear_b12x_caches() -> None:
    from b12x.integration.tp_moe import clear_tp_moe_caches

    clear_tp_moe_caches()


GRAPH_REPLAY_TOLERANCES = {
    "max_abs": 5e-4,
    "rmse": 1e-4,
    "mean_abs": 1e-4,
    "cos_min": 0.9999,
}


def bench_repeated(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> TimingStats:
    return summarize_timing_runs(
        [bench_events(fn, warmup=warmup, iters=iters) for _ in range(repeats)]
    )


def compare_graph_replay_outputs(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> OracleMetrics:
    metrics = compare_to_reference(actual, reference)
    actual_norm = actual.float().norm().item()
    reference_norm = reference.float().norm().item()
    if max(actual_norm, reference_norm) <= 1e-8:
        return OracleMetrics(
            max_abs=metrics.max_abs,
            rmse=metrics.rmse,
            mean_abs=metrics.mean_abs,
            cos=1.0,
        )
    return metrics


def allocate_layer_chain_workspace(
    weights_stack: Sequence[ExpertWeights],
    params_stack: Sequence[ScaleContractParams],
    x: torch.Tensor,
    topk_ids_per_layer: Sequence[torch.Tensor],
):
    from b12x.integration.tp_moe import allocate_tp_moe_workspace

    if not weights_stack:
        raise ValueError("weights_stack must not be empty")
    return allocate_tp_moe_workspace(
        x,
        params_stack[0].a1_gscale,
        weights_stack[0].w13_weight,
        params_stack[0].a2_gscale,
        weights_stack[0].w2_weight,
        topk_ids_per_layer[0],
        input_scales_static=True,
    )


def run_moe_layer_chain(
    weights_stack: Sequence[ExpertWeights],
    params_stack: Sequence[ScaleContractParams],
    x: torch.Tensor,
    topk_ids_per_layer: Sequence[torch.Tensor],
    topk_weights_per_layer: Sequence[torch.Tensor],
    *,
    fast_math: bool,
    fc2_tile_amax: bool = False,
    output_buffers: Sequence[torch.Tensor] | None = None,
    workspace,
) -> list[torch.Tensor]:
    from b12x.integration.tp_moe import b12x_moe_fp4

    if not (
        len(weights_stack)
        == len(params_stack)
        == len(topk_ids_per_layer)
        == len(topk_weights_per_layer)
    ):
        raise ValueError("layer-chain inputs must all have the same length")
    if output_buffers is not None and len(output_buffers) != len(weights_stack):
        raise ValueError("output_buffers must match the number of layers")

    layer_outputs: list[torch.Tensor] = []
    current = x
    for layer_idx, (weights, params, topk_ids, topk_weights) in enumerate(
        zip(weights_stack, params_stack, topk_ids_per_layer, topk_weights_per_layer, strict=True)
    ):
        output = None if output_buffers is None else output_buffers[layer_idx]
        current = b12x_moe_fp4(
            current,
            params.a1_gscale,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            params.g1_alphas,
            params.a2_gscale,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            params.g2_alphas,
            topk_weights,
            topk_ids,
            fast_math=fast_math,
            fc2_tile_amax=fc2_tile_amax,
            output=output,
            workspace=workspace,
            input_scales_static=True,
        )
        layer_outputs.append(current)
    return layer_outputs


def capture_moe_layer_chain(
    weights_stack: Sequence[ExpertWeights],
    params_stack: Sequence[ScaleContractParams],
    x: torch.Tensor,
    topk_ids_per_layer: Sequence[torch.Tensor],
    topk_weights_per_layer: Sequence[torch.Tensor],
    *,
    fast_math: bool,
    fc2_tile_amax: bool = False,
    output_buffers: Sequence[torch.Tensor],
    workspace,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_moe_layer_chain(
            weights_stack,
            params_stack,
            x,
            topk_ids_per_layer,
            topk_weights_per_layer,
            fast_math=fast_math,
            fc2_tile_amax=fc2_tile_amax,
            output_buffers=output_buffers,
            workspace=workspace,
        )
    return graph


def _check_graph_replay_metrics(
    label: str,
    metrics: OracleMetrics,
) -> list[str]:
    failures = []
    tol = GRAPH_REPLAY_TOLERANCES
    if metrics.max_abs > tol["max_abs"]:
        failures.append(f"{label}: max_abs={metrics.max_abs:.6f} > {tol['max_abs']}")
    if metrics.rmse > tol["rmse"]:
        failures.append(f"{label}: rmse={metrics.rmse:.6f} > {tol['rmse']}")
    if metrics.mean_abs > tol["mean_abs"]:
        failures.append(f"{label}: mean_abs={metrics.mean_abs:.6f} > {tol['mean_abs']}")
    if metrics.cos < tol["cos_min"]:
        failures.append(f"{label}: cos={metrics.cos:.6f} < {tol['cos_min']}")
    return failures


def bench_multilayer_graph_mode(
    args,
    spec: ModelSpec,
    batch_sizes: Sequence[int],
    device: torch.device,
) -> None:
    graph_num_layers = args.graph_num_layers
    if graph_num_layers < 2:
        raise ValueError("--graph-num-layers must be at least 2 in multi-layer graph mode")

    cfg = json.loads((MODEL_PATH / "config.json").read_text())["text_config"]
    total_layers = cfg["num_hidden_layers"]
    layer_start = args.graph_layer_start
    if layer_start < 0 or layer_start + graph_num_layers > total_layers:
        raise ValueError(
            f"requested layers [{layer_start}, {layer_start + graph_num_layers}) exceed model depth {total_layers}"
        )

    if args.reference != "none" or args.validate != "none":
        print("Note: multi-layer graph mode skips flashinfer/oracle checks and validates graph replay against an eager layer chain.")
    print("Multi-layer graph mode")
    print("Backend: b12x auto")
    print(f"Layers: {layer_start}..{layer_start + graph_num_layers - 1}")
    print(f"Patterns: disjoint, overlap, random")
    print()

    _clear_b12x_caches()
    weights_stack = load_expert_weight_stack(
        MODEL_PATH,
        spec,
        layer_start=layer_start,
        num_layers=graph_num_layers,
    )
    params_stack = [get_scale_contract_params(weights, args.scale_contract) for weights in weights_stack]

    scenario_specs = [
        ("disjoint", "disjoint", 1100),
        ("overlap", "overlap", 2200),
        ("random-a", "random", 3300),
        ("random-b", "random", 4400),
    ]
    validation_failures: list[str] = []

    for batch_size in batch_sizes:
        print(f"\n{'=' * 70}")
        print(
            f"  batch_size={batch_size}  "
            f"(layers={graph_num_layers}, tokens*top_k={batch_size * spec.top_k})"
        )
        print(f"{'=' * 70}")

        x_buf = make_input_activations(spec, batch_size, 10_000 + batch_size, device)
        initial_case = make_multilayer_routing_case(
            spec,
            batch_size,
            graph_num_layers,
            device,
            pattern="disjoint",
            seed=20_000 + batch_size,
        )
        topk_ids_bufs = [topk_ids.clone() for topk_ids, _ in initial_case]
        topk_weights_bufs = [topk_weights.clone() for _, topk_weights in initial_case]
        graph_output_bufs = [torch.empty_like(x_buf) for _ in range(graph_num_layers)]
        eager_output_bufs = [torch.empty_like(x_buf) for _ in range(graph_num_layers)]
        shared_workspace = allocate_layer_chain_workspace(
            weights_stack,
            params_stack,
            x_buf,
            topk_ids_bufs,
        )

        run_moe_layer_chain(
            weights_stack,
            params_stack,
            x_buf,
            topk_ids_bufs,
            topk_weights_bufs,
            fast_math=args.fast_math,
            fc2_tile_amax=args.fc2_tile_amax,
            output_buffers=graph_output_bufs,
            workspace=shared_workspace,
        )
        torch.cuda.synchronize()
        graph = capture_moe_layer_chain(
            weights_stack,
            params_stack,
            x_buf,
            topk_ids_bufs,
            topk_weights_bufs,
            fast_math=args.fast_math,
            fc2_tile_amax=args.fc2_tile_amax,
            output_buffers=graph_output_bufs,
            workspace=shared_workspace,
        )

        def eager_chain() -> None:
            run_moe_layer_chain(
                weights_stack,
                params_stack,
                x_buf,
                topk_ids_bufs,
                topk_weights_bufs,
                fast_math=args.fast_math,
                fc2_tile_amax=args.fc2_tile_amax,
                output_buffers=eager_output_bufs,
                workspace=shared_workspace,
            )

        for scenario_name, pattern, seed in scenario_specs:
            x_case = make_input_activations(
                spec,
                batch_size,
                30_000 + batch_size + seed,
                device,
            )
            routing_case = make_multilayer_routing_case(
                spec,
                batch_size,
                graph_num_layers,
                device,
                pattern=pattern,
                seed=40_000 + batch_size + seed,
            )

            x_buf.copy_(x_case)
            for layer_idx, (topk_ids, topk_weights) in enumerate(routing_case):
                topk_ids_bufs[layer_idx].copy_(topk_ids)
                topk_weights_bufs[layer_idx].copy_(topk_weights)

            graph.replay()
            torch.cuda.synchronize()
            graph_outputs = [buf.clone() for buf in graph_output_bufs]

            eager_chain()
            torch.cuda.synchronize()
            eager_outputs = [buf.clone() for buf in eager_output_bufs]

            final_metrics = compare_graph_replay_outputs(graph_outputs[-1], eager_outputs[-1])
            layer_metrics = [
                compare_graph_replay_outputs(graph_out, eager_out)
                for graph_out, eager_out in zip(graph_outputs, eager_outputs, strict=True)
            ]
            graph_stats = bench_repeated(
                graph.replay,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
            eager_stats = bench_repeated(
                eager_chain,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
            ratio_stats = RatioStats([graph_stats.median_ms / eager_stats.median_ms])

            print(
                f"  {scenario_name}: "
                f"graph {fmt_timing_stats(graph_stats)} | "
                f"eager {fmt_timing_stats(eager_stats)} | "
                f"ratio {fmt_ratio_stats(ratio_stats)}"
            )
            print(
                "    final:",
                format_oracle_metrics("graph vs eager", final_metrics),
            )
            for layer_idx, metrics in enumerate(layer_metrics):
                print(
                    f"    layer {layer_idx + layer_start}: "
                    f"max_abs={metrics.max_abs:.6f} rmse={metrics.rmse:.6f} cos={metrics.cos:.6f}"
                )
                validation_failures.extend(
                    _check_graph_replay_metrics(
                        f"bs={batch_size} {scenario_name} layer={layer_idx + layer_start}",
                        metrics,
                    )
                )
            validation_failures.extend(
                _check_graph_replay_metrics(
                    f"bs={batch_size} {scenario_name} final",
                    final_metrics,
                )
            )

    if validation_failures:
        print(f"\n\033[1;31m{'=' * 70}")
        print("  MULTI-LAYER GRAPH VALIDATION FAILED")
        print(f"{'=' * 70}")
        for failure in validation_failures:
            print(f"  {failure}")
        print(f"{'=' * 70}\033[0m")
        sys.exit(1)


def bench_e2e() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repeat the timed measurement this many times per batch size and aggregate the results.",
    )
    parser.add_argument("--batch-size-profile", choices=sorted(BATCH_SIZE_PROFILES), default="micro")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--graph-mode", choices=["single-op", "multi-layer"], default="single-op")
    parser.add_argument("--graph-num-layers", type=int, default=4)
    parser.add_argument("--graph-layer-start", type=int, default=0)
    parser.add_argument("--reference", choices=["flashinfer", "none"], default="flashinfer")
    parser.add_argument("--scale-contract", choices=["shared", "per-expert"], default="shared")
    parser.add_argument("--validate", choices=["none", "oracle"], default="oracle")
    parser.add_argument("--oracle-mode", choices=["nvfp4", "f32"], default="nvfp4")
    parser.add_argument("--include-routing", action="store_true")
    parser.set_defaults(cuda_graph=True)
    parser.add_argument(
        "--cuda-graph",
        dest="cuda_graph",
        action="store_true",
        help="Benchmark CUDA graph replay timings (default: enabled).",
    )
    parser.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="Disable CUDA graph capture/replay timing and use eager timings in the summary.",
    )
    parser.add_argument(
        "--profile-once",
        choices=["none", "backend", "flashinfer"],
        default="none",
    )
    parser.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fc2-tile-amax",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable compact static/micro FC2 tile-amax quantization.",
    )
    args = parser.parse_args()
    batch_sizes = (
        args.batch_sizes
        if args.batch_sizes is not None
        else BATCH_SIZE_PROFILES[args.batch_size_profile]
    )

    if args.scale_contract == "per-expert" and args.reference == "flashinfer":
        raise ValueError("--reference flashinfer is only valid with --scale-contract shared")

    require_sm120()
    torch.empty(1, device="cuda")
    device = torch.device("cuda")

    spec = ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )

    benchmark_scope = "Routing + MoE kernel" if args.include_routing else "Pre-routed MoE kernel only"
    print(f"MoE benchmark ({benchmark_scope})")
    print(
        f"Qwen3.5-397B  TP={spec.tp_size}, K={spec.hidden_size}, I_tp={spec.I_tp}, "
        f"E={spec.num_experts}, top_k={spec.top_k}"
    )
    print(f"Batch-size profile: {args.batch_size_profile} -> {batch_sizes}")
    print("Backend: b12x auto")
    print(f"Scale contract: {args.scale_contract}")
    print(f"Validation: {args.validate}")
    print(f"Fast math: {'on' if args.fast_math else 'off'}")
    print(f"FC2 tile amax: {'on' if args.fc2_tile_amax else 'off'}")
    print(f"Graph mode: {args.graph_mode}")
    print(f"Timing passes per batch size: {args.repeats} x {args.iters} iterations")
    print(
        "Timed region: "
        + ("top-k + softmax routing + backend launch" if args.include_routing else "backend launch only")
    )
    if args.validate == "oracle":
        print(f"Oracle mode: {args.oracle_mode}")
    print()

    if args.graph_mode == "multi-layer":
        bench_multilayer_graph_mode(args, spec, batch_sizes, device)
        return

    weights = load_expert_weights(MODEL_PATH, spec)
    params = get_scale_contract_params(weights, args.scale_contract)

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        allocate_tp_moe_workspace_pool,
        b12x_moe_fp4,
    )

    _clear_b12x_caches()

    print("  Warming up b12x (compilation)...", end="", flush=True)
    torch.manual_seed(42)
    x_warm = torch.randn(1, spec.hidden_size, dtype=torch.bfloat16, device=device)
    routing_warm = torch.randn(1, spec.num_experts, dtype=torch.float32, device=device)
    topk_logits_w, topk_ids_w = torch.topk(routing_warm, spec.top_k, dim=-1)
    topk_weights_w = torch.softmax(topk_logits_w, dim=-1)
    warmup_workspace = allocate_tp_moe_workspace_pool()
    b12x_moe_fp4(
        x_warm,
        params.a1_gscale,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        params.g1_alphas,
        params.a2_gscale,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        params.g2_alphas,
        topk_weights_w,
        topk_ids_w,
        workspace=warmup_workspace,
        fast_math=args.fast_math,
        fc2_tile_amax=args.fc2_tile_amax,
    )
    torch.cuda.synchronize()
    print(" done.")

    backend_label = "b12x"

    batch_results: dict[int, BatchResult] = {}
    accuracy_failures: list[str] = []
    reference_warnings: list[str] = []
    for batch_size in batch_sizes:
        print(f"\n{'=' * 70}")
        print(f"  batch_size={batch_size}  (tokens*top_k = {batch_size * spec.top_k} expert calls)")
        print(f"{'=' * 70}")

        torch.manual_seed(42 + batch_size)
        x = torch.randn(batch_size, spec.hidden_size, dtype=torch.bfloat16, device=device)
        routing_logits = torch.randn(batch_size, spec.num_experts, dtype=torch.float32, device=device)
        topk_logits, topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)
        backend_output = torch.empty_like(x)
        backend_workspace = allocate_tp_moe_workspace_pool()

        def make_backend_e2e() -> Callable[[], torch.Tensor]:

            def impl_launch(topk_ids_local: torch.Tensor, topk_weights_local: torch.Tensor) -> torch.Tensor:
                return b12x_moe_fp4(
                    x,
                    params.a1_gscale,
                    weights.w13_weight,
                    weights.w13_blockscale_swizzled,
                    params.g1_alphas,
                    params.a2_gscale,
                    weights.w2_weight,
                    weights.w2_blockscale_swizzled,
                    params.g2_alphas,
                    topk_weights_local,
                    topk_ids_local,
                    workspace=backend_workspace,
                    fast_math=args.fast_math,
                    fc2_tile_amax=args.fc2_tile_amax,
                    output=backend_output,
                )

            def impl_e2e() -> torch.Tensor:
                if args.include_routing:
                    timed_topk_logits, timed_topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
                    timed_topk_weights = torch.softmax(timed_topk_logits, dim=-1)
                    return impl_launch(timed_topk_ids, timed_topk_weights)
                return impl_launch(topk_ids, topk_weights)

            return impl_e2e

        backend_e2e = make_backend_e2e()

        ref_name = None
        ref_launch = None
        fi_output = None
        if args.reference == "flashinfer":
            ref_name = "FlashInfer"
            base_ref_launch, fi_output = bench_flashinfer(weights, x, topk_ids, topk_weights)
            fi_quant_scales = [
                weights.w13_input_scale_quant,
                weights.w13_blockscale_swizzled.view(torch.int32),
                weights.g1_alphas,
                weights.w2_input_scale_quant,
                weights.w2_blockscale_swizzled.view(torch.int32),
                weights.g2_alphas,
            ]

            if args.include_routing:
                def ref_launch() -> None:
                    timed_topk_logits, timed_topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
                    timed_topk_weights = torch.softmax(timed_topk_logits, dim=-1)
                    flashinfer_cutlass_fused_moe(
                        output=fi_output,
                        input=x,
                        token_selected_experts=timed_topk_ids.to(torch.int),
                        token_final_scales=timed_topk_weights,
                        fc1_expert_weights=weights.w13_weight.view(torch.long),
                        fc2_expert_weights=weights.w2_weight.view(torch.long),
                        output_dtype=torch.bfloat16,
                        quant_scales=fi_quant_scales,
                        input_sf=None,
                        tp_size=TP_SIZE,
                        tp_rank=TP_RANK,
                        ep_size=EP_SIZE,
                        ep_rank=EP_RANK,
                        tune_max_num_tokens=max(16, x.shape[0]),
                    )
            else:
                ref_launch = base_ref_launch

        oracle_ref = None
        if args.validate == "oracle":
            oracle_ref = make_oracle_reference(args.oracle_mode, x, weights, params, topk_ids, topk_weights)
            print(
                "  oracle:".ljust(28),
                f"norm={oracle_ref.float().norm().item():.5f}",
                f"max={oracle_ref.float().abs().max().item():.5f}",
            )

        ref_output = None
        if ref_launch is not None:
            ref_launch()
            torch.cuda.synchronize()
            ref_output = fi_output.clone()

        backend_out = backend_e2e().clone()
        torch.cuda.synchronize()

        if ref_output is not None:
            diff = (backend_out.float() - ref_output.float()).abs()
            print(
                f"  check vs {ref_name}: max_abs={diff.max().item():.5f} "
                f"rmse={diff.square().mean().sqrt().item():.5f}"
            )
            if oracle_ref is not None:
                backend_metrics = compare_to_reference(backend_out, oracle_ref)
                print(f"  {format_oracle_metrics(f'{backend_label} vs oracle', backend_metrics)}")
                accuracy_failures.extend(
                    check_oracle_metrics(f"{backend_label} vs oracle", backend_metrics, batch_size)
                )
                if fi_output is not None:
                    fi_metrics = compare_to_reference(fi_output, oracle_ref)
                    print(f"  {format_oracle_metrics('flashinfer vs oracle', fi_metrics)}")
                    reference_warnings.extend(
                        check_oracle_metrics("flashinfer vs oracle", fi_metrics, batch_size)
                    )

        if args.profile_once != "none":
            if args.profile_once == "flashinfer":
                if ref_launch is None:
                    raise ValueError("--profile-once flashinfer requires --reference flashinfer")
                profile_fn = ref_launch
                profile_name = ref_name or "flashinfer"
            else:
                profile_fn = backend_e2e
                profile_name = backend_label
            print(f"  profiling once: {profile_name}")
            torch.cuda.synchronize()
            cudart = torch.cuda.cudart()
            cudart.cudaProfilerStart()
            profile_fn()
            torch.cuda.synchronize()
            cudart.cudaProfilerStop()
            print("  profiler range complete")
            return

        ref_runs_ms: list[list[float]] = []
        backend_runs_ms: list[list[float]] = []
        ratio_runs: list[float] = []
        for _ in range(args.repeats):
            ref_run = None
            if ref_launch is not None:
                ref_run = bench_events(ref_launch, warmup=args.warmup, iters=args.iters)
                ref_runs_ms.append(ref_run)
            backend_run = bench_events(backend_e2e, warmup=args.warmup, iters=args.iters)
            backend_runs_ms.append(backend_run)
            if ref_run is not None:
                ratio_runs.append(statistics.median(backend_run) / statistics.median(ref_run))

        ref_stats = summarize_timing_runs(ref_runs_ms) if ref_runs_ms else None
        backend_stats = summarize_timing_runs(backend_runs_ms)
        ratio_nograph = RatioStats(ratio_runs) if ratio_runs else None

        if ref_stats is not None and ref_name is not None:
            print(f"  {ref_name} (no graph):".ljust(28), end="", flush=True)
            print(f" {fmt_timing_stats(ref_stats)}")

        print(f"  {backend_label} (no graph):".ljust(28), end="", flush=True)
        print(f" {fmt_timing_stats(backend_stats)}")
        if ratio_nograph is not None and ref_name is not None:
            print(f"    ratio vs {ref_name.lower()}:      {fmt_ratio_stats(ratio_nograph)}")

        if args.cuda_graph:
            graph_latencies: dict[str, float] = {}
            graph_launches = [(backend_label, backend_e2e)]
            if ref_launch is not None and ref_name is not None:
                graph_launches.insert(0, (ref_name, ref_launch))

            for name, fn in graph_launches:
                print(f"  {name} (CUDA graph):".ljust(28), end="", flush=True)
                try:
                    # Warm eager launch state before capture so compile/cache work
                    # does not leak into the replay measurement.
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fn()

                    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
                        g.replay()

                    # Warm graph replay separately; replay latency is the value
                    # that should drive the default summary.
                    graph_times = bench_events(replay, warmup=args.warmup, iters=args.iters)
                    graph_latencies[name] = statistics.median(graph_times)
                    print(f" {fmt_us(graph_times)}")
                except Exception as exc:
                    print(f" FAILED ({type(exc).__name__}: {exc})")

            if ref_name in graph_latencies and backend_label in graph_latencies:
                graph_ratio = graph_latencies[backend_label] / graph_latencies[ref_name]
                graph_ratio_stats = RatioStats([graph_ratio])
                print(f"    graph ratio vs {ref_name.lower()}: {fmt_ratio_stats(graph_ratio_stats)}")
                batch_results[batch_size] = BatchResult(
                    backend_stats=TimingStats([graph_latencies[backend_label]], [graph_latencies[backend_label]]),
                    ref_stats=TimingStats([graph_latencies[ref_name]], [graph_latencies[ref_name]]),
                    ratio_stats=graph_ratio_stats,
                )
            elif ratio_nograph is not None:
                batch_results[batch_size] = BatchResult(
                    backend_stats=backend_stats,
                    ref_stats=ref_stats,
                    ratio_stats=ratio_nograph,
                )
        elif ratio_nograph is not None or ref_stats is None:
            batch_results[batch_size] = BatchResult(
                backend_stats=backend_stats,
                ref_stats=ref_stats,
                ratio_stats=ratio_nograph,
            )

    ratio_results = {
        batch_size: result.ratio_stats.median
        for batch_size, result in batch_results.items()
        if result.ratio_stats is not None
    }
    if batch_results:
        print(f"\n{'=' * 70}")
        print("  Summary")
        print(f"{'=' * 70}")
        for batch_size in sorted(batch_results):
            result = batch_results[batch_size]
            parts = [f"bs={batch_size}"]
            if result.ref_stats is not None:
                parts.append(f"ref {result.ref_stats.median_us:.1f} us")
            parts.append(f"b12x {result.backend_stats.median_us:.1f} us")
            if result.ratio_stats is not None:
                parts.append(f"ratio {fmt_ratio_stats(result.ratio_stats)}")
            print("  " + " | ".join(parts))

        if ratio_results:
            geo = 1.0
            for ratio in ratio_results.values():
                geo *= ratio
            print(f"  geo mean: {geo ** (1.0 / len(ratio_results)):.2f}x")

    if accuracy_failures:
        print(f"\n\033[1;31m{'=' * 70}")
        print("  ACCURACY CHECK FAILED")
        print(f"{'=' * 70}")
        for f in accuracy_failures:
            print(f)
        print(f"{'=' * 70}\033[0m")
        sys.exit(1)
    if reference_warnings:
        print(f"\n\033[1;33m{'=' * 70}")
        print("  REFERENCE WARNING")
        print(f"{'=' * 70}")
        for f in reference_warnings:
            print(f)
        print(f"{'=' * 70}\033[0m")


def main() -> None:
    bench_e2e()


if __name__ == "__main__":
    main()
