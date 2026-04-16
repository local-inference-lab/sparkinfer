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
from b12x.moe.fused.nvfp4.reference import (
    OracleMetrics,
    compare_to_reference,
    moe_reference_f32,
    moe_reference_nvfp4,
)
from b12x.cute.fp4 import as_grouped_scale_view, swizzle_block_scale
from b12x.cute.utils import get_hardware_info


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
_L2_FLUSH_BUFFER_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_AUTO_L2_FLUSH_MULTIPLIER = 2
_FALLBACK_L2_FLUSH_BYTES = 32 << 20


def require_sm120() -> None:
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise RuntimeError(f"Requires sm_120 or sm_121, got sm_{major}{minor}")


def resolve_l2_flush_bytes(bytes_hint: int) -> int:
    if bytes_hint < 0:
        raise ValueError(f"l2 flush bytes must be non-negative, got {bytes_hint}")
    if bytes_hint > 0:
        return int(bytes_hint)
    try:
        l2_bytes = int(get_hardware_info().get_l2_cache_size_in_bytes())
    except Exception:
        l2_bytes = 0
    if l2_bytes > 0:
        return l2_bytes * _AUTO_L2_FLUSH_MULTIPLIER
    return _FALLBACK_L2_FLUSH_BYTES


def make_l2_flush_fn(
    *,
    enabled: bool,
    bytes_hint: int = 0,
) -> Callable[[], None] | None:
    if not enabled:
        return None
    flush_bytes = resolve_l2_flush_bytes(bytes_hint)
    device_idx = torch.cuda.current_device()
    key = (device_idx, flush_bytes)
    buffer = _L2_FLUSH_BUFFER_CACHE.get(key)
    if buffer is None:
        buffer = torch.empty(flush_bytes, dtype=torch.uint8, device=f"cuda:{device_idx}")
        _L2_FLUSH_BUFFER_CACHE[key] = buffer

    def flush(cache_buffer: torch.Tensor = buffer) -> None:
        cache_buffer.bitwise_not_()

    return flush


def bench_events(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    l2_flush: Callable[[], None] | None = None,
) -> list[float]:
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if l2_flush is not None:
            l2_flush()
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


@dataclass(frozen=True)
class ModelProfile:
    label: str
    checkpoint_family: str
    default_layer_idx: int
    tp_size: int
    hf_repo_id: str
    preferred_local_paths: tuple[str, ...] = ()


MODEL_PROFILES = {
    "qwen397b": ModelProfile(
        label="Qwen3.5-397B",
        checkpoint_family="qwen",
        default_layer_idx=0,
        tp_size=TP_SIZE,
        hf_repo_id="nvidia/Qwen3.5-397B-A17B-NVFP4",
        preferred_local_paths=(
            "/data/models/Qwen3.5-397B-A17B-NVFP4",
        ),
    ),
    "nemotron-backbone": ModelProfile(
        label="NVIDIA Nemotron Backbone",
        checkpoint_family="nemotron",
        default_layer_idx=1,
        tp_size=1,
        hf_repo_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        preferred_local_paths=(
            "/data/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        ),
    ),
}


def _cached_snapshot_path(repo_id: str) -> pathlib.Path | None:
    cache_root = pathlib.Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.is_dir():
        return None
    main_ref = cache_root / "refs" / "main"
    if main_ref.is_file():
        candidate = snapshots_root / main_ref.read_text().strip()
        if candidate.is_dir():
            return candidate
    snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
    if snapshots:
        return snapshots[-1]
    return None


def _default_legacy_model_path() -> pathlib.Path:
    local_qwen_path = pathlib.Path("/data/models/Qwen3.5-397B-A17B-NVFP4")
    if local_qwen_path.is_dir():
        return local_qwen_path
    cached_qwen_path = _cached_snapshot_path(MODEL_PROFILES["qwen397b"].hf_repo_id)
    if cached_qwen_path is not None:
        return cached_qwen_path
    return (
        pathlib.Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--nvidia--Qwen3.5-397B-A17B-NVFP4"
        / "snapshots"
        / "__missing__"
    )


def resolve_model_path(
    profile: ModelProfile,
    override: pathlib.Path | None,
) -> pathlib.Path:
    if override is not None:
        return override
    env_path = os.environ.get("B12X_MODEL_PATH")
    if env_path:
        return pathlib.Path(env_path)
    for candidate_str in profile.preferred_local_paths:
        candidate = pathlib.Path(candidate_str)
        if candidate.is_dir():
            return candidate
    cached_path = _cached_snapshot_path(profile.hf_repo_id)
    if cached_path is not None:
        return cached_path
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(repo_id=profile.hf_repo_id))

MODEL_PATH = _default_legacy_model_path()


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


def _load_config(model_path: pathlib.Path) -> dict:
    raw_cfg = json.loads((model_path / "config.json").read_text())
    return raw_cfg.get("text_config", raw_cfg)


def build_model_spec(model_path: pathlib.Path, profile: ModelProfile) -> ModelSpec:
    cfg = _load_config(model_path)
    if profile.checkpoint_family == "qwen":
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["num_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=profile.tp_size,
            tp_rank=0,
        )
    if profile.checkpoint_family == "nemotron":
        if cfg["hidden_size"] % TP_SIZE != 0:
            raise ValueError(
                f"expected hidden_size {cfg['hidden_size']} to be divisible by {TP_SIZE} for Nemotron local shard"
            )
        return ModelSpec(
            hidden_size=cfg["hidden_size"] // TP_SIZE,
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=profile.tp_size,
            tp_rank=0,
        )
    raise ValueError(f"unsupported checkpoint family {profile.checkpoint_family!r}")


def load_expert_weights(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_idx: int = 0,
    activation: str = "silu",
    checkpoint_family: str = "qwen",
) -> ExpertWeights:
    if activation not in {"silu", "relu2"}:
        raise ValueError(f"unsupported activation {activation!r}")

    cfg = _load_config(model_path)

    device = torch.device("cuda")
    E = spec.num_experts
    K = spec.hidden_size
    I_tp = spec.I_tp
    loader = IndexedSafetensorLoader(model_path)

    if checkpoint_family == "qwen":
        if activation != "silu":
            raise ValueError("Qwen FP4 benchmark only supports silu experts")
        assert cfg["num_experts"] == spec.num_experts
        assert cfg["moe_intermediate_size"] == spec.intermediate_size
        assert cfg["hidden_size"] == spec.hidden_size

        prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
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
            tp_off = spec.tp_rank * I_tp
            tp_off_packed = spec.tp_rank * (I_tp // 2)
            tp_sf_cols = I_tp // 16
            tp_sf_off = spec.tp_rank * tp_sf_cols

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
        w13_input_scale_per_expert = gate_is
        g1_alphas_per_expert = (gate_is * gate_gs).to(torch.float32)
    elif checkpoint_family == "nemotron":
        if activation != "relu2":
            raise ValueError("Nemotron backbone FP4 benchmark expects relu2 experts")
        assert cfg["n_routed_experts"] == spec.num_experts
        assert cfg["moe_intermediate_size"] == spec.intermediate_size
        assert cfg["hidden_size"] // TP_SIZE == spec.hidden_size

        prefix = f"backbone.layers.{layer_idx}.mixer.experts"
        up_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        down_w = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)

        up_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
        down_sf = torch.empty(E, K, I_tp // 16, dtype=torch.float8_e4m3fn, device=device)

        up_gs = torch.empty(E, dtype=torch.float32, device=device)
        down_gs = torch.empty(E, dtype=torch.float32, device=device)
        up_is = torch.empty(E, dtype=torch.float32, device=device)
        down_is = torch.empty(E, dtype=torch.float32, device=device)

        print(f"  Loading {E} experts...", end="", flush=True)
        for eid in range(E):
            ep = f"{prefix}.{eid}"
            up_w[eid] = loader.get_tensor(f"{ep}.up_proj.weight").to(device)
            up_sf[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale").to(device)
            up_gs[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale_2").to(device)
            up_is[eid] = loader.get_tensor(f"{ep}.up_proj.input_scale").to(device)

            down_w[eid] = loader.get_tensor(f"{ep}.down_proj.weight").to(device)
            down_sf[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale").to(device)
            down_gs[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale_2").to(device)
            down_is[eid] = loader.get_tensor(f"{ep}.down_proj.input_scale").to(device)
        print(" done.")

        w13_weight = up_w.contiguous()
        w13_sf = up_sf.contiguous()
        w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
        w2_weight = down_w.contiguous()
        w2_blockscale_swizzled = swizzle_block_scale(down_sf)

        w13_permuted = w13_weight.permute(1, 2, 0)
        w13_scale = as_grouped_scale_view(w13_blockscale_swizzled.view(torch.uint8), I_tp, K)
        down_permuted = w2_weight.permute(1, 2, 0)
        down_scale = as_grouped_scale_view(w2_blockscale_swizzled.view(torch.uint8), K, I_tp)

        w13_input_scale = up_is.max()
        w2_input_scale = down_is.max()
        g1_alphas = (w13_input_scale * up_gs).to(torch.float32)
        g2_alphas = (w2_input_scale * down_gs).to(torch.float32)
        w13_input_scale_per_expert = up_is
        g1_alphas_per_expert = (up_is * up_gs).to(torch.float32)
    else:
        raise ValueError(f"unsupported checkpoint family {checkpoint_family!r}")

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
        w13_input_scale_per_expert=w13_input_scale_per_expert,
        w2_input_scale_per_expert=down_is,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        g1_alphas_per_expert=g1_alphas_per_expert,
        g2_alphas_per_expert=g2_alphas_per_expert,
    )


def load_expert_weight_stack(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_start: int,
    num_layers: int,
    activation: str = "silu",
    checkpoint_family: str = "qwen",
) -> list[ExpertWeights]:
    return [
        load_expert_weights(
            model_path,
            spec,
            layer_idx=layer_start + layer_offset,
            activation=activation,
            checkpoint_family=checkpoint_family,
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
    cfg = _load_config(model_path)
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


def make_input_activations(
    spec: ModelSpec,
    m: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(m, spec.hidden_size, generator=generator, dtype=torch.float32)
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
            tp_size=weights.spec.tp_size,
            tp_rank=weights.spec.tp_rank,
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
    *,
    activation: str,
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
        activation=activation,
    )


ORACLE_TOLERANCES = {
    "silu": {
        "max_abs": 0.0005,
        "rmse": 0.0001,
        "mean_abs": 0.0001,
        "cos_min": 0.99925,
    },
    # relu2 outputs are ~1000x larger in magnitude than silu's, and the
    # activation's squaring step quadratically amplifies per-element noise.
    # Absolute thresholds don't transfer; cos is the correctness signal.
    "relu2": {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9915,
    },
}


def format_oracle_metrics(name: str, metrics: OracleMetrics) -> str:
    return (
        f"{name}: max_abs={metrics.max_abs:.5f} "
        f"rmse={metrics.rmse:.5f} "
        f"mean_abs={metrics.mean_abs:.5f} "
        f"cos={metrics.cos:.6f}"
    )


def check_oracle_metrics(
    label: str, metrics: OracleMetrics, batch_size: int, *, activation: str = "silu"
) -> list[str]:
    failures = []
    tol = ORACLE_TOLERANCES[activation]
    if tol["max_abs"] is not None and metrics.max_abs > tol["max_abs"]:
        failures.append(f"  bs={batch_size} {label}: max_abs={metrics.max_abs:.5f} > {tol['max_abs']}")
    if tol["rmse"] is not None and metrics.rmse > tol["rmse"]:
        failures.append(f"  bs={batch_size} {label}: rmse={metrics.rmse:.5f} > {tol['rmse']}")
    if tol["mean_abs"] is not None and metrics.mean_abs > tol["mean_abs"]:
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
    l2_flush: Callable[[], None] | None = None,
) -> TimingStats:
    return summarize_timing_runs(
        [bench_events(fn, warmup=warmup, iters=iters, l2_flush=l2_flush) for _ in range(repeats)]
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
    activation: str,
    fast_math: bool,
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
            output=output,
            workspace=workspace,
            input_scales_static=True,
            activation=activation,
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
    activation: str,
    fast_math: bool,
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
            activation=activation,
            fast_math=fast_math,
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
    model_path: pathlib.Path,
    profile: ModelProfile,
    spec: ModelSpec,
    batch_sizes: Sequence[int],
    device: torch.device,
) -> None:
    graph_num_layers = args.graph_num_layers
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    if graph_num_layers < 2:
        raise ValueError("--graph-num-layers must be at least 2 in multi-layer graph mode")

    cfg = _load_config(model_path)
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
        model_path,
        spec,
        layer_start=layer_start,
        num_layers=graph_num_layers,
        activation=args.activation,
        checkpoint_family=profile.checkpoint_family,
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
            activation=args.activation,
            fast_math=args.fast_math,
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
            activation=args.activation,
            fast_math=args.fast_math,
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
                activation=args.activation,
                fast_math=args.fast_math,
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
                l2_flush=l2_flush,
            )
            eager_stats = bench_repeated(
                eager_chain,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                l2_flush=l2_flush,
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
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="qwen397b")
    parser.add_argument("--model-path", type=pathlib.Path, default=None)
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument("--activation", choices=["silu", "relu2"], default="silu")
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
        "--graph-only",
        action="store_true",
        help="Skip eager timing and report CUDA graph replay only.",
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
    args = parser.parse_args()
    batch_sizes = (
        args.batch_sizes
        if args.batch_sizes is not None
        else BATCH_SIZE_PROFILES[args.batch_size_profile]
    )
    model_profile = MODEL_PROFILES[args.model_profile]
    model_path = resolve_model_path(model_profile, args.model_path)
    layer_idx = model_profile.default_layer_idx if args.layer_idx is None else args.layer_idx

    if args.scale_contract == "per-expert" and args.reference == "flashinfer":
        raise ValueError("--reference flashinfer is only valid with --scale-contract shared")
    if args.reference == "flashinfer" and args.activation != "silu":
        raise ValueError("--reference flashinfer is only valid with --activation silu")
    if args.graph_only and not args.cuda_graph:
        raise ValueError("--graph-only requires --cuda-graph")

    require_sm120()
    torch.empty(1, device="cuda")
    device = torch.device("cuda")
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0

    spec = build_model_spec(model_path, model_profile)

    benchmark_scope = "Routing + MoE kernel" if args.include_routing else "Pre-routed MoE kernel only"
    print(f"MoE benchmark ({benchmark_scope})")
    print(
        f"{model_profile.label}  TP={spec.tp_size}, K={spec.hidden_size}, I_tp={spec.I_tp}, "
        f"E={spec.num_experts}, top_k={spec.top_k}"
    )
    print(f"Model path: {model_path}")
    print(f"Layer: {layer_idx}")
    print(f"Activation: {args.activation}")
    print(f"Batch-size profile: {args.batch_size_profile} -> {batch_sizes}")
    print("Backend: b12x auto")
    print(f"Scale contract: {args.scale_contract}")
    print(f"Validation: {args.validate}")
    print(f"Fast math: {'on' if args.fast_math else 'off'}")
    if args.flush_l2:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")
    print(f"Graph mode: {args.graph_mode}")
    print(f"Graph only: {'yes' if args.graph_only else 'no'}")
    print(f"Timing passes per batch size: {args.repeats} x {args.iters} iterations")
    print(
        "Timed region: "
        + ("top-k + softmax routing + backend launch" if args.include_routing else "backend launch only")
    )
    if args.validate == "oracle":
        print(f"Oracle mode: {args.oracle_mode}")
    print()

    if args.graph_mode == "multi-layer":
        bench_multilayer_graph_mode(args, model_path, model_profile, spec, batch_sizes, device)
        return

    weights = load_expert_weights(
        model_path,
        spec,
        layer_idx=layer_idx,
        activation=args.activation,
        checkpoint_family=model_profile.checkpoint_family,
    )
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
        activation=args.activation,
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
                    output=backend_output,
                    activation=args.activation,
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
            from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe

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
                        tp_size=spec.tp_size,
                        tp_rank=spec.tp_rank,
                        ep_size=EP_SIZE,
                        ep_rank=EP_RANK,
                        tune_max_num_tokens=max(16, x.shape[0]),
                    )
            else:
                ref_launch = base_ref_launch

        oracle_ref = None
        if args.validate == "oracle":
            oracle_ref = make_oracle_reference(
                args.oracle_mode,
                x,
                weights,
                params,
                topk_ids,
                topk_weights,
                activation=args.activation,
            )
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
                check_oracle_metrics(
                    f"{backend_label} vs oracle", backend_metrics, batch_size,
                    activation=args.activation,
                )
            )
            if ref_output is not None and fi_output is not None:
                fi_metrics = compare_to_reference(fi_output, oracle_ref)
                print(f"  {format_oracle_metrics('flashinfer vs oracle', fi_metrics)}")
                reference_warnings.extend(
                    check_oracle_metrics(
                        "flashinfer vs oracle", fi_metrics, batch_size,
                        activation=args.activation,
                    )
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

        ref_stats = None
        backend_stats = None
        ratio_nograph = None
        if not args.graph_only:
            ref_runs_ms: list[list[float]] = []
            backend_runs_ms: list[list[float]] = []
            ratio_runs: list[float] = []
            for _ in range(args.repeats):
                ref_run = None
                if ref_launch is not None:
                    ref_run = bench_events(
                        ref_launch,
                        warmup=args.warmup,
                        iters=args.iters,
                        l2_flush=l2_flush,
                    )
                    ref_runs_ms.append(ref_run)
                backend_run = bench_events(
                    backend_e2e,
                    warmup=args.warmup,
                    iters=args.iters,
                    l2_flush=l2_flush,
                )
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
                    graph_times = bench_events(
                        replay,
                        warmup=args.warmup,
                        iters=args.iters,
                        l2_flush=l2_flush,
                    )
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
            elif backend_label in graph_latencies:
                ref_graph_stats = None
                if ref_name is not None and ref_name in graph_latencies:
                    ref_graph_stats = TimingStats([graph_latencies[ref_name]], [graph_latencies[ref_name]])
                batch_results[batch_size] = BatchResult(
                    backend_stats=TimingStats([graph_latencies[backend_label]], [graph_latencies[backend_label]]),
                    ref_stats=ref_graph_stats,
                    ratio_stats=None,
                )
            elif ratio_nograph is not None and backend_stats is not None:
                batch_results[batch_size] = BatchResult(
                    backend_stats=backend_stats,
                    ref_stats=ref_stats,
                    ratio_stats=ratio_nograph,
                )
        elif backend_stats is not None and (ratio_nograph is not None or ref_stats is None):
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
    backend_us_results = [
        result.backend_stats.median_us
        for result in batch_results.values()
        if result.ref_stats is None
    ]
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
        elif backend_us_results:
            print(f"  geo mean: {statistics.geometric_mean(backend_us_results):.1f} us")

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
