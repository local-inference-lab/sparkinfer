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
import torch.nn.functional as F

from benchmarks.checkpoint_loader import IndexedSafetensorLoader
from b12x.moe.fused.reference import (
    FlashInferTrtllmFP4E8M0K32Weights,
    OracleMetrics,
    compare_to_reference,
    moe_reference_f32,
    moe_reference_nvfp4,
    moe_reference_w4a16_fp4_e8m0_k32,
    moe_reference_w4a16_fp4_e8m0_k32_flashinfer_prepared,
    moe_reference_w4a16_f32,
    prepare_flashinfer_trtllm_fp4_e8m0_k32_weights,
)
from b12x.cute.fp4 import as_grouped_scale_view, swizzle_block_scale
from b12x.cute.utils import get_hardware_info
from tests.w4a16_reference import moe_reference_w4a16


LEGACY_BATCH_SIZES = [1, 2, 4, 8]
NANO35_MTP_BATCH_SIZES = [1, 4, 9]
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
    "nano35-mtp": NANO35_MTP_BATCH_SIZES,
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")


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
class ShapeSpec:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int


@dataclass(frozen=True)
class ModelProfile:
    label: str
    checkpoint_family: str
    default_layer_idx: int
    tp_size: int
    hf_repo_id: str | None
    default_model_path: pathlib.Path | None = None
    default_activation: str = "silu"
    default_quant_mode: str | None = None
    default_validate: str = "oracle"
    default_swiglu_limit: float | None = None
    default_routing: str = "synthetic"
    shape: ShapeSpec | None = None


MODEL_PROFILES = {
    "qwen397b": ModelProfile(
        label="Qwen3.5-397B",
        checkpoint_family="qwen",
        default_layer_idx=0,
        tp_size=TP_SIZE,
        hf_repo_id="nvidia/Qwen3.5-397B-A17B-NVFP4",
    ),
    "nemotron-backbone": ModelProfile(
        label="NVIDIA Nemotron Backbone",
        checkpoint_family="nemotron",
        default_layer_idx=1,
        tp_size=1,
        hf_repo_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    ),
    "nano35-w4a16": ModelProfile(
        label="NVIDIA Nano3.5 BF16 NVFP4 W4A16",
        checkpoint_family="nano35_w4a16",
        default_layer_idx=1,
        tp_size=1,
        hf_repo_id="nvidia/Nano3.5-BF16-NVFP4-W4A16-LMHEAD-CT",
        default_activation="relu2",
        default_quant_mode="w4a16",
        default_validate="oracle",
    ),
    "nano35-w4a16-shape": ModelProfile(
        label="NVIDIA Nano3.5 BF16 NVFP4 W4A16 (shape)",
        checkpoint_family="nano35_w4a16_shape",
        default_layer_idx=1,
        tp_size=1,
        hf_repo_id=None,
        default_activation="relu2",
        default_quant_mode="w4a16",
        default_validate="none",
        shape=ShapeSpec(
            hidden_size=2688,
            intermediate_size=1856,
            num_experts=128,
            top_k=6,
        ),
    ),
    "dsv4f": ModelProfile(
        label="DSV4F W4A16 (shape)",
        checkpoint_family="dsv4f_shape",
        default_layer_idx=0,
        tp_size=2,
        hf_repo_id=None,
        default_activation="silu",
        default_quant_mode="w4a16",
        default_validate="none",
        shape=ShapeSpec(
            hidden_size=6144,
            intermediate_size=2048,
            num_experts=256,
            top_k=8,
        ),
    ),
    "deepseek-v4-flash": ModelProfile(
        label="DeepSeek V4 Flash",
        checkpoint_family="deepseek_v4_flash",
        default_layer_idx=3,
        tp_size=4,
        hf_repo_id="deepseek-ai/DeepSeek-V4-Flash",
        default_activation="silu",
        default_quant_mode="w4a16",
        default_validate="oracle",
        default_swiglu_limit=10.0,
        default_routing="model",
    ),
    "glm51": ModelProfile(
        label="GLM-5.1",
        checkpoint_family="glm",
        default_layer_idx=3,
        tp_size=8,
        hf_repo_id=None,
        default_model_path=pathlib.Path("/data/models/GLM-5.1-NVFP4"),
    ),
    "minimax-m27": ModelProfile(
        label="MiniMax-M2.7",
        checkpoint_family="minimax_m2",
        default_layer_idx=0,
        tp_size=2,
        hf_repo_id=None,
        default_model_path=pathlib.Path("/data/models/MiniMax-M2.7-NVFP4"),
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
    if profile.default_model_path is not None and profile.default_model_path.is_dir():
        return profile.default_model_path
    if profile.hf_repo_id is not None:
        cached_path = _cached_snapshot_path(profile.hf_repo_id)
        if cached_path is not None:
            return cached_path
        from huggingface_hub import snapshot_download

        return pathlib.Path(snapshot_download(repo_id=profile.hf_repo_id))
    if profile.shape is not None:
        return pathlib.Path("<shape-only>")

    raise FileNotFoundError(
        f"no default path found for {profile.label}; pass --model-path explicitly"
    )

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
    w13_input_scale_quant_per_expert: torch.Tensor
    w2_input_scale_quant_per_expert: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor
    g1_alphas_per_expert: torch.Tensor
    g2_alphas_per_expert: torch.Tensor
    source_format: str = "modelopt_nvfp4"
    w4a16_w13_global_scale: torch.Tensor | None = None
    w4a16_w2_global_scale: torch.Tensor | None = None
    gate_weight: torch.Tensor | None = None
    gate_bias: torch.Tensor | None = None
    gate_tid2eid: torch.Tensor | None = None
    gate_score_func: str = "softmax"
    gate_route_scale: float = 1.0
    gate_norm_topk_prob: bool = True
    oracle_w13_weight: torch.Tensor | None = None
    oracle_w13_scale: torch.Tensor | None = None
    oracle_w2_weight: torch.Tensor | None = None
    oracle_w2_scale: torch.Tensor | None = None
    oracle_flashinfer_weights: FlashInferTrtllmFP4E8M0K32Weights | None = None
    w13_layout: str = "w31"


def _load_config(model_path: pathlib.Path) -> dict:
    raw_cfg = json.loads((model_path / "config.json").read_text())
    return raw_cfg.get("text_config", raw_cfg)


def _fp4_checkpoint_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.element_size() != 1:
        raise TypeError(f"expected one-byte packed FP4 tensor, got {tensor.dtype}")
    return tensor.view(torch.uint8)


def build_model_spec(model_path: pathlib.Path, profile: ModelProfile, *, tp_size_override: int | None = None, tp_rank: int = 0) -> ModelSpec:
    tp = tp_size_override if tp_size_override is not None else profile.tp_size
    if profile.shape is not None:
        return ModelSpec(
            hidden_size=profile.shape.hidden_size,
            intermediate_size=profile.shape.intermediate_size,
            num_experts=profile.shape.num_experts,
            top_k=profile.shape.top_k,
            tp_size=tp,
            tp_rank=tp_rank,
        )

    cfg = _load_config(model_path)
    if profile.checkpoint_family == "qwen":
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["num_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family == "glm":
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family == "minimax_m2":
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_experts=cfg["num_local_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family == "nano35_w4a16":
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family == "deepseek_v4_flash":
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family == "nemotron":
        if cfg["hidden_size"] % TP_SIZE != 0:
            raise ValueError(
                f"expected hidden_size {cfg['hidden_size']} to be divisible by {TP_SIZE} for Nemotron local shard"
            )
        # Nemotron expert tensors are stored for one hidden TP shard. The
        # benchmark --tp-size controls how the intermediate dimension is sliced,
        # not the hidden width loaded from each checkpoint shard.
        return ModelSpec(
            hidden_size=cfg["hidden_size"] // TP_SIZE,
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    raise ValueError(f"unsupported checkpoint family {profile.checkpoint_family!r}")


def make_shape_only_expert_weights(
    spec: ModelSpec,
    *,
    layer_idx: int,
    activation: str,
) -> ExpertWeights:
    if activation not in {"relu2", "silu"}:
        raise ValueError("shape-only W4A16 profile expects relu2 or silu experts")
    if spec.hidden_size % 16 != 0 or spec.I_tp % 16 != 0:
        raise ValueError(
            f"shape-only W4A16 profile requires K and I_tp divisible by 16, got K={spec.hidden_size}, I_tp={spec.I_tp}"
        )

    device = torch.device("cuda")
    E = spec.num_experts
    K = spec.hidden_size
    I_tp = spec.I_tp
    w13_rows = I_tp * (2 if activation == "silu" else 1)

    print(
        f"  Creating synthetic shape-only experts (E={E}, K={K}, I_tp={I_tp}, activation={activation})...",
        end="",
        flush=True,
    )
    w13_weight = torch.empty(E, w13_rows, K // 2, dtype=torch.uint8, device=device)
    w13_weight.fill_(0x11)
    w2_weight = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)
    w2_weight.fill_(0x11)

    w13_sf = torch.ones(E, w13_rows, K // 16, dtype=torch.float8_e4m3fn, device=device)
    down_sf = torch.ones(E, K, I_tp // 16, dtype=torch.float8_e4m3fn, device=device)
    w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
    w2_blockscale_swizzled = swizzle_block_scale(down_sf)

    w13_permuted = w13_weight.permute(1, 2, 0)
    w13_scale = as_grouped_scale_view(w13_blockscale_swizzled.view(torch.uint8), w13_rows, K)
    down_permuted = w2_weight.permute(1, 2, 0)
    down_scale = as_grouped_scale_view(w2_blockscale_swizzled.view(torch.uint8), K, I_tp)

    w13_input_scale_per_expert = torch.ones(E, dtype=torch.float32, device=device)
    w2_input_scale_per_expert = torch.ones(E, dtype=torch.float32, device=device)
    w13_input_scale = w13_input_scale_per_expert.max()
    w2_input_scale = w2_input_scale_per_expert.max()
    g1_alphas_per_expert = torch.ones(E, dtype=torch.float32, device=device)
    g2_alphas_per_expert = torch.ones(E, dtype=torch.float32, device=device)
    g1_alphas = g1_alphas_per_expert
    g2_alphas = g2_alphas_per_expert
    w13_input_scale_quant = (1.0 / w13_input_scale).to(torch.float32)
    w2_input_scale_quant = (1.0 / w2_input_scale).to(torch.float32)
    w13_input_scale_quant_per_expert = (1.0 / w13_input_scale_per_expert).to(torch.float32).contiguous()
    w2_input_scale_quant_per_expert = (1.0 / w2_input_scale_per_expert).to(torch.float32).contiguous()
    print(" done.")

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
        w2_input_scale_per_expert=w2_input_scale_per_expert,
        w13_input_scale_quant_per_expert=w13_input_scale_quant_per_expert,
        w2_input_scale_quant_per_expert=w2_input_scale_quant_per_expert,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        g1_alphas_per_expert=g1_alphas_per_expert,
        g2_alphas_per_expert=g2_alphas_per_expert,
    )


def load_expert_weights(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_idx: int = 0,
    activation: str = "silu",
    checkpoint_family: str = "qwen",
    keep_flashinfer_oracle_copy: bool = False,
) -> ExpertWeights:
    if activation not in {"silu", "relu2"}:
        raise ValueError(f"unsupported activation {activation!r}")

    device = torch.device("cuda")
    E = spec.num_experts
    K = spec.hidden_size
    I_tp = spec.I_tp
    source_format = "modelopt_nvfp4"
    w4a16_w13_global_scale = None
    w4a16_w2_global_scale = None
    gate_weight = None
    gate_bias = None
    gate_tid2eid = None
    gate_score_func = "softmax"
    gate_route_scale = 1.0
    gate_norm_topk_prob = True
    oracle_w13_weight = None
    oracle_w13_scale = None
    oracle_w2_weight = None
    oracle_w2_scale = None
    oracle_flashinfer_weights = None
    w13_layout = "w31"
    if checkpoint_family in {"nano35_w4a16_shape", "dsv4f_shape"}:
        return make_shape_only_expert_weights(spec, layer_idx=layer_idx, activation=activation)

    cfg = _load_config(model_path)
    loader = IndexedSafetensorLoader(model_path)

    if checkpoint_family in {"qwen", "glm", "minimax_m2"}:
        if checkpoint_family in {"glm", "minimax_m2"} and activation != "silu":
            raise ValueError(f"{checkpoint_family} FP4 benchmark only supports silu experts")
        if checkpoint_family == "qwen":
            cfg_num_experts = cfg["num_experts"]
            cfg_intermediate_size = cfg["moe_intermediate_size"]
            prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
            gate_proj = "gate_proj"
            up_proj = "up_proj"
            down_proj = "down_proj"
        elif checkpoint_family == "minimax_m2":
            cfg_num_experts = cfg["num_local_experts"]
            cfg_intermediate_size = cfg["intermediate_size"]
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts"
            gate_proj = "w1"
            up_proj = "w3"
            down_proj = "w2"
        else:
            cfg_num_experts = cfg["n_routed_experts"]
            cfg_intermediate_size = cfg["moe_intermediate_size"]
            prefix = f"model.layers.{layer_idx}.mlp.experts"
            gate_proj = "gate_proj"
            up_proj = "up_proj"
            down_proj = "down_proj"
        assert cfg_num_experts == spec.num_experts
        assert cfg_intermediate_size == spec.intermediate_size
        assert cfg["hidden_size"] == spec.hidden_size

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

            gate_w[eid] = loader.get_tensor(f"{ep}.{gate_proj}.weight").narrow(0, tp_off, I_tp).to(device)
            gate_sf[eid] = loader.get_tensor(f"{ep}.{gate_proj}.weight_scale").narrow(0, tp_off, I_tp).to(device)
            gate_gs[eid] = loader.get_tensor(f"{ep}.{gate_proj}.weight_scale_2").to(device)
            gate_is[eid] = loader.get_tensor(f"{ep}.{gate_proj}.input_scale").to(device)

            up_w[eid] = loader.get_tensor(f"{ep}.{up_proj}.weight").narrow(0, tp_off, I_tp).to(device)
            up_sf[eid] = loader.get_tensor(f"{ep}.{up_proj}.weight_scale").narrow(0, tp_off, I_tp).to(device)

            down_w[eid] = loader.get_tensor(f"{ep}.{down_proj}.weight").narrow(1, tp_off_packed, I_tp // 2).to(device)
            down_sf[eid] = loader.get_tensor(f"{ep}.{down_proj}.weight_scale").narrow(1, tp_sf_off, tp_sf_cols).to(device)
            down_gs[eid] = loader.get_tensor(f"{ep}.{down_proj}.weight_scale_2").to(device)
            down_is[eid] = loader.get_tensor(f"{ep}.{down_proj}.input_scale").to(device)
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
            tp_off = spec.tp_rank * I_tp
            tp_off_packed = spec.tp_rank * (I_tp // 2)
            tp_sf_cols = I_tp // 16
            tp_sf_off = spec.tp_rank * tp_sf_cols

            if spec.tp_size > 1:
                up_w[eid] = loader.get_tensor(f"{ep}.up_proj.weight").narrow(0, tp_off, I_tp).to(device)
                up_sf[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale").narrow(0, tp_off, I_tp).to(device)
                down_w[eid] = loader.get_tensor(f"{ep}.down_proj.weight").narrow(1, tp_off_packed, I_tp // 2).to(device)
                down_sf[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale").narrow(1, tp_sf_off, tp_sf_cols).to(device)
            else:
                up_w[eid] = loader.get_tensor(f"{ep}.up_proj.weight").to(device)
                up_sf[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale").to(device)
                down_w[eid] = loader.get_tensor(f"{ep}.down_proj.weight").to(device)
                down_sf[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale").to(device)
            up_gs[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale_2").to(device)
            up_is[eid] = loader.get_tensor(f"{ep}.up_proj.input_scale").to(device)
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
    elif checkpoint_family == "nano35_w4a16":
        if activation != "relu2":
            raise ValueError("Nano3.5 W4A16 benchmark expects relu2 experts")
        assert cfg["n_routed_experts"] == spec.num_experts
        assert cfg["moe_intermediate_size"] == spec.intermediate_size
        assert cfg["hidden_size"] == spec.hidden_size

        prefix = f"backbone.layers.{layer_idx}.mixer.experts"
        up_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        down_w = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)

        up_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
        down_sf = torch.empty(E, K, I_tp // 16, dtype=torch.float8_e4m3fn, device=device)

        up_weight_global_scale = torch.empty(E, dtype=torch.float32, device=device)
        down_weight_global_scale = torch.empty(E, dtype=torch.float32, device=device)

        print(f"  Loading {E} CT W4A16 experts...", end="", flush=True)
        for eid in range(E):
            ep = f"{prefix}.{eid}"
            tp_off = spec.tp_rank * I_tp
            tp_off_packed = spec.tp_rank * (I_tp // 2)
            tp_sf_cols = I_tp // 16
            tp_sf_off = spec.tp_rank * tp_sf_cols

            up_w[eid] = loader.get_tensor(f"{ep}.up_proj.weight_packed").narrow(0, tp_off, I_tp).to(device)
            up_sf[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale").narrow(0, tp_off, I_tp).to(device)
            up_weight_global_scale[eid] = loader.get_tensor(f"{ep}.up_proj.weight_global_scale").to(device)

            down_w[eid] = loader.get_tensor(f"{ep}.down_proj.weight_packed").narrow(1, tp_off_packed, I_tp // 2).to(device)
            down_sf[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale").narrow(1, tp_sf_off, tp_sf_cols).to(device)
            down_weight_global_scale[eid] = loader.get_tensor(f"{ep}.down_proj.weight_global_scale").to(device)
        print(" done.")

        # The compressed-tensors W4A16 checkpoint stores per-tensor global
        # scales as the inverse convention of the B12X W4A16 path.  Fold
        # 1 / weight_global_scale into the FP4 block scales and launch with
        # unit alphas so the benchmark exercises the same model contract.
        up_sf = (
            up_sf.float() * (1.0 / up_weight_global_scale).view(E, 1, 1)
        ).to(torch.float8_e4m3fn).contiguous()
        down_sf = (
            down_sf.float() * (1.0 / down_weight_global_scale).view(E, 1, 1)
        ).to(torch.float8_e4m3fn).contiguous()

        w13_weight = up_w.contiguous()
        w13_sf = up_sf
        w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
        w2_weight = down_w.contiguous()
        w2_blockscale_swizzled = swizzle_block_scale(down_sf)

        w13_permuted = w13_weight.permute(1, 2, 0)
        w13_scale = as_grouped_scale_view(w13_blockscale_swizzled.view(torch.uint8), I_tp, K)
        down_permuted = w2_weight.permute(1, 2, 0)
        down_scale = as_grouped_scale_view(w2_blockscale_swizzled.view(torch.uint8), K, I_tp)

        ones = torch.ones(E, dtype=torch.float32, device=device)
        w13_input_scale = torch.ones((), dtype=torch.float32, device=device)
        w2_input_scale = torch.ones((), dtype=torch.float32, device=device)
        w13_input_scale_per_expert = ones
        down_is = ones
        down_gs = ones
        g1_alphas = ones
        g2_alphas = ones
        g1_alphas_per_expert = ones
    elif checkpoint_family == "deepseek_v4_flash":
        if activation != "silu":
            raise ValueError("DeepSeek V4 Flash FP4 benchmark expects silu experts")
        if spec.hidden_size % 32 != 0 or spec.I_tp % 32 != 0:
            raise ValueError(
                f"DeepSeek V4 Flash W4A16 requires K and I_tp divisible by 32, "
                f"got K={spec.hidden_size}, I_tp={spec.I_tp}"
            )
        assert cfg["n_routed_experts"] == spec.num_experts
        assert cfg["moe_intermediate_size"] == spec.intermediate_size
        assert cfg["hidden_size"] == spec.hidden_size

        source_format = "fp4_e8m0_k32"
        w13_layout = "w31"
        prefix = f"layers.{layer_idx}.ffn.experts"
        gate_proj = "w1"
        up_proj = "w3"
        down_proj = "w2"
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            raise RuntimeError("DeepSeek V4 Flash FP4 scales require torch.float8_e8m0fnu")

        gate_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        up_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        down_w = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)

        gate_sf = torch.empty(E, I_tp, K // 32, dtype=e8m0_dtype, device=device)
        up_sf = torch.empty(E, I_tp, K // 32, dtype=e8m0_dtype, device=device)
        down_sf = torch.empty(E, K, I_tp // 32, dtype=e8m0_dtype, device=device)

        print(f"  Loading {E} DeepSeek V4 Flash FP4 experts...", end="", flush=True)
        for eid in range(E):
            ep = f"{prefix}.{eid}"
            tp_off = spec.tp_rank * I_tp
            tp_off_packed = spec.tp_rank * (I_tp // 2)
            tp_sf_cols = I_tp // 32
            tp_sf_off = spec.tp_rank * tp_sf_cols

            gate_w[eid] = _fp4_checkpoint_bytes(
                loader.get_tensor(f"{ep}.{gate_proj}.weight")
            ).narrow(0, tp_off, I_tp).to(device)
            gate_sf[eid] = loader.get_tensor(f"{ep}.{gate_proj}.scale").narrow(0, tp_off, I_tp).to(device)

            up_w[eid] = _fp4_checkpoint_bytes(
                loader.get_tensor(f"{ep}.{up_proj}.weight")
            ).narrow(0, tp_off, I_tp).to(device)
            up_sf[eid] = loader.get_tensor(f"{ep}.{up_proj}.scale").narrow(0, tp_off, I_tp).to(device)

            down_w[eid] = _fp4_checkpoint_bytes(
                loader.get_tensor(f"{ep}.{down_proj}.weight")
            ).narrow(1, tp_off_packed, I_tp // 2).to(device)
            down_sf[eid] = loader.get_tensor(f"{ep}.{down_proj}.scale").narrow(1, tp_sf_off, tp_sf_cols).to(device)
        print(" done.")

        # Match vLLM FusedMoE loading for the B12X backend: native DeepSeek V4
        # FP4 W13 source is contiguous [w1/gate, w3/up], and B12X receives it
        # without an additional row swap.
        w13_weight = torch.cat([gate_w, up_w], dim=1).contiguous()
        w13_sf = torch.cat([gate_sf, up_sf], dim=1).contiguous()
        if keep_flashinfer_oracle_copy:
            # Build the oracle copy in the exact FlashInfer/TRT-LLM structure
            # vLLM prepares from its independently loaded [w1, w3] source.
            oracle_w13_weight = w13_weight.clone()
            oracle_w13_scale = w13_sf.clone()
            oracle_w2_weight = down_w.contiguous().clone()
            oracle_w2_scale = down_sf.contiguous().clone()
            oracle_flashinfer_weights = prepare_flashinfer_trtllm_fp4_e8m0_k32_weights(
                oracle_w13_weight,
                oracle_w13_scale,
                oracle_w2_weight,
                oracle_w2_scale,
                K,
                I_tp,
                activation=activation,
                scale_byte_clamp=None,
            )
        w13_sf.view(torch.uint8).clamp_(max=247)
        down_sf.view(torch.uint8).clamp_(max=247)
        w13_blockscale_swizzled = w13_sf
        w2_weight = down_w.contiguous()
        w2_blockscale_swizzled = down_sf.contiguous()

        w13_permuted = w13_weight.permute(1, 2, 0)
        w13_scale = w13_sf
        down_permuted = w2_weight.permute(1, 2, 0)
        down_scale = down_sf

        ones = torch.ones(E, dtype=torch.float32, device=device)
        w13_input_scale = torch.ones((), dtype=torch.float32, device=device)
        w2_input_scale = torch.ones((), dtype=torch.float32, device=device)
        w13_input_scale_per_expert = ones
        down_is = ones
        down_gs = ones
        g1_alphas = ones
        g2_alphas = ones
        g1_alphas_per_expert = ones

        gate_prefix = f"layers.{layer_idx}.ffn.gate"
        gate_weight = loader.get_tensor(f"{gate_prefix}.weight").to(device=device).contiguous()
        bias_key = f"{gate_prefix}.bias"
        if bias_key in loader.weight_map:
            gate_bias = loader.get_tensor(bias_key).to(device=device).contiguous()
        tid2eid_key = f"{gate_prefix}.tid2eid"
        if tid2eid_key in loader.weight_map:
            gate_tid2eid = loader.get_tensor(tid2eid_key).to(device=device).contiguous()
        gate_score_func = str(cfg.get("scoring_func", "softmax")).lower()
        gate_route_scale = float(cfg.get("routed_scaling_factor", 1.0))
        gate_norm_topk_prob = bool(cfg.get("norm_topk_prob", True))
    else:
        raise ValueError(f"unsupported checkpoint family {checkpoint_family!r}")

    g2_alphas_per_expert = (down_is * down_gs).to(torch.float32)
    w13_input_scale_quant = (1.0 / w13_input_scale).to(torch.float32)
    w2_input_scale_quant = (1.0 / w2_input_scale).to(torch.float32)
    w13_input_scale_quant_per_expert = (1.0 / w13_input_scale_per_expert).to(torch.float32).contiguous()
    w2_input_scale_quant_per_expert = (1.0 / down_is).to(torch.float32).contiguous()

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
        w13_input_scale_quant_per_expert=w13_input_scale_quant_per_expert,
        w2_input_scale_quant_per_expert=w2_input_scale_quant_per_expert,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        g1_alphas_per_expert=g1_alphas_per_expert,
        g2_alphas_per_expert=g2_alphas_per_expert,
        source_format=source_format,
        w4a16_w13_global_scale=w4a16_w13_global_scale,
        w4a16_w2_global_scale=w4a16_w2_global_scale,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        gate_tid2eid=gate_tid2eid,
        gate_score_func=gate_score_func,
        gate_route_scale=gate_route_scale,
        gate_norm_topk_prob=gate_norm_topk_prob,
        oracle_w13_weight=oracle_w13_weight,
        oracle_w13_scale=oracle_w13_scale,
        oracle_w2_weight=oracle_w2_weight,
        oracle_w2_scale=oracle_w2_scale,
        oracle_flashinfer_weights=oracle_flashinfer_weights,
        w13_layout=w13_layout,
    )


def load_expert_weight_stack(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_start: int,
    num_layers: int,
    activation: str = "silu",
    checkpoint_family: str = "qwen",
    keep_flashinfer_oracle_copy: bool = False,
) -> list[ExpertWeights]:
    return [
        load_expert_weights(
            model_path,
            spec,
            layer_idx=layer_start + layer_offset,
            activation=activation,
            checkpoint_family=checkpoint_family,
            keep_flashinfer_oracle_copy=keep_flashinfer_oracle_copy,
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


def compute_model_gate_routing(
    weights: ExpertWeights,
    x: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weights.gate_weight is None:
        raise ValueError("model gate routing requires gate_weight")
    scores = F.linear(x.float(), weights.gate_weight.float())
    score_func = weights.gate_score_func
    if score_func == "softmax":
        original_scores = torch.softmax(scores, dim=-1)
    elif score_func == "sigmoid":
        original_scores = torch.sigmoid(scores)
    elif score_func == "sqrtsoftplus":
        original_scores = F.softplus(scores).sqrt()
    else:
        raise ValueError(f"unsupported model gate score function {score_func!r}")

    if weights.gate_tid2eid is not None:
        routing_generator = torch.Generator(device="cpu")
        routing_generator.manual_seed(seed + 17)
        input_ids = torch.randint(
            0,
            weights.gate_tid2eid.shape[0],
            (x.shape[0],),
            generator=routing_generator,
            dtype=torch.int64,
        ).to(device=x.device)
        topk_ids = weights.gate_tid2eid[input_ids].to(device=x.device)
    else:
        selection_scores = original_scores
        if weights.gate_bias is not None:
            selection_scores = selection_scores + weights.gate_bias.to(device=x.device)
        _topk_scores, topk_ids = torch.topk(selection_scores, weights.spec.top_k, dim=-1)

    topk_weights = original_scores.gather(1, topk_ids)
    if score_func != "softmax" and weights.gate_norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    topk_weights = topk_weights * weights.gate_route_scale
    return topk_ids, topk_weights


def make_profile_routed_inputs(
    profile: ModelProfile,
    weights: ExpertWeights,
    spec: ModelSpec,
    m: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = make_input_activations(spec, m, seed, device)
    if profile.default_routing == "model":
        topk_ids, topk_weights = compute_model_gate_routing(weights, x, seed=seed + 1)
        return x, topk_ids, topk_weights
    if profile.default_routing != "synthetic":
        raise ValueError(f"unsupported routing source {profile.default_routing!r}")
    _x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed, device)
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
            a1_gscale=weights.w13_input_scale_quant_per_expert,
            a2_gscale=weights.w2_input_scale_quant_per_expert,
            g1_alphas=weights.g1_alphas_per_expert,
            g2_alphas=weights.g2_alphas_per_expert,
        )
    if scale_contract == "shared":
        return ScaleContractParams(
            a1_gscale=weights.w13_input_scale_quant,
            a2_gscale=weights.w2_input_scale_quant,
            g1_alphas=weights.g1_alphas,
            g2_alphas=weights.g2_alphas,
        )
    raise ValueError(f"Unsupported scale contract: {scale_contract}")


def get_quant_mode_params(
    weights: ExpertWeights,
    scale_contract: str,
    quant_mode: str,
) -> ScaleContractParams:
    params = get_scale_contract_params(weights, scale_contract)
    quant_mode = quant_mode.lower()
    if quant_mode == "nvfp4":
        return params
    if quant_mode == "w4a16":
        # W4A16 keeps activations in BF16, so remove the activation input
        # scale factor from the fused alpha and leave only the weight global
        # scale component.
        return ScaleContractParams(
            a1_gscale=params.a1_gscale,
            a2_gscale=params.a2_gscale,
            g1_alphas=(params.g1_alphas * params.a1_gscale).to(torch.float32).contiguous(),
            g2_alphas=(params.g2_alphas * params.a2_gscale).to(torch.float32).contiguous(),
        )
    raise ValueError(f"Unsupported quant mode: {quant_mode}")


def get_w4a16_prepare_scales(
    weights: ExpertWeights,
    params: ScaleContractParams,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if weights.source_format == "compressed_tensors":
        if weights.w4a16_w13_global_scale is None or weights.w4a16_w2_global_scale is None:
            raise ValueError("compressed_tensors W4A16 weights require raw global scales")
        return (
            weights.w4a16_w13_global_scale,
            weights.w4a16_w2_global_scale,
            weights.source_format,
        )
    return params.g1_alphas, params.g2_alphas, weights.source_format


def get_w4a16_oracle_params(
    weights: ExpertWeights,
    params: ScaleContractParams,
) -> ScaleContractParams:
    if weights.source_format != "compressed_tensors":
        return params
    if weights.w4a16_w13_global_scale is None or weights.w4a16_w2_global_scale is None:
        raise ValueError("compressed_tensors W4A16 weights require raw global scales")
    return ScaleContractParams(
        a1_gscale=params.a1_gscale,
        a2_gscale=params.a2_gscale,
        g1_alphas=(1.0 / weights.w4a16_w13_global_scale).to(torch.float32).contiguous(),
        g2_alphas=(1.0 / weights.w4a16_w2_global_scale).to(torch.float32).contiguous(),
    )


def uses_unit_scale_contract(
    profile: ModelProfile,
    quant_mode: str,
    activation: str,
) -> bool:
    return (
        quant_mode.lower() == "w4a16"
        and activation == "relu2"
        and profile.checkpoint_family in {"nano35_w4a16", "nano35_w4a16_shape"}
    )


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
    quant_mode: str,
    x: torch.Tensor,
    weights: ExpertWeights,
    params: ScaleContractParams,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    spec = weights.spec
    quant_mode = quant_mode.lower()
    if quant_mode == "w4a16":
        if oracle_mode == "nvfp4":
            raise ValueError("--oracle-mode nvfp4 is not valid with --quant-mode w4a16")
        params = get_w4a16_oracle_params(weights, params)
        if weights.source_format == "fp4_e8m0_k32":
            if oracle_mode not in {"f32", "w4a16", "flashinfer"}:
                raise ValueError(f"unsupported W4A16 oracle mode {oracle_mode!r}")
            if oracle_mode == "flashinfer":
                if weights.oracle_flashinfer_weights is None:
                    raise ValueError("FlashInfer fp4_e8m0_k32 oracle requires FI-structured oracle tensors")
                return moe_reference_w4a16_fp4_e8m0_k32_flashinfer_prepared(
                    x,
                    weights.oracle_flashinfer_weights,
                    params.g1_alphas,
                    params.g2_alphas,
                    topk_ids,
                    topk_weights,
                    spec.num_experts,
                    spec.hidden_size,
                    spec.I_tp,
                    activation=activation,
                    swiglu_limit=swiglu_limit,
                )
            return moe_reference_w4a16_fp4_e8m0_k32(
                x,
                weights.w13_weight,
                weights.w13_blockscale_swizzled,
                params.g1_alphas,
                weights.w2_weight,
                weights.w2_blockscale_swizzled,
                params.g2_alphas,
                topk_ids,
                topk_weights,
                spec.num_experts,
                spec.hidden_size,
                spec.I_tp,
                activation=activation,
                swiglu_limit=swiglu_limit,
                w13_layout=weights.w13_layout,
            )
        if oracle_mode == "f32":
            return moe_reference_w4a16_f32(
                x,
                weights.w13_weight,
                weights.w13_blockscale_swizzled,
                params.g1_alphas,
                weights.w2_weight,
                weights.w2_blockscale_swizzled,
                params.g2_alphas,
                topk_ids,
                topk_weights,
                spec.num_experts,
                spec.hidden_size,
                spec.I_tp,
                activation=activation,
                swiglu_limit=swiglu_limit,
            )
        if oracle_mode != "w4a16":
            raise ValueError(f"unsupported W4A16 oracle mode {oracle_mode!r}")
        return moe_reference_w4a16(
            x,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            params.g1_alphas,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            params.g2_alphas,
            topk_ids,
            topk_weights,
            spec.num_experts,
            spec.hidden_size,
            spec.I_tp,
            activation=activation,
        )
    if oracle_mode == "flashinfer":
        raise ValueError("--oracle-mode flashinfer requires --quant-mode w4a16 and source_format='fp4_e8m0_k32'")
    if oracle_mode == "w4a16":
        raise ValueError("--oracle-mode w4a16 requires --quant-mode w4a16")
    oracle_fn = moe_reference_nvfp4 if oracle_mode == "nvfp4" else moe_reference_f32
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
        # Default fast_math keeps the fast FP4 dot path.  It is intentionally
        # not bit-exact against the f32-accum oracle, but should remain well
        # inside FlashInfer's oracle error on the MiniMax-M2.7 micro profile.
        "max_abs": 0.05,
        "rmse": 0.0075,
        "mean_abs": 0.005,
        "cos_min": 0.9999,
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

W4A16_ORACLE_TOLERANCES = {
    "silu": {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9975,
    },
    "relu2": {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9900,
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
    label: str,
    metrics: OracleMetrics,
    batch_size: int,
    *,
    activation: str = "silu",
    oracle_mode: str = "nvfp4",
) -> list[str]:
    failures = []
    if oracle_mode in {"w4a16", "flashinfer"}:
        tol = W4A16_ORACLE_TOLERANCES[activation]
    else:
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
    try:
        from b12x.moe.fused.w4a16.kernel import clear_w4a16_kernel_cache
    except ImportError:
        return
    clear_w4a16_kernel_cache()


def _validate_reference_case(
    args,
    spec: ModelSpec,
    model_profile: ModelProfile,
    batch_sizes: Sequence[int],
) -> None:
    if args.reference not in ("r4v2",):
        return
    raise ValueError(f"--reference {args.reference} is no longer supported")
    expected = {
        "hidden_size": 4096,
        "I_tp": 256,
        "num_experts": 512,
        "top_k": 10,
    }
    actual = {
        "hidden_size": spec.hidden_size,
        "I_tp": spec.I_tp,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
    }
    if actual != expected:
        raise ValueError(f"--reference r4v2 expects {expected}, got {actual}")
    unsupported = sorted(set(batch_sizes) - {1, 2, 4, 8})
    if unsupported:
        raise ValueError(f"--reference r4v2 only supports batch sizes 1, 2, 4, 8; got {unsupported}")


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
    *,
    quant_mode: str = "nvfp4",
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
        quant_mode=quant_mode,
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
    quant_mode: str = "nvfp4",
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
            quant_mode=quant_mode,
            source_format=weights.source_format,
            w13_layout=weights.w13_layout,
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
    quant_mode: str = "nvfp4",
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
            quant_mode=quant_mode,
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

    layer_start = args.graph_layer_start
    if profile.shape is None:
        cfg = _load_config(model_path)
        total_layers = cfg["num_hidden_layers"]
    else:
        total_layers = layer_start + graph_num_layers
    if layer_start < 0 or layer_start + graph_num_layers > total_layers:
        raise ValueError(
            f"requested layers [{layer_start}, {layer_start + graph_num_layers}) exceed model depth {total_layers}"
        )

    if args.reference != "none" or args.validate != "none":
        print(
            "Note: multi-layer graph mode skips flashinfer/oracle checks and validates graph replay "
            "against an eager layer chain."
        )
    print("Multi-layer graph mode")
    print("Backend: b12x")
    print(f"Quant mode: {args.quant_mode}")
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
        keep_flashinfer_oracle_copy=args.validate == "oracle" and args.oracle_mode == "flashinfer",
    )
    params_stack = [
        get_quant_mode_params(weights, args.scale_contract, args.quant_mode)
        for weights in weights_stack
    ]

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
            quant_mode=args.quant_mode,
        )

        run_moe_layer_chain(
            weights_stack,
            params_stack,
            x_buf,
            topk_ids_bufs,
            topk_weights_bufs,
            activation=args.activation,
            fast_math=args.fast_math,
            quant_mode=args.quant_mode,
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
            quant_mode=args.quant_mode,
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
                quant_mode=args.quant_mode,
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
    from b12x.integration.tp_moe import default_moe_quant_mode

    quant_mode_default = default_moe_quant_mode()
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
    parser.add_argument("--tp-size", type=int, default=None, help="Override TP size from model profile")
    parser.add_argument("--tp-parallel", action="store_true", help="Load all TP rank slices and replay per-rank CUDA graphs in parallel streams")
    parser.add_argument("--model-path", type=pathlib.Path, default=None)
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument("--activation", choices=["silu", "relu2"], default=None)
    parser.add_argument(
        "--swiglu-limit",
        type=float,
        default=None,
        help="Clamp gated W4A16 SwiGLU inputs before silu/up multiply; defaults to the model profile value.",
    )
    parser.add_argument(
        "--quant-mode",
        choices=["nvfp4", "w4a16"],
        default=None,
        help=(
            "Backend math mode. w4a16 keeps activations BF16 and dequantizes "
            "FP4 weights inline. If omitted, the model profile default is used, "
            "otherwise B12X_MOE_FORCE_A16=1 changes the default to w4a16."
        ),
    )
    parser.add_argument("--graph-mode", choices=["single-op", "multi-layer"], default="single-op")
    parser.add_argument("--graph-num-layers", type=int, default=4)
    parser.add_argument("--graph-layer-start", type=int, default=0)
    parser.add_argument(
        "--reference",
        choices=["flashinfer", "none"],
        default=None,
    )
    parser.add_argument("--scale-contract", choices=["shared", "per-expert"], default="shared")
    parser.add_argument(
        "--w4a16-native",
        action="store_true",
        help=(
            "Build the native (modelopt) W4A16 representation so small-M shapes "
            "route to the micro decode kernel. For w31 sources (e.g. DeepSeek V4 "
            "Flash) the W13 halves are reordered to w13 at prep time."
        ),
    )
    parser.add_argument("--validate", choices=["none", "oracle"], default=None)
    parser.add_argument("--oracle-mode", choices=["nvfp4", "w4a16", "f32", "flashinfer"], default=None)
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
    model_profile = MODEL_PROFILES[args.model_profile]
    if args.activation is None:
        args.activation = model_profile.default_activation
    if args.quant_mode is None:
        args.quant_mode = model_profile.default_quant_mode or quant_mode_default
    use_w4a16 = args.quant_mode == "w4a16"
    swiglu_limit = args.swiglu_limit if args.swiglu_limit is not None else model_profile.default_swiglu_limit
    if args.validate is None:
        args.validate = model_profile.default_validate
    if args.reference is None:
        args.reference = "none" if use_w4a16 else "flashinfer"
    if args.oracle_mode is None:
        args.oracle_mode = args.quant_mode
    keep_flashinfer_oracle_copy = args.validate == "oracle" and args.oracle_mode == "flashinfer"
    batch_sizes = (
        args.batch_sizes
        if args.batch_sizes is not None
        else BATCH_SIZE_PROFILES[args.batch_size_profile]
    )
    model_path = resolve_model_path(model_profile, args.model_path)
    layer_idx = model_profile.default_layer_idx if args.layer_idx is None else args.layer_idx

    if args.scale_contract == "per-expert" and args.reference == "flashinfer":
        raise ValueError("--reference flashinfer is only valid with --scale-contract shared")
    if args.reference == "flashinfer" and args.quant_mode != "nvfp4":
        raise ValueError("--reference flashinfer is only valid with --quant-mode nvfp4")
    if args.reference == "flashinfer" and args.activation != "silu":
        raise ValueError("--reference flashinfer is only valid with --activation silu")
    if model_profile.checkpoint_family == "deepseek_v4_flash" and not use_w4a16:
        raise ValueError("DeepSeek V4 Flash FP4 checkpoint profile requires --quant-mode w4a16")
    if use_w4a16 and args.graph_mode != "single-op":
        raise ValueError("--quant-mode w4a16 currently supports --graph-mode single-op")
    if use_w4a16 and args.tp_parallel:
        raise ValueError("--quant-mode w4a16 currently does not support --tp-parallel")
    if args.graph_only and not args.cuda_graph:
        raise ValueError("--graph-only requires --cuda-graph")

    require_sm120()
    torch.empty(1, device="cuda")
    device = torch.device("cuda")
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0

    spec = build_model_spec(model_path, model_profile, tp_size_override=args.tp_size)
    _validate_reference_case(args, spec, model_profile, batch_sizes)

    benchmark_scope = "Routing + MoE kernel" if args.include_routing else "Pre-routed MoE kernel only"
    print(f"MoE benchmark ({benchmark_scope})")
    print(
        f"{model_profile.label}  TP={spec.tp_size}, K={spec.hidden_size}, I_tp={spec.I_tp}, "
        f"E={spec.num_experts}, top_k={spec.top_k}"
    )
    print(f"Model path: {model_path}")
    if model_profile.shape is not None:
        print("Weights: synthetic shape-only")
    print(f"Layer: {layer_idx}")
    print(f"Activation: {args.activation}")
    print(f"Quant mode: {args.quant_mode}")
    print(f"Routing source: {model_profile.default_routing}")
    print(f"Batch-size profile: {args.batch_size_profile} -> {batch_sizes}")
    backend_label = "b12x"
    print(f"Backend: {backend_label}")
    print(f"Reference: {args.reference}")
    print(f"Scale contract: {args.scale_contract}")
    print(f"Validation: {args.validate}")
    print(f"Fast math: {'on' if args.fast_math else 'off'}")
    if use_w4a16:
        print("W4A16 kernel: fused W4A16 FC1+FC2")
    if swiglu_limit is not None:
        print(f"SwiGLU limit: {swiglu_limit:g}")
    if args.flush_l2:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")
    print(f"Graph mode: {args.graph_mode}")
    print(f"Graph only: {'yes' if args.graph_only else 'no'}")
    print(f"Timing passes per batch size: {args.repeats} x {args.iters} iterations")
    print(
        "Timed region: "
        + (
            "model gate routing + backend launch"
            if args.include_routing and model_profile.default_routing == "model"
            else "top-k + softmax routing + backend launch"
            if args.include_routing
            else "backend launch only"
        )
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
        keep_flashinfer_oracle_copy=keep_flashinfer_oracle_copy,
    )
    print(f"Source format: {weights.source_format}")
    params = get_quant_mode_params(weights, args.scale_contract, args.quant_mode)
    backend_w4a16_prepared = None
    make_backend_w4a16_buffers = None
    if use_w4a16:
        from b12x.moe.fused.w4a16.prepare import (
            make_w4a16_packed_buffers as make_w4a16_buffers,
            prepare_w4a16_e8m0_native_weights,
            prepare_w4a16_modelopt_native_weights,
            prepare_w4a16_packed_weights as prepare_w4a16_weights,
        )

        w4a16_g1, w4a16_g2, source_format = get_w4a16_prepare_scales(
            weights,
            params,
        )
        if args.w4a16_native:
            # Native (modelopt) prep so small-M routes to the micro decode kernel.
            # The micro kernel handles both w13 and w31 (gate_up) natively, so the
            # source layout (DeepSeek V4 Flash is w31) is passed through as-is.
            if source_format == "fp4_e8m0_k32":
                backend_w4a16_prepared = prepare_w4a16_e8m0_native_weights(
                    weights.w13_weight,
                    weights.w13_blockscale_swizzled,
                    w4a16_g1,
                    weights.w2_weight,
                    weights.w2_blockscale_swizzled,
                    w4a16_g2,
                    activation=args.activation,
                    params_dtype=torch.bfloat16,
                    w13_layout=weights.w13_layout,
                )
            else:
                backend_w4a16_prepared = prepare_w4a16_modelopt_native_weights(
                    weights.w13_weight,
                    weights.w13_blockscale_swizzled,
                    w4a16_g1,
                    weights.w2_weight,
                    weights.w2_blockscale_swizzled,
                    w4a16_g2,
                    activation=args.activation,
                    params_dtype=torch.bfloat16,
                    source_format=source_format,
                    w13_layout=weights.w13_layout,
                )
            print(
                "W4A16 prep: native (micro decode path enabled for small M, "
                f"w13_layout={weights.w13_layout})"
            )
        else:
            backend_w4a16_prepared = prepare_w4a16_weights(
                weights.w13_weight,
                weights.w13_blockscale_swizzled,
                w4a16_g1,
                weights.w2_weight,
                weights.w2_blockscale_swizzled,
                w4a16_g2,
                activation=args.activation,
                params_dtype=torch.bfloat16,
                source_format=source_format,
                w13_layout=weights.w13_layout,
            )
        print(
            "W4A16 scale format: "
            f"{getattr(backend_w4a16_prepared, 'scale_format', source_format)}"
        )
        make_backend_w4a16_buffers = make_w4a16_buffers

    unit_scale_contract = uses_unit_scale_contract(
        model_profile,
        args.quant_mode,
        args.activation,
    )

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        allocate_tp_moe_workspace_pool,
        b12x_moe_fp4,
    )
    w4a16_moe = None
    if use_w4a16:
        from b12x.moe.fused.w4a16.kernel import run_w4a16_moe

        w4a16_moe = run_w4a16_moe

    _clear_b12x_caches()

    print("  Warming up b12x (compilation)...", end="", flush=True)
    x_warm, topk_ids_w, topk_weights_w = make_profile_routed_inputs(
        model_profile,
        weights,
        spec,
        1,
        42,
        device,
    )
    if use_w4a16:
        assert w4a16_moe is not None
        assert backend_w4a16_prepared is not None
        assert make_backend_w4a16_buffers is not None
        warmup_buffers = make_backend_w4a16_buffers(
            backend_w4a16_prepared,
            m=x_warm.shape[0],
            topk=spec.top_k,
            dtype=torch.bfloat16,
            device=device,
        )
        w4a16_moe(
            x_warm,
            backend_w4a16_prepared,
            topk_weights_w,
            topk_ids_w,
            activation=args.activation,
            fast_math=args.fast_math,
            swiglu_limit=swiglu_limit,
            intermediate_cache13=warmup_buffers.intermediate_cache13,
            intermediate_cache2=warmup_buffers.intermediate_cache2,
            output=warmup_buffers.output,
            fc1_c_tmp=warmup_buffers.fc1_c_tmp,
            fc2_c_tmp=warmup_buffers.fc2_c_tmp,
            packed_route_indices=warmup_buffers.packed_route_indices,
            block_expert_ids=warmup_buffers.block_expert_ids,
            packed_route_count=warmup_buffers.packed_route_count,
            expert_offsets=warmup_buffers.expert_offsets,
        )
    else:
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
            quant_mode=args.quant_mode,
            unit_scale_contract=unit_scale_contract,
            source_format=weights.source_format,
            w13_layout=weights.w13_layout,
        )
    torch.cuda.synchronize()
    print(" done.")

    # ---- TP-parallel setup ----
    tp_parallel_ranks: list[tuple[ModelSpec, ExpertWeights, ScaleContractParams]] = []
    if args.tp_parallel and spec.tp_size > 1:
        print(f"  Loading TP-parallel ranks...", end="", flush=True)
        for r in range(spec.tp_size):
            rspec = build_model_spec(model_path, model_profile, tp_size_override=args.tp_size, tp_rank=r)
            rw = load_expert_weights(
                model_path, rspec, layer_idx=layer_idx,
                activation=args.activation, checkpoint_family=model_profile.checkpoint_family,
            )
            rp = get_quant_mode_params(rw, args.scale_contract, args.quant_mode)
            tp_parallel_ranks.append((rspec, rw, rp))
        # Warm up each rank's kernel
        for r, (rspec, rw, rp) in enumerate(tp_parallel_ranks):
            x_r = torch.randn(1, rspec.hidden_size, dtype=torch.bfloat16, device=device)
            rk_warm = torch.randn(1, rspec.num_experts, dtype=torch.float32, device=device)
            rk_logits, rk_ids = torch.topk(rk_warm, rspec.top_k, dim=-1)
            rk_weights = torch.softmax(rk_logits, dim=-1)
            ws_r = allocate_tp_moe_workspace_pool()
            b12x_moe_fp4(
                x_r, rp.a1_gscale, rw.w13_weight, rw.w13_blockscale_swizzled,
                rp.g1_alphas, rp.a2_gscale, rw.w2_weight, rw.w2_blockscale_swizzled,
                rp.g2_alphas, rk_weights, rk_ids,
                workspace=ws_r, fast_math=args.fast_math, activation=args.activation,
                quant_mode=args.quant_mode,
                unit_scale_contract=unit_scale_contract,
                source_format=rw.source_format,
                w13_layout=rw.w13_layout,
            )
        torch.cuda.synchronize()
        print(f" {spec.tp_size} ranks done.")

    batch_results: dict[int, BatchResult] = {}
    accuracy_failures: list[str] = []
    reference_warnings: list[str] = []
    for batch_size in batch_sizes:
        print(f"\n{'=' * 70}")
        print(f"  batch_size={batch_size}  (tokens*top_k = {batch_size * spec.top_k} expert calls)")
        print(f"{'=' * 70}")

        torch.manual_seed(42 + batch_size)
        x = make_input_activations(spec, batch_size, 42 + batch_size, device)
        routing_logits = None
        if model_profile.default_routing == "model":
            topk_ids, topk_weights = compute_model_gate_routing(weights, x, seed=43 + batch_size)
        else:
            routing_logits = torch.randn(batch_size, spec.num_experts, dtype=torch.float32, device=device)
            topk_logits, topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
            topk_weights = torch.softmax(topk_logits, dim=-1)

        def compute_timed_routing() -> tuple[torch.Tensor, torch.Tensor]:
            if model_profile.default_routing == "model":
                return compute_model_gate_routing(weights, x, seed=43 + batch_size)
            assert routing_logits is not None
            timed_topk_logits, timed_topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
            timed_topk_weights = torch.softmax(timed_topk_logits, dim=-1)
            return timed_topk_ids, timed_topk_weights

        backend_output = torch.empty_like(x)
        backend_workspace = (
            None
            if use_w4a16
            else allocate_tp_moe_workspace_pool()
        )
        backend_w4a16_buffers = (
            make_backend_w4a16_buffers(
                backend_w4a16_prepared,
                m=batch_size,
                topk=spec.top_k,
                dtype=torch.bfloat16,
                device=device,
            )
            if use_w4a16
            else None
        )

        def make_backend_e2e() -> Callable[[], torch.Tensor]:

            def impl_launch(topk_ids_local: torch.Tensor, topk_weights_local: torch.Tensor) -> torch.Tensor:
                if use_w4a16:
                    assert w4a16_moe is not None
                    assert backend_w4a16_prepared is not None
                    assert backend_w4a16_buffers is not None
                    return w4a16_moe(
                        x,
                        backend_w4a16_prepared,
                        topk_weights_local,
                        topk_ids_local,
                        activation=args.activation,
                        fast_math=args.fast_math,
                        swiglu_limit=swiglu_limit,
                        intermediate_cache13=backend_w4a16_buffers.intermediate_cache13,
                        intermediate_cache2=backend_w4a16_buffers.intermediate_cache2,
                        output=backend_output,
                        fc1_c_tmp=backend_w4a16_buffers.fc1_c_tmp,
                        fc2_c_tmp=backend_w4a16_buffers.fc2_c_tmp,
                        packed_route_indices=backend_w4a16_buffers.packed_route_indices,
                        block_expert_ids=backend_w4a16_buffers.block_expert_ids,
                        packed_route_count=backend_w4a16_buffers.packed_route_count,
                        expert_offsets=backend_w4a16_buffers.expert_offsets,
                    )
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
                    quant_mode=args.quant_mode,
                    unit_scale_contract=unit_scale_contract,
                    source_format=weights.source_format,
                    w13_layout=weights.w13_layout,
                )

            def impl_e2e() -> torch.Tensor:
                if args.include_routing:
                    timed_topk_ids, timed_topk_weights = compute_timed_routing()
                    return impl_launch(timed_topk_ids, timed_topk_weights)
                return impl_launch(topk_ids, topk_weights)

            return impl_e2e

        backend_e2e = make_backend_e2e()

        ref_name = None
        ref_launch = None
        ref_result_tensor = None
        if args.reference == "flashinfer":
            from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe

            ref_name = "FlashInfer"
            base_ref_launch, ref_result_tensor = bench_flashinfer(weights, x, topk_ids, topk_weights)
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
                    timed_topk_ids, timed_topk_weights = compute_timed_routing()
                    flashinfer_cutlass_fused_moe(
                        output=ref_result_tensor,
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
                args.quant_mode,
                x,
                weights,
                params,
                topk_ids,
                topk_weights,
                activation=args.activation,
                swiglu_limit=swiglu_limit,
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
            ref_output = ref_result_tensor.clone()

        backend_out = backend_e2e().clone()
        torch.cuda.synchronize()

        if ref_output is not None:
            ref_compare_metrics = compare_to_reference(backend_out, ref_output)
            print(f"  {format_oracle_metrics(f'{backend_label} vs {ref_name}', ref_compare_metrics)}")

        if oracle_ref is not None:
            backend_metrics = compare_to_reference(backend_out, oracle_ref)
            print(f"  {format_oracle_metrics(f'{backend_label} vs oracle', backend_metrics)}")
            accuracy_failures.extend(
                check_oracle_metrics(
                    f"{backend_label} vs oracle", backend_metrics, batch_size,
                    activation=args.activation,
                    oracle_mode=args.oracle_mode,
                )
            )
            if ref_output is not None and ref_name is not None:
                ref_metrics = compare_to_reference(ref_output, oracle_ref)
                print(f"  {format_oracle_metrics(f'{ref_name} vs oracle', ref_metrics)}")
                reference_warnings.extend(
                    check_oracle_metrics(
                        f"{ref_name} vs oracle", ref_metrics, batch_size,
                        activation=args.activation,
                        oracle_mode=args.oracle_mode,
                    )
                )

        if args.profile_once != "none":
            if args.profile_once == "backend":
                profile_fn = backend_e2e
                profile_name = backend_label
            else:
                if ref_launch is None or args.reference != args.profile_once:
                    raise ValueError(f"--profile-once {args.profile_once} requires --reference {args.profile_once}")
                profile_fn = ref_launch
                profile_name = ref_name or args.profile_once
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
            graph_stats_by_name: dict[str, TimingStats] = {}
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
                    graph_runs = [
                        bench_events(
                            replay,
                            warmup=args.warmup,
                            iters=args.iters,
                            l2_flush=l2_flush,
                        )
                        for _ in range(args.repeats)
                    ]
                    stats = summarize_timing_runs(graph_runs)
                    graph_stats_by_name[name] = stats
                    print(f" {fmt_timing_stats(stats)}")
                except Exception as exc:
                    print(f" FAILED ({type(exc).__name__}: {exc})")

            if ref_name in graph_stats_by_name and backend_label in graph_stats_by_name:
                ref_graph_stats = graph_stats_by_name[ref_name]
                backend_graph_stats = graph_stats_by_name[backend_label]
                graph_ratio_stats = RatioStats(
                    [
                        backend_ms / ref_ms
                        for backend_ms, ref_ms in zip(
                            backend_graph_stats.per_repeat_median_ms,
                            ref_graph_stats.per_repeat_median_ms,
                            strict=True,
                        )
                    ]
                )
                print(f"    graph ratio vs {ref_name.lower()}: {fmt_ratio_stats(graph_ratio_stats)}")
                batch_results[batch_size] = BatchResult(
                    backend_stats=backend_graph_stats,
                    ref_stats=ref_graph_stats,
                    ratio_stats=graph_ratio_stats,
                )
            elif backend_label in graph_stats_by_name:
                ref_graph_stats = None
                if ref_name is not None and ref_name in graph_stats_by_name:
                    ref_graph_stats = graph_stats_by_name[ref_name]
                batch_results[batch_size] = BatchResult(
                    backend_stats=graph_stats_by_name[backend_label],
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

        # ---- TP-parallel graph replay ----
        if tp_parallel_ranks:
            tp_n = len(tp_parallel_ranks)
            label = f"b12x TP={tp_n} parallel"
            print(f"  {label} (CUDA graph):".ljust(28), end="", flush=True)
            try:
                # Per-rank inputs, outputs, workspaces
                tp_x = [torch.randn(batch_size, rs.hidden_size, dtype=torch.bfloat16, device=device) for rs, _, _ in tp_parallel_ranks]
                tp_routing = [torch.randn(batch_size, rs.num_experts, dtype=torch.float32, device=device) for rs, _, _ in tp_parallel_ranks]
                tp_topk_ids: list[torch.Tensor] = []
                tp_topk_weights: list[torch.Tensor] = []
                for r_routing, (rspec, _, _) in zip(tp_routing, tp_parallel_ranks, strict=True):
                    r_logits, r_ids = torch.topk(r_routing, rspec.top_k, dim=-1)
                    tp_topk_ids.append(r_ids)
                    tp_topk_weights.append(torch.softmax(r_logits, dim=-1))
                tp_outputs = [torch.empty_like(tp_x[r]) for r in range(tp_n)]
                tp_workspaces = [allocate_tp_moe_workspace_pool() for _ in range(tp_n)]
                tp_streams = [torch.cuda.Stream() for _ in range(tp_n)]

                def launch_tp_rank(r: int) -> None:
                    _rspec, rw, rp = tp_parallel_ranks[r]
                    b12x_moe_fp4(
                        tp_x[r], rp.a1_gscale, rw.w13_weight, rw.w13_blockscale_swizzled,
                        rp.g1_alphas, rp.a2_gscale, rw.w2_weight, rw.w2_blockscale_swizzled,
                        rp.g2_alphas, tp_topk_weights[r], tp_topk_ids[r],
                        workspace=tp_workspaces[r], output=tp_outputs[r],
                        fast_math=args.fast_math, activation=args.activation,
                        quant_mode=args.quant_mode,
                        unit_scale_contract=unit_scale_contract,
                        source_format=rw.source_format,
                        w13_layout=rw.w13_layout,
                    )

                # Warm eager launches
                for r, stream in enumerate(tp_streams):
                    with torch.cuda.stream(stream):
                        launch_tp_rank(r)
                torch.cuda.synchronize()

                # Capture each rank on its own stream. A single graph context only
                # captures work issued to its active capture stream.
                tp_graphs: list[torch.cuda.CUDAGraph] = []
                for r, stream in enumerate(tp_streams):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream, capture_error_mode="relaxed"):
                        launch_tp_rank(r)
                    tp_graphs.append(graph)
                torch.cuda.synchronize()

                def tp_replay(
                    graphs: list[torch.cuda.CUDAGraph] = tp_graphs,
                    streams: list[torch.cuda.Stream] = tp_streams,
                ) -> None:
                    timing_stream = torch.cuda.current_stream()
                    for stream in streams:
                        stream.wait_stream(timing_stream)
                    for graph, stream in zip(graphs, streams, strict=True):
                        with torch.cuda.stream(stream):
                            graph.replay()
                    for stream in streams:
                        timing_stream.wait_stream(stream)

                tp_times = bench_events(
                    tp_replay, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush,
                )
                print(f" {fmt_us(tp_times)}")
            except Exception as exc:
                print(f" FAILED ({type(exc).__name__}: {exc})")

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
            parts.append(f"{backend_label} {result.backend_stats.median_us:.1f} us")
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

    if reference_warnings:
        print(f"\n\033[1;33m{'=' * 70}")
        print("  REFERENCE WARNING")
        print(f"{'=' * 70}")
        for f in reference_warnings:
            print(f)
        print(f"{'=' * 70}\033[0m")
    if accuracy_failures:
        print(f"\n\033[1;31m{'=' * 70}")
        print("  ACCURACY CHECK FAILED")
        print(f"{'=' * 70}")
        for f in accuracy_failures:
            print(f)
        print(f"{'=' * 70}\033[0m")
        sys.exit(1)


def main() -> None:
    bench_e2e()


if __name__ == "__main__":
    main()
