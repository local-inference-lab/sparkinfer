#!/usr/bin/env python3
"""BF16 MoE benchmark and shared BF16 weight-loading utilities.

This follows the structure of ``benchmark_moe.py`` but targets the additive
BF16 MoE family. It loads expert weights as dense BF16 tensors, either from a
native BF16 checkpoint or by dequantizing NVFP4 checkpoint weights once during
benchmark setup.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.checkpoint_loader import IndexedSafetensorLoader
from b12x.moe.fused.bf16.reference import (
    OracleMetrics,
    compare_to_reference,
    moe_reference_bf16,
)


DRAFT_GRAPH_BATCH_SIZES = [1, 2, 3, 4]
LEGACY_BATCH_SIZES = [1, 2, 4, 8]
RECORDED_SGLANG_SINGLE_REQUEST_BATCH_SIZES = [1, 23, 80]
EAGER_PREFILL_BATCH_SIZES = [16384, 32768]
CHUNKED_PREFILL_BATCH_SIZES = [8192, 16384, 24576, 32768]
BATCH_SIZE_PROFILES = {
    "draft-graph": DRAFT_GRAPH_BATCH_SIZES,
    "eager-prefill": EAGER_PREFILL_BATCH_SIZES,
    "micro": LEGACY_BATCH_SIZES,
    "sglang-single-request": RECORDED_SGLANG_SINGLE_REQUEST_BATCH_SIZES,
    "chunked-prefill": CHUNKED_PREFILL_BATCH_SIZES,
}
TP_SIZE = 4
TP_RANK = 0

ORACLE_TOLERANCES = {
    "max_abs": 0.10,
    "rmse": 0.02,
    "mean_abs": 0.01,
    "cos_min": 0.999,
}


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


def normalized_batch_latency_us(latency_ms: float, batch_size: int) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return latency_ms * 1000.0 / batch_size


def normalized_result_us(per_batch_latency_ms: dict[int, float]) -> float:
    if not per_batch_latency_ms:
        raise ValueError("normalized_result_us requires at least one batch result")
    normalized_us = [
        normalized_batch_latency_us(latency_ms, batch_size)
        for batch_size, latency_ms in sorted(per_batch_latency_ms.items())
    ]
    return statistics.geometric_mean(normalized_us)


def _timing_mode_suffix(execution_mode: str) -> str:
    if execution_mode == "graph":
        return "CUDA graph"
    if execution_mode == "eager":
        return "eager"
    raise ValueError(f"unsupported execution_mode={execution_mode!r}")


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
    default_activation: str
    tp_size: int
    hf_repo_id: str
    preferred_local_paths: tuple[str, ...] = ()


MODEL_PROFILES = {
    "qwen397b": ModelProfile(
        label="Qwen3.5-397B BF16",
        checkpoint_family="qwen",
        default_layer_idx=0,
        default_activation="silu",
        tp_size=TP_SIZE,
        hf_repo_id="nvidia/Qwen3.5-397B-A17B-NVFP4",
        preferred_local_paths=(
            "/data/models/Qwen3.5-397B-A17B-NVFP4-BF16shared",
            "/data/models/Qwen3.5-397B-A17B-NVFP4",
        ),
    ),
    "nemotron-backbone": ModelProfile(
        label="NVIDIA Nemotron Backbone BF16",
        checkpoint_family="nemotron",
        default_layer_idx=1,
        default_activation="relu2",
        tp_size=1,
        hf_repo_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        preferred_local_paths=(
            "/data/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        ),
    ),
}


@dataclass
class ExpertWeights:
    layer_idx: int
    spec: ModelSpec
    w1_weight: torch.Tensor
    w2_weight: torch.Tensor


@dataclass(frozen=True)
class _BF16BackendAPI:
    clear_caches: Callable[[], None]
    allocate_workspace_pool: Callable[[], object]
    b12x_moe: Callable[..., torch.Tensor]


@dataclass(frozen=True)
class _FlashInferAPI:
    cutlass_fused_moe: Callable[..., Any]
    activation_types: dict[str, Any]


@dataclass(frozen=True)
class _TritonAPI:
    outplace_fused_experts: Callable[..., torch.Tensor]


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


def resolve_model_path(
    profile: ModelProfile,
    override: pathlib.Path | None,
) -> pathlib.Path:
    if override is not None:
        return override
    env_path = os.environ.get("B12X_BF16_MODEL_PATH")
    if env_path:
        return pathlib.Path(env_path)
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


def resolve_activation(profile: ModelProfile, activation: str | None) -> str:
    resolved = profile.default_activation if activation is None else activation
    if resolved not in {"silu", "relu2"}:
        raise ValueError(f"unsupported activation {resolved!r}")
    expected = profile.default_activation
    if resolved != expected:
        raise ValueError(
            f"{profile.label} BF16 benchmark expects activation={expected!r}, got {resolved!r}"
        )
    return resolved


def _dequant_fp4(weight: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=weight.device,
    )
    packed = weight.to(torch.int32)
    lo = lut[packed & 0xF]
    hi = lut[(packed >> 4) & 0xF]
    vals = torch.stack([lo, hi], dim=-1).reshape(weight.shape[0], weight.shape[1] * 2)
    grouped_scale = scale.float().unsqueeze(-1).expand(-1, -1, 16).reshape(weight.shape[0], -1)
    return (vals * grouped_scale * float(scale2)).to(torch.bfloat16)


def _load_sharded_linear_weight(
    loader: IndexedSafetensorLoader,
    weight_key: str,
    *,
    shard_dim: int,
    shard_offset: int,
    shard_size: int,
    scale_key: str | None = None,
    scale2_key: str | None = None,
    device: torch.device,
) -> torch.Tensor:
    weight = loader.get_tensor(weight_key)
    if weight.dtype == torch.uint8:
        if scale_key is None or scale2_key is None:
            raise ValueError(f"quantized weight {weight_key} requires scale keys")
        scale = loader.get_tensor(scale_key)
        scale2 = loader.get_tensor(scale2_key)
        if shard_dim == 0:
            weight = weight.narrow(0, shard_offset, shard_size)
            scale = scale.narrow(0, shard_offset, shard_size)
            return _dequant_fp4(weight.to(device), scale.to(device), scale2.to(device))
        dense = _dequant_fp4(weight.to(device), scale.to(device), scale2.to(device))
        return dense.narrow(1, shard_offset, shard_size).contiguous()

    dense = weight.narrow(shard_dim, shard_offset, shard_size)
    return dense.to(device=device, dtype=torch.bfloat16).contiguous()


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

    up_rows = torch.empty(E, I_tp, K, dtype=torch.bfloat16, device=device)
    down_rows = torch.empty(E, K, I_tp, dtype=torch.bfloat16, device=device)
    tp_off = spec.tp_rank * I_tp

    if checkpoint_family == "qwen":
        if activation != "silu":
            raise ValueError("Qwen BF16 benchmark expects silu experts")
        assert cfg["num_experts"] == spec.num_experts
        assert cfg["moe_intermediate_size"] == spec.intermediate_size
        assert cfg["hidden_size"] == spec.hidden_size

        prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
        gate_rows = torch.empty(E, I_tp, K, dtype=torch.bfloat16, device=device)

        print(f"  Loading {E} BF16 experts...", end="", flush=True)
        for eid in range(E):
            ep = f"{prefix}.{eid}"
            gate_rows[eid] = _load_sharded_linear_weight(
                loader,
                f"{ep}.gate_proj.weight",
                shard_dim=0,
                shard_offset=tp_off,
                shard_size=I_tp,
                scale_key=f"{ep}.gate_proj.weight_scale",
                scale2_key=f"{ep}.gate_proj.weight_scale_2",
                device=device,
            )
            up_rows[eid] = _load_sharded_linear_weight(
                loader,
                f"{ep}.up_proj.weight",
                shard_dim=0,
                shard_offset=tp_off,
                shard_size=I_tp,
                scale_key=f"{ep}.up_proj.weight_scale",
                scale2_key=f"{ep}.up_proj.weight_scale_2",
                device=device,
            )
            down_rows[eid] = _load_sharded_linear_weight(
                loader,
                f"{ep}.down_proj.weight",
                shard_dim=1,
                shard_offset=tp_off,
                shard_size=I_tp,
                scale_key=f"{ep}.down_proj.weight_scale",
                scale2_key=f"{ep}.down_proj.weight_scale_2",
                device=device,
            )
        print(" done.")
        w1_weight = torch.cat([up_rows, gate_rows], dim=1).contiguous()
    elif checkpoint_family == "nemotron":
        if activation != "relu2":
            raise ValueError("Nemotron backbone BF16 benchmark expects relu2 experts")
        assert cfg["n_routed_experts"] == spec.num_experts
        assert cfg["moe_intermediate_size"] == spec.intermediate_size
        assert cfg["hidden_size"] // TP_SIZE == spec.hidden_size

        prefix = f"backbone.layers.{layer_idx}.mixer.experts"
        print(f"  Loading {E} BF16 experts...", end="", flush=True)
        for eid in range(E):
            ep = f"{prefix}.{eid}"
            up_rows[eid] = _load_sharded_linear_weight(
                loader,
                f"{ep}.up_proj.weight",
                shard_dim=0,
                shard_offset=0,
                shard_size=I_tp,
                scale_key=f"{ep}.up_proj.weight_scale",
                scale2_key=f"{ep}.up_proj.weight_scale_2",
                device=device,
            )
            down_rows[eid] = _load_sharded_linear_weight(
                loader,
                f"{ep}.down_proj.weight",
                shard_dim=1,
                shard_offset=0,
                shard_size=I_tp,
                scale_key=f"{ep}.down_proj.weight_scale",
                scale2_key=f"{ep}.down_proj.weight_scale_2",
                device=device,
            )
        print(" done.")
        w1_weight = up_rows.contiguous()
    else:
        raise ValueError(f"unsupported checkpoint family {checkpoint_family!r}")

    return ExpertWeights(
        layer_idx=layer_idx,
        spec=spec,
        w1_weight=w1_weight,
        w2_weight=down_rows.contiguous(),
    )


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return x, routing_logits, topk_ids.to(torch.int32), topk_weights


def make_oracle_reference(
    x: torch.Tensor,
    weights: ExpertWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
) -> torch.Tensor:
    return moe_reference_bf16(
        x,
        weights.w1_weight,
        weights.w2_weight,
        topk_ids,
        topk_weights,
        activation=activation,
    )


def format_oracle_metrics(name: str, metrics: OracleMetrics) -> str:
    return (
        f"{name}: max_abs={metrics.max_abs:.5f} "
        f"rmse={metrics.rmse:.5f} "
        f"mean_abs={metrics.mean_abs:.5f} "
        f"cos={metrics.cos:.6f}"
    )


def check_oracle_metrics(label: str, metrics: OracleMetrics, batch_size: int) -> list[str]:
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


def fail_on_cosine_discrepancy(label: str, metrics: OracleMetrics, batch_size: int) -> None:
    threshold = ORACLE_TOLERANCES["cos_min"]
    if metrics.cos >= threshold:
        return
    print(f"\n\033[1;31m{'=' * 70}")
    print("  COSINE CHECK FAILED")
    print(f"{'=' * 70}")
    print(f"  bs={batch_size} {label}: cos={metrics.cos:.6f} < {threshold}")
    print(f"{'=' * 70}\033[0m")
    raise SystemExit(1)


def _resolve_bf16_backend_api() -> _BF16BackendAPI:
    import b12x.integration.tp_moe_bf16 as tp_moe_bf16

    required_names = (
        "clear_tp_moe_bf16_caches",
        "allocate_tp_moe_bf16_workspace_pool",
        "b12x_moe_bf16",
    )
    missing = [name for name in required_names if not hasattr(tp_moe_bf16, name)]
    if missing:
        raise RuntimeError(
            "benchmark_moe_bf16.py expects the BF16 public API to be wired in "
            f"b12x.integration.tp_moe_bf16. Missing: {', '.join(missing)}"
        )
    return _BF16BackendAPI(
        clear_caches=getattr(tp_moe_bf16, "clear_tp_moe_bf16_caches"),
        allocate_workspace_pool=getattr(tp_moe_bf16, "allocate_tp_moe_bf16_workspace_pool"),
        b12x_moe=getattr(tp_moe_bf16, "b12x_moe_bf16"),
    )


def _resolve_flashinfer_api() -> _FlashInferAPI:
    try:
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
    except Exception as exc:
        raise RuntimeError(
            "FlashInfer BF16 MoE is unavailable. Run this benchmark from the "
            "~/projects/sglang/.venv interpreter or install flashinfer there."
        ) from exc

    return _FlashInferAPI(
        cutlass_fused_moe=cutlass_fused_moe,
        activation_types={
            "silu": ActivationType.Swiglu,
            "relu2": ActivationType.Relu2,
        },
    )


def _resolve_sglang_python_path() -> pathlib.Path | None:
    env_path = os.environ.get("SGLANG_PYTHON_PATH")
    if env_path:
        candidate = pathlib.Path(env_path)
        if candidate.is_dir():
            return candidate

    sibling_checkout = pathlib.Path(__file__).resolve().parents[2] / "sglang" / "python"
    if sibling_checkout.is_dir():
        return sibling_checkout

    return None


def _resolve_triton_api() -> _TritonAPI:
    sglang_python_path = _resolve_sglang_python_path()
    if sglang_python_path is None:
        raise RuntimeError(
            "SGLang Triton BF16 MoE is unavailable. Set SGLANG_PYTHON_PATH or "
            "check out ~/projects/sglang next to this repo."
        )

    sglang_python_str = str(sglang_python_path)
    if sglang_python_str not in sys.path:
        sys.path.insert(0, sglang_python_str)

    try:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts
    except Exception as exc:
        raise RuntimeError(
            "SGLang Triton BF16 MoE is unavailable. Run this benchmark from the "
            "~/projects/sglang/.venv interpreter or ensure its Python dependencies "
            "are installed there."
        ) from exc

    return _TritonAPI(outplace_fused_experts=outplace_fused_experts)


def _initialize_triton_runtime(model_path: pathlib.Path) -> None:
    sglang_python_path = _resolve_sglang_python_path()
    if sglang_python_path is None:
        raise RuntimeError(
            "SGLang Triton BF16 MoE is unavailable. Set SGLANG_PYTHON_PATH or "
            "check out ~/projects/sglang next to this repo."
        )

    sglang_python_str = str(sglang_python_path)
    if sglang_python_str not in sys.path:
        sys.path.insert(0, sglang_python_str)

    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path=str(model_path)))


def _run_flashinfer_bf16(
    api: _FlashInferAPI,
    *,
    x: torch.Tensor,
    weights: ExpertWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    activation: str,
    output: torch.Tensor,
) -> torch.Tensor:
    result = api.cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        fc1_expert_weights=weights.w1_weight,
        fc2_expert_weights=weights.w2_weight,
        output_dtype=torch.bfloat16,
        quant_scales=None,
        output=output,
        activation_type=api.activation_types[activation],
    )
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


def _swap_gate_up_halves_for_sglang(w1_weight: torch.Tensor) -> torch.Tensor:
    if w1_weight.shape[1] % 2 != 0:
        raise ValueError(f"expected even fused gate_up dim, got {w1_weight.shape[1]}")
    up_weight, gate_weight = torch.chunk(w1_weight, 2, dim=1)
    return torch.cat([gate_weight, up_weight], dim=1).contiguous()


def _run_triton_bf16(
    api: _TritonAPI,
    *,
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    weights: ExpertWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    return api.outplace_fused_experts(
        x,
        w1_weight,
        weights.w2_weight,
        topk_weights,
        topk_ids.to(torch.int32),
        activation=activation,
        is_gated=(activation == "silu"),
        filter_expert=False,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size-profile", choices=sorted(BATCH_SIZE_PROFILES), default="micro")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="nemotron-backbone")
    parser.add_argument("--model-path", type=pathlib.Path, default=None)
    parser.add_argument("--validate", choices=["none", "oracle"], default="oracle")
    parser.add_argument("--include-routing", action="store_true")
    parser.add_argument("--activation", choices=["silu", "relu2"], default=None)
    parser.add_argument(
        "--providers",
        choices=["b12x", "flashinfer", "triton"],
        nargs="+",
        default=["b12x"],
    )
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument(
        "--profile-once",
        choices=["none", "backend", "b12x", "flashinfer", "triton"],
        default="none",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["graph", "eager", "both"],
        default="graph",
    )
    return parser


def bench_e2e() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    batch_sizes = (
        args.batch_sizes
        if args.batch_sizes is not None
        else BATCH_SIZE_PROFILES[args.batch_size_profile]
    )
    batch_size_profile_label = (
        f"custom -> {batch_sizes}"
        if args.batch_sizes is not None
        else f"{args.batch_size_profile} -> {batch_sizes}"
    )
    model_profile = MODEL_PROFILES[args.model_profile]
    model_path = resolve_model_path(model_profile, args.model_path)
    layer_idx = model_profile.default_layer_idx if args.layer_idx is None else args.layer_idx
    activation = resolve_activation(model_profile, args.activation)

    require_sm120()
    torch.empty(1, device="cuda")
    device = torch.device("cuda")

    spec = build_model_spec(model_path, model_profile)

    benchmark_scope = "Routing + MoE kernel" if args.include_routing else "Pre-routed MoE kernel only"
    print(f"BF16 MoE benchmark ({benchmark_scope})")
    print(
        f"{model_profile.label}  TP={spec.tp_size}, K={spec.hidden_size}, I_tp={spec.I_tp}, "
        f"E={spec.num_experts}, top_k={spec.top_k}"
    )
    print(f"Model path: {model_path}")
    print(f"Layer: {layer_idx}")
    print(f"Activation: {activation}")
    print(f"Batch-size profile: {batch_size_profile_label}")
    print(f"Validation: {args.validate}")
    print(f"Timing passes per batch size: {args.repeats} x {args.iters} iterations")
    if args.execution_mode == "both":
        timing_mode_label = "eager + CUDA graph replay"
    else:
        timing_mode_label = _timing_mode_suffix(args.execution_mode)
    print(f"Timing mode: {timing_mode_label}")
    print(f"Providers: {', '.join(args.providers)}")
    print(
        "Timed region: "
        + ("top-k + softmax routing + backend launch" if args.include_routing else "backend launch only")
    )
    if (
        args.execution_mode in {"graph", "both"}
        and model_profile.checkpoint_family == "nemotron"
        and activation == "relu2"
        and any(batch_size > 4 for batch_size in batch_sizes)
    ):
        print(
            "Note: graph replay for relu2 batches >4 is not the live Nemotron draft-graph regime "
            "(serve-nemotron3-super.sh captures bs<=4)."
        )
    print()

    use_b12x = "b12x" in args.providers
    use_flashinfer = "flashinfer" in args.providers
    use_triton = "triton" in args.providers

    b12x_api = _resolve_bf16_backend_api() if use_b12x else None
    flashinfer_api = _resolve_flashinfer_api() if use_flashinfer else None
    triton_api = _resolve_triton_api() if use_triton else None
    if triton_api is not None:
        _initialize_triton_runtime(model_path)
    weights = load_expert_weights(
        model_path,
        spec,
        layer_idx=layer_idx,
        activation=activation,
        checkpoint_family=model_profile.checkpoint_family,
    )
    if b12x_api is not None:
        b12x_api.clear_caches()

    triton_w1_weight = weights.w1_weight
    if triton_api is not None and activation == "silu":
        print("  Reordering fused gate_up weights for SGLang Triton...", end="", flush=True)
        triton_w1_weight = _swap_gate_up_halves_for_sglang(weights.w1_weight)
        torch.cuda.synchronize()
        print(" done.")

    x_warm, _, topk_ids_w, topk_weights_w = make_routed_inputs(spec, 1, 42, device)
    if b12x_api is not None:
        print("  Warming up b12x BF16 (compilation)...", end="", flush=True)
        warmup_workspace = b12x_api.allocate_workspace_pool()
        warmup_output = torch.empty_like(x_warm)
        b12x_api.b12x_moe(
            x_warm,
            weights.w1_weight,
            weights.w2_weight,
            topk_weights_w,
            topk_ids_w,
            workspace=warmup_workspace,
            output=warmup_output,
            activation=activation,
        )
        torch.cuda.synchronize()
        print(" done.")

    if flashinfer_api is not None:
        print("  Warming up FlashInfer BF16 (compilation)...", end="", flush=True)
        flashinfer_output = torch.empty_like(x_warm)
        try:
            _run_flashinfer_bf16(
                flashinfer_api,
                x=x_warm,
                weights=weights,
                topk_ids=topk_ids_w,
                topk_weights=topk_weights_w,
                activation=activation,
                output=flashinfer_output,
            )
            torch.cuda.synchronize()
            print(" done.")
        except Exception as exc:
            message = str(exc).splitlines()[0]
            print(f" skipped ({type(exc).__name__}: {message})")
            flashinfer_api = None

    if triton_api is not None:
        print("  Warming up SGLang Triton BF16 (compilation)...", end="", flush=True)
        try:
            _run_triton_bf16(
                triton_api,
                x=x_warm,
                w1_weight=triton_w1_weight,
                weights=weights,
                topk_ids=topk_ids_w,
                topk_weights=topk_weights_w,
                activation=activation,
            )
            torch.cuda.synchronize()
            print(" done.")
        except Exception as exc:
            message = str(exc).splitlines()[0]
            print(f" skipped ({type(exc).__name__}: {message})")
            triton_api = None

    accuracy_failures: list[str] = []
    b12x_latency_ms = {
        mode: {}
        for mode in ("eager", "graph")
        if args.execution_mode in {mode, "both"}
    }
    flashinfer_latency_ms = {
        mode: {}
        for mode in ("eager", "graph")
        if args.execution_mode in {mode, "both"}
    }
    triton_latency_ms = {
        mode: {}
        for mode in ("eager", "graph")
        if args.execution_mode in {mode, "both"}
    }
    for batch_size in batch_sizes:
        print(f"\n{'=' * 70}")
        print(f"  batch_size={batch_size}  (tokens*top_k = {batch_size * spec.top_k} expert calls)")
        print(f"{'=' * 70}")

        x, routing_logits, topk_ids, topk_weights = make_routed_inputs(spec, batch_size, 1000 + batch_size, device)
        b12x_workspace = b12x_api.allocate_workspace_pool() if b12x_api is not None else None
        b12x_output = (
            torch.empty(batch_size, spec.hidden_size, dtype=torch.bfloat16, device=device)
            if b12x_api is not None
            else None
        )
        flashinfer_output = None
        if flashinfer_api is not None:
            reference_output = b12x_output if b12x_output is not None else x
            flashinfer_output = torch.empty_like(reference_output)

        def b12x_launch(topk_ids_local: torch.Tensor, topk_weights_local: torch.Tensor) -> torch.Tensor:
            assert b12x_api is not None
            assert b12x_workspace is not None
            assert b12x_output is not None
            return b12x_api.b12x_moe(
                x,
                weights.w1_weight,
                weights.w2_weight,
                topk_weights_local,
                topk_ids_local,
                workspace=b12x_workspace,
                output=b12x_output,
                activation=activation,
            )

        def flashinfer_launch(
            topk_ids_local: torch.Tensor,
            topk_weights_local: torch.Tensor,
        ) -> torch.Tensor:
            assert flashinfer_api is not None
            assert flashinfer_output is not None
            return _run_flashinfer_bf16(
                flashinfer_api,
                x=x,
                weights=weights,
                topk_ids=topk_ids_local,
                topk_weights=topk_weights_local,
                activation=activation,
                output=flashinfer_output,
            )

        def triton_launch(topk_ids_local: torch.Tensor, topk_weights_local: torch.Tensor) -> torch.Tensor:
            assert triton_api is not None
            return _run_triton_bf16(
                triton_api,
                x=x,
                w1_weight=triton_w1_weight,
                weights=weights,
                topk_ids=topk_ids_local,
                topk_weights=topk_weights_local,
                activation=activation,
            )

        def b12x_e2e() -> torch.Tensor:
            if args.include_routing:
                timed_topk_logits, timed_topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
                timed_topk_weights = torch.softmax(timed_topk_logits, dim=-1)
                return b12x_launch(timed_topk_ids.to(torch.int32), timed_topk_weights)
            return b12x_launch(topk_ids, topk_weights)

        def flashinfer_e2e() -> torch.Tensor:
            if args.include_routing:
                timed_topk_logits, timed_topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
                timed_topk_weights = torch.softmax(timed_topk_logits, dim=-1)
                return flashinfer_launch(timed_topk_ids.to(torch.int32), timed_topk_weights)
            return flashinfer_launch(topk_ids, topk_weights)

        def triton_e2e() -> torch.Tensor:
            if args.include_routing:
                timed_topk_logits, timed_topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
                timed_topk_weights = torch.softmax(timed_topk_logits, dim=-1)
                return triton_launch(timed_topk_ids.to(torch.int32), timed_topk_weights)
            return triton_launch(topk_ids, topk_weights)

        def _measure_timing(
            *,
            provider_label: str,
            fn: Callable[[], torch.Tensor],
            execution_mode: str,
            latencies_ms: dict[int, float],
        ) -> None:
            print(
                f"  {provider_label} ({_timing_mode_suffix(execution_mode)}):".ljust(28),
                end="",
                flush=True,
            )
            try:
                if execution_mode == "eager":
                    times = bench_events(fn, warmup=args.warmup, iters=args.iters)
                elif execution_mode == "graph":
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fn()

                    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
                        g.replay()

                    times = bench_events(replay, warmup=args.warmup, iters=args.iters)
                else:
                    raise ValueError(f"unsupported execution_mode={execution_mode!r}")
                latencies_ms[batch_size] = statistics.median(times)
                print(f" {statistics.median(times) * 1000.0:8.1f} us (min {min(times) * 1000.0:.1f})")
            except Exception as exc:
                print(f" FAILED ({type(exc).__name__}: {exc})")

        oracle_ref = None
        if args.validate == "oracle":
            oracle_ref = make_oracle_reference(
                x,
                weights,
                topk_ids,
                topk_weights,
                activation=activation,
            )
            print(
                "  oracle:".ljust(28),
                f"norm={oracle_ref.float().norm().item():.5f}",
                f"max={oracle_ref.float().abs().max().item():.5f}",
            )

        b12x_out = b12x_e2e().clone() if b12x_api is not None else None
        flashinfer_out = flashinfer_e2e().clone() if flashinfer_api is not None else None
        triton_out = triton_e2e().clone() if triton_api is not None else None
        torch.cuda.synchronize()

        if oracle_ref is not None:
            if b12x_out is not None:
                b12x_metrics = compare_to_reference(b12x_out, oracle_ref)
                print(f"  {format_oracle_metrics('b12x vs oracle', b12x_metrics)}")
                fail_on_cosine_discrepancy("b12x vs oracle", b12x_metrics, batch_size)
                accuracy_failures.extend(check_oracle_metrics("b12x vs oracle", b12x_metrics, batch_size))
            if flashinfer_out is not None:
                flashinfer_metrics = compare_to_reference(flashinfer_out, oracle_ref)
                print(f"  {format_oracle_metrics('flashinfer vs oracle', flashinfer_metrics)}")
                fail_on_cosine_discrepancy("flashinfer vs oracle", flashinfer_metrics, batch_size)
                accuracy_failures.extend(
                    check_oracle_metrics("flashinfer vs oracle", flashinfer_metrics, batch_size)
                )
            if triton_out is not None:
                triton_metrics = compare_to_reference(triton_out, oracle_ref)
                print(f"  {format_oracle_metrics('triton vs oracle', triton_metrics)}")
                fail_on_cosine_discrepancy("triton vs oracle", triton_metrics, batch_size)
                accuracy_failures.extend(check_oracle_metrics("triton vs oracle", triton_metrics, batch_size))
        if b12x_out is not None and flashinfer_out is not None:
            cross_metrics = compare_to_reference(b12x_out, flashinfer_out)
            print(f"  {format_oracle_metrics('b12x vs flashinfer', cross_metrics)}")
            fail_on_cosine_discrepancy("b12x vs flashinfer", cross_metrics, batch_size)
        if b12x_out is not None and triton_out is not None:
            cross_metrics = compare_to_reference(b12x_out, triton_out)
            print(f"  {format_oracle_metrics('b12x vs triton', cross_metrics)}")
            fail_on_cosine_discrepancy("b12x vs triton", cross_metrics, batch_size)

        if args.profile_once != "none":
            if args.execution_mode == "both":
                raise RuntimeError("--profile-once requires --execution-mode eager or graph")

            profile_target = "b12x" if args.profile_once == "backend" else args.profile_once
            profile_fns = {
                "b12x": b12x_e2e if b12x_api is not None else None,
                "flashinfer": flashinfer_e2e if flashinfer_api is not None else None,
                "triton": triton_e2e if triton_api is not None else None,
            }
            profile_fn = profile_fns.get(profile_target)
            if profile_fn is None:
                raise RuntimeError(
                    f"--profile-once {profile_target} requires --providers to include {profile_target}"
                )

            if args.execution_mode == "graph":
                for _ in range(3):
                    profile_fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    profile_fn()

                def profiled_launch(g: torch.cuda.CUDAGraph = graph) -> None:
                    g.replay()

            else:
                profiled_launch = profile_fn

            print(f"  profiling once: {profile_target} BF16 ({args.execution_mode})")
            torch.cuda.synchronize()
            cudart = torch.cuda.cudart()
            cudart.cudaProfilerStart()
            profiled_launch()
            torch.cuda.synchronize()
            cudart.cudaProfilerStop()
            print("  profiler range complete")
            return

        if b12x_api is not None:
            for execution_mode in b12x_latency_ms:
                _measure_timing(
                    provider_label="b12x BF16",
                    fn=b12x_e2e,
                    execution_mode=execution_mode,
                    latencies_ms=b12x_latency_ms[execution_mode],
                )

        if flashinfer_api is not None:
            for execution_mode in flashinfer_latency_ms:
                _measure_timing(
                    provider_label="FlashInfer BF16",
                    fn=flashinfer_e2e,
                    execution_mode=execution_mode,
                    latencies_ms=flashinfer_latency_ms[execution_mode],
                )

        if triton_api is not None:
            for execution_mode in triton_latency_ms:
                _measure_timing(
                    provider_label="Triton BF16",
                    fn=triton_e2e,
                    execution_mode=execution_mode,
                    latencies_ms=triton_latency_ms[execution_mode],
                )

    if b12x_latency_ms or flashinfer_latency_ms or triton_latency_ms:
        print(f"\n{'=' * 70}")
        print("  Summary")
        print(f"{'=' * 70}")
        for execution_mode in ("eager", "graph"):
            if not any(
                execution_mode in latency_map and latency_map[execution_mode]
                for latency_map in (b12x_latency_ms, flashinfer_latency_ms, triton_latency_ms)
            ):
                continue
            print(f"  [{_timing_mode_suffix(execution_mode)}]")
            for batch_size in sorted(
                set(b12x_latency_ms.get(execution_mode, {}))
                | set(flashinfer_latency_ms.get(execution_mode, {}))
                | set(triton_latency_ms.get(execution_mode, {}))
            ):
                parts = [f"bs={batch_size}"]
                if batch_size in b12x_latency_ms.get(execution_mode, {}):
                    latency_ms = b12x_latency_ms[execution_mode][batch_size]
                    parts.append(
                        "b12x "
                        f"{latency_ms * 1000.0:.1f} us "
                        f"(norm {normalized_batch_latency_us(latency_ms, batch_size):.1f} us/token)"
                    )
                if batch_size in flashinfer_latency_ms.get(execution_mode, {}):
                    latency_ms = flashinfer_latency_ms[execution_mode][batch_size]
                    parts.append(
                        "flashinfer "
                        f"{latency_ms * 1000.0:.1f} us "
                        f"(norm {normalized_batch_latency_us(latency_ms, batch_size):.1f} us/token)"
                    )
                if batch_size in triton_latency_ms.get(execution_mode, {}):
                    latency_ms = triton_latency_ms[execution_mode][batch_size]
                    parts.append(
                        "triton "
                        f"{latency_ms * 1000.0:.1f} us "
                        f"(norm {normalized_batch_latency_us(latency_ms, batch_size):.1f} us/token)"
                    )
                print("  " + " | ".join(parts))

            if b12x_latency_ms.get(execution_mode):
                print(
                    f"  geomean result ({execution_mode}, b12x): "
                    f"{normalized_result_us(b12x_latency_ms[execution_mode]):.1f} us/token"
                )
            if flashinfer_latency_ms.get(execution_mode):
                print(
                    f"  geomean result ({execution_mode}, flashinfer): "
                    f"{normalized_result_us(flashinfer_latency_ms[execution_mode]):.1f} us/token"
                )
            if triton_latency_ms.get(execution_mode):
                print(
                    f"  geomean result ({execution_mode}, triton): "
                    f"{normalized_result_us(triton_latency_ms[execution_mode]):.1f} us/token"
                )

    if accuracy_failures:
        print(f"\n\033[1;31m{'=' * 70}")
        print("  ACCURACY CHECK FAILED")
        print(f"{'=' * 70}")
        for failure in accuracy_failures:
            print(failure)
        print(f"{'=' * 70}\033[0m")
        sys.exit(1)


def main() -> None:
    bench_e2e()


if __name__ == "__main__":
    main()
