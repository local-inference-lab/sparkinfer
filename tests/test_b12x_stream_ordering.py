from __future__ import annotations

import contextlib
from typing import Callable, Iterator

import cuda.bindings.driver as cuda
import pytest
import torch

import b12x.integration.tp_moe as tp_moe
from benchmarks.benchmark_moe import MODEL_PATH, TP_RANK, TP_SIZE, ModelSpec, bench_flashinfer, load_expert_weights
from b12x.integration.tp_moe import allocate_tp_moe_workspace, b12x_moe_fp4, clear_tp_moe_caches
from b12x.moe.fused.reference import compare_to_reference

from .helpers import require_sm120


def _require_model_weights() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _driver_stream(stream: torch.cuda.Stream) -> cuda.CUstream:
    return cuda.CUstream(int(stream.cuda_stream))


@contextlib.contextmanager
def _override_launch_stream(
    provider: Callable[[], cuda.CUstream] | None,
) -> Iterator[None]:
    if provider is None:
        yield
        return
    original = tp_moe.current_cuda_stream
    tp_moe.current_cuda_stream = provider
    try:
        yield
    finally:
        tp_moe.current_cuda_stream = original


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


def _run_b12x(
    *,
    x: torch.Tensor,
    weights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    workspace,
) -> torch.Tensor:
    return b12x_moe_fp4(
        a=x,
        a1_gscale=weights.w13_input_scale_quant,
        w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled,
        w1_alphas=weights.g1_alphas,
        a2_gscale=weights.w2_input_scale_quant,
        w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled,
        w2_alphas=weights.g2_alphas,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        workspace=workspace,
    )


def _launch_on_stream(
    *,
    launch_stream: torch.cuda.Stream,
    x_src: torch.Tensor,
    topk_ids_src: torch.Tensor,
    topk_weights_src: torch.Tensor,
    weights,
    workspace,
    force_stream: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_stage = torch.zeros_like(x_src)
    topk_ids_stage = torch.zeros_like(topk_ids_src)
    topk_weights_stage = torch.zeros_like(topk_weights_src)

    provider: Callable[[], cuda.CUstream] | None
    if force_stream == "current":
        provider = None
    elif force_stream == "default":
        default_stream = torch.cuda.default_stream(x_src.device)
        provider = lambda: _driver_stream(default_stream)
    else:
        raise ValueError(f"Unsupported force_stream: {force_stream}")

    with torch.cuda.stream(launch_stream):
        x_stage.copy_(x_src, non_blocking=True)
        topk_ids_stage.copy_(topk_ids_src, non_blocking=True)
        topk_weights_stage.copy_(topk_weights_src, non_blocking=True)
        with _override_launch_stream(provider):
            out_alias = _run_b12x(
                x=x_stage,
                weights=weights,
                topk_ids=topk_ids_stage,
                topk_weights=topk_weights_stage,
                workspace=workspace,
            )
            eager_clone = out_alias.clone()
    launch_stream.synchronize()
    settled_clone = out_alias.clone()
    return eager_clone, settled_clone


def test_b12x_uses_current_cuda_stream() -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)

    torch.manual_seed(123)
    x = torch.randn(8, spec.hidden_size, dtype=torch.bfloat16, device=device)
    routing = torch.randn(8, spec.num_experts, dtype=torch.float32, device=device)
    topk_logits, topk_ids = torch.topk(routing, spec.top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_quant,
        weights.w13_weight,
        weights.w2_input_scale_quant,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )

    fi_launch, fi_output = bench_flashinfer(weights, x, topk_ids, topk_weights)
    fi_launch()
    torch.cuda.synchronize(device)
    fi_output = fi_output.clone()

    baseline_alias = _run_b12x(
        x=x,
        weights=weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        workspace=workspace,
    )
    baseline = baseline_alias.clone()
    torch.cuda.synchronize(device)

    test_stream = torch.cuda.Stream(device=device, priority=0)
    current_eager, current_settled = _launch_on_stream(
        launch_stream=test_stream,
        x_src=x,
        topk_ids_src=topk_ids,
        topk_weights_src=topk_weights,
        weights=weights,
        workspace=workspace,
        force_stream="current",
    )
    _forced_default_eager, forced_default_settled = _launch_on_stream(
        launch_stream=test_stream,
        x_src=x,
        topk_ids_src=topk_ids,
        topk_weights_src=topk_weights,
        weights=weights,
        workspace=workspace,
        force_stream="default",
    )

    for candidate in (baseline, current_eager, current_settled):
        metrics = compare_to_reference(candidate, fi_output)
        assert metrics.max_abs < 5e-4
        assert metrics.cos > 0.9999

    bad_settled = compare_to_reference(forced_default_settled, fi_output)
    assert bad_settled.max_abs < 5e-4
    assert bad_settled.cos > 0.9999
