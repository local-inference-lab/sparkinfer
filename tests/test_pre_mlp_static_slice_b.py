from __future__ import annotations

import functools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_moe import (
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    ModelSpec,
    load_gate_weight,
    load_shared_gate_weight,
    make_input_activations,
)
from b12x.integration.tp_moe import _shared_expert_gate_weights, b12x_route_experts_fast
from b12x.moe.fused.pre_mlp_static import (
    UnifiedPreMLPStaticLaunchConfig,
    slice_b_sparse_routing_shared_gate,
)
from b12x.moe.fused.reference import compare_to_reference


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        pytest.skip(f"Requires SM120, got sm_{major}{minor}")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


@functools.lru_cache(maxsize=1)
def _load_gate_weights():
    spec = _make_spec()
    return (
        load_gate_weight(MODEL_PATH, spec, layer_idx=0),
        load_shared_gate_weight(MODEL_PATH, layer_idx=0),
    )


def test_slice_b_matches_sparse_routing_and_shared_gate() -> None:
    _skip_if_unavailable()
    spec = _make_spec()
    sparse_gate_weight, shared_gate_weight = _load_gate_weights()
    normalized = make_input_activations(spec, 4, seed=6100, device=torch.device("cuda"))

    launch_config = UnifiedPreMLPStaticLaunchConfig()
    slice_b = slice_b_sparse_routing_shared_gate(
        normalized,
        sparse_gate_weight,
        shared_gate_weight,
        top_k=spec.top_k,
        launch_config=launch_config,
    )
    ref_routing = b12x_route_experts_fast(
        normalized,
        top_k=spec.top_k,
        gate_weight=sparse_gate_weight,
    )
    ref_shared = _shared_expert_gate_weights(
        normalized,
        gate_weight=shared_gate_weight,
    )
    torch.cuda.synchronize()

    assert torch.equal(slice_b.topk_ids, ref_routing.topk_ids)
    router_metrics = compare_to_reference(slice_b.router_logits, ref_routing.router_logits)
    topk_weight_metrics = compare_to_reference(slice_b.topk_weights, ref_routing.topk_weights)
    shared_gate_metrics = compare_to_reference(slice_b.shared_gate, ref_shared)
    assert router_metrics.cos > 0.999
    assert topk_weight_metrics.cos > 0.999
    assert shared_gate_metrics.cos > 0.999
