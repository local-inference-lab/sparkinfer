from __future__ import annotations

import torch

import b12x.integration.tp_moe as tp_moe
from b12x.integration._silu_dsv4_decode import is_exact_silu_dsv4_case
from b12x.integration.tp_moe import B12XFP4ExpertWeights, B12XTopKRouting


def test_dsv4_silu_exact_gate_accepts_tp4_decode_meta_tensors() -> None:
    a = torch.empty((8, 4096), dtype=torch.bfloat16, device="meta")
    w1 = torch.empty((64, 4096, 2048), dtype=torch.uint8, device="meta")
    w2 = torch.empty((64, 4096, 1024), dtype=torch.uint8, device="meta")
    topk_ids = torch.empty((8, 6), dtype=torch.int32, device="meta")
    topk_weights = torch.empty((8, 6), dtype=torch.float32, device="meta")
    scale = torch.empty((), dtype=torch.float32, device="meta")

    assert is_exact_silu_dsv4_case(
        activation="silu",
        a=a,
        w1_fp4=w1,
        w2_fp4=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        a1_gscale=scale,
        a2_gscale=scale,
    )

    assert not is_exact_silu_dsv4_case(
        activation="silu",
        a=torch.empty((9, 4096), dtype=torch.bfloat16, device="meta"),
        w1_fp4=w1,
        w2_fp4=w2,
        topk_weights=torch.empty((9, 6), dtype=torch.float32, device="meta"),
        topk_ids=torch.empty((9, 6), dtype=torch.int32, device="meta"),
        a1_gscale=scale,
        a2_gscale=scale,
    )


def test_b12x_moe_fp4_uses_dsv4_direct_before_generic_planning(monkeypatch) -> None:
    captured = {}

    def fake_direct(**kwargs):
        captured.update(kwargs)
        return torch.full_like(kwargs["a"], 3.0)

    monkeypatch.setattr(tp_moe, "_dsv4_silu_direct_enabled", lambda: True)
    monkeypatch.setattr(tp_moe, "is_exact_silu_dsv4_case", lambda **_kwargs: True)
    monkeypatch.setattr(tp_moe, "_launch_exact_silu_dsv4_decode", fake_direct)

    a = torch.randn(1, 4, dtype=torch.bfloat16)
    out = tp_moe.b12x_moe_fp4(
        a,
        torch.ones(()),
        torch.zeros((2, 4, 2), dtype=torch.uint8),
        torch.zeros((2, 1), dtype=torch.uint8),
        torch.ones(2),
        torch.ones(()),
        torch.zeros((2, 4, 1), dtype=torch.uint8),
        torch.zeros((2, 1), dtype=torch.uint8),
        torch.ones(2),
        torch.ones((1, 2)),
        torch.zeros((1, 2), dtype=torch.int32),
        workspace=object(),
    )

    torch.testing.assert_close(out, torch.full_like(a, 3.0))
    assert captured["input_scales_static"] is True


def test_sparse_wrapper_passes_preflattened_routing_to_dsv4_direct(monkeypatch) -> None:
    captured = {}

    def fake_direct(**kwargs):
        captured.update(kwargs)
        return torch.full_like(kwargs["a"], 5.0)

    monkeypatch.setattr(tp_moe, "_dsv4_silu_direct_enabled", lambda: True)
    monkeypatch.setattr(tp_moe, "is_exact_silu_dsv4_case", lambda **_kwargs: True)
    monkeypatch.setattr(tp_moe, "_launch_exact_silu_dsv4_decode", fake_direct)

    experts = B12XFP4ExpertWeights(
        a1_gscale=torch.ones(()),
        w1_fp4=torch.zeros((2, 4, 2), dtype=torch.uint8),
        w1_blockscale=torch.zeros((2, 1), dtype=torch.uint8),
        w1_alphas=torch.ones(2),
        a2_gscale=torch.ones(()),
        w2_fp4=torch.zeros((2, 4, 1), dtype=torch.uint8),
        w2_blockscale=torch.zeros((2, 1), dtype=torch.uint8),
        w2_alphas=torch.ones(2),
    )
    topk_ids = torch.tensor([[1, 0]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.75, 0.25]], dtype=torch.float32)
    routing = B12XTopKRouting(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        flat_ids=topk_ids.view(-1),
        flat_weights=topk_weights.view(-1),
    )
    hidden = torch.randn(1, 4, dtype=torch.bfloat16)

    out = tp_moe.b12x_sparse_moe_fp4(
        hidden,
        experts=experts,
        workspace=object(),
        routing=routing,
    )

    torch.testing.assert_close(out, torch.full_like(hidden, 5.0))
    assert captured["flat_ids"] is routing.flat_ids
    assert captured["flat_weights"] is routing.flat_weights
