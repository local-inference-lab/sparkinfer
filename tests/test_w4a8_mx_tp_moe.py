"""End-to-end w4a8_mx dispatch through b12x_moe_fp4 on synthetic MXFP4 weights.

Gates the e8m0_k32 serving prepare: checkpoint-native per-K/32 E8M0 grids
([E, rows, K//32] bytes) feed the dynamic w4a8_mx kernel directly — no vec16
scale stack, no residual grids. The micro (tiny-decode) band is not wired for
MXFP4 sources yet and must fail loudly rather than misread the grids.
"""

from __future__ import annotations

import functools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe

_E = 16
_K = 4096
_N = 1024
_TOPK = 4


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")


@functools.lru_cache(maxsize=1)
def _weights():
    return make_synthetic_mxfp4_moe(
        _E, _K, _N, seed=21, device=torch.device("cuda")
    )


def _routed_inputs(m: int, seed: int):
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = (torch.randn(m, _K, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(m, _E, generator=gen, device=device)
    topk_logits, topk_ids = torch.topk(logits, _TOPK, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).float()
    return x, topk_ids.to(torch.int32), topk_weights


def _run(
    m: int,
    w13_fp4: torch.Tensor,
    w13_mx: torch.Tensor,
    *,
    w13_layout: str = "w13",
    seed: int = 33,
) -> torch.Tensor:
    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    weights = _weights()
    x, topk_ids, topk_weights = _routed_inputs(m, seed)
    workspace = allocate_tp_moe_workspace(
        x,
        weights["input_scale"],
        w13_fp4,
        weights["input_scale"],
        weights["w2_fp4"],
        topk_ids,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )
    out = b12x_moe_fp4(
        x,
        weights["input_scale"],
        w13_fp4,
        w13_mx,
        weights["alphas"],
        weights["input_scale"],
        weights["w2_fp4"],
        weights["w2_mx"],
        weights["alphas"],
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        quant_mode="w4a8_mx",
        w13_layout=w13_layout,
    )
    torch.cuda.synchronize()
    return out


def test_w4a8_mx_dynamic_matches_oracle() -> None:
    _skip_if_unavailable()
    from b12x.moe.fused.reference import moe_reference_w4a8_mx

    weights = _weights()
    m = 16
    out = _run(m, weights["w13_fp4"], weights["w13_mx"])
    x, topk_ids, topk_weights = _routed_inputs(m, 33)

    ref = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        _N,
        activation="silu",
    )
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"w4a8_mx output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


def test_w4a8_mx_w31_layout_flip() -> None:
    _skip_if_unavailable()
    weights = _weights()
    m = 16
    baseline = _run(m, weights["w13_fp4"], weights["w13_mx"])
    repeat = _run(m, weights["w13_fp4"], weights["w13_mx"])

    u8 = weights["w13_fp4"]
    w13_w31 = torch.cat([u8[:, _N:], u8[:, :_N]], dim=1).contiguous()
    mx_w31 = torch.cat(
        [weights["w13_mx"][:, _N:], weights["w13_mx"][:, :_N]], dim=1
    ).contiguous()
    flipped = _run(m, w13_w31, mx_w31, w13_layout="w31")
    # Idempotency: a second pass over the same storage must not re-flip.
    flipped2 = _run(m, w13_w31, mx_w31, w13_layout="w31")

    noise = (baseline.float() - repeat.float()).abs().max().item()
    bound = max(8.0 * noise, 1e-6 * baseline.float().abs().max().item())
    for label, got in (("first", flipped), ("repeat", flipped2)):
        err = (got.float() - baseline.float()).abs().max().item()
        assert err <= bound, (
            f"w31 flip mismatch ({label}): err={err} noise={noise} bound={bound}"
        )


def test_w4a8_mx_micro_band_raises() -> None:
    _skip_if_unavailable()
    weights = _weights()
    with pytest.raises(NotImplementedError, match="w4a8_mx tiny-decode"):
        _run(4, weights["w13_fp4"], weights["w13_mx"])


# ---------------------------------------------------------------------------
# W4A8 throughput-tier dispatch (routed_rows >= B12X_W4A8_TIER_MIN_ROUTED_ROWS)
# ---------------------------------------------------------------------------


def _tier_dispatched() -> bool:
    from b12x.integration.tp_moe import _W4A8_TIER_WORKSPACE_CACHE

    return len(_W4A8_TIER_WORKSPACE_CACHE) > 0


def _oracle(m: int, weights: dict, seed: int = 33, *, n: int = _N):
    from b12x.moe.fused.reference import moe_reference_w4a8_mx

    x, topk_ids, topk_weights = _routed_inputs(m, seed)
    return moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="silu",
    )


def test_w4a8_mx_tier_band_matches_oracle() -> None:
    """m=1024 x topk=4 = 4096 routed rows = the default tier floor: the call
    must auto-dispatch to the throughput tier and still match the oracle."""
    _skip_if_unavailable()
    weights = _weights()
    m = 1024
    out = _run(m, weights["w13_fp4"], weights["w13_mx"])
    assert _tier_dispatched(), "expected the w4a8 tier to serve 4096 routed rows"
    ref = _oracle(m, weights)
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"tier output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


def test_w4a8_mx_tier_matches_dynamic(monkeypatch) -> None:
    """Forced-tier vs forced-dynamic on identical inputs: within 8x the
    dynamic path's run-to-run noise envelope (pattern from
    tests/test_tp_moe_w13_layout.py). Note threshold 0 DISABLES the tier;
    a huge threshold forces the dynamic kernel."""
    _skip_if_unavailable()
    weights = _weights()
    m = 256  # routed 1024: dynamic band, below the default tier floor

    monkeypatch.setenv("B12X_W4A8_TIER_MIN_ROUTED_ROWS", str(1 << 30))
    monkeypatch.delenv("B12X_MOE_FORCE_W4A8_TIER", raising=False)
    dyn1 = _run(m, weights["w13_fp4"], weights["w13_mx"])
    assert not _tier_dispatched()
    dyn2 = _run(m, weights["w13_fp4"], weights["w13_mx"])

    monkeypatch.delenv("B12X_W4A8_TIER_MIN_ROUTED_ROWS", raising=False)
    monkeypatch.setenv("B12X_MOE_FORCE_W4A8_TIER", "1")
    tier = _run(m, weights["w13_fp4"], weights["w13_mx"])
    assert _tier_dispatched(), "expected the forced w4a8 tier dispatch"

    noise = (dyn1.float() - dyn2.float()).abs().max().item()
    err = (tier.float() - dyn1.float()).abs().max().item()
    bound = max(8.0 * noise, 1e-6 * dyn1.float().abs().max().item())
    assert err <= bound, f"tier vs dynamic mismatch: err={err} noise={noise} bound={bound}"


def test_w4a8_mx_tier_graph_replay_tracks_routing_updates(monkeypatch) -> None:
    """Capture the tier path in a CUDA graph (caller-owned output buffer);
    replay with routing/activations mutated IN PLACE must track the update
    (vs a fresh eager call on the same inputs, and vs the oracle)."""
    _skip_if_unavailable()
    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )

    monkeypatch.setenv("B12X_MOE_FORCE_W4A8_TIER", "1")
    monkeypatch.delenv("B12X_W4A8_TIER_MIN_ROUTED_ROWS", raising=False)
    clear_tp_moe_caches()
    device = torch.device("cuda")
    weights = _weights()
    m = 256
    x, topk_ids, topk_weights = _routed_inputs(m, 33)
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    graph_out = torch.zeros(m, _K, dtype=torch.bfloat16, device=device)
    workspace = allocate_tp_moe_workspace(
        x,
        weights["input_scale"],
        weights["w13_fp4"],
        weights["input_scale"],
        weights["w2_fp4"],
        topk_ids,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )

    def _launch(out: torch.Tensor) -> None:
        b12x_moe_fp4(
            x,
            weights["input_scale"],
            weights["w13_fp4"],
            weights["w13_mx"],
            weights["alphas"],
            weights["input_scale"],
            weights["w2_fp4"],
            weights["w2_mx"],
            weights["alphas"],
            topk_weights,
            topk_ids,
            workspace=workspace,
            output=out,
            input_scales_static=True,
            quant_mode="w4a8_mx",
        )

    # Warm (compiles cute + triton, builds the tier workspace), then capture.
    _launch(graph_out)
    torch.cuda.synchronize()
    assert _tier_dispatched(), "expected the forced w4a8 tier dispatch"

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            _launch(graph_out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    from b12x.moe.fused.reference import moe_reference_w4a8_mx

    gen = torch.Generator(device=device)
    for round_idx in range(3):
        gen.manual_seed(500 + round_idx)
        new_x, new_ids, new_w = _routed_inputs(m, 500 + round_idx)
        x.copy_(new_x)
        topk_ids.copy_(new_ids)
        topk_weights.copy_(new_w)
        graph.replay()
        torch.cuda.synchronize()
        replayed = graph_out.clone()

        eager_out = torch.zeros_like(graph_out)
        _launch(eager_out)
        torch.cuda.synchronize()

        assert replayed.abs().sum().item() > 0, round_idx
        # The tier pipeline is deterministic (fixed j-order weighted sum), so
        # replay and a fresh eager call on identical inputs must agree bitwise.
        assert torch.equal(replayed, eager_out), (
            round_idx,
            (replayed.float() - eager_out.float()).abs().max().item(),
        )
        ref = moe_reference_w4a8_mx(
            x.float(),
            weights["w13_fp4"],
            weights["w13_mx"],
            None,
            weights["alphas"],
            weights["w2_fp4"],
            weights["w2_mx"],
            None,
            weights["alphas"],
            topk_ids,
            topk_weights,
            _E,
            _K,
            _N,
            activation="silu",
        )
        cos = torch.nn.functional.cosine_similarity(
            replayed.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        assert cos > 0.998, (round_idx, cos)


def test_w4a8_mx_tier_glm_shard_geometry(monkeypatch) -> None:
    """GLM per-rank shard geometry: E=16, K=4096, n=256 (FC1 N=512, FC2
    N=4096 with K=256) through the forced tier, gated vs the oracle."""
    _skip_if_unavailable()
    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )
    from b12x.moe.fused.reference import moe_reference_w4a8_mx

    monkeypatch.setenv("B12X_MOE_FORCE_W4A8_TIER", "1")
    monkeypatch.delenv("B12X_W4A8_TIER_MIN_ROUTED_ROWS", raising=False)
    clear_tp_moe_caches()
    n = 256
    weights = make_synthetic_mxfp4_moe(
        _E, _K, n, seed=77, device=torch.device("cuda")
    )
    m = 256
    x, topk_ids, topk_weights = _routed_inputs(m, 44)
    workspace = allocate_tp_moe_workspace(
        x,
        weights["input_scale"],
        weights["w13_fp4"],
        weights["input_scale"],
        weights["w2_fp4"],
        topk_ids,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )
    out = b12x_moe_fp4(
        x,
        weights["input_scale"],
        weights["w13_fp4"],
        weights["w13_mx"],
        weights["alphas"],
        weights["input_scale"],
        weights["w2_fp4"],
        weights["w2_mx"],
        weights["alphas"],
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )
    torch.cuda.synchronize()
    assert _tier_dispatched(), "expected the forced w4a8 tier dispatch"
    ref = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="silu",
    )
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"tier output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)
