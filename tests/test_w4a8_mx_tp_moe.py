"""End-to-end w4a8_mx dispatch through b12x_moe_fp4 on synthetic MXFP4 weights.

Gates the e8m0_k32 serving prepare: checkpoint-native per-K/32 E8M0 grids
([E, rows, K//32] bytes) feed the w4a8_mx kernels directly — no vec16 scale
stack, no residual grids. Tiny decode uses compile-time direct regimes inside
the unified dynamic kernel and consumes only its prepared weight layout.
"""

from __future__ import annotations

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


@pytest.mark.parametrize(
    ("routed_rows", "expected_tile_m"),
    [
        (16 * _E, 16),
        (16 * _E + 1, 32),
        (36 * _E - 1, 32),
        (36 * _E, 64),
    ],
)
def test_w4a8_mx_dynamic_tile_density_boundaries(
    monkeypatch, routed_rows: int, expected_tile_m: int
) -> None:
    from b12x.integration import tp_moe

    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    assert tp_moe._select_dynamic_tile_mn(
        routed_rows,
        _N,
        "w4a8_mx",
        num_experts=_E,
        activation="silu",
    ) == (expected_tile_m, 128)


@pytest.mark.parametrize("routed_rows", [384, 768, 1536, 2304])
def test_w4a8_mx_ds4_tp2_batch_m_uses_m32(
    monkeypatch, routed_rows: int
) -> None:
    from b12x.integration import tp_moe

    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    assert tp_moe._select_dynamic_tile_mn(
        routed_rows,
        1024,
        "w4a8_mx",
        num_experts=256,
        activation="silu",
    ) == (32, 128)


@pytest.mark.parametrize("routed_rows", [383, 2305])
def test_w4a8_mx_ds4_tp2_batch_m_tactic_is_band_limited(
    monkeypatch, routed_rows: int
) -> None:
    from b12x.integration import tp_moe

    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    assert tp_moe._select_dynamic_tile_mn(
        routed_rows,
        1024,
        "w4a8_mx",
        num_experts=256,
        activation="silu",
    ) == (16, 128)


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")


def _weights(*, n: int = _N, seed: int = 21):
    """Create one checkpoint allocation whose ownership may be transferred."""

    return make_synthetic_mxfp4_moe(
        _E, _K, n, seed=seed, device=torch.device("cuda")
    )


def _prepare(weights: dict, *, n: int = _N, w13_layout: str = "w13"):
    """Destructively turn checkpoint storage into the sole runtime layout."""

    from b12x.integration import (
        plan_b12x_fp4_moe_weights,
        prepare_b12x_fp4_moe_weights,
    )

    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=_E,
        hidden_size=_K,
        intermediate_size=n,
        w13_layout=w13_layout,
    )
    source_ptrs = tuple(
        weights[name].untyped_storage().data_ptr()
        for name in ("w13_fp4", "w13_mx", "w2_fp4", "w2_mx")
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=plan,
        w1_fp4=weights["w13_fp4"],
        w1_blockscale=weights["w13_mx"],
        w1_global_scale=weights["alphas"],
        a1_gscale=weights["input_scale"],
        w2_fp4=weights["w2_fp4"],
        w2_blockscale=weights["w2_mx"],
        w2_global_scale=weights["alphas"],
        a2_gscale=weights["input_scale"],
        params_dtype=torch.bfloat16,
    )
    runtime = prepared.representation_for("w4a8_mx")
    runtime_ptrs = tuple(
        tensor.untyped_storage().data_ptr()
        for tensor in (
            runtime.w13_rp,
            runtime.w13_sfb,
            runtime.w2_rp,
            runtime.w2_sfb,
        )
    )
    assert runtime_ptrs == source_ptrs
    return prepared


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
    experts,
    *,
    seed: int = 33,
) -> torch.Tensor:
    from b12x.integration.tp_moe import clear_tp_moe_caches
    from tests.helpers import run_tp_moe_fp4

    clear_tp_moe_caches()
    x, topk_ids, topk_weights = _routed_inputs(m, seed)
    out = run_tp_moe_fp4(
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )
    torch.cuda.synchronize()
    return out


def test_w4a8_mx_dynamic_matches_oracle() -> None:
    _skip_if_unavailable()
    from b12x.moe.fused.reference import moe_reference_w4a8_mx

    weights = _weights()
    m = 16
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
    prepared = _prepare(weights)
    out = _run(m, prepared)
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
    prepared = _prepare(weights)
    baseline = _run(m, prepared)
    repeat = _run(m, prepared)

    # Release the baseline owner, then build W31 from a fresh checkpoint.  The
    # half-row temporary avoids retaining complete W13 and W31 models together.
    del prepared, weights
    torch.cuda.empty_cache()
    weights = _weights()
    u8 = weights["w13_fp4"]
    tmp = u8[:, :_N].clone()
    u8[:, :_N].copy_(u8[:, _N:])
    u8[:, _N:].copy_(tmp)
    del tmp
    mx = weights["w13_mx"]
    tmp = mx[:, :_N].clone()
    mx[:, :_N].copy_(mx[:, _N:])
    mx[:, _N:].copy_(tmp)
    del tmp
    prepared = _prepare(weights, w13_layout="w31")
    flipped = _run(m, prepared)
    # Idempotency: a second pass over the same storage must not re-flip.
    flipped2 = _run(m, prepared)

    noise = (baseline.float() - repeat.float()).abs().max().item()
    bound = max(8.0 * noise, 1e-6 * baseline.float().abs().max().item())
    for label, got in (("first", flipped), ("repeat", flipped2)):
        err = (got.float() - baseline.float()).abs().max().item()
        assert err <= bound, (
            f"w31 flip mismatch ({label}): err={err} noise={noise} bound={bound}"
        )


@pytest.mark.parametrize("m", [1, 4])
def test_w4a8_mx_small_dynamic_band_matches_oracle(m: int) -> None:
    _skip_if_unavailable()
    from b12x.integration import plan_b12x_fp4_moe_weights, tp_moe

    weights = _weights()
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=_E,
        hidden_size=_K,
        intermediate_size=_N,
    )
    plan = tp_moe.plan_tp_moe_execution(
        num_tokens=m,
        num_topk=_TOPK,
        device=torch.device("cuda"),
        weight_plan=weight_plan,
        quant_mode="w4a8_mx",
    )
    assert plan.implementation == "dynamic"

    ref = _oracle(m, weights)
    prepared = _prepare(weights)
    out = _run(m, prepared)
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"w4a8_mx dynamic output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


# ---------------------------------------------------------------------------
# Unified dynamic W4A8
# ---------------------------------------------------------------------------


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


def test_w4a8_mx_dense_band_defaults_to_dynamic_and_matches_oracle() -> None:
    """The planner-selected dense band runs unified dynamic and matches its oracle."""
    _skip_if_unavailable()
    weights = _weights()
    m = 1024
    ref = _oracle(m, weights)
    prepared = _prepare(weights)
    out = _run(m, prepared)
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"dynamic output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


@pytest.mark.parametrize("tile_m", [64, 128])
def test_w4a8_mx_materialized_dense_override_matches_oracle(
    monkeypatch, tile_m: int
) -> None:
    """The split dense-prefill specializations remain correct when forced."""

    _skip_if_unavailable()
    monkeypatch.setenv("B12X_DYNAMIC_TILE_MN", f"{tile_m}x128")
    weights = _weights()
    m = 1024
    ref = _oracle(m, weights)
    prepared = _prepare(weights)
    out = _run(m, prepared)
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"M{tile_m} dynamic output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


def test_w4a8_mx_prepared_dynamic_runs_with_compacted_sources() -> None:
    """The serving representation must not retain logical checkpoint weights."""

    _skip_if_unavailable()
    from b12x.integration import (
        plan_b12x_fp4_moe_weights,
        prepare_b12x_fp4_moe_weights,
    )
    from b12x.moe.fused.reference import moe_reference_w4a8_mx
    from tests.helpers import run_tp_moe_fp4

    n = 256
    weights = _weights(n=n, seed=91)
    m = 16
    x, topk_ids, topk_weights = _routed_inputs(m, 92)
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
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=_E,
        hidden_size=_K,
        intermediate_size=n,
        w13_layout="w13",
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=weight_plan,
        w1_fp4=weights["w13_fp4"],
        w1_blockscale=weights["w13_mx"],
        w1_global_scale=weights["alphas"],
        a1_gscale=weights["input_scale"],
        w2_fp4=weights["w2_fp4"],
        w2_blockscale=weights["w2_mx"],
        w2_global_scale=weights["alphas"],
        a2_gscale=weights["input_scale"],
        params_dtype=torch.bfloat16,
    )
    runtime = prepared.representation_for("w4a8_mx")
    assert tuple(
        tensor.untyped_storage().data_ptr()
        for tensor in (
            runtime.w13_rp,
            runtime.w13_sfb,
            runtime.w2_rp,
            runtime.w2_sfb,
        )
    ) == tuple(
        weights[name].untyped_storage().data_ptr()
        for name in ("w13_fp4", "w13_mx", "w2_fp4", "w2_mx")
    )

    out = run_tp_moe_fp4(
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )
    torch.cuda.synchronize()

    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos


def test_w4a8_mx_dynamic_graph_replay_tracks_routing_updates() -> None:
    """Capture unified dynamic in a CUDA graph (caller-owned output buffer);
    replay with routing/activations mutated IN PLACE must track the update
    (vs a fresh eager call on the same inputs, and vs the oracle)."""
    _skip_if_unavailable()
    from b12x.integration.tp_moe import b12x_moe_fp4, clear_tp_moe_caches
    from tests.helpers import make_tp_moe_fp4_binding

    clear_tp_moe_caches()
    device = torch.device("cuda")
    weights = _weights()
    m = 256
    from b12x.moe.fused.reference import moe_reference_w4a8_mx

    initial_x, initial_ids, initial_w = _routed_inputs(m, 33)
    rounds = []
    for round_idx in range(3):
        new_x, new_ids, new_w = _routed_inputs(m, 500 + round_idx)
        ref = moe_reference_w4a8_mx(
            new_x.float(),
            weights["w13_fp4"],
            weights["w13_mx"],
            None,
            weights["alphas"],
            weights["w2_fp4"],
            weights["w2_mx"],
            None,
            weights["alphas"],
            new_ids,
            new_w,
            _E,
            _K,
            _N,
            activation="silu",
        )
        rounds.append((new_x, new_ids, new_w, ref))
    prepared = _prepare(weights)
    x, topk_ids, topk_weights = initial_x, initial_ids, initial_w
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    graph_out = torch.zeros(m, _K, dtype=torch.bfloat16, device=device)
    eager_out = torch.zeros_like(graph_out)

    def _make_binding(out: torch.Tensor):
        return make_tp_moe_fp4_binding(
            a=x,
            experts=prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=out,
            input_scales_static=True,
            quant_mode="w4a8_mx",
        )

    graph_binding = _make_binding(graph_out)
    eager_binding = _make_binding(eager_out)

    def _launch(binding) -> None:
        b12x_moe_fp4(binding=binding)

    # Warm the compiled dynamic launch, then capture.
    _launch(graph_binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        _launch(graph_binding)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    for round_idx, (new_x, new_ids, new_w, ref) in enumerate(rounds):
        x.copy_(new_x)
        topk_ids.copy_(new_ids)
        topk_weights.copy_(new_w)
        graph.replay()
        torch.cuda.synchronize()
        replayed = graph_out.clone()

        eager_out.zero_()
        _launch(eager_binding)
        torch.cuda.synchronize()

        assert replayed.abs().sum().item() > 0, round_idx
        replay_cos = torch.nn.functional.cosine_similarity(
            replayed.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        eager_cos = torch.nn.functional.cosine_similarity(
            eager_out.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        assert eager_cos > 0.998, (round_idx, "eager", eager_cos)
        assert replay_cos > 0.998, (round_idx, "replay", replay_cos)
        replay_eager_cos = torch.nn.functional.cosine_similarity(
            replayed.float().flatten(), eager_out.float().flatten(), dim=0
        ).item()
        assert replay_eager_cos > 0.9999, (round_idx, replay_eager_cos)


def test_w4a8_mx_dynamic_glm_shard_geometry() -> None:
    """GLM per-rank shard geometry: E=16, K=4096, n=256 (FC1 N=512, FC2
    N=4096 with K=256) through unified dynamic, gated vs the oracle."""
    _skip_if_unavailable()
    from b12x.integration.tp_moe import clear_tp_moe_caches
    from b12x.moe.fused.reference import moe_reference_w4a8_mx
    from tests.helpers import run_tp_moe_fp4

    clear_tp_moe_caches()
    n = 256
    weights = _weights(n=n, seed=77)
    m = 256
    x, topk_ids, topk_weights = _routed_inputs(m, 44)
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
    prepared = _prepare(weights, n=n)
    out = run_tp_moe_fp4(
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )
    torch.cuda.synchronize()
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"dynamic output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)
