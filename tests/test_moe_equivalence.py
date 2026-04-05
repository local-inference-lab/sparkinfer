"""Smoke tests for the auto-dispatched TP MoE path."""

from __future__ import annotations

import functools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_moe import (
    GRAPH_REPLAY_TOLERANCES,
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    allocate_layer_chain_workspace,
    ModelSpec,
    compare_graph_replay_outputs,
    capture_moe_layer_chain,
    get_scale_contract_params,
    load_expert_weight_stack,
    load_expert_weights,
    make_input_activations,
    make_multilayer_routing_case,
    make_routed_inputs,
    run_moe_layer_chain,
)


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
def _load_multilayer_weights() -> tuple:
    return tuple(
        load_expert_weight_stack(
            MODEL_PATH,
            _make_spec(),
            layer_start=0,
            num_layers=4,
        )
    )


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_moe_nonzero(m):
    """Validate `b12x_moe_fp4` produces non-zero output with real weights."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec)

    x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=99, device=device)
    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )

    out = b12x_moe_fp4(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights, topk_ids,
        workspace=workspace,
        input_scales_static=True,
    )
    torch.cuda.synchronize()

    out_norm = out.float().norm().item()
    print(f"\nm={m}: out_norm={out_norm:.4f}, shape={out.shape}")
    assert out_norm > 0.01, f"m={m}: output is near-zero (norm={out_norm})"
    assert out.shape == (m, spec.hidden_size)


@pytest.mark.parametrize("m", [1, 4])
def test_moe_fc2_tiled_oracle_matches_nvfp4_oracle_when_tile_amax_off(m):
    """The FC2 tiled oracle should reduce to the current oracle when tile-amax is off."""
    _skip_if_unavailable()

    from b12x.moe.fused.reference import (
        compare_to_reference,
        moe_reference_nvfp4,
        moe_reference_nvfp4_fc2_tiled,
    )

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec)
    x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=17_300 + m, device=device)

    reference = moe_reference_nvfp4(
        x,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        weights.w13_input_scale_per_expert,
        weights.w2_input_scale_per_expert,
        topk_ids,
        topk_weights,
        spec.num_experts,
        spec.hidden_size,
        spec.I_tp,
        fc2_tile_amax=False,
    )
    tiled_reference = moe_reference_nvfp4_fc2_tiled(
        x,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        weights.w13_input_scale_per_expert,
        weights.w2_input_scale_per_expert,
        topk_ids,
        topk_weights,
        spec.num_experts,
        spec.hidden_size,
        spec.I_tp,
        fc2_tile_amax=False,
    )

    metrics = compare_to_reference(tiled_reference, reference)
    assert metrics.max_abs == 0.0
    assert metrics.rmse == 0.0
    assert metrics.mean_abs == 0.0
    assert metrics.cos == 1.0


@pytest.mark.parametrize("m", [1, 4])
def test_moe_fc2_tile_amax_matches_nvfp4_reference(m):
    """Validate the FC2 tile-amax path against the NVFP4 torch reference."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )
    from b12x.moe.fused.reference import compare_to_reference, moe_reference_nvfp4_fc2_tiled

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec)

    x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=19_900 + m, device=device)
    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )

    actual = b12x_moe_fp4(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        fc2_tile_amax=True,
    )
    torch.cuda.synchronize()

    reference = moe_reference_nvfp4_fc2_tiled(
        x,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        weights.w13_input_scale_per_expert,
        weights.w2_input_scale_per_expert,
        topk_ids,
        topk_weights,
        spec.num_experts,
        spec.hidden_size,
        spec.I_tp,
        fc2_tile_amax=True,
    )

    metrics = compare_to_reference(actual, reference)
    assert metrics.max_abs <= 2e-3
    assert metrics.cos > 0.9995


@pytest.mark.parametrize(
    ("m", "fc1_tile_amax"),
    [
        (4, False),
        (4, True),
        (80, False),
        (80, True),
    ],
)
def test_moe_prequantized_fc1_matches_tiled_reference(m, fc1_tile_amax):
    """Validate the prequantized-activation FC1 path against the tiled NVFP4 oracle."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4_prequantized,
        clear_tp_moe_caches,
    )
    from b12x.moe.fused.reference import (
        compare_to_reference,
        moe_reference_nvfp4_fc1_fc2_tiled,
    )

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec)

    x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=27_100 + m + int(fc1_tile_amax), device=device)
    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )

    actual = b12x_moe_fp4_prequantized(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        fc1_tile_amax=fc1_tile_amax,
    )
    torch.cuda.synchronize()

    reference = moe_reference_nvfp4_fc1_fc2_tiled(
        x,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        weights.w13_input_scale_per_expert,
        weights.w2_input_scale_per_expert,
        topk_ids,
        topk_weights,
        spec.num_experts,
        spec.hidden_size,
        spec.I_tp,
        fc1_tile_amax=fc1_tile_amax,
        fc2_tile_amax=False,
    )

    metrics = compare_to_reference(actual, reference)
    assert metrics.max_abs <= 3e-3
    assert metrics.cos > 0.999


def test_moe_prequantized_dynamic_top1_multi_expert_matches_tiled_reference():
    """Exercise the dynamic prequantized FC1 path with multiple experts active."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4_prequantized,
        clear_tp_moe_caches,
    )
    from b12x.moe.fused.reference import (
        compare_to_reference,
        moe_reference_nvfp4_fc1_fc2_tiled,
    )

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec)

    m = 641
    x = make_input_activations(spec, m, seed=1234, device=device)
    topk_ids = (torch.arange(m, device=device, dtype=torch.int32) % 2)[:, None].contiguous()
    topk_weights = torch.ones((m, 1), dtype=torch.float32, device=device)
    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )
    assert workspace.implementation == "dynamic"

    actual = b12x_moe_fp4_prequantized(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        fc1_tile_amax=False,
    )
    torch.cuda.synchronize()

    reference = moe_reference_nvfp4_fc1_fc2_tiled(
        x,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        weights.w13_input_scale_per_expert,
        weights.w2_input_scale_per_expert,
        topk_ids,
        topk_weights,
        spec.num_experts,
        spec.hidden_size,
        spec.I_tp,
        fc1_tile_amax=False,
        fc2_tile_amax=False,
    )

    metrics = compare_to_reference(actual, reference)
    assert metrics.max_abs <= 1e-3
    assert metrics.rmse <= 1e-4
    assert metrics.cos > 0.9999


def test_static_gemma_sparse_moe_wrapper_matches_manual_prequantized_path():
    _skip_if_unavailable()

    from b12x.integration.tp_moe import (
        B12XFP4ExpertWeights,
        B12XTopKRouting,
        _b12x_gemma_sparse_moe_fp4_static,
        allocate_tp_moe_workspace,
        b12x_moe_fp4_prequantized,
        clear_tp_moe_caches,
    )
    from b12x.moe.fused.reference import compare_to_reference

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec)

    normed_hidden_states, topk_ids, topk_weights = make_routed_inputs(spec, 4, seed=31_337, device=device)
    hidden_states = torch.randn_like(normed_hidden_states)
    residual = torch.randn_like(normed_hidden_states)
    residual_expected = hidden_states + residual
    norm_weight = torch.randn(spec.hidden_size, dtype=torch.bfloat16, device=device).contiguous()

    class _FakePreMLPRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, object]] = []

        def allreduce_gemma_rmsnorm(
            self,
            inp: torch.Tensor,
            residual_in: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            *,
            peer_input_ptrs=None,
            out=None,
            residual_out=None,
        ):
            self.calls.append((inp, residual_in, weight, eps, peer_input_ptrs))
            if out is None:
                out = torch.empty_like(normed_hidden_states)
            out.copy_(normed_hidden_states)
            if residual_out is None:
                residual_out = torch.empty_like(normed_hidden_states)
            residual_out.copy_(residual_expected)
            return out, residual_out

    runtime = _FakePreMLPRuntime()
    routing = B12XTopKRouting(topk_weights=topk_weights, topk_ids=topk_ids)
    wrapper_workspace = allocate_tp_moe_workspace(
        normed_hidden_states,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )
    manual_workspace = allocate_tp_moe_workspace(
        normed_hidden_states,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )

    actual, residual_out = _b12x_gemma_sparse_moe_fp4_static(
        hidden_states,
        residual,
        pre_mlp_runtime=runtime,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        experts=B12XFP4ExpertWeights(
            a1_gscale=weights.w13_input_scale_per_expert,
            w1_fp4=weights.w13_weight,
            w1_blockscale=weights.w13_blockscale_swizzled,
            w1_alphas=weights.g1_alphas_per_expert,
            a2_gscale=weights.w2_input_scale_per_expert,
            w2_fp4=weights.w2_weight,
            w2_blockscale=weights.w2_blockscale_swizzled,
            w2_alphas=weights.g2_alphas_per_expert,
        ),
        workspace=wrapper_workspace,
        routing=routing,
        input_scales_static=True,
        fc1_tile_amax=True,
    )
    expected = b12x_moe_fp4_prequantized(
        normed_hidden_states,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        workspace=manual_workspace,
        input_scales_static=True,
        fc1_tile_amax=True,
    )
    torch.cuda.synchronize()

    metrics = compare_to_reference(actual, expected)
    assert metrics.max_abs <= 3e-3
    assert metrics.cos > 0.999
    torch.testing.assert_close(residual_out, residual_expected)
    assert len(runtime.calls) == 1
    assert runtime.calls[0][0] is hidden_states
    assert runtime.calls[0][1] is residual
    assert runtime.calls[0][2] is norm_weight


@pytest.mark.parametrize("m", [1, 2])
def test_moe_cuda_graph_replay_tracks_routing_updates(m):
    """Validate graph replay stays correct when routing contents change."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec)

    x0, topk_ids0, topk_weights0 = make_routed_inputs(spec, m, seed=123, device=device)
    x_buf = x0.clone()
    topk_ids_buf = topk_ids0.clone()
    topk_weights_buf = topk_weights0.clone()
    graph_output = torch.empty_like(x_buf)
    workspace = allocate_tp_moe_workspace(
        x_buf,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids_buf,
        input_scales_static=True,
    )

    # Compile once before capture; the replay check below is about routing safety.
    b12x_moe_fp4(
        x_buf,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights_buf,
        topk_ids_buf,
        output=graph_output,
        workspace=workspace,
        input_scales_static=True,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b12x_moe_fp4(
            x_buf,
            weights.w13_input_scale_per_expert,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            weights.g1_alphas_per_expert,
            weights.w2_input_scale_per_expert,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            weights.g2_alphas_per_expert,
            topk_weights_buf,
            topk_ids_buf,
            output=graph_output,
            workspace=workspace,
            input_scales_static=True,
        )

    for seed in (123, 456):
        x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=seed, device=device)
        x_buf.copy_(x)
        topk_ids_buf.copy_(topk_ids)
        topk_weights_buf.copy_(topk_weights)

        graph.replay()
        torch.cuda.synchronize()
        replay_out = graph_output.clone()

        eager_out = b12x_moe_fp4(
            x,
            weights.w13_input_scale_per_expert,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            weights.g1_alphas_per_expert,
            weights.w2_input_scale_per_expert,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            weights.g2_alphas_per_expert,
            topk_weights,
            topk_ids,
            workspace=workspace,
            input_scales_static=True,
        )
        torch.cuda.synchronize()

        metrics = compare_graph_replay_outputs(replay_out, eager_out)
        max_abs = metrics.max_abs
        cos = metrics.cos

        assert max_abs < 5e-4, f"m={m} seed={seed}: max_abs={max_abs:.6f}"
        assert cos > 0.9999, f"m={m} seed={seed}: cos={cos:.6f}"


@pytest.mark.parametrize("m", [1, 2, 4])
def test_moe_cuda_graph_replay_multilayer_tracks_routing_updates(m):
    """Validate a captured multi-layer graph stays correct under routing churn."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import clear_tp_moe_caches

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights_stack = list(_load_multilayer_weights())
    params_stack = [get_scale_contract_params(weights, "shared") for weights in weights_stack]
    num_layers = len(weights_stack)

    x_buf = make_input_activations(spec, m, seed=10_000 + m, device=device)
    initial_case = make_multilayer_routing_case(
        spec,
        m,
        num_layers,
        device,
        pattern="disjoint",
        seed=20_000 + m,
    )
    topk_ids_bufs = [topk_ids.clone() for topk_ids, _ in initial_case]
    topk_weights_bufs = [topk_weights.clone() for _, topk_weights in initial_case]
    graph_output_bufs = [torch.empty_like(x_buf) for _ in range(num_layers)]
    eager_output_bufs = [torch.empty_like(x_buf) for _ in range(num_layers)]
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
        fast_math=True,
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
        fast_math=True,
        output_buffers=graph_output_bufs,
        workspace=shared_workspace,
    )

    scenario_specs = [
        ("disjoint", "disjoint", 1100),
        ("overlap", "overlap", 2200),
        ("random-a", "random", 3300),
        ("random-b", "random", 4400),
    ]
    max_abs_tol = GRAPH_REPLAY_TOLERANCES["max_abs"]
    cos_tol = GRAPH_REPLAY_TOLERANCES["cos_min"]

    for scenario_name, pattern, seed in scenario_specs:
        x_case = make_input_activations(
            spec,
            m,
            seed=30_000 + m + seed,
            device=device,
        )
        routing_case = make_multilayer_routing_case(
            spec,
            m,
            num_layers,
            device,
            pattern=pattern,
            seed=40_000 + m + seed,
        )

        x_buf.copy_(x_case)
        for layer_idx, (topk_ids, topk_weights) in enumerate(routing_case):
            topk_ids_bufs[layer_idx].copy_(topk_ids)
            topk_weights_bufs[layer_idx].copy_(topk_weights)

        graph.replay()
        torch.cuda.synchronize()

        run_moe_layer_chain(
            weights_stack,
            params_stack,
            x_buf,
            topk_ids_bufs,
            topk_weights_bufs,
            fast_math=True,
            output_buffers=eager_output_bufs,
            workspace=shared_workspace,
        )
        torch.cuda.synchronize()

        for layer_idx, (replay_out, eager_out) in enumerate(
            zip(graph_output_bufs, eager_output_bufs, strict=True)
        ):
            metrics = compare_graph_replay_outputs(replay_out, eager_out)
            max_abs = metrics.max_abs
            cos = metrics.cos

            assert (
                max_abs < max_abs_tol
            ), f"m={m} scenario={scenario_name} layer={layer_idx}: max_abs={max_abs:.6f}"
            assert (
                cos > cos_tol
            ), f"m={m} scenario={scenario_name} layer={layer_idx}: cos={cos:.6f}"
