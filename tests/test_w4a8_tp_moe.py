"""End-to-end w4a8_nvfp4 dispatch through b12x_moe_fp4 with real weights.

Gates the Phase-5 serving integration: the same entry point, workspace
allocator, and checkpoint weights as the nvfp4 path, with
quant_mode="w4a8_nvfp4" deriving the UE8M0/residual grids on the fly.
"""

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
    load_expert_weights,
    make_routed_inputs,
)


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
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
def _weights():
    return load_expert_weights(MODEL_PATH, _make_spec())


def test_unswizzle_batched_matches_reference() -> None:
    _skip_if_unavailable()
    from b12x.integration.tp_moe import _unswizzle_block_scales_batched
    from b12x.moe.fused.reference import unswizzle_block_scale

    spec = _make_spec()
    weights = _weights()
    k = spec.hidden_size
    w1_n = weights.w13_weight.shape[1]
    bs_u8 = weights.w13_blockscale_swizzled.view(torch.uint8)
    batched = _unswizzle_block_scales_batched(bs_u8, w1_n, k // 16)
    for e in (0, 1, 7):
        ref = unswizzle_block_scale(bs_u8[e], w1_n, k // 16)
        torch.testing.assert_close(batched[e], ref, atol=0.0, rtol=0.0)


def _run_mode(m: int, quant_mode: str | None, seed: int) -> torch.Tensor:
    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    device = torch.device("cuda")
    spec = _make_spec()
    weights = _weights()
    x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=seed, device=device)
    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_quant_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_quant_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
        quant_mode=quant_mode,
    )
    out = b12x_moe_fp4(
        x,
        weights.w13_input_scale_quant_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_quant_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        quant_mode=quant_mode,
    )
    torch.cuda.synchronize()
    return out


def _w4a8_oracle(m: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run moe_reference_w4a8_mx on the real checkpoint weights."""
    from b12x.integration.tp_moe import _derive_w4a8_weight_grids
    from b12x.moe.fused.reference import moe_reference_w4a8_mx

    device = torch.device("cuda")
    spec = _make_spec()
    weights = _weights()
    k = spec.hidden_size
    n = weights.w2_weight.shape[2] * 2
    E = weights.w13_weight.shape[0]
    w1_n = weights.w13_weight.shape[1]
    x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=seed, device=device)

    w13_mx, w13_res = _derive_w4a8_weight_grids(
        weights.w13_blockscale_swizzled.view(torch.uint8), w1_n, k
    )
    w2_mx, w2_res = _derive_w4a8_weight_grids(
        weights.w2_blockscale_swizzled.view(torch.uint8), k, n
    )
    # w4a8 activations carry no global scale, so the pure weight-dequant
    # alpha is the combined nvfp4 alpha with the activation gs folded out.
    a1 = weights.w13_input_scale_quant_per_expert.float().expand(E)
    a2 = weights.w2_input_scale_quant_per_expert.float().expand(E)
    alpha1 = weights.g1_alphas_per_expert.float() * a1
    alpha2 = weights.g2_alphas_per_expert.float() * a2

    out = moe_reference_w4a8_mx(
        x.float(),
        weights.w13_weight.view(torch.uint8), w13_mx,
        w13_res.view(torch.float8_e4m3fn), alpha1,
        weights.w2_weight.view(torch.uint8), w2_mx,
        w2_res.view(torch.float8_e4m3fn), alpha2,
        topk_ids, topk_weights.float(), E, k, n,
        activation="silu",
    )
    return out, x


@pytest.mark.parametrize("m", [4, 64])
def test_w4a8_nvfp4_dispatch_matches_oracle(m: int) -> None:
    """w4a8_nvfp4 through the real serving entry, gated by the w4a8 oracle."""
    _skip_if_unavailable()
    out_w4a8 = _run_mode(m, "w4a8_nvfp4", seed=99)
    oracle, _ = _w4a8_oracle(m, seed=99)

    n_w4a8 = out_w4a8.float().norm().item()
    assert n_w4a8 > 0.01, f"m={m}: w4a8 output near-zero (norm={n_w4a8})"
    cos = torch.nn.functional.cosine_similarity(
        out_w4a8.float().flatten(), oracle.flatten(), dim=0
    ).item()
    assert cos > 0.998, (m, cos)


@pytest.mark.parametrize("m", [4, 256])
def test_w4a8_nvfp4_dispatch_tracks_nvfp4(m: int) -> None:
    """Cross-check vs the nvfp4 path: two approximations of the same layer.

    The nvfp4 output itself carries FP4 activation error, so the agreement
    bound is looser than the oracle gate above.
    """
    _skip_if_unavailable()
    out_ref = _run_mode(m, None, seed=99)
    out_w4a8 = _run_mode(m, "w4a8_nvfp4", seed=99)

    assert out_w4a8.shape == out_ref.shape
    n_ref = out_ref.float().norm().item()
    n_w4a8 = out_w4a8.float().norm().item()
    assert n_w4a8 > 0.01, f"m={m}: w4a8 output near-zero (norm={n_w4a8})"
    cos = torch.nn.functional.cosine_similarity(
        out_w4a8.float().flatten(), out_ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.97, (m, cos, n_ref, n_w4a8)
    assert 0.8 < n_w4a8 / n_ref < 1.25, (m, n_ref, n_w4a8)
