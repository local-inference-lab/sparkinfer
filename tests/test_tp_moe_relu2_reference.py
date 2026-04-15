from __future__ import annotations

import pytest
import torch

from b12x.cute.fp4 import FLOAT4_E2M1_MAX, fp4_quantize_values_torch, pack_grouped_fp4_values, swizzle_block_scale
from b12x.integration import tp_moe
from b12x.integration.tp_moe import allocate_tp_moe_workspace_pool, b12x_moe_fp4, clear_tp_moe_caches
from b12x.moe.fused.reference import compare_to_reference, moe_reference_nvfp4

from .helpers import require_sm120


BACKEND_CASES = [
    ("micro", 2, 10_000, 10_000),
    ("static", 128, 10_000, 0),
    ("dynamic", 768, 0, 0),
]


def _quantize_moe_weight_storage(
    input_tensor: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups, rows, cols = input_tensor.shape
    quantized = torch.zeros((num_groups, rows, cols), dtype=torch.float32, device=input_tensor.device)
    scales = torch.zeros((num_groups, rows, cols // 16), dtype=torch.float32, device=input_tensor.device)
    for group_idx in range(num_groups):
        x = input_tensor[group_idx].float()
        sliced = x.view(rows, cols // 16, 16)
        block_max = sliced.abs().amax(dim=-1, keepdim=True)
        scale = (global_scale[group_idx] * (block_max / FLOAT4_E2M1_MAX)).to(torch.float8_e4m3fn).to(torch.float32)
        output_scale = 1.0 / (scale * (1.0 / global_scale[group_idx]))
        clipped = torch.clamp(sliced * output_scale, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).view(rows, cols)
        quantized[group_idx] = fp4_quantize_values_torch(clipped)
        scales[group_idx] = scale.squeeze(-1)

    packed = pack_grouped_fp4_values(quantized).permute(2, 0, 1).contiguous()
    swizzled = swizzle_block_scale(scales.to(torch.float8_e4m3fn))
    return packed, swizzled


def _make_activation_case(
    *,
    device: torch.device,
    activation: str,
    m: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)

    E, k, n, topk = 1, 128, 128, 1
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    topk_ids = torch.zeros(m, topk, device=device, dtype=torch.int32)
    topk_weights = torch.ones(m, topk, device=device, dtype=torch.float32)

    w1_rows = 2 * n if activation == "silu" else n
    w1 = torch.randn(E, w1_rows, k, device=device, dtype=torch.bfloat16) * 0.5
    w2 = torch.randn(E, k, n, device=device, dtype=torch.bfloat16) * 0.25
    a1_gscale = torch.ones(E, device=device, dtype=torch.float32)
    a2_gscale = torch.ones(E, device=device, dtype=torch.float32)
    w1_fp4, w1_blockscale = _quantize_moe_weight_storage(w1, a1_gscale)
    w2_fp4, w2_blockscale = _quantize_moe_weight_storage(w2, a2_gscale)
    w1_alphas = torch.ones(E, device=device, dtype=torch.float32)
    w2_alphas = torch.ones(E, device=device, dtype=torch.float32)
    return (
        x,
        topk_ids,
        topk_weights,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        a1_gscale,
        a2_gscale,
        E,
        k,
        n,
    )


def _run_activation_case(
    *,
    activation: str,
    m: int,
    static_cutover: int,
    micro_cutover: int,
    fast_math: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = require_sm120()
    (
        x,
        topk_ids,
        topk_weights,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        a1_gscale,
        a2_gscale,
        E,
        k,
        n,
    ) = _make_activation_case(device=device, activation=activation, m=m)

    reference = moe_reference_nvfp4(
        x,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        a1_gscale,
        a2_gscale,
        topk_ids,
        topk_weights,
        E,
        k,
        n,
        activation=activation,
    )

    prev_static_cutover = tp_moe._STATIC_COMPACT_CUTOVER_PAIRS_CACHE
    prev_micro_cutover = tp_moe._MICRO_COMPACT_CUTOVER_PAIRS_CACHE
    try:
        clear_tp_moe_caches()
        tp_moe._STATIC_COMPACT_CUTOVER_PAIRS_CACHE = static_cutover
        tp_moe._MICRO_COMPACT_CUTOVER_PAIRS_CACHE = micro_cutover

        output = b12x_moe_fp4(
            x,
            a1_gscale,
            w1_fp4,
            w1_blockscale,
            w1_alphas,
            a2_gscale,
            w2_fp4,
            w2_blockscale,
            w2_alphas,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_workspace_pool(),
            input_scales_static=True,
            activation=activation,
            fast_math=fast_math,
        )
        torch.cuda.synchronize()
    finally:
        clear_tp_moe_caches()
        tp_moe._STATIC_COMPACT_CUTOVER_PAIRS_CACHE = prev_static_cutover
        tp_moe._MICRO_COMPACT_CUTOVER_PAIRS_CACHE = prev_micro_cutover

    return output, reference


@pytest.mark.parametrize(
    ("backend", "m", "static_cutover", "micro_cutover"),
    BACKEND_CASES,
)
@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_activation_exact_path_matches_reference_across_backends(
    activation: str,
    backend: str,
    m: int,
    static_cutover: int,
    micro_cutover: int,
) -> None:
    output, reference = _run_activation_case(
        activation=activation,
        m=m,
        static_cutover=static_cutover,
        micro_cutover=micro_cutover,
        fast_math=False,
    )
    metrics = compare_to_reference(output, reference)
    assert metrics.max_abs == 0.0, f"{activation}/{backend}: {metrics}"
    assert metrics.rmse == 0.0, f"{activation}/{backend}: {metrics}"


@pytest.mark.parametrize(
    ("backend", "m", "static_cutover", "micro_cutover"),
    BACKEND_CASES,
)
def test_relu2_matches_reference_across_backends(
    backend: str,
    m: int,
    static_cutover: int,
    micro_cutover: int,
) -> None:
    output, reference = _run_activation_case(
        activation="relu2",
        m=m,
        static_cutover=static_cutover,
        micro_cutover=micro_cutover,
        fast_math=True,
    )
    metrics = compare_to_reference(output, reference)
    assert metrics.max_abs == 0.0, f"{backend}: {metrics}"
    assert metrics.rmse == 0.0, f"{backend}: {metrics}"
