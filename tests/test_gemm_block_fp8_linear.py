from __future__ import annotations

import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm.block_fp8_linear import (
    BlockFP8LinearScratchCaps,
    block_fp8_linear_mxfp8,
    pack_block_fp8_linear_weight_mxfp8,
    plan_block_fp8_linear_scratch,
    quantize_block_fp8_linear_input_mxfp8,
)
from b12x.gemm.wo_projection import dequantize_mxfp8_rows_torch

from .helpers import require_sm120


def _make_block_fp8_weight(
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).to(torch.float8_e4m3fn)
    scale_u8 = (
        torch.arange(
            (out_features // 128) * (in_features // 128),
            device="cuda",
            dtype=torch.int32,
        )
        % 3
        + 126
    ).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).reshape(
        out_features // 128,
        in_features // 128,
    )
    return weight, scale


def _reference_from_quantized_operands(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    x_q = quantize_block_fp8_linear_input_mxfp8(x)
    w_q = pack_block_fp8_linear_weight_mxfp8(weight, scale)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    w_deq = dequantize_mxfp8_rows_torch(w_q.weight.values, w_q.weight.scale_rows)
    return x_deq @ w_deq.T


def test_block_fp8_linear_matches_quantized_reference() -> None:
    require_sm120()
    torch.manual_seed(20260523)

    tokens, in_features, out_features = 7, 256, 384
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    actual = block_fp8_linear_mxfp8(x, packed)
    expected = _reference_from_quantized_operands(x, weight, scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=0,
        atol=0,
    )


def test_block_fp8_linear_replays_under_cuda_graph() -> None:
    require_sm120()
    torch.manual_seed(20260524)

    tokens, in_features, out_features = 1, 128, 256
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)
    plan = plan_block_fp8_linear_scratch(
        BlockFP8LinearScratchCaps(
            device=x.device,
            max_tokens=tokens,
            in_features=in_features,
            out_features=out_features,
            output_dtype=x.dtype,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=x.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    output = torch.empty((tokens, out_features, 1), dtype=x.dtype, device=x.device)
    binding = plan.bind(
        scratch=scratch,
        source=x,
        packed_weight=packed,
        output=output,
    )

    def run_once() -> torch.Tensor:
        return block_fp8_linear_mxfp8(binding=binding)

    eager = run_once().clone()
    torch.cuda.synchronize()

    run_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output[:, :, 0], eager, rtol=0, atol=0)


def test_block_fp8_linear_scratch_binding_replays_under_cuda_graph() -> None:
    require_sm120()
    torch.manual_seed(20260526)

    tokens, in_features, out_features = 1, 128, 256
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)
    plan = plan_block_fp8_linear_scratch(
        BlockFP8LinearScratchCaps(
            device=x.device,
            max_tokens=tokens,
            in_features=in_features,
            out_features=out_features,
            output_dtype=x.dtype,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=x.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    output = torch.empty((tokens, out_features, 1), dtype=x.dtype, device=x.device)
    binding = plan.bind(
        scratch=scratch,
        source=x,
        packed_weight=packed,
        output=output,
    )

    def run_once() -> torch.Tensor:
        return block_fp8_linear_mxfp8(binding=binding)

    eager = run_once().clone()
    torch.cuda.synchronize()

    run_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)


def test_block_fp8_linear_default_fused_path_captures() -> None:
    require_sm120()
    torch.manual_seed(20260525)

    tokens, in_features, out_features = 1, 128, 256
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    eager = block_fp8_linear_mxfp8(x, packed).clone()
    torch.cuda.synchronize()

    block_fp8_linear_mxfp8(x, packed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = block_fp8_linear_mxfp8(x, packed)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)


def test_block_fp8_linear_live_m_does_not_resolve_new_dense_kernel() -> None:
    require_sm120()
    torch.manual_seed(20260528)

    warm_tokens, live_tokens = 4096, 1824
    in_features, out_features = 128, 1536
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    warm_x = (
        torch.randn((warm_tokens, in_features), device="cuda", dtype=torch.bfloat16)
        / 4
    ).contiguous()
    live_x = (
        torch.randn((live_tokens, in_features), device="cuda", dtype=torch.bfloat16)
        / 4
    ).contiguous()

    block_fp8_linear_mxfp8(warm_x, packed)
    torch.cuda.synchronize()

    freeze_kernel_resolution("block FP8 dense GEMM live M should be runtime")
    try:
        actual = block_fp8_linear_mxfp8(live_x, packed)
        torch.cuda.synchronize()
    finally:
        unfreeze_kernel_resolution()

    expected = _reference_from_quantized_operands(live_x, weight, scale)
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=1e-2,
        atol=1e-4,
    )


def test_block_fp8_linear_small_live_m_reuses_prefill_dense_kernel() -> None:
    require_sm120()
    torch.manual_seed(20260529)

    warm_tokens = 512
    live_token_counts = (16, 32, 128)
    in_features, out_features = 1024, 8192
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    warm_x = (
        torch.randn((warm_tokens, in_features), device="cuda", dtype=torch.bfloat16)
        / 4
    ).contiguous()
    live_xs = [
        (
            torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16)
            / 4
        ).contiguous()
        for tokens in live_token_counts
    ]

    block_fp8_linear_mxfp8(warm_x, packed)
    torch.cuda.synchronize()

    freeze_kernel_resolution("small live M should reuse the prefill dense kernel")
    try:
        for tokens, live_x in zip(live_token_counts, live_xs, strict=True):
            actual = block_fp8_linear_mxfp8(live_x, packed)
            torch.cuda.synchronize()
            assert actual.shape == (tokens, out_features)
    finally:
        unfreeze_kernel_resolution()


def test_block_fp8_linear_expected_m_decode_regime_reuses_kernel() -> None:
    # DeepGEMM-style expected_m hint: a decode-regime kernel (expected_m<=128 ->
    # 32x128 tile) must (a) produce byte-identical output to the default
    # (tile choice does not change the block-scaled MMA result) and (b) be
    # reused for every live M in the regime under frozen resolution.
    require_sm120()
    from b12x.gemm.dense import _select_default_mma_tiler_mn

    torch.manual_seed(20260530)
    in_features, out_features = 1024, 8192  # wide-N (>1536) MXFP8 regime
    expected_m = 64  # decode/small-batch regime
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    assert _select_default_mma_tiler_mn(
        expected_m, out_features, sm, is_mxfp8=True, expected_m=expected_m
    ) == (32, 128)

    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scale)

    # (a) tile-independence of numerics: hint (32x128) vs default (64x128).
    x = (
        torch.randn((32, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    default_out = block_fp8_linear_mxfp8(x, packed)
    hinted_out = block_fp8_linear_mxfp8(x, packed, expected_m=expected_m)
    torch.cuda.synchronize()
    torch.testing.assert_close(hinted_out.float(), default_out.float(), rtol=0, atol=0)

    # (b) warm the decode kernel once, freeze, serve a range of live M -> all
    # reuse the same warmed (32x128) kernel (no recompile under frozen
    # resolution). Live M stays in the persistent-scheduler policy class (m>=16),
    # matching the warm M; M==1 / m<16 are separate policy regimes
    # (use_m1_non_tma / direct scheduler) that must be warmed on their own -- a
    # pre-existing dense_gemm constraint independent of the expected_m hint.
    warm_x = (
        torch.randn((256, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    block_fp8_linear_mxfp8(warm_x, packed, expected_m=expected_m)
    torch.cuda.synchronize()

    freeze_kernel_resolution("decode-regime block FP8 reused for all live M")
    try:
        for tokens in (16, 32, 128):
            live_x = (
                torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16)
                / 4
            ).contiguous()
            out = block_fp8_linear_mxfp8(live_x, packed, expected_m=expected_m)
            torch.cuda.synchronize()
            assert out.shape == (tokens, out_features)
    finally:
        unfreeze_kernel_resolution()
