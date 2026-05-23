from __future__ import annotations

import torch

from b12x.gemm.block_fp8_linear import (
    block_fp8_linear_mxfp8,
    empty_block_fp8_linear_workspace,
    pack_block_fp8_linear_weight_mxfp8,
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
    workspace = empty_block_fp8_linear_workspace(
        tokens,
        in_features,
        out_features,
        device="cuda",
        output_dtype=x.dtype,
    )

    def run_once() -> torch.Tensor:
        return block_fp8_linear_mxfp8(x, packed, workspace=workspace)

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

    torch.testing.assert_close(workspace.output[:, :, 0], eager, rtol=0, atol=0)


def test_block_fp8_linear_default_workspace_path_captures() -> None:
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
