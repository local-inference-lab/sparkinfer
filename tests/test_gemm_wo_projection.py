from __future__ import annotations

import torch

from b12x.gemm.wo_projection import (
    dequantize_mxfp8_rows_torch,
    empty_dense_gemm_mnl_view,
    empty_wo_projection_workspace,
    pack_fp8_block_scaled_weight_mxfp8,
    pack_mxfp8_scales_for_dense_gemm,
    pack_wo_projection_fp8_block_scaled_weights_mxfp8,
    quantize_mxfp8_rows_torch,
    quantize_wo_a_input_mxfp8,
    quantize_wo_b_input_mxfp8,
    quantize_wo_projection_weights_mxfp8_torch,
    wo_a_dense_gemm_mxfp8,
    wo_b_dense_gemm_mxfp8,
    wo_projection_mxfp8,
)

from .helpers import require_sm120


def _assert_close_bf16(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual,
        expected.to(actual.dtype),
        rtol=0,
        atol=0,
    )


def test_pack_mxfp8_scales_round_trips_grouped_rows() -> None:
    require_sm120()

    groups, m, k = 3, 5, 256
    sf_k = k // 32
    scale_u8 = (
        torch.arange(groups * m * sf_k, device="cuda", dtype=torch.int32) % 16 + 120
    ).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).reshape(groups, m, sf_k)

    packed = pack_mxfp8_scales_for_dense_gemm(
        scale,
        m=m,
        k=k,
        num_groups=groups,
    )
    round_trip = (
        packed.permute(5, 2, 1, 0, 4, 3)
        .contiguous()
        .reshape(groups, 128, sf_k)
    )

    torch.testing.assert_close(
        round_trip[:, :m, :].view(torch.uint8),
        scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    assert bool((round_trip[:, m:, :].view(torch.uint8) == 127).all().item())


def test_pack_fp8_block_scaled_weight_expands_grouped_scales() -> None:
    require_sm120()
    torch.manual_seed(20260523)

    groups, m, k = 2, 129, 256
    m_tiles = 2
    k_tiles = 2
    raw_weight = (
        torch.randn((groups * m, k), device="cuda", dtype=torch.bfloat16) / 3
    ).to(torch.float8_e4m3fn)
    block_scale_u8 = (
        torch.arange(groups * m_tiles * k_tiles, device="cuda", dtype=torch.int32)
        % 8
        + 124
    ).to(torch.uint8)
    block_scale = block_scale_u8.view(torch.float8_e8m0fnu).reshape(
        groups * m_tiles,
        k_tiles,
    )

    packed = pack_fp8_block_scaled_weight_mxfp8(
        raw_weight,
        block_scale,
        m=m,
        k=k,
        num_groups=groups,
    )
    expected_scale_rows_u8 = (
        block_scale_u8.reshape(groups, m_tiles, k_tiles)[:, :, None, :, None]
        .expand(groups, m_tiles, 128, k_tiles, 4)
        .reshape(groups, m_tiles * 128, k_tiles * 4)[:, :m, : k // 32]
        .contiguous()
    )

    assert packed.values.shape == (m, k, groups)
    assert packed.values.stride() == (k, 1, m * k)
    torch.testing.assert_close(
        packed.scale_rows.view(torch.uint8),
        expected_scale_rows_u8,
        rtol=0,
        atol=0,
    )

    deq = dequantize_mxfp8_rows_torch(packed.values, packed.scale_rows)
    expected_values = raw_weight.view(groups, m, k).permute(1, 2, 0)
    expected = (
        expected_values.float().reshape(m, k // 32, 32, groups).permute(3, 0, 1, 2)
        * expected_scale_rows_u8.view(torch.float8_e8m0fnu).float()[..., None]
    ).permute(1, 2, 3, 0).reshape(m, k, groups)
    torch.testing.assert_close(deq, expected, rtol=0, atol=0)


def test_pack_fp8_block_scaled_weight_accepts_float_scales() -> None:
    require_sm120()
    torch.manual_seed(20260524)

    groups, m, k = 4, 1024, 4096
    m_tiles = m // 128
    k_tiles = k // 128
    raw_weight = (
        torch.randn((groups * m, k), device="cuda", dtype=torch.bfloat16) / 8
    ).to(torch.float8_e4m3fn)
    scale_u8 = (
        torch.arange(groups * m_tiles * k_tiles, device="cuda", dtype=torch.int32)
        % 4
        + 125
    ).to(torch.uint8)
    scale_e8m0 = scale_u8.view(torch.float8_e8m0fnu).reshape(
        groups * m_tiles,
        k_tiles,
    )
    scale_float = scale_e8m0.float()

    packed_e8m0 = pack_fp8_block_scaled_weight_mxfp8(
        raw_weight,
        scale_e8m0,
        m=m,
        k=k,
        num_groups=groups,
    )
    packed_float = pack_fp8_block_scaled_weight_mxfp8(
        raw_weight,
        scale_float,
        m=m,
        k=k,
        num_groups=groups,
    )

    torch.testing.assert_close(
        packed_float.scale_rows.view(torch.uint8),
        packed_e8m0.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        packed_float.scale_mma.view(torch.uint8),
        packed_e8m0.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_quantize_mxfp8_rows_dequantizes_on_gpu() -> None:
    require_sm120()
    torch.manual_seed(20260522)

    source = (torch.randn((3, 128, 2), device="cuda", dtype=torch.bfloat16) / 4).contiguous()
    q = quantize_mxfp8_rows_torch(source)
    deq = dequantize_mxfp8_rows_torch(q.values, q.scale_rows)

    assert q.values.shape == source.shape
    assert q.values.dtype == torch.float8_e4m3fn
    assert q.scale_rows.shape == (2, 3, 4)
    assert q.scale_mma.shape == (32, 4, 1, 4, 1, 2)
    assert bool(torch.isfinite(deq).all().item())
    max_abs = (deq.float() - source.float()).abs().max().item()
    assert max_abs < 0.05


def test_wo_activation_quant_kernels_match_gpu_reference() -> None:
    require_sm120()
    torch.manual_seed(31000)

    tokens, groups, group_width, rank = 3, 4, 128, 64
    source_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    actual_a = quantize_wo_a_input_mxfp8(source_tgd)
    expected_a = quantize_mxfp8_rows_torch(source_tgd.permute(0, 2, 1).contiguous())
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_a.values.view(torch.uint8),
        expected_a.values.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_a.scale_rows.view(torch.uint8),
        expected_a.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_a.scale_mma.view(torch.uint8),
        expected_a.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )

    source_trg = empty_dense_gemm_mnl_view(
        tokens,
        rank,
        groups,
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_trg.copy_(
        torch.randn((tokens, rank, groups), device="cuda", dtype=torch.bfloat16) / 4
    )
    actual_b = quantize_wo_b_input_mxfp8(source_trg)
    expected_b = quantize_mxfp8_rows_torch(
        source_trg.permute(0, 2, 1).contiguous().reshape(tokens, rank * groups)
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_b.values.view(torch.uint8),
        expected_b.values.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_b.scale_rows.view(torch.uint8),
        expected_b.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_b.scale_mma.view(torch.uint8),
        expected_b.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_wo_a_dense_gemm_mxfp8_matches_quantized_gpu_reference() -> None:
    require_sm120()
    torch.manual_seed(31001)

    tokens, groups, group_width, rank = 2, 2, 128, 64
    x_tgd = (torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16) / 4)
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    x_tdg = quantize_wo_a_input_mxfp8(x_tgd)
    wo_a_rdg = quantize_mxfp8_rows_torch(wo_a_grd.permute(1, 2, 0).contiguous())

    actual = wo_a_dense_gemm_mxfp8(x_tdg, wo_a_rdg)
    torch.cuda.synchronize()

    x_deq = dequantize_mxfp8_rows_torch(x_tdg.values, x_tdg.scale_rows).permute(0, 2, 1)
    wo_a_deq = dequantize_mxfp8_rows_torch(
        wo_a_rdg.values,
        wo_a_rdg.scale_rows,
    ).permute(2, 0, 1)
    expected = torch.einsum("tgd,grd->trg", x_deq, wo_a_deq)

    _assert_close_bf16(actual, expected)


def test_two_gemm_wo_projection_group_major_path_matches_quantized_reference() -> None:
    require_sm120()
    torch.manual_seed(31002)

    tokens, groups, group_width, rank, hidden = 2, 2, 128, 64, 128
    x_tgd = (torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16) / 4)
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    ).contiguous()

    x_tdg = quantize_wo_a_input_mxfp8(x_tgd)
    wo_a_rdg = quantize_mxfp8_rows_torch(wo_a_grd.permute(1, 2, 0).contiguous())
    tmp_trg = wo_a_dense_gemm_mxfp8(x_tdg, wo_a_rdg)
    tmp_q = quantize_wo_b_input_mxfp8(tmp_trg)

    wo_b_q = quantize_mxfp8_rows_torch(wo_b_hgr)
    actual = wo_b_dense_gemm_mxfp8(tmp_q, wo_b_q)
    torch.cuda.synchronize()

    tmp_deq = dequantize_mxfp8_rows_torch(tmp_q.values, tmp_q.scale_rows)
    wo_b_deq = dequantize_mxfp8_rows_torch(wo_b_q.values, wo_b_q.scale_rows)
    expected = tmp_deq @ wo_b_deq.T

    _assert_close_bf16(actual[:, :, 0], expected)


def test_two_gemm_wo_projection_replays_under_graph() -> None:
    require_sm120()
    torch.manual_seed(31003)

    tokens, groups, group_width, rank, hidden = 1, 2, 128, 64, 128
    x_tgd = (torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16) / 4)
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )

    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a_grd, wo_b_hgr)
    workspace = empty_wo_projection_workspace(
        tokens,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
        device="cuda",
    )

    def run_once() -> torch.Tensor:
        return wo_projection_mxfp8(x_tgd, weights, workspace)

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


def test_wo_projection_block_scaled_weight_pack_runs_graph() -> None:
    require_sm120()
    torch.manual_seed(31004)

    tokens, groups, group_width, rank, hidden = 1, 2, 128, 128, 128
    x_tgd = (torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16) / 4)
    wo_a_weight = (
        torch.randn((groups * rank, group_width), device="cuda", dtype=torch.bfloat16) / 8
    ).to(torch.float8_e4m3fn)
    wo_b_weight = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16) / 8
    ).to(torch.float8_e4m3fn)
    wo_a_scale = torch.full(
        (groups * (rank // 128), group_width // 128),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)
    wo_b_scale = torch.full(
        (hidden // 128, groups * rank // 128),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)

    weights = pack_wo_projection_fp8_block_scaled_weights_mxfp8(
        wo_a_weight,
        wo_a_scale,
        wo_b_weight,
        wo_b_scale,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )
    workspace = empty_wo_projection_workspace(
        tokens,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
        device="cuda",
    )

    eager = wo_projection_mxfp8(x_tgd, weights, workspace).clone()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wo_projection_mxfp8(x_tgd, weights, workspace)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert bool(torch.isfinite(workspace.output).all().item())
    torch.testing.assert_close(workspace.output[:, :, 0], eager, rtol=0, atol=0)
