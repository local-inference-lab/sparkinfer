from __future__ import annotations

import pytest
import torch

from sparkinfer._lib.intrinsics import quantize_grouped_nvfp4_torch
from sparkinfer._lib.dense_gemm import (
    dense_gemm,
    dense_gemm_fused_quant_a,
    dense_gemm_fused_quant_a_grouped,
)
from sparkinfer._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from sparkinfer.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    empty_mxfp8_rows_for_dense_gemm,
    quantize_mxfp8_rows_torch,
)

from tests._reference.helpers import dequantize_grouped_nvfp4, require_sparkinfer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _quantize_nvfp4_operand(
    source_gmk: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    groups, rows, _ = map(int, source_gmk.shape)
    row_counts = torch.full(
        (groups,), rows, dtype=torch.int32, device=source_gmk.device
    )
    amax = source_gmk.abs().amax().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0],
        dtype=torch.float32,
        device=source_gmk.device,
    ) / amax
    return quantize_grouped_nvfp4_torch(
        source_gmk, row_counts, global_scale
    ), global_scale


def _dequantize_nvfp4_dense_operand(
    operand: tuple[torch.Tensor, torch.Tensor],
    *,
    k: int,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    packed_mkl, scale_mma = operand
    packed_gmk = packed_mkl.permute(2, 0, 1).contiguous()
    return dequantize_grouped_nvfp4(
        packed_gmk,
        scale_mma,
        k,
        global_scale,
    )


def _mxfp8_gemm_reference(
    source_mkl: torch.Tensor,
    b_values: torch.Tensor,
    b_scale_rows: torch.Tensor,
) -> torch.Tensor:
    a_quant = quantize_mxfp8_rows_torch(source_mkl)
    a_dequant = dequantize_mxfp8_rows_torch(
        a_quant.values, a_quant.scale_rows
    ).to(torch.bfloat16)
    b_dequant = dequantize_mxfp8_rows_torch(
        b_values, b_scale_rows
    ).to(torch.bfloat16)
    return torch.einsum("mkl,nkl->mnl", a_dequant, b_dequant).to(
        torch.bfloat16
    )


def test_cute_migration_dense_nvfp4_gpu_oracle_and_graph() -> None:
    require_sparkinfer()
    generator = torch.Generator(device="cuda").manual_seed(46_001)
    m, n, k = 32, 128, 128
    a_source = (
        torch.randn(
            (1, m, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    b_source = (
        torch.randn(
            (1, n, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    a, a_global_scale = _quantize_nvfp4_operand(a_source)
    b, b_global_scale = _quantize_nvfp4_operand(b_source)
    alpha = (1.0 / (a_global_scale[0] * b_global_scale[0])).reshape(1)
    out = torch.empty((m, n, 1), dtype=torch.bfloat16, device="cuda")

    def run() -> torch.Tensor:
        return dense_gemm(
            a,
            b,
            out=out,
            alpha=alpha,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            mma_tiler_mn=(64, 64),
            load_path="tma",
            swap_ab=False,
        )

    run()
    torch.cuda.synchronize()
    a_dequant = _dequantize_nvfp4_dense_operand(
        a, k=k, global_scale=a_global_scale
    )
    b_dequant = _dequantize_nvfp4_dense_operand(
        b, k=k, global_scale=b_global_scale
    )
    expected = torch.einsum("gmk,gnk->mng", a_dequant, b_dequant).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_cute_migration_dense_fused_quant_gpu_oracle_and_graph() -> None:
    require_sparkinfer()
    generator = torch.Generator(device="cuda").manual_seed(46_002)
    m, n, k = 2, 128, 128
    source = (
        torch.randn(
            (m, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    ).contiguous()
    b_source = (
        torch.randn(
            (n, k, 1),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 32
    ).contiguous()
    b_quant = quantize_mxfp8_rows_torch(b_source)
    out = torch.empty((m, n, 1), dtype=torch.bfloat16, device="cuda")

    def run() -> torch.Tensor:
        return dense_gemm_fused_quant_a(
            source,
            b_quant.values,
            b_quant.scale_mma,
            out=out,
            mma_tiler_mn=(64, 64),
        )

    run()
    torch.cuda.synchronize()
    expected = _mxfp8_gemm_reference(
        source.unsqueeze(-1), b_quant.values, b_quant.scale_rows
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    source.copy_(
        torch.randn(
            source.shape,
            generator=generator,
            dtype=source.dtype,
            device=source.device,
        )
        / 4
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = _mxfp8_gemm_reference(
        source.unsqueeze(-1), b_quant.values, b_quant.scale_rows
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_cute_migration_dense_grouped_fused_quant_gpu_oracle_and_graph() -> None:
    require_sparkinfer()
    generator = torch.Generator(device="cuda").manual_seed(46_003)
    m, n, k, groups = 2, 128, 128, 2
    source = (
        torch.randn(
            (m, groups, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    ).contiguous()
    b_source = (
        torch.randn(
            (n, k, groups),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 32
    ).contiguous()
    b_quant = quantize_mxfp8_rows_torch(b_source)
    # The dense kernel writes C as physical [L,M,N]; retain that storage
    # contract while exposing the public logical [M,N,L] view.
    out = torch.empty(
        (groups, m, n), dtype=torch.bfloat16, device="cuda"
    ).as_strided((m, n, groups), (n, 1, m * n))

    def run() -> torch.Tensor:
        return dense_gemm_fused_quant_a_grouped(
            source,
            b_quant.values,
            b_quant.scale_mma,
            groups=groups,
            out=out,
            mma_tiler_mn=(64, 64),
        )

    run()
    torch.cuda.synchronize()
    expected = _mxfp8_gemm_reference(
        source.permute(0, 2, 1),
        b_quant.values,
        b_quant.scale_rows,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    source.copy_(
        torch.randn(
            source.shape,
            generator=generator,
            dtype=source.dtype,
            device=source.device,
        )
        / 4
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = _mxfp8_gemm_reference(
        source.permute(0, 2, 1),
        b_quant.values,
        b_quant.scale_rows,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [4, 16])
def test_cute_migration_mxfp8_quant_gpu_oracle_and_graph(m: int) -> None:
    require_sparkinfer()
    generator = torch.Generator(device="cuda").manual_seed(46_100 + m)
    source = (
        torch.randn(
            (m, 128),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    ).contiguous()
    expected = quantize_mxfp8_rows_torch(source)
    actual = empty_mxfp8_rows_for_dense_gemm(m, 128, device="cuda")
    values = actual.values
    scale_rows = actual.scale_rows
    scale_mma = actual.scale_mma

    def run() -> None:
        quantize_mxfp8_rows_cute(source, values, scale_rows, scale_mma)

    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        values.view(torch.uint8), expected.values.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        scale_rows.view(torch.uint8),
        expected.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scale_mma.view(torch.uint8),
        expected.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    source.copy_(
        torch.randn(
            source.shape,
            generator=generator,
            dtype=source.dtype,
            device=source.device,
        )
        / 4
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = quantize_mxfp8_rows_torch(source)
    torch.testing.assert_close(
        values.view(torch.uint8), expected.values.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        scale_rows.view(torch.uint8),
        expected.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scale_mma.view(torch.uint8),
        expected.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )
