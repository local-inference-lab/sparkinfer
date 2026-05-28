from __future__ import annotations

import pathlib
import sys

import pytest
import torch

from b12x.cute.fp4 import quantize_grouped_nvfp4_torch
from b12x.cute.utils import convert_sf_from_mma_layout, get_num_sm
from b12x.gemm.dense import _select_default_mma_tiler_mn, dense_gemm

_FLASHINFER_ROOT = pathlib.Path(__file__).resolve().parents[2] / "flashinfer"
if _FLASHINFER_ROOT.exists():
    sys.path.insert(0, str(_FLASHINFER_ROOT))

from .helpers import require_sm120


def _import_flashinfer_gemm():
    try:
        from flashinfer.gemm.gemm_base import CUDNN_AVAILABLE
        from flashinfer.gemm import mm_fp4
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"FlashInfer GEMM unavailable: {exc}")
    return CUDNN_AVAILABLE, mm_fp4


def _require_cudnn_fp4():
    CUDNN_AVAILABLE, mm_fp4 = _import_flashinfer_gemm()
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN Python bindings not installed")
    try:
        from flashinfer.gemm.gemm_base import _check_cudnn_fp4_availability
        _check_cudnn_fp4_availability()
    except RuntimeError as e:
        pytest.skip(f"cuDNN FP4 not available: {e}")
    return mm_fp4


def _make_quantized_operand(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    source = torch.randn(shape, device="cuda", dtype=dtype) / 4
    row_counts = torch.full((shape[0],), shape[1], dtype=torch.int32, device=source.device)
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32,
        device=source.device,
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return (packed, scales), global_scale


def _run_dense_gemm(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    *,
    c_dtype_str: str = "bfloat16",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)
    return dense_gemm(
        lhs,
        rhs,
        out=out,
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=c_dtype_str,
        sf_vec_size=16,
    )


@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),
    (256, 128, 128),
    (128, 256, 128),
    (128, 128, 256),
    (256, 256, 256),
    (256, 512, 128),
    (128, 256, 512),
    (512, 256, 256),
    (256, 256, 512),
])
@pytest.mark.parametrize("c_dtype_str", ["bfloat16", "float16"])
def test_dense_gemm_matches_flashinfer_cudnn(
    M: int, N: int, K: int, c_dtype_str: str,
) -> None:
    require_sm120()
    mm_fp4 = _require_cudnn_fp4()
    torch.manual_seed(42)

    lhs, lhs_scale = _make_quantized_operand((1, M, K), dtype=torch.bfloat16)
    rhs, rhs_scale = _make_quantized_operand((1, N, K), dtype=torch.bfloat16)
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)
    c_dtype = torch.bfloat16 if c_dtype_str == "bfloat16" else torch.float16

    dense_out = dense_gemm(
        lhs,
        rhs,
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=c_dtype_str,
        sf_vec_size=16,
    )

    packed_a, sfa = lhs
    packed_b, sfb = rhs

    a_fp4 = packed_a[:, :, 0].contiguous()
    b_fp4 = packed_b[:, :, 0].contiguous()

    sfa_2d = convert_sf_from_mma_layout(sfa, m=M, k=K, num_groups=1)
    sfb_2d = convert_sf_from_mma_layout(sfb, m=N, k=K, num_groups=1)

    cudnn_out = mm_fp4(
        a_fp4,
        b_fp4.T,
        sfa_2d,
        sfb_2d.T,
        alpha,
        c_dtype,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cudnn",
        use_nvfp4=True,
    )

    torch.testing.assert_close(dense_out[:, :, 0], cudnn_out, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("gate_shape", "down_shape"),
    [
        ((32, 2048, 512), (32, 1024, 2048)),
    ],
)
def test_dense_gemm_shared_expert_pair_replays_under_cuda_graph(
    gate_shape: tuple[int, int, int],
    down_shape: tuple[int, int, int],
) -> None:
    require_sm120()
    torch.manual_seed(1234)

    gate_m, gate_n, gate_k = gate_shape
    down_m, down_n, down_k = down_shape
    assert gate_m == down_m
    assert gate_n == down_k

    gate_lhs, gate_lhs_scale = _make_quantized_operand((1, gate_m, gate_k), dtype=torch.bfloat16)
    gate_rhs, gate_rhs_scale = _make_quantized_operand((1, gate_n, gate_k), dtype=torch.bfloat16)
    down_lhs, down_lhs_scale = _make_quantized_operand((1, down_m, down_k), dtype=torch.bfloat16)
    down_rhs, down_rhs_scale = _make_quantized_operand((1, down_n, down_k), dtype=torch.bfloat16)

    eager_gate = _run_dense_gemm(gate_lhs, gate_rhs, gate_lhs_scale, gate_rhs_scale)
    eager_down = _run_dense_gemm(down_lhs, down_rhs, down_lhs_scale, down_rhs_scale)
    torch.cuda.synchronize()

    graph_gate = torch.empty_like(eager_gate)
    graph_down = torch.empty_like(eager_down)

    # Prime the compiled kernels before capture to match the serving warmup path.
    _run_dense_gemm(gate_lhs, gate_rhs, gate_lhs_scale, gate_rhs_scale)
    _run_dense_gemm(down_lhs, down_rhs, down_lhs_scale, down_rhs_scale)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_dense_gemm(
            gate_lhs,
            gate_rhs,
            gate_lhs_scale,
            gate_rhs_scale,
            out=graph_gate,
        )
        _run_dense_gemm(
            down_lhs,
            down_rhs,
            down_lhs_scale,
            down_rhs_scale,
            out=graph_down,
        )

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_gate, eager_gate, rtol=0, atol=0)
    torch.testing.assert_close(graph_down, eager_down, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("m", "n", "sm_count", "expected"),
    [
        (2, 4096, 48, (64, 128)),
        (64, 4096, 48, (64, 128)),
        (96, 4096, 48, (128, 128)),
        (2, 1024, 48, (64, 64)),
    ],
)
def test_default_dense_tile_selector_handles_small_m_wide_n(
    m: int,
    n: int,
    sm_count: int,
    expected: tuple[int, int],
) -> None:
    assert _select_default_mma_tiler_mn(m, n, sm_count, is_mxfp8=False) == expected
