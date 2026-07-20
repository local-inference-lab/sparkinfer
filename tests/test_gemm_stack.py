from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import cutlass

import b12x.gemm.dense as dense_module
from b12x.cute.intrinsics import quantize_grouped_nvfp4_torch
from b12x.cute.utils import convert_sf_from_mma_layout, get_num_sm
from b12x.gemm.dense import (
    DenseGemmKernel,
    _select_default_dense_gemm_plan,
    _select_default_mma_tiler_mn,
    dense_gemm,
)
from b12x.gemm.wo_projection import (
    dequantize_mxfp8_rows_torch,
    pack_fp8_block_scaled_weight_mxfp8,
    quantize_mxfp8_rows_torch,
)

_FLASHINFER_ROOT = pathlib.Path(__file__).resolve().parents[2] / "flashinfer"
if _FLASHINFER_ROOT.exists():
    sys.path.insert(0, str(_FLASHINFER_ROOT))

from .helpers import require_sm12x


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
    mma_tiler_mn: tuple[int, int] | None = None,
    load_path: str = "tma",
    swap_ab: bool = False,
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
        mma_tiler_mn=mma_tiler_mn,
        load_path=load_path,
        swap_ab=swap_ab,
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
    require_sm12x()
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


@pytest.mark.parametrize("mma_tiler_mn", [(64, 32), (64, 16)])
@pytest.mark.parametrize("load_path", ["tma", "cpasync"])
def test_dense_gemm_fp4_swap_ab_small_tilen_matches_flashinfer_cudnn(
    mma_tiler_mn: tuple[int, int],
    load_path: str,
) -> None:
    require_sm12x()
    mm_fp4 = _require_cudnn_fp4()
    torch.manual_seed(7)

    M, N, K = 32, 64, 128
    lhs, lhs_scale = _make_quantized_operand((1, M, K), dtype=torch.bfloat16)
    rhs, rhs_scale = _make_quantized_operand((1, N, K), dtype=torch.bfloat16)
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)

    dense_out = dense_gemm(
        lhs,
        rhs,
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        mma_tiler_mn=mma_tiler_mn,
        load_path=load_path,
        swap_ab=True,
    )

    packed_a, sfa = lhs
    packed_b, sfb = rhs
    cudnn_out = mm_fp4(
        packed_a[:, :, 0].contiguous(),
        packed_b[:, :, 0].contiguous().T,
        convert_sf_from_mma_layout(sfa, m=M, k=K, num_groups=1),
        convert_sf_from_mma_layout(sfb, m=N, k=K, num_groups=1).T,
        alpha,
        torch.bfloat16,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cudnn",
        use_nvfp4=True,
    )

    torch.testing.assert_close(dense_out[:, :, 0], cudnn_out, rtol=0, atol=0)


def test_dense_gemm_fp4_small_tilen_support_matrix() -> None:
    base = dict(
        ab_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        c_dtype=cutlass.BFloat16,
        cluster_shape_mn=(1, 1),
        n=64,
        k=128,
        l=1,
        a_major="k",
        b_major="k",
        c_major="n",
    )

    assert not DenseGemmKernel.can_implement(
        **base,
        mma_tiler_mn=(64, 32),
        load_path="tma",
        swap_ab=False,
    )
    for tile in ((64, 32), (64, 16)):
        for load_path in ("tma", "cpasync"):
            assert DenseGemmKernel.can_implement(
                **base,
                mma_tiler_mn=tile,
                load_path=load_path,
                swap_ab=True,
            )


def test_dense_gemm_fp8_small_tile_and_swap_support_matrix() -> None:
    base = dict(
        ab_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        sf_vec_size=32,
        c_dtype=cutlass.BFloat16,
        cluster_shape_mn=(1, 1),
        n=1024,
        k=4096,
        l=1,
        a_major="k",
        b_major="k",
        c_major="n",
    )

    for tile in ((16, 64), (16, 128), (32, 64), (32, 128)):
        assert DenseGemmKernel.can_implement(
            **base,
            mma_tiler_mn=tile,
            load_path="tma",
            swap_ab=False,
        )

    assert not DenseGemmKernel.can_implement(
        **base,
        mma_tiler_mn=(64, 32),
        load_path="tma",
        swap_ab=False,
    )
    for tile in ((64, 32), (64, 16), (128, 32), (128, 16)):
        assert DenseGemmKernel.can_implement(
            **base,
            mma_tiler_mn=tile,
            load_path="tma",
            swap_ab=True,
        )

        assert not DenseGemmKernel.can_implement(
            **base,
            mma_tiler_mn=tile,
            load_path="cpasync",
            swap_ab=True,
        )


def test_dense_gemm_mxfp8_bk64_grouped_batches_use_their_own_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm12x()
    torch.manual_seed(29)

    # Use the real grouped WO-A geometry. The manual BK64 path is deliberately
    # shape-gated in production and does not claim arbitrary tiny N/K support.
    m, n, k = 64, 1024, 512
    groups = 4
    group_multipliers = torch.tensor(
        [1.0, 2.0, 4.0, 0.5], device="cuda", dtype=torch.bfloat16
    ).view(1, 1, groups)
    a = torch.randn((m, k, groups), device="cuda", dtype=torch.bfloat16) / 4
    a_q = quantize_mxfp8_rows_torch(a * group_multipliers)
    b_values = (
        torch.randn((groups * n, k), device="cuda", dtype=torch.bfloat16) / 32
    ).to(torch.float8_e4m3fn)
    b_scales = (
        torch.tensor([1.0, 2.0, 4.0, 0.5], device="cuda", dtype=torch.float32)
        .view(groups, 1, 1)
        .expand(groups, n // 128, k // 128)
        .reshape(groups * (n // 128), k // 128)
        .contiguous()
    )
    b_q = pack_fp8_block_scaled_weight_mxfp8(
        b_values,
        b_scales,
        m=n,
        k=k,
        num_groups=groups,
    )
    assert not torch.equal(a_q.scale_rows[0], a_q.scale_rows[1])
    assert not torch.equal(b_q.scale_rows[0], b_q.scale_rows[1])

    # Force the normally shape-gated BK64 specialization so this compact test
    # covers its manual packed-scale address arithmetic for L>1.
    monkeypatch.setattr(dense_module, "_select_mxfp8_tile_k", lambda *_: 64)
    out = dense_gemm(
        (a_q.values, a_q.scale_mma),
        (b_q.values, b_q.scale_mma),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        mma_tiler_mn=(128, 128),
        expected_m=2048,
        sfb_k_replicated=True,
    )
    a_deq = dequantize_mxfp8_rows_torch(a_q.values, a_q.scale_rows)
    b_deq = dequantize_mxfp8_rows_torch(
        b_q.values, b_q.scale_rows
    ).to(torch.bfloat16)
    a_deq = a_deq.to(torch.bfloat16)
    ref = torch.einsum("mkl,nkl->mnl", a_deq, b_deq).to(torch.bfloat16)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


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
    require_sm12x()
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


@pytest.mark.parametrize(
    ("m", "n", "k", "expected_tile", "expected_swap"),
    [
        (1, 4096, 5376, (64, 128), False),
        (1, 2048, 5376, (64, 128), False),
        (1, 1536, 4096, (64, 128), False),
        (1, 1024, 5376, (64, 64), False),
        (1, 512, 5376, (64, 32), True),
        (1, 512, 4096, (64, 32), True),
        (1, 512, 1024, (64, 64), False),
        (2, 512, 5376, (64, 64), False),
    ],
)
def test_default_dense_fp4_plan_handles_m1_probe_regimes(
    m: int,
    n: int,
    k: int,
    expected_tile: tuple[int, int],
    expected_swap: bool,
) -> None:
    plan = _select_default_dense_gemm_plan(
        m,
        n,
        k,
        188,
        is_mxfp8=False,
    )
    assert plan.mma_tiler_mn == expected_tile
    assert plan.load_path == "tma"
    assert plan.swap_ab is expected_swap


@pytest.mark.parametrize(
    ("n", "k"),
    [
        (4096, 5376),
        (2048, 5376),
        (1024, 5376),
        (512, 5376),
        (1536, 4096),
        (16384, 1024),
        (1024, 4096),
        (4096, 4096),
        (6144, 1536),
        (7168, 512),
    ],
)
def test_default_dense_fp8_plan_handles_m1_known_shapes(n: int, k: int) -> None:
    plan = _select_default_dense_gemm_plan(
        1,
        n,
        k,
        188,
        is_mxfp8=True,
    )
    assert plan.mma_tiler_mn == (16, 64)
    assert plan.load_path == "tma"
    assert plan.swap_ab is False
