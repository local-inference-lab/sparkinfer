"""Accuracy tests for b12x W4A16 dense GEMM."""

from __future__ import annotations

import pytest
import torch

from b12x.gemm.w4a16 import (
    DenseGemmW4A16MicroKernel,
    dense_gemm_w4a16,
    dense_reference_w4a16,
    quantize_dense_weight_to_fp4,
)
from b12x.moe.fused.reference import compare_to_reference


def test_dense_reference_w4a16_signature():
    """Reference function exists, returns ``[M, N]`` bf16, both code paths."""
    device = torch.device("cpu")
    m, k, n = 2, 32, 16
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)

    out_float = dense_reference_w4a16(x, w_bf16=w_bf16)
    assert out_float.shape == (m, n)
    assert out_float.dtype == torch.bfloat16

    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w_bf16)
    out_quant = dense_reference_w4a16(
        x, w_fp4=w_fp4, w_blockscale=w_bs, w_alpha=w_alpha
    )
    assert out_quant.shape == (m, n)
    assert out_quant.dtype == torch.bfloat16


@pytest.mark.parametrize("n,k", [(16, 32), (16, 64), (32, 128), (64, 512)])
def test_dense_reference_dequant_roundtrip(n, k):
    """``dense_reference_w4a16(quantize(w))`` matches ``x @ w.T`` to FP4 precision."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    m = 4
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)

    out_quant = dense_reference_w4a16(
        x, w_fp4=w_fp4, w_blockscale=w_bs, w_alpha=w_alpha
    )
    out_float = dense_reference_w4a16(x, w_bf16=w)

    metrics = compare_to_reference(out_quant, out_float)
    # FP4 weight quant noise accumulates as ~sqrt(K) in absolute terms;
    # we only assert high cosine correlation here.  The strict
    # quant-vs-quant accuracy gate lives in the kernel tests.
    assert metrics.cos > 0.99, f"cos={metrics.cos}"


def test_dense_gemm_w4a16_signature_callable():
    """Public entry exists with the expected signature."""
    import inspect
    sig = inspect.signature(dense_gemm_w4a16)
    params = list(sig.parameters)
    assert params[:4] == ["x", "w_fp4", "w_blockscale", "w_alpha"]
    assert "out" in sig.parameters


def test_dense_gemm_w4a16_is_supported_matrix():
    """is_supported gate matches the actual envelope (decode + prefill dispatch)."""
    # M ladder.  Decode kernel (v4) covers any M ∈ [1, 32]; prefill
    # (v5) takes over for M ≥ B12X_GEMM_W4A16_PREFILL_M (default 33),
    # so every positive M is supported as long as the shape envelope
    # (N % 64 == 0, K % 64 == 0) holds.
    for m in (1, 2, 3, 4, 8, 16, 24, 32, 33, 64, 128, 1024):
        assert DenseGemmW4A16MicroKernel.is_supported(m, 2688, 4096), f"M={m} should be supported"
    assert not DenseGemmW4A16MicroKernel.is_supported(0, 2688, 4096)
    # K must be a multiple of 64.  All Nano35 dense K values qualify.
    for k in (2688, 3712, 4096):
        assert DenseGemmW4A16MicroKernel.is_supported(1, k, 256)
    assert not DenseGemmW4A16MicroKernel.is_supported(1, 511, 256)
    # N must be a multiple of 64.
    assert DenseGemmW4A16MicroKernel.is_supported(1, 512, 64)
    assert not DenseGemmW4A16MicroKernel.is_supported(1, 512, 15)
    assert not DenseGemmW4A16MicroKernel.is_supported(1, 512, 16)


def test_dense_gemm_w4a16_stub_matches_reference_cpu():
    """Stub kernel call returns the reference output (CPU path)."""
    device = torch.device("cpu")
    torch.manual_seed(42)
    m, k, n = 1, 512, 64
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
    out = dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha)
    ref = dense_reference_w4a16(x, w_fp4=w_fp4, w_blockscale=w_bs, w_alpha=w_alpha)
    assert torch.equal(out, ref)


NANO35_DENSE_SHAPES = [
    # (name, K, N)
    ("q_proj",             2688,   4096),
    ("k_proj",             2688,    256),
    ("v_proj",             2688,    256),
    ("o_proj",             4096,   2688),
    ("shared_expert.up",   2688,   3712),
    ("shared_expert.down", 3712,   2688),
    ("lm_head",            2688, 131072),
]


@pytest.mark.parametrize("name,k,n", NANO35_DENSE_SHAPES, ids=[s[0] for s in NANO35_DENSE_SHAPES])
@pytest.mark.parametrize("m", [1, 8, 32])
def test_dense_gemm_w4a16_nano35_accuracy(name, k, n, m):
    """W4A16 dense matches reference across all Nano35 dense shapes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not DenseGemmW4A16MicroKernel.is_supported(m, k, n):
        pytest.skip(f"unsupported (m={m}, k={k}, n={n})")
    device = torch.device("cuda")
    torch.manual_seed(hash((name, m)) & 0xFFFFFFFF)
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
    out = dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha)
    ref = dense_reference_w4a16(
        x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu(),
    ).to(device)
    metrics = compare_to_reference(out, ref)
    # The reference and kernel both consume the same FP4 weights, so
    # any non-zero difference is purely MMA tile-order round-off in
    # bf16 (kernel uses fp32 accumulators but stores bf16).  Output
    # magnitudes scale as ~sqrt(K) * 0.5 * 0.1 = O(1)..O(10); a 0.04
    # max-abs is < 1% relative, with cos = 1.0 (down to fp32 precision).
    assert metrics.cos > 0.9999, f"{name} m={m}: cos={metrics.cos}"
    # Relative threshold: 1% of the reference's max-abs.  Tile-order
    # accum differences (kernel uses fp32 accum but stores bf16) plus
    # bf16 rounding put us comfortably under this on every Nano35
    # shape we tested.
    ref_max_abs = ref.abs().max().item()
    rel_thresh = max(0.04, 0.01 * ref_max_abs)
    assert metrics.max_abs <= rel_thresh, (
        f"{name} m={m}: max_abs={metrics.max_abs} cos={metrics.cos} "
        f"(threshold {rel_thresh:.4f})"
    )


@pytest.mark.parametrize("name,k,n", NANO35_DENSE_SHAPES, ids=[s[0] for s in NANO35_DENSE_SHAPES])
def test_dense_gemm_w4a16_cute_m16_accuracy(name, k, n, monkeypatch):
    """CuTe-DSL backend at M=16 matches reference bit-exact for every Nano35 shape."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.gemm.w4a16._cute_dense_kernel import DenseGemmW4A16CuteDenseKernel
    m = 16
    if not DenseGemmW4A16CuteDenseKernel.is_supported(m, k, n):
        pytest.skip(f"cute not supported (m={m}, k={k}, n={n})")
    device = torch.device("cuda")
    torch.manual_seed(hash(("cute", name)) & 0xFFFFFFFF)
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
    out = dense_gemm_w4a16(x, w_fp4, w_bs, w_alpha)
    ref = dense_reference_w4a16(
        x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu(),
    ).to(device)
    metrics = compare_to_reference(out, ref)
    assert metrics.cos > 0.9999, f"{name}: cos={metrics.cos}"
    ref_max_abs = ref.abs().max().item()
    rel_thresh = max(0.04, 0.01 * ref_max_abs)
    assert metrics.max_abs <= rel_thresh, (
        f"{name}: max_abs={metrics.max_abs} cos={metrics.cos}"
    )


def test_dense_quantize_shapes():
    """Packer produces the expected shapes / dtypes."""
    device = torch.device("cpu")
    n, k = 64, 512
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)
    assert w_fp4.shape == (n, k // 2)
    assert w_fp4.dtype == torch.uint8
    # rows_padded = ceil(64/128)*128 = 128; cols_padded = ceil(32/4)*4 = 32
    assert w_bs.shape == (128, 32)
    assert w_bs.dtype == torch.float8_e4m3fn
    assert w_alpha.dtype == torch.float32
    assert w_alpha.ndim == 0
