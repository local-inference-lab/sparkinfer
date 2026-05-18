"""Accuracy tests for the b12x W4A16 dense GEMM **Marlin-style** kernel (v6).

Mirrors ``tests/test_dense_gemm_w4a16_prefill.py`` for the v6 strip-port
of the MoE W4A16 kernel.  Runs every Nano3.5 dense linear shape × a
ladder of M values across the dispatch envelope.
"""

from __future__ import annotations

import pytest
import torch

from b12x.gemm.w4a16 import (
    dense_reference_w4a16,
    quantize_dense_weight_to_fp4,
)
from b12x.gemm.w4a16._cute_marlin_kernel import DenseGemmW4A16CuteMarlinKernel
from b12x.moe.fused.reference import compare_to_reference


NANO35_DENSE_SHAPES = [
    # (name, K, N) -- the v6-eligible subset (N >= 1024).  k_proj/v_proj
    # (N=256) is excluded since v6 isn't dispatched there in production.
    ("q_proj",             2688,   4096),
    ("o_proj",             4096,   2688),
    ("shared_expert.up",   2688,   3712),
    ("shared_expert.down", 3712,   2688),
    ("mamba_in_proj",      2688,  10304),
    ("mamba_output_proj",  4096,   2688),
]


@pytest.mark.parametrize(
    "name,k,n", NANO35_DENSE_SHAPES, ids=[s[0] for s in NANO35_DENSE_SHAPES],
)
@pytest.mark.parametrize("m", [64, 128, 256, 512, 1024, 2048, 4096])
def test_marlin_kernel_accuracy(name, k, n, m):
    """v6 marlin kernel matches the W4A16 reference for every Nano35 shape × M.

    Gates (same as v5 prefill):
    * ``cos > 0.9999`` — kernel + reference consume the same FP4 weights;
      any deviation is fp32-accum + bf16-round noise.
    * ``max_abs ≤ max(0.04, 1% × ref.abs().max())``.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not DenseGemmW4A16CuteMarlinKernel.is_supported(m, k, n):
        pytest.skip(f"v6 not supported (m={m}, k={k}, n={n})")

    device = torch.device("cuda")
    torch.manual_seed(hash((name, m)) & 0xFFFFFFFF)
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)

    kernel = DenseGemmW4A16CuteMarlinKernel()
    out = kernel(x, w_fp4, w_bs, w_alpha)
    ref = dense_reference_w4a16(
        x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu(),
    ).to(device)

    metrics = compare_to_reference(out, ref)
    assert metrics.cos > 0.9999, f"{name} m={m}: cos={metrics.cos}"
    ref_max_abs = ref.abs().max().item()
    rel_thresh = max(0.04, 0.01 * ref_max_abs)
    assert metrics.max_abs <= rel_thresh, (
        f"{name} m={m}: max_abs={metrics.max_abs} cos={metrics.cos} "
        f"(threshold {rel_thresh:.4f})"
    )


def test_marlin_is_supported_envelope():
    """v6 is_supported accepts the Nano3.5 (N % 64, K % 64) envelope."""
    kc = DenseGemmW4A16CuteMarlinKernel
    assert kc.is_supported(64, 2688, 4096)
    assert kc.is_supported(2048, 2688, 4096)
    assert kc.is_supported(4096, 4096, 2688)
    # N % 64 != 0
    assert not kc.is_supported(128, 2688, 100)
    # K % 64 != 0
    assert not kc.is_supported(128, 100, 2688)
    # M <= 0
    assert not kc.is_supported(0, 2688, 4096)


def test_marlin_padding_safe_for_m_below_cta_m_size():
    """M=64 (= default cta_m_size) writes exactly M rows; no OOB.

    The kernel pads x up to cta_m_size internally but passes the real M
    to compile, so the C-write path's block_valid_rows clipping only
    touches rows [0, M).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(123)
    m, k, n = 64, 2688, 3712
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)

    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    kernel = DenseGemmW4A16CuteMarlinKernel()
    kernel(x, w_fp4, w_bs, w_alpha, out=out)
    assert not torch.isnan(out).any(), "kernel left NaNs (incomplete coverage)"

    ref = dense_reference_w4a16(
        x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu(),
    ).to(device)
    metrics = compare_to_reference(out, ref)
    assert metrics.cos > 0.9999, f"cos={metrics.cos}"


@pytest.mark.parametrize("m", [40, 96, 200])
def test_marlin_handles_m_not_divisible_by_cta_m_size(m):
    """M values not aligned to cta_m_size are correctly handled via x-padding."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(hash(("nondiv", m)) & 0xFFFFFFFF)
    k, n = 2688, 4096
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_fp4, w_bs, w_alpha = quantize_dense_weight_to_fp4(w)

    kernel = DenseGemmW4A16CuteMarlinKernel()
    out = kernel(x, w_fp4, w_bs, w_alpha)
    assert out.shape == (m, n)
    ref = dense_reference_w4a16(
        x.cpu(), w_fp4=w_fp4.cpu(), w_blockscale=w_bs.cpu(), w_alpha=w_alpha.cpu(),
    ).to(device)
    metrics = compare_to_reference(out, ref)
    assert metrics.cos > 0.9999, f"m={m}: cos={metrics.cos}"


def test_micro_dispatches_prefill_above_threshold(monkeypatch):
    """When M ≥ PREFILL_M, micro routes through the v6 marlin prefill."""
    from importlib import reload
    from b12x.gemm.w4a16 import micro as micro_mod
    reload(micro_mod)
    # M ≥ default 256 → prefill (v6) for any v6-supported shape.
    assert micro_mod._use_prefill(256)
    assert micro_mod._use_prefill(2048)
    # M < 256 → decode (v4).
    assert not micro_mod._use_prefill(255)
    assert not micro_mod._use_prefill(1)
