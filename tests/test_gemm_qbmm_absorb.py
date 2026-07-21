from __future__ import annotations

import pytest
import torch

from b12x.gemm.qbmm_absorb import (
    _DEFAULT_WARMUP_BATCH_SIZES,
    _MAX_BATCH,
    warmup_qbmm_absorb,
)

from .helpers import require_sm120

pytestmark = require_sm120

# GLM-5.2 TP4 MLA geometry (the v1 kernel's fixed head layout).
N, HS, P, V, L = 16, 448, 192, 256, 512


def _make_pack(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Random kv_b mxfp8 pack: fp8 e4m3 values + per-32 e8m0 scale bytes."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    values = (
        torch.randn(N * HS, L, device="cuda", generator=g, dtype=torch.float32)
        * 0.1
    ).to(torch.float8_e4m3fn)
    scales = torch.randint(
        118, 132, (N * HS, L // 32), device="cuda", generator=g, dtype=torch.uint8
    )
    return values, scales


def _dequant_reference(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The kernel's bitwise contract: one RN bf16 multiply per element.

    (The float32-path helper dequantize_mxfp8_rows_torch is NOT the contract:
    f32-multiply-then-round can differ from the bf16 multiply in the last ulp.)
    """
    return values.to(torch.bfloat16) * scales.view(torch.float8_e8m0fnu).to(
        torch.bfloat16
    ).repeat_interleave(32, dim=1)


def _absorbed_pair(values: torch.Tensor, scales: torch.Tensor):
    wf = _dequant_reference(values, scales)
    kvw = wf.T.view(L, N, P + V)
    w_uk, w_uv = kvw.split([P, V], dim=-1)
    ukt = w_uk.permute(1, 2, 0).contiguous()  # (N, P, L)
    uv = w_uv.transpose(0, 1).contiguous()  # (N, L, V)
    return ukt, uv


@pytest.fixture(scope="module")
def pack() -> tuple[torch.Tensor, torch.Tensor]:
    values, scales = _make_pack()
    warmup_qbmm_absorb(values, scales)
    return values, scales


def test_dequant_bitwise_vs_torch_reference(pack) -> None:
    """One-hot activations turn the GEMM into a pure dequant readout: each
    fp32 accumulator receives exactly one product, so the output must be
    bitwise equal to the bf16-chain dequant of the pack."""
    values, scales = _make_pack(seed=11)
    ref = _dequant_reference(values, scales)

    # ukt: one-hot over p reads dequantized pack rows [448n + p, :].
    for p0 in range(0, P, 32):
        a = torch.zeros(N, 32, P, device="cuda", dtype=torch.bfloat16)
        for b in range(32):
            a[:, b, p0 + b] = 1.0
        out = torch.empty(N, 32, L, device="cuda", dtype=torch.bfloat16)
        torch.ops.b12x.qbmm_absorb_ukt(a, values, scales, out)
        want = torch.stack(
            [ref[torch.arange(N) * HS + p0 + b] for b in range(32)], dim=1
        )
        assert torch.equal(out, want), f"ukt dequant mismatch at p-chunk {p0}"

    # uv: one-hot over l reads dequantized pack columns of rows
    # [448n + 192 + v, :].
    for l0 in range(0, L, 32):
        a = torch.zeros(N, 32, L, device="cuda", dtype=torch.bfloat16)
        for b in range(32):
            a[:, b, l0 + b] = 1.0
        out_backing = torch.empty(32, N, V, device="cuda", dtype=torch.bfloat16)
        torch.ops.b12x.qbmm_absorb_uv(
            a, values, scales, out_backing.transpose(0, 1)
        )
        for b in range(4):  # spot-check 4 of the 32 one-hots per chunk
            for n in range(0, N, 5):
                want = ref[n * HS + P : n * HS + P + V, l0 + b]
                got = out_backing[b, n]
                assert torch.equal(got, want), (
                    f"uv dequant mismatch at l={l0 + b}, head {n}"
                )


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 25, 32])
def test_envelope_vs_cublas(pack, batch: int) -> None:
    """Acceptance: elementwise error vs fp64 ground truth no worse than the
    cuBLAS bf16 bmm of the materialized pair (x1.05 slack).  Bitwise equality
    with cuBLAS is NOT the contract -- different fp32 accumulation orders
    round differently in the last bf16 ulp."""
    values, scales = pack
    ukt, uv = _absorbed_pair(values, scales)

    a1 = torch.randn(N, batch, P, device="cuda", dtype=torch.bfloat16)
    o1 = torch.empty(N, batch, L, device="cuda", dtype=torch.bfloat16)
    torch.ops.b12x.qbmm_absorb_ukt(a1, values, scales, o1)
    r64 = torch.bmm(a1.double(), ukt.double())
    err_q = (o1.double() - r64).abs().max().item()
    err_c = (torch.bmm(a1, ukt).double() - r64).abs().max().item()
    assert err_q <= err_c * 1.05 + 1e-12

    a2 = torch.randn(N, batch, L, device="cuda", dtype=torch.bfloat16)
    o2 = torch.empty(batch, N, V, device="cuda", dtype=torch.bfloat16)
    torch.ops.b12x.qbmm_absorb_uv(a2, values, scales, o2.transpose(0, 1))
    r64 = torch.bmm(a2.double(), uv.double())
    err_q = (o2.transpose(0, 1).double() - r64).abs().max().item()
    err_c = (torch.bmm(a2, uv).double() - r64).abs().max().item()
    assert err_q <= err_c * 1.05 + 1e-12


def test_replay_determinism(pack) -> None:
    values, scales = pack
    a = torch.randn(N, 8, P, device="cuda", dtype=torch.bfloat16)
    o1 = torch.empty(N, 8, L, device="cuda", dtype=torch.bfloat16)
    o2 = torch.empty_like(o1)
    torch.ops.b12x.qbmm_absorb_ukt(a, values, scales, o1)
    torch.ops.b12x.qbmm_absorb_ukt(a, values, scales, o2)
    assert torch.equal(o1, o2)


def test_cudagraph_capture_replay(pack) -> None:
    """Serving requirement: a graph captured over a pre-warmed size must
    replay the kernel correctly for fresh input values in the static
    buffers."""
    values, scales = pack
    a = torch.zeros(N, 4, P, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(N, 4, L, device="cuda", dtype=torch.bfloat16)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops.b12x.qbmm_absorb_ukt(a, values, scales, out)

    fresh = torch.randn(N, 4, P, device="cuda", dtype=torch.bfloat16)
    a.copy_(fresh)
    g.replay()
    torch.cuda.synchronize()

    eager = torch.empty_like(out)
    torch.ops.b12x.qbmm_absorb_ukt(fresh, values, scales, eager)
    assert torch.equal(out, eager)


def test_capture_compile_miss_raises(pack) -> None:
    """A compile miss inside stream capture must raise, never silently
    capture around the JIT (a graph recorded that way replays garbage --
    observed as MTP acceptance 0.000 in serving)."""
    values, scales = pack
    b = 13  # deliberately un-warmed
    assert b not in _DEFAULT_WARMUP_BATCH_SIZES
    a = torch.zeros(N, b, P, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(N, b, L, device="cuda", dtype=torch.bfloat16)
    g = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="during\\s+CUDA-graph capture"):
        with torch.cuda.graph(g):
            torch.ops.b12x.qbmm_absorb_ukt(a, values, scales, out)


def test_warmup_covers_batch_envelope(pack) -> None:
    values, scales = pack
    assert max(_DEFAULT_WARMUP_BATCH_SIZES) == _MAX_BATCH
    # After module-fixture warmup, every default size launches with no
    # compile (and therefore stays capture-legal).
    for b in _DEFAULT_WARMUP_BATCH_SIZES:
        a = torch.zeros(N, b, P, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(N, b, L, device="cuda", dtype=torch.bfloat16)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.ops.b12x.qbmm_absorb_ukt(a, values, scales, out)
