"""Unit tests for the DeepGEMM-style expected_m regime hint in dense_gemm's
default tile selector (_select_default_mma_tiler_mn).

These are pure CPU/logic tests (no kernel launch): they pin the per-regime tile
mapping and the M-independence-within-regime contract that lets one compiled
kernel per (N,K,expected_m) be reused for all live M under frozen resolution.
"""
from __future__ import annotations

from b12x.gemm.dense import _select_default_mma_tiler_mn

SM = 188  # RTX PRO 6000 Blackwell
WIDE_N = 4096  # n > 1536 -> the MXFP8 wide-N regime that the hint tunes


def _tile(m, *, expected_m=None, n=WIDE_N):
    return _select_default_mma_tiler_mn(
        m, n, SM, is_mxfp8=True, expected_m=expected_m
    )


def test_expected_m_decode_regime_selects_32x128():
    # expected_m in the decode/small-batch regime -> 32x128 (probe optimum,
    # ~25% faster than 64x128 at M=32..128).
    for em in (2, 16, 32, 64, 128):
        assert _tile(64, expected_m=em) == (32, 128), em


def test_expected_m_single_token_selects_16x128():
    assert _tile(64, expected_m=1) == (16, 128)


def test_expected_m_prefill_regime_selects_64x128():
    for em in (129, 256, 512, 2048, 4096):
        assert _tile(64, expected_m=em) == (64, 128), em


def test_expected_m_is_independent_of_live_m():
    # The whole point: the tile is a function of (N,K,expected_m), NOT live M, so
    # one warmed kernel serves every live M in the regime. For a fixed
    # expected_m, the selected tile must be identical across wildly different
    # live M (16, 512, 4096).
    for em, want in ((64, (32, 128)), (1, (16, 128)), (2048, (64, 128))):
        tiles = {_tile(live_m, expected_m=em) for live_m in (1, 16, 128, 512, 4096)}
        assert tiles == {want}, (em, tiles)


def test_no_hint_preserves_graft_a_default():
    # expected_m=None must reproduce the M-independent Graft A behavior exactly:
    # m==1 -> 16x128, m>=2 -> 64x128 (never the decode 32x128 without a hint, so
    # the one-kernel-per-(N,K) freeze/reuse contract is preserved by default).
    assert _tile(1, expected_m=None) == (16, 128)
    for m in (2, 16, 32, 64, 128, 256, 4096):
        assert _tile(m, expected_m=None) == (64, 128), m


def test_expected_m_ignored_for_narrow_n():
    # The hint only governs the wide-N (n>1536) MXFP8 path; narrow-N keeps the
    # existing occupancy heuristic regardless of expected_m.
    narrow = 1024
    base = _select_default_mma_tiler_mn(64, narrow, SM, is_mxfp8=True)
    hinted = _select_default_mma_tiler_mn(
        64, narrow, SM, is_mxfp8=True, expected_m=64
    )
    assert base == hinted
