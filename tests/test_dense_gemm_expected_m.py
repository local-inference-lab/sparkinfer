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
    # expected_m in the small-batch regime (9..128) -> 32x128 (probe optimum,
    # ~25% faster than 64x128 at M=32..128).
    for em in (16, 32, 64, 128):
        assert _tile(64, expected_m=em) == (32, 128), em


def test_expected_m_tiny_m_decode_selects_probe_tiles():
    # Exact single-token decode uses the flushed common-shape winner (16x64).
    # The broader tiny-M regime keeps the prior 16x128 specialization.
    assert _tile(64, expected_m=1) == (16, 64)
    for em in (2, 4, 8):
        assert _tile(64, expected_m=em) == (16, 128), em


def test_expected_m_prefill_regime_selects_64x128():
    for em in (129, 256, 512, 2048, 4096):
        assert _tile(64, expected_m=em) == (64, 128), em


def test_expected_m_is_independent_of_live_m():
    # The whole point: the tile is a function of (N,K,expected_m), NOT live M, so
    # one warmed kernel serves every live M in the regime. For a fixed
    # expected_m, the selected tile must be identical across wildly different
    # live M (16, 512, 4096).
    for em, want in ((64, (32, 128)), (1, (16, 64)), (2048, (64, 128))):
        tiles = {_tile(live_m, expected_m=em) for live_m in (1, 16, 128, 512, 4096)}
        assert tiles == {want}, (em, tiles)


def test_no_hint_preserves_graft_a_default():
    # expected_m=None preserves the M-independent Graft A behavior outside the
    # tiny standalone decode range: m=1 -> 16x64, m=2..8 -> 16x128,
    # and m>=16 -> 64x128.
    assert _tile(1, expected_m=None) == (16, 64)
    for m in (2, 4, 8):
        assert _tile(m, expected_m=None) == (16, 128), m
    for m in (16, 32, 64, 128, 256, 4096):
        assert _tile(m, expected_m=None) == (64, 128), m


def test_expected_m_prefill_hint_for_narrow_n():
    # Narrow-N has its own occupancy heuristic. Exact M=1 uses the common-shape
    # decode winner, while declared prefill still moves to the prefill tile.
    narrow = 1024
    base = _select_default_mma_tiler_mn(64, narrow, SM, is_mxfp8=True)
    decode = _select_default_mma_tiler_mn(
        64, narrow, SM, is_mxfp8=True, expected_m=1
    )
    small = _select_default_mma_tiler_mn(
        64, narrow, SM, is_mxfp8=True, expected_m=64
    )
    prefill = _select_default_mma_tiler_mn(
        64, narrow, SM, is_mxfp8=True, expected_m=512
    )
    no_hint_decode = _select_default_mma_tiler_mn(1, narrow, SM, is_mxfp8=True)
    assert base == (64, 64)
    assert decode == (16, 64)
    assert small == base
    assert prefill == (64, 128)
    assert no_hint_decode == (16, 64)
