"""Unit tests for the DeepGEMM-style expected_m regime hint in dense_gemm's
default tile selector (_select_default_mma_tiler_mn).

These are pure CPU/logic tests (no kernel launch): they pin the per-regime tile
mapping and the M-independence-within-regime contract that lets one compiled
kernel per (N,K,expected_m) be reused for all live M under frozen resolution.
"""
from __future__ import annotations

import cutlass
import pytest

import sparkinfer._lib.dense_gemm as dense_module
from sparkinfer._lib.dense_gemm import (
    _DenseGemmFusedQuantALaunch,
    _DenseGemmLaunch,
    _DenseGemmPolicy,
    _dense_gemm_policy_for,
    _dense_gemm_target_occupancy,
    _dense_spark_policy_for_sm_count,
    _select_default_mma_tiler_mn,
    _select_mxfp8_tile_k,
    _validate_mxfp8_bk64_plan,
)

SM = 188  # RTX PRO 6000 Blackwell
SPARK_SM = 20  # DGX Spark GB10
WIDE_N = 4096  # n > 1536 -> the MXFP8 wide-N regime that the hint tunes


def _tile(m, *, expected_m=None, n=WIDE_N, k=None, sm_count=SM):
    return _select_default_mma_tiler_mn(
        m, n, sm_count, is_mxfp8=True, expected_m=expected_m, k=k
    )


def _bk64_launch(
    *,
    launch_type=_DenseGemmLaunch,
    policy: _DenseGemmPolicy | None = None,
    sfb_k_reuse: bool = True,
    n: int = 16384,
    k: int = 1024,
    l: int = 1,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    b_tile_major: bool = False,
) -> _DenseGemmLaunch:
    if policy is None:
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=False,
            direct_one_m_tile_scheduler=False,
            use_m1_non_tma=False,
            split_k_slices=1,
            split_k_atomic_bf16=False,
            large_m_unroll=True,
        )
    return launch_type(
        n=n,
        k=k,
        l=l,
        c_l=1,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_dtype=cutlass.BFloat16,
        alpha_dtype=cutlass.Float32,
        sf_vec_size=32,
        mma_k=32,
        tile_k=64,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        policy=policy,
        sm_count=SM,
        sm_version="sm_120",
        load_path="tma",
        swap_ab=False,
        sfb_k_reuse=sfb_k_reuse,
        b_tile_major=b_tile_major,
    )


def _compile_key_differences(
    lhs: _DenseGemmLaunch, rhs: _DenseGemmLaunch
) -> list[tuple[object, object]]:
    return [
        (lhs_value, rhs_value)
        for lhs_value, rhs_value in zip(
            lhs.compile_key(), rhs.compile_key(), strict=True
        )
        if lhs_value != rhs_value
    ]


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


def test_dense_compile_key_separates_replicated_sfb_reuse():
    generic_scales = _bk64_launch(sfb_k_reuse=False)
    replicated_scales = _bk64_launch()

    assert _compile_key_differences(generic_scales, replicated_scales) == [
        (False, True)
    ]


def test_dense_compile_key_covers_atom_shape_environment(monkeypatch):
    monkeypatch.setattr(dense_module, "_SPARKINFER_DENSE_ATOM_24", False)
    atom_42 = _bk64_launch()
    monkeypatch.setattr(dense_module, "_SPARKINFER_DENSE_ATOM_24", True)
    atom_24 = _bk64_launch()

    assert _compile_key_differences(atom_42, atom_24) == [(False, True)]


def test_fused_quant_compile_key_is_distinct_and_exhaustive():
    ordinary = _bk64_launch()
    fused = _bk64_launch(launch_type=_DenseGemmFusedQuantALaunch)

    assert fused.compile_key()[0] == "fused_quant_a"
    # Indexes 1-2 are the fused-only A inner-span and wide-M1 layout fields.
    assert fused.compile_key()[1] == 0
    assert fused.compile_key()[2] is False
    assert fused.compile_key()[3:] == ordinary.compile_key()


def test_expected_m_short_k_large_n_uses_production_bk64_plan():
    # RTX keeps the BM128/BK64 q_b prefill plan measured by the SM120 audit.
    for em in (2048, 4096, 8192):
        for live_m in (1, 64, 4096):
            assert _tile(
                live_m,
                expected_m=em,
                n=16384,
                k=1024,
            ) == (128, 128)
            assert _select_mxfp8_tile_k(live_m, 16384, 1024, em, SM) == 64
        policy = _dense_gemm_policy_for(
            m=64,
            n=16384,
            k=1024,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            sm_count=SM,
            expected_m=em,
        )
        assert policy.large_m_unroll == (em >= 8192)


def test_expected_m_short_k_large_n_keeps_spark_prefill_plan():
    # The candidate's BM64/BK128 and M>=4096 unroll choices remain selected on
    # DGX Spark instead of being reverted globally to the RTX winners.
    for em in (2048, 4096, 8192):
        for live_m in (1, 64, 4096):
            assert _tile(
                live_m,
                expected_m=em,
                n=16384,
                k=1024,
                sm_count=SPARK_SM,
            ) == (64, 128)
            assert (
                _select_mxfp8_tile_k(live_m, 16384, 1024, em, SPARK_SM)
                == 128
            )
        policy = _dense_gemm_policy_for(
            m=64,
            n=16384,
            k=1024,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=(64, 128),
            cluster_shape_mn=(1, 1),
            sm_count=SPARK_SM,
            expected_m=em,
        )
        assert policy.large_m_unroll == (em >= 4096)


def test_short_k_two_cta_occupancy_is_spark_only_for_q_b_decode():
    kwargs = dict(
        n=16384,
        k=1024,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        tile_k=128,
        mma_tiler_mn=(16, 128),
        cluster_shape_mn=(1, 1),
        load_path="tma",
        swap_ab=False,
        b_tile_major=False,
    )
    assert _dense_gemm_target_occupancy(sm_count=SPARK_SM, **kwargs) == 2
    assert _dense_gemm_target_occupancy(sm_count=SM, **kwargs) == 1
    assert _dense_spark_policy_for_sm_count(SPARK_SM)
    assert not _dense_spark_policy_for_sm_count(SM)


def test_wo_b_prefill_switches_to_bm128_bk64_at_2k():
    # Match the specialized DeepGEMM O-projection schedule without changing
    # the 1K schedule that already wins end to end.
    n, k = 4096, 4096
    assert _tile(1024, expected_m=1024, n=n, k=k) == (64, 128)
    assert _select_mxfp8_tile_k(1024, n, k, 1024, SM) == 128
    for em in (2048, 4096, 8192):
        tiles = {
            _tile(live_m, expected_m=em, n=n, k=k)
            for live_m in (1, 64, 2048, 8192)
        }
        assert tiles == {(128, 128)}, (n, k, em, tiles)
        assert _select_mxfp8_tile_k(1, n, k, em, SM) == 64


def test_grouped_wo_a_prefill_keeps_bm64_bk128():
    # The DeepGEMM schedule loses on sparkinfer's grouped WO-A kernel at every probed
    # prefill size, so keep the established narrow-N schedule.
    n, k = 1024, 512
    for em in (2048, 4096, 8192):
        assert _tile(1, expected_m=em, n=n, k=k) == (64, 128)
        assert _select_mxfp8_tile_k(1, n, k, em, SM) == 128


def test_wo_bk64_override_is_exact_shape_only():
    for n, k in ((1024, 640), (1152, 512), (4096, 3968), (4224, 4096)):
        assert _select_mxfp8_tile_k(2048, n, k, 2048, SM) == 128


def test_short_k_1024_and_2048_hints_have_stable_distinct_keys():
    def specialization(live_m: int, expected_m: int | None):
        tile = _tile(live_m, expected_m=expected_m, n=16384, k=1024)
        tile_k = _select_mxfp8_tile_k(live_m, 16384, 1024, expected_m, SM)
        policy = _dense_gemm_policy_for(
            m=live_m,
            n=16384,
            k=1024,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=tile,
            cluster_shape_mn=(1, 1),
            sm_count=SM,
            expected_m=expected_m,
        )
        return tile, tile_k, policy

    # Persistent live shapes share one specialization for a fixed hint.
    hint_1024 = {specialization(m, 1024) for m in (16, 1024, 2048, 8192)}
    hint_2048 = {specialization(m, 2048) for m in (16, 1024, 2048, 8192)}
    assert len(hint_1024) == 1
    assert len(hint_2048) == 1
    assert next(iter(hint_1024))[:2] == ((64, 128), 128)
    assert next(iter(hint_2048))[:2] == ((128, 128), 64)
    assert hint_1024 != hint_2048


def test_short_k_no_hint_does_not_cross_bk64_cache_boundary():
    # The no-hint API promises one prefill kernel across live M. BK64 is only
    # selected by an explicit regime hint, so crossing live M=2048 cannot cause
    # a new tile, tile-K, or policy key under frozen resolution.
    specializations = set()
    for live_m in (16, 1024, 2048, 4096, 8192):
        tile = _tile(live_m, expected_m=None, n=16384, k=1024)
        tile_k = _select_mxfp8_tile_k(live_m, 16384, 1024, None, SM)
        policy = _dense_gemm_policy_for(
            m=live_m,
            n=16384,
            k=1024,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=tile,
            cluster_shape_mn=(1, 1),
            sm_count=SM,
            expected_m=None,
        )
        specializations.add((tile, tile_k, policy))
    assert specializations == {
        (
            (64, 128),
            128,
            _DenseGemmPolicy(False, False, False, 1, True, True),
        )
    }


def test_bk64_rejects_unvalidated_short_row_and_swapped_tiles():
    _validate_mxfp8_bk64_plan(64, (128, 128), False)
    _validate_mxfp8_bk64_plan(64, (128, 64), False)
    _validate_mxfp8_bk64_plan(128, (64, 128), False)
    with pytest.raises(ValueError, match="requires an unswapped 128-row tile"):
        _validate_mxfp8_bk64_plan(64, (64, 128), False)
    with pytest.raises(ValueError, match="requires an unswapped 128-row tile"):
        _validate_mxfp8_bk64_plan(64, (128, 64), True)


def test_no_hint_persistent_policy_keeps_unroll_m_independent():
    # Public expected_m=None warms one persistent-policy kernel and reuses it
    # across live prefill sizes. In particular, crossing M=4096 must not change
    # the compile key solely to toggle mainloop unrolling.
    policies = {
        _dense_gemm_policy_for(
            m=live_m,
            n=1536,
            k=128,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=(64, 64),
            cluster_shape_mn=(1, 1),
            sm_count=SM,
            expected_m=None,
        )
        for live_m in (16, 512, 1824, 4096, 8192)
    }
    assert len(policies) == 1
    assert next(iter(policies)).large_m_unroll


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
