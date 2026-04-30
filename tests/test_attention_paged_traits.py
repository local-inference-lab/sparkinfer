from __future__ import annotations

import torch

from b12x.attention.paged.planner import create_paged_plan
from b12x.attention.paged.traits import (
    select_paged_forward_traits,
    select_paged_forward_traits_from_plan,
)

from .test_attention_paged_planner import _make_inputs


def test_paged_decode_fp8_traits_match_flashinfer_shape_rules() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 4096],
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
    )
    traits = select_paged_forward_traits_from_plan(plan)

    assert traits.cta_tile_q == 16
    assert traits.num_warps_q == 1
    assert traits.num_warps_kv == 4
    assert traits.num_mma_q == 1
    assert traits.num_mma_kv == 1
    assert traits.cta_tile_kv == 64
    assert traits.q_smem_bytes == 16 * 256 * 2
    assert traits.shared_storage_bytes == 66048


def test_paged_extend_fp8_traits_expand_kv_tile_without_duplicate_bf16_cache() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    traits = select_paged_forward_traits_from_plan(plan)

    assert traits.cta_tile_q == 64
    assert traits.num_warps_q == 4
    assert traits.num_warps_kv == 1
    assert traits.num_mma_q == 1
    assert traits.num_mma_kv == 2
    assert traits.cta_tile_kv == 32
    assert traits.shared_storage_bytes == 49152


def test_paged_fp8_traits_use_more_kv_mmas_than_bf16_traits() -> None:
    fp8_traits = select_paged_forward_traits(
        cta_tile_q=32,
        head_dim_qk=256,
        head_dim_vo=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    bf16_traits = select_paged_forward_traits(
        cta_tile_q=64,
        head_dim_qk=256,
        head_dim_vo=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        device="cuda",
    )

    assert fp8_traits.num_mma_kv == 4
    assert bf16_traits.num_mma_kv == 1
    assert fp8_traits.cta_tile_kv > bf16_traits.cta_tile_kv
