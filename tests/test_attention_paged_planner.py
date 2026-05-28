from __future__ import annotations

import pytest
import torch

from b12x.attention.paged.planner import (
    PagedPlanBudget,
    build_decode_chunk_pages_lut,
    create_paged_plan,
    decode_chunk_pages_for_graph,
    infer_paged_mode,
    resolve_decode_graph_ctas_per_sm,
)
from b12x.integration.attention import PagedAttentionWorkspace


def _make_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int = 64,
    q_heads: int = 8,
    kv_heads: int = 1,
    head_dim_qk: int = 256,
    head_dim_vo: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = "cuda"
    batch = len(q_seqlens)
    total_q = sum(q_seqlens)
    q = torch.randn(total_q, q_heads, head_dim_qk, dtype=dtype, device=device)
    max_pages = max((cache_len + page_size - 1) // page_size for cache_len in cache_seqlens)
    num_pages = sum((cache_len + page_size - 1) // page_size for cache_len in cache_seqlens) + 8
    k_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim_qk, dtype=torch.float32, device=device
    ).to(kv_dtype)
    v_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim_vo, dtype=torch.float32, device=device
    ).to(kv_dtype)
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    cursor = 0
    for request_idx, cache_len in enumerate(cache_seqlens):
        req_pages = (cache_len + page_size - 1) // page_size
        page_ids = torch.arange(cursor, cursor + req_pages, dtype=torch.int32, device=device)
        cursor += req_pages
        page_table[request_idx, :req_pages] = page_ids
        page_table[request_idx, req_pages:] = page_ids[-1]
    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    offsets = [0]
    for q_len in q_seqlens:
        offsets.append(offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q


def test_paged_infers_decode_mode() -> None:
    _, _, _, _, _, cu_seqlens_q = _make_inputs(q_seqlens=[1, 1, 1], cache_seqlens=[64, 128, 192])
    assert infer_paged_mode(cu_seqlens_q) == "decode"


def test_paged_infers_extend_mode() -> None:
    _, _, _, _, _, cu_seqlens_q = _make_inputs(q_seqlens=[1, 6], cache_seqlens=[64, 128])
    assert infer_paged_mode(cu_seqlens_q) == "extend"


def test_paged_decode_plan_emits_exact_split_metadata() -> None:
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

    assert plan.mode == "decode"
    assert plan.cta_tile_q == 16
    assert plan.split_kv is True
    assert plan.kv_chunk_size == 8 * 64
    assert plan.request_indices == (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)
    assert plan.qo_tile_indices == (0,) * 12
    assert plan.kv_tile_indices == (0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7)
    assert plan.merge_indptr == (0, 4, 12)
    assert plan.o_indptr == (0, 4, 12)
    assert plan.total_num_partial_rows == 12


def test_paged_workspace_shapes_follow_plan_metadata() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 6],
        cache_seqlens=[2048, 8192],
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    workspace = PagedAttentionWorkspace.for_tensors(
        mode="extend",
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert workspace.lse.shape == (8, 7)
    assert workspace.kv_chunk_size_ptr.item() == 128 * 64
    assert workspace.total_num_rows_ptr.item() == 7
    assert workspace.request_indices.shape[0] == plan.new_batch_size
    assert workspace.merge_indptr.shape[0] == plan.total_q + 1
    assert workspace.o_indptr.shape[0] == page_table.shape[0] + 1
    assert workspace.tmp_output is None
    assert workspace.tmp_lse is None


def test_paged_fp8_auto_chunk_heuristic_uses_larger_decode_chunks() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[8192] * 8,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "decode"
    assert plan.kv_chunk_size > 0
    assert plan.split_kv is True


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_paged_short_extend_plan_uses_cta_tile_q_16(
    kv_dtype: torch.dtype,
) -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[4],
        cache_seqlens=[4096],
        kv_dtype=kv_dtype,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.cta_tile_q == 16
    assert plan.split_kv is False


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_paged_verify_plan_uses_decode_style_split_kv(
    kv_dtype: torch.dtype,
) -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[4],
        cache_seqlens=[65536],
        kv_dtype=kv_dtype,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="verify",
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )

    assert plan.mode == "verify"
    assert plan.cta_tile_q == 16
    assert plan.split_kv is True
    assert plan.kv_chunk_size == 22 * 64
    assert plan.graph_ctas_per_sm == 2
    assert plan.total_num_partial_rows > 0
    assert plan.new_batch_size > 2


def test_paged_plan_disables_split_kv_when_merge_backend_is_unsupported() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[8192] * 8,
        q_heads=48,
        kv_heads=8,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )

    assert plan.split_kv is False


def test_paged_graph_budget_is_independent_of_cache_length() -> None:
    short_inputs = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[2048] * 8,
        kv_dtype=torch.bfloat16,
    )
    long_inputs = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[32768] * 8,
        kv_dtype=torch.bfloat16,
    )

    short_plan = create_paged_plan(
        *short_inputs,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=2,
    )
    long_plan = create_paged_plan(
        *long_inputs,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=2,
    )

    expected_budget = int(torch.cuda.get_device_properties("cuda").multi_processor_count) * 2
    assert short_plan.graph_ctas_per_sm == 2
    assert long_plan.graph_ctas_per_sm == 2
    assert short_plan.max_batch_size_if_split == expected_budget
    assert long_plan.max_batch_size_if_split == expected_budget
    assert short_plan.padded_batch_size == long_plan.padded_batch_size == expected_budget


def test_paged_graph_mode_falls_back_when_heuristic_overflows_budget() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[128000] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=2,
    )

    assert plan.new_batch_size <= plan.padded_batch_size


def test_paged_extend_plan_respects_fixed_partial_row_budget() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[128],
        cache_seqlens=[128 * 64],
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        plan_budget=PagedPlanBudget(
            max_total_q=128,
            max_batch=1,
            max_page_table_width=128,
            max_work_items=4096,
            max_partial_rows=512,
        ),
    )

    assert plan.mode == "extend"
    assert plan.split_kv is False
    assert plan.total_num_partial_rows == 0


def test_paged_extend_plan_budget_can_force_nosplit() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[64],
        cache_seqlens=[256 * 64],
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        plan_budget=PagedPlanBudget(
            max_total_q=64,
            max_batch=1,
            max_page_table_width=256,
            max_work_items=4096,
            max_partial_rows=0,
        ),
    )

    assert plan.mode == "extend"
    assert plan.split_kv is False
    assert plan.total_num_partial_rows == 0


def test_paged_extend_plan_rejects_fixed_split_smaller_than_full_span() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[8192] * 8,
        kv_dtype=torch.bfloat16,
    )

    with pytest.raises(
        ValueError,
        match="extend fixed_split_size must cover the full effective KV span",
    ):
        create_paged_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            fixed_split_size=8,
        )


def test_paged_graph_mode_uses_decode_graph_heuristic() -> None:
    batch = 7
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=[8192] * batch,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )
    expected_budget = int(torch.cuda.get_device_properties("cuda").multi_processor_count) * 2
    assert plan.graph_ctas_per_sm == 2
    assert plan.max_batch_size_if_split == expected_budget
    assert plan.kv_chunk_size == 5 * 64


def test_decode_graph_chunk_pages_for_graph_uses_heuristic() -> None:
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            batch=4,
            page_size=64,
            head_dim_qk=256,
            head_dim_vo=256,
            gqa_group_size=8,
            max_effective_kv_pages=32,
        )
        == 1
    )
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            batch=4,
            page_size=64,
            head_dim_qk=256,
            head_dim_vo=256,
            gqa_group_size=8,
            max_effective_kv_pages=256,
        )
        == 6
    )
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            batch=8,
            page_size=64,
            head_dim_qk=192,
            head_dim_vo=128,
            gqa_group_size=16,
            max_effective_kv_pages=256,
        )
        == 32
    )


def test_decode_graph_chunk_pages_uses_finer_fp8_minimax_bs1_splits() -> None:
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            batch=1,
            page_size=64,
            head_dim_qk=128,
            head_dim_vo=128,
            gqa_group_size=6,
            max_effective_kv_pages=257,
        )
        == 6
    )
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            batch=2,
            page_size=64,
            head_dim_qk=128,
            head_dim_vo=128,
            gqa_group_size=6,
            max_effective_kv_pages=257,
        )
        == 9
    )


def test_decode_graph_ctas_per_sm_uses_smaller_minimax_bs1_to_bs4_budget() -> None:
    for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        assert (
            resolve_decode_graph_ctas_per_sm(
                kv_dtype=kv_dtype,
                batch=1,
                page_size=64,
                head_dim_qk=128,
                head_dim_vo=128,
                gqa_group_size=6,
            )
            == 1
        )
        assert (
            resolve_decode_graph_ctas_per_sm(
                kv_dtype=kv_dtype,
                batch=5,
                page_size=64,
                head_dim_qk=128,
                head_dim_vo=128,
                gqa_group_size=6,
            )
            == 2
        )
    assert (
        resolve_decode_graph_ctas_per_sm(
            kv_dtype=torch.bfloat16,
            batch=2,
            page_size=64,
            head_dim_qk=128,
            head_dim_vo=128,
            gqa_group_size=6,
        )
        == 6
    )
    assert (
        resolve_decode_graph_ctas_per_sm(
            kv_dtype=torch.float8_e4m3fn,
            batch=2,
            page_size=64,
            head_dim_qk=128,
            head_dim_vo=128,
            gqa_group_size=6,
        )
        == 1
    )


def test_build_decode_chunk_pages_lut_uses_heuristic() -> None:
    lut = build_decode_chunk_pages_lut(
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        batch=8,
        page_size=64,
        head_dim_qk=192,
        head_dim_vo=128,
        gqa_group_size=16,
        max_effective_kv_pages=16,
    )

    assert lut[:8] == (1, 1, 1, 1, 1, 1, 1, 1)
    assert lut[8:] == (2, 2, 2, 2, 2, 2, 2, 2)


@pytest.mark.parametrize(
    ("q_seqlens", "cache_seqlens", "kv_dtype"),
    [
        ([1] * 8, [8192] * 8, torch.float8_e4m3fn),
        ([1] * 8, [32768] * 8, torch.bfloat16),
        ([6] * 8, [8192] * 8, torch.float8_e4m3fn),
        ([6] * 8, [32768] * 8, torch.bfloat16),
    ],
)
def test_paged_non_policy_chunk_selection_still_produces_valid_split_kv_plans(
    q_seqlens: list[int],
    cache_seqlens: list[int],
    kv_dtype: torch.dtype,
) -> None:
    q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = _make_inputs(
        q_seqlens=q_seqlens,
        cache_seqlens=cache_seqlens,
        kv_dtype=kv_dtype,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens_t,
        cu_seqlens_q,
    )

    assert plan.kv_chunk_size > 0
    assert plan.new_batch_size >= page_table.shape[0]
    if q_seqlens[0] == 1:
        assert plan.total_num_partial_rows >= plan.total_q
        assert plan.split_kv is True
    else:
        assert plan.total_num_partial_rows == 0
        assert plan.split_kv is False
