from __future__ import annotations

import gc
import math

import cutlass.base_dsl.dsl as cutlass_dsl
import pytest
import torch

from b12x.cute.compiler import clear_compile_cache, compile_cache_info
from b12x.attention.paged.api import _build_extend_forward_kernel, _resolve_native_fp8_attention_mma_flags
from b12x.attention.paged.traits import select_paged_forward_traits_from_plan
from b12x.attention.paged.reference import paged_attention_reference
from b12x.integration.attention import (
    PagedAttentionWorkspace,
    clear_attention_caches,
    infer_paged_mode,
)

from .helpers import require_sm120


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _lse_base2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * math.log(2.0)


def _make_paged_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int,
    q_heads: int = 8,
    kv_heads: int = 1,
    head_dim: int = 256,
    head_dim_qk: int | None = None,
    head_dim_vo: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    page_table_width: int | None = None,
    num_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(q_seqlens) != len(cache_seqlens):
        raise ValueError("q_seqlens and cache_seqlens must have the same length")
    torch.manual_seed(seed)
    device = "cuda"
    batch = len(q_seqlens)
    total_q = sum(q_seqlens)
    head_dim_qk = head_dim if head_dim_qk is None else head_dim_qk
    head_dim_vo = head_dim if head_dim_vo is None else head_dim_vo
    q = torch.randn(total_q, q_heads, head_dim_qk, device=device, dtype=dtype) / 4

    pages_per_request = [(cache_len + page_size - 1) // page_size for cache_len in cache_seqlens]
    max_pages = max(pages_per_request, default=0)
    if page_table_width is not None:
        if page_table_width < max_pages:
            raise ValueError(
                f"page_table_width={page_table_width} is smaller than the required max_pages={max_pages}"
            )
        max_pages = page_table_width
    total_pages_needed = sum(pages_per_request)
    if num_pages is None:
        num_pages = max(1, total_pages_needed * 2)
    if num_pages < total_pages_needed:
        raise ValueError(f"num_pages={num_pages} is smaller than the required total {total_pages_needed}")

    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim_qk, device=device, dtype=dtype) / 4
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim_vo, device=device, dtype=dtype) / 4
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    page_order = torch.randperm(num_pages, device=device)
    cursor = 0
    for request_idx, num_req_pages in enumerate(pages_per_request):
        if num_req_pages == 0:
            continue
        page_ids = page_order[cursor : cursor + num_req_pages].to(torch.int32)
        cursor += num_req_pages
        page_table[request_idx, :num_req_pages] = page_ids
        page_table[request_idx, num_req_pages:] = page_ids[-1]

    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    q_offsets = [0]
    for q_len in q_seqlens:
        q_offsets.append(q_offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(q_offsets, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q


def _quantize_paged_kv_cache_e4m3(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _max_pages = page_table.shape
    _, page_size, kv_heads, _head_dim = k_cache.shape
    finfo = torch.finfo(torch.float8_e4m3fn)
    k_quant = torch.empty_like(k_cache, dtype=torch.float8_e4m3fn)
    v_quant = torch.empty_like(v_cache, dtype=torch.float8_e4m3fn)
    k_descale = torch.ones((batch, kv_heads), dtype=torch.float32, device=k_cache.device)
    v_descale = torch.ones((batch, kv_heads), dtype=torch.float32, device=v_cache.device)
    for request_idx in range(batch):
        cache_len = int(cache_seqlens[request_idx].item())
        num_pages = (cache_len + page_size - 1) // page_size
        if num_pages == 0:
            continue
        page_ids = page_table[request_idx, :num_pages].to(torch.long)
        k_pages = k_cache.index_select(0, page_ids).to(torch.float32)
        v_pages = v_cache.index_select(0, page_ids).to(torch.float32)
        k_scale = k_pages.abs().amax(dim=(0, 1, 3)) / finfo.max
        v_scale = v_pages.abs().amax(dim=(0, 1, 3)) / finfo.max
        k_scale = torch.where(k_scale > 0, k_scale, torch.ones_like(k_scale))
        v_scale = torch.where(v_scale > 0, v_scale, torch.ones_like(v_scale))
        k_descale[request_idx] = k_scale
        v_descale[request_idx] = v_scale
        k_quant[page_ids] = (k_pages / k_scale.view(1, 1, kv_heads, 1)).clamp(
            min=finfo.min,
            max=finfo.max,
        ).to(torch.float8_e4m3fn)
        v_quant[page_ids] = (v_pages / v_scale.view(1, 1, kv_heads, 1)).clamp(
            min=finfo.min,
            max=finfo.max,
        ).to(torch.float8_e4m3fn)
    return k_quant.contiguous(), v_quant.contiguous(), k_descale.contiguous(), v_descale.contiguous()


def _make_workspace(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    use_cuda_graph: bool = False,
) -> PagedAttentionWorkspace:
    return PagedAttentionWorkspace.for_tensors(
        mode=infer_paged_mode(cu_seqlens_q),
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=use_cuda_graph,
    )


def _make_req_to_token(
    page_table: torch.Tensor,
    *,
    row_stride: int,
    page_size: int,
) -> torch.Tensor:
    batch, max_pages = [int(dim) for dim in page_table.shape]
    req_to_token = torch.zeros((batch, row_stride), dtype=torch.int64, device=page_table.device)
    for req_idx in range(batch):
        for page_idx in range(max_pages):
            req_to_token[req_idx, page_idx * page_size] = int(page_table[req_idx, page_idx].item()) * page_size
    return req_to_token


@pytest.mark.parametrize("fixed_split_size", [None, 4])
def test_paged_workspace_matches_reference_for_qwen_like_extend_shape(
    fixed_split_size: int | None,
) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=23,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=fixed_split_size,
    )
    output = torch.empty_like(q)
    out, lse = workspace.run(q, k_cache, v_cache, output=output)
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.03
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


def test_paged_workspace_exposes_primary_backend_metadata() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=29,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens_t, cu_seqlens_q)
    plan = workspace.plan

    assert plan.num_q_heads == 8
    assert plan.num_kv_heads == 1
    assert plan.gqa_group_size == 8
    assert plan.head_dim_qk == 256
    assert plan.head_dim_vo == 256
    assert plan.mode == "extend"
    assert plan.cta_tile_q == 64
    assert plan.kv_chunk_size == 2 * 64
    assert plan.split_kv is False
    assert plan.total_q == q.shape[0]
    assert plan.page_table_shape == tuple(page_table.shape)


def test_decode_cuda_graph_matches_reference_for_minimax_m2_head128_shape() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * 16,
        cache_seqlens=[128] * 16,
        page_size=64,
        q_heads=24,
        kv_heads=4,
        head_dim=128,
        seed=127,
        page_table_width=2,
        num_pages=48,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=16,
        total_q_capacity=16,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
    )

    bound_page_table = page_table.clone()
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    output = torch.empty_like(q)
    workspace.run(q, k_cache, v_cache, output=output)
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )

    output.zero_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=output,
            prepare_decode_graph_metadata=True,
        )

    graph.replay()
    torch.cuda.synchronize()

    assert torch.allclose(
        output.to(torch.float32),
        ref_out.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.allclose(
        _lse_base2_to_natural(workspace.current_lse_view()),
        ref_lse,
        atol=3e-2,
        rtol=3e-2,
    )


def test_decode_cuda_graph_matches_reference_for_minimax_m2_head128_shape_fp8_kv() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * 4,
        cache_seqlens=[128] * 4,
        page_size=64,
        q_heads=24,
        kv_heads=4,
        head_dim=128,
        seed=131,
        page_table_width=2,
        num_pages=16,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=4,
        total_q_capacity=4,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
    )

    bound_page_table = page_table.clone()
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    output = torch.empty_like(q)
    workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )

    output.zero_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_fp8,
            v_fp8,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
            prepare_decode_graph_metadata=True,
        )

    graph.replay()
    torch.cuda.synchronize()

    assert torch.allclose(
        output.to(torch.float32),
        ref_out.to(torch.float32),
        atol=8e-2,
        rtol=8e-2,
    )
    assert torch.allclose(
        _lse_base2_to_natural(workspace.current_lse_view()),
        ref_lse,
        atol=8e-2,
        rtol=8e-2,
    )
    assert _cosine_similarity(output, ref_out) >= 0.999


@pytest.mark.parametrize(
    ("turbo_value", "batch", "cache_len", "expect_native_fp8_qk"),
    [
        (None, 2, 16384, False),
        ("true", 2, 16384, False),
        ("1", 2, 16384, True),
        ("1", 4, 16384, False),
    ],
)
def test_decode_native_fp8_attention_dispatch_tracks_turbo_gate_batch_and_chunk_regime_eager(
    monkeypatch: pytest.MonkeyPatch,
    turbo_value: str | None,
    batch: int,
    cache_len: int,
    expect_native_fp8_qk: bool,
) -> None:
    require_sm120()
    clear_attention_caches()
    if turbo_value is None:
        monkeypatch.delenv("B12X_TURBO_ATTN", raising=False)
    else:
        monkeypatch.setenv("B12X_TURBO_ATTN", turbo_value)

    q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=[cache_len] * batch,
        page_size=64,
        seed=37 + batch + cache_len // 64,
        num_pages=(batch * cache_len) // 64,
    )
    k_fp8, v_fp8, _k_descale, _v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens_t,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=False,
    )
    workspace.prepare(page_table, cache_seqlens_t, cu_seqlens_q)

    use_native_fp8_qk, use_native_fp8_pv, decode_runtime_chunk_guard = _resolve_native_fp8_attention_mma_flags(
        plan=workspace.plan,
    )

    assert workspace.plan.mode == "decode"
    assert workspace.plan.kv_dtype == torch.float8_e4m3fn
    assert (workspace.plan.kv_chunk_size < 11 * workspace.plan.page_size) == (cache_len == 16384)
    assert use_native_fp8_qk is expect_native_fp8_qk
    assert use_native_fp8_pv is (expect_native_fp8_qk and workspace.plan.kv_chunk_size <= 384)
    assert decode_runtime_chunk_guard is False


def test_paged_workspace_matches_reference_for_fp8_kv_cache() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=123,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)
    out, lse = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    torch.cuda.synchronize()

    assert workspace.plan.kv_dtype == torch.float8_e4m3fn
    assert (out - ref_out).abs().max().item() <= 0.05
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(out, ref_out) >= 0.9999


def test_paged_workspace_matches_reference_for_bf16_large_extend_shape() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[128],
        cache_seqlens=[128],
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        seed=29,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    assert workspace.plan.mode == "extend"
    assert workspace.plan.split_kv is False
    assert workspace.plan.kv_chunk_size == 2 * 64

    output = torch.empty_like(q)
    out, lse = workspace.run(q, k_cache, v_cache, output=output)
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.02
    assert _cosine_similarity(out, ref_out) >= 0.9999


def test_paged_workspace_matches_reference_for_bf16_nosplit_extend_shape() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[4],
        cache_seqlens=[4096],
        page_size=64,
        q_heads=48,
        kv_heads=8,
        head_dim=128,
        seed=31,
        page_table_width=64,
        num_pages=512,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=64,
    )

    assert workspace.plan.mode == "extend"
    assert workspace.plan.split_kv is False

    output = torch.empty_like(q)
    out, lse = workspace.run(q, k_cache, v_cache, output=output)
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


def test_paged_fixed_capacity_extend_reuses_larger_compile_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
    monkeypatch.delenv("B12X_CUTE_COMPILE_MEMORY_CACHE", raising=False)
    clear_compile_cache()
    clear_attention_caches()

    q_large, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[128],
        cache_seqlens=[128],
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        seed=43,
    )
    q_small = q_large[:64].contiguous()
    cu_seqlens_q_small = torch.tensor([0, 64], dtype=torch.int32, device=q_large.device)

    workspace = PagedAttentionWorkspace.for_eager_extend_capacity(
        device=q_large.device,
        dtype=q_large.dtype,
        kv_dtype=k_cache.dtype,
        num_q_heads=q_large.shape[1],
        num_kv_heads=k_cache.shape[2],
        head_dim_qk=q_large.shape[2],
        head_dim_vo=v_cache.shape[3],
        page_size=int(k_cache.shape[1]),
        max_total_q=128,
        max_batch=1,
        max_page_table_width=int(page_table.shape[1]),
        num_cache_pages=int(k_cache.shape[0]),
    )

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output_large = torch.empty_like(q_large)
    workspace.run(q_large, k_cache, v_cache, output=output_large)
    first_compile_misses = compile_cache_info()["compile_misses"]
    assert first_compile_misses >= 1

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q_small)
    output_small = torch.empty_like(q_small)
    workspace.run(q_small, k_cache, v_cache, output=output_small)

    assert compile_cache_info()["compile_misses"] == first_compile_misses


def test_paged_workspace_matches_reference_for_fp8_nosplit_extend_shape() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[4],
        cache_seqlens=[4096],
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        seed=37,
        page_table_width=64,
        num_pages=256,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=64,
    )

    assert workspace.plan.mode == "extend"
    assert workspace.plan.split_kv is False
    assert workspace.plan.kv_dtype == torch.float8_e4m3fn

    output = torch.empty_like(q)
    out, lse = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.05
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(out, ref_out) >= 0.9999


def test_paged_mode_inference_distinguishes_decode_from_extend() -> None:
    require_sm120()
    clear_attention_caches()

    _, _, _, _, _, cu_seqlens_decode = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 64, 64, 64],
        page_size=64,
        seed=33,
    )
    _, _, _, _, _, cu_seqlens_extend = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=35,
    )

    assert infer_paged_mode(cu_seqlens_decode) == "decode"
    assert infer_paged_mode(cu_seqlens_extend) == "extend"


def test_decode_prepare_uses_small_q_tile() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 96, 128, 70],
        page_size=64,
        seed=39,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    assert workspace.plan.mode == "decode"
    assert workspace.plan.cta_tile_q == 16
    assert workspace.plan.kv_chunk_size == 2 * 64


@pytest.mark.parametrize(
    ("window_left", "use_sink"),
    [(-1, False), (128, True)],
)
def test_paged_workspace_matches_reference_for_mimo_v25_decode_shape(
    window_left: int,
    use_sink: bool,
) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 129, 192, 193],
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=117 + max(window_left, 0) + int(use_sink),
        page_table_width=4,
        num_pages=32,
    )
    attention_sink_bias = None
    if use_sink:
        attention_sink_bias = torch.linspace(
            -0.3,
            0.4,
            steps=q.shape[1],
            device=q.device,
            dtype=torch.float32,
        )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)

    assert workspace.plan.mode == "decode"
    assert workspace.plan.head_dim_qk == 192
    assert workspace.plan.head_dim_vo == 128
    assert workspace.plan.window_left == window_left
    if window_left >= 0:
        assert workspace.plan.kv_window_start_tokens == (0, 0, 0, 64)

    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)
    out, lse = workspace.run(
        q,
        k_cache,
        v_cache,
        output=output,
        attention_sink_bias=attention_sink_bias,
    )
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


@pytest.mark.parametrize(
    ("window_left", "use_sink"),
    [(-1, False), (128, True)],
)
def test_paged_workspace_matches_reference_for_mimo_v25_decode_shape_fp8_kv(
    window_left: int,
    use_sink: bool,
) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 129, 192, 193],
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=317 + max(window_left, 0) + int(use_sink),
        page_table_width=4,
        num_pages=32,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    attention_sink_bias = None
    if use_sink:
        attention_sink_bias = torch.linspace(
            -0.3,
            0.4,
            steps=q.shape[1],
            device=q.device,
            dtype=torch.float32,
        )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)

    assert workspace.plan.mode == "decode"
    assert workspace.plan.head_dim_qk == 192
    assert workspace.plan.head_dim_vo == 128
    assert workspace.plan.kv_dtype == torch.float8_e4m3fn
    assert workspace.plan.window_left == window_left
    if window_left >= 0:
        assert workspace.plan.kv_window_start_tokens == (0, 0, 0, 64)

    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)
    out, lse = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
        attention_sink_bias=attention_sink_bias,
    )
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.06
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.06
    assert _cosine_similarity(out, ref_out) >= 0.9995


@pytest.mark.parametrize(
    ("window_left", "use_sink"),
    [(-1, False), (128, True)],
)
def test_paged_workspace_matches_reference_for_mimo_v25_prefill_shape(
    window_left: int,
    use_sink: bool,
) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[39, 64, 8],
        cache_seqlens=[39, 192, 320],
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=211 + max(window_left, 0) + int(use_sink),
        page_table_width=5,
        num_pages=32,
    )
    attention_sink_bias = None
    if use_sink:
        attention_sink_bias = torch.linspace(
            -0.2,
            0.5,
            steps=q.shape[1],
            device=q.device,
            dtype=torch.float32,
        )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)

    assert workspace.plan.mode == "extend"
    assert workspace.plan.head_dim_qk == 192
    assert workspace.plan.head_dim_vo == 128
    assert workspace.plan.window_left == window_left
    if window_left >= 0:
        assert workspace.plan.kv_window_start_tokens == (0, 0, 128)

    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)
    out, lse = workspace.run(
        q,
        k_cache,
        v_cache,
        output=output,
        attention_sink_bias=attention_sink_bias,
    )
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


@pytest.mark.parametrize(
    ("window_left", "use_sink"),
    [(-1, False), (128, True)],
)
def test_paged_workspace_matches_reference_for_mimo_v25_prefill_shape_fp8_kv(
    window_left: int,
    use_sink: bool,
) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[39, 64, 8],
        cache_seqlens=[39, 192, 320],
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=411 + max(window_left, 0) + int(use_sink),
        page_table_width=5,
        num_pages=32,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    attention_sink_bias = None
    if use_sink:
        attention_sink_bias = torch.linspace(
            -0.2,
            0.5,
            steps=q.shape[1],
            device=q.device,
            dtype=torch.float32,
        )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)

    assert workspace.plan.mode == "extend"
    assert workspace.plan.head_dim_qk == 192
    assert workspace.plan.head_dim_vo == 128
    assert workspace.plan.kv_dtype == torch.float8_e4m3fn
    assert workspace.plan.window_left == window_left
    if window_left >= 0:
        assert workspace.plan.kv_window_start_tokens == (0, 0, 128)

    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)
    out, lse = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
        attention_sink_bias=attention_sink_bias,
    )
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.06
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.10
    assert _cosine_similarity(out, ref_out) >= 0.9985


def test_paged_workspace_matches_reference_for_mimo_v25_fp8_long_prefill_padded_k_stride() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[32],
        cache_seqlens=[4096],
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=419,
        page_table_width=64,
        num_pages=30000,
    )
    page_ids = (torch.arange(21000, 21064, device=q.device, dtype=torch.int32) * 37 % 29000) + 500
    page_table[0, :64] = page_ids
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    traits = select_paged_forward_traits_from_plan(workspace.plan)
    assert workspace.plan.mode == "extend"
    assert workspace.plan.cta_tile_q == 128
    assert workspace.plan.kv_chunk_size == 4096
    assert workspace.plan.kv_dtype == torch.float8_e4m3fn
    assert traits.upcast_stride_k == 16
    assert traits.upcast_stride_v == 8

    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)
    out, lse = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.06
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.10
    assert _cosine_similarity(out, ref_out) >= 0.9985


def test_workspace_eager_path_grows_with_larger_shape() -> None:
    require_sm120()
    clear_attention_caches()

    small = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 64, 64, 64],
        page_size=64,
        seed=41,
    )
    large = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[2048, 2048, 4096, 4096],
        page_size=64,
        seed=43,
        page_table_width=64,
        num_pages=512,
    )
    q_s, k_s, v_s, pt_s, cs_s, cu_s = small
    q_l, k_l, v_l, pt_l, cs_l, cu_l = large
    workspace = _make_workspace(
        q=q_s,
        k_cache=k_s,
        v_cache=v_s,
        cu_seqlens_q=cu_s,
    )
    workspace.prepare(pt_s, cs_s, cu_s)
    request_items_small = int(workspace.request_indices.shape[0])
    merge_small = int(workspace.merge_indptr.shape[0])

    workspace.prepare(pt_l, cs_l, cu_l)
    assert int(workspace.request_indices.shape[0]) >= request_items_small
    assert int(workspace.merge_indptr.shape[0]) >= merge_small


def test_fixed_capacity_extend_workspace_reuses_large_buffers() -> None:
    require_sm120()
    clear_attention_caches()

    small = _make_paged_inputs(
        q_seqlens=[32, 32],
        cache_seqlens=[64 * 16, 64 * 16],
        page_size=64,
        seed=91,
        page_table_width=64,
        num_pages=256,
    )
    large = _make_paged_inputs(
        q_seqlens=[32, 32],
        cache_seqlens=[64 * 64, 64 * 64],
        page_size=64,
        seed=93,
        page_table_width=64,
        num_pages=256,
    )
    q_s, k_s, v_s, pt_s, cs_s, cu_s = small
    _q_l, _k_l, _v_l, pt_l, cs_l, cu_l = large
    workspace = PagedAttentionWorkspace.for_fixed_capacity(
        mode="extend",
        device=q_s.device,
        dtype=q_s.dtype,
        kv_dtype=k_s.dtype,
        num_q_heads=int(q_s.shape[1]),
        num_kv_heads=int(k_s.shape[2]),
        head_dim_qk=int(q_s.shape[2]),
        head_dim_vo=int(v_s.shape[3]),
        page_size=int(k_s.shape[1]),
        max_total_q=int(q_s.shape[0]),
        max_batch=int(pt_s.shape[0]),
        max_page_table_width=int(pt_s.shape[1]),
        max_work_items=128,
        max_partial_rows=2048,
        num_cache_pages=int(k_s.shape[0]),
    )
    workspace.prepare(pt_s, cs_s, cu_s)
    ptrs = (
        workspace.request_indices.data_ptr(),
        workspace.page_table.data_ptr(),
        workspace.lse.data_ptr(),
        workspace.tmp_output.data_ptr(),
        workspace.tmp_lse.data_ptr(),
    )

    workspace.prepare(pt_l, cs_l, cu_l)

    assert ptrs == (
        workspace.request_indices.data_ptr(),
        workspace.page_table.data_ptr(),
        workspace.lse.data_ptr(),
        workspace.tmp_output.data_ptr(),
        workspace.tmp_lse.data_ptr(),
    )
    assert workspace.plan.total_num_partial_rows <= 2048


def test_fixed_capacity_extend_workspace_rejects_overflow() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[64],
        cache_seqlens=[64 * 256],
        page_size=64,
        seed=95,
        page_table_width=256,
        num_pages=512,
    )
    workspace = PagedAttentionWorkspace.for_fixed_capacity(
        mode="extend",
        device=q.device,
        dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        num_q_heads=int(q.shape[1]),
        num_kv_heads=int(k_cache.shape[2]),
        head_dim_qk=int(q.shape[2]),
        head_dim_vo=int(v_cache.shape[3]),
        page_size=int(k_cache.shape[1]),
        max_total_q=int(q.shape[0]),
        max_batch=1,
        max_page_table_width=int(page_table.shape[1]),
        max_work_items=0,
        max_partial_rows=0,
        num_cache_pages=int(k_cache.shape[0]),
    )

    with pytest.raises(ValueError, match="paged prefill plan exceeds the configured eager workspace budget"):
        workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)


def test_for_eager_extend_capacity_uses_nosplit_budget() -> None:
    require_sm120()
    clear_attention_caches()

    workspace = PagedAttentionWorkspace.for_eager_extend_capacity(
        device="cuda",
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=64,
        max_total_q=128,
        max_batch=8,
        max_page_table_width=64,
        num_cache_pages=512,
    )

    assert workspace.fixed_capacity is True
    assert workspace.tmp_output is None
    assert workspace.tmp_lse is None
    assert workspace.request_indices is not None
    assert int(workspace.request_indices.shape[0]) == PagedAttentionWorkspace.eager_extend_work_items_capacity(
        max_total_q=128,
        num_q_heads=8,
        num_kv_heads=1,
    )
    assert workspace.planner_budget is not None
    assert workspace.planner_budget.max_partial_rows == 0


def test_prepare_for_capacity_primes_extend_graph_bucket() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, _page_table, _cache_seqlens, _cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[4, 4, 4, 4],
        cache_seqlens=[2048, 2048, 2048, 2048],
        page_size=64,
        seed=97,
        page_table_width=64,
        num_pages=512,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device=q.device),
        use_cuda_graph=True,
    )

    workspace.prepare_for_capacity(
        batch=4,
        total_q_capacity=16,
        max_page_table_width=64,
        max_cache_seqlen=4096,
    )

    assert workspace.plan.mode == "extend"
    assert workspace.plan.split_kv is False
    assert workspace.plan.total_q == 16
    assert workspace.page_table is not None
    assert tuple(workspace.page_table.shape) == (4, 64)
    assert workspace.cache_seqlens is not None
    assert int(workspace.cache_seqlens[0].item()) == 4096


def test_prepare_for_capacity_uses_host_total_without_tensor_item(monkeypatch) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, _page_table, _cache_seqlens, _cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[4, 4, 4, 4],
        cache_seqlens=[2048, 2048, 2048, 2048],
        page_size=64,
        seed=98,
        page_table_width=64,
        num_pages=512,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device=q.device),
        use_cuda_graph=True,
    )

    def fail_item(self):
        raise AssertionError("prepare_for_capacity should use the host total-q")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    workspace.prepare_for_capacity(
        batch=4,
        total_q_capacity=16,
        max_page_table_width=64,
        max_cache_seqlen=4096,
    )

    assert workspace.plan.total_q == 16


def test_prepare_uses_host_total_without_tensor_item(monkeypatch) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[4, 4, 4, 4],
        cache_seqlens=[2048, 2048, 2048, 2048],
        page_size=64,
        seed=99,
        page_table_width=64,
        num_pages=512,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=False,
    )

    def fail_item(self):
        raise AssertionError("prepare should use the host total-q")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        active_total_q=q.shape[0],
    )

    assert workspace.plan.total_q == q.shape[0]


def test_prepare_decode_graph_replay_state_is_batch_specific() -> None:
    require_sm120()
    clear_attention_caches()

    num_pages = 512
    page_size = 64
    head_dim = 256
    num_kv_heads = 1

    q1 = torch.randn(1, 8, head_dim, device="cuda", dtype=torch.bfloat16) / 4
    q8 = torch.randn(8, 8, head_dim, device="cuda", dtype=torch.bfloat16) / 4
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16) / 4
    v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16) / 4

    workspace_bs1 = _make_workspace(
        q=q1,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        use_cuda_graph=True,
    )
    workspace_bs8 = _make_workspace(
        q=q8,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=torch.arange(0, 9, dtype=torch.int32, device="cuda"),
        use_cuda_graph=True,
    )

    workspace_bs1.prepare_decode_graph_replay_state(
        batch=1,
        total_q_capacity=1,
        max_page_table_width=128,
        max_cache_page_count=128,
    )
    workspace_bs8.prepare_decode_graph_replay_state(
        batch=8,
        total_q_capacity=8,
        max_page_table_width=128,
        max_cache_page_count=128,
    )

    page_table_bs1 = torch.arange(0, 128, dtype=torch.int32, device="cuda").view(1, 128)
    page_table_bs8 = torch.stack(
        [
            torch.remainder(
                torch.arange(req_idx * 128, (req_idx + 1) * 128, dtype=torch.int32, device="cuda"),
                num_pages,
            )
            for req_idx in range(8)
        ],
        dim=0,
    )
    cache_seqlens_bs1 = torch.tensor([128 * page_size], dtype=torch.int32, device="cuda")
    cache_seqlens_bs8 = torch.full((8,), 128 * page_size, dtype=torch.int32, device="cuda")

    workspace_bs1.bind_cuda_graph_runtime_metadata(
        page_table=page_table_bs1.clone(),
        cache_seqlens=cache_seqlens_bs1.clone(),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
    )
    workspace_bs8.bind_cuda_graph_runtime_metadata(
        page_table=page_table_bs8.clone(),
        cache_seqlens=cache_seqlens_bs8.clone(),
        cu_seqlens_q=torch.arange(0, 9, dtype=torch.int32, device="cuda"),
    )

    workspace_bs1.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(page_table_bs1, row_stride=128 * page_size, page_size=page_size),
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device="cuda"),
    )
    workspace_bs8.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(page_table_bs8, row_stride=128 * page_size, page_size=page_size),
        req_pool_indices=torch.arange(0, 8, dtype=torch.int64, device="cuda"),
    )

    assert int(workspace_bs1.kv_chunk_size_ptr[0].item()) == 64
    assert int(workspace_bs8.kv_chunk_size_ptr[0].item()) == 384


def test_prepare_decode_graph_replay_state_uses_shape_heuristic_without_tuned_policy() -> None:
    require_sm120()
    clear_attention_caches()

    num_pages = 256
    page_size = 64
    head_dim = 128
    q = torch.randn(1, 16, head_dim, device="cuda", dtype=torch.bfloat16) / 4
    k_cache = torch.randn(num_pages, page_size, 8, head_dim, device="cuda", dtype=torch.bfloat16) / 4
    v_cache = torch.randn(num_pages, page_size, 8, head_dim, device="cuda", dtype=torch.bfloat16) / 4

    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        use_cuda_graph=True,
    )

    workspace.prepare_decode_graph_replay_state(
        batch=1,
        total_q_capacity=1,
        max_page_table_width=64,
        max_cache_page_count=64,
    )

    assert workspace._decode_graph_chunk_pages_lut is not None
    assert workspace._decode_graph_chunk_pages_lut.shape[0] == 65
    assert int(workspace._decode_graph_chunk_pages_lut[-1].item()) == 2
    assert workspace.plan.mode == "decode"
    assert workspace.plan.enable_cuda_graph is True


def test_update_decode_graph_replay_metadata_updates_workspace_buffers() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, _cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[4096, 2048, 1024, 512],
        page_size=64,
        seed=101,
        page_table_width=64,
        num_pages=512,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=torch.arange(0, 5, dtype=torch.int32, device=q.device),
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=4,
        total_q_capacity=4,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
    )

    bound_page_table = torch.empty_like(page_table)
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = torch.arange(0, 5, dtype=torch.int32, device=q.device)
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )

    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(page_table, row_stride=int(page_table.shape[1]) * 64, page_size=64),
        req_pool_indices=torch.arange(0, 4, dtype=torch.int64, device=q.device),
    )

    assert torch.equal(workspace.page_table, page_table)
    assert torch.equal(workspace.cache_seqlens, cache_seqlens)
    assert torch.equal(workspace.o_indptr[:5], workspace.merge_indptr[:5])
    assert int(workspace.kv_chunk_size_ptr[0].item()) > 0


def test_decode_graph_replay_metadata_rewrites_dynamic_request_indices() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, _cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[4096, 512],
        page_size=64,
        seed=103,
        page_table_width=64,
        num_pages=512,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=torch.arange(0, 3, dtype=torch.int32, device=q.device),
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=2,
        total_q_capacity=2,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
    )
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=torch.empty_like(page_table),
        cache_seqlens=cache_seqlens.clone(),
        cu_seqlens_q=torch.arange(0, 3, dtype=torch.int32, device=q.device),
    )

    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(
            page_table, row_stride=int(page_table.shape[1]) * 64, page_size=64
        ),
        req_pool_indices=torch.arange(0, 2, dtype=torch.int64, device=q.device),
    )

    max_chunks_per_req = int(workspace.request_indices.shape[0]) // 2
    assert torch.equal(
        workspace.request_indices[:max_chunks_per_req],
        torch.zeros(max_chunks_per_req, dtype=torch.int32, device=q.device),
    )
    assert torch.equal(
        workspace.request_indices[max_chunks_per_req : 2 * max_chunks_per_req],
        torch.ones(max_chunks_per_req, dtype=torch.int32, device=q.device),
    )


@pytest.mark.parametrize(
    ("cache_seqlens_list", "expected_window_starts"),
    [
        ([193], [64]),
        ([64, 129, 192, 193, 257], [0, 0, 0, 64, 128]),
    ],
)
def test_decode_graph_replay_metadata_keeps_swa_page_boundary_tokens(
    cache_seqlens_list: list[int],
    expected_window_starts: list[int],
) -> None:
    require_sm120()
    clear_attention_caches()

    batch = len(cache_seqlens_list)
    q, k_cache, v_cache, page_table, cache_seqlens, _cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=cache_seqlens_list,
        page_size=64,
        seed=107 + batch,
        page_table_width=max(3, max((length + 63) // 64 for length in cache_seqlens_list)),
        num_pages=32,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=torch.arange(0, batch + 1, dtype=torch.int32, device=q.device),
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=batch,
        total_q_capacity=batch,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
        window_left=128,
    )
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=torch.empty_like(page_table),
        cache_seqlens=cache_seqlens.clone(),
        cu_seqlens_q=torch.arange(0, batch + 1, dtype=torch.int32, device=q.device),
    )

    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(page_table, row_stride=int(page_table.shape[1]) * 64, page_size=64),
        req_pool_indices=torch.arange(0, batch, dtype=torch.int64, device=q.device),
    )

    assert workspace.kv_window_start_tokens[:batch].detach().cpu().tolist() == expected_window_starts
    assert torch.all(workspace.merge_indptr[1 : batch + 1] > workspace.merge_indptr[:batch])


def test_regular_decode_graph_replay_runs_end_to_end_at_bs8() -> None:
    import gc

    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[512, 768, 1024, 1536, 2048, 2560, 3072, 3584],
        page_size=64,
        seed=103,
        page_table_width=64,
        num_pages=1024,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=8,
        total_q_capacity=8,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
    )

    bound_page_table = torch.empty_like(page_table)
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(page_table, row_stride=int(page_table.shape[1]) * 64, page_size=64),
        req_pool_indices=torch.arange(0, 8, dtype=torch.int64, device=q.device),
    )

    output = torch.empty_like(q)
    out, lse = workspace.run(q, k_cache, v_cache, output=output)
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )

    assert workspace._use_regular_decode_graph_replay is True
    assert torch.allclose(out.to(torch.float32), ref_out.to(torch.float32), atol=2e-2, rtol=2e-2)
    assert torch.allclose(_lse_base2_to_natural(lse), ref_lse, atol=3e-2, rtol=3e-2)

    del workspace, output, out, lse
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()


def test_regular_decode_graph_replay_runs_end_to_end_at_bs4() -> None:
    import gc

    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * 4,
        cache_seqlens=[512, 768, 1024, 1536],
        page_size=64,
        seed=107,
        page_table_width=64,
        num_pages=1024,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=4,
        total_q_capacity=4,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
    )

    bound_page_table = torch.empty_like(page_table)
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(page_table, row_stride=int(page_table.shape[1]) * 64, page_size=64),
        req_pool_indices=torch.arange(0, 4, dtype=torch.int64, device=q.device),
    )

    output = torch.empty_like(q)
    out, lse = workspace.run(q, k_cache, v_cache, output=output)
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )

    assert workspace._use_regular_decode_graph_replay is True
    assert torch.allclose(out.to(torch.float32), ref_out.to(torch.float32), atol=2e-2, rtol=2e-2)
    assert torch.allclose(_lse_base2_to_natural(lse), ref_lse, atol=3e-2, rtol=3e-2)

    del workspace, output, out, lse
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()


def test_mimo_v25_decode_cuda_graph_matches_reference_with_swa_and_sink() -> None:
    import gc

    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * 4,
        cache_seqlens=[129, 193, 257, 385],
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=227,
        page_table_width=8,
        num_pages=128,
    )
    attention_sink_bias = torch.linspace(
        -0.25,
        0.35,
        steps=q.shape[1],
        device=q.device,
        dtype=torch.float32,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=128)

    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)
    workspace.run(
        q,
        k_cache,
        v_cache,
        output=output,
        attention_sink_bias=attention_sink_bias,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=output,
            attention_sink_bias=attention_sink_bias,
        )
    graph.replay()
    torch.cuda.synchronize()
    lse = workspace.current_lse_view()
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=128,
        attention_sink_bias=attention_sink_bias,
    )

    assert torch.allclose(output.to(torch.float32), ref_out.to(torch.float32), atol=2e-2, rtol=2e-2)
    assert torch.allclose(_lse_base2_to_natural(lse), ref_lse, atol=3e-2, rtol=3e-2)

    del workspace, output, lse
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()


def test_mimo_v25_decode_cuda_graph_matches_reference_with_swa_and_sink_fp8_kv() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * 4,
        cache_seqlens=[129, 193, 257, 385],
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=427,
        page_table_width=8,
        num_pages=128,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    attention_sink_bias = torch.linspace(
        -0.25,
        0.35,
        steps=q.shape[1],
        device=q.device,
        dtype=torch.float32,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=128)

    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)
    workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
        attention_sink_bias=attention_sink_bias,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_fp8,
            v_fp8,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=attention_sink_bias,
        )
    graph.replay()
    torch.cuda.synchronize()
    lse = workspace.current_lse_view()
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
        window_left=128,
        attention_sink_bias=attention_sink_bias,
    )

    assert torch.allclose(output.to(torch.float32), ref_out.to(torch.float32), atol=6e-2, rtol=6e-2)
    assert torch.allclose(_lse_base2_to_natural(lse), ref_lse, atol=6e-2, rtol=6e-2)
    assert _cosine_similarity(output, ref_out) >= 0.9995

    del workspace, output, lse
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    ("cache_seqlens_a", "cache_seqlens_b"),
    [
        ([129], [193]),
        ([129, 193, 257, 385], [193, 257, 385, 449]),
    ],
)
def test_mimo_v25_decode_cuda_graph_replays_with_updated_metadata(
    cache_seqlens_a: list[int],
    cache_seqlens_b: list[int],
) -> None:
    import gc

    require_sm120()
    clear_attention_caches()

    batch = len(cache_seqlens_a)
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=cache_seqlens_a,
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=229,
        page_table_width=8,
        num_pages=128,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=batch,
        total_q_capacity=batch,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
        window_left=128,
    )
    bound_page_table = torch.empty_like(page_table)
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    req_pool_indices = torch.arange(0, batch, dtype=torch.int64, device=q.device)
    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(
            page_table,
            row_stride=int(page_table.shape[1]) * 64,
            page_size=64,
        ),
        req_pool_indices=req_pool_indices,
    )
    attention_sink_bias = torch.linspace(
        -0.25,
        0.35,
        steps=q.shape[1],
        device=q.device,
        dtype=torch.float32,
    )
    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)

    workspace.run(
        q,
        k_cache,
        v_cache,
        output=output,
        attention_sink_bias=attention_sink_bias,
        prepare_decode_graph_metadata=False,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=output,
            attention_sink_bias=attention_sink_bias,
            prepare_decode_graph_metadata=False,
        )

    graph.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=128,
        attention_sink_bias=attention_sink_bias,
    )
    assert torch.allclose(output.to(torch.float32), ref_out.to(torch.float32), atol=2e-2, rtol=2e-2)
    assert torch.allclose(
        _lse_base2_to_natural(workspace.current_lse_view()),
        ref_lse,
        atol=3e-2,
        rtol=3e-2,
    )

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = _make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=cache_seqlens_b,
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=233,
        page_table_width=int(page_table.shape[1]),
        num_pages=int(k_cache.shape[0]),
    )
    q.copy_(q_2)
    k_cache.copy_(k_cache_2)
    v_cache.copy_(v_cache_2)
    bound_cache_seqlens.copy_(cache_seqlens_2)
    bound_cu_seqlens_q.copy_(cu_seqlens_q_2)
    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(
            page_table_2,
            row_stride=int(page_table_2.shape[1]) * 64,
            page_size=64,
        ),
        req_pool_indices=req_pool_indices,
    )

    graph.replay()
    torch.cuda.synchronize()
    ref_out_2, ref_lse_2 = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table_2,
        cache_seqlens_2,
        cu_seqlens_q_2,
        causal=True,
        window_left=128,
        attention_sink_bias=attention_sink_bias,
    )
    assert torch.allclose(output.to(torch.float32), ref_out_2.to(torch.float32), atol=2e-2, rtol=2e-2)
    assert torch.allclose(
        _lse_base2_to_natural(workspace.current_lse_view()),
        ref_lse_2,
        atol=3e-2,
        rtol=3e-2,
    )

    del workspace, output
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()


def test_mimo_v25_decode_cuda_graph_replays_with_updated_metadata_fp8_kv() -> None:
    require_sm120()
    clear_attention_caches()

    batch = 4
    cache_seqlens_a = [129, 193, 257, 385]
    cache_seqlens_b = [193, 257, 385, 449]
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=cache_seqlens_a,
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=429,
        page_table_width=8,
        num_pages=128,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=batch,
        total_q_capacity=batch,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
        window_left=128,
    )
    bound_page_table = torch.empty_like(page_table)
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    req_pool_indices = torch.arange(0, batch, dtype=torch.int64, device=q.device)
    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(
            page_table,
            row_stride=int(page_table.shape[1]) * 64,
            page_size=64,
        ),
        req_pool_indices=req_pool_indices,
    )
    attention_sink_bias = torch.linspace(
        -0.25,
        0.35,
        steps=q.shape[1],
        device=q.device,
        dtype=torch.float32,
    )
    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)

    workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
        attention_sink_bias=attention_sink_bias,
        prepare_decode_graph_metadata=False,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_fp8,
            v_fp8,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=attention_sink_bias,
            prepare_decode_graph_metadata=False,
        )

    graph.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
        window_left=128,
        attention_sink_bias=attention_sink_bias,
    )
    assert torch.allclose(output.to(torch.float32), ref_out.to(torch.float32), atol=6e-2, rtol=6e-2)
    assert torch.allclose(
        _lse_base2_to_natural(workspace.current_lse_view()),
        ref_lse,
        atol=6e-2,
        rtol=6e-2,
    )

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = _make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=cache_seqlens_b,
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=433,
        page_table_width=int(page_table.shape[1]),
        num_pages=int(k_cache.shape[0]),
    )
    k_fp8_2, v_fp8_2, k_descale_2, v_descale_2 = _quantize_paged_kv_cache_e4m3(
        k_cache_2,
        v_cache_2,
        page_table_2,
        cache_seqlens_2,
    )
    q.copy_(q_2)
    k_fp8.copy_(k_fp8_2)
    v_fp8.copy_(v_fp8_2)
    k_descale.copy_(k_descale_2)
    v_descale.copy_(v_descale_2)
    bound_cache_seqlens.copy_(cache_seqlens_2)
    bound_cu_seqlens_q.copy_(cu_seqlens_q_2)
    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(
            page_table_2,
            row_stride=int(page_table_2.shape[1]) * 64,
            page_size=64,
        ),
        req_pool_indices=req_pool_indices,
    )

    graph.replay()
    torch.cuda.synchronize()
    ref_out_2, ref_lse_2 = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table_2,
        cache_seqlens_2,
        cu_seqlens_q_2,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
        window_left=128,
        attention_sink_bias=attention_sink_bias,
    )
    assert torch.allclose(output.to(torch.float32), ref_out_2.to(torch.float32), atol=6e-2, rtol=6e-2)
    assert torch.allclose(
        _lse_base2_to_natural(workspace.current_lse_view()),
        ref_lse_2,
        atol=6e-2,
        rtol=6e-2,
    )
    assert _cosine_similarity(output, ref_out_2) >= 0.9995

    del workspace, output
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()


def test_mimo_v25_full_decode_cuda_graph_replays_with_updated_metadata() -> None:
    import gc

    require_sm120()
    clear_attention_caches()

    batch = 4
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=[129, 193, 257, 385],
        page_size=64,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=239,
        page_table_width=8,
        num_pages=128,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=batch,
        total_q_capacity=batch,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
        window_left=-1,
    )
    bound_page_table = torch.empty_like(page_table)
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    req_pool_indices = torch.arange(0, batch, dtype=torch.int64, device=q.device)
    workspace.update_decode_graph_replay_metadata(
        req_to_token=_make_req_to_token(
            page_table,
            row_stride=int(page_table.shape[1]) * 64,
            page_size=64,
        ),
        req_pool_indices=req_pool_indices,
    )
    output = torch.empty((*q.shape[:-1], v_cache.shape[-1]), device=q.device, dtype=q.dtype)

    workspace.run(
        q,
        k_cache,
        v_cache,
        output=output,
        prepare_decode_graph_metadata=False,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=output,
            prepare_decode_graph_metadata=False,
        )

    for seed, lengths in ((241, [193, 257, 385, 449]), (251, [194, 258, 386, 450])):
        q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = (
            _make_paged_inputs(
                q_seqlens=[1] * batch,
                cache_seqlens=lengths,
                page_size=64,
                q_heads=16,
                kv_heads=1,
                head_dim_qk=192,
                head_dim_vo=128,
                seed=seed,
                page_table_width=int(page_table.shape[1]),
                num_pages=int(k_cache.shape[0]),
            )
        )
        q.copy_(q_2)
        k_cache.copy_(k_cache_2)
        v_cache.copy_(v_cache_2)
        bound_cache_seqlens.copy_(cache_seqlens_2)
        bound_cu_seqlens_q.copy_(cu_seqlens_q_2)
        workspace.update_decode_graph_replay_metadata(
            req_to_token=_make_req_to_token(
                page_table_2,
                row_stride=int(page_table_2.shape[1]) * 64,
                page_size=64,
            ),
            req_pool_indices=req_pool_indices,
        )

        graph.replay()
        torch.cuda.synchronize()
        ref_out, ref_lse = paged_attention_reference(
            q,
            k_cache,
            v_cache,
            page_table_2,
            cache_seqlens_2,
            cu_seqlens_q_2,
            causal=True,
        )
        assert torch.allclose(
            output.to(torch.float32),
            ref_out.to(torch.float32),
            atol=2e-2,
            rtol=2e-2,
        )
        assert torch.allclose(
            _lse_base2_to_natural(workspace.current_lse_view()),
            ref_lse,
            atol=3e-2,
            rtol=3e-2,
        )

    del workspace, output
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()


def test_decode_replay_metadata_can_be_captured_into_cuda_graph() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * 4,
        cache_seqlens=[512, 768, 1024, 1536],
        page_size=64,
        seed=109,
        page_table_width=64,
        num_pages=1024,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=4,
        total_q_capacity=4,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
    )

    bound_page_table = torch.empty_like(page_table)
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    output = torch.empty_like(q)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=output,
            prepare_decode_graph_metadata=True,
        )

    assert workspace._decode_graph_metadata_captured_in_graph is True

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = _make_paged_inputs(
        q_seqlens=[1] * 4,
        cache_seqlens=[2048, 2048, 4096, 4096],
        page_size=64,
        seed=113,
        page_table_width=int(page_table.shape[1]),
        num_pages=int(k_cache.shape[0]),
    )
    q.copy_(q_2)
    k_cache.copy_(k_cache_2)
    v_cache.copy_(v_cache_2)

    original_update = workspace.update_decode_graph_replay_metadata_from_runtime_cache_seqlens

    def _fail_if_called() -> None:
        raise AssertionError("decode replay metadata should come from the captured graph")

    workspace.update_decode_graph_replay_metadata_from_runtime_cache_seqlens = _fail_if_called
    workspace.prepare(page_table_2, cache_seqlens_2, cu_seqlens_q_2)

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table_2,
        cache_seqlens_2,
        cu_seqlens_q_2,
        causal=True,
    )

    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(output.to(torch.float32), ref_out.to(torch.float32), atol=2e-2, rtol=2e-2)
    assert torch.allclose(
        _lse_base2_to_natural(workspace.current_lse_view()),
        ref_lse,
        atol=3e-2,
        rtol=3e-2,
    )

    workspace.update_decode_graph_replay_metadata_from_runtime_cache_seqlens = original_update



@pytest.mark.parametrize(
    ("window_left", "use_sink"),
    [(-1, False), (128, True)],
)
def test_mimo_v25_decode_cuda_graph_page_boundary_matches_reference(
    window_left: int,
    use_sink: bool,
) -> None:
    require_sm120()
    clear_attention_caches()

    graph_bs = 8
    page_size = 64
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * graph_bs,
        cache_seqlens=[65] + [1] * (graph_bs - 1),
        page_size=page_size,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=263,
        page_table_width=4,
        num_pages=64,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=graph_bs,
        total_q_capacity=graph_bs,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
        window_left=window_left,
    )

    bound_page_table = page_table.clone()
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    output = torch.empty(
        (graph_bs, q.shape[1], v_cache.shape[-1]),
        dtype=q.dtype,
        device=q.device,
    )
    attention_sink_bias = None
    if use_sink:
        attention_sink_bias = torch.linspace(
            -0.2,
            0.2,
            q.shape[1],
            dtype=torch.float32,
            device=q.device,
        )

    workspace.run(
        q,
        k_cache,
        v_cache,
        output=output,
        attention_sink_bias=attention_sink_bias,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=output,
            attention_sink_bias=attention_sink_bias,
        )

    row0_cu = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
    for step, cache_len in enumerate((63, 64, 65, 66, 127, 128, 129)):
        torch.manual_seed(5000 + step)
        q.copy_(torch.randn_like(q) / 4)
        bound_cache_seqlens.fill_(1)
        bound_cache_seqlens[0] = cache_len
        bound_cu_seqlens_q.copy_(
            torch.arange(0, graph_bs + 1, dtype=torch.int32, device=q.device)
        )
        bound_page_table.copy_(page_table)

        graph.replay()
        torch.cuda.synchronize()

        row0_cache_seqlens = bound_cache_seqlens[:1]
        ref_out, ref_lse = paged_attention_reference(
            q[:1],
            k_cache,
            v_cache,
            bound_page_table[:1],
            row0_cache_seqlens,
            row0_cu,
            causal=True,
            window_left=window_left,
            attention_sink_bias=attention_sink_bias,
        )
        got = output[:1].to(torch.float32)
        ref = ref_out.to(torch.float32)
        assert torch.allclose(got, ref, atol=2e-2, rtol=2e-2), cache_len
        assert _cosine_similarity(got, ref) >= 0.99999, cache_len
        assert torch.allclose(
            _lse_base2_to_natural(workspace.current_lse_view()[:1]),
            ref_lse,
            atol=3e-2,
            rtol=3e-2,
        ), cache_len

    del workspace, output
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    ("window_left", "use_sink"),
    [(-1, False), (128, True)],
)
def test_mimo_v25_decode_cuda_graph_page_boundary_matches_reference_fp8_kv(
    window_left: int,
    use_sink: bool,
) -> None:
    require_sm120()
    clear_attention_caches()

    graph_bs = 8
    page_size = 64
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1] * graph_bs,
        cache_seqlens=[65] + [1] * (graph_bs - 1),
        page_size=page_size,
        q_heads=16,
        kv_heads=1,
        head_dim_qk=192,
        head_dim_vo=128,
        seed=463,
        page_table_width=4,
        num_pages=64,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare_decode_graph_replay_state(
        batch=graph_bs,
        total_q_capacity=graph_bs,
        max_page_table_width=int(page_table.shape[1]),
        max_cache_page_count=int(page_table.shape[1]),
        window_left=window_left,
    )

    bound_page_table = page_table.clone()
    bound_cache_seqlens = cache_seqlens.clone()
    bound_cu_seqlens_q = cu_seqlens_q.clone()
    workspace.bind_cuda_graph_runtime_metadata(
        page_table=bound_page_table,
        cache_seqlens=bound_cache_seqlens,
        cu_seqlens_q=bound_cu_seqlens_q,
    )
    output = torch.empty(
        (graph_bs, q.shape[1], v_cache.shape[-1]),
        dtype=q.dtype,
        device=q.device,
    )
    attention_sink_bias = None
    if use_sink:
        attention_sink_bias = torch.linspace(
            -0.2,
            0.2,
            q.shape[1],
            dtype=torch.float32,
            device=q.device,
        )

    workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
        attention_sink_bias=attention_sink_bias,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_fp8,
            v_fp8,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=attention_sink_bias,
        )

    row0_cu = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
    for step, cache_len in enumerate((63, 64, 65, 66, 127, 128, 129)):
        torch.manual_seed(6000 + step)
        q.copy_(torch.randn_like(q) / 4)
        bound_cache_seqlens.fill_(1)
        bound_cache_seqlens[0] = cache_len
        bound_cu_seqlens_q.copy_(
            torch.arange(0, graph_bs + 1, dtype=torch.int32, device=q.device)
        )
        bound_page_table.copy_(page_table)

        graph.replay()
        torch.cuda.synchronize()

        row0_cache_seqlens = bound_cache_seqlens[:1]
        ref_out, ref_lse = paged_attention_reference(
            q[:1],
            k_fp8,
            v_fp8,
            bound_page_table[:1],
            row0_cache_seqlens,
            row0_cu,
            k_descale=k_descale[:1],
            v_descale=v_descale[:1],
            causal=True,
            window_left=window_left,
            attention_sink_bias=attention_sink_bias,
        )
        got = output[:1].to(torch.float32)
        ref = ref_out.to(torch.float32)
        assert torch.allclose(got, ref, atol=6e-2, rtol=6e-2), cache_len
        assert _cosine_similarity(got, ref) >= 0.9995, cache_len
        assert torch.allclose(
            _lse_base2_to_natural(workspace.current_lse_view()[:1]),
            ref_lse,
            atol=6e-2,
            rtol=6e-2,
        ), cache_len

    del workspace, output
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()


def test_workspace_mode_validation_rejects_mismatched_prepare() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=47,
    )
    workspace = PagedAttentionWorkspace.for_tensors(
        mode="decode",
        q=q[:4],
        k_cache=k_cache,
        v_cache=v_cache,
    )
    with pytest.raises(ValueError, match="workspace mode decode does not match prepared mode extend"):
        workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)


def test_extend_workspace_accepts_single_token_extend_prepare() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, _ = _make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[1],
        page_size=64,
        seed=49,
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
    workspace = PagedAttentionWorkspace.for_tensors(
        mode="extend",
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
    )

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    assert workspace.plan.mode == "extend"
    assert workspace.active_total_q == 1


def test_workspace_fixed_split_size_pins_chunk_pages() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 4096],
        page_size=64,
        seed=51,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
    )

    assert workspace.plan.fixed_split_size == 8
    assert workspace.plan.kv_chunk_size == 8 * 64


def test_graph_workspace_reuses_stable_metadata_buffers_for_smaller_replay() -> None:
    require_sm120()
    clear_attention_caches()

    capture_inputs = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[2048, 2048, 4096, 4096],
        page_size=64,
        seed=59,
        page_table_width=64,
        num_pages=512,
    )
    replay_inputs = _make_paged_inputs(
        q_seqlens=[4, 4, 4, 4],
        cache_seqlens=[1024, 1536, 2048, 2048],
        page_size=64,
        seed=61,
        page_table_width=64,
        num_pages=512,
    )
    q0, k0, v0, pt0, cs0, cu0 = capture_inputs
    q1, k1, v1, pt1, cs1, cu1 = replay_inputs

    workspace = _make_workspace(
        q=q0,
        k_cache=k0,
        v_cache=v0,
        cu_seqlens_q=cu0,
        use_cuda_graph=True,
    )
    workspace.prepare(pt0, cs0, cu0)
    ptrs = (
        workspace.request_indices.data_ptr(),
        workspace.page_table.data_ptr(),
        workspace.cache_seqlens.data_ptr(),
        workspace.cu_seqlens_q.data_ptr(),
    )
    workspace.prepare(pt1, cs1, cu1)

    assert ptrs == (
        workspace.request_indices.data_ptr(),
        workspace.page_table.data_ptr(),
        workspace.cache_seqlens.data_ptr(),
        workspace.cu_seqlens_q.data_ptr(),
    )
    assert workspace.active_total_q == q1.shape[0]


def test_graph_workspace_uses_nosplit_fp8_extend_chunks_at_8192() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[8192] * 8,
        page_size=64,
        seed=73,
        page_table_width=128,
        num_pages=2048,
    )
    k_fp8, v_fp8, _, _ = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    assert workspace.plan.mode == "extend"
    assert workspace.plan.kv_dtype == torch.float8_e4m3fn
    assert workspace.plan.kv_chunk_size == 128 * 64
    assert workspace.plan.split_kv is False


def test_graph_workspace_rejects_capacity_growth() -> None:
    require_sm120()
    clear_attention_caches()

    small = _make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[64, 64],
        page_size=64,
        seed=67,
    )
    large = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[8192, 8192, 8192, 8192],
        page_size=64,
        seed=71,
        page_table_width=128,
        num_pages=1024,
    )
    q_s, k_s, v_s, pt_s, cs_s, cu_s = small
    q_l, k_l, v_l, pt_l, cs_l, cu_l = large
    workspace = _make_workspace(
        q=q_s,
        k_cache=k_s,
        v_cache=v_s,
        cu_seqlens_q=cu_s,
        use_cuda_graph=True,
    )
    workspace.prepare(pt_s, cs_s, cu_s)

    with pytest.raises(ValueError, match="graph-mode paged workspace capacity exceeded"):
        workspace.prepare(pt_l, cs_l, cu_l)


def _count_generate_original_ir_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], object]:
    calls = [0]
    original_generate_original_ir = cutlass_dsl.BaseDSL.generate_original_ir

    def counted_generate_original_ir(self, *args, **kwargs):
        calls[0] += 1
        return original_generate_original_ir(self, *args, **kwargs)

    monkeypatch.setattr(cutlass_dsl.BaseDSL, "generate_original_ir", counted_generate_original_ir)
    return calls, original_generate_original_ir


def test_eager_workspace_reuses_compiled_spec_for_identical_nosplit_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
    monkeypatch.delenv("B12X_CUTE_COMPILE_MEMORY_CACHE", raising=False)
    clear_compile_cache()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[4],
        cache_seqlens=[4096],
        page_size=64,
        q_heads=48,
        kv_heads=8,
        head_dim=128,
        seed=79,
        page_table_width=64,
        num_pages=512,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    first_compile_misses = compile_cache_info()["compile_misses"]
    assert first_compile_misses >= 1

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))

    assert compile_cache_info()["compile_misses"] == first_compile_misses


def test_eager_workspace_reuses_compiled_spec_for_identical_extend_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
    monkeypatch.delenv("B12X_CUTE_COMPILE_MEMORY_CACHE", raising=False)
    clear_compile_cache()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=83,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    calls, _ = _count_generate_original_ir_calls(monkeypatch)

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.split_kv is False
    workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    first_run_calls = calls[0]

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))

    assert first_run_calls > 0
    assert calls[0] == first_run_calls
