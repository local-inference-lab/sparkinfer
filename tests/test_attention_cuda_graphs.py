from __future__ import annotations

import math

import torch

from b12x.attention.paged.reference import paged_attention_reference
from b12x.integration.attention import PagedAttentionWorkspace, clear_attention_caches

from .helpers import require_sm120
from .test_paged_attention_workspace_api import _make_paged_inputs, _quantize_paged_kv_cache_e4m3


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _lse_base2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * math.log(2.0)


@torch.inference_mode()
def test_paged_attention_decode_replays_under_cuda_graph_with_variable_metadata() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 128, 192, 256],
        page_size=64,
        seed=73,
        page_table_width=64,
        num_pages=512,
    )
    _, _, _, page_table_max, cache_seqlens_max, cu_seqlens_q_max = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[4096, 4096, 4096, 4096],
        page_size=64,
        seed=74,
        page_table_width=page_table.shape[1],
        num_pages=k_cache.shape[0],
    )
    workspace = PagedAttentionWorkspace.for_tensors(
        mode="decode",
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=True,
    )
    workspace.prepare(page_table_max, cache_seqlens_max, cu_seqlens_q_max)
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(q, k_cache, v_cache, output=output)

    ref_out_1, ref_lse_1 = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out_1).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_1).abs().max().item() <= 0.03
    assert _cosine_similarity(output, ref_out_1) >= 0.99999

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[2048, 2048, 4096, 4096],
        page_size=64,
        seed=79,
        page_table_width=page_table.shape[1],
        num_pages=k_cache.shape[0],
    )
    q.copy_(q_2)
    k_cache.copy_(k_cache_2)
    v_cache.copy_(v_cache_2)
    workspace.prepare(page_table_2, cache_seqlens_2, cu_seqlens_q_2)

    ref_out_2, ref_lse_2 = paged_attention_reference(
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
    assert (output - ref_out_2).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_2).abs().max().item() <= 0.03
    assert _cosine_similarity(output, ref_out_2) >= 0.99999


@torch.inference_mode()
def test_paged_attention_extend_replays_under_cuda_graph_with_smaller_metadata() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=83,
        page_table_width=4,
    )
    workspace = PagedAttentionWorkspace.for_tensors(
        mode="extend",
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=True,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(q, k_cache, v_cache, output=output)

    ref_out_1, ref_lse_1 = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output[: q.shape[0]] - ref_out_1).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_1).abs().max().item() <= 0.03
    assert _cosine_similarity(output[: q.shape[0]], ref_out_1) >= 0.99999

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = _make_paged_inputs(
        q_seqlens=[4, 4, 4, 4],
        cache_seqlens=[64, 97, 81, 113],
        page_size=64,
        seed=89,
        page_table_width=page_table.shape[1],
        num_pages=k_cache.shape[0],
    )
    q.zero_()
    q[: q_2.shape[0]].copy_(q_2)
    k_cache.copy_(k_cache_2)
    v_cache.copy_(v_cache_2)
    workspace.prepare(page_table_2, cache_seqlens_2, cu_seqlens_q_2)

    ref_out_2, ref_lse_2 = paged_attention_reference(
        q[: q_2.shape[0]],
        k_cache,
        v_cache,
        page_table_2,
        cache_seqlens_2,
        cu_seqlens_q_2,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output[: q_2.shape[0]] - ref_out_2).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_2).abs().max().item() <= 0.03
    assert _cosine_similarity(output[: q_2.shape[0]], ref_out_2) >= 0.99999


@torch.inference_mode()
def test_paged_attention_fp8_kv_replays_under_cuda_graph() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=97,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = PagedAttentionWorkspace.for_tensors(
        mode="extend",
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        use_cuda_graph=True,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_fp8,
            v_fp8,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    ref_out_1, ref_lse_1 = paged_attention_reference(
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
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out_1).abs().max().item() <= 0.05
    assert (_lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_1).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out_1) >= 0.9999

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=101,
        page_table_width=page_table.shape[1],
        num_pages=k_cache.shape[0],
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
    workspace.prepare(page_table_2, cache_seqlens_2, cu_seqlens_q_2)

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
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out_2).abs().max().item() <= 0.05
    assert (_lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_2).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out_2) >= 0.9999
