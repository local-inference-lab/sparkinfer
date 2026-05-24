from __future__ import annotations

import math

import pytest
import torch

from benchmarks.benchmark_paged_attention import (
    _capture_backend_graph,
    _capture_flashinfer_fa2_graph,
    _make_uniform_paged_inputs,
    _quantize_paged_kv_cache_global_e4m3,
)
from b12x.attention.paged.reference import paged_attention_reference
from b12x.integration.attention import PagedAttentionWorkspace

from .helpers import require_sm120
from .test_attention_paged_planner import _make_inputs
from .test_paged_attention_workspace_api import _quantize_paged_kv_cache_e4m3


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _make_workspace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    mode: str,
) -> PagedAttentionWorkspace:
    return PagedAttentionWorkspace.for_tensors(
        mode=mode,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
    )


def _run_decode_graph_check(
    *,
    batch: int = 8,
    cache_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        capture_page_table,
        capture_cache_seqlens,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=batch,
        q_seqlen=1,
        cache_seqlen=cache_seqlen,
        capture_cache_seqlen=None,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        seed=1,
    )
    k_fp8, v_fp8, k_descale, v_descale, k_scale, v_scale = _quantize_paged_kv_cache_global_e4m3(
        k_cache,
        v_cache,
        batch=batch,
        kv_heads=1,
    )
    backend = _capture_backend_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=None,
        k_descale=k_descale,
        v_descale=v_descale,
        warmup=1,
        graph_ctas_per_sm=None,
    )
    _fa2_graph, fa2_out = _capture_flashinfer_fa2_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        q_seqlen=1,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        k_scale=k_scale,
        v_scale=v_scale,
        workspace_bytes=512 * 1024 * 1024,
        warmup=1,
    )
    return backend.output, fa2_out, backend.plan_desc


def _run_decode_reference_check(
    *,
    batch: int = 8,
    cache_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        capture_page_table,
        capture_cache_seqlens,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=batch,
        q_seqlen=1,
        cache_seqlen=cache_seqlen,
        capture_cache_seqlen=None,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        seed=1,
    )
    k_fp8, v_fp8, k_descale, v_descale, _k_scale, _v_scale = _quantize_paged_kv_cache_global_e4m3(
        k_cache,
        v_cache,
        batch=batch,
        kv_heads=1,
    )
    backend = _capture_backend_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=None,
        k_descale=k_descale,
        v_descale=v_descale,
        warmup=1,
        graph_ctas_per_sm=None,
    )
    backend.graph.replay()
    torch.cuda.synchronize()
    ref_out, _ref_lse = paged_attention_reference(
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
    return backend.output, ref_out, backend.plan_desc


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_short_context() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[64, 128, 192],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
    )
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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_fp8_decode_short_context_batch8() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1, 1, 1, 1, 1, 1],
        cache_seqlens=[64, 64, 64, 64, 64, 64, 64, 64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(q, k_fp8, v_fp8, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=torch.empty_like(q),
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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_with_sliding_window_and_sink() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[128, 192, 256],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 80
    attention_sink_bias = torch.linspace(-0.2, 0.2, q.shape[1], dtype=torch.float32, device=q.device)
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_mimo_gqa_shape_with_sliding_window_and_sink() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[128, 192, 256],
        q_heads=64,
        kv_heads=8,
        head_dim_qk=192,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 80
    attention_sink_bias = torch.linspace(-0.2, 0.2, q.shape[1], dtype=torch.float32, device=q.device)
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty(q.shape[0], q.shape[1], v_cache.shape[3], dtype=q.dtype, device=q.device),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.02
    assert (lse_natural - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(output, ref_out) >= 0.9999


@torch.inference_mode()
def test_paged_forward_attention_sink_affects_denominator_only() -> None:
    require_sm120()
    q_heads = 8
    kv_heads = 1
    head_dim = 256
    page_size = 64
    q = torch.zeros((1, q_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((1, page_size, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros((1, page_size, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v_cache[:, 0, :, :].fill_(1.0)
    page_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    cache_seqlens = torch.ones((1,), dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    attention_sink_bias = torch.full((q_heads,), math.log(3.0), dtype=torch.float32, device="cuda")

    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    expected_output = torch.full_like(output, 0.25)
    expected_lse = torch.full((1, q_heads), math.log(4.0), dtype=torch.float32, device="cuda")
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - expected_output).abs().max().item() <= 0.002
    assert (lse_natural - expected_lse).abs().max().item() <= 0.002


@torch.inference_mode()
def test_paged_forward_native_fp8_qkv_matches_reference_fp8_decode_short_context_batch8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    monkeypatch.setenv("B12X_TURBO_ATTN", "1")
    output, ref_out, plan_desc = _run_decode_reference_check(cache_seqlen=64)
    assert plan_desc.endswith(",split")
    assert (output - ref_out).abs().max().item() <= 0.02
    assert _cosine_similarity(output, ref_out) >= 0.995


@torch.inference_mode()
def test_paged_forward_matches_reference_without_split_bf16_extend() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[64, 64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_extend_with_sliding_window_and_sink() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[320, 384],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 96
    attention_sink_bias = torch.linspace(0.1, -0.1, q.shape[1], dtype=torch.float32, device=q.device)
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
        window_left=window_left,
    )
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_with_fp8_kv_extend() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[2048, 4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(q, k_fp8, v_fp8, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.split_kv is False
    output, lse_base2 = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=torch.empty_like(q),
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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_matches_reference_with_split_fp8_decode() -> None:
    require_sm120()
    output, fa2_out, plan_desc = _run_decode_graph_check(cache_seqlen=512)
    assert plan_desc.endswith(",split")
    assert (output - fa2_out).abs().max().item() <= 0.01
    assert _cosine_similarity(output, fa2_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_native_fp8_qkv_matches_reference_with_split_fp8_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    monkeypatch.setenv("B12X_TURBO_ATTN", "1")
    output, ref_out, plan_desc = _run_decode_reference_check(cache_seqlen=8192)
    assert plan_desc.endswith(",split")
    assert (output - ref_out).abs().max().item() <= 0.01
    assert _cosine_similarity(output, ref_out) >= 0.995


@torch.inference_mode()
def test_paged_forward_matches_reference_with_bf16_kv_extend() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[2048, 4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.split_kv is False
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999
