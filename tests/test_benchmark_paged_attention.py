from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_paged_attention import (
    _build_decode_replay_cases,
    _capture_b12x_decode_graph_bucket,
    _capture_flashinfer_decode_graph_bucket,
    _cosine_similarity,
    _decode_reference_output,
    _make_decode_bucket_shared_inputs,
    _relative_l2_error,
    _resolve_decode_graph_bucket_policy,
)

from .helpers import require_sm120


def test_decode_replay_cases_cover_requested_qwen35_batch_buckets() -> None:
    cases = _build_decode_replay_cases(
        batch_buckets=[1, 2, 4, 8, 12, 16],
        context_tokens=[128, 16_384],
    )

    assert sorted({case.batch for case in cases}) == [1, 2, 4, 8, 12, 16]
    assert sorted({case.context_tokens for case in cases}) == [128, 16_384]

    first_case = next(case for case in cases if case.batch == 1 and case.context_tokens == 128)
    assert first_case.effective_cache_tokens == 129


def test_decode_replay_cases_reject_zero_contexts() -> None:
    with pytest.raises(ValueError, match="decode graph bucket contexts must be positive"):
        _build_decode_replay_cases(
            batch_buckets=[1, 2, 4, 8, 12, 16],
            context_tokens=[0, 16_384],
        )


def test_decode_graph_bucket_policy_defaults_to_heuristic_qwen35_capture_contract() -> None:
    policy = _resolve_decode_graph_bucket_policy(
        batch=1,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        decode_contexts=[128, 16_384, 32_768, 65_536, 131_072],
        capture_context_override=0,
        fixed_split_pages_override=0,
        graph_ctas_per_sm_override=0,
    )

    assert policy.source == "heuristic"
    assert policy.capture_context_tokens == 262_143
    assert policy.capture_page_count == 4_096
    assert policy.capture_fixed_split_pages == 22
    assert policy.replay_fixed_split_pages is None
    assert policy.graph_ctas_per_sm == 6


@torch.inference_mode()
def test_decode_graph_buckets_reuse_single_graph_across_long_contexts_and_match_reference() -> None:
    require_sm120()

    policy = _resolve_decode_graph_bucket_policy(
        batch=1,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        decode_contexts=[128, 16_384, 32_768, 65_536, 131_072],
        capture_context_override=0,
        fixed_split_pages_override=0,
        graph_ctas_per_sm_override=0,
    )
    shared = _make_decode_bucket_shared_inputs(
        batch=1,
        capture_context_tokens=policy.capture_context_tokens,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        seed=17,
    )
    b12x_bucket = _capture_b12x_decode_graph_bucket(
        shared=shared,
        capture_fixed_split_pages=policy.capture_fixed_split_pages,
        replay_fixed_split_pages=policy.replay_fixed_split_pages,
        warmup=1,
        graph_ctas_per_sm=policy.graph_ctas_per_sm,
    )
    fa2_bucket = _capture_flashinfer_decode_graph_bucket(
        shared=shared,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        workspace_bytes=512 * 1024 * 1024,
        warmup=1,
    )

    b12x_graph_id = id(b12x_bucket.graph)
    fa2_graph_id = id(fa2_bucket.graph)

    for context_tokens in (16_384, 131_072):
        b12x_bucket.prepare_replay(context_tokens=context_tokens)
        fa2_bucket.prepare_replay(context_tokens=context_tokens)
        ref_out = _decode_reference_output(
            shared=shared,
            page_table=b12x_bucket.current_page_table,
            cache_seqlens=b12x_bucket.current_cache_seqlens,
            cu_seqlens_q=b12x_bucket.cu_seqlens_q,
        )

        b12x_bucket.graph.replay()
        fa2_bucket.graph.replay()
        torch.cuda.synchronize()

        assert id(b12x_bucket.graph) == b12x_graph_id
        assert id(fa2_bucket.graph) == fa2_graph_id

        assert _relative_l2_error(b12x_bucket.output, ref_out) <= 0.02
        assert _cosine_similarity(b12x_bucket.output, ref_out) >= 0.9999
        assert _relative_l2_error(fa2_bucket.output_view, ref_out) <= 0.005
        assert _cosine_similarity(fa2_bucket.output_view, ref_out) >= 0.99999
