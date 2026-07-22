from __future__ import annotations

import json

import pytest
import torch

from benchmarks.benchmark_paged_attention import (
    _active_split_kv_temporary_results,
    _balanced_graph_replay_schedule,
    _build_decode_replay_cases,
    _capture_sparkinfer_decode_graph_bucket,
    _capture_flashinfer_decode_graph_bucket,
    _cosine_similarity,
    _decode_reference_output,
    _decode_graph_replay_policy_metadata,
    _decode_graph_timing_metadata,
    _make_decode_bucket_shared_inputs,
    _kv_cache_layout_contract,
    _observe_decode_graph_replay_topology,
    _quantize_paged_kv_cache_global_e4m3,
    _record_samples,
    _reference_gate,
    _relative_l2_error,
    _resolve_decode_graph_bucket_policy,
    _strict_backend_replay_for_correctness,
    _strict_guarded_replay_for_correctness,
)

from tests._reference.helpers import require_sparkinfer


def test_balanced_graph_replay_schedule_alternates_ab_ba_pairs() -> None:
    schedule = _balanced_graph_replay_schedule(
        4,
        backend_a="a",
        backend_b="b",
    )

    assert schedule == (("a", "b"), ("b", "a"), ("a", "b"), ("b", "a"))


def test_raw_samples_record_balanced_timing_contract(tmp_path) -> None:
    output_path = tmp_path / "samples.jsonl"
    timing = {
        "method": "paired-interleaved-ab-ba",
        "sample_index": "replay-pair-index",
        "pair_count": 2,
        "even_pair_order": ["sparkinfer", "flashinfer-fa2"],
        "odd_pair_order": ["flashinfer-fa2", "sparkinfer"],
    }

    _record_samples(
        output_path,
        backend="sparkinfer",
        case={"case_contract_sha256": "test"},
        samples_ms=[0.01, 0.02],
        timing=timing,
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["samples"] == [10.0, 20.0]
    assert record["timing"] == timing


def test_observed_decode_graph_topology_reports_device_schedule() -> None:
    class FakeWorkspace:
        plan = type(
            "FakePlan",
            (),
            {"total_q": 2, "page_size": 128, "num_kv_heads": 4},
        )()
        _use_regular_decode_graph_replay = False
        request_indices = torch.zeros(8, dtype=torch.int32)
        block_valid_mask = torch.tensor(
            [1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int32
        )
        merge_indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
        o_indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
        kv_chunk_size_ptr = torch.tensor([512], dtype=torch.int32)

    observed = _observe_decode_graph_replay_topology(FakeWorkspace(), batch=2)

    assert observed == {
        "schema": "sparkinfer-decode-graph-observed-topology-v2",
        "source": "captured-device-lut-updater",
        "scheduling_mode": "compact-valid-mask",
        "kv_chunk_size_tokens": 512,
        "kv_chunk_size_pages": 4,
        "useful_work_items": 3,
        "work_item_capacity": 8,
        "padded_work_items": 5,
        "forward_grid_ctas": 32,
        "useful_forward_ctas": 12,
        "early_exit_forward_ctas": 20,
        "partial_rows": 3,
    }

    FakeWorkspace._use_regular_decode_graph_replay = True
    FakeWorkspace.block_valid_mask.fill_(1)
    regularized = _observe_decode_graph_replay_topology(FakeWorkspace(), batch=2)
    assert regularized["scheduling_mode"] == "regularized-fixed-grid"
    assert regularized["useful_work_items"] == 3
    assert regularized["padded_work_items"] == 5


def test_decode_graph_metadata_records_flashinfer_host_plan_asymmetry() -> None:
    policy = _decode_graph_replay_policy_metadata(include_flashinfer=True)
    backends = policy["backends"]
    sparkinfer = backends["sparkinfer"]
    flashinfer = backends["flashinfer-fa2"]

    assert policy["measurement_scope"] == "captured-cuda-graph-replay-only"
    assert sparkinfer["strict_live_length_graph_safe"] is True
    assert "no page-table" in sparkinfer["runtime_metadata_binding"]
    assert sparkinfer["live_length_dependent_host_planning"] is False
    assert sparkinfer["planning_excluded_from_timing"] == []
    assert sparkinfer["planning_timed"]
    assert flashinfer["strict_live_length_graph_safe"] is False
    assert flashinfer["live_length_dependent_host_planning"] is True
    assert flashinfer["planning_timed"] == []
    assert flashinfer["planning_excluded_from_timing"] == [
        "FlashInfer BatchDecodeWithPagedKVCacheWrapper.plan"
    ]
    assert "not strict graph-safe end-to-end" in policy["comparison_limitation"]

    base_timing = {
        "method": "paired-interleaved-ab-ba",
        "even_pair_order": ["sparkinfer", "flashinfer-fa2"],
        "odd_pair_order": ["flashinfer-fa2", "sparkinfer"],
    }
    timing = _decode_graph_timing_metadata(
        base_timing,
        include_flashinfer=True,
    )
    assert timing["method"] == "paired-interleaved-ab-ba"
    assert timing["even_pair_order"] == ["sparkinfer", "flashinfer-fa2"]
    assert timing["replay_policy"] == policy


def test_kv_cache_layout_contract_distinguishes_combined_strided_views() -> None:
    combined = torch.empty(3, 2, 4, 2, 8)
    k_combined = combined[:, 0]
    v_combined = combined[:, 1]

    combined_contract = _kv_cache_layout_contract(k_combined, v_combined)
    separate_contract = _kv_cache_layout_contract(
        k_combined.contiguous(),
        v_combined.contiguous(),
    )

    assert combined_contract["kind"] == "combined-pages-2-nhd-strided-views"
    assert combined_contract["shared_storage"] is True
    assert combined_contract["k_stride"] == [128, 16, 8, 1]
    assert combined_contract["v_stride"] == [128, 16, 8, 1]
    assert combined_contract["k_storage_offset_elements"] == 0
    assert combined_contract["v_storage_offset_elements"] == 64
    assert separate_contract["kind"] == "separate-contiguous-nhd"
    assert separate_contract["shared_storage"] is False


def test_fp8_quantization_preserves_requested_combined_kv_layout() -> None:
    k_cache = torch.randn(2, 4, 2, 8, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)

    k_fp8, v_fp8, *_ = _quantize_paged_kv_cache_global_e4m3(
        k_cache,
        v_cache,
        batch=1,
        kv_heads=2,
        combined_kv_cache=True,
    )

    layout = _kv_cache_layout_contract(k_fp8, v_fp8)
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert v_fp8.dtype == torch.float8_e4m3fn
    assert layout["kind"] == "combined-pages-2-nhd-strided-views"
    assert layout["shared_storage"] is True


def test_reference_gate_rejects_a_finite_but_incorrect_backend_output() -> None:
    reference = torch.ones(2, 3, dtype=torch.float32)
    output = -reference

    with pytest.raises(AssertionError, match="failed the Torch reference gate"):
        _reference_gate(
            backend="test-backend",
            output=output,
            reference=reference,
        )


def test_correctness_guard_selects_regular_decode_fixed_stride_rows() -> None:
    tmp_output = torch.full((16, 2, 3), float("nan"))
    tmp_lse = torch.full((16, 2), float("nan"))
    o_indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    active_fixed_rows = torch.tensor([0, 1, 8, 9])
    tmp_output[active_fixed_rows] = 1
    tmp_lse[active_fixed_rows] = 2

    active_output, active_lse = _active_split_kv_temporary_results(
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        o_indptr=o_indptr,
        batch=2,
        regular_decode_graph=True,
    )

    assert active_output.shape == (4, 2, 3)
    assert active_lse.shape == (4, 2)
    assert torch.isfinite(active_output).all()
    assert torch.isfinite(active_lse).all()
    assert torch.isnan(tmp_output[:4]).any()


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
    assert policy.graph_ctas_per_sm == 6
    assert policy.query_tiles_per_request == 1
    assert policy.max_chunks_per_request > 1
    assert policy.max_work_items == policy.max_chunks_per_request
    assert policy.max_partial_rows == policy.max_chunks_per_request


def test_decode_graph_bucket_policy_kv4_uses_kv_head_aware_chunk_budget() -> None:
    policy = _resolve_decode_graph_bucket_policy(
        batch=16,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        page_size=64,
        q_heads=8,
        kv_heads=4,
        head_dim=256,
        decode_contexts=[128, 131_072],
        capture_context_override=0,
        fixed_split_pages_override=0,
        graph_ctas_per_sm_override=0,
    )

    num_sms = int(torch.cuda.get_device_properties("cuda").multi_processor_count)
    expected_architecture_budget = max(
        (num_sms * policy.graph_ctas_per_sm)
        // (policy.batch * 4 * policy.query_tiles_per_request),
        1,
    )
    wrong_kv1_budget = max(
        (num_sms * policy.graph_ctas_per_sm)
        // (policy.batch * policy.query_tiles_per_request),
        1,
    )
    assert policy.architecture_max_chunks_per_request == expected_architecture_budget
    assert policy.architecture_max_chunks_per_request < wrong_kv1_budget
    assert policy.max_chunks_per_request <= expected_architecture_budget


@torch.inference_mode()
def test_decode_graph_bucket_kv4_captures_exact_production_grid() -> None:
    require_sparkinfer()
    policy = _resolve_decode_graph_bucket_policy(
        batch=16,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        page_size=64,
        q_heads=8,
        kv_heads=4,
        head_dim=256,
        decode_contexts=[128],
        capture_context_override=0,
        fixed_split_pages_override=0,
        graph_ctas_per_sm_override=0,
    )
    shared = _make_decode_bucket_shared_inputs(
        batch=16,
        capture_context_tokens=policy.capture_context_tokens,
        page_size=64,
        q_heads=8,
        kv_heads=4,
        head_dim=256,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        seed=23,
    )
    bucket = _capture_sparkinfer_decode_graph_bucket(
        shared=shared,
        policy=policy,
        warmup=1,
    )

    assert bucket.workspace.request_indices.numel() == policy.max_work_items
    assert policy.max_work_items == (
        policy.batch
        * policy.query_tiles_per_request
        * policy.max_chunks_per_request
    )
    bucket.prepare_replay(context_tokens=128)
    ref_out = _decode_reference_output(read_only_snapshot=bucket.read_only_snapshot)
    _strict_backend_replay_for_correctness(bucket)
    assert _relative_l2_error(bucket.output, ref_out) <= 0.02
    assert _cosine_similarity(bucket.output, ref_out) >= 0.9999


@pytest.mark.parametrize(
    ("fixed_split_pages", "graph_ctas_per_sm", "match"),
    [
        (8, 0, "--fixed-split-pages is only supported by legacy-matrix"),
        (0, 4, "--graph-ctas-per-sm is only supported by legacy-matrix"),
    ],
)
def test_decode_graph_bucket_policy_rejects_nonproduction_overrides(
    fixed_split_pages: int,
    graph_ctas_per_sm: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _resolve_decode_graph_bucket_policy(
            batch=1,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            page_size=64,
            q_heads=8,
            kv_heads=1,
            head_dim=256,
            decode_contexts=[128],
            capture_context_override=0,
            fixed_split_pages_override=fixed_split_pages,
            graph_ctas_per_sm_override=graph_ctas_per_sm,
        )


@torch.inference_mode()
def test_decode_graph_buckets_reuse_single_graph_across_long_contexts_and_match_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sparkinfer()

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
    sparkinfer_bucket = _capture_sparkinfer_decode_graph_bucket(
        shared=shared,
        policy=policy,
        warmup=1,
    )
    assert sparkinfer_bucket.current_plan_desc.endswith(",split")
    assert sparkinfer_bucket.current_plan_desc.startswith("chunk=device-lut,")
    assert sparkinfer_bucket.workspace._decode_graph_chunk_pages_lut is not None
    assert sparkinfer_bucket.workspace.request_indices.numel() == policy.max_work_items
    assert sparkinfer_bucket.workspace._uses_plan_owned_decode_graph_metadata is True
    assert sparkinfer_bucket.scratch_plan._decode_graph_replay_state_captured is True
    assert sparkinfer_bucket.forward_traits_contract["cta_tile_kv"] > 0
    assert sparkinfer_bucket.forward_traits_contract["num_mma_kv"] > 0
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

    sparkinfer_graph_id = id(sparkinfer_bucket.graph)
    fa2_graph_id = id(fa2_bucket.graph)
    sparkinfer_plan_id = id(sparkinfer_bucket.workspace.plan)
    sparkinfer_page_table_ptr = sparkinfer_bucket.current_page_table.data_ptr()
    sparkinfer_cache_seqlens_ptr = sparkinfer_bucket.current_cache_seqlens.data_ptr()

    def reject_host_replan(**_kwargs: object) -> None:
        raise AssertionError("decode graph replay must not build a live host PagedPlan")

    monkeypatch.setattr(
        "benchmarks.benchmark_paged_attention._build_backend_graph_plan",
        reject_host_replan,
    )

    for context_tokens in (16_384, 131_072):
        sparkinfer_bucket.prepare_replay(context_tokens=context_tokens)
        fa2_bucket.prepare_replay(context_tokens=context_tokens)
        ref_out = _decode_reference_output(
            read_only_snapshot=sparkinfer_bucket.read_only_snapshot,
        )

        _strict_backend_replay_for_correctness(sparkinfer_bucket)
        _strict_guarded_replay_for_correctness(
            backend="flashinfer-fa2",
            graph=fa2_bucket.graph,
            guarded_output=fa2_bucket.guarded_output,
            read_only_snapshot=fa2_bucket.read_only_snapshot,
            read_only_inputs=fa2_bucket.read_only_inputs,
        )

        assert id(sparkinfer_bucket.graph) == sparkinfer_graph_id
        assert id(sparkinfer_bucket.workspace.plan) == sparkinfer_plan_id
        assert sparkinfer_bucket.current_page_table.data_ptr() == sparkinfer_page_table_ptr
        assert (
            sparkinfer_bucket.current_cache_seqlens.data_ptr()
            == sparkinfer_cache_seqlens_ptr
        )
        assert id(fa2_bucket.graph) == fa2_graph_id

        assert _relative_l2_error(sparkinfer_bucket.output, ref_out) <= 0.02
        assert _cosine_similarity(sparkinfer_bucket.output, ref_out) >= 0.9999
        assert _relative_l2_error(fa2_bucket.output_view, ref_out) <= 0.005
        assert _cosine_similarity(fa2_bucket.output_view, ref_out) >= 0.99999
