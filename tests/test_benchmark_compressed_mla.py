from __future__ import annotations

import pytest
import torch

from benchmarks import benchmark_compressed_mla


def _case(rows: int) -> benchmark_compressed_mla.BenchmarkCase:
    return benchmark_compressed_mla.BenchmarkCase(
        name="c128",
        rows=rows,
        swa_width=0,
        indexed_width=512,
        indexed_page_size=2,
    )


def test_algorithm_cosine_threshold_is_hard_failure() -> None:
    case = _case(1)

    benchmark_compressed_mla._check_algorithm_sanity(
        case,
        benchmark_compressed_mla.Sanity(max_abs=0.0, rmse=0.0, cos=0.995),
    )

    try:
        benchmark_compressed_mla._check_algorithm_sanity(
            case,
            benchmark_compressed_mla.Sanity(max_abs=0.0, rmse=0.0, cos=0.994999),
        )
    except benchmark_compressed_mla.BenchmarkFailure as exc:
        assert "cos=0.994999" in str(exc)
        assert "threshold=0.995000" in str(exc)
    else:
        raise AssertionError("expected compressed MLA benchmark to fail on low cosine")


def test_main_prints_no_geomean_on_cosine_failure(monkeypatch, capsys) -> None:
    def fake_collect_case_reports(args, *, device=None):
        del args, device
        raise benchmark_compressed_mla.BenchmarkFailure("synthetic cosine failure")

    monkeypatch.setattr(
        benchmark_compressed_mla,
        "collect_case_reports",
        fake_collect_case_reports,
    )

    rc = benchmark_compressed_mla.main([])

    captured = capsys.readouterr()
    assert rc == 1
    assert "Summary" not in captured.out
    assert "target_ratio" not in captured.out
    assert "synthetic cosine failure" in captured.err


def test_target_summary_uses_rows1_and_rows4096_target_ratios() -> None:
    reports = [
        benchmark_compressed_mla.CaseReport(
            case=_case(1),
            replay_us=25.0,
            p90_replay_us=25.0,
            sanity_algorithm=None,
        ),
        benchmark_compressed_mla.CaseReport(
            case=_case(1),
            replay_us=100.0,
            p90_replay_us=100.0,
            sanity_algorithm=None,
        ),
        benchmark_compressed_mla.CaseReport(
            case=_case(4096),
            replay_us=2000.0,
            p90_replay_us=2000.0,
            sanity_algorithm=None,
        ),
        benchmark_compressed_mla.CaseReport(
            case=_case(4096),
            replay_us=8000.0,
            p90_replay_us=8000.0,
            sanity_algorithm=None,
        ),
    ]

    summary = benchmark_compressed_mla._compute_target_summary(reports)

    assert summary.rows1_geo_us == pytest.approx(50.0)
    assert summary.rows4096_geo_us == pytest.approx(4000.0)
    assert summary.rows1_target_ratio == pytest.approx(2.0)
    assert summary.rows4096_target_ratio == pytest.approx(2.0)
    assert summary.avg_target_ratio == pytest.approx(2.0)
    assert benchmark_compressed_mla._render_summary(reports, summary) == (
        "Summary | cases=4 | rows1_geo=50.00 us | rows1_target_ratio=2.0000 | "
        "rows4096_geo=4000.00 us | rows4096_target_ratio=2.0000 | "
        "avg_target_ratio=2.0000"
    )


def test_parse_cases_accepts_distinct_c4_and_c128_widths() -> None:
    cases = benchmark_compressed_mla._parse_cases(
        "swa-c4,swa-c128",
        [1],
        c4_indexed_width=512,
        c128_indexed_width=2176,
    )

    assert [case.name for case in cases] == ["swa-c4", "swa-c128"]
    assert cases[0].indexed_width == 512
    assert cases[1].indexed_width == 2176
    assert benchmark_compressed_mla._planned_split_chunks(cases[1]) == 192


def test_parse_args_accepts_non_flash_local_q_heads() -> None:
    args = benchmark_compressed_mla._parse_args(["--num-q-heads", "16"])

    assert args.num_q_heads == 16


def test_runtime_valid_widths_preserve_capture_capacity() -> None:
    cases = benchmark_compressed_mla._parse_cases(
        "model",
        [1],
        c4_indexed_width=512,
        c128_indexed_width=8192,
    )

    assert [
        benchmark_compressed_mla._runtime_valid_widths(
            case, context_length=16_384
        )
        for case in cases
    ] == [(128, 0), (128, 512), (128, 128)]
    assert [case.topk for case in cases] == [128, 640, 8320]


def test_parse_args_accepts_runtime_context_length() -> None:
    args = benchmark_compressed_mla._parse_args(["--context-length", "16384"])

    assert args.context_length == 16_384


def test_shared_indexed_cache_uses_one_production_prefill_pool() -> None:
    case = benchmark_compressed_mla.BenchmarkCase(
        name="swa-c128",
        rows=4096,
        swa_width=128,
        indexed_width=8192,
        indexed_page_size=2,
    )

    assert benchmark_compressed_mla._indexed_cache_tokens(
        case,
        shared_indexed_cache=True,
    ) == 8192
    assert benchmark_compressed_mla._indexed_cache_tokens(
        case,
        shared_indexed_cache=False,
    ) == 4096 * 8192

    args = benchmark_compressed_mla._parse_args(["--shared-indexed-cache"])
    assert args.shared_indexed_cache
    assert benchmark_compressed_mla._benchmark_workspace_mode(
        shared_indexed_cache=True
    ) == "extend"
    assert benchmark_compressed_mla._benchmark_workspace_mode(
        shared_indexed_cache=False
    ) == "decode"


def test_reused_cache_pool_preserves_decode_workspace() -> None:
    case = benchmark_compressed_mla.BenchmarkCase(
        name="swa-c128",
        rows=16,
        swa_width=128,
        indexed_width=8192,
        indexed_page_size=2,
    )

    assert benchmark_compressed_mla._indexed_cache_tokens(
        case,
        shared_indexed_cache=False,
        reuse_cache_pool=True,
        cache_num_pages=7792,
    ) == 15_584
    assert benchmark_compressed_mla._swa_cache_tokens(
        case,
        shared_indexed_cache=False,
        reuse_cache_pool=True,
        cache_num_pages=7792,
        swa_page_size=64,
    ) == 498_688
    args = benchmark_compressed_mla._parse_args(["--reuse-cache-pool"])
    assert args.reuse_cache_pool
    assert benchmark_compressed_mla._benchmark_workspace_mode(
        shared_indexed_cache=args.shared_indexed_cache
    ) == "decode"


def test_model_config_derives_live_dsv4_selected_widths() -> None:
    profile = benchmark_compressed_mla._derive_dsv4_compressed_mla_profile(
        {
            "sliding_window": 128,
            "index_topk": 512,
            "max_position_embeddings": 1_048_576,
            "compress_ratios": [0, 0, 4, 128, 4, 128],
        },
        full_token_capacity=340_480,
        c128_pool_size=2_660,
    )

    assert profile.swa_width == 128
    assert profile.c4_indexed_width == 512
    assert profile.c128_indexed_width == 2_688
    assert profile.selected_widths == (128, 640, 2_816)


def test_model_config_derives_c128_width_from_compression_ratio() -> None:
    profile = benchmark_compressed_mla._derive_dsv4_compressed_mla_profile(
        {
            "sliding_window": 128,
            "index_topk": 512,
            "max_position_embeddings": 1_048_576,
            "compress_ratios": [0, 4, 128],
        }
    )

    assert profile.c128_indexed_width == 8_192
    assert profile.selected_widths == (128, 640, 8_320)


def test_trace_preset_uses_full_packed_vllm_pool_geometry() -> None:
    args = benchmark_compressed_mla._parse_args(
        ["--preset", benchmark_compressed_mla.VLLM_DSV4_TRACE_PRESET]
    )

    assert args.cache_page_stride_bytes == 1_039_680
    assert args.cache_num_pages == 7_792
    span = (
        (args.cache_num_pages - 1) * args.cache_page_stride_bytes
        + benchmark_compressed_mla.compressed_mla_page_nbytes(args.swa_page_size)
    )
    assert span == benchmark_compressed_mla.VLLM_DSV4_TRACE_CACHE_VIEW_SPAN_BYTES


def test_cache_views_can_share_a_larger_strided_pool() -> None:
    page_size = 2
    page_nbytes = benchmark_compressed_mla.compressed_mla_page_nbytes(page_size)
    packed = torch.full((2, page_nbytes), 7, dtype=torch.uint8)
    backing = torch.zeros(4 * 4096, dtype=torch.uint8)

    views = benchmark_compressed_mla._make_cache_views(
        packed,
        page_size=page_size,
        block_stride_bytes=4096,
        num_pages=4,
        backing=backing,
        byte_offset=576,
    )

    assert views.sparkinfer.shape == (4, page_nbytes)
    assert views.sparkinfer.stride() == (4096, 1)
    assert views.sparkinfer.storage_offset() == 576
    assert views.flashinfer.shape == (4, page_size, 584)
    assert views.flashinfer.stride() == (4096, 584, 1)
    assert torch.equal(views.sparkinfer[:2], packed)


def test_trace_layer_weights_cover_all_target_layers() -> None:
    weights = benchmark_compressed_mla._trace_layer_weights(
        {
            "num_hidden_layers": 6,
            "compress_ratios": [0, 0, 4, 128, 4, 128, 0],
        }
    )

    assert weights == {"swa": 2, "swa-c4": 2, "swa-c128": 2}


def test_parse_model_cases_uses_runtime_variants() -> None:
    cases = benchmark_compressed_mla._parse_cases(
        "model",
        [1, 4096],
        c4_indexed_width=512,
        c128_indexed_width=2688,
    )

    assert [(case.name, case.rows, case.topk) for case in cases] == [
        ("swa", 1, 128),
        ("swa-c4", 1, 640),
        ("swa-c128", 1, 2816),
        ("swa", 4096, 128),
        ("swa-c4", 4096, 640),
        ("swa-c128", 4096, 2816),
    ]
    assert [benchmark_compressed_mla._planned_split_chunks(case) for case in cases] == [
        11,
        54,
        235,
        1,
        1,
        3,
    ]


def test_target_summary_requires_both_target_rows() -> None:
    try:
        benchmark_compressed_mla._compute_target_summary(
            [
                benchmark_compressed_mla.CaseReport(
                    case=_case(1),
                    replay_us=25.0,
                    p90_replay_us=25.0,
                    sanity_algorithm=None,
                )
            ]
        )
    except benchmark_compressed_mla.BenchmarkFailure as exc:
        assert "requires rows=1 and rows=4096" in str(exc)
        assert "missing rows=4096" in str(exc)
    else:
        raise AssertionError("expected target scoring to require rows=4096")


def test_main_prints_target_ratios(monkeypatch, capsys) -> None:
    reports = [
        benchmark_compressed_mla.CaseReport(
            case=_case(1),
            replay_us=50.0,
            p90_replay_us=50.0,
            sanity_algorithm=None,
        ),
        benchmark_compressed_mla.CaseReport(
            case=_case(4096),
            replay_us=4000.0,
            p90_replay_us=4000.0,
            sanity_algorithm=None,
        ),
    ]

    def fake_collect_case_reports(args, *, device=None):
        del args, device
        return reports

    monkeypatch.setattr(
        benchmark_compressed_mla,
        "collect_case_reports",
        fake_collect_case_reports,
    )
    monkeypatch.setattr(
        benchmark_compressed_mla,
        "resolve_l2_flush_bytes",
        lambda _: 1 << 20,
    )

    rc = benchmark_compressed_mla.main([])

    captured = capsys.readouterr()
    assert rc == 0
    assert "rows1_target_ratio=2.0000" in captured.out
    assert "rows4096_target_ratio=2.0000" in captured.out
    assert "avg_target_ratio=2.0000" in captured.out
    assert "replay_geo" not in captured.out


def test_main_prints_partial_case_reports_without_target_summary(
    monkeypatch, capsys
) -> None:
    reports = [
        benchmark_compressed_mla.CaseReport(
            case=_case(1),
            replay_us=50.0,
            p90_replay_us=50.0,
            sanity_algorithm=None,
        ),
    ]

    def fake_collect_case_reports(args, *, device=None):
        del args, device
        return reports

    monkeypatch.setattr(
        benchmark_compressed_mla,
        "collect_case_reports",
        fake_collect_case_reports,
    )
    monkeypatch.setattr(
        benchmark_compressed_mla,
        "resolve_l2_flush_bytes",
        lambda _: 1 << 20,
    )

    rc = benchmark_compressed_mla.main(["--rows", "1"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "compressed-mla-native case=c128" in captured.out
    assert "Summary skipped:" in captured.out
    assert "target_ratio" not in captured.out
