from __future__ import annotations

import pytest

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
