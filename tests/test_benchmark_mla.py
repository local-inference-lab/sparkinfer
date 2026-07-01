from __future__ import annotations

import json

import torch

from benchmarks import benchmark_mla


_GLM52_TEST_CONFIG = {
    "num_hidden_layers": 78,
    "num_attention_heads": 64,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 2048,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
}


def _write_glm52_config(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_GLM52_TEST_CONFIG))


def test_build_decode_cases_covers_requested_matrix_and_topk_cap() -> None:
    cases = benchmark_mla._build_decode_cases(
        modes=["decode"],
        batch_sizes=[1, 2, 4, 8],
        cache_lens=[1024, 32768, 131072],
        verify_q_lens=[benchmark_mla.DEFAULT_PREFILL_Q_LENS[0]],
        topk_cap=2048,
        decode_row_pattern="uniform",
        page_size=64,
    )

    assert len(cases) == 12
    assert cases[0] == benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=1024, topk=1024)
    assert cases[1] == benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=32768, topk=2048)
    assert cases[-1] == benchmark_mla.DecodeCase(mode="decode", batch_size=8, cache_len=131072, topk=2048)


def test_build_decode_cases_skips_prefill_chunks_larger_than_context() -> None:
    cases = benchmark_mla._build_decode_cases(
        modes=["decode", "prefill"],
        batch_sizes=[1],
        cache_lens=[1024, 32768],
        verify_q_lens=[16384],
        topk_cap=2048,
        decode_row_pattern="uniform",
        page_size=64,
    )

    assert cases == [
        benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=1024, topk=1024),
        benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=32768, topk=2048),
        benchmark_mla.DecodeCase(
            mode="prefill",
            batch_size=1,
            cache_len=32768,
            topk=2048,
            q_len=16384,
        ),
    ]


def test_build_decode_cases_supports_staggered_row_contexts() -> None:
    cases = benchmark_mla._build_decode_cases(
        modes=["decode"],
        batch_sizes=[1, 4, 8],
        cache_lens=[1024],
        verify_q_lens=[benchmark_mla.DEFAULT_PREFILL_Q_LENS[0]],
        topk_cap=2048,
        decode_row_pattern="staggered",
        page_size=64,
    )

    assert cases[0] == benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=1024, topk=1024)
    assert cases[1] == benchmark_mla.DecodeCase(
        mode="decode",
        batch_size=4,
        cache_len=1024,
        topk=1024,
        row_cache_lens=(1024, 768, 512, 256),
    )
    assert cases[2] == benchmark_mla.DecodeCase(
        mode="decode",
        batch_size=8,
        cache_len=1024,
        topk=1024,
        row_cache_lens=(1024, 896, 768, 640, 512, 384, 256, 128),
    )


def test_target_prefill64k_bs1_preset_sets_target_shape() -> None:
    args = benchmark_mla._parse_args(["--preset", benchmark_mla.TARGET_PREFILL64K_BS1_PRESET])

    assert args.modes == "prefill"
    assert args.batch_sizes == "1"
    assert args.cache_lens == "65536"
    assert args.verify_q_lens == "2048"
    assert args.topk_cap == 2048
    assert args.graph_width == 65536


def test_target_glm52_prefill4k_ctx16k_preset_sets_target_shape() -> None:
    args = benchmark_mla._parse_args(
        ["--preset", benchmark_mla.TARGET_GLM52_PREFILL4K_CTX16K_PRESET]
    )

    assert args.modes == "prefill"
    assert args.batch_sizes == "1"
    assert args.cache_lens == "16384"
    assert args.verify_q_lens == "4096"
    assert args.topk_cap == 2048
    assert args.graph_width == 16384
    assert args.use_tiled_topk
    assert args.prefill_indexer_layout == "paged"
    assert args.cache_page_stride_bytes == 0


def test_flashinfer_reference_is_opt_in() -> None:
    assert benchmark_mla._parse_args([]).reference == "none"
    assert (
        benchmark_mla._parse_args(["--reference", "flashinfer"]).reference
        == "flashinfer"
    )


def test_flashinfer_paged_kv_view_is_zero_copy() -> None:
    cache = torch.empty((128, 1, 656), dtype=torch.uint8)

    paged = benchmark_mla._flashinfer_paged_kv_view(cache, page_size=64)

    assert paged.shape == (2, 64, 656)
    assert paged.data_ptr() == cache.data_ptr()


def test_target_glm52_preset_preserves_explicit_packed_stride_regression() -> None:
    args = benchmark_mla._parse_args(
        [
            "--preset",
            benchmark_mla.TARGET_GLM52_PREFILL4K_CTX16K_PRESET,
            "--cache-page-stride-bytes",
            "-1",
        ]
    )

    assert args.cache_page_stride_bytes == -1


def test_resolve_cached_hf_config_follows_main_ref(tmp_path) -> None:
    repo_cache = tmp_path / "models--lukealonso--GLM-5.2-NVFP4"
    revision = "test-revision"
    config_path = repo_cache / "snapshots" / revision / "config.json"
    _write_glm52_config(config_path)
    main_ref = repo_cache / "refs" / "main"
    main_ref.parent.mkdir(parents=True)
    main_ref.write_text(revision)

    resolved = benchmark_mla._resolve_cached_hf_config(cache_root=tmp_path)

    assert resolved == config_path


def test_glm52_all_layer_cache_bytes_match_cached_config(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.json"
    _write_glm52_config(config_path)
    monkeypatch.setattr(
        benchmark_mla,
        "_resolve_cached_hf_config",
        lambda: config_path,
    )

    cfg = benchmark_mla._load_glm_contract_config(tp_size=8, tp_rank=0)

    assert cfg.index_cache_page_bytes == 8_448
    assert cfg.mla_cache_page_bytes == 41_984
    assert cfg.all_layer_cache_block_bytes == 3_933_696
    assert benchmark_mla._resolve_cache_page_stride_bytes(-1, cfg) == 3_933_696


def test_render_case_line_reports_public_step_metrics() -> None:
    report = benchmark_mla.CaseReport(
        case=benchmark_mla.DecodeCase(mode="prefill", batch_size=4, cache_len=32768, topk=2048, q_len=16384),
        graph_width=32768,
        metadata_us=12.34,
        replay_us=567.89,
        indexer_us=123.45,
        mla_us=456.78,
        split_enabled=True,
        chunk_size=256,
        num_chunks=8,
        mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
    )

    line = benchmark_mla._render_case_line(report)

    assert "glm52-prefill tp8" in line
    assert "bs= 4" in line
    assert "q=16384" in line
    assert "ctx= 32768" in line
    assert "graphw= 32768" in line
    assert "topk=2048" in line
    assert "split= on" in line
    assert "chunk=256" in line
    assert "nchunks=8" in line
    assert "step=" in line
    assert "meta=" in line
    assert "replay=" in line
    assert "indexer=" in line
    assert "mla=" in line
    assert "idx_bk=decode" in line


def test_render_case_line_reports_flashinfer_race_and_correctness() -> None:
    report = benchmark_mla.CaseReport(
        case=benchmark_mla.DecodeCase(
            mode="prefill",
            batch_size=1,
            cache_len=16384,
            topk=2048,
            q_len=4096,
        ),
        mla_us=200.0,
        flashinfer_mla_us=250.0,
        mla_sanity=benchmark_mla.SanityMetrics(
            max_abs=0.01,
            rmse=0.001,
            cos=0.9999,
        ),
        flashinfer_mla_sanity=benchmark_mla.SanityMetrics(
            max_abs=0.02,
            rmse=0.002,
            cos=0.9998,
        ),
        b12x_vs_flashinfer_sanity=benchmark_mla.SanityMetrics(
            max_abs=0.02,
            rmse=0.002,
            cos=0.9998,
        ),
    )

    line = benchmark_mla._render_case_line(report)

    assert "fi_mla=  250.00 us" in line
    assert "b12x/fi=0.800x" in line
    assert "fi_cos=0.9998000" in line
    assert "b12x_fi_cos=0.9998000" in line


def test_render_case_line_reports_heterogeneous_decode_context_range() -> None:
    report = benchmark_mla.CaseReport(
        case=benchmark_mla.DecodeCase(
            mode="decode",
            batch_size=4,
            cache_len=131072,
            topk=2048,
            row_cache_lens=(131072, 98304, 65536, 32768),
        ),
        graph_width=131072,
        metadata_us=12.34,
        replay_us=167.89,
        indexer_us=123.45,
        mla_us=44.44,
        split_enabled=True,
        chunk_size=32,
        num_chunks=64,
        mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
    )

    line = benchmark_mla._render_case_line(report)

    assert "glm52-decode tp8" in line
    assert "ctx=131072" in line
    assert "rowctx= 32768-131072" in line


def test_render_summary_lines_reports_geomeans() -> None:
    reports = [
        benchmark_mla.CaseReport(
            case=benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=1024, topk=1024),
            graph_width=8192,
            metadata_us=100.0,
            replay_us=400.0,
            indexer_us=100.0,
            mla_us=400.0,
            split_enabled=False,
            chunk_size=0,
            num_chunks=0,
            mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
        ),
        benchmark_mla.CaseReport(
            case=benchmark_mla.DecodeCase(mode="prefill", batch_size=2, cache_len=32768, topk=2048, q_len=16384),
            graph_width=32768,
            metadata_us=400.0,
            replay_us=100.0,
            indexer_us=400.0,
            mla_us=100.0,
            split_enabled=True,
            chunk_size=256,
            num_chunks=8,
            mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
        ),
    ]

    lines = benchmark_mla._render_summary_lines(reports)

    assert lines[0] == "Summary"
    assert lines[1] == "  cases: 2"
    assert lines[2] == "  total geo:   500.00 us"
    assert lines[3] == "  indexer geo: 200.00 us"
    assert lines[4] == "  mla geo:     200.00 us"
    assert lines[5] == "  step geo:    500.00 us"
    assert lines[6] == "  meta geo:    200.00 us"
    assert lines[7] == "  replay geo:  200.00 us"


def test_render_summary_lines_reports_flashinfer_ratio_direction() -> None:
    report = benchmark_mla.CaseReport(
        case=benchmark_mla.DecodeCase(
            mode="prefill",
            batch_size=1,
            cache_len=16384,
            topk=2048,
            q_len=4096,
        ),
        mla_us=200.0,
        flashinfer_mla_us=250.0,
    )

    lines = benchmark_mla._render_summary_lines([report])

    assert "  flashinfer:  250.00 us" in lines
    assert "  b12x/fi:     0.800x (<1 means b12x faster)" in lines


def test_main_prints_no_stdout_on_failure(monkeypatch, capsys) -> None:
    case = benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=1024, topk=1024)

    def fake_collect_case_reports(args, *, device=None):
        del args, device
        raise benchmark_mla.BenchmarkFailure(case, "synthetic failure")

    monkeypatch.setattr(benchmark_mla, "collect_case_reports", fake_collect_case_reports)

    rc = benchmark_mla.main([])

    captured = capsys.readouterr()
    assert rc == 1
    assert "glm52-" not in captured.out
    assert "Summary" not in captured.out
    assert "synthetic failure" in captured.err


def test_main_prints_buffered_case_lines_and_summary(monkeypatch, capsys) -> None:
    report = benchmark_mla.CaseReport(
        case=benchmark_mla.DecodeCase(mode="prefill", batch_size=1, cache_len=32768, topk=2048, q_len=16384),
        graph_width=8192,
        metadata_us=11.0,
        replay_us=322.0,
        indexer_us=111.0,
        mla_us=222.0,
        split_enabled=True,
        chunk_size=128,
        num_chunks=8,
        mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
    )

    def fake_collect_case_reports(args, *, device=None):
        del args, device
        return [report]

    monkeypatch.setattr(benchmark_mla, "collect_case_reports", fake_collect_case_reports)

    rc = benchmark_mla.main([])

    captured = capsys.readouterr()
    assert rc == 0
    assert "glm52-prefill tp8" in captured.out
    assert "Summary" in captured.out
    assert captured.err == ""


def test_select_paged_topk_matches_full_topk_set_on_cpu() -> None:
    logits = torch.tensor(
        [
            [0.2, 1.7, -0.4, 1.1, 0.5],
            [3.1, -2.0, 0.0, 2.5, 2.4],
        ],
        dtype=torch.float32,
    )
    page_table = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
        ],
        dtype=torch.int32,
    )
    seqlens = torch.tensor([5, 5], dtype=torch.int32)

    actual = benchmark_mla._select_paged_topk_from_logits(
        logits=logits,
        page_table_1=page_table,
        seqlens=seqlens,
        topk=3,
    )

    full_order = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :3]
    expected = torch.gather(page_table, 1, full_order.to(torch.long))
    assert torch.equal(torch.sort(actual, dim=1).values, torch.sort(expected, dim=1).values)


def test_select_paged_topk_uses_forced_persistent_backend(monkeypatch) -> None:
    logits = torch.randn((2, 8), dtype=torch.float32)
    page_table = torch.arange(16, dtype=torch.int32).reshape(2, 8)
    seqlens = torch.full((2,), 8, dtype=torch.int32)
    expected = torch.full((2, 2048), 7, dtype=torch.int32)
    calls: dict[str, object] = {}

    def fake_supports(logits_arg, lengths_arg, *, topk, page_table_1):
        calls["supports"] = (logits_arg, lengths_arg, topk, page_table_1)
        return True

    def fake_run(logits_arg, lengths_arg, *, page_table_1, max_seq_len):
        calls["run"] = (logits_arg, lengths_arg, page_table_1, max_seq_len)
        return expected

    monkeypatch.setattr(benchmark_mla, "supports_persistent_topk2048", fake_supports)
    monkeypatch.setattr(benchmark_mla, "run_persistent_topk2048", fake_run)

    actual = benchmark_mla._select_paged_topk_from_logits(
        logits=logits,
        page_table_1=page_table,
        seqlens=seqlens,
        topk=2048,
        backend="cute_persistent",
    )

    assert actual is expected
    supports_call = calls["supports"]
    assert isinstance(supports_call, tuple)
    assert supports_call[0] is logits
    assert supports_call[2] == 2048
    run_call = calls["run"]
    assert isinstance(run_call, tuple)
    assert run_call[0] is logits
    assert torch.equal(run_call[1], seqlens)
    assert run_call[1].data_ptr() == seqlens.data_ptr()
    assert run_call[2] is page_table
    assert run_call[3] == logits.shape[1]


def test_select_paged_topk_uses_base_page_table_mapping_for_topk_set_on_cpu() -> None:
    logits = torch.tensor(
        [
            [0.2, 1.7, -0.4, 1.1, 0.5],
            [1.4, -0.1, 0.0, 2.5, 2.4],
            [3.1, -2.0, 0.0, 2.5, 2.4],
            [0.4, 0.8, 0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )
    base_page_table = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
        ],
        dtype=torch.int32,
    )
    seqlens = torch.tensor([5, 5, 5, 5], dtype=torch.int32)
    query_row_to_batch = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    actual = benchmark_mla._select_paged_topk_from_logits(
        logits=logits,
        page_table_1=base_page_table,
        seqlens=seqlens,
        topk=3,
        query_row_to_batch=query_row_to_batch,
    )

    full_order = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :3]
    expected = base_page_table[
        query_row_to_batch.to(torch.long).unsqueeze(1),
        full_order.to(torch.long),
    ]
    assert torch.equal(torch.sort(actual, dim=1).values, torch.sort(expected, dim=1).values)


def test_select_ragged_topk_matches_full_stable_argsort_on_cpu() -> None:
    logits = torch.tensor(
        [
            [0.1, 2.2, 1.4, -3.0, 1.0, 0.5],
            [4.0, 1.5, -1.0, 3.5, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    k_start = torch.tensor([1, 2], dtype=torch.int32)
    lengths = torch.tensor([3, 2], dtype=torch.int32)

    actual = benchmark_mla._select_ragged_topk_from_logits(
        logits=logits,
        k_start=k_start,
        lengths=lengths,
        topk=4,
    )

    positions = torch.arange(logits.shape[1], dtype=torch.int32).unsqueeze(0)
    valid = (positions >= k_start.unsqueeze(1)) & (positions < (k_start + lengths).unsqueeze(1))
    masked_logits = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
    full_order = torch.argsort(masked_logits, dim=1, descending=True, stable=True)[:, :4]
    full_values = torch.gather(masked_logits, 1, full_order)
    expected = torch.where(
        torch.isfinite(full_values),
        full_order.to(torch.int32),
        torch.full_like(full_order, -1, dtype=torch.int32),
    )
    assert torch.equal(actual, expected)
