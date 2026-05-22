#!/usr/bin/env python3
"""Benchmark compressed MLA prep kernels with CUDA graph replay."""

from __future__ import annotations

import argparse
import gc
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.integration.mla import (
    COMPRESSED_MLA_C128_PAGE_SIZE,
    COMPRESSED_MLA_C4_PAGE_SIZE,
    COMPRESSED_MLA_HEAD_DIM,
    COMPRESSED_MLA_INDEX_TOPK,
    COMPRESSED_MLA_LOCAL_Q_HEADS_TP2,
    COMPRESSED_MLA_NOPE_DIM,
    COMPRESSED_MLA_ROPE_DIM,
    COMPRESSED_MLA_SWA_PAGE_SIZE,
    COMPRESSED_MLA_SWA_TOKENS,
    pack_compressed_mla_kv_cache_reference,
    prepare_compressed_mla_core_inputs,
)

from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
    resolve_l2_flush_bytes,
)


_SHARED_CORE_HEAD_DIM = 576
_SHARED_CORE_V_HEAD_DIM = 512


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    rows: int
    swa_width: int
    indexed_width: int
    indexed_page_size: int | None

    @property
    def topk(self) -> int:
        return self.swa_width + self.indexed_width


@dataclass
class PrepWorkspace:
    topk: int
    max_total_q: int
    use_cuda_graph: bool = True
    fixed_capacity: bool = True


@dataclass(frozen=True)
class CaseReport:
    impl: str
    case: BenchmarkCase
    replay_us: float
    p90_replay_us: float


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"all values must be positive, got {raw!r}")
    return values


def _parse_csv(raw: str) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _parse_cases(raw: str, rows: list[int]) -> list[BenchmarkCase]:
    names = _parse_csv(raw)
    if names == ["all"]:
        names = ["swa", "c4", "c128", "swa-c4", "swa-c128"]

    cases: list[BenchmarkCase] = []
    for row_count in rows:
        for name in names:
            if name == "swa":
                cases.append(BenchmarkCase(name, row_count, COMPRESSED_MLA_SWA_TOKENS, 0, None))
            elif name == "c4":
                cases.append(
                    BenchmarkCase(name, row_count, 0, COMPRESSED_MLA_INDEX_TOPK, COMPRESSED_MLA_C4_PAGE_SIZE)
                )
            elif name == "c128":
                cases.append(
                    BenchmarkCase(name, row_count, 0, COMPRESSED_MLA_INDEX_TOPK, COMPRESSED_MLA_C128_PAGE_SIZE)
                )
            elif name == "swa-c4":
                cases.append(
                    BenchmarkCase(
                        name,
                        row_count,
                        COMPRESSED_MLA_SWA_TOKENS,
                        COMPRESSED_MLA_INDEX_TOPK,
                        COMPRESSED_MLA_C4_PAGE_SIZE,
                    )
                )
            elif name == "swa-c128":
                cases.append(
                    BenchmarkCase(
                        name,
                        row_count,
                        COMPRESSED_MLA_SWA_TOKENS,
                        COMPRESSED_MLA_INDEX_TOPK,
                        COMPRESSED_MLA_C128_PAGE_SIZE,
                    )
                )
            else:
                raise argparse.ArgumentTypeError(
                    "cases must be one of all,swa,c4,c128,swa-c4,swa-c128; "
                    f"got {name!r}"
                )
    return cases


def _make_q(*, rows: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn(
        (rows, COMPRESSED_MLA_LOCAL_Q_HEADS_TP2, COMPRESSED_MLA_HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    )
    return (q * 0.04).to(dtype=torch.bfloat16)


def _make_compressed_cache(
    *,
    tokens: int,
    page_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k_nope = torch.randn(
        (tokens, COMPRESSED_MLA_NOPE_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    ) * 0.05
    k_rope = torch.randn(
        (tokens, COMPRESSED_MLA_ROPE_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    ) * 0.05
    return pack_compressed_mla_kv_cache_reference(
        k_nope,
        k_rope.to(dtype=torch.bfloat16),
        page_size=page_size,
    )


def _make_indices(
    *,
    rows: int,
    width: int,
    tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if width == 0:
        return torch.empty((rows, 0), dtype=torch.int32, device=device)
    stride = max(1, tokens // max(1, rows))
    offsets = (torch.arange(rows, dtype=torch.int64, device=device) * stride)[:, None]
    cols = torch.arange(width, dtype=torch.int64, device=device)[None, :]
    return ((offsets + cols) % tokens).to(torch.int32)


def _make_workspace(case: BenchmarkCase) -> PrepWorkspace:
    return PrepWorkspace(
        topk=max(1, case.topk),
        max_total_q=case.rows,
    )


def _prepare_case_inputs(case: BenchmarkCase, *, seed: int, device: torch.device):
    q = _make_q(rows=case.rows, seed=seed, device=device)
    swa_tokens = max(case.swa_width, 1)
    swa_cache = _make_compressed_cache(
        tokens=swa_tokens,
        page_size=COMPRESSED_MLA_SWA_PAGE_SIZE,
        seed=seed + 1,
        device=device,
    )
    swa_indices = _make_indices(rows=case.rows, width=case.swa_width, tokens=swa_tokens, device=device)
    swa_lengths = torch.full((case.rows,), case.swa_width, dtype=torch.int32, device=device)

    indexed_cache: torch.Tensor | None = None
    indexed_indices: torch.Tensor | None = None
    indexed_lengths: torch.Tensor | None = None
    if case.indexed_width:
        assert case.indexed_page_size is not None
        indexed_tokens = case.indexed_width * max(case.rows, 1)
        indexed_cache = _make_compressed_cache(
            tokens=indexed_tokens,
            page_size=case.indexed_page_size,
            seed=seed + 2,
            device=device,
        )
        indexed_indices = _make_indices(
            rows=case.rows,
            width=case.indexed_width,
            tokens=indexed_tokens,
            device=device,
        )
        indexed_lengths = torch.full((case.rows,), case.indexed_width, dtype=torch.int32, device=device)

    return q, swa_cache, swa_indices, swa_lengths, indexed_cache, indexed_indices, indexed_lengths


def _run_prep(
    case: BenchmarkCase,
    *,
    impl: str,
    workspace: PrepWorkspace,
    inputs,
):
    q, swa_cache, swa_indices, swa_lengths, indexed_cache, indexed_indices, indexed_lengths = inputs
    return prepare_compressed_mla_core_inputs(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        swa_page_size=COMPRESSED_MLA_SWA_PAGE_SIZE,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices,
        indexed_topk_lengths=indexed_lengths,
        indexed_page_size=case.indexed_page_size,
        workspace=workspace,
        kv_kernel_impl=impl,
    )


def _verify_case(case: BenchmarkCase, *, impl: str, inputs) -> None:
    if impl == "triton":
        return
    triton_workspace = _make_workspace(case)
    impl_workspace = _make_workspace(case)
    expected = _run_prep(case, impl="triton", workspace=triton_workspace, inputs=inputs)
    actual = _run_prep(case, impl=impl, workspace=impl_workspace, inputs=inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.q_all, expected.q_all, atol=0, rtol=0)
    torch.testing.assert_close(actual.cache_seqlens_int32, expected.cache_seqlens_int32, atol=0, rtol=0)
    actual_rows = actual.kv_cache.view(-1, 656)
    expected_rows = expected.kv_cache.view(-1, 656)
    torch.testing.assert_close(actual_rows[:, 528:], expected_rows[:, 528:], atol=0, rtol=0)
    torch.testing.assert_close(
        actual_rows[:, 512:528].view(torch.float32),
        expected_rows[:, 512:528].view(torch.float32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        _dequant_core_payload(actual_rows),
        _dequant_core_payload(expected_rows),
        atol=2e-6,
        rtol=0,
    )


def _dequant_core_payload(rows: torch.Tensor) -> torch.Tensor:
    fp8 = rows[:, :512].contiguous().view(torch.float8_e4m3fn).to(torch.float32).view(-1, 512)
    scales = rows[:, 512:528].view(torch.float32).repeat_interleave(128, dim=1)
    return fp8 * scales


def _profile_case(case: BenchmarkCase, *, impl: str, inputs, warmup: int) -> None:
    workspace = _make_workspace(case)
    for _ in range(warmup):
        _run_prep(case, impl=impl, workspace=workspace, inputs=inputs)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        _run_prep(case, impl=impl, workspace=workspace, inputs=inputs)
    torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


def _benchmark_case(
    case: BenchmarkCase,
    *,
    impl: str,
    inputs,
    warmup: int,
    replays: int,
    l2_flush,
    verify: bool,
    profile: bool,
) -> CaseReport:
    if verify:
        _verify_case(case, impl=impl, inputs=inputs)

    workspace = _make_workspace(case)
    output = None

    def run():
        nonlocal output
        output = _run_prep(case, impl=impl, workspace=workspace, inputs=inputs)
        return output.kv_cache

    graph = capture_cuda_graph(run, warmup=warmup)
    try:
        stats = bench_cuda_graph(graph, replays=replays, l2_flush=l2_flush)
    finally:
        torch.cuda.synchronize()
        del graph
        gc.collect()
        torch.cuda.empty_cache()
    if profile:
        _profile_case(case, impl=impl, inputs=inputs, warmup=warmup)

    replay_us = stats["replay_us"]
    return CaseReport(
        impl=impl,
        case=case,
        replay_us=statistics.median(replay_us),
        p90_replay_us=statistics.quantiles(replay_us, n=10)[8] if len(replay_us) >= 10 else max(replay_us),
    )


def _render_report(report: CaseReport) -> str:
    indexed_page = report.case.indexed_page_size if report.case.indexed_page_size is not None else 0
    return " | ".join(
        [
            f"compressed-mla-prep impl={report.impl:6s}",
            f"case={report.case.name:8s}",
            f"rows={report.case.rows:5d}",
            f"swa={report.case.swa_width:3d}",
            f"indexed={report.case.indexed_width:3d}",
            f"indexed_page={indexed_page:3d}",
            f"topk={report.case.topk:3d}",
            f"replay={report.replay_us:9.2f} us",
            f"p90={report.p90_replay_us:9.2f} us",
        ]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--impls", type=_parse_csv, default=_parse_csv("triton,cute"))
    parser.add_argument("--rows", type=_parse_csv_ints, default=_parse_csv_ints("1,4096"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--seed", type=int, default=101_000)
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.warmup <= 0 or args.replays <= 0:
        raise SystemExit("--warmup and --replays must be positive")
    for impl in args.impls:
        if impl not in {"triton", "cute"}:
            raise SystemExit(f"--impls entries must be triton or cute, got {impl!r}")

    device = require_sm120()
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, l2_flush_bytes)
    flush_desc = f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per replay)" if args.flush_l2 else "off"
    print(f"L2 flush: {flush_desc}")

    reports: list[CaseReport] = []
    cases = _parse_cases(args.cases, args.rows)
    for case_idx, case in enumerate(cases):
        inputs = _prepare_case_inputs(case, seed=args.seed + case_idx * 17, device=device)
        for impl in args.impls:
            report = _benchmark_case(
                case,
                impl=impl,
                inputs=inputs,
                warmup=args.warmup,
                replays=args.replays,
                l2_flush=l2_flush,
                verify=not args.skip_verify,
                profile=args.profile,
            )
            reports.append(report)
            print(_render_report(report))

    for impl in args.impls:
        impl_reports = [report for report in reports if report.impl == impl]
        replay_geo = math.exp(statistics.mean(math.log(report.replay_us) for report in impl_reports))
        print(f"Summary | impl={impl} | cases={len(impl_reports)} | replay_geo={replay_geo:.2f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
