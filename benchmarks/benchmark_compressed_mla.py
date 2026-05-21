#!/usr/bin/env python3
"""Benchmark compressed sparse MLA layouts through the shared MLA core."""

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
    B12XAttentionWorkspace,
    COMPRESSED_MLA_C128_PAGE_SIZE,
    COMPRESSED_MLA_C4_PAGE_SIZE,
    COMPRESSED_MLA_HEAD_DIM,
    COMPRESSED_MLA_INDEX_TOPK,
    COMPRESSED_MLA_LOCAL_Q_HEADS_TP2,
    COMPRESSED_MLA_NOPE_DIM,
    COMPRESSED_MLA_ROPE_DIM,
    COMPRESSED_MLA_SWA_PAGE_SIZE,
    COMPRESSED_MLA_SWA_TOKENS,
    clear_mla_caches,
    compressed_mla_decode_forward,
    compressed_sparse_mla_reference,
    pack_compressed_mla_kv_cache_reference,
    prepare_compressed_mla_core_inputs,
    sparse_mla_reference,
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
_SM_SCALE = 1.0 / math.sqrt(COMPRESSED_MLA_HEAD_DIM)


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


@dataclass(frozen=True)
class Sanity:
    max_abs: float
    rmse: float
    cos: float


@dataclass(frozen=True)
class CaseReport:
    case: BenchmarkCase
    replay_us: float
    p90_replay_us: float
    sanity_core: Sanity | None
    sanity_algorithm: Sanity | None


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"all values must be positive, got {raw!r}")
    return values


def _parse_cases(raw: str, rows: list[int]) -> list[BenchmarkCase]:
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names or names == ["all"]:
        names = ["swa", "c4", "c128", "swa-c4", "swa-c128"]

    cases: list[BenchmarkCase] = []
    for row_count in rows:
        for name in names:
            if name == "swa":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=COMPRESSED_MLA_SWA_TOKENS,
                        indexed_width=0,
                        indexed_page_size=None,
                    )
                )
            elif name == "c4":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=0,
                        indexed_width=COMPRESSED_MLA_INDEX_TOPK,
                        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
                    )
                )
            elif name == "c128":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=0,
                        indexed_width=COMPRESSED_MLA_INDEX_TOPK,
                        indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
                    )
                )
            elif name == "swa-c4":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=COMPRESSED_MLA_SWA_TOKENS,
                        indexed_width=COMPRESSED_MLA_INDEX_TOPK,
                        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
                    )
                )
            elif name == "swa-c128":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=COMPRESSED_MLA_SWA_TOKENS,
                        indexed_width=COMPRESSED_MLA_INDEX_TOPK,
                        indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
                    )
                )
            else:
                raise argparse.ArgumentTypeError(
                    "cases must be one of all,swa,c4,c128,swa-c4,swa-c128; "
                    f"got {name!r}"
                )
    return cases


def _make_q(*, rows: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q = torch.randn(
        (rows, COMPRESSED_MLA_LOCAL_Q_HEADS_TP2, COMPRESSED_MLA_HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
    )
    return (q * 0.04).to(dtype=torch.bfloat16, device=device)


def _make_compressed_cache(
    *,
    tokens: int,
    page_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    k_nope = torch.randn((tokens, COMPRESSED_MLA_NOPE_DIM), generator=gen, dtype=torch.float32) * 0.05
    k_rope = torch.randn((tokens, COMPRESSED_MLA_ROPE_DIM), generator=gen, dtype=torch.float32) * 0.05
    return pack_compressed_mla_kv_cache_reference(
        k_nope.to(device=device),
        k_rope.to(dtype=torch.bfloat16, device=device),
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
    if tokens < width:
        raise ValueError(f"tokens {tokens} must be at least width {width}")
    stride = max(1, tokens // max(1, rows))
    offsets = (torch.arange(rows, dtype=torch.int64, device=device) * stride)[:, None]
    cols = torch.arange(width, dtype=torch.int64, device=device)[None, :]
    return ((offsets + cols) % tokens).to(torch.int32)


def _make_workspace(
    *,
    case: BenchmarkCase,
    max_kv_rows: int,
    device: torch.device,
) -> B12XAttentionWorkspace:
    return B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=COMPRESSED_MLA_LOCAL_Q_HEADS_TP2,
        head_dim=_SHARED_CORE_HEAD_DIM,
        v_head_dim=_SHARED_CORE_V_HEAD_DIM,
        topk=max(1, case.topk),
        max_total_q=case.rows,
        max_batch=case.rows,
        max_kv_rows=max_kv_rows,
        use_cuda_graph=True,
    )


def _sanity(actual: torch.Tensor, expected: torch.Tensor) -> Sanity:
    diff = actual.float() - expected.float()
    flat_actual = actual.float().reshape(-1)
    flat_expected = expected.float().reshape(-1)
    return Sanity(
        max_abs=diff.abs().max().item(),
        rmse=torch.sqrt(torch.mean(diff * diff)).item(),
        cos=torch.nn.functional.cosine_similarity(flat_actual, flat_expected, dim=0).item(),
    )


def _benchmark_case(
    case: BenchmarkCase,
    *,
    device: torch.device,
    seed: int,
    warmup: int,
    replays: int,
    l2_flush,
    verify: bool,
    verify_algorithm: bool,
) -> CaseReport:
    clear_mla_caches()
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

    workspace = _make_workspace(
        case=case,
        max_kv_rows=case.rows * max(1, case.topk),
        device=device,
    )

    output: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal output
        output = compressed_mla_decode_forward(
            q_all=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_lengths,
            indexed_k_cache=indexed_cache,
            indexed_indices=indexed_indices,
            indexed_topk_lengths=indexed_lengths,
            indexed_page_size=case.indexed_page_size,
            workspace=workspace,
            sm_scale=_SM_SCALE,
        )
        return output

    graph = capture_cuda_graph(run, warmup=warmup)
    try:
        stats = bench_cuda_graph(graph, replays=replays, l2_flush=l2_flush)
    finally:
        torch.cuda.synchronize(device)
        del graph
        gc.collect()
        torch.cuda.empty_cache()

    if output is None:
        raise RuntimeError("benchmark graph did not produce an output tensor")

    sanity_core: Sanity | None = None
    sanity_algorithm: Sanity | None = None
    if verify:
        core = prepare_compressed_mla_core_inputs(
            q_all=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_lengths,
            indexed_k_cache=indexed_cache,
            indexed_indices=indexed_indices,
            indexed_topk_lengths=indexed_lengths,
            indexed_page_size=case.indexed_page_size,
        )
        expected_core = sparse_mla_reference(
            q_all=core.q_all,
            kv_cache=core.kv_cache,
            page_table_1=core.page_table_1,
            active_token_counts=core.nsa_cache_seqlens_int32,
            sm_scale=_SM_SCALE,
            v_head_dim=core.v_head_dim,
        )
        sanity_core = _sanity(output, expected_core)
    if verify_algorithm:
        expected_algorithm = compressed_sparse_mla_reference(
            q,
            swa_cache,
            swa_indices,
            swa_lengths,
            sm_scale=_SM_SCALE,
            extra_k_cache=indexed_cache,
            extra_indices=indexed_indices,
            extra_topk_lengths=indexed_lengths,
            extra_page_size=case.indexed_page_size,
        )
        sanity_algorithm = _sanity(output, expected_algorithm)

    replay_us = stats["replay_us"]
    return CaseReport(
        case=case,
        replay_us=statistics.median(replay_us),
        p90_replay_us=statistics.quantiles(replay_us, n=10)[8] if len(replay_us) >= 10 else max(replay_us),
        sanity_core=sanity_core,
        sanity_algorithm=sanity_algorithm,
    )


def _render_report(report: CaseReport) -> str:
    indexed_page = report.case.indexed_page_size if report.case.indexed_page_size is not None else 0
    parts = [
        f"compressed-mla-shared-core case={report.case.name:8s}",
        f"rows={report.case.rows:2d}",
        f"swa={report.case.swa_width:3d}",
        f"indexed={report.case.indexed_width:3d}",
        f"indexed_page={indexed_page:3d}",
        f"topk={report.case.topk:3d}",
        f"replay={report.replay_us:8.2f} us",
        f"p90={report.p90_replay_us:8.2f} us",
    ]
    if report.sanity_core is not None:
        parts.append(
            "core="
            f"max_abs:{report.sanity_core.max_abs:.4f},"
            f"rmse:{report.sanity_core.rmse:.5f},"
            f"cos:{report.sanity_core.cos:.6f}"
        )
    if report.sanity_algorithm is not None:
        parts.append(
            "algorithm="
            f"max_abs:{report.sanity_algorithm.max_abs:.4f},"
            f"rmse:{report.sanity_algorithm.rmse:.5f},"
            f"cos:{report.sanity_algorithm.cos:.6f}"
        )
    return " | ".join(parts)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="all",
        help="comma-separated cases: all,swa,c4,c128,swa-c4,swa-c128",
    )
    parser.add_argument("--rows", type=_parse_csv_ints, default=_parse_csv_ints("1,2,4"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--verify-algorithm",
        action="store_true",
        help="also compare against the compressed-layout algorithm reference",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.warmup <= 0 or args.replays <= 0:
        raise SystemExit("--warmup and --replays must be positive")

    device = require_sm120()
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, l2_flush_bytes)
    flush_desc = f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per replay)" if args.flush_l2 else "off"
    print(f"L2 flush: {flush_desc}")

    reports: list[CaseReport] = []
    for case_idx, case in enumerate(_parse_cases(args.cases, args.rows)):
        report = _benchmark_case(
            case,
            device=device,
            seed=args.seed + case_idx * 17,
            warmup=args.warmup,
            replays=args.replays,
            l2_flush=l2_flush,
            verify=not args.skip_verify,
            verify_algorithm=args.verify_algorithm,
        )
        reports.append(report)
        print(_render_report(report))

    replay_geo = math.exp(statistics.mean(math.log(report.replay_us) for report in reports))
    print(f"Summary | cases={len(reports)} | replay_geo={replay_geo:.2f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
