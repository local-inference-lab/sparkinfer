#!/usr/bin/env python3
"""Benchmark native compressed sparse MLA layouts."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.attention.workspace import B12XAttentionWorkspace
from b12x.attention.mla.compressed_reference import (
    COMPRESSED_MLA_C128_PAGE_SIZE,
    COMPRESSED_MLA_C4_PAGE_SIZE,
    COMPRESSED_MLA_DSV4_PAGE_SIZE,
    COMPRESSED_MLA_HEAD_DIM,
    COMPRESSED_MLA_NOPE_DIM,
    COMPRESSED_MLA_ROPE_DIM,
    COMPRESSED_MLA_SWA_TOKENS,
    compressed_sparse_mla_reference,
    pack_compressed_mla_kv_cache_reference,
)
from b12x.integration.mla import (
    clear_mla_caches,
    compressed_mla_decode_forward,
    compressed_mla_split_chunks_for_contract,
)

from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
    resolve_l2_flush_bytes,
)


_SM_SCALE = 1.0 / math.sqrt(COMPRESSED_MLA_HEAD_DIM)
_ALGORITHM_COS_TOL = 0.995
_DECODE_TARGET_US = 25.0
_PREFILL4096_TARGET_US = 2_000.0
_PAGE_INDEX_ALIGNMENT = 64
_DEFAULT_NUM_Q_HEADS = 32
_DEFAULT_INDEX_TOPK = 512


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
    sanity_algorithm: Sanity | None


@dataclass(frozen=True)
class TargetSummary:
    rows1_geo_us: float
    rows4096_geo_us: float
    rows1_target_ratio: float
    rows4096_target_ratio: float
    avg_target_ratio: float


@dataclass(frozen=True)
class DSV4CompressedMLAProfile:
    swa_width: int
    c4_indexed_width: int
    c128_indexed_width: int
    selected_widths: tuple[int, ...]


class BenchmarkFailure(RuntimeError):
    pass


def _align_up(value: int, alignment: int = _PAGE_INDEX_ALIGNMENT) -> int:
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def _load_model_config(path: pathlib.Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"model config must be a JSON object: {path}")
    return loaded


def _derive_dsv4_compressed_mla_profile(
    config: dict[str, object],
    *,
    full_token_capacity: int | None = None,
    c128_pool_size: int | None = None,
) -> DSV4CompressedMLAProfile:
    sliding_window = int(config.get("sliding_window", COMPRESSED_MLA_SWA_TOKENS))
    c4_topk = int(config.get("index_topk", _DEFAULT_INDEX_TOPK))
    max_positions = int(config.get("max_position_embeddings", 0))
    compress_ratios_raw = config.get("compress_ratios", ())
    if compress_ratios_raw is None:
        compress_ratios_raw = ()
    compress_ratios = tuple(int(value) for value in compress_ratios_raw)  # type: ignore[arg-type]

    uses_c4 = 4 in compress_ratios
    uses_c128 = 128 in compress_ratios
    swa_width = _align_up(sliding_window)
    c4_indexed_width = _align_up(c4_topk) if uses_c4 else 0

    c128_source_tokens = full_token_capacity
    if c128_source_tokens is None:
        if c128_pool_size is not None:
            c128_source_tokens = c128_pool_size * COMPRESSED_MLA_C128_PAGE_SIZE
        else:
            c128_source_tokens = max_positions

    c128_width = 0
    if uses_c128 and c128_source_tokens:
        c128_width = (int(c128_source_tokens) + COMPRESSED_MLA_C128_PAGE_SIZE - 1) // COMPRESSED_MLA_C128_PAGE_SIZE
        if c128_pool_size is not None:
            c128_width = min(c128_width, int(c128_pool_size))
    c128_indexed_width = _align_up(c128_width) if c128_width else 0

    selected_widths = {swa_width}
    if c4_indexed_width:
        selected_widths.add(swa_width + c4_indexed_width)
    if c128_indexed_width:
        selected_widths.add(swa_width + c128_indexed_width)

    return DSV4CompressedMLAProfile(
        swa_width=swa_width,
        c4_indexed_width=c4_indexed_width,
        c128_indexed_width=c128_indexed_width,
        selected_widths=tuple(sorted(selected_widths)),
    )


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"all values must be positive, got {raw!r}")
    return values


def _parse_cases(
    raw: str,
    rows: list[int],
    *,
    c4_indexed_width: int = _DEFAULT_INDEX_TOPK,
    c128_indexed_width: int = _DEFAULT_INDEX_TOPK,
) -> list[BenchmarkCase]:
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names or names == ["all"]:
        names = ["swa", "c4", "c128", "swa-c4", "swa-c128"]
    elif names == ["model"]:
        names = ["swa", "swa-c4", "swa-c128"]

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
                        indexed_width=c4_indexed_width,
                        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
                    )
                )
            elif name == "c128":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=0,
                        indexed_width=c128_indexed_width,
                        indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
                    )
                )
            elif name == "swa-c4":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=COMPRESSED_MLA_SWA_TOKENS,
                        indexed_width=c4_indexed_width,
                        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
                    )
                )
            elif name == "swa-c128":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=COMPRESSED_MLA_SWA_TOKENS,
                        indexed_width=c128_indexed_width,
                        indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
                    )
                )
            else:
                raise argparse.ArgumentTypeError(
                    "cases must be one of all,model,swa,c4,c128,swa-c4,swa-c128; "
                    f"got {name!r}"
                )
    return cases


def _resolve_case_widths(args: argparse.Namespace) -> tuple[int, int]:
    profile: DSV4CompressedMLAProfile | None = None
    if args.model_config is not None:
        profile = _derive_dsv4_compressed_mla_profile(
            _load_model_config(args.model_config),
            full_token_capacity=args.full_token_capacity,
            c128_pool_size=args.c128_pool_size,
        )

    c4_indexed_width = args.c4_indexed_width
    if c4_indexed_width is None:
        c4_indexed_width = (
            profile.c4_indexed_width
            if profile is not None and profile.c4_indexed_width
            else _DEFAULT_INDEX_TOPK
        )

    c128_indexed_width = args.c128_indexed_width
    if c128_indexed_width is None:
        c128_indexed_width = (
            profile.c128_indexed_width
            if profile is not None and profile.c128_indexed_width
            else _DEFAULT_INDEX_TOPK
        )

    if c4_indexed_width <= 0 or c128_indexed_width <= 0:
        raise ValueError(
            "--c4-indexed-width and --c128-indexed-width must be positive after model derivation"
        )
    return int(c4_indexed_width), int(c128_indexed_width)


def _planned_split_chunks(case: BenchmarkCase) -> int:
    return compressed_mla_split_chunks_for_contract(
        rows=case.rows,
        width=max(1, case.topk),
    )


def _make_q(*, rows: int, num_q_heads: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn(
        (rows, num_q_heads, COMPRESSED_MLA_HEAD_DIM),
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
    k_nope = (
        torch.randn(
            (tokens, COMPRESSED_MLA_NOPE_DIM),
            generator=gen,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    )
    k_rope = (
        torch.randn(
            (tokens, COMPRESSED_MLA_ROPE_DIM),
            generator=gen,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    )
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
    if tokens < width:
        raise ValueError(f"tokens {tokens} must be at least width {width}")
    stride = max(1, tokens // max(1, rows))
    offsets = (torch.arange(rows, dtype=torch.int64, device=device) * stride)[:, None]
    cols = torch.arange(width, dtype=torch.int64, device=device)[None, :]
    return ((offsets + cols) % tokens).to(torch.int32)


def _make_workspace(
    *,
    case: BenchmarkCase,
    num_q_heads: int,
    device: torch.device,
) -> B12XAttentionWorkspace:
    return B12XAttentionWorkspace.for_contract(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=num_q_heads,
        head_dim=COMPRESSED_MLA_HEAD_DIM,
        v_head_dim=COMPRESSED_MLA_HEAD_DIM,
        topk=max(1, case.topk),
        max_total_q=case.rows,
        max_batch=case.rows,
        use_cuda_graph=True,
        max_chunks_per_row=_planned_split_chunks(case),
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


def _check_algorithm_sanity(case: BenchmarkCase, sanity: Sanity) -> None:
    if not math.isfinite(sanity.cos) or sanity.cos < _ALGORITHM_COS_TOL:
        raise BenchmarkFailure(
            "compressed MLA algorithm cosine below threshold for "
            f"case={case.name} rows={case.rows}: "
            f"max_abs={sanity.max_abs:.6f} rmse={sanity.rmse:.6f} "
            f"cos={sanity.cos:.6f} threshold={_ALGORITHM_COS_TOL:.6f}"
        )


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geomean requires at least one value")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"geomean values must be positive, got {values}")
    return math.exp(statistics.mean(math.log(value) for value in values))


def _compute_target_summary(reports: list[CaseReport]) -> TargetSummary:
    by_rows: dict[int, list[float]] = {}
    for report in reports:
        by_rows.setdefault(report.case.rows, []).append(report.replay_us)

    missing = [rows for rows in (1, 4096) if rows not in by_rows]
    if missing:
        raise BenchmarkFailure(
            "compressed MLA target scoring requires rows=1 and rows=4096; "
            f"missing rows={','.join(str(row) for row in missing)}"
        )

    rows1_geo = _geomean(by_rows[1])
    rows4096_geo = _geomean(by_rows[4096])
    rows1_ratio = rows1_geo / _DECODE_TARGET_US
    rows4096_ratio = rows4096_geo / _PREFILL4096_TARGET_US
    return TargetSummary(
        rows1_geo_us=rows1_geo,
        rows4096_geo_us=rows4096_geo,
        rows1_target_ratio=rows1_ratio,
        rows4096_target_ratio=rows4096_ratio,
        avg_target_ratio=(rows1_ratio + rows4096_ratio) / 2.0,
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
    num_q_heads: int,
) -> CaseReport:
    clear_mla_caches()
    q = _make_q(rows=case.rows, num_q_heads=num_q_heads, seed=seed, device=device)

    swa_tokens = max(case.swa_width, 1)
    swa_cache = _make_compressed_cache(
        tokens=swa_tokens,
        page_size=COMPRESSED_MLA_DSV4_PAGE_SIZE,
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
        num_q_heads=num_q_heads,
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

        if output is None:
            raise RuntimeError("benchmark graph did not produce an output tensor")

        sanity_algorithm: Sanity | None = None
        if verify:
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
            _check_algorithm_sanity(case, sanity_algorithm)
    finally:
        torch.cuda.synchronize(device)
        del graph
        gc.collect()
        torch.cuda.empty_cache()

    replay_us = stats["replay_us"]
    return CaseReport(
        case=case,
        replay_us=statistics.median(replay_us),
        p90_replay_us=statistics.quantiles(replay_us, n=10)[8] if len(replay_us) >= 10 else max(replay_us),
        sanity_algorithm=sanity_algorithm,
    )


def collect_case_reports(args: argparse.Namespace, *, device: torch.device | None = None) -> list[CaseReport]:
    if device is None:
        device = require_sm120()
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, l2_flush_bytes)
    c4_indexed_width, c128_indexed_width = _resolve_case_widths(args)

    reports: list[CaseReport] = []
    for case_idx, case in enumerate(
        _parse_cases(
            args.cases,
            args.rows,
            c4_indexed_width=c4_indexed_width,
            c128_indexed_width=c128_indexed_width,
        )
    ):
        reports.append(
            _benchmark_case(
                case,
                device=device,
                seed=args.seed + case_idx * 17,
                warmup=args.warmup,
                replays=args.replays,
                l2_flush=l2_flush,
                verify=not args.skip_verify,
                num_q_heads=args.num_q_heads,
            )
        )
    return reports


def _render_report(report: CaseReport) -> str:
    indexed_page = report.case.indexed_page_size if report.case.indexed_page_size is not None else 0
    parts = [
        f"compressed-mla-native case={report.case.name:8s}",
        f"rows={report.case.rows:2d}",
        f"swa={report.case.swa_width:3d}",
        f"indexed={report.case.indexed_width:3d}",
        f"indexed_page={indexed_page:3d}",
        f"topk={report.case.topk:3d}",
        f"chunks={_planned_split_chunks(report.case):2d}",
        f"replay={report.replay_us:8.2f} us",
        f"p90={report.p90_replay_us:8.2f} us",
    ]
    if report.sanity_algorithm is not None:
        parts.append(
            "algorithm="
            f"max_abs:{report.sanity_algorithm.max_abs:.4f},"
            f"rmse:{report.sanity_algorithm.rmse:.5f},"
            f"cos:{report.sanity_algorithm.cos:.6f}"
        )
    return " | ".join(parts)


def _render_summary(reports: list[CaseReport], summary: TargetSummary) -> str:
    return " | ".join(
        [
            f"Summary | cases={len(reports)}",
            f"rows1_geo={summary.rows1_geo_us:.2f} us",
            f"rows1_target_ratio={summary.rows1_target_ratio:.4f}",
            f"rows4096_geo={summary.rows4096_geo_us:.2f} us",
            f"rows4096_target_ratio={summary.rows4096_target_ratio:.4f}",
            f"avg_target_ratio={summary.avg_target_ratio:.4f}",
        ]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="all",
        help="comma-separated cases: all,model,swa,c4,c128,swa-c4,swa-c128",
    )
    parser.add_argument("--rows", type=_parse_csv_ints, default=_parse_csv_ints("1,4096"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument(
        "--model-config",
        type=pathlib.Path,
        default=None,
        help=(
            "DeepSeek V4 config.json; with --cases model this derives the real "
            "SWA, SWA+C4, and SWA+C128 selected widths"
        ),
    )
    parser.add_argument(
        "--full-token-capacity",
        type=int,
        default=None,
        help=(
            "runtime full-token KV capacity used to derive C128 indexed width; "
            "matches the SGLang DSV4 pool log full=..."
        ),
    )
    parser.add_argument(
        "--c128-pool-size",
        type=int,
        default=None,
        help="optional runtime C128 pool size cap used by SGLang when deriving the C128 width",
    )
    parser.add_argument(
        "--c4-indexed-width",
        type=int,
        default=None,
        help="override indexed-token width for C4 cases; default comes from config or model top-k",
    )
    parser.add_argument(
        "--c128-indexed-width",
        type=int,
        default=None,
        help=(
            "override indexed-token width for C128 cases; by default this comes from "
            "config plus the runtime full-token/C128 pool capacity"
        ),
    )
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
        help="deprecated; compressed-layout algorithm verification is the default unless --skip-verify is set",
    )
    parser.add_argument(
        "--num-q-heads",
        type=int,
        default=_DEFAULT_NUM_Q_HEADS,
        help="local query-head count to benchmark; default is the synthetic 32-head profile",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.warmup <= 0 or args.replays <= 0:
        raise SystemExit("--warmup and --replays must be positive")
    if args.full_token_capacity is not None and args.full_token_capacity <= 0:
        raise SystemExit("--full-token-capacity must be positive")
    if args.c128_pool_size is not None and args.c128_pool_size <= 0:
        raise SystemExit("--c128-pool-size must be positive")
    if args.c4_indexed_width is not None and args.c4_indexed_width <= 0:
        raise SystemExit("--c4-indexed-width must be positive")
    if args.c128_indexed_width is not None and args.c128_indexed_width <= 0:
        raise SystemExit("--c128-indexed-width must be positive")
    if args.num_q_heads <= 0:
        raise SystemExit("--num-q-heads must be positive")
    try:
        _resolve_case_widths(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        reports = collect_case_reports(args)
    except BenchmarkFailure as exc:
        print(str(exc), file=sys.stderr)
        return 1

    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    flush_desc = f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per replay)" if args.flush_l2 else "off"
    print(f"L2 flush: {flush_desc}")
    for report in reports:
        print(_render_report(report))
    try:
        summary = _compute_target_summary(reports)
    except BenchmarkFailure as exc:
        print(f"Summary skipped: {exc}")
    else:
        print(_render_summary(reports, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
