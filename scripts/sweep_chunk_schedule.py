#!/usr/bin/env python3
"""Sweep fixed chunk-page candidates for paged attention under a shared capture bucket.

For each page count this script:

- captures one max-page CUDA graph/workspace per worker,
- replays fixed chunk-page candidates under that shared graph,
- races candidates in round-robin batches with CI-based elimination,
- picks the smallest surviving chunk-page candidate per page,
- run-length encodes the dense winner curve,
- emits a generated tuning module under ``sparkinfer.attention.paged.tuning``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
import pathlib
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from sparkinfer.integration.attention import PagedAttentionWorkspace
from sparkinfer.attention.paged.tuning.registry import normalize_kv_dtype_key

from benchmarks.benchmark_paged_attention import (
    _bench_graph,
    _capture_graph,
    _make_uniform_paged_inputs,
    _mean_ci,
    _quantize_paged_kv_cache_global_e4m3,
    _resolve_kv_dtype,
    clear_attention_caches,
    require_sm120,
)


@dataclass
class ChunkCandidate:
    fixed_split_pages: int
    graph: torch.cuda.CUDAGraph
    workspace: PagedAttentionWorkspace
    output: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    plan_cta_tile_q: int
    plan_chunk_pages: int
    plan_split: bool
    samples_ms: list[float]

    @property
    def label(self) -> str:
        return f"fixed={self.fixed_split_pages}"

    def prepare_replay(self) -> None:
        self.workspace.prepare_for_cuda_graph_replay(
            self.page_table,
            self.cache_seqlens,
            self.cu_seqlens_q,
            fixed_split_size=self.fixed_split_pages,
        )


@dataclass
class PageRaceResult:
    page_count: int
    cache_seqlen: int
    tied_winners: list[ChunkCandidate]
    best_candidate: ChunkCandidate
    best_mean_us: float
    best_ci_low_us: float
    best_ci_high_us: float


@dataclass(frozen=True)
class WinnerSummary:
    fixed_split_pages: int
    plan_cta_tile_q: int
    plan_chunk_pages: int
    plan_split: bool
    mean_us: float
    ci_low_us: float
    ci_high_us: float


@dataclass
class SweepCaptureContext:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    graph: torch.cuda.CUDAGraph
    workspace: PagedAttentionWorkspace
    output: torch.Tensor
    max_page_count: int


_VERBOSE = False
_SUMMARY = False
_WORKER_JSON_VERSION = 1


def _log(message: str) -> None:
    if _VERBOSE:
        print(message, file=sys.stderr, flush=True)


def _log_summary(message: str) -> None:
    if _SUMMARY or _VERBOSE:
        print(message, file=sys.stderr, flush=True)


def _parse_candidate_splits(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one candidate in --candidate-splits")
    if len(values) == 2:
        lo, hi = sorted(values)
        if lo < hi:
            return list(range(max(lo, 1), hi + 1))
    candidates = sorted({value for value in values if value > 0})
    if not candidates:
        raise ValueError("expected positive chunk-page candidates")
    return candidates


def _page_counts_for_args(args: argparse.Namespace) -> list[int]:
    return list(range(args.page_start, args.page_stop + 1, args.page_step))


def _collapse_smallest_winner_rows(
    rows: list[tuple[int, frozenset[int]]],
) -> tuple[tuple[int, int, int], ...]:
    if not rows:
        return ()
    collapsed: list[tuple[int, int, int]] = []
    current_start = rows[0][0]
    current_end = rows[0][0]
    current_winner = min(rows[0][1])
    for page, winners in rows[1:]:
        chosen = min(winners)
        if current_winner == chosen and current_end + 1 == page:
            current_end = page
        else:
            collapsed.append((current_start, current_end, current_winner))
            current_start = page
            current_end = page
            current_winner = chosen
    collapsed.append((current_start, current_end, current_winner))
    return tuple(collapsed)


def _collapsed_ladder_payload(*, rows: list[tuple[int, frozenset[int]]], page_size: int) -> list[dict[str, int]]:
    segments = _collapse_smallest_winner_rows(rows)
    payload: list[dict[str, int]] = []
    for start_page, end_page, winner in segments:
        payload.append(
            {
                "start_page": int(start_page),
                "end_page": int(end_page),
                "start_cache_tokens": int(start_page * page_size),
                "end_cache_tokens": int(end_page * page_size),
                "winner_fixed_split_pages": int(winner),
            }
        )
    return payload


def _family_module_path(*, mode: str, kv_dtype: str) -> pathlib.Path:
    dtype_key = normalize_kv_dtype_key(kv_dtype)
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "sparkinfer"
        / "attention"
        / "paged"
        / "tuning"
        / f"chunk_tuning_{dtype_key}_{mode}.py"
    )


def _policy_function_name(*, mode: str, kv_dtype: str) -> str:
    return f"{normalize_kv_dtype_key(kv_dtype)}_{mode}_chunk_pages"


def _print_ladder(payload: dict[str, object]) -> None:
    print()
    if "batch" in payload:
        print(f"# batch={payload['batch']}")
    print("start_page\tend_page\tstart_cache_tokens\tend_cache_tokens\twinner_fixed_split_pages")
    for segment in payload["collapsed_ladder"]:
        print(
            f"{segment['start_page']}\t{segment['end_page']}\t"
            f"{segment['start_cache_tokens']}\t{segment['end_cache_tokens']}\t"
            f"{segment['winner_fixed_split_pages']}"
        )


def _render_python_module(
    *,
    args: argparse.Namespace,
    candidate_splits: list[int],
    payload: dict[str, object],
) -> str:
    dtype_key = normalize_kv_dtype_key(args.kv_dtype)
    collapsed = list(payload["collapsed_ladder"])
    ladder = tuple(
        (
            int(segment["end_page"]),
            int(segment["winner_fixed_split_pages"]),
        )
        for segment in collapsed
    )
    fn_name = _policy_function_name(mode=args.mode, kv_dtype=args.kv_dtype)
    lines = [
        '"""Generated by scripts/sweep_chunk_schedule.py."""',
        "",
        "from .registry import register_chunk_ladder",
        "",
        "from bisect import bisect_left",
        "",
        f"KV_DTYPE = {dtype_key!r}",
        f"MODE = {args.mode!r}",
        f"BATCH = {int(args.batch)!r}",
        f"CANDIDATE_SPLITS = {tuple(candidate_splits)!r}",
        f"CAPTURE_PAGE_COUNT = {int(args.capture_page_count)!r}",
        f"PAGE_SIZE = {int(args.page_size)!r}",
        f"PAGE_RANGE = {(int(args.page_start), int(args.page_stop), int(args.page_step))!r}",
        "",
        f"LADDER = {ladder!r}",
        "UPPER_BOUNDS = tuple(end_page for end_page, _ in LADDER)",
        "WINNERS = tuple(chunk_pages for _, chunk_pages in LADDER)",
        "",
        f"def {fn_name}(page_count: int) -> int:",
        "    page_count = max(1, int(page_count))",
        "    if page_count <= UPPER_BOUNDS[-1]:",
        "        return WINNERS[bisect_left(UPPER_BOUNDS, page_count)]",
        "    return WINNERS[-1]",
        "",
        "register_chunk_ladder(",
        "    kv_dtype=KV_DTYPE,",
        "    mode=MODE,",
        "    ladder=LADDER,",
        ")",
        "",
            "__all__ = [",
            '    "BATCH",',
            '    "CANDIDATE_SPLITS",',
            '    "CAPTURE_PAGE_COUNT",',
            '    "KV_DTYPE",',
            '    "LADDER",',
        '    "MODE",',
        '    "PAGE_RANGE",',
        '    "PAGE_SIZE",',
        '    "UPPER_BOUNDS",',
        '    "WINNERS",',
        f'    "{fn_name}",',
        "]",
    ]
    return "\n".join(lines) + "\n"


def _capture_shared_graph(
    *,
    args: argparse.Namespace,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    fixed_split_pages: int,
) -> tuple[torch.cuda.CUDAGraph, PagedAttentionWorkspace, torch.Tensor]:
    output = torch.empty_like(q)
    workspace = PagedAttentionWorkspace.for_tensors(
        mode=args.mode,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=True,
    )
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=fixed_split_pages,
    )

    def run() -> None:
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    graph = _capture_graph(run, warmup=args.warmup)
    return graph, workspace, output


def _build_capture_context(
    *,
    args: argparse.Namespace,
    max_page_count: int,
) -> SweepCaptureContext:
    q_seqlen = 1 if args.mode == "decode" else args.q_seqlen
    cache_seqlen = max_page_count * args.page_size
    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        _capture_page_table,
        _capture_cache_seqlens,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=args.batch,
        q_seqlen=q_seqlen,
        cache_seqlen=cache_seqlen,
        capture_cache_seqlen=None,
        page_size=args.page_size,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype=torch.bfloat16,
        seed=1000 + max_page_count,
    )
    kv_dtype = _resolve_kv_dtype(args.kv_dtype, torch.bfloat16)
    k_descale = None
    v_descale = None
    if kv_dtype == torch.float8_e4m3fn:
        k_cache, v_cache, k_descale, v_descale, _, _ = _quantize_paged_kv_cache_global_e4m3(
            k_cache,
            v_cache,
            batch=args.batch,
            kv_heads=args.kv_heads,
        )
    graph, workspace, output = _capture_shared_graph(
        args=args,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=1,
    )
    return SweepCaptureContext(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        graph=graph,
        workspace=workspace,
        output=output,
        max_page_count=max_page_count,
    )


def _prepare_candidate(
    *,
    context: SweepCaptureContext,
    fixed_split_pages: int,
) -> ChunkCandidate:
    context.workspace.prepare_for_cuda_graph_replay(
        context.page_table,
        context.cache_seqlens,
        context.cu_seqlens_q,
        fixed_split_size=fixed_split_pages,
    )
    return ChunkCandidate(
        fixed_split_pages=fixed_split_pages,
        graph=context.graph,
        workspace=context.workspace,
        output=context.output,
        page_table=context.page_table,
        cache_seqlens=context.cache_seqlens,
        cu_seqlens_q=context.cu_seqlens_q,
        plan_cta_tile_q=int(context.workspace.plan.cta_tile_q),
        plan_chunk_pages=int(context.workspace.plan.kv_chunk_size // context.workspace.plan.page_size),
        plan_split=bool(context.workspace.plan.split_kv),
        samples_ms=[],
    )


def _candidate_stats_us(
    candidate: ChunkCandidate,
    *,
    ci_level: float,
) -> tuple[float, float, float]:
    ci_low_ms, ci_high_ms, _ = _mean_ci(candidate.samples_ms, ci_level=ci_level)
    return (
        statistics.fmean(candidate.samples_ms) * 1000.0,
        ci_low_ms * 1000.0,
        ci_high_ms * 1000.0,
    )


def _winner_summary_from_candidate(
    candidate: ChunkCandidate,
    *,
    ci_level: float,
) -> WinnerSummary:
    mean_us, ci_low_us, ci_high_us = _candidate_stats_us(candidate, ci_level=ci_level)
    return WinnerSummary(
        fixed_split_pages=int(candidate.fixed_split_pages),
        plan_cta_tile_q=int(candidate.plan_cta_tile_q),
        plan_chunk_pages=int(candidate.plan_chunk_pages),
        plan_split=bool(candidate.plan_split),
        mean_us=float(mean_us),
        ci_low_us=float(ci_low_us),
        ci_high_us=float(ci_high_us),
    )


def _preferred_winner_summary(winners: list[WinnerSummary]) -> WinnerSummary:
    return min(
        winners,
        key=lambda winner: (
            int(winner.fixed_split_pages),
            int(winner.plan_chunk_pages),
        ),
    )


def _best_candidate(
    candidates: list[ChunkCandidate],
    *,
    ci_level: float,
) -> tuple[ChunkCandidate, float, float, float]:
    best = min(candidates, key=lambda candidate: statistics.fmean(candidate.samples_ms))
    mean_us, ci_low_us, ci_high_us = _candidate_stats_us(best, ci_level=ci_level)
    return best, mean_us, ci_low_us, ci_high_us


def _run_candidate_race(
    *,
    candidates: list[ChunkCandidate],
    batch_replays: int,
    max_replays: int,
    ci_level: float,
    page_count: int,
) -> list[ChunkCandidate]:
    active = list(range(len(candidates)))
    while active:
        for idx in list(active):
            candidate = candidates[idx]
            remaining = max_replays - len(candidate.samples_ms)
            if remaining <= 0:
                continue
            replays = min(batch_replays, remaining)
            candidate.prepare_replay()
            candidate.samples_ms.extend(_bench_graph(candidate.graph, replays=replays))
        stats = {
            idx: _candidate_stats_us(candidates[idx], ci_level=ci_level)
            for idx in active
            if candidates[idx].samples_ms
        }
        if len(stats) <= 1:
            break
        best_idx = min(stats, key=lambda idx: stats[idx][0])
        best_mean_us, _best_ci_low_us, best_ci_high_us = stats[best_idx]
        next_active = [best_idx]
        for idx in active:
            if idx == best_idx:
                continue
            mean_us, ci_low_us, _ci_high_us = stats[idx]
            if ci_low_us <= best_ci_high_us:
                next_active.append(idx)
            else:
                _log(
                    f"# page={page_count} eliminate {candidates[idx].label} "
                    f"mean_us={mean_us:.3f} best={candidates[best_idx].label} "
                    f"best_mean_us={best_mean_us:.3f} best_ci_high_us={best_ci_high_us:.3f}"
                )
        if len(next_active) == len(active) and all(
            len(candidates[idx].samples_ms) >= max_replays for idx in active
        ):
            break
        active = next_active
    return [candidates[idx] for idx in sorted(active, key=lambda idx: _candidate_stats_us(candidates[idx], ci_level=ci_level)[0])]


def _evaluate_page_race(
    *,
    page_count: int,
    args: argparse.Namespace,
    candidate_splits: list[int],
    capture_context: SweepCaptureContext,
) -> PageRaceResult:
    if page_count > capture_context.max_page_count:
        raise ValueError(
            f"page_count {page_count} exceeds capture context max_page_count {capture_context.max_page_count}"
        )
    capture_context.cache_seqlens.fill_(page_count * args.page_size)
    candidates = [
        _prepare_candidate(context=capture_context, fixed_split_pages=fixed_split_pages)
        for fixed_split_pages in candidate_splits
    ]
    tied_winners = _run_candidate_race(
        candidates=candidates,
        batch_replays=args.probe_batch_replays,
        max_replays=args.replays,
        ci_level=args.ci_level,
        page_count=page_count,
    )
    best_candidate, best_mean_us, best_ci_low_us, best_ci_high_us = _best_candidate(
        tied_winners,
        ci_level=args.ci_level,
    )
    return PageRaceResult(
        page_count=page_count,
        cache_seqlen=page_count * args.page_size,
        tied_winners=tied_winners,
        best_candidate=best_candidate,
        best_mean_us=best_mean_us,
        best_ci_low_us=best_ci_low_us,
        best_ci_high_us=best_ci_high_us,
    )


def _measure_page_results(
    *,
    args: argparse.Namespace,
    candidate_splits: list[int],
    page_counts: list[int],
) -> list[PageRaceResult]:
    capture_context = _build_capture_context(args=args, max_page_count=args.capture_page_count)
    page_results: list[PageRaceResult] = []
    for page_count in page_counts:
        result = _evaluate_page_race(
            page_count=page_count,
            args=args,
            candidate_splits=candidate_splits,
            capture_context=capture_context,
        )
        page_results.append(result)
        _log_summary(
            f"# page={page_count} winners="
            f"{','.join(str(winner.fixed_split_pages) for winner in result.tied_winners)}"
        )
    return page_results


def _build_results_payload(
    *,
    batch: int,
    page_results: list[PageRaceResult],
    page_size: int,
    ci_level: float,
) -> dict[str, object]:
    winner_rows: list[tuple[int, frozenset[int]]] = []
    payload_pages: list[dict[str, object]] = []
    for result in page_results:
        winners = frozenset(winner.fixed_split_pages for winner in result.tied_winners)
        winner_rows.append((result.page_count, winners))
        tied_summaries = [
            _winner_summary_from_candidate(winner, ci_level=ci_level)
            for winner in result.tied_winners
        ]
        preferred_summary = _preferred_winner_summary(tied_summaries)
        payload_pages.append(
            {
                "page_count": int(result.page_count),
                "cache_seqlen": int(result.cache_seqlen),
                "winner_count": int(len(result.tied_winners)),
                "tied_winners": [asdict(summary) for summary in tied_summaries],
                "preferred_winner": asdict(preferred_summary),
            }
        )
    return {
        "batch": int(batch),
        "pages": payload_pages,
        "collapsed_ladder": _collapsed_ladder_payload(rows=winner_rows, page_size=page_size),
    }


def _worker_payload(*, page_payloads: list[dict[str, object]]) -> str:
    return json.dumps({"version": _WORKER_JSON_VERSION, "pages": page_payloads})


def _worker_cli_args() -> list[str]:
    passthrough: list[str] = []
    skip_next = False
    skip_flags = {
        "--output",
        "--parallel-workers",
        "--worker-pages",
    }
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in skip_flags:
            skip_next = True
            continue
        if any(arg.startswith(flag + "=") for flag in skip_flags):
            continue
        if arg in {"--verbose", "--summary"}:
            continue
        passthrough.append(arg)
    return passthrough


def _launch_worker(
    *,
    args: argparse.Namespace,
    gpu_id: int,
    worker_pages: list[int],
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        *_worker_cli_args(),
        "--worker-pages",
        ",".join(str(page) for page in worker_pages),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _partition_page_counts(*, page_counts: list[int], num_partitions: int) -> list[list[int]]:
    partitions = [[] for _ in range(num_partitions)]
    for idx, page_count in enumerate(page_counts):
        partitions[idx % num_partitions].append(page_count)
    return [partition for partition in partitions if partition]


def _run_parallel_workers(
    *,
    args: argparse.Namespace,
    candidate_splits: list[int],
) -> list[dict[str, object]]:
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        raise RuntimeError("parallel worker mode requires at least one visible CUDA device")
    page_counts = _page_counts_for_args(args)
    worker_count = args.parallel_workers
    if worker_count <= 0:
        worker_count = min(len(page_counts), visible_gpu_count)
    if worker_count <= 1:
        page_results = _measure_page_results(
            args=args,
            candidate_splits=candidate_splits,
            page_counts=page_counts,
        )
        return list(
            _build_results_payload(
                batch=args.batch,
                page_results=page_results,
                page_size=args.page_size,
                ci_level=args.ci_level,
            )["pages"]
        )

    worker_count = min(worker_count, len(page_counts), visible_gpu_count)
    gpu_ids = list(range(worker_count))
    page_chunks = _partition_page_counts(page_counts=page_counts, num_partitions=worker_count)
    procs: list[tuple[list[int], int, subprocess.Popen[str]]] = []
    for gpu_id, chunk in zip(gpu_ids, page_chunks):
        if not chunk:
            continue
        procs.append((chunk, gpu_id, _launch_worker(args=args, gpu_id=gpu_id, worker_pages=chunk)))
    all_pages: list[dict[str, object]] = []
    for worker_pages, gpu_id, proc in procs:
        stdout, stderr = proc.communicate()
        if stderr:
            sys.stderr.write(stderr)
            if not stderr.endswith("\n"):
                sys.stderr.write("\n")
        if proc.returncode != 0:
            raise RuntimeError(
                f"worker failed for pages={worker_pages[0]}..{worker_pages[-1]} "
                f"(count={len(worker_pages)}) on gpu={gpu_id}: exit_code={proc.returncode}"
            )
        payload = json.loads(stdout)
        if int(payload.get("version", -1)) != _WORKER_JSON_VERSION:
            raise RuntimeError(f"worker returned unsupported version: {payload.get('version')}")
        all_pages.extend(payload.get("pages", []))
    all_pages.sort(key=lambda page: int(page["page_count"]))
    return all_pages


def _write_generated_module(
    *,
    args: argparse.Namespace,
    candidate_splits: list[int],
    payload: dict[str, object],
) -> pathlib.Path:
    output_path = pathlib.Path(args.output) if args.output else _family_module_path(
        mode=args.mode,
        kv_dtype=args.kv_dtype,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_python_module(args=args, candidate_splits=candidate_splits, payload=payload),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--page-start", type=int, default=1)
    parser.add_argument("--page-stop", type=int, default=2048)
    parser.add_argument("--page-step", type=int, default=1)
    parser.add_argument("--capture-page-count", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--replays", type=int, default=500)
    parser.add_argument("--probe-batch-replays", type=int, default=50)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--candidate-splits", type=str, default="1,512")
    parser.add_argument("--mode", choices=["decode", "extend"], default="decode")
    parser.add_argument("--q-seqlen", type=int, default=6)
    parser.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp8_e4m3fn"], default="bf16")
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--worker-pages", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    global _VERBOSE, _SUMMARY
    _VERBOSE = bool(args.verbose)
    _SUMMARY = bool(args.summary)

    if args.page_start <= 0 or args.page_stop < args.page_start or args.page_step <= 0:
        raise ValueError("expected 1 <= page-start <= page-stop and page-step > 0")
    if args.page_size != 64:
        raise ValueError("primary paged backend expects page_size=64")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.q_seqlen <= 0:
        raise ValueError("--q-seqlen must be positive")
    if args.replays <= 0 or args.probe_batch_replays <= 0:
        raise ValueError("--replays and --probe-batch-replays must be positive")
    if args.probe_batch_replays > args.replays:
        raise ValueError("--probe-batch-replays must be <= --replays")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be between 0 and 1")
    if args.capture_page_count <= 0:
        args.capture_page_count = args.page_stop
    if args.capture_page_count < args.page_stop:
        raise ValueError("--capture-page-count must be at least page-stop")
    if args.parallel_workers < 0:
        raise ValueError("--parallel-workers must be non-negative")

    candidate_splits = _parse_candidate_splits(args.candidate_splits)

    require_sm120()
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if args.worker_pages:
        worker_page_counts = [int(part) for part in args.worker_pages.split(",") if part.strip()]
        if not worker_page_counts or any(page <= 0 for page in worker_page_counts):
            raise ValueError("worker page list must contain positive page counts")
        worker_args = argparse.Namespace(**vars(args))
        worker_args.page_start = int(worker_page_counts[0])
        worker_args.page_stop = int(worker_page_counts[-1])
        page_results = _measure_page_results(
            args=worker_args,
            candidate_splits=candidate_splits,
            page_counts=worker_page_counts,
        )
        payload = _build_results_payload(
            batch=args.batch,
            page_results=page_results,
            page_size=args.page_size,
            ci_level=args.ci_level,
        )
        print(_worker_payload(page_payloads=list(payload["pages"])))
        return

    if args.parallel_workers != 1:
        page_payloads = _run_parallel_workers(
            args=args,
            candidate_splits=candidate_splits,
        )
        payload = {
            "batch": int(args.batch),
            "pages": page_payloads,
            "collapsed_ladder": _collapsed_ladder_payload(
                rows=[
                    (
                        int(page["page_count"]),
                        frozenset(int(winner["fixed_split_pages"]) for winner in page["tied_winners"]),
                    )
                    for page in page_payloads
                ],
                page_size=args.page_size,
            ),
        }
    else:
        page_results = _measure_page_results(
            args=args,
            candidate_splits=candidate_splits,
            page_counts=_page_counts_for_args(args),
        )
        payload = _build_results_payload(
            batch=args.batch,
            page_results=page_results,
            page_size=args.page_size,
            ci_level=args.ci_level,
        )

    _print_ladder(payload)
    output_path = _write_generated_module(
        args=args,
        candidate_splits=candidate_splits,
        payload=payload,
    )
    print()
    print(f"# wrote {output_path}")


if __name__ == "__main__":
    main()
