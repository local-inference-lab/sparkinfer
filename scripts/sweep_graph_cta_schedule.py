#!/usr/bin/env python3
"""Sweep graph CTA budgets for paged attention under a shared capture bucket.

For each `graph_ctas_per_sm` candidate this script:

- captures one max-page CUDA graph for the chosen mode/family,
- replays smaller page counts under that candidate's own captured graph,
- races candidates in round-robin batches with CI-based elimination,
- reports tied surviving winners and a collapsed winner ladder.
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

from sparkinfer.attention.paged.planner import create_paged_plan
from sparkinfer.attention.paged.tuning.registry import normalize_kv_dtype_key
from sparkinfer.integration.attention import PagedAttentionWorkspace

from benchmarks.benchmark_paged_attention import (
    _bench_graph,
    _capture_graph,
    _mean_ci,
    _make_uniform_paged_inputs,
    _quantize_paged_kv_cache_global_e4m3,
    _resolve_kv_dtype,
    clear_attention_caches,
    require_sm120,
)
from benchmarks.sweep_search import SweepSegment


@dataclass
class CtaCandidateContext:
    graph_ctas_per_sm: int
    graph: torch.cuda.CUDAGraph
    workspace: PagedAttentionWorkspace
    output: torch.Tensor
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor


@dataclass
class CandidateState:
    context: CtaCandidateContext
    plan_cta_tile_q: int
    plan_chunk_pages: int
    plan_split: bool
    plan_new_batch_size: int
    plan_padded_batch_size: int
    samples_ms: list[float]

    @property
    def label(self) -> str:
        return f"ctas={self.context.graph_ctas_per_sm}"


@dataclass
class PageRaceResult:
    page_count: int
    cache_seqlen: int
    tied_winners: list[CandidateState]
    best_candidate: CandidateState
    best_mean_us: float
    best_ci_low_us: float
    best_ci_high_us: float


@dataclass(frozen=True)
class WinnerSummary:
    graph_ctas_per_sm: int
    plan_cta_tile_q: int
    plan_chunk_pages: int
    plan_split: bool
    plan_new_batch_size: int
    plan_padded_batch_size: int
    mean_us: float
    ci_low_us: float
    ci_high_us: float


_VERBOSE = False
_SUMMARY = False
_WORKER_JSON_VERSION = 1


def _log(message: str) -> None:
    if _VERBOSE:
        print(message, file=sys.stderr, flush=True)


def _log_summary(message: str) -> None:
    if _SUMMARY or _VERBOSE:
        print(message, file=sys.stderr, flush=True)


def _parse_candidate_ctas(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part]
    if not values:
        return list(range(1, 9))
    if len(values) == 2:
        lo, hi = sorted(values)
        if lo < hi:
            return list(range(lo, hi + 1))
    return sorted({value for value in values if value > 0})


def _parse_batch_list(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one batch size in --batch-list")
    batches = sorted({value for value in values if value > 0})
    if not batches:
        raise ValueError("expected positive batch sizes in --batch-list")
    return batches


def _collapse_page_winner_rows(
    rows: list[tuple[int, frozenset[int]]],
) -> list[SweepSegment]:
    if not rows:
        return []
    collapsed: list[SweepSegment] = []
    for page, winners in rows:
        chosen_winner = min(winners)
        if collapsed and collapsed[-1].winner == chosen_winner and collapsed[-1].end_page + 1 == page:
            collapsed[-1] = SweepSegment(
                winners=collapsed[-1].winners,
                start_page=collapsed[-1].start_page,
                end_page=page,
            )
        else:
            collapsed.append(
                SweepSegment(
                    winners=frozenset({chosen_winner}),
                    start_page=page,
                    end_page=page,
                )
            )
    return collapsed


def _collapsed_ladder_payload(*, rows: list[tuple[int, frozenset[int]]], page_size: int) -> list[dict[str, int]]:
    collapsed = _collapse_page_winner_rows(rows)
    return [
        {
            "start_page": int(segment.start_page),
            "end_page": int(segment.end_page),
            "start_cache_tokens": int(segment.start_page * page_size),
            "end_cache_tokens": int(segment.end_page * page_size),
            "winner_graph_ctas_per_sm": int(segment.winner),
        }
        for segment in collapsed
    ]


def _ladder_from_payload(payload: dict[str, object]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(segment["end_page"]),
            int(segment["winner_graph_ctas_per_sm"]),
        )
        for segment in payload["collapsed_ladder"]
    )


def _family_module_path(*, mode: str, kv_dtype: str) -> pathlib.Path:
    dtype_key = normalize_kv_dtype_key(kv_dtype)
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "sparkinfer"
        / "attention"
        / "paged"
        / "tuning"
        / f"cta_tuning_{dtype_key}_{mode}.py"
    )


def _print_batch_ladder(*, batch: int, payload: dict[str, object]) -> None:
    print()
    print(f"# batch={batch}")
    print("start_page\tend_page\tstart_cache_tokens\tend_cache_tokens\twinner_graph_ctas_per_sm")
    for segment in payload["collapsed_ladder"]:
        print(
            f"{segment['start_page']}\t{segment['end_page']}\t"
            f"{segment['start_cache_tokens']}\t{segment['end_cache_tokens']}\t"
            f"{segment['winner_graph_ctas_per_sm']}"
        )


def _render_python_module(
    *,
    args: argparse.Namespace,
    candidate_ctas_per_sm: list[int],
    batch_payloads: dict[int, dict[str, object]],
) -> str:
    dtype_key = normalize_kv_dtype_key(args.kv_dtype)
    batches = sorted(batch_payloads)
    lines = [
        '"""Generated by scripts/sweep_graph_cta_schedule.py."""',
        "",
        "from .registry import register_cta_ladder",
        "",
        f"KV_DTYPE = {dtype_key!r}",
        f"MODE = {args.mode!r}",
        f"BATCHES = {tuple(batches)!r}",
        f"CANDIDATE_CTAS_PER_SM = {tuple(candidate_ctas_per_sm)!r}",
        f"CAPTURE_PAGE_COUNT = {int(args.capture_page_count)!r}",
        f"PAGE_SIZE = {int(args.page_size)!r}",
        f"PAGE_RANGE = {(int(args.page_start), int(args.page_stop), int(args.page_step))!r}",
        "",
        "LADDERS = {",
    ]
    for batch in batches:
        lines.append(f"    {batch}: (")
        for end_page, ctas_per_sm in _ladder_from_payload(batch_payloads[batch]):
            lines.append(f"        ({end_page}, {ctas_per_sm}),")
        lines.append("    ),")
    lines.extend(
        [
            "}",
            "",
            "for _batch, _ladder in LADDERS.items():",
            "    register_cta_ladder(",
            "        kv_dtype=KV_DTYPE,",
            "        mode=MODE,",
            "        batch=_batch,",
            "        ladder=_ladder,",
            "    )",
            "",
            "__all__ = [",
            '    "BATCHES",',
            '    "CANDIDATE_CTAS_PER_SM",',
            '    "CAPTURE_PAGE_COUNT",',
            '    "KV_DTYPE",',
            '    "LADDERS",',
            '    "MODE",',
            '    "PAGE_RANGE",',
            '    "PAGE_SIZE",',
            "]",
        ]
    )
    return "\n".join(lines) + "\n"


def _winner_summary_from_candidate(
    candidate: CandidateState,
    *,
    ci_level: float,
) -> WinnerSummary:
    mean_us, ci_low_us, ci_high_us = _candidate_stats_us(candidate, ci_level=ci_level)
    return WinnerSummary(
        graph_ctas_per_sm=int(candidate.context.graph_ctas_per_sm),
        plan_cta_tile_q=int(candidate.plan_cta_tile_q),
        plan_chunk_pages=int(candidate.plan_chunk_pages),
        plan_split=bool(candidate.plan_split),
        plan_new_batch_size=int(candidate.plan_new_batch_size),
        plan_padded_batch_size=int(candidate.plan_padded_batch_size),
        mean_us=float(mean_us),
        ci_low_us=float(ci_low_us),
        ci_high_us=float(ci_high_us),
    )


def _page_counts_for_args(args: argparse.Namespace) -> list[int]:
    return list(range(args.page_start, args.page_stop + 1, args.page_step))


def _capture_candidate_context(
    *,
    mode: str,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    warmup: int,
    graph_ctas_per_sm: int,
    sweep_cache_seqlens: list[int],
) -> CtaCandidateContext:
    output = torch.empty_like(q)
    workspace = PagedAttentionWorkspace.for_tensors(
        mode=mode,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=False,
    )
    assert workspace._plan_q is not None
    assert workspace._plan_k_cache is not None
    assert workspace._plan_v_cache is not None
    active_total_q = int(cu_seqlens_q[-1].item())

    def graph_plan_for(runtime_cache_seqlens: torch.Tensor):
        return create_paged_plan(
            workspace._plan_q[:active_total_q],
            workspace._plan_k_cache,
            workspace._plan_v_cache,
            page_table,
            runtime_cache_seqlens,
            cu_seqlens_q,
            mode=mode,
            fixed_split_size=-1,
            disable_split_kv=False,
            enable_cuda_graph=True,
            graph_chunk_policy=True,
            graph_ctas_per_sm=graph_ctas_per_sm,
        )

    capture_cache_seqlen = int(cache_seqlens[0].item())
    capture_plan = graph_plan_for(cache_seqlens)
    workspace._ensure_capacity(capture_plan)
    for replay_cache_seqlen in sweep_cache_seqlens:
        cache_seqlens.fill_(int(replay_cache_seqlen))
        workspace._ensure_capacity(graph_plan_for(cache_seqlens))
    cache_seqlens.fill_(capture_cache_seqlen)
    workspace.use_cuda_graph = True
    workspace._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
    workspace._copy_plan_metadata(capture_plan)
    workspace._plan = capture_plan

    def run() -> None:
        workspace.run(
            q,
            k_cache,
            v_cache,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    graph = _capture_graph(run, warmup=warmup)
    return CtaCandidateContext(
        graph_ctas_per_sm=graph_ctas_per_sm,
        graph=graph,
        workspace=workspace,
        output=output,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )


def _prepare_candidate_for_page(
    *,
    context: CtaCandidateContext,
    mode: str,
    cache_seqlen: int,
) -> CandidateState:
    context.cache_seqlens.fill_(cache_seqlen)
    workspace = context.workspace
    assert workspace._plan_q is not None
    assert workspace._plan_k_cache is not None
    assert workspace._plan_v_cache is not None
    active_total_q = int(context.cu_seqlens_q[-1].item())
    replay_plan = create_paged_plan(
        workspace._plan_q[:active_total_q],
        workspace._plan_k_cache,
        workspace._plan_v_cache,
        context.page_table,
        context.cache_seqlens,
        context.cu_seqlens_q,
        mode=mode,
        fixed_split_size=-1,
        disable_split_kv=False,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=context.graph_ctas_per_sm,
    )
    workspace._ensure_capacity(replay_plan)
    workspace._copy_runtime_metadata(context.page_table, context.cache_seqlens, context.cu_seqlens_q)
    workspace._copy_plan_metadata(replay_plan)
    workspace._plan = replay_plan
    return CandidateState(
        context=context,
        plan_cta_tile_q=int(replay_plan.cta_tile_q),
        plan_chunk_pages=int(replay_plan.kv_chunk_size // replay_plan.page_size),
        plan_split=bool(replay_plan.split_kv),
        plan_new_batch_size=int(replay_plan.new_batch_size),
        plan_padded_batch_size=int(replay_plan.padded_batch_size),
        samples_ms=[],
    )


def _preferred_winner_summary(
    winners: list[WinnerSummary],
) -> WinnerSummary:
    if not winners:
        raise ValueError("expected at least one tied winner")
    return min(
        winners,
        key=lambda winner: (
            int(winner.graph_ctas_per_sm),
            int(winner.plan_chunk_pages),
            int(winner.plan_padded_batch_size),
        ),
    )


def _candidate_stats_us(
    candidate: CandidateState,
    *,
    ci_level: float,
) -> tuple[float, float, float]:
    ci_low_ms, ci_high_ms, _ = _mean_ci(candidate.samples_ms, ci_level=ci_level)
    return (
        statistics.fmean(candidate.samples_ms) * 1000.0,
        ci_low_ms * 1000.0,
        ci_high_ms * 1000.0,
    )


def _best_candidate(
    candidates: list[CandidateState],
    *,
    ci_level: float,
) -> tuple[CandidateState, float, float, float]:
    best = min(candidates, key=lambda candidate: statistics.fmean(candidate.samples_ms))
    mean_us, ci_low_us, ci_high_us = _candidate_stats_us(best, ci_level=ci_level)
    return best, mean_us, ci_low_us, ci_high_us


def _run_candidate_race(
    *,
    candidates: list[CandidateState],
    batch_replays: int,
    max_replays: int,
    ci_level: float,
    page_count: int,
) -> list[CandidateState]:
    active = list(range(len(candidates)))
    round_idx = 0
    while active:
        round_idx += 1
        _log(
            f"# page={page_count} round={round_idx} active="
            + ",".join(candidates[idx].label for idx in active)
        )
        for idx in list(active):
            candidate = candidates[idx]
            remaining = max_replays - len(candidate.samples_ms)
            if remaining <= 0:
                continue
            replays = min(batch_replays, remaining)
            candidate.samples_ms.extend(_bench_graph(candidate.context.graph, replays=replays))
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
        eliminated = False
        for idx in active:
            if idx == best_idx:
                continue
            mean_us, ci_low_us, _ci_high_us = stats[idx]
            if ci_low_us <= best_ci_high_us:
                next_active.append(idx)
            else:
                eliminated = True
                _log(
                    f"# page={page_count} round={round_idx} eliminate {candidates[idx].label} "
                    f"mean_us={mean_us:.3f} best={candidates[best_idx].label} "
                    f"best_mean_us={best_mean_us:.3f} best_ci_high_us={best_ci_high_us:.3f}"
                )
        if len(next_active) == len(active) and all(
            len(candidates[idx].samples_ms) >= max_replays for idx in active
        ):
            break
        if not eliminated and next_active == active:
            _log(f"# page={page_count} round={round_idx} no_eliminations")
        active = next_active
    return [candidates[idx] for idx in active]


def _worker_payload(*, page_payloads: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "version": _WORKER_JSON_VERSION,
            "pages": page_payloads,
        }
    )


def _build_capture_contexts(
    *,
    args: argparse.Namespace,
    candidate_ctas_per_sm: list[int],
    page_counts: list[int],
) -> list[CtaCandidateContext]:
    q_seqlen = 1 if args.mode == "decode" else args.q_seqlen
    cache_seqlen = args.capture_page_count * args.page_size
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
        seed=1000 + args.capture_page_count,
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
    contexts: list[CtaCandidateContext] = []
    sweep_cache_seqlens = [
        page_count * args.page_size
        for page_count in page_counts
    ]
    for graph_ctas_per_sm in candidate_ctas_per_sm:
        _log(f"# capture ctas_per_sm={graph_ctas_per_sm}")
        contexts.append(
            _capture_candidate_context(
                mode=args.mode,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                k_descale=k_descale,
                v_descale=v_descale,
                warmup=args.warmup,
                graph_ctas_per_sm=graph_ctas_per_sm,
                sweep_cache_seqlens=sweep_cache_seqlens,
            )
        )
    return contexts


def _measure_page_results(
    *,
    args: argparse.Namespace,
    candidate_ctas_per_sm: list[int],
    page_counts: list[int],
) -> list[PageRaceResult]:
    contexts = _build_capture_contexts(
        args=args,
        candidate_ctas_per_sm=candidate_ctas_per_sm,
        page_counts=page_counts,
    )
    page_results: list[PageRaceResult] = []
    for page_count in page_counts:
        result = _evaluate_page_race(page_count=page_count, args=args, contexts=contexts)
        page_results.append(result)
        _log_summary(
            f"# page={page_count} winners="
            f"{','.join(str(winner.context.graph_ctas_per_sm) for winner in result.tied_winners)}"
        )
    return page_results


def _run_batch_payload(
    *,
    args: argparse.Namespace,
    batch: int,
    candidate_ctas_per_sm: list[int],
) -> dict[str, object]:
    batch_args = argparse.Namespace(**vars(args))
    batch_args.batch = int(batch)
    if batch_args.parallel_workers != 1:
        page_payloads = _run_parallel_workers(
            args=batch_args,
            candidate_ctas_per_sm=candidate_ctas_per_sm,
        )
        return {
            "pages": page_payloads,
            "collapsed_ladder": _collapsed_ladder_payload(
                rows=[
                    (
                        int(page["page_count"]),
                        frozenset(
                            int(winner["graph_ctas_per_sm"])
                            for winner in page["tied_winners"]
                        ),
                    )
                    for page in page_payloads
                ],
                page_size=batch_args.page_size,
            ),
        }

    page_results = _measure_page_results(
        args=batch_args,
        candidate_ctas_per_sm=candidate_ctas_per_sm,
        page_counts=_page_counts_for_args(batch_args),
    )
    return _build_results_payload_from_page_results(
        page_results=page_results,
        page_size=batch_args.page_size,
        ci_level=batch_args.ci_level,
    )


def _worker_cli_args() -> list[str]:
    passthrough: list[str] = []
    skip_next = False
    skip_flags = {
        "--batch-list",
        "--output",
        "--parallel-workers",
        "--worker-graph-ctas-per-sm",
        "--worker-pages",
        "--worker-page-start",
        "--worker-page-stop",
    }
    for idx, arg in enumerate(sys.argv[1:]):
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
    worker_pages_arg = ",".join(str(page) for page in worker_pages)
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        *_worker_cli_args(),
        "--batch",
        str(args.batch),
        "--worker-pages",
        worker_pages_arg,
    ]
    _log(
        f"# launch_worker pages={worker_pages[0]}..{worker_pages[-1]} "
        f"count={len(worker_pages)} gpu={gpu_id} "
        f"cmd={' '.join(cmd)}"
    )
    return subprocess.Popen(
        cmd,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _partition_page_counts(
    *,
    page_counts: list[int],
    num_partitions: int,
) -> list[list[int]]:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    partitions = [[] for _ in range(num_partitions)]
    for idx, page_count in enumerate(page_counts):
        partitions[idx % num_partitions].append(page_count)
    return [partition for partition in partitions if partition]


def _run_parallel_workers(
    *,
    args: argparse.Namespace,
    candidate_ctas_per_sm: list[int],
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
            candidate_ctas_per_sm=candidate_ctas_per_sm,
            page_counts=page_counts,
        )
        return list(
            _build_results_payload_from_page_results(
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
        procs.append(
            (
                chunk,
                gpu_id,
                _launch_worker(
                    args=args,
                    gpu_id=gpu_id,
                    worker_pages=chunk,
                ),
            )
        )
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
                f"(count={len(worker_pages)}) on gpu={gpu_id}: "
                f"exit_code={proc.returncode}"
            )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"worker returned invalid JSON for pages={worker_pages[0]}..{worker_pages[-1]} "
                f"(count={len(worker_pages)}) on gpu={gpu_id}"
            ) from exc
        if int(payload.get("version", -1)) != _WORKER_JSON_VERSION:
            raise RuntimeError(
                f"worker returned unsupported version for pages={worker_pages[0]}..{worker_pages[-1]} "
                f"(count={len(worker_pages)}): "
                f"{payload.get('version')}"
            )
        all_pages.extend(payload.get("pages", []))
    all_pages.sort(key=lambda page: int(page["page_count"]))
    return all_pages


def _evaluate_page_race(
    *,
    page_count: int,
    args: argparse.Namespace,
    contexts: list[CtaCandidateContext],
) -> PageRaceResult:
    cache_seqlen = page_count * args.page_size
    candidates = [
        _prepare_candidate_for_page(
            context=context,
            mode=args.mode,
            cache_seqlen=cache_seqlen,
        )
        for context in contexts
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
        cache_seqlen=cache_seqlen,
        tied_winners=tied_winners,
        best_candidate=best_candidate,
        best_mean_us=best_mean_us,
        best_ci_low_us=best_ci_low_us,
        best_ci_high_us=best_ci_high_us,
    )


def _build_results_payload_from_page_results(
    *,
    page_results: list[PageRaceResult],
    page_size: int,
    ci_level: float,
) -> dict[str, object]:
    winner_rows: list[tuple[int, frozenset[int]]] = []
    payload_pages: list[dict[str, object]] = []
    for result in page_results:
        winners = frozenset(winner.context.graph_ctas_per_sm for winner in result.tied_winners)
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
        "pages": payload_pages,
        "collapsed_ladder": _collapsed_ladder_payload(rows=winner_rows, page_size=page_size),
    }


def _write_generated_module(
    *,
    args: argparse.Namespace,
    candidate_ctas_per_sm: list[int],
    batch_payloads: dict[int, dict[str, object]],
) -> pathlib.Path:
    output_path = pathlib.Path(args.output) if args.output else _family_module_path(
        mode=args.mode,
        kv_dtype=args.kv_dtype,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_python_module(
            args=args,
            candidate_ctas_per_sm=candidate_ctas_per_sm,
            batch_payloads=batch_payloads,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--batch-list", type=str, default="8")
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
    parser.add_argument("--candidate-ctas-per-sm", type=str, default="1,8")
    parser.add_argument("--mode", choices=["decode", "extend"], default="decode")
    parser.add_argument("--q-seqlen", type=int, default=6)
    parser.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp8_e4m3fn"], default="bf16")
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--batch", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--worker-pages", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-page-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-page-stop", type=int, default=0, help=argparse.SUPPRESS)
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
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be between 0 and 1")
    if args.capture_page_count <= 0:
        args.capture_page_count = args.page_stop
    if args.capture_page_count < args.page_stop:
        raise ValueError("--capture-page-count must be at least page-stop")

    require_sm120()
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    candidate_ctas_per_sm = _parse_candidate_ctas(args.candidate_ctas_per_sm)
    if not candidate_ctas_per_sm:
        raise ValueError("no positive CTA candidates to sweep")
    if args.parallel_workers < 0:
        raise ValueError("--parallel-workers must be non-negative")
    batch_list = _parse_batch_list(args.batch_list)

    if args.worker_pages or args.worker_page_start > 0 or args.worker_page_stop > 0:
        worker_args = argparse.Namespace(**vars(args))
        if args.worker_pages:
            worker_page_counts = [int(part) for part in args.worker_pages.split(",") if part.strip()]
            if not worker_page_counts or any(page <= 0 for page in worker_page_counts):
                raise ValueError("worker page list must contain positive page counts")
        else:
            if args.worker_page_start <= 0 or args.worker_page_stop < args.worker_page_start:
                raise ValueError("worker page slice must satisfy 1 <= worker-page-start <= worker-page-stop")
            worker_page_counts = list(range(int(args.worker_page_start), int(args.worker_page_stop) + 1, int(args.page_step)))
        worker_args.page_start = int(worker_page_counts[0])
        worker_args.page_stop = int(worker_page_counts[-1])
        page_results = _measure_page_results(
            args=worker_args,
            candidate_ctas_per_sm=candidate_ctas_per_sm,
            page_counts=worker_page_counts,
        )
        payload = _build_results_payload_from_page_results(
            page_results=page_results,
            page_size=args.page_size,
            ci_level=args.ci_level,
        )
        print(_worker_payload(page_payloads=list(payload["pages"])))
        return

    batch_payloads: dict[int, dict[str, object]] = {}
    for batch in batch_list:
        _log_summary(f"# batch={batch} start")
        payload = _run_batch_payload(
            args=args,
            batch=batch,
            candidate_ctas_per_sm=candidate_ctas_per_sm,
        )
        batch_payloads[int(batch)] = payload
        _print_batch_ladder(batch=batch, payload=payload)

    output_path = _write_generated_module(
        args=args,
        candidate_ctas_per_sm=candidate_ctas_per_sm,
        batch_payloads=batch_payloads,
    )
    print()
    print(f"# wrote {output_path}")


if __name__ == "__main__":
    main()
