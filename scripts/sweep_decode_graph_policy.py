#!/usr/bin/env python3
"""Sweep graph CTA and chunk policies and emit raw JSON measurements.

For each batch bucket this script:

- captures one shared max-page CUDA graph per CTA candidate,
- freezes that CTA for all replay pages in the bucket,
- races fixed chunk-page candidates under each frozen CTA,
- records the full per-page / per-CTA / per-chunk measurements to JSON,
- emits simple aggregate CTA summaries for downstream weighted search.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gc
import json
import math
import multiprocessing as mp
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from sparkinfer.attention.paged.planner import PagedPlan, create_paged_plan
from sparkinfer.attention.paged.tuning.registry import normalize_kv_dtype_key
from sparkinfer.integration.attention import PagedAttentionWorkspace

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
class CtaContext:
    graph_ctas_per_sm: int
    capture_fixed_split_pages: int
    graph: torch.cuda.CUDAGraph
    workspace: PagedAttentionWorkspace
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None
    replay_page_table_cpu: torch.Tensor
    replay_cache_seqlens_cpu: torch.Tensor
    replay_cu_seqlens_q_cpu: torch.Tensor


@dataclass(frozen=True)
class SharedPagedInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None
    replay_page_table_cpu: torch.Tensor
    replay_cache_seqlens_cpu: torch.Tensor
    replay_cu_seqlens_q_cpu: torch.Tensor


@dataclass
class ChunkCandidateState:
    fixed_split_pages: int
    context: CtaContext
    plan: PagedPlan
    plan_cta_tile_q: int
    plan_chunk_pages: int
    plan_split: bool
    plan_new_batch_size: int
    plan_padded_batch_size: int
    samples_ms: list[float]

    @property
    def label(self) -> str:
        return f"ctas={self.context.graph_ctas_per_sm}:fixed={self.fixed_split_pages}"

    def prepare_replay(self) -> None:
        workspace = self.context.workspace
        workspace._copy_runtime_metadata(
            self.context.replay_page_table_cpu,
            self.context.replay_cache_seqlens_cpu,
            self.context.replay_cu_seqlens_q_cpu,
        )
        workspace._copy_plan_metadata(self.plan)
        workspace._plan = self.plan


@dataclass(frozen=True)
class ChunkCandidateSummary:
    feasible: bool
    fixed_split_pages: int
    graph_ctas_per_sm: int
    plan_cta_tile_q: int | None
    plan_chunk_pages: int | None
    plan_split: bool | None
    plan_new_batch_size: int | None
    plan_padded_batch_size: int | None
    sample_count: int
    mean_us: float | None
    ci_low_us: float | None
    ci_high_us: float | None
    error: str | None = None


_VERBOSE = False
_SUMMARY = False
_WORKER_GPU_ID: int | None = None
_WORKER_CACHE_KEY: tuple[object, ...] | None = None
_WORKER_CONTEXTS: dict[int, CtaContext] | None = None
_WORKER_CAPTURE_ERRORS: dict[int, str] | None = None
_WORKER_CAPACITY_PROBE_OVERRIDE: tuple[int, ...] | None = None


def _log(message: str) -> None:
    if _VERBOSE:
        print(message, file=sys.stderr, flush=True)


def _log_summary(message: str) -> None:
    if _SUMMARY or _VERBOSE:
        print(message, file=sys.stderr, flush=True)


def _parse_candidate_ctas(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one candidate in --candidate-ctas-per-sm")
    if len(values) == 2:
        lo, hi = sorted(values)
        if lo < hi:
            return list(range(lo, hi + 1))
    candidates = sorted({value for value in values if value > 0})
    if not candidates:
        raise ValueError("expected positive CTA candidates")
    return candidates


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


def _parse_batch_list(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one batch size in --batch-list")
    batches = sorted({value for value in values if value > 0})
    if not batches:
        raise ValueError("expected positive batch sizes in --batch-list")
    return batches


def _page_counts_for_args(args: argparse.Namespace) -> list[int]:
    return list(range(args.page_start, args.page_stop + 1, args.page_step))


def _progressive_sample_page_counts(
    *,
    page_start: int,
    page_stop: int,
    sample_divisor: int,
) -> list[int]:
    if sample_divisor <= 0:
        raise ValueError("sample_divisor must be positive")
    pages = [int(page_start)]
    page = int(page_start)
    growth = 1.0 + (2.0 / float(sample_divisor))
    while page < page_stop:
        next_page = min(
            int(page_stop),
            max(page + 1, int(math.ceil(float(page) * growth))),
        )
        if next_page <= page:
            next_page = page + 1
        pages.append(int(next_page))
        page = int(next_page)
    return sorted(set(pages))


def _uniform_sample_page_counts(
    *,
    page_start: int,
    page_stop: int,
    stride: int,
) -> list[int]:
    if stride <= 0:
        return sorted({int(page_start), int(page_stop)})
    pages = {int(page_start), int(page_stop)}
    first_multiple = int(math.ceil(float(page_start) / float(stride)) * int(stride))
    page = max(int(stride), first_multiple)
    while page <= int(page_stop):
        pages.add(int(page))
        page += int(stride)
    return sorted(pages)


def _capacity_probe_page_counts(
    *,
    page_start: int,
    page_stop: int,
) -> list[int]:
    if page_stop < page_start:
        raise ValueError("expected page_stop >= page_start")
    span = int(page_stop - page_start)
    probes = set(
        _progressive_sample_page_counts(
            page_start=page_start,
            page_stop=page_stop,
            sample_divisor=6,
        )
    )
    probes.update(
        {
        int(page_start),
        int(page_stop),
        int((page_start + page_stop) // 2),
        }
    )
    if span > 0:
        probes.add(int(page_start + span // 4))
        probes.add(int(page_start + (3 * span) // 4))
    for page in (1, 2, 4, 8, 16, 32, 64):
        if page_start <= page <= page_stop:
            probes.add(int(page))
    return sorted(probe for probe in probes if page_start <= probe <= page_stop)


def _page_cta_result(page_payload: dict[str, object], graph_ctas_per_sm: int) -> dict[str, object]:
    return next(
        result
        for result in page_payload["cta_results"]
        if int(result["graph_ctas_per_sm"]) == int(graph_ctas_per_sm)
    )


def _page_cta_tied_winner_splits(page_payload: dict[str, object], graph_ctas_per_sm: int) -> list[int]:
    result = _page_cta_result(page_payload, graph_ctas_per_sm)
    tied = result.get("tied_chunk_winners")
    if not isinstance(tied, list):
        return []
    return sorted({int(summary["fixed_split_pages"]) for summary in tied})


def _page_cta_preferred_split(page_payload: dict[str, object], graph_ctas_per_sm: int) -> int | None:
    result = _page_cta_result(page_payload, graph_ctas_per_sm)
    preferred = result.get("preferred_chunk_winner")
    if not isinstance(preferred, dict):
        return None
    value = preferred.get("fixed_split_pages")
    return None if value is None else int(value)


def _page_cta_best_chunk_mean_us(page_payload: dict[str, object], graph_ctas_per_sm: int) -> float | None:
    result = _page_cta_result(page_payload, graph_ctas_per_sm)
    value = result["best_chunk_mean_us"]
    return None if value is None else float(value)


def _format_split_window(page_payload: dict[str, object]) -> str | None:
    searched = [int(value) for value in page_payload.get("searched_fixed_split_pages", [])]
    if not searched:
        return None
    if len(searched) == 1:
        return f"{searched[0]}"
    return f"{searched[0]}..{searched[-1]} n={len(searched)}"


def _relative_gap(best_value: float | None, other_value: float | None) -> float | None:
    if best_value is None or other_value is None:
        return None
    if best_value <= 0.0:
        return 0.0 if other_value <= 0.0 else float("inf")
    return max(0.0, float(other_value - best_value)) / float(best_value)


def _cta_refinement_pair(
    cta_scores: list[dict[str, object]],
) -> tuple[int, int, float | None] | None:
    if len(cta_scores) < 2:
        return None
    best = cta_scores[0]
    runner_up = cta_scores[1]
    if int(best["num_infeasible_pages"]) != int(runner_up["num_infeasible_pages"]):
        return None
    gap = _relative_gap(best["mean_best_chunk_us"], runner_up["mean_best_chunk_us"])
    if gap is None:
        return None
    return int(best["graph_ctas_per_sm"]), int(runner_up["graph_ctas_per_sm"]), float(gap)


def _cta_survivors(
    *,
    cta_scores: list[dict[str, object]],
    survivor_threshold: float,
    max_survivors: int,
) -> list[int]:
    if not cta_scores:
        return []
    best = cta_scores[0]
    best_mean = best["mean_best_chunk_us"]
    best_infeasible = int(best["num_infeasible_pages"])
    survivors = [int(best["graph_ctas_per_sm"])]
    for score in cta_scores[1:]:
        if len(survivors) >= max_survivors:
            break
        if int(score["num_infeasible_pages"]) != best_infeasible:
            continue
        gap = _relative_gap(best_mean, score["mean_best_chunk_us"])
        if gap is None:
            continue
        if gap <= survivor_threshold:
            survivors.append(int(score["graph_ctas_per_sm"]))
    return survivors


def _pair_winner(
    *,
    page_payload: dict[str, object],
    primary_cta: int,
    secondary_cta: int,
) -> int | None:
    primary = _page_cta_best_chunk_mean_us(page_payload, primary_cta)
    secondary = _page_cta_best_chunk_mean_us(page_payload, secondary_cta)
    if primary is None and secondary is None:
        return None
    if primary is None:
        return int(secondary_cta)
    if secondary is None:
        return int(primary_cta)
    return int(primary_cta) if primary <= secondary else int(secondary_cta)


def _refinement_pages_for_cta_pair(
    *,
    page_payloads_by_page: dict[int, dict[str, object]],
    sampled_page_counts: list[int],
    primary_cta: int,
    secondary_cta: int,
    close_threshold: float,
) -> list[int]:
    refine_pages: list[int] = []
    for left_page, right_page in zip(sampled_page_counts, sampled_page_counts[1:]):
        if right_page - left_page <= 1:
            continue
        left_payload = page_payloads_by_page[left_page]
        right_payload = page_payloads_by_page[right_page]
        left_primary = _page_cta_best_chunk_mean_us(left_payload, primary_cta)
        left_secondary = _page_cta_best_chunk_mean_us(left_payload, secondary_cta)
        right_primary = _page_cta_best_chunk_mean_us(right_payload, primary_cta)
        right_secondary = _page_cta_best_chunk_mean_us(right_payload, secondary_cta)
        left_gap = _relative_gap(
            min(value for value in (left_primary, left_secondary) if value is not None)
            if left_primary is not None or left_secondary is not None
            else None,
            max(value for value in (left_primary, left_secondary) if value is not None)
            if left_primary is not None or left_secondary is not None
            else None,
        )
        right_gap = _relative_gap(
            min(value for value in (right_primary, right_secondary) if value is not None)
            if right_primary is not None or right_secondary is not None
            else None,
            max(value for value in (right_primary, right_secondary) if value is not None)
            if right_primary is not None or right_secondary is not None
            else None,
        )
        left_winner = _pair_winner(
            page_payload=left_payload,
            primary_cta=primary_cta,
            secondary_cta=secondary_cta,
        )
        right_winner = _pair_winner(
            page_payload=right_payload,
            primary_cta=primary_cta,
            secondary_cta=secondary_cta,
        )
        feasibility_changed = (
            (left_primary is None) != (right_primary is None)
            or (left_secondary is None) != (right_secondary is None)
        )
        if (
            left_winner != right_winner
            or feasibility_changed
            or (left_gap is not None and left_gap <= close_threshold)
            or (right_gap is not None and right_gap <= close_threshold)
        ):
            refine_pages.append((left_page + right_page) // 2)
    return sorted(set(refine_pages))


def _render_output_payload(
    *,
    args: argparse.Namespace,
    batch_payloads: list[dict[str, object]],
    batch_list: list[int],
    candidate_ctas_per_sm: list[int],
    candidate_splits: list[int],
) -> dict[str, object]:
    return {
        "version": 1,
        "config": {
            "kv_dtype": normalize_kv_dtype_key(args.kv_dtype),
            "mode": str(args.mode),
            "q_seqlen": int(args.q_seqlen),
            "batch_list": batch_list,
            "page_range": [int(args.page_start), int(args.page_stop), int(args.page_step)],
            "capture_page_count": int(args.capture_page_count),
            "page_size": int(args.page_size),
            "q_heads": int(args.q_heads),
            "kv_heads": int(args.kv_heads),
            "head_dim": int(args.head_dim),
            "candidate_ctas_per_sm": candidate_ctas_per_sm,
            "candidate_splits": candidate_splits,
            "replays": int(args.replays),
            "probe_batch_replays": int(args.probe_batch_replays),
            "ci_level": float(args.ci_level),
            "cta_sample_divisor": int(args.cta_sample_divisor),
            "cta_close_threshold": float(args.cta_close_threshold),
            "cta_max_refinement_rounds": int(args.cta_max_refinement_rounds),
            "cta_survivor_threshold": float(args.cta_survivor_threshold),
            "cta_max_survivors": int(args.cta_max_survivors),
            "chunk_fill_windowed": bool(args.chunk_fill_windowed),
            "chunk_fill_window_probe_stride": int(args.chunk_fill_window_probe_stride),
            "chunk_fill_window_sample_divisor": int(args.chunk_fill_window_sample_divisor),
            "chunk_fill_window_relative_pad": float(args.chunk_fill_window_relative_pad),
            "chunk_fill_window_absolute_pad": int(args.chunk_fill_window_absolute_pad),
        },
        "batches": batch_payloads,
    }


def _write_json_atomic(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _checkpoint_output_path(output_path: pathlib.Path) -> pathlib.Path:
    if output_path.suffix:
        return output_path.with_name(output_path.name + ".checkpoint.jsonl")
    return output_path.with_name(output_path.name + ".checkpoint.jsonl")


def _worker_cache_key(
    *,
    args: argparse.Namespace,
    candidate_ctas_per_sm: list[int],
    candidate_splits: list[int],
    capacity_probe_page_counts: list[int],
) -> tuple[object, ...]:
    return (
        str(args.mode),
        int(args.q_seqlen),
        normalize_kv_dtype_key(args.kv_dtype),
        int(args.batch),
        int(args.capture_page_count),
        int(args.page_size),
        int(args.q_heads),
        int(args.kv_heads),
        int(args.head_dim),
        tuple(int(value) for value in candidate_ctas_per_sm),
        tuple(int(value) for value in candidate_splits),
        tuple(int(page) for page in capacity_probe_page_counts),
    )


def _is_graph_budget_infeasible(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "new_batch_size exceeds padded_batch_size" in str(exc)


def _is_workspace_capacity_exceeded(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "graph-mode paged workspace capacity exceeded" in str(exc)


def _collapsed_smallest_chunk_ladder(
    *,
    page_rows: list[tuple[int, frozenset[int]]],
    page_size: int,
) -> list[dict[str, int]]:
    if not page_rows:
        return []
    collapsed: list[dict[str, int]] = []
    current_start = page_rows[0][0]
    current_end = page_rows[0][0]
    current_winner = min(page_rows[0][1])
    for page, winners in page_rows[1:]:
        if current_winner in winners:
            chosen = int(current_winner)
        else:
            nondecreasing = [int(winner) for winner in winners if int(winner) >= int(current_winner)]
            chosen = min(nondecreasing) if nondecreasing else min(winners)
        if chosen < current_winner:
            chosen = int(current_winner)
        if chosen == current_winner and page == current_end + 1:
            current_end = page
            continue
        collapsed.append(
            {
                "start_page": int(current_start),
                "end_page": int(current_end),
                "start_cache_tokens": int(current_start * page_size),
                "end_cache_tokens": int(current_end * page_size),
                "winner_fixed_split_pages": int(current_winner),
            }
        )
        current_start = page
        current_end = page
        current_winner = chosen
    collapsed.append(
        {
            "start_page": int(current_start),
            "end_page": int(current_end),
            "start_cache_tokens": int(current_start * page_size),
            "end_cache_tokens": int(current_end * page_size),
            "winner_fixed_split_pages": int(current_winner),
        }
    )
    return collapsed


def _split_window_padding(*, value: int, relative_pad: float, absolute_pad: int) -> int:
    return max(int(absolute_pad), int(math.ceil(float(value) * float(relative_pad))))


def _split_window_without_padding(
    *,
    candidate_splits: list[int],
    winners: list[int],
) -> list[int]:
    if not winners:
        return [int(value) for value in candidate_splits]
    lo = min(int(value) for value in winners)
    hi = max(int(value) for value in winners)
    window = [
        int(value)
        for value in candidate_splits
        if int(lo) <= int(value) <= int(hi)
    ]
    return window if window else [int(value) for value in candidate_splits]


def _split_window_from_winners(
    *,
    candidate_splits: list[int],
    winners: list[int],
    relative_pad: float,
    absolute_pad: int,
) -> list[int]:
    if not winners:
        return [int(value) for value in candidate_splits]
    lo = min(int(value) for value in winners)
    hi = max(int(value) for value in winners)
    lower_bound = lo - _split_window_padding(
        value=lo,
        relative_pad=relative_pad,
        absolute_pad=absolute_pad,
    )
    upper_bound = hi + _split_window_padding(
        value=hi,
        relative_pad=relative_pad,
        absolute_pad=absolute_pad,
    )
    window = [
        int(value)
        for value in candidate_splits
        if int(lower_bound) <= int(value) <= int(upper_bound)
    ]
    return window if window else [int(value) for value in candidate_splits]


def _split_window_from_bounds(
    *,
    candidate_splits: list[int],
    lower_bound: int,
    upper_bound: int,
    relative_pad: float,
    absolute_pad: int,
) -> list[int]:
    lo = int(min(lower_bound, upper_bound))
    hi = int(max(lower_bound, upper_bound))
    return _split_window_from_winners(
        candidate_splits=candidate_splits,
        winners=[lo, hi],
        relative_pad=relative_pad,
        absolute_pad=absolute_pad,
    )


def _anchor_winner_window_for_page(
    *,
    page_count: int,
    anchor_pages: list[int],
    pages_by_page: dict[int, dict[str, object]],
    graph_ctas_per_sm: int,
    candidate_splits: list[int],
    relative_pad: float,
    absolute_pad: int,
) -> list[int]:
    if not anchor_pages:
        return [int(value) for value in candidate_splits]
    right_index = next(
        (idx for idx, anchor_page in enumerate(anchor_pages) if int(anchor_page) >= int(page_count)),
        len(anchor_pages) - 1,
    )
    left_index = max(0, right_index - 1)
    if int(anchor_pages[right_index]) < int(page_count):
        left_index = right_index
    winners: set[int] = set()
    for anchor_page in {int(anchor_pages[left_index]), int(anchor_pages[right_index])}:
        preferred = _page_cta_preferred_split(pages_by_page[anchor_page], graph_ctas_per_sm)
        if preferred is not None:
            winners.add(int(preferred))
            continue
        winners.update(_page_cta_tied_winner_splits(pages_by_page[anchor_page], graph_ctas_per_sm))
    return _split_window_without_padding(
        candidate_splits=candidate_splits,
        winners=sorted(winners),
    )


def _page_hits_split_window_edge(
    *,
    page_payload: dict[str, object],
    graph_ctas_per_sm: int,
    candidate_window: list[int],
    full_candidate_splits: list[int],
) -> bool:
    if not candidate_window or candidate_window == full_candidate_splits:
        return False
    winners = _page_cta_tied_winner_splits(page_payload, graph_ctas_per_sm)
    if not winners:
        return False
    return min(winners) <= int(candidate_window[0]) or max(winners) >= int(candidate_window[-1])


def _expanded_split_window(
    *,
    candidate_splits: list[int],
    current_window: list[int],
    winners: list[int],
    relative_pad: float,
    absolute_pad: int,
) -> list[int]:
    if not winners:
        return [int(value) for value in candidate_splits]
    lower_ref = int(current_window[0]) if min(winners) <= int(current_window[0]) else min(winners)
    upper_ref = int(current_window[-1]) if max(winners) >= int(current_window[-1]) else max(winners)
    lower_bound = lower_ref - _split_window_padding(
        value=lower_ref,
        relative_pad=relative_pad,
        absolute_pad=absolute_pad,
    )
    upper_bound = upper_ref + _split_window_padding(
        value=upper_ref,
        relative_pad=relative_pad,
        absolute_pad=absolute_pad,
    )
    expanded = [
        int(value)
        for value in candidate_splits
        if int(lower_bound) <= int(value) <= int(upper_bound)
    ]
    if not expanded:
        return [int(value) for value in candidate_splits]
    return expanded


def _union_split_windows(*windows: list[int]) -> list[int]:
    return sorted({int(value) for window in windows for value in window})


def _capture_cta_context(
    *,
    args: argparse.Namespace,
    graph_ctas_per_sm: int,
    candidate_splits: list[int],
    capacity_probe_page_counts: list[int],
    shared_inputs: SharedPagedInputs,
) -> CtaContext:
    _log_summary(
        f"# ctas={graph_ctas_per_sm} capture start capture_page_count={args.capture_page_count}"
    )
    q = shared_inputs.q
    k_cache = shared_inputs.k_cache
    v_cache = shared_inputs.v_cache
    k_descale = shared_inputs.k_descale
    v_descale = shared_inputs.v_descale

    workspace = PagedAttentionWorkspace.for_tensors(
        mode=str(args.mode),
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=False,
    )
    assert workspace._plan_q is not None
    assert workspace._plan_k_cache is not None
    assert workspace._plan_v_cache is not None
    replay_page_table_cpu = shared_inputs.replay_page_table_cpu
    replay_cache_seqlens_cpu = shared_inputs.replay_cache_seqlens_cpu
    replay_cu_seqlens_q_cpu = shared_inputs.replay_cu_seqlens_q_cpu
    active_total_q = int(replay_cu_seqlens_q_cpu[-1].item())
    batch = int(args.batch)
    max_pages_per_request = int(replay_page_table_cpu.shape[1])
    plan_page_table = replay_page_table_cpu.to(device=q.device, dtype=torch.int32).contiguous()
    plan_cache_seqlens = replay_cache_seqlens_cpu.to(device=q.device, dtype=torch.int32).contiguous()
    plan_cu_seqlens_q = replay_cu_seqlens_q_cpu.to(device=q.device, dtype=torch.int32).contiguous()

    def graph_plan_for(runtime_cache_seqlen: int, fixed_split_pages: int):
        plan_cache_seqlens.fill_(int(runtime_cache_seqlen))
        return create_paged_plan(
            workspace._plan_q[:active_total_q],
            workspace._plan_k_cache,
            workspace._plan_v_cache,
            plan_page_table[:batch, :max_pages_per_request],
            plan_cache_seqlens[:batch],
            plan_cu_seqlens_q[: batch + 1],
            mode=str(args.mode),
            fixed_split_size=int(fixed_split_pages),
            disable_split_kv=False,
            enable_cuda_graph=True,
            graph_chunk_policy=True,
            graph_ctas_per_sm=graph_ctas_per_sm,
        )

    capture_plan: PagedPlan | None = None
    capture_fixed_split_pages: int | None = None
    for fixed_split_pages in sorted(candidate_splits):
        try:
            capture_plan = graph_plan_for(args.capture_page_count * args.page_size, fixed_split_pages)
            capture_fixed_split_pages = int(fixed_split_pages)
            break
        except Exception as exc:
            if _is_graph_budget_infeasible(exc):
                continue
            raise
    if capture_plan is None or capture_fixed_split_pages is None:
        raise ValueError(
            f"no feasible capture split for graph_ctas_per_sm={graph_ctas_per_sm} "
            f"at capture_page_count={args.capture_page_count}"
        )
    workspace._ensure_capacity(capture_plan)
    for fixed_split_pages in candidate_splits:
        for page_count in capacity_probe_page_counts:
            try:
                workspace._ensure_capacity(graph_plan_for(int(page_count * args.page_size), fixed_split_pages))
            except Exception as exc:
                if _is_graph_budget_infeasible(exc):
                    continue
                raise

    workspace.use_cuda_graph = True
    workspace._copy_runtime_metadata(
        replay_page_table_cpu,
        replay_cache_seqlens_cpu,
        replay_cu_seqlens_q_cpu,
    )
    workspace._copy_plan_metadata(capture_plan)
    workspace._plan = capture_plan
    output = torch.empty_like(q)

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
    _log_summary(
        f"# ctas={graph_ctas_per_sm} capture ready seed_fixed={capture_fixed_split_pages} "
        f"padded_batch_size={capture_plan.padded_batch_size} new_batch_size={capture_plan.new_batch_size}"
    )
    return CtaContext(
        graph_ctas_per_sm=graph_ctas_per_sm,
        capture_fixed_split_pages=capture_fixed_split_pages,
        graph=graph,
        workspace=workspace,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        replay_page_table_cpu=replay_page_table_cpu,
        replay_cache_seqlens_cpu=replay_cache_seqlens_cpu,
        replay_cu_seqlens_q_cpu=replay_cu_seqlens_q_cpu,
    )


def _build_shared_paged_inputs(
    *,
    args: argparse.Namespace,
) -> SharedPagedInputs:
    cache_seqlen = int(args.capture_page_count * args.page_size)
    (
        q,
        k_cache,
        v_cache,
        replay_page_table,
        replay_cache_seqlens,
        _capture_page_table,
        _capture_cache_seqlens,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=args.batch,
        q_seqlen=int(args.q_seqlen),
        cache_seqlen=cache_seqlen,
        capture_cache_seqlen=None,
        page_size=args.page_size,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype=torch.bfloat16,
        seed=1000 + args.batch * 17,
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
    return SharedPagedInputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        replay_page_table_cpu=replay_page_table.detach().cpu().to(torch.int32).contiguous(),
        replay_cache_seqlens_cpu=replay_cache_seqlens.detach().cpu().to(torch.int32).contiguous(),
        replay_cu_seqlens_q_cpu=cu_seqlens_q.detach().cpu().to(torch.int32).contiguous(),
    )


def _prepare_chunk_candidate_for_page(
    *,
    mode: str,
    context: CtaContext,
    cache_seqlen: int,
    fixed_split_pages: int,
) -> ChunkCandidateState:
    context.replay_cache_seqlens_cpu.fill_(int(cache_seqlen))
    workspace = context.workspace
    assert workspace._plan_q is not None
    assert workspace._plan_k_cache is not None
    assert workspace._plan_v_cache is not None
    assert workspace.page_table is not None
    assert workspace.cache_seqlens is not None
    assert workspace.cu_seqlens_q is not None
    batch = int(context.replay_cache_seqlens_cpu.shape[0])
    max_pages_per_request = int(context.replay_page_table_cpu.shape[1])
    active_total_q = int(context.replay_cu_seqlens_q_cpu[-1].item())
    workspace._copy_runtime_metadata(
        context.replay_page_table_cpu,
        context.replay_cache_seqlens_cpu,
        context.replay_cu_seqlens_q_cpu,
    )
    replay_plan = create_paged_plan(
        workspace._plan_q[:active_total_q],
        workspace._plan_k_cache,
        workspace._plan_v_cache,
        workspace.page_table[:batch, :max_pages_per_request],
        workspace.cache_seqlens[:batch],
        workspace.cu_seqlens_q[: batch + 1],
        mode=str(mode),
        fixed_split_size=int(fixed_split_pages),
        disable_split_kv=False,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=context.graph_ctas_per_sm,
    )
    workspace._ensure_capacity(replay_plan)
    workspace._copy_plan_metadata(replay_plan)
    workspace._plan = replay_plan
    return ChunkCandidateState(
        fixed_split_pages=int(fixed_split_pages),
        context=context,
        plan=replay_plan,
        plan_cta_tile_q=int(replay_plan.cta_tile_q),
        plan_chunk_pages=int(replay_plan.kv_chunk_size // replay_plan.page_size),
        plan_split=bool(replay_plan.split_kv),
        plan_new_batch_size=int(replay_plan.new_batch_size),
        plan_padded_batch_size=int(replay_plan.padded_batch_size),
        samples_ms=[],
    )


def _candidate_stats_us(
    candidate: ChunkCandidateState,
    *,
    ci_level: float,
) -> tuple[float, float, float]:
    ci_low_ms, ci_high_ms, _ = _mean_ci(candidate.samples_ms, ci_level=ci_level)
    return (
        statistics.fmean(candidate.samples_ms) * 1000.0,
        ci_low_ms * 1000.0,
        ci_high_ms * 1000.0,
    )


def _candidate_summary(
    candidate: ChunkCandidateState,
    *,
    ci_level: float,
) -> ChunkCandidateSummary:
    mean_us, ci_low_us, ci_high_us = _candidate_stats_us(candidate, ci_level=ci_level)
    return ChunkCandidateSummary(
        feasible=True,
        fixed_split_pages=int(candidate.fixed_split_pages),
        graph_ctas_per_sm=int(candidate.context.graph_ctas_per_sm),
        plan_cta_tile_q=int(candidate.plan_cta_tile_q),
        plan_chunk_pages=int(candidate.plan_chunk_pages),
        plan_split=bool(candidate.plan_split),
        plan_new_batch_size=int(candidate.plan_new_batch_size),
        plan_padded_batch_size=int(candidate.plan_padded_batch_size),
        sample_count=int(len(candidate.samples_ms)),
        mean_us=float(mean_us),
        ci_low_us=float(ci_low_us),
        ci_high_us=float(ci_high_us),
        error=None,
    )


def _infeasible_candidate_summary(
    *,
    fixed_split_pages: int,
    graph_ctas_per_sm: int,
    error: str,
) -> ChunkCandidateSummary:
    return ChunkCandidateSummary(
        feasible=False,
        fixed_split_pages=int(fixed_split_pages),
        graph_ctas_per_sm=int(graph_ctas_per_sm),
        plan_cta_tile_q=None,
        plan_chunk_pages=None,
        plan_split=None,
        plan_new_batch_size=None,
        plan_padded_batch_size=None,
        sample_count=0,
        mean_us=None,
        ci_low_us=None,
        ci_high_us=None,
        error=str(error),
    )


def _run_chunk_candidate_race(
    *,
    candidates: list[ChunkCandidateState],
    batch_replays: int,
    max_replays: int,
    ci_level: float,
    page_count: int,
    graph_ctas_per_sm: int,
) -> list[ChunkCandidateState]:
    active = list(range(len(candidates)))
    while active:
        for idx in list(active):
            candidate = candidates[idx]
            remaining = max_replays - len(candidate.samples_ms)
            if remaining <= 0:
                continue
            replays = min(batch_replays, remaining)
            candidate.prepare_replay()
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
        for idx in active:
            if idx == best_idx:
                continue
            mean_us, ci_low_us, _ci_high_us = stats[idx]
            if ci_low_us <= best_ci_high_us:
                next_active.append(idx)
            else:
                _log(
                    f"# page={page_count} ctas={graph_ctas_per_sm} eliminate fixed={candidates[idx].fixed_split_pages} "
                    f"mean_us={mean_us:.3f} best_fixed={candidates[best_idx].fixed_split_pages} "
                    f"best_mean_us={best_mean_us:.3f} best_ci_high_us={best_ci_high_us:.3f}"
                )
        if len(next_active) == len(active) and all(
            len(candidates[idx].samples_ms) >= max_replays for idx in active
        ):
            break
        active = next_active
    return [candidates[idx] for idx in sorted(active, key=lambda idx: _candidate_stats_us(candidates[idx], ci_level=ci_level)[0])]


def _measure_page_for_cta(
    *,
    page_count: int,
    args: argparse.Namespace,
    context: CtaContext,
    candidate_splits: list[int],
) -> dict[str, object]:
    _log_summary(
        f"# page={page_count} ctas={context.graph_ctas_per_sm} chunk_sweep start "
        f"capture_seed={context.capture_fixed_split_pages}"
    )
    cache_seqlen = int(page_count * args.page_size)
    candidates: list[ChunkCandidateState] = []
    infeasible_summaries: list[ChunkCandidateSummary] = []
    for fixed_split_pages in candidate_splits:
        try:
            candidates.append(
                _prepare_chunk_candidate_for_page(
                    mode=str(args.mode),
                    context=context,
                    cache_seqlen=cache_seqlen,
                    fixed_split_pages=fixed_split_pages,
                )
            )
        except Exception as exc:
            if _is_graph_budget_infeasible(exc):
                infeasible_summaries.append(
                    _infeasible_candidate_summary(
                        fixed_split_pages=fixed_split_pages,
                        graph_ctas_per_sm=context.graph_ctas_per_sm,
                        error=str(exc),
                    )
                )
                continue
            raise
    if not candidates:
        all_summaries = sorted(
            infeasible_summaries,
            key=lambda summary: int(summary.fixed_split_pages),
        )
        _log_summary(
            f"# page={page_count} ctas={context.graph_ctas_per_sm} chunk_sweep done "
            f"preferred=NA feasible=0/{len(candidate_splits)}"
        )
        return {
            "graph_ctas_per_sm": int(context.graph_ctas_per_sm),
            "capture_fixed_split_pages": int(context.capture_fixed_split_pages),
            "capture_feasible": True,
            "capture_error": None,
            "best_chunk_mean_us": None,
            "best_chunk_ci_low_us": None,
            "best_chunk_ci_high_us": None,
            "preferred_chunk_winner": None,
            "tied_chunk_winners": [],
            "all_chunk_candidates": [asdict(summary) for summary in all_summaries],
        }
    tied_winners = _run_chunk_candidate_race(
        candidates=candidates,
        batch_replays=args.probe_batch_replays,
        max_replays=args.replays,
        ci_level=args.ci_level,
        page_count=page_count,
        graph_ctas_per_sm=context.graph_ctas_per_sm,
    )
    tied_summaries = [_candidate_summary(candidate, ci_level=args.ci_level) for candidate in tied_winners]
    all_summaries = [
        _candidate_summary(candidate, ci_level=args.ci_level)
        for candidate in sorted(candidates, key=lambda candidate: int(candidate.fixed_split_pages))
    ] + infeasible_summaries
    all_summaries.sort(key=lambda summary: int(summary.fixed_split_pages))
    best_summary = min(tied_summaries, key=lambda summary: float(summary.mean_us))
    preferred_summary = min(
        tied_summaries,
        key=lambda summary: (int(summary.fixed_split_pages), int(summary.plan_chunk_pages)),
    )
    _log_summary(
        f"# page={page_count} ctas={context.graph_ctas_per_sm} chunk_sweep done "
        f"preferred={preferred_summary.fixed_split_pages} tied={len(tied_summaries)} "
        f"feasible={len(candidates)}/{len(candidate_splits)} best_mean_us={best_summary.mean_us:.3f}"
    )
    return {
        "graph_ctas_per_sm": int(context.graph_ctas_per_sm),
        "capture_fixed_split_pages": int(context.capture_fixed_split_pages),
        "capture_feasible": True,
        "capture_error": None,
        "best_chunk_mean_us": float(best_summary.mean_us),
        "best_chunk_ci_low_us": float(best_summary.ci_low_us),
        "best_chunk_ci_high_us": float(best_summary.ci_high_us),
        "preferred_chunk_winner": asdict(preferred_summary),
        "tied_chunk_winners": [asdict(summary) for summary in tied_summaries],
        "all_chunk_candidates": [asdict(summary) for summary in all_summaries],
    }


def _capture_infeasible_cta_result(
    *,
    graph_ctas_per_sm: int,
    candidate_splits: list[int],
    error: str,
) -> dict[str, object]:
    all_summaries = [
        _infeasible_candidate_summary(
            fixed_split_pages=fixed_split_pages,
            graph_ctas_per_sm=graph_ctas_per_sm,
            error=error,
        )
        for fixed_split_pages in candidate_splits
    ]
    return {
        "graph_ctas_per_sm": int(graph_ctas_per_sm),
        "capture_fixed_split_pages": None,
        "capture_feasible": False,
        "capture_error": str(error),
        "best_chunk_mean_us": None,
        "best_chunk_ci_low_us": None,
        "best_chunk_ci_high_us": None,
        "preferred_chunk_winner": None,
        "tied_chunk_winners": [],
        "all_chunk_candidates": [asdict(summary) for summary in all_summaries],
    }


def _measure_single_page_payload(
    *,
    page_count: int,
    args: argparse.Namespace,
    candidate_ctas_per_sm: list[int],
    candidate_splits: list[int],
    contexts: dict[int, CtaContext],
    capture_errors: dict[int, str],
) -> dict[str, object]:
    cta_results = []
    for graph_ctas_per_sm in candidate_ctas_per_sm:
        context = contexts.get(graph_ctas_per_sm)
        if context is None:
            cta_results.append(
                _capture_infeasible_cta_result(
                    graph_ctas_per_sm=graph_ctas_per_sm,
                    candidate_splits=candidate_splits,
                    error=capture_errors[graph_ctas_per_sm],
                )
            )
            continue
        cta_results.append(
            _measure_page_for_cta(
                page_count=page_count,
                args=args,
                context=context,
                candidate_splits=candidate_splits,
            )
        )
    return {
        "page_count": int(page_count),
        "cache_seqlen": int(page_count * args.page_size),
        "searched_fixed_split_pages": [int(value) for value in candidate_splits],
        "cta_results": cta_results,
    }


def _measure_pages_payload(
    *,
    args: argparse.Namespace,
    candidate_ctas_per_sm: list[int],
    candidate_splits: list[int],
    page_counts: list[int],
    capacity_probe_page_counts: list[int],
    page_candidate_splits: dict[int, list[int]] | None = None,
    on_page_result: callable | None = None,
) -> tuple[dict[int, CtaContext], dict[int, str], list[dict[str, object]]]:
    contexts: dict[int, CtaContext] = {}
    capture_errors: dict[int, str] = {}
    for graph_ctas_per_sm in candidate_ctas_per_sm:
        try:
            contexts[graph_ctas_per_sm] = _capture_cta_context(
                args=args,
                graph_ctas_per_sm=graph_ctas_per_sm,
                candidate_splits=candidate_splits,
                capacity_probe_page_counts=capacity_probe_page_counts,
            )
        except Exception as exc:
            if _is_graph_budget_infeasible(exc) or "no feasible capture split" in str(exc):
                capture_errors[graph_ctas_per_sm] = str(exc)
                continue
            raise
    pages_payload: list[dict[str, object]] = []
    for page_count in page_counts:
        effective_candidate_splits = (
            [int(value) for value in candidate_splits]
            if page_candidate_splits is None
            else [int(value) for value in page_candidate_splits.get(int(page_count), candidate_splits)]
        )
        page_payload = _measure_single_page_payload(
            page_count=page_count,
            args=args,
            candidate_ctas_per_sm=candidate_ctas_per_sm,
            candidate_splits=effective_candidate_splits,
            contexts=contexts,
            capture_errors=capture_errors,
        )
        pages_payload.append(page_payload)
        if on_page_result is not None:
            on_page_result(page_payload)
        split_window = _format_split_window(page_payload)
        split_window_suffix = "" if split_window is None else f" split_window={split_window}"
        _log_summary(
            f"# page={page_count} cta_pref_chunks="
            + ",".join(
                f"{result['graph_ctas_per_sm']}:"
                f"{result['preferred_chunk_winner']['fixed_split_pages'] if result['preferred_chunk_winner'] is not None else 'NA'}"
                for result in page_payload["cta_results"]
            )
            + split_window_suffix
        )
    return contexts, capture_errors, pages_payload


def _build_batch_payload(
    *,
    args: argparse.Namespace,
    batch: int,
    page_payloads: list[dict[str, object]],
    candidate_ctas_per_sm: list[int],
) -> dict[str, object]:
    cta_scores: list[dict[str, object]] = []
    chunk_ladders: dict[str, list[dict[str, int]]] = {}
    capture_fixed_split_pages_by_cta: dict[str, int | None] = {}
    for graph_ctas_per_sm in candidate_ctas_per_sm:
        page_rows: list[tuple[int, frozenset[int]]] = []
        best_chunk_mean_us: list[float] = []
        num_infeasible_pages = 0
        first_cta_result: dict[str, object] | None = None
        for page in page_payloads:
            cta_result = next(
                result for result in page["cta_results"] if int(result["graph_ctas_per_sm"]) == graph_ctas_per_sm
            )
            if first_cta_result is None:
                first_cta_result = cta_result
            if cta_result["preferred_chunk_winner"] is None:
                num_infeasible_pages += 1
                continue
            winners = frozenset(int(summary["fixed_split_pages"]) for summary in cta_result["tied_chunk_winners"])
            page_rows.append((int(page["page_count"]), winners))
            best_chunk_mean_us.append(float(cta_result["best_chunk_mean_us"]))
        capture_fixed_split_pages_by_cta[str(graph_ctas_per_sm)] = (
            None if first_cta_result is None else first_cta_result["capture_fixed_split_pages"]
        )
        chunk_ladders[str(graph_ctas_per_sm)] = _collapsed_smallest_chunk_ladder(
            page_rows=page_rows,
            page_size=args.page_size,
        )
        cta_scores.append(
            {
                "graph_ctas_per_sm": int(graph_ctas_per_sm),
                "capture_fixed_split_pages": capture_fixed_split_pages_by_cta[str(graph_ctas_per_sm)],
                "mean_best_chunk_us": float(statistics.fmean(best_chunk_mean_us)) if best_chunk_mean_us else None,
                "max_best_chunk_us": float(max(best_chunk_mean_us)) if best_chunk_mean_us else None,
                "num_feasible_pages": int(len(best_chunk_mean_us)),
                "num_infeasible_pages": int(num_infeasible_pages),
            }
        )
    cta_scores.sort(
        key=lambda score: (
            int(score["num_infeasible_pages"]) > 0,
            float("inf") if score["mean_best_chunk_us"] is None else float(score["mean_best_chunk_us"]),
            int(score["graph_ctas_per_sm"]),
        )
    )
    best_cta = int(cta_scores[0]["graph_ctas_per_sm"])
    return {
        "batch": int(batch),
        "capture_page_count": int(args.capture_page_count),
        "candidate_ctas_per_sm": [int(value) for value in candidate_ctas_per_sm],
        "capture_fixed_split_pages_by_cta": capture_fixed_split_pages_by_cta,
        "pages": page_payloads,
        "aggregate_cta_scores": cta_scores,
        "best_cta_by_mean_best_chunk_us": best_cta,
        "best_cta_ladder": [
            {
                "start_page": int(args.page_start),
                "end_page": int(args.page_stop),
                "start_cache_tokens": int(args.page_start * args.page_size),
                "end_cache_tokens": int(args.page_stop * args.page_size),
                "winner_graph_ctas_per_sm": int(best_cta),
            }
        ],
        "chunk_ladders_by_cta": chunk_ladders,
        "best_cta_chunk_ladder": chunk_ladders[str(best_cta)],
    }


def _build_fixed_cta_search_payload(
    *,
    args: argparse.Namespace,
    batch: int,
    best_cta: int,
    chunk_fill_payload: dict[str, object],
) -> dict[str, object]:
    capture_fixed_split_pages = chunk_fill_payload["capture_fixed_split_pages_by_cta"][str(best_cta)]
    chunk_ladder = chunk_fill_payload["best_cta_chunk_ladder"]
    return {
        "batch": int(batch),
        "capture_page_count": int(args.capture_page_count),
        "candidate_ctas_per_sm": [int(best_cta)],
        "capture_fixed_split_pages_by_cta": {str(best_cta): capture_fixed_split_pages},
        "pages": [],
        "aggregate_cta_scores": [
            {
                "graph_ctas_per_sm": int(best_cta),
                "capture_fixed_split_pages": capture_fixed_split_pages,
                "mean_best_chunk_us": None,
                "max_best_chunk_us": None,
                "num_feasible_pages": 0,
                "num_infeasible_pages": 0,
            }
        ],
        "best_cta_by_mean_best_chunk_us": int(best_cta),
        "best_cta_ladder": [
            {
                "start_page": int(args.page_start),
                "end_page": int(args.page_stop),
                "start_cache_tokens": int(args.page_start * args.page_size),
                "end_cache_tokens": int(args.page_stop * args.page_size),
                "winner_graph_ctas_per_sm": int(best_cta),
            }
        ],
        "chunk_ladders_by_cta": {str(best_cta): chunk_ladder},
        "best_cta_chunk_ladder": chunk_ladder,
    }


def _single_cta_page_payload(
    *,
    page_payload: dict[str, object],
    graph_ctas_per_sm: int,
) -> dict[str, object]:
    return {
        "page_count": int(page_payload["page_count"]),
        "cache_seqlen": int(page_payload["cache_seqlen"]),
        "searched_fixed_split_pages": [int(value) for value in page_payload.get("searched_fixed_split_pages", [])],
        "cta_results": [_page_cta_result(page_payload, graph_ctas_per_sm)],
    }


def _build_two_stage_batch_payload(
    *,
    args: argparse.Namespace,
    batch: int,
    candidate_ctas_per_sm: list[int],
    cta_search_payload: dict[str, object],
    cta_search_sampled_page_counts: list[int],
    cta_search_rounds: list[dict[str, object]],
    chunk_fill_payload: dict[str, object] | None,
) -> dict[str, object]:
    best_cta = int(cta_search_payload["best_cta_by_mean_best_chunk_us"])
    chunk_fill_pages = [] if chunk_fill_payload is None else chunk_fill_payload["pages"]
    chunk_fill_ladder = (
        cta_search_payload["chunk_ladders_by_cta"][str(best_cta)]
        if chunk_fill_payload is None
        else chunk_fill_payload["best_cta_chunk_ladder"]
    )
    capture_fixed_split_pages = cta_search_payload["capture_fixed_split_pages_by_cta"][str(best_cta)]
    return {
        "batch": int(batch),
        "capture_page_count": int(args.capture_page_count),
        "candidate_ctas_per_sm": [int(value) for value in candidate_ctas_per_sm],
        "capture_fixed_split_pages_by_cta": cta_search_payload["capture_fixed_split_pages_by_cta"],
        "cta_search": {
            "sampled_page_counts": [int(page) for page in cta_search_sampled_page_counts],
            "rounds": cta_search_rounds,
            "pages": cta_search_payload["pages"],
            "aggregate_cta_scores": cta_search_payload["aggregate_cta_scores"],
            "best_cta_by_mean_best_chunk_us": int(best_cta),
            "capture_fixed_split_pages_by_cta": cta_search_payload["capture_fixed_split_pages_by_cta"],
            "chunk_ladders_by_cta": cta_search_payload["chunk_ladders_by_cta"],
        },
        "chunk_fill": {
            "graph_ctas_per_sm": int(best_cta),
            "capture_fixed_split_pages": capture_fixed_split_pages,
            "pages": chunk_fill_pages,
            "chunk_ladder": chunk_fill_ladder,
        },
        "aggregate_cta_scores": cta_search_payload["aggregate_cta_scores"],
        "best_cta_by_mean_best_chunk_us": int(best_cta),
        "best_cta_ladder": [
            {
                "start_page": int(args.page_start),
                "end_page": int(args.page_stop),
                "start_cache_tokens": int(args.page_start * args.page_size),
                "end_cache_tokens": int(args.page_stop * args.page_size),
                "winner_graph_ctas_per_sm": int(best_cta),
            }
        ],
        "best_cta_chunk_ladder": chunk_fill_ladder,
    }


def _summarize_page_completion(*, page: dict[str, object], phase: str) -> None:
    cta_pref = ",".join(
        f"{result['graph_ctas_per_sm']}:"
        f"{result['preferred_chunk_winner']['fixed_split_pages'] if result['preferred_chunk_winner'] is not None else 'NA'}"
        for result in page["cta_results"]
    )
    split_window = _format_split_window(page)
    split_window_suffix = "" if split_window is None else f" split_window={split_window}"
    _log_summary(f"# phase={phase} page={page['page_count']} cta_pref_chunks={cta_pref}{split_window_suffix}")


def _reset_worker_cache(*, clear_capacity_probe_override: bool = True) -> None:
    global _WORKER_CACHE_KEY, _WORKER_CONTEXTS, _WORKER_CAPTURE_ERRORS, _WORKER_CAPACITY_PROBE_OVERRIDE
    _WORKER_CACHE_KEY = None
    _WORKER_CONTEXTS = None
    _WORKER_CAPTURE_ERRORS = None
    if clear_capacity_probe_override:
        _WORKER_CAPACITY_PROBE_OVERRIDE = None
    clear_attention_caches()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _init_pool_worker(gpu_queue: object) -> None:
    global _SUMMARY, _VERBOSE, _WORKER_GPU_ID
    _SUMMARY = False
    _VERBOSE = False
    gpu_id = int(gpu_queue.get())
    _WORKER_GPU_ID = gpu_id
    torch.cuda.set_device(gpu_id)
    _reset_worker_cache()


def _build_worker_cache(
    *,
    args: argparse.Namespace,
    cache_candidate_ctas_per_sm: list[int],
    candidate_splits: list[int],
    capacity_probe_page_counts: list[int],
) -> None:
    global _WORKER_CACHE_KEY, _WORKER_CONTEXTS, _WORKER_CAPTURE_ERRORS
    contexts: dict[int, CtaContext] = {}
    capture_errors: dict[int, str] = {}
    shared_inputs = _build_shared_paged_inputs(args=args)
    for graph_ctas_per_sm in cache_candidate_ctas_per_sm:
        try:
            contexts[graph_ctas_per_sm] = _capture_cta_context(
                args=args,
                graph_ctas_per_sm=graph_ctas_per_sm,
                candidate_splits=candidate_splits,
                capacity_probe_page_counts=capacity_probe_page_counts,
                shared_inputs=shared_inputs,
            )
        except Exception as exc:
            if _is_graph_budget_infeasible(exc) or "no feasible capture split" in str(exc):
                capture_errors[graph_ctas_per_sm] = str(exc)
                continue
            raise
    _WORKER_CACHE_KEY = _worker_cache_key(
        args=args,
        candidate_ctas_per_sm=cache_candidate_ctas_per_sm,
        candidate_splits=candidate_splits,
        capacity_probe_page_counts=capacity_probe_page_counts,
    )
    _WORKER_CONTEXTS = contexts
    _WORKER_CAPTURE_ERRORS = capture_errors


def _worker_measure_page(task: dict[str, object]) -> dict[str, object]:
    global _WORKER_CACHE_KEY, _WORKER_CONTEXTS, _WORKER_CAPTURE_ERRORS, _WORKER_CAPACITY_PROBE_OVERRIDE
    args = argparse.Namespace(**task["args"])
    candidate_ctas_per_sm = [int(value) for value in task["candidate_ctas_per_sm"]]
    cache_candidate_ctas_per_sm = [int(value) for value in task["cache_candidate_ctas_per_sm"]]
    candidate_splits = [int(value) for value in task["candidate_splits"]]
    cache_candidate_splits = [int(value) for value in task.get("cache_candidate_splits", task["candidate_splits"])]
    page_count = int(task["page_count"])
    requested_capacity_probe_page_counts = [int(value) for value in task["capacity_probe_page_counts"]]
    effective_capacity_probe_page_counts = sorted(
        {
            *requested_capacity_probe_page_counts,
            *([] if _WORKER_CAPACITY_PROBE_OVERRIDE is None else _WORKER_CAPACITY_PROBE_OVERRIDE),
        }
    )
    cache_key = _worker_cache_key(
        args=args,
        candidate_ctas_per_sm=cache_candidate_ctas_per_sm,
        candidate_splits=cache_candidate_splits,
        capacity_probe_page_counts=effective_capacity_probe_page_counts,
    )
    if cache_key != _WORKER_CACHE_KEY or _WORKER_CONTEXTS is None or _WORKER_CAPTURE_ERRORS is None:
        _reset_worker_cache()
        _WORKER_CAPACITY_PROBE_OVERRIDE = tuple(int(page) for page in effective_capacity_probe_page_counts)
        _build_worker_cache(
            args=args,
            cache_candidate_ctas_per_sm=cache_candidate_ctas_per_sm,
            candidate_splits=cache_candidate_splits,
            capacity_probe_page_counts=effective_capacity_probe_page_counts,
        )
    for _attempt in range(2):
        assert _WORKER_CONTEXTS is not None
        assert _WORKER_CAPTURE_ERRORS is not None
        try:
            page = _measure_single_page_payload(
                page_count=page_count,
                args=args,
                candidate_ctas_per_sm=candidate_ctas_per_sm,
                candidate_splits=candidate_splits,
                contexts=_WORKER_CONTEXTS,
                capture_errors=_WORKER_CAPTURE_ERRORS,
            )
            break
        except Exception as exc:
            if not _is_workspace_capacity_exceeded(exc):
                raise
            effective_capacity_probe_page_counts = sorted({*effective_capacity_probe_page_counts, int(page_count)})
            _WORKER_CAPACITY_PROBE_OVERRIDE = tuple(int(page) for page in effective_capacity_probe_page_counts)
            _reset_worker_cache(clear_capacity_probe_override=False)
            _build_worker_cache(
                args=args,
                cache_candidate_ctas_per_sm=cache_candidate_ctas_per_sm,
                candidate_splits=cache_candidate_splits,
                capacity_probe_page_counts=effective_capacity_probe_page_counts,
            )
    else:
        raise RuntimeError(f"failed to recover worker capacity sizing for page_count={page_count}")
    return {
        "page": page,
        "page_count": int(page_count),
        "gpu_id": _WORKER_GPU_ID,
    }


def _run_parallel_workers(
    *,
    args: argparse.Namespace,
    candidate_ctas_per_sm: list[int],
    candidate_splits: list[int],
    page_counts: list[int],
    cache_candidate_ctas_per_sm: list[int] | None = None,
    cache_candidate_splits: list[int] | None = None,
    capacity_probe_page_counts: list[int] | None = None,
    page_candidate_splits: dict[int, list[int]] | None = None,
    executor: ProcessPoolExecutor | None = None,
    phase_label: str,
    on_page_result: callable | None = None,
) -> list[dict[str, object]]:
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        raise RuntimeError("parallel worker mode requires at least one visible CUDA device")
    worker_count = args.parallel_workers
    if worker_count <= 0:
        worker_count = min(len(page_counts), visible_gpu_count)
    if worker_count <= 1:
        capacity_probe_page_counts = (
            [int(value) for value in page_counts]
            if capacity_probe_page_counts is None
            else [int(value) for value in capacity_probe_page_counts]
        )
        _contexts, _capture_errors, pages = _measure_pages_payload(
            args=args,
            candidate_ctas_per_sm=candidate_ctas_per_sm,
            candidate_splits=candidate_splits,
            page_counts=page_counts,
            capacity_probe_page_counts=capacity_probe_page_counts,
            page_candidate_splits=page_candidate_splits,
            on_page_result=on_page_result,
        )
        return pages

    if executor is None:
        raise ValueError("parallel executor is required when parallel_workers > 1")
    cache_candidate_ctas_per_sm = (
        [int(value) for value in candidate_ctas_per_sm]
        if cache_candidate_ctas_per_sm is None
        else [int(value) for value in cache_candidate_ctas_per_sm]
    )
    cache_candidate_splits = (
        [int(value) for value in candidate_splits]
        if cache_candidate_splits is None
        else [int(value) for value in cache_candidate_splits]
    )
    capacity_probe_page_counts = (
        [int(value) for value in page_counts]
        if capacity_probe_page_counts is None
        else [int(value) for value in capacity_probe_page_counts]
    )
    all_pages: list[dict[str, object]] = []
    futures = {
        executor.submit(
            _worker_measure_page,
            {
                "args": {**vars(args)},
                "candidate_ctas_per_sm": [int(value) for value in candidate_ctas_per_sm],
                "cache_candidate_ctas_per_sm": [int(value) for value in cache_candidate_ctas_per_sm],
                "candidate_splits": [
                    int(value)
                    for value in (
                        candidate_splits
                        if page_candidate_splits is None
                        else page_candidate_splits.get(int(page_count), candidate_splits)
                    )
                ],
                "cache_candidate_splits": [int(value) for value in cache_candidate_splits],
                "capacity_probe_page_counts": [int(value) for value in capacity_probe_page_counts],
                "page_count": int(page_count),
            },
        ): int(page_count)
        for page_count in page_counts
    }
    for future in as_completed(futures):
        payload = future.result()
        page = payload["page"]
        all_pages.append(page)
        if on_page_result is not None:
            on_page_result(page)
        _summarize_page_completion(page=page, phase=phase_label)
    all_pages.sort(key=lambda page: int(page["page_count"]))
    return all_pages


def _run_windowed_chunk_fill(
    *,
    args: argparse.Namespace,
    batch: int,
    best_cta: int,
    all_page_counts: list[int],
    dense_pages_by_page: dict[int, dict[str, object]],
    candidate_splits: list[int],
    capacity_probe_page_counts: list[int],
    worker_count: int,
    executor: ProcessPoolExecutor | None,
    on_page_result: callable | None = None,
) -> None:
    probe_stride = int(getattr(args, "chunk_fill_window_probe_stride", 64))
    if probe_stride > 0:
        anchor_targets = _uniform_sample_page_counts(
            page_start=args.page_start,
            page_stop=args.page_stop,
            stride=probe_stride,
        )
    else:
        anchor_targets = _progressive_sample_page_counts(
            page_start=args.page_start,
            page_stop=args.page_stop,
            sample_divisor=args.chunk_fill_window_sample_divisor,
        )
    anchor_targets = sorted(set(int(page) for page in anchor_targets if int(page) in set(all_page_counts)))
    missing_anchor_pages = [int(page) for page in anchor_targets if int(page) not in dense_pages_by_page]
    if missing_anchor_pages:
        _log_summary(
            f"# batch={batch} chunk_fill coarse dispatch pages={len(missing_anchor_pages)} workers={worker_count}"
        )
        _run_parallel_workers(
            args=args,
            candidate_ctas_per_sm=[best_cta],
            candidate_splits=candidate_splits,
            page_counts=missing_anchor_pages,
            cache_candidate_ctas_per_sm=[best_cta],
            cache_candidate_splits=candidate_splits,
            capacity_probe_page_counts=capacity_probe_page_counts,
            executor=executor,
            phase_label="chunk_fill",
            on_page_result=on_page_result,
        )

    anchor_pages = sorted(int(page) for page in dense_pages_by_page)
    missing_dense_pages = [int(page) for page in all_page_counts if int(page) not in dense_pages_by_page]
    if not missing_dense_pages:
        return

    interval_pages_by_right_probe: dict[int, list[int]] = {
        int(right_probe): [
            int(page)
            for page in missing_dense_pages
            if int(left_probe) < int(page) < int(right_probe)
        ]
        for left_probe, right_probe in zip(anchor_pages, anchor_pages[1:])
    }
    previous_interval_min: int | None = None
    previous_interval_max: int | None = None
    for left_probe, right_probe in zip(anchor_pages, anchor_pages[1:]):
        interval_pages = interval_pages_by_right_probe[int(right_probe)]
        if not interval_pages:
            preferred = _page_cta_preferred_split(dense_pages_by_page[int(right_probe)], best_cta)
            if preferred is not None:
                previous_interval_min = int(preferred)
                previous_interval_max = int(preferred)
            continue
        right_ties = _page_cta_tied_winner_splits(dense_pages_by_page[int(right_probe)], best_cta)
        left_preferred = _page_cta_preferred_split(dense_pages_by_page[int(left_probe)], best_cta)
        left_ties = _page_cta_tied_winner_splits(dense_pages_by_page[int(left_probe)], best_cta)
        lower_bound = (
            int(previous_interval_min)
            if previous_interval_min is not None
            else (
                int(left_preferred)
                if left_preferred is not None
                else (int(min(left_ties)) if left_ties else int(candidate_splits[0]))
            )
        )
        if not right_ties:
            initial_window = [int(value) for value in candidate_splits]
        else:
            upper_bound = int(max(right_ties))
            if previous_interval_max is not None:
                upper_bound = max(int(previous_interval_max), upper_bound)
            if upper_bound < lower_bound:
                upper_bound = lower_bound
            initial_window = _split_window_from_bounds(
                candidate_splits=candidate_splits,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                relative_pad=float(args.chunk_fill_window_relative_pad),
                absolute_pad=int(args.chunk_fill_window_absolute_pad),
            )
        page_candidate_splits = {
            int(page_count): [int(value) for value in initial_window]
            for page_count in interval_pages
        }
        _log_summary(
            f"# batch={batch} chunk_fill windowed dispatch right_probe={right_probe} "
            f"pages={len(interval_pages)} workers={worker_count}"
        )
        _run_parallel_workers(
            args=args,
            candidate_ctas_per_sm=[best_cta],
            candidate_splits=candidate_splits,
            page_counts=interval_pages,
            cache_candidate_ctas_per_sm=[best_cta],
            cache_candidate_splits=candidate_splits,
            capacity_probe_page_counts=capacity_probe_page_counts,
            page_candidate_splits=page_candidate_splits,
            executor=executor,
            phase_label="chunk_fill",
            on_page_result=on_page_result,
        )

        retry_round = 0
        while True:
            retry_page_candidate_splits: dict[int, list[int]] = {}
            for page_count in interval_pages:
                page_payload = dense_pages_by_page[int(page_count)]
                current_window = [int(value) for value in page_candidate_splits[int(page_count)]]
                if not _page_hits_split_window_edge(
                    page_payload=page_payload,
                    graph_ctas_per_sm=best_cta,
                    candidate_window=current_window,
                    full_candidate_splits=candidate_splits,
                ):
                    continue
                winners = _page_cta_tied_winner_splits(page_payload, best_cta)
                expanded_window = _expanded_split_window(
                    candidate_splits=candidate_splits,
                    current_window=current_window,
                    winners=winners,
                    relative_pad=float(args.chunk_fill_window_relative_pad),
                    absolute_pad=int(args.chunk_fill_window_absolute_pad),
                )
                if expanded_window == current_window:
                    expanded_window = [int(value) for value in candidate_splits]
                if expanded_window == current_window:
                    continue
                retry_page_candidate_splits[int(page_count)] = expanded_window
            if not retry_page_candidate_splits:
                break
            retry_round += 1
            _log_summary(
                f"# batch={batch} chunk_fill windowed retry={retry_round} "
                f"right_probe={right_probe} pages={len(retry_page_candidate_splits)}"
            )
            _run_parallel_workers(
                args=args,
                candidate_ctas_per_sm=[best_cta],
                candidate_splits=candidate_splits,
                page_counts=sorted(retry_page_candidate_splits),
                cache_candidate_ctas_per_sm=[best_cta],
                cache_candidate_splits=candidate_splits,
                capacity_probe_page_counts=capacity_probe_page_counts,
                page_candidate_splits=retry_page_candidate_splits,
                executor=executor,
                phase_label="chunk_fill",
                on_page_result=on_page_result,
            )
            for page_count, window in retry_page_candidate_splits.items():
                page_candidate_splits[int(page_count)] = [int(value) for value in window]

        interval_preferreds = [
            preferred
            for preferred in (
                _page_cta_preferred_split(dense_pages_by_page[int(page_count)], best_cta)
                for page_count in interval_pages
            )
            if preferred is not None
        ]
        if interval_preferreds:
            previous_interval_min = int(min(interval_preferreds))
            previous_interval_max = int(max(interval_preferreds))
        else:
            preferred = _page_cta_preferred_split(dense_pages_by_page[int(right_probe)], best_cta)
            if preferred is not None:
                previous_interval_min = int(preferred)
                previous_interval_max = int(preferred)


def _run_batch(
    *,
    args: argparse.Namespace,
    batch: int,
    candidate_ctas_per_sm: list[int],
    candidate_splits: list[int],
    on_partial_payload: callable | None = None,
) -> dict[str, object]:
    batch_args = argparse.Namespace(**vars(args))
    batch_args.batch = int(batch)
    all_page_counts = _page_counts_for_args(batch_args)
    visible_gpu_count = torch.cuda.device_count()
    requested_workers = int(batch_args.parallel_workers)
    if requested_workers <= 0:
        requested_workers = visible_gpu_count
    worker_count = max(1, min(requested_workers, visible_gpu_count, len(all_page_counts)))
    # Keep downstream helpers on the same effective worker count after
    # clamping to the GPUs visible to this agent.
    batch_args.parallel_workers = int(worker_count)
    sampled_page_counts = _progressive_sample_page_counts(
        page_start=batch_args.page_start,
        page_stop=batch_args.page_stop,
        sample_divisor=batch_args.cta_sample_divisor,
    )
    capacity_probe_page_counts = _capacity_probe_page_counts(
        page_start=batch_args.page_start,
        page_stop=batch_args.page_stop,
    )
    active_ctas = [int(value) for value in candidate_ctas_per_sm]
    cta_search_pages_by_page: dict[int, dict[str, object]] = {}
    dense_pages_by_page: dict[int, dict[str, object]] = {}
    cta_search_rounds: list[dict[str, object]] = []
    cta_search_payload: dict[str, object] | None = None
    best_cta: int | None = None

    def _emit_partial(event: dict[str, object]) -> None:
        if on_partial_payload is None:
            return
        on_partial_payload(event)

    def _record_cta_search_page(page: dict[str, object]) -> None:
        nonlocal cta_search_payload, best_cta
        cta_search_pages_by_page[int(page["page_count"])] = page
        cta_search_payload = _build_batch_payload(
            args=batch_args,
            batch=batch,
            page_payloads=[cta_search_pages_by_page[page_count] for page_count in sorted(cta_search_pages_by_page)],
            candidate_ctas_per_sm=active_ctas,
        )
        best_cta = int(cta_search_payload["best_cta_by_mean_best_chunk_us"])
        _emit_partial(
            {
                "type": "page",
                "phase": "cta_search",
                "batch": int(batch),
                "best_cta_by_mean_best_chunk_us": int(best_cta),
                "page": page,
            }
        )

    def _record_dense_page(page: dict[str, object]) -> None:
        dense_pages_by_page[int(page["page_count"])] = page
        _emit_partial(
            {
                "type": "page",
                "phase": "chunk_fill",
                "batch": int(batch),
                "best_cta_by_mean_best_chunk_us": None if best_cta is None else int(best_cta),
                "page": page,
            }
        )

    pool_context = None
    if worker_count > 1:
        mp_context = mp.get_context("spawn")
        gpu_queue = mp_context.Queue()
        for gpu_id in range(worker_count):
            gpu_queue.put(gpu_id)
        pool_context = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=_init_pool_worker,
            initargs=(gpu_queue,),
        )
    try:
        fixed_cta = int(getattr(batch_args, "fixed_cta", 0))
        if fixed_cta > 0:
            best_cta = int(fixed_cta)
            cta_search_rounds.append(
                {
                    "round": 0,
                    "skipped": True,
                    "fixed_cta": int(best_cta),
                }
            )
            _log_summary(f"# batch={batch} cta_search skipped fixed_cta={best_cta}")
            _emit_partial(
                {
                    "type": "cta_search_skipped",
                    "batch": int(batch),
                    "fixed_cta": int(best_cta),
                }
            )
            _log_summary(f"# batch={batch} chunk_fill start best_cta={best_cta}")
            missing_dense_pages = list(all_page_counts)
            if missing_dense_pages:
                if batch_args.chunk_fill_windowed:
                    _run_windowed_chunk_fill(
                        args=batch_args,
                        batch=batch,
                        best_cta=best_cta,
                        all_page_counts=all_page_counts,
                        dense_pages_by_page=dense_pages_by_page,
                        candidate_splits=candidate_splits,
                        capacity_probe_page_counts=capacity_probe_page_counts,
                        worker_count=worker_count,
                        executor=pool_context,
                        on_page_result=_record_dense_page,
                    )
                else:
                    _log_summary(
                        f"# batch={batch} chunk_fill dispatch pages={len(missing_dense_pages)} workers={worker_count}"
                    )
                    _run_parallel_workers(
                        args=batch_args,
                        candidate_ctas_per_sm=[best_cta],
                        candidate_splits=candidate_splits,
                        page_counts=missing_dense_pages,
                        cache_candidate_ctas_per_sm=[best_cta],
                        capacity_probe_page_counts=capacity_probe_page_counts,
                        executor=pool_context,
                        phase_label="chunk_fill",
                        on_page_result=_record_dense_page,
                    )
            chunk_fill_payload = _build_batch_payload(
                args=batch_args,
                batch=batch,
                page_payloads=[dense_pages_by_page[page] for page in sorted(dense_pages_by_page)],
                candidate_ctas_per_sm=[best_cta],
            )
            cta_search_payload = _build_fixed_cta_search_payload(
                args=batch_args,
                batch=batch,
                best_cta=best_cta,
                chunk_fill_payload=chunk_fill_payload,
            )
            return _build_two_stage_batch_payload(
                args=batch_args,
                batch=batch,
                candidate_ctas_per_sm=candidate_ctas_per_sm,
                cta_search_payload=cta_search_payload,
                cta_search_sampled_page_counts=[],
                cta_search_rounds=cta_search_rounds,
                chunk_fill_payload=chunk_fill_payload,
            )

        remaining_sampled_pages = set(sampled_page_counts)
        round_index = 0
        while remaining_sampled_pages:
            new_pages = sorted(remaining_sampled_pages)
            _log_summary(
                f"# batch={batch} cta_search dispatch round={round_index} pages={len(new_pages)} "
                f"workers={worker_count} active_ctas={','.join(str(cta) for cta in active_ctas)} "
                f"capacity_probes={','.join(str(page) for page in capacity_probe_page_counts)}"
            )
            _run_parallel_workers(
                args=batch_args,
                candidate_ctas_per_sm=active_ctas,
                candidate_splits=candidate_splits,
                page_counts=new_pages,
                cache_candidate_ctas_per_sm=candidate_ctas_per_sm,
                capacity_probe_page_counts=capacity_probe_page_counts,
                executor=pool_context,
                phase_label="cta_search",
                on_page_result=_record_cta_search_page,
            )
            remaining_sampled_pages.clear()
            if cta_search_payload is None:
                break
            refinement_pair = _cta_refinement_pair(cta_search_payload["aggregate_cta_scores"])
            round_summary: dict[str, object] = {
                "round": int(round_index),
                "sampled_page_counts": [int(page) for page in sorted(cta_search_pages_by_page)],
                "new_page_counts": [int(page) for page in new_pages],
                "best_cta_by_mean_best_chunk_us": int(cta_search_payload["best_cta_by_mean_best_chunk_us"]),
            }
            if refinement_pair is not None:
                primary_cta, secondary_cta, relative_gap = refinement_pair
                round_summary.update(
                    {
                        "primary_cta": int(primary_cta),
                        "secondary_cta": int(secondary_cta),
                        "relative_gap": float(relative_gap),
                    }
                )
            round_max_survivors = min(
                int(batch_args.cta_max_survivors),
                max(1, len(active_ctas) // 2),
            )
            survivor_ctas = _cta_survivors(
                cta_scores=cta_search_payload["aggregate_cta_scores"],
                survivor_threshold=batch_args.cta_survivor_threshold,
                max_survivors=round_max_survivors,
            )
            if not survivor_ctas:
                survivor_ctas = [int(cta_search_payload["best_cta_by_mean_best_chunk_us"])]
            round_summary["active_ctas_before_prune"] = [int(cta) for cta in active_ctas]
            round_summary["surviving_ctas"] = [int(cta) for cta in survivor_ctas]
            round_summary["max_survivors_this_round"] = int(round_max_survivors)
            cta_search_rounds.append(round_summary)
            _log_summary(
                f"# batch={batch} cta_search round={round_index} "
                f"samples={len(cta_search_pages_by_page)} best_cta={cta_search_payload['best_cta_by_mean_best_chunk_us']}"
                + (
                    ""
                    if refinement_pair is None
                    else f" runner_up={refinement_pair[1]} rel_gap={refinement_pair[2]:.4f}"
                )
                + f" survivors={','.join(str(cta) for cta in survivor_ctas)}"
                + f" survivor_cap={round_max_survivors}"
            )
            active_ctas = survivor_ctas
            cta_search_payload = _build_batch_payload(
                args=batch_args,
                batch=batch,
                page_payloads=[cta_search_pages_by_page[page_count] for page_count in sorted(cta_search_pages_by_page)],
                candidate_ctas_per_sm=active_ctas,
            )
            best_cta = int(cta_search_payload["best_cta_by_mean_best_chunk_us"])
            _emit_partial(
                {
                    "type": "cta_search_round",
                    "batch": int(batch),
                    "best_cta_by_mean_best_chunk_us": int(best_cta),
                    "aggregate_cta_scores": cta_search_payload["aggregate_cta_scores"],
                    "round": dict(round_summary),
                }
            )
            refinement_pair = _cta_refinement_pair(cta_search_payload["aggregate_cta_scores"])
            if (
                refinement_pair is None
                or len(active_ctas) <= 1
                or round_index >= batch_args.cta_max_refinement_rounds
                or refinement_pair[2] > batch_args.cta_close_threshold
            ):
                break
            primary_cta, secondary_cta, _relative_gap_value = refinement_pair
            next_pages = [
                page
                for page in _refinement_pages_for_cta_pair(
                    page_payloads_by_page=cta_search_pages_by_page,
                    sampled_page_counts=sorted(cta_search_pages_by_page),
                    primary_cta=primary_cta,
                    secondary_cta=secondary_cta,
                    close_threshold=batch_args.cta_close_threshold,
                )
                if page not in cta_search_pages_by_page
            ]
            if not next_pages:
                next_pages = [
                    (left_page + right_page) // 2
                    for left_page, right_page in zip(
                        sorted(cta_search_pages_by_page),
                        sorted(cta_search_pages_by_page)[1:],
                    )
                    if right_page - left_page > 1 and (left_page + right_page) // 2 not in cta_search_pages_by_page
                ]
            remaining_sampled_pages.update(next_pages)
            round_index += 1

        if cta_search_payload is None:
            raise RuntimeError("CTA search produced no measurements")
        best_cta = int(cta_search_payload["best_cta_by_mean_best_chunk_us"])
        for page in cta_search_payload["pages"]:
            dense_pages_by_page[int(page["page_count"])] = _single_cta_page_payload(
                page_payload=page,
                graph_ctas_per_sm=best_cta,
            )
        _log_summary(
            f"# batch={batch} chunk_fill start best_cta={best_cta} "
            f"seed_fixed={cta_search_payload['capture_fixed_split_pages_by_cta'][str(best_cta)]}"
        )
        _emit_partial(
            {
                "type": "chunk_fill_start",
                "batch": int(batch),
                "best_cta_by_mean_best_chunk_us": int(best_cta),
                "capture_fixed_split_pages": cta_search_payload["capture_fixed_split_pages_by_cta"][str(best_cta)],
            }
        )
        missing_dense_pages = [
            page for page in all_page_counts if page not in dense_pages_by_page
        ]
        if missing_dense_pages:
            if batch_args.chunk_fill_windowed:
                _run_windowed_chunk_fill(
                    args=batch_args,
                    batch=batch,
                    best_cta=best_cta,
                    all_page_counts=all_page_counts,
                    dense_pages_by_page=dense_pages_by_page,
                    candidate_splits=candidate_splits,
                    capacity_probe_page_counts=capacity_probe_page_counts,
                    worker_count=worker_count,
                    executor=pool_context,
                    on_page_result=_record_dense_page,
                )
            else:
                _log_summary(
                    f"# batch={batch} chunk_fill dispatch pages={len(missing_dense_pages)} workers={worker_count}"
                )
                _run_parallel_workers(
                    args=batch_args,
                    candidate_ctas_per_sm=[best_cta],
                    candidate_splits=candidate_splits,
                    page_counts=missing_dense_pages,
                    cache_candidate_ctas_per_sm=[best_cta],
                    capacity_probe_page_counts=capacity_probe_page_counts,
                    executor=pool_context,
                    phase_label="chunk_fill",
                    on_page_result=_record_dense_page,
                )
    finally:
        if pool_context is not None:
            pool_context.shutdown(wait=True, cancel_futures=False)
    chunk_fill_payload = _build_batch_payload(
        args=batch_args,
        batch=batch,
        page_payloads=[dense_pages_by_page[page] for page in sorted(dense_pages_by_page)],
        candidate_ctas_per_sm=[best_cta],
    )
    return _build_two_stage_batch_payload(
        args=batch_args,
        batch=batch,
        candidate_ctas_per_sm=candidate_ctas_per_sm,
        cta_search_payload=cta_search_payload,
        cta_search_sampled_page_counts=sorted(cta_search_pages_by_page),
        cta_search_rounds=cta_search_rounds,
        chunk_fill_payload=chunk_fill_payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--mode", choices=["decode", "verify"], default="decode")
    parser.add_argument("--q-seqlen", type=int, default=1)
    parser.add_argument("--batch-list", type=str, default="8")
    parser.add_argument("--batch", type=int, default=8, help=argparse.SUPPRESS)
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
    parser.add_argument("--candidate-splits", type=str, default="1,512")
    parser.add_argument("--cta-sample-divisor", type=int, default=18)
    parser.add_argument("--cta-close-threshold", type=float, default=0.01)
    parser.add_argument("--cta-max-refinement-rounds", type=int, default=3)
    parser.add_argument("--cta-survivor-threshold", type=float, default=0.01)
    parser.add_argument("--cta-max-survivors", type=int, default=8)
    parser.add_argument("--chunk-fill-windowed", action="store_true")
    parser.add_argument("--chunk-fill-window-probe-stride", type=int, default=64)
    parser.add_argument("--chunk-fill-window-sample-divisor", type=int, default=18)
    parser.add_argument("--chunk-fill-window-relative-pad", type=float, default=0.10)
    parser.add_argument("--chunk-fill-window-absolute-pad", type=int, default=10)
    parser.add_argument("--fixed-cta", type=int, default=0)
    parser.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp8_e4m3fn"], default="bf16")
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    global _VERBOSE, _SUMMARY
    _VERBOSE = bool(args.verbose)
    _SUMMARY = bool(args.summary)

    if args.page_start <= 0 or args.page_stop < args.page_start or args.page_step <= 0:
        raise ValueError("expected 1 <= page-start <= page-stop and page-step > 0")
    if args.page_size != 64:
        raise ValueError("primary paged backend expects page_size=64")
    if args.q_seqlen <= 0:
        raise ValueError("--q-seqlen must be positive")
    if args.mode == "decode" and args.q_seqlen != 1:
        raise ValueError("--mode decode requires --q-seqlen=1")
    if args.mode == "verify" and args.q_seqlen <= 1:
        raise ValueError("--mode verify requires --q-seqlen > 1")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.replays <= 0 or args.probe_batch_replays <= 0:
        raise ValueError("--replays and --probe-batch-replays must be positive")
    if args.probe_batch_replays > args.replays:
        raise ValueError("--probe-batch-replays must be <= --replays")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be between 0 and 1")
    if args.cta_sample_divisor <= 0:
        raise ValueError("--cta-sample-divisor must be positive")
    if not 0.0 <= args.cta_close_threshold < 1.0:
        raise ValueError("--cta-close-threshold must be between 0 and 1")
    if args.cta_max_refinement_rounds < 0:
        raise ValueError("--cta-max-refinement-rounds must be non-negative")
    if not 0.0 <= args.cta_survivor_threshold < 1.0:
        raise ValueError("--cta-survivor-threshold must be between 0 and 1")
    if args.cta_max_survivors <= 0:
        raise ValueError("--cta-max-survivors must be positive")
    if args.chunk_fill_window_sample_divisor <= 0:
        raise ValueError("--chunk-fill-window-sample-divisor must be positive")
    if args.chunk_fill_window_probe_stride < 0:
        raise ValueError("--chunk-fill-window-probe-stride must be non-negative")
    if args.chunk_fill_window_relative_pad < 0.0:
        raise ValueError("--chunk-fill-window-relative-pad must be non-negative")
    if args.chunk_fill_window_absolute_pad < 0:
        raise ValueError("--chunk-fill-window-absolute-pad must be non-negative")
    if args.fixed_cta < 0:
        raise ValueError("--fixed-cta must be non-negative")
    if args.capture_page_count <= 0:
        args.capture_page_count = args.page_stop
    if args.capture_page_count < args.page_stop:
        raise ValueError("--capture-page-count must be at least page-stop")
    if args.parallel_workers < 0:
        raise ValueError("--parallel-workers must be non-negative")

    candidate_ctas_per_sm = _parse_candidate_ctas(args.candidate_ctas_per_sm)
    candidate_splits = _parse_candidate_splits(args.candidate_splits)
    batch_list = _parse_batch_list(args.batch_list)

    require_sm120()
    clear_attention_caches()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if not args.output:
        raise ValueError("--output is required")

    output_path = pathlib.Path(args.output)
    checkpoint_path = _checkpoint_output_path(output_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    _append_jsonl(
        checkpoint_path,
        {
            "type": "meta",
            "payload": _render_output_payload(
                args=args,
                batch_payloads=[],
                batch_list=batch_list,
                candidate_ctas_per_sm=candidate_ctas_per_sm,
                candidate_splits=candidate_splits,
            )["config"],
        },
    )
    batches_payload = []
    for batch in batch_list:
        _log_summary(f"# batch={batch} start")
        _append_jsonl(checkpoint_path, {"type": "batch_start", "batch": int(batch)})

        def _checkpoint_batch(event: dict[str, object]) -> None:
            _append_jsonl(checkpoint_path, event)

        batch_payload = _run_batch(
            args=args,
            batch=batch,
            candidate_ctas_per_sm=candidate_ctas_per_sm,
            candidate_splits=candidate_splits,
            on_partial_payload=_checkpoint_batch,
        )
        batches_payload.append(batch_payload)
        _append_jsonl(
            checkpoint_path,
            {
                "type": "batch_complete",
                "batch": int(batch),
                "payload": batch_payload,
            },
        )
        best_score = batch_payload["aggregate_cta_scores"][0]
        best_mean_us = best_score["mean_best_chunk_us"]
        _log_summary(
            f"# batch={batch} best_cta={batch_payload['best_cta_by_mean_best_chunk_us']} "
            f"mean_us={'NA' if best_mean_us is None else f'{best_mean_us:.3f}'}"
        )
    payload = _render_output_payload(
        args=args,
        batch_payloads=batches_payload,
        batch_list=batch_list,
        candidate_ctas_per_sm=candidate_ctas_per_sm,
        candidate_splits=candidate_splits,
    )
    _write_json_atomic(output_path, payload)
    print(f"# wrote {output_path}")


if __name__ == "__main__":
    main()
