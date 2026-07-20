#!/usr/bin/env python3
"""Sweep MoE decode MAX_ACTIVE_CLUSTERS over routed-row points.

The sweep key is routed rows, built from the union of:

    routed_rows = num_tokens * top_k

for every `(num_tokens, top_k)` pair in the requested decode token and top-k
lists.

For each routed-row point this script:

- records every source `(num_tokens, top_k)` pair that maps to that routed-row
  count,
- chooses one representative pair to execute,
- sweeps `MAX_ACTIVE_CLUSTERS` for the `micro` and `dynamic` regimes
  independently,
- uses a persistent process pool so every worker participates in each routed-row
  sweep,
- records raw per-backend candidate measurements plus preferred/tied winners to
  JSON.
"""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gc
import json
import multiprocessing as mp
import os
import pathlib
import statistics
import sys
from typing import Iterator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.benchmark_moe import (
    MODEL_PATH,
    ModelSpec,
    get_scale_contract_params,
    load_expert_weights,
    make_routed_inputs,
    prepare_sparkinfer_benchmark_weights,
    require_sm120,
)
from sparkinfer.cute.utils import get_max_active_clusters, get_num_sm
from sparkinfer.integration.tp_moe import (
    SPARKINFERFP4ExpertWeights,
    allocate_tp_moe_workspace_pool,
    sparkinfer_moe_fp4,
    build_tp_moe_fp4_binding,
    clear_tp_moe_caches,
    select_tp_moe_backend,
)


TP_SIZE = 4
TP_RANK = 0
_SUMMARY = False
_VERBOSE = False
_WORKER_GPU_ID = 0
_WORKER_SPEC_CACHE: dict[int, ModelSpec] = {}
_WORKER_EXPERT_CACHE: dict[tuple[str, str], SPARKINFERFP4ExpertWeights] = {}


@dataclass(frozen=True)
class CandidateSummary:
    num_tokens: int
    top_k: int
    routed_rows: int
    backend: str
    env_name: str
    requested_max_active_clusters: int
    feasible: bool
    sample_count: int
    mean_us: float | None
    ci_low_us: float | None
    ci_high_us: float | None
    error: str | None = None


@dataclass(frozen=True)
class SweepPoint:
    routed_rows: int
    num_tokens: int
    top_k: int
    source_pairs: tuple[tuple[int, int], ...]
    source_backends: tuple[str, ...]


def _log(message: str) -> None:
    if _VERBOSE:
        print(message, file=sys.stderr, flush=True)


def _log_summary(message: str) -> None:
    if _SUMMARY or _VERBOSE:
        print(message, file=sys.stderr, flush=True)


def _capture_graph(fn, *, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _bench_graph(graph: torch.cuda.CUDAGraph, *, replays: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for idx in range(replays):
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end)
        for start, end in zip(starts, ends, strict=True)
    ]


def _mean_ci(
    times_ms: list[float],
    *,
    ci_level: float,
) -> tuple[float, float, float]:
    if not times_ms:
        raise ValueError("mean CI inputs must be non-empty")
    mean = statistics.fmean(times_ms)
    if len(times_ms) == 1:
        return mean, mean, 0.0
    stdev = statistics.stdev(times_ms)
    sem = stdev / (len(times_ms) ** 0.5)
    alpha = (1.0 - ci_level) / 2.0
    z = statistics.NormalDist().inv_cdf(1.0 - alpha)
    half_width = z * sem
    return mean - half_width, mean + half_width, sem


def _parse_candidate_macs(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one candidate in --candidate-max-active-clusters")
    if len(values) == 2:
        lo, hi = sorted(values)
        if lo < hi:
            return list(range(max(lo, 1), hi + 1))
    candidates = sorted({value for value in values if value > 0})
    if not candidates:
        raise ValueError("expected positive MAX_ACTIVE_CLUSTERS candidates")
    return candidates


def _parse_positive_int_list(raw: str, *, arg_name: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"expected at least one value in {arg_name}")
    parsed = sorted({value for value in values if value > 0})
    if not parsed:
        raise ValueError(f"expected positive values in {arg_name}")
    return parsed


def _normalize_top_k_list(raw_values: list[int]) -> list[int]:
    filtered = [int(value) for value in raw_values if int(value) > 1]
    if not filtered:
        raise ValueError("expected at least one top-k value > 1 after filtering out top_k=1")
    if len(filtered) != len(raw_values):
        _log_summary("# skipping top_k=1 due to known forced-backend CuTe runtime issue")
    return filtered


def _parse_backends(raw: str) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one backend in --backends")
    valid = ("micro", "dynamic")
    parsed: list[str] = []
    for value in values:
        if value not in valid:
            raise ValueError(f"unsupported backend {value!r}; expected one of {', '.join(valid)}")
        if value not in parsed:
            parsed.append(value)
    return parsed


def _effective_backend(*, num_tokens: int, top_k: int) -> str:
    return select_tp_moe_backend(
        num_tokens=int(num_tokens),
        num_topk=int(top_k),
        quant_mode="nvfp4",
    )


def _build_sweep_points(*, token_list: list[int], top_k_list: list[int]) -> list[SweepPoint]:
    pairs_by_rows: dict[int, list[tuple[int, int]]] = {}
    for num_tokens in token_list:
        for top_k in top_k_list:
            routed_rows = int(num_tokens * top_k)
            pairs_by_rows.setdefault(routed_rows, []).append((int(num_tokens), int(top_k)))

    points: list[SweepPoint] = []
    for routed_rows in sorted(pairs_by_rows):
        source_pairs = sorted(set(pairs_by_rows[routed_rows]), key=lambda pair: (pair[0], -pair[1]))
        representative_tokens, representative_top_k = source_pairs[0]
        source_backends = tuple(
            _effective_backend(num_tokens=int(num_tokens), top_k=int(top_k))
            for num_tokens, top_k in source_pairs
        )
        points.append(
            SweepPoint(
                routed_rows=int(routed_rows),
                num_tokens=int(representative_tokens),
                top_k=int(representative_top_k),
                source_pairs=tuple((int(num_tokens), int(top_k)) for num_tokens, top_k in source_pairs),
                source_backends=source_backends,
            )
        )
    return points


def _mac_env_name(backend: str) -> str:
    return f"SPARKINFER_{backend.upper()}_MAX_ACTIVE_CLUSTERS"


def _mac_limit() -> int:
    return min(get_max_active_clusters(1), get_num_sm(torch.device("cuda")))


def _sweep_point_from_payload(payload: dict[str, object]) -> SweepPoint:
    return SweepPoint(
        routed_rows=int(payload["routed_rows"]),
        num_tokens=int(payload["num_tokens"]),
        top_k=int(payload["top_k"]),
        source_pairs=tuple(
            (int(num_tokens), int(top_k))
            for num_tokens, top_k in payload["source_pairs"]
        ),
        source_backends=tuple(str(value) for value in payload["source_backends"]),
    )


@contextlib.contextmanager
def _temporary_backend_env(
    *,
    point: SweepPoint,
    backend: str,
    requested_mac: int,
) -> Iterator[None]:
    env_names = (
        "SPARKINFER_MICRO_MAX_ACTIVE_CLUSTERS",
        "SPARKINFER_DYNAMIC_MAX_ACTIVE_CLUSTERS",
        "SPARKINFER_MICRO_DYNAMIC_CUTOVER_PAIRS",
    )
    previous = {name: os.environ.get(name) for name in env_names}
    try:
        for name in env_names:
            os.environ.pop(name, None)
        os.environ[_mac_env_name(backend)] = str(requested_mac)
        if backend == "micro":
            if point.num_tokens > 8:
                raise ValueError(
                    "micro is only defined for workloads with at most 8 tokens"
                )
            os.environ["SPARKINFER_MICRO_DYNAMIC_CUTOVER_PAIRS"] = str(
                point.routed_rows + 1
            )
        elif backend == "dynamic":
            os.environ["SPARKINFER_MICRO_DYNAMIC_CUTOVER_PAIRS"] = "0"
        else:
            raise ValueError(f"unsupported backend override: {backend}")
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _candidate_summary(
    *,
    point: SweepPoint,
    backend: str,
    requested_mac: int,
    env_name: str,
    samples_ms: list[float],
    ci_level: float,
) -> CandidateSummary:
    ci_low_ms, ci_high_ms, _ = _mean_ci(samples_ms, ci_level=ci_level)
    return CandidateSummary(
        num_tokens=int(point.num_tokens),
        top_k=int(point.top_k),
        routed_rows=int(point.routed_rows),
        backend=backend,
        env_name=env_name,
        requested_max_active_clusters=int(requested_mac),
        feasible=True,
        sample_count=int(len(samples_ms)),
        mean_us=float(statistics.fmean(samples_ms) * 1000.0),
        ci_low_us=float(ci_low_ms * 1000.0),
        ci_high_us=float(ci_high_ms * 1000.0),
    )


def _failed_candidate_summary(
    *,
    point: SweepPoint,
    backend: str,
    requested_mac: int,
    error: str,
    sample_count: int = 0,
) -> CandidateSummary:
    return CandidateSummary(
        num_tokens=int(point.num_tokens),
        top_k=int(point.top_k),
        routed_rows=int(point.routed_rows),
        backend=backend,
        env_name=_mac_env_name(backend),
        requested_max_active_clusters=int(requested_mac),
        feasible=False,
        sample_count=int(sample_count),
        mean_us=None,
        ci_low_us=None,
        ci_high_us=None,
        error=error,
    )


def _candidate_error_message(
    *,
    phase: str,
    point: SweepPoint,
    backend: str,
    requested_mac: int,
    exc: BaseException,
) -> str:
    return (
        f"# routed_rows={point.routed_rows} tokens={point.num_tokens} top_k={point.top_k} "
        f"backend={backend} mac={requested_mac} {phase}_failed "
        f"{type(exc).__name__}: {exc}"
    )


def _build_spec(*, top_k: int) -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=int(top_k),
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


def _capture_and_measure_candidate(
    *,
    args: argparse.Namespace,
    point: SweepPoint,
    spec: ModelSpec,
    experts: SPARKINFERFP4ExpertWeights,
    backend: str,
    requested_mac: int,
    device: torch.device,
) -> CandidateSummary:
    x, topk_ids, topk_weights = make_routed_inputs(
        spec,
        point.num_tokens,
        args.seed + point.routed_rows * 1009 + requested_mac * 17 + point.top_k * 31,
        device,
    )
    output = torch.empty_like(x)
    env_name = _mac_env_name(backend)

    with _temporary_backend_env(point=point, backend=backend, requested_mac=requested_mac):
        clear_tp_moe_caches()
        workspace = allocate_tp_moe_workspace_pool()
        binding = build_tp_moe_fp4_binding(
            scratch=workspace,
            a=x,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            fast_math=args.fast_math,
            output=output,
            input_scales_static=True,
            quant_mode="nvfp4",
        )
        if binding.implementation != backend:
            raise RuntimeError(
                f"requested {backend!r}, planner selected "
                f"{binding.implementation!r}"
            )

        def run() -> None:
            sparkinfer_moe_fp4(binding=binding)

        graph = _capture_graph(run, warmup=args.warmup)
        samples_ms = _bench_graph(graph, replays=args.replays)
    return _candidate_summary(
        point=point,
        backend=backend,
        requested_mac=requested_mac,
        env_name=env_name,
        samples_ms=samples_ms,
        ci_level=args.ci_level,
    )


def _reset_worker_cache() -> None:
    global _WORKER_SPEC_CACHE, _WORKER_EXPERT_CACHE
    _WORKER_SPEC_CACHE = {}
    _WORKER_EXPERT_CACHE = {}
    clear_tp_moe_caches()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _init_pool_worker(gpu_queue: object) -> None:
    global _SUMMARY, _VERBOSE, _WORKER_GPU_ID
    _SUMMARY = False
    _VERBOSE = False
    _WORKER_GPU_ID = int(gpu_queue.get())
    torch.cuda.set_device(_WORKER_GPU_ID)
    _reset_worker_cache()


def _ensure_worker_context(
    *,
    model_path: pathlib.Path,
    top_k: int,
    scale_contract: str,
) -> tuple[ModelSpec, SPARKINFERFP4ExpertWeights]:
    global _WORKER_SPEC_CACHE, _WORKER_EXPERT_CACHE
    require_sm120()
    torch.empty(1, device="cuda")
    spec = _WORKER_SPEC_CACHE.get(int(top_k))
    if spec is None:
        spec = _build_spec(top_k=top_k)
        _WORKER_SPEC_CACHE[int(top_k)] = spec
    weight_cache_key = (str(model_path), str(scale_contract))
    experts = _WORKER_EXPERT_CACHE.get(weight_cache_key)
    if experts is None:
        # Expert weights are shape-independent with respect to top_k. Keep one
        # prepared owner per worker and rebuild only the cheap ModelSpec.
        weights = load_expert_weights(model_path, spec)
        params = get_scale_contract_params(weights, scale_contract)
        experts, _ = prepare_sparkinfer_benchmark_weights(
            weights,
            params,
            quant_mode="nvfp4",
            activation="silu",
        )
        _WORKER_EXPERT_CACHE[weight_cache_key] = experts
    return spec, experts


def _chunk_candidates(values: list[int], *, chunk_count: int) -> list[list[int]]:
    if not values:
        return []
    chunk_count = max(1, min(int(chunk_count), len(values)))
    base, remainder = divmod(len(values), chunk_count)
    chunks: list[list[int]] = []
    start = 0
    for chunk_idx in range(chunk_count):
        width = base + (1 if chunk_idx < remainder else 0)
        stop = start + width
        if start < stop:
            chunks.append([int(value) for value in values[start:stop]])
        start = stop
    return chunks


def _worker_measure_mac_chunk(task: dict[str, object]) -> dict[str, object]:
    args = argparse.Namespace(**task["args"])
    point = _sweep_point_from_payload(task["point"])
    requested_macs = [int(value) for value in task["requested_macs"]]
    backends = [str(value) for value in task["backends"]]
    model_path = pathlib.Path(str(task["model_path"]))

    device = torch.device("cuda")
    spec, experts = _ensure_worker_context(
        model_path=model_path,
        top_k=point.top_k,
        scale_contract=args.scale_contract,
    )
    summaries: list[dict[str, object]] = []
    for backend in backends:
        for requested_mac in requested_macs:
            try:
                summary = _capture_and_measure_candidate(
                    args=args,
                    point=point,
                    spec=spec,
                    experts=experts,
                    backend=backend,
                    requested_mac=requested_mac,
                    device=device,
                )
            except Exception as exc:
                message = _candidate_error_message(
                    phase="candidate",
                    point=point,
                    backend=backend,
                    requested_mac=requested_mac,
                    exc=exc,
                )
                print(message, file=sys.stderr, flush=True)
                summary = _failed_candidate_summary(
                    point=point,
                    backend=backend,
                    requested_mac=requested_mac,
                    error=message,
                )
            summaries.append(asdict(summary))
    return {
        "routed_rows": int(point.routed_rows),
        "requested_macs": [int(value) for value in requested_macs],
        "backends": backends,
        "summaries": summaries,
        "gpu_id": int(_WORKER_GPU_ID),
    }


def _select_backend_winners(backend_summaries: list[CandidateSummary]) -> tuple[CandidateSummary | None, list[CandidateSummary], CandidateSummary | None]:
    feasible_summaries = [summary for summary in backend_summaries if bool(summary.feasible)]
    if not feasible_summaries:
        return None, [], None
    best_summary = min(feasible_summaries, key=lambda summary: float(summary.mean_us))
    tied_summaries = [
        summary
        for summary in feasible_summaries
        if float(summary.ci_low_us) <= float(best_summary.ci_high_us)
    ]
    preferred_summary = min(
        tied_summaries,
        key=lambda summary: int(summary.requested_max_active_clusters),
    )
    return best_summary, tied_summaries, preferred_summary


def _evaluate_point_parallel(
    *,
    args: argparse.Namespace,
    point: SweepPoint,
    candidate_macs: list[int],
    worker_count: int,
    executor: ProcessPoolExecutor,
) -> dict[str, object]:
    runtime_selected_backend = _effective_backend(num_tokens=point.num_tokens, top_k=point.top_k)
    source_backend_set = sorted(set(point.source_backends))
    mac_chunks = _chunk_candidates(candidate_macs, chunk_count=worker_count)
    active_backends = [str(value) for value in args.backend_values]
    _log_summary(
        f"# routed_rows={point.routed_rows} tokens={point.num_tokens} top_k={point.top_k} "
        f"source_pairs={len(point.source_pairs)} source_backends={','.join(source_backend_set)} "
        f"backends={','.join(active_backends)} mac_chunks={len(mac_chunks)}"
    )

    futures = {
        executor.submit(
            _worker_measure_mac_chunk,
            {
                "args": {**vars(args)},
                "point": asdict(point),
                "requested_macs": [int(value) for value in chunk],
                "backends": active_backends,
                "model_path": str(MODEL_PATH),
            },
        ): tuple(int(value) for value in chunk)
        for chunk in mac_chunks
    }

    all_summaries: list[CandidateSummary] = []
    for future in as_completed(futures):
        payload = future.result()
        _log_summary(
            f"# routed_rows={point.routed_rows} macs="
            f"{payload['requested_macs'][0]}..{payload['requested_macs'][-1]} "
            f"gpu={payload['gpu_id']} complete"
        )
        all_summaries.extend(CandidateSummary(**summary) for summary in payload["summaries"])

    all_summaries.sort(key=lambda summary: (summary.backend, int(summary.requested_max_active_clusters)))
    backend_results: dict[str, dict[str, object]] = {}
    for backend in active_backends:
        backend_summaries = [summary for summary in all_summaries if summary.backend == backend]
        best_summary, tied_summaries, preferred_summary = _select_backend_winners(backend_summaries)
        if preferred_summary is None:
            _log_summary(f"# routed_rows={point.routed_rows} backend={backend} no_feasible_candidates")
        else:
            assert best_summary is not None
            _log_summary(
                f"# routed_rows={point.routed_rows} backend={backend} "
                f"preferred_mac={preferred_summary.requested_max_active_clusters} "
                f"tied={len(tied_summaries)} best_mean_us={best_summary.mean_us:.3f}"
            )
        backend_results[backend] = {
            "backend": backend,
            "env_name": _mac_env_name(backend),
            "preferred_winner": None if preferred_summary is None else asdict(preferred_summary),
            "tied_winners": [asdict(summary) for summary in tied_summaries],
            "all_candidates": [asdict(summary) for summary in backend_summaries],
            "best_mean_us": None if best_summary is None else float(best_summary.mean_us),
            "best_ci_low_us": None if best_summary is None else float(best_summary.ci_low_us),
            "best_ci_high_us": None if best_summary is None else float(best_summary.ci_high_us),
        }

    runtime_result = backend_results.get(runtime_selected_backend)
    return {
        "routed_rows": int(point.routed_rows),
        "num_tokens": int(point.num_tokens),
        "top_k": int(point.top_k),
        "runtime_selected_backend": runtime_selected_backend,
        "source_pairs": [
            {
                "num_tokens": int(num_tokens),
                "top_k": int(top_k),
                "backend": _effective_backend(num_tokens=int(num_tokens), top_k=int(top_k)),
            }
            for num_tokens, top_k in point.source_pairs
        ],
        "source_backends": source_backend_set,
        "backend_ambiguous": len(source_backend_set) > 1,
        "backend_results": backend_results,
        "preferred_winner": (
            None if runtime_result is None else runtime_result["preferred_winner"]
        ),
        "tied_winners": [] if runtime_result is None else runtime_result["tied_winners"],
        "all_candidates": [] if runtime_result is None else runtime_result["all_candidates"],
        "best_mean_us": None if runtime_result is None else runtime_result["best_mean_us"],
        "best_ci_low_us": (
            None if runtime_result is None else runtime_result["best_ci_low_us"]
        ),
        "best_ci_high_us": (
            None if runtime_result is None else runtime_result["best_ci_high_us"]
        ),
    }


def _render_output_payload(
    *,
    args: argparse.Namespace,
    point_payloads: list[dict[str, object]],
    points: list[SweepPoint],
    candidate_macs: list[int],
) -> dict[str, object]:
    return {
        "version": 3,
        "config": {
            "token_list": [int(value) for value in args.token_list_values],
            "top_k_list": [int(value) for value in args.top_k_list_values],
            "backends": [str(value) for value in args.backend_values],
            "routed_rows": [int(point.routed_rows) for point in points],
            "candidate_max_active_clusters": [int(value) for value in candidate_macs],
            "parallel_workers": int(args.parallel_workers),
            "scale_contract": args.scale_contract,
            "fast_math": bool(args.fast_math),
            "warmup": int(args.warmup),
            "replays": int(args.replays),
            "ci_level": float(args.ci_level),
            "seed": int(args.seed),
            "mac_limit": int(_mac_limit()),
        },
        "points": point_payloads,
    }


def _write_json_atomic(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _checkpoint_output_path(output_path: pathlib.Path) -> pathlib.Path:
    return output_path.with_name(output_path.name + ".checkpoint.jsonl")


def _append_jsonl(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--token-list", type=str, required=True)
    parser.add_argument("--top-k-list", type=str, required=True)
    parser.add_argument("--backends", type=str, default="micro,dynamic")
    parser.add_argument("--candidate-max-active-clusters", type=str, default="1,16")
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--scale-contract", choices=["shared", "per-expert"], default="shared")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--fast-math", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    global _SUMMARY, _VERBOSE
    _SUMMARY = bool(args.summary)
    _VERBOSE = bool(args.verbose)

    if args.replays <= 0:
        raise ValueError("--replays must be positive")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be between 0 and 1")
    if args.parallel_workers < 0:
        raise ValueError("--parallel-workers must be non-negative")

    candidate_macs = _parse_candidate_macs(args.candidate_max_active_clusters)
    _log_summary(
        f"# candidate_max_active_clusters count={len(candidate_macs)} "
        f"min={candidate_macs[0]} max={candidate_macs[-1]}"
    )
    token_list = _parse_positive_int_list(args.token_list, arg_name="--token-list")
    top_k_list = _normalize_top_k_list(
        _parse_positive_int_list(args.top_k_list, arg_name="--top-k-list")
    )
    backend_values = _parse_backends(args.backends)
    args.token_list_values = token_list
    args.top_k_list_values = top_k_list
    args.backend_values = backend_values
    points = _build_sweep_points(token_list=token_list, top_k_list=top_k_list)
    _log_summary(
        f"# routed_row_points count={len(points)} min={points[0].routed_rows} max={points[-1].routed_rows}"
    )
    _log_summary(f"# backends {','.join(backend_values)}")

    require_sm120()
    torch.empty(1, device="cuda")

    output_path = pathlib.Path(args.output)
    checkpoint_path = _checkpoint_output_path(output_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    clear_tp_moe_caches()

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        raise RuntimeError("parallel worker mode requires at least one visible CUDA device")
    requested_workers = int(args.parallel_workers)
    if requested_workers <= 0:
        requested_workers = visible_gpu_count
    worker_count = max(1, min(requested_workers, visible_gpu_count, len(candidate_macs)))
    _log_summary(f"# parallel_workers requested={args.parallel_workers} effective={worker_count}")

    _append_jsonl(
        checkpoint_path,
        {
            "type": "meta",
            "payload": _render_output_payload(
                args=args,
                point_payloads=[],
                points=points,
                candidate_macs=candidate_macs,
            )["config"],
        },
    )

    point_payloads: list[dict[str, object]] = []
    mp_context = mp.get_context("spawn")
    gpu_queue = mp_context.Queue()
    for gpu_id in range(worker_count):
        gpu_queue.put(gpu_id % visible_gpu_count)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=_init_pool_worker,
        initargs=(gpu_queue,),
    ) as executor:
        for point in points:
            _append_jsonl(
                checkpoint_path,
                {
                    "type": "point_start",
                    "routed_rows": int(point.routed_rows),
                    "num_tokens": int(point.num_tokens),
                    "top_k": int(point.top_k),
                },
            )
            try:
                point_payload = _evaluate_point_parallel(
                    args=args,
                    point=point,
                    candidate_macs=candidate_macs,
                    worker_count=worker_count,
                    executor=executor,
                )
            except Exception as exc:
                _append_jsonl(
                    checkpoint_path,
                    {
                        "type": "point_failed",
                        "routed_rows": int(point.routed_rows),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                _write_json_atomic(
                    output_path,
                    _render_output_payload(
                        args=args,
                        point_payloads=point_payloads,
                        points=points,
                        candidate_macs=candidate_macs,
                    ),
                )
                raise
            point_payloads.append(point_payload)
            _append_jsonl(
                checkpoint_path,
                {
                    "type": "point_complete",
                    "routed_rows": int(point.routed_rows),
                    "payload": point_payload,
                },
            )

    payload = _render_output_payload(
        args=args,
        point_payloads=point_payloads,
        points=points,
        candidate_macs=candidate_macs,
    )
    _write_json_atomic(output_path, payload)

    if args.summary:
        print()
        print("routed_rows\truntime_backend\tmicro_mac\tdynamic_mac")
        for point_payload in point_payloads:
            def _preferred_mac(backend: str) -> str:
                backend_result = point_payload["backend_results"].get(backend)
                if backend_result is None:
                    return "-"
                preferred = backend_result["preferred_winner"]
                if preferred is None:
                    return "-"
                return str(preferred["requested_max_active_clusters"])

            print(
                f"{point_payload['routed_rows']}\t{point_payload['runtime_selected_backend']}\t"
                f"{_preferred_mac('micro')}\t"
                f"{_preferred_mac('dynamic')}"
            )


if __name__ == "__main__":
    main()
