#!/usr/bin/env python3
"""Optuna search over realized paged-attention plans for one target point.

This driver searches the realized plan directly rather than indirect scheduler
knobs. Each trial:

- benchmarks only sparkinfer
- captures one fixed FlashInfer FA2 reference output for correctness
- returns score=0 on compile/runtime failure
- returns score=0 on non-finite / bad-cosine outputs
- otherwise maximizes 1 / mean_us
"""

from __future__ import annotations

import argparse
import contextlib
import math
import pathlib
import statistics
import sys
import traceback
from dataclasses import dataclass
from typing import Literal

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL_OPTUNA = ROOT / ".deps" / "optuna"
if LOCAL_OPTUNA.exists():
    sys.path.insert(0, str(LOCAL_OPTUNA))

try:
    import optuna
except Exception as exc:  # pragma: no cover - env-time dependency
    raise ImportError(
        "optuna is required; install it with "
        "`pip install --target .deps/optuna --no-deps optuna colorlog` from the repo root."
    ) from exc

import torch

import benchmarks.benchmark_paged_attention as bench
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
import sparkinfer.attention.paged._forward as paged_api
from sparkinfer.attention.paged import merge as paged_merge
from sparkinfer.attention.paged import planner as paged_planner
from sparkinfer.attention.paged import workspace as paged_workspace
from sparkinfer.attention._shared.contiguous.api import clear_attention_caches

CHUNK_PAGE_LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
LONG_FORM_CUTOFF_TOKENS_LADDER = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 16384, 32768]

DEFAULT_TRIALS = 200
DEFAULT_REPLAYS = 1000
DEFAULT_WARMUP = 3
DEFAULT_COS_THRESHOLD = 0.999

Family = Literal["bf16_decode", "bf16_extend", "fp8_extend"]


@dataclass(frozen=True)
class TargetSpec:
    mode: Literal["decode", "extend"]
    kv_dtype: torch.dtype
    batch: int
    q_seqlen: int
    cache_seqlen: int
    page_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    q_dtype: torch.dtype

    @property
    def family(self) -> Family:
        if self.mode == "decode" and self.kv_dtype == torch.bfloat16:
            return "bf16_decode"
        if self.mode == "extend" and self.kv_dtype == torch.bfloat16:
            return "bf16_extend"
        if self.mode == "extend" and self.kv_dtype == torch.float8_e4m3fn:
            return "fp8_extend"
        raise ValueError(f"unsupported target family for mode={self.mode} kv_dtype={self.kv_dtype}")

    @property
    def name(self) -> str:
        kv = "fp8" if self.kv_dtype == torch.float8_e4m3fn else "bf16"
        return f"{kv}_{self.mode}_q{self.q_seqlen}_k{self.cache_seqlen}"


@dataclass(frozen=True)
class TrialResult:
    score: float
    sparkinfer_mean_us: float
    sparkinfer_ci_low_us: float
    sparkinfer_ci_high_us: float
    sparkinfer_sem_us: float
    plan_desc: str
    cta_tile_q: int
    kv_chunk_size: int
    split_kv: bool
    max_abs: float
    cos: float


def _parse_kv_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp8_e4m3fn":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported kv dtype {name}")


def _study_name_for_target(spec: TargetSpec) -> str:
    return f"{spec.name}_realized_plan"


def _journal_path_for_target(spec: TargetSpec) -> pathlib.Path:
    return pathlib.Path("/tmp/sparkinfer-optuna") / f"{_study_name_for_target(spec)}.journal"


def _sample_common_config(trial: optuna.Trial) -> dict[str, object]:
    return {
        "merge_cta_policy": trial.suggest_categorical("merge_cta_policy", ["formula", "absolute"]),
        "merge_blocks_per_sm_cap": trial.suggest_int("merge_blocks_per_sm_cap", 1, 6),
        "merge_ctas_per_sm": trial.suggest_int("merge_ctas_per_sm", 1, 6),
    }


def _cta_choices_for_family(spec: TargetSpec) -> list[int]:
    if spec.family == "bf16_decode":
        return [16, 64, 128]
    if spec.family == "bf16_extend":
        return [16, 64, 128]
    if spec.family == "fp8_extend":
        return [16, 32, 48, 64]
    raise AssertionError(f"unsupported family {spec.family}")


def _sample_config(trial: optuna.Trial, spec: TargetSpec) -> dict[str, object]:
    direct_split_kv = bool(trial.suggest_categorical("direct_split_kv", [False, True]))
    config: dict[str, object] = {
        **_sample_common_config(trial),
        "direct_cta_tile_q": int(trial.suggest_categorical("direct_cta_tile_q", _cta_choices_for_family(spec))),
        "direct_split_kv": direct_split_kv,
        "direct_kv_chunk_pages": (
            int(trial.suggest_categorical("direct_kv_chunk_pages", CHUNK_PAGE_LADDER)) if direct_split_kv else 0
        ),
    }
    if spec.family == "bf16_extend":
        config.update(
            {
                "bf16_long_form_mode": trial.suggest_categorical(
                    "bf16_long_form_mode", ["planner", "force_on", "force_off"]
                ),
                "bf16_long_form_cutoff_tokens": int(
                    trial.suggest_categorical("bf16_long_form_cutoff_tokens", LONG_FORM_CUTOFF_TOKENS_LADDER)
                ),
            }
        )
    return config


def _q_lengths_from_cu_seqlens(cu_seqlens_q: torch.Tensor) -> list[int]:
    cu = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    return [end - start for start, end in zip(cu[:-1], cu[1:])]


def _build_trial_inputs(spec: TargetSpec, seed: int):
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = bench._make_uniform_paged_inputs(
        batch=spec.batch,
        q_seqlen=spec.q_seqlen,
        cache_seqlen=spec.cache_seqlen,
        page_size=spec.page_size,
        q_heads=spec.q_heads,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        dtype=spec.q_dtype,
        seed=seed,
    )
    k_descale = None
    v_descale = None
    k_scale = None
    v_scale = None
    if spec.kv_dtype == torch.float8_e4m3fn:
        k_cache, v_cache, k_descale, v_descale, k_scale, v_scale = bench._quantize_paged_kv_cache_global_e4m3(
            k_cache,
            v_cache,
            batch=spec.batch,
            kv_heads=spec.kv_heads,
        )
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, k_descale, v_descale, k_scale, v_scale


@contextlib.contextmanager
def _scheduler_overrides(spec: TargetSpec, config: dict[str, object]):
    orig_paged_determine = paged_planner._paged_determine_cta_tile_q
    orig_workspace_create_plan = paged_workspace.create_paged_plan
    orig_api_default_merge = paged_api.default_paged_persistent_ctas
    orig_merge_default_merge = paged_merge.default_paged_persistent_ctas
    orig_bf16_long_form = paged_api._use_bf16_extend_raw_long_form

    def patched_paged_determine_cta_tile_q(*, mode, kv_dtype, packed_qo_len, head_dim, max_effective_kv_pages):
        if mode == spec.mode and kv_dtype == spec.kv_dtype:
            return int(config["direct_cta_tile_q"])
        return orig_paged_determine(
            mode=mode,
            kv_dtype=kv_dtype,
            packed_qo_len=packed_qo_len,
            head_dim=head_dim,
            max_effective_kv_pages=max_effective_kv_pages,
        )

    def patched_workspace_create_plan(*args, **kwargs):
        if bool(config["direct_split_kv"]):
            kwargs["disable_split_kv"] = False
            kwargs["fixed_split_size"] = int(config["direct_kv_chunk_pages"])
        else:
            kwargs["disable_split_kv"] = True
            kwargs["fixed_split_size"] = -1
        return orig_workspace_create_plan(*args, **kwargs)

    def patched_default_persistent_ctas(*, total_rows: int, num_heads: int, device=None) -> int:
        if device is None:
            device = torch.cuda.current_device()
        num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        if config["merge_cta_policy"] == "absolute":
            return int(num_sms * max(int(config["merge_ctas_per_sm"]), 1))
        total_work = max(int(total_rows) * int(num_heads), 1)
        blocks_per_sm = min(int(config["merge_blocks_per_sm_cap"]), math.ceil(total_work / num_sms))
        return int(num_sms * max(blocks_per_sm, 1))

    def patched_bf16_long_form(kv_chunk_size: int) -> bool:
        mode = str(config.get("bf16_long_form_mode", "planner"))
        if mode == "force_on":
            return True
        if mode == "force_off":
            return False
        return kv_chunk_size < int(config.get("bf16_long_form_cutoff_tokens", 2048))

    paged_planner._paged_determine_cta_tile_q = patched_paged_determine_cta_tile_q
    paged_workspace.create_paged_plan = patched_workspace_create_plan
    paged_api.default_paged_persistent_ctas = patched_default_persistent_ctas
    paged_merge.default_paged_persistent_ctas = patched_default_persistent_ctas
    if spec.family == "bf16_extend":
        paged_api._use_bf16_extend_raw_long_form = patched_bf16_long_form
    try:
        yield
    finally:
        paged_planner._paged_determine_cta_tile_q = orig_paged_determine
        paged_workspace.create_paged_plan = orig_workspace_create_plan
        paged_api.default_paged_persistent_ctas = orig_api_default_merge
        paged_merge.default_paged_persistent_ctas = orig_merge_default_merge
        paged_api._use_bf16_extend_raw_long_form = orig_bf16_long_form


def _materialize_current_plan(
    *,
    spec: TargetSpec,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> paged_planner.PagedPlan:
    clear_attention_caches()
    workspace = paged_workspace.PagedAttentionWorkspace.for_tensors(
        mode=spec.mode,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=True,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    return workspace.plan


def _baseline_params(spec: TargetSpec, plan: paged_planner.PagedPlan) -> dict[str, object]:
    config: dict[str, object] = {
        "merge_cta_policy": "formula",
        "merge_blocks_per_sm_cap": 3,
        "merge_ctas_per_sm": 3,
        "direct_cta_tile_q": int(plan.cta_tile_q),
        "direct_split_kv": bool(plan.split_kv),
        "direct_kv_chunk_pages": int(plan.kv_chunk_size // spec.page_size) if plan.split_kv else 0,
    }
    if spec.family == "bf16_extend":
        config.update(
            {
                "bf16_long_form_mode": "planner",
                "bf16_long_form_cutoff_tokens": 2048,
            }
        )
    return config


def _capture_sparkinfer_graph(
    *,
    spec: TargetSpec,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    warmup: int,
):
    output = torch.empty_like(q)
    workspace = paged_workspace.PagedAttentionWorkspace.for_tensors(
        mode=spec.mode,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=True,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    def run() -> None:
        workspace.run(q, k_cache, v_cache, output=output, k_descale=k_descale, v_descale=v_descale)

    graph = bench._capture_graph(run, warmup=warmup)
    return graph, output, workspace.plan


def _capture_reference_output(
    *,
    spec: TargetSpec,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    k_scale: float | None,
    v_scale: float | None,
):
    fa_graph, fa_output = bench._capture_flashinfer_fa2_graph(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        q_seqlen=spec.q_seqlen,
        page_size=spec.page_size,
        q_heads=spec.q_heads,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        q_dtype=spec.q_dtype,
        kv_dtype=spec.kv_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        workspace_bytes=512 * 1024 * 1024,
        warmup=1,
    )
    fa_graph.replay()
    torch.cuda.synchronize()
    return fa_output


def _bench_backend_mean_us(
    graph: torch.cuda.CUDAGraph,
    *,
    replays: int,
    l2_flush=None,
) -> tuple[float, float, float, float]:
    times_ms = bench._bench_graph(graph, replays=replays, l2_flush=l2_flush)
    ci_low_ms, ci_high_ms, sem_ms = bench._mean_ci(times_ms, ci_level=0.95)
    return (
        statistics.fmean(times_ms) * 1000.0,
        ci_low_ms * 1000.0,
        ci_high_ms * 1000.0,
        sem_ms * 1000.0,
    )


def _run_trial(
    *,
    spec: TargetSpec,
    config: dict[str, object],
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    fa_output: torch.Tensor,
    warmup: int,
    replays: int,
    cos_threshold: float,
    l2_flush,
) -> TrialResult:
    with _scheduler_overrides(spec, config):
        clear_attention_caches()
        sparkinfer_graph, sparkinfer_output, plan = _capture_sparkinfer_graph(
            spec=spec,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            k_descale=k_descale,
            v_descale=v_descale,
            warmup=warmup,
        )
        sparkinfer_mean_us, sparkinfer_ci_low_us, sparkinfer_ci_high_us, sparkinfer_sem_us = _bench_backend_mean_us(
            sparkinfer_graph,
            replays=replays,
            l2_flush=l2_flush,
        )
        max_abs = float((sparkinfer_output - fa_output).abs().max().item())
        cos = float(bench._cosine_similarity(sparkinfer_output, fa_output))
        valid = (
            math.isfinite(cos)
            and math.isfinite(max_abs)
            and math.isfinite(sparkinfer_mean_us)
            and sparkinfer_mean_us > 0.0
            and cos >= cos_threshold
        )
        score = (1.0 / sparkinfer_mean_us) if valid else 0.0
        return TrialResult(
            score=score,
            sparkinfer_mean_us=sparkinfer_mean_us,
            sparkinfer_ci_low_us=sparkinfer_ci_low_us,
            sparkinfer_ci_high_us=sparkinfer_ci_high_us,
            sparkinfer_sem_us=sparkinfer_sem_us,
            plan_desc=f"chunk={plan.kv_chunk_size},{'split' if plan.split_kv else 'nosplit'}",
            cta_tile_q=int(plan.cta_tile_q),
            kv_chunk_size=int(plan.kv_chunk_size),
            split_kv=bool(plan.split_kv),
            max_abs=max_abs,
            cos=cos,
        )


def _make_objective(
    *,
    spec: TargetSpec,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    fa_output: torch.Tensor,
    warmup: int,
    replays: int,
    cos_threshold: float,
    l2_flush,
):
    def objective(trial: optuna.Trial) -> float:
        config = _sample_config(trial, spec)
        try:
            result = _run_trial(
                spec=spec,
                config=config,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                k_descale=k_descale,
                v_descale=v_descale,
                fa_output=fa_output,
                warmup=warmup,
                replays=replays,
                cos_threshold=cos_threshold,
                l2_flush=l2_flush,
            )
        except Exception as exc:
            trial.set_user_attr("status", "crash")
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            trial.set_user_attr("traceback", traceback.format_exc(limit=20))
            trial.set_user_attr("score", 0.0)
            return 0.0

        valid = math.isfinite(result.cos) and result.cos >= cos_threshold and math.isfinite(result.max_abs)
        trial.set_user_attr("status", "ok" if valid else "bad_cos")
        trial.set_user_attr("score", result.score)
        trial.set_user_attr("plan_desc", result.plan_desc)
        trial.set_user_attr("cta_tile_q", result.cta_tile_q)
        trial.set_user_attr("kv_chunk_size", result.kv_chunk_size)
        trial.set_user_attr("split_kv", result.split_kv)
        trial.set_user_attr("sparkinfer_mean_us", result.sparkinfer_mean_us)
        trial.set_user_attr("sparkinfer_ci_low_us", result.sparkinfer_ci_low_us)
        trial.set_user_attr("sparkinfer_ci_high_us", result.sparkinfer_ci_high_us)
        trial.set_user_attr("sparkinfer_sem_us", result.sparkinfer_sem_us)
        trial.set_user_attr("max_abs", result.max_abs)
        trial.set_user_attr("cos", result.cos)
        return result.score

    return objective


def _print_top_trials(study: optuna.Study, *, limit: int) -> None:
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    complete.sort(key=lambda trial: float(trial.user_attrs.get("sparkinfer_mean_us", float("inf"))))
    print(f"top {min(limit, len(complete))} trials:")
    for trial in complete[:limit]:
        print(
            {
                "trial": trial.number,
                "score": round(float(trial.value), 9),
                "sparkinfer_mean_us": trial.user_attrs.get("sparkinfer_mean_us"),
                "sparkinfer_ci_low_us": trial.user_attrs.get("sparkinfer_ci_low_us"),
                "sparkinfer_ci_high_us": trial.user_attrs.get("sparkinfer_ci_high_us"),
                "plan": trial.user_attrs.get("plan_desc"),
                "cta_tile_q": trial.user_attrs.get("cta_tile_q"),
                "kv_chunk_size": trial.user_attrs.get("kv_chunk_size"),
                "split_kv": trial.user_attrs.get("split_kv"),
                "cos": trial.user_attrs.get("cos"),
                "params": trial.params,
            }
        )


def _build_target_from_args(args: argparse.Namespace) -> TargetSpec:
    return TargetSpec(
        mode=args.mode,
        kv_dtype=_parse_kv_dtype(args.kv_dtype),
        batch=args.batch,
        q_seqlen=args.q_seqlen,
        cache_seqlen=args.cache_seqlen,
        page_size=args.page_size,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        q_dtype=torch.bfloat16,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["decode", "extend"], default="decode")
    parser.add_argument("--kv-dtype", choices=["bf16", "fp8_e4m3fn"], default="bf16")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-seqlen", type=int, default=1)
    parser.add_argument("--cache-seqlen", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--replays", type=int, default=DEFAULT_REPLAYS)
    parser.add_argument("--cos-threshold", type=float, default=DEFAULT_COS_THRESHOLD)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--journal-path", type=str, default=None)
    parser.add_argument("--enqueue-baseline", action="store_true", default=True)
    parser.add_argument("--no-enqueue-baseline", action="store_false", dest="enqueue_baseline")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    args = parser.parse_args()

    bench.require_sm120()
    if args.replays < 100:
        raise ValueError("--replays must be at least 100")
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)

    spec = _build_target_from_args(args)
    if spec.mode == "decode" and spec.q_seqlen != 1:
        raise ValueError("decode target must use q_seqlen=1")
    if spec.mode == "extend" and spec.q_seqlen <= 1:
        raise ValueError("extend target must use q_seqlen > 1")

    study_name = args.study_name or _study_name_for_target(spec)
    journal_path = pathlib.Path(args.journal_path) if args.journal_path else _journal_path_for_target(spec)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        {
            "target": spec.name,
            "family": spec.family,
            "study_name": study_name,
            "journal_path": str(journal_path),
            "replays": args.replays,
            "trials": args.trials,
            "search_mode": "direct_plan_only",
            "l2_flush": args.flush_l2,
            "l2_flush_bytes": l2_flush_bytes if args.flush_l2 else 0,
        }
    )

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, k_descale, v_descale, k_scale, v_scale = (
        _build_trial_inputs(spec, args.seed)
    )
    clear_attention_caches()
    fa_output = _capture_reference_output(
        spec=spec,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    current_plan = _materialize_current_plan(
        spec=spec,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        n_startup_trials=20,
        constant_liar=True,
    )
    journal_backend = optuna.storages.journal.JournalFileBackend(str(journal_path))
    storage = optuna.storages.JournalStorage(journal_backend)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if args.enqueue_baseline:
        study.enqueue_trial(_baseline_params(spec, current_plan))

    objective = _make_objective(
        spec=spec,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        fa_output=fa_output,
        warmup=args.warmup,
        replays=args.replays,
        cos_threshold=args.cos_threshold,
        l2_flush=l2_flush,
    )
    study.optimize(objective, n_trials=args.trials, timeout=None if args.timeout <= 0 else args.timeout)

    best = study.best_trial
    print("best trial:")
    print(
        {
            "trial": best.number,
            "score": round(float(best.value), 9),
            "sparkinfer_mean_us": best.user_attrs.get("sparkinfer_mean_us"),
            "sparkinfer_ci_low_us": best.user_attrs.get("sparkinfer_ci_low_us"),
            "sparkinfer_ci_high_us": best.user_attrs.get("sparkinfer_ci_high_us"),
            "plan": best.user_attrs.get("plan_desc"),
            "cta_tile_q": best.user_attrs.get("cta_tile_q"),
            "kv_chunk_size": best.user_attrs.get("kv_chunk_size"),
            "split_kv": best.user_attrs.get("split_kv"),
            "cos": best.user_attrs.get("cos"),
            "params": best.params,
        }
    )
    _print_top_trials(study, limit=args.topk)


if __name__ == "__main__":
    main()
