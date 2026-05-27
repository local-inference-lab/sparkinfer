#!/usr/bin/env python3
"""Benchmark graph-replayed paged attention on Qwen-like GQA serving shapes."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from dataclasses import dataclass, replace
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from b12x.attention.paged.reference import paged_attention_reference
from b12x.integration.attention import (
    PagedAttentionWorkspace,
    clear_attention_caches,
)
from b12x.attention.paged.planner import (
    create_paged_plan,
    decode_chunk_pages_for_graph,
    resolve_decode_graph_ctas_per_sm,
)


def require_sm120() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")


def _capture_graph(fn: Callable[[], None], *, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _bench_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    replays: int,
    l2_flush=None,
) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for idx in range(replays):
        if l2_flush is not None:
            l2_flush()
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp8_e4m3fn":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported dtype {name}")


def _resolve_kv_dtype(name: str, q_dtype: torch.dtype) -> torch.dtype:
    if name == "same":
        return q_dtype
    return _dtype_from_name(name)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _relative_l2_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    diff_norm = (a_f - b_f).norm().item()
    ref_norm = max(b_f.norm().item(), 1e-12)
    return diff_norm / ref_norm


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


BENCHMARK_PROFILES: dict[str, dict[str, object]] = {
    "qwen-gqa": {
        "mode": "decode-graph-buckets",
        "batch": 8,
        "batch_buckets": "1,2,4,8,12,16",
        "decode_contexts": "128,16384,32768,65536,131072",
        "capture_context": 0,
        "q_seqlens": "1",
        "cache_seqlens": "64,512,2048,8192",
        "page_size": 64,
        "q_heads": 8,
        "kv_heads": 1,
        "head_dim": 256,
        "dtype": "bf16",
        "kv_dtype": "same",
    },
    "minimax-m2.7": {
        "mode": "decode-graph-buckets",
        "batch": 8,
        "batch_buckets": "1,2,4,8,12,16",
        "decode_contexts": "128,16384,32768,65536,131072",
        "capture_context": 0,
        "q_seqlens": "1",
        "cache_seqlens": "64,512,2048,8192",
        "page_size": 64,
        "q_heads": 24,
        "kv_heads": 4,
        "head_dim": 128,
        "dtype": "bf16",
        "kv_dtype": "same",
    },
}
BENCHMARK_PROFILE_ALIASES = {
    "minimax-m2": "minimax-m2.7",
    "mimimax-m2.7": "minimax-m2.7",
}


def _canonical_profile_name(name: str) -> str:
    return BENCHMARK_PROFILE_ALIASES.get(name, name)


def _profile_choices() -> list[str]:
    return sorted((*BENCHMARK_PROFILES.keys(), *BENCHMARK_PROFILE_ALIASES.keys()))


def _preparse_profile(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", choices=_profile_choices(), default="qwen-gqa")
    args, _ = parser.parse_known_args(argv)
    return _canonical_profile_name(args.profile)


def _gqa_group_size(*, q_heads: int, kv_heads: int) -> int:
    if q_heads <= 0 or kv_heads <= 0:
        raise ValueError("q_heads and kv_heads must be positive")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    return q_heads // kv_heads


def _import_flashinfer():
    try:
        import flashinfer
    except ImportError as exc:  # pragma: no cover - benchmark-time dependency
        raise ImportError(
            "flashinfer is required for --compare-fa2; install it in the benchmark env "
            "or add the repo to PYTHONPATH"
        ) from exc
    return flashinfer


@dataclass(frozen=True)
class ShapeCase:
    phase: str
    batch: int
    q_seqlen: int
    cache_seqlen: int

    @property
    def total_q(self) -> int:
        return self.batch * self.q_seqlen


@dataclass(frozen=True)
class CaseMetrics:
    backend: str
    mean_us: float


@dataclass(frozen=True)
class BackendCapture:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    plan_desc: str


def _build_shape_cases(
    *,
    batch: int,
    q_seqlens: list[int],
    cache_seqlens: list[int],
) -> list[ShapeCase]:
    cases: list[ShapeCase] = []
    for q_seqlen in q_seqlens:
        phase = "decode" if q_seqlen == 1 else "extend"
        for cache_seqlen in cache_seqlens:
            cases.append(
                ShapeCase(
                    phase=phase,
                    batch=batch,
                    q_seqlen=q_seqlen,
                    cache_seqlen=cache_seqlen,
                )
            )
    return cases


def _make_uniform_page_metadata(
    *,
    batch: int,
    cache_seqlen: int,
    page_size: int,
    num_pages: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = "cuda"
    pages_per_request = (cache_seqlen + page_size - 1) // page_size
    total_pages_needed = batch * pages_per_request
    if num_pages < total_pages_needed:
        raise ValueError(
            f"num_pages={num_pages} is too small for batch={batch}, cache_seqlen={cache_seqlen}, "
            f"page_size={page_size}; need at least {total_pages_needed}"
        )
    page_table = torch.zeros(batch, pages_per_request, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    page_order = torch.randperm(num_pages, generator=generator, device=device)
    for request_idx in range(batch):
        start = request_idx * pages_per_request
        page_ids = page_order[start : start + pages_per_request].to(torch.int32)
        page_table[request_idx] = page_ids
    cache_seqlens = torch.full((batch,), cache_seqlen, dtype=torch.int32, device=device)
    return page_table, cache_seqlens


def _make_uniform_paged_inputs(
    *,
    batch: int,
    q_seqlen: int,
    cache_seqlen: int,
    capture_cache_seqlen: int | None,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(seed)
    device = "cuda"
    total_q = batch * q_seqlen
    q = torch.randn(total_q, q_heads, head_dim, device=device, dtype=dtype) / 4
    capture_cache_seqlen = max(cache_seqlen, capture_cache_seqlen or cache_seqlen)
    capture_pages_per_request = (capture_cache_seqlen + page_size - 1) // page_size
    num_pages = batch * capture_pages_per_request
    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    page_table, cache_seqlens = _make_uniform_page_metadata(
        batch=batch,
        cache_seqlen=cache_seqlen,
        page_size=page_size,
        num_pages=num_pages,
        seed=seed,
    )
    capture_page_table, capture_cache_seqlens = _make_uniform_page_metadata(
        batch=batch,
        cache_seqlen=capture_cache_seqlen,
        page_size=page_size,
        num_pages=num_pages,
        seed=seed + 10_000,
    )
    cu_seqlens_q = torch.arange(0, total_q + 1, q_seqlen, dtype=torch.int32, device=device)
    return (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        capture_page_table,
        capture_cache_seqlens,
        cu_seqlens_q,
    )


def _quantize_paged_kv_cache_global_e4m3(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    batch: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    k_scale = k_cache.abs().amax().to(torch.float32) / finfo.max
    v_scale = v_cache.abs().amax().to(torch.float32) / finfo.max
    if float(k_scale.item()) == 0.0:
        k_scale = torch.ones_like(k_scale)
    if float(v_scale.item()) == 0.0:
        v_scale = torch.ones_like(v_scale)
    k_fp8 = (k_cache.to(torch.float32) / k_scale).clamp(min=finfo.min, max=finfo.max).to(
        torch.float8_e4m3fn
    )
    v_fp8 = (v_cache.to(torch.float32) / v_scale).clamp(min=finfo.min, max=finfo.max).to(
        torch.float8_e4m3fn
    )
    k_descale = torch.full(
        (batch, kv_heads),
        float(k_scale.item()),
        dtype=torch.float32,
        device=k_cache.device,
    )
    v_descale = torch.full(
        (batch, kv_heads),
        float(v_scale.item()),
        dtype=torch.float32,
        device=v_cache.device,
    )
    return (
        k_fp8.contiguous(),
        v_fp8.contiguous(),
        k_descale,
        v_descale,
        float(k_scale.item()),
        float(v_scale.item()),
    )


def _make_flashinfer_page_metadata(
    *,
    batch: int,
    q_seqlen: int,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qo_indptr = torch.arange(
        0,
        batch * q_seqlen + 1,
        q_seqlen,
        dtype=torch.int32,
        device=page_table.device,
    )
    pages_per_request = page_table.shape[1]
    paged_kv_indptr = torch.arange(
        0,
        batch * pages_per_request + 1,
        pages_per_request,
        dtype=torch.int32,
        device=page_table.device,
    )
    paged_kv_indices = page_table.reshape(-1).contiguous().to(torch.int32)
    paged_kv_last_page_len = ((cache_seqlens - 1) % page_size + 1).to(torch.int32)
    return qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len


def _format_plan_desc(*, kv_chunk_size: int, split_kv: bool) -> str:
    desc = f"chunk={int(kv_chunk_size)}"
    return f"{desc},split" if split_kv else f"{desc},nosplit"


def _build_backend_graph_plan(
    *,
    workspace: PagedAttentionWorkspace,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    fixed_split_pages: int | None,
    graph_ctas_per_sm: int | None,
) -> object:
    assert workspace._plan_q is not None
    assert workspace._plan_k_cache is not None
    assert workspace._plan_v_cache is not None
    active_total_q = int(cu_seqlens_q[-1].item())
    return create_paged_plan(
        workspace._plan_q[:active_total_q],
        workspace._plan_k_cache,
        workspace._plan_v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode=workspace.mode,
        fixed_split_size=-1 if fixed_split_pages is None else int(fixed_split_pages),
        disable_split_kv=False,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )


def _load_backend_graph_plan(
    *,
    workspace: PagedAttentionWorkspace,
    plan: object,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> str:
    workspace._ensure_capacity(plan)
    workspace._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
    workspace._copy_plan_metadata(plan)
    workspace._plan = plan
    return _format_plan_desc(kv_chunk_size=plan.kv_chunk_size, split_kv=plan.split_kv)


def _decode_effective_cache_tokens(
    *,
    context_tokens: int,
    q_seqlen: int = 1,
) -> int:
    if context_tokens < 0:
        raise ValueError("decode context_tokens must be non-negative")
    if q_seqlen <= 0:
        raise ValueError("decode q_seqlen must be positive")
    return int(context_tokens + q_seqlen)


def _make_decode_context_metadata(
    *,
    batch: int,
    context_tokens: int,
    page_size: int,
    num_pages: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _make_uniform_page_metadata(
        batch=batch,
        cache_seqlen=_decode_effective_cache_tokens(context_tokens=context_tokens),
        page_size=page_size,
        num_pages=num_pages,
        seed=seed,
    )


@dataclass(frozen=True)
class DecodeReplayCase:
    batch: int
    context_tokens: int

    @property
    def effective_cache_tokens(self) -> int:
        return _decode_effective_cache_tokens(context_tokens=self.context_tokens)


def _build_decode_replay_cases(
    *,
    batch_buckets: list[int],
    context_tokens: list[int],
) -> list[DecodeReplayCase]:
    if not batch_buckets:
        raise ValueError("expected at least one batch bucket")
    if not context_tokens:
        raise ValueError("expected at least one decode context")
    if any(batch <= 0 for batch in batch_buckets):
        raise ValueError("decode batch buckets must be positive")
    if any(context <= 0 for context in context_tokens):
        raise ValueError("decode graph bucket contexts must be positive")
    return [
        DecodeReplayCase(
            batch=int(batch),
            context_tokens=int(context),
        )
        for batch in sorted(dict.fromkeys(batch_buckets))
        for context in sorted(dict.fromkeys(context_tokens))
    ]


def _next_power_of_two(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class DecodeGraphBucketPolicy:
    batch: int
    capture_context_tokens: int
    capture_page_count: int
    capture_fixed_split_pages: int | None
    replay_fixed_split_pages: int | None
    graph_ctas_per_sm: int | None
    source: str

    @property
    def effective_capture_tokens(self) -> int:
        return _decode_effective_cache_tokens(context_tokens=self.capture_context_tokens)


def _resolve_decode_graph_bucket_policy(
    *,
    batch: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    decode_contexts: list[int],
    capture_context_override: int,
    fixed_split_pages_override: int,
    graph_ctas_per_sm_override: int,
) -> DecodeGraphBucketPolicy:
    if capture_context_override > 0:
        capture_context_tokens = int(capture_context_override)
        source = "manual"
    else:
        requested_capture_pages = (
            int(_decode_effective_cache_tokens(context_tokens=max(decode_contexts))) + page_size - 1
        ) // page_size
        capture_context_tokens = int(_next_power_of_two(requested_capture_pages) * page_size - 1)
        source = "heuristic"

    if capture_context_tokens < max(decode_contexts):
        raise ValueError("decode graph capture context must cover the largest replay context")
    capture_page_count = (
        int(_decode_effective_cache_tokens(context_tokens=capture_context_tokens)) + page_size - 1
    ) // page_size

    if fixed_split_pages_override > 0:
        capture_fixed_split_pages = int(fixed_split_pages_override)
        replay_fixed_split_pages = int(fixed_split_pages_override)
        source = "manual"
    else:
        graph_ctas_hint = resolve_decode_graph_ctas_per_sm(
            kv_dtype=kv_dtype,
            batch=batch,
            page_size=page_size,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            gqa_group_size=_gqa_group_size(q_heads=q_heads, kv_heads=kv_heads),
            graph_ctas_per_sm=graph_ctas_per_sm_override if graph_ctas_per_sm_override > 0 else None,
        )
        max_chunks_per_req = None
        if torch.cuda.is_available():
            num_sms = int(torch.cuda.get_device_properties("cuda").multi_processor_count)
            max_chunks_per_req = max((num_sms * graph_ctas_hint) // max(int(batch), 1), 1)
        capture_fixed_split_pages = decode_chunk_pages_for_graph(
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            batch=batch,
            page_size=page_size,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            gqa_group_size=_gqa_group_size(q_heads=q_heads, kv_heads=kv_heads),
            max_effective_kv_pages=capture_page_count,
            max_chunks_per_req=max_chunks_per_req,
        )
        replay_fixed_split_pages = None

    if graph_ctas_per_sm_override > 0:
        graph_ctas_per_sm = int(graph_ctas_per_sm_override)
        source = "manual"
    else:
        graph_ctas_per_sm = resolve_decode_graph_ctas_per_sm(
            kv_dtype=kv_dtype,
            batch=batch,
            page_size=page_size,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            gqa_group_size=_gqa_group_size(q_heads=q_heads, kv_heads=kv_heads),
        )

    return DecodeGraphBucketPolicy(
        batch=int(batch),
        capture_context_tokens=int(capture_context_tokens),
        capture_page_count=capture_page_count,
        capture_fixed_split_pages=capture_fixed_split_pages,
        replay_fixed_split_pages=replay_fixed_split_pages,
        graph_ctas_per_sm=graph_ctas_per_sm,
        source=source,
    )


@dataclass(frozen=True)
class DecodeBucketSharedInputs:
    batch: int
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    capture_page_table: torch.Tensor
    capture_cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None
    k_scale: float | None
    v_scale: float | None
    seed: int


def _make_decode_bucket_shared_inputs(
    *,
    batch: int,
    capture_context_tokens: int,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
    seed: int,
) -> DecodeBucketSharedInputs:
    (
        q,
        k_cache,
        v_cache,
        capture_page_table,
        capture_cache_seqlens,
        _capture_page_table_dup,
        _capture_cache_seqlens_dup,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=batch,
        q_seqlen=1,
        cache_seqlen=_decode_effective_cache_tokens(context_tokens=capture_context_tokens),
        capture_cache_seqlen=_decode_effective_cache_tokens(context_tokens=capture_context_tokens),
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        seed=seed,
    )
    k_descale = None
    v_descale = None
    k_scale = None
    v_scale = None
    if kv_dtype == torch.float8_e4m3fn:
        k_cache, v_cache, k_descale, v_descale, k_scale, v_scale = _quantize_paged_kv_cache_global_e4m3(
            k_cache,
            v_cache,
            batch=batch,
            kv_heads=kv_heads,
        )
    return DecodeBucketSharedInputs(
        batch=batch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        k_scale=k_scale,
        v_scale=v_scale,
        seed=seed,
    )


@dataclass
class B12xDecodeGraphBucket:
    shared: DecodeBucketSharedInputs
    workspace: PagedAttentionWorkspace
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    capture_fixed_split_pages: int | None
    replay_fixed_split_pages: int | None
    graph_ctas_per_sm: int | None
    current_page_table: torch.Tensor
    current_cache_seqlens: torch.Tensor
    current_plan_desc: str

    @property
    def batch(self) -> int:
        return self.shared.batch

    @property
    def q(self) -> torch.Tensor:
        return self.shared.q

    @property
    def k_cache(self) -> torch.Tensor:
        return self.shared.k_cache

    @property
    def v_cache(self) -> torch.Tensor:
        return self.shared.v_cache

    @property
    def cu_seqlens_q(self) -> torch.Tensor:
        return self.shared.cu_seqlens_q

    @property
    def k_descale(self) -> torch.Tensor | None:
        return self.shared.k_descale

    @property
    def v_descale(self) -> torch.Tensor | None:
        return self.shared.v_descale

    def prepare_replay(self, *, context_tokens: int) -> None:
        page_table, cache_seqlens = _make_decode_context_metadata(
            batch=self.batch,
            context_tokens=context_tokens,
            page_size=int(self.k_cache.shape[1]),
            num_pages=int(self.k_cache.shape[0]),
            seed=self.shared.seed,
        )
        replay_plan = _build_backend_graph_plan(
            workspace=self.workspace,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=self.cu_seqlens_q,
            fixed_split_pages=self.replay_fixed_split_pages,
            graph_ctas_per_sm=self.graph_ctas_per_sm,
        )
        self.current_plan_desc = _load_backend_graph_plan(
            workspace=self.workspace,
            plan=replay_plan,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=self.cu_seqlens_q,
        )
        self.current_page_table = page_table
        self.current_cache_seqlens = cache_seqlens


@dataclass
class FlashinferDecodeGraphBucket:
    shared: DecodeBucketSharedInputs
    wrapper: object
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    page_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    current_page_table: torch.Tensor
    current_cache_seqlens: torch.Tensor

    @property
    def batch(self) -> int:
        return self.shared.batch

    @property
    def output_view(self) -> torch.Tensor:
        return self.output.view(-1, self.q_heads, self.head_dim)

    def prepare_replay(self, *, context_tokens: int) -> None:
        page_table, cache_seqlens = _make_decode_context_metadata(
            batch=self.batch,
            context_tokens=context_tokens,
            page_size=self.page_size,
            num_pages=int(self.shared.k_cache.shape[0]),
            seed=self.shared.seed,
        )
        qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _make_flashinfer_page_metadata(
            batch=self.batch,
            q_seqlen=1,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            page_size=self.page_size,
        )
        self.wrapper.plan(
            indptr=paged_kv_indptr,
            indices=paged_kv_indices,
            last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.q_heads,
            num_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            sm_scale=self.head_dim ** -0.5,
        )
        self.current_page_table = page_table
        self.current_cache_seqlens = cache_seqlens


def _capture_backend_graph(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    capture_page_table: torch.Tensor,
    capture_cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    fixed_split_pages: int | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    warmup: int,
    graph_ctas_per_sm: int | None,
) -> BackendCapture:
    output = torch.empty_like(q)
    mode = "decode" if int(q.shape[0]) == int(page_table.shape[0]) else "extend"
    workspace = PagedAttentionWorkspace.for_tensors(
        mode=mode,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=False,
    )
    replay_plan = _build_backend_graph_plan(
        workspace=workspace,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=fixed_split_pages,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )
    capture_plan = _build_backend_graph_plan(
        workspace=workspace,
        page_table=capture_page_table,
        cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=fixed_split_pages,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )
    workspace._ensure_capacity(capture_plan)
    workspace.use_cuda_graph = True
    _load_backend_graph_plan(
        workspace=workspace,
        plan=capture_plan,
        page_table=capture_page_table,
        cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
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

    graph = _capture_graph(run, warmup=warmup)
    chunk_desc = _load_backend_graph_plan(
        workspace=workspace,
        plan=replay_plan,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    return BackendCapture(graph=graph, output=output, plan_desc=chunk_desc)


def _capture_flashinfer_fa2_graph(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    capture_page_table: torch.Tensor,
    capture_cache_seqlens: torch.Tensor,
    q_seqlen: int,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    k_scale: float | None,
    v_scale: float | None,
    workspace_bytes: int,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    flashinfer = _import_flashinfer()
    batch = int(page_table.shape[0])
    qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
        _make_flashinfer_page_metadata(
            batch=batch,
            q_seqlen=q_seqlen,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            page_size=page_size,
        )
    )
    (
        capture_qo_indptr,
        capture_paged_kv_indptr,
        capture_paged_kv_indices,
        capture_paged_kv_last_page_len,
    ) = _make_flashinfer_page_metadata(
        batch=batch,
        q_seqlen=q_seqlen,
        cache_seqlens=capture_cache_seqlens,
        page_table=capture_page_table,
        page_size=page_size,
    )
    float_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    sm_scale = head_dim ** -0.5

    if q_seqlen == 1:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            float_workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=capture_paged_kv_indptr.clone(),
            paged_kv_indices_buffer=capture_paged_kv_indices.clone(),
            paged_kv_last_page_len_buffer=capture_paged_kv_last_page_len.clone(),
            backend="fa2",
        )
        wrapper.plan(
            indptr=capture_paged_kv_indptr,
            indices=capture_paged_kv_indices,
            last_page_len=capture_paged_kv_last_page_len,
            num_qo_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            sm_scale=sm_scale,
        )
        q_input = q.view(batch, q_heads, head_dim)
        output = torch.empty_like(q_input)

        def run() -> None:
            wrapper.run(
                q_input,
                (k_cache, v_cache),
                out=output,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        graph = _capture_graph(run, warmup=warmup)
        wrapper.plan(
            indptr=paged_kv_indptr,
            indices=paged_kv_indices,
            last_page_len=paged_kv_last_page_len,
            num_qo_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            sm_scale=sm_scale,
        )
        return graph, output.view(-1, q_heads, head_dim)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=capture_qo_indptr.clone(),
        paged_kv_indptr_buf=capture_paged_kv_indptr.clone(),
        paged_kv_indices_buf=capture_paged_kv_indices.clone(),
        paged_kv_last_page_len_buf=capture_paged_kv_last_page_len.clone(),
        backend="fa2",
    )
    wrapper.plan(
        qo_indptr=capture_qo_indptr,
        paged_kv_indptr=capture_paged_kv_indptr,
        paged_kv_indices=capture_paged_kv_indices,
        paged_kv_last_page_len=capture_paged_kv_last_page_len,
        num_qo_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
        sm_scale=sm_scale,
    )
    output = torch.empty_like(q)

    def run() -> None:
        wrapper.run(q, (k_cache, v_cache), out=output, k_scale=k_scale, v_scale=v_scale)

    graph = _capture_graph(run, warmup=warmup)
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
        sm_scale=sm_scale,
    )
    return graph, output


def _capture_b12x_decode_graph_bucket(
    *,
    shared: DecodeBucketSharedInputs,
    capture_fixed_split_pages: int | None,
    replay_fixed_split_pages: int | None,
    warmup: int,
    graph_ctas_per_sm: int | None,
) -> B12xDecodeGraphBucket:
    workspace = PagedAttentionWorkspace.for_tensors(
        mode="decode",
        q=shared.q,
        k_cache=shared.k_cache,
        v_cache=shared.v_cache,
        use_cuda_graph=False,
    )
    capture_plan = _build_backend_graph_plan(
        workspace=workspace,
        page_table=shared.capture_page_table,
        cache_seqlens=shared.capture_cache_seqlens,
        cu_seqlens_q=shared.cu_seqlens_q,
        fixed_split_pages=capture_fixed_split_pages,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )
    workspace._ensure_capacity(capture_plan)
    workspace.use_cuda_graph = True
    capture_plan_desc = _load_backend_graph_plan(
        workspace=workspace,
        plan=capture_plan,
        page_table=shared.capture_page_table,
        cache_seqlens=shared.capture_cache_seqlens,
        cu_seqlens_q=shared.cu_seqlens_q,
    )
    output = torch.empty_like(shared.q)

    def run() -> None:
        workspace.run(
            shared.q,
            shared.k_cache,
            shared.v_cache,
            output=output,
            k_descale=shared.k_descale,
            v_descale=shared.v_descale,
        )

    graph = _capture_graph(run, warmup=warmup)
    return B12xDecodeGraphBucket(
        shared=shared,
        workspace=workspace,
        graph=graph,
        output=output,
        capture_fixed_split_pages=capture_fixed_split_pages,
        replay_fixed_split_pages=replay_fixed_split_pages,
        graph_ctas_per_sm=graph_ctas_per_sm,
        current_page_table=shared.capture_page_table,
        current_cache_seqlens=shared.capture_cache_seqlens,
        current_plan_desc=capture_plan_desc,
    )


def _capture_flashinfer_decode_graph_bucket(
    *,
    shared: DecodeBucketSharedInputs,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    workspace_bytes: int,
    warmup: int,
) -> FlashinferDecodeGraphBucket:
    flashinfer = _import_flashinfer()
    (
        capture_qo_indptr,
        capture_paged_kv_indptr,
        capture_paged_kv_indices,
        capture_paged_kv_last_page_len,
    ) = _make_flashinfer_page_metadata(
        batch=shared.batch,
        q_seqlen=1,
        cache_seqlens=shared.capture_cache_seqlens,
        page_table=shared.capture_page_table,
        page_size=page_size,
    )
    float_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=shared.q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        float_workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=capture_paged_kv_indptr.clone(),
        paged_kv_indices_buffer=capture_paged_kv_indices.clone(),
        paged_kv_last_page_len_buffer=capture_paged_kv_last_page_len.clone(),
        backend="fa2",
    )
    wrapper.plan(
        indptr=capture_paged_kv_indptr,
        indices=capture_paged_kv_indices,
        last_page_len=capture_paged_kv_last_page_len,
        num_qo_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
        sm_scale=head_dim ** -0.5,
    )
    q_input = shared.q.view(shared.batch, q_heads, head_dim)
    output = torch.empty_like(q_input)

    def run() -> None:
        wrapper.run(
            q_input,
            (shared.k_cache, shared.v_cache),
            out=output,
            k_scale=shared.k_scale,
            v_scale=shared.v_scale,
        )

    graph = _capture_graph(run, warmup=warmup)
    return FlashinferDecodeGraphBucket(
        shared=shared,
        wrapper=wrapper,
        graph=graph,
        output=output,
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        current_page_table=shared.capture_page_table,
        current_cache_seqlens=shared.capture_cache_seqlens,
    )


def _decode_reference_output(
    *,
    shared: DecodeBucketSharedInputs,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:
    ref_out, _ = paged_attention_reference(
        shared.q,
        shared.k_cache,
        shared.v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=shared.k_descale,
        v_descale=shared.v_descale,
        causal=True,
    )
    return ref_out


def _run_legacy_matrix(args: argparse.Namespace) -> None:
    dtype = _dtype_from_name(args.dtype)
    kv_dtype = _resolve_kv_dtype(args.kv_dtype, dtype)
    flashinfer_workspace_bytes = args.flashinfer_workspace_mb * 1024 * 1024
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    q_seqlens = _parse_csv_ints(args.q_seqlens)
    cache_seqlens = _parse_csv_ints(args.cache_seqlens)
    cases = _build_shape_cases(
        batch=args.batch,
        q_seqlens=q_seqlens,
        cache_seqlens=cache_seqlens,
    )

    print(
        "shape matrix:",
        {
            "mode": args.mode,
            "batch": args.batch,
            "q_seqlens": q_seqlens,
            "cache_seqlens": cache_seqlens,
            "page_size": args.page_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "q_dtype": str(dtype),
            "kv_dtype": str(kv_dtype),
            "fixed_split_pages": args.fixed_split_pages,
            "capture_cache_seqlen": args.capture_cache_seqlen,
            "graph_ctas_per_sm": args.graph_ctas_per_sm,
            "replays": args.replays,
            "flashinfer_fa2": args.compare_fa2,
            "l2_flush": args.flush_l2,
        },
    )

    speedups: list[float] = []
    for case_idx, case in enumerate(cases):
        (
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens_tensor,
            capture_page_table,
            capture_cache_seqlens,
            cu_seqlens_q,
        ) = _make_uniform_paged_inputs(
            batch=case.batch,
            q_seqlen=case.q_seqlen,
            cache_seqlen=case.cache_seqlen,
            capture_cache_seqlen=args.capture_cache_seqlen if args.capture_cache_seqlen > 0 else None,
            page_size=args.page_size,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            seed=1 + case_idx,
        )
        k_descale = None
        v_descale = None
        k_scale = None
        v_scale = None
        if kv_dtype == torch.float8_e4m3fn:
            k_cache, v_cache, k_descale, v_descale, k_scale, v_scale = _quantize_paged_kv_cache_global_e4m3(
                k_cache,
                v_cache,
                batch=case.batch,
                kv_heads=args.kv_heads,
            )
        backend_capture = _capture_backend_graph(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens_tensor,
            capture_page_table=capture_page_table,
            capture_cache_seqlens=capture_cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            fixed_split_pages=args.fixed_split_pages if args.fixed_split_pages > 0 else None,
            k_descale=k_descale,
            v_descale=v_descale,
            warmup=args.warmup,
            graph_ctas_per_sm=args.graph_ctas_per_sm if args.graph_ctas_per_sm > 0 else None,
        )
        backend_times_ms = _bench_graph(
            backend_capture.graph,
            replays=args.replays,
            l2_flush=l2_flush,
        )
        backend_metrics = CaseMetrics(
            backend="b12x",
            mean_us=statistics.fmean(backend_times_ms) * 1000.0,
        )

        flashinfer_metrics: CaseMetrics | None = None
        flashinfer_output: torch.Tensor | None = None
        if args.compare_fa2:
            flashinfer_graph, flashinfer_output = _capture_flashinfer_fa2_graph(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens_tensor,
                capture_page_table=capture_page_table,
                capture_cache_seqlens=capture_cache_seqlens,
                q_seqlen=case.q_seqlen,
                page_size=args.page_size,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                q_dtype=dtype,
                kv_dtype=kv_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
                workspace_bytes=flashinfer_workspace_bytes,
                warmup=args.warmup,
            )
            flashinfer_times_ms = _bench_graph(
                flashinfer_graph,
                replays=args.replays,
                l2_flush=l2_flush,
            )
            flashinfer_metrics = CaseMetrics(
                backend="flashinfer-fa2",
                mean_us=statistics.fmean(flashinfer_times_ms) * 1000.0,
            )
            speedups.append(flashinfer_metrics.mean_us / backend_metrics.mean_us)

        check_suffix = ""
        if args.check and flashinfer_output is not None:
            max_abs = (backend_capture.output - flashinfer_output).abs().max().item()
            cos = _cosine_similarity(backend_capture.output, flashinfer_output)
            check_suffix = f" max_abs={max_abs:.5f} cos={cos:.8f}"

        line = (
            f"{case.phase:>6s} "
            f"bs={case.batch:2d} "
            f"q={case.q_seqlen:2d} "
            f"k={case.cache_seqlen:5d} "
            f"{backend_capture.plan_desc:>17s} "
            f"| {backend_metrics.backend} mean={backend_metrics.mean_us:8.1f} us"
        )
        if flashinfer_metrics is not None:
            ratio = flashinfer_metrics.mean_us / backend_metrics.mean_us
            line += (
                f" | fa2 mean={flashinfer_metrics.mean_us:8.1f} us "
                f"| fa2/{backend_metrics.backend}="
                f"{ratio:6.3f}x"
            )
        print(line + check_suffix)

    if speedups:
        print(f"geomean fa2/b12x: {statistics.geometric_mean(speedups):.3f}x")


def _run_decode_graph_buckets(args: argparse.Namespace) -> None:
    if args.q_seqlens != "1":
        raise ValueError("decode-graph-buckets mode only supports --q-seqlens 1")
    dtype = _dtype_from_name(args.dtype)
    kv_dtype = _resolve_kv_dtype(args.kv_dtype, dtype)
    flashinfer_workspace_bytes = args.flashinfer_workspace_mb * 1024 * 1024
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    batch_buckets = _parse_csv_ints(args.batch_buckets)
    decode_contexts = _parse_csv_ints(args.decode_contexts)
    cases = _build_decode_replay_cases(
        batch_buckets=batch_buckets,
        context_tokens=decode_contexts,
    )

    print(
        "decode graph buckets:",
        {
            "profile": args.profile,
            "mode": args.mode,
            "batch_buckets": sorted(dict.fromkeys(batch_buckets)),
            "decode_context_tokens": sorted(dict.fromkeys(decode_contexts)),
            "capture_context_override": None if args.capture_context <= 0 else int(args.capture_context),
            "page_size": args.page_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "q_dtype": str(dtype),
            "kv_dtype": str(kv_dtype),
            "fixed_split_pages": args.fixed_split_pages,
            "graph_ctas_per_sm": args.graph_ctas_per_sm,
            "replays": args.replays,
            "flashinfer_fa2": args.compare_fa2,
            "l2_flush": args.flush_l2,
        },
    )

    speedups: list[float] = []
    for bucket_idx, batch in enumerate(sorted(dict.fromkeys(batch_buckets))):
        bucket_policy = _resolve_decode_graph_bucket_policy(
            batch=batch,
            q_dtype=dtype,
            kv_dtype=kv_dtype,
            page_size=args.page_size,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            decode_contexts=decode_contexts,
            capture_context_override=int(args.capture_context),
            fixed_split_pages_override=int(args.fixed_split_pages),
            graph_ctas_per_sm_override=int(args.graph_ctas_per_sm),
        )
        shared = _make_decode_bucket_shared_inputs(
            batch=batch,
            capture_context_tokens=bucket_policy.capture_context_tokens,
            page_size=args.page_size,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            kv_dtype=kv_dtype,
            seed=1 + bucket_idx,
        )
        capture_fallback_error: Exception | None = None
        try:
            b12x_bucket = _capture_b12x_decode_graph_bucket(
                shared=shared,
                capture_fixed_split_pages=bucket_policy.capture_fixed_split_pages,
                replay_fixed_split_pages=bucket_policy.replay_fixed_split_pages,
                warmup=args.warmup,
                graph_ctas_per_sm=bucket_policy.graph_ctas_per_sm,
            )
        except Exception as exc:
            if args.fixed_split_pages > 0 or bucket_policy.capture_fixed_split_pages is None:
                raise
            capture_fallback_error = exc
            bucket_policy = replace(
                bucket_policy,
                capture_fixed_split_pages=None,
                source=f"{bucket_policy.source}+capture-auto",
            )
            b12x_bucket = _capture_b12x_decode_graph_bucket(
                shared=shared,
                capture_fixed_split_pages=bucket_policy.capture_fixed_split_pages,
                replay_fixed_split_pages=bucket_policy.replay_fixed_split_pages,
                warmup=args.warmup,
                graph_ctas_per_sm=bucket_policy.graph_ctas_per_sm,
            )
        print(
            f"decode-graph-bucket "
            f"bs={batch:2d} "
            f"source={bucket_policy.source:>20s} "
            f"capture_ctx={bucket_policy.capture_context_tokens:6d} "
            f"capture_kv={bucket_policy.effective_capture_tokens:6d} "
            f"capture_pages={bucket_policy.capture_page_count:4d} "
            f"capture_split={str(bucket_policy.capture_fixed_split_pages):>4s} "
            f"replay_split={str(bucket_policy.replay_fixed_split_pages):>4s} "
            f"graph_ctas_per_sm={str(bucket_policy.graph_ctas_per_sm):>4s}"
        )
        if capture_fallback_error is not None:
            print(
                f"decode-graph-bucket "
                f"bs={batch:2d} "
                f"capture_split_fallback=None "
                f"reason={type(capture_fallback_error).__name__}:{capture_fallback_error}"
            )
        fa2_bucket = (
            _capture_flashinfer_decode_graph_bucket(
                shared=shared,
                page_size=args.page_size,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                q_dtype=dtype,
                kv_dtype=kv_dtype,
                workspace_bytes=flashinfer_workspace_bytes,
                warmup=args.warmup,
            )
            if args.compare_fa2
            else None
        )

        for case in (case for case in cases if case.batch == batch):
            try:
                b12x_bucket.prepare_replay(context_tokens=case.context_tokens)
            except Exception as exc:
                print(
                    f"decode-graph "
                    f"bs={case.batch:2d} "
                    f"ctx={case.context_tokens:6d} "
                    f"kv={case.effective_cache_tokens:6d} "
                    f"cap={bucket_policy.capture_context_tokens:6d} "
                    f"blocked={type(exc).__name__}:{exc}"
                )
                continue
            backend_times_ms = _bench_graph(
                b12x_bucket.graph,
                replays=args.replays,
                l2_flush=l2_flush,
            )
            backend_metrics = CaseMetrics(
                backend="b12x",
                mean_us=statistics.fmean(backend_times_ms) * 1000.0,
            )

            flashinfer_metrics: CaseMetrics | None = None
            flashinfer_output: torch.Tensor | None = None
            if fa2_bucket is not None:
                fa2_bucket.prepare_replay(context_tokens=case.context_tokens)
                flashinfer_times_ms = _bench_graph(
                    fa2_bucket.graph,
                    replays=args.replays,
                    l2_flush=l2_flush,
                )
                flashinfer_metrics = CaseMetrics(
                    backend="flashinfer-fa2",
                    mean_us=statistics.fmean(flashinfer_times_ms) * 1000.0,
                )
                flashinfer_output = fa2_bucket.output_view
                speedups.append(flashinfer_metrics.mean_us / backend_metrics.mean_us)

            check_suffix = ""
            if args.check:
                ref_out = _decode_reference_output(
                    shared=shared,
                    page_table=b12x_bucket.current_page_table,
                    cache_seqlens=b12x_bucket.current_cache_seqlens,
                    cu_seqlens_q=b12x_bucket.cu_seqlens_q,
                )
                b12x_ref_rel_l2 = _relative_l2_error(b12x_bucket.output, ref_out)
                b12x_ref_cos = _cosine_similarity(b12x_bucket.output, ref_out)
                check_suffix = f" | b12x/ref rel_l2={b12x_ref_rel_l2:.6f} cos={b12x_ref_cos:.8f}"
                if flashinfer_output is not None:
                    fa2_ref_rel_l2 = _relative_l2_error(flashinfer_output, ref_out)
                    fa2_ref_cos = _cosine_similarity(flashinfer_output, ref_out)
                    cross_rel_l2 = _relative_l2_error(b12x_bucket.output, flashinfer_output)
                    cross_cos = _cosine_similarity(b12x_bucket.output, flashinfer_output)
                    check_suffix += (
                        f" | fa2/ref rel_l2={fa2_ref_rel_l2:.6f} cos={fa2_ref_cos:.8f}"
                        f" | b12x/fa2 rel_l2={cross_rel_l2:.6f} cos={cross_cos:.8f}"
                    )

            line = (
                f"decode-graph "
                f"bs={case.batch:2d} "
                f"ctx={case.context_tokens:6d} "
                f"kv={case.effective_cache_tokens:6d} "
                f"cap={bucket_policy.capture_context_tokens:6d} "
                f"{b12x_bucket.current_plan_desc:>17s} "
                f"| {backend_metrics.backend} mean={backend_metrics.mean_us:8.1f} us"
            )
            if flashinfer_metrics is not None:
                ratio = flashinfer_metrics.mean_us / backend_metrics.mean_us
                line += (
                    f" | fa2 mean={flashinfer_metrics.mean_us:8.1f} us "
                    f"| fa2/{backend_metrics.backend}="
                    f"{ratio:6.3f}x"
                )
            print(line + check_suffix)

        del fa2_bucket
        del b12x_bucket
        del shared
        torch.cuda.empty_cache()

    if speedups:
        print(f"geomean fa2/b12x: {statistics.geometric_mean(speedups):.3f}x")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    profile_name = _preparse_profile(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=_profile_choices(),
        default=profile_name,
        help="Shape preset. Explicit CLI shape flags override preset defaults.",
    )
    parser.add_argument(
        "--mode",
        choices=["legacy-matrix", "decode-graph-buckets"],
        default="decode-graph-buckets",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--batch-buckets", type=str, default="1,2,4,8,12,16")
    parser.add_argument("--decode-contexts", type=str, default="128,16384,32768,65536,131072")
    parser.add_argument("--capture-context", type=int, default=0)
    parser.add_argument("--q-seqlens", type=str, default="1")
    parser.add_argument("--cache-seqlens", type=str, default="64,512,2048,8192")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--kv-dtype", choices=["same", "bf16", "fp16", "fp8_e4m3fn"], default="same")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--flashinfer-workspace-mb", type=int, default=512)
    parser.add_argument("--fixed-split-pages", type=int, default=0)
    parser.add_argument("--capture-cache-seqlen", type=int, default=0)
    parser.add_argument("--graph-ctas-per-sm", type=int, default=0)
    parser.add_argument("--ci-level", type=float, default=0.95, help=argparse.SUPPRESS)
    parser.add_argument("--compare-fa2", action="store_true", default=True)
    parser.add_argument("--no-compare-fa2", action="store_false", dest="compare_fa2")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    parser.set_defaults(**BENCHMARK_PROFILES[profile_name])
    args = parser.parse_args(argv)
    args.profile = _canonical_profile_name(args.profile)

    require_sm120()
    if args.replays < 100:
        raise ValueError("--replays must be at least 100 for graph-replay benchmarking")
    _gqa_group_size(q_heads=args.q_heads, kv_heads=args.kv_heads)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    flush_desc = (
        f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)"
        if args.flush_l2
        else "off"
    )
    print(f"L2 flush: {flush_desc}")
    clear_attention_caches()
    if args.mode == "legacy-matrix":
        _run_legacy_matrix(args)
        return
    _run_decode_graph_buckets(args)


if __name__ == "__main__":
    main()
