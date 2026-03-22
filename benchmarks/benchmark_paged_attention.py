#!/usr/bin/env python3
"""Benchmark graph-replayed paged attention on Qwen-like GQA serving shapes."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.integration.attention import (
    allocate_paged_attention_workspace_for_plan,
    b12x_paged_attention_forward,
    clear_attention_caches,
    create_paged_attention_plan,
)


def require_sm120() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(f"Requires sm_120, got sm_{major}{minor}")


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


def _bench_graph(graph: torch.cuda.CUDAGraph, *, replays: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for idx in range(replays):
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


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


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


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
    median_us: float
    min_us: float


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


def _make_uniform_paged_inputs(
    *,
    batch: int,
    q_seqlen: int,
    cache_seqlen: int,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    device = "cuda"
    total_q = batch * q_seqlen
    q = torch.randn(total_q, q_heads, head_dim, device=device, dtype=dtype) / 4
    pages_per_request = (cache_seqlen + page_size - 1) // page_size
    max_pages = pages_per_request
    total_pages_needed = batch * pages_per_request
    num_pages = max(1, total_pages_needed * 2)
    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    page_order = torch.randperm(num_pages, device=device)
    for request_idx in range(batch):
        start = request_idx * pages_per_request
        page_ids = page_order[start : start + pages_per_request].to(torch.int32)
        page_table[request_idx] = page_ids
    cache_seqlens = torch.full((batch,), cache_seqlen, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, total_q + 1, q_seqlen, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q


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


def _capture_b12x_graph(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    num_splits: int,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, int]:
    plan = create_paged_attention_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        num_splits=num_splits,
    )
    workspace = allocate_paged_attention_workspace_for_plan(plan, total_q=q.shape[0])

    def run() -> None:
        b12x_paged_attention_forward(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            workspace=workspace,
            plan=plan,
            output=workspace.output,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    graph = _capture_graph(run, warmup=warmup)
    return graph, workspace.output, plan.num_splits


def _capture_flashinfer_fa2_graph(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
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
    float_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    sm_scale = head_dim ** -0.5

    if q_seqlen == 1:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            float_workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=paged_kv_indptr.clone(),
            paged_kv_indices_buffer=paged_kv_indices.clone(),
            paged_kv_last_page_len_buffer=paged_kv_last_page_len.clone(),
            backend="fa2",
        )
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
        return graph, output.view(-1, q_heads, head_dim)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr.clone(),
        paged_kv_indptr_buf=paged_kv_indptr.clone(),
        paged_kv_indices_buf=paged_kv_indices.clone(),
        paged_kv_last_page_len_buf=paged_kv_last_page_len.clone(),
        backend="fa2",
    )
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
    output = torch.empty_like(q)

    def run() -> None:
        wrapper.run(q, (k_cache, v_cache), out=output, k_scale=k_scale, v_scale=v_scale)

    graph = _capture_graph(run, warmup=warmup)
    return graph, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-seqlens", type=str, default="1,6")
    parser.add_argument("--cache-seqlens", type=str, default="64,512,2048,8192")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--kv-dtype", choices=["same", "bf16", "fp16", "fp8_e4m3fn"], default="same")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--flashinfer-workspace-mb", type=int, default=512)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--compare-fa2", action="store_true", default=True)
    parser.add_argument("--no-compare-fa2", action="store_false", dest="compare_fa2")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    require_sm120()
    if args.replays < 100:
        raise ValueError("--replays must be at least 100 for graph-replay benchmarking")
    clear_attention_caches()

    dtype = _dtype_from_name(args.dtype)
    kv_dtype = _resolve_kv_dtype(args.kv_dtype, dtype)
    flashinfer_workspace_bytes = args.flashinfer_workspace_mb * 1024 * 1024
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
            "batch": args.batch,
            "q_seqlens": q_seqlens,
            "cache_seqlens": cache_seqlens,
            "page_size": args.page_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "q_dtype": str(dtype),
            "kv_dtype": str(kv_dtype),
            "num_splits": args.num_splits,
            "replays": args.replays,
            "flashinfer_fa2": args.compare_fa2,
        },
    )

    speedups: list[float] = []
    for case_idx, case in enumerate(cases):
        q, k_cache, v_cache, page_table, cache_seqlens_tensor, cu_seqlens_q = (
            _make_uniform_paged_inputs(
                batch=case.batch,
                q_seqlen=case.q_seqlen,
                cache_seqlen=case.cache_seqlen,
                page_size=args.page_size,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                seed=1 + case_idx,
            )
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
        b12x_graph, b12x_output, b12x_num_splits = _capture_b12x_graph(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens_tensor,
            cu_seqlens_q=cu_seqlens_q,
            num_splits=args.num_splits,
            k_descale=k_descale,
            v_descale=v_descale,
            warmup=args.warmup,
        )
        b12x_times_ms = _bench_graph(b12x_graph, replays=args.replays)
        b12x_metrics = CaseMetrics(
            backend="b12x",
            median_us=statistics.median(b12x_times_ms) * 1000.0,
            min_us=min(b12x_times_ms) * 1000.0,
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
            flashinfer_times_ms = _bench_graph(flashinfer_graph, replays=args.replays)
            flashinfer_metrics = CaseMetrics(
                backend="flashinfer-fa2",
                median_us=statistics.median(flashinfer_times_ms) * 1000.0,
                min_us=min(flashinfer_times_ms) * 1000.0,
            )
            speedups.append(flashinfer_metrics.median_us / b12x_metrics.median_us)

        check_suffix = ""
        if args.check and flashinfer_output is not None:
            max_abs = (b12x_output - flashinfer_output).abs().max().item()
            cos = _cosine_similarity(b12x_output, flashinfer_output)
            check_suffix = f" max_abs={max_abs:.5f} cos={cos:.8f}"

        line = (
            f"{case.phase:>6s} "
            f"bs={case.batch:2d} "
            f"q={case.q_seqlen:2d} "
            f"k={case.cache_seqlen:5d} "
            f"splits={b12x_num_splits:2d} "
            f"| b12x median={b12x_metrics.median_us:8.1f} us min={b12x_metrics.min_us:8.1f} us"
        )
        if flashinfer_metrics is not None:
            line += (
                f" | fa2 median={flashinfer_metrics.median_us:8.1f} us "
                f"min={flashinfer_metrics.min_us:8.1f} us "
                f"| fa2/b12x={flashinfer_metrics.median_us / b12x_metrics.median_us:6.3f}x"
            )
        print(line + check_suffix)

    if speedups:
        print(f"geomean fa2/b12x speedup: {statistics.geometric_mean(speedups):.3f}x")


if __name__ == "__main__":
    main()
