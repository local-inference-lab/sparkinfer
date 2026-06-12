#!/usr/bin/env python3
"""Graph-replay vLLM Triton paged decode probes for B12X comparisons."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.benchmark_paged_attention import (
    _cosine_similarity,
    _decode_effective_cache_tokens,
    _make_decode_context_metadata,
    _make_decode_bucket_shared_inputs,
    _relative_l2_error,
)
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from b12x.attention.paged.reference import paged_attention_reference
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode


PROFILES: dict[str, dict[str, int]] = {
    "minimax-m2.7": {"q_heads": 24, "kv_heads": 4, "head_dim": 128},
    "qwen-gqa": {"q_heads": 8, "kv_heads": 1, "head_dim": 256},
}


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


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
    l2_flush: Callable[[], None] | None,
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
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _reference_output(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:
    ref_out, _ = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    return ref_out


def _make_unified_runner(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    use_3d: bool,
    num_segments: int,
) -> tuple[Callable[[], None], torch.Tensor]:
    batch, q_heads, head_dim = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    head_dim_padded = 1 << (head_dim - 1).bit_length()
    max_seqlen_k = int(cache_seqlens.max().item())
    out = torch.empty_like(q)
    one = torch.ones((), dtype=torch.float32, device=q.device)
    segm_output = segm_max = segm_expsum = None
    seq_threshold = None
    if use_3d:
        seq_threshold = max(batch, 1)
        segm_output = torch.empty(
            (seq_threshold, q_heads, num_segments, head_dim_padded),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            (seq_threshold, q_heads, num_segments),
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty_like(segm_max)

    def run() -> None:
        unified_attention(
            q,
            k_cache,
            v_cache,
            out,
            cu_seqlens_q,
            1,
            cache_seqlens,
            max_seqlen_k,
            head_dim**-0.5,
            True,
            (-1, -1),
            page_table,
            0.0,
            one,
            one,
            one,
            seq_threshold_3D=seq_threshold,
            num_par_softmax_segments=num_segments if use_3d else None,
            softmax_segm_output=segm_output,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
            kv_quant_mode=KVQuantMode.NONE,
        )

    return run, out


def _make_split_decode_runner(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    num_splits: int,
) -> tuple[Callable[[], None], torch.Tensor]:
    batch, q_heads, head_dim = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    q_bhd = q.view(batch, q_heads, head_dim)
    k_flat = k_cache.view(-1, int(k_cache.shape[2]), head_dim)
    v_flat = v_cache.view(-1, int(v_cache.shape[2]), head_dim)
    out = torch.empty_like(q_bhd)
    lse = torch.empty((batch, q_heads), dtype=torch.float32, device=q.device)
    one = torch.ones((), dtype=torch.float32, device=q.device)
    attn_logits = torch.empty(
        (batch, q_heads, num_splits, head_dim + 1),
        dtype=torch.float32,
        device=q.device,
    )

    def run() -> None:
        decode_attention_fwd(
            q_bhd,
            k_flat,
            v_flat,
            out,
            lse,
            page_table,
            cache_seqlens,
            attn_logits,
            num_splits,
            head_dim**-0.5,
            page_size=int(k_cache.shape[1]),
            k_scale=one,
            v_scale=one,
        )

    return run, out.view_as(q)


def _summarize(times_us: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(times_us),
        "median_us": statistics.median(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="minimax-m2.7")
    parser.add_argument("--batch-buckets", default="1,4")
    parser.add_argument("--decode-contexts", default="128,16384,65536")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--splits", default="1,4,8,16")
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--json-out", type=pathlib.Path)
    args = parser.parse_args()

    if args.replays < 20:
        raise ValueError("--replays must be at least 20")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    profile = PROFILES[args.profile]
    batch_buckets = _parse_csv_ints(args.batch_buckets)
    contexts = _parse_csv_ints(args.decode_contexts)
    splits = _parse_csv_ints(args.splits)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    l2_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    print(
        "vllm triton paged decode:",
        {
            "profile": args.profile,
            "batch_buckets": batch_buckets,
            "decode_context_tokens": contexts,
            "page_size": args.page_size,
            **profile,
            "replays": args.replays,
            "warmup": args.warmup,
            "splits": splits,
            "segments": args.segments,
            "l2_flush": args.flush_l2,
            "l2_flush_bytes": l2_bytes if args.flush_l2 else 0,
        },
    )

    records: list[dict[str, object]] = []
    max_context = max(contexts)
    for batch_idx, batch in enumerate(batch_buckets):
        shared = _make_decode_bucket_shared_inputs(
            batch=batch,
            capture_context_tokens=max_context,
            page_size=args.page_size,
            q_heads=profile["q_heads"],
            kv_heads=profile["kv_heads"],
            head_dim=profile["head_dim"],
            dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            seed=args.seed + batch_idx,
        )
        for context in contexts:
            page_table, cache_seqlens = _make_decode_context_metadata(
                batch=batch,
                context_tokens=context,
                page_size=args.page_size,
                num_pages=int(shared.k_cache.shape[0]),
                seed=shared.seed,
            )
            ref_out = (
                _reference_output(
                    q=shared.q,
                    k_cache=shared.k_cache,
                    v_cache=shared.v_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=shared.cu_seqlens_q,
                )
                if args.check
                else None
            )
            runners: list[tuple[str, Callable[[], None], torch.Tensor]] = []
            run_2d, out_2d = _make_unified_runner(
                q=shared.q,
                k_cache=shared.k_cache,
                v_cache=shared.v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=shared.cu_seqlens_q,
                use_3d=False,
                num_segments=args.segments,
            )
            runners.append(("vllm-unified-2d", run_2d, out_2d))
            run_3d, out_3d = _make_unified_runner(
                q=shared.q,
                k_cache=shared.k_cache,
                v_cache=shared.v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=shared.cu_seqlens_q,
                use_3d=True,
                num_segments=args.segments,
            )
            runners.append(("vllm-unified-3d", run_3d, out_3d))
            for split in splits:
                run_split, out_split = _make_split_decode_runner(
                    q=shared.q,
                    k_cache=shared.k_cache,
                    v_cache=shared.v_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    num_splits=split,
                )
                runners.append((f"vllm-split-{split}", run_split, out_split))

            for backend, run, out in runners:
                graph = _capture_graph(run, warmup=args.warmup)
                times_us = _bench_graph(graph, replays=args.replays, l2_flush=l2_flush)
                summary = _summarize(times_us)
                check = {}
                if ref_out is not None:
                    check = {
                        "rel_l2": _relative_l2_error(out, ref_out),
                        "cos": _cosine_similarity(out, ref_out),
                    }
                record = {
                    "backend": backend,
                    "batch": batch,
                    "context_tokens": context,
                    "effective_cache_tokens": _decode_effective_cache_tokens(
                        context_tokens=context
                    ),
                    "times_us": times_us,
                    **summary,
                    **check,
                }
                records.append(record)
                check_suffix = (
                    f" | rel_l2={check['rel_l2']:.6f} cos={check['cos']:.8f}"
                    if check
                    else ""
                )
                print(
                    f"vllm-triton bs={batch:2d} ctx={context:6d} "
                    f"{backend:>16s} mean={summary['mean_us']:8.1f} us "
                    f"median={summary['median_us']:8.1f} us"
                    f"{check_suffix}"
                )
            torch.cuda.empty_cache()

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
