#!/usr/bin/env python3
"""Benchmark graph-replayed MiniMax-MSA paged decode attention."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from b12x.integration.attention import (
    B12XPagedAttentionScratchCaps,
    clear_attention_caches,
    paged_attention_forward,
    plan_paged_attention_scratch,
)


MSA_TOPK = 16
MSA_BLOCK_TOKENS = 128


def _parse_csv_ints(value: str) -> list[int]:
    vals = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(val <= 0 for val in vals):
        raise argparse.ArgumentTypeError("all values must be positive")
    return vals


def _msa_effective_selected_tokens(cache_len: int, *, page_size: int = 64) -> int:
    visible_blocks = max((int(cache_len) + MSA_BLOCK_TOKENS - 1) // MSA_BLOCK_TOKENS, 1)
    selected_blocks = min(MSA_TOPK, visible_blocks)
    if selected_blocks <= 1:
        tail_tokens = max(int(cache_len), 1)
        return min(
            max((tail_tokens + page_size - 1) // page_size, 1),
            MSA_BLOCK_TOKENS // page_size,
        ) * page_size
    tail_block_start = (visible_blocks - 1) * MSA_BLOCK_TOKENS
    tail_tokens = max(int(cache_len) - tail_block_start, 1)
    tail_pages = min(
        max((tail_tokens + page_size - 1) // page_size, 1),
        MSA_BLOCK_TOKENS // page_size,
    )
    return ((selected_blocks - 1) * (MSA_BLOCK_TOKENS // page_size) + tail_pages) * page_size


def _make_uniform_inputs(
    *,
    batch: int,
    cache_len: int,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    device = "cuda"
    pages_per_req = (int(cache_len) + int(page_size) - 1) // int(page_size)
    num_pages = batch * pages_per_req
    q = torch.randn(batch, q_heads, head_dim, device=device, dtype=dtype) / 4
    k_cache = (
        torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype)
        / 4
    )
    v_cache = (
        torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype)
        / 4
    )
    page_table = torch.empty(batch, pages_per_req, dtype=torch.int32, device=device)
    page_order = torch.randperm(num_pages, device=device)
    for req in range(batch):
        start = req * pages_per_req
        page_table[req] = page_order[start : start + pages_per_req].to(torch.int32)
    cache_seqlens = torch.full((batch,), int(cache_len), dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q


def _make_msa_q2k_indices(
    *,
    batch: int,
    cache_len: int,
    kv_heads: int,
    device: torch.device,
) -> torch.Tensor:
    visible_blocks = max((int(cache_len) + MSA_BLOCK_TOKENS - 1) // MSA_BLOCK_TOKENS, 1)
    count = min(MSA_TOPK, visible_blocks)
    if count == visible_blocks:
        selected = list(range(visible_blocks))
    else:
        selected = sorted(
            {
                int(round(i * (visible_blocks - 1) / max(count - 1, 1)))
                for i in range(count)
            }
        )
        cursor = visible_blocks - 1
        while len(selected) < count:
            if cursor not in selected:
                selected.append(cursor)
            cursor -= 1
        selected = sorted(selected[:count])
    q2k = torch.full(
        (kv_heads, batch, MSA_TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    values = torch.tensor(selected, dtype=torch.int32, device=device)
    q2k[:, :, : len(selected)] = values.view(1, 1, -1)
    return q2k.contiguous()


def _capture_graph(fn, *, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(max(int(warmup), 0)):
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
    l2_flush,
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
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends, strict=True)]


def _make_scratch_plan(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    max_work_items: int,
    max_partial_rows: int,
    msa_block_sparse: bool,
) -> object:
    return plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=q.shape[0],
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(int(max_work_items), 1),
            max_partial_rows=max(int(max_partial_rows), 0),
            num_cache_pages=k_cache.shape[0],
            use_cuda_graph=True,
            msa_block_sparse=msa_block_sparse,
        )
    )


def _capture_msa_case(
    *,
    batch: int,
    cache_len: int,
    chunk_pages: int,
    seed: int,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, int, int]:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_uniform_inputs(
        batch=batch,
        cache_len=cache_len,
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=seed,
    )
    q2k_indices = _make_msa_q2k_indices(
        batch=batch,
        cache_len=cache_len,
        kv_heads=4,
        device=q.device,
    )
    active_chunks_per_req = (MSA_TOPK * MSA_BLOCK_TOKENS + int(chunk_pages) * 64 - 1) // (
        int(chunk_pages) * 64
    )
    # MSA decode graph replay intentionally regularizes block-valid capacity to
    # the worst-case 64-token chunk fanout so all chunk policies share a stable
    # metadata shape.
    max_chunks_per_req = max(active_chunks_per_req, 32)
    scratch_plan = _make_scratch_plan(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        max_work_items=batch * max(max_chunks_per_req, 1),
        max_partial_rows=batch * max(max_chunks_per_req, 1),
        msa_block_sparse=True,
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=batch,
        max_page_table_width=page_table.shape[1],
        max_cache_page_count=page_table.shape[1],
        fixed_split_size=int(chunk_pages),
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
    )
    graph = _capture_graph(lambda: paged_attention_forward(binding=binding), warmup=warmup)
    selected_tokens = _msa_effective_selected_tokens(cache_len)
    active_chunks_per_req = (selected_tokens + int(chunk_pages) * 64 - 1) // (
        int(chunk_pages) * 64
    )
    launch_ctas = batch * 4 * active_chunks_per_req
    bytes_read = batch * 4 * selected_tokens * 128 * 2 * 2
    return graph, launch_ctas, bytes_read


def _capture_dense_case(
    *,
    batch: int,
    cache_len: int,
    seed: int,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, int, int]:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_uniform_inputs(
        batch=batch,
        cache_len=cache_len,
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=seed,
    )
    pages = int(page_table.shape[1])
    scratch_plan = _make_scratch_plan(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        max_work_items=max(batch * pages, batch),
        max_partial_rows=0,
        msa_block_sparse=False,
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=batch,
        max_page_table_width=pages,
        max_cache_page_count=pages,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    graph = _capture_graph(lambda: paged_attention_forward(binding=binding), warmup=warmup)
    launch_ctas = batch * 4
    bytes_read = batch * 4 * int(cache_len) * 128 * 2 * 2
    return graph, launch_ctas, bytes_read


def _summarize(samples_us: list[float]) -> tuple[float, float, float]:
    return (
        statistics.mean(samples_us),
        statistics.median(samples_us),
        min(samples_us),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=_parse_csv_ints, default="1,4,16")
    parser.add_argument("--contexts", type=_parse_csv_ints, default="32768,131072,524288,1048576")
    parser.add_argument("--chunks", type=_parse_csv_ints, default="64,128,256,512")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results.msa_decode.tsv"))
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    clear_attention_caches()
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "backend\tbatch\tcontext\tchunk_tokens\tmean_us\tmedian_us\tmin_us\t"
        "effective_gbs\tlaunch_ctas\tbytes_read"
    )
    rows = [header]
    print(f"L2 flush: {'on' if l2_flush is not None else 'off'} ({l2_flush_bytes} bytes)")

    for batch in args.batches:
        for context in args.contexts:
            if not args.skip_dense:
                graph, launch_ctas, bytes_read = _capture_dense_case(
                    batch=batch,
                    cache_len=context,
                    seed=args.seed + batch * 17 + context,
                    warmup=args.warmup,
                )
                samples = _bench_graph(graph, replays=args.replays, l2_flush=l2_flush)
                mean_us, median_us, min_us = _summarize(samples)
                gbs = bytes_read / (mean_us * 1e-6) / 1e9
                rows.append(
                    f"dense\t{batch}\t{context}\t0\t{mean_us:.3f}\t{median_us:.3f}\t"
                    f"{min_us:.3f}\t{gbs:.3f}\t{launch_ctas}\t{bytes_read}"
                )
                print(rows[-1])
            for chunk_tokens in args.chunks:
                if chunk_tokens % 64 != 0:
                    raise ValueError("MSA chunk tokens must be multiples of 64")
                graph, launch_ctas, bytes_read = _capture_msa_case(
                    batch=batch,
                    cache_len=context,
                    chunk_pages=chunk_tokens // 64,
                    seed=args.seed + batch * 31 + context + chunk_tokens,
                    warmup=args.warmup,
                )
                samples = _bench_graph(graph, replays=args.replays, l2_flush=l2_flush)
                mean_us, median_us, min_us = _summarize(samples)
                gbs = bytes_read / (mean_us * 1e-6) / 1e9
                rows.append(
                    f"msa\t{batch}\t{context}\t{chunk_tokens}\t{mean_us:.3f}\t{median_us:.3f}\t"
                    f"{min_us:.3f}\t{gbs:.3f}\t{launch_ctas}\t{bytes_read}"
                )
                print(rows[-1])
            args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
