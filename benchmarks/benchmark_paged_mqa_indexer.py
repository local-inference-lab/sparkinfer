#!/usr/bin/env python3
"""Benchmark the generic paged-MQA FP8 indexer logits path."""

from __future__ import annotations

import argparse
import statistics

import torch

from b12x.integration import (
    B12XAttentionWorkspace,
    clear_nsa_indexer_caches,
    pack_paged_mqa_index_k_cache_reference,
    paged_mqa_index_decode_logits_fp8,
    prepare_paged_mqa_indexer_metadata,
    resolve_replicated_num_q_heads,
)


def _make_page_table(
    *,
    rows: int,
    page_table_width: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    table = torch.full((rows, page_table_width), -1, dtype=torch.int32, device=device)
    pages_per_row = min((int(seq_len) + 63) // 64, int(page_table_width))
    for row in range(rows):
        start = row * int(page_table_width)
        table[row, :pages_per_row] = torch.arange(
            start,
            start + pages_per_row,
            dtype=torch.int32,
            device=device,
        )
    return table.contiguous()


def _cuda_time_us(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_evt.record()
        fn()
        stop_evt.record()
        torch.cuda.synchronize()
        samples.append(float(start_evt.elapsed_time(stop_evt)) * 1000.0)
    return statistics.median(samples), min(samples)


def _graph_time_us(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_evt.record()
        graph.replay()
        stop_evt.record()
        torch.cuda.synchronize()
        samples.append(float(start_evt.elapsed_time(stop_evt)) * 1000.0)
    return statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--global-heads", type=int, default=64)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--page-table-width", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=2304)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--eager", action="store_true", help="time eager launches instead of graph replay")
    parser.add_argument("--seed", type=int, default=91_100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    num_heads = resolve_replicated_num_q_heads(
        global_num_q_heads=args.global_heads,
        tensor_parallel_size=args.tp_size,
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)

    rows = int(args.rows)
    page_table_width = int(args.page_table_width)
    seq_len = int(args.seq_len)
    max_pages_needed = rows * page_table_width
    q_fp8 = (
        torch.randn((rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(device)
    index_k_cache = pack_paged_mqa_index_k_cache_reference(
        torch.randn((max_pages_needed * 64, 128), generator=gen, dtype=torch.float32).to(device)
        / 3
    )
    page_table = _make_page_table(
        rows=rows,
        page_table_width=page_table_width,
        seq_len=seq_len,
        device=device,
    )
    seqlens = torch.full((rows,), min(seq_len, page_table_width * 64), dtype=torch.int32, device=device)
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=512,
        max_page_table_width=page_table_width,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=index_k_cache.shape[0] * 64,
        page_size=64,
        use_cuda_graph=not args.eager,
    )
    schedule_out = None
    if workspace.paged_indexer_schedule_metadata_runtime is not None:
        schedule_out = workspace.paged_indexer_schedule_metadata_runtime
    metadata = prepare_paged_mqa_indexer_metadata(
        real_page_table=page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=num_heads,
        schedule_out=schedule_out,
    )

    clear_nsa_indexer_caches()

    def run() -> torch.Tensor:
        return paged_mqa_index_decode_logits_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            workspace=workspace,
        )

    # First call compiles the CuTe DSL kernel before timing or capture.
    out = run()
    torch.cuda.synchronize()
    if args.eager:
        median_us, min_us = _cuda_time_us(run, warmup=args.warmup, iters=args.iters)
        mode = "eager"
    else:
        median_us, min_us = _graph_time_us(run, warmup=args.warmup, iters=args.iters)
        mode = "graph"

    print(
        "paged_mqa_indexer "
        f"mode={mode} rows={rows} indexer_heads={num_heads} "
        f"page_table_width={page_table_width} seq_len={seq_len} "
        f"logits_shape={tuple(out.shape)} median_us={median_us:.2f} min_us={min_us:.2f}"
    )


if __name__ == "__main__":
    main()
