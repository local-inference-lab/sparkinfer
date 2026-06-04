#!/usr/bin/env python3
"""Benchmark the compressed-indexer FP8 logits path."""

from __future__ import annotations

import argparse
import statistics

import torch

from b12x.attention.indexer import uses_paged_mqa_schedule
from b12x.attention.indexer.kernel import (
    run_paged_windowed_tiled_logits_kernel,
)
from b12x.integration import (
    B12XAttentionWorkspace,
    clear_indexer_caches,
    pack_compressed_index_k_cache_reference,
    index_topk_fp8,
    prepare_compressed_indexer_metadata,
    resolve_replicated_num_q_heads,
)


def _make_page_table(
    *,
    rows: int,
    page_table_width: int,
    seq_len: int,
    page_stride: int,
    device: torch.device,
) -> torch.Tensor:
    table = torch.full((rows, page_table_width), -1, dtype=torch.int32, device=device)
    pages_per_row = min((int(seq_len) + 63) // 64, int(page_table_width))
    for row in range(rows):
        start = row * int(page_stride)
        table[row, :pages_per_row] = torch.arange(
            start,
            start + pages_per_row,
            dtype=torch.int32,
            device=device,
        )
    return table.contiguous()


def _event_time_us(fn, *, warmup: int, iters: int) -> tuple[float, float]:
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
    parser.add_argument(
        "--page-stride",
        type=int,
        default=0,
        help="physical page-id stride between rows; 0 shares pages across rows",
    )
    parser.add_argument(
        "--mode",
        choices=("supertile-logits", "supertile-topk", "fused-topk"),
        default="supertile-topk",
    )
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument(
        "--fused-ctas",
        type=int,
        default=0,
        help="fused-topk: override ctas_per_group (0 = auto heuristic)",
    )
    parser.add_argument(
        "--fused-coop",
        type=int,
        default=1,
        help="fused-topk: 1=cooperative grid-barrier merge, 0=last-CTA reduction",
    )
    parser.add_argument("--supertile-k", type=int, default=32768)
    parser.add_argument(
        "--persistent-ctas",
        type=int,
        default=0,
        help="benchmark-only override for paged scorer persistent CTAs",
    )
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
    page_stride = int(args.page_stride)
    if page_stride < 0:
        raise ValueError(f"page_stride must be non-negative, got {page_stride}")
    if page_stride == 0:
        max_pages_needed = page_table_width
    else:
        max_pages_needed = (rows - 1) * page_stride + page_table_width
    q_fp8 = (
        torch.randn((rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(device)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((max_pages_needed * 64, 128), generator=gen, dtype=torch.float32).to(device)
        / 3
    )
    page_table = _make_page_table(
        rows=rows,
        page_table_width=page_table_width,
        seq_len=seq_len,
        page_stride=page_stride,
        device=device,
    )
    seqlens = torch.full((rows,), min(seq_len, page_table_width * 64), dtype=torch.int32, device=device)
    bench_mode = str(args.mode)
    topk = int(args.topk)
    supertile_k = int(args.supertile_k)
    reserve_dense_logits = False  # dense logits/top-k paths removed; unified entry is supertile/fused
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=page_table_width,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=index_k_cache.shape[0] * 64,
        page_size=64,
        use_cuda_graph=not args.eager,
        reserve_paged_indexer_logits=reserve_dense_logits,
        paged_indexer_logits_q_rows=rows if reserve_dense_logits else 0,
        paged_indexer_logits_k_rows=page_table_width * 64 if reserve_dense_logits else 0,
        paged_indexer_tile_logits_k_rows=(
            supertile_k if bench_mode in {"supertile-logits", "supertile-topk"} else 0
        ),
    )
    if int(args.persistent_ctas) > 0:
        persistent_ctas = int(args.persistent_ctas)

        def _benchmark_paged_indexer_persistent_ctas() -> int:
            return persistent_ctas

        workspace.get_paged_indexer_persistent_ctas = _benchmark_paged_indexer_persistent_ctas  # type: ignore[method-assign]
    schedule_out = None
    build_schedule = None
    if bench_mode == "supertile-topk":
        build_schedule = False
    else:
        build_schedule = uses_paged_mqa_schedule(
            q_rows=rows,
            max_pages=page_table_width,
        )
    if build_schedule and workspace.paged_indexer_schedule_metadata_runtime is not None:
        schedule_out = workspace.paged_indexer_schedule_metadata_runtime
    metadata = prepare_compressed_indexer_metadata(
        real_page_table=page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=num_heads,
        schedule_out=schedule_out,
        build_schedule=build_schedule,
        shared_page_table=page_stride == 0,
    )

    clear_indexer_caches()
    if bench_mode in {"supertile-logits", "supertile-topk"}:
        workspace.prewarm_paged_indexer_tiled_topk()
        workspace.prewarm_paged_indexer_tiled_scorer(
            index_k_cache=index_k_cache,
            width_tokens=supertile_k,
        )

    def run() -> torch.Tensor:
        if bench_mode == "supertile-logits":
            tile_logits = workspace.get_indexer_extend_tile_logits()
            if tile_logits is None:
                raise RuntimeError("supertile-logits requires tiled-logits workspace")
            return run_paged_windowed_tiled_logits_kernel(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=index_k_cache,
                real_page_table=metadata.real_page_table,
                seqlens_per_query=metadata.cache_seqlens_int32,
                active_width=workspace.get_paged_indexer_active_width_cap(),
                tile_logits=tile_logits,
                source_page_offset=0,
                output_width_tokens=supertile_k,
                workspace=workspace,
                preinitialize_tile_logits=False,
                contract_phantoms=workspace.get_paged_indexer_contract_phantoms(),
                stage_runtime_metadata=False,
            )
        if bench_mode == "fused-topk":
            from b12x.attention.indexer.kernel import _split_index_k_cache_runtime_views
            from b12x.attention.indexer.fused_indexer import run_fused_indexer_c4

            quant, scales = _split_index_k_cache_runtime_views(index_k_cache)
            fused_ctas = int(args.fused_ctas) if int(args.fused_ctas) > 0 else None
            return run_fused_indexer_c4(
                q_bytes=q_fp8.view(torch.uint8),
                weights=weights,
                k_quant_bytes=quant,
                k_scales=scales,
                real_page_table=page_table,
                seqlens=seqlens,
                num_heads=num_heads,
                topk=topk,
                ctas_per_group=fused_ctas,
                coop_merge=bool(args.fused_coop),
            )[0]
        return index_topk_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=num_heads,
            workspace=workspace,
            supertile_k=supertile_k,
        )

    # First call compiles the CuTe DSL kernel before timing or capture.
    out = run()
    torch.cuda.synchronize()
    if args.eager:
        median_us, min_us = _event_time_us(run, warmup=args.warmup, iters=args.iters)
        timing_mode = "eager"
    else:
        median_us, min_us = _graph_time_us(run, warmup=args.warmup, iters=args.iters)
        timing_mode = "graph"

    print(
        "compressed_indexer "
        f"mode={bench_mode} timing={timing_mode} rows={rows} indexer_heads={num_heads} "
        f"page_table_width={page_table_width} seq_len={seq_len} "
        f"page_stride={page_stride} topk={topk} supertile_k={supertile_k} "
        f"logits_shape={tuple(out.shape)} median_us={median_us:.2f} min_us={min_us:.2f}"
    )


if __name__ == "__main__":
    main()
