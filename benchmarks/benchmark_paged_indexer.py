#!/usr/bin/env python3
"""Benchmark the vLLM-shaped paged-indexer FP8 top-k path."""

from __future__ import annotations

import argparse
import statistics

import torch

from b12x.attention.indexer import uses_paged_mqa_schedule
from b12x.attention.indexer.kernel import (
    run_paged_supertile_logits_kernel,
)
from b12x.attention.indexer import (
    B12XIndexerScratchCaps,
    INDEXER_SOURCE_LAYOUT_PAGED,
    clear_indexer_caches,
    index_topk_fp8,
    pack_paged_index_k_cache_reference,
    plan_indexer_scratch,
    prepare_paged_indexer_metadata,
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
    pages_per_row = min((int(seq_len) + 63) // 64, int(page_table_width))
    if int(page_stride) == 0:
        # vLLM's single-request prefill path expands one row without packing;
        # b12x receives a stride-0 row-shared page table.
        table = torch.full(
            (1, page_table_width), -1, dtype=torch.int32, device=device
        )
        table[0, :pages_per_row] = torch.arange(
            pages_per_row, dtype=torch.int32, device=device
        )
        return table.expand(rows, -1)

    table = torch.full((rows, page_table_width), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        start = row * int(page_stride)
        table[row, :pages_per_row] = torch.arange(
            start,
            start + pages_per_row,
            dtype=torch.int32,
            device=device,
        )
    return table.contiguous()


def _validate_analytic_topk(
    *,
    indices: torch.Tensor,
    scores: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
    analytic_scores: torch.Tensor,
) -> float:
    """Validate every target-shape row against a cheap analytic top-k oracle."""
    min_seq_len = int(seqlens.min().item())
    if topk > min_seq_len:
        raise ValueError(
            "analytic check requires topk <= every sequence length, got "
            f"{topk} > min_seqlen={min_seq_len}"
        )
    if not torch.isfinite(scores).all().item():
        raise AssertionError("top-k scores contain non-finite values")
    expected = seqlens[:, None] - topk + torch.arange(
        topk,
        dtype=torch.int32,
        device=indices.device,
    )
    actual_sorted = torch.sort(indices, dim=1).values
    if not torch.equal(actual_sorted, expected):
        mismatched = int((actual_sorted != expected).sum().item())
        raise AssertionError(
            f"analytic top-k index oracle mismatch: {mismatched} entries differ"
        )
    expected_scores = analytic_scores.index_select(
        0, indices.reshape(-1).to(torch.int64)
    ).view_as(scores)
    max_abs = float((scores - expected_scores).abs().max().item())
    torch.testing.assert_close(scores, expected_scores, rtol=2e-3, atol=2e-3)
    return max_abs


def _make_l2_flush(enabled: bool):
    """Evict L2 between timed replays so each measures HBM-bound (serving) speed,
    not L2-resident speed. Mirrors benchmark_moe: a 2x-L2 buffer, bitwise_not_ to
    write it through. Without this, replaying the same graph on the same inputs
    leaves the K-cache/page-table hot in L2 (128MB here) and inflates throughput."""
    if not enabled:
        return None
    try:
        from b12x.cute.utils import get_hardware_info

        l2_bytes = int(get_hardware_info().get_l2_cache_size_in_bytes())
    except Exception:
        l2_bytes = 0
    flush_bytes = (l2_bytes * 2) if l2_bytes > 0 else (128 << 20)
    buffer = torch.empty(flush_bytes, dtype=torch.uint8, device="cuda")

    def flush() -> None:
        buffer.bitwise_not_()

    return flush


def _event_time_us(fn, *, warmup: int, iters: int, l2_flush=None) -> list[float]:
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if l2_flush is not None:
            l2_flush()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(iters)]
    return samples


def _graph_time_us(
    fn,
    *,
    warmup: int,
    iters: int,
    l2_flush=None,
    nsys_capture: bool = False,
    torch_profile: bool = False,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        graph.replay()
    torch.cuda.synchronize()

    if nsys_capture:
        if l2_flush is not None:
            l2_flush()
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    if torch_profile:
        if l2_flush is not None:
            l2_flush()
            torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            graph.replay()
            torch.cuda.synchronize()
        print(
            profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=40
            )
        )

    # Flush L2 BEFORE start.record() each iter (stream-ordered, so excluded from the
    # timed interval); record all events and synchronize once (vs per-iter sync, which
    # serializes launches and adds its own jitter).
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if l2_flush is not None:
            l2_flush()
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(iters)]
    return samples


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
        "--cache-page-stride-bytes",
        type=int,
        default=0,
        help=(
            "physical byte stride between index-cache pages; 0 uses the packed "
            "page width. Some vLLM multi-group packed-KV allocations use a "
            "larger per-block stride shared by multiple cache layers"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("supertile-logits", "supertile-topk", "fused-topk"),
        default="supertile-topk",
    )
    parser.add_argument(
        "--route",
        choices=("auto", "packed-contiguous", "paged-tiled"),
        default="auto",
        help="force the paged prefill route; auto uses production policy",
    )
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument(
        "--fused-ctas",
        type=int,
        default=0,
        help="fused-topk: override ctas_per_group (0 = auto heuristic)",
    )
    parser.add_argument(
        "--fused-merge-threshold",
        type=int,
        default=-1,
        help="fused-topk cross-CTA merge auto-switch: seq_len<=thr uses last-CTA "
        "reduction, else cooperative radix. -1=topk-aware auto (default), "
        "0=force coop, large=force last-CTA.",
    )
    parser.add_argument(
        "--supertile-k",
        type=int,
        default=0,
        help="supertile width in K rows; 0 uses the production capacity-aware default",
    )
    parser.add_argument(
        "--persistent-ctas",
        type=int,
        default=0,
        help="benchmark-only override for paged scorer persistent CTAs",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--eager", action="store_true", help="time eager launches instead of graph replay")
    parser.add_argument(
        "--no-l2-flush",
        action="store_true",
        help="disable the 2x-L2 flush between timed replays (default: flush, "
        "for HBM-bound serving-realistic timings instead of L2-hot)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="use analytic inputs and validate every output row before timing",
    )
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="bracket one warmed graph replay with cudaProfilerStart/Stop",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="print per-kernel timings for one warmed graph replay",
    )
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
    cache_page_stride_bytes = int(args.cache_page_stride_bytes)
    logical_cache_page_bytes = 64 * 132
    if cache_page_stride_bytes == 0:
        cache_page_stride_bytes = logical_cache_page_bytes
    if cache_page_stride_bytes < logical_cache_page_bytes:
        raise ValueError(
            "cache_page_stride_bytes must be zero or at least one logical page "
            f"({logical_cache_page_bytes}), got {cache_page_stride_bytes}"
        )
    if page_stride == 0:
        max_pages_needed = page_table_width
    else:
        max_pages_needed = (rows - 1) * page_stride + page_table_width
    cache_tokens = max_pages_needed * 64
    analytic_scores = None
    if args.check:
        if str(args.mode) != "supertile-topk":
            raise ValueError("--check currently validates --mode supertile-topk")
        q_source = torch.zeros((rows, num_heads, 128), dtype=torch.float32, device=device)
        q_source[..., 0] = 1.0
        q_fp8 = q_source.to(torch.float8_e4m3fn)
        weights = torch.full(
            (rows, num_heads), 1.0 / num_heads, dtype=torch.float32, device=device
        )
        analytic_scores = torch.linspace(
            0.25, 1.0, cache_tokens, dtype=torch.float32, device=device
        )
        k_source = torch.zeros((cache_tokens, 128), dtype=torch.float32, device=device)
        k_source[:, 0] = analytic_scores
    else:
        q_fp8 = (
            torch.randn((rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device)
            / 2
        ).to(torch.float8_e4m3fn)
        weights = torch.randn(
            (rows, num_heads), generator=gen, dtype=torch.float32
        ).to(device)
        k_source = (
            torch.randn((cache_tokens, 128), generator=gen, dtype=torch.float32).to(device)
            / 3
        )
    packed_index_k_cache = pack_paged_index_k_cache_reference(k_source)
    # Match vLLM's rank-3 allocation, whose page bytes are planar
    # [64*128 quant][64*4 scales], then reproduce its zero-copy flattening for
    # b12x. The apparent [64, 132] shape is an allocation contract, not an
    # interleaved per-token byte layout.
    if cache_page_stride_bytes == logical_cache_page_bytes:
        vllm_kv_cache = packed_index_k_cache.view(max_pages_needed, 64, 132)
        index_k_cache = vllm_kv_cache.as_strided(
            (max_pages_needed, logical_cache_page_bytes),
            (int(vllm_kv_cache.stride(0)), 1),
        )
    else:
        # Packed multi-group allocators can give each layer a view into one
        # per-block allocation. Consecutive logical pages for that layer are
        # then separated by the aggregate bytes for every cache slot.
        backing_bytes = (
            (max_pages_needed - 1) * cache_page_stride_bytes
            + logical_cache_page_bytes
        )
        packed_backing = torch.empty(backing_bytes, dtype=torch.uint8, device=device)
        index_k_cache = torch.as_strided(
            packed_backing,
            size=(max_pages_needed, logical_cache_page_bytes),
            stride=(cache_page_stride_bytes, 1),
        )
        index_k_cache.copy_(packed_index_k_cache)
    page_table = _make_page_table(
        rows=rows,
        page_table_width=page_table_width,
        seq_len=seq_len,
        page_stride=page_stride,
        device=device,
    )
    bench_mode = str(args.mode)
    requested_route = str(args.route).replace("-", "_")
    topk = int(args.topk)
    requested_supertile_k = int(args.supertile_k)
    shared_page_table = page_stride == 0
    max_seq_len = min(seq_len, page_table_width * 64)
    if shared_page_table:
        # A vLLM chunked-prefill request exposes one shared page table and one
        # causal length per query token. For a 4k chunk ending at 16k these are
        # 12289..16384, rather than 4096 copies of the final length.
        seqlens = (
            max_seq_len
            - rows
            + 1
            + torch.arange(rows, dtype=torch.int32, device=device)
        ).clamp_(min=1, max=max_seq_len)
    else:
        seqlens = torch.full(
            (rows,), max_seq_len, dtype=torch.int32, device=device
        )
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=device,
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=num_heads,
            max_q_rows=rows,
            max_page_table_width=page_table_width,
            topk=topk,
            page_size=64,
            reserve_paged_logits=False,
            supertile_k=requested_supertile_k,
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=shared_page_table,
            route=(
                "paged_tiled"
                if bench_mode == "supertile-logits" and requested_route == "auto"
                else requested_route
            ),
        ),
    )
    supertile_k = int(plan.layout.supertile_tokens)
    if int(args.persistent_ctas) > 0:
        raise ValueError("--persistent-ctas was removed with the workspace-backed indexer path")
    schedule_out = None
    build_schedule = None
    if bench_mode == "supertile-topk":
        build_schedule = False
    else:
        build_schedule = uses_paged_mqa_schedule(
            q_rows=rows,
            max_pages=page_table_width,
        )
    metadata = prepare_paged_indexer_metadata(
        real_page_table=page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=num_heads,
        schedule_out=schedule_out,
        build_schedule=build_schedule,
        shared_page_table=shared_page_table,
    )
    scratch_specs = plan.shapes_and_dtypes()
    scratch = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in scratch_specs
    ]
    binding = plan.bind(
        scratch=scratch,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        schedule_metadata=metadata.schedule_metadata,
        expected_num_q_heads=num_heads,
        shared_page_table=shared_page_table,
    )

    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    out_scores = torch.empty((rows, topk), dtype=torch.float32, device=device)

    clear_indexer_caches()

    def run() -> torch.Tensor:
        if bench_mode == "supertile-logits":
            tile_logits = binding.scratch.get_indexer_contiguous_tile_logits()
            return run_paged_supertile_logits_kernel(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=index_k_cache,
                real_page_table=metadata.real_page_table,
                seqlens_per_query=metadata.cache_seqlens_int32,
                active_width=binding.active_width,
                tile_logits=tile_logits,
                source_page_offset=0,
                output_width_tokens=supertile_k,
                preinitialize_tile_logits=False,
            )
        if bench_mode == "fused-topk":
            from b12x.attention.indexer.kernel import _split_index_k_cache_runtime_views
            from b12x.attention.indexer.fused_indexer import run_fused_paged_indexer

            quant, scales = _split_index_k_cache_runtime_views(index_k_cache)
            fused_ctas = int(args.fused_ctas) if int(args.fused_ctas) > 0 else None
            return run_fused_paged_indexer(
                q_bytes=q_fp8.view(torch.uint8),
                weights=weights,
                k_quant_bytes=quant,
                k_scales=scales,
                real_page_table=page_table,
                seqlens=seqlens,
                num_heads=num_heads,
                topk=topk,
                ctas_per_group=fused_ctas,
                merge_threshold=(
                    None if args.fused_merge_threshold < 0
                    else int(args.fused_merge_threshold)
                ),
            )[0]
        return index_topk_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            binding=binding,
            topk=topk,
            expected_num_q_heads=num_heads,
            out_indices=out_indices,
            out_scores=out_scores,
            supertile_k=supertile_k,
        )

    # First call compiles the CuTe DSL kernel before timing or capture.
    out = run()
    torch.cuda.synchronize()
    oracle_max_abs = None
    if args.check:
        assert analytic_scores is not None
        oracle_max_abs = _validate_analytic_topk(
            indices=out_indices,
            scores=out_scores,
            seqlens=seqlens,
            topk=topk,
            analytic_scores=analytic_scores,
        )
    l2_flush = _make_l2_flush(not args.no_l2_flush)
    if args.eager:
        samples_us = _event_time_us(
            run, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush
        )
        timing_mode = "eager"
    else:
        samples_us = _graph_time_us(
            run,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
            nsys_capture=bool(args.nsys_capture),
            torch_profile=bool(args.torch_profile),
        )
        timing_mode = "graph"

    median_us = statistics.median(samples_us)
    min_us = min(samples_us)
    oracle_state = (
        f"analytic_pass(max_abs={oracle_max_abs:.3g})"
        if oracle_max_abs is not None
        else "not_run"
    )
    raw_us = ",".join(f"{sample:.2f}" for sample in samples_us)

    print(
        "paged_indexer "
        f"mode={bench_mode} timing={timing_mode} rows={rows} indexer_heads={num_heads} "
        f"page_table_width={page_table_width} seq_len={seq_len} "
        f"seqlen_range={int(seqlens.min().item())}-{int(seqlens.max().item())} "
        f"page_stride={page_stride} cache_page_stride_bytes={cache_page_stride_bytes} "
        f"cache_span_mib={((max_pages_needed - 1) * cache_page_stride_bytes + logical_cache_page_bytes) / (1024 * 1024):.2f} "
        f"topk={topk} supertile_k={supertile_k} "
        f"requested_supertile_k={requested_supertile_k} "
        f"route={plan.layout.route} prefill_block_k={plan.layout.prefill_block_k} "
        f"scratch_mib={plan.layout.nbytes / (1024 * 1024):.2f} "
        f"output_shape={tuple(out.shape)} correctness={oracle_state} "
        f"median_us={median_us:.2f} min_us={min_us:.2f} raw_us=[{raw_us}]"
    )


if __name__ == "__main__":
    main()
