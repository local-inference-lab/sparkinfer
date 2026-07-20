"""NCU harness for isolated topk graph replay profiling."""

from __future__ import annotations

import argparse
import statistics

import cutlass
import torch
from cutlass import Int32

from sparkinfer.attention.nsa_indexer.tiled_topk import (
    SparseNSATiledTopkKernel,
    _run_cached_host_launcher,
    _tensor_meta_key,
    _to_kernel_tensor,
)
from sparkinfer.attention.nsa_indexer.persistent_topk import run_persistent_topk2048
from sparkinfer._lib.utils import current_cuda_stream


def _make_inputs(rows: int, cols: int, topk: int, seed: int):
    del topk
    torch.manual_seed(seed)
    scores = torch.randn((rows, cols), device="cuda", dtype=torch.float32)
    lengths = torch.full((rows,), cols, device="cuda", dtype=torch.int32)
    row_starts = torch.zeros((rows,), device="cuda", dtype=torch.int32)
    return scores, lengths, row_starts


def _make_cute_runner(scores: torch.Tensor, lengths: torch.Tensor, row_starts: torch.Tensor, topk: int):
    rows, cols = scores.shape
    values = torch.empty((rows, topk), dtype=torch.float32, device=scores.device)
    output = torch.empty((rows, topk), dtype=torch.int32, device=scores.device)
    flat_scores = scores.reshape(-1).contiguous()
    flat_values = values.reshape(-1).contiguous()
    flat_output = output.reshape(-1).contiguous()
    kernel = SparseNSATiledTopkKernel(is_tiled=False)

    def run():
        args = (
            _to_kernel_tensor(flat_scores, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(row_starts, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(lengths, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(flat_values, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(flat_output, cutlass.Int32, assumed_align=4),
            Int32(rows),
            Int32(cols),
            Int32(0),
            Int32(0),
            Int32(1),
            Int32(cols),
            Int32(topk),
            Int32(0),
            Int32(0),
            Int32(0),
            current_cuda_stream(),
        )
        cache_key = (
            _tensor_meta_key(flat_scores),
            _tensor_meta_key(row_starts),
            _tensor_meta_key(lengths),
            _tensor_meta_key(flat_values),
            _tensor_meta_key(flat_output),
            ("profile_topk_ncu_row", rows, cols, topk),
        )
        _run_cached_host_launcher(kernel, cache_key, args)

    return run, output


def _make_persistent_runner(scores: torch.Tensor, lengths: torch.Tensor, row_starts: torch.Tensor, topk: int):
    del row_starts
    if topk != 2048:
        raise ValueError("persistent mode supports topk=2048")
    output = torch.empty((scores.shape[0], topk), dtype=torch.int32, device=scores.device)

    def run():
        run_persistent_topk2048(
            scores,
            lengths,
            output_indices=output,
            max_seq_len=scores.shape[1],
        )

    return run, output


def _make_sgl_runner(scores: torch.Tensor, lengths: torch.Tensor, row_starts: torch.Tensor, topk: int):
    from sgl_kernel.top_k import fast_topk_transform_ragged_fused

    output_holder: list[torch.Tensor | None] = [None]

    def run():
        output_holder[0] = fast_topk_transform_ragged_fused(
            scores,
            lengths,
            row_starts,
            topk=topk,
            row_starts=row_starts,
        )

    return run, output_holder


def _capture(fn, warmup: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, replays: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for idx in range(replays):
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cute", "sgl", "persistent"), required=True)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-replays", type=int, default=1)
    parser.add_argument("--time-replays", type=int, default=20)
    parser.add_argument("--skip-profiler-range", action="store_true")
    args = parser.parse_args()

    if args.topk > args.cols:
        raise ValueError(f"topk ({args.topk}) must be <= cols ({args.cols})")

    scores, lengths, row_starts = _make_inputs(args.rows, args.cols, args.topk, args.seed)
    if args.mode == "cute":
        run, _ = _make_cute_runner(scores, lengths, row_starts, args.topk)
    elif args.mode == "persistent":
        run, _ = _make_persistent_runner(scores, lengths, row_starts, args.topk)
    else:
        run, _ = _make_sgl_runner(scores, lengths, row_starts, args.topk)

    graph = _capture(run, args.warmup)
    timings = _time_graph(graph, args.time_replays)
    print(
        "timing "
        f"mode={args.mode} rows={args.rows} cols={args.cols} topk={args.topk} "
        f"median_us={statistics.median(timings):.3f} min_us={min(timings):.3f} "
        f"max_us={max(timings):.3f}",
        flush=True,
    )

    torch.cuda.synchronize()
    if not args.skip_profiler_range:
        torch.cuda.profiler.start()
    for _ in range(args.profile_replays):
        graph.replay()
    torch.cuda.synchronize()
    if not args.skip_profiler_range:
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
