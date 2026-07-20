#!/usr/bin/env python3
"""Benchmark MiniMax-M3 sparse-attention indexer score/pool/select paths."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from sparkinfer.attention.indexer import (
    IndexerContiguousMetadata,
    IndexerPagedDecodeMetadata,
    build_paged_mqa_schedule_metadata,
    clear_indexer_caches,
    msa_q2k_indices_decode,
    msa_q2k_indices_prefill,
    quantize_msa_q_fp8,
)
from sparkinfer.attention.indexer.reference import pack_index_k_cache_reference

from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
    resolve_l2_flush_bytes,
)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _quantize_rows_to_kv_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scale = k.abs().amax(dim=1) / fp8_max
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (k / scale.unsqueeze(1)).clamp(-fp8_max, fp8_max)
    return quant.to(torch.float8_e4m3fn), scale.to(torch.float32)


def _make_q(
    *,
    rows: int,
    heads: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn((rows, heads, 128), generator=gen, dtype=torch.float32, device=device) / 3
    return quantize_msa_q_fp8(q)


def _make_decode_case(
    *,
    rows: int,
    heads: int,
    ctx_tokens: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, IndexerPagedDecodeMetadata]:
    q_fp8, q_scale = _make_q(rows=rows, heads=heads, seed=seed, device=device)
    page_size = 64
    pages = (ctx_tokens + page_size - 1) // page_size
    k_rows = pages * page_size
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32, device=device) / 3
    index_k_cache = pack_index_k_cache_reference(k)
    real_page_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
    real_page_table = real_page_table.expand(rows, pages).contiguous()
    seqlens = torch.full((rows,), ctx_tokens, dtype=torch.int32, device=device)
    metadata = IndexerPagedDecodeMetadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        paged_mqa_schedule_metadata=build_paged_mqa_schedule_metadata(seqlens, page_size),
    )
    return q_fp8, q_scale, index_k_cache, metadata


def _make_prefill_case(
    *,
    rows: int,
    heads: int,
    k_rows: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], IndexerContiguousMetadata]:
    q_fp8, q_scale = _make_q(rows=rows, heads=heads, seed=seed, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32, device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros((rows,), dtype=torch.int32, device=device)
    k_end = torch.arange(1, rows + 1, dtype=torch.int32, device=device).clamp(max=k_rows)
    metadata = IndexerContiguousMetadata(k_start=k_start, k_end=k_end)
    return q_fp8, q_scale, kv_fp8, metadata


def _bench_graph(fn, *, warmup: int, replays: int, l2_flush) -> float:
    graph = capture_cuda_graph(fn, warmup=warmup)
    stats = bench_cuda_graph(graph, replays=replays, l2_flush=l2_flush)
    return statistics.median(stats["replay_us"])


def _gbps(bytes_touched: int, us: float) -> float:
    if us <= 0:
        return float("nan")
    return float(bytes_touched) / (us * 1000.0)


def _run_decode(args: argparse.Namespace, device: torch.device, l2_flush) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for q_rows in args.rows:
        for heads in args.heads:
            for ctx_tokens in args.ctx:
                clear_indexer_caches()
                q_fp8, q_scale, index_k_cache, metadata = _make_decode_case(
                    rows=q_rows,
                    heads=heads,
                    ctx_tokens=ctx_tokens,
                    seed=args.seed + q_rows * 17 + heads,
                    device=device,
                )

                def run():
                    return msa_q2k_indices_decode(
                        q_fp8=q_fp8,
                        q_scale=q_scale,
                        index_k_cache=index_k_cache,
                        metadata=metadata,
                    )

                median_us = _bench_graph(run, warmup=args.warmup, replays=args.iters, l2_flush=l2_flush)
                bytes_touched = q_rows * heads * ctx_tokens * (128 + 4)
                rows_out.append(
                    {
                        "mode": "decode",
                        "rows": q_rows,
                        "heads": heads,
                        "ctx": ctx_tokens,
                        "us": median_us,
                        "gbps": _gbps(bytes_touched, median_us),
                    }
                )
    return rows_out


def _run_prefill(args: argparse.Namespace, device: torch.device, l2_flush) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for q_rows in args.rows:
        for heads in args.heads:
            for k_rows in args.ctx:
                clear_indexer_caches()
                q_fp8, q_scale, kv_fp8, metadata = _make_prefill_case(
                    rows=q_rows,
                    heads=heads,
                    k_rows=k_rows,
                    seed=args.seed + q_rows * 31 + heads,
                    device=device,
                )

                def run():
                    return msa_q2k_indices_prefill(
                        q_fp8=q_fp8,
                        q_scale=q_scale,
                        kv_fp8=kv_fp8,
                        metadata=metadata,
                    )

                median_us = _bench_graph(run, warmup=args.warmup, replays=args.iters, l2_flush=l2_flush)
                bytes_touched = q_rows * heads * k_rows * (128 + 4)
                rows_out.append(
                    {
                        "mode": "prefill",
                        "rows": q_rows,
                        "heads": heads,
                        "ctx": k_rows,
                        "us": median_us,
                        "gbps": _gbps(bytes_touched, median_us),
                    }
                )
    return rows_out


def _write_tsv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    header = ["mode", "rows", "heads", "ctx", "us", "gbps"]
    exists = path.exists()
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row[key]) for key in header) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--rows", type=_parse_csv_ints, default=[1, 4, 16, 64])
    parser.add_argument("--heads", type=_parse_csv_ints, default=[1, 4])
    parser.add_argument("--ctx", type=_parse_csv_ints, default=[8192, 32768, 131072, 262144])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=93_001)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results.msa_indexer.tsv"))
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    device = require_sm120()
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    if args.flush_l2:
        flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
        print(f"L2 flush: on ({flush_bytes / (1 << 20):.1f} MiB per replay)")
    else:
        print("L2 flush: off")
    rows = _run_decode(args, device, l2_flush) if args.mode == "decode" else _run_prefill(args, device, l2_flush)
    for row in rows:
        print(
            f"{row['mode']} rows={row['rows']} heads={row['heads']} ctx={row['ctx']} "
            f"{row['us']:.2f} us {row['gbps']:.1f} GB/s"
        )
    _write_tsv(args.output, rows)


if __name__ == "__main__":
    main()
