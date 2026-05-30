#!/usr/bin/env python3
"""End-to-end WO-projection decode latency at the DeepSeek-V4-Flash TP=2 shape.

groups=4, group_width=1024 (8 heads/group x 128), rank=1024, hidden=4096
(o_groups=8, num_heads=64, o_lora_rank=1024 at TP=2). Times the full
wo_projection (quant + wo_a + wo_b) under CUDA-graph replay at a decode token
count. Run on HEAD vs the pre-graft baseline to see the decode latency delta.

Run:  CUDA_VISIBLE_DEVICES=0 .../vllm-other/.venv/bin/python \
        benchmarks/probe_wo_decode_latency.py [tokens]
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmark_dense_gemm import bench_events, capture_graph_replay, make_l2_flush_fn
from b12x.gemm.wo_projection import (
    empty_wo_projection_workspace,
    quantize_wo_projection_weights_mxfp8_torch,
    wo_projection_mxfp8,
)

GROUPS, GROUP_WIDTH, RANK, HIDDEN = 4, 1024, 1024, 4096


def main():
    tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    torch.manual_seed(0)
    x = torch.randn((tokens, GROUPS, GROUP_WIDTH), device="cuda", dtype=torch.bfloat16) / 4
    wo_a = torch.randn((GROUPS, RANK, GROUP_WIDTH), device="cuda", dtype=torch.bfloat16) / GROUP_WIDTH**0.5
    wo_b = torch.randn((HIDDEN, GROUPS * RANK), device="cuda", dtype=torch.bfloat16) / (GROUPS * RANK) ** 0.5
    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a, wo_b)
    ws = empty_wo_projection_workspace(
        tokens, groups=GROUPS, group_width=GROUP_WIDTH, rank=RANK, hidden=HIDDEN, device="cuda"
    )
    l2 = make_l2_flush_fn(enabled=True, bytes_hint=0)

    def run():
        wo_projection_mxfp8(x, weights, ws)

    replay = capture_graph_replay(run)
    import statistics
    t = bench_events(replay, warmup=20, iters=100, l2_flush=l2)
    print(f"DSV4-Flash TP=2 wo_projection  tokens={tokens}  "
          f"median={statistics.median(t)*1000:.1f}us  min={min(t)*1000:.1f}us")


if __name__ == "__main__":
    main()
