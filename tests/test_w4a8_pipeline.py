"""End-to-end oracle gates for the W4A8 throughput-tier pipeline driver.

w4a8_tier_forward (route_pack -> mxfp8 quant -> FC1 gather GEMM -> fused
silu+quant -> FC2 GEMM -> weighted topk sum) vs moe_reference_w4a8_mx on
synthetic MXFP4 experts, with the test_w4a8_mx_tp_moe.py gate set: nonzero
output, cosine > 0.998, norm ratio in (0.8, 1.25); plus bitwise run-to-run
determinism of the full pipeline.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe
from b12x.moe.fused.reference import moe_reference_w4a8_mx
from b12x.moe.fused.w4a8.pipeline import (
    build_w4a8_tier_workspace,
    prepare_w4a8_tier_weights,
    w4a8_tier_forward,
)
from b12x.moe.fused.w4a8.route import pack_routes_w4a8, w4a8_route_capacity

from .helpers import require_sm120


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("m,topk,e,group_rows,seed", [(200, 4, 7, 48, 5), (33, 6, 256, 48, 6)])
def test_w4a8_route_pack_oracle(
    m: int, topk: int, e: int, group_rows: int, seed: int
) -> None:
    """pack_routes_w4a8 vs a torch oracle: per-expert slot sets, group-padded
    offsets, padding-slot fill, per-group expert ids, and untouched tail."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    logits = torch.randn(m, e, generator=gen, device=device)
    topk_eff = min(topk, e)
    _, topk_ids = torch.topk(logits, topk_eff, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    numel = m * topk_eff

    cap_rows, cap_groups = w4a8_route_capacity(numel, e, group_rows)
    pri = torch.full(
        (cap_rows,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device
    )
    beids = torch.empty(cap_groups, dtype=torch.int32, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)
    eoff = torch.empty(e + 1, dtype=torch.int32, device=device)
    ecnt = torch.zeros(e, dtype=torch.int32, device=device)
    pack_routes_w4a8(
        topk_ids, e, group_rows,
        packed_route_indices=pri, block_expert_ids=beids,
        packed_route_count=count, expert_offsets=eoff, expert_counts=ecnt,
    )
    torch.cuda.synchronize()

    flat = topk_ids.reshape(-1).cpu()
    counts = torch.bincount(flat, minlength=e)
    padded = ((counts + group_rows - 1) // group_rows) * group_rows
    offsets = torch.zeros(e + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(padded, dim=0)
    total = int(offsets[-1])
    assert int(count.item()) == total
    pri_h = pri.cpu()
    for ex in range(e):
        start, cnt = int(offsets[ex]), int(counts[ex])
        got = set(pri_h[start : start + cnt].tolist())
        want = set((flat == ex).nonzero().flatten().tolist())
        assert got == want, f"expert {ex} slot set mismatch"
        pad = pri_h[start + cnt : start + int(padded[ex])]
        assert torch.all(pad == numel), f"expert {ex} padding fill"
    # Every slot beyond the live packed routes is filled with numel each
    # call (the post kernel covers the whole rounded buffer).
    assert torch.all(pri_h[total:] == numel)
    beids_h = beids.cpu()
    for g in range(cap_groups):
        row0 = g * group_rows
        want_e = -1
        for ex in range(e):
            if int(offsets[ex]) <= row0 < int(offsets[ex + 1]):
                want_e = ex
                break
        assert int(beids_h[g]) == want_e, f"group {g} expert id"


def _routed_inputs(m: int, e: int, k: int, topk: int, seed: int):
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = (torch.randn(m, k, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(m, e, generator=gen, device=device)
    topk_logits, topk_ids = torch.topk(logits, topk, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).float()
    return x, topk_ids.to(torch.int32), topk_weights


def _run_pipeline(e: int, k: int, n: int, m: int, topk: int, seed: int):
    device = torch.device("cuda")
    weights = make_synthetic_mxfp4_moe(e, k, n, seed=seed, device=device)
    x, topk_ids, topk_weights = _routed_inputs(m, e, k, topk, seed + 1)
    prep = prepare_w4a8_tier_weights(
        weights["w13_fp4"], weights["w13_mx"], weights["w2_fp4"], weights["w2_mx"]
    )
    ws = build_w4a8_tier_workspace(
        m=m,
        hidden_size=k,
        intermediate_size=n,
        num_experts=e,
        topk=topk,
        device=device,
    )
    out = w4a8_tier_forward(
        x,
        prep["w13_rp"],
        prep["w13_sfb"],
        prep["w2_rp"],
        prep["w2_sfb"],
        topk_ids,
        topk_weights,
        ws,
    )
    torch.cuda.synchronize()
    first = out.clone()
    # Bitwise determinism: a second pass over the same inputs must agree.
    out2 = w4a8_tier_forward(
        x,
        prep["w13_rp"],
        prep["w13_sfb"],
        prep["w2_rp"],
        prep["w2_sfb"],
        topk_ids,
        topk_weights,
        ws,
    )
    torch.cuda.synchronize()
    assert torch.equal(first, out2), "pipeline output is not deterministic"
    return first, weights, x, topk_ids, topk_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "e,k,n,m,topk,seed",
    [
        (8, 512, 256, 64, 4, 50),     # small dims
        (32, 4096, 1024, 32, 6, 60),  # DS4 dims, small m
    ],
)
def test_w4a8_tier_forward_matches_oracle(
    e: int, k: int, n: int, m: int, topk: int, seed: int
) -> None:
    require_sm120()
    out, weights, x, topk_ids, topk_weights = _run_pipeline(e, k, n, m, topk, seed)

    ref = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        e,
        k,
        n,
        activation="silu",
    )
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"w4a8 tier output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)
