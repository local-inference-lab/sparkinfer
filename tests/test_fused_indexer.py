"""Parity tests for the unified fused indexer kernel (score + top-k, paged path).

The fused kernel scores q·k via the paged mxfp8 MMA and selects the row-wise
top-k in one launch (no global logits blob). Golden = the exact scorer math
(e4m3 q·k → ReLU·weight → sum heads → ×k_scale) followed by torch.topk, which
is the same contract the production scorer + tiled_topk path satisfies.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention.indexer.fused_indexer import (
    _COOP_STATE_WORDS,
    fused_indexer_scratch_capacity,
    resolve_fused_indexer_path,
    run_fused_paged_indexer,
    run_fused_indexer_mla,
)

_PS = 64  # compressed-index page size


@pytest.mark.parametrize(
    "heads,width,rows,expected",
    [
        (32, 8192, 1, True),
        (32, 8192, 2, False),
        (32, 16384, 5, True),
        (32, 16384, 6, False),
        (32, 32768, 5, True),
        (32, 32768, 6, False),
        (32, 65536, 6, True),
        (64, 16384, 6, True),
    ],
)
def test_fused_indexer_route_uses_capture_static_glm_buckets(
    heads, width, rows, expected
):
    assert (
        resolve_fused_indexer_path(
            topk=2048,
            num_rows=rows,
            width=width,
            num_heads=heads,
        )
        is expected
    )


def _build_case(rows, heads, seqlen, topk, *, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pr = (seqlen + _PS - 1) // _PS
    npages = rows * pr
    q_fp8 = (torch.randn((rows, heads, 128), generator=g) / 3).to(torch.float8_e4m3fn).to(device)
    weights = torch.randn((rows, heads), generator=g, dtype=torch.float32).to(device)
    k_fp8 = (torch.randn((npages, _PS, 128), generator=g) / 3).to(torch.float8_e4m3fn).to(device)
    k_scales = torch.rand((npages, _PS), generator=g, dtype=torch.float32).to(device) + 0.1
    page_table = torch.arange(npages, dtype=torch.int32, device=device).view(rows, pr).contiguous()
    seqlens = torch.full((rows,), seqlen, dtype=torch.int32, device=device)
    return q_fp8, weights, k_fp8, k_scales, page_table, seqlens


def _golden_topk(q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk):
    rows, seqlen = int(q_fp8.shape[0]), int(seqlens[0])
    qf, kf = q_fp8.float(), k_fp8.float()
    vals, idxs = [], []
    for r in range(rows):
        pages = page_table[r].long()
        kr = kf[pages].reshape(-1, 128)[:seqlen]
        sc = k_scales[pages].reshape(-1)[:seqlen]
        logit = (torch.relu(torch.einsum("hd,td->ht", qf[r], kr)) * weights[r].unsqueeze(1)).sum(0) * sc
        tk = torch.topk(logit, topk, largest=True, sorted=True)
        vals.append(tk.values)
        idxs.append(set(tk.indices.tolist()))
    return torch.stack(vals), idxs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused indexer")
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("rows,seqlen", [(1, 4096), (2, 4096), (4, 8192)])
def test_fused_indexer_paged_matches_reference(topk, rows, seqlen, monkeypatch):
    if seqlen <= topk:
        seqlen = topk * 2  # ensure the radix path (not just no-selection copy) engages
    device = torch.device("cuda")
    heads = 16
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=7, device=device
    )
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
    )

    assert idx.shape == (rows, topk)
    assert bool((idx >= 0).all())  # every slot filled (seqlen > topk)
    # value-multiset parity (tie-robust); fp8 e4m3 dot in f32 matches the MMA.
    fused_sorted = torch.sort(val, dim=1, descending=True).values
    assert torch.allclose(fused_sorted, gold_vals, atol=1e-2, rtol=0)
    # exact selected-index set per row (random fp8 logits have no rank-k ties here)
    for r in range(rows):
        assert set(idx[r].tolist()) == gold_idx_sets[r]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused indexer")
def test_fused_indexer_paged_emits_native_physical_slots():
    device = torch.device("cuda")
    rows, heads, topk, seqlen = 2, 16, 512, 1024
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=71, device=device
    )
    page_table = torch.flip(page_table, dims=(1,)).contiguous()
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
        output_physical_slots=True,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
    )
    assert torch.allclose(
        torch.sort(val, dim=1, descending=True).values,
        gold_vals,
        atol=1e-2,
        rtol=0,
    )
    for row in range(rows):
        logical = torch.tensor(
            sorted(gold_idx_sets[row]), dtype=torch.int64, device=device
        )
        page_ids = page_table[row, torch.div(logical, _PS, rounding_mode="floor")]
        physical = page_ids * _PS + torch.remainder(logical, _PS)
        assert set(idx[row].tolist()) == set(physical.tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused indexer")
@pytest.mark.parametrize("seqlen", [4101, 8255, 5377])
def test_fused_indexer_paged_partial_last_page(seqlen):
    # seqlen % 64 != 0 -> the last page is partial (valid_slots < 64). Exercises
    # the masked-tail path (no partial-logits pre-zero must still be exact).
    device = torch.device("cuda")
    rows, heads, topk = 2, 16, 2048
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=23, device=device
    )
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8), weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(), k_scales=k_scales,
        real_page_table=page_table, seqlens=seqlens, num_heads=heads, topk=topk,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
    )
    fused_sorted = torch.sort(val, dim=1, descending=True).values
    assert torch.allclose(fused_sorted, gold_vals, atol=1e-2, rtol=0)
    for r in range(rows):
        assert set(idx[r].tolist()) == gold_idx_sets[r]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused indexer")
def test_fused_indexer_paged_short_context_no_radix():
    # seqlen <= topk: every token is selected (the no-selection fold path).
    device = torch.device("cuda")
    rows, heads, topk, seqlen = 2, 16, 512, 64
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=11, device=device
    )
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
    )
    torch.cuda.synchronize(device)
    for r in range(rows):
        valid = idx[r][idx[r] >= 0]
        assert valid.numel() == seqlen
        assert set(valid.tolist()) == set(range(seqlen))
        assert int((idx[r] == -1).sum()) == topk - seqlen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused indexer")
def test_fused_indexer_paged_strided_q_scalar_load_fallback():
    device = torch.device("cuda")
    rows, heads, topk, seqlen = 2, 16, 512, 1024
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=17, device=device
    )
    padded_q = torch.empty(
        (rows, heads, 129), dtype=torch.float8_e4m3fn, device=device
    )
    padded_q[..., :128].copy_(q_fp8)
    strided_q = padded_q[..., :128]
    assert strided_q.stride(1) == 129

    idx, val = run_fused_paged_indexer(
        q_bytes=strided_q.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        strided_q, weights, k_fp8, k_scales, page_table, seqlens, topk
    )
    fused_sorted = torch.sort(val, dim=1, descending=True).values
    assert torch.allclose(fused_sorted, gold_vals, atol=1e-2, rtol=0)
    for row in range(rows):
        assert set(idx[row].tolist()) == gold_idx_sets[row]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused indexer")
@pytest.mark.parametrize("ctas_per_group", [96, 188])
def test_fused_indexer_paged_long_context_merge_exact(ctas_per_group):
    # Long context + many CTAs stresses the in-kernel last-CTA merge: each CTA's
    # local top-k is a cluster of high values, so the merged radix threshold bin
    # is the worst case for the bounded SMEM candidate buffer. Must stay exact.
    device = torch.device("cuda")
    rows, heads, topk, seqlen = 2, 16, 2048, 65536
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, seqlen, topk, seed=3, device=device
    )
    idx, val = run_fused_paged_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
        ctas_per_group=ctas_per_group,
    )
    torch.cuda.synchronize(device)
    gold_vals, gold_idx_sets = _golden_topk(
        q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
    )
    fused_sorted = torch.sort(val, dim=1, descending=True).values
    assert torch.allclose(fused_sorted, gold_vals, atol=1e-2, rtol=0)
    for r in range(rows):
        assert set(idx[r].tolist()) == gold_idx_sets[r]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused indexer")
def test_fused_indexer_preinitialized_state_graph_replay_switches_merge_mode():
    """One initialized workspace remains exact across serial/coop graph replays."""
    device = torch.device("cuda")
    rows, heads, topk, max_seqlen = 2, 32, 512, 4097
    ctas_per_group = 8
    merge_threshold = 3000
    q_fp8, weights, k_fp8, k_scales, page_table, seqlens = _build_case(
        rows, heads, max_seqlen, topk, seed=41, device=device
    )
    pack_capacity, _ = fused_indexer_scratch_capacity(
        rows, topk, torch.cuda.get_device_properties(device).multi_processor_count
    )
    pack_values = torch.empty(pack_capacity, dtype=torch.float32, device=device)
    pack_indices = torch.empty(pack_capacity, dtype=torch.int32, device=device)
    merge_state = torch.zeros(
        rows * _COOP_STATE_WORDS, dtype=torch.int32, device=device
    )
    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    out_values = torch.empty((rows, topk), dtype=torch.float32, device=device)

    kwargs = dict(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(),
        k_scales=k_scales,
        real_page_table=page_table,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
        out_indices=out_indices,
        out_values=out_values,
        ctas_per_group=ctas_per_group,
        merge_threshold=merge_threshold,
        pack_values=pack_values,
        pack_indices=pack_indices,
        merge_state=merge_state,
        merge_state_preinitialized=True,
    )
    # Compile and exercise one launch before capture. The kernel must restore
    # its counters; no memset is part of the captured graph below.
    run_fused_paged_indexer(**kwargs)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fused_paged_indexer(**kwargs)

    # Alternate across the runtime dispatch boundary and repeat both arms. This
    # catches stale arrival/output/total/cleanup counters on graph replay.
    for live_seqlen in (2048, max_seqlen, max_seqlen, 2048, max_seqlen):
        seqlens.fill_(live_seqlen)
        graph.replay()
        torch.cuda.synchronize(device)
        gold_values, gold_idx_sets = _golden_topk(
            q_fp8, weights, k_fp8, k_scales, page_table, seqlens, topk
        )
        fused_sorted = torch.sort(out_values, dim=1, descending=True).values
        assert torch.allclose(fused_sorted, gold_values, atol=1e-2, rtol=0)
        for row in range(rows):
            assert set(out_indices[row].tolist()) == gold_idx_sets[row]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused indexer")
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_fused_indexer_mla_matches_reference(rows):
    # FLAT/MLA: contiguous K, per-row [k_start, k_end) windows -> absolute indices.
    device = torch.device("cuda")
    heads, topk, krows = 16, 512, 8192
    g = torch.Generator(device="cpu").manual_seed(5)
    q_fp8 = (torch.randn((rows, heads, 128), generator=g) / 3).to(torch.float8_e4m3fn).to(device)
    weights = torch.randn((rows, heads), generator=g, dtype=torch.float32).to(device)
    k_fp8 = (torch.randn((krows, 128), generator=g) / 3).to(torch.float8_e4m3fn).to(device)
    k_scales = torch.rand((krows,), generator=g, dtype=torch.float32).to(device) + 0.1
    k_start = torch.zeros((rows,), dtype=torch.int32, device=device)
    k_end = torch.tensor(
        [min(krows, topk + (i + 1) * 512) for i in range(rows)],
        dtype=torch.int32, device=device,
    )
    idx, val = run_fused_indexer_mla(
        q_bytes=q_fp8.view(torch.uint8), weights=weights,
        k_quant_bytes=k_fp8.view(torch.uint8).contiguous(), k_scales=k_scales,
        k_start=k_start, k_end=k_end, num_heads=heads, topk=topk,
    )
    torch.cuda.synchronize(device)
    qf, kf = q_fp8.float(), k_fp8.float()
    for r in range(rows):
        a, b = int(k_start[r]), int(k_end[r])
        logit = (torch.relu(torch.einsum("hd,td->ht", qf[r], kf[a:b])) * weights[r].unsqueeze(1)).sum(0) * k_scales[a:b]
        gset = set((torch.topk(logit, topk).indices + a).tolist())  # absolute index
        assert set(idx[r].tolist()) == gset
