#!/usr/bin/env python3
"""Long-context DSV4 compressed-attention correctness battery.

This is intentionally a standalone opt-in verifier rather than a normal pytest
test.  It targets the long-context surfaces that are easy to miss in short unit
tests: C4 indexer multi-supertile selection, shared-page-table prefill scoring,
and compressed MLA over far C4/C128 page positions under CUDA graph replay.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch

from b12x.attention.mla.compressed_reference import (
    COMPRESSED_MLA_C128_PAGE_SIZE,
    COMPRESSED_MLA_C4_PAGE_SIZE,
    COMPRESSED_MLA_DSV4_PAGE_SIZE,
    COMPRESSED_MLA_HEAD_DIM,
    COMPRESSED_MLA_INDEX_TOPK,
    COMPRESSED_MLA_NOPE_DIM,
    COMPRESSED_MLA_ROPE_DIM,
    COMPRESSED_MLA_SWA_TOKENS,
    compressed_mla_page_nbytes,
    compressed_sparse_mla_reference,
    pack_compressed_mla_kv_cache_reference,
)
from b12x.integration import (
    B12XAttentionWorkspace,
    clear_indexer_caches,
    clear_mla_caches,
    compressed_index_logits_reference,
    compressed_index_decode_supertile_topk_fp8,
    compressed_mla_decode_forward,
    compressed_mla_split_chunks_for_contract,
    pack_compressed_index_k_cache_reference,
    prepare_compressed_indexer_metadata,
    unpack_compressed_index_k_cache_reference,
)


INDEX_HEAD_DIM = 128
INDEX_HEADS = 64
LOCAL_MLA_HEADS_TP2 = 32
INDEX_PAGE_SIZE = 64
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
SM_SCALE = 1.0 / math.sqrt(COMPRESSED_MLA_HEAD_DIM)


@dataclass(frozen=True)
class Metrics:
    max_abs: float
    rmse: float
    cos: float


def _deepgemm_calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """DeepGEMM's cosine-like numeric diff from deep_gemm/testing/numeric.py."""

    x64 = x.double()
    y64 = y.double()
    denominator = (x64 * x64 + y64 * y64).sum()
    if float(denominator.item()) == 0.0:
        return 0.0
    sim = 2 * (x64 * y64).sum() / denominator
    return float((1 - sim).item())


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    return torch.device("cuda")


def _device_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _make_permuted_page_table(
    *,
    rows: int,
    page_table_width: int,
    seqlens: torch.Tensor,
    device: torch.device,
    shared: bool,
    multiplier: int = 17,
    offset: int = 3,
) -> torch.Tensor:
    cols = torch.arange(page_table_width, dtype=torch.int32, device=device)
    page_ids = ((cols.to(torch.int64) * multiplier + offset) % page_table_width).to(torch.int32)
    table = page_ids.unsqueeze(0).expand(rows, -1).clone()
    required_pages = torch.div(
        seqlens.to(torch.int64) + INDEX_PAGE_SIZE - 1,
        INDEX_PAGE_SIZE,
        rounding_mode="floor",
    )
    mask = cols.to(torch.int64).unsqueeze(0) >= required_pages.unsqueeze(1)
    table[mask] = -1
    if not shared:
        row_offsets = torch.arange(rows, dtype=torch.int32, device=device).unsqueeze(1)
        table = torch.where(table >= 0, (table + row_offsets * page_table_width), table)
    return table.contiguous()


def _pack_index_k_cache_vectorized(k: torch.Tensor) -> torch.Tensor:
    if k.ndim != 2 or k.shape[1] != INDEX_HEAD_DIM:
        raise ValueError(f"k must have shape [tokens, {INDEX_HEAD_DIM}], got {tuple(k.shape)}")
    k = k.contiguous().float()
    num_tokens = int(k.shape[0])
    num_pages = max(1, (num_tokens + INDEX_PAGE_SIZE - 1) // INDEX_PAGE_SIZE)
    padded_tokens = num_pages * INDEX_PAGE_SIZE
    if padded_tokens != num_tokens:
        pad = torch.zeros(
            (padded_tokens - num_tokens, INDEX_HEAD_DIM),
            dtype=k.dtype,
            device=k.device,
        )
        k = torch.cat((k, pad), dim=0)
    rows = k.view(num_pages, INDEX_PAGE_SIZE, INDEX_HEAD_DIM)
    max_abs = rows.abs().amax(dim=-1)
    scale = torch.where(max_abs > 0, max_abs / FP8_MAX, torch.ones_like(max_abs))
    quant = (rows / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    cache = torch.zeros(
        (num_pages, INDEX_PAGE_SIZE * (INDEX_HEAD_DIM + 4)),
        dtype=torch.uint8,
        device=k.device,
    )
    data_bytes = INDEX_PAGE_SIZE * INDEX_HEAD_DIM
    cache[:, :data_bytes] = quant.view(torch.uint8).view(num_pages, data_bytes)
    cache[:, data_bytes:] = scale.contiguous().view(torch.uint8).view(num_pages, INDEX_PAGE_SIZE * 4)
    return cache.contiguous()


def _deepgemm_style_paged_index_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
    page_size: int = INDEX_PAGE_SIZE,
) -> torch.Tensor:
    """Independent paged-MQA-logit reference adapted from DeepGEMM.

    This mirrors /home/luke/projects/DeepGEMM/tests/test_attention.py:
    ref_paged_mqa_logits.  The only adaptation is b12x's packed FP8+scale
    K-cache layout and DSV4's single-token-per-row indexer shape.
    """

    rows, heads, head_dim = q_fp8.shape
    if head_dim != INDEX_HEAD_DIM:
        raise ValueError(f"q head_dim must be {INDEX_HEAD_DIM}, got {head_dim}")
    weights_f = weights.squeeze(-1).float() if weights.ndim == 3 else weights.float()
    if weights_f.shape != (rows, heads):
        raise ValueError(f"weights must have shape {(rows, heads)}, got {tuple(weights_f.shape)}")
    if real_page_table.shape[0] != rows:
        raise ValueError("real_page_table row count must match q rows")

    width_tokens = int(real_page_table.shape[1]) * int(page_size)
    logits = torch.full((rows, width_tokens), -float("inf"), dtype=torch.float32, device=q_fp8.device)
    if width_tokens == 0:
        return logits

    k_dequant = unpack_compressed_index_k_cache_reference(
        index_k_cache,
        num_tokens=int(index_k_cache.shape[0]) * int(page_size),
        page_size=page_size,
    )
    max_pages = int(index_k_cache.shape[0])
    q_f32 = q_fp8.float()
    positions = torch.arange(width_tokens, dtype=torch.int64, device=q_fp8.device)
    page_cols = positions // int(page_size)
    page_offsets = positions % int(page_size)

    for row in range(rows):
        page_ids = real_page_table[row, page_cols].to(torch.int64)
        valid = (positions < seqlens[row].to(torch.int64)) & (page_ids >= 0) & (page_ids < max_pages)
        physical = page_ids * int(page_size) + page_offsets
        physical = torch.where(valid, physical, torch.zeros_like(physical))
        k = k_dequant[physical]
        score = torch.matmul(q_f32[row], k.t())
        row_logits = (torch.relu(score) * weights_f[row].unsqueeze(1)).sum(dim=0)
        logits[row] = torch.where(valid, row_logits, torch.full_like(row_logits, -float("inf")))
    return logits


def _assert_logits_match(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    max_abs_tol: float = 2.5e-4,
    diff_tol: float = 1e-8,
) -> None:
    actual_inf = actual == float("-inf")
    expected_inf = expected == float("-inf")
    if not torch.equal(actual_inf, expected_inf):
        mismatch = torch.nonzero(actual_inf != expected_inf, as_tuple=False)
        row = int(mismatch[0, 0].item())
        col = int(mismatch[0, 1].item())
        raise AssertionError(f"{label}: -inf mask mismatch at row={row} col={col}")
    actual_masked = actual.masked_fill(actual_inf, 0)
    expected_masked = expected.masked_fill(expected_inf, 0)
    max_abs = float((actual_masked - expected_masked).abs().max().item())
    diff = _deepgemm_calc_diff(actual_masked, expected_masked)
    print(f"{label}: max_abs={max_abs:.6g} deepgemm_diff={diff:.6g}")
    if max_abs > max_abs_tol or diff > diff_tol:
        raise AssertionError(
            f"{label}: mismatch max_abs={max_abs:.6g} diff={diff:.6g} "
            f"tol=({max_abs_tol}, {diff_tol})"
        )


def _topk_from_logits(logits: torch.Tensor, topk: int) -> torch.Tensor:
    return torch.topk(logits, k=int(topk), dim=1, largest=True, sorted=False).indices.to(torch.int32)


def _topk_from_logits_with_invalid_fill(logits: torch.Tensor, topk: int) -> torch.Tensor:
    rows = int(logits.shape[0])
    out = torch.full((rows, int(topk)), -1, dtype=torch.int32, device=logits.device)
    for row in range(rows):
        valid = logits[row] != float("-inf")
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            continue
        k = min(int(topk), valid_count)
        out[row, :k] = torch.topk(logits[row], k=k, largest=True, sorted=False).indices.to(torch.int32)
    return out


def _make_index_q_weights(
    *,
    rows: int,
    seed: int,
    device: torch.device,
    needle_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if needle_mode:
        q = torch.zeros((rows, INDEX_HEADS, INDEX_HEAD_DIM), dtype=torch.float32, device=device)
        weights = torch.zeros((rows, INDEX_HEADS), dtype=torch.float32, device=device)
        q[:, 0, 0] = 1.0
        weights[:, 0] = 1.0
        return q.to(torch.float8_e4m3fn).contiguous(), weights.contiguous()

    gen = _device_generator(seed, device)
    q = torch.randn((rows, INDEX_HEADS, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32, device=device) / 2
    weights = torch.randn((rows, INDEX_HEADS), generator=gen, dtype=torch.float32, device=device)
    return q.to(torch.float8_e4m3fn).contiguous(), weights.contiguous()


def _make_index_workspace(
    *,
    device: torch.device,
    rows: int,
    page_table_width: int,
    supertile_k: int,
    topk: int,
    use_cuda_graph: bool,
) -> B12XAttentionWorkspace:
    return B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=INDEX_HEADS,
        indexer_num_q_heads=INDEX_HEADS,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=page_table_width,
        max_total_q=max(rows, 1),
        max_batch=max(rows, 1),
        max_paged_q_rows=max(rows, 1),
        max_kv_rows=0,
        indexer_max_k_rows=supertile_k,
        page_size=INDEX_PAGE_SIZE,
        use_cuda_graph=use_cuda_graph,
        reserve_paged_indexer_logits=False,
        paged_indexer_tile_logits_k_rows=supertile_k,
    )


def _make_mla_workspace(
    *,
    device: torch.device,
    rows: int,
    width: int,
    use_cuda_graph: bool,
) -> B12XAttentionWorkspace:
    max_chunks = compressed_mla_split_chunks_for_contract(rows=rows, width=width)
    return B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=LOCAL_MLA_HEADS_TP2,
        head_dim=COMPRESSED_MLA_HEAD_DIM,
        v_head_dim=COMPRESSED_MLA_HEAD_DIM,
        topk=width,
        max_total_q=rows,
        max_batch=rows,
        max_kv_rows=0,
        use_cuda_graph=use_cuda_graph,
        max_chunks_per_row=max_chunks,
    )


def _capture_and_replay(fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    out = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return out


def _compare_tensors(actual: torch.Tensor, expected: torch.Tensor, *, label: str, cos_min: float) -> Metrics:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    max_abs = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    cos = float(
        torch.nn.functional.cosine_similarity(
            actual_f.reshape(-1),
            expected_f.reshape(-1),
            dim=0,
        ).item()
    )
    print(f"{label}: max_abs={max_abs:.6g} rmse={rmse:.6g} cos={cos:.8f}")
    if cos < cos_min:
        raise AssertionError(f"{label} cosine {cos:.8f} below threshold {cos_min}")
    return Metrics(max_abs=max_abs, rmse=rmse, cos=cos)


def _streaming_index_topk_reference(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
    chunk_tokens: int,
) -> torch.Tensor:
    rows = int(q_fp8.shape[0])
    page_table_width = int(real_page_table.shape[1])
    max_tokens = page_table_width * INDEX_PAGE_SIZE
    k_dequant = unpack_compressed_index_k_cache_reference(
        index_k_cache,
        num_tokens=int(index_k_cache.shape[0]) * INDEX_PAGE_SIZE,
    )
    q_f32 = q_fp8.float()
    weights_f32 = weights.float()
    best_values = torch.full((rows, 0), -float("inf"), dtype=torch.float32, device=q_fp8.device)
    best_indices = torch.empty((rows, 0), dtype=torch.int64, device=q_fp8.device)
    max_len = min(int(seqlens.max().item()), max_tokens)
    for start in range(0, max_len, chunk_tokens):
        end = min(start + int(chunk_tokens), max_len)
        pos = torch.arange(start, end, dtype=torch.int64, device=q_fp8.device)
        page_cols = pos // INDEX_PAGE_SIZE
        page_offsets = pos % INDEX_PAGE_SIZE
        row_logits = []
        for row in range(rows):
            page_ids = real_page_table[row, page_cols].to(torch.int64)
            valid = (pos < seqlens[row].to(torch.int64)) & (page_ids >= 0)
            physical = page_ids * INDEX_PAGE_SIZE + page_offsets
            physical = torch.where(valid, physical, torch.zeros_like(physical))
            k = k_dequant[physical]
            scores = torch.matmul(q_f32[row], k.t())
            logits = (torch.relu(scores) * weights_f32[row].unsqueeze(1)).sum(dim=0)
            logits = torch.where(valid, logits, torch.full_like(logits, -float("inf")))
            row_logits.append(logits)
        logits = torch.stack(row_logits, dim=0)
        candidate_values = torch.cat((best_values, logits), dim=1)
        candidate_indices = torch.cat((best_indices, pos.unsqueeze(0).expand(rows, -1)), dim=1)
        next_k = min(int(topk), int(candidate_values.shape[1]))
        values, local = torch.topk(candidate_values, k=next_k, dim=1, largest=True, sorted=False)
        best_values = values
        best_indices = torch.gather(candidate_indices, 1, local)
    return best_indices.to(torch.int32)


def run_deepgemm_reference_alignment(args: argparse.Namespace, device: torch.device) -> None:
    rows = 8
    heads = 16
    page_table_width = 257
    width_tokens = page_table_width * INDEX_PAGE_SIZE
    topk = 64
    gen = _device_generator(args.seed + 101, device)

    physical_pages = rows * page_table_width
    k = torch.randn((physical_pages * INDEX_PAGE_SIZE, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32, device=device) / 3
    packed_loop = pack_compressed_index_k_cache_reference(k, page_size=INDEX_PAGE_SIZE)
    packed_vectorized = _pack_index_k_cache_vectorized(k)
    if not torch.equal(packed_loop, packed_vectorized):
        raise AssertionError("compressed index K-cache packers disagree")

    q_fp8, weights = _make_index_q_weights(rows=rows, seed=args.seed + 102, device=device, needle_mode=False)
    q_fp8 = q_fp8[:, :heads].contiguous()
    weights = weights[:, :heads].contiguous()
    seqlens = torch.tensor(
        [0, 1, 63, 64, 65, 511, 8192, width_tokens - 17],
        dtype=torch.int32,
        device=device,
    )
    page_table = _make_permuted_page_table(
        rows=rows,
        page_table_width=page_table_width,
        seqlens=torch.full((rows,), width_tokens, dtype=torch.int32, device=device),
        device=device,
        shared=False,
        multiplier=29,
        offset=11,
    )
    page_table[0].fill_(-1)
    page_table[2, 3:] = -1
    page_table[5, 11] = -1
    page_table[7, -3:] = -1
    page_table = page_table.contiguous()

    b12x_ref = compressed_index_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=packed_loop,
        real_page_table=page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )
    deepgemm_ref = _deepgemm_style_paged_index_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=packed_loop,
        real_page_table=page_table,
        seqlens=seqlens,
    )
    _assert_logits_match(
        b12x_ref,
        deepgemm_ref,
        label="reference-audit-b12x-vs-deepgemm-paged-mqa",
    )
    _assert_topk_sets_equal(
        _topk_from_logits_with_invalid_fill(b12x_ref, topk),
        _topk_from_logits_with_invalid_fill(deepgemm_ref, topk),
        label="reference-audit-topk",
    )


def run_c4_indexer_dense_equivalence(args: argparse.Namespace, device: torch.device) -> None:
    rows = 32 if args.tier == "smoke" else 96
    topk = 512
    page_table_width = 257 if args.tier == "smoke" else 521
    seq_len = page_table_width * INDEX_PAGE_SIZE - 19
    supertile_k = 8192 if args.tier == "smoke" else 16384
    gen = _device_generator(args.seed + 111, device)
    seqlens = torch.full((rows,), seq_len, dtype=torch.int32, device=device)
    if rows >= 8:
        seqlens[0] = 1
        seqlens[1] = 63
        seqlens[2] = 64
        seqlens[3] = 65
        seqlens[4] = supertile_k - 1
        seqlens[5] = supertile_k
        seqlens[6] = min(seq_len, supertile_k + 1)
        seqlens[7] = min(seq_len, 2 * supertile_k + 1)
    page_table = _make_permuted_page_table(
        rows=rows,
        page_table_width=page_table_width,
        seqlens=torch.full((rows,), seq_len, dtype=torch.int32, device=device),
        device=device,
        shared=True,
        multiplier=31,
        offset=7,
    )
    q_fp8, weights = _make_index_q_weights(rows=rows, seed=args.seed + 112, device=device, needle_mode=False)
    k = torch.randn(
        (page_table_width * INDEX_PAGE_SIZE, INDEX_HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    ) / 4
    index_k_cache = _pack_index_k_cache_vectorized(k)
    workspace = _make_index_workspace(
        device=device,
        rows=rows,
        page_table_width=page_table_width,
        supertile_k=supertile_k,
        topk=topk,
        use_cuda_graph=True,
    )
    clear_indexer_caches()
    workspace.prewarm_paged_indexer_tiled_topk()
    workspace.prewarm_paged_indexer_tiled_scorer(
        index_k_cache=index_k_cache,
        width_tokens=supertile_k,
    )
    metadata = prepare_compressed_indexer_metadata(
        real_page_table=page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=INDEX_HEADS,
        build_schedule=False,
        shared_page_table=True,
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def run() -> torch.Tensor:
        return compressed_index_decode_supertile_topk_fp8(
            q_fp8=q_fp8,
            weights=weights.unsqueeze(-1),
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=INDEX_HEADS,
            workspace=workspace,
            out_indices=actual,
            supertile_k=supertile_k,
        )

    out = _capture_and_replay(run)
    deepgemm_ref = _deepgemm_style_paged_index_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        seqlens=seqlens,
    )
    expected = _topk_from_logits_with_invalid_fill(deepgemm_ref, topk)
    _assert_topk_sets_equal(
        out,
        expected,
        label=f"c4-indexer-shared-prefill-dense-deepgemm-graph rows={rows} width={page_table_width * INDEX_PAGE_SIZE}",
    )


def _assert_topk_sets_equal(actual: torch.Tensor, expected: torch.Tensor, *, label: str) -> None:
    actual_sorted = torch.sort(actual, dim=1).values
    expected_sorted = torch.sort(expected, dim=1).values
    if not torch.equal(actual_sorted, expected_sorted):
        mismatch = torch.nonzero(actual_sorted != expected_sorted, as_tuple=False)
        row = int(mismatch[0, 0].item())
        missing = sorted(set(expected[row].tolist()) - set(actual[row].tolist()))[:16]
        extra = sorted(set(actual[row].tolist()) - set(expected[row].tolist()))[:16]
        raise AssertionError(f"{label}: top-k mismatch at row {row}; missing={missing} extra={extra}")
    print(f"{label}: top-k exact set match rows={actual.shape[0]} width={actual.shape[1]}")


def _assert_needles_present(
    actual: torch.Tensor,
    *,
    needle_positions: list[int],
    seqlens: torch.Tensor,
    label: str,
    rows_to_check: list[int],
) -> None:
    actual_cpu = actual.detach().cpu()
    seqlens_cpu = seqlens.detach().cpu()
    for row in rows_to_check:
        expected = {pos for pos in needle_positions if pos < int(seqlens_cpu[row].item())}
        present = set(int(v) for v in actual_cpu[row].tolist())
        missing = sorted(expected - present)
        if missing:
            raise AssertionError(f"{label}: row {row} missing planted C4 needles {missing[:32]}")
    print(f"{label}: all planted needles present in rows {rows_to_check}")


def run_c4_indexer_random_decode(args: argparse.Namespace, device: torch.device) -> None:
    rows = 1
    topk = int(args.topk)
    page_table_width = int(args.page_table_width)
    seq_len = min(int(args.c4_seq_len), page_table_width * INDEX_PAGE_SIZE)
    gen = _device_generator(args.seed + 1, device)
    seqlens = torch.full((rows,), seq_len, dtype=torch.int32, device=device)
    page_table = _make_permuted_page_table(
        rows=rows,
        page_table_width=page_table_width,
        seqlens=seqlens,
        device=device,
        shared=True,
    )
    q_fp8, weights = _make_index_q_weights(rows=rows, seed=args.seed + 2, device=device, needle_mode=False)
    k = torch.randn(
        (page_table_width * INDEX_PAGE_SIZE, INDEX_HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    ) / 3
    index_k_cache = _pack_index_k_cache_vectorized(k)
    workspace = _make_index_workspace(
        device=device,
        rows=rows,
        page_table_width=page_table_width,
        supertile_k=int(args.supertile_k),
        topk=topk,
        use_cuda_graph=True,
    )
    clear_indexer_caches()
    workspace.prewarm_paged_indexer_tiled_topk()
    workspace.prewarm_paged_indexer_tiled_scorer(
        index_k_cache=index_k_cache,
        width_tokens=int(args.supertile_k),
    )
    metadata = prepare_compressed_indexer_metadata(
        real_page_table=page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=INDEX_HEADS,
        build_schedule=False,
        shared_page_table=False,
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def run() -> torch.Tensor:
        return compressed_index_decode_supertile_topk_fp8(
            q_fp8=q_fp8,
            weights=weights.unsqueeze(-1),
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=INDEX_HEADS,
            workspace=workspace,
            out_indices=actual,
            supertile_k=int(args.supertile_k),
        )

    start = time.perf_counter()
    out = _capture_and_replay(run)
    expected = _streaming_index_topk_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        seqlens=seqlens,
        topk=topk,
        chunk_tokens=int(args.reference_chunk_tokens),
    )
    _assert_topk_sets_equal(out, expected, label="c4-indexer-random-decode-graph")
    print(f"c4-indexer-random-decode-graph: elapsed={time.perf_counter() - start:.2f}s")


def run_c4_indexer_shared_prefill_needles(args: argparse.Namespace, device: torch.device) -> None:
    rows = int(args.prefill_rows)
    topk = int(args.topk)
    page_table_width = int(args.page_table_width)
    seq_len = min(int(args.c4_seq_len), page_table_width * INDEX_PAGE_SIZE)
    supertile_k = int(args.supertile_k)
    seqlens = torch.full((rows,), seq_len, dtype=torch.int32, device=device)
    if rows >= 4:
        seqlens[1::4] = min(seq_len, supertile_k)
        seqlens[2::4] = min(seq_len, supertile_k + 1)
        seqlens[3::4] = min(seq_len, page_table_width * INDEX_PAGE_SIZE - INDEX_PAGE_SIZE)
    page_table = _make_permuted_page_table(
        rows=rows,
        page_table_width=page_table_width,
        seqlens=torch.full((rows,), seq_len, dtype=torch.int32, device=device),
        device=device,
        shared=True,
    )
    q_fp8, weights = _make_index_q_weights(rows=rows, seed=args.seed + 3, device=device, needle_mode=True)
    k = torch.zeros((page_table_width * INDEX_PAGE_SIZE, INDEX_HEAD_DIM), dtype=torch.float32, device=device)
    raw_needles = {
        0,
        1,
        63,
        64,
        511,
        512,
        supertile_k - 1,
        supertile_k,
        supertile_k + 1,
        2 * supertile_k - 1,
        2 * supertile_k,
        max(0, seq_len - 2),
        max(0, seq_len - 1),
    }
    needle_positions = sorted(pos for pos in raw_needles if 0 <= pos < seq_len)
    for rank, logical_pos in enumerate(needle_positions):
        page_col = logical_pos // INDEX_PAGE_SIZE
        page_off = logical_pos % INDEX_PAGE_SIZE
        physical_page = int(page_table[0, page_col].item())
        k[physical_page * INDEX_PAGE_SIZE + page_off, 0] = 64.0 + rank
    index_k_cache = _pack_index_k_cache_vectorized(k)
    workspace = _make_index_workspace(
        device=device,
        rows=rows,
        page_table_width=page_table_width,
        supertile_k=supertile_k,
        topk=topk,
        use_cuda_graph=True,
    )
    clear_indexer_caches()
    workspace.prewarm_paged_indexer_tiled_topk()
    workspace.prewarm_paged_indexer_tiled_scorer(
        index_k_cache=index_k_cache,
        width_tokens=supertile_k,
    )
    metadata = prepare_compressed_indexer_metadata(
        real_page_table=page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=INDEX_HEADS,
        build_schedule=False,
        shared_page_table=True,
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def run() -> torch.Tensor:
        return compressed_index_decode_supertile_topk_fp8(
            q_fp8=q_fp8,
            weights=weights.unsqueeze(-1),
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=INDEX_HEADS,
            workspace=workspace,
            out_indices=actual,
            supertile_k=supertile_k,
        )

    start = time.perf_counter()
    out = _capture_and_replay(run)
    rows_to_check = sorted({0, min(1, rows - 1), min(2, rows - 1), min(3, rows - 1), rows - 1})
    _assert_needles_present(
        out,
        needle_positions=needle_positions,
        seqlens=seqlens,
        label="c4-indexer-shared-prefill-needles-graph",
        rows_to_check=rows_to_check,
    )
    print(f"c4-indexer-shared-prefill-needles-graph: elapsed={time.perf_counter() - start:.2f}s")


def _make_compressed_mla_q(rows: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = _device_generator(seed, device)
    q = torch.randn(
        (rows, LOCAL_MLA_HEADS_TP2, COMPRESSED_MLA_HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    ) * 0.04
    return q.to(torch.bfloat16).contiguous()


def _make_compressed_mla_cache(tokens: int, page_size: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = _device_generator(seed, device)
    k_nope = torch.randn(
        (tokens, COMPRESSED_MLA_NOPE_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    ) * 0.05
    k_rope = torch.randn(
        (tokens, COMPRESSED_MLA_ROPE_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    ) * 0.05
    return pack_compressed_mla_kv_cache_reference(
        k_nope,
        k_rope.to(torch.bfloat16),
        page_size=page_size,
    )


def _map_indexed_indices(indices: torch.Tensor, page_table: torch.Tensor, page_size: int) -> torch.Tensor:
    mapped = torch.full_like(indices, -1)
    rows, width = indices.shape
    for row in range(rows):
        raw = indices[row].to(torch.int64)
        valid = raw >= 0
        page_col = raw // page_size
        page_off = raw % page_size
        valid &= page_col < page_table.shape[1]
        safe_col = torch.where(valid, page_col, torch.zeros_like(page_col))
        page_id = page_table[row, safe_col].to(torch.int64)
        valid &= page_id >= 0
        mapped[row] = torch.where(valid, page_id * page_size + page_off, torch.full_like(raw, -1)).to(torch.int32)
    return mapped.contiguous()


def run_compressed_mla_long_cases(args: argparse.Namespace, device: torch.device) -> None:
    rows = int(args.mla_rows)
    q = _make_compressed_mla_q(rows, args.seed + 11, device)
    swa_width = COMPRESSED_MLA_SWA_TOKENS
    swa_cache = _make_compressed_mla_cache(
        tokens=swa_width + rows + 8,
        page_size=COMPRESSED_MLA_DSV4_PAGE_SIZE,
        seed=args.seed + 12,
        device=device,
    )
    swa_indices = torch.empty((rows, swa_width), dtype=torch.int32, device=device)
    for row in range(rows):
        swa_indices[row] = torch.arange(row, row + swa_width, dtype=torch.int32, device=device)
    swa_lengths = torch.full((rows,), swa_width, dtype=torch.int32, device=device)
    empty_swa_cache = _make_compressed_mla_cache(
        tokens=1,
        page_size=COMPRESSED_MLA_DSV4_PAGE_SIZE,
        seed=args.seed + 113,
        device=device,
    )
    empty_swa_indices = torch.empty((rows, 0), dtype=torch.int32, device=device)
    empty_swa_lengths = torch.zeros((rows,), dtype=torch.int32, device=device)
    attn_sink = torch.linspace(-0.1, 0.15, LOCAL_MLA_HEADS_TP2, dtype=torch.float32, device=device)

    c128_width = int(args.c128_width)
    c128_valid = min(int(args.c128_valid), c128_width)
    c128_cache = _make_compressed_mla_cache(
        tokens=max(c128_valid, 1),
        page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
        seed=args.seed + 13,
        device=device,
    )
    c128_indices = torch.full((rows, c128_width), -1, dtype=torch.int32, device=device)
    c128_base = torch.linspace(0, max(c128_valid - 1, 0), steps=c128_valid, device=device).round().to(torch.int32)
    if c128_valid:
        c128_base[0] = 0
        c128_base[-1] = c128_valid - 1
    c128_indices[:, :c128_valid] = c128_base.unsqueeze(0).expand(rows, -1)
    c128_lengths = torch.full((rows,), c128_valid, dtype=torch.int32, device=device)

    c4_width = COMPRESSED_MLA_INDEX_TOPK
    c4_page_table_width = int(args.page_table_width)
    c4_physical_pages = 256
    c4_cache = _make_compressed_mla_cache(
        tokens=c4_physical_pages * COMPRESSED_MLA_C4_PAGE_SIZE,
        page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        seed=args.seed + 14,
        device=device,
    )
    c4_page_table = torch.arange(c4_page_table_width, dtype=torch.int32, device=device)
    c4_page_table = ((c4_page_table.to(torch.int64) * 13 + 5) % c4_physical_pages).to(torch.int32)
    c4_page_table = c4_page_table.unsqueeze(0).expand(rows, -1).contiguous()
    far = c4_page_table_width * COMPRESSED_MLA_C4_PAGE_SIZE - 1
    c4_seed_indices = torch.tensor(
        [0, 1, 63, 64, 127, 128, 511, 512, far - 2, far - 1, far],
        dtype=torch.int32,
        device=device,
    )
    fill = torch.linspace(0, far, steps=c4_width, device=device).round().to(torch.int32)
    fill[: c4_seed_indices.numel()] = c4_seed_indices
    fill = torch.clamp(fill, 0, far)
    c4_indices = fill.unsqueeze(0).expand(rows, -1).contiguous()
    c4_lengths = torch.full((rows,), c4_width, dtype=torch.int32, device=device)
    mapped_c4_indices = _map_indexed_indices(
        c4_indices,
        c4_page_table,
        COMPRESSED_MLA_C4_PAGE_SIZE,
    )

    def run_case(
        *,
        label: str,
        case_swa_cache: torch.Tensor,
        case_swa_indices: torch.Tensor,
        case_swa_lengths: torch.Tensor,
        indexed_cache: torch.Tensor | None = None,
        indexed_indices: torch.Tensor | None = None,
        indexed_lengths: torch.Tensor | None = None,
        indexed_page_size: int | None = None,
        indexed_page_table: torch.Tensor | None = None,
        expected_indexed_indices: torch.Tensor | None = None,
    ) -> None:
        has_indexed = indexed_cache is not None
        if has_indexed:
            assert indexed_indices is not None
            assert indexed_lengths is not None
            assert indexed_page_size is not None
        total_width = int(case_swa_indices.shape[1]) + (int(indexed_indices.shape[1]) if has_indexed else 0)
        workspace = _make_mla_workspace(
            device=device,
            rows=rows,
            width=total_width,
            use_cuda_graph=True,
        )
        clear_mla_caches()

        def run_kernel() -> torch.Tensor:
            return compressed_mla_decode_forward(
                q_all=q,
                swa_k_cache=case_swa_cache,
                swa_indices=case_swa_indices,
                swa_topk_lengths=case_swa_lengths,
                indexed_k_cache=indexed_cache,
                indexed_indices=indexed_indices,
                indexed_topk_lengths=indexed_lengths,
                indexed_page_size=indexed_page_size,
                indexed_page_table=indexed_page_table,
                attn_sink=attn_sink,
                workspace=workspace,
                sm_scale=SM_SCALE,
            )

        actual = _capture_and_replay(run_kernel)
        expected = compressed_sparse_mla_reference(
            q,
            case_swa_cache,
            case_swa_indices,
            case_swa_lengths,
            extra_k_cache=indexed_cache,
            extra_indices=expected_indexed_indices if expected_indexed_indices is not None else indexed_indices,
            extra_topk_lengths=indexed_lengths,
            extra_page_size=indexed_page_size,
            attn_sink=attn_sink,
            sm_scale=SM_SCALE,
        )
        _compare_tensors(
            actual,
            expected,
            label=f"{label} rows={rows} width={total_width}",
            cos_min=float(args.cos_min),
        )

    run_case(
        label="compressed-mla-swa-only-graph",
        case_swa_cache=swa_cache,
        case_swa_indices=swa_indices,
        case_swa_lengths=swa_lengths,
    )
    run_case(
        label="compressed-mla-c128-only-graph",
        case_swa_cache=empty_swa_cache,
        case_swa_indices=empty_swa_indices,
        case_swa_lengths=empty_swa_lengths,
        indexed_cache=c128_cache,
        indexed_indices=c128_indices,
        indexed_lengths=c128_lengths,
        indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
    )
    run_case(
        label="compressed-mla-c4-only-pagetable-graph",
        case_swa_cache=empty_swa_cache,
        case_swa_indices=empty_swa_indices,
        case_swa_lengths=empty_swa_lengths,
        indexed_cache=c4_cache,
        indexed_indices=c4_indices,
        indexed_lengths=c4_lengths,
        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        indexed_page_table=c4_page_table,
        expected_indexed_indices=mapped_c4_indices,
    )
    run_case(
        label="compressed-mla-c128-plus-swa-graph",
        case_swa_cache=swa_cache,
        case_swa_indices=swa_indices,
        case_swa_lengths=swa_lengths,
        indexed_cache=c128_cache,
        indexed_indices=c128_indices,
        indexed_lengths=c128_lengths,
        indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
    )
    run_case(
        label="compressed-mla-c4-plus-swa-pagetable-graph",
        case_swa_cache=swa_cache,
        case_swa_indices=swa_indices,
        case_swa_lengths=swa_lengths,
        indexed_cache=c4_cache,
        indexed_indices=c4_indices,
        indexed_lengths=c4_lengths,
        indexed_page_size=COMPRESSED_MLA_C4_PAGE_SIZE,
        indexed_page_table=c4_page_table,
        expected_indexed_indices=mapped_c4_indices,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "reference", "indexer", "mla"), default="all")
    parser.add_argument("--tier", choices=("smoke", "full", "stress"), default="smoke")
    parser.add_argument("--seed", type=int, default=924_001)
    parser.add_argument("--page-table-width", type=int, default=4160)
    parser.add_argument("--c4-seq-len", type=int, default=266_240)
    parser.add_argument("--supertile-k", type=int, default=65_024)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--prefill-rows", type=int, default=1024)
    parser.add_argument("--mla-rows", type=int, default=1)
    parser.add_argument("--c128-width", type=int, default=2688)
    parser.add_argument("--c128-valid", type=int, default=2660)
    parser.add_argument("--reference-chunk-tokens", type=int, default=8192)
    parser.add_argument("--cos-min", type=float, default=0.999)
    args = parser.parse_args()
    if args.tier == "full":
        args.prefill_rows = max(int(args.prefill_rows), 4096)
        args.mla_rows = max(int(args.mla_rows), 4)
    elif args.tier == "stress":
        args.prefill_rows = max(int(args.prefill_rows), 4096)
        args.mla_rows = max(int(args.mla_rows), 16)
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = _require_cuda()
    print(
        "dsv4-long-context-battery "
        f"mode={args.mode} tier={args.tier} device={device} "
        f"page_table_width={args.page_table_width} c4_seq_len={args.c4_seq_len} "
        f"supertile_k={args.supertile_k} prefill_rows={args.prefill_rows}"
    )
    start = time.perf_counter()
    if args.mode in ("all", "indexer"):
        run_c4_indexer_dense_equivalence(args, device)
        run_c4_indexer_random_decode(args, device)
        run_c4_indexer_shared_prefill_needles(args, device)
    if args.mode in ("all", "reference"):
        run_deepgemm_reference_alignment(args, device)
    if args.mode in ("all", "mla"):
        run_compressed_mla_long_cases(args, device)
    torch.cuda.synchronize()
    print(f"dsv4-long-context-battery: PASS elapsed={time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
