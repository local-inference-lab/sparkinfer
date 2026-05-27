#!/usr/bin/env python3
"""Standalone numerical reference battery for NSA Indexer and Sparse MLA.

This file is intentionally not named test_*.py. Run it directly when you want a
large SGLang-shaped correctness pass against the B12X reference implementations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from b12x.attention.workspace import B12XAttentionWorkspace  # noqa: E402
from b12x.attention.mla.reference import (  # noqa: E402
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
)
from b12x.attention.indexer.reference import (  # noqa: E402
    extend_logits_reference,
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
)
from b12x.integration.mla import (  # noqa: E402
    MLASparseDecodeMetadata,
    MLASparseExtendMetadata,
    clear_mla_caches,
    sparse_mla_decode_forward,
    sparse_mla_extend_forward,
)
from b12x.integration.indexer import (  # noqa: E402
    IndexerExtendMetadata,
    IndexerPagedDecodeMetadata,
    clear_indexer_caches,
    build_paged_mqa_schedule_metadata,
    paged_decode_logits,
    extend_logits,
)


PAGE_SIZE = 64
INDEX_HEAD_DIM = 128
MLA_NOPE_DIM = 512
MLA_ROPE_DIM = 64
MLA_Q_DIM = MLA_NOPE_DIM + MLA_ROPE_DIM
MLA_PACKED_DIM = 656
MLA_V_HEAD_DIM = MLA_NOPE_DIM
MLA_SM_SCALE = MLA_Q_DIM**-0.5
DEFAULT_TOPK = 2048
DEFAULT_INDEX_HEADS = 32
DEFAULT_MLA_HEADS = 16
FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
Tier = Literal["smoke", "full", "stress"]
Mode = Literal["all", "indexer", "mla", "e2e", "sglang"]
Execution = Literal["eager", "graph"]


TIER_ORDER: dict[Tier, int] = {"smoke": 0, "full": 1, "stress": 2}


@dataclass(frozen=True)
class CaseResult:
    name: str
    tier: Tier
    mode: Mode
    execution: Execution
    elapsed_ms: float
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class IndexerPagedCase:
    name: str
    tier: Tier
    q_rows: int
    row_seqlens_a: tuple[int, ...]
    row_seqlens_b: tuple[int, ...] | None
    index_heads: int
    graph_width_tokens: int
    topk: int
    seed: int


@dataclass(frozen=True)
class IndexerExtendCase:
    name: str
    tier: Tier
    request_shapes: tuple[tuple[int, int], ...]
    index_heads: int
    topk: int
    seed: int
    preinitialize_invalid_logits: bool


@dataclass(frozen=True)
class MLACase:
    name: str
    tier: Tier
    q_rows: int
    batch: int
    cache_len: int
    topk: int
    valid_pattern: tuple[int, ...]
    mla_heads: int
    seed: int
    mode: Literal["decode", "extend", "verify", "target_verify", "draft_extend"]


@dataclass(frozen=True)
class E2EDecodeCase:
    name: str
    tier: Tier
    row_seqlens_a: tuple[int, ...]
    row_seqlens_b: tuple[int, ...] | None
    index_heads: int
    mla_heads: int
    topk: int
    graph_width_tokens: int
    seed: int


@dataclass(frozen=True)
class E2EExtendCase:
    name: str
    tier: Tier
    request_shapes: tuple[tuple[int, int], ...]
    index_heads: int
    mla_heads: int
    topk: int
    seed: int


@dataclass(frozen=True)
class SGLangPagedCase:
    name: str
    tier: Tier
    row_seqlens_a: tuple[int, ...]
    row_seqlens_b: tuple[int, ...] | None
    extend_lens: tuple[int, ...]
    index_heads: int
    mla_heads: int
    graph_width_tokens: int
    q_padding: int
    seed: int
    mode_name: Literal["decode", "target_verify", "draft_extend"]
    run_mla: bool


@dataclass(frozen=True)
class SGLangRaggedCase:
    name: str
    tier: Tier
    request_shapes: tuple[tuple[int, int], ...]
    index_heads: int
    mla_heads: int
    q_padding: int
    seed: int
    run_mla: bool


@dataclass(frozen=True)
class SGLangImports:
    NSAMetadata: Any
    NSAIndexerMetadata: Any
    TopkTransformMethod: Any
    NSATokenToKVPool: Any
    envs: Any


class BatteryFailure(AssertionError):
    pass


def _cpu_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scale: float = 1.0,
) -> torch.Tensor:
    tensor = torch.randn(shape, generator=_cpu_generator(seed), dtype=torch.float32)
    tensor = tensor.mul(float(scale)).to(device=device)
    return tensor.to(dtype=dtype)


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _page_count(tokens: int) -> int:
    return max(1, _align_up(tokens, PAGE_SIZE) // PAGE_SIZE)


def _device_name(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    return f"{props.name} sm_{major}{minor}"


def _require_sm120() -> torch.device | None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available")
        return None
    device = torch.device("cuda")
    return device


def _ensure_sglang_imports(sglang_root: Path) -> SGLangImports:
    root = sglang_root.expanduser().resolve()
    python_root = root / "python"
    if not python_root.exists():
        raise BatteryFailure(f"SGLang python root does not exist: {python_root}")
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    try:
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.nsa_backend import (
            NSAMetadata,
            NSAIndexerMetadata,
            TopkTransformMethod,
        )
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
    except Exception as exc:  # noqa: BLE001
        raise BatteryFailure(f"failed to import SGLang from {python_root}: {exc}") from exc
    if not envs.SGLANG_NSA_FUSE_TOPK.get():
        raise BatteryFailure(
            "SGLang fused NSA top-k is disabled; these integration-shaped cases "
            "expect SGLANG_NSA_FUSE_TOPK=true"
        )
    return SGLangImports(
        NSAMetadata=NSAMetadata,
        NSAIndexerMetadata=NSAIndexerMetadata,
        TopkTransformMethod=TopkTransformMethod,
        NSATokenToKVPool=NSATokenToKVPool,
        envs=envs,
    )


def _make_index_q_and_weights(
    *,
    q_rows: int,
    index_heads: int,
    seed: int,
    device: torch.device,
    structured: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if structured:
        q = torch.ones(
            (q_rows, index_heads, INDEX_HEAD_DIM),
            dtype=torch.float32,
            device=device,
        ).to(torch.float8_e4m3fn)
        weights = torch.linspace(
            0.75,
            1.25,
            index_heads,
            dtype=torch.float32,
            device=device,
        ).repeat(q_rows, 1)
        return q.contiguous(), weights.contiguous()

    q = _randn(
        (q_rows, index_heads, INDEX_HEAD_DIM),
        seed=seed,
        device=device,
        scale=0.5,
    ).to(torch.float8_e4m3fn)
    weights = _randn(
        (q_rows, index_heads),
        seed=seed + 1,
        device=device,
        scale=1.0 / math.sqrt(max(index_heads, 1)),
    )
    return q.contiguous(), weights.contiguous()


def _quantize_rows_to_index_kv_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = k.abs().amax(dim=1) / FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (k / scale.unsqueeze(1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return quant.to(torch.float8_e4m3fn).contiguous(), scale.to(torch.float32).contiguous()


def _make_disjoint_real_page_table(
    *,
    seqlens: tuple[int, ...],
    graph_width_tokens: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    width_pages = max(_page_count(graph_width_tokens), max(_page_count(s) for s in seqlens))
    row_pages = [_page_count(seq_len) for seq_len in seqlens]
    required_pages = sum(row_pages)
    pool_pages = max(required_pages + 17, width_pages * max(2, len(seqlens)))
    perm = torch.randperm(pool_pages, generator=_cpu_generator(seed), dtype=torch.int64)

    table_cpu = torch.full((len(seqlens), width_pages), -1, dtype=torch.int32)
    cursor = 0
    for row, pages in enumerate(row_pages):
        chosen = perm[cursor : cursor + pages].to(torch.int32)
        table_cpu[row, :pages] = chosen
        cursor += pages

    return table_cpu.to(device=device), table_cpu, pool_pages


def _page_table_1_from_real(
    *,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
) -> torch.Tensor:
    rows, width_pages = real_page_table.shape
    width_tokens = width_pages * PAGE_SIZE
    out = torch.full(
        (rows, width_tokens),
        -1,
        dtype=torch.int32,
        device=real_page_table.device,
    )
    for row in range(rows):
        seq_len = int(seqlens[row].item())
        if seq_len <= 0:
            continue
        pos = torch.arange(seq_len, dtype=torch.int32, device=real_page_table.device)
        page_col = torch.div(pos, PAGE_SIZE, rounding_mode="floor").to(torch.long)
        slot = pos % PAGE_SIZE
        out[row, :seq_len] = real_page_table[row, page_col] * PAGE_SIZE + slot
    return out


def _logical_positions_to_physical(
    *,
    real_page_table: torch.Tensor,
    row: int,
    logical_positions: torch.Tensor,
) -> torch.Tensor:
    page_col = torch.div(logical_positions, PAGE_SIZE, rounding_mode="floor").to(torch.long)
    slot = logical_positions.to(torch.int32) % PAGE_SIZE
    return real_page_table[row, page_col] * PAGE_SIZE + slot


def _make_random_index_cache(
    *,
    pool_pages: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    k_cpu = torch.randn(
        (pool_pages * PAGE_SIZE, INDEX_HEAD_DIM),
        generator=_cpu_generator(seed),
        dtype=torch.float32,
    ) / 3.0
    return pack_index_k_cache_reference(k_cpu).to(device=device)


def _make_structured_index_cache(
    *,
    real_page_table_cpu: torch.Tensor,
    seqlens: tuple[int, ...],
    pool_pages: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    del seed
    k_cpu = torch.zeros((pool_pages * PAGE_SIZE, INDEX_HEAD_DIM), dtype=torch.float32)
    for row, seq_len in enumerate(seqlens):
        if seq_len <= 0:
            continue
        values = torch.linspace(0.01, 1.01, seq_len, dtype=torch.float32)
        for pos in range(seq_len):
            page = int(real_page_table_cpu[row, pos // PAGE_SIZE].item())
            physical = page * PAGE_SIZE + (pos % PAGE_SIZE)
            k_cpu[physical].fill_(float(values[pos].item()))
    return pack_index_k_cache_reference(k_cpu).to(device=device)


def _make_structured_index_rows(
    *,
    real_page_table_cpu: torch.Tensor,
    seqlens: tuple[int, ...],
    pool_pages: int,
    device: torch.device,
) -> torch.Tensor:
    k_cpu = torch.zeros((pool_pages * PAGE_SIZE, INDEX_HEAD_DIM), dtype=torch.float32)
    for row, seq_len in enumerate(seqlens):
        if seq_len <= 0:
            continue
        values = torch.linspace(0.01, 1.01, seq_len, dtype=torch.float32)
        for pos in range(seq_len):
            page = int(real_page_table_cpu[row, pos // PAGE_SIZE].item())
            physical = page * PAGE_SIZE + (pos % PAGE_SIZE)
            k_cpu[physical].fill_(float(values[pos].item()))
    return k_cpu.to(device=device)


def _make_mla_pool(
    *,
    pool_tokens: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_nope = _randn(
        (pool_tokens, 1, MLA_NOPE_DIM),
        seed=seed,
        device=device,
        dtype=torch.bfloat16,
        scale=0.20,
    )
    k_rope = _randn(
        (pool_tokens, 1, MLA_ROPE_DIM),
        seed=seed + 1,
        device=device,
        dtype=torch.bfloat16,
        scale=0.20,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    if packed.dtype != torch.uint8 or packed.shape[-1] != MLA_PACKED_DIM:
        raise BatteryFailure(f"unexpected packed MLA cache shape {tuple(packed.shape)}")
    return k_nope, k_rope, packed


def _make_q_all(
    *,
    q_rows: int,
    mla_heads: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    return _randn(
        (q_rows, mla_heads, MLA_Q_DIM),
        seed=seed,
        device=device,
        dtype=torch.bfloat16,
        scale=0.20,
    ).contiguous()


def _make_workspace(
    *,
    mode: Literal["decode", "extend", "verify", "draft_extend"],
    device: torch.device,
    topk: int,
    max_total_q: int,
    max_batch: int,
    max_kv_rows: int = 0,
    max_paged_q_rows: int | None = None,
    max_page_table_width: int | None = None,
    mla_heads: int = DEFAULT_MLA_HEADS,
    index_heads: int = DEFAULT_INDEX_HEADS,
    use_cuda_graph: bool = False,
) -> B12XAttentionWorkspace:
    return B12XAttentionWorkspace.for_fixed_capacity(
        mode=mode,
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=mla_heads,
        indexer_num_q_heads=index_heads,
        head_dim=MLA_Q_DIM,
        v_head_dim=MLA_V_HEAD_DIM,
        topk=topk,
        max_page_table_width=max_page_table_width,
        max_total_q=max_total_q,
        max_batch=max_batch,
        max_paged_q_rows=max_paged_q_rows,
        max_kv_rows=max_kv_rows,
        page_size=PAGE_SIZE,
        use_cuda_graph=use_cuda_graph,
    )


def _select_paged_topk_from_logits(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    rows = logits.shape[0]
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    gather_k = min(int(topk), int(logits.shape[1]), int(page_table_1.shape[1]))
    if gather_k <= 0:
        return out
    values, positions = torch.topk(logits, k=gather_k, dim=1, largest=True, sorted=True)
    gathered = page_table_1.gather(1, positions.to(torch.long))
    valid = torch.isfinite(values) & (gathered >= 0)
    out[:, :gather_k] = torch.where(valid, gathered, torch.full_like(gathered, -1))
    return out


def _select_ragged_topk_from_logits(
    *,
    logits: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    rows = logits.shape[0]
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    gather_k = min(int(topk), int(logits.shape[1]))
    if gather_k <= 0:
        return out
    masked = torch.full_like(logits, float("-inf"))
    for row in range(rows):
        ks = max(0, int(k_start[row].item()))
        ke = min(int(k_end[row].item()), int(logits.shape[1]))
        if ke > ks:
            masked[row, ks:ke] = logits[row, ks:ke]
    values, positions = torch.topk(masked, k=gather_k, dim=1, largest=True, sorted=True)
    valid = torch.isfinite(values)
    out[:, :gather_k] = torch.where(
        valid,
        positions.to(torch.int32),
        torch.full_like(positions, -1, dtype=torch.int32),
    )
    return out


def _map_compact_topk_to_physical(
    *,
    compact_topk: torch.Tensor,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    mapped = torch.full_like(compact_topk, -1)
    mask = compact_topk >= 0
    if torch.any(mask):
        mapped[mask] = row_ids.index_select(0, compact_topk[mask].to(torch.long))
    return mapped


def _tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.to(torch.float32) - expected.to(torch.float32)).reshape(-1)
    actual_f = actual.to(torch.float32).reshape(-1)
    expected_f = expected.to(torch.float32).reshape(-1)
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    rmse = float(torch.sqrt(diff.square().mean()).item()) if diff.numel() else 0.0
    denom = float(actual_f.norm().item() * expected_f.norm().item())
    if denom == 0.0:
        cos = 1.0 if max_abs == 0.0 else 0.0
    else:
        cos = float(torch.dot(actual_f, expected_f).item() / denom)
    return {"max_abs": max_abs, "rmse": rmse, "cos": cos}


def _assert_indexer_logits_close(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    require_invalid_neginf: bool,
) -> dict[str, float | int]:
    if actual.shape != expected.shape:
        raise BatteryFailure(f"logit shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    expected_finite = torch.isfinite(expected)
    if require_invalid_neginf:
        expected_invalid = torch.isneginf(expected)
        actual_invalid = torch.isneginf(actual)
        if not torch.equal(actual_invalid, expected_invalid):
            mismatch = int(torch.count_nonzero(actual_invalid != expected_invalid).item())
            raise BatteryFailure(f"invalid -inf mask mismatch at {mismatch} positions")
    if torch.any(expected_finite):
        torch.testing.assert_close(
            actual[expected_finite],
            expected[expected_finite],
            atol=1e-4,
            rtol=1e-4,
        )
        metrics = _tensor_metrics(actual[expected_finite], expected[expected_finite])
    else:
        metrics = {"max_abs": 0.0, "rmse": 0.0, "cos": 1.0}
    return {
        **metrics,
        "finite_logits": int(torch.count_nonzero(expected_finite).item()),
        "total_logits": int(expected.numel()),
    }


def _assert_mla_close(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise BatteryFailure(f"MLA output shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if not torch.isfinite(actual).all():
        bad = int(torch.count_nonzero(~torch.isfinite(actual)).item())
        raise BatteryFailure(f"MLA output contains {bad} non-finite values")
    metrics = _tensor_metrics(actual, expected)
    if metrics["max_abs"] > 0.10 or metrics["rmse"] > 0.005 or metrics["cos"] < 0.9995:
        raise BatteryFailure(
            "MLA mismatch: "
            f"max_abs={metrics['max_abs']:.6f} rmse={metrics['rmse']:.6f} "
            f"cos={metrics['cos']:.6f}"
        )
    return metrics


def _assert_topk_equal(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, int]:
    if actual.shape != expected.shape:
        raise BatteryFailure(f"topk shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if not torch.equal(actual, expected):
        mismatch = int(torch.count_nonzero(actual != expected).item())
        first = torch.nonzero(actual != expected, as_tuple=False)[0].tolist()
        raise BatteryFailure(f"topk mismatch at {mismatch} entries, first={first}")
    return {"topk_entries": int(actual.numel())}


def _assert_topk_set_equal(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, int]:
    if actual.shape != expected.shape:
        raise BatteryFailure(f"topk shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    actual_sorted = torch.sort(actual.to(torch.int64), dim=1).values
    expected_sorted = torch.sort(expected.to(torch.int64), dim=1).values
    if not torch.equal(actual_sorted, expected_sorted):
        mismatch = actual_sorted != expected_sorted
        mismatch_entries = int(torch.count_nonzero(mismatch).item())
        first_row = int(torch.nonzero(mismatch, as_tuple=False)[0, 0].item())
        actual_preview = actual_sorted[first_row, :16].detach().cpu().tolist()
        expected_preview = expected_sorted[first_row, :16].detach().cpu().tolist()
        raise BatteryFailure(
            "topk set mismatch: "
            f"entries={mismatch_entries} row={first_row} "
            f"actual_first16={actual_preview} expected_first16={expected_preview}"
        )
    return {
        "topk_entries": int(actual.numel()),
        "topk_rows": int(actual.shape[0]),
    }


def _topk_sets_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.shape != expected.shape:
        return False
    actual_sorted = torch.sort(actual.to(torch.int64), dim=1).values
    expected_sorted = torch.sort(expected.to(torch.int64), dim=1).values
    return bool(torch.equal(actual_sorted, expected_sorted))


def _suffix_paged_topk(
    *,
    page_table_1: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    rows = int(page_table_1.shape[0])
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=page_table_1.device)
    for row in range(rows):
        seq_len = int(seqlens[row].item())
        valid = min(max(seq_len, 0), int(page_table_1.shape[1]))
        keep = min(valid, int(topk))
        if keep <= 0:
            continue
        start = valid - keep
        out[row, :keep] = page_table_1[row, start:valid]
    return out


def _suffix_ragged_topk(
    *,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    rows = int(k_start.shape[0])
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=k_start.device)
    for row in range(rows):
        ks = int(k_start[row].item())
        ke = int(k_end[row].item())
        keep = min(max(ke - ks, 0), int(topk))
        if keep <= 0:
            continue
        out[row, :keep] = torch.arange(ke - keep, ke, dtype=torch.int32, device=k_start.device)
    return out


def _cu_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    out = torch.empty((int(lengths.numel()) + 1,), dtype=torch.int32, device=lengths.device)
    out[0] = 0
    if lengths.numel():
        out[1:] = torch.cumsum(lengths.to(torch.int32), dim=0)
    return out


def _topk_gap_diagnostics(
    *,
    logits: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    topk: int,
    ambiguity_epsilon: float = 1e-4,
) -> dict[str, int]:
    checked_rows = 0
    ambiguous_rows = 0
    for row in range(logits.shape[0]):
        ks = max(0, int(k_start[row].item()))
        ke = min(int(k_end[row].item()), int(logits.shape[1]))
        width = ke - ks
        if width <= 0:
            continue
        gather_k = min(int(topk), width)
        if gather_k >= width:
            continue
        checked_rows += 1
        values = torch.topk(logits[row, ks:ke], k=gather_k + 1, largest=True, sorted=True).values
        gap = float((values[gather_k - 1] - values[gather_k]).abs().item())
        if gap <= ambiguity_epsilon:
            ambiguous_rows += 1
    return {
        "topk_margin_checked_rows": checked_rows,
        "topk_margin_ambiguous_rows": ambiguous_rows,
    }


def _run_indexer_paged_eager(case: IndexerPagedCase, device: torch.device) -> dict[str, float | int]:
    seqlens = torch.tensor(case.row_seqlens_a, dtype=torch.int32, device=device)
    real_page_table, _real_cpu, pool_pages = _make_disjoint_real_page_table(
        seqlens=case.row_seqlens_a,
        graph_width_tokens=case.graph_width_tokens,
        seed=case.seed,
        device=device,
    )
    index_k_cache = _make_random_index_cache(
        pool_pages=pool_pages,
        seed=case.seed + 10,
        device=device,
    )
    q_fp8, weights = _make_index_q_and_weights(
        q_rows=case.q_rows,
        index_heads=case.index_heads,
        seed=case.seed + 20,
        device=device,
    )
    workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=case.topk,
        max_total_q=case.q_rows,
        max_batch=case.q_rows,
        max_paged_q_rows=case.q_rows,
        max_page_table_width=real_page_table.shape[1],
        mla_heads=1,
        index_heads=case.index_heads,
    )
    metadata = IndexerPagedDecodeMetadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        paged_mqa_schedule_metadata=build_paged_mqa_schedule_metadata(seqlens.contiguous(), PAGE_SIZE),
    )
    actual = paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        page_size=PAGE_SIZE,
        contract_phantoms=workspace.get_paged_indexer_contract_phantoms(),
        workspace=workspace,
    )
    expected = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(case.q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
        page_size=PAGE_SIZE,
    )
    torch.cuda.synchronize(device)
    return _assert_indexer_logits_close(
        actual=actual,
        expected=expected,
        require_invalid_neginf=True,
    )


def _run_indexer_paged_graph(case: IndexerPagedCase, device: torch.device) -> dict[str, float | int]:
    if case.row_seqlens_b is None:
        raise BatteryFailure("graph paged case requires row_seqlens_b")
    seqlens_a = torch.tensor(case.row_seqlens_a, dtype=torch.int32, device=device)
    seqlens_b = torch.tensor(case.row_seqlens_b, dtype=torch.int32, device=device)
    real_a, _real_a_cpu, pool_pages_a = _make_disjoint_real_page_table(
        seqlens=case.row_seqlens_a,
        graph_width_tokens=case.graph_width_tokens,
        seed=case.seed,
        device=device,
    )
    real_b, _real_b_cpu, pool_pages_b = _make_disjoint_real_page_table(
        seqlens=case.row_seqlens_b,
        graph_width_tokens=case.graph_width_tokens,
        seed=case.seed + 1,
        device=device,
    )
    pool_pages = max(pool_pages_a, pool_pages_b)
    index_k_cache = _make_random_index_cache(
        pool_pages=pool_pages,
        seed=case.seed + 10,
        device=device,
    )
    q_fp8, weights = _make_index_q_and_weights(
        q_rows=case.q_rows,
        index_heads=case.index_heads,
        seed=case.seed + 20,
        device=device,
    )

    graph_pages = _page_count(case.graph_width_tokens)
    graph_real = torch.full(
        (case.q_rows, graph_pages),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((case.q_rows,), dtype=torch.int32, device=device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    graph_schedule = torch.empty((num_sms + 1, 2), dtype=torch.int32, device=device)
    workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=case.topk,
        max_total_q=case.q_rows,
        max_batch=case.q_rows,
        max_paged_q_rows=case.q_rows,
        max_page_table_width=graph_pages,
        mla_heads=1,
        index_heads=case.index_heads,
        use_cuda_graph=True,
    )
    metadata = IndexerPagedDecodeMetadata(
        real_page_table=graph_real,
        cache_seqlens_int32=graph_seqlens,
        paged_mqa_schedule_metadata=graph_schedule,
    )

    def prepare(real_page_table: torch.Tensor, seqlens: torch.Tensor) -> None:
        graph_real.fill_(-1)
        graph_real[:, : real_page_table.shape[1]].copy_(real_page_table)
        graph_seqlens.copy_(seqlens)
        build_paged_mqa_schedule_metadata(
            graph_seqlens.contiguous(),
            PAGE_SIZE,
            num_sms,
            out=graph_schedule,
        )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = paged_decode_logits(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            page_size=PAGE_SIZE,
            contract_phantoms=workspace.get_paged_indexer_contract_phantoms(),
            workspace=workspace,
        )
        return captured_out

    clear_indexer_caches()
    prepare(real_a, seqlens_a)
    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)
    if captured_out is None:
        raise BatteryFailure("graph did not produce indexer output")
    actual_a = captured_out.clone()
    expected_a = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real,
        query_row_to_batch=torch.arange(case.q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
        page_size=PAGE_SIZE,
    )
    metrics_a = _assert_indexer_logits_close(
        actual=actual_a,
        expected=expected_a,
        require_invalid_neginf=True,
    )

    prepare(real_b, seqlens_b)
    graph.replay()
    torch.cuda.synchronize(device)
    actual_b = captured_out.clone()
    expected_b = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real,
        query_row_to_batch=torch.arange(case.q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
        page_size=PAGE_SIZE,
    )
    metrics_b = _assert_indexer_logits_close(
        actual=actual_b,
        expected=expected_b,
        require_invalid_neginf=True,
    )
    return {
        "replay_a_finite_logits": int(metrics_a["finite_logits"]),
        "replay_b_finite_logits": int(metrics_b["finite_logits"]),
        "replay_a_max_abs": float(metrics_a["max_abs"]),
        "replay_b_max_abs": float(metrics_b["max_abs"]),
    }


def _build_extend_compact_layout(
    *,
    request_shapes: tuple[tuple[int, int], ...],
    topk: int,
    index_heads: int,
    seed: int,
    device: torch.device,
    structured: bool,
) -> dict[str, torch.Tensor | int]:
    del topk
    q_rows = sum(q_len for _prefix, q_len in request_shapes)
    total_k = sum(prefix + q_len for prefix, q_len in request_shapes)
    padded_k = _align_up(total_k, PAGE_SIZE)

    if structured:
        k = torch.zeros((padded_k, INDEX_HEAD_DIM), dtype=torch.float32, device=device)
        cursor = 0
        for prefix, q_len in request_shapes:
            seq_len = prefix + q_len
            values = torch.linspace(0.01, 1.01, seq_len, dtype=torch.float32, device=device)
            k[cursor : cursor + seq_len] = values.unsqueeze(1)
            cursor += seq_len
    else:
        k = _randn(
            (padded_k, INDEX_HEAD_DIM),
            seed=seed + 30,
            device=device,
            scale=0.35,
        )

    k_quant, k_scale = _quantize_rows_to_index_kv_fp8(k)
    q_fp8, weights = _make_index_q_and_weights(
        q_rows=q_rows,
        index_heads=index_heads,
        seed=seed + 40,
        device=device,
        structured=structured,
    )
    k_start = torch.empty((q_rows,), dtype=torch.int32, device=device)
    k_end = torch.empty((q_rows,), dtype=torch.int32, device=device)
    row = 0
    cursor = 0
    for prefix, q_len in request_shapes:
        for token_idx in range(q_len):
            k_start[row] = cursor
            k_end[row] = cursor + prefix + token_idx + 1
            row += 1
        cursor += prefix + q_len
    return {
        "q_fp8": q_fp8,
        "weights": weights,
        "k_quant": k_quant,
        "k_scale": k_scale,
        "k_start": k_start,
        "k_end": k_end,
        "q_rows": q_rows,
        "total_k": total_k,
        "padded_k": padded_k,
    }


def _run_indexer_extend_eager(case: IndexerExtendCase, device: torch.device) -> dict[str, float | int]:
    layout = _build_extend_compact_layout(
        request_shapes=case.request_shapes,
        topk=case.topk,
        index_heads=case.index_heads,
        seed=case.seed,
        device=device,
        structured=False,
    )
    q_rows = int(layout["q_rows"])
    padded_k = int(layout["padded_k"])
    workspace = _make_workspace(
        mode="extend",
        device=device,
        topk=case.topk,
        max_total_q=q_rows,
        max_batch=len(case.request_shapes),
        max_kv_rows=padded_k,
        mla_heads=1,
        index_heads=case.index_heads,
    )
    metadata = IndexerExtendMetadata(
        k_start=layout["k_start"],
        k_end=layout["k_end"],
    )
    actual = extend_logits(
        q_fp8=layout["q_fp8"],
        weights=layout["weights"],
        kv_fp8=(layout["k_quant"], layout["k_scale"]),
        metadata=metadata,
        contract_phantoms=workspace.get_indexer_contract_phantoms(),
        workspace=workspace,
        preinitialize_invalid_logits=case.preinitialize_invalid_logits,
    )
    expected = extend_logits_reference(
        q_fp8=layout["q_fp8"],
        weights=layout["weights"],
        kv_fp8=(layout["k_quant"], layout["k_scale"]),
        k_start=layout["k_start"],
        k_end=layout["k_end"],
    )
    torch.cuda.synchronize(device)
    metrics = _assert_indexer_logits_close(
        actual=actual,
        expected=expected,
        require_invalid_neginf=case.preinitialize_invalid_logits,
    )
    topk_metrics = _topk_gap_diagnostics(
        logits=expected,
        k_start=layout["k_start"],
        k_end=layout["k_end"],
        topk=case.topk,
    )
    return {**metrics, **topk_metrics}


def _make_random_selected_table(
    *,
    rows: int,
    cache_len: int,
    pool_tokens: int,
    width: int,
    valid_pattern: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pool_tokens < cache_len:
        raise BatteryFailure("pool_tokens must cover cache_len")
    selected = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    active_counts = torch.empty((rows,), dtype=torch.int32, device=device)
    gen = _cpu_generator(seed)
    for row in range(rows):
        valid = max(0, min(width, cache_len, valid_pattern[row % len(valid_pattern)]))
        active_counts[row] = valid
        if valid == 0:
            continue
        positions = torch.randperm(cache_len, generator=gen, dtype=torch.int64)[:valid]
        selected[row, :valid] = positions.to(device=device, dtype=torch.int32)
    return selected.contiguous(), active_counts.contiguous()


def _run_mla_eager(case: MLACase, device: torch.device) -> dict[str, float | int]:
    pool_tokens = _align_up(max(case.cache_len, 1), PAGE_SIZE)
    _k_nope, _k_rope, packed = _make_mla_pool(
        pool_tokens=pool_tokens,
        seed=case.seed,
        device=device,
    )
    q_all = _make_q_all(
        q_rows=case.q_rows,
        mla_heads=case.mla_heads,
        seed=case.seed + 20,
        device=device,
    )
    selected, active_counts = _make_random_selected_table(
        rows=case.q_rows,
        cache_len=case.cache_len,
        pool_tokens=pool_tokens,
        width=case.topk,
        valid_pattern=case.valid_pattern,
        seed=case.seed + 30,
        device=device,
    )
    cache_seqlens = torch.full((case.batch,), case.cache_len, dtype=torch.int32, device=device)
    workspace_mode = "verify" if case.mode == "target_verify" else case.mode
    workspace = _make_workspace(
        mode=workspace_mode,
        device=device,
        topk=case.topk,
        max_total_q=case.q_rows,
        max_batch=case.batch,
        max_kv_rows=pool_tokens if workspace_mode != "decode" else 0,
        mla_heads=case.mla_heads,
        index_heads=1,
    )
    if case.mode == "decode":
        metadata = MLASparseDecodeMetadata(
            page_table_1=selected,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=active_counts,
            max_seq_len_k=case.cache_len,
        )
        actual = sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=packed,
            page_table_1=metadata.page_table_1,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
            workspace=workspace,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
    else:
        q_per_batch = max(1, math.ceil(case.q_rows / case.batch))
        cu_q = torch.arange(0, case.batch + 1, dtype=torch.int32, device=device) * q_per_batch
        cu_q[-1] = case.q_rows
        cu_k = torch.arange(0, case.batch + 1, dtype=torch.int32, device=device) * case.cache_len
        metadata = MLASparseExtendMetadata(
            selected_token_offsets=selected,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=active_counts,
            nsa_cu_seqlens_q=cu_q,
            nsa_cu_seqlens_k=cu_k,
            max_seq_len_q=q_per_batch,
            max_seq_len_k=case.cache_len,
            mode=case.mode,
        )
        actual = sparse_mla_extend_forward(
            q_all=q_all,
            kv_cache=packed,
            selected_token_offsets=metadata.selected_token_offsets,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
            workspace=workspace,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
    expected = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=selected,
        active_token_counts=active_counts,
        sm_scale=MLA_SM_SCALE,
        v_head_dim=MLA_V_HEAD_DIM,
    )
    torch.cuda.synchronize(device)
    metrics = _assert_mla_close(actual=actual, expected=expected)
    return {
        **metrics,
        "rows": case.q_rows,
        "heads": case.mla_heads,
        "topk": case.topk,
        "active_tokens": int(active_counts.sum().item()),
    }


def _run_mla_graph(case: MLACase, device: torch.device) -> dict[str, float | int]:
    if case.mode != "decode":
        raise BatteryFailure("MLA graph case currently expects decode mode")
    pool_tokens = _align_up(max(case.cache_len, 1), PAGE_SIZE)
    _k_nope, _k_rope, packed = _make_mla_pool(
        pool_tokens=pool_tokens,
        seed=case.seed,
        device=device,
    )
    q_all = _make_q_all(
        q_rows=case.q_rows,
        mla_heads=case.mla_heads,
        seed=case.seed + 20,
        device=device,
    )
    selected_a, active_a = _make_random_selected_table(
        rows=case.q_rows,
        cache_len=case.cache_len,
        pool_tokens=pool_tokens,
        width=case.topk,
        valid_pattern=case.valid_pattern,
        seed=case.seed + 30,
        device=device,
    )
    selected_b, active_b = _make_random_selected_table(
        rows=case.q_rows,
        cache_len=case.cache_len,
        pool_tokens=pool_tokens,
        width=case.topk,
        valid_pattern=tuple(reversed(case.valid_pattern)),
        seed=case.seed + 31,
        device=device,
    )
    graph_selected = torch.full_like(selected_a, -1)
    graph_active = torch.empty_like(active_a)
    graph_cache_seqlens = torch.full((case.batch,), case.cache_len, dtype=torch.int32, device=device)
    workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=case.topk,
        max_total_q=case.q_rows,
        max_batch=case.batch,
        mla_heads=case.mla_heads,
        index_heads=1,
        use_cuda_graph=True,
    )
    metadata = MLASparseDecodeMetadata(
        page_table_1=graph_selected,
        cache_seqlens_int32=graph_cache_seqlens,
        nsa_cache_seqlens_int32=graph_active,
        max_seq_len_k=case.cache_len,
    )

    def prepare(selected: torch.Tensor, active: torch.Tensor) -> None:
        graph_selected.copy_(selected)
        graph_active.copy_(active)

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=packed,
            page_table_1=metadata.page_table_1,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
            workspace=workspace,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
        return captured_out

    clear_mla_caches()
    prepare(selected_a, active_a)
    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)
    if captured_out is None:
        raise BatteryFailure("graph did not produce MLA output")
    actual_a = captured_out.clone()
    expected_a = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=graph_selected,
        active_token_counts=graph_active,
        sm_scale=MLA_SM_SCALE,
        v_head_dim=MLA_V_HEAD_DIM,
    )
    metrics_a = _assert_mla_close(actual=actual_a, expected=expected_a)

    prepare(selected_b, active_b)
    graph.replay()
    torch.cuda.synchronize(device)
    actual_b = captured_out.clone()
    expected_b = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=graph_selected,
        active_token_counts=graph_active,
        sm_scale=MLA_SM_SCALE,
        v_head_dim=MLA_V_HEAD_DIM,
    )
    metrics_b = _assert_mla_close(actual=actual_b, expected=expected_b)
    return {
        "replay_a_max_abs": float(metrics_a["max_abs"]),
        "replay_b_max_abs": float(metrics_b["max_abs"]),
        "replay_a_rmse": float(metrics_a["rmse"]),
        "replay_b_rmse": float(metrics_b["rmse"]),
    }


def _run_e2e_decode_eager(case: E2EDecodeCase, device: torch.device) -> dict[str, float | int]:
    q_rows = len(case.row_seqlens_a)
    seqlens = torch.tensor(case.row_seqlens_a, dtype=torch.int32, device=device)
    real_page_table, real_cpu, pool_pages = _make_disjoint_real_page_table(
        seqlens=case.row_seqlens_a,
        graph_width_tokens=case.graph_width_tokens,
        seed=case.seed,
        device=device,
    )
    page_table_1 = _page_table_1_from_real(real_page_table=real_page_table, seqlens=seqlens)
    index_k_cache = _make_structured_index_cache(
        real_page_table_cpu=real_cpu,
        seqlens=case.row_seqlens_a,
        pool_pages=pool_pages,
        seed=case.seed + 1,
        device=device,
    )
    _k_nope, _k_rope, packed = _make_mla_pool(
        pool_tokens=pool_pages * PAGE_SIZE,
        seed=case.seed + 2,
        device=device,
    )
    q_fp8, weights = _make_index_q_and_weights(
        q_rows=q_rows,
        index_heads=case.index_heads,
        seed=case.seed + 3,
        device=device,
        structured=True,
    )
    q_all = _make_q_all(
        q_rows=q_rows,
        mla_heads=case.mla_heads,
        seed=case.seed + 4,
        device=device,
    )
    indexer_workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=case.topk,
        max_total_q=q_rows,
        max_batch=q_rows,
        max_paged_q_rows=q_rows,
        max_page_table_width=real_page_table.shape[1],
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
    )
    metadata = IndexerPagedDecodeMetadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        paged_mqa_schedule_metadata=build_paged_mqa_schedule_metadata(seqlens.contiguous(), PAGE_SIZE),
    )
    logits = paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        page_size=PAGE_SIZE,
        contract_phantoms=indexer_workspace.get_paged_indexer_contract_phantoms(),
        workspace=indexer_workspace,
    )
    expected_logits = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
        page_size=PAGE_SIZE,
    )
    logit_metrics = _assert_indexer_logits_close(
        actual=logits,
        expected=expected_logits,
        require_invalid_neginf=True,
    )
    selected = _select_paged_topk_from_logits(
        logits=logits,
        page_table_1=page_table_1,
        topk=case.topk,
    )
    expected_selected = _select_paged_topk_from_logits(
        logits=expected_logits,
        page_table_1=page_table_1,
        topk=case.topk,
    )
    topk_metrics = _assert_topk_equal(selected, expected_selected)
    active_counts = torch.count_nonzero(selected >= 0, dim=1).to(torch.int32)
    mla_workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=case.topk,
        max_total_q=q_rows,
        max_batch=q_rows,
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
    )
    mla_metadata = MLASparseDecodeMetadata(
        page_table_1=selected,
        cache_seqlens_int32=seqlens,
        nsa_cache_seqlens_int32=active_counts,
        max_seq_len_k=int(seqlens.max().item()),
    )
    actual_mla = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=mla_metadata.page_table_1,
        cache_seqlens_int32=mla_metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=mla_metadata.nsa_cache_seqlens_int32,
        workspace=mla_workspace,
        sm_scale=MLA_SM_SCALE,
        v_head_dim=MLA_V_HEAD_DIM,
    )
    expected_mla = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=expected_selected,
        active_token_counts=active_counts,
        sm_scale=MLA_SM_SCALE,
        v_head_dim=MLA_V_HEAD_DIM,
    )
    torch.cuda.synchronize(device)
    mla_metrics = _assert_mla_close(actual=actual_mla, expected=expected_mla)
    return {
        "logit_max_abs": float(logit_metrics["max_abs"]),
        **topk_metrics,
        "mla_max_abs": float(mla_metrics["max_abs"]),
        "mla_rmse": float(mla_metrics["rmse"]),
        "mla_cos": float(mla_metrics["cos"]),
    }


def _run_e2e_decode_graph(case: E2EDecodeCase, device: torch.device) -> dict[str, float | int]:
    if case.row_seqlens_b is None:
        raise BatteryFailure("graph e2e case requires row_seqlens_b")
    q_rows = len(case.row_seqlens_a)
    seqlens_a = torch.tensor(case.row_seqlens_a, dtype=torch.int32, device=device)
    seqlens_b = torch.tensor(case.row_seqlens_b, dtype=torch.int32, device=device)
    real_a, real_a_cpu, pool_pages_a = _make_disjoint_real_page_table(
        seqlens=case.row_seqlens_a,
        graph_width_tokens=case.graph_width_tokens,
        seed=case.seed,
        device=device,
    )
    real_b, real_b_cpu, pool_pages_b = _make_disjoint_real_page_table(
        seqlens=case.row_seqlens_b,
        graph_width_tokens=case.graph_width_tokens,
        seed=case.seed + 1,
        device=device,
    )
    pool_pages = max(pool_pages_a, pool_pages_b)
    index_k_cache_a = _make_structured_index_cache(
        real_page_table_cpu=real_a_cpu,
        seqlens=case.row_seqlens_a,
        pool_pages=pool_pages,
        seed=case.seed + 2,
        device=device,
    )
    index_k_cache_b = _make_structured_index_cache(
        real_page_table_cpu=real_b_cpu,
        seqlens=case.row_seqlens_b,
        pool_pages=pool_pages,
        seed=case.seed + 3,
        device=device,
    )
    index_k_cache = torch.empty_like(index_k_cache_a)
    _k_nope, _k_rope, packed = _make_mla_pool(
        pool_tokens=pool_pages * PAGE_SIZE,
        seed=case.seed + 4,
        device=device,
    )
    q_fp8, weights = _make_index_q_and_weights(
        q_rows=q_rows,
        index_heads=case.index_heads,
        seed=case.seed + 5,
        device=device,
        structured=True,
    )
    q_all = _make_q_all(
        q_rows=q_rows,
        mla_heads=case.mla_heads,
        seed=case.seed + 6,
        device=device,
    )
    graph_pages = _page_count(case.graph_width_tokens)
    graph_real = torch.full((q_rows, graph_pages), -1, dtype=torch.int32, device=device)
    graph_page_table_1 = torch.full(
        (q_rows, graph_pages * PAGE_SIZE),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((q_rows,), dtype=torch.int32, device=device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    graph_schedule = torch.empty((num_sms + 1, 2), dtype=torch.int32, device=device)
    indexer_workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=case.topk,
        max_total_q=q_rows,
        max_batch=q_rows,
        max_paged_q_rows=q_rows,
        max_page_table_width=graph_pages,
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
        use_cuda_graph=True,
    )
    mla_workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=case.topk,
        max_total_q=q_rows,
        max_batch=q_rows,
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
        use_cuda_graph=True,
    )
    indexer_metadata = IndexerPagedDecodeMetadata(
        real_page_table=graph_real,
        cache_seqlens_int32=graph_seqlens,
        paged_mqa_schedule_metadata=graph_schedule,
    )

    def prepare(
        *,
        real_page_table: torch.Tensor,
        seqlens: torch.Tensor,
        page_table_1: torch.Tensor,
        cache: torch.Tensor,
    ) -> None:
        graph_real.fill_(-1)
        graph_real[:, : real_page_table.shape[1]].copy_(real_page_table)
        graph_page_table_1.fill_(-1)
        graph_page_table_1[:, : page_table_1.shape[1]].copy_(page_table_1)
        graph_seqlens.copy_(seqlens)
        index_k_cache.copy_(cache)
        build_paged_mqa_schedule_metadata(
            graph_seqlens.contiguous(),
            PAGE_SIZE,
            num_sms,
            out=graph_schedule,
        )

    captured_logits: torch.Tensor | None = None
    captured_selected: torch.Tensor | None = None
    captured_mla: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_logits, captured_selected, captured_mla
        captured_logits = paged_decode_logits(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=indexer_metadata,
            page_size=PAGE_SIZE,
            contract_phantoms=indexer_workspace.get_paged_indexer_contract_phantoms(),
            workspace=indexer_workspace,
        )
        captured_selected = _select_paged_topk_from_logits(
            logits=captured_logits,
            page_table_1=graph_page_table_1,
            topk=case.topk,
        )
        active_counts = torch.count_nonzero(captured_selected >= 0, dim=1).to(torch.int32)
        mla_metadata = MLASparseDecodeMetadata(
            page_table_1=captured_selected,
            cache_seqlens_int32=graph_seqlens,
            nsa_cache_seqlens_int32=active_counts,
            max_seq_len_k=case.graph_width_tokens,
        )
        captured_mla = sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=packed,
            page_table_1=mla_metadata.page_table_1,
            cache_seqlens_int32=mla_metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=mla_metadata.nsa_cache_seqlens_int32,
            workspace=mla_workspace,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
        return captured_mla

    clear_indexer_caches()
    clear_mla_caches()
    page_table_a = _page_table_1_from_real(real_page_table=real_a, seqlens=seqlens_a)
    page_table_b = _page_table_1_from_real(real_page_table=real_b, seqlens=seqlens_b)
    prepare(real_page_table=real_a, seqlens=seqlens_a, page_table_1=page_table_a, cache=index_k_cache_a)
    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    replay_metrics: dict[str, float | int] = {}
    for label, real, seqlens, page_table, cache in (
        ("a", real_a, seqlens_a, page_table_a, index_k_cache_a),
        ("b", real_b, seqlens_b, page_table_b, index_k_cache_b),
    ):
        prepare(real_page_table=real, seqlens=seqlens, page_table_1=page_table, cache=cache)
        graph.replay()
        torch.cuda.synchronize(device)
        if captured_logits is None or captured_selected is None or captured_mla is None:
            raise BatteryFailure("graph e2e did not produce all outputs")
        actual_logits = captured_logits.clone()
        actual_selected = captured_selected.clone()
        actual_mla = captured_mla.clone()
        expected_logits = paged_decode_logits_reference(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            real_page_table=graph_real,
            query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
            seqlens_per_query=graph_seqlens,
            page_size=PAGE_SIZE,
        )
        logit_metrics = _assert_indexer_logits_close(
            actual=actual_logits,
            expected=expected_logits,
            require_invalid_neginf=True,
        )
        expected_selected = _select_paged_topk_from_logits(
            logits=expected_logits,
            page_table_1=graph_page_table_1,
            topk=case.topk,
        )
        _assert_topk_equal(actual_selected, expected_selected)
        active_counts = torch.count_nonzero(expected_selected >= 0, dim=1).to(torch.int32)
        expected_mla = sparse_mla_reference(
            q_all=q_all,
            kv_cache=packed,
            page_table_1=expected_selected,
            active_token_counts=active_counts,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
        mla_metrics = _assert_mla_close(actual=actual_mla, expected=expected_mla)
        replay_metrics[f"replay_{label}_logit_max_abs"] = float(logit_metrics["max_abs"])
        replay_metrics[f"replay_{label}_mla_max_abs"] = float(mla_metrics["max_abs"])
        replay_metrics[f"replay_{label}_mla_rmse"] = float(mla_metrics["rmse"])
    return replay_metrics


def _run_e2e_extend_eager(case: E2EExtendCase, device: torch.device) -> dict[str, float | int]:
    q_rows = sum(q_len for _prefix, q_len in case.request_shapes)
    total_k = sum(prefix + q_len for prefix, q_len in case.request_shapes)
    padded_k = _align_up(total_k, PAGE_SIZE)
    pool_tokens = padded_k * 2
    row_ids = torch.randperm(
        pool_tokens,
        generator=_cpu_generator(case.seed),
        dtype=torch.int64,
    )[:total_k].to(device=device, dtype=torch.int32)
    padded_row_ids = torch.full((padded_k,), 0, dtype=torch.int32, device=device)
    padded_row_ids[:total_k] = row_ids

    k_compact = torch.zeros((padded_k, INDEX_HEAD_DIM), dtype=torch.float32, device=device)
    cursor = 0
    for prefix, q_len in case.request_shapes:
        seq_len = prefix + q_len
        values = torch.linspace(0.01, 1.01, seq_len, dtype=torch.float32, device=device)
        k_compact[cursor : cursor + seq_len] = values.unsqueeze(1)
        cursor += seq_len
    k_quant, k_scale = _quantize_rows_to_index_kv_fp8(k_compact)
    q_fp8, weights = _make_index_q_and_weights(
        q_rows=q_rows,
        index_heads=case.index_heads,
        seed=case.seed + 1,
        device=device,
        structured=True,
    )
    k_start = torch.empty((q_rows,), dtype=torch.int32, device=device)
    k_end = torch.empty((q_rows,), dtype=torch.int32, device=device)
    row = 0
    cursor = 0
    cache_seqlens = []
    for prefix, q_len in case.request_shapes:
        cache_seqlens.append(prefix + q_len)
        for token_idx in range(q_len):
            k_start[row] = cursor
            k_end[row] = cursor + prefix + token_idx + 1
            row += 1
        cursor += prefix + q_len
    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)

    indexer_workspace = _make_workspace(
        mode="extend",
        device=device,
        topk=case.topk,
        max_total_q=q_rows,
        max_batch=len(case.request_shapes),
        max_kv_rows=padded_k,
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
    )
    metadata = IndexerExtendMetadata(k_start=k_start, k_end=k_end)
    logits = extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=(k_quant, k_scale),
        metadata=metadata,
        contract_phantoms=indexer_workspace.get_indexer_contract_phantoms(),
        workspace=indexer_workspace,
        preinitialize_invalid_logits=False,
    )
    expected_logits = extend_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=(k_quant, k_scale),
        k_start=k_start,
        k_end=k_end,
    )
    logit_metrics = _assert_indexer_logits_close(
        actual=logits,
        expected=expected_logits,
        require_invalid_neginf=False,
    )
    compact_topk = _select_ragged_topk_from_logits(
        logits=logits,
        k_start=k_start,
        k_end=k_end,
        topk=case.topk,
    )
    expected_compact_topk = _select_ragged_topk_from_logits(
        logits=expected_logits,
        k_start=k_start,
        k_end=k_end,
        topk=case.topk,
    )
    topk_metrics = _assert_topk_equal(compact_topk, expected_compact_topk)
    selected = _map_compact_topk_to_physical(compact_topk=compact_topk, row_ids=padded_row_ids)
    expected_selected = _map_compact_topk_to_physical(
        compact_topk=expected_compact_topk,
        row_ids=padded_row_ids,
    )
    _assert_topk_equal(selected, expected_selected)

    _k_nope, _k_rope, packed = _make_mla_pool(
        pool_tokens=pool_tokens,
        seed=case.seed + 2,
        device=device,
    )
    q_all = _make_q_all(
        q_rows=q_rows,
        mla_heads=case.mla_heads,
        seed=case.seed + 3,
        device=device,
    )
    active_counts = torch.count_nonzero(selected >= 0, dim=1).to(torch.int32)
    q_cu = torch.empty((len(case.request_shapes) + 1,), dtype=torch.int32, device=device)
    k_cu = torch.empty_like(q_cu)
    q_cu[0] = 0
    k_cu[0] = 0
    q_cursor = 0
    k_cursor = 0
    for idx, (prefix, q_len) in enumerate(case.request_shapes):
        q_cursor += q_len
        k_cursor += prefix + q_len
        q_cu[idx + 1] = q_cursor
        k_cu[idx + 1] = k_cursor
    mla_workspace = _make_workspace(
        mode="extend",
        device=device,
        topk=case.topk,
        max_total_q=q_rows,
        max_batch=len(case.request_shapes),
        max_kv_rows=padded_k,
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
    )
    mla_metadata = MLASparseExtendMetadata(
        selected_token_offsets=selected,
        cache_seqlens_int32=cache_seqlens_t,
        nsa_cache_seqlens_int32=active_counts,
        nsa_cu_seqlens_q=q_cu,
        nsa_cu_seqlens_k=k_cu,
        max_seq_len_q=max(q_len for _prefix, q_len in case.request_shapes),
        max_seq_len_k=max(cache_seqlens),
        mode="extend",
    )
    actual_mla = sparse_mla_extend_forward(
        q_all=q_all,
        kv_cache=packed,
        selected_token_offsets=mla_metadata.selected_token_offsets,
        cache_seqlens_int32=mla_metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=mla_metadata.nsa_cache_seqlens_int32,
        workspace=mla_workspace,
        sm_scale=MLA_SM_SCALE,
        v_head_dim=MLA_V_HEAD_DIM,
    )
    expected_mla = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=expected_selected,
        active_token_counts=active_counts,
        sm_scale=MLA_SM_SCALE,
        v_head_dim=MLA_V_HEAD_DIM,
    )
    torch.cuda.synchronize(device)
    mla_metrics = _assert_mla_close(actual=actual_mla, expected=expected_mla)
    return {
        "logit_max_abs": float(logit_metrics["max_abs"]),
        **topk_metrics,
        "mla_max_abs": float(mla_metrics["max_abs"]),
        "mla_rmse": float(mla_metrics["rmse"]),
        "mla_cos": float(mla_metrics["cos"]),
    }


def _make_sglang_index_pool(
    *,
    sglang: SGLangImports,
    pool_tokens: int,
    device: torch.device,
) -> Any:
    return sglang.NSATokenToKVPool(
        size=int(pool_tokens),
        page_size=PAGE_SIZE,
        kv_lora_rank=MLA_NOPE_DIM,
        dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=MLA_ROPE_DIM,
        layer_num=1,
        device=str(device),
        index_head_dim=INDEX_HEAD_DIM,
        enable_memory_saver=False,
        kv_cache_dim=MLA_PACKED_DIM,
        index_buf_size=int(pool_tokens),
    )


def _set_sglang_index_pool_rows(
    *,
    pool: Any,
    k_rows: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_quant, k_scale = _quantize_rows_to_index_kv_fp8(k_rows)
    loc = torch.arange(k_rows.shape[0], dtype=torch.int64, device=device)
    pool.set_index_k_scale_buffer(0, loc, k_quant, k_scale)
    torch.cuda.synchronize(device)
    return k_quant, k_scale


def _build_sglang_paged_shape(
    *,
    row_seqlens: tuple[int, ...],
    extend_lens: tuple[int, ...],
    graph_width_tokens: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(row_seqlens) != len(extend_lens):
        raise BatteryFailure("row_seqlens and extend_lens must have the same length")
    if any(ext <= 0 for ext in extend_lens):
        raise BatteryFailure("extend_lens must be positive")
    if any(ext > seq_len for seq_len, ext in zip(row_seqlens, extend_lens, strict=True)):
        raise BatteryFailure("extend_lens cannot exceed row_seqlens")

    base_seqlens = torch.tensor(row_seqlens, dtype=torch.int32, device=device)
    real_base, real_cpu, pool_pages = _make_disjoint_real_page_table(
        seqlens=row_seqlens,
        graph_width_tokens=graph_width_tokens,
        seed=seed,
        device=device,
    )
    page_table_1_base = _page_table_1_from_real(
        real_page_table=real_base,
        seqlens=base_seqlens,
    )
    repeats = torch.tensor(extend_lens, dtype=torch.long, device=device)
    real_expanded = torch.repeat_interleave(real_base, repeats=repeats, dim=0)
    page_table_1_expanded = torch.repeat_interleave(page_table_1_base, repeats=repeats, dim=0)

    expanded_cpu: list[int] = []
    for seq_len, ext in zip(row_seqlens, extend_lens, strict=True):
        expanded_cpu.extend(range(seq_len - ext + 1, seq_len + 1))
    seqlens_expanded = torch.tensor(expanded_cpu, dtype=torch.int32, device=device)
    q_group_lens = torch.tensor(extend_lens, dtype=torch.int32, device=device)
    return {
        "base_seqlens": base_seqlens,
        "real_base": real_base,
        "real_cpu": real_cpu,
        "pool_pages": pool_pages,
        "page_table_1_base": page_table_1_base,
        "real_expanded": real_expanded,
        "page_table_1_expanded": page_table_1_expanded,
        "seqlens_expanded": seqlens_expanded,
        "q_group_lens": q_group_lens,
        "q_offset": int(seqlens_expanded.numel()),
    }


def _make_sglang_paged_metadata(
    *,
    sglang: SGLangImports,
    page_table_1: torch.Tensor,
    real_page_table: torch.Tensor,
    base_seqlens: torch.Tensor,
    seqlens_expanded: torch.Tensor,
    q_group_lens: torch.Tensor,
    extend_lens: tuple[int, ...],
    topk: int,
) -> Any:
    nsa_cache = torch.clamp(seqlens_expanded, max=topk).to(torch.int32)
    nsa_cu_k = _cu_from_lengths(nsa_cache)
    attn_metadata = sglang.NSAMetadata(
        page_size=PAGE_SIZE,
        cache_seqlens_int32=base_seqlens,
        max_seq_len_q=int(q_group_lens.max().item()) if q_group_lens.numel() else 1,
        max_seq_len_k=int(base_seqlens.max().item()) if base_seqlens.numel() else 0,
        cu_seqlens_q=_cu_from_lengths(q_group_lens),
        cu_seqlens_k=_cu_from_lengths(base_seqlens),
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        nsa_cache_seqlens_int32=nsa_cache,
        nsa_cu_seqlens_q=torch.arange(
            int(seqlens_expanded.numel()) + 1,
            dtype=torch.int32,
            device=seqlens_expanded.device,
        ),
        nsa_cu_seqlens_k=nsa_cu_k,
        nsa_extend_seq_lens_list=list(extend_lens),
        nsa_seqlens_expanded=seqlens_expanded,
        paged_mqa_schedule_metadata=build_paged_mqa_schedule_metadata(
            seqlens_expanded.contiguous(),
            PAGE_SIZE,
        ),
    )
    return sglang.NSAIndexerMetadata(
        attn_metadata=attn_metadata,
        topk_transform_method=sglang.TopkTransformMethod.PAGED,
        paged_mqa_schedule_metadata=attn_metadata.paged_mqa_schedule_metadata,
    )


def _run_sglang_paged_eager(
    case: SGLangPagedCase,
    device: torch.device,
    sglang_root: Path,
) -> dict[str, float | int | str]:
    sglang = _ensure_sglang_imports(sglang_root)
    shape = _build_sglang_paged_shape(
        row_seqlens=case.row_seqlens_a,
        extend_lens=case.extend_lens,
        graph_width_tokens=case.graph_width_tokens,
        seed=case.seed,
        device=device,
    )
    q_offset = int(shape["q_offset"])
    q_rows = q_offset + int(case.q_padding)
    k_rows = _make_structured_index_rows(
        real_page_table_cpu=shape["real_cpu"],
        seqlens=case.row_seqlens_a,
        pool_pages=int(shape["pool_pages"]),
        device=device,
    )
    index_pool = _make_sglang_index_pool(
        sglang=sglang,
        pool_tokens=int(shape["pool_pages"]) * PAGE_SIZE,
        device=device,
    )
    _set_sglang_index_pool_rows(pool=index_pool, k_rows=k_rows, device=device)
    index_k_cache = index_pool.get_index_k_with_scale_buffer(0)

    q_fp8, weights = _make_index_q_and_weights(
        q_rows=q_rows,
        index_heads=case.index_heads,
        seed=case.seed + 20,
        device=device,
        structured=True,
    )
    workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=DEFAULT_TOPK,
        max_total_q=max(q_offset, 1),
        max_batch=max(q_offset, 1),
        max_paged_q_rows=max(q_offset, 1),
        max_page_table_width=shape["real_expanded"].shape[1],
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
    )
    b12x_metadata = IndexerPagedDecodeMetadata(
        real_page_table=shape["real_expanded"],
        cache_seqlens_int32=shape["seqlens_expanded"],
        paged_mqa_schedule_metadata=build_paged_mqa_schedule_metadata(
            shape["seqlens_expanded"].contiguous(),
            PAGE_SIZE,
        ),
    )
    logits = paged_decode_logits(
        q_fp8=q_fp8[:q_offset],
        weights=weights[:q_offset],
        index_k_cache=index_k_cache,
        metadata=b12x_metadata,
        page_size=PAGE_SIZE,
        contract_phantoms=workspace.get_paged_indexer_contract_phantoms(),
        workspace=workspace,
    )
    expected_logits = paged_decode_logits_reference(
        q_fp8=q_fp8[:q_offset],
        weights=weights[:q_offset],
        index_k_cache=index_k_cache,
        real_page_table=shape["real_expanded"],
        query_row_to_batch=torch.arange(q_offset, dtype=torch.int32, device=device),
        seqlens_per_query=shape["seqlens_expanded"],
        page_size=PAGE_SIZE,
    )
    logit_metrics = _assert_indexer_logits_close(
        actual=logits,
        expected=expected_logits,
        require_invalid_neginf=True,
    )

    indexer_metadata = _make_sglang_paged_metadata(
        sglang=sglang,
        page_table_1=shape["page_table_1_expanded"],
        real_page_table=shape["real_expanded"],
        base_seqlens=shape["base_seqlens"],
        seqlens_expanded=shape["seqlens_expanded"],
        q_group_lens=shape["q_group_lens"],
        extend_lens=case.extend_lens,
        topk=DEFAULT_TOPK,
    )
    topk_kwargs: dict[str, torch.Tensor] = {}
    if case.mode_name in ("target_verify", "draft_extend"):
        topk_kwargs["cu_seqlens_q_cumsum"] = (
            indexer_metadata.attn_metadata.nsa_cu_seqlens_q[: q_offset + 1]
        )
    selected_core = indexer_metadata.topk_transform(logits, DEFAULT_TOPK, **topk_kwargs)
    expected_core = _suffix_paged_topk(
        page_table_1=shape["page_table_1_expanded"],
        seqlens=shape["seqlens_expanded"],
        topk=DEFAULT_TOPK,
    )
    if case.q_padding:
        pad = torch.full(
            (case.q_padding, DEFAULT_TOPK),
            -1,
            dtype=torch.int32,
            device=device,
        )
        selected = torch.cat([selected_core, pad], dim=0)
        expected_selected = torch.cat([expected_core, pad.clone()], dim=0)
    else:
        selected = selected_core
        expected_selected = expected_core
    try:
        topk_metrics = _assert_topk_set_equal(selected, expected_selected)
    except BatteryFailure as exc:
        torch_topk_core = _select_paged_topk_from_logits(
            logits=expected_logits,
            page_table_1=shape["page_table_1_expanded"],
            topk=DEFAULT_TOPK,
        )
        verdict = (
            "torch.topk(logits) matches the structured reference"
            if _topk_sets_equal(torch_topk_core, expected_core)
            else "torch.topk(logits) also differs from the structured reference"
        )
        raise BatteryFailure(f"{exc}; {verdict}") from exc
    metrics: dict[str, float | int | str] = {
        "logit_max_abs": float(logit_metrics["max_abs"]),
        "q_offset": q_offset,
        "q_rows": q_rows,
        **topk_metrics,
    }
    if case.q_padding:
        padded_bad = int(torch.count_nonzero(selected[q_offset:] != -1).item())
        if padded_bad:
            raise BatteryFailure(f"SGLang paged q-padding rows contain {padded_bad} non--1 entries")
        metrics["padded_rows"] = int(case.q_padding)

    if case.run_mla:
        if case.q_padding:
            raise BatteryFailure("run_mla SGLang paged cases should not include q padding")
        _k_nope, _k_rope, packed = _make_mla_pool(
            pool_tokens=int(shape["pool_pages"]) * PAGE_SIZE,
            seed=case.seed + 30,
            device=device,
        )
        q_all = _make_q_all(
            q_rows=q_offset,
            mla_heads=case.mla_heads,
            seed=case.seed + 31,
            device=device,
        )
        active_counts = torch.count_nonzero(selected >= 0, dim=1).to(torch.int32)
        mla_workspace = _make_workspace(
            mode="decode",
            device=device,
            topk=DEFAULT_TOPK,
            max_total_q=q_offset,
            max_batch=q_offset,
            mla_heads=case.mla_heads,
            index_heads=case.index_heads,
        )
        mla_metadata = MLASparseDecodeMetadata(
            page_table_1=selected,
            cache_seqlens_int32=shape["seqlens_expanded"],
            nsa_cache_seqlens_int32=active_counts,
            max_seq_len_k=int(shape["seqlens_expanded"].max().item()),
        )
        actual_mla = sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=packed,
            page_table_1=mla_metadata.page_table_1,
            cache_seqlens_int32=mla_metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=mla_metadata.nsa_cache_seqlens_int32,
            workspace=mla_workspace,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
        expected_mla = sparse_mla_reference(
            q_all=q_all,
            kv_cache=packed,
            page_table_1=expected_selected,
            active_token_counts=active_counts,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
        mla_metrics = _assert_mla_close(actual=actual_mla, expected=expected_mla)
        metrics.update(
            {
                "mla_max_abs": float(mla_metrics["max_abs"]),
                "mla_rmse": float(mla_metrics["rmse"]),
                "mla_cos": float(mla_metrics["cos"]),
            }
        )
    torch.cuda.synchronize(device)
    return metrics


def _run_sglang_paged_graph(
    case: SGLangPagedCase,
    device: torch.device,
    sglang_root: Path,
) -> dict[str, float | int]:
    if case.row_seqlens_b is None:
        raise BatteryFailure("SGLang graph paged case requires row_seqlens_b")
    if case.q_padding:
        raise BatteryFailure("SGLang graph paged cases keep q rows fixed without padding")
    if case.mode_name != "decode":
        raise BatteryFailure("SGLang graph paged case currently covers decode-shaped replay")

    sglang = _ensure_sglang_imports(sglang_root)
    shape_a = _build_sglang_paged_shape(
        row_seqlens=case.row_seqlens_a,
        extend_lens=case.extend_lens,
        graph_width_tokens=case.graph_width_tokens,
        seed=case.seed,
        device=device,
    )
    shape_b = _build_sglang_paged_shape(
        row_seqlens=case.row_seqlens_b,
        extend_lens=case.extend_lens,
        graph_width_tokens=case.graph_width_tokens,
        seed=case.seed + 1,
        device=device,
    )
    q_offset = int(shape_a["q_offset"])
    if int(shape_b["q_offset"]) != q_offset:
        raise BatteryFailure("SGLang graph replay requires a fixed q row count")
    graph_pages = _page_count(case.graph_width_tokens)
    pool_pages = max(int(shape_a["pool_pages"]), int(shape_b["pool_pages"]))
    pool_tokens = pool_pages * PAGE_SIZE
    index_pool = _make_sglang_index_pool(sglang=sglang, pool_tokens=pool_tokens, device=device)
    index_k_cache = index_pool.get_index_k_with_scale_buffer(0)
    k_rows_a = _make_structured_index_rows(
        real_page_table_cpu=shape_a["real_cpu"],
        seqlens=case.row_seqlens_a,
        pool_pages=pool_pages,
        device=device,
    )
    k_rows_b = _make_structured_index_rows(
        real_page_table_cpu=shape_b["real_cpu"],
        seqlens=case.row_seqlens_b,
        pool_pages=pool_pages,
        device=device,
    )

    graph_real = torch.full((q_offset, graph_pages), -1, dtype=torch.int32, device=device)
    graph_page_table_1 = torch.full(
        (q_offset, graph_pages * PAGE_SIZE),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((q_offset,), dtype=torch.int32, device=device)
    graph_base_seqlens = torch.empty((q_offset,), dtype=torch.int32, device=device)
    q_group_lens = torch.ones((q_offset,), dtype=torch.int32, device=device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    graph_schedule = torch.empty((num_sms + 1, 2), dtype=torch.int32, device=device)

    q_fp8, weights = _make_index_q_and_weights(
        q_rows=q_offset,
        index_heads=case.index_heads,
        seed=case.seed + 20,
        device=device,
        structured=True,
    )
    q_all = _make_q_all(
        q_rows=q_offset,
        mla_heads=case.mla_heads,
        seed=case.seed + 21,
        device=device,
    )
    _k_nope, _k_rope, packed = _make_mla_pool(
        pool_tokens=pool_tokens,
        seed=case.seed + 22,
        device=device,
    )
    indexer_workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=DEFAULT_TOPK,
        max_total_q=q_offset,
        max_batch=q_offset,
        max_paged_q_rows=q_offset,
        max_page_table_width=graph_pages,
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
        use_cuda_graph=True,
    )
    mla_workspace = _make_workspace(
        mode="decode",
        device=device,
        topk=DEFAULT_TOPK,
        max_total_q=q_offset,
        max_batch=q_offset,
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
        use_cuda_graph=True,
    )
    b12x_metadata = IndexerPagedDecodeMetadata(
        real_page_table=graph_real,
        cache_seqlens_int32=graph_seqlens,
        paged_mqa_schedule_metadata=graph_schedule,
    )
    indexer_metadata = _make_sglang_paged_metadata(
        sglang=sglang,
        page_table_1=graph_page_table_1,
        real_page_table=graph_real,
        base_seqlens=graph_base_seqlens,
        seqlens_expanded=graph_seqlens,
        q_group_lens=q_group_lens,
        extend_lens=tuple(1 for _ in range(q_offset)),
        topk=DEFAULT_TOPK,
    )

    def prepare(shape: dict[str, Any], k_rows: torch.Tensor) -> None:
        graph_real.fill_(-1)
        graph_real[:, : shape["real_expanded"].shape[1]].copy_(shape["real_expanded"])
        graph_page_table_1.fill_(-1)
        graph_page_table_1[:, : shape["page_table_1_expanded"].shape[1]].copy_(
            shape["page_table_1_expanded"]
        )
        graph_seqlens.copy_(shape["seqlens_expanded"])
        graph_base_seqlens.copy_(shape["seqlens_expanded"])
        build_paged_mqa_schedule_metadata(
            graph_seqlens.contiguous(),
            PAGE_SIZE,
            num_sms,
            out=graph_schedule,
        )
        _set_sglang_index_pool_rows(pool=index_pool, k_rows=k_rows, device=device)

    captured_logits: torch.Tensor | None = None
    captured_selected: torch.Tensor | None = None
    captured_mla: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_logits, captured_selected, captured_mla
        captured_logits = paged_decode_logits(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=b12x_metadata,
            page_size=PAGE_SIZE,
            contract_phantoms=indexer_workspace.get_paged_indexer_contract_phantoms(),
            workspace=indexer_workspace,
        )
        captured_selected = indexer_metadata.topk_transform(captured_logits, DEFAULT_TOPK)
        active_counts = torch.count_nonzero(captured_selected >= 0, dim=1).to(torch.int32)
        mla_metadata = MLASparseDecodeMetadata(
            page_table_1=captured_selected,
            cache_seqlens_int32=graph_seqlens,
            nsa_cache_seqlens_int32=active_counts,
            max_seq_len_k=case.graph_width_tokens,
        )
        captured_mla = sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=packed,
            page_table_1=mla_metadata.page_table_1,
            cache_seqlens_int32=mla_metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=mla_metadata.nsa_cache_seqlens_int32,
            workspace=mla_workspace,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
        return captured_mla

    clear_indexer_caches()
    clear_mla_caches()
    prepare(shape_a, k_rows_a)
    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    replay_metrics: dict[str, float | int] = {}
    for label, shape, k_rows in (("a", shape_a, k_rows_a), ("b", shape_b, k_rows_b)):
        prepare(shape, k_rows)
        graph.replay()
        torch.cuda.synchronize(device)
        if captured_logits is None or captured_selected is None or captured_mla is None:
            raise BatteryFailure("SGLang graph replay did not produce all outputs")
        actual_logits = captured_logits.clone()
        actual_selected = captured_selected.clone()
        actual_mla = captured_mla.clone()
        expected_logits = paged_decode_logits_reference(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            real_page_table=graph_real,
            query_row_to_batch=torch.arange(q_offset, dtype=torch.int32, device=device),
            seqlens_per_query=graph_seqlens,
            page_size=PAGE_SIZE,
        )
        logit_metrics = _assert_indexer_logits_close(
            actual=actual_logits,
            expected=expected_logits,
            require_invalid_neginf=True,
        )
        expected_selected = _suffix_paged_topk(
            page_table_1=graph_page_table_1,
            seqlens=graph_seqlens,
            topk=DEFAULT_TOPK,
        )
        try:
            _assert_topk_set_equal(actual_selected, expected_selected)
        except BatteryFailure as exc:
            torch_topk_selected = _select_paged_topk_from_logits(
                logits=expected_logits,
                page_table_1=graph_page_table_1,
                topk=DEFAULT_TOPK,
            )
            verdict = (
                "torch.topk(logits) matches the structured reference"
                if _topk_sets_equal(torch_topk_selected, expected_selected)
                else "torch.topk(logits) also differs from the structured reference"
            )
            raise BatteryFailure(f"{exc}; {verdict}") from exc
        active_counts = torch.count_nonzero(expected_selected >= 0, dim=1).to(torch.int32)
        expected_mla = sparse_mla_reference(
            q_all=q_all,
            kv_cache=packed,
            page_table_1=expected_selected,
            active_token_counts=active_counts,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
        mla_metrics = _assert_mla_close(actual=actual_mla, expected=expected_mla)
        replay_metrics[f"replay_{label}_logit_max_abs"] = float(logit_metrics["max_abs"])
        replay_metrics[f"replay_{label}_mla_max_abs"] = float(mla_metrics["max_abs"])
        replay_metrics[f"replay_{label}_mla_rmse"] = float(mla_metrics["rmse"])
    return replay_metrics


def _run_sglang_ragged_eager(
    case: SGLangRaggedCase,
    device: torch.device,
    sglang_root: Path,
) -> dict[str, float | int]:
    sglang = _ensure_sglang_imports(sglang_root)
    final_lens = tuple(prefix + q_len for prefix, q_len in case.request_shapes)
    q_lens = tuple(q_len for _prefix, q_len in case.request_shapes)
    q_rows = sum(q_lens)
    q_rows_total = q_rows + int(case.q_padding)
    total_k = sum(final_lens)
    real_page_table, _real_cpu, pool_pages = _make_disjoint_real_page_table(
        seqlens=final_lens,
        graph_width_tokens=max(final_lens),
        seed=case.seed,
        device=device,
    )
    final_lens_t = torch.tensor(final_lens, dtype=torch.int32, device=device)
    page_table_1 = _page_table_1_from_real(
        real_page_table=real_page_table,
        seqlens=final_lens_t,
    )
    row_ids = torch.cat(
        [page_table_1[row, :seq_len] for row, seq_len in enumerate(final_lens)],
        dim=0,
    ).to(torch.int32)
    if int(row_ids.numel()) != total_k:
        raise BatteryFailure("unexpected SGLang ragged row id count")

    padded_total_k = _align_up(total_k, PAGE_SIZE)
    k_compact = torch.zeros((padded_total_k, INDEX_HEAD_DIM), dtype=torch.float32, device=device)
    cursor = 0
    for prefix, q_len in case.request_shapes:
        seq_len = prefix + q_len
        values = torch.linspace(0.01, 1.01, seq_len, dtype=torch.float32, device=device)
        k_compact[cursor : cursor + seq_len] = values.unsqueeze(1)
        cursor += seq_len
    k_quant_expected, k_scale_expected = _quantize_rows_to_index_kv_fp8(k_compact)

    index_pool = _make_sglang_index_pool(
        sglang=sglang,
        pool_tokens=int(pool_pages) * PAGE_SIZE,
        device=device,
    )
    loc = row_ids.to(torch.int64)
    index_pool.set_index_k_scale_buffer(
        0,
        loc,
        k_quant_expected[:total_k],
        k_scale_expected[:total_k],
    )
    torch.cuda.synchronize(device)
    workspace = _make_workspace(
        mode="extend",
        device=device,
        topk=DEFAULT_TOPK,
        max_total_q=max(q_rows, 1),
        max_batch=len(case.request_shapes),
        max_kv_rows=padded_total_k,
        mla_heads=case.mla_heads,
        index_heads=case.index_heads,
    )
    gather_k_out, gather_s_out = workspace.get_indexer_gather_outputs(row_count=total_k)
    k_gather_u8, s_gather_u8 = index_pool.get_index_k_scale_buffer(
        0,
        final_lens_t,
        real_page_table,
        total_k,
        max(final_lens),
        k_out=gather_k_out,
        s_out=gather_s_out,
    )
    torch.cuda.synchronize(device)
    k_gather = k_gather_u8.view(torch.float8_e4m3fn)
    s_gather = s_gather_u8.view(torch.float32).squeeze(-1)
    if not torch.equal(k_gather[:total_k], k_quant_expected[:total_k]):
        mismatch = int(torch.count_nonzero(k_gather[:total_k] != k_quant_expected[:total_k]).item())
        raise BatteryFailure(f"SGLang index K gather mismatch at {mismatch} entries")
    torch.testing.assert_close(s_gather[:total_k], k_scale_expected[:total_k], atol=0.0, rtol=0.0)

    k_start = torch.empty((q_rows,), dtype=torch.int32, device=device)
    k_end = torch.empty((q_rows,), dtype=torch.int32, device=device)
    token_to_batch_idx = torch.empty((q_rows,), dtype=torch.int32, device=device)
    seqlens_expanded_list: list[int] = []
    row = 0
    cursor = 0
    for batch_idx, (prefix, q_len) in enumerate(case.request_shapes):
        for token_idx in range(q_len):
            k_start[row] = cursor
            k_end[row] = cursor + prefix + token_idx + 1
            token_to_batch_idx[row] = batch_idx
            seqlens_expanded_list.append(prefix + token_idx + 1)
            row += 1
        cursor += prefix + q_len
    seqlens_expanded = torch.tensor(seqlens_expanded_list, dtype=torch.int32, device=device)
    q_fp8, weights = _make_index_q_and_weights(
        q_rows=q_rows_total,
        index_heads=case.index_heads,
        seed=case.seed + 20,
        device=device,
        structured=True,
    )
    logits = extend_logits(
        q_fp8=q_fp8[:q_rows],
        weights=weights[:q_rows],
        kv_fp8=(k_gather, s_gather),
        metadata=IndexerExtendMetadata(k_start=k_start, k_end=k_end),
        contract_phantoms=workspace.get_indexer_contract_phantoms(),
        workspace=workspace,
        preinitialize_invalid_logits=False,
    )
    expected_logits = extend_logits_reference(
        q_fp8=q_fp8[:q_rows],
        weights=weights[:q_rows],
        kv_fp8=(k_gather, s_gather),
        k_start=k_start,
        k_end=k_end,
    )
    logit_metrics = _assert_indexer_logits_close(
        actual=logits,
        expected=expected_logits,
        require_invalid_neginf=False,
    )

    q_lens_t = torch.tensor(q_lens, dtype=torch.int32, device=device)
    nsa_cache = torch.clamp(seqlens_expanded, max=DEFAULT_TOPK).to(torch.int32)
    attn_metadata = sglang.NSAMetadata(
        page_size=PAGE_SIZE,
        cache_seqlens_int32=final_lens_t,
        max_seq_len_q=max(q_lens),
        max_seq_len_k=max(final_lens),
        cu_seqlens_q=_cu_from_lengths(q_lens_t),
        cu_seqlens_k=_cu_from_lengths(final_lens_t),
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        nsa_cache_seqlens_int32=nsa_cache,
        nsa_cu_seqlens_q=torch.arange(q_rows + 1, dtype=torch.int32, device=device),
        nsa_cu_seqlens_k=_cu_from_lengths(nsa_cache),
        nsa_extend_seq_lens_list=list(q_lens),
        nsa_seqlens_expanded=seqlens_expanded,
        seq_lens_sum=total_k,
        page_table_1_flattened=row_ids,
        topk_indices_offset=k_start,
        indexer_k_start_end=(k_start, k_end),
        indexer_seq_lens_cpu=final_lens_t.detach().cpu(),
        indexer_seq_lens=final_lens_t,
        seq_lens_cpu_list=list(final_lens),
        indexer_seq_lens_sum=total_k,
        indexer_max_seq_len=max(final_lens),
        token_to_batch_idx=token_to_batch_idx,
    )
    indexer_metadata = sglang.NSAIndexerMetadata(
        attn_metadata=attn_metadata,
        topk_transform_method=sglang.TopkTransformMethod.RAGGED,
    )
    compact_core = indexer_metadata.topk_transform(logits, DEFAULT_TOPK, ks=k_start)
    expected_core = _suffix_ragged_topk(k_start=k_start, k_end=k_end, topk=DEFAULT_TOPK)
    if case.q_padding:
        pad = torch.full((case.q_padding, DEFAULT_TOPK), -1, dtype=torch.int32, device=device)
        compact_topk = torch.cat([compact_core, pad], dim=0)
        expected_compact = torch.cat([expected_core, pad.clone()], dim=0)
    else:
        compact_topk = compact_core
        expected_compact = expected_core
    try:
        topk_metrics = _assert_topk_set_equal(compact_topk, expected_compact)
    except BatteryFailure as exc:
        torch_topk_compact = _select_ragged_topk_from_logits(
            logits=expected_logits,
            k_start=k_start,
            k_end=k_end,
            topk=DEFAULT_TOPK,
        )
        verdict = (
            "torch.topk(logits) matches the structured reference"
            if _topk_sets_equal(torch_topk_compact, expected_core)
            else "torch.topk(logits) also differs from the structured reference"
        )
        raise BatteryFailure(f"{exc}; {verdict}") from exc
    selected = _map_compact_topk_to_physical(compact_topk=compact_topk, row_ids=row_ids)
    expected_selected = _map_compact_topk_to_physical(
        compact_topk=expected_compact,
        row_ids=row_ids,
    )
    _assert_topk_set_equal(selected, expected_selected)

    metrics: dict[str, float | int] = {
        "gather_rows": int(k_gather.shape[0]),
        "logit_max_abs": float(logit_metrics["max_abs"]),
        "q_rows": q_rows_total,
        **topk_metrics,
    }
    if case.q_padding:
        padded_bad = int(torch.count_nonzero(compact_topk[q_rows:] != -1).item())
        if padded_bad:
            raise BatteryFailure(f"SGLang ragged q-padding rows contain {padded_bad} non--1 entries")
        metrics["padded_rows"] = int(case.q_padding)

    if case.run_mla:
        if case.q_padding:
            raise BatteryFailure("run_mla SGLang ragged cases should not include q padding")
        _k_nope, _k_rope, packed = _make_mla_pool(
            pool_tokens=int(pool_pages) * PAGE_SIZE,
            seed=case.seed + 30,
            device=device,
        )
        q_all = _make_q_all(
            q_rows=q_rows,
            mla_heads=case.mla_heads,
            seed=case.seed + 31,
            device=device,
        )
        active_counts = torch.count_nonzero(selected >= 0, dim=1).to(torch.int32)
        mla_workspace = _make_workspace(
            mode="extend",
            device=device,
            topk=DEFAULT_TOPK,
            max_total_q=q_rows,
            max_batch=len(case.request_shapes),
            max_kv_rows=total_k,
            mla_heads=case.mla_heads,
            index_heads=case.index_heads,
        )
        mla_metadata = MLASparseExtendMetadata(
            selected_token_offsets=selected,
            cache_seqlens_int32=final_lens_t,
            nsa_cache_seqlens_int32=active_counts,
            nsa_cu_seqlens_q=_cu_from_lengths(q_lens_t),
            nsa_cu_seqlens_k=_cu_from_lengths(final_lens_t),
            max_seq_len_q=max(q_lens),
            max_seq_len_k=max(final_lens),
            mode="extend",
        )
        actual_mla = sparse_mla_extend_forward(
            q_all=q_all,
            kv_cache=packed,
            selected_token_offsets=mla_metadata.selected_token_offsets,
            cache_seqlens_int32=mla_metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=mla_metadata.nsa_cache_seqlens_int32,
            workspace=mla_workspace,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
        expected_mla = sparse_mla_reference(
            q_all=q_all,
            kv_cache=packed,
            page_table_1=expected_selected,
            active_token_counts=active_counts,
            sm_scale=MLA_SM_SCALE,
            v_head_dim=MLA_V_HEAD_DIM,
        )
        mla_metrics = _assert_mla_close(actual=actual_mla, expected=expected_mla)
        metrics.update(
            {
                "mla_max_abs": float(mla_metrics["max_abs"]),
                "mla_rmse": float(mla_metrics["rmse"]),
                "mla_cos": float(mla_metrics["cos"]),
            }
        )
    torch.cuda.synchronize(device)
    return metrics


def _indexer_paged_cases() -> list[IndexerPagedCase]:
    return [
        IndexerPagedCase(
            name="paged-boundaries-bs4",
            tier="smoke",
            q_rows=4,
            row_seqlens_a=(1, 63, 64, 129),
            row_seqlens_b=(65, 17, 128, 2),
            index_heads=8,
            graph_width_tokens=256,
            topk=128,
            seed=10_001,
        ),
        IndexerPagedCase(
            name="paged-mixed-long-bs8",
            tier="full",
            q_rows=8,
            row_seqlens_a=(129, 2048, 4097, 8192, 1024, 65, 32768, 16384),
            row_seqlens_b=(65, 1024, 2048, 4096, 8192, 129, 16384, 32768),
            index_heads=32,
            graph_width_tokens=32768,
            topk=DEFAULT_TOPK,
            seed=10_002,
        ),
        IndexerPagedCase(
            name="paged-stress-64k-bs4",
            tier="stress",
            q_rows=4,
            row_seqlens_a=(65536, 49152, 32768, 8192),
            row_seqlens_b=(8192, 32768, 65536, 129),
            index_heads=32,
            graph_width_tokens=65536,
            topk=DEFAULT_TOPK,
            seed=10_003,
        ),
    ]


def _indexer_extend_cases() -> list[IndexerExtendCase]:
    return [
        IndexerExtendCase(
            name="extend-prefill-boundaries",
            tier="smoke",
            request_shapes=((64, 2), (127, 3)),
            index_heads=8,
            topk=128,
            seed=20_001,
            preinitialize_invalid_logits=True,
        ),
        IndexerExtendCase(
            name="extend-ragged-sglang-preinit-false",
            tier="full",
            request_shapes=((2048, 8), (8192, 16), (4096, 7)),
            index_heads=32,
            topk=DEFAULT_TOPK,
            seed=20_002,
            preinitialize_invalid_logits=False,
        ),
        IndexerExtendCase(
            name="extend-stress-64k",
            tier="stress",
            request_shapes=((65536, 32), (32768, 32)),
            index_heads=32,
            topk=DEFAULT_TOPK,
            seed=20_003,
            preinitialize_invalid_logits=False,
        ),
    ]


def _mla_cases() -> list[MLACase]:
    return [
        MLACase(
            name="mla-decode-boundaries",
            tier="smoke",
            q_rows=4,
            batch=4,
            cache_len=257,
            topk=128,
            valid_pattern=(0, 1, 63, 128),
            mla_heads=8,
            seed=30_001,
            mode="decode",
        ),
        MLACase(
            name="mla-decode-topk2048",
            tier="full",
            q_rows=8,
            batch=8,
            cache_len=8192,
            topk=DEFAULT_TOPK,
            valid_pattern=(1, 64, 129, 2048),
            mla_heads=16,
            seed=30_002,
            mode="decode",
        ),
        MLACase(
            name="mla-extend-topk2048",
            tier="full",
            q_rows=32,
            batch=4,
            cache_len=8192,
            topk=DEFAULT_TOPK,
            valid_pattern=(64, 512, 2048, 129),
            mla_heads=16,
            seed=30_003,
            mode="extend",
        ),
        MLACase(
            name="mla-target-verify-draft-shaped",
            tier="full",
            q_rows=16,
            batch=4,
            cache_len=4096,
            topk=DEFAULT_TOPK,
            valid_pattern=(2048, 1024, 65, 1),
            mla_heads=8,
            seed=30_004,
            mode="target_verify",
        ),
        MLACase(
            name="mla-stress-decode-64k",
            tier="stress",
            q_rows=4,
            batch=4,
            cache_len=65536,
            topk=DEFAULT_TOPK,
            valid_pattern=(2048, 2048, 129, 64),
            mla_heads=16,
            seed=30_005,
            mode="decode",
        ),
    ]


def _e2e_decode_cases() -> list[E2EDecodeCase]:
    return [
        E2EDecodeCase(
            name="e2e-decode-boundaries",
            tier="smoke",
            row_seqlens_a=(129, 257),
            row_seqlens_b=(65, 128),
            index_heads=8,
            mla_heads=8,
            topk=128,
            graph_width_tokens=512,
            seed=40_001,
        ),
        E2EDecodeCase(
            name="e2e-decode-topk2048-mixed",
            tier="full",
            row_seqlens_a=(2050, 8192, 4097, 129),
            row_seqlens_b=(65, 4096, 8192, 2048),
            index_heads=32,
            mla_heads=16,
            topk=DEFAULT_TOPK,
            graph_width_tokens=8192,
            seed=40_002,
        ),
        E2EDecodeCase(
            name="e2e-decode-stress-64k",
            tier="stress",
            row_seqlens_a=(65536, 32768),
            row_seqlens_b=(8192, 65536),
            index_heads=32,
            mla_heads=16,
            topk=DEFAULT_TOPK,
            graph_width_tokens=65536,
            seed=40_003,
        ),
    ]


def _e2e_extend_cases() -> list[E2EExtendCase]:
    return [
        E2EExtendCase(
            name="e2e-extend-ragged-boundaries",
            tier="smoke",
            request_shapes=((128, 2), (257, 3)),
            index_heads=8,
            mla_heads=8,
            topk=128,
            seed=50_001,
        ),
        E2EExtendCase(
            name="e2e-extend-topk2048",
            tier="full",
            request_shapes=((4096, 8), (8192, 8), (2048, 4)),
            index_heads=32,
            mla_heads=16,
            topk=DEFAULT_TOPK,
            seed=50_002,
        ),
        E2EExtendCase(
            name="e2e-extend-stress-64k",
            tier="stress",
            request_shapes=((65536, 16), (32768, 16)),
            index_heads=32,
            mla_heads=16,
            topk=DEFAULT_TOPK,
            seed=50_003,
        ),
    ]


def _sglang_paged_cases() -> list[SGLangPagedCase]:
    return [
        SGLangPagedCase(
            name="sglang-paged-decode-fused-padding",
            tier="smoke",
            row_seqlens_a=(1, 63, 2049, 4096),
            row_seqlens_b=None,
            extend_lens=(1, 1, 1, 1),
            index_heads=32,
            mla_heads=16,
            graph_width_tokens=4096,
            q_padding=2,
            seed=60_001,
            mode_name="decode",
            run_mla=False,
        ),
        SGLangPagedCase(
            name="sglang-paged-decode-e2e",
            tier="full",
            row_seqlens_a=(2050, 8192, 4097, 129),
            row_seqlens_b=(65, 4096, 8192, 2048),
            extend_lens=(1, 1, 1, 1),
            index_heads=32,
            mla_heads=16,
            graph_width_tokens=8192,
            q_padding=0,
            seed=60_002,
            mode_name="decode",
            run_mla=True,
        ),
        SGLangPagedCase(
            name="sglang-paged-target-verify-expanded",
            tier="full",
            row_seqlens_a=(2052, 8192, 4097),
            row_seqlens_b=None,
            extend_lens=(4, 3, 5),
            index_heads=32,
            mla_heads=16,
            graph_width_tokens=8192,
            q_padding=1,
            seed=60_003,
            mode_name="target_verify",
            run_mla=False,
        ),
        SGLangPagedCase(
            name="sglang-paged-draft-extend-expanded",
            tier="full",
            row_seqlens_a=(4096, 1024, 8192, 257),
            row_seqlens_b=None,
            extend_lens=(2, 5, 3, 1),
            index_heads=32,
            mla_heads=16,
            graph_width_tokens=8192,
            q_padding=3,
            seed=60_004,
            mode_name="draft_extend",
            run_mla=False,
        ),
        SGLangPagedCase(
            name="sglang-paged-stress-64k",
            tier="stress",
            row_seqlens_a=(65536, 32768, 8192, 49152),
            row_seqlens_b=(8192, 65536, 32768, 129),
            extend_lens=(1, 1, 1, 1),
            index_heads=32,
            mla_heads=16,
            graph_width_tokens=65536,
            q_padding=0,
            seed=60_005,
            mode_name="decode",
            run_mla=True,
        ),
    ]


def _sglang_ragged_cases() -> list[SGLangRaggedCase]:
    return [
        SGLangRaggedCase(
            name="sglang-ragged-gather-padding",
            tier="smoke",
            request_shapes=((128, 2), (2048, 3), (63, 1)),
            index_heads=32,
            mla_heads=16,
            q_padding=2,
            seed=70_001,
            run_mla=False,
        ),
        SGLangRaggedCase(
            name="sglang-ragged-e2e-mla",
            tier="full",
            request_shapes=((2048, 4), (8192, 5), (4096, 3)),
            index_heads=32,
            mla_heads=16,
            q_padding=0,
            seed=70_002,
            run_mla=True,
        ),
        SGLangRaggedCase(
            name="sglang-ragged-stress-64k",
            tier="stress",
            request_shapes=((65536, 8), (32768, 8)),
            index_heads=32,
            mla_heads=16,
            q_padding=0,
            seed=70_003,
            run_mla=True,
        ),
    ]


class BatteryRunner:
    def __init__(
        self,
        *,
        tier: Tier,
        mode: Mode,
        execution: Literal["eager", "graph", "both"],
        fail_fast: bool,
        device: torch.device,
        sglang_root: Path,
    ) -> None:
        self.tier = tier
        self.mode = mode
        self.execution = execution
        self.fail_fast = fail_fast
        self.device = device
        self.sglang_root = sglang_root
        self.results: list[CaseResult] = []
        self.failures: list[dict[str, str]] = []

    def _enabled(self, *, tier: Tier, mode: Mode, execution: Execution) -> bool:
        if TIER_ORDER[tier] > TIER_ORDER[self.tier]:
            return False
        if self.mode != "all" and self.mode != mode:
            return False
        if self.execution != "both" and self.execution != execution:
            return False
        return True

    def run_case(
        self,
        *,
        name: str,
        tier: Tier,
        mode: Mode,
        execution: Execution,
        fn: Callable[[], dict[str, float | int | str]],
    ) -> None:
        if not self._enabled(tier=tier, mode=mode, execution=execution):
            return
        torch.cuda.empty_cache()
        start = time.perf_counter()
        print(f"RUN  {mode:7s} {execution:5s} {tier:6s} {name}", flush=True)
        try:
            metrics = fn()
            torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.results.append(
                CaseResult(
                    name=name,
                    tier=tier,
                    mode=mode,
                    execution=execution,
                    elapsed_ms=elapsed_ms,
                    metrics=metrics,
                )
            )
            metric_text = " ".join(
                f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                for key, value in sorted(metrics.items())
            )
            print(f"PASS {elapsed_ms:9.2f} ms {name} {metric_text}", flush=True)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            tb = traceback.format_exc()
            self.failures.append({"name": name, "error": str(exc), "traceback": tb})
            print(f"FAIL {elapsed_ms:9.2f} ms {name}: {exc}", flush=True)
            print(tb, flush=True)
            if self.fail_fast:
                raise

    def run_all(self) -> None:
        for case in _indexer_paged_cases():
            self.run_case(
                name=case.name,
                tier=case.tier,
                mode="indexer",
                execution="eager",
                fn=lambda case=case: _run_indexer_paged_eager(case, self.device),
            )
            self.run_case(
                name=f"{case.name}-graph",
                tier=case.tier,
                mode="indexer",
                execution="graph",
                fn=lambda case=case: _run_indexer_paged_graph(case, self.device),
            )

        for case in _indexer_extend_cases():
            self.run_case(
                name=case.name,
                tier=case.tier,
                mode="indexer",
                execution="eager",
                fn=lambda case=case: _run_indexer_extend_eager(case, self.device),
            )

        for case in _mla_cases():
            self.run_case(
                name=case.name,
                tier=case.tier,
                mode="mla",
                execution="eager",
                fn=lambda case=case: _run_mla_eager(case, self.device),
            )
            if case.mode == "decode":
                self.run_case(
                    name=f"{case.name}-graph",
                    tier=case.tier,
                    mode="mla",
                    execution="graph",
                    fn=lambda case=case: _run_mla_graph(case, self.device),
                )

        for case in _e2e_decode_cases():
            self.run_case(
                name=case.name,
                tier=case.tier,
                mode="e2e",
                execution="eager",
                fn=lambda case=case: _run_e2e_decode_eager(case, self.device),
            )
            self.run_case(
                name=f"{case.name}-graph",
                tier=case.tier,
                mode="e2e",
                execution="graph",
                fn=lambda case=case: _run_e2e_decode_graph(case, self.device),
            )

        for case in _e2e_extend_cases():
            self.run_case(
                name=case.name,
                tier=case.tier,
                mode="e2e",
                execution="eager",
                fn=lambda case=case: _run_e2e_extend_eager(case, self.device),
            )

        for case in _sglang_paged_cases():
            self.run_case(
                name=case.name,
                tier=case.tier,
                mode="sglang",
                execution="eager",
                fn=lambda case=case: _run_sglang_paged_eager(
                    case,
                    self.device,
                    self.sglang_root,
                ),
            )
            if case.row_seqlens_b is not None:
                self.run_case(
                    name=f"{case.name}-graph",
                    tier=case.tier,
                    mode="sglang",
                    execution="graph",
                    fn=lambda case=case: _run_sglang_paged_graph(
                        case,
                        self.device,
                        self.sglang_root,
                    ),
                )

        for case in _sglang_ragged_cases():
            self.run_case(
                name=case.name,
                tier=case.tier,
                mode="sglang",
                execution="eager",
                fn=lambda case=case: _run_sglang_ragged_eager(
                    case,
                    self.device,
                    self.sglang_root,
                ),
            )

    def report(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "mode": self.mode,
            "execution": self.execution,
            "device": _device_name(self.device),
            "passed": len(self.results),
            "failed": len(self.failures),
            "results": [
                {
                    "name": result.name,
                    "tier": result.tier,
                    "mode": result.mode,
                    "execution": result.execution,
                    "elapsed_ms": result.elapsed_ms,
                    "metrics": result.metrics,
                }
                for result in self.results
            ],
            "failures": self.failures,
        }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone NSA Indexer and Sparse MLA reference battery."
    )
    parser.add_argument("--tier", choices=("smoke", "full", "stress"), default="full")
    parser.add_argument("--mode", choices=("all", "indexer", "mla", "e2e", "sglang"), default="all")
    parser.add_argument("--execution", choices=("eager", "graph", "both"), default="both")
    parser.add_argument("--sglang-root", type=Path, default=Path("~/projects/sglang"))
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    device = _require_sm120()
    if device is None:
        return 0
    print(
        "NSA/MLA numerical reference battery "
        f"tier={args.tier} mode={args.mode} execution={args.execution} "
        f"device={_device_name(device)}",
        flush=True,
    )
    runner = BatteryRunner(
        tier=args.tier,
        mode=args.mode,
        execution=args.execution,
        fail_fast=bool(args.fail_fast),
        device=device,
        sglang_root=args.sglang_root,
    )
    try:
        runner.run_all()
    finally:
        report = runner.report()
        if args.report_json is not None:
            args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(
        f"SUMMARY passed={len(runner.results)} failed={len(runner.failures)}",
        flush=True,
    )
    return 1 if runner.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
