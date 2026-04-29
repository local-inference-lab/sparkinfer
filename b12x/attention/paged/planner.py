"""Host planner for the primary paged-attention backend.

This module models the host-side work decomposition used by FlashInfer's paged
attention kernels:

- choose `CTA_TILE_Q` from packed Q rows,
- choose `kv_chunk_size` on the host,
- emit exact `(request_idx, qo_tile_idx, kv_tile_idx)` worklists,
- emit `merge_indptr` / `o_indptr` for split reduction.

No kernel-side split LUT or legacy scheduler assumptions live here.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import torch

_FP8_KV_DTYPE = torch.float8_e4m3fn
_PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 6),
    (512, 24),
    (1024, 24),
    (2048, 24),
)
_PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 6),
    (512, 32),
    (1024, 32),
    (2048, 32),
)

_DEFAULT_GRAPH_CTAS_PER_SM = 2
_PagedMode = Literal["decode", "extend", "verify"]


def _merge_backend_supports_split_kv(
    *,
    output_dtype: torch.dtype,
    head_dim_vo: int,
) -> bool:
    return output_dtype in (torch.float16, torch.bfloat16, torch.float32) and head_dim_vo == 256


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def _graph_max_batch_size_if_split(
    *,
    device: torch.device,
    num_kv_heads: int,
    graph_ctas_per_sm: int,
) -> int:
    blocks_per_sm = int(graph_ctas_per_sm)
    if blocks_per_sm <= 0:
        raise ValueError("graph_ctas_per_sm must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return max((num_sms * blocks_per_sm) // num_kv_heads, 1)


def _kv_dtype_tuning_key(kv_dtype: torch.dtype) -> str:
    if kv_dtype == torch.bfloat16:
        return "bf16"
    if kv_dtype == torch.float16:
        return "fp16"
    if kv_dtype == _FP8_KV_DTYPE:
        return "fp8_e4m3fn"
    raise TypeError(f"unsupported kv dtype for CTA tuning lookup: {kv_dtype}")


def _resolve_graph_ctas_per_sm(
    *,
    mode: _PagedMode,
    kv_dtype: torch.dtype,
    policy_batch: int,
    max_effective_kv_pages: int,
    graph_ctas_per_sm: int | None,
) -> int:
    if graph_ctas_per_sm is not None:
        resolved_graph_ctas_per_sm = int(graph_ctas_per_sm)
        if resolved_graph_ctas_per_sm <= 0:
            raise ValueError("graph_ctas_per_sm must be positive")
        return resolved_graph_ctas_per_sm

    tuning_regime = "decode" if mode == "verify" else mode
    if tuning_regime != "decode":
        return _DEFAULT_GRAPH_CTAS_PER_SM

    try:
        from .tuning import get_decode_graph_policy
    except ImportError:
        return _DEFAULT_GRAPH_CTAS_PER_SM

    try:
        resolved_graph_ctas_per_sm = int(
            get_decode_graph_policy(
                kv_dtype=_kv_dtype_tuning_key(kv_dtype),
                regime=tuning_regime,
                batch=policy_batch,
            ).graph_ctas_per_sm
        )
    except KeyError:
        return _DEFAULT_GRAPH_CTAS_PER_SM

    if resolved_graph_ctas_per_sm <= 0:
        raise ValueError("resolved graph_ctas_per_sm must be positive")
    return resolved_graph_ctas_per_sm


def _metadata_to_cpu_int_list(t: torch.Tensor, *, name: str) -> list[int]:
    if t.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be torch.int32 or torch.int64")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return [int(v) for v in t.detach().cpu().tolist()]


def _estimate_new_batch_size(
    *,
    packed_qo_len_arr: list[int],
    kv_len_arr: list[int],
    qo_chunk_size: int,
    kv_chunk_size_pages: int,
) -> int:
    new_batch_size = 0
    for packed_qo_len, kv_len in zip(packed_qo_len_arr, kv_len_arr):
        num_tiles_q = _ceil_div(packed_qo_len, qo_chunk_size)
        num_chunks_kv = _ceil_div(max(kv_len, 1), kv_chunk_size_pages)
        new_batch_size += num_tiles_q * num_chunks_kv
    return int(new_batch_size)


def _lookup_chunk_pages_from_table(
    max_effective_kv_pages: int,
    table: tuple[tuple[int, int | None], ...],
) -> int | None:
    for max_pages, chunk_pages in table:
        if max_effective_kv_pages <= max_pages:
            return chunk_pages
    return None


def _q_lengths_from_cu_seqlens(cu_seqlens_q: torch.Tensor) -> list[int]:
    cu_seqlens_q_list = _metadata_to_cpu_int_list(cu_seqlens_q, name="cu_seqlens_q")
    q_lengths: list[int] = []
    for start, end in zip(cu_seqlens_q_list[:-1], cu_seqlens_q_list[1:]):
        if end < start:
            raise ValueError("cu_seqlens_q must be non-decreasing")
        q_lengths.append(end - start)
    return q_lengths


def _graph_policy_batch(
    *,
    mode: _PagedMode,
    batch: int,
    total_q: int,
) -> int:
    if mode == "verify":
        return max(int(total_q), 1)
    return max(int(batch), 1)


def infer_paged_mode(cu_seqlens_q: torch.Tensor) -> Literal["decode", "extend"]:
    q_lengths = _q_lengths_from_cu_seqlens(cu_seqlens_q)
    return "decode" if q_lengths and all(q_len == 1 for q_len in q_lengths) else "extend"


def _fa2_determine_cta_tile_q(avg_packed_qo_len: int, head_dim: int) -> int:
    # Faithful to FlashInfer's FA2DetermineCtaTileQ.
    if avg_packed_qo_len > 64 and head_dim < 256:
        return 128
    if avg_packed_qo_len > 16:
        return 64
    return 16


def _paged_determine_cta_tile_q(
    *,
    mode: _PagedMode,
    kv_dtype: torch.dtype,
    packed_qo_len: int,
    head_dim: int,
    max_effective_kv_pages: int,
) -> int:
    if mode == "verify":
        del kv_dtype, head_dim, max_effective_kv_pages
        return 16
    if mode == "extend" and packed_qo_len <= 32:
        del kv_dtype, head_dim, max_effective_kv_pages
        return 16

    cta_tile_q = _fa2_determine_cta_tile_q(packed_qo_len, head_dim)
    del max_effective_kv_pages
    del mode, kv_dtype
    return cta_tile_q


def chunk_pages_for_family(
    *,
    mode: _PagedMode,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    policy_batch: int | None,
    graph_chunk_policy: bool,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
) -> int | None:
    if page_size != 64 or head_dim_qk != 256 or head_dim_vo != 256 or gqa_group_size != 8:
        return None
    if q_dtype != torch.bfloat16:
        return None
    if mode == "extend" and not graph_chunk_policy and kv_dtype == _FP8_KV_DTYPE:
        return _lookup_chunk_pages_from_table(max_effective_kv_pages, _PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES)
    if mode == "extend" and not graph_chunk_policy and kv_dtype == torch.bfloat16:
        return _lookup_chunk_pages_from_table(
            max_effective_kv_pages,
            _PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES,
        )

    tuning_regime = "decode" if mode == "verify" else mode
    if (
        tuning_regime != "decode"
        or not graph_chunk_policy
        or policy_batch is None
        or q_dtype != torch.bfloat16
        or page_size != 64
        or head_dim_qk != 256
        or head_dim_vo != 256
        or gqa_group_size != 8
    ):
        return None

    try:
        from .tuning import lookup_decode_graph_chunk_pages
    except ImportError:
        return None

    try:
        return int(
            lookup_decode_graph_chunk_pages(
                kv_dtype=_kv_dtype_tuning_key(kv_dtype),
                regime=tuning_regime,
                batch=int(policy_batch),
                page_count=max(max_effective_kv_pages, 1),
            )
        )
    except KeyError:
        return None


@dataclass(frozen=True)
class PagedPlanBudget:
    max_total_q: int | None = None
    max_batch: int | None = None
    max_page_table_width: int | None = None
    max_work_items: int | None = None
    max_partial_rows: int | None = None


def _estimate_prefill_plan_usage(
    *,
    packed_qo_len_arr: list[int],
    q_lengths: list[int],
    kv_len_arr: list[int],
    qo_chunk_size: int,
    kv_chunk_size_pages: int,
    force_split_kv: bool,
) -> tuple[int, int]:
    new_batch_size = 0
    total_num_partial_rows = 0
    split_kv = False
    for packed_qo_len, qo_len, kv_len in zip(packed_qo_len_arr, q_lengths, kv_len_arr):
        num_tiles_q = _ceil_div(packed_qo_len, qo_chunk_size)
        num_chunks_kv = _ceil_div(max(kv_len, 1), kv_chunk_size_pages)
        split_kv = split_kv or num_chunks_kv > 1
        new_batch_size += num_tiles_q * num_chunks_kv
        total_num_partial_rows += qo_len * num_chunks_kv
    if force_split_kv:
        split_kv = True
    return int(new_batch_size), int(total_num_partial_rows if split_kv else 0)


def _prefill_usage_fits_budget(
    *,
    budget: PagedPlanBudget | None,
    new_batch_size: int,
    total_num_partial_rows: int,
) -> bool:
    if budget is None:
        return True
    if budget.max_work_items is not None and new_batch_size > int(budget.max_work_items):
        return False
    if budget.max_partial_rows is not None and total_num_partial_rows > int(budget.max_partial_rows):
        return False
    return True


def _prefill_binary_search_kv_chunk_size(
    *,
    enable_cuda_graph: bool,
    max_batch_size_if_split: int,
    packed_qo_len_arr: list[int],
    q_lengths: list[int],
    kv_len_arr: list[int],
    qo_chunk_size: int,
    force_split_kv: bool,
    plan_budget: PagedPlanBudget | None = None,
    min_kv_chunk_size: int = 1,
) -> tuple[bool, int]:
    batch_size = len(packed_qo_len_arr)
    max_kv_len = max(max(kv_len_arr, default=1), 1)
    low = min_kv_chunk_size
    high = max_kv_len
    while low < high:
        mid = (low + high) // 2
        new_batch_size, total_num_partial_rows = _estimate_prefill_plan_usage(
            packed_qo_len_arr=packed_qo_len_arr,
            q_lengths=q_lengths,
            kv_len_arr=kv_len_arr,
            qo_chunk_size=qo_chunk_size,
            kv_chunk_size_pages=mid,
            force_split_kv=force_split_kv,
        )
        if (
            new_batch_size > max_batch_size_if_split
            or not _prefill_usage_fits_budget(
                budget=plan_budget,
                new_batch_size=new_batch_size,
                total_num_partial_rows=total_num_partial_rows,
            )
        ):
            low = mid + 1
        else:
            high = mid
    final_new_batch_size, final_total_num_partial_rows = _estimate_prefill_plan_usage(
        packed_qo_len_arr=packed_qo_len_arr,
        q_lengths=q_lengths,
        kv_len_arr=kv_len_arr,
        qo_chunk_size=qo_chunk_size,
        kv_chunk_size_pages=low,
        force_split_kv=force_split_kv,
    )
    if final_new_batch_size > max_batch_size_if_split or not _prefill_usage_fits_budget(
        budget=plan_budget,
        new_batch_size=final_new_batch_size,
        total_num_partial_rows=final_total_num_partial_rows,
    ):
        raise ValueError("paged prefill plan exceeds the configured eager workspace budget")
    return (enable_cuda_graph or low < max_kv_len, low)


def decode_chunk_pages_for_graph(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    batch: int | None = None,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
) -> int | None:
    return chunk_pages_for_family(
        mode="decode",
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        policy_batch=batch,
        graph_chunk_policy=True,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
        max_effective_kv_pages=max_effective_kv_pages,
    )


def build_decode_chunk_pages_lut(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    batch: int | None = None,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
) -> tuple[int, ...]:
    if batch is None:
        raise ValueError("batch is required for decode graph chunk policy lookup")
    max_effective_kv_pages = max(int(max_effective_kv_pages), 1)
    lut: list[int] = []
    for page_count in range(1, max_effective_kv_pages + 1):
        chunk_pages = decode_chunk_pages_for_graph(
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            batch=batch,
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=page_count,
        )
        if chunk_pages is None:
            raise KeyError(
                f"no registered decode graph policy for kv_dtype={_kv_dtype_tuning_key(kv_dtype)!r}, batch={batch}"
            )
        lut.append(int(chunk_pages))
    return tuple(lut)


@dataclass(frozen=True)
class PagedPlanKey:
    total_q: int
    num_q_heads: int
    head_dim_qk: int
    head_dim_vo: int
    k_cache_shape: tuple[int, ...]
    v_cache_shape: tuple[int, ...]
    page_table_shape: tuple[int, ...]
    dtype: torch.dtype
    kv_dtype: torch.dtype
    mode: _PagedMode
    cta_tile_q: int
    kv_chunk_size: int
    split_kv: bool
    fixed_split_size: int
    disable_split_kv: bool
    enable_cuda_graph: bool
    graph_chunk_policy: bool
    graph_ctas_per_sm: int
    window_left: int
    max_batch_size_if_split: int
    padded_batch_size: int
    new_batch_size: int
    num_qo_tiles: int
    total_num_partial_rows: int
    page_size: int
    num_kv_heads: int
    gqa_group_size: int
    device_index: int


@dataclass(frozen=True, kw_only=True)
class PagedPlan:
    key: PagedPlanKey
    request_indices: tuple[int, ...]
    qo_tile_indices: tuple[int, ...]
    kv_tile_indices: tuple[int, ...]
    merge_indptr: tuple[int, ...]
    o_indptr: tuple[int, ...]
    block_valid_mask: tuple[bool, ...]
    kv_window_start_tokens: tuple[int, ...]

    def __getattr__(self, name: str):
        return getattr(self.key, name)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.device_index)


def create_paged_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    mode: _PagedMode | None = None,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
    force_split_kv: bool | None = None,
    enable_cuda_graph: bool = False,
    graph_chunk_policy: bool = False,
    max_batch_size_if_split: int | None = None,
    graph_ctas_per_sm: int | None = None,
    plan_budget: PagedPlanBudget | None = None,
    window_left: int = -1,
) -> PagedPlan:
    if q.ndim != 3:
        raise ValueError(f"q must be rank-3 [total_q, q_heads, head_dim], got {tuple(q.shape)}")
    if k_cache.ndim != 4:
        raise ValueError(
            f"k_cache must be rank-4 [num_pages, page_size, kv_heads, head_dim], got {tuple(k_cache.shape)}"
        )
    if v_cache.ndim != 4:
        raise ValueError(
            f"v_cache must be rank-4 [num_pages, page_size, kv_heads, head_dim_v], got {tuple(v_cache.shape)}"
        )
    if page_table.ndim != 2:
        raise ValueError(f"page_table must be rank-2 [batch, max_pages], got {tuple(page_table.shape)}")
    if cache_seqlens.ndim != 1:
        raise ValueError(f"cache_seqlens must be rank-1 [batch], got {tuple(cache_seqlens.shape)}")
    if cu_seqlens_q.ndim != 1:
        raise ValueError(f"cu_seqlens_q must be rank-1 [batch+1], got {tuple(cu_seqlens_q.shape)}")
    if q.device.type != "cuda":
        raise ValueError("q must be on CUDA")
    if not (k_cache.device == v_cache.device == page_table.device == cache_seqlens.device == cu_seqlens_q.device == q.device):
        raise ValueError("all inputs must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported q dtype {q.dtype}")
    if k_cache.dtype != v_cache.dtype:
        raise TypeError("k_cache and v_cache must have matching dtypes")
    if k_cache.dtype not in (torch.float16, torch.bfloat16, _FP8_KV_DTYPE):
        raise TypeError(f"unsupported kv dtype {k_cache.dtype}")
    if window_left < -1:
        raise ValueError("window_left must be -1 for full attention or a non-negative token count")

    total_q, num_q_heads, head_dim_qk = [int(dim) for dim in q.shape]
    num_pages, page_size, num_kv_heads, head_dim_k = [int(dim) for dim in k_cache.shape]
    v_num_pages, v_page_size, v_num_kv_heads, head_dim_vo = [int(dim) for dim in v_cache.shape]
    batch, max_pages_per_request = [int(dim) for dim in page_table.shape]

    if num_pages != v_num_pages or page_size != v_page_size or num_kv_heads != v_num_kv_heads:
        raise ValueError("k_cache and v_cache structural shapes must match except head_dim")
    if head_dim_k != head_dim_qk:
        raise ValueError("primary paged backend expects head_dim_qk to match k_cache head_dim")
    if page_size != 64:
        raise ValueError(f"primary paged backend expects page_size=64, got {page_size}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if tuple(cache_seqlens.shape) != (batch,):
        raise ValueError("cache_seqlens shape must match page_table batch")
    if tuple(cu_seqlens_q.shape) != (batch + 1,):
        raise ValueError("cu_seqlens_q shape must be [batch + 1]")

    q_lengths = _q_lengths_from_cu_seqlens(cu_seqlens_q)
    cache_lengths = _metadata_to_cpu_int_list(cache_seqlens, name="cache_seqlens")
    if any(cache_len <= 0 for cache_len in cache_lengths):
        raise ValueError("primary paged backend requires cache_seqlens > 0")
    cache_pages_arr = [_ceil_div(cache_len, page_size) for cache_len in cache_lengths]
    if any(cache_pages > max_pages_per_request for cache_pages in cache_pages_arr):
        raise ValueError("page_table width is smaller than required by cache_seqlens")

    inferred_mode = infer_paged_mode(cu_seqlens_q)
    mode = inferred_mode if mode is None else mode
    if mode == "verify" and inferred_mode != "extend":
        raise ValueError(f"verify mode requires q_len > 1, got inferred mode {inferred_mode}")
    if force_split_kv is None:
        force_split_kv = mode in ("decode", "verify")
    if mode == "extend" and force_split_kv:
        raise ValueError("extend plans no longer support split-kv")
    if plan_budget is not None:
        if plan_budget.max_total_q is not None and total_q > int(plan_budget.max_total_q):
            raise ValueError(
                f"paged plan total_q={total_q} exceeds workspace capacity {int(plan_budget.max_total_q)}"
            )
        if plan_budget.max_batch is not None and batch > int(plan_budget.max_batch):
            raise ValueError(
                f"paged plan batch={batch} exceeds workspace capacity {int(plan_budget.max_batch)}"
            )
        if (
            plan_budget.max_page_table_width is not None
            and max_pages_per_request > int(plan_budget.max_page_table_width)
        ):
            raise ValueError(
                "paged plan page_table width="
                f"{max_pages_per_request} exceeds workspace capacity {int(plan_budget.max_page_table_width)}"
            )

    gqa_group_size = num_q_heads // num_kv_heads
    packed_qo_len_arr = [q_len * gqa_group_size for q_len in q_lengths]
    kv_len_arr = list(cache_pages_arr)
    policy_batch = _graph_policy_batch(
        mode=mode,
        batch=batch,
        total_q=total_q,
    )

    if enable_cuda_graph:
        total_num_rows = total_q
        max_seq_len = total_num_rows - batch + 1
        max_qo_len = max_seq_len * gqa_group_size
        max_effective_kv_pages = max(kv_len_arr) if window_left < 0 else min(
            _ceil_div(window_left + _fa2_determine_cta_tile_q(max_qo_len, head_dim_qk), page_size),
            max(kv_len_arr),
        )
        cta_tile_q = _paged_determine_cta_tile_q(
            mode=mode,
            kv_dtype=k_cache.dtype,
            packed_qo_len=max_qo_len,
            head_dim=head_dim_qk,
            max_effective_kv_pages=max(max_effective_kv_pages, 1),
        )
        total_num_qo_tiles = _ceil_div(total_num_rows * gqa_group_size, cta_tile_q) + batch - 1
    else:
        avg_packed_qo_len = sum(packed_qo_len_arr) // max(batch, 1)
        max_effective_kv_pages = max(kv_len_arr) if window_left < 0 else min(
            _ceil_div(window_left + _fa2_determine_cta_tile_q(avg_packed_qo_len, head_dim_qk), page_size),
            max(kv_len_arr),
        )
        cta_tile_q = _paged_determine_cta_tile_q(
            mode=mode,
            kv_dtype=k_cache.dtype,
            packed_qo_len=avg_packed_qo_len,
            head_dim=head_dim_qk,
            max_effective_kv_pages=max(max_effective_kv_pages, 1),
        )
        total_num_qo_tiles = sum(_ceil_div(packed_qo_len, cta_tile_q) for packed_qo_len in packed_qo_len_arr)

    effective_kv_len_arr = [
        min(_ceil_div(window_left + cta_tile_q, page_size), kv_len) if window_left >= 0 else kv_len
        for kv_len in kv_len_arr
    ]
    kv_window_start_pages = [
        max(kv_pages - effective_pages, 0)
        for kv_pages, effective_pages in zip(kv_len_arr, effective_kv_len_arr)
    ]
    kv_window_start_tokens = tuple(start_page * page_size for start_page in kv_window_start_pages)
    resolved_graph_ctas_per_sm = 0
    if enable_cuda_graph:
        resolved_graph_ctas_per_sm = _resolve_graph_ctas_per_sm(
            mode=mode,
            kv_dtype=k_cache.dtype,
            policy_batch=policy_batch,
            max_effective_kv_pages=max(max(effective_kv_len_arr), 1),
            graph_ctas_per_sm=graph_ctas_per_sm,
        )
    if max_batch_size_if_split is None:
        if enable_cuda_graph:
            max_batch_size_if_split = _graph_max_batch_size_if_split(
                device=q.device,
                num_kv_heads=num_kv_heads,
                graph_ctas_per_sm=resolved_graph_ctas_per_sm,
            )
        else:
            max_batch_size_if_split = max(total_num_qo_tiles, 1) * max(max(effective_kv_len_arr), 1)
    regularized_decode_graph = bool(enable_cuda_graph and mode == "decode" and total_num_qo_tiles == batch and batch == 16)
    padded_batch_size = max(max_batch_size_if_split, total_num_qo_tiles) if enable_cuda_graph else 0
    if regularized_decode_graph:
        padded_batch_size = _align_up(padded_batch_size, max(batch, 1))

    if not _merge_backend_supports_split_kv(output_dtype=q.dtype, head_dim_vo=head_dim_vo):
        disable_split_kv = True

    min_kv_chunk_size = max(128 // page_size, 1)
    if mode == "extend":
        split_kv = False
        disable_split_kv = True
        required_kv_chunk_size_pages = max(max(effective_kv_len_arr), 1)
        if fixed_split_size > 0:
            if fixed_split_size < required_kv_chunk_size_pages:
                raise ValueError(
                    "extend fixed_split_size must cover the full effective KV span when split-kv is disabled"
                )
            kv_chunk_size_pages = fixed_split_size
        else:
            kv_chunk_size_pages = required_kv_chunk_size_pages
    elif disable_split_kv and not force_split_kv:
        split_kv = False
        kv_chunk_size_pages = 1 << 30
    elif fixed_split_size > 0:
        split_kv = False
        kv_chunk_size_pages = fixed_split_size
    else:
        heuristic_kv_chunk_size_pages = chunk_pages_for_family(
            mode=mode,
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            policy_batch=policy_batch,
            graph_chunk_policy=bool(enable_cuda_graph and graph_chunk_policy),
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=max(max(effective_kv_len_arr), 1),
        )
        heuristic_fits_graph_budget = True
        heuristic_fits_plan_budget = True
        if heuristic_kv_chunk_size_pages is not None and enable_cuda_graph:
            heuristic_new_batch_size = _estimate_new_batch_size(
                packed_qo_len_arr=packed_qo_len_arr,
                kv_len_arr=effective_kv_len_arr,
                qo_chunk_size=cta_tile_q,
                kv_chunk_size_pages=heuristic_kv_chunk_size_pages,
            )
            heuristic_fits_graph_budget = heuristic_new_batch_size <= padded_batch_size
        if heuristic_kv_chunk_size_pages is not None:
            heuristic_new_batch_size, heuristic_total_num_partial_rows = _estimate_prefill_plan_usage(
                packed_qo_len_arr=packed_qo_len_arr,
                q_lengths=q_lengths,
                kv_len_arr=effective_kv_len_arr,
                qo_chunk_size=cta_tile_q,
                kv_chunk_size_pages=heuristic_kv_chunk_size_pages,
                force_split_kv=bool(force_split_kv),
            )
            heuristic_fits_plan_budget = _prefill_usage_fits_budget(
                budget=plan_budget,
                new_batch_size=heuristic_new_batch_size,
                total_num_partial_rows=heuristic_total_num_partial_rows,
            )
        if (
            heuristic_kv_chunk_size_pages is not None
            and heuristic_fits_graph_budget
            and heuristic_fits_plan_budget
        ):
            split_kv = False
            kv_chunk_size_pages = heuristic_kv_chunk_size_pages
        else:
            split_kv, kv_chunk_size_pages = _prefill_binary_search_kv_chunk_size(
                enable_cuda_graph=enable_cuda_graph,
                max_batch_size_if_split=max_batch_size_if_split,
                packed_qo_len_arr=packed_qo_len_arr,
                q_lengths=q_lengths,
                kv_len_arr=effective_kv_len_arr,
                qo_chunk_size=cta_tile_q,
                force_split_kv=bool(force_split_kv),
                plan_budget=plan_budget,
                min_kv_chunk_size=min_kv_chunk_size,
            )

    request_indices: list[int] = []
    qo_tile_indices: list[int] = []
    kv_tile_indices: list[int] = []
    merge_indptr: list[int] = [0]
    o_indptr: list[int] = [0]
    request_num_chunks_kv: list[int] = []
    new_batch_size = 0

    for request_idx, (packed_qo_len, qo_len, kv_len) in enumerate(
        zip(packed_qo_len_arr, q_lengths, effective_kv_len_arr)
    ):
        num_tiles_q = _ceil_div(packed_qo_len, cta_tile_q)
        num_chunks_kv = 1 if disable_split_kv and not force_split_kv else _ceil_div(max(kv_len, 1), kv_chunk_size_pages)
        request_num_chunks_kv.append(int(num_chunks_kv))
        if not disable_split_kv or force_split_kv:
            split_kv = split_kv or num_chunks_kv > 1
        for q_tile_idx in range(num_tiles_q):
            for kv_tile_idx in range(num_chunks_kv):
                new_batch_size += 1
                request_indices.append(request_idx)
                qo_tile_indices.append(q_tile_idx)
                kv_tile_indices.append(kv_tile_idx)
        for _ in range(qo_len):
            merge_indptr.append(merge_indptr[-1] + num_chunks_kv)
        o_indptr.append(o_indptr[-1] + qo_len * num_chunks_kv)

    padded_batch_size = max(max_batch_size_if_split, total_num_qo_tiles) if enable_cuda_graph else new_batch_size
    if regularized_decode_graph:
        padded_batch_size = _align_up(padded_batch_size, max(batch, 1))
    if new_batch_size > padded_batch_size:
        raise ValueError(
            "new_batch_size exceeds padded_batch_size; fixed_split_size is incompatible with the chosen graph budget"
        )
    if force_split_kv:
        split_kv = True
    total_num_partial_rows = o_indptr[-1] if split_kv else 0
    if not _prefill_usage_fits_budget(
        budget=plan_budget,
        new_batch_size=new_batch_size,
        total_num_partial_rows=total_num_partial_rows,
    ):
        raise ValueError("paged prefill plan exceeds the configured eager workspace budget")

    if regularized_decode_graph:
        max_chunks_per_req = padded_batch_size // max(batch, 1)
        block_valid_mask = [False for _ in range(padded_batch_size)]
        for request_idx, num_chunks_kv in enumerate(request_num_chunks_kv):
            base_idx = request_idx * max_chunks_per_req
            for kv_tile_idx in range(num_chunks_kv):
                block_valid_mask[base_idx + kv_tile_idx] = True
    else:
        block_valid_mask = [idx < new_batch_size for idx in range(padded_batch_size)]
    kv_chunk_size = kv_chunk_size_pages * page_size

    if (
        os.environ.get("B12X_DEBUG_PAGED_POLICY") == "1"
        and enable_cuda_graph
        and graph_chunk_policy
        and mode == "decode"
    ):
        print(
            "# b12x_paged_policy"
            f" batch={batch}"
            f" pages={max(max(effective_kv_len_arr), 1)}"
            f" graph_ctas_per_sm={resolved_graph_ctas_per_sm}"
            f" chunk_pages={kv_chunk_size_pages}"
            f" split_kv={int(split_kv)}"
            f" padded_batch_size={padded_batch_size}"
            f" new_batch_size={new_batch_size}",
            file=sys.stderr,
            flush=True,
        )

    key = PagedPlanKey(
        total_q=total_q,
        num_q_heads=num_q_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        k_cache_shape=tuple(int(dim) for dim in k_cache.shape),
        v_cache_shape=tuple(int(dim) for dim in v_cache.shape),
        page_table_shape=tuple(int(dim) for dim in page_table.shape),
        dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        mode=mode,
        cta_tile_q=cta_tile_q,
        kv_chunk_size=kv_chunk_size,
        split_kv=split_kv,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
        enable_cuda_graph=enable_cuda_graph,
        graph_chunk_policy=graph_chunk_policy,
        graph_ctas_per_sm=resolved_graph_ctas_per_sm,
        window_left=window_left,
        max_batch_size_if_split=max_batch_size_if_split,
        padded_batch_size=padded_batch_size,
        new_batch_size=new_batch_size,
        num_qo_tiles=total_num_qo_tiles,
        total_num_partial_rows=total_num_partial_rows,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        gqa_group_size=gqa_group_size,
        device_index=q.device.index if q.device.index is not None else torch.cuda.current_device(),
    )
    return PagedPlan(
        key=key,
        request_indices=tuple(request_indices),
        qo_tile_indices=tuple(qo_tile_indices),
        kv_tile_indices=tuple(kv_tile_indices),
        merge_indptr=tuple(merge_indptr),
        o_indptr=tuple(o_indptr),
        block_valid_mask=tuple(block_valid_mask),
        kv_window_start_tokens=kv_window_start_tokens,
    )
