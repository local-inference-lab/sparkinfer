"""Generic paged-MQA indexer integration surface.

This module exposes the paged FP8 MQA scorer behind algorithmic names.  The
implementation is shared with the NSA indexer path, but callers should use this
surface when they only need paged indexer logits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from b12x.attention.nsa_indexer import (
    NSAIndexerPagedDecodeMetadata,
    get_paged_mqa_logits_metadata,
    make_nsa_indexer_contract_phantoms,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_logits_paged,
    sparse_nsa_paged_logits_reference,
    unpack_nsa_index_k_cache_reference,
    uses_paged_mqa_schedule_metadata,
)
from b12x.attention.nsa_indexer.kernel import run_sparse_nsa_paged_tiled_logits_kernel
from b12x.attention.nsa_indexer.tiled_topk import (
    merge_tiled_topk_candidates,
    run_row_topk,
    run_tiled_topk,
)


INDEX_HEAD_DIM = 128
PAGED_MQA_INDEX_PAGE_SIZE = 64
_PAGED_MQA_INDEX_SUPERTILE_K_ENV = "B12X_PAGED_MQA_INDEX_SUPERTILE_K"
_PAGED_MQA_INDEX_SUPERTILE_K_DEFAULT = 32768
_PAGED_MQA_INDEX_TILE_BLOCK_Q = 32
_PAGED_MQA_INDEX_TILE_BLOCK_K = 512


@triton.jit
def _prepare_c4_supertile_metadata_kernel(
    src_page_table,
    src_lengths,
    dst_page_table,
    dst_lengths,
    dst_active_width,
    q_rows: tl.constexpr,
    src_page_width: tl.constexpr,
    src_row_stride: tl.constexpr,
    src_col_stride: tl.constexpr,
    dst_page_width: tl.constexpr,
    dst_row_stride: tl.constexpr,
    dst_col_stride: tl.constexpr,
    total_table: tl.constexpr,
    page_begin,
    chunk_pages,
    chunk_start_token,
    chunk_width_tokens,
    active_width_value,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    table_mask = offs < total_table
    table_row = offs // dst_page_width
    table_col = offs - table_row * dst_page_width
    src_col = page_begin + table_col
    copy_mask = table_mask & (table_col < chunk_pages) & (src_col < src_page_width)
    page = tl.load(
        src_page_table + table_row * src_row_stride + src_col * src_col_stride,
        mask=copy_mask,
        other=-1,
    )
    tl.store(
        dst_page_table + table_row * dst_row_stride + table_col * dst_col_stride,
        page,
        mask=table_mask,
    )

    length_mask = offs < q_rows
    full_length = tl.load(src_lengths + offs, mask=length_mask, other=0)
    chunk_length = full_length - chunk_start_token
    chunk_length = tl.maximum(chunk_length, 0)
    chunk_length = tl.minimum(chunk_length, chunk_width_tokens)
    tl.store(dst_lengths + offs, chunk_length, mask=length_mask)
    tl.store(dst_active_width, active_width_value, mask=pid == 0)


@dataclass(frozen=True)
class PagedMQAIndexerMetadata:
    """Metadata for paged FP8 MQA indexer logits.

    ``expected_num_q_heads`` is optional for the generic path, but integrations
    should set it to the exact indexer-head count they pass to b12x. Replicated
    selector paths such as C4 use the full model-global selector-head count on
    every attention TP rank.
    """

    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    paged_mqa_schedule_metadata: torch.Tensor | None = None
    expected_num_q_heads: int | None = None


def resolve_replicated_num_q_heads(
    *,
    global_num_q_heads: int,
    tensor_parallel_size: int | None = None,
) -> int:
    """Return the replicated query/index head count used on every TP rank."""

    global_num_q_heads = int(global_num_q_heads)
    if global_num_q_heads <= 0:
        raise ValueError(f"global_num_q_heads must be positive, got {global_num_q_heads}")
    if tensor_parallel_size is not None and int(tensor_parallel_size) <= 0:
        raise ValueError(
            f"tensor_parallel_size must be positive, got {int(tensor_parallel_size)}"
        )
    return global_num_q_heads


def resolve_local_num_q_heads(
    *,
    global_num_q_heads: int,
    tensor_parallel_size: int,
) -> int:
    """Return a TP-local head count for legacy sharded-indexer callers."""

    global_num_q_heads = int(global_num_q_heads)
    tensor_parallel_size = int(tensor_parallel_size)
    if global_num_q_heads <= 0:
        raise ValueError(f"global_num_q_heads must be positive, got {global_num_q_heads}")
    if tensor_parallel_size <= 0:
        raise ValueError(
            f"tensor_parallel_size must be positive, got {tensor_parallel_size}"
        )
    if global_num_q_heads % tensor_parallel_size != 0:
        raise ValueError(
            f"global_num_q_heads={global_num_q_heads} is not divisible by "
            f"tensor_parallel_size={tensor_parallel_size}"
        )
    return global_num_q_heads // tensor_parallel_size


def make_paged_mqa_indexer_contract_phantoms(
    *,
    max_q_rows: int,
    num_heads: int,
    max_pages: int,
    page_size: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Create fixed-shape phantoms for the paged-MQA indexer launcher cache."""

    return make_nsa_indexer_contract_phantoms(
        max_q_rows=max_q_rows,
        num_heads=num_heads,
        max_pages=max_pages,
        page_size=page_size,
        device=device,
    )


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _validate_i32_contiguous(
    tensor: torch.Tensor,
    *,
    name: str,
    ndim: int,
) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank-{ndim}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_raw_page_lengths(
    *,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    page_size: int,
) -> None:
    """Reject positive lengths whose active page-table entries are missing."""

    if _is_cuda_graph_capture_active(real_page_table.device):
        raise RuntimeError("paged-MQA metadata prep must run outside CUDA graph capture")
    if real_page_table.device.type == "cuda" and os.getenv(
        "B12X_VALIDATE_PAGED_MQA_INDEXER_CUDA_VALUES", "0"
    ) != "1":
        return
    if cache_seqlens_int32.numel() == 0:
        return
    if torch.any(cache_seqlens_int32 < 0).item():
        raise ValueError("cache_seqlens_int32 must be non-negative")

    max_width_tokens = int(real_page_table.shape[1]) * int(page_size)
    if torch.any(cache_seqlens_int32 > max_width_tokens).item():
        max_len = int(cache_seqlens_int32.max().item())
        raise ValueError(
            f"cache_seqlens_int32 contains length {max_len}, but page-table capacity "
            f"is {max_width_tokens} tokens"
        )

    required_pages = torch.div(
        cache_seqlens_int32.to(torch.int64) + int(page_size) - 1,
        int(page_size),
        rounding_mode="floor",
    )
    if real_page_table.shape[1] == 0:
        return
    cols = torch.arange(
        int(real_page_table.shape[1]),
        dtype=torch.int64,
        device=real_page_table.device,
    ).unsqueeze(0)
    active_page_mask = cols < required_pages.unsqueeze(1)
    if torch.any(active_page_mask & (real_page_table.to(torch.int64) < 0)).item():
        raise ValueError(
            "cache_seqlens_int32 marks page-table slots active, but real_page_table "
            "contains -1 in those slots; pass raw unclamped compressed lengths"
        )


def _validate_schedule_metadata(
    schedule_metadata: torch.Tensor,
    *,
    device: torch.device,
) -> None:
    _validate_i32_contiguous(
        schedule_metadata,
        name="paged_mqa_schedule_metadata",
        ndim=2,
    )
    if schedule_metadata.shape[1] != 2:
        raise ValueError(
            "paged_mqa_schedule_metadata must have trailing dimension 2, got "
            f"{tuple(schedule_metadata.shape)}"
        )
    if schedule_metadata.device != device:
        raise ValueError(
            "paged_mqa_schedule_metadata device "
            f"{schedule_metadata.device} does not match real_page_table device {device}"
        )


def prepare_paged_mqa_indexer_metadata(
    *,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    page_size: int = PAGED_MQA_INDEX_PAGE_SIZE,
    expected_num_q_heads: int | None = None,
    paged_mqa_schedule_metadata: torch.Tensor | None = None,
    schedule_out: torch.Tensor | None = None,
    schedule_num_sms: int | None = None,
    build_schedule: bool | None = None,
    validate_raw_lengths: bool = True,
) -> PagedMQAIndexerMetadata:
    """Validate and optionally build metadata for paged-MQA indexer logits.

    ``cache_seqlens_int32`` must be the raw compressed-token length for this
    indexer layout.  Do not pass attention-kernel clamp-to-1 lengths here.
    """

    page_size = int(page_size)
    if page_size != PAGED_MQA_INDEX_PAGE_SIZE:
        raise ValueError(
            f"paged-MQA indexer currently supports page_size={PAGED_MQA_INDEX_PAGE_SIZE}, "
            f"got {page_size}"
        )
    _validate_i32_contiguous(real_page_table, name="real_page_table", ndim=2)
    _validate_i32_contiguous(cache_seqlens_int32, name="cache_seqlens_int32", ndim=1)
    if real_page_table.shape[0] != cache_seqlens_int32.shape[0]:
        raise ValueError(
            f"real_page_table rows {real_page_table.shape[0]} do not match "
            f"cache_seqlens_int32 rows {cache_seqlens_int32.shape[0]}"
        )
    if real_page_table.device != cache_seqlens_int32.device:
        raise ValueError(
            f"real_page_table device {real_page_table.device} does not match "
            f"cache_seqlens_int32 device {cache_seqlens_int32.device}"
        )
    if expected_num_q_heads is not None:
        expected_num_q_heads = int(expected_num_q_heads)
        if expected_num_q_heads <= 0:
            raise ValueError(
                f"expected_num_q_heads must be positive, got {expected_num_q_heads}"
            )
    if validate_raw_lengths:
        _validate_raw_page_lengths(
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            page_size=page_size,
        )

    if build_schedule is None:
        build_schedule = uses_paged_mqa_schedule_metadata(
            q_rows=int(real_page_table.shape[0]),
            max_pages=int(real_page_table.shape[1]),
        )
    if build_schedule:
        if paged_mqa_schedule_metadata is not None and schedule_out is not None:
            raise ValueError(
                "pass only one of paged_mqa_schedule_metadata or schedule_out"
            )
        if paged_mqa_schedule_metadata is None:
            if _is_cuda_graph_capture_active(real_page_table.device):
                raise RuntimeError(
                    "paged-MQA schedule metadata must be built before CUDA graph capture"
                )
            paged_mqa_schedule_metadata = get_paged_mqa_logits_metadata(
                cache_seqlens_int32,
                page_size,
                schedule_num_sms,
                out=schedule_out,
            )
        else:
            _validate_schedule_metadata(
                paged_mqa_schedule_metadata,
                device=real_page_table.device,
            )
    elif paged_mqa_schedule_metadata is not None:
        _validate_schedule_metadata(
            paged_mqa_schedule_metadata,
            device=real_page_table.device,
        )
    elif schedule_out is not None:
        raise ValueError("schedule_out was provided, but build_schedule is false")

    return PagedMQAIndexerMetadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens_int32,
        paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
        expected_num_q_heads=expected_num_q_heads,
    )


def _metadata_to_nsa(metadata: PagedMQAIndexerMetadata) -> NSAIndexerPagedDecodeMetadata:
    return NSAIndexerPagedDecodeMetadata(
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        paged_mqa_schedule_metadata=metadata.paged_mqa_schedule_metadata,
    )


def _validate_q_head_contract(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata: PagedMQAIndexerMetadata,
    expected_num_q_heads: int | None,
    allow_partial_rows: bool,
) -> int:
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != INDEX_HEAD_DIM:
        raise ValueError(f"q_fp8 head_dim must be {INDEX_HEAD_DIM}, got {q_fp8.shape[2]}")
    if expected_num_q_heads is not None and metadata.expected_num_q_heads is not None:
        if int(expected_num_q_heads) != int(metadata.expected_num_q_heads):
            raise ValueError(
                "expected_num_q_heads argument does not match metadata "
                f"({expected_num_q_heads} vs {metadata.expected_num_q_heads})"
            )
    expected_heads = (
        int(expected_num_q_heads)
        if expected_num_q_heads is not None
        else metadata.expected_num_q_heads
    )
    if expected_heads is not None and q_fp8.shape[1] != int(expected_heads):
        raise ValueError(
            f"q_fp8 must use expected indexer head count {int(expected_heads)}, got "
            f"{q_fp8.shape[1]}"
        )
    if weights.ndim == 3:
        if weights.shape[2] != 1:
            raise ValueError(
                f"weights rank-3 input must have trailing dimension 1, got {tuple(weights.shape)}"
            )
        weight_shape = (weights.shape[0], weights.shape[1])
    elif weights.ndim == 2:
        weight_shape = tuple(weights.shape)
    else:
        raise ValueError(f"weights must be rank-2 or rank-3, got {tuple(weights.shape)}")
    if weight_shape != (q_fp8.shape[0], q_fp8.shape[1]):
        raise ValueError(
            f"weights must have shape {(q_fp8.shape[0], q_fp8.shape[1])}, got "
            f"{tuple(weights.shape)}"
        )
    metadata_rows = int(metadata.real_page_table.shape[0])
    if allow_partial_rows:
        if metadata_rows > q_fp8.shape[0]:
            raise ValueError(
                f"metadata rows {metadata_rows} exceed q rows {q_fp8.shape[0]}"
            )
    elif metadata_rows != q_fp8.shape[0]:
        raise ValueError(
            f"metadata rows {metadata_rows} must match q rows {q_fp8.shape[0]}"
        )
    return int(expected_heads) if expected_heads is not None else int(q_fp8.shape[1])


def _weights_as_2d(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim == 3:
        return weights.squeeze(-1)
    return weights


def paged_mqa_index_decode_logits_fp8(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: PagedMQAIndexerMetadata,
    page_size: int = PAGED_MQA_INDEX_PAGE_SIZE,
    expected_num_q_heads: int | None = None,
    contract_phantoms: dict[str, torch.Tensor] | None = None,
    workspace=None,
    preinitialize_invalid_logits: bool = True,
    active_width_override: torch.Tensor | None = None,
    allow_partial_rows: bool = False,
) -> torch.Tensor:
    """Compute paged FP8 MQA indexer logits with an explicit head contract."""

    page_size = int(page_size)
    if page_size != PAGED_MQA_INDEX_PAGE_SIZE:
        raise ValueError(
            f"paged-MQA indexer currently supports page_size={PAGED_MQA_INDEX_PAGE_SIZE}, "
            f"got {page_size}"
        )
    _validate_q_head_contract(
        q_fp8=q_fp8,
        weights=weights,
        metadata=metadata,
        expected_num_q_heads=expected_num_q_heads,
        allow_partial_rows=allow_partial_rows,
    )
    weights = _weights_as_2d(weights)
    return sparse_nsa_index_decode_logits_paged(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=_metadata_to_nsa(metadata),
        page_size=page_size,
        contract_phantoms=contract_phantoms,
        workspace=workspace,
        preinitialize_invalid_logits=preinitialize_invalid_logits,
        active_width_override=active_width_override,
    )


def _resolve_supertile_k(supertile_k: int | None, *, page_size: int) -> int:
    if supertile_k is None:
        raw = os.environ.get(_PAGED_MQA_INDEX_SUPERTILE_K_ENV)
        if raw is None:
            supertile_k = _PAGED_MQA_INDEX_SUPERTILE_K_DEFAULT
        else:
            try:
                supertile_k = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{_PAGED_MQA_INDEX_SUPERTILE_K_ENV} must be an integer, got {raw!r}"
                ) from exc
    alignment = _PAGED_MQA_INDEX_TILE_BLOCK_K
    if alignment % int(page_size) != 0:
        raise ValueError(
            f"internal C4 supertile alignment {alignment} must be divisible by page_size={page_size}"
        )
    supertile_k = max(int(supertile_k), alignment)
    return ((supertile_k + alignment - 1) // alignment) * alignment


def _prepare_c4_supertile_metadata(
    *,
    source_page_table: torch.Tensor,
    source_lengths: torch.Tensor,
    destination_page_table: torch.Tensor,
    destination_lengths: torch.Tensor,
    destination_active_width: torch.Tensor,
    page_begin: int,
    chunk_pages: int,
    chunk_start_token: int,
    chunk_width_tokens: int,
    active_width_value: int,
) -> None:
    if source_page_table.ndim != 2 or destination_page_table.ndim != 2:
        raise ValueError("C4 supertile page tables must be rank-2")
    if source_lengths.ndim != 1 or destination_lengths.ndim != 1:
        raise ValueError("C4 supertile lengths must be rank-1")
    q_rows = int(destination_page_table.shape[0])
    if source_page_table.shape[0] != q_rows or source_lengths.shape[0] != q_rows:
        raise ValueError("C4 supertile source metadata rows must match destination rows")
    if destination_lengths.shape[0] != q_rows:
        raise ValueError("C4 supertile destination lengths rows must match page-table rows")
    if destination_active_width.shape != (1,):
        raise ValueError("C4 supertile active-width buffer must have shape (1,)")
    if q_rows == 0:
        return
    block = 256
    total_table = q_rows * int(destination_page_table.shape[1])
    grid = (triton.cdiv(max(total_table, q_rows), block),)
    _prepare_c4_supertile_metadata_kernel[grid](
        source_page_table,
        source_lengths,
        destination_page_table,
        destination_lengths,
        destination_active_width,
        q_rows,
        int(source_page_table.shape[1]),
        int(source_page_table.stride(0)),
        int(source_page_table.stride(1)),
        int(destination_page_table.shape[1]),
        int(destination_page_table.stride(0)),
        int(destination_page_table.stride(1)),
        int(total_table),
        int(page_begin),
        int(chunk_pages),
        int(chunk_start_token),
        int(chunk_width_tokens),
        int(active_width_value),
        BLOCK=block,
    )


def paged_mqa_index_decode_supertile_topk_fp8(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: PagedMQAIndexerMetadata,
    page_size: int = PAGED_MQA_INDEX_PAGE_SIZE,
    topk: int = 512,
    expected_num_q_heads: int | None = None,
    workspace=None,
    out_indices: torch.Tensor | None = None,
    supertile_k: int | None = None,
) -> torch.Tensor:
    """Score paged C4 supertiles and select top-k with the shared NSA top-k core."""

    page_size = int(page_size)
    if page_size != PAGED_MQA_INDEX_PAGE_SIZE:
        raise ValueError(
            f"paged-MQA indexer currently supports page_size={PAGED_MQA_INDEX_PAGE_SIZE}, "
            f"got {page_size}"
        )
    topk = int(topk)
    _validate_q_head_contract(
        q_fp8=q_fp8,
        weights=weights,
        metadata=metadata,
        expected_num_q_heads=expected_num_q_heads,
        allow_partial_rows=False,
    )
    weights = _weights_as_2d(weights)
    if q_fp8.device.type != "cuda":
        raise NotImplementedError("paged MQA index supertile top-k requires CUDA")
    if workspace is None:
        raise RuntimeError("paged MQA index supertile top-k requires a b12x workspace")
    if metadata.real_page_table.device != q_fp8.device:
        raise ValueError("real_page_table must be on the same device as q_fp8")
    if not metadata.real_page_table.is_contiguous():
        raise ValueError("metadata.real_page_table must be contiguous")

    q_rows = int(q_fp8.shape[0])
    if out_indices is not None:
        if out_indices.shape != (q_rows, topk):
            raise ValueError(
                f"out_indices must have shape {(q_rows, topk)}, got "
                f"{tuple(out_indices.shape)}"
            )
        if out_indices.dtype != torch.int32 or not out_indices.is_contiguous():
            raise ValueError("out_indices must be contiguous torch.int32")

    page_table_width = int(metadata.real_page_table.shape[1])
    supertile_tokens = _resolve_supertile_k(supertile_k, page_size=page_size)
    supertile_pages = max(1, supertile_tokens // page_size)
    num_chunks = max(1, (page_table_width + supertile_pages - 1) // supertile_pages)
    if int(getattr(workspace, "max_page_table_width", 0)) < supertile_pages:
        raise RuntimeError(
            "paged MQA index supertile top-k workspace page-table capacity is too small: "
            f"need={supertile_pages}, have={getattr(workspace, 'max_page_table_width', None)}"
        )
    tile_logits = workspace.get_indexer_extend_tile_logits()
    if tile_logits is None:
        raise RuntimeError(
            "paged MQA index supertile top-k requires the workspace tiled-logits buffer"
        )
    contract_phantoms = workspace.get_paged_indexer_contract_phantoms()

    final_values, workspace_raw_indices = workspace.get_indexer_extend_topk_buffers(
        row_count=q_rows,
    )
    final_values = final_values[:, :topk]
    workspace_raw_indices = workspace_raw_indices[:, :topk]
    final_raw_indices = out_indices if out_indices is not None else workspace_raw_indices
    if final_values.shape != (q_rows, topk) or final_raw_indices.shape != (q_rows, topk):
        raise ValueError(
            f"workspace top-k buffers are smaller than requested C4 top-k {topk}"
        )
    candidate_values = None
    candidate_indices = None
    if num_chunks > 1:
        candidate_values, candidate_indices = workspace.get_indexer_extend_candidate_buffers()
        if candidate_values.shape[0] < num_chunks or candidate_indices.shape[0] < num_chunks:
            raise RuntimeError(
                "workspace C4 candidate buffers cannot hold all supertile chunks: "
                f"need={num_chunks}, have={candidate_values.shape[0]}"
            )
        candidate_values = candidate_values[:num_chunks, :q_rows, :topk]
        candidate_indices = candidate_indices[:num_chunks, :q_rows, :topk]

    workspace._allocate_paged_indexer_runtime_metadata()
    real_page_table_runtime = workspace.paged_indexer_real_page_table_runtime
    if real_page_table_runtime is None:
        raise RuntimeError("workspace did not allocate paged indexer page-table buffer")
    if real_page_table_runtime.shape[0] < q_rows or real_page_table_runtime.shape[1] < supertile_pages:
        raise RuntimeError(
            "workspace paged indexer page-table buffer is too small for C4 supertiles: "
            f"need={(q_rows, supertile_pages)}, have={tuple(real_page_table_runtime.shape)}"
        )
    active_width = workspace.paged_indexer_active_width_runtime
    if active_width is None:
        raise RuntimeError("workspace did not allocate paged indexer active-width scalar")
    chunk_lengths_runtime = workspace.paged_indexer_seqlens_per_query_runtime
    if chunk_lengths_runtime is None:
        raise RuntimeError("workspace did not allocate paged indexer length buffer")
    chunk_lengths = chunk_lengths_runtime[:q_rows]

    for chunk_idx in range(num_chunks):
        page_begin = chunk_idx * supertile_pages
        page_end = min(page_begin + supertile_pages, page_table_width)
        chunk_pages = page_end - page_begin
        chunk_width_tokens = chunk_pages * page_size
        chunk_start_token = page_begin * page_size
        if uses_paged_mqa_schedule_metadata(q_rows=q_rows, max_pages=chunk_pages):
            raise RuntimeError(
                "C4 supertile top-k requires an unscheduled paged scorer tile; "
                f"reduce {_PAGED_MQA_INDEX_SUPERTILE_K_ENV} below "
                f"{chunk_width_tokens} tokens"
            )

        fixed_page_table = real_page_table_runtime[:q_rows, :supertile_pages]
        _prepare_c4_supertile_metadata(
            source_page_table=metadata.real_page_table,
            source_lengths=metadata.cache_seqlens_int32,
            destination_page_table=fixed_page_table,
            destination_lengths=chunk_lengths,
            destination_active_width=active_width,
            page_begin=page_begin,
            chunk_pages=chunk_pages,
            chunk_start_token=chunk_start_token,
            chunk_width_tokens=chunk_width_tokens,
            active_width_value=supertile_tokens,
        )
        logits = run_sparse_nsa_paged_tiled_logits_kernel(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            real_page_table=fixed_page_table,
            seqlens_per_query=chunk_lengths,
            active_width=active_width,
            tile_logits=tile_logits,
            page_size=page_size,
            tile_block_q=_PAGED_MQA_INDEX_TILE_BLOCK_Q,
            tile_block_k=_PAGED_MQA_INDEX_TILE_BLOCK_K,
            workspace=workspace,
            preinitialize_tile_logits=False,
            contract_phantoms=contract_phantoms,
        )
        if not logits.is_contiguous():
            raise RuntimeError("C4 supertile scorer returned non-contiguous tiled logits")

        out_values = final_values
        out_indices = final_raw_indices
        if candidate_values is not None and candidate_indices is not None:
            out_values = candidate_values[chunk_idx]
            out_indices = candidate_indices[chunk_idx]
        run_tiled_topk(
            tile_logits=tile_logits,
            k_start=None,
            lengths=chunk_lengths,
            topk=topk,
            block_q=_PAGED_MQA_INDEX_TILE_BLOCK_Q,
            block_k=_PAGED_MQA_INDEX_TILE_BLOCK_K,
            output_values=out_values,
            output_indices=out_indices,
            num_k_tiles=supertile_tokens // _PAGED_MQA_INDEX_TILE_BLOCK_K,
            input_index_offset=0,
            input_extent=chunk_width_tokens,
            output_index_offset=chunk_start_token,
            zero_row_start=True,
            contract_phantoms=contract_phantoms,
        )

    if candidate_values is not None and candidate_indices is not None:
        merged_values, merged_indices = merge_tiled_topk_candidates(
            candidate_values=candidate_values,
            candidate_indices=candidate_indices,
            topk=topk,
        )
        final_values.copy_(merged_values)
        final_raw_indices.copy_(merged_indices)

    return final_raw_indices


def paged_mqa_index_decode_dense_topk_fp8(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: PagedMQAIndexerMetadata,
    page_size: int = PAGED_MQA_INDEX_PAGE_SIZE,
    topk: int = 512,
    expected_num_q_heads: int | None = None,
    workspace=None,
    out_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score the full paged C4 row and select top-k with the dense top-k core."""

    page_size = int(page_size)
    if page_size != PAGED_MQA_INDEX_PAGE_SIZE:
        raise ValueError(
            f"paged-MQA indexer currently supports page_size={PAGED_MQA_INDEX_PAGE_SIZE}, "
            f"got {page_size}"
        )
    topk = int(topk)
    _validate_q_head_contract(
        q_fp8=q_fp8,
        weights=weights,
        metadata=metadata,
        expected_num_q_heads=expected_num_q_heads,
        allow_partial_rows=False,
    )
    weights = _weights_as_2d(weights)
    if q_fp8.device.type != "cuda":
        raise NotImplementedError("paged MQA index dense top-k requires CUDA")
    if workspace is None:
        raise RuntimeError("paged MQA index dense top-k requires a b12x workspace")
    if metadata.real_page_table.device != q_fp8.device:
        raise ValueError("real_page_table must be on the same device as q_fp8")
    if not metadata.real_page_table.is_contiguous():
        raise ValueError("metadata.real_page_table must be contiguous")

    q_rows = int(q_fp8.shape[0])
    if out_indices is not None:
        if out_indices.shape != (q_rows, topk):
            raise ValueError(
                f"out_indices must have shape {(q_rows, topk)}, got "
                f"{tuple(out_indices.shape)}"
            )
        if out_indices.dtype != torch.int32 or not out_indices.is_contiguous():
            raise ValueError("out_indices must be contiguous torch.int32")

    scorer_metadata = metadata
    workspace_page_width = int(getattr(workspace, "max_page_table_width", 0) or 0)
    if 0 < workspace_page_width < int(metadata.real_page_table.shape[1]):
        compact_page_table = metadata.real_page_table[:, :workspace_page_width]
        if not bool(getattr(workspace, "use_cuda_graph", False)) and not compact_page_table.is_contiguous():
            raise RuntimeError(
                "paged MQA index dense top-k requires a CUDA-graph workspace when "
                "scoring a compact view of a padded page table"
            )
        scorer_metadata = PagedMQAIndexerMetadata(
            real_page_table=compact_page_table,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            paged_mqa_schedule_metadata=metadata.paged_mqa_schedule_metadata,
            expected_num_q_heads=metadata.expected_num_q_heads,
        )

    active_width = None
    if workspace is not None and hasattr(workspace, "get_paged_indexer_active_width_cap"):
        active_width = workspace.get_paged_indexer_active_width_cap()

    logits = paged_mqa_index_decode_logits_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=scorer_metadata,
        page_size=page_size,
        expected_num_q_heads=expected_num_q_heads,
        workspace=workspace,
        preinitialize_invalid_logits=False,
        active_width_override=active_width,
    )
    contract_phantoms = workspace.get_paged_indexer_contract_phantoms()
    final_values, workspace_raw_indices = workspace.get_indexer_extend_topk_buffers(
        row_count=q_rows,
    )
    final_values = final_values[:, :topk]
    workspace_raw_indices = workspace_raw_indices[:, :topk]
    final_raw_indices = out_indices if out_indices is not None else workspace_raw_indices
    if final_values.shape != (q_rows, topk) or final_raw_indices.shape != (q_rows, topk):
        raise ValueError(
            f"workspace top-k buffers are smaller than requested C4 top-k {topk}"
        )
    run_row_topk(
        row_logits=logits,
        lengths=metadata.cache_seqlens_int32,
        topk=topk,
        output_values=final_values,
        output_indices=final_raw_indices,
        contract_phantoms=contract_phantoms,
    )

    return final_raw_indices


pack_paged_mqa_index_k_cache_reference = pack_nsa_index_k_cache_reference
unpack_paged_mqa_index_k_cache_reference = unpack_nsa_index_k_cache_reference
paged_mqa_index_logits_reference = sparse_nsa_paged_logits_reference


__all__ = [
    "INDEX_HEAD_DIM",
    "PAGED_MQA_INDEX_PAGE_SIZE",
    "PagedMQAIndexerMetadata",
    "get_paged_mqa_logits_metadata",
    "make_paged_mqa_indexer_contract_phantoms",
    "pack_paged_mqa_index_k_cache_reference",
    "paged_mqa_index_decode_dense_topk_fp8",
    "paged_mqa_index_decode_logits_fp8",
    "paged_mqa_index_decode_supertile_topk_fp8",
    "paged_mqa_index_logits_reference",
    "prepare_paged_mqa_indexer_metadata",
    "resolve_local_num_q_heads",
    "resolve_replicated_num_q_heads",
    "unpack_paged_mqa_index_k_cache_reference",
    "uses_paged_mqa_schedule_metadata",
]
