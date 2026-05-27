"""Compressed sparse-indexer integration surface.

This module adapts the generic indexer scorer/top-k kernels to the DS4/C4
compressed page layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from b12x.attention.workspace import B12XIndexerTopKPositionBufferUnavailable
from b12x.attention.indexer import (
    IndexerPagedDecodeMetadata,
    build_paged_mqa_schedule_metadata,
    make_indexer_contract_phantoms,
    paged_decode_logits,
    uses_paged_mqa_schedule,
)
from b12x.attention.indexer.extend_kernel import (
    resolve_extend_prefill_block_k,
    run_extend_logits_kernel,
)
from b12x.attention.indexer.kernel import (
    run_paged_windowed_tiled_logits_kernel,
)
from b12x.attention.indexer.tiled_topk import (
    merge_tiled_topk_candidates,
    run_row_topk,
    run_tiled_topk,
)
from b12x.attention.indexer.reference import (
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
    unpack_index_k_cache_reference,
)


INDEX_HEAD_DIM = 128
COMPRESSED_INDEX_PAGE_SIZE = 64
_COMPRESSED_INDEX_SUPERTILE_K_ENV = "B12X_COMPRESSED_INDEX_SUPERTILE_K"
_COMPRESSED_INDEX_SUPERTILE_K_DEFAULT = 32768
_COMPRESSED_INDEX_TILE_BLOCK_Q = 32
_COMPRESSED_INDEX_TILE_BLOCK_K = 512
_COMPRESSED_INDEX_CACHE_ROW_BYTES = COMPRESSED_INDEX_PAGE_SIZE * (INDEX_HEAD_DIM + 4)
_COMPRESSED_INDEX_CACHE_DATA_BYTES = COMPRESSED_INDEX_PAGE_SIZE * INDEX_HEAD_DIM


@triton.jit
def _gather_shared_compressed_supertile_kernel(
    index_k_cache,
    real_page_table,
    seqlens_per_query,
    k_quant_out,
    k_scale_bytes_out,
    k_start_out,
    k_end_out,
    q_rows,
    page_table_width,
    source_page_offset,
    supertile_tokens: tl.constexpr,
    block_tokens: tl.constexpr,
    page_size: tl.constexpr,
    index_head_dim: tl.constexpr,
    cache_row_bytes: tl.constexpr,
    cache_data_bytes: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = pid * block_tokens + tl.arange(0, block_tokens)
    token_mask = token_offsets < supertile_tokens

    page_cols = source_page_offset + token_offsets // page_size
    slot_offsets = token_offsets % page_size
    page_ids = tl.load(
        real_page_table + page_cols,
        mask=token_mask & (page_cols < page_table_width),
        other=-1,
    )
    valid_tokens = token_mask & (page_ids >= 0)

    byte_offsets = tl.arange(0, index_head_dim)
    k_bytes = tl.load(
        index_k_cache
        + page_ids[:, None] * cache_row_bytes
        + slot_offsets[:, None] * index_head_dim
        + byte_offsets[None, :],
        mask=valid_tokens[:, None],
        other=0,
    )
    tl.store(
        k_quant_out + token_offsets[:, None] * index_head_dim + byte_offsets[None, :],
        k_bytes,
        mask=token_mask[:, None],
    )

    scale_byte_offsets = tl.arange(0, 4)
    scale_bytes = tl.load(
        index_k_cache
        + page_ids[:, None] * cache_row_bytes
        + cache_data_bytes
        + slot_offsets[:, None] * 4
        + scale_byte_offsets[None, :],
        mask=valid_tokens[:, None],
        other=0,
    )
    tl.store(
        k_scale_bytes_out + token_offsets[:, None] * 4 + scale_byte_offsets[None, :],
        scale_bytes,
        mask=token_mask[:, None],
    )

    row_offsets = token_offsets
    row_mask = row_offsets < q_rows
    row_lengths = tl.load(seqlens_per_query + row_offsets, mask=row_mask, other=0)
    local_ends = row_lengths - source_page_offset * page_size
    local_ends = tl.minimum(tl.maximum(local_ends, 0), supertile_tokens)
    tl.store(k_start_out + row_offsets, tl.zeros((block_tokens,), tl.int32), mask=row_mask)
    tl.store(k_end_out + row_offsets, local_ends, mask=row_mask)


@dataclass(frozen=True)
class CompressedIndexerMetadata:
    """Metadata for paged FP8 MQA indexer logits.

    ``expected_num_q_heads`` is optional for the generic path, but integrations
    should set it to the exact indexer-head count they pass to b12x. Replicated
    selector paths such as C4 use the full model-global selector-head count on
    every attention TP rank.
    """

    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    schedule_metadata: torch.Tensor | None = None
    expected_num_q_heads: int | None = None
    shared_page_table: bool = False


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


def make_compressed_indexer_contract_phantoms(
    *,
    max_q_rows: int,
    num_heads: int,
    max_pages: int,
    page_size: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Create fixed-shape phantoms for the compressed indexer launcher cache."""

    return make_indexer_contract_phantoms(
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
        raise RuntimeError("compressed-index metadata prep must run outside CUDA graph capture")
    if real_page_table.device.type == "cuda" and os.getenv(
        "B12X_VALIDATE_COMPRESSED_INDEXER_CUDA_VALUES", "0"
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
        name="schedule_metadata",
        ndim=2,
    )
    if schedule_metadata.shape[1] != 2:
        raise ValueError(
            "schedule_metadata must have trailing dimension 2, got "
            f"{tuple(schedule_metadata.shape)}"
        )
    if schedule_metadata.device != device:
        raise ValueError(
            "schedule_metadata device "
            f"{schedule_metadata.device} does not match real_page_table device {device}"
        )


def prepare_compressed_indexer_metadata(
    *,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    page_size: int = COMPRESSED_INDEX_PAGE_SIZE,
    expected_num_q_heads: int | None = None,
    schedule_metadata: torch.Tensor | None = None,
    schedule_out: torch.Tensor | None = None,
    schedule_num_sms: int | None = None,
    build_schedule: bool | None = None,
    validate_raw_lengths: bool = True,
    shared_page_table: bool = False,
) -> CompressedIndexerMetadata:
    """Validate and optionally build metadata for compressed indexer logits.

    ``cache_seqlens_int32`` must be the raw compressed-token length for this
    indexer layout.  Do not pass attention-kernel clamp-to-1 lengths here.
    """

    page_size = int(page_size)
    if page_size != COMPRESSED_INDEX_PAGE_SIZE:
        raise ValueError(
            f"compressed indexer currently supports page_size={COMPRESSED_INDEX_PAGE_SIZE}, "
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
        build_schedule = uses_paged_mqa_schedule(
            q_rows=int(real_page_table.shape[0]),
            max_pages=int(real_page_table.shape[1]),
        )
    if build_schedule:
        if schedule_metadata is not None and schedule_out is not None:
            raise ValueError(
                "pass only one of schedule_metadata or schedule_out"
            )
        if schedule_metadata is None:
            if _is_cuda_graph_capture_active(real_page_table.device):
                raise RuntimeError(
                    "compressed-indexer schedule metadata must be built before CUDA graph capture"
                )
            schedule_metadata = build_paged_mqa_schedule_metadata(
                cache_seqlens_int32,
                page_size,
                schedule_num_sms,
                out=schedule_out,
            )
        else:
            _validate_schedule_metadata(
                schedule_metadata,
                device=real_page_table.device,
            )
    elif schedule_metadata is not None:
        _validate_schedule_metadata(
            schedule_metadata,
            device=real_page_table.device,
        )
    elif schedule_out is not None:
        raise ValueError("schedule_out was provided, but build_schedule is false")

    return CompressedIndexerMetadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens_int32,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=expected_num_q_heads,
        shared_page_table=bool(shared_page_table),
    )


def _metadata_to_indexer(metadata: CompressedIndexerMetadata) -> IndexerPagedDecodeMetadata:
    return IndexerPagedDecodeMetadata(
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        paged_mqa_schedule_metadata=metadata.schedule_metadata,
    )


def _prepare_shared_compressed_supertile(
    *,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    workspace,
    q_rows: int,
    page_table_width: int,
    page_begin: int,
    supertile_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if index_k_cache.ndim != 2 or index_k_cache.dtype != torch.uint8:
        raise ValueError(
            "shared compressed-index supertile gather requires uint8 index_k_cache with "
            f"rank 2, got shape={tuple(index_k_cache.shape)} dtype={index_k_cache.dtype}"
        )
    if not index_k_cache.is_contiguous():
        raise ValueError("shared compressed-index supertile gather requires contiguous index_k_cache")
    expected_width = COMPRESSED_INDEX_PAGE_SIZE * (INDEX_HEAD_DIM + 4)
    if int(index_k_cache.shape[1]) != expected_width:
        raise ValueError(
            f"index_k_cache width must be {expected_width}, got {int(index_k_cache.shape[1])}"
        )

    k_quant_bytes, k_scale_bytes = workspace.get_indexer_gather_outputs(
        row_count=supertile_tokens,
    )
    k_start = workspace.get_indexer_extend_lengths(row_count=q_rows)
    get_end = getattr(workspace, "get_paged_indexer_runtime_lengths", None)
    if get_end is None:
        raise RuntimeError("workspace is missing paged indexer runtime length scratch")
    k_end = get_end(row_count=q_rows)

    block_tokens = 128
    grid_elems = max(int(supertile_tokens), int(q_rows))
    grid = (triton.cdiv(grid_elems, block_tokens),)
    _gather_shared_compressed_supertile_kernel[grid](
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        k_quant_bytes,
        k_scale_bytes,
        k_start,
        k_end,
        q_rows,
        int(page_table_width),
        int(page_begin),
        int(supertile_tokens),
        block_tokens,
        COMPRESSED_INDEX_PAGE_SIZE,
        INDEX_HEAD_DIM,
        _COMPRESSED_INDEX_CACHE_ROW_BYTES,
        _COMPRESSED_INDEX_CACHE_DATA_BYTES,
        num_warps=4,
    )

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is required for shared compressed-index scoring")
    return (
        k_quant_bytes.view(fp8_dtype),
        k_scale_bytes.view(torch.float32).view(-1),
        k_start,
        k_end,
    )


def _validate_q_head_contract(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata: CompressedIndexerMetadata,
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


def compressed_index_decode_logits_fp8(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: CompressedIndexerMetadata,
    page_size: int = COMPRESSED_INDEX_PAGE_SIZE,
    expected_num_q_heads: int | None = None,
    contract_phantoms: dict[str, torch.Tensor] | None = None,
    workspace=None,
    preinitialize_invalid_logits: bool = True,
    active_width_override: torch.Tensor | None = None,
    allow_partial_rows: bool = False,
) -> torch.Tensor:
    """Compute paged FP8 MQA indexer logits with an explicit head contract."""

    page_size = int(page_size)
    if page_size != COMPRESSED_INDEX_PAGE_SIZE:
        raise ValueError(
            f"compressed indexer currently supports page_size={COMPRESSED_INDEX_PAGE_SIZE}, "
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
    return paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=_metadata_to_indexer(metadata),
        page_size=page_size,
        contract_phantoms=contract_phantoms,
        workspace=workspace,
        preinitialize_invalid_logits=preinitialize_invalid_logits,
        active_width_override=active_width_override,
    )


def _resolve_supertile_k(supertile_k: int | None, *, page_size: int) -> int:
    if supertile_k is None:
        raw = os.environ.get(_COMPRESSED_INDEX_SUPERTILE_K_ENV)
        if raw is None:
            supertile_k = _COMPRESSED_INDEX_SUPERTILE_K_DEFAULT
        else:
            try:
                supertile_k = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{_COMPRESSED_INDEX_SUPERTILE_K_ENV} must be an integer, got {raw!r}"
                ) from exc
    alignment = _COMPRESSED_INDEX_TILE_BLOCK_K
    if alignment % int(page_size) != 0:
        raise ValueError(
            f"internal C4 supertile alignment {alignment} must be divisible by page_size={page_size}"
        )
    supertile_k = max(int(supertile_k), alignment)
    return ((supertile_k + alignment - 1) // alignment) * alignment


def compressed_index_decode_supertile_topk_fp8(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: CompressedIndexerMetadata,
    page_size: int = COMPRESSED_INDEX_PAGE_SIZE,
    topk: int = 512,
    expected_num_q_heads: int | None = None,
    workspace=None,
    out_indices: torch.Tensor | None = None,
    supertile_k: int | None = None,
) -> torch.Tensor:
    """Score paged C4 supertiles and select top-k with the shared indexer top-k core."""

    page_size = int(page_size)
    if page_size != COMPRESSED_INDEX_PAGE_SIZE:
        raise ValueError(
            f"compressed indexer currently supports page_size={COMPRESSED_INDEX_PAGE_SIZE}, "
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
        raise NotImplementedError("compressed index supertile top-k requires CUDA")
    if workspace is None:
        raise RuntimeError("compressed index supertile top-k requires a b12x workspace")
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
    supertile_k_tiles = supertile_tokens // _COMPRESSED_INDEX_TILE_BLOCK_K
    num_chunks = max(1, (page_table_width + supertile_pages - 1) // supertile_pages)
    if int(getattr(workspace, "max_page_table_width", 0)) < page_table_width:
        raise RuntimeError(
            "compressed index supertile top-k workspace page-table capacity is too small: "
            f"need={page_table_width}, have={getattr(workspace, 'max_page_table_width', None)}"
        )
    require_topk_plan = getattr(workspace, "require_paged_indexer_tiled_topk_plan", None)
    if require_topk_plan is not None and bool(getattr(workspace, "fixed_capacity", False)):
        require_topk_plan(
            topk=topk,
            block_q=_COMPRESSED_INDEX_TILE_BLOCK_Q,
            block_k=_COMPRESSED_INDEX_TILE_BLOCK_K,
            num_k_tiles=supertile_k_tiles,
        )
    require_scorer_plan = getattr(workspace, "require_paged_indexer_tiled_scorer_plan", None)
    if require_scorer_plan is not None and bool(getattr(workspace, "fixed_capacity", False)):
        require_scorer_plan(
            block_q=_COMPRESSED_INDEX_TILE_BLOCK_Q,
            block_k=_COMPRESSED_INDEX_TILE_BLOCK_K,
            width_tokens=supertile_tokens,
            source_page_width=page_table_width,
        )
    tile_logits = workspace.get_indexer_extend_tile_logits()
    if tile_logits is None:
        raise RuntimeError(
            "compressed index supertile top-k requires the workspace tiled-logits buffer"
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
    merge_positions = None
    if num_chunks > 1:
        candidate_values, candidate_indices = workspace.get_indexer_extend_candidate_buffers()
        if candidate_values.shape[0] < num_chunks or candidate_indices.shape[0] < num_chunks:
            raise RuntimeError(
                "workspace C4 candidate buffers cannot hold all supertile chunks: "
                f"need={num_chunks}, have={candidate_values.shape[0]}"
            )
        candidate_values = candidate_values[:num_chunks, :q_rows, :topk]
        candidate_indices = candidate_indices[:num_chunks, :q_rows, :topk]
        get_position_buffer = getattr(
            workspace,
            "get_indexer_extend_topk_position_buffer",
            None,
        )
        if get_position_buffer is not None:
            try:
                merge_positions = get_position_buffer(row_count=q_rows)[:, :topk]
            except B12XIndexerTopKPositionBufferUnavailable:
                merge_positions = None

    active_width = workspace.get_paged_indexer_active_width_cap()
    page_table_for_kernel = metadata.real_page_table
    lengths_for_kernel = metadata.cache_seqlens_int32
    shared_prefill_candidate = bool(metadata.shared_page_table) and q_rows >= 1024
    if shared_prefill_candidate:
        shared_prefill_block_k = resolve_extend_prefill_block_k(
            valid_q_rows=q_rows,
            k_rows=supertile_tokens,
            num_heads=int(q_fp8.shape[1]),
        )
        if shared_prefill_block_k != _COMPRESSED_INDEX_TILE_BLOCK_K:
            raise RuntimeError(
                "shared compressed-index prefill scorer requires the 512-wide indexer prefill "
                "scorer to match the C4 tiled top-k contract: "
                f"resolved_block_k={shared_prefill_block_k}, "
                f"q_rows={q_rows}, k_rows={supertile_tokens}, "
                f"num_heads={int(q_fp8.shape[1])}"
            )
        if supertile_tokens % _COMPRESSED_INDEX_TILE_BLOCK_K != 0:
            raise RuntimeError(
                "shared compressed-index prefill scorer requires a supertile width "
                f"that is divisible by {_COMPRESSED_INDEX_TILE_BLOCK_K}, got {supertile_tokens}"
            )
        indexer_q_capacity = max(
            int(getattr(workspace, "max_total_q", 0)),
            int(getattr(workspace, "max_paged_q_rows", 0)),
        )
        if indexer_q_capacity < q_rows:
            raise RuntimeError(
                "shared compressed-index prefill scorer requires an indexer-capable workspace: "
                f"q_rows={q_rows}, indexer_q_capacity={indexer_q_capacity}, "
                f"max_total_q={getattr(workspace, 'max_total_q', None)}, "
                f"max_paged_q_rows={getattr(workspace, 'max_paged_q_rows', None)}"
            )
        if int(getattr(workspace, "max_paged_q_rows", 0)) < q_rows:
            raise RuntimeError(
                "shared compressed-index prefill scorer requires paged metadata capacity: "
                f"q_rows={q_rows}, max_paged_q_rows={getattr(workspace, 'max_paged_q_rows', None)}"
            )
        if getattr(workspace, "indexer_k_quant_bytes", None) is None:
            raise RuntimeError(
                "shared compressed-index prefill scorer requires workspace-backed indexer gather buffers"
            )
    use_shared_prefill_scorer = shared_prefill_candidate
    paged_contract_phantoms = contract_phantoms
    extend_contract_phantoms = (
        workspace.get_indexer_contract_phantoms() if use_shared_prefill_scorer else None
    )

    for chunk_idx in range(num_chunks):
        page_begin = chunk_idx * supertile_pages
        page_end = min(page_begin + supertile_pages, page_table_width)
        chunk_pages = page_end - page_begin
        chunk_width_tokens = chunk_pages * page_size
        chunk_start_token = page_begin * page_size
        if uses_paged_mqa_schedule(q_rows=q_rows, max_pages=chunk_pages):
            raise RuntimeError(
                "C4 supertile top-k requires an unscheduled paged scorer tile; "
                f"reduce {_COMPRESSED_INDEX_SUPERTILE_K_ENV} below "
                f"{chunk_width_tokens} tokens"
            )

        if use_shared_prefill_scorer:
            k_quant, k_scale, k_start, k_end = _prepare_shared_compressed_supertile(
                index_k_cache=index_k_cache,
                real_page_table=page_table_for_kernel,
                seqlens_per_query=lengths_for_kernel,
                workspace=workspace,
                q_rows=q_rows,
                page_table_width=page_table_width,
                page_begin=page_begin,
                supertile_tokens=supertile_tokens,
            )
            logits = run_extend_logits_kernel(
                q_fp8=q_fp8,
                weights=weights,
                k_quant=k_quant,
                k_scale=k_scale,
                k_start=k_start,
                k_end=k_end,
                contract_phantoms=extend_contract_phantoms,
                workspace=workspace,
                tile_logits=tile_logits,
                tile_k_offset=0,
                tile_num_k_tiles=supertile_k_tiles,
            )
            # The shared scorer consumes local supertile K bounds, but tiled
            # top-k clips against global raw-token offsets below.
            topk_lengths = lengths_for_kernel
        else:
            logits = run_paged_windowed_tiled_logits_kernel(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=index_k_cache,
                real_page_table=page_table_for_kernel,
                seqlens_per_query=lengths_for_kernel,
                active_width=active_width,
                tile_logits=tile_logits,
                source_page_offset=page_begin,
                output_width_tokens=supertile_tokens,
                page_size=page_size,
                tile_block_q=_COMPRESSED_INDEX_TILE_BLOCK_Q,
                tile_block_k=_COMPRESSED_INDEX_TILE_BLOCK_K,
                workspace=workspace,
                preinitialize_tile_logits=False,
                contract_phantoms=contract_phantoms,
                stage_runtime_metadata=False,
            )
            topk_lengths = lengths_for_kernel
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
            lengths=topk_lengths,
            topk=topk,
            block_q=_COMPRESSED_INDEX_TILE_BLOCK_Q,
            block_k=_COMPRESSED_INDEX_TILE_BLOCK_K,
            output_values=out_values,
            output_indices=out_indices,
            num_k_tiles=supertile_k_tiles,
            input_index_offset=chunk_start_token,
            input_extent=chunk_width_tokens,
            output_index_offset=chunk_start_token,
            zero_row_start=True,
            contract_phantoms=paged_contract_phantoms,
        )

    if candidate_values is not None and candidate_indices is not None:
        merged_values, merged_indices = merge_tiled_topk_candidates(
            candidate_values=candidate_values,
            candidate_indices=candidate_indices,
            topk=topk,
            output_values=final_values,
            output_indices=final_raw_indices,
            merge_positions=merge_positions,
        )
        final_values = merged_values
        final_raw_indices = merged_indices

    return final_raw_indices


def compressed_index_decode_dense_topk_fp8(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: CompressedIndexerMetadata,
    page_size: int = COMPRESSED_INDEX_PAGE_SIZE,
    topk: int = 512,
    expected_num_q_heads: int | None = None,
    workspace=None,
    out_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score the full paged C4 row and select top-k with the dense top-k core."""

    page_size = int(page_size)
    if page_size != COMPRESSED_INDEX_PAGE_SIZE:
        raise ValueError(
            f"compressed indexer currently supports page_size={COMPRESSED_INDEX_PAGE_SIZE}, "
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
        raise NotImplementedError("compressed index dense top-k requires CUDA")
    if workspace is None:
        raise RuntimeError("compressed index dense top-k requires a b12x workspace")
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
                "compressed index dense top-k requires a CUDA-graph workspace when "
                "scoring a compact view of a padded page table"
            )
        scorer_metadata = CompressedIndexerMetadata(
            real_page_table=compact_page_table,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            schedule_metadata=metadata.schedule_metadata,
            expected_num_q_heads=metadata.expected_num_q_heads,
        )

    active_width = None
    if workspace is not None and hasattr(workspace, "get_paged_indexer_active_width_cap"):
        active_width = workspace.get_paged_indexer_active_width_cap()

    logits = compressed_index_decode_logits_fp8(
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


pack_compressed_index_k_cache_reference = pack_index_k_cache_reference
unpack_compressed_index_k_cache_reference = unpack_index_k_cache_reference
compressed_index_logits_reference = paged_decode_logits_reference


__all__ = [
    "INDEX_HEAD_DIM",
    "COMPRESSED_INDEX_PAGE_SIZE",
    "CompressedIndexerMetadata",
    "make_compressed_indexer_contract_phantoms",
    "pack_compressed_index_k_cache_reference",
    "compressed_index_decode_dense_topk_fp8",
    "compressed_index_decode_logits_fp8",
    "compressed_index_decode_supertile_topk_fp8",
    "compressed_index_logits_reference",
    "prepare_compressed_indexer_metadata",
    "resolve_local_num_q_heads",
    "resolve_replicated_num_q_heads",
    "unpack_compressed_index_k_cache_reference",
]
