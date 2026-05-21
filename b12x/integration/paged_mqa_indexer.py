"""Generic paged-MQA indexer integration surface.

This module exposes the paged FP8 MQA scorer behind algorithmic names.  The
implementation is shared with the NSA indexer path, but callers should use this
surface when they only need paged indexer logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

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


INDEX_HEAD_DIM = 128
PAGED_MQA_INDEX_PAGE_SIZE = 64


@dataclass(frozen=True)
class PagedMQAIndexerMetadata:
    """Metadata for paged FP8 MQA indexer logits.

    ``expected_num_q_heads`` is optional for the generic path, but SGLang-style
    tensor-parallel integrations should set it to the true local head count.
    That catches accidental full-global-head padding at the b12x boundary.
    """

    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    paged_mqa_schedule_metadata: torch.Tensor | None = None
    expected_num_q_heads: int | None = None


def resolve_local_num_q_heads(
    *,
    global_num_q_heads: int,
    tensor_parallel_size: int,
) -> int:
    """Return the TP-local query/index head count with divisibility checks."""

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
            f"q_fp8 must use the TP-local head count {int(expected_heads)}, got "
            f"{q_fp8.shape[1]}; do not pass full global-head padded indexer tensors"
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
    """Compute paged FP8 MQA indexer logits with a TP-local head contract."""

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
    "paged_mqa_index_decode_logits_fp8",
    "paged_mqa_index_logits_reference",
    "prepare_paged_mqa_indexer_metadata",
    "resolve_local_num_q_heads",
    "unpack_paged_mqa_index_k_cache_reference",
    "uses_paged_mqa_schedule_metadata",
]
