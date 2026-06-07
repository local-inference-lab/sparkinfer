"""NSA indexer API for paged and contiguous logits contracts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import torch

from .kernel import (
    PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT,
    _should_use_schedule_multi_row_kernel,
    clear_indexer_kernel_cache,
    _should_use_schedule_single_row_kernel,
    run_paged_logits_kernel,
    supports_paged_logits_kernel,
)
from .contiguous_kernel import (
    _PREFILL512_BLOCK_K,
    _PREFILL512_BLOCK_Q,
    _PREFILL_BLOCK_K,
    _PREFILL_BLOCK_Q,
    build_indexer_contiguous_logits_kernel_binding,
    resolve_contiguous_prefill_block_k,
    run_contiguous_logits_kernel,
    supports_contiguous_logits_kernel,
)
from .reference import contiguous_logits_reference
from .schedule_metadata import (
    build_paged_mqa_schedule_metadata_torch,
    build_paged_mqa_schedule_metadata_triton,
)
from .tiled_topk import (
    _resolve_supertile_k,
    clear_tiled_topk_kernel_cache,
    run_tiled_topk,
)
from .persistent_topk import clear_persistent_topk2048_kernel_cache


_INDEX_HEAD_DIM = 128
_VALIDATE_PAGE_IDS = bool(int(os.getenv("B12X_NSA_VALIDATE_PAGE_IDS", "0")))


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


@dataclass(frozen=True)
class IndexerPagedDecodeMetadata:
    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    paged_mqa_schedule_metadata: torch.Tensor | None = None


@dataclass(frozen=True)
class IndexerContiguousMetadata:
    k_start: torch.Tensor
    k_end: torch.Tensor


def build_paged_mqa_schedule_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build paged-MQA schedule metadata on the input device."""

    if context_lens.ndim not in (1, 2):
        raise ValueError(
            f"context_lens must be rank-1 or rank-2, got {tuple(context_lens.shape)}"
        )
    if context_lens.ndim == 2 and context_lens.shape[1] == 0:
        raise ValueError("context_lens rank-2 input must have a non-empty trailing dimension")
    if context_lens.dtype != torch.int32:
        raise ValueError(
            f"context_lens must have dtype torch.int32, got {context_lens.dtype}"
        )
    if not context_lens.is_contiguous():
        raise ValueError("context_lens must be contiguous")
    if block_kv <= 0:
        raise ValueError(f"block_kv must be positive, got {block_kv}")
    if out is not None:
        if out.ndim != 2 or out.shape[1] != 2:
            raise ValueError(f"out must have shape (num_sms + 1, 2), got {tuple(out.shape)}")
        if out.dtype != torch.int32:
            raise ValueError(f"out must have dtype torch.int32, got {out.dtype}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.device != context_lens.device:
            raise ValueError(
                f"out device {out.device} does not match context_lens device {context_lens.device}"
            )
        if num_sms is None:
            num_sms = out.shape[0] - 1
    if num_sms is None:
        if context_lens.device.type == "cuda":
            num_sms = torch.cuda.get_device_properties(context_lens.device).multi_processor_count
        else:
            num_sms = 1
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    if out is not None and out.shape[0] != num_sms + 1:
        raise ValueError(
            f"out leading dimension {out.shape[0]} does not match num_sms + 1 ({num_sms + 1})"
        )

    schedule = out
    if schedule is None:
        schedule = torch.empty(
            (num_sms + 1, 2),
            dtype=torch.int32,
            device=context_lens.device,
        )
    builder = (
        build_paged_mqa_schedule_metadata_triton
        if context_lens.device.type == "cuda"
        else build_paged_mqa_schedule_metadata_torch
    )
    return builder(
        context_lens,
        block_kv=block_kv,
        num_sms=num_sms,
        pages_per_split=PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT,
        out=schedule,
    )


def clear_indexer_caches() -> None:
    """Clear any cached NSA indexer runtime state."""
    clear_indexer_kernel_cache()
    clear_tiled_topk_kernel_cache()
    clear_persistent_topk2048_kernel_cache()
    _cached_width_cap_tensor.cache_clear()


def uses_paged_mqa_schedule(
    *,
    q_rows: int,
    max_pages: int,
) -> bool:
    """Return whether decode should use a schedule-driven scorer path."""
    return _should_use_schedule_single_row_kernel(
        q_rows=q_rows,
        max_pages=max_pages,
    ) or _should_use_schedule_multi_row_kernel(
        q_rows=q_rows,
        max_pages=max_pages,
    )


def _normalize_weights(
    weights: torch.Tensor,
    *,
    q_rows: int,
    num_heads: int,
    require_float32: bool = False,
) -> torch.Tensor:
    if weights.ndim == 3:
        if weights.shape[2] != 1:
            raise ValueError(
                f"weights rank-3 input must have trailing dimension 1, got {tuple(weights.shape)}"
            )
        weights = weights.squeeze(2)
    if weights.ndim != 2:
        raise ValueError(f"weights must be rank-2 or rank-3, got {tuple(weights.shape)}")
    if weights.shape != (q_rows, num_heads):
        raise ValueError(f"weights shape must be {(q_rows, num_heads)}, got {tuple(weights.shape)}")
    if require_float32 and weights.dtype != torch.float32:
        raise ValueError(
            f"strict indexer contiguous requires torch.float32 weights, got {weights.dtype}"
        )
    return weights.to(torch.float32)


@lru_cache(maxsize=64)
def _cached_width_cap_tensor(
    width: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    return torch.tensor([width], dtype=torch.int32, device=torch.device(device_type, device_index))


def _make_active_width_tensor(
    *,
    seqlens_per_query: torch.Tensor,
    width: int,
) -> torch.Tensor:
    if seqlens_per_query.ndim != 1:
        raise ValueError(
            "seqlens_per_query must be rank-1 when computing active width, got "
            f"{tuple(seqlens_per_query.shape)}"
        )
    active_width = seqlens_per_query.amax().reshape(1)
    if _is_cuda_graph_capture_active(seqlens_per_query.device):
        return active_width.clamp_(min=0, max=int(width))
    width_cap = _cached_width_cap_tensor(
        int(width),
        seqlens_per_query.device.type,
        seqlens_per_query.device.index,
    )
    return torch.minimum(active_width, width_cap)


def _validate_paged_decode_inputs(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    paged_mqa_schedule_metadata: torch.Tensor | None,
) -> torch.Tensor:
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(f"q_fp8 head_dim must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}")
    if real_page_table.ndim != 2:
        raise ValueError(f"real_page_table must be rank-2, got {tuple(real_page_table.shape)}")
    if real_page_table.dtype != torch.int32:
        raise ValueError(
            f"real_page_table must have dtype torch.int32, got {real_page_table.dtype}"
        )
    if cache_seqlens_int32.ndim != 1:
        raise ValueError(
            "cache_seqlens_int32 must be rank-1, got "
            f"{tuple(cache_seqlens_int32.shape)}"
        )
    if real_page_table.shape[0] != cache_seqlens_int32.shape[0]:
        raise ValueError(
            f"real_page_table rows {real_page_table.shape[0]} do not match "
            f"cache_seqlens rows {cache_seqlens_int32.shape[0]}"
        )
    if real_page_table.shape[0] > q_fp8.shape[0]:
        raise ValueError(
            f"real_page_table rows {real_page_table.shape[0]} exceed q rows {q_fp8.shape[0]}"
        )
    if real_page_table.device != q_fp8.device:
        raise ValueError(
            f"real_page_table device {real_page_table.device} does not match q_fp8 device {q_fp8.device}"
        )
    if cache_seqlens_int32.device != q_fp8.device:
        raise ValueError(
            f"cache_seqlens_int32 device {cache_seqlens_int32.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )
    if paged_mqa_schedule_metadata is not None:
        if paged_mqa_schedule_metadata.ndim != 2:
            raise ValueError(
                "paged_mqa_schedule_metadata must be rank-2, got "
                f"{tuple(paged_mqa_schedule_metadata.shape)}"
            )
        if paged_mqa_schedule_metadata.shape[1] != 2:
            raise ValueError(
                "paged_mqa_schedule_metadata trailing dimension must be 2, got "
                f"{tuple(paged_mqa_schedule_metadata.shape)}"
            )
        if paged_mqa_schedule_metadata.shape[0] < 2:
            raise ValueError(
                "paged_mqa_schedule_metadata must have at least two rows, got "
                f"{tuple(paged_mqa_schedule_metadata.shape)}"
            )
        if paged_mqa_schedule_metadata.dtype != torch.int32:
            raise ValueError(
                "paged_mqa_schedule_metadata must have dtype torch.int32, got "
                f"{paged_mqa_schedule_metadata.dtype}"
            )
        if not paged_mqa_schedule_metadata.is_contiguous():
            raise ValueError("paged_mqa_schedule_metadata must be contiguous")
        if paged_mqa_schedule_metadata.device != q_fp8.device:
            raise ValueError(
                "paged_mqa_schedule_metadata device "
                f"{paged_mqa_schedule_metadata.device} does not match q_fp8 device {q_fp8.device}"
            )
    return _normalize_weights(weights, q_rows=q_fp8.shape[0], num_heads=q_fp8.shape[1])


def paged_decode_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: IndexerPagedDecodeMetadata | None = None,
    page_size: int = 64,
    preinitialize_invalid_logits: bool = True,
    active_width_override: torch.Tensor | None = None,
    binding=None,
) -> torch.Tensor:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("active_width_override", active_width_override),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "paged indexer binding owns metadata, scratch, and active width; "
                f"do not also pass {', '.join(extras)}"
            )
        metadata = binding.metadata
        active_width_override = binding.active_width
    if metadata is None:
        raise TypeError("paged_decode_logits requires metadata or binding")

    weights_f = _validate_paged_decode_inputs(
        q_fp8=q_fp8,
        weights=weights,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        paged_mqa_schedule_metadata=metadata.paged_mqa_schedule_metadata,
    )

    valid_q_rows = metadata.real_page_table.shape[0]
    full_q_rows = q_fp8.shape[0]
    width_tokens = metadata.real_page_table.shape[1] * page_size
    if valid_q_rows == 0 or width_tokens == 0:
        return torch.full(
            (full_q_rows, width_tokens),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )

    seqlens_valid = metadata.cache_seqlens_int32.contiguous()
    if active_width_override is None:
        active_width = _make_active_width_tensor(seqlens_per_query=seqlens_valid, width=width_tokens)
    else:
        if active_width_override.shape != (1,):
            raise ValueError(
                f"active_width_override must have shape (1,), got {tuple(active_width_override.shape)}"
            )
        if active_width_override.dtype != torch.int32:
            raise ValueError(
                "active_width_override must have dtype torch.int32, got "
                f"{active_width_override.dtype}"
            )
        if active_width_override.device != q_fp8.device:
            raise ValueError(
                "active_width_override device "
                f"{active_width_override.device} does not match q_fp8 device {q_fp8.device}"
            )
        active_width = active_width_override

    validate_page_ids = q_fp8.device.type != "cuda" or (
        _VALIDATE_PAGE_IDS and not _is_cuda_graph_capture_active(q_fp8.device)
    )
    if validate_page_ids:
        active_width_host = min(width_tokens, int(active_width.item()))
        if active_width_host > 0:
            max_page_capacity = index_k_cache.shape[0]
            positions = torch.arange(
                active_width_host,
                dtype=torch.int32,
                device=q_fp8.device,
            ).unsqueeze(0)
            page_cols = torch.div(positions, page_size, rounding_mode="floor").to(torch.long)
            page_cols = page_cols.expand(valid_q_rows, -1)
            candidate_pages = metadata.real_page_table.gather(1, page_cols)
            candidate_valid_mask = (positions < seqlens_valid.unsqueeze(1)) & (candidate_pages >= 0)
            overflow_mask = candidate_valid_mask & (candidate_pages >= max_page_capacity)
            if torch.any(overflow_mask):
                bad = int(candidate_pages[overflow_mask].max().item())
                raise ValueError(
                    f"real_page_table page id {bad} exceeds index_k_cache page capacity {max_page_capacity}"
                )

    if not supports_paged_logits_kernel(
        q_fp8=q_fp8[:valid_q_rows],
        weights=weights_f[:valid_q_rows],
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        seqlens_per_query=seqlens_valid,
        page_size=page_size,
    ):
        raise NotImplementedError(
            "B12X sparse NSA paged logits requires the production CUDA FP8 "
            "kernel contract; refusing to run the reference fallback. "
            f"q_fp8 shape={tuple(q_fp8.shape)} dtype={q_fp8.dtype} "
            f"device={q_fp8.device}, weights shape={tuple(weights_f.shape)} "
            f"dtype={weights_f.dtype}, index_k_cache shape={tuple(index_k_cache.shape)} "
            f"dtype={index_k_cache.dtype}, real_page_table shape="
            f"{tuple(metadata.real_page_table.shape)} dtype="
            f"{metadata.real_page_table.dtype}, cache_seqlens shape="
            f"{tuple(seqlens_valid.shape)} dtype={seqlens_valid.dtype}, "
            f"page_size={page_size}"
        )

    schedule_metadata = None
    if uses_paged_mqa_schedule(
        q_rows=valid_q_rows,
        max_pages=int(metadata.real_page_table.shape[1]),
    ):
        schedule_metadata = metadata.paged_mqa_schedule_metadata
        if schedule_metadata is None:
            if _is_cuda_graph_capture_active(q_fp8.device):
                raise ValueError(
                    "paged_mqa_schedule_metadata must be precomputed before CUDA graph capture "
                    "for the scheduled decode path"
                )
            schedule_metadata = build_paged_mqa_schedule_metadata(seqlens_valid, page_size)
    logits_valid = run_paged_logits_kernel(
        q_fp8=q_fp8[:valid_q_rows],
        weights=weights_f[:valid_q_rows],
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        seqlens_per_query=seqlens_valid,
        schedule_metadata=schedule_metadata,
        active_width=active_width,
        page_size=page_size,
        preinitialize_invalid_logits=preinitialize_invalid_logits,
    )
    if valid_q_rows == full_q_rows:
        return logits_valid

    logits = torch.full(
        (full_q_rows, width_tokens),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    logits[:valid_q_rows].copy_(logits_valid)
    return logits


def contiguous_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    metadata: IndexerContiguousMetadata | None = None,
    preinitialize_invalid_logits: bool = True,
    tile_logits: torch.Tensor | None = None,
    binding=None,
) -> torch.Tensor:
    strict_binding = False
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("tile_logits", tile_logits),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "indexer contiguous binding owns metadata, scratch, "
                f"and tile logits; do not also pass {', '.join(extras)}"
            )
        strict_binding = bool(getattr(binding, "strict", False))
        metadata = binding.metadata
        tile_logits = binding.tile_logits
    if metadata is None:
        raise TypeError("contiguous_logits requires metadata or binding")
    k_start = metadata.k_start
    k_end = metadata.k_end
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(f"q_fp8 head_dim must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}")
    _normalize_weights(weights, q_rows=q_fp8.shape[0], num_heads=q_fp8.shape[1])
    if k_start.ndim != 1 or k_end.ndim != 1:
        raise ValueError(
            f"k_start and k_end must be rank-1, got {tuple(k_start.shape)} and {tuple(k_end.shape)}"
        )
    if k_start.shape != k_end.shape:
        raise ValueError(
            f"k_start and k_end must have the same shape, got {tuple(k_start.shape)} vs {tuple(k_end.shape)}"
        )
    if k_start.device != q_fp8.device or k_end.device != q_fp8.device:
        raise ValueError("k_start and k_end must be on the same device as q_fp8")

    weights_f = _normalize_weights(
        weights,
        q_rows=q_fp8.shape[0],
        num_heads=q_fp8.shape[1],
        require_float32=strict_binding,
    )
    k_quant, k_scale = kv_fp8
    if supports_contiguous_logits_kernel(
        q_fp8=q_fp8,
        weights=weights_f,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    ):
        result = run_contiguous_logits_kernel(
            q_fp8=q_fp8,
            weights=weights_f,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
            preinitialize_invalid_logits=preinitialize_invalid_logits,
            tile_logits=tile_logits,
        )
        return result

    return contiguous_logits_reference(
        q_fp8=q_fp8,
        weights=weights_f,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )


def _reference_topk_indices_from_logits(
    logits: torch.Tensor,
    *,
    topk: int,
    output_values: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    topk = int(topk)
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    num_rows = int(logits.shape[0])
    result = torch.full((num_rows, topk), -1, dtype=torch.int32, device=logits.device)
    values = torch.full((num_rows, topk), float("-inf"), dtype=torch.float32, device=logits.device)
    gather_k = min(topk, int(logits.shape[1]))
    if gather_k:
        topk_pos = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :gather_k]
        topk_values = torch.gather(logits, 1, topk_pos)
        result[:, :gather_k] = torch.where(
            torch.isfinite(topk_values),
            topk_pos.to(torch.int32),
            torch.full_like(topk_pos, -1, dtype=torch.int32),
        )
        values[:, :gather_k] = topk_values

    if output_indices is not None:
        if output_indices.dtype != torch.int32:
            raise ValueError(f"output_indices must have dtype torch.int32, got {output_indices.dtype}")
        if output_indices.device != logits.device:
            raise ValueError("output_indices device must match logits")
        if output_indices.ndim != 2 or output_indices.shape[0] < num_rows or output_indices.shape[1] < topk:
            raise ValueError(
                f"output_indices must have shape at least ({num_rows}, {topk}), got {tuple(output_indices.shape)}"
            )
        output_indices[:num_rows, :topk].copy_(result)
        result = output_indices[:num_rows, :topk]

    if output_values is not None:
        if output_values.dtype != torch.float32:
            raise ValueError(f"output_values must have dtype torch.float32, got {output_values.dtype}")
        if output_values.device != logits.device:
            raise ValueError("output_values device must match logits")
        if output_values.ndim != 2 or output_values.shape[0] < num_rows or output_values.shape[1] < topk:
            raise ValueError(
                f"output_values must have shape at least ({num_rows}, {topk}), got {tuple(output_values.shape)}"
            )
        output_values[:num_rows, :topk].copy_(values)

    return result


def contiguous_tiled_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    metadata: IndexerContiguousMetadata | None = None,
    topk: int | None = None,
    tile_logits: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    output_values: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
    candidate_values: torch.Tensor | None = None,
    candidate_indices: torch.Tensor | None = None,
    merge_positions: torch.Tensor | None = None,
    supertile_k: int | None = None,
    binding=None,
) -> torch.Tensor:
    """Run the prefill NSA scorer in K-supertiles and consume each tile with tiled topk."""

    strict_binding = False
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("tile_logits", tile_logits),
                ("lengths", lengths),
                ("output_values", output_values),
                ("output_indices", output_indices),
                ("candidate_values", candidate_values),
                ("candidate_indices", candidate_indices),
                ("merge_positions", merge_positions),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "indexer contiguous binding owns metadata, scratch, "
                "and top-k scratch buffers; do not also pass "
                f"{', '.join(extras)}"
            )
        if topk is None:
            topk = binding.topk
        elif binding.topk is not None and int(topk) != int(binding.topk):
            raise ValueError(
                f"topk {int(topk)} does not match bound topk {int(binding.topk)}"
            )
        strict_binding = bool(getattr(binding, "strict", False))
        metadata = binding.metadata
        tile_logits = binding.tile_logits
        lengths = binding.lengths
        output_values = binding.output_values
        output_indices = binding.output_indices
        candidate_values = binding.candidate_values
        candidate_indices = binding.candidate_indices
        merge_positions = binding.merge_positions
        if binding.supertile_k is not None:
            supertile_k = int(binding.supertile_k)
    if metadata is None:
        raise TypeError("contiguous_tiled_topk requires metadata or binding")
    if topk is None:
        raise TypeError("contiguous_tiled_topk requires topk or a binding with topk")
    topk = int(topk)
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    k_start = metadata.k_start
    k_end = metadata.k_end
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if k_start.ndim != 1 or k_end.ndim != 1 or k_start.shape != k_end.shape:
        raise ValueError("tiled topk requires matching rank-1 k_start and k_end tensors")
    weights_f = _normalize_weights(weights, q_rows=q_fp8.shape[0], num_heads=q_fp8.shape[1])
    k_quant, k_scale = kv_fp8
    if not supports_contiguous_logits_kernel(
        q_fp8=q_fp8,
        weights=weights_f,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    ):
        if strict_binding:
            raise RuntimeError(
                "strict indexer contiguous binding requires the CUDA FP8 scorer "
                "contract; reference logits fallback is disabled"
            )
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] < int(k_start.shape[0]):
                raise ValueError(
                    f"lengths must have shape at least ({int(k_start.shape[0])},), got {tuple(lengths.shape)}"
                )
            if lengths.dtype != torch.int32:
                raise ValueError(f"lengths must have dtype torch.int32, got {lengths.dtype}")
            if lengths.device != q_fp8.device:
                raise ValueError(f"lengths device {lengths.device} does not match q_fp8 device {q_fp8.device}")
            torch.sub(k_end, k_start, out=lengths[: int(k_start.shape[0])])
        logits = contiguous_logits_reference(
            q_fp8=q_fp8,
            weights=weights_f,
            kv_fp8=kv_fp8,
            k_start=k_start,
            k_end=k_end,
        )
        return _reference_topk_indices_from_logits(
            logits[: int(k_start.shape[0])],
            topk=topk,
            output_values=output_values,
            output_indices=output_indices,
        )
    prefill_block_k = (
        int(binding.prefill_block_k)
        if binding is not None and binding.prefill_block_k is not None
        else resolve_contiguous_prefill_block_k(
            valid_q_rows=int(k_start.shape[0]),
            k_rows=int(k_quant.shape[0]),
            num_heads=int(q_fp8.shape[1]),
        )
    )
    if prefill_block_k is None:
        # This API explicitly requests tiled logits for immediate tiled top-k.
        # The decode scorer does not produce that layout, so force the standard
        # prefill scorer for small q batches instead of failing.
        prefill_block_k = _PREFILL_BLOCK_K
    block_q = _PREFILL512_BLOCK_Q if prefill_block_k == _PREFILL512_BLOCK_K else _PREFILL_BLOCK_Q

    num_q_rows = int(k_start.shape[0])
    num_q_tiles = (num_q_rows + block_q - 1) // block_q
    num_k_tiles = (int(k_quant.shape[0]) + prefill_block_k - 1) // prefill_block_k
    tile_size = block_q * prefill_block_k
    resolved_supertile_k = _resolve_supertile_k(supertile_k, block_k=prefill_block_k)
    supertile_tiles = max(1, resolved_supertile_k // prefill_block_k)
    num_chunks = (num_k_tiles + supertile_tiles - 1) // supertile_tiles
    max_chunk_tiles = min(supertile_tiles, num_k_tiles)
    chunk_tile_elements = num_q_tiles * max_chunk_tiles * tile_size

    if tile_logits is None:
        if strict_binding:
            raise RuntimeError("strict indexer contiguous binding is missing tile_logits")
        tile_logits = torch.empty(
            (chunk_tile_elements,),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    elif int(tile_logits.numel()) < chunk_tile_elements:
        raise ValueError(
            f"tile_logits has {int(tile_logits.numel())} elements, expected at least "
            f"{chunk_tile_elements} for the largest K-supertile"
        )

    if lengths is None:
        if strict_binding:
            raise RuntimeError("strict indexer contiguous binding is missing lengths")
        global_lengths = (k_end - k_start).contiguous()
    else:
        if lengths.ndim != 1 or lengths.shape[0] < num_q_rows:
            raise ValueError(
                f"lengths must have shape at least ({num_q_rows},), got {tuple(lengths.shape)}"
            )
        if lengths.dtype != torch.int32:
            raise ValueError(f"lengths must have dtype torch.int32, got {lengths.dtype}")
        if lengths.device != q_fp8.device:
            raise ValueError(f"lengths device {lengths.device} does not match q_fp8 device {q_fp8.device}")
        if not lengths.is_contiguous():
            raise ValueError("lengths must be contiguous")
        global_lengths = lengths[:num_q_rows]
        torch.sub(k_end, k_start, out=global_lengths)
    if strict_binding and (output_values is None or output_indices is None):
        raise RuntimeError(
            "strict indexer contiguous binding is missing output top-k buffers"
        )

    def _run_contiguous_scorer(
        *,
        tile_k_offset: int,
        tile_num_k_tiles: int,
    ) -> None:
        if not strict_binding:
            run_contiguous_logits_kernel(
                q_fp8=q_fp8,
                weights=weights_f,
                k_quant=k_quant,
                k_scale=k_scale,
                k_start=k_start,
                k_end=k_end,
                preinitialize_invalid_logits=True,
                tile_logits=tile_logits,
                tile_k_offset=tile_k_offset,
                tile_num_k_tiles=tile_num_k_tiles,
                prefill_block_k=prefill_block_k,
            )
            return

        if binding is None:
            raise RuntimeError("strict indexer contiguous path requires a binding")
        scratch = binding.scratch
        if tile_logits is None:
            raise RuntimeError("strict indexer contiguous binding is missing tile logits")
        if not hasattr(scratch, "prepare_k_padding"):
            raise RuntimeError(
                "strict indexer contiguous binding requires plan-owned scratch"
            )
        if not q_fp8.is_contiguous():
            raise ValueError("strict indexer contiguous requires contiguous q_fp8")
        if not weights_f.is_contiguous():
            raise ValueError("strict indexer contiguous requires contiguous weights")
        if not k_quant.is_contiguous() or not k_scale.is_contiguous():
            raise ValueError("strict indexer contiguous requires contiguous K tensors")
        scratch.prepare_k_padding(k_rows=int(k_quant.shape[0]))
        scratch_k_quant = scratch.k_quant
        scratch_k_scale = scratch.k_scale
        if k_quant.data_ptr() != scratch_k_quant.data_ptr():
            raise ValueError("strict indexer contiguous K values must be a scratch prefix")
        if k_scale.data_ptr() != scratch_k_scale.data_ptr():
            raise ValueError("strict indexer contiguous K scales must be a scratch prefix")
        q_bytes = q_fp8.view(torch.uint8)
        q_u32 = q_bytes.view(torch.uint32).view(
            int(q_fp8.shape[0]),
            int(q_fp8.shape[1]),
            _INDEX_HEAD_DIM // 4,
        )
        kernel_binding = build_indexer_contiguous_logits_kernel_binding(
            q_fp8=q_fp8,
            weights=weights_f,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
            preinitialize_invalid_logits=True,
            tile_logits=tile_logits,
            tile_k_offset=tile_k_offset,
            tile_num_k_tiles=tile_num_k_tiles,
            prefill_block_k=prefill_block_k,
            q_u32=q_u32,
            q_bytes=q_bytes,
            weights_kernel=weights_f,
            k_quant_bytes=scratch_k_quant.view(torch.uint8),
            k_scale_kernel=scratch_k_scale,
            k_start_kernel=k_start,
            k_end_kernel=k_end,
            out_kernel=scratch.dummy_logits,
            out_view=scratch.dummy_logits,
            k_tma_desc_ptrs=scratch.k_tma_desc_ptrs,
            k_tma_prefill_desc_ptrs=scratch.k_tma_prefill_desc_ptrs,
        )
        run_contiguous_logits_kernel(binding=kernel_binding)

    if num_chunks <= 1:
        # Dead tiles (entirely out of causal/length range) are left UNWRITTEN by
        # the tiled-output contiguous kernel (it `pass`es, trusting run_tiled_topk's
        # k_start/k_end mask). Pre-clear to -inf so any stale value in those slots
        # of the (reused) scratch can never win the tiled top-k.
        tile_logits[:chunk_tile_elements].fill_(float("-inf"))
        _run_contiguous_scorer(
            tile_k_offset=0,
            tile_num_k_tiles=num_k_tiles,
        )
        _, topk_indices = run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            lengths=global_lengths,
            topk=topk,
            block_q=block_q,
            block_k=prefill_block_k,
            output_values=output_values,
            output_indices=output_indices,
            num_k_tiles=num_k_tiles,
        )
        return topk_indices

    # Streaming fold over K-supertiles: each chunk folds the previous chunk's running
    # top-k (carry) into its own radix selection, so the final chunk's output is the
    # exact global top-k. The reused candidate_values/candidate_indices buffers serve
    # as a (2, M, topk) carry double-buffer (read prev half, write next half); the
    # final chunk writes the user output. No (num_chunks, ...) slab, no merge.
    if (candidate_values is None) != (candidate_indices is None):
        raise ValueError("candidate_values and candidate_indices must be provided together")
    if candidate_values is None:
        if strict_binding:
            raise RuntimeError(
                "strict indexer contiguous binding is missing carry buffers"
            )
        candidate_values = torch.empty(
            (2, num_q_rows, topk),
            dtype=torch.float32,
            device=q_fp8.device,
        )
        candidate_indices = torch.empty(
            (2, num_q_rows, topk),
            dtype=torch.int32,
            device=q_fp8.device,
        )
    else:
        assert candidate_indices is not None
        if candidate_values.ndim != 3 or candidate_indices.ndim != 3:
            raise ValueError(
                "carry buffers must have shape at least "
                f"(2, {num_q_rows}, {topk})"
            )
        if candidate_values.shape[0] < 2 or candidate_values.shape[1] < num_q_rows:
            raise ValueError(
                "candidate_values shape "
                f"{tuple(candidate_values.shape)} is smaller than required "
                f"(2, {num_q_rows}, {topk})"
            )
        if candidate_indices.shape[0] < 2 or candidate_indices.shape[1] < num_q_rows:
            raise ValueError(
                "candidate_indices shape "
                f"{tuple(candidate_indices.shape)} is smaller than required "
                f"(2, {num_q_rows}, {topk})"
            )
        if candidate_values.shape[2] != topk or candidate_indices.shape[2] != topk:
            raise ValueError(
                "carry buffer top-k dimension must match requested topk "
                f"{topk}, got {candidate_values.shape[2]} and {candidate_indices.shape[2]}"
            )
        if candidate_values.dtype != torch.float32:
            raise ValueError(f"candidate_values must have dtype torch.float32, got {candidate_values.dtype}")
        if candidate_indices.dtype != torch.int32:
            raise ValueError(f"candidate_indices must have dtype torch.int32, got {candidate_indices.dtype}")
        if candidate_values.device != q_fp8.device or candidate_indices.device != q_fp8.device:
            raise ValueError("carry buffer devices must match q_fp8")
    carry_buf_values = candidate_values[:2, :num_q_rows, :]
    carry_buf_indices = candidate_indices[:2, :num_q_rows, :]

    topk_indices = output_indices
    for chunk_idx in range(num_chunks):
        chunk_tile_begin = chunk_idx * supertile_tiles
        chunk_tile_end = min(chunk_tile_begin + supertile_tiles, num_k_tiles)
        chunk_tiles = chunk_tile_end - chunk_tile_begin
        chunk_start = chunk_tile_begin * prefill_block_k
        chunk_rows = chunk_tiles * prefill_block_k
        # tile_logits is reused across chunks; dead tiles are not rewritten by the
        # kernel, so a stale logit from a previous chunk at the same offset could
        # otherwise survive into this chunk's tiled top-k. Pre-clear to -inf.
        tile_logits[: num_q_tiles * chunk_tiles * tile_size].fill_(float("-inf"))
        _run_contiguous_scorer(
            tile_k_offset=chunk_tile_begin,
            tile_num_k_tiles=chunk_tiles,
        )
        is_first = chunk_idx == 0
        is_last = chunk_idx == num_chunks - 1
        carry_values = carry_buf_values[(chunk_idx - 1) % 2]
        carry_indices = carry_buf_indices[(chunk_idx - 1) % 2]
        out_v = output_values if is_last else carry_buf_values[chunk_idx % 2]
        out_i = output_indices if is_last else carry_buf_indices[chunk_idx % 2]
        _, topk_indices = run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            lengths=global_lengths,
            topk=topk,
            block_q=block_q,
            block_k=prefill_block_k,
            output_values=out_v,
            output_indices=out_i,
            num_k_tiles=chunk_tiles,
            input_index_offset=chunk_start,
            input_extent=chunk_rows,
            output_index_offset=chunk_start,
            carry_values=carry_values,
            carry_indices=carry_indices,
            is_first=is_first,
        )
    return topk_indices
