"""Device-side decode graph replay helpers for the paged attention backend."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

_DECODE_BLOCK_CHUNKS = 128
_DECODE_BLOCK_PAGES = 128


@triton.jit
def build_decode_graph_page_table_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_ptr,
    active_max_pages_ptr,
    req_to_token_row_stride,
    page_table_row_stride,
    PAGE_SIZE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    page_block_idx = tl.program_id(axis=1)

    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx).to(tl.int64)
    active_max_pages = tl.load(active_max_pages_ptr).to(tl.int32)
    page_offsets = page_block_idx * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < active_max_pages
    flat_token_offsets = req_pool_idx * req_to_token_row_stride + page_offsets.to(tl.int64) * PAGE_SIZE
    token_indices = tl.load(req_to_token_ptr + flat_token_offsets, mask=page_mask, other=0)
    tl.store(
        page_table_ptr + req_idx * page_table_row_stride + page_offsets,
        (token_indices // PAGE_SIZE).to(tl.int32),
        mask=page_mask,
    )


@triton.jit
def update_decode_graph_metadata_triton(
    cache_seqlens_ptr,
    merge_indptr_ptr,
    block_valid_mask_ptr,
    kv_window_start_tokens_ptr,
    chunk_pages_ptr,
    max_chunks_per_req,
    PAGE_SIZE: tl.constexpr,
    WINDOW_PAGE_SPAN: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    chunk_block_idx = tl.program_id(axis=1)

    cache_len = tl.load(cache_seqlens_ptr + req_idx).to(tl.int32)
    chunk_pages = tl.load(chunk_pages_ptr).to(tl.int32)
    num_pages = tl.maximum((cache_len + (PAGE_SIZE - 1)) // PAGE_SIZE, 1)
    effective_pages = num_pages
    if WINDOW_PAGE_SPAN > 0:
        effective_pages = tl.minimum(num_pages, WINDOW_PAGE_SPAN)
    window_start_page = num_pages - effective_pages
    tl.store(kv_window_start_tokens_ptr + req_idx, window_start_page * PAGE_SIZE)
    num_chunks = (effective_pages + chunk_pages - 1) // chunk_pages

    tl.store(merge_indptr_ptr + req_idx + 1, num_chunks)

    chunk_offsets = chunk_block_idx * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunk_offsets < max_chunks_per_req
    is_active = chunk_offsets < num_chunks
    tl.store(
        block_valid_mask_ptr + req_idx * max_chunks_per_req + chunk_offsets,
        is_active.to(tl.int32),
        mask=chunk_mask,
    )


@triton.jit
def update_regular_decode_graph_metadata_triton(
    cache_seqlens_ptr,
    merge_indptr_ptr,
    kv_window_start_tokens_ptr,
    chunk_pages_ptr,
    PAGE_SIZE: tl.constexpr,
    WINDOW_PAGE_SPAN: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)

    cache_len = tl.load(cache_seqlens_ptr + req_idx).to(tl.int32)
    chunk_pages = tl.load(chunk_pages_ptr).to(tl.int32)
    kv_chunk_size = chunk_pages * PAGE_SIZE
    num_pages = tl.maximum((cache_len + (PAGE_SIZE - 1)) // PAGE_SIZE, 1)
    effective_pages = num_pages
    if WINDOW_PAGE_SPAN > 0:
        effective_pages = tl.minimum(num_pages, WINDOW_PAGE_SPAN)
    window_start_page = num_pages - effective_pages
    tl.store(kv_window_start_tokens_ptr + req_idx, window_start_page * PAGE_SIZE)
    num_chunks = tl.maximum((effective_pages * PAGE_SIZE + kv_chunk_size - 1) // kv_chunk_size, 1)

    tl.store(merge_indptr_ptr + req_idx + 1, num_chunks)


def make_decode_chunk_pages_lut_tensor(
    decode_chunk_pages_lut: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not decode_chunk_pages_lut:
        raise ValueError("decode chunk-pages LUT must be non-empty")
    if any(int(chunk_pages) <= 0 for chunk_pages in decode_chunk_pages_lut):
        raise ValueError("decode chunk-pages LUT must contain only positive values")
    return torch.tensor(
        (int(decode_chunk_pages_lut[0]), *(int(chunk_pages) for chunk_pages in decode_chunk_pages_lut)),
        dtype=torch.int32,
        device=device,
    )


def summarize_decode_chunk_pages_lut(
    decode_chunk_pages_lut: Sequence[int],
) -> tuple[int, int]:
    if not decode_chunk_pages_lut:
        raise ValueError("decode chunk-pages LUT must be non-empty")
    worst_page_count = 1
    max_chunks_per_req = 1
    for page_count, chunk_pages in enumerate(decode_chunk_pages_lut, start=1):
        num_chunks = (page_count + int(chunk_pages) - 1) // int(chunk_pages)
        if num_chunks > max_chunks_per_req:
            max_chunks_per_req = num_chunks
            worst_page_count = page_count
    return int(worst_page_count), int(max_chunks_per_req)


def update_decode_graph_replay_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
) -> None:
    if req_to_token.device != page_table.device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != page_table.device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if cache_seqlens.device != page_table.device:
        raise ValueError("cache_seqlens and page_table must be on the same device")
    if decode_chunk_pages_lut.device != page_table.device:
        raise ValueError("decode_chunk_pages_lut and page_table must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if int(req_pool_indices.shape[0]) != bs:
        raise ValueError("req_pool_indices shape must match cache_seqlens batch")
    work_items_capacity = int(request_indices.shape[0])
    if work_items_capacity % bs != 0:
        raise RuntimeError("decode graph workspace request_indices shape is incompatible with the batch bucket")
    max_chunks_per_req = work_items_capacity // bs
    if max_chunks_per_req <= 0:
        raise RuntimeError("decode graph workspace must allocate at least one chunk per request")

    max_cache_pages = torch.div(
        cache_seqlens[:bs].amax() + (page_size - 1),
        page_size,
        rounding_mode="floor",
    ).clamp_(min=1, max=page_table.shape[1]).to(torch.int64)
    active_max_pages = max_cache_pages.to(torch.int32)
    if window_page_span > 0:
        effective_max_pages = torch.minimum(
            max_cache_pages,
            torch.tensor(int(window_page_span), dtype=torch.int64, device=max_cache_pages.device),
        )
    else:
        effective_max_pages = max_cache_pages
    effective_max_pages = effective_max_pages.clamp_(min=1, max=decode_chunk_pages_lut.shape[0] - 1)
    decode_chunk_pages = torch.index_select(decode_chunk_pages_lut, 0, effective_max_pages.view(1))

    page_blocks = triton.cdiv(int(page_table.shape[1]), _DECODE_BLOCK_PAGES)
    build_decode_graph_page_table_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        page_table,
        active_max_pages,
        req_to_token.stride(0),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )

    block_valid_mask.zero_()
    merge_indptr.zero_()
    chunk_blocks = triton.cdiv(max_chunks_per_req, _DECODE_BLOCK_CHUNKS)
    update_decode_graph_metadata_triton[(bs, chunk_blocks)](
        cache_seqlens,
        merge_indptr,
        block_valid_mask,
        kv_window_start_tokens,
        decode_chunk_pages,
        max_chunks_per_req,
        PAGE_SIZE=page_size,
        WINDOW_PAGE_SPAN=int(window_page_span),
        BLOCK_CHUNKS=_DECODE_BLOCK_CHUNKS,
    )
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    kv_chunk_size_ptr.copy_(decode_chunk_pages * page_size)


def update_regular_decode_graph_replay_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
) -> None:
    if req_to_token.device != page_table.device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != page_table.device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if cache_seqlens.device != page_table.device:
        raise ValueError("cache_seqlens and page_table must be on the same device")
    if decode_chunk_pages_lut.device != page_table.device:
        raise ValueError("decode_chunk_pages_lut and page_table must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if int(req_pool_indices.shape[0]) != bs:
        raise ValueError("req_pool_indices shape must match cache_seqlens batch")

    max_cache_pages = torch.div(
        cache_seqlens[:bs].amax() + (page_size - 1),
        page_size,
        rounding_mode="floor",
    ).clamp_(min=1, max=page_table.shape[1]).to(torch.int64)
    active_max_pages = max_cache_pages.to(torch.int32)
    if window_page_span > 0:
        effective_max_pages = torch.minimum(
            max_cache_pages,
            torch.tensor(int(window_page_span), dtype=torch.int64, device=max_cache_pages.device),
        )
    else:
        effective_max_pages = max_cache_pages
    effective_max_pages = effective_max_pages.clamp_(min=1, max=decode_chunk_pages_lut.shape[0] - 1)
    decode_chunk_pages = torch.index_select(decode_chunk_pages_lut, 0, effective_max_pages.view(1))

    page_blocks = triton.cdiv(int(page_table.shape[1]), _DECODE_BLOCK_PAGES)
    build_decode_graph_page_table_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        page_table,
        active_max_pages,
        req_to_token.stride(0),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )

    merge_indptr.zero_()
    update_regular_decode_graph_metadata_triton[(bs,)](
        cache_seqlens,
        merge_indptr,
        kv_window_start_tokens,
        decode_chunk_pages,
        PAGE_SIZE=page_size,
        WINDOW_PAGE_SPAN=int(window_page_span),
    )
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    kv_chunk_size_ptr.copy_(decode_chunk_pages * page_size)


def update_regular_decode_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_chunk_size: int | torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    max_chunks_per_req: int,
    page_size: int,
    window_page_span: int = 0,
) -> None:
    device = cache_seqlens.device
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if kv_chunk_size_ptr.device != device:
        raise ValueError("decode graph buffers and cache_seqlens must be on the same device")
    if kv_window_start_tokens.device != device:
        raise ValueError("kv_window_start_tokens and cache_seqlens must be on the same device")
    if max_chunks_per_req <= 0:
        raise ValueError("max_chunks_per_req must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")

    if isinstance(kv_chunk_size, torch.Tensor):
        if kv_chunk_size.device != device:
            raise ValueError("kv_chunk_size tensor and cache_seqlens must be on the same device")
        if kv_chunk_size.numel() != 1:
            raise ValueError("kv_chunk_size tensor must contain exactly one element")
        kv_chunk_size_i32 = kv_chunk_size.reshape(1).to(torch.int32)
    else:
        if kv_chunk_size <= 0:
            raise ValueError("kv_chunk_size must be positive")
        kv_chunk_size_i32 = None

    merge_indptr.zero_()
    num_pages = torch.maximum(
        torch.div(
            cache_seqlens[:bs] + (int(page_size) - 1),
            int(page_size),
            rounding_mode="floor",
        ),
        torch.ones(bs, dtype=torch.int32, device=device),
    )
    if window_page_span > 0:
        effective_pages = torch.minimum(
            num_pages,
            torch.full((bs,), int(window_page_span), dtype=torch.int32, device=device),
        )
    else:
        effective_pages = num_pages
    kv_window_start_tokens[:bs].copy_((num_pages - effective_pages) * int(page_size))
    effective_tokens = effective_pages * int(page_size)
    if kv_chunk_size_i32 is None:
        num_chunks = torch.maximum(
            torch.div(
                effective_tokens + (int(kv_chunk_size) - 1),
                int(kv_chunk_size),
                rounding_mode="floor",
            ),
            torch.ones(bs, dtype=torch.int32, device=device),
        )
    else:
        num_chunks = torch.maximum(
            torch.div(
                effective_tokens + (kv_chunk_size_i32 - 1),
                kv_chunk_size_i32,
                rounding_mode="floor",
            ),
            torch.ones(bs, dtype=torch.int32, device=device),
        )
    merge_indptr[1 : bs + 1].copy_(num_chunks)
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    if kv_chunk_size_i32 is None:
        kv_chunk_size_ptr[0] = int(kv_chunk_size)
    else:
        kv_chunk_size_ptr[:1].copy_(kv_chunk_size_i32.to(kv_chunk_size_ptr.dtype))


def update_decode_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
) -> None:
    device = cache_seqlens.device
    if request_indices.device != device:
        raise ValueError("request_indices and cache_seqlens must be on the same device")
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if block_valid_mask.device != device or kv_chunk_size_ptr.device != device:
        raise ValueError("decode graph buffers and cache_seqlens must be on the same device")
    if decode_chunk_pages_lut.device != device:
        raise ValueError("decode_chunk_pages_lut and cache_seqlens must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    work_items_capacity = int(request_indices.shape[0])
    if work_items_capacity % bs != 0:
        raise RuntimeError("decode graph workspace request_indices shape is incompatible with the batch bucket")
    max_chunks_per_req = work_items_capacity // bs
    if max_chunks_per_req <= 0:
        raise RuntimeError("decode graph workspace must allocate at least one chunk per request")

    max_cache_pages = torch.div(
        cache_seqlens[:bs].amax() + (page_size - 1),
        page_size,
        rounding_mode="floor",
    ).clamp_(min=1, max=decode_chunk_pages_lut.shape[0] - 1).to(torch.int64)
    if window_page_span > 0:
        effective_max_pages = torch.minimum(
            max_cache_pages,
            torch.tensor(int(window_page_span), dtype=torch.int64, device=max_cache_pages.device),
        )
    else:
        effective_max_pages = max_cache_pages
    effective_max_pages = effective_max_pages.clamp_(min=1, max=decode_chunk_pages_lut.shape[0] - 1)
    decode_chunk_pages = torch.index_select(decode_chunk_pages_lut, 0, effective_max_pages.view(1))

    block_valid_mask.zero_()
    merge_indptr.zero_()
    chunk_blocks = triton.cdiv(max_chunks_per_req, _DECODE_BLOCK_CHUNKS)
    update_decode_graph_metadata_triton[(bs, chunk_blocks)](
        cache_seqlens,
        merge_indptr,
        block_valid_mask,
        kv_window_start_tokens,
        decode_chunk_pages,
        max_chunks_per_req,
        PAGE_SIZE=page_size,
        WINDOW_PAGE_SPAN=int(window_page_span),
        BLOCK_CHUNKS=_DECODE_BLOCK_CHUNKS,
    )
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    kv_chunk_size_ptr.copy_(decode_chunk_pages * page_size)
