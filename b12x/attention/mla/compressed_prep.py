"""Graph-capturable prep for compressed MLA pages into the shared MLA core layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .compressed_reference import (
    COMPRESSED_MLA_HEAD_DIM,
    compressed_mla_page_nbytes,
)
from .reference import _MLA_NOPE_DIM, _MLA_PACKED_DIM, _MLA_ROPE_DIM


_CORE_HEAD_DIM = _MLA_NOPE_DIM + _MLA_ROPE_DIM


@dataclass(frozen=True)
class CompressedMLAPrepScratch:
    q_all: torch.Tensor
    kv_cache: torch.Tensor
    page_table_1: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor

    @property
    def v_head_dim(self) -> int:
        return _MLA_NOPE_DIM


@triton.jit
def _prepare_compressed_mla_q_kernel(
    q_ptr,
    q_core_ptr,
    HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    core_mask = dims < 576
    src_dims = tl.where(dims < 448, dims, dims - 64)
    copy_mask = (dims < 448) | (dims >= 512)
    vals = tl.load(
        q_ptr + (row * HEADS + head) * 512 + src_dims,
        mask=core_mask & copy_mask,
        other=0.0,
    )
    vals = tl.where(copy_mask, vals, tl.zeros((BLOCK_D,), tl.bfloat16))
    tl.store(q_core_ptr + (row * HEADS + head) * 576 + dims, vals, mask=core_mask)


@triton.jit
def _ue8m0_to_f32(scale_u8):
    return tl.exp2(scale_u8.to(tl.float32) - 127.0)


@triton.jit
def _compute_valid_prefix_length(indices_ptr, lengths_ptr, row, width: tl.constexpr, block: tl.constexpr):
    requested = tl.load(lengths_ptr + row).to(tl.int32)
    requested = tl.minimum(tl.maximum(requested, 0), width)
    offs = tl.arange(0, block)
    vals = tl.load(
        indices_ptr + row * width + offs,
        mask=offs < width,
        other=-1,
    ).to(tl.int32)
    invalid_pos = tl.where((offs < requested) & (vals < 0), offs, requested)
    return tl.min(invalid_pos, axis=0).to(tl.int32)


@triton.jit
def _sanitize_compressed_mla_lengths_kernel(
    swa_indices_ptr,
    swa_lengths_ptr,
    indexed_indices_ptr,
    indexed_lengths_ptr,
    swa_valid_lengths_ptr,
    indexed_valid_lengths_ptr,
    SWA_WIDTH: tl.constexpr,
    INDEXED_WIDTH: tl.constexpr,
    HAS_INDEXED: tl.constexpr,
    BLOCK_SWA: tl.constexpr,
    BLOCK_INDEXED: tl.constexpr,
):
    row = tl.program_id(0)
    swa_len = _compute_valid_prefix_length(
        swa_indices_ptr,
        swa_lengths_ptr,
        row,
        SWA_WIDTH,
        BLOCK_SWA,
    )
    tl.store(swa_valid_lengths_ptr + row, swa_len)
    indexed_len = tl.full((), 0, tl.int32)
    if HAS_INDEXED:
        indexed_len = _compute_valid_prefix_length(
            indexed_indices_ptr,
            indexed_lengths_ptr,
            row,
            INDEXED_WIDTH,
            BLOCK_INDEXED,
        )
    tl.store(indexed_valid_lengths_ptr + row, indexed_len)


@triton.jit
def _prepare_compressed_mla_kv_kernel(
    swa_fp8_ptr,
    swa_u8_ptr,
    swa_bf16_ptr,
    swa_indices_ptr,
    swa_lengths_ptr,
    indexed_fp8_ptr,
    indexed_u8_ptr,
    indexed_bf16_ptr,
    indexed_indices_ptr,
    indexed_lengths_ptr,
    kv_fp8_ptr,
    kv_f32_ptr,
    kv_bf16_ptr,
    page_table_ptr,
    active_counts_ptr,
    SWA_WIDTH: tl.constexpr,
    INDEXED_WIDTH: tl.constexpr,
    TOTAL_WIDTH: tl.constexpr,
    SWA_PAGE_SIZE: tl.constexpr,
    SWA_PAGE_NBYTES: tl.constexpr,
    INDEXED_PAGE_SIZE: tl.constexpr,
    INDEXED_PAGE_NBYTES: tl.constexpr,
    HAS_INDEXED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    group = tl.program_id(2)
    offs = tl.arange(0, BLOCK_D)

    swa_len = tl.load(swa_lengths_ptr + row).to(tl.int32)
    indexed_len = tl.full((), 0, tl.int32)
    if HAS_INDEXED:
        indexed_len = tl.load(indexed_lengths_ptr + row).to(tl.int32)

    core_row = row * TOTAL_WIDTH + slot
    if group == 0:
        tl.store(page_table_ptr + row * TOTAL_WIDTH + slot, core_row)
        if slot == 0:
            tl.store(active_counts_ptr + row, swa_len + indexed_len)

    use_swa = slot < swa_len
    extra_slot = slot - swa_len
    active_swa = use_swa
    active_indexed = (~use_swa) & HAS_INDEXED & (extra_slot < indexed_len)
    active = active_swa | active_indexed

    swa_index = tl.load(
        swa_indices_ptr + row * SWA_WIDTH + slot,
        mask=active_swa,
        other=0,
    ).to(tl.int64)
    indexed_index = tl.load(
        indexed_indices_ptr + row * INDEXED_WIDTH + extra_slot,
        mask=active_indexed,
        other=0,
    ).to(tl.int64)
    token_index = tl.where(active_swa, swa_index, indexed_index)

    page_size = tl.where(use_swa, SWA_PAGE_SIZE, INDEXED_PAGE_SIZE)
    page_nbytes = tl.where(use_swa, SWA_PAGE_NBYTES, INDEXED_PAGE_NBYTES)
    page = token_index // page_size
    token_offset = token_index - page * page_size
    payload_base = page * page_nbytes + token_offset * 576
    scale_base = page * page_nbytes + page_size * 576 + token_offset * 8

    dims = group * 128 + offs
    fp8_ptr = tl.where(use_swa, swa_fp8_ptr, indexed_fp8_ptr)
    u8_ptr = tl.where(use_swa, swa_u8_ptr, indexed_u8_ptr)
    bf16_ptr = tl.where(use_swa, swa_bf16_ptr, indexed_bf16_ptr)

    vals = tl.zeros((BLOCK_D,), tl.float32)
    if group < 3:
        scale_ids = dims // 64
        raw = tl.load(fp8_ptr + payload_base + dims, mask=active, other=0.0).to(tl.float32)
        scale_u8_vec = tl.load(u8_ptr + scale_base + scale_ids, mask=active, other=127).to(tl.uint32)
        vals = raw * _ue8m0_to_f32(scale_u8_vec)
    else:
        nope_mask = offs < 64
        nope_dims = 384 + offs
        raw_nope = tl.load(
            fp8_ptr + payload_base + nope_dims,
            mask=active & nope_mask,
            other=0.0,
        ).to(tl.float32)
        scale_u8_scalar = tl.load(u8_ptr + scale_base + 6, mask=active, other=127).to(tl.uint32)
        nope_vals = raw_nope * _ue8m0_to_f32(scale_u8_scalar)
        rope_vals = tl.load(
            bf16_ptr + (payload_base + 448) // 2 + (offs - 64),
            mask=active & (~nope_mask),
            other=0.0,
        ).to(tl.float32)
        vals = tl.where(nope_mask, nope_vals, rope_vals)

        rope_offsets = tl.arange(0, 64)
        rope_copy = tl.load(
            bf16_ptr + (payload_base + 448) // 2 + rope_offsets,
            mask=active,
            other=0.0,
        )
        tl.store(
            kv_bf16_ptr + (core_row * 656 + 528) // 2 + rope_offsets,
            rope_copy,
            mask=rope_offsets < 64,
        )

    vals = tl.where(active, vals, tl.zeros((BLOCK_D,), tl.float32))
    max_abs = tl.max(tl.abs(vals), axis=0)
    scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    quant = tl.minimum(tl.maximum(vals / scale, -448.0), 448.0).to(tl.float8e4nv)

    tl.store(kv_fp8_ptr + core_row * 656 + group * 128 + offs, quant, mask=offs < 128)
    tl.store(kv_f32_ptr + (core_row * 656 + 512) // 4 + group, scale)


def prepare_compressed_mla_core_inputs_cuda(
    *,
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    workspace: object,
    swa_page_size: int,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_topk_lengths: torch.Tensor | None = None,
    indexed_page_size: int | None = None,
) -> CompressedMLAPrepScratch:
    """Prepare compressed MLA pages for the current shared sparse-MLA CUDA core."""

    _validate_cuda_inputs(
        q_all=q_all,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_topk_lengths,
        indexed_k_cache=indexed_k_cache,
        indexed_indices=indexed_indices,
        indexed_topk_lengths=indexed_topk_lengths,
        indexed_page_size=indexed_page_size,
    )
    rows = int(q_all.shape[0])
    heads = int(q_all.shape[1])
    swa_width = int(swa_indices.shape[1])
    indexed_width = int(indexed_indices.shape[1]) if indexed_indices is not None else 0
    live_width = swa_width + indexed_width
    if live_width <= 0:
        raise ValueError("compressed MLA requires at least one SWA or indexed slot")

    graph_capacity = bool(
        getattr(workspace, "fixed_capacity", False) or getattr(workspace, "use_cuda_graph", False)
    )
    q_capacity = int(getattr(workspace, "max_total_q", rows)) if graph_capacity else rows
    width_capacity = int(getattr(workspace, "topk", live_width)) if graph_capacity else live_width
    if rows > q_capacity:
        raise ValueError(f"q rows {rows} exceed compressed prep capacity {q_capacity}")
    if live_width > width_capacity:
        raise ValueError(f"compressed MLA width {live_width} exceeds workspace topk {width_capacity}")

    (
        q_core,
        kv_cache,
        page_table,
        active_counts,
        swa_valid_lengths,
        indexed_valid_lengths,
    ) = _get_or_alloc_prep_buffers(
        workspace=workspace,
        q_capacity=q_capacity,
        rows=rows,
        heads=heads,
        width_capacity=width_capacity,
        device=q_all.device,
    )

    q_live = q_core[:rows, :heads, :]
    kv_live = kv_cache[: rows * width_capacity, :, :]
    page_table_live = page_table[:rows, :width_capacity]
    active_counts_live = active_counts[:rows]
    swa_valid_lengths_live = swa_valid_lengths[:rows]
    indexed_valid_lengths_live = indexed_valid_lengths[:rows]

    _prepare_compressed_mla_q_kernel[(rows, heads)](
        q_all,
        q_live,
        HEADS=heads,
        BLOCK_D=triton.next_power_of_2(_CORE_HEAD_DIM),
        num_warps=8,
    )

    has_indexed = indexed_k_cache is not None
    if not has_indexed:
        indexed_k_cache = swa_k_cache
        indexed_indices = swa_indices
        indexed_topk_lengths = swa_topk_lengths
        indexed_page_size = swa_page_size

    assert indexed_k_cache is not None
    assert indexed_indices is not None
    assert indexed_topk_lengths is not None
    assert indexed_page_size is not None

    _sanitize_compressed_mla_lengths_kernel[(rows,)](
        swa_indices,
        swa_topk_lengths,
        indexed_indices,
        indexed_topk_lengths,
        swa_valid_lengths_live,
        indexed_valid_lengths_live,
        SWA_WIDTH=swa_width,
        INDEXED_WIDTH=indexed_width,
        HAS_INDEXED=has_indexed,
        BLOCK_SWA=triton.next_power_of_2(max(swa_width, 1)),
        BLOCK_INDEXED=triton.next_power_of_2(max(indexed_width, 1)),
        num_warps=8,
    )

    _prepare_compressed_mla_kv_kernel[(rows, width_capacity, 4)](
        swa_k_cache.view(torch.float8_e4m3fn),
        swa_k_cache,
        swa_k_cache.view(torch.bfloat16),
        swa_indices,
        swa_valid_lengths_live,
        indexed_k_cache.view(torch.float8_e4m3fn),
        indexed_k_cache,
        indexed_k_cache.view(torch.bfloat16),
        indexed_indices,
        indexed_valid_lengths_live,
        kv_live.view(torch.float8_e4m3fn),
        kv_live.view(torch.float32),
        kv_live.view(torch.bfloat16),
        page_table_live,
        active_counts_live,
        SWA_WIDTH=swa_width,
        INDEXED_WIDTH=indexed_width,
        TOTAL_WIDTH=width_capacity,
        SWA_PAGE_SIZE=int(swa_page_size),
        SWA_PAGE_NBYTES=compressed_mla_page_nbytes(int(swa_page_size)),
        INDEXED_PAGE_SIZE=int(indexed_page_size),
        INDEXED_PAGE_NBYTES=compressed_mla_page_nbytes(int(indexed_page_size)),
        HAS_INDEXED=has_indexed,
        BLOCK_D=128,
        num_warps=4,
    )

    return CompressedMLAPrepScratch(
        q_all=q_live,
        kv_cache=kv_live,
        page_table_1=page_table_live,
        cache_seqlens_int32=active_counts_live,
        nsa_cache_seqlens_int32=active_counts_live,
    )


def _validate_cuda_inputs(
    *,
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    indexed_k_cache: torch.Tensor | None,
    indexed_indices: torch.Tensor | None,
    indexed_topk_lengths: torch.Tensor | None,
    indexed_page_size: int | None,
) -> None:
    if q_all.device.type != "cuda":
        raise ValueError("compressed MLA CUDA prep requires CUDA q_all")
    if q_all.dtype != torch.bfloat16:
        raise TypeError(f"q_all must have dtype torch.bfloat16, got {q_all.dtype}")
    if q_all.ndim != 3 or q_all.shape[-1] != COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(f"q_all must have shape [rows, heads, {COMPRESSED_MLA_HEAD_DIM}], got {tuple(q_all.shape)}")
    if not q_all.is_contiguous():
        raise ValueError("q_all must be contiguous")
    _validate_cache(swa_k_cache, "swa_k_cache")
    _validate_indices(swa_indices, "swa_indices", rows=int(q_all.shape[0]))
    _validate_lengths(swa_topk_lengths, "swa_topk_lengths", rows=int(q_all.shape[0]))

    has_indexed = indexed_k_cache is not None or indexed_indices is not None or indexed_topk_lengths is not None
    if not has_indexed:
        return
    if indexed_k_cache is None or indexed_indices is None or indexed_topk_lengths is None:
        raise ValueError("indexed_k_cache, indexed_indices, and indexed_topk_lengths must be provided together")
    if indexed_page_size is None:
        raise ValueError("indexed_page_size is required when indexed_k_cache is provided")
    _validate_cache(indexed_k_cache, "indexed_k_cache")
    _validate_indices(indexed_indices, "indexed_indices", rows=int(q_all.shape[0]))
    _validate_lengths(indexed_topk_lengths, "indexed_topk_lengths", rows=int(q_all.shape[0]))


def _validate_cache(cache: torch.Tensor, name: str) -> None:
    if cache.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA")
    if cache.dtype != torch.uint8:
        raise TypeError(f"{name} must have dtype torch.uint8, got {cache.dtype}")
    if cache.ndim != 2:
        raise ValueError(f"{name} must have shape [pages, page_nbytes], got {tuple(cache.shape)}")
    if not cache.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_indices(indices: torch.Tensor, name: str, *, rows: int) -> None:
    if indices.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA")
    if indices.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32 for CUDA prep, got {indices.dtype}")
    if indices.ndim != 2 or indices.shape[0] != rows:
        raise ValueError(f"{name} must have shape [{rows}, width], got {tuple(indices.shape)}")
    if not indices.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_lengths(lengths: torch.Tensor, name: str, *, rows: int) -> None:
    if lengths.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA")
    if lengths.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32 for CUDA prep, got {lengths.dtype}")
    if lengths.shape != (rows,):
        raise ValueError(f"{name} must have shape [{rows}], got {tuple(lengths.shape)}")
    if not lengths.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _get_or_alloc_prep_buffers(
    *,
    workspace: object,
    q_capacity: int,
    rows: int,
    heads: int,
    width_capacity: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_shape = (q_capacity, heads, _CORE_HEAD_DIM)
    kv_shape = (q_capacity * width_capacity, 1, _MLA_PACKED_DIM)
    page_table_shape = (q_capacity, width_capacity)
    counts_shape = (q_capacity,)
    q_core = _workspace_buffer(workspace, "_compressed_mla_q_core", q_shape, torch.bfloat16, device)
    kv_cache = _workspace_buffer(workspace, "_compressed_mla_kv_core", kv_shape, torch.uint8, device)
    page_table = _workspace_buffer(workspace, "_compressed_mla_page_table", page_table_shape, torch.int32, device)
    active_counts = _workspace_buffer(workspace, "_compressed_mla_active_counts", counts_shape, torch.int32, device)
    if rows <= 0:
        raise ValueError("q rows must be positive")
    swa_valid_lengths = _workspace_buffer(
        workspace,
        "_compressed_mla_swa_valid_lengths",
        counts_shape,
        torch.int32,
        device,
    )
    indexed_valid_lengths = _workspace_buffer(
        workspace,
        "_compressed_mla_indexed_valid_lengths",
        counts_shape,
        torch.int32,
        device,
    )
    return q_core, kv_cache, page_table, active_counts, swa_valid_lengths, indexed_valid_lengths


def _workspace_buffer(
    workspace: object,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    existing = getattr(workspace, name, None)
    valid = (
        isinstance(existing, torch.Tensor)
        and existing.device == device
        and existing.dtype == dtype
        and tuple(existing.shape) == tuple(shape)
    )
    if valid:
        return existing
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"compressed MLA prep buffer {name} was not allocated before CUDA graph capture")
    buffer = torch.empty(shape, dtype=dtype, device=device)
    setattr(workspace, name, buffer)
    return buffer
