"""Caller-owned scratch plans for compressed MLA and compressed indexer paths."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x.attention.mla.compressed_config import (
    compressed_mla_split_config_for_contract,
)
from b12x.attention.mla.compressed_reference import COMPRESSED_MLA_HEAD_DIM
from b12x.attention.workspace import (
    _ARENA_ALIGN_BYTES,
    B12XIndexerTopKPositionBufferUnavailable,
    B12XAttentionWorkspace,
    _align_up,
    _dtype_nbytes,
    _materialize_arena_strided_view,
    _materialize_arena_view,
    _split_output_buffer_from_tmp,
    _split_tmp_output_stride,
)
from b12x.integration.scratch import (
    B12XScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)

_COMPRESSED_INDEX_SUPERTILE_K_ENV = "B12X_COMPRESSED_INDEX_SUPERTILE_K"
_COMPRESSED_INDEX_SUPERTILE_K_DEFAULT = 32768
_COMPRESSED_INDEX_TILE_BLOCK_Q = 32
_COMPRESSED_INDEX_TILE_BLOCK_K = 512
_COMPRESSED_INDEX_HEAD_DIM = 128


@dataclass(frozen=True, kw_only=True)
class B12XCompressedMLAScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_width: int
    max_page_table_width: int | None = None
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.uint8
    head_dim: int = COMPRESSED_MLA_HEAD_DIM
    v_head_dim: int = COMPRESSED_MLA_HEAD_DIM
    max_batch: int | None = None
    max_kv_rows: int = 0
    max_chunks_per_row: int = 64
    max_q_chunks: int | None = None
    page_size: int = 64

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "max_q_rows", max(int(self.max_q_rows), 1))
        object.__setattr__(self, "max_width", max(int(self.max_width), 1))
        max_page_table_width = self.max_width if self.max_page_table_width is None else self.max_page_table_width
        object.__setattr__(self, "max_page_table_width", max(int(max_page_table_width), 1))
        object.__setattr__(self, "head_dim", max(int(self.head_dim), 1))
        object.__setattr__(self, "v_head_dim", max(int(self.v_head_dim), 1))
        max_batch = self.max_q_rows if self.max_batch is None else self.max_batch
        object.__setattr__(self, "max_batch", max(int(max_batch), 1))
        object.__setattr__(self, "max_kv_rows", max(int(self.max_kv_rows), 0))
        object.__setattr__(self, "max_chunks_per_row", max(int(self.max_chunks_per_row), 1))
        if self.max_q_chunks is not None:
            object.__setattr__(self, "max_q_chunks", max(int(self.max_q_chunks), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))


@dataclass(frozen=True, kw_only=True)
class B12XCompressedIndexerScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_page_table_width: int
    topk: int
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.uint8
    max_batch: int | None = None
    page_size: int = 64
    max_k_rows: int = 0
    reserve_paged_logits: bool = True
    paged_logits_k_rows: int = 0
    paged_tile_logits_k_rows: int = 0

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "max_q_rows", max(int(self.max_q_rows), 1))
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(self.max_page_table_width), 1),
        )
        object.__setattr__(self, "topk", max(int(self.topk), 1))
        max_batch = self.max_q_rows if self.max_batch is None else self.max_batch
        object.__setattr__(self, "max_batch", max(int(max_batch), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))
        object.__setattr__(self, "max_k_rows", max(int(self.max_k_rows), 0))
        object.__setattr__(self, "reserve_paged_logits", bool(self.reserve_paged_logits))
        object.__setattr__(self, "paged_logits_k_rows", max(int(self.paged_logits_k_rows), 0))
        object.__setattr__(
            self,
            "paged_tile_logits_k_rows",
            max(int(self.paged_tile_logits_k_rows), 0),
        )


@dataclass(frozen=True, kw_only=True)
class _B12XCompressedMLAScratchLayout:
    nbytes: int
    max_q_chunks: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int
    final_lse_offset_bytes: int
    kv_chunk_size_offset_bytes: int
    num_chunks_offset_bytes: int
    sm_scale_offset_bytes: int


@dataclass(frozen=True, kw_only=True)
class _B12XCompressedIndexerScratchLayout:
    nbytes: int
    supertile_tokens: int
    max_chunks: int
    tile_logits_elements: int
    tile_logits_offset_bytes: int
    topk_values_offset_bytes: int
    topk_indices_offset_bytes: int
    candidate_values_offset_bytes: int
    candidate_indices_offset_bytes: int
    merge_positions_offset_bytes: int
    active_width_offset_bytes: int


@dataclass(kw_only=True)
class B12XCompressedMLAScratch:
    """Component-owned compressed MLA scratch views over caller-owned storage."""

    shared_scratch: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    head_dim: int
    v_head_dim: int
    topk: int
    max_page_table_width: int
    max_total_q: int
    max_batch: int
    max_kv_rows: int
    max_chunks_per_row: int
    page_size: int
    mode: str = "decode"
    fixed_capacity: bool = True
    use_cuda_graph: bool = False
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    output_buffer: torch.Tensor | None = None
    final_lse: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    num_chunks_ptr: torch.Tensor | None = None
    sm_scale_tensor: torch.Tensor | None = None
    kv_chunk_size_value: int | None = None
    num_chunks_value: int | None = None
    sm_scale_value: float | None = None
    _contract_q: torch.Tensor | None = None
    _contract_page_table: torch.Tensor | None = None
    _contract_indexer_cache_seqlens: torch.Tensor | None = None
    _contract_output: torch.Tensor | None = None
    _contract_tmp_output: torch.Tensor | None = None
    _contract_tmp_lse: torch.Tensor | None = None

    def set_split_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        if num_chunks <= 0 or num_chunks > self.max_chunks_per_row:
            raise ValueError(
                f"num_chunks must be in [1, {self.max_chunks_per_row}], got {num_chunks}"
            )
        if kv_chunk_size <= 0:
            raise ValueError(f"kv_chunk_size must be positive, got {kv_chunk_size}")
        if self.kv_chunk_size_ptr is None or self.num_chunks_ptr is None:
            raise RuntimeError("compressed MLA scratch is missing split-control tensors")
        if self.kv_chunk_size_value != int(kv_chunk_size):
            self.kv_chunk_size_ptr.fill_(int(kv_chunk_size))
            self.kv_chunk_size_value = int(kv_chunk_size)
        if self.num_chunks_value != int(num_chunks):
            self.num_chunks_ptr.fill_(int(num_chunks))
            self.num_chunks_value = int(num_chunks)

    def bind(
        self,
        *,
        q: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        indexed_indices: torch.Tensor | None = None,
        indexed_lengths: torch.Tensor | None = None,
        indexed_page_table: torch.Tensor | None = None,
    ) -> "B12XCompressedMLABinding":
        return build_compressed_mla_binding(
            scratch=self,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_table=indexed_page_table,
        )


@dataclass(kw_only=True)
class B12XCompressedIndexerScratch:
    """Compressed indexer scratch views over caller-owned storage."""

    shared_scratch: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    topk: int
    max_page_table_width: int
    max_total_q: int
    max_paged_q_rows: int
    max_batch: int
    page_size: int
    paged_tile_logits_k_rows: int
    max_chunks: int
    fixed_capacity: bool = True
    use_cuda_graph: bool = False
    indexer_extend_tile_logits: torch.Tensor | None = None
    indexer_extend_topk_values: torch.Tensor | None = None
    indexer_extend_topk_indices: torch.Tensor | None = None
    indexer_extend_candidate_values: torch.Tensor | None = None
    indexer_extend_candidate_indices: torch.Tensor | None = None
    indexer_extend_topk_positions: torch.Tensor | None = None
    paged_indexer_active_width_cap: torch.Tensor | None = None
    _contract_paged_indexer_q_bytes: torch.Tensor | None = None
    _contract_paged_indexer_weights: torch.Tensor | None = None
    _contract_paged_real_page_table: torch.Tensor | None = None
    _contract_paged_indexer_cache_seqlens: torch.Tensor | None = None
    _contract_paged_indexer_tile_logits: torch.Tensor | None = None
    _contract_paged_indexer_topk_values: torch.Tensor | None = None
    _contract_paged_indexer_topk_indices: torch.Tensor | None = None

    def bind(
        self,
        *,
        real_page_table: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        active_width: torch.Tensor | None = None,
        schedule_metadata: torch.Tensor | None = None,
        expected_num_q_heads: int | None = None,
        shared_page_table: bool = False,
    ) -> "B12XCompressedIndexerBinding":
        return build_compressed_indexer_binding(
            scratch=self,
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            active_width=active_width,
            schedule_metadata=schedule_metadata,
            expected_num_q_heads=expected_num_q_heads,
            shared_page_table=shared_page_table,
        )

    def get_indexer_extend_tile_logits(self) -> torch.Tensor:
        if self.indexer_extend_tile_logits is None:
            raise RuntimeError("compressed indexer scratch is missing tiled logits")
        return self.indexer_extend_tile_logits

    def get_indexer_extend_topk_buffers(
        self,
        *,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.indexer_extend_topk_values is None
            or self.indexer_extend_topk_indices is None
        ):
            raise RuntimeError("compressed indexer scratch is missing top-k buffers")
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > int(self.indexer_extend_topk_indices.shape[0]):
            raise ValueError(
                "row_count "
                f"{row_count} exceeds compressed indexer scratch top-k capacity "
                f"{int(self.indexer_extend_topk_indices.shape[0])}"
            )
        return (
            self.indexer_extend_topk_values[:row_count],
            self.indexer_extend_topk_indices[:row_count],
        )

    def get_indexer_extend_candidate_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.indexer_extend_candidate_values is None
            or self.indexer_extend_candidate_indices is None
        ):
            raise RuntimeError("compressed indexer scratch is missing candidate buffers")
        return self.indexer_extend_candidate_values, self.indexer_extend_candidate_indices

    def get_indexer_extend_topk_position_buffer(self, *, row_count: int) -> torch.Tensor:
        if self.indexer_extend_topk_positions is None:
            raise B12XIndexerTopKPositionBufferUnavailable(
                "compressed indexer scratch is missing top-k position buffer"
            )
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > int(self.indexer_extend_topk_positions.shape[0]):
            raise ValueError(
                "row_count "
                f"{row_count} exceeds compressed indexer scratch position capacity "
                f"{int(self.indexer_extend_topk_positions.shape[0])}"
            )
        return self.indexer_extend_topk_positions[:row_count]

    def get_paged_indexer_active_width_cap(self) -> torch.Tensor:
        if self.paged_indexer_active_width_cap is None:
            raise RuntimeError("compressed indexer scratch is missing active-width cap")
        return self.paged_indexer_active_width_cap

    def get_paged_indexer_contract_phantoms(self) -> dict[str, torch.Tensor]:
        if (
            self._contract_paged_indexer_q_bytes is None
            or self._contract_paged_indexer_weights is None
            or self._contract_paged_real_page_table is None
            or self._contract_paged_indexer_cache_seqlens is None
            or self._contract_paged_indexer_tile_logits is None
            or self._contract_paged_indexer_topk_values is None
            or self._contract_paged_indexer_topk_indices is None
        ):
            raise RuntimeError("compressed indexer scratch is missing contract phantoms")
        return {
            "q_bytes": self._contract_paged_indexer_q_bytes,
            "weights": self._contract_paged_indexer_weights,
            "real_page_table": self._contract_paged_real_page_table,
            "seqlens_per_query": self._contract_paged_indexer_cache_seqlens,
            "tile_logits": self._contract_paged_indexer_tile_logits,
            "topk_values": self._contract_paged_indexer_topk_values,
            "topk_indices": self._contract_paged_indexer_topk_indices,
        }


@dataclass(frozen=True, kw_only=True)
class B12XCompressedMLABinding:
    scratch: object
    q: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    indexed_indices: torch.Tensor | None = None
    indexed_lengths: torch.Tensor | None = None
    indexed_page_table: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class B12XCompressedIndexerBinding:
    scratch: object
    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    active_width: torch.Tensor
    schedule_metadata: torch.Tensor | None = None
    expected_num_q_heads: int | None = None
    shared_page_table: bool = False


def _compressed_mla_scratch_layout(
    caps: B12XCompressedMLAScratchCaps,
) -> _B12XCompressedMLAScratchLayout:
    max_total_q = max(int(caps.max_q_rows), 1)
    max_chunks_per_row = max(int(caps.max_chunks_per_row), 1)
    default_q_chunks = max_total_q * max_chunks_per_row
    max_q_chunks = (
        default_q_chunks
        if caps.max_q_chunks is None
        else max(int(caps.max_q_chunks), default_q_chunks)
    )

    cursor = 0
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)
    tmp_output_offset_bytes = cursor
    cursor += (
        max_q_chunks
        * int(caps.num_q_heads)
        * int(caps.v_head_dim)
        * _dtype_nbytes(caps.dtype)
    )
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    tmp_lse_offset_bytes = cursor
    cursor += (
        max_q_chunks
        * int(caps.num_q_heads)
        * _dtype_nbytes(torch.float32)
    )
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    final_lse_offset_bytes = cursor
    cursor += (
        max_total_q
        * int(caps.num_q_heads)
        * _dtype_nbytes(torch.float32)
    )
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    kv_chunk_size_offset_bytes = cursor
    cursor += _dtype_nbytes(torch.int32)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    num_chunks_offset_bytes = cursor
    cursor += _dtype_nbytes(torch.int32)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    sm_scale_offset_bytes = cursor
    cursor += _dtype_nbytes(torch.float32)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    return _B12XCompressedMLAScratchLayout(
        nbytes=max(int(cursor), _ARENA_ALIGN_BYTES),
        max_q_chunks=max_q_chunks,
        tmp_output_offset_bytes=tmp_output_offset_bytes,
        tmp_lse_offset_bytes=tmp_lse_offset_bytes,
        final_lse_offset_bytes=final_lse_offset_bytes,
        kv_chunk_size_offset_bytes=kv_chunk_size_offset_bytes,
        num_chunks_offset_bytes=num_chunks_offset_bytes,
        sm_scale_offset_bytes=sm_scale_offset_bytes,
    )


def _shape_only_scratch_tensor(
    scratch: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = scratch.narrow(0, 0, _dtype_nbytes(dtype)).view(dtype)
    return base.as_strided(shape, (0,) * len(shape))


def _install_compressed_mla_contract_phantoms(
    scratch: B12XCompressedMLAScratch,
) -> None:
    storage = scratch.shared_scratch
    scratch._contract_q = _shape_only_scratch_tensor(
        storage,
        (
            int(scratch.max_total_q),
            int(scratch.num_q_heads),
            int(scratch.head_dim) // 4,
        ),
        dtype=torch.uint32,
    )
    scratch._contract_page_table = _shape_only_scratch_tensor(
        storage,
        (int(scratch.max_total_q), int(scratch.topk)),
        dtype=torch.int32,
    )
    scratch._contract_indexer_cache_seqlens = _shape_only_scratch_tensor(
        storage,
        (int(scratch.max_total_q),),
        dtype=torch.int32,
    )
    scratch._contract_output = _shape_only_scratch_tensor(
        storage,
        (
            int(scratch.max_total_q),
            int(scratch.num_q_heads),
            int(scratch.v_head_dim),
        ),
        dtype=scratch.dtype,
    )
    scratch._contract_tmp_output = _shape_only_scratch_tensor(
        storage,
        (
            int(scratch.max_total_q),
            int(scratch.num_q_heads),
            int(scratch.max_chunks_per_row),
            int(scratch.v_head_dim),
        ),
        dtype=scratch.dtype,
    )
    scratch._contract_tmp_lse = _shape_only_scratch_tensor(
        storage,
        (
            int(scratch.max_total_q),
            int(scratch.num_q_heads),
            int(scratch.max_chunks_per_row),
        ),
        dtype=torch.float32,
    )


def _resolve_compressed_indexer_supertile_tokens(raw_tokens: int) -> int:
    if int(raw_tokens) <= 0:
        raw = os.environ.get(_COMPRESSED_INDEX_SUPERTILE_K_ENV)
        if raw is None:
            raw_tokens = _COMPRESSED_INDEX_SUPERTILE_K_DEFAULT
        else:
            try:
                raw_tokens = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{_COMPRESSED_INDEX_SUPERTILE_K_ENV} must be an integer, got {raw!r}"
                ) from exc
    tokens = max(int(raw_tokens), _COMPRESSED_INDEX_TILE_BLOCK_K)
    return (
        (tokens + _COMPRESSED_INDEX_TILE_BLOCK_K - 1)
        // _COMPRESSED_INDEX_TILE_BLOCK_K
    ) * _COMPRESSED_INDEX_TILE_BLOCK_K


def _compressed_indexer_scratch_layout(
    caps: B12XCompressedIndexerScratchCaps,
) -> _B12XCompressedIndexerScratchLayout:
    max_q_rows = max(int(caps.max_q_rows), 1)
    page_size = max(int(caps.page_size), 1)
    supertile_tokens = _resolve_compressed_indexer_supertile_tokens(
        int(caps.paged_tile_logits_k_rows)
    )
    if supertile_tokens % page_size != 0:
        raise ValueError(
            "compressed indexer supertile width must be divisible by page_size, "
            f"got supertile_tokens={supertile_tokens}, page_size={page_size}"
        )
    supertile_pages = max(1, supertile_tokens // page_size)
    max_chunks = max(
        1,
        (int(caps.max_page_table_width) + supertile_pages - 1) // supertile_pages,
    )
    num_q_tiles = (max_q_rows + _COMPRESSED_INDEX_TILE_BLOCK_Q - 1) // _COMPRESSED_INDEX_TILE_BLOCK_Q
    num_k_tiles = supertile_tokens // _COMPRESSED_INDEX_TILE_BLOCK_K
    tile_logits_elements = max(
        1,
        num_q_tiles
        * num_k_tiles
        * _COMPRESSED_INDEX_TILE_BLOCK_Q
        * _COMPRESSED_INDEX_TILE_BLOCK_K,
    )

    cursor = 0
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)
    tile_logits_offset_bytes = cursor
    cursor += tile_logits_elements * _dtype_nbytes(torch.float32)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    topk_values_offset_bytes = cursor
    cursor += max_q_rows * int(caps.topk) * _dtype_nbytes(torch.float32)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    topk_indices_offset_bytes = cursor
    cursor += max_q_rows * int(caps.topk) * _dtype_nbytes(torch.int32)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    # Streaming-fold carry double-buffer: two (M, topk) halves ping-pong across
    # supertile chunks (replaces the old max_chunks-deep candidate slab and the
    # merge_positions buffer, both of which the merge step required).
    fold_carry_chunks = 2 if int(max_chunks) > 1 else 0

    candidate_values_offset_bytes = cursor
    cursor += fold_carry_chunks * max_q_rows * int(caps.topk) * _dtype_nbytes(torch.float32)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    candidate_indices_offset_bytes = cursor
    cursor += fold_carry_chunks * max_q_rows * int(caps.topk) * _dtype_nbytes(torch.int32)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    # merge_positions is gone with the merge step; keep the offset for layout
    # compatibility but reserve no bytes.
    merge_positions_offset_bytes = cursor

    active_width_offset_bytes = cursor
    cursor += _dtype_nbytes(torch.int32)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    return _B12XCompressedIndexerScratchLayout(
        nbytes=max(int(cursor), _ARENA_ALIGN_BYTES),
        supertile_tokens=supertile_tokens,
        max_chunks=max_chunks,
        tile_logits_elements=tile_logits_elements,
        tile_logits_offset_bytes=tile_logits_offset_bytes,
        topk_values_offset_bytes=topk_values_offset_bytes,
        topk_indices_offset_bytes=topk_indices_offset_bytes,
        candidate_values_offset_bytes=candidate_values_offset_bytes,
        candidate_indices_offset_bytes=candidate_indices_offset_bytes,
        merge_positions_offset_bytes=merge_positions_offset_bytes,
        active_width_offset_bytes=active_width_offset_bytes,
    )


def _install_compressed_indexer_contract_phantoms(
    scratch: B12XCompressedIndexerScratch,
) -> None:
    storage = scratch.shared_scratch
    scratch._contract_paged_indexer_q_bytes = _shape_only_scratch_tensor(
        storage,
        (
            int(scratch.max_paged_q_rows),
            int(scratch.num_q_heads),
            _COMPRESSED_INDEX_HEAD_DIM,
        ),
        dtype=torch.uint8,
    )
    scratch._contract_paged_indexer_weights = _shape_only_scratch_tensor(
        storage,
        (int(scratch.max_paged_q_rows), int(scratch.num_q_heads)),
        dtype=torch.float32,
    )
    scratch._contract_paged_real_page_table = _shape_only_scratch_tensor(
        storage,
        (int(scratch.max_paged_q_rows), int(scratch.max_page_table_width)),
        dtype=torch.int32,
    )
    scratch._contract_paged_indexer_cache_seqlens = _shape_only_scratch_tensor(
        storage,
        (int(scratch.max_paged_q_rows),),
        dtype=torch.int32,
    )
    scratch._contract_paged_indexer_tile_logits = _shape_only_scratch_tensor(
        storage,
        (int(scratch.indexer_extend_tile_logits.numel()),),
        dtype=torch.float32,
    )
    scratch._contract_paged_indexer_topk_values = _shape_only_scratch_tensor(
        storage,
        (int(scratch.max_paged_q_rows), int(scratch.topk)),
        dtype=torch.float32,
    )
    scratch._contract_paged_indexer_topk_indices = _shape_only_scratch_tensor(
        storage,
        (int(scratch.max_paged_q_rows), int(scratch.topk)),
        dtype=torch.int32,
    )


def _materialize_compressed_indexer_scratch(
    caps: B12XCompressedIndexerScratchCaps,
    scratch_storage: torch.Tensor,
    layout: _B12XCompressedIndexerScratchLayout,
) -> B12XCompressedIndexerScratch:
    max_q_rows = max(int(caps.max_q_rows), 1)
    topk = max(int(caps.topk), 1)
    tile_logits, _ = _materialize_arena_view(
        scratch_storage,
        offset_bytes=layout.tile_logits_offset_bytes,
        shape=(int(layout.tile_logits_elements),),
        dtype=torch.float32,
    )
    topk_values, _ = _materialize_arena_view(
        scratch_storage,
        offset_bytes=layout.topk_values_offset_bytes,
        shape=(max_q_rows, topk),
        dtype=torch.float32,
    )
    topk_indices, _ = _materialize_arena_view(
        scratch_storage,
        offset_bytes=layout.topk_indices_offset_bytes,
        shape=(max_q_rows, topk),
        dtype=torch.int32,
    )
    # Streaming-fold carry double-buffer (two halves) when chunking is possible;
    # otherwise no carry buffer. merge_positions is gone with the merge step.
    fold_carry_chunks = 2 if int(layout.max_chunks) > 1 else 0
    if fold_carry_chunks:
        candidate_values, _ = _materialize_arena_view(
            scratch_storage,
            offset_bytes=layout.candidate_values_offset_bytes,
            shape=(fold_carry_chunks, max_q_rows, topk),
            dtype=torch.float32,
        )
        candidate_indices, _ = _materialize_arena_view(
            scratch_storage,
            offset_bytes=layout.candidate_indices_offset_bytes,
            shape=(fold_carry_chunks, max_q_rows, topk),
            dtype=torch.int32,
        )
    else:
        candidate_values = None
        candidate_indices = None
    merge_positions = None
    active_width_cap, _ = _materialize_arena_view(
        scratch_storage,
        offset_bytes=layout.active_width_offset_bytes,
        shape=(1,),
        dtype=torch.int32,
    )
    width_cap = max(
        int(caps.max_page_table_width) * int(caps.page_size),
        int(layout.supertile_tokens),
        1,
    )
    active_width_cap.fill_(int(width_cap))

    scratch = B12XCompressedIndexerScratch(
        shared_scratch=scratch_storage,
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=caps.num_q_heads,
        topk=caps.topk,
        max_page_table_width=caps.max_page_table_width,
        max_total_q=caps.max_q_rows,
        max_paged_q_rows=caps.max_q_rows,
        max_batch=caps.max_batch,
        page_size=caps.page_size,
        paged_tile_logits_k_rows=layout.supertile_tokens,
        max_chunks=layout.max_chunks,
        indexer_extend_tile_logits=tile_logits,
        indexer_extend_topk_values=topk_values,
        indexer_extend_topk_indices=topk_indices,
        indexer_extend_candidate_values=candidate_values,
        indexer_extend_candidate_indices=candidate_indices,
        indexer_extend_topk_positions=merge_positions,
        paged_indexer_active_width_cap=active_width_cap,
    )
    _install_compressed_indexer_contract_phantoms(scratch)
    return scratch


def _materialize_compressed_mla_scratch(
    caps: B12XCompressedMLAScratchCaps,
    scratch_storage: torch.Tensor,
    layout: _B12XCompressedMLAScratchLayout,
) -> B12XCompressedMLAScratch:
    max_total_q = max(int(caps.max_q_rows), 1)
    tmp_output, _ = _materialize_arena_strided_view(
        scratch_storage,
        offset_bytes=layout.tmp_output_offset_bytes,
        shape=(
            max_total_q,
            int(caps.num_q_heads),
            int(caps.max_chunks_per_row),
            int(caps.v_head_dim),
        ),
        stride=_split_tmp_output_stride(
            max_total_q=max_total_q,
            num_q_heads=int(caps.num_q_heads),
            max_chunks_per_row=int(caps.max_chunks_per_row),
            v_head_dim=int(caps.v_head_dim),
        ),
        dtype=caps.dtype,
    )
    tmp_lse, _ = _materialize_arena_view(
        scratch_storage,
        offset_bytes=layout.tmp_lse_offset_bytes,
        shape=(max_total_q, int(caps.num_q_heads), int(caps.max_chunks_per_row)),
        dtype=torch.float32,
    )
    final_lse, _ = _materialize_arena_view(
        scratch_storage,
        offset_bytes=layout.final_lse_offset_bytes,
        shape=(max_total_q, int(caps.num_q_heads)),
        dtype=torch.float32,
    )
    kv_chunk_size_ptr, _ = _materialize_arena_view(
        scratch_storage,
        offset_bytes=layout.kv_chunk_size_offset_bytes,
        shape=(1,),
        dtype=torch.int32,
    )
    num_chunks_ptr, _ = _materialize_arena_view(
        scratch_storage,
        offset_bytes=layout.num_chunks_offset_bytes,
        shape=(1,),
        dtype=torch.int32,
    )
    sm_scale_tensor, _ = _materialize_arena_view(
        scratch_storage,
        offset_bytes=layout.sm_scale_offset_bytes,
        shape=(1,),
        dtype=torch.float32,
    )
    scratch = B12XCompressedMLAScratch(
        shared_scratch=scratch_storage,
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=caps.num_q_heads,
        head_dim=caps.head_dim,
        v_head_dim=caps.v_head_dim,
        topk=caps.max_width,
        max_page_table_width=caps.max_page_table_width,
        max_total_q=caps.max_q_rows,
        max_batch=caps.max_batch,
        max_kv_rows=caps.max_kv_rows,
        max_chunks_per_row=caps.max_chunks_per_row,
        page_size=caps.page_size,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        output_buffer=_split_output_buffer_from_tmp(tmp_output),
        final_lse=final_lse,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        sm_scale_tensor=sm_scale_tensor,
    )
    _install_compressed_mla_contract_phantoms(scratch)
    split_cfg = compressed_mla_split_config_for_contract(
        rows=caps.max_q_rows,
        width=caps.max_width,
        max_chunks=caps.max_chunks_per_row,
    )
    scratch.set_split_chunk_config(
        kv_chunk_size=split_cfg.chunk_size,
        num_chunks=split_cfg.num_chunks,
    )
    return scratch


def _validate_device(
    tensor: torch.Tensor,
    *,
    scratch: object | None = None,
    workspace: B12XAttentionWorkspace | None = None,
    name: str,
) -> None:
    resource = scratch if scratch is not None else workspace
    if resource is None:
        raise TypeError("_validate_device requires scratch or workspace")
    if tensor.device != resource.device:
        raise ValueError(f"{name} device {tensor.device} does not match resource device {resource.device}")


def _normalize_q(q: torch.Tensor, *, scratch: object) -> torch.Tensor:
    if q.ndim == 4 and q.shape[1] == 1:
        q = q[:, 0]
    if q.ndim != 3:
        raise ValueError(f"q must be rank-3 or [rows, 1, heads, dim], got {tuple(q.shape)}")
    if int(q.shape[1]) != int(scratch.num_q_heads):
        raise ValueError(f"q heads {int(q.shape[1])} do not match scratch heads {scratch.num_q_heads}")
    if int(q.shape[2]) != COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(f"q head_dim must be {COMPRESSED_MLA_HEAD_DIM}, got {int(q.shape[2])}")
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must have dtype torch.bfloat16, got {q.dtype}")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    _validate_device(q, scratch=scratch, name="q")
    if int(q.shape[0]) > int(scratch.max_total_q):
        raise ValueError(f"q rows {int(q.shape[0])} exceed scratch capacity {scratch.max_total_q}")
    return q.detach()


def _is_row_shared_i32_matrix(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 and int(tensor.stride(0)) == 0 and int(tensor.stride(1)) == 1


def _normalize_i32_matrix(
    tensor: torch.Tensor,
    *,
    scratch: object,
    rows: int,
    name: str,
    allow_row_shared: bool = False,
) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2 or [rows, 1, width], got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous() and not (allow_row_shared and _is_row_shared_i32_matrix(tensor)):
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, scratch=scratch, name=name)
    if int(tensor.shape[0]) != int(rows):
        raise ValueError(f"{name} rows {int(tensor.shape[0])} do not match q rows {rows}")
    return tensor


def _validate_i32_vector(tensor: torch.Tensor, *, scratch: object, rows: int, name: str) -> torch.Tensor:
    if tensor.shape != (int(rows),):
        raise ValueError(f"{name} must have shape ({rows},), got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, scratch=scratch, name=name)
    return tensor


def build_compressed_mla_binding(
    *,
    scratch: object | None = None,
    workspace: B12XAttentionWorkspace | None = None,
    q: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_indices: torch.Tensor | None = None,
    indexed_lengths: torch.Tensor | None = None,
    indexed_page_table: torch.Tensor | None = None,
) -> B12XCompressedMLABinding:
    if scratch is None:
        if workspace is None:
            raise TypeError("build_compressed_mla_binding requires scratch or workspace")
        scratch = workspace
    elif workspace is not None and workspace is not scratch:
        raise ValueError("scratch and workspace refer to different compressed MLA resources")

    q = _normalize_q(q, scratch=scratch)
    rows = int(q.shape[0])
    swa_indices = _normalize_i32_matrix(
        swa_indices,
        scratch=scratch,
        rows=rows,
        name="swa_indices",
    )
    if int(swa_indices.shape[1]) > int(scratch.topk):
        raise ValueError(f"swa_indices width {int(swa_indices.shape[1])} exceeds scratch topk {scratch.topk}")
    swa_lengths = _validate_i32_vector(
        swa_lengths,
        scratch=scratch,
        rows=rows,
        name="swa_lengths",
    )
    if (indexed_indices is None) != (indexed_lengths is None):
        raise ValueError("indexed_indices and indexed_lengths must be provided together")
    indexed_width = 0
    if indexed_indices is not None:
        indexed_indices = _normalize_i32_matrix(
            indexed_indices,
            scratch=scratch,
            rows=rows,
            name="indexed_indices",
        )
        indexed_width = int(indexed_indices.shape[1])
        indexed_lengths = _validate_i32_vector(
            indexed_lengths,  # type: ignore[arg-type]
            scratch=scratch,
            rows=rows,
            name="indexed_lengths",
        )
    if indexed_page_table is not None:
        indexed_page_table = _normalize_i32_matrix(
            indexed_page_table,
            scratch=scratch,
            rows=rows,
            name="indexed_page_table",
            allow_row_shared=True,
        )
        if int(indexed_page_table.shape[1]) > int(scratch.max_page_table_width):
            raise ValueError(
                "indexed_page_table width "
                f"{int(indexed_page_table.shape[1])} exceeds scratch capacity {scratch.max_page_table_width}"
            )
    total_width = int(swa_indices.shape[1]) + indexed_width
    if total_width > int(scratch.topk):
        raise ValueError(f"compressed MLA width {total_width} exceeds scratch topk {scratch.topk}")
    return B12XCompressedMLABinding(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
    )


def _validate_i32_contiguous(
    tensor: torch.Tensor,
    *,
    scratch: object | None = None,
    workspace: B12XAttentionWorkspace | None = None,
    name: str,
    ndim: int,
) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank-{ndim}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, scratch=scratch, workspace=workspace, name=name)


def build_compressed_indexer_binding(
    *,
    scratch: object | None = None,
    workspace: B12XAttentionWorkspace | None = None,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    active_width: torch.Tensor | None = None,
    schedule_metadata: torch.Tensor | None = None,
    expected_num_q_heads: int | None = None,
    shared_page_table: bool = False,
) -> B12XCompressedIndexerBinding:
    if scratch is None:
        if workspace is None:
            raise TypeError("build_compressed_indexer_binding requires scratch or workspace")
        scratch = workspace
    elif workspace is not None and workspace is not scratch:
        raise ValueError("scratch and workspace refer to different compressed indexer resources")

    _validate_i32_contiguous(
        real_page_table,
        scratch=scratch,
        name="real_page_table",
        ndim=2,
    )
    _validate_i32_contiguous(
        cache_seqlens_int32,
        scratch=scratch,
        name="cache_seqlens_int32",
        ndim=1,
    )
    if active_width is None:
        active_width = scratch.get_paged_indexer_active_width_cap()
    _validate_i32_contiguous(
        active_width,
        scratch=scratch,
        name="active_width",
        ndim=1,
    )
    if active_width.shape != (1,):
        raise ValueError(f"active_width must have shape (1,), got {tuple(active_width.shape)}")
    if int(real_page_table.shape[0]) != int(cache_seqlens_int32.shape[0]):
        raise ValueError(
            f"real_page_table rows {int(real_page_table.shape[0])} do not match "
            f"cache_seqlens_int32 rows {int(cache_seqlens_int32.shape[0])}"
        )
    if int(real_page_table.shape[0]) > int(scratch.max_paged_q_rows):
        raise ValueError(
            f"real_page_table rows {int(real_page_table.shape[0])} exceed workspace paged capacity "
            f"{scratch.max_paged_q_rows}"
        )
    if int(real_page_table.shape[1]) > int(scratch.max_page_table_width):
        raise ValueError(
            f"real_page_table width {int(real_page_table.shape[1])} exceeds workspace capacity "
            f"{scratch.max_page_table_width}"
        )
    if schedule_metadata is not None:
        _validate_i32_contiguous(
            schedule_metadata,
            scratch=scratch,
            name="schedule_metadata",
            ndim=2,
        )
        if int(schedule_metadata.shape[1]) != 2:
            raise ValueError(f"schedule_metadata must have shape (num_sms + 1, 2), got {tuple(schedule_metadata.shape)}")
    if expected_num_q_heads is not None:
        expected_num_q_heads = int(expected_num_q_heads)
        if expected_num_q_heads <= 0:
            raise ValueError(f"expected_num_q_heads must be positive, got {expected_num_q_heads}")
    return B12XCompressedIndexerBinding(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens_int32,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=expected_num_q_heads,
        shared_page_table=bool(shared_page_table),
    )


@dataclass(frozen=True)
class B12XCompressedMLAScratchPlan:
    caps: B12XCompressedMLAScratchCaps
    layout: _B12XCompressedMLAScratchLayout
    _scratch_specs: tuple[B12XScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[B12XScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        q: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        indexed_indices: torch.Tensor | None = None,
        indexed_lengths: torch.Tensor | None = None,
        indexed_page_table: torch.Tensor | None = None,
    ) -> B12XCompressedMLABinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="compressed MLA",
        )
        scratch_views = _materialize_compressed_mla_scratch(
            self.caps,
            scratch_storage,
            self.layout,
        )
        return build_compressed_mla_binding(
            scratch=scratch_views,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_table=indexed_page_table,
        )


@dataclass(frozen=True)
class B12XCompressedIndexerScratchPlan:
    caps: B12XCompressedIndexerScratchCaps
    layout: _B12XCompressedIndexerScratchLayout
    _scratch_specs: tuple[B12XScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[B12XScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        real_page_table: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        active_width: torch.Tensor | None = None,
        schedule_metadata: torch.Tensor | None = None,
        expected_num_q_heads: int | None = None,
        shared_page_table: bool = False,
    ) -> B12XCompressedIndexerBinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="compressed indexer",
        )
        scratch_views = _materialize_compressed_indexer_scratch(
            self.caps,
            scratch_storage,
            self.layout,
        )
        return build_compressed_indexer_binding(
            scratch=scratch_views,
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            active_width=active_width,
            schedule_metadata=schedule_metadata,
            expected_num_q_heads=expected_num_q_heads,
            shared_page_table=shared_page_table,
        )


def plan_compressed_mla_scratch(
    caps: B12XCompressedMLAScratchCaps,
) -> B12XCompressedMLAScratchPlan:
    layout = _compressed_mla_scratch_layout(caps)
    return B12XCompressedMLAScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "compressed_mla.scratch",
                nbytes=int(layout.nbytes),
                device=caps.device,
            ),
        ),
    )


def plan_compressed_indexer_scratch(
    caps: B12XCompressedIndexerScratchCaps,
) -> B12XCompressedIndexerScratchPlan:
    layout = _compressed_indexer_scratch_layout(caps)
    return B12XCompressedIndexerScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "compressed_indexer.scratch",
                nbytes=int(layout.nbytes),
                device=caps.device,
            ),
        ),
    )


__all__ = [
    "B12XScratchBufferSpec",
    "B12XCompressedIndexerBinding",
    "B12XCompressedIndexerScratch",
    "B12XCompressedIndexerScratchCaps",
    "B12XCompressedIndexerScratchPlan",
    "B12XCompressedMLABinding",
    "B12XCompressedMLAScratch",
    "B12XCompressedMLAScratchCaps",
    "B12XCompressedMLAScratchPlan",
    "build_compressed_indexer_binding",
    "build_compressed_mla_binding",
    "plan_compressed_indexer_scratch",
    "plan_compressed_mla_scratch",
]
