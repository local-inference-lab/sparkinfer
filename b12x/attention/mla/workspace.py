"""Workspace state for sparse MLA execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .split import default_sparse_mla_split_decode_config_for_width
from .reference import _MLA_PACKED_DIM


_INDEX_HEAD_DIM = 128
_NSA_INDEXER_BLOCK_K = 64
_NSA_INDEXER_PREFILL_BLOCK_K = 256
_NSA_INDEXER_TILE_BLOCK_Q = 32
_ARENA_ALIGN_BYTES = 1024


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _shape_only_cuda_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a tiny CUDA tensor whose shape/stride/dtype/device are stable.

    Used as a phantom in host-launcher cache keys so that varying batch sizes
    do not trigger CUTLASS recompilation.  The tensor is never read by kernels.
    """
    base = torch.empty(1, dtype=dtype, device=device)
    return base.as_strided(shape, (0,) * len(shape))


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + alignment - 1) // alignment) * alignment


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _resolve_extend_topk_supertile_k(value: int) -> int:
    value = int(value)
    if value <= 0:
        return 0
    return _align_up(value, _NSA_INDEXER_BLOCK_K)


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _materialize_arena_view(
    arena: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    offset_bytes = _align_up(offset_bytes, max(_ARENA_ALIGN_BYTES, _dtype_nbytes(dtype)))
    nbytes = _shape_numel(shape) * _dtype_nbytes(dtype)
    view_bytes = arena.narrow(0, offset_bytes, nbytes)
    typed_view = view_bytes.view(dtype).view(shape)
    return typed_view, offset_bytes + nbytes


def _encode_indexer_k_tma_descriptor(
    k_quant_bytes: torch.Tensor,
    *,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a stable TMA descriptor for the fixed NSA indexer K workspace."""
    if k_quant_bytes.ndim != 2 or k_quant_bytes.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"k_quant_bytes must have shape (rows, {_INDEX_HEAD_DIM}), got {tuple(k_quant_bytes.shape)}"
        )
    if k_quant_bytes.dtype != torch.uint8:
        raise TypeError(f"k_quant_bytes must be dtype torch.uint8, got {k_quant_bytes.dtype}")

    import cuda.bindings.driver as cuda

    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    row_bytes = int(k_quant_bytes.stride(0)) * k_quant_bytes.element_size()
    base_ptr = int(k_quant_bytes.data_ptr())
    total_rows = int(k_quant_bytes.shape[0])

    result, tensor_map = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        base_ptr,
        [U64(_INDEX_HEAD_DIM), U64(total_rows)],
        [U64(row_bytes)],
        [U32(_INDEX_HEAD_DIM), U32(int(block_k))],
        [U32(1), U32(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed for NSA indexer K workspace: {result}")

    desc = torch.tensor(
        [int(word) for word in tensor_map.opaque],
        dtype=torch.uint64,
        device=k_quant_bytes.device,
    )
    desc_ptrs = torch.tensor([int(desc.data_ptr())], dtype=torch.int64, device=k_quant_bytes.device)
    return desc, desc_ptrs


B12XWorkspaceMode = Literal["decode", "extend", "verify", "draft_extend"]


@dataclass(frozen=True, kw_only=True)
class B12XAttentionArenaCaps:
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    indexer_num_q_heads: int
    head_dim: int
    max_v_head_dim: int
    topk: int
    max_page_table_width: int
    extend_max_total_q: int
    extend_max_batch: int
    extend_max_kv_rows: int
    paged_max_q_rows: int
    paged_max_batch: int
    page_size: int = 64
    padded_heads: int = 128
    max_chunks_per_row: int = 64
    reserve_extend_indexer_logits: bool = True
    extend_indexer_tile_logits_k_rows: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(
            self,
            "indexer_num_q_heads",
            max(int(self.indexer_num_q_heads), 1),
        )
        object.__setattr__(self, "head_dim", max(int(self.head_dim), 1))
        object.__setattr__(self, "max_v_head_dim", max(int(self.max_v_head_dim), 1))
        object.__setattr__(self, "topk", max(int(self.topk), 1))
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(self.max_page_table_width), 1),
        )
        object.__setattr__(
            self,
            "extend_max_total_q",
            max(int(self.extend_max_total_q), 1),
        )
        object.__setattr__(
            self,
            "extend_max_batch",
            max(int(self.extend_max_batch), 1),
        )
        object.__setattr__(
            self,
            "extend_max_kv_rows",
            max(int(self.extend_max_kv_rows), 0),
        )
        object.__setattr__(
            self,
            "paged_max_q_rows",
            max(int(self.paged_max_q_rows), 1),
        )
        object.__setattr__(
            self,
            "paged_max_batch",
            max(int(self.paged_max_batch), 1),
        )
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))
        object.__setattr__(self, "padded_heads", max(int(self.padded_heads), 1))
        object.__setattr__(
            self,
            "max_chunks_per_row",
            max(int(self.max_chunks_per_row), 1),
        )
        object.__setattr__(
            self,
            "extend_indexer_tile_logits_k_rows",
            _resolve_extend_topk_supertile_k(self.extend_indexer_tile_logits_k_rows),
        )


@dataclass(frozen=True, kw_only=True)
class B12XAttentionWorkspaceContract:
    mode: B12XWorkspaceMode
    max_total_q: int
    max_batch: int
    max_paged_q_rows: int
    max_kv_rows: int
    v_head_dim: int
    indexer_num_q_heads: int
    max_page_table_width: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_total_q", max(int(self.max_total_q), 1))
        object.__setattr__(self, "max_batch", max(int(self.max_batch), 1))
        object.__setattr__(
            self,
            "max_paged_q_rows",
            max(int(self.max_paged_q_rows), 1),
        )
        object.__setattr__(self, "max_kv_rows", max(int(self.max_kv_rows), 0))
        object.__setattr__(self, "v_head_dim", max(int(self.v_head_dim), 1))
        object.__setattr__(
            self,
            "indexer_num_q_heads",
            max(int(self.indexer_num_q_heads), 1),
        )
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(self.max_page_table_width), 1),
        )


@dataclass(frozen=True, kw_only=True)
class _B12XAttentionArenaLayout:
    arena_nbytes: int
    mla_phase_nbytes: int
    indexer_phase_nbytes: int
    ragged_kv_nbytes: int
    indexer_logits_nbytes: int
    indexer_extend_logits_nbytes: int
    indexer_extend_tile_logits_nbytes: int
    indexer_extend_topk_indices_nbytes: int
    indexer_extend_topk_values_nbytes: int
    indexer_extend_topk_scratch_indices_nbytes: int
    indexer_extend_topk_scratch_values_nbytes: int
    indexer_extend_candidate_values_nbytes: int
    indexer_extend_candidate_indices_nbytes: int
    indexer_extend_lengths_nbytes: int
    indexer_extend_mapped_indices_nbytes: int
    indexer_paged_logits_nbytes: int
    ragged_kv_offset_bytes: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int
    indexer_k_quant_offset_bytes: int
    indexer_k_scale_offset_bytes: int
    indexer_extend_logits_offset_bytes: int
    indexer_extend_tile_logits_offset_bytes: int
    indexer_extend_topk_indices_offset_bytes: int
    indexer_extend_topk_values_offset_bytes: int
    indexer_extend_topk_scratch_indices_offset_bytes: int
    indexer_extend_topk_scratch_values_offset_bytes: int
    indexer_extend_candidate_values_offset_bytes: int
    indexer_extend_candidate_indices_offset_bytes: int
    indexer_extend_lengths_offset_bytes: int
    indexer_extend_mapped_indices_offset_bytes: int
    indexer_paged_logits_offset_bytes: int


@dataclass(kw_only=True)
class B12XAttentionArena:
    caps: B12XAttentionArenaCaps
    shared_arena: torch.Tensor
    shared_arena_nbytes: int
    mla_phase_nbytes: int
    indexer_phase_nbytes: int
    ragged_kv_nbytes: int
    indexer_logits_nbytes: int
    indexer_extend_logits_nbytes: int
    indexer_extend_tile_logits_nbytes: int
    indexer_extend_topk_indices_nbytes: int
    indexer_extend_topk_values_nbytes: int
    indexer_extend_topk_scratch_indices_nbytes: int
    indexer_extend_topk_scratch_values_nbytes: int
    indexer_extend_candidate_values_nbytes: int
    indexer_extend_candidate_indices_nbytes: int
    indexer_extend_lengths_nbytes: int
    indexer_extend_mapped_indices_nbytes: int
    indexer_paged_logits_nbytes: int
    ragged_kv_offset_bytes: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int
    indexer_k_quant_offset_bytes: int
    indexer_k_scale_offset_bytes: int
    indexer_extend_logits_offset_bytes: int
    indexer_extend_tile_logits_offset_bytes: int
    indexer_extend_topk_indices_offset_bytes: int
    indexer_extend_topk_values_offset_bytes: int
    indexer_extend_topk_scratch_indices_offset_bytes: int
    indexer_extend_topk_scratch_values_offset_bytes: int
    indexer_extend_candidate_values_offset_bytes: int
    indexer_extend_candidate_indices_offset_bytes: int
    indexer_extend_lengths_offset_bytes: int
    indexer_extend_mapped_indices_offset_bytes: int
    indexer_paged_logits_offset_bytes: int

    @classmethod
    def _layout(cls, caps: B12XAttentionArenaCaps) -> _B12XAttentionArenaLayout:
        max_total_q = max(int(caps.extend_max_total_q), int(caps.paged_max_q_rows), 1)
        max_paged_q_rows = max(int(caps.paged_max_q_rows), 1)
        max_kv_rows = max(int(caps.extend_max_kv_rows), 1)
        indexer_k_rows = _align_up(max_kv_rows, _NSA_INDEXER_BLOCK_K)
        paged_width_tokens = max(
            int(caps.max_page_table_width) * int(caps.page_size),
            1,
        )

        mla_offset = 0
        mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
        ragged_kv_offset_bytes = mla_offset
        mla_offset += max_kv_rows * _MLA_PACKED_DIM * _dtype_nbytes(caps.kv_dtype)
        mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
        tmp_output_offset_bytes = mla_offset
        mla_offset += (
            max_total_q
            * int(caps.num_q_heads)
            * int(caps.max_chunks_per_row)
            * int(caps.max_v_head_dim)
            * _dtype_nbytes(caps.dtype)
        )
        mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
        tmp_lse_offset_bytes = mla_offset
        mla_offset += (
            max_total_q
            * int(caps.num_q_heads)
            * int(caps.max_chunks_per_row)
            * _dtype_nbytes(torch.float32)
        )
        mla_phase_nbytes = int(mla_offset)

        extend_offset = 0
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_k_quant_offset_bytes = extend_offset
        extend_offset += indexer_k_rows * _INDEX_HEAD_DIM
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_k_scale_offset_bytes = extend_offset
        extend_offset += indexer_k_rows * _dtype_nbytes(torch.float32)
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_logits_offset_bytes = extend_offset
        if caps.reserve_extend_indexer_logits:
            extend_logits_nbytes = (
                int(caps.extend_max_total_q)
                * indexer_k_rows
                * _dtype_nbytes(torch.float32)
            )
        else:
            extend_logits_nbytes = 0
        extend_offset += extend_logits_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_tile_logits_offset_bytes = extend_offset
        extend_tile_logits_k_rows = min(
            indexer_k_rows,
            _resolve_extend_topk_supertile_k(caps.extend_indexer_tile_logits_k_rows),
        )
        if extend_tile_logits_k_rows:
            extend_tile_logits_q_rows = _align_up(
                int(caps.extend_max_total_q),
                _NSA_INDEXER_TILE_BLOCK_Q,
            )
            extend_tile_logits_nbytes = (
                extend_tile_logits_q_rows
                * extend_tile_logits_k_rows
                * _dtype_nbytes(torch.float32)
            )
            extend_candidate_chunks = (
                indexer_k_rows + extend_tile_logits_k_rows - 1
            ) // extend_tile_logits_k_rows
        else:
            extend_tile_logits_nbytes = 0
            extend_candidate_chunks = 0
        extend_offset += extend_tile_logits_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_topk_indices_offset_bytes = extend_offset
        extend_topk_indices_nbytes = (
            int(caps.extend_max_total_q)
            * int(caps.topk)
            * _dtype_nbytes(torch.int32)
        )
        extend_offset += extend_topk_indices_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_topk_values_offset_bytes = extend_offset
        extend_topk_values_nbytes = (
            int(caps.extend_max_total_q)
            * int(caps.topk)
            * _dtype_nbytes(torch.float32)
        )
        extend_offset += extend_topk_values_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_topk_scratch_indices_offset_bytes = extend_offset
        extend_topk_scratch_indices_nbytes = (
            int(caps.extend_max_total_q)
            * int(caps.topk)
            * _dtype_nbytes(torch.int32)
        )
        extend_offset += extend_topk_scratch_indices_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_topk_scratch_values_offset_bytes = extend_offset
        extend_topk_scratch_values_nbytes = (
            int(caps.extend_max_total_q)
            * int(caps.topk)
            * _dtype_nbytes(torch.float32)
        )
        extend_offset += extend_topk_scratch_values_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_candidate_values_offset_bytes = extend_offset
        extend_candidate_values_nbytes = (
            int(extend_candidate_chunks)
            * int(caps.extend_max_total_q)
            * int(caps.topk)
            * _dtype_nbytes(torch.float32)
        )
        extend_offset += extend_candidate_values_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_candidate_indices_offset_bytes = extend_offset
        extend_candidate_indices_nbytes = (
            int(extend_candidate_chunks)
            * int(caps.extend_max_total_q)
            * int(caps.topk)
            * _dtype_nbytes(torch.int32)
        )
        extend_offset += extend_candidate_indices_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_lengths_offset_bytes = extend_offset
        extend_lengths_nbytes = int(caps.extend_max_total_q) * _dtype_nbytes(torch.int32)
        extend_offset += extend_lengths_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_extend_mapped_indices_offset_bytes = extend_offset
        extend_mapped_indices_nbytes = (
            int(caps.extend_max_total_q)
            * int(caps.topk)
            * _dtype_nbytes(torch.int32)
        )
        extend_offset += extend_mapped_indices_nbytes

        paged_offset = 0
        paged_offset = _align_up(paged_offset, _ARENA_ALIGN_BYTES)
        indexer_paged_logits_offset_bytes = paged_offset
        paged_logits_nbytes = (
            max_paged_q_rows * paged_width_tokens * _dtype_nbytes(torch.float32)
        )
        paged_offset += paged_logits_nbytes

        indexer_phase_nbytes = int(max(extend_offset, paged_offset))
        arena_nbytes = max(mla_phase_nbytes, indexer_phase_nbytes, 1)
        ragged_kv_nbytes = max_kv_rows * _MLA_PACKED_DIM * _dtype_nbytes(caps.kv_dtype)
        return _B12XAttentionArenaLayout(
            arena_nbytes=int(arena_nbytes),
            mla_phase_nbytes=mla_phase_nbytes,
            indexer_phase_nbytes=indexer_phase_nbytes,
            ragged_kv_nbytes=ragged_kv_nbytes,
            indexer_logits_nbytes=max(
                extend_logits_nbytes,
                extend_tile_logits_nbytes,
                extend_topk_indices_nbytes,
                extend_topk_values_nbytes,
                extend_topk_scratch_indices_nbytes,
                extend_topk_scratch_values_nbytes,
                extend_candidate_values_nbytes,
                extend_candidate_indices_nbytes,
                extend_lengths_nbytes,
                extend_mapped_indices_nbytes,
                paged_logits_nbytes,
            ),
            indexer_extend_logits_nbytes=extend_logits_nbytes,
            indexer_extend_tile_logits_nbytes=extend_tile_logits_nbytes,
            indexer_extend_topk_indices_nbytes=extend_topk_indices_nbytes,
            indexer_extend_topk_values_nbytes=extend_topk_values_nbytes,
            indexer_extend_topk_scratch_indices_nbytes=extend_topk_scratch_indices_nbytes,
            indexer_extend_topk_scratch_values_nbytes=extend_topk_scratch_values_nbytes,
            indexer_extend_candidate_values_nbytes=extend_candidate_values_nbytes,
            indexer_extend_candidate_indices_nbytes=extend_candidate_indices_nbytes,
            indexer_extend_lengths_nbytes=extend_lengths_nbytes,
            indexer_extend_mapped_indices_nbytes=extend_mapped_indices_nbytes,
            indexer_paged_logits_nbytes=paged_logits_nbytes,
            ragged_kv_offset_bytes=ragged_kv_offset_bytes,
            tmp_output_offset_bytes=tmp_output_offset_bytes,
            tmp_lse_offset_bytes=tmp_lse_offset_bytes,
            indexer_k_quant_offset_bytes=indexer_k_quant_offset_bytes,
            indexer_k_scale_offset_bytes=indexer_k_scale_offset_bytes,
            indexer_extend_logits_offset_bytes=indexer_extend_logits_offset_bytes,
            indexer_extend_tile_logits_offset_bytes=indexer_extend_tile_logits_offset_bytes,
            indexer_extend_topk_indices_offset_bytes=indexer_extend_topk_indices_offset_bytes,
            indexer_extend_topk_values_offset_bytes=indexer_extend_topk_values_offset_bytes,
            indexer_extend_topk_scratch_indices_offset_bytes=indexer_extend_topk_scratch_indices_offset_bytes,
            indexer_extend_topk_scratch_values_offset_bytes=indexer_extend_topk_scratch_values_offset_bytes,
            indexer_extend_candidate_values_offset_bytes=indexer_extend_candidate_values_offset_bytes,
            indexer_extend_candidate_indices_offset_bytes=indexer_extend_candidate_indices_offset_bytes,
            indexer_extend_lengths_offset_bytes=indexer_extend_lengths_offset_bytes,
            indexer_extend_mapped_indices_offset_bytes=indexer_extend_mapped_indices_offset_bytes,
            indexer_paged_logits_offset_bytes=indexer_paged_logits_offset_bytes,
        )

    @classmethod
    def _build(
        cls,
        caps: B12XAttentionArenaCaps,
        *,
        shared_arena: torch.Tensor | None,
        storage: str,
    ) -> "B12XAttentionArena":
        layout = cls._layout(caps)
        if shared_arena is None:
            shared_arena = torch.empty(
                (layout.arena_nbytes,),
                dtype=torch.uint8,
                device=caps.device,
            )
        elif shared_arena.dtype != torch.uint8:
            raise TypeError(f"shared_arena must have dtype torch.uint8, got {shared_arena.dtype}")
        elif shared_arena.device != caps.device:
            raise ValueError(f"shared_arena device {shared_arena.device} does not match caps device {caps.device}")
        elif shared_arena.numel() < layout.arena_nbytes:
            raise ValueError(
                f"shared_arena has {shared_arena.numel()} bytes, but attention arena requires {layout.arena_nbytes}"
            )
        arena = cls(
            caps=caps,
            shared_arena=shared_arena,
            shared_arena_nbytes=layout.arena_nbytes,
            mla_phase_nbytes=layout.mla_phase_nbytes,
            indexer_phase_nbytes=layout.indexer_phase_nbytes,
            ragged_kv_nbytes=layout.ragged_kv_nbytes,
            indexer_logits_nbytes=layout.indexer_logits_nbytes,
            indexer_extend_logits_nbytes=layout.indexer_extend_logits_nbytes,
            indexer_extend_tile_logits_nbytes=layout.indexer_extend_tile_logits_nbytes,
            indexer_extend_topk_indices_nbytes=layout.indexer_extend_topk_indices_nbytes,
            indexer_extend_topk_values_nbytes=layout.indexer_extend_topk_values_nbytes,
            indexer_extend_topk_scratch_indices_nbytes=layout.indexer_extend_topk_scratch_indices_nbytes,
            indexer_extend_topk_scratch_values_nbytes=layout.indexer_extend_topk_scratch_values_nbytes,
            indexer_extend_candidate_values_nbytes=layout.indexer_extend_candidate_values_nbytes,
            indexer_extend_candidate_indices_nbytes=layout.indexer_extend_candidate_indices_nbytes,
            indexer_extend_lengths_nbytes=layout.indexer_extend_lengths_nbytes,
            indexer_extend_mapped_indices_nbytes=layout.indexer_extend_mapped_indices_nbytes,
            indexer_paged_logits_nbytes=layout.indexer_paged_logits_nbytes,
            ragged_kv_offset_bytes=layout.ragged_kv_offset_bytes,
            tmp_output_offset_bytes=layout.tmp_output_offset_bytes,
            tmp_lse_offset_bytes=layout.tmp_lse_offset_bytes,
            indexer_k_quant_offset_bytes=layout.indexer_k_quant_offset_bytes,
            indexer_k_scale_offset_bytes=layout.indexer_k_scale_offset_bytes,
            indexer_extend_logits_offset_bytes=layout.indexer_extend_logits_offset_bytes,
            indexer_extend_tile_logits_offset_bytes=layout.indexer_extend_tile_logits_offset_bytes,
            indexer_extend_topk_indices_offset_bytes=layout.indexer_extend_topk_indices_offset_bytes,
            indexer_extend_topk_values_offset_bytes=layout.indexer_extend_topk_values_offset_bytes,
            indexer_extend_topk_scratch_indices_offset_bytes=layout.indexer_extend_topk_scratch_indices_offset_bytes,
            indexer_extend_topk_scratch_values_offset_bytes=layout.indexer_extend_topk_scratch_values_offset_bytes,
            indexer_extend_candidate_values_offset_bytes=layout.indexer_extend_candidate_values_offset_bytes,
            indexer_extend_candidate_indices_offset_bytes=layout.indexer_extend_candidate_indices_offset_bytes,
            indexer_extend_lengths_offset_bytes=layout.indexer_extend_lengths_offset_bytes,
            indexer_extend_mapped_indices_offset_bytes=layout.indexer_extend_mapped_indices_offset_bytes,
            indexer_paged_logits_offset_bytes=layout.indexer_paged_logits_offset_bytes,
        )
        return arena

    @classmethod
    def allocate(cls, caps: B12XAttentionArenaCaps) -> "B12XAttentionArena":
        return cls._build(caps, shared_arena=None, storage="standalone")

    @classmethod
    def from_shared_arena(
        cls,
        caps: B12XAttentionArenaCaps,
        shared_arena: torch.Tensor,
    ) -> "B12XAttentionArena":
        """Materialize an attention arena over caller-owned uint8 storage."""
        return cls._build(caps, shared_arena=shared_arena, storage="shared")

    @classmethod
    def required_nbytes(cls, caps: B12XAttentionArenaCaps) -> int:
        """Return the backing-store byte requirement without retaining storage."""
        return cls._layout(caps).arena_nbytes

    def make_workspace(
        self,
        contract: B12XAttentionWorkspaceContract,
        *,
        use_cuda_graph: bool = False,
    ) -> "B12XAttentionWorkspace":
        if contract.v_head_dim > self.caps.max_v_head_dim:
            raise ValueError(
                f"workspace v_head_dim {contract.v_head_dim} exceeds arena max_v_head_dim {self.caps.max_v_head_dim}"
            )
        if contract.max_total_q > self.caps.extend_max_total_q and contract.max_total_q > self.caps.paged_max_q_rows:
            raise ValueError(
                f"workspace max_total_q {contract.max_total_q} exceeds arena capacities "
                f"(extend={self.caps.extend_max_total_q}, paged={self.caps.paged_max_q_rows})"
            )
        if contract.max_batch > max(self.caps.extend_max_batch, self.caps.paged_max_batch):
            raise ValueError(
                f"workspace max_batch {contract.max_batch} exceeds arena capacities "
                f"(extend={self.caps.extend_max_batch}, paged={self.caps.paged_max_batch})"
            )
        if contract.max_paged_q_rows > self.caps.paged_max_q_rows:
            raise ValueError(
                f"workspace max_paged_q_rows {contract.max_paged_q_rows} exceeds arena paged_max_q_rows {self.caps.paged_max_q_rows}"
            )
        if contract.max_kv_rows > self.caps.extend_max_kv_rows:
            raise ValueError(
                f"workspace max_kv_rows {contract.max_kv_rows} exceeds arena extend_max_kv_rows {self.caps.extend_max_kv_rows}"
            )
        if contract.indexer_num_q_heads > self.caps.indexer_num_q_heads:
            raise ValueError(
                "workspace indexer_num_q_heads "
                f"{contract.indexer_num_q_heads} exceeds arena indexer_num_q_heads "
                f"{self.caps.indexer_num_q_heads}"
            )
        if contract.max_page_table_width > self.caps.max_page_table_width:
            raise ValueError(
                "workspace max_page_table_width "
                f"{contract.max_page_table_width} exceeds arena max_page_table_width "
                f"{self.caps.max_page_table_width}"
            )
        workspace = B12XAttentionWorkspace(
            arena=self,
            contract=contract,
            mode=contract.mode,
            device=self.caps.device,
            dtype=self.caps.dtype,
            kv_dtype=self.caps.kv_dtype,
            num_q_heads=self.caps.num_q_heads,
            indexer_num_q_heads=contract.indexer_num_q_heads,
            head_dim=self.caps.head_dim,
            v_head_dim=contract.v_head_dim,
            topk=self.caps.topk,
            max_page_table_width=contract.max_page_table_width,
            max_total_q=contract.max_total_q,
            max_batch=contract.max_batch,
            max_paged_q_rows=contract.max_paged_q_rows,
            max_kv_rows=contract.max_kv_rows,
            page_size=self.caps.page_size,
            padded_heads=self.caps.padded_heads,
            use_cuda_graph=use_cuda_graph,
            fixed_capacity=True,
            max_chunks_per_row=self.caps.max_chunks_per_row,
            shared_arena=self.shared_arena,
            shared_arena_nbytes=self.shared_arena_nbytes,
            mla_phase_nbytes=self.mla_phase_nbytes,
            indexer_phase_nbytes=self.indexer_phase_nbytes,
            ragged_kv_nbytes=self.ragged_kv_nbytes,
            indexer_logits_nbytes=self.indexer_logits_nbytes,
            indexer_extend_logits_nbytes=self.indexer_extend_logits_nbytes,
            indexer_extend_tile_logits_nbytes=self.indexer_extend_tile_logits_nbytes,
            indexer_extend_topk_indices_nbytes=self.indexer_extend_topk_indices_nbytes,
            indexer_extend_topk_values_nbytes=self.indexer_extend_topk_values_nbytes,
            indexer_extend_topk_scratch_indices_nbytes=self.indexer_extend_topk_scratch_indices_nbytes,
            indexer_extend_topk_scratch_values_nbytes=self.indexer_extend_topk_scratch_values_nbytes,
            indexer_extend_candidate_values_nbytes=self.indexer_extend_candidate_values_nbytes,
            indexer_extend_candidate_indices_nbytes=self.indexer_extend_candidate_indices_nbytes,
            indexer_extend_lengths_nbytes=self.indexer_extend_lengths_nbytes,
            indexer_extend_mapped_indices_nbytes=self.indexer_extend_mapped_indices_nbytes,
            indexer_paged_logits_nbytes=self.indexer_paged_logits_nbytes,
        )
        workspace._allocate_fixed_capacity_views()
        workspace._initialize_split_chunk_config_if_needed()
        workspace._allocate_contract_phantoms()
        if use_cuda_graph:
            workspace._allocate_runtime_metadata()
        return workspace


@dataclass(kw_only=True)
class B12XAttentionWorkspace:
    arena: B12XAttentionArena | None = None
    contract: B12XAttentionWorkspaceContract | None = None
    mode: B12XWorkspaceMode
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    indexer_num_q_heads: int = 0
    head_dim: int
    v_head_dim: int
    topk: int
    max_page_table_width: int = 1
    max_total_q: int
    max_batch: int
    max_paged_q_rows: int = 0
    max_kv_rows: int = 0
    page_size: int = 64
    padded_heads: int = 128
    use_cuda_graph: bool = False
    fixed_capacity: bool = False
    max_chunks_per_row: int = 64
    page_table_1: torch.Tensor | None = None
    cache_seqlens_int32: torch.Tensor | None = None
    nsa_cache_seqlens_int32: torch.Tensor | None = None
    page_table_1_runtime: torch.Tensor | None = None
    cache_seqlens_int32_runtime: torch.Tensor | None = None
    nsa_cache_seqlens_int32_runtime: torch.Tensor | None = None
    paged_indexer_real_page_table_runtime: torch.Tensor | None = None
    paged_indexer_seqlens_per_query_runtime: torch.Tensor | None = None
    paged_indexer_active_width_runtime: torch.Tensor | None = None
    paged_indexer_schedule_metadata_runtime: torch.Tensor | None = None
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    ragged_kv_cache: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    num_chunks_ptr: torch.Tensor | None = None
    sm_scale_tensor: torch.Tensor | None = None
    sm_scale_value: float | None = None
    kv_chunk_size_value: int | None = None
    num_chunks_value: int | None = None
    shared_arena: torch.Tensor | None = None
    shared_arena_nbytes: int = 0
    mla_phase_nbytes: int = 0
    indexer_phase_nbytes: int = 0
    ragged_kv_nbytes: int = 0
    indexer_logits_nbytes: int = 0
    indexer_extend_logits_nbytes: int = 0
    indexer_extend_tile_logits_nbytes: int = 0
    indexer_extend_topk_indices_nbytes: int = 0
    indexer_extend_topk_values_nbytes: int = 0
    indexer_extend_topk_scratch_indices_nbytes: int = 0
    indexer_extend_topk_scratch_values_nbytes: int = 0
    indexer_extend_candidate_values_nbytes: int = 0
    indexer_extend_candidate_indices_nbytes: int = 0
    indexer_extend_lengths_nbytes: int = 0
    indexer_extend_mapped_indices_nbytes: int = 0
    indexer_paged_logits_nbytes: int = 0
    indexer_k_quant_bytes: torch.Tensor | None = None
    indexer_k_scales: torch.Tensor | None = None
    indexer_extend_logits: torch.Tensor | None = None
    indexer_extend_tile_logits: torch.Tensor | None = None
    indexer_extend_topk_indices: torch.Tensor | None = None
    indexer_extend_topk_values: torch.Tensor | None = None
    indexer_extend_topk_scratch_indices: torch.Tensor | None = None
    indexer_extend_topk_scratch_values: torch.Tensor | None = None
    indexer_extend_candidate_values: torch.Tensor | None = None
    indexer_extend_candidate_indices: torch.Tensor | None = None
    indexer_extend_lengths: torch.Tensor | None = None
    indexer_extend_mapped_indices: torch.Tensor | None = None
    indexer_paged_logits: torch.Tensor | None = None
    indexer_k_tma_desc: torch.Tensor | None = None
    indexer_k_tma_desc_ptrs: torch.Tensor | None = None
    indexer_k_tma_prefill_desc: torch.Tensor | None = None
    indexer_k_tma_prefill_desc_ptrs: torch.Tensor | None = None
    # Phantom tensors for stable host-launcher cache keys (fixed_capacity only).
    _contract_q: torch.Tensor | None = None
    _contract_kv_rows: torch.Tensor | None = None
    _contract_kv_scales: torch.Tensor | None = None
    _contract_page_table: torch.Tensor | None = None
    _contract_nsa_cache_seqlens: torch.Tensor | None = None
    _contract_output: torch.Tensor | None = None
    _contract_tmp_output: torch.Tensor | None = None
    _contract_tmp_lse: torch.Tensor | None = None
    _contract_indexer_q_bytes: torch.Tensor | None = None
    _contract_indexer_q_u32: torch.Tensor | None = None
    _contract_indexer_weights: torch.Tensor | None = None
    _contract_indexer_k_quant: torch.Tensor | None = None
    _contract_indexer_k_scale: torch.Tensor | None = None
    _contract_indexer_k_start: torch.Tensor | None = None
    _contract_indexer_k_end: torch.Tensor | None = None
    _contract_indexer_logits: torch.Tensor | None = None
    _contract_indexer_tile_logits: torch.Tensor | None = None
    _contract_paged_indexer_q_bytes: torch.Tensor | None = None
    _contract_paged_indexer_weights: torch.Tensor | None = None
    _contract_paged_real_page_table: torch.Tensor | None = None
    _contract_paged_nsa_cache_seqlens: torch.Tensor | None = None
    _contract_paged_indexer_logits: torch.Tensor | None = None
    _nsa_extend_tiled_topk_prewarmed: bool = False

    def __post_init__(self) -> None:
        self.device = _canonical_device(self.device)
        self.num_q_heads = int(self.num_q_heads)
        self.indexer_num_q_heads = int(self.indexer_num_q_heads) or int(self.num_q_heads)
        self.max_page_table_width = max(int(self.max_page_table_width), 1)
        self.max_paged_q_rows = max(int(self.max_paged_q_rows), 1)

    def runtime_metadata_nbytes(self) -> int:
        if not self.use_cuda_graph:
            return 0
        num_sms = 1
        if self.device.type == "cuda":
            num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        return (
            int(self.max_total_q) * int(self.topk) * _dtype_nbytes(torch.int32)
            + int(self.max_batch) * _dtype_nbytes(torch.int32)
            + int(self.max_total_q) * _dtype_nbytes(torch.int32)
            + int(self.max_paged_q_rows)
            * int(self.max_page_table_width)
            * _dtype_nbytes(torch.int32)
            + int(self.max_paged_q_rows) * _dtype_nbytes(torch.int32)
            + _dtype_nbytes(torch.int32)
            + (int(num_sms) + 1) * 2 * _dtype_nbytes(torch.int32)
        )

    def standalone_scratch_nbytes(self) -> int:
        if self.fixed_capacity:
            return 0
        return (
            int(self.max_total_q)
            * int(self.num_q_heads)
            * int(self.max_chunks_per_row)
            * int(self.v_head_dim)
            * _dtype_nbytes(self.dtype)
            + int(self.max_total_q)
            * int(self.num_q_heads)
            * int(self.max_chunks_per_row)
            * _dtype_nbytes(torch.float32)
            + 2 * _dtype_nbytes(torch.int32)
        )

    @classmethod
    def for_contract(
        cls,
        *,
        mode: Literal["decode", "extend", "verify", "draft_extend"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        indexer_num_q_heads: int | None = None,
        head_dim: int,
        v_head_dim: int,
        topk: int,
        max_page_table_width: int | None = None,
        max_total_q: int,
        max_batch: int,
        max_paged_q_rows: int | None = None,
        max_kv_rows: int | None = None,
        page_size: int = 64,
        use_cuda_graph: bool = False,
        padded_heads: int = 128,
    ) -> B12XAttentionWorkspace:
        device = _canonical_device(device)
        if indexer_num_q_heads is None:
            indexer_num_q_heads = num_q_heads
        if max_page_table_width is None:
            max_page_table_width = topk
        if max_paged_q_rows is None:
            max_paged_q_rows = max_batch
        workspace = cls(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            indexer_num_q_heads=indexer_num_q_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            topk=topk,
            max_page_table_width=max_page_table_width,
            max_total_q=int(max_total_q),
            max_batch=int(max_batch),
            max_paged_q_rows=int(max_paged_q_rows),
            max_kv_rows=max(0, int(max_kv_rows)) if max_kv_rows is not None else 0,
            page_size=page_size,
            padded_heads=padded_heads,
            use_cuda_graph=use_cuda_graph,
        )
        workspace._allocate_split_buffers()
        if use_cuda_graph:
            workspace._allocate_runtime_metadata()
        return workspace

    @classmethod
    def for_fixed_capacity(
        cls,
        *,
        mode: Literal["decode", "extend", "verify", "draft_extend"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        indexer_num_q_heads: int | None = None,
        head_dim: int,
        v_head_dim: int,
        topk: int,
        max_page_table_width: int | None = None,
        max_total_q: int,
        max_batch: int,
        max_paged_q_rows: int | None = None,
        max_kv_rows: int | None = None,
        page_size: int = 64,
        use_cuda_graph: bool = False,
        padded_heads: int = 128,
    ) -> B12XAttentionWorkspace:
        device = _canonical_device(device)
        if indexer_num_q_heads is None:
            indexer_num_q_heads = num_q_heads
        topk = int(topk)
        if max_page_table_width is None:
            max_page_table_width = topk
        max_page_table_width = max(int(max_page_table_width), topk, 1)
        if max_paged_q_rows is None:
            max_paged_q_rows = max_batch
        max_paged_q_rows = max(int(max_paged_q_rows), 1)
        caps = B12XAttentionArenaCaps(
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            indexer_num_q_heads=indexer_num_q_heads,
            head_dim=head_dim,
            max_v_head_dim=v_head_dim,
            topk=topk,
            max_page_table_width=max_page_table_width,
            extend_max_total_q=max_total_q,
            extend_max_batch=max_batch,
            extend_max_kv_rows=max(0, int(max_kv_rows)) if max_kv_rows is not None else 0,
            paged_max_q_rows=max_paged_q_rows,
            paged_max_batch=max_batch,
            page_size=page_size,
            padded_heads=padded_heads,
        )
        arena = B12XAttentionArena.allocate(caps)
        contract = B12XAttentionWorkspaceContract(
            mode=mode,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_paged_q_rows=max_paged_q_rows,
            max_kv_rows=max(0, int(max_kv_rows)) if max_kv_rows is not None else 0,
            v_head_dim=v_head_dim,
            indexer_num_q_heads=indexer_num_q_heads,
            max_page_table_width=max_page_table_width,
        )
        return arena.make_workspace(contract, use_cuda_graph=use_cuda_graph)

    def _allocate_runtime_metadata(self) -> None:
        if self.page_table_1_runtime is None:
            self.page_table_1_runtime = torch.empty(
                (self.max_total_q, self.topk),
                dtype=torch.int32,
                device=self.device,
            )
        if self.cache_seqlens_int32_runtime is None:
            self.cache_seqlens_int32_runtime = torch.empty(
                (self.max_batch,),
                dtype=torch.int32,
                device=self.device,
            )
        if self.nsa_cache_seqlens_int32_runtime is None:
            self.nsa_cache_seqlens_int32_runtime = torch.empty(
                (self.max_total_q,),
                dtype=torch.int32,
                device=self.device,
            )
        if self.paged_indexer_real_page_table_runtime is None:
            self.paged_indexer_real_page_table_runtime = torch.empty(
                (self.max_paged_q_rows, self.max_page_table_width),
                dtype=torch.int32,
                device=self.device,
            )
        if self.paged_indexer_seqlens_per_query_runtime is None:
            self.paged_indexer_seqlens_per_query_runtime = torch.empty(
                (self.max_paged_q_rows,),
                dtype=torch.int32,
                device=self.device,
            )
        if self.paged_indexer_active_width_runtime is None:
            self.paged_indexer_active_width_runtime = torch.empty(
                (1,),
                dtype=torch.int32,
                device=self.device,
            )
        if self.paged_indexer_schedule_metadata_runtime is None:
            num_sms = 1
            if self.device.type == "cuda":
                num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            self.paged_indexer_schedule_metadata_runtime = torch.empty(
                (int(num_sms) + 1, 2),
                dtype=torch.int32,
                device=self.device,
            )

    def _allocate_fixed_capacity_views(self) -> None:
        if self.arena is None:
            raise RuntimeError("_allocate_fixed_capacity_views requires an arena-backed workspace")
        max_total_q = max(int(self.max_total_q), 1)
        max_paged_q_rows = max(int(self.max_paged_q_rows), 1)
        max_kv_rows = max(int(self.max_kv_rows), 1)
        indexer_k_rows = _align_up(max_kv_rows, _NSA_INDEXER_BLOCK_K)
        paged_width_tokens = max(
            int(self.max_page_table_width) * int(self.page_size),
            1,
        )
        self.shared_arena = self.arena.shared_arena
        self.shared_arena_nbytes = self.arena.shared_arena_nbytes
        self.mla_phase_nbytes = self.arena.mla_phase_nbytes
        self.indexer_phase_nbytes = self.arena.indexer_phase_nbytes
        self.ragged_kv_nbytes = self.arena.ragged_kv_nbytes
        self.indexer_extend_logits_nbytes = self.arena.indexer_extend_logits_nbytes
        self.indexer_extend_tile_logits_nbytes = self.arena.indexer_extend_tile_logits_nbytes
        self.indexer_extend_topk_indices_nbytes = self.arena.indexer_extend_topk_indices_nbytes
        self.indexer_extend_topk_values_nbytes = self.arena.indexer_extend_topk_values_nbytes
        self.indexer_extend_topk_scratch_indices_nbytes = self.arena.indexer_extend_topk_scratch_indices_nbytes
        self.indexer_extend_topk_scratch_values_nbytes = self.arena.indexer_extend_topk_scratch_values_nbytes
        self.indexer_extend_candidate_values_nbytes = self.arena.indexer_extend_candidate_values_nbytes
        self.indexer_extend_candidate_indices_nbytes = self.arena.indexer_extend_candidate_indices_nbytes
        self.indexer_extend_lengths_nbytes = self.arena.indexer_extend_lengths_nbytes
        self.indexer_extend_mapped_indices_nbytes = self.arena.indexer_extend_mapped_indices_nbytes
        self.indexer_paged_logits_nbytes = self.arena.indexer_paged_logits_nbytes
        self.indexer_logits_nbytes = self.arena.indexer_logits_nbytes

        assert self.shared_arena is not None
        self.ragged_kv_cache, mla_offset = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.arena.ragged_kv_offset_bytes,
            shape=(max_kv_rows, 1, _MLA_PACKED_DIM),
            dtype=self.kv_dtype,
        )
        self.tmp_output, mla_offset = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.arena.tmp_output_offset_bytes,
            shape=(
                max_total_q,
                int(self.num_q_heads),
                int(self.max_chunks_per_row),
                int(self.v_head_dim),
            ),
            dtype=self.dtype,
        )
        self.tmp_lse, _ = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.arena.tmp_lse_offset_bytes,
            shape=(max_total_q, int(self.num_q_heads), int(self.max_chunks_per_row)),
            dtype=torch.float32,
        )

        self.indexer_k_quant_bytes, extend_offset = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.arena.indexer_k_quant_offset_bytes,
            shape=(indexer_k_rows, _INDEX_HEAD_DIM),
            dtype=torch.uint8,
        )
        self.indexer_k_scales, extend_offset = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.arena.indexer_k_scale_offset_bytes,
            shape=(indexer_k_rows,),
            dtype=torch.float32,
        )
        self.indexer_k_tma_desc, self.indexer_k_tma_desc_ptrs = _encode_indexer_k_tma_descriptor(
            self.indexer_k_quant_bytes,
            block_k=_NSA_INDEXER_BLOCK_K,
        )
        self.indexer_k_tma_prefill_desc, self.indexer_k_tma_prefill_desc_ptrs = (
            _encode_indexer_k_tma_descriptor(
                self.indexer_k_quant_bytes,
                block_k=_NSA_INDEXER_PREFILL_BLOCK_K,
            )
        )
        if self.indexer_extend_logits_nbytes:
            self.indexer_extend_logits, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_logits_offset_bytes,
                shape=(max_total_q * indexer_k_rows,),
                dtype=torch.float32,
            )
        else:
            self.indexer_extend_logits = None
        if self.indexer_extend_tile_logits_nbytes:
            self.indexer_extend_tile_logits, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_tile_logits_offset_bytes,
                shape=(self.indexer_extend_tile_logits_nbytes // _dtype_nbytes(torch.float32),),
                dtype=torch.float32,
            )
        else:
            self.indexer_extend_tile_logits = None
        if self.indexer_extend_topk_indices_nbytes:
            self.indexer_extend_topk_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_topk_indices_offset_bytes,
                shape=(max_total_q, int(self.topk)),
                dtype=torch.int32,
            )
        else:
            self.indexer_extend_topk_indices = None
        if self.indexer_extend_topk_values_nbytes:
            self.indexer_extend_topk_values, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_topk_values_offset_bytes,
                shape=(max_total_q, int(self.topk)),
                dtype=torch.float32,
            )
        else:
            self.indexer_extend_topk_values = None
        if self.indexer_extend_topk_scratch_indices_nbytes:
            self.indexer_extend_topk_scratch_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_topk_scratch_indices_offset_bytes,
                shape=(max_total_q, int(self.topk)),
                dtype=torch.int32,
            )
        else:
            self.indexer_extend_topk_scratch_indices = None
        if self.indexer_extend_topk_scratch_values_nbytes:
            self.indexer_extend_topk_scratch_values, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_topk_scratch_values_offset_bytes,
                shape=(max_total_q, int(self.topk)),
                dtype=torch.float32,
            )
        else:
            self.indexer_extend_topk_scratch_values = None
        if self.indexer_extend_candidate_values_nbytes:
            candidate_chunks = self.indexer_extend_candidate_values_nbytes // (
                max_total_q * int(self.topk) * _dtype_nbytes(torch.float32)
            )
            self.indexer_extend_candidate_values, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_candidate_values_offset_bytes,
                shape=(candidate_chunks, max_total_q, int(self.topk)),
                dtype=torch.float32,
            )
        else:
            self.indexer_extend_candidate_values = None
        if self.indexer_extend_candidate_indices_nbytes:
            candidate_chunks = self.indexer_extend_candidate_indices_nbytes // (
                max_total_q * int(self.topk) * _dtype_nbytes(torch.int32)
            )
            self.indexer_extend_candidate_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_candidate_indices_offset_bytes,
                shape=(candidate_chunks, max_total_q, int(self.topk)),
                dtype=torch.int32,
            )
        else:
            self.indexer_extend_candidate_indices = None
        if self.indexer_extend_lengths_nbytes:
            self.indexer_extend_lengths, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_lengths_offset_bytes,
                shape=(max_total_q,),
                dtype=torch.int32,
            )
        else:
            self.indexer_extend_lengths = None
        if self.indexer_extend_mapped_indices_nbytes:
            self.indexer_extend_mapped_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_extend_mapped_indices_offset_bytes,
                shape=(max_total_q, int(self.topk)),
                dtype=torch.int32,
            )
        else:
            self.indexer_extend_mapped_indices = None
        if self.indexer_paged_logits_nbytes:
            self.indexer_paged_logits, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_paged_logits_offset_bytes,
                shape=(max_paged_q_rows * paged_width_tokens,),
                dtype=torch.float32,
            )
        else:
            self.indexer_paged_logits = None

    def _allocate_split_buffers(self) -> None:
        if self.mode not in ("decode", "extend", "verify", "draft_extend"):
            return
        if self.fixed_capacity:
            if self.shared_arena is None:
                self._allocate_fixed_capacity_views()
        elif self.tmp_output is None:
            self.tmp_output = torch.empty(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row, self.v_head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        if self.tmp_lse is None:
            self.tmp_lse = torch.empty(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row),
                dtype=torch.float32,
                device=self.device,
            )
        if self.kv_chunk_size_ptr is None:
            self.kv_chunk_size_ptr = torch.empty((1,), dtype=torch.int32, device=self.device)
            self.kv_chunk_size_value = None
        if self.num_chunks_ptr is None:
            self.num_chunks_ptr = torch.empty((1,), dtype=torch.int32, device=self.device)
            self.num_chunks_value = None

    def _initialize_split_chunk_config_if_needed(self) -> None:
        self._allocate_split_buffers()
        if not (self.fixed_capacity or self.use_cuda_graph):
            return
        if self.kv_chunk_size_value is not None and self.num_chunks_value is not None:
            return
        split_cfg = default_sparse_mla_split_decode_config_for_width(
            int(self.topk),
            max_chunks=self.max_chunks_per_row,
        )
        if split_cfg is None:
            return
        assert self.kv_chunk_size_ptr is not None
        assert self.num_chunks_ptr is not None
        self.kv_chunk_size_ptr[0] = int(split_cfg.chunk_size)
        self.num_chunks_ptr[0] = int(split_cfg.num_chunks)
        self.kv_chunk_size_value = int(split_cfg.chunk_size)
        self.num_chunks_value = int(split_cfg.num_chunks)

    def set_split_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        if num_chunks <= 0 or num_chunks > self.max_chunks_per_row:
            raise ValueError(
                f"num_chunks must be in [1, {self.max_chunks_per_row}], got {num_chunks}"
            )
        if kv_chunk_size <= 0:
            raise ValueError(f"kv_chunk_size must be positive, got {kv_chunk_size}")
        self._allocate_split_buffers()
        assert self.kv_chunk_size_ptr is not None
        assert self.num_chunks_ptr is not None
        if self.kv_chunk_size_value != int(kv_chunk_size):
            self.kv_chunk_size_ptr[0] = int(kv_chunk_size)
            self.kv_chunk_size_value = int(kv_chunk_size)
        if self.num_chunks_value != int(num_chunks):
            self.num_chunks_ptr[0] = int(num_chunks)
            self.num_chunks_value = int(num_chunks)

    def set_decode_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        self.set_split_chunk_config(kv_chunk_size=kv_chunk_size, num_chunks=num_chunks)

    def prepare_decode(
        self,
        page_table_1: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> None:
        self._prepare_sparse(
            page_table_1=page_table_1,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )

    def prepare_extend(
        self,
        selected_token_offsets: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> None:
        self._prepare_sparse(
            page_table_1=selected_token_offsets,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )

    def bind_cuda_graph_runtime_metadata(
        self,
        *,
        page_table_1: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> None:
        if not self.use_cuda_graph:
            raise RuntimeError("bind_cuda_graph_runtime_metadata is only valid for graph-mode workspaces")
        self._prepare_sparse(
            page_table_1=page_table_1,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )

    def _prepare_sparse(
        self,
        *,
        page_table_1: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> None:
        if page_table_1.ndim != 2:
            raise ValueError(f"page_table_1 must be rank-2, got {tuple(page_table_1.shape)}")
        if cache_seqlens_int32.ndim != 1:
            raise ValueError(
                f"cache_seqlens_int32 must be rank-1, got {tuple(cache_seqlens_int32.shape)}"
            )
        if nsa_cache_seqlens_int32.ndim != 1:
            raise ValueError(
                "nsa_cache_seqlens_int32 must be rank-1, "
                f"got {tuple(nsa_cache_seqlens_int32.shape)}"
            )
        if page_table_1.device != self.device:
            raise ValueError(
                f"page_table_1 device {page_table_1.device} does not match workspace device {self.device}"
            )
        if cache_seqlens_int32.device != self.device:
            raise ValueError(
                "cache_seqlens_int32 device "
                f"{cache_seqlens_int32.device} does not match workspace device {self.device}"
            )
        if nsa_cache_seqlens_int32.device != self.device:
            raise ValueError(
                "nsa_cache_seqlens_int32 device "
                f"{nsa_cache_seqlens_int32.device} does not match workspace device {self.device}"
            )
        if page_table_1.dtype != torch.int32:
            raise ValueError(f"page_table_1 must have dtype torch.int32, got {page_table_1.dtype}")
        if cache_seqlens_int32.dtype != torch.int32:
            raise ValueError(
                "cache_seqlens_int32 must have dtype torch.int32, "
                f"got {cache_seqlens_int32.dtype}"
            )
        if nsa_cache_seqlens_int32.dtype != torch.int32:
            raise ValueError(
                "nsa_cache_seqlens_int32 must have dtype torch.int32, "
                f"got {nsa_cache_seqlens_int32.dtype}"
            )
        if page_table_1.shape[0] > self.max_total_q:
            raise ValueError(
                f"page_table_1 rows {page_table_1.shape[0]} exceed workspace capacity {self.max_total_q}"
            )
        if cache_seqlens_int32.shape[0] > self.max_batch:
            raise ValueError(
                "cache_seqlens_int32 batch "
                f"{cache_seqlens_int32.shape[0]} exceeds workspace capacity {self.max_batch}"
            )
        if page_table_1.shape[1] > self.topk:
            raise ValueError(
                f"page_table_1 width {page_table_1.shape[1]} exceeds topk capacity {self.topk}"
            )
        if page_table_1.shape[0] != nsa_cache_seqlens_int32.shape[0]:
            raise ValueError(
                "page_table_1 rows "
                f"{page_table_1.shape[0]} do not match nsa_cache_seqlens_int32 rows "
                f"{nsa_cache_seqlens_int32.shape[0]}"
            )
        use_runtime_buffers = self.use_cuda_graph
        if not use_runtime_buffers:
            if self.device.type == "cuda":
                self.page_table_1 = page_table_1
                self.cache_seqlens_int32 = cache_seqlens_int32
                self.nsa_cache_seqlens_int32 = nsa_cache_seqlens_int32
            else:
                self.page_table_1 = page_table_1.clone()
                self.cache_seqlens_int32 = cache_seqlens_int32.clone()
                self.nsa_cache_seqlens_int32 = nsa_cache_seqlens_int32.clone()
            return

        self._allocate_runtime_metadata()
        assert self.page_table_1_runtime is not None
        assert self.cache_seqlens_int32_runtime is not None
        assert self.nsa_cache_seqlens_int32_runtime is not None
        rows, width = page_table_1.shape
        batch = cache_seqlens_int32.shape[0]
        self.page_table_1_runtime[:rows, :width].copy_(page_table_1)
        self.cache_seqlens_int32_runtime[:batch].copy_(cache_seqlens_int32)
        self.nsa_cache_seqlens_int32_runtime[:rows].copy_(nsa_cache_seqlens_int32)
        self.page_table_1 = self.page_table_1_runtime[:rows, :width]
        self.cache_seqlens_int32 = self.cache_seqlens_int32_runtime[:batch]
        self.nsa_cache_seqlens_int32 = self.nsa_cache_seqlens_int32_runtime[:rows]

    def gather_ragged_kv_rows(
        self,
        *,
        kv_cache: torch.Tensor,
        row_ids: torch.Tensor,
    ) -> torch.Tensor:
        if kv_cache.ndim != 3:
            raise ValueError(f"kv_cache must be rank-3, got {tuple(kv_cache.shape)}")
        if row_ids.ndim != 1:
            raise ValueError(f"row_ids must be rank-1, got {tuple(row_ids.shape)}")
        if kv_cache.device != self.device:
            raise ValueError(
                f"kv_cache device {kv_cache.device} does not match workspace device {self.device}"
            )
        if row_ids.device != self.device:
            raise ValueError(
                f"row_ids device {row_ids.device} does not match workspace device {self.device}"
            )
        if kv_cache.dtype != self.kv_dtype:
            raise ValueError(
                f"kv_cache dtype {kv_cache.dtype} does not match workspace kv_dtype {self.kv_dtype}"
            )

        row_count = int(row_ids.shape[0])
        capacity = max(int(self.max_kv_rows), row_count, 1)
        expected_row_shape = tuple(int(dim) for dim in kv_cache.shape[1:])
        buffer = self.ragged_kv_cache
        if (
            buffer is None
            or buffer.device != self.device
            or buffer.dtype != kv_cache.dtype
            or tuple(int(dim) for dim in buffer.shape[1:]) != expected_row_shape
            or buffer.shape[0] < capacity
        ):
            if self.fixed_capacity and buffer is not None:
                raise ValueError(
                    f"row_count {row_count} exceeds fixed-capacity ragged KV workspace {buffer.shape[0]}"
                )
            buffer = torch.empty(
                (capacity, *expected_row_shape),
                dtype=kv_cache.dtype,
                device=self.device,
            )
            self.ragged_kv_cache = buffer
            self.max_kv_rows = capacity
            self._refresh_ragged_kv_contracts()
        elif self._contract_kv_rows is None or self._contract_kv_scales is None:
            self._refresh_ragged_kv_contracts()

        assert buffer is not None
        if row_count != 0:
            kv_bytes = kv_cache.view(torch.uint8)
            gathered_bytes = buffer[:row_count].view(torch.uint8)
            torch.index_select(kv_bytes, 0, row_ids.to(torch.long), out=gathered_bytes)
        # Return the full-capacity scratch buffer so launcher cache keys follow
        # workspace capacity instead of the live ragged row count for this prefill.
        return buffer

    def get_indexer_contract_phantoms(self) -> dict[str, torch.Tensor]:
        if (
            self._contract_indexer_q_u32 is None
            or self._contract_indexer_q_bytes is None
            or self._contract_indexer_weights is None
            or self._contract_indexer_k_quant is None
            or self._contract_indexer_k_scale is None
            or self._contract_indexer_k_start is None
            or self._contract_indexer_k_end is None
        ):
            raise RuntimeError("fixed-capacity workspace is missing NSA indexer phantoms")
        phantoms = {
            "extend_q_u32": self._contract_indexer_q_u32,
            "extend_q_bytes": self._contract_indexer_q_bytes,
            "extend_weights": self._contract_indexer_weights,
            "extend_k_quant": self._contract_indexer_k_quant,
            "extend_k_scale": self._contract_indexer_k_scale,
            "extend_k_start": self._contract_indexer_k_start,
            "extend_k_end": self._contract_indexer_k_end,
        }
        if self._contract_indexer_logits is not None:
            phantoms["extend_logits"] = self._contract_indexer_logits
        if self._contract_indexer_tile_logits is not None:
            phantoms["extend_tile_logits"] = self._contract_indexer_tile_logits
        return phantoms

    def get_indexer_extend_tile_logits(self) -> torch.Tensor | None:
        return self.indexer_extend_tile_logits

    def get_indexer_extend_topk_buffers(
        self,
        *,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.indexer_extend_topk_values is None or self.indexer_extend_topk_indices is None:
            raise RuntimeError("fixed-capacity workspace is missing NSA extend top-k buffers")
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > self.indexer_extend_topk_indices.shape[0]:
            raise ValueError(
                "row_count "
                f"{row_count} exceeds workspace top-k capacity {self.indexer_extend_topk_indices.shape[0]}"
            )
        return (
            self.indexer_extend_topk_values[:row_count],
            self.indexer_extend_topk_indices[:row_count],
        )

    def get_indexer_extend_topk_scratch_buffers(
        self,
        *,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.indexer_extend_topk_scratch_values is None
            or self.indexer_extend_topk_scratch_indices is None
        ):
            raise RuntimeError("fixed-capacity workspace is missing NSA extend top-k scratch buffers")
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > self.indexer_extend_topk_scratch_indices.shape[0]:
            raise ValueError(
                "row_count "
                f"{row_count} exceeds workspace top-k scratch capacity "
                f"{self.indexer_extend_topk_scratch_indices.shape[0]}"
            )
        return (
            self.indexer_extend_topk_scratch_values[:row_count],
            self.indexer_extend_topk_scratch_indices[:row_count],
        )

    def get_indexer_extend_candidate_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.indexer_extend_candidate_values is None
            or self.indexer_extend_candidate_indices is None
        ):
            raise RuntimeError("fixed-capacity workspace is missing NSA extend candidate buffers")
        return self.indexer_extend_candidate_values, self.indexer_extend_candidate_indices

    def get_indexer_extend_lengths(self, *, row_count: int) -> torch.Tensor:
        if self.indexer_extend_lengths is None:
            raise RuntimeError("fixed-capacity workspace is missing NSA extend length buffer")
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > self.indexer_extend_lengths.shape[0]:
            raise ValueError(
                f"row_count {row_count} exceeds workspace length capacity {self.indexer_extend_lengths.shape[0]}"
            )
        return self.indexer_extend_lengths[:row_count]

    def get_indexer_extend_topk_result(self, *, row_count: int) -> torch.Tensor:
        if self.indexer_extend_topk_indices is None:
            raise RuntimeError("fixed-capacity workspace is missing NSA extend top-k result buffer")
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > self.indexer_extend_topk_indices.shape[0]:
            raise ValueError(
                "row_count "
                f"{row_count} exceeds workspace top-k capacity {self.indexer_extend_topk_indices.shape[0]}"
            )
        return self.indexer_extend_topk_indices[:row_count]

    def get_indexer_extend_mapped_indices(self, *, row_count: int, width: int) -> torch.Tensor:
        if self.indexer_extend_mapped_indices is None:
            raise RuntimeError("fixed-capacity workspace is missing NSA mapped-index buffer")
        row_count = int(row_count)
        width = int(width)
        if row_count < 0 or width < 0:
            raise ValueError(f"row_count and width must be non-negative, got {row_count}, {width}")
        if row_count > self.indexer_extend_mapped_indices.shape[0]:
            raise ValueError(
                "row_count "
                f"{row_count} exceeds workspace mapped-index capacity {self.indexer_extend_mapped_indices.shape[0]}"
            )
        if width > self.indexer_extend_mapped_indices.shape[1]:
            raise ValueError(
                f"width {width} exceeds workspace mapped-index width {self.indexer_extend_mapped_indices.shape[1]}"
            )
        return self.indexer_extend_mapped_indices[:row_count, :width]

    def prewarm_nsa_extend_tiled_topk(self) -> None:
        """Compile the arena-backed extend scorer/topk variants before serving.

        The production extend path uses fixed-capacity phantom tensors so live
        K lengths do not change the host-launcher cache key.  There are still
        two intentional scorer variants today: BK=256 and BK=512.  This warmup
        runs representative single-supertile calls for both, when supported by
        the workspace capacity, so chunked prefill traffic does not pay the CuTe
        compile cost at the first request that reaches each variant.
        """

        if self._nsa_extend_tiled_topk_prewarmed:
            return
        if self.mode != "extend":
            return
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return
        if (
            self.indexer_extend_tile_logits is None
            or self.indexer_k_quant_bytes is None
            or self.indexer_k_scales is None
            or self.max_total_q < 256
            or self.topk <= 0
        ):
            return

        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            return

        from b12x.attention.nsa_indexer.api import (
            NSAIndexerExtendLogitsMetadata,
            sparse_nsa_index_extend_tiled_topk,
        )
        from b12x.attention.nsa_indexer.extend_kernel import (
            _PREFILL512_BLOCK_K,
            _PREFILL512_BLOCK_Q,
            _PREFILL_BLOCK_Q,
            resolve_sparse_nsa_extend_prefill_block_k,
        )

        q_rows = min(int(self.max_total_q), 4096)
        heads = int(self.indexer_num_q_heads)
        warm_cases: list[int] = []
        for requested_k_rows in (4096, 32768):
            if requested_k_rows > int(self.max_kv_rows):
                continue
            if requested_k_rows < int(self.topk):
                continue
            block_k = resolve_sparse_nsa_extend_prefill_block_k(
                valid_q_rows=q_rows,
                k_rows=requested_k_rows,
                num_heads=heads,
            )
            if block_k is None:
                continue
            block_q = (
                _PREFILL512_BLOCK_Q
                if block_k == _PREFILL512_BLOCK_K
                else _PREFILL_BLOCK_Q
            )
            num_q_tiles = (q_rows + block_q - 1) // block_q
            num_k_tiles = (requested_k_rows + block_k - 1) // block_k
            required_tile_logits = num_q_tiles * num_k_tiles * block_q * block_k
            if int(self.indexer_extend_tile_logits.numel()) < required_tile_logits:
                continue
            warm_cases.append(requested_k_rows)

        if not warm_cases:
            return

        with torch.cuda.device(self.device):
            q_fp8 = torch.empty(
                (q_rows, heads, _INDEX_HEAD_DIM),
                dtype=fp8_dtype,
                device=self.device,
            )
            weights = torch.empty(
                (q_rows, heads),
                dtype=torch.float32,
                device=self.device,
            )
            k_start = torch.empty((q_rows,), dtype=torch.int32, device=self.device)
            k_end = torch.empty((q_rows,), dtype=torch.int32, device=self.device)
            q_fp8.zero_()
            weights.fill_(1.0)

            phantoms = self.get_indexer_contract_phantoms()
            tile_logits = self.get_indexer_extend_tile_logits()
            assert tile_logits is not None
            for k_rows in warm_cases:
                self.indexer_k_quant_bytes[:k_rows].zero_()
                self.indexer_k_scales[:k_rows].fill_(1.0)
                k_start.zero_()
                k_end.fill_(k_rows)
                sparse_nsa_index_extend_tiled_topk(
                    q_fp8=q_fp8,
                    weights=weights,
                    kv_fp8=(
                        self.indexer_k_quant_bytes[:k_rows].view(fp8_dtype),
                        self.indexer_k_scales[:k_rows],
                    ),
                    metadata=NSAIndexerExtendLogitsMetadata(k_start=k_start, k_end=k_end),
                    topk=int(self.topk),
                    contract_phantoms=phantoms,
                    tile_logits=tile_logits,
                    supertile_k=k_rows,
                )
            torch.cuda.synchronize(self.device)

        self._nsa_extend_tiled_topk_prewarmed = True

    def get_paged_indexer_contract_phantoms(self) -> dict[str, torch.Tensor]:
        if (
            self._contract_paged_indexer_q_bytes is None
            or self._contract_paged_indexer_weights is None
            or self._contract_paged_real_page_table is None
            or self._contract_paged_nsa_cache_seqlens is None
            or self._contract_paged_indexer_logits is None
        ):
            raise RuntimeError("fixed-capacity workspace is missing paged NSA indexer phantoms")
        return {
            "q_bytes": self._contract_paged_indexer_q_bytes,
            "weights": self._contract_paged_indexer_weights,
            "real_page_table": self._contract_paged_real_page_table,
            "seqlens_per_query": self._contract_paged_nsa_cache_seqlens,
            "logits": self._contract_paged_indexer_logits,
        }

    def get_indexer_gather_outputs(
        self,
        *,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.indexer_k_quant_bytes is None or self.indexer_k_scales is None:
            raise RuntimeError("fixed-capacity workspace is missing NSA gather buffers")
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > self.indexer_k_quant_bytes.shape[0]:
            raise ValueError(
                f"row_count {row_count} exceeds workspace gather capacity {self.indexer_k_quant_bytes.shape[0]}"
            )
        k_scale_bytes = self.indexer_k_scales.view(torch.uint8).view(
            self.indexer_k_scales.shape[0], 4
        )
        return self.indexer_k_quant_bytes[:row_count], k_scale_bytes[:row_count]

    def stage_nsa_indexer_extend(
        self,
        *,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        k_quant: torch.Tensor,
        k_scale: torch.Tensor,
        k_start: torch.Tensor,
        k_end: torch.Tensor,
        preinitialize_invalid_logits: bool = True,
    ) -> dict[str, torch.Tensor]:
        if (
            self.indexer_k_quant_bytes is None
            or self.indexer_k_scales is None
            or self.indexer_extend_logits is None
        ):
            raise RuntimeError("fixed-capacity workspace is missing NSA indexer buffers")
        if not q_fp8.is_contiguous():
            raise ValueError("workspace-backed NSA indexer extend requires contiguous q_fp8")
        if not weights.is_contiguous():
            raise ValueError("workspace-backed NSA indexer extend requires contiguous weights")
        if not k_quant.is_contiguous():
            raise ValueError("workspace-backed NSA indexer extend requires contiguous k_quant")
        if not k_scale.is_contiguous():
            raise ValueError("workspace-backed NSA indexer extend requires contiguous k_scale")
        if not k_start.is_contiguous():
            raise ValueError("workspace-backed NSA indexer extend requires contiguous k_start")
        if not k_end.is_contiguous():
            raise ValueError("workspace-backed NSA indexer extend requires contiguous k_end")

        q_rows_total = int(q_fp8.shape[0])
        valid_q_rows = int(k_start.shape[0])
        k_rows = int(k_quant.shape[0])
        padded_k_rows = _align_up(max(k_rows, 1), _NSA_INDEXER_BLOCK_K)
        if not preinitialize_invalid_logits and valid_q_rows != q_rows_total:
            raise ValueError(
                "preinitialize_invalid_logits=False requires all q rows to be valid; "
                f"got q_rows={q_rows_total} and valid_q_rows={valid_q_rows}"
            )

        if q_rows_total > self.max_total_q:
            raise ValueError(
                f"q rows {q_rows_total} exceed workspace NSA indexer capacity {self.max_total_q}"
            )
        if padded_k_rows > self.indexer_k_quant_bytes.shape[0]:
            raise ValueError(
                f"k rows {k_rows} exceed workspace NSA indexer capacity {self.indexer_k_quant_bytes.shape[0]}"
            )
        if q_fp8.ndim != 3 or q_fp8.shape[1] != self.indexer_num_q_heads:
            raise ValueError(
                "q_fp8 must have shape "
                f"(q_rows, {self.indexer_num_q_heads}, {_INDEX_HEAD_DIM}), got {tuple(q_fp8.shape)}"
            )
        if q_fp8.shape[2] != _INDEX_HEAD_DIM:
            raise ValueError(
                f"q_fp8 trailing dimension must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}"
            )
        if weights.ndim != 2 or weights.shape != (q_rows_total, self.indexer_num_q_heads):
            raise ValueError(
                "weights must have shape "
                f"({q_rows_total}, {self.indexer_num_q_heads}), got {tuple(weights.shape)}"
            )

        q_bytes = q_fp8.view(torch.uint8)
        q_u32 = q_bytes.view(torch.uint32).view(
            q_rows_total,
            int(self.indexer_num_q_heads),
            _INDEX_HEAD_DIM // 4,
        )

        k_quant_bytes = k_quant.view(torch.uint8)
        k_quant_aliases_workspace = (
            k_quant_bytes.data_ptr() == self.indexer_k_quant_bytes.data_ptr()
            and k_quant_bytes.storage_offset() == self.indexer_k_quant_bytes.storage_offset()
        )
        k_scale_aliases_workspace = (
            k_scale.data_ptr() == self.indexer_k_scales.data_ptr()
            and k_scale.storage_offset() == self.indexer_k_scales.storage_offset()
        )
        if k_quant_aliases_workspace:
            k_quant_kernel = self.indexer_k_quant_bytes[:padded_k_rows]
            k_scale_kernel = self.indexer_k_scales[:padded_k_rows]
            if padded_k_rows > k_rows:
                self.indexer_k_quant_bytes[k_rows:padded_k_rows].zero_()
                self.indexer_k_scales[k_rows:padded_k_rows].zero_()
        else:
            if padded_k_rows != k_rows:
                raise ValueError(
                    "workspace-backed NSA indexer extend requires pre-padded K/scale rows "
                    "or workspace gather outputs; refusing an implicit pad copy"
                )
            k_quant_kernel = k_quant_bytes
            k_scale_kernel = k_scale
        if k_quant_aliases_workspace != k_scale_aliases_workspace:
            raise ValueError(
                "workspace-backed NSA indexer extend requires k_quant and k_scale to either "
                "both alias the workspace gather buffers or both use live storage"
            )

        logits_view = self.indexer_extend_logits.narrow(0, 0, q_rows_total * k_rows).view(
            q_rows_total, k_rows
        )
        if preinitialize_invalid_logits and q_rows_total != 0 and k_rows != 0:
            logits_view.fill_(float("-inf"))
        return {
            "q_u32": q_u32,
            "weights": weights,
            "k_quant_bytes": k_quant_kernel,
            "k_scales": k_scale_kernel,
            "k_start": k_start,
            "k_end": k_end,
            "logits": logits_view,
            "logits_view": logits_view,
            "k_tma_desc_ptrs": self.indexer_k_tma_desc_ptrs if k_quant_aliases_workspace else None,
            "k_tma_prefill_desc_ptrs": (
                self.indexer_k_tma_prefill_desc_ptrs if k_quant_aliases_workspace else None
            ),
        }

    def stage_nsa_indexer_paged_decode(
        self,
        *,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        real_page_table: torch.Tensor,
        seqlens_per_query: torch.Tensor,
        active_width: torch.Tensor,
        schedule_metadata: torch.Tensor | None = None,
        width_tokens: int,
    ) -> dict[str, torch.Tensor]:
        if self.indexer_paged_logits is None:
            raise RuntimeError("fixed-capacity workspace is missing paged NSA indexer buffers")
        if q_fp8.device != self.device:
            raise ValueError(f"q_fp8 device {q_fp8.device} does not match workspace device {self.device}")
        if weights.device != self.device:
            raise ValueError(
                f"weights device {weights.device} does not match workspace device {self.device}"
            )
        if real_page_table.device != self.device:
            raise ValueError(
                "real_page_table device "
                f"{real_page_table.device} does not match workspace device {self.device}"
            )
        if seqlens_per_query.device != self.device:
            raise ValueError(
                "seqlens_per_query device "
                f"{seqlens_per_query.device} does not match workspace device {self.device}"
            )
        if active_width.device != self.device:
            raise ValueError(
                f"active_width device {active_width.device} does not match workspace device {self.device}"
            )
        if not q_fp8.is_contiguous():
            raise ValueError("workspace-backed paged decode requires contiguous q_fp8")
        if not weights.is_contiguous():
            raise ValueError("workspace-backed paged decode requires contiguous weights")
        if not real_page_table.is_contiguous():
            raise ValueError("workspace-backed paged decode requires contiguous real_page_table")
        if not seqlens_per_query.is_contiguous():
            raise ValueError("workspace-backed paged decode requires contiguous seqlens_per_query")
        if not active_width.is_contiguous():
            raise ValueError("workspace-backed paged decode requires contiguous active_width")
        if schedule_metadata is not None and not schedule_metadata.is_contiguous():
            raise ValueError("workspace-backed paged decode requires contiguous schedule_metadata")

        q_rows = int(q_fp8.shape[0])
        width_tokens = int(width_tokens)
        if q_rows > self.max_paged_q_rows:
            raise ValueError(
                f"q rows {q_rows} exceed workspace NSA paged capacity {self.max_paged_q_rows}"
            )
        if q_fp8.ndim != 3 or q_fp8.shape[1] != self.indexer_num_q_heads:
            raise ValueError(
                "q_fp8 must have shape "
                f"(q_rows, {self.indexer_num_q_heads}, {_INDEX_HEAD_DIM}), got "
                f"{tuple(q_fp8.shape)}"
            )
        if q_fp8.shape[2] != _INDEX_HEAD_DIM:
            raise ValueError(
                f"q_fp8 trailing dimension must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}"
            )
        if weights.ndim != 2 or weights.shape != (q_rows, self.indexer_num_q_heads):
            raise ValueError(
                "weights must have shape "
                f"({q_rows}, {self.indexer_num_q_heads}), got {tuple(weights.shape)}"
            )
        if real_page_table.ndim != 2:
            raise ValueError(
                f"real_page_table must be rank-2, got {tuple(real_page_table.shape)}"
            )
        if real_page_table.shape[0] != q_rows:
            raise ValueError(
                f"real_page_table rows {real_page_table.shape[0]} do not match q rows {q_rows}"
            )
        if real_page_table.shape[1] > self.max_page_table_width:
            raise ValueError(
                "real_page_table width "
                f"{real_page_table.shape[1]} exceeds workspace page-table capacity "
                f"{self.max_page_table_width}"
            )
        if real_page_table.dtype != torch.int32:
            raise ValueError(
                f"real_page_table must have dtype torch.int32, got {real_page_table.dtype}"
            )
        if seqlens_per_query.ndim != 1 or seqlens_per_query.shape[0] != q_rows:
            raise ValueError(
                "seqlens_per_query must be rank-1 with q_rows entries, got "
                f"{tuple(seqlens_per_query.shape)} for q_rows={q_rows}"
            )
        if seqlens_per_query.dtype != torch.int32:
            raise ValueError(
                "seqlens_per_query must have dtype torch.int32, got "
                f"{seqlens_per_query.dtype}"
            )
        if active_width.shape != (1,):
            raise ValueError(f"active_width must have shape (1,), got {tuple(active_width.shape)}")
        if active_width.dtype != torch.int32:
            raise ValueError(
                f"active_width must have dtype torch.int32, got {active_width.dtype}"
            )
        if schedule_metadata is not None:
            if schedule_metadata.device != self.device:
                raise ValueError(
                    "schedule_metadata device "
                    f"{schedule_metadata.device} does not match workspace device {self.device}"
                )
            if schedule_metadata.ndim != 2 or schedule_metadata.shape[1] != 2:
                raise ValueError(
                    "schedule_metadata must have shape (num_sms + 1, 2), got "
                    f"{tuple(schedule_metadata.shape)}"
                )
            if schedule_metadata.dtype != torch.int32:
                raise ValueError(
                    "schedule_metadata must have dtype torch.int32, got "
                    f"{schedule_metadata.dtype}"
                )
        if width_tokens < 0:
            raise ValueError(f"width_tokens must be non-negative, got {width_tokens}")
        max_width_tokens = int(self.max_page_table_width) * int(self.page_size)
        if width_tokens > max_width_tokens:
            raise ValueError(
                f"width_tokens {width_tokens} exceed workspace logits capacity {max_width_tokens}"
            )

        q_bytes = q_fp8.view(torch.uint8)
        logits_view = self.indexer_paged_logits.narrow(0, 0, q_rows * width_tokens).view(
            q_rows, width_tokens
        )
        if q_rows != 0 and width_tokens != 0:
            logits_view.fill_(float("-inf"))
        real_page_table_kernel = real_page_table
        seqlens_per_query_kernel = seqlens_per_query
        active_width_kernel = active_width
        schedule_metadata_kernel = schedule_metadata
        if self.use_cuda_graph:
            self._allocate_runtime_metadata()
            assert self.paged_indexer_real_page_table_runtime is not None
            assert self.paged_indexer_seqlens_per_query_runtime is not None
            assert self.paged_indexer_active_width_runtime is not None
            rows, page_width = real_page_table.shape
            self.paged_indexer_real_page_table_runtime[:rows, :page_width].copy_(
                real_page_table
            )
            self.paged_indexer_seqlens_per_query_runtime[:q_rows].copy_(seqlens_per_query)
            self.paged_indexer_active_width_runtime.copy_(active_width)
            real_page_table_kernel = self.paged_indexer_real_page_table_runtime[
                :rows, :page_width
            ]
            seqlens_per_query_kernel = self.paged_indexer_seqlens_per_query_runtime[:q_rows]
            active_width_kernel = self.paged_indexer_active_width_runtime
            if schedule_metadata is not None:
                assert self.paged_indexer_schedule_metadata_runtime is not None
                schedule_rows = schedule_metadata.shape[0]
                if schedule_rows > self.paged_indexer_schedule_metadata_runtime.shape[0]:
                    raise ValueError(
                        "schedule_metadata rows "
                        f"{schedule_rows} exceed workspace schedule capacity "
                        f"{self.paged_indexer_schedule_metadata_runtime.shape[0]}"
                    )
                self.paged_indexer_schedule_metadata_runtime[
                    :schedule_rows, :
                ].copy_(schedule_metadata)
                schedule_metadata_kernel = self.paged_indexer_schedule_metadata_runtime[
                    :schedule_rows, :
                ]
        return {
            "q_bytes": q_bytes,
            "weights": weights,
            "real_page_table": real_page_table_kernel,
            "seqlens_per_query": seqlens_per_query_kernel,
            "active_width": active_width_kernel,
            "schedule_metadata": schedule_metadata_kernel,
            "logits": logits_view,
            "logits_view": logits_view,
        }

    def contract_kv_tensors_for(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return stable KV phantoms only for the ragged scratch allocation.

        Extend/verify share a workspace in SGLang. After a ragged prefill allocates
        `ragged_kv_cache`, later paged launches must not reuse those KV phantoms or
        they can collide with a launcher compiled for a different KV layout.
        """
        buffer = self.ragged_kv_cache
        if buffer is None:
            return None, None
        if kv_cache.device != buffer.device or kv_cache.dtype != buffer.dtype:
            return None, None
        if kv_cache.ndim != buffer.ndim:
            return None, None
        if kv_cache.data_ptr() != buffer.data_ptr():
            return None, None
        if tuple(int(dim) for dim in kv_cache.shape[1:]) != tuple(
            int(dim) for dim in buffer.shape[1:]
        ):
            return None, None
        return self._contract_kv_rows, self._contract_kv_scales

    def _refresh_ragged_kv_contracts(self) -> None:
        if self.ragged_kv_cache is None:
            self._contract_kv_rows = None
            self._contract_kv_scales = None
            return

        from .kernel import _extract_packed_kv_runtime_views

        kv_rows_u32, kv_scales = _extract_packed_kv_runtime_views(self.ragged_kv_cache)
        self._contract_kv_rows = _shape_only_cuda_tensor(
            tuple(int(dim) for dim in kv_rows_u32.shape),
            dtype=kv_rows_u32.dtype,
            device=self.device,
        )
        self._contract_kv_scales = _shape_only_cuda_tensor(
            tuple(int(dim) for dim in kv_scales.shape),
            dtype=kv_scales.dtype,
            device=self.device,
        )

    def _allocate_contract_phantoms(self) -> None:
        """Create zero-stride phantom tensors at max capacity for stable cache keys."""
        # q is viewed as uint32 in the kernel: (max_total_q, num_q_heads, head_dim // 4).
        self._contract_q = _shape_only_cuda_tensor(
            (self.max_total_q, self.num_q_heads, self.head_dim // 4),
            dtype=torch.uint32,
            device=self.device,
        )
        self._contract_page_table = _shape_only_cuda_tensor(
            (self.max_total_q, self.topk),
            dtype=torch.int32,
            device=self.device,
        )
        self._contract_nsa_cache_seqlens = _shape_only_cuda_tensor(
            (self.max_total_q,),
            dtype=torch.int32,
            device=self.device,
        )
        self._contract_output = _shape_only_cuda_tensor(
            (self.max_total_q, self.num_q_heads, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        if self.tmp_output is not None and self.tmp_lse is not None:
            self._contract_tmp_output = _shape_only_cuda_tensor(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row, self.v_head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self._contract_tmp_lse = _shape_only_cuda_tensor(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row),
                dtype=torch.float32,
                device=self.device,
            )
        if self.ragged_kv_cache is not None:
            self._refresh_ragged_kv_contracts()
        self._contract_indexer_q_u32 = _shape_only_cuda_tensor(
            (self.max_total_q, self.indexer_num_q_heads, _INDEX_HEAD_DIM // 4),
            dtype=torch.uint32,
            device=self.device,
        )
        self._contract_indexer_q_bytes = _shape_only_cuda_tensor(
            (self.max_total_q, self.indexer_num_q_heads, _INDEX_HEAD_DIM),
            dtype=torch.uint8,
            device=self.device,
        )
        self._contract_indexer_weights = _shape_only_cuda_tensor(
            (self.max_total_q, self.indexer_num_q_heads),
            dtype=torch.float32,
            device=self.device,
        )
        if self.indexer_k_quant_bytes is not None:
            self._contract_indexer_k_quant = _shape_only_cuda_tensor(
                tuple(int(dim) for dim in self.indexer_k_quant_bytes.shape),
                dtype=torch.uint8,
                device=self.device,
            )
        if self.indexer_k_scales is not None:
            self._contract_indexer_k_scale = _shape_only_cuda_tensor(
                tuple(int(dim) for dim in self.indexer_k_scales.shape),
                dtype=torch.float32,
                device=self.device,
            )
        self._contract_indexer_k_start = _shape_only_cuda_tensor(
            (self.max_total_q,),
            dtype=torch.int32,
            device=self.device,
        )
        self._contract_indexer_k_end = _shape_only_cuda_tensor(
            (self.max_total_q,),
            dtype=torch.int32,
            device=self.device,
        )
        if self.indexer_extend_logits is not None and self.indexer_k_quant_bytes is not None:
            self._contract_indexer_logits = _shape_only_cuda_tensor(
                (self.max_total_q, int(self.indexer_k_quant_bytes.shape[0])),
                dtype=torch.float32,
                device=self.device,
            )
        if self.indexer_extend_tile_logits is not None:
            self._contract_indexer_tile_logits = _shape_only_cuda_tensor(
                tuple(int(dim) for dim in self.indexer_extend_tile_logits.shape),
                dtype=torch.float32,
                device=self.device,
            )
        if self.indexer_paged_logits is not None:
            paged_width_tokens = int(self.max_page_table_width) * int(self.page_size)
            self._contract_paged_indexer_q_bytes = _shape_only_cuda_tensor(
                (self.max_paged_q_rows, self.indexer_num_q_heads, _INDEX_HEAD_DIM),
                dtype=torch.uint8,
                device=self.device,
            )
            self._contract_paged_indexer_weights = _shape_only_cuda_tensor(
                (self.max_paged_q_rows, self.indexer_num_q_heads),
                dtype=torch.float32,
                device=self.device,
            )
            self._contract_paged_real_page_table = _shape_only_cuda_tensor(
                (self.max_paged_q_rows, self.max_page_table_width),
                dtype=torch.int32,
                device=self.device,
            )
            self._contract_paged_nsa_cache_seqlens = _shape_only_cuda_tensor(
                (self.max_paged_q_rows,),
                dtype=torch.int32,
                device=self.device,
            )
            self._contract_paged_indexer_logits = _shape_only_cuda_tensor(
                (self.max_paged_q_rows, paged_width_tokens),
                dtype=torch.float32,
                device=self.device,
            )
