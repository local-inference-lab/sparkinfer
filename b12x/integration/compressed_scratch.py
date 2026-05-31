"""Caller-owned scratch plans for compressed MLA and compressed indexer paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x.attention.mla.compressed_config import (
    compressed_mla_split_config_for_contract,
)
from b12x.attention.mla.compressed_reference import COMPRESSED_MLA_HEAD_DIM
from b12x.attention.workspace import (
    _ARENA_ALIGN_BYTES,
    B12XAttentionArena,
    B12XAttentionArenaCaps,
    B12XAttentionWorkspace,
    B12XAttentionWorkspaceContract,
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
    scratch: B12XAttentionWorkspace
    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    active_width: torch.Tensor
    schedule_metadata: torch.Tensor | None = None
    expected_num_q_heads: int | None = None
    shared_page_table: bool = False


def _arena_spec(name: str, caps: B12XAttentionArenaCaps) -> B12XScratchBufferSpec:
    nbytes = B12XAttentionArena.required_nbytes(caps)
    return scratch_buffer_spec(name, nbytes=int(nbytes), device=caps.device)


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


def _validate_i32_contiguous(tensor: torch.Tensor, *, workspace: B12XAttentionWorkspace, name: str, ndim: int) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank-{ndim}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, workspace=workspace, name=name)


def build_compressed_indexer_binding(
    *,
    workspace: B12XAttentionWorkspace,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    active_width: torch.Tensor | None = None,
    schedule_metadata: torch.Tensor | None = None,
    expected_num_q_heads: int | None = None,
    shared_page_table: bool = False,
) -> B12XCompressedIndexerBinding:
    _validate_i32_contiguous(real_page_table, workspace=workspace, name="real_page_table", ndim=2)
    _validate_i32_contiguous(cache_seqlens_int32, workspace=workspace, name="cache_seqlens_int32", ndim=1)
    if active_width is None:
        active_width = workspace.get_paged_indexer_active_width_cap()
    _validate_i32_contiguous(active_width, workspace=workspace, name="active_width", ndim=1)
    if active_width.shape != (1,):
        raise ValueError(f"active_width must have shape (1,), got {tuple(active_width.shape)}")
    if int(real_page_table.shape[0]) != int(cache_seqlens_int32.shape[0]):
        raise ValueError(
            f"real_page_table rows {int(real_page_table.shape[0])} do not match "
            f"cache_seqlens_int32 rows {int(cache_seqlens_int32.shape[0])}"
        )
    if int(real_page_table.shape[0]) > int(workspace.max_paged_q_rows):
        raise ValueError(
            f"real_page_table rows {int(real_page_table.shape[0])} exceed workspace paged capacity "
            f"{workspace.max_paged_q_rows}"
        )
    if int(real_page_table.shape[1]) > int(workspace.max_page_table_width):
        raise ValueError(
            f"real_page_table width {int(real_page_table.shape[1])} exceeds workspace capacity "
            f"{workspace.max_page_table_width}"
        )
    if schedule_metadata is not None:
        _validate_i32_contiguous(schedule_metadata, workspace=workspace, name="schedule_metadata", ndim=2)
        if int(schedule_metadata.shape[1]) != 2:
            raise ValueError(f"schedule_metadata must have shape (num_sms + 1, 2), got {tuple(schedule_metadata.shape)}")
    if expected_num_q_heads is not None:
        expected_num_q_heads = int(expected_num_q_heads)
        if expected_num_q_heads <= 0:
            raise ValueError(f"expected_num_q_heads must be positive, got {expected_num_q_heads}")
    return B12XCompressedIndexerBinding(
        scratch=workspace,
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
    arena_caps: B12XAttentionArenaCaps
    contract: B12XAttentionWorkspaceContract
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
        arena_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="compressed indexer",
        )
        arena = B12XAttentionArena.from_shared_arena(self.arena_caps, arena_storage)
        workspace = arena._make_workspace_views(self.contract, use_cuda_graph=False)
        return build_compressed_indexer_binding(
            workspace=workspace,
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
    arena_caps = B12XAttentionArenaCaps(
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=1,
        indexer_num_q_heads=caps.num_q_heads,
        head_dim=1,
        max_v_head_dim=1,
        topk=caps.topk,
        indexer_topk=caps.topk,
        max_page_table_width=caps.max_page_table_width,
        extend_max_total_q=caps.max_q_rows,
        extend_max_batch=caps.max_batch,
        extend_max_kv_rows=0,
        paged_max_q_rows=caps.max_q_rows,
        paged_max_batch=caps.max_batch,
        indexer_max_k_rows=caps.max_k_rows,
        page_size=caps.page_size,
        max_chunks_per_row=1,
        reserve_extend_indexer_logits=False,
        reserve_paged_indexer_logits=caps.reserve_paged_logits,
        paged_indexer_logits_q_rows=caps.max_q_rows,
        paged_indexer_logits_k_rows=caps.paged_logits_k_rows,
        paged_indexer_tile_logits_k_rows=caps.paged_tile_logits_k_rows,
    )
    contract = B12XAttentionWorkspaceContract(
        mode="decode",
        max_total_q=caps.max_q_rows,
        max_batch=caps.max_batch,
        max_paged_q_rows=caps.max_q_rows,
        max_kv_rows=0,
        v_head_dim=1,
        indexer_num_q_heads=caps.num_q_heads,
        max_page_table_width=caps.max_page_table_width,
        topk=caps.topk,
        max_chunks_per_row=1,
    )
    return B12XCompressedIndexerScratchPlan(
        caps=caps,
        arena_caps=arena_caps,
        contract=contract,
        _scratch_specs=(_arena_spec("compressed_indexer.arena", arena_caps),),
    )


__all__ = [
    "B12XScratchBufferSpec",
    "B12XCompressedIndexerBinding",
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
