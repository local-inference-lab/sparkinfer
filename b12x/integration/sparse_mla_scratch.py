"""Caller-owned scratch plans for sparse MLA paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from b12x.attention.workspace import (
    B12XAttentionArena,
    B12XAttentionArenaCaps,
    B12XAttentionWorkspace,
    B12XAttentionWorkspaceContract,
)
from b12x.integration.scratch import (
    B12XScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)


@dataclass(frozen=True, kw_only=True)
class B12XSparseMLAScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_width: int
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.bfloat16
    head_dim: int = 576
    v_head_dim: int = 512
    mode: Literal["decode", "extend", "verify", "draft_extend"] = "decode"
    max_batch: int | None = None
    max_kv_rows: int = 0
    max_page_table_width: int | None = None
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
        object.__setattr__(self, "head_dim", max(int(self.head_dim), 1))
        object.__setattr__(self, "v_head_dim", max(int(self.v_head_dim), 1))
        max_batch = self.max_q_rows if self.max_batch is None else self.max_batch
        object.__setattr__(self, "max_batch", max(int(max_batch), 1))
        object.__setattr__(self, "max_kv_rows", max(int(self.max_kv_rows), 0))
        max_page_table_width = (
            self.max_width if self.max_page_table_width is None else self.max_page_table_width
        )
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(max_page_table_width), 1),
        )
        object.__setattr__(
            self,
            "max_chunks_per_row",
            max(int(self.max_chunks_per_row), 1),
        )
        if self.max_q_chunks is not None:
            object.__setattr__(self, "max_q_chunks", max(int(self.max_q_chunks), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))


@dataclass(frozen=True, kw_only=True)
class B12XSparseMLABinding:
    scratch: B12XAttentionWorkspace
    q: torch.Tensor
    selected_indices: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor


def _validate_device(
    tensor: torch.Tensor,
    *,
    workspace: B12XAttentionWorkspace,
    name: str,
) -> None:
    if tensor.device != workspace.device:
        raise ValueError(f"{name} device {tensor.device} does not match workspace device {workspace.device}")


def _validate_q(q: torch.Tensor, *, workspace: B12XAttentionWorkspace) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError(f"q must be rank-3, got {tuple(q.shape)}")
    if q.dtype != workspace.dtype:
        raise TypeError(f"q must have dtype {workspace.dtype}, got {q.dtype}")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    _validate_device(q, workspace=workspace, name="q")
    if int(q.shape[0]) > int(workspace.max_total_q):
        raise ValueError(f"q rows {int(q.shape[0])} exceed workspace capacity {workspace.max_total_q}")
    if int(q.shape[1]) != int(workspace.num_q_heads):
        raise ValueError(f"q heads {int(q.shape[1])} do not match workspace heads {workspace.num_q_heads}")
    if int(q.shape[2]) != int(workspace.head_dim):
        raise ValueError(f"q head_dim {int(q.shape[2])} does not match workspace head_dim {workspace.head_dim}")
    return q.detach()


def _validate_selected_indices(
    selected_indices: torch.Tensor,
    *,
    workspace: B12XAttentionWorkspace,
    rows: int,
) -> torch.Tensor:
    if selected_indices.ndim != 2:
        raise ValueError(f"selected_indices must be rank-2, got {tuple(selected_indices.shape)}")
    if selected_indices.dtype != torch.int32:
        raise TypeError(f"selected_indices must have dtype torch.int32, got {selected_indices.dtype}")
    if not selected_indices.is_contiguous():
        raise ValueError("selected_indices must be contiguous")
    _validate_device(selected_indices, workspace=workspace, name="selected_indices")
    if int(selected_indices.shape[0]) != int(rows):
        raise ValueError(
            f"selected_indices rows {int(selected_indices.shape[0])} do not match q rows {rows}"
        )
    if int(selected_indices.shape[1]) > int(workspace.topk):
        raise ValueError(
            f"selected_indices width {int(selected_indices.shape[1])} exceeds workspace topk {workspace.topk}"
        )
    return selected_indices


def _validate_i32_vector(
    tensor: torch.Tensor,
    *,
    workspace: B12XAttentionWorkspace,
    name: str,
    max_rows: int | None = None,
    rows: int | None = None,
) -> torch.Tensor:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, workspace=workspace, name=name)
    if rows is not None and int(tensor.shape[0]) != int(rows):
        raise ValueError(f"{name} rows {int(tensor.shape[0])} do not match q rows {rows}")
    if max_rows is not None and int(tensor.shape[0]) > int(max_rows):
        raise ValueError(f"{name} rows {int(tensor.shape[0])} exceed capacity {max_rows}")
    return tensor


def build_sparse_mla_binding(
    *,
    workspace: B12XAttentionWorkspace,
    q: torch.Tensor,
    selected_indices: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    nsa_cache_seqlens_int32: torch.Tensor,
) -> B12XSparseMLABinding:
    q = _validate_q(q, workspace=workspace)
    rows = int(q.shape[0])
    selected_indices = _validate_selected_indices(
        selected_indices,
        workspace=workspace,
        rows=rows,
    )
    cache_seqlens_int32 = _validate_i32_vector(
        cache_seqlens_int32,
        workspace=workspace,
        name="cache_seqlens_int32",
        max_rows=workspace.max_batch,
    )
    nsa_cache_seqlens_int32 = _validate_i32_vector(
        nsa_cache_seqlens_int32,
        workspace=workspace,
        name="nsa_cache_seqlens_int32",
        rows=rows,
    )
    return B12XSparseMLABinding(
        scratch=workspace,
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
    )


@dataclass(frozen=True)
class B12XSparseMLAScratchPlan:
    caps: B12XSparseMLAScratchCaps
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
        q: torch.Tensor,
        selected_indices: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> B12XSparseMLABinding:
        arena_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="sparse MLA",
        )
        arena = B12XAttentionArena.from_shared_arena(self.arena_caps, arena_storage)
        workspace = arena._make_workspace_views(self.contract, use_cuda_graph=False)
        return build_sparse_mla_binding(
            workspace=workspace,
            q=q,
            selected_indices=selected_indices,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )


def plan_sparse_mla_scratch(
    caps: B12XSparseMLAScratchCaps,
) -> B12XSparseMLAScratchPlan:
    arena_caps = B12XAttentionArenaCaps(
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=caps.num_q_heads,
        indexer_num_q_heads=1,
        head_dim=caps.head_dim,
        max_v_head_dim=caps.v_head_dim,
        topk=caps.max_width,
        indexer_topk=1,
        max_page_table_width=caps.max_page_table_width,
        extend_max_total_q=caps.max_q_rows,
        extend_max_batch=caps.max_batch,
        extend_max_kv_rows=caps.max_kv_rows,
        paged_max_q_rows=1,
        paged_max_batch=1,
        mla_max_total_q=caps.max_q_rows,
        mla_max_q_chunks=caps.max_q_chunks,
        page_size=caps.page_size,
        max_chunks_per_row=caps.max_chunks_per_row,
        reserve_extend_indexer_logits=False,
        reserve_paged_indexer_logits=False,
        reserve_compressed_mla_staging=False,
    )
    contract = B12XAttentionWorkspaceContract(
        mode=caps.mode,
        max_total_q=caps.max_q_rows,
        max_batch=caps.max_batch,
        max_paged_q_rows=1,
        max_kv_rows=caps.max_kv_rows,
        v_head_dim=caps.v_head_dim,
        indexer_num_q_heads=1,
        max_page_table_width=caps.max_page_table_width,
        topk=caps.max_width,
        max_chunks_per_row=caps.max_chunks_per_row,
    )
    return B12XSparseMLAScratchPlan(
        caps=caps,
        arena_caps=arena_caps,
        contract=contract,
        _scratch_specs=(
            scratch_buffer_spec(
                "sparse_mla.arena",
                nbytes=B12XAttentionArena.required_nbytes(arena_caps),
                device=arena_caps.device,
            ),
        ),
    )


__all__ = [
    "B12XSparseMLABinding",
    "B12XSparseMLAScratchCaps",
    "B12XSparseMLAScratchPlan",
    "build_sparse_mla_binding",
    "plan_sparse_mla_scratch",
]
