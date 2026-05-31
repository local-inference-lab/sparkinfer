"""Caller-owned scratch plans for sparse NSA indexer paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x.attention.indexer.api import IndexerExtendMetadata, IndexerPagedDecodeMetadata
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
class B12XIndexerPagedScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_page_table_width: int
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.uint8
    topk: int = 1
    max_batch: int | None = None
    page_size: int = 64
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
        object.__setattr__(self, "reserve_paged_logits", bool(self.reserve_paged_logits))
        object.__setattr__(self, "paged_logits_k_rows", max(int(self.paged_logits_k_rows), 0))
        object.__setattr__(
            self,
            "paged_tile_logits_k_rows",
            max(int(self.paged_tile_logits_k_rows), 0),
        )


@dataclass(frozen=True, kw_only=True)
class B12XIndexerExtendScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_k_rows: int
    topk: int
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.uint8
    max_batch: int | None = None
    page_size: int = 64
    reserve_extend_logits: bool = True
    extend_tile_logits_k_rows: int = 0

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "max_q_rows", max(int(self.max_q_rows), 1))
        object.__setattr__(self, "max_k_rows", max(int(self.max_k_rows), 1))
        object.__setattr__(self, "topk", max(int(self.topk), 1))
        max_batch = self.max_q_rows if self.max_batch is None else self.max_batch
        object.__setattr__(self, "max_batch", max(int(max_batch), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))
        object.__setattr__(self, "reserve_extend_logits", bool(self.reserve_extend_logits))
        object.__setattr__(
            self,
            "extend_tile_logits_k_rows",
            max(int(self.extend_tile_logits_k_rows), 0),
        )


@dataclass(frozen=True, kw_only=True)
class B12XIndexerPagedBinding:
    scratch: B12XAttentionWorkspace
    metadata: IndexerPagedDecodeMetadata
    active_width: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class B12XIndexerExtendBinding:
    scratch: B12XAttentionWorkspace
    metadata: IndexerExtendMetadata
    gather_k_quant: torch.Tensor | None = None
    gather_k_scale: torch.Tensor | None = None
    topk: int | None = None
    contract_phantoms: dict[str, torch.Tensor] | None = None
    tile_logits: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    output_values: torch.Tensor | None = None
    output_indices: torch.Tensor | None = None
    candidate_values: torch.Tensor | None = None
    candidate_indices: torch.Tensor | None = None
    merge_positions: torch.Tensor | None = None


def _validate_device(
    tensor: torch.Tensor,
    *,
    workspace: B12XAttentionWorkspace,
    name: str,
) -> None:
    if tensor.device != workspace.device:
        raise ValueError(f"{name} device {tensor.device} does not match workspace device {workspace.device}")


def _validate_i32_contiguous(
    tensor: torch.Tensor,
    *,
    workspace: B12XAttentionWorkspace,
    name: str,
    ndim: int,
) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank-{ndim}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, workspace=workspace, name=name)


def _maybe_indexer_extend_contract_phantoms(
    workspace: B12XAttentionWorkspace,
) -> dict[str, torch.Tensor] | None:
    if not workspace.fixed_capacity or workspace.contract is None:
        return None
    return workspace.get_indexer_contract_phantoms()


def _maybe_workspace_buffer(fn, **kwargs):
    try:
        return fn(**kwargs)
    except RuntimeError:
        return None


def build_indexer_paged_binding(
    *,
    workspace: B12XAttentionWorkspace,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    active_width: torch.Tensor | None = None,
    paged_mqa_schedule_metadata: torch.Tensor | None = None,
) -> B12XIndexerPagedBinding:
    _validate_i32_contiguous(real_page_table, workspace=workspace, name="real_page_table", ndim=2)
    _validate_i32_contiguous(
        cache_seqlens_int32,
        workspace=workspace,
        name="cache_seqlens_int32",
        ndim=1,
    )
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
    if paged_mqa_schedule_metadata is not None:
        _validate_i32_contiguous(
            paged_mqa_schedule_metadata,
            workspace=workspace,
            name="paged_mqa_schedule_metadata",
            ndim=2,
        )
        if int(paged_mqa_schedule_metadata.shape[1]) != 2:
            raise ValueError(
                "paged_mqa_schedule_metadata must have shape (num_sms + 1, 2), "
                f"got {tuple(paged_mqa_schedule_metadata.shape)}"
            )
    return B12XIndexerPagedBinding(
        scratch=workspace,
        metadata=IndexerPagedDecodeMetadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
        ),
        active_width=active_width,
    )


def build_indexer_extend_binding(
    *,
    workspace: B12XAttentionWorkspace,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    gather_rows: int | None = None,
    topk: int | None = None,
    include_topk_buffers: bool = True,
    include_candidate_buffers: bool = True,
    include_lengths: bool = True,
    include_merge_positions: bool = True,
) -> B12XIndexerExtendBinding:
    _validate_i32_contiguous(k_start, workspace=workspace, name="k_start", ndim=1)
    _validate_i32_contiguous(k_end, workspace=workspace, name="k_end", ndim=1)
    if k_start.shape != k_end.shape:
        raise ValueError(
            f"k_start and k_end must have the same shape, got {tuple(k_start.shape)} "
            f"vs {tuple(k_end.shape)}"
        )
    row_count = int(k_start.shape[0])
    if row_count > int(workspace.max_total_q):
        raise ValueError(
            f"k_start rows {row_count} exceed workspace extend capacity {workspace.max_total_q}"
        )

    resolved_topk = None
    if topk is not None:
        resolved_topk = int(topk)
        if resolved_topk < 0:
            raise ValueError(f"topk must be non-negative, got {resolved_topk}")
        if resolved_topk > int(workspace.indexer_topk):
            raise ValueError(
                f"topk {resolved_topk} exceeds workspace indexer topk {workspace.indexer_topk}"
            )

    contract_phantoms = _maybe_indexer_extend_contract_phantoms(workspace)
    gather_k_quant = None
    gather_k_scale = None
    if gather_rows is not None:
        gather_k_quant, gather_k_scale = workspace.get_indexer_gather_outputs(
            row_count=int(gather_rows),
        )
    tile_logits = workspace.get_indexer_extend_tile_logits()
    lengths = None
    output_values = None
    output_indices = None
    candidate_values = None
    candidate_indices = None
    merge_positions = None

    if resolved_topk is not None:
        if include_lengths:
            lengths = _maybe_workspace_buffer(
                workspace.get_indexer_extend_lengths,
                row_count=row_count,
            )
        if include_topk_buffers:
            topk_buffers = _maybe_workspace_buffer(
                workspace.get_indexer_extend_topk_buffers,
                row_count=row_count,
            )
            if topk_buffers is not None:
                output_values, output_indices = topk_buffers
                output_values = output_values[:, :resolved_topk]
                output_indices = output_indices[:, :resolved_topk]
        if include_candidate_buffers:
            candidate_buffers = _maybe_workspace_buffer(
                workspace.get_indexer_extend_candidate_buffers,
            )
            if candidate_buffers is not None:
                candidate_values, candidate_indices = candidate_buffers
                candidate_values = candidate_values[:, :row_count, :resolved_topk]
                candidate_indices = candidate_indices[:, :row_count, :resolved_topk]
        if include_merge_positions:
            merge_positions = _maybe_workspace_buffer(
                workspace.get_indexer_extend_topk_position_buffer,
                row_count=row_count,
            )
            if merge_positions is not None:
                merge_positions = merge_positions[:, :resolved_topk]

    return B12XIndexerExtendBinding(
        scratch=workspace,
        metadata=IndexerExtendMetadata(k_start=k_start, k_end=k_end),
        gather_k_quant=gather_k_quant,
        gather_k_scale=gather_k_scale,
        topk=resolved_topk,
        contract_phantoms=contract_phantoms,
        tile_logits=tile_logits,
        lengths=lengths,
        output_values=output_values,
        output_indices=output_indices,
        candidate_values=candidate_values,
        candidate_indices=candidate_indices,
        merge_positions=merge_positions,
    )


@dataclass(frozen=True)
class B12XIndexerPagedScratchPlan:
    caps: B12XIndexerPagedScratchCaps
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
        paged_mqa_schedule_metadata: torch.Tensor | None = None,
    ) -> B12XIndexerPagedBinding:
        arena_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="paged indexer",
        )
        arena = B12XAttentionArena.from_shared_arena(self.arena_caps, arena_storage)
        workspace = arena._make_workspace_views(self.contract, use_cuda_graph=False)
        return build_indexer_paged_binding(
            workspace=workspace,
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            active_width=active_width,
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
    )


@dataclass(frozen=True)
class B12XIndexerExtendScratchPlan:
    caps: B12XIndexerExtendScratchCaps
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
        k_start: torch.Tensor,
        k_end: torch.Tensor,
        gather_rows: int | None = None,
        topk: int | None = None,
        include_topk_buffers: bool = True,
        include_candidate_buffers: bool = True,
        include_lengths: bool = True,
        include_merge_positions: bool = True,
    ) -> B12XIndexerExtendBinding:
        arena_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="indexer extend",
        )
        arena = B12XAttentionArena.from_shared_arena(self.arena_caps, arena_storage)
        workspace = arena._make_workspace_views(self.contract, use_cuda_graph=False)
        return build_indexer_extend_binding(
            workspace=workspace,
            k_start=k_start,
            k_end=k_end,
            gather_rows=gather_rows,
            topk=self.caps.topk if topk is None else topk,
            include_topk_buffers=include_topk_buffers,
            include_candidate_buffers=include_candidate_buffers,
            include_lengths=include_lengths,
            include_merge_positions=include_merge_positions,
        )


def plan_indexer_paged_scratch(
    caps: B12XIndexerPagedScratchCaps,
) -> B12XIndexerPagedScratchPlan:
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
    return B12XIndexerPagedScratchPlan(
        caps=caps,
        arena_caps=arena_caps,
        contract=contract,
        _scratch_specs=(
            scratch_buffer_spec(
                "indexer_paged.arena",
                nbytes=B12XAttentionArena.required_nbytes(arena_caps),
                device=arena_caps.device,
            ),
        ),
    )


def plan_indexer_extend_scratch(
    caps: B12XIndexerExtendScratchCaps,
) -> B12XIndexerExtendScratchPlan:
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
        max_page_table_width=1,
        extend_max_total_q=caps.max_q_rows,
        extend_max_batch=caps.max_batch,
        extend_max_kv_rows=0,
        paged_max_q_rows=1,
        paged_max_batch=1,
        indexer_max_k_rows=caps.max_k_rows,
        page_size=caps.page_size,
        max_chunks_per_row=1,
        reserve_extend_indexer_logits=caps.reserve_extend_logits,
        reserve_paged_indexer_logits=False,
        extend_indexer_tile_logits_k_rows=caps.extend_tile_logits_k_rows,
    )
    contract = B12XAttentionWorkspaceContract(
        mode="extend",
        max_total_q=caps.max_q_rows,
        max_batch=caps.max_batch,
        max_paged_q_rows=1,
        max_kv_rows=0,
        v_head_dim=1,
        indexer_num_q_heads=caps.num_q_heads,
        max_page_table_width=1,
        topk=caps.topk,
        max_chunks_per_row=1,
    )
    return B12XIndexerExtendScratchPlan(
        caps=caps,
        arena_caps=arena_caps,
        contract=contract,
        _scratch_specs=(
            scratch_buffer_spec(
                "indexer_extend.arena",
                nbytes=B12XAttentionArena.required_nbytes(arena_caps),
                device=arena_caps.device,
            ),
        ),
    )


__all__ = [
    "B12XIndexerExtendBinding",
    "B12XIndexerExtendScratchCaps",
    "B12XIndexerExtendScratchPlan",
    "B12XIndexerPagedBinding",
    "B12XIndexerPagedScratchCaps",
    "B12XIndexerPagedScratchPlan",
    "build_indexer_extend_binding",
    "build_indexer_paged_binding",
    "plan_indexer_extend_scratch",
    "plan_indexer_paged_scratch",
]
