"""Caller-owned scratch plans for the primary paged-attention backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from b12x.attention.paged.workspace import (
    PagedAttentionArena,
    PagedAttentionArenaCaps,
    PagedAttentionWorkspace,
    PagedAttentionWorkspaceContract,
)
from b12x.integration.scratch import (
    B12XScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)


@dataclass(frozen=True, kw_only=True)
class B12XPagedAttentionScratchCaps:
    device: torch.device | str
    mode: Literal["decode", "extend", "verify"]
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    max_total_q: int
    max_batch: int
    max_page_table_width: int
    max_work_items: int
    max_partial_rows: int
    num_cache_pages: int
    use_cuda_graph: bool = False

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "num_kv_heads", max(int(self.num_kv_heads), 1))
        object.__setattr__(self, "head_dim_qk", max(int(self.head_dim_qk), 1))
        object.__setattr__(self, "head_dim_vo", max(int(self.head_dim_vo), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))
        object.__setattr__(self, "max_total_q", max(int(self.max_total_q), 1))
        object.__setattr__(self, "max_batch", max(int(self.max_batch), 1))
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(self.max_page_table_width), 1),
        )
        object.__setattr__(self, "max_work_items", max(int(self.max_work_items), 0))
        object.__setattr__(self, "max_partial_rows", max(int(self.max_partial_rows), 0))
        object.__setattr__(self, "num_cache_pages", max(int(self.num_cache_pages), 1))
        object.__setattr__(self, "use_cuda_graph", bool(self.use_cuda_graph))


@dataclass(frozen=True, kw_only=True)
class B12XPagedAttentionBinding:
    workspace: PagedAttentionWorkspace
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    output: torch.Tensor
    k_descale: torch.Tensor | None = None
    v_descale: torch.Tensor | None = None
    attention_sink_bias: torch.Tensor | None = None

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        from b12x.attention.paged.api import paged_attention_forward

        return paged_attention_forward(binding=self)


def _validate_optional_tensor_device(
    tensor: torch.Tensor | None,
    *,
    workspace: PagedAttentionWorkspace,
    name: str,
) -> None:
    if tensor is not None and tensor.device != workspace.device:
        raise ValueError(f"{name} device {tensor.device} does not match workspace device {workspace.device}")


def _validate_output(
    output: torch.Tensor,
    *,
    workspace: PagedAttentionWorkspace,
) -> None:
    if output.device != workspace.device:
        raise ValueError(f"output device {output.device} does not match workspace device {workspace.device}")
    if output.dtype != workspace.dtype:
        raise TypeError(f"output must have dtype {workspace.dtype}, got {output.dtype}")
    if output.ndim != 3:
        raise ValueError(f"output must be rank-3 [total_q, heads, head_dim], got {tuple(output.shape)}")
    if tuple(output.shape[1:]) != (workspace.num_q_heads, workspace.head_dim_vo):
        raise ValueError(
            "output shape does not match the workspace contract: "
            f"expected (*, {workspace.num_q_heads}, {workspace.head_dim_vo}), got {tuple(output.shape)}"
        )


def _metadata_tuple(
    *,
    page_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    values = (page_table, cache_seqlens, cu_seqlens_q)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("page_table, cache_seqlens, and cu_seqlens_q must be provided together")
    return page_table, cache_seqlens, cu_seqlens_q


def build_paged_attention_binding(
    *,
    workspace: PagedAttentionWorkspace,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    page_table: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    fixed_split_size: int | None = None,
    disable_split_kv: bool = False,
    window_left: int = -1,
    active_total_q: int | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    attention_sink_bias: torch.Tensor | None = None,
) -> B12XPagedAttentionBinding:
    workspace._validate_static_shapes(q, k_cache, v_cache)
    _validate_output(output, workspace=workspace)
    _validate_optional_tensor_device(k_descale, workspace=workspace, name="k_descale")
    _validate_optional_tensor_device(v_descale, workspace=workspace, name="v_descale")
    _validate_optional_tensor_device(
        attention_sink_bias,
        workspace=workspace,
        name="attention_sink_bias",
    )

    metadata = _metadata_tuple(
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    if metadata is not None:
        workspace.prepare(
            metadata[0],
            metadata[1],
            metadata[2],
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            window_left=window_left,
            active_total_q=active_total_q,
        )
    elif not workspace.prepared:
        raise RuntimeError("paged attention binding requires prepared workspace metadata")

    return B12XPagedAttentionBinding(
        workspace=workspace,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
        attention_sink_bias=attention_sink_bias,
    )


@dataclass(frozen=True)
class B12XPagedAttentionScratchPlan:
    caps: B12XPagedAttentionScratchCaps
    arena_caps: PagedAttentionArenaCaps
    contract: PagedAttentionWorkspaceContract
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
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
        window_left: int = -1,
        active_total_q: int | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        attention_sink_bias: torch.Tensor | None = None,
    ) -> B12XPagedAttentionBinding:
        arena_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="paged attention",
        )
        arena = PagedAttentionArena.from_shared_arena(self.arena_caps, arena_storage)
        workspace = arena.make_workspace(
            self.contract,
            use_cuda_graph=self.caps.use_cuda_graph,
        )
        return build_paged_attention_binding(
            workspace=workspace,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            window_left=window_left,
            active_total_q=active_total_q,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=attention_sink_bias,
        )


def plan_paged_attention_scratch(
    caps: B12XPagedAttentionScratchCaps,
) -> B12XPagedAttentionScratchPlan:
    arena_caps = PagedAttentionArenaCaps(
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=caps.num_q_heads,
        num_kv_heads=caps.num_kv_heads,
        head_dim_qk=caps.head_dim_qk,
        max_head_dim_vo=caps.head_dim_vo,
        page_size=caps.page_size,
        max_total_q=caps.max_total_q,
        max_batch=caps.max_batch,
        max_page_table_width=caps.max_page_table_width,
        max_work_items=caps.max_work_items,
        max_partial_rows=caps.max_partial_rows,
    )
    contract = PagedAttentionWorkspaceContract(
        mode=caps.mode,
        max_total_q=caps.max_total_q,
        max_batch=caps.max_batch,
        max_page_table_width=caps.max_page_table_width,
        max_work_items=caps.max_work_items,
        max_partial_rows=caps.max_partial_rows,
        num_q_heads=caps.num_q_heads,
        num_kv_heads=caps.num_kv_heads,
        head_dim_qk=caps.head_dim_qk,
        head_dim_vo=caps.head_dim_vo,
        num_cache_pages=caps.num_cache_pages,
    )
    return B12XPagedAttentionScratchPlan(
        caps=caps,
        arena_caps=arena_caps,
        contract=contract,
        _scratch_specs=(
            scratch_buffer_spec(
                "paged_attention.arena",
                nbytes=PagedAttentionArena.required_nbytes(arena_caps),
                device=arena_caps.device,
            ),
        ),
    )


__all__ = [
    "B12XPagedAttentionBinding",
    "B12XPagedAttentionScratchCaps",
    "B12XPagedAttentionScratchPlan",
    "build_paged_attention_binding",
    "plan_paged_attention_scratch",
]
