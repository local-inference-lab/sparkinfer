"""Compressed sparse MLA integration through the shared sparse-MLA core."""

from __future__ import annotations

import math
from typing import Literal

import torch

from .api import _final_lse_from_split_workspace, _get_sm_scale_tensor
from .compressed_prep import (
    prepare_compressed_mla_native_metadata,
)
from .compressed_reference import (
    COMPRESSED_MLA_HEAD_DIM,
    COMPRESSED_MLA_LOCAL_Q_HEADS_TP2,
    COMPRESSED_MLA_SWA_PAGE_SIZE,
    compressed_mla_page_nbytes,
)
from .split import (
    forced_sparse_mla_split_decode_config_for_width,
    run_compressed_mla_split_decode_forward,
    run_sparse_mla_split_decode_merge,
)
from .workspace import B12XAttentionWorkspace


_LN2 = math.log(2.0)


def compressed_mla_decode_forward(
    *,
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    workspace: B12XAttentionWorkspace,
    sm_scale: float,
    swa_page_size: int = COMPRESSED_MLA_SWA_PAGE_SIZE,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_topk_lengths: torch.Tensor | None = None,
    indexed_page_size: int | None = None,
    indexed_page_table: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    expected_num_q_heads: int | None = COMPRESSED_MLA_LOCAL_Q_HEADS_TP2,
    return_lse: bool = False,
    lse_scale: Literal["base2", "natural"] = "base2",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run compressed sparse MLA decode directly from compressed KV pages."""

    if lse_scale not in ("base2", "natural"):
        raise ValueError(f"lse_scale must be 'base2' or 'natural', got {lse_scale!r}")

    q3 = _normalize_compressed_q(q_all)
    rows, heads, _ = q3.shape
    if expected_num_q_heads is not None and heads != int(expected_num_q_heads):
        raise ValueError(
            f"q_all local heads must be {int(expected_num_q_heads)} for this contract, got {heads}"
        )

    if attn_sink is not None:
        attn_sink = attn_sink.detach()
        if attn_sink.shape != (heads,):
            raise ValueError(f"attn_sink must have shape [{heads}], got {tuple(attn_sink.shape)}")
        if attn_sink.device != q3.device:
            raise ValueError(f"attn_sink device {attn_sink.device} does not match q_all device {q3.device}")
        if attn_sink.dtype != torch.float32:
            raise TypeError(f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}")
        if not attn_sink.is_contiguous():
            raise ValueError("attn_sink must be contiguous")

    swa_indices_2d = _normalize_index_matrix(swa_indices, name="swa_indices")
    if swa_indices_2d.shape[0] != rows:
        raise ValueError("swa_indices row count must match q_all")
    _validate_lengths(swa_topk_lengths, rows=rows, name="swa_topk_lengths")

    has_indexed = indexed_k_cache is not None or indexed_indices is not None or indexed_topk_lengths is not None
    if has_indexed:
        if indexed_k_cache is None or indexed_indices is None or indexed_topk_lengths is None:
            raise ValueError("indexed_k_cache, indexed_indices, and indexed_topk_lengths must be provided together")
        if indexed_page_size is None:
            raise ValueError("indexed_page_size is required when indexed_k_cache is provided")
        indexed_indices_2d = _normalize_index_matrix(indexed_indices, name="indexed_indices")
        if indexed_indices_2d.shape[0] != rows:
            raise ValueError("indexed_indices row count must match q_all")
        _validate_lengths(indexed_topk_lengths, rows=rows, name="indexed_topk_lengths")
        if indexed_page_table is not None:
            indexed_page_table_2d = _normalize_index_matrix(indexed_page_table, name="indexed_page_table")
            if indexed_page_table_2d.shape[0] != rows:
                raise ValueError("indexed_page_table row count must match q_all")
        else:
            indexed_page_table_2d = None
    else:
        indexed_indices_2d = None
        indexed_page_table_2d = None
        if indexed_page_table is not None:
            raise ValueError("indexed_page_table requires indexed_k_cache/indices/lengths")

    _validate_native_workspace(
        workspace=workspace,
        rows=rows,
        heads=heads,
        width=swa_indices_2d.shape[1] + (indexed_indices_2d.shape[1] if has_indexed else 0),
    )

    metadata = prepare_compressed_mla_native_metadata(
        q_all=q3,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices_2d,
        swa_topk_lengths=swa_topk_lengths,
        workspace=workspace,
        swa_page_size=swa_page_size,
        indexed_k_cache=indexed_k_cache,
        indexed_indices=indexed_indices_2d,
        indexed_topk_lengths=indexed_topk_lengths,
        indexed_page_size=indexed_page_size,
        indexed_page_table=indexed_page_table_2d,
    )

    if not has_indexed:
        indexed_k_cache_for_kernel = swa_k_cache
        indexed_indices_for_kernel = swa_indices_2d
        indexed_lengths_for_kernel = swa_topk_lengths
        indexed_page_size_for_kernel = int(swa_page_size)
        indexed_page_table_for_kernel = swa_indices_2d
        map_indexed_page_table = False
        indexed_page_table_width = 1
    else:
        assert indexed_k_cache is not None
        assert indexed_indices_2d is not None
        assert indexed_topk_lengths is not None
        assert indexed_page_size is not None
        indexed_k_cache_for_kernel = indexed_k_cache
        indexed_indices_for_kernel = indexed_indices_2d
        indexed_lengths_for_kernel = indexed_topk_lengths
        indexed_page_size_for_kernel = int(indexed_page_size)
        map_indexed_page_table = indexed_page_table_2d is not None
        indexed_page_table_for_kernel = (
            indexed_page_table_2d if map_indexed_page_table else indexed_indices_2d
        )
        indexed_page_table_width = int(indexed_page_table_for_kernel.shape[1]) if map_indexed_page_table else 1

    total_width = int(swa_indices_2d.shape[1]) + (
        int(indexed_indices_2d.shape[1]) if has_indexed else 0
    )
    min_effective_chunk = 32 if rows <= 1 else 64
    compressed_max_chunks = min(
        workspace.max_chunks_per_row,
        max(1, (total_width + min_effective_chunk - 1) // min_effective_chunk),
    )
    split_cfg = forced_sparse_mla_split_decode_config_for_width(
        total_width,
        max_chunks=compressed_max_chunks,
    )
    if split_cfg is None:
        raise RuntimeError(
            f"compressed MLA width {total_width} has no supported split configuration"
        )
    if workspace.tmp_output is None or workspace.tmp_lse is None:
        raise RuntimeError("workspace is missing split MLA buffers")
    if workspace.kv_chunk_size_ptr is None or workspace.num_chunks_ptr is None:
        raise RuntimeError("workspace is missing split MLA chunk metadata")

    graph_stable_split = workspace.fixed_capacity or workspace.use_cuda_graph
    graph_capture_active = q3.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    if not graph_capture_active or not graph_stable_split:
        workspace.set_split_chunk_config(
            kv_chunk_size=split_cfg.chunk_size,
            num_chunks=split_cfg.num_chunks,
        )
    launch_num_chunks = workspace.max_chunks_per_row if graph_stable_split else split_cfg.num_chunks

    output = torch.empty(
        (rows, heads, COMPRESSED_MLA_HEAD_DIM),
        dtype=q3.dtype,
        device=q3.device,
    )
    sm_scale_tensor = _get_sm_scale_tensor(
        workspace=workspace,
        device=q3.device,
        sm_scale=sm_scale,
    )
    run_compressed_mla_split_decode_forward(
        q_all=q3,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices_2d,
        swa_lengths=metadata.swa_valid_lengths,
        indexed_k_cache=indexed_k_cache_for_kernel,
        indexed_indices=indexed_indices_for_kernel,
        indexed_lengths=(
            metadata.indexed_valid_lengths if has_indexed else indexed_lengths_for_kernel
        ),
        indexed_page_table=indexed_page_table_for_kernel,
        sm_scale=sm_scale_tensor,
        kv_chunk_size_ptr=workspace.kv_chunk_size_ptr,
        num_chunks_ptr=workspace.num_chunks_ptr,
        tmp_output=workspace.tmp_output,
        tmp_lse=workspace.tmp_lse,
        launch_num_chunks=launch_num_chunks,
        swa_page_size=int(swa_page_size),
        swa_page_nbytes=compressed_mla_page_nbytes(int(swa_page_size)),
        indexed_page_size=indexed_page_size_for_kernel,
        indexed_page_nbytes=compressed_mla_page_nbytes(indexed_page_size_for_kernel),
        has_indexed=has_indexed,
        map_indexed_page_table=map_indexed_page_table,
        indexed_page_table_width=indexed_page_table_width,
        workspace=workspace,
    )

    fused_sink_output = attn_sink is not None and not return_lse
    needs_lse = return_lse or (attn_sink is not None and not fused_sink_output)
    run_sparse_mla_split_decode_merge(
        tmp_output=workspace.tmp_output,
        tmp_lse=workspace.tmp_lse,
        num_chunks_ptr=workspace.num_chunks_ptr,
        output=output,
        attn_sink=attn_sink if fused_sink_output else None,
        workspace=workspace,
    )
    if not needs_lse:
        return output

    lse_natural = _final_lse_from_split_workspace(
        workspace=workspace,
        q_rows=rows,
        num_heads=heads,
        launch_num_chunks=int(launch_num_chunks),
        scale="natural",
    )
    if attn_sink is not None:
        sink = attn_sink.float().view(1, heads)
        lse_with_sink = torch.logaddexp(lse_natural.float(), sink)
        scale = torch.exp(lse_natural.float() - lse_with_sink).view(rows, heads, 1)
        output = (output.float() * scale).to(output.dtype)
        lse_natural = lse_with_sink

    if not return_lse:
        return output
    if lse_scale == "base2":
        return output, lse_natural / _LN2
    return output, lse_natural


def _normalize_compressed_q(q: torch.Tensor) -> torch.Tensor:
    if q.ndim == 4 and q.shape[1] == 1:
        q = q[:, 0]
    if q.ndim != 3 or q.shape[-1] != COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(f"q_all must have shape [rows, heads, {COMPRESSED_MLA_HEAD_DIM}], got {tuple(q.shape)}")
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q_all must have dtype torch.bfloat16, got {q.dtype}")
    if not q.is_contiguous():
        raise ValueError("q_all must be contiguous for compressed MLA")
    return q.detach()


def _normalize_index_matrix(indices: torch.Tensor, *, name: str) -> torch.Tensor:
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0]
    if indices.ndim != 2:
        raise ValueError(f"{name} must have shape [rows, width] or [rows, 1, width], got {tuple(indices.shape)}")
    if indices.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {indices.dtype}")
    if not indices.is_contiguous():
        raise ValueError(f"{name} must be contiguous for compressed MLA")
    return indices


def _validate_lengths(lengths: torch.Tensor, *, rows: int, name: str) -> None:
    if lengths.shape != (rows,):
        raise ValueError(f"{name} must have shape [{rows}], got {tuple(lengths.shape)}")
    if lengths.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {lengths.dtype}")
    if not lengths.is_contiguous():
        raise ValueError(f"{name} must be contiguous for compressed MLA")


def _validate_native_workspace(
    workspace: B12XAttentionWorkspace,
    *,
    rows: int,
    heads: int,
    width: int,
) -> None:
    if rows > workspace.max_total_q:
        raise ValueError(f"q rows {rows} exceed workspace max_total_q {workspace.max_total_q}")
    if rows > workspace.max_batch and workspace.mode == "decode":
        raise ValueError(f"decode rows {rows} exceed workspace max_batch {workspace.max_batch}")
    if heads != workspace.num_q_heads:
        raise ValueError(f"q_all num_heads {heads} does not match workspace num_q_heads {workspace.num_q_heads}")
    if workspace.head_dim != COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(
            f"compressed MLA workspace head_dim must be {COMPRESSED_MLA_HEAD_DIM}, got {workspace.head_dim}"
        )
    if workspace.v_head_dim != COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(
            f"compressed MLA workspace v_head_dim must be {COMPRESSED_MLA_HEAD_DIM}, got {workspace.v_head_dim}"
        )
    if width > workspace.topk:
        raise ValueError(f"compressed MLA width {width} exceeds workspace topk {workspace.topk}")
