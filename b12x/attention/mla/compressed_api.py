"""Compressed sparse MLA integration through the shared sparse-MLA core."""

from __future__ import annotations

import math
from typing import Literal

import torch

from .api import sparse_mla_decode_forward
from .compressed_prep import (
    CompressedMLAPrepScratch as CompressedMLACoreInputs,
    prepare_compressed_mla_core_inputs,
)
from .compressed_reference import (
    COMPRESSED_MLA_HEAD_DIM,
    COMPRESSED_MLA_LOCAL_Q_HEADS_TP2,
    COMPRESSED_MLA_SWA_PAGE_SIZE,
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
    """Run compressed sparse MLA decode through the shared sparse-MLA core."""

    if lse_scale not in ("base2", "natural"):
        raise ValueError(f"lse_scale must be 'base2' or 'natural', got {lse_scale!r}")

    q3 = _normalize_compressed_q(q_all)
    rows, heads, _ = q3.shape
    if expected_num_q_heads is not None and heads != int(expected_num_q_heads):
        raise ValueError(
            f"q_all local heads must be {int(expected_num_q_heads)} for this contract, got {heads}"
        )

    if attn_sink is not None:
        if attn_sink.shape != (heads,):
            raise ValueError(f"attn_sink must have shape [{heads}], got {tuple(attn_sink.shape)}")
        if attn_sink.device != q3.device:
            raise ValueError(f"attn_sink device {attn_sink.device} does not match q_all device {q3.device}")

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

    core = prepare_compressed_mla_core_inputs(
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

    fused_sink_output = attn_sink is not None and not return_lse
    needs_lse = return_lse or (attn_sink is not None and not fused_sink_output)
    result = sparse_mla_decode_forward(
        q_all=core.q_all,
        kv_cache=core.kv_cache,
        page_table_1=core.page_table_1,
        cache_seqlens_int32=core.cache_seqlens_int32,
        nsa_cache_seqlens_int32=core.nsa_cache_seqlens_int32,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=core.v_head_dim,
        return_lse=needs_lse,
        lse_scale="natural" if needs_lse else lse_scale,
        attn_sink=attn_sink if fused_sink_output else None,
        identity_page_table=getattr(core, "identity_page_table", False),
    )
    if not needs_lse:
        return result

    output, lse_natural = result
    assert isinstance(lse_natural, torch.Tensor)
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
    return q.contiguous()


def _normalize_index_matrix(indices: torch.Tensor, *, name: str) -> torch.Tensor:
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0]
    if indices.ndim != 2:
        raise ValueError(f"{name} must have shape [rows, width] or [rows, 1, width], got {tuple(indices.shape)}")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must have dtype torch.int32 or torch.int64, got {indices.dtype}")
    return indices


def _validate_lengths(lengths: torch.Tensor, *, rows: int, name: str) -> None:
    if lengths.shape != (rows,):
        raise ValueError(f"{name} must have shape [{rows}], got {tuple(lengths.shape)}")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must have dtype torch.int32 or torch.int64, got {lengths.dtype}")
