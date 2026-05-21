"""Compressed sparse MLA integration through the shared sparse-MLA core."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

from .api import sparse_mla_decode_forward
from .compressed_prep import prepare_compressed_mla_core_inputs_cuda
from .compressed_reference import (
    COMPRESSED_MLA_HEAD_DIM,
    COMPRESSED_MLA_LOCAL_Q_HEADS_TP2,
    COMPRESSED_MLA_NOPE_DIM,
    COMPRESSED_MLA_SWA_PAGE_SIZE,
    gather_compressed_mla_kv_cache_reference,
)
from .reference import _MLA_NOPE_DIM, _MLA_ROPE_DIM, pack_mla_kv_cache_reference
from .workspace import B12XAttentionWorkspace


_LN2 = math.log(2.0)


@dataclass(frozen=True)
class CompressedMLACoreInputs:
    """Inputs after adapting compressed sparse MLA to the shared MLA core contract."""

    q_all: torch.Tensor
    kv_cache: torch.Tensor
    page_table_1: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor

    @property
    def v_head_dim(self) -> int:
        return _MLA_NOPE_DIM


def prepare_compressed_mla_core_inputs(
    *,
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    swa_page_size: int = COMPRESSED_MLA_SWA_PAGE_SIZE,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_topk_lengths: torch.Tensor | None = None,
    indexed_page_size: int | None = None,
    expected_num_q_heads: int | None = COMPRESSED_MLA_LOCAL_Q_HEADS_TP2,
) -> CompressedMLACoreInputs:
    """Prepare compressed sparse MLA data for the existing 576-d MLA core.

    The current core scores a 512-d noPE block plus a 64-d RoPE block and returns
    512 value lanes.  Compressed MLA uses 448 noPE lanes plus 64 RoPE lanes.  The
    adapter pads Q's noPE block with zeros and stores RoPE values in those padded
    value lanes, so the shared core can produce a 512-d output without changing
    the GLM hot path.
    """

    q3 = _normalize_compressed_q(q_all)
    rows, heads, _ = q3.shape
    if expected_num_q_heads is not None and heads != int(expected_num_q_heads):
        raise ValueError(
            f"q_all local heads must be {int(expected_num_q_heads)} for this contract, got {heads}"
        )
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
    else:
        indexed_indices_2d = None

    swa_width = int(swa_indices_2d.shape[1])
    indexed_width = int(indexed_indices_2d.shape[1]) if indexed_indices_2d is not None else 0
    total_width = max(1, swa_width + indexed_width)
    device = q3.device

    q_core = torch.zeros((rows, heads, _MLA_NOPE_DIM + _MLA_ROPE_DIM), dtype=q3.dtype, device=device)
    q_core[:, :, :COMPRESSED_MLA_NOPE_DIM] = q3[:, :, :COMPRESSED_MLA_NOPE_DIM]
    q_core[:, :, _MLA_NOPE_DIM:] = q3[:, :, COMPRESSED_MLA_NOPE_DIM:]

    core_k_nope = torch.zeros((rows * total_width, _MLA_NOPE_DIM), dtype=torch.float32, device=device)
    core_k_rope = torch.zeros((rows * total_width, _MLA_ROPE_DIM), dtype=torch.bfloat16, device=device)
    page_table_1 = torch.arange(rows * total_width, dtype=torch.int32, device=device).view(rows, total_width)
    active_counts = torch.zeros((rows,), dtype=torch.int32, device=device)

    for row in range(rows):
        write_start = row * total_width
        write_pos = 0
        swa_len = _valid_prefix_length(swa_topk_lengths[row], swa_indices_2d[row], swa_width)
        if swa_len:
            k_swa, _ = gather_compressed_mla_kv_cache_reference(
                swa_k_cache,
                swa_indices_2d[row, :swa_len],
                page_size=swa_page_size,
            )
            _write_core_rows(
                core_k_nope=core_k_nope,
                core_k_rope=core_k_rope,
                write_start=write_start + write_pos,
                k_full=k_swa,
            )
            write_pos += swa_len

        if has_indexed and indexed_indices_2d is not None and indexed_topk_lengths is not None:
            indexed_len = _valid_prefix_length(indexed_topk_lengths[row], indexed_indices_2d[row], indexed_width)
            if indexed_len:
                assert indexed_k_cache is not None
                assert indexed_page_size is not None
                k_indexed, _ = gather_compressed_mla_kv_cache_reference(
                    indexed_k_cache,
                    indexed_indices_2d[row, :indexed_len],
                    page_size=indexed_page_size,
                )
                _write_core_rows(
                    core_k_nope=core_k_nope,
                    core_k_rope=core_k_rope,
                    write_start=write_start + write_pos,
                    k_full=k_indexed,
                )
                write_pos += indexed_len

        active_counts[row] = write_pos

    kv_cache = pack_mla_kv_cache_reference(core_k_nope, core_k_rope)
    return CompressedMLACoreInputs(
        q_all=q_core.contiguous(),
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=active_counts.clone(),
        nsa_cache_seqlens_int32=active_counts,
    )


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
    else:
        indexed_indices_2d = None

    if q3.device.type == "cuda":
        core = prepare_compressed_mla_core_inputs_cuda(
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
        )
    else:
        core = prepare_compressed_mla_core_inputs(
            q_all=q3,
            swa_k_cache=swa_k_cache,
            swa_indices=swa_indices_2d,
            swa_topk_lengths=swa_topk_lengths,
            swa_page_size=swa_page_size,
            indexed_k_cache=indexed_k_cache,
            indexed_indices=indexed_indices_2d,
            indexed_topk_lengths=indexed_topk_lengths,
            indexed_page_size=indexed_page_size,
            expected_num_q_heads=None,
        )

    needs_lse = return_lse or attn_sink is not None
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


def _write_core_rows(
    *,
    core_k_nope: torch.Tensor,
    core_k_rope: torch.Tensor,
    write_start: int,
    k_full: torch.Tensor,
) -> None:
    count = int(k_full.shape[0])
    if count == 0:
        return
    dst = slice(write_start, write_start + count)
    core_k_nope[dst, :COMPRESSED_MLA_NOPE_DIM] = k_full[:, :COMPRESSED_MLA_NOPE_DIM]
    core_k_nope[dst, COMPRESSED_MLA_NOPE_DIM:_MLA_NOPE_DIM] = k_full[:, COMPRESSED_MLA_NOPE_DIM:]
    core_k_rope[dst] = k_full[:, COMPRESSED_MLA_NOPE_DIM:].to(torch.bfloat16)


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


def _bounded_length(length: torch.Tensor, width: int) -> int:
    value = int(length.item())
    if value < 0:
        raise ValueError(f"topk length must be non-negative, got {value}")
    return min(value, width)


def _valid_prefix_length(length: torch.Tensor, indices: torch.Tensor, width: int) -> int:
    limit = _bounded_length(length, width)
    if limit == 0:
        return 0
    active = indices[:limit].to(torch.int64)
    negative = torch.nonzero(active < 0, as_tuple=False)
    if negative.numel() == 0:
        return limit
    return int(negative[0, 0].item())
