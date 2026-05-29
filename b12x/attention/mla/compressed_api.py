"""Compressed sparse MLA integration through the shared sparse-MLA core."""

from __future__ import annotations

import math
from typing import Literal

import torch

from .api import (
    _final_lse_from_split_workspace,
    _get_mla_output_view,
    _get_sm_scale_tensor,
    _use_unified_sm120,
    _validate_split_control_tensors,
    _validate_tensor_storage_bounds,
)
from .compressed_config import (
    compressed_mla_split_chunks_for_contract,
    compressed_mla_split_config_for_contract,
)
from .compressed_reference import (
    COMPRESSED_MLA_DSV4_PAGE_SIZE,
    COMPRESSED_MLA_HEAD_DIM,
    compressed_mla_page_nbytes,
)
from .split import (
    _compressed_mla_cache_byte_view,
    build_compressed_mla_split_decode_forward_binding,
    build_sparse_mla_split_decode_merge_binding,
    run_compressed_mla_split_decode_forward,
    run_sparse_mla_split_decode_merge,
)
from b12x.attention.workspace import B12XAttentionWorkspace


_LN2 = math.log(2.0)
def compressed_mla_decode_forward(
    *,
    q_all: torch.Tensor | None = None,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor | None = None,
    swa_topk_lengths: torch.Tensor | None = None,
    workspace: B12XAttentionWorkspace | None = None,
    binding=None,
    sm_scale: float,
    swa_page_size: int = COMPRESSED_MLA_DSV4_PAGE_SIZE,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_topk_lengths: torch.Tensor | None = None,
    indexed_page_size: int | None = None,
    indexed_page_table: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    expected_num_q_heads: int | None = None,
    return_lse: bool = False,
    lse_scale: Literal["base2", "natural"] = "base2",
    backend: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run compressed sparse MLA decode directly from compressed KV pages."""

    if lse_scale not in ("base2", "natural"):
        raise ValueError(f"lse_scale must be 'base2' or 'natural', got {lse_scale!r}")

    scratch = workspace
    if binding is not None:
        binding_scratch = getattr(binding, "scratch", None)
        if binding_scratch is None:
            raise TypeError("compressed MLA binding is missing scratch")
        if workspace is not None and workspace is not binding_scratch:
            raise ValueError("workspace argument does not match compressed MLA binding scratch")
        scratch = binding_scratch
        if q_all is None:
            q_all = getattr(binding, "q")
        if swa_indices is None:
            swa_indices = getattr(binding, "swa_indices")
        if swa_topk_lengths is None:
            swa_topk_lengths = getattr(binding, "swa_lengths")
        if indexed_indices is None:
            indexed_indices = getattr(binding, "indexed_indices", None)
        if indexed_topk_lengths is None:
            indexed_topk_lengths = getattr(binding, "indexed_lengths", None)
        if indexed_page_table is None:
            indexed_page_table = getattr(binding, "indexed_page_table", None)

    if q_all is None:
        raise TypeError("compressed_mla_decode_forward requires q_all or binding")
    if swa_indices is None:
        raise TypeError("compressed_mla_decode_forward requires swa_indices or binding")
    if swa_topk_lengths is None:
        raise TypeError("compressed_mla_decode_forward requires swa_topk_lengths or binding")
    if scratch is None:
        raise TypeError("compressed_mla_decode_forward requires workspace or binding")

    q3 = _normalize_compressed_q(q_all)
    rows, heads, _ = q3.shape
    live_rows = rows

    # Dispatch into the unified_sm120 backend (DSV4 decode). The unified kernel is
    # the SM120 sparse-MLA path and supports the FULL upstream DSV4 decode surface:
    #   * q_head_dim == COMPRESSED_MLA_HEAD_DIM == 512 (DSV4; NOT GLM q=576)
    #   * MAIN cache, OR the DSV4 DUAL-CACHE (indexed/extra tokens) when the full
    #     extra trio is provided together with indexed_page_size
    #   * attn_sink fold (sink-in-merge, upstream design) and return_lse
    #   * VALID_HPB<16 / non-multiple-of-16 head shards
    # GENUINELY-UPSTREAM-UNSUPPORTED contracts RAISE (never fall back to legacy):
    #   * a mapped indexed_page_table (upstream is raw-slot-id only)
    #   * a partial dual-cache trio (upstream ICHECKs required-together)
    # A non-512 q_head_dim here is a compressed-API contract error -> RAISE.
    # When the flag is OFF / backend is None _use_unified_sm120 is False and the
    # legacy path below stays byte-identical (that is the flag being off, not a
    # fallback; the default-flip is a separate validated step).
    if _use_unified_sm120(backend=backend, device=q3.device):
        _unified_has_extra = (
            indexed_k_cache is not None
            or indexed_indices is not None
            or indexed_topk_lengths is not None
        )
        if _unified_has_extra:
            # Partial dual-cache trio is a HARD ERROR (upstream required-together
            # ICHECK, sparse_mla_sm120.cu:171-174). NOT a fallback.
            if not (
                indexed_k_cache is not None
                and indexed_indices is not None
                and indexed_topk_lengths is not None
                and indexed_page_size is not None
            ):
                raise ValueError(
                    "SM120 sparse-MLA dual-cache decode requires indexed_k_cache, "
                    "indexed_indices, indexed_topk_lengths, and indexed_page_size "
                    "together (partial extra trio is unsupported, matching upstream "
                    "sparse_mla_sm120.cu:171-174)"
                )
            # Mapped extra page table is GENUINELY-UPSTREAM-UNSUPPORTED (upstream
            # addresses the extra cache by raw slot id only). RAISE, not fallback.
            if indexed_page_table is not None:
                raise ValueError(
                    "SM120 sparse-MLA decode does not support a mapped "
                    "indexed_page_table (extra cache is raw-slot-id only on the "
                    "unified backend, matching upstream FlashInfer)"
                )
        if int(q3.shape[-1]) != COMPRESSED_MLA_HEAD_DIM:
            raise ValueError(
                f"compressed_mla_decode_forward is DSV4 (q_head_dim="
                f"{COMPRESSED_MLA_HEAD_DIM}) by construction; got q_head_dim="
                f"{int(q3.shape[-1])}"
            )
        from . import unified_sm120

        return unified_sm120.run_unified_decode(
            q_all=q3,
            swa_k_cache=swa_k_cache,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_topk_lengths,
            workspace=scratch,
            sm_scale=sm_scale,
            swa_page_size=swa_page_size,
            indexed_k_cache=indexed_k_cache,
            indexed_indices=indexed_indices,
            indexed_topk_lengths=indexed_topk_lengths,
            indexed_page_size=indexed_page_size,
            indexed_page_table=indexed_page_table,
            attn_sink=attn_sink,
            return_lse=return_lse,
            lse_scale=lse_scale,
        )

    swa_k_cache = _compressed_mla_cache_byte_view(swa_k_cache, name="swa_k_cache")
    _validate_compressed_cache_layout(
        swa_k_cache,
        page_size=swa_page_size,
        name="swa_k_cache",
    )
    if expected_num_q_heads is not None and heads != int(expected_num_q_heads):
        raise ValueError(
            f"q_all local heads must match expected_num_q_heads={int(expected_num_q_heads)}, got {heads}"
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
    if swa_indices_2d.device != q3.device:
        raise ValueError("swa_indices must be on the same device as q_all")
    if swa_indices_2d.shape[0] != rows:
        raise ValueError("swa_indices row count must match q_all")
    _validate_lengths(
        swa_topk_lengths,
        rows=rows,
        name="swa_topk_lengths",
    )
    if swa_topk_lengths.device != q3.device:
        raise ValueError("swa_topk_lengths must be on the same device as q_all")

    has_indexed = indexed_k_cache is not None or indexed_indices is not None or indexed_topk_lengths is not None
    if has_indexed:
        if indexed_k_cache is None or indexed_indices is None or indexed_topk_lengths is None:
            raise ValueError("indexed_k_cache, indexed_indices, and indexed_topk_lengths must be provided together")
        if indexed_page_size is None:
            raise ValueError("indexed_page_size is required when indexed_k_cache is provided")
        indexed_k_cache = _compressed_mla_cache_byte_view(indexed_k_cache, name="indexed_k_cache")
        _validate_compressed_cache_layout(
            indexed_k_cache,
            page_size=int(indexed_page_size),
            name="indexed_k_cache",
        )
        indexed_indices_2d = _normalize_index_matrix(indexed_indices, name="indexed_indices")
        if indexed_indices_2d.device != q3.device:
            raise ValueError("indexed_indices must be on the same device as q_all")
        if indexed_indices_2d.shape[0] != rows:
            raise ValueError("indexed_indices row count must match q_all")
        _validate_lengths(
            indexed_topk_lengths,
            rows=rows,
            name="indexed_topk_lengths",
        )
        if indexed_topk_lengths.device != q3.device:
            raise ValueError("indexed_topk_lengths must be on the same device as q_all")
        if indexed_page_table is not None:
            indexed_page_table_2d = _normalize_index_matrix(
                indexed_page_table,
                name="indexed_page_table",
                allow_row_shared=True,
            )
            if indexed_page_table_2d.device != q3.device:
                raise ValueError("indexed_page_table must be on the same device as q_all")
            if indexed_page_table_2d.shape[0] != rows:
                raise ValueError("indexed_page_table row count must match q_all")
        else:
            indexed_page_table_2d = None
    else:
        indexed_indices_2d = None
        indexed_page_table_2d = None
        if indexed_page_table is not None:
            raise ValueError("indexed_page_table requires indexed_k_cache/indices/lengths")

    _validate_compressed_mla_scratch(
        scratch=scratch,
        rows=rows,
        heads=heads,
        width=swa_indices_2d.shape[1] + (indexed_indices_2d.shape[1] if has_indexed else 0),
    )

    if not has_indexed:
        indexed_k_cache_for_kernel = swa_k_cache
        indexed_indices_for_kernel = swa_indices_2d
        indexed_lengths_for_kernel = swa_topk_lengths
        indexed_page_size_for_kernel = int(swa_page_size)
        indexed_page_table_for_kernel = swa_indices_2d
        map_indexed_page_table = False
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

    total_width = int(swa_indices_2d.shape[1]) + (
        int(indexed_indices_2d.shape[1]) if has_indexed else 0
    )
    if scratch.tmp_output is None or scratch.tmp_lse is None:
        raise RuntimeError("compressed MLA scratch is missing split buffers")
    _validate_split_control_tensors(workspace=scratch)

    graph_stable_split = scratch.fixed_capacity or scratch.use_cuda_graph
    if graph_stable_split:
        split_cfg = compressed_mla_split_config_for_contract(
            rows=scratch.max_total_q,
            width=scratch.topk,
            max_chunks=scratch.max_chunks_per_row,
        )
    else:
        split_cfg = compressed_mla_split_config_for_contract(
            rows=rows,
            width=total_width,
            max_chunks=scratch.max_chunks_per_row,
        )
    graph_capture_active = q3.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    if not graph_capture_active or not graph_stable_split:
        scratch.set_split_chunk_config(
            kv_chunk_size=split_cfg.chunk_size,
            num_chunks=split_cfg.num_chunks,
        )
    elif scratch.kv_chunk_size_value is None or scratch.num_chunks_value is None:
        raise RuntimeError("compressed MLA fixed scratch split config was not preplanned before graph capture")
    elif int(scratch.kv_chunk_size_value) != int(split_cfg.chunk_size) or int(
        scratch.num_chunks_value
    ) != int(split_cfg.num_chunks):
        raise RuntimeError(
            "compressed MLA fixed scratch split config was not preplanned before graph capture: "
            f"scratch has chunk_size={scratch.kv_chunk_size_value} "
            f"num_chunks={scratch.num_chunks_value}, expected "
            f"chunk_size={split_cfg.chunk_size} num_chunks={split_cfg.num_chunks}"
        )
    launch_num_chunks = (
        scratch.max_chunks_per_row
        if graph_stable_split
        else split_cfg.num_chunks
    )

    output = _get_mla_output_view(
        workspace=scratch,
        q_all=q3,
        v_head_dim=COMPRESSED_MLA_HEAD_DIM,
    )
    q_kernel = q3
    swa_indices_kernel = swa_indices_2d
    swa_lengths_kernel = swa_topk_lengths
    indexed_indices_kernel = indexed_indices_for_kernel
    indexed_lengths_kernel = indexed_lengths_for_kernel
    indexed_page_table_kernel = indexed_page_table_for_kernel
    output_kernel = output
    if graph_stable_split:
        if scratch.output_buffer is None:
            raise RuntimeError("fixed compressed MLA scratch is missing output buffer")
        if (
            scratch.output_buffer.ndim != 3
            or int(scratch.output_buffer.shape[0]) < int(scratch.max_total_q)
            or int(scratch.output_buffer.shape[1]) < heads
            or int(scratch.output_buffer.shape[2]) < COMPRESSED_MLA_HEAD_DIM
        ):
            raise ValueError(
                "fixed compressed MLA scratch output buffer is too small: "
                f"buffer={tuple(scratch.output_buffer.shape)} required>="
                f"({int(scratch.max_total_q)}, {heads}, {COMPRESSED_MLA_HEAD_DIM})"
            )
        if binding is None:
            (
                q_kernel,
                swa_indices_kernel,
                swa_lengths_kernel,
                indexed_indices_kernel,
                indexed_lengths_kernel,
                indexed_page_table_kernel,
            ) = _stage_fixed_compressed_mla_inputs(
                workspace=workspace,
                q_all=q3,
                swa_indices=swa_indices_2d,
                swa_lengths=swa_topk_lengths,
                indexed_indices=indexed_indices_for_kernel,
                indexed_lengths=indexed_lengths_for_kernel,
                indexed_page_table=indexed_page_table_for_kernel,
            )
            output_kernel = scratch.output_buffer[
                : scratch.max_total_q,
                :heads,
                :COMPRESSED_MLA_HEAD_DIM,
            ]
        elif int(q_kernel.shape[0]) == int(scratch.max_total_q):
            output_kernel = scratch.output_buffer[
                : scratch.max_total_q,
                :heads,
                :COMPRESSED_MLA_HEAD_DIM,
            ]
    fused_sink_output = attn_sink is not None and not return_lse
    needs_lse = return_lse or (attn_sink is not None and not fused_sink_output)
    direct_single_chunk_output = (
        split_cfg.num_chunks == 1
        and attn_sink is None
        and not needs_lse
    )
    direct_sink_output = False
    single_tile_chunks = split_cfg.chunk_size <= 64
    sm_scale_tensor = _get_sm_scale_tensor(
        workspace=scratch,
        device=q3.device,
        sm_scale=sm_scale,
    )
    launch_chunks_for_kernel = 1 if direct_single_chunk_output else int(launch_num_chunks)
    _validate_compressed_launch_views(
        tmp_output=output_kernel if direct_single_chunk_output else scratch.tmp_output,
        tmp_lse=scratch.tmp_lse,
        q_rows=int(q_kernel.shape[0]),
        heads=heads,
        launch_num_chunks=launch_chunks_for_kernel,
        direct_output=direct_single_chunk_output,
    )
    forward_binding = build_compressed_mla_split_decode_forward_binding(
        q_all=q_kernel,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices_kernel,
        swa_lengths=swa_lengths_kernel,
        indexed_k_cache=indexed_k_cache_for_kernel,
        indexed_indices=indexed_indices_kernel,
        indexed_lengths=indexed_lengths_kernel,
        indexed_page_table=indexed_page_table_kernel,
        sm_scale=sm_scale_tensor,
        kv_chunk_size_ptr=scratch.kv_chunk_size_ptr,
        num_chunks_ptr=scratch.num_chunks_ptr,
        tmp_output=output_kernel if direct_single_chunk_output else scratch.tmp_output,
        tmp_lse=scratch.tmp_lse,
        launch_num_chunks=launch_chunks_for_kernel,
        swa_page_size=int(swa_page_size),
        swa_page_nbytes=compressed_mla_page_nbytes(int(swa_page_size)),
        indexed_page_size=indexed_page_size_for_kernel,
        indexed_page_nbytes=compressed_mla_page_nbytes(indexed_page_size_for_kernel),
        has_indexed=has_indexed,
        map_indexed_page_table=map_indexed_page_table,
        workspace=scratch,
        direct_output=direct_single_chunk_output,
        single_tile_chunks=single_tile_chunks,
        attn_sink=attn_sink,
        direct_sink_output=direct_sink_output,
    )
    run_compressed_mla_split_decode_forward(binding=forward_binding)

    if direct_single_chunk_output:
        pass
    elif split_cfg.num_chunks == 1 and attn_sink is None:
        output_kernel.copy_(
            scratch.tmp_output[
                : int(q_kernel.shape[0]),
                :heads,
                0,
                :COMPRESSED_MLA_HEAD_DIM,
            ]
        )
    else:
        merge_binding = build_sparse_mla_split_decode_merge_binding(
            tmp_output=scratch.tmp_output,
            tmp_lse=scratch.tmp_lse,
            num_chunks_ptr=scratch.num_chunks_ptr,
            output=output_kernel,
            attn_sink=attn_sink if fused_sink_output else None,
            workspace=scratch,
        )
        run_sparse_mla_split_decode_merge(binding=merge_binding)
    if not needs_lse:
        return output

    lse_natural = _final_lse_from_split_workspace(
        workspace=scratch,
        q_rows=live_rows,
        num_heads=heads,
        launch_num_chunks=int(launch_num_chunks),
        scale="natural",
    )
    if attn_sink is not None:
        sink = attn_sink.float().view(1, heads)
        lse_with_sink = torch.logaddexp(lse_natural.float(), sink)
        scale = torch.exp(lse_natural.float() - lse_with_sink).view(live_rows, heads, 1)
        output = (output.float() * scale).to(output.dtype)
        lse_natural = lse_with_sink

    if not return_lse:
        return output
    if lse_scale == "base2":
        lse_natural.div_(_LN2)
        return output, lse_natural
    return output, lse_natural


def _stage_fixed_compressed_mla_inputs(
    *,
    workspace: B12XAttentionWorkspace,
    q_all: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_indices: torch.Tensor,
    indexed_lengths: torch.Tensor,
    indexed_page_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_stage = workspace.compressed_mla_q_stage
    swa_indices_stage = workspace.compressed_mla_swa_indices_stage
    swa_lengths_stage = workspace.compressed_mla_swa_lengths_stage
    indexed_indices_stage = workspace.compressed_mla_indexed_indices_stage
    indexed_lengths_stage = workspace.compressed_mla_indexed_lengths_stage
    indexed_page_table_stage = workspace.compressed_mla_indexed_page_table_stage
    if (
        q_stage is None
        or swa_indices_stage is None
        or swa_lengths_stage is None
        or indexed_indices_stage is None
        or indexed_lengths_stage is None
        or indexed_page_table_stage is None
    ):
        raise RuntimeError(
            "fixed compressed MLA workspace is missing capacity staging buffers; "
            "set reserve_compressed_mla_staging=True when building the attention arena"
        )

    rows = int(q_all.shape[0])
    cap_rows = int(workspace.max_total_q)
    if rows > cap_rows:
        raise ValueError(f"q rows {rows} exceed fixed compressed MLA staging capacity {cap_rows}")
    if q_stage.shape != (cap_rows, int(workspace.num_q_heads), COMPRESSED_MLA_HEAD_DIM):
        raise ValueError(
            "compressed MLA q staging buffer shape mismatch: "
            f"got {tuple(q_stage.shape)}, expected "
            f"({cap_rows}, {int(workspace.num_q_heads)}, {COMPRESSED_MLA_HEAD_DIM})"
        )
    for name, stage in (
        ("swa_lengths", swa_lengths_stage),
        ("indexed_lengths", indexed_lengths_stage),
    ):
        if stage.shape != (cap_rows,):
            raise ValueError(
                f"compressed MLA {name} staging buffer shape mismatch: "
                f"got {tuple(stage.shape)}, expected ({cap_rows},)"
            )
        if stage.dtype != torch.int32:
            raise TypeError(f"compressed MLA {name} staging buffer must be int32, got {stage.dtype}")
        if stage.device != q_all.device:
            raise ValueError(f"compressed MLA {name} staging buffer must be on {q_all.device}")
    q_stage[:rows].copy_(q_all.detach())

    swa_indices_view = _stage_fixed_int_matrix(
        swa_indices_stage,
        swa_indices,
        rows=rows,
        cap_rows=cap_rows,
        name="swa_indices",
    )
    indexed_indices_view = _stage_fixed_int_matrix(
        indexed_indices_stage,
        indexed_indices,
        rows=rows,
        cap_rows=cap_rows,
        name="indexed_indices",
    )
    indexed_page_table_view = _stage_fixed_int_matrix(
        indexed_page_table_stage,
        indexed_page_table,
        rows=rows,
        cap_rows=cap_rows,
        name="indexed_page_table",
    )
    swa_lengths_view = swa_lengths_stage[:cap_rows]
    indexed_lengths_view = indexed_lengths_stage[:cap_rows]
    swa_lengths_view[:rows].copy_(swa_lengths.detach())
    indexed_lengths_view[:rows].copy_(indexed_lengths.detach())
    if rows < cap_rows:
        swa_lengths_view[rows:].zero_()
        indexed_lengths_view[rows:].zero_()

    return (
        q_stage,
        swa_indices_view,
        swa_lengths_view,
        indexed_indices_view,
        indexed_lengths_view,
        indexed_page_table_view,
    )


def _stage_fixed_int_matrix(
    stage: torch.Tensor,
    source: torch.Tensor,
    *,
    rows: int,
    cap_rows: int,
    name: str,
) -> torch.Tensor:
    width = int(source.shape[1])
    if int(stage.shape[0]) < cap_rows or int(stage.shape[1]) < width:
        raise ValueError(
            f"{name} staging buffer is too small: stage={tuple(stage.shape)} "
            f"required=({cap_rows}, {width})"
        )
    view = stage.reshape(-1)[: cap_rows * width].view(cap_rows, width)
    if rows:
        view[:rows].copy_(source.detach())
    return view


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


def _validate_compressed_cache_layout(
    cache: torch.Tensor,
    *,
    page_size: int,
    name: str,
) -> None:
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError(f"{name} page_size must be positive, got {page_size}")
    expected_page_nbytes = compressed_mla_page_nbytes(page_size)
    if int(cache.shape[1]) != expected_page_nbytes:
        raise ValueError(
            f"{name} page byte width must be {expected_page_nbytes} for page_size "
            f"{page_size}, got {int(cache.shape[1])}"
        )


def _validate_compressed_launch_views(
    *,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    q_rows: int,
    heads: int,
    launch_num_chunks: int,
    direct_output: bool,
) -> None:
    if tmp_output is None or tmp_lse is None:
        raise RuntimeError("compressed MLA launch is missing scratch/output buffers")
    q_rows = int(q_rows)
    heads = int(heads)
    launch_num_chunks = int(launch_num_chunks)
    if tmp_output.dtype != torch.bfloat16:
        raise TypeError(f"compressed MLA tmp_output must be BF16, got {tmp_output.dtype}")
    if tmp_lse.dtype != torch.float32:
        raise TypeError(f"compressed MLA tmp_lse must be FP32, got {tmp_lse.dtype}")
    if tmp_output.device != tmp_lse.device:
        raise ValueError("compressed MLA tmp_output and tmp_lse must be on the same device")
    _validate_tensor_storage_bounds(tmp_output, name="compressed MLA tmp_output")
    _validate_tensor_storage_bounds(tmp_lse, name="compressed MLA tmp_lse")
    if direct_output:
        if tmp_output.ndim != 3:
            raise ValueError(
                f"compressed MLA direct output must be rank-3, got {tuple(tmp_output.shape)}"
            )
        required = (q_rows, heads, COMPRESSED_MLA_HEAD_DIM)
        if (
            int(tmp_output.shape[0]) < q_rows
            or int(tmp_output.shape[1]) < heads
            or int(tmp_output.shape[2]) < COMPRESSED_MLA_HEAD_DIM
        ):
            raise ValueError(
                "compressed MLA direct output is too small: "
                f"buffer={tuple(tmp_output.shape)} required>={required}"
            )
    else:
        if tmp_output.ndim != 4:
            raise ValueError(
                f"compressed MLA split output must be rank-4, got {tuple(tmp_output.shape)}"
            )
        required = (q_rows, heads, launch_num_chunks, COMPRESSED_MLA_HEAD_DIM)
        if (
            int(tmp_output.shape[0]) < q_rows
            or int(tmp_output.shape[1]) < heads
            or int(tmp_output.shape[2]) < launch_num_chunks
            or int(tmp_output.shape[3]) < COMPRESSED_MLA_HEAD_DIM
        ):
            raise ValueError(
                "compressed MLA split output is too small: "
                f"buffer={tuple(tmp_output.shape)} required>={required}"
            )
    if tmp_lse.ndim != 3:
        raise ValueError(f"compressed MLA tmp_lse must be rank-3, got {tuple(tmp_lse.shape)}")
    required_lse = (q_rows, heads, max(1, launch_num_chunks))
    if (
        int(tmp_lse.shape[0]) < q_rows
        or int(tmp_lse.shape[1]) < heads
        or int(tmp_lse.shape[2]) < max(1, launch_num_chunks)
    ):
        raise ValueError(
            "compressed MLA tmp_lse is too small: "
            f"buffer={tuple(tmp_lse.shape)} required>={required_lse}"
        )


def _is_row_shared_index_matrix(indices: torch.Tensor) -> bool:
    return indices.ndim == 2 and int(indices.stride(0)) == 0 and int(indices.stride(1)) == 1


def _normalize_index_matrix(
    indices: torch.Tensor,
    *,
    name: str,
    allow_row_shared: bool = False,
) -> torch.Tensor:
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0]
    if indices.ndim != 2:
        raise ValueError(f"{name} must have shape [rows, width] or [rows, 1, width], got {tuple(indices.shape)}")
    if indices.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {indices.dtype}")
    if not indices.is_contiguous() and not (allow_row_shared and _is_row_shared_index_matrix(indices)):
        raise ValueError(f"{name} must be contiguous for compressed MLA")
    return indices


def _validate_lengths(
    lengths: torch.Tensor,
    *,
    rows: int,
    name: str,
) -> None:
    if lengths.shape != (rows,):
        raise ValueError(f"{name} must have shape [{rows}], got {tuple(lengths.shape)}")
    if lengths.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {lengths.dtype}")
    if not lengths.is_contiguous():
        raise ValueError(f"{name} must be contiguous for compressed MLA")


def _validate_compressed_mla_scratch(
    scratch: object,
    *,
    rows: int,
    heads: int,
    width: int,
) -> None:
    if rows > scratch.max_total_q:
        raise ValueError(f"q rows {rows} exceed compressed MLA scratch max_total_q {scratch.max_total_q}")
    if rows > scratch.max_batch and scratch.mode == "decode":
        raise ValueError(f"decode rows {rows} exceed compressed MLA scratch max_batch {scratch.max_batch}")
    if heads != scratch.num_q_heads:
        raise ValueError(
            f"q_all num_heads {heads} does not match compressed MLA scratch num_q_heads {scratch.num_q_heads}"
        )
    if scratch.head_dim != COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(
            f"compressed MLA scratch head_dim must be {COMPRESSED_MLA_HEAD_DIM}, got {scratch.head_dim}"
        )
    if scratch.v_head_dim != COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(
            f"compressed MLA scratch v_head_dim must be {COMPRESSED_MLA_HEAD_DIM}, got {scratch.v_head_dim}"
        )
    if width > scratch.topk:
        raise ValueError(f"compressed MLA width {width} exceeds scratch topk {scratch.topk}")
