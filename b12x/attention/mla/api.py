"""Sparse MLA API oriented around the NSA runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

import torch

from .kernel import (
    clear_sparse_mla_kernel_cache,
    run_sparse_mla_kernel,
    supports_sparse_mla_kernel,
)
from .reference import sparse_mla_reference
from .split import (
    clear_sparse_mla_split_kernel_cache,
    forced_sparse_mla_split_decode_config_for_width,
    run_sparse_mla_split_decode,
    select_sparse_mla_split_decode_config,
)
from .workspace import B12XAttentionWorkspace


_MLA_STRATEGY_ENV = "B12X_MLA_PREFILL_STRATEGY"
_MLA_FORCE_SINGLE_PASS_ENV = "B12X_MLA_FORCE_SINGLE_PASS"
_MLA_FORCE_SPLIT_ENV = "B12X_MLA_FORCE_SPLIT"
_MLA_SINGLE_PASS_TARGET_Q_ROWS = 2048
_MLA_SINGLE_PASS_TARGET_TOPK = 2048


@dataclass(frozen=True)
class MLASparseDecodeMetadata:
    page_table_1: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor
    max_seq_len_k: int


@dataclass(frozen=True)
class MLASparseExtendMetadata:
    selected_token_offsets: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor
    nsa_cu_seqlens_q: torch.Tensor
    nsa_cu_seqlens_k: torch.Tensor
    max_seq_len_q: int
    max_seq_len_k: int
    mode: Literal["extend", "verify", "target_verify", "draft_extend"] = "extend"

    @property
    def page_table_1(self) -> torch.Tensor:
        # Compatibility alias for existing internal callers/tests during migration.
        return self.selected_token_offsets


def clear_mla_caches() -> None:
    """Clear any cached MLA runtime state."""
    clear_sparse_mla_kernel_cache()
    clear_sparse_mla_split_kernel_cache()


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _resolve_mla_prefill_strategy() -> Literal["auto", "single", "split"]:
    if _env_flag(_MLA_FORCE_SINGLE_PASS_ENV):
        return "single"
    if _env_flag(_MLA_FORCE_SPLIT_ENV):
        return "split"

    value = os.environ.get(_MLA_STRATEGY_ENV, "auto").strip().lower()
    if value in ("auto", ""):
        return "auto"
    if value in ("single", "single-pass", "nonsplit", "onepass"):
        return "single"
    if value == "split":
        return "split"
    raise ValueError(
        f"{_MLA_STRATEGY_ENV} must be auto, split, or single-pass/nonsplit; got {value!r}"
    )


def _apply_mla_prefill_strategy(
    *,
    split_cfg,
    workspace: B12XAttentionWorkspace,
    active_token_counts: torch.Tensor,
    device: torch.device,
    q_rows: int,
    topk_width: int,
):
    if split_cfg is None:
        return None

    strategy = _resolve_mla_prefill_strategy()
    if strategy == "single":
        return None
    if strategy == "split":
        return split_cfg
    if (
        workspace.mode in ("extend", "verify", "draft_extend")
        and int(workspace.max_batch) == 1
        and int(q_rows) >= _MLA_SINGLE_PASS_TARGET_Q_ROWS
        and int(topk_width) >= _MLA_SINGLE_PASS_TARGET_TOPK
    ):
        return None

    return split_cfg


def sparse_mla_decode_forward(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata: MLASparseDecodeMetadata,
    workspace: B12XAttentionWorkspace,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    workspace.prepare_decode(
        metadata.page_table_1,
        metadata.cache_seqlens_int32,
        metadata.nsa_cache_seqlens_int32,
    )
    return _run_sparse_mla(
        q_all=q_all,
        kv_cache=kv_cache,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
    )


def sparse_mla_extend_forward(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata: MLASparseExtendMetadata,
    workspace: B12XAttentionWorkspace,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    workspace.prepare_extend(
        metadata.selected_token_offsets,
        metadata.cache_seqlens_int32,
        metadata.nsa_cache_seqlens_int32,
    )
    return _run_sparse_mla(
        q_all=q_all,
        kv_cache=kv_cache,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
    )


def _run_sparse_mla(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace: B12XAttentionWorkspace,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    selected_indices = workspace.page_table_1
    active_token_counts = workspace.nsa_cache_seqlens_int32
    if selected_indices is None:
        raise RuntimeError("workspace selected-index metadata is not prepared")
    if active_token_counts is None:
        raise RuntimeError("workspace active token counts are not prepared")
    if q_all.ndim != 3:
        raise ValueError(f"q_all must be rank-3, got {tuple(q_all.shape)}")
    if kv_cache.ndim != 3:
        raise ValueError(f"kv_cache must be rank-3, got {tuple(kv_cache.shape)}")
    if q_all.device != workspace.device:
        raise ValueError(
            f"q_all device {q_all.device} does not match workspace device {workspace.device}"
        )
    if kv_cache.device != workspace.device:
        raise ValueError(
            f"kv_cache device {kv_cache.device} does not match workspace device {workspace.device}"
        )
    if selected_indices.device != workspace.device:
        raise ValueError(
            "selected indices device "
            f"{selected_indices.device} does not match workspace device {workspace.device}"
        )
    if q_all.dtype != workspace.dtype:
        raise ValueError(f"q_all dtype {q_all.dtype} does not match workspace dtype {workspace.dtype}")
    if kv_cache.dtype != workspace.kv_dtype:
        raise ValueError(
            f"kv_cache dtype {kv_cache.dtype} does not match workspace kv_dtype {workspace.kv_dtype}"
        )
    if selected_indices.dtype != torch.int32:
        raise ValueError(
            f"selected indices must have dtype torch.int32, got {selected_indices.dtype}"
        )
    if int(v_head_dim) != workspace.v_head_dim:
        raise ValueError(
            f"v_head_dim {v_head_dim} does not match workspace v_head_dim {workspace.v_head_dim}"
        )
    if q_all.shape[0] > workspace.max_total_q:
        raise ValueError(
            f"q_all rows {q_all.shape[0]} exceed workspace capacity {workspace.max_total_q}"
        )
    if q_all.shape[0] != selected_indices.shape[0]:
        raise ValueError(
            f"selected-index rows {selected_indices.shape[0]} do not match q_all rows {q_all.shape[0]}"
        )
    if selected_indices.shape[1] > workspace.topk:
        raise ValueError(
            f"selected-index width {selected_indices.shape[1]} exceeds workspace topk {workspace.topk}"
        )
    if q_all.shape[1] != workspace.num_q_heads:
        raise ValueError(
            f"q_all num_heads {q_all.shape[1]} does not match workspace num_q_heads {workspace.num_q_heads}"
        )
    if q_all.shape[-1] != workspace.head_dim:
        raise ValueError(
            f"q_all head_dim {q_all.shape[-1]} does not match workspace head_dim {workspace.head_dim}"
        )

    sm_scale_tensor = _get_sm_scale_tensor(workspace=workspace, device=q_all.device, sm_scale=sm_scale)
    split_cfg = None
    force_split = workspace.mode in ("extend", "verify", "draft_extend")
    graph_stable_split = workspace.fixed_capacity or workspace.use_cuda_graph
    split_cfg = select_sparse_mla_split_decode_config(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=selected_indices,
        active_token_counts=active_token_counts,
        output_dtype=q_all.dtype,
        v_head_dim=v_head_dim,
        max_chunks=workspace.max_chunks_per_row,
    )
    if (
        force_split
        and split_cfg is None
        and q_all.device.type == "cuda"
        and supports_sparse_mla_kernel(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=selected_indices,
            v_head_dim=v_head_dim,
        )
    ):
        forced_width = int(selected_indices.shape[1])
        if active_token_counts is not None and active_token_counts.numel() > 0:
            if (
                not graph_stable_split
                and (
                    active_token_counts.device.type != "cuda"
                    or not torch.cuda.is_current_stream_capturing()
                )
            ):
                forced_width = min(
                    forced_width,
                    max(0, int(active_token_counts.max().item())),
                )
        split_cfg = forced_sparse_mla_split_decode_config_for_width(
            forced_width,
            max_chunks=workspace.max_chunks_per_row,
        )
    if graph_stable_split and split_cfg is not None:
        split_cfg = forced_sparse_mla_split_decode_config_for_width(
            int(selected_indices.shape[1]),
            max_chunks=workspace.max_chunks_per_row,
        )
    split_cfg = _apply_mla_prefill_strategy(
        split_cfg=split_cfg,
        workspace=workspace,
        active_token_counts=active_token_counts,
        device=q_all.device,
        q_rows=int(q_all.shape[0]),
        topk_width=int(selected_indices.shape[1]),
    )
    if split_cfg is not None:
        if workspace.tmp_output is None or workspace.tmp_lse is None:
            raise RuntimeError("workspace is missing split MLA buffers")
        if (
            not _is_cuda_graph_capture_active(q_all.device)
            or not (workspace.fixed_capacity or workspace.use_cuda_graph)
        ):
            workspace.set_split_chunk_config(
                kv_chunk_size=split_cfg.chunk_size,
                num_chunks=split_cfg.num_chunks,
            )
        launch_num_chunks = (
            workspace.max_chunks_per_row if (workspace.fixed_capacity or workspace.use_cuda_graph) else split_cfg.num_chunks
        )
        output = torch.empty(
            (q_all.shape[0], q_all.shape[1], v_head_dim),
            dtype=q_all.dtype,
            device=q_all.device,
        )
        assert workspace.kv_chunk_size_ptr is not None
        assert workspace.num_chunks_ptr is not None
        run_sparse_mla_split_decode(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=selected_indices,
            active_token_counts=active_token_counts,
            sm_scale=sm_scale_tensor,
            kv_chunk_size_ptr=workspace.kv_chunk_size_ptr,
            num_chunks_ptr=workspace.num_chunks_ptr,
            tmp_output=workspace.tmp_output,
            tmp_lse=workspace.tmp_lse,
            output=output,
            launch_num_chunks=launch_num_chunks,
            workspace=workspace,
        )
    elif supports_sparse_mla_kernel(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=selected_indices,
        v_head_dim=v_head_dim,
    ):
        output = torch.empty(
            (q_all.shape[0], q_all.shape[1], v_head_dim),
            dtype=q_all.dtype,
            device=q_all.device,
        )
        run_sparse_mla_kernel(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=selected_indices,
            active_token_counts=active_token_counts,
            sm_scale=sm_scale_tensor,
            output=output,
            workspace=workspace,
        )
    else:
        if _is_cuda_graph_capture_active(q_all.device):
            raise RuntimeError(
                "b12x MLA fell back to the PyTorch reference during CUDA graph capture; "
                "the current q/kv/page-table contract is not supported by the compiled kernel path"
            )
        output = sparse_mla_reference(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=selected_indices,
            active_token_counts=active_token_counts,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
        )
    return output


def _get_sm_scale_tensor(
    *,
    workspace: B12XAttentionWorkspace,
    device: torch.device,
    sm_scale: float,
) -> torch.Tensor:
    sm_scale_tensor = workspace.sm_scale_tensor
    if (
        sm_scale_tensor is None
        or sm_scale_tensor.device != device
        or sm_scale_tensor.dtype != torch.float32
    ):
        sm_scale_tensor = torch.empty((1,), dtype=torch.float32, device=device)
        workspace.sm_scale_tensor = sm_scale_tensor
        workspace.sm_scale_value = None
    sm_scale_value = float(sm_scale)
    if workspace.sm_scale_value != sm_scale_value:
        sm_scale_tensor[0] = sm_scale_value
        workspace.sm_scale_value = sm_scale_value
    return sm_scale_tensor
