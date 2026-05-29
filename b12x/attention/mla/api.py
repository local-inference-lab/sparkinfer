"""Sparse MLA API oriented around the NSA runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
from b12x.attention.workspace import B12XAttentionWorkspace


_MLA_STRATEGY_ENV = "B12X_MLA_PREFILL_STRATEGY"
_MLA_FORCE_SINGLE_PASS_ENV = "B12X_MLA_FORCE_SINGLE_PASS"
_MLA_FORCE_SPLIT_ENV = "B12X_MLA_FORCE_SPLIT"
# Opt-in dispatch into the parallel unified_sm120 sparse-MLA backend. Parsed once at
# import time via the same _env_flag helper (api.py:86); the legacy path is byte-identical
# when this is off and no backend="sm120_unified" kwarg is supplied.
_MLA_SM120_UNIFIED_ENV = "B12X_MLA_SM120_UNIFIED"
_MLA_SM120_UNIFIED_BACKEND = "sm120_unified"
# GLM_NSA uncompressed decode contract (q_head_dim = d_nope+d_rope = 512+64).
_MLA_UNIFIED_GLM_Q_HEAD_DIM = 576
_MLA_SINGLE_PASS_TARGET_Q_ROWS = 2048
_MLA_SINGLE_PASS_TARGET_TOPK = 2048
_LN2 = math.log(2.0)


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


def clear_mla_caches() -> None:
    """Clear any cached MLA runtime state."""
    clear_sparse_mla_kernel_cache()
    clear_sparse_mla_split_kernel_cache()


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _validate_tensor_storage_bounds(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.numel() == 0:
        return
    min_offset = int(tensor.storage_offset())
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        extent = (int(size) - 1) * int(stride)
        if extent >= 0:
            max_offset += extent
        else:
            min_offset += extent
    storage_elems = tensor.untyped_storage().nbytes() // tensor.element_size()
    if min_offset < 0 or max_offset >= storage_elems:
        raise ValueError(
            f"{name} view is out of storage bounds: shape={tuple(tensor.shape)} "
            f"stride={tuple(tensor.stride())} storage_offset={int(tensor.storage_offset())} "
            f"storage_elems={storage_elems}"
        )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


# Parsed once at import; honored alongside the per-call backend kwarg (which OR-combines).
_MLA_SM120_UNIFIED_DEFAULT = _env_flag(_MLA_SM120_UNIFIED_ENV)


def _use_unified_sm120(*, backend: str | None, device: torch.device) -> bool:
    """Gate for the opt-in unified_sm120 backend.

    Returns True only when (env flag OR backend="sm120_unified") AND the device is
    SM120+. Re-reads the env flag each call so monkeypatched os.environ in tests is
    honored without re-import. The contract (q_head_dim) is checked at the call site.
    """
    if backend is not None and backend != _MLA_SM120_UNIFIED_BACKEND:
        raise ValueError(
            f"backend must be None or {_MLA_SM120_UNIFIED_BACKEND!r}, got {backend!r}"
        )
    requested = backend == _MLA_SM120_UNIFIED_BACKEND or (
        _MLA_SM120_UNIFIED_DEFAULT or _env_flag(_MLA_SM120_UNIFIED_ENV)
    )
    if not requested:
        return False
    if device.type != "cuda":
        return False
    # Imported lazily: get_sm_version lives in b12x.cute.fp4 (CuTe surface) and we only
    # need it on the opt-in path so the legacy import graph is unchanged.
    from b12x.cute.fp4 import get_sm_version

    return get_sm_version(device) >= 120


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


def _get_mla_output_view(
    *,
    workspace: B12XAttentionWorkspace,
    q_all: torch.Tensor,
    v_head_dim: int,
) -> torch.Tensor:
    rows = int(q_all.shape[0])
    heads = int(q_all.shape[1])
    v_head_dim = int(v_head_dim)
    output_buffer = workspace.output_buffer
    if output_buffer is None:
        raise RuntimeError("workspace is missing MLA output buffer")
    if output_buffer.device != q_all.device:
        raise ValueError(
            f"workspace MLA output buffer is on {output_buffer.device}, expected {q_all.device}"
        )
    if output_buffer.dtype != q_all.dtype:
        raise TypeError(
            f"workspace MLA output buffer has dtype {output_buffer.dtype}, expected {q_all.dtype}"
        )
    if output_buffer.ndim != 3:
        raise ValueError(
            f"workspace MLA output buffer must be rank 3, got {output_buffer.ndim}"
        )
    if (
        int(output_buffer.shape[0]) < rows
        or int(output_buffer.shape[1]) < heads
        or int(output_buffer.shape[2]) < v_head_dim
    ):
        raise ValueError(
            "workspace MLA output buffer is too small: "
            f"buffer={tuple(output_buffer.shape)} required=({rows}, {heads}, {v_head_dim})"
        )
    _validate_tensor_storage_bounds(output_buffer, name="workspace MLA output buffer")
    return output_buffer[:rows, :heads, :v_head_dim]


def _validate_split_control_tensors(
    *,
    workspace: object,
) -> None:
    for name, tensor in (
        ("kv_chunk_size_ptr", workspace.kv_chunk_size_ptr),
        ("num_chunks_ptr", workspace.num_chunks_ptr),
    ):
        if tensor is None:
            raise RuntimeError(f"workspace is missing {name}")
        if tensor.shape != (1,):
            raise ValueError(f"{name} must have shape (1,), got {tuple(tensor.shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if tensor.device != workspace.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match workspace device {workspace.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _validate_split_workspace_views(
    *,
    workspace: B12XAttentionWorkspace,
    q_rows: int,
    num_heads: int,
    v_head_dim: int,
    launch_num_chunks: int,
) -> None:
    _validate_split_control_tensors(workspace=workspace)
    if workspace.tmp_output is None or workspace.tmp_lse is None:
        raise RuntimeError("workspace is missing split MLA buffers")

    q_rows = int(q_rows)
    num_heads = int(num_heads)
    v_head_dim = int(v_head_dim)
    launch_num_chunks = int(launch_num_chunks)
    tmp_output = workspace.tmp_output
    tmp_lse = workspace.tmp_lse

    if tmp_output.device != workspace.device or tmp_lse.device != workspace.device:
        raise ValueError("split MLA scratch buffers must be on the workspace device")
    if tmp_output.dtype != workspace.dtype:
        raise TypeError(
            f"split MLA tmp_output dtype {tmp_output.dtype} does not match workspace dtype {workspace.dtype}"
        )
    if tmp_lse.dtype != torch.float32:
        raise TypeError(f"split MLA tmp_lse must have dtype torch.float32, got {tmp_lse.dtype}")
    if tmp_output.ndim != 4:
        raise ValueError(f"split MLA tmp_output must be rank-4, got {tuple(tmp_output.shape)}")
    if tmp_lse.ndim != 3:
        raise ValueError(f"split MLA tmp_lse must be rank-3, got {tuple(tmp_lse.shape)}")
    _validate_tensor_storage_bounds(tmp_output, name="split MLA tmp_output")
    _validate_tensor_storage_bounds(tmp_lse, name="split MLA tmp_lse")
    required_output = (q_rows, num_heads, launch_num_chunks, v_head_dim)
    if (
        int(tmp_output.shape[0]) < q_rows
        or int(tmp_output.shape[1]) < num_heads
        or int(tmp_output.shape[2]) < launch_num_chunks
        or int(tmp_output.shape[3]) < v_head_dim
    ):
        raise ValueError(
            "split MLA tmp_output is too small: "
            f"buffer={tuple(tmp_output.shape)} required>={required_output}"
        )
    required_lse = (q_rows, num_heads, launch_num_chunks)
    if (
        int(tmp_lse.shape[0]) < q_rows
        or int(tmp_lse.shape[1]) < num_heads
        or int(tmp_lse.shape[2]) < launch_num_chunks
    ):
        raise ValueError(
            "split MLA tmp_lse is too small: "
            f"buffer={tuple(tmp_lse.shape)} required>={required_lse}"
        )


def _resolve_sparse_mla_binding(
    *,
    binding,
    q_all: torch.Tensor | None,
    selected_indices: torch.Tensor | None,
    cache_seqlens_int32: torch.Tensor | None,
    nsa_cache_seqlens_int32: torch.Tensor | None,
    workspace: B12XAttentionWorkspace | None,
    selected_name: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    B12XAttentionWorkspace,
]:
    if binding is None:
        missing = [
            name
            for name, value in (
                ("q_all", q_all),
                (selected_name, selected_indices),
                ("cache_seqlens_int32", cache_seqlens_int32),
                ("nsa_cache_seqlens_int32", nsa_cache_seqlens_int32),
                ("workspace", workspace),
            )
            if value is None
        ]
        if missing:
            raise TypeError(f"missing required sparse MLA arguments: {', '.join(missing)}")
        return (
            q_all,
            selected_indices,
            cache_seqlens_int32,
            nsa_cache_seqlens_int32,
            workspace,
        )

    extras = [
        name
        for name, value in (
            ("q_all", q_all),
            (selected_name, selected_indices),
            ("cache_seqlens_int32", cache_seqlens_int32),
            ("nsa_cache_seqlens_int32", nsa_cache_seqlens_int32),
            ("workspace", workspace),
        )
        if value is not None
    ]
    if extras:
        raise ValueError(
            "sparse MLA binding owns runtime tensors and scratch; "
            f"do not also pass {', '.join(extras)}"
        )
    return (
        binding.q,
        binding.selected_indices,
        binding.cache_seqlens_int32,
        binding.nsa_cache_seqlens_int32,
        binding.scratch,
    )


def sparse_mla_decode_forward(
    *,
    q_all: torch.Tensor | None = None,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor | None = None,
    cache_seqlens_int32: torch.Tensor | None = None,
    nsa_cache_seqlens_int32: torch.Tensor | None = None,
    workspace: B12XAttentionWorkspace | None = None,
    binding=None,
    sm_scale: float,
    v_head_dim: int | None = None,
    return_lse: bool = False,
    lse_scale: Literal["base2", "natural"] = "base2",
    attn_sink: torch.Tensor | None = None,
    identity_page_table: bool = False,
    backend: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    q_all, page_table_1, cache_seqlens_int32, nsa_cache_seqlens_int32, workspace = (
        _resolve_sparse_mla_binding(
            binding=binding,
            q_all=q_all,
            selected_indices=page_table_1,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            workspace=workspace,
            selected_name="page_table_1",
        )
    )
    if v_head_dim is None:
        v_head_dim = workspace.v_head_dim
    return _run_sparse_mla(
        q_all=q_all,
        kv_cache=kv_cache,
        selected_indices=page_table_1,
        cache_seqlens_int32=cache_seqlens_int32,
        active_token_counts=nsa_cache_seqlens_int32,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
        return_lse=return_lse,
        lse_scale=lse_scale,
        attn_sink=attn_sink,
        identity_page_table=identity_page_table,
        backend=backend,
    )


def sparse_mla_extend_forward(
    *,
    q_all: torch.Tensor | None = None,
    kv_cache: torch.Tensor,
    selected_token_offsets: torch.Tensor | None = None,
    cache_seqlens_int32: torch.Tensor | None = None,
    nsa_cache_seqlens_int32: torch.Tensor | None = None,
    workspace: B12XAttentionWorkspace | None = None,
    binding=None,
    sm_scale: float,
    v_head_dim: int | None = None,
    return_lse: bool = False,
    lse_scale: Literal["base2", "natural"] = "base2",
    identity_page_table: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    q_all, selected_token_offsets, cache_seqlens_int32, nsa_cache_seqlens_int32, workspace = (
        _resolve_sparse_mla_binding(
            binding=binding,
            q_all=q_all,
            selected_indices=selected_token_offsets,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            workspace=workspace,
            selected_name="selected_token_offsets",
        )
    )
    if v_head_dim is None:
        v_head_dim = workspace.v_head_dim
    return _run_sparse_mla(
        q_all=q_all,
        kv_cache=kv_cache,
        selected_indices=selected_token_offsets,
        cache_seqlens_int32=cache_seqlens_int32,
        active_token_counts=nsa_cache_seqlens_int32,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
        return_lse=return_lse,
        lse_scale=lse_scale,
        identity_page_table=identity_page_table,
    )


def _run_sparse_mla(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    active_token_counts: torch.Tensor,
    workspace: B12XAttentionWorkspace,
    sm_scale: float,
    v_head_dim: int,
    return_lse: bool = False,
    lse_scale: Literal["base2", "natural"] = "base2",
    attn_sink: torch.Tensor | None = None,
    identity_page_table: bool = False,
    backend: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if q_all.ndim != 3:
        raise ValueError(f"q_all must be rank-3, got {tuple(q_all.shape)}")
    if kv_cache.ndim != 3:
        raise ValueError(f"kv_cache must be rank-3, got {tuple(kv_cache.shape)}")
    if selected_indices.ndim != 2:
        raise ValueError(
            f"selected indices must be rank-2, got {tuple(selected_indices.shape)}"
        )
    if cache_seqlens_int32.ndim != 1:
        raise ValueError(
            "cache_seqlens_int32 must be rank-1, "
            f"got {tuple(cache_seqlens_int32.shape)}"
        )
    if active_token_counts.ndim != 1:
        raise ValueError(
            "nsa_cache_seqlens_int32 must be rank-1, "
            f"got {tuple(active_token_counts.shape)}"
        )
    if not q_all.is_contiguous():
        raise ValueError("q_all must be contiguous for sparse MLA")
    if not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous for sparse MLA")
    if not selected_indices.is_contiguous():
        raise ValueError("selected indices must be contiguous for sparse MLA")
    if not cache_seqlens_int32.is_contiguous():
        raise ValueError("cache_seqlens_int32 must be contiguous for sparse MLA")
    if not active_token_counts.is_contiguous():
        raise ValueError("nsa_cache_seqlens_int32 must be contiguous for sparse MLA")
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
    if cache_seqlens_int32.device != workspace.device:
        raise ValueError(
            "cache_seqlens_int32 device "
            f"{cache_seqlens_int32.device} does not match workspace device {workspace.device}"
        )
    if active_token_counts.device != workspace.device:
        raise ValueError(
            "nsa_cache_seqlens_int32 device "
            f"{active_token_counts.device} does not match workspace device {workspace.device}"
        )
    if q_all.dtype != workspace.dtype:
        raise ValueError(
            f"q_all dtype {q_all.dtype} does not match workspace dtype {workspace.dtype}"
        )
    if kv_cache.dtype != workspace.kv_dtype:
        raise ValueError(
            f"kv_cache dtype {kv_cache.dtype} does not match workspace kv_dtype {workspace.kv_dtype}"
        )
    if selected_indices.dtype != torch.int32:
        raise ValueError(
            f"selected indices must have dtype torch.int32, got {selected_indices.dtype}"
        )
    if cache_seqlens_int32.dtype != torch.int32:
        raise ValueError(
            "cache_seqlens_int32 must have dtype torch.int32, "
            f"got {cache_seqlens_int32.dtype}"
        )
    if active_token_counts.dtype != torch.int32:
        raise ValueError(
            "nsa_cache_seqlens_int32 must have dtype torch.int32, "
            f"got {active_token_counts.dtype}"
        )
    if int(v_head_dim) != workspace.v_head_dim:
        raise ValueError(
            f"v_head_dim {v_head_dim} does not match workspace v_head_dim {workspace.v_head_dim}"
        )
    # The unified SM120 backend supports attn_sink together with return_lse (the
    # sink folds into O in the merge and into the returned LSE). The legacy fused
    # path is output-only, so the return_lse+attn_sink restriction applies ONLY
    # when the unified path will NOT be taken.
    _sm120_unified_route = _use_unified_sm120(backend=backend, device=q_all.device)
    if attn_sink is not None:
        attn_sink = attn_sink.detach()
        if return_lse and not _sm120_unified_route:
            raise ValueError("fused sparse MLA attn_sink currently supports output-only calls")
        if attn_sink.ndim != 1 or int(attn_sink.shape[0]) != int(q_all.shape[1]):
            raise ValueError(
                f"attn_sink must have shape ({int(q_all.shape[1])},), got {tuple(attn_sink.shape)}"
            )
        if attn_sink.device != workspace.device:
            raise ValueError(
                f"attn_sink device {attn_sink.device} does not match workspace device {workspace.device}"
            )
        if attn_sink.dtype != torch.float32:
            raise ValueError(f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}")
        if not attn_sink.is_contiguous():
            raise ValueError("attn_sink must be contiguous for fused sparse MLA")
    if q_all.shape[0] > workspace.max_total_q:
        raise ValueError(
            f"q_all rows {q_all.shape[0]} exceed workspace capacity {workspace.max_total_q}"
        )
    if q_all.shape[0] != selected_indices.shape[0]:
        raise ValueError(
            f"selected-index rows {selected_indices.shape[0]} do not match q_all rows {q_all.shape[0]}"
        )
    if selected_indices.shape[0] != active_token_counts.shape[0]:
        raise ValueError(
            "selected-index rows "
            f"{selected_indices.shape[0]} do not match nsa_cache_seqlens_int32 rows "
            f"{active_token_counts.shape[0]}"
        )
    if cache_seqlens_int32.shape[0] > workspace.max_batch:
        raise ValueError(
            "cache_seqlens_int32 batch "
            f"{cache_seqlens_int32.shape[0]} exceeds workspace capacity {workspace.max_batch}"
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
    # Opt-in dispatch into the parallel unified_sm120 backend (DSV3.2 dropped; this is the
    # GLM_NSA uncompressed q_head_dim==576 contract). Gated so the legacy path below is
    # byte-identical when the flag is off and no backend kwarg is supplied.
    #
    # P7 SCOPE: the unified backend implements the DSV4 MAIN-CACHE compressed
    # decode ONLY (routed via compressed_api.compressed_mla_decode_forward). The
    # GLM_NSA uncompressed path here is DEFERRED to P7b -- so when the gate is on
    # and the GLM contract is presented we raise NotImplementedError (never
    # silently route an unsupported GLM call to the DSV4 kernel). q != 576 is a
    # contract error.
    if _use_unified_sm120(backend=backend, device=q_all.device):
        if int(q_all.shape[-1]) != _MLA_UNIFIED_GLM_Q_HEAD_DIM:
            raise ValueError(
                f"unified_sm120 sparse decode requires the GLM_NSA contract "
                f"(q_head_dim={_MLA_UNIFIED_GLM_Q_HEAD_DIM}); got q_head_dim={int(q_all.shape[-1])}"
            )
        # GLM_NSA (uncompressed q=576, ARBITRARY_FP32 inline scales) decode via the
        # unified SM120 kernel (cute.constexpr GLM branches beside DSV4). The
        # unified path now covers the upstream DSV3.2/GLM decode surface:
        # return_lse, attn_sink fold (sink-in-merge), and VALID_HPB<16 /
        # non-multiple-of-16 head shards. Multi-token rows (q_rows>1) remain a
        # SEPARATE later step (per-token topk_length threading), so single-token is
        # still required here; a >1-row GLM decode falls through to legacy until
        # that lands. Everything else routes to unified unconditionally.
        _glm_unified_ok = int(q_all.shape[0]) == 1
        if _glm_unified_ok:
            from .unified_sm120.launch import run_unified_decode

            return run_unified_decode(
                q_all=q_all,
                swa_k_cache=kv_cache,
                swa_indices=selected_indices,
                swa_topk_lengths=active_token_counts,
                workspace=workspace,
                sm_scale=float(sm_scale),
                swa_page_size=int(workspace.page_size),
                attn_sink=attn_sink,
                return_lse=return_lse,
                lse_scale=lse_scale,
            )
    sm_scale_tensor = _get_sm_scale_tensor(
        workspace=workspace, device=q_all.device, sm_scale=sm_scale
    )
    split_cfg = None
    force_split = return_lse or attn_sink is not None or workspace.mode in ("extend", "verify", "draft_extend")
    graph_stable_split = workspace.fixed_capacity or workspace.use_cuda_graph
    split_cfg = select_sparse_mla_split_decode_config(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=selected_indices,
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
        split_cfg = forced_sparse_mla_split_decode_config_for_width(
            forced_width,
            max_chunks=workspace.max_chunks_per_row,
        )
    if graph_stable_split and split_cfg is not None:
        split_cfg = forced_sparse_mla_split_decode_config_for_width(
            int(selected_indices.shape[1]),
            max_chunks=workspace.max_chunks_per_row,
        )
    if not return_lse:
        split_cfg = _apply_mla_prefill_strategy(
            split_cfg=split_cfg,
            workspace=workspace,
            active_token_counts=active_token_counts,
            device=q_all.device,
            q_rows=int(q_all.shape[0]),
            topk_width=int(selected_indices.shape[1]),
        )
    if split_cfg is not None:
        if not _is_cuda_graph_capture_active(q_all.device) or not (
            workspace.fixed_capacity or workspace.use_cuda_graph
        ):
            workspace.set_split_chunk_config(
                kv_chunk_size=split_cfg.chunk_size,
                num_chunks=split_cfg.num_chunks,
            )
        launch_num_chunks = (
            workspace.max_chunks_per_row
            if (workspace.fixed_capacity or workspace.use_cuda_graph)
            else split_cfg.num_chunks
        )
        _validate_split_workspace_views(
            workspace=workspace,
            q_rows=int(q_all.shape[0]),
            num_heads=int(q_all.shape[1]),
            v_head_dim=int(v_head_dim),
            launch_num_chunks=int(launch_num_chunks),
        )
        output = _get_mla_output_view(
            workspace=workspace,
            q_all=q_all,
            v_head_dim=v_head_dim,
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
            attn_sink=attn_sink,
            workspace=workspace,
            identity_page_table=identity_page_table,
        )
        if return_lse:
            lse = _final_lse_from_split_workspace(
                workspace=workspace,
                q_rows=int(q_all.shape[0]),
                num_heads=int(q_all.shape[1]),
                launch_num_chunks=int(launch_num_chunks),
                scale=lse_scale,
            )
    elif supports_sparse_mla_kernel(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=selected_indices,
        v_head_dim=v_head_dim,
    ):
        if return_lse:
            raise RuntimeError(
                "B12X sparse MLA LSE output requires the split path, but no split "
                "configuration was available for this contract."
            )
        if attn_sink is not None:
            raise RuntimeError(
                "B12X sparse MLA attn_sink output requires the split path, but no split "
                "configuration was available for this contract."
            )
        output = _get_mla_output_view(
            workspace=workspace,
            q_all=q_all,
            v_head_dim=v_head_dim,
        )
        run_sparse_mla_kernel(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=selected_indices,
            active_token_counts=active_token_counts,
            sm_scale=sm_scale_tensor,
            output=output,
            workspace=workspace,
            identity_page_table=identity_page_table,
        )
    else:
        if _is_cuda_graph_capture_active(q_all.device):
            raise RuntimeError(
                "b12x MLA fell back to the PyTorch reference during CUDA graph capture; "
                "the current q/kv/page-table contract is not supported by the compiled kernel path"
            )
        if identity_page_table:
            raise RuntimeError("identity page-table sparse MLA requires the compiled CUDA kernel path")
        reference_kwargs = dict(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=selected_indices,
            active_token_counts=active_token_counts,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
        )
        if return_lse:
            reference_kwargs["return_lse"] = True
        output = sparse_mla_reference(**reference_kwargs)
        if return_lse:
            output, lse = output
            if lse_scale == "natural":
                lse = lse * _LN2
    if return_lse:
        return output, lse
    return output


def _final_lse_from_split_workspace(
    *,
    workspace: object,
    q_rows: int,
    num_heads: int,
    launch_num_chunks: int,
    scale: Literal["base2", "natural"] = "base2",
) -> torch.Tensor:
    if workspace.tmp_lse is None:
        raise RuntimeError("workspace is missing split MLA LSE buffer")
    if workspace.final_lse is None:
        raise RuntimeError("workspace is missing final MLA LSE buffer")
    q_rows = int(q_rows)
    num_heads = int(num_heads)
    chunk_count = int(launch_num_chunks)
    if chunk_count <= 0:
        raise ValueError(f"launch_num_chunks must be positive, got {chunk_count}")
    if workspace.tmp_lse.ndim != 3:
        raise ValueError(f"workspace split MLA LSE buffer must be rank-3, got {tuple(workspace.tmp_lse.shape)}")
    _validate_tensor_storage_bounds(workspace.tmp_lse, name="workspace split MLA LSE buffer")
    if (
        int(workspace.tmp_lse.shape[0]) < q_rows
        or int(workspace.tmp_lse.shape[1]) < num_heads
        or int(workspace.tmp_lse.shape[2]) < chunk_count
    ):
        raise ValueError(
            "workspace split MLA LSE buffer is too small: "
            f"buffer={tuple(workspace.tmp_lse.shape)} required>=({q_rows}, {num_heads}, {chunk_count})"
        )
    if workspace.final_lse.ndim != 2:
        raise ValueError(f"workspace final MLA LSE buffer must be rank-2, got {tuple(workspace.final_lse.shape)}")
    _validate_tensor_storage_bounds(workspace.final_lse, name="workspace final MLA LSE buffer")
    if int(workspace.final_lse.shape[0]) < q_rows or int(workspace.final_lse.shape[1]) < num_heads:
        raise ValueError(
            "workspace final MLA LSE buffer is too small: "
            f"buffer={tuple(workspace.final_lse.shape)} required>=({q_rows}, {num_heads})"
        )
    final_lse = workspace.final_lse[:q_rows, :num_heads]
    if final_lse.dtype != torch.float32:
        raise TypeError(
            f"workspace final MLA LSE buffer must be FP32, got {final_lse.dtype}"
        )
    chunk_lse = workspace.tmp_lse[:q_rows, :num_heads, :chunk_count]
    if chunk_lse.dtype != torch.float32:
        raise TypeError(
            f"workspace split MLA LSE buffer must be FP32, got {chunk_lse.dtype}"
        )
    chunk_lse.mul_(_LN2)
    torch.logsumexp(chunk_lse, dim=-1, out=final_lse)
    if scale == "natural":
        return final_lse
    final_lse.div_(_LN2)
    return final_lse


def _get_sm_scale_tensor(
    *,
    workspace: object,
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
        sm_scale_tensor.fill_(sm_scale_value)
        workspace.sm_scale_value = sm_scale_value
    return sm_scale_tensor
