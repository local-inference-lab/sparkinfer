"""Trait selection for sparse MLA kernels under the current GLM-5.1 contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_MLA_EXACT_HEAD_DIM = 576
_MLA_EXACT_V_HEAD_DIM = 512
_MLA_EXACT_PACKED_WIDTH = 656
_MLA_EXACT_NOPE_GROUPS = _MLA_EXACT_V_HEAD_DIM // 128


def _dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    if dtype == torch.uint8:
        return 1
    raise TypeError(f"unsupported dtype {dtype}")


@dataclass(frozen=True)
class SparseMLATraits:
    heads_per_cta: int
    num_threads: int
    head_dim: int
    v_head_dim: int
    num_q_heads: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    o_dtype: torch.dtype
    q_smem_bytes: int
    kv_stage_bytes: int
    q_register_elements_per_lane: int
    shared_storage_bytes: int


def select_sparse_mla_traits(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    output_dtype: torch.dtype,
    v_head_dim: int,
) -> SparseMLATraits | None:
    if q_all.device.type != "cuda" or kv_cache.device.type != "cuda":
        return None
    if page_table_1.device != q_all.device:
        return None
    if q_all.ndim != 3 or kv_cache.ndim != 3 or page_table_1.ndim != 2:
        return None
    if q_all.dtype != torch.bfloat16 or output_dtype != torch.bfloat16:
        return None
    if kv_cache.dtype not in (torch.uint8, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return None
    if q_all.shape[0] != page_table_1.shape[0]:
        return None
    if q_all.shape[1] <= 0:
        return None
    if q_all.shape[2] != _MLA_EXACT_HEAD_DIM:
        return None
    if kv_cache.shape[1:] != (1, _MLA_EXACT_PACKED_WIDTH):
        return None
    if int(v_head_dim) != _MLA_EXACT_V_HEAD_DIM:
        return None

    heads_per_cta = 1
    kv_stage_bytes = 0
    q_register_elements_per_lane = _MLA_EXACT_NOPE_GROUPS * 4 + 2
    shared_storage_bytes = 0
    return SparseMLATraits(
        heads_per_cta=heads_per_cta,
        num_threads=heads_per_cta * 32,
        head_dim=_MLA_EXACT_HEAD_DIM,
        v_head_dim=_MLA_EXACT_V_HEAD_DIM,
        num_q_heads=int(q_all.shape[1]),
        q_dtype=q_all.dtype,
        kv_dtype=kv_cache.dtype,
        o_dtype=output_dtype,
        q_smem_bytes=0,
        kv_stage_bytes=kv_stage_bytes,
        q_register_elements_per_lane=q_register_elements_per_lane,
        shared_storage_bytes=shared_storage_bytes,
    )
