"""FlashInfer-inspired forward trait selection for the primary paged backend."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .planner import PagedPlan

_FP8_KV_DTYPE = torch.float8_e4m3fn


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    if dtype == _FP8_KV_DTYPE:
        return 1
    raise TypeError(f"unsupported dtype {dtype}")


def paged_get_num_warps_q(cta_tile_q: int) -> int:
    return 4 if cta_tile_q > 16 else 1


def paged_get_num_warps_kv(cta_tile_q: int) -> int:
    return 4 // paged_get_num_warps_q(cta_tile_q)


def paged_get_num_mma_q(cta_tile_q: int) -> int:
    return 2 if cta_tile_q > 64 else 1


@dataclass(frozen=True)
class PagedForwardTraits:
    cta_tile_q: int
    cta_tile_kv: int
    num_mma_q: int
    num_mma_kv: int
    num_mma_d_qk: int
    num_mma_d_vo: int
    num_warps_q: int
    num_warps_kv: int
    num_threads: int
    head_dim_qk: int
    head_dim_vo: int
    upcast_stride_q: int
    upcast_stride_k: int
    upcast_stride_v: int
    upcast_stride_o: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    o_dtype: torch.dtype
    q_smem_bytes: int
    shared_storage_bytes: int
    max_smem_per_sm: int
    num_ctas_per_sm: int
    max_smem_per_threadblock: int

    @property
    def uses_fp8_kv(self) -> bool:
        return self.kv_dtype == _FP8_KV_DTYPE


def _paged_is_invalid(
    *,
    num_mma_q: int,
    num_mma_kv: int,
    num_mma_d_vo: int,
    num_warps_q: int,
    kv_dtype: torch.dtype,
) -> bool:
    kv_is_fp8 = kv_dtype == _FP8_KV_DTYPE
    if num_mma_d_vo < 4:
        return True
    if num_mma_d_vo == 4 and num_mma_kv % 2 == 1:
        return True
    if num_mma_q * (8 * num_mma_d_vo + 8 * num_mma_kv) >= 256:
        return True
    if kv_is_fp8 and (num_mma_kv * 2) % num_warps_q != 0:
        return True
    return False


def select_paged_forward_traits(
    *,
    cta_tile_q: int,
    head_dim_qk: int,
    head_dim_vo: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    o_dtype: torch.dtype | None = None,
    device: torch.device | int | None = None,
) -> PagedForwardTraits:
    if head_dim_qk % 16 != 0 or head_dim_vo % 16 != 0:
        raise ValueError("head_dim_qk and head_dim_vo must be multiples of 16")
    if q_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported q dtype {q_dtype}")
    if kv_dtype not in (torch.float16, torch.bfloat16, _FP8_KV_DTYPE):
        raise TypeError(f"unsupported kv dtype {kv_dtype}")
    if kv_dtype == _FP8_KV_DTYPE and q_dtype != torch.bfloat16:
        raise TypeError("primary paged backend only supports bf16 queries with fp8 kv")
    o_dtype = q_dtype if o_dtype is None else o_dtype
    if o_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported output dtype {o_dtype}")

    if kv_dtype == _FP8_KV_DTYPE and cta_tile_q == 48:
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device)
        max_smem_per_sm = int(device_props.shared_memory_per_multiprocessor)
        kv_bytes = _dtype_num_bytes(kv_dtype)
        upcast_stride_k = _align_up(head_dim_qk // (16 // kv_bytes), 8)
        upcast_stride_v = _align_up(head_dim_vo // (16 // kv_bytes), 8)
        return PagedForwardTraits(
            cta_tile_q=48,
            cta_tile_kv=32,
            num_mma_q=1,
            num_mma_kv=2,
            num_mma_d_qk=head_dim_qk // 16,
            num_mma_d_vo=head_dim_vo // 16,
            num_warps_q=3,
            num_warps_kv=1,
            num_threads=96,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            upcast_stride_q=head_dim_qk // 8,
            upcast_stride_k=upcast_stride_k,
            upcast_stride_v=upcast_stride_v,
            upcast_stride_o=head_dim_vo // (16 // _dtype_num_bytes(o_dtype)),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=o_dtype,
            q_smem_bytes=48 * head_dim_qk * _dtype_num_bytes(q_dtype),
            shared_storage_bytes=49152,
            max_smem_per_sm=max_smem_per_sm,
            num_ctas_per_sm=2 if max_smem_per_sm >= 2 * 49152 else 1,
            max_smem_per_threadblock=max_smem_per_sm // (2 if max_smem_per_sm >= 2 * 49152 else 1),
        )

    num_mma_d_qk = head_dim_qk // 16
    num_mma_d_vo = head_dim_vo // 16
    num_warps_q = paged_get_num_warps_q(cta_tile_q)
    num_warps_kv = paged_get_num_warps_kv(cta_tile_q)
    num_mma_q = paged_get_num_mma_q(cta_tile_q)

    device_props = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device)
    max_smem_per_sm = int(device_props.shared_memory_per_multiprocessor)

    q_bytes = _dtype_num_bytes(q_dtype)
    kv_bytes = _dtype_num_bytes(kv_dtype)
    o_bytes = _dtype_num_bytes(o_dtype)
    upcast_stride_q = head_dim_qk // (16 // q_bytes)
    upcast_stride_k = head_dim_qk // (16 // kv_bytes)
    upcast_stride_v = head_dim_vo // (16 // kv_bytes)
    if kv_dtype == _FP8_KV_DTYPE:
        upcast_stride_k = _align_up(upcast_stride_k, 8)
        upcast_stride_v = _align_up(upcast_stride_v, 8)
    upcast_stride_o = head_dim_vo // (16 // o_bytes)
    q_smem_bytes = cta_tile_q * head_dim_qk * q_bytes
    kv_bytes_per_mma = (upcast_stride_k + upcast_stride_v) * 16 * 16 * num_warps_kv
    num_ctas_per_sm = 2 if max_smem_per_sm >= 2 * (q_smem_bytes + kv_bytes_per_mma) else 1
    max_smem_per_threadblock = max_smem_per_sm // num_ctas_per_sm
    max_num_mma_kv_reg = 8 // num_mma_q
    max_num_mma_kv_smem = max((max_smem_per_threadblock - q_smem_bytes) // kv_bytes_per_mma, 0)
    num_mma_kv = min(max_num_mma_kv_smem, max_num_mma_kv_reg)
    if (
        kv_dtype == _FP8_KV_DTYPE
        and cta_tile_q == 16
        and head_dim_qk == 192
        and head_dim_vo == 128
    ):
        num_mma_kv = min(num_mma_kv, 1)
    if num_mma_kv <= 0:
        raise ValueError("no valid NUM_MMA_KV fits the current paged forward trait constraints")
    if _paged_is_invalid(
        num_mma_q=num_mma_q,
        num_mma_kv=num_mma_kv,
        num_mma_d_vo=num_mma_d_vo,
        num_warps_q=num_warps_q,
        kv_dtype=kv_dtype,
    ):
        raise ValueError("selected paged forward traits are invalid under FlashInfer rules")

    cta_tile_kv = num_mma_kv * num_warps_kv * 16

    k_smem_bytes = cta_tile_kv * upcast_stride_k * 16
    v_smem_bytes = cta_tile_kv * upcast_stride_v * 16
    qkv_storage_bytes = q_smem_bytes + k_smem_bytes + v_smem_bytes
    cta_sync_o_bytes = 4 if num_warps_kv == 1 else num_warps_kv * cta_tile_q * head_dim_vo * 4
    cta_sync_md_bytes = 8 if num_warps_kv == 1 else num_warps_kv * cta_tile_q * 8
    cta_sync_storage_bytes = cta_sync_o_bytes + cta_sync_md_bytes
    smem_o_bytes = cta_tile_q * head_dim_vo * o_bytes
    shared_storage_bytes = _align_up(max(qkv_storage_bytes, cta_sync_storage_bytes, smem_o_bytes), 16)

    return PagedForwardTraits(
        cta_tile_q=cta_tile_q,
        cta_tile_kv=cta_tile_kv,
        num_mma_q=num_mma_q,
        num_mma_kv=num_mma_kv,
        num_mma_d_qk=num_mma_d_qk,
        num_mma_d_vo=num_mma_d_vo,
        num_warps_q=num_warps_q,
        num_warps_kv=num_warps_kv,
        num_threads=num_warps_q * num_warps_kv * 32,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        upcast_stride_q=upcast_stride_q,
        upcast_stride_k=upcast_stride_k,
        upcast_stride_v=upcast_stride_v,
        upcast_stride_o=upcast_stride_o,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        o_dtype=o_dtype,
        q_smem_bytes=q_smem_bytes,
        shared_storage_bytes=shared_storage_bytes,
        max_smem_per_sm=max_smem_per_sm,
        num_ctas_per_sm=num_ctas_per_sm,
        max_smem_per_threadblock=max_smem_per_threadblock,
    )


def select_paged_forward_traits_from_plan(
    plan: PagedPlan,
    *,
    o_dtype: torch.dtype | None = None,
) -> PagedForwardTraits:
    return select_paged_forward_traits(
        cta_tile_q=plan.cta_tile_q,
        head_dim_qk=plan.head_dim_qk,
        head_dim_vo=plan.head_dim_vo,
        q_dtype=plan.dtype,
        kv_dtype=plan.kv_dtype,
        o_dtype=o_dtype,
        device=plan.device,
    )
