"""FlashInfer-shaped MG shared-memory layout for DSV4 SM120 prefill.

This layout mirrors the DSV4 ``SmemLayoutMG`` contract used by FlashInfer prefill:
two HPB head groups per CTA, one double-buffered NoPE KV stage, separate UE8M0
scale buffers, and no DSV4 RoPE KV staging. RoPE is loaded from global/L2 by the
math path.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from .smem import SM120_SMEM_CARVEOUT_BYTES
from .traits import UnifiedMLATraits


_MG_N_HG = 2
_KV_BUF_COUNT = 2
_W_FP8_BUF_COUNT = 2
_TOKEN_IDX_BUF_COUNT = _KV_BUF_COUNT
_MATH_WARPS = 8
_W_FP8_PAD = 16
_MBAR_BYTES = _KV_BUF_COUNT * 8


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class SmemLayoutMG:
    mg_n_hg: int
    heads_per_cta: int

    q_rope_off: int
    q_rope_group_bytes: int
    q_rope_bytes: int
    q_rope_stride: int

    q_fp8_off: int
    q_fp8_group_bytes: int
    q_fp8_bytes: int
    q_nope_stride: int

    q_sc_off: int
    q_sc_group_bytes: int
    q_sc_bytes: int
    q_sc_stride: int

    kv_fp8_off: int
    kv_fp8_buf_bytes: int
    kv_smem_stride: int

    kv_sc_off: int
    kv_sc_buf_bytes: int
    kv_sc_stride: int

    kv_bufs: int

    mbar_off: int
    mbar_bytes: int

    reduce_off: int
    reduce_group_bytes: int
    reduce_warp_max_group_off: int
    reduce_warp_sum_group_off: int
    reduce_bytes: int

    w_head_sc_off: int
    w_head_sc_group_bytes: int
    w_head_sc_bytes: int
    w_head_sc_stride: int

    w_fp8_off: int
    w_fp8_group_bytes: int
    w_fp8_parity_bytes: int
    w_fp8_stride: int
    w_fp8_bufs: int

    token_idx_off: int
    token_idx_buf_bytes: int
    token_idx_bufs: int

    sm_p_full_off: int
    sm_p_full_group_bytes: int
    sm_p_full_bytes: int
    sm_p_full_stride: int

    total_bytes: int


def make_smem_layout_mg(traits: UnifiedMLATraits) -> SmemLayoutMG:
    if not traits.has_extra_cache:
        raise ValueError("MG prefill layout is DSV4-only")

    bi = traits.bi
    hpb = traits.hpb
    d_rope = traits.d_rope
    num_scales = traits.num_scales
    n_v_chunks = traits.n_v_chunks
    kv_smem_stride = traits.kv_smem_stride
    q_nope_stride = traits.q_nope_stride
    bufs = _KV_BUF_COUNT
    mg_n_hg = _MG_N_HG
    heads_per_cta = mg_n_hg * hpb

    off = 0

    q_rope_off = off
    q_rope_stride = d_rope
    q_rope_group_bytes = hpb * d_rope * 2
    q_rope_bytes = mg_n_hg * q_rope_group_bytes
    off = q_rope_off + q_rope_bytes

    q_fp8_off = off
    q_fp8_group_bytes = hpb * q_nope_stride
    q_fp8_bytes = mg_n_hg * q_fp8_group_bytes
    off = q_fp8_off + q_fp8_bytes

    q_sc_off = off
    q_sc_stride = num_scales
    q_sc_group_bytes = hpb * num_scales * 4
    q_sc_bytes = mg_n_hg * q_sc_group_bytes
    off = q_sc_off + q_sc_bytes

    off = _align_up(off, 128)
    kv_fp8_off = off
    kv_fp8_buf_bytes = bi * kv_smem_stride
    off = kv_fp8_off + kv_fp8_buf_bytes * bufs

    kv_sc_off = off
    kv_sc_stride = 8
    kv_sc_buf_bytes = bi * kv_sc_stride
    off = kv_sc_off + kv_sc_buf_bytes * bufs

    off = _align_up(off, 16)
    mbar_off = off
    mbar_bytes = _MBAR_BYTES
    off = mbar_off + mbar_bytes

    reduce_off = off
    reduce_group_bytes = 2 * _MATH_WARPS * hpb * 4
    reduce_warp_max_group_off = 0
    reduce_warp_sum_group_off = _MATH_WARPS * hpb * 4
    reduce_bytes = mg_n_hg * reduce_group_bytes
    off = reduce_off + reduce_bytes

    w_head_sc_off = off
    w_head_sc_stride = hpb
    w_head_sc_group_bytes = n_v_chunks * hpb * 4
    w_head_sc_bytes = mg_n_hg * w_head_sc_group_bytes
    off = w_head_sc_off + w_head_sc_bytes

    w_fp8_off = off
    w_fp8_stride = bi + _W_FP8_PAD
    w_fp8_group_bytes = hpb * w_fp8_stride
    w_fp8_parity_bytes = mg_n_hg * w_fp8_group_bytes
    w_fp8_bufs = _W_FP8_BUF_COUNT
    off = w_fp8_off + w_fp8_parity_bytes * w_fp8_bufs

    off = _align_up(off, 16)
    token_idx_off = off
    token_idx_buf_bytes = bi * 4
    token_idx_bufs = _TOKEN_IDX_BUF_COUNT
    off = token_idx_off + token_idx_buf_bytes * token_idx_bufs

    off = _align_up(off, 128)
    sm_p_full_off = off
    sm_p_full_stride = bi
    sm_p_full_group_bytes = hpb * bi * 2
    sm_p_full_bytes = mg_n_hg * sm_p_full_group_bytes
    off = sm_p_full_off + sm_p_full_bytes

    total_bytes = _align_up(off, 128)

    return SmemLayoutMG(
        mg_n_hg=mg_n_hg,
        heads_per_cta=heads_per_cta,
        q_rope_off=q_rope_off,
        q_rope_group_bytes=q_rope_group_bytes,
        q_rope_bytes=q_rope_bytes,
        q_rope_stride=q_rope_stride,
        q_fp8_off=q_fp8_off,
        q_fp8_group_bytes=q_fp8_group_bytes,
        q_fp8_bytes=q_fp8_bytes,
        q_nope_stride=q_nope_stride,
        q_sc_off=q_sc_off,
        q_sc_group_bytes=q_sc_group_bytes,
        q_sc_bytes=q_sc_bytes,
        q_sc_stride=q_sc_stride,
        kv_fp8_off=kv_fp8_off,
        kv_fp8_buf_bytes=kv_fp8_buf_bytes,
        kv_smem_stride=kv_smem_stride,
        kv_sc_off=kv_sc_off,
        kv_sc_buf_bytes=kv_sc_buf_bytes,
        kv_sc_stride=kv_sc_stride,
        kv_bufs=bufs,
        mbar_off=mbar_off,
        mbar_bytes=mbar_bytes,
        reduce_off=reduce_off,
        reduce_group_bytes=reduce_group_bytes,
        reduce_warp_max_group_off=reduce_warp_max_group_off,
        reduce_warp_sum_group_off=reduce_warp_sum_group_off,
        reduce_bytes=reduce_bytes,
        w_head_sc_off=w_head_sc_off,
        w_head_sc_group_bytes=w_head_sc_group_bytes,
        w_head_sc_bytes=w_head_sc_bytes,
        w_head_sc_stride=w_head_sc_stride,
        w_fp8_off=w_fp8_off,
        w_fp8_group_bytes=w_fp8_group_bytes,
        w_fp8_parity_bytes=w_fp8_parity_bytes,
        w_fp8_stride=w_fp8_stride,
        w_fp8_bufs=w_fp8_bufs,
        token_idx_off=token_idx_off,
        token_idx_buf_bytes=token_idx_buf_bytes,
        token_idx_bufs=token_idx_bufs,
        sm_p_full_off=sm_p_full_off,
        sm_p_full_group_bytes=sm_p_full_group_bytes,
        sm_p_full_bytes=sm_p_full_bytes,
        sm_p_full_stride=sm_p_full_stride,
        total_bytes=total_bytes,
    )


def get_prefill_mg_shared_storage_cls(traits: UnifiedMLATraits):
    layout = make_smem_layout_mg(traits)

    class SharedStorageMG:
        pass

    SharedStorageMG.__annotations__ = {
        "q_rope": cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, int(layout.q_rope_bytes // 2)],
            128,
        ],
        "q_fp8": cute.struct.MemRange[cutlass.Uint8, int(layout.q_fp8_bytes)],
        "q_sc": cute.struct.MemRange[cutlass.Float32, int(layout.q_sc_bytes // 4)],
        "kv_fp8": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(layout.kv_fp8_buf_bytes * layout.kv_bufs)],
            128,
        ],
        "kv_sc": cute.struct.MemRange[
            cutlass.Uint8, int(layout.kv_sc_buf_bytes * layout.kv_bufs)
        ],
        "mbar": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint64, int(layout.mbar_bytes // 8)],
            16,
        ],
        "reduce": cute.struct.MemRange[cutlass.Float32, int(layout.reduce_bytes // 4)],
        "w_head_sc": cute.struct.MemRange[
            cutlass.Float32, int(layout.w_head_sc_bytes // 4)
        ],
        "w_fp8": cute.struct.MemRange[
            cutlass.Uint8, int(layout.w_fp8_parity_bytes * layout.w_fp8_bufs)
        ],
        "token_idx": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Int32,
                int(layout.token_idx_buf_bytes * layout.token_idx_bufs // 4),
            ],
            16,
        ],
        "sm_p_full": cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, int(layout.sm_p_full_bytes // 2)],
            128,
        ],
    }
    return cute.struct(SharedStorageMG)


def _run_module_asserts() -> None:
    from .traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits

    traits = make_unified_traits(ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE)
    layout = make_smem_layout_mg(traits)
    assert layout.kv_smem_stride == 464
    assert layout.kv_fp8_buf_bytes == 64 * 464
    assert layout.kv_sc_buf_bytes == 64 * 8
    assert layout.q_rope_off == 0
    assert layout.total_bytes < SM120_SMEM_CARVEOUT_BYTES, (
        f"MG prefill smem {layout.total_bytes}B exceeds SM120 carveout "
        f"{SM120_SMEM_CARVEOUT_BYTES}B"
    )


_run_module_asserts()
