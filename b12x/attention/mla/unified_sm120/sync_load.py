"""Synchronous (non-TMA) gather KV loader for the unified SM120 sparse-MLA
decode -- the P5 stand-in for the P6 cp.async.bulk / mbarrier pipeline.

This loader lets the DSV4 decode MATH be validated end-to-end BEFORE the
orthogonal P6 IO/pipeline plumbing exists. It is a pure-software gather: the 256
math threads cooperatively read the ``BI`` (=64) topk candidate rows for ONE
chunk from the paged FP8 KV cache in gmem and write them into the flat-linear
smem KV regions finalized in ``smem.py``, using plain vectorized ``ld.global``
(NO ``cp.async.bulk``, NO mbarrier). When P6 lands, ``sync_gather_kv`` is
replaced by the mbarrier producer/consumer pipeline and the math is unchanged.

Data path (modeled on FlashInfer ``decode_dsv4_kernel.cuh::issue_gather`` +
``common/kv_cache_io.cuh``, but synchronous):

  * DSV4 gmem KV is BLOCK-STRUCTURED (FlashMLA footer ABI). For a paged cache
    with ``page_block_size = pbs`` tokens per block and per-block byte stride
    ``stride_kv_block``:
        block_idx = idx // pbs ;  local_idx = idx - block_idx*pbs
        data_base  = kv + block_idx*stride_kv_block + local_idx*IO_STRIDE(576)
        scale_base = kv + block_idx*stride_kv_block + pbs*IO_STRIDE + local_idx*8
    The 576-byte data row is [0:448) FP8 nope + [448:576) BF16 rope; the 8-byte
    footer is 7 UE8M0 scale bytes + 1 pad.

  * Per candidate ``entry`` in [0, BI):
      - ``cand_pos = chunk_cand_start + entry``. The RAW topk index is
        ``topk_indices[cand_pos]`` when ``cand_pos < section_len`` else -1.
      - Write that raw index (INCLUDING the -1 sentinel) into the staged
        ``token_idx`` validity buffer -- the gap #9 single source of truth for
        the S3 consumer mask (NOT re-read from gmem in the math).
      - Clamp the index to slot 0 for the data read (idx<0 -> 0), exactly as
        FlashInfer clamps before the bulk copy; the -inf masking in S3 kills the
        clamped junk.
      - Copy 448 nope bytes  -> ``kv_fp8[entry*KV_SMEM_STRIDE]`` (FLAT linear).
      - Copy 128 rope bytes  -> ``kv_rope[entry*D_ROPE_bf16]``     (FLAT linear).
      - Copy 8  footer bytes -> ``kv_sc[entry*8]``                 (DSV4 only).

  * NO ``bfloat2_mul`` pre-dequant (kernel_onepass.py:550): DSV4 keeps the FP8
    nope bytes RAW in smem and feeds the separate UE8M0 footer byte as the ``sfb``
    selector into the block-scaled QK MMA (the KEY divergence from the GLM
    onepass idiom). This loader is therefore a dequant-FREE raw byte copy.

The smem regions are addressed FLAT (``entry*STRIDE + dim``) -- matching the
DSV4 ldmatrix/d2_load_b math path, which does NOT use the GLM onepass
``_permuted_offset_128b`` XOR swizzle (see ``smem.py`` module docstring).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x.cute.fp4 import (
    get_ptr_as_int64,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    st_shared_u32,
    st_shared_v4_u32,
)

# BI: candidates per chunk for the supported decode tile (64 for both DSV4 and
# GLM). Compile-time constant so the cooperative work loops fold.
_BI = 64

# DSV4 KV gmem IO stride: data portion only (448 nope + 64 rope * 2B), 16B
# aligned for the eventual cp.async.bulk. The footer (8B) is excluded and
# addressed separately. == FlashInfer KVIOTraits<DSV4>::IO_STRIDE.
_DSV4_IO_STRIDE = 576

# Per-entry byte spans within the 576B data row.
_DSV4_NOPE_BYTES = 448  # FP8 nope; == D_NOPE
_DSV4_ROPE_BYTES = 128  # BF16 rope; == D_ROPE * 2
_DSV4_FOOTER_BYTES = 8  # 7 UE8M0 + 1 pad; == SCALE_BYTES_PER_TOKEN

# Vectorized-copy granularity: 128-bit (4 x u32) loads/stores.
_V4_BYTES = 16
_NOPE_V4 = _DSV4_NOPE_BYTES // _V4_BYTES  # 28 vec4 / entry (448 = 28*16)
_ROPE_V4 = _DSV4_ROPE_BYTES // _V4_BYTES  # 8  vec4 / entry (128 = 8*16)
_DATA_V4_PER_ENTRY = _NOPE_V4 + _ROPE_V4  # 36 vec4 / entry


@cute.jit
def _smem_byte_addr(base_addr: Int32, byte_off) -> Int32:
    """Flat smem byte address: base (u32) + byte offset (no XOR swizzle)."""
    return base_addr + Int32(byte_off)


@cute.jit
def sync_gather_kv(
    kv_cache_u8: cute.Tensor,      # flat 1-D u8 view of the paged DSV4 KV cache
    topk_indices: cute.Tensor,     # 1-D int32 topk slice for this query token
    kv_fp8_base_addr: Int32,       # u32 smem addr of kv_fp8[buf] (BI x KV_SMEM_STRIDE)
    kv_rope_base_addr: Int32,      # u32 smem addr of kv_rope[buf] (BI x D_ROPE bf16)
    kv_sc_base_addr: Int32,        # u32 smem addr of kv_sc[buf] (BI x 8); DSV4 only
    sTokenIdx: cute.Tensor,        # smem int32 validity buffer (BI,) for this buf
    chunk_cand_start: Int32,       # absolute candidate offset of entry 0 (g_start)
    section_len: Int32,            # valid topk length for this section
    kv_smem_stride: cutlass.Constexpr,   # 464 (DSV4) -- smem nope row stride
    rope_smem_stride: cutlass.Constexpr,  # 64 (D_ROPE bf16 elems) -- smem rope row stride
    page_block_size: Int32,        # pbs: tokens per paged block
    stride_kv_block: Int64,        # per-block byte stride in gmem
    tid: Int32,                    # flat thread id in [0, MATH_THREADS)
    num_threads: cutlass.Constexpr,  # MATH_THREADS (256)
    has_footer: cutlass.Constexpr = True,  # DSV4 footer scales (False -> inline GLM)
):
    """Cooperatively gather one chunk (BI=64 candidates) of KV from gmem to smem.

    Synchronous: plain vectorized ``ld.global.nc`` -> ``st.shared`` per byte
    span; the CALLER fences (``bar.sync``) after this returns so the math reads a
    coherent KV stage (P5 stands in for the P6 mbarrier full-arrival).

    Work split: ``num_threads`` threads stride over ``BI * _DATA_V4_PER_ENTRY``
    128-bit data chunks (nope+rope) so every thread copies a contiguous vec4 of
    one entry; the per-entry index resolution (gmem block/local + clamp) is
    recomputed per chunk (cheap, fully predicated). The validity buffer + footer
    are filled in separate one-thread-per-entry passes.
    """
    bi = Int32(_BI)
    rope_elem_off = Int32(_DSV4_NOPE_BYTES)  # rope starts at byte 448 in the data row

    # --- Pass 1: stage the raw topk index (incl -1) into the validity buffer. ---
    # gap #9 single source of truth; one thread per candidate entry. sTokenIdx is
    # a flat int32 smem cute.Tensor view (stride 1); the S3 mask reads it directly.
    entry = tid
    while entry < bi:
        cand_pos = chunk_cand_start + entry
        idx_raw = Int32(-1)
        if cand_pos < section_len:
            idx_raw = Int32(topk_indices[cand_pos])
        sTokenIdx[entry] = idx_raw
        entry += Int32(num_threads)

    # --- Pass 2: gather NoPE (448B) + RoPE (128B) data, FLAT linear smem. ---
    total_data = bi * Int32(_DATA_V4_PER_ENTRY)
    work = tid
    while work < total_data:
        entry = work // Int32(_DATA_V4_PER_ENTRY)
        v4 = work - entry * Int32(_DATA_V4_PER_ENTRY)

        cand_pos = chunk_cand_start + entry
        idx_raw = Int32(-1)
        if cand_pos < section_len:
            idx_raw = Int32(topk_indices[cand_pos])
        # Clamp invalid (idx<0) to slot 0 for the data read (masked in S3).
        idx = idx_raw
        if idx < Int32(0):
            idx = Int32(0)

        # Block-structured (footer ABI) gmem addressing.
        block_idx = idx // page_block_size
        local_idx = idx - block_idx * page_block_size
        data_base_off = (
            Int64(block_idx) * stride_kv_block
            + Int64(local_idx) * Int64(_DSV4_IO_STRIDE)
        )

        if v4 < Int32(_NOPE_V4):
            # NoPE vec4: gmem byte (v4*16) -> smem kv_fp8[entry*KV_SMEM_STRIDE + v4*16].
            g_byte = data_base_off + Int64(v4) * Int64(_V4_BYTES)
            d0, d1, d2, d3 = ld_global_nc_v4_u32(
                get_ptr_as_int64(kv_cache_u8, g_byte)
            )
            s_byte = entry * Int32(kv_smem_stride) + v4 * Int32(_V4_BYTES)
            st_shared_v4_u32(
                _smem_byte_addr(kv_fp8_base_addr, s_byte), d0, d1, d2, d3
            )
        else:
            # RoPE vec4: gmem byte (448 + r*16) -> smem kv_rope[entry*ROPE + r*16].
            r = v4 - Int32(_NOPE_V4)
            g_byte = data_base_off + Int64(rope_elem_off) + Int64(r) * Int64(_V4_BYTES)
            d0, d1, d2, d3 = ld_global_nc_v4_u32(
                get_ptr_as_int64(kv_cache_u8, g_byte)
            )
            # kv_rope is bf16 (2B); stride in bytes == rope_smem_stride * 2.
            s_byte = entry * Int32(rope_smem_stride * 2) + r * Int32(_V4_BYTES)
            st_shared_v4_u32(
                _smem_byte_addr(kv_rope_base_addr, s_byte), d0, d1, d2, d3
            )
        work += Int32(num_threads)

    # --- Pass 3: gather the 8B UE8M0 footer scales (DSV4 only). ---
    if cutlass.const_expr(has_footer):
        pbs_io = Int64(_DSV4_IO_STRIDE)
        entry = tid
        while entry < bi:
            cand_pos = chunk_cand_start + entry
            idx_raw = Int32(-1)
            if cand_pos < section_len:
                idx_raw = Int32(topk_indices[cand_pos])
            f0 = Uint32(0)
            f1 = Uint32(0)
            if idx_raw >= Int32(0):
                idx = idx_raw
                block_idx = idx // page_block_size
                local_idx = idx - block_idx * page_block_size
                scale_base_off = (
                    Int64(block_idx) * stride_kv_block
                    + Int64(page_block_size) * pbs_io
                    + Int64(local_idx) * Int64(_DSV4_FOOTER_BYTES)
                )
                f0, f1 = ld_global_nc_v2_u32(
                    get_ptr_as_int64(kv_cache_u8, scale_base_off)
                )
            s_byte = entry * Int32(_DSV4_FOOTER_BYTES)
            st_shared_u32(_smem_byte_addr(kv_sc_base_addr, s_byte), f0)
            st_shared_u32(_smem_byte_addr(kv_sc_base_addr, s_byte + Int32(4)), f1)
            entry += Int32(num_threads)


