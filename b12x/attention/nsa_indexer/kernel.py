"""CuTeDSL paged decode score kernel for the paged NSA contract."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import os
import warnings

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass import Float32, Int32, Uint32
from cutlass.cutlass_dsl import Int64, dsl_user_op
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from b12x.attention import copy_utils
from b12x.attention import pipeline
from b12x.attention import utils as attention_utils
from b12x.cute.fp4 import get_sm_version
from b12x.cute.fp4 import (
    frag_layout_swizzle_16b_to_8b,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    st_shared_v4_u32,
)
from b12x.runtime_control import raise_if_kernel_resolution_frozen
from b12x.cute.utils import current_cuda_stream


_INDEX_HEAD_DIM = 128
_PAGE_SIZE = 64
_SCALE_BYTES = 4
_WARP_THREADS = 32
_PAGED_WARPS_PER_CTA = 4
_PAGED_THREADS_PER_CTA = _WARP_THREADS * _PAGED_WARPS_PER_CTA
_PAGED_TOKENS_PER_GROUP = 8
PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT = 4
_SCHEDULE_MIN_PAGES = 1024
_ENABLE_MULTI_ROW_SCHEDULE = True
_SCHEDULE_SINGLE_ROW_PARALLEL_CTAS = 4
_SCHEDULE_MULTI_ROW_PARALLEL_CTAS = 4
_SCHEDULE_MULTI_ROW_MAX_Q_ROWS = 8
_MAX_SUPPORTED_Q_HEADS = 64
_PAGED_TILED_BLOCK_Q = 32
_PAGED_TILED_BLOCK_K = 512
_EAGER_HOST_LAUNCHER_CACHE_SIZE = int(
    os.getenv("B12X_EAGER_HOST_LAUNCHER_CACHE_SIZE", "512")
)
_PAGED_Q_HEAD_TILE = 16
_PAGED_K_TMA_DESC_CACHE_SIZE = 32
_BLACKWELL_TMA_DESC_WORDS = 16
_BLACKWELL_TMA_DESC_MIN_BACKING_BYTES = 128 * 1024
_BLACKWELL_TMA_DESC_PATCH_WORD = 1
_BLACKWELL_TMA_DESC_PATCH_CLEAR_MASK = 0xFFFFFFFFFFDFFFFF


def _num_q_head_tiles(num_heads: int) -> int:
    return max((num_heads + _PAGED_Q_HEAD_TILE - 1) // _PAGED_Q_HEAD_TILE, 1)


def _page_splits_for_num_heads(num_heads: int) -> int:
    num_q_head_tiles = _num_q_head_tiles(num_heads)
    token_groups = _PAGED_WARPS_PER_CTA // num_q_head_tiles
    tokens_per_work = _PAGED_TOKENS_PER_GROUP * token_groups
    return _PAGE_SIZE // tokens_per_work


@lru_cache(maxsize=16)
def _paged_indexer_shared_storage_cls(
    padded_q_heads: int,
    tokens_per_work: int,
    num_q_head_tiles: int,
):
    class SharedStorage:
        pass

    def k_page_storage():
        return cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, _PAGE_SIZE * _INDEX_HEAD_DIM],
            1024,
        ]

    annotations = {
        "mbar_ptr_k": cute.struct.MemRange[cutlass.Int64, 1],
        "q_bytes": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(padded_q_heads) * _INDEX_HEAD_DIM],
            16,
        ],
        "weights": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, int(padded_q_heads)],
            16,
        ],
    }
    annotations["k_page"] = k_page_storage()
    annotations.update(
        {
            "k_page_perm": k_page_storage(),
            "scales": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _PAGE_SIZE],
                16,
            ],
            "partial_logits": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    int(tokens_per_work) * int(num_q_head_tiles),
                ],
                16,
            ],
        }
    )
    SharedStorage.__annotations__ = annotations
    return cute.struct(SharedStorage)


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> cutlass.cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    leading_dim = next((idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None)
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_meta_key(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _launcher_cache_lookup(
    kernel: object,
    cache_key: tuple[object, ...],
):
    cache = getattr(kernel, "_eager_host_launchers", None)
    if cache is None:
        cache = OrderedDict()
        setattr(kernel, "_eager_host_launchers", cache)
        return cache, None
    compiled = cache.get(cache_key)
    if compiled is not None:
        cache.move_to_end(cache_key)
    return cache, compiled


def _run_cached_host_launcher(
    kernel: object,
    cache_key: tuple[object, ...],
    args: tuple[object, ...],
) -> None:
    cache, compiled = _launcher_cache_lookup(kernel, cache_key)
    if compiled is None:
        raise_if_kernel_resolution_frozen(
            "eager host launcher compile",
            target=kernel,
            cache_key=cache_key,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cache is disabled as user wants to compile only.",
                category=UserWarning,
            )
            compiled = kernel(*args, compile_only=True)
        cache[cache_key] = compiled
        if len(cache) > _EAGER_HOST_LAUNCHER_CACHE_SIZE:
            cache.popitem(last=False)
    exe_args, _ = compiled.generate_execution_args(*args)
    compiled.run_compiled_program(exe_args)


def _assume_paged_k_tma_source_aligned(t: cute.Tensor):
    divby = 128 // t.element_type.width
    strides = []
    for dim, stride in enumerate(t.stride):
        if dim == 1 or isinstance(stride, int):
            strides.append(stride)
        else:
            strides.append(cute.assume(stride, divby=divby))
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=tuple(strides)))


def _make_paged_index_k_tma_source(k_quant_bytes: cutlass.cute.Tensor) -> cutlass.cute.Tensor:
    t_pages = cute.make_tensor(k_quant_bytes.iterator, cute.select(k_quant_bytes.layout, mode=[1, 2, 0]))
    return _assume_paged_k_tma_source_aligned(t_pages)


@lru_cache(maxsize=16)
def _default_sparse_nsa_persistent_ctas(device_index: int) -> int:
    num_sms = int(torch.cuda.get_device_properties(device_index).multi_processor_count)
    return max(num_sms * 4, 1)


def _resolve_sparse_nsa_persistent_ctas(
    *,
    device_index: int,
    q_rows: int,
) -> int:
    persistent_ctas = _default_sparse_nsa_persistent_ctas(device_index)
    if q_rows >= 4:
        persistent_ctas = max(persistent_ctas // 2, 1)
    return persistent_ctas


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    storage = tensor.untyped_storage() if hasattr(tensor, "untyped_storage") else tensor.storage()
    if hasattr(storage, "nbytes"):
        return int(storage.nbytes())
    return int(storage.size()) * tensor.element_size()


def _is_dense_non_overlapping(tensor: torch.Tensor) -> bool:
    expected_stride = 1
    strides_and_sizes = []
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if int(size) <= 1:
            continue
        strides_and_sizes.append((abs(int(stride)), int(size)))
    for stride, size in sorted(strides_and_sizes):
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _needs_paged_index_k_tma_descriptor_workaround(
    index_k_cache: torch.Tensor,
    k_quant_bytes: torch.Tensor,
) -> bool:
    if get_sm_version(index_k_cache.device) // 10 != 12:
        return False
    if _tensor_storage_nbytes(index_k_cache) >= _BLACKWELL_TMA_DESC_MIN_BACKING_BYTES:
        return False
    return not _is_dense_non_overlapping(k_quant_bytes)


def _encode_paged_index_k_tma_descriptor(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_quant_bytes.ndim != 3 or k_quant_bytes.shape[1:] != (_PAGE_SIZE, _INDEX_HEAD_DIM):
        raise ValueError(
            "k_quant_bytes must have shape "
            f"(num_pages, {_PAGE_SIZE}, {_INDEX_HEAD_DIM}), got {tuple(k_quant_bytes.shape)}"
        )
    if k_quant_bytes.dtype != torch.uint8:
        raise TypeError(f"k_quant_bytes must have dtype torch.uint8, got {k_quant_bytes.dtype}")

    num_pages = int(k_quant_bytes.shape[0])
    row_stride_bytes = int(k_quant_bytes.stride(1)) * k_quant_bytes.element_size()
    page_stride_bytes = int(k_quant_bytes.stride(0)) * k_quant_bytes.element_size()
    base_ptr = int(k_quant_bytes.data_ptr())
    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t

    result, tensor_map = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3,
        base_ptr,
        [U64(_INDEX_HEAD_DIM), U64(_PAGE_SIZE), U64(num_pages)],
        [U64(row_stride_bytes), U64(page_stride_bytes)],
        [U32(_INDEX_HEAD_DIM), U32(_PAGE_SIZE), U32(1)],
        [U32(1), U32(1), U32(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")

    host_desc_words = [int(word) for word in tensor_map.opaque]
    if len(host_desc_words) != _BLACKWELL_TMA_DESC_WORDS:
        raise RuntimeError(
            f"unexpected TMA descriptor size: expected {_BLACKWELL_TMA_DESC_WORDS} words, "
            f"got {len(host_desc_words)}"
        )
    host_desc = torch.tensor(host_desc_words, dtype=torch.uint64)
    host_desc[_BLACKWELL_TMA_DESC_PATCH_WORD] &= _BLACKWELL_TMA_DESC_PATCH_CLEAR_MASK
    desc = host_desc.to(device=k_quant_bytes.device, non_blocking=False)
    desc_ptrs = torch.tensor([int(desc.data_ptr())], dtype=torch.int64, device=k_quant_bytes.device)
    return desc, desc_ptrs


def _get_cached_paged_index_k_tma_descriptor(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        int(k_quant_bytes.data_ptr()),
        tuple(k_quant_bytes.shape),
        tuple(k_quant_bytes.stride()),
        str(k_quant_bytes.dtype),
        (k_quant_bytes.device.type, k_quant_bytes.device.index),
    )
    cache = getattr(_get_cached_paged_index_k_tma_descriptor, "_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(_get_cached_paged_index_k_tma_descriptor, "_cache", cache)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    desc = _encode_paged_index_k_tma_descriptor(k_quant_bytes)
    cache[key] = desc
    if len(cache) > _PAGED_K_TMA_DESC_CACHE_SIZE:
        cache.popitem(last=False)
    return desc


@lru_cache(maxsize=16)
def _dummy_paged_index_k_tma_desc_ptrs(device_index: int) -> torch.Tensor:
    return torch.zeros((1,), dtype=torch.int64, device=torch.device("cuda", device_index))


@lru_cache(maxsize=32)
def _cached_int32_scalar(value: int, device_index: int) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int32, device=torch.device("cuda", device_index))


@dsl_user_op
def _cp_async_bulk_tensor_3d(
    dst_smem_addr: Int32,
    tensor_map_ptr: Int64,
    coord0: Int32,
    coord1: Int32,
    coord2: Int32,
    mbar_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(dst_smem_addr).ir_value(loc=loc, ip=ip),
            Int64(tensor_map_ptr).ir_value(loc=loc, ip=ip),
            Int32(coord0).ir_value(loc=loc, ip=ip),
            Int32(coord1).ir_value(loc=loc, ip=ip),
            Int32(coord2).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[$0], [$1, {$2, $3, $4}], [$5];",
        "r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _permuted_offset_128b(row_idx, vec_idx, row_stride_128b):
    return row_idx * row_stride_128b + (vec_idx ^ (row_idx % 8))


@cute.jit
def _smem_addr_from_b128_offset(base_addr: Int32, offset_128b):
    return base_addr + Int32(offset_128b * 16)


@cute.jit
def _advance_offset_by_row_128b(offset_128b, step_size, row_stride_128b):
    return offset_128b + step_size * row_stride_128b


@cute.jit
def _advance_offset_by_column_128b_2(offset_128b, step_idx):
    xor_term = Int32(0x2) + (Int32(0x4) if step_idx % 2 == 1 else Int32(0))
    extra = Int32(8) if step_idx % 4 == 3 else Int32(0)
    return (offset_128b ^ xor_term) + extra


@cute.jit
def _pack_q_mxfp8_reg(
    s_q_bytes: cute.Tensor,
    row: Int32,
    col_pair_base: Int32,
) -> Uint32:
    b0 = Uint32(s_q_bytes[row, col_pair_base + Int32(0)])
    b1 = Uint32(s_q_bytes[row, col_pair_base + Int32(1)])
    b2 = Uint32(s_q_bytes[row, col_pair_base + Int32(8)])
    b3 = Uint32(s_q_bytes[row, col_pair_base + Int32(9)])
    return b0 | (b1 << Int32(8)) | (b2 << Int32(16)) | (b3 << Int32(24))


@cute.jit
def _repack_k_page_to_permuted(
    k_linear_base_addr: Int32,
    k_perm_base_addr: Int32,
    lane_linear: Int32,
):
    linear = lane_linear
    total = Int32(_PAGE_SIZE * (_INDEX_HEAD_DIM // 16))
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM // 16)
        vec_idx = linear - row * Int32(_INDEX_HEAD_DIM // 16)
        src_addr = k_linear_base_addr + Int32(row * _INDEX_HEAD_DIM + vec_idx * 16)
        dst_addr = _smem_addr_from_b128_offset(
            k_perm_base_addr,
            _permuted_offset_128b(row, vec_idx, Int32(_INDEX_HEAD_DIM // 16)),
        )
        v0, v1, v2, v3 = ld_shared_v4_u32(src_addr)
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(_PAGED_THREADS_PER_CTA)


@cute.jit
def _reduce_column_pair_sum(value: Float32) -> Float32:
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=4))
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=8))
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=16))
    return value


@cute.jit
def _compute_mxfp8_tile_partials(
    s_q_bytes: cute.Tensor,
    s_w: cute.Tensor,
    num_heads: Int32,
    k_perm_base_addr: Int32,
    token_base: Int32,
    head_tile_base: Int32,
    lane: Int32,
    s_partial_logits: cute.Tensor,
    partial_row_base: Int32,
    head_tile_slot: Int32,
):
    group_id = lane // Int32(4)
    thread_id_in_group = lane % Int32(4)
    col_pair_base = thread_id_in_group * Int32(2)
    q0_acc = Float32(0.0)
    q1_acc = Float32(0.0)
    q2_acc = Float32(0.0)
    q3_acc = Float32(0.0)
    k_offset = _permuted_offset_128b(
        token_base + Int32(8) * (lane // Int32(16)) + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        Int32(_INDEX_HEAD_DIM // 16),
    )
    for mma_pair in cutlass.range_constexpr(_INDEX_HEAD_DIM // 32):
        pair_base = Int32(mma_pair * 32) + col_pair_base
        q0 = _pack_q_mxfp8_reg(s_q_bytes, head_tile_base + group_id, pair_base)
        q1 = _pack_q_mxfp8_reg(s_q_bytes, head_tile_base + group_id + Int32(8), pair_base)
        q2 = _pack_q_mxfp8_reg(s_q_bytes, head_tile_base + group_id, pair_base + Int32(16))
        q3 = _pack_q_mxfp8_reg(
            s_q_bytes,
            head_tile_base + group_id + Int32(8),
            pair_base + Int32(16),
        )
        b0_k0, _ = ldmatrix_m8n8x4_left_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k1, _ = ldmatrix_m8n8x4_right_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
        k_offset_cur = _advance_offset_by_row_128b(
            k_offset,
            Int32(16),
            Int32(_INDEX_HEAD_DIM // 16),
        )
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            q0_acc,
            q1_acc,
            q2_acc,
            q3_acc,
            q0,
            q1,
            q2,
            q3,
            b0_k0,
            b0_k1,
            Uint32(0x7F7F7F7F),
            Uint32(0x7F7F7F7F),
        )
        q0_acc = d0
        q1_acc = d1
        q2_acc = d2
        q3_acc = d3
        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            16 * (_INDEX_HEAD_DIM // 16)
        )

    head0 = head_tile_base + group_id
    head1 = head0 + Int32(8)
    w0 = Float32(0.0)
    w1 = Float32(0.0)
    if head0 < num_heads:
        w0 = Float32(s_w[head0])
    if head1 < num_heads:
        w1 = Float32(s_w[head1])
    col0 = col_pair_base
    col1 = col_pair_base + Int32(1)
    partial0 = Float32(attention_utils.fmax(q0_acc, Float32(0.0)) * w0)
    partial0 = Float32(partial0 + attention_utils.fmax(q2_acc, Float32(0.0)) * w1)
    partial1 = Float32(attention_utils.fmax(q1_acc, Float32(0.0)) * w0)
    partial1 = Float32(partial1 + attention_utils.fmax(q3_acc, Float32(0.0)) * w1)
    partial0 = _reduce_column_pair_sum(partial0)
    partial1 = _reduce_column_pair_sum(partial1)
    if group_id == Int32(0):
        s_partial_logits[partial_row_base + col0, head_tile_slot] = partial0
        s_partial_logits[partial_row_base + col1, head_tile_slot] = partial1


@cute.jit
def _issue_index_k_tma_copy(
    load_tma,
    producer_state,
    mbar_ptr,
    expected_bytes,
    page_id,
    dst_smem_addr,
    tma_desc_ptr=None,
    use_external_tma_desc=Int32(0),
):
    full_mbar_ptr = mbar_ptr + producer_state.index
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(full_mbar_ptr, expected_bytes)
    if use_external_tma_desc != Int32(0):
        _cp_async_bulk_tensor_3d(
            Int32(dst_smem_addr),
            Int64(tma_desc_ptr),
            Int32(0),
            Int32(0),
            page_id,
            Int32(shared_ptr_to_u32(full_mbar_ptr)),
        )
    else:
        load_tma(src_idx=page_id, dst_idx=producer_state.index, tma_bar_ptr=full_mbar_ptr)


class SparseNSAPagedLogitsKernel:
    """One CTA reuses a query row across a small tile of paged candidate positions."""

    def __init__(
        self,
        persistent_ctas: int,
        num_heads_static: int,
        *,
        tiled_output: bool = False,
        tile_block_q: int = _PAGED_TILED_BLOCK_Q,
        tile_block_k: int = _PAGED_TILED_BLOCK_K,
    ):
        self.persistent_ctas = int(persistent_ctas)
        self.num_heads_static = int(num_heads_static)
        self.tiled_output = bool(tiled_output)
        self.tile_block_q = int(tile_block_q)
        self.tile_block_k = int(tile_block_k)
        if self.tiled_output and self.tile_block_k != _PAGED_TILED_BLOCK_K:
            raise ValueError(
                f"paged tiled logits currently require block_k={_PAGED_TILED_BLOCK_K}, "
                f"got {self.tile_block_k}"
            )
        self.num_q_head_tiles = _num_q_head_tiles(self.num_heads_static)
        if self.num_q_head_tiles not in (1, 2, 4):
            raise ValueError(
                f"paged logits kernel only supports 1/2/4 head tiles, got {self.num_q_head_tiles}"
            )
        self.padded_q_heads = self.num_q_head_tiles * _PAGED_Q_HEAD_TILE
        self.token_groups = _PAGED_WARPS_PER_CTA // self.num_q_head_tiles
        self.tokens_per_work = _PAGED_TOKENS_PER_GROUP * self.token_groups
        self.page_splits = _PAGE_SIZE // self.tokens_per_work

    def _get_shared_storage_cls(self):
        return _paged_indexer_shared_storage_cls(
            self.padded_q_heads,
            self.tokens_per_work,
            self.num_q_head_tiles,
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_patched_k_tma_desc: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_paged_index_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            cute.make_layout((_PAGE_SIZE, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)),
            (_PAGE_SIZE, _INDEX_HEAD_DIM),
            1,
        )
        SharedStorage = self._get_shared_storage_cls()
        self.kernel(
            q_bytes,
            weights,
            tma_tensor_k,
            k_tma_desc_ptrs,
            use_patched_k_tma_desc,
            k_scales,
            real_page_table,
            seqlens_per_query,
            active_width,
            Int32(0),
            Int32(0),
            logits_out,
            tma_atom_k,
        ).launch(
            grid=(
                q_bytes.shape[0],
                self.persistent_ctas,
                1,
            ),
            block=[_PAGED_THREADS_PER_CTA, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_tma_tensor: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_patched_k_tma_desc: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        source_page_offset: Int32,
        output_width_tokens: Int32,
        logits_out: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
    ):
        tx, _, _ = cute.arch.thread_idx()
        q_idx, cta_idx, _ = cute.arch.block_idx()
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()

        source_width_tokens = Int32(real_page_table.shape[1]) * Int32(_PAGE_SIZE)
        source_offset_pages = source_page_offset
        if source_offset_pages < Int32(0):
            source_offset_pages = Int32(0)
        source_offset_tokens = source_offset_pages * Int32(_PAGE_SIZE)
        remaining_width_tokens = source_width_tokens - source_offset_tokens
        if remaining_width_tokens < Int32(0):
            remaining_width_tokens = Int32(0)

        width_tokens = output_width_tokens
        if width_tokens <= Int32(0):
            width_tokens = remaining_width_tokens
        valid_width_tokens = width_tokens
        if valid_width_tokens > remaining_width_tokens:
            valid_width_tokens = remaining_width_tokens

        live_width = Int32(active_width[Int32(0)])
        if live_width > source_width_tokens:
            live_width = source_width_tokens
        live_width = live_width - source_offset_tokens
        if live_width < Int32(0):
            live_width = Int32(0)
        if live_width > valid_width_tokens:
            live_width = valid_width_tokens

        seq_len = Int32(seqlens_per_query[q_idx]) - source_offset_tokens
        if seq_len < Int32(0):
            seq_len = Int32(0)
        if seq_len > live_width:
            seq_len = live_width
        total_work = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)

        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        s_q = storage.q_bytes.get_tensor(
            cute.make_layout((self.padded_q_heads, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1))
        )
        s_w = storage.weights.get_tensor(cute.make_layout((self.padded_q_heads,), stride=(1,)))
        k_page_base_addr = shared_ptr_to_u32(storage.k_page.data_ptr())
        k_page_perm_base_addr = shared_ptr_to_u32(storage.k_page_perm.data_ptr())
        s_k_page_stage = storage.k_page.get_tensor(
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM, 1),
                stride=(_INDEX_HEAD_DIM, 1, _PAGE_SIZE * _INDEX_HEAD_DIM),
            )
        )
        s_scale = storage.scales.get_tensor(cute.make_layout((_PAGE_SIZE,), stride=(1,)))
        s_partial_logits = storage.partial_logits.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        load_k_tma, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.local_tile(k_tma_tensor, (_PAGE_SIZE, _INDEX_HEAD_DIM), (0, 0, None)),
            s_k_page_stage,
        )
        use_external_k_tma_desc = Int32(use_patched_k_tma_desc[Int32(0)])
        k_tma_desc_ptr = Int64(k_tma_desc_ptrs[Int32(0)])
        if cta_idx < total_work:
            if tx == 0:
                cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
            if warp_idx == Int32(0) and use_external_k_tma_desc == Int32(0):
                cpasync.prefetch_descriptor(tma_atom_k)

            num_heads = Int32(self.num_heads_static)
            q_linear = tx
            total_q_bytes = Int32(self.padded_q_heads * _INDEX_HEAD_DIM)
            while q_linear < total_q_bytes:
                head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
                col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
                s_q[head_idx, col_idx] = (
                    q_bytes[q_idx, head_idx, col_idx]
                    if head_idx < num_heads
                    else cutlass.Uint8(0)
                )
                q_linear += Int32(_PAGED_THREADS_PER_CTA)

            w_linear = tx
            while w_linear < Int32(self.padded_q_heads):
                s_w[w_linear] = (
                    Float32(weights[q_idx, w_linear]) if w_linear < num_heads else Float32(0.0)
                )
                w_linear += Int32(_PAGED_THREADS_PER_CTA)
            cute.arch.sync_threads()

            producer_state = pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = pipeline.PipelineStateSimple(1, Int32(0))
            head_tile_slot = warp_idx % Int32(self.num_q_head_tiles)
            token_group = warp_idx // Int32(self.num_q_head_tiles)
            work_idx = cta_idx
            while work_idx < total_work:
                page_col = work_idx
                source_page_col = source_offset_pages + page_col
                page_base = page_col * Int32(_PAGE_SIZE)
                if page_base < seq_len and source_page_col < Int32(real_page_table.shape[1]):
                    page_id = Int32(real_page_table[q_idx, source_page_col])
                    if page_id >= Int32(0):
                        if warp_idx == Int32(0):
                            _issue_index_k_tma_copy(
                                load_k_tma,
                                producer_state,
                                mbar_ptr_k,
                                Int32(_PAGE_SIZE * _INDEX_HEAD_DIM),
                                page_id,
                                k_page_base_addr,
                                tma_desc_ptr=k_tma_desc_ptr,
                                use_external_tma_desc=use_external_k_tma_desc,
                            )
                        scale_idx = tx
                        while scale_idx < Int32(_PAGE_SIZE):
                            s_scale[scale_idx] = Float32(k_scales[page_id, scale_idx])
                            scale_idx += Int32(_PAGED_THREADS_PER_CTA)
                        cute.arch.mbarrier_wait(
                            mbar_ptr_k + consumer_state.index,
                            phase=consumer_state.phase,
                        )
                        cute.arch.sync_threads()
                        _repack_k_page_to_permuted(k_page_base_addr, k_page_perm_base_addr, tx)
                        cute.arch.sync_threads()

                        valid_slots = seq_len - page_base
                        if valid_slots > Int32(_PAGE_SIZE):
                            valid_slots = Int32(_PAGE_SIZE)
                        split_idx = Int32(0)
                        while split_idx < Int32(self.page_splits):
                            token_base = (
                                split_idx * Int32(self.tokens_per_work)
                                + token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                            )
                            zero_idx = tx
                            while zero_idx < Int32(self.tokens_per_work * self.num_q_head_tiles):
                                token_idx = zero_idx // Int32(self.num_q_head_tiles)
                                head_tile_idx = zero_idx - token_idx * Int32(self.num_q_head_tiles)
                                s_partial_logits[token_idx, head_tile_idx] = Float32(0.0)
                                zero_idx += Int32(_PAGED_THREADS_PER_CTA)
                            cute.arch.sync_threads()
                            if token_base < valid_slots:
                                head_tile_base = head_tile_slot * Int32(_PAGED_Q_HEAD_TILE)
                                _compute_mxfp8_tile_partials(
                                    s_q,
                                    s_w,
                                    num_heads,
                                    k_page_perm_base_addr,
                                    token_base,
                                    head_tile_base,
                                    lane,
                                    s_partial_logits,
                                    token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                    head_tile_slot,
                                )
                            cute.arch.sync_threads()
                            if head_tile_slot == Int32(0) and lane < Int32(_PAGED_TOKENS_PER_GROUP):
                                slot_idx = token_base + lane
                                if slot_idx < valid_slots:
                                    logit = Float32(0.0)
                                    head_tile_idx = Int32(0)
                                    partial_row = token_group * Int32(_PAGED_TOKENS_PER_GROUP) + lane
                                    while head_tile_idx < Int32(self.num_q_head_tiles):
                                        logit = Float32(
                                            logit + s_partial_logits[partial_row, head_tile_idx]
                                        )
                                        head_tile_idx += Int32(1)
                                    value = Float32(logit * s_scale[slot_idx])
                                    if cutlass.const_expr(self.tiled_output):
                                        tile_block_q = Int32(self.tile_block_q)
                                        tile_block_k = Int32(self.tile_block_k)
                                        tile_size = Int32(self.tile_block_q * self.tile_block_k)
                                        num_k_tiles = (
                                            width_tokens + tile_block_k - Int32(1)
                                        ) // tile_block_k
                                        q_tile_idx = q_idx // tile_block_q
                                        q_local = q_idx - q_tile_idx * tile_block_q
                                        k_tile_idx = page_base // tile_block_k
                                        k_local = page_base - k_tile_idx * tile_block_k + slot_idx
                                        flat_offset = (
                                            q_tile_idx * num_k_tiles * tile_size
                                            + k_tile_idx * tile_size
                                            + q_local * tile_block_k
                                            + k_local
                                        )
                                        logits_out[flat_offset] = value
                                    else:
                                        logits_out[q_idx, page_base + slot_idx] = value
                            cute.arch.sync_threads()
                            split_idx += Int32(1)
                        producer_state.advance()
                        consumer_state.advance()
                        cute.arch.sync_threads()

                work_idx += Int32(self.persistent_ctas)


class SparseNSAPagedWindowedTiledLogitsKernel(SparseNSAPagedLogitsKernel):
    """C4 supertile adapter over a full paged table.

    The shared paged NSA scorer keeps its historical public ABI.  This wrapper
    exposes the extra page offset/window scalars only to compressed-indexer
    supertile callers.
    """

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_patched_k_tma_desc: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        source_page_offset: Int32,
        output_width_tokens: Int32,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_paged_index_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            cute.make_layout((_PAGE_SIZE, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)),
            (_PAGE_SIZE, _INDEX_HEAD_DIM),
            1,
        )
        SharedStorage = self._get_shared_storage_cls()
        self.kernel(
            q_bytes,
            weights,
            tma_tensor_k,
            k_tma_desc_ptrs,
            use_patched_k_tma_desc,
            k_scales,
            real_page_table,
            seqlens_per_query,
            active_width,
            source_page_offset,
            output_width_tokens,
            logits_out,
            tma_atom_k,
        ).launch(
            grid=(
                q_bytes.shape[0],
                self.persistent_ctas,
                1,
            ),
            block=[_PAGED_THREADS_PER_CTA, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )


def _should_use_schedule_kernel(
    *,
    q_rows: int,
    max_pages: int,
) -> bool:
    return q_rows > 0 and max_pages >= _SCHEDULE_MIN_PAGES


def _should_use_schedule_single_row_kernel(
    *,
    q_rows: int,
    max_pages: int,
) -> bool:
    return q_rows == 1 and _should_use_schedule_kernel(q_rows=q_rows, max_pages=max_pages)


def _should_use_schedule_multi_row_kernel(
    *,
    q_rows: int,
    max_pages: int,
) -> bool:
    return (
        _ENABLE_MULTI_ROW_SCHEDULE
        and
        q_rows > 1
        and q_rows <= _SCHEDULE_MULTI_ROW_MAX_Q_ROWS
        and _should_use_schedule_kernel(q_rows=q_rows, max_pages=max_pages)
    )


class SparseNSAScheduledSingleRowLogitsKernel:
    """Schedule-driven scorer for the long single-row decode case."""

    def __init__(self, parallel_ctas: int, num_heads_static: int):
        self.parallel_ctas = int(parallel_ctas)
        self.num_heads_static = int(num_heads_static)
        self.num_q_head_tiles = _num_q_head_tiles(self.num_heads_static)
        if self.num_q_head_tiles not in (1, 2, 4):
            raise ValueError(
                f"paged logits kernel only supports 1/2/4 head tiles, got {self.num_q_head_tiles}"
            )
        self.padded_q_heads = self.num_q_head_tiles * _PAGED_Q_HEAD_TILE
        self.token_groups = _PAGED_WARPS_PER_CTA // self.num_q_head_tiles
        self.tokens_per_work = _PAGED_TOKENS_PER_GROUP * self.token_groups
        self.page_splits = _PAGE_SIZE // self.tokens_per_work
        self.schedule_pages_per_split = int(PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT)

    def _get_shared_storage_cls(self):
        return _paged_indexer_shared_storage_cls(
            self.padded_q_heads,
            self.tokens_per_work,
            self.num_q_head_tiles,
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_patched_k_tma_desc: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        schedule_metadata: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_paged_index_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            cute.make_layout((_PAGE_SIZE, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)),
            (_PAGE_SIZE, _INDEX_HEAD_DIM),
            1,
        )
        SharedStorage = self._get_shared_storage_cls()
        self.kernel(
            q_bytes,
            weights,
            tma_tensor_k,
            k_tma_desc_ptrs,
            use_patched_k_tma_desc,
            k_scales,
            real_page_table,
            seqlens_per_query,
            schedule_metadata,
            active_width,
            logits_out,
            tma_atom_k,
        ).launch(
            grid=(
                schedule_metadata.shape[0] - 1,
                self.parallel_ctas,
                1,
            ),
            block=[_PAGED_THREADS_PER_CTA, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_tma_tensor: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_patched_k_tma_desc: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        schedule_metadata: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
    ):
        tx, _, _ = cute.arch.thread_idx()
        interval_idx, cta_lane_idx, _ = cute.arch.block_idx()
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()

        width_tokens = Int32(real_page_table.shape[1]) * Int32(_PAGE_SIZE)
        live_width = Int32(active_width[Int32(0)])
        if live_width > width_tokens:
            live_width = width_tokens
        seq_len = Int32(seqlens_per_query[Int32(0)])
        if seq_len > live_width:
            seq_len = live_width
        live_pages = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)

        start_q_idx = Int32(schedule_metadata[interval_idx, Int32(0)])
        start_split_idx = Int32(schedule_metadata[interval_idx, Int32(1)])
        end_q_idx = Int32(schedule_metadata[interval_idx + Int32(1), Int32(0)])
        end_split_idx = Int32(schedule_metadata[interval_idx + Int32(1), Int32(1)])

        interval_page_start = start_split_idx * Int32(self.schedule_pages_per_split)
        interval_page_end = live_pages
        if end_q_idx == Int32(0):
            interval_page_end = end_split_idx * Int32(self.schedule_pages_per_split)
            if interval_page_end > live_pages:
                interval_page_end = live_pages

        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        s_q = storage.q_bytes.get_tensor(
            cute.make_layout((self.padded_q_heads, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1))
        )
        s_w = storage.weights.get_tensor(cute.make_layout((self.padded_q_heads,), stride=(1,)))
        k_page_base_addr = shared_ptr_to_u32(storage.k_page.data_ptr())
        k_page_perm_base_addr = shared_ptr_to_u32(storage.k_page_perm.data_ptr())
        s_k_page_stage = storage.k_page.get_tensor(
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM, 1),
                stride=(_INDEX_HEAD_DIM, 1, _PAGE_SIZE * _INDEX_HEAD_DIM),
            )
        )
        s_scale = storage.scales.get_tensor(cute.make_layout((_PAGE_SIZE,), stride=(1,)))
        s_partial_logits = storage.partial_logits.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        load_k_tma, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.local_tile(k_tma_tensor, (_PAGE_SIZE, _INDEX_HEAD_DIM), (0, 0, None)),
            s_k_page_stage,
        )
        use_external_k_tma_desc = Int32(use_patched_k_tma_desc[Int32(0)])
        k_tma_desc_ptr = Int64(k_tma_desc_ptrs[Int32(0)])

        if (
            start_q_idx == Int32(0)
            and interval_page_start < interval_page_end
            and cta_lane_idx < Int32(self.parallel_ctas)
        ):
            if tx == 0:
                cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
            if warp_idx == Int32(0) and use_external_k_tma_desc == Int32(0):
                cpasync.prefetch_descriptor(tma_atom_k)

            num_heads = Int32(self.num_heads_static)
            q_linear = tx
            total_q_bytes = Int32(self.padded_q_heads * _INDEX_HEAD_DIM)
            while q_linear < total_q_bytes:
                head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
                col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
                s_q[head_idx, col_idx] = (
                    q_bytes[Int32(0), head_idx, col_idx]
                    if head_idx < num_heads
                    else cutlass.Uint8(0)
                )
                q_linear += Int32(_PAGED_THREADS_PER_CTA)

            w_linear = tx
            while w_linear < Int32(self.padded_q_heads):
                s_w[w_linear] = (
                    Float32(weights[Int32(0), w_linear]) if w_linear < num_heads else Float32(0.0)
                )
                w_linear += Int32(_PAGED_THREADS_PER_CTA)
            cute.arch.sync_threads()

            producer_state = pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = pipeline.PipelineStateSimple(1, Int32(0))
            head_tile_slot = warp_idx % Int32(self.num_q_head_tiles)
            token_group = warp_idx // Int32(self.num_q_head_tiles)
            page_col = interval_page_start + cta_lane_idx
            while page_col < interval_page_end:
                page_base = page_col * Int32(_PAGE_SIZE)
                if page_base < seq_len:
                    page_id = Int32(real_page_table[Int32(0), page_col])
                    if page_id >= Int32(0):
                        if warp_idx == Int32(0):
                            _issue_index_k_tma_copy(
                                load_k_tma,
                                producer_state,
                                mbar_ptr_k,
                                Int32(_PAGE_SIZE * _INDEX_HEAD_DIM),
                                page_id,
                                k_page_base_addr,
                                tma_desc_ptr=k_tma_desc_ptr,
                                use_external_tma_desc=use_external_k_tma_desc,
                            )
                        scale_idx = tx
                        while scale_idx < Int32(_PAGE_SIZE):
                            s_scale[scale_idx] = Float32(k_scales[page_id, scale_idx])
                            scale_idx += Int32(_PAGED_THREADS_PER_CTA)
                        cute.arch.mbarrier_wait(
                            mbar_ptr_k + consumer_state.index,
                            phase=consumer_state.phase,
                        )
                        cute.arch.sync_threads()
                        _repack_k_page_to_permuted(k_page_base_addr, k_page_perm_base_addr, tx)
                        cute.arch.sync_threads()

                        valid_slots = seq_len - page_base
                        if valid_slots > Int32(_PAGE_SIZE):
                            valid_slots = Int32(_PAGE_SIZE)
                        split_idx = Int32(0)
                        while split_idx < Int32(self.page_splits):
                            token_base = (
                                split_idx * Int32(self.tokens_per_work)
                                + token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                            )
                            zero_idx = tx
                            while zero_idx < Int32(self.tokens_per_work * self.num_q_head_tiles):
                                token_idx = zero_idx // Int32(self.num_q_head_tiles)
                                head_tile_idx = zero_idx - token_idx * Int32(self.num_q_head_tiles)
                                s_partial_logits[token_idx, head_tile_idx] = Float32(0.0)
                                zero_idx += Int32(_PAGED_THREADS_PER_CTA)
                            cute.arch.sync_threads()
                            if token_base < valid_slots:
                                head_tile_base = head_tile_slot * Int32(_PAGED_Q_HEAD_TILE)
                                _compute_mxfp8_tile_partials(
                                    s_q,
                                    s_w,
                                    num_heads,
                                    k_page_perm_base_addr,
                                    token_base,
                                    head_tile_base,
                                    lane,
                                    s_partial_logits,
                                    token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                    head_tile_slot,
                                )
                            cute.arch.sync_threads()
                            if head_tile_slot == Int32(0) and lane < Int32(_PAGED_TOKENS_PER_GROUP):
                                slot_idx = token_base + lane
                                if slot_idx < valid_slots:
                                    logit = Float32(0.0)
                                    head_tile_idx = Int32(0)
                                    partial_row = token_group * Int32(_PAGED_TOKENS_PER_GROUP) + lane
                                    while head_tile_idx < Int32(self.num_q_head_tiles):
                                        logit = Float32(
                                            logit + s_partial_logits[partial_row, head_tile_idx]
                                        )
                                        head_tile_idx += Int32(1)
                                    logits_out[Int32(0), page_base + slot_idx] = Float32(
                                        logit * s_scale[slot_idx]
                                    )
                            cute.arch.sync_threads()
                            split_idx += Int32(1)
                        producer_state.advance()
                        consumer_state.advance()
                        cute.arch.sync_threads()

                page_col += Int32(self.parallel_ctas)


class SparseNSAScheduledMultiRowLogitsKernel:
    """Schedule-driven scorer for long multi-row decode."""

    def __init__(self, parallel_ctas: int, num_heads_static: int):
        self.parallel_ctas = int(parallel_ctas)
        self.num_heads_static = int(num_heads_static)
        self.num_q_head_tiles = _num_q_head_tiles(self.num_heads_static)
        if self.num_q_head_tiles not in (1, 2, 4):
            raise ValueError(
                f"paged logits kernel only supports 1/2/4 head tiles, got {self.num_q_head_tiles}"
            )
        self.padded_q_heads = self.num_q_head_tiles * _PAGED_Q_HEAD_TILE
        self.token_groups = _PAGED_WARPS_PER_CTA // self.num_q_head_tiles
        self.tokens_per_work = _PAGED_TOKENS_PER_GROUP * self.token_groups
        self.page_splits = _PAGE_SIZE // self.tokens_per_work
        self.schedule_pages_per_split = int(PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT)

    def _get_shared_storage_cls(self):
        return _paged_indexer_shared_storage_cls(
            self.padded_q_heads,
            self.tokens_per_work,
            self.num_q_head_tiles,
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_patched_k_tma_desc: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        schedule_metadata: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_paged_index_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            cute.make_layout((_PAGE_SIZE, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)),
            (_PAGE_SIZE, _INDEX_HEAD_DIM),
            1,
        )
        SharedStorage = self._get_shared_storage_cls()
        self.kernel(
            q_bytes,
            weights,
            tma_tensor_k,
            k_tma_desc_ptrs,
            use_patched_k_tma_desc,
            k_scales,
            real_page_table,
            seqlens_per_query,
            schedule_metadata,
            active_width,
            logits_out,
            tma_atom_k,
        ).launch(
            grid=(
                schedule_metadata.shape[0] - 1,
                self.parallel_ctas,
                1,
            ),
            block=[_PAGED_THREADS_PER_CTA, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_tma_tensor: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_patched_k_tma_desc: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        schedule_metadata: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
    ):
        tx, _, _ = cute.arch.thread_idx()
        interval_idx, cta_lane_idx, _ = cute.arch.block_idx()
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()

        width_tokens = Int32(real_page_table.shape[1]) * Int32(_PAGE_SIZE)
        live_width = Int32(active_width[Int32(0)])
        if live_width > width_tokens:
            live_width = width_tokens
        q_rows = Int32(seqlens_per_query.shape[0])
        total_schedule_intervals = Int32(schedule_metadata.shape[0] - 1)
        start_q_idx = Int32(schedule_metadata[interval_idx, Int32(0)])
        start_split_idx = Int32(schedule_metadata[interval_idx, Int32(1)])
        end_q_idx = Int32(schedule_metadata[interval_idx + Int32(1), Int32(0)])
        end_split_idx = Int32(schedule_metadata[interval_idx + Int32(1), Int32(1)])

        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        s_q = storage.q_bytes.get_tensor(
            cute.make_layout((self.padded_q_heads, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1))
        )
        s_w = storage.weights.get_tensor(cute.make_layout((self.padded_q_heads,), stride=(1,)))
        k_page_base_addr = shared_ptr_to_u32(storage.k_page.data_ptr())
        k_page_perm_base_addr = shared_ptr_to_u32(storage.k_page_perm.data_ptr())
        s_k_page_stage = storage.k_page.get_tensor(
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM, 1),
                stride=(_INDEX_HEAD_DIM, 1, _PAGE_SIZE * _INDEX_HEAD_DIM),
            )
        )
        s_scale = storage.scales.get_tensor(cute.make_layout((_PAGE_SIZE,), stride=(1,)))
        s_partial_logits = storage.partial_logits.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        load_k_tma, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.local_tile(k_tma_tensor, (_PAGE_SIZE, _INDEX_HEAD_DIM), (0, 0, None)),
            s_k_page_stage,
        )
        use_external_k_tma_desc = Int32(use_patched_k_tma_desc[Int32(0)])
        k_tma_desc_ptr = Int64(k_tma_desc_ptrs[Int32(0)])

        if interval_idx < total_schedule_intervals and cta_lane_idx < Int32(self.parallel_ctas):
            if tx == 0:
                cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
            if warp_idx == Int32(0) and use_external_k_tma_desc == Int32(0):
                cpasync.prefetch_descriptor(tma_atom_k)

            num_heads = Int32(self.num_heads_static)
            producer_state = pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = pipeline.PipelineStateSimple(1, Int32(0))
            head_tile_slot = warp_idx % Int32(self.num_q_head_tiles)
            token_group = warp_idx // Int32(self.num_q_head_tiles)
            current_q_idx = start_q_idx
            current_split_idx = start_split_idx

            while current_q_idx < q_rows and (
                current_q_idx < end_q_idx
                or (current_q_idx == end_q_idx and end_split_idx > Int32(0))
            ):
                seq_len = Int32(seqlens_per_query[current_q_idx])
                if seq_len > live_width:
                    seq_len = live_width
                live_pages = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)
                row_total_splits = (
                    live_pages + Int32(self.schedule_pages_per_split - 1)
                ) // Int32(self.schedule_pages_per_split)
                row_split_start = current_split_idx if current_q_idx == start_q_idx else Int32(0)
                row_split_end = row_total_splits
                if current_q_idx == end_q_idx:
                    row_split_end = end_split_idx
                    if row_split_end > row_total_splits:
                        row_split_end = row_total_splits

                if row_split_start < row_split_end:
                    q_linear = tx
                    total_q_bytes = Int32(self.padded_q_heads * _INDEX_HEAD_DIM)
                    while q_linear < total_q_bytes:
                        head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
                        col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
                        s_q[head_idx, col_idx] = (
                            q_bytes[current_q_idx, head_idx, col_idx]
                            if head_idx < num_heads
                            else cutlass.Uint8(0)
                        )
                        q_linear += Int32(_PAGED_THREADS_PER_CTA)

                    w_linear = tx
                    while w_linear < Int32(self.padded_q_heads):
                        s_w[w_linear] = (
                            Float32(weights[current_q_idx, w_linear])
                            if w_linear < num_heads
                            else Float32(0.0)
                        )
                        w_linear += Int32(_PAGED_THREADS_PER_CTA)
                    cute.arch.sync_threads()

                    row_page_end = row_split_end * Int32(self.schedule_pages_per_split)
                    if row_page_end > live_pages:
                        row_page_end = live_pages
                    page_col = row_split_start * Int32(self.schedule_pages_per_split) + cta_lane_idx
                    while page_col < row_page_end:
                        page_base = page_col * Int32(_PAGE_SIZE)
                        if page_base < seq_len:
                            page_id = Int32(real_page_table[current_q_idx, page_col])
                            if page_id >= Int32(0):
                                if warp_idx == Int32(0):
                                    _issue_index_k_tma_copy(
                                        load_k_tma,
                                        producer_state,
                                        mbar_ptr_k,
                                        Int32(_PAGE_SIZE * _INDEX_HEAD_DIM),
                                        page_id,
                                        k_page_base_addr,
                                        tma_desc_ptr=k_tma_desc_ptr,
                                        use_external_tma_desc=use_external_k_tma_desc,
                                    )
                                scale_idx = tx
                                while scale_idx < Int32(_PAGE_SIZE):
                                    s_scale[scale_idx] = Float32(k_scales[page_id, scale_idx])
                                    scale_idx += Int32(_PAGED_THREADS_PER_CTA)
                                cute.arch.mbarrier_wait(
                                    mbar_ptr_k + consumer_state.index,
                                    phase=consumer_state.phase,
                                )
                                cute.arch.sync_threads()
                                _repack_k_page_to_permuted(k_page_base_addr, k_page_perm_base_addr, tx)
                                cute.arch.sync_threads()

                                valid_slots = seq_len - page_base
                                if valid_slots > Int32(_PAGE_SIZE):
                                    valid_slots = Int32(_PAGE_SIZE)
                                split_idx = Int32(0)
                                while split_idx < Int32(self.page_splits):
                                    token_base = (
                                        split_idx * Int32(self.tokens_per_work)
                                        + token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                                    )
                                    zero_idx = tx
                                    while zero_idx < Int32(
                                        self.tokens_per_work * self.num_q_head_tiles
                                    ):
                                        token_idx = zero_idx // Int32(self.num_q_head_tiles)
                                        head_tile_idx = (
                                            zero_idx - token_idx * Int32(self.num_q_head_tiles)
                                        )
                                        s_partial_logits[token_idx, head_tile_idx] = Float32(0.0)
                                        zero_idx += Int32(_PAGED_THREADS_PER_CTA)
                                    cute.arch.sync_threads()
                                    if token_base < valid_slots:
                                        head_tile_base = head_tile_slot * Int32(_PAGED_Q_HEAD_TILE)
                                        _compute_mxfp8_tile_partials(
                                            s_q,
                                            s_w,
                                            num_heads,
                                            k_page_perm_base_addr,
                                            token_base,
                                            head_tile_base,
                                            lane,
                                            s_partial_logits,
                                            token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                            head_tile_slot,
                                        )
                                    cute.arch.sync_threads()
                                    if (
                                        head_tile_slot == Int32(0)
                                        and lane < Int32(_PAGED_TOKENS_PER_GROUP)
                                    ):
                                        slot_idx = token_base + lane
                                        if slot_idx < valid_slots:
                                            logit = Float32(0.0)
                                            head_tile_idx = Int32(0)
                                            partial_row = (
                                                token_group * Int32(_PAGED_TOKENS_PER_GROUP) + lane
                                            )
                                            while head_tile_idx < Int32(self.num_q_head_tiles):
                                                logit = Float32(
                                                    logit
                                                    + s_partial_logits[partial_row, head_tile_idx]
                                                )
                                                head_tile_idx += Int32(1)
                                            logits_out[current_q_idx, page_base + slot_idx] = Float32(
                                                logit * s_scale[slot_idx]
                                            )
                                    cute.arch.sync_threads()
                                    split_idx += Int32(1)
                                producer_state.advance()
                                consumer_state.advance()
                                cute.arch.sync_threads()

                        page_col += Int32(self.parallel_ctas)

                current_q_idx += Int32(1)
                current_split_idx = Int32(0)


@lru_cache(maxsize=32)
def _build_sparse_nsa_paged_kernel(
    persistent_ctas: int,
    num_heads_static: int,
) -> SparseNSAPagedLogitsKernel:
    return SparseNSAPagedLogitsKernel(persistent_ctas, num_heads_static)


@lru_cache(maxsize=32)
def _build_sparse_nsa_paged_tiled_kernel(
    persistent_ctas: int,
    num_heads_static: int,
    tile_block_q: int,
    tile_block_k: int,
) -> SparseNSAPagedLogitsKernel:
    return SparseNSAPagedLogitsKernel(
        persistent_ctas,
        num_heads_static,
        tiled_output=True,
        tile_block_q=tile_block_q,
        tile_block_k=tile_block_k,
    )


@lru_cache(maxsize=32)
def _build_sparse_nsa_paged_windowed_tiled_kernel(
    persistent_ctas: int,
    num_heads_static: int,
    tile_block_q: int,
    tile_block_k: int,
) -> SparseNSAPagedWindowedTiledLogitsKernel:
    return SparseNSAPagedWindowedTiledLogitsKernel(
        persistent_ctas,
        num_heads_static,
        tiled_output=True,
        tile_block_q=tile_block_q,
        tile_block_k=tile_block_k,
    )


@lru_cache(maxsize=16)
def _build_sparse_nsa_schedule_single_row_kernel(
    parallel_ctas: int,
    num_heads_static: int,
) -> SparseNSAScheduledSingleRowLogitsKernel:
    return SparseNSAScheduledSingleRowLogitsKernel(parallel_ctas, num_heads_static)


@lru_cache(maxsize=16)
def _build_sparse_nsa_schedule_multi_row_kernel(
    parallel_ctas: int,
    num_heads_static: int,
) -> SparseNSAScheduledMultiRowLogitsKernel:
    return SparseNSAScheduledMultiRowLogitsKernel(parallel_ctas, num_heads_static)


def _split_index_k_cache_runtime_views(index_k_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if index_k_cache.stride(-1) != 1:
        raise ValueError(
            f"index_k_cache must have a contiguous last dimension, got stride={index_k_cache.stride()}"
        )
    num_pages = index_k_cache.shape[0]
    data_bytes = _PAGE_SIZE * _INDEX_HEAD_DIM
    k_quant_bytes = index_k_cache[:, :data_bytes].view(num_pages, _PAGE_SIZE, _INDEX_HEAD_DIM)
    k_scales = (
        index_k_cache[:, data_bytes : data_bytes + _PAGE_SIZE * _SCALE_BYTES]
        .view(num_pages, _PAGE_SIZE, _SCALE_BYTES)
        .view(torch.float32)
        .squeeze(-1)
    )
    return k_quant_bytes, k_scales


def clear_sparse_nsa_indexer_kernel_cache() -> None:
    _build_sparse_nsa_paged_kernel.cache_clear()
    _build_sparse_nsa_paged_tiled_kernel.cache_clear()
    _build_sparse_nsa_schedule_single_row_kernel.cache_clear()
    _build_sparse_nsa_schedule_multi_row_kernel.cache_clear()
    cache = getattr(_get_cached_paged_index_k_tma_descriptor, "_cache", None)
    if cache is not None:
        cache.clear()


def supports_sparse_nsa_paged_logits_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    page_size: int,
) -> bool:
    if page_size != _PAGE_SIZE:
        return False
    if q_fp8.device.type != "cuda":
        return False
    if not (
        weights.device
        == index_k_cache.device
        == real_page_table.device
        == seqlens_per_query.device
        == q_fp8.device
    ):
        return False
    if q_fp8.ndim != 3 or q_fp8.shape[2] != _INDEX_HEAD_DIM:
        return False
    if q_fp8.shape[1] > _MAX_SUPPORTED_Q_HEADS:
        return False
    if _num_q_head_tiles(q_fp8.shape[1]) not in (1, 2, 4):
        return False
    if weights.ndim != 2 or weights.shape != q_fp8.shape[:2]:
        return False
    if real_page_table.ndim != 2:
        return False
    if seqlens_per_query.ndim != 1 or seqlens_per_query.shape[0] != q_fp8.shape[0]:
        return False
    if index_k_cache.ndim != 2 or index_k_cache.shape[1] != _PAGE_SIZE * (_INDEX_HEAD_DIM + _SCALE_BYTES):
        return False
    if q_fp8.dtype != torch.float8_e4m3fn:
        return False
    if weights.dtype != torch.float32:
        return False
    if index_k_cache.dtype != torch.uint8:
        return False
    if index_k_cache.stride(-1) != 1:
        return False
    if (
        real_page_table.dtype != torch.int32
        or seqlens_per_query.dtype != torch.int32
    ):
        return False
    return True


def run_sparse_nsa_paged_logits_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    active_width: torch.Tensor | None = None,
    page_size: int = _PAGE_SIZE,
    contract_phantoms: dict[str, torch.Tensor] | None = None,
    workspace=None,
    preinitialize_invalid_logits: bool = True,
) -> torch.Tensor:
    if not supports_sparse_nsa_paged_logits_kernel(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        page_size=page_size,
    ):
        raise ValueError(
            "sparse NSA paged logits kernel only supports the exact CUDA page_size=64 FP8 contract"
        )

    rows = q_fp8.shape[0]
    width_tokens = real_page_table.shape[1] * page_size
    if rows == 0 or width_tokens == 0:
        return torch.empty((rows, width_tokens), dtype=torch.float32, device=q_fp8.device)
    if active_width is None:
        active_width = torch.tensor([width_tokens], dtype=torch.int32, device=q_fp8.device)
    if active_width.shape != (1,):
        raise ValueError(f"active_width must have shape (1,), got {tuple(active_width.shape)}")
    if active_width.dtype != torch.int32:
        raise ValueError(f"active_width must have dtype torch.int32, got {active_width.dtype}")
    if active_width.device != q_fp8.device:
        raise ValueError(
            f"active_width device {active_width.device} does not match q_fp8 device {q_fp8.device}"
        )

    k_quant_bytes, k_scales = _split_index_k_cache_runtime_views(index_k_cache)
    use_patched_k_tma_desc = _needs_paged_index_k_tma_descriptor_workaround(
        index_k_cache,
        k_quant_bytes,
    )
    device_index = q_fp8.device.index or 0
    if use_patched_k_tma_desc:
        _, k_tma_desc_ptrs = _get_cached_paged_index_k_tma_descriptor(k_quant_bytes)
    else:
        k_tma_desc_ptrs = _dummy_paged_index_k_tma_desc_ptrs(device_index)
    use_patched_k_tma_desc_tensor = _cached_int32_scalar(
        int(use_patched_k_tma_desc),
        device_index,
    )
    if workspace is not None:
        staged = workspace.stage_nsa_indexer_paged_decode(
            q_fp8=q_fp8,
            weights=weights,
            real_page_table=real_page_table,
            seqlens_per_query=seqlens_per_query,
            active_width=active_width,
            schedule_metadata=schedule_metadata,
            width_tokens=width_tokens,
            preinitialize_invalid_logits=preinitialize_invalid_logits,
        )
        q_bytes = staged["q_bytes"]
        weights_kernel = staged["weights"]
        real_page_table_kernel = staged["real_page_table"]
        seqlens_per_query_kernel = staged["seqlens_per_query"]
        active_width_kernel = staged["active_width"]
        schedule_metadata_kernel = staged["schedule_metadata"]
        logits = staged["logits"]
        logits_view = staged["logits_view"]
        if contract_phantoms is None:
            contract_phantoms = workspace.get_paged_indexer_contract_phantoms()
    else:
        q_bytes = q_fp8.contiguous().view(torch.uint8)
        weights_kernel = weights.contiguous()
        real_page_table_kernel = real_page_table.contiguous()
        seqlens_per_query_kernel = seqlens_per_query.contiguous()
        active_width_kernel = active_width.contiguous()
        schedule_metadata_kernel = None
        if preinitialize_invalid_logits:
            logits = torch.full(
                (rows, width_tokens),
                float("-inf"),
                dtype=torch.float32,
                device=q_fp8.device,
            )
        else:
            logits = torch.empty((rows, width_tokens), dtype=torch.float32, device=q_fp8.device)
        logits_view = logits
    _cp = contract_phantoms or {}
    common_args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8),
        _to_kernel_tensor(weights_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
        _to_kernel_tensor(k_tma_desc_ptrs, cutlass.Int64, assumed_align=8),
        _to_kernel_tensor(use_patched_k_tma_desc_tensor, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(real_page_table_kernel, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(seqlens_per_query_kernel, cutlass.Int32, assumed_align=4),
    )
    common_cache_key = (
        q_fp8.shape[1],
        _tensor_meta_key(_cp.get("q_bytes", q_bytes)),
        _tensor_meta_key(_cp.get("weights", weights_kernel)),
        _tensor_meta_key(k_quant_bytes),
        _tensor_meta_key(k_tma_desc_ptrs),
        _tensor_meta_key(use_patched_k_tma_desc_tensor),
        _tensor_meta_key(k_scales),
        _tensor_meta_key(_cp.get("real_page_table", real_page_table_kernel)),
        _tensor_meta_key(_cp.get("seqlens_per_query", seqlens_per_query_kernel)),
        _tensor_meta_key(active_width_kernel),
        _tensor_meta_key(_cp.get("logits", logits)),
    )
    max_pages = int(real_page_table.shape[1])
    if schedule_metadata is not None and schedule_metadata_kernel is None:
        if workspace is not None and not schedule_metadata.is_contiguous():
            raise ValueError(
                "workspace-backed paged decode requires contiguous schedule_metadata"
            )
        schedule_metadata_kernel = (
            schedule_metadata
            if schedule_metadata.is_contiguous()
            else schedule_metadata.contiguous()
        )
    if _should_use_schedule_single_row_kernel(q_rows=rows, max_pages=max_pages):
        if schedule_metadata is None:
            raise ValueError("schedule_metadata is required for the scheduled single-row decode path")
        kernel = _build_sparse_nsa_schedule_single_row_kernel(
            _SCHEDULE_SINGLE_ROW_PARALLEL_CTAS,
            q_fp8.shape[1],
        )
        args = (
            *common_args,
            _to_kernel_tensor(schedule_metadata_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "schedule_single_row",
            _SCHEDULE_SINGLE_ROW_PARALLEL_CTAS,
            *common_cache_key,
            _tensor_meta_key(schedule_metadata_kernel),
        )
    elif _should_use_schedule_multi_row_kernel(q_rows=rows, max_pages=max_pages):
        if schedule_metadata is None:
            raise ValueError("schedule_metadata is required for the scheduled multi-row decode path")
        kernel = _build_sparse_nsa_schedule_multi_row_kernel(
            _SCHEDULE_MULTI_ROW_PARALLEL_CTAS,
            q_fp8.shape[1],
        )
        args = (
            *common_args,
            _to_kernel_tensor(schedule_metadata_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "schedule_multi_row",
            _SCHEDULE_MULTI_ROW_PARALLEL_CTAS,
            *common_cache_key,
            _tensor_meta_key(schedule_metadata_kernel),
        )
    else:
        persistent_ctas = _resolve_sparse_nsa_persistent_ctas(
            device_index=device_index,
            q_rows=rows,
        )
        kernel = _build_sparse_nsa_paged_kernel(persistent_ctas, q_fp8.shape[1])
        args = (
            *common_args,
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "persistent",
            persistent_ctas,
            *common_cache_key,
        )
    _run_cached_host_launcher(kernel, cache_key, args)
    return logits_view


def run_sparse_nsa_paged_tiled_logits_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    tile_logits: torch.Tensor,
    page_size: int = _PAGE_SIZE,
    tile_block_q: int = _PAGED_TILED_BLOCK_Q,
    tile_block_k: int = _PAGED_TILED_BLOCK_K,
    contract_phantoms: dict[str, torch.Tensor] | None = None,
    workspace=None,
    preinitialize_tile_logits: bool = True,
) -> torch.Tensor:
    return _run_sparse_nsa_paged_tiled_logits_kernel_common(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        page_size=page_size,
        tile_block_q=tile_block_q,
        tile_block_k=tile_block_k,
        contract_phantoms=contract_phantoms,
        workspace=workspace,
        preinitialize_tile_logits=preinitialize_tile_logits,
        source_page_offset=0,
        output_width_tokens=None,
        windowed=False,
    )


def run_sparse_nsa_paged_windowed_tiled_logits_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    tile_logits: torch.Tensor,
    source_page_offset: int,
    output_width_tokens: int,
    page_size: int = _PAGE_SIZE,
    tile_block_q: int = _PAGED_TILED_BLOCK_Q,
    tile_block_k: int = _PAGED_TILED_BLOCK_K,
    contract_phantoms: dict[str, torch.Tensor] | None = None,
    workspace=None,
    preinitialize_tile_logits: bool = True,
    stage_runtime_metadata: bool = True,
) -> torch.Tensor:
    return _run_sparse_nsa_paged_tiled_logits_kernel_common(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        page_size=page_size,
        tile_block_q=tile_block_q,
        tile_block_k=tile_block_k,
        contract_phantoms=contract_phantoms,
        workspace=workspace,
        preinitialize_tile_logits=preinitialize_tile_logits,
        source_page_offset=source_page_offset,
        output_width_tokens=output_width_tokens,
        windowed=True,
        stage_runtime_metadata=stage_runtime_metadata,
    )


def _run_sparse_nsa_paged_tiled_logits_kernel_common(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    tile_logits: torch.Tensor,
    page_size: int,
    tile_block_q: int,
    tile_block_k: int,
    contract_phantoms: dict[str, torch.Tensor] | None,
    workspace,
    preinitialize_tile_logits: bool,
    source_page_offset: int = 0,
    output_width_tokens: int | None = None,
    windowed: bool,
    stage_runtime_metadata: bool = True,
) -> torch.Tensor:
    if page_size != _PAGE_SIZE:
        raise ValueError(f"paged tiled logits kernel requires page_size={_PAGE_SIZE}, got {page_size}")
    if int(tile_block_k) != _PAGED_TILED_BLOCK_K:
        raise ValueError(
            f"paged tiled logits kernel requires tile_block_k={_PAGED_TILED_BLOCK_K}, got {tile_block_k}"
        )
    if not supports_sparse_nsa_paged_logits_kernel(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        page_size=page_size,
    ):
        raise ValueError(
            "sparse NSA paged tiled logits kernel only supports the exact CUDA page_size=64 FP8 contract"
        )
    if active_width.shape != (1,):
        raise ValueError(f"active_width must have shape (1,), got {tuple(active_width.shape)}")
    if active_width.dtype != torch.int32:
        raise ValueError(f"active_width must have dtype torch.int32, got {active_width.dtype}")
    if active_width.device != q_fp8.device:
        raise ValueError(
            f"active_width device {active_width.device} does not match q_fp8 device {q_fp8.device}"
        )
    if tile_logits.dtype != torch.float32 or tile_logits.device != q_fp8.device:
        raise ValueError("tile_logits must be a CUDA torch.float32 tensor on the q_fp8 device")
    if not tile_logits.is_contiguous():
        raise ValueError("tile_logits must be contiguous")

    rows = int(q_fp8.shape[0])
    source_width_tokens = int(real_page_table.shape[1]) * int(page_size)
    source_page_offset = int(source_page_offset)
    if source_page_offset < 0:
        raise ValueError(f"source_page_offset must be non-negative, got {source_page_offset}")
    if source_page_offset > int(real_page_table.shape[1]):
        raise ValueError(
            "source_page_offset exceeds real_page_table width: "
            f"offset={source_page_offset}, width={int(real_page_table.shape[1])}"
        )
    if windowed:
        if output_width_tokens is None:
            raise ValueError("windowed paged tiled logits require output_width_tokens")
        width_tokens = int(output_width_tokens)
    else:
        width_tokens = source_width_tokens
    if rows == 0 or width_tokens == 0:
        return tile_logits
    if width_tokens < 0:
        raise ValueError(f"output_width_tokens must be non-negative, got {width_tokens}")
    if width_tokens % int(tile_block_k) != 0:
        raise ValueError(
            f"paged tiled logits width {width_tokens} must be divisible by tile_block_k={tile_block_k}"
        )
    num_q_tiles = (rows + int(tile_block_q) - 1) // int(tile_block_q)
    num_k_tiles = width_tokens // int(tile_block_k)
    required_elements = num_q_tiles * num_k_tiles * int(tile_block_q) * int(tile_block_k)
    if int(tile_logits.numel()) < required_elements:
        raise ValueError(
            f"tile_logits has {int(tile_logits.numel())} elements, expected at least {required_elements}"
        )

    k_quant_bytes, k_scales = _split_index_k_cache_runtime_views(index_k_cache)
    use_patched_k_tma_desc = _needs_paged_index_k_tma_descriptor_workaround(
        index_k_cache,
        k_quant_bytes,
    )
    device_index = q_fp8.device.index or 0
    if use_patched_k_tma_desc:
        _, k_tma_desc_ptrs = _get_cached_paged_index_k_tma_descriptor(k_quant_bytes)
    else:
        k_tma_desc_ptrs = _dummy_paged_index_k_tma_desc_ptrs(device_index)
    use_patched_k_tma_desc_tensor = _cached_int32_scalar(
        int(use_patched_k_tma_desc),
        device_index,
    )
    if workspace is not None and stage_runtime_metadata:
        staged = workspace.stage_nsa_indexer_paged_tiled_decode(
            q_fp8=q_fp8,
            weights=weights,
            real_page_table=real_page_table,
            seqlens_per_query=seqlens_per_query,
            active_width=active_width,
            width_tokens=width_tokens,
            tile_logits=tile_logits,
            tile_block_q=int(tile_block_q),
            tile_block_k=int(tile_block_k),
            preinitialize_tile_logits=bool(preinitialize_tile_logits),
        )
        q_bytes = staged["q_bytes"]
        weights_kernel = staged["weights"]
        real_page_table_kernel = staged["real_page_table"]
        seqlens_per_query_kernel = staged["seqlens_per_query"]
        active_width_kernel = staged["active_width"]
        logits = staged["tile_logits"]
        logits_view = staged["tile_logits_view"]
        if contract_phantoms is None:
            contract_phantoms = workspace.get_paged_indexer_contract_phantoms()
    else:
        if not q_fp8.is_contiguous():
            raise ValueError("paged tiled logits requires contiguous q_fp8")
        if not weights.is_contiguous():
            raise ValueError("paged tiled logits requires contiguous weights")
        if not real_page_table.is_contiguous():
            raise ValueError("paged tiled logits requires contiguous real_page_table")
        if not seqlens_per_query.is_contiguous():
            raise ValueError("paged tiled logits requires contiguous seqlens_per_query")
        if not active_width.is_contiguous():
            raise ValueError("paged tiled logits requires contiguous active_width")
        q_bytes = q_fp8.view(torch.uint8)
        weights_kernel = weights
        real_page_table_kernel = real_page_table
        seqlens_per_query_kernel = seqlens_per_query
        active_width_kernel = active_width
        logits = tile_logits
        logits_view = tile_logits[:required_elements]
        if preinitialize_tile_logits:
            logits_view.fill_(float("-inf"))
        if workspace is not None and contract_phantoms is None:
            contract_phantoms = workspace.get_paged_indexer_contract_phantoms()

    _cp = contract_phantoms or {}
    common_args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8),
        _to_kernel_tensor(weights_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
        _to_kernel_tensor(k_tma_desc_ptrs, cutlass.Int64, assumed_align=8),
        _to_kernel_tensor(use_patched_k_tma_desc_tensor, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(real_page_table_kernel, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(seqlens_per_query_kernel, cutlass.Int32, assumed_align=4),
    )
    common_cache_key = (
        q_fp8.shape[1],
        _tensor_meta_key(_cp.get("q_bytes", q_bytes)),
        _tensor_meta_key(_cp.get("weights", weights_kernel)),
        _tensor_meta_key(k_quant_bytes),
        _tensor_meta_key(k_tma_desc_ptrs),
        _tensor_meta_key(use_patched_k_tma_desc_tensor),
        _tensor_meta_key(k_scales),
        _tensor_meta_key(_cp.get("real_page_table", real_page_table_kernel)),
        _tensor_meta_key(_cp.get("seqlens_per_query", seqlens_per_query_kernel)),
        _tensor_meta_key(active_width_kernel),
        _tensor_meta_key(_cp.get("tile_logits", _cp.get("logits", logits))),
    )
    workspace_persistent_ctas = (
        workspace.get_paged_indexer_persistent_ctas()
        if workspace is not None and hasattr(workspace, "get_paged_indexer_persistent_ctas")
        else None
    )
    persistent_ctas = (
        int(workspace_persistent_ctas)
        if workspace_persistent_ctas is not None
        else _resolve_sparse_nsa_persistent_ctas(
            device_index=device_index,
            q_rows=rows,
        )
    )
    if windowed:
        kernel = _build_sparse_nsa_paged_windowed_tiled_kernel(
            persistent_ctas,
            q_fp8.shape[1],
            int(tile_block_q),
            int(tile_block_k),
        )
        args = (
            *common_args,
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            Int32(source_page_offset),
            Int32(width_tokens),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "persistent_windowed_tiled",
            persistent_ctas,
            int(tile_block_q),
            int(tile_block_k),
            *common_cache_key,
        )
    else:
        kernel = _build_sparse_nsa_paged_tiled_kernel(
            persistent_ctas,
            q_fp8.shape[1],
            int(tile_block_q),
            int(tile_block_k),
        )
        args = (
            *common_args,
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "persistent_tiled",
            persistent_ctas,
            int(tile_block_q),
            int(tile_block_k),
            *common_cache_key,
        )
    _run_cached_host_launcher(kernel, cache_key, args)
    logits_view._b12x_num_q_tiles = num_q_tiles
    logits_view._b12x_num_k_tiles = num_k_tiles
    logits_view._b12x_block_q = int(tile_block_q)
    logits_view._b12x_block_k = int(tile_block_k)
    return logits_view
