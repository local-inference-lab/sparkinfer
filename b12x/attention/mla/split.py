"""Token-split sparse MLA decode kernels and runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cuda.bindings.driver as cuda
import os
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from b12x.attention._cute import ops as attention_ops
from b12x.attention.workspace import (
    SparseMLASplitDecodeConfig,
    _SPLIT_MAX_CHUNKS,
    _ceil_div,
    default_sparse_mla_split_decode_config_for_width,
    forced_sparse_mla_split_decode_config_for_width,  # noqa: F401
)
from b12x.cute.compiler import DimKey, KernelCompileSpec, launch as b12x_launch
from b12x.cute.fp4 import shared_ptr_to_u32
from b12x.cute.utils import current_cuda_stream, make_ptr

from .compressed_reference import compressed_mla_page_nbytes
from .kernel import (
    _COMPRESSED_MLA_HEAD_DIM,
    _MLA_GROUP_SIZE,
    _MLA_HEADS_PER_TILE,
    _MLA_KV_STAGE_BYTES,
    _MLA_Q_GROUP_STAGE_BYTES,
    _MLA_SCALE_GROUPS,
    _MLA_SHARED_SCALE_STAGE_ELEMS,
    _MLA_TOKEN_TILE,
    _MLA_WARP_THREADS,
    _extract_packed_kv_runtime_views,
    _exp2_approx_ftz_f32,
    _clamp_active_token_count,
    _run_one_pass_compressed_mla_tile,
    _run_one_pass_sparse_mla_tile,
    _run_single_tile_compressed_mla_tile,
    _tensor_meta_key,
    _to_kernel_tensor,
    _torch_to_cutlass_dtype,
    _view_last_dim_as_u32,
    _workspace_contract_kv_tensors,
)
from .traits import SparseMLATraits, select_sparse_mla_traits


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _raise_binding_extras(api_name: str, extras: list[str]) -> None:
    raise ValueError(
        f"{api_name} binding owns runtime tensors, scratch, and kernel options; "
        f"do not also pass {', '.join(extras)}"
    )


def _require_bound_arg(value, *, api_name: str, name: str):
    if value is None:
        raise TypeError(f"{api_name} requires {name} or binding")
    return value


@dataclass(frozen=True, kw_only=True)
class SparseMLASplitDecodeForwardBinding:
    q_all: torch.Tensor
    kv_cache: torch.Tensor
    page_table_1: torch.Tensor
    active_token_counts: torch.Tensor
    sm_scale: torch.Tensor
    kv_chunk_size_ptr: torch.Tensor
    num_chunks_ptr: torch.Tensor
    tmp_output: torch.Tensor
    tmp_lse: torch.Tensor
    launch_num_chunks: int
    scratch: object | None = None
    identity_page_table: bool = False

    def run(self) -> None:
        run_sparse_mla_split_decode_forward(binding=self)


@dataclass(frozen=True, kw_only=True)
class CompressedMLASplitDecodeForwardBinding:
    q_all: torch.Tensor
    swa_k_cache: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    indexed_k_cache: torch.Tensor
    indexed_indices: torch.Tensor
    indexed_lengths: torch.Tensor
    indexed_page_table: torch.Tensor
    sm_scale: torch.Tensor
    kv_chunk_size_ptr: torch.Tensor
    num_chunks_ptr: torch.Tensor
    tmp_output: torch.Tensor
    tmp_lse: torch.Tensor
    launch_num_chunks: int
    swa_page_size: int
    swa_page_nbytes: int
    indexed_page_size: int
    indexed_page_nbytes: int
    has_indexed: bool
    map_indexed_page_table: bool
    scratch: object | None = None
    direct_output: bool = False
    single_tile_chunks: bool = False
    attn_sink: torch.Tensor | None = None
    direct_sink_output: bool = False

    def run(self) -> None:
        run_compressed_mla_split_decode_forward(binding=self)


@dataclass(frozen=True, kw_only=True)
class SparseMLASplitDecodeMergeBinding:
    tmp_output: torch.Tensor
    tmp_lse: torch.Tensor
    num_chunks_ptr: torch.Tensor
    output: torch.Tensor
    attn_sink: torch.Tensor | None = None
    scratch: object | None = None

    def run(self) -> None:
        run_sparse_mla_split_decode_merge(binding=self)


@dataclass(frozen=True, kw_only=True)
class SparseMLASplitDecodeBinding:
    q_all: torch.Tensor
    kv_cache: torch.Tensor
    page_table_1: torch.Tensor
    active_token_counts: torch.Tensor
    sm_scale: torch.Tensor
    kv_chunk_size_ptr: torch.Tensor
    num_chunks_ptr: torch.Tensor
    tmp_output: torch.Tensor
    tmp_lse: torch.Tensor
    output: torch.Tensor
    launch_num_chunks: int
    attn_sink: torch.Tensor | None = None
    scratch: object | None = None
    identity_page_table: bool = False

    def run(self) -> None:
        run_sparse_mla_split_decode(binding=self)


def build_sparse_mla_split_decode_forward_binding(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    active_token_counts: torch.Tensor,
    sm_scale: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    launch_num_chunks: int,
    scratch: object | None = None,
    identity_page_table: bool = False,
) -> SparseMLASplitDecodeForwardBinding:
    return SparseMLASplitDecodeForwardBinding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        launch_num_chunks=int(launch_num_chunks),
        scratch=scratch,
        identity_page_table=bool(identity_page_table),
    )


def build_compressed_mla_split_decode_forward_binding(
    *,
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_k_cache: torch.Tensor,
    indexed_indices: torch.Tensor,
    indexed_lengths: torch.Tensor,
    indexed_page_table: torch.Tensor,
    sm_scale: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    launch_num_chunks: int,
    swa_page_size: int,
    swa_page_nbytes: int,
    indexed_page_size: int,
    indexed_page_nbytes: int,
    has_indexed: bool,
    map_indexed_page_table: bool,
    scratch: object | None = None,
    direct_output: bool = False,
    single_tile_chunks: bool = False,
    attn_sink: torch.Tensor | None = None,
    direct_sink_output: bool = False,
) -> CompressedMLASplitDecodeForwardBinding:
    return CompressedMLASplitDecodeForwardBinding(
        q_all=q_all,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_k_cache=indexed_k_cache,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
        sm_scale=sm_scale,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        launch_num_chunks=int(launch_num_chunks),
        swa_page_size=int(swa_page_size),
        swa_page_nbytes=int(swa_page_nbytes),
        indexed_page_size=int(indexed_page_size),
        indexed_page_nbytes=int(indexed_page_nbytes),
        has_indexed=bool(has_indexed),
        map_indexed_page_table=bool(map_indexed_page_table),
        scratch=scratch,
        direct_output=bool(direct_output),
        single_tile_chunks=bool(single_tile_chunks),
        attn_sink=attn_sink,
        direct_sink_output=bool(direct_sink_output),
    )


def build_sparse_mla_split_decode_merge_binding(
    *,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor | None = None,
    scratch: object | None = None,
) -> SparseMLASplitDecodeMergeBinding:
    return SparseMLASplitDecodeMergeBinding(
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        num_chunks_ptr=num_chunks_ptr,
        output=output,
        attn_sink=attn_sink,
        scratch=scratch,
    )


def build_sparse_mla_split_decode_binding(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    active_token_counts: torch.Tensor,
    sm_scale: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    output: torch.Tensor,
    launch_num_chunks: int,
    attn_sink: torch.Tensor | None = None,
    scratch: object | None = None,
    identity_page_table: bool = False,
) -> SparseMLASplitDecodeBinding:
    return SparseMLASplitDecodeBinding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        output=output,
        launch_num_chunks=int(launch_num_chunks),
        attn_sink=attn_sink,
        scratch=scratch,
        identity_page_table=bool(identity_page_table),
    )


def _validate_tensor_storage_bounds(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.numel() == 0:
        return
    min_offset = int(tensor.storage_offset())
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
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


def _validate_split_control_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if tensor.shape != (1,):
        raise ValueError(f"{name} must have shape (1,), got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _compressed_mla_cache_byte_view(cache: torch.Tensor, *, name: str) -> torch.Tensor:
    if cache.dtype == torch.uint8:
        byte_cache = cache
    elif cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        byte_cache = cache.detach().view(torch.uint8)
    else:
        raise TypeError(
            f"{name} must have dtype torch.uint8 or FP8 byte view, got {cache.dtype}"
        )

    if byte_cache.ndim != 2 or not byte_cache.is_contiguous():
        raise ValueError(f"{name} must be contiguous with shape [pages, page_nbytes]")
    return byte_cache


def _gmem_ptr(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int,
) -> cute.Pointer:
    return make_ptr(
        dtype,
        int(tensor.data_ptr()),
        cute.AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def _fake_gmem_ptr(
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int,
) -> cute.Pointer:
    return make_ptr(
        dtype,
        assumed_align,
        cute.AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def _fake_compact_tensor(
    dtype: type[cutlass.Numeric],
    shape: tuple[int, ...],
    *,
    assumed_align: int,
) -> cute.Tensor:
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        assumed_align=assumed_align,
    )


def _tensor_compile_key(
    name: str,
    tensor: torch.Tensor,
    *,
    dynamic_dims: tuple[int, ...] = (),
    dynamic_strides: tuple[int, ...] = (),
) -> tuple[object, ...]:
    dynamic_dim_set = set(dynamic_dims)
    dynamic_stride_set = set(dynamic_strides)
    dims = tuple(
        DimKey.dynamic() if idx in dynamic_dim_set else DimKey.exact(int(dim))
        for idx, dim in enumerate(tensor.shape)
    )
    strides = tuple(
        DimKey.dynamic() if idx in dynamic_stride_set else DimKey.exact(int(stride))
        for idx, stride in enumerate(tensor.stride())
    )
    return (
        "tensor",
        name,
        str(tensor.dtype),
        int(tensor.ndim),
        dims,
        strides,
        (tensor.device.type, tensor.device.index),
    )


def get_sparse_mla_split_shared_storage_cls():
    """SharedStorage for split kernel: no kv_stage_b (single-tile path only)."""

    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "q_group_stage": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(_MLA_Q_GROUP_STAGE_BYTES)],
            128,
        ],
        "kv_stage_a": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(_MLA_KV_STAGE_BYTES)],
            128,
        ],
        "token_idx": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, _MLA_TOKEN_TILE],
            16,
        ],
        "token_scale_a": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MLA_SHARED_SCALE_STAGE_ELEMS],
            16,
        ],
    }
    return cute.struct(SharedStorage)


@cute.jit
def _split_output_lane_view(
    tmp_output: cute.Tensor,
    q_idx: Int32,
    head_idx: Int32,
    out_base: Int32,
) -> cute.Tensor:
    return cute.make_tensor(
        attention_ops.elem_pointer(tmp_output, (q_idx, head_idx, Int32(0), out_base)),
        cute.make_layout(
            (tmp_output.shape[2], 4),
            stride=(tmp_output.stride[2], 1),
        ),
    )


@cute.jit
def _split_lse_head_view(
    tmp_lse: cute.Tensor,
    q_idx: Int32,
    head_idx: Int32,
) -> cute.Tensor:
    return cute.make_tensor(
        attention_ops.elem_pointer(tmp_lse, (q_idx, head_idx, Int32(0))),
        cute.make_layout(
            (tmp_lse.shape[2],),
            stride=(tmp_lse.stride[2],),
        ),
    )


def select_sparse_mla_split_decode_config(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    output_dtype: torch.dtype,
    v_head_dim: int,
    max_chunks: int = _SPLIT_MAX_CHUNKS,
) -> SparseMLASplitDecodeConfig | None:
    traits = select_sparse_mla_traits(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        output_dtype=output_dtype,
        v_head_dim=v_head_dim,
    )
    if traits is None:
        return None

    width = int(page_table_1.shape[1])
    env_chunk = os.environ.get("B12X_MLA_SPLIT_CHUNK_SIZE", None)
    if env_chunk is not None:
        chunk_size = int(env_chunk)
        num_chunks = _ceil_div(width, chunk_size)
        if num_chunks > max(1, min(int(max_chunks), _SPLIT_MAX_CHUNKS)):
            return None
        return SparseMLASplitDecodeConfig(chunk_size=chunk_size, num_chunks=num_chunks)
    return default_sparse_mla_split_decode_config_for_width(
        width, max_chunks=max_chunks
    )


@cute.jit
def _zero_partial_head_tile(
    tmp_output: cute.Tensor,
    tmp_lse: cute.Tensor,
    q_idx: Int32,
    chunk_idx: Int32,
    head_tile_start: Int32,
    lane: Int32,
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    for row_slot in cutlass.range_constexpr(2):
        head_local = lane_group + Int32(8) * row_slot
        head_idx = head_tile_start + head_local
        if head_idx < Int32(tmp_output.shape[1]):
            for group_idx in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
                out_base = Int32(group_idx * _MLA_GROUP_SIZE) + lane_pair_base
                for mma_d in cutlass.range_constexpr(8):
                    dim_base = out_base + mma_d * Int32(16)
                    tmp_output[q_idx, head_idx, chunk_idx, dim_base + Int32(0)] = (
                        Float32(0.0).to(tmp_output.element_type)
                    )
                    tmp_output[q_idx, head_idx, chunk_idx, dim_base + Int32(1)] = (
                        Float32(0.0).to(tmp_output.element_type)
                    )
                    tmp_output[q_idx, head_idx, chunk_idx, dim_base + Int32(8)] = (
                        Float32(0.0).to(tmp_output.element_type)
                    )
                    tmp_output[q_idx, head_idx, chunk_idx, dim_base + Int32(9)] = (
                        Float32(0.0).to(tmp_output.element_type)
                    )
            if lane % Int32(4) == Int32(0):
                tmp_lse[q_idx, head_idx, chunk_idx] = Float32(-Float32.inf)


class SparseMLASplitDecodeForwardKernel:
    """Chunk-local sparse MLA partial forward for decode."""

    def __init__(
        self, launch_num_chunks: int, head_tiles: int, identity_page_table: bool = False
    ):
        self.launch_num_chunks = int(launch_num_chunks)
        self.head_tiles = int(head_tiles)
        self.identity_page_table = bool(identity_page_table)

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        active_token_counts: cute.Tensor,
        sm_scale: cute.Tensor,
        kv_chunk_size_ptr: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_u32,
            kv_rows_u32,
            kv_scales,
            page_table_1,
            active_token_counts,
            sm_scale,
            kv_chunk_size_ptr,
            num_chunks_ptr,
            tmp_output,
            tmp_lse,
        ).launch(
            grid=(
                q_u32.shape[0],
                self.head_tiles,
                self.launch_num_chunks,
            ),
            block=[_MLA_WARP_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        kv_rows_u32: cute.Tensor,
        kv_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        active_token_counts: cute.Tensor,
        sm_scale: cute.Tensor,
        kv_chunk_size_ptr: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        q_idx, head_tile_idx, chunk_idx = cute.arch.block_idx()
        q_idx = Int32(q_idx)
        head_tile_start = Int32(head_tile_idx * _MLA_HEADS_PER_TILE)
        chunk_idx = Int32(chunk_idx)

        active_num_chunks = Int32(num_chunks_ptr[Int32(0)])
        if active_num_chunks > Int32(_SPLIT_MAX_CHUNKS):
            active_num_chunks = Int32(_SPLIT_MAX_CHUNKS)
        row_token_end = _clamp_active_token_count(
            active_token_counts, q_idx, Int32(page_table_1.shape[1])
        )
        chunk_size = Int32(kv_chunk_size_ptr[Int32(0)])
        token_start = Int32(chunk_idx) * chunk_size
        if chunk_idx >= active_num_chunks or token_start >= row_token_end:
            _zero_partial_head_tile(
                tmp_output, tmp_lse, q_idx, chunk_idx, head_tile_start, lane
            )
        else:
            token_end = token_start + chunk_size
            if token_end > row_token_end:
                token_end = row_token_end

            smem = cutlass.utils.SmemAllocator()
            SharedStorage = get_sparse_mla_split_shared_storage_cls()
            storage = smem.allocate(SharedStorage)
            sTokenIdx = storage.token_idx.get_tensor(
                cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,))
            )
            sScale = storage.token_scale_a.get_tensor(
                cute.make_layout((_MLA_SHARED_SCALE_STAGE_ELEMS,), stride=(1,))
            )

            q_base_addr = shared_ptr_to_u32(storage.q_group_stage.data_ptr())
            kv_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

            _run_one_pass_sparse_mla_tile(
                q_u32,
                kv_rows_u32,
                kv_scales,
                page_table_1,
                sTokenIdx,
                sScale,
                q_base_addr,
                kv_base_addr,
                q_idx,
                head_tile_start,
                token_start,
                token_end,
                Float32(sm_scale[Int32(0)] * attention_ops.LOG2_E),
                lane,
                tmp_output,
                q_idx,
                chunk_idx,
                tmp_lse,
                self.identity_page_table,
            )


class CompressedMLASplitDecodeForwardKernel:
    """Chunk-local compressed-layout sparse MLA partial forward."""

    def __init__(
        self,
        *,
        launch_num_chunks: int,
        head_tiles: int,
        swa_page_size: int,
        swa_page_nbytes: int,
        indexed_page_size: int,
        indexed_page_nbytes: int,
        num_heads: int,
        swa_cache_nbytes: int,
        indexed_cache_nbytes: int,
        swa_indices_width: int,
        indexed_indices_width: int,
        tmp_output_chunks: int,
        tmp_lse_chunks: int,
        has_swa: bool,
        has_indexed: bool,
        map_indexed_page_table: bool,
        direct_output: bool = False,
        single_tile_chunks: bool = False,
        direct_sink_output: bool = False,
    ):
        self.launch_num_chunks = int(launch_num_chunks)
        self.head_tiles = int(head_tiles)
        self.swa_page_size = int(swa_page_size)
        self.swa_page_nbytes = int(swa_page_nbytes)
        self.indexed_page_size = int(indexed_page_size)
        self.indexed_page_nbytes = int(indexed_page_nbytes)
        self.num_heads = int(num_heads)
        self.q_u32_width = _COMPRESSED_MLA_HEAD_DIM // 2
        self.swa_cache_nbytes = max(int(swa_cache_nbytes), 1)
        self.indexed_cache_nbytes = max(int(indexed_cache_nbytes), 1)
        self.swa_indices_width = max(int(swa_indices_width), 1)
        self.indexed_indices_width = max(int(indexed_indices_width), 1)
        self.tmp_output_chunks = int(tmp_output_chunks)
        self.tmp_lse_chunks = int(tmp_lse_chunks)
        self.has_swa = bool(has_swa)
        self.has_indexed = bool(has_indexed)
        self.map_indexed_page_table = bool(map_indexed_page_table)
        self.direct_output = bool(direct_output)
        self.single_tile_chunks = bool(single_tile_chunks)
        self.direct_sink_output = bool(direct_sink_output)

    @cute.jit
    def __call__(
        self,
        q_u32_ptr: cute.Pointer,
        swa_u8_ptr: cute.Pointer,
        swa_indices_ptr: cute.Pointer,
        swa_lengths_ptr: cute.Pointer,
        indexed_u8_ptr: cute.Pointer,
        indexed_indices_ptr: cute.Pointer,
        indexed_lengths_ptr: cute.Pointer,
        indexed_page_table_ptr: cute.Pointer,
        sm_scale: cute.Tensor,
        kv_chunk_size_ptr: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        tmp_output_ptr: cute.Pointer,
        tmp_lse_ptr: cute.Pointer,
        attn_sink: cute.Tensor,
        indexed_page_table_width: cutlass.Int32,
        indexed_page_table_stride0: cutlass.Int32,
        tmp_output_stride0: cutlass.Int32,
        tmp_output_stride1: cutlass.Int32,
        tmp_output_stride2: cutlass.Int32,
        tmp_output_stride3: cutlass.Int32,
        tmp_lse_stride0: cutlass.Int32,
        tmp_lse_stride1: cutlass.Int32,
        tmp_lse_stride2: cutlass.Int32,
        launch_num_chunks_runtime: cutlass.Int32,
        num_rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        q_u32 = cute.make_tensor(
            q_u32_ptr,
            layout=cute.make_layout(
                (num_rows, self.num_heads, self.q_u32_width),
                stride=(
                    self.num_heads * self.q_u32_width,
                    self.q_u32_width,
                    1,
                ),
            ),
        )
        swa_u8 = cute.make_tensor(
            swa_u8_ptr,
            layout=cute.make_layout((self.swa_cache_nbytes,), stride=(1,)),
        )
        swa_indices = cute.make_tensor(
            swa_indices_ptr,
            layout=cute.make_layout(
                (num_rows, self.swa_indices_width),
                stride=(self.swa_indices_width, 1),
            ),
        )
        swa_lengths = cute.make_tensor(
            swa_lengths_ptr,
            layout=cute.make_layout((num_rows,), stride=(1,)),
        )
        indexed_u8 = cute.make_tensor(
            indexed_u8_ptr,
            layout=cute.make_layout((self.indexed_cache_nbytes,), stride=(1,)),
        )
        indexed_indices = cute.make_tensor(
            indexed_indices_ptr,
            layout=cute.make_layout(
                (num_rows, self.indexed_indices_width),
                stride=(self.indexed_indices_width, 1),
            ),
        )
        indexed_lengths = cute.make_tensor(
            indexed_lengths_ptr,
            layout=cute.make_layout((num_rows,), stride=(1,)),
        )
        indexed_page_table = cute.make_tensor(
            indexed_page_table_ptr,
            layout=cute.make_layout(
                (num_rows, indexed_page_table_width),
                stride=(indexed_page_table_stride0, 1),
            ),
        )
        if cutlass.const_expr(self.direct_output):
            tmp_output = cute.make_tensor(
                tmp_output_ptr,
                layout=cute.make_layout(
                    (num_rows, self.num_heads, _COMPRESSED_MLA_HEAD_DIM),
                    stride=(tmp_output_stride0, tmp_output_stride1, tmp_output_stride2),
                ),
            )
        else:
            tmp_output = cute.make_tensor(
                tmp_output_ptr,
                layout=cute.make_layout(
                    (
                        num_rows,
                        self.num_heads,
                        self.tmp_output_chunks,
                        _COMPRESSED_MLA_HEAD_DIM,
                    ),
                    stride=(
                        tmp_output_stride0,
                        tmp_output_stride1,
                        tmp_output_stride2,
                        tmp_output_stride3,
                    ),
                ),
            )
        tmp_lse = cute.make_tensor(
            tmp_lse_ptr,
            layout=cute.make_layout(
                (num_rows, self.num_heads, self.tmp_lse_chunks),
                stride=(tmp_lse_stride0, tmp_lse_stride1, tmp_lse_stride2),
            ),
        )
        self.kernel(
            q_u32,
            swa_u8,
            swa_indices,
            swa_lengths,
            indexed_u8,
            indexed_indices,
            indexed_lengths,
            indexed_page_table,
            sm_scale,
            kv_chunk_size_ptr,
            num_chunks_ptr,
            tmp_output,
            tmp_lse,
            attn_sink,
        ).launch(
            grid=(
                num_rows,
                self.head_tiles,
                launch_num_chunks_runtime,
            ),
            block=[_MLA_WARP_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        swa_u8: cute.Tensor,
        swa_indices: cute.Tensor,
        swa_lengths: cute.Tensor,
        indexed_u8: cute.Tensor,
        indexed_indices: cute.Tensor,
        indexed_lengths: cute.Tensor,
        indexed_page_table: cute.Tensor,
        sm_scale: cute.Tensor,
        kv_chunk_size_ptr: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        attn_sink: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        q_idx, head_tile_idx, chunk_idx = cute.arch.block_idx()
        q_idx = Int32(q_idx)
        head_tile_start = Int32(head_tile_idx * _MLA_HEADS_PER_TILE)
        chunk_idx = Int32(chunk_idx)
        indexed_page_table_width = Int32(indexed_page_table.shape[1])

        if cutlass.const_expr(self.direct_output):
            swa_len = Int32(0)
            if cutlass.const_expr(self.has_swa):
                swa_len = _clamp_active_token_count(
                    swa_lengths, q_idx, Int32(swa_indices.shape[1])
                )
            indexed_len = Int32(0)
            if cutlass.const_expr(self.has_indexed):
                indexed_len = _clamp_active_token_count(
                    indexed_lengths,
                    q_idx,
                    Int32(indexed_indices.shape[1]),
                )
            row_token_end = swa_len + indexed_len

            direct_smem = cutlass.utils.SmemAllocator()
            DirectSharedStorage = get_sparse_mla_split_shared_storage_cls()
            direct_storage = direct_smem.allocate(DirectSharedStorage)
            direct_sTokenIdx = direct_storage.token_idx.get_tensor(
                cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,))
            )
            direct_sScale = direct_storage.token_scale_a.get_tensor(
                cute.make_layout((_MLA_SHARED_SCALE_STAGE_ELEMS,), stride=(1,))
            )

            direct_q_base_addr = shared_ptr_to_u32(
                direct_storage.q_group_stage.data_ptr()
            )
            direct_kv_base_addr = shared_ptr_to_u32(
                direct_storage.kv_stage_a.data_ptr()
            )

            if cutlass.const_expr(self.single_tile_chunks):
                _run_single_tile_compressed_mla_tile(
                    q_u32,
                    swa_u8,
                    swa_indices,
                    swa_lengths,
                    indexed_u8,
                    indexed_indices,
                    indexed_lengths,
                    indexed_page_table,
                    direct_sTokenIdx,
                    direct_sScale,
                    direct_q_base_addr,
                    direct_kv_base_addr,
                    q_idx,
                    head_tile_start,
                    Int32(0),
                    row_token_end,
                    Float32(sm_scale[Int32(0)] * attention_ops.LOG2_E),
                    lane,
                    tmp_output,
                    q_idx,
                    Int32(0),
                    None,
                    attn_sink,
                    self.direct_sink_output,
                    self.swa_page_size,
                    self.swa_page_nbytes,
                    self.indexed_page_size,
                    self.indexed_page_nbytes,
                    self.has_swa,
                    self.has_indexed,
                    self.map_indexed_page_table,
                    indexed_page_table_width,
                )
            else:
                _run_one_pass_compressed_mla_tile(
                    q_u32,
                    swa_u8,
                    swa_indices,
                    swa_lengths,
                    indexed_u8,
                    indexed_indices,
                    indexed_lengths,
                    indexed_page_table,
                    direct_sTokenIdx,
                    direct_sScale,
                    direct_q_base_addr,
                    direct_kv_base_addr,
                    q_idx,
                    head_tile_start,
                    Int32(0),
                    row_token_end,
                    Float32(sm_scale[Int32(0)] * attention_ops.LOG2_E),
                    lane,
                    tmp_output,
                    q_idx,
                    Int32(0),
                    None,
                    attn_sink,
                    self.direct_sink_output,
                    self.swa_page_size,
                    self.swa_page_nbytes,
                    self.indexed_page_size,
                    self.indexed_page_nbytes,
                    self.has_swa,
                    self.has_indexed,
                    self.map_indexed_page_table,
                    indexed_page_table_width,
                )
        else:
            active_num_chunks = Int32(num_chunks_ptr[Int32(0)])
            if active_num_chunks > Int32(_SPLIT_MAX_CHUNKS):
                active_num_chunks = Int32(_SPLIT_MAX_CHUNKS)
            swa_len = Int32(0)
            if cutlass.const_expr(self.has_swa):
                swa_len = _clamp_active_token_count(
                    swa_lengths, q_idx, Int32(swa_indices.shape[1])
                )
            indexed_len = Int32(0)
            if cutlass.const_expr(self.has_indexed):
                indexed_len = _clamp_active_token_count(
                    indexed_lengths,
                    q_idx,
                    Int32(indexed_indices.shape[1]),
                )
            row_token_end = swa_len + indexed_len
            chunk_size = Int32(kv_chunk_size_ptr[Int32(0)])
            token_start = Int32(chunk_idx) * chunk_size
            if chunk_idx >= active_num_chunks or token_start >= row_token_end:
                _zero_partial_head_tile(
                    tmp_output, tmp_lse, q_idx, chunk_idx, head_tile_start, lane
                )
            else:
                token_end = token_start + chunk_size
                if token_end > row_token_end:
                    token_end = row_token_end

                split_smem = cutlass.utils.SmemAllocator()
                SplitSharedStorage = get_sparse_mla_split_shared_storage_cls()
                split_storage = split_smem.allocate(SplitSharedStorage)
                split_sTokenIdx = split_storage.token_idx.get_tensor(
                    cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,))
                )
                split_sScale = split_storage.token_scale_a.get_tensor(
                    cute.make_layout((_MLA_SHARED_SCALE_STAGE_ELEMS,), stride=(1,))
                )

                split_q_base_addr = shared_ptr_to_u32(
                    split_storage.q_group_stage.data_ptr()
                )
                split_kv_base_addr = shared_ptr_to_u32(
                    split_storage.kv_stage_a.data_ptr()
                )

                if cutlass.const_expr(self.single_tile_chunks):
                    _run_single_tile_compressed_mla_tile(
                        q_u32,
                        swa_u8,
                        swa_indices,
                        swa_lengths,
                        indexed_u8,
                        indexed_indices,
                        indexed_lengths,
                        indexed_page_table,
                        split_sTokenIdx,
                        split_sScale,
                        split_q_base_addr,
                        split_kv_base_addr,
                        q_idx,
                        head_tile_start,
                        token_start,
                        token_end,
                        Float32(sm_scale[Int32(0)] * attention_ops.LOG2_E),
                        lane,
                        tmp_output,
                        q_idx,
                        chunk_idx,
                        tmp_lse,
                        attn_sink,
                        False,
                        self.swa_page_size,
                        self.swa_page_nbytes,
                        self.indexed_page_size,
                        self.indexed_page_nbytes,
                        self.has_swa,
                        self.has_indexed,
                        self.map_indexed_page_table,
                        indexed_page_table_width,
                    )
                else:
                    _run_one_pass_compressed_mla_tile(
                        q_u32,
                        swa_u8,
                        swa_indices,
                        swa_lengths,
                        indexed_u8,
                        indexed_indices,
                        indexed_lengths,
                        indexed_page_table,
                        split_sTokenIdx,
                        split_sScale,
                        split_q_base_addr,
                        split_kv_base_addr,
                        q_idx,
                        head_tile_start,
                        token_start,
                        token_end,
                        Float32(sm_scale[Int32(0)] * attention_ops.LOG2_E),
                        lane,
                        tmp_output,
                        q_idx,
                        chunk_idx,
                        tmp_lse,
                        attn_sink,
                        False,
                        self.swa_page_size,
                        self.swa_page_nbytes,
                        self.indexed_page_size,
                        self.indexed_page_nbytes,
                        self.has_swa,
                        self.has_indexed,
                        self.map_indexed_page_table,
                        indexed_page_table_width,
                    )


class SparseMLASplitDecodeMergeKernel:
    """Reduce normalized chunk partials into the final decode output."""

    @cute.jit
    def __call__(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            tmp_output,
            tmp_lse,
            num_chunks_ptr,
            output,
        ).launch(
            grid=(output.shape[0], output.shape[1], _MLA_SCALE_GROUPS),
            block=[_MLA_WARP_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        output: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        q_idx, head_idx, group_idx = cute.arch.block_idx()
        q_idx = Int32(q_idx)
        head_idx = Int32(head_idx)
        group_idx = Int32(group_idx)

        acc = cute.make_rmem_tensor((4,), Float32)
        for frag_idx in cutlass.range_constexpr(4):
            acc[frag_idx] = Float32(0.0)

        out_base = group_idx * Int32(_MLA_GROUP_SIZE) + lane * Int32(4)
        tmp_output_lane = _split_output_lane_view(tmp_output, q_idx, head_idx, out_base)
        tmp_lse_head = _split_lse_head_view(tmp_lse, q_idx, head_idx)
        merged_m = Float32(-Float32.inf)
        merged_d = Float32(1.0)
        chunk_idx = Int32(0)
        num_chunks = Int32(num_chunks_ptr[Int32(0)])
        if num_chunks > Int32(_SPLIT_MAX_CHUNKS):
            num_chunks = Int32(_SPLIT_MAX_CHUNKS)

        while chunk_idx < num_chunks and merged_m == Float32(-Float32.inf):
            part_lse = Float32(tmp_lse_head[chunk_idx])
            if part_lse != Float32(-Float32.inf):
                acc[0] = Float32(tmp_output_lane[chunk_idx, Int32(0)])
                acc[1] = Float32(tmp_output_lane[chunk_idx, Int32(1)])
                acc[2] = Float32(tmp_output_lane[chunk_idx, Int32(2)])
                acc[3] = Float32(tmp_output_lane[chunk_idx, Int32(3)])
                merged_m = Float32(part_lse)
                merged_d = Float32(1.0)
            chunk_idx += Int32(1)

        while chunk_idx < num_chunks:
            part_lse = Float32(tmp_lse_head[chunk_idx])
            if part_lse != Float32(-Float32.inf):
                new_m = attention_ops.fmax(merged_m, part_lse)
                prev_scale = _exp2_approx_ftz_f32(merged_m - new_m)
                part_scale = _exp2_approx_ftz_f32(part_lse - new_m)
                merged_d = Float32(merged_d * prev_scale + part_scale)
                acc[0] = Float32(
                    acc[0] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(0)]) * part_scale
                )
                acc[1] = Float32(
                    acc[1] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(1)]) * part_scale
                )
                acc[2] = Float32(
                    acc[2] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(2)]) * part_scale
                )
                acc[3] = Float32(
                    acc[3] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(3)]) * part_scale
                )
                merged_m = Float32(new_m)
            chunk_idx += Int32(1)

        if merged_m == Float32(-Float32.inf):
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(0.0).to(
                output.element_type
            )
        else:
            inv_d = cute.arch.rcp_approx(merged_d)
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(acc[0] * inv_d).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(acc[1] * inv_d).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(acc[2] * inv_d).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(acc[3] * inv_d).to(
                output.element_type
            )


class SparseMLASplitDecodeSinkMergeKernel:
    """Reduce chunk partials and fold a zero-value attention sink into softmax."""

    @cute.jit
    def __call__(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        attn_sink: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            tmp_output,
            tmp_lse,
            num_chunks_ptr,
            attn_sink,
            output,
        ).launch(
            grid=(output.shape[0], output.shape[1], _MLA_SCALE_GROUPS),
            block=[_MLA_WARP_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        attn_sink: cute.Tensor,
        output: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        q_idx, head_idx, group_idx = cute.arch.block_idx()
        q_idx = Int32(q_idx)
        head_idx = Int32(head_idx)
        group_idx = Int32(group_idx)

        acc = cute.make_rmem_tensor((4,), Float32)
        for frag_idx in cutlass.range_constexpr(4):
            acc[frag_idx] = Float32(0.0)

        out_base = group_idx * Int32(_MLA_GROUP_SIZE) + lane * Int32(4)
        tmp_output_lane = _split_output_lane_view(tmp_output, q_idx, head_idx, out_base)
        tmp_lse_head = _split_lse_head_view(tmp_lse, q_idx, head_idx)
        merged_m = Float32(-Float32.inf)
        merged_d = Float32(1.0)
        chunk_idx = Int32(0)
        num_chunks = Int32(num_chunks_ptr[Int32(0)])
        if num_chunks > Int32(_SPLIT_MAX_CHUNKS):
            num_chunks = Int32(_SPLIT_MAX_CHUNKS)

        while chunk_idx < num_chunks and merged_m == Float32(-Float32.inf):
            part_lse = Float32(tmp_lse_head[chunk_idx])
            if part_lse != Float32(-Float32.inf):
                acc[0] = Float32(tmp_output_lane[chunk_idx, Int32(0)])
                acc[1] = Float32(tmp_output_lane[chunk_idx, Int32(1)])
                acc[2] = Float32(tmp_output_lane[chunk_idx, Int32(2)])
                acc[3] = Float32(tmp_output_lane[chunk_idx, Int32(3)])
                merged_m = Float32(part_lse)
                merged_d = Float32(1.0)
            chunk_idx += Int32(1)

        while chunk_idx < num_chunks:
            part_lse = Float32(tmp_lse_head[chunk_idx])
            if part_lse != Float32(-Float32.inf):
                new_m = attention_ops.fmax(merged_m, part_lse)
                prev_scale = _exp2_approx_ftz_f32(merged_m - new_m)
                part_scale = _exp2_approx_ftz_f32(part_lse - new_m)
                merged_d = Float32(merged_d * prev_scale + part_scale)
                acc[0] = Float32(
                    acc[0] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(0)]) * part_scale
                )
                acc[1] = Float32(
                    acc[1] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(1)]) * part_scale
                )
                acc[2] = Float32(
                    acc[2] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(2)]) * part_scale
                )
                acc[3] = Float32(
                    acc[3] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(3)]) * part_scale
                )
                merged_m = Float32(new_m)
            chunk_idx += Int32(1)

        if merged_m == Float32(-Float32.inf):
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(0.0).to(
                output.element_type
            )
        else:
            sink_m = Float32(attn_sink[head_idx] * attention_ops.LOG2_E)
            new_m = attention_ops.fmax(merged_m, sink_m)
            prev_scale = _exp2_approx_ftz_f32(merged_m - new_m)
            sink_scale = _exp2_approx_ftz_f32(sink_m - new_m)
            merged_d = Float32(merged_d * prev_scale + sink_scale)
            inv_d = cute.arch.rcp_approx(merged_d)
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(
                acc[0] * prev_scale * inv_d
            ).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(
                acc[1] * prev_scale * inv_d
            ).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(
                acc[2] * prev_scale * inv_d
            ).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(
                acc[3] * prev_scale * inv_d
            ).to(output.element_type)


@lru_cache(maxsize=16)
def _build_sparse_mla_split_forward_kernel(
    traits: SparseMLATraits,
    launch_num_chunks: int,
    head_tiles: int,
    identity_page_table: bool,
) -> SparseMLASplitDecodeForwardKernel:
    del traits
    return SparseMLASplitDecodeForwardKernel(
        launch_num_chunks,
        head_tiles,
        identity_page_table,
    )


@lru_cache(maxsize=64)
def _build_compressed_mla_split_forward_kernel(
    launch_num_chunks: int,
    head_tiles: int,
    swa_page_size: int,
    swa_page_nbytes: int,
    indexed_page_size: int,
    indexed_page_nbytes: int,
    num_heads: int,
    swa_cache_nbytes: int,
    indexed_cache_nbytes: int,
    swa_indices_width: int,
    indexed_indices_width: int,
    tmp_output_chunks: int,
    tmp_lse_chunks: int,
    has_swa: bool,
    has_indexed: bool,
    map_indexed_page_table: bool,
    direct_output: bool,
    single_tile_chunks: bool,
    direct_sink_output: bool,
) -> CompressedMLASplitDecodeForwardKernel:
    return CompressedMLASplitDecodeForwardKernel(
        launch_num_chunks=launch_num_chunks,
        head_tiles=head_tiles,
        swa_page_size=swa_page_size,
        swa_page_nbytes=swa_page_nbytes,
        indexed_page_size=indexed_page_size,
        indexed_page_nbytes=indexed_page_nbytes,
        num_heads=num_heads,
        swa_cache_nbytes=swa_cache_nbytes,
        indexed_cache_nbytes=indexed_cache_nbytes,
        swa_indices_width=swa_indices_width,
        indexed_indices_width=indexed_indices_width,
        tmp_output_chunks=tmp_output_chunks,
        tmp_lse_chunks=tmp_lse_chunks,
        has_swa=has_swa,
        has_indexed=has_indexed,
        map_indexed_page_table=map_indexed_page_table,
        direct_output=direct_output,
        single_tile_chunks=single_tile_chunks,
        direct_sink_output=direct_sink_output,
    )


@lru_cache(maxsize=1)
def _build_sparse_mla_split_merge_kernel() -> SparseMLASplitDecodeMergeKernel:
    return SparseMLASplitDecodeMergeKernel()


@lru_cache(maxsize=1)
def _build_sparse_mla_split_sink_merge_kernel() -> SparseMLASplitDecodeSinkMergeKernel:
    return SparseMLASplitDecodeSinkMergeKernel()


def clear_sparse_mla_split_kernel_cache() -> None:
    _build_sparse_mla_split_forward_kernel.cache_clear()
    _build_compressed_mla_split_forward_kernel.cache_clear()
    _build_sparse_mla_split_merge_kernel.cache_clear()
    _build_sparse_mla_split_sink_merge_kernel.cache_clear()


def run_sparse_mla_split_decode_forward(
    *,
    q_all: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    page_table_1: torch.Tensor | None = None,
    active_token_counts: torch.Tensor | None = None,
    sm_scale: torch.Tensor | None = None,
    kv_chunk_size_ptr: torch.Tensor | None = None,
    num_chunks_ptr: torch.Tensor | None = None,
    tmp_output: torch.Tensor | None = None,
    tmp_lse: torch.Tensor | None = None,
    launch_num_chunks: int | None = None,
    workspace: object | None = None,
    identity_page_table: bool | None = None,
    binding: SparseMLASplitDecodeForwardBinding | None = None,
) -> None:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q_all", q_all),
                ("kv_cache", kv_cache),
                ("page_table_1", page_table_1),
                ("active_token_counts", active_token_counts),
                ("sm_scale", sm_scale),
                ("kv_chunk_size_ptr", kv_chunk_size_ptr),
                ("num_chunks_ptr", num_chunks_ptr),
                ("tmp_output", tmp_output),
                ("tmp_lse", tmp_lse),
                ("launch_num_chunks", launch_num_chunks),
                ("workspace", workspace),
                ("identity_page_table", identity_page_table),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_sparse_mla_split_decode_forward", extras)
        q_all = binding.q_all
        kv_cache = binding.kv_cache
        page_table_1 = binding.page_table_1
        active_token_counts = binding.active_token_counts
        sm_scale = binding.sm_scale
        kv_chunk_size_ptr = binding.kv_chunk_size_ptr
        num_chunks_ptr = binding.num_chunks_ptr
        tmp_output = binding.tmp_output
        tmp_lse = binding.tmp_lse
        launch_num_chunks = binding.launch_num_chunks
        workspace = binding.scratch
        identity_page_table = binding.identity_page_table

    q_all = _require_bound_arg(
        q_all, api_name="run_sparse_mla_split_decode_forward", name="q_all"
    )
    kv_cache = _require_bound_arg(
        kv_cache,
        api_name="run_sparse_mla_split_decode_forward",
        name="kv_cache",
    )
    page_table_1 = _require_bound_arg(
        page_table_1,
        api_name="run_sparse_mla_split_decode_forward",
        name="page_table_1",
    )
    active_token_counts = _require_bound_arg(
        active_token_counts,
        api_name="run_sparse_mla_split_decode_forward",
        name="active_token_counts",
    )
    sm_scale = _require_bound_arg(
        sm_scale,
        api_name="run_sparse_mla_split_decode_forward",
        name="sm_scale",
    )
    kv_chunk_size_ptr = _require_bound_arg(
        kv_chunk_size_ptr,
        api_name="run_sparse_mla_split_decode_forward",
        name="kv_chunk_size_ptr",
    )
    num_chunks_ptr = _require_bound_arg(
        num_chunks_ptr,
        api_name="run_sparse_mla_split_decode_forward",
        name="num_chunks_ptr",
    )
    tmp_output = _require_bound_arg(
        tmp_output,
        api_name="run_sparse_mla_split_decode_forward",
        name="tmp_output",
    )
    tmp_lse = _require_bound_arg(
        tmp_lse,
        api_name="run_sparse_mla_split_decode_forward",
        name="tmp_lse",
    )
    launch_num_chunks = _require_bound_arg(
        launch_num_chunks,
        api_name="run_sparse_mla_split_decode_forward",
        name="launch_num_chunks",
    )
    launch_num_chunks = int(launch_num_chunks)
    identity_page_table = (
        False if identity_page_table is None else bool(identity_page_table)
    )

    traits = select_sparse_mla_traits(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        output_dtype=tmp_output.dtype,
        v_head_dim=tmp_output.shape[-1],
    )
    if traits is None:
        raise ValueError(
            "sparse MLA split decode only supports the exact CUDA GLM-5.1 contract"
        )
    if active_token_counts.dtype != torch.int32:
        raise ValueError(
            f"active_token_counts must have dtype torch.int32, got {active_token_counts.dtype}"
        )
    if active_token_counts.device != q_all.device:
        raise ValueError("active_token_counts must be on the same device as q_all")
    if active_token_counts.ndim != 1 or active_token_counts.shape[0] != q_all.shape[0]:
        raise ValueError(
            "active_token_counts must be rank-1 with one entry per query row, "
            f"got {tuple(active_token_counts.shape)} for q rows {q_all.shape[0]}"
        )
    if not q_all.is_contiguous():
        raise ValueError("q_all must be contiguous for sparse MLA split decode")
    if not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous for sparse MLA split decode")
    if page_table_1.device != q_all.device:
        raise ValueError("page_table_1 must be on the same device as q_all")
    if page_table_1.dtype != torch.int32:
        raise TypeError(
            f"page_table_1 must have dtype torch.int32, got {page_table_1.dtype}"
        )
    if page_table_1.ndim != 2 or int(page_table_1.shape[0]) != int(q_all.shape[0]):
        raise ValueError(
            f"page_table_1 must have shape [{int(q_all.shape[0])}, width], got {tuple(page_table_1.shape)}"
        )
    if not page_table_1.is_contiguous():
        raise ValueError("page_table_1 must be contiguous for sparse MLA split decode")
    if not active_token_counts.is_contiguous():
        raise ValueError(
            "active_token_counts must be contiguous for sparse MLA split decode"
        )
    if launch_num_chunks <= 0 or launch_num_chunks > _SPLIT_MAX_CHUNKS:
        raise ValueError(
            f"launch_num_chunks must be in [1, {_SPLIT_MAX_CHUNKS}], got {launch_num_chunks}"
        )
    head_tiles = (
        int(tmp_output.shape[1]) + _MLA_HEADS_PER_TILE - 1
    ) // _MLA_HEADS_PER_TILE

    kv_rows_u32, kv_scales = _extract_packed_kv_runtime_views(kv_cache)
    q_u32 = _view_last_dim_as_u32(q_all)
    if sm_scale.shape != (1,) or sm_scale.dtype != torch.float32:
        raise ValueError("sm_scale tensor must have shape (1,) and dtype float32")
    if sm_scale.device != q_all.device:
        raise ValueError("sm_scale tensor must be on the same device as q_all")
    _validate_split_control_tensor(
        kv_chunk_size_ptr,
        name="kv_chunk_size_ptr",
        device=q_all.device,
    )
    _validate_split_control_tensor(
        num_chunks_ptr,
        name="num_chunks_ptr",
        device=q_all.device,
    )
    if tmp_output.device != q_all.device or tmp_lse.device != q_all.device:
        raise ValueError(
            "split MLA scratch buffers must be on the same device as q_all"
        )
    if tmp_lse.dtype != torch.float32:
        raise TypeError(f"tmp_lse must have dtype torch.float32, got {tmp_lse.dtype}")
    if tmp_output.ndim != 4:
        raise ValueError(
            f"tmp_output must have shape [rows, heads, chunks, dim], got {tuple(tmp_output.shape)}"
        )
    if tmp_lse.ndim != 3:
        raise ValueError(
            f"tmp_lse must have shape [rows, heads, chunks], got {tuple(tmp_lse.shape)}"
        )
    _validate_tensor_storage_bounds(tmp_output, name="split MLA tmp_output")
    _validate_tensor_storage_bounds(tmp_lse, name="split MLA tmp_lse")
    if (
        int(tmp_output.shape[0]) < int(q_all.shape[0])
        or int(tmp_output.shape[1]) < int(q_all.shape[1])
        or int(tmp_output.shape[2]) < int(launch_num_chunks)
        or int(tmp_lse.shape[0]) < int(q_all.shape[0])
        or int(tmp_lse.shape[1]) < int(q_all.shape[1])
        or int(tmp_lse.shape[2]) < int(launch_num_chunks)
    ):
        raise ValueError(
            "split MLA scratch buffers are too small: "
            f"tmp_output={tuple(tmp_output.shape)} tmp_lse={tuple(tmp_lse.shape)} "
            f"required rows={int(q_all.shape[0])} heads={int(q_all.shape[1])} "
            f"chunks={int(launch_num_chunks)}"
        )

    forward_kernel = _build_sparse_mla_split_forward_kernel(
        traits,
        int(launch_num_chunks),
        head_tiles,
        bool(identity_page_table),
    )
    forward_args = (
        _to_kernel_tensor(q_u32, cutlass.Uint32, assumed_align=16),
        _to_kernel_tensor(kv_rows_u32, cutlass.Uint32, assumed_align=16),
        _to_kernel_tensor(kv_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(page_table_1, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(active_token_counts, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(sm_scale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(kv_chunk_size_ptr, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(num_chunks_ptr, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(tmp_output, _torch_to_cutlass_dtype(tmp_output.dtype)),
        _to_kernel_tensor(tmp_lse, cutlass.Float32, assumed_align=4),
        current_cuda_stream(),
    )
    _cq = getattr(workspace, "_contract_q", None)
    _ckv, _cks = _workspace_contract_kv_tensors(workspace, kv_cache)
    _cpt = getattr(workspace, "_contract_page_table", None)
    _cnt = getattr(workspace, "_contract_indexer_cache_seqlens", None)
    _cto = getattr(workspace, "_contract_tmp_output", None)
    _ctl = getattr(workspace, "_contract_tmp_lse", None)
    forward_cache_key = (
        _tensor_meta_key(_cq if _cq is not None else q_u32, dynamic_dims=(0,)),
        _tensor_meta_key(_ckv if _ckv is not None else kv_rows_u32),
        _tensor_meta_key(_cks if _cks is not None else kv_scales),
        _tensor_meta_key(
            _cpt if _cpt is not None else page_table_1,
            dynamic_dims=(0,),
        ),
        _tensor_meta_key(
            _cnt if _cnt is not None else active_token_counts,
            dynamic_dims=(0,),
        ),
        _tensor_meta_key(kv_chunk_size_ptr),
        _tensor_meta_key(num_chunks_ptr),
        _tensor_meta_key(
            _cto if _cto is not None else tmp_output,
            dynamic_dims=(0,),
        ),
        _tensor_meta_key(
            _ctl if _ctl is not None else tmp_lse,
            dynamic_dims=(0,),
        ),
        traits,
        int(launch_num_chunks),
        head_tiles,
        str(tmp_output.dtype),
        bool(identity_page_table),
    )
    forward_spec = KernelCompileSpec.from_key(
        "attention.mla.split_forward",
        1,
        forward_cache_key,
        labels=(
            "q",
            "kv_rows",
            "kv_scales",
            "page_table",
            "active_token_counts",
            "kv_chunk_size_ptr",
            "num_chunks_ptr",
            "tmp_output",
            "tmp_lse",
            "traits",
            "launch_num_chunks",
            "head_tiles",
            "tmp_output_dtype",
            "identity_page_table",
        ),
    )
    b12x_launch(
        forward_kernel,
        compile_spec=forward_spec,
        compile_args=forward_args,
        runtime_args=forward_args,
    )


def _compressed_mla_split_decode_forward_flat_launch(
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_k_cache: torch.Tensor,
    indexed_indices: torch.Tensor,
    indexed_lengths: torch.Tensor,
    indexed_page_table: torch.Tensor,
    sm_scale: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    attn_sink_for_kernel: torch.Tensor,
    launch_num_chunks: int,
    swa_page_size: int,
    swa_page_nbytes: int,
    indexed_page_size: int,
    indexed_page_nbytes: int,
    has_swa: bool,
    has_indexed: bool,
    map_indexed_page_table: bool,
    direct_output: bool,
    single_tile_chunks: bool,
    direct_sink_output: bool,
) -> None:
    head_tiles = (
        int(tmp_output.shape[1]) + _MLA_HEADS_PER_TILE - 1
    ) // _MLA_HEADS_PER_TILE
    q_u32 = _view_last_dim_as_u32(q_all)
    swa_u8 = swa_k_cache.reshape(-1)
    indexed_u8 = indexed_k_cache.reshape(-1)
    num_heads = int(q_all.shape[1])
    q_u32_width = int(q_u32.shape[2])
    swa_indices_width = int(swa_indices.shape[1])
    indexed_indices_width = int(indexed_indices.shape[1])
    indexed_page_table_width = int(indexed_page_table.shape[1])
    tmp_output_chunk_capacity = 1 if direct_output else _SPLIT_MAX_CHUNKS
    tmp_lse_chunk_capacity = _SPLIT_MAX_CHUNKS
    swa_cache_nbytes = int(swa_u8.numel())
    indexed_cache_nbytes = int(indexed_u8.numel())

    forward_kernel = _build_compressed_mla_split_forward_kernel(
        1,
        head_tiles,
        int(swa_page_size),
        int(swa_page_nbytes),
        int(indexed_page_size),
        int(indexed_page_nbytes),
        num_heads,
        swa_cache_nbytes,
        indexed_cache_nbytes,
        swa_indices_width,
        indexed_indices_width,
        tmp_output_chunk_capacity,
        tmp_lse_chunk_capacity,
        bool(has_swa),
        bool(has_indexed),
        bool(map_indexed_page_table),
        bool(direct_output),
        bool(single_tile_chunks),
        bool(direct_sink_output),
    )
    forward_args = (
        _gmem_ptr(q_u32, cutlass.Uint32, assumed_align=16),
        _gmem_ptr(swa_u8, cutlass.Uint8, assumed_align=16),
        _gmem_ptr(swa_indices, cutlass.Int32, assumed_align=4),
        _gmem_ptr(swa_lengths, cutlass.Int32, assumed_align=4),
        _gmem_ptr(indexed_u8, cutlass.Uint8, assumed_align=16),
        _gmem_ptr(indexed_indices, cutlass.Int32, assumed_align=4),
        _gmem_ptr(indexed_lengths, cutlass.Int32, assumed_align=4),
        _gmem_ptr(indexed_page_table, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(sm_scale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(kv_chunk_size_ptr, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(num_chunks_ptr, cutlass.Int32, assumed_align=4),
        _gmem_ptr(
            tmp_output,
            _torch_to_cutlass_dtype(tmp_output.dtype),
            assumed_align=16,
        ),
        _gmem_ptr(tmp_lse, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(attn_sink_for_kernel, cutlass.Float32, assumed_align=4),
        indexed_page_table_width,
        int(indexed_page_table.stride(0)),
        int(tmp_output.stride(0)),
        int(tmp_output.stride(1)),
        int(tmp_output.stride(2)),
        int(tmp_output.stride(3)) if tmp_output.ndim == 4 else 1,
        int(tmp_lse.stride(0)),
        int(tmp_lse.stride(1)),
        int(tmp_lse.stride(2)),
        int(launch_num_chunks),
        int(q_all.shape[0]),
        current_cuda_stream(),
    )
    attn_sink_fake_shape = (num_heads,) if direct_sink_output else (1,)
    forward_compile_args = (
        _fake_gmem_ptr(cutlass.Uint32, assumed_align=16),
        _fake_gmem_ptr(cutlass.Uint8, assumed_align=16),
        _fake_gmem_ptr(cutlass.Int32, assumed_align=4),
        _fake_gmem_ptr(cutlass.Int32, assumed_align=4),
        _fake_gmem_ptr(cutlass.Uint8, assumed_align=16),
        _fake_gmem_ptr(cutlass.Int32, assumed_align=4),
        _fake_gmem_ptr(cutlass.Int32, assumed_align=4),
        _fake_gmem_ptr(cutlass.Int32, assumed_align=4),
        _fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4),
        _fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4),
        _fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4),
        _fake_gmem_ptr(
            _torch_to_cutlass_dtype(tmp_output.dtype),
            assumed_align=16,
        ),
        _fake_gmem_ptr(cutlass.Float32, assumed_align=4),
        _fake_compact_tensor(cutlass.Float32, attn_sink_fake_shape, assumed_align=4),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
    )
    forward_cache_key = (
        "compressed_mla_split_forward_ptr",
        (q_all.device.type, q_all.device.index),
        num_heads,
        q_u32_width,
        swa_cache_nbytes,
        indexed_cache_nbytes,
        swa_indices_width,
        indexed_indices_width,
        tmp_output_chunk_capacity,
        tmp_lse_chunk_capacity,
        _tensor_meta_key(kv_chunk_size_ptr),
        _tensor_meta_key(num_chunks_ptr),
        _tensor_meta_key(attn_sink_for_kernel),
        "dynamic",
        head_tiles,
        int(swa_page_size),
        int(swa_page_nbytes),
        int(indexed_page_size),
        int(indexed_page_nbytes),
        bool(has_swa),
        bool(has_indexed),
        bool(map_indexed_page_table),
        str(tmp_output.dtype),
        bool(direct_output),
        bool(single_tile_chunks),
        bool(direct_sink_output),
    )
    forward_spec = KernelCompileSpec.from_key(
        "attention.mla.compressed_split_forward",
        4,
        forward_cache_key,
        labels=(
            "kind",
            "device",
            "num_heads",
            "q_u32_width",
            "swa_cache_nbytes",
            "indexed_cache_nbytes",
            "swa_indices_width",
            "indexed_indices_width",
            "tmp_output_chunk_capacity",
            "tmp_lse_chunk_capacity",
            "kv_chunk_size_ptr",
            "num_chunks_ptr",
            "attn_sink",
            "launch_num_chunks_policy",
            "head_tiles",
            "swa_page_size",
            "swa_page_nbytes",
            "indexed_page_size",
            "indexed_page_nbytes",
            "has_swa",
            "has_indexed",
            "map_indexed_page_table",
            "tmp_output_dtype",
            "direct_output",
            "single_tile_chunks",
            "direct_sink_output",
        ),
    )
    b12x_launch(
        forward_kernel,
        compile_spec=forward_spec,
        compile_args=forward_compile_args,
        runtime_args=forward_args,
    )


@torch.library.custom_op(
    "b12x::compressed_mla_split_decode_forward",
    mutates_args=("tmp_output", "tmp_lse"),
)
def _compressed_mla_split_decode_forward_op(
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_k_cache: torch.Tensor,
    indexed_indices: torch.Tensor,
    indexed_lengths: torch.Tensor,
    indexed_page_table: torch.Tensor,
    sm_scale: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    attn_sink_for_kernel: torch.Tensor,
    launch_num_chunks: int,
    swa_page_size: int,
    swa_page_nbytes: int,
    indexed_page_size: int,
    indexed_page_nbytes: int,
    has_swa: bool,
    has_indexed: bool,
    map_indexed_page_table: bool,
    direct_output: bool,
    single_tile_chunks: bool,
    direct_sink_output: bool,
) -> None:
    _compressed_mla_split_decode_forward_flat_launch(
        q_all,
        swa_k_cache,
        swa_indices,
        swa_lengths,
        indexed_k_cache,
        indexed_indices,
        indexed_lengths,
        indexed_page_table,
        sm_scale,
        kv_chunk_size_ptr,
        num_chunks_ptr,
        tmp_output,
        tmp_lse,
        attn_sink_for_kernel,
        launch_num_chunks,
        swa_page_size,
        swa_page_nbytes,
        indexed_page_size,
        indexed_page_nbytes,
        has_swa,
        has_indexed,
        map_indexed_page_table,
        direct_output,
        single_tile_chunks,
        direct_sink_output,
    )


@_compressed_mla_split_decode_forward_op.register_fake
def _compressed_mla_split_decode_forward_fake(
    q_all: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_k_cache: torch.Tensor,
    indexed_indices: torch.Tensor,
    indexed_lengths: torch.Tensor,
    indexed_page_table: torch.Tensor,
    sm_scale: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    attn_sink_for_kernel: torch.Tensor,
    launch_num_chunks: int,
    swa_page_size: int,
    swa_page_nbytes: int,
    indexed_page_size: int,
    indexed_page_nbytes: int,
    has_swa: bool,
    has_indexed: bool,
    map_indexed_page_table: bool,
    direct_output: bool,
    single_tile_chunks: bool,
    direct_sink_output: bool,
) -> None:
    return None


def _sparse_mla_split_decode_merge_flat_launch(
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    contract_tmp_output: torch.Tensor,
    contract_tmp_lse: torch.Tensor,
    contract_output: torch.Tensor,
    has_attn_sink: bool,
) -> None:
    if not has_attn_sink:
        merge_kernel = _build_sparse_mla_split_merge_kernel()
        merge_args = (
            _to_kernel_tensor(tmp_output, _torch_to_cutlass_dtype(tmp_output.dtype)),
            _to_kernel_tensor(tmp_lse, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(num_chunks_ptr, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
            current_cuda_stream(),
        )
        merge_cache_key = (
            _tensor_compile_key(
                "tmp_output",
                contract_tmp_output,
                dynamic_dims=(0, 2),
                dynamic_strides=(2,),
            ),
            _tensor_compile_key(
                "tmp_lse",
                contract_tmp_lse,
                dynamic_dims=(0, 2),
                dynamic_strides=(0, 1),
            ),
            _tensor_meta_key(num_chunks_ptr),
            _tensor_compile_key(
                "output",
                contract_output,
                dynamic_dims=(0,),
            ),
            str(tmp_output.dtype),
            str(output.dtype),
        )
        merge_spec = KernelCompileSpec.from_key(
            "attention.mla.split_merge",
            3,
            merge_cache_key,
            labels=(
                "tmp_output",
                "tmp_lse",
                "num_chunks_ptr",
                "output",
                "tmp_output_dtype",
                "output_dtype",
            ),
        )
        b12x_launch(
            merge_kernel,
            compile_spec=merge_spec,
            compile_args=merge_args,
            runtime_args=merge_args,
        )
        return

    merge_kernel = _build_sparse_mla_split_sink_merge_kernel()
    merge_args = (
        _to_kernel_tensor(tmp_output, _torch_to_cutlass_dtype(tmp_output.dtype)),
        _to_kernel_tensor(tmp_lse, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(num_chunks_ptr, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(attn_sink, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
        current_cuda_stream(),
    )
    merge_cache_key = (
        _tensor_compile_key(
            "tmp_output",
            contract_tmp_output,
            dynamic_dims=(0, 2),
            dynamic_strides=(2,),
        ),
        _tensor_compile_key(
            "tmp_lse",
            contract_tmp_lse,
            dynamic_dims=(0, 2),
            dynamic_strides=(0, 1),
        ),
        _tensor_meta_key(num_chunks_ptr),
        _tensor_meta_key(attn_sink),
        _tensor_compile_key(
            "output",
            contract_output,
            dynamic_dims=(0,),
        ),
        str(tmp_output.dtype),
        str(output.dtype),
        "attn_sink",
    )
    merge_spec = KernelCompileSpec.from_key(
        "attention.mla.split_sink_merge",
        3,
        merge_cache_key,
        labels=(
            "tmp_output",
            "tmp_lse",
            "num_chunks_ptr",
            "attn_sink",
            "output",
            "tmp_output_dtype",
            "output_dtype",
            "kind",
        ),
    )
    b12x_launch(
        merge_kernel,
        compile_spec=merge_spec,
        compile_args=merge_args,
        runtime_args=merge_args,
    )


@torch.library.custom_op(
    "b12x::sparse_mla_split_decode_merge",
    mutates_args=("output",),
)
def _sparse_mla_split_decode_merge_op(
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    contract_tmp_output: torch.Tensor,
    contract_tmp_lse: torch.Tensor,
    contract_output: torch.Tensor,
    has_attn_sink: bool,
) -> None:
    _sparse_mla_split_decode_merge_flat_launch(
        tmp_output,
        tmp_lse,
        num_chunks_ptr,
        output,
        attn_sink,
        contract_tmp_output,
        contract_tmp_lse,
        contract_output,
        has_attn_sink,
    )


@_sparse_mla_split_decode_merge_op.register_fake
def _sparse_mla_split_decode_merge_fake(
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    contract_tmp_output: torch.Tensor,
    contract_tmp_lse: torch.Tensor,
    contract_output: torch.Tensor,
    has_attn_sink: bool,
) -> None:
    return None


def run_compressed_mla_split_decode_forward(
    *,
    q_all: torch.Tensor | None = None,
    swa_k_cache: torch.Tensor | None = None,
    swa_indices: torch.Tensor | None = None,
    swa_lengths: torch.Tensor | None = None,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_lengths: torch.Tensor | None = None,
    indexed_page_table: torch.Tensor | None = None,
    sm_scale: torch.Tensor | None = None,
    kv_chunk_size_ptr: torch.Tensor | None = None,
    num_chunks_ptr: torch.Tensor | None = None,
    tmp_output: torch.Tensor | None = None,
    tmp_lse: torch.Tensor | None = None,
    launch_num_chunks: int | None = None,
    swa_page_size: int | None = None,
    swa_page_nbytes: int | None = None,
    indexed_page_size: int | None = None,
    indexed_page_nbytes: int | None = None,
    has_indexed: bool | None = None,
    map_indexed_page_table: bool | None = None,
    workspace: object | None = None,
    direct_output: bool | None = None,
    single_tile_chunks: bool | None = None,
    attn_sink: torch.Tensor | None = None,
    direct_sink_output: bool | None = None,
    binding: CompressedMLASplitDecodeForwardBinding | None = None,
) -> None:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q_all", q_all),
                ("swa_k_cache", swa_k_cache),
                ("swa_indices", swa_indices),
                ("swa_lengths", swa_lengths),
                ("indexed_k_cache", indexed_k_cache),
                ("indexed_indices", indexed_indices),
                ("indexed_lengths", indexed_lengths),
                ("indexed_page_table", indexed_page_table),
                ("sm_scale", sm_scale),
                ("kv_chunk_size_ptr", kv_chunk_size_ptr),
                ("num_chunks_ptr", num_chunks_ptr),
                ("tmp_output", tmp_output),
                ("tmp_lse", tmp_lse),
                ("launch_num_chunks", launch_num_chunks),
                ("swa_page_size", swa_page_size),
                ("swa_page_nbytes", swa_page_nbytes),
                ("indexed_page_size", indexed_page_size),
                ("indexed_page_nbytes", indexed_page_nbytes),
                ("has_indexed", has_indexed),
                ("map_indexed_page_table", map_indexed_page_table),
                ("workspace", workspace),
                ("direct_output", direct_output),
                ("single_tile_chunks", single_tile_chunks),
                ("attn_sink", attn_sink),
                ("direct_sink_output", direct_sink_output),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_compressed_mla_split_decode_forward", extras)
        q_all = binding.q_all
        swa_k_cache = binding.swa_k_cache
        swa_indices = binding.swa_indices
        swa_lengths = binding.swa_lengths
        indexed_k_cache = binding.indexed_k_cache
        indexed_indices = binding.indexed_indices
        indexed_lengths = binding.indexed_lengths
        indexed_page_table = binding.indexed_page_table
        sm_scale = binding.sm_scale
        kv_chunk_size_ptr = binding.kv_chunk_size_ptr
        num_chunks_ptr = binding.num_chunks_ptr
        tmp_output = binding.tmp_output
        tmp_lse = binding.tmp_lse
        launch_num_chunks = binding.launch_num_chunks
        swa_page_size = binding.swa_page_size
        swa_page_nbytes = binding.swa_page_nbytes
        indexed_page_size = binding.indexed_page_size
        indexed_page_nbytes = binding.indexed_page_nbytes
        has_indexed = binding.has_indexed
        map_indexed_page_table = binding.map_indexed_page_table
        workspace = binding.scratch
        direct_output = binding.direct_output
        single_tile_chunks = binding.single_tile_chunks
        attn_sink = binding.attn_sink
        direct_sink_output = binding.direct_sink_output

    q_all = _require_bound_arg(
        q_all, api_name="run_compressed_mla_split_decode_forward", name="q_all"
    )
    swa_k_cache = _require_bound_arg(
        swa_k_cache,
        api_name="run_compressed_mla_split_decode_forward",
        name="swa_k_cache",
    )
    swa_indices = _require_bound_arg(
        swa_indices,
        api_name="run_compressed_mla_split_decode_forward",
        name="swa_indices",
    )
    swa_lengths = _require_bound_arg(
        swa_lengths,
        api_name="run_compressed_mla_split_decode_forward",
        name="swa_lengths",
    )
    indexed_k_cache = _require_bound_arg(
        indexed_k_cache,
        api_name="run_compressed_mla_split_decode_forward",
        name="indexed_k_cache",
    )
    indexed_indices = _require_bound_arg(
        indexed_indices,
        api_name="run_compressed_mla_split_decode_forward",
        name="indexed_indices",
    )
    indexed_lengths = _require_bound_arg(
        indexed_lengths,
        api_name="run_compressed_mla_split_decode_forward",
        name="indexed_lengths",
    )
    indexed_page_table = _require_bound_arg(
        indexed_page_table,
        api_name="run_compressed_mla_split_decode_forward",
        name="indexed_page_table",
    )
    sm_scale = _require_bound_arg(
        sm_scale,
        api_name="run_compressed_mla_split_decode_forward",
        name="sm_scale",
    )
    kv_chunk_size_ptr = _require_bound_arg(
        kv_chunk_size_ptr,
        api_name="run_compressed_mla_split_decode_forward",
        name="kv_chunk_size_ptr",
    )
    num_chunks_ptr = _require_bound_arg(
        num_chunks_ptr,
        api_name="run_compressed_mla_split_decode_forward",
        name="num_chunks_ptr",
    )
    tmp_output = _require_bound_arg(
        tmp_output,
        api_name="run_compressed_mla_split_decode_forward",
        name="tmp_output",
    )
    tmp_lse = _require_bound_arg(
        tmp_lse,
        api_name="run_compressed_mla_split_decode_forward",
        name="tmp_lse",
    )
    launch_num_chunks = int(
        _require_bound_arg(
            launch_num_chunks,
            api_name="run_compressed_mla_split_decode_forward",
            name="launch_num_chunks",
        )
    )
    swa_page_size = int(
        _require_bound_arg(
            swa_page_size,
            api_name="run_compressed_mla_split_decode_forward",
            name="swa_page_size",
        )
    )
    swa_page_nbytes = int(
        _require_bound_arg(
            swa_page_nbytes,
            api_name="run_compressed_mla_split_decode_forward",
            name="swa_page_nbytes",
        )
    )
    indexed_page_size = int(
        _require_bound_arg(
            indexed_page_size,
            api_name="run_compressed_mla_split_decode_forward",
            name="indexed_page_size",
        )
    )
    indexed_page_nbytes = int(
        _require_bound_arg(
            indexed_page_nbytes,
            api_name="run_compressed_mla_split_decode_forward",
            name="indexed_page_nbytes",
        )
    )
    has_indexed = bool(
        _require_bound_arg(
            has_indexed,
            api_name="run_compressed_mla_split_decode_forward",
            name="has_indexed",
        )
    )
    map_indexed_page_table = bool(
        _require_bound_arg(
            map_indexed_page_table,
            api_name="run_compressed_mla_split_decode_forward",
            name="map_indexed_page_table",
        )
    )
    direct_output = False if direct_output is None else bool(direct_output)
    single_tile_chunks = (
        False if single_tile_chunks is None else bool(single_tile_chunks)
    )
    direct_sink_output = (
        False if direct_sink_output is None else bool(direct_sink_output)
    )

    if q_all.device.type != "cuda":
        raise ValueError("compressed MLA split decode requires CUDA q_all")
    if q_all.dtype != torch.bfloat16:
        raise TypeError(f"q_all must have dtype torch.bfloat16, got {q_all.dtype}")
    if q_all.ndim != 3 or int(q_all.shape[-1]) != _COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(
            f"q_all must have shape [rows, heads, {_COMPRESSED_MLA_HEAD_DIM}], got {tuple(q_all.shape)}"
        )
    if not q_all.is_contiguous():
        raise ValueError("q_all must be contiguous for compressed MLA split decode")
    for name, cache in (
        ("swa_k_cache", swa_k_cache),
        ("indexed_k_cache", indexed_k_cache),
    ):
        if cache.device != q_all.device:
            raise ValueError(f"{name} must be on the same device as q_all")
    swa_k_cache = _compressed_mla_cache_byte_view(swa_k_cache, name="swa_k_cache")
    indexed_k_cache = _compressed_mla_cache_byte_view(
        indexed_k_cache, name="indexed_k_cache"
    )
    if int(swa_page_size) <= 0 or int(indexed_page_size) <= 0:
        raise ValueError(
            f"compressed MLA page sizes must be positive, got swa={swa_page_size} indexed={indexed_page_size}"
        )
    expected_swa_nbytes = compressed_mla_page_nbytes(int(swa_page_size))
    expected_indexed_nbytes = compressed_mla_page_nbytes(int(indexed_page_size))
    if int(swa_page_nbytes) != expected_swa_nbytes:
        raise ValueError(
            f"swa_page_nbytes must be {expected_swa_nbytes} for page_size {int(swa_page_size)}, got {swa_page_nbytes}"
        )
    if int(indexed_page_nbytes) != expected_indexed_nbytes:
        raise ValueError(
            "indexed_page_nbytes must be "
            f"{expected_indexed_nbytes} for page_size {int(indexed_page_size)}, got {indexed_page_nbytes}"
        )
    if int(swa_k_cache.shape[1]) != expected_swa_nbytes:
        raise ValueError(
            f"swa_k_cache page byte width must be {expected_swa_nbytes}, got {int(swa_k_cache.shape[1])}"
        )
    if int(indexed_k_cache.shape[1]) != expected_indexed_nbytes:
        raise ValueError(
            "indexed_k_cache page byte width must be "
            f"{expected_indexed_nbytes}, got {int(indexed_k_cache.shape[1])}"
        )
    rows = int(q_all.shape[0])
    for name, tensor in (
        ("swa_indices", swa_indices),
        ("indexed_indices", indexed_indices),
    ):
        if tensor.device != q_all.device:
            raise ValueError(f"{name} must be on the same device as q_all")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if (
            tensor.ndim != 2
            or int(tensor.shape[0]) != rows
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous with shape [{rows}, width]")
    if indexed_page_table.device != q_all.device:
        raise ValueError("indexed_page_table must be on the same device as q_all")
    if indexed_page_table.dtype != torch.int32:
        raise TypeError(
            f"indexed_page_table must have dtype torch.int32, got {indexed_page_table.dtype}"
        )
    if indexed_page_table.ndim != 2 or int(indexed_page_table.shape[0]) != rows:
        raise ValueError(f"indexed_page_table must have shape [{rows}, width]")
    if not indexed_page_table.is_contiguous():
        if (
            int(indexed_page_table.stride(0)) != 0
            or int(indexed_page_table.stride(1)) != 1
        ):
            raise ValueError(
                "indexed_page_table must be contiguous or row-shared with stride (0, 1)"
            )
        _validate_tensor_storage_bounds(
            indexed_page_table, name="compressed MLA indexed_page_table"
        )
    for name, tensor in (
        ("swa_lengths", swa_lengths),
        ("indexed_lengths", indexed_lengths),
    ):
        if tensor.device != q_all.device:
            raise ValueError(f"{name} must be on the same device as q_all")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if tensor.shape != (rows,) or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous with shape [{rows}]")
    if tmp_output.dtype != torch.bfloat16 or tmp_lse.dtype != torch.float32:
        raise TypeError("tmp_output must be BF16 and tmp_lse must be FP32")
    expected_tmp_rank = 3 if direct_output else 4
    expected_tmp_shape = (
        f"[rows, heads, {_COMPRESSED_MLA_HEAD_DIM}]"
        if direct_output
        else f"[rows, heads, chunks, {_COMPRESSED_MLA_HEAD_DIM}]"
    )
    if (
        tmp_output.ndim != expected_tmp_rank
        or int(tmp_output.shape[-1]) != _COMPRESSED_MLA_HEAD_DIM
    ):
        raise ValueError(
            f"tmp_output must have shape {expected_tmp_shape}, got {tuple(tmp_output.shape)}"
        )
    if tmp_lse.ndim != 3:
        raise ValueError("tmp_lse must have shape [rows, heads, chunks]")
    if tmp_output.device != q_all.device or tmp_lse.device != q_all.device:
        raise ValueError(
            "compressed MLA scratch/output buffers must be on the same device as q_all"
        )
    _validate_tensor_storage_bounds(tmp_output, name="compressed MLA tmp_output")
    _validate_tensor_storage_bounds(tmp_lse, name="compressed MLA tmp_lse")
    if launch_num_chunks <= 0 or launch_num_chunks > _SPLIT_MAX_CHUNKS:
        raise ValueError(
            f"launch_num_chunks must be in [1, {_SPLIT_MAX_CHUNKS}], got {launch_num_chunks}"
        )
    if direct_output:
        if (
            int(tmp_output.shape[0]) < rows
            or int(tmp_output.shape[1]) < int(q_all.shape[1])
            or int(tmp_output.shape[2]) < _COMPRESSED_MLA_HEAD_DIM
        ):
            raise ValueError(
                "compressed MLA direct output is too small: "
                f"buffer={tuple(tmp_output.shape)} required>=({rows}, {int(q_all.shape[1])}, {_COMPRESSED_MLA_HEAD_DIM})"
            )
    elif (
        int(tmp_output.shape[0]) < rows
        or int(tmp_output.shape[1]) < int(q_all.shape[1])
        or int(tmp_output.shape[2]) < int(launch_num_chunks)
        or int(tmp_output.shape[3]) < _COMPRESSED_MLA_HEAD_DIM
    ):
        raise ValueError(
            "compressed MLA split output is too small: "
            f"buffer={tuple(tmp_output.shape)} required>=({rows}, {int(q_all.shape[1])}, "
            f"{int(launch_num_chunks)}, {_COMPRESSED_MLA_HEAD_DIM})"
        )
    if (
        int(tmp_lse.shape[0]) < rows
        or int(tmp_lse.shape[1]) < int(q_all.shape[1])
        or int(tmp_lse.shape[2]) < int(launch_num_chunks)
    ):
        raise ValueError(
            "compressed MLA tmp_lse is too small: "
            f"buffer={tuple(tmp_lse.shape)} required>=({rows}, {int(q_all.shape[1])}, {int(launch_num_chunks)})"
        )
    if sm_scale.shape != (1,) or sm_scale.dtype != torch.float32:
        raise ValueError("sm_scale tensor must have shape (1,) and dtype float32")
    if sm_scale.device != q_all.device:
        raise ValueError("sm_scale must be on the same device as q_all")
    _validate_split_control_tensor(
        kv_chunk_size_ptr,
        name="kv_chunk_size_ptr",
        device=q_all.device,
    )
    _validate_split_control_tensor(
        num_chunks_ptr,
        name="num_chunks_ptr",
        device=q_all.device,
    )
    if direct_sink_output:
        if not direct_output:
            raise ValueError("direct_sink_output requires direct_output=True")
        if attn_sink is None:
            raise ValueError("direct_sink_output requires attn_sink")
        if attn_sink.device != q_all.device:
            raise ValueError("attn_sink must be on the same device as q_all")
        if attn_sink.dtype != torch.float32:
            raise TypeError(
                f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}"
            )
        if attn_sink.ndim != 1 or int(attn_sink.shape[0]) != int(tmp_output.shape[1]):
            raise ValueError(
                f"attn_sink must have shape ({int(tmp_output.shape[1])},), got {tuple(attn_sink.shape)}"
            )
        if not attn_sink.is_contiguous():
            raise ValueError("attn_sink must be contiguous")
    attn_sink_for_kernel = attn_sink if attn_sink is not None else sm_scale
    has_swa = int(swa_indices.shape[1]) > 0

    torch.ops.b12x.compressed_mla_split_decode_forward(
        q_all,
        swa_k_cache,
        swa_indices,
        swa_lengths,
        indexed_k_cache,
        indexed_indices,
        indexed_lengths,
        indexed_page_table,
        sm_scale,
        kv_chunk_size_ptr,
        num_chunks_ptr,
        tmp_output,
        tmp_lse,
        attn_sink_for_kernel,
        int(launch_num_chunks),
        int(swa_page_size),
        int(swa_page_nbytes),
        int(indexed_page_size),
        int(indexed_page_nbytes),
        bool(has_swa),
        bool(has_indexed),
        bool(map_indexed_page_table),
        bool(direct_output),
        bool(single_tile_chunks),
        bool(direct_sink_output),
    )


def run_sparse_mla_split_decode_merge(
    *,
    tmp_output: torch.Tensor | None = None,
    tmp_lse: torch.Tensor | None = None,
    num_chunks_ptr: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    workspace: object | None = None,
    binding: SparseMLASplitDecodeMergeBinding | None = None,
) -> None:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("tmp_output", tmp_output),
                ("tmp_lse", tmp_lse),
                ("num_chunks_ptr", num_chunks_ptr),
                ("output", output),
                ("attn_sink", attn_sink),
                ("workspace", workspace),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_sparse_mla_split_decode_merge", extras)
        tmp_output = binding.tmp_output
        tmp_lse = binding.tmp_lse
        num_chunks_ptr = binding.num_chunks_ptr
        output = binding.output
        attn_sink = binding.attn_sink
        workspace = binding.scratch

    tmp_output = _require_bound_arg(
        tmp_output,
        api_name="run_sparse_mla_split_decode_merge",
        name="tmp_output",
    )
    tmp_lse = _require_bound_arg(
        tmp_lse,
        api_name="run_sparse_mla_split_decode_merge",
        name="tmp_lse",
    )
    num_chunks_ptr = _require_bound_arg(
        num_chunks_ptr,
        api_name="run_sparse_mla_split_decode_merge",
        name="num_chunks_ptr",
    )
    output = _require_bound_arg(
        output,
        api_name="run_sparse_mla_split_decode_merge",
        name="output",
    )

    if tmp_output.device != output.device or tmp_lse.device != output.device:
        raise ValueError("split MLA merge tensors must be on the same device")
    if tmp_lse.dtype != torch.float32:
        raise TypeError(f"tmp_lse must have dtype torch.float32, got {tmp_lse.dtype}")
    if tmp_output.dtype != output.dtype:
        raise TypeError(
            f"tmp_output dtype {tmp_output.dtype} must match output dtype {output.dtype}"
        )
    if tmp_output.ndim != 4:
        raise ValueError(
            f"tmp_output must have shape [rows, heads, chunks, dim], got {tuple(tmp_output.shape)}"
        )
    if tmp_lse.ndim != 3:
        raise ValueError(
            f"tmp_lse must have shape [rows, heads, chunks], got {tuple(tmp_lse.shape)}"
        )
    if output.ndim != 3:
        raise ValueError(
            f"output must have shape [rows, heads, dim], got {tuple(output.shape)}"
        )
    if (
        int(tmp_output.shape[0]) < int(output.shape[0])
        or int(tmp_output.shape[1]) < int(output.shape[1])
        or int(tmp_output.shape[3]) < int(output.shape[2])
        or int(tmp_lse.shape[0]) < int(output.shape[0])
        or int(tmp_lse.shape[1]) < int(output.shape[1])
        or int(tmp_lse.shape[2]) < int(tmp_output.shape[2])
    ):
        raise ValueError(
            "split MLA merge scratch/output shapes are inconsistent: "
            f"tmp_output={tuple(tmp_output.shape)} tmp_lse={tuple(tmp_lse.shape)} "
            f"output={tuple(output.shape)}"
        )
    _validate_tensor_storage_bounds(tmp_output, name="split MLA merge tmp_output")
    _validate_tensor_storage_bounds(tmp_lse, name="split MLA merge tmp_lse")
    _validate_tensor_storage_bounds(output, name="split MLA merge output")
    _validate_split_control_tensor(
        num_chunks_ptr,
        name="num_chunks_ptr",
        device=output.device,
    )
    _cto = getattr(workspace, "_contract_tmp_output", None)
    _ctl = getattr(workspace, "_contract_tmp_lse", None)
    _co = getattr(workspace, "_contract_output", None)

    has_attn_sink = attn_sink is not None
    if has_attn_sink:
        attn_sink = attn_sink.detach()
        if attn_sink.dtype != torch.float32:
            raise ValueError(
                f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}"
            )
        if attn_sink.device != output.device:
            raise ValueError("attn_sink must be on the same CUDA device as output")
        if attn_sink.ndim != 1 or int(attn_sink.shape[0]) != int(output.shape[1]):
            raise ValueError(
                f"attn_sink must have shape ({int(output.shape[1])},), got {tuple(attn_sink.shape)}"
            )
        if not attn_sink.is_contiguous():
            raise ValueError(
                "attn_sink must be contiguous for the fused split-merge path"
            )

    torch.ops.b12x.sparse_mla_split_decode_merge(
        tmp_output,
        tmp_lse,
        num_chunks_ptr,
        output,
        attn_sink if attn_sink is not None else tmp_lse,
        _cto if _cto is not None else tmp_output,
        _ctl if _ctl is not None else tmp_lse,
        _co if _co is not None else output,
        bool(has_attn_sink),
    )


def run_sparse_mla_split_decode(
    *,
    q_all: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    page_table_1: torch.Tensor | None = None,
    active_token_counts: torch.Tensor | None = None,
    sm_scale: torch.Tensor | None = None,
    kv_chunk_size_ptr: torch.Tensor | None = None,
    num_chunks_ptr: torch.Tensor | None = None,
    tmp_output: torch.Tensor | None = None,
    tmp_lse: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    launch_num_chunks: int | None = None,
    attn_sink: torch.Tensor | None = None,
    workspace: object | None = None,
    identity_page_table: bool | None = None,
    binding: SparseMLASplitDecodeBinding | None = None,
) -> None:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q_all", q_all),
                ("kv_cache", kv_cache),
                ("page_table_1", page_table_1),
                ("active_token_counts", active_token_counts),
                ("sm_scale", sm_scale),
                ("kv_chunk_size_ptr", kv_chunk_size_ptr),
                ("num_chunks_ptr", num_chunks_ptr),
                ("tmp_output", tmp_output),
                ("tmp_lse", tmp_lse),
                ("output", output),
                ("launch_num_chunks", launch_num_chunks),
                ("attn_sink", attn_sink),
                ("workspace", workspace),
                ("identity_page_table", identity_page_table),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_sparse_mla_split_decode", extras)
        q_all = binding.q_all
        kv_cache = binding.kv_cache
        page_table_1 = binding.page_table_1
        active_token_counts = binding.active_token_counts
        sm_scale = binding.sm_scale
        kv_chunk_size_ptr = binding.kv_chunk_size_ptr
        num_chunks_ptr = binding.num_chunks_ptr
        tmp_output = binding.tmp_output
        tmp_lse = binding.tmp_lse
        output = binding.output
        launch_num_chunks = binding.launch_num_chunks
        attn_sink = binding.attn_sink
        workspace = binding.scratch
        identity_page_table = binding.identity_page_table

    q_all = _require_bound_arg(
        q_all, api_name="run_sparse_mla_split_decode", name="q_all"
    )
    kv_cache = _require_bound_arg(
        kv_cache, api_name="run_sparse_mla_split_decode", name="kv_cache"
    )
    page_table_1 = _require_bound_arg(
        page_table_1,
        api_name="run_sparse_mla_split_decode",
        name="page_table_1",
    )
    active_token_counts = _require_bound_arg(
        active_token_counts,
        api_name="run_sparse_mla_split_decode",
        name="active_token_counts",
    )
    sm_scale = _require_bound_arg(
        sm_scale, api_name="run_sparse_mla_split_decode", name="sm_scale"
    )
    kv_chunk_size_ptr = _require_bound_arg(
        kv_chunk_size_ptr,
        api_name="run_sparse_mla_split_decode",
        name="kv_chunk_size_ptr",
    )
    num_chunks_ptr = _require_bound_arg(
        num_chunks_ptr,
        api_name="run_sparse_mla_split_decode",
        name="num_chunks_ptr",
    )
    tmp_output = _require_bound_arg(
        tmp_output,
        api_name="run_sparse_mla_split_decode",
        name="tmp_output",
    )
    tmp_lse = _require_bound_arg(
        tmp_lse, api_name="run_sparse_mla_split_decode", name="tmp_lse"
    )
    output = _require_bound_arg(
        output, api_name="run_sparse_mla_split_decode", name="output"
    )
    launch_num_chunks = int(
        _require_bound_arg(
            launch_num_chunks,
            api_name="run_sparse_mla_split_decode",
            name="launch_num_chunks",
        )
    )
    identity_page_table = (
        False if identity_page_table is None else bool(identity_page_table)
    )

    run_sparse_mla_split_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        launch_num_chunks=launch_num_chunks,
        workspace=workspace,
        identity_page_table=identity_page_table,
    )
    run_sparse_mla_split_decode_merge(
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        num_chunks_ptr=num_chunks_ptr,
        output=output,
        attn_sink=attn_sink,
        workspace=workspace,
    )
