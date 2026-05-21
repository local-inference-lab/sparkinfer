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

from b12x.attention import utils as attention_utils
from b12x.cute.fp4 import shared_ptr_to_u32
from b12x.cute.utils import current_cuda_stream

from .kernel import (
    _MLA_GROUP_SIZE,
    _MLA_HEADS_PER_TILE,
    _MLA_KV_STAGE_BYTES,
    _MLA_NOPE_DIM,
    _MLA_OUTPUT_FRAGMENTS_PER_LANE,
    _MLA_Q_STAGE_BYTES,
    _MLA_Q_GROUP_STAGE_BYTES,
    _MLA_SCALE_GROUPS,
    _MLA_SCALE_STAGE_ELEMS,
    _MLA_TOKEN_TILE,
    _MLA_WARP_THREADS,
    _extract_packed_kv_runtime_views,
    _exp2_approx_ftz_f32,
    _log2_approx_ftz_f32,
    _clamp_active_token_count,
    _run_cached_host_launcher,
    _run_one_pass_sparse_mla_tile,
    _tensor_meta_key,
    _to_kernel_tensor,
    _torch_to_cutlass_dtype,
    _view_last_dim_as_u32,
    _workspace_contract_kv_tensors,
    get_sparse_mla_shared_storage_cls,
)
from .traits import SparseMLATraits, select_sparse_mla_traits


_SPLIT_CHUNK_LADDER = (32, 64, 128, 256, 512, 1024)
_SPLIT_MAX_CHUNKS = 64
_SPLIT_MAX_WIDTH = _SPLIT_CHUNK_LADDER[-1] * _SPLIT_MAX_CHUNKS


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


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
            cute.struct.MemRange[cutlass.Float32, _MLA_SCALE_STAGE_ELEMS],
            16,
        ],
    }
    return cute.struct(SharedStorage)


@dataclass(frozen=True)
class SparseMLASplitDecodeConfig:
    chunk_size: int
    num_chunks: int


def default_sparse_mla_split_decode_config_for_width(
    width: int,
    *,
    max_chunks: int = _SPLIT_MAX_CHUNKS,
) -> SparseMLASplitDecodeConfig | None:
    if width <= _SPLIT_CHUNK_LADDER[0] or width > _SPLIT_MAX_WIDTH:
        return None

    max_chunks = max(1, min(int(max_chunks), _SPLIT_MAX_CHUNKS))
    for chunk_size in _SPLIT_CHUNK_LADDER:
        num_chunks = _ceil_div(width, chunk_size)
        if num_chunks <= max_chunks:
            return SparseMLASplitDecodeConfig(chunk_size=chunk_size, num_chunks=num_chunks)
    return None


def forced_sparse_mla_split_decode_config_for_width(
    width: int,
    *,
    max_chunks: int = _SPLIT_MAX_CHUNKS,
) -> SparseMLASplitDecodeConfig | None:
    if width <= 0 or width > _SPLIT_MAX_WIDTH:
        return None

    max_chunks = max(1, min(int(max_chunks), _SPLIT_MAX_CHUNKS))
    for chunk_size in _SPLIT_CHUNK_LADDER:
        num_chunks = _ceil_div(width, chunk_size)
        if num_chunks <= max_chunks:
            return SparseMLASplitDecodeConfig(chunk_size=chunk_size, num_chunks=num_chunks)
    return None


@cute.jit
def _split_output_lane_view(
    tmp_output: cute.Tensor,
    q_idx: Int32,
    head_idx: Int32,
    out_base: Int32,
) -> cute.Tensor:
    return cute.make_tensor(
        attention_utils.elem_pointer(tmp_output, (q_idx, head_idx, Int32(0), out_base)),
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
        attention_utils.elem_pointer(tmp_lse, (q_idx, head_idx, Int32(0))),
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
    return default_sparse_mla_split_decode_config_for_width(width, max_chunks=max_chunks)


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
                    tmp_output[q_idx, head_idx, chunk_idx, dim_base + Int32(0)] = Float32(
                        0.0
                    ).to(tmp_output.element_type)
                    tmp_output[q_idx, head_idx, chunk_idx, dim_base + Int32(1)] = Float32(
                        0.0
                    ).to(tmp_output.element_type)
                    tmp_output[q_idx, head_idx, chunk_idx, dim_base + Int32(8)] = Float32(
                        0.0
                    ).to(tmp_output.element_type)
                    tmp_output[q_idx, head_idx, chunk_idx, dim_base + Int32(9)] = Float32(
                        0.0
                    ).to(tmp_output.element_type)
            if lane % Int32(4) == Int32(0):
                tmp_lse[q_idx, head_idx, chunk_idx] = Float32(-Float32.inf)


class SparseMLASplitDecodeForwardKernel:
    """Chunk-local sparse MLA partial forward for decode."""

    def __init__(self, launch_num_chunks: int, head_tiles: int, identity_page_table: bool = False):
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
            _zero_partial_head_tile(tmp_output, tmp_lse, q_idx, chunk_idx, head_tile_start, lane)
        else:
            token_end = token_start + chunk_size
            if token_end > row_token_end:
                token_end = row_token_end

            smem = cutlass.utils.SmemAllocator()
            SharedStorage = get_sparse_mla_split_shared_storage_cls()
            storage = smem.allocate(SharedStorage)
            sTokenIdx = storage.token_idx.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
            sScale = storage.token_scale_a.get_tensor(
                cute.make_layout((_MLA_TOKEN_TILE * _MLA_SCALE_GROUPS,), stride=(1,))
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
                Float32(sm_scale[Int32(0)] * attention_utils.LOG2_E),
                lane,
                tmp_output,
                q_idx,
                chunk_idx,
                tmp_lse,
                self.identity_page_table,
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
                new_m = attention_utils.fmax(merged_m, part_lse)
                prev_scale = _exp2_approx_ftz_f32(merged_m - new_m)
                part_scale = _exp2_approx_ftz_f32(part_lse - new_m)
                merged_d = Float32(merged_d * prev_scale + part_scale)
                acc[0] = Float32(
                    acc[0] * prev_scale + Float32(tmp_output_lane[chunk_idx, Int32(0)]) * part_scale
                )
                acc[1] = Float32(
                    acc[1] * prev_scale + Float32(tmp_output_lane[chunk_idx, Int32(1)]) * part_scale
                )
                acc[2] = Float32(
                    acc[2] * prev_scale + Float32(tmp_output_lane[chunk_idx, Int32(2)]) * part_scale
                )
                acc[3] = Float32(
                    acc[3] * prev_scale + Float32(tmp_output_lane[chunk_idx, Int32(3)]) * part_scale
                )
                merged_m = Float32(new_m)
            chunk_idx += Int32(1)

        if merged_m == Float32(-Float32.inf):
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(0.0).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(0.0).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(0.0).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(0.0).to(output.element_type)
        else:
            inv_d = cute.arch.rcp_approx(merged_d)
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(acc[0] * inv_d).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(acc[1] * inv_d).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(acc[2] * inv_d).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(acc[3] * inv_d).to(output.element_type)


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
                new_m = attention_utils.fmax(merged_m, part_lse)
                prev_scale = _exp2_approx_ftz_f32(merged_m - new_m)
                part_scale = _exp2_approx_ftz_f32(part_lse - new_m)
                merged_d = Float32(merged_d * prev_scale + part_scale)
                acc[0] = Float32(
                    acc[0] * prev_scale + Float32(tmp_output_lane[chunk_idx, Int32(0)]) * part_scale
                )
                acc[1] = Float32(
                    acc[1] * prev_scale + Float32(tmp_output_lane[chunk_idx, Int32(1)]) * part_scale
                )
                acc[2] = Float32(
                    acc[2] * prev_scale + Float32(tmp_output_lane[chunk_idx, Int32(2)]) * part_scale
                )
                acc[3] = Float32(
                    acc[3] * prev_scale + Float32(tmp_output_lane[chunk_idx, Int32(3)]) * part_scale
                )
                merged_m = Float32(new_m)
            chunk_idx += Int32(1)

        if merged_m == Float32(-Float32.inf):
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(0.0).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(0.0).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(0.0).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(0.0).to(output.element_type)
        else:
            sink_m = Float32(attn_sink[head_idx] * attention_utils.LOG2_E)
            new_m = attention_utils.fmax(merged_m, sink_m)
            prev_scale = _exp2_approx_ftz_f32(merged_m - new_m)
            sink_scale = _exp2_approx_ftz_f32(sink_m - new_m)
            merged_d = Float32(merged_d * prev_scale + sink_scale)
            inv_d = cute.arch.rcp_approx(merged_d)
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(acc[0] * prev_scale * inv_d).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(acc[1] * prev_scale * inv_d).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(acc[2] * prev_scale * inv_d).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(acc[3] * prev_scale * inv_d).to(output.element_type)


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


@lru_cache(maxsize=1)
def _build_sparse_mla_split_merge_kernel() -> SparseMLASplitDecodeMergeKernel:
    return SparseMLASplitDecodeMergeKernel()


@lru_cache(maxsize=1)
def _build_sparse_mla_split_sink_merge_kernel() -> SparseMLASplitDecodeSinkMergeKernel:
    return SparseMLASplitDecodeSinkMergeKernel()


def clear_sparse_mla_split_kernel_cache() -> None:
    _build_sparse_mla_split_forward_kernel.cache_clear()
    _build_sparse_mla_split_merge_kernel.cache_clear()
    _build_sparse_mla_split_sink_merge_kernel.cache_clear()


def run_sparse_mla_split_decode_forward(
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
    workspace: object | None = None,
    identity_page_table: bool = False,
) -> None:
    traits = select_sparse_mla_traits(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        output_dtype=tmp_output.dtype,
        v_head_dim=tmp_output.shape[-1],
    )
    if traits is None:
        raise ValueError("sparse MLA split decode only supports the exact CUDA GLM-5.1 contract")
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
    if launch_num_chunks <= 0 or launch_num_chunks > _SPLIT_MAX_CHUNKS:
        raise ValueError(
            f"launch_num_chunks must be in [1, {_SPLIT_MAX_CHUNKS}], got {launch_num_chunks}"
        )
    head_tiles = (int(tmp_output.shape[1]) + _MLA_HEADS_PER_TILE - 1) // _MLA_HEADS_PER_TILE

    kv_rows_u32, kv_scales = _extract_packed_kv_runtime_views(kv_cache)
    q_u32 = _view_last_dim_as_u32(q_all)
    if sm_scale.shape != (1,) or sm_scale.dtype != torch.float32:
        raise ValueError("sm_scale tensor must have shape (1,) and dtype float32")

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
    _cnt = getattr(workspace, "_contract_nsa_cache_seqlens", None)
    _cto = getattr(workspace, "_contract_tmp_output", None)
    _ctl = getattr(workspace, "_contract_tmp_lse", None)
    forward_cache_key = (
        _tensor_meta_key(_cq if _cq is not None else q_u32),
        _tensor_meta_key(_ckv if _ckv is not None else kv_rows_u32),
        _tensor_meta_key(_cks if _cks is not None else kv_scales),
        _tensor_meta_key(_cpt if _cpt is not None else page_table_1),
        _tensor_meta_key(_cnt if _cnt is not None else active_token_counts),
        _tensor_meta_key(kv_chunk_size_ptr),
        _tensor_meta_key(num_chunks_ptr),
        _tensor_meta_key(_cto if _cto is not None else tmp_output),
        _tensor_meta_key(_ctl if _ctl is not None else tmp_lse),
        traits,
        int(launch_num_chunks),
        head_tiles,
        str(tmp_output.dtype),
        bool(identity_page_table),
    )
    _run_cached_host_launcher(forward_kernel, forward_cache_key, forward_args)


def run_sparse_mla_split_decode_merge(
    *,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor | None = None,
    workspace: object | None = None,
) -> None:
    _cto = getattr(workspace, "_contract_tmp_output", None)
    _ctl = getattr(workspace, "_contract_tmp_lse", None)
    _co = getattr(workspace, "_contract_output", None)
    if attn_sink is None:
        merge_kernel = _build_sparse_mla_split_merge_kernel()
        merge_args = (
            _to_kernel_tensor(tmp_output, _torch_to_cutlass_dtype(tmp_output.dtype)),
            _to_kernel_tensor(tmp_lse, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(num_chunks_ptr, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
            current_cuda_stream(),
        )
        merge_cache_key = (
            _tensor_meta_key(_cto if _cto is not None else tmp_output),
            _tensor_meta_key(_ctl if _ctl is not None else tmp_lse),
            _tensor_meta_key(num_chunks_ptr),
            _tensor_meta_key(_co if _co is not None else output),
            str(tmp_output.dtype),
            str(output.dtype),
        )
        _run_cached_host_launcher(merge_kernel, merge_cache_key, merge_args)
        return

    attn_sink = attn_sink.detach()
    if attn_sink.dtype != torch.float32:
        raise ValueError(f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}")
    if attn_sink.device != output.device:
        raise ValueError("attn_sink must be on the same CUDA device as output")
    if attn_sink.ndim != 1 or int(attn_sink.shape[0]) != int(output.shape[1]):
        raise ValueError(
            f"attn_sink must have shape ({int(output.shape[1])},), got {tuple(attn_sink.shape)}"
        )
    if not attn_sink.is_contiguous():
        raise ValueError("attn_sink must be contiguous for the fused split-merge path")

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
        _tensor_meta_key(_cto if _cto is not None else tmp_output),
        _tensor_meta_key(_ctl if _ctl is not None else tmp_lse),
        _tensor_meta_key(num_chunks_ptr),
        _tensor_meta_key(attn_sink),
        _tensor_meta_key(_co if _co is not None else output),
        str(tmp_output.dtype),
        str(output.dtype),
        "attn_sink",
    )
    _run_cached_host_launcher(merge_kernel, merge_cache_key, merge_args)


def run_sparse_mla_split_decode(
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
    workspace: object | None = None,
    identity_page_table: bool = False,
) -> None:
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
