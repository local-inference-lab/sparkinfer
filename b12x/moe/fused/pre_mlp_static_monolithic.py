"""Monolithic static pre-MLP producer for the compact Qwen3.5 MoE path.

This kernel owns the full producer-side dependency chain in one launch:
  oneshot allreduce -> residual add -> Gemma RMSNorm -> sparse routing
  -> shared gate -> compact row reservation -> FC1 prequantized packing

V1 is correctness-first and intentionally serial after the norm stage:
lane 0 computes routing and packs each routed row into the existing compact
prequantized workspace contract consumed by the static/micro MoE kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32

from b12x.cute.fp4 import (
    atomic_add_global_i32,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    quantize_block_fp4,
    st_global_f32,
    st_global_i32,
    st_global_u8,
    st_global_u64,
    warp_reduce,
)
from b12x.cute.utils import current_cuda_stream, make_ptr
from b12x.distributed._oneshot_common import (
    MAX_BLOCKS,
    THREADS_PER_BLOCK,
    add_f32,
    align_bytes,
    cutlass_dtype,
    reduce_peer_row_sum,
    sqrt_f32,
    wait_for_peer_signals,
)
from b12x.moe.fused.pre_mlp_static import UnifiedPreMLPIPC, _exp2_approx_ftz_f32
from b12x.moe.fused.static import (
    _atomic_cas_global_i32,
    _ld_global_acquire_i32,
    _spin_wait_global_eq_i32,
    _st_global_release_i32,
    _threadfence,
)


LOG2_E = 1.4426950408889634
_FC1_TILE_ROWS = 128
_FC1_BLOCK_SIZE = 16


@dataclass(frozen=True, kw_only=True)
class MonolithicPreMLPStaticOutputs:
    normalized: torch.Tensor
    residual_out: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@cute.jit
def _resident_grid_barrier(
    barrier_count: cute.Tensor,
    barrier_epoch: cute.Tensor,
    grid_x: Int32,
    is_cta_leader: Int32,
):
    cute.arch.sync_threads()
    _threadfence()
    if is_cta_leader > Int32(0):
        barrier_count_addr = get_ptr_as_int64(barrier_count, Int32(0))
        barrier_epoch_addr = get_ptr_as_int64(barrier_epoch, Int32(0))
        old_epoch = _ld_global_acquire_i32(barrier_epoch_addr)
        arrived = atomic_add_global_i32(barrier_count_addr, Int32(1))
        if arrived == grid_x - Int32(1):
            st_global_i32(barrier_count_addr, Int32(0))
            _st_global_release_i32(barrier_epoch_addr, old_epoch + Int32(1))
        else:
            _spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)
    cute.arch.sync_threads()


@cute.jit
def _append_compact_prequantized_row(
    *,
    token_idx: Int32,
    expert_id: Int32,
    route_weight: cutlass.Float32,
    normalized_out: cute.Tensor,
    expert_input_scale: cute.Tensor,
    expert_alpha: cute.Tensor,
    active_expert_count: cute.Tensor,
    weight_expert_ids: cute.Tensor,
    global_to_local_expert: cute.Tensor,
    row_counts: cute.Tensor,
    token_map_flat: cute.Tensor,
    token_weights_flat: cute.Tensor,
    packed_input_flat: cute.Tensor,
    packed_input_scale_flat: cute.Tensor,
    fc1_tile_scale_flat: cute.Tensor,
    fc1_tile_alpha_flat: cute.Tensor,
    max_rows: Int32,
    output_bytes_per_row: Int32,
    scale_rows: Int32,
    scale_cols: Int32,
    tiles_per_expert: Int32,
    sf_blocks_per_row: Int32,
    num_k_tiles: Int32,
):
    local_expert_id = Int32(0)
    prior_local_expert_id = _atomic_cas_global_i32(
        get_ptr_as_int64(global_to_local_expert, expert_id),
        Int32(-1),
        Int32(-2),
    )
    if prior_local_expert_id == Int32(-1):
        local_expert_id = atomic_add_global_i32(
            get_ptr_as_int64(active_expert_count, Int32(0)),
            Int32(1),
        )
        st_global_i32(get_ptr_as_int64(weight_expert_ids, local_expert_id), expert_id)
        _st_global_release_i32(
            get_ptr_as_int64(global_to_local_expert, expert_id),
            local_expert_id,
        )
    else:
        if prior_local_expert_id == Int32(-2):
            _spin_wait_global_eq_i32(
                get_ptr_as_int64(global_to_local_expert, expert_id),
                Int32(-2),
            )
            prior_local_expert_id = _ld_global_acquire_i32(
                get_ptr_as_int64(global_to_local_expert, expert_id),
            )
        local_expert_id = prior_local_expert_id

    row = atomic_add_global_i32(
        get_ptr_as_int64(row_counts, local_expert_id),
        Int32(1),
    )
    map_idx = local_expert_id * max_rows + row
    st_global_i32(get_ptr_as_int64(token_map_flat, map_idx), token_idx)
    st_global_f32(get_ptr_as_int64(token_weights_flat, map_idx), route_weight)

    gs_value = expert_input_scale[expert_id].to(cutlass.Float32)
    alpha_value = expert_alpha[expert_id].to(cutlass.Float32)
    tile_idx = row // Int32(_FC1_TILE_ROWS)
    tile_offset = local_expert_id * tiles_per_expert + tile_idx
    st_global_f32(get_ptr_as_int64(fc1_tile_scale_flat, tile_offset), gs_value)
    st_global_f32(get_ptr_as_int64(fc1_tile_alpha_flat, tile_offset), alpha_value)

    expert_scale_stride = scale_rows * scale_cols
    sf_idx = Int32(0)
    while sf_idx < sf_blocks_per_row:
        block_start = sf_idx * Int32(_FC1_BLOCK_SIZE)
        values = cute.make_rmem_tensor((_FC1_BLOCK_SIZE,), cutlass.Float32)
        block_max = cutlass.Float32(0.0)
        for elem_idx in cutlass.range_constexpr(_FC1_BLOCK_SIZE):
            value = cutlass.Float32(
                normalized_out[token_idx, block_start + Int32(elem_idx)]
            )
            values[elem_idx] = value
            block_max = fmax_f32(block_max, fabs_f32(value))
        packed64, scale_byte = quantize_block_fp4(values, block_max, gs_value)

        output_offset = (
            local_expert_id * max_rows * output_bytes_per_row
            + row * output_bytes_per_row
            + sf_idx * Int32(8)
        )
        st_global_u64(get_ptr_as_int64(packed_input_flat, output_offset), packed64)

        m_tile_idx = row // Int32(_FC1_TILE_ROWS)
        k_tile_idx = sf_idx // Int32(4)
        outer_m_idx = row % Int32(32)
        inner_m_idx = (row % Int32(_FC1_TILE_ROWS)) // Int32(32)
        inner_k_idx = sf_idx % Int32(4)
        scale_offset = (
            local_expert_id * expert_scale_stride
            + m_tile_idx * num_k_tiles * Int32(32 * 4 * 4)
            + k_tile_idx * Int32(32 * 4 * 4)
            + outer_m_idx * Int32(4 * 4)
            + inner_m_idx * Int32(4)
            + inner_k_idx
        )
        st_global_u8(get_ptr_as_int64(packed_input_scale_flat, scale_offset), scale_byte)
        sf_idx += Int32(1)


class _MonolithicPreMLPKernel:
    def __init__(
        self,
        *,
        world_size: int,
        num_tokens: int,
        hidden_size: int,
        num_sparse_experts: int,
        top_k: int,
        state_E: int,
        weight_E: int,
        max_rows: int,
        scale_rows: int,
        scale_cols: int,
        tiles_per_expert: int,
        element_dtype,
        renormalize_topk: bool,
    ):
        self.world_size = world_size
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_sparse_experts = num_sparse_experts
        self.top_k = top_k
        self.combined_top_k = top_k + 1
        self.state_E = state_E
        self.weight_E = weight_E
        self.max_rows = max_rows
        self.scale_rows = scale_rows
        self.scale_cols = scale_cols
        self.tiles_per_expert = tiles_per_expert
        self.element_dtype = element_dtype
        self.renormalize_topk = renormalize_topk

    def __call__(self, *args):
        stream = args[-1]
        self.kernel(*args[:-1]).launch(
            grid=[self.num_tokens, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        inp0: cute.Tensor,
        inp1: cute.Tensor,
        inp2: cute.Tensor,
        inp3: cute.Tensor,
        inp4: cute.Tensor,
        inp5: cute.Tensor,
        inp6: cute.Tensor,
        inp7: cute.Tensor,
        residual_in: cute.Tensor,
        normalized_out: cute.Tensor,
        residual_out: cute.Tensor,
        norm_weight: cute.Tensor,
        sparse_gate_weight: cute.Tensor,
        shared_gate_weight: cute.Tensor,
        expert_input_scale: cute.Tensor,
        expert_alpha: cute.Tensor,
        active_expert_count: cute.Tensor,
        weight_expert_ids: cute.Tensor,
        global_to_local_expert: cute.Tensor,
        row_counts: cute.Tensor,
        token_map_flat: cute.Tensor,
        token_weights_flat: cute.Tensor,
        packed_input_flat: cute.Tensor,
        packed_input_scale_flat: cute.Tensor,
        fc1_tile_scale_flat: cute.Tensor,
        fc1_tile_alpha_flat: cute.Tensor,
        topk_ids_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        self_signal: cute.Pointer,
        rank: Int32,
        eps: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tidx = Int32(tidx)
        bidx = Int32(bidx)
        grid_x = Int32(self.num_tokens)
        flat_tid = bidx * Int32(THREADS_PER_BLOCK) + tidx
        flat_stride = grid_x * Int32(THREADS_PER_BLOCK)

        i = flat_tid
        while i < Int32(self.state_E):
            row_counts[i] = Int32(0)
            weight_expert_ids[i] = Int32(0)
            i += flat_stride
        i = flat_tid
        while i < Int32(self.weight_E):
            global_to_local_expert[i] = Int32(-1)
            i += flat_stride
        i = flat_tid
        while i < Int32(self.state_E * self.max_rows):
            token_map_flat[i] = Int32(0)
            token_weights_flat[i] = cutlass.Float32(0.0)
            i += flat_stride
        i = flat_tid
        while i < Int32(self.state_E * self.max_rows * (self.hidden_size // 2)):
            packed_input_flat[i] = cutlass.Uint8(0)
            i += flat_stride
        i = flat_tid
        while i < Int32(self.state_E * self.scale_rows * self.scale_cols):
            packed_input_scale_flat[i] = cutlass.Uint8(0)
            i += flat_stride
        i = flat_tid
        while i < Int32(self.state_E * self.tiles_per_expert):
            fc1_tile_scale_flat[i] = cutlass.Float32(0.0)
            fc1_tile_alpha_flat[i] = cutlass.Float32(0.0)
            i += flat_stride
        if flat_tid == Int32(0):
            active_expert_count[Int32(0)] = Int32(0)
        _resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            grid_x,
            Int32(1) if tidx == Int32(0) else Int32(0),
        )

        signal_ptrs = [signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7]
        inputs = [inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7]
        wait_for_peer_signals(
            signal_ptrs=signal_ptrs,
            self_signal=self_signal,
            rank=rank,
            world_size=self.world_size,
            bidx=bidx,
            tidx=tidx,
        )

        local_sum_sq = cutlass.Float32(0.0)
        col = tidx
        while col < Int32(self.hidden_size):
            acc = reduce_peer_row_sum(
                inputs=inputs,
                world_size=self.world_size,
                bidx=bidx,
                col=col,
                element_dtype=self.element_dtype,
            )
            residual_val = self.element_dtype(
                acc + self.element_dtype(residual_in[bidx, col])
            )
            residual_out[bidx, col] = residual_val
            residual_f32 = cutlass.Float32(residual_val)
            local_sum_sq += residual_f32 * residual_f32
            col += Int32(THREADS_PER_BLOCK)

        sum_sq = warp_reduce(local_sum_sq, add_f32)
        inv_scale = cutlass.Float32(1.0) / sqrt_f32(
            sum_sq / cutlass.Float32(self.hidden_size) + eps
        )

        col = tidx
        while col < Int32(self.hidden_size):
            gamma = cutlass.Float32(1.0) + cutlass.Float32(norm_weight[col])
            out = cutlass.Float32(residual_out[bidx, col]) * inv_scale * gamma
            normalized_out[bidx, col] = self.element_dtype(out)
            col += Int32(THREADS_PER_BLOCK)
        cute.arch.sync_threads()

        if tidx == Int32(0):
            neg_inf = cutlass.Float32(-3.4028235e38)
            top_vals = [neg_inf for _ in range(self.top_k)]
            top_ids = [Int32(-1) for _ in range(self.top_k)]

            shared_acc = cutlass.Float32(0.0)
            col = Int32(0)
            while col < Int32(self.hidden_size):
                x = cutlass.Float32(normalized_out[bidx, col])
                w = cutlass.Float32(shared_gate_weight[col])
                shared_acc += x * w
                col += Int32(1)
            shared_logit = self.element_dtype(shared_acc)
            neg_exp = _exp2_approx_ftz_f32(
                cutlass.Float32(-LOG2_E) * cutlass.Float32(shared_logit)
            )
            shared_gate = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + neg_exp)

            for slot in cutlass.range_constexpr(self.top_k):
                best_val = neg_inf
                best_id = Int32(-1)
                expert = Int32(0)
                while expert < Int32(self.num_sparse_experts):
                    skip = Int32(0)
                    for prev in cutlass.range_constexpr(self.top_k):
                        if prev < slot:
                            if expert == top_ids[prev]:
                                skip = Int32(1)
                    if skip == Int32(0):
                        acc = cutlass.Float32(0.0)
                        col = Int32(0)
                        while col < Int32(self.hidden_size):
                            x = cutlass.Float32(normalized_out[bidx, col])
                            w = cutlass.Float32(sparse_gate_weight[expert, col])
                            acc += x * w
                            col += Int32(1)
                        rounded = self.element_dtype(acc)
                        candidate_val = cutlass.Float32(rounded)
                        if candidate_val > best_val or (
                            candidate_val == best_val and expert > best_id
                        ):
                            best_val = candidate_val
                            best_id = expert
                    expert += Int32(1)
                top_vals[slot] = best_val
                top_ids[slot] = best_id

            if self.renormalize_topk:
                max_logit = top_vals[0]
                exp_vals = [cutlass.Float32(0.0) for _ in range(self.top_k)]
                denom = cutlass.Float32(0.0)
                for slot in cutlass.range_constexpr(self.top_k):
                    exp_val = _exp2_approx_ftz_f32(
                        (top_vals[slot] - max_logit) * cutlass.Float32(LOG2_E)
                    )
                    exp_vals[slot] = exp_val
                    denom += exp_val
                inv_denom = cutlass.Float32(1.0) / denom
                for slot in cutlass.range_constexpr(self.top_k):
                    topk_offset = bidx * Int32(self.combined_top_k) + Int32(slot)
                    weight = exp_vals[slot] * inv_denom
                    st_global_i32(get_ptr_as_int64(topk_ids_flat, topk_offset), top_ids[slot])
                    st_global_f32(get_ptr_as_int64(topk_weights_flat, topk_offset), weight)
                    _append_compact_prequantized_row(
                        token_idx=bidx,
                        expert_id=top_ids[slot],
                        route_weight=weight,
                        normalized_out=normalized_out,
                        expert_input_scale=expert_input_scale,
                        expert_alpha=expert_alpha,
                        active_expert_count=active_expert_count,
                        weight_expert_ids=weight_expert_ids,
                        global_to_local_expert=global_to_local_expert,
                        row_counts=row_counts,
                        token_map_flat=token_map_flat,
                        token_weights_flat=token_weights_flat,
                        packed_input_flat=packed_input_flat,
                        packed_input_scale_flat=packed_input_scale_flat,
                        fc1_tile_scale_flat=fc1_tile_scale_flat,
                        fc1_tile_alpha_flat=fc1_tile_alpha_flat,
                        max_rows=Int32(self.max_rows),
                        output_bytes_per_row=Int32(self.hidden_size // 2),
                        scale_rows=Int32(self.scale_rows),
                        scale_cols=Int32(self.scale_cols),
                        tiles_per_expert=Int32(self.tiles_per_expert),
                        sf_blocks_per_row=Int32(self.hidden_size // _FC1_BLOCK_SIZE),
                        num_k_tiles=Int32((self.hidden_size // _FC1_BLOCK_SIZE) // 4),
                    )
            else:
                for slot in cutlass.range_constexpr(self.top_k):
                    topk_offset = bidx * Int32(self.combined_top_k) + Int32(slot)
                    weight = top_vals[slot]
                    st_global_i32(get_ptr_as_int64(topk_ids_flat, topk_offset), top_ids[slot])
                    st_global_f32(get_ptr_as_int64(topk_weights_flat, topk_offset), weight)
                    _append_compact_prequantized_row(
                        token_idx=bidx,
                        expert_id=top_ids[slot],
                        route_weight=weight,
                        normalized_out=normalized_out,
                        expert_input_scale=expert_input_scale,
                        expert_alpha=expert_alpha,
                        active_expert_count=active_expert_count,
                        weight_expert_ids=weight_expert_ids,
                        global_to_local_expert=global_to_local_expert,
                        row_counts=row_counts,
                        token_map_flat=token_map_flat,
                        token_weights_flat=token_weights_flat,
                        packed_input_flat=packed_input_flat,
                        packed_input_scale_flat=packed_input_scale_flat,
                        fc1_tile_scale_flat=fc1_tile_scale_flat,
                        fc1_tile_alpha_flat=fc1_tile_alpha_flat,
                        max_rows=Int32(self.max_rows),
                        output_bytes_per_row=Int32(self.hidden_size // 2),
                        scale_rows=Int32(self.scale_rows),
                        scale_cols=Int32(self.scale_cols),
                        tiles_per_expert=Int32(self.tiles_per_expert),
                        sf_blocks_per_row=Int32(self.hidden_size // _FC1_BLOCK_SIZE),
                        num_k_tiles=Int32((self.hidden_size // _FC1_BLOCK_SIZE) // 4),
                    )

            shared_offset = bidx * Int32(self.combined_top_k) + Int32(self.top_k)
            st_global_i32(
                get_ptr_as_int64(topk_ids_flat, shared_offset),
                Int32(self.num_sparse_experts),
            )
            st_global_f32(get_ptr_as_int64(topk_weights_flat, shared_offset), shared_gate)
            _append_compact_prequantized_row(
                token_idx=bidx,
                expert_id=Int32(self.num_sparse_experts),
                route_weight=shared_gate,
                normalized_out=normalized_out,
                expert_input_scale=expert_input_scale,
                expert_alpha=expert_alpha,
                active_expert_count=active_expert_count,
                weight_expert_ids=weight_expert_ids,
                global_to_local_expert=global_to_local_expert,
                row_counts=row_counts,
                token_map_flat=token_map_flat,
                token_weights_flat=token_weights_flat,
                packed_input_flat=packed_input_flat,
                packed_input_scale_flat=packed_input_scale_flat,
                fc1_tile_scale_flat=fc1_tile_scale_flat,
                fc1_tile_alpha_flat=fc1_tile_alpha_flat,
                max_rows=Int32(self.max_rows),
                output_bytes_per_row=Int32(self.hidden_size // 2),
                scale_rows=Int32(self.scale_rows),
                scale_cols=Int32(self.scale_cols),
                tiles_per_expert=Int32(self.tiles_per_expert),
                sf_blocks_per_row=Int32(self.hidden_size // _FC1_BLOCK_SIZE),
                num_k_tiles=Int32((self.hidden_size // _FC1_BLOCK_SIZE) // 4),
            )


class _MonolithicPreMLPLaunch:
    def __init__(
        self,
        *,
        world_size: int,
        num_tokens: int,
        hidden_size: int,
        num_sparse_experts: int,
        top_k: int,
        state_E: int,
        weight_E: int,
        max_rows: int,
        scale_rows: int,
        scale_cols: int,
        tiles_per_expert: int,
        input_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        renormalize_topk: bool,
    ):
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_sparse_experts = num_sparse_experts
        self.top_k = top_k
        self.combined_top_k = top_k + 1
        self.state_E = state_E
        self.weight_E = weight_E
        self.max_rows = max_rows
        self.scale_rows = scale_rows
        self.scale_cols = scale_cols
        self.tiles_per_expert = tiles_per_expert
        self.input_cutlass_dtype = cutlass_dtype(input_dtype)
        self.weight_cutlass_dtype = cutlass_dtype(weight_dtype)
        self.kernel = _MonolithicPreMLPKernel(
            world_size=world_size,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_sparse_experts=num_sparse_experts,
            top_k=top_k,
            state_E=state_E,
            weight_E=weight_E,
            max_rows=max_rows,
            scale_rows=scale_rows,
            scale_cols=scale_cols,
            tiles_per_expert=tiles_per_expert,
            element_dtype=self.input_cutlass_dtype,
            renormalize_topk=renormalize_topk,
        )

    @cute.jit
    def __call__(
        self,
        inp0_ptr: cute.Pointer,
        inp1_ptr: cute.Pointer,
        inp2_ptr: cute.Pointer,
        inp3_ptr: cute.Pointer,
        inp4_ptr: cute.Pointer,
        inp5_ptr: cute.Pointer,
        inp6_ptr: cute.Pointer,
        inp7_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        normalized_ptr: cute.Pointer,
        residual_out_ptr: cute.Pointer,
        norm_weight_ptr: cute.Pointer,
        sparse_gate_weight_ptr: cute.Pointer,
        shared_gate_weight_ptr: cute.Pointer,
        expert_input_scale_ptr: cute.Pointer,
        expert_alpha_ptr: cute.Pointer,
        active_expert_count_ptr: cute.Pointer,
        weight_expert_ids_ptr: cute.Pointer,
        global_to_local_expert_ptr: cute.Pointer,
        row_counts_ptr: cute.Pointer,
        token_map_flat_ptr: cute.Pointer,
        token_weights_flat_ptr: cute.Pointer,
        packed_input_flat_ptr: cute.Pointer,
        packed_input_scale_flat_ptr: cute.Pointer,
        fc1_tile_scale_flat_ptr: cute.Pointer,
        fc1_tile_alpha_flat_ptr: cute.Pointer,
        topk_ids_flat_ptr: cute.Pointer,
        topk_weights_flat_ptr: cute.Pointer,
        barrier_count_ptr: cute.Pointer,
        barrier_epoch_ptr: cute.Pointer,
        signal0_ptr: cute.Pointer,
        signal1_ptr: cute.Pointer,
        signal2_ptr: cute.Pointer,
        signal3_ptr: cute.Pointer,
        signal4_ptr: cute.Pointer,
        signal5_ptr: cute.Pointer,
        signal6_ptr: cute.Pointer,
        signal7_ptr: cute.Pointer,
        self_signal_ptr: cute.Pointer,
        rank: Int32,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        row_layout = cute.make_layout(
            (self.num_tokens, self.hidden_size),
            stride=(self.hidden_size, 1),
        )
        sparse_gate_layout = cute.make_layout(
            (self.num_sparse_experts, self.hidden_size),
            stride=(self.hidden_size, 1),
        )
        vector_weight_E = cute.make_layout((self.weight_E,), stride=(1,))
        vector_state_E = cute.make_layout((self.state_E,), stride=(1,))
        flat_token_rows = cute.make_layout((self.state_E * self.max_rows,), stride=(1,))
        flat_packed = cute.make_layout(
            (self.state_E * self.max_rows * (self.hidden_size // 2),),
            stride=(1,),
        )
        flat_scales = cute.make_layout(
            (self.state_E * self.scale_rows * self.scale_cols,),
            stride=(1,),
        )
        flat_tiles = cute.make_layout((self.state_E * self.tiles_per_expert,), stride=(1,))
        flat_topk = cute.make_layout((self.num_tokens * self.combined_top_k,), stride=(1,))
        scalar_layout = cute.make_layout((1,), stride=(1,))

        inputs = [
            cute.make_tensor(inp0_ptr, layout=row_layout),
            cute.make_tensor(inp1_ptr, layout=row_layout),
            cute.make_tensor(inp2_ptr, layout=row_layout),
            cute.make_tensor(inp3_ptr, layout=row_layout),
            cute.make_tensor(inp4_ptr, layout=row_layout),
            cute.make_tensor(inp5_ptr, layout=row_layout),
            cute.make_tensor(inp6_ptr, layout=row_layout),
            cute.make_tensor(inp7_ptr, layout=row_layout),
        ]
        self.kernel(
            *inputs,
            cute.make_tensor(residual_ptr, layout=row_layout),
            cute.make_tensor(normalized_ptr, layout=row_layout),
            cute.make_tensor(residual_out_ptr, layout=row_layout),
            cute.make_tensor(norm_weight_ptr, layout=cute.make_layout((self.hidden_size,), stride=(1,))),
            cute.make_tensor(sparse_gate_weight_ptr, layout=sparse_gate_layout),
            cute.make_tensor(shared_gate_weight_ptr, layout=cute.make_layout((self.hidden_size,), stride=(1,))),
            cute.make_tensor(expert_input_scale_ptr, layout=vector_weight_E),
            cute.make_tensor(expert_alpha_ptr, layout=vector_weight_E),
            cute.make_tensor(active_expert_count_ptr, layout=scalar_layout),
            cute.make_tensor(weight_expert_ids_ptr, layout=vector_state_E),
            cute.make_tensor(global_to_local_expert_ptr, layout=vector_weight_E),
            cute.make_tensor(row_counts_ptr, layout=vector_state_E),
            cute.make_tensor(token_map_flat_ptr, layout=flat_token_rows),
            cute.make_tensor(token_weights_flat_ptr, layout=flat_token_rows),
            cute.make_tensor(packed_input_flat_ptr, layout=flat_packed),
            cute.make_tensor(packed_input_scale_flat_ptr, layout=flat_scales),
            cute.make_tensor(fc1_tile_scale_flat_ptr, layout=flat_tiles),
            cute.make_tensor(fc1_tile_alpha_flat_ptr, layout=flat_tiles),
            cute.make_tensor(topk_ids_flat_ptr, layout=flat_topk),
            cute.make_tensor(topk_weights_flat_ptr, layout=flat_topk),
            cute.make_tensor(barrier_count_ptr, layout=scalar_layout),
            cute.make_tensor(barrier_epoch_ptr, layout=scalar_layout),
            signal0_ptr,
            signal1_ptr,
            signal2_ptr,
            signal3_ptr,
            signal4_ptr,
            signal5_ptr,
            signal6_ptr,
            signal7_ptr,
            self_signal_ptr,
            rank,
            eps,
            stream,
        )


@lru_cache(maxsize=128)
def _get_monolithic_kernel(
    world_size: int,
    num_tokens: int,
    hidden_size: int,
    num_sparse_experts: int,
    top_k: int,
    state_E: int,
    weight_E: int,
    max_rows: int,
    scale_rows: int,
    scale_cols: int,
    tiles_per_expert: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    renormalize_topk: bool,
):
    launch = _MonolithicPreMLPLaunch(
        world_size=world_size,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_sparse_experts=num_sparse_experts,
        top_k=top_k,
        state_E=state_E,
        weight_E=weight_E,
        max_rows=max_rows,
        scale_rows=scale_rows,
        scale_cols=scale_cols,
        tiles_per_expert=tiles_per_expert,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        renormalize_topk=renormalize_topk,
    )
    input_align = align_bytes(input_dtype)
    weight_align = align_bytes(weight_dtype)
    row_fake = make_ptr(
        cutlass_dtype(input_dtype),
        max(16, input_align),
        cute.AddressSpace.gmem,
        assumed_align=input_align,
    )
    weight_fake = make_ptr(
        cutlass_dtype(weight_dtype),
        max(16, weight_align),
        cute.AddressSpace.gmem,
        assumed_align=weight_align,
    )
    f32_fake = make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4)
    i32_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    u8_fake = make_ptr(cutlass.Uint8, 1, cute.AddressSpace.gmem, assumed_align=1)
    signal_fake = make_ptr(cutlass.Int32, 128, cute.AddressSpace.gmem, assumed_align=128)
    return cute.compile(
        launch,
        row_fake,
        row_fake,
        row_fake,
        row_fake,
        row_fake,
        row_fake,
        row_fake,
        row_fake,
        row_fake,
        row_fake,
        row_fake,
        weight_fake,
        weight_fake,
        weight_fake,
        f32_fake,
        f32_fake,
        i32_fake,
        i32_fake,
        i32_fake,
        i32_fake,
        i32_fake,
        f32_fake,
        u8_fake,
        u8_fake,
        f32_fake,
        f32_fake,
        i32_fake,
        f32_fake,
        i32_fake,
        i32_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        signal_fake,
        0,
        1.0,
        current_cuda_stream(),
    )


def monolithic_allreduce_route_prepack_static(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    sparse_gate_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    *,
    expert_input_scale: torch.Tensor,
    expert_alpha: torch.Tensor,
    workspace,
    ipc: UnifiedPreMLPIPC,
    top_k: int,
    renormalize_topk: bool = True,
    eps: float,
    normalized_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    topk_ids_out: torch.Tensor | None = None,
    topk_weights_out: torch.Tensor | None = None,
) -> MonolithicPreMLPStaticOutputs:
    ipc.validate()
    if hidden_states.ndim != 2 or residual.ndim != 2:
        raise ValueError("hidden_states and residual must both be [num_tokens, hidden_size]")
    if hidden_states.shape != residual.shape:
        raise ValueError("hidden_states and residual must have the same shape")
    if hidden_states.shape[0] > MAX_BLOCKS:
        raise ValueError(f"monolithic producer supports at most {MAX_BLOCKS} tokens")
    if not hidden_states.is_contiguous() or not residual.is_contiguous():
        raise ValueError("hidden_states and residual must be contiguous")
    if not norm_weight.is_contiguous() or not sparse_gate_weight.is_contiguous():
        raise ValueError("norm_weight and sparse_gate_weight must be contiguous")
    if expert_input_scale.ndim != 1 or expert_alpha.ndim != 1:
        raise ValueError("expert_input_scale and expert_alpha must be rank-1")

    num_tokens, hidden_size = hidden_states.shape
    num_sparse_experts = sparse_gate_weight.shape[0]
    combined_top_k = top_k + 1
    if normalized_out is None:
        normalized_out = torch.empty_like(hidden_states)
    if residual_out is None:
        residual_out = torch.empty_like(hidden_states)
    if topk_ids_out is None:
        topk_ids_out = torch.empty(
            (num_tokens, combined_top_k),
            dtype=torch.int32,
            device=hidden_states.device,
        )
    if topk_weights_out is None:
        topk_weights_out = torch.empty(
            (num_tokens, combined_top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
    if shared_gate_weight.ndim == 2:
        if shared_gate_weight.shape[0] != 1:
            raise ValueError("shared_gate_weight must have leading dimension 1 when rank-2")
        shared_gate_weight = shared_gate_weight.view(-1)
    if shared_gate_weight.ndim != 1 or shared_gate_weight.numel() != hidden_size:
        raise ValueError("shared_gate_weight must be [hidden_size]")

    compiled = _get_monolithic_kernel(
        ipc.world_size,
        num_tokens,
        hidden_size,
        num_sparse_experts,
        top_k,
        int(workspace.state_E),
        int(workspace.weight_E),
        int(workspace.max_rows),
        int(workspace.packed_input_scale.shape[1]),
        int(workspace.packed_input_scale.shape[2]),
        int(workspace.fc1_tile_scale.shape[1]),
        hidden_states.dtype,
        sparse_gate_weight.dtype,
        renormalize_topk,
    )
    input_ptr_args = [
        make_ptr(
            cutlass_dtype(hidden_states.dtype),
            ptr,
            cute.AddressSpace.gmem,
            assumed_align=align_bytes(hidden_states.dtype),
        )
        for ptr in ipc.peer_input_ptrs
    ]
    while len(input_ptr_args) < 8:
        input_ptr_args.append(input_ptr_args[0])
    signal_ptr_args = [
        make_ptr(cutlass.Int32, ptr, cute.AddressSpace.gmem, assumed_align=128)
        for ptr in ipc.signal_ptrs
    ]
    while len(signal_ptr_args) < 8:
        signal_ptr_args.append(signal_ptr_args[0])

    compiled(
        *input_ptr_args[:8],
        make_ptr(
            cutlass_dtype(hidden_states.dtype),
            int(residual.data_ptr()),
            cute.AddressSpace.gmem,
            assumed_align=align_bytes(hidden_states.dtype),
        ),
        make_ptr(
            cutlass_dtype(hidden_states.dtype),
            int(normalized_out.data_ptr()),
            cute.AddressSpace.gmem,
            assumed_align=align_bytes(hidden_states.dtype),
        ),
        make_ptr(
            cutlass_dtype(hidden_states.dtype),
            int(residual_out.data_ptr()),
            cute.AddressSpace.gmem,
            assumed_align=align_bytes(hidden_states.dtype),
        ),
        make_ptr(
            cutlass_dtype(norm_weight.dtype),
            int(norm_weight.data_ptr()),
            cute.AddressSpace.gmem,
            assumed_align=align_bytes(norm_weight.dtype),
        ),
        make_ptr(
            cutlass_dtype(sparse_gate_weight.dtype),
            int(sparse_gate_weight.data_ptr()),
            cute.AddressSpace.gmem,
            assumed_align=align_bytes(sparse_gate_weight.dtype),
        ),
        make_ptr(
            cutlass_dtype(shared_gate_weight.dtype),
            int(shared_gate_weight.data_ptr()),
            cute.AddressSpace.gmem,
            assumed_align=align_bytes(shared_gate_weight.dtype),
        ),
        make_ptr(cutlass.Float32, int(expert_input_scale.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, int(expert_alpha.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, int(workspace.active_expert_count.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, int(workspace.weight_expert_ids.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, int(workspace.global_to_local_expert.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, int(workspace.row_counts.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, int(workspace.token_map.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, int(workspace.token_weights.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, int(workspace.packed_input.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=1),
        make_ptr(cutlass.Uint8, int(workspace.packed_input_scale.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=1),
        make_ptr(cutlass.Float32, int(workspace.fc1_tile_scale.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, int(workspace.fc1_tile_alpha.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, int(topk_ids_out.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, int(topk_weights_out.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, int(workspace.barrier_count.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, int(workspace.barrier_epoch.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        *signal_ptr_args[:8],
        make_ptr(cutlass.Int32, ipc.signal_ptrs[ipc.rank], cute.AddressSpace.gmem, assumed_align=128),
        ipc.rank,
        float(eps),
        current_cuda_stream(),
    )
    return MonolithicPreMLPStaticOutputs(
        normalized=normalized_out,
        residual_out=residual_out,
        topk_ids=topk_ids_out,
        topk_weights=topk_weights_out,
    )


__all__ = [
    "MonolithicPreMLPStaticOutputs",
    "monolithic_allreduce_route_prepack_static",
]
