"""Correctness-first one-kernel micro megafused MoE path.

This is an intentionally dumb prototype for the m=1 decode regime. One CTA
owns one token and executes:

  oneshot allreduce -> residual add -> Gemma RMSNorm -> sparse routing
  -> shared gate -> naive NVFP4 FC1/FC2 compute -> output write

It does not roundtrip routed activations through the compact prequantized
workspace and does not attempt to reuse the existing micro compute half.
The goal is simply to prove the one-kernel shape.
"""

from __future__ import annotations

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32

from b12x.cute.fp4 import (
    fabs_f32,
    fmax_f32,
    fp8_e4m3_to_f32,
    get_ptr_as_int64,
    quantize_block_fp4,
    st_global_f32,
    st_global_i32,
    warp_reduce,
)
from b12x.cute.utils import current_cuda_stream, make_ptr
from b12x.distributed._oneshot_common import (
    THREADS_PER_BLOCK,
    add_f32,
    align_bytes,
    cutlass_dtype,
    reduce_peer_row_sum,
    sqrt_f32,
    wait_for_peer_signals,
)
from b12x.moe.fused.pre_mlp_static import UnifiedPreMLPIPC, _exp2_approx_ftz_f32


LOG2_E = 1.4426950408889634
_FP4_BLOCK = 16


def _unswizzle_scale_u8_batch(scale: torch.Tensor, rows: int, cols_blocks: int) -> torch.Tensor:
    rows_padded = ((rows + 127) // 128) * 128
    cols_padded = ((cols_blocks + 3) // 4) * 4
    batch = scale.shape[0]
    sf_u8 = scale.view(torch.uint8)
    unswizzled = sf_u8.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    unswizzled = unswizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    unswizzled = unswizzled.view(batch, rows_padded, cols_padded)
    return unswizzled[:, :rows, :cols_blocks].contiguous()


@cute.jit
def _fp4_nibble_to_f32(nibble: Uint8) -> cutlass.Float32:
    mag = nibble & Uint8(0x7)
    val = cutlass.Float32(0.0)
    if mag == Uint8(1):
        val = cutlass.Float32(0.5)
    elif mag == Uint8(2):
        val = cutlass.Float32(1.0)
    elif mag == Uint8(3):
        val = cutlass.Float32(1.5)
    elif mag == Uint8(4):
        val = cutlass.Float32(2.0)
    elif mag == Uint8(5):
        val = cutlass.Float32(3.0)
    elif mag == Uint8(6):
        val = cutlass.Float32(4.0)
    elif mag == Uint8(7):
        val = cutlass.Float32(6.0)
    if (nibble & Uint8(0x8)) != Uint8(0) and val != cutlass.Float32(0.0):
        val = cutlass.Float32(0.0) - val
    return val


@cute.jit
def _sigmoid_f32(x: cutlass.Float32) -> cutlass.Float32:
    neg_exp = _exp2_approx_ftz_f32(cutlass.Float32(-LOG2_E) * x)
    return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + neg_exp)


@cute.jit
def _fp4_quantize_dequant_value(value: cutlass.Float32, value_scale: cutlass.Float32) -> cutlass.Float32:
    t025 = value_scale * cutlass.Float32(0.25)
    t075 = value_scale * cutlass.Float32(0.75)
    t125 = value_scale * cutlass.Float32(1.25)
    t175 = value_scale * cutlass.Float32(1.75)
    t250 = value_scale * cutlass.Float32(2.5)
    t350 = value_scale * cutlass.Float32(3.5)
    t500 = value_scale * cutlass.Float32(5.0)
    mag = fabs_f32(value)
    q = cutlass.Float32(0.0)
    if mag > t025 and mag < t075:
        q = cutlass.Float32(0.5)
    elif mag >= t075 and mag <= t125:
        q = cutlass.Float32(1.0)
    elif mag > t125 and mag < t175:
        q = cutlass.Float32(1.5)
    elif mag >= t175 and mag <= t250:
        q = cutlass.Float32(2.0)
    elif mag > t250 and mag < t350:
        q = cutlass.Float32(3.0)
    elif mag >= t350 and mag <= t500:
        q = cutlass.Float32(4.0)
    elif mag > t500:
        q = cutlass.Float32(6.0)
    if q != cutlass.Float32(0.0) and value < cutlass.Float32(0.0):
        q = cutlass.Float32(0.0) - q
    return q


class _MicroMegaFusedKernel:
    def __init__(
        self,
        *,
        world_size: int,
        num_tokens: int,
        hidden_size: int,
        intermediate_size_local: int,
        num_sparse_experts: int,
        top_k: int,
        input_dtype,
        renormalize_topk: bool,
    ):
        self.world_size = world_size
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.intermediate_size_local = intermediate_size_local
        self.num_sparse_experts = num_sparse_experts
        self.top_k = top_k
        self.combined_top_k = top_k + 1
        self.input_dtype = input_dtype
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
        residual_out: cute.Tensor,
        normalized_out: cute.Tensor,
        norm_weight: cute.Tensor,
        sparse_gate_weight: cute.Tensor,
        shared_gate_weight: cute.Tensor,
        w1_fp4_u8: cute.Tensor,          # [E, 2*I_tp, K/2]
        w1_scale_u8: cute.Tensor,        # [E, 2*I_tp, K/16]
        w1_alpha: cute.Tensor,           # [E]
        input_global_scale: cute.Tensor, # [E] effective FC1 scale
        w2_fp4_u8: cute.Tensor,          # [E, K, I_tp/2]
        w2_scale_u8: cute.Tensor,        # [E, K, I_tp/16]
        w2_alpha: cute.Tensor,           # [E]
        fc2_global_scale: cute.Tensor,   # [E] effective FC2 scale
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        output: cute.Tensor,
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

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            norm_row: cute.struct.MemRange[cutlass.BFloat16, self.hidden_size]
            x_qdq: cute.struct.MemRange[cutlass.Float32, self.hidden_size]
            gate_vec: cute.struct.MemRange[cutlass.Float32, self.intermediate_size_local]
            up_vec: cute.struct.MemRange[cutlass.Float32, self.intermediate_size_local]
            int_qdq: cute.struct.MemRange[cutlass.Float32, self.intermediate_size_local]
            out_acc: cute.struct.MemRange[cutlass.Float32, self.hidden_size]
            selected_ids: cute.struct.MemRange[cutlass.Int32, self.combined_top_k]
            selected_weights: cute.struct.MemRange[cutlass.Float32, self.combined_top_k]

        storage = smem.allocate(Storage)
        sNorm = storage.norm_row.get_tensor(cute.make_layout((self.hidden_size,), stride=(1,)))
        sX = storage.x_qdq.get_tensor(cute.make_layout((self.hidden_size,), stride=(1,)))
        sGate = storage.gate_vec.get_tensor(cute.make_layout((self.intermediate_size_local,), stride=(1,)))
        sUp = storage.up_vec.get_tensor(cute.make_layout((self.intermediate_size_local,), stride=(1,)))
        sInt = storage.int_qdq.get_tensor(cute.make_layout((self.intermediate_size_local,), stride=(1,)))
        sOut = storage.out_acc.get_tensor(cute.make_layout((self.hidden_size,), stride=(1,)))
        sSelectedIds = storage.selected_ids.get_tensor(cute.make_layout((self.combined_top_k,), stride=(1,)))
        sSelectedWeights = storage.selected_weights.get_tensor(cute.make_layout((self.combined_top_k,), stride=(1,)))

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
                element_dtype=self.input_dtype,
            )
            residual_val = self.input_dtype(acc + self.input_dtype(residual_in[bidx, col]))
            residual_out[bidx, col] = residual_val
            sNorm[col] = residual_val
            residual_f32 = cutlass.Float32(residual_val)
            local_sum_sq += residual_f32 * residual_f32
            col += Int32(THREADS_PER_BLOCK)
        sum_sq = warp_reduce(local_sum_sq, add_f32)
        inv_scale = cutlass.Float32(1.0) / sqrt_f32(
            sum_sq / cutlass.Float32(self.hidden_size) + eps
        )
        cute.arch.sync_threads()

        col = tidx
        while col < Int32(self.hidden_size):
            gamma = cutlass.Float32(1.0) + cutlass.Float32(norm_weight[col])
            normed = self.input_dtype(cutlass.Float32(sNorm[col]) * inv_scale * gamma)
            sNorm[col] = normed
            normalized_out[bidx, col] = normed
            sOut[col] = cutlass.Float32(0.0)
            col += Int32(THREADS_PER_BLOCK)
        cute.arch.sync_threads()

        if tidx == Int32(0):
            neg_inf = cutlass.Float32(-3.4028235e38)
            top_vals = [neg_inf for _ in range(self.top_k)]
            top_ids_local = [Int32(-1) for _ in range(self.top_k)]

            shared_acc = cutlass.Float32(0.0)
            col0 = Int32(0)
            while col0 < Int32(self.hidden_size):
                shared_acc += cutlass.Float32(sNorm[col0]) * cutlass.Float32(shared_gate_weight[col0])
                col0 += Int32(1)
            shared_gate = _sigmoid_f32(cutlass.Float32(self.input_dtype(shared_acc)))

            for slot in cutlass.range_constexpr(self.top_k):
                best_val = neg_inf
                best_id = Int32(-1)
                expert = Int32(0)
                while expert < Int32(self.num_sparse_experts):
                    skip = Int32(0)
                    for prev in cutlass.range_constexpr(self.top_k):
                        if prev < slot and expert == top_ids_local[prev]:
                            skip = Int32(1)
                    if skip == Int32(0):
                        acc = cutlass.Float32(0.0)
                        col1 = Int32(0)
                        while col1 < Int32(self.hidden_size):
                            acc += cutlass.Float32(sNorm[col1]) * cutlass.Float32(sparse_gate_weight[expert, col1])
                            col1 += Int32(1)
                        candidate_val = cutlass.Float32(self.input_dtype(acc))
                        if candidate_val > best_val or (candidate_val == best_val and expert > best_id):
                            best_val = candidate_val
                            best_id = expert
                    expert += Int32(1)
                top_vals[slot] = best_val
                top_ids_local[slot] = best_id

            if self.renormalize_topk:
                max_logit = top_vals[0]
                exp_vals = [cutlass.Float32(0.0) for _ in range(self.top_k)]
                denom = cutlass.Float32(0.0)
                for slot in cutlass.range_constexpr(self.top_k):
                    exp_val = _exp2_approx_ftz_f32((top_vals[slot] - max_logit) * cutlass.Float32(LOG2_E))
                    exp_vals[slot] = exp_val
                    denom += exp_val
                inv_denom = cutlass.Float32(1.0) / denom
                for slot in cutlass.range_constexpr(self.top_k):
                    weight = exp_vals[slot] * inv_denom
                    slot_i = Int32(slot)
                    sSelectedIds[slot_i] = top_ids_local[slot]
                    sSelectedWeights[slot_i] = weight
                    topk_offset = bidx * Int32(self.combined_top_k) + slot_i
                    st_global_i32(get_ptr_as_int64(topk_ids, topk_offset), top_ids_local[slot])
                    st_global_f32(get_ptr_as_int64(topk_weights, topk_offset), weight)
            else:
                for slot in cutlass.range_constexpr(self.top_k):
                    weight = top_vals[slot]
                    slot_i = Int32(slot)
                    sSelectedIds[slot_i] = top_ids_local[slot]
                    sSelectedWeights[slot_i] = weight
                    topk_offset = bidx * Int32(self.combined_top_k) + slot_i
                    st_global_i32(get_ptr_as_int64(topk_ids, topk_offset), top_ids_local[slot])
                    st_global_f32(get_ptr_as_int64(topk_weights, topk_offset), weight)

            shared_slot = Int32(self.top_k)
            sSelectedIds[shared_slot] = Int32(self.num_sparse_experts)
            sSelectedWeights[shared_slot] = shared_gate
            topk_offset = bidx * Int32(self.combined_top_k) + shared_slot
            st_global_i32(get_ptr_as_int64(topk_ids, topk_offset), Int32(self.num_sparse_experts))
            st_global_f32(get_ptr_as_int64(topk_weights, topk_offset), shared_gate)

            for slot in cutlass.range_constexpr(self.combined_top_k):
                slot_i = Int32(slot)
                expert_id = sSelectedIds[slot_i]
                route_weight = sSelectedWeights[slot_i]
                gs1 = cutlass.Float32(input_global_scale[expert_id])
                alpha1 = cutlass.Float32(w1_alpha[expert_id])

                block_idx = Int32(0)
                while block_idx < Int32(self.hidden_size // _FP4_BLOCK):
                    block_start = block_idx * Int32(_FP4_BLOCK)
                    values = cute.make_rmem_tensor((_FP4_BLOCK,), cutlass.Float32)
                    block_max = cutlass.Float32(0.0)
                    elem = Int32(0)
                    while elem < Int32(_FP4_BLOCK):
                        value = cutlass.Float32(sNorm[block_start + elem])
                        values[elem] = value
                        block_max = fmax_f32(block_max, fabs_f32(value))
                        elem += Int32(1)
                    _, scale_byte = quantize_block_fp4(values, block_max, gs1)
                    # Thresholds are in true value-space (sf * gs), but the A operand
                    # dequant contract carries only the FP8 blockscale (sf). The coarse
                    # global scale is reflected in alpha on the consumer side.
                    qscale = fp8_e4m3_to_f32(Uint32(scale_byte))
                    threshold_scale = qscale * gs1
                    elem = Int32(0)
                    while elem < Int32(_FP4_BLOCK):
                        sX[block_start + elem] = _fp4_quantize_dequant_value(values[elem], threshold_scale) * qscale
                        elem += Int32(1)
                    block_idx += Int32(1)

                row = Int32(0)
                while row < Int32(self.intermediate_size_local):
                    up_acc = cutlass.Float32(0.0)
                    gate_acc = cutlass.Float32(0.0)
                    block_idx = Int32(0)
                    while block_idx < Int32(self.hidden_size // _FP4_BLOCK):
                        scale_up = fp8_e4m3_to_f32(Uint32(w1_scale_u8[expert_id, row, block_idx]))
                        scale_gate = fp8_e4m3_to_f32(Uint32(w1_scale_u8[expert_id, row + Int32(self.intermediate_size_local), block_idx]))
                        byte_base = block_idx * Int32(8)
                        elem = Int32(0)
                        while elem < Int32(_FP4_BLOCK):
                            packed_byte_up = Uint8(w1_fp4_u8[expert_id, row, byte_base + (elem >> Int32(1))])
                            packed_byte_gate = Uint8(w1_fp4_u8[expert_id, row + Int32(self.intermediate_size_local), byte_base + (elem >> Int32(1))])
                            nibble_up = packed_byte_up & Uint8(0x0F) if (elem & Int32(1)) == Int32(0) else ((packed_byte_up >> Uint8(4)) & Uint8(0x0F))
                            nibble_gate = packed_byte_gate & Uint8(0x0F) if (elem & Int32(1)) == Int32(0) else ((packed_byte_gate >> Uint8(4)) & Uint8(0x0F))
                            xval = sX[block_idx * Int32(_FP4_BLOCK) + elem]
                            up_acc += (_fp4_nibble_to_f32(nibble_up) * scale_up) * xval
                            gate_acc += (_fp4_nibble_to_f32(nibble_gate) * scale_gate) * xval
                            elem += Int32(1)
                        block_idx += Int32(1)
                    sUp[row] = up_acc * alpha1
                    sGate[row] = gate_acc * alpha1
                    row += Int32(1)

                gs2 = cutlass.Float32(fc2_global_scale[expert_id])
                row = Int32(0)
                while row < Int32(self.intermediate_size_local):
                    inter = _sigmoid_f32(sGate[row]) * sGate[row] * sUp[row]
                    sInt[row] = inter
                    row += Int32(1)

                block_idx = Int32(0)
                while block_idx < Int32(self.intermediate_size_local // _FP4_BLOCK):
                    block_start = block_idx * Int32(_FP4_BLOCK)
                    values = cute.make_rmem_tensor((_FP4_BLOCK,), cutlass.Float32)
                    block_max = cutlass.Float32(0.0)
                    elem = Int32(0)
                    while elem < Int32(_FP4_BLOCK):
                        value = sInt[block_start + elem]
                        values[elem] = value
                        block_max = fmax_f32(block_max, fabs_f32(value))
                        elem += Int32(1)
                    _, scale_byte = quantize_block_fp4(values, block_max, gs2)
                    qscale = fp8_e4m3_to_f32(Uint32(scale_byte))
                    threshold_scale = qscale * gs2
                    elem = Int32(0)
                    while elem < Int32(_FP4_BLOCK):
                        sInt[block_start + elem] = _fp4_quantize_dequant_value(values[elem], threshold_scale) * qscale
                        elem += Int32(1)
                    block_idx += Int32(1)

                alpha2 = cutlass.Float32(w2_alpha[expert_id])
                out_col = Int32(0)
                while out_col < Int32(self.hidden_size):
                    acc = cutlass.Float32(0.0)
                    block_idx = Int32(0)
                    while block_idx < Int32(self.intermediate_size_local // _FP4_BLOCK):
                        scale_down = fp8_e4m3_to_f32(Uint32(w2_scale_u8[expert_id, out_col, block_idx]))
                        byte_base = block_idx * Int32(8)
                        elem = Int32(0)
                        while elem < Int32(_FP4_BLOCK):
                            packed_byte = Uint8(w2_fp4_u8[expert_id, out_col, byte_base + (elem >> Int32(1))])
                            nibble = packed_byte & Uint8(0x0F) if (elem & Int32(1)) == Int32(0) else ((packed_byte >> Uint8(4)) & Uint8(0x0F))
                            acc += (_fp4_nibble_to_f32(nibble) * scale_down) * sInt[block_idx * Int32(_FP4_BLOCK) + elem]
                            elem += Int32(1)
                        block_idx += Int32(1)
                    sOut[out_col] = sOut[out_col] + route_weight * (acc * alpha2)
                    out_col += Int32(1)
        cute.arch.sync_threads()

        col = tidx
        while col < Int32(self.hidden_size):
            output[bidx, col] = self.input_dtype(sOut[col])
            col += Int32(THREADS_PER_BLOCK)


class _MicroMegaFusedLaunch:
    def __init__(
        self,
        *,
        world_size: int,
        num_tokens: int,
        hidden_size: int,
        intermediate_size_local: int,
        num_sparse_experts: int,
        top_k: int,
        input_dtype: torch.dtype,
        renormalize_topk: bool,
    ):
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.intermediate_size_local = intermediate_size_local
        self.num_sparse_experts = num_sparse_experts
        self.top_k = top_k
        self.combined_top_k = top_k + 1
        self.input_cutlass_dtype = cutlass_dtype(input_dtype)
        self.kernel = _MicroMegaFusedKernel(
            world_size=world_size,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size_local=intermediate_size_local,
            num_sparse_experts=num_sparse_experts,
            top_k=top_k,
            input_dtype=self.input_cutlass_dtype,
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
        residual_out_ptr: cute.Pointer,
        normalized_ptr: cute.Pointer,
        norm_weight_ptr: cute.Pointer,
        sparse_gate_weight_ptr: cute.Pointer,
        shared_gate_weight_ptr: cute.Pointer,
        w1_fp4_u8_ptr: cute.Pointer,
        w1_scale_u8_ptr: cute.Pointer,
        w1_alpha_ptr: cute.Pointer,
        input_scale_ptr: cute.Pointer,
        w2_fp4_u8_ptr: cute.Pointer,
        w2_scale_u8_ptr: cute.Pointer,
        w2_alpha_ptr: cute.Pointer,
        fc2_scale_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
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
        row_layout = cute.make_layout((self.num_tokens, self.hidden_size), stride=(self.hidden_size, 1))
        sparse_gate_layout = cute.make_layout((self.num_sparse_experts, self.hidden_size), stride=(self.hidden_size, 1))
        shared_gate_layout = cute.make_layout((self.hidden_size,), stride=(1,))
        w1_layout = cute.make_layout(
            (self.num_sparse_experts + 1, 2 * self.intermediate_size_local, self.hidden_size // 2),
            stride=(2 * self.intermediate_size_local * (self.hidden_size // 2), self.hidden_size // 2, 1),
        )
        w1s_layout = cute.make_layout(
            (self.num_sparse_experts + 1, 2 * self.intermediate_size_local, self.hidden_size // _FP4_BLOCK),
            stride=(2 * self.intermediate_size_local * (self.hidden_size // _FP4_BLOCK), self.hidden_size // _FP4_BLOCK, 1),
        )
        w2_layout = cute.make_layout(
            (self.num_sparse_experts + 1, self.hidden_size, self.intermediate_size_local // 2),
            stride=(self.hidden_size * (self.intermediate_size_local // 2), self.intermediate_size_local // 2, 1),
        )
        w2s_layout = cute.make_layout(
            (self.num_sparse_experts + 1, self.hidden_size, self.intermediate_size_local // _FP4_BLOCK),
            stride=(self.hidden_size * (self.intermediate_size_local // _FP4_BLOCK), self.intermediate_size_local // _FP4_BLOCK, 1),
        )
        vector_layout = cute.make_layout((self.num_sparse_experts + 1,), stride=(1,))
        topk_layout = cute.make_layout((self.num_tokens * self.combined_top_k,), stride=(1,))
        self.kernel(
            cute.make_tensor(inp0_ptr, layout=row_layout),
            cute.make_tensor(inp1_ptr, layout=row_layout),
            cute.make_tensor(inp2_ptr, layout=row_layout),
            cute.make_tensor(inp3_ptr, layout=row_layout),
            cute.make_tensor(inp4_ptr, layout=row_layout),
            cute.make_tensor(inp5_ptr, layout=row_layout),
            cute.make_tensor(inp6_ptr, layout=row_layout),
            cute.make_tensor(inp7_ptr, layout=row_layout),
            cute.make_tensor(residual_ptr, layout=row_layout),
            cute.make_tensor(residual_out_ptr, layout=row_layout),
            cute.make_tensor(normalized_ptr, layout=row_layout),
            cute.make_tensor(norm_weight_ptr, layout=shared_gate_layout),
            cute.make_tensor(sparse_gate_weight_ptr, layout=sparse_gate_layout),
            cute.make_tensor(shared_gate_weight_ptr, layout=shared_gate_layout),
            cute.make_tensor(w1_fp4_u8_ptr, layout=w1_layout),
            cute.make_tensor(w1_scale_u8_ptr, layout=w1s_layout),
            cute.make_tensor(w1_alpha_ptr, layout=vector_layout),
            cute.make_tensor(input_scale_ptr, layout=vector_layout),
            cute.make_tensor(w2_fp4_u8_ptr, layout=w2_layout),
            cute.make_tensor(w2_scale_u8_ptr, layout=w2s_layout),
            cute.make_tensor(w2_alpha_ptr, layout=vector_layout),
            cute.make_tensor(fc2_scale_ptr, layout=vector_layout),
            cute.make_tensor(topk_ids_ptr, layout=topk_layout),
            cute.make_tensor(topk_weights_ptr, layout=topk_layout),
            cute.make_tensor(output_ptr, layout=row_layout),
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


@lru_cache(maxsize=64)
def _get_megafused_kernel(
    world_size: int,
    num_tokens: int,
    hidden_size: int,
    intermediate_size_local: int,
    num_sparse_experts: int,
    top_k: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    renormalize_topk: bool,
):
    launch = _MicroMegaFusedLaunch(
        world_size=world_size,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size_local=intermediate_size_local,
        num_sparse_experts=num_sparse_experts,
        top_k=top_k,
        input_dtype=input_dtype,
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
    u8_fake = make_ptr(cutlass.Uint8, 1, cute.AddressSpace.gmem, assumed_align=1)
    i32_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    signal_fake = make_ptr(cutlass.Int32, 128, cute.AddressSpace.gmem, assumed_align=128)
    return cute.compile(
        launch,
        row_fake, row_fake, row_fake, row_fake,
        row_fake, row_fake, row_fake, row_fake,
        row_fake, row_fake, row_fake,
        weight_fake, weight_fake, weight_fake,
        u8_fake, u8_fake,
        f32_fake, f32_fake,
        u8_fake, u8_fake,
        f32_fake, f32_fake,
        i32_fake, f32_fake, row_fake,
        signal_fake, signal_fake, signal_fake, signal_fake,
        signal_fake, signal_fake, signal_fake, signal_fake,
        signal_fake,
        0, 1.0, current_cuda_stream(),
    )


def megafused_allreduce_route_compute_micro(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    sparse_gate_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    *,
    w1_fp4_u8: torch.Tensor,
    w1_scale_u8: torch.Tensor,
    w1_alpha: torch.Tensor,
    input_global_scale: torch.Tensor,
    w2_fp4_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    w2_alpha: torch.Tensor,
    fc2_global_scale: torch.Tensor,
    ipc: UnifiedPreMLPIPC,
    top_k: int,
    renormalize_topk: bool,
    eps: float,
    output: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    normalized_out: torch.Tensor | None = None,
):
    if output is None:
        output = torch.empty_like(hidden_states)
    if residual_out is None:
        residual_out = torch.empty_like(hidden_states)
    if normalized_out is None:
        normalized_out = torch.empty_like(hidden_states)
    topk_ids_out = torch.empty(
        (hidden_states.shape[0], top_k + 1),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    topk_weights_out = torch.empty(
        (hidden_states.shape[0], top_k + 1),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    w1_scale_unswizzled = _unswizzle_scale_u8_batch(
        w1_scale_u8.view(torch.float8_e4m3fn),
        rows=w1_fp4_u8.shape[1],
        cols_blocks=hidden_states.shape[1] // _FP4_BLOCK,
    )
    w2_scale_unswizzled = _unswizzle_scale_u8_batch(
        w2_scale_u8.view(torch.float8_e4m3fn),
        rows=hidden_states.shape[1],
        cols_blocks=(w2_fp4_u8.shape[2] * 2) // _FP4_BLOCK,
    )

    compiled = _get_megafused_kernel(
        ipc.world_size,
        hidden_states.shape[0],
        hidden_states.shape[1],
        w2_fp4_u8.shape[2] * 2,
        sparse_gate_weight.shape[0],
        top_k,
        hidden_states.dtype,
        norm_weight.dtype,
        renormalize_topk,
    )

    input_ptr_args = [
        make_ptr(cutlass_dtype(hidden_states.dtype), ptr, cute.AddressSpace.gmem, assumed_align=align_bytes(hidden_states.dtype))
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
        make_ptr(cutlass_dtype(hidden_states.dtype), int(residual.data_ptr()), cute.AddressSpace.gmem, assumed_align=align_bytes(hidden_states.dtype)),
        make_ptr(cutlass_dtype(hidden_states.dtype), int(residual_out.data_ptr()), cute.AddressSpace.gmem, assumed_align=align_bytes(hidden_states.dtype)),
        make_ptr(cutlass_dtype(hidden_states.dtype), int(normalized_out.data_ptr()), cute.AddressSpace.gmem, assumed_align=align_bytes(hidden_states.dtype)),
        make_ptr(cutlass_dtype(norm_weight.dtype), int(norm_weight.data_ptr()), cute.AddressSpace.gmem, assumed_align=align_bytes(norm_weight.dtype)),
        make_ptr(cutlass_dtype(sparse_gate_weight.dtype), int(sparse_gate_weight.data_ptr()), cute.AddressSpace.gmem, assumed_align=align_bytes(sparse_gate_weight.dtype)),
        make_ptr(cutlass_dtype(shared_gate_weight.dtype), int(shared_gate_weight.data_ptr()), cute.AddressSpace.gmem, assumed_align=align_bytes(shared_gate_weight.dtype)),
        make_ptr(cutlass.Uint8, int(w1_fp4_u8.data_ptr()), cute.AddressSpace.gmem, assumed_align=1),
        make_ptr(cutlass.Uint8, int(w1_scale_unswizzled.data_ptr()), cute.AddressSpace.gmem, assumed_align=1),
        make_ptr(cutlass.Float32, int(w1_alpha.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, int(input_global_scale.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, int(w2_fp4_u8.data_ptr()), cute.AddressSpace.gmem, assumed_align=1),
        make_ptr(cutlass.Uint8, int(w2_scale_unswizzled.data_ptr()), cute.AddressSpace.gmem, assumed_align=1),
        make_ptr(cutlass.Float32, int(w2_alpha.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, int(fc2_global_scale.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, int(topk_ids_out.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, int(topk_weights_out.view(-1).data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass_dtype(hidden_states.dtype), int(output.data_ptr()), cute.AddressSpace.gmem, assumed_align=align_bytes(hidden_states.dtype)),
        *signal_ptr_args[:8],
        make_ptr(cutlass.Int32, ipc.signal_ptrs[ipc.rank], cute.AddressSpace.gmem, assumed_align=128),
        ipc.rank,
        float(eps),
        current_cuda_stream(),
    )
    return output, residual_out, normalized_out, topk_ids_out, topk_weights_out


__all__ = [
    "megafused_allreduce_route_compute_micro",
]
