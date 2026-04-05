"""Unified static pre-MLP producer scaffolding and slice-A kernel."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, T, dsl_user_op

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
    warp_reduce,
)
from b12x.moe.fused.pre_mlp_route_pack import build_compact_route_metadata

LOG2_E = 1.4426950408889634


LEVEL_TILE_M = 128
NVFP4_BLOCK_SIZE = 16


@dataclass(frozen=True, kw_only=True)
class UnifiedPreMLPStaticContract:
    """Logical contract for the future unified static producer kernel."""

    hidden_size: int
    sparse_experts: int
    sparse_top_k: int
    shared_experts: int = 1
    tile_rows: int = LEVEL_TILE_M
    fp4_block_size: int = NVFP4_BLOCK_SIZE

    @property
    def total_experts(self) -> int:
        return self.sparse_experts + self.shared_experts

    @property
    def combined_top_k(self) -> int:
        return self.sparse_top_k + self.shared_experts

    @property
    def shared_expert_id(self) -> int:
        return self.sparse_experts

    @property
    def packed_hidden_size(self) -> int:
        return self.hidden_size // 2

    @property
    def blockscale_cols(self) -> int:
        cols_blocks = self.hidden_size // self.fp4_block_size
        return ((cols_blocks + 3) // 4) * 4

    def routed_rows(self, num_tokens: int) -> int:
        return num_tokens * self.combined_top_k

    def static_tile_count(self, num_tokens: int) -> int:
        rows = self.routed_rows(num_tokens)
        return (rows + self.tile_rows - 1) // self.tile_rows

    def active_expert_capacity(self, num_tokens: int) -> int:
        return min(self.total_experts, self.routed_rows(num_tokens))

    def workspace_shapes(self, num_tokens: int) -> dict[str, tuple[int, ...]]:
        routed_rows = self.routed_rows(num_tokens)
        state_e = self.active_expert_capacity(num_tokens)
        tiles = self.static_tile_count(num_tokens)
        return {
            "row_counts": (state_e,),
            "active_expert_count": (1,),
            "weight_expert_ids": (state_e,),
            "global_to_local_expert": (self.total_experts,),
            "token_map": (state_e, routed_rows),
            "token_weights": (state_e, routed_rows),
            "packed_input": (state_e, routed_rows, self.packed_hidden_size),
            "packed_input_scale": (
                state_e,
                ((routed_rows + self.tile_rows - 1) // self.tile_rows) * self.tile_rows,
                self.blockscale_cols,
            ),
            "fc1_tile_scale": (state_e, tiles),
            "fc1_tile_alpha": (state_e, tiles),
        }


@dataclass(frozen=True, kw_only=True)
class UnifiedPreMLPIPC:
    """Generic pointer-based IPC descriptor consumed by the producer kernel."""

    rank: int
    world_size: int
    signal_ptrs: tuple[int, ...]
    peer_input_ptrs: tuple[int, ...]

    def validate(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if len(self.signal_ptrs) != self.world_size:
            raise ValueError("signal_ptrs must match world_size")
        if len(self.peer_input_ptrs) != self.world_size:
            raise ValueError("peer_input_ptrs must match world_size")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank {self.rank} out of range for world_size {self.world_size}")

    @classmethod
    def from_oneshot_runtime(
        cls,
        runtime,
        *,
        inp: torch.Tensor,
        peer_input_ptrs: Sequence[int] | None = None,
    ) -> "UnifiedPreMLPIPC":
        if peer_input_ptrs is None:
            if not hasattr(runtime, "_select_peer_input_ptrs"):
                raise TypeError("runtime does not expose _select_peer_input_ptrs")
            peer_ptrs = tuple(int(ptr) for ptr in runtime._select_peer_input_ptrs(inp, None))
        else:
            peer_ptrs = tuple(int(ptr) for ptr in peer_input_ptrs)
        return cls(
            rank=int(runtime.rank),
            world_size=int(runtime.world_size),
            signal_ptrs=tuple(int(ptr) for ptr in runtime.signal_ptrs),
            peer_input_ptrs=peer_ptrs,
        )


@dataclass(frozen=True, kw_only=True)
class UnifiedPreMLPStaticLaunchConfig:
    """Launch-shape contract for the first producer slice."""

    threads_per_block: int = THREADS_PER_BLOCK
    max_blocks: int = MAX_BLOCKS
    ctas_per_token: int = 1
    control_warps: int = 1

    @property
    def compute_warps(self) -> int:
        return max(0, self.threads_per_block // 32 - self.control_warps)

    def validate(self) -> None:
        if self.ctas_per_token != 1:
            raise ValueError("slice A only supports one CTA per token")
        if self.threads_per_block != THREADS_PER_BLOCK:
            raise ValueError(f"slice A expects {THREADS_PER_BLOCK} threads per block")
        if self.max_blocks > MAX_BLOCKS:
            raise ValueError(f"slice A supports at most {MAX_BLOCKS} token CTAs")


@dataclass(frozen=True, kw_only=True)
class UnifiedPreMLPSliceAOutputs:
    normalized: torch.Tensor
    residual_out: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class UnifiedPreMLPSliceBOutputs:
    router_logits: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    shared_gate: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class UnifiedPreMLPSliceCOutputs:
    active_expert_count: torch.Tensor
    weight_expert_ids: torch.Tensor
    global_to_local_expert: torch.Tensor
    row_counts: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class UnifiedPreMLPSliceDOutputs:
    packed_input: torch.Tensor
    packed_input_scale: torch.Tensor
    fc1_tile_scale: torch.Tensor
    fc1_tile_alpha: torch.Tensor


@dsl_user_op
def _exp2_approx_ftz_f32(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class _UnifiedPreMLPSliceAKernel:
    def __init__(self, *, world_size: int, num_tokens: int, hidden_size: int, element_dtype):
        self.world_size = world_size
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.element_dtype = element_dtype

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
        weight: cute.Tensor,
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
            residual_val = self.element_dtype(acc + self.element_dtype(residual_in[bidx, col]))
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
            gamma = cutlass.Float32(1.0) + cutlass.Float32(weight[col])
            out = cutlass.Float32(residual_out[bidx, col]) * inv_scale * gamma
            normalized_out[bidx, col] = self.element_dtype(out)
            col += Int32(THREADS_PER_BLOCK)


class _UnifiedPreMLPSliceALaunch:
    def __init__(
        self,
        *,
        world_size: int,
        num_tokens: int,
        hidden_size: int,
        input_dtype: torch.dtype,
        weight_dtype: torch.dtype,
    ):
        self.world_size = world_size
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.input_cutlass_dtype = cutlass_dtype(input_dtype)
        self.weight_cutlass_dtype = cutlass_dtype(weight_dtype)
        self.kernel = _UnifiedPreMLPSliceAKernel(
            world_size=world_size,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            element_dtype=self.input_cutlass_dtype,
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
        weight_ptr: cute.Pointer,
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
        weight_layout = cute.make_layout((self.hidden_size,), stride=(1,))
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
        residual = cute.make_tensor(residual_ptr, layout=row_layout)
        normalized_out = cute.make_tensor(normalized_ptr, layout=row_layout)
        residual_out = cute.make_tensor(residual_out_ptr, layout=row_layout)
        weight = cute.make_tensor(weight_ptr, layout=weight_layout)
        self.kernel(
            *inputs,
            residual,
            normalized_out,
            residual_out,
            weight,
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


class _UnifiedPreMLPSliceBKernel:
    def __init__(
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        input_element_dtype,
        weight_element_dtype,
    ):
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_element_dtype = input_element_dtype
        self.weight_element_dtype = weight_element_dtype

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
        normalized_in: cute.Tensor,
        sparse_gate_weight: cute.Tensor,
        shared_gate_weight: cute.Tensor,
        router_logits_out: cute.Tensor,
        topk_ids_out: cute.Tensor,
        topk_weights_out: cute.Tensor,
        shared_gate_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tidx = Int32(tidx)
        bidx = Int32(bidx)
        if tidx == Int32(0):
            top_vals = [cutlass.Float32(-3.4028235e38) for _ in range(self.top_k)]
            top_ids = [Int32(-1) for _ in range(self.top_k)]

            shared_acc = cutlass.Float32(0.0)
            col = Int32(0)
            while col < Int32(self.hidden_size):
                x = cutlass.Float32(normalized_in[bidx, col])
                w = cutlass.Float32(shared_gate_weight[0, col])
                shared_acc += x * w
                col += Int32(1)
            shared_logit = self.input_element_dtype(shared_acc)
            neg_exp = _exp2_approx_ftz_f32(cutlass.Float32(-LOG2_E) * cutlass.Float32(shared_logit))
            shared_gate = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + neg_exp)
            shared_gate_out[bidx, 0] = shared_gate

            expert = Int32(0)
            while expert < Int32(self.num_experts):
                acc = cutlass.Float32(0.0)
                col = Int32(0)
                while col < Int32(self.hidden_size):
                    x = cutlass.Float32(normalized_in[bidx, col])
                    w = cutlass.Float32(sparse_gate_weight[expert, col])
                    acc += x * w
                    col += Int32(1)
                rounded = self.input_element_dtype(acc)
                rounded_f32 = cutlass.Float32(rounded)
                router_logits_out[bidx, expert] = rounded
                candidate_val = rounded_f32
                candidate_id = expert
                for slot in cutlass.range_constexpr(self.top_k):
                    if candidate_val > top_vals[slot]:
                        prev_val = top_vals[slot]
                        prev_id = top_ids[slot]
                        top_vals[slot] = candidate_val
                        top_ids[slot] = candidate_id
                        candidate_val = prev_val
                        candidate_id = prev_id
                expert += Int32(1)

            max_logit = top_vals[0]
            exp_vals = [cutlass.Float32(0.0) for _ in range(self.top_k)]
            denom = cutlass.Float32(0.0)
            for slot in cutlass.range_constexpr(self.top_k):
                exp_val = _exp2_approx_ftz_f32((top_vals[slot] - max_logit) * cutlass.Float32(LOG2_E))
                exp_vals[slot] = exp_val
                denom += exp_val
            inv_denom = cutlass.Float32(1.0) / denom
            for slot in cutlass.range_constexpr(self.top_k):
                topk_ids_out[bidx, slot] = top_ids[slot]
                topk_weights_out[bidx, slot] = exp_vals[slot] * inv_denom


class _UnifiedPreMLPSliceBLaunch:
    def __init__(
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        input_dtype: torch.dtype,
        weight_dtype: torch.dtype,
    ):
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_cutlass_dtype = cutlass_dtype(input_dtype)
        self.weight_cutlass_dtype = cutlass_dtype(weight_dtype)
        self.kernel = _UnifiedPreMLPSliceBKernel(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            input_element_dtype=self.input_cutlass_dtype,
            weight_element_dtype=self.weight_cutlass_dtype,
        )

    @cute.jit
    def __call__(
        self,
        normalized_ptr: cute.Pointer,
        sparse_gate_weight_ptr: cute.Pointer,
        shared_gate_weight_ptr: cute.Pointer,
        router_logits_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        shared_gate_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        row_layout = cute.make_layout((self.num_tokens, self.hidden_size), stride=(self.hidden_size, 1))
        sparse_gate_layout = cute.make_layout((self.num_experts, self.hidden_size), stride=(self.hidden_size, 1))
        shared_gate_layout = cute.make_layout((1, self.hidden_size), stride=(self.hidden_size, 1))
        router_logits_layout = cute.make_layout((self.num_tokens, self.num_experts), stride=(self.num_experts, 1))
        topk_ids_layout = cute.make_layout((self.num_tokens, self.top_k), stride=(self.top_k, 1))
        topk_weights_layout = cute.make_layout((self.num_tokens, self.top_k), stride=(self.top_k, 1))
        shared_gate_out_layout = cute.make_layout((self.num_tokens, 1), stride=(1, 1))
        self.kernel(
            cute.make_tensor(normalized_ptr, layout=row_layout),
            cute.make_tensor(sparse_gate_weight_ptr, layout=sparse_gate_layout),
            cute.make_tensor(shared_gate_weight_ptr, layout=shared_gate_layout),
            cute.make_tensor(router_logits_ptr, layout=router_logits_layout),
            cute.make_tensor(topk_ids_ptr, layout=topk_ids_layout),
            cute.make_tensor(topk_weights_ptr, layout=topk_weights_layout),
            cute.make_tensor(shared_gate_ptr, layout=shared_gate_out_layout),
            stream,
        )


@lru_cache(maxsize=128)
def _get_slice_a_kernel(
    world_size: int,
    num_tokens: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
):
    launch = _UnifiedPreMLPSliceALaunch(
        world_size=world_size,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
    )
    ptr_align = align_bytes(input_dtype)
    weight_align = align_bytes(weight_dtype)
    input_fake = make_ptr(cutlass_dtype(input_dtype), max(16, ptr_align), cute.AddressSpace.gmem, assumed_align=ptr_align)
    residual_fake = make_ptr(cutlass_dtype(input_dtype), max(16, ptr_align), cute.AddressSpace.gmem, assumed_align=ptr_align)
    normalized_fake = make_ptr(cutlass_dtype(input_dtype), max(16, ptr_align), cute.AddressSpace.gmem, assumed_align=ptr_align)
    residual_out_fake = make_ptr(cutlass_dtype(input_dtype), max(16, ptr_align), cute.AddressSpace.gmem, assumed_align=ptr_align)
    weight_fake = make_ptr(cutlass_dtype(weight_dtype), max(16, weight_align), cute.AddressSpace.gmem, assumed_align=weight_align)
    signal_fake = make_ptr(cutlass.Int32, 128, cute.AddressSpace.gmem, assumed_align=128)
    return cute.compile(
        launch,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        input_fake,
        residual_fake,
        normalized_fake,
        residual_out_fake,
        weight_fake,
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


@lru_cache(maxsize=128)
def _get_slice_b_kernel(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
):
    launch = _UnifiedPreMLPSliceBLaunch(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
    )
    input_align = align_bytes(input_dtype)
    weight_align = align_bytes(weight_dtype)
    normalized_fake = make_ptr(cutlass_dtype(input_dtype), max(16, input_align), cute.AddressSpace.gmem, assumed_align=input_align)
    sparse_gate_weight_fake = make_ptr(cutlass_dtype(weight_dtype), max(16, weight_align), cute.AddressSpace.gmem, assumed_align=weight_align)
    shared_gate_weight_fake = make_ptr(cutlass_dtype(weight_dtype), max(16, weight_align), cute.AddressSpace.gmem, assumed_align=weight_align)
    router_logits_fake = make_ptr(cutlass_dtype(input_dtype), max(16, input_align), cute.AddressSpace.gmem, assumed_align=input_align)
    topk_ids_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    topk_weights_fake = make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4)
    shared_gate_fake = make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4)
    return cute.compile(
        launch,
        normalized_fake,
        sparse_gate_weight_fake,
        shared_gate_weight_fake,
        router_logits_fake,
        topk_ids_fake,
        topk_weights_fake,
        shared_gate_fake,
        current_cuda_stream(),
    )


def qwen35_static_contract() -> UnifiedPreMLPStaticContract:
    return UnifiedPreMLPStaticContract(
        hidden_size=4096,
        sparse_experts=512,
        sparse_top_k=10,
    )


def slice_a_allreduce_residual_gemma_rmsnorm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    ipc: UnifiedPreMLPIPC,
    launch_config: UnifiedPreMLPStaticLaunchConfig | None = None,
    normalized_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> UnifiedPreMLPSliceAOutputs:
    ipc.validate()
    config = launch_config or UnifiedPreMLPStaticLaunchConfig()
    config.validate()
    if hidden_states.shape != residual.shape:
        raise ValueError("hidden_states and residual must have the same shape")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [num_tokens, hidden_size]")
    if hidden_states.shape[0] > config.max_blocks:
        raise ValueError(
            f"slice A supports at most {config.max_blocks} tokens, got {hidden_states.shape[0]}"
        )
    if hidden_states.device != residual.device or hidden_states.device != norm_weight.device:
        raise ValueError("all tensors must be on the same CUDA device")
    if not hidden_states.is_contiguous() or not residual.is_contiguous() or not norm_weight.is_contiguous():
        raise ValueError("slice A expects contiguous inputs")
    if norm_weight.ndim != 1 or norm_weight.numel() != hidden_states.shape[1]:
        raise ValueError("norm_weight must be [hidden_size]")
    if normalized_out is None:
        normalized_out = torch.empty_like(hidden_states)
    if residual_out is None:
        residual_out = torch.empty_like(hidden_states)
    if normalized_out.shape != hidden_states.shape or residual_out.shape != hidden_states.shape:
        raise ValueError("slice A outputs must match input shape")
    if normalized_out.dtype != hidden_states.dtype or residual_out.dtype != hidden_states.dtype:
        raise ValueError("slice A outputs must match input dtype")
    compiled = _get_slice_a_kernel(
        ipc.world_size,
        int(hidden_states.shape[0]),
        int(hidden_states.shape[1]),
        hidden_states.dtype,
        norm_weight.dtype,
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
        *signal_ptr_args[:8],
        make_ptr(cutlass.Int32, ipc.signal_ptrs[ipc.rank], cute.AddressSpace.gmem, assumed_align=128),
        ipc.rank,
        float(eps),
        current_cuda_stream(),
    )
    return UnifiedPreMLPSliceAOutputs(
        normalized=normalized_out,
        residual_out=residual_out,
    )


def slice_b_sparse_routing_shared_gate(
    normalized_hidden_states: torch.Tensor,
    sparse_gate_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    *,
    top_k: int,
    launch_config: UnifiedPreMLPStaticLaunchConfig | None = None,
    router_logits_out: torch.Tensor | None = None,
    topk_ids_out: torch.Tensor | None = None,
    topk_weights_out: torch.Tensor | None = None,
    shared_gate_out: torch.Tensor | None = None,
) -> UnifiedPreMLPSliceBOutputs:
    config = launch_config or UnifiedPreMLPStaticLaunchConfig()
    config.validate()
    if normalized_hidden_states.ndim != 2:
        raise ValueError("normalized_hidden_states must be [num_tokens, hidden_size]")
    if normalized_hidden_states.shape[0] > config.max_blocks:
        raise ValueError(
            f"slice B supports at most {config.max_blocks} tokens, got {normalized_hidden_states.shape[0]}"
        )
    if sparse_gate_weight.ndim != 2:
        raise ValueError("sparse_gate_weight must be [num_experts, hidden_size]")
    if sparse_gate_weight.shape[1] != normalized_hidden_states.shape[1]:
        raise ValueError("sparse_gate_weight hidden-size mismatch")
    if shared_gate_weight.ndim == 1:
        shared_gate_weight = shared_gate_weight.unsqueeze(0)
    if shared_gate_weight.shape != (1, normalized_hidden_states.shape[1]):
        raise ValueError("shared_gate_weight must be [hidden_size] or [1, hidden_size]")
    if top_k <= 0 or top_k > sparse_gate_weight.shape[0]:
        raise ValueError(f"invalid top_k={top_k} for num_experts={sparse_gate_weight.shape[0]}")
    if not normalized_hidden_states.is_contiguous():
        raise ValueError("slice B expects contiguous normalized_hidden_states")
    if not sparse_gate_weight.is_contiguous() or not shared_gate_weight.is_contiguous():
        raise ValueError("slice B expects contiguous gate weights")
    device = normalized_hidden_states.device
    num_tokens, num_experts = normalized_hidden_states.shape[0], sparse_gate_weight.shape[0]
    if router_logits_out is None:
        router_logits_out = torch.empty(
            num_tokens,
            num_experts,
            device=device,
            dtype=normalized_hidden_states.dtype,
        )
    if topk_ids_out is None:
        topk_ids_out = torch.empty(num_tokens, top_k, device=device, dtype=torch.int32)
    if topk_weights_out is None:
        topk_weights_out = torch.empty(num_tokens, top_k, device=device, dtype=torch.float32)
    if shared_gate_out is None:
        shared_gate_out = torch.empty(num_tokens, 1, device=device, dtype=torch.float32)
    compiled = _get_slice_b_kernel(
        num_tokens,
        normalized_hidden_states.shape[1],
        num_experts,
        top_k,
        normalized_hidden_states.dtype,
        sparse_gate_weight.dtype,
    )
    compiled(
        make_ptr(
            cutlass_dtype(normalized_hidden_states.dtype),
            int(normalized_hidden_states.data_ptr()),
            cute.AddressSpace.gmem,
            assumed_align=align_bytes(normalized_hidden_states.dtype),
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
        make_ptr(
            cutlass_dtype(normalized_hidden_states.dtype),
            int(router_logits_out.data_ptr()),
            cute.AddressSpace.gmem,
            assumed_align=align_bytes(normalized_hidden_states.dtype),
        ),
        make_ptr(cutlass.Int32, int(topk_ids_out.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, int(topk_weights_out.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, int(shared_gate_out.data_ptr()), cute.AddressSpace.gmem, assumed_align=4),
        current_cuda_stream(),
    )
    return UnifiedPreMLPSliceBOutputs(
        router_logits=router_logits_out,
        topk_ids=topk_ids_out,
        topk_weights=topk_weights_out,
        shared_gate=shared_gate_out,
    )


def slice_c_compact_route_assignment(
    workspace,
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> UnifiedPreMLPSliceCOutputs:
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("topk_ids and topk_weights must both have rank 2")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            "topk_ids/topk_weights shape mismatch: "
            f"{tuple(topk_ids.shape)} vs {tuple(topk_weights.shape)}"
        )
    if topk_ids.device != topk_weights.device:
        raise ValueError("topk_ids and topk_weights must be on the same device")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("slice C expects contiguous routing tensors")
    if workspace.row_counts.device != topk_ids.device:
        raise ValueError("workspace and routing tensors must be on the same device")
    total_pairs = int(topk_ids.numel())
    if total_pairs > int(workspace.token_map.shape[1]):
        raise ValueError(
            "workspace token_map capacity mismatch: "
            f"expected at least {total_pairs}, got {workspace.token_map.shape[1]}"
        )

    workspace.row_counts.zero_()
    workspace.token_map.zero_()
    workspace.token_weights.zero_()
    workspace.active_expert_count.zero_()
    workspace.weight_expert_ids.zero_()
    workspace.global_to_local_expert.fill_(-1)

    metadata = build_compact_route_metadata(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        weight_E=int(workspace.weight_E),
    )
    active_expert_count = metadata.active_expert_count
    if active_expert_count > int(workspace.state_E):
        raise ValueError(
            "workspace active-expert capacity mismatch: "
            f"expected at least {active_expert_count}, got {workspace.state_E}"
        )

    workspace.active_expert_count[0] = active_expert_count
    if active_expert_count > 0:
        workspace.weight_expert_ids[:active_expert_count].copy_(metadata.weight_expert_ids)
        workspace.global_to_local_expert.copy_(metadata.local_of_global)
        workspace.row_counts[:active_expert_count].copy_(metadata.counts)
        workspace.token_map[metadata.sorted_local, metadata.row_idx] = metadata.sorted_tokens
        workspace.token_weights[metadata.sorted_local, metadata.row_idx] = metadata.sorted_weights

    return UnifiedPreMLPSliceCOutputs(
        active_expert_count=workspace.active_expert_count,
        weight_expert_ids=workspace.weight_expert_ids,
        global_to_local_expert=workspace.global_to_local_expert,
        row_counts=workspace.row_counts,
        token_map=workspace.token_map,
        token_weights=workspace.token_weights,
    )


def slice_d_quantize_fc1_inputs(
    workspace,
    *,
    normalized_hidden_states: torch.Tensor,
    expert_input_scale: torch.Tensor,
    expert_alpha: torch.Tensor,
    fc1_tile_amax: bool = False,
) -> UnifiedPreMLPSliceDOutputs:
    if fc1_tile_amax:
        raise NotImplementedError("slice D currently supports fc1_tile_amax=False only")
    if normalized_hidden_states.ndim != 2:
        raise ValueError("normalized_hidden_states must be [num_tokens, hidden_size]")
    if expert_input_scale.ndim != 1 or expert_alpha.ndim != 1:
        raise ValueError("expert_input_scale and expert_alpha must both be rank-1")
    if expert_input_scale.numel() != int(workspace.weight_E):
        raise ValueError(
            "expert_input_scale expert mismatch: expected "
            f"{workspace.weight_E}, got {expert_input_scale.numel()}"
        )
    if expert_alpha.numel() != int(workspace.weight_E):
        raise ValueError(
            "expert_alpha expert mismatch: expected "
            f"{workspace.weight_E}, got {expert_alpha.numel()}"
        )
    if normalized_hidden_states.device != workspace.packed_input.device:
        raise ValueError("workspace and normalized_hidden_states must be on the same device")

    from b12x.cute.fp4 import quantize_grouped_nvfp4_torch
    from b12x.integration.tp_moe import _grouped_scale_view_to_swizzled_u8

    device = normalized_hidden_states.device
    hidden_size = int(normalized_hidden_states.shape[1])
    cols_pad_k = int(workspace.packed_input_scale.shape[-1])

    workspace.packed_input.zero_()
    workspace.packed_input_scale.zero_()
    workspace.fc1_tile_scale.zero_()
    workspace.fc1_tile_alpha.zero_()

    active_expert_count = int(workspace.active_expert_count.item())
    for local_idx in range(active_expert_count):
        row_count = int(workspace.row_counts[local_idx].item())
        if row_count == 0:
            continue
        expert_idx = int(workspace.weight_expert_ids[local_idx].item())
        token_idx = workspace.token_map[local_idx, :row_count].to(torch.long)
        rows_f32 = normalized_hidden_states.index_select(0, token_idx).float()
        num_tiles = (row_count + LEVEL_TILE_M - 1) // LEVEL_TILE_M
        rows_pad = num_tiles * LEVEL_TILE_M
        tile_rows = torch.zeros(
            (num_tiles, LEVEL_TILE_M, hidden_size),
            dtype=torch.float32,
            device=device,
        )
        tile_rows.view(rows_pad, hidden_size)[:row_count].copy_(rows_f32)
        row_counts = torch.full((num_tiles,), LEVEL_TILE_M, dtype=torch.int32, device=device)
        row_counts[-1] = row_count - (num_tiles - 1) * LEVEL_TILE_M

        expert_scale_value = float(expert_input_scale[expert_idx].item())
        effective_scale = expert_scale_value if expert_scale_value > 0.0 else 1.0
        tile_scale = torch.full((num_tiles,), effective_scale, dtype=torch.float32, device=device)
        packed_grouped, scale_view = quantize_grouped_nvfp4_torch(tile_rows, row_counts, tile_scale)
        packed_tiles = packed_grouped.permute(2, 0, 1).contiguous().view(rows_pad, hidden_size // 2)
        swizzled_tiles = _grouped_scale_view_to_swizzled_u8(
            scale_view,
            rows=LEVEL_TILE_M,
            cols=hidden_size,
        ).view(rows_pad, cols_pad_k)

        workspace.packed_input[local_idx, :row_count].copy_(packed_tiles[:row_count])
        workspace.packed_input_scale[local_idx, :rows_pad].copy_(swizzled_tiles)
        workspace.fc1_tile_scale[local_idx, :num_tiles].copy_(tile_scale)
        workspace.fc1_tile_alpha[local_idx, :num_tiles].fill_(float(expert_alpha[expert_idx].item()))

    return UnifiedPreMLPSliceDOutputs(
        packed_input=workspace.packed_input,
        packed_input_scale=workspace.packed_input_scale,
        fc1_tile_scale=workspace.fc1_tile_scale,
        fc1_tile_alpha=workspace.fc1_tile_alpha,
    )


__all__ = [
    "UnifiedPreMLPIPC",
    "UnifiedPreMLPStaticContract",
    "UnifiedPreMLPStaticLaunchConfig",
    "UnifiedPreMLPSliceAOutputs",
    "UnifiedPreMLPSliceBOutputs",
    "UnifiedPreMLPSliceCOutputs",
    "UnifiedPreMLPSliceDOutputs",
    "qwen35_static_contract",
    "slice_a_allreduce_residual_gemma_rmsnorm",
    "slice_b_sparse_routing_shared_gate",
    "slice_c_compact_route_assignment",
    "slice_d_quantize_fc1_inputs",
]
