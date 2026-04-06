from __future__ import annotations

import pathlib
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

from tests.test_gemma_moe_block_paths import (  # noqa: E402
    _gemma_rmsnorm_after_allreduce,
    _load_norm_weight,
    _make_spec,
    _pack_shared_expert,
    _pack_sparse_experts_per_expert,
)
from benchmarks.benchmark_moe import (  # noqa: E402
    MODEL_PATH,
    load_expert_weights,
    load_gate_weight,
    load_shared_expert_weights,
    load_shared_gate_weight,
    make_input_activations,
)
from b12x.cute.utils import current_cuda_stream, make_ptr  # noqa: E402
from b12x.distributed._oneshot_common import SIGNAL_BYTES, align_bytes, cutlass_dtype  # noqa: E402
from b12x.integration.tp_moe import (  # noqa: E402
    B12XTopKRouting,
    _append_expert_bank,
    _b12x_gemma_moe_block_fp4_static_producer,
    _effective_input_scales,
    _get_weight_views,
    _launch_prequantized_moe_consumer,
    _make_workspace_plan,
    _resolve_workspace,
    allocate_tp_moe_workspace_pool,
)
from b12x.moe.fused.monolithic_superfused_static_parity import (  # noqa: E402
    MoESuperfusedStaticKernel,
)
from b12x.moe.fused.reference import compare_to_reference  # noqa: E402


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype,
    *,
    assumed_align: int = 16,
):
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    if tensor.ndim >= 2:
        leading_dim = next((idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None)
        if leading_dim is not None:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


class _CompiledSuperfusedLaunch:
    def __init__(self, *, kernel: MoESuperfusedStaticKernel, num_tokens: int, hidden_size: int):
        self.kernel = kernel
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size

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
        residual_in: cute.Tensor,
        normalized_out: cute.Tensor,
        residual_out: cute.Tensor,
        norm_weight: cute.Tensor,
        sparse_gate_weight: cute.Tensor,
        shared_gate_weight: cute.Tensor,
        topk_ids_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        packed_a_u8: cute.Tensor,
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        b_w13_u8: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down_u8: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        active_expert_count: cute.Tensor,
        weight_expert_ids: cute.Tensor,
        global_to_local_expert: cute.Tensor,
        input_global_scale: cute.Tensor,
        expert_alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        fc1_tile_scale: cute.Tensor,
        fc1_tile_alpha: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        row_layout = cute.make_layout((self.num_tokens, self.hidden_size), stride=(self.hidden_size, 1))
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
            residual_in,
            normalized_out,
            residual_out,
            norm_weight,
            sparse_gate_weight,
            shared_gate_weight,
            topk_ids_flat,
            topk_weights_flat,
            packed_a_u8,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            b_w13_u8,
            sfb_w13_ptr,
            b_down_u8,
            sfb_down_ptr,
            row_counts,
            active_expert_count,
            weight_expert_ids,
            global_to_local_expert,
            input_global_scale,
            expert_alpha,
            down_alpha,
            global_scale,
            fc1_tile_scale,
            fc1_tile_alpha,
            scatter_output,
            token_map,
            token_weights,
            max_active_clusters,
            eps,
            stream,
        )


def main() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    spec = _make_spec()
    sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=0)
    gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=0)
    shared_gate_weight = load_shared_gate_weight(MODEL_PATH, layer_idx=0)
    sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
    shared_expert = _pack_shared_expert(shared_weights)
    combined_experts = _append_expert_bank(sparse_experts, shared_expert)
    norm_weight = _load_norm_weight()

    m = 4
    hidden_states = make_input_activations(spec, m, seed=5500, device=device)
    residual = make_input_activations(spec, m, seed=5600, device=device)
    normed_hidden_states, residual_expected = _gemma_rmsnorm_after_allreduce(
        hidden_states,
        residual,
        norm_weight,
        1e-6,
    )

    class _FakePreMLPRuntime:
        def allreduce_gemma_rmsnorm(
            self,
            inp: torch.Tensor,
            residual_in: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            *,
            peer_input_ptrs=None,
            out=None,
            residual_out=None,
        ):
            del inp, residual_in, weight, eps, peer_input_ptrs
            if out is None:
                out = torch.empty_like(normed_hidden_states)
            out.copy_(normed_hidden_states)
            if residual_out is None:
                residual_out = torch.empty_like(residual_expected)
            residual_out.copy_(residual_expected)
            return out, residual_out

    ref_pool = allocate_tp_moe_workspace_pool()
    ref_out, ref_residual_out, ref_routing = _b12x_gemma_moe_block_fp4_static_producer(
        hidden_states,
        residual,
        pre_mlp_runtime=_FakePreMLPRuntime(),
        norm_weight=norm_weight,
        norm_eps=1e-6,
        sparse_experts=sparse_experts,
        shared_expert=shared_expert,
        shared_gate_weight=shared_gate_weight,
        combined_experts=combined_experts,
        workspace=ref_pool,
        top_k=spec.top_k,
        gate_weight=gate_weight,
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        fc1_tile_amax=False,
        return_routing=True,
    )

    weight_E = combined_experts.w1_fp4.shape[0]
    n = combined_experts.w2_fp4.shape[2] * 2
    num_topk = spec.top_k + 1
    pool = allocate_tp_moe_workspace_pool()
    plan = _make_workspace_plan(
        num_tokens=m,
        weight_E=weight_E,
        k=spec.hidden_size,
        n=n,
        num_topk=num_topk,
        device=device,
        dtype=hidden_states.dtype,
    )
    ws = _resolve_workspace(
        pool,
        plan=plan,
        a1_gscale=combined_experts.a1_gscale,
        a2_gscale=combined_experts.a2_gscale,
        input_scales_static=True,
    )

    signal_storage = [torch.zeros(SIGNAL_BYTES // 4, dtype=torch.int32, device=device) for _ in range(8)]
    input_align = align_bytes(torch.bfloat16)
    inp_ptr = make_ptr(
        cutlass_dtype(torch.bfloat16),
        int(hidden_states.data_ptr()),
        cute.AddressSpace.gmem,
        assumed_align=input_align,
    )
    sig_ptrs = [
        make_ptr(cutlass.Int32, int(sig.data_ptr()), cute.AddressSpace.gmem, assumed_align=128)
        for sig in signal_storage
    ]

    kernel = MoESuperfusedStaticKernel(
        world_size=1,
        num_sparse_experts=sparse_experts.w1_fp4.shape[0],
        top_k=spec.top_k,
        sf_vec_size=16,
        mma_tiler_mn=(128, 128),
        output_tile_count_n=1,
        input_scales_are_reciprocal=True,
        fast_math=False,
        fc2_tile_amax=False,
        emit_normalized=False,
        renormalize_topk=True,
        prequantized_input=False,
    )
    launch = _CompiledSuperfusedLaunch(
        kernel=kernel,
        num_tokens=m,
        hidden_size=spec.hidden_size,
    )
    prequantized_kernel = MoESuperfusedStaticKernel(
        world_size=1,
        num_sparse_experts=sparse_experts.w1_fp4.shape[0],
        top_k=spec.top_k,
        sf_vec_size=16,
        mma_tiler_mn=(128, 128),
        output_tile_count_n=1,
        input_scales_are_reciprocal=True,
        fast_math=False,
        fc2_tile_amax=False,
        emit_normalized=False,
        renormalize_topk=True,
        prequantized_input=True,
    )
    prequantized_launch = _CompiledSuperfusedLaunch(
        kernel=prequantized_kernel,
        num_tokens=m,
        hidden_size=spec.hidden_size,
    )

    out = torch.empty_like(hidden_states)
    prequantized_out = torch.empty_like(hidden_states)
    residual_out = torch.empty_like(hidden_states)
    normalized_out = torch.empty_like(hidden_states)
    rt_topk_ids = torch.empty((m * (spec.top_k + 1),), dtype=torch.int32, device=device)
    rt_topk_weights = torch.empty((m * (spec.top_k + 1),), dtype=torch.float32, device=device)
    weights_view = _get_weight_views(
        combined_experts.w1_fp4,
        combined_experts.w1_blockscale,
        combined_experts.w2_fp4,
        combined_experts.w2_blockscale,
        combined_experts.w1_alphas,
        combined_experts.w2_alphas,
        n,
        spec.hidden_size,
    )
    effective_fc1_input_scale = _effective_input_scales(
        combined_experts.a1_gscale,
        weight_E,
        input_scales_are_reciprocal=True,
    )
    packed_a_u8 = ws.packed_input.permute(1, 2, 0)

    launch(
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        sig_ptrs[0],
        sig_ptrs[1],
        sig_ptrs[2],
        sig_ptrs[3],
        sig_ptrs[4],
        sig_ptrs[5],
        sig_ptrs[6],
        sig_ptrs[7],
        sig_ptrs[0],
        Int32(0),
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(normalized_out, cutlass.BFloat16),
        _to_kernel_tensor(residual_out, cutlass.BFloat16),
        _to_kernel_tensor(norm_weight, cutlass.BFloat16),
        _to_kernel_tensor(gate_weight, cutlass.BFloat16),
        _to_kernel_tensor(shared_gate_weight, cutlass.BFloat16),
        _to_kernel_tensor(rt_topk_ids, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(rt_topk_weights, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(packed_a_u8, cutlass.Uint8, assumed_align=16),
        ws.sfa_ptr,
        _to_kernel_tensor(ws.packed_a_flat, cutlass.Uint8, assumed_align=1),
        _to_kernel_tensor(ws.scale_flat, cutlass.Uint8, assumed_align=1),
        _to_kernel_tensor(ws.barrier_count, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.barrier_epoch, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(weights_view.w13, cutlass.Uint8, assumed_align=16),
        weights_view.sfb_w13_ptr,
        _to_kernel_tensor(weights_view.down, cutlass.Uint8, assumed_align=16),
        weights_view.sfb_down_ptr,
        _to_kernel_tensor(ws.row_counts, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.active_expert_count, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.weight_expert_ids, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.global_to_local_expert, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(effective_fc1_input_scale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(combined_experts.w1_alphas, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(combined_experts.w2_alphas, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(combined_experts.a2_gscale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(ws.fc1_tile_scale.view(-1), cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(ws.fc1_tile_alpha.view(-1), cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(out, cutlass.BFloat16),
        _to_kernel_tensor(ws.token_map, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.token_weights, cutlass.Float32, assumed_align=4),
        1,
        cutlass.Float32(1e-6),
        current_cuda_stream(),
    )
    prequantized_launch(
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        inp_ptr,
        sig_ptrs[0],
        sig_ptrs[1],
        sig_ptrs[2],
        sig_ptrs[3],
        sig_ptrs[4],
        sig_ptrs[5],
        sig_ptrs[6],
        sig_ptrs[7],
        sig_ptrs[0],
        Int32(0),
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(normalized_out, cutlass.BFloat16),
        _to_kernel_tensor(residual_out, cutlass.BFloat16),
        _to_kernel_tensor(norm_weight, cutlass.BFloat16),
        _to_kernel_tensor(gate_weight, cutlass.BFloat16),
        _to_kernel_tensor(shared_gate_weight, cutlass.BFloat16),
        _to_kernel_tensor(rt_topk_ids, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(rt_topk_weights, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(packed_a_u8, cutlass.Uint8, assumed_align=16),
        ws.sfa_ptr,
        _to_kernel_tensor(ws.packed_a_flat, cutlass.Uint8, assumed_align=1),
        _to_kernel_tensor(ws.scale_flat, cutlass.Uint8, assumed_align=1),
        _to_kernel_tensor(ws.barrier_count, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.barrier_epoch, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(weights_view.w13, cutlass.Uint8, assumed_align=16),
        weights_view.sfb_w13_ptr,
        _to_kernel_tensor(weights_view.down, cutlass.Uint8, assumed_align=16),
        weights_view.sfb_down_ptr,
        _to_kernel_tensor(ws.row_counts, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.active_expert_count, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.weight_expert_ids, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.global_to_local_expert, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(effective_fc1_input_scale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(combined_experts.w1_alphas, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(combined_experts.w2_alphas, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(combined_experts.a2_gscale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(ws.fc1_tile_scale.view(-1), cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(ws.fc1_tile_alpha.view(-1), cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(prequantized_out, cutlass.BFloat16),
        _to_kernel_tensor(ws.token_map, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.token_weights, cutlass.Float32, assumed_align=4),
        1,
        cutlass.Float32(1e-6),
        current_cuda_stream(),
    )
    torch.cuda.synchronize()

    routing_ids = rt_topk_ids.view(m, spec.top_k + 1)
    routing_weights = rt_topk_weights.view(m, spec.top_k + 1)
    parity_consumer_out = _launch_prequantized_moe_consumer(
        normed_hidden_states,
        experts=combined_experts,
        workspace=ws,
        routing=B12XTopKRouting(topk_ids=routing_ids, topk_weights=routing_weights),
        input_scales_are_reciprocal=True,
        fc1_tile_amax=False,
        fc2_tile_amax=False,
    )
    out_metrics = compare_to_reference(out, ref_out)
    prequantized_kernel_metrics = compare_to_reference(prequantized_out, ref_out)
    parity_consumer_metrics = compare_to_reference(parity_consumer_out, ref_out)
    residual_metrics = compare_to_reference(residual_out, ref_residual_out)
    routing_metrics = compare_to_reference(routing_weights, ref_routing.topk_weights)
    print(
        "superfused parity e2e:",
        f"out(max_abs={out_metrics.max_abs:.3e}, cos={out_metrics.cos:.6f})",
        f"self_prequantized(max_abs={prequantized_kernel_metrics.max_abs:.3e}, cos={prequantized_kernel_metrics.cos:.6f})",
        f"static_consumer_from_parity_ws(max_abs={parity_consumer_metrics.max_abs:.3e}, cos={parity_consumer_metrics.cos:.6f})",
        f"residual(max_abs={residual_metrics.max_abs:.3e}, cos={residual_metrics.cos:.6f})",
        f"routing(max_abs={routing_metrics.max_abs:.3e}, cos={routing_metrics.cos:.6f}, ids_equal={torch.equal(routing_ids, ref_routing.topk_ids)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
