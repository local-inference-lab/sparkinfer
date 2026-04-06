from __future__ import annotations

import pathlib
import sys

import cutlass
import cutlass.cute as cute
import torch

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
from b12x.distributed._oneshot_common import SIGNAL_BYTES  # noqa: E402
from b12x.integration.tp_moe import (  # noqa: E402
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
        hidden_states, residual, norm_weight, 1e-6,
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
    sig_ptrs = [
        make_ptr(cutlass.Int32, int(sig.data_ptr()), cute.AddressSpace.gmem, assumed_align=128)
        for sig in signal_storage
    ]
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

    kernel = MoESuperfusedStaticKernel(
        world_size=1,
        num_sparse_experts=sparse_experts.w1_fp4.shape[0],
        top_k=spec.top_k,
        sf_vec_size=16,
        mma_tiler_mn=(128, 128),
        output_tile_count_n=max(1, (spec.hidden_size + 128 - 1) // 128),
        input_scales_are_reciprocal=True,
        fast_math=False,
        fc2_tile_amax=False,
        emit_normalized=False,
        renormalize_topk=True,
        prequantized_input=False,
    )

    row_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m, spec.hidden_size), stride_order=(1, 0), assumed_align=16
    )
    norm_weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (spec.hidden_size,), assumed_align=16
    )
    sparse_gate_weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, tuple(gate_weight.shape), stride_order=(1, 0), assumed_align=16
    )
    shared_gate_weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, tuple(shared_gate_weight.shape), stride_order=(1, 0), assumed_align=16
    )
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (m * num_topk,), assumed_align=4
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m * num_topk,), assumed_align=4
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, tuple(packed_a_u8.shape), stride_order=(1, 0, 2), assumed_align=16
    )
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (ws.packed_a_flat.numel(),), assumed_align=16
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (ws.scale_flat.numel(),), assumed_align=16
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, tuple(weights_view.w13.shape), stride_order=(1, 0, 2), assumed_align=16
    )
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, tuple(weights_view.down.shape), stride_order=(1, 0, 2), assumed_align=16
    )
    row_counts_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, tuple(ws.row_counts.shape), assumed_align=4)
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, tuple(ws.weight_expert_ids.shape), assumed_align=4
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, tuple(ws.global_to_local_expert.shape), assumed_align=4
    )
    effective_fc1_input_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, tuple(effective_fc1_input_scale.shape), assumed_align=4
    )
    w1_alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, tuple(combined_experts.w1_alphas.shape), assumed_align=4
    )
    w2_alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, tuple(combined_experts.w2_alphas.shape), assumed_align=4
    )
    a2_gscale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, tuple(combined_experts.a2_gscale.shape), assumed_align=4
    )
    fc1_tile_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, tuple(ws.fc1_tile_scale.view(-1).shape), assumed_align=4
    )
    fc1_tile_alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, tuple(ws.fc1_tile_alpha.view(-1).shape), assumed_align=4
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, tuple(hidden_states.shape), stride_order=(1, 0), assumed_align=16
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, tuple(ws.token_map.shape), stride_order=(1, 0), assumed_align=4
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, tuple(ws.token_weights.shape), stride_order=(1, 0), assumed_align=4
    )
    signal_fake = make_ptr(cutlass.Int32, 128, cute.AddressSpace.gmem, assumed_align=128)
    compiled = cute.compile(
        kernel,
        row_fake, row_fake, row_fake, row_fake, row_fake, row_fake, row_fake, row_fake,
        signal_fake, signal_fake, signal_fake, signal_fake, signal_fake, signal_fake, signal_fake, signal_fake,
        signal_fake,
        cutlass.Int32(0),
        row_fake, row_fake, row_fake,
        norm_weight_fake,
        sparse_gate_weight_fake,
        shared_gate_weight_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        packed_a_storage_fake,
        scale_storage_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        b_w13_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        b_down_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        row_counts_fake,
        active_expert_count_fake,
        weight_expert_ids_fake,
        global_to_local_expert_fake,
        effective_fc1_input_scale_fake,
        w1_alpha_fake,
        w2_alpha_fake,
        a2_gscale_fake,
        fc1_tile_scale_fake,
        fc1_tile_alpha_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        1,
        cutlass.Float32(1e-6),
        current_cuda_stream(),
    )

    out = torch.empty_like(hidden_states)
    residual_out = torch.empty_like(hidden_states)
    normalized_out = torch.empty_like(hidden_states)
    rt_topk_ids = torch.empty((m * num_topk,), dtype=torch.int32, device=device)
    rt_topk_weights = torch.empty((m * num_topk,), dtype=torch.float32, device=device)
    compiled(
        hidden_states,
        hidden_states,
        hidden_states,
        hidden_states,
        hidden_states,
        hidden_states,
        hidden_states,
        hidden_states,
        sig_ptrs[0],
        sig_ptrs[1],
        sig_ptrs[2],
        sig_ptrs[3],
        sig_ptrs[4],
        sig_ptrs[5],
        sig_ptrs[6],
        sig_ptrs[7],
        sig_ptrs[0],
        cutlass.Int32(0),
        residual,
        normalized_out,
        residual_out,
        norm_weight,
        gate_weight,
        shared_gate_weight,
        rt_topk_ids,
        rt_topk_weights,
        packed_a_u8,
        ws.sfa_ptr,
        ws.packed_a_flat,
        ws.scale_flat,
        ws.barrier_count,
        ws.barrier_epoch,
        weights_view.w13,
        weights_view.sfb_w13_ptr,
        weights_view.down,
        weights_view.sfb_down_ptr,
        ws.row_counts,
        ws.active_expert_count,
        ws.weight_expert_ids,
        ws.global_to_local_expert,
        effective_fc1_input_scale,
        combined_experts.w1_alphas,
        combined_experts.w2_alphas,
        combined_experts.a2_gscale,
        ws.fc1_tile_scale.view(-1),
        ws.fc1_tile_alpha.view(-1),
        out,
        ws.token_map,
        ws.token_weights,
        1,
        cutlass.Float32(1e-6),
        current_cuda_stream(),
    )
    torch.cuda.synchronize()

    routing_ids = rt_topk_ids.view(m, num_topk)
    routing_weights = rt_topk_weights.view(m, num_topk)
    parity_consumer_out = _launch_prequantized_moe_consumer(
        normed_hidden_states,
        experts=combined_experts,
        workspace=ws,
        routing=ref_routing.__class__(topk_ids=routing_ids, topk_weights=routing_weights),
        input_scales_are_reciprocal=True,
        fast_math=False,
        fc1_tile_amax=False,
        fc2_tile_amax=False,
    )
    print(
        "superfused parity direct e2e:",
        f"out={compare_to_reference(out, ref_out)}",
        f"static_consumer_from_parity_ws={compare_to_reference(parity_consumer_out, ref_out)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
