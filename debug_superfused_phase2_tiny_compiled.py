from __future__ import annotations

from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32
from cutlass.cute.runtime import from_dlpack

from benchmarks.benchmark_moe import ModelSpec, load_expert_weights, make_input_activations
from benchmarks.benchmark_gemma_moe_block_paths import _pack_sparse_experts_per_expert
from b12x.cute.utils import current_cuda_stream, make_ptr
from b12x.distributed._oneshot_common import align_bytes, cutlass_dtype
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    _get_weight_views,
    _make_workspace_plan,
    _populate_static_prequantized_workspace_host,
    _resolve_workspace,
    allocate_tp_moe_workspace_pool,
)
from b12x.moe.fused.monolithic_superfused_static import MoESuperfusedStaticKernel
from b12x.moe.fused.reference import (
    _apply_block_scales,
    _dequant_fp4,
    _make_fp4_lut,
    compare_to_reference,
    unswizzle_block_scale,
)

MODEL_PATH = Path("/data/models/Qwen3.5-397B-A17B-NVFP4")
DEVICE = torch.device("cuda", 0)


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
        packed_a: cute.Tensor,
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
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
            packed_a,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            b_w13,
            sfb_w13_ptr,
            b_down,
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


def _compile_launch(num_tokens: int, hidden_size: int):
    kernel = MoESuperfusedStaticKernel(
        world_size=1,
        num_sparse_experts=1,
        top_k=1,
        sf_vec_size=16,
        mma_tiler_mn=(128, 128),
        output_tile_count_n=1,
        input_scales_are_reciprocal=True,
        fast_math=False,
        fc2_tile_amax=False,
        emit_normalized=False,
        renormalize_topk=True,
        skip_phase1=True,
    )
    launch = _CompiledSuperfusedLaunch(kernel=kernel, num_tokens=num_tokens, hidden_size=hidden_size)
    input_align = align_bytes(torch.bfloat16)
    row_fake = make_ptr(cutlass_dtype(torch.bfloat16), max(16, input_align), cute.AddressSpace.gmem, assumed_align=input_align)
    signal_fake = make_ptr(cutlass.Int32, 128, cute.AddressSpace.gmem, assumed_align=128)
    f32_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    i32_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    bf16_row_fake = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (num_tokens, hidden_size), stride_order=(1, 0), assumed_align=16)
    sparse_gate_fake = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (1, hidden_size), stride_order=(1, 0), assumed_align=16)
    shared_gate_fake = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (hidden_size,), assumed_align=16)
    flat_i32_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (16,), assumed_align=4)
    flat_f32_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (16,), assumed_align=4)
    packed_a_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float4E2M1FN, (128, hidden_size, 1), stride_order=(1, 0, 2), assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (128 * (hidden_size // 2),), assumed_align=16)
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (128 * (((hidden_size // 16 + 3) // 4) * 4),), assumed_align=16)
    b_w13_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float4E2M1FN, (256, hidden_size, 1), stride_order=(1, 0, 2), assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float4E2M1FN, (hidden_size, 128, 1), stride_order=(1, 0, 2), assumed_align=16)
    sfb_fake = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1, 128), stride_order=(1, 0), assumed_align=4)
    token_weights_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1, 128), stride_order=(1, 0), assumed_align=16)
    return cute.compile(
        launch,
        row_fake, row_fake, row_fake, row_fake, row_fake, row_fake, row_fake, row_fake,
        signal_fake, signal_fake, signal_fake, signal_fake, signal_fake, signal_fake, signal_fake, signal_fake,
        signal_fake,
        Int32(0),
        bf16_row_fake, bf16_row_fake, bf16_row_fake,
        cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (hidden_size,), assumed_align=16),
        sparse_gate_fake, shared_gate_fake,
        flat_i32_fake, flat_f32_fake,
        packed_a_fake, sfb_fake, packed_a_storage_fake, scale_storage_fake,
        i32_fake, i32_fake,
        b_w13_fake, sfb_fake, b_down_fake, sfb_fake,
        i32_fake, i32_fake, i32_fake, i32_fake,
        f32_fake, f32_fake, f32_fake, f32_fake,
        f32_fake, f32_fake,
        bf16_row_fake,
        token_map_fake, token_weights_fake,
        1,
        cutlass.Float32(1e-6),
        current_cuda_stream(),
    )


def main():
    torch.cuda.set_device(DEVICE)
    torch.set_grad_enabled(False)

    spec = ModelSpec(hidden_size=4096, intermediate_size=1024, num_experts=512, top_k=1, tp_size=8, tp_rank=0)
    sparse = _pack_sparse_experts_per_expert(load_expert_weights(MODEL_PATH, spec, layer_idx=0))
    chosen_e = 48
    one = B12XFP4ExpertWeights(
        a1_gscale=sparse.a1_gscale[chosen_e:chosen_e + 1].contiguous(),
        w1_fp4=sparse.w1_fp4[chosen_e:chosen_e + 1].contiguous(),
        w1_blockscale=sparse.w1_blockscale[chosen_e:chosen_e + 1].contiguous(),
        w1_alphas=sparse.w1_alphas[chosen_e:chosen_e + 1].contiguous(),
        a2_gscale=sparse.a2_gscale[chosen_e:chosen_e + 1].contiguous(),
        w2_fp4=sparse.w2_fp4[chosen_e:chosen_e + 1].contiguous(),
        w2_blockscale=sparse.w2_blockscale[chosen_e:chosen_e + 1].contiguous(),
        w2_alphas=sparse.w2_alphas[chosen_e:chosen_e + 1].contiguous(),
    )

    m = 1
    k = spec.hidden_size
    n = one.w2_fp4.shape[2] * 2
    a = make_input_activations(spec, m, seed=777, device=DEVICE).float().mul_(1024.0).to(torch.bfloat16)
    topk_ids = torch.zeros((m, 1), dtype=torch.int32, device=DEVICE)
    topk_weights = torch.ones((m, 1), dtype=torch.float32, device=DEVICE)

    pool = allocate_tp_moe_workspace_pool()
    plan = _make_workspace_plan(num_tokens=m, weight_E=1, k=k, n=n, num_topk=1, device=DEVICE, dtype=a.dtype)
    ws = _resolve_workspace(pool, plan=plan, a1_gscale=one.a1_gscale, a2_gscale=one.a2_gscale, input_scales_static=True)
    _populate_static_prequantized_workspace_host(
        ws,
        a=a,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        expert_input_scale=one.a1_gscale,
        expert_alpha=one.w1_alphas,
        fc1_tile_amax=False,
    )

    fp4_lut = _make_fp4_lut(DEVICE)
    x_raw = _dequant_fp4(ws.packed_input[0, 0].view(torch.uint8), 1, k, fp4_lut)
    x_sf = unswizzle_block_scale(ws.packed_input_scale[0], 1, k // 16)
    x_dequant = _apply_block_scales(x_raw, x_sf, 1, k, 16)[0]
    i_tp = n
    w13_sf = unswizzle_block_scale(one.w1_blockscale[0], 2 * i_tp, k // 16)
    up_dequant = _apply_block_scales(_dequant_fp4(one.w1_fp4[0, :i_tp].view(torch.uint8), i_tp, k, fp4_lut), w13_sf[:i_tp], i_tp, k, 16)
    gate_dequant = _apply_block_scales(_dequant_fp4(one.w1_fp4[0, i_tp:].view(torch.uint8), i_tp, k, fp4_lut), w13_sf[i_tp:], i_tp, k, 16)
    w2_sf = unswizzle_block_scale(one.w2_blockscale[0], k, i_tp // 16)
    down_dequant = _apply_block_scales(_dequant_fp4(one.w2_fp4[0].view(torch.uint8), k, i_tp, fp4_lut), w2_sf, k, i_tp, 16)
    alpha1 = float(ws.fc1_tile_alpha[0, 0].item())
    gate = (gate_dequant @ x_dequant) * alpha1
    up = (up_dequant @ x_dequant) * alpha1
    silu = gate * torch.sigmoid(gate) * up
    ref_out = ((down_dequant @ silu) * float(one.w2_alphas[0].item()) * float(one.a2_gscale[0].item())).unsqueeze(0).to(torch.bfloat16)

    dummy = torch.zeros((m, k), dtype=torch.bfloat16, device=DEVICE)
    signals = [torch.zeros((1,), dtype=torch.int32, device=DEVICE) for _ in range(8)]
    output = torch.zeros_like(a)
    residual_out = torch.zeros_like(a)
    normalized_out = torch.empty_like(a)
    dbg_ids = torch.zeros((16,), dtype=torch.int32, device=DEVICE)
    dbg_weights = torch.zeros((16,), dtype=torch.float32, device=DEVICE)
    compiled = _compile_launch(m, k)
    weights_view = _get_weight_views(
        one.w1_fp4, one.w1_blockscale, one.w2_fp4, one.w2_blockscale, one.w1_alphas, one.w2_alphas, n, k
    )
    norm_weight = torch.ones((k,), dtype=torch.bfloat16, device=DEVICE)
    sparse_gate = torch.zeros((1, k), dtype=torch.bfloat16, device=DEVICE)
    shared_gate = torch.zeros((k,), dtype=torch.bfloat16, device=DEVICE)
    dummy_cute = _to_kernel_tensor(dummy, cutlass.BFloat16)
    normalized_cute = _to_kernel_tensor(normalized_out, cutlass.BFloat16)
    residual_out_cute = _to_kernel_tensor(residual_out, cutlass.BFloat16)
    norm_weight_cute = _to_kernel_tensor(norm_weight, cutlass.BFloat16)
    sparse_gate_cute = _to_kernel_tensor(sparse_gate, cutlass.BFloat16)
    shared_gate_cute = _to_kernel_tensor(shared_gate, cutlass.BFloat16)
    dbg_ids_cute = _to_kernel_tensor(dbg_ids, cutlass.Int32, assumed_align=4)
    dbg_weights_cute = _to_kernel_tensor(dbg_weights, cutlass.Float32, assumed_align=4)
    packed_a_cute = _to_kernel_tensor(ws.packed_a_view, cutlass.Float4E2M1FN, assumed_align=16)
    packed_a_storage_cute = _to_kernel_tensor(ws.packed_a_flat, cutlass.Uint8, assumed_align=1)
    scale_storage_cute = _to_kernel_tensor(ws.scale_flat, cutlass.Uint8, assumed_align=1)
    barrier_count_cute = _to_kernel_tensor(ws.barrier_count, cutlass.Int32, assumed_align=4)
    barrier_epoch_cute = _to_kernel_tensor(ws.barrier_epoch, cutlass.Int32, assumed_align=4)
    w13_cute = _to_kernel_tensor(weights_view.w13_fp4, cutlass.Float4E2M1FN, assumed_align=16)
    down_cute = _to_kernel_tensor(weights_view.down_fp4, cutlass.Float4E2M1FN, assumed_align=16)
    row_counts_cute = _to_kernel_tensor(ws.row_counts, cutlass.Int32, assumed_align=4)
    active_expert_count_cute = _to_kernel_tensor(ws.active_expert_count, cutlass.Int32, assumed_align=4)
    weight_expert_ids_cute = _to_kernel_tensor(ws.weight_expert_ids, cutlass.Int32, assumed_align=4)
    global_to_local_expert_cute = _to_kernel_tensor(ws.global_to_local_expert, cutlass.Int32, assumed_align=4)
    a1_gscale_cute = _to_kernel_tensor(one.a1_gscale, cutlass.Float32, assumed_align=4)
    w1_alphas_cute = _to_kernel_tensor(one.w1_alphas, cutlass.Float32, assumed_align=4)
    w2_alphas_cute = _to_kernel_tensor(one.w2_alphas, cutlass.Float32, assumed_align=4)
    a2_gscale_cute = _to_kernel_tensor(one.a2_gscale, cutlass.Float32, assumed_align=4)
    fc1_tile_scale_cute = _to_kernel_tensor(ws.fc1_tile_scale.view(-1), cutlass.Float32, assumed_align=4)
    fc1_tile_alpha_cute = _to_kernel_tensor(ws.fc1_tile_alpha.view(-1), cutlass.Float32, assumed_align=4)
    output_cute = _to_kernel_tensor(output, cutlass.BFloat16)
    token_map_cute = _to_kernel_tensor(ws.token_map, cutlass.Int32, assumed_align=4)
    token_weights_cute = _to_kernel_tensor(ws.token_weights, cutlass.Float32, assumed_align=4)
    input_ptrs = [make_ptr(cutlass_dtype(dummy.dtype), int(dummy.data_ptr()), cute.AddressSpace.gmem, assumed_align=align_bytes(dummy.dtype)) for _ in range(8)]
    signal_ptrs = [make_ptr(cutlass.Int32, int(t.data_ptr()), cute.AddressSpace.gmem, assumed_align=128) for t in signals]

    compiled(
        *input_ptrs,
        *signal_ptrs,
        signal_ptrs[0],
        0,
        dummy_cute,
        normalized_cute,
        residual_out_cute,
        norm_weight_cute,
        sparse_gate_cute,
        shared_gate_cute,
        dbg_ids_cute,
        dbg_weights_cute,
        packed_a_cute,
        ws.sfa_ptr,
        packed_a_storage_cute,
        scale_storage_cute,
        barrier_count_cute,
        barrier_epoch_cute,
        w13_cute,
        weights_view.sfb_w13_ptr,
        down_cute,
        weights_view.sfb_down_ptr,
        row_counts_cute,
        active_expert_count_cute,
        weight_expert_ids_cute,
        global_to_local_expert_cute,
        a1_gscale_cute,
        w1_alphas_cute,
        w2_alphas_cute,
        a2_gscale_cute,
        fc1_tile_scale_cute,
        fc1_tile_alpha_cute,
        output_cute,
        token_map_cute,
        token_weights_cute,
        1,
        1e-6,
        current_cuda_stream(),
    )
    torch.cuda.synchronize()

    metrics = compare_to_reference(output, ref_out)
    print(
        "tiny compiled validate:",
        f"out(max_abs={metrics.max_abs:.3e}, cos={metrics.cos:.6f})",
        f"out_norm={float(output.float().norm().item()):.6e}",
        f"ref_norm={float(ref_out.float().norm().item()):.6e}",
        f"valid_rows={int(dbg_ids[1].item())}",
        f"weight_e={int(dbg_ids[2].item())}",
        f"inter_tiles={int(dbg_ids[3].item())}",
        f"gate_tiles={int(dbg_ids[4].item())}",
        f"gate_sfb_max_u8={int(dbg_ids[5].item())}",
        f"up_sfb_max_u8={int(dbg_ids[6].item())}",
        f"gate_b_part={int(dbg_ids[7].item())}",
        f"up_b_part={int(dbg_ids[8].item())}",
        f"gate_sfb_src={int(dbg_ids[9].item())}",
        f"up_sfb_src={int(dbg_ids[10].item())}",
        f"gate_b_dst={int(dbg_ids[11].item())}",
        f"up_b_dst={int(dbg_ids[12].item())}",
        f"gate_frag_max={float(dbg_weights[1].item()):.6e}",
        f"up_frag_max={float(dbg_weights[2].item()):.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
