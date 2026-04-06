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
from b12x.cute.utils import current_cuda_stream
from b12x.distributed._oneshot_common import cutlass_dtype
from b12x.integration.tp_moe import (
    B12XFP4ExpertWeights,
    _get_weight_views,
    _make_workspace_plan,
    _populate_static_prequantized_workspace_host,
    _resolve_workspace,
    allocate_tp_moe_workspace_pool,
)
from b12x.moe.fused.reference import (
    _apply_block_scales,
    _dequant_fp4,
    _make_fp4_lut,
    compare_to_reference,
    unswizzle_block_scale,
)
from b12x.moe.fused.static import MoEStaticKernel

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


class _CompiledStaticLaunch:
    def __init__(self):
        self.kernel = MoEStaticKernel(
            sf_vec_size=16,
            mma_tiler_mn=(128, 128),
            output_tile_count_n=1,
            input_scales_are_reciprocal=True,
            fast_math=False,
            fc2_tile_amax=False,
            prequantized_input=True,
        )

    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
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
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        fc1_tile_scale: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            a_input,
            topk_ids,
            topk_weights,
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
            alpha,
            down_alpha,
            global_scale,
            fc1_tile_scale,
            scatter_output,
            token_map,
            token_weights,
            1,
            stream,
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

    output = torch.zeros_like(a)
    compiled = _CompiledStaticLaunch()
    weights_view = _get_weight_views(
        one.w1_fp4, one.w1_blockscale, one.w2_fp4, one.w2_blockscale, one.w1_alphas, one.w2_alphas, n, k
    )
    compiled(
        _to_kernel_tensor(a, cutlass.BFloat16),
        _to_kernel_tensor(topk_ids.view(-1), cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(topk_weights.view(-1), cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(ws.packed_a_view, cutlass.Float4E2M1FN, assumed_align=16),
        ws.sfa_ptr,
        _to_kernel_tensor(ws.packed_a_flat, cutlass.Uint8, assumed_align=1),
        _to_kernel_tensor(ws.scale_flat, cutlass.Uint8, assumed_align=1),
        _to_kernel_tensor(ws.barrier_count, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.barrier_epoch, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(weights_view.w13_fp4, cutlass.Float4E2M1FN, assumed_align=16),
        weights_view.sfb_w13_ptr,
        _to_kernel_tensor(weights_view.down_fp4, cutlass.Float4E2M1FN, assumed_align=16),
        weights_view.sfb_down_ptr,
        _to_kernel_tensor(ws.row_counts, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.active_expert_count, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.weight_expert_ids, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.global_to_local_expert, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(one.a1_gscale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(one.w1_alphas, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(one.w2_alphas, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(one.a2_gscale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(ws.fc1_tile_scale.view(-1), cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(output, cutlass.BFloat16),
        _to_kernel_tensor(ws.token_map, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(ws.token_weights, cutlass.Float32, assumed_align=4),
        current_cuda_stream(),
    )
    torch.cuda.synchronize()

    metrics = compare_to_reference(output, ref_out)
    print(
        "static tiny validate:",
        f"out(max_abs={metrics.max_abs:.3e}, cos={metrics.cos:.6f})",
        f"out_norm={float(output.float().norm().item()):.6e}",
        f"ref_norm={float(ref_out.float().norm().item()):.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
