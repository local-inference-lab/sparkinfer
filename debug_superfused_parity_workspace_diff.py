from __future__ import annotations

import pathlib
import sys

import cutlass
import cutlass.cute as cute
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

from debug_superfused_parity_e2e import _CompiledSuperfusedLaunch, _to_kernel_tensor  # noqa: E402
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
    _append_expert_bank,
    _effective_input_scales,
    _get_weight_views,
    _launch_prequantized_moe_consumer,
    _make_workspace_plan,
    _prepare_gemma_moe_block_fp4_static_producer,
    _resolve_workspace,
    allocate_tp_moe_workspace_pool,
)
from b12x.moe.fused.monolithic_superfused_static_parity import (  # noqa: E402
    MoESuperfusedStaticKernel,
)


def _first_diff(a: torch.Tensor, b: torch.Tensor):
    if a.shape != b.shape:
        return ("shape", tuple(a.shape), tuple(b.shape))
    diff = a != b if a.dtype != torch.float32 and a.dtype != torch.float16 and a.dtype != torch.bfloat16 else (a != b)
    nz = diff.nonzero()
    if nz.numel() == 0:
        return None
    idx = tuple(int(x) for x in nz[0].tolist())
    return idx, a[idx].item(), b[idx].item()


def _float_stats(a: torch.Tensor, b: torch.Tensor):
    d = (a.float() - b.float()).abs()
    nz = (d > 0).nonzero()
    first = None
    if nz.numel():
        idx = tuple(int(x) for x in nz[0].tolist())
        first = (idx, a[idx].item(), b[idx].item(), d[idx].item())
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "first": first,
    }


def _build_semantic_rows(ws):
    active = int(ws.active_expert_count.item())
    rows = {}
    for local_e in range(active):
        expert_id = int(ws.weight_expert_ids[local_e].item())
        row_count = int(ws.row_counts[local_e].item())
        for row in range(row_count):
            token_idx = int(ws.token_map[local_e, row].item())
            rows[(expert_id, token_idx)] = {
                "weight": float(ws.token_weights[local_e, row].item()),
                "packed": ws.packed_input[local_e, row].clone(),
                "scale": ws.packed_input_scale[local_e, row].clone(),
                "tile_scale": float(ws.fc1_tile_scale[local_e, row // 128].item()),
                "tile_alpha": float(ws.fc1_tile_alpha[local_e, row // 128].item()),
            }
    return rows


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
    expected_normed, expected_residual = _gemma_rmsnorm_after_allreduce(
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
            del peer_input_ptrs
            x = inp + residual_in
            x_fp32 = x.float()
            inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
            normed = (x_fp32 * inv_rms * (1.0 + weight.float())).to(dtype=inp.dtype)
            if out is None:
                out = torch.empty_like(normed)
            out.copy_(normed)
            if residual_out is None:
                residual_out = torch.empty_like(x)
            residual_out.copy_(x)
            return out, residual_out

    ref_pool = allocate_tp_moe_workspace_pool()
    _, _, ref_ws, _, ref_routing = _prepare_gemma_moe_block_fp4_static_producer(
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
        emit_normalized=True,
        renormalize_topk=True,
        prequantized_input=False,
    )
    launch = _CompiledSuperfusedLaunch(kernel=kernel, num_tokens=m, hidden_size=spec.hidden_size)
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

    out = torch.empty_like(hidden_states)
    residual_out = torch.empty_like(hidden_states)
    normalized_out = torch.empty_like(hidden_states)
    rt_topk_ids = torch.empty((m * (spec.top_k + 1),), dtype=torch.int32, device=device)
    rt_topk_weights = torch.empty((m * (spec.top_k + 1),), dtype=torch.float32, device=device)

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
        cutlass.Int32(0),
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(normalized_out, cutlass.BFloat16),
        _to_kernel_tensor(residual_out, cutlass.BFloat16),
        _to_kernel_tensor(norm_weight, cutlass.BFloat16),
        _to_kernel_tensor(gate_weight, cutlass.BFloat16),
        _to_kernel_tensor(shared_gate_weight, cutlass.BFloat16),
        _to_kernel_tensor(rt_topk_ids, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(rt_topk_weights, cutlass.Float32, assumed_align=4),
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
    torch.cuda.synchronize()

    active = int(ref_ws.active_expert_count.item())
    max_rows = int(ref_ws.row_counts[:active].max().item()) if active else 0
    print("active_expert_count", int(ws.active_expert_count.item()), int(ref_ws.active_expert_count.item()))
    print("weight_expert_ids", _first_diff(ws.weight_expert_ids[:active], ref_ws.weight_expert_ids[:active]))
    print("row_counts", _first_diff(ws.row_counts[:active], ref_ws.row_counts[:active]))
    print("global_to_local_expert", _first_diff(ws.global_to_local_expert, ref_ws.global_to_local_expert))
    print("token_map", _first_diff(ws.token_map[:active, :max_rows], ref_ws.token_map[:active, :max_rows]))
    print("token_weights", _float_stats(ws.token_weights[:active, :max_rows], ref_ws.token_weights[:active, :max_rows]))
    print("fc1_tile_scale", _float_stats(ws.fc1_tile_scale[:active], ref_ws.fc1_tile_scale[:active]))
    print("fc1_tile_alpha", _float_stats(ws.fc1_tile_alpha[:active], ref_ws.fc1_tile_alpha[:active]))
    print("packed_input", _first_diff(ws.packed_input[:active, :max_rows], ref_ws.packed_input[:active, :max_rows]))
    rows_pad = ref_ws.packed_input_scale.shape[1]
    print(
        "packed_input_scale",
        _first_diff(
            ws.packed_input_scale[:active, :rows_pad],
            ref_ws.packed_input_scale[:active, :rows_pad],
        ),
    )
    print(
        "routing ids",
        _first_diff(rt_topk_ids.view(m, -1), ref_routing.topk_ids),
    )
    print(
        "routing weights",
        _float_stats(rt_topk_weights.view(m, -1), ref_routing.topk_weights),
    )
    print("normalized_out", _float_stats(normalized_out, expected_normed))
    print("residual_out", _float_stats(residual_out, expected_residual))

    fused_rows = _build_semantic_rows(ws)
    ref_rows = _build_semantic_rows(ref_ws)
    fused_keys = set(fused_rows)
    ref_keys = set(ref_rows)
    print("semantic key sets", len(fused_keys), len(ref_keys), "missing", len(ref_keys - fused_keys), "extra", len(fused_keys - ref_keys))
    if fused_keys == ref_keys:
        packed_mismatch = None
        scale_mismatch = None
        weight_mismatch = None
        tile_scale_mismatch = None
        tile_alpha_mismatch = None
        for key in sorted(fused_keys):
            f = fused_rows[key]
            r = ref_rows[key]
            if weight_mismatch is None and abs(f["weight"] - r["weight"]) > 0:
                weight_mismatch = (key, f["weight"], r["weight"], abs(f["weight"] - r["weight"]))
            if tile_scale_mismatch is None and abs(f["tile_scale"] - r["tile_scale"]) > 0:
                tile_scale_mismatch = (key, f["tile_scale"], r["tile_scale"], abs(f["tile_scale"] - r["tile_scale"]))
            if tile_alpha_mismatch is None and abs(f["tile_alpha"] - r["tile_alpha"]) > 0:
                tile_alpha_mismatch = (key, f["tile_alpha"], r["tile_alpha"], abs(f["tile_alpha"] - r["tile_alpha"]))
            if packed_mismatch is None:
                pd = (f["packed"] != r["packed"]).nonzero()
                if pd.numel():
                    idx = int(pd[0].item())
                    packed_mismatch = (key, idx, int(f["packed"][idx].item()), int(r["packed"][idx].item()))
            if scale_mismatch is None:
                sd = (f["scale"] != r["scale"]).nonzero()
                if sd.numel():
                    idx = int(sd[0].item())
                    scale_mismatch = (key, idx, int(f["scale"][idx].item()), int(r["scale"][idx].item()))
            if all(x is not None for x in [weight_mismatch, tile_scale_mismatch, tile_alpha_mismatch, packed_mismatch, scale_mismatch]):
                break
        print("semantic weight", weight_mismatch)
        print("semantic tile_scale", tile_scale_mismatch)
        print("semantic tile_alpha", tile_alpha_mismatch)
        print("semantic packed", packed_mismatch)
        print("semantic scale", scale_mismatch)

    consumer_from_fused = torch.empty_like(expected_normed)
    _launch_prequantized_moe_consumer(
        expected_normed,
        experts=combined_experts,
        workspace=ws,
        routing=ref_routing,
        output=consumer_from_fused,
        input_scales_are_reciprocal=True,
        fast_math=False,
        fc1_tile_amax=False,
        fc2_tile_amax=False,
    )
    consumer_metrics = _float_stats(consumer_from_fused, torch.empty_like(consumer_from_fused).copy_(consumer_from_fused))
    del consumer_metrics
    from b12x.moe.fused.reference import compare_to_reference  # local import to keep top short
    print("consumer_from_fused_ws", compare_to_reference(consumer_from_fused, _launch_prequantized_moe_consumer(
        expected_normed,
        experts=combined_experts,
        workspace=ref_ws,
        routing=ref_routing,
        output=torch.empty_like(expected_normed),
        input_scales_are_reciprocal=True,
        fast_math=False,
        fc1_tile_amax=False,
        fc2_tile_amax=False,
    )))


if __name__ == "__main__":
    main()
