from __future__ import annotations

import os
import pathlib
import sys

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

from benchmarks.benchmark_gemma_moe_block_paths import (  # noqa: E402
    _gemma_rmsnorm_after_allreduce,
    _load_post_attention_layernorm_weight,
    _pack_shared_expert,
    _pack_sparse_experts_per_expert,
)
from benchmarks.benchmark_moe import (  # noqa: E402
    MODEL_PATH,
    ModelSpec,
    load_expert_weights,
    load_gate_weight,
    load_shared_expert_weights,
    load_shared_gate_weight,
    make_input_activations,
)
from b12x.cute.utils import current_cuda_stream, make_ptr  # noqa: E402
from b12x.distributed._oneshot_common import SIGNAL_BYTES  # noqa: E402
from b12x.distributed.pcie_oneshot import PCIeOneshotAllReduce  # noqa: E402
from b12x.integration.tp_moe import (  # noqa: E402
    _append_expert_bank,
    _append_shared_expert_routing,
    _b12x_gemma_moe_block_fp4_static_producer,
    _effective_input_scales,
    _get_weight_views,
    _launch_prequantized_moe_consumer,
    _prepare_expert_scale,
    _shared_expert_gate_weights,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    b12x_route_experts_fast,
    clear_tp_moe_caches,
)
from b12x.moe.fused.monolithic_superfused_static_parity_runtime import (  # noqa: E402
    _get_superfused_static_parity_launch,
    launch_superfused_static_parity,
)
from b12x.moe.fused.pre_mlp_static import UnifiedPreMLPIPC  # noqa: E402
from b12x.moe.fused.reference import compare_to_reference  # noqa: E402


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank0_print(msg: str) -> None:
    if _rank() == 0:
        print(msg, flush=True)


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=_world_size(),
        tp_rank=_rank(),
    )


def main() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", _local_rank()))
    torch.cuda.set_device(_local_rank())
    device = torch.device("cuda", _local_rank())
    torch.set_grad_enabled(False)

    clear_tp_moe_caches()
    spec = _make_spec()
    m = 2
    hidden_local = make_input_activations(spec, m, seed=22100 + _rank(), device=device)
    residual = make_input_activations(spec, m, seed=23100 + _rank(), device=device)

    max_input_bytes = m * spec.hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()
    runtime = PCIeOneshotAllReduce.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=max_input_bytes,
    )
    try:
        sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
        shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=0)
        gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=0)
        shared_gate_weight = load_shared_gate_weight(MODEL_PATH, layer_idx=0)
        norm_weight = _load_post_attention_layernorm_weight(MODEL_PATH, layer_idx=0, device=device)
        sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
        shared_expert = _pack_shared_expert(shared_weights)
        combined_experts = _append_expert_bank(sparse_experts, shared_expert)

        reduced = hidden_local.clone()
        dist.all_reduce(reduced)
        normed_hidden_states, semi_residual_out = _gemma_rmsnorm_after_allreduce(
            reduced, residual, norm_weight, 1e-6
        )
        sparse_routing = b12x_route_experts_fast(
            normed_hidden_states,
            top_k=spec.top_k,
            gate_weight=gate_weight,
            workspace=allocate_tp_moe_workspace_pool(),
        )
        combined_routing = _append_shared_expert_routing(
            sparse_routing,
            shared_gate_weights=_shared_expert_gate_weights(
                normed_hidden_states,
                gate_weight=shared_gate_weight,
            ),
            shared_expert_id=sparse_experts.w1_fp4.shape[0],
        )
        semi_out = b12x_moe_fp4(
            normed_hidden_states,
            combined_experts.a1_gscale,
            combined_experts.w1_fp4,
            combined_experts.w1_blockscale,
            combined_experts.w1_alphas,
            combined_experts.a2_gscale,
            combined_experts.w2_fp4,
            combined_experts.w2_blockscale,
            combined_experts.w2_alphas,
            combined_routing.topk_weights,
            combined_routing.topk_ids,
            workspace=allocate_tp_moe_workspace_pool(),
            input_scales_are_reciprocal=True,
            input_scales_static=True,
            fc2_tile_amax=False,
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
                    residual_out = torch.empty_like(semi_residual_out)
                residual_out.copy_(semi_residual_out)
                return out, residual_out

        fused_pool = allocate_tp_moe_workspace_pool()
        _, _, routing = _b12x_gemma_moe_block_fp4_static_producer(
            hidden_local,
            residual,
            pre_mlp_runtime=_FakePreMLPRuntime(),
            norm_weight=norm_weight,
            norm_eps=1e-6,
            sparse_experts=sparse_experts,
            shared_expert=shared_expert,
            shared_gate_weight=shared_gate_weight,
            combined_experts=combined_experts,
            workspace=fused_pool,
            top_k=spec.top_k,
            gate_weight=gate_weight,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
            fc1_tile_amax=False,
            return_routing=True,
        )
        ws = fused_pool.workspaces[next(iter(fused_pool.workspaces))]

        weights = _get_weight_views(
            combined_experts.w1_fp4,
            combined_experts.w1_blockscale,
            combined_experts.w2_fp4,
            combined_experts.w2_blockscale,
            combined_experts.w1_alphas,
            combined_experts.w2_alphas,
            combined_experts.w2_fp4.shape[2] * 2,
            spec.hidden_size,
        )
        ipc = UnifiedPreMLPIPC.from_oneshot_runtime(runtime, inp=hidden_local)
        signal_ptr_args = [
            make_ptr(cutlass.Int32, ptr, cute.AddressSpace.gmem, assumed_align=128)
            for ptr in ipc.signal_ptrs
        ]
        while len(signal_ptr_args) < 8:
            signal_ptr_args.append(signal_ptr_args[0])
        packed_a_u8 = ws.packed_input.permute(1, 2, 0)

        def _run_direct(*, world_size: int) -> torch.Tensor:
            compiled = _get_superfused_static_parity_launch(
                world_size=world_size,
                num_tokens=int(hidden_local.shape[0]),
                hidden_size=int(hidden_local.shape[1]),
                output_n=int(weights.down.shape[1] * 2),
                num_sparse_experts=sparse_experts.w1_fp4.shape[0],
                top_k=spec.top_k,
                state_E=int(ws.state_E),
                weight_E=int(ws.weight_E),
                max_rows=int(ws.max_rows),
                intermediate_size=int(weights.w13.shape[0] // 2),
                input_scales_are_reciprocal=True,
                fast_math=False,
                fc2_tile_amax=False,
                emit_normalized=False,
                renormalize_topk=True,
                prequantized_input=True,
            )
            out = torch.empty_like(hidden_local)
            direct_residual = torch.empty_like(hidden_local)
            direct_topk_ids = torch.empty((hidden_local.shape[0] * (spec.top_k + 1),), dtype=torch.int32, device=device)
            direct_topk_weights = torch.empty((hidden_local.shape[0] * (spec.top_k + 1),), dtype=torch.float32, device=device)
            compiled(
                hidden_local,
                hidden_local,
                hidden_local,
                hidden_local,
                hidden_local,
                hidden_local,
                hidden_local,
                hidden_local,
                *signal_ptr_args[:8],
                make_ptr(cutlass.Int32, ipc.signal_ptrs[ipc.rank], cute.AddressSpace.gmem, assumed_align=128),
                ipc.rank,
                residual,
                hidden_local,
                direct_residual,
                norm_weight,
                gate_weight,
                shared_gate_weight,
                direct_topk_ids,
                direct_topk_weights,
                packed_a_u8,
                ws.sfa_ptr,
                ws.packed_a_flat,
                ws.scale_flat,
                ws.barrier_count,
                ws.barrier_epoch,
                weights.w13,
                weights.sfb_w13_ptr,
                weights.down,
                weights.sfb_down_ptr,
                ws.row_counts,
                ws.active_expert_count,
                ws.weight_expert_ids,
                ws.global_to_local_expert,
                _effective_input_scales(
                    combined_experts.a1_gscale,
                    combined_experts.w1_fp4.shape[0],
                    input_scales_are_reciprocal=True,
                ),
                combined_experts.w1_alphas,
                combined_experts.w2_alphas,
                _prepare_expert_scale(
                    combined_experts.a2_gscale,
                    combined_experts.w2_fp4.shape[0],
                ),
                ws.fc1_tile_scale.view(-1),
                ws.fc1_tile_alpha.view(-1),
                out,
                ws.token_map,
                ws.token_weights,
                1,
                1e-6,
                current_cuda_stream(),
            )
            torch.cuda.synchronize(device)
            return out

        direct_out = _run_direct(world_size=ipc.world_size)
        direct_ws1_out = _run_direct(world_size=1)

        wrapped_out, _, _, _, _ = launch_superfused_static_parity(
            hidden_states=hidden_local,
            residual=residual,
            norm_weight=norm_weight,
            gate_weight=gate_weight,
            shared_gate_weight=shared_gate_weight,
            ipc=ipc,
            workspace=ws,
            weights=weights,
            input_global_scale=_effective_input_scales(
                combined_experts.a1_gscale,
                combined_experts.w1_fp4.shape[0],
                input_scales_are_reciprocal=True,
            ),
            expert_alpha=combined_experts.w1_alphas,
            down_alpha=combined_experts.w2_alphas,
            global_scale=_prepare_expert_scale(
                combined_experts.a2_gscale,
                combined_experts.w2_fp4.shape[0],
            ),
            num_sparse_experts=sparse_experts.w1_fp4.shape[0],
            top_k=spec.top_k,
            output=torch.empty_like(hidden_local),
            residual_out=torch.empty_like(hidden_local),
            input_scales_are_reciprocal=True,
            fast_math=False,
            fc2_tile_amax=False,
            renormalize_topk=True,
            eps=1e-6,
            prequantized_input=True,
        )
        torch.cuda.synchronize(device)

        parity_static_consumer = _launch_prequantized_moe_consumer(
            normed_hidden_states,
            experts=combined_experts,
            workspace=ws,
            routing=routing,
            input_scales_are_reciprocal=True,
            fc1_tile_amax=False,
            fc2_tile_amax=False,
        )

        direct_metrics = compare_to_reference(direct_out, semi_out)
        direct_ws1_metrics = compare_to_reference(direct_ws1_out, semi_out)
        wrapped_metrics = compare_to_reference(wrapped_out, semi_out)
        static_metrics = compare_to_reference(parity_static_consumer, semi_out)
        direct_vs_wrap = compare_to_reference(direct_out, wrapped_out)
        direct_ws1_vs_direct = compare_to_reference(direct_ws1_out, direct_out)
        _rank0_print(
            "distributed direct prequantized: "
            f"direct(max_abs={direct_metrics.max_abs:.3e}, cos={direct_metrics.cos:.6f}) "
            f"direct_ws1(max_abs={direct_ws1_metrics.max_abs:.3e}, cos={direct_ws1_metrics.cos:.6f}) "
            f"wrapped(max_abs={wrapped_metrics.max_abs:.3e}, cos={wrapped_metrics.cos:.6f}) "
            f"static(max_abs={static_metrics.max_abs:.3e}, cos={static_metrics.cos:.6f}) "
            f"direct_vs_wrapped(max_abs={direct_vs_wrap.max_abs:.3e}, cos={direct_vs_wrap.cos:.6f}) "
            f"direct_ws1_vs_direct(max_abs={direct_ws1_vs_direct.max_abs:.3e}, cos={direct_ws1_vs_direct.cos:.6f})"
        )
    finally:
        runtime.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
