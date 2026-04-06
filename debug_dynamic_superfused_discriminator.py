from __future__ import annotations

import os
import pathlib
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

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
from b12x.distributed.pcie_oneshot import PCIeOneshotAllReduce  # noqa: E402
from b12x.integration.tp_moe import (  # noqa: E402
    _alloc_dynamic_superfused_workspace,
    _alloc_workspace,
    _append_expert_bank,
    _append_shared_expert_routing,
    _b12x_gemma_moe_block_fp4_dynamic_superfused,
    _effective_input_scales,
    _launch_prequantized_moe_consumer,
    _populate_dynamic_prequantized_workspace_host,
    _shared_expert_gate_weights,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    b12x_route_experts_fast,
    clear_tp_moe_caches,
)
from b12x.moe.fused.reference import (  # noqa: E402
    _dequant_fp4,
    compare_to_reference,
    unswizzle_block_scale,
)


def _decode_rows_by_key(workspace, *, hidden_size: int, task_tail: int) -> dict[tuple[int, int], torch.Tensor]:
    fp4_lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=workspace.packed_input.device,
    )
    cols_blocks = hidden_size // 16
    by_tile: dict[int, tuple[int, int]] = {}
    for slot in range(task_tail):
        phys_tile = int(workspace.task_m_tile[slot].item())
        if phys_tile not in by_tile:
            by_tile[phys_tile] = (
                int(workspace.task_expert[slot].item()),
                int(workspace.task_valid_rows[slot].item()),
            )
    decoded: dict[tuple[int, int], torch.Tensor] = {}
    for phys_tile, (expert_id, valid_rows) in by_tile.items():
        phys_row = phys_tile * 128
        packed = workspace.packed_input[0, phys_row : phys_row + 128]
        scales = workspace.packed_input_scale[phys_row : phys_row + 128]
        raw = _dequant_fp4(packed, 128, hidden_size, fp4_lut)
        sf = unswizzle_block_scale(scales, 128, cols_blocks)
        dequant = raw.reshape(128, cols_blocks, 16) * sf.unsqueeze(-1)
        dequant = dequant.reshape(128, hidden_size)
        for row_idx in range(valid_rows):
            token_idx = int(workspace.token_map[phys_row + row_idx].item())
            decoded[(expert_id, token_idx)] = dequant[row_idx].clone()
    return decoded


def _decode_weights_by_key(workspace, *, task_tail: int) -> dict[tuple[int, int], float]:
    by_tile: dict[int, tuple[int, int]] = {}
    for slot in range(task_tail):
        phys_tile = int(workspace.task_m_tile[slot].item())
        if phys_tile not in by_tile:
            by_tile[phys_tile] = (
                int(workspace.task_expert[slot].item()),
                int(workspace.task_valid_rows[slot].item()),
            )
    decoded: dict[tuple[int, int], float] = {}
    for phys_tile, (expert_id, valid_rows) in by_tile.items():
        phys_row = phys_tile * 128
        for row_idx in range(valid_rows):
            token_idx = int(workspace.token_map[phys_row + row_idx].item())
            decoded[(expert_id, token_idx)] = float(workspace.token_weights[phys_row + row_idx].item())
    return decoded


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

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    clear_tp_moe_caches()
    spec = _make_spec()
    m = 2
    hidden_local = make_input_activations(spec, m, seed=19100 + _rank(), device=device)
    residual = make_input_activations(spec, m, seed=20100 + _rank(), device=device)
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

        fused_workspace = _alloc_dynamic_superfused_workspace(
            weight_E=combined_experts.w1_fp4.shape[0],
            k=spec.hidden_size,
            n=combined_experts.w2_fp4.shape[2] * 2,
            num_topk=spec.top_k + 1,
            device=device,
            dtype=hidden_local.dtype,
            a1_gscale=combined_experts.a1_gscale,
            a2_gscale=combined_experts.a2_gscale,
            routed_rows=m * (spec.top_k + 1),
            input_scales_static=True,
        )
        fused_out, fused_residual_out = _b12x_gemma_moe_block_fp4_dynamic_superfused(
            hidden_local,
            residual,
            pre_mlp_runtime=runtime,
            norm_weight=norm_weight,
            norm_eps=1e-6,
            sparse_experts=sparse_experts,
            shared_expert=shared_expert,
            shared_gate_weight=shared_gate_weight,
            combined_experts=combined_experts,
            workspace=fused_workspace,
            top_k=spec.top_k,
            gate_weight=gate_weight,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
        )

        reduced = hidden_local.clone()
        dist.all_reduce(reduced)
        normed_hidden_states, semi_residual_out = _gemma_rmsnorm_after_allreduce(
            reduced,
            residual,
            norm_weight,
            1e-6,
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

        dynamic_ws = _alloc_workspace(
            implementation="dynamic",
            state_E=combined_experts.w1_fp4.shape[0],
            weight_E=combined_experts.w1_fp4.shape[0],
            k=spec.hidden_size,
            n=combined_experts.w2_fp4.shape[2] * 2,
            num_topk=spec.top_k + 1,
            device=device,
            dtype=hidden_local.dtype,
            a1_gscale=combined_experts.a1_gscale,
            a2_gscale=combined_experts.a2_gscale,
            routed_rows=m * (spec.top_k + 1),
            max_rows=fused_workspace.max_rows,
            input_scales_static=True,
            dynamic_physical_tiles=fused_workspace.physical_tiles_capacity,
            dynamic_task_capacity=fused_workspace.task_capacity,
        )
        dynamic_ws.row_counts.copy_(fused_workspace.row_counts)
        dynamic_ws.token_map.copy_(fused_workspace.token_map)
        dynamic_ws.token_weights.copy_(fused_workspace.token_weights)
        dynamic_ws.packed_input.copy_(fused_workspace.packed_input)
        dynamic_ws.packed_input_scale.copy_(fused_workspace.packed_input_scale)
        dynamic_ws.task_head.zero_()
        dynamic_ws.task_tail.copy_(fused_workspace.task_tail)
        dynamic_ws.task_ready.copy_(fused_workspace.task_ready)
        dynamic_ws.task_expert.copy_(fused_workspace.task_expert)
        dynamic_ws.task_m_tile.copy_(fused_workspace.task_m_tile)
        dynamic_ws.task_slice_begin.copy_(fused_workspace.task_slice_begin)
        dynamic_ws.task_slice_count.copy_(fused_workspace.task_slice_count)
        dynamic_ws.task_valid_rows.copy_(fused_workspace.task_valid_rows)
        dynamic_ws.all_work_published.copy_(fused_workspace.all_work_published)

        dynamic_from_fused = _launch_prequantized_moe_consumer(
            normed_hidden_states,
            experts=combined_experts,
            workspace=dynamic_ws,
            routing=combined_routing,
            input_scales_are_reciprocal=True,
            fc1_tile_amax=False,
            fc2_tile_amax=False,
        )

        host_ws = _alloc_workspace(
            implementation="dynamic",
            state_E=combined_experts.w1_fp4.shape[0],
            weight_E=combined_experts.w1_fp4.shape[0],
            k=spec.hidden_size,
            n=combined_experts.w2_fp4.shape[2] * 2,
            num_topk=spec.top_k + 1,
            device=device,
            dtype=hidden_local.dtype,
            a1_gscale=combined_experts.a1_gscale,
            a2_gscale=combined_experts.a2_gscale,
            routed_rows=m * (spec.top_k + 1),
            max_rows=fused_workspace.max_rows,
            input_scales_static=True,
            dynamic_physical_tiles=fused_workspace.physical_tiles_capacity,
            dynamic_task_capacity=fused_workspace.task_capacity,
        )
        _populate_dynamic_prequantized_workspace_host(
            host_ws,
            a=normed_hidden_states,
            topk_ids=combined_routing.topk_ids,
            topk_weights=combined_routing.topk_weights,
            expert_input_scale=_effective_input_scales(
                combined_experts.a1_gscale,
                combined_experts.w1_fp4.shape[0],
                input_scales_are_reciprocal=True,
            ),
            expert_alpha=combined_experts.w1_alphas,
            n=combined_experts.w2_fp4.shape[2] * 2,
            fc1_tile_amax=False,
        )

        host_out = _launch_prequantized_moe_consumer(
            normed_hidden_states,
            experts=combined_experts,
            workspace=host_ws,
            routing=combined_routing,
            input_scales_are_reciprocal=True,
            fc1_tile_amax=False,
            fc2_tile_amax=False,
        )

        payload_swapped_ws = _alloc_workspace(
            implementation="dynamic",
            state_E=combined_experts.w1_fp4.shape[0],
            weight_E=combined_experts.w1_fp4.shape[0],
            k=spec.hidden_size,
            n=combined_experts.w2_fp4.shape[2] * 2,
            num_topk=spec.top_k + 1,
            device=device,
            dtype=hidden_local.dtype,
            a1_gscale=combined_experts.a1_gscale,
            a2_gscale=combined_experts.a2_gscale,
            routed_rows=m * (spec.top_k + 1),
            max_rows=fused_workspace.max_rows,
            input_scales_static=True,
            dynamic_physical_tiles=fused_workspace.physical_tiles_capacity,
            dynamic_task_capacity=fused_workspace.task_capacity,
        )
        payload_swapped_ws.row_counts.copy_(host_ws.row_counts)
        payload_swapped_ws.token_map.copy_(host_ws.token_map)
        payload_swapped_ws.token_weights.copy_(host_ws.token_weights)
        payload_swapped_ws.packed_input.copy_(fused_workspace.packed_input)
        payload_swapped_ws.packed_input_scale.copy_(fused_workspace.packed_input_scale)
        payload_swapped_ws.task_head.zero_()
        payload_swapped_ws.task_tail.copy_(host_ws.task_tail)
        payload_swapped_ws.task_ready.copy_(host_ws.task_ready)
        payload_swapped_ws.task_expert.copy_(host_ws.task_expert)
        payload_swapped_ws.task_m_tile.copy_(host_ws.task_m_tile)
        payload_swapped_ws.task_slice_begin.copy_(host_ws.task_slice_begin)
        payload_swapped_ws.task_slice_count.copy_(host_ws.task_slice_count)
        payload_swapped_ws.task_valid_rows.copy_(host_ws.task_valid_rows)
        payload_swapped_ws.all_work_published.copy_(host_ws.all_work_published)

        metadata_swapped_ws = _alloc_workspace(
            implementation="dynamic",
            state_E=combined_experts.w1_fp4.shape[0],
            weight_E=combined_experts.w1_fp4.shape[0],
            k=spec.hidden_size,
            n=combined_experts.w2_fp4.shape[2] * 2,
            num_topk=spec.top_k + 1,
            device=device,
            dtype=hidden_local.dtype,
            a1_gscale=combined_experts.a1_gscale,
            a2_gscale=combined_experts.a2_gscale,
            routed_rows=m * (spec.top_k + 1),
            max_rows=fused_workspace.max_rows,
            input_scales_static=True,
            dynamic_physical_tiles=fused_workspace.physical_tiles_capacity,
            dynamic_task_capacity=fused_workspace.task_capacity,
        )
        metadata_swapped_ws.row_counts.copy_(fused_workspace.row_counts)
        metadata_swapped_ws.token_map.copy_(fused_workspace.token_map)
        metadata_swapped_ws.token_weights.copy_(fused_workspace.token_weights)
        metadata_swapped_ws.packed_input.copy_(host_ws.packed_input)
        metadata_swapped_ws.packed_input_scale.copy_(host_ws.packed_input_scale)
        metadata_swapped_ws.task_head.zero_()
        metadata_swapped_ws.task_tail.copy_(fused_workspace.task_tail)
        metadata_swapped_ws.task_ready.copy_(fused_workspace.task_ready)
        metadata_swapped_ws.task_expert.copy_(fused_workspace.task_expert)
        metadata_swapped_ws.task_m_tile.copy_(fused_workspace.task_m_tile)
        metadata_swapped_ws.task_slice_begin.copy_(fused_workspace.task_slice_begin)
        metadata_swapped_ws.task_slice_count.copy_(fused_workspace.task_slice_count)
        metadata_swapped_ws.task_valid_rows.copy_(fused_workspace.task_valid_rows)
        metadata_swapped_ws.all_work_published.copy_(fused_workspace.all_work_published)

        payload_swapped_out = _launch_prequantized_moe_consumer(
            normed_hidden_states,
            experts=combined_experts,
            workspace=payload_swapped_ws,
            routing=combined_routing,
            input_scales_are_reciprocal=True,
            fc1_tile_amax=False,
            fc2_tile_amax=False,
        )
        metadata_swapped_out = _launch_prequantized_moe_consumer(
            normed_hidden_states,
            experts=combined_experts,
            workspace=metadata_swapped_ws,
            routing=combined_routing,
            input_scales_are_reciprocal=True,
            fc1_tile_amax=False,
            fc2_tile_amax=False,
        )
        torch.cuda.synchronize(device)

        fused_metrics = compare_to_reference(fused_out, semi_out)
        copied_consumer_metrics = compare_to_reference(dynamic_from_fused, semi_out)
        host_metrics = compare_to_reference(host_out, semi_out)
        payload_swapped_metrics = compare_to_reference(payload_swapped_out, semi_out)
        metadata_swapped_metrics = compare_to_reference(metadata_swapped_out, semi_out)
        residual_metrics = compare_to_reference(fused_residual_out, semi_residual_out)
        fused_rows = _decode_rows_by_key(
            fused_workspace,
            hidden_size=spec.hidden_size,
            task_tail=int(fused_workspace.task_tail.item()),
        )
        host_rows = _decode_rows_by_key(
            host_ws,
            hidden_size=spec.hidden_size,
            task_tail=int(host_ws.task_tail.item()),
        )
        fused_keys = set(fused_rows)
        host_keys = set(host_rows)
        shared_keys = sorted(fused_keys & host_keys)
        missing_keys = len(host_keys - fused_keys)
        extra_keys = len(fused_keys - host_keys)
        if shared_keys:
            fused_stack = torch.stack([fused_rows[key] for key in shared_keys])
            host_stack = torch.stack([host_rows[key] for key in shared_keys])
            row_metrics = compare_to_reference(fused_stack, host_stack)
            host_norms = host_stack.float().norm(dim=1).clamp_min(1e-12)
            fused_norms = fused_stack.float().norm(dim=1)
            norm_ratio = fused_norms / host_norms
            mean_norm_ratio = float(norm_ratio.mean().item())
            max_norm_ratio = float(norm_ratio.max().item())
            min_norm_ratio = float(norm_ratio.min().item())
            worst_key = max(
                shared_keys,
                key=lambda key: (fused_rows[key].float() - host_rows[key].float()).abs().max().item(),
            )
            worst_row_max_abs = (
                fused_rows[worst_key].float() - host_rows[worst_key].float()
            ).abs().max().item()
            worst_row_diff = (
                fused_rows[worst_key].float() - host_rows[worst_key].float()
            ).abs().view(-1, 16).amax(dim=1)
            worst_block_idx = int(worst_row_diff.argmax().item())
            worst_block_max_abs = float(worst_row_diff[worst_block_idx].item())
            worst_same_key_cos = torch.nn.functional.cosine_similarity(
                fused_rows[worst_key].float().view(1, -1),
                host_rows[worst_key].float().view(1, -1),
            ).item()
            same_token_keys = [key for key in shared_keys if key[1] == worst_key[1]]
            best_alt_key = max(
                same_token_keys,
                key=lambda key: torch.nn.functional.cosine_similarity(
                    fused_rows[worst_key].float().view(1, -1),
                    host_rows[key].float().view(1, -1),
                ).item(),
            )
            best_alt_cos = torch.nn.functional.cosine_similarity(
                fused_rows[worst_key].float().view(1, -1),
                host_rows[best_alt_key].float().view(1, -1),
            ).item()
        else:
            row_metrics = None
            mean_norm_ratio = 0.0
            max_norm_ratio = 0.0
            min_norm_ratio = 0.0
            worst_key = None
            worst_row_max_abs = 0.0
            worst_block_idx = -1
            worst_block_max_abs = 0.0
            worst_same_key_cos = 0.0
            best_alt_key = None
            best_alt_cos = 0.0
        fused_weights = _decode_weights_by_key(
            fused_workspace,
            task_tail=int(fused_workspace.task_tail.item()),
        )
        host_weights = _decode_weights_by_key(
            host_ws,
            task_tail=int(host_ws.task_tail.item()),
        )
        if shared_keys:
            fused_weight_t = torch.tensor([fused_weights[key] for key in shared_keys], device=device)
            host_weight_t = torch.tensor([host_weights[key] for key in shared_keys], device=device)
            weight_metrics = compare_to_reference(
                fused_weight_t.view(1, -1),
                host_weight_t.view(1, -1),
            )
        else:
            weight_metrics = None
        _rank0_print(
            "dynamic superfused discriminator: "
            f"fused(max_abs={fused_metrics.max_abs:.3e}, cos={fused_metrics.cos:.6f}) "
            f"dynamic_consumer_from_fused_ws(max_abs={copied_consumer_metrics.max_abs:.3e}, cos={copied_consumer_metrics.cos:.6f}) "
            f"host_ws(max_abs={host_metrics.max_abs:.3e}, cos={host_metrics.cos:.6f}) "
            f"host_meta_fused_payload(max_abs={payload_swapped_metrics.max_abs:.3e}, cos={payload_swapped_metrics.cos:.6f}) "
            f"fused_meta_host_payload(max_abs={metadata_swapped_metrics.max_abs:.3e}, cos={metadata_swapped_metrics.cos:.6f}) "
            f"residual(max_abs={residual_metrics.max_abs:.3e}, cos={residual_metrics.cos:.6f}) "
            f"task_tail={int(fused_workspace.task_tail.item())} "
            f"shared_row_keys={len(shared_keys)} missing_row_keys={missing_keys} extra_row_keys={extra_keys}"
            + (
                f" row_payload(max_abs={row_metrics.max_abs:.3e}, cos={row_metrics.cos:.6f})"
                if row_metrics is not None
                else ""
            )
            + (
                f" row_norm_ratio(mean={mean_norm_ratio:.6f}, min={min_norm_ratio:.6f}, max={max_norm_ratio:.6f})"
                if row_metrics is not None
                else ""
            )
            + (
                f" row_weights(max_abs={weight_metrics.max_abs:.3e}, cos={weight_metrics.cos:.6f})"
                if weight_metrics is not None
                else ""
            )
            + (
                f" worst_row_key={worst_key} worst_row_max_abs={worst_row_max_abs:.3e}"
                if worst_key is not None
                else ""
            )
            + (
                f" worst_same_key_cos={worst_same_key_cos:.6f} best_same_token_match={best_alt_key} best_same_token_cos={best_alt_cos:.6f}"
                if best_alt_key is not None
                else ""
            )
            + (
                f" worst_block_idx={worst_block_idx} worst_block_max_abs={worst_block_max_abs:.3e}"
                if worst_key is not None
                else ""
            )
        )
    finally:
        runtime.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
