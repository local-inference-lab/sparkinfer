"""MoEMicroKernelBackend — BF16 micro routed MoE backend."""

from __future__ import annotations

from typing import Tuple

import torch

from b12x.cute.utils import get_num_sm
from b12x.moe.fused.bf16.indexed_dense import (
    ExpertIndexedDenseGemmKernel,
    run_dense_bf16_expert_ids,
)
from b12x.moe.fused.bf16.static import (
    DenseGemmKernel,
    MoEStaticKernelBackend,
    current_cuda_stream,
    run_dense_bf16,
)


class MoEMicroKernelBackend(MoEStaticKernelBackend):
    implementation = "micro"
    default_expert_chunk_size = 24
    relu2_expert_chunk_size = 88
    group_bmm_min_experts = 1
    row1_dense_max_active_clusters = 96
    row1_multi_token_dense_max_active_clusters = 128
    single_token_dense_tile_shape_mnk = (16, 128, 64)
    static_fallback_tail_experts = 4

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        input_scales_are_reciprocal: bool = False,
        activation: str = "silu",
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            exact_mma_m_tiles=False,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            activation=activation,
        )
        self._indexed_fc1_dense_kernel: ExpertIndexedDenseGemmKernel | None = None
        self._indexed_fc2_dense_kernel: ExpertIndexedDenseGemmKernel | None = None
        self._indexed_max_active_clusters: int | None = None
        self._indexed_device_index: int | None = None

    def get_expert_indexed_dense_runtime(
        self, device: torch.device
    ) -> tuple[ExpertIndexedDenseGemmKernel, ExpertIndexedDenseGemmKernel, int]:
        device_index = device.index or 0
        if (
            self._indexed_fc1_dense_kernel is None
            or self._indexed_fc2_dense_kernel is None
            or self._indexed_device_index != device_index
        ):
            self._indexed_fc1_dense_kernel = ExpertIndexedDenseGemmKernel(
                self.single_token_dense_tile_shape_mnk,
                epilogue="relu2",
            )
            self._indexed_fc2_dense_kernel = ExpertIndexedDenseGemmKernel(
                self.single_token_dense_tile_shape_mnk
            )
            self._indexed_fc1_dense_kernel.configure_atom_layout((1, 2, 1))
            self._indexed_fc2_dense_kernel.configure_atom_layout((1, 2, 1))
            self._indexed_max_active_clusters = get_num_sm(device)
            self._indexed_device_index = device_index
        return (
            self._indexed_fc1_dense_kernel,  # type: ignore[return-value]
            self._indexed_fc2_dense_kernel,  # type: ignore[return-value]
            self._indexed_max_active_clusters,  # type: ignore[return-value]
        )

    def _should_fallback_to_static(self, routing) -> bool:
        if self.activation != "relu2":
            return False
        if not routing.chunk_plans:
            return False
        if len(routing.micro_group_plans) <= 1:
            return False
        if routing.micro_group_plans[0].rows != 1:
            return False
        return max(
            (len(group.expert_ids_cpu) for group in routing.micro_group_plans[1:]),
            default=0,
        ) < self.static_fallback_tail_experts

    def run(
        self,
        *,
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routing,
        workspace,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self._should_fallback_to_static(routing):
            return super().run(
                a=a,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                routing=routing,
                workspace=workspace,
                output=output,
            )
        routed_rows = topk_ids.numel()
        use_grouped_relu2_gather = (
            self.activation == "relu2"
            and routing.micro_group_token_indices is not None
            and routing.micro_group_route_indices is not None
        )
        use_fast_relu2_combine = self.activation == "relu2" and topk_ids.shape[1] > 2
        use_row1_direct_input_gather = (
            use_grouped_relu2_gather
            and len(routing.micro_group_plans) > 0
            and routing.micro_group_plans[0].rows == 1
            and workspace.micro_row1_routed_input_chunk is not None
        )
        use_single_token_row1_direct_reduce = (
            use_fast_relu2_combine
            and workspace.micro_topk_weights_bf16 is not None
            and a.shape[0] == 1
            and len(routing.micro_group_plans) == 1
            and routing.micro_group_plans[0].rows == 1
        )
        use_fast_relu2_bmm_combine = (
            use_fast_relu2_combine
            and not use_single_token_row1_direct_reduce
            and workspace.micro_topk_weights_bf16 is not None
        )
        need_sorted_weights = not use_fast_relu2_bmm_combine
        flat_topk_weights = (
            topk_weights.reshape(-1).to(torch.float32) if need_sorted_weights else None
        )
        if (
            workspace.micro_fc1_output_flat is None
            or workspace.micro_fc2_output_flat is None
            or workspace.micro_accum_output_float is None
            or (self.is_gated and workspace.micro_intermediate_flat is None)
            or (
                not use_grouped_relu2_gather
                and (
                    workspace.micro_routed_input_flat is None
                    or (
                        need_sorted_weights
                        and workspace.micro_sorted_weights_flat is None
                    )
                )
            )
        ):
            raise RuntimeError("micro workspace scratch is not initialized")

        if use_grouped_relu2_gather:
            # Relu2 group order is static for a captured routing. Gather once
            # into routed-row scratch, then hand each group a flat slice.
            if use_row1_direct_input_gather:
                row1_group = routing.micro_group_plans[0]
                row1_group_rows = len(row1_group.expert_ids_cpu)
                row1_routed_dense = workspace.micro_row1_routed_input_chunk[
                    0, :, :row1_group_rows
                ].transpose(0, 1)
                torch.index_select(
                    a,
                    0,
                    row1_group.token_indices_gpu,
                    out=row1_routed_dense,
                )
                if row1_group_rows < routed_rows:
                    torch.index_select(
                        a,
                        0,
                        routing.micro_group_token_indices[row1_group_rows:routed_rows],
                        out=workspace.routed_input[row1_group_rows:routed_rows],
                    )
            else:
                torch.index_select(
                    a,
                    0,
                    routing.micro_group_token_indices,
                    out=workspace.routed_input[:routed_rows],
                )
            if need_sorted_weights:
                assert flat_topk_weights is not None
                torch.index_select(
                    flat_topk_weights,
                    0,
                    routing.micro_group_route_indices,
                    out=workspace.sorted_weights[:routed_rows],
                )

        dense_runtime = (
            self._get_dense_runtime(a.device) if self.activation == "relu2" else None
        )
        indexed_dense_runtime = (
            self.get_expert_indexed_dense_runtime(a.device)
            if self.activation == "relu2"
            else None
        )
        direct_w1_view = w1.permute(1, 2, 0) if self.activation == "relu2" else None
        direct_w2_view = w2.permute(1, 2, 0) if self.activation == "relu2" else None
        topk_weights_bf16 = (
            workspace.micro_topk_weights_bf16[: a.shape[0]]
            if use_fast_relu2_bmm_combine
            else None
        )
        if topk_weights_bf16 is not None:
            topk_weights_bf16.copy_(topk_weights)

        for group in routing.micro_group_plans:
            group_e = len(group.expert_ids_cpu)
            rows = group.rows
            group_rows = group_e * rows
            intermediate_n = (
                group.w2_group.shape[1] if group.w2_group is not None else w2.shape[1]
            )
            if use_grouped_relu2_gather:
                flat_begin = group.flat_offset
                flat_end = flat_begin + group_rows
                routed_input_flat = workspace.routed_input[flat_begin:flat_end]
                sorted_weights_flat = (
                    workspace.sorted_weights[flat_begin:flat_end]
                    if need_sorted_weights
                    else None
                )
            else:
                assert workspace.micro_routed_input_flat is not None
                routed_input_flat = workspace.micro_routed_input_flat[:group_rows]
                sorted_weights_flat = (
                    workspace.micro_sorted_weights_flat[:group_rows]
                    if need_sorted_weights
                    else None
                )

            fc1_output_flat = workspace.micro_fc1_output_flat[:group_rows]
            if self.is_gated:
                assert workspace.micro_intermediate_flat is not None
                intermediate_flat = workspace.micro_intermediate_flat[
                    :group_rows, :intermediate_n
                ]
            else:
                intermediate_flat = fc1_output_flat
            fc2_output_flat = workspace.micro_fc2_output_flat[:group_rows]

            if not use_grouped_relu2_gather:
                torch.index_select(
                    a,
                    0,
                    group.token_indices_gpu,
                    out=routed_input_flat,
                )
                if need_sorted_weights:
                    assert flat_topk_weights is not None
                    assert sorted_weights_flat is not None
                    torch.index_select(
                        flat_topk_weights,
                        0,
                        group.flat_route_indices_gpu,
                        out=sorted_weights_flat,
                    )

            use_direct_weight_lookup = group.weight_expert_ids_gpu is not None
            use_row1_dense = self.activation == "relu2" and rows == 1
            if use_row1_dense:
                assert dense_runtime is not None
                assert workspace.micro_row1_routed_input_chunk is not None
                assert workspace.micro_row1_fc1_output_chunk is not None
                assert workspace.micro_row1_fc2_output_chunk is not None
                if use_direct_weight_lookup:
                    assert indexed_dense_runtime is not None
                    fc1_dense_kernel = indexed_dense_runtime[0]
                    fc2_dense_kernel = indexed_dense_runtime[1]
                    max_active_clusters = indexed_dense_runtime[2]
                else:
                    (
                        fc1_dense_kernel,
                        fc2_dense_kernel,
                        max_active_clusters,
                    ) = dense_runtime

                # The dense row1 path needs exact 1-row batched scratch.
                # Slicing [:1, ...] out of a wider micro chunk changes the
                # K stride when other routing groups have rows > 1, which
                # corrupts the dense GEMM.
                routed_dense = workspace.micro_row1_routed_input_chunk[:, :, :group_e]
                fc1_dense = workspace.micro_row1_fc1_output_chunk[:, :, :group_e]
                fc2_dense = workspace.micro_row1_fc2_output_chunk[:, :, :group_e]
                dense_cluster_limit = min(
                    max_active_clusters,
                    self.row1_multi_token_dense_max_active_clusters
                    if a.shape[0] > 1
                    else self.row1_dense_max_active_clusters,
                )
                if not (use_row1_direct_input_gather and group.flat_offset == 0):
                    routed_dense[0].copy_(routed_input_flat.transpose(0, 1))
                if use_direct_weight_lookup:
                    run_dense_bf16_expert_ids(
                        fc1_dense_kernel,
                        routed_dense,
                        w1.permute(1, 2, 0),
                        group.weight_expert_ids_gpu,
                        fc1_dense,
                        dense_cluster_limit,
                        current_cuda_stream(),
                    )
                    run_dense_bf16_expert_ids(
                        fc2_dense_kernel,
                        fc1_dense,
                        w2.permute(1, 2, 0),
                        group.weight_expert_ids_gpu,
                        fc2_dense,
                        dense_cluster_limit,
                        current_cuda_stream(),
                    )
                else:
                    assert group.w1_group is not None
                    assert group.w2_group is not None
                    run_dense_bf16(
                        fc1_dense_kernel,
                        routed_dense,
                        group.w1_group,
                        fc1_dense,
                        dense_cluster_limit,
                        current_cuda_stream(),
                    )
                    run_dense_bf16(
                        fc2_dense_kernel,
                        fc1_dense,
                        group.w2_group,
                        fc2_dense,
                        dense_cluster_limit,
                        current_cuda_stream(),
                    )
                if use_single_token_row1_direct_reduce:
                    assert workspace.micro_topk_weights_bf16 is not None
                    assert sorted_weights_flat is not None
                    single_token_weights_bf16 = workspace.micro_topk_weights_bf16[
                        0, :group_e
                    ]
                    single_token_weights_bf16.copy_(sorted_weights_flat)
                    torch.bmm(
                        single_token_weights_bf16.view(1, 1, group_e),
                        fc2_dense[0].transpose(0, 1).unsqueeze(0),
                        out=output.view(1, 1, a.shape[1]),
                    )
                    return output
                if use_fast_relu2_bmm_combine:
                    workspace.routed_output_unsorted[:routed_rows].index_copy_(
                        0,
                        group.flat_route_indices_gpu,
                        fc2_dense[0].transpose(0, 1),
                    )
                    continue
                fc2_output_flat.copy_(fc2_dense[0].transpose(0, 1))
            else:
                if use_direct_weight_lookup:
                    assert indexed_dense_runtime is not None
                    assert direct_w1_view is not None
                    assert direct_w2_view is not None
                    grouped_routed_chunk = workspace.routed_input_chunk[
                        :rows, :, :group_e
                    ]
                    grouped_fc1_chunk = workspace.fc1_output_chunk[
                        :rows, : w1.shape[1], :group_e
                    ]
                    grouped_fc2_chunk = workspace.fc2_output_chunk[
                        :rows, :, :group_e
                    ]
                    grouped_routed_chunk.copy_(
                        routed_input_flat.view(group_e, rows, -1).permute(1, 2, 0)
                    )
                    fc1_dense_kernel = indexed_dense_runtime[0]
                    fc2_dense_kernel = indexed_dense_runtime[1]
                    max_active_clusters = indexed_dense_runtime[2]
                    run_dense_bf16_expert_ids(
                        fc1_dense_kernel,
                        grouped_routed_chunk,
                        direct_w1_view,
                        group.weight_expert_ids_gpu,
                        grouped_fc1_chunk,
                        max_active_clusters,
                        current_cuda_stream(),
                    )
                    run_dense_bf16_expert_ids(
                        fc2_dense_kernel,
                        grouped_fc1_chunk,
                        direct_w2_view,
                        group.weight_expert_ids_gpu,
                        grouped_fc2_chunk,
                        max_active_clusters,
                        current_cuda_stream(),
                    )
                    fc2_output_flat.copy_(
                        grouped_fc2_chunk.permute(2, 0, 1).reshape(group_rows, -1)
                    )
                else:
                    assert group.w1_group is not None
                    assert group.w2_group is not None
                    routed_batch = routed_input_flat.view(group_e, rows, -1)
                    fc1_batch = fc1_output_flat.view(group_e, rows, -1)
                    intermediate_batch = intermediate_flat.view(group_e, rows, -1)
                    fc2_batch = fc2_output_flat.view(group_e, rows, -1)
                    w1_batch = group.w1_group.permute(2, 1, 0)
                    w2_batch = group.w2_group.permute(2, 1, 0)

                    # Keep each GEMM at one routed row per expert. Batched 1xK/KxN
                    # BF16 matmuls match the per-expert reference path exactly,
                    # while still collapsing hundreds of tiny launches into a
                    # handful of group-wide calls without paying chunk padding for
                    # experts with fewer routed rows than their neighbors.
                    if rows > 1 and group_e >= self.group_bmm_min_experts:
                        torch.bmm(
                            routed_batch,
                            w1_batch,
                            out=fc1_batch,
                        )
                    else:
                        for row_idx in range(rows):
                            torch.bmm(
                                routed_batch[:, row_idx : row_idx + 1, :],
                                w1_batch,
                                out=fc1_batch[:, row_idx : row_idx + 1, :],
                            )

                    if self.is_gated:
                        up = fc1_batch[:, :, :intermediate_n]
                        gate = fc1_batch[:, :, intermediate_n:]
                        intermediate_batch.copy_(
                            (
                                torch.sigmoid(gate.float())
                                * gate.float()
                                * up.float()
                            ).to(torch.bfloat16)
                        )
                    else:
                        fc1_output_flat.relu_()
                        fc1_output_flat.mul_(fc1_output_flat)

                    if rows > 1 and group_e >= self.group_bmm_min_experts:
                        torch.bmm(
                            intermediate_batch,
                            w2_batch,
                            out=fc2_batch,
                        )
                    else:
                        for row_idx in range(rows):
                            torch.bmm(
                                intermediate_batch[:, row_idx : row_idx + 1, :],
                                w2_batch,
                                out=fc2_batch[:, row_idx : row_idx + 1, :],
                            )

            if sorted_weights_flat is not None:
                fc2_output_flat.mul_(sorted_weights_flat[:, None])
            workspace.routed_output_unsorted[:routed_rows].index_copy_(
                0,
                group.flat_route_indices_gpu,
                fc2_output_flat,
            )

        route_outputs = workspace.routed_output_unsorted[:routed_rows].view(
            a.shape[0], topk_ids.shape[1], a.shape[1]
        )
        if use_fast_relu2_bmm_combine:
            assert topk_weights_bf16 is not None
            torch.bmm(
                topk_weights_bf16.unsqueeze(1),
                route_outputs,
                out=output.unsqueeze(1),
            )
        else:
            torch.sum(
                route_outputs,
                dim=1,
                dtype=torch.float32,
                out=workspace.micro_accum_output_float[: a.shape[0]],
            )
            output.copy_(workspace.micro_accum_output_float[: a.shape[0]])
        return output


__all__ = ["MoEMicroKernelBackend"]
