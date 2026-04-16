"""MoEDynamicKernelBackend — BF16 dynamic routed MoE backend.

The BF16 dynamic backend shares the live fused chunk kernel with the static
backend and keeps the larger expert chunking policy that the host-side
selector uses for dynamic workloads.
"""

from __future__ import annotations

from typing import Tuple

import torch

from b12x.moe.fused.bf16.static import MoEStaticKernelBackend


class MoEDynamicKernelBackend(MoEStaticKernelBackend):
    implementation = "dynamic"
    default_expert_chunk_size = 16
    relu2_expert_chunk_size = 64
    vectorized_row_limit = 4

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        input_scales_are_reciprocal: bool = False,
        activation: str = "silu",
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            1,
            exact_mma_m_tiles=False,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            activation=activation,
        )

    def _populate_routed_chunk(
        self,
        *,
        a: torch.Tensor,
        workspace,
        chunk,
        routed_chunk: torch.Tensor,
    ) -> None:
        if self.activation != "relu2" or chunk.max_rows > self.vectorized_row_limit:
            return super()._populate_routed_chunk(
                a=a,
                workspace=workspace,
                chunk=chunk,
                routed_chunk=routed_chunk,
            )

        return self._populate_small_row_routed_chunk(
            a=a,
            chunk=chunk,
            routed_chunk=routed_chunk,
        )

    def _store_sorted_chunk_output(
        self,
        *,
        workspace,
        flat_topk_weights: torch.Tensor,
        chunk,
        fc2_chunk: torch.Tensor,
        use_route_order_output: bool,
    ) -> None:
        if self.activation != "relu2" or chunk.max_rows > self.vectorized_row_limit:
            return super()._store_sorted_chunk_output(
                workspace=workspace,
                flat_topk_weights=flat_topk_weights,
                chunk=chunk,
                fc2_chunk=fc2_chunk,
                use_route_order_output=use_route_order_output,
            )

        return self._store_small_row_chunk_output(
            workspace=workspace,
            flat_topk_weights=flat_topk_weights,
            chunk=chunk,
            fc2_chunk=fc2_chunk,
            use_route_order_output=use_route_order_output,
        )


__all__ = ["MoEDynamicKernelBackend"]
