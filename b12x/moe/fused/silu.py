"""SwiGLU wrappers for the activation-specialized fused MoE backends."""

from __future__ import annotations

from typing import Tuple

from b12x.moe.fused.dynamic import MoEDynamicKernelBackend
from b12x.moe.fused.micro import MoEMicroKernelBackend


class MoEMicroKernelSilu(MoEMicroKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        fast_math: bool = False,
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        single_token: bool = False,
        dynamic_down_scale: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            fast_math=fast_math,
            activation="silu",
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=single_token,
            dynamic_down_scale=dynamic_down_scale,
        )

    @classmethod
    def is_supported(
        cls,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
    ) -> bool:
        return super().is_supported(m, k, n, num_topk, weight_E)


class MoEDynamicKernelSilu(MoEDynamicKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        fast_math: bool = False,
        dynamic_down_scale: bool = False,
        share_input_across_experts: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            fast_math=fast_math,
            activation="silu",
            dynamic_down_scale=dynamic_down_scale,
            share_input_across_experts=share_input_across_experts,
        )


__all__ = [
    "MoEDynamicKernelSilu",
    "MoEMicroKernelSilu",
]
