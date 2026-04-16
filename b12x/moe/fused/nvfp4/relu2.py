"""ReLU2 wrappers for the activation-specialized fused MoE backends."""

from __future__ import annotations

from typing import Tuple

from b12x.moe.fused.nvfp4.dynamic import MoEDynamicKernelBackend
from b12x.moe.fused.nvfp4.micro import MoEMicroKernelBackend
from b12x.moe.fused.nvfp4.static import MoEStaticKernelBackend


class MoEMicroKernelRelu2(MoEMicroKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        single_token: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation="relu2",
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=single_token,
        )


class MoEStaticKernelRelu2(MoEStaticKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        exact_mma_m_tiles: bool = False,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            exact_mma_m_tiles=exact_mma_m_tiles,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation="relu2",
        )


class MoEDynamicKernelRelu2(MoEDynamicKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation="relu2",
        )


__all__ = [
    "MoEDynamicKernelRelu2",
    "MoEMicroKernelRelu2",
    "MoEStaticKernelRelu2",
]
