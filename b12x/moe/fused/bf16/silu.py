"""SwiGLU wrappers for the activation-specialized fused MoE backends."""

from __future__ import annotations

from typing import Tuple

from b12x.moe.fused.bf16.dynamic import MoEDynamicKernelBackend
from b12x.moe.fused.bf16.micro import MoEMicroKernelBackend
from b12x.moe.fused.bf16.static import MoEStaticKernelBackend


class MoEMicroKernelSilu(MoEMicroKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        input_scales_are_reciprocal: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            activation="silu",
        )


class MoEStaticKernelSilu(MoEStaticKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        exact_mma_m_tiles: bool = False,
        input_scales_are_reciprocal: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            exact_mma_m_tiles=exact_mma_m_tiles,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            activation="silu",
        )


class MoEDynamicKernelSilu(MoEDynamicKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        input_scales_are_reciprocal: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            activation="silu",
        )


__all__ = [
    "MoEDynamicKernelSilu",
    "MoEMicroKernelSilu",
    "MoEStaticKernelSilu",
]
