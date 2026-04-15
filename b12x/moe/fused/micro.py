"""SwiGLU micro-kernel wrapper over the shared activation-specialized body."""

from __future__ import annotations

from typing import Tuple

from b12x.moe.fused.micro_relu2 import MoEMicroKernel as _MoEMicroKernelShared


class MoEMicroKernel(_MoEMicroKernelShared):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation="silu",
        )


__all__ = ["MoEMicroKernel"]
