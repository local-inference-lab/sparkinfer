"""ReLU2 dynamic-kernel wrapper over the shared activation-specialized body."""

from __future__ import annotations

from typing import Tuple

from b12x.moe.fused.dynamic import MoEDynamicKernel as _MoEDynamicKernelShared


class MoEDynamicKernel(_MoEDynamicKernelShared):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        activation: str = "relu2",
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
        )


__all__ = ["MoEDynamicKernel"]
