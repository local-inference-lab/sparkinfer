"""W4A16 dense GEMM micro kernel for SM120 / SM121.

Two-tier dispatch:

* **Decode** (M < ``B12X_GEMM_W4A16_PREFILL_M``, default 256):
  v4 forked CuTe-DSL kernel (``_cute_dense_kernel.py``), warp-level
  bf16 MMA, tile (32, 64, 64), 4 MMA warps.  Sweet spot for the
  Nano3.5 decode linears (M ≤ 32 originally; v4 still runs correctly
  at any M).
* **Prefill** (M ≥ ``B12X_GEMM_W4A16_PREFILL_M``):
  v6 Marlin-style cp.async + register-pipelined kernel
  (``_cute_marlin_kernel.py``), MoE-stripped to dense.  Includes
  tile-keyed ``num_stages`` autotune and in-kernel N-padding so the
  wider tile_n=128 geometry works even when N doesn't divide 128
  (e.g. mamba_in_proj N=10304 → kernel sees N=10368, C-write path
  skips the padded tail).  Beats Marlin on 4/5 traced Nano3.5 shapes
  at M=2048; ~25-44% faster than v5/prefill on all v6-eligible shapes.

All backends share weight layout and accuracy gates.

Set ``B12X_GEMM_W4A16_FORCE_REFERENCE=1`` to fall back to the Python
reference (useful for accuracy debugging — runs on CPU).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from ._cute_dense_kernel import DenseGemmW4A16CuteDenseKernel
from ._cute_marlin_kernel import DenseGemmW4A16CuteMarlinKernel
from .reference import dense_reference_w4a16


# Crossover threshold empirically: v4 wins at M ≤ 128 (warp-level 32×64
# tile keeps work granular), v6 takes over at M ≥ 256 (Marlin-style
# cp.async pipeline amortizes per-CTA overhead).  At M=128 they're
# roughly equal — default to v6 from 256 onward to be safe.
_DEFAULT_PREFILL_M = int(os.environ.get("B12X_GEMM_W4A16_PREFILL_M", "256"))


def _use_prefill(m: int) -> bool:
    return m >= _DEFAULT_PREFILL_M


class DenseGemmW4A16MicroKernel:
    """W4A16 dense GEMM kernel with decode + prefill dispatch."""

    @classmethod
    def is_supported(cls, m: int, k: int, n: int) -> bool:
        if _use_prefill(m):
            return DenseGemmW4A16CuteMarlinKernel.is_supported(m, k, n)
        return DenseGemmW4A16CuteDenseKernel.is_supported(m, k, n)

    def __init__(self) -> None:
        self._decode: DenseGemmW4A16CuteDenseKernel | None = None
        # v6 is shape-agnostic at construction time: it picks cta_m_size
        # + tile_n + tile_k + num_stages per call from M/N/K.
        self._prefill: DenseGemmW4A16CuteMarlinKernel | None = None

    def __call__(
        self,
        x: torch.Tensor,
        w_fp4: torch.Tensor,
        w_blockscale: torch.Tensor,
        w_alpha: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m, k = x.shape
        n = w_fp4.shape[0]
        if out is None:
            out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)

        if not x.is_cuda:
            out_cpu = dense_reference_w4a16(
                x, w_fp4=w_fp4, w_blockscale=w_blockscale, w_alpha=w_alpha,
            )
            out.copy_(out_cpu)
            return out

        if _use_prefill(m):
            if self._prefill is None:
                self._prefill = DenseGemmW4A16CuteMarlinKernel()
            return self._prefill(
                x.contiguous(), w_fp4.contiguous(), w_blockscale.contiguous(),
                w_alpha.contiguous(), out=out,
            )
        if self._decode is None:
            self._decode = DenseGemmW4A16CuteDenseKernel()
        return self._decode(
            x.contiguous(), w_fp4.contiguous(), w_blockscale.contiguous(),
            w_alpha.contiguous(), out=out,
        )


_KERNEL_CACHE: Optional[DenseGemmW4A16MicroKernel] = None


def _get_cached_kernel() -> DenseGemmW4A16MicroKernel:
    global _KERNEL_CACHE
    if _KERNEL_CACHE is None:
        _KERNEL_CACHE = DenseGemmW4A16MicroKernel()
    return _KERNEL_CACHE


def dense_gemm_w4a16(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale: torch.Tensor,
    w_alpha: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Public entry point — W4A16 dense GEMM, decode + prefill backends.

    Dispatch picks the v6 Marlin-style prefill kernel when M ≥
    ``B12X_GEMM_W4A16_PREFILL_M`` (default 256), otherwise the v4 decode
    kernel.  ``w_blockscale`` must be the swizzled FP8 e4m3 tensor (as
    produced by ``quantize_dense_weight_to_fp4``).
    """
    if os.environ.get("B12X_GEMM_W4A16_FORCE_REFERENCE") == "1":
        result = dense_reference_w4a16(
            x.detach().cpu(),
            w_fp4=w_fp4.detach().cpu(),
            w_blockscale=w_blockscale.detach().cpu(),
            w_alpha=w_alpha.detach().cpu(),
        ).to(x.device)
        if out is None:
            return result
        out.copy_(result)
        return out

    m, k = x.shape
    n = w_fp4.shape[0]
    if not DenseGemmW4A16MicroKernel.is_supported(m, k, n):
        raise NotImplementedError(
            f"dense_gemm_w4a16 supports N % 64 == 0 and K % 64 == 0; "
            f"got M={m}, K={k}, N={n}."
        )
    kernel = _get_cached_kernel()
    return kernel(x, w_fp4, w_blockscale, w_alpha, out=out)


__all__ = ["DenseGemmW4A16MicroKernel", "dense_gemm_w4a16"]
