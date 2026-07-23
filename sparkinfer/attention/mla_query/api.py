"""Public API for :mod:`sparkinfer.attention.mla_query`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch

from ..._lib.gating import default_is_supported
from ...gemm._bmm import _mxfp8
from . import META


def run(
    q_nope: torch.Tensor,
    weight: tuple[torch.Tensor, torch.Tensor],
    q_pe: torch.Tensor,
    q_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    stream: Optional[object] = None,
) -> torch.Tensor:
    """Assemble an MLA query into caller-owned ``out``.

    ``q_nope`` is head-major ``[H,M,192]``; ``weight`` is the N-major native
    MXFP8 ``W_UK_T`` pack; ``q_pe`` and ``out`` are token-major
    ``[M,H,64]`` and ``[M,H,576]``.  ``out`` may be BF16 or E4M3.
    """
    return _mxfp8.mla_query(
        q_nope,
        weight,
        q_pe,
        q_scale,
        out,
        b_major="n",
        sf_axis="n",
        stream=stream,
    )


def prewarm(
    weight: tuple[torch.Tensor, torch.Tensor],
    m_values: Iterable[int],
    *,
    output_dtype: torch.dtype,
    stream: Optional[object] = None,
    synchronize: bool = True,
) -> int:
    """Compile and first-launch each caller-declared graph-visible ``M``."""
    return _mxfp8.prewarm_mla_query(
        weight,
        m_values,
        output_dtype=output_dtype,
        b_major="n",
        sf_axis="n",
        stream=stream,
        synchronize=synchronize,
    )


def can_implement(
    *,
    num_heads: int,
    max_m: int,
    nope_dim: int,
    latent_dim: int,
    output_dtype: torch.dtype,
    device=None,
) -> bool:
    """Return whether the qualified fused-query specialization covers a plan."""
    return is_supported(device) and _mxfp8.can_implement_mla_query(
        batch=num_heads,
        max_m=max_m,
        n=latent_dim,
        k=nope_dim,
        output_dtype=output_dtype,
        b_major="n",
        sf_axis="n",
    )


def is_supported(device=None) -> bool:
    """True when an SM120/SM121 target and CUTLASS DSL are available."""
    return default_is_supported(device, requires=META.requires)


def clear_caches() -> None:
    _mxfp8.clear_mla_query_caches()


__all__ = list(META.entry_points)
