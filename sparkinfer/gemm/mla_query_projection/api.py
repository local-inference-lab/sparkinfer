"""Public API for :mod:`sparkinfer.gemm.mla_query_projection`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Optional, TypeAlias

import torch

from ..._lib.gating import default_is_supported
from .._shared import mxfp8_bmm as _mxfp8
from . import META
from . import _bf16

Mxfp8Weight: TypeAlias = tuple[torch.Tensor, torch.Tensor]
MlaQueryWeight: TypeAlias = torch.Tensor | Mxfp8Weight


def run(
    q_nope: torch.Tensor,
    weight: MlaQueryWeight,
    q_pe: torch.Tensor,
    out: torch.Tensor,
    *,
    q_scale: Optional[torch.Tensor] = None,
    stream: Optional[object] = None,
) -> torch.Tensor:
    """Assemble an MLA query into caller-owned ``out``.

    ``q_nope`` is head-major ``[H,M,192]``; ``weight`` is either a BF16
    ``[H,192,512]`` tensor or the N-major native MXFP8 ``W_UK_T`` pack.
    ``q_pe`` and ``out`` are token-major
    ``[M,H,64]`` and ``[M,H,576]``.  ``out`` may be BF16 or E4M3.
    ``q_scale`` is required only for E4M3 output.
    """
    if isinstance(weight, torch.Tensor):
        return _bf16.run(
            q_nope,
            weight,
            q_pe,
            out,
            q_scale=q_scale,
            stream=stream,
        )
    return _mxfp8.mla_query_projection(
        q_nope,
        weight,
        q_pe,
        out,
        q_scale=q_scale,
        b_major="n",
        sf_axis="n",
        stream=stream,
    )


def prewarm(
    weight: MlaQueryWeight,
    m_values: Iterable[int],
    *,
    output_dtype: torch.dtype,
    stream: Optional[object] = None,
    synchronize: bool = True,
) -> int:
    """Compile and first-launch each caller-declared graph-visible ``M``."""
    if isinstance(weight, torch.Tensor):
        return _bf16.prewarm(
            weight,
            m_values,
            output_dtype=output_dtype,
            stream=stream,
            synchronize=synchronize,
        )
    return _mxfp8.prewarm_mla_query_projection(
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
    weight_format: Literal["bf16", "mxfp8"] = "mxfp8",
    device=None,
) -> bool:
    """Return whether the qualified fused-query specialization covers a plan."""
    if not is_supported(device):
        return False
    if weight_format == "bf16":
        return _bf16.can_implement(
            num_heads=num_heads,
            max_m=max_m,
            nope_dim=nope_dim,
            latent_dim=latent_dim,
            output_dtype=output_dtype,
            device=device,
        )
    if weight_format == "mxfp8":
        return _mxfp8.can_implement_mla_query_projection(
            batch=num_heads,
            max_m=max_m,
            n=latent_dim,
            k=nope_dim,
            output_dtype=output_dtype,
            b_major="n",
            sf_axis="n",
        )
    return False


def is_supported(device=None) -> bool:
    """True when an SM120/SM121 target and CUTLASS DSL are available."""
    return default_is_supported(device, requires=META.requires)


def clear_caches() -> None:
    """Clear compiled fused-query projection specializations."""
    _mxfp8.clear_mla_query_projection_caches()
    _bf16.clear_caches()


__all__ = list(META.entry_points)
