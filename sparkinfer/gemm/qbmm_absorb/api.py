"""Public surface for gemm.qbmm_absorb (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._kernel import (
    QbmmAbsorbKernel,
    qbmm_absorb,
    qbmm_absorb_supported,
    qbmm_absorb_ukt,
    qbmm_absorb_uv,
    warmup_qbmm_absorb,
)
from . import META

# The default geometry every launch entry point assumes (GLM-5.2 TP4 MLA);
# ``qbmm_absorb_supported(**other_geometry)`` gates non-default layouts.
_DEFAULT_GEOMETRY = dict(
    num_heads=16, head_stride=448, p_dim=192, v_dim=256, latent_dim=512
)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and the default
    MLA geometry inside the v1 kernel envelope."""
    return default_is_supported(device, requires=META.requires) and bool(
        qbmm_absorb_supported(**_DEFAULT_GEOMETRY)
    )


__all__ = [
    "QbmmAbsorbKernel",
    "qbmm_absorb",
    "qbmm_absorb_supported",
    "qbmm_absorb_ukt",
    "qbmm_absorb_uv",
    "warmup_qbmm_absorb",
    "is_supported",
]
