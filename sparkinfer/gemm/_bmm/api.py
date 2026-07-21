"""Dtype dispatch for :func:`sparkinfer.gemm.bmm`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Optional

import torch

from ..._lib.gating import default_is_supported
from . import META
from . import _mxfp8

_MXFP8_SPECIALIZATION = (
    "bfloat16",
    "float8_e4m3fn",
    "float8_e8m0fnu",
    "bfloat16",
    32,
)


def _require_specialization(
    *,
    a_dtype: str,
    b_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
) -> None:
    requested = (a_dtype, b_dtype, sf_dtype, c_dtype, sf_vec_size)
    if requested != _MXFP8_SPECIALIZATION:
        raise NotImplementedError(
            "gemm.bmm supports only "
            "a_dtype='bfloat16', b_dtype='float8_e4m3fn', "
            "sf_dtype='float8_e8m0fnu', c_dtype='bfloat16', "
            f"sf_vec_size=32; got {requested!r}"
        )


def bmm(
    lhs: torch.Tensor,
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    *,
    a_dtype: str,
    b_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    b_major: Literal["k", "n"] = "k",
    sf_axis: Literal["k", "n"] = "k",
    stream: Optional[object] = None,
) -> torch.Tensor:
    """Compute a batch of ``C = A @ B`` into caller-owned ``out``.

    ``lhs`` and ``out`` have logical shapes ``[B,M,K]`` and ``[B,M,N]``.
    The MXFP8 specialization represents ``rhs`` as ``(values, scales)``.
    Values use ``torch.float8_e4m3fn``; scales contain E8M0 encodings as either
    ``torch.uint8`` or ``torch.float8_e8m0fnu``.  ``b_major='k'`` stores those
    tensors as ``[B,N,K]`` and ``[B,N,K/32]``; ``b_major='n'`` stores them as
    ``[B,K,N]`` and ``[B,K,N/32]``.  This backend requires
    ``sf_axis == b_major``.
    """
    _require_specialization(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
    )
    return _mxfp8.mm(
        lhs,
        rhs,
        out,
        b_major=b_major,
        sf_axis=sf_axis,
        stream=stream,
    )


def prewarm_bmm(
    rhs: tuple[torch.Tensor, torch.Tensor],
    m_values: Iterable[int],
    *,
    a_dtype: str,
    b_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    b_major: Literal["k", "n"] = "k",
    sf_axis: Literal["k", "n"] = "k",
    stream: Optional[object] = None,
    synchronize: bool = True,
) -> int:
    """Compile and first-launch every caller-declared graph-visible M."""
    _require_specialization(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
    )
    return _mxfp8.prewarm(
        rhs,
        m_values,
        b_major=b_major,
        sf_axis=sf_axis,
        stream=stream,
        synchronize=synchronize,
    )


def can_implement_bmm(
    *,
    batch: int,
    max_m: int,
    n: int,
    k: int,
    a_dtype: str,
    b_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    b_major: Literal["k", "n"] = "k",
    sf_axis: Literal["k", "n"] = "k",
    device=None,
) -> bool:
    """Return whether the qualified specialization covers a plan."""
    requested = (a_dtype, b_dtype, sf_dtype, c_dtype, sf_vec_size)
    return (
        requested == _MXFP8_SPECIALIZATION
        and is_bmm_supported(device)
        and _mxfp8.can_implement(
            batch=batch,
            max_m=max_m,
            n=n,
            k=k,
            b_major=b_major,
            sf_axis=sf_axis,
        )
    )


def is_bmm_supported(device=None) -> bool:
    """True when an SM120/SM121 target and CUTLASS DSL are available."""
    return default_is_supported(device, requires=META.requires)


def clear_bmm_caches() -> None:
    _mxfp8.clear_caches()


__all__ = [
    "bmm",
    "can_implement_bmm",
    "is_bmm_supported",
    "prewarm_bmm",
]
