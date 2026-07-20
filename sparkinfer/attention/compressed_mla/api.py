"""Public surface for attention.compressed_mla (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from .._shared.mla.api import (
    clear_mla_caches as clear_caches,
)
from .._shared.mla.compressed_api import (
    compressed_mla_decode_forward as run,
)
from .._shared.mla.compressed_api import (
    compressed_mla_split_chunks_for_contract as split_chunks_for_contract,
)
from ._scratch import (
    SPARKINFERCompressedMLABinding as Binding,
)
from ._scratch import (
    SPARKINFERCompressedMLAScratch as Scratch,
)
from ._scratch import (
    SPARKINFERCompressedMLAScratchCaps as Caps,
)
from ._scratch import (
    SPARKINFERCompressedMLAScratchPlan as Plan,
)
from ._scratch import (
    plan_compressed_mla_scratch as plan,
)
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "Scratch",
    "plan",
    "bind",
    "run",
    "split_chunks_for_contract",
    "is_supported",
    "clear_caches",
]
