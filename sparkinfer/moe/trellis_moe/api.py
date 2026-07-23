"""Public surface for :mod:`sparkinfer.moe.trellis_moe`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from . import META
from ._impl import (
    TrellisMoEBinding as Binding,
)
from ._impl import (
    TrellisMoECaps as Caps,
)
from ._impl import (
    TrellisMoEPlan as Plan,
)
from ._impl import (
    TrellisMoEWeights as Weights,
)
from ._impl import (
    bind_trellis_moe as bind,
)
from ._impl import (
    clear_trellis_moe_caches as clear_caches,
)
from ._impl import (
    plan_trellis_moe as plan,
)
from ._impl import (
    prepare_trellis_moe_weights as prepare_weights,
)
from ._impl import (
    run_trellis_moe as run,
)


def is_supported(device=None) -> bool:
    """Return whether ``device`` supports the SM12x Trellis kernel stack."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Binding",
    "Caps",
    "Plan",
    "Weights",
    "bind",
    "clear_caches",
    "is_supported",
    "plan",
    "prepare_weights",
    "run",
]
