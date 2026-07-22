"""Public surface for attention.paged (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._forward import (
    clear_paged_caches as clear_caches,
)
from ._forward import (
    paged_attention_forward as run,
)
from ._scratch import (
    SPARKINFERPagedAttentionBinding as Binding,
)
from ._scratch import (
    SPARKINFERPagedAttentionScratchCaps as Caps,
)
from ._scratch import (
    SPARKINFERPagedAttentionScratchPlan as Plan,
)
from ._scratch import (
    SPARKINFERPagedDecodeGraphScratchEnvelope as DecodeGraphScratchEnvelope,
)
from ._scratch import (
    plan_decode_graph_scratch_envelope as decode_graph_scratch_envelope,
)
from ._scratch import (
    plan_paged_attention_scratch as plan,
)
from .planner import (
    PagedDecodeGraphCapacity as DecodeGraphCapacity,
)
from .planner import (
    PagedVerifyGraphCapacity as VerifyGraphCapacity,
)
from .planner import (
    PagedPlanBudget as Budget,
)
from .planner import (
    infer_paged_mode as infer_mode,
)
from .planner import (
    plan_decode_graph_capacity as decode_graph_capacity,
)
from .planner import (
    plan_verify_graph_capacity as verify_graph_capacity,
)
from .workspace import (
    PagedAttentionWorkspace as Workspace,
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
    "Workspace",
    "Budget",
    "DecodeGraphCapacity",
    "VerifyGraphCapacity",
    "DecodeGraphScratchEnvelope",
    "decode_graph_capacity",
    "verify_graph_capacity",
    "decode_graph_scratch_envelope",
    "plan",
    "bind",
    "run",
    "infer_mode",
    "is_supported",
    "clear_caches",
]
