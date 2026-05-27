"""Public paged-attention integration surface for the primary backend."""

from __future__ import annotations

from b12x.attention.paged.api import clear_paged_caches, paged_attention_forward
from b12x.attention.paged.planner import (
    create_paged_plan,
    infer_paged_mode,
)
from b12x.attention.paged.workspace import (
    PagedAttentionArena,
    PagedAttentionArenaCaps,
    PagedAttentionWorkspace,
    PagedAttentionWorkspaceContract,
)
from b12x.integration.paged_attention_scratch import (
    B12XPagedAttentionBinding,
    B12XPagedAttentionScratchCaps,
    B12XPagedAttentionScratchPlan,
    plan_paged_attention_scratch,
)


def clear_attention_caches() -> None:
    clear_paged_caches()


__all__ = [
    "B12XPagedAttentionBinding",
    "B12XPagedAttentionScratchCaps",
    "B12XPagedAttentionScratchPlan",
    "PagedAttentionArena",
    "PagedAttentionArenaCaps",
    "PagedAttentionWorkspace",
    "PagedAttentionWorkspaceContract",
    "clear_attention_caches",
    "create_paged_plan",
    "infer_paged_mode",
    "paged_attention_forward",
    "plan_paged_attention_scratch",
]
