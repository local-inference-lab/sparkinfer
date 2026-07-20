"""Public paged-attention integration surface for the primary backend."""

from __future__ import annotations

from sparkinfer.attention.paged.api import clear_paged_caches, paged_attention_forward
from sparkinfer.attention.paged.planner import (
    create_paged_plan,
    infer_paged_mode,
)
from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace
from sparkinfer.integration.paged_attention_scratch import (
    SPARKINFERPagedAttentionBinding,
    SPARKINFERPagedAttentionScratchCaps,
    SPARKINFERPagedAttentionScratchPlan,
    plan_paged_attention_scratch,
)


def clear_attention_caches() -> None:
    clear_paged_caches()


__all__ = [
    "SPARKINFERPagedAttentionBinding",
    "SPARKINFERPagedAttentionScratchCaps",
    "SPARKINFERPagedAttentionScratchPlan",
    "PagedAttentionWorkspace",
    "clear_attention_caches",
    "create_paged_plan",
    "infer_paged_mode",
    "paged_attention_forward",
    "plan_paged_attention_scratch",
]
