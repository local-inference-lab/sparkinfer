"""Public paged-attention integration surface for the primary backend."""

from __future__ import annotations

from b12x.attention.paged.api import clear_paged_caches, paged_attention_forward
from b12x.attention.paged.planner import (
    create_paged_plan,
    infer_paged_mode as infer_paged_attention_mode,
)
from b12x.attention.paged.workspace import (
    PagedAttentionArena,
    PagedAttentionArenaCaps,
    PagedAttentionWorkspace,
    PagedAttentionWorkspaceContract,
)


def clear_attention_caches() -> None:
    clear_paged_caches()


__all__ = [
    "PagedAttentionArena",
    "PagedAttentionArenaCaps",
    "PagedAttentionWorkspace",
    "PagedAttentionWorkspaceContract",
    "clear_attention_caches",
    "create_paged_plan",
    "infer_paged_attention_mode",
    "paged_attention_forward",
]
