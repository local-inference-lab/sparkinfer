"""Per-request KV cache management and batch tensor assembly.

Tracks which pages belong to which request, handles extend/decode
allocation, builds the page_table and cache_seqlens tensors that
b12x_paged_attention_forward expects.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from b12x.profiling import record_function

from serve.cache.page_pool import PagePool, _PAGE_SIZE


@dataclass
class RequestKVState:
    """KV cache bookkeeping for one live request."""

    request_id: int
    page_ids: list[int] = field(default_factory=list)
    cache_len: int = 0

    @property
    def num_pages(self) -> int:
        return len(self.page_ids)

    def pages_needed(self, new_tokens: int) -> int:
        """How many new pages are required to hold *new_tokens* more tokens."""
        current_capacity = self.num_pages * _PAGE_SIZE
        total_needed = self.cache_len + new_tokens
        if total_needed <= current_capacity:
            return 0
        new_capacity_needed = total_needed - current_capacity
        return (new_capacity_needed + _PAGE_SIZE - 1) // _PAGE_SIZE


class KVCacheManager:
    """Manages per-request page allocations and builds batch tensors.

    Requests are tracked in insertion order for LRU eviction.
    """

    def __init__(self, pool: PagePool):
        self.pool = pool
        # OrderedDict preserves insertion order for LRU eviction.
        self._requests: OrderedDict[int, RequestKVState] = OrderedDict()

    # -- request lifecycle -------------------------------------------------

    def allocate_request(self, request_id: int) -> RequestKVState:
        """Register a new request. No pages allocated yet."""
        if request_id in self._requests:
            raise ValueError(f"request {request_id} already active")
        state = RequestKVState(request_id=request_id)
        self._requests[request_id] = state
        return state

    def extend_request(self, request_id: int, new_tokens: int) -> None:
        """Allocate pages for *new_tokens* additional KV entries."""
        state = self._requests[request_id]
        pages_needed = state.pages_needed(new_tokens)
        if pages_needed > 0:
            new_pages = self.pool.alloc(pages_needed)
            state.page_ids.extend(new_pages)
        state.cache_len += new_tokens
        # Touch for LRU.
        self._requests.move_to_end(request_id)

    def free_request(self, request_id: int) -> None:
        """Release all pages held by a request."""
        state = self._requests.pop(request_id)
        if state.page_ids:
            self.pool.free(state.page_ids)

    def get_state(self, request_id: int) -> RequestKVState:
        return self._requests[request_id]

    @property
    def active_request_ids(self) -> list[int]:
        return list(self._requests.keys())

    @property
    def num_active(self) -> int:
        return len(self._requests)

    # -- eviction ----------------------------------------------------------

    def evict_lru(self) -> int:
        """Evict the least-recently-used request. Returns its request_id."""
        if not self._requests:
            raise RuntimeError("no requests to evict")
        # First item in OrderedDict is the oldest.
        oldest_id = next(iter(self._requests))
        self.free_request(oldest_id)
        return oldest_id

    def try_alloc_or_evict(self, pages_needed: int) -> list[int]:
        """Evict requests until *pages_needed* can be satisfied, then alloc.

        Returns the list of evicted request_ids (empty if none were evicted).
        """
        evicted: list[int] = []
        while self.pool.num_free < pages_needed and self._requests:
            evicted.append(self.evict_lru())
        return evicted

    # -- batch tensor assembly ---------------------------------------------

    def build_page_table(
        self,
        request_ids: list[int],
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        """Build ``[batch, max_pages]`` page table for the given requests."""
        with record_function("kv.build_page_table"):
            if not request_ids:
                return torch.zeros((0, 1), dtype=torch.int32, device=device)
            states = [self._requests[rid] for rid in request_ids]
            max_pages = max(s.num_pages for s in states)
            max_pages = max(max_pages, 1)
            table = torch.zeros(
                (len(request_ids), max_pages), dtype=torch.int32, device=device
            )
            for i, state in enumerate(states):
                if state.page_ids:
                    with record_function("kv.page_ids_to_device"):
                        table[i, : len(state.page_ids)] = torch.tensor(
                            state.page_ids, dtype=torch.int32, device=device
                        )
            return table

    def build_cache_seqlens(
        self,
        request_ids: list[int],
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        """Build ``[batch]`` cache sequence lengths."""
        with record_function("kv.build_cache_seqlens"):
            lens = [self._requests[rid].cache_len for rid in request_ids]
            return torch.tensor(lens, dtype=torch.int32, device=device)

    def build_cu_seqlens_q(
        self,
        q_seqlens: list[int],
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        """Build ``[batch + 1]`` cumulative Q sequence lengths."""
        with record_function("kv.build_cu_seqlens_q"):
            cu = [0]
            for s in q_seqlens:
                cu.append(cu[-1] + s)
            return torch.tensor(cu, dtype=torch.int32, device=device)
