"""MiMo-V2.5 FP8 decode graph policy for qk=192, v=128, GQA=16."""

from __future__ import annotations

from .registry import register_decode_graph_policy


def _ladder(max_chunks_per_request: int) -> tuple[tuple[int, int], ...]:
    chunk_pages = 1
    ladder: list[tuple[int, int]] = []
    while max_chunks_per_request * chunk_pages < 4096:
        ladder.append((max_chunks_per_request * chunk_pages, chunk_pages))
        chunk_pages *= 2
    ladder.append((4096, 4096 // max_chunks_per_request))
    return tuple(ladder)


_MAX_CHUNKS_PER_REQUEST = {
    1: 32,
    2: 32,
    4: 16,
    8: 8,
    16: 4,
    32: 4,
    64: 4,
    128: 4,
}

for _batch, _max_chunks_per_request in _MAX_CHUNKS_PER_REQUEST.items():
    register_decode_graph_policy(
        kv_dtype="fp8",
        regime="decode_qk192_vo128_gqa16",
        batch=_batch,
        graph_ctas_per_sm=2,
        capture_fixed_split_pages=4096 // _max_chunks_per_request,
        capture_page_count=4096,
        page_size=64,
        chunk_ladder=_ladder(_max_chunks_per_request),
    )
