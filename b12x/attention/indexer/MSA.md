# MiniMax Sparse Attention Index Contract

This document describes the tensor boundary between the MSA index branch and
the paged block-sparse attention backend.

## Selection Tensor

`q2k_indices` is a CUDA `torch.int32` tensor with shape
`[num_kv_heads, total_q_capacity, 16]`. It is contiguous and owned by the
caller or indexer scratch. The attention backend captures the tensor pointer in
CUDA graph mode, so serving code must keep the same allocation and rewrite the
contents in place before the attention segment.

Entries are batch-local 128-token block ids. At page_size=64, block `b` maps
to logical pages `2*b` and `2*b + 1` in that request's page table. At
page_size=128 (the vLLM contract), the page table is indexed by block id
directly: one table entry per 128-token block. Each row is sorted ascending,
has no duplicates, includes the local causal block, and pads unused tail
slots with `-1`. **Ascending order with the local block in the last valid
slot is load-bearing**: the kernel reads the last valid entry to size the
tail of its walk.

The attention kernel derives the live block count from `cache_seqlens`, not
from the tensor contents. It still guards negative ids defensively, so `-1`
tails are ignored. Other out-of-range ids are invalid input.

## Main KV cache (page_size=128, vLLM)

The attention backend accepts `page_size ∈ {64, 128}` for MSA plans, BF16 or
FP8 e4m3 K/V (matching dtypes; FP8 requires `k_descale`/`v_descale`,
per-request or per-(request, kv-head)). At page_size=128 the K/V cache views
may be strided slices of a combined `[num_blocks, 2, 128, kv_heads, head_dim]`
cache (`k = cache[:, 0]`, `v = cache[:, 1]`). Constraints on each view
`[num_pages, 128, kv_heads, head_dim]`:

1. `stride(3) == 1` and rows within a page contiguous
   (`stride(1) == kv_heads * head_dim`).
2. `stride(0)` must be a whole, 64-row-aligned number of token rows
   (`stride(0) % stride(1) == 0` and `(stride(0)/stride(1)) % 64 == 0`).
3. K and V must have equal page-stride geometry (automatic for slices of one
   combined cache).

## Decode

For decode, `total_q_capacity >= batch` and row `q` corresponds to request
`q`. The indexer writes all selected rows before the graph-captured attention
kernel runs:

1. Score index queries against the paged index-K cache.
2. Max-pool token scores into 128-token block scores.
3. Select top-16 blocks by raw score, force the local block, sort ascending.
4. Copy the result into the stable `q2k_indices` allocation.
5. Run MSA paged decode attention with the same page table and seqlens.

## Prefill

For extend/prefill, `total_q_capacity >= cu_seqlens_q[-1]`. Row `q` is the
packed query row, and the local block is computed from
`token_local + cache_len - qo_len`.

The union-tile attention path consumes the same `q2k_indices` contract. Its
workspace pre-pass builds per-8-token tile membership metadata from the stable
selection tensor; the source tensor pointer remains graph-safe.

## Index-K Cache

The MSA index-K cache uses the existing paged indexer reference layout:
`uint8[num_pages, 64 * (128 + 4)]`. Each page stores 64 FP8 e4m3 rows followed
by one FP32 scale per row. Page ids are physical cache page ids, and the request
page table supplies the logical-to-physical mapping.

The current attention backend only consumes `q2k_indices`. Production MSA
indexer kernels should produce exactly this tensor contract.
