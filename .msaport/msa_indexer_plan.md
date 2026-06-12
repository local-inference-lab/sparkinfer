# MSA Indexer — Phased Implementation Plan

Track 1 of 2 for MiniMax-M3 sparse attention (MSA) support. Companion: `.msaport/msa_attention_plan.md` (paged GQA sparse block-list attention). Algorithm reference: `~/projects/MSA` (paper `docs/MiniMaxSparseAttention.pdf` + SM100 kernels).

## Context

MSA's Index Branch: one bilinear index q-head per GQA group (4 groups; 1/GPU at TP=4) + ONE shared index k-head, d_idx=128, RMSNorm+RoPE caller-side. Token score = `q_idx·k_idx/√128` — **no ReLU, no per-head weights, no head-sum** (the degenerate point of the NSA/DSA lightning indexer this repo already implements). Scores column-max-pool over 128-token KV blocks (causal, empty → -inf), then per (query token, kv head): top-k=16 blocks, local block always forced, exp-free (raw-score ranking), output ascending-sorted with -1 padding.

Strategy: **specialize the existing NSA indexer kernels via const_expr**, add a block-max epilogue, do v1 top-k in torch (block scores are tiny: `[h, Q, ≤8K]`; `tiled_topk` only supports k∈{512,1024,2048}).

## Grounding (verified in code)

- The only ReLU sites in scorer hot loops: `kernel.py:557-560` (`_compute_mxfp8_tile_partials`, shared by all paged decode kernels — plain :844, scheduled single-row :1227, scheduled multi-row :1520) and `contiguous_kernel.py:1096` (BK=64), `:1750` (BK=256 prefill), `:2189` (BK=512).
- Head-sum is an epilogue concern: decode = butterfly `_reduce_column_pair_sum` (kernel.py:470) + `s_partial_logits` fold (:1240-1253); prefill = `acc_frag += relu(score)*w` across the serialized head loop (contiguous_kernel.py:1739-1755).
- BK=256 prefill tile geometry (contiguous_kernel.py:57-64): 2 Q-warps × 4 K-warps, each K-warp covers 64 K rows (num_mma_kv=4). Fragment mapping (:1797-1808): `q_local = warp_q_idx*16 + lane//4 + 8*row_slot`; `k_local = warp_k_idx*64 + mma_kv*16 + 2*(lane%4) + 8*(reg_id//4) + reg_id%2`. **A 128-token MSA block = 2 adjacent K-warps; one CTA tile covers exactly 2 blocks.**
- Decode dispatch (api.py:139-151, kernel.py:959-987): single-row schedule (q_rows==1 & pages≥1024), multi-row (2–8 rows & pages≥1024), else plain/tiled fallback.
- Index K cache: page = 64 rows × (128B fp8 e4m3 payload + 4B f32 scale); `index_k_cache (num_pages, 64*132) uint8` (reference.py:57-78; quantizer `pack_index_k_cache_reference` :81-115, scale = absmax/e4m3_max, 0→1.0 — **k_scale > 0 always, so scale-then-max is order-safe**).
- `tiled_topk.py:42` `_SUPPORTED_TOPK=(512,1024,2048)`; output NOT sorted; sentinels -inf/-1.
- `fused_indexer.py` is a SCAFFOLD (launch config/dispatch/scratch trio; kernel body not landed).
- Conventions: int-enum const_expr keys (`mla/unified_sm120/traits.py:21-33`); lru-cached kernel builders + `KernelCompileSpec` cache keys (contiguous_kernel.py:2383-2407, :2927-2931); dummy-tensor-for-unused-output (`tile_logits_kernel = torch.empty((1,1))`, :2793); `preinitialize_invalid_logits` host -inf prefill pattern.

## Locked design decisions

- **D1 — Block-score layout: `[num_idx_heads, total_q, num_blocks] f32`, -inf invalid.** Head-major matches the `q2k_indices [num_kv_heads, total_q, 16]` output; per-head torch.topk emits it with zero permutes; degenerates to headless at TP=4 (h=1). Decode page scores: same convention `[h, Q, num_pages]`.
- **D2 — `IndexerScoreMode` int const_expr key** (new, in kernel.py; imported by contiguous_kernel.py): `NSA_RELU_SUM=0`, `MSA_BILINEAR=1`. **In MSA mode the existing `weights (Q,h) f32` slot carries `q_scale * (1/√128)`** — the existing weight-multiply + per-K-row `s_scales` multiply then produce exactly `q·k · q_scale · k_scale / √128`. Only the ReLU is removed; zero new tensor plumbing in the inner loop.
- **D3 — Decode v1 via row expansion**: `q_fp8 (Q,h,128) → (Q*h,1,128)`, page table/seqlens repeated ×h. Zero overhead at the production TP=4 config (h=1); accepted at TP=1 for v1; fixed in Phase A5.
- **D4 — Block ids batch-local.** Decode: page tables are per-row batch-local → `block = token//128` directly. Contiguous prefill: packed-K sequence starts MUST be 128-aligned (assert in metadata builder; prefill pad is 256, a multiple). Kernel emits GLOBAL block ids; selection helper subtracts `block_base = k_start[q]//128`.
- **D5 — Causal/-inf at block granularity**: all masking token-level BEFORE the max (`k_start ≤ j < k_end` prefill; `j < cache_seqlen` decode). Partial local block ⇒ max over j ≤ i only. Fully-future blocks, **padded/zero-filled K rows** (a 0 score beats negative real maxes — must mask), and dead tiles ⇒ -inf via host prefill.
- **D6 — Selection contract**: exclude -inf; force local block (`+inf` scatter at `pos//128`); top-16 raw score; ascending sort with trailing -1 via INT32_MAX-sentinel sort. Fully graph-capturable (no `.item()`, fixed shapes).

---

## Phase A1 — Torch reference + contracts

Files: **create** `b12x/attention/indexer/msa_reference.py`, **create** `tests/test_attention_msa_indexer_reference.py`, modify `b12x/attention/indexer/__init__.py` (exports).

```python
MSA_BLOCK_TOKENS = 128; MSA_TOPK_BLOCKS = 16; MSA_SM_SCALE = 1.0 / math.sqrt(128.0)
def quantize_msa_q_fp8_reference(q) -> (q_fp8 (Q,h,128) e4m3, q_scale (Q,h) f32)   # per-row absmax, 0→1.0
def msa_contiguous_block_scores_reference(*, q_fp8, q_scale, kv_fp8, k_start, k_end) -> (h, Q, ceil(K/128)) f32
def msa_paged_decode_block_scores_reference(*, q_fp8, q_scale, index_k_cache,
        real_page_table, cache_seqlens_int32, page_size=64) -> (h, Q, ceil(width/128)) f32   # block = 2 pages
def msa_select_blocks_reference(*, block_scores, query_positions, block_base=None, topk=16) -> (h, Q, 16) i32
def msa_q2k_indices_reference(...)   # end-to-end composition
```
Reuse `_split_index_k_cache_reference` gather logic (reference.py:140-232) minus relu/w-sum. Keyword-only, f32 math, explicit validation (reference.py conventions).

Tests (CPU-runnable): block-max vs naive token loop; partial local block at lens {1,127,128,129,255,256}; empty-block -inf; <16 valid blocks → trailing -1; forced local when it would rank 17th; ascending property; nonzero block_base; h∈{1,4}; spec-decode-style rows.

## Phase A2 — Host selection helper (block scores → q2k_indices)

Files: modify `b12x/attention/indexer/api.py` (mirror `_reference_topk_indices_from_logits` style, api.py:514), extend the A1 test file.

```python
def msa_topk_blocks(*, block_scores (h,Q,nb) f32, query_positions (Q,) i32,
    block_base: (Q,) i32 | None = None, topk=16, out_indices: (h,Q,16) i32 | None = None) -> Tensor
def msa_decode_query_positions(cache_seqlens_int32) -> Tensor   # seqlens - 1
def msa_prefill_query_positions(cu_seqlens_q, total_q) -> Tensor
```
All batched torch, no sync/alloc-in-loop: scatter +inf at local block (reusable scratch copy), `torch.topk(16)`, global→batch-local, invalidate gathered -inf non-local → INT32_MAX, `sort(-1)`, → -1. Honors `out_indices` copy-out (api.py:538-548 pattern).

Tests: exact-match vs reference on jittered scores (fp8 ties — add `arange*1e-6`); nb>16 and nb<16; all -inf except local; CUDA-graph capture/replay smoke with mutated scores.

## Phase A3 — BILINEAR score mode + decode E2E (token logits → torch pooling)

First kernel touch, deliberately minimal: const_expr switch at the 4 ReLU sites; **no epilogue changes**. Delivers working MSA decode on every existing route + the contiguous token-logit mode A4 cross-checks against.

- `kernel.py`: add `IndexerScoreMode`; `_compute_mxfp8_tile_partials(..., score_mode: cutlass.Constexpr)` — `if const_expr(score_mode == MSA_BILINEAR): partial = q_acc * w` else existing `fmax(q_acc,0)*w`. (Padded zero q-rows × w=0 keep the butterfly head-sum an identity — leave it.) Thread score_mode through `SparseNSAPagedLogitsKernel` / scheduled single-row / multi-row constructors, `_build_sparse_nsa_*` lru keys (kernel.py:1566-1619), launch cache_key tuples. `run_paged_logits_kernel(..., score_mode=NSA_RELU_SUM)`.
- `contiguous_kernel.py`: same const_expr at :1096/:1750/:2189 + builder/cache keys; `run_contiguous_logits_kernel(..., score_mode=...)`.
- `api.py`:
```python
def quantize_msa_q_fp8(q) -> (q_fp8, q_scale)                       # production quantizer
def msa_paged_decode_block_scores(*, q_fp8, q_scale, index_k_cache,
    metadata: IndexerPagedDecodeMetadata, page_size=64, out=None, binding=None) -> (h,Q,nb)
def msa_q2k_indices_decode(*, q_fp8, q_scale, index_k_cache, metadata, topk=16,
    out_indices=None, binding=None) -> (h,Q,16)
```
  v1 internals: `weights = q_scale * MSA_SM_SCALE`; reshape `(Q,h)→(Q*h,1)`; repeat_interleave page table/seqlens (binding-owned buffers when captured); `paged_decode_logits(score_mode=MSA_BILINEAR)` → `(Q*h, width)`; pool `view(Q,h,nb,128).amax(-1).permute(1,0,2)` (width padded to 128 multiple; -inf propagates).
- `__init__.py`: export `msa_*`, `IndexerScoreMode`.

Tests (**create** `tests/test_attention_msa_indexer_api.py`):
- Token-logit parity vs bilinear reference, atol/rtol 1e-4 (`_assert_logits_close` convention, tests/test_attention_nsa_indexer_api.py:84), across all 3 dispatch routes (force via pages≥1024 / q-rows 1, 4, 64) — **negative scores must survive** (the property ReLU destroyed).
- Block scores vs reference; seqlens straddling page and block boundaries {63,64,65,127,128,129,8191,...}; h∈{1,4}.
- E2E `msa_q2k_indices_decode` vs reference (exact, jittered).
- **Full NSA regression** (`tests/test_attention_nsa_indexer_*.py`) unchanged; compile-cache key distinctness via `compile_cache_info`.
- CUDA-graph capture/replay of the whole decode E2E with mutating seqlens (mirror the live-width replay test at tests/.../api.py:520).

Benchmark: **create** `benchmarks/benchmark_msa_indexer.py` (argparse + `benchmarks.common.bench_cuda_graph` + L2 flush, clone `benchmark_nsa_indexer.py`): `--mode decode`, rows {1,4,16,64}, h {1,4}, ctx {8K,32K,128K,256K}; µs + effective K-cache GB/s; NSA `paged_decode_logits` baseline column (expect parity at h=1; ~h× gap at TP=1 until A5).

## Phase A4 — Prefill scorer: BK=256 contiguous kernel + block-max epilogue

Main new kernel work. Token logits never reach gmem; output `[h, Q, nb]`, host-prefilled -inf.

Kernel (`SparseNSAContiguousLogitsPrefillKernel` only — BK=512's head gates {32,64} don't apply, BK=64 tiles can't see a whole block):
1. Constructor `block_score_output=False` const_expr; builder/lru + cache-key variant `"prefill_msa_blockmax"`.
2. Shared storage (param. `get_sparse_nsa_contiguous_prefill_shared_storage_cls`, contiguous_kernel.py:352): add `block_partial: MemRange[Float32, 4*32]` (512B; ample headroom under ~72KB current).
3. `__call__`/`kernel` gain `block_scores: cute.Tensor` (flat f32) + `num_blocks_out: Int32`; NSA mode passes the dummy `(1,1)` tensor (pattern at :2793).
4. Under `const_expr(block_score_output)`, replace the acc_frag accumulate in the head loop (:1707-1755) with a per-head epilogue:
   - Per register (reuse index math at :1797-1808 **verbatim**): `valid = (k_row >= s_k_start[q_local]) & (k_row < s_k_end[q_local]) & (k_row < k_total_rows)`; `val = valid ? score*w_val*s_scales[k_local] : -inf` (w_val = staged `w_rs0/w_rs1` = `q_scale/√128` per D2).
   - Per-thread fmax over its 16 values (4 regs × 4 mma_kv) per row_slot; quad-reduce via new `_reduce_quad_max` (bfly offsets 1,2); lane%4==0 writes `s_block_partial[warp_k_idx, q_local]`; `sync_threads()`.
   - Combine: threads tx<64: `q_local=tx%32, blk=tx//32`; `v = fmax(partial[2blk], partial[2blk+1])`; bounds-checked write `block_scores[head, q_row, k_tile_idx*2+blk]`; `sync_threads()` before next head reuses the buffer.
   - Dead tiles (`s_tile_live==0`, :1827): write nothing (host -inf prefill covers).
5. `run_contiguous_logits_kernel`: when block_score_output force `_prefill_block_k=256` (precedent: tile_logits forcing :2641-2645); require `num_heads ≤ 8` (single head batch); `_pad_kv_rows` pad 256; allocate/validate + `fill_(-inf)`. Prefer a thin `run_contiguous_block_scores_kernel` wrapper sharing staging, keeping the NSA signature stable.

API: `msa_contiguous_block_scores(...)` (GLOBAL block ids), `msa_q2k_indices_prefill(...)`; metadata builder asserts D4 128-alignment.

Tests: vs reference 1e-4 — Q∈{1,32,257,4096} × K∈{128,256,8192,131072}, h∈{1,4}, ragged causal `k_end=k_start+i+1`; multi-sequence packing with 128-but-not-256-aligned starts (CTA tile straddling two sequences); K%256≠0 **pad-leak regression** (D5); **cross-check vs A3 BILINEAR token logits `.amax(...)` — same MMA order ⇒ expect bitwise; fall back 1e-6**; E2E prefill q2k; NSA regression.

Benchmark `--mode prefill`: Q chunks {2048,8192} × K {32K,128K} × h {1,4}; vs NSA dense token-logit output (expect large win: 128× smaller output traffic); optional flag comparing the A3-style tiled-logits+torch-pool bridge to quantify the epilogue.

## Phase A5 — Decode scorer: per-head page-max epilogue in scheduled kernels

Removes D3 reshape (h heads stay in MMA M, one shared K stream) and the `(Q,width)` logits round-trip. **Reuse the M=heads scheduled decode kernels** (single-row + multi-row), not the tiled path: decode is HBM-bound on K pages; M-underutilization (1-4 of 16 rows) costs nothing measurable; the schedule/TMA/persistent machinery is what long-context decode needs. Plain (pages<1024)/tiled routes keep the A3 path.

Kernel (`kernel.py`):
1. `output_mode` const_expr (TOKEN_LOGITS=0 / PAGE_HEAD_MAX=1) on both scheduled kernels; output arg `page_scores` flat f32 `[h, Q, max_pages]` (dummy in NSA mode); builder/cache keys.
2. New `_compute_mxfp8_tile_head_token_max` beside `_compute_mxfp8_tile_partials` (kernel.py:478): identical MMA body; epilogue keeps per-(head,token) values (thread holds heads `g=lane//4`, `g+8` for token cols col0/col1), applies `val = q_acc * s_w[head] * s_scale[token]`, masks `token < valid_slots` else -inf, in-thread fmax(col0,col1), bfly offsets 1,2 over the token quad → per-(head, 8-token-group) max in lane%4==0.
3. Page-loop epilogue (:1212-1255): re-shape `s_partial_logits` as `[4 token-groups, 16 head slots]`, init -inf **once per page** (replacing per-split zeroing :1219-1223), single-writer fmax-fold per split; after the split loop threads tx<16 reduce the 4 groups and (tx<num_heads) write `page_scores[tx, q_idx, page_col]`. Host -inf prefill; pages ≥ live_pages never visited.
4. **Per-PAGE, not per-block**: the schedule strides single pages across `parallel_ctas` (:1174, :1260), so a block's 2 pages generally land in different CTAs — per-page output keeps it write-race-free; host pairs `view(h,Q,-1,2).amax(-1)` (pad page capacity even). In-kernel pairing deferred to A7 (fused kernel re-aligns CTA stride to page pairs).
5. `run_paged_logits_kernel` plumbing; `msa_paged_decode_block_scores` routes here when scheduled routes apply; env escape `B12X_MSA_DECODE_PAGEMAX=0`.

Tests: parity vs A3 path ≤1e-6 (ideally bitwise) and vs reference; h∈{1,2,4}; q_rows {1, 2..8}; page/block-boundary seqlens; odd live_pages; graph replay with seqlen growth; NSA regression. Benchmark: A5 vs A3 route at h∈{1,4} (expect ~parity h=1; ~4× at TP=1).

## Phase A6 — Scratch/binding consolidation + graph hardening

- `scratch.py`: `B12XIndexerScratchCaps`/`B12XIndexerPagedScratchCaps` (scratch.py:65/:159) gain `score_mode: str = "nsa"`, `num_idx_heads: int = 1`; MSA layouts (`_indexer_paged_scratch_layout`, :660) reserve `page_scores (h·maxQ·even(max_pages))`, `block_scores (h·maxQ·ceil(width/128))`, `q2k_indices (h·maxQ·16) i32`, topk scratch (+inf-scatter copy), and A3-fallback page_table/seqlen replicas `(maxQ·h)`. Contiguous caps (:223): `block_scores (h·maxQ·ceil(max_k_rows/128))`. MSA = 1 row/token (no C4 compression) — keep the row-capacity doc convention.
- `B12XIndexerMSABinding` + `build_indexer_msa_paged_binding` / `build_indexer_msa_contiguous_binding` mirroring `B12XIndexerPagedBinding` (scratch.py:541); strict mode forbids alloc inside the call (graph guarantee; `strict_binding` pattern, api.py:730-733). `msa_q2k_indices_*` accept `binding=` and enforce "binding owns X" errors (api.py:291-308).
- Tests: zero-allocation under capture (allocator stats); binding-extras errors; `clear_indexer_caches` covers new caches; replay correctness after seqlen mutation. Benchmark: graph-captured E2E decode (score→pool→select) single-replay → `results.msa_indexer.tsv`.

## Phase A7 (later, optional) — Fused decode score→blockmax→top16

On the `fused_indexer.py` scaffold (row-cooperative persistent kernel, KV_LAYOUT_PAGED, scratch trio, `persistent_topk` group-barrier merge):
- Score: reuse A5's `_compute_mxfp8_tile_head_token_max`; CTA streams page **pairs** (block max completes in-CTA — re-align per-CTA K-slice stride to 2 pages).
- Selection: replace the radix running-topk (sized for k∈{512..2048}) with a per-head 16-entry running top-k in registers/smem (insertion vs current min; candidates ~1 block-max per 2 pages, O(1) amortized); cross-CTA merge via existing `_group_barrier` state machinery; final CTA does forced-local + 16-elt bitonic ascending sort + -1 pad, writes `q2k_indices [h,Q,16]` directly. Logits/block scores never reach gmem; per-(q,head) working set = 16 entries. Exp-free raw f32 compares.
- Decode-only (m=heads); gated by row-count crossover like `FUSED_MAX_ROWS`. Tests: exact index parity vs A5; benchmark at 32K–256K ctx, rows 1–6.

## Risk register

1. **A4 fragment-mapping bugs** (highest): mitigated by the A3 bitwise cross-check (token logits and block scores from the same MMA) + verbatim reuse of validated index math.
2. **Padded-K zero rows winning a max** (D5): dedicated A4 regression.
3. **Compile-cache key collisions**: every new const_expr in BOTH the lru builder key and the KernelCompileSpec cache_key; explicit `compile_cache_info` distinctness test.
4. **fp8 score ties** breaking exact-index tests: jittered inputs + recall-style assertion (selected set min ≥ 16th-best − ε) for unjittered cases.
5. **Graph capture**: no `.item()`/alloc in MSA host glue; A6 enforces with strict bindings.

## Verification

Per phase: `pytest tests/test_attention_msa_indexer_*.py -x -q` + NSA regression files. First JIT compile 30s+ is normal; >30s silence on a warm cache = hang (CLAUDE.md rule). Benchmarks graph-captured with L2 flush; results → `results.msa_indexer.tsv`. PTX-identity spot-check with score_mode=NSA after each kernel-touching phase.

## Critical files

`b12x/attention/indexer/{kernel.py, contiguous_kernel.py, api.py, scratch.py, msa_reference.py(new), __init__.py}` · `tests/test_attention_msa_indexer_{reference,api}.py(new)` · `benchmarks/benchmark_msa_indexer.py(new)` · pattern sources: `reference.py`, `b12x/attention/mla/unified_sm120/traits.py`.
