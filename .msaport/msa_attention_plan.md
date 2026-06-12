# MSA Sparse Block-List Attention — Phased Implementation Plan

Track 2 of 2 for MiniMax-M3 sparse attention (MSA). Companion: `.msaport/msa_indexer_plan.md` (produces the `q2k_indices` this track consumes). Built as a const_expr "sparse block walk" mode on the **paged GQA backend** (`b12x/attention/paged/`) — NOT the sparse-MLA backend: MSA has standard dual-stream paged K/V `[num_pages, page_size, kv_heads, head_dim]`, head_dim 128, and its gather granularity *is the page*.

## Context

MSA main branch: GQA 64 q / 4 kv heads, head_dim 128 (qk+vo). Each (query token, kv head) attends ONLY to its top-16 selected 128-token KV blocks. page_size=64 everywhere in this repo ⇒ **1 MSA block = 2 consecutive logical pages**: walking block j touches pages `(2*block_id, 2*block_id+1)` of the request's page table. Causality: all non-local selected blocks are fully past; only the local block needs masking; the tail block may straddle `cache_len`. Per-(query,head) block count = `min(16, ceil(visible/128))` — **head-invariant ⇒ no head dim in worklists** (kv_head stays a pure grid-Y replica, forward_paged.py:3127).

Priorities (user-confirmed): **decode first** (local agentic use; batch often 1; ctx → 1M; occupancy at batch=1 is first-class — 16 blocks × 4 heads of work must fill ~188 SMs ⇒ split-KV mandatory). KV dtype: **BF16 bring-up → FP8 production**. Prefill: per-token Q-outer bring-up, then **union-tile committed** (not optional). KV-outer (MiniMax's prefill design) out of scope. Scope ends at `b12x/integration` bindings + e2e numeric tests.

## q2k contract (shared with indexer track)

`q2k_indices: int32 [num_kv_heads=4, total_q_capacity, 16]`, contiguous; batch-local block ids, ascending, -1 tail padding, local block always present. Attention never reads contents on host (counts derive from `cache_seqlens`); kernel guards `block_id >= 0` defensively. **Graph capture: fixed-capacity device buffer owned by the caller/indexer, captured by pointer, contents rewritten each step before the attention segment** (same contract as page_table/cache_seqlens via `_copy_runtime_metadata`); binding validates `data_ptr` identity across prepares (mirror `_validate_runtime_metadata_reference`, paged_attention_scratch.py:578).

## Findings: decode split + merge machinery (read before touching)

**Decode split-KV was disabled at HEAD by `254b0a7a`** ("Long overdue API cleanups", 2026-06-12): planner.py:798-801 now `force_split_kv = mode == "verify"` + decode-split raises; :918-919 `disable_split_kv = True` for decode. Sole external caller (`b12x/integration/paged_attention_scratch.py:1396`) passes `force_split_kv=False` ⇒ every decode plan at HEAD is single-chunk and the merge never runs for decode. The machinery is intact (worked at parent `866ac1cd`):

1. **Plan time** (`prepare_decode_graph_replay_state`, paged_attention_scratch.py:1289-1416): capacity plan vs worst-case seqlen; `build_decode_chunk_pages_lut` (planner.py:632) ← `heuristic_decode_graph_chunk_pages` (consults `_BF16_MINIMAX_DECODE_MAX_CHUNKS` planner.py:54 via `_bf16_minimax_decode_max_chunks` :412 for the existing g=6 head128 MiniMax model), capped by `decode_graph_max_chunks_per_request_budget ≈ num_sms*ctas_per_sm/kv_heads/batch`. `max_chunks_per_req` sizes shape-stable arena views: worklists `(batch*max_chunks)`, `tmp_output [partial_rows, 64, 128]` bf16 **normalized** partial O, `tmp_lse [partial_rows, 64]` f32 **base-2**, `merge_indptr [total_q+1]`, `o_indptr [batch+1]`, `kv_chunk_size_ptr [1]`.
2. **Replay time** (graph_replay.py): in-graph Triton rewrites from live `cache_seqlens`: `update_decode_graph_metadata_fused_triton` (:198) → per-request num_chunks, indptr prefix sums, chunk size, valid-mask zeroing; `update_decode_graph_compact_work_metadata_triton` (:267) expands worklists. Regularized variant skips worklists (grid `(max_chunks_per_req, kv_heads, batch)`, CTA self-decodes from o_indptr, forward_paged.py:3129-3146). Both shape-stable (grid off static-capacity tensors :3063-3070, early-exit on valid mask).
3. **Forward split epilogue**: chunk window `[kv_window_start + kv_tile_idx*kv_chunk_size, min(..., cache_len))` (:3209-3218); `decode_partial_row_idx = o_indptr[req] + kv_tile_idx` (:3866-3870); normalized O + base-2 LSE writes (:5179-5195).
4. **Merge** (`merge.py PagedPersistentMergeKernel`, launched api.py:802-908): (row, head)-persistent FlashInfer split-state fold (`_state_merge_normalized_lse_base2`), cp.async staged. **Supports head_dim 128** (vec_size = 128/32 = 4; there's even a head128 `merge_bdy=3` regular-decode tuning at api.py:316-331). The `_merge_backend_supports_split_kv` head_dim_vo==256 gate (planner.py:66-71) only kills *non-forced* split.

**MSA rides this as-is**: re-open split behind the msa plan flavor; same worklists/regularized grid/partials/merge; one new Triton metadata variant computing chunks from the **virtual selected length** `v_len = min(16, ceil(cache_len/128)) * 128` with a constant chunk size (no LUT — work bounded at 2048 tokens).

**Occupancy at batch=1** (188 SMs, 4 kv heads, ≤32 logical pages/request/head): no split = 4 CTAs (~2%, unusable); chunk=128 virtual tokens ⇒ 16×4 = 64 CTAs (34%); chunk=64 (the stage-tile floor) ⇒ 128 CTAs (68%), merge fan-in 32/row comfortably in range. Starting chunk policy: `{bs≤2: 128, bs≤8: 256, else 512}` virtual tokens; `max_chunks_per_req = ceil(2048/chunk) ≤ 32`. Tune in P3 benchmarks.

## Findings: gqa_group_size gates at g=16

- api.py:511/:520 (`<= 8`): gate single_request/single_qtile decode-graph fast paths assuming one qo tile per request. At g=16 with cta_tile_q=16 the assumption holds exactly — **relax to `<= plan.cta_tile_q`**. Same relaxation at workspace.py:1141-1145 and paged_attention_scratch.py:1412-1414.
- api.py:824 (`== 6`): `pair_bf16_merge_partial_loads` micro-opt for the g=6 model. Leave off for g=16; optional P7 (`head_dim_vo==128 and bf16`).
- Kernel-internal: `decode_qwen_single_row_fastpath` (forward_paged.py:3652) and bf16/fp8 "row0" fastpaths (:3667-3695) require g∈{6,8} — at g=16 **both mma row slots are live, these must stay off (correctness)**; the generic `decode_row_metadata_fastpath` masking branch (:4236-4254) handles g=16. The role-specialized TMA producer (`bf16_minimax_role_specialized_decode`, :2492-2506) is g==6-gated; g=16 takes the generic single-stage bf16 TMA decode family (`use_paged_kv_tma_exact_plane_bf16_layout`, :2542-2553) whose trait requirements (cta_tile_q=16, cta_tile_kv=64, 1 q-warp, 4 kv-warps) are exactly what `select_paged_forward_traits` yields for head128 g=16 decode. **No traits.py change needed.**
- Planner heuristics have untuned `>8`/fallback branches; MSA gets its own chunk policy + `graph_ctas_per_sm=2` default.

## Cross-cutting rules

- Every msa branch const_expr-keyed; msa off + `mQ2KIndices=None` ⇒ **byte-identical PTX** for existing paths (verify via `cutlass___call___*.ptx` dump on one dense decode spec before/after each kernel phase).
- `window_left >= 0` (SWA) and `attention_sink_bias` rejected for msa plans (assert in `create_paged_plan`).
- Planner `regularized_decode_graph=False` hardcode (planner.py:911) stays; msa rides the metadata-driven single-qtile path first; regularized grid only if P3 benchmarks justify.
- Known pre-existing breakage (don't chase): stale `,split` asserts tests/test_attention_paged_forward.py:604/:613; `PagedAttentionWorkspace.run` (workspace.py:1496-1500) passes a removed `workspace=` kwarg — live path is the binding.

---

## Phase P0 — Reference + fixtures (no kernel changes)

- `b12x/attention/paged/reference.py`: add `msa_attention_reference(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices, k_descale=None, v_descale=None) -> (out, lse)`: per (request, kv_head, q token) gather blocks `b = q2k[h,q_row,j] (b≥0)`, keys = rows `[b*128, b*128+128) ∩ [0, causal_limit]` with `causal_limit = token_local + cache_len - qo_len` (decode: `cache_len-1`), via pages (2b, 2b+1); fp32 softmax; natural-log LSE (tests convert kernel base-2 like `test_attention_cuda_graphs.py:_lse_base2_to_natural`). Self-check vs brute-force masked dense for small shapes.
- `tests/paged_attention_helpers.py`: `make_msa_q2k_indices(...)` — sorted unique random blocks, local block forced, -1 tail, adversarial poison option for padded slots.
- **Create** `tests/test_attention_paged_msa.py` (reference battery only at this stage). Pin dense baselines green at HEAD (decode currently non-split — clean perf denominator).

## Phase P1 — Decode sparse block walk: eager, single-chunk, BF16 KV

Highest-risk kernel change on the smallest surface (decode_only path, generic bf16 head128 TMA family). BF16 first: most-exercised head128 decode family (the existing MiniMax path minus its g==6 role-specialization), no descales, avoids fp8 native-MMA heuristics keyed off chunk size (api.py:78-97).

forward_paged.py (`PagedForwardKernel`, const_expr `self.msa_block_sparse`; class consts MSA_BLOCK_TOKENS=128, MSA_TOPK=16; assert page_size 64):
- **Work decode** (:3171-3230): `nblocks = min_(16, ceil_div(cache_len, 128))`; walk in **virtual selected-token space** `[0, v_len = nblocks*128)`; P1 forces one chunk (`chunk_start=0, chunk_end=v_len`).
- **Tile→page mapping** at the 3 walk sites (role-producer :3722-3748 — inactive at g=16 but keep consistent; initial prefetch :3921-4052; main loop :4054-4057): currently `tile_token_base` feeds `_issue_paged_kv_tma_copy_{1,2,3}planes` which do `page_idx = tile_token_base // 64` (:2789/:2813/:2835/:2855). Add const_expr msa coordinates: `j = s_base // 128; block_id = mQ2KIndices[kv_head_idx, q_start + q_token_local, j]; page_idx = 2*block_id + ((s_base // 64) & 1); key_base = block_id*128 + (s_base % 128)`. **Refactor the issue helpers to take precomputed `page_idx`** (dense passes `tile_token_base // page_size`) — one shared TMA body. Decode q2k row = `q_start` (formulated as `q_start + q_token_local` so spec-decode later changes only the planner). ≤16 ids read inline (one scalar ldg per 128 tokens; register staging optional P7).
- **Masking** (:4212-4288): decode-fastpath live predicate becomes `tile_tokens = clamp(cache_len - key_base, 0, min(64, chunk_end - s_base))` — covers tail-block straddle (partial/empty second sub-tile), exact multiples, short sequences (<16 blocks never visited since chunk_end=v_len). Defensive `block_id < 0 ⇒ tile_tokens = 0` (page table untouched).
- Ascending lists keep `key_base` monotone ⇒ streaming-softmax order assumptions hold; blocks disjoint ⇒ LSE exact.

planner.py: `create_paged_plan(msa_block_sparse=False)` → `PagedPlanKey` (+ msa_topk=16, msa_block_tokens=128 key fields); decode msa effective work = `2*min(16, ceil(len/128))` logical pages, head-invariant, same `(request, 0, 0)` items; `kv_chunk_size = 2048` (single chunk); validate window_left==-1, g==cta_tile_q, page_size 64.

api.py: q2k plumbing — `_to_kernel_tensor(q2k, cutlass.Int32, assumed_align=4)` appended after `block_valid_mask_arg`; `_tensor_meta_key(q2k, dynamic_dims=() if static_decode_graph_bucket else (1,))` into forward_cache_key; `bool(msa)` in kernel_policy tuple (~:683); `_build_forward_kernel` lru key gains the flag.

b12x/integration/paged_attention_scratch.py: `B12XPagedAttentionBinding.q2k_indices` field + validation (shape `(4, ≥capacity_total_q, 16)`, int32, contiguous, device, data_ptr stability); `B12XPagedAttentionScratchCaps.msa_block_sparse: bool = False`; optional env kill-switch `B12X_PAGED_MSA=0` ⇒ hard error, never silent dense fallback. `b12x/integration/attention.py` re-exports untouched (binding carries it).

Tests (vs `msa_attention_reference`, eager `bind(...).run()`, bf16, 64/4/128):
1. bs1 cache_len ∈ {1, 64, 127, 128, 129, 200, 2047, 2048, 5000} — <16 blocks, tail straddle inside/at page boundary, full lists.
2. Multi-batch varlen bs ∈ {2,5} mixed lengths.
3. **-1 padding poison** (garbage ids + poisoned KV pages behind padded slots ⇒ no output effect).
4. Non-contiguous sorted lists incl. block 0 + far blocks against a **shuffled page table** (indirection correctness).
5. LSE parity ≤2e-3 rel, O cosine ≥0.999 (existing tolerances).
Regressions: `tests/test_attention_paged_forward.py`, `test_attention_paged_planner.py`, `test_paged_attention_scratch_bindings.py` unchanged; PTX-identity msa-off.

## Phase P2 — Split-KV re-enable for MSA decode + merge (eager)

- planner.py: msa decode plans take the split path — bypass the two 254b0a7a gates **for msa only**; chunk in virtual tokens per the policy table; emit `(req, 0, kv_tile)` worklists, merge_indptr/o_indptr (qo_len=1), `total_partial_rows = Σ num_chunks_r`; `kv_chunk_size` stored in tokens, interpreted in virtual space under msa. Reject non-multiple-of-64 chunk sizes.
- forward_paged.py: generalize P1 to `chunk_start = kv_tile_idx*kv_chunk_size`, `chunk_end = min(+kv_chunk_size, v_len)`. Partial epilogue untouched (:5179-5195).
- api.py/merge.py: split branch already launches merge; head128 works (vec_size=4). Verify `_build_merge_kernel` bdy heuristic at head128/small total_q. Eager only (`merge_regular_decode_graph=False`).
- Capacity: eager scratch `max_partial_rows ≥ batch*32`; existing arena fields.

Tests: P1 battery × chunk ∈ {64,128,256,2048} — **split-invariance** (all chunkings agree within fp tolerance and match reference); merge LSE base-2→natural parity; planner rejects chunk%64≠0.

## Phase P3 — CUDA-graph decode replay + benchmarks (the priority deliverable)

- graph_replay.py: `update_msa_decode_graph_metadata_fused_triton` + host wrapper (model on :198-265): `num_chunks = ceil(min(16,ceil(len/128))*128 / kv_chunk_size)` (constant chunk size — reuse the `HAS_KV_CHUNK_SIZE_TENSOR/FIXED_KV_CHUNK_SIZE` pattern of :295-359; no LUT), write indptr prefix sums + kv_chunk_size_ptr, zero valid mask; reuse `update_decode_graph_compact_work_metadata_triton` (:267-292) **verbatim** for worklist expansion.
- workspace.py / paged_attention_scratch.py: `prepare_decode_graph_replay_state(msa_block_sparse=True)` branch — skip page-count LUT; `max_chunks_per_req = ceil(2048/chunk)`; capacity plan caps partials at 32 pages/request **regardless of 1M-context page tables**. Wire into `update_decode_graph_replay_metadata_from_runtime_cache_seqlens` (workspace.py:874-961, scratch :754+) ahead of split/regular dispatch. Relax the two `gqa_group_size <= 8` replay gates to `<= plan.cta_tile_q`. The metadata update captures in-graph via the existing `_capture_decode_graph_replay_metadata_if_needed` hook (api.py:390-412) — msa adds nothing there.
- q2k stable-buffer contract enforced (pointer-captured; tests rewrite contents manually; production indexer rewrites pre-segment).

Tests (extend `_PagedGraphScratchHarness`, tests/test_attention_cuda_graphs.py): capture at (bs, max_pages) bucket; replay with (i) shorter seqlens (fewer blocks/chunks), (ii) mutated q2k contents, (iii) mutated page_table — each vs eager P2; padded-batch replay (replay bs < captured bs via seqlen fill convention).

Benchmark: **create** `benchmarks/benchmark_paged_msa.py` (clone `benchmark_paged_attention.py` `_capture_graph`/`_bench_graph`): decode bs {1,4,16} × ctx {32K,128K,512K,1M} × chunk {64,128,256,512}; columns µs/step, effective GB/s (bytes ≈ bs·4·nblocks·128·128·2B·2 + partials), launch CTAs, dense-decode baseline (expect MSA flat in ctx, dense linear) → `results.msa_decode.tsv`. Bake tuned chunk table + `graph_ctas_per_sm` into the planner.

Exit: bs1/1M decode attention few-tens-of-µs, ≥64-128 active CTAs, replay-correct under mutating metadata.

## Phase P4 — FP8 KV decode

- Walk is dtype-agnostic; verify fp8 plane path under msa (1 K plane + 1 V plane of 128 cols, :2601-2614). Initially force `use_native_fp8_qk/pv = False` for msa (`_resolve_native_fp8_attention_mma_flags` heuristics api.py:78-97 are dense-chunk-tuned); after correctness, evaluate native fp8 MMA with msa-specific thresholds (`B12X_TURBO_ATTN` path), gated on benchmark + cosine ≥0.995.
- Tests: fp8 via `quantize_paged_kv_cache_e4m3` (tests/paged_attention_helpers.py), per-request and per-(request,head) descales. Benchmark grid repeated with kv_dtype axis.

## Phase P5 — Prefill v1: per-token Q-outer extend tiles

- planner.py: `mode="extend" and msa` ⇒ force `cta_tile_q=16` in `_paged_determine_cta_tile_q` — each packed qo tile = exactly one token × 16 heads (`qo_tile_idx == token_local` identity). One work item per (request, token) × 4 head planes (8K chunk ⇒ 32K CTAs, saturating); no split (extend split is gone anyway); per-tile work ≤ 2048 keys.
- forward_extend_generic.py (head128 extend = cp.async ingress; TMA plane path is head256-only :2227-2243; page lookups at :157/:181/:232/:267, :2581-2601): under msa const_expr — `token_local = qo_tile_idx`; `visible = cache_len - qo_len + 1 + token_local`; `nblocks = min(16, ceil(visible/128))`; same walk as P1 with q2k row `q_start + token_local`; keep per-row `causal_k_limit` machinery but compare `key_pos = key_base + key_local`. 64-token stage tiles sit inside one logical page ⇒ single-page-per-tile holds on the cp.async path. All 16 rows of a tile share one token ⇒ uniform causal limit; only the local block partially masks.
- api.py: msa flag + q2k through `_build_extend_forward_kernel` (q2k inserted before the TMA-desc-ptr arg tail; cache key updated).

Tests: qo_len ∈ {1, 5, 300} with per-token growing lists (every token's local block present), multi-request varlen, vs reference; **qo_len=1 extend ≡ decode** cross-check. Benchmark: extend at prefix 128K, chunk N ∈ {2K,8K,32K}, ms + effective TFLOP/s vs dense extend (M=16 tiles are mma.sync m16n8-native; target ≥~25-30% of dense extend MMA efficiency for this bring-up phase).

## Phase P6 — Union-tile prefill (COMMITTED)

- `cta_tile_q = 128` (8 tokens × 16 heads). **Pre-pass** (small Triton kernel alongside graph_replay.py utilities): per 8-token group, union of the 8 sorted 16-entry lists → per-tile union block list (≤128 entries; expected ~20-40 given diagonal/sink/stripe selection overlap) + per-row 16-bit membership masks; written to a workspace buffer `[num_tiles, max_union + mask words]`.
- forward_extend_generic.py: `msa_union_tile` const_expr — walk the tile's union list; in the S-mask stage AND the per-row membership bit for the current block into the existing causal mask. Per-row causal_k_limit machinery unchanged.
- planner/workspace: tile count = ceil(qo_len/8) per request; union buffer capacity = num_tiles × 128 worst case (cap + assert; fall back to P5 per-token path for pathological tiles if cap exceeded — planner decides per tile group is overkill, per plan is fine for v1).
- Tests: parity vs P5 (same q2k ⇒ same output within fp tolerance); adversarial zero-overlap (union=128) and full-overlap lists; membership-mask correctness for tokens that didn't select the current block; union-buffer cap fallback.
- Benchmark: vs P5 at chunk {2K,8K,32K} × prefix {32K,128K,512K}; report measured union sizes (selection-overlap statistics) alongside speedup. Acceptance: union beats per-token by ≥1.3× at the 8K-chunk/128K-prefix point; below that it ships opt-in (env) rather than default.

## Phase P7 — Optional follow-ups (separately benchmarked, keep-if-wins)

- **Broader split-KV re-enable for dense decode** (user-approved exploration): hard constraint — graph capture must be unaffected; benchmark dense decode split vs non-split before proposing any default change.
- g=16 role-specialized TMA producer warp (extend :2492-2506 beyond g==6); 3-stage pipeline for the msa walk.
- Merge pairing relaxation (api.py:824 → `head_dim_vo==128 and bf16`); `merge_bdy` retune at fan-in 16-32.
- Register/smem staging of the 16 block ids at CTA prologue.
- Chunk-policy autotune loop on `results.msa_decode.tsv` (b12x-autobench style).
- Spec-decode (qo_len>1 decode): planner-only work given the `q_start + q_token_local` formulation.

## E2E integration test (after P3 + indexer A5)

**Create** `tests/test_attention_msa_e2e.py`: synthetic pipeline — pack K_idx rows into the index cache → block scores (A5/A3) → selection (A2) → sparse decode (P3) [and P5/P6 prefill] — vs composed torch reference (`msa_q2k_indices_reference` + `msa_attention_reference`). Decode variant fully graph-captured (one graph: score → pool → select → attention) and replayed across mutating seqlens. Document the tensor contracts in `b12x/attention/indexer/MSA.md` — this is what MiniMax-M3 serving integration consumes.

## Verification

Per phase: `pytest tests/test_attention_paged_msa.py -x -q` + regression files (`test_attention_paged_forward.py`, `test_attention_paged_planner.py`, `test_paged_attention_scratch_bindings.py`, `test_attention_cuda_graphs.py`). JIT first-compile 30s+ normal; >30s silence warm = hang (CLAUDE.md). Benchmarks graph-captured with L2 flush → `results.msa_decode.tsv`. PTX-identity spot-check msa-off after each kernel phase.

## Critical files

`b12x/attention/paged/{forward_paged.py, planner.py, api.py, graph_replay.py, merge.py, reference.py, workspace.py, forward_extend_generic.py}` · `b12x/integration/paged_attention_scratch.py` · `tests/test_attention_paged_msa.py(new)`, `tests/test_attention_msa_e2e.py(new)` · `benchmarks/benchmark_paged_msa.py(new)`.
