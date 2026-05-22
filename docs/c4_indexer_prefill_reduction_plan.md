# C4 Indexer Prefill Reduction Plan

This is the approved implementation plan for bringing long-prefill C4 indexing
back toward the shared GLM/NSA indexer structure without regressing GLM.

## Invariants

- C4 prefill uses only the tiled/supertile path. Dense logits are decode-only.
- Kernel selection comes from the fixed workspace contract: `block_q`, `block_k`,
  `topk`, `supertile_pages`, `q_capacity`, and `index_heads`.
- Live request values such as active width, chunk page offset, chunk token offset,
  and row lengths are runtime inputs. They must not enter CuTeDSL constexprs,
  host-launcher cache keys, or workspace sizing.
- The GLM/NSA paged scorer front door keeps its original API. C4-only
  offset/window behavior lives behind the separate paged-windowed wrapper.
- There are no CPU fallbacks, DeepGEMM paths, FlashMLA paths, or silent fallback
  paths in this integration. Unsupported contracts fail hard.

## Implementation Phases

1. Lock the invariants with tests and explicit workspace checks.
2. Remove C4 supertile metadata staging by having the paged scorer consume the
   full page table plus runtime `source_page_offset` and `output_width_tokens`.
3. Keep the scorer as a layout variant of the NSA indexer family:
   GLM uses the original contiguous/paged front door, C4 uses the paged-windowed
   front door, and both reuse the tiled top-k/merge path.
4. Keep workspace and arena sizing fixed and bounded: tiled C4 prefill logits are
   sized for one fixed supertile, dense decode logits are sized separately, and
   non-concurrent attention/MoE scratch overlaps in the joint arena.
5. Profile long prefill. Only after direct paged scoring is stable, consider a
   scorer-local top-k variant that emits per-supertile candidates directly.

## Test And Benchmark Contract

- Correctness compares C4 tiled/supertile output with reference top-k on small
  and multi-supertile page tables.
- CUDA graph tests must cover decode and fixed-size prefill chunks.
- Regression tests must prove live page-table width or sequence length cannot
  create a new kernel contract after fixed workspace prewarm.
- `benchmark_mla` remains the GLM/NSA guardrail.
- C4 indexer benchmarks should report scorer, top-k, merge, and total where
  possible, with rows covering decode (`1`) and prefill chunk sizes up to `4096`.
- SGLang V4 Flash E2E runs under `~/projects/sglang/.venv` with
  `CUTE_DSL_ARCH=sm_120a`, CUDA graph capture enabled, and no DeepGEMM/FlashMLA.

## Acceptance Criteria

- Long prefill never calls the dense-logit C4 indexer.
- No C4 prefill per-chunk metadata-prep kernel remains.
- No CuTeDSL compile appears during steady-state long prefill chunks.
- GLM/NSA benchmark numbers remain within noise of baseline.
- V4 Flash produces sensible responses after long prefill and decode.
- Decode dense path remains available only for small decode shapes.

## Current Implementation Status

- The GLM/NSA paged scorer API is guarded by tests and does not expose C4
  `source_page_offset` or `output_width_tokens` parameters.
- C4 uses the paged-windowed scorer wrapper and fixed workspace prewarm for both
  the tiled top-k and paged scorer contracts.
- The C4 supertile path passes the live page table and lengths directly into
  the scorer and does not stage paged metadata through the workspace at runtime.
- Dense C4 logits remain a decode-only path; long prefill uses tiled/supertile
  logits with fixed workspace capacity.
- The current long-prefill supertile cap is selected by the fixed workspace
  contract, not live sequence length. SGLang caps this below the current b12x
  paged-scorer schedule threshold, so a 4096-row prefill chunk plans a
  65024-token C4 supertile while still consuming the live full page table at
  runtime.

## Current Benchmark Snapshot

Measured on GPU 7 with `CUTE_DSL_ARCH=sm_120a` using graph replay:

- C4 tiled prefill stress, rows=4096, page_table_width=4160, seq_len=266240:
  32768-token supertile median 453951.90 us; 65024-token supertile median
  429003.88 us.
- C4 dense decode, rows=1, page_table_width=4160, seq_len=266240:
  median 229.01 us. This remains a decode-only path.
- GLM/NSA guardrail: decode replay 73.73 us / step 81.55 us; prefill replay
  2488.32 us / step 2498.88 us.
