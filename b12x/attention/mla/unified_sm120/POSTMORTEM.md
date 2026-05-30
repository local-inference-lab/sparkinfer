# unified_sm120 — Port Postmortem & Handoff

Porting FlashInfer's hand-written SM120 sparse-MLA CUDA kernels to 100% CuTeDSL as a
unified, `cute.constexpr`-specialized backend in b12x. Built on branch `rs-1` over 14
commits (`5f342b2e` base → `9349aa31`), validated on RTX PRO 6000 Blackwell (sm_120, 188 SMs).

---

## 1. TL;DR / current state

- **`b12x/attention/mla/unified_sm120/`** is a new, 100%-CuTeDSL backend that reimplements
  FlashInfer's SM120 sparse-MLA **decode + prefill** for **two model types in ONE traced
  kernel family**, specialized at trace time via `cute.constexpr`:
  - **DSV4** (compressed, UE8M0 footer scales, `V_HAS_ROPE`) — hot-op **PTX byte-matches
    FlashInfer** (MMA 14/14/8, `cp.async.bulk`, mbarrier; prefill 384-thread/4-IO + setmaxnreg).
  - **GLM_NSA** (uncompressed, arbitrary-FP32 inline scales) — matches **legacy accuracy**
    (≥0.9995 vs the dense FP32 oracle) using the legacy raw-e4m3 + post-MMA-FP32-scale design.
- **Feature parity with upstream:** attn_sink, return_lse, `VALID_HPB<16` (TP shards),
  multi-token mixed per-token-length decode, dual-cache (extra tokens, incl. zero-width main),
  split-K + base-2 merge.
- **It is the SM120 default.** The existing front-door functions (`sparse_mla_decode_forward`,
  `sparse_mla_extend_forward`, `compressed_mla_decode_forward`) route here automatically on
  SM120+ CUDA. Escape hatch: `backend="legacy"` kwarg or `B12X_MLA_SM120_UNIFIED=0`.
  Genuinely-upstream-unsupported contracts (mapped `indexed_page_table`, partial dual-cache
  trio) **RAISE** — no silent legacy fallback.
- **Faster than legacy** at batch=1 decode: DSV4 **3.2–4.1×**, GLM **1.2–1.6×** (after
  replicating FlashInfer's wave-balanced split-K launch tuning).
- **Legacy kernels are untouched** (byte-identical) and remain the escape-hatch / non-SM120
  path. The whole effort was additive (+7,929 / −4).
- **Tests:** full MLA surface 157 passed, 0 real failures (6 benign env/diagnostic skips).

### Files
| file | role |
|---|---|
| `traits.py` | `ModelType`/`ComputeMode`/`ScaleFormat` int enums (const_expr keys) + `UnifiedMLATraits` + `infer_model_type` |
| `smem.py` | per-model SharedStorage layout (const_expr offsets), ~91 KB DSV4 / ~99 KB GLM |
| `io.py` | IO-warp producer: `cp_async_bulk_g2s_mbar` gather + mbarrier; `io_threads` param (32 decode / 128 prefill); per-chunk main/extra section dispatch |
| `decode_math.py` | the shared math stages S0–S7 (Q-quant, QK block-scaled / GLM post-MMA-scale, RoPE, online softmax, P/W-quant, XV, epilogue) — used by BOTH decode and prefill |
| `decode` kernel (in `launch.py`) | 288-thread warp-specialized decode (8 math + 1 IO) + split-K planner + merge wiring |
| `prefill.py` | 384-thread/4-IO single-pass prefill reusing the decode stages |
| `launch.py` | `run_unified_decode` / `run_unified_prefill` launchers, KernelCompileSpec keying, wave-balanced split-K heuristic, the multiple device entries (see §7) |

Deeper artifacts (gitignored working notes): `.sm120port/` (phase-1 design, parity plan,
benchmark findings, verified traits, reference python oracles). Extracted **FlashInfer
reference PTX/SASS** is archived at `~/projects/archive/sm120port/` (not committed).

---

## 2. How it was built (process)

A dynamic multi-agent **workflow** orchestration, phase by phase, with three rules that paid off:
1. **Gate-first.** Two empirical gates were proven on real SM120 hardware *before* any kernel
   body: (G1) emit `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` from CuTeDSL,
   (G2) launch a ~98 KB dynamic-smem kernel. Both passed → the whole architecture was viable.
2. **Numeric gate as truth.** Every kernel was gated by cosine vs a PyTorch/FP32 reference, not
   by an agent's say-so. PTX parity was checked by diffing `CUTE_DSL_KEEP=ptx` dumps against the
   extracted FlashInfer reference.
3. **Adversarial verify + orchestrator re-run.** Each phase had an independent verifier (anti-cheat:
   tolerances not loosened, real sparse topk, sink actually applied, no silent fallback), and the
   orchestrator re-ran every gate itself before committing. This caught several overstated claims.

Commit-per-phase kept every checkpoint green and bisectable.

---

## 3. What worked

- **Reusing the existing b12x kernels as idiom references.** The mbarrier producer/consumer
  pipeline (`_cute/pipeline.py`), the production-grade `const_expr` specialization style
  (`w4a16`), the inline-PTX op vocabulary (`fp4.py`: mxfp8 MMA, ldmatrix, cvt, byte_perm), and
  the base-2 split/merge (`split.py`, reused verbatim) meant most "novel" mechanisms already
  existed. Only **one** genuinely new PTX op (`cp_async_bulk_g2s_mbar`) had to be written.
- **`cute.constexpr` for unification.** One traced kernel covers DSV4 + GLM + FP8; each
  (model, scale_format, compute_mode, v_has_rope, has_extra) tuple constant-folds to its own
  cubin. Dead-code elimination is *real*: adding the GLM branches left DSV4 PTX **byte-identical**
  (verified by stash-diff). This is the single most important design lever.
- **Replicating FlashInfer's launch tuning.** Porting the `CEIL_WAVES_MAX=3` wave-balanced
  `num_splits`/`chunks_per_block` heuristic verbatim turned decode from *slower* than legacy
  into 3–4× *faster* — the kernel was already right; only the launch decision was wrong.
- **Holding the unified backend to the FULL legacy test surface.** Flipping the default and
  running all 7 MLA suites through unified surfaced 4 latent edge-case bugs the unified-only
  tests never hit (see §4). This is the highest-leverage QA step we did.
- **Matching the right reference per model.** DSV4 → FlashInfer (PTX parity). GLM → legacy
  (correctness), because FlashInfer has no arbitrary-scale GLM kernel to match.

---

## 4. What didn't work / pitfalls (and the fixes)

- **The Phase-1 design claimed "no in-repo precedent" for warp-spec/mbarrier/split-merge — it
  was simply wrong.** A skim-level design agent missed `_cute/pipeline.py`, the paged kernels,
  and `bf16_to_fp4_tma.py`. **Lesson:** validate design assumptions against the codebase before
  trusting them; an explicit "precedent audit" phase (Phase 2) corrected it with `file:line`.
- **Building the prefill 4-IO pipeline from scratch deadlocked** (~45 min of timeout-cycling on
  the mbarrier parity). The fix: **scale the proven 1-IO decode pipeline** to 4 IO warps
  (`io_threads=32→128`) keeping the mbarrier protocol *bit-identical* — no deadlock.
  **Lesson:** for warp-spec pipelines, reuse a working protocol; don't re-derive it.
- **Forcing GLM through the DSV4 block-scaled MMA cost ~0.003 cosine.** GLM's arbitrary FP32
  scales (~5e-5, non-power-of-2) **cannot be represented** by FlashInfer's `ue8m0` block-scale
  selectors, so the "dequant→requant to e4m3" path discards per-group mantissa headroom. Fix:
  GLM-only `const_expr` path that keeps raw e4m3 K/V and applies the FP32 group scale *after* the
  MMA (the legacy design). **Lesson:** PTX parity is only a goal where upstream actually has a
  matching kernel.
- **The default-flip exposed 4 latent correctness bugs** the unified tests missed (because they
  used `-1`-padded indices, fixed shapes, no graph capture, non-zero main cache):
  1. **zero-length row** in single-pass prefill → spurious `exp2(0)=1` softmax mass → garbage
     (cos 0.20). Fix: empty-row guard → `O=0/LSE=-inf`.
  2. **short single row** decode took the uniform scalar path and over-attended. Fix: per-token
     clamp even at `rows==1`.
  3. **graph-capture host-sync** (`torch.all(...).item()` in the launcher) — illegal under stream
     capture. Fix: under capture, skip the sync and take the (always-correct) per-token path.
  4. **zero-width main cache** (extra-only dual-cache) → `cute.make_layout(0)` crash. Fix:
     const_expr 1-extent guard; attend only the extra section.
- **Agent claims needed verification.** One verifier reported "graph-capture failures" I couldn't
  reproduce; one "byte-identical" was actually a single `min.s32` reordered (semantically
  identical). **Lesson:** re-run and diff yourself before believing a structured result.
- **One source-mapping agent failed** to emit structured output (prefill) and had to be re-run.

---

## 5. What we learned (technical)

- **CuTeDSL can hit the exact PTX.** `cp.async.bulk` (CTA-scope, mbarrier `complete_tx::bytes`),
  block-scaled `mxf8f6f4 ... ue8m0` MMA, ldmatrix variants, named barriers with explicit thread
  counts — all emit verbatim. Instruction-class-for-instruction-class parity with hand-written
  CUDA is achievable.
- **`cp.async.bulk` requires a 16-byte-multiple transfer size.** DSV4's 8-byte UE8M0 footer
  (→ 584 B/token) is not 16-aligned; gather the *grouped* footer (`BI×8=512 B`) instead.
- **Decode at batch=1 is latency-bound, not BW-bound.** The perf lever is **split-K
  parallelism** (flash-decoding) to fill the SMs, *not* memory bandwidth. `num_splits=1` starves
  a 188-SM GPU with 8 head-blocks.
- **GLM's arbitrary FP32 scales are fundamentally incompatible with `ue8m0` block-scaling** —
  the legacy "raw-e4m3 + post-MMA FP32 scale" is the correct design, and a 2-pass e4m3 W
  (HIGH + LOW residual) recovers ~7 mantissa bits for P·V without a bf16 MMA.
- **`const_expr` dead-code elimination gives true byte-identical specialization** — but only if
  divergent stages are *separate* const_expr arms (a shared loop drifted DSV4 by 14 registers
  until split out).
- **Launchers must be graph-capture safe** — no data-dependent `.item()` host syncs.

---

## 6. Status & remaining work

**Done (committed on `rs-1`):** full DSV4+GLM decode+prefill, dual-cache, all feature parity,
default-flipped, no fallbacks, faster-than-legacy decode, 157 tests green.

**Optional / future:**
- **P9c — FlashInfer AutoTuner.** Per-shape `chunks_per_block` sweep+cache (the env override hook
  `B12X_MLA_SM120_NUM_SPLITS` exists). Marginal — the wave-balanced default already wins.
- **BF16 compute mode.** Deferred; FP8-only today. A perf/coverage item (upstream uses BF16-QK
  for some prefill shapes), not a correctness gap.
- **`prmt.b32` soft parity.** DSV4 decode emits ~84 vs FlashInfer's ~98 (d2_load_b fragment
  synthesis). Non-blocking; numerics + MMA/ldmatrix/mbarrier counts already match.
- **DSV3.2 POW2_FP32 path.** Dropped per scope (the user runs GLM = arbitrary-FP32, not the
  pow2-quantized DSV3.2 cache). The `ScaleFormat` enum has only `UE8M0_BYTE` + `ARBITRARY_FP32`.
  Re-adding it needs a `quantize_kv_dsv3_2` reference cache.

---

## 7. Maintenance notes / gotchas for the next person

- **Device-entry pattern.** To keep DSV4 byte-identical while adding GLM / dual-cache /
  per-token-length, `launch.py` uses *distinct* `@cute.kernel` entries that share one
  `_kernel_body(has_extra, per_token_len, ...)` via const_expr: `kernel` (8-param, the
  byte-identical no-extra path), `kernel_extra`, `kernel_pertok`, `kernel_extra_pertok`. Prefill
  mirrors this. **Don't merge these into one entry** — the separate arms are what preserves
  byte-identity / the regression gates.
- **Two scale worlds.** `scale_format==UE8M0_BYTE` (DSV4): K scale is a footer byte → block-scaled
  MMA selector. `scale_format==ARBITRARY_FP32` (GLM): raw-e4m3 K + per-group FP32 scale applied
  *post-MMA* (S1) and as the V scale (S6, with the 2-pass W residual). These are mutually-elided
  const_expr arms; touching one must not perturb the other (re-verify DSV4 PTX byte-identity).
- **Cache versioning.** `KernelCompileSpec` versions are bumped on any device-trace change
  (decode and prefill independently) to invalidate stale cubins. Bump them if you change a kernel.
- **Re-verifying PTX parity.** `B12X_CUTE_COMPILE_DISK_CACHE=0 B12X_CUTE_COMPILE_MEMORY_CACHE=0
  CUTE_DSL_KEEP=ptx python <probe>`; diff the hot-op histogram against the reference in
  `~/projects/archive/sm120port/ref_ptx/` (env var is `CUTE_DSL_KEEP=ptx`, not the deprecated
  `CUTE_DSL_KEEP_PTX`). DSV4 byte-identity is checked by stash-diffing a recompiled pre-change tree.
- **Dispatch.** The single gate is `api.py::_use_unified_sm120(backend, device)`; `compressed_api`
  imports it. Default = unified on SM120+ CUDA; `backend="legacy"` / `B12X_MLA_SM120_UNIFIED=0`
  forces legacy; `backend="sm120_unified"` forces unified even with env=0. Non-CUDA/pre-SM120 →
  legacy. Unsupported-upstream contracts RAISE (never silent legacy).
- **Numerics.** DSV4 ≈ 0.9998 vs reference; GLM ≥ 0.9995. FP8 tolerance is `atol≈2e-2` /
  `cos>0.999` for the unified suites; the GLM dense-oracle tests use the legacy bar `cos≥0.9995`.

---

*Built via Claude Code dynamic-workflow orchestration; every gate independently re-run before commit.*
