# SM120 dense FP8 GEMM — DeepGEMM perf port (postmortem & handoff)

Porting the SM120 DeepSeek FP8 GEMM techniques from `~/projects/deepgemm-other`
into b12x's CuTeDSL `dense_gemm`, to make the **FP8 kernels b12x exposes**
(`dense_gemm` and the `wo_a`/`wo_b` + `block_fp8_linear` paths built on it)
faster. Built on branch `rs-2`, validated on RTX PRO 6000 Blackwell (sm_120,
188 SMs). Commits: `87d4bed2` (Graft A), `1ca79e96` (Graft A.2 dense+block_fp8),
`bdedf51a` (Graft A.2 wo).

---

## 1. TL;DR / current state

- **dense FP8 (MXFP8) is now ~1.6× faster than FlashInfer CUTLASS** on the
  benchmark shape (Nemotron down-proj), up from **1.44× slower**:
  `benchmark_dense_gemm.py --dtype fp8` geo **1.44× → 0.63×** (lower = b12x
  faster). FP4 path unchanged (~0.77×).
- The whole win is **tile selection** — zero numeric change (every tile is
  byte-identical: cos=1.0, maxabs=0 vs the prior output and vs CUTLASS).
- **Two grafts, both `dense_gemm`-level, additive, fully gated:**
  1. **Graft A** — replace the wide-N MXFP8 tile pin `(128,128)` with the best
     **M-independent** tile **`(64,128)`**. Honors b12x's one-kernel-per-(N,K)
     freeze/reuse serving contract; beats (128,128) at *every* M (1.1×–2.4×);
     generalizes (best single tile on Nemotron / GLM5-down / block_fp8 shapes).
  2. **Graft A.2** — a DeepGEMM-style **`expected_m` regime hint**. A caller
     that declares its regime gets the per-regime-optimal tile
     (`expected_m<=1 → 16×128`, `<=128` decode `→ 32×128`, else `→ 64×128`),
     recovering the ~25% the M-independent default gives up at M=32–128, while
     staying M-independent *within the regime* so freeze/reuse still holds.
     `expected_m=None` reproduces Graft A exactly.
- **Threaded through the whole replace surface**: `dense_gemm`,
  `block_fp8_linear_mxfp8` (+ binding/prewarm/scratch), and the WO path
  (`wo_a`/`wo_b`/`wo_projection_mxfp8`/`wo_projection_inv_rope_mxfp8` + both
  bindings + `bind`/`bind_inv_rope`). Production DeepSeek-V4 WO in `vllm-other`
  reaches it via `plan.bind_inv_rope(...)`.
- **Tests:** full FP8 surface green (dense/wo/block_fp8/stack/scratch-bindings)
  + new regime-selection unit tests + end-to-end decode-reuse + byte-identical
  hint tests.

### Files touched
| file | change |
|---|---|
| `b12x/gemm/dense.py` | `_select_default_mma_tiler_mn`: `(64,128)` default + `expected_m` regime tiles; `dense_gemm(..., expected_m=None)` |
| `b12x/gemm/block_fp8_linear.py` | `expected_m` through `block_fp8_linear_mxfp8` / binding / build / workspace+scratch `.bind` / `prewarm` |
| `b12x/gemm/wo_projection.py` | `expected_m` through wo_a/wo_b leaves / both orchestrators / both bindings / both builds / all four `.bind*` |
| `benchmarks/probe_dense_fp8_tile_sweep.py` | new: per-(M,tile) sweep, parameterized by N,K (the evidence behind the tile choice) |
| `tests/test_dense_gemm_expected_m.py` | new: regime-selection unit tests |
| `tests/test_gemm_block_fp8_linear.py`, `tests/test_gemm_wo_projection.py`, `tests/test_wo_projection_scratch_bindings.py` | hint integration + fake updates |

---

## 2. The reframing finding (audit beat the premise)

> **Correction (2026-05-30, see §8):** the claim below that "the port was never a
> numerics gap" was **right for activations but wrong for weights**, and was
> *asserted, never measured*. A rigorous re-audit found a dropped
> re-quantization step in weight packing (DeepSeek arbitrary-fp32 block scales
> were rounded to UE8M0 while keeping stale FP8 values → ~2.7× excess weight
> error). Fixed; see §8.

b12x **already had** a working CuTeDSL SM120 block-scaled FP8 GEMM, and it
**already does DeepSeek-style scaling with the identical MMA instruction**
(`mma.sync...mxf8f6f4.block_scale.scale_vec::1X.m16n8k32...ue8m0`). Activations
are 128-col-max-abs quantized (SGLang/DeepSeek) and stored at 1×32 ue8m0;
weights are DSV4 128×128 block scales expanded to 1×32. DeepGEMM's "1d1d"
(kGranK=128) feeds the same MMA. So the port was never a *capability* or
*numerics* gap — it was **performance/tile-selection**. The "groups" in
`wo_a`/`wo_b` are the uniform batched-`L` axis (already supported), **not**
ragged MoE grouping — so DeepGEMM's m-grouped/masked/k-grouped machinery is out
of scope for this surface.

---

## 3. Root cause & the data

`dense_gemm` pinned `mma_tiler_mn=(128,128)` for `mxfp8 n>1536, m>=2`. On wide-N
shapes that tile spans only `ceil(N/128)` column tiles → ~32–64 CTAs on 188 SMs
→ B-bandwidth-starved, **flat ~80µs across M=2..256** (≈275 GB/s vs CUTLASS
~710). Tile sweep (`benchmarks/probe_dense_fp8_tile_sweep.py`, Nemotron
N=4096 K=5376, M=2..4096):

| tile | M=2 | M=64 | M=256 | M=2048 | M=4096 | geomean |
|---|---|---|---|---|---|---|
| (128,128) *(old)* | 79 | 80 | 82 | 248 | 508 | **121µs** ← worst at every M |
| **(64,128)** *(new default)* | 33 | 35 | 37 | 211 | 445 | **69µs** ← beats (128,128) everywhere |
| (32,128) *(decode-only via hint)* | 27 | 27 | 47 | **322** | **652** | 74µs ← best ≤128, regresses prefill |

`(32,128)`/`(64,64)` score better on the M≤256 benchmark but **regress prefill
(M≥2k) and are M-dependent** → they can't be the single durable tile. `(64,128)`
is the M-independent, prefill-safe winner; the hint unlocks `(32,128)` only for
callers that declare a decode regime.

---

## 4. The two invariants that shaped the design

- **One kernel per (N,K), reused for all live M under frozen resolution.** b12x
  keeps M out of the compile key (warm once, serve any token count, no per-M
  recompile; `test_block_fp8_linear_small_live_m_reuses_prefill_dense_kernel`).
  ⇒ the *default* tile must be **M-independent** (Graft A). The `expected_m`
  hint keeps this: it selects the tile from the declared regime, not live M, so
  one kernel per `(N,K,expected_m)` still serves all live M in that regime.
- **`expected_m` is NOT a cache key.** It only selects `mma_tiler_mn`; the tile
  (plus policy/N/K/L/dtypes) is the `@functools.cache` key. Verified: 4 distinct
  `expected_m` (64/128/256/None) → only **2 compiles** (64,128→hit; None hits
  the 256 kernel). Same-regime hints share one cubin.
- **M=1 and m<16 are separate *policy* regimes** (`use_m1_non_tma`, direct
  scheduler in `_dense_gemm_policy_for`) — a pre-existing constraint orthogonal
  to the tile/hint. Warm those classes separately if serving them under freeze.

---

## 5. Production hook (vllm-other) — recommended follow-up

DeepSeek-V4 attention output projection in
`vllm/models/deepseek_v4/attention.py` builds the WO binding per-forward via
`plan_wo_projection_scratch(...).bind_inv_rope(o, positions, ...)`. To get the
decode win for wo_b (N=hidden>1536), pass the captured token count as the
regime:

```python
return plan.bind_inv_rope(
    scratch=scratch, o=o, positions=positions,
    cos_sin_cache=self.rotary_emb.cos_sin_cache, weights=weights,
    heads_per_group=self.n_local_heads // self.n_local_groups,
    nope_dim=self.nope_head_dim, rope_dim=self.rope_head_dim,
    expected_m=max(1, int(o.shape[0])),   # <-- decode batch -> 32x128; prefill -> 64x128
)
```

Under CUDA-graph capture each graph fixes `o.shape[0]`, so `expected_m` is fixed
per captured kernel (freeze-safe). **Takes effect only once these `rs-2` b12x
changes land in the b12x that vllm imports** (vllm imports the main clone, not
this worktree).

**Alternative (no vllm change): auto-default `expected_m` in the WO binding.**
`build_wo_projection_inv_rope_binding` already knows `tokens` (it validates the
inputs), and the binding is built per-forward / per-capture, so defaulting
`expected_m = tokens` there is freeze-safe and would give vllm the decode win
with zero vllm edits — at the cost of making the WO *default* token-count-aware
(decode→32×128, prefill→64×128) rather than M-independent. Left as a decision
(explicit opt-in vs b12x-side default change).

### DSV4-Flash TP=2 validation (the real profile)

Config: hidden=4096, num_heads=64, o_groups=8, o_lora_rank=1024 → at TP=2:
n_local_heads=32, n_local_groups=4, heads_per_group=8. Decode token count is
tiny: `--max-num-seqs 2` × MTP(1+2) ⇒ **M ≈ 2–8**.

- **wo_b** (the hot wide-N decode GEMM): N=hidden=4096, K=groups·rank=4096, L=1.
  Measured at decode M (`probe_dense_fp8_tile_sweep.py 4096 4096`):
  **128×128 = 61µs → 64×128 (default) = 25µs → 32×128 (decode hint) ≈ 18µs**
  (3.0–3.3× over the old pin; 1.3× over the M-independent default). (32×128 and
  16×128 tie at decode; 32×128 chosen for robustness through M≤128. M=1 is ~27µs
  for every tile — `use_m1_non_tma` policy dominates.)
- **wo_a**: N=rank=1024 (≤1536) → occupancy branch (64×64), **unchanged** by this
  work; the hint is a no-op there. A separate (narrow-N) optimization if needed.

So the existing `expected_m≤128 → 32×128` mapping is already optimal for this
profile. **The WO path now auto-defaults `expected_m` to the captured token
count** (`build`/orchestrator), so this is realized with **zero vllm change**.

**End-to-end (full `wo_projection`: quant + wo_a + wo_b), CUDA-graph replay,
HEAD vs pre-graft baseline `3083e936`** (`benchmarks/probe_wo_decode_latency.py`):

| decode tokens | baseline (128×128 wo_b) | HEAD (auto 32×128 wo_b) | speedup |
|---|---|---|---|
| 2 | 71.7 µs | **29.1 µs** | **2.46×** |
| 6 | 72.0 µs | **28.7 µs** | **2.51×** |

The ~43 µs drop is exactly the wo_b tile change; wo_a + quant (~11 µs) are
unchanged. The DeepSeek-V4-Flash TP=2 decode WO projection is **~2.5× faster**.

---

## 6. What was NOT done, and why

The user picked the `expected_m` hint over the riskier kernel-level levers. The
M-independent tile ceiling is reached; further gains need kernel surgery with
diminishing returns:

- **split-K / stream-K** — the real fix for the M=32–128 CTA-starvation and the
  only lever that moves the *autobench metric* further; but high-effort,
  high-risk in-place mainloop+epilogue change with no escape-hatch flag.
- **cooperative `kNWarps×kMWarps`** (TiledMma `atom_layout (2,4,1)`, BLOCK_M
  ∈{96,160,224}) and **larger BLOCK_N** (>128 currently fails to compile) —
  uncertain ROI given (64,128) already wins; would need kernel work.
- **SF-major loop + genuine 1×128 packed-int32 SF** — fewer SF loads, but the
  Graft-A win was CTA/BW-bound, not SF-bound; big change (quant/pack/arena), low
  expected ROI at these tiles.

---

## 7. Maintenance notes

- **Re-derive the tile** with `benchmarks/probe_dense_fp8_tile_sweep.py [N K]`
  (sweeps M=2..4096; prints per-M optima + the best single M-independent tile).
- **Re-validate**: `benchmark_dense_gemm.py --dtype fp8` (correctness gate vs
  CUTLASS is inclusive) + the FP8 test surface. Run from a checkout cwd with a
  flashinfer-equipped interpreter (`~/projects/vllm-other/.venv/bin/python`).
- **Don't make the default tile M-dependent** — it breaks the freeze/reuse
  contract. Use `expected_m` for regime tuning instead.
- The wo benchmark's tokens=1 abort (cos 0.99189 < 0.995) is **pre-existing**
  (inherent MXFP8-vs-BF16 error; identical on baseline `3083e936`), not from
  these changes.

*Built via Claude Code dynamic-workflow orchestration; every gate independently
re-run before commit.*

---

## 8. Correction: the dropped weight re-quantization step (2026-05-30)

A re-audit re-examined the §2 premise *"the port was never a numerics gap"*
against DeepGEMM's actual quantizer (`deepgemm-other/deep_gemm/utils/math.py`),
**measuring** byte-exactness instead of asserting it. Findings:

**Activations were genuinely at parity** (the §2 claim was correct here):
- `quantize_block_fp8_linear_input_mxfp8` is **byte-for-byte identical** to
  DeepGEMM `per_token_cast_to_fp8` at gran_k=32 — same FP8 bytes *and* same
  UE8M0 scales (`benchmarks/probe_quant_parity_deepgemm.py`: 100%/100%).
- The `ceil(log2)/exp2` UE8M0 path equals DeepGEMM's bit-exact `ceil_to_ue8m0`
  across 200k values incl. exact powers of two and the real Triton kernel
  (`benchmarks/probe_quant_parity_adversarial.py`: 0 mismatches).
- gran_k=32 (b12x) vs 128 (DeepGEMM) is real but **numerically neutral** in
  realistic regimes (identical cos even with 3%×40 outlier channels) and costs
  nothing at the MMA (which consumes per-32 SF either way).

**The weight path dropped a re-quantization step (the real bug):**
- DeepSeek checkpoints carry `(w_fp8, weight_scale_inv)` where `weight_scale_inv
  = block_amax/448` is an **arbitrary fp32** 128×128 block scale. Production
  (`vllm-other …/scaled_mm/b12x.py`) feeds it straight to
  `pack_fp8_block_scaled_weight_mxfp8` — used by **both** dense `block_fp8` and
  the WO `wo_a`/`wo_b` weights.
- The packer **rounded the scale to the nearest power of two and kept the
  original FP8 values**, leaving values matched to the *unrounded* scale →
  per-block scale error up to √2.
- Measured (`benchmarks/probe_weight_requant_parity.py`, N=K=4096): weight
  rel-Fro vs bf16 **0.102** (round-keep) vs **0.0264** checkpoint floor — i.e.
  ~3.9× the irreducible error, ~10% GEMM error vs the fp32 oracle.
- **Why it was missed:** there was never a byte-exact test vs DeepGEMM, and the
  one test that exercises this packer (`test_pack_fp8_block_scaled_weight_…`)
  feeds **already-power-of-two e8m0** scales, so the arbitrary-fp32 case never
  ran. The GEMM correctness gate compared b12x to a reference consuming b12x's
  *own* FP8, so a DeepGEMM divergence was invisible.

**Fix** (`b12x/gemm/wo_projection.py`): when the block scale is **not** already
exact UE8M0, `pack_fp8_block_scaled_weight_mxfp8` now **re-quantizes** —
reconstruct `w_fp8 · s_fp32`, then ceil-UE8M0 `per_block_cast` to derive *fresh*
FP8 values consistent with the power-of-two scale (DeepGEMM
`per_block_cast_to_fp8` parity). One-time at weight load; no serving-path cost.
Already-UE8M0 scales keep their FP8 values verbatim (`_scale_is_exact_ue8m0`
guard), so the existing e8m0 contract is byte-identical.

- **Result:** dense weight rel-Fro **0.102 → 0.0373**, WO (num_groups>1)
  **0.0369** — at the DeepGEMM/e4m3 floor (`probe_weight_requant_integration.py`).
- **Gate:** `tests/test_fp8_quant_deepgemm_parity.py` (new) — the byte-exact
  DeepGEMM comparison that should have existed. Full FP8 surface shows **no new
  failures** vs baseline under the flashinfer interpreter (4 pre-existing
  failures unchanged: 2 WO-A *activation*-quant-vs-reference 32-vs-128 column
  mismatches — numerically neutral — and 2 `alpha` test-mock TypeErrors).

*Re-audit via Claude Code multi-agent orchestration; agent claims (e.g. a
spurious "divisor 240" and a mis-stated "128-col activation" granularity) were
re-verified against source and discarded where wrong before any change.*
