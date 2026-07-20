# CUTLASS DSL 4.6 migration gate

This migration is evaluated on physical GPUs 4 and 5. CPU-only results are not
part of the acceptance gate.

## Comparisons

Two comparisons are required so source changes cannot hide a compiler
regression:

1. **Compiler-only:** identical final sparkinfer source compiled once with CUTLASS DSL
   4.5.2 and once with 4.6.0.
2. **End to end:** the pre-migration 4.5.2 source compared with the final 4.6.0
   candidate.

Both runs use fresh compile caches. Every cache object must have a valid
semantic manifest and exact launch-time dynamic shared-memory accounting.
The corpus populates each fresh cache in a separate GPU process before starting
Nsight, then repeats the identical node set under Nsight and requires disk-hit
events for the same specialization keys with zero profiled misses.  This keeps
CUPTI out of compiler helper subprocesses (which can deadlock after exit under
injection) while still tracing every exact production CUDA entry point and the
serving disk-cache load path.
Every performance log must also carry the exact sparkinfer package-content
fingerprint and benchmark-script SHA-256; the A-B-B-A comparator rejects a
missing or mixed fingerprint even when the commit, dirty-path list, and branch
names happen to match.
`python -m validation.cutlass_migration evidence compare-resources --fail-on-unmatched`
requires the two runs to
contain the same compile specializations; the production family inventory in
`validation/cutlass_migration/data/cute_production_kernel_coverage.txt` prevents a symmetric but empty
or partial run from passing.

### Evidence status

The fresh same-source 4.5.2/4.6.0 static census matches all 140 reviewed
semantic-specialization/entry-point rows, with no unmatched rows or source-
coverage gaps.  It flags 21 allocated-GPR increases, 73 rows with at least one
positive exact R/UR/P/UP SASS-use metric, and six rows with an increased local-
memory metric.  It finds no new register ceiling, no newly local-memory-using
row, no launch-SMEM change, and no occupancy regression.  These counts describe
alarms, not performance results: every positive delta remains visible for
causal review even when the row stays spill-free and occupancy is unchanged.

The default closeout is deliberately lean: run the exhaustive static resource
and register census once, then run GPU correctness and exact-object graph
timing only for changed implementations, known cliffs, and resource changes
that could plausibly affect execution.  The 104-process, all-family/all-position
evidence-set constructor and four-corpus release index remain preserved as the
optional formal full-matrix workflow; they are not rerun by default merely to
repeat low-risk runtime coverage.  Any performance disposition still uses the
real target path on physical GPUs 4 and 5 and must retain its raw evidence.

The final resource commands are:

```bash
python -m validation.cutlass_migration evidence resources "$CACHE_45" \
  --format csv --output "$REPORT_45" \
  --require-semantic-manifest --require-launch-dynamic-smem \
  --require-cutlass-dsl-version 4.5.2 \
  --require-cutlass-libs-base-version 4.5.2 \
  --require-cutlass-libs-cu13-version 4.5.2 \
  --require-kernel-id-pattern-file validation/cutlass_migration/data/cute_production_kernel_coverage.txt \
  --require-kernel-symbol-pattern-file validation/cutlass_migration/data/cute_kernel_symbol_coverage.txt

python -m validation.cutlass_migration evidence resources "$CACHE_46" \
  --format csv --output "$REPORT_46" \
  --require-semantic-manifest --require-launch-dynamic-smem \
  --require-cutlass-dsl-version 4.6.0 \
  --require-cutlass-libs-base-version 4.6.0 \
  --require-cutlass-libs-core-version 4.6.0 \
  --require-cutlass-libs-cu12-version 4.6.0 \
  --require-cutlass-libs-cu13-version 4.6.0 \
  --require-kernel-id-pattern-file validation/cutlass_migration/data/cute_production_kernel_coverage.txt \
  --require-kernel-symbol-pattern-file validation/cutlass_migration/data/cute_kernel_symbol_coverage.txt

# When GPU corpus shards use separate caches/reports, merge each toolchain
# independently. Conflicting duplicate semantic-key/symbol rows, mixed sparkinfer
# package fingerprints, and mixed CUTLASS/toolchain versions are rejected.
python -m validation.cutlass_migration evidence merge-resources \
  "$ATTENTION_REPORT_45" "$COMPUTE_REPORT_45" "$W4A16_REPORT_45" \
  --output "$REPORT_45"
python -m validation.cutlass_migration evidence merge-resources \
  "$ATTENTION_REPORT_46" "$COMPUTE_REPORT_46" "$W4A16_REPORT_46" \
  --output "$REPORT_46"

python -m validation.cutlass_migration evidence compare-resources \
  "$REPORT_45" "$REPORT_46" --output "$DELTA_REPORT" \
  --require-semantic-manifest --require-exact-launch-dynamic-smem \
  --require-matching-package-fingerprint \
  --fail-on-unmatched

# This is an alarm pass, not the report-generation pass above.  It is expected
# to return nonzero whenever a positive resource delta still needs review.
python -m validation.cutlass_migration evidence compare-resources \
  "$REPORT_45" "$REPORT_46" --output "$DELTA_ALARM_REPORT" \
  --require-semantic-manifest --require-exact-launch-dynamic-smem \
  --require-matching-package-fingerprint \
  --fail-on-unmatched --fail-on-resource-regression

# EIATTR_REGCOUNT is the allocation ceiling, but it can stay flat while the
# generated program names more physical operands.  Re-open the exact cubins
# from both reports and retain the complete R/UR/P/UP index sets for every
# semantic-specialization/entry-point pair.
python -m validation.cutlass_migration evidence sass \
  "$REPORT_45" --output "$SASS_SETS_45"
python -m validation.cutlass_migration evidence sass \
  "$REPORT_46" --output "$SASS_SETS_46"
python -m validation.cutlass_migration evidence compare-sass \
  "$SASS_SETS_45" "$SASS_SETS_46" \
  --output "$SASS_SET_DELTA_REPORT"

# Likewise, retain the complete sidecar first, then run the intentionally
# nonzero alarm that flags every positive exact SASS-use/local/SMEM delta.
python -m validation.cutlass_migration evidence compare-sass \
  "$SASS_SETS_45" "$SASS_SETS_46" \
  --output "$SASS_SET_ALARM_REPORT" \
  --fail-on-register-usage-increase \
  --fail-on-local-memory-increase \
  --fail-on-shared-memory-increase

python -m validation.cutlass_migration evidence source-inventory "$REPORT_46"

# The optional formal release workflow publishes its all-specialization table
# only after every exceptional resource row has a causal/disposition/performance
# annotation.  The lean migration closeout instead retains the compact alarm
# inventory in docs/cutlass-46-register-accounting.csv.
python -m validation.cutlass_migration evidence register-accounting "$DELTA_REPORT" \
  --sass-register-delta "$SASS_SET_DELTA_REPORT" \
  --annotations "$RESOURCE_ANNOTATIONS" \
  --require-annotations-for-exceptions \
  --output docs/cutlass-46-register-accounting.csv

# A formal release also publishes SASS_SET_DELTA_REPORT beside the accounting
# table as the raw disassembly sidecar.  The accounting builder joins it
# one-to-one and requires annotations for exact usage increases, so a flat
# EIATTR_REGCOUNT row cannot disappear from review.

python -m validation.cutlass_migration diagnostic graph-abba \
  "$A1_JSONL" "$B1_JSONL" "$B2_JSONL" "$A2_JSONL" \
  --backend sparkinfer --output "$ABBA_REPORT" \
  --expected-a-cutlass-version 4.5.2 \
  --expected-b-cutlass-version 4.6.0 \
  --allowed-physical-gpu "$PHYSICAL_GPU" \
  --require-l2-flush \
  --require-serving-contract \
  --max-mean-regression-pct 0.5 \
  --max-median-regression-pct 0.5 \
  --max-p95-regression-pct 1.0 \
  --max-run-mean-drift-pct 1.0
```

### Optional formal full-matrix release artifact index

After producing the complete physical-GPU matrix, run the offline final gate:

```bash
python -m validation.cutlass_migration acceptance release-index \
  --gpu4-cutlass45-corpus "$GPU4_CORPUS_45" \
  --gpu4-cutlass46-corpus "$GPU4_CORPUS_46" \
  --gpu4-resource-delta "$GPU4_RESOURCE_DELTA" \
  --gpu4-sass-delta "$GPU4_SASS_DELTA" \
  --gpu4-accounting "$GPU4_ACCOUNTING" \
  --gpu4-abba-root "$GPU4_EXACT_CACHE_ABBA_ROOT" \
  --gpu5-cutlass45-corpus "$GPU5_CORPUS_45" \
  --gpu5-cutlass46-corpus "$GPU5_CORPUS_46" \
  --gpu5-resource-delta "$GPU5_RESOURCE_DELTA" \
  --gpu5-sass-delta "$GPU5_SASS_DELTA" \
  --gpu5-accounting "$GPU5_ACCOUNTING" \
  --gpu5-abba-root "$GPU5_EXACT_CACHE_ABBA_ROOT" \
  --output-json "$RELEASE_ARTIFACT_INDEX" \
  --output-csv "$RELEASE_EXCEPTION_INDEX"
```

The JSON uses schema `sparkinfer.cute.migration.release_artifact_index.v1`; the CSV
uses `sparkinfer.cute.migration.release_exception_index.v1`. The command imports no
GPU runtime and fails closed unless all four corpora have the same frozen
source and exact production specialization coverage. Every resource symbol
must be case-bound. Every production-bound compile spec—not only rows with a
resource exception—must have two independent warm-L2 and cold-L2 exact-cache
A-B-B-A artifacts on each GPU with identical GPU-4/GPU-5 coverage. Artifacts
must retain at least 1,000 raw samples per arm, exact before/after cache-object
hashes, paired physical-GPU mode snapshots, and independent oracle, arm
equality, poison-overwrite, input-immutability, fixed-capacity, stable-address,
allocator, and graph-topology gates. The release ceilings cannot be loosened:
0.5% for mean and median regression and 1.0% for p95 and independent-run mean
drift. Outputs are written only after every check passes and hash both their
inputs and the compact exception index.

The non-W4 producer-to-gate map is
`validation/cutlass_migration/data/cute_migration_abba_producer_contracts.json`.
Audit it offline
after changing either a producer or the final validator:

```bash
python -m validation.cutlass_migration acceptance paired-contract-audit \
  --output /tmp/sparkinfer-abba-producer-contract-audit.json

python -m validation.cutlass_migration integrity-check exact-cache-abba
python -m validation.cutlass_migration integrity-check release-aggregate
```

These are static developer checks, not migration acceptance. The shared-timer
self-test permanently covers fixed-pool ordering, balanced cycle inputs,
duration overshoot, P1/exact-throttle-policy gates, and zero clock-delta acceptance;
the release self-test independently reconstructs and mutates the emitted
artifact envelope. The producer-contract audit's
`runtime_follow_up` entries identify proofs that still need a GPU scenario;
the four-corpus release index remains the formal full-matrix gate when that
optional workflow is selected.  The default lean closeout instead combines the
complete static census with targeted GPU runtime dispositions.

The alarm invocations of the resource comparators deliberately return nonzero
for every register-usage increase, not only spills; the preceding report passes
still write complete inputs for causal annotation. `EIATTR_REGCOUNT` supplies the exact
per-thread GPR allocation. The SASS sidecar additionally records the complete
physical index set plus distinct count, minimum, maximum, inclusive span, and
`max(index) + 1` index span for `R`, `UR`, `P`, and `UP`. These are labeled as
SASS-use metrics and never presented as occupancy allocation counts. Any
allocation, distinct-count, maximum-index, or inclusive-span increase remains
a flagged exception; every exact addition/removal remains visible even when
the aggregate metrics improve. The CSV retains the complete compile-spec JSON and raw
toolchain/options for each row so any reviewed exception is tied to an exact
specialization rather than a symbol-name guess. Pairing uses both the semantic
compile key and the exact CUDA entry-point symbol, because one host object may
contain several kernels. The source inventory independently
requires GPU-produced cubins for all 47 `@cute.kernel` bodies: 43 production
entries plus four explicitly classified diagnostic entries. Diagnostic entries
are resource-accounted but do not support serving performance claims.
The `evidence register-accounting` command retains all matched rows, including
flat and improved rows, joins the exact R/UR/P/UP sets and deltas, adds the
per-CTA register footprint, and refuses to publish while any allocation,
distinct-count, maximum-index, or inclusive-span increase (or other resource
exception) lacks a cause, disposition, evidence, or graph-performance status.

## Order of gates

1. **Correctness and serving behavior**
   - Validate the real GPU kernel output against the appropriate oracle.
   - Check nonzero outputs, cosine/top-k equality, quantization semantics, and
     boundary shapes.
   - Capture and replay CUDA graphs only after all scratch, descriptors,
     metadata, and fixed-capacity workspaces have been allocated and prepared.
   - Replay with changed metadata without changing captured tensor addresses.
2. **Complete resource accounting**
   - Record exact GPR allocation plus uniform/predicate SASS register use for
     every cubin kernel, including every increase even when it remains
     spill-free.
   - Record frame/stack bytes, LDL/STL counts, the 255-register ceiling, thread
     geometry, parameter bytes, exact inferred launch SMEM, and cubin shared
     sections.
   - Treat new local memory, a new register ceiling, any static or dynamic SMEM
     change, changed launch metadata, or unmatched semantic/kernel
     specializations as hard failures pending explanation.
3. **Cause isolation**
   - Reassemble PTX from both compiler versions with the same PTXAS to separate
     DSL/LLVM lowering changes from assembler allocation.
   - Compare PTX/LLVM/SASS instruction classes, live ranges, uniform-register
     use, warp-role register limits, and occupancy.
   - Use a small structural canary before applying a change to the whole kernel
     family.
4. **Performance**
   - Use the real graph-replay path, stable addresses, fixed workspace capacity,
     warmup, L2 flushing, and raw repeated samples.
   - Run A-B-B-A ordering on one physical GPU for each comparison. Report the
     command, source revision, worktree, GPU, correctness state, all raw samples,
     medians, and the direction of `4.6 / 4.5`.
   - A resource-only improvement is not accepted if graph replay regresses.

## Shape matrix

- **W4A16:** both NF3 and E8M0/K32 weight contracts; native small-M, fused,
  activation, and top-k helpers; decode/single-request sizes `1, 2, 4, 8, 23,
  33, 80`; packed prefill sizes `8192, 16384, 24576, 32768` where the model
  profile has capacity.
- **W4A8:** direct M16/M32/M64/M128 and materialized routing/phase1/phase2;
  `w4a8_mx` and `w4a8_nvfp4`; SiLU and ReLU2; decode and packed-prefill routing.
- **Paged attention:** decode plus q lengths `8, 16, 64, 128, 256, 1024`, BF16
  and FP8 KV, direct and split/merge paths, variable replay metadata.
- **MLA/NSA and contiguous attention:** decode, unified prefill, sparse/split,
  merge, indexer/top-k, contiguous and varlen paths.
- **Other production CuTe families:** dense/fused/grouped GEMM, MXFP8 and
  weight-only projection, BF16-to-FP4 quantization, TP-MoE micro/dynamic/tiny,
  and both 4096/7168 residual-mHC decode and prefill branches.

## Structural rule for compiler regressions

The governing principle is to preserve the kernel contract and shorten the
compiler-visible live range, rather than trading correct work or serving
behavior for a lower register number.  Mutable register tensors are fragmented
at their true dependency boundary; control-flow joins are replaced only when
the scalar predicate has identical semantics; shared-memory addresses are
derived from one aligned root; and independent asynchronous pipelines own
independent tail states.  Math, quantization, synchronization, launch geometry,
and planner policy stay unchanged unless a separate correctness and performance
argument justifies changing them.

Every positive register delta remains an accounting exception even if it does
not cross an occupancy boundary.  New local memory, a new 255-register ceiling,
or changed SMEM/launch metadata must first be treated as an unresolved defect.
A smaller spill-free kernel is accepted only after the real graph path is no
slower; conversely, a spill-free positive register delta may be retained when
same-PTXAS evidence identifies the compiler cause, occupancy is unchanged, and
the complete serving benchmark stays inside the A-B-B-A thresholds.  Such a
row is annotated, never silently waived or removed from the table.

### Current family findings

**W4A16.** The original 4.6 lowering of the affected SM120 prefill kernel
reached the 255-register ceiling and emitted a 256-byte stack frame per thread.
Wide mutable FP32 accumulator tensors crossed pipelined/control-flow joins as
vector PHIs.  The retained fix expresses each accumulator element as a scalar
rmem fragment at the true dependency boundary, without changing the W4A16
contract: activations remain BF16, weights remain inline FP4/NVFP4 dequantized,
and no activation-scale math is introduced.

In the fresh same-source census, the shared prefill specialization exercised by
M8192, M16384, M24576, and M32768 improves from 244 to 239 allocated GPRs.  It
has zero frame, stack, local loads, and local stores in both toolchains;
launch SMEM remains 84,992 bytes and one-CTA occupancy is unchanged.  Thus the
original spill is fixed.

The targeted current-source exact-object graph rerun also closes the historical
performance cliff.  On both physical GPUs 4 and 5, every M8192, M16384, M24576,
and M32768 warm- and cold-L2 comparison favors 4.6.0: the 16 trimmed-mean
`4.6 / 4.5.2` ratios span `0.98880--0.98982`, a `1.02--1.12%` improvement.
Each arm contributes 1,000 raw samples per condition; bit-exact arm equality,
the GPU oracle, live-input graph replay, stable addresses, fixed workspace,
zero replay allocation, and before/after object verification all pass.  The
Max-Q software-power-cap bit transitions between `0x0` and `0x4` in some
conditions, so this targeted diagnostic explicitly permits only those two
states while retaining P1, equal memory clocks, and at most 60 MHz SM-clock
delta.  The compact result and raw-report hashes are retained in
`docs/cutlass-46-w4a16-prefill-performance.csv`; the full reports are diagnostic
artifacts rather than a claim that the optional 104-process formal release gate
ran.  Other W4A16 increases, including standalone GEMM `234 -> 238` and
direct-decode `109 -> 110`, remain flagged in the accounting table.

**W4A8.** The fresh resource pair still shows frontend aggregate-copy pressure.
The M32 MX/SiLU row rises `168 -> 192` with no local memory.  The M128
NVFP4/ReLU2 row changes frame/LDL/STL `64/23/16 -> 80/27/20`; two M128 SiLU
rows also change existing local traffic.  These rows remain resource alarms;
no performance disposition is inferred from allocation or spill counts alone.

**Residual-mHC.** The compact 4096/7168 rows rise from 96 to 128 allocated GPRs
without introducing local memory or reducing occupancy.  The SASS evidence
attributes that increase to compiler load hoisting, so it remains flagged even
though a higher register count is not by itself a performance regression.

The separate hidden-7168 block-M false cliff is closed by the retained narrow
F32 materialization at the Gram dependency boundary.  It restores the intended
FFMA dataflow while leaving R128 allocation, zero local memory, and 4,480 bytes
of launch SMEM unchanged.  Fresh same-source A-B-B-A checks pass correctness,
graph, and stability gates on both target GPUs; observed `4.6 / 4.5.2` ratios
span approximately `0.980--1.003` on GPU 4 and `0.989--1.001` on GPU 5.  This
closes that targeted finding without standing in for the optional full-matrix
release workflow.

**Other families.** The exhaustive table retains every exact positive metric,
including W4A8, TP-MoE, MLA, paged-attention merge, dense fused-quantization,
and diagnostic-entry rows.  Targeted runtime work is selected from known
cliffs, changed source, new ceilings/local memory/SMEM/occupancy, and material
resource shifts; unchanged low-risk rows are not mechanically subjected to the
full runtime cross product.  A row without fresh GPU evidence remains labeled
pending rather than inheriting a result from an older source snapshot.

CUTLASS 4.6 also changes `PipelineAsync.producer_tail` from the old final
acquire sequence to a wait over every stage that advances the supplied pipeline
state after each wait.  Contiguous attention reused one lockstep state for its K
and V pipelines and called K tail followed by V tail.  For its one-stage
pipeline, K therefore flipped the phase before V observed it; V waited forever
on an empty-barrier phase that no consumer could signal.  The generated SASS
shows the phase XOR immediately between the K and V tail waits.  Passing an
independent state clone to each tail fixes the deadlock without changing the
pipeline's steady-state lockstep behavior.  The same repair is required at the
latent paged-extend K/V tail site.  The general rule is that a 4.6 tail consumes
and mutates its state: never pass one state object to tails for multiple
pipelines, even when those pipelines advance together in the main loop.

Shared-memory allocation is a real 4.6 compiler-contract change.  In 4.5,
`SmemAllocator` requested one 1,024-byte-aligned dynamic-SMEM base pointer,
manually aligned and advanced that pointer for each object, and tracked a
Python-side `_allocated_bytes` total.  In 4.6 the allocator constructor has no
base pointer.  Every allocation instead becomes a typed `cute.memref.alloca`
with its own alignment and optional swizzle metadata, is assigned to the USER
SMEM partition, and is laid out by the compiler's SMEM inference pass.  Kernel
launch then queries the inferred partition usage and supplies it when `smem` is
omitted.

The migration rule is therefore to describe every shared object, alignment,
swizzle, and lifetime through the 4.6 allocator and let the compiler resolve
the partition; old hand-computed `.launch(smem=...)` values must not remain as
a second source of truth.  Equal total bytes are necessary but not sufficient:
the new typed allocation/address lowering can change register pressure even
when the inferred launch size is identical.  That possibility must be tested
per specialization rather than inferred from byte totals.  Production launches
use the inferred value, which is extracted
from the fresh final LLVM launch configuration for every specialization.
Cubin `.nv.shared` section size is reported independently and is never added
to, or substituted for, launch-time dynamic SMEM.  A change in either value is
a hard comparison failure until its source layout and launch occupancy have
been checked.

## Resource annotation taxonomy

For the optional formal release gate, generate
`docs/cutlass-46-resource-annotations.csv`, keyed by the exact
`semantic_key,symbol_sha256` pair, with four nonempty fields for every
exceptional final row.  It is deliberately not fabricated for the lean
closeout: `docs/cutlass-46-register-accounting.csv` remains the complete alarm
inventory, and unresolved rows stay visible there.  Use short controlled
prefixes followed by row-specific detail; never use a family-wide annotation
that hides specialization-specific deltas.

- `cause`: one of `frontend:wide-phi-pack`,
  `frontend:aggregate-copy`, `frontend:smem-address-lifetime`,
  `frontend:predicate-schedule`, `frontend:load-hoist`,
  `assembler:allocation`, `source:launch-geometry`,
  `source:typed-smem-contract`, `source:pipeline-tail-semantics`, or
  `unresolved:<scope>`.  Use `assembler:allocation` only when identical PTX
  assembled with different PTXAS versions reproduces the delta; use a
  `frontend:` cause when common-PTXAS output preserves it.
- `disposition`: `fixed-residual-flagged` when the defect was removed but a
  positive metric remains, `retained-beneficial` or `retained-neutral` only
  after the required targeted GPU gates (and the full matrix when selected),
  and `pending-fix` or `blocking-regression` otherwise.
  Rejected canaries belong in the evidence text, not as the disposition of a
  final-source row.
- `evidence`: record the exact same-source resource/SASS row, common-PTXAS or
  PTX/LLVM/SASS diff that establishes cause, correctness node IDs, and raw
  GPU-4/GPU-5 graph A-B-B-A artifact names.  Include the source/package
  fingerprint and ratio direction.  Static instruction evidence alone cannot
  justify `retained-*`.
- `performance_status`: use `beneficial:gpu4,gpu5`, `pass:gpu4,gpu5`,
  `regressed:<shapes>;pending-fix`, or `pending:<missing-gates>`.  A row with a
  measured cliff remains pending even if its resource delta is small, and a
  positive register delta can pass only when correctness, graph serving, and
  both physical GPUs pass.

Current high-signal cause mapping is: the fixed W4A16 spill to
`frontend:wide-phi-pack`; W4A8 M32/M128 to `frontend:aggregate-copy`; and the
compact residual 4096/7168 positives to `frontend:load-hoist`.  The hidden-7168
block-M issue is separately closed by its source-DAG/F32 materialization fix.
Other positive rows keep their row-specific reviewed cause or remain
`unresolved:<scope>`; they do not inherit a family-wide explanation.  Typed
SMEM describes the 4.6 allocation contract but is not a default explanation
for an unrelated register delta.

Graph benchmarks keep the in-process compiled-program cache enabled.  Disabling
it can trigger program reload/initialization during capture and invalidate the
graph; that is not a serving configuration.  Each benchmark performs a
correctness replay in the same cold/hot cache state before recording any timing,
completes L2 eviction outside the timed interval, and writes every replay plus
worktree/commit/toolchain/GPU UUID provenance with `--raw-samples-jsonl PATH`.
The comparator consumes the individual replay samples (not only repeat
medians), verifies their count and microsecond unit, and requires the four runs
to record stable allocations, fixed/preplanned workspace capacity, and the same
physical GPU.
