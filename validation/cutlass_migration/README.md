# CUTLASS migration qualification

This package is the durable qualification and evidence system for CUTLASS DSL
compiler migrations. It is not a collection of ordinary benchmarks.

The release path is GPU-only. It exercises the real production launch paths on
the explicitly selected physical GPUs, binds exact compiled artifacts, checks
correctness and CUDA-graph invariants before timing, and fails closed on missing
kernels, specializations, shapes, or evidence. Offline integrity checks validate
the machinery but never count as kernel acceptance.

## Layout

- `acceptance/`: corpus capture, isolated single-arm producers, and the
  separate-process end-to-end release gates.
- `evidence/`: SMEM, resource, SASS, register-accounting, source-inventory, and
  specialization-contract builders.
- `diagnostics/`: paired ABBA and compiler root-cause experiments. These are
  retained for investigation but do not satisfy release acceptance.
- `integrity_checks/`: offline negative checks of schemas, timers, producer
  contracts, and exact per-test Nsight/NVTX CUDA-launch ownership. They are
  evidence-infrastructure checks, not CPU kernel tests.
- `data/`: reviewed closed-world corpus, producer, source, and symbol contracts.
- `core/`: shared GPU scope, timing, identity, and single-arm primitives.

General-purpose performance benchmarks remain under `benchmarks/`. GPU pytest
cases remain under `tests/`; the corpus runner selects and launches them in fresh
GPU-bound processes.

## Default lean closeout

The default migration closeout runs one exhaustive same-source static census
over every reviewed production specialization and CUDA entry point. It compares
allocated GPRs, exact R/UR/P/UP use, local-memory metrics, SMEM, occupancy,
instruction count, and code size, and fails closed on missing or unmatched
rows. GPU runtime checks are then targeted at changed implementations, known
performance cliffs, and resource changes with credible execution impact. They
still require correctness and the real exact-object CUDA-graph serving path on
physical GPUs 4 and 5; a static improvement is never presented as a timing
result.

Exact throttle-mask matching remains the default for formal evidence. A paired
diagnostic may opt into the explicitly recorded `0x0`/`0x4` Max-Q
software-power-cap transition only when P1, device identity, memory clock, and
the configured SM-clock envelope still pass. This avoids retry loops around a
non-semantic status-bit flicker without accepting thermal or hardware-brake
throttling.

The formal full-matrix machinery is intentionally preserved but is optional
for routine closeout. Selecting it expands all 13 families across both GPUs and
all A1/B1/B2/A2 positions into 104 independent process results. The lean
default does not run that cross product merely to repeat unchanged, low-risk
runtime coverage.

## Entry point

Run the complete command inventory with:

```bash
python -m validation.cutlass_migration --help
```

Every leaf command forwards its remaining arguments to the underlying tool, so
the leaf's own help is available in the usual way, for example:

```bash
python -m validation.cutlass_migration evidence smem --help
python -m validation.cutlass_migration acceptance corpus --help
python -m validation.cutlass_migration diagnostic paired w4a16_serving --help
```

Family discovery is a two-step offline review boundary. Each GPU producer
writes a hashed `b12x.cute.migration.family_discovery.v1` fragment. Assembly
requires exactly one fragment for every closed-set family, converts exact cache
paths into immutable file records, and writes a pending
`b12x.cute.migration.end_to_end_contract_discovery.v1` artifact. A separate
command must revalidate and stamp that artifact with an explicit review ID:

```bash
python -m validation.cutlass_migration acceptance discovery assemble \
  --side current --source-manifest "$SOURCE_MANIFEST" \
  --expected-physical-gpu 4 \
  --fragment "$FAMILY_FRAGMENT_1" \
  --fragment "$FAMILY_FRAGMENT_2" \
  --output "$PENDING_DISCOVERY"
# Repeat --fragment exactly once for every required family.

python -m validation.cutlass_migration acceptance discovery review \
  --input "$PENDING_DISCOVERY" --source-manifest "$SOURCE_MANIFEST" \
  --review-id "$REVIEW_ID" --output "$REVIEWED_DISCOVERY"
```

When the optional formal full-matrix workflow is selected, construct the final
closed evidence manifest from a dedicated run tree after the reviewed contract
and all isolated A1/B1/B2/A2 processes are complete. The constructor requires
all 13 families on both physical GPUs in all four positions (104 independent
process results), verifies their embedded hashes and contract binding, and
rejects extra, duplicate, diagnostic, or partial inputs:

```bash
python -m validation.cutlass_migration acceptance evidence-set \
  --contract "$CONTRACT" --run-root "$FINAL_RUN_ROOT" \
  --output "$EVIDENCE_SET"
```

The paired diagnostics and integrity checks are deliberately labeled. Targeted
diagnostics support row-level dispositions in the lean workflow, but only the
GPU acceptance and final release-gate commands can produce formal full-matrix
acceptance evidence.
