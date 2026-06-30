# MoE execution model and scheduler convergence

All kernels implement the same logical graph:

1. route each token to `top_k` experts;
2. compute expert-local FC1;
3. apply the activation and any intermediate quantization;
4. compute expert-local FC2;
5. reduce weighted routes to token order.

The public `quant_mode` names historically bundled numeric semantics, source
storage, prepared storage, scheduling, and graph partitioning.  `MoESpec` and
`MoEExecutionPlan` in `b12x.moe.execution` name those choices independently.

## Independent axes

`MoESpec` describes semantics and numerics:

- checkpoint source format and source scale encoding;
- activation operand and scale encodings;
- compute-time weight scale encoding;
- activation, IO and accumulator types, W13 order, and router-weight placement.

`MoEExecutionPlan` describes a lowering:

- route layout: direct top-k, append-only expert rows, or sorted/padded rows;
- work availability: inline, precomputed, or streamed;
- scheduler: direct, persistent grid, materialized atomic queue, or ready queue;
- GEMM engine and prepared weight layout;
- output reduction and tile geometry.

The canonical numeric recipes are:

| Recipe | Activation operand / scale | Weight source scale | Compute weight scale |
|---|---|---|---|
| NVFP4 W4A4 | FP4 E2M1 / E4M3 K/16 | E4M3 K/16 | E4M3 K/16 |
| W4A8 on NVFP4 | MXFP8 E4M3 / E8M0 K/32 | E4M3 K/16 | E8M0 K/32 base × K/16 residual |
| native W4A8-MX | MXFP8 E4M3 / E8M0 K/32 | E8M0 K/32 | E8M0 K/32 |
| W4A16 | BF16 / none | E4M3 K/16 or E8M0 K/32 | source-preserving |

The current kernel families map onto those axes as follows:

| Family | Route layout | Availability | Scheduler | Weight memory |
|---|---|---|---|---|
| direct micro W4A4 / W4A8-on-NVFP4 | direct top-k | inline | direct | source-native |
| unified dynamic W4A4/W4A8 | append-only expert rows | precomputed, or experimental streaming | atomic queue, persistent grid, or ready queue | MMA views; native W4A8 uses N256/K128 QMMA repack |
| W4A16 tensor-core | sorted/padded; direct at small M | precomputed; inline at small M | persistent grid | MMA-packed |

## Why queue versus grid exists

There are two scheduler questions:

1. **When is a compute tile knowable?** This determines whether the work
   source must understand readiness.
2. **How uniform is the cost of the knowable tiles?** This determines whether
   fixed arithmetic ownership or dynamic work stealing is preferable.

The production dynamic kernel histograms routes, appends expert rows, crosses a
resident-grid barrier, materializes every work item, and crosses a second
barrier before expert compute.  Its consumer therefore sees a complete,
addressable domain.  An atomic claim is not enforcing a data dependency; it is
load balancing already-published work.

That makes both persistent-grid assignment and a materialized queue legal.  It
does not make them equally fast.  Partial expert tiles, differing slice counts,
memory locality, and scatter contention give fused items enough cost variance
that work stealing often wins.  Where a phase becomes rectangular and
equal-cost, arithmetic assignment wins: native W4A8's materialized FC2 phase
uses a flattened grid-stride domain after the queued FC1 phase.

The resulting rule is:

- streamed publication requires a readiness-aware queue;
- materialized work permits either a grid or a non-readiness queue;
- choose per phase from measured cost variance, not activation precision.

## Common work-source contract

The dynamic consumer acquires one register-resident item:

```text
DynamicWorkItem
  expert
  m_tile
  slice_begin, slice_count
  valid_rows
```

The acquisition policy is compile-time specialized:

```text
PersistentRangeSource   stride through an arithmetic materialized domain
MaterializedQueueSource atomically claim from a complete domain
ReadyQueueSource        pop incrementally published items
```

All policies feed the same FC1/activation/FC2 body.  Materialized grid and
queue sources derive most metadata arithmetically; the ready source loads all
fields because publication order is not arithmetic.  The item lives in a CuTe
rmem tensor so roles do not repeatedly reload shared control state.

This is the useful unification boundary.  W4A4, W4A8, and W4A16 need not share
one GEMM mainloop: their operand transport and MMA contracts are genuinely
different.  Route readiness and work ownership no longer need to be entangled
with those arithmetic choices.

## Native W4A8 convergence

Native W4A8 now uses one dynamic-kernel family across the full routed range.
Its dense specialization combines:

- token-major input quantization with expert-major route metadata;
- N256/K128 lane-major prepared weights;
- M32 FC1 work with queue or persistent-grid acquisition;
- a materialized E4M3 intermediate with transposed scale planes;
- flattened grid-stride full-K FC2 work;
- register-resident work metadata and compile-time-specialized control flow.

The standalone staged pipeline was removed after the unified kernel won every
tested common DSV4 point on GPU 9 with `benchmark_moe.py`, CUDA graph replay,
and L2 flushing:

| TP | Tokens | Unified dynamic (µs) | Removed staged baseline (µs) | Change |
|---:|---:|---:|---:|---:|
| 2 | 256 | 1715.2 | 1891.9 | -9.3% |
| 2 | 512 | 1855.6 | 1901.1 | -2.4% |
| 2 | 768 | 1763.9 | 1920.0 | -8.1% |
| 2 | 1024 | 1905.7 | 1948.7 | -2.2% |
| 2 | 2048 | 2405.5 | 2617.4 | -8.1% |
| 4 | 256 | 923.6 | 1026.6 | -10.0% |
| 4 | 512 | 969.8 | 1026.1 | -5.5% |
| 4 | 768 | 936.5 | 1033.2 | -9.4% |
| 4 | 1024 | 1003.6 | 1061.4 | -5.4% |
| 4 | 2048 | 1300.1 | 1464.9 | -11.2% |

The geometric-mean improvements are 6.1% at TP2 and 8.3% at TP4.  Removing
the old path also removes its 48-row route padding, capacity workspace,
standalone input quantizer, activation/quantizer, grouped GEMMs, inverse-route
map, weighted-sum kernel, dispatch thresholds, and graph-warmup contract.

## Scheduler measurements beyond W4A8

Direct arithmetic decoding is still useful when the fused task domain is
regular.  The true persistent-grid specialization removed queue-control
broadcasts and reduced executed instructions for W4A4.  At DSV4 batch 1024 it
improved TP4 from about 1153 to 1134 µs and TP2 from about 2072 to 2051 µs.

It is not universally better.  At TP2 batch 2048 the grid measured about 2959
µs versus 2802 µs for the queue.  This is the same rule at a different point:
static ownership is best in the regular middle band, while work stealing
absorbs the large-domain tail.  Scheduler selection is therefore a shape and
phase policy under one kernel, not a separate precision-specific kernel.

## One-owner API

Preparation is a lifecycle state, not a promise of a second allocation. The
public path has one direction:

```text
plan weights -> prepare one expert owner -> plan scratch -> bind -> run
```

`plan_b12x_fp4_moe_weights` chooses the representation and storage policy.
`prepare_b12x_fp4_moe_weights` executes that decision once and returns the sole
`B12XFP4ExpertWeights` owner. Scratch and launch planning derive source format,
activation, dtype, geometry, and W13 order from that owner's plan. Binding takes
the owner—not raw weight tensors—and execution takes only the completed binding.

| Requested runtime recipe | Planned storage | Preparation action |
|---|---|---|
| NVFP4 or W4A8-on-NVFP4 | source-native bytes with MMA views | keep the source allocation; derive runtime alphas |
| W4A16 source-native | source-native bytes | transfer the source allocation into the expert owner |
| W4A16 MMA-packed | packed MMA layout | repack the source allocation in place |
| native W4A8-MX | N256/K128 QMMA weights plus SFB scales | repack weights and scales in place |
| source-native plus an incompatible model-sized repack, or two incompatible repacks | none | reject during planning |

There is no runtime raw-weight overload, prepared-payload override, old cache-key
fallback, or source-plus-repack serving state. Internal representation payloads
are not exported from `b12x.integration`.

## End state

The code now has:

1. one semantic/execution vocabulary;
2. one rmem work-item interface in the dynamic kernel;
3. compile-time queue, persistent-grid, and ready-queue acquisition policies;
4. one native-W4A8 prepared representation consumed by unified dynamic;
5. phase-local scheduling where rectangular work justifies it;
6. one planner-owned preparation and execution API with one weight owner.

The remaining specialization is intentional: precision selects operand
transport and MMA instructions; availability constrains legal schedulers; and
measured cost variance selects among those schedulers.
