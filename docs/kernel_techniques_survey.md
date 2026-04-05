# Kernel Techniques Survey

Scope:
- `b12x/moe/fused/{static,micro,dynamic}.py`
- `b12x/attention/paged/{forward_paged,forward_extend_generic,merge}.py`
- shared CuTe DSL support used by both families:
  - `b12x/cute/fp4.py`
  - `b12x/cute/utils.py`
  - `b12x/gemm/dense.py`

This is a survey of the kernel programming techniques actually used in the current code, with emphasis on CuTe DSL patterns, low-level control flow, data movement strategy, and reusable architectural ideas.

## 1. Core Architectural Pattern Across The Repo

The dominant pattern is "host planner + device resident control plane + literal tensor-core inner loops".

Common traits:
- The host computes enough metadata to make the device schedule regular or at least bounded.
- The kernel owns the hot loop and often also part of the control plane.
- Shared memory and register layouts are treated as first-class design objects.
- The implementation prefers literal MMA / ldmatrix / cp.async / TMA driven code over opaque library calls.
- CuTe layout algebra is used to describe storage and movement, but inline PTX is used wherever CuTe or CUTLASS DSL does not expose the needed primitive.

The repo is not using CuTe DSL as a "pretty syntax for CUDA". It is using CuTe as:
- a layout compiler
- a typed codegen front-end
- a pipeline builder
- a way to stage descriptors and shared-memory aliases

and then filling the gaps with `@dsl_user_op` inline assembly.

## 2. Cross-Cutting Techniques

### 2.1 Inline PTX via `@dsl_user_op`

This is one of the most important recurring techniques.

Used for:
- global/shared loads and stores with exact instruction forms
- memory ordering (`acquire`, `release`, fences)
- cluster shared-memory addressing
- atomics not conveniently exposed by CuTe
- transcendental approximations
- `cp.async` variants and TMA issue helpers
- specialized FP4/FP8 conversion and packing support

Representative files:
- `b12x/cute/fp4.py`
- `b12x/attention/paged/merge.py`
- `b12x/attention/paged/forward_extend_generic.py`
- `b12x/distributed/pcie_oneshot.py`

Design takeaway:
- CuTe DSL is the structural layer.
- Inline PTX is the performance and semantics escape hatch.

### 2.2 Explicit Memory-Space And Alignment Modeling

Common moves:
- `make_ptr(...)` wrappers carrying dtype, address space, alignment
- `cute.assume(..., divby=...)` to strengthen alignment facts for codegen
- explicit reconstruction of typed views over byte payloads
- "flat backing store + logical typed view" for packed activations and block scales

Examples:
- `packed_a_storage` / `scale_storage` in MoE
- payload slicing and typed aliasing in paged attention shared storage
- page-oriented TMA source tensor reshaping in paged attention

This is not cosmetic. The code relies on these facts to make:
- vectorized 128-bit accesses legal
- TMA descriptor lowering work
- swizzled shared-memory layouts compile correctly

### 2.3 Persistent CTAs / Resident Grids

This shows up in both MoE and paged attention merge.

Patterns:
- size grid by useful work or max active clusters, not by naive output tiles
- use a persistent work loop inside the kernel
- maintain scheduler state explicitly instead of relying on launch geometry alone

Examples:
- static MoE uses resident CTAs plus a route/pack -> compute barrier
- dynamic MoE uses a ready-task queue
- paged merge uses persistent CTAs over `(row, head)` work and `griddepcontrol`

Why it matters:
- lets kernels absorb control-plane work without extra launches
- improves reuse of staged state
- makes small decode shapes more viable

### 2.4 Producer / Consumer Pipelining

There are several pipeline forms in the repo:

1. TMA async pipelines
- `pipeline.PipelineTmaAsync.create(...)`
- producer/consumer cooperative groups
- explicit producer and consumer states

2. Simple mbarrier-driven single-stage pipelines
- used when the full CUTLASS pipeline abstraction is too heavy or the flow is special-case

3. Named warp-group barriers
- especially in dense GEMM support for coordinating DMA vs MMA vs epilogue phases

The repo uses pipeline constructs as real scheduling devices, not abstractions for convenience.

### 2.5 Warp Specialization

Common split:
- one warp (or a small set) handles DMA / descriptor issue / task management
- the rest do MMA and epilogue math

Examples:
- dense GEMM skeleton: dedicated TMA load warp
- dynamic MoE: compute warps plus a control warp / CTA leader behavior
- paged attention: Q-warps, KV-warps, and special staging flows depending on kernel variant

### 2.6 Multi-Level Synchronization

The code uses several distinct synchronization mechanisms, each for a different scope:

- `cute.arch.sync_threads()` for CTA-wide sync
- `pipeline.NamedBarrier` for warp-group phase sync
- `cute.arch.mbarrier_*` for async copy completion and cluster reductions
- resident-grid barrier patterns in MoE static
- `griddepcontrol_wait/launch_dependents` in paged merge

Important point:
- synchronization is scoped very intentionally
- the code is not leaning only on block-wide barriers

### 2.7 Data Movement As A First-Class Design Problem

The kernels are designed around minimizing or restructuring data movement:
- route/pack directly into expert-major FP4 contract
- stage paged KV with TMA or manual async copy
- keep intermediate tiles in shared memory for FC2 reuse
- use swizzled layouts to match tensor-core consumption directly

This is the main architectural theme of the repo.

## 3. CuTe DSL Specific Idioms

### 3.1 `@cute.jit` vs `@cute.kernel`

Usage pattern:
- `@cute.jit` for reusable code fragments, microkernels, helpers, layout-sensitive code
- `@cute.kernel` for the actual launchable entrypoint

This lets the code treat the kernel as a composition of smaller generated building blocks rather than one giant function.

### 3.2 Layout Algebra Everywhere

Common operations:
- `cute.make_layout`
- `cute.select`
- `cute.make_tensor`
- `cute.tile_to_shape`
- `cute.recast_tensor`
- `cute.flatten`
- `retile(...)`

This is how the code expresses:
- tensor-core fragment layout
- swizzled shared storage
- TMA source and destination shapes
- packed scale layout expectations

### 3.3 Fake Tensors And Compile-Time Cache Keys

The integration layer uses fake tensors plus `cute.compile(...)` to build and cache specialized kernels.

This shows up strongly in `tp_moe.py`:
- fake compact tensors
- compile cache keys including shapes, dtypes, implementation flags, and feature toggles

This is a very CuTe DSL specific codebase pattern:
- compile many specialized kernels lazily
- keep feature flags in the cache key
- use fake tensor shapes to drive specialization

### 3.4 Runtime Pointer Interop

`b12x/cute/utils.py` defines a runtime pointer wrapper so compiled CuTe kernels can cleanly accept:
- raw addresses
- ctypes pointers
- memory-space aware typed pointers

This is important because the repo frequently mixes:
- Torch tensors
- IPC-opened memory
- descriptor tables
- byte payload aliases

### 3.5 Custom MLIR Value Plumbing

Dynamic MoE includes custom runtime launch state objects with:
- `__extract_mlir_values__`
- `__new_from_mlir_values__`

This is a notable CuTe DSL specific technique for passing structured runtime state into generated code.

## 4. Fused MoE Kernel Family

## 4.1 Static MoE

Main file:
- `b12x/moe/fused/static.py`

Main techniques:
- resident-grid two-phase kernel
- route/pack and compute in one launch
- expert-major packed FP4 activation contract
- blockscaled GEMM consumption
- cooperative FC2 reuse from shared intermediate slices

Frontend techniques:
- atomically append expert rows
- write `token_map` and `token_weights` during routing
- quantize each routed row inline into packed activation storage

Backend techniques:
- grouped static scheduler over `(expert, m_tile, output_tile)`
- FC1 once per slice, FC2 sweep over all output tiles
- direct scatter via BF16x2 atomics

Why static is interesting:
- it already contains the Phase-1 route/pack producer pattern needed for the planned fused pre-MLP producer
- it is the clearest example of "control plane fused into compute kernel"

## 4.2 Micro MoE

Main file:
- `b12x/moe/fused/micro.py`

Role:
- same basic control-plane idea as static
- specialized for tiny routed working sets

Typical techniques:
- same packed-input contract
- same route/pack and compute fusion
- more aggressive specialization to tiny shapes

The micro kernel is less conceptually different than static; it is mostly a specialization strategy.

## 4.3 Dynamic MoE

Main file:
- `b12x/moe/fused/dynamic.py`

Main technique:
- global ready-task queue replaces the static route/pack -> compute barrier

Important patterns:
- CTAs start as producers
- routed rows are appended and quantized immediately
- when an `(expert, tile)` is complete, a compute task is published
- producers turn into consumers dynamically

Low-level synchronization details:
- acquire/release global loads/stores
- explicit spin polling
- global task queue metadata
- conservative one-CTA-per-SM model

This is the repo's clearest example of moving from a resident barrier model to a queue-driven model.

## 4.4 Shared MoE Support Layer

`b12x/cute/fp4.py` is not just "FP4 math helpers". It is the kernel support library for much of the repo.

It contains:
- 128-bit vectorized load/store ops
- cluster-shared helpers
- atomics
- BF16x2 scatter helpers
- FP4 quantization primitives
- warp and cluster reductions
- ldmatrix and tensor-core helper code
- scale swizzle/oracle helpers used in tests and benchmarks

This file is the main low-level vocabulary that the fused MoE kernels speak.

## 5. Paged Attention Kernel Family

## 5.1 Forward Paged

Main file:
- `b12x/attention/paged/forward_paged.py`

Core techniques:
- host planner worklists drive kernel launch
- staged paged K/V ingress
- literal QK and PV tensor-core inner loops
- support for BF16 and FP8 KV storage
- base-2 LSE representation to match split-state merge

Notable implementation details:
- multiple K/V ingress modes:
  - TMA based
  - manual cp.async based
  - raw FP8 special issue path
- explicit plane layouts and swizzle handling
- payload byte-buffer slicing into typed shared-memory aliases
- manual fragment dump/debug helpers

This file is one of the best examples in the repo of combining:
- CuTe layout programming
- manual PTX
- multiple ingress backends under one kernel contract

## 5.2 Forward Extend Generic

Main file:
- `b12x/attention/paged/forward_extend_generic.py`

Role:
- historical / generic extend kernels
- very similar design vocabulary to `forward_paged.py`

Key techniques:
- custom bulk tensor async issue helpers
- TMA descriptor prefetch
- mbarrier-managed staging
- pipeline producer/consumer states
- literal BF16 and FP8 MMA loops
- split-plane raw FP8 handling

This file is useful because it contains several variants of the same underlying ideas, making the design space more explicit than the primary kernel sometimes does.

## 5.3 Merge

Main file:
- `b12x/attention/paged/merge.py`

Core techniques:
- persistent CTAs over split-state merge work
- base-2 log-sum-exp arithmetic
- explicit state objects (`m`, `d`, partial `o`)
- direct-grid vs persistent-grid options
- `griddepcontrol_wait` / `griddepcontrol_launch_dependents`

This file is an example of:
- control-flow-heavy kernel programming without tensor-core dominance
- persistent scheduling and arithmetic-state fusion

It is less about MMA micro-optimization and more about:
- scheduling
- numerical state merging
- dependency control

## 6. Dense GEMM Support Skeleton

Main file:
- `b12x/gemm/dense.py`

This file is the reusable dense blockscaled GEMM scaffold from which a lot of the MoE compute style is inherited.

Important techniques:
- persistent tile scheduler
- TMA async pipeline with producer and consumer groups
- dedicated DMA warp
- named barriers for mainloop and epilogue sync
- shared-memory layout generation for A/B/SFA/SFB
- descriptor prefetch
- cluster-aware scaffolding, even though current target keeps cluster `(1,1,1)`

This file is important because it demonstrates the repo's "canonical" way to build:
- a high-throughput blockscaled GEMM in CuTe DSL
- with reusable staging and scheduling components

## 7. What The Codebase Is Actually Good At

The current codebase is especially strong in these areas:

1. Fusing control plane into kernels
- route/pack inside MoE kernels
- merge scheduling inside merge kernel

2. Treating layout and movement as the primary optimization target
- swizzled scale layouts
- TMA source reshaping
- payload aliasing

3. Filling CuTe DSL gaps safely with low-level PTX
- the support layer is extensive and deliberate

4. Keeping the literal tensor-core math visible
- no black-box GEMM calls in the hot path
- the MMA loops are inspectable

## 8. Relevance To The Planned Pre-MLP Producer

For the planned two-kernel architecture:

Kernel 1:
- allreduce + residual + Gemma RMSNorm
- router logits + top-k
- expert row assignment
- packed NVFP4 writeout

Kernel 2:
- existing prequantized MoE consumer

The survey suggests the right implementation strategy is:

1. Reuse the route/pack frontend pattern from `static.py` / `micro.py`
- do not preserve the current Python/reference-style prequantized workspace population path

2. Reuse the low-level support from `b12x/cute/fp4.py`
- especially quantization, vectorized stores, atomics, and layout helpers

3. Keep the consumer boundary exactly where it is now
- packed input
- scale storage
- token map / token weights
- row counts and expert maps

4. Treat the new producer as a control-plane fusion kernel
- much closer in spirit to MoE static Phase 1 than to the current host populator

5. Prefer one additional producer kernel over a large host/device mixed path
- this is aligned with the rest of the repo's design language

## 9. Bottom Line

The repo already contains the important ingredients for the next step:
- a real fused route/pack frontend in the static and micro MoE kernels
- a queue-based control-plane alternative in dynamic MoE
- a reusable blockscaled GEMM scaffold
- a very capable low-level CuTe DSL + PTX support layer
- attention kernels that show how to build sophisticated staged ingress and persistent scheduling with the same vocabulary

The main missing piece is not technique. It is integration:
- moving the MoE route/pack frontend earlier so it becomes the producer for the prequantized consumer contract
- and doing that without roundtripping through a generic BF16 activation buffer.
