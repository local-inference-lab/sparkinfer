# sparkinfer

`sparkinfer` (formerly `b12x`) is an SM120/SM121 CuTe DSL kernel library for LLM
inference. It specifically targets DGX Spark, RTX Spark and the
Blackwell-based RTX cards (RTX 6000 Pro, RTX 5090).

## Install

```bash
pip install sparkinfer
```

You need Python 3.10+, `torch >= 2.12`, and an SM120/SM121 GPU. The CuTe DSL
compiler and its CUDA 13 libraries come in as wheel dependencies
(`nvidia-cutlass-dsl == 4.6.0`), so there is no build step — kernels are
JIT-compiled on first use and cached.

## What's in here

Every kernel is one op at `sparkinfer.<group>.<op>` (15 total; `list_ops()`
enumerates them). The op owns its `plan`/`bind`/`run` facade in `api.py`; the
kernel guts sit in `_impl.py`/`_kernel.py`; cross-op lowering lives in
`<group>/_shared/` and the universal compile/scratch spine in `sparkinfer/_lib/`.

**`gemm`** — a dense block-scaled GEMM (NVFP4/MXFP8 operands, BF16/FP16/FP32
out) plus fused linears on top of it: `gemm.blockscaled` (one-shot), MXFP8
(`gemm.mxfp8_linear`), 128×128 block-FP8 (`gemm.block_fp8_linear`), and the
grouped WO-projection (`gemm.wo_projection`) used by MLA attention output.

**`attention`** — `attention.paged` (paged-KV decode/extend, FP8 KV, MSA block
sparse, CUDA-graph-replayable), `attention.sparse_mla` and
`attention.compressed_mla` (top-k / compressed-page MLA — distinct contracts,
kept separate on purpose), `attention.nsa_indexer` (the NSA/MSA quantize →
score → select pipeline), and `attention.varlen` (contiguous batched/varlen).

**`moe`** — `moe.fused_moe`, fused FP4 TP MoE across a micro-kernel decode
path, a unified dynamic path (persistent grid, `nvfp4`/`w4a8_mx`/`w4a8_nvfp4`),
and W4A16 (BF16 activations, inline FP4 weight dequant — no activation-scale
math), with SiLU/ReLU2/SwiGLU-OAI activations; plus `moe.ep_moe` (expert
parallel).

**the rest** — `norm.mhc` (fused RMSNorm + hyper-connection residual),
`quantization.{nvfp4,mxfp8}` (row quantizers), and `comm.pcie` (IPC-backed PCIe
collectives). `sparkinfer` owns planning, scratch layout, and policy, so
serving stacks only supply metadata and capacity limits.

## Using it

Every stateful kernel lives at `sparkinfer.<group>.<op>` and shares the **same
shape** — `plan` the work, size scratch from the plan, `bind` your tensors as
views, `run`. The module path carries the context, so the verbs and role
classes (`Caps`/`Plan`/`Binding`) are uniform across families:

```python
# norm — fused RMSNorm + hyper-connection residual mixing
from sparkinfer.norm import mhc

plan    = mhc.plan(mhc.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = mhc.bind(plan, scratch=scratch, ...)
residual, post, comb, y = mhc.run_post_pre(..., binding=binding)
```

```python
# moe — fused tensor-parallel routed-expert FFN (weights prepped once per model)
from sparkinfer.moe import fused_moe

wplan   = fused_moe.plan_weights(quant_modes="nvfp4",
                                 source_format="modelopt_nvfp4", ...)
experts = fused_moe.prepare_weights(plan=wplan, ...)
plan    = fused_moe.plan(fused_moe.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=experts,
                         topk_weights=tw, topk_ids=ti)
out     = fused_moe.run(binding=binding)
```

```python
# attention — MLA decode from compressed KV pages (DeepSeek-V3.2)
from sparkinfer.attention import compressed_mla

plan    = compressed_mla.plan(compressed_mla.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = compressed_mla.bind(plan, scratch=scratch, q=q,
                              swa_indices=idx, swa_lengths=lens, ...)
out = compressed_mla.run(swa_k_cache=swa, binding=binding, sm_scale=scale, ...)
```

`plan` is host-side and may allocate; `bind` only narrows/views (never
allocates), which is what makes captured graphs safe; `run*` executes and is
CUDA-graph-capture safe. One-shot ops (`gemm.blockscaled.mm`,
`quantization.mxfp8.quantize_rows`) are plain functions; `comm.pcie`
collectives are stateful classes. `sparkinfer.list_ops()` enumerates the full
set; every op exports `is_supported()`. Underneath, kernels register as torch
custom ops in the private `sparkinfer::` namespace (torch.compile / CUDA-graph
integration) — prefer the Python API.

Compilation happens lazily per shape/config and is cached. For serving, warm
up the shapes you need, then freeze:

```python
import sparkinfer

# ... run warmup traffic covering every shape you will serve ...
sparkinfer.freeze_kernel_resolution("serving")
```

After the freeze, any request that would trigger a new kernel compile raises
`KernelResolutionFrozenError` instead of stalling a live request (or worse,
compiling inside CUDA graph capture).

Set `SPARKINFER_PRINT_COMPILE_PROGRESS=1` to log each compiler invocation with its
cache-key parameters and duration — useful for figuring out what warmup
actually covered. `SPARKINFER_TIMING=1` enables per-kernel timing logs.

## Where to look next

- `tests/` is the executable spec — per-group API and numerical-reference
  tests showing exact tensor layouts and `plan`/`bind`/`run` call sequences.
  (`tests/_legacy/` holds the pre-namespace flat-API suite, being migrated.)
- `benchmarks/` has tuned invocations per kernel family (and `probe_*` scripts
  from tile-sweep experiments).
- `docs/` has design notes: the MoE execution model, the eager-plan-bind
  architecture, and an SM120 MLA postmortem.

Failing that, ask your friendly neighborhood AI agent — it does fine here.
