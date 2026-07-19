# b12x

`b12x` is an SM120/SM121 CuTe DSL kernel library for (primarily) NVFP4 LLM
inference — the Blackwell GeForce and workstation parts (RTX 50-series,
RTX PRO) and GB10.

It is intentionally narrow. This is not a generic CUDA kernel collection or a
full model-serving stack, and it does not target any other architecture,
including SM100. It is a focused set of high-performance kernels plus the
runtime glue needed to launch them cleanly from `sglang`/`vllm`.

## Install

```bash
pip install b12x
```

You need Python 3.10+, `torch >= 2.12`, and an SM120/SM121 GPU. The CuTe DSL
compiler and its CUDA 13 libraries come in as wheel dependencies
(`nvidia-cutlass-dsl == 4.6.0`), so there is no build step — kernels are
JIT-compiled on first use and cached.

## What's in here

**GEMM** (`b12x/gemm/`) — a dense block-scaled GEMM (`DenseGemmKernel`,
exposed as `b12x::dense_gemm_launch`) covering NVFP4 and MXFP8 operands with
BF16/FP16/FP32 outputs, plus fused linear layers on top of it: MXFP8
(`b12x::mxfp8_linear_fused`), 128x128 block-FP8
(`b12x::block_fp8_linear_mxfp8_fused`), and the grouped WO-projection paths
used by MLA attention output.

**Attention** (`b12x/attention/`) — contiguous (fixed-shape and packed-varlen)
and paged attention forward kernels, with BF16/FP16 and FP8 E4M3 KV caches,
GQA, sliding window, and attention sinks. Sparse MLA decode/prefill lives in
`mla/`, and the NSA/MSA logits indexer plus its top-k and scheduling kernels
in `indexer/`. Compressed MLA and GLM MLA/NSA are distinct contracts and kept
separate on purpose. `paged/graph_replay.py` has the metadata staging kernels
that make decode replayable under CUDA graphs.

**MoE** (`b12x/moe/`) — fused FP4 TP MoE in three flavors: a direct
micro-kernel decode path, a unified dynamic path (persistent grid, dynamic M
tiles, `nvfp4`/`w4a8_mx`/`w4a8_nvfp4` weights), and W4A16 (BF16 activations
with inline FP4 weight dequant — no activation-scale math). SiLU, ReLU2, and
SwiGLU-OAI activations throughout.

**Everything else** — BF16→NVFP4 TMA quantization (`b12x/quantization/`), mHC
residual/projection kernels (`b12x/integration/residual*.py`), and an
IPC-backed PCIe one-shot allreduce (`b12x/distributed/`). The
`b12x/integration/` layer is the boundary serving stacks talk to: it owns
planning, scratch layout, and policy, so integrations only supply metadata and
capacity limits.

## Using it

Kernels are registered as torch custom ops under the `b12x::` namespace, so
after `import b12x` they are callable as `torch.ops.b12x.*` and compose with
`torch.compile` and CUDA graphs. Higher-level Python entry points (kernel
classes, planners) live next to each kernel.

Compilation happens lazily per shape/config and is cached. For serving, warm
up the shapes you need, then freeze:

```python
import b12x

# ... run warmup traffic covering every shape you will serve ...
b12x.freeze_kernel_resolution("serving")
```

After the freeze, any request that would trigger a new kernel compile raises
`KernelResolutionFrozenError` instead of stalling a live request (or worse,
compiling inside CUDA graph capture).

Set `B12X_PRINT_COMPILE_PROGRESS=1` to log each compiler invocation with its
cache-key parameters and duration — useful for figuring out what warmup
actually covered. `B12X_TIMING=1` enables per-kernel timing logs.

## Where to look next

- `tests/` is the executable spec — every kernel has API and numerical
  reference tests showing exact tensor layouts and call sequences.
- `benchmarks/` has tuned invocations per kernel family (and `probe_*` scripts
  from tile-sweep experiments).
- `docs/` has design notes: the MoE execution model, the eager-plan-bind
  architecture, and an SM120 MLA postmortem.

Failing that, ask your friendly neighborhood AI agent — it does fine here.
