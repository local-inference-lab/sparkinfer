# Adding ReLU2 Activation Support to b12x MoE Kernels

## Motivation

Nemotron-3-Super-120B uses non-gated ReLU2 (`relu(x)^2`) in its MoE experts.
b12x currently hardcodes SiLU/SwiGLU activation, forcing Nemotron to fall back
to FlashInfer CUTLASS (~85 tok/s on SM120). Adding ReLU2 would give Nemotron
the same +18-26% speedup Qwen3.5 gets, plus an inherent FC1 speedup (1 GEMM
instead of 2).

## SiLU (SwiGLU) vs ReLU2 (non-gated) data flow

```
SiLU/SwiGLU (gated):
  w1 shape: [E, 2*I, K//2]    (gate + up stacked)
  FC1 gate GEMM: gate_out = A @ gate_weights^T
  FC1 up GEMM:   up_out   = A @ up_weights^T
  Activation:    result = SiLU(gate_out) * up_out
  FC2 GEMM:      output = result @ down_weights^T

ReLU2 (non-gated):
  w1 shape: [E, I, K//2]      (single FC1 weight)
  FC1 GEMM:     fc1_out = A @ w1^T
  Activation:   result = relu(fc1_out)^2
  FC2 GEMM:     output = result @ down_weights^T
```

The non-gated path has half the FC1 compute (1 GEMM instead of 2) and does not
use a gate/up split.

## CuTe DSL constraint: no conditional control flow in kernels

**Critical finding from implementation attempt (2026-04-13):**

CuTe DSL's `@dsl_user_op` kernel methods and `__call__` methods are traced by
the MLIR code generator. The tracer treats ALL `if` statements as dynamic
control flow, even when the condition is a plain Python `bool` attribute like
`self.is_gated`.

### What fails

```python
# FAILS: NameError — variable not visible outside if block
if self.is_gated:
    gate_tile_cnt = intermediate_tile_cnt // Int32(2)
else:
    gate_tile_cnt = Int32(0)
# gate_tile_cnt is not defined here

# FAILS: ICE IR Verification — ternary traces both arms
gate_tile_cnt = intermediate_tile_cnt // Int32(2) if self.is_gated else Int32(0)
# error: operand #19 does not dominate this use

# FAILS: wrapping existing code blocks
if self.is_gated:
    up_acc.fill(0.0)        # up GEMM setup
    ...                      # up GEMM consumer loop
# ICE during CUDA graph capture for some batch sizes
```

### What works (but is insufficient for relu2)

```python
# OK: self.fast_math gates between two quantization paths
# that both write to the same output variable
if self.fast_math:
    packed64, scale_byte = quantize_block_fp4_fast(...)
else:
    packed64, scale_byte = quantize_block_fp4(...)
# Works because both branches assign the same variables
```

The `fast_math` pattern works because both branches produce the same output
variables. The relu2 case is fundamentally different: it needs to **skip an
entire GEMM pass** and change the epilogue data flow.

### What works in host code (not DSL)

Python `if/else` on `self.is_gated` works fine in:
- `__init__()` constructors
- Functions that are NOT `@dsl_user_op` decorated
- The `_get_*_kernel()` functions in `tp_moe.py` (host-side Python)

But NOT in `__call__()` (which IS a DSL-traced method in the dynamic kernel).

## Required approach: separate kernel classes

The only viable approach is to create separate kernel classes for each
activation:

```
b12x/moe/fused/
  static.py          — MoEStaticKernel        (SiLU, existing)
  static_relu2.py    — MoEStaticKernelRelu2   (ReLU2, new)
  dynamic.py         — MoEDynamicKernel       (SiLU, existing)
  dynamic_relu2.py   — MoEDynamicKernelRelu2  (ReLU2, new)
  micro.py           — MoEMicroKernel         (SiLU, existing)
  micro_relu2.py     — MoEMicroKernelRelu2    (ReLU2, new)
```

Each relu2 variant is a copy of its SiLU counterpart (~1700 lines) with:

### Changes per kernel file (~150 lines removed/modified)

1. **Remove up pipeline**: `up_pipeline_array`, `sB_up`, `sSFB_up` from shared
   memory struct; `up_pipeline` creation; `up_prod_state`/`up_cons_state`

2. **Remove up GEMM consumer loop**: ~70 lines of MMA warp code

3. **Remove up DMA producer loop**: ~10 lines of DMA warp code

4. **Change epilogue**: from `SiLU(gate) * up` to `relu(gate)^2`
   ```python
   # SiLU (existing):
   g = alpha_value * gate_slice[elem_idx]
   u = alpha_value * up_slice[elem_idx]
   sigmoid_g = cute.arch.rcp_approx(Float32(1.0) + cute.math.exp(-g))
   tRS_rD_slice[elem_idx] = g * sigmoid_g * u

   # ReLU2 (new):
   g = alpha_value * gate_slice[elem_idx]
   relu_g = fmax_f32(g, Float32(0.0))
   tRS_rD_slice[elem_idx] = relu_g * relu_g
   ```

5. **Remove `pass_sync_barrier` between gate and up**: Keep barrier for
   gate→FC2 sync, remove the gate→up sync comment

6. **Change `gate_tile_cnt` computation**: Remove the `// 2` divisor
   (`gate_tile_cnt = intermediate_tile_cnt` instead of `intermediate_tile_cnt // 2`)

7. **Change DMA gate B-tile offset**: Use `intermediate_slice` directly
   (no `+ gate_tile_cnt` offset since there's no up/gate split)

### Changes to integration layer (tp_moe.py)

1. Add `activation: str = "silu"` parameter to `b12x_moe_fp4()` and launcher
   functions
2. Select kernel class based on activation
3. In `_get_weight_views()`: use `w1_fp4.shape[1]` instead of `2 * n`
4. In `__call__` of dynamic kernel: compute `gate_tile_cnt` without `/2` for
   non-gated (this is in host code, so `if/else` is fine)
5. TMA fake tensor shapes: use actual `w1_N` dimension

### Changes to vLLM integration (b12x_moe.py)

1. Add `MoEActivation.RELU2_NO_MUL` to `_supports_activation()`
2. Map `MoEActivation.RELU2_NO_MUL` → `"relu2"` and pass to `b12x_moe_fp4()`

## Existing safe code (already committed)

The following additions are safe and don't touch the SiLU kernels:

- `b12x/cute/fp4.py`: `relu2_16()`, `relu2_quantize_block_fp4()`,
  `relu2_quantize_grouped_nvfp4_torch()`
- `b12x/moe/fused/reference.py`: `activation` parameter in reference paths
- `b12x/quant/`: relu2 quantize exports

## Estimated effort

- Fork the 3 kernel files + modify: 2-3 days for someone familiar with the code
- Integration layer + vLLM wiring: 1 day
- Testing with Nemotron checkpoint: 1 day

## Testing plan

1. Unit test: reference paths (relu2) vs naive PyTorch implementation
2. Kernel test: relu2 kernel vs reference path (extend `test_tp_moe_reference.py`)
3. End-to-end: Nemotron-3-Super-120B-A12B-NVFP4 with b12x relu2 backend
4. Regression: Qwen3.5-122B with SiLU kernels (zero changes to SiLU path)
