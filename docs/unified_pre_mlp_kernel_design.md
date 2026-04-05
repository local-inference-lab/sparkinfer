# Unified Pre-MLP Kernel Design

Scope:
- Qwen3.5
- compact/static MoE path only
- shared expert integrated into the same routed contract as sparse experts
- `fc1_tile_amax = False` in v1

This design is based on the now-validated static full-block path:
- `b12x.integration.tp_moe._b12x_gemma_moe_block_fp4_static(...)`
- `tests/test_gemma_moe_block_paths.py`
- `benchmarks/benchmark_gemma_moe_block_paths.py`

That path proves the contract we want is correct:
- sparse experts can be combined with the shared expert as a virtual extra expert
- the shared gate can be represented as one more routing weight per token
- the existing prequantized static/micro consumer kernels can consume the full output directly

## 1. Goal

Replace this sequence:

1. PCIe oneshot allreduce
2. residual add
3. Gemma RMSNorm
4. sparse router GEMM + top-k
5. shared-expert gate GEMV
6. route/pack into the compact prequantized FC1 contract
7. prequantized MoE consumer

with:

1. one unified producer kernel
2. one existing expert consumer kernel

The producer kernel should avoid materializing a generic BF16 post-norm tensor for the sparse path.

## 2. Current Validated Contract

For Qwen3.5 in `/data/models/Qwen3.5-397B-A17B-NVFP4`:
- hidden size: `4096`
- sparse experts: `512`
- sparse top-k: `10`
- shared experts: `1`
- shared expert width matches sparse expert width: `1024`
- shared expert weights are already NVFP4
- shared expert gate is BF16 and replicated

That means the shared expert can be appended as expert `512`, and the routed contract becomes:
- total experts: `513`
- routed rows per token: `11`

The consumer already accepts this if we fill:
- `row_counts`
- `active_expert_count`
- `weight_expert_ids`
- `global_to_local_expert`
- `token_map`
- `token_weights`
- `packed_input`
- `packed_input_scale`
- `fc1_tile_scale`
- `fc1_tile_alpha`

## 3. Final Architecture

### Kernel 1: Unified Pre-MLP Producer

Responsibilities:
- oneshot allreduce from caller-supplied IPC peer pointers
- residual add
- Gemma RMSNorm
- sparse gate GEMM
- sparse top-k selection + renormalization
- shared gate GEMV + sigmoid
- route assignment into expert-major compact rows
- FC1 input quantization into routed NVFP4
- write the full compact prequantized workspace contract

### Kernel 2: Existing Expert Consumer

Reuse the current static/micro prequantized consumer path:
- FC1 MMA
- SiLU
- FC2 quantization / MMA
- scatter to output

No separate shared-expert path remains outside `b12x` for this mode.

## 4. Producer Inputs

Required logical inputs:
- local hidden-state shard for this TP rank
- residual tensor
- Gemma RMSNorm weight
- sparse router weight / bias
- shared gate weight / bias
- combined expert-bank weight metadata:
  - FC1 input scales
  - FC1 alphas
  - packed sparse + shared expert weights
  - packed blockscale tensors
- IPC peer input pointers
- IPC signal pointers
- rank / world size

Caller-owned inputs should remain generic:
- no dependency on `sglang` communicator objects
- only pointer-based IPC descriptors and rank metadata

## 5. Producer Outputs

The producer writes the exact compact prequantized contract consumed today:

- `active_expert_count`
- `weight_expert_ids`
- `global_to_local_expert`
- `row_counts`
- `token_map`
- `token_weights`
- `packed_input`
- `packed_input_scale`
- `fc1_tile_scale`
- `fc1_tile_alpha`

And optionally:
- `residual_out`

It does **not** write a reusable BF16 post-norm tensor for the sparse path.

## 6. Shared Expert Treatment

The shared expert is represented as a virtual extra expert:
- sparse experts: `0..511`
- shared expert: `512`

Per token:
- sparse top-k contributes `10` routed rows
- shared gate contributes `1` routed row

The shared gate output is the routing weight for expert `512`.

This keeps the consumer unchanged:
- it only sees expert-major rows plus routing weights
- it does not need to know which expert is “shared”

## 7. Launch Geometry

The most natural producer geometry is:
- one CTA per token row for v1
- one warp for the oneshot/allreduce control plane
- remaining warps for local GEMV/GEMM work and packing

Why:
- decode / compact shapes are small
- one-CTA-per-token aligns well with:
  - norm over hidden dim
  - sparse router logits for one token
  - shared gate for one token
  - writing that token’s `11` routed rows

This is not necessarily the final highest-throughput shape, but it is the cleanest first implementation.

## 8. Phase Breakdown Inside The Producer

### Phase A: IPC Allreduce

Use the existing oneshot signaling pattern from `b12x.distributed.pcie_oneshot`:
- publish local block completion
- wait on peer counters
- read peer input pointers directly

Numerical contract must match the validated BF16 path:
- reduce in input dtype
- form `residual_out` from the rounded reduced value
- compute RMS statistic from that rounded `residual_out`

### Phase B: Residual + Gemma RMSNorm

For each token:
- add reduced hidden row + residual row
- compute row RMS
- apply `1 + gamma`

The normalized row should stay in registers/shared memory for subsequent phases.

### Phase C: Sparse Router

Compute sparse router logits:
- `512 x 4096` GEMV for one token row

Then:
- top-k select
- renormalize sparse weights with softmax over the selected logits

We do not need to materialize full router logits in global memory for the fused path.

### Phase D: Shared Gate

Compute the shared gate scalar:
- `1 x 4096` GEMV
- apply sigmoid

This yields one more routing weight for the virtual shared expert.

### Phase E: Route Assignment

The CTA now knows all `11` routed pairs for its token:
- `10` sparse expert ids + weights
- `1` shared expert id + weight

It appends rows into the compact expert-major workspace:
- atomically reserve one row per `(token, expert)` pair
- write `token_map`
- write `token_weights`
- update `row_counts`
- update `active_expert_count` / `weight_expert_ids` / `global_to_local_expert`

The producer should lift the route/pack mechanics directly from Phase 1 of `b12x/moe/fused/static.py`.

### Phase F: FC1 Input Quantization

For v1:
- use the existing expert-global input-scale contract
- `fc1_tile_amax = False`

So after the row is assigned, quantize the normalized BF16 row directly into:
- `packed_input`
- `packed_input_scale`
- `fc1_tile_scale`
- `fc1_tile_alpha`

This again should be lifted from the existing static route/pack frontend.

## 9. Data Movement Strategy

The key optimization goal is:
- never write a generic BF16 sparse-input tensor
- never reread that tensor just to route and quantize it

Instead:
- read peer/local BF16 hidden once
- normalize once
- route once
- write final routed FP4 contract once

The only unavoidable large global writes between kernels should be:
- `residual_out` if needed
- the routed compact FP4 contract

## 10. Synchronization Strategy

Inside the producer:
- IPC signal synchronization for oneshot allreduce
- CTA-wide syncs between:
  - reduce/norm
  - router/shared-gate
  - route/pack writes

No inter-CTA reduction is needed in v1 because:
- `fc1_tile_amax` is off
- we only need per-row quantization metadata

This is one reason to keep `fc1_tile_amax` out of the first producer kernel.

## 11. Weight Model

Prepare a combined expert bank once:
- sparse expert bank of size `512`
- append shared expert bank as expert `512`

This combined bank should include:
- `a1_gscale`
- `w1_fp4`
- `w1_blockscale`
- `w1_alphas`
- `a2_gscale`
- `w2_fp4`
- `w2_blockscale`
- `w2_alphas`

The current validated helper path in `tp_moe.py` already proves this composition works.

## 12. Internal API Shape

Recommended internal-only wrapper:

```python
b12x_qwen35_moe_block_fp4_static(
    hidden_states,
    residual,
    *,
    ipc,
    norm_weight,
    norm_eps,
    sparse_gate_weight,
    sparse_gate_bias,
    shared_gate_weight,
    shared_gate_bias,
    combined_experts,
    workspace,
    output=None,
    residual_out=None,
)
```

Where `ipc` is a generic pointer-based descriptor:
- `rank`
- `world_size`
- `signal_ptrs`
- `peer_input_ptrs`

This keeps `sglang` in control of its own process lifecycle and IPC setup.

## 13. Validation Plan

### Contract Tests
- producer workspace vs current host/device route-pack reference
- combined routing parity vs validated Python composition

### Functional Tests
- producer + existing consumer vs current validated `_b12x_gemma_moe_block_fp4_static(...)`
- cosine threshold stays strict: `> 0.999`

### Distributed Smoke
- 2-rank compare against:
  - NCCL allreduce
  - Gemma RMSNorm
  - sparse route selection
  - shared gate
  - BF16 `b12x_moe_fp4(...)` on the combined expert bank

### Benchmark
- producer only
- consumer only
- full path
- same batch-size matrix as `benchmark_gemma_moe_block_paths.py`

## 14. Deferred Work

Not part of producer v1:
- dynamic path
- `fc1_tile_amax = True`
- generic BF16 side output for non-fused consumers
- fused shared-gate + sparse-gate kernel sharing
- multi-CTA producer ownership per token

Those can be layered once the static producer is correct and faster than the current reference-style route-pack step.
