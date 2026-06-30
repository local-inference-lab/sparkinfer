
`b12x` is an SM120/SM121 CuTe DSL kernel library for (primarily) NVFP4 LLM inference.

It is intentionally narrow. This is not a generic CUDA kernel collection or a
full model-serving stack. It does not intend to target any other GPU architectures,
including SM100. It is a focused package for a small number of high-performance
kernels plus the runtime glue needed to launch them cleanly from `sglang`/`vllm`.

## Kernel inventory

Scope: package kernels shipped under `b12x/`. Benchmark and probe kernels under
`benchmarks/` are not listed as supported package surface.

### GEMM and projection

| Surface | Kernels / ops | Files | Variants |
| --- | --- | --- | --- |
| Dense block-scaled GEMM | `DenseGemmKernel`<br>`b12x::dense_gemm_launch`<br>`b12x::dense_gemm_launch_functional`<br>2-way split-K BF16 reducer | `b12x/gemm/dense.py` | NVFP4 (`float4_e2m1fn`), MXFP8 (`float8_e4m3fn`); BF16/FP16/FP32 outputs; E8M0/block scales; TMA or cp.async; A/B swap; expected-M tile regimes; out-buffer or functional output |
| MXFP8 linear | `b12x::mxfp8_linear_fused` | `b12x/gemm/mxfp8_linear.py` | ModelOpt-style MXFP8 weights; activation quantization plus dense GEMM |
| Block-FP8 linear | `b12x::block_fp8_linear_mxfp8_fused`<br>`b12x::quantize_block_fp8_linear_input_mxfp8_alloc` | `b12x/gemm/block_fp8_linear.py` | 128x128 block-FP8 weights; Triton `TK -> TK` activation quantizer; dense GEMM backend |
| WO projection | grouped WO-A projection<br>grouped WO-B projection<br>`b12x::wo_projection_inv_rope_mxfp8_fused` | `b12x/gemm/wo_projection.py` | MXFP8 dense-GEMM projections; grouped `[T,G,D] -> [T,D,G]` quantizer; inverse-RoPE attention-output quantizer; group-major `[T,R,G] -> [T,K]` quantizer |

### Attention

| Surface | Kernels / ops | Files | Variants |
| --- | --- | --- | --- |
| Contiguous attention | `ContiguousAttentionForwardKernel`<br>`b12x_attention_forward`<br>`b12x_varlen_attention_forward` | `b12x/attention/contiguous/` | Fixed-shape and packed-varlen; BF16/FP16; causal or non-causal; local/window attention; attention-sink bias; GQA packing; head dims `<=64`, `<=128`, `256` |
| Paged attention | decode/verify `PagedForwardKernel`<br>extend `PagedForwardKernel`<br>`PagedFp8DecodeRawForwardKernel`<br>`PagedBf16ExtendRawForwardKernel`<br>`PagedFp8ExtendRawForwardKernel`<br>`PagedFp8RawPlaneDumpKernel`<br>`PagedPersistentMergeKernel` | `b12x/attention/paged/` | BF16/FP16 and FP8 E4M3 KV caches; page sizes 64 and 128; split-KV or direct output; native FP8 QK/PV option; sliding window; attention-sink bias; GQA; MSA block-sparse and union-tile prefill |
| Paged graph replay helpers | decode metadata stage/patch/update kernels<br>MSA metadata update kernels<br>chunk and union metadata builders | `b12x/attention/paged/graph_replay.py` | CUDA-graph replay for single-request, single-qtile, regular decode, and MSA decode metadata |
| Sparse MLA | `UnifiedDecodeKernel`<br>`UnifiedPrefillMGKernel`<br>`SparseMLASplitDecodeMergeKernel`<br>`SparseMLASplitDecodeSinkMergeKernel`<br>`sparse_mla_decode_forward`<br>`sparse_mla_extend_forward` | `b12x/attention/mla/` | Single-cache decode; dual-cache extra-section decode; uniform section length; per-token length; single-cache MG prefill; dual-cache MG prefill |
| Sparse MLA public ops | `b12x::sparse_mla_sm120_decode_grid`<br>`b12x::sparse_mla_sm120_prefill_mg`<br>`b12x::sparse_mla_sm120_prefill_mg_dual`<br>`b12x::sparse_mla_sm120_split_decode_merge` | `b12x/attention/mla/` | SM120 sparse MLA decode, extend, MG prefill, and split-decode merge entry points |
| Legacy and compressed MLA | sparse MLA one-pass<br>sparse split-decode<br>compressed split-decode<br>sparse split-merge<br>sink-merge | `b12x/attention/mla/legacy/`<br>`b12x/attention/mla/compressed_api.py` | Compatibility paths; `b12x::compressed_mla_split_decode_forward`; `b12x::sparse_mla_split_decode_merge`; compressed MLA stays separate from GLM MLA/NSA contracts |
| NSA/MSA logits indexer | `SparseNSAPagedLogitsKernel`<br>`SparseNSAPagedSupertileLogitsKernel`<br>`SparseNSAScheduledSingleRowLogitsKernel`<br>`SparseNSAScheduledMultiRowLogitsKernel`<br>`SparseNSAContiguousLogitsKernel`<br>`SparseNSAContiguousLogitsPrefillKernel`<br>`SparseNSAContiguousLogitsPrefill512Kernel` | `b12x/attention/indexer/kernel.py`<br>`b12x/attention/indexer/contiguous_kernel.py` | Paged logits; paged supertile logits; scheduled long single-row and multi-row decode; contiguous logits; BK=256 prefill; experimental BK=512 prefill |
| NSA/MSA top-k and scheduling | `SparseNSAFusedIndexerKernel`<br>`SparseNSAPersistentTopK2048Kernel`<br>`SparseNSATiledTopkKernel`<br>paged supertile gather helper<br>paged-MQA schedule builder | `b12x/attention/indexer/fused_indexer.py`<br>`b12x/attention/indexer/persistent_topk.py`<br>`b12x/attention/indexer/tiled_topk.py`<br>`b12x/attention/indexer/paged.py`<br>`b12x/attention/indexer/schedule_metadata.py` | Fused score plus top-k; paged and contiguous-MLA layouts; persistent top-k 2048; tiled, row, and supertile top-k; paged-MQA metadata |

### MoE

| Surface | Kernels / ops | Files | Variants |
| --- | --- | --- | --- |
| Direct-micro FP4 TP MoE | `MoEMicroKernelBackend`<br>`MoEMicroKernelSilu`<br>`MoEMicroKernelRelu2`<br>`MoEMicroKernelSwiGLUOAI`<br>`b12x::tp_moe_compact_micro_launch` | `b12x/moe/fused/micro.py`<br>`b12x/moe/fused/silu.py`<br>`b12x/moe/fused/relu2.py`<br>`b12x/integration/tp_moe.py` | SiLU, ReLU2, SwiGLU-OAI; direct decode; shared input or expert scales; E4M3 K/16 or E8M0 K/32 scale formats; `w13` or `w31` weight layouts |
| Unified dynamic FP4 TP MoE | `MoEDynamicKernelBackend`<br>`MoEDynamicKernelSilu`<br>`MoEDynamicKernelRelu2`<br>`MoEDynamicKernelSwiGLUOAI`<br>`b12x::tp_moe_dynamic_launch` | `b12x/moe/fused/dynamic.py`<br>`b12x/moe/fused/silu.py`<br>`b12x/moe/fused/relu2.py`<br>`b12x/moe/fused/w4a8/weights.py`<br>`b12x/integration/tp_moe.py` | Compile-time materialized-queue, persistent-grid, or ready-queue work source; dynamic M tiles 16/32/64/128 by N128; `nvfp4`, `w4a8_mx`, `w4a8_nvfp4`; token-major W4A8 input and materialized FC2; N256/K128 prepared W4A8 weights; deterministic-output top-k sum; SiLU, ReLU2, SwiGLU-OAI |
| W4A16 MoE | `_W4A16SmallMDirectKernel`<br>`W4A16GemmKernel`<br>`W4A16FusedMoeKernel`<br>`W4A16ActivationKernel`<br>`W4A16TopKSumKernel`<br>`b12x::w4a16_small_m_direct_launch`<br>`b12x::w4a16_fused_moe_launch`<br>`b12x::w4a16_topk_sum_launch` | `b12x/moe/fused/w4a16/kernel.py` | BF16 activations with inline FP4/NVFP4 weight dequantization; packed or ModelOpt weight layouts; E4M3 K/16 or E8M0 K/32 scales; W13/W31 order; direct top-k routes or route-pack; small-M direct decode; persistent packed GEMM; fused FC1+activation+FC2; TC-decode fused-sum epilogue |
| Route and layout helpers | W4A16 route-pack Triton kernels<br>TP-MoE repack/conversion kernels<br>router top-k kernel | `b12x/moe/fused/w4a16/route_pack.py`<br>`b12x/integration/tp_moe.py`<br>`b12x/integration/triton_route.py` | W4A16 packed weights; W4A8 row-panel weights; E8M0 scale-grid/SFB layouts; optional router-weight renormalization |

### Residual, quantization, and distributed support

| Surface | Kernels / ops | Files | Variants |
| --- | --- | --- | --- |
| mHC residual/projection | `MHCPostPrePartialKernel`<br>`MHCPostPrePrefillPartialKernel`<br>`MHCPostPrePrefillBlockMPartialKernel`<br>`MHCPostPrePrefillGramKernel`<br>`MHCPrefillBf16ProjectTmaKernel`<br>`MHCPrefillTf32ProjectTmaKernel`<br>`MHCPrefillBf16ProjectKernel`<br>`MHCFinalizeGramKernel` | `b12x/integration/residual_kernels.py`<br>`b12x/integration/residual.py` | Post, pre, and post-pre partial reductions; prefill full-hidden, block-M, and Gram paths; BF16 TMA projection; BF16 non-TMA projection; TF32 projection; finalize-Gram; planned `mhc_pre`, `mhc_post_pre`, and `mhc_post` wrappers |
| BF16-to-FP4 TMA quantization | BF16-to-packed-NVFP4 CuTe TMA kernel<br>`compile_bf16_to_fp4_tma` | `b12x/quantization/bf16_to_fp4_tma.py` | BF16 input tiles to packed NVFP4 plus scale tiles |
| PCIe one-shot allreduce | `pcie_allreduce_kernel`<br>`PCIeOneshotAllReduce` | `b12x/distributed/pcie_oneshot.cu` | IPC-backed PCIe allreduce for FP32/FP16/BF16 |

```bash
pip install b12x
```

Ask your friendly neighborhood AI agent for further information on how to use this library.
