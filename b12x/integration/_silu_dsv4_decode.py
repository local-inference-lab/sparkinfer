"""DSv4-Flash specialized SiLU MoE decode dispatch for SM_120.

EXACT shape specialization for DSv4-Flash decode:
- Hidden K = 4096
- Intermediate N = 2048
- Num experts E = 256 (TP=4 → 64 per rank)
- Top_k = 6 (avg 1.5 per rank)
- Activation = silu (gated)
- m ∈ {1..8} (decode bs=1 + EAGLE expansion)

Microbench data (single GPU, E=256):
  m=1: b12x_moe_fp4 = 0.246ms (efficiency 23.6% of bandwidth)
  m=4: b12x_moe_fp4 = 0.323ms (efficiency 18.0%)
  Theoretical floor: 0.058ms

Goal: replace dispatch overhead with an exact-shape direct decode path.
Target speedup: remove the remaining Python/backend-selection overhead around
the already-tuned b12x NVFP4 compact microkernel for m=1-8 SiLU DSv4 shapes.

CURRENT DESIGN:
1. Static exact-shape gate: no generic backend selection on DSv4 decode.
2. Force the compact static/micro topology for m<=8, top_k=6.
3. Reuse the existing NVFP4 SiLU microkernel and workspace pool.
4. Reuse preflattened routing when sparse routing already produced it.

Memory footprint per CTA at m=1, top_k=6:
- Input a: 1 * 4096 * 2B = 8KB BF16
- Per-expert w1+w3: 2 * 2048 * 4096 * 0.5B = 8MB FP4 (need streamed access)
- Per-expert w2: 4096 * 2048 * 0.5B = 4MB FP4
- TOO BIG for SMEM. Need TMA + cooperative loading

This module owns the exact DSv4 predicate and a narrow launcher shim. The
launcher delegates to a private tp_moe helper, not to the public b12x_moe_fp4
entrypoint, so the direct path cannot recurse through generic dispatch.
"""
from __future__ import annotations

import os

import torch


# DSv4-Flash exact shapes (must match these to take fast path)
DSV4_HIDDEN_K = 4096
DSV4_INTERMEDIATE_N = 2048
DSV4_NUM_EXPERTS = 256
DSV4_TOP_K = 6
DSV4_W1_FP4_SHAPE_GLOBAL = (DSV4_NUM_EXPERTS, 2 * DSV4_INTERMEDIATE_N, DSV4_HIDDEN_K // 2)  # (256, 4096, 2048) gated
DSV4_W2_FP4_SHAPE_GLOBAL = (DSV4_NUM_EXPERTS, DSV4_HIDDEN_K, DSV4_INTERMEDIATE_N // 2)      # (256, 4096, 1024)
DSV4_BS_SUPPORTED = (1, 2, 3, 4, 5, 6, 7, 8)

# TP=4 partition shapes (each rank has E/4 experts)
DSV4_W1_FP4_SHAPE_TP4 = (DSV4_NUM_EXPERTS // 4, 2 * DSV4_INTERMEDIATE_N, DSV4_HIDDEN_K // 2)
DSV4_W2_FP4_SHAPE_TP4 = (DSV4_NUM_EXPERTS // 4, DSV4_HIDDEN_K, DSV4_INTERMEDIATE_N // 2)


def is_exact_silu_dsv4_case(
    *, activation: str, a: torch.Tensor, w1_fp4: torch.Tensor, w2_fp4: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    a1_gscale: torch.Tensor, a2_gscale: torch.Tensor,
) -> bool:
    """Gate for DSv4-silu specialized path. Accepts both global and TP=4 partition shapes."""
    if os.environ.get("B12X_DSV4_SILU_DIRECT", "1") == "0":
        return False
    if activation != "silu":
        return False
    if a.dtype != torch.bfloat16:
        return False
    if a.dim() != 2 or a.shape[1] != DSV4_HIDDEN_K:
        return False
    if topk_ids.dim() != 2 or topk_ids.shape[1] != DSV4_TOP_K:
        return False
    if topk_weights.shape != topk_ids.shape:
        return False
    if a1_gscale.numel() != 1 or a2_gscale.numel() != 1:
        return False
    if a.shape[0] not in DSV4_BS_SUPPORTED:
        return False
    if topk_ids.shape[0] != a.shape[0]:
        return False
    # Accept either global or TP=4-sliced shapes
    if tuple(w1_fp4.shape) == DSV4_W1_FP4_SHAPE_GLOBAL and tuple(w2_fp4.shape) == DSV4_W2_FP4_SHAPE_GLOBAL:
        return True
    if tuple(w1_fp4.shape) == DSV4_W1_FP4_SHAPE_TP4 and tuple(w2_fp4.shape) == DSV4_W2_FP4_SHAPE_TP4:
        return True
    return False


def launch_exact_silu_dsv4(
    *, workspace, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas,
    a2_gscale, w2_fp4, w2_blockscale, w2_alphas,
    topk_weights, topk_ids, output,
    apply_router_weight_on_input, input_scales_are_reciprocal,
    fast_math: bool = True,
    flat_ids: torch.Tensor | None = None,
    flat_weights: torch.Tensor | None = None,
):
    """Launch the exact DSv4-SiLU decode path without public-entry recursion."""
    if apply_router_weight_on_input:
        raise NotImplementedError("apply_router_weight_on_input is not implemented in DSv4 SiLU direct path")
    from b12x.integration.tp_moe import _launch_exact_silu_dsv4_decode

    return _launch_exact_silu_dsv4_decode(
        a=a, a1_gscale=a1_gscale,
        w1_fp4=w1_fp4, w1_blockscale=w1_blockscale, w1_alphas=w1_alphas,
        a2_gscale=a2_gscale, w2_fp4=w2_fp4, w2_blockscale=w2_blockscale, w2_alphas=w2_alphas,
        topk_weights=topk_weights, topk_ids=topk_ids,
        workspace=workspace, output=output,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        input_scales_static=(a1_gscale.numel() == 1 and a2_gscale.numel() == 1),
        fast_math=fast_math,
        flat_ids=flat_ids,
        flat_weights=flat_weights,
    )
