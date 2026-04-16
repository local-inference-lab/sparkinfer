"""Graph-safe smoke test for b12x BF16 MoE on Nemotron-MTP-shaped routing.

Verifies that `b12x_moe_bf16` with an externally-supplied `output=` buffer can
be captured in a CUDA graph and replayed with consistent outputs.  Targets the
exact shape profile used by the Nemotron 3 Super MTP predictor.
"""

from __future__ import annotations

import torch

from b12x.integration.tp_moe_bf16 import (
    allocate_tp_moe_bf16_workspace_pool,
    b12x_moe_bf16,
)


E = 512
K = 1024
N = 2688
TOP_K = 22
ACTIVATION = "relu2"
# Anchor this smoke test to the actual Nemotron relu2 served regime.
BATCH_SIZES = (1, 2, 4, 8)


def _relu2_w1_rows(n: int) -> int:
    return n


def _build_topk(m: int, device: torch.device, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(m, E, device=device, dtype=torch.float32, generator=generator)
    top = torch.topk(logits, TOP_K, dim=-1)
    weights = torch.softmax(top.values, dim=-1).to(torch.float32)
    ids = top.indices.to(torch.int32)
    return weights, ids


def _reference(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(a, dtype=torch.float32)
    a_f32 = a.to(torch.float32)
    w1_f32 = w1.to(torch.float32)
    w2_f32 = w2.to(torch.float32)
    for token in range(a.shape[0]):
        for slot in range(TOP_K):
            expert = int(topk_ids[token, slot].item())
            weight = float(topk_weights[token, slot].item())
            x = a_f32[token]
            h = x @ w1_f32[expert].T
            h = (h.clamp_min(0.0)) ** 2
            y = h @ w2_f32[expert].T
            out[token] += weight * y
    return out.to(torch.bfloat16)


def _check_case(m: int, device: torch.device, pool, generator: torch.Generator) -> None:
    a = torch.randn(m, K, device=device, dtype=torch.bfloat16, generator=generator) * 0.1
    w1 = (
        torch.randn(E, _relu2_w1_rows(N), K, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.02
    )
    w2 = (
        torch.randn(E, K, N, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.02
    )
    topk_weights, topk_ids = _build_topk(m, device, generator)

    output_buf = torch.empty_like(a)

    # Warmup (jit compile, workspace realize) — outside capture.
    out_eager = b12x_moe_bf16(
        a=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=ACTIVATION,
        workspace=pool,
        output=output_buf,
    )
    eager_snapshot = out_eager.clone()
    torch.cuda.synchronize()

    # Capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            b12x_moe_bf16(
                a=a,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=ACTIVATION,
                workspace=pool,
                output=output_buf,
            )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        b12x_moe_bf16(
            a=a,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=ACTIVATION,
            workspace=pool,
            output=output_buf,
        )

    # Replay: refresh input/topk data in-place (simulating new decode step)
    # but keep shapes + ptrs identical so the graph can reuse them.
    a.normal_(generator=generator).mul_(0.1)
    new_weights, new_ids = _build_topk(m, device, generator)
    topk_weights.copy_(new_weights)
    topk_ids.copy_(new_ids)

    graph.replay()
    torch.cuda.synchronize()

    # Run the same inputs eagerly with a fresh pool so stale route metadata in
    # the capture workspace cannot hide replay bugs.
    eager_pool = allocate_tp_moe_bf16_workspace_pool()
    eager_post = b12x_moe_bf16(
        a=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=ACTIVATION,
        workspace=eager_pool,
        output=torch.empty_like(a),
    )

    max_abs = (output_buf.float() - eager_post.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        output_buf.float().flatten(), eager_post.float().flatten(), dim=0
    ).item()
    print(
        f"  m={m:>4}  graph_max_abs_vs_eager={max_abs:.5g}  cos={cos:.6f}  "
        f"warmup_max={eager_snapshot.abs().max().item():.4g}"
    )
    assert cos > 0.999, f"cos too low: {cos}"
    # max_abs check is loose — MoE path is slightly non-deterministic across
    # scatter orderings, but graph replay should match eager to within fp16-ish bounds.
    assert max_abs < 0.1, f"max_abs too large: {max_abs}"


def main() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    pool = allocate_tp_moe_bf16_workspace_pool()

    print(f"E={E}  K={K}  N={N}  topk={TOP_K}  act={ACTIVATION}")
    for m in BATCH_SIZES:
        _check_case(m, device, pool, generator)
    print("ok")


if __name__ == "__main__":
    main()
