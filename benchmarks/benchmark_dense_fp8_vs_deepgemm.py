"""Head-to-head: b12x dense MXFP8 GEMM vs DeepGEMM SM120 fp8_gemm_nt (1d1d).

Graph-replay timed, correctness-gated (each vs its own fp32 oracle). Both run
their native FP8 block-scaled path (b12x: per-32 ue8m0; DeepGEMM: per_token/
per_block gran_k=128 ue8m0). We time only the GEMM (operands pre-quantized
outside the captured region).

Run: CUDA_VISIBLE_DEVICES=0 trafficcontrol --timeout 600 -- \
       ~/projects/vllm-other/.venv/bin/python \
       benchmarks/benchmark_dense_fp8_vs_deepgemm.py
"""

import statistics
import sys

sys.path.insert(0, "/home/luke/projects/vllm-other/vllm/third_party")
sys.path.insert(0, "/home/luke/projects/b12x/benchmarks")

import torch
import deep_gemm

print("deep_gemm from:", deep_gemm.__file__)

from benchmark_dense_gemm import (
    bench_events,
    capture_graph_replay,
    make_l2_flush_fn,
    make_mxfp8_operand,
)
from b12x.gemm.dense import dense_gemm


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)).item()


def med_us(times_ms):
    return statistics.median(times_ms) * 1000.0


def deepgemm_setup(M, N, K):
    A = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 4)
    B = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 4)
    a = deep_gemm.per_token_cast_to_fp8(A, True)   # (fp8 [M,K], sf), gran_k=128
    b = deep_gemm.per_block_cast_to_fp8(B, True)    # (fp8 [N,K], sf), 128x128 block
    d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    oracle = A.float() @ B.float().T

    def launch():
        deep_gemm.fp8_gemm_nt(a, b, d)

    return launch, d, oracle


def b12x_setup(M, N, K):
    a_q, a_s, a_mma, a_src = make_mxfp8_operand(M, K)
    b_q, b_s, b_mma, b_src = make_mxfp8_operand(N, K)
    out = torch.empty((M, N, 1), device="cuda", dtype=torch.bfloat16)

    def launch():
        dense_gemm(
            (a_q.view(M, K, 1), a_mma),
            (b_q.view(N, K, 1), b_mma),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32, out=out,
            expected_m=M,  # vllm scaled_mm passes expected_m=tokens
        )

    oracle = a_src.float() @ b_src.float().T
    return launch, out, oracle


def run(M, N, K, l2, warmup, iters):
    print(f"\n(M={M}, N={N}, K={K})")
    row = {"M": M}
    for name, setup in (("deepgemm", deepgemm_setup), ("b12x", b12x_setup)):
        try:
            launch, out, oracle = setup(M, N, K)
            replay = capture_graph_replay(launch)
            replay()
            torch.cuda.synchronize()
            c = cos(out.reshape(M, N), oracle)
            t = bench_events(replay, warmup=warmup, iters=iters, l2_flush=l2)
            us = med_us(t)
            row[name] = us
            flag = "" if c > 0.99 else "  !! LOW COS"
            print(f"  {name:9s} {us:8.1f} us   cos={c:.5f}{flag}")
        except Exception as exc:
            row[name] = None
            print(f"  {name:9s} FAILED: {exc}")
    if row.get("b12x") and row.get("deepgemm"):
        r = row["b12x"] / row["deepgemm"]
        print(f"  -> b12x/deepgemm = {r:.2f}x  ({'b12x slower' if r > 1 else 'b12x faster'})")
    return row


def main():
    torch.manual_seed(0)
    l2 = make_l2_flush_fn(enabled=True)
    warmup, iters = 10, 50
    # DeepSeek-V4-Flash q/k/v projections at TP=2 (hidden=4096, heads=64->32 local,
    # q_lora_rank=1024, head_dim=512). N=out_features, K=in_features.
    configs = [
        ("qkv_a_down", 1536, 4096),    # fused wqa_wkv: hidden(4096) -> q_lora(1024)+kv_latent(512)
        ("q_b_up", 16384, 1024),       # wq_b: q_lora(1024) -> n_local_heads(32)*head_dim(512)
        ("wo_a (per-group; L=4)", 1024, 4096),  # group_width(8*512) -> o_lora_rank(1024)
        ("wo_b", 4096, 4096),          # n_local_groups*o_lora(4*1024) -> hidden(4096)
    ]
    Ms = [2, 8, 32, 128, 512, 2048, 4096]  # decode (2-8) -> prefill

    by_cfg = {}
    for label, N, K in configs:
        print(f"\n######## {label}: N={N} K={K} ########")
        ratios = []
        for M in Ms:
            r = run(M, N, K, l2, warmup, iters)
            if r.get("b12x") and r.get("deepgemm"):
                ratios.append((M, r["b12x"], r["deepgemm"], r["b12x"] / r["deepgemm"]))
        by_cfg[(label, N, K)] = ratios

    print("\n==== summary: DeepSeek-V4-Flash q/k/v projections, TP=2 (b12x/dg; >1 = b12x slower) ====")
    all_r = []
    for (label, N, K), ratios in by_cfg.items():
        print(f"\n  {label}  N={N} K={K}")
        for M, b, d, r in ratios:
            all_r.append(r)
            print(f"    M={M:5d}: b12x={b:8.1f}us  dg={d:8.1f}us  {r:.2f}x  ({'b12x slower' if r > 1.02 else 'b12x faster' if r < 0.98 else 'tie'})")
        if ratios:
            print(f"    -> {label} geomean = {statistics.geometric_mean([x[3] for x in ratios]):.2f}x")
    if all_r:
        print(f"\n  OVERALL geomean ratio = {statistics.geometric_mean(all_r):.2f}x")


if __name__ == "__main__":
    main()
