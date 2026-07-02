"""Does b12x's weight packing drop a re-quantization step vs DeepGEMM?

DeepSeek-style FP8 checkpoints store (w_fp8, s_fp32) where s_fp32 = blockamax/448
is an ARBITRARY fp32 per-128x128-block scale (NOT a power of two). To run a
UE8M0 block-scaled MMA you must reconcile this with power-of-two scales.

  A) b12x today: keep checkpoint w_fp8, round s_fp32 to nearest power-of-two
     (pack_fp8_block_scaled_weight_mxfp8 -> _scale_to_e8m0_u8 round). The fp8
     values stay matched to s_fp32, but are now multiplied by a DIFFERENT
     (rounded) scale -> per-block scale error up to sqrt(2).

  B) DeepGEMM-parity: dequantize w_recovered = w_fp8 * s_fp32, then
     per_block_cast_to_fp8(w_recovered, use_ue8m0=True) -> FRESH (w'_fp8, s'_e8m0)
     with ceil; fp8 values re-optimized for the ue8m0 scale.

Ground truth = the original bf16 weight (and the checkpoint intent C = w_fp8*s_fp32).
If B is much closer than A, b12x is dropping the re-quantization step.

Run with the assigned GPU:
  python benchmarks/probe_weight_requant_parity.py

GPU serialization, when enabled, is managed outside this command.
"""

import torch


def ceil_to_ue8m0(x):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def per_block_cast_to_fp8(x, use_ue8m0=True, gran_k=128):
    m, n = x.shape
    x_view = x.view(m // gran_k, gran_k, n // gran_k, gran_k)
    amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    q = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    sf2 = sf.view(m // gran_k, n // gran_k)
    return q.view(m, n).contiguous(), sf2


def round_pow2(s):
    # b12x _scale_to_e8m0_u8: round(log2(s)) -> nearest power of two
    e = torch.round(torch.log2(s.clamp_min(1e-30))).clamp(-127, 127)
    return torch.exp2(e)


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)).item()


def relfro(a, b):
    return ((a - b).flatten().float().norm() / b.flatten().float().norm() + 1e-30).item()


def expand_block(sf_block, gran_k, shape):
    # sf_block [N/128, K/128] -> [N, K]
    n, k = shape
    return sf_block.repeat_interleave(gran_k, 0).repeat_interleave(gran_k, 1)[:n, :k]


def main():
    torch.manual_seed(0)
    dev = "cuda"
    N, K = 4096, 4096
    w_orig = (torch.randn(N, K, device=dev) * 0.2).to(torch.bfloat16).float()

    # --- simulate a DeepSeek-style checkpoint: arbitrary fp32 block scale, fixed fp8 values ---
    wv = w_orig.view(N // 128, 128, K // 128, 128)
    blk_amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    s_fp32 = blk_amax / 448.0                                   # arbitrary fp32, NOT pow2
    w_fp8 = (wv / s_fp32).to(torch.float8_e4m3fn).view(N, K)     # checkpoint fp8 values
    s_fp32 = s_fp32.view(N // 128, K // 128)
    C = w_fp8.float() * expand_block(s_fp32, 128, (N, K))        # checkpoint intent (best estimate of w_orig)

    # --- A) b12x today: keep w_fp8, round the scale to pow2 ---
    s_round = round_pow2(s_fp32)
    deq_A = w_fp8.float() * expand_block(s_round, 128, (N, K))

    # --- B) DeepGEMM-parity: requantize from recovered weight ---
    wB_fp8, sB = per_block_cast_to_fp8(C, use_ue8m0=True)
    deq_B = wB_fp8.float() * expand_block(sB.float(), 128, (N, K))

    print("Weight reconstruction (N=K=4096, DeepSeek-style arbitrary-fp32-scale checkpoint)")
    print(f"  scale-rounding ratio s_round/s_fp32: "
          f"mean={ (s_round/s_fp32).mean().item():.4f}  "
          f"min={(s_round/s_fp32).min().item():.4f}  max={(s_round/s_fp32).max().item():.4f}")
    print()
    print("  vs checkpoint intent C = w_fp8*s_fp32:")
    print(f"    A) b12x round-scale, stale values : cos={cos(deq_A, C):.6f}  rel_fro={relfro(deq_A, C):.5f}")
    print(f"    B) requantize (DeepGEMM-parity)   : cos={cos(deq_B, C):.6f}  rel_fro={relfro(deq_B, C):.5f}")
    print()
    print("  vs ORIGINAL bf16 weight w_orig:")
    print(f"    C) checkpoint intent              : cos={cos(C, w_orig):.6f}  rel_fro={relfro(C, w_orig):.5f}")
    print(f"    A) b12x round-scale, stale values : cos={cos(deq_A, w_orig):.6f}  rel_fro={relfro(deq_A, w_orig):.5f}")
    print(f"    B) requantize (DeepGEMM-parity)   : cos={cos(deq_B, w_orig):.6f}  rel_fro={relfro(deq_B, w_orig):.5f}")

    # --- end-to-end GEMM impact (A @ W^T), activation kept bf16-exact to isolate weight quant ---
    M = 256
    x = (torch.randn(M, K, device=dev) * 0.5).float()
    oracle = x @ w_orig.T
    print()
    print("  GEMM x@W^T cos vs fp32-oracle (weight quant only):")
    print(f"    A) b12x round-scale  : cos={cos(x @ deq_A.T, oracle):.6f}  rel_fro={relfro(x @ deq_A.T, oracle):.5f}")
    print(f"    B) requantize        : cos={cos(x @ deq_B.T, oracle):.6f}  rel_fro={relfro(x @ deq_B.T, oracle):.5f}")
    print(f"    C) checkpoint intent : cos={cos(x @ C.T, oracle):.6f}  rel_fro={relfro(x @ C.T, oracle):.5f}")


if __name__ == "__main__":
    main()
