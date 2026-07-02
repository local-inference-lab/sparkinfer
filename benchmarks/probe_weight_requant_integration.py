"""Integration gate: the REAL b12x packer now re-quantizes fp32 checkpoint
scales onto exact UE8M0, achieving DeepGEMM-parity weight error -- and leaves
the already-UE8M0 (e8m0) path byte-for-byte unchanged.

Run with the assigned GPU:
  python benchmarks/probe_weight_requant_integration.py

GPU serialization, when enabled, is managed outside this command.
"""

import torch

from b12x.gemm.block_fp8_linear import pack_block_fp8_linear_weight_mxfp8
from b12x.gemm.wo_projection import (
    dequantize_mxfp8_rows_torch,
    pack_fp8_block_scaled_weight_mxfp8,
)


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)).item()


def relfro(a, b):
    return ((a - b).flatten().float().norm() / b.flatten().float().norm()).item()


def deepseek_checkpoint(N, K, dev):
    """(w_fp8, s_fp32, w_orig) with arbitrary fp32 128x128 block scales."""
    w_orig = (torch.randn(N, K, device=dev) * 0.2).to(torch.bfloat16).float()
    wv = w_orig.view(N // 128, 128, K // 128, 128)
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    s = amax / 448.0
    w_fp8 = (wv / s).to(torch.float8_e4m3fn).view(N, K)
    s = s.view(N // 128, K // 128).to(torch.float32)
    C = w_fp8.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return w_fp8, s, w_orig, C


def main():
    torch.manual_seed(0)
    dev = "cuda"

    print("=" * 74)
    print("(1) DENSE ng=1: arbitrary fp32 checkpoint scale -> re-quantized UE8M0")
    N, K = 4096, 4096
    w_fp8, s_fp32, w_orig, C = deepseek_checkpoint(N, K, dev)
    packed = pack_block_fp8_linear_weight_mxfp8(w_fp8, s_fp32)
    deq = dequantize_mxfp8_rows_torch(packed.weight.values, packed.weight.scale_rows)
    print(f"    weight cos vs bf16 = {cos(deq, w_orig):.6f}  rel_fro = {relfro(deq, w_orig):.5f}")
    print(f"    weight cos vs ckpt = {cos(deq, C):.6f}  rel_fro = {relfro(deq, C):.5f}")
    print(f"    (was ~0.102 rel_fro with the old round-scale path; ~0.037 is DeepGEMM-parity)")
    ok1 = relfro(deq, w_orig) < 0.05
    print(f"    PASS (< 0.05) = {ok1}")

    print("=" * 74)
    print("(2) e8m0 passthrough: already-UE8M0 scale must be UNCHANGED (keep values)")
    # build an e8m0 (power-of-two) scale; values quantized against it
    w_orig2 = (torch.randn(N, K, device=dev) * 0.2).to(torch.bfloat16).float()
    wv = w_orig2.view(N // 128, 128, K // 128, 128)
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    s_pow2 = torch.exp2(exp)
    w_fp8_2 = (wv / s_pow2).to(torch.float8_e4m3fn).view(N, K)
    s_e8m0 = (exp.view(N // 128, K // 128) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    packed2 = pack_block_fp8_linear_weight_mxfp8(w_fp8_2, s_e8m0)
    # values must be byte-identical to the input fp8 (no re-quant on e8m0 path)
    same = (packed2.weight.values.view(torch.uint8) == w_fp8_2.view(torch.uint8)).all().item()
    print(f"    fp8 values byte-identical to checkpoint = {same}")
    print(f"    PASS = {same}")

    print("=" * 74)
    print("(3) WO num_groups>1: arbitrary fp32 per-group block scale -> re-quantized")
    groups, mg, kg = 4, 1024, 512
    w_orig3 = (torch.randn(groups * mg, kg, device=dev) * 0.2).to(torch.bfloat16).float()
    wv = w_orig3.view(groups * mg // 128, 128, kg // 128, 128)
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    s = (amax / 448.0)
    w_fp8_3 = (wv / s).to(torch.float8_e4m3fn).view(groups * mg, kg)
    s3 = s.view(groups * (mg // 128), kg // 128).to(torch.float32)  # [g*m_tiles, k_tiles]
    C3 = w_fp8_3.float() * s.view(groups * mg // 128, kg // 128).repeat_interleave(128, 0).repeat_interleave(128, 1)
    packed3 = pack_fp8_block_scaled_weight_mxfp8(w_fp8_3, s3, m=mg, k=kg, num_groups=groups)
    deq3 = dequantize_mxfp8_rows_torch(packed3.values, packed3.scale_rows)  # [mg, kg, groups]
    deq3 = deq3.permute(2, 0, 1).reshape(groups * mg, kg)
    print(f"    weight cos vs bf16 = {cos(deq3, w_orig3):.6f}  rel_fro = {relfro(deq3, w_orig3):.5f}")
    print(f"    weight cos vs ckpt = {cos(deq3, C3):.6f}  rel_fro = {relfro(deq3, C3):.5f}")
    ok3 = relfro(deq3, w_orig3) < 0.06
    print(f"    PASS (< 0.06) = {ok3}")

    print("=" * 74)
    print(f"ALL PASS = {ok1 and same and ok3}")


if __name__ == "__main__":
    main()
