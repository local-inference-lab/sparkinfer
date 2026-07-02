"""Numeric gate: is b12x's MXFP8 activation quant at parity with DeepGEMM?

Compares b12x `quantize_block_fp8_linear_input_mxfp8` against DeepGEMM's
reference `per_token_cast_to_fp8` (the 1d1d dense path quantizer) on identical
BF16 inputs, at both gran_k=32 and gran_k=128, and reports:
  - byte-exact fp8 match + e8m0 scale match (b12x vs each DeepGEMM granularity)
  - reconstruction error of each quant vs the original BF16 (accuracy)
  - which gran_k b12x actually implements

Run with the assigned GPU:
  python benchmarks/probe_quant_parity_deepgemm.py

GPU serialization, when enabled, is managed outside this command.
"""

import torch

from b12x.gemm.block_fp8_linear import quantize_block_fp8_linear_input_mxfp8


# ---- DeepGEMM reference (verbatim from deepgemm-other/deep_gemm/utils/math.py) ----
def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    return x_fp8, sf  # sf is fp32 power-of-two, shape [m, n//gran_k]


def sf_fp32_to_e8m0_u8(sf_fp32: torch.Tensor) -> torch.Tensor:
    # ue8m0 power-of-two -> stored exponent byte (exp+127). log2 is exact here.
    return (torch.log2(sf_fp32).round().clamp(-127, 127) + 127).to(torch.uint8)


def dequant_per32(values_fp8: torch.Tensor, scale_e8m0_u8: torch.Tensor, gran_k: int) -> torch.Tensor:
    # scale_e8m0_u8: [m, n//gran_k] exponent bytes; broadcast across gran_k cols.
    m, n = values_fp8.shape
    scale = torch.exp2((scale_e8m0_u8.to(torch.int32) - 127).to(torch.float32))  # [m, n//gran_k]
    scale = scale.repeat_interleave(gran_k, dim=1)[:, :n]
    return values_fp8.to(torch.float32) * scale


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)).item()


def report(name: str, recon: torch.Tensor, ref: torch.Tensor) -> None:
    err = (recon - ref).abs()
    print(f"  {name:28s} cos={cos(recon, ref):.6f}  max_abs={err.max():.4g}  "
          f"mean_abs={err.mean():.4g}  rel_fro={(err.norm()/ref.norm()).item():.4g}")


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    M, K = 64, 5376  # Nemotron down-proj K (5376 = 42*128, divisible by 32 and 128)
    x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 3.0)
    xf = x.float()

    # --- b12x quant ---
    rows = quantize_block_fp8_linear_input_mxfp8(x)
    b_vals = rows.values.view(torch.float8_e4m3fn)        # [M, K]
    b_sf = rows.scale_rows.view(torch.uint8)[0]           # [M, K//32]
    print(f"b12x scale_rows shape {tuple(b_sf.shape)} -> implies gran_k={K // b_sf.shape[1]}")

    # --- DeepGEMM quant at both granularities ---
    dg32_vals, dg32_sf_f = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
    dg128_vals, dg128_sf_f = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128)
    dg32_sf = sf_fp32_to_e8m0_u8(dg32_sf_f)
    dg128_sf = sf_fp32_to_e8m0_u8(dg128_sf_f)

    # --- byte-exact parity: b12x vs DeepGEMM gran_k=32 ---
    print("\n[byte-exact] b12x vs DeepGEMM gran_k=32")
    v_eq = (b_vals.view(torch.uint8) == dg32_vals.view(torch.uint8)).float().mean().item()
    s_eq = (b_sf == dg32_sf).float().mean().item()
    print(f"  fp8 byte match = {v_eq*100:.3f}%   e8m0 scale match = {s_eq*100:.3f}%")

    print("[byte-exact] b12x scale vs DeepGEMM gran_k=128 (after 4x broadcast)")
    dg128_sf_bc = dg128_sf.repeat_interleave(4, dim=1)  # [M, K//32]
    s128_eq = (b_sf == dg128_sf_bc).float().mean().item()
    print(f"  e8m0 scale match (b12x per-32 vs DG per-128 broadcast) = {s128_eq*100:.3f}%")

    # --- reconstruction accuracy vs original bf16 ---
    print("\n[reconstruction error vs original bf16]")
    report("b12x (per-32)", dequant_per32(b_vals, b_sf, 32), xf)
    report("DeepGEMM gran_k=32", dequant_per32(dg32_vals, dg32_sf, 32), xf)
    report("DeepGEMM gran_k=128", dequant_per32(dg128_vals, dg128_sf, 128), xf)


if __name__ == "__main__":
    main()
