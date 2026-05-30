"""Adversarial quant-parity probe: hunt for ANY divergence between b12x's
MXFP8 quantization and DeepGEMM's reference, focusing on the cases the random
test cannot exercise:

  (1) ROUNDING: does b12x's exp2(clamp(ceil(log2(amax/448)),-127,127)) ever
      differ from DeepGEMM's bit-exact ceil_to_ue8m0(amax/448)?  Swept over a
      dense range incl. exact powers of two and just-above/below boundaries,
      in pure torch (the formula) AND through the real b12x Triton kernel
      (libdevice log2/ceil/exp2).
  (2) WEIGHT round-vs-ceil: _scale_to_e8m0_u8 (round) vs _scale_u8_from_max_abs
      (ceil) divergence on FP32 block scales.
  (3) GRAN_K on OUTLIER-channel data: where 32 vs 128 actually matters --
      reconstruction + GEMM-vs-fp32-oracle for b12x(32) vs DeepGEMM(128).

Run: CUDA_VISIBLE_DEVICES=0 trafficcontrol -- \
       ~/projects/b12x/.venv/bin/python benchmarks/probe_quant_parity_adversarial.py
"""

import torch

from b12x.gemm.block_fp8_linear import quantize_block_fp8_linear_input_mxfp8
from b12x.gemm.wo_projection import _scale_to_e8m0_u8, _scale_u8_from_max_abs


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def b12x_scale_formula(amax: torch.Tensor) -> torch.Tensor:
    """Exactly b12x's Triton/torch scale: exp2(clamp(ceil(log2(amax/448)),-127,127))."""
    safe = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    e = torch.clamp(torch.ceil(torch.log2(safe)), -127.0, 127.0)
    return torch.exp2(e)


def test_rounding_formula() -> None:
    print("=" * 78)
    print("(1a) ROUNDING formula sweep: b12x ceil(log2)+exp2  vs  bit-exact ceil_to_ue8m0")
    dev = "cuda"
    # exact powers of two (sf == 2^k exactly): amax = 448 * 2^k
    ks = torch.arange(-30, 30, device=dev, dtype=torch.float32)
    pow2 = 448.0 * torch.exp2(ks)
    # boundaries: just below / just above each power of two scale
    eps_lo = pow2 * (1 - 1e-6)
    eps_hi = pow2 * (1 + 1e-6)
    # dense log sweep
    dense = torch.logspace(-12, 12, 200000, device=dev, dtype=torch.float32)
    amax = torch.cat([pow2, eps_lo, eps_hi, dense])

    dg = ceil_to_ue8m0(amax / 448.0)            # bit-exact power-of-two scale
    b12 = b12x_scale_formula(amax)              # b12x formula
    # compare as exponents (avoid fp equality pitfalls)
    dg_e = torch.log2(dg).round().to(torch.int64)
    b12_e = torch.log2(b12).round().to(torch.int64)
    # clamp dg to b12x's [-127,127] range for fair comparison (b12x clamps)
    dg_e = dg_e.clamp(-127, 127)
    ndiff = (dg_e != b12_e).sum().item()
    print(f"   total={amax.numel()}  exponent mismatches={ndiff}")
    if ndiff:
        idx = (dg_e != b12_e).nonzero().flatten()[:10]
        for i in idx.tolist():
            print(f"     amax={amax[i].item():.6e}  sf=amax/448={amax[i].item()/448:.6e}  "
                  f"DG_exp={dg_e[i].item()}  b12x_exp={b12_e[i].item()}")
    else:
        print("   -> IDENTICAL across all powers of two, boundaries, and dense sweep.")


def test_rounding_triton() -> None:
    print("=" * 78)
    print("(1b) ROUNDING via REAL b12x Triton kernel on power-of-2-aligned input")
    dev = "cuda"
    M, K = 32, 256
    # craft rows whose per-32 amax is exactly 448 * 2^k for varied k
    x = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
    for c in range(K // 32):
        k = (c % 20) - 10
        target = 448.0 * (2.0 ** k)
        x[:, c * 32] = torch.tensor(target, dtype=torch.bfloat16)  # one big element per group
        x[:, c * 32 + 1 : c * 32 + 32] = (torch.randn(M, 31, device=dev) * (target * 1e-3)).to(torch.bfloat16)
    xf = x.float()
    rows = quantize_block_fp8_linear_input_mxfp8(x)
    b_sf = rows.scale_rows.view(torch.uint8)[0].to(torch.int64) - 127  # exponent per 32-group
    # reference: bit-exact per-32
    blocked = xf.view(M, K // 32, 32)
    amax = blocked.abs().amax(dim=2).clamp(1e-4)
    dg = ceil_to_ue8m0(amax / 448.0)
    dg_e = torch.log2(dg).round().to(torch.int64).clamp(-127, 127)
    ndiff = (b_sf != dg_e).sum().item()
    print(f"   per-32 scale exponents compared={b_sf.numel()}  mismatches={ndiff}")
    if ndiff:
        bi, gi = (b_sf != dg_e).nonzero()[0].tolist()
        print(f"     e.g. amax={amax[bi,gi].item():.6e} b12x_exp={b_sf[bi,gi].item()} DG_exp={dg_e[bi,gi].item()}")
    else:
        print("   -> Triton kernel byte-exact with bit-exact ceil_to_ue8m0 even at boundaries.")


def test_round_vs_ceil() -> None:
    print("=" * 78)
    print("(2) _scale_to_e8m0_u8 (round) vs _scale_u8_from_max_abs (ceil) vs DeepGEMM ceil")
    dev = "cuda"
    # FP32 block scales as a checkpoint would carry: sf = blockamax/448 (NOT pre-rounded)
    amax = torch.logspace(-6, 4, 50000, device=dev, dtype=torch.float32)
    sf_fp32 = amax / 448.0
    # _scale_to_e8m0_u8 takes an already-scale tensor and rounds it to e8m0
    round_u8 = _scale_to_e8m0_u8(sf_fp32).to(torch.int64) - 127
    ceil_u8 = _scale_u8_from_max_abs(amax).to(torch.int64) - 127
    dg_e = torch.log2(ceil_to_ue8m0(sf_fp32)).round().to(torch.int64).clamp(-127, 127)
    n_rc = (round_u8 != ceil_u8).sum().item()
    n_cd = (ceil_u8 != dg_e).sum().item()
    print(f"   round vs ceil mismatches = {n_rc}/{amax.numel()} ({100*n_rc/amax.numel():.2f}%)")
    print(f"   ceil  vs DeepGEMM        = {n_cd}/{amax.numel()} ({100*n_cd/amax.numel():.2f}%)")
    # does round ever UNDERscale (round down) -> x/scale > 448 -> fp8 saturation?
    under = (round_u8 < dg_e).sum().item()
    print(f"   round produces SMALLER exponent than bit-exact ceil (=> fp8 saturation risk) "
          f"in {under}/{amax.numel()} ({100*under/amax.numel():.2f}%) cases")


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)).item()


def per_token_cast(x, gran_k):
    m, n = x.shape
    xv = x.view(m, n // gran_k, gran_k)
    amax = xv.abs().float().amax(dim=2).clamp(1e-4)
    sf = ceil_to_ue8m0(amax / 448.0)
    q = (xv * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n)
    deq = (q.float().view(m, n // gran_k, gran_k) * sf.unsqueeze(2)).view(m, n)
    return q, deq


def test_outlier_granularity() -> None:
    print("=" * 78)
    print("(3) GRAN_K on OUTLIER-channel data (where 32 vs 128 should matter)")
    dev = "cuda"
    M, K, N = 256, 4096, 4096
    x = (torch.randn(M, K, device=dev) * 0.5)
    # inject heavy per-channel outliers in ~3% of channels (activation-like)
    out_ch = torch.randperm(K, device=dev)[: K // 32]
    x[:, out_ch] *= 40.0
    xb = x.to(torch.bfloat16)
    xf = xb.float()
    w = (torch.randn(N, K, device=dev) * 0.3).to(torch.bfloat16)

    # b12x actual quant (gran_k=32)
    rows = quantize_block_fp8_linear_input_mxfp8(xb)
    b_vals = rows.values.view(torch.float8_e4m3fn)
    b_sf = rows.scale_rows.view(torch.uint8)[0]
    scale32 = torch.exp2((b_sf.to(torch.int32) - 127).float()).repeat_interleave(32, dim=1)
    b_deq = b_vals.float() * scale32

    _, dg32_deq = per_token_cast(xf, 32)
    _, dg128_deq = per_token_cast(xf, 128)

    print(f"   reconstruction cos vs bf16:  b12x(32)={cos(b_deq, xf):.6f}  "
          f"DG(32)={cos(dg32_deq, xf):.6f}  DG(128)={cos(dg128_deq, xf):.6f}")
    print(f"   reconstruction rel_fro:      b12x(32)={((b_deq-xf).norm()/xf.norm()).item():.5f}  "
          f"DG(32)={((dg32_deq-xf).norm()/xf.norm()).item():.5f}  "
          f"DG(128)={((dg128_deq-xf).norm()/xf.norm()).item():.5f}")
    # GEMM vs fp32 oracle (weight kept bf16 to isolate activation-quant granularity)
    oracle = xf @ w.float().T
    wf = w.float()
    print(f"   GEMM cos vs fp32 oracle:     b12x(32)={cos(b_deq @ wf.T, oracle):.6f}  "
          f"DG(32)={cos(dg32_deq @ wf.T, oracle):.6f}  DG(128)={cos(dg128_deq @ wf.T, oracle):.6f}")


def main():
    torch.manual_seed(0)
    test_rounding_formula()
    test_rounding_triton()
    test_round_vs_ceil()
    test_outlier_granularity()


if __name__ == "__main__":
    main()
