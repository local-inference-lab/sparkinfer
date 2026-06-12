"""Oracle gate for the W4A8 throughput-tier MXFP8 activation quant kernel.

The kernel must match ``quant_dequant_mxfp8_torch`` (the spec shared with the
in-kernel ``quantize_block_fp8_mx``) bit-for-bit: identical UE8M0 scale bytes
(exponent-bump ceil of amax/448, zero block -> byte 0) and identical E4M3
payload bytes, hence identical dequantized values.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from b12x.cute.fp4 import (
    _ue8m0_output_scale_torch,
    pow2_ceil_ue8m0_torch,
    quant_dequant_mxfp8_torch,
)
from b12x.moe.fused.w4a8.act import silu_mul_mxfp8_quantize_rows
from b12x.moe.fused.w4a8.quant import mxfp8_quantize_rows


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")


def _oracle_bytes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference payload/scale bytes via the torch oracle helpers."""
    m, k = x.shape
    blocked = x.to(torch.float32).reshape(m, k // 32, 32)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    inv448 = torch.tensor(1.0 / 448.0, dtype=torch.float32, device=x.device)
    _, byte = pow2_ceil_ue8m0_torch(block_max * inv448)
    inv = _ue8m0_output_scale_torch(byte)
    payload = (
        (blocked * inv)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(m, k)
    )
    return payload, byte.reshape(m, k // 32).to(torch.uint8)


@pytest.mark.parametrize("shape", [(1, 32), (3, 64), (16, 4096), (257, 1024)])
def test_mxfp8_rows_match_oracle_bytes(shape: tuple[int, int]) -> None:
    _skip_if_unavailable()
    m, k = shape
    torch.manual_seed(m * 1000 + k)
    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 4.0).to(torch.bfloat16)
    # Adversarial content: zero blocks, single huge values, tiny values, and
    # amax values that land exactly on 448 * 2^j (power-of-two scale boundary).
    if m >= 3:
        x[0, :32] = 0.0
        x[1, 0] = 30000.0
        x[2, :] = 2.0**-20
        x[0, 32 : min(64, k)] = 448.0
    q, sf = mxfp8_quantize_rows(x)

    ref_payload, ref_sf = _oracle_bytes(x)
    assert torch.equal(sf, ref_sf)
    assert torch.equal(q.view(torch.uint8), ref_payload.view(torch.uint8))

    # Dequant equality against the roundtrip oracle (redundant with the byte
    # equality above, but anchors the value-domain contract).
    scale = torch.exp2(sf.to(torch.float32) - 127.0)
    scale = torch.where(sf == 0, torch.zeros_like(scale), scale)
    deq = q.float().reshape(m, k // 32, 32) * scale.reshape(m, k // 32, 1)
    torch.testing.assert_close(
        deq.reshape(m, k),
        quant_dequant_mxfp8_torch(x.float()),
        atol=0.0,
        rtol=0.0,
    )


def test_mxfp8_rows_preallocated_outputs_and_validation() -> None:
    _skip_if_unavailable()
    device = torch.device("cuda")
    x = torch.randn(4, 128, device=device).to(torch.bfloat16)
    q = torch.empty(4, 128, dtype=torch.float8_e4m3fn, device=device)
    sf = torch.empty(4, 4, dtype=torch.uint8, device=device)
    q2, sf2 = mxfp8_quantize_rows(x, out_values=q, out_scales=sf)
    assert q2 is q and sf2 is sf
    with pytest.raises(ValueError, match="divisible"):
        mxfp8_quantize_rows(torch.randn(2, 33, device=device).to(torch.bfloat16))


def _act_oracle_y(fc1: torch.Tensor) -> torch.Tensor:
    """silu(gate)*up in fp32, the moe_reference_w4a8_mx convention."""
    n = fc1.shape[1] // 2
    up = fc1[:, :n].float()
    gate = fc1[:, n:].float()
    return torch.sigmoid(gate) * gate * up


def _dequant_mx(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    m, k = q.shape
    scale = torch.where(
        sf == 0,
        torch.zeros((), device=q.device),
        torch.exp2(sf.to(torch.float32) - 127.0),
    )
    return (q.float().view(m, k // 32, 32) * scale.unsqueeze(-1)).view(m, k)


@pytest.mark.parametrize("shape", [(1, 64), (16, 2048), (37, 512)])
def test_silu_mul_mxfp8_matches_oracle(shape: tuple[int, int]) -> None:
    """Fused silu(gate)*up + quant vs torch silu/mul + the quant oracle.

    The quant bit-math is shared with mxfp8_quantize_rows (byte-exact given
    identical inputs); the only divergence source is the fp32 sigmoid/mul
    (libdevice exp vs torch's CUDA sigmoid), so the dequant gate allows one
    E4M3 ulp where the activations differ in the last fp32 ulp.
    """
    _skip_if_unavailable()
    rows, two_n = shape
    n = two_n // 2
    torch.manual_seed(rows * 1000 + two_n)
    device = torch.device("cuda")
    fc1 = (torch.randn(rows, two_n, device=device) * 8.0).to(torch.bfloat16)
    if rows >= 3:
        fc1[0, n : n + 32] = 0.0      # zero gate block -> zero outputs
        fc1[1, 0] = 30000.0           # huge up
        fc1[2, :n] = 2.0**-14         # tiny up row
    q, sf = silu_mul_mxfp8_quantize_rows(fc1)

    y_ref = _act_oracle_y(fc1)
    ref_dq = quant_dequant_mxfp8_torch(y_ref)
    got_dq = _dequant_mx(q, sf)
    # One E4M3 ulp at the block scale = 2^(sf-127) * 2^-3 relative to the
    # payload magnitude; gate elementwise against that bound.
    scale = torch.where(
        sf == 0, torch.zeros((), device=device),
        torch.exp2(sf.to(torch.float32) - 127.0),
    )
    ulp = (scale * 2.0 ** -3).repeat_interleave(32, dim=1)
    err = (got_dq - ref_dq).abs()
    assert torch.all(err <= ulp + 1e-30), (
        f"max err {err.max().item()} vs ulp bound {ulp.max().item()}"
    )
    # The overwhelming majority of bytes must be bit-identical.
    ref_q, ref_sf = _oracle_bytes(y_ref)
    byte_match = (
        (q.view(torch.uint8) == ref_q.view(torch.uint8)).float().mean().item()
    )
    sf_match = (sf == ref_sf).float().mean().item()
    assert byte_match > 0.999, byte_match
    assert sf_match > 0.999, sf_match


def test_silu_mul_mxfp8_valid_rows_guard() -> None:
    """Rows at/beyond the device valid_rows scalar are left untouched."""
    _skip_if_unavailable()
    device = torch.device("cuda")
    torch.manual_seed(11)
    rows, two_n = 8, 256
    n = two_n // 2
    fc1 = (torch.randn(rows, two_n, device=device) * 4.0).to(torch.bfloat16)
    live = 5
    q = torch.full((rows, n), 0xAB, dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    sf = torch.full((rows, n // 32), 0xCD, dtype=torch.uint8, device=device)
    valid = torch.tensor([live], dtype=torch.int32, device=device)
    silu_mul_mxfp8_quantize_rows(fc1, out_values=q, out_scales=sf, valid_rows=valid)

    q_full, sf_full = silu_mul_mxfp8_quantize_rows(fc1)
    assert torch.equal(
        q.view(torch.uint8)[:live], q_full.view(torch.uint8)[:live]
    )
    assert torch.equal(sf[:live], sf_full[:live])
    assert torch.all(q.view(torch.uint8)[live:] == 0xAB)
    assert torch.all(sf[live:] == 0xCD)
