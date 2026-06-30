from __future__ import annotations

import sys
import types

import torch

from benchmarks.benchmark_moe import (
    _dequant_mxfp4_expert,
    _dequant_nvfp4_expert,
    _requantize_flashinfer_mxfp4_stack,
)
from b12x.cute.fp4 import swizzle_block_scale


def test_dequant_mxfp4_expert_applies_e8m0_k32_scale() -> None:
    # Low nibble 1 is +0.5 and high nibble 2 is +1.0. E8M0 byte 128
    # contributes 2**(128 - 127) = 2.
    packed = torch.full((1, 16), 0x21, dtype=torch.uint8)
    scales = torch.full((1, 1), 128, dtype=torch.uint8)

    result = _dequant_mxfp4_expert(packed, scales, rows=1, cols=32).float()

    expected = torch.tensor([1.0, 2.0] * 16).reshape(1, 32)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_dequant_nvfp4_expert_applies_k16_and_global_scales() -> None:
    packed = torch.full((1, 8), 0x21, dtype=torch.uint8)
    logical_scales = torch.full((1, 1), 2.0, dtype=torch.float8_e4m3fn)
    swizzled_scales = swizzle_block_scale(logical_scales)

    result = _dequant_nvfp4_expert(
        packed,
        swizzled_scales,
        rows=1,
        cols=16,
        global_scale=3.0,
    ).float()

    expected = torch.tensor([3.0, 6.0] * 8).reshape(1, 16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_flashinfer_requantization_swaps_checkpoint_native_w31(monkeypatch) -> None:
    captured: list[torch.Tensor] = []

    def fake_mxfp4_quantize(logical: torch.Tensor):
        captured.append(logical.clone())
        rows, cols = logical.shape
        return (
            torch.zeros(rows, cols // 2, dtype=torch.uint8),
            torch.zeros(rows, cols // 32, dtype=torch.uint8),
        )

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.mxfp4_quantize = fake_mxfp4_quantize
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)

    # Four constant rows decode to [0.5, 1.0, 1.5, 2.0]. A checkpoint-native
    # w31 stack is [gate; up], while FlashInfer consumes [up; gate].
    packed = torch.tensor([0x11, 0x22, 0x33, 0x44], dtype=torch.uint8)
    packed = packed.view(1, 4, 1).expand(1, 4, 16).contiguous()
    scales = torch.full((1, 4, 1), 127, dtype=torch.uint8)

    quantized, quantized_scales = _requantize_flashinfer_mxfp4_stack(
        packed,
        scales,
        rows=4,
        cols=32,
        swap_halves=True,
    )

    assert quantized.shape == (1, 4, 16)
    assert quantized_scales.shape == (1, 4, 1)
    assert len(captured) == 1
    expected_rows = torch.tensor([1.5, 2.0, 0.5, 1.0], dtype=torch.bfloat16)
    torch.testing.assert_close(captured[0][:, 0], expected_rows, rtol=0, atol=0)
