from __future__ import annotations

import pytest
import torch

from b12x.cute.fp4 import swizzle_block_scale
from b12x.moe.fused.w4a16.prepare import (
    prepare_w4a16_packed_weights as prepare_w4a16_weights,
)


def _positive_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.rand(shape, device="cuda") * 1.75 + 0.03125).to(torch.float8_e4m3fn)


def _make_case(
    *,
    experts: int = 3,
    hidden_size: int = 128,
    intermediate_size: int = 128,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    is_gated = activation == "silu"
    w13_rows = intermediate_size * (2 if is_gated else 1)
    w13 = torch.randint(
        0,
        256,
        (experts, w13_rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_blockscale = swizzle_block_scale(
        _positive_fp8((experts, w13_rows, hidden_size // 16))
    )
    w2_blockscale = swizzle_block_scale(
        _positive_fp8((experts, hidden_size, intermediate_size // 16))
    )
    return w13, w13_blockscale, w2, w2_blockscale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
def test_packed_weight_preparation_shapes_and_dtypes(
    activation: str,
    params_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260514)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13_rows = intermediate_size * (2 if activation == "silu" else 1)
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    prepared = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
    )

    assert prepared.w13.shape == (
        experts,
        hidden_size // 16,
        (w13_rows // 64) * 128,
    )
    assert prepared.w2.shape == (
        experts,
        intermediate_size // 16,
        (hidden_size // 64) * 128,
    )
    assert prepared.w13_scale.dtype == torch.float8_e4m3fn
    assert prepared.w2_scale.dtype == torch.float8_e4m3fn
    assert prepared.w13_global_scale.dtype == torch.float32
    assert prepared.w2_global_scale.dtype == torch.float32
    assert prepared.workspace.dtype == torch.int32
    assert prepared.params_dtype == params_dtype
    assert prepared.source_format == "modelopt"
    assert prepared.w13.is_contiguous()
    assert prepared.w2.is_contiguous()
    assert prepared.w13_scale.is_contiguous()
    assert prepared.w2_scale.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
def test_packed_compressed_tensors_source_matches_reciprocal_modelopt_contract(
    activation: str,
    params_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260520)
    experts = 3
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        activation=activation,
    )
    w13_weight_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 1.25).to(
        torch.float32
    )
    w2_weight_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 1.25).to(
        torch.float32
    )

    modelopt = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        (1.0 / w13_weight_global_scale).to(torch.float32),
        w2,
        w2_blockscale,
        (1.0 / w2_weight_global_scale).to(torch.float32),
        activation=activation,
        params_dtype=params_dtype,
        source_format="modelopt",
    )
    compressed_tensors = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_weight_global_scale,
        w2,
        w2_blockscale,
        w2_weight_global_scale,
        activation=activation,
        params_dtype=params_dtype,
        source_format="compressed_tensors",
    )

    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(modelopt, name), getattr(compressed_tensors, name)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_packed_weight_preparation_can_reuse_input_storage(activation: str) -> None:
    torch.manual_seed(20260521)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    expected = prepare_w4a16_weights(
        w13.clone(),
        w13_blockscale,
        w13_global_scale,
        w2.clone(),
        w2_blockscale,
        w2_global_scale,
        activation=activation,
    )
    actual = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        reuse_input_storage=True,
    )

    assert actual.w13.data_ptr() == w13.data_ptr()
    assert actual.w2.data_ptr() == w2.data_ptr()
    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
