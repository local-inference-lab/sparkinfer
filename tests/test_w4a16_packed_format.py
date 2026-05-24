from __future__ import annotations

import pytest
import torch

from b12x.cute.fp4 import swizzle_block_scale
from b12x.integration.tp_moe import prepare_b12x_fp4_moe_weights
from b12x.moe.fused.w4a16.prepare import (
    prepare_w4a16_compressed_tensors_weights,
    prepare_w4a16_modelopt_nvfp4_weights as prepare_w4a16_weights,
    prepare_w4a16_mxfp4_native_weights,
    prepare_w4a16_packed_weights,
)


def _positive_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.rand(shape, device="cuda") * 1.75 + 0.03125).to(torch.float8_e4m3fn)


def test_prepare_fp4_moe_weights_modelopt_runtime_alphas_fuses_input_scales() -> None:
    w1_global_scale = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    w2_global_scale = torch.tensor([3.0, 5.0], dtype=torch.float32)
    a1_gscale = torch.tensor([0.5, 0.25], dtype=torch.float32)
    a2_gscale = torch.tensor([1.0 / 3.0, 0.2], dtype=torch.float32)

    prepared = prepare_b12x_fp4_moe_weights(
        source_format="modelopt_nvfp4",
        w1_global_scale=w1_global_scale,
        a1_gscale=a1_gscale,
        w2_global_scale=w2_global_scale,
        a2_gscale=a2_gscale,
        prepare_runtime_alphas=True,
    )

    assert prepared.w1_runtime_alphas is not None
    assert prepared.w2_runtime_alphas is not None
    w1_runtime = prepared.w1_runtime_alphas
    w2_runtime = prepared.w2_runtime_alphas
    assert w1_runtime.shape == (2,)
    assert w2_runtime.shape == (2,)
    torch.testing.assert_close(w1_runtime, torch.tensor([4.0, 16.0]))
    torch.testing.assert_close(w2_runtime, torch.tensor([9.0, 25.0]))


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
    assert prepared.source_format == "modelopt_nvfp4"
    assert prepared.w13.is_contiguous()
    assert prepared.w2.is_contiguous()
    assert prepared.w13_scale.is_contiguous()
    assert prepared.w2_scale.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
def test_modelopt_nvfp4_preparation_packs_runtime_weights(
    activation: str,
    params_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260523)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    actual = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
    )

    assert actual.weight_layout == "packed"
    assert actual.source_format == "modelopt_nvfp4"
    assert actual.w13.dtype == torch.int32
    assert actual.w2.dtype == torch.int32
    assert actual.w13.data_ptr() != w13.data_ptr()
    assert actual.w2.data_ptr() != w2.data_ptr()
    assert actual.w13.shape != w13.shape
    assert actual.w2.shape != w2.shape
    assert actual.w13_scale.dtype == torch.float8_e4m3fn
    assert actual.w2_scale.dtype == torch.float8_e4m3fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_legacy_modelopt_source_format_is_not_accepted() -> None:
    torch.manual_seed(20260523)
    experts = 3
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        activation="relu2",
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    with pytest.raises(ValueError, match="modelopt_nvfp4"):
        prepare_w4a16_packed_weights(
            w13,
            w13_blockscale,
            w13_global_scale,
            w2,
            w2_blockscale,
            w2_global_scale,
            activation="relu2",
            source_format="modelopt",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
def test_compressed_tensors_source_matches_reciprocal_modelopt_nvfp4_contract(
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
    )
    compressed_tensors = prepare_w4a16_compressed_tensors_weights(
        w13,
        w13_blockscale,
        w13_weight_global_scale,
        w2,
        w2_blockscale,
        w2_weight_global_scale,
        activation=activation,
        params_dtype=params_dtype,
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
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
def test_mxfp4_native_scales_expand_to_w4a16_scale_grid(
    activation: str,
    params_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260524)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13_rows = intermediate_size * (2 if activation == "silu" else 1)
    w13, _, w2, _ = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_native_scale = _positive_fp8((experts, w13_rows, hidden_size // 32))
    w2_native_scale = _positive_fp8((experts, hidden_size, intermediate_size // 32))
    w13_blockscale = swizzle_block_scale(w13_native_scale.repeat_interleave(2, dim=2))
    w2_blockscale = swizzle_block_scale(w2_native_scale.repeat_interleave(2, dim=2))
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)

    expected = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
    )
    actual = prepare_w4a16_mxfp4_native_weights(
        w13,
        w13_native_scale,
        w13_global_scale,
        w2,
        w2_native_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
    )

    assert actual.source_format == "mxfp4_native"
    assert actual.weight_layout == "packed"
    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_integration_packed_preparation_uses_raw_a16_alpha_contract(
    monkeypatch: pytest.MonkeyPatch,
    activation: str,
) -> None:
    torch.manual_seed(20260522)
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    w2_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(torch.float32)
    a1_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)
    a2_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)

    expected = prepare_w4a16_weights(
        w13.clone(),
        w13_blockscale,
        w13_alphas,
        w2.clone(),
        w2_blockscale,
        w2_alphas,
        activation=activation,
    )
    prepared = prepare_b12x_fp4_moe_weights(
        source_format="modelopt_nvfp4",
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_global_scale=w13_alphas,
        a1_gscale=a1_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_global_scale=w2_alphas,
        a2_gscale=a2_gscale,
        activation=activation,
        params_dtype=torch.bfloat16,
        prepare_w4a16=True,
        reuse_input_storage=True,
    )
    actual = prepared.w4a16

    assert actual is not None
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_integration_modelopt_nvfp4_preparation_uses_raw_weight_global_scales(
    activation: str,
) -> None:
    torch.manual_seed(20260523)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    w2_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    a1_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)
    a2_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)

    expected = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_alphas,
        w2,
        w2_blockscale,
        w2_alphas,
        activation=activation,
    )
    prepared = prepare_b12x_fp4_moe_weights(
        source_format="modelopt_nvfp4",
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_global_scale=w13_alphas,
        a1_gscale=a1_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_global_scale=w2_alphas,
        a2_gscale=a2_gscale,
        activation=activation,
        params_dtype=torch.bfloat16,
        prepare_runtime_alphas=True,
        prepare_w4a16=True,
    )
    actual = prepared.w4a16

    assert actual is not None
    assert prepared.w1_runtime_alphas is not None
    assert prepared.w2_runtime_alphas is not None
    torch.testing.assert_close(prepared.w1_runtime_alphas, w13_alphas / a1_gscale)
    torch.testing.assert_close(prepared.w2_runtime_alphas, w2_alphas / a2_gscale)
    assert actual.weight_layout == "packed"
    assert actual.source_format == "modelopt_nvfp4"
    for name in (
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_integration_modelopt_nvfp4_preparation_can_reuse_input_storage(
    activation: str,
) -> None:
    torch.manual_seed(20260524)
    experts, hidden_size, intermediate_size = 3, 128, 128
    w13, w13_blockscale, w2, w2_blockscale = _make_case(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    w2_alphas = (torch.rand(experts, device="cuda") * 0.5 + 0.25).to(
        torch.float32
    )
    a1_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)
    a2_gscale = (torch.rand(experts, device="cuda") * 0.5 + 0.75).to(torch.float32)

    expected_prepared = prepare_b12x_fp4_moe_weights(
        source_format="modelopt_nvfp4",
        w1_fp4=w13.clone(),
        w1_blockscale=w13_blockscale,
        w1_global_scale=w13_alphas,
        a1_gscale=a1_gscale,
        w2_fp4=w2.clone(),
        w2_blockscale=w2_blockscale,
        w2_global_scale=w2_alphas,
        a2_gscale=a2_gscale,
        activation=activation,
        params_dtype=torch.bfloat16,
        prepare_w4a16=True,
    )
    actual_prepared = prepare_b12x_fp4_moe_weights(
        source_format="modelopt_nvfp4",
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_global_scale=w13_alphas,
        a1_gscale=a1_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_global_scale=w2_alphas,
        a2_gscale=a2_gscale,
        activation=activation,
        params_dtype=torch.bfloat16,
        prepare_w4a16=True,
        reuse_input_storage=True,
    )
    expected = expected_prepared.w4a16
    actual = actual_prepared.w4a16

    assert expected is not None
    assert actual is not None
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
