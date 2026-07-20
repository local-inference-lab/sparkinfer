from __future__ import annotations

import pytest
import torch

import b12x.integration.tp_moe as tp_moe
from b12x.cute.intrinsics import pack_grouped_fp4_values, swizzle_block_scale
from b12x.moe.fused.reference import (
    moe_reference_w4a16_f32,
    moe_reference_w4a16_fp4_e8m0_k32,
)
from b12x.moe.fused.reference_flashinfer import (
    prepare_flashinfer_trtllm_fp4_e8m0_k32_weights,
)
from b12x.moe.fused.micro import MoEMicroKernelBackend as NVFP4MoEMicroKernelBackend
from tests.w4a16_reference import moe_reference_w4a16


def _packed_fp4_constant(
    value: float,
    *,
    groups: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    dense = torch.full((groups, rows, cols), value, dtype=torch.float32)
    return pack_grouped_fp4_values(dense).permute(2, 0, 1).contiguous()


def _blockscale_constant(
    value: float,
    *,
    groups: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scales = torch.full(
        (groups, rows, cols // 16),
        value,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    return swizzle_block_scale(scales)


def _e8m0_blockscale_constant(
    value: float,
    *,
    groups: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scales = torch.full(
        (groups, rows, cols // 32),
        value,
        dtype=torch.float32,
    )
    return scales.to(torch.float8_e8m0fnu)


def _require_flashinfer_trtllm_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("requires FlashInfer TRT-LLM SM100+ kernels")
    try:
        from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
        from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache
    except Exception as exc:
        pytest.skip(f"requires FlashInfer TRT-LLM helpers: {exc}")
    return nvfp4_block_scale_interleave, get_w2_permute_indices_with_cache


def _vllm_style_flashinfer_trtllm_prepare_expected(
    w13_fp4: torch.Tensor,
    w13_e8m0_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nvfp4_block_scale_interleave, get_w2_permute_indices_with_cache = _require_flashinfer_trtllm_cuda()
    w13_u8 = w13_fp4.view(torch.uint8).contiguous()
    w2_u8 = w2_fp4.view(torch.uint8).contiguous()
    w13_s_u8 = w13_e8m0_scale.view(torch.uint8).contiguous()
    w2_s_u8 = w2_e8m0_scale.view(torch.uint8).contiguous()

    # vLLM native DeepSeek loading stores W13 as [w1/gate, w3/up].
    # Its TRT-LLM conversion interleaves that into [w3_0, w1_0, ...].
    w1_weight = w13_u8[:, :intermediate_size, :]
    w3_weight = w13_u8[:, intermediate_size:, :]
    w13_u8 = torch.stack([w3_weight, w1_weight], dim=2).reshape(w13_u8.shape)

    w1_scale = w13_s_u8[:, :intermediate_size, :]
    w3_scale = w13_s_u8[:, intermediate_size:, :]
    w13_s_u8 = torch.stack([w3_scale, w1_scale], dim=2).reshape(w13_s_u8.shape)

    cache: dict = {}
    epilogue_tile_m = 128
    w13_perm = get_w2_permute_indices_with_cache(
        cache,
        w13_u8[0],
        epilogue_tile_m,
    ).to(w13_u8.device)
    w13_out = w13_u8[:, w13_perm].contiguous()

    w13_sf_perm = get_w2_permute_indices_with_cache(
        cache,
        w13_s_u8[0],
        epilogue_tile_m,
        num_elts_per_sf=16,
    ).to(w13_s_u8.device)
    w13_s = w13_s_u8[:, w13_sf_perm].contiguous()
    E, N_s, K_s = w13_s.shape
    w13_scale_out = (
        nvfp4_block_scale_interleave(w13_s.reshape(E * N_s, K_s))
        .reshape(E, 2 * intermediate_size, hidden_size // 32)
        .view(torch.float8_e4m3fn)
    )

    w2_perm = get_w2_permute_indices_with_cache(
        cache,
        w2_u8[0],
        epilogue_tile_m,
    ).to(w2_u8.device)
    w2_out = w2_u8[:, w2_perm].contiguous()

    w2_sf_perm = get_w2_permute_indices_with_cache(
        cache,
        w2_s_u8[0],
        epilogue_tile_m,
        num_elts_per_sf=16,
    ).to(w2_s_u8.device)
    w2_s = w2_s_u8[:, w2_sf_perm].contiguous()
    E2, N2_s, K2_s = w2_s.shape
    w2_scale_out = (
        nvfp4_block_scale_interleave(w2_s.reshape(E2 * N2_s, K2_s))
        .reshape(E2, hidden_size, intermediate_size // 32)
        .view(torch.float8_e4m3fn)
    )
    return w13_out, w13_scale_out, w2_out, w2_scale_out


def test_w4a16_reference_uses_bf16_activation_and_intermediate_without_activation_scales() -> None:
    experts, hidden, intermediate, topk = 1, 16, 16, 1
    x = torch.full((1, hidden), 0.25, dtype=torch.bfloat16)
    topk_ids = torch.zeros(1, topk, dtype=torch.int32)
    topk_weights = torch.ones(1, topk, dtype=torch.float32)

    w1_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )
    w1_blockscale = _blockscale_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_blockscale = _blockscale_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )

    actual = moe_reference_w4a16(
        x,
        w1_fp4,
        w1_blockscale,
        torch.ones(experts, dtype=torch.float32),
        w2_fp4,
        w2_blockscale,
        torch.ones(experts, dtype=torch.float32),
        topk_ids,
        topk_weights,
        experts,
        hidden,
        intermediate,
        activation="relu2",
    )

    torch.testing.assert_close(
        actual.float(),
        torch.full((1, hidden), 256.0, dtype=torch.float32),
    )


def test_w4a16_f32_oracle_uses_weight_only_scales_without_activation_quant() -> None:
    experts, hidden, intermediate, topk = 1, 16, 16, 1
    x = torch.full((1, hidden), 0.25, dtype=torch.bfloat16)
    topk_ids = torch.zeros(1, topk, dtype=torch.int32)
    topk_weights = torch.ones(1, topk, dtype=torch.float32)

    w1_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )
    w1_blockscale = _blockscale_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_blockscale = _blockscale_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )

    actual = moe_reference_w4a16_f32(
        x,
        w1_fp4,
        w1_blockscale,
        torch.full((experts,), 2.0, dtype=torch.float32),
        w2_fp4,
        w2_blockscale,
        torch.full((experts,), 3.0, dtype=torch.float32),
        topk_ids,
        topk_weights,
        experts,
        hidden,
        intermediate,
        activation="relu2",
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual,
        torch.full((1, hidden), 3072.0, dtype=torch.float32),
    )


def test_w4a16_fp4_e8m0_k32_oracle_uses_raw_k32_scale_grid() -> None:
    experts, hidden, intermediate, topk = 1, 32, 32, 1
    x = torch.full((1, hidden), 0.25, dtype=torch.bfloat16)
    topk_ids = torch.zeros(1, topk, dtype=torch.int32)
    topk_weights = torch.ones(1, topk, dtype=torch.float32)

    w1_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )
    w1_blockscale = _e8m0_blockscale_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    ).view(torch.uint8)
    w2_blockscale = _e8m0_blockscale_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    ).view(torch.uint8)

    actual = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w1_fp4,
        w1_blockscale,
        torch.full((experts,), 2.0, dtype=torch.float32),
        w2_fp4,
        w2_blockscale,
        torch.full((experts,), 3.0, dtype=torch.float32),
        topk_ids,
        topk_weights,
        experts,
        hidden,
        intermediate,
        activation="relu2",
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual,
        torch.full((1, hidden), 24576.0, dtype=torch.float32),
    )


def test_flashinfer_trtllm_fp4_e8m0_k32_prep_matches_vllm_deepseek_style() -> None:
    _require_flashinfer_trtllm_cuda()
    experts, hidden, intermediate = 2, 128, 128
    generator = torch.Generator(device="cuda").manual_seed(123)
    w13_fp4 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2_fp4 = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w13_scale = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2_scale = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w13_scale[0, 0, 0] = 254
    w2_scale[0, 0, 0] = 248

    expected = _vllm_style_flashinfer_trtllm_prepare_expected(
        w13_fp4,
        w13_scale,
        w2_fp4,
        w2_scale,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    actual = prepare_flashinfer_trtllm_fp4_e8m0_k32_weights(
        w13_fp4,
        w13_scale,
        w2_fp4,
        w2_scale,
        hidden,
        intermediate,
        scale_byte_clamp=None,
    )

    assert actual.w13_scale.dtype == torch.float8_e4m3fn
    assert actual.w2_scale.dtype == torch.float8_e4m3fn
    assert actual.w13_scale.numel() == experts * (2 * intermediate) * (hidden // 32)
    assert actual.w2_scale.numel() == experts * hidden * (intermediate // 32)
    torch.testing.assert_close(actual.w13, expected[0])
    torch.testing.assert_close(actual.w13_scale.view(torch.uint8), expected[1].view(torch.uint8))
    torch.testing.assert_close(actual.w2, expected[2])
    torch.testing.assert_close(actual.w2_scale.view(torch.uint8), expected[3].view(torch.uint8))


def test_flashinfer_fp4_e8m0_k32_oracle_matches_python_oracle_on_deepseek_v4_flash_layer() -> None:
    _require_flashinfer_trtllm_cuda()
    from benchmarks.benchmark_moe import (
        ActivationParams,
        MODEL_PROFILES,
        _cached_snapshot_path,
        build_model_spec,
        get_quant_mode_params,
        load_expert_weights,
        make_oracle_reference,
        make_profile_routed_inputs,
    )

    profile = MODEL_PROFILES["deepseek-v4-flash"]
    assert profile.hf_repo_id is not None
    model_path = _cached_snapshot_path(profile.hf_repo_id)
    if model_path is None:
        pytest.skip(f"DeepSeek V4 Flash checkpoint is not cached for {profile.hf_repo_id}")

    spec = build_model_spec(model_path, profile)
    weights = load_expert_weights(
        model_path,
        spec,
        layer_idx=profile.default_layer_idx,
        activation=profile.default_activation,
        checkpoint_family=profile.checkpoint_family,
        keep_flashinfer_oracle_copy=True,
    )
    assert weights.source_format == "fp4_e8m0_k32"
    assert weights.w13_layout == "w31"
    assert weights.oracle_w13_weight is not None
    assert weights.oracle_w13_scale is not None
    assert weights.oracle_w2_weight is not None
    assert weights.oracle_w2_scale is not None
    assert weights.oracle_flashinfer_weights is not None
    assert weights.oracle_w13_weight.data_ptr() != weights.w13_weight.data_ptr()
    assert weights.oracle_w13_scale.data_ptr() != weights.w13_blockscale_swizzled.data_ptr()
    assert weights.oracle_w2_weight.data_ptr() != weights.w2_weight.data_ptr()
    assert weights.oracle_w2_scale.data_ptr() != weights.w2_blockscale_swizzled.data_ptr()
    assert weights.oracle_flashinfer_weights.w13.shape == weights.oracle_w13_weight.shape
    assert weights.oracle_flashinfer_weights.w13_scale.dtype == torch.float8_e4m3fn
    assert weights.oracle_flashinfer_weights.w13_scale.numel() == weights.oracle_w13_scale.numel()
    assert int(weights.w13_blockscale_swizzled.view(torch.uint8).max().item()) <= 247
    assert int(weights.w2_blockscale_swizzled.view(torch.uint8).max().item()) <= 247

    params = get_quant_mode_params(weights, "shared", "w4a16")
    x, topk_ids, topk_weights = make_profile_routed_inputs(
        profile,
        weights,
        spec,
        1,
        20240525,
        torch.device("cuda"),
    )
    assert topk_ids.shape == (1, spec.top_k)
    assert topk_weights.shape == (1, spec.top_k)
    activation_params = ActivationParams(
        swiglu_limit=profile.default_swiglu_limit,
    )

    expected = make_oracle_reference(
        "w4a16",
        "w4a16",
        x,
        weights,
        params,
        topk_ids,
        topk_weights,
        activation=profile.default_activation,
        activation_params=activation_params,
    )
    try:
        actual = make_oracle_reference(
            "flashinfer",
            "w4a16",
            x,
            weights,
            params,
            topk_ids,
            topk_weights,
            activation=profile.default_activation,
            activation_params=activation_params,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        unavailable_markers = (
            "error occurred when running gemm",
            "no kernel image",
            "no such file",
            "not supported",
            "unsupported",
            "requires flashinfer",
        )
        if any(marker in msg for marker in unavailable_markers):
            pytest.skip(f"FlashInfer TRT-LLM oracle unavailable: {exc}")
        raise

    torch.testing.assert_close(actual.float(), expected.to(torch.bfloat16).float(), rtol=1e-2, atol=1.0)


def test_nvfp4_direct_micro_supports_partial_512_k_groups() -> None:
    for batch_size in (1, 2, 4, 8):
        assert NVFP4MoEMicroKernelBackend.is_supported(
            m=batch_size,
            k=2688,
            n=1856,
            num_topk=6,
            weight_E=128,
        )

    assert not NVFP4MoEMicroKernelBackend.is_supported(
        m=1,
        k=2720,
        n=1856,
        num_topk=6,
        weight_E=128,
    )

    plan = tp_moe._plan_core_workspace(
        "micro",
        "nvfp4",
        state_E=128,
        weight_E=128,
        k=2688,
        n=1856,
        num_topk=6,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        routed_rows=6,
        max_rows=6,
    )
    barrier_spec = next(spec for spec in plan.tensor_specs if spec.name == "barrier_count")
    assert barrier_spec.shape == (22,)
