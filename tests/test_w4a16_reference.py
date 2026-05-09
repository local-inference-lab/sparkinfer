from __future__ import annotations

import torch

import b12x.integration.tp_moe as tp_moe
from b12x.cute.fp4 import pack_grouped_fp4_values, swizzle_block_scale
from b12x.moe.fused.w4a16.micro import MoEMicroKernelBackend
from b12x.moe.fused.w4a16.reference import moe_reference_w4a16


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


def test_w4a16_workspace_plan_uses_bf16_activation_scratch() -> None:
    static_plan = tp_moe._plan_core_workspace(
        "static",
        "w4a16",
        state_E=2,
        weight_E=4,
        k=16,
        n=16,
        num_topk=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        routed_rows=2,
        max_rows=2,
    )
    static_specs = {spec.name: spec for spec in static_plan.tensor_specs}
    assert static_plan.quant_mode == "w4a16"
    assert static_specs["packed_input"].shape == (2, 128, 16)
    assert static_specs["packed_input"].dtype == torch.bfloat16

    dynamic_plan = tp_moe._plan_core_workspace(
        "dynamic",
        "w4a16",
        state_E=4,
        weight_E=4,
        k=16,
        n=16,
        num_topk=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        routed_rows=129,
        max_rows=256,
        dynamic_physical_tiles=2,
        dynamic_task_capacity=3,
    )
    dynamic_specs = {spec.name: spec for spec in dynamic_plan.tensor_specs}
    assert dynamic_plan.quant_mode == "w4a16"
    assert dynamic_specs["packed_input"].shape == (
        1,
        2 * tp_moe._dynamic_tile_m("w4a16"),
        16,
    )
    assert dynamic_specs["packed_input"].dtype == torch.bfloat16


def test_w4a16_dynamic_geometry_uses_bf16_tile_contract() -> None:
    tile_m = tp_moe._dynamic_tile_m("w4a16")
    tile_n = tp_moe._dynamic_tile_n("w4a16")
    max_tiles, gate_tile_cnt, max_tasks = tp_moe._dynamic_task_geometry(
        8,
        256,
        641,
        tile_m=tile_m,
        tile_n=tile_n,
    )

    assert tile_m == 32
    assert tile_n == 64
    assert gate_tile_cnt == 4
    assert max_tasks == max_tiles * 4
    assert tp_moe._dynamic_rows_padded_limit(
        4096,
        quant_mode="w4a16",
    ) < tp_moe._dynamic_rows_padded_limit(4096)


def test_w4a16_backend_cutover_uses_dynamic_after_small_static_window() -> None:
    tp_moe.clear_tp_moe_caches()
    assert tp_moe._get_static_compact_cutover_pairs("w4a16") == 128
    assert tp_moe._get_static_compact_cutover_pairs("nvfp4") == 640
    assert (
        tp_moe.select_tp_moe_backend(num_tokens=16, num_topk=8, quant_mode="w4a16")
        == "static"
    )
    assert (
        tp_moe.select_tp_moe_backend(num_tokens=17, num_topk=8, quant_mode="w4a16")
        == "dynamic"
    )


def test_w4a16_direct_micro_supports_static_decode_batches() -> None:
    for k_segments in range(1, 13):
        assert MoEMicroKernelBackend.is_supported(
            m=1,
            k=k_segments * 32 * 16,
            n=256,
            num_topk=8,
            weight_E=256,
        )

    for batch_size in (1, 2, 4, 8):
        assert MoEMicroKernelBackend.is_supported(
            m=batch_size,
            k=3072,
            n=768,
            num_topk=8,
            weight_E=256,
        )

    for batch_size in (10, 12, 16, 24, 32):
        assert MoEMicroKernelBackend.is_supported(
            m=batch_size,
            k=4096,
            n=256,
            num_topk=10,
            weight_E=512,
        )

    for batch_size in (1, 2, 4, 8):
        assert MoEMicroKernelBackend.is_supported(
            m=batch_size,
            k=6144,
            n=256,
            num_topk=8,
            weight_E=256,
        )

    assert not MoEMicroKernelBackend.is_supported(
        m=40,
        k=4096,
        n=256,
        num_topk=10,
        weight_E=512,
    )
