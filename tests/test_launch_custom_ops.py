from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def test_mhc_decode_split_n_environment_override(monkeypatch) -> None:
    from b12x.integration import residual_kernels

    monkeypatch.setenv("B12X_MHC_DECODE_SPLITS", "4")
    monkeypatch.setenv("B12X_MHC_DECODE_TILE_N", "3")
    assert residual_kernels._selected_post_pre_decode_split_n(
        num_tokens=16,
        hidden_size=4096,
        compute_capability=(12, 1),
    ) == (4, 3)


def test_mhc_sm121_decode_split_n_policy(monkeypatch) -> None:
    from b12x.integration import residual_kernels

    monkeypatch.delenv("B12X_MHC_DECODE_SPLITS", raising=False)
    monkeypatch.delenv("B12X_MHC_DECODE_TILE_N", raising=False)
    select = residual_kernels._selected_post_pre_decode_split_n

    assert select(num_tokens=4, hidden_size=4096, compute_capability=(12, 1)) == (0, 0)
    assert select(num_tokens=8, hidden_size=4096, compute_capability=(12, 1)) == (4, 6)
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 1)) == (8, 6)
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 0)) == (0, 0)
    assert select(num_tokens=16, hidden_size=7168, compute_capability=(12, 1)) == (0, 0)


def test_mhc_decode_finalize_threads_environment_override(monkeypatch) -> None:
    from b12x.integration import residual_kernels

    monkeypatch.setenv("B12X_MHC_DECODE_FINALIZE_THREADS", "128")
    assert (
        residual_kernels._selected_mhc_decode_finalize_threads(
            num_tokens=16,
            hidden_size=4096,
            compute_capability=(12, 1),
        )
        == 128
    )


def test_mhc_sm121_decode_finalize_policy(monkeypatch) -> None:
    from b12x.integration import residual_kernels

    monkeypatch.delenv("B12X_MHC_DECODE_FINALIZE_THREADS", raising=False)
    select = residual_kernels._selected_mhc_decode_finalize_threads

    assert select(num_tokens=4, hidden_size=4096, compute_capability=(12, 1)) == 0
    assert select(num_tokens=8, hidden_size=4096, compute_capability=(12, 1)) == 512
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 1)) == 1024
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 0)) == 0
    assert select(num_tokens=16, hidden_size=7168, compute_capability=(12, 1)) == 0


def test_mhc_sm121_decode_partial_group_policy(monkeypatch) -> None:
    from b12x.integration import residual_kernels

    monkeypatch.delenv("B12X_MHC_PARTIALS_PER_CTA", raising=False)
    select = residual_kernels._selected_post_pre_partials_per_cta

    assert select(num_tokens=2, hidden_size=4096, compute_capability=(12, 1)) == 4
    assert select(num_tokens=4, hidden_size=4096, compute_capability=(12, 1)) == 9
    assert select(num_tokens=8, hidden_size=4096, compute_capability=(12, 1)) == 25
    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 1)) == 25


def test_mhc_decode_partial_group_policy_preserves_sm120(monkeypatch) -> None:
    from b12x.integration import residual_kernels

    monkeypatch.delenv("B12X_MHC_PARTIALS_PER_CTA", raising=False)
    select = residual_kernels._selected_post_pre_partials_per_cta

    assert select(num_tokens=16, hidden_size=4096, compute_capability=(12, 0)) == 4
    assert select(num_tokens=16, hidden_size=7168, compute_capability=(12, 1)) == 4


def test_mhc_decode_partial_group_environment_override(monkeypatch) -> None:
    from b12x.integration import residual_kernels

    monkeypatch.setenv("B12X_MHC_PARTIALS_PER_CTA", "7")
    assert (
        residual_kernels._selected_post_pre_partials_per_cta(
            num_tokens=16,
            hidden_size=4096,
            compute_capability=(12, 1),
        )
        == 7
    )


def test_dense_gemm_launch_has_fake_dispatch() -> None:
    __import__("b12x.gemm.dense")

    with FakeTensorMode():
        a = torch.empty((2, 32, 1), dtype=torch.float8_e4m3fn)
        b = torch.empty((4, 32, 1), dtype=torch.float8_e4m3fn)
        sfa = torch.empty((32, 4, 1, 4, 1, 1), dtype=torch.float8_e8m0fnu)
        sfb = torch.empty((32, 4, 1, 4, 1, 1), dtype=torch.float8_e8m0fnu)
        c = torch.empty((2, 4, 1), dtype=torch.bfloat16)
        alpha = torch.empty((1,), dtype=torch.float32)

        torch.ops.b12x.dense_gemm_launch(
            a,
            b,
            sfa,
            sfb,
            c,
            alpha,
            4,
            32,
            1,
            1,
            "float8_e4m3fn",
            "float8_e8m0fnu",
            "bfloat16",
            "float32",
            32,
            32,
            128,
            64,
            64,
            1,
            1,
            188,
            False,
            False,
            False,
            1,
            False,
            False,
            "tma",
            False,
            False,
            None,
        )
        torch.ops.b12x.dense_gemm_launch(
            a,
            b,
            sfa,
            sfb,
            c,
            alpha,
            4,
            32,
            1,
            1,
            "float8_e4m3fn",
            "float8_e8m0fnu",
            "bfloat16",
            "float32",
            32,
            32,
            128,
            64,
            64,
            1,
            1,
            188,
            False,
            False,
            False,
            1,
            False,
            False,
            "tma",
            False,
            False,
            123,
        )


def test_mhc_launch_ops_have_fake_dispatch() -> None:
    __import__("b12x.integration.residual_kernels")

    with FakeTensorMode():
        residual = torch.empty((2, 4096), dtype=torch.bfloat16)
        x = torch.empty((2, 4096), dtype=torch.bfloat16)
        fn = torch.empty((4, 4096), dtype=torch.float32)
        partials = torch.empty((2, 64, 25), dtype=torch.float32)
        prev_post = torch.empty((2, 24), dtype=torch.float32)
        prev_comb = torch.empty((2, 24), dtype=torch.float32)
        out = torch.empty((2, 4096), dtype=torch.bfloat16)
        scale = torch.empty((24,), dtype=torch.float32)
        bias = torch.empty((24,), dtype=torch.float32)
        y = torch.empty((2, 4096), dtype=torch.bfloat16)
        post = torch.empty((2, 24), dtype=torch.float32)
        comb = torch.empty((2, 24), dtype=torch.float32)
        norm_weight = torch.empty((4096,), dtype=torch.float32)

        torch.ops.b12x.mhc_pre_partial_launch(residual, fn, partials, True)
        torch.ops.b12x.mhc_post_pre_partial_launch(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            partials,
            out,
            True,
        )
        torch.ops.b12x.mhc_finalize_gram_launch(
            residual,
            partials,
            scale,
            bias,
            y,
            post,
            comb,
            norm_weight,
            1e-6,
            1e-6,
            4,
            1e-6,
            True,
            False,
            1,
            0,
        )


def test_tp_moe_launch_ops_have_fake_dispatch() -> None:
    __import__("b12x.integration.tp_moe")

    with FakeTensorMode():
        a = torch.empty((2, 128), dtype=torch.bfloat16)
        flat_ids = torch.empty((4,), dtype=torch.int32)
        flat_weights = torch.empty((4,), dtype=torch.float32)
        scatter = torch.empty((2, 64), dtype=torch.bfloat16)
        packed_a_view = torch.empty((64, 8, 2), dtype=torch.uint8)
        packed_a_flat = torch.empty((1024,), dtype=torch.uint8)
        scale_flat = torch.empty((256,), dtype=torch.uint8)
        scalar_i32 = torch.empty((1,), dtype=torch.int32)
        row_counts = torch.empty((4,), dtype=torch.int32)
        token_map = torch.empty((4,), dtype=torch.int32)
        token_weights = torch.empty((4,), dtype=torch.float32)
        compact_topk_ids = torch.empty((4,), dtype=torch.int32)
        micro_intermediate = torch.empty((2, 64), dtype=torch.float32)
        w13 = torch.empty((128, 64, 4), dtype=torch.uint8)
        down = torch.empty((128, 32, 4), dtype=torch.uint8)
        w13_sf = torch.empty((4, 128, 4), dtype=torch.uint8)
        down_sf = torch.empty((4, 128, 4), dtype=torch.uint8)
        w1_storage = torch.empty((4, 128, 64), dtype=torch.uint8)
        w2_storage = torch.empty((4, 128, 32), dtype=torch.uint8)
        w1_scale = torch.empty((4, 8, 4), dtype=torch.uint8)
        w2_scale = torch.empty((4, 8, 4), dtype=torch.uint8)
        alpha = torch.empty((4,), dtype=torch.float32)
        task = torch.empty((16,), dtype=torch.int32)

        torch.ops.b12x.tp_moe_dynamic_launch(
            packed_a_view,
            packed_a_flat,
            scale_flat,
            scalar_i32,
            scalar_i32,
            scalar_i32,
            scalar_i32,
            scalar_i32,
            scalar_i32,
            scalar_i32,
            task,
            task,
            task,
            task,
            task,
            task,
            task,
            task,
            row_counts,
            row_counts,
            row_counts,
            alpha,
            alpha,
            token_map,
            token_weights,
            w13,
            w13_sf,
            down,
            down_sf,
            alpha,
            alpha,
            w1_storage,
            w2_storage,
            w13,
            w13_sf,
            down,
            down_sf,
            alpha,
            alpha,
            a,
            flat_ids,
            flat_weights,
            scatter,
            4,
            2,
            128,
            64,
            2,
            4,
            4,
            2,
            2,
            16,
            True,
            True,
            "silu",
            "nvfp4",
            False,
            False,
            False,
            None,
            1.0,
            0.0,
            False,
        )

        torch.ops.b12x.tp_moe_compact_micro_launch(
            scalar_i32,
            scalar_i32,
            micro_intermediate,
            w1_storage,
            w1_scale,
            w2_storage,
            w2_scale,
            alpha,
            alpha,
            a,
            compact_topk_ids,
            flat_weights,
            alpha,
            alpha,
            scatter,
            4,
            2,
            128,
            64,
            2,
            True,
            False,
            False,
            "silu",
            "nvfp4",
            None,
            1.0,
            0.0,
            False,
        )


def test_w4a16_moe_launch_ops_have_fake_dispatch() -> None:
    __import__("b12x.moe.fused.w4a16.kernel")

    with FakeTensorMode():
        a = torch.empty((2, 128), dtype=torch.bfloat16)
        w13_u8 = torch.empty((4096,), dtype=torch.uint8)
        w2_u8 = torch.empty((4096,), dtype=torch.uint8)
        scale_u8 = torch.empty((512,), dtype=torch.uint8)
        global_scale = torch.empty((4,), dtype=torch.float32)
        inter_u32 = torch.empty((1024,), dtype=torch.uint32)
        topk_ids = torch.empty((2, 2), dtype=torch.int32)
        topk_weights = torch.empty((2, 2), dtype=torch.float32)
        output = torch.empty((2, 128), dtype=torch.bfloat16)
        barrier = torch.empty((1,), dtype=torch.int32)

        torch.ops.b12x.w4a16_small_m_direct_launch(
            a,
            w13_u8,
            scale_u8,
            global_scale,
            global_scale,
            inter_u32,
            w2_u8,
            scale_u8,
            topk_ids,
            topk_weights,
            output,
            barrier,
            barrier,
            2,
            128,
            64,
            4,
            2,
            "silu",
            True,
            "e8m0_k32",
            False,
            0.0,
            1.0,
            0.0,
            "w13",
            0,
        )

        fc1_out = torch.empty((256,), dtype=torch.bfloat16)
        activated = torch.empty((256,), dtype=torch.bfloat16)
        fc2_out = torch.empty((256,), dtype=torch.bfloat16)
        scale_i32 = torch.empty((128,), dtype=torch.int32)
        packed_routes = torch.empty((16,), dtype=torch.int32)
        block_experts = torch.empty((4,), dtype=torch.int32)
        route_count = torch.empty((1,), dtype=torch.int32)
        scratch = torch.empty((1024,), dtype=torch.float32)
        workspace = torch.empty((512,), dtype=torch.int32)
        activation_amax = torch.empty((2, 4, 2), dtype=torch.float32)

        torch.ops.b12x.w4a16_fused_moe_launch(
            a,
            w13_u8,
            w2_u8,
            fc1_out,
            activated,
            fc2_out,
            scale_i32,
            scale_i32,
            global_scale,
            global_scale,
            packed_routes,
            block_experts,
            route_count,
            topk_weights,
            scratch,
            scratch,
            workspace,
            2,
            2,
            128,
            64,
            4,
            2,
            "silu",
            False,
            False,
            16,
            4,
            "bf16",
            True,
            120,
            101376,
            False,
            0.0,
            1.0,
            0.0,
            "modelopt",
            "e8m0_k32",
            "w13",
            False,
            False,
            0,
        )

        torch.ops.b12x.w4a16_fused_moe_calibrated_launch(
            a,
            w13_u8,
            w2_u8,
            fc1_out,
            activated,
            fc2_out,
            scale_i32,
            scale_i32,
            global_scale,
            global_scale,
            packed_routes,
            block_experts,
            route_count,
            activation_amax,
            1,
            topk_weights,
            scratch,
            scratch,
            workspace,
            2,
            2,
            128,
            64,
            4,
            2,
            "silu",
            False,
            False,
            16,
            4,
            "bf16",
            True,
            120,
            101376,
            False,
            0.0,
            1.0,
            0.0,
            "modelopt",
            "e8m0_k32",
            "w13",
            0,
        )

        torch.ops.b12x.w4a16_topk_sum_launch(
            fc2_out,
            output,
            2,
            2,
            128,
            "bf16",
            0,
        )
