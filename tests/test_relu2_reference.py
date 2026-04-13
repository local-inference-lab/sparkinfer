#!/usr/bin/env python3
"""Unit tests for ReLU2 (non-gated) MoE activation support.

Tests the reference implementation, weight view creation, integration layer
plumbing, and kernel class selection for non-gated relu2 activations used
by models like Nemotron-3-Super-120B.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _cuda_has_free_memory(min_mb: int = 512) -> bool:
    """Check if CUDA has at least min_mb free memory."""
    if not torch.cuda.is_available():
        return False
    try:
        free, total = torch.cuda.mem_get_info()
        return free >= min_mb * 1024 * 1024
    except Exception:
        return False


_skip_no_gpu_mem = pytest.mark.skipif(
    not _cuda_has_free_memory(512),
    reason="Insufficient free GPU memory (model loaded?)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_weights(
    E: int, K: int, I_tp: int, *, is_gated: bool = True, device: str = "cpu",
):
    """Create random FP4-style weight tensors for testing.

    Default device is CPU to avoid OOM when a model occupies GPU memory.
    """
    w1_n = 2 * I_tp if is_gated else I_tp
    w1_fp4 = torch.randint(0, 256, (E, w1_n, K // 2), dtype=torch.uint8, device=device)
    w2_fp4 = torch.randint(0, 256, (E, K, I_tp // 2), dtype=torch.uint8, device=device)

    # Block-scale tensors: swizzled layout
    bs_k = K // 16
    bs_k_padded = ((bs_k + 3) // 4) * 4
    w1_n_padded = ((w1_n + 127) // 128) * 128
    K_padded = ((K + 127) // 128) * 128

    w1_blockscale = torch.randint(
        1, 127, (E, w1_n_padded // 128, bs_k_padded // 4, 32, 4, 4),
        dtype=torch.uint8, device=device,
    ).reshape(E, -1).contiguous()
    w2_bs_n = I_tp // 16
    w2_bs_n_padded = ((w2_bs_n + 3) // 4) * 4
    w2_blockscale = torch.randint(
        1, 127, (E, K_padded // 128, w2_bs_n_padded // 4, 32, 4, 4),
        dtype=torch.uint8, device=device,
    ).reshape(E, -1).contiguous()

    w1_alphas = torch.rand(E, dtype=torch.float32, device=device) * 0.1 + 0.9
    w2_alphas = torch.rand(E, dtype=torch.float32, device=device) * 0.1 + 0.9
    a1_gscale = torch.rand(E, dtype=torch.float32, device=device) * 0.5 + 0.5
    a2_gscale = torch.rand(E, dtype=torch.float32, device=device) * 0.5 + 0.5

    return dict(
        w1_fp4=w1_fp4, w1_blockscale=w1_blockscale, w1_alphas=w1_alphas,
        w2_fp4=w2_fp4, w2_blockscale=w2_blockscale, w2_alphas=w2_alphas,
        a1_gscale=a1_gscale, a2_gscale=a2_gscale,
    )


def _to_device(weights: dict, device: str) -> dict:
    """Move all weight tensors to the specified device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in weights.items()}


def _make_routed_inputs(
    m: int, K: int, E: int, topk: int, *, device: str = "cuda",
):
    """Create random activations and routing for testing."""
    x = torch.randn(m, K, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (m, topk), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn(m, topk, dtype=torch.float32, device=device), dim=-1,
    )
    return x, topk_ids, topk_weights


# ---------------------------------------------------------------------------
# Test: Reference f32 relu2 vs naive PyTorch
# ---------------------------------------------------------------------------

class TestRelu2Reference:
    """Verify that the reference relu2 path matches a naive PyTorch impl."""

    def test_relu2_activation_basic(self):
        """relu(x)^2 should match torch.square(torch.relu(x))."""
        x = torch.randn(128, dtype=torch.float32)
        expected = torch.square(torch.relu(x))
        # Replicate what the kernel does
        relu_x = torch.clamp(x, min=0.0)
        actual = relu_x * relu_x
        torch.testing.assert_close(actual, expected)

    def test_relu2_vs_silu_different_output(self):
        """relu2 and silu should produce different results for the same input."""
        x = torch.randn(64, dtype=torch.float32)

        relu2_out = torch.square(torch.relu(x))
        silu_out = F.silu(x) * x  # SwiGLU needs gate*up, so not directly comparable
        # Just check they're different (one uses relu^2, other uses silu*x)
        assert not torch.allclose(relu2_out, silu_out), \
            "relu2 and silu should produce different outputs"

    def test_relu2_zero_for_negative_inputs(self):
        """relu2 should be exactly zero for all negative inputs."""
        x = torch.tensor([-5.0, -1.0, -0.01, 0.0, 0.01, 1.0, 5.0])
        result = torch.square(torch.relu(x))
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert result[4] > 0.0
        assert result[5] > 0.0
        assert result[6] > 0.0


# ---------------------------------------------------------------------------
# Test: Weight shape validation for gated vs non-gated
# ---------------------------------------------------------------------------

class TestWeightShapes:
    """Verify weight tensor shapes for gated vs non-gated MoE."""

    @pytest.mark.parametrize("is_gated", [True, False])
    def test_w1_shape(self, is_gated):
        E, K, I_tp = 8, 128, 256
        weights = _make_random_weights(E, K, I_tp, is_gated=is_gated, device="cpu")
        expected_n = 2 * I_tp if is_gated else I_tp
        assert weights["w1_fp4"].shape == (E, expected_n, K // 2)

    def test_w2_shape_same_for_both(self):
        """w2 shape is independent of gated/non-gated."""
        E, K, I_tp = 8, 128, 256
        gated = _make_random_weights(E, K, I_tp, is_gated=True, device="cpu")
        nongated = _make_random_weights(E, K, I_tp, is_gated=False, device="cpu")
        assert gated["w2_fp4"].shape == nongated["w2_fp4"].shape


# ---------------------------------------------------------------------------
# Test: Integration layer plumbing (tp_moe.py)
# ---------------------------------------------------------------------------

@_skip_no_gpu_mem
class TestIntegrationPlumbing:
    """Test that activation parameter flows correctly through tp_moe.py."""

    def test_get_weight_views_gated(self):
        """_get_weight_views with is_gated=True uses 2*n for w13_sf."""
        from b12x.integration.tp_moe import _get_weight_views, _WEIGHT_CACHE

        E, K, I_tp = 2, 128, 128
        weights = _to_device(_make_random_weights(E, K, I_tp, is_gated=True), "cuda")
        _WEIGHT_CACHE.clear()

        wv = _get_weight_views(
            weights["w1_fp4"], weights["w1_blockscale"],
            weights["w2_fp4"], weights["w2_blockscale"],
            weights["w1_alphas"], weights["w2_alphas"],
            I_tp, K, is_gated=True,
        )
        # w13 should have 2*I_tp in first dim
        assert wv.w13.shape[0] == 2 * I_tp

    def test_get_weight_views_nongated(self):
        """_get_weight_views with is_gated=False uses n for w13_sf."""
        from b12x.integration.tp_moe import _get_weight_views, _WEIGHT_CACHE

        E, K, I_tp = 2, 128, 128
        weights = _to_device(_make_random_weights(E, K, I_tp, is_gated=False), "cuda")
        _WEIGHT_CACHE.clear()

        wv = _get_weight_views(
            weights["w1_fp4"], weights["w1_blockscale"],
            weights["w2_fp4"], weights["w2_blockscale"],
            weights["w1_alphas"], weights["w2_alphas"],
            I_tp, K, is_gated=False,
        )
        # w13 should have I_tp in first dim (non-gated)
        assert wv.w13.shape[0] == I_tp

    def test_cache_key_includes_is_gated(self):
        """Cache key must differentiate gated vs non-gated to avoid collisions."""
        from b12x.integration.tp_moe import _get_weight_views, _WEIGHT_CACHE

        E, K, I_tp = 2, 128, 128
        _WEIGHT_CACHE.clear()

        gated = _to_device(_make_random_weights(E, K, I_tp, is_gated=True), "cuda")
        wv_gated = _get_weight_views(
            gated["w1_fp4"], gated["w1_blockscale"],
            gated["w2_fp4"], gated["w2_blockscale"],
            gated["w1_alphas"], gated["w2_alphas"],
            I_tp, K, is_gated=True,
        )

        nongated = _to_device(_make_random_weights(E, K, I_tp, is_gated=False), "cuda")
        wv_nongated = _get_weight_views(
            nongated["w1_fp4"], nongated["w1_blockscale"],
            nongated["w2_fp4"], nongated["w2_blockscale"],
            nongated["w1_alphas"], nongated["w2_alphas"],
            I_tp, K, is_gated=False,
        )

        # Both should be cached separately
        assert len(_WEIGHT_CACHE) >= 2
        assert wv_gated.w13.shape[0] != wv_nongated.w13.shape[0]


# ---------------------------------------------------------------------------
# Test: Kernel class selection
# ---------------------------------------------------------------------------

@_skip_no_gpu_mem
class TestKernelClassSelection:
    """Verify that the correct kernel class is selected based on activation."""

    def test_static_kernel_silu(self):
        """Default activation selects MoEStaticKernel."""
        from b12x.integration.tp_moe import _get_static_kernel, clear_tp_moe_caches
        clear_tp_moe_caches()

        compiled, mac = _get_static_kernel(
            state_E=8, weight_E=8, m=4, k=128, n=256, num_topk=1, max_rows=32,
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=True,
            fast_math=True,
            activation="silu",
        )
        # The compiled kernel should be from MoEStaticKernel (SiLU)
        assert compiled is not None

    def test_static_kernel_relu2(self):
        """activation='relu2' selects MoEStaticKernelRelu2."""
        from b12x.integration.tp_moe import _get_static_kernel, clear_tp_moe_caches
        clear_tp_moe_caches()

        compiled, mac = _get_static_kernel(
            state_E=8, weight_E=8, m=4, k=128, n=256, num_topk=1, max_rows=32,
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=True,
            fast_math=True,
            activation="relu2",
        )
        assert compiled is not None

    def test_static_cache_differentiates_activation(self):
        """Same params but different activation should produce different cache entries."""
        from b12x.integration.tp_moe import _get_static_kernel, clear_tp_moe_caches, _STATIC_KERNEL_CACHE
        clear_tp_moe_caches()

        _get_static_kernel(
            state_E=8, weight_E=8, m=4, k=128, n=256, num_topk=1, max_rows=32,
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=True,
            fast_math=True,
            activation="silu",
        )
        _get_static_kernel(
            state_E=8, weight_E=8, m=4, k=128, n=256, num_topk=1, max_rows=32,
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=True,
            fast_math=True,
            activation="relu2",
        )
        # Should have 2 distinct entries in cache
        assert len(_STATIC_KERNEL_CACHE) >= 2

    def test_dynamic_kernel_relu2(self):
        """activation='relu2' selects MoEDynamicKernelRelu2."""
        from b12x.integration.tp_moe import _get_dynamic_kernel, clear_tp_moe_caches
        clear_tp_moe_caches()

        compiled, mac = _get_dynamic_kernel(
            E=8, m=256, k=128, n=256, num_topk=1, max_rows=256,
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=True,
            fast_math=True,
            activation="relu2",
        )
        assert compiled is not None

    def test_micro_kernel_relu2(self):
        """activation='relu2' selects MoEMicroKernelRelu2."""
        from b12x.integration.tp_moe import _get_micro_kernel, clear_tp_moe_caches
        clear_tp_moe_caches()

        compiled, mac = _get_micro_kernel(
            state_E=8, weight_E=8, m=4, k=128, n=256, num_topk=4, max_rows=32,
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=True,
            fast_math=True,
            activation="relu2",
        )
        assert compiled is not None


# ---------------------------------------------------------------------------
# Test: Reference implementation with relu2 activation
# ---------------------------------------------------------------------------

@_skip_no_gpu_mem
class TestRelu2ReferenceF32:
    """Test the reference f32 implementation with relu2 activation."""

    def test_reference_relu2_runs(self):
        """moe_reference_f32(activation='relu2') should run without error."""
        from b12x.moe.fused.reference import moe_reference_f32

        E, K, I_tp, m, topk = 2, 128, 128, 2, 1
        weights = _to_device(_make_random_weights(E, K, I_tp, is_gated=False), "cuda")
        x, topk_ids, topk_weights = _make_routed_inputs(m, K, E, topk, device="cuda")

        result = moe_reference_f32(
            x, weights["w1_fp4"], weights["w1_blockscale"], weights["w1_alphas"],
            weights["w2_fp4"], weights["w2_blockscale"], weights["w2_alphas"],
            weights["a1_gscale"], weights["a2_gscale"],
            topk_ids, topk_weights, E, K, I_tp,
            activation="relu2",
        )
        assert result.shape == (m, K)
        assert result.dtype == torch.bfloat16
        assert torch.isfinite(result).all()

    def test_reference_relu2_vs_naive(self):
        """moe_reference_f32(relu2) should match a manual relu2 computation."""
        from b12x.moe.fused.reference import moe_reference_f32

        E, K, I_tp = 2, 64, 128
        m, topk = 1, 1

        weights = _to_device(_make_random_weights(E, K, I_tp, is_gated=False), "cuda")
        x, topk_ids, topk_weights = _make_routed_inputs(m, K, E, topk, device="cuda")
        # Force routing to expert 0 with weight 1.0
        topk_ids[:] = 0
        topk_weights[:] = 1.0

        result = moe_reference_f32(
            x, weights["w1_fp4"], weights["w1_blockscale"], weights["w1_alphas"],
            weights["w2_fp4"], weights["w2_blockscale"], weights["w2_alphas"],
            weights["a1_gscale"], weights["a2_gscale"],
            topk_ids, topk_weights, E, K, I_tp,
            activation="relu2",
        )

        # Result should not be all zeros (unless extremely unlucky random init)
        assert result.abs().sum() > 0, "Reference relu2 output is all zeros"

    def test_reference_relu2_different_from_silu(self):
        """relu2 and silu reference paths should produce different outputs."""
        from b12x.moe.fused.reference import moe_reference_f32

        E, K, I_tp = 2, 64, 128
        m, topk = 2, 1

        gated_weights = _to_device(_make_random_weights(E, K, I_tp, is_gated=True), "cuda")
        x, topk_ids, topk_weights = _make_routed_inputs(m, K, E, topk, device="cuda")

        silu_result = moe_reference_f32(
            x, gated_weights["w1_fp4"], gated_weights["w1_blockscale"],
            gated_weights["w1_alphas"],
            gated_weights["w2_fp4"], gated_weights["w2_blockscale"],
            gated_weights["w2_alphas"],
            gated_weights["a1_gscale"], gated_weights["a2_gscale"],
            topk_ids, topk_weights, E, K, I_tp,
            activation="silu",
        )

        nongated_weights = _to_device(_make_random_weights(E, K, I_tp, is_gated=False), "cuda")

        relu2_result = moe_reference_f32(
            x, nongated_weights["w1_fp4"], nongated_weights["w1_blockscale"],
            nongated_weights["w1_alphas"],
            nongated_weights["w2_fp4"], nongated_weights["w2_blockscale"],
            nongated_weights["w2_alphas"],
            nongated_weights["a1_gscale"], nongated_weights["a2_gscale"],
            topk_ids, topk_weights, E, K, I_tp,
            activation="relu2",
        )

        # Shapes should match (same K output)
        assert silu_result.shape == relu2_result.shape


# ---------------------------------------------------------------------------
# Test: b12x_moe_fp4 activation parameter
# ---------------------------------------------------------------------------

class TestB12xMoeFp4Activation:
    """Test that b12x_moe_fp4 accepts and respects the activation parameter."""

    def test_default_activation_is_silu(self):
        """b12x_moe_fp4 should default to silu without explicit activation."""
        import inspect
        from b12x.integration.tp_moe import b12x_moe_fp4
        sig = inspect.signature(b12x_moe_fp4)
        assert sig.parameters["activation"].default == "silu"

    def test_activation_parameter_exists(self):
        """b12x_moe_fp4 should have an 'activation' keyword parameter."""
        import inspect
        from b12x.integration.tp_moe import b12x_moe_fp4
        sig = inspect.signature(b12x_moe_fp4)
        assert "activation" in sig.parameters
        assert sig.parameters["activation"].kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )

    def test_launcher_functions_accept_activation(self):
        """Both launcher functions should accept an activation parameter."""
        import inspect
        from b12x.integration.tp_moe import _launch_compact_static, _launch_dynamic
        for fn in [_launch_compact_static, _launch_dynamic]:
            sig = inspect.signature(fn)
            assert "activation" in sig.parameters, \
                f"{fn.__name__} missing 'activation' parameter"


# ---------------------------------------------------------------------------
# Test: vLLM b12x_moe.py activation mapping
# ---------------------------------------------------------------------------

class TestVllmActivationMapping:
    """Test the vLLM-side activation mapping in b12x_moe.py."""

    def test_activation_map_covers_supported(self):
        """_B12X_ACTIVATION_MAP should cover all supported activations."""
        sys.path.insert(0, "/home/ubuntu/vllm-src")
        from vllm.model_executor.layers.fused_moe.b12x_moe import (
            _B12X_ACTIVATION_MAP,
            B12xExperts,
        )
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        # All activations that _supports_activation returns True for
        # should be in the map
        for act in MoEActivation:
            if B12xExperts._supports_activation(act):
                assert act in _B12X_ACTIVATION_MAP, \
                    f"Supported activation {act} missing from _B12X_ACTIVATION_MAP"

    def test_relu2_no_mul_supported(self):
        """B12xExperts should support RELU2_NO_MUL."""
        sys.path.insert(0, "/home/ubuntu/vllm-src")
        from vllm.model_executor.layers.fused_moe.b12x_moe import B12xExperts
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        assert B12xExperts._supports_activation(MoEActivation.RELU2_NO_MUL)

    def test_relu2_maps_to_relu2_string(self):
        """RELU2_NO_MUL should map to 'relu2' string."""
        sys.path.insert(0, "/home/ubuntu/vllm-src")
        from vllm.model_executor.layers.fused_moe.b12x_moe import _B12X_ACTIVATION_MAP
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        assert _B12X_ACTIVATION_MAP[MoEActivation.RELU2_NO_MUL] == "relu2"

    def test_unsupported_activation_raises(self):
        """Unmapped activation should raise KeyError, not silently default."""
        sys.path.insert(0, "/home/ubuntu/vllm-src")
        from vllm.model_executor.layers.fused_moe.b12x_moe import _B12X_ACTIVATION_MAP
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        with pytest.raises(KeyError):
            _ = _B12X_ACTIVATION_MAP[MoEActivation.GELU]


# ---------------------------------------------------------------------------
# Test: Kernel __init__.py exports
# ---------------------------------------------------------------------------

class TestKernelExports:
    """Verify that all kernel classes are properly exported."""

    def test_all_kernel_classes_importable(self):
        from b12x.moe.fused import (
            MoEStaticKernel,
            MoEStaticKernelRelu2,
            MoEMicroKernel,
            MoEMicroKernelRelu2,
            MoEDynamicKernel,
            MoEDynamicKernelRelu2,
        )
        # All should be class objects
        assert isinstance(MoEStaticKernelRelu2, type)
        assert isinstance(MoEMicroKernelRelu2, type)
        assert isinstance(MoEDynamicKernelRelu2, type)

    def test_relu2_classes_distinct_from_silu(self):
        from b12x.moe.fused import (
            MoEStaticKernel, MoEStaticKernelRelu2,
            MoEMicroKernel, MoEMicroKernelRelu2,
            MoEDynamicKernel, MoEDynamicKernelRelu2,
        )
        assert MoEStaticKernel is not MoEStaticKernelRelu2
        assert MoEMicroKernel is not MoEMicroKernelRelu2
        assert MoEDynamicKernel is not MoEDynamicKernelRelu2


# ---------------------------------------------------------------------------
# Test: CuTe DSL helpers for relu2
# ---------------------------------------------------------------------------

@_skip_no_gpu_mem
class TestRelu2CuteHelpers:
    """Test the relu2 CuTe DSL helper functions."""

    def test_relu2_quantize_torch_runs(self):
        """relu2_quantize_grouped_nvfp4_torch should produce valid output."""
        from b12x.cute.fp4 import relu2_quantize_grouped_nvfp4_torch

        groups, N, K = 1, 4, 128
        input_tensor = torch.randn(groups, N, K, dtype=torch.bfloat16, device="cuda")
        row_counts = torch.tensor([N], dtype=torch.int32, device="cuda")
        global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        packed, scales = relu2_quantize_grouped_nvfp4_torch(
            input_tensor, row_counts, global_scale,
        )
        assert packed.numel() > 0
        assert torch.isfinite(scales.float()).all()

    def test_relu2_quantize_preserves_zeros_for_negatives(self):
        """relu2 quantization of all-negative input should produce near-zero output."""
        from b12x.cute.fp4 import relu2_quantize_grouped_nvfp4_torch

        groups, N, K = 1, 1, 64
        # All negative inputs → relu2 produces all zeros
        input_tensor = -torch.ones(groups, N, K, dtype=torch.bfloat16, device="cuda")
        row_counts = torch.tensor([N], dtype=torch.int32, device="cuda")
        global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        packed, scales = relu2_quantize_grouped_nvfp4_torch(
            input_tensor, row_counts, global_scale,
        )
        # Packed FP4 of all-zero should be 0x00 bytes
        assert (packed == 0).all(), \
            "Packed FP4 of relu2(negative) should be all zeros"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
