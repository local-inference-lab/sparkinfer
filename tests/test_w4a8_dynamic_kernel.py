"""Direct-drive integration test for the w4a8 dynamic MoE kernel recipe.

Drives MoEDynamicKernelBackend(quant_recipe="w4a8_mx"/"w4a8_nvfp4") through
the _DynamicMoEW4A8Launch adapter with synthetic experts and gates the output
against the pure-Torch oracle (moe_reference_w4a8_mx).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import make_ptr

from b12x.cute.fp4 import _fp4_encode_nibbles, fp4_quantize_values_torch
from b12x.cute.compiler import compile as b12x_compile
from cutlass.base_dsl.compiler import OptLevel as _DSLOptLevel

_OPT_LEVEL_2 = _DSLOptLevel(2)
from b12x.integration.tp_moe import _DynamicMoEW4A8Launch, current_cuda_stream
from b12x.moe.fused.dynamic import MoEDynamicKernelBackend
from b12x.moe.fused.reference import (
    compare_to_reference,
    decompose_nvfp4_scales_to_mx_residual,
    moe_reference_w4a8_mx,
)

from .helpers import require_sm120

_TILE_M = 128
_TILE_N = 128


def _pack_fp4_rows(values: torch.Tensor) -> torch.Tensor:
    nib = _fp4_encode_nibbles(values)
    pair = nib.view(*values.shape[:-1], values.shape[-1] // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).contiguous()


def _quantize_weight_mxfp4(w: torch.Tensor):
    rows, cols = w.shape
    blocked = w.view(rows, cols // 32, 32)
    bmax = blocked.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(bmax > 0, bmax / 6.0, torch.ones_like(bmax))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    byte = torch.where(bmax > 0, exponent + 127, torch.zeros_like(exponent)).to(torch.uint8)
    scale = torch.where(bmax > 0, torch.exp2(exponent), torch.zeros_like(exponent))
    q = fp4_quantize_values_torch(
        torch.where(scale > 0, blocked / scale.clamp(min=1e-30), torch.zeros_like(blocked)).view(rows, cols)
    )
    return _pack_fp4_rows(q), byte.squeeze(-1)


def _quantize_weight_nvfp4(w: torch.Tensor):
    rows, cols = w.shape
    blocked = w.view(rows, cols // 16, 16)
    bmax = blocked.abs().amax(dim=-1, keepdim=True)
    scale = (bmax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).to(torch.float32)
    q = fp4_quantize_values_torch((blocked / scale.clamp(min=1e-30)).view(rows, cols))
    return _pack_fp4_rows(q), scale.squeeze(-1)


def _gptr(dtype, t: torch.Tensor, align: int = 16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _fake_i32(shape):
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, shape, assumed_align=4)


def _fake_f32(shape):
    return cute.runtime.make_fake_compact_tensor(cutlass.Float32, shape, assumed_align=16)


def _run_w4a8_dynamic(
    *,
    recipe: str,
    activation: str,
    E: int,
    m: int,
    K: int,
    n: int,
    top_k: int,
    seed: int,
    tile_m: int = _TILE_M,
    return_launcher: bool = False,
):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    is_gated = activation == "silu"
    w1_n = 2 * n if is_gated else n

    x = (torch.randn(m, K, device=device) * 2.0).to(torch.bfloat16)
    w13_full = torch.randn(E, w1_n, K, device=device) * 0.05
    w2_full = torch.randn(E, K, n, device=device) * 0.05
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, top_k, device=device), dim=-1).float()

    if recipe == "w4a8_mx":
        w13_q = [_quantize_weight_mxfp4(w13_full[e]) for e in range(E)]
        w2_q = [_quantize_weight_mxfp4(w2_full[e]) for e in range(E)]
        w13_mx = torch.stack([q[1] for q in w13_q]).contiguous()
        w2_mx = torch.stack([q[1] for q in w2_q]).contiguous()
        w13_res = torch.zeros(E, w1_n, K // 16, dtype=torch.uint8, device=device)
        w2_res = torch.zeros(E, K, n // 16, dtype=torch.uint8, device=device)
        ref_res_w13 = None
        ref_res_w2 = None
    else:
        w13_q = [_quantize_weight_nvfp4(w13_full[e]) for e in range(E)]
        w2_q = [_quantize_weight_nvfp4(w2_full[e]) for e in range(E)]
        w13_scales = torch.stack([q[1] for q in w13_q])
        w2_scales = torch.stack([q[1] for q in w2_q])
        w13_mx, w13_res_e4m3 = decompose_nvfp4_scales_to_mx_residual(w13_scales)
        w2_mx, w2_res_e4m3 = decompose_nvfp4_scales_to_mx_residual(w2_scales)
        w13_mx = w13_mx.contiguous()
        w2_mx = w2_mx.contiguous()
        w13_res = w13_res_e4m3.view(torch.uint8).contiguous()
        w2_res = w2_res_e4m3.view(torch.uint8).contiguous()
        ref_res_w13 = w13_res_e4m3
        ref_res_w2 = w2_res_e4m3

    w13_packed = torch.stack([q[0] for q in w13_q]).contiguous()
    w2_packed = torch.stack([q[0] for q in w2_q]).contiguous()
    ones = torch.ones(E, device=device)

    reference = moe_reference_w4a8_mx(
        x.float(),
        w13_packed, w13_mx, ref_res_w13, ones,
        w2_packed, w2_mx, ref_res_w2, ones,
        topk_ids, topk_weights, E, K, n,
        activation=activation,
    )

    # ---- workspace ----
    # Worst-case physical tiles: every expert pays one partial tile plus the
    # full tiles its routed rows occupy.
    phys_tiles = E + (m * top_k + tile_m - 1) // tile_m
    rows_padded = phys_tiles * tile_m
    gate_tile_cnt = (w1_n // _TILE_N) // (2 if is_gated else 1)
    max_tasks = phys_tiles * max(gate_tile_cnt, 1)
    mac = 4

    packed_a = torch.zeros(rows_padded * K, dtype=torch.uint8, device=device)
    # Sized for the (unused) vec16 SF TMA descriptor view: rows * K/8 bytes.
    scale_flat = torch.zeros(rows_padded * (K // 8), dtype=torch.uint8, device=device)
    def z1():
        return torch.zeros(1, dtype=torch.int32, device=device)

    barrier_count, barrier_epoch = z1(), z1()
    pair_head, producers_done, all_pub = z1(), z1(), z1()
    task_head, task_tail = z1(), z1()

    def zt():
        return torch.zeros(max_tasks, dtype=torch.int32, device=device)

    task_ready, task_expert, task_m_tile = zt(), zt(), zt()
    task_slice_begin, task_slice_count, task_valid_rows = zt(), zt(), zt()
    tile_write_count = torch.zeros(phys_tiles, dtype=torch.int32, device=device)
    row_counts = torch.zeros(E, dtype=torch.int32, device=device)
    expert_write_rows = torch.zeros(E, dtype=torch.int32, device=device)
    expert_tile_base = torch.zeros(E + 1, dtype=torch.int32, device=device)
    token_map = torch.zeros(rows_padded, dtype=torch.int32, device=device)
    token_weights = torch.zeros(rows_padded, dtype=torch.float32, device=device)
    scatter_output = torch.zeros(m, K, dtype=torch.bfloat16, device=device)
    flat_ids = topk_ids.reshape(-1).contiguous()
    flat_weights = topk_weights.reshape(-1).contiguous()

    kernel = MoEDynamicKernelBackend(
        16,
        (tile_m, _TILE_N),
        activation=activation,
        quant_recipe=recipe,
    )
    launch = _DynamicMoEW4A8Launch(kernel, k=K, n=n, w1_n=w1_n, num_topk=top_k)

    weight_dtype = cutlass.Float4E2M1FN
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype, (w1_n, K, E), stride_order=(1, 0, 2), assumed_align=16
    )
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype, (K, n, E), stride_order=(1, 0, 2), assumed_align=16
    )
    def fake_ptr_u8():
        return make_ptr(
            cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
        )

    def fake_ptr_i32():
        return make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)

    compiled = b12x_compile(
        launch,
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(weight_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_u8(),
        fake_ptr_u8(),
        _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
        _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
        fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
        fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
        b_w13_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        b_down_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(),
        _fake_i32((E,)), _fake_i32((E,)), _fake_i32((E + 1,)),
        _fake_f32((E,)), _fake_f32((E,)), _fake_f32((E,)), _fake_f32((E,)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        1, 1, 1, 1, 1, 1, 1,
        current_cuda_stream(),
        dsl_compile_options=_OPT_LEVEL_2,
    )

    compiled(
        _gptr(cutlass.BFloat16, x),
        _gptr(cutlass.Int32, flat_ids, 4),
        _gptr(cutlass.Float32, flat_weights, 4),
        _gptr(weight_dtype, packed_a),
        _gptr(cutlass.Float8E4M3FN, scale_flat),
        _gptr(cutlass.Uint8, packed_a),
        _gptr(cutlass.Uint8, scale_flat),
        barrier_count, barrier_epoch, pair_head, producers_done, all_pub,
        task_head, task_tail,
        _gptr(cutlass.Int32, task_ready, 4),
        _gptr(cutlass.Int32, task_expert, 4),
        _gptr(cutlass.Int32, task_m_tile, 4),
        _gptr(cutlass.Int32, task_slice_begin, 4),
        _gptr(cutlass.Int32, task_slice_count, 4),
        _gptr(cutlass.Int32, task_valid_rows, 4),
        _gptr(cutlass.Int32, tile_write_count, 4),
        w13_packed,
        _gptr(cutlass.Float8E4M3FN, scale_flat),
        w2_packed,
        _gptr(cutlass.Float8E4M3FN, scale_flat),
        _gptr(cutlass.Uint8, w13_mx),
        _gptr(cutlass.Uint8, w2_mx),
        _gptr(cutlass.Uint8, w13_res),
        _gptr(cutlass.Uint8, w2_res),
        row_counts, expert_write_rows, expert_tile_base,
        ones, ones, ones, ones,
        _gptr(cutlass.BFloat16, scatter_output),
        _gptr(cutlass.Int32, token_map, 4),
        _gptr(cutlass.Float32, token_weights, 4),
        m,
        m * top_k,
        m,
        rows_padded,
        max_tasks,
        phys_tiles,
        mac,
        current_cuda_stream(),
    )
    torch.cuda.synchronize()
    if return_launcher:
        def _relaunch():
            compiled(
                _gptr(cutlass.BFloat16, x),
                _gptr(cutlass.Int32, flat_ids, 4),
                _gptr(cutlass.Float32, flat_weights, 4),
                _gptr(weight_dtype, packed_a),
                _gptr(cutlass.Float8E4M3FN, scale_flat),
                _gptr(cutlass.Uint8, packed_a),
                _gptr(cutlass.Uint8, scale_flat),
                barrier_count, barrier_epoch, pair_head, producers_done, all_pub,
                task_head, task_tail,
                _gptr(cutlass.Int32, task_ready, 4),
                _gptr(cutlass.Int32, task_expert, 4),
                _gptr(cutlass.Int32, task_m_tile, 4),
                _gptr(cutlass.Int32, task_slice_begin, 4),
                _gptr(cutlass.Int32, task_slice_count, 4),
                _gptr(cutlass.Int32, task_valid_rows, 4),
                _gptr(cutlass.Int32, tile_write_count, 4),
                w13_packed,
                _gptr(cutlass.Float8E4M3FN, scale_flat),
                w2_packed,
                _gptr(cutlass.Float8E4M3FN, scale_flat),
                _gptr(cutlass.Uint8, w13_mx),
                _gptr(cutlass.Uint8, w2_mx),
                _gptr(cutlass.Uint8, w13_res),
                _gptr(cutlass.Uint8, w2_res),
                row_counts, expert_write_rows, expert_tile_base,
                ones, ones, ones, ones,
                _gptr(cutlass.BFloat16, scatter_output),
                _gptr(cutlass.Int32, token_map, 4),
                _gptr(cutlass.Float32, token_weights, 4),
                m, m * top_k, m, rows_padded, max_tasks, phys_tiles, mac,
                current_cuda_stream(),
            )
        return scatter_output, reference, _relaunch
    return scatter_output, reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("recipe", ["w4a8_mx", "w4a8_nvfp4"])
@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_w4a8_dynamic_matches_oracle(recipe: str, activation: str) -> None:
    require_sm120()
    out, ref = _run_w4a8_dynamic(
        recipe=recipe, activation=activation,
        E=4, m=8, K=256, n=128, top_k=2, seed=11,
    )
    assert out.abs().sum().item() > 0, "kernel produced all zeros"
    metrics = compare_to_reference(out.float(), ref)
    assert metrics.cos > 0.999, metrics
    ref_rms = ref.float().square().mean().sqrt().item()
    assert metrics.rmse <= max(0.03 * ref_rms, 5e-3), (metrics, ref_rms)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_w4a8_dynamic_small_tile_parallel_regime_matches_oracle(
    activation: str,
) -> None:
    """Exercise the production M16/four-MMA/two-DMA regime directly."""
    require_sm120()
    kernel = MoEDynamicKernelBackend(
        16,
        (16, _TILE_N),
        activation=activation,
        quant_recipe="w4a8_mx",
    )
    assert kernel.atom_shape == (1, 4, 1)
    assert kernel.num_mma_warps == 4
    assert kernel.num_dma_warps == 2
    out, ref = _run_w4a8_dynamic(
        recipe="w4a8_mx",
        activation=activation,
        E=4,
        m=17,
        K=256,
        n=128,
        top_k=2,
        seed=29,
        tile_m=16,
    )
    metrics = compare_to_reference(out.float(), ref)
    assert metrics.cos > 0.999, metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_dynamic_boundary_m_sizes() -> None:
    require_sm120()
    for m in (1, 3, 127, 129):
        out, ref = _run_w4a8_dynamic(
            recipe="w4a8_mx", activation="silu",
            E=4, m=m, K=256, n=128, top_k=2, seed=100 + m,
        )
        assert out.abs().sum().item() > 0, m
        metrics = compare_to_reference(out.float(), ref)
        assert metrics.cos > 0.999, (m, metrics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_dynamic_graph_replay_tracks_routing_updates() -> None:
    """Capture the w4a8 launch in a CUDA graph; replay must track routing."""
    require_sm120()
    device = torch.device("cuda")
    torch.manual_seed(7)
    E, m, K, n, top_k = 4, 8, 256, 128, 2
    w1_n = 2 * n

    x = (torch.randn(m, K, device=device) * 2.0).to(torch.bfloat16)
    w13_full = torch.randn(E, w1_n, K, device=device) * 0.05
    w2_full = torch.randn(E, K, n, device=device) * 0.05
    w13_q = [_quantize_weight_mxfp4(w13_full[e]) for e in range(E)]
    w2_q = [_quantize_weight_mxfp4(w2_full[e]) for e in range(E)]
    w13_packed = torch.stack([q[0] for q in w13_q]).contiguous()
    w2_packed = torch.stack([q[0] for q in w2_q]).contiguous()
    w13_mx = torch.stack([q[1] for q in w13_q]).contiguous()
    w2_mx = torch.stack([q[1] for q in w2_q]).contiguous()
    w13_res = torch.zeros(E, w1_n, K // 16, dtype=torch.uint8, device=device)
    w2_res = torch.zeros(E, K, n // 16, dtype=torch.uint8, device=device)
    ones = torch.ones(E, device=device)

    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, top_k, device=device), dim=-1).float()
    flat_ids = topk_ids.reshape(-1).contiguous()
    flat_weights = topk_weights.reshape(-1).contiguous()

    phys_tiles = E + 1
    rows_padded = phys_tiles * _TILE_M
    max_tasks = phys_tiles
    packed_a = torch.zeros(rows_padded * K, dtype=torch.uint8, device=device)
    scale_flat = torch.zeros(rows_padded * (K // 8), dtype=torch.uint8, device=device)
    def z1():
        return torch.zeros(1, dtype=torch.int32, device=device)

    barrier_count, barrier_epoch = z1(), z1()
    pair_head, producers_done, all_pub = z1(), z1(), z1()
    task_head, task_tail = z1(), z1()

    def zt():
        return torch.zeros(max_tasks, dtype=torch.int32, device=device)

    task_ready, task_expert, task_m_tile = zt(), zt(), zt()
    task_slice_begin, task_slice_count, task_valid_rows = zt(), zt(), zt()
    tile_write_count = torch.zeros(phys_tiles, dtype=torch.int32, device=device)
    row_counts = torch.zeros(E, dtype=torch.int32, device=device)
    expert_write_rows = torch.zeros(E, dtype=torch.int32, device=device)
    expert_tile_base = torch.zeros(E + 1, dtype=torch.int32, device=device)
    token_map = torch.zeros(rows_padded, dtype=torch.int32, device=device)
    token_weights = torch.zeros(rows_padded, dtype=torch.float32, device=device)
    scatter_output = torch.zeros(m, K, dtype=torch.bfloat16, device=device)

    kernel = MoEDynamicKernelBackend(
        16, (_TILE_M, _TILE_N), activation="silu", quant_recipe="w4a8_mx"
    )
    launch = _DynamicMoEW4A8Launch(kernel, k=K, n=n, w1_n=w1_n, num_topk=top_k)
    weight_dtype = cutlass.Float4E2M1FN
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype, (w1_n, K, E), stride_order=(1, 0, 2), assumed_align=16
    )
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype, (K, n, E), stride_order=(1, 0, 2), assumed_align=16
    )
    def fake_ptr_u8():
        return make_ptr(
            cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
        )

    def fake_ptr_i32():
        return make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)

    compiled = b12x_compile(
        launch,
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(weight_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_u8(), fake_ptr_u8(),
        _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
        _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
        fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
        fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
        b_w13_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        b_down_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(),
        _fake_i32((E,)), _fake_i32((E,)), _fake_i32((E + 1,)),
        _fake_f32((E,)), _fake_f32((E,)), _fake_f32((E,)), _fake_f32((E,)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        1, 1, 1, 1, 1, 1, 1,
        current_cuda_stream(),
        dsl_compile_options=_OPT_LEVEL_2,
    )

    def _launch():
        compiled(
            _gptr(cutlass.BFloat16, x),
            _gptr(cutlass.Int32, flat_ids, 4),
            _gptr(cutlass.Float32, flat_weights, 4),
            _gptr(weight_dtype, packed_a),
            _gptr(cutlass.Float8E4M3FN, scale_flat),
            _gptr(cutlass.Uint8, packed_a),
            _gptr(cutlass.Uint8, scale_flat),
            barrier_count, barrier_epoch, pair_head, producers_done, all_pub,
            task_head, task_tail,
            _gptr(cutlass.Int32, task_ready, 4),
            _gptr(cutlass.Int32, task_expert, 4),
            _gptr(cutlass.Int32, task_m_tile, 4),
            _gptr(cutlass.Int32, task_slice_begin, 4),
            _gptr(cutlass.Int32, task_slice_count, 4),
            _gptr(cutlass.Int32, task_valid_rows, 4),
            _gptr(cutlass.Int32, tile_write_count, 4),
            w13_packed,
            _gptr(cutlass.Float8E4M3FN, scale_flat),
            w2_packed,
            _gptr(cutlass.Float8E4M3FN, scale_flat),
            _gptr(cutlass.Uint8, w13_mx),
            _gptr(cutlass.Uint8, w2_mx),
            _gptr(cutlass.Uint8, w13_res),
            _gptr(cutlass.Uint8, w2_res),
            row_counts, expert_write_rows, expert_tile_base,
            ones, ones, ones, ones,
            _gptr(cutlass.BFloat16, scatter_output),
            _gptr(cutlass.Int32, token_map, 4),
            _gptr(cutlass.Float32, token_weights, 4),
            m, m * top_k, m, rows_padded, max_tasks, phys_tiles, 4,
            current_cuda_stream(),
        )

    def _oracle():
        return moe_reference_w4a8_mx(
            x.float(), w13_packed, w13_mx, None, ones,
            w2_packed, w2_mx, None, ones,
            flat_ids.view(m, top_k), flat_weights.view(m, top_k), E, K, n,
            activation="silu",
        )

    # Warm up (also pre-zeros barrier state path), then capture.
    _launch()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        _launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    for round_idx in range(3):
        torch.manual_seed(100 + round_idx)
        new_ids = torch.stack(
            [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
        ).to(torch.int32)
        new_w = torch.softmax(torch.randn(m, top_k, device=device), dim=-1).float()
        flat_ids.copy_(new_ids.reshape(-1))
        flat_weights.copy_(new_w.reshape(-1))
        x.copy_((torch.randn(m, K, device=device) * 2.0).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        ref = _oracle()
        metrics = compare_to_reference(scatter_output.float(), ref)
        assert scatter_output.abs().sum().item() > 0, round_idx
        assert metrics.cos > 0.999, (round_idx, metrics)
