from __future__ import annotations

import pytest
import torch

from b12x.cute.fp4 import swizzle_block_scale
from b12x.integration.tp_moe import allocate_tp_moe_workspace_pool, b12x_moe_fp4
from b12x.moe.fused.w4a16.kernel import run_w4a16_moe
from b12x.moe.fused.w4a16.prepare import (
    make_w4a16_packed_buffers as make_w4a16_buffers,
    prepare_w4a16_packed_weights as prepare_w4a16_weights,
)
from tests.w4a16_reference import compare_to_reference, moe_reference_w4a16


def _positive_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.rand(shape, device="cuda") * 0.25 + 0.03125).to(torch.float8_e4m3fn)


def _make_weights(
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.1 + 0.05).to(torch.float32)
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.1 + 0.05).to(torch.float32)
    return w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale


def _run_w4a16(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
    expert_map: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    prepared = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=x.shape[0],
        topk=topk_ids.shape[1],
        dtype=x.dtype,
        device=x.device,
        route_num_experts=None if expert_map is None else int(expert_map.numel()),
    )
    return run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        expert_map=expert_map,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        swiglu_limit=swiglu_limit,
    )


def _reference_w4a16(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
    expert_map: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    reference_topk_ids = topk_ids
    if expert_map is not None:
        reference_topk_ids = expert_map[topk_ids.long()].to(torch.int32)
        assert bool((reference_topk_ids >= 0).all().item())
    return moe_reference_w4a16(
        x,
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        reference_topk_ids,
        topk_weights,
        w13.shape[0],
        w2.shape[1],
        w2.shape[2] * 2,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )


def _assert_matches_oracle(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    activation: str,
) -> None:
    metrics = compare_to_reference(actual, expected)
    min_cos = 0.9975 if activation == "silu" else 0.9900
    assert metrics.cos >= min_cos, metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize(
    ("routed_size", "m"),
    [(8, 16), (16, 32), (32, 64), (48, 128), (64, 192)],
)
def test_w4a16_moe_matches_oracle(
    activation: str,
    routed_size: int,
    m: int,
) -> None:
    torch.manual_seed(20260515 + routed_size + (1000 if activation == "silu" else 0))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk = 2
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_moe_matches_oracle_with_expert_map(
    activation: str,
) -> None:
    torch.manual_seed(20260516 + (1000 if activation == "silu" else 0))
    global_experts, local_experts = 8, 4
    hidden_size, intermediate_size = 128, 128
    topk, m = 2, 24
    weights = _make_weights(
        experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[::2] = torch.arange(local_experts, dtype=torch.int32, device="cuda")

    valid_global_ids = torch.arange(0, global_experts, 2, dtype=torch.int32, device="cuda")
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = valid_global_ids[
        torch.randint(0, local_experts, (m, topk), device="cuda")
    ].to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        expert_map=expert_map,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        expert_map=expert_map,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_moe_swiglu_limit_matches_oracle_under_cuda_graph() -> None:
    torch.manual_seed(20260519)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    activation = "silu"
    swiglu_limit = 10.0
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, _, w2, w2_blockscale, w2_global_scale = weights
    w13_global_scale = torch.full(
        (experts,), 8.0, dtype=torch.float32, device="cuda"
    )
    weights = (
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 2.0).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=x.shape[0],
        topk=topk_ids.shape[1],
        dtype=x.dtype,
        device=x.device,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )

    eager = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        swiglu_limit=swiglu_limit,
    )
    torch.cuda.synchronize()
    _assert_matches_oracle(eager, expected, activation=activation)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            swiglu_limit=swiglu_limit,
        )
    graph.replay()
    torch.cuda.synchronize()

    _assert_matches_oracle(buffers.output, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tp_moe_w4a16_dispatch_uses_native_path() -> None:
    torch.manual_seed(20260518)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    activation = "relu2"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights

    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    output = torch.empty_like(x)
    actual = b12x_moe_fp4(
        x,
        torch.ones((), dtype=torch.float32, device="cuda"),
        w13,
        w13_blockscale,
        w13_global_scale,
        torch.ones((), dtype=torch.float32, device="cuda"),
        w2,
        w2_blockscale,
        w2_global_scale,
        topk_weights,
        topk_ids,
        workspace=allocate_tp_moe_workspace_pool(),
        output=output,
        activation=activation,
        quant_mode="w4a16",
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert actual is output
    _assert_matches_oracle(actual, expected, activation=activation)
