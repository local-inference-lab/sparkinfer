from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from cutlass.utils.layout import LayoutEnum

import b12x.integration.tp_moe_bf16 as tp_moe_bf16
from b12x.cute.utils import current_cuda_stream
from b12x.integration.triton_bf16_scatter import gather_rows_bf16, permute_rows_bf16
from b12x.integration.triton_compact import build_bucketed_compact_route
from b12x.integration.tp_moe_bf16 import (
    _select_tp_moe_backend,
    allocate_tp_moe_bf16_workspace_pool,
    b12x_moe_bf16,
    clear_tp_moe_bf16_caches,
)
from b12x.moe.fused.bf16.indexed_dense import (
    ExpertIndexedDenseGemmKernel,
    ExpertIndexedDenseRow1GridKernel,
    run_dense_bf16_expert_ids,
)
from b12x.moe.fused.bf16.reference import compare_to_reference, moe_reference_bf16
from b12x.moe.fused.bf16.static import (
    MoEStaticKernelBackend,
    _to_dense_kernel_tensor,
    run_fused_relu2_bf16_expert_ids,
)

from .helpers import require_sm120


BACKEND_CASES = [
    ("micro", 2),
    ("static", 256),
    ("dynamic", 640),
]


def _assert_bf16_close(
    metrics,
    label: str,
    *,
    max_abs: float = 0.01,
    rmse: float = 1e-4,
    cos: float = 0.99999,
) -> None:
    assert metrics.max_abs <= max_abs, f"{label}: {metrics}"
    assert metrics.rmse <= rmse, f"{label}: {metrics}"
    assert metrics.cos >= cos, f"{label}: {metrics}"


def _assert_legacy_routing_removed() -> None:
    assert not hasattr(tp_moe_bf16, "_build_routing_layout")


def _run_legacy_relu2_without_compact(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    orig = tp_moe_bf16._should_use_compact_relu2
    try:
        tp_moe_bf16._should_use_compact_relu2 = lambda **kwargs: False
        clear_tp_moe_bf16_caches()
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(x.device)
        return output
    finally:
        tp_moe_bf16._should_use_compact_relu2 = orig
        clear_tp_moe_bf16_caches()


def _run_legacy_relu2_without_special_routes(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    orig_compact = tp_moe_bf16._should_use_compact_relu2
    orig_fp4_shaped = tp_moe_bf16._should_use_fp4_shaped_relu2_route
    try:
        tp_moe_bf16._should_use_compact_relu2 = lambda **kwargs: False
        tp_moe_bf16._should_use_fp4_shaped_relu2_route = lambda **kwargs: False
        clear_tp_moe_bf16_caches()
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(x.device)
        return output
    finally:
        tp_moe_bf16._should_use_compact_relu2 = orig_compact
        tp_moe_bf16._should_use_fp4_shaped_relu2_route = orig_fp4_shaped
        clear_tp_moe_bf16_caches()


def _make_case(
    *,
    device: torch.device,
    activation: str,
    m: int,
    e: int = 8,
    n: int = 128,
    topk: int = 2,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)

    k = 128
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    topk_logits = torch.randn(m, topk, device=device, dtype=torch.float32)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    topk_ids = torch.randint(0, e, (m, topk), device=device, dtype=torch.int32)

    w1_rows = 2 * n if activation == "silu" else n
    w1 = torch.randn(e, w1_rows, k, device=device, dtype=torch.bfloat16) * 0.25
    w2 = torch.randn(e, k, n, device=device, dtype=torch.bfloat16) * 0.125
    return x, w1, w2, topk_weights, topk_ids, e, k, n


def _sample_unique_topk_ids(
    *,
    device: torch.device,
    num_tokens: int,
    num_experts: int,
    topk: int,
) -> torch.Tensor:
    routing_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    return torch.topk(routing_logits, topk, dim=-1).indices.to(torch.int32)


def _make_nemotron_case(
    *,
    device: torch.device,
    activation: str = "relu2",
    m: int,
    e: int = 128,
) -> tuple[torch.Tensor, ...]:
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation=activation,
        m=m,
        e=e,
        topk=22,
    )
    topk_ids.copy_(
        _sample_unique_topk_ids(
            device=device,
            num_tokens=m,
            num_experts=e,
            topk=topk_ids.shape[1],
        )
    )
    return x, w1, w2, topk_weights, topk_ids, e, k, n


def _make_mixed_repeat_nemotron_ids(
    *,
    device: torch.device,
    num_tokens: int,
    num_experts: int,
    topk: int,
    shared_experts: int = 8,
) -> torch.Tensor:
    if shared_experts <= 0 or shared_experts >= topk:
        raise ValueError("shared_experts must be in [1, topk)")
    unique_per_token = topk - shared_experts
    required_experts = shared_experts + num_tokens * unique_per_token
    if required_experts > num_experts:
        raise ValueError(
            f"num_experts={num_experts} is too small for num_tokens={num_tokens}, "
            f"topk={topk}, shared_experts={shared_experts}"
        )

    shared_ids = torch.arange(shared_experts, device=device, dtype=torch.int32)
    topk_ids = torch.empty((num_tokens, topk), device=device, dtype=torch.int32)
    next_unique = shared_experts
    for token_idx in range(num_tokens):
        unique_ids = torch.arange(
            next_unique,
            next_unique + unique_per_token,
            device=device,
            dtype=torch.int32,
        )
        next_unique += unique_per_token
        topk_ids[token_idx].copy_(torch.cat((shared_ids, unique_ids), dim=0))
    return topk_ids


def _alloc_row1_dense(
    *,
    feature_dim: int,
    routed_rows: int,
    device: torch.device,
) -> torch.Tensor:
    return tp_moe_bf16._alloc_batched_matrix(
        1,
        feature_dim,
        routed_rows,
        mode0_major=True,
        dtype=torch.bfloat16,
        device=device,
    )


@pytest.mark.parametrize("activation", ["silu", "relu2"])
@pytest.mark.parametrize(("backend", "m"), BACKEND_CASES)
def test_bf16_backend_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    activation: str,
    backend: str,
    m: int,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation=activation,
        m=m,
    )
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation=activation,
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", backend)
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation=activation,
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    if activation == "relu2" and backend in {"static", "dynamic"}:
        _assert_bf16_close(
            metrics,
            f"{activation}/{backend}",
            max_abs=0.26,
            rmse=0.035,
            cos=0.99999,
        )
    else:
        assert metrics.max_abs == 0.0, f"{activation}/{backend}: {metrics}"
        assert metrics.rmse == 0.0, f"{activation}/{backend}: {metrics}"


@pytest.mark.parametrize("backend", ["micro", "static", "dynamic"])
def test_bf16_silu_runtime_does_not_build_routing_layout(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, _, _, _ = _make_case(
        device=device,
        activation="silu",
        m=8 if backend != "dynamic" else 32,
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", backend)
    _assert_legacy_routing_removed()
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="silu",
        )
        reference = moe_reference_bf16(
            x,
            w1,
            w2,
            topk_ids,
            topk_weights,
            activation="silu",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    assert metrics.max_abs == 0.0, f"silu/{backend}: {metrics}"
    assert metrics.rmse == 0.0, f"silu/{backend}: {metrics}"


@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_bf16_static_wide_intermediate_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    activation: str,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation=activation,
        m=256,
        n=256,
    )
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation=activation,
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation=activation,
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    if activation == "relu2":
        _assert_bf16_close(
            metrics,
            f"{activation}/static-wide",
            max_abs=0.51,
            rmse=0.05,
            cos=0.99999,
        )
    else:
        _assert_bf16_close(metrics, f"{activation}/static-wide")


@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_bf16_cuda_graph_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    activation: str,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation=activation,
        m=4,
    )
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation=activation,
    )

    pool = allocate_tp_moe_bf16_workspace_pool()
    output = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()

    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            output=output,
            activation=activation,
        )
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            b12x_moe_bf16(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                workspace=pool,
                output=output,
                activation=activation,
            )
        graph.replay()
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    assert metrics.max_abs == 0.0, f"{activation}: {metrics}"
    assert metrics.rmse == 0.0, f"{activation}: {metrics}"


def test_bf16_relu2_micro_pool_does_not_cache_routing_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=16,
    )

    pool = allocate_tp_moe_bf16_workspace_pool()
    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert len(pool.workspaces) == 1
    workspace = next(iter(pool.workspaces.values()))
    assert not hasattr(workspace, "routing_layout")


@pytest.mark.parametrize("m", [1, 16, 80])
def test_bf16_relu2_direct_path_does_not_build_routing_layout(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=m,
        topk=4,
    )
    pool = allocate_tp_moe_bf16_workspace_pool()

    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    _assert_legacy_routing_removed()
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()


def test_bf16_relu2_direct_path_clears_stale_routing_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=16,
        topk=4,
    )
    pool = allocate_tp_moe_bf16_workspace_pool()

    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert len(pool.workspaces) == 1
    workspace = next(iter(pool.workspaces.values()))
    stale_routing = object()
    stale_weight_key = (123, 456)
    workspace.routing_layout = stale_routing  # type: ignore[attr-defined]
    workspace.weight_key = stale_weight_key  # type: ignore[attr-defined]

    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    _assert_legacy_routing_removed()
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert workspace.routing_layout is stale_routing  # type: ignore[attr-defined]
    assert workspace.weight_key == stale_weight_key  # type: ignore[attr-defined]


@pytest.mark.parametrize("implementation", ["static", "dynamic"])
def test_bf16_relu2_chunk_routing_uses_expert_ids_not_grouped_weights(
    implementation: str,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=256 if implementation == "static" else 640,
        topk=4,
    )
    _assert_legacy_routing_removed()

    backend = tp_moe_bf16._get_static_kernel(activation="relu2", num_topk=topk_ids.shape[1])
    expert_chunk_size = tp_moe_bf16._expert_chunk_size(
        backend,
        num_tokens=x.shape[0],
        num_topk=topk_ids.shape[1],
    )
    workspace = tp_moe_bf16._resolve_workspace(
        allocate_tp_moe_bf16_workspace_pool(),
        implementation="static",
        max_tokens=x.shape[0],
        k=k,
        n=n,
        weight_E=e,
        num_topk=topk_ids.shape[1],
        activation="relu2",
        expert_chunk_size=expert_chunk_size,
        device=device,
        routing_layout=tp_moe_bf16._route_state_workspace_sizing(
            num_tokens=x.shape[0],
            num_topk=topk_ids.shape[1],
            expert_chunk_size=expert_chunk_size,
            weight_E=e,
        ),
    )
    route = tp_moe_bf16._prepare_compact_route_state(
        workspace,
        topk_ids,
        topk_weights,
        route_kind="fp4_shaped",
        build_fp4_metadata=True,
    )
    chunk_ranges = backend._compact_route_chunk_ranges(
        route=route,
        expert_chunk_size=expert_chunk_size,
    )

    assert route.kernel_weight_expert_ids is not None
    assert route.kernel_weight_expert_ids.dtype == torch.int32
    assert chunk_ranges
    for expert_begin, expert_end in chunk_ranges:
        chunk_ids = route.kernel_weight_expert_ids[expert_begin:expert_end]
        assert chunk_ids.numel() == expert_end - expert_begin
        assert chunk_ids.dtype == torch.int32


def test_bf16_relu2_small_row_static_scheduler_keeps_multirow_groups_bucketed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    e, k, n, topk = 8, 128, 128, 2
    x = torch.randn(8, k, device=device, dtype=torch.bfloat16)
    topk_weights = torch.full((8, topk), 0.5, device=device, dtype=torch.float32)
    topk_ids = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
        device=device,
        dtype=torch.int32,
    )
    w1 = torch.randn(e, n, k, device=device, dtype=torch.bfloat16) * 0.25
    w2 = torch.randn(e, k, n, device=device, dtype=torch.bfloat16) * 0.125
    _assert_legacy_routing_removed()

    monkeypatch.setenv("B12X_BF16_ENABLE_BUCKETED_COMPACT_RELU2_STATIC", "1")
    clear_tp_moe_bf16_caches()
    try:
        pool = allocate_tp_moe_bf16_workspace_pool()
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert output.shape == x.shape
    workspace = next(iter(pool.workspaces.values()))
    assert workspace.compact_bucket_expert_ids is not None
    assert workspace.compact_bucket_token_map is not None
    assert workspace.compact_bucket_token_weights is not None
    assert workspace.direct_route_expert_ids_i32 is not None
    assert workspace.direct_route_expert_ids_i32.shape[0] >= x.shape[0] * topk_ids.shape[1]
    assert workspace.direct_w1_view is not None
    assert workspace.direct_w2_view is not None


def test_bf16_relu2_micro_large_eager_stays_close_to_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=80,
        topk=4,
    )
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation="relu2",
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    _assert_bf16_close(
        metrics,
        "relu2/micro-large-direct",
        max_abs=0.26,
        rmse=0.04,
        cos=0.99999,
    )


def test_bf16_relu2_micro_multirow_direct_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=256,
        topk=1,
    )
    topk_ids = torch.randint(0, e, (x.shape[0], 1), device=device, dtype=torch.int32)
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation="relu2",
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    _assert_bf16_close(
        metrics,
        "relu2/micro-multirow-direct-topk1",
    )


def test_bf16_relu2_micro_topk4_tracks_reference_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=256,
        topk=4,
    )
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation="relu2",
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    _assert_bf16_close(
        metrics,
        "relu2/micro-topk4-envelope",
        max_abs=0.3,
        rmse=0.04,
        cos=0.99999,
    )


def test_bf16_relu2_backend_cutover_matches_nemotron_bs_regime() -> None:
    # Nemotron uses top_k=22. Keep the relu2 selector aligned with the
    # shipped route path we actually care about:
    # bs=1/2/4/8 -> static.
    assert _select_tp_moe_backend(
        num_tokens=1, num_topk=22, activation="relu2"
    ) == "static"
    assert _select_tp_moe_backend(
        num_tokens=2, num_topk=22, activation="relu2"
    ) == "static"
    assert _select_tp_moe_backend(
        num_tokens=4, num_topk=22, activation="relu2"
    ) == "static"
    assert _select_tp_moe_backend(
        num_tokens=8, num_topk=22, activation="relu2"
    ) == "static"


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_bf16_relu2_compact_direct_route_is_default_for_nemotron_bs_regime(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_nemotron_case(
        device=device,
        m=m,
    )

    implementation = _select_tp_moe_backend(
        num_tokens=m,
        num_topk=22,
        activation="relu2",
    )
    backend_cls = type(
        tp_moe_bf16._resolve_backend(
            implementation=implementation,
            activation="relu2",
            num_topk=22,
        )
    )
    orig_bucketed = backend_cls._run_bucketed_compact_static_route
    orig_direct = backend_cls._run_compact_direct_route
    bucketed_call_count = 0
    direct_call_count = 0

    def _wrapped_bucketed(self, *args, **kwargs):
        nonlocal bucketed_call_count
        bucketed_call_count += 1
        return orig_bucketed(self, *args, **kwargs)

    def _wrapped_direct(self, *args, **kwargs):
        nonlocal direct_call_count
        direct_call_count += 1
        return orig_direct(self, *args, **kwargs)

    monkeypatch.setattr(
        backend_cls,
        "_run_bucketed_compact_static_route",
        _wrapped_bucketed,
    )
    monkeypatch.setattr(
        backend_cls,
        "_run_compact_direct_route",
        _wrapped_direct,
    )
    monkeypatch.delenv("B12X_BF16_BACKEND", raising=False)
    monkeypatch.delenv("B12X_BF16_ENABLE_COMPACT_RELU2", raising=False)
    monkeypatch.delenv("B12X_BF16_ENABLE_BUCKETED_COMPACT_RELU2_STATIC", raising=False)
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert bucketed_call_count == 0
    assert direct_call_count > 0


def test_bf16_relu2_compact_route_precomputes_sorted_direct_buffers() -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        topk=22,
    )

    backend = tp_moe_bf16._resolve_backend(
        implementation="static",
        activation="relu2",
        num_topk=topk_ids.shape[1],
    )
    expert_chunk_size = tp_moe_bf16._expert_chunk_size(
        backend,
        num_tokens=x.shape[0],
        num_topk=topk_ids.shape[1],
    )
    workspace = tp_moe_bf16._resolve_workspace(
        None,
        implementation="static",
        max_tokens=x.shape[0],
        k=k,
        n=n,
        weight_E=e,
        num_topk=topk_ids.shape[1],
        activation="relu2",
        expert_chunk_size=expert_chunk_size,
        device=device,
        routing_layout=tp_moe_bf16._compact_relu2_workspace_sizing(
            num_tokens=x.shape[0],
            num_topk=topk_ids.shape[1],
            weight_E=e,
        ),
    )
    route = tp_moe_bf16._prepare_compact_route_layout(
        workspace,
        topk_ids,
        topk_weights,
    )

    assert route.sorted_route_order_i64 is not None
    assert route.sorted_flat_ids_i32 is not None
    assert route.sorted_flat_token_indices is not None

    precomputed = backend._build_sorted_compact_direct_route(
        route=route,
        workspace=workspace,
    )
    precomputed_clones = tuple(t.clone() for t in precomputed)
    fallback_route = replace(
        route,
        sorted_route_order_i64=None,
        sorted_flat_ids_i32=None,
        sorted_flat_token_indices=None,
    )
    rebuilt = backend._build_sorted_compact_direct_route(
        route=fallback_route,
        workspace=workspace,
    )

    for got, expected in zip(precomputed_clones, rebuilt):
        torch.testing.assert_close(got, expected)


def test_bf16_relu2_chunked_direct_route_skips_unused_compact_metadata() -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        topk=22,
    )

    backend = tp_moe_bf16._resolve_backend(
        implementation="static",
        activation="relu2",
        num_topk=topk_ids.shape[1],
    )
    expert_chunk_size = tp_moe_bf16._expert_chunk_size(
        backend,
        num_tokens=x.shape[0],
        num_topk=topk_ids.shape[1],
    )
    workspace = tp_moe_bf16._resolve_workspace(
        None,
        implementation="static",
        max_tokens=x.shape[0],
        k=k,
        n=n,
        weight_E=e,
        num_topk=topk_ids.shape[1],
        activation="relu2",
        expert_chunk_size=expert_chunk_size,
        device=device,
        routing_layout=tp_moe_bf16._route_state_workspace_sizing(
            num_tokens=x.shape[0],
            num_topk=topk_ids.shape[1],
            expert_chunk_size=expert_chunk_size,
            weight_E=e,
        ),
    )
    route = tp_moe_bf16._prepare_compact_route_state(
        workspace,
        topk_ids,
        topk_weights,
        route_kind="chunked",
        build_fp4_metadata=False,
    )

    assert route.sorted_route_order_i64 is not None
    assert route.sorted_flat_ids_i32 is not None
    assert route.sorted_flat_token_indices is not None
    assert route.compact_topk_ids.numel() == topk_ids.numel()
    assert route.route_row_indices.numel() == topk_ids.numel()
    assert route.row_counts.numel() > 0
    assert route.weight_expert_ids.numel() == route.row_counts.numel()
    assert route.kernel_weight_expert_ids.numel() == route.row_counts.numel()
    assert int(route.active_expert_count.item()) == int(
        (route.row_counts > 0).sum().item()
    )

    direct_sorted = backend._build_sorted_compact_direct_route(
        route=route,
        workspace=workspace,
    )
    for cached, built in zip(
        (
            route.sorted_route_order_i64,
            route.sorted_flat_ids_i32,
            route.sorted_flat_token_indices,
        ),
        direct_sorted,
    ):
        assert cached is built


def test_bf16_relu2_bucketed_compact_route_builder_matches_reference() -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        topk=22,
    )

    backend = tp_moe_bf16._resolve_backend(
        implementation="static",
        activation="relu2",
        num_topk=topk_ids.shape[1],
    )
    expert_chunk_size = tp_moe_bf16._expert_chunk_size(
        backend,
        num_tokens=x.shape[0],
        num_topk=topk_ids.shape[1],
    )
    workspace = tp_moe_bf16._resolve_workspace(
        None,
        implementation="static",
        max_tokens=x.shape[0],
        k=k,
        n=n,
        weight_E=e,
        num_topk=topk_ids.shape[1],
        activation="relu2",
        expert_chunk_size=expert_chunk_size,
        device=device,
        routing_layout=tp_moe_bf16._compact_relu2_workspace_sizing(
            num_tokens=x.shape[0],
            num_topk=topk_ids.shape[1],
            weight_E=e,
        ),
    )
    route = tp_moe_bf16._prepare_compact_route_state(
        workspace,
        topk_ids,
        topk_weights,
        route_kind="compact",
        build_fp4_metadata=True,
    )

    assert route.token_map is not None
    assert route.token_weights is not None
    bucket_rows = 2
    bucket_capacity = route.routed_rows // bucket_rows
    compact_expert_ids = workspace.compact_bucket_expert_ids[:bucket_capacity]
    compact_token_map = workspace.compact_bucket_token_map[:bucket_capacity, :bucket_rows]
    compact_token_weights = workspace.compact_bucket_token_weights[
        :bucket_capacity, :bucket_rows
    ]

    build_bucketed_compact_route(
        route.row_counts,
        route.kernel_weight_expert_ids,
        route.token_map,
        route.token_weights,
        bucket_rows,
        compact_expert_ids,
        compact_token_map,
        compact_token_weights,
    )

    row_counts_cpu = route.row_counts.cpu()
    expected_slots = (row_counts_cpu == bucket_rows).nonzero(as_tuple=False).flatten()
    expected_expert_ids = torch.full_like(compact_expert_ids.cpu(), -1)
    expected_token_map = torch.full_like(compact_token_map.cpu(), -1)
    expected_token_weights = torch.zeros_like(compact_token_weights.cpu())
    count = expected_slots.numel()
    if count > 0:
        expected_expert_ids[:count] = route.kernel_weight_expert_ids.cpu()[expected_slots]
        expected_token_map[:count] = route.token_map.cpu()[expected_slots, :bucket_rows]
        expected_token_weights[:count] = route.token_weights.cpu()[
            expected_slots, :bucket_rows
        ]

    torch.testing.assert_close(compact_expert_ids.cpu(), expected_expert_ids)
    torch.testing.assert_close(compact_token_map.cpu(), expected_token_map)
    torch.testing.assert_close(compact_token_weights.cpu(), expected_token_weights)


def test_bf16_triton_row_movement_matches_torch_reference() -> None:
    device = require_sm120()
    src = torch.randn((7, 16), device=device, dtype=torch.bfloat16)
    row_indices = torch.tensor([5, 2, 6, 1], device=device, dtype=torch.int64)
    gather_storage = torch.empty((1, 16, row_indices.numel()), device=device, dtype=torch.bfloat16)
    gathered = gather_storage[0].transpose(0, 1)

    gather_rows_bf16(src, row_indices, gathered)
    torch.testing.assert_close(gathered, src.index_select(0, row_indices))

    scatter_storage = torch.randn((1, 16, row_indices.numel()), device=device, dtype=torch.bfloat16)
    scatter_src = scatter_storage[0].transpose(0, 1)
    scatter_order = torch.tensor([2, 0, 3, 1], device=device, dtype=torch.int64)
    scattered = torch.empty((row_indices.numel(), 16), device=device, dtype=torch.bfloat16)
    expected = torch.empty_like(scattered)
    expected.index_copy_(0, scatter_order, scatter_src)

    permute_rows_bf16(scatter_src, scatter_order, scattered)
    torch.testing.assert_close(scattered, expected)


def test_bf16_row1_indexed_dense_default_runtime_runs_fc1_then_fc2() -> None:
    device = require_sm120()
    num_experts = 32
    hidden_size = 1024
    intermediate_size = 2688
    routed_rows = 22

    backend = MoEStaticKernelBackend(16, (128, 128), 1, activation="relu2")
    fc1_kernel, fc2_kernel, max_active_clusters = backend._get_row1_indexed_dense_runtime(
        device
    )

    routed_input = _alloc_row1_dense(
        feature_dim=hidden_size,
        routed_rows=routed_rows,
        device=device,
    )
    fc1_output = _alloc_row1_dense(
        feature_dim=intermediate_size,
        routed_rows=routed_rows,
        device=device,
    )
    fc2_output = _alloc_row1_dense(
        feature_dim=hidden_size,
        routed_rows=routed_rows,
        device=device,
    )
    routed_input.normal_()
    fc1_output.zero_()
    fc2_output.zero_()

    w1 = torch.randn(
        (num_experts, intermediate_size, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )
    direct_w1_view = w1.permute(1, 2, 0)
    direct_w2_view = w2.permute(1, 2, 0)
    expert_ids = torch.arange(routed_rows, device=device, dtype=torch.int32) % num_experts

    run_dense_bf16_expert_ids(
        fc1_kernel,
        routed_input,
        direct_w1_view,
        expert_ids,
        fc1_output,
        max_active_clusters,
        current_cuda_stream(),
    )
    run_dense_bf16_expert_ids(
        fc2_kernel,
        fc1_output,
        direct_w2_view,
        expert_ids,
        fc2_output,
        max_active_clusters,
        current_cuda_stream(),
    )
    torch.cuda.synchronize(device)

    assert torch.isfinite(fc2_output).all()
    assert float(fc2_output.abs().sum()) > 0.0


def test_bf16_fused_relu2_flat_persistent_runtime_runs_row1_direct_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    num_experts = 32
    hidden_size = 1024
    intermediate_size = 2688
    routed_rows = 22

    backend = MoEStaticKernelBackend(16, (128, 128), 1, activation="relu2")
    monkeypatch.setenv("B12X_BF16_FUSED_DIRECT_RELU2_VARIANT", "flat_persistent")
    kernel, max_active_clusters = backend._get_indexed_fused_relu2_runtime(device)

    routed_input = _alloc_row1_dense(
        feature_dim=hidden_size,
        routed_rows=routed_rows,
        device=device,
    )
    fused_output = _alloc_row1_dense(
        feature_dim=hidden_size,
        routed_rows=routed_rows,
        device=device,
    )
    w1 = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        (num_experts, intermediate_size, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    direct_w1_view = w1.permute(1, 2, 0)
    direct_w2_view = w2.permute(1, 2, 0)
    expert_ids = torch.arange(routed_rows, device=device, dtype=torch.int32) % num_experts

    run_fused_relu2_bf16_expert_ids(
        kernel,
        routed_input,
        direct_w1_view,
        direct_w2_view,
        expert_ids,
        fused_output,
        max_active_clusters,
        current_cuda_stream(),
    )
    torch.cuda.synchronize(device)

    assert torch.isfinite(fused_output).all()
    assert float(fused_output.abs().sum()) > 0.0


def test_bf16_direct_weight_view_preserves_valid_cutlass_layout() -> None:
    device = require_sm120()
    w = torch.empty((4, 8, 16), device=device, dtype=torch.bfloat16)
    direct_w_view = w.permute(1, 2, 0)

    assert (
        LayoutEnum.from_tensor(_to_dense_kernel_tensor(direct_w_view))
        is LayoutEnum.ROW_MAJOR
    )
    with pytest.raises(ValueError, match="Invalid leading dimension"):
        LayoutEnum.from_tensor(_to_dense_kernel_tensor(direct_w_view.contiguous()))


def test_bf16_direct_runtime_can_mix_indexed_and_row1_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    backend = MoEStaticKernelBackend(16, (128, 128), 1, activation="relu2")

    monkeypatch.setattr(
        "b12x.moe.fused.bf16.static._ENABLE_ROW1_GRID_INDEXED_DENSE",
        True,
    )
    monkeypatch.setenv("B12X_BF16_ENABLE_ROW1_GRID_INDEXED_FC1", "0")
    monkeypatch.setenv("B12X_BF16_ENABLE_ROW1_GRID_INDEXED_FC2", "1")

    fc1_kernel, fc2_kernel, _ = backend._get_direct_expert_indexed_dense_runtime(device)
    assert isinstance(fc1_kernel, ExpertIndexedDenseGemmKernel)
    assert not isinstance(fc1_kernel, ExpertIndexedDenseRow1GridKernel)
    assert isinstance(fc2_kernel, ExpertIndexedDenseRow1GridKernel)

    monkeypatch.setenv("B12X_BF16_ENABLE_ROW1_GRID_INDEXED_FC1", "1")
    monkeypatch.setenv("B12X_BF16_ENABLE_ROW1_GRID_INDEXED_FC2", "0")
    fc1_kernel, fc2_kernel, _ = backend._get_direct_expert_indexed_dense_runtime(device)
    assert isinstance(fc1_kernel, ExpertIndexedDenseRow1GridKernel)
    assert isinstance(fc2_kernel, ExpertIndexedDenseGemmKernel)
    assert not isinstance(fc2_kernel, ExpertIndexedDenseRow1GridKernel)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_bf16_relu2_nemotron_served_bs_regime_does_not_build_routing_layout(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_nemotron_case(
        device=device,
        m=m,
    )

    monkeypatch.delenv("B12X_BF16_BACKEND", raising=False)
    _assert_legacy_routing_removed()
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert output.shape == x.shape


def test_bf16_relu2_static_compact_path_does_not_build_routing_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        topk=22,
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_COMPACT_RELU2", "1")
    _assert_legacy_routing_removed()
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert output.shape == x.shape


def test_bf16_relu2_static_compact_path_uses_routed_pair_capacity_not_token_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=128,
        topk=2,
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_COMPACT_RELU2", "1")
    _assert_legacy_routing_removed()
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert output.shape == x.shape


def test_bf16_relu2_static_compact_path_matches_reference_nemotron_bs8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_nemotron_case(
        device=device,
        m=8,
    )
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation="relu2",
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_COMPACT_RELU2", "1")
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    _assert_bf16_close(
        metrics,
        "relu2/static-compact-nemotron-bs8",
        max_abs=0.26,
        rmse=0.04,
        cos=0.99998,
    )


def test_bf16_relu2_static_compact_path_matches_reference_with_duplicate_experts_nemotron_bs8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        topk=22,
    )
    repeated_pattern = torch.arange(e, device=device, dtype=torch.int32).repeat(
        (topk_ids.shape[1] + e - 1) // e
    )[: topk_ids.shape[1]]
    repeated_ids = repeated_pattern.repeat(x.shape[0], 1)
    repeated_weights = torch.softmax(
        torch.randn_like(topk_weights, dtype=torch.float32),
        dim=-1,
    )
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        repeated_ids,
        repeated_weights,
        activation="relu2",
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_COMPACT_RELU2", "1")
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            repeated_weights,
            repeated_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    _assert_bf16_close(
        metrics,
        "relu2/static-compact-duplicates-nemotron-bs8",
        max_abs=0.26,
        rmse=0.04,
        cos=0.99998,
    )


def test_bf16_relu2_static_compact_path_falls_back_to_direct_on_guaranteed_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        topk=22,
    )

    backend_cls = type(
        tp_moe_bf16._resolve_backend(
            implementation="static",
            activation="relu2",
            num_topk=topk_ids.shape[1],
        )
    )
    orig_bucketed = backend_cls._run_bucketed_compact_static_route
    orig_direct = backend_cls._run_compact_direct_route
    bucketed_call_count = 0
    direct_call_count = 0

    def _wrapped_bucketed(self, *args, **kwargs):
        nonlocal bucketed_call_count
        bucketed_call_count += 1
        return orig_bucketed(self, *args, **kwargs)

    def _wrapped_direct(self, *args, **kwargs):
        nonlocal direct_call_count
        direct_call_count += 1
        return orig_direct(self, *args, **kwargs)

    monkeypatch.setattr(
        backend_cls,
        "_run_bucketed_compact_static_route",
        _wrapped_bucketed,
    )
    monkeypatch.setattr(
        backend_cls,
        "_run_compact_direct_route",
        _wrapped_direct,
    )
    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_COMPACT_RELU2", "1")
    monkeypatch.setenv("B12X_BF16_ENABLE_BUCKETED_COMPACT_RELU2_STATIC", "1")
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert output.shape == x.shape
    assert bucketed_call_count == 0
    assert direct_call_count > 0


def test_bf16_relu2_static_compact_graph_replay_tracks_inplace_route_updates_nemotron_bs8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_nemotron_case(
        device=device,
        m=8,
    )
    pool = allocate_tp_moe_bf16_workspace_pool()
    output = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_COMPACT_RELU2", "1")
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            output=output,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            b12x_moe_bf16(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                workspace=pool,
                output=output,
                activation="relu2",
            )

        x.normal_()
        new_logits = torch.randn_like(topk_weights)
        topk_weights.copy_(torch.softmax(new_logits, dim=-1))
        topk_ids.copy_(
            _sample_unique_topk_ids(
                device=device,
                num_tokens=x.shape[0],
                num_experts=e,
                topk=topk_ids.shape[1],
            )
        )
        graph.replay()
        torch.cuda.synchronize(device)

        eager = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            output=torch.empty_like(x),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert torch.equal(torch.isnan(output), torch.isnan(eager))
    diff = (output.float() - eager.float()).abs()
    assert diff.max().item() <= 1.0 / 1024.0


def test_bf16_relu2_static_compact_bucketed_path_matches_reference_nemotron_bs8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_nemotron_case(
        device=device,
        m=8,
    )
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation="relu2",
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_COMPACT_RELU2", "1")
    monkeypatch.setenv("B12X_BF16_ENABLE_BUCKETED_COMPACT_RELU2_STATIC", "1")
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    _assert_bf16_close(
        metrics,
        "relu2/static-compact-bucketed-nemotron-bs8",
        max_abs=0.26,
        rmse=0.04,
        cos=0.99998,
    )


def test_bf16_relu2_static_compact_bucketed_graph_replay_tracks_inplace_route_updates_nemotron_bs8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_nemotron_case(
        device=device,
        m=8,
    )
    pool = allocate_tp_moe_bf16_workspace_pool()
    output = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_COMPACT_RELU2", "1")
    monkeypatch.setenv("B12X_BF16_ENABLE_BUCKETED_COMPACT_RELU2_STATIC", "1")
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            output=output,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            b12x_moe_bf16(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                workspace=pool,
                output=output,
                activation="relu2",
            )

        x.normal_()
        new_logits = torch.randn_like(topk_weights)
        topk_weights.copy_(torch.softmax(new_logits, dim=-1))
        topk_ids.copy_(
            _sample_unique_topk_ids(
                device=device,
                num_tokens=x.shape[0],
                num_experts=e,
                topk=topk_ids.shape[1],
            )
        )
        graph.replay()
        torch.cuda.synchronize(device)

        eager = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            output=torch.empty_like(x),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert torch.equal(torch.isnan(output), torch.isnan(eager))
    diff = (output.float() - eager.float()).abs()
    assert diff.max().item() <= 1.0 / 1024.0


def test_bf16_relu2_static_fp4_shaped_path_does_not_build_routing_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        e=64,
        topk=22,
    )
    topk_ids.copy_(
        _sample_unique_topk_ids(
            device=device,
            num_tokens=x.shape[0],
            num_experts=e,
            topk=topk_ids.shape[1],
        )
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_FP4_SHAPED_RELU2", "1")
    _assert_legacy_routing_removed()
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert output.shape == x.shape


def test_bf16_relu2_static_fp4_shaped_default_path_uses_backend_compact_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        e=64,
        topk=22,
    )
    topk_ids.copy_(
        _sample_unique_topk_ids(
            device=device,
            num_tokens=x.shape[0],
            num_experts=e,
            topk=topk_ids.shape[1],
        )
    )

    run_calls = 0
    backend = tp_moe_bf16._resolve_backend(
        implementation="static",
        activation="relu2",
        num_topk=topk_ids.shape[1],
    )
    backend_cls = type(backend)
    orig_run = backend_cls.run_compact_route

    def _wrapped_run(self, *args, **kwargs):
        nonlocal run_calls
        run_calls += 1
        return orig_run(self, *args, **kwargs)

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_FP4_SHAPED_RELU2", "1")
    _assert_legacy_routing_removed()
    monkeypatch.setattr(backend_cls, "run_compact_route", _wrapped_run)
    workspace = allocate_tp_moe_bf16_workspace_pool()
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=workspace,
            activation="relu2",
        )
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=workspace,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert output.shape == x.shape
    assert run_calls == 2


def test_bf16_relu2_static_fp4_shaped_route_splits_bs8_into_fixed_chunks() -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        e=64,
        topk=22,
    )
    topk_ids.copy_(
        _sample_unique_topk_ids(
            device=device,
            num_tokens=x.shape[0],
            num_experts=e,
            topk=topk_ids.shape[1],
        )
    )
    backend = tp_moe_bf16._resolve_backend(
        implementation="static",
        activation="relu2",
        num_topk=topk_ids.shape[1],
    )
    expert_chunk_size = tp_moe_bf16._expert_chunk_size(
        backend,
        num_tokens=x.shape[0],
        num_topk=topk_ids.shape[1],
    )
    workspace = tp_moe_bf16._resolve_workspace(
        None,
        implementation="static",
        max_tokens=x.shape[0],
        k=k,
        n=n,
        weight_E=e,
        num_topk=topk_ids.shape[1],
        activation="relu2",
        expert_chunk_size=expert_chunk_size,
        device=device,
        routing_layout=tp_moe_bf16._fp4_shaped_static_workspace_sizing(
            num_tokens=x.shape[0],
            num_topk=topk_ids.shape[1],
            weight_E=e,
        ),
    )
    route = tp_moe_bf16._prepare_fp4_shaped_relu2_route_layout(
        workspace,
        topk_ids,
        topk_weights,
    )
    chunk_ranges = backend._compact_route_chunk_ranges(
        route=route,
        expert_chunk_size=expert_chunk_size,
    )

    assert int(route.active_expert_count.item()) == 64
    assert chunk_ranges == [(0, 64)]


def test_bf16_relu2_static_fp4_shaped_route_builds_fp4_metadata() -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        e=128,
        topk=22,
    )
    topk_ids.copy_(
        _sample_unique_topk_ids(
            device=device,
            num_tokens=x.shape[0],
            num_experts=e,
            topk=topk_ids.shape[1],
        )
    )
    backend = tp_moe_bf16._resolve_backend(
        implementation="static",
        activation="relu2",
        num_topk=topk_ids.shape[1],
    )
    expert_chunk_size = tp_moe_bf16._expert_chunk_size(
        backend,
        num_tokens=x.shape[0],
        num_topk=topk_ids.shape[1],
    )
    workspace = tp_moe_bf16._resolve_workspace(
        None,
        implementation="static",
        max_tokens=x.shape[0],
        k=k,
        n=n,
        weight_E=e,
        num_topk=topk_ids.shape[1],
        activation="relu2",
        expert_chunk_size=expert_chunk_size,
        device=device,
        routing_layout=tp_moe_bf16._fp4_shaped_static_workspace_sizing(
            num_tokens=x.shape[0],
            num_topk=topk_ids.shape[1],
            weight_E=e,
        ),
    )
    route = tp_moe_bf16._prepare_fp4_shaped_relu2_route_layout(
        workspace,
        topk_ids,
        topk_weights,
    )

    active_expert_count = int(route.active_expert_count.item())
    flat_topk_ids = topk_ids.reshape(-1)
    flat_topk_weights = topk_weights.reshape(-1).float()
    flat_token_indices = torch.arange(
        x.shape[0], device=device, dtype=torch.int32
    ).repeat_interleave(topk_ids.shape[1])

    expected_global_to_local = torch.full((e,), -1, device=device, dtype=torch.int32)
    compact_capacity = route.weight_expert_ids.shape[0]
    assert compact_capacity == min(e, route.routed_rows)
    expected_token_map = torch.full(
        (compact_capacity, x.shape[0]),
        -1,
        device=device,
        dtype=torch.int32,
    )
    expected_token_weights = torch.zeros(
        (compact_capacity, x.shape[0]),
        device=device,
        dtype=torch.float32,
    )

    for pair_idx in range(route.routed_rows):
        local_expert = int(route.compact_topk_ids[pair_idx].item())
        route_row = int(route.route_row_indices[pair_idx].item())
        token_idx = int(flat_token_indices[pair_idx].item())
        expected_token_map[local_expert, route_row] = token_idx
        expected_token_weights[local_expert, route_row] = float(
            flat_topk_weights[pair_idx].item()
        )

    for local_expert in range(active_expert_count):
        global_expert = int(route.weight_expert_ids[local_expert].item())
        expected_global_to_local[global_expert] = local_expert

    torch.testing.assert_close(route.global_to_local_expert, expected_global_to_local)
    torch.testing.assert_close(route.token_map, expected_token_map)
    torch.testing.assert_close(route.token_weights, expected_token_weights)


def test_bf16_relu2_static_fp4_shaped_path_matches_reference_nemotron_bs8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        e=64,
        topk=22,
    )
    topk_ids.copy_(
        _sample_unique_topk_ids(
            device=device,
            num_tokens=x.shape[0],
            num_experts=e,
            topk=topk_ids.shape[1],
        )
    )
    reference = moe_reference_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation="relu2",
    )

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_FP4_SHAPED_RELU2", "1")
    clear_tp_moe_bf16_caches()
    try:
        output = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, reference)
    _assert_bf16_close(
        metrics,
        "relu2/static-fp4-shaped-nemotron-bs8",
        max_abs=0.26,
        rmse=0.04,
        cos=0.99998,
    )


def test_bf16_relu2_static_fp4_shaped_graph_replay_tracks_inplace_route_updates_nemotron_bs8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        e=64,
        topk=22,
    )
    topk_ids.copy_(
        _sample_unique_topk_ids(
            device=device,
            num_tokens=x.shape[0],
            num_experts=e,
            topk=topk_ids.shape[1],
        )
    )
    pool = allocate_tp_moe_bf16_workspace_pool()
    output = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_FP4_SHAPED_RELU2", "1")
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            output=output,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            b12x_moe_bf16(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                workspace=pool,
                output=output,
                activation="relu2",
            )

        x.normal_()
        new_logits = torch.randn_like(topk_weights)
        topk_weights.copy_(torch.softmax(new_logits, dim=-1))
        topk_ids.copy_(
            _sample_unique_topk_ids(
                device=device,
                num_tokens=x.shape[0],
                num_experts=e,
                topk=topk_ids.shape[1],
            )
        )
        graph.replay()
        torch.cuda.synchronize(device)

        eager = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            output=torch.empty_like(x),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert torch.equal(torch.isnan(output), torch.isnan(eager))
    diff = (output.float() - eager.float()).abs()
    assert diff.max().item() == 0.0


def test_bf16_relu2_static_fp4_shaped_graph_replay_tracks_inplace_route_updates_nemotron_bs8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=8,
        e=64,
        topk=22,
    )
    topk_ids.copy_(
        _sample_unique_topk_ids(
            device=device,
            num_tokens=x.shape[0],
            num_experts=e,
            topk=topk_ids.shape[1],
        )
    )
    pool = allocate_tp_moe_bf16_workspace_pool()
    output = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()

    monkeypatch.setenv("B12X_BF16_BACKEND", "static")
    monkeypatch.setenv("B12X_BF16_ENABLE_FP4_SHAPED_RELU2", "1")
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            output=output,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            b12x_moe_bf16(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                workspace=pool,
                output=output,
                activation="relu2",
            )

        x.normal_()
        new_logits = torch.randn_like(topk_weights)
        topk_weights.copy_(torch.softmax(new_logits, dim=-1))
        topk_ids.copy_(
            _sample_unique_topk_ids(
                device=device,
                num_tokens=x.shape[0],
                num_experts=e,
                topk=topk_ids.shape[1],
            )
        )
        graph.replay()
        torch.cuda.synchronize(device)

        eager = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            output=torch.empty_like(x),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    assert torch.equal(torch.isnan(output), torch.isnan(eager))
    diff = (output.float() - eager.float()).abs()
    assert diff.max().item() == 0.0


def test_bf16_relu2_single_token_graph_replay_tracks_inplace_route_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=1,
        topk=4,
    )
    pool = allocate_tp_moe_bf16_workspace_pool()
    output = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()

    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            output=output,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            b12x_moe_bf16(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                workspace=pool,
                output=output,
                activation="relu2",
            )

        x.normal_()
        new_logits = torch.randn_like(topk_weights)
        topk_weights.copy_(torch.softmax(new_logits, dim=-1))
        topk_ids.copy_(torch.randint(0, e, topk_ids.shape, device=device, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize(device)

        eager = b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=allocate_tp_moe_bf16_workspace_pool(),
            output=torch.empty_like(x),
            activation="relu2",
        )
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()

    metrics = compare_to_reference(output, eager)
    assert metrics.max_abs == 0.0, metrics
    assert metrics.rmse == 0.0, metrics


def test_bf16_relu2_graph_replay_does_not_build_routing_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    x, w1, w2, topk_weights, topk_ids, e, k, n = _make_case(
        device=device,
        activation="relu2",
        m=16,
        topk=4,
    )
    pool = allocate_tp_moe_bf16_workspace_pool()
    output = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()

    monkeypatch.setenv("B12X_BF16_BACKEND", "micro")
    _assert_legacy_routing_removed()
    clear_tp_moe_bf16_caches()
    try:
        b12x_moe_bf16(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            workspace=pool,
            output=output,
            activation="relu2",
        )
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            b12x_moe_bf16(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                workspace=pool,
                output=output,
                activation="relu2",
            )

        x.normal_()
        topk_weights.copy_(torch.softmax(torch.randn_like(topk_weights), dim=-1))
        topk_ids.copy_(torch.randint(0, e, topk_ids.shape, device=device, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize(device)
    finally:
        clear_tp_moe_bf16_caches()
