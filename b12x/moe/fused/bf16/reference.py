from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OracleMetrics:
    max_abs: float
    rmse: float
    mean_abs: float
    cos: float


@dataclass(frozen=True)
class MoERouteTrace:
    token_idx: int
    route_idx: int
    expert_idx: int
    activation: str
    router_weight: float
    x: torch.Tensor
    fc1_out: torch.Tensor | None
    gate_out: torch.Tensor | None
    up_out: torch.Tensor | None
    intermediate: torch.Tensor
    down_out: torch.Tensor
    routed_out: torch.Tensor


def compare_to_reference(actual: torch.Tensor, reference: torch.Tensor) -> OracleMetrics:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    diff = actual_fp32 - reference_fp32
    cos = F.cosine_similarity(
        actual_fp32.reshape(actual_fp32.shape[0], -1),
        reference_fp32.reshape(reference_fp32.shape[0], -1),
        dim=1,
    ).mean().item()
    return OracleMetrics(
        max_abs=diff.abs().max().item(),
        rmse=diff.square().mean().sqrt().item(),
        mean_abs=diff.abs().mean().item(),
        cos=cos,
    )


def _validate_reference_inputs(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
) -> tuple[int, int, int]:
    if activation not in {"silu", "relu2"}:
        raise ValueError(f"unsupported activation {activation!r}")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"expected x.dtype=torch.bfloat16, got {x.dtype}")
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 expert weights, got w1={w1.dtype}, w2={w2.dtype}")
    if x.ndim != 2:
        raise ValueError(f"expected x.ndim == 2, got {x.ndim}")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError(f"expected rank-3 weights, got w1.ndim={w1.ndim}, w2.ndim={w2.ndim}")
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("expected topk_ids/topk_weights to be rank-2")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"topk_ids and topk_weights must have the same shape, got "
            f"{topk_ids.shape} vs {topk_weights.shape}"
        )

    m, k = x.shape
    experts, w1_rows, w1_k = w1.shape
    w2_experts, output_k, intermediate_n = w2.shape
    if w1_k != k:
        raise ValueError(f"expected w1.shape[2] == {k}, got {w1_k}")
    if w2_experts != experts:
        raise ValueError(f"expected w2.shape[0] == {experts}, got {w2_experts}")
    if topk_ids.shape[0] != m:
        raise ValueError(f"expected topk_ids.shape[0] == {m}, got {topk_ids.shape[0]}")
    if output_k != k:
        raise ValueError(f"expected w2.shape[1] == hidden_size {k}, got {output_k}")
    if intermediate_n * (2 if activation == "silu" else 1) != w1_rows:
        raise ValueError(
            f"expected w2.shape[2] == intermediate_size {w1_rows // (2 if activation == 'silu' else 1)}, "
            f"got {intermediate_n}"
        )
    expected_w1_rows = 2 * intermediate_n if activation == "silu" else intermediate_n
    if w1_rows != expected_w1_rows:
        raise ValueError(
            f"expected w1.shape[1] == {expected_w1_rows} for activation {activation!r}, "
            f"got {w1_rows}"
        )
    return experts, output_k, intermediate_n


def _compute_intermediate(
    x_row: torch.Tensor,
    w1_expert: torch.Tensor,
    *,
    activation: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    if activation == "silu":
        half = w1_expert.shape[0] // 2
        fc1 = torch.matmul(w1_expert, x_row).to(torch.bfloat16)
        up = fc1[:half]
        gate = fc1[half:]
        intermediate = (
            torch.sigmoid(gate.float()) * gate.float() * up.float()
        ).to(torch.bfloat16)
        return None, gate, up, intermediate

    fc1 = torch.matmul(w1_expert, x_row).to(torch.bfloat16)
    intermediate = torch.square(torch.relu(fc1)).to(torch.bfloat16)
    return fc1, None, None, intermediate


def trace_moe_reference_bf16_route(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    token_idx: int,
    route_idx: int,
    activation: str = "silu",
) -> MoERouteTrace:
    _validate_reference_inputs(x, w1, w2, topk_ids, topk_weights, activation=activation)
    if token_idx < 0 or token_idx >= x.shape[0]:
        raise IndexError(f"token_idx {token_idx} is out of range for batch {x.shape[0]}")
    if route_idx < 0 or route_idx >= topk_ids.shape[1]:
        raise IndexError(f"route_idx {route_idx} is out of range for top_k {topk_ids.shape[1]}")

    expert_idx = int(topk_ids[token_idx, route_idx].item())
    router_weight = float(topk_weights[token_idx, route_idx].item())
    x_row = x[token_idx]
    fc1_out, gate_out, up_out, intermediate = _compute_intermediate(
        x_row,
        w1[expert_idx],
        activation=activation,
    )
    down_out = torch.matmul(w2[expert_idx], intermediate).to(torch.bfloat16)
    routed_out = (router_weight * down_out.float()).to(torch.bfloat16)
    return MoERouteTrace(
        token_idx=token_idx,
        route_idx=route_idx,
        expert_idx=expert_idx,
        activation=activation,
        router_weight=router_weight,
        x=x_row,
        fc1_out=fc1_out,
        gate_out=gate_out,
        up_out=up_out,
        intermediate=intermediate,
        down_out=down_out,
        routed_out=routed_out,
    )


def moe_reference_bf16(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    experts, output_k, intermediate_n = _validate_reference_inputs(
        x, w1, w2, topk_ids, topk_weights, activation=activation
    )
    m = x.shape[0]
    top_k = topk_ids.shape[1]
    routed_rows = m * top_k

    flat_experts = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts.index_select(0, order)
    sorted_token_indices = torch.arange(
        m, device=x.device, dtype=torch.int64
    ).repeat_interleave(top_k).index_select(0, order)
    sorted_weights = topk_weights.reshape(-1).float().index_select(0, order)
    routed_input = x.index_select(0, sorted_token_indices)
    routed_output_sorted = torch.empty(
        routed_rows, output_k, dtype=torch.bfloat16, device=x.device
    )

    row_counts = torch.bincount(sorted_experts, minlength=experts)
    offsets = row_counts.cumsum(0)
    start = 0
    for expert_idx in range(experts):
        rows = int(row_counts[expert_idx].item())
        if rows == 0:
            continue
        x_rows = routed_input[start : start + rows]
        w1_expert = w1[expert_idx]
        fc1 = torch.matmul(x_rows, w1_expert.transpose(0, 1)).to(torch.bfloat16)
        if activation == "silu":
            up = fc1[:, :intermediate_n]
            gate = fc1[:, intermediate_n:]
            intermediate = (
                torch.sigmoid(gate.float()) * gate.float() * up.float()
            ).to(torch.bfloat16)
        else:
            intermediate = torch.square(torch.relu(fc1)).to(torch.bfloat16)
        down = torch.matmul(intermediate, w2[expert_idx].transpose(0, 1)).to(
            torch.bfloat16
        )
        routed_output_sorted[start : start + rows] = (
            down.float() * sorted_weights[start : start + rows, None]
        ).to(torch.bfloat16)
        start = int(offsets[expert_idx].item())

    routed_output = torch.empty_like(routed_output_sorted)
    routed_output.index_copy_(0, order, routed_output_sorted)
    route_outputs = routed_output.view(m, top_k, output_k)

    output = torch.zeros(m, output_k, dtype=torch.bfloat16, device=x.device)
    for route_idx in range(top_k):
        output = (output.float() + route_outputs[:, route_idx, :].float()).to(
            torch.bfloat16
        )
    return output
