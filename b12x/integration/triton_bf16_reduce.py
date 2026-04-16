from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_fc2_chunk_grouped_bf16_kernel(
    fc2_chunk_ptr,
    route_weights_grouped_ptr,
    route_local_expert_slots_grouped_ptr,
    route_row_indices_grouped_ptr,
    output_ptr,
    fc2_row_stride,
    fc2_col_stride,
    fc2_expert_stride,
    route_weights_row_stride,
    route_metadata_row_stride,
    output_row_stride,
    hidden_size,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size

    weight_ptr = route_weights_grouped_ptr + token_idx * route_weights_row_stride
    expert_ptr = (
        route_local_expert_slots_grouped_ptr + token_idx * route_metadata_row_stride
    )
    row_ptr = route_row_indices_grouped_ptr + token_idx * route_metadata_row_stride

    first_row = tl.load(row_ptr)
    first_expert = tl.load(expert_ptr)
    first_weight = tl.load(weight_ptr)
    acc = tl.load(
        fc2_chunk_ptr
        + first_row * fc2_row_stride
        + col_offsets * fc2_col_stride
        + first_expert * fc2_expert_stride,
        mask=col_mask,
        other=0,
    ).to(tl.float32) * first_weight

    for route_idx in range(1, TOP_K):
        row = tl.load(row_ptr + route_idx)
        expert = tl.load(expert_ptr + route_idx)
        weight = tl.load(weight_ptr + route_idx)
        vals = tl.load(
            fc2_chunk_ptr
            + row * fc2_row_stride
            + col_offsets * fc2_col_stride
            + expert * fc2_expert_stride,
            mask=col_mask,
            other=0,
        ).to(tl.float32) * weight
        acc += vals

    tl.store(
        output_ptr + token_idx * output_row_stride + col_offsets,
        acc.to(tl.bfloat16),
        mask=col_mask,
    )


def reduce_fc2_chunk_grouped_bf16(
    fc2_chunk: torch.Tensor,
    route_weights_grouped: torch.Tensor,
    route_local_expert_slots_grouped: torch.Tensor,
    route_row_indices_grouped: torch.Tensor,
    output: torch.Tensor,
    *,
    num_topk: int,
) -> None:
    if num_topk <= 0:
        output.zero_()
        return
    if (
        not fc2_chunk.is_cuda
        or not route_weights_grouped.is_cuda
        or not route_local_expert_slots_grouped.is_cuda
        or not route_row_indices_grouped.is_cuda
        or not output.is_cuda
    ):
        raise ValueError("reduce_fc2_chunk_grouped_bf16 requires CUDA tensors")
    if fc2_chunk.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 fc2_chunk, got {fc2_chunk.dtype}")
    if route_weights_grouped.dtype != torch.float32:
        raise ValueError(
            f"expected FP32 route_weights_grouped, got {route_weights_grouped.dtype}"
        )
    if output.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 output, got {output.dtype}")
    if route_local_expert_slots_grouped.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            "expected int32/int64 route_local_expert_slots_grouped, got "
            f"{route_local_expert_slots_grouped.dtype}"
        )
    if route_row_indices_grouped.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            "expected int32/int64 route_row_indices_grouped, got "
            f"{route_row_indices_grouped.dtype}"
        )
    if (
        fc2_chunk.ndim != 3
        or route_weights_grouped.ndim != 1
        or route_local_expert_slots_grouped.ndim != 1
        or route_row_indices_grouped.ndim != 1
        or output.ndim != 2
    ):
        raise ValueError(
            "expected rank-3 fc2_chunk, rank-1 grouped route metadata, and rank-2 output"
        )
    if output.stride(1) != 1:
        raise ValueError("output must be contiguous in the hidden dimension")
    if route_weights_grouped.stride(0) != 1:
        raise ValueError("route_weights_grouped must be contiguous")
    if route_local_expert_slots_grouped.stride(0) != 1:
        raise ValueError("route_local_expert_slots_grouped must be contiguous")
    if route_row_indices_grouped.stride(0) != 1:
        raise ValueError("route_row_indices_grouped must be contiguous")
    if route_weights_grouped.numel() != output.shape[0] * num_topk:
        raise ValueError("route_weights_grouped shape does not match output rows * num_topk")
    if route_local_expert_slots_grouped.numel() != output.shape[0] * num_topk:
        raise ValueError(
            "route_local_expert_slots_grouped shape does not match output rows * num_topk"
        )
    if route_row_indices_grouped.numel() != output.shape[0] * num_topk:
        raise ValueError("route_row_indices_grouped shape does not match output rows * num_topk")

    hidden_size = output.shape[1]
    block_k = int(
        os.environ.get(
            "B12X_BF16_DIRECT_REDUCE_BLOCK_K",
            os.environ.get(
                "B12X_BF16_REDUCE_BLOCK_K",
                "64" if hidden_size <= 1024 else "128",
            ),
        )
    )
    num_warps = int(
        os.environ.get(
            "B12X_BF16_DIRECT_REDUCE_NUM_WARPS",
            os.environ.get(
                "B12X_BF16_REDUCE_NUM_WARPS",
                "2" if hidden_size <= 1024 else "4",
            ),
        )
    )

    _reduce_fc2_chunk_grouped_bf16_kernel[
        (output.shape[0], triton.cdiv(hidden_size, block_k))
    ](
        fc2_chunk,
        route_weights_grouped,
        route_local_expert_slots_grouped,
        route_row_indices_grouped,
        output,
        fc2_chunk.stride(0),
        fc2_chunk.stride(1),
        fc2_chunk.stride(2),
        num_topk,
        num_topk,
        output.stride(0),
        hidden_size,
        TOP_K=num_topk,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
