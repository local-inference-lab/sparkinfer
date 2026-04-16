from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_rows_bf16_kernel(
    src_ptr,
    row_indices_ptr,
    out_ptr,
    src_row_stride,
    src_col_stride,
    out_row_stride,
    out_col_stride,
    hidden_size,
    BLOCK_K: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size
    src_row_idx = tl.load(row_indices_ptr + row_idx).to(tl.int64)
    vals = tl.load(
        src_ptr + src_row_idx * src_row_stride + col_offsets * src_col_stride,
        mask=col_mask,
        other=0,
    )
    tl.store(
        out_ptr + row_idx * out_row_stride + col_offsets * out_col_stride,
        vals,
        mask=col_mask,
    )


@triton.jit
def _permute_rows_bf16_kernel(
    src_ptr,
    dst_row_indices_ptr,
    out_ptr,
    src_row_stride,
    src_col_stride,
    out_row_stride,
    out_col_stride,
    hidden_size,
    BLOCK_K: tl.constexpr,
):
    src_row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size
    dst_row_idx = tl.load(dst_row_indices_ptr + src_row_idx).to(tl.int64)
    vals = tl.load(
        src_ptr + src_row_idx * src_row_stride + col_offsets * src_col_stride,
        mask=col_mask,
        other=0,
    )
    tl.store(
        out_ptr + dst_row_idx * out_row_stride + col_offsets * out_col_stride,
        vals,
        mask=col_mask,
    )


@triton.jit
def _scatter_routed_input_grouped_bf16_kernel(
    a_ptr,
    flat_token_indices_ptr,
    route_local_expert_slots_ptr,
    route_row_indices_ptr,
    routed_chunk_ptr,
    a_row_stride,
    routed_row_stride,
    routed_col_stride,
    routed_expert_stride,
    hidden_size,
    BLOCK_K: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size

    token_idx = tl.load(flat_token_indices_ptr + pair_idx).to(tl.int64)
    local_expert_idx = tl.load(route_local_expert_slots_ptr + pair_idx).to(tl.int64)
    route_row_idx = tl.load(route_row_indices_ptr + pair_idx).to(tl.int64)

    vals = tl.load(
        a_ptr + token_idx * a_row_stride + col_offsets,
        mask=col_mask,
        other=0,
    )
    tl.store(
        routed_chunk_ptr
        + route_row_idx * routed_row_stride
        + col_offsets * routed_col_stride
        + local_expert_idx * routed_expert_stride,
        vals,
        mask=col_mask,
    )


@triton.jit
def _scatter_add_grouped_fc2_bf16_kernel(
    fc2_chunk_ptr,
    routed_weights_ptr,
    flat_token_indices_ptr,
    route_local_expert_slots_ptr,
    route_row_indices_ptr,
    accum_output_ptr,
    fc2_row_stride,
    fc2_col_stride,
    fc2_expert_stride,
    accum_output_row_stride,
    hidden_size,
    ROUND_WEIGHTED_TO_BF16: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size

    token_idx = tl.load(flat_token_indices_ptr + pair_idx).to(tl.int64)
    local_expert_idx = tl.load(route_local_expert_slots_ptr + pair_idx).to(tl.int64)
    route_row_idx = tl.load(route_row_indices_ptr + pair_idx).to(tl.int64)
    routed_weight = tl.load(routed_weights_ptr + pair_idx).to(tl.float32)

    vals = tl.load(
        fc2_chunk_ptr
        + route_row_idx * fc2_row_stride
        + col_offsets * fc2_col_stride
        + local_expert_idx * fc2_expert_stride,
        mask=col_mask,
        other=0,
    ).to(tl.float32) * routed_weight
    if ROUND_WEIGHTED_TO_BF16:
        vals = vals.to(tl.bfloat16).to(tl.float32)
    tl.atomic_add(
        accum_output_ptr + token_idx * accum_output_row_stride + col_offsets,
        vals,
        mask=col_mask,
    )


@triton.jit
def _scatter_routed_input_compact_chunk_bf16_kernel(
    a_ptr,
    flat_token_indices_ptr,
    compact_topk_ids_ptr,
    route_row_indices_ptr,
    routed_chunk_ptr,
    expert_begin,
    expert_end,
    a_row_stride,
    routed_row_stride,
    routed_col_stride,
    routed_expert_stride,
    hidden_size,
    BLOCK_K: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size

    compact_id = tl.load(compact_topk_ids_ptr + pair_idx).to(tl.int32)
    valid = (compact_id >= expert_begin) & (compact_id < expert_end)
    if not valid:
        return

    token_idx = tl.load(flat_token_indices_ptr + pair_idx).to(tl.int64)
    route_row_idx = tl.load(route_row_indices_ptr + pair_idx).to(tl.int64)
    local_expert_idx = (compact_id - expert_begin).to(tl.int64)

    vals = tl.load(
        a_ptr + token_idx * a_row_stride + col_offsets,
        mask=col_mask,
        other=0,
    )
    tl.store(
        routed_chunk_ptr
        + route_row_idx * routed_row_stride
        + col_offsets * routed_col_stride
        + local_expert_idx * routed_expert_stride,
        vals,
        mask=col_mask,
    )


@triton.jit
def _scatter_add_compact_chunk_fc2_bf16_kernel(
    fc2_chunk_ptr,
    routed_weights_ptr,
    flat_token_indices_ptr,
    compact_topk_ids_ptr,
    route_row_indices_ptr,
    accum_output_ptr,
    expert_begin,
    expert_end,
    fc2_row_stride,
    fc2_col_stride,
    fc2_expert_stride,
    accum_output_row_stride,
    hidden_size,
    ROUND_WEIGHTED_TO_BF16: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size

    compact_id = tl.load(compact_topk_ids_ptr + pair_idx).to(tl.int32)
    valid = (compact_id >= expert_begin) & (compact_id < expert_end)
    if not valid:
        return

    token_idx = tl.load(flat_token_indices_ptr + pair_idx).to(tl.int64)
    route_row_idx = tl.load(route_row_indices_ptr + pair_idx).to(tl.int64)
    local_expert_idx = (compact_id - expert_begin).to(tl.int64)
    routed_weight = tl.load(routed_weights_ptr + pair_idx).to(tl.float32)

    vals = tl.load(
        fc2_chunk_ptr
        + route_row_idx * fc2_row_stride
        + col_offsets * fc2_col_stride
        + local_expert_idx * fc2_expert_stride,
        mask=col_mask,
        other=0,
    ).to(tl.float32) * routed_weight
    if ROUND_WEIGHTED_TO_BF16:
        vals = vals.to(tl.bfloat16).to(tl.float32)
    tl.atomic_add(
        accum_output_ptr + token_idx * accum_output_row_stride + col_offsets,
        vals,
        mask=col_mask,
    )


@triton.jit
def _reduce_compact_chunk_fc2_by_token_bf16_kernel(
    fc2_chunk_ptr,
    routed_weights_ptr,
    compact_topk_ids_ptr,
    route_row_indices_ptr,
    accum_output_ptr,
    expert_begin,
    expert_end,
    fc2_row_stride,
    fc2_col_stride,
    fc2_expert_stride,
    accum_output_row_stride,
    hidden_size,
    ROUND_WEIGHTED_TO_BF16: tl.constexpr,
    NUM_TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size
    acc = tl.load(
        accum_output_ptr + token_idx * accum_output_row_stride + col_offsets,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    token_pair_base = token_idx * NUM_TOPK
    for route_idx in tl.static_range(NUM_TOPK):
        pair_idx = token_pair_base + route_idx
        compact_id = tl.load(compact_topk_ids_ptr + pair_idx).to(tl.int32)
        valid = (compact_id >= expert_begin) & (compact_id < expert_end)
        if valid:
            route_row_idx = tl.load(route_row_indices_ptr + pair_idx).to(tl.int64)
            local_expert_idx = (compact_id - expert_begin).to(tl.int64)
            routed_weight = tl.load(routed_weights_ptr + pair_idx).to(tl.float32)
            vals = tl.load(
                fc2_chunk_ptr
                + route_row_idx * fc2_row_stride
                + col_offsets * fc2_col_stride
                + local_expert_idx * fc2_expert_stride,
                mask=col_mask,
                other=0,
            ).to(tl.float32) * routed_weight
            if ROUND_WEIGHTED_TO_BF16:
                vals = vals.to(tl.bfloat16).to(tl.float32)
            acc += vals

    tl.store(
        accum_output_ptr + token_idx * accum_output_row_stride + col_offsets,
        acc,
        mask=col_mask,
    )


@triton.jit
def _scatter_routed_input_token_map_bf16_kernel(
    a_ptr,
    token_map_ptr,
    routed_chunk_ptr,
    token_map_expert_stride,
    token_map_row_stride,
    a_row_stride,
    routed_row_stride,
    routed_col_stride,
    routed_expert_stride,
    max_rows,
    hidden_size,
    BLOCK_K: tl.constexpr,
):
    pair_slot = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    local_expert_idx = pair_slot // max_rows
    route_row_idx = pair_slot % max_rows
    token_idx = tl.load(
        token_map_ptr
        + local_expert_idx * token_map_expert_stride
        + route_row_idx * token_map_row_stride
    ).to(tl.int64)
    if token_idx < 0:
        return

    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size
    vals = tl.load(
        a_ptr + token_idx * a_row_stride + col_offsets,
        mask=col_mask,
        other=0,
    )
    tl.store(
        routed_chunk_ptr
        + route_row_idx * routed_row_stride
        + col_offsets * routed_col_stride
        + local_expert_idx * routed_expert_stride,
        vals,
        mask=col_mask,
    )


@triton.jit
def _scatter_add_token_map_fc2_bf16_kernel(
    fc2_chunk_ptr,
    token_map_ptr,
    token_weights_ptr,
    accum_output_ptr,
    token_map_expert_stride,
    token_map_row_stride,
    token_weights_expert_stride,
    token_weights_row_stride,
    fc2_row_stride,
    fc2_col_stride,
    fc2_expert_stride,
    accum_output_row_stride,
    max_rows,
    hidden_size,
    ROUND_WEIGHTED_TO_BF16: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_slot = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    local_expert_idx = pair_slot // max_rows
    route_row_idx = pair_slot % max_rows
    token_idx = tl.load(
        token_map_ptr
        + local_expert_idx * token_map_expert_stride
        + route_row_idx * token_map_row_stride
    ).to(tl.int64)
    if token_idx < 0:
        return

    routed_weight = tl.load(
        token_weights_ptr
        + local_expert_idx * token_weights_expert_stride
        + route_row_idx * token_weights_row_stride
    ).to(tl.float32)
    col_offsets = col_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = col_offsets < hidden_size
    vals = tl.load(
        fc2_chunk_ptr
        + route_row_idx * fc2_row_stride
        + col_offsets * fc2_col_stride
        + local_expert_idx * fc2_expert_stride,
        mask=col_mask,
        other=0,
    ).to(tl.float32) * routed_weight
    if ROUND_WEIGHTED_TO_BF16:
        vals = vals.to(tl.bfloat16).to(tl.float32)
    tl.atomic_add(
        accum_output_ptr + token_idx * accum_output_row_stride + col_offsets,
        vals,
        mask=col_mask,
    )


def scatter_routed_input_grouped_bf16(
    a: torch.Tensor,
    flat_token_indices: torch.Tensor,
    route_local_expert_slots: torch.Tensor,
    route_row_indices: torch.Tensor,
    routed_chunk: torch.Tensor,
) -> None:
    if (
        not a.is_cuda
        or not flat_token_indices.is_cuda
        or not route_local_expert_slots.is_cuda
        or not route_row_indices.is_cuda
        or not routed_chunk.is_cuda
    ):
        raise ValueError("scatter_routed_input_grouped_bf16 requires CUDA tensors")
    if a.dtype != torch.bfloat16 or routed_chunk.dtype != torch.bfloat16:
        raise ValueError(
            f"expected BF16 tensors for a/routed_chunk, got {a.dtype} and {routed_chunk.dtype}"
        )
    if flat_token_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            f"expected int32/int64 flat_token_indices, got {flat_token_indices.dtype}"
        )
    if route_local_expert_slots.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            "expected int32/int64 route_local_expert_slots, got "
            f"{route_local_expert_slots.dtype}"
        )
    if route_row_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            f"expected int32/int64 route_row_indices, got {route_row_indices.dtype}"
        )
    if a.ndim != 2 or routed_chunk.ndim != 3:
        raise ValueError(
            f"expected rank-2 a and rank-3 routed_chunk, got {a.ndim} and {routed_chunk.ndim}"
        )
    if (
        flat_token_indices.ndim != 1
        or route_local_expert_slots.ndim != 1
        or route_row_indices.ndim != 1
    ):
        raise ValueError("expected rank-1 route metadata tensors")
    routed_rows = flat_token_indices.numel()
    if (
        route_local_expert_slots.numel() != routed_rows
        or route_row_indices.numel() != routed_rows
    ):
        raise ValueError("route metadata lengths must match routed_rows")
    if flat_token_indices.stride(0) != 1:
        raise ValueError("flat_token_indices must be contiguous")
    if route_local_expert_slots.stride(0) != 1:
        raise ValueError("route_local_expert_slots must be contiguous")
    if route_row_indices.stride(0) != 1:
        raise ValueError("route_row_indices must be contiguous")
    if routed_rows == 0:
        return

    hidden_size = a.shape[1]
    block_k = int(os.environ.get("B12X_BF16_SCATTER_BLOCK_K", "64"))
    num_warps = int(os.environ.get("B12X_BF16_SCATTER_NUM_WARPS", "2"))

    _scatter_routed_input_grouped_bf16_kernel[
        (routed_rows, triton.cdiv(hidden_size, block_k))
    ](
        a,
        flat_token_indices,
        route_local_expert_slots,
        route_row_indices,
        routed_chunk,
        a.stride(0),
        routed_chunk.stride(0),
        routed_chunk.stride(1),
        routed_chunk.stride(2),
        hidden_size,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def gather_rows_bf16(
    src: torch.Tensor,
    row_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if not src.is_cuda or not row_indices.is_cuda or not out.is_cuda:
        raise ValueError("gather_rows_bf16 requires CUDA tensors")
    if src.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 src/out, got {src.dtype} and {out.dtype}")
    if row_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"expected int32/int64 row_indices, got {row_indices.dtype}")
    if src.ndim != 2 or out.ndim != 2 or row_indices.ndim != 1:
        raise ValueError(
            f"expected rank-2 src/out and rank-1 row_indices, got {src.ndim}, {out.ndim}, {row_indices.ndim}"
        )
    routed_rows = row_indices.numel()
    if out.shape[0] != routed_rows or src.shape[1] != out.shape[1]:
        raise ValueError("gather_rows_bf16 shape mismatch")
    if routed_rows == 0:
        return

    hidden_size = src.shape[1]
    block_k = int(os.environ.get("B12X_BF16_SCATTER_BLOCK_K", "64"))
    num_warps = int(os.environ.get("B12X_BF16_SCATTER_NUM_WARPS", "2"))

    _gather_rows_bf16_kernel[(routed_rows, triton.cdiv(hidden_size, block_k))](
        src,
        row_indices,
        out,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        hidden_size,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def permute_rows_bf16(
    src: torch.Tensor,
    dst_row_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if not src.is_cuda or not dst_row_indices.is_cuda or not out.is_cuda:
        raise ValueError("permute_rows_bf16 requires CUDA tensors")
    if src.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 src/out, got {src.dtype} and {out.dtype}")
    if dst_row_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            f"expected int32/int64 dst_row_indices, got {dst_row_indices.dtype}"
        )
    if src.ndim != 2 or out.ndim != 2 or dst_row_indices.ndim != 1:
        raise ValueError(
            "expected rank-2 src/out and rank-1 dst_row_indices, got "
            f"{src.ndim}, {out.ndim}, {dst_row_indices.ndim}"
        )
    routed_rows = dst_row_indices.numel()
    if src.shape[0] != routed_rows or out.shape[0] != routed_rows or src.shape[1] != out.shape[1]:
        raise ValueError("permute_rows_bf16 shape mismatch")
    if routed_rows == 0:
        return

    hidden_size = src.shape[1]
    block_k = int(os.environ.get("B12X_BF16_SCATTER_BLOCK_K", "64"))
    num_warps = int(os.environ.get("B12X_BF16_SCATTER_NUM_WARPS", "2"))

    _permute_rows_bf16_kernel[(routed_rows, triton.cdiv(hidden_size, block_k))](
        src,
        dst_row_indices,
        out,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        hidden_size,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def scatter_add_grouped_fc2_bf16(
    fc2_chunk: torch.Tensor,
    routed_weights: torch.Tensor,
    flat_token_indices: torch.Tensor,
    route_local_expert_slots: torch.Tensor,
    route_row_indices: torch.Tensor,
    accum_output: torch.Tensor,
    *,
    round_weighted_to_bf16: bool,
) -> None:
    if (
        not fc2_chunk.is_cuda
        or not routed_weights.is_cuda
        or not flat_token_indices.is_cuda
        or not route_local_expert_slots.is_cuda
        or not route_row_indices.is_cuda
        or not accum_output.is_cuda
    ):
        raise ValueError("scatter_add_grouped_fc2_bf16 requires CUDA tensors")
    if fc2_chunk.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 fc2_chunk, got {fc2_chunk.dtype}")
    if routed_weights.dtype != torch.float32:
        raise ValueError(f"expected FP32 routed_weights, got {routed_weights.dtype}")
    if accum_output.dtype != torch.float32:
        raise ValueError(f"expected FP32 accum_output, got {accum_output.dtype}")
    if flat_token_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            f"expected int32/int64 flat_token_indices, got {flat_token_indices.dtype}"
        )
    if route_local_expert_slots.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            "expected int32/int64 route_local_expert_slots, got "
            f"{route_local_expert_slots.dtype}"
        )
    if route_row_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            f"expected int32/int64 route_row_indices, got {route_row_indices.dtype}"
        )
    if fc2_chunk.ndim != 3 or accum_output.ndim != 2:
        raise ValueError(
            f"expected rank-3 fc2_chunk and rank-2 accum_output, got {fc2_chunk.ndim} and {accum_output.ndim}"
        )
    if (
        routed_weights.ndim != 1
        or flat_token_indices.ndim != 1
        or route_local_expert_slots.ndim != 1
        or route_row_indices.ndim != 1
    ):
        raise ValueError("expected rank-1 route metadata tensors")
    routed_rows = routed_weights.numel()
    if (
        flat_token_indices.numel() != routed_rows
        or route_local_expert_slots.numel() != routed_rows
        or route_row_indices.numel() != routed_rows
    ):
        raise ValueError("route metadata lengths must match routed_weights")
    if routed_weights.stride(0) != 1:
        raise ValueError("routed_weights must be contiguous")
    if flat_token_indices.stride(0) != 1:
        raise ValueError("flat_token_indices must be contiguous")
    if route_local_expert_slots.stride(0) != 1:
        raise ValueError("route_local_expert_slots must be contiguous")
    if route_row_indices.stride(0) != 1:
        raise ValueError("route_row_indices must be contiguous")
    if accum_output.stride(1) != 1:
        raise ValueError("accum_output must be contiguous in the hidden dimension")
    if routed_rows == 0:
        return

    hidden_size = accum_output.shape[1]
    block_k = int(os.environ.get("B12X_BF16_SCATTER_ADD_BLOCK_K", "64"))
    num_warps = int(os.environ.get("B12X_BF16_SCATTER_ADD_NUM_WARPS", "2"))

    _scatter_add_grouped_fc2_bf16_kernel[
        (routed_rows, triton.cdiv(hidden_size, block_k))
    ](
        fc2_chunk,
        routed_weights,
        flat_token_indices,
        route_local_expert_slots,
        route_row_indices,
        accum_output,
        fc2_chunk.stride(0),
        fc2_chunk.stride(1),
        fc2_chunk.stride(2),
        accum_output.stride(0),
        hidden_size,
        ROUND_WEIGHTED_TO_BF16=round_weighted_to_bf16,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def scatter_routed_input_compact_chunk_bf16(
    a: torch.Tensor,
    flat_token_indices: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    route_row_indices: torch.Tensor,
    routed_chunk: torch.Tensor,
    *,
    expert_begin: int,
    expert_end: int,
) -> None:
    if (
        not a.is_cuda
        or not flat_token_indices.is_cuda
        or not compact_topk_ids.is_cuda
        or not route_row_indices.is_cuda
        or not routed_chunk.is_cuda
    ):
        raise ValueError("scatter_routed_input_compact_chunk_bf16 requires CUDA tensors")
    if a.dtype != torch.bfloat16 or routed_chunk.dtype != torch.bfloat16:
        raise ValueError(
            f"expected BF16 tensors for a/routed_chunk, got {a.dtype} and {routed_chunk.dtype}"
        )
    if flat_token_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            f"expected int32/int64 flat_token_indices, got {flat_token_indices.dtype}"
        )
    if compact_topk_ids.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            f"expected int32/int64 compact_topk_ids, got {compact_topk_ids.dtype}"
        )
    if route_row_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError(
            f"expected int32/int64 route_row_indices, got {route_row_indices.dtype}"
        )
    routed_rows = flat_token_indices.numel()
    if compact_topk_ids.numel() != routed_rows or route_row_indices.numel() != routed_rows:
        raise ValueError("route metadata lengths must match routed_rows")
    if routed_rows == 0:
        return

    hidden_size = a.shape[1]
    block_k = int(os.environ.get("B12X_BF16_SCATTER_BLOCK_K", "64"))
    num_warps = int(os.environ.get("B12X_BF16_SCATTER_NUM_WARPS", "2"))

    _scatter_routed_input_compact_chunk_bf16_kernel[
        (routed_rows, triton.cdiv(hidden_size, block_k))
    ](
        a,
        flat_token_indices,
        compact_topk_ids,
        route_row_indices,
        routed_chunk,
        expert_begin,
        expert_end,
        a.stride(0),
        routed_chunk.stride(0),
        routed_chunk.stride(1),
        routed_chunk.stride(2),
        hidden_size,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def scatter_add_compact_chunk_fc2_bf16(
    fc2_chunk: torch.Tensor,
    routed_weights: torch.Tensor,
    flat_token_indices: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    route_row_indices: torch.Tensor,
    accum_output: torch.Tensor,
    *,
    expert_begin: int,
    expert_end: int,
    round_weighted_to_bf16: bool,
) -> None:
    if (
        not fc2_chunk.is_cuda
        or not routed_weights.is_cuda
        or not flat_token_indices.is_cuda
        or not compact_topk_ids.is_cuda
        or not route_row_indices.is_cuda
        or not accum_output.is_cuda
    ):
        raise ValueError("scatter_add_compact_chunk_fc2_bf16 requires CUDA tensors")
    if fc2_chunk.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 fc2_chunk, got {fc2_chunk.dtype}")
    if routed_weights.dtype != torch.float32:
        raise ValueError(f"expected FP32 routed_weights, got {routed_weights.dtype}")
    if accum_output.dtype != torch.float32:
        raise ValueError(f"expected FP32 accum_output, got {accum_output.dtype}")
    routed_rows = routed_weights.numel()
    if (
        flat_token_indices.numel() != routed_rows
        or compact_topk_ids.numel() != routed_rows
        or route_row_indices.numel() != routed_rows
    ):
        raise ValueError("route metadata lengths must match routed_weights")
    if routed_rows == 0:
        return

    hidden_size = accum_output.shape[1]
    block_k = int(os.environ.get("B12X_BF16_SCATTER_ADD_BLOCK_K", "64"))
    num_warps = int(os.environ.get("B12X_BF16_SCATTER_ADD_NUM_WARPS", "2"))

    _scatter_add_compact_chunk_fc2_bf16_kernel[
        (routed_rows, triton.cdiv(hidden_size, block_k))
    ](
        fc2_chunk,
        routed_weights,
        flat_token_indices,
        compact_topk_ids,
        route_row_indices,
        accum_output,
        expert_begin,
        expert_end,
        fc2_chunk.stride(0),
        fc2_chunk.stride(1),
        fc2_chunk.stride(2),
        accum_output.stride(0),
        hidden_size,
        ROUND_WEIGHTED_TO_BF16=round_weighted_to_bf16,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def reduce_compact_chunk_fc2_by_token_bf16(
    fc2_chunk: torch.Tensor,
    routed_weights: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    route_row_indices: torch.Tensor,
    accum_output: torch.Tensor,
    *,
    expert_begin: int,
    expert_end: int,
    num_topk: int,
    round_weighted_to_bf16: bool,
) -> None:
    if (
        not fc2_chunk.is_cuda
        or not routed_weights.is_cuda
        or not compact_topk_ids.is_cuda
        or not route_row_indices.is_cuda
        or not accum_output.is_cuda
    ):
        raise ValueError("reduce_compact_chunk_fc2_by_token_bf16 requires CUDA tensors")
    if fc2_chunk.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 fc2_chunk, got {fc2_chunk.dtype}")
    if routed_weights.dtype != torch.float32:
        raise ValueError(f"expected FP32 routed_weights, got {routed_weights.dtype}")
    if accum_output.dtype != torch.float32:
        raise ValueError(f"expected FP32 accum_output, got {accum_output.dtype}")
    if compact_topk_ids.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"expected int32/int64 compact_topk_ids, got {compact_topk_ids.dtype}")
    if route_row_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"expected int32/int64 route_row_indices, got {route_row_indices.dtype}")
    if fc2_chunk.ndim != 3 or accum_output.ndim != 2:
        raise ValueError(
            f"expected rank-3 fc2_chunk and rank-2 accum_output, got {fc2_chunk.ndim} and {accum_output.ndim}"
        )
    if (
        routed_weights.ndim != 1
        or compact_topk_ids.ndim != 1
        or route_row_indices.ndim != 1
    ):
        raise ValueError("expected rank-1 route metadata tensors")
    routed_rows = routed_weights.numel()
    if compact_topk_ids.numel() != routed_rows or route_row_indices.numel() != routed_rows:
        raise ValueError("route metadata lengths must match routed_weights")
    if num_topk <= 0:
        raise ValueError(f"expected positive num_topk, got {num_topk}")
    if routed_rows % num_topk != 0:
        raise ValueError("routed_weights length must be divisible by num_topk")
    if accum_output.shape[0] != routed_rows // num_topk:
        raise ValueError("accum_output token dimension must match routed_weights//num_topk")
    if accum_output.stride(1) != 1:
        raise ValueError("accum_output must be contiguous in the hidden dimension")
    if routed_rows == 0:
        return

    hidden_size = accum_output.shape[1]
    block_k = int(os.environ.get("B12X_BF16_SCATTER_ADD_BLOCK_K", "64"))
    num_warps = int(os.environ.get("B12X_BF16_SCATTER_ADD_NUM_WARPS", "2"))

    _reduce_compact_chunk_fc2_by_token_bf16_kernel[
        (accum_output.shape[0], triton.cdiv(hidden_size, block_k))
    ](
        fc2_chunk,
        routed_weights,
        compact_topk_ids,
        route_row_indices,
        accum_output,
        expert_begin,
        expert_end,
        fc2_chunk.stride(0),
        fc2_chunk.stride(1),
        fc2_chunk.stride(2),
        accum_output.stride(0),
        hidden_size,
        ROUND_WEIGHTED_TO_BF16=round_weighted_to_bf16,
        NUM_TOPK=num_topk,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def scatter_routed_input_token_map_bf16(
    a: torch.Tensor,
    token_map: torch.Tensor,
    routed_chunk: torch.Tensor,
) -> None:
    if not a.is_cuda or not token_map.is_cuda or not routed_chunk.is_cuda:
        raise ValueError("scatter_routed_input_token_map_bf16 requires CUDA tensors")
    if a.dtype != torch.bfloat16 or routed_chunk.dtype != torch.bfloat16:
        raise ValueError(
            f"expected BF16 tensors for a/routed_chunk, got {a.dtype} and {routed_chunk.dtype}"
        )
    if token_map.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"expected int32/int64 token_map, got {token_map.dtype}")
    if a.ndim != 2 or token_map.ndim != 2 or routed_chunk.ndim != 3:
        raise ValueError(
            f"expected rank-2 a/token_map and rank-3 routed_chunk, got {a.ndim}, {token_map.ndim}, {routed_chunk.ndim}"
        )
    local_experts, max_rows = token_map.shape
    if local_experts == 0 or max_rows == 0:
        return
    if routed_chunk.shape[0] < max_rows or routed_chunk.shape[2] < local_experts:
        raise ValueError("routed_chunk is smaller than token_map capacity")

    hidden_size = a.shape[1]
    block_k = int(os.environ.get("B12X_BF16_SCATTER_BLOCK_K", "64"))
    num_warps = int(os.environ.get("B12X_BF16_SCATTER_NUM_WARPS", "2"))
    pair_slots = local_experts * max_rows
    _scatter_routed_input_token_map_bf16_kernel[
        (pair_slots, triton.cdiv(hidden_size, block_k))
    ](
        a,
        token_map,
        routed_chunk,
        token_map.stride(0),
        token_map.stride(1),
        a.stride(0),
        routed_chunk.stride(0),
        routed_chunk.stride(1),
        routed_chunk.stride(2),
        max_rows,
        hidden_size,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def scatter_add_token_map_fc2_bf16(
    fc2_chunk: torch.Tensor,
    token_map: torch.Tensor,
    token_weights: torch.Tensor,
    accum_output: torch.Tensor,
    *,
    round_weighted_to_bf16: bool,
) -> None:
    if (
        not fc2_chunk.is_cuda
        or not token_map.is_cuda
        or not token_weights.is_cuda
        or not accum_output.is_cuda
    ):
        raise ValueError("scatter_add_token_map_fc2_bf16 requires CUDA tensors")
    if fc2_chunk.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 fc2_chunk, got {fc2_chunk.dtype}")
    if token_map.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"expected int32/int64 token_map, got {token_map.dtype}")
    if token_weights.dtype != torch.float32:
        raise ValueError(f"expected FP32 token_weights, got {token_weights.dtype}")
    if accum_output.dtype != torch.float32:
        raise ValueError(f"expected FP32 accum_output, got {accum_output.dtype}")
    if fc2_chunk.ndim != 3 or token_map.ndim != 2 or token_weights.ndim != 2 or accum_output.ndim != 2:
        raise ValueError(
            f"expected rank-3 fc2_chunk, rank-2 token_map/token_weights/accum_output, got {fc2_chunk.ndim}, {token_map.ndim}, {token_weights.ndim}, {accum_output.ndim}"
        )
    if token_map.shape != token_weights.shape:
        raise ValueError("token_map/token_weights shape mismatch")
    local_experts, max_rows = token_map.shape
    if local_experts == 0 or max_rows == 0:
        return
    if fc2_chunk.shape[0] < max_rows or fc2_chunk.shape[2] < local_experts:
        raise ValueError("fc2_chunk is smaller than token_map capacity")
    if accum_output.stride(1) != 1:
        raise ValueError("accum_output must be contiguous in the hidden dimension")

    hidden_size = accum_output.shape[1]
    block_k = int(os.environ.get("B12X_BF16_SCATTER_ADD_BLOCK_K", "64"))
    num_warps = int(os.environ.get("B12X_BF16_SCATTER_ADD_NUM_WARPS", "2"))
    pair_slots = local_experts * max_rows
    _scatter_add_token_map_fc2_bf16_kernel[
        (pair_slots, triton.cdiv(hidden_size, block_k))
    ](
        fc2_chunk,
        token_map,
        token_weights,
        accum_output,
        token_map.stride(0),
        token_map.stride(1),
        token_weights.stride(0),
        token_weights.stride(1),
        fc2_chunk.stride(0),
        fc2_chunk.stride(1),
        fc2_chunk.stride(2),
        accum_output.stride(0),
        max_rows,
        hidden_size,
        ROUND_WEIGHTED_TO_BF16=round_weighted_to_bf16,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
