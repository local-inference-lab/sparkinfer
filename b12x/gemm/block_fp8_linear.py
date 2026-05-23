from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import triton
import triton.language as tl

from b12x.gemm.dense import dense_gemm
from b12x.gemm.wo_projection import (
    MXFP8Rows,
    MXFP8_SCALE_VEC_SIZE,
    _check_gpu_tensor,
    _check_mxfp8_k,
    _check_mxfp8_rows_storage,
    empty_dense_gemm_mnl_view,
    empty_mxfp8_rows_for_dense_gemm,
    pack_fp8_block_scaled_weight_mxfp8,
)


@dataclass(frozen=True)
class BlockFP8LinearWeight:
    weight: MXFP8Rows
    in_features: int
    out_features: int
    block_size: tuple[int, int]


@dataclass(frozen=True)
class BlockFP8LinearWorkspace:
    x_q: MXFP8Rows
    output: torch.Tensor


@triton.jit
def _quantize_dense_tk_to_tk_kernel(
    source,
    values,
    scale_rows,
    scale_mma,
    tokens,
    source_stride_t,
    source_stride_k,
    values_stride_t,
    values_stride_k,
    scale_rows_stride_l,
    scale_rows_stride_t,
    scale_rows_stride_k,
    scale_mma_s0,
    scale_mma_s1,
    scale_mma_s2,
    scale_mma_s3,
    scale_mma_s4,
    scale_mma_s5,
    BLOCK: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    k = chunk * BLOCK + offs

    src = tl.load(source + token * source_stride_t + k * source_stride_k).to(tl.float32)
    max_abs = tl.max(tl.abs(src), axis=0)
    safe = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scale_exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(safe)), -127.0), 127.0)
    scale = tl.exp2(scale_exp)
    scale_u8 = (scale_exp + 127.0).to(tl.uint8)

    tl.store(
        values + token * values_stride_t + k * values_stride_k,
        (src / scale).to(tl.float8e4nv),
    )

    tl.store(
        scale_rows
        + scale_rows_stride_l * 0
        + token * scale_rows_stride_t
        + chunk * scale_rows_stride_k,
        scale_u8,
    )

    row32 = token % 32
    row4 = (token // 32) % 4
    tile_m = token // 128
    k4 = chunk % 4
    tile_k = chunk // 4
    tl.store(
        scale_mma
        + row32 * scale_mma_s0
        + row4 * scale_mma_s1
        + tile_m * scale_mma_s2
        + k4 * scale_mma_s3
        + tile_k * scale_mma_s4
        + scale_mma_s5 * 0,
        scale_u8,
    )


def _check_block_size(block_size: Sequence[int]) -> tuple[int, int]:
    if len(block_size) != 2:
        raise ValueError(f"block_size must have two elements, got {block_size}")
    block_n, block_k = int(block_size[0]), int(block_size[1])
    if (block_n, block_k) != (128, 128):
        raise ValueError(
            f"b12x block FP8 linear currently supports 128x128 weight blocks, got {block_size}"
        )
    return block_n, block_k


def _c_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"b12x block FP8 linear output dtype must be bf16/fp16, got {dtype}")


def pack_block_fp8_linear_weight_mxfp8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    block_size: Sequence[int] = (128, 128),
) -> BlockFP8LinearWeight:
    """Pack serialized block-FP8 linear weights for the native b12x MXFP8 GEMM.

    The checkpoint weight stays in E4M3. The 128x128 DSV-style block scales are
    expanded once to the row/32-column UE8M0 scale layout consumed by SM120 MMA.
    """

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("weight_scale", weight_scale)
    _check_block_size(block_size)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    _check_mxfp8_k(in_features)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    packed = pack_fp8_block_scaled_weight_mxfp8(
        weight.detach(),
        weight_scale.detach(),
        m=out_features,
        k=in_features,
        num_groups=1,
    )
    return BlockFP8LinearWeight(
        weight=packed,
        in_features=in_features,
        out_features=out_features,
        block_size=(128, 128),
    )


def empty_block_fp8_linear_workspace(
    tokens: int,
    in_features: int,
    out_features: int,
    *,
    device: torch.device | str,
    output_dtype: torch.dtype = torch.bfloat16,
) -> BlockFP8LinearWorkspace:
    if tokens <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("tokens, in_features, and out_features must be positive")
    _check_mxfp8_k(in_features)
    x_q = empty_mxfp8_rows_for_dense_gemm(
        tokens,
        in_features,
        num_groups=1,
        device=device,
    )
    output = empty_dense_gemm_mnl_view(
        tokens,
        out_features,
        1,
        device=device,
        dtype=output_dtype,
    )
    return BlockFP8LinearWorkspace(x_q=x_q, output=output)


def quantize_block_fp8_linear_input_mxfp8(
    source_tk: torch.Tensor,
    *,
    out: MXFP8Rows | None = None,
) -> MXFP8Rows:
    """Quantize dense BF16/FP16 rows `[tokens, K]` to native MXFP8 rows."""

    _check_gpu_tensor("source_tk", source_tk)
    if source_tk.ndim != 2:
        raise ValueError(f"source_tk must have shape [tokens,K], got {tuple(source_tk.shape)}")
    tokens, in_features = source_tk.shape
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    _check_mxfp8_k(in_features)
    if out is None:
        out = empty_mxfp8_rows_for_dense_gemm(
            tokens,
            in_features,
            num_groups=1,
            device=source_tk.device,
        )
    else:
        _check_mxfp8_rows_storage(out, m=tokens, k=in_features, num_groups=1)

    _quantize_dense_tk_to_tk_kernel[(tokens, in_features // MXFP8_SCALE_VEC_SIZE)](
        source_tk,
        out.values,
        out.scale_rows.view(torch.uint8),
        out.scale_mma.view(torch.uint8),
        tokens,
        source_tk.stride(0),
        source_tk.stride(1),
        out.values.stride(0),
        out.values.stride(1),
        out.scale_rows.stride(0),
        out.scale_rows.stride(1),
        out.scale_rows.stride(2),
        out.scale_mma.stride(0),
        out.scale_mma.stride(1),
        out.scale_mma.stride(2),
        out.scale_mma.stride(3),
        out.scale_mma.stride(4),
        out.scale_mma.stride(5),
        BLOCK=MXFP8_SCALE_VEC_SIZE,
    )
    return out


def block_fp8_linear_mxfp8(
    source: torch.Tensor,
    packed_weight: BlockFP8LinearWeight,
    *,
    workspace: BlockFP8LinearWorkspace | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a serialized block-FP8 linear through the native b12x MXFP8 GEMM."""

    _check_gpu_tensor("source", source)
    if not isinstance(packed_weight, BlockFP8LinearWeight):
        raise TypeError("packed_weight must be a BlockFP8LinearWeight")
    source_2d = source.view(-1, source.shape[-1])
    tokens, in_features = source_2d.shape
    if in_features != packed_weight.in_features:
        raise ValueError(
            f"input K={in_features} does not match packed weight K={packed_weight.in_features}"
        )
    if source_2d.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"source dtype must be bf16/fp16, got {source_2d.dtype}")

    if workspace is None:
        workspace = empty_block_fp8_linear_workspace(
            tokens,
            packed_weight.in_features,
            packed_weight.out_features,
            device=source_2d.device,
            output_dtype=source_2d.dtype,
        )
    else:
        _check_mxfp8_rows_storage(
            workspace.x_q,
            m=tokens,
            k=packed_weight.in_features,
            num_groups=1,
        )
        if workspace.output.shape != (tokens, packed_weight.out_features, 1):
            raise ValueError(
                "workspace.output must have shape "
                f"{(tokens, packed_weight.out_features, 1)}, got {tuple(workspace.output.shape)}"
            )
        if workspace.output.dtype != source_2d.dtype:
            raise ValueError(
                f"workspace.output dtype {workspace.output.dtype} does not match input {source_2d.dtype}"
            )

    x_q = quantize_block_fp8_linear_input_mxfp8(source_2d, out=workspace.x_q)
    output = dense_gemm(
        (x_q.values.reshape(tokens, packed_weight.in_features, 1), x_q.scale_mma),
        (
            packed_weight.weight.values.reshape(
                packed_weight.out_features,
                packed_weight.in_features,
                1,
            ),
            packed_weight.weight.scale_mma,
        ),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_c_dtype_name(source_2d.dtype),
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        out=workspace.output,
    )[:, :, 0]
    if bias is not None:
        output += bias
    return output.view(*source.shape[:-1], packed_weight.out_features)


__all__ = [
    "BlockFP8LinearWeight",
    "BlockFP8LinearWorkspace",
    "block_fp8_linear_mxfp8",
    "empty_block_fp8_linear_workspace",
    "pack_block_fp8_linear_weight_mxfp8",
    "quantize_block_fp8_linear_input_mxfp8",
]
