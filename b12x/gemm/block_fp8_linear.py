from __future__ import annotations

import math
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import triton
import triton.language as tl

from b12x.gemm.dense import dense_gemm
from b12x.gemm.wo_projection import (
    MXFP8Rows,
    MXFP8_SCALE_K_TILE,
    MXFP8_SCALE_ROW_TILE,
    MXFP8_SCALE_VEC_SIZE,
    _check_gpu_tensor,
    _check_mxfp8_k,
    _check_mxfp8_rows_storage,
    empty_dense_gemm_mnl_view,
    empty_mxfp8_rows_bases,
    empty_mxfp8_rows_for_dense_gemm,
    mxfp8_rows_from_bases,
    pack_fp8_block_scaled_weight_mxfp8,
)
from b12x.cute.scratch import B12XScratchBufferSpec, scratch_buffer_spec, scratch_tensor

logger = logging.getLogger(__name__)
_B12X_TIMING = os.getenv("B12X_TIMING", "0") == "1" or os.getenv(
    "VLLM_B12X_TIMING", "0"
) == "1"
_B12X_TIMING_THRESHOLD_MS = float(
    os.getenv(
        "B12X_TIMING_THRESHOLD_MS",
        os.getenv("VLLM_B12X_TIMING_THRESHOLD_MS", "0"),
    )
)

_SCRATCH_ALIGN_BYTES = 1024


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

    def bind(
        self,
        *,
        source: torch.Tensor,
        packed_weight: "BlockFP8LinearWeight",
        bias: torch.Tensor | None = None,
        expected_m: int | None = None,
    ) -> "BlockFP8LinearBinding":
        return build_block_fp8_linear_binding(
            source=source,
            packed_weight=packed_weight,
            x_q=self.x_q,
            output=self.output,
            bias=bias,
            expected_m=expected_m,
        )


@dataclass(frozen=True, kw_only=True)
class BlockFP8LinearBinding:
    source: torch.Tensor
    packed_weight: BlockFP8LinearWeight
    x_q: MXFP8Rows
    output: torch.Tensor
    bias: torch.Tensor | None = None
    # DeepGEMM-style regime hint forwarded to dense_gemm (decode vs prefill tile).
    # None keeps the M-independent default; set it at bind time so the warmed
    # kernel matches the regime this binding serves.
    expected_m: int | None = None

    def run(self) -> torch.Tensor:
        return block_fp8_linear_mxfp8(binding=self)


@dataclass(frozen=True, kw_only=True)
class BlockFP8LinearScratchCaps:
    device: torch.device | str
    max_tokens: int
    in_features: int
    out_features: int
    output_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        object.__setattr__(self, "in_features", max(int(self.in_features), 1))
        object.__setattr__(self, "out_features", max(int(self.out_features), 1))
        _check_mxfp8_k(self.in_features)
        if self.output_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"output_dtype must be bf16/fp16, got {self.output_dtype}")


@dataclass(frozen=True)
class BlockFP8LinearScratchPlan:
    caps: BlockFP8LinearScratchCaps
    _scratch_specs: tuple[B12XScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[B12XScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        source: torch.Tensor,
        packed_weight: BlockFP8LinearWeight,
        output: torch.Tensor,
        bias: torch.Tensor | None = None,
        expected_m: int | None = None,
    ) -> BlockFP8LinearBinding:
        source_2d = _source_2d(source)
        tokens, in_features = map(int, source_2d.shape)
        if tokens > int(self.caps.max_tokens):
            raise ValueError(
                f"source tokens {tokens} exceed block-FP8 scratch capacity {self.caps.max_tokens}"
            )
        if in_features != int(self.caps.in_features):
            raise ValueError(
                f"source K={in_features} does not match scratch in_features={self.caps.in_features}"
            )
        if int(packed_weight.out_features) != int(self.caps.out_features):
            raise ValueError(
                "packed weight out_features "
                f"{packed_weight.out_features} does not match scratch out_features={self.caps.out_features}"
            )
        if source_2d.dtype != self.caps.output_dtype:
            raise ValueError(
                f"source dtype {source_2d.dtype} does not match scratch output_dtype={self.caps.output_dtype}"
            )
        scratch = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="block FP8 linear",
        )
        x_q = _block_fp8_linear_x_q_from_scratch(
            scratch,
            tokens=tokens,
            in_features=self.caps.in_features,
            output_dtype=self.caps.output_dtype,
        )
        return build_block_fp8_linear_binding(
            source=source,
            packed_weight=packed_weight,
            x_q=x_q,
            output=output,
            bias=bias,
            expected_m=expected_m,
        )


@dataclass(frozen=True, kw_only=True)
class _BlockFP8LinearScratchLayout:
    nbytes: int
    x_values_offset_bytes: int
    x_scale_rows_offset_bytes: int
    x_scale_mma_offset_bytes: int
    x_scale_mma_physical_shape: tuple[int, int, int, int, int, int]


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


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _shape_numel(shape: Sequence[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _block_fp8_linear_scratch_layout(
    *,
    tokens: int,
    in_features: int,
    out_features: int,
    output_dtype: torch.dtype,
) -> _BlockFP8LinearScratchLayout:
    tokens = max(int(tokens), 1)
    in_features = max(int(in_features), 1)
    del out_features
    _check_mxfp8_k(in_features)
    _c_dtype_name(output_dtype)

    offset = 0
    offset = _align_up(offset, _SCRATCH_ALIGN_BYTES)
    x_values_offset_bytes = offset
    offset += tokens * in_features * _dtype_nbytes(torch.float8_e4m3fn)

    offset = _align_up(offset, _SCRATCH_ALIGN_BYTES)
    x_scale_rows_offset_bytes = offset
    offset += (
        tokens
        * (in_features // MXFP8_SCALE_VEC_SIZE)
        * _dtype_nbytes(torch.float8_e8m0fnu)
    )

    offset = _align_up(offset, _SCRATCH_ALIGN_BYTES)
    x_scale_mma_offset_bytes = offset
    sf_k = in_features // MXFP8_SCALE_VEC_SIZE
    x_scale_mma_physical_shape = (
        1,
        math.ceil(tokens / MXFP8_SCALE_ROW_TILE),
        math.ceil(sf_k / MXFP8_SCALE_K_TILE),
        32,
        4,
        4,
    )
    offset += _shape_numel(x_scale_mma_physical_shape) * _dtype_nbytes(torch.uint8)

    return _BlockFP8LinearScratchLayout(
        nbytes=max(int(offset), 1),
        x_values_offset_bytes=x_values_offset_bytes,
        x_scale_rows_offset_bytes=x_scale_rows_offset_bytes,
        x_scale_mma_offset_bytes=x_scale_mma_offset_bytes,
        x_scale_mma_physical_shape=x_scale_mma_physical_shape,
    )


def _scratch_view(
    scratch: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    offset_bytes = _align_up(offset_bytes, max(_SCRATCH_ALIGN_BYTES, _dtype_nbytes(dtype)))
    nbytes = _shape_numel(shape) * _dtype_nbytes(dtype)
    return scratch.narrow(0, offset_bytes, nbytes).view(dtype).view(shape)


def _block_fp8_linear_x_q_from_scratch(
    scratch: torch.Tensor,
    *,
    tokens: int,
    in_features: int,
    output_dtype: torch.dtype,
) -> MXFP8Rows:
    layout = _block_fp8_linear_scratch_layout(
        tokens=tokens,
        in_features=in_features,
        out_features=1,
        output_dtype=output_dtype,
    )
    if scratch.dtype != torch.uint8:
        raise TypeError(
            f"block FP8 linear scratch must have dtype torch.uint8, got {scratch.dtype}"
        )
    if not scratch.is_contiguous():
        raise ValueError("block FP8 linear scratch must be contiguous")
    if int(scratch.numel()) < int(layout.nbytes):
        raise ValueError(
            f"block FP8 linear scratch has {int(scratch.numel())} bytes, requires {layout.nbytes}"
        )
    x_values = _scratch_view(
        scratch,
        offset_bytes=layout.x_values_offset_bytes,
        shape=(int(tokens), int(in_features)),
        dtype=torch.float8_e4m3fn,
    )
    x_scale_rows_u8 = _scratch_view(
        scratch,
        offset_bytes=layout.x_scale_rows_offset_bytes,
        shape=(1, int(tokens), int(in_features) // MXFP8_SCALE_VEC_SIZE),
        dtype=torch.uint8,
    )
    x_scale_mma_u8 = _scratch_view(
        scratch,
        offset_bytes=layout.x_scale_mma_offset_bytes,
        shape=layout.x_scale_mma_physical_shape,
        dtype=torch.uint8,
    )
    x_scale_rows_u8.fill_(127)
    x_scale_mma_u8.fill_(127)
    x_scale_mma = x_scale_mma_u8.view(torch.float8_e8m0fnu).permute(
        3,
        4,
        1,
        5,
        2,
        0,
    )
    return MXFP8Rows(
        values=x_values,
        scale_rows=x_scale_rows_u8.view(torch.float8_e8m0fnu),
        scale_mma=x_scale_mma,
    )


def _source_2d(source: torch.Tensor) -> torch.Tensor:
    if source.ndim == 0:
        raise ValueError("source must have at least one dimension")
    return source.view(-1, source.shape[-1])


def _check_block_fp8_linear_tensors(
    x_q: MXFP8Rows,
    output: torch.Tensor,
    *,
    tokens: int,
    packed_weight: BlockFP8LinearWeight,
    output_dtype: torch.dtype,
) -> None:
    _check_mxfp8_rows_storage(
        x_q,
        m=tokens,
        k=packed_weight.in_features,
        num_groups=1,
    )
    if output.shape != (tokens, packed_weight.out_features, 1):
        raise ValueError(
            "output must have shape "
            f"{(tokens, packed_weight.out_features, 1)}, got {tuple(output.shape)}"
        )
    if output.dtype != output_dtype:
        raise ValueError(f"output dtype {output.dtype} does not match input {output_dtype}")


def _check_block_fp8_linear_workspace(
    workspace: BlockFP8LinearWorkspace,
    *,
    tokens: int,
    packed_weight: BlockFP8LinearWeight,
    output_dtype: torch.dtype,
) -> None:
    if not isinstance(workspace, BlockFP8LinearWorkspace):
        raise TypeError("workspace must be a BlockFP8LinearWorkspace")
    _check_block_fp8_linear_tensors(
        workspace.x_q,
        workspace.output,
        tokens=tokens,
        packed_weight=packed_weight,
        output_dtype=output_dtype,
    )


def build_block_fp8_linear_binding(
    *,
    source: torch.Tensor,
    packed_weight: BlockFP8LinearWeight,
    x_q: MXFP8Rows,
    output: torch.Tensor,
    bias: torch.Tensor | None = None,
    expected_m: int | None = None,
) -> BlockFP8LinearBinding:
    if not isinstance(packed_weight, BlockFP8LinearWeight):
        raise TypeError("packed_weight must be a BlockFP8LinearWeight")
    source_2d = _source_2d(source)
    tokens, in_features = map(int, source_2d.shape)
    if in_features != packed_weight.in_features:
        raise ValueError(
            f"input K={in_features} does not match packed weight K={packed_weight.in_features}"
        )
    if source_2d.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"source dtype must be bf16/fp16, got {source_2d.dtype}")
    _check_block_fp8_linear_tensors(
        x_q,
        output,
        tokens=tokens,
        packed_weight=packed_weight,
        output_dtype=source_2d.dtype,
    )
    return BlockFP8LinearBinding(
        source=source,
        packed_weight=packed_weight,
        x_q=x_q,
        output=output,
        bias=bias,
        expected_m=expected_m,
    )


def plan_block_fp8_linear_scratch(
    caps: BlockFP8LinearScratchCaps,
) -> BlockFP8LinearScratchPlan:
    layout = _block_fp8_linear_scratch_layout(
        tokens=caps.max_tokens,
        in_features=caps.in_features,
        out_features=caps.out_features,
        output_dtype=caps.output_dtype,
    )
    return BlockFP8LinearScratchPlan(
        caps=caps,
        _scratch_specs=(
            scratch_buffer_spec(
                "block_fp8_linear.scratch",
                nbytes=layout.nbytes,
                device=caps.device,
            ),
        ),
    )


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


def _run_block_fp8_quant_kernel(
    source_tk: torch.Tensor,
    out_values: torch.Tensor,
    out_scale_rows: torch.Tensor,
    out_scale_mma: torch.Tensor,
    tokens: int,
    in_features: int,
) -> None:
    _quantize_dense_tk_to_tk_kernel[(tokens, in_features // MXFP8_SCALE_VEC_SIZE)](
        source_tk,
        out_values,
        out_scale_rows.view(torch.uint8),
        out_scale_mma.view(torch.uint8),
        tokens,
        source_tk.stride(0),
        source_tk.stride(1),
        out_values.stride(0),
        out_values.stride(1),
        out_scale_rows.stride(0),
        out_scale_rows.stride(1),
        out_scale_rows.stride(2),
        out_scale_mma.stride(0),
        out_scale_mma.stride(1),
        out_scale_mma.stride(2),
        out_scale_mma.stride(3),
        out_scale_mma.stride(4),
        out_scale_mma.stride(5),
        BLOCK=MXFP8_SCALE_VEC_SIZE,
    )


@torch.library.custom_op(
    "b12x::quantize_block_fp8_linear_input_mxfp8_alloc",
    mutates_args=(),
)
def _quantize_block_fp8_linear_input_mxfp8_alloc_op(
    source_tk: torch.Tensor,
    tokens: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Functional (allocate + return) quantizer: the raw Triton kernel writes the
    # MXFP8 output views INSIDE this opaque op, so torch.compile never sees a
    # triton_kernel_wrapper_mutation on 3 caller-visible views. Returns the
    # CONTIGUOUS bases (not the views) so no symbolic-shaped view output reaches a
    # downstream subgraph (which trips AOT merge_view_inputs). Used on the no-`out`
    # (compile) path.
    values_base, scale_rows_base, scale_physical_base = empty_mxfp8_rows_bases(
        tokens, in_features, num_groups=1, device=source_tk.device
    )
    out = mxfp8_rows_from_bases(
        values_base, scale_rows_base, scale_physical_base,
        tokens, in_features, num_groups=1,
    )
    _run_block_fp8_quant_kernel(
        source_tk, out.values, out.scale_rows, out.scale_mma, tokens, in_features
    )
    return values_base, scale_rows_base, scale_physical_base


@_quantize_block_fp8_linear_input_mxfp8_alloc_op.register_fake
def _quantize_block_fp8_linear_input_mxfp8_alloc_fake(
    source_tk: torch.Tensor,
    tokens: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return empty_mxfp8_rows_bases(
        tokens, in_features, num_groups=1, device=source_tk.device
    )


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
        values_base, scale_rows_base, scale_physical_base = (
            torch.ops.b12x.quantize_block_fp8_linear_input_mxfp8_alloc(
                source_tk, tokens, in_features
            )
        )
        return mxfp8_rows_from_bases(
            values_base, scale_rows_base, scale_physical_base,
            tokens, in_features, num_groups=1,
        )

    _check_mxfp8_rows_storage(out, m=tokens, k=in_features, num_groups=1)
    _run_block_fp8_quant_kernel(
        source_tk, out.values, out.scale_rows, out.scale_mma, tokens, in_features
    )
    return out


@torch.library.custom_op(
    "b12x::block_fp8_linear_mxfp8_fused",
    mutates_args=(),
)
def _block_fp8_linear_mxfp8_fused_op(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_rows: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    in_features: int,
    out_features: int,
    expected_m: int,
) -> torch.Tensor:
    # Fused, fully opaque block-FP8 linear: quantize + dense GEMM run INSIDE this
    # one op, so the token-shaped activation MXFP8 views (x_q.values/scale_mma)
    # are never graph values -- they can't become symbolic-shaped view inputs to
    # a downstream subgraph (which trips torch AOT merge_view_inputs under dynamic
    # shapes). The weight views passed in are static-shaped, so they don't hit
    # that path. Returns a contiguous [tokens, out_features] base.
    tokens = int(source_2d.shape[0])
    x_q = quantize_block_fp8_linear_input_mxfp8(source_2d)
    return dense_gemm(
        (x_q.values.reshape(tokens, in_features, 1), x_q.scale_mma),
        (weight_values.reshape(out_features, in_features, 1), weight_scale_mma),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_c_dtype_name(source_2d.dtype),
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        expected_m=expected_m,
    )[:, :, 0]


@_block_fp8_linear_mxfp8_fused_op.register_fake
def _block_fp8_linear_mxfp8_fused_fake(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_rows: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    in_features: int,
    out_features: int,
    expected_m: int,
) -> torch.Tensor:
    return torch.empty(
        (source_2d.shape[0], out_features),
        dtype=source_2d.dtype,
        device=source_2d.device,
    )


def block_fp8_linear_mxfp8(
    source: torch.Tensor | None = None,
    packed_weight: BlockFP8LinearWeight | None = None,
    *,
    workspace: BlockFP8LinearWorkspace | None = None,
    bias: torch.Tensor | None = None,
    binding: BlockFP8LinearBinding | None = None,
    expected_m: int | None = None,
) -> torch.Tensor:
    """Run a serialized block-FP8 linear through the native b12x MXFP8 GEMM.

    expected_m forwards a DeepGEMM-style regime hint to dense_gemm (decode vs
    prefill tile); None keeps the M-independent default. When a binding is given
    its stored expected_m is used.
    """

    if binding is not None:
        extras = [
            name
            for name, value in (
                ("source", source),
                ("packed_weight", packed_weight),
                ("workspace", workspace),
                ("bias", bias),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "block FP8 linear binding owns source, packed weight, scratch tensors, and bias; "
                f"do not also pass {', '.join(extras)}"
            )
        source = binding.source
        packed_weight = binding.packed_weight
        x_q_storage = binding.x_q
        output_storage = binding.output
        bias = binding.bias
        expected_m = binding.expected_m
    else:
        x_q_storage = None
        output_storage = None
    if source is None or packed_weight is None:
        raise TypeError("block_fp8_linear_mxfp8 requires source and packed_weight or binding")
    _check_gpu_tensor("source", source)
    if not isinstance(packed_weight, BlockFP8LinearWeight):
        raise TypeError("packed_weight must be a BlockFP8LinearWeight")
    source_2d = _source_2d(source)
    tokens, in_features = source_2d.shape
    if in_features != packed_weight.in_features:
        raise ValueError(
            f"input K={in_features} does not match packed weight K={packed_weight.in_features}"
        )
    if source_2d.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"source dtype must be bf16/fp16, got {source_2d.dtype}")

    if binding is None and workspace is None:
        # No caller-owned buffers (e.g. the torch.compile path): one fully opaque
        # fused op runs quantize + dense GEMM internally, so the token-shaped
        # activation MXFP8 views stay inside the op and never reach the graph (see
        # _block_fp8_linear_mxfp8_fused_op). No is_compiling -- purely caller-intent.
        output = torch.ops.b12x.block_fp8_linear_mxfp8_fused(
            source_2d,
            packed_weight.weight.values,
            packed_weight.weight.scale_rows,
            packed_weight.weight.scale_mma,
            packed_weight.in_features,
            packed_weight.out_features,
            int(expected_m) if expected_m is not None else tokens,
        )
        if bias is not None:
            output = output + bias
        return output.view(*source.shape[:-1], packed_weight.out_features)

    if workspace is None:
        if x_q_storage is None or output_storage is None:
            workspace = empty_block_fp8_linear_workspace(
                tokens,
                packed_weight.in_features,
                packed_weight.out_features,
                device=source_2d.device,
                output_dtype=source_2d.dtype,
            )
            x_q_storage = workspace.x_q
            output_storage = workspace.output
        else:
            _check_block_fp8_linear_tensors(
                x_q_storage,
                output_storage,
                tokens=tokens,
                packed_weight=packed_weight,
                output_dtype=source_2d.dtype,
            )
    else:
        _check_block_fp8_linear_workspace(
            workspace,
            tokens=tokens,
            packed_weight=packed_weight,
            output_dtype=source_2d.dtype,
        )
        x_q_storage = workspace.x_q
        output_storage = workspace.output

    assert x_q_storage is not None
    assert output_storage is not None
    t0 = time.perf_counter() if _B12X_TIMING else 0.0
    x_q = quantize_block_fp8_linear_input_mxfp8(source_2d, out=x_q_storage)
    t_quant = time.perf_counter() if _B12X_TIMING else 0.0
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
        out=output_storage,
        expected_m=expected_m,
    )[:, :, 0]
    t_gemm = time.perf_counter() if _B12X_TIMING else 0.0
    if bias is not None:
        output += bias
    if _B12X_TIMING:
        t_done = time.perf_counter()
        total_ms = (t_done - t0) * 1000.0
        if total_ms >= _B12X_TIMING_THRESHOLD_MS:
            logger.warning(
                "b12x_block_fp8_linear timing tokens=%d in=%d out=%d "
                "quant_enqueue=%.3fms dense_gemm=%.3fms bias=%.3fms total=%.3fms",
                int(tokens),
                int(packed_weight.in_features),
                int(packed_weight.out_features),
                (t_quant - t0) * 1000.0,
                (t_gemm - t_quant) * 1000.0,
                (t_done - t_gemm) * 1000.0,
                total_ms,
            )
    return output.view(*source.shape[:-1], packed_weight.out_features)


def prewarm_block_fp8_linear_mxfp8(
    packed_weight: BlockFP8LinearWeight,
    token_counts: Iterable[int],
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    expected_m: int | None = None,
) -> None:
    """Compile and warm the native block-FP8 linear kernels for planned M values.

    Pass the same expected_m the serving path will use so the warmed tile matches
    the regime kernel that live calls reuse under frozen resolution.
    """

    if not isinstance(packed_weight, BlockFP8LinearWeight):
        raise TypeError("packed_weight must be a BlockFP8LinearWeight")
    if output_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"output_dtype must be bf16/fp16, got {output_dtype}")
    device = packed_weight.weight.values.device
    counts = sorted({int(tokens) for tokens in token_counts if int(tokens) > 0})
    if not counts:
        return

    with torch.inference_mode():
        for tokens in counts:
            workspace = empty_block_fp8_linear_workspace(
                tokens,
                packed_weight.in_features,
                packed_weight.out_features,
                device=device,
                output_dtype=output_dtype,
            )
            source = torch.zeros(
                (tokens, packed_weight.in_features),
                dtype=output_dtype,
                device=device,
            )
            block_fp8_linear_mxfp8(
                source, packed_weight, workspace=workspace, expected_m=expected_m
            )
        torch.cuda.synchronize(device)


__all__ = [
    "BlockFP8LinearBinding",
    "BlockFP8LinearScratchCaps",
    "BlockFP8LinearScratchPlan",
    "BlockFP8LinearWeight",
    "BlockFP8LinearWorkspace",
    "build_block_fp8_linear_binding",
    "block_fp8_linear_mxfp8",
    "empty_block_fp8_linear_workspace",
    "pack_block_fp8_linear_weight_mxfp8",
    "plan_block_fp8_linear_scratch",
    "prewarm_block_fp8_linear_mxfp8",
    "quantize_block_fp8_linear_input_mxfp8",
]
