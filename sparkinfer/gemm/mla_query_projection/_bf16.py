"""Tiny-M BF16 absorbed-query GEMM with fused MLA query assembly."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import triton
import triton.language as tl

from .._shared.mxfp8_bmm import _overlaps, _torch_stream

_NOPE_DIM = 192
_LATENT_DIM = 512
_ROPE_DIM = 64
_QUERY_DIM = _LATENT_DIM + _ROPE_DIM
_MAX_M = 32
_QUALIFIED_HEADS = frozenset((8, 11, 16))
_BLOCK_N = 32
_BLOCK_K = 64
_COMPILED_SIGNATURES: set[tuple[int, int, bool]] = set()


@triton.jit
def _mla_query_projection_bf16_kernel(
    q_nope_ptr,
    weight_ptr,
    q_pe_ptr,
    q_scale_ptr,
    out_ptr,
    m,
    q_stride_h,
    q_stride_m,
    w_stride_h,
    w_stride_k,
    pe_stride_m,
    pe_stride_h,
    out_stride_m,
    out_stride_h,
    OUTPUT_FP8: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_block = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    cols = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in tl.static_range(0, NOPE_DIM, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        q_ptrs = (
            q_nope_ptr + head * q_stride_h + rows[:, None] * q_stride_m + ks[None, :]
        )
        w_ptrs = (
            weight_ptr + head * w_stride_h + ks[:, None] * w_stride_k + cols[None, :]
        )
        q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)
        w_tile = tl.load(w_ptrs)
        acc += tl.dot(q_tile, w_tile, out_dtype=tl.float32)

    # Preserve the established BMM contract: project to BF16 before an
    # optional static E4M3 quantization of the assembled attention query.
    projected = acc.to(tl.bfloat16)
    out_ptrs = (
        out_ptr + rows[:, None] * out_stride_m + head * out_stride_h + cols[None, :]
    )
    if OUTPUT_FP8:
        inv_scale = 1.0 / tl.load(q_scale_ptr)
        projected_fp32 = projected.to(tl.float32) * inv_scale
        projected_fp32 = tl.maximum(tl.minimum(projected_fp32, 448.0), -448.0)
        tl.store(out_ptrs, projected_fp32, mask=row_mask[:, None])
    else:
        tl.store(out_ptrs, projected, mask=row_mask[:, None])

    # The first N tile also copies the existing RoPE suffix. This is disjoint
    # from the projected columns and removes the separate concat/quant launch.
    if n_block == 0:
        rope_cols = tl.arange(0, ROPE_DIM)
        pe_ptrs = (
            q_pe_ptr
            + rows[:, None] * pe_stride_m
            + head * pe_stride_h
            + rope_cols[None, :]
        )
        rope = tl.load(pe_ptrs, mask=row_mask[:, None], other=0.0)
        rope_out_ptrs = (
            out_ptr
            + rows[:, None] * out_stride_m
            + head * out_stride_h
            + LATENT_DIM
            + rope_cols[None, :]
        )
        if OUTPUT_FP8:
            rope_fp32 = rope.to(tl.float32) * inv_scale
            rope_fp32 = tl.maximum(tl.minimum(rope_fp32, 448.0), -448.0)
            tl.store(rope_out_ptrs, rope_fp32, mask=row_mask[:, None])
        else:
            tl.store(rope_out_ptrs, rope, mask=row_mask[:, None])


def _validate(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
) -> tuple[int, int, bool]:
    if q_nope.ndim != 3:
        raise ValueError(f"q_nope must have shape [H,M,192], got {q_nope.shape}")
    heads, m, nope_dim = map(int, q_nope.shape)
    if heads not in _QUALIFIED_HEADS or not 1 <= m <= _MAX_M or nope_dim != _NOPE_DIM:
        raise NotImplementedError(
            "the BF16 fused MLA query specialization requires "
            f"H in {sorted(_QUALIFIED_HEADS)}, 1<=M<=32, K=192; "
            f"got H={heads}, M={m}, K={nope_dim}"
        )
    if tuple(weight.shape) != (heads, _NOPE_DIM, _LATENT_DIM):
        raise ValueError(
            "weight must have shape "
            f"{(heads, _NOPE_DIM, _LATENT_DIM)}, got {tuple(weight.shape)}"
        )
    if tuple(q_pe.shape) != (m, heads, _ROPE_DIM):
        raise ValueError(
            f"q_pe must have shape {(m, heads, _ROPE_DIM)}, got {tuple(q_pe.shape)}"
        )
    if tuple(out.shape) != (m, heads, _QUERY_DIM):
        raise ValueError(
            f"out must have shape {(m, heads, _QUERY_DIM)}, got {tuple(out.shape)}"
        )
    if q_nope.dtype != torch.bfloat16:
        raise TypeError(f"q_nope must be bfloat16, got {q_nope.dtype}")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"weight must be bfloat16, got {weight.dtype}")
    if q_pe.dtype != torch.bfloat16:
        raise TypeError(f"q_pe must be bfloat16, got {q_pe.dtype}")
    output_fp8 = out.dtype == torch.float8_e4m3fn
    if not output_fp8 and out.dtype != torch.bfloat16:
        raise ValueError(f"out must be bfloat16 or float8_e4m3fn, got {out.dtype}")
    if output_fp8:
        if q_scale is None:
            raise ValueError("q_scale is required for float8_e4m3fn output")
        if q_scale.dtype != torch.float32 or q_scale.numel() != 1:
            raise ValueError(
                "q_scale must be a scalar float32 tensor, "
                f"got shape={tuple(q_scale.shape)}, dtype={q_scale.dtype}"
            )
    tensors = [q_nope, weight, q_pe, out]
    if output_fp8:
        assert q_scale is not None
        tensors.append(q_scale)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("BF16 MLA query operands must be CUDA tensors")
    if any(tensor.device != q_nope.device for tensor in tensors[1:]):
        raise ValueError("BF16 MLA query operands must be on the same CUDA device")
    for name, tensor in (
        ("q_nope", q_nope),
        ("weight", weight),
        ("q_pe", q_pe),
        ("out", out),
    ):
        if int(tensor.stride(-1)) != 1:
            raise ValueError(f"{name} innermost dimension must be contiguous")
    sources = [
        ("q_nope", q_nope),
        ("weight", weight),
        ("q_pe", q_pe),
    ]
    if output_fp8:
        assert q_scale is not None
        sources.append(("q_scale", q_scale))
    for name, source in sources:
        if _overlaps(out, source):
            raise ValueError(f"out must not overlap {name}")
    return heads, m, output_fp8


def _launch(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
) -> None:
    heads, m, output_fp8 = _validate(q_nope, weight, q_pe, q_scale, out)
    block_m = 16 if m <= 16 else 32
    device_index = int(
        q_nope.device.index
        if q_nope.device.index is not None
        else torch.cuda.current_device()
    )
    signature = (device_index, block_m, output_fp8)
    if (
        torch.cuda.is_current_stream_capturing()
        and signature not in _COMPILED_SIGNATURES
    ):
        raise RuntimeError(
            "BF16 MLA query compile miss during CUDA-graph capture for "
            f"M={m}, output_fp8={output_fp8}; "
            "call mla_query_projection.prewarm first"
        )
    # This geometry wins across the complete qualified envelope (M=1..32 and
    # H=8/11/16). Keeping one shape also bounds prewarm to two M regimes.
    grid = (triton.cdiv(_LATENT_DIM, _BLOCK_N), heads)
    _mla_query_projection_bf16_kernel[grid](
        q_nope,
        weight,
        q_pe,
        q_scale if q_scale is not None else out,
        out,
        m,
        q_nope.stride(0),
        q_nope.stride(1),
        weight.stride(0),
        weight.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        out.stride(0),
        out.stride(1),
        OUTPUT_FP8=output_fp8,
        NOPE_DIM=_NOPE_DIM,
        LATENT_DIM=_LATENT_DIM,
        ROPE_DIM=_ROPE_DIM,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    _COMPILED_SIGNATURES.add(signature)


@torch.library.custom_op("sparkinfer::mla_query_projection_bf16", mutates_args=("out",))
def _op(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
) -> None:
    _launch(q_nope, weight, q_pe, q_scale, out)


@_op.register_fake
def _fake(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    out: torch.Tensor,
) -> None:
    del q_nope, weight, q_pe, q_scale, out


def run(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    out: torch.Tensor,
    *,
    q_scale: Optional[torch.Tensor] = None,
    stream: Optional[object] = None,
) -> torch.Tensor:
    """Run the BF16 absorbed projection and query-assembly epilogue."""
    if stream is None:
        torch.ops.sparkinfer.mla_query_projection_bf16(
            q_nope, weight, q_pe, q_scale, out
        )
        return out
    target = _torch_stream(stream, q_nope.device)
    with torch.cuda.stream(target):
        torch.ops.sparkinfer.mla_query_projection_bf16(
            q_nope, weight, q_pe, q_scale, out
        )
        tensors = [q_nope, weight, q_pe, out]
        if q_scale is not None:
            tensors.append(q_scale)
        for tensor in tensors:
            tensor.record_stream(target)
    return out


def prewarm(
    weight: torch.Tensor,
    m_values: Iterable[int],
    *,
    output_dtype: torch.dtype,
    stream: Optional[object] = None,
    synchronize: bool = True,
) -> int:
    """Compile and first-launch every required BF16 fused-query regime."""
    if output_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(
            f"output_dtype must be bfloat16 or float8_e4m3fn, got {output_dtype}"
        )
    if weight.ndim != 3:
        raise ValueError(f"weight must have shape [H,192,512], got {weight.shape}")
    heads = int(weight.shape[0])
    values = tuple(dict.fromkeys(int(value) for value in m_values if int(value) > 0))
    for m in values:
        if not can_implement(
            num_heads=heads,
            max_m=m,
            nope_dim=int(weight.shape[1]),
            latent_dim=int(weight.shape[2]),
            output_dtype=output_dtype,
            device=weight.device,
        ):
            raise NotImplementedError(
                "the BF16 fused MLA query specialization cannot prewarm "
                f"H={heads}, M={m}, weight={tuple(weight.shape)}, "
                f"output_dtype={output_dtype}"
            )
    target = _torch_stream(stream, weight.device)
    q_scale = (
        torch.ones(1, dtype=torch.float32, device=weight.device)
        if output_dtype == torch.float8_e4m3fn
        else None
    )
    warmed_regimes: set[int] = set()
    with torch.cuda.stream(target):
        for m in values:
            block_m = 16 if m <= 16 else 32
            if block_m in warmed_regimes:
                continue
            warmed_regimes.add(block_m)
            q_nope = torch.zeros(
                (heads, m, _NOPE_DIM), dtype=torch.bfloat16, device=weight.device
            )
            q_pe = torch.zeros(
                (m, heads, _ROPE_DIM), dtype=torch.bfloat16, device=weight.device
            )
            out = torch.empty(
                (m, heads, _QUERY_DIM), dtype=output_dtype, device=weight.device
            )
            torch.ops.sparkinfer.mla_query_projection_bf16(
                q_nope, weight, q_pe, q_scale, out
            )
            for tensor in (q_nope, q_pe, out):
                tensor.record_stream(target)
        if q_scale is not None:
            q_scale.record_stream(target)
    if synchronize:
        target.synchronize()
    return len(warmed_regimes)


def can_implement(
    *,
    num_heads: int,
    max_m: int,
    nope_dim: int,
    latent_dim: int,
    output_dtype: torch.dtype,
    device=None,
) -> bool:
    """Return whether the qualified BF16 fused-query kernel covers a plan."""
    del device
    return bool(
        int(num_heads) in _QUALIFIED_HEADS
        and 1 <= int(max_m) <= _MAX_M
        and int(nope_dim) == _NOPE_DIM
        and int(latent_dim) == _LATENT_DIM
        and output_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    )


def clear_caches() -> None:
    _COMPILED_SIGNATURES.clear()


__all__ = ["can_implement", "clear_caches", "prewarm", "run"]
