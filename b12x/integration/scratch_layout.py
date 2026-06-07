"""Shared byte-layout helpers for caller-owned scratch buffers."""

from __future__ import annotations

import torch

SCRATCH_ALIGN_BYTES = 1024


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + alignment - 1) // alignment) * alignment


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def materialize_scratch_view(
    scratch: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    offset_bytes = align_up(offset_bytes, max(SCRATCH_ALIGN_BYTES, dtype_nbytes(dtype)))
    nbytes = shape_numel(shape) * dtype_nbytes(dtype)
    view_bytes = scratch.narrow(0, offset_bytes, nbytes)
    typed_view = view_bytes.view(dtype).view(shape)
    return typed_view, offset_bytes + nbytes


def materialize_scratch_strided_view(
    scratch: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    offset_bytes = align_up(offset_bytes, max(SCRATCH_ALIGN_BYTES, dtype_nbytes(dtype)))
    nbytes = shape_numel(shape) * dtype_nbytes(dtype)
    view_bytes = scratch.narrow(0, offset_bytes, nbytes)
    typed_storage = view_bytes.view(dtype)
    return typed_storage.as_strided(shape, stride), offset_bytes + nbytes
