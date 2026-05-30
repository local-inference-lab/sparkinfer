"""CuTeDSL mHC residual helpers for DeepSeek-style residual mixing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x.attention.workspace import (
    _ARENA_ALIGN_BYTES,
    _align_up,
    _dtype_nbytes,
    _materialize_arena_view,
)
from b12x.integration.scratch import (
    B12XScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)


MHC_MULT = 4
MHC_MIXES = (2 + MHC_MULT) * MHC_MULT
MHC_PARTIALS = 1 + MHC_MIXES
MHC_DEFAULT_SPLIT_K = 64
MHC_DEFAULT_BLOCK_K = 256
MHC_DEFAULT_BLOCK_H = 512


@dataclass(frozen=True)
class MHCPreWorkspace:
    partials: torch.Tensor
    split_k: int


@dataclass(frozen=True)
class MHCWorkspace:
    partials: torch.Tensor
    y: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    out: torch.Tensor
    split_k: int

    @property
    def capacity(self) -> int:
        return int(self.partials.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.y.shape[1])

    def slice(self, num_tokens: int) -> "MHCWorkspace":
        num_tokens = int(num_tokens)
        if num_tokens < 0 or num_tokens > self.capacity:
            raise ValueError(
                f"num_tokens={num_tokens} exceeds MHC workspace capacity {self.capacity}"
            )
        return MHCWorkspace(
            partials=self.partials[:num_tokens],
            y=self.y[:num_tokens],
            post=self.post[:num_tokens],
            comb=self.comb[:num_tokens],
            out=self.out[:num_tokens],
            split_k=self.split_k,
        )

    def bind(
        self,
        *,
        tokens: int | None = None,
        out: torch.Tensor | None = None,
    ) -> "B12XMHCBinding":
        return build_mhc_binding(workspace=self, tokens=tokens, out=out)


@dataclass(frozen=True, kw_only=True)
class B12XMHCBinding:
    partials: torch.Tensor | None = None
    y: torch.Tensor | None = None
    post_buffer: torch.Tensor | None = None
    comb_buffer: torch.Tensor | None = None
    out: torch.Tensor | None = None
    split_k: int = MHC_DEFAULT_SPLIT_K

    def post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        prev_post: torch.Tensor,
        prev_comb: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
        block_k: int = MHC_DEFAULT_BLOCK_K,
        block_h: int = MHC_DEFAULT_BLOCK_H,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return b12x_mhc_post_pre(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            binding=self,
            block_k=block_k,
            block_h=block_h,
        )


@dataclass(frozen=True, kw_only=True)
class B12XMHCScratchCaps:
    device: torch.device | str
    max_tokens: int
    hidden_size: int
    dtype: torch.dtype = torch.bfloat16
    split_k: int = MHC_DEFAULT_SPLIT_K

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        object.__setattr__(self, "hidden_size", max(int(self.hidden_size), 1))
        object.__setattr__(self, "split_k", max(int(self.split_k), 1))
        if self.dtype != torch.bfloat16:
            raise ValueError(f"mHC scratch currently supports torch.bfloat16 outputs, got {self.dtype}")


@dataclass(frozen=True)
class _MHCScratchLayout:
    nbytes: int
    partials_offset_bytes: int


@dataclass(frozen=True)
class B12XMHCScratchPlan:
    caps: B12XMHCScratchCaps
    layout: _MHCScratchLayout
    _scratch_specs: tuple[B12XScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[B12XScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def make_pre_workspace(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    ) -> MHCPreWorkspace:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="mHC",
        )
        max_tokens = int(self.caps.max_tokens)
        split_k = int(self.caps.split_k)
        partials, _ = _materialize_arena_view(
            scratch_storage,
            offset_bytes=self.layout.partials_offset_bytes,
            shape=(max_tokens, split_k, MHC_PARTIALS),
            dtype=torch.float32,
        )
        return MHCPreWorkspace(partials=partials, split_k=split_k)

    def make_workspace(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    ) -> MHCPreWorkspace:
        return self.make_pre_workspace(scratch=scratch)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        tokens: int | None = None,
        y: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> B12XMHCBinding:
        workspace = self.make_pre_workspace(scratch=scratch)
        live_tokens = int(self.caps.max_tokens) if tokens is None else int(tokens)
        if live_tokens < 0 or live_tokens > int(self.caps.max_tokens):
            raise ValueError(
                f"tokens={live_tokens} exceeds MHC scratch capacity {self.caps.max_tokens}"
            )
        partials = workspace.partials[:live_tokens]
        _validate_mhc_binding_views(
            partials=partials,
            y=y,
            post=post,
            comb=comb,
            out=out,
            tokens=live_tokens,
            hidden_size=int(self.caps.hidden_size),
            split_k=int(self.caps.split_k),
            dtype=self.caps.dtype,
            device=self.caps.device,
        )
        return B12XMHCBinding(
            partials=partials,
            y=y,
            post_buffer=post,
            comb_buffer=comb,
            out=out,
            split_k=int(self.caps.split_k),
        )


def build_mhc_binding(
    *,
    workspace: MHCWorkspace,
    tokens: int | None = None,
    out: torch.Tensor | None = None,
) -> B12XMHCBinding:
    if not isinstance(workspace, MHCWorkspace):
        raise TypeError("workspace must be an MHCWorkspace")
    live_tokens = int(workspace.capacity) if tokens is None else int(tokens)
    if live_tokens < 0 or live_tokens > int(workspace.capacity):
        raise ValueError(
            f"tokens={live_tokens} exceeds MHC workspace capacity {workspace.capacity}"
        )
    live = workspace if live_tokens == int(workspace.capacity) else workspace.slice(live_tokens)
    if out is None:
        out = live.out
    _validate_mhc_binding_views(
        partials=live.partials,
        y=live.y,
        post=live.post,
        comb=live.comb,
        out=out,
        tokens=live_tokens,
        hidden_size=int(workspace.hidden_size),
        split_k=int(live.split_k),
        dtype=live.out.dtype,
        device=live.out.device,
    )
    return B12XMHCBinding(
        partials=live.partials,
        y=live.y,
        post_buffer=live.post,
        comb_buffer=live.comb,
        out=out,
        split_k=int(live.split_k),
    )


def _validate_optional_view(
    tensor: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> None:
    if tensor is None:
        return
    if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(
            f"{name} must have shape {shape}, dtype {dtype}, and device {device}; "
            f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
        )
    _require_contiguous(tensor, name=name)


def _validate_mhc_binding_views(
    *,
    partials: torch.Tensor | None,
    y: torch.Tensor | None,
    post: torch.Tensor | None,
    comb: torch.Tensor | None,
    out: torch.Tensor | None,
    tokens: int,
    hidden_size: int,
    split_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if partials is not None:
        _validate_optional_view(
            partials,
            shape=(tokens, split_k, MHC_PARTIALS),
            dtype=torch.float32,
            device=device,
            name="mHC partials",
        )
    _validate_optional_view(
        y,
        shape=(tokens, hidden_size),
        dtype=dtype,
        device=device,
        name="mHC y",
    )
    _validate_optional_view(
        post,
        shape=(tokens, MHC_MULT),
        dtype=torch.float32,
        device=device,
        name="mHC post",
    )
    _validate_optional_view(
        comb,
        shape=(tokens, MHC_MULT, MHC_MULT),
        dtype=torch.float32,
        device=device,
        name="mHC comb",
    )
    _validate_optional_view(
        out,
        shape=(tokens, MHC_MULT, hidden_size),
        dtype=dtype,
        device=device,
        name="mHC out",
    )


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _slice_capacity_view(
    tensor: torch.Tensor | None,
    *,
    tokens: int,
    tail_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    expected = (tokens, *tail_shape)
    if tuple(tensor.shape) == expected:
        return tensor
    if (
        tensor.ndim == len(expected)
        and int(tensor.shape[0]) >= tokens
        and tuple(tensor.shape[1:]) == tail_shape
        and tensor.dtype == dtype
        and tensor.device == device
    ):
        return tensor[:tokens]
    raise ValueError(
        f"{name} must have shape {expected} or capacity >= {tokens} with tail "
        f"{tail_shape}, dtype {dtype}, and device {device}; got "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    )


def _layout_mhc_scratch(caps: B12XMHCScratchCaps) -> _MHCScratchLayout:
    cursor = 0

    def reserve(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[int, int]:
        nonlocal cursor
        offset = _align_up(cursor, max(_ARENA_ALIGN_BYTES, _dtype_nbytes(dtype)))
        cursor = offset + _shape_numel(shape) * _dtype_nbytes(dtype)
        return offset, cursor

    partials_offset_bytes, _ = reserve(
        (int(caps.max_tokens), int(caps.split_k), MHC_PARTIALS),
        torch.float32,
    )
    return _MHCScratchLayout(
        nbytes=cursor,
        partials_offset_bytes=partials_offset_bytes,
    )


def plan_mhc_scratch(caps: B12XMHCScratchCaps) -> B12XMHCScratchPlan:
    layout = _layout_mhc_scratch(caps)
    return B12XMHCScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "mhc.scratch",
                nbytes=layout.nbytes,
                device=caps.device,
            ),
        ),
    )


def _capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _require_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_pre_inputs(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> tuple[int, int, int]:
    if residual.device.type != "cuda":
        raise ValueError("residual must be a CUDA tensor")
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be torch.bfloat16, got {residual.dtype}")
    if residual.ndim != 3:
        raise ValueError(f"residual must be rank-3 [tokens, 4, hidden], got {tuple(residual.shape)}")
    tokens, hc_mult, hidden_size = map(int, residual.shape)
    if hc_mult != MHC_MULT:
        raise ValueError(f"residual hc dimension must be {MHC_MULT}, got {hc_mult}")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if fn.dtype != torch.float32:
        raise ValueError(f"fn must be torch.float32, got {fn.dtype}")
    if fn.shape != (MHC_MIXES, MHC_MULT * hidden_size):
        raise ValueError(
            f"fn must have shape {(MHC_MIXES, MHC_MULT * hidden_size)}, got {tuple(fn.shape)}"
        )
    if hc_scale.dtype != torch.float32 or tuple(hc_scale.shape) != (3,):
        raise ValueError(f"hc_scale must be float32 shape [3], got {hc_scale.dtype} {tuple(hc_scale.shape)}")
    if hc_base.dtype != torch.float32 or tuple(hc_base.shape) != (MHC_MIXES,):
        raise ValueError(
            f"hc_base must be float32 shape [{MHC_MIXES}], got {hc_base.dtype} {tuple(hc_base.shape)}"
        )
    if fn.device != residual.device or hc_scale.device != residual.device or hc_base.device != residual.device:
        raise ValueError("fn, hc_scale, and hc_base must be on the residual device")
    _require_contiguous(residual, name="residual")
    _require_contiguous(fn, name="fn")
    _require_contiguous(hc_scale, name="hc_scale")
    _require_contiguous(hc_base, name="hc_base")
    return tokens, hidden_size, MHC_MULT * hidden_size


def _validate_norm_weight(
    norm_weight: torch.Tensor | None,
    *,
    hidden_size: int,
    device: torch.device,
) -> None:
    if norm_weight is None:
        return
    if norm_weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"norm_weight must be bf16 or fp32, got {norm_weight.dtype}")
    if norm_weight.device != device:
        raise ValueError("norm_weight must be on the residual device")
    if tuple(norm_weight.shape) != (hidden_size,):
        raise ValueError(
            f"norm_weight must have shape {(hidden_size,)}, got {tuple(norm_weight.shape)}"
        )
    _require_contiguous(norm_weight, name="norm_weight")


def _canonicalize_post_mix_input(
    post: torch.Tensor,
    *,
    tokens: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if post.dtype != torch.float32 or post.device != device:
        raise ValueError(
            f"{name} must be float32 on device {device}, got {post.dtype} on {post.device}"
        )
    if tuple(post.shape) == (tokens, MHC_MULT, 1):
        post = post.squeeze(-1)
    elif tuple(post.shape) != (tokens, MHC_MULT):
        raise ValueError(
            f"{name} must have shape {(tokens, MHC_MULT)} or {(tokens, MHC_MULT, 1)}, "
            f"got {tuple(post.shape)}"
        )
    _require_contiguous(post, name=name)
    return post


def empty_mhc_pre_workspace(
    *,
    num_tokens: int,
    split_k: int = MHC_DEFAULT_SPLIT_K,
    device: torch.device | str | None = None,
) -> MHCPreWorkspace:
    device_obj = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    if int(num_tokens) < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if int(split_k) <= 0:
        raise ValueError(f"split_k must be positive, got {split_k}")
    partials = torch.empty(
        (int(num_tokens), int(split_k), MHC_PARTIALS),
        device=device_obj,
        dtype=torch.float32,
    )
    return MHCPreWorkspace(partials=partials, split_k=int(split_k))


def empty_mhc_workspace(
    *,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    split_k: int = MHC_DEFAULT_SPLIT_K,
    device: torch.device | str | None = None,
) -> MHCWorkspace:
    device_obj = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    num_tokens = int(num_tokens)
    hidden_size = int(hidden_size)
    split_k = int(split_k)
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if split_k <= 0:
        raise ValueError(f"split_k must be positive, got {split_k}")
    if dtype != torch.bfloat16:
        raise ValueError(f"mHC workspace currently supports torch.bfloat16 outputs, got {dtype}")
    partials = torch.empty(
        (num_tokens, split_k, MHC_PARTIALS),
        device=device_obj,
        dtype=torch.float32,
    )
    y = torch.empty((num_tokens, hidden_size), device=device_obj, dtype=dtype)
    post = torch.empty((num_tokens, MHC_MULT), device=device_obj, dtype=torch.float32)
    comb = torch.empty(
        (num_tokens, MHC_MULT, MHC_MULT),
        device=device_obj,
        dtype=torch.float32,
    )
    out = torch.empty(
        (num_tokens, MHC_MULT, hidden_size),
        device=device_obj,
        dtype=dtype,
    )
    return MHCWorkspace(
        partials=partials,
        y=y,
        post=post,
        comb=comb,
        out=out,
        split_k=split_k,
    )


def mhc_workspace_nbytes(
    *,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    split_k: int = MHC_DEFAULT_SPLIT_K,
) -> int:
    num_tokens = max(int(num_tokens), 0)
    hidden_size = max(int(hidden_size), 1)
    split_k = max(int(split_k), 1)
    dtype_nbytes = torch.empty((), dtype=dtype).element_size()
    return (
        num_tokens * split_k * MHC_PARTIALS * torch.empty((), dtype=torch.float32).element_size()
        + num_tokens * hidden_size * dtype_nbytes
        + num_tokens * MHC_MULT * torch.empty((), dtype=torch.float32).element_size()
        + num_tokens * MHC_MULT * MHC_MULT * torch.empty((), dtype=torch.float32).element_size()
        + num_tokens * MHC_MULT * hidden_size * dtype_nbytes
    )


def _workspace_views_for_pre(
    workspace: MHCWorkspace,
    *,
    tokens: int,
    hidden_size: int,
    split_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(workspace.split_k) != split_k:
        raise ValueError(f"workspace split_k={workspace.split_k} does not match split_k={split_k}")
    if workspace.capacity < tokens:
        raise ValueError(
            f"MHC workspace capacity {workspace.capacity} is smaller than requested tokens={tokens}"
        )
    if workspace.hidden_size != hidden_size:
        raise ValueError(
            f"MHC workspace hidden_size={workspace.hidden_size} does not match requested hidden_size={hidden_size}"
        )
    sliced = workspace.slice(tokens)
    partials = sliced.partials
    y_out = sliced.y
    post_out = sliced.post
    comb_out = sliced.comb
    if partials.dtype != torch.float32 or partials.device != device:
        raise ValueError("MHC workspace partials must be float32 on the residual device")
    if y_out.dtype != dtype or y_out.device != device:
        raise ValueError("MHC workspace y must match residual dtype and device")
    if post_out.dtype != torch.float32 or post_out.device != device:
        raise ValueError("MHC workspace post must be float32 on the residual device")
    if comb_out.dtype != torch.float32 or comb_out.device != device:
        raise ValueError("MHC workspace comb must be float32 on the residual device")
    return partials, y_out, post_out, comb_out


def b12x_mhc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    workspace: MHCWorkspace | MHCPreWorkspace | torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    y_out: torch.Tensor | None = None,
    post_out: torch.Tensor | None = None,
    comb_out: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    binding: B12XMHCBinding | None = None,
    split_k: int = MHC_DEFAULT_SPLIT_K,
    block_k: int = MHC_DEFAULT_BLOCK_K,
    block_h: int = MHC_DEFAULT_BLOCK_H,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("workspace", workspace),
                ("residual_out", residual_out),
                ("y_out", y_out),
                ("post_out", post_out),
                ("comb_out", comb_out),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "mHC binding owns workspace and output buffers; "
                f"do not also pass {', '.join(extras)}"
            )
        workspace = binding.partials
        residual_out = binding.out
        y_out = binding.y
        post_out = binding.post_buffer
        comb_out = binding.comb_buffer
        split_k = int(binding.split_k)

    tokens, hidden_size, _ = _validate_pre_inputs(residual, fn, hc_scale, hc_base)
    _validate_norm_weight(norm_weight, hidden_size=hidden_size, device=residual.device)
    if x.dtype != residual.dtype or x.dtype != torch.bfloat16:
        raise ValueError(f"x and residual must both be torch.bfloat16, got {x.dtype} and {residual.dtype}")
    if x.ndim != 2 or tuple(x.shape) != (tokens, hidden_size):
        raise ValueError(f"x must have shape {(tokens, hidden_size)}, got {tuple(x.shape)}")
    if x.device != residual.device:
        raise ValueError("x, residual, fn, hc_scale, and hc_base must be on the same device")
    prev_post = _canonicalize_post_mix_input(
        prev_post,
        tokens=tokens,
        device=residual.device,
        name="prev_post",
    )
    if prev_comb.dtype != torch.float32 or tuple(prev_comb.shape) != (tokens, MHC_MULT, MHC_MULT):
        raise ValueError(
            f"prev_comb must be float32 shape {(tokens, MHC_MULT, MHC_MULT)}, "
            f"got {prev_comb.dtype} {tuple(prev_comb.shape)}"
        )
    if prev_comb.device != residual.device:
        raise ValueError("prev_comb must be on the residual device")
    _require_contiguous(x, name="x")
    _require_contiguous(prev_comb, name="prev_comb")

    split_k = int(split_k)
    block_k = int(block_k)
    block_h = int(block_h)
    sinkhorn_iters = int(sinkhorn_iters)
    if sinkhorn_iters <= 0:
        raise ValueError(f"sinkhorn_iters must be positive, got {sinkhorn_iters}")
    if split_k <= 0:
        raise ValueError(f"split_k must be positive, got {split_k}")
    if block_k <= 0:
        raise ValueError(f"block_k must be positive, got {block_k}")
    if block_h <= 0:
        raise ValueError(f"block_h must be positive, got {block_h}")

    capture = _capture_active(residual.device)
    if workspace is None:
        # The Gram post_pre needs a partials scratch buffer (the launch boundary
        # between the partial pass and the multi-CTA finalize). Callers that
        # capture CUDA graphs must own it; outside capture we allocate here.
        if capture:
            raise ValueError(
                "b12x_mhc_post_pre requires a caller-owned workspace (partials "
                "scratch) during CUDA graph capture"
            )
        partials = torch.empty(
            (tokens, split_k, MHC_PARTIALS), dtype=torch.float32, device=residual.device
        )
    elif isinstance(workspace, MHCWorkspace):
        partials, workspace_y, workspace_post, workspace_comb = _workspace_views_for_pre(
            workspace,
            tokens=tokens,
            hidden_size=hidden_size,
            split_k=split_k,
            dtype=residual.dtype,
            device=residual.device,
        )
        sliced_workspace = workspace.slice(tokens)
        if residual_out is None:
            residual_out = sliced_workspace.out
        if y_out is None:
            y_out = workspace_y
        if post_out is None:
            post_out = workspace_post
        if comb_out is None:
            comb_out = workspace_comb
    elif isinstance(workspace, MHCPreWorkspace):
        if int(workspace.split_k) != split_k:
            raise ValueError(f"workspace split_k={workspace.split_k} does not match split_k={split_k}")
        partials = workspace.partials
    else:
        partials = workspace
    # The source-tile CuTe post_pre path uses the shared workspace partials as
    # the launch boundary between post+partial reduction and y/post/comb finalize.
    if partials is not None:
        partials = _slice_capacity_view(
            partials,
            tokens=tokens,
            tail_shape=(split_k, MHC_PARTIALS),
            dtype=torch.float32,
            device=residual.device,
            name="workspace partials",
        )
        if partials.dtype != torch.float32 or partials.device != residual.device:
            raise ValueError("workspace partials must be float32 on the residual device")
        _require_contiguous(partials, name="workspace partials")

    if residual_out is None:
        if capture:
            raise ValueError("b12x_mhc_post_pre requires caller-owned residual_out during CUDA graph capture")
        residual_out = torch.empty_like(residual)
    else:
        residual_out = _slice_capacity_view(
            residual_out,
            tokens=tokens,
            tail_shape=(MHC_MULT, hidden_size),
            dtype=residual.dtype,
            device=residual.device,
            name="residual_out",
        )
    if y_out is None:
        if capture:
            raise ValueError("b12x_mhc_post_pre requires caller-owned y_out during CUDA graph capture")
        y_out = torch.empty((tokens, hidden_size), dtype=residual.dtype, device=residual.device)
    else:
        y_out = _slice_capacity_view(
            y_out,
            tokens=tokens,
            tail_shape=(hidden_size,),
            dtype=residual.dtype,
            device=residual.device,
            name="y_out",
        )
    if post_out is None:
        if capture:
            raise ValueError("b12x_mhc_post_pre requires caller-owned post_out during CUDA graph capture")
        post_out = torch.empty((tokens, MHC_MULT), dtype=torch.float32, device=residual.device)
    else:
        post_out = _slice_capacity_view(
            post_out,
            tokens=tokens,
            tail_shape=(MHC_MULT,),
            dtype=torch.float32,
            device=residual.device,
            name="post_out",
        )
    if comb_out is None:
        if capture:
            raise ValueError("b12x_mhc_post_pre requires caller-owned comb_out during CUDA graph capture")
        comb_out = torch.empty((tokens, MHC_MULT, MHC_MULT), dtype=torch.float32, device=residual.device)
    else:
        comb_out = _slice_capacity_view(
            comb_out,
            tokens=tokens,
            tail_shape=(MHC_MULT, MHC_MULT),
            dtype=torch.float32,
            device=residual.device,
            name="comb_out",
        )

    if residual_out.shape != residual.shape or residual_out.dtype != residual.dtype or residual_out.device != residual.device:
        raise ValueError("residual_out must match residual shape, dtype, and device")
    if y_out.shape != (tokens, hidden_size) or y_out.dtype != residual.dtype or y_out.device != residual.device:
        raise ValueError("y_out must match shape [tokens, hidden_size], residual dtype, and residual device")
    if post_out.shape != (tokens, MHC_MULT) or post_out.dtype != torch.float32 or post_out.device != residual.device:
        raise ValueError("post_out must match shape [tokens, 4], dtype float32, and residual device")
    if comb_out.shape != (tokens, MHC_MULT, MHC_MULT) or comb_out.dtype != torch.float32 or comb_out.device != residual.device:
        raise ValueError("comb_out must match shape [tokens, 4, 4], dtype float32, and residual device")
    _require_contiguous(residual_out, name="residual_out")
    _require_contiguous(y_out, name="y_out")
    _require_contiguous(post_out, name="post_out")
    _require_contiguous(comb_out, name="comb_out")

    if tokens == 0:
        return residual_out, post_out, comb_out, y_out

    if (
        partials is not None
        and hidden_size == 4096
        and split_k == MHC_DEFAULT_SPLIT_K
        and block_k == MHC_DEFAULT_BLOCK_K
        and block_h == MHC_DEFAULT_BLOCK_H
        and float(rms_eps) == 1.0e-6
        and float(hc_eps) == 1.0e-6
        and sinkhorn_iters == 20
    ):
        # The Gram-trick fused post_pre is THE mHC decode post_pre kernel: one
        # partial pass (POST + the fn@flat reduction + residual_out's 4x4 Gram)
        # feeds a multi-CTA finalize whose RMSNorm uses sum_h y^2 = pre^T G pre
        # (no per-hidden reduction, so no single-CTA bottleneck). Sinkhorn runs
        # the caller's iteration count (the full 20, matching vLLM and the
        # reference; Sinkhorn is ~0.015us/iter so the cost is negligible). The
        # Gram + RMSNorm are skipped when there is no fused norm_weight.
        from b12x.integration.residual_kernels import (
            run_mhc_finalize_gram,
            run_mhc_post_pre_partial,
        )

        run_mhc_post_pre_partial(
            x=x,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=fn,
            partials=partials,
            out=residual_out,
            compute_gram=norm_weight is not None,
        )
        run_mhc_finalize_gram(
            residual=residual_out,
            partials=partials,
            scale=hc_scale,
            bias=hc_base,
            y=y_out,
            post=post_out,
            comb=comb_out,
            rms_eps=float(rms_eps),
            hc_eps=float(hc_eps),
            sinkhorn_iters=sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=float(norm_eps),
        )
        return residual_out, post_out, comb_out, y_out

    raise ValueError(
        "b12x_mhc_post_pre is served only by the fused Gram kernel, which "
        "supports the decode config "
        f"(hidden_size=4096, split_k={MHC_DEFAULT_SPLIT_K}, "
        f"block_k={MHC_DEFAULT_BLOCK_K}, block_h={MHC_DEFAULT_BLOCK_H}, "
        "sinkhorn_iters=20); got "
        f"hidden_size={hidden_size}, split_k={split_k}, block_k={block_k}, "
        f"block_h={block_h}, sinkhorn_iters={sinkhorn_iters}"
    )


__all__ = [
    "B12XMHCBinding",
    "B12XMHCScratchCaps",
    "B12XMHCScratchPlan",
    "MHC_DEFAULT_BLOCK_H",
    "MHC_DEFAULT_BLOCK_K",
    "MHC_DEFAULT_SPLIT_K",
    "MHC_MULT",
    "MHC_MIXES",
    "MHC_PARTIALS",
    "MHCWorkspace",
    "MHCPreWorkspace",
    "build_mhc_binding",
    "b12x_mhc_post_pre",
    "empty_mhc_workspace",
    "empty_mhc_pre_workspace",
    "mhc_workspace_nbytes",
    "plan_mhc_scratch",
]
