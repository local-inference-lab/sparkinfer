"""Fused mHC residual helpers for DeepSeek-style residual mixing.

The pre path is deliberately parallel: a split-K stage computes the 24 mHC
projection terms plus the RMS square sum, and a fused finalize stage performs
the Sinkhorn normalization and residual collapse.  There is no single-CTA
fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


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


@triton.jit
def _sigmoid_f32(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _pick_mix(mixes, mix_ids, idx: tl.constexpr):
    return tl.sum(tl.where(mix_ids == idx, mixes, 0.0), axis=0)


@triton.jit
def _mhc_pre_partial_kernel(
    residual,
    fn,
    partials,
    total_k: tl.constexpr,
    split_k: tl.constexpr,
    split_size: tl.constexpr,
    block_k: tl.constexpr,
    mix_block: tl.constexpr,
):
    token = tl.program_id(0)
    split = tl.program_id(1)

    offs = tl.arange(0, block_k)
    mix_ids = tl.arange(0, mix_block)
    acc = tl.zeros((mix_block,), tl.float32)
    sqsum = tl.full((), 0.0, tl.float32)

    for rel in tl.static_range(0, split_size, block_k):
        k = split * split_size + rel + offs
        x = tl.load(residual + token * total_k + k).to(tl.float32)
        sqsum += tl.sum(x * x, axis=0)
        weights = tl.load(
            fn + mix_ids[:, None] * total_k + k[None, :],
            mask=mix_ids[:, None] < 24,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(weights * x[None, :], axis=1)

    base = (token * split_k + split) * 25
    tl.store(partials + base, sqsum)
    tl.store(
        partials + base + 1 + mix_ids,
        acc,
        mask=mix_ids < 24,
    )


@triton.jit
def _mhc_pre_finalize_kernel(
    residual,
    partials,
    scale,
    bias,
    y,
    post,
    comb,
    hidden_size: tl.constexpr,
    total_k: tl.constexpr,
    split_k: tl.constexpr,
    split_block: tl.constexpr,
    block_h: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    mix_block: tl.constexpr,
):
    token = tl.program_id(0)
    tile_h = tl.program_id(1)

    split_ids = tl.arange(0, split_block)
    split_mask = split_ids < split_k
    partial_base = (token * split_k + split_ids) * 25
    sqsum = tl.sum(
        tl.load(partials + partial_base, mask=split_mask, other=0.0),
        axis=0,
    )

    mix_ids = tl.arange(0, mix_block)
    mix_partials = tl.load(
        partials + partial_base[None, :] + 1 + mix_ids[:, None],
        mask=(mix_ids[:, None] < 24) & split_mask[None, :],
        other=0.0,
    )
    mixes = tl.sum(mix_partials, axis=1)
    inv_rms = tl.rsqrt(sqsum / total_k + rms_eps)
    mixes = mixes * inv_rms

    s0 = tl.load(scale + 0).to(tl.float32)
    s1 = tl.load(scale + 1).to(tl.float32)
    s2 = tl.load(scale + 2).to(tl.float32)

    pre0 = _sigmoid_f32(_pick_mix(mixes, mix_ids, 0) * s0 + tl.load(bias + 0)) + hc_eps
    pre1 = _sigmoid_f32(_pick_mix(mixes, mix_ids, 1) * s0 + tl.load(bias + 1)) + hc_eps
    pre2 = _sigmoid_f32(_pick_mix(mixes, mix_ids, 2) * s0 + tl.load(bias + 2)) + hc_eps
    pre3 = _sigmoid_f32(_pick_mix(mixes, mix_ids, 3) * s0 + tl.load(bias + 3)) + hc_eps

    post0 = 2.0 * _sigmoid_f32(_pick_mix(mixes, mix_ids, 4) * s1 + tl.load(bias + 4))
    post1 = 2.0 * _sigmoid_f32(_pick_mix(mixes, mix_ids, 5) * s1 + tl.load(bias + 5))
    post2 = 2.0 * _sigmoid_f32(_pick_mix(mixes, mix_ids, 6) * s1 + tl.load(bias + 6))
    post3 = 2.0 * _sigmoid_f32(_pick_mix(mixes, mix_ids, 7) * s1 + tl.load(bias + 7))

    c00 = _pick_mix(mixes, mix_ids, 8) * s2 + tl.load(bias + 8)
    c01 = _pick_mix(mixes, mix_ids, 9) * s2 + tl.load(bias + 9)
    c02 = _pick_mix(mixes, mix_ids, 10) * s2 + tl.load(bias + 10)
    c03 = _pick_mix(mixes, mix_ids, 11) * s2 + tl.load(bias + 11)
    c10 = _pick_mix(mixes, mix_ids, 12) * s2 + tl.load(bias + 12)
    c11 = _pick_mix(mixes, mix_ids, 13) * s2 + tl.load(bias + 13)
    c12 = _pick_mix(mixes, mix_ids, 14) * s2 + tl.load(bias + 14)
    c13 = _pick_mix(mixes, mix_ids, 15) * s2 + tl.load(bias + 15)
    c20 = _pick_mix(mixes, mix_ids, 16) * s2 + tl.load(bias + 16)
    c21 = _pick_mix(mixes, mix_ids, 17) * s2 + tl.load(bias + 17)
    c22 = _pick_mix(mixes, mix_ids, 18) * s2 + tl.load(bias + 18)
    c23 = _pick_mix(mixes, mix_ids, 19) * s2 + tl.load(bias + 19)
    c30 = _pick_mix(mixes, mix_ids, 20) * s2 + tl.load(bias + 20)
    c31 = _pick_mix(mixes, mix_ids, 21) * s2 + tl.load(bias + 21)
    c32 = _pick_mix(mixes, mix_ids, 22) * s2 + tl.load(bias + 22)
    c33 = _pick_mix(mixes, mix_ids, 23) * s2 + tl.load(bias + 23)

    m0 = tl.maximum(tl.maximum(c00, c01), tl.maximum(c02, c03))
    m1 = tl.maximum(tl.maximum(c10, c11), tl.maximum(c12, c13))
    m2 = tl.maximum(tl.maximum(c20, c21), tl.maximum(c22, c23))
    m3 = tl.maximum(tl.maximum(c30, c31), tl.maximum(c32, c33))
    c00 = tl.exp(c00 - m0)
    c01 = tl.exp(c01 - m0)
    c02 = tl.exp(c02 - m0)
    c03 = tl.exp(c03 - m0)
    c10 = tl.exp(c10 - m1)
    c11 = tl.exp(c11 - m1)
    c12 = tl.exp(c12 - m1)
    c13 = tl.exp(c13 - m1)
    c20 = tl.exp(c20 - m2)
    c21 = tl.exp(c21 - m2)
    c22 = tl.exp(c22 - m2)
    c23 = tl.exp(c23 - m2)
    c30 = tl.exp(c30 - m3)
    c31 = tl.exp(c31 - m3)
    c32 = tl.exp(c32 - m3)
    c33 = tl.exp(c33 - m3)
    r0 = c00 + c01 + c02 + c03
    r1 = c10 + c11 + c12 + c13
    r2 = c20 + c21 + c22 + c23
    r3 = c30 + c31 + c32 + c33
    c00 = c00 / r0 + hc_eps
    c01 = c01 / r0 + hc_eps
    c02 = c02 / r0 + hc_eps
    c03 = c03 / r0 + hc_eps
    c10 = c10 / r1 + hc_eps
    c11 = c11 / r1 + hc_eps
    c12 = c12 / r1 + hc_eps
    c13 = c13 / r1 + hc_eps
    c20 = c20 / r2 + hc_eps
    c21 = c21 / r2 + hc_eps
    c22 = c22 / r2 + hc_eps
    c23 = c23 / r2 + hc_eps
    c30 = c30 / r3 + hc_eps
    c31 = c31 / r3 + hc_eps
    c32 = c32 / r3 + hc_eps
    c33 = c33 / r3 + hc_eps

    col0 = c00 + c10 + c20 + c30 + hc_eps
    col1 = c01 + c11 + c21 + c31 + hc_eps
    col2 = c02 + c12 + c22 + c32 + hc_eps
    col3 = c03 + c13 + c23 + c33 + hc_eps
    c00 = c00 / col0
    c10 = c10 / col0
    c20 = c20 / col0
    c30 = c30 / col0
    c01 = c01 / col1
    c11 = c11 / col1
    c21 = c21 / col1
    c31 = c31 / col1
    c02 = c02 / col2
    c12 = c12 / col2
    c22 = c22 / col2
    c32 = c32 / col2
    c03 = c03 / col3
    c13 = c13 / col3
    c23 = c23 / col3
    c33 = c33 / col3

    for _ in tl.static_range(0, sinkhorn_iters - 1):
        r0 = c00 + c01 + c02 + c03 + hc_eps
        r1 = c10 + c11 + c12 + c13 + hc_eps
        r2 = c20 + c21 + c22 + c23 + hc_eps
        r3 = c30 + c31 + c32 + c33 + hc_eps
        c00 = c00 / r0
        c01 = c01 / r0
        c02 = c02 / r0
        c03 = c03 / r0
        c10 = c10 / r1
        c11 = c11 / r1
        c12 = c12 / r1
        c13 = c13 / r1
        c20 = c20 / r2
        c21 = c21 / r2
        c22 = c22 / r2
        c23 = c23 / r2
        c30 = c30 / r3
        c31 = c31 / r3
        c32 = c32 / r3
        c33 = c33 / r3

        col0 = c00 + c10 + c20 + c30 + hc_eps
        col1 = c01 + c11 + c21 + c31 + hc_eps
        col2 = c02 + c12 + c22 + c32 + hc_eps
        col3 = c03 + c13 + c23 + c33 + hc_eps
        c00 = c00 / col0
        c10 = c10 / col0
        c20 = c20 / col0
        c30 = c30 / col0
        c01 = c01 / col1
        c11 = c11 / col1
        c21 = c21 / col1
        c31 = c31 / col1
        c02 = c02 / col2
        c12 = c12 / col2
        c22 = c22 / col2
        c32 = c32 / col2
        c03 = c03 / col3
        c13 = c13 / col3
        c23 = c23 / col3
        c33 = c33 / col3

    if tile_h == 0:
        post_base = token * 4
        tl.store(post + post_base + 0, post0)
        tl.store(post + post_base + 1, post1)
        tl.store(post + post_base + 2, post2)
        tl.store(post + post_base + 3, post3)

        comb_base = token * 16
        tl.store(comb + comb_base + 0, c00)
        tl.store(comb + comb_base + 1, c01)
        tl.store(comb + comb_base + 2, c02)
        tl.store(comb + comb_base + 3, c03)
        tl.store(comb + comb_base + 4, c10)
        tl.store(comb + comb_base + 5, c11)
        tl.store(comb + comb_base + 6, c12)
        tl.store(comb + comb_base + 7, c13)
        tl.store(comb + comb_base + 8, c20)
        tl.store(comb + comb_base + 9, c21)
        tl.store(comb + comb_base + 10, c22)
        tl.store(comb + comb_base + 11, c23)
        tl.store(comb + comb_base + 12, c30)
        tl.store(comb + comb_base + 13, c31)
        tl.store(comb + comb_base + 14, c32)
        tl.store(comb + comb_base + 15, c33)

    h = tile_h * block_h + tl.arange(0, block_h)
    mask = h < hidden_size
    token_base = token * total_k
    r0v = tl.load(residual + token_base + h, mask=mask, other=0.0).to(tl.float32)
    r1v = tl.load(residual + token_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
    r2v = tl.load(
        residual + token_base + 2 * hidden_size + h, mask=mask, other=0.0
    ).to(tl.float32)
    r3v = tl.load(
        residual + token_base + 3 * hidden_size + h, mask=mask, other=0.0
    ).to(tl.float32)
    out = pre0 * r0v + pre1 * r1v + pre2 * r2v + pre3 * r3v
    tl.store(y + token * hidden_size + h, out, mask=mask)


@triton.jit
def _mhc_post_kernel(
    x,
    residual,
    post,
    comb,
    out,
    hidden_size: tl.constexpr,
    total_k: tl.constexpr,
    block_h: tl.constexpr,
):
    token = tl.program_id(0)
    tile_h = tl.program_id(1)
    h = tile_h * block_h + tl.arange(0, block_h)
    mask = h < hidden_size

    xh = tl.load(x + token * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
    token_base = token * total_k
    r0 = tl.load(residual + token_base + h, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(residual + token_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(
        residual + token_base + 2 * hidden_size + h, mask=mask, other=0.0
    ).to(tl.float32)
    r3 = tl.load(
        residual + token_base + 3 * hidden_size + h, mask=mask, other=0.0
    ).to(tl.float32)

    post_base = token * 4
    p0 = tl.load(post + post_base + 0).to(tl.float32)
    p1 = tl.load(post + post_base + 1).to(tl.float32)
    p2 = tl.load(post + post_base + 2).to(tl.float32)
    p3 = tl.load(post + post_base + 3).to(tl.float32)

    comb_base = token * 16
    c00 = tl.load(comb + comb_base + 0).to(tl.float32)
    c01 = tl.load(comb + comb_base + 1).to(tl.float32)
    c02 = tl.load(comb + comb_base + 2).to(tl.float32)
    c03 = tl.load(comb + comb_base + 3).to(tl.float32)
    c10 = tl.load(comb + comb_base + 4).to(tl.float32)
    c11 = tl.load(comb + comb_base + 5).to(tl.float32)
    c12 = tl.load(comb + comb_base + 6).to(tl.float32)
    c13 = tl.load(comb + comb_base + 7).to(tl.float32)
    c20 = tl.load(comb + comb_base + 8).to(tl.float32)
    c21 = tl.load(comb + comb_base + 9).to(tl.float32)
    c22 = tl.load(comb + comb_base + 10).to(tl.float32)
    c23 = tl.load(comb + comb_base + 11).to(tl.float32)
    c30 = tl.load(comb + comb_base + 12).to(tl.float32)
    c31 = tl.load(comb + comb_base + 13).to(tl.float32)
    c32 = tl.load(comb + comb_base + 14).to(tl.float32)
    c33 = tl.load(comb + comb_base + 15).to(tl.float32)

    o0 = p0 * xh + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3
    o1 = p1 * xh + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3
    o2 = p2 * xh + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3
    o3 = p3 * xh + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3

    tl.store(out + token_base + h, o0, mask=mask)
    tl.store(out + token_base + hidden_size + h, o1, mask=mask)
    tl.store(out + token_base + 2 * hidden_size + h, o2, mask=mask)
    tl.store(out + token_base + 3 * hidden_size + h, o3, mask=mask)


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


def b12x_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    workspace: MHCWorkspace | MHCPreWorkspace | torch.Tensor | None = None,
    y_out: torch.Tensor | None = None,
    post_out: torch.Tensor | None = None,
    comb_out: torch.Tensor | None = None,
    split_k: int = MHC_DEFAULT_SPLIT_K,
    block_k: int = MHC_DEFAULT_BLOCK_K,
    block_h: int = MHC_DEFAULT_BLOCK_H,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, hidden_size, total_k = _validate_pre_inputs(residual, fn, hc_scale, hc_base)
    split_k = int(split_k)
    block_k = int(block_k)
    block_h = int(block_h)
    sinkhorn_iters = int(sinkhorn_iters)
    if sinkhorn_iters <= 0:
        raise ValueError(f"sinkhorn_iters must be positive, got {sinkhorn_iters}")
    if split_k <= 1:
        raise ValueError("b12x_mhc_pre requires split_k > 1; no single-CTA fallback is provided")
    if total_k % split_k != 0:
        raise ValueError(f"total_k={total_k} must be divisible by split_k={split_k}")
    split_size = total_k // split_k
    if split_size % block_k != 0:
        raise ValueError(f"split_size={split_size} must be divisible by block_k={block_k}")
    if block_h <= 0:
        raise ValueError(f"block_h must be positive, got {block_h}")

    capture = _capture_active(residual.device)
    if workspace is None:
        if capture:
            raise ValueError("b12x_mhc_pre requires caller-owned workspace during CUDA graph capture")
        workspace_obj = empty_mhc_workspace(
            num_tokens=tokens,
            hidden_size=hidden_size,
            dtype=residual.dtype,
            split_k=split_k,
            device=residual.device,
        )
        partials = workspace_obj.partials
        if y_out is None:
            y_out = workspace_obj.y
        if post_out is None:
            post_out = workspace_obj.post
        if comb_out is None:
            comb_out = workspace_obj.comb
    elif isinstance(workspace, MHCWorkspace):
        partials, workspace_y, workspace_post, workspace_comb = _workspace_views_for_pre(
            workspace,
            tokens=tokens,
            hidden_size=hidden_size,
            split_k=split_k,
            dtype=residual.dtype,
            device=residual.device,
        )
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
    if partials.dtype != torch.float32 or partials.device != residual.device:
        raise ValueError("workspace partials must be float32 on the residual device")
    if tuple(partials.shape) != (tokens, split_k, MHC_PARTIALS):
        raise ValueError(
            f"workspace partials must have shape {(tokens, split_k, MHC_PARTIALS)}, got {tuple(partials.shape)}"
        )
    _require_contiguous(partials, name="workspace partials")

    if y_out is None:
        if capture:
            raise ValueError("b12x_mhc_pre requires caller-owned y_out during CUDA graph capture")
        y_out = torch.empty((tokens, hidden_size), dtype=residual.dtype, device=residual.device)
    if post_out is None:
        if capture:
            raise ValueError("b12x_mhc_pre requires caller-owned post_out during CUDA graph capture")
        post_out = torch.empty((tokens, MHC_MULT), dtype=torch.float32, device=residual.device)
    if comb_out is None:
        if capture:
            raise ValueError("b12x_mhc_pre requires caller-owned comb_out during CUDA graph capture")
        comb_out = torch.empty((tokens, MHC_MULT, MHC_MULT), dtype=torch.float32, device=residual.device)
    if y_out.shape != (tokens, hidden_size) or y_out.dtype != residual.dtype or y_out.device != residual.device:
        raise ValueError("y_out must match shape [tokens, hidden_size], residual dtype, and residual device")
    if post_out.shape != (tokens, MHC_MULT) or post_out.dtype != torch.float32 or post_out.device != residual.device:
        raise ValueError("post_out must match shape [tokens, 4], dtype float32, and residual device")
    if comb_out.shape != (tokens, MHC_MULT, MHC_MULT) or comb_out.dtype != torch.float32 or comb_out.device != residual.device:
        raise ValueError("comb_out must match shape [tokens, 4, 4], dtype float32, and residual device")
    _require_contiguous(y_out, name="y_out")
    _require_contiguous(post_out, name="post_out")
    _require_contiguous(comb_out, name="comb_out")

    if tokens == 0:
        return y_out, post_out, comb_out

    mix_block = triton.next_power_of_2(MHC_MIXES)
    split_block = triton.next_power_of_2(split_k)
    _mhc_pre_partial_kernel[(tokens, split_k)](
        residual,
        fn,
        partials,
        total_k,
        split_k,
        split_size,
        block_k,
        mix_block,
        num_warps=8,
    )
    _mhc_pre_finalize_kernel[(tokens, triton.cdiv(hidden_size, block_h))](
        residual,
        partials,
        hc_scale,
        hc_base,
        y_out,
        post_out,
        comb_out,
        hidden_size,
        total_k,
        split_k,
        split_block,
        block_h,
        float(rms_eps),
        float(hc_eps),
        sinkhorn_iters,
        mix_block,
        num_warps=4,
    )
    return y_out, post_out, comb_out


def b12x_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    workspace: MHCWorkspace | None = None,
    out: torch.Tensor | None = None,
    block_h: int = MHC_DEFAULT_BLOCK_H,
) -> torch.Tensor:
    if residual.device.type != "cuda":
        raise ValueError("residual must be a CUDA tensor")
    if x.dtype != residual.dtype or x.dtype != torch.bfloat16:
        raise ValueError(f"x and residual must both be torch.bfloat16, got {x.dtype} and {residual.dtype}")
    if residual.ndim != 3 or x.ndim != 2:
        raise ValueError(f"expected x [tokens, hidden] and residual [tokens, 4, hidden], got {tuple(x.shape)} {tuple(residual.shape)}")
    tokens, hc_mult, hidden_size = map(int, residual.shape)
    if hc_mult != MHC_MULT:
        raise ValueError(f"residual hc dimension must be {MHC_MULT}, got {hc_mult}")
    if tuple(x.shape) != (tokens, hidden_size):
        raise ValueError(f"x must have shape {(tokens, hidden_size)}, got {tuple(x.shape)}")
    if post.dtype != torch.float32 or tuple(post.shape) != (tokens, MHC_MULT):
        raise ValueError(f"post must be float32 shape {(tokens, MHC_MULT)}, got {post.dtype} {tuple(post.shape)}")
    if comb.dtype != torch.float32 or tuple(comb.shape) != (tokens, MHC_MULT, MHC_MULT):
        raise ValueError(f"comb must be float32 shape {(tokens, MHC_MULT, MHC_MULT)}, got {comb.dtype} {tuple(comb.shape)}")
    if x.device != residual.device or post.device != residual.device or comb.device != residual.device:
        raise ValueError("x, post, comb, and residual must be on the same device")
    _require_contiguous(x, name="x")
    _require_contiguous(residual, name="residual")
    _require_contiguous(post, name="post")
    _require_contiguous(comb, name="comb")
    block_h = int(block_h)
    if block_h <= 0:
        raise ValueError(f"block_h must be positive, got {block_h}")
    if out is None and workspace is not None:
        if workspace.capacity < tokens:
            raise ValueError(
                f"MHC workspace capacity {workspace.capacity} is smaller than requested tokens={tokens}"
            )
        if workspace.hidden_size != hidden_size:
            raise ValueError(
                f"MHC workspace hidden_size={workspace.hidden_size} does not match requested hidden_size={hidden_size}"
            )
        out = workspace.slice(tokens).out
    if out is None:
        if _capture_active(residual.device):
            raise ValueError("b12x_mhc_post requires caller-owned out during CUDA graph capture")
        out = torch.empty_like(residual)
    if out.shape != residual.shape or out.dtype != residual.dtype or out.device != residual.device:
        raise ValueError("out must match residual shape, dtype, and device")
    _require_contiguous(out, name="out")
    if tokens == 0:
        return out

    _mhc_post_kernel[(tokens, triton.cdiv(hidden_size, block_h))](
        x,
        residual,
        post,
        comb,
        out,
        hidden_size,
        MHC_MULT * hidden_size,
        block_h,
        num_warps=4,
    )
    return out


__all__ = [
    "MHC_DEFAULT_BLOCK_H",
    "MHC_DEFAULT_BLOCK_K",
    "MHC_DEFAULT_SPLIT_K",
    "MHC_MULT",
    "MHC_MIXES",
    "MHC_PARTIALS",
    "MHCWorkspace",
    "MHCPreWorkspace",
    "b12x_mhc_post",
    "b12x_mhc_pre",
    "empty_mhc_workspace",
    "empty_mhc_pre_workspace",
    "mhc_workspace_nbytes",
]
