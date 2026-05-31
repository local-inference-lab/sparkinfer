from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from b12x.attention.workspace import (
    _layout_wo_projection,
    _materialize_arena_strided_view,
    _materialize_arena_view,
    _wo_mxfp8_scale_physical_shape,
)
from b12x.gemm.dense import dense_gemm
from b12x.cute.scratch import B12XScratchBufferSpec, scratch_buffer_spec, scratch_tensor

FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
MXFP8_SCALE_VEC_SIZE = 32
MXFP8_SCALE_ROW_TILE = 128
MXFP8_SCALE_K_TILE = 4
WO_A_INPUT_QUANT_GROUP_SIZE = MXFP8_SCALE_VEC_SIZE
_ALPHA_ONE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


@dataclass(frozen=True)
class MXFP8Rows:
    """Row-wise MXFP8 operand and scales for `dense_gemm`.

    `values` has logical dense-GEMM shape `[M, K, L]`, but for `L > 1` it is a
    strided view over physical `[L, M, K]` storage because the current CuTe
    dense kernel consumes raw pointers and reconstructs its own K-major layout.
    `scale_rows` is the compact row/chunk view `[L, M, K/32]`, and `scale_mma`
    is the strided `[32, 4, ceil(M/128), 4, ceil(K/128), L]` view consumed by
    the CuTe kernel.
    """

    values: torch.Tensor
    scale_rows: torch.Tensor
    scale_mma: torch.Tensor


@dataclass(frozen=True)
class WOProjectionMXFP8Weights:
    """MXFP8 WO-A/WO-B weights in the layouts consumed by the two GEMMs."""

    wo_a: MXFP8Rows
    wo_b: MXFP8Rows
    groups: int
    group_width: int
    rank: int
    hidden: int


@dataclass(frozen=True)
class WOProjectionWorkspace:
    """Fixed workspace for one WO projection graph contract."""

    x_q: MXFP8Rows
    tmp: torch.Tensor
    tmp_q: MXFP8Rows
    output: torch.Tensor

    def bind(
        self,
        *,
        source_tgd: torch.Tensor,
        weights: "WOProjectionMXFP8Weights",
        return_3d: bool = False,
        expected_m: int | None = None,
    ) -> "WOProjectionBinding":
        return build_wo_projection_binding(
            workspace=self,
            source_tgd=source_tgd,
            weights=weights,
            return_3d=return_3d,
            expected_m=expected_m,
        )

    def bind_inv_rope(
        self,
        *,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        weights: "WOProjectionMXFP8Weights",
        heads_per_group: int,
        nope_dim: int = 448,
        rope_dim: int = 64,
        return_3d: bool = False,
        expected_m: int | None = None,
    ) -> "WOProjectionInvRopeBinding":
        return build_wo_projection_inv_rope_binding(
            workspace=self,
            o=o,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            weights=weights,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            return_3d=return_3d,
            expected_m=expected_m,
        )


@dataclass(frozen=True)
class _WOProjectionScratchViews:
    x_q: MXFP8Rows
    tmp: torch.Tensor
    tmp_q: MXFP8Rows
    output: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class WOProjectionBinding:
    source_tgd: torch.Tensor
    weights: WOProjectionMXFP8Weights
    x_q: MXFP8Rows
    tmp: torch.Tensor
    tmp_q: MXFP8Rows
    output: torch.Tensor
    return_3d: bool = False
    # DeepGEMM-style regime hint forwarded to the wo_b up-projection (N=hidden,
    # the n>1536 path). None keeps the M-independent default tile.
    expected_m: int | None = None

    def run(self) -> torch.Tensor:
        return wo_projection_mxfp8(binding=self)


@dataclass(frozen=True, kw_only=True)
class WOProjectionInvRopeBinding:
    o: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    weights: WOProjectionMXFP8Weights
    x_q: MXFP8Rows
    tmp: torch.Tensor
    tmp_q: MXFP8Rows
    output: torch.Tensor
    heads_per_group: int
    nope_dim: int = 448
    rope_dim: int = 64
    return_3d: bool = False
    # DeepGEMM-style regime hint forwarded to the wo_b up-projection.
    expected_m: int | None = None

    def run(self) -> torch.Tensor:
        return wo_projection_inv_rope_mxfp8(binding=self)


@dataclass(frozen=True, kw_only=True)
class WOProjectionScratchCaps:
    device: torch.device | str
    max_tokens: int
    groups: int
    group_width: int
    rank: int
    hidden: int
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        object.__setattr__(self, "groups", max(int(self.groups), 1))
        object.__setattr__(self, "group_width", max(int(self.group_width), 1))
        object.__setattr__(self, "rank", max(int(self.rank), 1))
        object.__setattr__(self, "hidden", max(int(self.hidden), 1))
        if self.dtype != torch.bfloat16:
            raise ValueError(
                "WO projection scratch currently supports torch.bfloat16 outputs, "
                f"got {self.dtype}"
            )
        _check_mxfp8_k(self.group_width)
        _check_mxfp8_k(self.rank * self.groups)


@dataclass(frozen=True)
class WOProjectionScratchPlan:
    caps: WOProjectionScratchCaps
    layout: object
    _scratch_specs: tuple[B12XScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[B12XScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        source_tgd: torch.Tensor,
        weights: WOProjectionMXFP8Weights,
        return_3d: bool = False,
        expected_m: int | None = None,
        ) -> WOProjectionBinding:
        tokens = _validate_wo_projection_inputs(source_tgd, weights)
        self._check_live_capacity(tokens=tokens, weights=weights)
        views = self._views_from_scratch(scratch=scratch, tokens=tokens)
        return _build_wo_projection_binding_from_views(
            x_q=views.x_q,
            tmp=views.tmp,
            tmp_q=views.tmp_q,
            output=views.output,
            source_tgd=source_tgd,
            weights=weights,
            return_3d=return_3d,
            expected_m=expected_m,
        )

    def bind_inv_rope(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        weights: WOProjectionMXFP8Weights,
        heads_per_group: int,
        nope_dim: int = 448,
        rope_dim: int = 64,
        return_3d: bool = False,
        expected_m: int | None = None,
    ) -> WOProjectionInvRopeBinding:
        tokens = _validate_wo_projection_inv_rope_inputs(
            o=o,
            weights=weights,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        self._check_live_capacity(tokens=tokens, weights=weights)
        views = self._views_from_scratch(scratch=scratch, tokens=tokens)
        return _build_wo_projection_inv_rope_binding_from_views(
            x_q=views.x_q,
            tmp=views.tmp,
            tmp_q=views.tmp_q,
            output=views.output,
            o=o,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            weights=weights,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            return_3d=return_3d,
            expected_m=expected_m,
        )

    def make_workspace(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        tokens: int | None = None,
    ) -> WOProjectionWorkspace:
        views = self._views_from_scratch(scratch=scratch, tokens=tokens)
        return WOProjectionWorkspace(
            x_q=views.x_q,
            tmp=views.tmp,
            tmp_q=views.tmp_q,
            output=views.output,
        )

    def _views_from_scratch(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        tokens: int | None = None,
    ) -> _WOProjectionScratchViews:
        max_tokens = int(self.caps.max_tokens)
        tokens = max_tokens if tokens is None else int(tokens)
        if tokens <= 0 or tokens > max_tokens:
            raise ValueError(
                f"WO projection tokens={tokens} exceeds scratch capacity {max_tokens}"
            )
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="WO projection",
        )
        groups = int(self.caps.groups)
        group_width = int(self.caps.group_width)
        rank = int(self.caps.rank)
        hidden = int(self.caps.hidden)
        layout = _layout_wo_projection(
            offset_bytes=0,
            tokens=tokens,
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
        )
        if int(layout.nbytes) > int(self.layout.nbytes):
            raise RuntimeError(
                "WO projection workspace layout exceeds reserved scratch "
                f"capacity: requested={layout.nbytes}, reserved={self.layout.nbytes}"
            )

        def mxfp8_rows(
            *,
            values_offset_bytes: int,
            scale_rows_offset_bytes: int,
            scale_mma_offset_bytes: int,
            m: int,
            k: int,
            num_groups: int,
        ) -> MXFP8Rows:
            if num_groups == 1:
                values, _ = _materialize_arena_view(
                    scratch_storage,
                    offset_bytes=values_offset_bytes,
                    shape=(m, k),
                    dtype=torch.float8_e4m3fn,
                )
            else:
                values, _ = _materialize_arena_strided_view(
                    scratch_storage,
                    offset_bytes=values_offset_bytes,
                    shape=(m, k, num_groups),
                    stride=(k, 1, m * k),
                    dtype=torch.float8_e4m3fn,
                )
            scale_rows, _ = _materialize_arena_view(
                scratch_storage,
                offset_bytes=scale_rows_offset_bytes,
                shape=(num_groups, m, k // MXFP8_SCALE_VEC_SIZE),
                dtype=torch.float8_e8m0fnu,
            )
            scale_physical_u8, _ = _materialize_arena_view(
                scratch_storage,
                offset_bytes=scale_mma_offset_bytes,
                shape=_wo_mxfp8_scale_physical_shape(
                    m=m,
                    k=k,
                    num_groups=num_groups,
                ),
                dtype=torch.uint8,
            )
            if m % MXFP8_SCALE_ROW_TILE:
                scale_physical_u8.fill_(127)
            scale_mma = scale_physical_u8.view(torch.float8_e8m0fnu).permute(
                3,
                4,
                1,
                5,
                2,
                0,
            )
            return MXFP8Rows(
                values=values,
                scale_rows=scale_rows,
                scale_mma=scale_mma,
            )

        x_q = mxfp8_rows(
            values_offset_bytes=layout.x_q_values_offset_bytes,
            scale_rows_offset_bytes=layout.x_q_scale_rows_offset_bytes,
            scale_mma_offset_bytes=layout.x_q_scale_mma_offset_bytes,
            m=tokens,
            k=group_width,
            num_groups=groups,
        )
        tmp, _ = _materialize_arena_strided_view(
            scratch_storage,
            offset_bytes=layout.tmp_offset_bytes,
            shape=(tokens, rank, groups),
            stride=(rank, 1, tokens * rank),
            dtype=torch.bfloat16,
        )
        tmp_q = mxfp8_rows(
            values_offset_bytes=layout.tmp_q_values_offset_bytes,
            scale_rows_offset_bytes=layout.tmp_q_scale_rows_offset_bytes,
            scale_mma_offset_bytes=layout.tmp_q_scale_mma_offset_bytes,
            m=tokens,
            k=rank * groups,
            num_groups=1,
        )
        output, _ = _materialize_arena_view(
            scratch_storage,
            offset_bytes=layout.output_offset_bytes,
            shape=(tokens, hidden, 1),
            dtype=torch.bfloat16,
        )
        return _WOProjectionScratchViews(
            x_q=x_q,
            tmp=tmp,
            tmp_q=tmp_q,
            output=output,
        )

    def _check_live_capacity(
        self,
        *,
        tokens: int,
        weights: WOProjectionMXFP8Weights,
    ) -> None:
        if tokens > int(self.caps.max_tokens):
            raise ValueError(
                f"WO projection tokens {tokens} exceed scratch capacity {self.caps.max_tokens}"
            )
        expected = (
            int(self.caps.groups),
            int(self.caps.group_width),
            int(self.caps.rank),
            int(self.caps.hidden),
        )
        actual = (
            int(weights.groups),
            int(weights.group_width),
            int(weights.rank),
            int(weights.hidden),
        )
        if actual != expected:
            raise ValueError(
                "WO projection weights do not match scratch caps: "
                f"weights={actual}, caps={expected}"
            )


def _check_gpu_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")


def _as_grouped_mkl(source: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    _check_gpu_tensor("source", source)
    if source.ndim == 2:
        m, k = source.shape
        grouped = source.reshape(m, k, 1).permute(2, 0, 1).contiguous()
        return grouped, m, k, 1
    if source.ndim == 3:
        m, k, groups = source.shape
        grouped = source.permute(2, 0, 1).contiguous()
        return grouped, m, k, groups
    raise ValueError(f"source must have shape [M,K] or [M,K,L], got {tuple(source.shape)}")


def _check_mxfp8_k(k: int) -> None:
    if k <= 0 or k % 128 != 0:
        raise ValueError(f"MXFP8 dense_gemm K must be a positive multiple of 128, got {k}")


def _cached_alpha_one(device: torch.device | str) -> torch.Tensor:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    key = (resolved.type, resolved.index)
    alpha = _ALPHA_ONE_CACHE.get(key)
    if alpha is None or alpha.device != resolved:
        alpha = torch.ones((1,), dtype=torch.float32, device=resolved)
        _ALPHA_ONE_CACHE[key] = alpha
    return alpha


@triton.jit
def _quantize_grouped_tgd_to_tdg_kernel(
    source,
    values,
    scale_rows,
    scale_mma,
    tokens,
    groups: tl.constexpr,
    group_width: tl.constexpr,
    source_stride_t,
    source_stride_g,
    source_stride_d,
    values_stride_t,
    values_stride_d,
    values_stride_g,
    scale_mma_s0,
    scale_mma_s1,
    scale_mma_s2,
    scale_mma_s3,
    scale_mma_s4,
    scale_mma_s5,
    SCALE_CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    group = tl.program_id(1)
    chunk = tl.program_id(2)
    offs = tl.arange(0, BLOCK)
    d = chunk * BLOCK + offs

    src = tl.load(
        source
        + token * source_stride_t
        + group * source_stride_g
        + d * source_stride_d,
    ).to(tl.float32)
    max_abs = tl.max(tl.abs(src), axis=0)
    quant_scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scale_exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(quant_scale)), -127.0), 127.0)
    scale = tl.exp2(scale_exp)
    scale_u8 = (scale_exp + 127.0).to(tl.uint8)

    tl.store(
        values
        + token * values_stride_t
        + d * values_stride_d
        + group * values_stride_g,
        (src / scale).to(tl.float8e4nv),
    )

    sf_cols = group_width // 32
    tl.store(
        scale_rows + group * tokens * sf_cols + token * sf_cols + chunk,
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
        + group * scale_mma_s5,
        scale_u8,
    )


@triton.jit
def _quantize_attention_inv_rope_to_tdg_kernel(
    o,
    positions,
    cos_sin_cache,
    values,
    scale_rows,
    scale_mma,
    tokens,
    groups: tl.constexpr,
    heads_per_group: tl.constexpr,
    group_width: tl.constexpr,
    o_stride_t,
    o_stride_h,
    o_stride_d,
    cos_sin_stride_pos,
    values_stride_t,
    values_stride_d,
    values_stride_g,
    scale_mma_s0,
    scale_mma_s1,
    scale_mma_s2,
    scale_mma_s3,
    scale_mma_s4,
    scale_mma_s5,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    SCALE_CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    group = tl.program_id(1)
    chunk = tl.program_id(2)
    offs = tl.arange(0, BLOCK)
    d = chunk * BLOCK + offs

    head_in_group = d // HEAD_DIM
    head_d = d - head_in_group * HEAD_DIM
    head = group * heads_per_group + head_in_group

    src = tl.load(
        o + token * o_stride_t + head * o_stride_h + head_d * o_stride_d
    ).to(tl.float32)

    is_rope = head_d >= NOPE_DIM
    rope_local = head_d - NOPE_DIM
    partner_d = NOPE_DIM + (rope_local ^ 1)
    partner = tl.load(
        o + token * o_stride_t + head * o_stride_h + partner_d * o_stride_d,
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)

    pos = tl.load(positions + token)
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cache_base = cos_sin_cache + pos * cos_sin_stride_pos
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE_DIM + cs_idx, mask=is_rope, other=0.0)
    x_add = src * cos_v + partner * sin_v
    x_sub = src * cos_v - partner * sin_v
    rotated = tl.where((rope_local & 1) == 0, x_add, x_sub)
    src = tl.where(is_rope, rotated, src)

    max_abs = tl.max(tl.abs(src), axis=0)
    quant_scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scale_exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(quant_scale)), -127.0), 127.0)
    scale = tl.exp2(scale_exp)
    scale_u8 = (scale_exp + 127.0).to(tl.uint8)

    tl.store(
        values
        + token * values_stride_t
        + d * values_stride_d
        + group * values_stride_g,
        (src / scale).to(tl.float8e4nv),
    )

    sf_cols = group_width // 32
    tl.store(
        scale_rows + group * tokens * sf_cols + token * sf_cols + chunk,
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
        + group * scale_mma_s5,
        scale_u8,
    )


@triton.jit
def _quantize_group_major_trg_to_tk_kernel(
    source,
    values,
    scale_rows,
    scale_mma,
    tokens,
    rank: tl.constexpr,
    groups: tl.constexpr,
    source_stride_t,
    source_stride_r,
    source_stride_g,
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
    cols = chunk * BLOCK + offs
    g = cols // rank
    r = cols - g * rank

    src = tl.load(
        source
        + token * source_stride_t
        + r * source_stride_r
        + g * source_stride_g,
    ).to(tl.float32)
    max_abs = tl.max(tl.abs(src), axis=0)
    safe = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scale_exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(safe)), -127.0), 127.0)
    scale = tl.exp2(scale_exp)
    scale_u8 = (scale_exp + 127.0).to(tl.uint8)

    width = rank * groups
    tl.store(values + token * width + cols, (src / scale).to(tl.float8e4nv))

    sf_cols = width // 32
    tl.store(scale_rows + token * sf_cols + chunk, scale_u8)

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


def empty_dense_gemm_mnl_view(
    m: int,
    n: int,
    l: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate an `[M,N,L]` view backed by dense-GEMM physical `[L,M,N]` storage."""

    if m <= 0 or n <= 0 or l <= 0:
        raise ValueError("m, n, and l must be positive")
    if l == 1:
        return torch.empty((m, n, 1), device=device, dtype=dtype)
    physical = torch.empty((l, m, n), device=device, dtype=dtype)
    return physical.as_strided((m, n, l), (n, 1, m * n))


def empty_mxfp8_rows_for_dense_gemm(
    m: int,
    k: int,
    *,
    num_groups: int = 1,
    device: torch.device | str,
) -> MXFP8Rows:
    """Allocate MXFP8 row storage in the layout consumed by `dense_gemm`."""

    if m <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError("m, k, and num_groups must be positive")
    _check_mxfp8_k(k)
    if num_groups == 1:
        values = torch.empty((m, k), device=device, dtype=torch.float8_e4m3fn)
    else:
        values = empty_dense_gemm_mnl_view(
            m,
            k,
            num_groups,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
    scale_rows_u8 = torch.full(
        (num_groups, m, k // MXFP8_SCALE_VEC_SIZE),
        127,
        dtype=torch.uint8,
        device=device,
    )
    m_tiles = math.ceil(m / MXFP8_SCALE_ROW_TILE)
    k_tiles = math.ceil((k // MXFP8_SCALE_VEC_SIZE) / MXFP8_SCALE_K_TILE)
    scale_physical_u8 = torch.full(
        (num_groups, m_tiles, k_tiles, 32, 4, 4),
        127,
        dtype=torch.uint8,
        device=device,
    )
    scale_mma = scale_physical_u8.view(torch.float8_e8m0fnu).permute(3, 4, 1, 5, 2, 0)
    return MXFP8Rows(
        values=values,
        scale_rows=scale_rows_u8.view(torch.float8_e8m0fnu),
        scale_mma=scale_mma,
    )


def _check_dense_gemm_mnl_view(name: str, tensor: torch.Tensor) -> None:
    _check_gpu_tensor(name, tensor)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [M,N,L], got {tuple(tensor.shape)}")
    m, n, l = tensor.shape
    expected_stride = (n, 1, m * n) if l > 1 else tensor.stride()
    if l > 1 and tensor.stride() != expected_stride:
        raise ValueError(
            f"{name} must be backed by dense-GEMM physical [L,M,N] storage: "
            f"expected stride {expected_stride}, got {tensor.stride()}"
        )


def _scale_u8_from_max_abs(max_abs: torch.Tensor) -> torch.Tensor:
    safe = torch.where(
        max_abs > 0,
        max_abs.to(torch.float32) / FP8_E4M3_MAX,
        torch.ones_like(max_abs, dtype=torch.float32),
    )
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    return (exponent + 127).to(torch.uint8)


def _scale_to_e8m0_u8(scale: torch.Tensor) -> torch.Tensor:
    _check_gpu_tensor("scale", scale)
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.view(torch.uint8)
    if scale.dtype == torch.uint8:
        return scale
    if not scale.is_floating_point():
        raise ValueError(f"scale must be e8m0, uint8, or floating-point, got {scale.dtype}")
    safe = torch.where(
        scale > 0,
        scale.to(torch.float32),
        torch.ones_like(scale, dtype=torch.float32),
    )
    exponent = torch.round(torch.log2(safe)).clamp(-127, 127)
    return (exponent + 127).to(torch.uint8)


def _expand_block_scales_to_mxfp8_rows(
    scale: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int,
) -> torch.Tensor:
    _check_gpu_tensor("scale", scale)
    if m <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError("m, k, and num_groups must be positive")
    _check_mxfp8_k(k)

    m_tiles = math.ceil(m / MXFP8_SCALE_ROW_TILE)
    k_tiles = math.ceil((k // MXFP8_SCALE_VEC_SIZE) / MXFP8_SCALE_K_TILE)
    expected_2d = (num_groups * m_tiles, k_tiles)
    expected_3d = (num_groups, m_tiles, k_tiles)
    if scale.shape == expected_2d:
        block_u8 = _scale_to_e8m0_u8(scale).reshape(num_groups, m_tiles, k_tiles)
    elif scale.shape == expected_3d:
        block_u8 = _scale_to_e8m0_u8(scale).reshape(expected_3d)
    elif num_groups == 1 and scale.shape == (m_tiles, k_tiles):
        block_u8 = _scale_to_e8m0_u8(scale).reshape(1, m_tiles, k_tiles)
    else:
        raise ValueError(
            "block scale must have shape "
            f"{expected_2d}, {expected_3d}"
            + (f", or {(m_tiles, k_tiles)}" if num_groups == 1 else "")
            + f"; got {tuple(scale.shape)}"
        )

    scale_rows_u8 = (
        block_u8[:, :, None, :, None]
        .expand(num_groups, m_tiles, MXFP8_SCALE_ROW_TILE, k_tiles, MXFP8_SCALE_K_TILE)
        .reshape(
            num_groups,
            m_tiles * MXFP8_SCALE_ROW_TILE,
            k_tiles * MXFP8_SCALE_K_TILE,
        )[:, :m, : k // MXFP8_SCALE_VEC_SIZE]
        .contiguous()
    )
    return scale_rows_u8.view(torch.float8_e8m0fnu)


def pack_mxfp8_scales_for_dense_gemm(
    scale_rows: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int = 1,
) -> torch.Tensor:
    """Pack compact MXFP8 row/chunk scales into b12x dense-GEMM MMA layout.

    `scale_rows` must be UE8M0 scales in either `[M, K/32]`,
    `[num_groups, M, K/32]`, or `[num_groups * M, K/32]` form. Missing padded
    rows/chunks are filled with UE8M0 1.0, so the kernel can safely read the
    fixed 128-row scale tile for small-M contracts.
    """

    _check_gpu_tensor("scale_rows", scale_rows)
    if scale_rows.dtype == torch.uint8:
        scale_rows = scale_rows.view(torch.float8_e8m0fnu)
    if scale_rows.dtype != torch.float8_e8m0fnu:
        raise ValueError(f"scale_rows must be uint8/e8m0, got {scale_rows.dtype}")
    if m <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError("m, k, and num_groups must be positive")
    _check_mxfp8_k(k)

    sf_k = k // MXFP8_SCALE_VEC_SIZE
    if scale_rows.ndim == 2:
        if scale_rows.shape == (m, sf_k):
            grouped = scale_rows.reshape(1, m, sf_k)
            if num_groups != 1:
                raise ValueError(
                    "2D scale_rows with shape [M,K/32] requires num_groups=1"
                )
        elif scale_rows.shape == (num_groups * m, sf_k):
            grouped = scale_rows.reshape(num_groups, m, sf_k)
        else:
            raise ValueError(
                "scale_rows must have shape [M,K/32] or [num_groups*M,K/32], "
                f"got {tuple(scale_rows.shape)} for m={m}, k={k}, num_groups={num_groups}"
            )
    elif scale_rows.ndim == 3:
        if scale_rows.shape != (num_groups, m, sf_k):
            raise ValueError(
                f"scale_rows must have shape {(num_groups, m, sf_k)}, "
                f"got {tuple(scale_rows.shape)}"
            )
        grouped = scale_rows
    else:
        raise ValueError(
            "scale_rows must have shape [M,K/32], [num_groups*M,K/32], "
            f"or [num_groups,M,K/32], got {tuple(scale_rows.shape)}"
        )

    m_tiles = math.ceil(m / MXFP8_SCALE_ROW_TILE)
    k_tiles = math.ceil(sf_k / MXFP8_SCALE_K_TILE)
    padded_m = m_tiles * MXFP8_SCALE_ROW_TILE
    padded_sf_k = k_tiles * MXFP8_SCALE_K_TILE

    padded_u8 = torch.full(
        (num_groups, padded_m, padded_sf_k),
        127,
        dtype=torch.uint8,
        device=scale_rows.device,
    )
    padded = padded_u8.view(torch.float8_e8m0fnu)
    padded[:, :m, :sf_k] = grouped

    physical = (
        padded.view(num_groups, m_tiles, 4, 32, k_tiles, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
    )
    return physical.permute(3, 4, 1, 5, 2, 0)


def _scale_is_exact_ue8m0(scale: torch.Tensor) -> bool:
    """True if `scale` is already an exact UE8M0 scale (no re-quant needed).

    UE8M0-typed tensors (`float8_e8m0fnu`/`uint8`) qualify by construction. A
    floating-point tensor qualifies only if every entry is a positive power of
    two (zero mantissa) — e.g. a previously-UE8M0 scale widened to fp32. Genuine
    checkpoint scales (`weight_scale_inv = block_amax / 448`) are not powers of
    two and fail this test, so they get re-quantized. Runs once at weight load.
    """

    if scale.dtype in (torch.uint8, torch.float8_e8m0fnu):
        return True
    if not scale.is_floating_point():
        return False
    bits = scale.detach().to(torch.float32).view(torch.int32)
    mantissa = bits & 0x7FFFFF
    positive = scale.detach().to(torch.float32) > 0
    return bool(torch.all(positive & (mantissa == 0)).item())


def _requantize_block_fp8_to_ue8m0(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize an FP8 checkpoint weight onto exact UE8M0 block scales.

    DeepSeek-style checkpoints carry `(w_fp8, s_fp32)` where `s_fp32 =
    block_amax / 448` is an *arbitrary* fp32 128x128 block scale. The b12x MMA
    can only apply power-of-two (UE8M0) scales. Rounding `s_fp32` to the nearest
    power of two while keeping the original `w_fp8` values leaves the values
    matched to the *unrounded* scale, baking in a per-block scale error up to
    sqrt(2) (~10% weight RMS error, measured).

    Parity with DeepGEMM's `per_block_cast_to_fp8(use_ue8m0=True)`: reconstruct
    the recovered weight `w_fp8 * s_fp32` and re-cast it onto a fresh ceil-UE8M0
    block scale, so the FP8 values are re-derived *consistently* with the
    power-of-two scale (~3.7% RMS error, near the irreducible e4m3 floor). This
    runs once at weight load, not in the serving path.

    Returns `(weight_e4m3 [num_groups*m, k] or [m, k], block_scale_e8m0_u8
    [num_groups, m_tiles, k_tiles])`.
    """

    gk = MXFP8_SCALE_ROW_TILE  # 128
    m_tiles = math.ceil(m / gk)
    k_tiles = math.ceil((k // MXFP8_SCALE_VEC_SIZE) / MXFP8_SCALE_K_TILE)

    # Normalize the FP8 weight to per-group [num_groups, m, k].
    if num_groups == 1:
        w_g = weight.reshape(1, m, k)
    elif tuple(weight.shape) == (num_groups * m, k):
        w_g = weight.reshape(num_groups, m, k)
    elif tuple(weight.shape) == (m, k, num_groups):
        w_g = weight.permute(2, 0, 1)
    else:
        raise ValueError(
            f"weight must have shape {(num_groups * m, k)} or {(m, k, num_groups)} "
            f"or {(m, k)}, got {tuple(weight.shape)}"
        )

    # Normalize the arbitrary fp32 block scale to [num_groups, m_tiles, k_tiles].
    s = scale.to(torch.float32)
    if tuple(s.shape) == (num_groups * m_tiles, k_tiles):
        s_g = s.reshape(num_groups, m_tiles, k_tiles)
    elif tuple(s.shape) == (num_groups, m_tiles, k_tiles):
        s_g = s
    elif num_groups == 1 and tuple(s.shape) == (m_tiles, k_tiles):
        s_g = s.reshape(1, m_tiles, k_tiles)
    else:
        raise ValueError(
            f"block scale must have shape {(num_groups * m_tiles, k_tiles)}, "
            f"{(num_groups, m_tiles, k_tiles)}"
            + (f", or {(m_tiles, k_tiles)}" if num_groups == 1 else "")
            + f"; got {tuple(scale.shape)}"
        )

    # Reconstruct the recovered weight: w_fp8 * (block scale broadcast to elems).
    s_elem = (
        s_g.repeat_interleave(gk, dim=1).repeat_interleave(gk, dim=2)[:, :m, :k]
    )
    w_rec = w_g.to(torch.float32) * s_elem

    # Pad to whole 128x128 blocks, take per-block amax, ceil-UE8M0 cast.
    m_pad, k_pad = m_tiles * gk, k_tiles * gk
    w_pad = w_rec.new_zeros(num_groups, m_pad, k_pad)
    w_pad[:, :m, :k] = w_rec
    blocks = w_pad.view(num_groups, m_tiles, gk, k_tiles, gk)
    amax = blocks.abs().amax(dim=(2, 4), keepdim=True)  # [g, m_tiles, 1, k_tiles, 1]
    safe = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
    exponent = torch.clamp(torch.ceil(torch.log2(safe)), -127.0, 127.0)
    sf = torch.exp2(exponent)
    new_vals = (
        (blocks / sf)
        .to(torch.float8_e4m3fn)
        .view(num_groups, m_pad, k_pad)[:, :m, :k]
        .contiguous()
    )
    new_scale_u8 = (exponent.view(num_groups, m_tiles, k_tiles) + 127).to(torch.uint8)

    if num_groups == 1:
        new_weight = new_vals.reshape(m, k)
    else:
        new_weight = new_vals.reshape(num_groups * m, k)
    return new_weight, new_scale_u8


def pack_fp8_block_scaled_weight_mxfp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int = 1,
) -> MXFP8Rows:
    """Pack checkpoint FP8 block-scaled weights for native MXFP8 dense GEMM.

    `weight` is FP8 E4M3 and `scale` is the DSV4-style 128x128 block scale.

    When `scale` is an *arbitrary* fp32 block scale (the DeepSeek
    `weight_scale_inv` checkpoint format), the weight is **re-quantized** onto an
    exact ceil-UE8M0 block scale so the FP8 values stay consistent with the
    power-of-two scale the MMA applies (parity with DeepGEMM
    `per_block_cast_to_fp8`). When `scale` is already UE8M0 (e8m0/uint8), the
    FP8 values are kept verbatim and the scale is expanded as-is.
    """

    _check_gpu_tensor("weight", weight)
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if m <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError("m, k, and num_groups must be positive")
    _check_mxfp8_k(k)

    # Arbitrary fp32 checkpoint scales (e.g. DeepSeek `weight_scale_inv`) are not
    # powers of two; re-quantize onto exact UE8M0 instead of rounding the scale
    # and leaving the FP8 values stale (which costs ~2.7x more error). Scales
    # that are already exact UE8M0 (e8m0/uint8, or a float tensor holding only
    # powers of two) keep their FP8 values verbatim.
    if not _scale_is_exact_ue8m0(scale):
        _check_gpu_tensor("scale", scale)
        weight, scale = _requantize_block_fp8_to_ue8m0(
            weight, scale, m=m, k=k, num_groups=num_groups
        )

    if num_groups == 1:
        if weight.shape != (m, k):
            raise ValueError(f"weight must have shape {(m, k)}, got {tuple(weight.shape)}")
        values = weight.contiguous()
    else:
        if weight.shape == (num_groups * m, k):
            values = weight.contiguous().view(num_groups, m, k).permute(1, 2, 0)
        elif weight.shape == (m, k, num_groups):
            values = weight
            _check_dense_gemm_mnl_view("weight", values)
        else:
            raise ValueError(
                f"weight must have shape {(num_groups * m, k)} or {(m, k, num_groups)}, "
                f"got {tuple(weight.shape)}"
            )

    scale_rows = _expand_block_scales_to_mxfp8_rows(
        scale,
        m=m,
        k=k,
        num_groups=num_groups,
    )
    scale_mma = pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=m,
        k=k,
        num_groups=num_groups,
    )
    return MXFP8Rows(values=values, scale_rows=scale_rows, scale_mma=scale_mma)


def pack_wo_projection_fp8_block_scaled_weights_mxfp8(
    wo_a_weight: torch.Tensor,
    wo_a_scale: torch.Tensor,
    wo_b_weight: torch.Tensor,
    wo_b_scale: torch.Tensor,
    *,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
) -> WOProjectionMXFP8Weights:
    """Pack local DSV4 WO-A/WO-B checkpoint FP8 weights for the b12x WO path."""

    wo_a = pack_fp8_block_scaled_weight_mxfp8(
        wo_a_weight,
        wo_a_scale,
        m=rank,
        k=group_width,
        num_groups=groups,
    )
    wo_b = pack_fp8_block_scaled_weight_mxfp8(
        wo_b_weight,
        wo_b_scale,
        m=hidden,
        k=groups * rank,
        num_groups=1,
    )
    return WOProjectionMXFP8Weights(
        wo_a=wo_a,
        wo_b=wo_b,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )


def _check_mxfp8_rows_storage(
    out: MXFP8Rows,
    *,
    m: int,
    k: int,
    num_groups: int,
) -> None:
    _check_gpu_tensor("out.values", out.values)
    _check_gpu_tensor("out.scale_rows", out.scale_rows)
    _check_gpu_tensor("out.scale_mma", out.scale_mma)
    if out.values.dtype != torch.float8_e4m3fn:
        raise ValueError(f"out.values must be float8_e4m3fn, got {out.values.dtype}")
    if out.scale_rows.dtype != torch.float8_e8m0fnu:
        raise ValueError(f"out.scale_rows must be float8_e8m0fnu, got {out.scale_rows.dtype}")
    if out.scale_mma.dtype != torch.float8_e8m0fnu:
        raise ValueError(f"out.scale_mma must be float8_e8m0fnu, got {out.scale_mma.dtype}")
    if num_groups == 1:
        if out.values.shape != (m, k):
            raise ValueError(f"out.values must have shape {(m, k)}, got {tuple(out.values.shape)}")
    else:
        if out.values.shape != (m, k, num_groups):
            raise ValueError(
                f"out.values must have shape {(m, k, num_groups)}, got {tuple(out.values.shape)}"
            )
        _check_dense_gemm_mnl_view("out.values", out.values)
    sf_k = k // MXFP8_SCALE_VEC_SIZE
    if out.scale_rows.shape != (num_groups, m, sf_k):
        raise ValueError(
            f"out.scale_rows must have shape {(num_groups, m, sf_k)}, "
            f"got {tuple(out.scale_rows.shape)}"
        )
    expected_scale_mma = (
        32,
        4,
        math.ceil(m / MXFP8_SCALE_ROW_TILE),
        4,
        math.ceil(sf_k / MXFP8_SCALE_K_TILE),
        num_groups,
    )
    if out.scale_mma.shape != expected_scale_mma:
        raise ValueError(
            f"out.scale_mma must have shape {expected_scale_mma}, got {tuple(out.scale_mma.shape)}"
        )


def quantize_wo_a_input_mxfp8(
    source_tgd: torch.Tensor,
    *,
    out: MXFP8Rows | None = None,
) -> MXFP8Rows:
    """Quantize grouped WO-A input into per-32-column MXFP8 rows."""

    _check_gpu_tensor("source_tgd", source_tgd)
    if source_tgd.ndim != 3:
        raise ValueError(
            f"source_tgd must have shape [tokens, groups, group_width], got {tuple(source_tgd.shape)}"
        )
    tokens, groups, group_width = source_tgd.shape
    _check_mxfp8_k(group_width)
    if out is None:
        out = empty_mxfp8_rows_for_dense_gemm(
            tokens,
            group_width,
            num_groups=groups,
            device=source_tgd.device,
        )
    else:
        _check_mxfp8_rows_storage(out, m=tokens, k=group_width, num_groups=groups)
    _quantize_grouped_tgd_to_tdg_kernel[
        (tokens, groups, group_width // WO_A_INPUT_QUANT_GROUP_SIZE)
    ](
        source_tgd,
        out.values,
        out.scale_rows.view(torch.uint8),
        out.scale_mma.view(torch.uint8),
        tokens,
        groups,
        group_width,
        source_tgd.stride(0),
        source_tgd.stride(1),
        source_tgd.stride(2),
        out.values.stride(0),
        out.values.stride(1),
        out.values.stride(2),
        out.scale_mma.stride(0),
        out.scale_mma.stride(1),
        out.scale_mma.stride(2),
        out.scale_mma.stride(3),
        out.scale_mma.stride(4),
        out.scale_mma.stride(5),
        SCALE_CHUNKS=WO_A_INPUT_QUANT_GROUP_SIZE // MXFP8_SCALE_VEC_SIZE,
        BLOCK=WO_A_INPUT_QUANT_GROUP_SIZE,
    )
    return out


def quantize_wo_a_input_inv_rope_mxfp8(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    out: MXFP8Rows | None = None,
) -> MXFP8Rows:
    """Inverse-RoPE attention output and quantize with 128-column WO-A scales."""

    _check_gpu_tensor("o", o)
    _check_gpu_tensor("positions", positions)
    _check_gpu_tensor("cos_sin_cache", cos_sin_cache)
    if o.ndim != 3:
        raise ValueError(f"o must have shape [tokens, heads, head_dim], got {tuple(o.shape)}")
    tokens, heads, head_dim = o.shape
    if head_dim != nope_dim + rope_dim:
        raise ValueError(
            f"o head_dim must equal nope_dim + rope_dim ({nope_dim + rope_dim}), "
            f"got {head_dim}"
        )
    if heads != groups * heads_per_group:
        raise ValueError(
            f"o heads must equal groups * heads_per_group ({groups * heads_per_group}), "
            f"got {heads}"
        )
    if positions.shape != (tokens,):
        raise ValueError(
            f"positions must have shape {(tokens,)}, got {tuple(positions.shape)}"
        )
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != rope_dim:
        raise ValueError(
            "cos_sin_cache must have shape [max_position, rope_dim], "
            f"got {tuple(cos_sin_cache.shape)}"
        )
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")

    group_width = heads_per_group * head_dim
    _check_mxfp8_k(group_width)
    if out is None:
        out = empty_mxfp8_rows_for_dense_gemm(
            tokens,
            group_width,
            num_groups=groups,
            device=o.device,
        )
    else:
        _check_mxfp8_rows_storage(out, m=tokens, k=group_width, num_groups=groups)

    _quantize_attention_inv_rope_to_tdg_kernel[
        (tokens, groups, group_width // WO_A_INPUT_QUANT_GROUP_SIZE)
    ](
        o,
        positions,
        cos_sin_cache,
        out.values,
        out.scale_rows.view(torch.uint8),
        out.scale_mma.view(torch.uint8),
        tokens,
        groups,
        heads_per_group,
        group_width,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        cos_sin_cache.stride(0),
        out.values.stride(0),
        out.values.stride(1),
        out.values.stride(2),
        out.scale_mma.stride(0),
        out.scale_mma.stride(1),
        out.scale_mma.stride(2),
        out.scale_mma.stride(3),
        out.scale_mma.stride(4),
        out.scale_mma.stride(5),
        HEAD_DIM=head_dim,
        NOPE_DIM=nope_dim,
        HALF_ROPE_DIM=rope_dim // 2,
        SCALE_CHUNKS=WO_A_INPUT_QUANT_GROUP_SIZE // MXFP8_SCALE_VEC_SIZE,
        BLOCK=WO_A_INPUT_QUANT_GROUP_SIZE,
    )
    return out


def quantize_wo_b_input_mxfp8(
    source_trg: torch.Tensor,
    *,
    out: MXFP8Rows | None = None,
) -> MXFP8Rows:
    """Quantize WO-A intermediate `[tokens, rank, groups]` into group-major `[tokens, groups * rank]`."""

    _check_gpu_tensor("source_trg", source_trg)
    if source_trg.ndim != 3:
        raise ValueError(
            f"source_trg must have shape [tokens, rank, groups], got {tuple(source_trg.shape)}"
        )
    tokens, rank, groups = source_trg.shape
    width = rank * groups
    _check_mxfp8_k(width)
    if out is None:
        out = empty_mxfp8_rows_for_dense_gemm(
            tokens,
            width,
            num_groups=1,
            device=source_trg.device,
        )
    else:
        _check_mxfp8_rows_storage(out, m=tokens, k=width, num_groups=1)
    _quantize_group_major_trg_to_tk_kernel[(tokens, width // MXFP8_SCALE_VEC_SIZE)](
        source_trg,
        out.values,
        out.scale_rows.view(torch.uint8),
        out.scale_mma.view(torch.uint8),
        tokens,
        rank,
        groups,
        source_trg.stride(0),
        source_trg.stride(1),
        source_trg.stride(2),
        out.scale_mma.stride(0),
        out.scale_mma.stride(1),
        out.scale_mma.stride(2),
        out.scale_mma.stride(3),
        out.scale_mma.stride(4),
        out.scale_mma.stride(5),
        BLOCK=MXFP8_SCALE_VEC_SIZE,
    )
    return out


def quantize_mxfp8_rows_torch(source: torch.Tensor) -> MXFP8Rows:
    """Quantize `[M,K]` or `[M,K,L]` rows to MXFP8 on GPU.

    This is a graph-capturable GPU Torch prep/reference helper. It is not the
    final production activation-quant kernel for WO-A/WO-B.
    """

    grouped, m, k, num_groups = _as_grouped_mkl(source)
    _check_mxfp8_k(k)

    chunks = k // MXFP8_SCALE_VEC_SIZE
    grouped_f32 = grouped.to(torch.float32)
    blocked = grouped_f32.reshape(num_groups, m, chunks, MXFP8_SCALE_VEC_SIZE)
    max_abs = blocked.abs().amax(dim=-1)
    scale_u8 = _scale_u8_from_max_abs(max_abs)
    scale_rows = scale_u8.view(torch.float8_e8m0fnu)
    scale = scale_rows.to(torch.float32)
    quant_grouped = (
        (blocked / scale[..., None])
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(num_groups, m, k)
        .contiguous()
    )
    if source.ndim == 2:
        values = quant_grouped.reshape(m, k).contiguous()
    else:
        values = quant_grouped.as_strided((m, k, num_groups), (k, 1, m * k))
    scale_mma = pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=m,
        k=k,
        num_groups=num_groups,
    )
    return MXFP8Rows(values=values, scale_rows=scale_rows, scale_mma=scale_mma)


def dequantize_mxfp8_rows_torch(values: torch.Tensor, scale_rows: torch.Tensor) -> torch.Tensor:
    """Dequantize compact row-wise MXFP8 values on GPU for tests/oracles."""

    grouped, m, k, num_groups = _as_grouped_mkl(values)
    _check_mxfp8_k(k)
    sf_k = k // MXFP8_SCALE_VEC_SIZE
    if scale_rows.dtype == torch.uint8:
        scale_rows = scale_rows.view(torch.float8_e8m0fnu)
    if scale_rows.shape != (num_groups, m, sf_k):
        raise ValueError(
            f"scale_rows must have shape {(num_groups, m, sf_k)}, got {tuple(scale_rows.shape)}"
        )
    scale = scale_rows.to(torch.float32)
    out_grouped = (
        grouped.to(torch.float32)
        .reshape(num_groups, m, sf_k, MXFP8_SCALE_VEC_SIZE)
        * scale[..., None]
    ).reshape(num_groups, m, k)
    out = out_grouped.permute(1, 2, 0).contiguous()
    if values.ndim == 2:
        out = out[:, :, 0].contiguous()
    return out


def quantize_wo_projection_weights_mxfp8_torch(
    wo_a_grd: torch.Tensor,
    wo_b_hgr: torch.Tensor,
) -> WOProjectionMXFP8Weights:
    """Quantize BF16 WO weights into the native MXFP8 two-GEMM layouts.

    This is a GPU Torch setup helper for tests and benchmarks. Serving should
    prepare the same layouts at model load from checkpoint FP8 weights/scales.
    """

    _check_gpu_tensor("wo_a_grd", wo_a_grd)
    _check_gpu_tensor("wo_b_hgr", wo_b_hgr)
    if wo_a_grd.ndim != 3:
        raise ValueError(
            f"wo_a_grd must have shape [groups, rank, group_width], got {tuple(wo_a_grd.shape)}"
        )
    if wo_b_hgr.ndim != 2:
        raise ValueError(
            f"wo_b_hgr must have shape [hidden, groups * rank], got {tuple(wo_b_hgr.shape)}"
        )
    groups, rank, group_width = wo_a_grd.shape
    hidden, wo_b_width = wo_b_hgr.shape
    if wo_b_width != groups * rank:
        raise ValueError(
            f"wo_b_hgr width must equal groups * rank, got {wo_b_width} vs {groups * rank}"
        )
    _check_mxfp8_k(group_width)
    _check_mxfp8_k(groups * rank)

    wo_a = quantize_mxfp8_rows_torch(wo_a_grd.permute(1, 2, 0).contiguous())
    wo_b = quantize_mxfp8_rows_torch(wo_b_hgr)
    return WOProjectionMXFP8Weights(
        wo_a=wo_a,
        wo_b=wo_b,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )


def _check_wo_projection_weights(weights: WOProjectionMXFP8Weights) -> None:
    if not isinstance(weights, WOProjectionMXFP8Weights):
        raise TypeError("weights must be a WOProjectionMXFP8Weights instance")
    if (
        weights.groups <= 0
        or weights.group_width <= 0
        or weights.rank <= 0
        or weights.hidden <= 0
    ):
        raise ValueError("WO projection dimensions must be positive")
    _check_mxfp8_rows_storage(
        weights.wo_a,
        m=weights.rank,
        k=weights.group_width,
        num_groups=weights.groups,
    )
    _check_mxfp8_rows_storage(
        weights.wo_b,
        m=weights.hidden,
        k=weights.rank * weights.groups,
        num_groups=1,
    )


def empty_wo_projection_workspace(
    tokens: int,
    *,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
    device: torch.device | str,
    output: torch.Tensor | None = None,
) -> WOProjectionWorkspace:
    """Allocate fixed scratch/output tensors for one WO projection graph shape."""

    if tokens <= 0 or groups <= 0 or group_width <= 0 or rank <= 0 or hidden <= 0:
        raise ValueError("tokens, groups, group_width, rank, and hidden must be positive")
    _check_mxfp8_k(group_width)
    _check_mxfp8_k(rank * groups)

    x_q = empty_mxfp8_rows_for_dense_gemm(
        tokens,
        group_width,
        num_groups=groups,
        device=device,
    )
    tmp = empty_dense_gemm_mnl_view(
        tokens,
        rank,
        groups,
        device=device,
        dtype=torch.bfloat16,
    )
    tmp_q = empty_mxfp8_rows_for_dense_gemm(
        tokens,
        rank * groups,
        num_groups=1,
        device=device,
    )
    if output is None:
        output = torch.empty((tokens, hidden, 1), device=device, dtype=torch.bfloat16)
    else:
        _check_dense_gemm_mnl_view("output", output)
        if output.shape != (tokens, hidden, 1):
            raise ValueError(
                f"output must have shape {(tokens, hidden, 1)}, got {tuple(output.shape)}"
            )
        if output.dtype != torch.bfloat16:
            raise ValueError(f"output must be bfloat16, got {output.dtype}")
    return WOProjectionWorkspace(x_q=x_q, tmp=tmp, tmp_q=tmp_q, output=output)


def _check_wo_projection_workspace(
    workspace: WOProjectionWorkspace,
    *,
    tokens: int,
    weights: WOProjectionMXFP8Weights,
) -> None:
    if not isinstance(workspace, WOProjectionWorkspace):
        raise TypeError("workspace must be a WOProjectionWorkspace instance")
    _check_wo_projection_views(
        x_q=workspace.x_q,
        tmp=workspace.tmp,
        tmp_q=workspace.tmp_q,
        output=workspace.output,
        tokens=tokens,
        weights=weights,
    )


def _check_wo_projection_views(
    *,
    x_q: MXFP8Rows,
    tmp: torch.Tensor,
    tmp_q: MXFP8Rows,
    output: torch.Tensor,
    tokens: int,
    weights: WOProjectionMXFP8Weights,
) -> None:
    _check_mxfp8_rows_storage(
        x_q,
        m=tokens,
        k=weights.group_width,
        num_groups=weights.groups,
    )
    _check_dense_gemm_mnl_view("tmp", tmp)
    if tmp.shape != (tokens, weights.rank, weights.groups):
        raise ValueError(
            "tmp must have shape "
            f"{(tokens, weights.rank, weights.groups)}, got {tuple(tmp.shape)}"
        )
    if tmp.dtype != torch.bfloat16:
        raise ValueError(f"tmp must be bfloat16, got {tmp.dtype}")
    _check_mxfp8_rows_storage(
        tmp_q,
        m=tokens,
        k=weights.rank * weights.groups,
        num_groups=1,
    )
    _check_dense_gemm_mnl_view("output", output)
    if output.shape != (tokens, weights.hidden, 1):
        raise ValueError(
            "output must have shape "
            f"{(tokens, weights.hidden, 1)}, got {tuple(output.shape)}"
        )
    if output.dtype != torch.bfloat16:
        raise ValueError(f"output must be bfloat16, got {output.dtype}")


def _validate_wo_projection_inputs(
    source_tgd: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
) -> int:
    _check_gpu_tensor("source_tgd", source_tgd)
    _check_wo_projection_weights(weights)
    if source_tgd.ndim != 3:
        raise ValueError(
            f"source_tgd must have shape [tokens, groups, group_width], got {tuple(source_tgd.shape)}"
        )
    tokens, groups, group_width = source_tgd.shape
    if (groups, group_width) != (weights.groups, weights.group_width):
        raise ValueError(
            "source_tgd shape does not match weights: "
            f"source={(groups, group_width)}, weights={(weights.groups, weights.group_width)}"
        )
    return int(tokens)


def _validate_wo_projection_inv_rope_inputs(
    *,
    o: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> int:
    _check_gpu_tensor("o", o)
    _check_wo_projection_weights(weights)
    heads_per_group = int(heads_per_group)
    nope_dim = int(nope_dim)
    rope_dim = int(rope_dim)
    if heads_per_group <= 0 or nope_dim <= 0 or rope_dim <= 0:
        raise ValueError("heads_per_group, nope_dim, and rope_dim must be positive")
    if o.ndim != 3:
        raise ValueError(f"o must have shape [tokens, heads, head_dim], got {tuple(o.shape)}")
    tokens, heads, head_dim = o.shape
    if head_dim != nope_dim + rope_dim:
        raise ValueError(
            f"o head_dim must equal nope_dim + rope_dim ({nope_dim + rope_dim}), "
            f"got {head_dim}"
        )
    if heads != weights.groups * heads_per_group:
        raise ValueError(
            f"o heads must equal weights.groups * heads_per_group "
            f"({weights.groups * heads_per_group}), got {heads}"
        )
    if weights.group_width != heads_per_group * head_dim:
        raise ValueError(
            "weights.group_width does not match heads_per_group * head_dim: "
            f"{weights.group_width} != {heads_per_group * head_dim}"
        )
    return int(tokens)


def build_wo_projection_binding(
    *,
    workspace: WOProjectionWorkspace,
    source_tgd: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    return_3d: bool = False,
    expected_m: int | None = None,
) -> WOProjectionBinding:
    tokens = _validate_wo_projection_inputs(source_tgd, weights)
    _check_wo_projection_workspace(workspace, tokens=tokens, weights=weights)
    return _build_wo_projection_binding_from_views(
        x_q=workspace.x_q,
        tmp=workspace.tmp,
        tmp_q=workspace.tmp_q,
        output=workspace.output,
        source_tgd=source_tgd,
        weights=weights,
        return_3d=return_3d,
        expected_m=expected_m,
    )


def _build_wo_projection_binding_from_views(
    *,
    x_q: MXFP8Rows,
    tmp: torch.Tensor,
    tmp_q: MXFP8Rows,
    output: torch.Tensor,
    source_tgd: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    return_3d: bool = False,
    expected_m: int | None = None,
) -> WOProjectionBinding:
    tokens = _validate_wo_projection_inputs(source_tgd, weights)
    _check_wo_projection_views(
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        tokens=tokens,
        weights=weights,
    )
    return WOProjectionBinding(
        source_tgd=source_tgd,
        weights=weights,
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        return_3d=bool(return_3d),
        expected_m=expected_m,
    )


def build_wo_projection_inv_rope_binding(
    *,
    workspace: WOProjectionWorkspace,
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    return_3d: bool = False,
    expected_m: int | None = None,
) -> WOProjectionInvRopeBinding:
    tokens = _validate_wo_projection_inv_rope_inputs(
        o=o,
        weights=weights,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    _check_gpu_tensor("positions", positions)
    _check_gpu_tensor("cos_sin_cache", cos_sin_cache)
    _check_wo_projection_workspace(workspace, tokens=tokens, weights=weights)
    return _build_wo_projection_inv_rope_binding_from_views(
        x_q=workspace.x_q,
        tmp=workspace.tmp,
        tmp_q=workspace.tmp_q,
        output=workspace.output,
        o=o,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        return_3d=return_3d,
        expected_m=expected_m,
    )


def _build_wo_projection_inv_rope_binding_from_views(
    *,
    x_q: MXFP8Rows,
    tmp: torch.Tensor,
    tmp_q: MXFP8Rows,
    output: torch.Tensor,
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    return_3d: bool = False,
    expected_m: int | None = None,
) -> WOProjectionInvRopeBinding:
    tokens = _validate_wo_projection_inv_rope_inputs(
        o=o,
        weights=weights,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    _check_gpu_tensor("positions", positions)
    _check_gpu_tensor("cos_sin_cache", cos_sin_cache)
    _check_wo_projection_views(
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        tokens=tokens,
        weights=weights,
    )
    return WOProjectionInvRopeBinding(
        o=o,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        heads_per_group=int(heads_per_group),
        nope_dim=int(nope_dim),
        rope_dim=int(rope_dim),
        return_3d=bool(return_3d),
        expected_m=expected_m,
    )


def plan_wo_projection_scratch(caps: WOProjectionScratchCaps) -> WOProjectionScratchPlan:
    layout = _layout_wo_projection(
        offset_bytes=0,
        tokens=caps.max_tokens,
        groups=caps.groups,
        group_width=caps.group_width,
        rank=caps.rank,
        hidden=caps.hidden,
    )
    return WOProjectionScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "wo_projection.scratch",
                nbytes=layout.nbytes,
                device=caps.device,
            ),
        ),
    )


def wo_a_dense_gemm_mxfp8(
    x_tdg: MXFP8Rows,
    wo_a_rdg: MXFP8Rows,
    *,
    out: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    expected_m: int | None = None,
) -> torch.Tensor:
    """Run WO-A as grouped MXFP8 dense GEMM.

    Inputs are `x.values [tokens, group_width, groups]` and
    `wo_a.values [rank, group_width, groups]`; output is `[tokens, rank, groups]`.
    """

    if x_tdg.values.ndim != 3 or wo_a_rdg.values.ndim != 3:
        raise ValueError("WO-A operands must have shape [M,K,groups] and [N,K,groups]")
    if x_tdg.values.shape[1:] != wo_a_rdg.values.shape[1:]:
        raise ValueError(
            f"WO-A K/groups mismatch: x={tuple(x_tdg.values.shape)} "
            f"wo_a={tuple(wo_a_rdg.values.shape)}"
        )
    if out is None:
        out = empty_dense_gemm_mnl_view(
            x_tdg.values.shape[0],
            wo_a_rdg.values.shape[0],
            x_tdg.values.shape[2],
            device=x_tdg.values.device,
            dtype=torch.bfloat16,
        )
    else:
        _check_dense_gemm_mnl_view("out", out)
    mma_tiler_mn = (
        (16, 64)
        if expected_m is not None
        and 1 <= expected_m <= 8
        and wo_a_rdg.values.shape[0] <= 1536
        else None
    )
    return dense_gemm(
        (x_tdg.values, x_tdg.scale_mma),
        (wo_a_rdg.values, wo_a_rdg.scale_mma),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        out=out,
        alpha=alpha,
        mma_tiler_mn=mma_tiler_mn,
        expected_m=expected_m,
    )


def wo_b_dense_gemm_mxfp8(
    tmp_tgr_group_major: MXFP8Rows,
    wo_b_hgr: MXFP8Rows,
    *,
    out: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    expected_m: int | None = None,
) -> torch.Tensor:
    """Run group-major WO-B as MXFP8 dense GEMM.

    Inputs are `tmp.values [tokens, rank * groups]` and
    `wo_b.values [hidden, rank * groups]`; output is `[tokens, hidden, 1]`.
    """

    if tmp_tgr_group_major.values.ndim != 2 or wo_b_hgr.values.ndim != 2:
        raise ValueError("WO-B operands must have shape [M,K] and [N,K]")
    if tmp_tgr_group_major.values.shape[1] != wo_b_hgr.values.shape[1]:
        raise ValueError(
            f"WO-B K mismatch: tmp={tuple(tmp_tgr_group_major.values.shape)} "
            f"wo_b={tuple(wo_b_hgr.values.shape)}"
        )
    if out is None:
        out = torch.empty(
            (
                tmp_tgr_group_major.values.shape[0],
                wo_b_hgr.values.shape[0],
                1,
            ),
            device=tmp_tgr_group_major.values.device,
            dtype=torch.bfloat16,
        )
    else:
        _check_dense_gemm_mnl_view("out", out)
    return dense_gemm(
        (
            tmp_tgr_group_major.values.reshape(
                tmp_tgr_group_major.values.shape[0],
                tmp_tgr_group_major.values.shape[1],
                1,
            ),
            tmp_tgr_group_major.scale_mma,
        ),
        (
            wo_b_hgr.values.reshape(
                wo_b_hgr.values.shape[0],
                wo_b_hgr.values.shape[1],
                1,
            ),
            wo_b_hgr.scale_mma,
        ),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        out=out,
        alpha=alpha,
        expected_m=expected_m,
    )


def wo_projection_mxfp8(
    source_tgd: torch.Tensor | None = None,
    weights: WOProjectionMXFP8Weights | None = None,
    workspace: WOProjectionWorkspace | None = None,
    *,
    return_3d: bool = False,
    binding: WOProjectionBinding | None = None,
    expected_m: int | None = None,
) -> torch.Tensor:
    """Run the native MXFP8 WO-A/WO-B projection.

    `source_tgd` is `[tokens, groups, group_width]`. The default return value
    is the SGLang-friendly `[tokens, hidden]` view over `workspace.output`.
    """

    if binding is not None:
        extras = [
            name
            for name, value in (
                ("source_tgd", source_tgd),
                ("weights", weights),
                ("workspace", workspace),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "WO projection binding owns source_tgd, weights, and workspace; "
                f"do not also pass {', '.join(extras)}"
            )
        if return_3d:
            raise ValueError(
                "WO projection binding owns return_3d; do not also pass return_3d=True"
            )
        source_tgd = binding.source_tgd
        weights = binding.weights
        x_q = binding.x_q
        tmp = binding.tmp
        tmp_q = binding.tmp_q
        output = binding.output
        return_3d = binding.return_3d
        expected_m = binding.expected_m
    else:
        x_q = None
        tmp = None
        tmp_q = None
        output = None
    if source_tgd is None or weights is None:
        raise TypeError("wo_projection_mxfp8 requires source_tgd, weights, and workspace or binding")
    tokens = _validate_wo_projection_inputs(source_tgd, weights)
    # WO auto-defaults the regime hint to the (capture-fixed) token count so the
    # wo_b up-projection (N=hidden>1536) picks the decode tile (32x128) at small
    # M and the prefill tile (64x128) at large M, with no caller change.
    # Freeze-safe: the binding is built per-forward / per-capture, so `tokens`
    # is fixed per compiled kernel. Pass expected_m explicitly to override.
    if expected_m is None:
        expected_m = int(tokens)
    if workspace is not None:
        _check_wo_projection_workspace(workspace, tokens=tokens, weights=weights)
        x_q = workspace.x_q
        tmp = workspace.tmp
        tmp_q = workspace.tmp_q
        output = workspace.output
    if x_q is None or tmp is None or tmp_q is None or output is None:
        raise TypeError("wo_projection_mxfp8 requires source_tgd, weights, and workspace or binding")

    alpha_one = _cached_alpha_one(source_tgd.device)
    quantize_wo_a_input_mxfp8(source_tgd, out=x_q)
    wo_a_dense_gemm_mxfp8(
        x_q,
        weights.wo_a,
        out=tmp,
        alpha=alpha_one,
        expected_m=expected_m,
    )
    quantize_wo_b_input_mxfp8(tmp, out=tmp_q)
    wo_b_dense_gemm_mxfp8(
        tmp_q,
        weights.wo_b,
        out=output,
        alpha=alpha_one,
        expected_m=expected_m,
    )
    if return_3d:
        return output
    return output[:, :, 0]


def wo_projection_inv_rope_mxfp8(
    o: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    weights: WOProjectionMXFP8Weights | None = None,
    workspace: WOProjectionWorkspace | None = None,
    *,
    heads_per_group: int | None = None,
    nope_dim: int = 448,
    rope_dim: int = 64,
    return_3d: bool = False,
    binding: WOProjectionInvRopeBinding | None = None,
    expected_m: int | None = None,
) -> torch.Tensor:
    """Run WO projection from attention output without BF16 inverse-RoPE storage."""

    if binding is not None:
        extras = [
            name
            for name, value in (
                ("o", o),
                ("positions", positions),
                ("cos_sin_cache", cos_sin_cache),
                ("weights", weights),
                ("workspace", workspace),
                ("heads_per_group", heads_per_group),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "WO projection inverse-RoPE binding owns runtime tensors, weights, "
                f"workspace, and heads_per_group; do not also pass {', '.join(extras)}"
            )
        extra_options = []
        if nope_dim != 448:
            extra_options.append("nope_dim")
        if rope_dim != 64:
            extra_options.append("rope_dim")
        if return_3d:
            extra_options.append("return_3d")
        if extra_options:
            raise ValueError(
                "WO projection inverse-RoPE binding owns options; "
                f"do not also pass {', '.join(extra_options)}"
            )
        o = binding.o
        positions = binding.positions
        cos_sin_cache = binding.cos_sin_cache
        weights = binding.weights
        x_q = binding.x_q
        tmp = binding.tmp
        tmp_q = binding.tmp_q
        output = binding.output
        heads_per_group = binding.heads_per_group
        nope_dim = binding.nope_dim
        rope_dim = binding.rope_dim
        return_3d = binding.return_3d
        expected_m = binding.expected_m
    else:
        x_q = None
        tmp = None
        tmp_q = None
        output = None
    if (
        o is None
        or positions is None
        or cos_sin_cache is None
        or weights is None
        or heads_per_group is None
    ):
        raise TypeError(
            "wo_projection_inv_rope_mxfp8 requires o, positions, cos_sin_cache, "
            "weights, workspace, and heads_per_group or binding"
        )
    tokens = _validate_wo_projection_inv_rope_inputs(
        o=o,
        weights=weights,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    _check_gpu_tensor("positions", positions)
    _check_gpu_tensor("cos_sin_cache", cos_sin_cache)
    # Auto-default the regime hint to the (capture-fixed) token count (see
    # wo_projection_mxfp8); decode -> 32x128, prefill -> 64x128, no caller change.
    if expected_m is None:
        expected_m = int(tokens)
    if workspace is not None:
        _check_wo_projection_workspace(workspace, tokens=tokens, weights=weights)
        x_q = workspace.x_q
        tmp = workspace.tmp
        tmp_q = workspace.tmp_q
        output = workspace.output
    if x_q is None or tmp is None or tmp_q is None or output is None:
        raise TypeError(
            "wo_projection_inv_rope_mxfp8 requires o, positions, cos_sin_cache, "
            "weights, workspace, and heads_per_group or binding"
        )

    alpha_one = _cached_alpha_one(o.device)
    quantize_wo_a_input_inv_rope_mxfp8(
        o,
        positions,
        cos_sin_cache,
        groups=weights.groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        out=x_q,
    )
    wo_a_dense_gemm_mxfp8(
        x_q,
        weights.wo_a,
        out=tmp,
        alpha=alpha_one,
        expected_m=expected_m,
    )
    quantize_wo_b_input_mxfp8(tmp, out=tmp_q)
    wo_b_dense_gemm_mxfp8(
        tmp_q,
        weights.wo_b,
        out=output,
        alpha=alpha_one,
        expected_m=expected_m,
    )
    if return_3d:
        return output
    return output[:, :, 0]

__all__ = [
    "FP8_E4M3_MAX",
    "MXFP8Rows",
    "MXFP8_SCALE_VEC_SIZE",
    "WO_A_INPUT_QUANT_GROUP_SIZE",
    "WOProjectionBinding",
    "WOProjectionInvRopeBinding",
    "WOProjectionMXFP8Weights",
    "WOProjectionScratchCaps",
    "WOProjectionScratchPlan",
    "WOProjectionWorkspace",
    "build_wo_projection_binding",
    "build_wo_projection_inv_rope_binding",
    "dequantize_mxfp8_rows_torch",
    "empty_dense_gemm_mnl_view",
    "empty_mxfp8_rows_for_dense_gemm",
    "empty_wo_projection_workspace",
    "pack_fp8_block_scaled_weight_mxfp8",
    "pack_mxfp8_scales_for_dense_gemm",
    "pack_wo_projection_fp8_block_scaled_weights_mxfp8",
    "plan_wo_projection_scratch",
    "quantize_mxfp8_rows_torch",
    "quantize_wo_a_input_inv_rope_mxfp8",
    "quantize_wo_a_input_mxfp8",
    "quantize_wo_b_input_mxfp8",
    "quantize_wo_projection_weights_mxfp8_torch",
    "wo_a_dense_gemm_mxfp8",
    "wo_b_dense_gemm_mxfp8",
    "wo_projection_inv_rope_mxfp8",
    "wo_projection_mxfp8",
]
