from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from b12x.gemm.dense import dense_gemm

FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
MXFP8_SCALE_VEC_SIZE = 32
MXFP8_SCALE_ROW_TILE = 128
MXFP8_SCALE_K_TILE = 4


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
    safe = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scale_exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(safe)), -127.0), 127.0)
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
    tl.store(scale_rows + group * tokens * sf_cols + token * sf_cols + chunk, scale_u8)

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


def pack_fp8_block_scaled_weight_mxfp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int = 1,
) -> MXFP8Rows:
    """Pack checkpoint FP8 block-scaled weights for native MXFP8 dense GEMM.

    `weight` is kept in FP8 E4M3 form. `scale` is the DSV4-style 128x128 block
    scale and is expanded to the MXFP8 row/32-column scale layout.
    """

    _check_gpu_tensor("weight", weight)
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if m <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError("m, k, and num_groups must be positive")
    _check_mxfp8_k(k)

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
    """Quantize grouped WO-A input `[tokens, groups, group_width]` for dense GEMM."""

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
    _quantize_grouped_tgd_to_tdg_kernel[(tokens, groups, group_width // MXFP8_SCALE_VEC_SIZE)](
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
        BLOCK=MXFP8_SCALE_VEC_SIZE,
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
    _check_mxfp8_rows_storage(
        workspace.x_q,
        m=tokens,
        k=weights.group_width,
        num_groups=weights.groups,
    )
    _check_dense_gemm_mnl_view("workspace.tmp", workspace.tmp)
    if workspace.tmp.shape != (tokens, weights.rank, weights.groups):
        raise ValueError(
            "workspace.tmp must have shape "
            f"{(tokens, weights.rank, weights.groups)}, got {tuple(workspace.tmp.shape)}"
        )
    if workspace.tmp.dtype != torch.bfloat16:
        raise ValueError(f"workspace.tmp must be bfloat16, got {workspace.tmp.dtype}")
    _check_mxfp8_rows_storage(
        workspace.tmp_q,
        m=tokens,
        k=weights.rank * weights.groups,
        num_groups=1,
    )
    _check_dense_gemm_mnl_view("workspace.output", workspace.output)
    if workspace.output.shape != (tokens, weights.hidden, 1):
        raise ValueError(
            "workspace.output must have shape "
            f"{(tokens, weights.hidden, 1)}, got {tuple(workspace.output.shape)}"
        )
    if workspace.output.dtype != torch.bfloat16:
        raise ValueError(f"workspace.output must be bfloat16, got {workspace.output.dtype}")


def wo_a_dense_gemm_mxfp8(
    x_tdg: MXFP8Rows,
    wo_a_rdg: MXFP8Rows,
    *,
    out: torch.Tensor | None = None,
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
    return dense_gemm(
        (x_tdg.values, x_tdg.scale_mma),
        (wo_a_rdg.values, wo_a_rdg.scale_mma),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        out=out,
    )


def wo_b_dense_gemm_mxfp8(
    tmp_tgr_group_major: MXFP8Rows,
    wo_b_hgr: MXFP8Rows,
    *,
    out: torch.Tensor | None = None,
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
    )


def wo_projection_mxfp8(
    source_tgd: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    workspace: WOProjectionWorkspace,
    *,
    return_3d: bool = False,
) -> torch.Tensor:
    """Run the native MXFP8 WO-A/WO-B projection.

    `source_tgd` is `[tokens, groups, group_width]`. The default return value
    is the SGLang-friendly `[tokens, hidden]` view over `workspace.output`.
    """

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
    _check_wo_projection_workspace(workspace, tokens=tokens, weights=weights)

    quantize_wo_a_input_mxfp8(source_tgd, out=workspace.x_q)
    wo_a_dense_gemm_mxfp8(workspace.x_q, weights.wo_a, out=workspace.tmp)
    quantize_wo_b_input_mxfp8(workspace.tmp, out=workspace.tmp_q)
    wo_b_dense_gemm_mxfp8(workspace.tmp_q, weights.wo_b, out=workspace.output)
    if return_3d:
        return workspace.output
    return workspace.output[:, :, 0]


__all__ = [
    "FP8_E4M3_MAX",
    "MXFP8Rows",
    "MXFP8_SCALE_VEC_SIZE",
    "WOProjectionMXFP8Weights",
    "WOProjectionWorkspace",
    "dequantize_mxfp8_rows_torch",
    "empty_dense_gemm_mnl_view",
    "empty_mxfp8_rows_for_dense_gemm",
    "empty_wo_projection_workspace",
    "pack_fp8_block_scaled_weight_mxfp8",
    "pack_mxfp8_scales_for_dense_gemm",
    "pack_wo_projection_fp8_block_scaled_weights_mxfp8",
    "quantize_mxfp8_rows_torch",
    "quantize_wo_a_input_mxfp8",
    "quantize_wo_b_input_mxfp8",
    "quantize_wo_projection_weights_mxfp8_torch",
    "wo_a_dense_gemm_mxfp8",
    "wo_b_dense_gemm_mxfp8",
    "wo_projection_mxfp8",
]
