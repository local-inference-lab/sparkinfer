"""CuTeDSL kernels for the mHC residual path."""

from __future__ import annotations

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

from b12x.cute.compiler import KernelCompileSpec, launch as b12x_launch
from b12x.cute.utils import current_cuda_stream


_MHC_MULT = 4
_TOKENS = 1
_HIDDEN = 4096
_TOTAL_K = _MHC_MULT * _HIDDEN
_SPLIT_K = 64
_SOURCE_TILE_H = 128
_SOURCE_TILES = _HIDDEN // _SOURCE_TILE_H
_MIXES = 24
_PARTIALS = 1 + _MIXES
_PARTIALS_PER_CTA = 2
# Partials handled per post_pre-partial CTA. mix_groups = ceil(25/this), so the
# partial-kernel grid is (32 source tiles x mix_groups). 4 (-> 7 groups, 224
# CTAs) maximizes fn-read parallelism without excess grid-scheduling overhead.
_POST_PRE_PARTIALS_PER_CTA = 4
_THREADS = 128
_POST_PRE_CHUNK = 12

# --- Gram-trick split finalize (multi-CTA fuse_norm, no per-h norm reduction) -
# The post_pre-partial kernel additionally reduces the 4x4 Gram of residual_out
# G[m,m'] = sum_h ro[m,h] ro[m',h] (10 unique entries) into free partials rows
# [32, 64) (one row per source tile). The finalize then gets sum_h y^2 =
# pre^T G pre as a scalar, so it no longer reduces over hidden and can run
# multi-CTA (one block per hidden tile) like the no-norm path. 10 packed pairs:
#   0:(0,0) 1:(1,1) 2:(2,2) 3:(3,3) 4:(0,1) 5:(0,2) 6:(0,3) 7:(1,2) 8:(1,3) 9:(2,3)
_GRAM_PAIRS = 10
_GRAM_ROW0 = 32  # gram[tile] stored at partials[token, 32 + tile, 0:10]
# 1024 threads/CTA -> 4 hidden tiles (CTAs); fastest finalize in the sweep.
_GRAM_BLOCK_H = 1024


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> cutlass.cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    return cute_tensor


def _tensor_meta_key(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _norm_weight_kernel_tensor(
    norm_weight: torch.Tensor | None,
    fallback: torch.Tensor,
) -> cutlass.cute.Tensor:
    if norm_weight is None:
        return _to_kernel_tensor(fallback, cutlass.BFloat16)
    if norm_weight.dtype == torch.bfloat16:
        return _to_kernel_tensor(norm_weight, cutlass.BFloat16)
    if norm_weight.dtype == torch.float32:
        return _to_kernel_tensor(norm_weight, cutlass.Float32)
    raise ValueError(f"norm_weight must be bf16 or fp32, got {norm_weight.dtype}")


@lru_cache(maxsize=8)
def _finalize_storage_cls(num_threads: int, include_y: bool):
    class FinalizeStorage:
        pass

    annotations = {
        "partials": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, num_threads],
            16,
        ],
        "pre": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT],
            16,
        ],
        "post": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT],
            16,
        ],
        "comb": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT * _MHC_MULT],
            16,
        ],
    }
    if include_y:
        annotations["y"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, _HIDDEN],
            16,
        ]
    FinalizeStorage.__annotations__ = annotations
    return cute.struct(FinalizeStorage)


@lru_cache(maxsize=2)
def _post_pre_partial_group_storage_cls(compute_gram: bool = False):
    class PostPrePartialGroupStorage:
        pass

    annotations = {
        "warp_sums": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Float32, _POST_PRE_PARTIALS_PER_CTA * (_THREADS // 32)
            ],
            16,
        ],
    }
    if compute_gram:
        annotations["gram_sums"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _GRAM_PAIRS * (_THREADS // 32)],
            16,
        ]
    PostPrePartialGroupStorage.__annotations__ = annotations
    return cute.struct(PostPrePartialGroupStorage)


@cute.jit
def _warp_allreduce_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


class MHCPostPrePartialKernel:
    num_threads = _THREADS
    hidden_size = _HIDDEN
    total_k = _TOTAL_K
    split_k = _SPLIT_K
    source_tile_h = _SOURCE_TILE_H
    source_tiles = _SOURCE_TILES
    partials = _PARTIALS
    partials_per_cta = _POST_PRE_PARTIALS_PER_CTA

    def __init__(self, *, tokens: int = _TOKENS, compute_gram: bool = False):
        self.tokens = int(tokens)
        # When True, the partial_group==0 CTAs also reduce the 4x4 Gram of
        # residual_out into partials rows [32, 64) for the Gram-trick finalize.
        self.compute_gram = bool(compute_gram)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(x.element_type != cutlass.BFloat16):
            raise TypeError("x must be BFloat16")
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(prev_post.element_type != cutlass.Float32):
            raise TypeError("prev_post must be Float32")
        if const_expr(prev_comb.element_type != cutlass.Float32):
            raise TypeError("prev_comb must be Float32")
        if const_expr(fn.element_type != cutlass.Float32):
            raise TypeError("fn must be Float32")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        if const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        if const_expr(x.shape != (self.tokens, _HIDDEN)):
            raise ValueError("x must have shape (tokens, 4096)")
        if const_expr(residual.shape != (self.tokens, _MHC_MULT, _HIDDEN)):
            raise ValueError("residual must have shape (tokens, 4, 4096)")
        if const_expr(prev_post.shape != (self.tokens, _MHC_MULT)):
            raise ValueError("prev_post must have shape (tokens, 4)")
        if const_expr(prev_comb.shape != (self.tokens, _MHC_MULT, _MHC_MULT)):
            raise ValueError("prev_comb must have shape (tokens, 4, 4)")
        if const_expr(fn.shape != (_MIXES, _TOTAL_K)):
            raise ValueError("fn must have shape (24, 16384)")
        if const_expr(partials.shape != (self.tokens, _SPLIT_K, _PARTIALS)):
            raise ValueError("partials must have shape (tokens, 64, 25)")
        if const_expr(out.shape != (self.tokens, _MHC_MULT, _HIDDEN)):
            raise ValueError("out must have shape (tokens, 4, 4096)")

        self.kernel(x, residual, prev_post, prev_comb, fn, partials, out).launch(
            grid=(
                self.source_tiles,
                (self.partials + self.partials_per_cta - 1) // self.partials_per_cta,
                self.tokens,
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
    ):
        hidden_tile, partial_group, token = cute.arch.block_idx()
        tidx = cute.arch.thread_idx()[0]
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        nwarps = self.num_threads // 32
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_post_pre_partial_group_storage_cls(self.compute_gram))
        warp_sums = storage.warp_sums.get_tensor(
            cute.make_layout(
                (_POST_PRE_PARTIALS_PER_CTA, self.num_threads // 32),
                stride=(self.num_threads // 32, 1),
            )
        )
        if const_expr(self.compute_gram):
            gram_sums = storage.gram_sums.get_tensor(
                cute.make_layout((_GRAM_PAIRS, nwarps), stride=(nwarps, 1))
            )

        partial0 = partial_group * Int32(self.partials_per_cta)
        h = hidden_tile * Int32(self.source_tile_h) + tidx
        xh = Float32(x[token, h])
        r0 = Float32(residual[token, Int32(0), h])
        r1 = Float32(residual[token, Int32(1), h])
        r2 = Float32(residual[token, Int32(2), h])
        r3 = Float32(residual[token, Int32(3), h])

        o0 = (
            Float32(prev_post[token, Int32(0)]) * xh
            + Float32(prev_comb[token, Int32(0), Int32(0)]) * r0
            + Float32(prev_comb[token, Int32(1), Int32(0)]) * r1
            + Float32(prev_comb[token, Int32(2), Int32(0)]) * r2
            + Float32(prev_comb[token, Int32(3), Int32(0)]) * r3
        ).to(cutlass.BFloat16)
        o1 = (
            Float32(prev_post[token, Int32(1)]) * xh
            + Float32(prev_comb[token, Int32(0), Int32(1)]) * r0
            + Float32(prev_comb[token, Int32(1), Int32(1)]) * r1
            + Float32(prev_comb[token, Int32(2), Int32(1)]) * r2
            + Float32(prev_comb[token, Int32(3), Int32(1)]) * r3
        ).to(cutlass.BFloat16)
        o2 = (
            Float32(prev_post[token, Int32(2)]) * xh
            + Float32(prev_comb[token, Int32(0), Int32(2)]) * r0
            + Float32(prev_comb[token, Int32(1), Int32(2)]) * r1
            + Float32(prev_comb[token, Int32(2), Int32(2)]) * r2
            + Float32(prev_comb[token, Int32(3), Int32(2)]) * r3
        ).to(cutlass.BFloat16)
        o3 = (
            Float32(prev_post[token, Int32(3)]) * xh
            + Float32(prev_comb[token, Int32(0), Int32(3)]) * r0
            + Float32(prev_comb[token, Int32(1), Int32(3)]) * r1
            + Float32(prev_comb[token, Int32(2), Int32(3)]) * r2
            + Float32(prev_comb[token, Int32(3), Int32(3)]) * r3
        ).to(cutlass.BFloat16)

        if partial_group == Int32(0):
            out[token, Int32(0), h] = o0
            out[token, Int32(1), h] = o1
            out[token, Int32(2), h] = o2
            out[token, Int32(3), h] = o3

        r0 = Float32(o0)
        r1 = Float32(o1)
        r2 = Float32(o2)
        r3 = Float32(o3)

        # Optional 4x4 Gram of residual_out (only the residual_out-owning group).
        if const_expr(self.compute_gram):
            if partial_group == Int32(0):
                gvals = cute.make_rmem_tensor(
                    cute.make_layout((_GRAM_PAIRS,), stride=(1,)),
                    Float32,
                )
                gvals[0] = r0 * r0
                gvals[1] = r1 * r1
                gvals[2] = r2 * r2
                gvals[3] = r3 * r3
                gvals[4] = r0 * r1
                gvals[5] = r0 * r2
                gvals[6] = r0 * r3
                gvals[7] = r1 * r2
                gvals[8] = r1 * r3
                gvals[9] = r2 * r3
                for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                    gvals[gp] = _warp_allreduce_sum(gvals[gp])
                if lane == Int32(0):
                    for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                        gram_sums[gp, warp] = gvals[gp]

        values = cute.make_rmem_tensor(
            cute.make_layout((_POST_PRE_PARTIALS_PER_CTA,), stride=(1,)),
            Float32,
        )
        for slot in cutlass.range_constexpr(_POST_PRE_PARTIALS_PER_CTA):
            partial = partial0 + Int32(slot)
            value = Float32(0.0)
            if partial == Int32(0):
                value = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
            elif partial < Int32(self.partials):
                mix = partial - Int32(1)
                value = (
                    Float32(fn[mix, h]) * r0
                    + Float32(fn[mix, Int32(self.hidden_size) + h]) * r1
                    + Float32(fn[mix, Int32(2 * self.hidden_size) + h]) * r2
                    + Float32(fn[mix, Int32(3 * self.hidden_size) + h]) * r3
                )
            values[slot] = _warp_allreduce_sum(value)
        if lane == Int32(0):
            for slot in cutlass.range_constexpr(_POST_PRE_PARTIALS_PER_CTA):
                warp_sums[slot, warp] = values[slot]
        cute.arch.sync_threads()

        if tidx == Int32(0):
            for slot in cutlass.range_constexpr(_POST_PRE_PARTIALS_PER_CTA):
                total = Float32(0.0)
                src_warp = Int32(0)
                while src_warp < Int32(self.num_threads // 32):
                    total += Float32(warp_sums[slot, src_warp])
                    src_warp += Int32(1)
                partial = partial0 + Int32(slot)
                if partial < Int32(self.partials):
                    partials[token, hidden_tile, partial] = total

        if const_expr(self.compute_gram):
            if partial_group == Int32(0):
                if tidx == Int32(0):
                    for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                        gtotal = Float32(0.0)
                        src_warp = Int32(0)
                        while src_warp < Int32(nwarps):
                            gtotal += Float32(gram_sums[gp, src_warp])
                            src_warp += Int32(1)
                        partials[token, Int32(_GRAM_ROW0) + hidden_tile, gp] = gtotal


class MHCFinalizeGramKernel:
    """Multi-CTA fuse_norm finalize using the residual_out Gram matrix.

    The partial kernel (compute_gram=True) provides G[m,m'] in partials rows
    [32, 64), so sum_h y^2 = pre^T G pre is a scalar -- no per-h norm reduction.
    Each CTA owns one hidden tile, redundantly reduces partials+Gram and runs the
    Sinkhorn (cheap), then writes its y tile in a single pass (no cross-CTA sync).
    """

    num_threads = _GRAM_BLOCK_H
    block_h = _GRAM_BLOCK_H
    hidden_size = _HIDDEN
    source_tiles = _SOURCE_TILES
    mixes = _MIXES
    partials = _PARTIALS
    gram_row0 = _GRAM_ROW0
    gram_pairs = _GRAM_PAIRS

    def __init__(
        self,
        *,
        tokens: int = _TOKENS,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_eps: float,
        fuse_norm: bool = True,
    ):
        self.tokens = int(tokens)
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.norm_eps = float(norm_eps)
        # When False, norm_weight is ignored: y is the raw collapsed activation
        # (no Gram reduction, no RMSNorm). The partial then skips the Gram.
        self.fuse_norm = bool(fuse_norm)

    @cute.jit
    def __call__(
        self,
        residual: cute.Tensor,
        partials: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        y: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        norm_weight: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        if const_expr(y.element_type != cutlass.BFloat16):
            raise TypeError("y must be BFloat16")
        if const_expr(
            self.fuse_norm
            and norm_weight.element_type != cutlass.BFloat16
            and norm_weight.element_type != cutlass.Float32
        ):
            raise TypeError("norm_weight must be BFloat16 or Float32")
        if const_expr(residual.shape != (self.tokens, _MHC_MULT, _HIDDEN)):
            raise ValueError("residual must have shape (tokens, 4, 4096)")
        if const_expr(partials.shape != (self.tokens, _SPLIT_K, _PARTIALS)):
            raise ValueError("partials must have shape (tokens, 64, 25)")
        if const_expr(y.shape != (self.tokens, _HIDDEN)):
            raise ValueError("y must have shape (tokens, 4096)")
        if const_expr(self.fuse_norm and norm_weight.shape != (_HIDDEN,)):
            raise ValueError("norm_weight must have shape (4096,)")

        self.kernel(residual, partials, scale, bias, y, post, comb, norm_weight).launch(
            grid=(self.hidden_size // self.block_h, self.tokens, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        residual: cute.Tensor,
        partials: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        y: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        norm_weight: cute.Tensor,
    ):
        tile_h, token, _ = cute.arch.block_idx()
        tile_h = Int32(tile_h)
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_finalize_storage_cls(self.num_threads, False))
        s_pre = storage.pre.get_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)))
        s_post = storage.post.get_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)))

        sums = cute.make_rmem_tensor(
            cute.make_layout((self.partials,), stride=(1,)), Float32
        )
        gram = cute.make_rmem_tensor(
            cute.make_layout((self.gram_pairs,), stride=(1,)), Float32
        )
        if tidx < Int32(32):
            for column in cutlass.range_constexpr(_PARTIALS):
                value = Float32(0.0)
                if tidx < Int32(self.source_tiles):
                    value = Float32(partials[token, tidx, column])
                sums[column] = _warp_allreduce_sum(value)
            if const_expr(self.fuse_norm):
                for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                    gvalue = Float32(0.0)
                    if tidx < Int32(self.source_tiles):
                        gvalue = Float32(
                            partials[token, Int32(self.gram_row0) + tidx, gp]
                        )
                    gram[gp] = _warp_allreduce_sum(gvalue)

        if tidx == Int32(0):
            total_sqsum = Float32(sums[0])
            mixes = cute.make_rmem_tensor(
                cute.make_layout((self.mixes,), stride=(1,)), Float32
            )
            for mix in cutlass.range_constexpr(_MIXES):
                mixes[mix] = Float32(sums[mix + 1])
            inv_rms = cute.math.rsqrt(
                total_sqsum / Float32(_TOTAL_K) + Float32(self.rms_eps), fastmath=True
            )
            for mix in cutlass.range_constexpr(_MIXES):
                mixes[mix] = mixes[mix] * inv_rms

            s0 = Float32(scale[0])
            s1 = Float32(scale[1])
            s2 = Float32(scale[2])
            one = Float32(1.0)
            two = Float32(2.0)
            eps = Float32(self.hc_eps)

            pre0 = one / (one + cute.math.exp(-(mixes[0] * s0 + Float32(bias[0])), fastmath=True)) + eps
            pre1 = one / (one + cute.math.exp(-(mixes[1] * s0 + Float32(bias[1])), fastmath=True)) + eps
            pre2 = one / (one + cute.math.exp(-(mixes[2] * s0 + Float32(bias[2])), fastmath=True)) + eps
            pre3 = one / (one + cute.math.exp(-(mixes[3] * s0 + Float32(bias[3])), fastmath=True)) + eps
            s_pre[0] = pre0
            s_pre[1] = pre1
            s_pre[2] = pre2
            s_pre[3] = pre3

            post0 = two / (one + cute.math.exp(-(mixes[4] * s1 + Float32(bias[4])), fastmath=True))
            post1 = two / (one + cute.math.exp(-(mixes[5] * s1 + Float32(bias[5])), fastmath=True))
            post2 = two / (one + cute.math.exp(-(mixes[6] * s1 + Float32(bias[6])), fastmath=True))
            post3 = two / (one + cute.math.exp(-(mixes[7] * s1 + Float32(bias[7])), fastmath=True))
            if tile_h == Int32(0):
                post[token, 0] = post0
                post[token, 1] = post1
                post[token, 2] = post2
                post[token, 3] = post3

            c00 = mixes[8] * s2 + Float32(bias[8])
            c01 = mixes[9] * s2 + Float32(bias[9])
            c02 = mixes[10] * s2 + Float32(bias[10])
            c03 = mixes[11] * s2 + Float32(bias[11])
            c10 = mixes[12] * s2 + Float32(bias[12])
            c11 = mixes[13] * s2 + Float32(bias[13])
            c12 = mixes[14] * s2 + Float32(bias[14])
            c13 = mixes[15] * s2 + Float32(bias[15])
            c20 = mixes[16] * s2 + Float32(bias[16])
            c21 = mixes[17] * s2 + Float32(bias[17])
            c22 = mixes[18] * s2 + Float32(bias[18])
            c23 = mixes[19] * s2 + Float32(bias[19])
            c30 = mixes[20] * s2 + Float32(bias[20])
            c31 = mixes[21] * s2 + Float32(bias[21])
            c32 = mixes[22] * s2 + Float32(bias[22])
            c33 = mixes[23] * s2 + Float32(bias[23])

            m0 = c00
            if c01 > m0:
                m0 = c01
            if c02 > m0:
                m0 = c02
            if c03 > m0:
                m0 = c03
            m1 = c10
            if c11 > m1:
                m1 = c11
            if c12 > m1:
                m1 = c12
            if c13 > m1:
                m1 = c13
            m2 = c20
            if c21 > m2:
                m2 = c21
            if c22 > m2:
                m2 = c22
            if c23 > m2:
                m2 = c23
            m3 = c30
            if c31 > m3:
                m3 = c31
            if c32 > m3:
                m3 = c32
            if c33 > m3:
                m3 = c33

            c00 = cute.math.exp(c00 - m0, fastmath=True)
            c01 = cute.math.exp(c01 - m0, fastmath=True)
            c02 = cute.math.exp(c02 - m0, fastmath=True)
            c03 = cute.math.exp(c03 - m0, fastmath=True)
            c10 = cute.math.exp(c10 - m1, fastmath=True)
            c11 = cute.math.exp(c11 - m1, fastmath=True)
            c12 = cute.math.exp(c12 - m1, fastmath=True)
            c13 = cute.math.exp(c13 - m1, fastmath=True)
            c20 = cute.math.exp(c20 - m2, fastmath=True)
            c21 = cute.math.exp(c21 - m2, fastmath=True)
            c22 = cute.math.exp(c22 - m2, fastmath=True)
            c23 = cute.math.exp(c23 - m2, fastmath=True)
            c30 = cute.math.exp(c30 - m3, fastmath=True)
            c31 = cute.math.exp(c31 - m3, fastmath=True)
            c32 = cute.math.exp(c32 - m3, fastmath=True)
            c33 = cute.math.exp(c33 - m3, fastmath=True)

            r0s = c00 + c01 + c02 + c03
            r1s = c10 + c11 + c12 + c13
            r2s = c20 + c21 + c22 + c23
            r3s = c30 + c31 + c32 + c33
            inv_r0 = cute.arch.rcp_approx(r0s)
            inv_r1 = cute.arch.rcp_approx(r1s)
            inv_r2 = cute.arch.rcp_approx(r2s)
            inv_r3 = cute.arch.rcp_approx(r3s)
            c00 = c00 * inv_r0 + eps
            c01 = c01 * inv_r0 + eps
            c02 = c02 * inv_r0 + eps
            c03 = c03 * inv_r0 + eps
            c10 = c10 * inv_r1 + eps
            c11 = c11 * inv_r1 + eps
            c12 = c12 * inv_r1 + eps
            c13 = c13 * inv_r1 + eps
            c20 = c20 * inv_r2 + eps
            c21 = c21 * inv_r2 + eps
            c22 = c22 * inv_r2 + eps
            c23 = c23 * inv_r2 + eps
            c30 = c30 * inv_r3 + eps
            c31 = c31 * inv_r3 + eps
            c32 = c32 * inv_r3 + eps
            c33 = c33 * inv_r3 + eps

            col0 = c00 + c10 + c20 + c30 + eps
            col1 = c01 + c11 + c21 + c31 + eps
            col2 = c02 + c12 + c22 + c32 + eps
            col3 = c03 + c13 + c23 + c33 + eps
            inv_col0 = cute.arch.rcp_approx(col0)
            inv_col1 = cute.arch.rcp_approx(col1)
            inv_col2 = cute.arch.rcp_approx(col2)
            inv_col3 = cute.arch.rcp_approx(col3)
            c00 = c00 * inv_col0
            c10 = c10 * inv_col0
            c20 = c20 * inv_col0
            c30 = c30 * inv_col0
            c01 = c01 * inv_col1
            c11 = c11 * inv_col1
            c21 = c21 * inv_col1
            c31 = c31 * inv_col1
            c02 = c02 * inv_col2
            c12 = c12 * inv_col2
            c22 = c22 * inv_col2
            c32 = c32 * inv_col2
            c03 = c03 * inv_col3
            c13 = c13 * inv_col3
            c23 = c23 * inv_col3
            c33 = c33 * inv_col3

            for _ in cutlass.range_constexpr(self.sinkhorn_iters - 1):
                r0s = c00 + c01 + c02 + c03 + eps
                r1s = c10 + c11 + c12 + c13 + eps
                r2s = c20 + c21 + c22 + c23 + eps
                r3s = c30 + c31 + c32 + c33 + eps
                inv_r0 = cute.arch.rcp_approx(r0s)
                inv_r1 = cute.arch.rcp_approx(r1s)
                inv_r2 = cute.arch.rcp_approx(r2s)
                inv_r3 = cute.arch.rcp_approx(r3s)
                c00 = c00 * inv_r0
                c01 = c01 * inv_r0
                c02 = c02 * inv_r0
                c03 = c03 * inv_r0
                c10 = c10 * inv_r1
                c11 = c11 * inv_r1
                c12 = c12 * inv_r1
                c13 = c13 * inv_r1
                c20 = c20 * inv_r2
                c21 = c21 * inv_r2
                c22 = c22 * inv_r2
                c23 = c23 * inv_r2
                c30 = c30 * inv_r3
                c31 = c31 * inv_r3
                c32 = c32 * inv_r3
                c33 = c33 * inv_r3

                col0 = c00 + c10 + c20 + c30 + eps
                col1 = c01 + c11 + c21 + c31 + eps
                col2 = c02 + c12 + c22 + c32 + eps
                col3 = c03 + c13 + c23 + c33 + eps
                inv_col0 = cute.arch.rcp_approx(col0)
                inv_col1 = cute.arch.rcp_approx(col1)
                inv_col2 = cute.arch.rcp_approx(col2)
                inv_col3 = cute.arch.rcp_approx(col3)
                c00 = c00 * inv_col0
                c10 = c10 * inv_col0
                c20 = c20 * inv_col0
                c30 = c30 * inv_col0
                c01 = c01 * inv_col1
                c11 = c11 * inv_col1
                c21 = c21 * inv_col1
                c31 = c31 * inv_col1
                c02 = c02 * inv_col2
                c12 = c12 * inv_col2
                c22 = c22 * inv_col2
                c32 = c32 * inv_col2
                c03 = c03 * inv_col3
                c13 = c13 * inv_col3
                c23 = c23 * inv_col3
                c33 = c33 * inv_col3

            if tile_h == Int32(0):
                comb[token, 0, 0] = c00
                comb[token, 0, 1] = c01
                comb[token, 0, 2] = c02
                comb[token, 0, 3] = c03
                comb[token, 1, 0] = c10
                comb[token, 1, 1] = c11
                comb[token, 1, 2] = c12
                comb[token, 1, 3] = c13
                comb[token, 2, 0] = c20
                comb[token, 2, 1] = c21
                comb[token, 2, 2] = c22
                comb[token, 2, 3] = c23
                comb[token, 3, 0] = c30
                comb[token, 3, 1] = c31
                comb[token, 3, 2] = c32
                comb[token, 3, 3] = c33

            if const_expr(self.fuse_norm):
                # sum_h y^2 = pre^T G pre  (G symmetric, 10 packed entries)
                sy2 = (
                    pre0 * pre0 * Float32(gram[0])
                    + pre1 * pre1 * Float32(gram[1])
                    + pre2 * pre2 * Float32(gram[2])
                    + pre3 * pre3 * Float32(gram[3])
                    + two
                    * (
                        pre0 * pre1 * Float32(gram[4])
                        + pre0 * pre2 * Float32(gram[5])
                        + pre0 * pre3 * Float32(gram[6])
                        + pre1 * pre2 * Float32(gram[7])
                        + pre1 * pre3 * Float32(gram[8])
                        + pre2 * pre3 * Float32(gram[9])
                    )
                )
                s_post[0] = cute.math.rsqrt(
                    sy2 / Float32(self.hidden_size) + Float32(self.norm_eps),
                    fastmath=True,
                )

        cute.arch.sync_threads()

        p0 = Float32(s_pre[0])
        p1 = Float32(s_pre[1])
        p2 = Float32(s_pre[2])
        p3 = Float32(s_pre[3])
        h = tile_h * Int32(self.block_h) + tidx
        ro0 = Float32(residual[token, 0, h])
        ro1 = Float32(residual[token, 1, h])
        ro2 = Float32(residual[token, 2, h])
        ro3 = Float32(residual[token, 3, h])
        # Round y_prenorm to bf16 before applying the norm, matching the
        # reference (and vLLM), so the only difference is the (negligible)
        # fp32-vs-bf16 sum-of-squares used for rms.
        y_pre = (p0 * ro0 + p1 * ro1 + p2 * ro2 + p3 * ro3).to(cutlass.BFloat16)
        if const_expr(self.fuse_norm):
            rms = Float32(s_post[0])
            y[token, h] = (
                Float32(y_pre) * rms * Float32(norm_weight[h])
            ).to(cutlass.BFloat16)
        else:
            y[token, h] = y_pre


@lru_cache(maxsize=64)
def _post_pre_partial_kernel(
    tokens: int, compute_gram: bool = False
) -> MHCPostPrePartialKernel:
    return MHCPostPrePartialKernel(
        tokens=tokens, compute_gram=compute_gram
    )


@lru_cache(maxsize=64)
def _finalize_gram_kernel(
    tokens: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
) -> MHCFinalizeGramKernel:
    return MHCFinalizeGramKernel(
        tokens=tokens,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        fuse_norm=fuse_norm,
    )


def run_mhc_post_pre_partial(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool = False,
) -> None:
    tokens = int(x.shape[0])
    compute_gram = bool(compute_gram)
    args = (
        _to_kernel_tensor(x, cutlass.BFloat16),
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(prev_post, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(prev_comb, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(partials, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(out, cutlass.BFloat16),
        current_cuda_stream(),
    )
    cache_key = (
        _tensor_meta_key(x),
        _tensor_meta_key(residual),
        _tensor_meta_key(prev_post),
        _tensor_meta_key(prev_comb),
        _tensor_meta_key(fn),
        _tensor_meta_key(partials),
        _tensor_meta_key(out),
    )
    b12x_launch(
        _post_pre_partial_kernel(tokens, compute_gram),
        compile_spec=KernelCompileSpec.from_key(
            "integration.residual.mhc_post_pre_partial_hidden4096_hctile128_all4",
            2,
            (
                ("tokens", tokens),
                ("partials_per_cta", _POST_PRE_PARTIALS_PER_CTA),
                ("compute_gram", compute_gram),
                cache_key,
            ),
        ),
        compile_args=args,
        runtime_args=args,
    )


def run_mhc_finalize_gram(
    *,
    residual: torch.Tensor,
    partials: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> None:
    rms_eps = float(rms_eps)
    hc_eps = float(hc_eps)
    sinkhorn_iters = int(sinkhorn_iters)
    norm_eps = float(norm_eps)
    fuse_norm = norm_weight is not None
    norm_weight_tensor = _norm_weight_kernel_tensor(norm_weight, y)
    tokens = int(residual.shape[0])
    args = (
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(partials, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(scale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(bias, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(y, cutlass.BFloat16),
        _to_kernel_tensor(post, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(comb, cutlass.Float32, assumed_align=4),
        norm_weight_tensor,
        current_cuda_stream(),
    )
    cache_key = (
        ("tokens", tokens),
        ("block_h", _GRAM_BLOCK_H),
        ("source_tiles", _SOURCE_TILES),
        ("gram_row0", _GRAM_ROW0),
        ("impl", "finalize_gram_multicta_v1"),
        ("math", "fast_exp_exact_sigmoid_rcp_approx_sinkhorn"),
        ("fuse_norm", fuse_norm),
        ("norm_eps", norm_eps if fuse_norm else 0.0),
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        _tensor_meta_key(residual),
        _tensor_meta_key(partials),
        _tensor_meta_key(scale),
        _tensor_meta_key(bias),
        _tensor_meta_key(y),
        _tensor_meta_key(post),
        _tensor_meta_key(comb),
        _tensor_meta_key(norm_weight) if norm_weight is not None else None,
    )
    b12x_launch(
        _finalize_gram_kernel(
            tokens, rms_eps, hc_eps, sinkhorn_iters, norm_eps, fuse_norm
        ),
        compile_spec=KernelCompileSpec.from_key(
            "integration.residual.mhc_finalize_gram_hidden4096",
            1,
            cache_key,
        ),
        compile_args=args,
        runtime_args=args,
    )


__all__ = [
    "run_mhc_finalize_gram",
    "run_mhc_post_pre_partial",
]
