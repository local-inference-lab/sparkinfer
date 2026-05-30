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
_POST_PRE_PARTIALS_PER_CTA = 4
_THREADS = 128
_POST_PRE_CHUNK = 12


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


@lru_cache(maxsize=4)
def _shared_storage_cls(num_threads: int = _THREADS):
    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "partials": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _PARTIALS * num_threads],
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
        "y": cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, _HIDDEN],
            16,
        ],
    }
    return cute.struct(SharedStorage)


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


@lru_cache(maxsize=1)
def _partial_group_storage_cls():
    class PartialGroupStorage:
        pass

    PartialGroupStorage.__annotations__ = {
        "warp_sums": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Float32, _PARTIALS_PER_CTA * (_THREADS // 32)
            ],
            16,
        ],
    }
    return cute.struct(PartialGroupStorage)


@lru_cache(maxsize=1)
def _post_pre_partial_group_storage_cls():
    class PostPrePartialGroupStorage:
        pass

    PostPrePartialGroupStorage.__annotations__ = {
        "warp_sums": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Float32, _POST_PRE_PARTIALS_PER_CTA * (_THREADS // 32)
            ],
            16,
        ],
    }
    return cute.struct(PostPrePartialGroupStorage)


@cute.jit
def _warp_allreduce_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


class MHCPartialToken1Hidden4096Kernel:
    num_threads = _THREADS
    hidden_size = _HIDDEN
    total_k = _TOTAL_K
    split_k = _SPLIT_K
    source_tile_h = _SOURCE_TILE_H
    source_tiles = _SOURCE_TILES
    mixes = _MIXES
    partials = _PARTIALS

    @cute.jit
    def __call__(
        self,
        residual: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(fn.element_type != cutlass.Float32):
            raise TypeError("fn must be Float32")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        if const_expr(residual.shape != (_TOKENS, _MHC_MULT, _HIDDEN)):
            raise ValueError("residual must have shape (1, 4, 4096)")
        if const_expr(fn.shape != (_MIXES, _TOTAL_K)):
            raise ValueError("fn must have shape (24, 16384)")
        if const_expr(partials.shape != (_TOKENS, _SPLIT_K, _PARTIALS)):
            raise ValueError("partials must have shape (1, 64, 25)")

        self.kernel(residual, fn, partials).launch(
            grid=(
                self.source_tiles,
                (self.partials + _PARTIALS_PER_CTA - 1) // _PARTIALS_PER_CTA,
                1,
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        residual: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
    ):
        hidden_tile, partial_group, _ = cute.arch.block_idx()
        tidx = cute.arch.thread_idx()[0]
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_partial_group_storage_cls())
        warp_sums = storage.warp_sums.get_tensor(
            cute.make_layout(
                (_PARTIALS_PER_CTA, self.num_threads // 32),
                stride=(self.num_threads // 32, 1),
            )
        )

        partial0 = partial_group * Int32(_PARTIALS_PER_CTA)
        partial1 = partial0 + Int32(1)
        value0 = Float32(0.0)
        value1 = Float32(0.0)
        h = hidden_tile * Int32(self.source_tile_h) + tidx
        for hc_const in cutlass.range_constexpr(_MHC_MULT):
            linear = Int32(hc_const * _HIDDEN) + h
            rv = Float32(residual[Int32(0), Int32(hc_const), h])
            if partial0 == Int32(0):
                value0 += rv * rv
            else:
                value0 += Float32(fn[partial0 - Int32(1), linear]) * rv
            if partial1 < Int32(self.partials):
                value1 += Float32(fn[partial1 - Int32(1), linear]) * rv
        value0 = _warp_allreduce_sum(value0)
        value1 = _warp_allreduce_sum(value1)
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        if lane == Int32(0):
            warp_sums[0, warp] = value0
            warp_sums[1, warp] = value1
        cute.arch.sync_threads()

        if tidx == Int32(0):
            total0 = Float32(0.0)
            total1 = Float32(0.0)
            src_warp = Int32(0)
            while src_warp < Int32(self.num_threads // 32):
                total0 += Float32(warp_sums[0, src_warp])
                total1 += Float32(warp_sums[1, src_warp])
                src_warp += Int32(1)
            partials[0, hidden_tile, partial0] = total0
            if partial1 < Int32(self.partials):
                partials[0, hidden_tile, partial1] = total1


class MHCPostPrePartialToken1Hidden4096Kernel:
    num_threads = _THREADS
    hidden_size = _HIDDEN
    total_k = _TOTAL_K
    split_k = _SPLIT_K
    source_tile_h = _SOURCE_TILE_H
    source_tiles = _SOURCE_TILES
    partials = _PARTIALS
    partials_per_cta = _POST_PRE_PARTIALS_PER_CTA

    def __init__(self, *, tokens: int = _TOKENS):
        self.tokens = int(tokens)

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
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_post_pre_partial_group_storage_cls())
        warp_sums = storage.warp_sums.get_tensor(
            cute.make_layout(
                (_POST_PRE_PARTIALS_PER_CTA, self.num_threads // 32),
                stride=(self.num_threads // 32, 1),
            )
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
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
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


class MHCPreHidden4096Kernel:
    num_threads = _THREADS
    hidden_size = _HIDDEN
    total_k = _TOTAL_K
    mixes = _MIXES
    partials = _PARTIALS

    def __init__(
        self,
        *,
        tokens: int,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_eps: float = 0.0,
        fuse_norm: bool = False,
    ):
        self.tokens = int(tokens)
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.norm_eps = float(norm_eps)
        self.fuse_norm = bool(fuse_norm)
        self.emit_out = False
        self.source_post = False

    @cute.jit
    def __call__(
        self,
        residual: cute.Tensor,
        x: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        y: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        out: cute.Tensor,
        norm_weight: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr((self.emit_out or self.source_post) and x.element_type != cutlass.BFloat16):
            raise TypeError("x must be BFloat16")
        if const_expr(self.source_post and prev_post.element_type != cutlass.Float32):
            raise TypeError("prev_post must be Float32")
        if const_expr(self.source_post and prev_comb.element_type != cutlass.Float32):
            raise TypeError("prev_comb must be Float32")
        if const_expr(y.element_type != cutlass.BFloat16):
            raise TypeError("y must be BFloat16")
        if const_expr((self.emit_out or self.source_post) and out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        if const_expr(fn.element_type != cutlass.Float32):
            raise TypeError("fn must be Float32")
        if const_expr(scale.element_type != cutlass.Float32):
            raise TypeError("scale must be Float32")
        if const_expr(bias.element_type != cutlass.Float32):
            raise TypeError("bias must be Float32")
        if const_expr(post.element_type != cutlass.Float32):
            raise TypeError("post must be Float32")
        if const_expr(comb.element_type != cutlass.Float32):
            raise TypeError("comb must be Float32")
        if const_expr(
            self.fuse_norm
            and norm_weight.element_type != cutlass.BFloat16
            and norm_weight.element_type != cutlass.Float32
        ):
            raise TypeError("norm_weight must be BFloat16 or Float32")
        if const_expr(residual.shape != (self.tokens, _MHC_MULT, _HIDDEN)):
            raise ValueError("residual must have shape (tokens, 4, 4096)")
        if const_expr((self.emit_out or self.source_post) and x.shape != (self.tokens, _HIDDEN)):
            raise ValueError("x must have shape (tokens, 4096)")
        if const_expr(self.source_post and prev_post.shape != (self.tokens, _MHC_MULT)):
            raise ValueError("prev_post must have shape (tokens, 4)")
        if const_expr(self.source_post and prev_comb.shape != (self.tokens, _MHC_MULT, _MHC_MULT)):
            raise ValueError("prev_comb must have shape (tokens, 4, 4)")
        if const_expr(fn.shape != (_MIXES, _TOTAL_K)):
            raise ValueError("fn must have shape (24, 16384)")
        if const_expr(scale.shape != (3,)):
            raise ValueError("scale must have shape (3,)")
        if const_expr(bias.shape != (_MIXES,)):
            raise ValueError("bias must have shape (24,)")
        if const_expr(y.shape != (self.tokens, _HIDDEN)):
            raise ValueError("y must have shape (tokens, 4096)")
        if const_expr(post.shape != (self.tokens, _MHC_MULT)):
            raise ValueError("post must have shape (tokens, 4)")
        if const_expr(comb.shape != (self.tokens, _MHC_MULT, _MHC_MULT)):
            raise ValueError("comb must have shape (tokens, 4, 4)")
        if const_expr((self.emit_out or self.source_post) and out.shape != (self.tokens, _MHC_MULT, _HIDDEN)):
            raise ValueError("out must have shape (tokens, 4, 4096)")
        if const_expr(self.fuse_norm and norm_weight.shape != (_HIDDEN,)):
            raise ValueError("norm_weight must have shape (4096,)")

        self.kernel(residual, x, prev_post, prev_comb, fn, scale, bias, y, post, comb, out, norm_weight).launch(
            grid=(self.tokens, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        residual: cute.Tensor,
        x: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        y: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        out: cute.Tensor,
        norm_weight: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        token, _, _ = cute.arch.block_idx()
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_shared_storage_cls(self.num_threads))
        partial_layout = cute.make_layout(
            (self.partials, self.num_threads),
            stride=(self.num_threads, 1),
        )
        s_partials = storage.partials.get_tensor(partial_layout)
        s_pre = storage.pre.get_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)))
        s_post = storage.post.get_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)))
        s_comb = storage.comb.get_tensor(
            cute.make_layout((_MHC_MULT, _MHC_MULT), stride=(_MHC_MULT, 1))
        )
        s_y = storage.y.get_tensor(cute.make_layout((_HIDDEN,), stride=(1,)))

        if const_expr(self.source_post):
            prev_p0 = Float32(prev_post[token, 0])
            prev_p1 = Float32(prev_post[token, 1])
            prev_p2 = Float32(prev_post[token, 2])
            prev_p3 = Float32(prev_post[token, 3])
            prev_c00 = Float32(prev_comb[token, 0, 0])
            prev_c01 = Float32(prev_comb[token, 0, 1])
            prev_c02 = Float32(prev_comb[token, 0, 2])
            prev_c03 = Float32(prev_comb[token, 0, 3])
            prev_c10 = Float32(prev_comb[token, 1, 0])
            prev_c11 = Float32(prev_comb[token, 1, 1])
            prev_c12 = Float32(prev_comb[token, 1, 2])
            prev_c13 = Float32(prev_comb[token, 1, 3])
            prev_c20 = Float32(prev_comb[token, 2, 0])
            prev_c21 = Float32(prev_comb[token, 2, 1])
            prev_c22 = Float32(prev_comb[token, 2, 2])
            prev_c23 = Float32(prev_comb[token, 2, 3])
            prev_c30 = Float32(prev_comb[token, 3, 0])
            prev_c31 = Float32(prev_comb[token, 3, 1])
            prev_c32 = Float32(prev_comb[token, 3, 2])
            prev_c33 = Float32(prev_comb[token, 3, 3])

            sqsum = Float32(0.0)
            mix_acc = cute.make_rmem_tensor(
                cute.make_layout((_POST_PRE_CHUNK,), stride=(1,)),
                Float32,
            )
            for mix in cutlass.range_constexpr(_POST_PRE_CHUNK):
                mix_acc[mix] = Float32(0.0)

            post_h = tidx
            while post_h < Int32(self.hidden_size):
                xh = Float32(x[token, post_h])
                src0 = Float32(residual[token, 0, post_h])
                src1 = Float32(residual[token, 1, post_h])
                src2 = Float32(residual[token, 2, post_h])
                src3 = Float32(residual[token, 3, post_h])
                o0 = (
                    prev_p0 * xh
                    + prev_c00 * src0
                    + prev_c10 * src1
                    + prev_c20 * src2
                    + prev_c30 * src3
                ).to(cutlass.BFloat16)
                o1 = (
                    prev_p1 * xh
                    + prev_c01 * src0
                    + prev_c11 * src1
                    + prev_c21 * src2
                    + prev_c31 * src3
                ).to(cutlass.BFloat16)
                o2 = (
                    prev_p2 * xh
                    + prev_c02 * src0
                    + prev_c12 * src1
                    + prev_c22 * src2
                    + prev_c32 * src3
                ).to(cutlass.BFloat16)
                o3 = (
                    prev_p3 * xh
                    + prev_c03 * src0
                    + prev_c13 * src1
                    + prev_c23 * src2
                    + prev_c33 * src3
                ).to(cutlass.BFloat16)
                out[token, 0, post_h] = o0
                out[token, 1, post_h] = o1
                out[token, 2, post_h] = o2
                out[token, 3, post_h] = o3

                r0v = Float32(o0)
                r1v = Float32(o1)
                r2v = Float32(o2)
                r3v = Float32(o3)
                sqsum += r0v * r0v + r1v * r1v + r2v * r2v + r3v * r3v
                for mix in cutlass.range_constexpr(_POST_PRE_CHUNK):
                    mix_acc[mix] += (
                        Float32(fn[mix, post_h]) * r0v
                        + Float32(fn[mix, Int32(self.hidden_size) + post_h]) * r1v
                        + Float32(fn[mix, Int32(2 * self.hidden_size) + post_h]) * r2v
                        + Float32(fn[mix, Int32(3 * self.hidden_size) + post_h]) * r3v
                    )
                post_h += Int32(self.num_threads)

            s_partials[0, tidx] = sqsum
            for mix in cutlass.range_constexpr(_POST_PRE_CHUNK):
                s_partials[mix + 1, tidx] = mix_acc[mix]
            for mix in cutlass.range_constexpr(_POST_PRE_CHUNK):
                mix_acc[mix] = Float32(0.0)

            post_h = tidx
            while post_h < Int32(self.hidden_size):
                r0v = Float32(out[token, 0, post_h])
                r1v = Float32(out[token, 1, post_h])
                r2v = Float32(out[token, 2, post_h])
                r3v = Float32(out[token, 3, post_h])
                for mix in cutlass.range_constexpr(_POST_PRE_CHUNK):
                    src_mix = mix + _POST_PRE_CHUNK
                    mix_acc[mix] += (
                        Float32(fn[src_mix, post_h]) * r0v
                        + Float32(fn[src_mix, Int32(self.hidden_size) + post_h]) * r1v
                        + Float32(fn[src_mix, Int32(2 * self.hidden_size) + post_h]) * r2v
                        + Float32(fn[src_mix, Int32(3 * self.hidden_size) + post_h]) * r3v
                    )
                post_h += Int32(self.num_threads)

            for mix in cutlass.range_constexpr(_POST_PRE_CHUNK):
                s_partials[mix + _POST_PRE_CHUNK + 1, tidx] = mix_acc[mix]
        else:
            mix_acc = cute.make_rmem_tensor(
                cute.make_layout((self.mixes,), stride=(1,)),
                Float32,
            )
            for mix in cutlass.range_constexpr(_MIXES):
                mix_acc[mix] = Float32(0.0)
            sqsum = Float32(0.0)
            linear = tidx
            while linear < Int32(self.total_k):
                hc = linear // Int32(self.hidden_size)
                lin_h = linear - hc * Int32(self.hidden_size)
                rv = Float32(residual[token, hc, lin_h])
                sqsum += rv * rv
                for mix in cutlass.range_constexpr(_MIXES):
                    mix_acc[mix] += Float32(fn[mix, linear]) * rv
                linear += Int32(self.num_threads)

            s_partials[0, tidx] = sqsum
            for mix in cutlass.range_constexpr(_MIXES):
                s_partials[mix + 1, tidx] = mix_acc[mix]
        cute.arch.sync_threads()

        if tidx == Int32(0):
            total_sqsum = Float32(0.0)
            mixes = cute.make_rmem_tensor(
                cute.make_layout((self.mixes,), stride=(1,)),
                Float32,
            )
            for mix in cutlass.range_constexpr(_MIXES):
                mixes[mix] = Float32(0.0)
            src_thread = Int32(0)
            while src_thread < Int32(self.num_threads):
                total_sqsum += Float32(s_partials[0, src_thread])
                for mix in cutlass.range_constexpr(_MIXES):
                    mixes[mix] += Float32(s_partials[mix + 1, src_thread])
                src_thread += Int32(1)

            inv_rms = cute.rsqrt(
                total_sqsum / Float32(self.total_k) + Float32(self.rms_eps)
            )
            for mix in cutlass.range_constexpr(_MIXES):
                mixes[mix] = mixes[mix] * inv_rms

            s0 = Float32(scale[0])
            s1 = Float32(scale[1])
            s2 = Float32(scale[2])
            one = Float32(1.0)
            two = Float32(2.0)
            eps = Float32(self.hc_eps)

            pre0 = one / (one + cute.exp(-(mixes[0] * s0 + Float32(bias[0])))) + eps
            pre1 = one / (one + cute.exp(-(mixes[1] * s0 + Float32(bias[1])))) + eps
            pre2 = one / (one + cute.exp(-(mixes[2] * s0 + Float32(bias[2])))) + eps
            pre3 = one / (one + cute.exp(-(mixes[3] * s0 + Float32(bias[3])))) + eps
            s_pre[0] = pre0
            s_pre[1] = pre1
            s_pre[2] = pre2
            s_pre[3] = pre3

            post0 = two / (one + cute.exp(-(mixes[4] * s1 + Float32(bias[4]))))
            post1 = two / (one + cute.exp(-(mixes[5] * s1 + Float32(bias[5]))))
            post2 = two / (one + cute.exp(-(mixes[6] * s1 + Float32(bias[6]))))
            post3 = two / (one + cute.exp(-(mixes[7] * s1 + Float32(bias[7]))))
            s_post[0] = post0
            s_post[1] = post1
            s_post[2] = post2
            s_post[3] = post3
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

            c00 = cute.exp(c00 - m0)
            c01 = cute.exp(c01 - m0)
            c02 = cute.exp(c02 - m0)
            c03 = cute.exp(c03 - m0)
            c10 = cute.exp(c10 - m1)
            c11 = cute.exp(c11 - m1)
            c12 = cute.exp(c12 - m1)
            c13 = cute.exp(c13 - m1)
            c20 = cute.exp(c20 - m2)
            c21 = cute.exp(c21 - m2)
            c22 = cute.exp(c22 - m2)
            c23 = cute.exp(c23 - m2)
            c30 = cute.exp(c30 - m3)
            c31 = cute.exp(c31 - m3)
            c32 = cute.exp(c32 - m3)
            c33 = cute.exp(c33 - m3)

            r0 = c00 + c01 + c02 + c03
            r1 = c10 + c11 + c12 + c13
            r2 = c20 + c21 + c22 + c23
            r3 = c30 + c31 + c32 + c33
            c00 = c00 / r0 + eps
            c01 = c01 / r0 + eps
            c02 = c02 / r0 + eps
            c03 = c03 / r0 + eps
            c10 = c10 / r1 + eps
            c11 = c11 / r1 + eps
            c12 = c12 / r1 + eps
            c13 = c13 / r1 + eps
            c20 = c20 / r2 + eps
            c21 = c21 / r2 + eps
            c22 = c22 / r2 + eps
            c23 = c23 / r2 + eps
            c30 = c30 / r3 + eps
            c31 = c31 / r3 + eps
            c32 = c32 / r3 + eps
            c33 = c33 / r3 + eps

            col0 = c00 + c10 + c20 + c30 + eps
            col1 = c01 + c11 + c21 + c31 + eps
            col2 = c02 + c12 + c22 + c32 + eps
            col3 = c03 + c13 + c23 + c33 + eps
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

            for _ in cutlass.range_constexpr(self.sinkhorn_iters - 1):
                r0 = c00 + c01 + c02 + c03 + eps
                r1 = c10 + c11 + c12 + c13 + eps
                r2 = c20 + c21 + c22 + c23 + eps
                r3 = c30 + c31 + c32 + c33 + eps
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

                col0 = c00 + c10 + c20 + c30 + eps
                col1 = c01 + c11 + c21 + c31 + eps
                col2 = c02 + c12 + c22 + c32 + eps
                col3 = c03 + c13 + c23 + c33 + eps
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
            s_comb[0, 0] = c00
            s_comb[0, 1] = c01
            s_comb[0, 2] = c02
            s_comb[0, 3] = c03
            s_comb[1, 0] = c10
            s_comb[1, 1] = c11
            s_comb[1, 2] = c12
            s_comb[1, 3] = c13
            s_comb[2, 0] = c20
            s_comb[2, 1] = c21
            s_comb[2, 2] = c22
            s_comb[2, 3] = c23
            s_comb[3, 0] = c30
            s_comb[3, 1] = c31
            s_comb[3, 2] = c32
            s_comb[3, 3] = c33

        cute.arch.sync_threads()

        norm_sumsq = Float32(0.0)
        out_h = tidx
        while out_h < Int32(self.hidden_size):
            if const_expr(self.source_post):
                r0v = Float32(out[token, 0, out_h])
                r1v = Float32(out[token, 1, out_h])
                r2v = Float32(out[token, 2, out_h])
                r3v = Float32(out[token, 3, out_h])
            else:
                r0v = Float32(residual[token, 0, out_h])
                r1v = Float32(residual[token, 1, out_h])
                r2v = Float32(residual[token, 2, out_h])
                r3v = Float32(residual[token, 3, out_h])
            yv = (
                Float32(s_pre[0]) * r0v
                + Float32(s_pre[1]) * r1v
                + Float32(s_pre[2]) * r2v
                + Float32(s_pre[3]) * r3v
            )
            y_bf16 = yv.to(cutlass.BFloat16)
            if const_expr(self.fuse_norm):
                s_y[out_h] = y_bf16
                y_norm_src = Float32(y_bf16)
                norm_sumsq += y_norm_src * y_norm_src
            else:
                y[token, out_h] = y_bf16
            if const_expr(self.emit_out):
                xh = Float32(x[token, out_h])
                o0 = (
                    Float32(s_post[0]) * xh
                    + Float32(s_comb[0, 0]) * r0v
                    + Float32(s_comb[1, 0]) * r1v
                    + Float32(s_comb[2, 0]) * r2v
                    + Float32(s_comb[3, 0]) * r3v
                )
                o1 = (
                    Float32(s_post[1]) * xh
                    + Float32(s_comb[0, 1]) * r0v
                    + Float32(s_comb[1, 1]) * r1v
                    + Float32(s_comb[2, 1]) * r2v
                    + Float32(s_comb[3, 1]) * r3v
                )
                o2 = (
                    Float32(s_post[2]) * xh
                    + Float32(s_comb[0, 2]) * r0v
                    + Float32(s_comb[1, 2]) * r1v
                    + Float32(s_comb[2, 2]) * r2v
                    + Float32(s_comb[3, 2]) * r3v
                )
                o3 = (
                    Float32(s_post[3]) * xh
                    + Float32(s_comb[0, 3]) * r0v
                    + Float32(s_comb[1, 3]) * r1v
                    + Float32(s_comb[2, 3]) * r2v
                    + Float32(s_comb[3, 3]) * r3v
                )
                out[token, 0, out_h] = o0.to(cutlass.BFloat16)
                out[token, 1, out_h] = o1.to(cutlass.BFloat16)
                out[token, 2, out_h] = o2.to(cutlass.BFloat16)
                out[token, 3, out_h] = o3.to(cutlass.BFloat16)
            out_h += Int32(self.num_threads)

        if const_expr(self.fuse_norm):
            norm_sumsq = _warp_allreduce_sum(norm_sumsq)
            lane = tidx % Int32(32)
            warp = tidx // Int32(32)
            if lane == Int32(0):
                s_partials[0, warp] = norm_sumsq
            cute.arch.sync_threads()
            if tidx == Int32(0):
                total_norm_sumsq = Float32(0.0)
                src_warp = Int32(0)
                while src_warp < Int32(self.num_threads // 32):
                    total_norm_sumsq += Float32(s_partials[0, src_warp])
                    src_warp += Int32(1)
                s_post[0] = cute.rsqrt(
                    total_norm_sumsq / Float32(self.hidden_size)
                    + Float32(self.norm_eps)
                )
            cute.arch.sync_threads()

            norm_h = tidx
            while norm_h < Int32(self.hidden_size):
                y[token, norm_h] = (
                    Float32(s_y[norm_h])
                    * Float32(s_post[0])
                    * Float32(norm_weight[norm_h])
                ).to(cutlass.BFloat16)
                norm_h += Int32(self.num_threads)


class MHCPrePostHidden4096Kernel(MHCPreHidden4096Kernel):
    def __init__(
        self,
        *,
        tokens: int,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_eps: float = 0.0,
        fuse_norm: bool = False,
    ):
        super().__init__(
            tokens=tokens,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
        )
        self.emit_out = True


class MHCPostPreHidden4096Kernel(MHCPreHidden4096Kernel):
    def __init__(
        self,
        *,
        tokens: int,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_eps: float = 0.0,
        fuse_norm: bool = False,
    ):
        super().__init__(
            tokens=tokens,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
        )
        self.source_post = True


class MHCPostHidden4096Kernel:
    num_threads = _THREADS
    hidden_size = _HIDDEN

    def __init__(self, *, tokens: int):
        self.tokens = int(tokens)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(x.element_type != cutlass.BFloat16):
            raise TypeError("x must be BFloat16")
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(post.element_type != cutlass.Float32):
            raise TypeError("post must be Float32")
        if const_expr(comb.element_type != cutlass.Float32):
            raise TypeError("comb must be Float32")
        if const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        if const_expr(x.shape != (self.tokens, _HIDDEN)):
            raise ValueError("x must have shape (tokens, 4096)")
        if const_expr(residual.shape != (self.tokens, _MHC_MULT, _HIDDEN)):
            raise ValueError("residual must have shape (tokens, 4, 4096)")
        if const_expr(post.shape != (self.tokens, _MHC_MULT)):
            raise ValueError("post must have shape (tokens, 4)")
        if const_expr(comb.shape != (self.tokens, _MHC_MULT, _MHC_MULT)):
            raise ValueError("comb must have shape (tokens, 4, 4)")
        if const_expr(out.shape != (self.tokens, _MHC_MULT, _HIDDEN)):
            raise ValueError("out must have shape (tokens, 4, 4096)")

        self.kernel(x, residual, post, comb, out).launch(
            grid=(self.tokens, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        out: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        token, _, _ = cute.arch.block_idx()
        p0 = Float32(post[token, 0])
        p1 = Float32(post[token, 1])
        p2 = Float32(post[token, 2])
        p3 = Float32(post[token, 3])
        c00 = Float32(comb[token, 0, 0])
        c01 = Float32(comb[token, 0, 1])
        c02 = Float32(comb[token, 0, 2])
        c03 = Float32(comb[token, 0, 3])
        c10 = Float32(comb[token, 1, 0])
        c11 = Float32(comb[token, 1, 1])
        c12 = Float32(comb[token, 1, 2])
        c13 = Float32(comb[token, 1, 3])
        c20 = Float32(comb[token, 2, 0])
        c21 = Float32(comb[token, 2, 1])
        c22 = Float32(comb[token, 2, 2])
        c23 = Float32(comb[token, 2, 3])
        c30 = Float32(comb[token, 3, 0])
        c31 = Float32(comb[token, 3, 1])
        c32 = Float32(comb[token, 3, 2])
        c33 = Float32(comb[token, 3, 3])

        h = tidx
        while h < Int32(self.hidden_size):
            xh = Float32(x[token, h])
            r0v = Float32(residual[token, 0, h])
            r1v = Float32(residual[token, 1, h])
            r2v = Float32(residual[token, 2, h])
            r3v = Float32(residual[token, 3, h])
            o0 = p0 * xh + c00 * r0v + c10 * r1v + c20 * r2v + c30 * r3v
            o1 = p1 * xh + c01 * r0v + c11 * r1v + c21 * r2v + c31 * r3v
            o2 = p2 * xh + c02 * r0v + c12 * r1v + c22 * r2v + c32 * r3v
            o3 = p3 * xh + c03 * r0v + c13 * r1v + c23 * r2v + c33 * r3v
            out[token, 0, h] = o0.to(cutlass.BFloat16)
            out[token, 1, h] = o1.to(cutlass.BFloat16)
            out[token, 2, h] = o2.to(cutlass.BFloat16)
            out[token, 3, h] = o3.to(cutlass.BFloat16)
            h += Int32(self.num_threads)


class MHCFinalizeYFromPartialsToken1Hidden4096Kernel:
    num_threads = 1024
    hidden_size = _HIDDEN
    block_h = 512
    source_tiles = _SOURCE_TILES
    mixes = _MIXES
    partials = _PARTIALS

    def __init__(
        self,
        *,
        tokens: int = _TOKENS,
        block_h: int = 512,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_eps: float = 0.0,
        fuse_norm: bool = False,
    ):
        self.tokens = int(tokens)
        self.block_h = int(block_h)
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.norm_eps = float(norm_eps)
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
        if const_expr(scale.element_type != cutlass.Float32):
            raise TypeError("scale must be Float32")
        if const_expr(bias.element_type != cutlass.Float32):
            raise TypeError("bias must be Float32")
        if const_expr(y.element_type != cutlass.BFloat16):
            raise TypeError("y must be BFloat16")
        if const_expr(post.element_type != cutlass.Float32):
            raise TypeError("post must be Float32")
        if const_expr(comb.element_type != cutlass.Float32):
            raise TypeError("comb must be Float32")
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
        if const_expr(scale.shape != (3,)):
            raise ValueError("scale must have shape (3,)")
        if const_expr(bias.shape != (_MIXES,)):
            raise ValueError("bias must have shape (24,)")
        if const_expr(y.shape != (self.tokens, _HIDDEN)):
            raise ValueError("y must have shape (tokens, 4096)")
        if const_expr(post.shape != (self.tokens, _MHC_MULT)):
            raise ValueError("post must have shape (tokens, 4)")
        if const_expr(comb.shape != (self.tokens, _MHC_MULT, _MHC_MULT)):
            raise ValueError("comb must have shape (tokens, 4, 4)")
        if const_expr(self.fuse_norm and norm_weight.shape != (_HIDDEN,)):
            raise ValueError("norm_weight must have shape (4096,)")

        self.kernel(residual, partials, scale, bias, y, post, comb, norm_weight).launch(
            grid=(
                1 if self.fuse_norm else self.hidden_size // self.block_h,
                self.tokens,
                1,
            ),
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
        tidx = Int32(cute.arch.thread_idx()[0])
        tile_h, token, _ = cute.arch.block_idx()
        tile_h = Int32(tile_h)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_finalize_storage_cls(self.num_threads, self.fuse_norm))
        partial_layout = cute.make_layout(
            (1, self.num_threads),
            stride=(self.num_threads, 1),
        )
        s_partials = storage.partials.get_tensor(partial_layout)
        s_pre = storage.pre.get_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)))
        s_post = storage.post.get_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)))
        s_comb = storage.comb.get_tensor(
            cute.make_layout((_MHC_MULT, _MHC_MULT), stride=(_MHC_MULT, 1))
        )
        if const_expr(self.fuse_norm):
            s_y = storage.y.get_tensor(cute.make_layout((_HIDDEN,), stride=(1,)))
        sums = cute.make_rmem_tensor(
            cute.make_layout((self.partials,), stride=(1,)),
            Float32,
        )
        if tidx < Int32(32):
            for column in cutlass.range_constexpr(_PARTIALS):
                value = Float32(0.0)
                if tidx < Int32(self.source_tiles):
                    value = Float32(partials[token, tidx, column])
                sums[column] = _warp_allreduce_sum(value)

        if tidx == Int32(0):
            total_sqsum = Float32(sums[0])
            mixes = cute.make_rmem_tensor(
                cute.make_layout((self.mixes,), stride=(1,)),
                Float32,
            )
            for mix in cutlass.range_constexpr(_MIXES):
                mixes[mix] = Float32(sums[mix + 1])

            inv_rms = cute.math.rsqrt(
                total_sqsum / Float32(_TOTAL_K) + Float32(self.rms_eps),
                fastmath=True,
            )
            for mix in cutlass.range_constexpr(_MIXES):
                mixes[mix] = mixes[mix] * inv_rms

            s0 = Float32(scale[0])
            s1 = Float32(scale[1])
            s2 = Float32(scale[2])
            one = Float32(1.0)
            two = Float32(2.0)
            eps = Float32(self.hc_eps)

            pre0 = (
                one
                / (
                    one
                    + cute.math.exp(-(mixes[0] * s0 + Float32(bias[0])), fastmath=True)
                )
                + eps
            )
            pre1 = (
                one
                / (
                    one
                    + cute.math.exp(-(mixes[1] * s0 + Float32(bias[1])), fastmath=True)
                )
                + eps
            )
            pre2 = (
                one
                / (
                    one
                    + cute.math.exp(-(mixes[2] * s0 + Float32(bias[2])), fastmath=True)
                )
                + eps
            )
            pre3 = (
                one
                / (
                    one
                    + cute.math.exp(-(mixes[3] * s0 + Float32(bias[3])), fastmath=True)
                )
                + eps
            )
            s_pre[0] = pre0
            s_pre[1] = pre1
            s_pre[2] = pre2
            s_pre[3] = pre3

            post0 = two / (
                one
                + cute.math.exp(-(mixes[4] * s1 + Float32(bias[4])), fastmath=True)
            )
            post1 = two / (
                one
                + cute.math.exp(-(mixes[5] * s1 + Float32(bias[5])), fastmath=True)
            )
            post2 = two / (
                one
                + cute.math.exp(-(mixes[6] * s1 + Float32(bias[6])), fastmath=True)
            )
            post3 = two / (
                one
                + cute.math.exp(-(mixes[7] * s1 + Float32(bias[7])), fastmath=True)
            )
            s_post[0] = post0
            s_post[1] = post1
            s_post[2] = post2
            s_post[3] = post3
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

            r0 = c00 + c01 + c02 + c03
            r1 = c10 + c11 + c12 + c13
            r2 = c20 + c21 + c22 + c23
            r3 = c30 + c31 + c32 + c33
            inv_r0 = cute.arch.rcp_approx(r0)
            inv_r1 = cute.arch.rcp_approx(r1)
            inv_r2 = cute.arch.rcp_approx(r2)
            inv_r3 = cute.arch.rcp_approx(r3)
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
                r0 = c00 + c01 + c02 + c03 + eps
                r1 = c10 + c11 + c12 + c13 + eps
                r2 = c20 + c21 + c22 + c23 + eps
                r3 = c30 + c31 + c32 + c33 + eps
                inv_r0 = cute.arch.rcp_approx(r0)
                inv_r1 = cute.arch.rcp_approx(r1)
                inv_r2 = cute.arch.rcp_approx(r2)
                inv_r3 = cute.arch.rcp_approx(r3)
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
            s_comb[0, 0] = c00
            s_comb[0, 1] = c01
            s_comb[0, 2] = c02
            s_comb[0, 3] = c03
            s_comb[1, 0] = c10
            s_comb[1, 1] = c11
            s_comb[1, 2] = c12
            s_comb[1, 3] = c13
            s_comb[2, 0] = c20
            s_comb[2, 1] = c21
            s_comb[2, 2] = c22
            s_comb[2, 3] = c23
            s_comb[3, 0] = c30
            s_comb[3, 1] = c31
            s_comb[3, 2] = c32
            s_comb[3, 3] = c33

        cute.arch.sync_threads()

        norm_sumsq = Float32(0.0)
        if const_expr(self.fuse_norm):
            for step in cutlass.range_constexpr(self.hidden_size // self.num_threads):
                h = tidx + Int32(step * self.num_threads)
                fused_r0v = Float32(residual[token, 0, h])
                fused_r1v = Float32(residual[token, 1, h])
                fused_r2v = Float32(residual[token, 2, h])
                fused_r3v = Float32(residual[token, 3, h])
                fused_yv = (
                    Float32(s_pre[0]) * fused_r0v
                    + Float32(s_pre[1]) * fused_r1v
                    + Float32(s_pre[2]) * fused_r2v
                    + Float32(s_pre[3]) * fused_r3v
                )
                fused_y_bf16 = fused_yv.to(cutlass.BFloat16)
                s_y[h] = fused_y_bf16
                y_norm_src = Float32(fused_y_bf16)
                norm_sumsq += y_norm_src * y_norm_src
        else:
            h = tile_h * Int32(self.block_h) + tidx
            tile_end = (tile_h + Int32(1)) * Int32(self.block_h)
            while h < tile_end:
                r0v = Float32(residual[token, 0, h])
                r1v = Float32(residual[token, 1, h])
                r2v = Float32(residual[token, 2, h])
                r3v = Float32(residual[token, 3, h])
                yv = (
                    Float32(s_pre[0]) * r0v
                    + Float32(s_pre[1]) * r1v
                    + Float32(s_pre[2]) * r2v
                    + Float32(s_pre[3]) * r3v
                )
                y_bf16 = yv.to(cutlass.BFloat16)
                y[token, h] = y_bf16
                h += Int32(self.num_threads)

        if const_expr(self.fuse_norm):
            norm_sumsq = _warp_allreduce_sum(norm_sumsq)
            lane = tidx % Int32(32)
            warp = tidx // Int32(32)
            if lane == Int32(0):
                s_partials[0, warp] = norm_sumsq
            cute.arch.sync_threads()
            if tidx == Int32(0):
                total_norm_sumsq = Float32(0.0)
                src_warp = Int32(0)
                while src_warp < Int32(self.num_threads // 32):
                    total_norm_sumsq += Float32(s_partials[0, src_warp])
                    src_warp += Int32(1)
                s_post[0] = cute.math.rsqrt(
                    total_norm_sumsq / Float32(self.hidden_size)
                    + Float32(self.norm_eps),
                    fastmath=True,
                )
            cute.arch.sync_threads()

            for step in cutlass.range_constexpr(self.hidden_size // self.num_threads):
                norm_h = tidx + Int32(step * self.num_threads)
                y[token, norm_h] = (
                    Float32(s_y[norm_h])
                    * Float32(s_post[0])
                    * Float32(norm_weight[norm_h])
                ).to(cutlass.BFloat16)


@lru_cache(maxsize=64)
def _pre_kernel(
    tokens: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
) -> MHCPreHidden4096Kernel:
    return MHCPreHidden4096Kernel(
        tokens=tokens,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        fuse_norm=fuse_norm,
    )


@lru_cache(maxsize=1)
def _partial_kernel() -> MHCPartialToken1Hidden4096Kernel:
    return MHCPartialToken1Hidden4096Kernel()


@lru_cache(maxsize=64)
def _post_pre_partial_kernel(tokens: int) -> MHCPostPrePartialToken1Hidden4096Kernel:
    return MHCPostPrePartialToken1Hidden4096Kernel(tokens=tokens)


@lru_cache(maxsize=64)
def _finalize_y_from_partials_kernel(
    tokens: int,
    block_h: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
) -> MHCFinalizeYFromPartialsToken1Hidden4096Kernel:
    return MHCFinalizeYFromPartialsToken1Hidden4096Kernel(
        tokens=tokens,
        block_h=block_h,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        fuse_norm=fuse_norm,
    )


@lru_cache(maxsize=64)
def _post_kernel(tokens: int) -> MHCPostHidden4096Kernel:
    return MHCPostHidden4096Kernel(tokens=tokens)


@lru_cache(maxsize=64)
def _pre_post_kernel(
    tokens: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
) -> MHCPrePostHidden4096Kernel:
    return MHCPrePostHidden4096Kernel(
        tokens=tokens,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        fuse_norm=fuse_norm,
    )


@lru_cache(maxsize=64)
def _post_pre_kernel(
    tokens: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
) -> MHCPostPreHidden4096Kernel:
    return MHCPostPreHidden4096Kernel(
        tokens=tokens,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        fuse_norm=fuse_norm,
    )


def run_mhc_partial_token1_hidden4096(
    *,
    residual: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    args = (
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(partials, cutlass.Float32, assumed_align=4),
        current_cuda_stream(),
    )
    cache_key = (
        _tensor_meta_key(residual),
        _tensor_meta_key(fn),
        _tensor_meta_key(partials),
    )
    b12x_launch(
        _partial_kernel(),
        compile_spec=KernelCompileSpec.from_key(
            "integration.residual.mhc_partial_token1_hidden4096_hctile128_all4",
            1,
            cache_key,
        ),
        compile_args=args,
        runtime_args=args,
    )


def run_mhc_post_pre_partial_hidden4096(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
) -> None:
    tokens = int(x.shape[0])
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
        _post_pre_partial_kernel(tokens),
        compile_spec=KernelCompileSpec.from_key(
            "integration.residual.mhc_post_pre_partial_hidden4096_hctile128_all4",
            2,
            (
                ("tokens", tokens),
                ("partials_per_cta", _POST_PRE_PARTIALS_PER_CTA),
                cache_key,
            ),
        ),
        compile_args=args,
        runtime_args=args,
    )


def run_mhc_post_pre_partial_token1_hidden4096(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
) -> None:
    run_mhc_post_pre_partial_hidden4096(
        x=x,
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        partials=partials,
        out=out,
    )


def run_mhc_finalize_y_from_partials_hidden4096(
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
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> None:
    rms_eps = float(rms_eps)
    hc_eps = float(hc_eps)
    sinkhorn_iters = int(sinkhorn_iters)
    norm_eps = float(norm_eps)
    fuse_norm = norm_weight is not None
    norm_weight_tensor = _norm_weight_kernel_tensor(norm_weight, y)
    tokens = int(residual.shape[0])
    block_h = 256 if tokens >= 8 else 512
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
        ("source_tile_h", _SOURCE_TILE_H),
        ("source_tiles", _SOURCE_TILES),
        ("block_h", block_h),
        ("source_finalize_loop", "warp32_per_hidden_tile_y_only"),
        ("threads", MHCFinalizeYFromPartialsToken1Hidden4096Kernel.num_threads),
        ("math", "fast_exp_exact_sigmoid_rcp_approx_sinkhorn"),
        ("norm_impl", "single_cta_shared_y_warp_reduce_v4" if fuse_norm else "none"),
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
        _finalize_y_from_partials_kernel(
            tokens,
            block_h,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            norm_eps,
            fuse_norm,
        ),
        compile_spec=KernelCompileSpec.from_key(
            "integration.residual.mhc_finalize_y_from_partials_hidden4096",
            4,
            cache_key,
        ),
        compile_args=args,
        runtime_args=args,
    )


def run_mhc_finalize_y_from_partials_token1_hidden4096(
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
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> None:
    run_mhc_finalize_y_from_partials_hidden4096(
        residual=residual,
        partials=partials,
        scale=scale,
        bias=bias,
        y=y,
        post=post,
        comb=comb,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def run_mhc_pre_hidden4096(
    *,
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> None:
    tokens = int(residual.shape[0])
    rms_eps = float(rms_eps)
    hc_eps = float(hc_eps)
    sinkhorn_iters = int(sinkhorn_iters)
    norm_eps = float(norm_eps)
    fuse_norm = norm_weight is not None
    norm_weight_tensor = _norm_weight_kernel_tensor(norm_weight, y)
    if sinkhorn_iters <= 0:
        raise ValueError(f"sinkhorn_iters must be positive, got {sinkhorn_iters}")
    args = (
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(y, cutlass.BFloat16),
        _to_kernel_tensor(post, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(comb, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(scale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(bias, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(y, cutlass.BFloat16),
        _to_kernel_tensor(post, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(comb, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(residual, cutlass.BFloat16),
        norm_weight_tensor,
        current_cuda_stream(),
    )
    cache_key = (
        tokens,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        ("fuse_norm", fuse_norm),
        ("norm_eps", norm_eps if fuse_norm else 0.0),
        ("norm_impl", "bf16_sumsq_shared_y_warp_reduce_v4" if fuse_norm else "none"),
        _tensor_meta_key(residual),
        _tensor_meta_key(fn),
        _tensor_meta_key(scale),
        _tensor_meta_key(bias),
        _tensor_meta_key(y),
        _tensor_meta_key(post),
        _tensor_meta_key(comb),
        _tensor_meta_key(norm_weight) if norm_weight is not None else None,
    )
    b12x_launch(
        _pre_kernel(tokens, rms_eps, hc_eps, sinkhorn_iters, norm_eps, fuse_norm),
        compile_spec=KernelCompileSpec.from_key(
            "integration.residual.mhc_pre_hidden4096",
            5,
            cache_key,
        ),
        compile_args=args,
        runtime_args=args,
    )


def run_mhc_pre_post_hidden4096(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    out: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> None:
    tokens = int(residual.shape[0])
    rms_eps = float(rms_eps)
    hc_eps = float(hc_eps)
    sinkhorn_iters = int(sinkhorn_iters)
    norm_eps = float(norm_eps)
    fuse_norm = norm_weight is not None
    norm_weight_tensor = _norm_weight_kernel_tensor(norm_weight, y)
    if sinkhorn_iters <= 0:
        raise ValueError(f"sinkhorn_iters must be positive, got {sinkhorn_iters}")
    args = (
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(x, cutlass.BFloat16),
        _to_kernel_tensor(post, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(comb, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(scale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(bias, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(y, cutlass.BFloat16),
        _to_kernel_tensor(post, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(comb, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(out, cutlass.BFloat16),
        norm_weight_tensor,
        current_cuda_stream(),
    )
    cache_key = (
        tokens,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        ("fuse_norm", fuse_norm),
        ("norm_eps", norm_eps if fuse_norm else 0.0),
        ("norm_impl", "bf16_sumsq_shared_y_warp_reduce_v4" if fuse_norm else "none"),
        _tensor_meta_key(residual),
        _tensor_meta_key(x),
        _tensor_meta_key(fn),
        _tensor_meta_key(scale),
        _tensor_meta_key(bias),
        _tensor_meta_key(y),
        _tensor_meta_key(post),
        _tensor_meta_key(comb),
        _tensor_meta_key(out),
        _tensor_meta_key(norm_weight) if norm_weight is not None else None,
    )
    b12x_launch(
        _pre_post_kernel(tokens, rms_eps, hc_eps, sinkhorn_iters, norm_eps, fuse_norm),
        compile_spec=KernelCompileSpec.from_key(
            "integration.residual.mhc_pre_post_hidden4096",
            4,
            cache_key,
        ),
        compile_args=args,
        runtime_args=args,
    )


def run_mhc_post_pre_hidden4096(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    residual_out: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> None:
    tokens = int(residual.shape[0])
    rms_eps = float(rms_eps)
    hc_eps = float(hc_eps)
    sinkhorn_iters = int(sinkhorn_iters)
    norm_eps = float(norm_eps)
    fuse_norm = norm_weight is not None
    norm_weight_tensor = _norm_weight_kernel_tensor(norm_weight, y)
    if sinkhorn_iters <= 0:
        raise ValueError(f"sinkhorn_iters must be positive, got {sinkhorn_iters}")
    args = (
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(x, cutlass.BFloat16),
        _to_kernel_tensor(prev_post, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(prev_comb, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(scale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(bias, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(y, cutlass.BFloat16),
        _to_kernel_tensor(post, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(comb, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(residual_out, cutlass.BFloat16),
        norm_weight_tensor,
        current_cuda_stream(),
    )
    cache_key = (
        tokens,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        ("fuse_norm", fuse_norm),
        ("norm_eps", norm_eps if fuse_norm else 0.0),
        ("norm_impl", "bf16_sumsq_shared_y_warp_reduce_v4" if fuse_norm else "none"),
        _tensor_meta_key(residual),
        _tensor_meta_key(x),
        _tensor_meta_key(prev_post),
        _tensor_meta_key(prev_comb),
        _tensor_meta_key(fn),
        _tensor_meta_key(scale),
        _tensor_meta_key(bias),
        _tensor_meta_key(y),
        _tensor_meta_key(post),
        _tensor_meta_key(comb),
        _tensor_meta_key(residual_out),
        _tensor_meta_key(norm_weight) if norm_weight is not None else None,
    )
    b12x_launch(
        _post_pre_kernel(tokens, rms_eps, hc_eps, sinkhorn_iters, norm_eps, fuse_norm),
        compile_spec=KernelCompileSpec.from_key(
            "integration.residual.mhc_post_pre_hidden4096",
            3,
            cache_key,
        ),
        compile_args=args,
        runtime_args=args,
    )


def run_mhc_post_hidden4096(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    out: torch.Tensor,
) -> None:
    tokens = int(residual.shape[0])
    args = (
        _to_kernel_tensor(x, cutlass.BFloat16),
        _to_kernel_tensor(residual, cutlass.BFloat16),
        _to_kernel_tensor(post, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(comb, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(out, cutlass.BFloat16),
        current_cuda_stream(),
    )
    cache_key = (
        tokens,
        _tensor_meta_key(x),
        _tensor_meta_key(residual),
        _tensor_meta_key(post),
        _tensor_meta_key(comb),
        _tensor_meta_key(out),
    )
    b12x_launch(
        _post_kernel(tokens),
        compile_spec=KernelCompileSpec.from_key(
            "integration.residual.mhc_post_hidden4096",
            2,
            cache_key,
        ),
        compile_args=args,
        runtime_args=args,
    )


__all__ = [
    "run_mhc_finalize_y_from_partials_hidden4096",
    "run_mhc_finalize_y_from_partials_token1_hidden4096",
    "run_mhc_partial_token1_hidden4096",
    "run_mhc_post_pre_partial_hidden4096",
    "run_mhc_post_pre_partial_token1_hidden4096",
    "run_mhc_pre_hidden4096",
    "run_mhc_pre_post_hidden4096",
    "run_mhc_post_pre_hidden4096",
    "run_mhc_post_hidden4096",
]
