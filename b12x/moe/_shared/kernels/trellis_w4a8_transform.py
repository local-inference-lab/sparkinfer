"""Exact EXL outer transforms for the route-major QSRT W4A8 path."""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Float32, Int32, Uint8, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    FLOAT8_E4M3_MAX,
    cvt_f32x4_to_e4m3x4,
    fabs_f32,
    fmax_f32,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe._shared.kernels.activations import (
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)


_THREADS = 256


class _Hadamard128:
    @cute.jit
    def _had128_quad(
        self,
        v0: Float32,
        v1: Float32,
        v2: Float32,
        v3: Float32,
        lane: Int32,
    ):
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        h0 = s0 + s1
        h1 = d0 + d1
        h2 = s0 - s1
        h3 = d0 - d1
        for i in cutlass.range_constexpr(5):
            st = 1 << i
            p0 = cute.arch.shuffle_sync_bfly(h0, offset=st)
            p1 = cute.arch.shuffle_sync_bfly(h1, offset=st)
            p2 = cute.arch.shuffle_sync_bfly(h2, offset=st)
            p3 = cute.arch.shuffle_sync_bfly(h3, offset=st)
            if (lane & Int32(st)) != Int32(0):
                h0 = p0 - h0
                h1 = p1 - h1
                h2 = p2 - h2
                h3 = p3 - h3
            else:
                h0 = p0 + h0
                h1 = p1 + h1
                h2 = p2 + h2
                h3 = p3 + h3
        scale = Float32(0.088388347648)
        return h0 * scale, h1 * scale, h2 * scale, h3 * scale


class _NativeMXFP8:
    """Quantize one H128 warp directly into the native trellis MMA order."""

    @cute.jit
    def _pack_native_mxfp8(
        self,
        v0: Float32,
        v1: Float32,
        v2: Float32,
        v3: Float32,
        lane: Int32,
    ):
        # Match the standalone quantizer exactly: its input has already been
        # rounded through FP16 by the transform kernel, and one 8-lane
        # subgroup owns each K32 scale group.
        v0 = cutlass.Float16(v0).to(Float32)
        v1 = cutlass.Float16(v1).to(Float32)
        v2 = cutlass.Float16(v2).to(Float32)
        v3 = cutlass.Float16(v3).to(Float32)
        max_abs = fmax_f32(fabs_f32(v0), fabs_f32(v1))
        max_abs = fmax_f32(max_abs, fabs_f32(v2))
        max_abs = fmax_f32(max_abs, fabs_f32(v3))
        for shift in cutlass.range_constexpr(3):
            max_abs = fmax_f32(
                max_abs,
                cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
            )

        _, scale_byte = pow2_ceil_ue8m0(
            max_abs * Float32(1.0 / FLOAT8_E4M3_MAX)
        )
        if max_abs == Float32(0.0):
            scale_byte = Uint32(127)
        inv_scale = ue8m0_to_output_scale(scale_byte)

        # The native E4M3 A fragment is not linear K order.  Reproduce the
        # fixed permutation used by quantize_mxfp8_rows_cute without ever
        # materializing the transformed FP16 row in global memory.
        subgroup_base = (lane >> Int32(3)) << Int32(3)
        q = lane & Int32(7)
        source0 = Int32(0)
        source1 = Int32(2)
        use_high = Int32(0)
        if q < Int32(4):
            source0 = q & Int32(1)
            source1 = source0 + Int32(2)
            if q >= Int32(2):
                use_high = Int32(1)
        else:
            q4 = q - Int32(4)
            source0 = Int32(5) - (q4 & Int32(1))
            source1 = source0 + Int32(2)
            if q4 >= Int32(2):
                use_high = Int32(1)
        source0 += subgroup_base
        source1 += subgroup_base

        lo0 = cute.arch.shuffle_sync(v0, source0)
        lo1 = cute.arch.shuffle_sync(v1, source0)
        lo2 = cute.arch.shuffle_sync(v0, source1)
        lo3 = cute.arch.shuffle_sync(v1, source1)
        hi0 = cute.arch.shuffle_sync(v2, source0)
        hi1 = cute.arch.shuffle_sync(v3, source0)
        hi2 = cute.arch.shuffle_sync(v2, source1)
        hi3 = cute.arch.shuffle_sync(v3, source1)
        x0 = lo0
        x1 = lo1
        x2 = lo2
        x3 = lo3
        if use_high != Int32(0):
            x0 = hi0
            x1 = hi1
            x2 = hi2
            x3 = hi3
        return (
            cvt_f32x4_to_e4m3x4(
                x0 * inv_scale,
                x1 * inv_scale,
                x2 * inv_scale,
                x3 * inv_scale,
            ),
            scale_byte,
        )

    @cute.jit
    def _store_native_scale(
        self,
        scale_rows: cute.Tensor,
        scale_mma: cute.Tensor,
        row: Int32,
        group: Int32,
        scale_byte: Uint32,
        groups_k: cutlass.Constexpr[int],
    ) -> None:
        scale_u8 = Uint8(scale_byte)
        scale_rows[row, group] = scale_u8
        row32 = row % Int32(32)
        row4 = (row // Int32(32)) % Int32(4)
        tile_m = row // Int32(128)
        k4 = group % Int32(4)
        tile_k = group // Int32(4)
        scale_mma_offset = (
            row32 * Int32(16)
            + row4 * Int32(4)
            + tile_m * Int32(((int(groups_k) + 3) // 4) * 512)
            + k4
            + tile_k * Int32(512)
        )
        scale_mma[scale_mma_offset] = scale_u8


class TrellisW4A8InputRotationKernel(_Hadamard128):
    """Apply projection-specific SUH and H128 to each token/top-k route."""

    threads = _THREADS

    def __init__(
        self,
        *,
        hidden_size: int,
        topk: int,
        source_type: type[cutlass.Numeric],
        broadcast_suh: bool,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.topk = int(topk)
        self.source_type = source_type
        self.broadcast_suh = bool(broadcast_suh)

    @cute.jit
    def __call__(
        self,
        source_ptr: cute.Pointer,
        route_experts_ptr: cute.Pointer,
        gate_suh_ptr: cute.Pointer,
        up_suh_ptr: cute.Pointer,
        gate_out_ptr: cute.Pointer,
        up_out_ptr: cute.Pointer,
        tokens: Int32,
        num_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        routes = tokens * Int32(self.topk)
        source = cute.make_tensor(
            source_ptr,
            cute.make_ordered_layout((tokens, self.hidden_size), order=(1, 0)),
        )
        route_experts = cute.make_tensor(
            route_experts_ptr, cute.make_layout((routes,))
        )
        suh_rows = Int32(1) if self.broadcast_suh else num_experts
        gate_suh = cute.make_tensor(
            gate_suh_ptr,
            cute.make_ordered_layout((suh_rows, self.hidden_size), order=(1, 0)),
        )
        up_suh = cute.make_tensor(
            up_suh_ptr,
            cute.make_ordered_layout((suh_rows, self.hidden_size), order=(1, 0)),
        )
        gate_out = cute.make_tensor(
            gate_out_ptr,
            cute.make_ordered_layout((routes, self.hidden_size), order=(1, 0)),
        )
        up_out = cute.make_tensor(
            up_out_ptr,
            cute.make_ordered_layout((routes, self.hidden_size), order=(1, 0)),
        )
        warps = self.threads // 32
        units = routes * Int32(self.hidden_size // 128)
        grid_x = cute.ceil_div(units, warps)
        self.kernel(
            source,
            route_experts,
            gate_suh,
            up_suh,
            gate_out,
            up_out,
            tokens,
            num_experts,
        ).launch(
            grid=(grid_x, 1, 1), block=[self.threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        route_experts: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        gate_out: cute.Tensor,
        up_out: cute.Tensor,
        tokens: Int32,
        num_experts: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        gwarp = Int32(bidx) * Int32(self.threads // 32) + warp
        routes = tokens * Int32(self.topk)
        nblk = Int32(self.hidden_size // 128)
        total = routes * nblk
        if gwarp < total:
            route = gwarp // nblk
            blk = gwarp - route * nblk
            token = route // Int32(self.topk)
            expert = route_experts[route].to(Int32)
            valid = Int32(0)
            if expert >= Int32(0) and expert < num_experts:
                valid = Int32(1)
            scale_row = Int32(0)
            if cutlass.const_expr(not self.broadcast_suh):
                if valid != Int32(0):
                    scale_row = expert
            col = blk * Int32(128) + lane * Int32(4)

            x0 = cutlass.Float16(source[token, col]).to(Float32)
            x1 = cutlass.Float16(source[token, col + Int32(1)]).to(Float32)
            x2 = cutlass.Float16(source[token, col + Int32(2)]).to(Float32)
            x3 = cutlass.Float16(source[token, col + Int32(3)]).to(Float32)
            sg0 = gate_suh[scale_row, col].to(Float32)
            sg1 = gate_suh[scale_row, col + Int32(1)].to(Float32)
            sg2 = gate_suh[scale_row, col + Int32(2)].to(Float32)
            sg3 = gate_suh[scale_row, col + Int32(3)].to(Float32)
            su0 = up_suh[scale_row, col].to(Float32)
            su1 = up_suh[scale_row, col + Int32(1)].to(Float32)
            su2 = up_suh[scale_row, col + Int32(2)].to(Float32)
            su3 = up_suh[scale_row, col + Int32(3)].to(Float32)
            g0 = cutlass.Float16(x0 * sg0).to(Float32)
            g1 = cutlass.Float16(x1 * sg1).to(Float32)
            g2 = cutlass.Float16(x2 * sg2).to(Float32)
            g3 = cutlass.Float16(x3 * sg3).to(Float32)
            u0 = cutlass.Float16(x0 * su0).to(Float32)
            u1 = cutlass.Float16(x1 * su1).to(Float32)
            u2 = cutlass.Float16(x2 * su2).to(Float32)
            u3 = cutlass.Float16(x3 * su3).to(Float32)
            if valid == Int32(0):
                g0 = Float32(0.0)
                g1 = Float32(0.0)
                g2 = Float32(0.0)
                g3 = Float32(0.0)
                u0 = Float32(0.0)
                u1 = Float32(0.0)
                u2 = Float32(0.0)
                u3 = Float32(0.0)
            gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
            uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
            gate_out[route, col] = cutlass.Float16(gh0)
            gate_out[route, col + Int32(1)] = cutlass.Float16(gh1)
            gate_out[route, col + Int32(2)] = cutlass.Float16(gh2)
            gate_out[route, col + Int32(3)] = cutlass.Float16(gh3)
            up_out[route, col] = cutlass.Float16(uh0)
            up_out[route, col + Int32(1)] = cutlass.Float16(uh1)
            up_out[route, col + Int32(2)] = cutlass.Float16(uh2)
            up_out[route, col + Int32(3)] = cutlass.Float16(uh3)


class TrellisW4A8InputRotationQuantKernel(_Hadamard128, _NativeMXFP8):
    """Fuse projection-specific input rotations with both MXFP8 quantizers."""

    threads = _THREADS

    def __init__(
        self,
        *,
        hidden_size: int,
        topk: int,
        source_type: type[cutlass.Numeric],
        broadcast_suh: bool,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.groups_k = self.hidden_size // 32
        self.topk = int(topk)
        self.source_type = source_type
        self.broadcast_suh = bool(broadcast_suh)

    @cute.jit
    def __call__(
        self,
        source_ptr: cute.Pointer,
        route_experts_ptr: cute.Pointer,
        gate_suh_ptr: cute.Pointer,
        up_suh_ptr: cute.Pointer,
        gate_values_ptr: cute.Pointer,
        gate_scale_rows_ptr: cute.Pointer,
        gate_scale_mma_ptr: cute.Pointer,
        up_values_ptr: cute.Pointer,
        up_scale_rows_ptr: cute.Pointer,
        up_scale_mma_ptr: cute.Pointer,
        tokens: Int32,
        num_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        routes = tokens * Int32(self.topk)
        output_rows = tokens if self.broadcast_suh else routes
        source = cute.make_tensor(
            source_ptr,
            cute.make_ordered_layout((tokens, self.hidden_size), order=(1, 0)),
        )
        route_experts = cute.make_tensor(
            route_experts_ptr, cute.make_layout((routes,))
        )
        suh_rows = Int32(1) if self.broadcast_suh else num_experts
        gate_suh = cute.make_tensor(
            gate_suh_ptr,
            cute.make_ordered_layout((suh_rows, self.hidden_size), order=(1, 0)),
        )
        up_suh = cute.make_tensor(
            up_suh_ptr,
            cute.make_ordered_layout((suh_rows, self.hidden_size), order=(1, 0)),
        )
        gate_values = cute.make_tensor(
            gate_values_ptr,
            cute.make_ordered_layout(
                (output_rows, self.hidden_size // 4), order=(1, 0)
            ),
        )
        gate_scale_rows = cute.make_tensor(
            gate_scale_rows_ptr,
            cute.make_ordered_layout((output_rows, self.groups_k), order=(1, 0)),
        )
        gate_scale_mma = cute.make_tensor(
            gate_scale_mma_ptr,
            cute.make_layout((max(512, ((self.groups_k + 3) // 4) * 512),)),
        )
        up_values = cute.make_tensor(
            up_values_ptr,
            cute.make_ordered_layout(
                (output_rows, self.hidden_size // 4), order=(1, 0)
            ),
        )
        up_scale_rows = cute.make_tensor(
            up_scale_rows_ptr,
            cute.make_ordered_layout((output_rows, self.groups_k), order=(1, 0)),
        )
        up_scale_mma = cute.make_tensor(
            up_scale_mma_ptr,
            cute.make_layout((max(512, ((self.groups_k + 3) // 4) * 512),)),
        )
        warps = self.threads // 32
        units = output_rows * Int32(self.hidden_size // 128)
        self.kernel(
            source,
            route_experts,
            gate_suh,
            up_suh,
            gate_values,
            gate_scale_rows,
            gate_scale_mma,
            up_values,
            up_scale_rows,
            up_scale_mma,
            tokens,
            num_experts,
        ).launch(
            grid=(cute.ceil_div(units, warps), 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        route_experts: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        gate_values: cute.Tensor,
        gate_scale_rows: cute.Tensor,
        gate_scale_mma: cute.Tensor,
        up_values: cute.Tensor,
        up_scale_rows: cute.Tensor,
        up_scale_mma: cute.Tensor,
        tokens: Int32,
        num_experts: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        gwarp = Int32(bidx) * Int32(self.threads // 32) + warp
        routes = tokens * Int32(self.topk)
        nblk = Int32(self.hidden_size // 128)
        output_rows = tokens if cutlass.const_expr(self.broadcast_suh) else routes
        total = output_rows * nblk
        if gwarp < total:
            output_row = gwarp // nblk
            blk = gwarp - output_row * nblk
            route = output_row
            token = output_row
            expert = Int32(0)
            valid = Int32(1)
            if cutlass.const_expr(not self.broadcast_suh):
                token = route // Int32(self.topk)
                expert = route_experts[route].to(Int32)
                valid = Int32(0)
                if expert >= Int32(0) and expert < num_experts:
                    valid = Int32(1)
            scale_row = Int32(0)
            if cutlass.const_expr(not self.broadcast_suh):
                if valid != Int32(0):
                    scale_row = expert
            col = blk * Int32(128) + lane * Int32(4)

            x0 = cutlass.Float16(source[token, col]).to(Float32)
            x1 = cutlass.Float16(source[token, col + Int32(1)]).to(Float32)
            x2 = cutlass.Float16(source[token, col + Int32(2)]).to(Float32)
            x3 = cutlass.Float16(source[token, col + Int32(3)]).to(Float32)
            g0 = cutlass.Float16(x0 * gate_suh[scale_row, col].to(Float32)).to(
                Float32
            )
            g1 = cutlass.Float16(
                x1 * gate_suh[scale_row, col + Int32(1)].to(Float32)
            ).to(Float32)
            g2 = cutlass.Float16(
                x2 * gate_suh[scale_row, col + Int32(2)].to(Float32)
            ).to(Float32)
            g3 = cutlass.Float16(
                x3 * gate_suh[scale_row, col + Int32(3)].to(Float32)
            ).to(Float32)
            u0 = cutlass.Float16(x0 * up_suh[scale_row, col].to(Float32)).to(
                Float32
            )
            u1 = cutlass.Float16(
                x1 * up_suh[scale_row, col + Int32(1)].to(Float32)
            ).to(Float32)
            u2 = cutlass.Float16(
                x2 * up_suh[scale_row, col + Int32(2)].to(Float32)
            ).to(Float32)
            u3 = cutlass.Float16(
                x3 * up_suh[scale_row, col + Int32(3)].to(Float32)
            ).to(Float32)
            if valid == Int32(0):
                g0 = Float32(0.0)
                g1 = Float32(0.0)
                g2 = Float32(0.0)
                g3 = Float32(0.0)
                u0 = Float32(0.0)
                u1 = Float32(0.0)
                u2 = Float32(0.0)
                u3 = Float32(0.0)
            gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
            uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
            gate_payload, gate_scale = self._pack_native_mxfp8(
                gh0, gh1, gh2, gh3, lane
            )
            up_payload, up_scale = self._pack_native_mxfp8(
                uh0, uh1, uh2, uh3, lane
            )
            subgroup = lane >> Int32(3)
            lane8 = lane & Int32(7)
            group = blk * Int32(4) + subgroup
            word = group * Int32(8) + lane8
            gate_values[output_row, word] = gate_payload
            up_values[output_row, word] = up_payload
            if lane8 == Int32(0):
                self._store_native_scale(
                    gate_scale_rows,
                    gate_scale_mma,
                    output_row,
                    group,
                    gate_scale,
                    self.groups_k,
                )
                self._store_native_scale(
                    up_scale_rows,
                    up_scale_mma,
                    output_row,
                    group,
                    up_scale,
                    self.groups_k,
                )


class TrellisW4A8ActivationRotationKernel(_Hadamard128):
    """Undo FC1 output rotations, apply the activation, then apply down SUH/H128."""

    threads = _THREADS

    def __init__(self, *, activation: str, fast_math: bool = False) -> None:
        activation = str(activation).lower()
        if activation not in {"silu", "situ"}:
            raise ValueError(
                f"trellis W4A8 activation must be 'silu' or 'situ', got {activation!r}"
            )
        self.activation = activation
        self.is_situ = activation == "situ"
        self.fast_math = bool(fast_math)

    @cute.jit
    def _sigmoid(self, x: Float32) -> Float32:
        e = cute.math.exp(-x, fastmath=self.fast_math)
        return Float32(1.0) / (Float32(1.0) + e)

    @cute.jit
    def _activate(self, gate: Float32, up: Float32) -> Float32:
        if cutlass.const_expr(self.is_situ):
            beta = Float32(SITU_DEFAULT_BETA)
            linear_beta = Float32(SITU_DEFAULT_LINEAR_BETA)
            return (
                beta
                * cute.math.tanh(gate / beta, fastmath=self.fast_math)
                * self._sigmoid(gate)
                * linear_beta
                * cute.math.tanh(up / linear_beta, fastmath=self.fast_math)
            )
        return gate * self._sigmoid(gate) * up

    @cute.jit
    def __call__(
        self,
        gate_ptr: cute.Pointer,
        up_ptr: cute.Pointer,
        rotations_ptr: cute.Pointer,
        route_experts_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        routes: Int32,
        num_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        gate = cute.make_tensor(
            gate_ptr, cute.make_ordered_layout((routes, 256), order=(1, 0))
        )
        up = cute.make_tensor(
            up_ptr, cute.make_ordered_layout((routes, 256), order=(1, 0))
        )
        rotations = cute.make_tensor(
            rotations_ptr,
            cute.make_ordered_layout((num_experts, 3 * 256), order=(1, 0)),
        )
        route_experts = cute.make_tensor(
            route_experts_ptr, cute.make_layout((routes,))
        )
        output = cute.make_tensor(
            output_ptr, cute.make_ordered_layout((routes, 256), order=(1, 0))
        )
        warps = self.threads // 32
        units = routes * Int32(2)
        self.kernel(
            gate, up, rotations, route_experts, output, routes, num_experts
        ).launch(
            grid=(cute.ceil_div(units, warps), 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        rotations: cute.Tensor,
        route_experts: cute.Tensor,
        output: cute.Tensor,
        routes: Int32,
        num_experts: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        unit = Int32(bidx) * Int32(self.threads // 32) + warp
        total = routes * Int32(2)
        if unit < total:
            route = unit >> Int32(1)
            blk = unit & Int32(1)
            expert = route_experts[route].to(Int32)
            valid = Int32(0)
            if expert >= Int32(0) and expert < num_experts:
                valid = Int32(1)
            scale_expert = Int32(0)
            if valid != Int32(0):
                scale_expert = expert
            col = blk * Int32(128) + lane * Int32(4)
            g0 = gate[route, col].to(Float32)
            g1 = gate[route, col + Int32(1)].to(Float32)
            g2 = gate[route, col + Int32(2)].to(Float32)
            g3 = gate[route, col + Int32(3)].to(Float32)
            u0 = up[route, col].to(Float32)
            u1 = up[route, col + Int32(1)].to(Float32)
            u2 = up[route, col + Int32(2)].to(Float32)
            u3 = up[route, col + Int32(3)].to(Float32)
            gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
            uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
            ig0 = gh0 * rotations[scale_expert, col].to(Float32)
            ig1 = gh1 * rotations[scale_expert, col + Int32(1)].to(Float32)
            ig2 = gh2 * rotations[scale_expert, col + Int32(2)].to(Float32)
            ig3 = gh3 * rotations[scale_expert, col + Int32(3)].to(Float32)
            iu0 = uh0 * rotations[scale_expert, 256 + col].to(Float32)
            iu1 = uh1 * rotations[scale_expert, 256 + col + Int32(1)].to(Float32)
            iu2 = uh2 * rotations[scale_expert, 256 + col + Int32(2)].to(Float32)
            iu3 = uh3 * rotations[scale_expert, 256 + col + Int32(3)].to(Float32)
            sd0 = rotations[scale_expert, 512 + col].to(Float32)
            sd1 = rotations[scale_expert, 512 + col + Int32(1)].to(Float32)
            sd2 = rotations[scale_expert, 512 + col + Int32(2)].to(Float32)
            sd3 = rotations[scale_expert, 512 + col + Int32(3)].to(Float32)
            a0 = self._activate(ig0, iu0) * sd0
            a1 = self._activate(ig1, iu1) * sd1
            a2 = self._activate(ig2, iu2) * sd2
            a3 = self._activate(ig3, iu3) * sd3
            if valid == Int32(0):
                a0 = Float32(0.0)
                a1 = Float32(0.0)
                a2 = Float32(0.0)
                a3 = Float32(0.0)
            o0, o1, o2, o3 = self._had128_quad(a0, a1, a2, a3, lane)
            output[route, col] = cutlass.Float16(o0)
            output[route, col + Int32(1)] = cutlass.Float16(o1)
            output[route, col + Int32(2)] = cutlass.Float16(o2)
            output[route, col + Int32(3)] = cutlass.Float16(o3)


class TrellisW4A8ActivationRotationQuantKernel(
    TrellisW4A8ActivationRotationKernel, _NativeMXFP8
):
    """Fuse inverse FC1 rotation, activation, down rotation, and MXFP8 quantization."""

    groups_k = 8

    @cute.jit
    def __call__(
        self,
        gate_ptr: cute.Pointer,
        up_ptr: cute.Pointer,
        rotations_ptr: cute.Pointer,
        route_experts_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        scale_mma_ptr: cute.Pointer,
        routes: Int32,
        num_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        gate = cute.make_tensor(
            gate_ptr, cute.make_ordered_layout((routes, 256), order=(1, 0))
        )
        up = cute.make_tensor(
            up_ptr, cute.make_ordered_layout((routes, 256), order=(1, 0))
        )
        rotations = cute.make_tensor(
            rotations_ptr,
            cute.make_ordered_layout((num_experts, 3 * 256), order=(1, 0)),
        )
        route_experts = cute.make_tensor(
            route_experts_ptr, cute.make_layout((routes,))
        )
        values = cute.make_tensor(
            values_ptr, cute.make_ordered_layout((routes, 256 // 4), order=(1, 0))
        )
        scale_rows = cute.make_tensor(
            scale_rows_ptr,
            cute.make_ordered_layout((routes, self.groups_k), order=(1, 0)),
        )
        scale_mma = cute.make_tensor(
            scale_mma_ptr,
            cute.make_layout((max(512, ((self.groups_k + 3) // 4) * 512),)),
        )
        warps = self.threads // 32
        units = routes * Int32(2)
        self.kernel_quantized(
            gate,
            up,
            rotations,
            route_experts,
            values,
            scale_rows,
            scale_mma,
            routes,
            num_experts,
        ).launch(
            grid=(cute.ceil_div(units, warps), 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel_quantized(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        rotations: cute.Tensor,
        route_experts: cute.Tensor,
        values: cute.Tensor,
        scale_rows: cute.Tensor,
        scale_mma: cute.Tensor,
        routes: Int32,
        num_experts: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        unit = Int32(bidx) * Int32(self.threads // 32) + warp
        total = routes * Int32(2)
        if unit < total:
            route = unit >> Int32(1)
            blk = unit & Int32(1)
            expert = route_experts[route].to(Int32)
            valid = Int32(0)
            if expert >= Int32(0) and expert < num_experts:
                valid = Int32(1)
            scale_expert = Int32(0)
            if valid != Int32(0):
                scale_expert = expert
            col = blk * Int32(128) + lane * Int32(4)
            g0 = gate[route, col].to(Float32)
            g1 = gate[route, col + Int32(1)].to(Float32)
            g2 = gate[route, col + Int32(2)].to(Float32)
            g3 = gate[route, col + Int32(3)].to(Float32)
            u0 = up[route, col].to(Float32)
            u1 = up[route, col + Int32(1)].to(Float32)
            u2 = up[route, col + Int32(2)].to(Float32)
            u3 = up[route, col + Int32(3)].to(Float32)
            gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
            uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
            ig0 = gh0 * rotations[scale_expert, col].to(Float32)
            ig1 = gh1 * rotations[scale_expert, col + Int32(1)].to(Float32)
            ig2 = gh2 * rotations[scale_expert, col + Int32(2)].to(Float32)
            ig3 = gh3 * rotations[scale_expert, col + Int32(3)].to(Float32)
            iu0 = uh0 * rotations[scale_expert, 256 + col].to(Float32)
            iu1 = uh1 * rotations[scale_expert, 256 + col + Int32(1)].to(Float32)
            iu2 = uh2 * rotations[scale_expert, 256 + col + Int32(2)].to(Float32)
            iu3 = uh3 * rotations[scale_expert, 256 + col + Int32(3)].to(Float32)
            sd0 = rotations[scale_expert, 512 + col].to(Float32)
            sd1 = rotations[scale_expert, 512 + col + Int32(1)].to(Float32)
            sd2 = rotations[scale_expert, 512 + col + Int32(2)].to(Float32)
            sd3 = rotations[scale_expert, 512 + col + Int32(3)].to(Float32)
            a0 = self._activate(ig0, iu0) * sd0
            a1 = self._activate(ig1, iu1) * sd1
            a2 = self._activate(ig2, iu2) * sd2
            a3 = self._activate(ig3, iu3) * sd3
            if valid == Int32(0):
                a0 = Float32(0.0)
                a1 = Float32(0.0)
                a2 = Float32(0.0)
                a3 = Float32(0.0)
            o0, o1, o2, o3 = self._had128_quad(a0, a1, a2, a3, lane)
            payload, scale_byte = self._pack_native_mxfp8(
                o0, o1, o2, o3, lane
            )
            subgroup = lane >> Int32(3)
            lane8 = lane & Int32(7)
            group = blk * Int32(4) + subgroup
            values[route, group * Int32(8) + lane8] = payload
            if lane8 == Int32(0):
                self._store_native_scale(
                    scale_rows,
                    scale_mma,
                    route,
                    group,
                    scale_byte,
                    self.groups_k,
                )


def _ptr(dtype, address: int):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=16)


@functools.cache
def _compile_input_rotation(
    hidden_size: int,
    topk: int,
    source_dtype: torch.dtype,
    broadcast_suh: bool,
    device_index: int,
):
    if source_dtype == torch.bfloat16:
        source_type = cutlass.BFloat16
    elif source_dtype == torch.float16:
        source_type = cutlass.Float16
    else:
        raise TypeError(f"trellis W4A8 input must be bf16/fp16, got {source_dtype}")
    launch = TrellisW4A8InputRotationKernel(
        hidden_size=hidden_size,
        topk=topk,
        source_type=source_type,
        broadcast_suh=broadcast_suh,
    )
    key = (
        int(hidden_size),
        int(topk),
        str(source_dtype),
        bool(broadcast_suh),
        int(device_index),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(source_type, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.trellis_w4a8_input_rotation", 1, key
        ),
    )


@functools.cache
def _compile_input_rotation_quant(
    hidden_size: int,
    topk: int,
    source_dtype: torch.dtype,
    broadcast_suh: bool,
    device_index: int,
):
    if source_dtype == torch.bfloat16:
        source_type = cutlass.BFloat16
    elif source_dtype == torch.float16:
        source_type = cutlass.Float16
    else:
        raise TypeError(f"trellis W4A8 input must be bf16/fp16, got {source_dtype}")
    launch = TrellisW4A8InputRotationQuantKernel(
        hidden_size=hidden_size,
        topk=topk,
        source_type=source_type,
        broadcast_suh=broadcast_suh,
    )
    key = (
        int(hidden_size),
        int(topk),
        str(source_dtype),
        bool(broadcast_suh),
        int(device_index),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(source_type, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint8, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.trellis_w4a8_input_rotation_quant", 1, key
        ),
    )


@functools.cache
def _compile_activation_rotation(
    fast_math: bool, activation: str, device_index: int
):
    launch = TrellisW4A8ActivationRotationKernel(
        activation=activation, fast_math=fast_math
    )
    key = (bool(fast_math), str(activation), int(device_index))
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.trellis_w4a8_activation_rotation", 2, key
        ),
    )


@functools.cache
def _compile_activation_rotation_quant(
    fast_math: bool, activation: str, device_index: int
):
    launch = TrellisW4A8ActivationRotationQuantKernel(
        activation=activation, fast_math=fast_math
    )
    key = (bool(fast_math), str(activation), int(device_index))
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint8, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.trellis_w4a8_activation_rotation_quant", 2, key
        ),
    )


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def run_trellis_w4a8_input_rotation(
    source: torch.Tensor,
    route_experts: torch.Tensor,
    prepared,
    gate_out: torch.Tensor,
    up_out: torch.Tensor,
    *,
    topk: int,
) -> None:
    broadcast_suh = int(prepared.gate_suh.shape[0]) == 1
    compiled = _compile_input_rotation(
        int(prepared.hidden_size),
        int(topk),
        source.dtype,
        broadcast_suh,
        _device_index(source.device),
    )
    compiled(
        _ptr(cutlass.BFloat16 if source.dtype == torch.bfloat16 else cutlass.Float16, source.data_ptr()),
        _ptr(cutlass.Int32, route_experts.data_ptr()),
        _ptr(cutlass.Float16, prepared.gate_suh.data_ptr()),
        _ptr(cutlass.Float16, prepared.up_suh.data_ptr()),
        _ptr(cutlass.Float16, gate_out.data_ptr()),
        _ptr(cutlass.Float16, up_out.data_ptr()),
        int(source.shape[0]),
        int(prepared.num_experts),
        current_cuda_stream(),
    )


def run_trellis_w4a8_input_rotation_quant(
    source: torch.Tensor,
    route_experts: torch.Tensor,
    prepared,
    gate_out,
    up_out,
    *,
    topk: int,
) -> None:
    """Apply both input rotations directly into native-order MXFP8 rows."""

    broadcast_suh = int(prepared.gate_suh.shape[0]) == 1
    compiled = _compile_input_rotation_quant(
        int(prepared.hidden_size),
        int(topk),
        source.dtype,
        broadcast_suh,
        _device_index(source.device),
    )
    source_type = (
        cutlass.BFloat16 if source.dtype == torch.bfloat16 else cutlass.Float16
    )
    compiled(
        _ptr(source_type, source.data_ptr()),
        _ptr(cutlass.Int32, route_experts.data_ptr()),
        _ptr(cutlass.Float16, prepared.gate_suh.data_ptr()),
        _ptr(cutlass.Float16, prepared.up_suh.data_ptr()),
        _ptr(cutlass.Uint32, gate_out.values.data_ptr()),
        _ptr(cutlass.Uint8, gate_out.scale_rows.data_ptr()),
        _ptr(cutlass.Uint8, gate_out.scale_mma.data_ptr()),
        _ptr(cutlass.Uint32, up_out.values.data_ptr()),
        _ptr(cutlass.Uint8, up_out.scale_rows.data_ptr()),
        _ptr(cutlass.Uint8, up_out.scale_mma.data_ptr()),
        int(source.shape[0]),
        int(prepared.num_experts),
        current_cuda_stream(),
    )


def run_trellis_w4a8_activation_rotation(
    gate: torch.Tensor,
    up: torch.Tensor,
    route_experts: torch.Tensor,
    prepared,
    output: torch.Tensor,
    *,
    fast_math: bool = False,
) -> None:
    compiled = _compile_activation_rotation(
        bool(fast_math),
        str(getattr(prepared, "activation", "")).lower(),
        _device_index(gate.device),
    )
    compiled(
        _ptr(cutlass.Float16, gate.data_ptr()),
        _ptr(cutlass.Float16, up.data_ptr()),
        _ptr(cutlass.Float16, prepared.intermediate_rotations.data_ptr()),
        _ptr(cutlass.Int32, route_experts.data_ptr()),
        _ptr(cutlass.Float16, output.data_ptr()),
        int(route_experts.numel()),
        int(prepared.num_experts),
        current_cuda_stream(),
    )


def run_trellis_w4a8_activation_rotation_quant(
    gate: torch.Tensor,
    up: torch.Tensor,
    route_experts: torch.Tensor,
    prepared,
    output,
    *,
    fast_math: bool = False,
) -> None:
    """Apply activation/rotation directly into native-order MXFP8 rows."""

    compiled = _compile_activation_rotation_quant(
        bool(fast_math),
        str(getattr(prepared, "activation", "")).lower(),
        _device_index(gate.device),
    )
    compiled(
        _ptr(cutlass.Float16, gate.data_ptr()),
        _ptr(cutlass.Float16, up.data_ptr()),
        _ptr(cutlass.Float16, prepared.intermediate_rotations.data_ptr()),
        _ptr(cutlass.Int32, route_experts.data_ptr()),
        _ptr(cutlass.Uint32, output.values.data_ptr()),
        _ptr(cutlass.Uint8, output.scale_rows.data_ptr()),
        _ptr(cutlass.Uint8, output.scale_mma.data_ptr()),
        int(route_experts.numel()),
        int(prepared.num_experts),
        current_cuda_stream(),
    )


__all__ = [
    "run_trellis_w4a8_activation_rotation",
    "run_trellis_w4a8_activation_rotation_quant",
    "run_trellis_w4a8_input_rotation",
    "run_trellis_w4a8_input_rotation_quant",
]
