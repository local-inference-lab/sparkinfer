"""MoETinyDecodeKernelBackend — tiny-decode (M<=4) W4A8-MX MoE for SM120.

Reads the N256/K128 in-place-REPACKED weight storage (the "rp" layout produced
by _logical_weight_to_w4a8_rp_inplace) directly.

Consumes the N256/K128 in-place-repacked FP4 weights and e8m0 sfb grids
directly (inverse mappings verified in tests/test_w4a8_rp_inverse_mapping.py).
Both the BF16 input and FC1 activation are quantized/dequantized in-register
as MXFP8 E4M3 with one UE8M0 scale per 32 values, matching the W4A8 contract.
Two plain (non-cooperative) launches: FC1 dots gate+up rows into an fp32
intermediate, FC2 applies SiLU and the second MXFP8 round-trip inline, then
applies router weights after FC2. The wrapper zeroes the intermediate and
output first; there are no grid barriers and no CTA co-residency assumptions,
so the kernels are safe on busy serving streams.

Thread mapping (256 threads/CTA), the core of the rp coalescing story: one rp
(nt, kt) tile is 4096 contiguous int32 words whose flat index decomposes as
``k32<<10 | n8c<<7 | r8<<4 | cgrp<<2 | n8i``. Thread t = (n8c=t>>5, r8=(t>>2)&7,
cgrp=t&3) issues one 16 B ``ld.global.nc.v4`` per k32 covering n8i=0..3 — four
8-apart logical rows at one k window — so a warp (fixed n8c) covers a fully
coalesced 512 B run per k32, and the 4-lane cgrp butterfly folds the k dim.

The prep rotation normalizes both declared w13 layouts to the same rp tile
order (tiles [0, N/256) hold the "up" rows, [N/256, 2N/256) the "gate" rows of
the same channels), so FC1 stores inter rows through ``(p + n) % 2n`` and FC2
reads gate at [0, n), up at [n, 2n) unconditionally.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cutlass_dsl import Int32, Int64

from b12x.cute.utils import current_cuda_stream, make_ptr
from b12x.cute.fp4 import (
    cvt_bf16x2_to_f16x2,
    cvt_e8m0x4_to_f32x4,
    fmax_f32,
    fp4_dot4_sum_f32acc,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    pack_f32x2_to_f16x2,
    mx_scale_from_amax32,
    quant_dequant_e4m3_2,
    red_add_global_f32,
    scatter_add_bf16x2,
    warp_reduce,
)

_BLOCK_THREADS = 256
_FC1_KT_PER_TASK = 4
_FC2_KT_PER_TASK = 2


class MoETinyDecodeKernelBackend:
    """Tiny-M (decode) W4A8-MX kernel reading the repacked weight layout."""

    def __init__(
        self,
        *,
        activation: str = "silu",
        w13_layout: str = "w31",
        compile_time_phase: int = 1,
    ):
        if activation != "silu":
            raise ValueError(f"tiny_decode supports silu only, got {activation!r}")
        if int(compile_time_phase) not in (1, 2):
            raise ValueError(f"unsupported tiny_decode phase {compile_time_phase!r}")
        self.compile_time_phase = int(compile_time_phase)
        self.activation = activation
        self.w13_layout = w13_layout
        self._cfg_key = None
        self._c = None
        self.grid_x = 0

    def configure(
        self,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        del device
        if k % 256 != 0 or n % 256 != 0:
            raise ValueError("tiny_decode requires k % 256 == 0 and n % 256 == 0")
        if m < 1 or m > 4:
            raise ValueError("tiny_decode supports 1 <= m <= 4")
        rt = m * num_topk
        kt13 = k // 128
        kt2 = n // 128
        if kt13 % _FC1_KT_PER_TASK != 0 or kt2 % _FC2_KT_PER_TASK != 0:
            raise ValueError("tiny_decode k-tile counts not divisible by task sizes")
        cfg = dict(
            m=m,
            k=k,
            n=n,
            two_n=2 * n,
            num_topk=num_topk,
            weight_E=weight_E,
            rt=rt,
            nt13=(2 * n) // 256,
            kt13=kt13,
            fc1_ktg=kt13 // _FC1_KT_PER_TASK,
            nt2=k // 256,
            kt2=kt2,
            fc2_ktg=kt2 // _FC2_KT_PER_TASK,
            w13_words=(2 * n) * k // 8,
            w2_words=k * n // 8,
            sfb13_bytes=(2 * n) * (k // 32),
            sfb2_bytes=k * (n // 32),
            fc1_tasks=rt * ((2 * n) // 256) * (kt13 // _FC1_KT_PER_TASK),
            fc2_tasks=rt * (k // 256) * (kt2 // _FC2_KT_PER_TASK),
        )
        self._c = cfg
        self._cfg_key = tuple(sorted(cfg.items()))
        self.grid_x = (
            cfg["fc1_tasks"] if self.compile_time_phase == 1 else cfg["fc2_tasks"]
        )

    @property
    def __cache_key__(self):
        return (
            self.activation,
            self.w13_layout,
            self.compile_time_phase,
            self._cfg_key,
            self.grid_x,
        )

    @cute.jit
    def _row_block_dot(
        self,
        tile_word: Int64,
        srow: Int64,
        n8c: Int32,
        r8: Int32,
        cgrp: Int32,
        x0_0: cutlass.Uint32, x1_0: cutlass.Uint32, x2_0: cutlass.Uint32, x3_0: cutlass.Uint32,
        x0_1: cutlass.Uint32, x1_1: cutlass.Uint32, x2_1: cutlass.Uint32, x3_1: cutlass.Uint32,
        x0_2: cutlass.Uint32, x1_2: cutlass.Uint32, x2_2: cutlass.Uint32, x3_2: cutlass.Uint32,
        x0_3: cutlass.Uint32, x1_3: cutlass.Uint32, x2_3: cutlass.Uint32, x3_3: cutlass.Uint32,
    ):
        """Dot 4 n8i rows (v=0..3) x 128 k window against packed activations.

        Activation f16x2 quads are per k32 (x*_<k32>). Returns 4 row partials.
        """
        word_off = Int64(n8c * Int32(128) + r8 * Int32(16) + cgrp * Int32(4)) * Int64(4)
        acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
        xq = (
            (x0_0, x1_0, x2_0, x3_0),
            (x0_1, x1_1, x2_1, x3_1),
            (x0_2, x1_2, x2_2, x3_2),
            (x0_3, x1_3, x2_3, x3_3),
        )
        for v in cutlass.range_constexpr(4):
            sv = ld_global_nc_u32(srow + Int64(v * 32))
            sk = cvt_e8m0x4_to_f32x4(sv)
            accv = Float32(0.0)
            for k32 in cutlass.range_constexpr(4):
                words = ld_global_nc_v4_u32(tile_word + Int64(k32 * 4096) + word_off)
                x0, x1, x2, x3 = xq[k32]
                accv += sk[k32] * fp4_dot4_sum_f32acc(words[v], x0, x1, x2, x3)
            acc[v] += accv
        return acc[0], acc[1], acc[2], acc[3]

    @cute.jit
    def _mxfp8_roundtrip_8(
        self,
        v0: Float32, v1: Float32, v2: Float32, v3: Float32,
        v4: Float32, v5: Float32, v6: Float32, v7: Float32,
    ):
        """Round-trip this lane's 8 values through its CTA group's K32 scale.

        Four adjacent ``cgrp`` lanes own one logical K32 block.  The width-4
        warp reduction therefore produces precisely the UE8M0 scale used by
        the W4A8 MXFP8 reference, without a shared-memory staging buffer.
        """
        peak = fmax_f32(v0, -v0)
        peak = fmax_f32(peak, fmax_f32(v1, -v1))
        peak = fmax_f32(peak, fmax_f32(v2, -v2))
        peak = fmax_f32(peak, fmax_f32(v3, -v3))
        peak = fmax_f32(peak, fmax_f32(v4, -v4))
        peak = fmax_f32(peak, fmax_f32(v5, -v5))
        peak = fmax_f32(peak, fmax_f32(v6, -v6))
        peak = fmax_f32(peak, fmax_f32(v7, -v7))
        peak = warp_reduce(peak, fmax_f32, width=4)
        scale, inv_scale = mx_scale_from_amax32(peak)
        q0, q1 = quant_dequant_e4m3_2(v0, v1, inv_scale, scale)
        q2, q3 = quant_dequant_e4m3_2(v2, v3, inv_scale, scale)
        q4, q5 = quant_dequant_e4m3_2(v4, v5, inv_scale, scale)
        q6, q7 = quant_dequant_e4m3_2(v6, v7, inv_scale, scale)
        return (
            pack_f32x2_to_f16x2(q0, q1),
            pack_f32x2_to_f16x2(q2, q3),
            pack_f32x2_to_f16x2(q4, q5),
            pack_f32x2_to_f16x2(q6, q7),
        )

    @cute.jit
    def _load_mxfp8_input_8(
        self,
        qinput: cute.Tensor,
        word_idx: Int32,
    ):
        """Load eight pre-quantized/dequantized f16 values as four pairs."""
        return (
            qinput[word_idx + Int32(0)],
            qinput[word_idx + Int32(1)],
            qinput[word_idx + Int32(2)],
            qinput[word_idx + Int32(3)],
        )

    @cute.kernel
    def kernel(
        self,
        a_input: cute.Tensor,       # bf16 [m*k]
        qinput: cute.Tensor,        # f16x2 words [m*k/2]
        qinter: cute.Tensor,        # f16x2 words [rt*n/2]
        w13: cute.Tensor,           # u8 rp bytes, expert-major
        sfb13: cute.Tensor,         # u8 sfb bytes, expert-major
        inter: cute.Tensor,         # f32 [rt * 2n] (pre-zeroed for phase 1)
        w2: cute.Tensor,            # u8 rp bytes
        sfb2: cute.Tensor,          # u8 sfb bytes
        topk_ids: cute.Tensor,      # i32 [rt]
        topk_weights: cute.Tensor,  # f32 [rt]
        out: cute.Tensor,           # bf16 [m*k] (pre-zeroed for phase 2)
    ):
        c = self._c
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n8c = tidx // Int32(32)
        r8 = (tidx // Int32(4)) % Int32(8)
        cgrp = tidx % Int32(4)

        w13_base = get_ptr_as_int64(w13, Int32(0))
        sfb13_base = get_ptr_as_int64(sfb13, Int32(0))
        inter_base = get_ptr_as_int64(inter, Int32(0))
        w2_base = get_ptr_as_int64(w2, Int32(0))
        sfb2_base = get_ptr_as_int64(sfb2, Int32(0))
        out_base = get_ptr_as_int64(out, Int32(0))

        if cutlass.const_expr(self.compile_time_phase == 1):
            # ---- FC1: block = (rt, nt13, ktg) ----
            fc1_per_rt = Int32(c["nt13"] * c["fc1_ktg"])
            rt_idx = bidx // fc1_per_rt
            rem = bidx % fc1_per_rt
            nt = rem // Int32(c["fc1_ktg"])
            ktg = rem % Int32(c["fc1_ktg"])
            eid = Int64(Int32(topk_ids[rt_idx]))
            tok = rt_idx // Int32(c["num_topk"])
            we_base = w13_base + eid * Int64(c["w13_words"] * 4)
            se_base = sfb13_base + eid * Int64(c["sfb13_bytes"])
            srow_base = se_base + Int64(r8 * Int32(4) + n8c * Int32(128))

            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            acc2 = Float32(0.0)
            acc3 = Float32(0.0)
            for kt_i in cutlass.range_constexpr(_FC1_KT_PER_TASK):
                kt = ktg * Int32(_FC1_KT_PER_TASK) + Int32(kt_i)
                col_tile = nt * Int32(c["kt13"]) + kt
                tile_word = we_base + Int64(col_tile) * Int64(4096 * 4)
                srow = srow_base + Int64(col_tile) * Int64(1024)
                xs = []
                for k32 in cutlass.range_constexpr(4):
                    x_idx = tok * Int32(c["k"]) + kt * Int32(128) + cgrp * Int32(8) + Int32(k32 * 32)
                    xs.append(self._load_mxfp8_input_8(
                        qinput,
                        (x_idx >> Int32(1)),
                    ))
                d0, d1, d2, d3 = self._row_block_dot(
                    tile_word, srow, n8c, r8, cgrp,
                    xs[0][0], xs[0][1], xs[0][2], xs[0][3],
                    xs[1][0], xs[1][1], xs[1][2], xs[1][3],
                    xs[2][0], xs[2][1], xs[2][2], xs[2][3],
                    xs[3][0], xs[3][1], xs[3][2], xs[3][3],
                )
                acc0 += d0
                acc1 += d1
                acc2 += d2
                acc3 += d3
            acc0 = warp_reduce(acc0, lambda a, b: a + b, width=4)
            acc1 = warp_reduce(acc1, lambda a, b: a + b, width=4)
            acc2 = warp_reduce(acc2, lambda a, b: a + b, width=4)
            acc3 = warp_reduce(acc3, lambda a, b: a + b, width=4)
            if cgrp == Int32(0):
                ibase_rt = inter_base + Int64(rt_idx) * Int64(c["two_n"] * 4)
                accs = (acc0, acc1, acc2, acc3)
                for v in cutlass.range_constexpr(4):
                    p = nt * Int32(256) + n8c * Int32(32) + Int32(v * 8) + r8
                    r_log = p + Int32(c["n"])
                    if r_log >= Int32(c["two_n"]):
                        r_log -= Int32(c["two_n"])
                    red_add_global_f32(ibase_rt + Int64(r_log) * Int64(4), accs[v])

        if cutlass.const_expr(self.compile_time_phase == 2):
            # ---- FC2: block = (rt, nt2, ktg2) ----
            fc2_per_rt = Int32(c["nt2"] * c["fc2_ktg"])
            rt_idx = bidx // fc2_per_rt
            rem = bidx % fc2_per_rt
            nt = rem // Int32(c["fc2_ktg"])
            ktg = rem % Int32(c["fc2_ktg"])
            eid = Int64(Int32(topk_ids[rt_idx]))
            tok = rt_idx // Int32(c["num_topk"])
            rw = Float32(topk_weights[rt_idx])
            we_base = w2_base + eid * Int64(c["w2_words"] * 4)
            se_base = sfb2_base + eid * Int64(c["sfb2_bytes"])
            srow_base = se_base + Int64(r8 * Int32(4) + n8c * Int32(128))
            ibase = rt_idx * Int32(c["n"])

            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            acc2 = Float32(0.0)
            acc3 = Float32(0.0)
            for kt_i in cutlass.range_constexpr(_FC2_KT_PER_TASK):
                kt = ktg * Int32(_FC2_KT_PER_TASK) + Int32(kt_i)
                col_tile = nt * Int32(c["kt2"]) + kt
                tile_word = we_base + Int64(col_tile) * Int64(4096 * 4)
                srow = srow_base + Int64(col_tile) * Int64(1024)
                xs = []
                for k32 in cutlass.range_constexpr(4):
                    ich = ibase + kt * Int32(128) + cgrp * Int32(8) + Int32(k32 * 32)
                    xs.append(self._load_mxfp8_input_8(qinter, ich >> Int32(1)))
                d0, d1, d2, d3 = self._row_block_dot(
                    tile_word, srow, n8c, r8, cgrp,
                    xs[0][0], xs[0][1], xs[0][2], xs[0][3],
                    xs[1][0], xs[1][1], xs[1][2], xs[1][3],
                    xs[2][0], xs[2][1], xs[2][2], xs[2][3],
                    xs[3][0], xs[3][1], xs[3][2], xs[3][3],
                )
                acc0 += d0
                acc1 += d1
                acc2 += d2
                acc3 += d3
            acc0 = warp_reduce(acc0, lambda a, b: a + b, width=4)
            acc1 = warp_reduce(acc1, lambda a, b: a + b, width=4)
            acc2 = warp_reduce(acc2, lambda a, b: a + b, width=4)
            acc3 = warp_reduce(acc3, lambda a, b: a + b, width=4)
            acc0 = acc0 * rw
            acc1 = acc1 * rw
            acc2 = acc2 * rw
            acc3 = acc3 * rw
            # pair consecutive output rows: partner lane differs in r8 bit0 (lane^4)
            o0 = cute.arch.shuffle_sync_bfly(acc0, offset=4)
            o1 = cute.arch.shuffle_sync_bfly(acc1, offset=4)
            o2 = cute.arch.shuffle_sync_bfly(acc2, offset=4)
            o3 = cute.arch.shuffle_sync_bfly(acc3, offset=4)
            if cgrp == Int32(0):
                if (r8 % Int32(2)) == Int32(0):
                    ob = out_base + Int64(tok) * Int64(c["k"] * 2)
                    accs = (acc0, acc1, acc2, acc3)
                    others = (o0, o1, o2, o3)
                    for v in cutlass.range_constexpr(4):
                        p2 = nt * Int32(256) + n8c * Int32(32) + Int32(v * 8) + r8
                        scatter_add_bf16x2(
                            ob + Int64(p2) * Int64(2), accs[v], others[v]
                        )

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        qinput_ptr: cute.Pointer,
        qinter_ptr: cute.Pointer,
        w13_ptr: cute.Pointer,
        sfb13_ptr: cute.Pointer,
        inter_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        tid_ptr: cute.Pointer,
        tw_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        stream,
    ):
        c = self._c
        a_input = cute.make_tensor(x_ptr, cute.make_layout(Int32(c["m"] * c["k"])))
        qinput = cute.make_tensor(
            qinput_ptr, cute.make_layout(Int32(c["m"] * (c["k"] // 2)))
        )
        qinter = cute.make_tensor(
            qinter_ptr, cute.make_layout(Int32(c["rt"] * (c["n"] // 2)))
        )
        w13 = cute.make_tensor(
            w13_ptr, cute.make_layout(Int64(c["weight_E"] * c["w13_words"] * 4))
        )
        sfb13 = cute.make_tensor(
            sfb13_ptr, cute.make_layout(Int64(c["weight_E"] * c["sfb13_bytes"]))
        )
        inter = cute.make_tensor(
            inter_ptr, cute.make_layout(Int32(c["rt"] * c["two_n"]))
        )
        w2 = cute.make_tensor(
            w2_ptr, cute.make_layout(Int64(c["weight_E"] * c["w2_words"] * 4))
        )
        sfb2 = cute.make_tensor(
            sfb2_ptr, cute.make_layout(Int64(c["weight_E"] * c["sfb2_bytes"]))
        )
        topk_ids = cute.make_tensor(tid_ptr, cute.make_layout(Int32(c["rt"])))
        topk_weights = cute.make_tensor(tw_ptr, cute.make_layout(Int32(c["rt"])))
        out = cute.make_tensor(out_ptr, cute.make_layout(Int32(c["m"] * c["k"])))

        self.kernel(
            a_input, qinput, qinter, w13, sfb13, inter, w2, sfb2, topk_ids, topk_weights, out,
        ).launch(
            grid=(Int32(self.grid_x), Int32(1), Int32(1)),
            block=(_BLOCK_THREADS, 1, 1),
            smem=0,
            stream=stream,
        )

    @staticmethod
    def launch(
        compiled_fc1,
        compiled_fc2,
        compiled_input_quant,
        compiled_intermediate_quant,
        *,
        x: torch.Tensor,
        qinput: torch.Tensor,
        qinter: torch.Tensor,
        w13_rp: torch.Tensor,
        sfb13: torch.Tensor,
        inter_fp32: torch.Tensor,
        w2_rp: torch.Tensor,
        sfb2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor,
    ):
        def ptr(dt, t, align=16):
            return make_ptr(
                dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
            )

        MoETinyDecodeInputQuantizer.launch(compiled_input_quant, x=x, out=qinput)
        inter_fp32.zero_()
        out.zero_()
        stream = current_cuda_stream()
        args = (
            ptr(cutlass.BFloat16, x),
            ptr(cutlass.Uint32, qinput, 4),
            ptr(cutlass.Uint32, qinter, 4),
            ptr(cutlass.Uint8, w13_rp.view(torch.uint8)),
            ptr(cutlass.Uint8, sfb13.view(torch.uint8)),
            ptr(cutlass.Float32, inter_fp32),
            ptr(cutlass.Uint8, w2_rp.view(torch.uint8)),
            ptr(cutlass.Uint8, sfb2.view(torch.uint8)),
            ptr(cutlass.Int32, topk_ids, 4),
            ptr(cutlass.Float32, topk_weights, 4),
            ptr(cutlass.BFloat16, out),
            stream,
        )
        compiled_fc1(*args)
        MoETinyDecodeIntermediateQuantizer.launch(
            compiled_intermediate_quant, inter=inter_fp32, out=qinter
        )
        compiled_fc2(*args)


class MoETinyDecodeInputQuantizer:
    """K32 MXFP8 round-trip prepass for the tiny FC1 schedule.

    It writes the f16x2 values consumed by the scalar FP4 dot directly.  This
    is deliberately not a general MXFP8 packing kernel: avoiding a payload
    decode in every FC1 CTA is the point of the tiny-decode specialization.
    """

    def __init__(self, m: int, k: int):
        if m < 1 or k <= 0 or k % 256 != 0:
            raise ValueError("tiny input quantizer requires 1<=m and k % 256 == 0")
        self.m = int(m)
        self.k = int(k)
        self.tiles_per_row = self.k // 256
        # One CTA has eight warps; each warp quantizes eight K32 blocks.
        self.grid_x = self.m * ((self.tiles_per_row + 7) // 8)

    @property
    def __cache_key__(self):
        return (self.m, self.k, self.tiles_per_row, self.grid_x)

    @cute.kernel
    def kernel(self, x: cute.Tensor, out_u32: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // Int32(32)
        lane = tidx % Int32(32)
        subgroup = lane // Int32(4)
        lane8 = lane % Int32(4)
        task = bidx * Int32(8) + warp
        row = task // Int32(self.tiles_per_row)
        tile = task - row * Int32(self.tiles_per_row)
        group = tile * Int32(8) + subgroup
        valid = Int32(1) if row < Int32(self.m) else Int32(0)

        value_idx = row * Int32(self.k) + group * Int32(32) + lane8 * Int32(8)
        v0 = Float32(x[value_idx + Int32(0)]) if valid > Int32(0) else Float32(0.0)
        v1 = Float32(x[value_idx + Int32(1)]) if valid > Int32(0) else Float32(0.0)
        v2 = Float32(x[value_idx + Int32(2)]) if valid > Int32(0) else Float32(0.0)
        v3 = Float32(x[value_idx + Int32(3)]) if valid > Int32(0) else Float32(0.0)
        v4 = Float32(x[value_idx + Int32(4)]) if valid > Int32(0) else Float32(0.0)
        v5 = Float32(x[value_idx + Int32(5)]) if valid > Int32(0) else Float32(0.0)
        v6 = Float32(x[value_idx + Int32(6)]) if valid > Int32(0) else Float32(0.0)
        v7 = Float32(x[value_idx + Int32(7)]) if valid > Int32(0) else Float32(0.0)

        peak = fmax_f32(v0, -v0)
        peak = fmax_f32(peak, fmax_f32(v1, -v1))
        peak = fmax_f32(peak, fmax_f32(v2, -v2))
        peak = fmax_f32(peak, fmax_f32(v3, -v3))
        peak = fmax_f32(peak, fmax_f32(v4, -v4))
        peak = fmax_f32(peak, fmax_f32(v5, -v5))
        peak = fmax_f32(peak, fmax_f32(v6, -v6))
        peak = fmax_f32(peak, fmax_f32(v7, -v7))
        peak = warp_reduce(peak, fmax_f32, width=4)
        scale, inv_scale = mx_scale_from_amax32(peak)
        q0, q1 = quant_dequant_e4m3_2(v0, v1, inv_scale, scale)
        q2, q3 = quant_dequant_e4m3_2(v2, v3, inv_scale, scale)
        q4, q5 = quant_dequant_e4m3_2(v4, v5, inv_scale, scale)
        q6, q7 = quant_dequant_e4m3_2(v6, v7, inv_scale, scale)
        out_idx = row * Int32(self.k // 2) + group * Int32(16) + lane8 * Int32(4)
        if valid > Int32(0):
            out_u32[out_idx + Int32(0)] = pack_f32x2_to_f16x2(q0, q1)
            out_u32[out_idx + Int32(1)] = pack_f32x2_to_f16x2(q2, q3)
            out_u32[out_idx + Int32(2)] = pack_f32x2_to_f16x2(q4, q5)
            out_u32[out_idx + Int32(3)] = pack_f32x2_to_f16x2(q6, q7)

    @cute.jit
    def __call__(self, x_ptr: cute.Pointer, out_ptr: cute.Pointer, stream):
        x = cute.make_tensor(x_ptr, cute.make_layout(Int32(self.m * self.k)))
        out_u32 = cute.make_tensor(
            out_ptr, cute.make_layout(Int32(self.m * (self.k // 2)))
        )
        self.kernel(x, out_u32).launch(
            grid=(Int32(self.grid_x), Int32(1), Int32(1)),
            block=(_BLOCK_THREADS, 1, 1),
            smem=0,
            stream=stream,
        )

    @staticmethod
    def launch(compiled, *, x: torch.Tensor, out: torch.Tensor) -> None:
        compiled(
            make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            current_cuda_stream(),
        )


class MoETinyDecodeIntermediateQuantizer:
    """Route-local SiLU and MXFP8 K32 round-trip for tiny FC2.

    FC2 has many N tiles for one route but they all consume the same 256-wide
    activated intermediate.  Quantizing that intermediate once avoids doing
    the nonlinear operation and K32 reduction in every output CTA.
    """

    def __init__(self, rt: int, n: int):
        if rt < 1 or n <= 0 or n % 256 != 0:
            raise ValueError("tiny intermediate quantizer requires rt>=1 and n % 256 == 0")
        self.rt = int(rt)
        self.n = int(n)
        self.tiles_per_route = self.n // 256

    @property
    def __cache_key__(self):
        return (self.rt, self.n, self.tiles_per_route)

    @cute.kernel
    def kernel(self, inter: cute.Tensor, out_u32: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        route = bidx // Int32(self.tiles_per_route)
        tile = bidx - route * Int32(self.tiles_per_route)
        subgroup = tidx // Int32(4)
        lane8 = tidx % Int32(4)
        group = tile * Int32(8) + subgroup
        base = route * Int32(2 * self.n) + group * Int32(32) + lane8 * Int32(8)

        vals = []
        for i in cutlass.range_constexpr(8):
            g = Float32(inter[base + Int32(i)])
            u = Float32(inter[base + Int32(self.n) + Int32(i)])
            sigmoid = Float32(1.0) / (
                Float32(1.0) + cute.math.exp(-g, fastmath=False)
            )
            vals.append(sigmoid * g * u)
        peak = fmax_f32(vals[0], -vals[0])
        for i in cutlass.range_constexpr(1, 8):
            peak = fmax_f32(peak, fmax_f32(vals[i], -vals[i]))
        peak = warp_reduce(peak, fmax_f32, width=4)
        scale, inv_scale = mx_scale_from_amax32(peak)
        q0, q1 = quant_dequant_e4m3_2(vals[0], vals[1], inv_scale, scale)
        q2, q3 = quant_dequant_e4m3_2(vals[2], vals[3], inv_scale, scale)
        q4, q5 = quant_dequant_e4m3_2(vals[4], vals[5], inv_scale, scale)
        q6, q7 = quant_dequant_e4m3_2(vals[6], vals[7], inv_scale, scale)
        out_idx = route * Int32(self.n // 2) + group * Int32(16) + lane8 * Int32(4)
        out_u32[out_idx + Int32(0)] = pack_f32x2_to_f16x2(q0, q1)
        out_u32[out_idx + Int32(1)] = pack_f32x2_to_f16x2(q2, q3)
        out_u32[out_idx + Int32(2)] = pack_f32x2_to_f16x2(q4, q5)
        out_u32[out_idx + Int32(3)] = pack_f32x2_to_f16x2(q6, q7)

    @cute.jit
    def __call__(self, inter_ptr: cute.Pointer, out_ptr: cute.Pointer, stream):
        inter = cute.make_tensor(
            inter_ptr, cute.make_layout(Int32(self.rt * 2 * self.n))
        )
        out_u32 = cute.make_tensor(
            out_ptr, cute.make_layout(Int32(self.rt * (self.n // 2)))
        )
        self.kernel(inter, out_u32).launch(
            grid=(Int32(self.rt * self.tiles_per_route), Int32(1), Int32(1)),
            block=(32, 1, 1),
            smem=0,
            stream=stream,
        )

    @staticmethod
    def launch(compiled, *, inter: torch.Tensor, out: torch.Tensor) -> None:
        compiled(
            make_ptr(cutlass.Float32, inter.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            current_cuda_stream(),
        )
