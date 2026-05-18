"""W4A16 dense GEMM kernel — forked from b12x/moe/fused/w4a16/static.py.

Stripped of MoE routing / gated SwiGLU / scatter-output. Pure dense:

    out[M, N] = (A[M, K] @ dequant(W_fp4[N, K/2], W_sf[N, K/16]).T) * alpha

* A: bf16, row-major (M, K)
* W: FP4-packed uint8, (N, K/2), one expert (no expert dim)
* SF: FP8 e4m3, swizzled per ``_swizzled_e4m3_offset`` (same as MoE)
* alpha: scalar fp32
* C: bf16, row-major (M, N)

Architecture (kept from static.py):
* TMA load for A (async, multi-stage).
* Multi-stage AB pipeline (``ab_stage = 2``) via ``PipelineTmaAsync``.
* Warp specialization: ``num_mma_warps`` MMA warps + 1 DMA warp.
* Swizzled smem layouts via ``sm90_utils.get_smem_layout_atom``.
* Hardware FP4 decode: ``cvt.rn.f16x2.e2m1x2`` PTX (``fp4_decode_4bytes``).
* LdMatrix.x4 sA/sB → reg, StMatrix.x2 reg → sC.
* Persistent CTA scheduler over (m_tile, n_tile).

Enable via ``B12X_GEMM_W4A16_USE_CUTE=1``; falls back to Triton on
compile / runtime errors (see ``micro.py``).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.hopper_helpers as sm90_utils
    from cutlass import BFloat16, Float32, Int32, Uint8
    from cutlass.cute.nvgpu import cpasync
    from cutlass.cutlass_dsl import Int64, Uint32, T, dsl_user_op
    from cutlass._mlir.dialects import llvm

    from b12x.cute.utils import current_cuda_stream, make_ptr
    from b12x.cute.fp4 import (
        cvt_e4m3_to_f32_via_f16,
        f16x2_to_f32x2,
        fp4_decode_4bytes,
        ld_global_nc_u32,
    )

    _CUTE_AVAILABLE = True
except ImportError:
    _CUTE_AVAILABLE = False


_SF_VEC_SIZE = 16


def _cute_backend_enabled() -> bool:
    return os.environ.get("B12X_GEMM_W4A16_USE_CUTE") == "1" and _CUTE_AVAILABLE


if _CUTE_AVAILABLE:

    @dsl_user_op
    def _ld_global_nc_u8(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
                "ld.global.nc.u8 $0, [$1];",
                "=r,l",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    class _DenseGemmW4A16CuteJit:
        """JIT-compiled forked dense W4A16 kernel.

        Per-call attributes (``a_dtype``, ``b_dtype``, layouts) are set on
        ``__call__`` from the input tensors; MMA + smem layouts are
        derived in ``_setup_attributes`` inside the ``@cute.jit`` scope.
        """

        def __init__(
            self,
            m: int,
            n: int,
            k: int,
            mma_tiler_mn: Tuple[int, int] = (32, 64),
            tile_k: Optional[int] = None,
            ab_stage: int = 2,
            n_per_cta: int = 1,
            sf_vec_size: int = _SF_VEC_SIZE,
        ):
            self.m = m
            self.n = n
            self.k = k
            self.sf_vec_size = sf_vec_size
            self.acc_dtype = Float32
            # Allow tile_k != mma_tiler_mn[1] so the K-tile can be tuned
            # independently of N.  Default keeps the legacy behavior
            # (tile_K = tile_N).
            if tile_k is None:
                tile_k = mma_tiler_mn[1]
            self._ab_stage_cfg = ab_stage
            # n_per_cta: how many consecutive N-tiles each CTA processes
            # in one work-pair, sharing the A K-tile loads.  >1 reduces
            # A traffic at the cost of more registers for the extra
            # accumulators.  Must divide num_n_tiles.
            self.n_per_cta = n_per_cta
            self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
            self.sa_tile_shape_mk = (mma_tiler_mn[0], tile_k)
            self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
            self.cluster_shape_mnk = (1, 1, 1)
            self.cluster_shape_mn = (1, 1)
            self.num_mma_warps = 4
            self.tma_load_warp_id = self.num_mma_warps
            self.num_threads_per_warp = 32
            self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
            self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
            self.buffer_align_bytes = 1024
            self.load_register_requirement = 32
            self.mma_register_requirement = 232
            self.epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=self.num_mma_warps * self.num_threads_per_warp,
            )

        @staticmethod
        def _make_tma_atom_and_tensor(tensor, smem_layout_staged, smem_tile):
            op = cpasync.CopyBulkTensorTileG2SOp()
            smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
            return cpasync.make_tiled_tma_atom(
                op, tensor, smem_layout, smem_tile, num_multicast=1,
            )

        # --- smem layout builders (lifted from MoE static, unchanged) ---

        def _make_a_smem_layout(self, ab_stage: int):
            a_is_k_major = self.a_layout.is_k_major_a()
            a_major_mode_size = self.sa_tile_shape_mk[1 if a_is_k_major else 0]
            a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(
                    self.a_layout, self.a_dtype, a_major_mode_size,
                ),
                self.a_dtype,
            )
            return cute.tile_to_shape(
                a_smem_layout_atom,
                cute.append(self.sa_tile_shape_mk, ab_stage),
                order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
            )

        def _make_b_smem_layout(self, ab_stage: int):
            b_smem_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
            b_is_k_major = self.b_layout.is_k_major_b()
            b_major_mode_size = self.tile_shape_mnk[2 if b_is_k_major else 1]
            b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(
                    self.b_layout, self.b_dtype, b_major_mode_size,
                ),
                self.b_dtype,
            )
            return cute.tile_to_shape(
                b_smem_layout_atom,
                cute.append(b_smem_shape, ab_stage),
                order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
            )

        def _make_staged_layouts(self, ab_stage: int):
            a_smem_staged = self._make_a_smem_layout(ab_stage)
            b_smem_staged = self._make_b_smem_layout(ab_stage)
            epi_smem_staged = sm90_utils.make_smem_layout_epi(
                BFloat16, self.c_layout, self.epi_tile, self.epi_stage,
            )
            return a_smem_staged, b_smem_staged, epi_smem_staged

        def _setup_attributes(self):
            self.mma_inst_mnk = (16, 8, 16)
            mma_op = cute.nvgpu.warp.MmaF16BF16Op(
                self.a_dtype, self.acc_dtype, self.mma_inst_mnk,
            )
            atom_layout = cute.make_layout((2, 2, 1))
            permutation_mnk = (
                2 * self.mma_inst_mnk[0],
                2 * self.mma_inst_mnk[1] * 2,
                self.mma_inst_mnk[2],
            )
            self.tiled_mma = cute.make_tiled_mma(
                mma_op, atom_layout, permutation_mnk=permutation_mnk,
            )
            self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
            self.num_k_blocks = self.tile_shape_mnk[2] // self.mma_inst_mnk[2]
            epi_stage_max = (self.tile_shape_mnk[1] // self.epi_tile[1]) * (
                self.tile_shape_mnk[0] // self.epi_tile[0]
            )
            self.epi_stage = min(epi_stage_max, 4)
            # AB pipeline depth — tunable per backend instance.
            self.ab_stage = self._ab_stage_cfg
            (
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.epi_smem_layout_staged,
            ) = self._make_staged_layouts(self.ab_stage)

        # --- helpers for FP4 weight staging (lifted from MoE static) ---

        @cute.jit
        def _swizzled_e4m3_offset(
            self, row: Int32, sf_block: Int32, sf_cols: Int32,
        ) -> Int64:
            row_rb = row >> Int32(7)
            mode_a = (row >> Int32(5)) & Int32(3)
            mode_32 = row & Int32(31)
            cb_idx = sf_block >> Int32(2)
            mode_c = sf_block & Int32(3)
            return (
                Int64(row_rb) * Int64(sf_cols * Int32(128))
                + Int64(cb_idx) * Int64(512)
                + Int64(mode_32) * Int64(16)
                + Int64(mode_a) * Int64(4)
                + Int64(mode_c)
            )

        @cute.jit
        def _stage_b_fp4_tile(
            self,
            packed_w: cute.Tensor,    # W view as Uint8
            sfb_ptr: cute.Pointer,    # SF as Uint8 ptr
            sB: cute.Tensor,
            stage_idx: Int32,
            n_tile_idx: Int32,
            k_tile_idx: Int32,
            weight_rows: Int32,       # N
            weight_cols: Int32,       # K (logical, unpacked)
            sf_cols: Int32,           # padded ceil(K/16, 4)
            copy_start: Int32,
            copy_stride: Int32,
        ):
            """Cooperative FP4 → bf16 stage from gmem W into sB[stage].

            Each thread loads 8 packed bytes (= 16 FP4 elems) via
            ``ld.global.nc.u32`` ×2, decodes with hardware cvt, multiplies
            by per-block FP8 scale, casts to bf16, stores to swizzled sB.

            Pattern verbatim from b12x/moe/fused/w4a16/static.py:457+
            with the expert offset stripped.
            """
            w_base = packed_w.iterator.toint()
            sf_base = sfb_ptr.toint()
            packed_cols = weight_cols // Int32(2)
            tile_n = Int32(self.tile_shape_mnk[1])
            tile_k = Int32(self.tile_shape_mnk[2])
            blocks_per_row = tile_k // Int32(self.sf_vec_size)
            total_blocks = tile_n * blocks_per_row
            copy_idx = copy_start
            while copy_idx < total_blocks:
                local_n = copy_idx // blocks_per_row
                local_sf_block = copy_idx - local_n * blocks_per_row
                local_k = local_sf_block * Int32(self.sf_vec_size)
                global_n = n_tile_idx * tile_n + local_n
                global_k = k_tile_idx * tile_k + local_k
                packed_offset = (
                    Int64(global_n) * Int64(packed_cols)
                    + Int64(global_k // Int32(2))
                )
                scale_offset = self._swizzled_e4m3_offset(
                    global_n, global_k // Int32(self.sf_vec_size), sf_cols,
                )
                scale_byte = _ld_global_nc_u8(sf_base + scale_offset)
                scale = cvt_e4m3_to_f32_via_f16(scale_byte)
                q_word0 = ld_global_nc_u32(w_base + packed_offset)
                q_word1 = ld_global_nc_u32(w_base + packed_offset + Int64(4))
                d0, d1, d2, d3 = fp4_decode_4bytes(q_word0)
                f0, f1 = f16x2_to_f32x2(d0)
                sB[local_n, local_k, stage_idx] = BFloat16(f0 * scale)
                sB[local_n, local_k + Int32(1), stage_idx] = BFloat16(f1 * scale)
                f0, f1 = f16x2_to_f32x2(d1)
                sB[local_n, local_k + Int32(2), stage_idx] = BFloat16(f0 * scale)
                sB[local_n, local_k + Int32(3), stage_idx] = BFloat16(f1 * scale)
                f0, f1 = f16x2_to_f32x2(d2)
                sB[local_n, local_k + Int32(4), stage_idx] = BFloat16(f0 * scale)
                sB[local_n, local_k + Int32(5), stage_idx] = BFloat16(f1 * scale)
                f0, f1 = f16x2_to_f32x2(d3)
                sB[local_n, local_k + Int32(6), stage_idx] = BFloat16(f0 * scale)
                sB[local_n, local_k + Int32(7), stage_idx] = BFloat16(f1 * scale)
                d0, d1, d2, d3 = fp4_decode_4bytes(q_word1)
                f0, f1 = f16x2_to_f32x2(d0)
                sB[local_n, local_k + Int32(8), stage_idx] = BFloat16(f0 * scale)
                sB[local_n, local_k + Int32(9), stage_idx] = BFloat16(f1 * scale)
                f0, f1 = f16x2_to_f32x2(d1)
                sB[local_n, local_k + Int32(10), stage_idx] = BFloat16(f0 * scale)
                sB[local_n, local_k + Int32(11), stage_idx] = BFloat16(f1 * scale)
                f0, f1 = f16x2_to_f32x2(d2)
                sB[local_n, local_k + Int32(12), stage_idx] = BFloat16(f0 * scale)
                sB[local_n, local_k + Int32(13), stage_idx] = BFloat16(f1 * scale)
                f0, f1 = f16x2_to_f32x2(d3)
                sB[local_n, local_k + Int32(14), stage_idx] = BFloat16(f0 * scale)
                sB[local_n, local_k + Int32(15), stage_idx] = BFloat16(f1 * scale)
                copy_idx += copy_stride

        # --- kernel entry: GEMM tile loop ---

        @cute.kernel
        def kernel(
            self,
            a_input: cute.Tensor,       # [M_pad, K] bf16 (padded to tile_M)
            tma_a: cute.CopyAtom, mA: cute.Tensor,
            b_w: cute.Tensor,           # [N, K/2] uint8 packed FP4
            sfb_ptr: cute.Pointer,
            alpha: cute.Tensor,         # [1] fp32
            c_out: cute.Tensor,         # [M, N] bf16
            tiled_mma: cute.TiledMma,
            cta_layout_mnk: cute.Layout,
            a_smem_staged: cute.ComposedLayout,
            b_smem_staged: cute.ComposedLayout,
            epi_smem_staged: cute.ComposedLayout,
            num_m_tiles: Int32,
            num_n_tiles: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            _, _, bidz = cute.arch.block_idx()
            _, _, gdim_z = cute.arch.grid_dim()
            warp_idx = cute.arch.warp_idx()
            warp_idx = cute.arch.make_warp_uniform(warp_idx)

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_a)

            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

            a_smem_one = cute.slice_(a_smem_staged, (None, None, 0))
            tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_one)

            smem = cutlass.utils.SmemAllocator()

            @cute.struct
            class Storage:
                pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
                sA: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                    self.buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[BFloat16, cute.cosize(epi_smem_staged)],
                    self.buffer_align_bytes,
                ]

            storage = smem.allocate(Storage)

            prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            cons_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_warps,
            )
            cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
            ml_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=self.ab_stage,
                producer_group=prod_group,
                consumer_group=cons_group,
                tx_count=tma_copy_bytes,
                barrier_storage=storage.pipeline_array.data_ptr(),
                cta_layout_vmnk=cta_layout_vmnk,
            )

            cute.arch.sync_threads()

            sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
            sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
            sC = storage.sC.get_tensor(epi_smem_staged.outer, swizzle=epi_smem_staged.inner)

            gA = cute.local_tile(mA, self.sa_tile_shape_mk, (None, None))
            thr_mma = tiled_mma.get_slice(tidx)
            a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
            a_cta_crd = cluster_coord[1]
            tAsA, tAgA = cpasync.tma_partition(
                tma_a, a_cta_crd, a_cta_layout,
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2),
            )

            tCsA = thr_mma.partition_A(sA)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCsB = thr_mma.partition_B(sB)
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

            tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
            epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
            sub_shape = tCsC_for_shape.shape[:3]
            acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
            # One accumulator per N-tile in the work-pair (n_per_cta accs).
            accs = []
            for _ in cutlass.range_constexpr(self.n_per_cta):
                accs.append(cute.make_rmem_tensor(acc_shape, self.acc_dtype))

            k_tile_cnt = cute.size(gA, mode=[3])
            k_logical = Int32(self.k)
            sf_cols = ((k_logical // Int32(self.sf_vec_size)) + Int32(3)) // Int32(4) * Int32(4)

            alpha_value = Float32(alpha[Int32(0)])

            # Persistent scheduler: each work-pair covers n_per_cta
            # consecutive N-tiles for the same M-tile.  Total work-pairs
            # = num_m_tiles * num_n_tiles / n_per_cta.  Caller ensures
            # num_n_tiles is divisible by n_per_cta.
            n_per_cta_c = cutlass.const_expr(self.n_per_cta)
            total_pairs = num_m_tiles * (num_n_tiles // Int32(n_per_cta_c))
            work_idx = Int32(bidz)

            # ===================================================================
            # MMA warps (warps 0..num_mma_warps-1)
            # ===================================================================
            if warp_idx < self.num_mma_warps:
                # setmaxregister_{increase,decrease} only exist on
                # cutlass-dsl >= 4.4; on Spark (4.3.4) we skip them.
                # The kernel still runs correctly without dedicated
                # register budgets; only perf may suffer slightly.
                if cutlass.const_expr(hasattr(cute.arch, "setmaxregister_increase")):
                    cute.arch.setmaxregister_increase(self.mma_register_requirement)
                num_k_blocks = cute.size(tCrA, mode=[2])

                atom_ld_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                    self.a_dtype,
                )
                atom_ld_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                    self.b_dtype,
                )
                smem_copy_A = cute.make_tiled_copy_A(atom_ld_A, tiled_mma)
                smem_copy_B = cute.make_tiled_copy_B(atom_ld_B, tiled_mma)
                thr_ld_A = smem_copy_A.get_slice(tidx)
                thr_ld_B = smem_copy_B.get_slice(tidx)
                csA = thr_ld_A.partition_S(sA)
                crA = thr_ld_A.retile(tCrA)
                csB = thr_ld_B.partition_S(sB)
                crB = thr_ld_B.retile(tCrB)

                _is_m_major = self.c_layout.is_m_major_c()
                copy_atom_r2s = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2), BFloat16,
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2), BFloat16,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                # rD_out is laid out like one accumulator's retile —
                # reused across n_per_cta epilogue passes.
                tRS_rAcc0 = tiled_copy_r2s.retile(accs[0])
                rD_out = cute.make_fragment_like(tRS_rAcc0, BFloat16)

                cons_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage,
                )

                while work_idx < total_pairs:
                    # Each work-pair = n_per_cta consecutive N-tiles
                    # for the same M-tile.  DMA warp issues ONE round of
                    # A K-tile TMA loads; MMA warps reuse rA across the
                    # n_per_cta inner iterations.
                    pairs_per_m = num_n_tiles // Int32(n_per_cta_c)
                    m_tile = work_idx // pairs_per_m
                    pair_in_m = work_idx - m_tile * pairs_per_m
                    n_tile_base = pair_in_m * Int32(n_per_cta_c)

                    for nn in cutlass.range_constexpr(n_per_cta_c):
                        accs[nn].fill(0.0)
                    cons_state.reset_count()

                    for k_tile in range(0, k_tile_cnt, 1):
                        peek = ml_pipeline.consumer_try_wait(cons_state)
                        ml_pipeline.consumer_wait(cons_state, peek)
                        csA_p = csA[None, None, None, cons_state.index]

                        if cutlass.const_expr(n_per_cta_c == 1):
                            # n_per_cta=1: keep the original progressive
                            # rA/rB k_block load pattern so smem→reg loads
                            # overlap with MMA.
                            self._stage_b_fp4_tile(
                                b_w, sfb_ptr, sB, cons_state.index,
                                n_tile_base, k_tile,
                                Int32(self.n), Int32(self.k), sf_cols,
                                Int32(tidx),
                                Int32(self.num_mma_warps * self.num_threads_per_warp),
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            self.epilog_sync_barrier.arrive_and_wait()
                            csB_p = csB[None, None, None, cons_state.index]
                            cute.copy(smem_copy_A, csA_p[None, None, 0], crA[None, None, 0])
                            cute.copy(smem_copy_B, csB_p[None, None, 0], crB[None, None, 0])
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                                cute.gemm(
                                    tiled_mma, accs[0],
                                    tCrA[None, None, k_block_idx],
                                    tCrB[None, None, k_block_idx],
                                    accs[0],
                                )
                                if k_next > 0:
                                    cute.copy(smem_copy_A, csA_p[None, None, k_next],
                                              crA[None, None, k_next])
                                    cute.copy(smem_copy_B, csB_p[None, None, k_next],
                                              crB[None, None, k_next])
                        else:
                            # n_per_cta > 1: pre-load all k_blocks of rA
                            # once, then reuse across n_per_cta inner MMA
                            # passes (each with its own rB).
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                cute.copy(smem_copy_A, csA_p[None, None, k_block_idx],
                                          crA[None, None, k_block_idx])
                            for nn in cutlass.range_constexpr(n_per_cta_c):
                                # Between nn iterations, wait for the
                                # prior nn's incremental LdMatrix reads
                                # of sB to finish before restaging sB.
                                # Otherwise lagging warps can read sB
                                # while fast warps overwrite it for the
                                # next nn (race shows up as silent
                                # accuracy failure at larger K).
                                if cutlass.const_expr(nn > 0):
                                    self.epilog_sync_barrier.arrive_and_wait()
                                n_tile_inner = n_tile_base + Int32(nn)
                                self._stage_b_fp4_tile(
                                    b_w, sfb_ptr, sB, cons_state.index,
                                    n_tile_inner, k_tile,
                                    Int32(self.n), Int32(self.k), sf_cols,
                                    Int32(tidx),
                                    Int32(self.num_mma_warps * self.num_threads_per_warp),
                                )
                                cute.arch.fence_proxy("async.shared", space="cta")
                                self.epilog_sync_barrier.arrive_and_wait()
                                csB_p = csB[None, None, None, cons_state.index]
                                cute.copy(smem_copy_B, csB_p[None, None, 0],
                                          crB[None, None, 0])
                                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                    k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                                    cute.gemm(
                                        tiled_mma, accs[nn],
                                        tCrA[None, None, k_block_idx],
                                        tCrB[None, None, k_block_idx],
                                        accs[nn],
                                    )
                                    if k_next > 0:
                                        cute.copy(smem_copy_B, csB_p[None, None, k_next],
                                                  crB[None, None, k_next])

                        ml_pipeline.consumer_release(cons_state)
                        cons_state.advance()

                    # Epilogue: alpha-scale + cast + store each acc.
                    tile_m_base = m_tile * Int32(self.tile_shape_mnk[0])
                    m_valid = Int32(self.m) - tile_m_base
                    if m_valid > Int32(self.tile_shape_mnk[0]):
                        m_valid = Int32(self.tile_shape_mnk[0])
                    if m_valid < Int32(0):
                        m_valid = Int32(0)
                    for nn in cutlass.range_constexpr(n_per_cta_c):
                        n_tile_inner = n_tile_base + Int32(nn)
                        tRS_rAcc_nn = tiled_copy_r2s.retile(accs[nn])
                        acc_vec = tRS_rAcc_nn.load()
                        acc_vec = acc_vec * alpha_value
                        rD_out.store(acc_vec.to(BFloat16))
                        cute.copy(tiled_copy_r2s, rD_out,
                                  tRS_sD[(None, None, None, Int32(0))])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        tile_n_base = n_tile_inner * Int32(self.tile_shape_mnk[1])
                        n_valid = Int32(self.n) - tile_n_base
                        if n_valid > Int32(self.tile_shape_mnk[1]):
                            n_valid = Int32(self.tile_shape_mnk[1])
                        if n_valid < Int32(0):
                            n_valid = Int32(0)
                        copy_idx = Int32(tidx)
                        copy_total = m_valid * n_valid
                        while copy_idx < copy_total:
                            local_m = copy_idx // n_valid
                            local_n = copy_idx - local_m * n_valid
                            c_out[tile_m_base + local_m, tile_n_base + local_n] = sC[
                                local_m, local_n, Int32(0)
                            ]
                            copy_idx += Int32(self.num_mma_warps * self.num_threads_per_warp)
                        # Sync between epilogue passes only when there's
                        # a next one — protects sC reuse across nn.
                        if cutlass.const_expr(n_per_cta_c > 1):
                            self.epilog_sync_barrier.arrive_and_wait()

                    work_idx += Int32(gdim_z)

            # ===================================================================
            # DMA warp (warps == num_mma_warps): drives TMA loads for A.
            # With n_per_cta > 1, each work-pair = n_per_cta N-tiles for
            # the same M-tile, but the DMA warp still issues k_tile_cnt
            # TMA loads (one per K-tile) — the MMA warps reuse rA across
            # the inner n_per_cta MMA passes.
            # ===================================================================
            else:
                if cutlass.const_expr(hasattr(cute.arch, "setmaxregister_decrease")):
                    cute.arch.setmaxregister_decrease(self.load_register_requirement)
                prod_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage,
                )

                while work_idx < total_pairs:
                    pairs_per_m = num_n_tiles // Int32(n_per_cta_c)
                    m_tile = work_idx // pairs_per_m

                    for k_tile in range(0, k_tile_cnt, 1):
                        ml_pipeline.producer_acquire(prod_state)
                        cute.copy(
                            tma_a,
                            tAgA[(None, m_tile, k_tile)],
                            tAsA[(None, prod_state.index)],
                            tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                        )
                        ml_pipeline.producer_commit(prod_state)
                        prod_state.advance()

                    work_idx += Int32(gdim_z)

        @cute.jit
        def __call__(
            self,
            x_ptr: cute.Pointer,
            w_ptr: cute.Pointer,
            sf_ptr: cute.Pointer,
            alpha_ptr: cute.Pointer,
            out_ptr: cute.Pointer,
            max_active_clusters: Int32,
            stream,
        ):
            # Build tensors. A is padded along M to tile_M (caller's
            # responsibility) so we can use a single TMA descriptor.
            tile_m = self.tile_shape_mnk[0]
            tile_n = self.tile_shape_mnk[1]
            m_padded = ((self.m + tile_m - 1) // tile_m) * tile_m

            self.a_dtype = BFloat16
            self.b_dtype = BFloat16
            x = cute.make_tensor(
                x_ptr,
                layout=cute.make_ordered_layout((m_padded, self.k), order=(1, 0)),
            )
            w = cute.make_tensor(
                w_ptr,
                layout=cute.make_ordered_layout((self.n, self.k // 2), order=(1, 0)),
            )
            out = cute.make_tensor(
                out_ptr,
                layout=cute.make_ordered_layout((self.m, self.n), order=(1, 0)),
            )
            self.a_layout = utils.LayoutEnum.from_tensor(x)
            self.b_layout = utils.LayoutEnum.from_tensor(w)
            self.c_layout = utils.LayoutEnum.ROW_MAJOR
            self._setup_attributes()

            alpha_t = cute.make_tensor(alpha_ptr, layout=cute.make_layout((1,)))

            tma_a, gA = self._make_tma_atom_and_tensor(
                x, self.a_smem_layout_staged, self.sa_tile_shape_mk,
            )

            num_m_tiles = (self.m + tile_m - 1) // tile_m
            num_n_tiles = (self.n + tile_n - 1) // tile_n
            total_tiles = num_m_tiles * num_n_tiles

            grid = (*self.cluster_shape_mn, min(total_tiles, max_active_clusters))
            self.kernel(
                x, tma_a, gA, w, sf_ptr, alpha_t, out,
                self.tiled_mma, self.cta_layout_mnk,
                self.a_smem_layout_staged, self.b_smem_layout_staged,
                self.epi_smem_layout_staged,
                Int32(num_m_tiles), Int32(num_n_tiles),
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
            )

else:

    class _DenseGemmW4A16CuteJit:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("cutlass.cute (DSL) not available")


class DenseGemmW4A16CuteDenseKernel:
    """W4A16 dense GEMM cute backend, v4 (forked from MoE static).

    Consumes the *swizzled* FP8 block-scales directly (no host-side
    unswizzle round-trip).  Supports M ∈ {1..tile_M}, N % tile_N == 0,
    K % tile_K == 0; tile_M=32, tile_N=tile_K=64 by default.
    """

    _TILE_M = 32
    _TILE_N = 64
    _TILE_K = 64
    _AB_STAGE = 2
    _N_PER_CTA = 1

    @classmethod
    def is_supported(cls, m: int, k: int, n: int) -> bool:
        # v4 was tuned for decode (M ≤ tile_M = 32) but the kernel
        # itself loops over m_tiles, so any positive M runs correctly
        # — just suboptimally at large M (where v5 prefill takes over).
        # ``micro.py`` picks v4 vs v5 by M; both should accept the
        # full N/K envelope.
        if m <= 0 or k <= 0 or n <= 0:
            return False
        if n % cls._TILE_N != 0:
            return False
        if k % cls._TILE_K != 0:
            return False
        return True

    def is_supported_instance(self, m: int, k: int, n: int) -> bool:
        if m <= 0 or k <= 0 or n <= 0:
            return False
        if n % self._tile_n != 0:
            return False
        if k % self._tile_k != 0:
            return False
        # n_per_cta must divide num_n_tiles.
        num_n_tiles = n // self._tile_n
        if num_n_tiles % self._n_per_cta != 0:
            return False
        return True

    def __init__(
        self,
        tile_m: int = None,
        tile_n: int = None,
        tile_k: int = None,
        ab_stage: int = None,
        n_per_cta: int = None,
    ) -> None:
        self._tile_m = tile_m if tile_m is not None else self._TILE_M
        self._tile_n = tile_n if tile_n is not None else self._TILE_N
        self._tile_k = tile_k if tile_k is not None else self._TILE_K
        self._ab_stage = ab_stage if ab_stage is not None else self._AB_STAGE
        self._n_per_cta = n_per_cta if n_per_cta is not None else self._N_PER_CTA
        # Cache the cute.compile output per (m, n, k).  Pre-compiling once
        # avoids the @cute.jit-traced-per-call overhead (~140ms each) that
        # bites the TMA-pipelined v4 kernel because re-emitting the IR
        # per call is expensive.  Pattern follows
        # b12x/integration/tp_moe.py:2398.
        self._compile_cache: dict = {}

    def _get_compiled(self, m: int, n: int, k: int):
        if not _CUTE_AVAILABLE:
            raise NotImplementedError("cutlass.cute (DSL) not available")
        key = (m, n, k, self._tile_m, self._tile_n, self._tile_k,
               self._ab_stage, self._n_per_cta)
        if key not in self._compile_cache:
            jit_instance = _DenseGemmW4A16CuteJit(
                m=m, n=n, k=k,
                mma_tiler_mn=(self._tile_m, self._tile_n),
                tile_k=self._tile_k,
                ab_stage=self._ab_stage,
                n_per_cta=self._n_per_cta,
            )

            def _dummy(dt):
                return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)

            self._compile_cache[key] = cute.compile(
                jit_instance,
                _dummy(BFloat16),       # x_ptr
                _dummy(cutlass.Uint8),  # w_ptr
                _dummy(cutlass.Uint8),  # sf_ptr (FP8 e4m3 viewed as u8)
                _dummy(Float32),        # alpha_ptr
                _dummy(BFloat16),       # out_ptr
                Int32(128),             # max_active_clusters
                current_cuda_stream(),
            )
        return self._compile_cache[key]

    def __call__(
        self,
        x: torch.Tensor,
        w_fp4: torch.Tensor,
        w_blockscale_swizzled_u8: torch.Tensor,
        w_alpha: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not _CUTE_AVAILABLE:
            raise NotImplementedError("cutlass.cute (DSL) not available")
        m, k = x.shape
        n = w_fp4.shape[0]
        if out is None:
            out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)

        # Pad A along M to tile_M if needed (one-time per call) so the
        # TMA descriptor covers full tiles. Zero pad is safe — the kernel
        # bounds-checks on M in the epilogue store.
        if m < self._tile_m:
            x_pad = torch.zeros(self._tile_m, k, dtype=x.dtype, device=x.device)
            x_pad[:m].copy_(x)
            x_used = x_pad
        else:
            x_used = x

        compiled = self._get_compiled(m, n, k)
        stream = current_cuda_stream()
        # Persistent scheduler: scale with the device's SM count.
        # ~3x SM count over-subscribes to keep pipelines full.
        # Override via B12X_GEMM_W4A16_MAX_ACTIVE for sweeps.
        max_active_env = os.environ.get("B12X_GEMM_W4A16_MAX_ACTIVE")
        if max_active_env is not None:
            max_active = int(max_active_env)
        else:
            sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
            max_active = min(sm_count * 3, 256)
        compiled(
            make_ptr(BFloat16, x_used.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint8, w_fp4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint8, w_blockscale_swizzled_u8.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Float32, w_alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            max_active,
            stream,
        )
        return out


__all__ = ["DenseGemmW4A16CuteDenseKernel", "_cute_backend_enabled"]
