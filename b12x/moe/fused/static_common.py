from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils

from b12x.gemm.dense import (
    DenseGemmKernel,
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)


_COMPACT_STATIC_TILE_M = 128


class MoEStaticKernelBase:
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        exact_mma_m_tiles: bool = False,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        activation: str = "silu",
    ):
        if activation not in {"silu", "relu2"}:
            raise ValueError(f"unsupported activation {activation!r}")
        self._dense_cls = DenseGemmKernel
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.exact_mma_m_tiles = exact_mma_m_tiles
        self.input_scales_are_reciprocal = input_scales_are_reciprocal
        self.fast_math = fast_math
        self.activation = activation
        self.is_gated = activation == "silu"
        tile_k = sf_vec_size * 8
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        self.sa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sa_tiles_per_block = self.sa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sfa_tiles_per_block = self.sfa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfb_tile_shape_nk = (max(128, mma_tiler_mn[1]), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // mma_tiler_mn[1]
        self.output_tile_count_n = output_tile_count_n
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.occupancy = 1
        self.num_mma_warps = 4
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.pass_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )
        self.load_register_requirement = 32
        self.mma_register_requirement = 232

    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFA(self, sfa_tensor, tiled_mma)

    def _thrfrg_SFB(self, sfb_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFB(self, sfb_tensor, tiled_mma)

    def _get_layoutSFA_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFA_TV(self, tiled_mma)

    def _get_layoutSFB_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFB_TV(self, tiled_mma)

    def _make_a_smem_layout(self, ab_stage: int):
        import cutlass.utils.hopper_helpers as sm90_utils

        a_is_k_major = self.a_layout.is_k_major_a()
        a_major_mode_size = self.sa_tile_shape_mk[1 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.a_layout,
                self.a_dtype,
                a_major_mode_size,
            ),
            self.a_dtype,
        )
        return cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(self.sa_tile_shape_mk, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

    def _make_staged_layouts(self, ab_stage: int):
        (
            _,
            b_smem_staged,
            sfa_smem_staged,
            sfb_smem_staged,
            epi_smem_staged,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            ab_stage,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )
        a_smem_staged = self._make_a_smem_layout(ab_stage)
        return (
            a_smem_staged,
            b_smem_staged,
            sfa_smem_staged,
            sfb_smem_staged,
            epi_smem_staged,
        )

    def _shared_storage_size_bytes(
        self,
        a_smem_staged,
        b_smem_staged,
        sfa_smem_staged,
        sfb_smem_staged,
        epi_smem_staged,
    ) -> int:
        def _align_up(value: int, align: int) -> int:
            return ((value + align - 1) // align) * align

        pipeline_count = 3 if self.is_gated else 2
        offset = (
            3 * 4
            + pipeline_count * (self.ab_stage * 2 * 8)
            + _COMPACT_STATIC_TILE_M * 4
            + _COMPACT_STATIC_TILE_M * 4
        )
        buffers = [
            cute.size_in_bytes(self.a_dtype, a_smem_staged),
            cute.size_in_bytes(self.b_dtype, b_smem_staged),
            cute.size_in_bytes(self.sf_dtype, sfa_smem_staged),
            cute.size_in_bytes(self.sf_dtype, sfb_smem_staged),
            cute.size_in_bytes(cutlass.BFloat16, epi_smem_staged),
        ]
        if self.is_gated:
            buffers.insert(2, cute.size_in_bytes(self.b_dtype, b_smem_staged))
            buffers.insert(5, cute.size_in_bytes(self.sf_dtype, sfb_smem_staged))
        offset = _align_up(offset, self.buffer_align_bytes)
        for idx, size in enumerate(buffers):
            offset += size
            if idx + 1 != len(buffers):
                offset = _align_up(offset, self.buffer_align_bytes)
        return offset

    def _setup_attributes(self):
        import cutlass.utils.blackwell_helpers as sm120_utils

        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        atom_layout = cute.make_layout((2, 2, 1))
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk,
            self.sf_vec_size,
            False,
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        self.mma_atom = cute.make_mma_atom(mma_op)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        m_tile_divisor = 2 if self.exact_mma_m_tiles else 4
        self.num_m_tiles = self.tile_shape_mnk[0] // (16 * m_tile_divisor)
        self.num_n_tiles = self.tile_shape_mnk[1] // (8 * 2)
        self.num_k_blocks = self.tile_shape_mnk[2] // 64

        sfa_smem = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        sfb_smem = sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        self.ab_stage, self.epi_stage = self._dense_cls._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            sfa_smem,
            sfb_smem,
            self.epi_tile,
            cutlass.BFloat16,
            self.smem_capacity,
            self.occupancy,
        )
        self.ab_stage = max(1, min(self.ab_stage, 2))
        while self.ab_stage > 1 and 32 % self.ab_stage != 0:
            self.ab_stage -= 1
        self.epi_stage = 1
        while True:
            (
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.sfa_smem_layout_staged,
                self.sfb_smem_layout_staged,
                self.epi_smem_layout_staged,
            ) = self._make_staged_layouts(self.ab_stage)
            if (
                self._shared_storage_size_bytes(
                    self.a_smem_layout_staged,
                    self.b_smem_layout_staged,
                    self.sfa_smem_layout_staged,
                    self.sfb_smem_layout_staged,
                    self.epi_smem_layout_staged,
                ) <= self.smem_capacity
                or self.ab_stage <= 1
            ):
                break
            self.ab_stage -= 1
            while self.ab_stage > 1 and 32 % self.ab_stage != 0:
                self.ab_stage -= 1
