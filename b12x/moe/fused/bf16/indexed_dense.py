"""BF16 dense GEMM variant that reads full expert weights by expert id."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch

from b12x.attention.utils import fmax as fmax_f32
from b12x.moe.fused.bf16.static import (
    DenseGemmKernel,
    _consumer_dense_pass,
    _producer_load_dense_pass,
    _run_cached_host_launcher,
    _tensor_meta_key,
    _to_dense_kernel_tensor,
)


def run_dense_bf16_expert_ids(
    kernel: "ExpertIndexedDenseGemmKernel",
    a: torch.Tensor,
    b: torch.Tensor,
    expert_ids: torch.Tensor,
    c: torch.Tensor,
    max_active_clusters: int,
    stream: cuda.CUstream,
) -> None:
    if expert_ids.ndim != 1:
        raise ValueError(f"expert_ids must be rank-1, got {tuple(expert_ids.shape)}")
    if expert_ids.dtype != torch.int32 or expert_ids.stride(0) != 1:
        expert_ids = expert_ids.to(torch.int32).contiguous()
    args = (
        _to_dense_kernel_tensor(a),
        _to_dense_kernel_tensor(b),
        _to_dense_kernel_tensor(expert_ids, cutlass.Int32, assumed_align=4),
        _to_dense_kernel_tensor(c),
        max_active_clusters,
        stream,
    )
    cache_key = (
        tuple(kernel.tile_shape_mnk),
        _tensor_meta_key(a),
        _tensor_meta_key(b),
        _tensor_meta_key(expert_ids),
        _tensor_meta_key(c),
        max_active_clusters,
    )
    _run_cached_host_launcher(kernel, cache_key, args)


class ExpertIndexedDenseGemmKernel(DenseGemmKernel):
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        expert_ids: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.expert_ids_dtype = expert_ids.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 A, got {self.a_dtype}")
        if cutlass.const_expr(self.b_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 B, got {self.b_dtype}")
        if cutlass.const_expr(self.expert_ids_dtype != cutlass.Int32):
            raise TypeError(f"expected Int32 expert_ids, got {self.expert_ids_dtype}")
        if cutlass.const_expr(self.c_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 C, got {self.c_dtype}")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._get_or_make_tma_load(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_b, tma_tensor_b = self._get_or_make_tma_load(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_c, tma_tensor_c = self._get_or_make_tma_store(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )
        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            expert_ids,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        expert_ids: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        consumer_arrive_cnt = self.num_mma_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )
        pipeline.sync(barrier_id=1)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(tidx)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_mnk[1],
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            cluster_coord_mnk[0],
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCgC = thr_mma.partition_C(gC_mnl)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        if warp_idx < self.num_mma_warps:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                weight_expert_idx = expert_ids[tile_coord_mnl[2]].to(cutlass.Int32)
                if weight_expert_idx >= cutlass.Int32(0):
                    gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                    accumulators.fill(0.0)
                    mainloop_consumer_state.reset_count()

                    peek_ab_full_status = cutlass.Boolean(1)
                    if mainloop_consumer_state.count < k_tile_cnt:
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            mainloop_consumer_state
                        )
                    mainloop_pipeline.consumer_wait(
                        mainloop_consumer_state, peek_ab_full_status
                    )
                    tCsA_p = tCsA_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]
                    tCsB_p = tCsB_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]
                    cute.copy(
                        smem_tiled_copy_A,
                        tCsA_p[None, None, 0],
                        tCrA_copy_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_B,
                        tCsB_p[None, None, 0],
                        tCrB_copy_view[None, None, 0],
                    )

                    for _k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                        del _k_tile
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_next = (
                                0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                            )
                            if k_block_idx == num_k_blocks - 1:
                                mainloop_pipeline.consumer_release(mainloop_consumer_state)
                                mainloop_consumer_state.advance()
                                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                    mainloop_consumer_state
                                )
                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsB_p = tCsB_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                            cute.copy(
                                smem_tiled_copy_B,
                                tCsB_p[None, None, k_block_next],
                                tCrB_copy_view[None, None, k_block_next],
                            )
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[None, None, k_block_idx],
                                tCrB[None, None, k_block_idx],
                                accumulators,
                            )

                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                        if k_block_next > 0:
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                            cute.copy(
                                smem_tiled_copy_B,
                                tCsB_p[None, None, k_block_next],
                                tCrB_copy_view[None, None, k_block_next],
                            )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )

                    copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                        self.c_layout,
                        elem_ty_d=self.c_dtype,
                        elem_ty_acc=self.acc_dtype,
                    )
                    copy_atom_C = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(
                            self.c_layout.is_m_major_c(), 4
                        ),
                        self.c_dtype,
                    )
                    tiled_copy_C_atom = cute.make_tiled_copy_C_atom(
                        copy_atom_C, tiled_mma
                    )
                    tiled_copy_r2s = cute.make_tiled_copy_S(
                        copy_atom_r2s, tiled_copy_C_atom
                    )
                    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                    tRS_sD = thr_copy_r2s.partition_D(sC)
                    tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                    rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                    tRS_rD_layout = cute.make_layout(rD_shape[:3])
                    tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                    size_tRS_rD = cute.size(tRS_rD)

                    sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                    tcgc_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile
                    )
                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        sepi_for_tma_partition,
                        tcgc_for_tma_partition,
                    )

                    epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
                    epi_tile_shape = tcgc_for_tma_partition.shape[1]
                    epi_tile_layout = cute.make_layout(
                        epi_tile_shape, stride=(1, epi_tile_shape[0])
                    )
                    tma_store_producer_group = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp,
                    )
                    tma_store_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.epi_stage,
                        producer_group=tma_store_producer_group,
                    )

                    for epi_idx in cutlass.range_constexpr(epi_tile_num):
                        for epi_v in cutlass.range_constexpr(size_tRS_rD):
                            tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                        tRS_rD_out = cute.make_rmem_tensor(
                            tRS_rD_layout.shape, self.c_dtype
                        )
                        if cutlass.const_expr(self.epilogue == "relu2"):
                            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                                activated = cutlass.Float32(
                                    cutlass.BFloat16(tRS_rD[epi_v])
                                )
                                activated = fmax_f32(activated, cutlass.Float32(0.0))
                                tRS_rD_out[epi_v] = cutlass.BFloat16(
                                    activated * activated
                                )
                        else:
                            tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                        epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, gmem_coord)],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()
                    tma_store_pipeline.producer_tail()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        elif warp_idx == self.num_mma_warps:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                weight_expert_idx = expert_ids[tile_coord_mnl[2]].to(cutlass.Int32)
                if weight_expert_idx >= cutlass.Int32(0):
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, weight_expert_idx)]
                    mainloop_producer_state.reset_count()

                    for _k_tile in range(0, k_tile_cnt, 1, unroll=1):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                        tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                        tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                        tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)


class ExpertIndexedDenseRow1GridKernel(ExpertIndexedDenseGemmKernel):
    _uses_row1_grid_launch = True

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        expert_ids: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        del max_active_clusters
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.expert_ids_dtype = expert_ids.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 A, got {self.a_dtype}")
        if cutlass.const_expr(self.b_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 B, got {self.b_dtype}")
        if cutlass.const_expr(self.expert_ids_dtype != cutlass.Int32):
            raise TypeError(f"expected Int32 expert_ids, got {self.expert_ids_dtype}")
        if cutlass.const_expr(self.c_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 C, got {self.c_dtype}")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._get_or_make_tma_load(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_b, tma_tensor_b = self._get_or_make_tma_load(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_c, tma_tensor_c = self._get_or_make_tma_store(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )
        grid = (
            (a.shape[0] + self.tile_shape_mnk[0] - 1) // self.tile_shape_mnk[0],
            c.shape[1] // self.tile_shape_mnk[1],
            c.shape[2],
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            expert_ids,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        expert_ids: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        weight_expert_idx = cute.arch.make_warp_uniform(
            expert_ids[bidz].to(cutlass.Int32)
        )

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_warps
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        pipeline.sync(barrier_id=1)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )
        gC_mnl_tile = gC_mnl[(None, None, bidx, bidy, bidz)]

        thr_mma = tiled_mma.get_slice(tidx)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCgC = thr_mma.partition_C(gC_mnl_tile)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

        atom_copy_ldmatrix_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
            self.a_dtype,
        )
        atom_copy_ldmatrix_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
            self.b_dtype,
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
        thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
        thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        num_k_blocks = cute.size(tCrA, mode=[2])

        if warp_idx < self.num_mma_warps:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)
            mainloop_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            _consumer_dense_pass(
                tiled_mma,
                mainloop_pipeline,
                mainloop_consumer_state,
                k_tile_cnt,
                num_k_blocks,
                tCsA_copy_view,
                tCrA_copy_view,
                tCsB_copy_view,
                tCrB_copy_view,
                smem_tiled_copy_A,
                smem_tiled_copy_B,
                tCrA,
                tCrB,
                accumulators,
            )

            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(), 4
                ),
                self.c_dtype,
            )
            tiled_copy_C_atom = cute.make_tiled_copy_C_atom(
                copy_atom_C, tiled_mma
            )
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s, tiled_copy_C_atom
            )
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
            tcgc_for_tma_partition = cute.zipped_divide(
                gC_mnl_tile, self.epi_tile
            )
            bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                tma_atom_c,
                0,
                cute.make_layout(1),
                sepi_for_tma_partition,
                tcgc_for_tma_partition,
            )

            epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
            epi_tile_shape = tcgc_for_tma_partition.shape[1]
            epi_tile_layout = cute.make_layout(
                epi_tile_shape, stride=(1, epi_tile_shape[0])
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps * self.num_threads_per_warp,
                ),
            )

            for epi_idx in cutlass.range_constexpr(epi_tile_num):
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                tRS_rD_out = cute.make_rmem_tensor(
                    tRS_rD_layout.shape, self.c_dtype
                )
                if cutlass.const_expr(self.epilogue == "relu2"):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        activated = cutlass.Float32(
                            cutlass.BFloat16(tRS_rD[epi_v])
                        )
                        activated = fmax_f32(activated, cutlass.Float32(0.0))
                        tRS_rD_out[epi_v] = cutlass.BFloat16(
                            activated * activated
                        )
                else:
                    tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                cute.copy(
                    tiled_copy_r2s,
                    tRS_rD_out,
                    tRS_sD[(None, None, None, epi_buffer)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()
                gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                if warp_idx == 0:
                    cute.copy(
                        tma_atom_c,
                        bSG_sD[(None, epi_buffer)],
                        bSG_gD[(None, gmem_coord)],
                    )
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()
            tma_store_pipeline.producer_tail()

        elif warp_idx == self.num_mma_warps:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)
            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            _producer_load_dense_pass(
                mainloop_pipeline,
                mainloop_producer_state,
                k_tile_cnt,
                tma_atom_a,
                tAgA[(None, bidx, None, bidz)],
                tAsA,
                tma_atom_b,
                tBgB[(None, bidy, None, weight_expert_idx)],
                tBsB,
            )
            mainloop_pipeline.producer_tail(mainloop_producer_state)


__all__ = [
    "ExpertIndexedDenseGemmKernel",
    "ExpertIndexedDenseRow1GridKernel",
    "run_dense_bf16_expert_ids",
]
