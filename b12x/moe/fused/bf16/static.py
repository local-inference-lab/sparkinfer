"""MoEStaticKernelBackend — compact static routed BF16 MoE backend."""

from __future__ import annotations

import warnings
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import from_dlpack

from b12x.attention.utils import fmax as fmax_f32
from b12x.integration.triton_bf16_scatter import (
    gather_rows_bf16,
    permute_rows_bf16,
    scatter_add_compact_chunk_fc2_bf16,
    scatter_add_grouped_fc2_bf16,
    scatter_add_token_map_fc2_bf16,
    scatter_routed_input_compact_chunk_bf16,
    scatter_routed_input_grouped_bf16,
    scatter_routed_input_token_map_bf16,
)
from b12x.integration.triton_compact import (
    build_bucketed_compact_route,
    build_compact_route_sorted_singleton_direct_state,
)
from b12x.cute.utils import current_cuda_stream, get_num_sm
from b12x.integration.triton_bf16_reduce import reduce_fc2_chunk_grouped_bf16


_EAGER_HOST_LAUNCHER_CACHE_SIZE = 32
_FUSED_TILE_SHAPE_MNK = (128, 64, 64)
_RELU2_INDEXED_DENSE_MAX_ROWS = 256
# For Nemotron relu2, expert-sorted direct execution starts paying off once we
# reach bs=2 (44 routed rows at top_k=22). bs=1 stays flat, so keep the
# unsorted path there.
_RELU2_SORTED_DIRECT_ROUTE_MIN_ROWS = 44
_ENABLE_FUSED_DIRECT_RELU2 = (
    os.getenv("B12X_BF16_ENABLE_FUSED_DIRECT_RELU2", "0") == "1"
)
_ENABLE_ROW1_GRID_INDEXED_DENSE = (
    os.getenv("B12X_BF16_ENABLE_ROW1_GRID_INDEXED_DENSE", "0") == "1"
)
_ENABLE_BUCKETED_COMPACT_RELU2_STATIC = (
    os.getenv("B12X_BF16_ENABLE_BUCKETED_COMPACT_RELU2_STATIC", "0") == "1"
)
_BMM_CUBLAS_WARMED_DEVICES: set[int] = set()


def _parse_int_tuple_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    values = tuple(int(part.strip()) for part in raw.split(","))
    if len(values) != len(default):
        raise ValueError(
            f"{name} must contain {len(default)} comma-separated integers, got {raw!r}"
        )
    return values


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}")


def _parse_optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _get_fused_direct_relu2_variant() -> str:
    raw = os.getenv("B12X_BF16_FUSED_DIRECT_RELU2_VARIANT", "grid").strip().lower()
    if raw in {"grid", "flat_persistent"}:
        return raw
    raise ValueError(
        "B12X_BF16_FUSED_DIRECT_RELU2_VARIANT must be 'grid' or "
        f"'flat_persistent', got {raw!r}"
    )


@dataclass(frozen=True)
class _CompactRouteChunk:
    expert_ids_i32: torch.Tensor | None
    expert_ids_i64: torch.Tensor | None
    compact_flat_token_indices_gpu: torch.Tensor | None
    compact_topk_ids_gpu: torch.Tensor | None
    compact_route_row_indices_gpu: torch.Tensor | None
    compact_expert_begin: int
    compact_expert_end: int
    token_map_gpu: torch.Tensor | None = None
    token_weights_gpu: torch.Tensor | None = None


def _round_up_tile_m(rows: int) -> int:
    tile_m = _FUSED_TILE_SHAPE_MNK[0]
    return max(tile_m, ((rows + tile_m - 1) // tile_m) * tile_m)


def _round_up_rows(rows: int, tile_m: int) -> int:
    return max(tile_m, ((rows + tile_m - 1) // tile_m) * tile_m)


def _alloc_batched_matrix(
    mode0: int,
    mode1: int,
    batch: int,
    *,
    mode0_major: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if mode0_major:
        return torch.empty(
            (batch, mode1, mode0), dtype=dtype, device=device
        ).permute(2, 1, 0)
    return torch.empty((batch, mode0, mode1), dtype=dtype, device=device).permute(
        1, 2, 0
    )


def _tensor_meta_key(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _launcher_cache_lookup(kernel: object, cache_key: tuple[object, ...]):
    cache = getattr(kernel, "_eager_host_launchers", None)
    if cache is None:
        cache = OrderedDict()
        setattr(kernel, "_eager_host_launchers", cache)
        return cache, None
    compiled = cache.get(cache_key)
    if compiled is not None:
        cache.move_to_end(cache_key)
    return cache, compiled


def _ensure_bmm_cublas_ready(device: torch.device) -> None:
    device_index = device.index or 0
    if device_index in _BMM_CUBLAS_WARMED_DEVICES:
        return
    warm_a = torch.empty((1, 1, 1), dtype=torch.bfloat16, device=device)
    warm_b = torch.empty((1, 1, 1), dtype=torch.bfloat16, device=device)
    warm_out = torch.empty((1, 1, 1), dtype=torch.bfloat16, device=device)
    torch.bmm(warm_a, warm_b, out=warm_out)
    _BMM_CUBLAS_WARMED_DEVICES.add(device_index)


def _run_cached_host_launcher(
    kernel: object,
    cache_key: tuple[object, ...],
    args: tuple[object, ...],
) -> None:
    cache, compiled = _launcher_cache_lookup(kernel, cache_key)
    if compiled is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cache is disabled as user wants to compile only.",
                category=UserWarning,
            )
            compiled = kernel(*args, compile_only=True)
        cache[cache_key] = compiled
        if len(cache) > _EAGER_HOST_LAUNCHER_CACHE_SIZE:
            cache.popitem(last=False)
    exe_args, _ = compiled.generate_execution_args(*args)
    compiled.run_compiled_program(exe_args)


def _to_dense_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric] = cutlass.BFloat16,
    *,
    assumed_align: int = 16,
) -> cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    # Prefer a stride-1 axis with size > 1: size-1 axes carry an ambiguous stride.
    leading_dim = next(
        (
            idx
            for idx, (stride, size) in enumerate(zip(tensor.stride(), tensor.shape))
            if stride == 1 and size > 1
        ),
        None,
    )
    if leading_dim is None:
        leading_dim = next(
            (idx for idx, stride in enumerate(tensor.stride()) if stride == 1),
            None,
        )
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def run_dense_bf16(
    kernel: "DenseGemmKernel",
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    max_active_clusters: int,
    stream: cuda.CUstream,
) -> None:
    args = (
        _to_dense_kernel_tensor(a),
        _to_dense_kernel_tensor(b),
        _to_dense_kernel_tensor(c),
        max_active_clusters,
        stream,
    )
    cache_key = (
        tuple(kernel.tile_shape_mnk),
        _tensor_meta_key(a),
        _tensor_meta_key(b),
        _tensor_meta_key(c),
        max_active_clusters,
    )
    _run_cached_host_launcher(kernel, cache_key, args)


class DenseGemmKernel:
    def __init__(
        self,
        tile_shape_mnk: Tuple[int, int, int] = (128, 128, 64),
        *,
        acc_dtype=cutlass.Float32,
        epilogue: str = "identity",
    ):
        if epilogue not in {"identity", "relu2"}:
            raise ValueError(f"unsupported dense epilogue {epilogue!r}")
        self.acc_dtype = acc_dtype
        self.epilogue = epilogue
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.tiled_mma = None

        self.occupancy = 1
        self.atom_layout = (2, 2, 1)
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024
        self._load_tma_cache = {}
        self._store_tma_cache = {}

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def configure_atom_layout(self, atom_layout: tuple[int, int, int]) -> None:
        self.atom_layout = tuple(atom_layout)
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )

    def _setup_attributes(self):
        self.mma_inst_mnk = (16, 8, 16)
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk,
            self.c_dtype,
            is_cooperative=False,
        )
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )
        if self.ab_stage <= 0:
            raise RuntimeError("ab_stage <= 0 for BF16 dense GEMM")
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 A, got {self.a_dtype}")
        if cutlass.const_expr(self.b_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 B, got {self.b_dtype}")
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
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
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
                    k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
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
                    cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
                    self.c_dtype,
                )
                tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                size_tRS_rD = cute.size(tRS_rD)

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)
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
                    tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
                    if cutlass.const_expr(self.epilogue == "relu2"):
                        for epi_v in cutlass.range_constexpr(size_tRS_rD):
                            activated = cutlass.Float32(cutlass.BFloat16(tRS_rD[epi_v]))
                            activated = fmax_f32(activated, cutlass.Float32(0.0))
                            tRS_rD_out[epi_v] = cutlass.BFloat16(activated * activated)
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

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                tma_store_pipeline.producer_tail()

        elif warp_idx == self.num_mma_warps:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
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

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype,
        b_dtype,
        epi_tile: tuple[int, int],
        c_dtype,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        epi_stage = 8
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype,
        a_layout: cute.Layout,
        b_dtype,
        b_layout: cute.Layout,
        ab_stage: int,
        c_dtype,
        c_layout: cute.Layout,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )
        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ):
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    def _get_or_make_tma_store(
        self,
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ):
        key = (
            repr(tensor_c.iterator),
            tuple(tensor_c.shape),
            tuple(tensor_c.stride),
            tuple(epi_tile),
        )
        cached = self._store_tma_cache.get(key)
        if cached is not None:
            return cached
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        cached = (tma_atom_c, tma_tensor_c)
        self._store_tma_cache[key] = cached
        return cached

    def _get_or_make_tma_load(
        self,
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ):
        key = (
            repr(tensor.iterator),
            tuple(tensor.shape),
            tuple(tensor.stride),
            tuple(smem_tile),
            mcast_dim,
        )
        cached = self._load_tma_cache.get(key)
        if cached is not None:
            return cached
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        cached = (tma_atom, tma_tensor)
        self._load_tma_cache[key] = cached
        return cached


@cute.jit
def _warp_mma_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    A_in_regs: cutlass.Constexpr = False,
    B_in_regs: cutlass.Constexpr = False,
):
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    if cutlass.const_expr(not A_in_regs):
        cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    if cutlass.const_expr(not B_in_regs):
        cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        if k < cute.size(tCsA.shape[2]) - 1:
            if cutlass.const_expr(not A_in_regs):
                cute.copy(
                    smem_thr_copy_A,
                    tCsA[None, None, k + 1],
                    tCrA_copy_view[None, None, k + 1],
                )
            if cutlass.const_expr(not B_in_regs):
                cute.copy(
                    smem_thr_copy_B,
                    tCsB[None, None, k + 1],
                    tCrB_copy_view[None, None, k + 1],
                )
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def _producer_load_dense_pass(
    load_pipeline,
    producer_state,
    k_tile_cnt,
    tma_atom_a,
    tAgA_mkl,
    tAsA,
    tma_atom_b,
    tBgB_nkl,
    tBsB,
):
    producer_state.reset_count()
    for k_tile in range(0, k_tile_cnt, 1, unroll=4):
        load_pipeline.producer_acquire(producer_state)
        cute.copy(
            tma_atom_a,
            tAgA_mkl[(None, k_tile)],
            tAsA[(None, producer_state.index)],
            tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
        )
        cute.copy(
            tma_atom_b,
            tBgB_nkl[(None, k_tile)],
            tBsB[(None, producer_state.index)],
            tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
        )
        load_pipeline.producer_commit(producer_state)
        producer_state.advance()


@cute.jit
def _consumer_dense_pass(
    tiled_mma: cute.TiledMma,
    load_pipeline,
    consumer_state,
    k_tile_cnt,
    num_k_blocks: cutlass.Constexpr,
    tCsA_copy_view: cute.Tensor,
    tCrA_copy_view: cute.Tensor,
    tCsB_copy_view: cute.Tensor,
    tCrB_copy_view: cute.Tensor,
    smem_tiled_copy_A: cute.TiledCopy,
    smem_tiled_copy_B: cute.TiledCopy,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    acc: cute.Tensor,
):
    acc.fill(0.0)
    consumer_state.reset_count()

    peek_status = cutlass.Boolean(1)
    if consumer_state.count < k_tile_cnt:
        peek_status = load_pipeline.consumer_try_wait(consumer_state)
    load_pipeline.consumer_wait(consumer_state, peek_status)

    tCsA_p = tCsA_copy_view[None, None, None, consumer_state.index]
    tCsB_p = tCsB_copy_view[None, None, None, consumer_state.index]
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
            k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
            if k_block_idx == num_k_blocks - 1:
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
                peek_status = load_pipeline.consumer_try_wait(consumer_state)
                tCsA_p = tCsA_copy_view[None, None, None, consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, consumer_state.index]
                load_pipeline.consumer_wait(consumer_state, peek_status)
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
                acc,
                tCrA[None, None, k_block_idx],
                tCrB[None, None, k_block_idx],
                acc,
            )

    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
        k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
        if k_block_idx == num_k_blocks - 1:
            load_pipeline.consumer_release(consumer_state)
            consumer_state.advance()
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
            acc,
            tCrA[None, None, k_block_idx],
            tCrB[None, None, k_block_idx],
            acc,
        )


class _FusedChunkKernel(DenseGemmKernel):
    def __init__(self, *, activation: str):
        super().__init__(_FUSED_TILE_SHAPE_MNK)
        if activation not in {"silu", "relu2"}:
            raise ValueError(f"unsupported activation {activation!r}")
        self.activation = activation
        self.is_gated = activation == "silu"
        self.load_warp_id = self.num_mma_warps
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.pass_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_cta,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        w1: cute.Tensor,
        w2: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = w1.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(w1)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 A, got {self.a_dtype}")
        if cutlass.const_expr(self.b_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 W1, got {self.b_dtype}")
        if cutlass.const_expr(w2.element_type != cutlass.BFloat16):
            raise TypeError(f"expected BF16 W2, got {w2.element_type}")
        if cutlass.const_expr(self.c_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 C, got {self.c_dtype}")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._get_or_make_tma_load(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_w1, tma_tensor_w1 = self._get_or_make_tma_load(
            w1,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_w2, tma_tensor_w2 = self._get_or_make_tma_load(
            w2,
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
            a.shape[0] // self.tile_shape_mnk[0],
            c.shape[1] // self.tile_shape_mnk[1],
            c.shape[2],
        )

        @cute.struct
        class SharedStorageGated:
            fc1_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            fc1_up_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            fc2_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, 2]
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
            sB_up: cute.struct.Align[
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

        @cute.struct
        class SharedStorageRelu2:
            fc1_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            fc2_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, 2]
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

        self.shared_storage = SharedStorageGated if self.is_gated else SharedStorageRelu2

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_w1,
            tma_tensor_w1,
            tma_atom_w2,
            tma_tensor_w2,
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
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_w1: cute.CopyAtom,
        mW1_nkl: cute.Tensor,
        tma_atom_w2: cute.CopyAtom,
        mW2_nkl: cute.Tensor,
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
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        fc1_tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        fc2_tma_copy_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        fc1_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps,
            ),
            tx_count=fc1_tma_copy_bytes,
            barrier_storage=storage.fc1_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        fc1_up_pipeline = (
            pipeline.PipelineTmaAsync.create(
                num_stages=self.ab_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps,
                ),
                tx_count=fc1_tma_copy_bytes,
                barrier_storage=storage.fc1_up_pipeline_array_ptr.data_ptr(),
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            if self.is_gated
            else fc1_pipeline
        )
        fc2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps,
            ),
            tx_count=fc2_tma_copy_bytes,
            barrier_storage=storage.fc2_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        pipeline.sync(barrier_id=1)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sB_up = (
            storage.sB_up.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            if self.is_gated
            else sB
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gW1_nkl = cute.local_tile(
            mW1_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gW2_nkl = cute.local_tile(
            mW2_nkl,
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
        tBsW1, tBgW1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w1,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gW1_nkl, 0, 2),
        )
        tBsW1_up, _ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w1,
            0,
            cute.make_layout(1),
            cute.group_modes(sB_up, 0, 2),
            cute.group_modes(gW1_nkl, 0, 2),
        )
        tBsW2, tBgW2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w2,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gW2_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsB_up = thr_mma.partition_B(sB_up)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrB_up = tiled_mma.make_fragment_B(tCsB_up[None, None, None, 0])
        tCgC = thr_mma.partition_C(gC_mnl_tile)
        out_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        out_acc.fill(0.0)

        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        epi_n_scale = self.tile_shape_mnk[1] // self.epi_tile[1]
        gate_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        up_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

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
        tCsB_up_copy_view = thr_copy_ldmatrix_B.partition_S(sB_up)
        tCrB_up_copy_view = thr_copy_ldmatrix_B.retile(tCrB_up)

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=cutlass.BFloat16,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
            cutlass.BFloat16,
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
        tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)
        tRS_rGate = tiled_copy_r2s.retile(gate_acc)
        if cutlass.const_expr(self.is_gated):
            tRS_rUp = tiled_copy_r2s.retile(up_acc)

        k_tile_cnt = cute.size(mA_mkl, mode=[1]) // self.tile_shape_mnk[2]
        inter_tile_cnt = cute.size(mW2_nkl, mode=[1]) // self.tile_shape_mnk[2]
        gate_tile_offset = cutlass.Int32(inter_tile_cnt) if self.is_gated else cutlass.Int32(0)
        num_k_blocks = cute.size(tCrA, mode=[2])
        epi_rest_m = epi_m_scale
        mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
        mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rGate, mode=[2])
        mma_m_per_epi_m = self.epi_tile[0] // mma_tile_m
        mma_n_per_epi_n = self.epi_tile[1] // mma_tile_n
        mma_thread_count = cutlass.Int32(
            self.num_mma_warps * self.num_threads_per_warp
        )

        if warp_idx < self.num_mma_warps:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)

            fc2_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )

            for _inter_tile_idx in range(0, inter_tile_cnt, 1, unroll=1):
                inter_tile_idx = cutlass.Int32(_inter_tile_idx)
                fc1_gate_cons_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                tBgW1_gate = tBgW1[(None, inter_tile_idx + gate_tile_offset, None, bidz)]
                _consumer_dense_pass(
                    tiled_mma,
                    fc1_pipeline,
                    fc1_gate_cons_state,
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
                    gate_acc,
                )
                self.pass_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.is_gated):
                    fc1_up_cons_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.ab_stage
                    )
                    tBgW1_up = tBgW1[(None, inter_tile_idx, None, bidz)]
                    _consumer_dense_pass(
                        tiled_mma,
                        fc1_up_pipeline,
                        fc1_up_cons_state,
                        k_tile_cnt,
                        num_k_blocks,
                        tCsA_copy_view,
                        tCrA_copy_view,
                        tCsB_up_copy_view,
                        tCrB_up_copy_view,
                        smem_tiled_copy_A,
                        smem_tiled_copy_B,
                        tCrA,
                        tCrB_up,
                        up_acc,
                    )

                for epi_n in cutlass.range_constexpr(epi_n_scale):
                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        epi_idx = epi_m * epi_n_scale + epi_n
                        epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                        tRS_rD.fill(0.0)
                        for mma_n_in_epi in cutlass.range_constexpr(mma_n_per_epi_n):
                            for mma_m_in_epi in cutlass.range_constexpr(mma_m_per_epi_m):
                                mma_m = epi_m * mma_m_per_epi_m + mma_m_in_epi
                                mma_n = epi_n * mma_n_per_epi_n + mma_n_in_epi
                                tRS_rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                                gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                if cutlass.const_expr(self.is_gated):
                                    up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                    for elem_idx in cutlass.range_constexpr(cute.size(tRS_rD_slice)):
                                        g = cutlass.Float32(
                                            cutlass.BFloat16(gate_slice[elem_idx])
                                        )
                                        u = cutlass.Float32(
                                            cutlass.BFloat16(up_slice[elem_idx])
                                        )
                                        sigmoid_g = cute.arch.rcp_approx(
                                            cutlass.Float32(1.0) + cute.math.exp(-g)
                                        )
                                        tRS_rD_slice[elem_idx] = g * sigmoid_g * u
                                else:
                                    for elem_idx in cutlass.range_constexpr(cute.size(tRS_rD_slice)):
                                        g = cutlass.Float32(
                                            cutlass.BFloat16(gate_slice[elem_idx])
                                        )
                                        relu_g = fmax_f32(g, cutlass.Float32(0.0))
                                        tRS_rD_slice[elem_idx] = relu_g * relu_g

                        acc_vec = tRS_rD.load()
                        acc_vec = acc_vec.to(cutlass.BFloat16)
                        tRS_rD_out.store(acc_vec)
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        copy_idx = cutlass.Int32(tidx)
                        epi_rows = cutlass.Int32(self.epi_tile[0])
                        epi_cols = cutlass.Int32(self.epi_tile[1])
                        row_base = cutlass.Int32(epi_m) * epi_rows
                        col_base = cutlass.Int32(epi_n) * epi_cols
                        total_copy = epi_rows * epi_cols
                        while copy_idx < total_copy:
                            local_row = copy_idx // epi_cols
                            col = copy_idx - local_row * epi_cols
                            sA[row_base + local_row, col_base + col, 0] = sC[
                                local_row, col, epi_buffer
                            ]
                            copy_idx += mma_thread_count
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                # All MMA warps must finish materializing the activated BF16
                # intermediate into sA before phase2 starts reading it.
                self.epilog_sync_barrier.arrive_and_wait()

                self.pass_sync_barrier.arrive_and_wait()

                phase2_peek = fc2_pipeline.consumer_try_wait(fc2_cons_state)
                fc2_pipeline.consumer_wait(fc2_cons_state, phase2_peek)
                csB_phase2 = tCsB_copy_view[None, None, None, fc2_cons_state.index]
                csA_phase2 = tCsA_copy_view[None, None, None, 0]
                _warp_mma_gemm(
                    tiled_mma,
                    out_acc,
                    tCrA,
                    tCrB,
                    csA_phase2,
                    csB_phase2,
                    smem_tiled_copy_A,
                    smem_tiled_copy_B,
                )
                fc2_pipeline.consumer_release(fc2_cons_state)
                fc2_cons_state.advance()
                self.pass_sync_barrier.arrive_and_wait()

            sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
            tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_tile, self.epi_tile)
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

            tRS_rAcc = tiled_copy_r2s.retile(out_acc)
            for epi_idx in cutlass.range_constexpr(epi_tile_num):
                for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
                    tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]
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

        elif warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)

            fc2_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            fc2_prod_state.reset_count()

            tAgA_mkl = tAgA[(None, bidx, None, bidz)]
            for _inter_tile_idx in range(0, inter_tile_cnt, 1, unroll=1):
                inter_tile_idx = cutlass.Int32(_inter_tile_idx)
                fc1_gate_prod_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                tBgW1_gate = tBgW1[(None, inter_tile_idx + gate_tile_offset, None, bidz)]
                _producer_load_dense_pass(
                    fc1_pipeline,
                    fc1_gate_prod_state,
                    k_tile_cnt,
                    tma_atom_a,
                    tAgA_mkl,
                    tAsA,
                    tma_atom_w1,
                    tBgW1_gate,
                    tBsW1,
                )
                self.pass_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.is_gated):
                    fc1_up_prod_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.ab_stage
                    )
                    tBgW1_up = tBgW1[(None, inter_tile_idx, None, bidz)]
                    _producer_load_dense_pass(
                        fc1_up_pipeline,
                        fc1_up_prod_state,
                        k_tile_cnt,
                        tma_atom_a,
                        tAgA_mkl,
                        tAsA,
                        tma_atom_w1,
                        tBgW1_up,
                        tBsW1_up,
                    )
                    self.pass_sync_barrier.arrive_and_wait()
                else:
                    # Keep the producer/consumer barrier cadence identical to
                    # the gated specialization without issuing a dummy relu2
                    # FC1 replay pass.
                    self.pass_sync_barrier.arrive_and_wait()

                fc2_pipeline.producer_acquire(fc2_prod_state)
                cute.copy(
                    tma_atom_w2,
                    tBgW2[(None, bidy, inter_tile_idx, bidz)],
                    tBsW2[(None, fc2_prod_state.index)],
                    tma_bar_ptr=fc2_pipeline.producer_get_barrier(fc2_prod_state),
                )
                fc2_pipeline.producer_commit(fc2_prod_state)
                fc2_prod_state.advance()
                self.pass_sync_barrier.arrive_and_wait()

            fc2_pipeline.producer_tail(fc2_prod_state)
        return


class _FC1ActivationChunkKernel(_FusedChunkKernel):
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        w1: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = w1.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(w1)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 A, got {self.a_dtype}")
        if cutlass.const_expr(self.b_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 W1, got {self.b_dtype}")
        if cutlass.const_expr(self.c_dtype != cutlass.BFloat16):
            raise TypeError(f"expected BF16 C, got {self.c_dtype}")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._get_or_make_tma_load(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_w1, tma_tensor_w1 = self._get_or_make_tma_load(
            w1,
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
            a.shape[0] // self.tile_shape_mnk[0],
            1,
            c.shape[2],
        )

        @cute.struct
        class SharedStorageGated:
            fc1_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            fc1_up_pipeline_array_ptr: cute.struct.MemRange[
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
            sB_up: cute.struct.Align[
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

        @cute.struct
        class SharedStorageRelu2:
            fc1_pipeline_array_ptr: cute.struct.MemRange[
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

        self.shared_storage = SharedStorageGated if self.is_gated else SharedStorageRelu2

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_w1,
            tma_tensor_w1,
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
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_w1: cute.CopyAtom,
        mW1_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w1)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        fc1_tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        fc1_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps,
            ),
            tx_count=fc1_tma_copy_bytes,
            barrier_storage=storage.fc1_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        fc1_up_pipeline = (
            pipeline.PipelineTmaAsync.create(
                num_stages=self.ab_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps,
                ),
                tx_count=fc1_tma_copy_bytes,
                barrier_storage=storage.fc1_up_pipeline_array_ptr.data_ptr(),
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            if self.is_gated
            else fc1_pipeline
        )
        pipeline.sync(barrier_id=1)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sB_up = (
            storage.sB_up.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            if self.is_gated
            else sB
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gW1_nkl = cute.local_tile(
            mW1_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )
        gC_mnl_tile = gC_mnl[(None, None, bidx, 0, bidz)]

        thr_mma = tiled_mma.get_slice(tidx)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        tBsW1, tBgW1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w1,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gW1_nkl, 0, 2),
        )
        tBsW1_up, _ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w1,
            0,
            cute.make_layout(1),
            cute.group_modes(sB_up, 0, 2),
            cute.group_modes(gW1_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsB_up = thr_mma.partition_B(sB_up)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrB_up = tiled_mma.make_fragment_B(tCsB_up[None, None, None, 0])
        tCgC = thr_mma.partition_C(gC_mnl_tile)

        gate_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        up_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

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
        tCsB_up_copy_view = thr_copy_ldmatrix_B.partition_S(sB_up)
        tCrB_up_copy_view = thr_copy_ldmatrix_B.retile(tCrB_up)

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=cutlass.BFloat16,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
            cutlass.BFloat16,
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
        tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)
        tRS_rGate = tiled_copy_r2s.retile(gate_acc)
        if cutlass.const_expr(self.is_gated):
            tRS_rUp = tiled_copy_r2s.retile(up_acc)

        k_tile_cnt = cute.size(mA_mkl, mode=[1]) // self.tile_shape_mnk[2]
        gate_tile_offset = cutlass.Int32(1) if self.is_gated else cutlass.Int32(0)
        num_k_blocks = cute.size(tCrA, mode=[2])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        epi_n_scale = self.tile_shape_mnk[1] // self.epi_tile[1]
        mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
        mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rGate, mode=[2])
        mma_m_per_epi_m = self.epi_tile[0] // mma_tile_m
        mma_n_per_epi_n = self.epi_tile[1] // mma_tile_n
        mma_thread_count = cutlass.Int32(
            self.num_mma_warps * self.num_threads_per_warp
        )

        if warp_idx < self.num_mma_warps:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)

            fc1_gate_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            tBgW1_gate = tBgW1[(None, gate_tile_offset, None, bidz)]
            _consumer_dense_pass(
                tiled_mma,
                fc1_pipeline,
                fc1_gate_cons_state,
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
                gate_acc,
            )
            self.pass_sync_barrier.arrive_and_wait()

            if cutlass.const_expr(self.is_gated):
                fc1_up_cons_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                tBgW1_up = tBgW1[(None, 0, None, bidz)]
                _consumer_dense_pass(
                    tiled_mma,
                    fc1_up_pipeline,
                    fc1_up_cons_state,
                    k_tile_cnt,
                    num_k_blocks,
                    tCsA_copy_view,
                    tCrA_copy_view,
                    tCsB_up_copy_view,
                    tCrB_up_copy_view,
                    smem_tiled_copy_A,
                    smem_tiled_copy_B,
                    tCrA,
                    tCrB_up,
                    up_acc,
                )
            self.pass_sync_barrier.arrive_and_wait()

            for epi_n in cutlass.range_constexpr(epi_n_scale):
                for epi_m in cutlass.range_constexpr(epi_m_scale):
                    epi_idx = epi_m * epi_n_scale + epi_n
                    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                    tRS_rD.fill(0.0)
                    for mma_n_in_epi in cutlass.range_constexpr(mma_n_per_epi_n):
                        for mma_m_in_epi in cutlass.range_constexpr(mma_m_per_epi_m):
                            mma_m = epi_m * mma_m_per_epi_m + mma_m_in_epi
                            mma_n = epi_n * mma_n_per_epi_n + mma_n_in_epi
                            tRS_rD_slice = tRS_rD[
                                (None, mma_m_in_epi, mma_n_in_epi)
                            ]
                            gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                            if cutlass.const_expr(self.is_gated):
                                up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    g = cutlass.Float32(
                                        cutlass.BFloat16(gate_slice[elem_idx])
                                    )
                                    u = cutlass.Float32(
                                        cutlass.BFloat16(up_slice[elem_idx])
                                    )
                                    sigmoid_g = cute.arch.rcp_approx(
                                        cutlass.Float32(1.0)
                                        + cute.math.exp(-g)
                                    )
                                    tRS_rD_slice[elem_idx] = g * sigmoid_g * u
                            else:
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    g = cutlass.Float32(
                                        cutlass.BFloat16(gate_slice[elem_idx])
                                    )
                                    relu_g = fmax_f32(g, cutlass.Float32(0.0))
                                    tRS_rD_slice[elem_idx] = relu_g * relu_g

                    tRS_rD_out.store(tRS_rD.load().to(cutlass.BFloat16))
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    copy_idx = cutlass.Int32(tidx)
                    epi_rows = cutlass.Int32(self.epi_tile[0])
                    epi_cols = cutlass.Int32(self.epi_tile[1])
                    row_base = cutlass.Int32(bidx) * cutlass.Int32(self.tile_shape_mnk[0]) + cutlass.Int32(epi_m) * epi_rows
                    col_base = cutlass.Int32(epi_n) * epi_cols
                    total_copy = epi_rows * epi_cols
                    while copy_idx < total_copy:
                        local_row = copy_idx // epi_cols
                        col = copy_idx - local_row * epi_cols
                        mC_mnl[row_base + local_row, col_base + col, bidz] = sC[
                            local_row, col, epi_buffer
                        ]
                        copy_idx += mma_thread_count
                    self.epilog_sync_barrier.arrive_and_wait()

        elif warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)

            fc1_gate_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            tAgA_mkl = tAgA[(None, bidx, None, bidz)]
            tBgW1_gate = tBgW1[(None, gate_tile_offset, None, bidz)]
            _producer_load_dense_pass(
                fc1_pipeline,
                fc1_gate_prod_state,
                k_tile_cnt,
                tma_atom_a,
                tAgA_mkl,
                tAsA,
                tma_atom_w1,
                tBgW1_gate,
                tBsW1,
            )
            self.pass_sync_barrier.arrive_and_wait()

            if cutlass.const_expr(self.is_gated):
                fc1_up_prod_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                tBgW1_up = tBgW1[(None, 0, None, bidz)]
                _producer_load_dense_pass(
                    fc1_up_pipeline,
                    fc1_up_prod_state,
                    k_tile_cnt,
                    tma_atom_a,
                    tAgA_mkl,
                    tAsA,
                    tma_atom_w1,
                    tBgW1_up,
                    tBsW1_up,
                )
            self.pass_sync_barrier.arrive_and_wait()
        return


def _run_fused_chunk_bf16_single_slice(
    kernel: _FusedChunkKernel,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    c: torch.Tensor,
    stream: cuda.CUstream,
) -> None:
    if a.shape[0] % kernel.tile_shape_mnk[0] != 0:
        raise ValueError("expected routed rows padded to 128")
    if c.shape[1] % kernel.tile_shape_mnk[1] != 0:
        raise ValueError("expected hidden size padded to 64")
    if w2.shape[1] % kernel.tile_shape_mnk[2] != 0:
        raise ValueError("expected intermediate size padded to 64")
    args = (a, w1, w2, c, stream)
    cache_key = (
        kernel.activation,
        tuple(kernel.tile_shape_mnk),
        _tensor_meta_key(a),
        _tensor_meta_key(w1),
        _tensor_meta_key(w2),
        _tensor_meta_key(c),
    )
    _run_cached_host_launcher(kernel, cache_key, args)


def _run_fc1_activation_chunk_bf16_single_slice(
    kernel: _FC1ActivationChunkKernel,
    a: torch.Tensor,
    w1: torch.Tensor,
    c: torch.Tensor,
    stream: cuda.CUstream,
) -> None:
    if a.shape[0] % kernel.tile_shape_mnk[0] != 0:
        raise ValueError("expected routed rows padded to 128")
    if c.shape[1] != kernel.tile_shape_mnk[1]:
        raise ValueError("expected intermediate slice width 64")
    args = (a, w1, c, stream)
    cache_key = (
        kernel.activation,
        tuple(kernel.tile_shape_mnk),
        _tensor_meta_key(a),
        _tensor_meta_key(w1),
        _tensor_meta_key(c),
    )
    _run_cached_host_launcher(kernel, cache_key, args)


def run_fused_chunk_bf16(
    kernel: _FusedChunkKernel,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    c: torch.Tensor,
    stream: cuda.CUstream,
) -> None:
    if a.shape[0] % kernel.tile_shape_mnk[0] != 0:
        raise ValueError("expected routed rows padded to 128")
    if c.shape[1] % kernel.tile_shape_mnk[1] != 0:
        raise ValueError("expected hidden size padded to 64")
    if w2.shape[1] % kernel.tile_shape_mnk[2] != 0:
        raise ValueError("expected intermediate size padded to 64")

    inter_tile = kernel.tile_shape_mnk[2]
    inter_tile_cnt = w2.shape[1] // inter_tile
    if inter_tile_cnt == 1:
        _run_fused_chunk_bf16_single_slice(kernel, a, w1, w2, c, stream)
        return

    partial_out = torch.empty_like(c)
    accum_out = torch.zeros_like(c, dtype=torch.float32)
    gate_offset = w2.shape[1]
    for inter_tile_idx in range(inter_tile_cnt):
        inter_begin = inter_tile_idx * inter_tile
        inter_end = inter_begin + inter_tile
        if kernel.is_gated:
            w1_slice = torch.empty(
                (inter_tile * 2, w1.shape[1], w1.shape[2]),
                dtype=w1.dtype,
                device=w1.device,
            )
            w1_slice[:inter_tile].copy_(w1[inter_begin:inter_end])
            w1_slice[inter_tile:].copy_(
                w1[gate_offset + inter_begin : gate_offset + inter_end]
            )
        else:
            w1_slice = w1[inter_begin:inter_end]
        w2_slice = w2[:, inter_begin:inter_end, :]
        partial_out.zero_()
        _run_fused_chunk_bf16_single_slice(
            kernel,
            a,
            w1_slice,
            w2_slice,
            partial_out,
            stream,
        )
        accum_out.add_(partial_out.float())
    c.copy_(accum_out.to(torch.bfloat16))


class _ExpertIndexedFusedRelu2Kernel(DenseGemmKernel):
    def __init__(self, tile_shape_mnk: Tuple[int, int, int] = (16, 64, 64)):
        super().__init__(tile_shape_mnk)
        self.load_warp_id = self.num_mma_warps
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.pass_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_cta,
        )

    def configure_atom_layout(self, atom_layout: tuple[int, int, int]) -> None:
        super().configure_atom_layout(atom_layout)
        self.load_warp_id = self.num_mma_warps
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.pass_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_cta,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        w1: cute.Tensor,
        w2: cute.Tensor,
        expert_ids: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = w1.element_type
        self.expert_ids_dtype = expert_ids.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(w1)
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
        tma_atom_w1, tma_tensor_w1 = self._get_or_make_tma_load(
            w1,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_w2, tma_tensor_w2 = self._get_or_make_tma_load(
            w2,
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
            self.cluster_shape_mnk,
        )

        @cute.struct
        class SharedStorage:
            fc1_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            fc2_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, 2]
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
            tma_atom_w1,
            tma_tensor_w1,
            tma_atom_w2,
            tma_tensor_w2,
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
        tma_atom_w1: cute.CopyAtom,
        mW1_nkl: cute.Tensor,
        tma_atom_w2: cute.CopyAtom,
        mW2_nkl: cute.Tensor,
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
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        fc1_tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        fc2_tma_copy_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        fc1_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps,
            ),
            tx_count=fc1_tma_copy_bytes,
            barrier_storage=storage.fc1_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
        )
        fc2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps,
            ),
            tx_count=fc2_tma_copy_bytes,
            barrier_storage=storage.fc2_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
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
        gW1_nkl = cute.local_tile(
            mW1_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gW2_nkl = cute.local_tile(
            mW2_nkl,
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
        tBsW1, tBgW1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w1,
            cluster_coord_mnk[0],
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gW1_nkl, 0, 2),
        )
        tBsW2, tBgW2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w2,
            cluster_coord_mnk[0],
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gW2_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCgC = thr_mma.partition_C(gC_mnl)

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

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=cutlass.BFloat16,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
            cutlass.BFloat16,
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
        tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)

        k_tile_cnt = cute.size(mA_mkl, mode=[1]) // self.tile_shape_mnk[2]
        inter_tile_cnt = cute.size(mW2_nkl, mode=[1]) // self.tile_shape_mnk[2]
        num_k_blocks = cute.size(tCrA, mode=[2])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        epi_n_scale = self.tile_shape_mnk[1] // self.epi_tile[1]
        mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rD, mode=[1])
        mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rD, mode=[2])
        mma_m_per_epi_m = self.epi_tile[0] // mma_tile_m
        mma_n_per_epi_n = self.epi_tile[1] // mma_tile_n
        mma_thread_count = cutlass.Int32(
            self.num_mma_warps * self.num_threads_per_warp
        )

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        if warp_idx < self.num_mma_warps:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)

            gate_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
            out_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
            tRS_rGate = tiled_copy_r2s.retile(gate_acc)
            fc2_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tile_m = cute.arch.make_warp_uniform(tile_coord_mnl[0])
                tile_n = cute.arch.make_warp_uniform(tile_coord_mnl[1])
                tile_l = cute.arch.make_warp_uniform(tile_coord_mnl[2])
                weight_expert_idx = cute.arch.make_warp_uniform(
                    expert_ids[tile_l].to(cutlass.Int32)
                )
                if weight_expert_idx >= cutlass.Int32(0):
                    gC_mnl_slice = gC_mnl[(None, None, tile_m, tile_n, tile_l)]
                    out_acc.fill(0.0)
                    fc2_cons_state.reset_count()
                    for _inter_tile_idx in range(0, inter_tile_cnt, 1, unroll=1):
                        inter_tile_idx = cutlass.Int32(_inter_tile_idx)
                        fc1_cons_state = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Consumer, self.ab_stage
                        )
                        _consumer_dense_pass(
                            tiled_mma,
                            fc1_pipeline,
                            fc1_cons_state,
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
                            gate_acc,
                        )
                        self.pass_sync_barrier.arrive_and_wait()

                        for epi_n in cutlass.range_constexpr(epi_n_scale):
                            for epi_m in cutlass.range_constexpr(epi_m_scale):
                                epi_idx = epi_m * epi_n_scale + epi_n
                                epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                                tRS_rD.fill(0.0)
                                for mma_n_in_epi in cutlass.range_constexpr(
                                    mma_n_per_epi_n
                                ):
                                    for mma_m_in_epi in cutlass.range_constexpr(
                                        mma_m_per_epi_m
                                    ):
                                        mma_m = (
                                            epi_m * mma_m_per_epi_m + mma_m_in_epi
                                        )
                                        mma_n = (
                                            epi_n * mma_n_per_epi_n + mma_n_in_epi
                                        )
                                        tRS_rD_slice = tRS_rD[
                                            (None, mma_m_in_epi, mma_n_in_epi)
                                        ]
                                        gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size(tRS_rD_slice)
                                        ):
                                            g = cutlass.Float32(
                                                cutlass.BFloat16(gate_slice[elem_idx])
                                            )
                                            relu_g = fmax_f32(
                                                g, cutlass.Float32(0.0)
                                            )
                                            tRS_rD_slice[elem_idx] = relu_g * relu_g

                                tRS_rD_out.store(tRS_rD.load().to(cutlass.BFloat16))
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rD_out,
                                    tRS_sD[(None, None, None, epi_buffer)],
                                )
                                cute.arch.fence_proxy("async.shared", space="cta")
                                self.epilog_sync_barrier.arrive_and_wait()
                                copy_idx = cutlass.Int32(tidx)
                                epi_rows = cutlass.Int32(self.epi_tile[0])
                                epi_cols = cutlass.Int32(self.epi_tile[1])
                                row_base = (
                                    cutlass.Int32(epi_m) * epi_rows
                                )
                                col_base = cutlass.Int32(epi_n) * epi_cols
                                total_copy = epi_rows * epi_cols
                                while copy_idx < total_copy:
                                    local_row = copy_idx // epi_cols
                                    col = copy_idx - local_row * epi_cols
                                    sA[row_base + local_row, col_base + col, 0] = sC[
                                        local_row, col, epi_buffer
                                    ]
                                    copy_idx += mma_thread_count
                                cute.arch.fence_proxy("async.shared", space="cta")
                                self.epilog_sync_barrier.arrive_and_wait()
                        self.epilog_sync_barrier.arrive_and_wait()
                        self.pass_sync_barrier.arrive_and_wait()

                        phase2_peek = fc2_pipeline.consumer_try_wait(fc2_cons_state)
                        fc2_pipeline.consumer_wait(fc2_cons_state, phase2_peek)
                        csB_phase2 = tCsB_copy_view[
                            None, None, None, fc2_cons_state.index
                        ]
                        csA_phase2 = tCsA_copy_view[None, None, None, 0]
                        _warp_mma_gemm(
                            tiled_mma,
                            out_acc,
                            tCrA,
                            tCrB,
                            csA_phase2,
                            csB_phase2,
                            smem_tiled_copy_A,
                            smem_tiled_copy_B,
                        )
                        fc2_pipeline.consumer_release(fc2_cons_state)
                        fc2_cons_state.advance()
                        self.pass_sync_barrier.arrive_and_wait()

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
                    tma_store_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.epi_stage,
                        producer_group=pipeline.CooperativeGroup(
                            pipeline.Agent.Thread,
                            self.num_mma_warps * self.num_threads_per_warp,
                        ),
                    )

                    tRS_rAcc = tiled_copy_r2s.retile(out_acc)
                    for epi_idx in cutlass.range_constexpr(epi_tile_num):
                        for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
                            tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]
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
                self.pass_sync_barrier.arrive_and_wait()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        elif warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)
            fc2_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tile_m = cute.arch.make_warp_uniform(tile_coord_mnl[0])
                tile_n = cute.arch.make_warp_uniform(tile_coord_mnl[1])
                tile_l = cute.arch.make_warp_uniform(tile_coord_mnl[2])
                weight_expert_idx = cute.arch.make_warp_uniform(
                    expert_ids[tile_l].to(cutlass.Int32)
                )
                if weight_expert_idx >= cutlass.Int32(0):
                    tAgA_mkl = tAgA[(None, tile_m, None, tile_l)]
                    fc2_prod_state.reset_count()
                    for _inter_tile_idx in range(0, inter_tile_cnt, 1, unroll=1):
                        inter_tile_idx = cutlass.Int32(_inter_tile_idx)
                        fc1_prod_state = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Producer, self.ab_stage
                        )
                        _producer_load_dense_pass(
                            fc1_pipeline,
                            fc1_prod_state,
                            k_tile_cnt,
                            tma_atom_a,
                            tAgA_mkl,
                            tAsA,
                            tma_atom_w1,
                            tBgW1[
                                (None, inter_tile_idx, None, weight_expert_idx)
                            ],
                            tBsW1,
                        )
                        self.pass_sync_barrier.arrive_and_wait()

                        # Keep the barrier cadence aligned with the MMA warps:
                        # after FC1 they spend one phase materializing the
                        # relu2-activated intermediate into shared memory
                        # before phase2 starts consuming W2.
                        self.pass_sync_barrier.arrive_and_wait()

                        fc2_pipeline.producer_acquire(fc2_prod_state)
                        cute.copy(
                            tma_atom_w2,
                            tBgW2[
                                (
                                    None,
                                    tile_n,
                                    inter_tile_idx,
                                    weight_expert_idx,
                                )
                            ],
                            tBsW2[(None, fc2_prod_state.index)],
                            tma_bar_ptr=fc2_pipeline.producer_get_barrier(
                                fc2_prod_state
                            ),
                        )
                        fc2_pipeline.producer_commit(fc2_prod_state)
                        fc2_prod_state.advance()
                        self.pass_sync_barrier.arrive_and_wait()
                    fc2_pipeline.producer_tail(fc2_prod_state)
                self.pass_sync_barrier.arrive_and_wait()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        return


class _ExpertIndexedFusedRelu2GridKernel(_ExpertIndexedFusedRelu2Kernel):
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        w1: cute.Tensor,
        w2: cute.Tensor,
        expert_ids: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        del max_active_clusters
        self.a_dtype = a.element_type
        self.b_dtype = w1.element_type
        self.expert_ids_dtype = expert_ids.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(w1)
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
        tma_atom_w1, tma_tensor_w1 = self._get_or_make_tma_load(
            w1,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_w2, tma_tensor_w2 = self._get_or_make_tma_load(
            w2,
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
            fc1_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            fc2_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, 2]
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
            tma_atom_w1,
            tma_tensor_w1,
            tma_atom_w2,
            tma_tensor_w2,
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
        tma_atom_w1: cute.CopyAtom,
        mW1_nkl: cute.Tensor,
        tma_atom_w2: cute.CopyAtom,
        mW2_nkl: cute.Tensor,
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
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        weight_expert_idx = cute.arch.make_warp_uniform(
            expert_ids[bidz].to(cutlass.Int32)
        )

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        fc1_tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        fc2_tma_copy_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        fc1_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps,
            ),
            tx_count=fc1_tma_copy_bytes,
            barrier_storage=storage.fc1_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        fc2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps,
            ),
            tx_count=fc2_tma_copy_bytes,
            barrier_storage=storage.fc2_pipeline_array_ptr.data_ptr(),
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
        gW1_nkl = cute.local_tile(
            mW1_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gW2_nkl = cute.local_tile(
            mW2_nkl,
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
        tBsW1, tBgW1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w1,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gW1_nkl, 0, 2),
        )
        tBsW2, tBgW2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w2,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gW2_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCgC = thr_mma.partition_C(gC_mnl_tile)
        out_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        out_acc.fill(0.0)
        gate_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

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

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=cutlass.BFloat16,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
            cutlass.BFloat16,
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
        tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)
        tRS_rGate = tiled_copy_r2s.retile(gate_acc)

        k_tile_cnt = cute.size(mA_mkl, mode=[1]) // self.tile_shape_mnk[2]
        inter_tile_cnt = cute.size(mW2_nkl, mode=[1]) // self.tile_shape_mnk[2]
        num_k_blocks = cute.size(tCrA, mode=[2])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        epi_n_scale = self.tile_shape_mnk[1] // self.epi_tile[1]
        mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
        mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rGate, mode=[2])
        mma_m_per_epi_m = self.epi_tile[0] // mma_tile_m
        mma_n_per_epi_n = self.epi_tile[1] // mma_tile_n
        mma_thread_count = cutlass.Int32(
            self.num_mma_warps * self.num_threads_per_warp
        )

        if warp_idx < self.num_mma_warps:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)

            fc2_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            fc2_cons_state.reset_count()

            for _inter_tile_idx in range(0, inter_tile_cnt, 1, unroll=1):
                inter_tile_idx = cutlass.Int32(_inter_tile_idx)
                fc1_cons_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                _consumer_dense_pass(
                    tiled_mma,
                    fc1_pipeline,
                    fc1_cons_state,
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
                    gate_acc,
                )
                self.pass_sync_barrier.arrive_and_wait()

                for epi_n in cutlass.range_constexpr(epi_n_scale):
                    for epi_m in cutlass.range_constexpr(epi_m_scale):
                        epi_idx = epi_m * epi_n_scale + epi_n
                        epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                        tRS_rD.fill(0.0)
                        for mma_n_in_epi in cutlass.range_constexpr(mma_n_per_epi_n):
                            for mma_m_in_epi in cutlass.range_constexpr(mma_m_per_epi_m):
                                mma_m = epi_m * mma_m_per_epi_m + mma_m_in_epi
                                mma_n = epi_n * mma_n_per_epi_n + mma_n_in_epi
                                tRS_rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                                gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    g = cutlass.Float32(
                                        cutlass.BFloat16(gate_slice[elem_idx])
                                    )
                                    relu_g = fmax_f32(g, cutlass.Float32(0.0))
                                    tRS_rD_slice[elem_idx] = relu_g * relu_g

                        tRS_rD_out.store(tRS_rD.load().to(cutlass.BFloat16))
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        copy_idx = cutlass.Int32(tidx)
                        epi_rows = cutlass.Int32(self.epi_tile[0])
                        epi_cols = cutlass.Int32(self.epi_tile[1])
                        row_base = cutlass.Int32(epi_m) * epi_rows
                        col_base = cutlass.Int32(epi_n) * epi_cols
                        total_copy = epi_rows * epi_cols
                        while copy_idx < total_copy:
                            local_row = copy_idx // epi_cols
                            col = copy_idx - local_row * epi_cols
                            sA[row_base + local_row, col_base + col, 0] = sC[
                                local_row, col, epi_buffer
                            ]
                            copy_idx += mma_thread_count
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                self.epilog_sync_barrier.arrive_and_wait()
                self.pass_sync_barrier.arrive_and_wait()

                phase2_peek = fc2_pipeline.consumer_try_wait(fc2_cons_state)
                fc2_pipeline.consumer_wait(fc2_cons_state, phase2_peek)
                csB_phase2 = tCsB_copy_view[None, None, None, fc2_cons_state.index]
                csA_phase2 = tCsA_copy_view[None, None, None, 0]
                _warp_mma_gemm(
                    tiled_mma,
                    out_acc,
                    tCrA,
                    tCrB,
                    csA_phase2,
                    csB_phase2,
                    smem_tiled_copy_A,
                    smem_tiled_copy_B,
                )
                fc2_pipeline.consumer_release(fc2_cons_state)
                fc2_cons_state.advance()
                self.pass_sync_barrier.arrive_and_wait()

            sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
            tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_tile, self.epi_tile)
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

            tRS_rAcc = tiled_copy_r2s.retile(out_acc)
            for epi_idx in cutlass.range_constexpr(epi_tile_num):
                for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
                    tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]
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

        elif warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)

            fc2_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            fc2_prod_state.reset_count()
            tAgA_mkl = tAgA[(None, bidx, None, bidz)]
            for _inter_tile_idx in range(0, inter_tile_cnt, 1, unroll=1):
                inter_tile_idx = cutlass.Int32(_inter_tile_idx)
                fc1_prod_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                _producer_load_dense_pass(
                    fc1_pipeline,
                    fc1_prod_state,
                    k_tile_cnt,
                    tma_atom_a,
                    tAgA_mkl,
                    tAsA,
                    tma_atom_w1,
                    tBgW1[(None, inter_tile_idx, None, weight_expert_idx)],
                    tBsW1,
                )
                self.pass_sync_barrier.arrive_and_wait()
                self.pass_sync_barrier.arrive_and_wait()

                fc2_pipeline.producer_acquire(fc2_prod_state)
                cute.copy(
                    tma_atom_w2,
                    tBgW2[(None, bidy, inter_tile_idx, weight_expert_idx)],
                    tBsW2[(None, fc2_prod_state.index)],
                    tma_bar_ptr=fc2_pipeline.producer_get_barrier(fc2_prod_state),
                )
                fc2_pipeline.producer_commit(fc2_prod_state)
                fc2_prod_state.advance()
                self.pass_sync_barrier.arrive_and_wait()

            fc2_pipeline.producer_tail(fc2_prod_state)
        return


class _ExpertIndexedFusedRelu2FlatPersistentKernel(_ExpertIndexedFusedRelu2Kernel):
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        w1: cute.Tensor,
        w2: cute.Tensor,
        expert_ids: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = w1.element_type
        self.expert_ids_dtype = expert_ids.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(w1)
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
        tma_atom_w1, tma_tensor_w1 = self._get_or_make_tma_load(
            w1,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_w2, tma_tensor_w2 = self._get_or_make_tma_load(
            w2,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_c, tma_tensor_c = self._get_or_make_tma_store(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        num_tiles_m = (a.shape[0] + self.tile_shape_mnk[0] - 1) // self.tile_shape_mnk[0]
        num_tiles_n = c.shape[1] // self.tile_shape_mnk[1]
        num_tiles_l = c.shape[2]
        total_tiles = num_tiles_m * num_tiles_n * num_tiles_l
        grid = (min(max_active_clusters, total_tiles), 1, 1)

        @cute.struct
        class SharedStorage:
            fc1_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            fc2_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, 2]
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
            tma_atom_w1,
            tma_tensor_w1,
            tma_atom_w2,
            tma_tensor_w2,
            tma_atom_c,
            tma_tensor_c,
            total_tiles,
            num_tiles_n,
            num_tiles_l,
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
        tma_atom_w1: cute.CopyAtom,
        mW1_nkl: cute.Tensor,
        tma_atom_w2: cute.CopyAtom,
        mW2_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        total_tiles,
        num_tiles_n,
        num_tiles_l,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        fc1_tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        fc2_tma_copy_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        fc1_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps,
            ),
            tx_count=fc1_tma_copy_bytes,
            barrier_storage=storage.fc1_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        fc2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps,
            ),
            tx_count=fc2_tma_copy_bytes,
            barrier_storage=storage.fc2_pipeline_array_ptr.data_ptr(),
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
        gW1_nkl = cute.local_tile(
            mW1_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gW2_nkl = cute.local_tile(
            mW2_nkl,
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
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        tBsW1, tBgW1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w1,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gW1_nkl, 0, 2),
        )
        tBsW2, tBgW2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w2,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gW2_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCgC = thr_mma.partition_C(gC_mnl)
        out_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        gate_acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

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

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=cutlass.BFloat16,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
            cutlass.BFloat16,
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
        tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)
        tRS_rGate = tiled_copy_r2s.retile(gate_acc)

        k_tile_cnt = cute.size(mA_mkl, mode=[1]) // self.tile_shape_mnk[2]
        inter_tile_cnt = cute.size(mW2_nkl, mode=[1]) // self.tile_shape_mnk[2]
        num_k_blocks = cute.size(tCrA, mode=[2])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        epi_n_scale = self.tile_shape_mnk[1] // self.epi_tile[1]
        mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
        mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rGate, mode=[2])
        mma_m_per_epi_m = self.epi_tile[0] // mma_tile_m
        mma_n_per_epi_n = self.epi_tile[1] // mma_tile_n
        mma_thread_count = cutlass.Int32(
            self.num_mma_warps * self.num_threads_per_warp
        )

        tile_linear_idx = cutlass.Int32(cute.arch.block_idx()[0])
        tile_stride = cutlass.Int32(cute.arch.grid_dim()[0])
        tiles_per_m = cutlass.Int32(num_tiles_n * num_tiles_l)

        if warp_idx < self.num_mma_warps:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)

            while tile_linear_idx < total_tiles:
                tile_m = cute.arch.make_warp_uniform(tile_linear_idx // tiles_per_m)
                tile_rem = tile_linear_idx - tile_m * tiles_per_m
                tile_l = cute.arch.make_warp_uniform(tile_rem // num_tiles_n)
                tile_n = cute.arch.make_warp_uniform(tile_rem - tile_l * num_tiles_n)
                weight_expert_idx = cute.arch.make_warp_uniform(
                    expert_ids[tile_l].to(cutlass.Int32)
                )

                if weight_expert_idx >= cutlass.Int32(0):
                    gC_mnl_slice = gC_mnl[(None, None, tile_m, tile_n, tile_l)]
                    out_acc.fill(0.0)
                    fc2_cons_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, 1
                    )
                    fc2_cons_state.reset_count()

                    for _inter_tile_idx in range(0, inter_tile_cnt, 1, unroll=1):
                        inter_tile_idx = cutlass.Int32(_inter_tile_idx)
                        fc1_cons_state = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Consumer, self.ab_stage
                        )
                        _consumer_dense_pass(
                            tiled_mma,
                            fc1_pipeline,
                            fc1_cons_state,
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
                            gate_acc,
                        )
                        self.pass_sync_barrier.arrive_and_wait()

                        for epi_n in cutlass.range_constexpr(epi_n_scale):
                            for epi_m in cutlass.range_constexpr(epi_m_scale):
                                epi_idx = epi_m * epi_n_scale + epi_n
                                epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                                tRS_rD.fill(0.0)
                                for mma_n_in_epi in cutlass.range_constexpr(mma_n_per_epi_n):
                                    for mma_m_in_epi in cutlass.range_constexpr(mma_m_per_epi_m):
                                        mma_m = epi_m * mma_m_per_epi_m + mma_m_in_epi
                                        mma_n = epi_n * mma_n_per_epi_n + mma_n_in_epi
                                        tRS_rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                                        gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size(tRS_rD_slice)
                                        ):
                                            g = cutlass.Float32(
                                                cutlass.BFloat16(gate_slice[elem_idx])
                                            )
                                            relu_g = fmax_f32(g, cutlass.Float32(0.0))
                                            tRS_rD_slice[elem_idx] = relu_g * relu_g

                                tRS_rD_out.store(tRS_rD.load().to(cutlass.BFloat16))
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rD_out,
                                    tRS_sD[(None, None, None, epi_buffer)],
                                )
                                cute.arch.fence_proxy("async.shared", space="cta")
                                self.epilog_sync_barrier.arrive_and_wait()
                                copy_idx = cutlass.Int32(tidx)
                                epi_rows = cutlass.Int32(self.epi_tile[0])
                                epi_cols = cutlass.Int32(self.epi_tile[1])
                                row_base = cutlass.Int32(epi_m) * epi_rows
                                col_base = cutlass.Int32(epi_n) * epi_cols
                                total_copy = epi_rows * epi_cols
                                while copy_idx < total_copy:
                                    local_row = copy_idx // epi_cols
                                    col = copy_idx - local_row * epi_cols
                                    sA[row_base + local_row, col_base + col, 0] = sC[
                                        local_row, col, epi_buffer
                                    ]
                                    copy_idx += mma_thread_count
                                cute.arch.fence_proxy("async.shared", space="cta")
                                self.epilog_sync_barrier.arrive_and_wait()
                        self.epilog_sync_barrier.arrive_and_wait()
                        self.pass_sync_barrier.arrive_and_wait()

                        phase2_peek = fc2_pipeline.consumer_try_wait(fc2_cons_state)
                        fc2_pipeline.consumer_wait(fc2_cons_state, phase2_peek)
                        csB_phase2 = tCsB_copy_view[None, None, None, fc2_cons_state.index]
                        csA_phase2 = tCsA_copy_view[None, None, None, 0]
                        _warp_mma_gemm(
                            tiled_mma,
                            out_acc,
                            tCrA,
                            tCrB,
                            csA_phase2,
                            csB_phase2,
                            smem_tiled_copy_A,
                            smem_tiled_copy_B,
                        )
                        fc2_pipeline.consumer_release(fc2_cons_state)
                        fc2_cons_state.advance()
                        self.pass_sync_barrier.arrive_and_wait()

                    sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                    tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)
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

                    tRS_rAcc = tiled_copy_r2s.retile(out_acc)
                    for epi_idx in cutlass.range_constexpr(epi_tile_num):
                        for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
                            tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]
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

                self.pass_sync_barrier.arrive_and_wait()
                tile_linear_idx += tile_stride

        elif warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)
            fc2_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )

            while tile_linear_idx < total_tiles:
                tile_m = cute.arch.make_warp_uniform(tile_linear_idx // tiles_per_m)
                tile_rem = tile_linear_idx - tile_m * tiles_per_m
                tile_l = cute.arch.make_warp_uniform(tile_rem // num_tiles_n)
                tile_n = cute.arch.make_warp_uniform(tile_rem - tile_l * num_tiles_n)
                weight_expert_idx = cute.arch.make_warp_uniform(
                    expert_ids[tile_l].to(cutlass.Int32)
                )

                if weight_expert_idx >= cutlass.Int32(0):
                    fc2_prod_state.reset_count()
                    tAgA_mkl = tAgA[(None, tile_m, None, tile_l)]
                    for _inter_tile_idx in range(0, inter_tile_cnt, 1, unroll=1):
                        inter_tile_idx = cutlass.Int32(_inter_tile_idx)
                        fc1_prod_state = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Producer, self.ab_stage
                        )
                        _producer_load_dense_pass(
                            fc1_pipeline,
                            fc1_prod_state,
                            k_tile_cnt,
                            tma_atom_a,
                            tAgA_mkl,
                            tAsA,
                            tma_atom_w1,
                            tBgW1[(None, inter_tile_idx, None, weight_expert_idx)],
                            tBsW1,
                        )
                        self.pass_sync_barrier.arrive_and_wait()
                        self.pass_sync_barrier.arrive_and_wait()

                        fc2_pipeline.producer_acquire(fc2_prod_state)
                        cute.copy(
                            tma_atom_w2,
                            tBgW2[(None, tile_n, inter_tile_idx, weight_expert_idx)],
                            tBsW2[(None, fc2_prod_state.index)],
                            tma_bar_ptr=fc2_pipeline.producer_get_barrier(fc2_prod_state),
                        )
                        fc2_pipeline.producer_commit(fc2_prod_state)
                        fc2_prod_state.advance()
                        self.pass_sync_barrier.arrive_and_wait()

                self.pass_sync_barrier.arrive_and_wait()
                tile_linear_idx += tile_stride
            fc2_pipeline.producer_tail(fc2_prod_state)
        return


def run_fused_relu2_bf16_expert_ids(
    kernel: _ExpertIndexedFusedRelu2Kernel,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
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
        _to_dense_kernel_tensor(w1),
        _to_dense_kernel_tensor(w2),
        _to_dense_kernel_tensor(expert_ids, cutlass.Int32, assumed_align=4),
        _to_dense_kernel_tensor(c),
        max_active_clusters,
        stream,
    )
    cache_key = (
        tuple(kernel.tile_shape_mnk),
        _tensor_meta_key(a),
        _tensor_meta_key(w1),
        _tensor_meta_key(w2),
        _tensor_meta_key(expert_ids),
        _tensor_meta_key(c),
        max_active_clusters,
    )
    _run_cached_host_launcher(kernel, cache_key, args)


class MoEStaticKernelBackend:
    implementation = "static"
    default_expert_chunk_size = 8
    relu2_expert_chunk_size = 88
    vectorized_row_limit = 4
    compact_direct_routed_rows_limit = 176
    row1_dense_max_active_clusters = 96
    row1_multi_token_dense_max_active_clusters = 128

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        exact_mma_m_tiles: bool = False,
        input_scales_are_reciprocal: bool = False,
        activation: str = "silu",
    ):
        if activation not in {"silu", "relu2"}:
            raise ValueError(f"unsupported activation {activation!r}")
        self.sf_vec_size = sf_vec_size
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.output_tile_count_n = output_tile_count_n
        self.exact_mma_m_tiles = exact_mma_m_tiles
        self.input_scales_are_reciprocal = input_scales_are_reciprocal
        self.fast_math = True
        self.activation = activation
        self.is_gated = activation == "silu"
        self.tile_shape_mnk = _FUSED_TILE_SHAPE_MNK
        self._fused_kernel: _FusedChunkKernel | None = None
        self._phase1_kernel: _FC1ActivationChunkKernel | None = None
        self._fc1_dense_kernel: DenseGemmKernel | None = None
        self._fc2_dense_kernel: DenseGemmKernel | None = None
        self._indexed_fc1_dense_kernel = None
        self._indexed_fc2_dense_kernel = None
        self._row1_indexed_fc1_dense_kernel = None
        self._row1_indexed_fc2_dense_kernel = None
        self._row1_indexed_runtime_key: tuple[
            tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
        ] | None = None
        self._indexed_fused_relu2_kernel: _ExpertIndexedFusedRelu2Kernel | None = None
        self._indexed_fused_relu2_variant: str | None = None
        self._indexed_max_active_clusters: int | None = None
        self._indexed_device_index: int | None = None
        self._max_active_clusters: int | None = None
        self._device_index: int | None = None

    @property
    def expert_chunk_size(self) -> int:
        if self.activation == "relu2":
            return max(self.default_expert_chunk_size, self.relu2_expert_chunk_size)
        return self.default_expert_chunk_size

    def resolve_expert_chunk_size(self, *, num_tokens: int, num_topk: int) -> int:
        if self.activation != "relu2":
            return self.expert_chunk_size
        routed_rows = num_tokens * num_topk
        return max(self.expert_chunk_size, min(192, routed_rows))

    def _get_fused_runtime(self) -> _FusedChunkKernel:
        if self._fused_kernel is None:
            self._fused_kernel = _FusedChunkKernel(activation=self.activation)
        return self._fused_kernel

    def _get_phase1_runtime(self) -> _FC1ActivationChunkKernel:
        if self._phase1_kernel is None:
            self._phase1_kernel = _FC1ActivationChunkKernel(
                activation=self.activation
            )
        return self._phase1_kernel

    def _get_dense_runtime(
        self,
        device: torch.device,
    ) -> tuple[DenseGemmKernel, DenseGemmKernel, int]:
        device_index = device.index or 0
        if self._fc1_dense_kernel is None or self._device_index != device_index:
            if self.activation == "relu2":
                self._fc1_dense_kernel = DenseGemmKernel(
                    (16, 128, 64),
                    epilogue="relu2",
                )
                self._fc2_dense_kernel = DenseGemmKernel((16, 64, 64))
                self._fc1_dense_kernel.configure_atom_layout((1, 2, 1))
                self._fc2_dense_kernel.configure_atom_layout((1, 2, 1))
                self._max_active_clusters = min(120, get_num_sm(device))
            else:
                self._fc1_dense_kernel = DenseGemmKernel((128, 128, 64))
                self._fc2_dense_kernel = self._fc1_dense_kernel
                self._max_active_clusters = get_num_sm(device)
            self._device_index = device_index
        return (
            self._fc1_dense_kernel,  # type: ignore[return-value]
            self._fc2_dense_kernel,  # type: ignore[return-value]
            self._max_active_clusters,  # type: ignore[return-value]
        )

    def _get_indexed_dense_runtime(
        self,
        device: torch.device,
    ) -> tuple[object, object, int]:
        from b12x.moe.fused.bf16.indexed_dense import ExpertIndexedDenseGemmKernel

        device_index = device.index or 0
        if (
            self._indexed_fc1_dense_kernel is None
            or self._indexed_fc2_dense_kernel is None
            or self._indexed_device_index != device_index
        ):
            fc1_tile = _parse_int_tuple_env(
                "B12X_BF16_INDEXED_FC1_TILE_MNK", (16, 128, 64)
            )
            fc2_tile = _parse_int_tuple_env(
                "B12X_BF16_INDEXED_FC2_TILE_MNK", (16, 64, 64)
            )
            fc1_atom = _parse_int_tuple_env(
                "B12X_BF16_INDEXED_FC1_ATOM_LAYOUT", (1, 2, 1)
            )
            fc2_atom = _parse_int_tuple_env(
                "B12X_BF16_INDEXED_FC2_ATOM_LAYOUT", (1, 2, 1)
            )
            self._indexed_fc1_dense_kernel = ExpertIndexedDenseGemmKernel(
                fc1_tile,
                epilogue="relu2",
            )
            self._indexed_fc2_dense_kernel = ExpertIndexedDenseGemmKernel(fc2_tile)
            self._indexed_fc1_dense_kernel.configure_atom_layout(fc1_atom)
            self._indexed_fc2_dense_kernel.configure_atom_layout(fc2_atom)
            self._indexed_fc1_dense_kernel.occupancy = _parse_int_env(
                "B12X_BF16_INDEXED_FC1_OCCUPANCY", 1
            )
            self._indexed_fc2_dense_kernel.occupancy = _parse_int_env(
                "B12X_BF16_INDEXED_FC2_OCCUPANCY", 1
            )
            self._indexed_max_active_clusters = min(
                _parse_int_env("B12X_BF16_INDEXED_MAX_ACTIVE_CLUSTERS", 120),
                get_num_sm(device),
            )
            self._indexed_device_index = device_index
        return (
            self._indexed_fc1_dense_kernel,
            self._indexed_fc2_dense_kernel,
            self._indexed_max_active_clusters,  # type: ignore[return-value]
        )

    def _get_row1_indexed_dense_runtime(
        self,
        device: torch.device,
    ) -> tuple[object, object, int]:
        from b12x.moe.fused.bf16.indexed_dense import ExpertIndexedDenseRow1GridKernel

        device_index = device.index or 0
        fc1_tile = _parse_int_tuple_env(
            "B12X_BF16_ROW1_INDEXED_FC1_TILE_MNK", (16, 128, 64)
        )
        fc2_tile = _parse_int_tuple_env(
            # The original (16, 64, 64) FC2 row1-grid tile can launch-fail on
            # SM120 in the relu2 direct path. Keep the experimental row1-grid
            # path on a shape that actually runs end-to-end by default.
            "B12X_BF16_ROW1_INDEXED_FC2_TILE_MNK", (16, 128, 64)
        )
        fc1_atom = _parse_int_tuple_env(
            "B12X_BF16_ROW1_INDEXED_FC1_ATOM_LAYOUT", (1, 2, 1)
        )
        fc2_atom = _parse_int_tuple_env(
            "B12X_BF16_ROW1_INDEXED_FC2_ATOM_LAYOUT", (1, 2, 1)
        )
        runtime_key = (fc1_tile, fc2_tile, fc1_atom, fc2_atom)
        if (
            self._row1_indexed_fc1_dense_kernel is None
            or self._row1_indexed_fc2_dense_kernel is None
            or self._indexed_device_index != device_index
            or self._row1_indexed_runtime_key != runtime_key
        ):
            self._row1_indexed_fc1_dense_kernel = ExpertIndexedDenseRow1GridKernel(
                fc1_tile,
                epilogue="relu2",
            )
            self._row1_indexed_fc2_dense_kernel = ExpertIndexedDenseRow1GridKernel(
                fc2_tile
            )
            self._row1_indexed_fc1_dense_kernel.configure_atom_layout(fc1_atom)
            self._row1_indexed_fc2_dense_kernel.configure_atom_layout(fc2_atom)
            self._row1_indexed_fc1_dense_kernel.occupancy = _parse_int_env(
                "B12X_BF16_ROW1_INDEXED_FC1_OCCUPANCY",
                _parse_int_env("B12X_BF16_INDEXED_FC1_OCCUPANCY", 1),
            )
            self._row1_indexed_fc2_dense_kernel.occupancy = _parse_int_env(
                "B12X_BF16_ROW1_INDEXED_FC2_OCCUPANCY",
                _parse_int_env("B12X_BF16_INDEXED_FC2_OCCUPANCY", 1),
            )
            self._indexed_max_active_clusters = min(120, get_num_sm(device))
            self._indexed_device_index = device_index
            self._row1_indexed_runtime_key = runtime_key
        return (
            self._row1_indexed_fc1_dense_kernel,
            self._row1_indexed_fc2_dense_kernel,
            self._indexed_max_active_clusters,  # type: ignore[return-value]
        )

    def _get_direct_expert_indexed_dense_runtime(
        self,
        device: torch.device,
    ) -> tuple[object, object, int]:
        (
            indexed_fc1_kernel,
            indexed_fc2_kernel,
            indexed_max_active_clusters,
        ) = self._get_indexed_dense_runtime(device)
        if not _ENABLE_ROW1_GRID_INDEXED_DENSE:
            return indexed_fc1_kernel, indexed_fc2_kernel, indexed_max_active_clusters

        (
            row1_fc1_kernel,
            row1_fc2_kernel,
            row1_max_active_clusters,
        ) = self._get_row1_indexed_dense_runtime(device)
        use_row1_fc1 = _parse_bool_env(
            "B12X_BF16_ENABLE_ROW1_GRID_INDEXED_FC1",
            True,
        )
        use_row1_fc2 = _parse_bool_env(
            "B12X_BF16_ENABLE_ROW1_GRID_INDEXED_FC2",
            True,
        )
        return (
            row1_fc1_kernel if use_row1_fc1 else indexed_fc1_kernel,
            row1_fc2_kernel if use_row1_fc2 else indexed_fc2_kernel,
            min(indexed_max_active_clusters, row1_max_active_clusters),
        )

    def _get_indexed_fused_relu2_runtime(
        self,
        device: torch.device,
    ) -> tuple[_ExpertIndexedFusedRelu2Kernel, int]:
        device_index = device.index or 0
        variant = _get_fused_direct_relu2_variant()
        if (
            self._indexed_fused_relu2_kernel is None
            or self._indexed_device_index != device_index
            or self._indexed_fused_relu2_variant != variant
        ):
            fused_tile = _parse_int_tuple_env(
                "B12X_BF16_FUSED_DIRECT_RELU2_TILE_MNK", (16, 64, 64)
            )
            fused_atom = _parse_int_tuple_env(
                "B12X_BF16_FUSED_DIRECT_RELU2_ATOM_LAYOUT", (1, 2, 1)
            )
            if variant == "flat_persistent":
                self._indexed_fused_relu2_kernel = (
                    _ExpertIndexedFusedRelu2FlatPersistentKernel(fused_tile)
                )
            else:
                self._indexed_fused_relu2_kernel = _ExpertIndexedFusedRelu2GridKernel(
                    fused_tile
                )
            self._indexed_fused_relu2_kernel.configure_atom_layout(fused_atom)
            self._indexed_fused_relu2_kernel.occupancy = _parse_int_env(
                "B12X_BF16_FUSED_DIRECT_RELU2_OCCUPANCY", 1
            )
            self._indexed_fused_relu2_variant = variant
            if (
                self._indexed_max_active_clusters is None
                or self._indexed_device_index != device_index
            ):
                self._indexed_max_active_clusters = min(120, get_num_sm(device))
            self._indexed_device_index = device_index
        return (
            self._indexed_fused_relu2_kernel,
            self._indexed_max_active_clusters,  # type: ignore[return-value]
        )

    def _resolve_max_active_clusters(
        self, *, max_active_clusters: int, routed_rows: int
    ) -> int:
        if self.activation != "relu2":
            return max_active_clusters
        if routed_rows <= 44:
            return max_active_clusters
        return min(96, max_active_clusters)

    def _get_direct_weight_views(
        self,
        *,
        workspace,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_key = (w1.data_ptr(), w2.data_ptr())
        if (
            workspace.direct_weight_key != weight_key
            or workspace.direct_w1_view is None
            or workspace.direct_w2_view is None
        ):
            # The expert-indexed CUTLASS path only accepts weight tensors that
            # stay row/col-major in the first two modes. The plain
            # permute(1, 2, 0) view preserves that leading-dimension contract;
            # a naive contiguous repack does not.
            workspace.direct_w1_view = w1.permute(1, 2, 0)
            workspace.direct_w2_view = w2.permute(1, 2, 0)
            workspace.direct_weight_key = weight_key
        return workspace.direct_w1_view, workspace.direct_w2_view

    def _should_use_compact_direct_route(
        self,
        *,
        route,
    ) -> bool:
        return (
            self.activation == "relu2"
            and route.routed_rows <= self.compact_direct_routed_rows_limit
            and route.flat_ids_i32 is not None
            and route.flat_weights is not None
            and route.flat_token_indices is not None
        )

    def _should_use_sorted_compact_direct_route(
        self,
        *,
        route,
    ) -> bool:
        return (
            self.activation == "relu2"
            and route.routed_rows >= _RELU2_SORTED_DIRECT_ROUTE_MIN_ROWS
            and (
                (
                    route.sorted_route_order_i64 is not None
                    and route.sorted_flat_ids_i32 is not None
                    and route.sorted_flat_token_indices is not None
                )
                or (
                    route.compact_topk_ids_i64 is not None
                    and route.route_row_indices_i64 is not None
                )
            )
        )

    def _build_sorted_compact_direct_route(
        self,
        *,
        route,
        workspace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            route.sorted_route_order_i64 is not None
            and route.sorted_flat_ids_i32 is not None
            and route.sorted_flat_token_indices is not None
        ):
            return (
                route.sorted_route_order_i64,
                route.sorted_flat_ids_i32,
                route.sorted_flat_token_indices,
            )

        if (
            workspace.compact_route_order_i64 is None
            or workspace.compact_route_order_i32 is None
            or workspace.compact_sorted_flat_ids_i32 is None
            or workspace.compact_pair_arange_i64 is None
            or workspace.compact_row_counts_i64 is None
            or workspace.compact_expert_offsets_i64 is None
            or workspace.compact_route_positions_i64 is None
        ):
            raise RuntimeError("compact direct route order scratch is not initialized")
        if route.flat_token_indices.dtype == torch.int32:
            if workspace.compact_sorted_flat_token_indices_i32 is None:
                raise RuntimeError("compact direct int32 token-index scratch is not initialized")
        elif workspace.compact_sorted_flat_token_indices is None:
            raise RuntimeError("compact direct int64 token-index scratch is not initialized")

        routed_rows = route.routed_rows
        state_e = route.row_counts.shape[0]
        row_counts_i64 = workspace.compact_row_counts_i64[:state_e]
        expert_offsets_i64 = workspace.compact_expert_offsets_i64[:state_e]
        route_positions_i64 = workspace.compact_route_positions_i64[:routed_rows]
        route_order_i64 = workspace.compact_route_order_i64[:routed_rows]
        route_order_i32 = workspace.compact_route_order_i32[:routed_rows]
        sorted_flat_ids_i32 = workspace.compact_sorted_flat_ids_i32[:routed_rows]
        if route.flat_token_indices.dtype == torch.int32:
            sorted_flat_token_indices = workspace.compact_sorted_flat_token_indices_i32[
                :routed_rows
            ]
        else:
            sorted_flat_token_indices = workspace.compact_sorted_flat_token_indices[
                :routed_rows
            ]
        compact_pair_arange_i64 = workspace.compact_pair_arange_i64[:routed_rows]

        row_counts_i64.copy_(route.row_counts)
        torch.cumsum(row_counts_i64, dim=0, out=expert_offsets_i64)
        expert_offsets_i64.sub_(row_counts_i64)
        compact_topk_ids_i64 = (
            route.compact_topk_ids_i64
            if route.compact_topk_ids_i64 is not None
            else route.compact_topk_ids.to(torch.int64)
        )
        route_row_indices_i64 = (
            route.route_row_indices_i64
            if route.route_row_indices_i64 is not None
            else route.route_row_indices.to(torch.int64)
        )
        torch.index_select(
            expert_offsets_i64,
            0,
            compact_topk_ids_i64,
            out=route_positions_i64,
        )
        route_positions_i64.add_(route_row_indices_i64)
        route_order_i64.scatter_(0, route_positions_i64, compact_pair_arange_i64)
        route_order_i32.copy_(route_order_i64)
        torch.index_select(
            route.flat_ids_i32,
            0,
            route_order_i64,
            out=sorted_flat_ids_i32,
        )
        torch.index_select(
            route.flat_token_indices,
            0,
            route_order_i64,
            out=sorted_flat_token_indices,
        )
        return route_order_i32, sorted_flat_ids_i32, sorted_flat_token_indices

    def _build_sorted_compact_singleton_direct_route(
        self,
        *,
        route,
        workspace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            workspace.compact_route_order_i64 is None
            or workspace.compact_sorted_flat_ids_i32 is None
        ):
            raise RuntimeError("compact singleton direct route scratch is not initialized")
        if route.flat_token_indices.dtype == torch.int32:
            if workspace.compact_sorted_flat_token_indices_i32 is None:
                raise RuntimeError(
                    "compact singleton direct int32 token-index scratch is not initialized"
                )
            sorted_flat_token_indices = workspace.compact_sorted_flat_token_indices_i32[
                : route.routed_rows
            ]
        else:
            if workspace.compact_sorted_flat_token_indices is None:
                raise RuntimeError(
                    "compact singleton direct int64 token-index scratch is not initialized"
                )
            sorted_flat_token_indices = workspace.compact_sorted_flat_token_indices[
                : route.routed_rows
            ]

        route_order_i64 = workspace.compact_route_order_i64[: route.routed_rows]
        sorted_flat_ids_i32 = workspace.compact_sorted_flat_ids_i32[: route.routed_rows]
        build_compact_route_sorted_singleton_direct_state(
            route.flat_ids_i32,
            route.flat_token_indices,
            route_order_i64,
            sorted_flat_ids_i32,
            sorted_flat_token_indices,
        )
        return route_order_i64, sorted_flat_ids_i32, sorted_flat_token_indices

    def _run_compact_direct_route(
        self,
        *,
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        route,
        workspace,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if (
            workspace.micro_topk_weights_bf16 is None
            or workspace.micro_row1_routed_input_chunk is None
            or workspace.micro_row1_fc1_output_chunk is None
            or workspace.micro_row1_fc2_output_chunk is None
        ):
            raise RuntimeError("relu2 compact direct workspace scratch is not initialized")

        num_tokens, hidden_size = a.shape
        num_topk = topk_ids.shape[1]
        routed_rows = route.routed_rows
        _ensure_bmm_cublas_ready(a.device)
        get_direct_runtime = getattr(self, "get_expert_indexed_dense_runtime", None)
        if callable(get_direct_runtime):
            (
                fc1_dense_kernel,
                fc2_dense_kernel,
                max_active_clusters,
            ) = get_direct_runtime(a.device)
        else:
            (
                fc1_dense_kernel,
                fc2_dense_kernel,
                max_active_clusters,
            ) = self._get_direct_expert_indexed_dense_runtime(a.device)
        fused_direct_kernel = None
        if self.activation == "relu2" and _ENABLE_FUSED_DIRECT_RELU2:
            fused_direct_kernel, fused_max_active_clusters = (
                self._get_indexed_fused_relu2_runtime(a.device)
            )
            max_active_clusters = min(max_active_clusters, fused_max_active_clusters)
        from b12x.moe.fused.bf16.indexed_dense import run_dense_bf16_expert_ids

        dense_cluster_limit = min(
            max_active_clusters,
            self.row1_multi_token_dense_max_active_clusters
            if num_tokens > 1
            else self.row1_dense_max_active_clusters,
        )
        fc1_dense_cluster_limit = min(
            dense_cluster_limit,
            _parse_optional_int_env("B12X_BF16_INDEXED_FC1_MAX_ACTIVE_CLUSTERS")
            or dense_cluster_limit,
        )
        fc2_dense_cluster_limit = min(
            dense_cluster_limit,
            _parse_optional_int_env("B12X_BF16_INDEXED_FC2_MAX_ACTIVE_CLUSTERS")
            or dense_cluster_limit,
        )
        direct_w1_view, direct_w2_view = self._get_direct_weight_views(
            workspace=workspace,
            w1=w1,
            w2=w2,
        )

        topk_weights_bf16 = workspace.micro_topk_weights_bf16[:num_tokens, :num_topk]
        row1_routed_dense = workspace.micro_row1_routed_input_chunk[:, :, :routed_rows]
        row1_routed_dense_rows = row1_routed_dense[0].transpose(0, 1)
        fc1_dense = workspace.micro_row1_fc1_output_chunk[:, :, :routed_rows]
        fc2_dense = workspace.micro_row1_fc2_output_chunk[:, :, :routed_rows]
        route_order_i64 = None
        dense_expert_ids_i32 = route.flat_ids_i32
        routed_token_indices = route.flat_token_indices

        if self._should_use_sorted_compact_direct_route(route=route):
            (
                route_order_i64,
                dense_expert_ids_i32,
                routed_token_indices,
            ) = self._build_sorted_compact_direct_route(
                route=route,
                workspace=workspace,
            )
        workspace.direct_route_expert_ids_i32 = dense_expert_ids_i32

        gather_rows_bf16(
            a,
            routed_token_indices,
            row1_routed_dense_rows,
        )
        topk_weights_bf16.copy_(route.flat_weights.view(num_tokens, num_topk))

        if fused_direct_kernel is not None:
            run_fused_relu2_bf16_expert_ids(
                fused_direct_kernel,
                row1_routed_dense,
                direct_w1_view,
                direct_w2_view,
                dense_expert_ids_i32,
                fc2_dense,
                dense_cluster_limit,
                current_cuda_stream(),
            )
        else:
            run_dense_bf16_expert_ids(
                fc1_dense_kernel,
                row1_routed_dense,
                direct_w1_view,
                dense_expert_ids_i32,
                fc1_dense,
                fc1_dense_cluster_limit,
                current_cuda_stream(),
            )
            run_dense_bf16_expert_ids(
                fc2_dense_kernel,
                fc1_dense,
                direct_w2_view,
                dense_expert_ids_i32,
                fc2_dense,
                fc2_dense_cluster_limit,
                current_cuda_stream(),
            )

        route_output_flat = workspace.routed_output_unsorted[:routed_rows]
        if route_order_i64 is None:
            route_output_flat.copy_(fc2_dense[0].transpose(0, 1))
        else:
            permute_rows_bf16(
                fc2_dense[0].transpose(0, 1),
                route_order_i64,
                route_output_flat,
            )
        route_outputs = route_output_flat.view(num_tokens, num_topk, hidden_size)
        if num_topk > 2:
            torch.bmm(
                topk_weights_bf16.unsqueeze(1),
                route_outputs,
                out=output.unsqueeze(1),
            )
            return output

        routed_output_tmp = workspace.routed_output_sorted[:num_tokens]
        output.zero_()
        flat_weights_2d = route.flat_weights.view(num_tokens, num_topk)
        for route_idx in range(num_topk):
            routed_output_tmp.copy_(
                (
                    route_outputs[:, route_idx, :].float()
                    * flat_weights_2d[:, route_idx : route_idx + 1]
                ).to(torch.bfloat16)
            )
            output.copy_((output.float() + routed_output_tmp.float()).to(torch.bfloat16))
        return output

    def _should_use_bucketed_compact_static_route(
        self,
        *,
        a: torch.Tensor,
        route,
    ) -> bool:
        return (
            self.activation == "relu2"
            and route.kind == "compact"
            and route.token_map is not None
            and route.token_weights is not None
            and _parse_bool_env("B12X_BF16_ENABLE_BUCKETED_COMPACT_RELU2_STATIC", False)
        )

    def _run_compact_singleton_direct_bucket(
        self,
        *,
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        bucket_expert_ids: torch.Tensor,
        bucket_token_map: torch.Tensor,
        bucket_token_weights: torch.Tensor,
        workspace,
    ) -> None:
        if (
            workspace.micro_row1_routed_input_chunk is None
            or workspace.micro_row1_fc1_output_chunk is None
            or workspace.micro_row1_fc2_output_chunk is None
            or workspace.compact_sorted_flat_token_indices_i32 is None
        ):
            raise RuntimeError("relu2 compact singleton direct scratch is not initialized")

        _ensure_bmm_cublas_ready(a.device)
        get_direct_runtime = getattr(self, "get_expert_indexed_dense_runtime", None)
        if callable(get_direct_runtime):
            (
                fc1_dense_kernel,
                fc2_dense_kernel,
                max_active_clusters,
            ) = get_direct_runtime(a.device)
        else:
            (
                fc1_dense_kernel,
                fc2_dense_kernel,
                max_active_clusters,
            ) = self._get_direct_expert_indexed_dense_runtime(a.device)
        fused_direct_kernel = None
        if self.activation == "relu2" and _ENABLE_FUSED_DIRECT_RELU2:
            fused_direct_kernel, fused_max_active_clusters = (
                self._get_indexed_fused_relu2_runtime(a.device)
            )
            max_active_clusters = min(max_active_clusters, fused_max_active_clusters)
        from b12x.moe.fused.bf16.indexed_dense import run_dense_bf16_expert_ids

        dense_cluster_limit = min(
            max_active_clusters,
            self.row1_multi_token_dense_max_active_clusters
            if a.shape[0] > 1
            else self.row1_dense_max_active_clusters,
        )
        fc1_dense_cluster_limit = min(
            dense_cluster_limit,
            _parse_optional_int_env("B12X_BF16_INDEXED_FC1_MAX_ACTIVE_CLUSTERS")
            or dense_cluster_limit,
        )
        fc2_dense_cluster_limit = min(
            dense_cluster_limit,
            _parse_optional_int_env("B12X_BF16_INDEXED_FC2_MAX_ACTIVE_CLUSTERS")
            or dense_cluster_limit,
        )
        direct_w1_view, direct_w2_view = self._get_direct_weight_views(
            workspace=workspace,
            w1=w1,
            w2=w2,
        )

        bucket_capacity = bucket_expert_ids.shape[0]
        row1_routed_dense = workspace.micro_row1_routed_input_chunk[:, :, :bucket_capacity]
        row1_routed_dense_rows = row1_routed_dense[0].transpose(0, 1)
        fc1_dense = workspace.micro_row1_fc1_output_chunk[:, :, :bucket_capacity]
        fc2_dense = workspace.micro_row1_fc2_output_chunk[:, :, :bucket_capacity]
        bucket_token_indices = workspace.compact_sorted_flat_token_indices_i32[:bucket_capacity]
        bucket_token_indices.copy_(bucket_token_map[:, 0])
        torch.clamp_min(bucket_token_indices, 0, out=bucket_token_indices)

        gather_rows_bf16(
            a,
            bucket_token_indices,
            row1_routed_dense_rows,
        )
        valid_token_mask = (bucket_token_map[:, :1] >= 0).to(row1_routed_dense_rows.dtype)
        row1_routed_dense_rows.mul_(valid_token_mask)

        if fused_direct_kernel is not None:
            run_fused_relu2_bf16_expert_ids(
                fused_direct_kernel,
                row1_routed_dense,
                direct_w1_view,
                direct_w2_view,
                bucket_expert_ids,
                fc2_dense,
                dense_cluster_limit,
                current_cuda_stream(),
            )
        else:
            run_dense_bf16_expert_ids(
                fc1_dense_kernel,
                row1_routed_dense,
                direct_w1_view,
                bucket_expert_ids,
                fc1_dense,
                fc1_dense_cluster_limit,
                current_cuda_stream(),
            )
            run_dense_bf16_expert_ids(
                fc2_dense_kernel,
                fc1_dense,
                direct_w2_view,
                bucket_expert_ids,
                fc2_dense,
                fc2_dense_cluster_limit,
                current_cuda_stream(),
            )

        scatter_add_token_map_fc2_bf16(
            fc2_dense,
            bucket_token_map,
            bucket_token_weights,
            workspace.accum_output,
            round_weighted_to_bf16=workspace.num_topk <= 4,
        )

    def _accumulate_bucketed_compact_static_route(
        self,
        *,
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        route,
        workspace,
        bucket_row_begin: int,
        zero_accum_output: bool,
    ) -> None:
        if (
            workspace.compact_bucket_expert_ids is None
            or workspace.compact_bucket_token_map is None
            or workspace.compact_bucket_token_weights is None
        ):
            raise RuntimeError("bucketed compact route scratch is not initialized")

        from b12x.moe.fused.bf16.indexed_dense import run_dense_bf16_expert_ids

        state_e = route.kernel_weight_expert_ids.shape[0]
        if state_e <= 0:
            if zero_accum_output:
                workspace.accum_output[: a.shape[0]].zero_()
            return

        (
            indexed_fc1_dense_kernel,
            indexed_fc2_dense_kernel,
            indexed_max_active_clusters,
        ) = self._get_indexed_dense_runtime(a.device)
        max_active_clusters = self._resolve_max_active_clusters(
            max_active_clusters=indexed_max_active_clusters,
            routed_rows=route.routed_rows,
        )
        direct_w1_view, direct_w2_view = self._get_direct_weight_views(
            workspace=workspace,
            w1=w1,
            w2=w2,
        )
        tile_m = indexed_fc1_dense_kernel.tile_shape_mnk[0]
        accum_output = workspace.accum_output[: a.shape[0]]
        if zero_accum_output:
            accum_output.zero_()

        for bucket_rows in range(bucket_row_begin, a.shape[0] + 1):
            bucket_capacity = max(
                1,
                min(state_e, route.routed_rows // bucket_rows),
            )
            bucket_expert_ids = workspace.compact_bucket_expert_ids[:bucket_capacity]
            bucket_token_map = workspace.compact_bucket_token_map[
                :bucket_capacity, :bucket_rows
            ]
            bucket_token_weights = workspace.compact_bucket_token_weights[
                :bucket_capacity, :bucket_rows
            ]
            build_bucketed_compact_route(
                route.row_counts[:state_e],
                route.kernel_weight_expert_ids[:state_e],
                route.token_map[:state_e, :bucket_rows],
                route.token_weights[:state_e, :bucket_rows],
                bucket_rows,
                bucket_expert_ids,
                bucket_token_map,
                bucket_token_weights,
            )
            torch.clamp_min(bucket_expert_ids, 0, out=bucket_expert_ids)
            if bucket_rows == 1:
                self._run_compact_singleton_direct_bucket(
                    a=a,
                    w1=w1,
                    w2=w2,
                    bucket_expert_ids=bucket_expert_ids,
                    bucket_token_map=bucket_token_map,
                    bucket_token_weights=bucket_token_weights,
                    workspace=workspace,
                )
                continue
            padded_rows = _round_up_rows(bucket_rows, tile_m)
            routed_chunk = workspace.routed_input_chunk[:padded_rows, :, :bucket_capacity]
            fc1_chunk = workspace.fc1_output_chunk[
                :padded_rows, : w1.shape[1], :bucket_capacity
            ]
            fc2_chunk = workspace.fc2_output_chunk[:padded_rows, :, :bucket_capacity]
            chunk = _CompactRouteChunk(
                expert_ids_i32=bucket_expert_ids,
                expert_ids_i64=None,
                compact_flat_token_indices_gpu=None,
                compact_topk_ids_gpu=None,
                compact_route_row_indices_gpu=None,
                compact_expert_begin=0,
                compact_expert_end=bucket_capacity,
                token_map_gpu=bucket_token_map,
                token_weights_gpu=bucket_token_weights,
            )

            self._populate_small_row_routed_chunk(
                a=a,
                chunk=chunk,
                routed_chunk=routed_chunk,
            )
            fc1_chunk.zero_()
            fc2_chunk.zero_()
            run_dense_bf16_expert_ids(
                indexed_fc1_dense_kernel,
                routed_chunk,
                direct_w1_view,
                bucket_expert_ids,
                fc1_chunk,
                max_active_clusters,
                current_cuda_stream(),
            )
            run_dense_bf16_expert_ids(
                indexed_fc2_dense_kernel,
                fc1_chunk,
                direct_w2_view,
                bucket_expert_ids,
                fc2_chunk,
                max_active_clusters,
                current_cuda_stream(),
            )
            self._store_small_row_chunk_output(
                workspace=workspace,
                flat_topk_weights=route.flat_weights,
                chunk=chunk,
                fc2_chunk=fc2_chunk,
                use_route_order_output=True,
            )

    def _run_bucketed_compact_static_route(
        self,
        *,
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        route,
        workspace,
        output: torch.Tensor,
    ) -> torch.Tensor:
        self._accumulate_bucketed_compact_static_route(
            a=a,
            w1=w1,
            w2=w2,
            route=route,
            workspace=workspace,
            bucket_row_begin=1,
            zero_accum_output=True,
        )
        output.copy_(workspace.accum_output[: a.shape[0]].to(torch.bfloat16))
        return output

    def _compact_route_chunk_ranges(
        self,
        *,
        route,
        expert_chunk_size: int,
    ) -> list[tuple[int, int]]:
        state_e = route.kernel_weight_expert_ids.shape[0]
        if state_e <= 0:
            return []
        if route.kind == "compact":
            return [(0, state_e)]
        chunk_experts = max(1, min(state_e, expert_chunk_size))
        if self.activation == "relu2":
            chunk_experts = min(chunk_experts, self.relu2_expert_chunk_size)
        return [
            (expert_begin, min(state_e, expert_begin + chunk_experts))
            for expert_begin in range(0, state_e, chunk_experts)
        ]

    def run_compact_route(
        self,
        *,
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        route,
        workspace,
        output: torch.Tensor,
        expert_chunk_size: int,
    ) -> torch.Tensor:
        routed_rows = route.routed_rows
        if routed_rows <= 0:
            output.zero_()
            return output
        # The small-row bucketed scheduler assumes each expert can receive at
        # most one routed row per token, so per-expert row counts stay within
        # `[1, num_tokens]`. Real Nemotron routing satisfies that because top-k
        # ids are unique per token. Synthetic relu2 tests still exercise
        # duplicate-heavy cases like `topk=22, e=8`, where duplicates are
        # guaranteed by shape alone; route those through the direct path instead
        # of trying to express them with token_map columns.
        guaranteed_duplicate_experts_per_token = topk_ids.shape[1] > w1.shape[0]
        if self._should_use_bucketed_compact_static_route(a=a, route=route):
            if not guaranteed_duplicate_experts_per_token:
                return self._run_bucketed_compact_static_route(
                    a=a,
                    w1=w1,
                    w2=w2,
                    topk_ids=topk_ids,
                    route=route,
                    workspace=workspace,
                    output=output,
                )
        if self._should_use_compact_direct_route(route=route):
            return self._run_compact_direct_route(
                a=a,
                w1=w1,
                w2=w2,
                topk_ids=topk_ids,
                route=route,
                workspace=workspace,
                output=output,
            )

        chunk_ranges = self._compact_route_chunk_ranges(
            route=route,
            expert_chunk_size=expert_chunk_size,
        )
        if not chunk_ranges:
            output.zero_()
            return output

        kernel = self._get_fused_runtime()
        fc1_dense_kernel, fc2_dense_kernel, max_active_clusters = self._get_dense_runtime(
            a.device
        )
        indexed_fc1_dense_kernel = None
        indexed_fc2_dense_kernel = None
        direct_w1_view = None
        direct_w2_view = None
        if self.activation == "relu2":
            (
                indexed_fc1_dense_kernel,
                indexed_fc2_dense_kernel,
                indexed_max_active_clusters,
            ) = self._get_indexed_dense_runtime(a.device)
            max_active_clusters = min(max_active_clusters, indexed_max_active_clusters)
            direct_w1_view = w1.permute(1, 2, 0)
            direct_w2_view = w2.permute(1, 2, 0)
        max_active_clusters = self._resolve_max_active_clusters(
            max_active_clusters=max_active_clusters,
            routed_rows=routed_rows,
        )
        max_rows = max(1, a.shape[0])
        if self.activation == "relu2":
            tile_m = (
                indexed_fc1_dense_kernel.tile_shape_mnk[0]
                if indexed_fc1_dense_kernel is not None
                else fc1_dense_kernel.tile_shape_mnk[0]
            )
            padded_rows = _round_up_rows(max_rows, tile_m)
        elif route.kind == "compact" or w2.shape[2] // _FUSED_TILE_SHAPE_MNK[2] == 1:
            padded_rows = _round_up_tile_m(max_rows)
        else:
            tile_m = fc1_dense_kernel.tile_shape_mnk[0]
            padded_rows = _round_up_rows(max_rows, tile_m)
        max_chunk_experts = max(expert_end - expert_begin for expert_begin, expert_end in chunk_ranges)
        if (
            workspace.routed_input_chunk.shape[0] < padded_rows
            or workspace.routed_input_chunk.shape[2] < max_chunk_experts
            or workspace.fc1_output_chunk.shape[2] < max_chunk_experts
            or workspace.fc2_output_chunk.shape[2] < max_chunk_experts
        ):
            raise RuntimeError("relu2 compact-route static workspace scratch is undersized")

        use_large_relu2_route_order_combine = (
            self.activation == "relu2" and padded_rows > _RELU2_INDEXED_DENSE_MAX_ROWS
        )
        use_grouped_reduce = (
            self.activation == "relu2"
            and route.compact_topk_ids is not None
            and route.route_row_indices is not None
            and len(chunk_ranges) == 1
            and not use_large_relu2_route_order_combine
        )
        accum_output = workspace.accum_output[: a.shape[0]]
        accum_output.zero_()
        route_output_flat = (
            workspace.routed_output_unsorted[:routed_rows]
            if use_large_relu2_route_order_combine
            else None
        )
        for expert_begin, expert_end in chunk_ranges:
            chunk = _CompactRouteChunk(
                expert_ids_i32=(
                    None
                    if route.kernel_weight_expert_ids is None
                    else route.kernel_weight_expert_ids[expert_begin:expert_end]
                ),
                expert_ids_i64=(
                    None
                    if route.kernel_weight_expert_ids_i64 is None
                    else route.kernel_weight_expert_ids_i64[expert_begin:expert_end]
                ),
                compact_flat_token_indices_gpu=route.flat_token_indices,
                compact_topk_ids_gpu=route.compact_topk_ids,
                compact_route_row_indices_gpu=route.route_row_indices,
                compact_expert_begin=expert_begin,
                compact_expert_end=expert_end,
                token_map_gpu=(
                    None
                    if route.token_map is None
                    else route.token_map[expert_begin:expert_end, : a.shape[0]]
                ),
                token_weights_gpu=(
                    None
                    if route.token_weights is None
                    else route.token_weights[expert_begin:expert_end, : a.shape[0]]
                ),
            )
            chunk_e = expert_end - expert_begin
            routed_chunk = workspace.routed_input_chunk[:padded_rows, :, :chunk_e]
            fc1_chunk = workspace.fc1_output_chunk[:padded_rows, : w1.shape[1], :chunk_e]
            fc2_chunk = workspace.fc2_output_chunk[:padded_rows, :, :chunk_e]
            inter_chunk = workspace.intermediate_chunk[:padded_rows, : w2.shape[2], :chunk_e]

            self._populate_small_row_routed_chunk(
                a=a,
                chunk=chunk,
                routed_chunk=routed_chunk,
            )
            fc1_chunk.zero_()
            fc2_chunk.zero_()
            if self.activation == "relu2":
                use_indexed_dense = padded_rows <= _RELU2_INDEXED_DENSE_MAX_ROWS
                if use_indexed_dense:
                    from b12x.moe.fused.bf16.indexed_dense import (
                        run_dense_bf16_expert_ids,
                    )

                    assert indexed_fc1_dense_kernel is not None
                    assert indexed_fc2_dense_kernel is not None
                    assert direct_w1_view is not None
                    assert direct_w2_view is not None
                    assert chunk.expert_ids_i32 is not None
                    run_dense_bf16_expert_ids(
                        indexed_fc1_dense_kernel,
                        routed_chunk,
                        direct_w1_view,
                        chunk.expert_ids_i32,
                        fc1_chunk,
                        max_active_clusters,
                        current_cuda_stream(),
                    )
                    run_dense_bf16_expert_ids(
                        indexed_fc2_dense_kernel,
                        fc1_chunk,
                        direct_w2_view,
                        chunk.expert_ids_i32,
                        fc2_chunk,
                        max_active_clusters,
                        current_cuda_stream(),
                    )
                else:
                    assert chunk.expert_ids_i64 is not None
                    routed_batch = routed_chunk.permute(2, 0, 1).float()
                    w1_batch = w1.index_select(0, chunk.expert_ids_i64).float()
                    fc1_batch = torch.bmm(routed_batch, w1_batch.transpose(1, 2))
                    fc1_batch = torch.where(
                        fc1_batch > 0,
                        fc1_batch * fc1_batch,
                        torch.zeros_like(fc1_batch),
                    )
                    w2_batch = w2.index_select(0, chunk.expert_ids_i64).float()
                    fc2_batch = torch.bmm(fc1_batch, w2_batch.transpose(1, 2))
                    fc2_chunk.copy_(fc2_batch.permute(1, 2, 0).to(torch.bfloat16))
                    assert route_output_flat is not None
                    pair_mask = (route.compact_topk_ids >= expert_begin) & (
                        route.compact_topk_ids < expert_end
                    )
                    if pair_mask.any():
                        fc2_route_major = fc2_chunk.permute(0, 2, 1)
                        pair_rows = route.route_row_indices[pair_mask].to(torch.int64)
                        pair_experts = (
                            route.compact_topk_ids[pair_mask] - expert_begin
                        ).to(torch.int64)
                        route_output_flat[pair_mask].copy_(
                            (
                                fc2_route_major[pair_rows, pair_experts].float()
                                * route.flat_weights[pair_mask, None]
                            ).to(torch.bfloat16)
                        )
            else:
                assert chunk.expert_ids_i64 is not None
                w1_chunk = w1.index_select(0, chunk.expert_ids_i64).permute(1, 2, 0)
                w2_chunk = w2.index_select(0, chunk.expert_ids_i64).permute(1, 2, 0)
                inter_tile_cnt = w2_chunk.shape[1] // _FUSED_TILE_SHAPE_MNK[2]
                if inter_tile_cnt == 1:
                    run_fused_chunk_bf16(
                        kernel,
                        routed_chunk,
                        w1_chunk,
                        w2_chunk,
                        fc2_chunk,
                        current_cuda_stream(),
                    )
                else:
                    run_dense_bf16(
                        fc1_dense_kernel,
                        routed_chunk,
                        w1_chunk,
                        fc1_chunk,
                        max_active_clusters,
                        current_cuda_stream(),
                    )
                    gated_n = w2_chunk.shape[1]
                    up_chunk = fc1_chunk[:, :gated_n, :]
                    gate_chunk = fc1_chunk[:, gated_n:, :]
                    inter_chunk.copy_(
                        (
                            torch.sigmoid(gate_chunk.float())
                            * gate_chunk.float()
                            * up_chunk.float()
                        ).to(torch.bfloat16)
                    )
                    run_dense_bf16(
                        fc2_dense_kernel,
                        inter_chunk,
                        w2_chunk,
                        fc2_chunk,
                        max_active_clusters,
                        current_cuda_stream(),
                    )
            if use_grouped_reduce:
                reduce_fc2_chunk_grouped_bf16(
                    fc2_chunk,
                    route.flat_weights,
                    route.compact_topk_ids,
                    route.route_row_indices,
                    output,
                    num_topk=topk_ids.shape[1],
                )
                return output
            if use_large_relu2_route_order_combine:
                continue
            self._store_small_row_chunk_output(
                workspace=workspace,
                flat_topk_weights=route.flat_weights,
                chunk=chunk,
                fc2_chunk=fc2_chunk,
                use_route_order_output=True,
            )

        if route_output_flat is not None:
            route_outputs = route_output_flat.view(a.shape[0], topk_ids.shape[1], output.shape[1])
            output.zero_()
            for route_idx in range(topk_ids.shape[1]):
                output.copy_(
                    (
                        output.float() + route_outputs[:, route_idx, :].float()
                    ).to(torch.bfloat16)
                )
            return output
        output.copy_(accum_output.to(torch.bfloat16))
        return output

    def _populate_small_row_routed_chunk(
        self,
        *,
        a: torch.Tensor,
        chunk,
        routed_chunk: torch.Tensor,
    ) -> None:
        routed_chunk.zero_()
        if chunk.token_map_gpu is not None:
            scatter_routed_input_token_map_bf16(
                a,
                chunk.token_map_gpu,
                routed_chunk,
            )
            return
        if chunk.compact_topk_ids_gpu is not None:
            if (
                chunk.compact_flat_token_indices_gpu is None
                or chunk.compact_route_row_indices_gpu is None
                or chunk.compact_expert_begin is None
                or chunk.compact_expert_end is None
            ):
                raise RuntimeError("compact BF16 chunk metadata is incomplete")
            scatter_routed_input_compact_chunk_bf16(
                a,
                chunk.compact_flat_token_indices_gpu,
                chunk.compact_topk_ids_gpu,
                chunk.compact_route_row_indices_gpu,
                routed_chunk,
                expert_begin=chunk.compact_expert_begin,
                expert_end=chunk.compact_expert_end,
            )
            return
        scatter_routed_input_grouped_bf16(
            a,
            chunk.flat_token_indices_gpu,
            chunk.flat_local_experts_gpu,
            chunk.flat_row_indices_gpu,
            routed_chunk,
        )

    def _store_small_row_chunk_output(
        self,
        *,
        workspace,
        flat_topk_weights: torch.Tensor,
        chunk,
        fc2_chunk: torch.Tensor,
        use_route_order_output: bool,
    ) -> None:
        if chunk.token_map_gpu is not None and chunk.token_weights_gpu is not None:
            if not use_route_order_output:
                raise RuntimeError(
                    "token-map BF16 chunk scatter-add requires route-order accumulation"
                )
            scatter_add_token_map_fc2_bf16(
                fc2_chunk,
                chunk.token_map_gpu,
                chunk.token_weights_gpu,
                workspace.accum_output,
                round_weighted_to_bf16=workspace.num_topk <= 4,
            )
            return
        if chunk.compact_topk_ids_gpu is not None:
            if not use_route_order_output:
                raise RuntimeError(
                    "compact BF16 chunk scatter-add requires route-order accumulation"
                )
            if (
                chunk.compact_flat_token_indices_gpu is None
                or chunk.compact_route_row_indices_gpu is None
                or chunk.compact_expert_begin is None
                or chunk.compact_expert_end is None
            ):
                raise RuntimeError("compact BF16 chunk metadata is incomplete")
            scatter_add_compact_chunk_fc2_bf16(
                fc2_chunk,
                flat_topk_weights,
                chunk.compact_flat_token_indices_gpu,
                chunk.compact_topk_ids_gpu,
                chunk.compact_route_row_indices_gpu,
                workspace.accum_output,
                expert_begin=chunk.compact_expert_begin,
                expert_end=chunk.compact_expert_end,
                round_weighted_to_bf16=workspace.num_topk <= 4,
            )
            return
        routed_weights = flat_topk_weights.index_select(0, chunk.flat_route_indices_gpu)
        if use_route_order_output:
            scatter_add_grouped_fc2_bf16(
                fc2_chunk,
                routed_weights,
                chunk.flat_token_indices_gpu,
                chunk.flat_local_experts_gpu,
                chunk.flat_row_indices_gpu,
                workspace.accum_output,
                round_weighted_to_bf16=workspace.num_topk <= 4,
            )
            return
        local_outputs = fc2_chunk[
            chunk.flat_row_indices_gpu,
            :,
            chunk.flat_local_experts_gpu,
        ]
        weighted_outputs = (local_outputs * routed_weights[:, None]).to(torch.bfloat16)
        workspace.routed_output_sorted.index_copy_(
            0,
            chunk.flat_source_rows_gpu,
            weighted_outputs,
        )

    def _populate_routed_chunk(
        self,
        *,
        a: torch.Tensor,
        workspace,
        chunk,
        routed_chunk: torch.Tensor,
    ) -> None:
        if self.activation == "relu2" and chunk.max_rows <= self.vectorized_row_limit:
            return self._populate_small_row_routed_chunk(
                a=a,
                chunk=chunk,
                routed_chunk=routed_chunk,
            )

        routed_chunk.zero_()
        if chunk.total_rows > 0:
            routed_chunk[chunk.dst_rows_gpu, :, chunk.dst_expert_gpu] = (
                workspace.routed_input[chunk.src_rows_gpu]
            )

    def _store_sorted_chunk_output(
        self,
        *,
        workspace,
        flat_topk_weights: torch.Tensor,
        chunk,
        fc2_chunk: torch.Tensor,
        use_route_order_output: bool,
    ) -> None:
        if self.activation == "relu2" and chunk.max_rows <= self.vectorized_row_limit:
            return self._store_small_row_chunk_output(
                workspace=workspace,
                flat_topk_weights=flat_topk_weights,
                chunk=chunk,
                fc2_chunk=fc2_chunk,
                use_route_order_output=use_route_order_output,
            )

        if chunk.total_rows > 0:
            workspace.routed_output_sorted.index_copy_(
                0,
                chunk.src_rows_gpu,
                (
                    fc2_chunk[chunk.dst_rows_gpu, :, chunk.dst_expert_gpu]
                    * workspace.sorted_weights[chunk.src_rows_gpu, None]
                ).to(torch.bfloat16),
            )

    def _finalize_output(
        self,
        *,
        workspace,
        routing,
        num_tokens: int,
        num_topk: int,
        hidden_size: int,
        output: torch.Tensor,
        use_route_order_output: bool,
    ) -> torch.Tensor:
        routed_rows = num_tokens * num_topk
        if use_route_order_output:
            output.copy_(workspace.accum_output[:num_tokens].to(torch.bfloat16))
            return output

        workspace.routed_output_unsorted[:routed_rows].index_copy_(
            0,
            routing.order,
            workspace.routed_output_sorted[:routed_rows],
        )
        route_outputs = workspace.routed_output_unsorted[:routed_rows].view(
            num_tokens, num_topk, hidden_size
        )
        output.copy_(route_outputs.sum(dim=1, dtype=torch.float32).to(torch.bfloat16))
        return output

    def run(
        self,
        *,
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routing,
        workspace,
        output: torch.Tensor,
    ) -> torch.Tensor:
        kernel = self._get_fused_runtime()
        fc1_dense_kernel, fc2_dense_kernel, max_active_clusters = self._get_dense_runtime(
            a.device
        )
        indexed_fc1_dense_kernel = None
        indexed_fc2_dense_kernel = None
        direct_w1_view = None
        direct_w2_view = None
        if self.activation == "relu2":
            (
                indexed_fc1_dense_kernel,
                indexed_fc2_dense_kernel,
                indexed_max_active_clusters,
            ) = self._get_indexed_dense_runtime(a.device)
            max_active_clusters = min(max_active_clusters, indexed_max_active_clusters)
            direct_w1_view = w1.permute(1, 2, 0)
            direct_w2_view = w2.permute(1, 2, 0)

        routed_rows = topk_ids.numel()
        max_active_clusters = self._resolve_max_active_clusters(
            max_active_clusters=max_active_clusters,
            routed_rows=routed_rows,
        )
        flat_topk_weights = topk_weights.reshape(-1)
        if flat_topk_weights.dtype != torch.float32 or flat_topk_weights.stride(0) != 1:
            flat_topk_weights = flat_topk_weights.to(torch.float32).contiguous()
        use_direct_small_row_path = (
            self.activation == "relu2"
            and routing.max_rows_any_chunk <= self.vectorized_row_limit
        )
        use_direct_fc2_grouped_reduce = (
            self.activation == "relu2"
            and len(routing.chunk_plans) == 1
            and routing.route_local_expert_slots_grouped is not None
            and routing.route_row_indices_grouped is not None
        )
        if use_direct_small_row_path:
            workspace.accum_output[: a.shape[0]].zero_()
        if not use_direct_small_row_path:
            torch.index_select(
                a,
                0,
                routing.sorted_token_indices,
                out=workspace.routed_input[:routed_rows],
            )
        if not use_direct_small_row_path and not use_direct_fc2_grouped_reduce:
            torch.index_select(
                flat_topk_weights,
                0,
                routing.order,
                out=workspace.sorted_weights[:routed_rows],
            )

        for chunk in routing.chunk_plans:
            chunk_e = len(chunk.expert_ids_cpu)
            max_rows = chunk.max_rows
            use_direct_weight_lookup = (
                self.activation == "relu2"
                and chunk.expert_ids_i32 is not None
                and direct_w1_view is not None
                and direct_w2_view is not None
            )
            hidden_n = w2.shape[2] if use_direct_weight_lookup else chunk.w2_chunk.shape[1]
            w1_rows = w1.shape[1] if use_direct_weight_lookup else chunk.w1_chunk.shape[0]
            inter_tile = kernel.tile_shape_mnk[2]
            inter_tile_cnt = hidden_n // inter_tile
            if inter_tile_cnt == 1:
                padded_rows = _round_up_tile_m(max_rows)
            else:
                padded_rows = _round_up_rows(max_rows, fc1_dense_kernel.tile_shape_mnk[0])
            routed_chunk = workspace.routed_input_chunk[:padded_rows, :, :chunk_e]
            fc1_chunk = workspace.fc1_output_chunk[:padded_rows, :w1_rows, :chunk_e]
            inter_chunk = workspace.intermediate_chunk[:padded_rows, :, :chunk_e]
            fc2_chunk = workspace.fc2_output_chunk[:padded_rows, :, :chunk_e]

            self._populate_routed_chunk(
                a=a,
                workspace=workspace,
                chunk=chunk,
                routed_chunk=routed_chunk,
            )

            if use_direct_weight_lookup:
                from b12x.moe.fused.bf16.indexed_dense import run_dense_bf16_expert_ids

                run_dense_bf16_expert_ids(
                    indexed_fc1_dense_kernel,
                    routed_chunk,
                    direct_w1_view,
                    chunk.expert_ids_i32,
                    fc1_chunk,
                    max_active_clusters,
                    current_cuda_stream(),
                )
                run_dense_bf16_expert_ids(
                    indexed_fc2_dense_kernel,
                    fc1_chunk,
                    direct_w2_view,
                    chunk.expert_ids_i32,
                    fc2_chunk,
                    max_active_clusters,
                    current_cuda_stream(),
                )
            elif inter_tile_cnt == 1:
                assert chunk.w1_chunk is not None
                assert chunk.w2_chunk is not None
                run_fused_chunk_bf16(
                    kernel,
                    routed_chunk,
                    chunk.w1_chunk,
                    chunk.w2_chunk,
                    fc2_chunk,
                    current_cuda_stream(),
                )
            else:
                assert chunk.w1_chunk is not None
                assert chunk.w2_chunk is not None
                run_dense_bf16(
                    fc1_dense_kernel,
                    routed_chunk,
                    chunk.w1_chunk,
                    fc1_chunk,
                    max_active_clusters,
                    current_cuda_stream(),
                )
                if self.is_gated:
                    gated_n = chunk.w2_chunk.shape[1]
                    up_chunk = fc1_chunk[:, :gated_n, :]
                    gate_chunk = fc1_chunk[:, gated_n:, :]
                    inter_chunk.copy_(
                        (
                            torch.sigmoid(gate_chunk.float())
                            * gate_chunk.float()
                            * up_chunk.float()
                        ).to(torch.bfloat16)
                    )
                    activation_chunk = inter_chunk
                else:
                    activation_chunk = fc1_chunk
                run_dense_bf16(
                    fc2_dense_kernel,
                    activation_chunk,
                    chunk.w2_chunk,
                    fc2_chunk,
                    max_active_clusters,
                    current_cuda_stream(),
                )

            if use_direct_fc2_grouped_reduce:
                reduce_fc2_chunk_grouped_bf16(
                    fc2_chunk,
                    flat_topk_weights,
                    routing.route_local_expert_slots_grouped,
                    routing.route_row_indices_grouped,
                    output,
                    num_topk=topk_ids.shape[1],
                )
                return output

            self._store_sorted_chunk_output(
                workspace=workspace,
                flat_topk_weights=flat_topk_weights,
                chunk=chunk,
                fc2_chunk=fc2_chunk,
                use_route_order_output=use_direct_small_row_path,
            )

        return self._finalize_output(
            workspace=workspace,
            routing=routing,
            num_tokens=a.shape[0],
            num_topk=topk_ids.shape[1],
            hidden_size=a.shape[1],
            output=output,
            use_route_order_output=use_direct_small_row_path,
        )


__all__ = ["MoEStaticKernelBackend"]
