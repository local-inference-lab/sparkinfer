"""
MoESuperfusedStaticKernel — one-launch producer+consumer static NVFP4 MoE kernel for SM120 (Blackwell).

This kernel keeps the proven compact-static FC1/FC2 compute body, but replaces the
old route/pack frontend with the producer-side pre-MLP work in the same resident kernel.
The result is still a two-phase algorithm, just without a host-side handoff:

  Phase 0: cooperative init / clear row counts
  Phase 1: walk routed (token, topk_slot) pairs, append rows per expert,
           write token_map + token_weights, and quantize each routed
           token row directly into expert-major packed A + scale storage
  Barrier: resident-grid barrier after all expert rows are finalized
  Phase 2: run the FC1 -> SiLU -> quant -> FC2 -> scatter datapath
           over the finalized expert-major packed input

The compute half is intentionally the same design as the earlier two-kernel
implementation:
  FC1:     A x gate^T, A x up^T     (paired FP4 block-scaled GEMMs)
  SiLU:    SiLU(gate) * up          (fused SwiGLU activation)
  Quant:   intermediate -> FP4      (cooperative quantization into shared A)
  FC2:     sweep all output tiles   (reuse the cached intermediate slice)
  Scatter: bf16x2 atomic add        (directly into token-major output)

What changes relative to the old split path:
  the compute launch used to expect the frontend to have already produced:
    - expert row counts
    - expert-major packed A
    - token_map / token_weights
  static.py builds those GPU-side before entering the same grouped compute
  schedule. That is why this file owns the resident-grid barriers and the
  route/pack bookkeeping itself.

Work decomposition
  Frontend:
    One CTA leader handles one routed pair at a time. It atomically appends a
    row to row_counts[expert], writes the source token + router weight, then
    quantizes the source token row into that expert-major destination row.
  Compute:
    The compact static work loop assigns (m_tile, intermediate_slice, expert).
    FC1 is computed once per slice, the slice is quantized into shared A, and
    FC2 sweeps all output tiles from that cached slice. FC1 cost is therefore
    amortized across every FC2 output tile.

Layouts and dataflow
  packed_a_storage:
    Flat uint8 backing store for expert-major FP4 activations.
    Logical view used by the compute path is [max_rows, K, E] fp4x2.
  scale_storage:
    Flat uint8 backing store for expert-major activation scale factors laid
    out in the CUTLASS/CuTe block-scaled MMA layout expected by the compute
    mainloop.
  token_map / token_weights:
    Expert-row metadata used by the FC2 scatter path to accumulate the final
    output directly into [num_tokens, K].

Why the barriers exist
  row_counts drives the grouped scheduler shape. The compute phase cannot begin
  until every routed pair has claimed its expert row and packed A/scales have
  been written. The static kernel therefore uses a resident-grid barrier between
  route/pack and compute instead of the host-side sequencing used previously.

Scale-contract note
  This kernel supports per-expert FC1 activation scales by quantizing each
  routed pair with input_global_scale[expert]. That is checkpoint-correct for
  models where gate/up input scales vary across experts.

Design boundary
  The static kernel is the compact decode backend. It keeps route/pack and
  compute in one resident launch for small routed working sets, and relies on
  the resident-grid barrier between those phases instead of overlapping them.
  Large routed workloads dispatch to the dynamic backend instead.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from cutlass.cutlass_dsl import (
    Int32, Int64, Uint8, Uint64, T, Integer,
    dsl_user_op,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync

from b12x.cute.utils import (
    get_num_sm,
    get_max_active_clusters,
    make_ptr,
)
from b12x.cute.fp4 import (
    atomic_add_global_i32,
    bfloat2_add,
    bfloat2_mul,
    elem_pointer,
    fabs_f32,
    fmax_f32,
    rcp_approx_ftz,
    warp_reduce,
    quantize_block_fp4,
    quantize_block_fp4_fast,
    get_ptr_as_int64,
    ld_global_v4_u32,
    st_global_f32,
    st_global_i32,
    shared_ptr_to_u32,
    st_shared_u8,
    st_global_u64,
)
from b12x.gemm.dense import (
    DenseGemmKernel,
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)
from b12x.cute.fp4 import scatter_add_bf16x2

from b12x.distributed._oneshot_common import (
    add_f32,
    reduce_peer_row_sum,
    sqrt_f32,
    wait_for_peer_signals,
)
from b12x.moe.fused.pre_mlp_static import _exp2_approx_ftz_f32



_SF_VEC_SIZE = 16
_COMPACT_STATIC_TILE_M = 128
_FC1_TILE_ROWS = 128
_FC1_BLOCK_SIZE = 16
_FC2_TILE_AMAX_GS_RCP = 1.0 / (6.0 * 448.0)
LOG2_E = 1.4426950408889634
_WARP_SIZE = 32
_SUPERFUSED_NORM_WARP_IDX = 0
_SUPERFUSED_ROUTE_WARP_IDX = 1
_SUPERFUSED_PACK_WARP_BASE = 2


@cute.jit
def _compact_static_get_work_tile(
    row_counts: cute.Tensor,
    active_expert_count: cute.Tensor,
    *,
    num_tiles_n: Int32,
    cluster_shape_mn: Tuple[Int32, Int32],
    current_work_linear_idx: Int32,
    current_local_expert_idx: Int32,
    accum_tile_m: Int32,
    cta_id_in_cluster: cute.Coord,
) -> Tuple[Tuple[Int32, Int32, Int32], Integer, Int32, Int32]:
    num_active_experts = active_expert_count[Int32(0)]
    scan_local_expert_idx = current_local_expert_idx
    tile_m = Int32(_COMPACT_STATIC_TILE_M)
    tile_m_minus_one = Int32(_COMPACT_STATIC_TILE_M - 1)

    while scan_local_expert_idx < num_active_experts:
        batch_rows = row_counts[scan_local_expert_idx]
        batch_m_tiles = (batch_rows + tile_m_minus_one) // tile_m
        if (accum_tile_m + batch_m_tiles) * num_tiles_n > current_work_linear_idx:
            current_local_expert_idx = scan_local_expert_idx
            scan_local_expert_idx = num_active_experts
        else:
            accum_tile_m += batch_m_tiles
            scan_local_expert_idx += Int32(1)
            current_local_expert_idx = scan_local_expert_idx

    is_valid = current_local_expert_idx < num_active_experts
    if is_valid:
        batch_rows = row_counts[current_local_expert_idx]
        is_valid = (
            accum_tile_m + (batch_rows + tile_m_minus_one) // tile_m
        ) * num_tiles_n > current_work_linear_idx

    cur_cluster_coord = (
        current_work_linear_idx // num_tiles_n - accum_tile_m,
        current_work_linear_idx % num_tiles_n,
        current_local_expert_idx,
    )
    cur_tile_coord = (
        Int32(cur_cluster_coord[0]) * cluster_shape_mn[0] + cta_id_in_cluster[0],
        Int32(cur_cluster_coord[1]) * cluster_shape_mn[1] + cta_id_in_cluster[1],
        Int32(cur_cluster_coord[2]),
    )
    return cur_tile_coord, is_valid, current_local_expert_idx, accum_tile_m


@dsl_user_op
def _st_shared_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_shared_i32(addr, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(
        T.i32(),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.s32 $0, [$1];",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def _st_shared_f32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_shared_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.f32 $0, [$1];",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def _ld_shared_u8(addr, *, loc=None, ip=None):
    return Uint8(llvm.inline_asm(
        T.i8(),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.u8 $0, [$1];",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def _ld_global_u64(addr, *, loc=None, ip=None):
    return Uint64(llvm.inline_asm(
        T.i64(),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.u64 $0, [$1];",
        "=l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def _ld_global_acquire_i32(addr, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(
        T.i32(),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.acquire.gpu.s32 $0, [$1];",
        "=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def _st_global_release_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.global.release.gpu.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _spin_wait_global_eq_i32(addr, expected, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(expected).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred %p0;\n"
        ".reg .s32 %val;\n"
        "spin_loop:\n"
        "  ld.global.acquire.gpu.s32 %val, [$0];\n"
        "  setp.eq.s32 %p0, %val, $1;\n"
        "  @%p0 bra spin_loop;\n"
        "}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _threadfence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _atomic_cas_global_i32(addr, compare, value, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(
        T.i32(),
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(compare).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "atom.global.cas.b32 $0, [$1], $2, $3;",
        "=r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))



@dsl_user_op
def bfloat2_hsum_to_f32(x: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Float32:
    """Extract sum of two bf16 lanes in a bfloat2 register as float32."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1, sum;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                add.f32 sum, f0, f1;
                mov.f32 $0, sum;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _bf16x16_dot_shared_x_global_w(
    x_words_flat: cute.Tensor,
    w_flat: cute.Tensor,
    x_word_base: Int32,
    w_elem_base: Int32,
):
    acc_h2 = cutlass.Uint32(0)
    w0, w1, w2, w3 = ld_global_v4_u32(get_ptr_as_int64(w_flat, w_elem_base))
    w4, w5, w6, w7 = ld_global_v4_u32(get_ptr_as_int64(w_flat, w_elem_base + Int32(8)))
    acc_h2 = bfloat2_add(acc_h2, bfloat2_mul(x_words_flat[x_word_base + Int32(0)], w0))
    acc_h2 = bfloat2_add(acc_h2, bfloat2_mul(x_words_flat[x_word_base + Int32(1)], w1))
    acc_h2 = bfloat2_add(acc_h2, bfloat2_mul(x_words_flat[x_word_base + Int32(2)], w2))
    acc_h2 = bfloat2_add(acc_h2, bfloat2_mul(x_words_flat[x_word_base + Int32(3)], w3))
    acc_h2 = bfloat2_add(acc_h2, bfloat2_mul(x_words_flat[x_word_base + Int32(4)], w4))
    acc_h2 = bfloat2_add(acc_h2, bfloat2_mul(x_words_flat[x_word_base + Int32(5)], w5))
    acc_h2 = bfloat2_add(acc_h2, bfloat2_mul(x_words_flat[x_word_base + Int32(6)], w6))
    acc_h2 = bfloat2_add(acc_h2, bfloat2_mul(x_words_flat[x_word_base + Int32(7)], w7))
    return bfloat2_hsum_to_f32(acc_h2)


@cute.jit
def _resident_grid_barrier(
    barrier_count: cute.Tensor,
    barrier_epoch: cute.Tensor,
    grid_x: Int32,
    is_cta_leader: Int32,
):
    cute.arch.sync_threads()
    _threadfence()
    if is_cta_leader > Int32(0):
        barrier_count_addr = get_ptr_as_int64(barrier_count, Int32(0))
        barrier_epoch_addr = get_ptr_as_int64(barrier_epoch, Int32(0))
        old_epoch = _ld_global_acquire_i32(barrier_epoch_addr)
        arrived = atomic_add_global_i32(barrier_count_addr, Int32(1))
        if arrived == grid_x - Int32(1):
            st_global_i32(barrier_count_addr, Int32(0))
            _st_global_release_i32(barrier_epoch_addr, old_epoch + Int32(1))
        else:
            _spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)
    cute.arch.sync_threads()


@cute.jit
def _append_compact_prequantized_row_warp(
    *,
    lane_id: Int32,
    token_idx: Int32,
    expert_id: Int32,
    route_weight: cutlass.Float32,
    normalized_row: cute.Tensor,
    expert_input_scale: cute.Tensor,
    expert_alpha: cute.Tensor,
    active_expert_count: cute.Tensor,
    weight_expert_ids: cute.Tensor,
    global_to_local_expert: cute.Tensor,
    row_counts: cute.Tensor,
    token_map_flat: cute.Tensor,
    token_weights_flat: cute.Tensor,
    packed_input_flat: cute.Tensor,
    packed_input_scale_flat: cute.Tensor,
    fc1_tile_scale_flat: cute.Tensor,
    fc1_tile_alpha_flat: cute.Tensor,
    max_rows: Int32,
    output_bytes_per_row: Int32,
    expert_scale_stride: Int32,
    tiles_per_expert: Int32,
    sf_blocks_per_row: Int32,
    num_k_tiles: Int32,
    fast_math: bool,
):
    local_expert_id = Int32(0)
    row = Int32(0)
    gs_value = expert_input_scale[expert_id].to(cutlass.Float32)
    alpha_value = expert_alpha[expert_id].to(cutlass.Float32)

    if lane_id == Int32(0):
        prior_local_expert_id = _atomic_cas_global_i32(
            get_ptr_as_int64(global_to_local_expert, expert_id),
            Int32(-1),
            Int32(-2),
        )
        if prior_local_expert_id == Int32(-1):
            local_expert_id = atomic_add_global_i32(
                get_ptr_as_int64(active_expert_count, Int32(0)),
                Int32(1),
            )
            st_global_i32(get_ptr_as_int64(weight_expert_ids, local_expert_id), expert_id)
            _st_global_release_i32(
                get_ptr_as_int64(global_to_local_expert, expert_id),
                local_expert_id,
            )
        else:
            if prior_local_expert_id == Int32(-2):
                _spin_wait_global_eq_i32(
                    get_ptr_as_int64(global_to_local_expert, expert_id),
                    Int32(-2),
                )
                prior_local_expert_id = _ld_global_acquire_i32(
                    get_ptr_as_int64(global_to_local_expert, expert_id),
                )
            local_expert_id = prior_local_expert_id

        row = atomic_add_global_i32(
            get_ptr_as_int64(row_counts, local_expert_id),
            Int32(1),
        )
        map_idx = local_expert_id * max_rows + row
        st_global_i32(get_ptr_as_int64(token_map_flat, map_idx), token_idx)
        st_global_f32(get_ptr_as_int64(token_weights_flat, map_idx), route_weight)

        if row % Int32(_FC1_TILE_ROWS) == Int32(0):
            tile_idx = row // Int32(_FC1_TILE_ROWS)
            tile_offset = local_expert_id * tiles_per_expert + tile_idx
            st_global_f32(get_ptr_as_int64(fc1_tile_scale_flat, tile_offset), gs_value)
            st_global_f32(get_ptr_as_int64(fc1_tile_alpha_flat, tile_offset), alpha_value)

    local_expert_id = cute.arch.shuffle_sync(local_expert_id, Int32(0))
    row = cute.arch.shuffle_sync(row, Int32(0))

    sf_idx = lane_id
    while sf_idx < sf_blocks_per_row:
        block_start = sf_idx * Int32(_FC1_BLOCK_SIZE)
        values = cute.make_rmem_tensor((_FC1_BLOCK_SIZE,), cutlass.Float32)
        block_max = cutlass.Float32(0.0)
        for elem_idx in cutlass.range_constexpr(_FC1_BLOCK_SIZE):
            value = cutlass.Float32(normalized_row[block_start + Int32(elem_idx)])
            values[elem_idx] = value
            block_max = fmax_f32(block_max, fabs_f32(value))
        packed64 = cutlass.Uint64(0)
        scale_byte = cutlass.Uint8(0)
        if cutlass.const_expr(fast_math):
            packed64, scale_byte = quantize_block_fp4_fast(values, block_max, gs_value)
        else:
            packed64, scale_byte = quantize_block_fp4(values, block_max, gs_value)

        output_offset = (
            local_expert_id * max_rows * output_bytes_per_row
            + row * output_bytes_per_row
            + sf_idx * Int32(8)
        )
        st_global_u64(get_ptr_as_int64(packed_input_flat, output_offset), packed64)

        m_tile_idx = row // Int32(_FC1_TILE_ROWS)
        k_tile_idx = sf_idx // Int32(4)
        outer_m_idx = row % Int32(32)
        inner_m_idx = (row % Int32(_FC1_TILE_ROWS)) // Int32(32)
        inner_k_idx = sf_idx % Int32(4)
        scale_offset = (
            local_expert_id * expert_scale_stride
            + m_tile_idx * num_k_tiles * Int32(32 * 4 * 4)
            + k_tile_idx * Int32(32 * 4 * 4)
            + outer_m_idx * Int32(4 * 4)
            + inner_m_idx * Int32(4)
            + inner_k_idx
        )
        packed_input_scale_flat[scale_offset] = scale_byte
        sf_idx += Int32(_WARP_SIZE)



class MoESuperfusedStaticKernel:
    """Resident producer+consumer compact-static MoE kernel."""

    def __init__(
        self,
        world_size: int,
        num_sparse_experts: int,
        top_k: int,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        fc2_tile_amax: bool = False,
        emit_normalized: bool = False,
        renormalize_topk: bool = True,
        skip_phase1: bool = False,
    ):
        self._dense_cls = DenseGemmKernel
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.world_size = world_size
        self.num_sparse_experts = num_sparse_experts
        self.top_k = top_k
        self.combined_top_k = top_k + 1
        self.input_scales_are_reciprocal = input_scales_are_reciprocal
        self.fast_math = fast_math
        self.fc2_tile_amax = fc2_tile_amax
        self.emit_normalized = emit_normalized
        self.renormalize_topk = renormalize_topk
        self.skip_phase1 = skip_phase1
        tile_k = sf_vec_size * 8
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
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
            barrier_id=1, num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.pass_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta,
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

    def _setup_attributes(self):
        import cutlass.utils.blackwell_helpers as sm120_utils

        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype, self.acc_dtype, self.sf_dtype,
        )
        atom_layout = cute.make_layout((2, 2, 1))
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False,
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op, atom_layout, permutation_mnk=permutation_mnk,
        )
        self.mma_atom = cute.make_mma_atom(mma_op)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_m_tiles = self.tile_shape_mnk[0] // (16 * 4)
        self.num_n_tiles = self.tile_shape_mnk[1] // (8 * 2)
        self.num_k_blocks = self.tile_shape_mnk[2] // 64

        sfa_smem = sm120_make_smem_layout_sfa(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1,
        )
        sfb_smem = sm120_make_smem_layout_sfb(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1,
        )

        self.ab_stage, self.epi_stage = self._dense_cls._compute_stages(
            self.tile_shape_mnk, self.a_dtype, self.b_dtype, self.sf_dtype,
            sfa_smem, sfb_smem, self.epi_tile, cutlass.BFloat16,
            self.smem_capacity, self.occupancy,
        )
        # ab_stage must divide k_tile_cnt (K/tile_K = 4096/128 = 32) evenly.
        # _compute_stages returns the max that fits in smem (e.g. 3), but
        # 32%3!=0 causes pipeline phase mismatch. Round down to nearest divisor.
        while self.ab_stage > 1 and 32 % self.ab_stage != 0:
            self.ab_stage -= 1
        self.epi_stage = 1
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk, self.epi_tile,
            self.a_dtype, self.a_layout,
            self.b_dtype, self.b_layout,
            self.ab_stage,
            cutlass.BFloat16, self.c_layout,
            self.epi_stage,
            self.sf_vec_size, self.tiled_mma,
        )

    @cute.jit
    def _resident_grid_barrier(
        self,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        grid_x: Int32,
        is_cta_leader: Int32,
    ):
        cute.arch.sync_threads()
        _threadfence()
        if is_cta_leader > Int32(0):
            barrier_count_addr = get_ptr_as_int64(barrier_count, Int32(0))
            barrier_epoch_addr = get_ptr_as_int64(barrier_epoch, Int32(0))
            old_epoch = _ld_global_acquire_i32(barrier_epoch_addr)
            arrived = atomic_add_global_i32(barrier_count_addr, Int32(1))
            if arrived == grid_x - Int32(1):
                st_global_i32(barrier_count_addr, Int32(0))
                _st_global_release_i32(barrier_epoch_addr, old_epoch + Int32(1))
            else:
                _spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)
        cute.arch.sync_threads()


    @cute.jit
    def __call__(
        self,
        inp0: cute.Tensor,
        inp1: cute.Tensor,
        inp2: cute.Tensor,
        inp3: cute.Tensor,
        inp4: cute.Tensor,
        inp5: cute.Tensor,
        inp6: cute.Tensor,
        inp7: cute.Tensor,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        self_signal: cute.Pointer,
        rank: Int32,
        residual_in: cute.Tensor,
        normalized_out: cute.Tensor,
        residual_out: cute.Tensor,
        norm_weight: cute.Tensor,
        sparse_gate_weight: cute.Tensor,
        shared_gate_weight: cute.Tensor,
        topk_ids_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        packed_a: cute.Tensor,
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        active_expert_count: cute.Tensor,
        weight_expert_ids: cute.Tensor,
        global_to_local_expert: cute.Tensor,
        input_global_scale: cute.Tensor,
        expert_alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        fc1_tile_scale: cute.Tensor,
        fc1_tile_alpha: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.a_dtype = packed_a.element_type
        self.b_dtype = b_w13.element_type
        self.sf_dtype = sfa_ptr.dtype
        self.a_layout = utils.LayoutEnum.from_tensor(packed_a)
        self.b_layout = utils.LayoutEnum.from_tensor(b_w13)
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        self._setup_attributes()

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(packed_a.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_w13_layout = blockscaled_utils.tile_atom_to_shape_SF(b_w13.shape, self.sf_vec_size)
        sfb_w13_tensor = cute.make_tensor(sfb_w13_ptr, sfb_w13_layout)

        tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
            packed_a, self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]), 1,
        )
        tma_sfa, gSFA = self._dense_cls._make_tma_atoms_and_tensors(
            sfa_tensor, self.sfa_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]), 1,
            internal_type=cutlass.Int16,
        )
        tma_b_w13, gB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            b_w13, self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), 1,
        )
        tma_sfb_w13, gSFB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_w13_tensor, self.sfb_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), 1,
            internal_type=cutlass.Int16,
        )
        sfb_down_layout = blockscaled_utils.tile_atom_to_shape_SF(b_down.shape, self.sf_vec_size)
        sfb_down_tensor = cute.make_tensor(sfb_down_ptr, sfb_down_layout)
        tma_b_down, gB_down = self._dense_cls._make_tma_atoms_and_tensors(
            b_down, self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), 1,
        )
        tma_sfb_down, gSFB_down = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_down_tensor, self.sfb_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), 1,
            internal_type=cutlass.Int16,
        )

        grid = (*self.cluster_shape_mn, max_active_clusters)
        self.kernel(
            inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7,
            signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7,
            self_signal, rank,
            residual_in, normalized_out, residual_out,
            norm_weight, sparse_gate_weight, shared_gate_weight,
            topk_ids_flat, topk_weights_flat,
            packed_a_storage, scale_storage,
            barrier_count, barrier_epoch,
            tma_a, gA, tma_sfa, gSFA,
            tma_b_w13, gB_w13, tma_sfb_w13, gSFB_w13,
            tma_b_down, gB_down, tma_sfb_down, gSFB_down,
            self.tiled_mma, self.mma_atom, self.cta_layout_mnk,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            row_counts, active_expert_count, weight_expert_ids, global_to_local_expert,
            input_global_scale, expert_alpha, down_alpha, global_scale,
            fc1_tile_scale, fc1_tile_alpha,
            scatter_output, token_map, token_weights,
            eps,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        inp0: cute.Tensor,
        inp1: cute.Tensor,
        inp2: cute.Tensor,
        inp3: cute.Tensor,
        inp4: cute.Tensor,
        inp5: cute.Tensor,
        inp6: cute.Tensor,
        inp7: cute.Tensor,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        self_signal: cute.Pointer,
        rank: Int32,
        residual_in: cute.Tensor,
        normalized_out: cute.Tensor,
        residual_out: cute.Tensor,
        norm_weight: cute.Tensor,
        sparse_gate_weight: cute.Tensor,
        shared_gate_weight: cute.Tensor,
        topk_ids_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        tma_a: cute.CopyAtom, mA: cute.Tensor,
        tma_sfa: cute.CopyAtom, mSFA: cute.Tensor,
        tma_b_w13: cute.CopyAtom, mB_w13: cute.Tensor,
        tma_sfb_w13: cute.CopyAtom, mSFB_w13: cute.Tensor,
        tma_b_down: cute.CopyAtom, mB_down: cute.Tensor,
        tma_sfb_down: cute.CopyAtom, mSFB_down: cute.Tensor,
        tiled_mma: cute.TiledMma, mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout, b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout, sfb_smem_staged: cute.Layout,
        epi_smem_staged: cute.ComposedLayout,
        row_counts: cute.Tensor,
        active_expert_count: cute.Tensor,
        weight_expert_ids: cute.Tensor,
        global_to_local_expert: cute.Tensor,
        input_global_scale: cute.Tensor,
        expert_alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        fc1_tile_scale: cute.Tensor,
        fc1_tile_alpha: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        eps: cutlass.Float32,
    ):
        """Kernel entry point."""
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        _, _, gdim_z = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_sfa)
            cpasync.prefetch_descriptor(tma_b_w13)
            cpasync.prefetch_descriptor(tma_sfb_w13)
            cpasync.prefetch_descriptor(tma_b_down)
            cpasync.prefetch_descriptor(tma_sfb_down)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

        a_smem_one = cute.slice_(a_smem_staged, (None, None, 0))
        b_smem_one = cute.slice_(b_smem_staged, (None, None, 0))
        sfa_smem_one = cute.slice_(sfa_smem_staged, (None, None, 0))
        sfb_smem_one = cute.slice_(sfb_smem_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_one)
            + cute.size_in_bytes(self.b_dtype, b_smem_one)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_one)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)
        )
        phase2_tma_copy_bytes = (
            cute.size_in_bytes(self.b_dtype, b_smem_one)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)
        )

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            ctrl: cute.struct.MemRange[cutlass.Int32, 8]
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            up_pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            phase2_pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            scatter_tok_cache: cute.struct.MemRange[cutlass.Int32, _COMPACT_STATIC_TILE_M]
            scatter_weight_cache: cute.struct.MemRange[cutlass.Float32, _COMPACT_STATIC_TILE_M]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB_up: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB_up: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]

        storage = smem.allocate(Storage)

        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps)
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        ml_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=tma_copy_bytes,
            barrier_storage=storage.pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        up_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=tma_copy_bytes,
            barrier_storage=storage.up_pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        phase2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=phase2_tma_copy_bytes,
            barrier_storage=storage.phase2_pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )

        cute.arch.sync_threads()

        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sB_up = storage.sB_up.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sA_in_u8 = cute.recast_tensor(sA, cutlass.Uint8)
        sB_u8 = cute.recast_tensor(sB, cutlass.Uint8)
        sB_up_u8 = cute.recast_tensor(sB_up, cutlass.Uint8)
        sSFA = storage.sSFA.get_tensor(sfa_smem_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_staged)
        sSFB_up = storage.sSFB_up.get_tensor(sfb_smem_staged)
        sSFA_u8 = cute.recast_tensor(sSFA, cutlass.Uint8)
        sSFB_u8 = cute.recast_tensor(sSFB, cutlass.Uint8)
        sSFB_up_u8 = cute.recast_tensor(sSFB_up, cutlass.Uint8)
        sC = storage.sC.get_tensor(
            epi_smem_staged.outer, swizzle=epi_smem_staged.inner,
        )
        a_base_addr = shared_ptr_to_u32(storage.sA.data_ptr())
        b_base_addr = shared_ptr_to_u32(storage.sB.data_ptr())
        b_up_base_addr = shared_ptr_to_u32(storage.sB_up.data_ptr())
        sfa_base_addr = shared_ptr_to_u32(storage.sSFA.data_ptr())
        sfb_base_addr = shared_ptr_to_u32(storage.sSFB.data_ptr())
        sfb_up_base_addr = shared_ptr_to_u32(storage.sSFB_up.data_ptr())
        ctrl_base_addr = shared_ptr_to_u32(storage.ctrl.data_ptr())
        scatter_tok_base_addr = shared_ptr_to_u32(storage.scatter_tok_cache.data_ptr())
        scatter_weight_base_addr = shared_ptr_to_u32(storage.scatter_weight_cache.data_ptr())
        c_base_addr = shared_ptr_to_u32(storage.sC.data_ptr())


        num_tokens = Int32(residual_in.shape[0])
        cols = Int32(residual_in.shape[1])
        num_experts = Int32(row_counts.shape[0])
        sf_blocks_per_row = cols // Int32(16)
        padded_sf_cols = ((cols + Int32(63)) // Int32(64)) * Int32(4)
        output_bytes_per_row = cols // Int32(2)
        tile_m = Int32(self.tile_shape_mnk[0])
        max_rows = Int32(token_map.shape[1])
        fc1_tiles_per_expert = (max_rows + tile_m - Int32(1)) // tile_m
        expert_scale_stride = Int32(scale_storage.shape[0]) // num_experts
        num_global_experts = Int32(global_to_local_expert.shape[0])
        flat_tid = Int32(bidz) * Int32(self.threads_per_cta) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(self.threads_per_cta)
        num_k_tiles = (cols + Int32(63)) // Int32(64)

        token_map_flat = cute.flatten(token_map)
        token_weights_flat = cute.flatten(token_weights)
        fc1_tile_scale_flat = cute.flatten(fc1_tile_scale)
        fc1_tile_alpha_flat = cute.flatten(fc1_tile_alpha)

        sNorm = cute.make_tensor(
            cute.recast_ptr(storage.sC.data_ptr().align(16), dtype=cutlass.BFloat16),
            cute.make_layout((cols,), stride=(1,)),
        )
        sNormWords = cute.flatten(cute.recast_tensor(sNorm, cutlass.Uint32))
        sSelectedIds = storage.scatter_tok_cache.get_tensor(
            cute.make_layout((self.combined_top_k,), stride=(1,))
        )
        sSelectedWeights = storage.scatter_weight_cache.get_tensor(
            cute.make_layout((self.combined_top_k,), stride=(1,))
        )

        if not self.skip_phase1:
            i = flat_tid
            while i < num_experts:
                row_counts[i] = Int32(0)
                i += flat_stride
            i = flat_tid
            while i < num_global_experts:
                global_to_local_expert[i] = Int32(-1)
                i += flat_stride
            total_fc1_tiles = Int32(fc1_tile_scale_flat.shape[0])
            i = flat_tid
            while i < total_fc1_tiles:
                fc1_tile_scale_flat[i] = cutlass.Float32(0.0)
                fc1_tile_alpha_flat[i] = cutlass.Float32(0.0)
                i += flat_stride
            if flat_tid == Int32(0):
                active_expert_count[Int32(0)] = Int32(0)

        scatter_total = num_tokens * cols
        j = flat_tid
        while j < scatter_total:
            scatter_output[j // cols, j % cols] = cutlass.BFloat16(0.0)
            j += flat_stride
        if self.skip_phase1 and flat_tid == Int32(0):
            topk_ids_flat[Int32(0)] = Int32(0)

        cute.arch.sync_threads()
        self._resident_grid_barrier(
            barrier_count, barrier_epoch, Int32(gdim_z), is_cta_leader,
        )

        if not self.skip_phase1:
            signal_ptrs = [signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7]
            inputs = [inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7]

            token_idx = Int32(bidz)
            while token_idx < num_tokens:
                wait_for_peer_signals(
                    signal_ptrs=signal_ptrs,
                    self_signal=self_signal,
                    rank=rank,
                    world_size=self.world_size,
                    bidx=token_idx,
                    tidx=Int32(tidx),
                )

            local_sum_sq = cutlass.Float32(0.0)
            inv_scale = cutlass.Float32(0.0)
            if warp_idx == Int32(_SUPERFUSED_NORM_WARP_IDX):
                col = Int32(tidx) & Int32(_WARP_SIZE - 1)
                while col < cols:
                    acc = reduce_peer_row_sum(
                        inputs=inputs,
                        world_size=self.world_size,
                        bidx=token_idx,
                        col=col,
                        element_dtype=cutlass.BFloat16,
                    )
                    residual_val = cutlass.BFloat16(
                        acc + cutlass.Float32(residual_in[token_idx, col])
                    )
                    residual_out[token_idx, col] = residual_val
                    sNorm[col] = residual_val
                    residual_f32 = cutlass.Float32(residual_val)
                    local_sum_sq += residual_f32 * residual_f32
                    col += Int32(_WARP_SIZE)

                sum_sq = warp_reduce(local_sum_sq, add_f32)
                denom = sqrt_f32(sum_sq / cutlass.Float32(cols) + eps)
                if self.fast_math:
                    inv_scale = rcp_approx_ftz(denom)
                else:
                    inv_scale = cutlass.Float32(1.0) / denom
            cute.arch.sync_threads()

            if warp_idx == Int32(_SUPERFUSED_NORM_WARP_IDX):
                col = Int32(tidx) & Int32(_WARP_SIZE - 1)
                while col < cols:
                    gamma = cutlass.Float32(1.0) + cutlass.Float32(norm_weight[col])
                    out = cutlass.Float32(sNorm[col]) * inv_scale * gamma
                    out_cast = cutlass.BFloat16(out)
                    sNorm[col] = out_cast
                    if self.emit_normalized:
                        normalized_out[token_idx, col] = out_cast
                    col += Int32(_WARP_SIZE)
            cute.arch.sync_threads()

            neg_inf = cutlass.Float32(-3.4028235e38)
            top_vals = [neg_inf for _ in range(self.top_k)]
            top_ids = [Int32(-1) for _ in range(self.top_k)]

            shared_acc = cutlass.Float32(0.0)
            shared_local = cutlass.Float32(0.0)
            lane_id = Int32(tidx) & Int32(_WARP_SIZE - 1)

            if warp_idx == Int32(_SUPERFUSED_ROUTE_WARP_IDX):
                shared_gate_flat = cute.flatten(shared_gate_weight)
                sparse_gate_flat = cute.flatten(sparse_gate_weight)
                word_base = lane_id * Int32(8)
                word_stride = Int32(_WARP_SIZE * 8)
                hidden_words = cols // Int32(2)
                while word_base + Int32(7) < hidden_words:
                    shared_local += _bf16x16_dot_shared_x_global_w(
                        sNormWords,
                        shared_gate_flat,
                        word_base,
                        word_base * Int32(2),
                    )
                    word_base += word_stride
                shared_acc = warp_reduce(shared_local, add_f32)

                expert = Int32(0)
                while expert < Int32(self.num_sparse_experts):
                    expert_local = cutlass.Float32(0.0)
                    word_base = lane_id * Int32(8)
                    expert_word_base = expert * hidden_words * Int32(2)
                    while word_base + Int32(7) < hidden_words:
                        expert_local += _bf16x16_dot_shared_x_global_w(
                            sNormWords,
                            sparse_gate_flat,
                            word_base,
                            expert_word_base + word_base * Int32(2),
                        )
                        word_base += word_stride
                    expert_acc = warp_reduce(expert_local, add_f32)
                    if lane_id == Int32(0):
                        candidate_val = cutlass.Float32(cutlass.BFloat16(expert_acc))
                        insert_slot = Int32(-1)
                        for slot in cutlass.range_constexpr(self.top_k):
                            if insert_slot == Int32(-1) and (
                                candidate_val > top_vals[slot] or (
                                    candidate_val == top_vals[slot] and expert > top_ids[slot]
                                )
                            ):
                                insert_slot = Int32(slot)
                        for shift in cutlass.range_constexpr(self.top_k - 1, -1, -1):
                            if insert_slot != Int32(-1):
                                if Int32(shift) > insert_slot:
                                    top_vals[shift] = top_vals[shift - 1]
                                    top_ids[shift] = top_ids[shift - 1]
                                elif Int32(shift) == insert_slot:
                                    top_vals[shift] = candidate_val
                                    top_ids[shift] = expert
                    expert += Int32(1)

            if warp_idx == Int32(_SUPERFUSED_ROUTE_WARP_IDX):
                if lane_id == Int32(0):
                    shared_logit = cutlass.BFloat16(shared_acc)
                    neg_exp = _exp2_approx_ftz_f32(
                        cutlass.Float32(-LOG2_E) * cutlass.Float32(shared_logit)
                    )
                    shared_gate = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + neg_exp)

                    if self.renormalize_topk:
                        max_logit = top_vals[0]
                        exp_vals = [cutlass.Float32(0.0) for _ in range(self.top_k)]
                        denom = cutlass.Float32(0.0)
                        for slot in cutlass.range_constexpr(self.top_k):
                            exp_val = _exp2_approx_ftz_f32(
                                (top_vals[slot] - max_logit) * cutlass.Float32(LOG2_E)
                            )
                            exp_vals[slot] = exp_val
                            denom += exp_val
                        inv_denom = cutlass.Float32(1.0) / denom
                        for slot in cutlass.range_constexpr(self.top_k):
                            weight = exp_vals[slot] * inv_denom
                            sSelectedIds[Int32(slot)] = top_ids[slot]
                            sSelectedWeights[Int32(slot)] = weight
                            topk_offset = token_idx * Int32(self.combined_top_k) + Int32(slot)
                            st_global_i32(get_ptr_as_int64(topk_ids_flat, topk_offset), top_ids[slot])
                            st_global_f32(get_ptr_as_int64(topk_weights_flat, topk_offset), weight)
                    else:
                        for slot in cutlass.range_constexpr(self.top_k):
                            weight = top_vals[slot]
                            sSelectedIds[Int32(slot)] = top_ids[slot]
                            sSelectedWeights[Int32(slot)] = weight
                            topk_offset = token_idx * Int32(self.combined_top_k) + Int32(slot)
                            st_global_i32(get_ptr_as_int64(topk_ids_flat, topk_offset), top_ids[slot])
                            st_global_f32(get_ptr_as_int64(topk_weights_flat, topk_offset), weight)

                    shared_slot = Int32(self.top_k)
                    sSelectedIds[shared_slot] = Int32(self.num_sparse_experts)
                    sSelectedWeights[shared_slot] = shared_gate
                    topk_offset = token_idx * Int32(self.combined_top_k) + shared_slot
                    st_global_i32(get_ptr_as_int64(topk_ids_flat, topk_offset), Int32(self.num_sparse_experts))
                    st_global_f32(get_ptr_as_int64(topk_weights_flat, topk_offset), shared_gate)
            cute.arch.sync_threads()

            if warp_idx >= Int32(_SUPERFUSED_PACK_WARP_BASE):
                pack_warp_idx = warp_idx - Int32(_SUPERFUSED_PACK_WARP_BASE)
                slot = pack_warp_idx
                while slot < Int32(self.combined_top_k):
                    _append_compact_prequantized_row_warp(
                        lane_id=lane_id,
                        token_idx=token_idx,
                        expert_id=sSelectedIds[slot],
                        route_weight=sSelectedWeights[slot],
                        normalized_row=sNorm,
                        expert_input_scale=input_global_scale,
                        expert_alpha=expert_alpha,
                        active_expert_count=active_expert_count,
                        weight_expert_ids=weight_expert_ids,
                        global_to_local_expert=global_to_local_expert,
                        row_counts=row_counts,
                        token_map_flat=token_map_flat,
                        token_weights_flat=token_weights_flat,
                        packed_input_flat=packed_a_storage,
                        packed_input_scale_flat=scale_storage,
                        fc1_tile_scale_flat=fc1_tile_scale_flat,
                        fc1_tile_alpha_flat=fc1_tile_alpha_flat,
                        max_rows=max_rows,
                        output_bytes_per_row=output_bytes_per_row,
                        expert_scale_stride=expert_scale_stride,
                        tiles_per_expert=fc1_tiles_per_expert,
                        sf_blocks_per_row=sf_blocks_per_row,
                        num_k_tiles=num_k_tiles,
                        fast_math=self.fast_math,
                    )
                    slot += Int32(self.num_mma_warps + 1 - _SUPERFUSED_PACK_WARP_BASE)
            cute.arch.sync_threads()

            token_idx += Int32(gdim_z)

            # Phase 2 consumes the phase-1 compact workspace through the async/TMA
            # proxy, so publish generic global stores before the resident barrier.
            cute.arch.fence_proxy("async.global")
            self._resident_grid_barrier(
                barrier_count, barrier_epoch, Int32(gdim_z), is_cta_leader,
            )
            cute.arch.fence_proxy("async.global")

        gA = cute.local_tile(mA, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None))
        # Single tiled view over concatenated w13 [2*I_tp, K, E].
        # W13 is packed as [up, gate] across the concatenated N dimension.
        # Up tiles: N-indices 0..gate_tile_cnt-1
        # Gate tiles: N-indices gate_tile_cnt..2*gate_tile_cnt-1
        gB_w13_tiled = cute.local_tile(mB_w13, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None))
        gSFA = cute.local_tile(mSFA, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None))
        gSFB_w13_tiled = cute.local_tile(mSFB_w13, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None))
        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord[1]
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord[0]

        tAsA, tAgA = cpasync.tma_partition(tma_a, a_cta_crd, a_cta_layout, cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tAsSFA, tAgSFA = cpasync.tma_partition(tma_sfa, a_cta_crd, a_cta_layout, cute.group_modes(sSFA, 0, 2), cute.group_modes(gSFA, 0, 2))
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # Single w13 TMA partition (gate+up concatenated)
        tBsB_w13, tBgB_w13 = cpasync.tma_partition(
            tma_b_w13, b_cta_crd, b_cta_layout,
            cute.group_modes(sB, 0, 2), cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsB_w13_up, _ = cpasync.tma_partition(
            tma_b_w13, b_cta_crd, b_cta_layout,
            cute.group_modes(sB_up, 0, 2), cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsSFB_w13, tBgSFB_w13 = cpasync.tma_partition(
            tma_sfb_w13, b_cta_crd, b_cta_layout,
            cute.group_modes(sSFB, 0, 2), cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFB_w13_up, _ = cpasync.tma_partition(
            tma_sfb_w13, b_cta_crd, b_cta_layout,
            cute.group_modes(sSFB_up, 0, 2), cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsB_w13 = cute.filter_zeros(tBsB_w13)
        tBsB_w13_up = cute.filter_zeros(tBsB_w13_up)
        tBsSFB_w13 = cute.filter_zeros(tBsSFB_w13)
        tBgSFB_w13 = cute.filter_zeros(tBgSFB_w13)
        tBsSFB_w13_up = cute.filter_zeros(tBsSFB_w13_up)
        gate_sfb_copy_elems = cute.size(tBsSFB_w13)
        up_sfb_copy_elems = cute.size(tBsSFB_w13_up)

        # B_down TMA partitions
        gB_down = cute.local_tile(mB_down, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None))
        gSFB_down = cute.local_tile(mSFB_down, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None))
        tBsB_down, tBgB_down = cpasync.tma_partition(tma_b_down, b_cta_crd, b_cta_layout, cute.group_modes(sB, 0, 2), cute.group_modes(gB_down, 0, 2))
        tBsSFB_down, tBgSFB_down = cpasync.tma_partition(tma_sfb_down, b_cta_crd, b_cta_layout, cute.group_modes(sSFB, 0, 2), cute.group_modes(gSFB_down, 0, 2))
        tBsSFB_down = cute.filter_zeros(tBsSFB_down)
        tBgSFB_down = cute.filter_zeros(tBgSFB_down)

        # MMA fragment partitions
        tCsA = thr_mma.partition_A(sA)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrSFA = self._dense_cls._partition_fragment_SFA(self, sSFA[None, None, 0], thr_mma, tidx)
        tCsB = thr_mma.partition_B(sB)
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFB = self._dense_cls._partition_fragment_SFB(self, sSFB[None, None, 0], thr_mma, tidx)

        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        sub_shape = tCsC_for_shape.shape[:3]
        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
        gate_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        k_tile_cnt = cute.size(gA, mode=[3])
        fc1_k_tile_cnt = k_tile_cnt
        # w13 has 2*I_tp/tile_N N-tiles. Gate = first half, up = second half.
        intermediate_tile_cnt = cute.size(gB_w13_tiled, mode=[2])
        sfb_intermediate_tile_cnt = cute.size(gSFB_w13_tiled, mode=[2])
        gate_tile_cnt = intermediate_tile_cnt // Int32(2)
        sfb_gate_tile_cnt = sfb_intermediate_tile_cnt // Int32(2)
        output_tile_cnt = cute.size(gB_down, mode=[2])
        if self.skip_phase1 and flat_tid == Int32(0):
            topk_ids_flat[Int32(3)] = intermediate_tile_cnt
            topk_ids_flat[Int32(4)] = gate_tile_cnt
            topk_ids_flat[Int32(7)] = sfb_intermediate_tile_cnt
            topk_ids_flat[Int32(8)] = sfb_gate_tile_cnt

        prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
        cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
        up_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
        up_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
        phase2_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
        phase2_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)

        # ===================================================================
        # MMA WARP GROUP (warps 0-3)
        # ===================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            num_k_blocks = cute.size(tCrA, mode=[2])

            atom_ld_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4), self.a_dtype)
            atom_ld_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4), self.b_dtype)
            smem_copy_A = cute.make_tiled_copy_A(atom_ld_A, tiled_mma)
            smem_copy_B = cute.make_tiled_copy_B(atom_ld_B, tiled_mma)
            atom_ld_SF = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
            smem_copy_SFA = cute.make_tiled_copy(
                atom_ld_SF, self._dense_cls._get_layoutSFA_TV(self, tiled_mma),
                (cute.size(tiled_mma.permutation_mnk[0]), cute.size(tiled_mma.permutation_mnk[2])),
            )
            smem_copy_SFB = cute.make_tiled_copy(
                atom_ld_SF, self._dense_cls._get_layoutSFB_TV(self, tiled_mma),
                (cute.size(tiled_mma.permutation_mnk[1]), cute.size(tiled_mma.permutation_mnk[2])),
            )

            thr_ld_A = smem_copy_A.get_slice(tidx)
            thr_ld_B = smem_copy_B.get_slice(tidx)
            csA = thr_ld_A.partition_S(sA)
            crA = thr_ld_A.retile(tCrA)
            csB = thr_ld_B.partition_S(sB)
            csB_up = thr_ld_B.partition_S(sB_up)
            crB = thr_ld_B.retile(tCrB)

            thr_ld_SFA = smem_copy_SFA.get_slice(tidx)
            thr_ld_SFB = smem_copy_SFB.get_slice(tidx)
            csSFA = thr_ld_SFA.partition_S(sSFA)
            crSFA = thr_ld_SFA.retile(tCrSFA)
            csSFB = thr_ld_SFB.partition_S(sSFB)
            csSFB_up = thr_ld_SFB.partition_S(sSFB_up)
            crSFB = thr_ld_SFB.retile(tCrSFB)
            csA_phase2_write = csA[None, None, None, 0]
            src_a_phase2_u8 = cute.flatten(
                cute.recast_tensor(csA_phase2_write, cutlass.Uint8)
            )
            csSFA_phase2_write = cute.filter_zeros(csSFA[None, None, None, 0])
            src_sfa_phase2_u8 = cute.flatten(
                cute.recast_tensor(csSFA_phase2_write, cutlass.Uint8)
            )

            num_persistent_clusters = Int32(gdim_z)
            cluster_shape_mn = (
                Int32(self.cluster_shape_mn[0]),
                Int32(self.cluster_shape_mn[1]),
            )
            cta_id_in_cluster = (
                Int32(bidx % cluster_shape_mn[0]),
                Int32(bidy % cluster_shape_mn[1]),
                Int32(0),
            )
            current_work_linear_idx = Int32(bidz)
            current_local_expert_idx = Int32(0)
            accum_tile_m = Int32(0)
            tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = _compact_static_get_work_tile(
                row_counts,
                active_expert_count,
                num_tiles_n=Int32(self.output_tile_count_n),
                cluster_shape_mn=cluster_shape_mn,
                current_work_linear_idx=current_work_linear_idx,
                current_local_expert_idx=current_local_expert_idx,
                accum_tile_m=accum_tile_m,
                cta_id_in_cluster=cta_id_in_cluster,
            )

            while is_valid_tile:
                if self.skip_phase1 and tidx == Int32(0):
                    atomic_add_global_i32(get_ptr_as_int64(topk_ids_flat, Int32(0)), Int32(1))
                # tile_coord = (m_tile, intermediate_slice, local_expert_idx)
                local_expert_idx = tile_coord[2]
                weight_expert_idx = weight_expert_ids[local_expert_idx]
                alpha_value = cutlass.Float32(0.0)
                alpha_offset = local_expert_idx * fc1_tiles_per_expert + tile_coord[0]
                alpha_value = fc1_tile_alpha_flat[alpha_offset].to(cutlass.Float32)
                valid_rows = row_counts[local_expert_idx]
                if self.skip_phase1 and bidz == Int32(0) and tidx == Int32(0) and current_work_linear_idx == Int32(0):
                    topk_ids_flat[Int32(1)] = valid_rows
                    topk_ids_flat[Int32(2)] = weight_expert_idx
                    topk_weights_flat[Int32(0)] = alpha_value
                tile_m_base = tile_coord[0] * Int32(self.tile_shape_mnk[0])
                intermediate_slice = tile_coord[1]
                valid_tile_rows = valid_rows - tile_m_base
                if valid_tile_rows > Int32(_COMPACT_STATIC_TILE_M):
                    valid_tile_rows = Int32(_COMPACT_STATIC_TILE_M)
                if valid_tile_rows < Int32(0):
                    valid_tile_rows = Int32(0)

                cache_row = Int32(tidx)
                if cache_row < Int32(_COMPACT_STATIC_TILE_M):
                    tok = Int32(0)
                    wv = cutlass.Float32(0.0)
                    if cache_row < valid_tile_rows:
                        tok = token_map[local_expert_idx, tile_m_base + cache_row].to(Int32)
                        wv = token_weights[local_expert_idx, tile_m_base + cache_row].to(cutlass.Float32)
                    _st_shared_i32(scatter_tok_base_addr + cache_row * Int32(4), tok)
                    _st_shared_f32(scatter_weight_base_addr + cache_row * Int32(4), wv)
                self.epilog_sync_barrier.arrive_and_wait()

                _is_m_major = self.c_layout.is_m_major_c()
                copy_atom_r2s = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2), cutlass.BFloat16,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rGate = tiled_copy_r2s.retile(gate_acc)
                tRS_rUp = tiled_copy_r2s.retile(up_acc)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)

                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rGate, mode=[2])
                epi_buffer = Int32(0)

                down_alpha_value = down_alpha[weight_expert_idx].to(cutlass.Float32)
                down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

                epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
                MmaMPerEpiM = self.epi_tile[0] // mma_tile_m
                MmaNPerEpiN = self.epi_tile[1] // mma_tile_n

                # ============================================================
                # PHASE A: FC1 for this slice (gate + up)
                # ============================================================

                # Gate GEMM (inlined to avoid @cute.jit pass-by-value for acc)
                fz_crSFA = cute.filter_zeros(crSFA)
                fz_crSFB = cute.filter_zeros(crSFB)
                gate_acc.fill(0.0)
                cons_state.reset_count()
                peek = ml_pipeline.consumer_try_wait(cons_state)
                ml_pipeline.consumer_wait(cons_state, peek)
                csA_p = csA[None, None, None, cons_state.index]
                csB_p = csB[None, None, None, cons_state.index]
                csSFA_p = csSFA[None, None, None, cons_state.index]
                csSFB_p = csSFB[None, None, None, cons_state.index]
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and Int32(tidx) == Int32(0)
                ):
                    gate_first_off = Int32(-1)
                    gate_first_val = Int32(0)
                    up_first_off = Int32(-1)
                    up_first_val = Int32(0)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(512):
                        gate_v = Int32(_ld_shared_u8(sfb_base_addr + scan_idx))
                        up_v = Int32(_ld_shared_u8(sfb_up_base_addr + scan_idx))
                        if gate_first_off < Int32(0) and gate_v != Int32(0):
                            gate_first_off = scan_idx
                            gate_first_val = gate_v
                        if up_first_off < Int32(0) and up_v != Int32(0):
                            up_first_off = scan_idx
                            up_first_val = up_v
                        scan_idx += Int32(1)
                    cute.printf(
                        "gate_sfb_raw work={} local_e={} weight_e={} {} {} {} {} up_sfb_raw {} {} {} {} first gate(off={},val={}) up(off={},val={})",
                        current_work_linear_idx,
                        local_expert_idx,
                        weight_expert_idx,
                        Int32(_ld_shared_u8(sfb_base_addr + Int32(0))),
                        Int32(_ld_shared_u8(sfb_base_addr + Int32(1))),
                        Int32(_ld_shared_u8(sfb_base_addr + Int32(2))),
                        Int32(_ld_shared_u8(sfb_base_addr + Int32(3))),
                        Int32(_ld_shared_u8(sfb_up_base_addr + Int32(0))),
                        Int32(_ld_shared_u8(sfb_up_base_addr + Int32(1))),
                        Int32(_ld_shared_u8(sfb_up_base_addr + Int32(2))),
                        Int32(_ld_shared_u8(sfb_up_base_addr + Int32(3))),
                        gate_first_off,
                        gate_first_val,
                        up_first_off,
                        up_first_val,
                    )
                cute.copy(smem_copy_A, csA_p[None, None, 0], crA[None, None, 0])
                cute.copy(smem_copy_B, csB_p[None, None, 0], crB[None, None, 0])
                fz_csSFA_p = cute.filter_zeros(csSFA_p)
                fz_csSFB_p = cute.filter_zeros(csSFB_p)
                cute.copy(smem_copy_SFA, fz_csSFA_p[None, None, 0], fz_crSFA[None, None, 0])
                cute.copy(smem_copy_SFB, fz_csSFB_p[None, None, 0], fz_crSFB[None, None, 0])
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and Int32(tidx) == Int32(0)
                ):
                    gate_b_first_off = Int32(-1)
                    gate_b_first_val = Int32(0)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(512):
                        v = Int32(_ld_shared_u8(b_base_addr + scan_idx))
                        if gate_b_first_off < Int32(0) and v != Int32(0):
                            gate_b_first_off = scan_idx
                            gate_b_first_val = v
                        scan_idx += Int32(1)
                    cute.printf(
                        "gate_b_raw 0={} {} {} {} 8={} {} {} {} 16={} {} {} {} first(off={},val={})",
                        Int32(_ld_shared_u8(b_base_addr + Int32(0))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(1))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(2))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(3))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(8))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(9))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(10))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(11))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(16))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(17))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(18))),
                        Int32(_ld_shared_u8(b_base_addr + Int32(19))),
                        gate_b_first_off,
                        gate_b_first_val,
                    )
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and (Int32(tidx) & Int32(31)) == Int32(0)
                ):
                    gate_sfb_src = cute.flatten(
                        cute.recast_tensor(fz_csSFB_p[None, None, Int32(0)], cutlass.Uint8)
                    )
                    gate_sfb_first_off = Int32(-1)
                    gate_sfb_first_val = Int32(0)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(cute.size(gate_sfb_src)):
                        v = Int32(gate_sfb_src[scan_idx])
                        if gate_sfb_first_off < Int32(0) and v != Int32(0):
                            gate_sfb_first_off = scan_idx
                            gate_sfb_first_val = v
                        scan_idx += Int32(1)
                    gate_b_frag = cute.flatten(
                        cute.recast_tensor(crB[None, None, Int32(0)], cutlass.Uint32)
                    )
                    gate_sfb_frag = cute.flatten(
                        cute.recast_tensor(fz_crSFB[None, None, Int32(0)], cutlass.Uint32)
                    )
                    cute.printf(
                        "gate_frag warp={} src={} {} {} {} first(off={},val={}) phys={} {} {} {} b={} {} sfb={} {}",
                        warp_idx,
                        Int32(gate_sfb_src[Int32(0)]),
                        Int32(gate_sfb_src[Int32(1)]),
                        Int32(gate_sfb_src[Int32(2)]),
                        Int32(gate_sfb_src[Int32(3)]),
                        gate_sfb_first_off,
                        gate_sfb_first_val,
                        shared_ptr_to_u32(elem_pointer(gate_sfb_src, Int32(0))) - sfb_base_addr,
                        shared_ptr_to_u32(elem_pointer(gate_sfb_src, Int32(1))) - sfb_base_addr,
                        shared_ptr_to_u32(elem_pointer(gate_sfb_src, Int32(2))) - sfb_base_addr,
                        shared_ptr_to_u32(elem_pointer(gate_sfb_src, Int32(3))) - sfb_base_addr,
                        Int32(gate_b_frag[Int32(0)]),
                        Int32(gate_b_frag[Int32(1)]),
                        Int32(gate_sfb_frag[Int32(0)]),
                        Int32(gate_sfb_frag[Int32(1)]),
                    )
                for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        if k_block_idx == num_k_blocks - 1:
                            ml_pipeline.consumer_release(cons_state)
                            cons_state.advance()
                            peek = ml_pipeline.consumer_try_wait(cons_state)
                            csA_p = csA[None, None, None, cons_state.index]
                            csB_p = csB[None, None, None, cons_state.index]
                            csSFA_p = csSFA[None, None, None, cons_state.index]
                            csSFB_p = csSFB[None, None, None, cons_state.index]
                            fz_csSFA_p = cute.filter_zeros(csSFA_p)
                            fz_csSFB_p = cute.filter_zeros(csSFB_p)
                            ml_pipeline.consumer_wait(cons_state, peek)
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                mma_atom.set(WarpField.SFA, tCrSFA[None, _mt, k_block_idx].iterator)
                                mma_atom.set(WarpField.SFB, tCrSFB[None, _nt, k_block_idx].iterator)
                                cute.gemm(
                                    mma_atom,
                                    gate_acc[None, _mt, _nt],
                                    tCrA[None, _mt, k_block_idx],
                                    tCrB[None, _nt, k_block_idx],
                                    gate_acc[None, _mt, _nt],
                                )
                        cute.copy(smem_copy_A, csA_p[None, None, k_next], crA[None, None, k_next])
                        cute.copy(smem_copy_B, csB_p[None, None, k_next], crB[None, None, k_next])
                        fz_csSFA_cur = cute.filter_zeros(csSFA[None, None, None, cons_state.index])
                        fz_csSFB_cur = cute.filter_zeros(csSFB[None, None, None, cons_state.index])
                        cute.copy(smem_copy_SFA, fz_csSFA_cur[None, None, k_next], fz_crSFA[None, None, k_next])
                        cute.copy(smem_copy_SFB, fz_csSFB_cur[None, None, k_next], fz_crSFB[None, None, k_next])
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    if k_block_idx == num_k_blocks - 1:
                        ml_pipeline.consumer_release(cons_state)
                        cons_state.advance()
                    if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                        cute.copy(smem_copy_A, csA_p[None, None, k_next], crA[None, None, k_next])
                        cute.copy(smem_copy_B, csB_p[None, None, k_next], crB[None, None, k_next])
                        cute.copy(smem_copy_SFA, fz_csSFA_p[None, None, k_next], fz_crSFA[None, None, k_next])
                        cute.copy(smem_copy_SFB, fz_csSFB_p[None, None, k_next], fz_crSFB[None, None, k_next])
                    for _mt in range(self.num_m_tiles):
                        for _nt in range(self.num_n_tiles):
                            mma_atom.set(WarpField.SFA, tCrSFA[None, _mt, k_block_idx].iterator)
                            mma_atom.set(WarpField.SFB, tCrSFB[None, _nt, k_block_idx].iterator)
                            cute.gemm(
                                mma_atom,
                                gate_acc[None, _mt, _nt],
                                tCrA[None, _mt, k_block_idx],
                                tCrB[None, _nt, k_block_idx],
                                gate_acc[None, _mt, _nt],
                            )
                # Gate and up share the A/SFA staging buffers. Drain gate
                # consumers before the DMA warp starts refilling those stages
                # for the up pass.
                self.pass_sync_barrier.arrive_and_wait()

                # Up GEMM (inlined, same pattern)
                up_acc.fill(0.0)
                up_cons_state.reset_count()
                peek = up_pipeline.consumer_try_wait(up_cons_state)
                up_pipeline.consumer_wait(up_cons_state, peek)
                csA_p = csA[None, None, None, up_cons_state.index]
                csB_p = csB_up[None, None, None, up_cons_state.index]
                csSFA_p = csSFA[None, None, None, up_cons_state.index]
                csSFB_p = csSFB_up[None, None, None, up_cons_state.index]
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and Int32(tidx) == Int32(0)
                ):
                    cute.printf(
                        "up_sfb_raw work={} local_e={} weight_e={} {} {} {} {}",
                        current_work_linear_idx,
                        local_expert_idx,
                        weight_expert_idx,
                        Int32(_ld_shared_u8(sfb_up_base_addr + Int32(0))),
                        Int32(_ld_shared_u8(sfb_up_base_addr + Int32(1))),
                        Int32(_ld_shared_u8(sfb_up_base_addr + Int32(2))),
                        Int32(_ld_shared_u8(sfb_up_base_addr + Int32(3))),
                    )
                cute.copy(smem_copy_A, csA_p[None, None, 0], crA[None, None, 0])
                cute.copy(smem_copy_B, csB_p[None, None, 0], crB[None, None, 0])
                fz_csSFA_p = cute.filter_zeros(csSFA_p)
                fz_csSFB_p = cute.filter_zeros(csSFB_p)
                cute.copy(smem_copy_SFA, fz_csSFA_p[None, None, 0], fz_crSFA[None, None, 0])
                cute.copy(smem_copy_SFB, fz_csSFB_p[None, None, 0], fz_crSFB[None, None, 0])
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and Int32(tidx) == Int32(0)
                ):
                    up_b_first_off = Int32(-1)
                    up_b_first_val = Int32(0)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(512):
                        v = Int32(_ld_shared_u8(b_up_base_addr + scan_idx))
                        if up_b_first_off < Int32(0) and v != Int32(0):
                            up_b_first_off = scan_idx
                            up_b_first_val = v
                        scan_idx += Int32(1)
                    cute.printf(
                        "up_b_raw 0={} {} {} {} 8={} {} {} {} 16={} {} {} {} first(off={},val={})",
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(0))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(1))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(2))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(3))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(8))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(9))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(10))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(11))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(16))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(17))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(18))),
                        Int32(_ld_shared_u8(b_up_base_addr + Int32(19))),
                        up_b_first_off,
                        up_b_first_val,
                    )
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and (Int32(tidx) & Int32(31)) == Int32(0)
                ):
                    up_sfb_src = cute.flatten(
                        cute.recast_tensor(fz_csSFB_p[None, None, Int32(0)], cutlass.Uint8)
                    )
                    up_sfb_first_off = Int32(-1)
                    up_sfb_first_val = Int32(0)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(cute.size(up_sfb_src)):
                        v = Int32(up_sfb_src[scan_idx])
                        if up_sfb_first_off < Int32(0) and v != Int32(0):
                            up_sfb_first_off = scan_idx
                            up_sfb_first_val = v
                        scan_idx += Int32(1)
                    up_b_frag = cute.flatten(
                        cute.recast_tensor(crB[None, None, Int32(0)], cutlass.Uint32)
                    )
                    up_sfb_frag = cute.flatten(
                        cute.recast_tensor(fz_crSFB[None, None, Int32(0)], cutlass.Uint32)
                    )
                    cute.printf(
                        "up_frag warp={} src={} {} {} {} first(off={},val={}) phys={} {} {} {} b={} {} sfb={} {}",
                        warp_idx,
                        Int32(up_sfb_src[Int32(0)]),
                        Int32(up_sfb_src[Int32(1)]),
                        Int32(up_sfb_src[Int32(2)]),
                        Int32(up_sfb_src[Int32(3)]),
                        up_sfb_first_off,
                        up_sfb_first_val,
                        shared_ptr_to_u32(elem_pointer(up_sfb_src, Int32(0))) - sfb_up_base_addr,
                        shared_ptr_to_u32(elem_pointer(up_sfb_src, Int32(1))) - sfb_up_base_addr,
                        shared_ptr_to_u32(elem_pointer(up_sfb_src, Int32(2))) - sfb_up_base_addr,
                        shared_ptr_to_u32(elem_pointer(up_sfb_src, Int32(3))) - sfb_up_base_addr,
                        Int32(up_b_frag[Int32(0)]),
                        Int32(up_b_frag[Int32(1)]),
                        Int32(up_sfb_frag[Int32(0)]),
                        Int32(up_sfb_frag[Int32(1)]),
                    )
                for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        if k_block_idx == num_k_blocks - 1:
                            up_pipeline.consumer_release(up_cons_state)
                            up_cons_state.advance()
                            peek = up_pipeline.consumer_try_wait(up_cons_state)
                            csA_p = csA[None, None, None, up_cons_state.index]
                            csB_p = csB_up[None, None, None, up_cons_state.index]
                            csSFA_p = csSFA[None, None, None, up_cons_state.index]
                            csSFB_p = csSFB_up[None, None, None, up_cons_state.index]
                            fz_csSFA_p = cute.filter_zeros(csSFA_p)
                            fz_csSFB_p = cute.filter_zeros(csSFB_p)
                            up_pipeline.consumer_wait(up_cons_state, peek)
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                mma_atom.set(WarpField.SFA, tCrSFA[None, _mt, k_block_idx].iterator)
                                mma_atom.set(WarpField.SFB, tCrSFB[None, _nt, k_block_idx].iterator)
                                cute.gemm(
                                    mma_atom,
                                    up_acc[None, _mt, _nt],
                                    tCrA[None, _mt, k_block_idx],
                                    tCrB[None, _nt, k_block_idx],
                                    up_acc[None, _mt, _nt],
                                )
                        cute.copy(smem_copy_A, csA_p[None, None, k_next], crA[None, None, k_next])
                        cute.copy(smem_copy_B, csB_p[None, None, k_next], crB[None, None, k_next])
                        cute.copy(smem_copy_SFA, fz_csSFA_p[None, None, k_next], fz_crSFA[None, None, k_next])
                        cute.copy(smem_copy_SFB, fz_csSFB_p[None, None, k_next], fz_crSFB[None, None, k_next])
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    if k_block_idx == num_k_blocks - 1:
                        up_pipeline.consumer_release(up_cons_state)
                        up_cons_state.advance()
                    if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                        cute.copy(smem_copy_A, csA_p[None, None, k_next], crA[None, None, k_next])
                        cute.copy(smem_copy_B, csB_p[None, None, k_next], crB[None, None, k_next])
                        cute.copy(smem_copy_SFA, fz_csSFA_p[None, None, k_next], fz_crSFA[None, None, k_next])
                        cute.copy(smem_copy_SFB, fz_csSFB_p[None, None, k_next], fz_crSFB[None, None, k_next])
                    for _mt in range(self.num_m_tiles):
                        for _nt in range(self.num_n_tiles):
                            mma_atom.set(WarpField.SFA, tCrSFA[None, _mt, k_block_idx].iterator)
                            mma_atom.set(WarpField.SFB, tCrSFB[None, _nt, k_block_idx].iterator)
                            cute.gemm(
                                mma_atom,
                                up_acc[None, _mt, _nt],
                                tCrA[None, _mt, k_block_idx],
                                tCrB[None, _nt, k_block_idx],
                                up_acc[None, _mt, _nt],
                            )
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and (Int32(tidx) & Int32(31)) == Int32(0)
                ):
                    gate_acc_max = cutlass.Float32(0.0)
                    up_acc_max = cutlass.Float32(0.0)
                    for idx in cutlass.range_constexpr(cute.size(gate_acc)):
                        gate_acc_max = fmax_f32(gate_acc_max, fabs_f32(gate_acc[idx]))
                    for idx in cutlass.range_constexpr(cute.size(up_acc)):
                        up_acc_max = fmax_f32(up_acc_max, fabs_f32(up_acc[idx]))
                    cute.printf(
                        "fc1_acc warp={} gate_max={} up_max={}",
                        warp_idx,
                        gate_acc_max,
                        up_acc_max,
                    )
                # SiLU + quant into sA
                sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
                packed_cols = Int32(self.tile_shape_mnk[2] // 2)
                sf_blocks_per_row = Int32(self.tile_shape_mnk[2] // 16)
                gs_value = global_scale[weight_expert_idx].to(cutlass.Float32)
                if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(0.0):
                    if self.fast_math:
                        gs_value = rcp_approx_ftz(gs_value)
                    else:
                        gs_value = cutlass.Float32(1.0) / gs_value
                # FC2 tile-amax is a compact-kernel specialization. The current
                # epilogue shape has exactly one M-slice per tile, so a single
                # tile-local alpha is sufficient for the whole cached sA tile.
                fc2_down_alpha_value = down_alpha_value

                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    epi_m_valid = valid_rows - tile_m_base - Int32(epi_m) * Int32(self.epi_tile[0])
                    silu_epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                    if epi_m_valid > Int32(0):
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                mma_n = mma_n_in_epi
                                tRS_rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                                gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                if (
                                    self.skip_phase1
                                    and bidz == Int32(0)
                                    and current_work_linear_idx == Int32(0)
                                    and epi_m == Int32(0)
                                    and mma_m_in_epi == Int32(0)
                                    and mma_n_in_epi == Int32(0)
                                    and (Int32(tidx) & Int32(31)) == Int32(0)
                                ):
                                    gate_abs_max = cutlass.Float32(0.0)
                                    up_abs_max = cutlass.Float32(0.0)
                                    for sample_idx in cutlass.range_constexpr(cute.size(gate_slice)):
                                        gate_abs_max = fmax_f32(gate_abs_max, fabs_f32(alpha_value * gate_slice[sample_idx]))
                                        up_abs_max = fmax_f32(up_abs_max, fabs_f32(alpha_value * up_slice[sample_idx]))
                                    topk_weights_flat[Int32(1)] = gate_abs_max
                                    topk_weights_flat[Int32(2)] = up_abs_max
                                for elem_idx in cutlass.range_constexpr(cute.size(tRS_rD_slice)):
                                    g = alpha_value * gate_slice[elem_idx]
                                    u = alpha_value * up_slice[elem_idx]
                                    sigmoid_g = cute.arch.rcp_approx(
                                        cutlass.Float32(1.0) + cute.math.exp(-g, fastmath=self.fast_math),
                                    )
                                    tRS_rD_slice[elem_idx] = g * sigmoid_g * u

                        acc_vec = tRS_rD.load()
                        acc_vec = acc_vec.to(cutlass.BFloat16)
                        tRS_rD_out.store(acc_vec)
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, silu_epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                    epi_rows = epi_m_valid
                    if epi_rows > Int32(self.epi_tile[0]):
                        epi_rows = Int32(self.epi_tile[0])
                    if epi_rows < Int32(0):
                        epi_rows = Int32(0)
                    quant_gs_value = gs_value
                    if self.fc2_tile_amax and epi_rows > Int32(0):
                        lane_id = Int32(tidx) & Int32(31)
                        local_tile_amax = cutlass.Float32(0.0)
                        reduce_idx = Int32(tidx)
                        while reduce_idx < epi_rows * sf_blocks_per_row:
                            local_row = reduce_idx // sf_blocks_per_row
                            sf_block = reduce_idx - local_row * sf_blocks_per_row
                            block_start = sf_block * Int32(16)
                            block_amax = cutlass.Float32(0.0)
                            for elem_idx in cutlass.range_constexpr(16):
                                value = cutlass.Float32(
                                    sC[local_row, block_start + elem_idx, silu_epi_buffer]
                                )
                                block_amax = fmax_f32(block_amax, fabs_f32(value))
                            local_tile_amax = fmax_f32(local_tile_amax, block_amax)
                            reduce_idx += Int32(self.threads_per_cta)

                        warp_tile_amax = warp_reduce(local_tile_amax, fmax_f32)
                        if lane_id == Int32(0):
                            _st_shared_f32(ctrl_base_addr + warp_idx * Int32(4), warp_tile_amax)
                        self.epilog_sync_barrier.arrive_and_wait()

                        if warp_idx == 0:
                            tile_amax = cutlass.Float32(0.0)
                            if lane_id < Int32(self.num_mma_warps):
                                tile_amax = _ld_shared_f32(ctrl_base_addr + lane_id * Int32(4))
                            tile_amax = warp_reduce(tile_amax, fmax_f32)
                            if lane_id == Int32(0):
                                _st_shared_f32(ctrl_base_addr, tile_amax)
                        self.epilog_sync_barrier.arrive_and_wait()

                        tile_amax = _ld_shared_f32(ctrl_base_addr)
                        if tile_amax != cutlass.Float32(0.0) and gs_value != cutlass.Float32(0.0):
                            tile_gs_value = tile_amax * cutlass.Float32(_FC2_TILE_AMAX_GS_RCP)
                            fc2_down_alpha_value = down_alpha_value * (tile_gs_value / gs_value)
                            quant_gs_value = tile_gs_value

                    quant_lane = Int32(tidx) & Int32(31)
                    quant_idx = Int32(0)
                    quant_stride = Int32(1)
                    if self.skip_phase1:
                        quant_idx = Int32(0) if quant_lane == Int32(0) else epi_rows * sf_blocks_per_row
                        quant_stride = Int32(1)
                    else:
                        quant_idx = Int32(tidx)
                        quant_stride = Int32(self.num_mma_warps * self.num_threads_per_warp)
                    while quant_idx < epi_rows * sf_blocks_per_row:
                        local_row = quant_idx // sf_blocks_per_row
                        row = rows_offset + local_row
                        sf_block = quant_idx - local_row * sf_blocks_per_row
                        block_start = sf_block * Int32(16)

                        values = cute.make_rmem_tensor((16,), cutlass.Float32)
                        block_max = cutlass.Float32(0.0)
                        for elem_idx in cutlass.range_constexpr(16):
                            value = cutlass.Float32(
                                sC[local_row, block_start + elem_idx, silu_epi_buffer]
                            )
                            values[elem_idx] = value
                            block_max = fmax_f32(block_max, fabs_f32(value))

                        packed64 = Uint64(0)
                        scale_byte = Uint8(0)
                        if self.fast_math:
                            packed64, scale_byte = quantize_block_fp4_fast(values, block_max, quant_gs_value)
                        else:
                            packed64, scale_byte = quantize_block_fp4(values, block_max, quant_gs_value)
                        if (
                            self.skip_phase1
                            and bidz == Int32(0)
                            and current_work_linear_idx == Int32(0)
                            and row == Int32(0)
                            and sf_block == Int32(0)
                            and (Int32(tidx) & Int32(31)) == Int32(0)
                        ):
                            cute.printf(
                                "fused_fc2_quant_in warp={} tidx={} block_max={} v0={} v1={} v2={} v3={}",
                                warp_idx,
                                tidx,
                                block_max,
                                values[Int32(0)],
                                values[Int32(1)],
                                values[Int32(2)],
                                values[Int32(3)],
                            )
                            topk_weights_flat[Int32(5)] = block_max
                            topk_weights_flat[Int32(6)] = quant_gs_value
                            topk_weights_flat[Int32(7)] = fc2_down_alpha_value
                        packed_base = sf_block << Int32(3)
                        dst_pcol = row & Int32(63)
                        xor_bits = ((dst_pcol >> Int32(1)) & Int32(0x3)) << Int32(4)
                        row_high = row >> Int32(6)
                        for byte_idx in cutlass.range_constexpr(8):
                            src_pcol = packed_base + Int32(byte_idx)
                            dst_row = ((src_pcol ^ xor_bits) << Int32(1)) + row_high
                            dst_flat = dst_row * packed_cols + dst_pcol
                            byte_val = Uint8(
                                (packed64 >> Uint64(byte_idx * 8)) & Uint64(0xFF)
                            )
                            owner_tidx = (dst_flat & Int32(15)) + (((dst_flat >> Int32(4)) & Int32(1)) << Int32(5))
                            src_idx = dst_flat >> Int32(7)
                            if (
                                self.skip_phase1
                                and bidz == Int32(0)
                                and current_work_linear_idx == Int32(0)
                                and row == Int32(0)
                                and sf_block == Int32(0)
                                and byte_idx == Int32(0)
                                and (Int32(tidx) & Int32(31)) == Int32(0)
                            ):
                                cute.printf(
                                    "fused_fc2_write warp={} tidx={} src_idx={} src_a_base={} owner_tidx={} dst_flat={} byte={}",
                                    warp_idx,
                                    tidx,
                                    src_idx,
                                    shared_ptr_to_u32(elem_pointer(src_a_phase2_u8, Int32(0))) - a_base_addr,
                                    owner_tidx,
                                    dst_flat,
                                    Int32(byte_val),
                                )
                            if self.skip_phase1:
                                src_a_phase2_u8[src_idx] = byte_val
                            else:
                                sA_u8[dst_flat] = byte_val

                        outer_m_idx = row % Int32(32)
                        inner_m_idx = row // Int32(32)
                        inner_k_idx = sf_block % Int32(4)
                        k_tile_idx = sf_block // Int32(4)
                        sf_raw_idx = (
                            k_tile_idx * Int32(32 * 4 * 4)
                            + outer_m_idx * Int32(4 * 4)
                            + inner_m_idx * Int32(4)
                            + inner_k_idx
                        )
                        if self.skip_phase1:
                            sfa_src_base = shared_ptr_to_u32(
                                elem_pointer(
                                    src_sfa_phase2_u8,
                                    Int32(k_tile_idx * 16),
                                )
                            )
                            if (
                                bidz == Int32(0)
                                and current_work_linear_idx == Int32(0)
                                and row == Int32(0)
                                and sf_block == Int32(0)
                                and (Int32(tidx) & Int32(31)) == Int32(0)
                            ):
                                cute.printf(
                                    "fused_fc2_sfa_write warp={} tidx={} sfa_src_base={} inner_k={} byte={}",
                                    warp_idx,
                                    tidx,
                                    sfa_src_base - sfa_base_addr,
                                    inner_k_idx,
                                    Int32(scale_byte),
                                )
                            st_shared_u8(sfa_src_base + inner_k_idx, scale_byte)
                        else:
                            st_shared_u8(sfa_base_addr + sf_raw_idx, scale_byte)
                        quant_idx += quant_stride
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and rows_offset == Int32(0)
                    and (Int32(tidx) & Int32(31)) == Int32(0)
                ):
                    sA_u8_max = Int32(0)
                    for sample_idx in cutlass.range_constexpr(64):
                        byte_val = Int32(sA_u8[Int32(sample_idx)])
                        if byte_val > sA_u8_max:
                            sA_u8_max = byte_val
                    topk_ids_flat[Int32(5)] = sA_u8_max
                    sSFA_u8_max = Int32(0)
                    sample_count = Int32(64)
                    sample_idx = Int32(0)
                    while sample_idx < sample_count:
                        byte_val = Int32(_ld_shared_u8(sfa_base_addr + sample_idx))
                        if byte_val > sSFA_u8_max:
                            sSFA_u8_max = byte_val
                        sample_idx += Int32(1)
                    topk_ids_flat[Int32(6)] = sSFA_u8_max
                    cute.printf(
                        "fused_fc2_a work={} local_e={} weight_e={} bytes={} {} {} {} sfa={} {} {} {}",
                        current_work_linear_idx,
                        local_expert_idx,
                        weight_expert_idx,
                        Int32(sA_u8[Int32(0)]),
                        Int32(sA_u8[Int32(1)]),
                        Int32(sA_u8[Int32(2)]),
                        Int32(sA_u8[Int32(3)]),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(0))),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(1))),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(2))),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(3))),
                    )
                    cute.printf(
                        "fused_fc2_a tail work={} local_e={} weight_e={} bytes={} {} {} {} sfa={} {} {} {}",
                        current_work_linear_idx,
                        local_expert_idx,
                        weight_expert_idx,
                        Int32(sA_u8[Int32(512)]),
                        Int32(sA_u8[Int32(513)]),
                        Int32(sA_u8[Int32(514)]),
                        Int32(sA_u8[Int32(515)]),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(512))),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(513))),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(514))),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(515))),
                    )
                cute.arch.fence_proxy("async.shared", space="cta")
                # epilog_sync: MMA-only barrier. DMA warp doesn't need to wait
                # for quant — it only loads B_down into sB (separate buffer).
                # This allows DMA to prefetch B_down tiles earlier.
                self.epilog_sync_barrier.arrive_and_wait()
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and (Int32(tidx) & Int32(31)) == Int32(0)
                ):
                    raw_a_first_off2 = Int32(-1)
                    raw_a_first_val2 = Int32(0)
                    raw_sfa_first_off2 = Int32(-1)
                    raw_sfa_first_val2 = Int32(0)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(cute.size(cute.flatten(cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)))):
                        v = Int32(_ld_shared_u8(a_base_addr + scan_idx))
                        if raw_a_first_off2 < Int32(0) and v != Int32(0):
                            raw_a_first_off2 = scan_idx
                            raw_a_first_val2 = v
                        scan_idx += Int32(1)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(cute.size(cute.flatten(cute.recast_tensor(sSFA[None, None, 0], cutlass.Uint8)))):
                        v = Int32(_ld_shared_u8(sfa_base_addr + scan_idx))
                        if raw_sfa_first_off2 < Int32(0) and v != Int32(0):
                            raw_sfa_first_off2 = scan_idx
                            raw_sfa_first_val2 = v
                        scan_idx += Int32(1)
                    cute.printf(
                        "fused_fc2_prephase2_raw work={} local_e={} weight_e={} first_a(off={},val={}) first_sfa(off={},val={}) base_delta_a_sfa={} base_delta_sfa_sfb={} base_delta_sfb_c={}",
                        current_work_linear_idx,
                        local_expert_idx,
                        weight_expert_idx,
                        raw_a_first_off2,
                        raw_a_first_val2,
                        raw_sfa_first_off2,
                        raw_sfa_first_val2,
                        sfa_base_addr - a_base_addr,
                        sfb_base_addr - sfa_base_addr,
                        c_base_addr - sfb_base_addr,
                    )

                # ============================================================
                # PHASE B: Sweep ALL FC2 output tiles using cached sA
                # No CTA-wide barrier needed here: gate is done with sB/sSFB
                # (barrier at line 925 ensured that), up uses sB_up/sSFB_up,
                # and DMA's B_down loads into sB/sSFB don't conflict with
                # MMA's SiLU+quant on sC/sA/sSFA. The phase2_pipeline
                # handles B_down availability for FC2 GEMM.
                # ============================================================
                scatter_N = Int32(scatter_output.shape[1])
                lane_id = Int32(tidx) & Int32(31)
                warp_in_tile = Int32(tidx) >> Int32(5)
                warp_m_base = (warp_in_tile >> Int32(1)) * Int32(64)
                warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)

                csA_phase2 = csA[None, None, None, 0]
                csSFA_phase2 = csSFA[None, None, None, 0]

                # Consume all output tiles continuously from phase2_pipeline.

                # Hoist A-side register loads: sA is constant across all
                # FC2 output tiles (quantized intermediate). Load crA and
                # crSFA for all k-blocks once, reuse for all 32 tiles.
                fz_crSFA_p2 = cute.filter_zeros(crSFA)
                cute.copy(smem_copy_A, csA_phase2[None, None, 0], crA[None, None, 0])
                fz_csSFA_p2 = cute.filter_zeros(csSFA_phase2)
                if (
                    self.skip_phase1
                    and bidz == Int32(0)
                    and current_work_linear_idx == Int32(0)
                    and lane_id == Int32(0)
                ):
                    raw_a_first_off = Int32(-1)
                    raw_a_first_val = Int32(0)
                    raw_sfa_first_off = Int32(-1)
                    raw_sfa_first_val = Int32(0)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(cute.size(cute.flatten(cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)))):
                        v = Int32(_ld_shared_u8(a_base_addr + scan_idx))
                        if raw_a_first_off < Int32(0) and v != Int32(0):
                            raw_a_first_off = scan_idx
                            raw_a_first_val = v
                        scan_idx += Int32(1)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(cute.size(cute.flatten(cute.recast_tensor(sSFA[None, None, 0], cutlass.Uint8)))):
                        v = Int32(_ld_shared_u8(sfa_base_addr + scan_idx))
                        if raw_sfa_first_off < Int32(0) and v != Int32(0):
                            raw_sfa_first_off = scan_idx
                            raw_sfa_first_val = v
                        scan_idx += Int32(1)
                    cute.printf(
                        "fused_fc2_postbarrier_raw work={} local_e={} weight_e={} a={} {} {} {} first_a(off={},val={}) sfa={} {} {} {} first_sfa(off={},val={})",
                        current_work_linear_idx,
                        local_expert_idx,
                        weight_expert_idx,
                        Int32(_ld_shared_u8(a_base_addr + Int32(0))),
                        Int32(_ld_shared_u8(a_base_addr + Int32(1))),
                        Int32(_ld_shared_u8(a_base_addr + Int32(2))),
                        Int32(_ld_shared_u8(a_base_addr + Int32(3))),
                        raw_a_first_off,
                        raw_a_first_val,
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(0))),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(1))),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(2))),
                        Int32(_ld_shared_u8(sfa_base_addr + Int32(3))),
                        raw_sfa_first_off,
                        raw_sfa_first_val,
                    )
                    src_a_u8 = cute.flatten(
                        cute.recast_tensor(csA_phase2[None, None, Int32(0)], cutlass.Uint8)
                    )
                    src_sfa_u8 = cute.flatten(
                        cute.recast_tensor(fz_csSFA_p2[None, None, Int32(0)], cutlass.Uint8)
                    )
                    src_a_first_off = Int32(-1)
                    src_a_first_val = Int32(0)
                    src_sfa_first_off = Int32(-1)
                    src_sfa_first_val = Int32(0)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(cute.size(src_a_u8)):
                        v = Int32(src_a_u8[scan_idx])
                        if src_a_first_off < Int32(0) and v != Int32(0):
                            src_a_first_off = scan_idx
                            src_a_first_val = v
                        scan_idx += Int32(1)
                    scan_idx = Int32(0)
                    while scan_idx < Int32(cute.size(src_sfa_u8)):
                        v = Int32(src_sfa_u8[scan_idx])
                        if src_sfa_first_off < Int32(0) and v != Int32(0):
                            src_sfa_first_off = scan_idx
                            src_sfa_first_val = v
                        scan_idx += Int32(1)
                    cute.printf(
                        "fused_fc2_a_src {} {} {} {} first(off={},val={})",
                        Int32(src_a_u8[Int32(0)]),
                        Int32(src_a_u8[Int32(1)]),
                        Int32(src_a_u8[Int32(2)]),
                        Int32(src_a_u8[Int32(3)]),
                        src_a_first_off,
                        src_a_first_val,
                    )
                    cute.printf(
                        "fused_fc2_sfa_src {} {} {} {} first(off={},val={}) phys={} {} {} {} {} {} {} {}",
                        Int32(src_sfa_u8[Int32(0)]),
                        Int32(src_sfa_u8[Int32(1)]),
                        Int32(src_sfa_u8[Int32(2)]),
                        Int32(src_sfa_u8[Int32(3)]),
                        src_sfa_first_off,
                        src_sfa_first_val,
                        shared_ptr_to_u32(elem_pointer(src_sfa_u8, Int32(0))) - sfa_base_addr,
                        shared_ptr_to_u32(elem_pointer(src_sfa_u8, Int32(1))) - sfa_base_addr,
                        shared_ptr_to_u32(elem_pointer(src_sfa_u8, Int32(2))) - sfa_base_addr,
                        shared_ptr_to_u32(elem_pointer(src_sfa_u8, Int32(3))) - sfa_base_addr,
                        shared_ptr_to_u32(elem_pointer(src_sfa_u8, Int32(16))) - sfa_base_addr,
                        shared_ptr_to_u32(elem_pointer(src_sfa_u8, Int32(17))) - sfa_base_addr,
                        shared_ptr_to_u32(elem_pointer(src_sfa_u8, Int32(18))) - sfa_base_addr,
                        shared_ptr_to_u32(elem_pointer(src_sfa_u8, Int32(19))) - sfa_base_addr,
                    )
                cute.copy(smem_copy_SFA, fz_csSFA_p2[None, None, 0], fz_crSFA_p2[None, None, 0])
                for _kb_pre in cutlass.range_constexpr(num_k_blocks - 1):
                    k_pre = _kb_pre + 1
                    cute.copy(smem_copy_A, csA_phase2[None, None, k_pre], crA[None, None, k_pre])
                    cute.copy(smem_copy_SFA, fz_csSFA_p2[None, None, k_pre], fz_crSFA_p2[None, None, k_pre])

                phase2_cons_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):
                    phase2_peek = phase2_pipeline.consumer_try_wait(phase2_cons_state)
                    phase2_pipeline.consumer_wait(phase2_cons_state, phase2_peek)
                    csB_phase2 = csB[None, None, None, phase2_cons_state.index]
                    csSFB_phase2 = csSFB[None, None, None, phase2_cons_state.index]

                    # Only load B-side (B_down changes per output tile; A is hoisted)
                    cute.copy(smem_copy_B, csB_phase2[None, None, 0], crB[None, None, 0])
                    f2 = cute.filter_zeros(csSFB_phase2)
                    f4 = cute.filter_zeros(crSFB)
                    cute.copy(smem_copy_SFB, f2[None, None, 0], f4[None, None, 0])
                    if (
                        self.skip_phase1
                        and bidz == Int32(0)
                        and current_work_linear_idx == Int32(0)
                        and output_tile_idx == Int32(0)
                        and lane_id == Int32(0)
                    ):
                        a_frag_u32 = cute.flatten(
                            cute.recast_tensor(tCrA[None, Int32(0), Int32(0)], cutlass.Uint32)
                        )
                        sfa_frag_u32 = cute.flatten(
                            cute.recast_tensor(tCrSFA[None, Int32(0), Int32(0)], cutlass.Uint32)
                        )
                        b_frag_u32 = cute.flatten(
                            cute.recast_tensor(tCrB[None, Int32(0), Int32(0)], cutlass.Uint32)
                        )
                        sfb_frag_u32 = cute.flatten(
                            cute.recast_tensor(tCrSFB[None, Int32(0), Int32(0)], cutlass.Uint32)
                        )
                        cute.printf(
                            "fused_fc2_frag a={} {} sfa={} {} b={} {} sfb={} {}",
                            a_frag_u32[Int32(0)],
                            a_frag_u32[Int32(1)],
                            sfa_frag_u32[Int32(0)],
                            sfa_frag_u32[Int32(1)],
                            b_frag_u32[Int32(0)],
                            b_frag_u32[Int32(1)],
                            sfb_frag_u32[Int32(0)],
                            sfb_frag_u32[Int32(1)],
                        )
                        cute.printf(
                            "fused_fc2_b bytes={} {} {} {} sfb={} {} {} {}",
                            Int32(sB_u8[Int32(0)]),
                            Int32(sB_u8[Int32(1)]),
                            Int32(sB_u8[Int32(2)]),
                            Int32(sB_u8[Int32(3)]),
                            Int32(sSFB_u8[Int32(0)]),
                            Int32(sSFB_u8[Int32(1)]),
                            Int32(sSFB_u8[Int32(2)]),
                            Int32(sSFB_u8[Int32(3)]),
                        )
                    down_acc.fill(0.0)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        if k_block_idx == num_k_blocks - 1:
                            phase2_pipeline.consumer_release(phase2_cons_state)
                            phase2_cons_state.advance()
                        if k_next > 0:
                            # Only B-side for next k-block (A already in registers)
                            cute.copy(smem_copy_B, csB_phase2[None, None, k_next], crB[None, None, k_next])
                            f2 = cute.filter_zeros(csSFB_phase2)
                            f4 = cute.filter_zeros(crSFB)
                            cute.copy(smem_copy_SFB, f2[None, None, k_next], f4[None, None, k_next])
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                mma_atom.set(WarpField.SFA, tCrSFA[None, _mt, k_block_idx].iterator)
                                mma_atom.set(WarpField.SFB, tCrSFB[None, _nt, k_block_idx].iterator)
                                cute.gemm(mma_atom, down_acc[None, _mt, _nt], tCrA[None, _mt, k_block_idx], tCrB[None, _nt, k_block_idx], down_acc[None, _mt, _nt])
                    if (
                        self.skip_phase1
                        and bidz == Int32(0)
                        and current_work_linear_idx == Int32(0)
                        and output_tile_idx == Int32(0)
                        and (Int32(tidx) & Int32(31)) == Int32(0)
                    ):
                        down_abs_max = cutlass.Float32(0.0)
                        down_slice = down_acc[(None, Int32(0), Int32(0))]
                        for sample_idx in cutlass.range_constexpr(cute.size(down_slice)):
                            down_abs_max = fmax_f32(
                                down_abs_max,
                                fabs_f32(fc2_down_alpha_value * down_slice[sample_idx]),
                            )
                        topk_weights_flat[Int32(3)] = down_abs_max

                    # Scatter using precomputed metadata (no redundant gmem loads)
                    tile_n_base_cur = output_tile_idx * Int32(self.tile_shape_mnk[1])
                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_n = mma_n_in_epi
                                mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                tRS_rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                                down_epi_acc_slice = down_acc[(None, mma_m, mma_n)]
                                for elem_idx in cutlass.range_constexpr(cute.size(tRS_rD_slice)):
                                    tRS_rD_slice[elem_idx] = fc2_down_alpha_value * down_epi_acc_slice[elem_idx]
                                if (
                                    self.skip_phase1
                                    and bidz == Int32(0)
                                    and current_work_linear_idx == Int32(0)
                                    and output_tile_idx == Int32(0)
                                    and epi_m == Int32(0)
                                    and mma_n_in_epi == Int32(0)
                                    and mma_m_in_epi == Int32(0)
                                    and lane_id == Int32(0)
                                ):
                                    cute.printf(
                                        "fc2_epi we={} d0={} d1={} r0={} r1={}",
                                        weight_expert_idx,
                                        down_epi_acc_slice[Int32(0)],
                                        down_epi_acc_slice[Int32(1)],
                                        tRS_rD_slice[Int32(0)],
                                        tRS_rD_slice[Int32(1)],
                                    )

                        acc_vec = tRS_rD.load()
                        acc_vec = acc_vec.to(cutlass.BFloat16)
                        tRS_rD_out.store(acc_vec)
                        epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        # No cross-warp barrier needed before scatter:
                        # StMatrix is warp-local, and each warp only reads
                        # its own 64×64 quadrant of sC below.

                        rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])

                        # Per-warp scatter: each warp scatters its own quadrant
                        # of sC (64 M-rows × 64 N-cols). No cross-warp read
                        # dependencies, so no pre-scatter barrier is needed.
                        warp_epi_rows = valid_rows - tile_m_base - rows_offset - warp_m_base
                        if warp_epi_rows > Int32(64):
                            warp_epi_rows = Int32(64)
                        if warp_epi_rows < Int32(0):
                            warp_epi_rows = Int32(0)

                        pair_idx = lane_id
                        while pair_idx < warp_epi_rows * Int32(32):
                            local_row = pair_idx >> Int32(5)  # / 32
                            local_pair_col = pair_idx & Int32(31)  # % 32
                            global_row = tile_m_base + rows_offset + warp_m_base + local_row
                            global_col = tile_n_base_cur + warp_n_base + local_pair_col * Int32(2)
                            cached_row = rows_offset + warp_m_base + local_row
                            # Only lane 0 loads tok/wv from gmem; broadcast via shuffle.
                            tok = Int32(0)
                            wv = cutlass.Float32(0.0)
                            if lane_id == Int32(0):
                                tok = _ld_shared_i32(scatter_tok_base_addr + cached_row * Int32(4))
                                wv = _ld_shared_f32(scatter_weight_base_addr + cached_row * Int32(4))
                            tok = cute.arch.shuffle_sync(tok, Int32(0))
                            wv = cute.arch.shuffle_sync(wv, Int32(0))
                            sc_v0 = cutlass.Float32(
                                sC[warp_m_base + local_row, warp_n_base + local_pair_col * Int32(2), epi_buffer]
                            )
                            sc_v1 = cutlass.Float32(
                                sC[warp_m_base + local_row, warp_n_base + local_pair_col * Int32(2) + Int32(1), epi_buffer]
                            )
                            if (
                                self.skip_phase1
                                and bidz == Int32(0)
                                and current_work_linear_idx == Int32(0)
                                and output_tile_idx == Int32(0)
                                and rows_offset == Int32(0)
                                and warp_m_base == Int32(0)
                                and warp_n_base == Int32(0)
                                and local_row == Int32(0)
                                and local_pair_col == Int32(0)
                                and lane_id == Int32(0)
                            ):
                                topk_weights_flat[Int32(4)] = fabs_f32(wv * sc_v0)
                                cute.printf(
                                    "fc2_scatter we={} alpha={} gs={} tok={} wv={} sc0={} sc1={}",
                                    weight_expert_idx,
                                    fc2_down_alpha_value,
                                    gs_value,
                                    tok,
                                    wv,
                                    sc_v0,
                                    sc_v1,
                                )
                            scatter_add_bf16x2(
                                get_ptr_as_int64(scatter_output, tok * scatter_N + global_col),
                                wv * sc_v0,
                                wv * sc_v1,
                            )
                            pair_idx += Int32(self.num_threads_per_warp)

                        # Post-scatter barrier: needed to ensure all warps
                        # finish scatter before next output tile's pipeline ops
                        # (pipeline consumer is collective across all MMA warps).
                        self.epilog_sync_barrier.arrive_and_wait()

                # Final pass_sync: protect sA from next task's FC1 loads.
                # DMA warp waits here too after finishing all B_down loads.
                self.pass_sync_barrier.arrive_and_wait()

                current_work_linear_idx += num_persistent_clusters
                tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = _compact_static_get_work_tile(
                    row_counts,
                    active_expert_count,
                    num_tiles_n=Int32(self.output_tile_count_n),
                    cluster_shape_mn=cluster_shape_mn,
                    current_work_linear_idx=current_work_linear_idx,
                    current_local_expert_idx=current_local_expert_idx,
                    accum_tile_m=accum_tile_m,
                    cta_id_in_cluster=cta_id_in_cluster,
                )

        # ===================================================================
        # DMA WARP (warp 4)
        # ===================================================================
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            lane_id = Int32(tidx) & Int32(31)

            num_persistent_clusters = Int32(gdim_z)
            cluster_shape_mn = (
                Int32(self.cluster_shape_mn[0]),
                Int32(self.cluster_shape_mn[1]),
            )
            cta_id_in_cluster = (
                Int32(bidx % cluster_shape_mn[0]),
                Int32(bidy % cluster_shape_mn[1]),
                Int32(0),
            )
            current_work_linear_idx = Int32(bidz)
            current_local_expert_idx = Int32(0)
            accum_tile_m = Int32(0)
            tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = _compact_static_get_work_tile(
                row_counts,
                active_expert_count,
                num_tiles_n=Int32(self.output_tile_count_n),
                cluster_shape_mn=cluster_shape_mn,
                current_work_linear_idx=current_work_linear_idx,
                current_local_expert_idx=current_local_expert_idx,
                accum_tile_m=accum_tile_m,
                cta_id_in_cluster=cta_id_in_cluster,
            )

            while is_valid_tile:
                tc = tile_coord
                intermediate_slice = tc[1]
                local_expert_idx = tc[2]
                weight_expert_idx = weight_expert_ids[local_expert_idx]

                tAgA_mk = tAgA[(None, tc[0], None, local_expert_idx)]
                tAgSFA_mk = tAgSFA[(None, tc[0], None, local_expert_idx)]

                # W13 is laid out as [up, gate] across the concatenated N dimension.
                tBgB_w13_up_nk = tBgB_w13[(None, intermediate_slice, None, weight_expert_idx)]
                tBgSFB_w13_up_nk = tBgSFB_w13[(None, intermediate_slice, None, weight_expert_idx)]
                tBgB_w13_gate_nk = tBgB_w13[(None, intermediate_slice + gate_tile_cnt, None, weight_expert_idx)]
                tBgSFB_w13_gate_nk = tBgSFB_w13[(None, intermediate_slice + gate_tile_cnt, None, weight_expert_idx)]
                gate_sfb_src_copy_elems = cute.size(tBgSFB_w13_gate_nk[(None, Int32(0))])
                up_sfb_src_copy_elems = cute.size(tBgSFB_w13_up_nk[(None, Int32(0))])
                if self.skip_phase1 and bidz == Int32(0) and lane_id == Int32(0) and current_work_linear_idx == Int32(0):
                    topk_ids_flat[Int32(7)] = cute.size(tBgB_w13_gate_nk)
                    topk_ids_flat[Int32(8)] = cute.size(tBgB_w13_up_nk)
                    topk_ids_flat[Int32(9)] = gate_sfb_src_copy_elems
                    topk_ids_flat[Int32(10)] = up_sfb_src_copy_elems
                    topk_ids_flat[Int32(11)] = cute.size(tBsB_w13)
                    topk_ids_flat[Int32(12)] = cute.size(tBsB_w13_up)

                # ---- FC1 gate pass ----
                prod_state.reset_count()
                if self.skip_phase1 and bidz == Int32(0) and current_work_linear_idx == Int32(0):
                    zero_idx = lane_id
                    while zero_idx < Int32(512):
                        st_shared_u8(b_base_addr + zero_idx, Uint8(0))
                        zero_idx += Int32(32)
                for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):
                    ml_pipeline.producer_acquire(prod_state)
                    cute.copy(tma_a, tAgA_mk[(None, k_tile)], tAsA[(None, prod_state.index)], tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state))
                    if self.skip_phase1 and bidz == Int32(0) and current_work_linear_idx == Int32(0):
                        cute.copy(tma_b_w13, tBgB_w13_up_nk[(None, k_tile)], tBsB_w13[(None, prod_state.index)], tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state))
                    else:
                        cute.copy(tma_b_w13, tBgB_w13_gate_nk[(None, k_tile)], tBsB_w13[(None, prod_state.index)], tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state))
                    cute.copy(tma_sfa, tAgSFA_mk[(None, k_tile)], tAsSFA[(None, prod_state.index)], tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state))
                    if self.skip_phase1 and bidz == Int32(0) and current_work_linear_idx == Int32(0):
                        cute.copy(tma_sfb_w13, tBgSFB_w13_up_nk[(None, k_tile)], tBsSFB_w13[(None, prod_state.index)], tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state))
                    else:
                        cute.copy(tma_sfb_w13, tBgSFB_w13_gate_nk[(None, k_tile)], tBsSFB_w13[(None, prod_state.index)], tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state))
                    ml_pipeline.producer_commit(prod_state)
                    prod_state.advance()

                # Gate and up share the A/SFA staging buffers. Wait for MMA
                # warps to drain the gate pass before refilling those stages.
                self.pass_sync_barrier.arrive_and_wait()

                # ---- FC1 up pass ----
                up_prod_state.reset_count()
                for k_tile in range(0, fc1_k_tile_cnt, 1, unroll=4):
                    up_pipeline.producer_acquire(up_prod_state)
                    cute.copy(tma_a, tAgA_mk[(None, k_tile)], tAsA[(None, up_prod_state.index)], tma_bar_ptr=up_pipeline.producer_get_barrier(up_prod_state))
                    cute.copy(tma_b_w13, tBgB_w13_up_nk[(None, k_tile)], tBsB_w13_up[(None, up_prod_state.index)], tma_bar_ptr=up_pipeline.producer_get_barrier(up_prod_state))
                    cute.copy(tma_sfa, tAgSFA_mk[(None, k_tile)], tAsSFA[(None, up_prod_state.index)], tma_bar_ptr=up_pipeline.producer_get_barrier(up_prod_state))
                    cute.copy(tma_sfb_w13, tBgSFB_w13_up_nk[(None, k_tile)], tBsSFB_w13_up[(None, up_prod_state.index)], tma_bar_ptr=up_pipeline.producer_get_barrier(up_prod_state))
                    up_pipeline.producer_commit(up_prod_state)
                    up_prod_state.advance()

                # ---- FC2 B_down loads: continuous pipeline ----
                # No barrier needed: sB/sSFB are free (gate done, up uses
                # sB_up/sSFB_up). phase2_pipeline handles data availability.
                # intermediate_slice selects the K-tile of GEMM2 (FC1 output N-tile
                # = GEMM2 K-tile since intermediate dim is the reduction dim).
                # Load ALL FC2 tiles continuously once stage1 no longer needs
                # the gate staging buffers.
                phase2_prod_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):
                    phase2_pipeline.producer_acquire(phase2_prod_state)
                    cute.copy(tma_b_down, tBgB_down[(None, output_tile_idx, intermediate_slice, weight_expert_idx)], tBsB_down[(None, phase2_prod_state.index)], tma_bar_ptr=phase2_pipeline.producer_get_barrier(phase2_prod_state))
                    cute.copy(tma_sfb_down, tBgSFB_down[(None, output_tile_idx, intermediate_slice, weight_expert_idx)], tBsSFB_down[(None, phase2_prod_state.index)], tma_bar_ptr=phase2_pipeline.producer_get_barrier(phase2_prod_state))
                    phase2_pipeline.producer_commit(phase2_prod_state)
                    phase2_prod_state.advance()

                # Final pass_sync: match MMA warps' barrier after FC2 sweep.
                # Ensures MMA warps finish scatter before DMA starts next task's FC1.
                self.pass_sync_barrier.arrive_and_wait()

                current_work_linear_idx += num_persistent_clusters
                tile_coord, is_valid_tile, current_local_expert_idx, accum_tile_m = _compact_static_get_work_tile(
                    row_counts,
                    active_expert_count,
                    num_tiles_n=Int32(self.output_tile_count_n),
                    cluster_shape_mn=cluster_shape_mn,
                    current_work_linear_idx=current_work_linear_idx,
                    current_local_expert_idx=current_local_expert_idx,
                    accum_tile_m=accum_tile_m,
                    cta_id_in_cluster=cta_id_in_cluster,
                )

            ml_pipeline.producer_tail(prod_state)
            up_pipeline.producer_tail(up_prod_state)
            phase2_pipeline.producer_tail(phase2_prod_state)
        return


__all__ = ["MoESuperfusedStaticKernel"]
