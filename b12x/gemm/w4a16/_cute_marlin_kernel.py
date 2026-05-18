"""CuTeDSL W4A16 NVFP4/BF16 W4A16 MoE kernels."""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Uint32

from b12x.cute.fp4 import (
    atomic_add_global_i32,
    bf16_mma_m16n8k16_f32,
    bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    bfloat2_broadcast_lane,
    bfloat2_mul,
    bfloat2_to_float2_scaled,
    broadcast_f32_to_half2,
    broadcast_f32_to_bfloat2,
    cp_async4_shared_global,
    cp_async4_shared_global_pred,
    f16_mma_m16n8k16_f32,
    f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    half2_to_float2_scaled,
    get_ptr_as_int64,
    half2_mul,
    ld_global_acquire_i32,
    ld_global_v4_f32,
    ld_shared_i32_relaxed,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_f32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x2_b16,
    packed_dequant_e2m1x4_to_bfloat2x2,
    packed_dequant_e2m1x4_to_half2x2,
    packed_dequant_e4m3x4_to_bfloat2x2,
    packed_dequant_e4m3x4_to_half2x2,
    pack_f32x2_to_bfloat2,
    pack_f32x2_to_f16x2,
    red_add_global_release_i32,
    shared_ptr_to_u32,
    st_global_v4_u32,
    st_global_i32,
    st_global_v4_f32,
    st_shared_bf16_from_f32,
    st_shared_f16_from_f32,
    st_shared_i32,
    st_shared_u32,
    st_shared_v4_f32,
    threadfence,
)
from b12x.cute.utils import current_cuda_stream
from b12x.moe.fused.w4a16.host import unswizzle_block_scale
from b12x.moe.fused.w4a16.prepare import (
    _nvfp4_compute_scale_factor,
    _permute_packed_scales,
    _process_nvfp4_packed_global_scale,
    _process_nvfp4_packed_scales,
    _repack_4bit_no_perm,
)
from b12x.runtime_control import raise_if_kernel_resolution_frozen


_SF_VEC_SIZE = 16

# CTA M-tile granularity.  Inherited from the MoE donor kernel.  8 picks a
# special small-M code path; others use the large-M path.
_ALLOWED_ROUTED_SIZES = (8, 16, 32, 48, 64)
_PACK_FACTOR = 8
# Pipeline depth of the cp.async + dequant + MMA K-pipeline.  Higher
# values hide more memory latency but increase smem footprint
# (sh_a_size, sh_b_size, sh_s_size all scale linearly with stages),
# which can drop blocks_per_sm.  Configurable per kernel instance via
# W4A16GemmKernel(num_stages=...) — default 4.  Lowering to 2 unlocks
# 2 blocks/SM on mamba_in_proj (N=10304, single-config-only shape)
# but loses pipeline parallelism elsewhere; see _select_num_stages.
_DEFAULT_NUM_STAGES = 4
_DEVICE_MAX_REG_BYTES = 255 * 1024
_DEFAULT_MAX_SHARED_MEM = 101_376

# The W4A16 launch model chooses blocks/SM from static resource usage
# for each specialization.  These register counts were measured from the local
# SM121 JIT output and keep launch occupancy stable across refactors.
_W4A16_REGS_SM121 = {
    (256, 1, 8, 8, True): 118,
    (128, 1, 4, 8, True): 118,
    (128, 1, 8, 4, True): 120,
    (256, 1, 8, 8, False): 158,
    (128, 1, 4, 8, False): 154,
    (128, 1, 8, 4, False): 143,
    (256, 2, 16, 4, False): 212,
    (128, 2, 4, 8, False): 215,
    (128, 2, 8, 4, False): 214,
    (256, 3, 16, 4, False): 249,
    (128, 3, 4, 8, False): 249,
    (128, 3, 8, 4, False): 250,
    (256, 4, 16, 4, False): 255,
    (128, 4, 4, 8, False): 255,
    (128, 4, 8, 4, False): 255,
}
# Tile config tables — (tile_k, tile_n, cta_threads).
#
# The kernel's register-pressure table (``_W4A16_REGS_SM121`` above) only
# enumerates four valid ``(cta_n_blocks, cta_k_blocks)`` pairs per
# cta_threads class:
#
#   cta_threads=128: (cta_n=4, cta_k=8) ↔ (tile_n=64,  tile_k=128)
#                    (cta_n=8, cta_k=4) ↔ (tile_n=128, tile_k=64)
#   cta_threads=256: (cta_n=8, cta_k=8) ↔ (tile_n=128, tile_k=128)
#                    (cta_n=16,cta_k=4) ↔ (tile_n=256, tile_k=64)
#
# A 2026-05-17 Spark autotune sweep over all (cta_m_size ∈ {16,32,48,64})
# × (tile_n, tile_k) combos for the 6 Nano3.5 v6-eligible shapes × M ∈
# {1024, 2048, 4096} confirmed ``_select_tile_config`` already picks the
# best config for every shape; the only outlier was shared.up where
# cta_m_size=32 won by ~1-2% but the same change hurt o_proj (-1.8%) and
# shared.dn (-1.4%), so no shape-agnostic improvement was applied.
#
# The Marlin gap on mamba_in_proj M=2048 (v6 = 1724 µs vs Marlin = 810 µs)
# is *structural*: N=10304 only divides by tile_n=64, which forces
# tile_k=128 as the single valid config.  ``num_stages`` (pipeline depth)
# was made configurable on 2026-05-18 — see W4A16GemmKernel.__init__.
# Lowering num_stages can free smem for 2 blocks/SM on this shape, but
# losing pipeline parallelism does not pay back the wave-parallelism gain
# (see _select_num_stages below).  Closing the residual gap further would
# need register-table entries for new (cta_n, cta_k) combinations or
# persistent split-K (Marlin-style).
_SMALL_BATCH_TILE_CONFIGS = (
    (128, 128, 256),
    (64, 128, 128),
    (128, 64, 128),
)
_LARGE_BATCH_TILE_CONFIGS = (
    (64, 256, 256),
    (64, 128, 128),
    (128, 64, 128),
)


def _covering_count(total: int, quantum: int) -> int:
    return (total + quantum - 1) // quantum


def _w4a16_num_regs(
    *,
    cta_threads: int,
    cta_m_blocks: int,
    cta_n_blocks: int,
    cta_k_blocks: int,
    uses_m_block_8: bool,
) -> int:
    key = (
        int(cta_threads),
        int(cta_m_blocks),
        int(cta_n_blocks),
        int(cta_k_blocks),
        bool(uses_m_block_8),
    )
    try:
        return _W4A16_REGS_SM121[key]
    except KeyError as exc:
        raise ValueError(
            f"missing W4A16 register count for NVFP4 BF16 specialization {key}"
        ) from exc


def _shared_memory_footprint(
    *,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> int:
    cta_m = int(cta_m_blocks) * 16
    cta_n = int(tile_n)
    cta_k = int(tile_k)
    sh_block_meta_size = cta_m * 16
    sh_a_size = int(num_stages) * (cta_m * cta_k) * 2
    sh_b_size = int(num_stages) * (cta_k * cta_n // _PACK_FACTOR) * 4
    sh_red_size = cta_m * (cta_n + 8) * 2
    sh_bias_size = cta_n * 2
    tmp_size = min(sh_b_size, sh_red_size) + sh_bias_size
    tmp_size = max(max(sh_b_size, sh_red_size), tmp_size)
    sh_s_size = _covering_count(cta_k, 16) * cta_n * 2 * int(num_stages)
    return tmp_size + sh_a_size + sh_s_size + sh_block_meta_size


def _determine_blocks_per_sm(
    *,
    problem_m: int,
    problem_n: int,
    top_k: int,
    cta_threads: int,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    uses_m_block_8: bool,
    sms: int,
    max_shared_mem: int,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> int:
    num_regs = _w4a16_num_regs(
        cta_threads=cta_threads,
        cta_m_blocks=cta_m_blocks,
        cta_n_blocks=tile_n // 16,
        cta_k_blocks=tile_k // 16,
        uses_m_block_8=uses_m_block_8,
    )
    register_bytes = max(num_regs, 1) * int(cta_threads) * 4
    smem_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=tile_n,
        tile_k=tile_k,
        num_stages=num_stages,
    )
    blocks_per_sm_limit = min(
        _DEVICE_MAX_REG_BYTES // register_bytes,
        int(max_shared_mem) // (smem_bytes + 1536),
    )
    if cta_m_blocks == 1:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 4), 1)
    else:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 2), 1)

    work_cta_count = (int(problem_n) // int(tile_n)) * int(problem_m) * int(top_k) * 4
    if work_cta_count < int(sms) * blocks_per_sm_limit:
        blocks_per_sm_limit = max(work_cta_count // int(sms), 1)
    return int(blocks_per_sm_limit)


def _candidate_tile_fits(
    *,
    problem_n: int,
    problem_k: int,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    cta_threads: int,
    max_shared_mem: int,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> bool:
    if int(tile_k) == -1 or int(tile_n) == -1 or int(cta_threads) == -1:
        return False
    if int(problem_k) % int(tile_k) != 0 or int(problem_n) % int(tile_n) != 0:
        return False
    if int(tile_n) < 64 or int(tile_k) < 64 or int(cta_threads) < 128:
        return False
    smem_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=tile_n,
        tile_k=tile_k,
        num_stages=num_stages,
    )
    return smem_bytes <= int(max_shared_mem)


def _select_tile_config(
    *,
    problem_m: int,
    problem_n: int,
    problem_k: int,
    top_k: int,
    moe_block_size: int,
    sms: int,
    max_shared_mem: int,
    required_cta_threads: int | None = None,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> tuple[int, int, int, int]:
    cta_m_blocks = _covering_count(moe_block_size, 16)
    uses_m_block_8 = moe_block_size == 8
    configs = (
        _LARGE_BATCH_TILE_CONFIGS if cta_m_blocks > 1 else _SMALL_BATCH_TILE_CONFIGS
    )
    best_blocks_per_sm = 0
    best_tile_config: tuple[int, int, int, int] | None = None
    for tile_k, tile_n, cta_threads in configs:
        if required_cta_threads is not None and int(cta_threads) != int(
            required_cta_threads
        ):
            continue
        if not _candidate_tile_fits(
            problem_n=problem_n,
            problem_k=problem_k,
            cta_m_blocks=cta_m_blocks,
            tile_n=tile_n,
            tile_k=tile_k,
            cta_threads=cta_threads,
            max_shared_mem=int(max_shared_mem) - 512,
            num_stages=num_stages,
        ):
            continue
        blocks_per_sm_limit = _determine_blocks_per_sm(
            problem_m=problem_m,
            problem_n=problem_n,
            top_k=top_k,
            cta_threads=cta_threads,
            cta_m_blocks=cta_m_blocks,
            tile_n=tile_n,
            tile_k=tile_k,
            uses_m_block_8=uses_m_block_8,
            sms=sms,
            max_shared_mem=max_shared_mem,
            num_stages=num_stages,
        )
        if blocks_per_sm_limit > best_blocks_per_sm:
            best_blocks_per_sm = blocks_per_sm_limit
            best_tile_config = (tile_k, tile_n, cta_threads, blocks_per_sm_limit)
    if best_tile_config is None:
        cta_thread_msg = (
            ""
            if required_cta_threads is None
            else f", required_cta_threads={required_cta_threads}"
        )
        raise ValueError(
            "no valid W4A16 tile config for "
            f"M/N/K={problem_m}/{problem_n}/{problem_k}, moe_block_size={moe_block_size}"
            f"{cta_thread_msg}"
        )
    return best_tile_config


# Tile-keyed ``num_stages`` overrides (Spark autotune 2026-05-18).
#
# The kernel's MMA pipeline depth (``num_stages``) trades smem footprint
# against latency hiding.  Spark sweeps showed the winning depth is
# determined by the *tile shape*, not the problem shape:
#
#   tile_n=128, tile_k=64   (cta_threads=128): stages=3 — frees smem for
#       2 blocks/SM, wins +7-12% on o_proj / shared.up / shared.dn /
#       mamba_output_proj across M ∈ {512..4096}.
#   tile_n=64,  tile_k=128  (cta_threads=128): stages=2 — frees enough
#       smem for 2 blocks/SM on the wide-K narrow-N case (mamba_in_proj
#       N=10304), wins +2-4%.
#   tile_n=256, tile_k=64   (cta_threads=256): stages=4 — already at
#       blocks_per_sm=1 ceiling (tile too big for 2/SM at any depth);
#       deeper pipeline = better latency hiding.  +0.3% over stages=3.
#   tile_n=128, tile_k=128  (cta_threads=256): stages=4 (default;
#       smem too tight to drop stages without underutilization).
#
# Any tile not in the table uses _DEFAULT_NUM_STAGES.
_NUM_STAGES_BY_TILE: dict[tuple[int, int], int] = {
    # (tile_n, tile_k): num_stages
    (128, 64): 3,
    (64, 128): 2,
    (256, 64): 4,
}


def _select_num_stages_for_tile(*, tile_n: int, tile_k: int) -> int:
    return _NUM_STAGES_BY_TILE.get((int(tile_n), int(tile_k)), _DEFAULT_NUM_STAGES)


@dataclass(frozen=True)
class W4A16GemmCompileResult:
    compiled: object
    tile_n: int
    tile_k: int
    moe_block_size: int
    max_m_blocks: int
    blocks_per_sm: int
    num_stages: int = _DEFAULT_NUM_STAGES


@dataclass(frozen=True)
class _W4A16GemmLaunch:
    kernel: W4A16GemmCompileResult
    c_tmp: torch.Tensor


class W4A16GemmKernel:
    """Dense W4A16 GEMM kernel — Marlin-style cp.async + register pipeline.

    Ported from the MoE W4A16 kernel (b12x/moe/fused/w4a16/kernel.py) by
    stripping the routing/expert/topk axes.  Each CTA computes one
    (cta_m_size x tile_n) output tile.  Internal field names keep the
    MoE donor's terminology (e.g. ``cta_m_blocks``, ``moe_block_size``)
    for line-by-line traceability against the donor; semantically these
    are now just dense CTA-tile dimensions.
    """

    def __init__(
        self,
        *,
        size_m: int,
        size_n: int,
        size_k: int,
        tile_n: int,
        tile_k: int,
        cta_m_size: int,
        element_dtype: str = "bf16",
        num_stages: int = _DEFAULT_NUM_STAGES,
        size_n_real: int | None = None,
    ):
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if size_n % tile_n != 0:
            raise ValueError("size_n must be divisible by tile_n")
        if size_k % tile_k != 0:
            raise ValueError("size_k must be divisible by tile_k")
        if tile_n % 16 != 0 or tile_k % 16 != 0:
            raise ValueError("tile_n/tile_k must be multiples of 16")
        if cta_m_size not in _ALLOWED_ROUTED_SIZES:
            raise ValueError(f"unsupported cta_m_size {cta_m_size}")
        if cta_m_size != 8 and cta_m_size % 16 != 0:
            raise ValueError("cta_m_size must be 8 or a multiple of 16")
        if int(num_stages) < 2:
            raise ValueError("num_stages must be at least 2")
        cta_threads = tile_n * tile_k // 64
        if cta_threads not in (128, 256):
            raise ValueError("W4A16 GEMM expects 128 or 256 CTA threads")
        # size_n is the *padded* N used for tile geometry, weight/scale
        # layout, and c_gl_wr arithmetic (must satisfy size_n % tile_n == 0).
        # size_n_real is the *actual* N to write to the C buffer; defaults
        # to size_n when no padding is in play.  Letting size_n_real <
        # size_n lets callers pad shapes (e.g. N=10304 → 10368) to unlock
        # wider tile geometries without copying the output afterwards.
        if size_n_real is None:
            size_n_real = size_n
        if int(size_n_real) > int(size_n):
            raise ValueError("size_n_real must be <= size_n (padded)")
        if int(size_n_real) % 8 != 0:
            raise ValueError(
                "size_n_real must be a multiple of 8 (per-thread store vector width)"
            )
        self.size_m = int(size_m)
        self.size_n = int(size_n)
        self.size_n_real = int(size_n_real)
        self.size_k = int(size_k)
        self.tile_n = int(tile_n)
        self.tile_k = int(tile_k)
        self.cta_n_blocks = int(tile_n // 16)
        self.cta_k_blocks = int(tile_k // 16)
        self.cta_threads = int(cta_threads)
        self.num_stages = int(num_stages)
        # Keep ``moe_block_size`` field name for parity with the donor's
        # internal methods (which heavily reference self.moe_block_size).
        # For dense it's just the per-CTA M-tile size.
        self.moe_block_size = int(cta_m_size)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        # Dense path has no in-kernel activation epilogue.
        self.epilogue_relu2 = False
        # Routing-axis constants pinned for the dense port.  The MoE donor
        # branches on these via ``cutlass.const_expr`` so when they're
        # hard-coded the dead branches compile away.
        self.top_k = 1
        self.mul_topk_weights = False
        self.cta_m_blocks = int(_covering_count(cta_m_size, 16))
        self.uses_m_block_8 = cta_m_size == 8
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.sms = int(props.multi_processor_count)
            max_shared_mem = int(
                getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
            )
        else:
            self.sms = 120
            max_shared_mem = _DEFAULT_MAX_SHARED_MEM
        self.blocks_per_sm = _determine_blocks_per_sm(
            problem_m=self.size_m,
            problem_n=self.size_n,
            top_k=1,
            cta_threads=self.cta_threads,
            cta_m_blocks=self.cta_m_blocks,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            uses_m_block_8=self.uses_m_block_8,
            sms=self.sms,
            max_shared_mem=max_shared_mem,
            num_stages=self.num_stages,
        )

        # W4A16 shared-memory geometry, in int4 units unless noted.
        # (Identical to the MoE donor's smem layout — the MMA pipeline +
        # cp.async staging doesn't change between MoE and dense.)
        self.a_sh_stride = 16 * self.cta_k_blocks // 8
        self.a_sh_stage = self.a_sh_stride * (16 * self.cta_m_blocks)
        self.a_gl_rd_delta_o = 16 * self.cta_k_blocks // 8
        self.a_sh_wr_delta = self.a_sh_stride * (
            self.cta_threads // self.a_gl_rd_delta_o
        )
        self.a_sh_wr_iters = _covering_count(self.a_sh_stage, self.a_sh_wr_delta)
        self.a_sh_rd_delta_i = self.a_sh_stride * 16

        self.b_sh_stride = ((self.cta_n_blocks * 16) * 16 // _PACK_FACTOR) // 4
        self.b_thread_vecs = 1
        self.b_sh_stride_threads = self.b_sh_stride
        self.b_sh_stage = self.b_sh_stride * self.cta_k_blocks
        self.b_sh_wr_iters = self.b_sh_stage // self.cta_threads

        self.s_sh_stride = 16 * self.cta_n_blocks // 16
        self.s_tb_groups = self.cta_k_blocks
        self.s_sh_stage = self.s_tb_groups * self.s_sh_stride
        self.tb_n_warps = self.cta_n_blocks // 4

        # Dense kernel has no routing tables -- the donor's
        # sh_block_route_indices / sh_rd_block_route_indices /
        # sh_block_topk_weights / sh_valid_count_off smem region is gone.
        # Set the corresponding offsets to 0 so downstream arithmetic
        # is structurally identical but reads from "no-op" smem slots
        # (or, where the field is used as a routing-table base, we
        # intercept the access at the point of use).
        self.sh_valid_count_off = 0
        self.sh_route_off = 0
        self.sh_rd_route_off = 0
        self.sh_topk_off = 0

        sh_red_size = (2 * self.cta_n_blocks + 1) * 16 * self.cta_m_blocks
        sh_b_size = self.num_stages * self.b_sh_stage
        sh_size_min = min(sh_red_size, sh_b_size)
        sh_size_max = max(sh_red_size, sh_b_size)
        sh_bias_size = self.cta_n_blocks * 16 // 8
        sh_b_red_bias_size = max(sh_size_max, sh_size_min + sh_bias_size)
        self.sh_b_off = self.sh_valid_count_off
        self.sh_red_off = self.sh_valid_count_off
        self.sh_s_off = self.sh_valid_count_off + sh_b_red_bias_size
        self.sh_a_off = self.sh_s_off + self.num_stages * self.s_sh_stage
        self.shared_int4 = self.sh_a_off + self.num_stages * self.a_sh_stage
        self.shared_words = self.shared_int4 * 4

    @cute.jit
    def _activation_smem_permuted_offset(self, i: Int32) -> Int32:
        row = i // Int32(self.a_gl_rd_delta_o)
        return Int32(self.a_gl_rd_delta_o) * row + (
            (i - row * Int32(self.a_gl_rd_delta_o)) ^ (row & Int32(7))
        )

    @cute.jit
    def _int4_addr(self, smem_base: Int32, int4_off: Int32) -> Int32:
        return smem_base + int4_off * Int32(16)

    @cute.jit
    def _dequant_e2m1x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e2m1x4_to_half2x2(packed)
        return packed_dequant_e2m1x4_to_bfloat2x2(packed)

    @cute.jit
    def _dequant_e4m3x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e4m3x4_to_half2x2(packed)
        return packed_dequant_e4m3x4_to_bfloat2x2(packed)

    @cute.jit
    def _elem2_mul(self, a: Uint32, b: Uint32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return half2_mul(a, b)
        return bfloat2_mul(a, b)

    @cute.jit
    def _broadcast_f32_to_elem2(self, x: cutlass.Float32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return broadcast_f32_to_half2(x)
        return broadcast_f32_to_bfloat2(x)

    @cute.jit
    def _pack_f32x2_to_elem2(self, x0: cutlass.Float32, x1: cutlass.Float32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return pack_f32x2_to_f16x2(x0, x1)
        return pack_f32x2_to_bfloat2(x0, x1)

    @cute.jit
    def _elem2_to_f32x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return half2_to_float2_scaled(packed, cutlass.Float32(1.0))
        return bfloat2_to_float2_scaled(packed, cutlass.Float32(1.0))

    @cute.jit
    def _relu2_elem2(self, packed: Uint32) -> Uint32:
        x0, x1 = self._elem2_to_f32x2(packed)
        if x0 < cutlass.Float32(0.0):
            x0 = cutlass.Float32(0.0)
        if x1 < cutlass.Float32(0.0):
            x1 = cutlass.Float32(0.0)
        return self._pack_f32x2_to_elem2(x0 * x0, x1 * x1)

    @cute.jit
    def _st_shared_elem_from_f32(self, addr: Int32, val: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            st_shared_f16_from_f32(addr, val)
        else:
            st_shared_bf16_from_f32(addr, val)

    @cute.jit
    def _mma_m16n8k16_f32(
        self,
        d0: cutlass.Float32,
        d1: cutlass.Float32,
        d2: cutlass.Float32,
        d3: cutlass.Float32,
        a0: Uint32,
        a1: Uint32,
        a2: Uint32,
        a3: Uint32,
        b0: Uint32,
        b1: Uint32,
    ):
        if cutlass.const_expr(self.is_fp16):
            return f16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)
        return bf16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)

    @cute.jit
    def _mma_rhs_fragments_as_mma_a_m16n8k16_f32(
        self,
        d0: cutlass.Float32,
        d1: cutlass.Float32,
        d2: cutlass.Float32,
        d3: cutlass.Float32,
        b0_0: Uint32,
        b1_0: Uint32,
        b0_1: Uint32,
        b1_1: Uint32,
        a0: Uint32,
        a1: Uint32,
    ):
        if cutlass.const_expr(self.is_fp16):
            return f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                d0, d1, d2, d3, b0_0, b1_0, b0_1, b1_1, a0, a1
            )
        return bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            d0, d1, d2, d3, b0_0, b1_0, b0_1, b1_1, a0, a1
        )

    @cute.jit
    def __call__(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = self.sms * self.blocks_per_sm
        grid = (grid_x, 1, 1)
        self.kernel(
            a_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            c_tmp_f32_flat,
            locks_i32_flat,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        grid_x, _, _ = cute.arch.grid_dim()
        self._run_persistent_gemm(
            a_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            cta,
            Int32(grid_x),
        )

    @cute.jit
    def _run_persistent_gemm(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
    ):
        n_tiles = Int32(self.size_n // self.tile_n)
        # Number of M-tiles this CTA-grid must cover.  For dense this is
        # ceil(M / cta_m_size); donor used a route-count tensor here.
        route_blocks = Int32(
            (self.size_m + self.moe_block_size - 1) // self.moe_block_size
        )
        k_tiles = Int32(self.size_k // self.tile_k)
        global_mn_tiles = route_blocks * n_tiles

        tail_mn_tiles = global_mn_tiles
        full_grid_mn_iters = Int32(0)
        if global_mn_tiles > grid_x:
            tail_mn_tiles = global_mn_tiles - (global_mn_tiles // grid_x) * grid_x
            if tail_mn_tiles * Int32(3) <= grid_x:
                tail_mn_tiles += grid_x
            full_grid_mn_iters = (global_mn_tiles - tail_mn_tiles) // grid_x

        iters = (k_tiles * tail_mn_tiles + grid_x - Int32(1)) // grid_x

        lock_slot = Int32(0)
        if tail_mn_tiles >= grid_x:
            lock_slot = cta
        else:
            lock_slot = (iters * cta) // k_tiles - Int32(1)

        in_tail_region = Int32(0)
        has_work = Int32(1)
        work_mn_tile = cta
        reduce_k_tile = Int32(0)
        route_block_idx = Int32(0)
        output_n_tile = Int32(0)
        if iters == Int32(0) and full_grid_mn_iters == Int32(0):
            has_work = Int32(0)

        while has_work != Int32(0):
            reduce_tile_count = Int32(0)
            reduce_slice_count = Int32(1)
            reduce_slice_idx = Int32(0)

            if in_tail_region == Int32(0) and full_grid_mn_iters > Int32(0):
                route_block_idx = work_mn_tile // n_tiles
                output_n_tile = work_mn_tile - route_block_idx * n_tiles
                reduce_k_tile = Int32(0)
                reduce_tile_count = k_tiles
                full_grid_mn_iters -= Int32(1)
            else:
                if in_tail_region == Int32(0):
                    in_tail_region = Int32(1)
                    tail_mn_base = global_mn_tiles - tail_mn_tiles
                    cta_iter_start = iters * cta
                    work_mn_tile = cta_iter_start // k_tiles
                    reduce_k_tile = cta_iter_start - work_mn_tile * k_tiles
                    global_mn_tile = work_mn_tile + tail_mn_base
                    route_block_idx = global_mn_tile // n_tiles
                    output_n_tile = global_mn_tile - route_block_idx * n_tiles

                if work_mn_tile < tail_mn_tiles and iters > Int32(0):
                    reduce_tile_count = iters * (cta + Int32(1)) - (
                        k_tiles * work_mn_tile + reduce_k_tile
                    )
                    if reduce_tile_count < Int32(0):
                        reduce_tile_count = Int32(0)
                    if reduce_k_tile + reduce_tile_count > k_tiles:
                        reduce_tile_count = k_tiles - reduce_k_tile

                    if reduce_tile_count > Int32(0):
                        first_reduce_boundary = iters * (
                            (k_tiles * work_mn_tile + iters - Int32(1)) // iters
                        )
                        if first_reduce_boundary <= k_tiles * (work_mn_tile + Int32(1)):
                            reduce_boundary_offset = (
                                first_reduce_boundary - k_tiles * work_mn_tile
                            )
                            reduce_slice_count = (
                                k_tiles - reduce_boundary_offset + iters - Int32(1)
                            ) // iters
                            if reduce_boundary_offset > Int32(0):
                                reduce_slice_count += Int32(1)
                            reduce_boundary_delta = iters * cta - first_reduce_boundary
                            if reduce_boundary_delta < Int32(0):
                                reduce_slice_idx = reduce_slice_count - Int32(1)
                            else:
                                if reduce_boundary_offset == Int32(
                                    0
                                ) and reduce_boundary_delta == Int32(0):
                                    reduce_slice_idx = reduce_slice_count - Int32(1)
                                else:
                                    reduce_slice_idx = (
                                        reduce_slice_count
                                        - Int32(1)
                                        - reduce_boundary_delta // iters
                                    )
                                    if reduce_boundary_offset > Int32(0):
                                        reduce_slice_idx -= Int32(1)

                        if tail_mn_tiles >= grid_x:
                            if reduce_slice_count > Int32(
                                1
                            ) and reduce_slice_idx == reduce_slice_count - Int32(1):
                                lock_slot += Int32(1)
                        else:
                            lock_slot += Int32(1)
                    else:
                        has_work = Int32(0)
                else:
                    has_work = Int32(0)

            if (
                has_work != Int32(0)
                and reduce_tile_count > Int32(0)
                and route_block_idx < route_blocks
            ):
                # Dense: expert_idx is always 0, no need for the donor's
                # block_expert_ids[route_block_idx] lookup or the
                # ``expert_idx >= 0`` guard (which was the MoE drop-the-
                # tile-if-no-expert-assigned path).
                self._run_tile(
                    a_bf16_flat,
                    b_i32_flat,
                    c_bf16_flat,
                    scales_i32_flat,
                    global_scale,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    smem_base,
                    tid,
                    route_block_idx,
                    output_n_tile,
                    reduce_k_tile,
                    reduce_tile_count,
                    reduce_slice_count,
                    reduce_slice_idx,
                    lock_slot,
                )

            if has_work != Int32(0):
                if in_tail_region == Int32(0):
                    work_mn_tile += grid_x
                else:
                    reduce_k_tile = Int32(0)
                    work_mn_tile += Int32(1)
                    output_n_tile += Int32(1)
                    if output_n_tile == n_tiles:
                        output_n_tile = Int32(0)
                        route_block_idx += Int32(1)
    @cute.jit
    def _run_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        m_tile_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        if cutlass.const_expr(self.uses_m_block_8):
            self._run_tile_m8(
                a_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                m_tile_idx,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                lock_slot,
            )
        else:
            self._run_tile_large_m(
                a_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                m_tile_idx,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                lock_slot,
            )

    @cute.jit
    def _tile_common_prologue(
        self,
        global_scale: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        m_tile_idx: Int32,
        output_n_tile: Int32,
    ):
        """Dense replacement for the donor's MoE prologue.

        Computes the global scale + how many valid M-rows this CTA owns,
        and the staging stream offsets.  No route-table read, no topk
        weight prefetch, no expert lookup -- expert_idx is implicit zero.
        """
        global_scale_f32 = global_scale[Int32(0)].to(cutlass.Float32)
        # Valid M-rows in this CTA's tile (clipped at the M boundary).
        m_tile_base = m_tile_idx * Int32(self.moe_block_size)
        block_valid_rows = Int32(self.size_m) - m_tile_base
        if block_valid_rows > Int32(self.moe_block_size):
            block_valid_rows = Int32(self.moe_block_size)
        if block_valid_rows < Int32(0):
            block_valid_rows = Int32(0)
        (
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_stream_offsets(tid, Int32(0), output_n_tile)
        return (
            global_scale_f32,
            block_valid_rows,
            m_tile_base,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        )

    @cute.jit
    def _tile_stream_offsets(self, tid: Int32, expert_idx: Int32, output_n_tile: Int32):
        a_gl_stride = Int32(self.size_k // 8)
        b_gl_stride = Int32(16 * self.size_n // (_PACK_FACTOR * 4))
        s_gl_stride = Int32(self.size_n // 16)
        scales_expert_stride = Int32((self.size_n * self.size_k) // (16 * 16))
        b_expert_off = (
            Int32((self.size_n * self.size_k) // (_PACK_FACTOR * 4)) * expert_idx
        )
        scales_expert_off = scales_expert_stride * expert_idx

        a_gl_rd_row = tid // Int32(self.a_gl_rd_delta_o)
        a_gl_rd_col0 = tid - a_gl_rd_row * Int32(self.a_gl_rd_delta_o)
        a_sh_wr = Int32(self.a_sh_stride) * (tid // Int32(self.a_gl_rd_delta_o)) + (
            tid - (tid // Int32(self.a_gl_rd_delta_o)) * Int32(self.a_gl_rd_delta_o)
        )
        a_rows_per_iter = Int32(self.cta_threads // self.a_gl_rd_delta_o)

        if cutlass.const_expr(self.cta_threads <= self.b_sh_stride):
            b_gl_rd_base = tid
        else:
            b_gl_rd_base = b_gl_stride * (tid // Int32(self.b_sh_stride)) + (
                tid % Int32(self.b_sh_stride)
            )
        b_gl_rd_base += b_expert_off + Int32(self.b_sh_stride) * output_n_tile
        b_sh_rd = tid
        b_sh_rd += (b_sh_rd // Int32(self.b_sh_stride)) * Int32(
            self.b_sh_stride * (self.b_sh_wr_iters - 1)
        )

        s_sh_rd = Int32(8) * ((tid // Int32(32)) % Int32(self.tb_n_warps)) + (
            tid & Int32(31)
        ) // Int32(4)
        return (
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        )

    @cute.jit
    def _a_shared_read_offset(self, tid: Int32, lanes_per_row: cutlass.Constexpr[int]):
        a_sh_rd = Int32(self.a_sh_stride) * (
            (tid & Int32(31)) % Int32(lanes_per_row)
        ) + (tid & Int32(31)) // Int32(lanes_per_row)
        a_sh_rd += (
            Int32(2)
            * ((tid // Int32(32)) // Int32(self.tb_n_warps))
            * Int32(self.b_sh_wr_iters)
        )
        return a_sh_rd

    @cute.jit
    def _run_tile_m8(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        m_tile_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        (
            global_scale_f32,
            block_valid_rows,
            m_tile_base,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            smem_base,
            tid,
            m_tile_idx,
            output_n_tile,
        )
        a_sh_rd = self._a_shared_read_offset(tid, 8)

        acc = cute.make_rmem_tensor((4, 4), cutlass.Float32)
        acc.fill(0.0)

        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            m_tile_base,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
        )

        b_scale_cur = cute.make_rmem_tensor((2, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((2, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
        )
        a_regs_cur = cute.make_rmem_tensor((2,), Uint32)
        a_regs_next = cute.make_rmem_tensor((2,), Uint32)
        self._load_a_register_bundle(
            a_regs_cur,
            smem_base,
            a_sh_rd,
            Int32(0),
            Int32(0),
            True,
        )
        self._run_mma_pipeline(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            acc,
            b_scale_cur,
            b_scale_next,
            a_regs_cur,
            a_regs_next,
            b_sh_rd,
            s_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            m_tile_base,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            True,
        )

        self._finish_tile(
            acc,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
            m_tile_base,
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            True,
        )

    @cute.jit
    def _run_tile_large_m(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        m_tile_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        (
            global_scale_f32,
            block_valid_rows,
            m_tile_base,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            smem_base,
            tid,
            m_tile_idx,
            output_n_tile,
        )
        a_sh_rd = self._a_shared_read_offset(tid, 16)

        acc = cute.make_rmem_tensor((self.cta_m_blocks, 4, 2, 4), cutlass.Float32)
        acc.fill(0.0)

        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            m_tile_base,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
        )

        b_scale_cur = cute.make_rmem_tensor((2, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((2, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
        )
        a_regs = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        a_regs_next = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        self._load_a_register_bundle(
            a_regs,
            smem_base,
            a_sh_rd,
            Int32(0),
            Int32(0),
            False,
        )
        self._run_mma_pipeline(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            acc,
            b_scale_cur,
            b_scale_next,
            a_regs,
            a_regs_next,
            b_sh_rd,
            s_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            m_tile_base,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            False,
        )

        self._finish_tile(
            acc,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
            m_tile_base,
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            False,
        )

    @cute.jit
    def _run_mma_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        acc: cute.Tensor,
        b_scale_cur: cute.Tensor,
        b_scale_next: cute.Tensor,
        a_regs_cur: cute.Tensor,
        a_regs_next: cute.Tensor,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a_sh_rd: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        m_tile_base: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        b_frag = cute.make_rmem_tensor((2, 2), Uint32)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range_constexpr(self.num_stages):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        self._load_next_fragment_bundle(
                            b_scale_next,
                            a_regs_next,
                            smem_base,
                            tid,
                            b_sh_rd,
                            s_sh_rd,
                            a_sh_rd,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            uses_m_block_8,
                        )

                        self._prefetch_pipeline_step(
                            a_bf16_flat,
                            b_i32_flat,
                            scales_i32_flat,
                            smem_base,
                            tid,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            reduce_k_tile,
                            block_valid_rows,
                            m_tile_base,
                            a_gl_stride,
                            b_gl_stride,
                            s_gl_stride,
                            scales_expert_off,
                            b_gl_rd_base,
                            a_gl_rd_row,
                            a_gl_rd_col0,
                            a_sh_wr,
                            a_rows_per_iter,
                            output_n_tile,
                        )

                        for jj in cutlass.range_constexpr(4):
                            q, s = self._select_b_scale_register(jj, b_scale_cur)
                            self._scaled_dequant_b_fragment(b_frag, q, s)
                            if cutlass.const_expr(uses_m_block_8):
                                self._mma_accumulate_m8(
                                    acc,
                                    jj,
                                    a_regs_cur,
                                    b_frag,
                                )
                            else:
                                for mb in cutlass.range_constexpr(self.cta_m_blocks):
                                    self._mma_accumulate_large_m(
                                        acc,
                                        a_regs_cur,
                                        mb,
                                        jj,
                                        b_frag,
                                    )

                        if cutlass.const_expr(uses_m_block_8):
                            self._copy_a_register_bundle(
                                a_regs_cur,
                                a_regs_next,
                                uses_m_block_8,
                            )
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next
                            )
                        else:
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next
                            )
                            self._copy_a_register_bundle(
                                a_regs_cur,
                                a_regs_next,
                                uses_m_block_8,
                            )
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_cur,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(0),
                    Int32(0),
                )
                self._load_a_register_bundle(
                    a_regs_cur,
                    smem_base,
                    a_sh_rd,
                    Int32(0),
                    Int32(0),
                    uses_m_block_8,
                )

    @cute.jit
    def _finish_tile(
        self,
        acc: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        m_tile_base: Int32,
        global_scale_f32: cutlass.Float32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._fold_cta_partials_m8(acc, smem_base, tid)
        else:
            self._fold_cta_partials_large_m(acc, smem_base, tid)

        if reduce_slice_count > Int32(1):
            self._wait_for_reduction_turn(
                locks_i32_flat, lock_slot, reduce_slice_idx, tid
            )
            self._combine_splitk_accumulators(
                acc,
                c_tmp_f32_flat,
                block_valid_rows,
                lock_slot,
                reduce_slice_idx,
                reduce_slice_count,
                tid,
                uses_m_block_8,
            )
            self._publish_reduction_turn(
                locks_i32_flat,
                lock_slot,
                reduce_slice_idx == reduce_slice_count - Int32(1),
                tid,
            )

        if reduce_slice_idx == reduce_slice_count - Int32(1):
            if cutlass.const_expr(uses_m_block_8):
                self._store_tile_m8(
                    acc,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    m_tile_base,
                    global_scale_f32,
                )
            else:
                self._store_tile_large_m(
                    acc,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    m_tile_base,
                    global_scale_f32,
                )

    @cute.jit
    def _wait_for_reduction_turn(
        self,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        count: Int32,
        tid: Int32,
    ):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        if tid == Int32(0):
            state = Int32(-1)
            while state != count:
                state = ld_global_acquire_i32(lock_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _publish_reduction_turn(
        self,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        reset,
        tid: Int32,
    ):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        cute.arch.sync_threads()
        if tid == Int32(0):
            if reset:
                st_global_i32(lock_addr, Int32(0))
            else:
                red_add_global_release_i32(lock_addr, Int32(1))

    @cute.jit
    def _merge_splitk_vec4(
        self,
        c_tmp_f32_flat: cute.Tensor,
        f32_off: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        c2: cutlass.Float32,
        c3: cutlass.Float32,
    ):
        if reduce_slice_idx != Int32(0):
            r0, r1, r2, r3 = ld_global_v4_f32(get_ptr_as_int64(c_tmp_f32_flat, f32_off))
            c0 = c0 + r0
            c1 = c1 + r1
            c2 = c2 + r2
            c3 = c3 + r3
        if reduce_slice_idx != reduce_slice_count - Int32(1):
            st_global_v4_f32(
                get_ptr_as_int64(c_tmp_f32_flat, f32_off),
                c0,
                c1,
                c2,
                c3,
            )
        return c0, c1, c2, c3

    @cute.jit
    def _combine_splitk_accumulators(
        self,
        acc: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        block_valid_rows: Int32,
        lock_slot: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        tid: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        active_threads = Int32(32 * self.tb_n_warps)
        c_size_int4 = Int32((self.cta_m_blocks * 16 * self.cta_n_blocks * 16) // 4)
        c_cur_offset = lock_slot * c_size_int4
        if cutlass.const_expr(uses_m_block_8):
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    k = jj * 2
                    acc[jj, 0], acc[jj, 1], acc[jj, 2], acc[jj, 3] = (
                        self._merge_splitk_slot(
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            Int32(k),
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                            acc[jj, 0],
                            acc[jj, 1],
                            acc[jj, 2],
                            acc[jj, 3],
                        )
                    )
        else:
            lane_row = (tid & Int32(31)) // Int32(4)
            if tid < active_threads:
                for k in cutlass.range_constexpr(self.cta_m_blocks * 8):
                    mb = k // 8
                    flat_j = k % 8
                    jj = flat_j // 2
                    half = flat_j % 2
                    row_valid = Int32(mb * 16) + lane_row < block_valid_rows
                    if row_valid:
                        (
                            acc[mb, jj, half, 0],
                            acc[mb, jj, half, 1],
                            acc[mb, jj, half, 2],
                            acc[mb, jj, half, 3],
                        ) = self._merge_splitk_slot(
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            Int32(k),
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                            acc[mb, jj, half, 0],
                            acc[mb, jj, half, 1],
                            acc[mb, jj, half, 2],
                            acc[mb, jj, half, 3],
                        )

    @cute.jit
    def _merge_splitk_slot(
        self,
        c_tmp_f32_flat: cute.Tensor,
        c_cur_offset: Int32,
        active_threads: Int32,
        slot: Int32,
        tid: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        c2: cutlass.Float32,
        c3: cutlass.Float32,
    ):
        int4_off = c_cur_offset + active_threads * slot + tid
        return self._merge_splitk_vec4(
            c_tmp_f32_flat,
            int4_off * Int32(4),
            reduce_slice_idx,
            reduce_slice_count,
            c0,
            c1,
            c2,
            c3,
        )

    @cute.jit
    def _load_a_registers_large_m(
        self,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        m_block: Int32,
    ):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off)
            + pipe * Int32(self.a_sh_stage)
            + self._activation_smem_permuted_offset(
                Int32(2) * kk + m_block * Int32(self.a_sh_rd_delta_i) + a_sh_rd
            ),
        )
        return ldmatrix_m8n8x4_b16(a_addr)

    @cute.jit
    def _load_a_registers_m8(
        self,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off)
            + pipe * Int32(self.a_sh_stage)
            + self._activation_smem_permuted_offset(Int32(2) * kk + a_sh_rd),
        )
        return ldmatrix_m8n8x2_b16(a_addr)

    @cute.jit
    def _load_a_registers_large_m_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            a0, a1, a2, a3 = self._load_a_registers_large_m(
                smem_base,
                a_sh_rd,
                pipe,
                kk,
                Int32(mb),
            )
            regs[mb, 0] = a0
            regs[mb, 1] = a1
            regs[mb, 2] = a2
            regs[mb, 3] = a3

    @cute.jit
    def _load_a_registers_m8_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        a0, a1 = self._load_a_registers_m8(smem_base, a_sh_rd, pipe, kk)
        regs[0] = a0
        regs[1] = a1

    @cute.jit
    def _clear_a_register_bundle_large_m(self, regs: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                regs[mb, reg] = Uint32(0)

    @cute.jit
    def _clear_a_register_bundle_m8(self, regs: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            regs[reg] = Uint32(0)

    @cute.jit
    def _copy_a_register_bundle_large_m(self, dst: cute.Tensor, src: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                dst[mb, reg] = src[mb, reg]

    @cute.jit
    def _copy_a_register_bundle_m8(self, dst: cute.Tensor, src: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            dst[reg] = src[reg]

    @cute.jit
    def _load_a_register_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._load_a_registers_m8_bundle(regs, smem_base, a_sh_rd, pipe, kk)
        else:
            self._load_a_registers_large_m_bundle(regs, smem_base, a_sh_rd, pipe, kk)

    @cute.jit
    def _clear_a_register_bundle(
        self,
        regs: cute.Tensor,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._clear_a_register_bundle_m8(regs)
        else:
            self._clear_a_register_bundle_large_m(regs)

    @cute.jit
    def _copy_a_register_bundle(
        self,
        dst: cute.Tensor,
        src: cute.Tensor,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._copy_a_register_bundle_m8(dst, src)
        else:
            self._copy_a_register_bundle_large_m(dst, src)

    @cute.jit
    def _load_b_scale_registers(
        self,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        b_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_b_off)
            + pipe * Int32(self.b_sh_stage)
            + Int32(self.b_sh_stride) * kk
            + b_sh_rd,
        )
        q0, q1, q2, q3 = ld_shared_v4_u32(b_addr)

        warp_id = tid // Int32(32)
        warp_row = warp_id // Int32(self.tb_n_warps)
        cur_group_id = Int32(self.b_sh_wr_iters) * warp_row + kk
        s_addr = (
            smem_base
            + Int32(self.sh_s_off * 16)
            + pipe * Int32(self.s_sh_stage * 16)
            + (s_sh_rd + cur_group_id * Int32(2 * self.s_sh_stride)) * Int32(8)
        )
        s_pack0, s_pack1 = ld_shared_v2_u32(s_addr)
        s0, s1 = self._dequant_e4m3x4_to_elem2x2(s_pack0)
        s2, s3 = self._dequant_e4m3x4_to_elem2x2(s_pack1)
        return q0, q1, q2, q3, s0, s1, s2, s3

    @cute.jit
    def _load_b_scale_register_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        q0, q1, q2, q3, s0, s1, s2, s3 = self._load_b_scale_registers(
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            pipe,
            kk,
        )
        regs[0, 0] = q0
        regs[0, 1] = q1
        regs[0, 2] = q2
        regs[0, 3] = q3
        regs[1, 0] = s0
        regs[1, 1] = s1
        regs[1, 2] = s2
        regs[1, 3] = s3

    @cute.jit
    def _clear_b_scale_register_bundle(self, regs: cute.Tensor):
        for row in cutlass.range_constexpr(2):
            for col in cutlass.range_constexpr(4):
                regs[row, col] = Uint32(0)

    @cute.jit
    def _copy_b_scale_register_bundle(self, dst: cute.Tensor, src: cute.Tensor):
        for row in cutlass.range_constexpr(2):
            for col in cutlass.range_constexpr(4):
                dst[row, col] = src[row, col]

    @cute.jit
    def _select_b_scale_register(self, jj: cutlass.Constexpr[int], regs: cute.Tensor):
        return regs[0, jj], regs[1, jj]

    @cute.jit
    def _load_next_fragment_bundle(
        self,
        b_scale_next: cute.Tensor,
        a_regs_next: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a_sh_rd: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        self._clear_b_scale_register_bundle(b_scale_next)
        self._clear_a_register_bundle(a_regs_next, uses_m_block_8)

        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                )
                self._load_a_register_bundle(
                    a_regs_next,
                    smem_base,
                    a_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                    uses_m_block_8,
                )
        else:
            next_tile = tile_idx + Int32(1)
            if next_tile < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32((pipe + 1) % self.num_stages),
                    Int32(0),
                )
                self._load_a_register_bundle(
                    a_regs_next,
                    smem_base,
                    a_sh_rd,
                    Int32((pipe + 1) % self.num_stages),
                    Int32(0),
                    uses_m_block_8,
                )

    @cute.jit
    def _scaled_dequant_b_fragment(self, frag: cute.Tensor, q: Uint32, s: Uint32):
        bq1 = q
        bq0 = bq1 << Uint32(8)
        b0_0, b0_1 = self._dequant_e2m1x4_to_elem2x2(bq0)
        b1_0, b1_1 = self._dequant_e2m1x4_to_elem2x2(bq1)
        s_lane0 = bfloat2_broadcast_lane(s, Int32(0))
        s_lane1 = bfloat2_broadcast_lane(s, Int32(1))
        b0_0 = self._elem2_mul(b0_0, s_lane0)
        b0_1 = self._elem2_mul(b0_1, s_lane0)
        b1_0 = self._elem2_mul(b1_0, s_lane1)
        b1_1 = self._elem2_mul(b1_1, s_lane1)
        frag[0, 0] = b0_0
        frag[0, 1] = b0_1
        frag[1, 0] = b1_0
        frag[1, 1] = b1_1

    @cute.jit
    def _mma_accumulate_m8(
        self,
        acc: cute.Tensor,
        jj: cutlass.Constexpr[int],
        a_regs: cute.Tensor,
        b_frag: cute.Tensor,
    ):
        d0, d1, d2, d3 = self._mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            acc[jj, 0],
            acc[jj, 1],
            acc[jj, 2],
            acc[jj, 3],
            b_frag[0, 0],
            b_frag[1, 0],
            b_frag[0, 1],
            b_frag[1, 1],
            a_regs[0],
            a_regs[1],
        )
        acc[jj, 0] = d0
        acc[jj, 1] = d1
        acc[jj, 2] = d2
        acc[jj, 3] = d3

    @cute.jit
    def _mma_accumulate_large_m(
        self,
        acc: cute.Tensor,
        a_regs: cute.Tensor,
        mb: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
        b_frag: cute.Tensor,
    ):
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[mb, jj, 0, 0],
            acc[mb, jj, 0, 1],
            acc[mb, jj, 0, 2],
            acc[mb, jj, 0, 3],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[0, 0],
            b_frag[0, 1],
        )
        acc[mb, jj, 0, 0] = d0
        acc[mb, jj, 0, 1] = d1
        acc[mb, jj, 0, 2] = d2
        acc[mb, jj, 0, 3] = d3
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[mb, jj, 1, 0],
            acc[mb, jj, 1, 1],
            acc[mb, jj, 1, 2],
            acc[mb, jj, 1, 3],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[1, 0],
            b_frag[1, 1],
        )
        acc[mb, jj, 1, 0] = d0
        acc[mb, jj, 1, 1] = d1
        acc[mb, jj, 1, 2] = d2
        acc[mb, jj, 1, 3] = d3

    @cute.jit
    def _stage_k_tile_async(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        tile_idx: Int32,
        block_valid_rows: Int32,
        m_tile_base: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
            row = a_rows_per_iter * Int32(i) + a_gl_rd_row
            # Dense: A is contiguous in M, so the global row is just
            # ``m_tile_base + row``.  The donor read this from a per-tile
            # smem route table.
            global_row = m_tile_base + row
            a_int4 = (
                global_row * a_gl_stride
                + tile_idx * Int32(self.a_gl_rd_delta_o)
                + a_gl_rd_col0
            )
            a_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_a_off)
                + pipe * Int32(self.a_sh_stage)
                + self._activation_smem_permuted_offset(
                    Int32(i * self.a_sh_wr_delta) + a_sh_wr
                ),
            )
            cp_async4_shared_global_pred(
                a_dst,
                get_ptr_as_int64(a_bf16_flat, a_int4 * Int32(8)),
                (row < block_valid_rows).to(Int32),
            )

        for i in cutlass.range_constexpr(self.b_sh_wr_iters):
            b_src_int4 = (
                b_gl_rd_base
                + tile_idx * Int32(self.cta_k_blocks) * b_gl_stride
                + Int32(i * (self.cta_threads // self.b_sh_stride)) * b_gl_stride
            )
            b_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_b_off)
                + pipe * Int32(self.b_sh_stage)
                + Int32(i * self.cta_threads)
                + tid,
            )
            cp_async4_shared_global(
                b_dst,
                get_ptr_as_int64(b_i32_flat, b_src_int4 * Int32(4)),
            )

        if tid < Int32(self.s_sh_stage):
            s_src_int4 = (
                scales_expert_off
                + s_gl_stride
                * (tile_idx * Int32(self.cta_k_blocks) + tid // Int32(self.s_sh_stride))
                + Int32(self.s_sh_stride) * output_n_tile
                + (tid % Int32(self.s_sh_stride))
            )
            s_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_s_off) + pipe * Int32(self.s_sh_stage) + tid,
            )
            cp_async4_shared_global(
                s_dst,
                get_ptr_as_int64(scales_i32_flat, s_src_int4 * Int32(4)),
            )

        cute.arch.cp_async_commit_group()

    @cute.jit
    def _prefetch_pipeline_step(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        m_tile_base: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        if cutlass.const_expr(kk == self.b_sh_wr_iters - 2):
            self._prefetch_lookahead_tile(
                a_bf16_flat,
                b_i32_flat,
                scales_i32_flat,
                smem_base,
                tid,
                pipe,
                tile_idx,
                k_tiles,
                reduce_k_tile,
                block_valid_rows,
                m_tile_base,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
            )

    @cute.jit
    def _prefetch_initial_tiles(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        m_tile_base: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        for pipe in cutlass.range_constexpr(self.num_stages - 1):
            if Int32(pipe) < k_tiles:
                self._stage_k_tile_async(
                    a_bf16_flat,
                    b_i32_flat,
                    scales_i32_flat,
                    smem_base,
                    tid,
                    Int32(pipe),
                    reduce_k_tile + Int32(pipe),
                    block_valid_rows,
                    m_tile_base,
                    a_gl_stride,
                    b_gl_stride,
                    s_gl_stride,
                    scales_expert_off,
                    b_gl_rd_base,
                    a_gl_rd_row,
                    a_gl_rd_col0,
                    a_sh_wr,
                    a_rows_per_iter,
                    output_n_tile,
                )
            else:
                cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.num_stages - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _prefetch_lookahead_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        m_tile_base: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        fetch_tile = tile_idx + Int32(self.num_stages - 1)
        if fetch_tile < k_tiles:
            self._stage_k_tile_async(
                a_bf16_flat,
                b_i32_flat,
                scales_i32_flat,
                smem_base,
                tid,
                Int32((pipe + self.num_stages - 1) % self.num_stages),
                reduce_k_tile + fetch_tile,
                block_valid_rows,
                m_tile_base,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
            )
        else:
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.num_stages - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _reduction_offsets(self, tid: Int32):
        red_idx = tid // Int32(self.b_sh_stride_threads)
        red_sh_stride = Int32(self.b_sh_stride_threads * 4 * 2)
        red_sh_delta = Int32(self.b_sh_stride_threads)
        red_sh_rd = red_sh_stride * (tid // Int32(self.b_sh_stride_threads)) + (
            tid % Int32(self.b_sh_stride_threads)
        )
        return red_idx, red_sh_stride, red_sh_delta, red_sh_rd

    @cute.jit
    def _fold_cta_partials_m8(self, acc: cute.Tensor, smem_base: Int32, tid: Int32):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid
            )
            if cutlass.const_expr(red_off == 2):
                if Int32(2) <= red_idx and red_idx < Int32(4):
                    for jj in cutlass.range_constexpr(4):
                        red_sh_wr = red_sh_delta * Int32(jj * 2) + (
                            red_sh_rd - red_sh_stride * Int32(2)
                        )
                        st_shared_v4_f32(
                            self._int4_addr(
                                smem_base, Int32(self.sh_red_off) + red_sh_wr
                            ),
                            acc[jj, 0],
                            acc[jj, 1],
                            acc[jj, 2],
                            acc[jj, 3],
                        )
                cute.arch.sync_threads()

            if Int32(1) <= red_idx and red_idx < Int32(2):
                for jj in cutlass.range_constexpr(4):
                    red_sh_wr = red_sh_delta * Int32(jj * 2) + (
                        red_sh_rd - red_sh_stride
                    )
                    if cutlass.const_expr(red_off > 1):
                        rd_addr = self._int4_addr(
                            smem_base,
                            Int32(self.sh_red_off)
                            + red_sh_delta * Int32(jj * 2)
                            + red_sh_rd,
                        )
                        wr_addr = self._int4_addr(
                            smem_base, Int32(self.sh_red_off) + red_sh_wr
                        )
                        r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                        w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                        acc[jj, 0] = acc[jj, 0] + r0 + w0
                        acc[jj, 1] = acc[jj, 1] + r1 + w1
                        acc[jj, 2] = acc[jj, 2] + r2 + w2
                        acc[jj, 3] = acc[jj, 3] + r3 + w3
                    st_shared_v4_f32(
                        self._int4_addr(smem_base, Int32(self.sh_red_off) + red_sh_wr),
                        acc[jj, 0],
                        acc[jj, 1],
                        acc[jj, 2],
                        acc[jj, 3],
                    )
            cute.arch.sync_threads()

            if red_idx == Int32(0):
                for jj in cutlass.range_constexpr(4):
                    rd_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off)
                        + red_sh_delta * Int32(jj * 2)
                        + red_sh_rd,
                    )
                    r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                    acc[jj, 0] = acc[jj, 0] + r0
                    acc[jj, 1] = acc[jj, 1] + r1
                    acc[jj, 2] = acc[jj, 2] + r2
                    acc[jj, 3] = acc[jj, 3] + r3
            cute.arch.sync_threads()

    @cute.jit
    def _output_store_cursor(self, tid: Int32, output_n_tile: Int32):
        c_gl_stride = Int32(self.size_n // 8)
        c_sh_stride = Int32(2 * self.cta_n_blocks + 1)
        c_gl_wr_delta = c_gl_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_sh_rd_delta = c_sh_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_gl_wr = (
            c_gl_stride * (tid // Int32(2 * self.cta_n_blocks))
            + (tid % Int32(2 * self.cta_n_blocks))
            + Int32(2 * self.cta_n_blocks) * output_n_tile
        )
        c_sh_rd = c_sh_stride * (tid // Int32(2 * self.cta_n_blocks)) + (
            tid % Int32(2 * self.cta_n_blocks)
        )
        return c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd

    @cute.jit
    def _drain_output_smem(
        self,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        c_gl_stride: Int32,
        c_gl_wr: Int32,
        c_gl_wr_delta: Int32,
        c_sh_rd: Int32,
        c_sh_rd_delta: Int32,
        block_valid_rows: Int32,
        m_tile_base: Int32,
        store_iters: cutlass.Constexpr[int],
    ):
        # c_gl_stride is in 8-bf16 vector units; it equals size_n // 8
        # (size_n = padded N for tile arithmetic).  The actual output
        # buffer has row stride size_n_real // 8 — equal to c_gl_stride
        # when no padding is active, smaller when N was padded up.
        c_gl_stride_real = Int32(self.size_n_real // 8)
        for _ in cutlass.range_constexpr(store_iters):
            row = c_gl_wr // c_gl_stride
            col_vec = c_gl_wr % c_gl_stride
            # Skip writes that would land in the (padded N – real N) tail
            # of the last N-tile.  With c_gl_stride == c_gl_stride_real,
            # this check is a no-op (col_vec is naturally bounded).
            if row < block_valid_rows and col_vec < c_gl_stride_real:
                # Dense: output row is m_tile_base + row (no route lookup).
                # Donor read this from the sh_route_off smem table.
                global_row = m_tile_base + row
                true_idx = global_row * c_gl_stride_real + col_vec
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + c_sh_rd)
                )
                # ``mul_topk_weights`` is hard-pinned to False for dense,
                # so the topk-weight multiplication branch is dead.
                if cutlass.const_expr(self.epilogue_relu2):
                    q0 = self._relu2_elem2(q0)
                    q1 = self._relu2_elem2(q1)
                    q2 = self._relu2_elem2(q2)
                    q3 = self._relu2_elem2(q3)
                st_global_v4_u32(
                    get_ptr_as_int64(c_bf16_flat, true_idx * Int32(8)),
                    q0,
                    q1,
                    q2,
                    q3,
                )
            c_gl_wr += c_gl_wr_delta
            c_sh_rd += c_sh_rd_delta
        cute.arch.sync_threads()

    @cute.jit
    def _store_tile_m8(
        self,
        acc: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        m_tile_base: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd = (
            self._output_store_cursor(tid, output_n_tile)
        )
        c_sh_wr = (
            Int32(8) * c_sh_stride * (((tid & Int32(31)) % Int32(4)) * Int32(2))
            + (tid & Int32(31)) // Int32(4)
            + Int32(64) * (tid // Int32(32))
        )

        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for jj in cutlass.range_constexpr(4):
                wr = c_sh_wr + Int32(16 * jj)
                self._st_shared_elem_from_f32(
                    smem_base + Int32(self.sh_red_off * 16) + (wr * Int32(2)),
                    acc[jj, 0] * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8) * c_sh_stride) * Int32(2)),
                    acc[jj, 1] * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8)) * Int32(2)),
                    acc[jj, 2] * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8) + Int32(8) * c_sh_stride) * Int32(2)),
                    acc[jj, 3] * write_scale,
                )
        cute.arch.sync_threads()

        store_iters = _covering_count(16, self.cta_threads // (2 * self.cta_n_blocks))
        self._drain_output_smem(
            c_bf16_flat,
            smem_base,
            c_gl_stride,
            c_gl_wr,
            c_gl_wr_delta,
            c_sh_rd,
            c_sh_rd_delta,
            block_valid_rows,
            m_tile_base,
            store_iters,
        )

    @cute.jit
    def _fold_cta_partials_large_m(
        self, acc: cute.Tensor, smem_base: Int32, tid: Int32
    ):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid
            )

            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                if cutlass.const_expr(red_off == 2):
                    if Int32(2) <= red_idx and red_idx < Int32(4):
                        for flat_j in cutlass.range_constexpr(8):
                            jj = flat_j // 2
                            half = flat_j % 2
                            red_sh_wr = red_sh_delta * Int32(flat_j) + (
                                red_sh_rd - red_sh_stride * Int32(2)
                            )
                            st_shared_v4_f32(
                                self._int4_addr(
                                    smem_base, Int32(self.sh_red_off) + red_sh_wr
                                ),
                                acc[mb, jj, half, 0],
                                acc[mb, jj, half, 1],
                                acc[mb, jj, half, 2],
                                acc[mb, jj, half, 3],
                            )
                    cute.arch.sync_threads()

                if Int32(1) <= red_idx and red_idx < Int32(2):
                    for flat_j in cutlass.range_constexpr(8):
                        jj = flat_j // 2
                        half = flat_j % 2
                        red_sh_wr = red_sh_delta * Int32(flat_j) + (
                            red_sh_rd - red_sh_stride
                        )
                        if cutlass.const_expr(red_off > 1):
                            rd_addr = self._int4_addr(
                                smem_base,
                                Int32(self.sh_red_off)
                                + red_sh_delta * Int32(flat_j)
                                + red_sh_rd,
                            )
                            wr_addr = self._int4_addr(
                                smem_base,
                                Int32(self.sh_red_off) + red_sh_wr,
                            )
                            r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                            w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                            acc[mb, jj, half, 0] = acc[mb, jj, half, 0] + r0 + w0
                            acc[mb, jj, half, 1] = acc[mb, jj, half, 1] + r1 + w1
                            acc[mb, jj, half, 2] = acc[mb, jj, half, 2] + r2 + w2
                            acc[mb, jj, half, 3] = acc[mb, jj, half, 3] + r3 + w3
                        st_shared_v4_f32(
                            self._int4_addr(
                                smem_base, Int32(self.sh_red_off) + red_sh_wr
                            ),
                            acc[mb, jj, half, 0],
                            acc[mb, jj, half, 1],
                            acc[mb, jj, half, 2],
                            acc[mb, jj, half, 3],
                        )
                cute.arch.sync_threads()

                if red_idx == Int32(0):
                    for flat_j in cutlass.range_constexpr(8):
                        jj = flat_j // 2
                        half = flat_j % 2
                        rd_addr = self._int4_addr(
                            smem_base,
                            Int32(self.sh_red_off)
                            + red_sh_delta * Int32(flat_j)
                            + red_sh_rd,
                        )
                        r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                        acc[mb, jj, half, 0] = acc[mb, jj, half, 0] + r0
                        acc[mb, jj, half, 1] = acc[mb, jj, half, 1] + r1
                        acc[mb, jj, half, 2] = acc[mb, jj, half, 2] + r2
                        acc[mb, jj, half, 3] = acc[mb, jj, half, 3] + r3
                cute.arch.sync_threads()

    @cute.jit
    def _write_bf16x2_shared(
        self,
        smem_base: Int32,
        half2_idx: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        write_scale: cutlass.Float32,
    ):
        packed = self._pack_f32x2_to_elem2(c0 * write_scale, c1 * write_scale)
        st_shared_u32(
            smem_base + Int32(self.sh_red_off * 16) + half2_idx * Int32(4),
            packed,
        )

    @cute.jit
    def _store_tile_large_m(
        self,
        acc: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        m_tile_base: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd = (
            self._output_store_cursor(tid, output_n_tile)
        )
        c_sh_wr = (
            Int32(4) * c_sh_stride * ((tid & Int32(31)) // Int32(4))
            + (tid & Int32(31)) % Int32(4)
            + Int32(32) * (tid // Int32(32))
        )

        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                for jj in cutlass.range_constexpr(4):
                    wr = c_sh_wr + Int32(8 * jj)
                    self._write_bf16x2_shared(
                        smem_base,
                        wr,
                        acc[mb, jj, 0, 0],
                        acc[mb, jj, 0, 1],
                        write_scale,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(0),
                        acc[mb, jj, 0, 2],
                        acc[mb, jj, 0, 3],
                        write_scale,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + Int32(4),
                        acc[mb, jj, 1, 0],
                        acc[mb, jj, 1, 1],
                        write_scale,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(4),
                        acc[mb, jj, 1, 2],
                        acc[mb, jj, 1, 3],
                        write_scale,
                    )
                c_sh_wr += Int32(16 * (4 * (2 * self.cta_n_blocks + 1)))
        cute.arch.sync_threads()

        store_iters = _covering_count(
            16 * self.cta_m_blocks,
            self.cta_threads // (2 * self.cta_n_blocks),
        )
        self._drain_output_smem(
            c_bf16_flat,
            smem_base,
            c_gl_stride,
            c_gl_wr,
            c_gl_wr_delta,
            c_sh_rd,
            c_sh_rd_delta,
            block_valid_rows,
            m_tile_base,
            store_iters,
        )


_CACHE: dict[tuple, W4A16GemmCompileResult] = {}


def _normalize_element_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    raise TypeError(f"unsupported W4A16 activation dtype {dtype}")


def _cutlass_element_dtype(element_dtype: str):
    if element_dtype == "bf16":
        return cutlass.BFloat16
    if element_dtype == "fp16":
        return cutlass.Float16
    raise ValueError(f"unsupported element_dtype {element_dtype!r}")


def compile_w4a16_gemm(
    *,
    size_m: int,
    size_n: int,
    size_k: int,
    tile_n: int,
    tile_k: int,
    cta_m_size: int,
    element_dtype: str = "bf16",
    num_stages: int = _DEFAULT_NUM_STAGES,
    size_n_real: int | None = None,
) -> W4A16GemmCompileResult:
    """Compile a dense W4A16 GEMM kernel for the given problem shape + tile.

    Returns a cached ``W4A16GemmCompileResult``.  The compiled callable
    expects 7 device tensors:
      a, b_packed_i32, c, scales_i32, global_scale, c_tmp_f32, locks_i32
    plus a CUDA stream.  Tile arithmetic comes from the donor MoE kernel.
    """
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    if torch.cuda.is_available():
        device = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device)
        sms = int(props.multi_processor_count)
        max_shared_mem = int(
            getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
        )
    else:
        device = None
        sms = 120
        max_shared_mem = _DEFAULT_MAX_SHARED_MEM
    if size_n_real is None:
        size_n_real = size_n
    cache_key = (
        "w4a16_marlin_dense",
        device,
        sms,
        max_shared_mem,
        element_dtype,
        size_m,
        size_n,
        int(size_n_real),
        size_k,
        tile_n,
        tile_k,
        cta_m_size,
        int(num_stages),
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    max_m_blocks = (size_m + cta_m_size - 1) // cta_m_size

    a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (size_m * size_k,),
        assumed_align=16,
    )
    # Packed B is in donor's _repack_4bit_no_perm output shape:
    # (size_k // 16) * (size_n // 16 * 32) int32 words = K*N/8 int32 = K*N/2 bytes.
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        ((size_k // 16) * (size_n // 16 * 32),),
        assumed_align=16,
    )
    # C buffer is laid out at the *real* N (not padded) so the wrapper
    # can hand the caller a clean contiguous (M, N_real) output without
    # any post-kernel slice/copy.
    c_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (size_m * int(size_n_real),),
        assumed_align=16,
    )
    # Permuted FP8 scales (donor's _permute_packed_scales): (K/16) * (N/4) i32.
    # N is the *padded* N -- the scales tensor lives in padded layout when
    # size_n_real < size_n_padded.
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        ((size_k // 16) * (size_n // 4),),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=16,
    )
    # Split-K scratch + atomic locks.  Sized like the donor; the dense
    # kernel uses split-K when reduce_slice_count > 1 in the scheduler.
    c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                size_n * max_m_blocks * cta_m_size,
                4 * 256 * cta_m_size * 256,
            ),
        ),
        assumed_align=16,
    )
    locks_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (4 * 256,),
        assumed_align=16,
    )

    kernel = W4A16GemmKernel(
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        tile_n=tile_n,
        tile_k=tile_k,
        cta_m_size=cta_m_size,
        element_dtype=element_dtype,
        num_stages=int(num_stages),
        size_n_real=int(size_n_real),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = cute.compile(
        kernel,
        a_fake,
        b_fake,
        c_fake,
        scales_fake,
        global_scale_fake,
        c_tmp_fake,
        locks_fake,
        current_cuda_stream(),
    )
    result = W4A16GemmCompileResult(
        compiled=compiled,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=cta_m_size,
        max_m_blocks=max_m_blocks,
        blocks_per_sm=kernel.blocks_per_sm,
        num_stages=int(num_stages),
    )
    _CACHE[cache_key] = result
    return result


@dataclass(frozen=True)
class _DenseMarlinPackedWeights:
    """v5-format dense weights repacked for the MoE-stripped kernel.

    * ``b_packed_i32``: flat int32, ``K*N/8`` elements, ``_repack_4bit_no_perm`` layout.
    * ``scales_i32``: flat int32, ``K*N/64`` elements, permuted + processed FP8 scales.
    * ``global_scale``: shape ``(1,)`` float32, processed via
      ``_process_nvfp4_packed_global_scale`` and divided by the combined scale factor.
    """

    b_packed_i32: torch.Tensor
    scales_i32: torch.Tensor
    global_scale: torch.Tensor


def _pack_dense_weights_for_marlin(
    w_fp4: torch.Tensor,
    w_blockscale_swizzled: torch.Tensor,
    w_alpha: torch.Tensor,
    *,
    size_n: int,
    size_k: int,
    a_dtype: torch.dtype = torch.bfloat16,
    size_n_padded: int | None = None,
) -> _DenseMarlinPackedWeights:
    """Convert v5-format quantized weights to the MoE kernel's packed layout.

    Steps mirror ``prepare_w4a16_packed_weights`` minus the expert dim:

    1. ``w_fp4 (N, K/2) u8`` → view as int32 ``(N, K/8) i32`` → transpose
       to ``(K/8, N) i32`` → ``_repack_4bit_no_perm`` → ``(K/16, 2N) i32``.
    2. swizzled FP8 scales → ``unswizzle_block_scale`` → ``(N, K/16) f32``
       → cast to ``a_dtype`` → ``_permute_packed_scales(scales.T, ...)``
       → ``_process_nvfp4_packed_scales`` → ``(K/16, N) fp8`` viewed as i32.
    3. ``w_alpha`` (scalar f32) → ``_process_nvfp4_packed_global_scale``
       (fold FP4 → ``a_dtype`` exponent-bias shift) → divide by
       ``combined_scale_factor`` from step 2.
    """
    # When ``size_n_padded > size_n``, the caller wants the kernel to
    # see a larger N (e.g. to unlock tile_n=128 when actual N=10304).
    # We pad both the FP4 weight and the unswizzled scales with zero
    # rows; zero-weight rows contribute zero to the matmul, and the
    # kernel's per-column write bound (``size_n_real``) skips writes
    # to the padded tail.
    if size_n_padded is None:
        size_n_padded = size_n
    size_n_for_kernel = int(size_n_padded)
    if size_n_for_kernel < size_n:
        raise ValueError("size_n_padded must be >= size_n")
    pad_rows = size_n_for_kernel - int(size_n)

    if pad_rows > 0:
        zero_rows = torch.zeros(
            (pad_rows, w_fp4.shape[1]), dtype=w_fp4.dtype, device=w_fp4.device,
        )
        w_fp4_padded = torch.cat([w_fp4, zero_rows], dim=0).contiguous()
    else:
        w_fp4_padded = w_fp4
    qweight_i32 = w_fp4_padded.view(torch.int32).T.contiguous()
    b_packed = _repack_4bit_no_perm(
        qweight_i32, size_k=size_k, size_n=size_n_for_kernel,
    )
    b_packed_flat = b_packed.contiguous().view(-1)

    cols_blocks = size_k // _SF_VEC_SIZE
    # ``unswizzle_block_scale`` pads its rows-padded layout up to the
    # nearest multiple of 128 internally; requesting rows=size_n_for_kernel
    # returns the unswizzled slice including the zero-padded tail when
    # the FP8 swizzle was already wide enough (the standard case for
    # quantize_dense_weight_to_fp4 output).  When the swizzle layout
    # isn't wide enough, fall back to padding the unswizzled scales.
    swizzle_rows_padded = ((int(size_n) + 127) // 128) * 128
    if size_n_for_kernel <= swizzle_rows_padded:
        scales_unswizzled_f32 = unswizzle_block_scale(
            w_blockscale_swizzled,
            rows=size_n_for_kernel,
            cols_blocks=cols_blocks,
        )
    else:
        scales_unswizzled_real = unswizzle_block_scale(
            w_blockscale_swizzled, rows=int(size_n), cols_blocks=cols_blocks,
        )
        pad = torch.zeros(
            (size_n_for_kernel - int(size_n), cols_blocks),
            dtype=scales_unswizzled_real.dtype,
            device=scales_unswizzled_real.device,
        )
        scales_unswizzled_f32 = torch.cat(
            [scales_unswizzled_real, pad], dim=0,
        ).contiguous()
    scales_fp8 = scales_unswizzled_f32.to(torch.float8_e4m3fn)
    scales_a_dtype = scales_fp8.to(a_dtype)
    combined_scale_factor = _nvfp4_compute_scale_factor(scales_a_dtype, a_dtype)
    packed_scales = _permute_packed_scales(
        scales_a_dtype.T,
        size_k=size_k,
        size_n=size_n_for_kernel,
        group_size=_SF_VEC_SIZE,
    )
    processed_scales = _process_nvfp4_packed_scales(
        packed_scales, scale_factor=combined_scale_factor,
    )
    scales_i32_flat = (
        processed_scales.view(torch.uint8).view(torch.int32).contiguous().view(-1)
    )

    alpha_1d = w_alpha.reshape(1).contiguous()
    processed_global = _process_nvfp4_packed_global_scale(
        alpha_1d, a_dtype=a_dtype,
    ).to(torch.float32)
    processed_global = (processed_global / combined_scale_factor).contiguous()

    return _DenseMarlinPackedWeights(
        b_packed_i32=b_packed_flat,
        scales_i32=scales_i32_flat,
        global_scale=processed_global,
    )


class DenseGemmW4A16CuteMarlinKernel:
    """W4A16 dense GEMM cute backend, v6 (Marlin-style cp.async + register pipeline).

    Strip-port of ``b12x/moe/fused/w4a16/kernel.py`` to the dense case
    (one expert, no top-k routing).  Same FP4 dequant + bf16 MMA atom as
    the MoE donor; the routing/expert/topk axes have been stripped so each
    CTA computes one ``(cta_m_size x tile_n)`` output tile of a dense
    matmul against the donor's pre-pipelined cp.async + register-tiled
    epilogue.

    Public surface mirrors ``DenseGemmW4A16CutePrefillKernel`` (v5): takes
    v5-format ``(w_fp4, w_blockscale, w_alpha)`` and repacks internally
    to the MoE kernel's input layout (cached per weight tensor identity).
    """

    _ALLOWED_CTA_M_SIZES = _ALLOWED_ROUTED_SIZES

    @classmethod
    def is_supported(cls, m: int, k: int, n: int) -> bool:
        if m <= 0 or k <= 0 or n <= 0:
            return False
        # _repack_4bit_no_perm requires K%16==0 and N%64==0.
        # The smallest tile_k/tile_n in the config tables is 64, and tile_k|k
        # + tile_n|n must hold.
        if k % 64 != 0 or n % 64 != 0:
            return False
        return True

    def is_supported_instance(self, m: int, k: int, n: int) -> bool:
        return self.is_supported(m, k, n)

    def __init__(
        self,
        *,
        cta_m_size: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        num_stages: int | None = None,
        element_dtype: str = "bf16",
    ) -> None:
        if cta_m_size is not None and cta_m_size not in self._ALLOWED_CTA_M_SIZES:
            raise ValueError(
                f"cta_m_size must be one of {self._ALLOWED_CTA_M_SIZES}, got {cta_m_size}"
            )
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if num_stages is not None and int(num_stages) < 2:
            raise ValueError("num_stages must be at least 2")
        self._cta_m_size = cta_m_size
        self._tile_n = tile_n
        self._tile_k = tile_k
        self._num_stages = num_stages
        self._element_dtype = element_dtype
        self._a_dtype = (
            torch.bfloat16 if element_dtype == "bf16" else torch.float16
        )
        self._weight_cache: dict = {}
        self._scratch_cache: dict = {}

    def _pick_cta_m_size(self, m: int) -> int:
        if self._cta_m_size is not None:
            return int(self._cta_m_size)
        if m <= 8:
            return 8
        if m <= 16:
            return 16
        if m <= 32:
            return 32
        if m <= 48:
            return 48
        return 64

    def _pick_n_padded(self, *, n: int, k: int) -> int:
        """Round N up to the next multiple of 128 (the largest tile_n).

        When N already divides 128 the result equals N and no padding
        happens.  When it doesn't (e.g. mamba_in_proj N=10304), padding
        unlocks tile_n=128 which is materially faster than the only
        otherwise-valid tile_n=64.

        Skip the padding if it grows N by more than ~1% — the per-call
        wasted compute on the padding tail then outweighs the geometry
        win.  The Nano3.5 outlier (10304→10368, +0.6%) lands well
        inside the budget.
        """
        n = int(n)
        if n % 128 == 0:
            return n
        n_padded = ((n + 127) // 128) * 128
        if n_padded - n > max(64, n // 64):
            return n  # padding overhead too high; keep the narrow tile.
        return n_padded

    def _pick_num_stages(self, *, tile_n: int, tile_k: int) -> int:
        """Pick pipeline depth (cp.async → MMA stages) by tile geometry.

        The optimal depth is a property of the tile shape, not the
        problem shape.  See ``_NUM_STAGES_BY_TILE`` for the autotune
        table — Spark autotune 2026-05-18.
        """
        if self._num_stages is not None:
            return int(self._num_stages)
        return _select_num_stages_for_tile(tile_n=int(tile_n), tile_k=int(tile_k))

    def _pick_tile_config(
        self, *, m: int, n: int, k: int, cta_m_size: int, num_stages: int,
    ) -> tuple[int, int]:
        if self._tile_n is not None and self._tile_k is not None:
            return int(self._tile_n), int(self._tile_k)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            sms = int(props.multi_processor_count)
            max_shared_mem = int(
                getattr(
                    props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM
                )
            )
        else:
            sms = 120
            max_shared_mem = _DEFAULT_MAX_SHARED_MEM
        tile_k, tile_n, _, _ = _select_tile_config(
            problem_m=m,
            problem_n=n,
            problem_k=k,
            top_k=1,
            moe_block_size=cta_m_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            num_stages=num_stages,
        )
        return int(tile_n), int(tile_k)

    def _get_packed_weights(
        self,
        w_fp4: torch.Tensor,
        w_blockscale: torch.Tensor,
        w_alpha: torch.Tensor,
        *,
        size_n: int,
        size_k: int,
        size_n_padded: int | None = None,
    ) -> _DenseMarlinPackedWeights:
        size_n_padded = int(size_n if size_n_padded is None else size_n_padded)
        cache_key = (
            int(w_fp4.data_ptr()),
            int(w_blockscale.data_ptr()),
            float(w_alpha.item()),
            int(size_n),
            size_n_padded,
            int(size_k),
            self._element_dtype,
        )
        cached = self._weight_cache.get(cache_key)
        if cached is not None:
            return cached
        packed = _pack_dense_weights_for_marlin(
            w_fp4, w_blockscale, w_alpha,
            size_n=size_n, size_k=size_k, a_dtype=self._a_dtype,
            size_n_padded=size_n_padded,
        )
        self._weight_cache[cache_key] = packed
        return packed

    def _get_scratch(
        self,
        *,
        size_m_padded: int,
        size_n: int,
        cta_m_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            sms = int(props.multi_processor_count)
        else:
            sms = 48
        max_m_blocks = (size_m_padded + cta_m_size - 1) // cta_m_size
        # Mirror donor's ``packed_gemm_scratch_elements`` with
        # ``route_slots = padded_M``: c_tmp's size is bounded by either
        # the full per-output-element scratch (N * padded_M) or a
        # per-CTA upper bound based on grid_x * c_size.  Small dense
        # problems hit the first bound; large problems with split-K hit
        # the second.
        c_tmp_elements = min(
            int(size_n) * int(max_m_blocks) * int(cta_m_size),
            int(sms) * 4 * int(cta_m_size) * 256,
        )
        if cta_m_size == 8:
            c_tmp_elements *= 2
        c_tmp_elements = max(int(c_tmp_elements), 1)
        # Locks: one int32 per CTA (grid_x ≤ sms * 4 = blocks_per_sm cap).
        locks_elements = int(sms) * 4
        cache_key = (
            int(getattr(device, "index", 0) or 0),
            int(size_m_padded),
            int(size_n),
            int(cta_m_size),
        )
        cached = self._scratch_cache.get(cache_key)
        if cached is not None:
            c_tmp, locks = cached
            if int(c_tmp.numel()) >= c_tmp_elements and int(locks.numel()) >= locks_elements:
                # Locks: kernel resets them to 0 at the end of each launch
                # when split-K is used, but a zero-initialized scratch from
                # creation stays zero between calls when reduce_slice_count=1.
                # Zero defensively to avoid any cross-call interference.
                locks.zero_()
                return c_tmp[:c_tmp_elements], locks[:locks_elements]
        c_tmp = torch.empty(c_tmp_elements, dtype=torch.float32, device=device)
        locks = torch.zeros(locks_elements, dtype=torch.int32, device=device)
        self._scratch_cache[cache_key] = (c_tmp, locks)
        return c_tmp, locks

    def __call__(
        self,
        x: torch.Tensor,
        w_fp4: torch.Tensor,
        w_blockscale_swizzled_u8: torch.Tensor,
        w_alpha: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
        if x.dtype != self._a_dtype:
            raise TypeError(
                f"x dtype {x.dtype} does not match kernel dtype {self._a_dtype}"
            )
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if w_fp4.dim() != 2:
            raise ValueError(f"w_fp4 must be 2D, got shape {tuple(w_fp4.shape)}")
        m, k = int(x.shape[0]), int(x.shape[1])
        n = int(w_fp4.shape[0])
        if int(w_fp4.shape[1]) * 2 != k:
            raise ValueError(
                f"w_fp4 shape {tuple(w_fp4.shape)} (K={int(w_fp4.shape[1]) * 2}) "
                f"does not match x.shape[1]={k}"
            )
        if not self.is_supported(m, k, n):
            raise ValueError(
                f"unsupported shape for v6 marlin dense: M={m}, K={k}, N={n}"
            )

        cta_m_size = self._pick_cta_m_size(m)
        # N-padding: when the actual N doesn't divide 128, pad up to the
        # next multiple of 128 internally so the kernel can use the
        # wider tile_n=128 geometry (typically ~25% faster than the
        # narrow tile_n=64 fallback).  ``size_n_real`` < ``size_n``
        # tells the kernel to skip writes to the padded tail; the
        # caller still receives a clean (M, N) output buffer.
        n_padded = self._pick_n_padded(n=n, k=k)
        # Tile pick is stable across num_stages for the Nano3.5 shapes
        # (the candidate set is determined by N%tile_n, K%tile_k, smem
        # cap at the deepest stages; tile_n/tile_k are picked before
        # depth so order doesn't matter).  Use _DEFAULT_NUM_STAGES for
        # the tile selection's smem-fit check.
        tile_n, tile_k = self._pick_tile_config(
            m=m, n=n_padded, k=k, cta_m_size=cta_m_size,
            num_stages=_DEFAULT_NUM_STAGES,
        )
        num_stages = self._pick_num_stages(tile_n=tile_n, tile_k=tile_k)

        m_padded = ((m + cta_m_size - 1) // cta_m_size) * cta_m_size
        if m_padded != m:
            x_used = torch.zeros(m_padded, k, dtype=x.dtype, device=x.device)
            x_used[:m].copy_(x)
        else:
            x_used = x.contiguous() if not x.is_contiguous() else x

        if out is None:
            out = torch.empty(m, n, dtype=self._a_dtype, device=x.device)
        else:
            if tuple(out.shape) != (m, n):
                raise ValueError(
                    f"out shape {tuple(out.shape)} != expected {(m, n)}"
                )
            if out.dtype != self._a_dtype:
                raise TypeError(
                    f"out dtype {out.dtype} does not match kernel dtype {self._a_dtype}"
                )
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")

        packed = self._get_packed_weights(
            w_fp4, w_blockscale_swizzled_u8, w_alpha,
            size_n=n, size_k=k, size_n_padded=n_padded,
        )
        c_tmp, locks = self._get_scratch(
            size_m_padded=m_padded, size_n=n_padded, cta_m_size=cta_m_size,
            device=x.device,
        )

        # The kernel bakes ``size_m`` into the JIT and uses it as the
        # M-bound for the C-write path.  Passing the real M means only
        # rows ``[0, M)`` get written -- the padded ``x_used`` rows
        # produce no output.  Pre-padded ``x_used`` ensures the A-row
        # indexing never reads OOB (``global_row = m_tile_base + row``
        # ranges over ``[0, m_padded)``).
        result = compile_w4a16_gemm(
            size_m=m,
            size_n=n_padded,
            size_n_real=n,
            size_k=k,
            tile_n=tile_n,
            tile_k=tile_k,
            cta_m_size=cta_m_size,
            element_dtype=self._element_dtype,
            num_stages=num_stages,
        )
        stream = current_cuda_stream()
        result.compiled(
            x_used.view(-1),
            packed.b_packed_i32,
            out.view(-1),
            packed.scales_i32,
            packed.global_scale,
            c_tmp,
            locks,
            stream,
        )
        return out


__all__ = [
    "DenseGemmW4A16CuteMarlinKernel",
    "W4A16GemmCompileResult",
    "W4A16GemmKernel",
    "compile_w4a16_gemm",
]

