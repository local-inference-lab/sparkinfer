from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, T, Uint8, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from b12x.cute.utils import current_cuda_stream, make_ptr
from b12x.cute.fp4 import elem_pointer, shared_ptr_to_u32, st_shared_u8
from b12x.distributed._oneshot_common import align_bytes, cutlass_dtype
from b12x.gemm.dense import DenseGemmKernel, sm120_make_smem_layout_sfa, sm120_make_smem_layout_sfb


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


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype,
    *,
    assumed_align: int = 16,
):
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    if tensor.ndim >= 2:
        leading_dim = next((idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None)
        if leading_dim is not None:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


class FC2SharedLayoutProbe:
    def __init__(self, *, hidden_size: int):
        self._dense_cls = DenseGemmKernel
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = 16
        tile_k = self.sf_vec_size * 8
        self.tile_shape_mnk = (128, 128, tile_k)
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.epi_tile = (128, 128)
        self.occupancy = 1
        self.num_mma_warps = 4
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.buffer_align_bytes = 1024
        self.hidden_size = hidden_size

    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFA(self, sfa_tensor, tiled_mma)

    def _get_layoutSFA_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFA_TV(self, tiled_mma)

    def _setup_attributes(self):
        import cutlass.utils.blackwell_helpers as sm120_utils

        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype, self.acc_dtype, self.sf_dtype,
        )
        atom_layout = cute.make_layout((2, 2, 1))
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False,
        )
        self.tiled_mma = cute.make_tiled_mma(mma_op, atom_layout, permutation_mnk=permutation_mnk)
        self.mma_atom = cute.make_mma_atom(mma_op)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        sfa_smem = sm120_make_smem_layout_sfa(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1,
        )
        sfb_smem = sm120_make_smem_layout_sfb(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1,
        )

        ab_stage, epi_stage = self._dense_cls._compute_stages(
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
        while ab_stage > 1 and 32 % ab_stage != 0:
            ab_stage -= 1
        self.ab_stage = ab_stage
        self.epi_stage = 1
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )

    @cute.jit
    def __call__(
        self,
        packed_a: cute.Tensor,
        sfa_ptr: cute.Pointer,
        b_w13: cute.Tensor,
        target_tidx: Int32,
        target_src_idx: Int32,
        write_sfa: Int32,
        write_mode: Int32,
        stream: cuda.CUstream,
    ):
        self.a_dtype = packed_a.element_type
        self.b_dtype = b_w13.element_type
        self.sf_dtype = sfa_ptr.dtype
        self.a_layout = utils.LayoutEnum.from_tensor(packed_a)
        self.b_layout = utils.LayoutEnum.from_tensor(b_w13)
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        self._setup_attributes()

        self.kernel(
            packed_a,
            sfa_ptr,
            b_w13,
            target_tidx,
            target_src_idx,
            write_sfa,
            write_mode,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=[1, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        packed_a: cute.Tensor,
        sfa_ptr: cute.Pointer,
        b_w13: cute.Tensor,
        target_tidx: Int32,
        target_src_idx: Int32,
        write_sfa: Int32,
        write_mode: Int32,
        tiled_mma: cute.TiledMma,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.ComposedLayout,
        sfb_smem_staged: cute.ComposedLayout,
        epi_smem_staged: cute.ComposedLayout,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = tidx // Int32(32)
        lane_id = tidx & Int32(31)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_staged)],
                self.buffer_align_bytes,
            ]

        storage = smem.allocate(Storage)
        cute.arch.sync_threads()

        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_staged)
        sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
        sA_u8_view = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
        sSFA_u8_view = cute.recast_tensor(sSFA, cutlass.Uint8)

        a_base = shared_ptr_to_u32(storage.sA.data_ptr())
        sfa_base = shared_ptr_to_u32(storage.sSFA.data_ptr())

        atom_ld_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4), self.a_dtype
        )
        atom_ld_SF = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrSFA = self._dense_cls._partition_fragment_SFA(self, sSFA[None, None, 0], thr_mma, tidx)
        smem_copy_A = cute.make_tiled_copy_A(atom_ld_A, tiled_mma)
        smem_copy_SFA = cute.make_tiled_copy(
            atom_ld_SF,
            self._dense_cls._get_layoutSFA_TV(self, tiled_mma),
            (cute.size(tiled_mma.permutation_mnk[0]), cute.size(tiled_mma.permutation_mnk[2])),
        )

        thr_ld_A = smem_copy_A.get_slice(tidx)
        thr_ld_SFA = smem_copy_SFA.get_slice(tidx)
        csA = thr_ld_A.partition_S(sA)
        crA = thr_ld_A.retile(tCrA)
        csSFA = thr_ld_SFA.partition_S(sSFA)
        crSFA = thr_ld_SFA.retile(tCrSFA)
        csA_phase2 = csA[None, None, None, 0]
        csSFA_phase2 = csSFA[None, None, None, 0]

        src_a_all_u8 = cute.flatten(cute.recast_tensor(csA_phase2, cutlass.Uint8))
        src_sfa_all_u8 = cute.flatten(
            cute.recast_tensor(cute.filter_zeros(csSFA_phase2), cutlass.Uint8)
        )

        fill_idx = Int32(tidx)
        while fill_idx < Int32(cute.size(sA_u8)):
            sA_u8[fill_idx] = Uint8(0)
            fill_idx += Int32(self.threads_per_cta)
        sSFA_u8 = cute.flatten(cute.recast_tensor(sSFA, cutlass.Uint8))
        fill_idx = Int32(tidx)
        while fill_idx < Int32(cute.size(sSFA_u8)):
            sSFA_u8[fill_idx] = Uint8(0)
            fill_idx += Int32(self.threads_per_cta)

        if tidx == target_tidx:
            if write_mode == Int32(0):
                if write_sfa == Int32(0):
                    src_a_all_u8[target_src_idx] = Uint8(99)
                else:
                    src_sfa_all_u8[target_src_idx] = Uint8(99)
            elif write_mode == Int32(1):
                if write_sfa == Int32(0):
                    sA_u8[target_src_idx] = Uint8(99)
                else:
                    sSFA_u8[target_src_idx] = Uint8(99)
            elif write_mode == Int32(2):
                logical_a_dim0 = Int32(cute.size(sA_u8_view, mode=[0]))
                if write_sfa == Int32(0):
                    coord0 = target_src_idx % logical_a_dim0
                    coord1 = target_src_idx // logical_a_dim0
                    sA_u8_view[coord0, coord1] = Uint8(99)
                else:
                    src_sfa_all_u8[target_src_idx] = Uint8(99)
            else:
                if write_sfa == Int32(0):
                    st_shared_u8(a_base + target_src_idx, Uint8(99))
                else:
                    st_shared_u8(sfa_base + target_src_idx, Uint8(99))
        cute.arch.sync_threads()
        cute.copy(smem_copy_A, csA_phase2[None, None, Int32(0)], crA[None, None, Int32(0)])
        fz_csSFA_p2 = cute.filter_zeros(csSFA_phase2)
        fz_crSFA = cute.filter_zeros(crSFA)
        cute.copy(smem_copy_SFA, fz_csSFA_p2[None, None, Int32(0)], fz_crSFA[None, None, Int32(0)])

        if tidx == target_tidx:
            src_a_u8_view = cute.recast_tensor(csA_phase2, cutlass.Uint8)
            src_sfa_u8_view = cute.recast_tensor(cute.filter_zeros(csSFA_phase2), cutlass.Uint8)
            cute.printf(
                "probe target_tidx={} warp={} lane={} target_src_idx={} write_sfa={} write_mode={} sizes rawA={} rawSFA={} srcA={} srcSFA={}",
                target_tidx,
                warp_idx,
                lane_id,
                target_src_idx,
                write_sfa,
                write_mode,
                Int32(cute.size(sA_u8)),
                Int32(cute.size(sSFA_u8)),
                Int32(cute.size(src_a_all_u8)),
                Int32(cute.size(src_sfa_all_u8)),
            )
            cute.printf(
                "probe srcA modes={} {} {}",
                Int32(cute.size(src_a_u8_view, mode=[0])),
                Int32(cute.size(src_a_u8_view, mode=[1])),
                Int32(cute.size(src_a_u8_view, mode=[2])),
            )
            cute.printf(
                "probe srcSFA modes={} {} {}",
                Int32(cute.size(src_sfa_u8_view, mode=[0])),
                Int32(cute.size(src_sfa_u8_view, mode=[1])),
                Int32(cute.size(src_sfa_u8_view, mode=[2])),
            )
            cute.printf(
                "probe logicalA modes={} {}",
                Int32(cute.size(sA_u8_view, mode=[0])),
                Int32(cute.size(sA_u8_view, mode=[1])),
            )
            cute.printf(
                "probe logicalSFA modes={} {}",
                Int32(cute.size(sSFA_u8_view, mode=[0])),
                Int32(cute.size(sSFA_u8_view, mode=[1])),
            )
            found = Int32(0)
            scan_idx = Int32(0)
            while scan_idx < Int32(2048) and found < Int32(32):
                raw_a_val = Int32(_ld_shared_u8(a_base + scan_idx))
                if raw_a_val != Int32(0):
                    cute.printf("probe physA nz off={} val={}", scan_idx, raw_a_val)
                    found += Int32(1)
                scan_idx += Int32(1)
            found = Int32(0)
            scan_idx = Int32(0)
            while scan_idx < Int32(2048) and found < Int32(32):
                raw_sfa_val = Int32(_ld_shared_u8(sfa_base + scan_idx))
                if raw_sfa_val != Int32(0):
                    cute.printf("probe physSFA nz off={} val={}", scan_idx, raw_sfa_val)
                    found += Int32(1)
                scan_idx += Int32(1)
            frag_a = cute.flatten(
                cute.recast_tensor(tCrA[None, Int32(0), Int32(0)], cutlass.Uint32)
            )
            frag_sfa = cute.flatten(
                cute.recast_tensor(tCrSFA[None, Int32(0), Int32(0)], cutlass.Uint32)
            )
            cute.printf(
                "probe frag a={} {} sfa={} {}",
                Int32(frag_a[Int32(0)]),
                Int32(frag_a[Int32(1)]),
                Int32(frag_sfa[Int32(0)]),
                Int32(frag_sfa[Int32(1)]),
            )


def _compile_probe(hidden_size: int):
    probe = FC2SharedLayoutProbe(hidden_size=hidden_size)
    fake_packed_a = cute.runtime.make_fake_compact_tensor(
        cutlass.Float4E2M1FN, (128, hidden_size, 1), stride_order=(1, 0, 2), assumed_align=16
    )
    fake_b_w13 = cute.runtime.make_fake_compact_tensor(
        cutlass.Float4E2M1FN, (256, hidden_size, 1), stride_order=(1, 0, 2), assumed_align=16
    )
    fake_sfa_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(
        probe,
        fake_packed_a,
        fake_sfa_ptr,
        fake_b_w13,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        current_cuda_stream(),
    )


def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    hidden_size = 4096

    packed_a = torch.empty((128, hidden_size, 1), dtype=torch.uint8, device=device)
    b_w13 = torch.empty((256, hidden_size, 1), dtype=torch.uint8, device=device)
    sfa = torch.empty(
        (128 * (((hidden_size // 16 + 3) // 4) * 4),),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    probe = FC2SharedLayoutProbe(hidden_size=hidden_size)
    packed_a_t = _to_kernel_tensor(packed_a, cutlass.Float4E2M1FN)
    sfa_ptr = from_dlpack(sfa, assumed_align=16).iterator
    b_w13_t = _to_kernel_tensor(b_w13, cutlass.Float4E2M1FN)

    cases = [(0, 0, 0, 0)]
    for target_tidx, target_src_idx, write_sfa, write_mode in cases:
        probe(
            packed_a_t,
            sfa_ptr,
            b_w13_t,
            Int32(target_tidx),
            Int32(target_src_idx),
            Int32(write_sfa),
            Int32(write_mode),
            current_cuda_stream(),
        )
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
