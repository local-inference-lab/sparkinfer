#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Int64, dsl_user_op

from b12x.attention._cute import copy as cute_copy
from b12x.attention._cute import pipeline as cute_pipeline
from b12x.cute.intrinsics import get_ptr_as_int64, shared_ptr_to_u32


_ROWS = 64
_PLANE_COLS = 128
_OUT_WORDS = _ROWS * _PLANE_COLS // 4


def _source_byte(idx: int) -> int:
    x = (idx * 1103515245 + 12345) & 0xFFFFFFFF
    x ^= (x >> 11) & 0xFFFFFFFF
    x ^= (x << 7) & 0xFFFFFFFF
    x ^= (x >> 13) & 0xFFFFFFFF
    return (x >> 16) & 0xFF


def _pack_words(bytes_: torch.Tensor) -> torch.Tensor:
    raw = bytes_.to(torch.int64).reshape(-1)
    return raw[0::4] | (raw[1::4] << 8) | (raw[2::4] << 16) | (raw[3::4] << 24)


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


@dsl_user_op
def _cp_async_bulk_tensor_2d(
    dst_smem_addr: Int32,
    tensor_map_ptr: Int64,
    coord0: Int32,
    coord1: Int32,
    mbar_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(dst_smem_addr).ir_value(loc=loc, ip=ip),
            Int64(tensor_map_ptr).ir_value(loc=loc, ip=ip),
            Int32(coord0).ir_value(loc=loc, ip=ip),
            Int32(coord1).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[$0], [$1, {$2, $3}], [$4];",
        "r,l,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class CutePlaneProbe:
    def __init__(self, *, transport: str, swizzle: str, use_internal_type: bool):
        if transport not in {"u8", "u16"}:
            raise ValueError("transport must be one of {u8,u16}")
        self.transport = transport
        self.swizzle = swizzle
        self.use_internal_type = use_internal_type
        self.transport_dtype = cutlass.Uint8 if transport == "u8" else cutlass.Uint16
        self.plane_head_dim = 128 if transport == "u8" else 64
        self.plane_bytes = _ROWS * _PLANE_COLS
        self.plane_total_elems = _ROWS * self.plane_head_dim
        self.internal_type = self.transport_dtype if use_internal_type else None

    @cute.jit
    def __call__(
        self, mSrc: cute.Tensor, mOutWords: cute.Tensor, stream: cuda.CUstream
    ):
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mSrc,
            self._get_plane_layout(),
            (_ROWS, self.plane_head_dim),
            1,
            internal_type=self.internal_type,
        )
        self.kernel(tma_tensor, mOutWords, tma_atom).launch(
            grid=(1, 1, 1),
            block=[32, 1, 1],
            stream=stream,
        )

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, 2],
            "payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, int(_ROWS * _PLANE_COLS)],
                1024,
            ],
        }
        return cute.struct(SharedStorage)

    def _get_plane_layout(self):
        if self.swizzle == "none":
            return cute.make_layout(
                (_ROWS, self.plane_head_dim),
                stride=(self.plane_head_dim, 1),
            )
        mbase, bbits, sshift = [int(part) for part in self.swizzle.split(",")]
        return cute.make_composed_layout(
            cute.make_swizzle(mbase, bbits, sshift),
            0,
            cute.make_layout(
                (_ROWS, self.plane_head_dim),
                stride=(self.plane_head_dim, 1),
            ),
        )

    @cute.kernel
    def kernel(
        self,
        mSrcTma: cute.Tensor,
        mOutWords: cute.Tensor,
        tma_atom: cute.CopyAtom,
    ):
        lane, _, _ = cute.arch.thread_idx()
        tidx = lane

        SharedStorage = self._get_shared_storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        if lane == 0:
            cpasync.prefetch_descriptor(tma_atom)

        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((_ROWS * _PLANE_COLS,), stride=(1,))
        )
        mbar_ptr = storage.mbar_ptr.data_ptr()

        plane_layout = self._get_plane_layout()
        sPlane = cute.make_tensor(
            cute.recast_tensor(
                cute.make_tensor(
                    payload_u8.iterator + cutlass.Int32(0),
                    cute.make_layout((self.plane_bytes,), stride=(1,)),
                ),
                self.transport_dtype,
            ).iterator,
            plane_layout,
        )
        gPlane = cute.local_tile(mSrcTma, (_ROWS, self.plane_head_dim), (0, 0))

        consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 1
        )
        producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipe = cute_pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.plane_bytes,
            defer_sync=False,
        )
        producer_state = cute_pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state = cute_pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )

        load, _, _ = cute_copy.tma_get_copy_fn(
            tma_atom, 0, cute.make_layout(1), gPlane, sPlane
        )
        load = cute_copy.tma_producer_copy_fn(load, pipe)

        if lane == 0:
            pipe.producer_acquire(producer_state)
            load(src_idx=0, producer_state=producer_state)

        pipe.consumer_wait(consumer_state, pipe.consumer_try_wait(consumer_state))
        out_words_u32 = cute.flatten(mOutWords)
        payload_u32 = cute.flatten(
            cute.recast_tensor(
                cute.make_tensor(
                    payload_u8.iterator,
                    cute.make_layout((_ROWS * _PLANE_COLS,), stride=(1,)),
                ),
                cutlass.Uint32,
            )
        )
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(_OUT_WORDS, 32)):
            word_idx = tidx + idx_iter * 32
            if word_idx < _OUT_WORDS:
                out_words_u32[word_idx] = payload_u32[word_idx]
        pipe.consumer_release(consumer_state)
        if lane == 0:
            pipe.producer_tail(producer_state)


class RawPtxPlaneProbe:
    def __init__(self, *, swizzle: str):
        self.swizzle = swizzle
        self.plane_bytes = _ROWS * _PLANE_COLS

    @cute.jit
    def __call__(
        self, mDescWords: cute.Tensor, mOutWords: cute.Tensor, stream: cuda.CUstream
    ):
        self.kernel(mDescWords, mOutWords).launch(
            grid=(1, 1, 1),
            block=[32, 1, 1],
            stream=stream,
        )

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, 1],
            "payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, int(_ROWS * _PLANE_COLS)],
                1024,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.kernel
    def kernel(
        self,
        mDescWords: cute.Tensor,
        mOutWords: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        SharedStorage = self._get_shared_storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_ptr = storage.mbar_ptr.data_ptr()
        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((_ROWS * _PLANE_COLS,), stride=(1,))
        )

        if lane == 0:
            cute.arch.mbarrier_init(mbar_ptr, Int32(1))
        cute.arch.sync_threads()

        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, self.plane_bytes)
            _cp_async_bulk_tensor_2d(
                shared_ptr_to_u32(payload_u8.iterator),
                get_ptr_as_int64(mDescWords, Int32(0)),
                Int32(0),
                Int32(0),
                shared_ptr_to_u32(mbar_ptr),
            )
        cute.arch.sync_threads()
        cute.arch.mbarrier_wait(mbar_ptr, phase=0)
        cute.arch.sync_threads()

        out_words_u32 = cute.flatten(mOutWords)
        payload_u32 = cute.flatten(
            cute.recast_tensor(
                cute.make_tensor(
                    payload_u8.iterator,
                    cute.make_layout((_ROWS * _PLANE_COLS,), stride=(1,)),
                ),
                cutlass.Uint32,
            )
        )
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(_OUT_WORDS, 32)):
            word_idx = lane + idx_iter * 32
            if word_idx < _OUT_WORDS:
                out_words_u32[word_idx] = payload_u32[word_idx]


def _build_src_bytes() -> torch.Tensor:
    host = torch.empty((_ROWS, _PLANE_COLS), dtype=torch.uint8)
    flat = host.view(-1)
    for idx in range(flat.numel()):
        flat[idx] = _source_byte(idx)
    return host.cuda()


def _expected_words(swizzle: str) -> torch.Tensor:
    src = _build_src_bytes().cpu()
    expected = torch.empty((_ROWS, _PLANE_COLS), dtype=torch.uint8)
    if swizzle == "none":
        expected.copy_(src)
    else:
        for row in range(_ROWS):
            for chunk in range(8):
                src_chunk = chunk ^ (row % 8)
                expected[row, chunk * 16 : (chunk + 1) * 16] = src[
                    row, src_chunk * 16 : (src_chunk + 1) * 16
                ]
    return _pack_words(expected.reshape(-1)).cpu()


def _build_tensor_map_words(src: torch.Tensor, *, swizzle: str) -> torch.Tensor:
    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    swizzle_enum = (
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
        if swizzle == "none"
        else cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    )
    result, tensor_map = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        src.data_ptr(),
        [U64(_PLANE_COLS), U64(_ROWS)],
        [U64(_PLANE_COLS)],
        [U32(_PLANE_COLS), U32(_ROWS)],
        [U32(1), U32(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_enum,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")
    host_words = [int(word) for word in tensor_map.opaque]
    return torch.tensor(host_words, dtype=torch.uint64, device="cuda")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal CuTe-only FP8 plane TMA landing probe."
    )
    parser.add_argument("--issue", choices=("cute", "ptx"), default="ptx")
    parser.add_argument("--transport", choices=("u8", "u16"), default="u8")
    parser.add_argument("--swizzle", default="3,4,3")
    parser.add_argument("--no-internal-type", action="store_true")
    args = parser.parse_args()

    src_u8 = _build_src_bytes()
    out_words = torch.zeros((_OUT_WORDS,), dtype=torch.int32, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if args.issue == "cute":
        if args.transport == "u8":
            src = src_u8
        else:
            src = src_u8.view(torch.uint16)
        probe = CutePlaneProbe(
            transport=args.transport,
            swizzle=args.swizzle,
            use_internal_type=not args.no_internal_type,
        )
        src_dtype = cutlass.Uint8 if args.transport == "u8" else cutlass.Uint16
        compiled = cute.compile(
            probe,
            _to_cute_tensor(src, src_dtype),
            _to_cute_tensor(out_words, cutlass.Int32),
            stream,
        )
        compiled(
            _to_cute_tensor(src, src_dtype),
            _to_cute_tensor(out_words, cutlass.Int32),
            stream,
        )
    else:
        if args.transport != "u8":
            raise ValueError("raw PTX probe currently supports only --transport u8")
        desc_words = _build_tensor_map_words(src_u8, swizzle=args.swizzle)
        probe = RawPtxPlaneProbe(swizzle=args.swizzle)
        compiled = cute.compile(
            probe,
            _to_cute_tensor(desc_words, cutlass.Uint64),
            _to_cute_tensor(out_words, cutlass.Int32),
            stream,
        )
        compiled(
            _to_cute_tensor(desc_words, cutlass.Uint64),
            _to_cute_tensor(out_words, cutlass.Int32),
            stream,
        )
    torch.cuda.synchronize()

    got = out_words.cpu().to(torch.int64) & 0xFFFFFFFF
    exp = _expected_words(args.swizzle).to(torch.int64) & 0xFFFFFFFF
    mismatches = (got != exp).nonzero(as_tuple=False).reshape(-1)
    first = int(mismatches[0].item()) if mismatches.numel() else -1
    report = {
        "issue": args.issue,
        "transport": args.transport,
        "swizzle": args.swizzle,
        "use_internal_type": not args.no_internal_type,
        "mismatch_count": int(mismatches.numel()),
        "first_mismatch": (
            {
                "word": first,
                "got": f"0x{int(got[first].item()) & 0xFFFFFFFF:08x}",
                "expected": f"0x{int(exp[first].item()) & 0xFFFFFFFF:08x}",
            }
            if first >= 0
            else None
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
