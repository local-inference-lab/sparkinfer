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
from cutlass import Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Int64, dsl_user_op

from sparkinfer.attention._cute import pipeline as cute_pipeline
from sparkinfer.cute.intrinsics import shared_ptr_to_u32

_ROWS = 64
_HEAD_DIM = 256
_PLANE_COLS = 128
_OUT_WORDS = _ROWS * _HEAD_DIM // 4


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


class RawPtxPagedFp8Probe:
    def __init__(self, *, mode: str, stages: int):
        self.mode = mode
        self.stages = stages

    @cute.jit
    def __call__(
        self,
        mDescPtrs: cute.Tensor,
        mOutWords: cute.Tensor,
        page_row_base: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mDescPtrs, mOutWords, page_row_base).launch(
            grid=(1, 1, 1),
            block=[32, 1, 1],
            stream=stream,
        )

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.stages],
            "payload": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, int(_ROWS * _HEAD_DIM * self.stages)
                ],
                1024,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.kernel
    def kernel(
        self,
        mDescPtrs: cute.Tensor,
        mOutWords: cute.Tensor,
        page_row_base: Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        SharedStorage = self._get_shared_storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_ptr = storage.mbar_ptr.data_ptr()
        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((_ROWS * _HEAD_DIM * self.stages,), stride=(1,))
        )

        if const_expr(self.mode == "pipeline"):
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
                tx_count=_ROWS * _HEAD_DIM,
                defer_sync=False,
            )
            producer_state = cute_pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, 1
            )
            consumer_state = cute_pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, 1
            )
            if lane == 0:
                desc_ptr = Int64(cute.flatten(mDescPtrs)[0])
                pipe.sync_object_empty.wait(producer_state.index, producer_state.phase)
                pipe.sync_object_full.arrive_and_expect_tx(
                    producer_state.index,
                    pipe.sync_object_full.tx_count,
                )
                _cp_async_bulk_tensor_2d(
                    shared_ptr_to_u32(payload_u8.iterator + 0),
                    desc_ptr,
                    Int32(0),
                    page_row_base,
                    shared_ptr_to_u32(pipe.producer_get_barrier(producer_state)),
                )
                _cp_async_bulk_tensor_2d(
                    shared_ptr_to_u32(payload_u8.iterator + _ROWS * _PLANE_COLS),
                    desc_ptr,
                    Int32(_PLANE_COLS),
                    page_row_base,
                    shared_ptr_to_u32(pipe.producer_get_barrier(producer_state)),
                )
            pipe.consumer_wait(consumer_state, pipe.consumer_try_wait(consumer_state))
            cute.arch.sync_threads()
        elif const_expr(self.mode == "state"):
            producer_state = cute_pipeline.PipelineStateSimple(self.stages, Int32(0))
            consumer_state = cute_pipeline.PipelineStateSimple(self.stages, Int32(0))
            if lane < self.stages:
                cute.arch.mbarrier_init(mbar_ptr + lane, Int32(1))
            cute.arch.sync_threads()
            if lane == 0:
                desc_ptr = Int64(cute.flatten(mDescPtrs)[0])
                stage_offset = producer_state.index * Int32(_ROWS * _HEAD_DIM)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_ptr + producer_state.index,
                    _ROWS * _HEAD_DIM,
                )
                _cp_async_bulk_tensor_2d(
                    shared_ptr_to_u32(payload_u8.iterator + stage_offset + Int32(0)),
                    desc_ptr,
                    Int32(0),
                    page_row_base,
                    shared_ptr_to_u32(mbar_ptr + producer_state.index),
                )
                _cp_async_bulk_tensor_2d(
                    shared_ptr_to_u32(
                        payload_u8.iterator + stage_offset + Int32(_ROWS * _PLANE_COLS)
                    ),
                    desc_ptr,
                    Int32(_PLANE_COLS),
                    page_row_base,
                    shared_ptr_to_u32(mbar_ptr + producer_state.index),
                )
            cute.arch.sync_threads()
            cute.arch.mbarrier_wait(
                mbar_ptr + consumer_state.index,
                phase=consumer_state.phase,
            )
            cute.arch.sync_threads()
        else:
            if lane == 0:
                cute.arch.mbarrier_init(mbar_ptr, Int32(1))
            cute.arch.sync_threads()
            if lane == 0:
                desc_ptr = Int64(cute.flatten(mDescPtrs)[0])
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, _ROWS * _HEAD_DIM)
                _cp_async_bulk_tensor_2d(
                    shared_ptr_to_u32(payload_u8.iterator + 0),
                    desc_ptr,
                    Int32(0),
                    page_row_base,
                    shared_ptr_to_u32(mbar_ptr),
                )
                _cp_async_bulk_tensor_2d(
                    shared_ptr_to_u32(payload_u8.iterator + _ROWS * _PLANE_COLS),
                    desc_ptr,
                    Int32(_PLANE_COLS),
                    page_row_base,
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
                    cute.make_layout((_ROWS * _HEAD_DIM,), stride=(1,)),
                ),
                cutlass.Uint32,
            )
        )
        payload_word_offset = Int32(0)
        if const_expr(self.mode == "state"):
            payload_word_offset = consumer_state.index * Int32(_OUT_WORDS)
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(_OUT_WORDS, 32)):
            word_idx = lane + idx_iter * 32
            if word_idx < _OUT_WORDS:
                out_words_u32[word_idx] = payload_u32[payload_word_offset + word_idx]


def _build_src_bytes(pages: int) -> torch.Tensor:
    host = torch.empty((pages * _ROWS, _HEAD_DIM), dtype=torch.uint8)
    flat = host.view(-1)
    for idx in range(flat.numel()):
        flat[idx] = _source_byte(idx)
    return host.cuda()


def _build_desc_ptrs(src_u8: torch.Tensor, *, swizzle: str) -> torch.Tensor:
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
        src_u8.data_ptr(),
        [U64(_HEAD_DIM), U64(int(src_u8.shape[0]))],
        [U64(_HEAD_DIM)],
        [U32(_PLANE_COLS), U32(_ROWS)],
        [U32(1), U32(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_enum,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")
    desc_words = torch.tensor(
        [int(word) for word in tensor_map.opaque], dtype=torch.uint64, device="cuda"
    )
    return torch.tensor([int(desc_words.data_ptr())], dtype=torch.int64, device="cuda")


def _expected_words(
    src_u8: torch.Tensor, *, page_index: int, swizzle: str
) -> torch.Tensor:
    page = src_u8[page_index * _ROWS : (page_index + 1) * _ROWS].cpu()
    slabs = []
    for plane in range(2):
        slab = torch.empty((_ROWS, _PLANE_COLS), dtype=torch.uint8)
        col_base = plane * _PLANE_COLS
        for row in range(_ROWS):
            for chunk in range(8):
                src_chunk = chunk if swizzle == "none" else (chunk ^ (row % 8))
                slab[row, chunk * 16 : (chunk + 1) * 16] = page[
                    row, col_base + src_chunk * 16 : col_base + (src_chunk + 1) * 16
                ]
        slabs.append(slab.reshape(-1))
    return _pack_words(torch.cat(slabs))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw PTX paged FP8 row-offset TMA probe."
    )
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--page-index", type=int, default=2)
    parser.add_argument("--mode", choices=("bare", "pipeline", "state"), default="bare")
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--swizzle", choices=("none", "128b"), default="128b")
    args = parser.parse_args()

    src_u8 = _build_src_bytes(args.pages)
    desc_ptrs = _build_desc_ptrs(src_u8, swizzle=args.swizzle)
    out_words = torch.zeros((_OUT_WORDS,), dtype=torch.int32, device="cuda")
    probe = RawPtxPagedFp8Probe(mode=args.mode, stages=args.stages)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        probe,
        _to_cute_tensor(desc_ptrs, cutlass.Int64),
        _to_cute_tensor(out_words, cutlass.Int32),
        Int32(args.page_index * _ROWS),
        stream,
    )
    compiled(
        _to_cute_tensor(desc_ptrs, cutlass.Int64),
        _to_cute_tensor(out_words, cutlass.Int32),
        Int32(args.page_index * _ROWS),
        stream,
    )
    torch.cuda.synchronize()

    got = out_words.cpu().to(torch.int64) & 0xFFFFFFFF
    exp = (
        _expected_words(src_u8, page_index=args.page_index, swizzle=args.swizzle).to(
            torch.int64
        )
        & 0xFFFFFFFF
    )
    mismatches = (got != exp).nonzero(as_tuple=False).reshape(-1)
    first = int(mismatches[0].item()) if mismatches.numel() else -1
    report = {
        "pages": args.pages,
        "page_index": args.page_index,
        "mode": args.mode,
        "stages": args.stages,
        "swizzle": args.swizzle,
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
