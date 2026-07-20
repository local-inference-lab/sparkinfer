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
from cutlass.cute.runtime import from_dlpack

import sparkinfer.attention._shared.cute.pipeline as cute_pipeline
from sparkinfer.attention.paged.forward_paged import (
    _async_copy_q_tile_permuted_128b_fp8_decode_impl,
    _issue_paged_kv_tma_copy_2planes_fp8_raw_impl,
)

_ROWS = 64
_HEAD_DIM = 256
_PLANE_COLS = 128
_PLANE_BYTES = _ROWS * _PLANE_COLS
_TILE_BYTES = _ROWS * _HEAD_DIM
_OUT_WORDS = _TILE_BYTES // 4

_Q_BYTES = 16 * 256 * 2
_K_BYTES = _TILE_BYTES
_V_BYTES = _TILE_BYTES
_PAYLOAD_BYTES = _Q_BYTES + _K_BYTES + _V_BYTES


def _source_byte(idx: int, *, salt: int) -> int:
    x = (idx * 1103515245 + 12345 + salt) & 0xFFFFFFFF
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


class RawPtxPagedFp8DecodeTransportProbe:
    def __init__(
        self,
        *,
        issue_k: bool,
        issue_v: bool,
        k_dst_offset: int,
        v_dst_offset: int,
        warps_kv: int,
        wait_mode: str,
        with_qcopy: bool,
    ):
        self.issue_k = issue_k
        self.issue_v = issue_v
        self.k_dst_offset = k_dst_offset
        self.v_dst_offset = v_dst_offset
        self.warps_kv = warps_kv
        self.wait_mode = wait_mode
        self.with_qcopy = with_qcopy

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr_K": cute.struct.MemRange[cutlass.Int64, 2],
            "mbar_ptr_V": cute.struct.MemRange[cutlass.Int64, 2],
            "payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _PAYLOAD_BYTES],
                1024,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mPageTable: cute.Tensor,
        mKDescPtrs: cute.Tensor,
        mVDescPtrs: cute.Tensor,
        mOutWords: cute.Tensor,
        request_idx: Int32,
        kv_head_idx: Int32,
        page_index: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mQ,
            mPageTable,
            mKDescPtrs,
            mVDescPtrs,
            mOutWords,
            request_idx,
            kv_head_idx,
            page_index,
        ).launch(
            grid=(1, 1, 1),
            block=[32, 1, self.warps_kv],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mPageTable: cute.Tensor,
        mKDescPtrs: cute.Tensor,
        mVDescPtrs: cute.Tensor,
        mOutWords: cute.Tensor,
        request_idx: Int32,
        kv_head_idx: Int32,
        page_index: Int32,
    ):
        lane, _, warp_kv_idx = cute.arch.thread_idx()
        tidx = lane + 32 * warp_kv_idx
        SharedStorage = self._get_shared_storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_ptr_K = storage.mbar_ptr_K.data_ptr()
        mbar_ptr_V = storage.mbar_ptr_V.data_ptr()
        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((_PAYLOAD_BYTES,), stride=(1,))
        )
        sQ = cute.make_tensor(
            cute.recast_ptr(payload_u8.iterator.align(16), dtype=cutlass.BFloat16),
            cute.make_layout((16, 256), stride=(256, 1)),
        )
        if tidx == Int32(0):
            cute.arch.mbarrier_init(mbar_ptr_K, Int32(1))
            cute.arch.mbarrier_init(mbar_ptr_V, Int32(1))
        cute.arch.sync_threads()

        if const_expr(self.with_qcopy):
            sQBytes = cute.flatten(cute.recast_tensor(sQ, cutlass.Uint8))
            mQBytes = cute.flatten(cute.recast_tensor(mQ, cutlass.Uint8))
            if warp_kv_idx == Int32(0):
                _async_copy_q_tile_permuted_128b_fp8_decode_impl(
                    mQBytes,
                    request_idx,
                    Int32(0),
                    Int32(2),
                    kv_head_idx,
                    Int32(2),
                    Int32(mQ.shape[1]),
                    Int32(512),
                    sQBytes,
                    lane,
                    Int32(32),
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        producer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
        consumer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
        tile_base = page_index * Int32(_ROWS)

        if warp_kv_idx == Int32(0):
            if const_expr(self.issue_k):
                _issue_paged_kv_tma_copy_2planes_fp8_raw_impl(
                    cute.flatten(mKDescPtrs),
                    kv_head_idx,
                    Int32(_PLANE_COLS),
                    payload_u8,
                    Int32(self.k_dst_offset),
                    Int32(_PLANE_BYTES),
                    producer_state,
                    mbar_ptr_K,
                    Int32(_TILE_BYTES),
                    mPageTable,
                    request_idx,
                    tile_base,
                    Int32(_ROWS),
                )
            if const_expr(self.issue_v):
                _issue_paged_kv_tma_copy_2planes_fp8_raw_impl(
                    cute.flatten(mVDescPtrs),
                    kv_head_idx,
                    Int32(_PLANE_COLS),
                    payload_u8,
                    Int32(self.v_dst_offset),
                    Int32(_PLANE_BYTES),
                    producer_state,
                    mbar_ptr_V,
                    Int32(_TILE_BYTES),
                    mPageTable,
                    request_idx,
                    tile_base,
                    Int32(_ROWS),
                )
            producer_state.advance()
        cute.arch.sync_threads()

        if const_expr(self.wait_mode != "none"):
            if const_expr(self.wait_mode == "all") or warp_kv_idx == Int32(0):
                if const_expr(self.issue_k):
                    cute.arch.mbarrier_wait(
                        mbar_ptr_K + consumer_state.index, phase=consumer_state.phase
                    )
                if const_expr(self.issue_v):
                    cute.arch.mbarrier_wait(
                        mbar_ptr_V + consumer_state.index, phase=consumer_state.phase
                    )
        cute.arch.sync_threads()

        payload_u32 = cute.flatten(
            cute.recast_tensor(
                cute.make_tensor(
                    payload_u8.iterator,
                    cute.make_layout((_PAYLOAD_BYTES,), stride=(1,)),
                ),
                cutlass.Uint32,
            )
        )
        out_words = cute.flatten(mOutWords)
        word_offset = tidx
        if const_expr(self.issue_k):
            k_src_word = Int32(self.k_dst_offset // 4)
            while word_offset < _OUT_WORDS:
                out_words[word_offset] = payload_u32[k_src_word + word_offset]
                word_offset += 128
        if const_expr(self.issue_v):
            v_src_word = Int32(self.v_dst_offset // 4)
            dst_base = Int32(_OUT_WORDS)
            word_offset = tidx
            while word_offset < _OUT_WORDS:
                out_words[dst_base + word_offset] = payload_u32[
                    v_src_word + word_offset
                ]
                word_offset += 128


def _build_src_bytes(pages: int, *, salt: int) -> torch.Tensor:
    host = torch.empty((pages * _ROWS, _HEAD_DIM), dtype=torch.uint8)
    flat = host.view(-1)
    for idx in range(flat.numel()):
        flat[idx] = _source_byte(idx, salt=salt)
    return host.cuda()


def _build_desc_words(src_u8: torch.Tensor, *, swizzle: str) -> torch.Tensor:
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
    return torch.tensor(
        [int(word) for word in tensor_map.opaque], dtype=torch.uint64, device="cuda"
    )


def _descriptor_row_ptrs(desc_words: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [int(desc_words.data_ptr())], dtype=torch.int64, device=desc_words.device
    )


def _expected_words(
    src_u8: torch.Tensor, *, page_index: int, swizzle: str
) -> torch.Tensor:
    if src_u8.ndim == 3:
        src_u8 = src_u8.reshape(-1, src_u8.shape[-1])
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


def _offset_value(name: str) -> int:
    named = {
        "0": 0,
        "q": _Q_BYTES,
        "qk": _Q_BYTES + _K_BYTES,
        "q_midk": _Q_BYTES + (_K_BYTES // 2),
    }
    if name in named:
        return named[name]
    return int(name)


def _compare(got: torch.Tensor, exp: torch.Tensor) -> dict:
    got64 = got.cpu().to(torch.int64) & 0xFFFFFFFF
    exp64 = exp.to(torch.int64) & 0xFFFFFFFF
    mismatches = (got64 != exp64).nonzero(as_tuple=False).reshape(-1)
    first = int(mismatches[0].item()) if mismatches.numel() else -1
    return {
        "mismatch_count": int(mismatches.numel()),
        "first_mismatch": (
            {
                "word": first,
                "got": f"0x{int(got64[first].item()) & 0xFFFFFFFF:08x}",
                "expected": f"0x{int(exp64[first].item()) & 0xFFFFFFFF:08x}",
            }
            if first >= 0
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="128-thread raw PTX paged FP8 decode transport probe."
    )
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--swizzle", choices=("none", "128b"), default="128b")
    parser.add_argument("--case", choices=("k", "v", "kv"), default="kv")
    parser.add_argument("--k-dst-offset", default="q")
    parser.add_argument("--v-dst-offset", default="qk")
    parser.add_argument("--warps-kv", type=int, default=4)
    parser.add_argument(
        "--wait-mode", choices=("warp0", "all", "none"), default="warp0"
    )
    parser.add_argument("--with-qcopy", action="store_true")
    parser.add_argument("--request-idx", type=int, default=0)
    parser.add_argument("--kv-head-idx", type=int, default=0)
    args = parser.parse_args()

    issue_k = args.case in ("k", "kv")
    issue_v = args.case in ("v", "kv")
    k_dst_offset = _offset_value(args.k_dst_offset)
    v_dst_offset = _offset_value(args.v_dst_offset)

    src_k_u8 = (
        _build_src_bytes(args.pages, salt=0)
        .view(args.pages, _ROWS, 1, _HEAD_DIM)
        .repeat(1, 1, 8, 1)
    )
    src_v_u8 = (
        _build_src_bytes(args.pages, salt=17)
        .view(args.pages, _ROWS, 1, _HEAD_DIM)
        .repeat(1, 1, 8, 1)
    )
    src_k = src_k_u8.to(torch.float8_e4m3fn)
    src_v = src_v_u8.to(torch.float8_e4m3fn)
    page_table = (
        torch.arange(args.pages, dtype=torch.int32, device="cuda")
        .view(1, -1)
        .repeat(8, 1)
    )
    from sparkinfer.attention.paged._forward import (
        _descriptor_row_ptrs as _api_descriptor_row_ptrs,
    )
    from sparkinfer.attention.paged._forward import _encode_fp8_plane_tma_descriptors

    desc_words_k = _encode_fp8_plane_tma_descriptors(src_k, plane_cols=_PLANE_COLS)
    desc_words_v = _encode_fp8_plane_tma_descriptors(src_v, plane_cols=_PLANE_COLS)
    desc_k = _api_descriptor_row_ptrs(desc_words_k)
    desc_v = _api_descriptor_row_ptrs(desc_words_v)
    out_words = torch.zeros((2 * _OUT_WORDS,), dtype=torch.int32, device="cuda")

    probe = RawPtxPagedFp8DecodeTransportProbe(
        issue_k=issue_k,
        issue_v=issue_v,
        k_dst_offset=k_dst_offset,
        v_dst_offset=v_dst_offset,
        warps_kv=args.warps_kv,
        wait_mode=args.wait_mode,
        with_qcopy=args.with_qcopy,
    )
    q = torch.randn((8, 16, 256), dtype=torch.bfloat16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        probe,
        _to_cute_tensor(q, cutlass.BFloat16),
        _to_cute_tensor(page_table, cutlass.Int32),
        _to_cute_tensor(desc_k, cutlass.Int64),
        _to_cute_tensor(desc_v, cutlass.Int64),
        _to_cute_tensor(out_words, cutlass.Int32),
        Int32(args.request_idx),
        Int32(args.kv_head_idx),
        Int32(args.page_index),
        stream,
    )
    compiled(
        _to_cute_tensor(q, cutlass.BFloat16),
        _to_cute_tensor(page_table, cutlass.Int32),
        _to_cute_tensor(desc_k, cutlass.Int64),
        _to_cute_tensor(desc_v, cutlass.Int64),
        _to_cute_tensor(out_words, cutlass.Int32),
        Int32(args.request_idx),
        Int32(args.kv_head_idx),
        Int32(args.page_index),
        stream,
    )
    torch.cuda.synchronize()

    report = {
        "case": args.case,
        "swizzle": args.swizzle,
        "page_index": args.page_index,
        "k_dst_offset": k_dst_offset,
        "v_dst_offset": v_dst_offset,
        "warps_kv": args.warps_kv,
        "wait_mode": args.wait_mode,
        "with_qcopy": args.with_qcopy,
        "request_idx": args.request_idx,
        "kv_head_idx": args.kv_head_idx,
    }
    got = out_words.cpu()
    src_k_head = src_k.view(torch.uint8)[:, :, args.kv_head_idx, :].contiguous()
    src_v_head = src_v.view(torch.uint8)[:, :, args.kv_head_idx, :].contiguous()
    if issue_k:
        report["k"] = _compare(
            got[:_OUT_WORDS],
            _expected_words(
                src_k_head, page_index=args.page_index, swizzle=args.swizzle
            ),
        )
    if issue_v:
        report["v"] = _compare(
            got[_OUT_WORDS:],
            _expected_words(
                src_v_head, page_index=args.page_index, swizzle=args.swizzle
            ),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
