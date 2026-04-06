#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as cutlass_pipeline
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import fabs_f32, fmax_f32, quantize_block_fp4_fast

_TILE_M = 128
_TILE_N = 128
_PACKED_COLS = _TILE_N // 2
_SCALE_COLS = _TILE_N // 16


def _to_cute_tensor(x: torch.Tensor, dtype, *, assumed_align: int = 16) -> cute.Tensor:
    t = from_dlpack(x, assumed_align=assumed_align)
    t.element_type = dtype
    return t


class CuteNvfp4TileProbe:
    num_threads = 64

    def __init__(self, rows_per_tile: int):
        self.rows_per_tile = rows_per_tile

    def _x_layout(self, rows: int):
        return cute.make_layout((rows, _TILE_N), stride=(_TILE_N, 1))

    def _packed_layout(self):
        return cute.make_layout((_TILE_M, _PACKED_COLS), stride=(_PACKED_COLS, 1))

    def _scale_layout(self):
        return cute.make_layout((_TILE_M, _SCALE_COLS), stride=(_SCALE_COLS, 1))

    def _storage_cls(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, 2],
            "x_payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, _TILE_M * _TILE_N],
                1024,
            ],
            "packed_payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _TILE_M * _PACKED_COLS],
                1024,
            ],
            "scale_payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _TILE_M * _SCALE_COLS],
                1024,
            ],
            "amax_partial": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_threads],
                16,
            ],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Tensor, packed: cute.Tensor, scale: cute.Tensor, tile_amax: cute.Tensor, stream: cuda.CUstream):
        x_layout = self._x_layout(self.rows_per_tile)
        packed_layout = self._packed_layout()
        scale_layout = self._scale_layout()

        x_tma_atom, x_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            x,
            x_layout,
            (self.rows_per_tile, _TILE_N),
            1,
        )
        packed_tma_atom, packed_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            packed,
            packed_layout,
            (_TILE_M, _PACKED_COLS),
        )
        scale_tma_atom, scale_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            scale,
            scale_layout,
            (_TILE_M, _SCALE_COLS),
        )

        Storage = self._storage_cls()
        grid = (1, 1, 1)
        self.kernel(
            x_tma_tensor,
            packed_tma_tensor,
            scale_tma_tensor,
            tile_amax,
            x_tma_atom,
            packed_tma_atom,
            scale_tma_atom,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=Storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x_tma: cute.Tensor,
        packed_tma: cute.Tensor,
        scale_tma: cute.Tensor,
        tile_amax: cute.Tensor,
        x_tma_atom: cute.CopyAtom,
        packed_tma_atom: cute.CopyAtom,
        scale_tma_atom: cute.CopyAtom,
    ):
        tidx = cute.arch.thread_idx()[0]
        Storage = self._storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)

        if tidx == 0:
            cpasync.prefetch_descriptor(x_tma_atom)
            cpasync.prefetch_descriptor(packed_tma_atom)
            cpasync.prefetch_descriptor(scale_tma_atom)

        x_payload = storage.x_payload.get_tensor(cute.make_layout((_TILE_M * _TILE_N,), stride=(1,)))
        sX_full = cute.make_tensor(
            cute.recast_tensor(
                cute.make_tensor(
                    x_payload.iterator,
                    cute.make_layout((_TILE_M * _TILE_N,), stride=(1,)),
                ),
                cutlass.BFloat16,
            ).iterator,
            self._x_layout(_TILE_M),
        )
        sX_load = cute.make_tensor(sX_full.iterator, self._x_layout(self.rows_per_tile))
        sPacked = storage.packed_payload.get_tensor(self._packed_layout())
        sScale = storage.scale_payload.get_tensor(self._scale_layout())
        sAmaxPartial = storage.amax_partial.get_tensor(cute.make_layout((self.num_threads,), stride=(1,)))

        producer_group = cutlass_pipeline.CooperativeGroup(cutlass_pipeline.Agent.Thread, 32)
        consumer_group = cutlass_pipeline.CooperativeGroup(cutlass_pipeline.Agent.Thread, 32)
        load_pipe = cutlass_pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.rows_per_tile * _TILE_N * 2,
            defer_sync=False,
        )
        producer_state = cutlass_pipeline.make_pipeline_state(cutlass_pipeline.PipelineUserType.Producer, 1)
        consumer_state = cutlass_pipeline.make_pipeline_state(cutlass_pipeline.PipelineUserType.Consumer, 1)
        store_pipe = cutlass_pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=cutlass_pipeline.CooperativeGroup(cutlass_pipeline.Agent.Thread, 32),
        )
        cute.arch.sync_threads()

        gX = cute.local_tile(x_tma, (self.rows_per_tile, _TILE_N), (0, 0))
        gPacked = cute.local_tile(packed_tma, (_TILE_M, _PACKED_COLS), (0, 0))
        gScale = cute.local_tile(scale_tma, (_TILE_M, _SCALE_COLS), (0, 0))
        warp_idx = tidx >> cutlass.Int32(5)
        lane_id = tidx & cutlass.Int32(31)

        idx = tidx
        while idx < _TILE_M * _PACKED_COLS:
            sPacked[idx // _PACKED_COLS, idx % _PACKED_COLS] = cutlass.Uint8(0)
            idx += self.num_threads
        idx = tidx
        while idx < _TILE_M * _SCALE_COLS:
            sScale[idx // _SCALE_COLS, idx % _SCALE_COLS] = cutlass.Uint8(0)
            idx += self.num_threads
        cute.arch.sync_threads()

        if warp_idx == 0:
            if lane_id == 0:
                cute.printf("probe: producer acquire")
            load_pipe.producer_acquire(producer_state)
            cute.copy(
                x_tma_atom,
                gX,
                sX_load,
                tma_bar_ptr=load_pipe.producer_get_barrier(producer_state),
            )
            load_pipe.producer_commit(producer_state)
            if lane_id == 0:
                cute.printf("probe: load issued")
        elif warp_idx == 1:
            load_pipe.consumer_wait(consumer_state, load_pipe.consumer_try_wait(consumer_state))
            if lane_id == 0:
                cute.printf("probe: load complete")

            local_amax = cutlass.Float32(0.0)
            block_idx = lane_id
            while block_idx < self.rows_per_tile * _SCALE_COLS:
                row = block_idx // _SCALE_COLS
                sf_block = block_idx % _SCALE_COLS
                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                block_amax = cutlass.Float32(0.0)
                for elem_idx in cutlass.range_constexpr(16):
                    value = cutlass.Float32(sX_full[row, sf_block * 16 + elem_idx])
                    values[elem_idx] = value
                    block_amax = fmax_f32(block_amax, fabs_f32(value))
                local_amax = fmax_f32(local_amax, block_amax)
                packed64, scale_byte = quantize_block_fp4_fast(values, block_amax, cutlass.Float32(1.0))
                for byte_idx in cutlass.range_constexpr(8):
                    sPacked[row, sf_block * 8 + byte_idx] = cutlass.Uint8(
                        (packed64 >> cutlass.Uint64(byte_idx * 8)) & cutlass.Uint64(0xFF)
                    )
                sScale[row, sf_block] = scale_byte
                block_idx += cutlass.Int32(32)

            sAmaxPartial[lane_id] = local_amax
            cute.arch.sync_warp()
            if lane_id == 0:
                tile_max = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(32):
                    tile_max = fmax_f32(tile_max, sAmaxPartial[i])
                tile_amax[cutlass.Int32(0)] = tile_max
                cute.printf(
                    "probe: amax={} packed00={} scale00={}",
                    tile_max,
                    cutlass.Int32(sPacked[cutlass.Int32(0), cutlass.Int32(0)]),
                    cutlass.Int32(sScale[cutlass.Int32(0), cutlass.Int32(0)]),
                )
            load_pipe.consumer_release(consumer_state)

        cute.arch.sync_threads()
        if warp_idx == 0:
            load_pipe.producer_tail(producer_state)
        cute.arch.fence_proxy("async.shared", space="cta")
        if warp_idx == 0:
            if lane_id == 0:
                cute.printf("probe: store packed")
            cute.copy(packed_tma_atom, sPacked, gPacked)
            store_pipe.producer_commit()
            store_pipe.producer_acquire()
            if lane_id == 0:
                cute.printf("probe: store scale")
            cute.copy(scale_tma_atom, sScale, gScale)
            store_pipe.producer_commit()
            store_pipe.producer_acquire()
            if lane_id == 0:
                cute.printf("probe: store tail")
            store_pipe.producer_tail()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    rows = min(max(args.rows, 1), _TILE_M)
    cols = _TILE_N

    x = torch.randn((rows, cols), device=device, dtype=torch.bfloat16)
    packed = torch.empty((_TILE_M, _PACKED_COLS), device=device, dtype=torch.uint8)
    scale = torch.empty((_TILE_M, _SCALE_COLS), device=device, dtype=torch.uint8)
    tile_amax = torch.empty((1,), device=device, dtype=torch.float32)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    probe = CuteNvfp4TileProbe(rows)
    compiled = cute.compile(
        probe,
        _to_cute_tensor(x, cutlass.BFloat16),
        _to_cute_tensor(packed, cutlass.Uint8),
        _to_cute_tensor(scale, cutlass.Uint8),
        _to_cute_tensor(tile_amax, cutlass.Float32),
        stream,
    )

    compiled(
        _to_cute_tensor(x, cutlass.BFloat16),
        _to_cute_tensor(packed, cutlass.Uint8),
        _to_cute_tensor(scale, cutlass.Uint8),
        _to_cute_tensor(tile_amax, cutlass.Float32),
        stream,
    )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        compiled(
            _to_cute_tensor(x, cutlass.BFloat16),
            _to_cute_tensor(packed, cutlass.Uint8),
            _to_cute_tensor(scale, cutlass.Uint8),
            _to_cute_tensor(tile_amax, cutlass.Float32),
            stream,
        )
    torch.cuda.synchronize()
    dt_us = (time.perf_counter() - start) * 1e6 / args.iters
    print(
        f"rows={rows} cols={cols} mean_us={dt_us:.1f} "
        f"amax0={float(tile_amax[0].item()):.6f} "
        f"packed0={int(packed[0, 0].item())} scale0={int(scale[0, 0].item())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
