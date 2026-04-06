#!/usr/bin/env python3
"""Benchmark dynamic superfused Gemma/Qwen block paths against graph-safe baselines.

Run under torchrun, for example:

  torchrun --nproc-per-node=2 benchmarks/benchmark_gemma_dynamic_superfused_paths.py \
      --batch-sizes 1 2 4 8
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from benchmarks.benchmark_gemma_moe_block_paths import (
    _gemma_rmsnorm_after_allreduce,
    _load_post_attention_layernorm_weight,
    _pack_shared_expert,
    _pack_sparse_experts_per_expert,
)
from benchmarks.benchmark_moe import (
    BATCH_SIZE_PROFILES,
    MODEL_PATH,
    ModelSpec,
    bench_events,
    load_expert_weights,
    load_gate_weight,
    load_shared_expert_weights,
    load_shared_gate_weight,
    make_input_activations,
    require_sm120,
)
from b12x.distributed.pcie_oneshot import PCIeOneshotAllReduce
from b12x.attention import copy_utils, pipeline
from b12x.integration.tp_moe import (
    _append_expert_bank,
    _append_shared_expert_routing,
    _b12x_gemma_moe_block_fp4_dynamic_superfused,
    _shared_expert_gate_weights,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    b12x_route_experts_fast,
    clear_tp_moe_caches,
)
from b12x.cute.fp4 import fabs_f32, fmax_f32, quantize_block_fp4_fast
from b12x.moe.fused.reference import compare_to_reference


def _rank0_print(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=_world_size(),
        tp_rank=_rank(),
    )


def _fmt_us(times_ms: list[float]) -> str:
    median_us = statistics.median(times_ms) * 1000.0
    min_us = min(times_ms) * 1000.0
    return f"{median_us:8.1f} us (min {min_us:.1f})"


def _gather_rank_medians(times_ms: list[float]) -> list[float]:
    local = statistics.median(times_ms)
    gathered: list[float] = [0.0 for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered


def _bench_graph_replay(
    fn,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> list[float]:
    # Warm eager launch state so compile/cache work does not leak into capture
    # or replay timing.
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    return bench_events(replay, warmup=warmup, iters=iters)


_TILE_M = 128
_TILE_N = 128
_TILE_PACKED_COLS = _TILE_N // 2
_TILE_SCALE_COLS = _TILE_N // 16
_CUTE_TILE_QUANT_CACHE: dict[tuple[int, int, int], object] = {}


def _to_cute_tensor(x: torch.Tensor, dtype, *, assumed_align: int = 16) -> cute.Tensor:
    t = from_dlpack(x, assumed_align=assumed_align)
    t.element_type = dtype
    return t


class _CuteTileAmaxPackProbe:
    num_threads = 128

    def __init__(self, load_rows: int):
        self.load_rows = load_rows

    def _x_smem_layout(self, rows: int):
        return cute.make_layout((rows, _TILE_N), stride=(_TILE_N, 1))

    def _packed_smem_layout(self):
        return cute.make_layout((_TILE_M, _TILE_PACKED_COLS), stride=(_TILE_PACKED_COLS, 1))

    def _scale_smem_layout(self):
        return cute.make_layout((_TILE_M, _TILE_SCALE_COLS), stride=(_TILE_SCALE_COLS, 1))

    def _storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, 2],
            "x_payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, _TILE_M * _TILE_N],
                1024,
            ],
            "packed_payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _TILE_M * _TILE_PACKED_COLS],
                1024,
            ],
            "scale_payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _TILE_M * _TILE_SCALE_COLS],
                1024,
            ],
            "amax_partial": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_threads],
                16,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        packed_tiles: cute.Tensor,
        scale_tiles: cute.Tensor,
        tile_amax: cute.Tensor,
        stream: cuda.CUstream,
    ):
        x_smem_layout = self._x_smem_layout(self.load_rows)
        packed_smem_layout = self._packed_smem_layout()
        scale_smem_layout = self._scale_smem_layout()
        x_tma_atom, x_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            x,
            x_smem_layout,
            (self.load_rows, _TILE_N),
            1,
        )
        packed_tma_atom, packed_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            packed_tiles,
            packed_smem_layout,
            (_TILE_M, _TILE_PACKED_COLS),
        )
        scale_tma_atom, scale_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            scale_tiles,
            scale_smem_layout,
            (_TILE_M, _TILE_SCALE_COLS),
        )
        Storage = self._storage_cls()
        grid = (
            packed_tiles.shape[0] // _TILE_M,
            packed_tiles.shape[1] // _TILE_PACKED_COLS,
            1,
        )
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
        pid_m = cute.arch.block_idx()[0]
        pid_n = cute.arch.block_idx()[1]

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
                cute.make_tensor(x_payload.iterator, cute.make_layout((_TILE_M * _TILE_N,), stride=(1,))),
                cutlass.BFloat16,
            ).iterator,
            self._x_smem_layout(_TILE_M),
        )
        sX_load = cute.make_tensor(sX_full.iterator, self._x_smem_layout(self.load_rows))
        sPacked = storage.packed_payload.get_tensor(self._packed_smem_layout())
        sScale = storage.scale_payload.get_tensor(self._scale_smem_layout())
        sAmaxPartial = storage.amax_partial.get_tensor(cute.make_layout((self.num_threads,), stride=(1,)))

        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.load_rows * _TILE_N * 2,
            defer_sync=False,
        )
        store_pipe = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
        )
        producer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        consumer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)

        gX = cute.local_tile(x_tma, (self.load_rows, _TILE_N), (pid_m, pid_n))
        gPacked = cute.local_tile(packed_tma, (_TILE_M, _TILE_PACKED_COLS), (pid_m, pid_n))
        gScale = cute.local_tile(scale_tma, (_TILE_M, _TILE_SCALE_COLS), (pid_m, pid_n))
        load, _, _ = copy_utils.tma_get_copy_fn(
            x_tma_atom,
            0,
            cute.make_layout(1),
            gX,
            sX_load,
        )
        load = copy_utils.tma_producer_copy_fn(load, load_pipe)
        store_packed, _, _ = copy_utils.tma_get_copy_fn(
            packed_tma_atom,
            0,
            cute.make_layout(1),
            sPacked,
            gPacked,
            single_stage=True,
        )
        store_scale, _, _ = copy_utils.tma_get_copy_fn(
            scale_tma_atom,
            0,
            cute.make_layout(1),
            sScale,
            gScale,
            single_stage=True,
        )

        zero_idx = tidx
        while zero_idx < _TILE_M * _TILE_PACKED_COLS:
            sPacked[zero_idx // _TILE_PACKED_COLS, zero_idx % _TILE_PACKED_COLS] = cutlass.Uint8(0)
            zero_idx += self.num_threads
        zero_idx = tidx
        while zero_idx < _TILE_M * _TILE_SCALE_COLS:
            sScale[zero_idx // _TILE_SCALE_COLS, zero_idx % _TILE_SCALE_COLS] = cutlass.Uint8(0)
            zero_idx += self.num_threads
        cute.arch.sync_threads()

        if tidx == 0:
            load_pipe.producer_acquire(producer_state)
            load(src_idx=0, producer_state=producer_state)

        load_pipe.consumer_wait(consumer_state, load_pipe.consumer_try_wait(consumer_state))
        cute.arch.sync_threads()

        local_tile_amax = cutlass.Float32(0.0)
        block_idx = tidx
        while block_idx < self.load_rows * _TILE_SCALE_COLS:
            row = block_idx // _TILE_SCALE_COLS
            sf_block = block_idx % _TILE_SCALE_COLS
            values = cute.make_rmem_tensor((16,), cutlass.Float32)
            block_max = cutlass.Float32(0.0)
            for elem_idx in cutlass.range_constexpr(16):
                value = cutlass.Float32(sX_full[row, sf_block * 16 + elem_idx])
                values[elem_idx] = value
                block_max = fmax_f32(block_max, fabs_f32(value))
            local_tile_amax = fmax_f32(local_tile_amax, block_max)
            packed64, scale_byte = quantize_block_fp4_fast(values, block_max, cutlass.Float32(1.0))
            for byte_idx in cutlass.range_constexpr(8):
                sPacked[row, sf_block * 8 + byte_idx] = cutlass.Uint8(
                    (packed64 >> cutlass.Uint64(byte_idx * 8)) & cutlass.Uint64(0xFF)
                )
            sScale[row, sf_block] = scale_byte
            block_idx += self.num_threads

        sAmaxPartial[tidx] = local_tile_amax
        cute.arch.sync_threads()
        if tidx == 0:
            tile_max = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(self.num_threads):
                tile_max = fmax_f32(tile_max, sAmaxPartial[i])
            tile_amax[pid_m, pid_n] = tile_max

        if tidx == 0:
            load_pipe.consumer_release(consumer_state)
            load_pipe.producer_tail(producer_state)

        cute.arch.sync_threads()
        cute.arch.fence_proxy("async.shared", space="cta")
        if tidx < 32:
            store_packed()
            store_pipe.producer_commit()
            store_pipe.producer_acquire()
            store_scale()
            store_pipe.producer_commit()
            store_pipe.producer_acquire()
            store_pipe.producer_tail()


def _cute_tiled_amax_pack_fp4_128x128(
    x: torch.Tensor,
    packed_tiles: torch.Tensor,
    scale_tiles: torch.Tensor,
    tile_amax: torch.Tensor,
) -> None:
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key = (
        int(x.shape[0]),
        int(x.shape[1]),
        int(x.device.index or 0),
    )
    compiled = _CUTE_TILE_QUANT_CACHE.get(key)
    x_cute = _to_cute_tensor(x, cutlass.BFloat16)
    packed_cute = _to_cute_tensor(packed_tiles, cutlass.Uint8)
    scale_cute = _to_cute_tensor(scale_tiles, cutlass.Uint8)
    amax_cute = _to_cute_tensor(tile_amax, cutlass.Float32)
    if compiled is None:
        compiled = cute.compile(
            _CuteTileAmaxPackProbe(min(int(x.shape[0]), _TILE_M)),
            x_cute,
            packed_cute,
            scale_cute,
            amax_cute,
            stream,
        )
        _CUTE_TILE_QUANT_CACHE[key] = compiled
    compiled(
        x_cute,
        packed_cute,
        scale_cute,
        amax_cute,
        stream,
    )


def _pick_batch_sizes(args: argparse.Namespace) -> list[int]:
    if args.batch_sizes:
        return list(args.batch_sizes)
    return list(BATCH_SIZE_PROFILES[args.batch_size_profile])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size-profile", choices=sorted(BATCH_SIZE_PROFILES), default="micro")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip output validation between the dynamic superfused and semi-fused paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", _local_rank()))
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    require_sm120()
    torch.set_grad_enabled(False)
    clear_tp_moe_caches()

    spec = _make_spec()
    batch_sizes = _pick_batch_sizes(args)
    max_tokens = max(batch_sizes)
    max_input_bytes = max_tokens * spec.hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()

    _rank0_print(
        "Gemma dynamic superfused benchmark | "
        f"world_size={dist.get_world_size()} K={spec.hidden_size} I_tp={spec.I_tp} "
        f"E={spec.num_experts}+1 top_k={spec.top_k}+1 | timings use CUDA graph replay"
    )

    runtime = PCIeOneshotAllReduce.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=max_input_bytes,
    )
    try:
        with torch.no_grad():
            sparse_weights = load_expert_weights(MODEL_PATH, spec, layer_idx=args.layer_idx)
            shared_weights = load_shared_expert_weights(MODEL_PATH, spec, layer_idx=args.layer_idx)
            gate_weight = load_gate_weight(MODEL_PATH, spec, layer_idx=args.layer_idx)
            shared_gate_weight = load_shared_gate_weight(MODEL_PATH, layer_idx=args.layer_idx)
            norm_weight = _load_post_attention_layernorm_weight(
                MODEL_PATH,
                layer_idx=args.layer_idx,
                device=device,
            )
            sparse_experts = _pack_sparse_experts_per_expert(sparse_weights)
            shared_expert = _pack_shared_expert(shared_weights)
            combined_experts = _append_expert_bank(sparse_experts, shared_expert)
            shared_expert_id = sparse_experts.w1_fp4.shape[0]

            for m in batch_sizes:
                hidden_local = make_input_activations(
                    spec,
                    m,
                    seed=args.seed + 17 * _rank() + m,
                    device=device,
                )
                residual = make_input_activations(
                    spec,
                    m,
                    seed=args.seed + 10_000 + m,
                    device=device,
                )

                fused_pool = allocate_tp_moe_workspace_pool()
                semi_pool = allocate_tp_moe_workspace_pool()
                fused_output = torch.empty_like(hidden_local)
                fused_residual_out = torch.empty_like(hidden_local)
                semi_output = torch.empty_like(hidden_local)
                tiles_m = (m + _TILE_M - 1) // _TILE_M
                tiles_n = (spec.hidden_size + _TILE_N - 1) // _TILE_N
                semi_tile_amax = torch.empty(
                    (tiles_m, tiles_n),
                    device=device,
                    dtype=torch.float32,
                )
                semi_sA_tiles = torch.empty(
                    (tiles_m * _TILE_M, tiles_n * _TILE_PACKED_COLS),
                    device=device,
                    dtype=torch.uint8,
                )
                semi_sSFA_tiles = torch.empty(
                    (tiles_m * _TILE_M, tiles_n * _TILE_SCALE_COLS),
                    device=device,
                    dtype=torch.uint8,
                )

                def superfused_path() -> None:
                    _b12x_gemma_moe_block_fp4_dynamic_superfused(
                        hidden_local,
                        residual,
                        pre_mlp_runtime=runtime,
                        norm_weight=norm_weight,
                        norm_eps=args.norm_eps,
                        sparse_experts=sparse_experts,
                        shared_expert=shared_expert,
                        shared_gate_weight=shared_gate_weight,
                        combined_experts=combined_experts,
                        workspace=fused_pool,
                        top_k=spec.top_k,
                        gate_weight=gate_weight,
                        output=fused_output,
                        residual_out=fused_residual_out,
                        input_scales_are_reciprocal=True,
                        input_scales_static=True,
                    )

                def semi_fused_path() -> tuple[torch.Tensor, torch.Tensor]:
                    reduced = hidden_local.clone()
                    dist.all_reduce(reduced)
                    normed_hidden_states, semi_residual_out = _gemma_rmsnorm_after_allreduce(
                        reduced,
                        residual,
                        norm_weight,
                        args.norm_eps,
                    )
                    _cute_tiled_amax_pack_fp4_128x128(
                        normed_hidden_states,
                        semi_sA_tiles,
                        semi_sSFA_tiles,
                        semi_tile_amax,
                    )
                    sparse_routing = b12x_route_experts_fast(
                        normed_hidden_states,
                        top_k=spec.top_k,
                        gate_weight=gate_weight,
                        workspace=semi_pool,
                    )
                    combined_routing = _append_shared_expert_routing(
                        sparse_routing,
                        shared_gate_weights=_shared_expert_gate_weights(
                            normed_hidden_states,
                            gate_weight=shared_gate_weight,
                        ),
                        shared_expert_id=shared_expert_id,
                    )
                    out = b12x_moe_fp4(
                        normed_hidden_states,
                        combined_experts.a1_gscale,
                        combined_experts.w1_fp4,
                        combined_experts.w1_blockscale,
                        combined_experts.w1_alphas,
                        combined_experts.a2_gscale,
                        combined_experts.w2_fp4,
                        combined_experts.w2_blockscale,
                        combined_experts.w2_alphas,
                        combined_routing.topk_weights,
                        combined_routing.topk_ids,
                        workspace=semi_pool,
                        output=semi_output,
                        input_scales_are_reciprocal=True,
                        input_scales_static=True,
                        fc2_tile_amax=False,
                    )
                    return out, semi_residual_out

                superfused_path()
                semi_out, semi_residual_out = semi_fused_path()
                torch.cuda.synchronize(device)

                if not args.skip_validate:
                    output_metrics = compare_to_reference(fused_output, semi_out)
                    residual_metrics = compare_to_reference(fused_residual_out, semi_residual_out)
                    if output_metrics.max_abs > 2e-2 or output_metrics.cos <= 0.999:
                        raise RuntimeError(
                            f"m={m} output mismatch on rank {_rank()}: "
                            f"max_abs={output_metrics.max_abs:.3e} cos={output_metrics.cos:.6f}"
                        )
                    if residual_metrics.max_abs > 0.0 or residual_metrics.cos <= 0.999999:
                        raise RuntimeError(
                            f"m={m} residual mismatch on rank {_rank()}: "
                            f"max_abs={residual_metrics.max_abs:.3e} cos={residual_metrics.cos:.6f}"
                        )
                    _rank0_print(
                        f"m={m} validate: out(max_abs={output_metrics.max_abs:.3e}, cos={output_metrics.cos:.6f}) "
                        f"residual(max_abs={residual_metrics.max_abs:.3e}, cos={residual_metrics.cos:.6f})"
                    )

                superfused_times = _bench_graph_replay(
                    superfused_path,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                semi_times = _bench_graph_replay(
                    lambda: semi_fused_path()[0],
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                superfused_rank_medians = _gather_rank_medians(superfused_times)
                semi_rank_medians = _gather_rank_medians(semi_times)
                if _rank() == 0:
                    superfused_med_us = max(superfused_rank_medians) * 1000.0
                    semi_med_us = max(semi_rank_medians) * 1000.0
                    superfused_ratio = superfused_med_us / semi_med_us
                    print(
                        f"m={m:5d} | dynamic_superfused { _fmt_us(superfused_times) } | "
                        f"semi_fused { _fmt_us(semi_times) } | "
                        f"ratio dyn {superfused_ratio:.3f}x",
                        flush=True,
                    )
    finally:
        runtime.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
