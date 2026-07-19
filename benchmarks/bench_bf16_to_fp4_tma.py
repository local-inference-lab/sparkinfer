"""Benchmark the reusable BF16->FP4 TMA quantization kernel module."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import statistics as _stats
import subprocess
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from b12x.cute.compiler import b12x_package_fingerprint
from b12x.cute.fp4 import quantize_grouped_nvfp4_torch
from b12x.quantization import allocate_bf16_to_fp4_tma_outputs, compile_bf16_to_fp4_tma


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--raw-samples-jsonl",
        type=pathlib.Path,
        help="Append provenance, exact-output hashes, and every timing sample.",
    )
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    return parser.parse_args()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _cutlass_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in (
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-core",
        "nvidia-cutlass-dsl-libs-cu12",
        "nvidia-cutlass-dsl-libs-cu13",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


def _physical_reference(
    source: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = source.shape
    packed, scale_view = quantize_grouped_nvfp4_torch(
        source.unsqueeze(0),
        torch.tensor([rows], dtype=torch.int32, device=source.device),
        global_scale,
    )
    scale_storage = (
        scale_view.permute(5, 2, 4, 0, 1, 3)
        .contiguous()
        .view(torch.uint8)
        .reshape(-1)
    )
    return packed, scale_storage


def main() -> None:
    args = _parse_args()
    m = int(args.M)
    k = int(args.K)
    dev = torch.device("cuda")
    torch.manual_seed(42)
    bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    gs = torch.tensor([args.global_scale], dtype=torch.float32, device=dev)
    rows_padded = ((m + 127) // 128) * 128
    csf = ((k // 16 + 3) // 4) * 4
    inp = (
        bf16
        if rows_padded == m and bf16.is_contiguous()
        else torch.zeros((rows_padded, k), dtype=torch.bfloat16, device=dev)
    )
    if rows_padded != m:
        inp[:m].copy_(bf16)
    out = allocate_bf16_to_fp4_tma_outputs(m, k, device=dev)
    compiled = compile_bf16_to_fp4_tma(rows_padded, k)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    flush_desc = f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)" if l2_flush else "off"
    print("Compiled OK")
    print(f"L2 flush: {flush_desc}")

    def launch() -> None:
        compiled(inp, gs, out.packed_a_flat, out.scale_flat)

    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    packed_ref, scale_ref = _physical_reference(inp, gs)
    packed_actual = out.packed_a_storage.permute(1, 2, 0)
    torch.testing.assert_close(packed_actual, packed_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out.scale_flat, scale_ref, rtol=0.0, atol=0.0)
    input_hash = _tensor_sha256(inp)
    global_scale_hash = _tensor_sha256(gs)
    packed_reference_hash = _tensor_sha256(packed_ref)
    scale_reference_hash = _tensor_sha256(scale_ref)
    output_ptrs = (out.packed_a_flat.data_ptr(), out.scale_flat.data_ptr())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        if l2_flush is not None:
            l2_flush()
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for idx in range(args.iters):
        if l2_flush is not None:
            l2_flush()
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    times_ms = [starts[idx].elapsed_time(ends[idx]) for idx in range(args.iters)]
    samples_us = [sample * 1000.0 for sample in times_ms]
    med_us = _stats.median(samples_us)
    min_us = min(samples_us)
    read_bytes = rows_padded * k * 2
    write_bytes = rows_padded * k // 2 + rows_padded * csf
    bw = (read_bytes + write_bytes) / (med_us * 1e-6) / 1e9
    print(f"M={m} K={k}  graph replay median: {med_us:.1f} us  (min {min_us:.1f})  BW: {bw:.1f} GB/s")

    assert output_ptrs == (out.packed_a_flat.data_ptr(), out.scale_flat.data_ptr())
    assert input_hash == _tensor_sha256(inp)
    assert global_scale_hash == _tensor_sha256(gs)
    torch.testing.assert_close(packed_actual, packed_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out.scale_flat, scale_ref, rtol=0.0, atol=0.0)
    packed_output_hash = _tensor_sha256(packed_actual)
    scale_output_hash = _tensor_sha256(out.scale_flat)
    assert packed_output_hash == packed_reference_hash
    assert scale_output_hash == scale_reference_hash

    if args.raw_samples_jsonl is not None:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        sorted_samples = sorted(samples_us)
        p95_index = min(len(sorted_samples) - 1, int(0.95 * len(sorted_samples)))
        record = {
            "schema": "b12x.bf16_to_fp4_tma.graph_abba.v1",
            "run_label": args.run_label,
            "command": [sys.executable, *sys.argv],
            "cwd": os.getcwd(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "worktree": _git_value("rev-parse", "--show-toplevel"),
            "dirty": bool(_git_value("status", "--short")),
            "b12x_package_fingerprint": b12x_package_fingerprint(),
            "cutlass_versions": _cutlass_versions(),
            "gpu": {
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_index": device_index,
                "name": props.name,
                "capability": list(torch.cuda.get_device_capability(device_index)),
                "uuid": str(getattr(props, "uuid", "")),
            },
            "shape": {"M": m, "K": k},
            "global_scale": float(args.global_scale),
            "graph_replay": True,
            "warmup": int(args.warmup),
            "iterations": int(args.iters),
            "l2_flush_bytes": l2_flush_bytes if l2_flush is not None else 0,
            "samples_us": samples_us,
            "median_us": med_us,
            "mean_us": _stats.mean(samples_us),
            "min_us": min_us,
            "p95_us": sorted_samples[p95_index],
            "bandwidth_gbps": bw,
            "exact_output_hashes": {
                "input": input_hash,
                "global_scale": global_scale_hash,
                "packed_reference": packed_reference_hash,
                "packed_output": packed_output_hash,
                "scale_reference": scale_reference_hash,
                "scale_output": scale_output_hash,
            },
            "output_pointer_stable": True,
        }
        args.raw_samples_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.raw_samples_jsonl.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
