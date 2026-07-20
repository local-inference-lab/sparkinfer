#!/usr/bin/env python3
"""Strict CUDA-graph benchmark for SM120 W4A8 NVFP4/ReLU2 tile kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Mapping

import torch

from sparkinfer.cute import sparkinfer_package_fingerprint
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument,
    require_target_gpu,
)
from validation.cutlass_migration.paths import REPO_ROOT
from tests.test_w4a8_dynamic_kernel import _run_w4a8_dynamic


_CUTLASS_PACKAGES = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
)
_RUNTIME_ENVIRONMENT_PREFIXES = (
    "SPARKINFER_",
    "CUTE_",
    "CUTLASS_",
    "CUDA_",
    "TORCH_",
    "PYTORCH_",
    "TRITON_",
    "NVCC_",
    "PTXAS_",
    "NCCL_",
)
_RUNTIME_ENVIRONMENT_EXPLICIT_CONTROLS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_PATH",
    "CUDA_CACHE_MAXSIZE",
    "CUDA_FORCE_PTX_JIT",
    "CUDA_DISABLE_PTX_JIT",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_TF32_OVERRIDE",
)
_TENSOR_HASH_CHUNK_BYTES = 64 * 1024 * 1024
_MINIMUM_COSINE = 0.999
_MAXIMUM_RELATIVE_L2 = 0.03
_RELATIVE_TOLERANCE = 0.15
# The W4A8 oracle compares two independently rounded quantized pipelines.
# Cosine and relative-L2 remain the tight gates; this absolute bound prevents
# near-zero elements from making the required elementwise-allclose field
# reject an otherwise accurate result.
_ABSOLUTE_TOLERANCE = 0.20


@dataclass(frozen=True)
class _LiveVariant:
    x: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    sha256: str


@dataclass(frozen=True)
class _ReadOnlySnapshot:
    tensor_sha256: dict[str, str]
    aggregate_sha256: str


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "layout": str(tensor.layout),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    byte_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    for offset in range(0, byte_view.numel(), _TENSOR_HASH_CHUNK_BYTES):
        chunk = byte_view.narrow(
            0,
            offset,
            min(_TENSOR_HASH_CHUNK_BYTES, byte_view.numel() - offset),
        ).cpu()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _snapshot_read_only(inputs: Mapping[str, torch.Tensor]) -> _ReadOnlySnapshot:
    tensor_hashes = {
        name: _tensor_content_sha256(tensor) for name, tensor in sorted(inputs.items())
    }
    return _ReadOnlySnapshot(
        tensor_sha256=tensor_hashes,
        aggregate_sha256=_json_sha256(
            {
                "schema": "sparkinfer-read-only-inputs-v1",
                "tensor_sha256": tensor_hashes,
            }
        ),
    )


def _assert_read_only_unchanged(
    snapshot: _ReadOnlySnapshot,
    inputs: Mapping[str, torch.Tensor],
) -> None:
    current = _snapshot_read_only(inputs)
    if current != snapshot:
        changed = [
            name
            for name, digest in snapshot.tensor_sha256.items()
            if current.tensor_sha256.get(name) != digest
        ]
        raise AssertionError(f"W4A8 read-only inputs were mutated: {changed}")


def _read_only_provenance(snapshot: _ReadOnlySnapshot) -> dict[str, object]:
    return {
        "schema": "sparkinfer-read-only-inputs-v1",
        "tensor_sha256": dict(snapshot.tensor_sha256),
        "aggregate_sha256": snapshot.aggregate_sha256,
    }


def _generate_variants(args: argparse.Namespace) -> tuple[list[_LiveVariant], str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    variants: list[_LiveVariant] = []
    token = torch.arange(args.m, dtype=torch.int64)
    for variant_idx in range(args.variants):
        x_cpu = (
            torch.randn(
                args.m,
                args.hidden_size,
                generator=generator,
                dtype=torch.float32,
            )
            * 2.0
        ).to(torch.bfloat16)
        ids_cpu = torch.empty(args.m, 2, dtype=torch.int32)
        ids_cpu[:, 0] = 0
        ids_cpu[:, 1] = 1 + ((token + variant_idx) % (args.experts - 1)).to(torch.int32)
        primary = 0.55 + 0.05 * ((token * 3 + variant_idx) % 7).to(torch.float32)
        weights_cpu = torch.stack((primary, 1.0 - primary), dim=1).contiguous()
        component_hashes = {
            "x": _tensor_content_sha256(x_cpu),
            "topk_ids": _tensor_content_sha256(ids_cpu),
            "topk_weights": _tensor_content_sha256(weights_cpu),
        }
        variant_sha = _json_sha256(
            {"variant_index": variant_idx, "tensor_sha256": component_hashes}
        )
        variants.append(
            _LiveVariant(
                x=x_cpu.cuda(),
                topk_ids=ids_cpu.cuda(),
                topk_weights=weights_cpu.cuda(),
                sha256=variant_sha,
            )
        )
    generation = {
        "generator": "torch-cpu-philox-w4a8-live-inputs-v1",
        "seed": args.seed,
        "variant_sha256": [variant.sha256 for variant in variants],
    }
    return variants, _json_sha256(generation)


def _apply_variant(
    live_inputs: Mapping[str, torch.Tensor], variant: _LiveVariant
) -> None:
    live_inputs["x"].copy_(variant.x)
    live_inputs["topk_ids"].copy_(variant.topk_ids)
    live_inputs["topk_weights"].copy_(variant.topk_weights)


def _capture_graph(
    launch: Callable[[], None],
    *,
    live_inputs: Mapping[str, torch.Tensor],
    variants: list[_LiveVariant],
    warmup: int,
) -> torch.cuda.CUDAGraph:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for idx in range(warmup):
            _apply_variant(live_inputs, variants[idx % len(variants)])
            launch()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        launch()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def _relative_l2_error(a: torch.Tensor, b: torch.Tensor) -> float:
    diff_norm = (a.float() - b.float()).norm().item()
    return diff_norm / max(b.float().norm().item(), 1e-12)


def _correctness_metrics(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    phase: str,
    variant_index: int,
) -> dict[str, object]:
    finite = bool(torch.isfinite(output).all().item())
    nonzero = int(torch.count_nonzero(output).item())
    max_abs = float((output.float() - reference.float()).abs().max().item())
    relative_l2 = _relative_l2_error(output, reference)
    cosine = _cosine_similarity(output, reference)
    allclose = bool(
        torch.allclose(
            output.float(),
            reference.float(),
            rtol=_RELATIVE_TOLERANCE,
            atol=_ABSOLUTE_TOLERANCE,
        )
    )
    passed = bool(
        finite
        and nonzero > 0
        and math.isfinite(relative_l2)
        and relative_l2 <= _MAXIMUM_RELATIVE_L2
        and math.isfinite(cosine)
        and cosine >= _MINIMUM_COSINE
        and allclose
    )
    metrics: dict[str, object] = {
        "phase": phase,
        "variant_index": variant_index,
        "passed": passed,
        "finite": finite,
        "nonzero": nonzero,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "allclose": allclose,
    }
    if not passed:
        raise AssertionError(f"W4A8 graph correctness failed: {metrics}")
    return metrics


def _correctness_sweep(
    graph: torch.cuda.CUDAGraph,
    *,
    phase: str,
    live_inputs: Mapping[str, torch.Tensor],
    variants: list[_LiveVariant],
    references: list[torch.Tensor],
) -> list[dict[str, object]]:
    results = []
    for variant_idx, (variant, reference) in enumerate(
        zip(variants, references, strict=True)
    ):
        _apply_variant(live_inputs, variant)
        graph.replay()
        torch.cuda.synchronize()
        results.append(
            _correctness_metrics(
                live_inputs["output"],
                reference,
                phase=phase,
                variant_index=variant_idx,
            )
        )
    return results


def _benchmark_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    live_inputs: Mapping[str, torch.Tensor],
    variants: list[_LiveVariant],
    replays: int,
    l2_flush: Callable[[], None] | None,
) -> tuple[list[float], list[int]]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    schedule = [(idx + 1) % len(variants) for idx in range(replays)]
    for idx, variant_idx in enumerate(schedule):
        _apply_variant(live_inputs, variants[variant_idx])
        torch.cuda.synchronize()
        if l2_flush is not None:
            l2_flush()
            torch.cuda.synchronize()
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    samples_ms = [
        start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
    ]
    return samples_ms, schedule


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "missing"


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _runtime_environment_provenance() -> dict[str, object]:
    set_variables = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(_RUNTIME_ENVIRONMENT_PREFIXES)
    }
    explicit_controls = {
        name: (
            {"status": "set", "value": os.environ[name]}
            if name in os.environ
            else {"status": "missing"}
        )
        for name in _RUNTIME_ENVIRONMENT_EXPLICIT_CONTROLS
    }
    payload: dict[str, object] = {
        "schema": "sparkinfer-runtime-environment-v1",
        "complete_set_variable_prefixes": list(_RUNTIME_ENVIRONMENT_PREFIXES),
        "set_variables": set_variables,
        "explicit_controls": explicit_controls,
        "nvidia_enumeration": {
            "policy": "explicit-only",
            "included": [
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "NVIDIA_TF32_OVERRIDE",
            ],
            "reason": "avoid collecting unrelated NVIDIA_ variables that may contain secrets",
        },
    }
    payload["sha256"] = _json_sha256(payload)
    return payload


def _benchmark_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: value for key, value in vars(args).items() if key != "raw_samples_jsonl"
    }


def _expected_case_contract(args: argparse.Namespace) -> dict[str, object]:
    identity_fields = [
        "recipe",
        "activation",
        "experts",
        "m",
        "hidden_size",
        "intermediate_size",
        "top_k",
        "tile_m",
        "live_input_variants",
    ]
    expected = [
        {
            "recipe": "w4a8_nvfp4",
            "activation": "relu2",
            "experts": args.experts,
            "m": args.m,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "top_k": args.top_k,
            "tile_m": args.tile_m,
            "live_input_variants": args.variants,
        }
    ]
    contract: dict[str, object] = {
        "identity_fields": identity_fields,
        "expected": expected,
    }
    contract["sha256"] = _json_sha256(contract)
    return contract


def _initialize_log(
    path: pathlib.Path,
    *,
    args: argparse.Namespace,
    argv: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    repo = REPO_ROOT
    benchmark_path = pathlib.Path(__file__).resolve()
    dependencies = (
        benchmark_path,
        repo / "benchmarks" / "common.py",
        repo / "tests" / "test_w4a8_dynamic_kernel.py",
    )
    benchmark_dependencies = {
        str(dependency.relative_to(repo)): hashlib.sha256(
            dependency.read_bytes()
        ).hexdigest()
        for dependency in dependencies
    }
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logical_device = torch.cuda.current_device()
    visible = [part.strip() for part in visible_devices.split(",") if part.strip()]
    physical_device = visible[logical_device] if logical_device < len(visible) else None
    properties = torch.cuda.get_device_properties(logical_device)
    config = _benchmark_config(args)
    runtime_environment = _runtime_environment_provenance()
    provenance = {
        "type": "provenance",
        "command": shlex.join(
            [
                sys.executable,
                "-m",
                "validation.cutlass_migration",
                "diagnostic",
                "w4a8-m32",
                *argv,
            ]
        ),
        "argv": argv,
        "worktree": str(repo),
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "dirty_paths": _git_value("status", "--short").splitlines(),
        "sparkinfer_package_fingerprint": sparkinfer_package_fingerprint(),
        "benchmark_sha256": benchmark_dependencies[
            str(benchmark_path.relative_to(repo))
        ],
        "benchmark_dependencies_sha256": benchmark_dependencies,
        "benchmark_config": config,
        "benchmark_config_sha256": _json_sha256(config),
        "benchmark_case_contract": _expected_case_contract(args),
        "runtime_environment": runtime_environment,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cutlass": {
            package: _package_version(package) for package in _CUTLASS_PACKAGES
        },
        "runtime_packages": {
            package: _package_version(package)
            for package in ("cuda-python", "cuda-bindings")
        },
        "gpu": {
            "cuda_visible_devices": visible_devices,
            "logical_index": logical_device,
            "physical_index": physical_device,
            "name": torch.cuda.get_device_name(logical_device),
            "uuid": str(properties.uuid),
            "capability": list(torch.cuda.get_device_capability(logical_device)),
            "l2_cache_bytes": int(properties.L2_cache_size),
        },
        "serving_mode": {
            "cuda_graph_replay": True,
            "stable_allocations": True,
            "fixed_workspace_capacity": True,
            "live_inputs_mutated_between_replays": True,
            "mutation_outside_timed_interval": True,
            "oracle_outside_timed_interval": True,
            "warmup": args.warmup,
            "replays": args.replays,
            "l2_flush": args.flush_l2,
            "l2_flush_bytes": resolve_l2_flush_bytes(args.l2_flush_bytes),
            "correctness": "torch-reference",
        },
    }
    with path.open("a", encoding="utf-8") as output:
        json.dump(provenance, output, sort_keys=True, allow_nan=False)
        output.write("\n")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _record_samples(
    path: pathlib.Path,
    *,
    case: dict[str, object],
    samples_ms: list[float],
    correctness: dict[str, object],
) -> None:
    samples_us = [sample * 1000.0 for sample in samples_ms]
    record = {
        "type": "graph-replay-samples",
        "backend": "sparkinfer",
        "case": case,
        "unit": "us",
        "samples": samples_us,
        "count": len(samples_us),
        "mean": statistics.fmean(samples_us),
        "median": statistics.median(samples_us),
        "p95": _percentile(samples_us, 0.95),
        "sample_stdev": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
        "minimum": min(samples_us),
        "maximum": max(samples_us),
        "correctness": correctness,
    }
    with path.open("a", encoding="utf-8") as output:
        json.dump(record, output, sort_keys=True, allow_nan=False)
        output.write("\n")


def _case(args: argparse.Namespace, input_generation_sha256: str) -> dict[str, object]:
    case: dict[str, object] = {
        "recipe": "w4a8_nvfp4",
        "activation": "relu2",
        "experts": args.experts,
        "m": args.m,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "top_k": args.top_k,
        "tile_m": args.tile_m,
        "live_input_variants": args.variants,
        "forced_primary_expert": 0,
        "boundary_rows": [args.tile_m // 2, args.tile_m // 2 + 1],
        "input_seed": args.seed,
        "input_generation_sha256": input_generation_sha256,
    }
    case["case_contract_sha256"] = _json_sha256(case)
    return case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_gpu_argument(parser)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--tile-m", type=int, choices=(32, 128), default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1032)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument(
        "--flush-l2", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--raw-samples-jsonl", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    parsed_argv = list(sys.argv[1:] if argv is None else argv)

    if args.experts < 2:
        parser.error("--experts must be at least 2")
    if args.tile_m == 32 and not 18 <= args.m <= 32:
        parser.error(
            "with --tile-m=32, --m must be in [18, 32] to exercise the "
            "row-16/17 boundary"
        )
    if args.tile_m == 128 and args.m != 129:
        parser.error(
            "with --tile-m=128, --m must be 129 to exercise the exact direct "
            "M128 tail specialization"
        )
    if args.hidden_size != 256 or args.intermediate_size != 128:
        parser.error("this strict specialization benchmark requires K=256 and n=128")
    if args.top_k != 2:
        parser.error("this strict specialization benchmark requires --top-k=2")
    if args.variants < 2 or args.warmup < 1 or args.replays < 1:
        parser.error("--variants must be >=2 and warmup/replays must be positive")
    if args.l2_flush_bytes < 0:
        parser.error("--l2-flush-bytes must be nonnegative")
    require_target_gpu(args.expected_physical_gpu)

    _initialize_log(args.raw_samples_jsonl, args=args, argv=parsed_argv)
    variants, input_generation_sha256 = _generate_variants(args)
    initial_ids = variants[0].topk_ids.cpu()
    output, _, launch, state = _run_w4a8_dynamic(
        recipe="w4a8_nvfp4",
        activation="relu2",
        E=args.experts,
        m=args.m,
        K=args.hidden_size,
        n=args.intermediate_size,
        top_k=args.top_k,
        seed=args.seed,
        tile_m=args.tile_m,
        return_launcher=True,
        return_state=True,
        topk_ids_override=initial_ids,
    )
    live_inputs = dict(state["live_inputs"])
    live_inputs["output"] = output
    read_only_inputs = state["read_only_inputs"]
    all_allocations = {
        **live_inputs,
        **{f"read_only.{name}": tensor for name, tensor in read_only_inputs.items()},
        **{
            f"workspace.{name}": tensor
            for name, tensor in state["mutable_allocations"].items()
        },
    }
    allocation_pointers = {
        name: tensor.data_ptr() for name, tensor in all_allocations.items()
    }

    references = []
    for variant in variants:
        _apply_variant(live_inputs, variant)
        torch.cuda.synchronize()
        references.append(state["current_reference"]().detach())
    torch.cuda.synchronize()
    read_only_snapshot = _snapshot_read_only(read_only_inputs)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    if l2_flush is not None:
        l2_flush()
        torch.cuda.synchronize()

    graph = _capture_graph(
        launch,
        live_inputs=live_inputs,
        variants=variants,
        warmup=args.warmup,
    )
    correctness_results = _correctness_sweep(
        graph,
        phase="pre-timing",
        live_inputs=live_inputs,
        variants=variants,
        references=references,
    )
    samples_ms, schedule = _benchmark_graph(
        graph,
        live_inputs=live_inputs,
        variants=variants,
        replays=args.replays,
        l2_flush=l2_flush,
    )
    correctness_results.extend(
        _correctness_sweep(
            graph,
            phase="post-timing",
            live_inputs=live_inputs,
            variants=variants,
            references=references,
        )
    )
    _assert_read_only_unchanged(read_only_snapshot, read_only_inputs)
    final_pointers = {
        name: tensor.data_ptr() for name, tensor in all_allocations.items()
    }
    if final_pointers != allocation_pointers:
        raise AssertionError("W4A8 graph allocation addresses changed")

    correctness = {
        "oracle": "torch-reference",
        "passed": True,
        "finite": all(bool(result["finite"]) for result in correctness_results),
        "nonzero": min(int(result["nonzero"]) for result in correctness_results),
        "max_abs": max(float(result["max_abs"]) for result in correctness_results),
        "relative_l2": max(
            float(result["relative_l2"]) for result in correctness_results
        ),
        "cosine": min(float(result["cosine"]) for result in correctness_results),
        "allclose": all(bool(result["allclose"]) for result in correctness_results),
        "minimum_cosine": _MINIMUM_COSINE,
        "maximum_relative_l2": _MAXIMUM_RELATIVE_L2,
        "relative_tolerance": _RELATIVE_TOLERANCE,
        "absolute_tolerance": _ABSOLUTE_TOLERANCE,
        "read_only_inputs": _read_only_provenance(read_only_snapshot),
        "variant_results": correctness_results,
        "live_input_mutation": {
            "schema": "sparkinfer-live-input-mutation-v1",
            "variant_sha256": [variant.sha256 for variant in variants],
            "distinct_variants": len(set(variant.sha256 for variant in variants)),
            "timed_schedule_sha256": _json_sha256(schedule),
            "mutations_outside_timed_interval": True,
            "allocation_addresses_stable": True,
        },
    }
    _record_samples(
        args.raw_samples_jsonl,
        case=_case(args, input_generation_sha256),
        samples_ms=samples_ms,
        correctness=correctness,
    )
    print(
        json.dumps(
            {
                "raw_samples_jsonl": str(args.raw_samples_jsonl),
                "samples": len(samples_ms),
                "mean_us": statistics.fmean(samples_ms) * 1000.0,
                "median_us": statistics.median(samples_ms) * 1000.0,
                "p95_us": _percentile(samples_ms, 0.95) * 1000.0,
                "minimum_cosine": correctness["cosine"],
                "maximum_relative_l2": correctness["relative_l2"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
