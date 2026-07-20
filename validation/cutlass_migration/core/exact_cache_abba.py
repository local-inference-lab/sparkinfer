"""Shared strict machinery for exact-object CUDA-graph ABBA benchmarks.

This module is benchmark infrastructure, not a CPU acceptance path.  Its
entrypoints deliberately fail unless the caller selected physical GPU 4 or 5
and the visible device is SM120.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import tempfile
import time
from types import ModuleType
from typing import Any

from cuda.bindings import driver as cuda_driver
import torch

from benchmarks.common import (
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    resolve_l2_flush_bytes,
)
from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument as add_target_gpu_argument,
    require_target_gpu as require_target_gpu,
)
import sparkinfer.cute.compiler as cute_compiler


SINGLE_ARM_E2E_RUN_SCHEMA = "sparkinfer.cute.migration.end_to_end_process_result.v4"
_ABBA_EVENT_POOL_SCHEMA = "sparkinfer.cuda_event_pool.v1"
_ABBA_MODE_STABILITY_SCHEMA = "sparkinfer.gpu_mode_stability.v1"
GPU_TIMING_MODE_POLICY_SCHEMA = "sparkinfer.gpu_timing_mode_policy.v1"
_PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS = frozenset((0, 0x4))
_CUTLASS_PACKAGE_NAMES = (
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
_RUNTIME_ENVIRONMENT_OPERATIONAL_PATH_EXCEPTIONS = (
    "SPARKINFER_CUTE_COMPILE_CACHE_DIR",
    "CUTE_DSL_CACHE_DIR",
    "CUTE_DSL_LIBS",
)


@dataclass(frozen=True)
class _PreparedAbbaEventPool:
    """A fixed, eagerly initialized CUDA-event pool for one ABBA condition."""

    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]
    metadata: dict[str, object]
    cycles: int
    event_batch_cycles: int
    replays_per_reported_sample: int


@dataclass(frozen=True)
class _PreparedSingleGraphEventPool:
    """A fixed, eagerly initialized event pool for one single-graph condition."""

    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]
    metadata: dict[str, object]
    replays: int
    event_batch_replays: int
    replays_per_reported_sample: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _manifest_path_for_key(cache: Path, key: str) -> Path:
    if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
        raise ValueError(f"cache key is not a lowercase SHA-256 digest: {key!r}")
    return cache / key[:2] / f"{key}.json"


def manifest_for_spec(
    cache: Path,
    spec_hash: str,
    *,
    cache_key: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    cache = cache.resolve()
    candidates = (
        [_manifest_path_for_key(cache, cache_key)]
        if cache_key is not None
        else sorted(cache.rglob("*.json"))
    )
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        if path.name.endswith(".ptx.json"):
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema") == "sparkinfer.cute.compile_manifest.v3"
            and manifest.get("compile_spec_hash") == spec_hash
            and manifest.get("object_sha256")
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one manifest for spec {spec_hash} in {cache}, "
            f"found {len(matches)}"
        )
    path, manifest = matches[0]
    manifest_key = str(manifest.get("cache_key", ""))
    if path.resolve() != _manifest_path_for_key(cache, manifest_key).resolve():
        raise RuntimeError(f"manifest path/cache-key mismatch: {path}")
    if cache_key is not None and manifest_key != cache_key:
        raise RuntimeError(
            f"requested cache key {cache_key}, manifest contains {manifest_key}"
        )
    return path, manifest


def artifact_provenance(
    cache: Path,
    spec_hash: str,
    *,
    cache_key: str | None = None,
) -> dict[str, Any]:
    """Resolve and verify one exact cache artifact without loading it.

    Release ABBA reports must retain both the manifest and object identities.
    Keeping this construction in one place prevents older producers from
    accidentally recording only the object digest (which is insufficient to
    prove the compile specification and toolchain remained immutable).
    """

    cache = cache.resolve()
    manifest_path, manifest = manifest_for_spec(
        cache,
        spec_hash,
        cache_key=cache_key,
    )
    object_path = manifest_path.with_suffix(".o")
    object_digest = sha256_file(object_path)
    if object_digest != manifest["object_sha256"]:
        raise RuntimeError(f"object digest mismatch: {object_path}")
    object_bytes = object_path.stat().st_size
    if "object_bytes" in manifest and object_bytes != int(manifest["object_bytes"]):
        raise RuntimeError(f"object byte-count mismatch: {object_path}")
    return {
        "cache": str(cache),
        "cache_key": manifest["cache_key"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "object_path": str(object_path),
        "object_sha256": object_digest,
        "object_bytes": object_bytes,
        "compile_spec_hash": manifest["compile_spec_hash"],
        "compile_spec_json": manifest["compile_spec_json"],
        "semantic_key": manifest["semantic_key"],
        "kernel_id": manifest["kernel_id"],
        "package_fingerprint": manifest["package_fingerprint"],
        "toolchain": manifest["toolchain"],
    }


def load_exact(
    cache: Path,
    spec_hash: str,
    *,
    cache_key: str | None = None,
) -> tuple[object, dict[str, Any]]:
    cache = cache.resolve()
    provenance = artifact_provenance(cache, spec_hash, cache_key=cache_key)
    object_path = Path(str(provenance["object_path"]))
    manifest_path = Path(str(provenance["manifest_path"]))
    key = str(provenance["cache_key"])
    # CUTLASS' ExternalBinaryModule may finalize/patch the ELF object while
    # loading it. Release evidence must remain immutable, so load a verified
    # staging copy and keep the manifest-bound source object untouched.
    with tempfile.TemporaryDirectory(prefix="sparkinfer-exact-cache-load-") as raw_stage:
        stage_cache = Path(raw_stage)
        stage_shard = stage_cache / key[:2]
        stage_shard.mkdir(parents=True)
        shutil.copy2(object_path, stage_shard / f"{key}.o")
        shutil.copy2(manifest_path, stage_shard / f"{key}.json")
        previous = os.environ.get("SPARKINFER_CUTE_COMPILE_CACHE_DIR")
        os.environ["SPARKINFER_CUTE_COMPILE_CACHE_DIR"] = str(stage_cache)
        try:
            compiled = cute_compiler._load_cute_compile_from_disk(key)
        finally:
            if previous is None:
                os.environ.pop("SPARKINFER_CUTE_COMPILE_CACHE_DIR", None)
            else:
                os.environ["SPARKINFER_CUTE_COMPILE_CACHE_DIR"] = previous
    if compiled is None:
        raise RuntimeError(f"failed to load exact cached object {object_path}")
    verify_artifact(provenance)
    return compiled, provenance


def verify_artifact(provenance: Mapping[str, Any]) -> dict[str, object]:
    manifest_path = Path(str(provenance["manifest_path"]))
    object_path = Path(str(provenance["object_path"]))
    observed = {
        "manifest_sha256": sha256_file(manifest_path),
        "object_sha256": sha256_file(object_path),
        "object_bytes": object_path.stat().st_size,
    }
    expected = {name: provenance[name] for name in observed}
    if observed != expected:
        raise RuntimeError(
            "cached artifact changed during benchmark: "
            f"expected={expected}, observed={observed}"
        )
    return {"passed": True, **observed}


def gpu_mode_snapshot(expected_physical_gpu: int) -> dict[str, object]:
    """Capture a complete physical-GPU mode snapshot or fail closed."""

    snapshot = nvidia_smi_gpu_mode_snapshot()
    fields = snapshot.get("fields")
    if not snapshot.get("available") or not isinstance(fields, Mapping):
        raise RuntimeError(f"physical GPU mode snapshot unavailable: {snapshot}")
    try:
        observed = int(str(fields["index"]))
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            f"GPU mode snapshot has no physical index: {snapshot}"
        ) from error
    if observed != expected_physical_gpu:
        raise RuntimeError(
            f"expected physical GPU {expected_physical_gpu}, observed {observed}"
        )
    return snapshot


@contextmanager
def pin_module_launches(
    module: ModuleType,
    compiled_by_spec: Mapping[str, object],
    observed_specs: list[str],
) -> Iterator[None]:
    """Replace one module-local ``sparkinfer_launch`` with exact-object execution."""

    original = module.sparkinfer_launch

    def pinned_launch(
        _kernel,
        *,
        compile_spec,
        compile_args,
        runtime_args,
    ) -> None:
        del compile_args
        spec_hash = str(compile_spec.hash_key)
        observed_specs.append(spec_hash)
        compiled = compiled_by_spec.get(spec_hash)
        if compiled is None:
            raise RuntimeError(
                f"production launcher requested unpinned compile spec {spec_hash}; "
                f"allowed={sorted(compiled_by_spec)}"
            )
        cute_compiler.run_compiled(compiled, runtime_args)

    module.sparkinfer_launch = pinned_launch
    try:
        yield
    finally:
        module.sparkinfer_launch = original


def graph_topology(graph: torch.cuda.CUDAGraph) -> dict[str, object]:
    raw_graph = graph.raw_cuda_graph()
    error, _, node_count = cuda_driver.cuGraphGetNodes(raw_graph)
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGraphGetNodes count failed: {error}")
    error, nodes, returned = cuda_driver.cuGraphGetNodes(raw_graph, node_count)
    if error != cuda_driver.CUresult.CUDA_SUCCESS or returned != node_count:
        raise RuntimeError(f"cuGraphGetNodes failed: {error}, {returned}/{node_count}")
    metadata: list[dict[str, object]] = []
    for index, node in enumerate(nodes):
        error, node_type = cuda_driver.cuGraphNodeGetType(node)
        if error != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphNodeGetType failed: {error}")
        item: dict[str, object] = {"index": index, "type": node_type.name}
        if node_type == cuda_driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            error, params = cuda_driver.cuGraphKernelNodeGetParams(node)
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuGraphKernelNodeGetParams failed: {error}")
            error, name = cuda_driver.cuFuncGetName(params.func)
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuFuncGetName failed: {error}")
            item.update(
                {
                    "kernel_name": name.decode(),
                    "grid": [params.gridDimX, params.gridDimY, params.gridDimZ],
                    "block": [params.blockDimX, params.blockDimY, params.blockDimZ],
                    "dynamic_smem_bytes": params.sharedMemBytes,
                }
            )
        metadata.append(item)
    return {
        "node_count": node_count,
        "kernel_node_count": sum(
            node["type"] == "CU_GRAPH_NODE_TYPE_KERNEL" for node in metadata
        ),
        "nodes": metadata,
    }


def topology_signature(topology: Mapping[str, Any]) -> dict[str, object]:
    return {
        "node_count": topology["node_count"],
        "kernel_node_count": topology["kernel_node_count"],
        "nodes": [
            {key: value for key, value in node.items() if key != "kernel_name"}
            for node in topology["nodes"]
        ],
    }


def allocator_counters() -> dict[str, int]:
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def summary(samples: list[float]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(samples)
    trim = int(0.01 * len(ordered))
    trimmed = ordered[trim:-trim] if trim else ordered
    return {
        "count": len(samples),
        "mean_us": statistics.mean(samples),
        "trimmed_mean_us": statistics.mean(trimmed),
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "p05_us": ordered[int(0.05 * (len(ordered) - 1))],
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))],
        "samples_us": samples,
    }


def _require_summary_matches(
    observed: Mapping[str, object],
    expected_samples: list[float],
    *,
    context: str,
) -> None:
    expected = summary(expected_samples)
    if observed.get("count") != expected["count"]:
        raise AssertionError(f"{context}: reported sample count is inconsistent")
    observed_samples = observed.get("samples_us")
    if not isinstance(observed_samples, list) or sorted(observed_samples) != sorted(
        expected_samples
    ):
        raise AssertionError(f"{context}: reported samples are inconsistent")
    for field in (
        "mean_us",
        "trimmed_mean_us",
        "median_us",
        "min_us",
        "p05_us",
        "p95_us",
    ):
        value = observed.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isclose(
                float(value),
                float(expected[field]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise AssertionError(f"{context}: {field} is inconsistent")


def validate_time_abba_aggregation(
    timing: Mapping[str, object],
    *,
    labels: tuple[str, str],
) -> bool:
    """Fail closed if aggregate timing cannot be reconstructed from raw events."""

    replays = timing.get("replays_per_reported_sample")
    if not isinstance(replays, int) or isinstance(replays, bool) or replays < 1:
        raise AssertionError("aggregate timing has an invalid replay count")
    policy = timing.get("aggregation")
    if not isinstance(policy, Mapping) or policy != {
        "reported_sample": "arithmetic_mean_us",
        "inner_event_bracketing": "independent_per_graph_replay",
        "inner_schedule": "full_abba_order_per_repetition",
        "flush_before_every_inner_replay": timing.get("cold_l2") is True,
        "flush_inside_timed_interval": False,
    }:
        raise AssertionError("aggregate timing policy is missing or inconsistent")
    event_pool = timing.get("event_pool")
    if not isinstance(event_pool, Mapping) or event_pool.get("schema") != (
        _ABBA_EVENT_POOL_SCHEMA
    ):
        raise AssertionError("aggregate timing lacks fixed event-pool provenance")
    event_batch_cycles = event_pool.get("event_batch_cycles")
    if (
        not isinstance(event_batch_cycles, int)
        or isinstance(event_batch_cycles, bool)
        or event_batch_cycles < 1
        or event_pool.get("allocation_phase") != "before_reported_samples"
        or event_pool.get("prewarm_phase") != "before_reported_samples"
        or event_pool.get("prewarm_each_event") is not True
        or event_pool.get("one_pair_per_inner_replay") is not True
        or event_pool.get("event_creation_inside_sample_schedule") is not False
        or event_pool.get("initialized_before_target_graph_preconditioning") is not True
        or event_pool.get("reuse_boundary")
        != "after_stream_synchronize_and_elapsed_query"
    ):
        raise AssertionError("aggregate timing event-pool policy is inconsistent")
    event_handle_sha256 = event_pool.get("event_handle_sha256")
    prewarm_elapsed_sha256 = event_pool.get("prewarm_elapsed_sha256")
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        for digest in (event_handle_sha256, prewarm_elapsed_sha256)
    ):
        raise AssertionError("aggregate timing event-pool digest is malformed")
    expected_orders = (
        (labels[0], labels[1], labels[1], labels[0]),
        (labels[1], labels[0], labels[0], labels[1]),
    )
    orders = timing.get("orders")
    if (
        not isinstance(orders, (list, tuple))
        or tuple(
            tuple(order) if isinstance(order, (list, tuple)) else () for order in orders
        )
        != expected_orders
    ):
        raise AssertionError("aggregate timing does not declare exact ABBA/BAAB orders")
    raw_by_position = timing.get("inner_samples_by_position")
    position_summaries = timing.get("position_summaries")
    summaries = timing.get("summaries")
    inner_counts = timing.get("inner_sample_count_by_label")
    if (
        not isinstance(raw_by_position, Mapping)
        or not isinstance(position_summaries, Mapping)
        or not isinstance(summaries, Mapping)
        or not isinstance(inner_counts, Mapping)
        or set(raw_by_position) != set(position_summaries)
        or set(summaries) != set(labels)
        or set(inner_counts) != set(labels)
    ):
        raise AssertionError("aggregate timing maps are incomplete")
    expected_position_keys = {
        f"{order_index}:{position}:{label}"
        for order_index, order in enumerate(expected_orders)
        for position, label in enumerate(order)
    }
    if set(raw_by_position) != expected_position_keys:
        raise AssertionError("aggregate timing does not contain all eight positions")

    reported_by_label: dict[str, list[float]] = {label: [] for label in labels}
    observed_inner_counts = {label: 0 for label in labels}
    position_counts: set[int] = set()
    for position_key, raw_groups in raw_by_position.items():
        if not isinstance(position_key, str):
            raise AssertionError("aggregate timing position key is not a string")
        label = position_key.rsplit(":", 1)[-1]
        if label not in reported_by_label:
            raise AssertionError(f"aggregate timing has unknown label {label!r}")
        position_summary = position_summaries[position_key]
        if not isinstance(position_summary, Mapping) or not isinstance(
            raw_groups, list
        ):
            raise AssertionError("aggregate timing position data is malformed")
        reported_samples = position_summary.get("samples_us")
        if not isinstance(reported_samples, list) or len(raw_groups) != len(
            reported_samples
        ):
            raise AssertionError("aggregate timing position counts differ")
        recomputed_reported: list[float] = []
        for reported_sample, raw_group in zip(
            reported_samples, raw_groups, strict=True
        ):
            if (
                not isinstance(raw_group, list)
                or len(raw_group) != replays
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in raw_group
                )
            ):
                raise AssertionError("aggregate timing inner sample group is invalid")
            recomputed = statistics.mean(float(value) for value in raw_group)
            if (
                not isinstance(reported_sample, (int, float))
                or isinstance(reported_sample, bool)
                or not math.isclose(
                    float(reported_sample),
                    recomputed,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ):
                raise AssertionError(
                    "aggregate timing reported sample is not the inner mean"
                )
            recomputed_reported.append(recomputed)
        _require_summary_matches(
            position_summary,
            recomputed_reported,
            context=f"aggregate timing position {position_key}",
        )
        position_counts.add(len(recomputed_reported))
        reported_by_label[label].extend(recomputed_reported)
        observed_inner_counts[label] += len(raw_groups) * replays

    if len(position_counts) != 1 or next(iter(position_counts)) <= 0:
        raise AssertionError("aggregate timing position/order counts are unbalanced")

    position_count = next(iter(position_counts))
    total_cycles = 2 * position_count
    batch_cycle_capacity = min(total_cycles, event_batch_cycles)
    pair_count = batch_cycle_capacity * len(expected_orders[0]) * replays
    batch_count = math.ceil(total_cycles / event_batch_cycles)
    expected_pool_counts = {
        "batch_cycle_capacity": batch_cycle_capacity,
        "pair_count": pair_count,
        "event_count": 2 * pair_count,
        "unique_event_handle_count": 2 * pair_count,
        "prewarm_elapsed_query_count": pair_count,
        "batch_count": batch_count,
        "reuse_count": max(0, batch_count - 1),
    }
    if any(
        event_pool.get(field) != value for field, value in expected_pool_counts.items()
    ):
        raise AssertionError("aggregate timing event-pool counts are inconsistent")

    for label in labels:
        label_summary = summaries[label]
        if not isinstance(label_summary, Mapping):
            raise AssertionError(f"aggregate timing summary for {label} is malformed")
        _require_summary_matches(
            label_summary,
            reported_by_label[label],
            context=f"aggregate timing label {label}",
        )
        if inner_counts[label] != observed_inner_counts[label]:
            raise AssertionError(f"aggregate timing inner count for {label} differs")
    return True


def _prepare_abba_event_pool(
    *,
    cycles: int,
    event_batch_cycles: int,
    replays_per_reported_sample: int,
    stream: torch.cuda.Stream,
) -> _PreparedAbbaEventPool:
    """Allocate and force lazy CUDA-event initialization before GPU preconditioning."""

    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 2:
        raise ValueError("cycles must be an even integer of at least two")
    if cycles % 2 != 0:
        raise ValueError("cycles must be even for balanced ABBA/BAAB positions")
    if (
        not isinstance(event_batch_cycles, int)
        or isinstance(event_batch_cycles, bool)
        or event_batch_cycles < 1
    ):
        raise ValueError("event_batch_cycles must be a positive integer")
    if (
        not isinstance(replays_per_reported_sample, int)
        or isinstance(replays_per_reported_sample, bool)
        or replays_per_reported_sample < 1
    ):
        raise ValueError("replays_per_reported_sample must be a positive integer")
    batch_cycle_capacity = min(cycles, event_batch_cycles)
    if batch_cycle_capacity < 1:
        raise ValueError("cycles and event_batch_cycles must both be positive")
    positions_per_cycle = 4
    pool_pair_count = (
        batch_cycle_capacity * positions_per_cycle * replays_per_reported_sample
    )
    event_pairs = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(pool_pair_count)
    ]
    # CUDA event construction is lazy. Record and query every event now, before
    # target-graph duration/P1 preconditioning. Sampling can then follow the
    # final warmup and mode snapshot without an event-initialization idle gap.
    with torch.cuda.stream(stream):
        for start, end in event_pairs:
            start.record(stream)
            end.record(stream)
    stream.synchronize()
    prewarm_elapsed_us = [
        start.elapsed_time(end) * 1000.0 for start, end in event_pairs
    ]
    if any(
        not isinstance(value, float) or not math.isfinite(value) or value < 0.0
        for value in prewarm_elapsed_us
    ):
        raise AssertionError("CUDA event-pool prewarm emitted an invalid elapsed time")
    event_handles = [
        int(handle)
        for pair in event_pairs
        for handle in (pair[0].cuda_event, pair[1].cuda_event)
    ]
    if len(set(event_handles)) != len(event_handles):
        raise AssertionError("CUDA event pool contains duplicate event handles")
    return _PreparedAbbaEventPool(
        event_pairs=event_pairs,
        metadata={
            "schema": _ABBA_EVENT_POOL_SCHEMA,
            "allocation_phase": "before_reported_samples",
            "prewarm_phase": "before_reported_samples",
            "prewarm_each_event": True,
            "one_pair_per_inner_replay": True,
            "event_creation_inside_sample_schedule": False,
            "initialized_before_target_graph_preconditioning": True,
            "reuse_boundary": "after_stream_synchronize_and_elapsed_query",
            "event_batch_cycles": event_batch_cycles,
            "batch_cycle_capacity": batch_cycle_capacity,
            "pair_count": pool_pair_count,
            "event_count": len(event_handles),
            "unique_event_handle_count": len(set(event_handles)),
            "event_handle_sha256": json_sha256(event_handles),
            "prewarm_elapsed_query_count": len(prewarm_elapsed_us),
            "prewarm_elapsed_sha256": json_sha256(prewarm_elapsed_us),
        },
        cycles=cycles,
        event_batch_cycles=event_batch_cycles,
        replays_per_reported_sample=replays_per_reported_sample,
    )


def time_abba(
    graphs: Mapping[str, torch.cuda.CUDAGraph],
    *,
    labels: tuple[str, str],
    cycles: int,
    event_batch_cycles: int,
    stream: torch.cuda.Stream,
    flush: Callable[[], None] | None,
    prepared_event_pool: _PreparedAbbaEventPool,
    replays_per_reported_sample: int = 1,
) -> dict[str, object]:
    if (
        not isinstance(replays_per_reported_sample, int)
        or isinstance(replays_per_reported_sample, bool)
        or replays_per_reported_sample < 1
    ):
        raise ValueError("replays_per_reported_sample must be a positive integer")
    if (
        prepared_event_pool.cycles != cycles
        or prepared_event_pool.event_batch_cycles != event_batch_cycles
        or prepared_event_pool.replays_per_reported_sample
        != replays_per_reported_sample
    ):
        raise ValueError("prepared CUDA event pool does not match timing parameters")
    orders = (
        (labels[0], labels[1], labels[1], labels[0]),
        (labels[1], labels[0], labels[0], labels[1]),
    )
    by_label = {label: [] for label in labels}
    by_position = {
        f"{order_index}:{position}:{label}": []
        for order_index, order in enumerate(orders)
        for position, label in enumerate(order)
    }
    inner_samples_by_position: dict[str, list[list[float]]] = {
        key: [] for key in by_position
    }
    event_pairs = prepared_event_pool.event_pairs

    batch_count = 0
    for batch_start in range(0, cycles, event_batch_cycles):
        records: list[tuple[str, str, list[int]]] = []
        next_pair = 0
        with torch.cuda.stream(stream):
            for cycle in range(
                batch_start,
                min(cycles, batch_start + event_batch_cycles),
            ):
                order_index = cycle & 1
                order = orders[order_index]
                cycle_records: list[list[int]] = [[] for _ in order]
                for _ in range(replays_per_reported_sample):
                    for position, label in enumerate(order):
                        if flush is not None:
                            flush()
                        if next_pair >= len(event_pairs):
                            raise AssertionError(
                                "CUDA event pool exhausted within one batch"
                            )
                        start, end = event_pairs[next_pair]
                        start.record(stream)
                        graphs[label].replay()
                        end.record(stream)
                        cycle_records[position].append(next_pair)
                        next_pair += 1
                for position, label in enumerate(order):
                    records.append(
                        (
                            f"{order_index}:{position}:{label}",
                            label,
                            cycle_records[position],
                        )
                    )
        stream.synchronize()
        for key, label, event_indices in records:
            inner_us = [
                event_pairs[index][0].elapsed_time(event_pairs[index][1]) * 1000.0
                for index in event_indices
            ]
            elapsed_us = statistics.mean(inner_us)
            inner_samples_by_position[key].append(inner_us)
            by_position[key].append(elapsed_us)
            by_label[label].append(elapsed_us)
        batch_count += 1
    summaries = {label: summary(values) for label, values in by_label.items()}
    result = {
        "orders": orders,
        "cold_l2": flush is not None,
        "replays_per_reported_sample": replays_per_reported_sample,
        "aggregation": {
            "reported_sample": "arithmetic_mean_us",
            "inner_event_bracketing": "independent_per_graph_replay",
            "inner_schedule": "full_abba_order_per_repetition",
            "flush_before_every_inner_replay": flush is not None,
            "flush_inside_timed_interval": False,
        },
        "event_pool": {
            **prepared_event_pool.metadata,
            "batch_count": batch_count,
            "reuse_count": max(0, batch_count - 1),
        },
        "inner_samples_by_position": inner_samples_by_position,
        "inner_sample_count_by_label": {
            label: len(by_label[label]) * replays_per_reported_sample
            for label in labels
        },
        "summaries": summaries,
        "position_summaries": {
            key: summary(values) for key, values in by_position.items()
        },
        "ratios_b_over_a": {
            metric: float(summaries[labels[1]][f"{metric}_us"])
            / float(summaries[labels[0]][f"{metric}_us"])
            for metric in ("mean", "trimmed_mean", "median")
        },
    }
    validate_time_abba_aggregation(result, labels=labels)
    return result


def _mode_clock_mhz(snapshot: Mapping[str, object], field: str) -> float:
    fields = snapshot.get("fields")
    if snapshot.get("available") is not True or not isinstance(fields, Mapping):
        raise AssertionError(f"GPU mode snapshot is unavailable: {snapshot}")
    raw = fields.get(field)
    try:
        value = float(str(raw).split()[0])
    except (IndexError, ValueError) as error:
        raise AssertionError(
            f"GPU mode snapshot has invalid {field}: {raw!r}"
        ) from error
    if not math.isfinite(value) or value <= 0.0:
        raise AssertionError(f"GPU mode snapshot has invalid {field}: {raw!r}")
    return value


def _mode_active_throttle_reasons(snapshot: Mapping[str, object]) -> int:
    fields = snapshot.get("fields")
    if snapshot.get("available") is not True or not isinstance(fields, Mapping):
        raise AssertionError(f"GPU mode snapshot is unavailable: {snapshot}")
    raw = fields.get("clocks_throttle_reasons.active")
    try:
        value = int(str(raw).strip(), 0)
    except ValueError as error:
        raise AssertionError(
            f"GPU mode snapshot has invalid active throttle reasons: {raw!r}"
        ) from error
    if value < 0:
        raise AssertionError(
            f"GPU mode snapshot has negative active throttle reasons: {raw!r}"
        )
    return value


def _validate_required_active_throttle_reasons(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value not in _PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS
    ):
        raise ValueError(
            "required_active_throttle_reasons must be exactly 0 or 0x4 "
            "(NVIDIA SW power cap)"
        )
    return value


def _allowed_active_throttle_reasons(
    required_active_throttle_reasons: object,
    *,
    allow_sw_power_cap_transition: bool,
) -> tuple[int, ...]:
    required = _validate_required_active_throttle_reasons(
        required_active_throttle_reasons
    )
    if not isinstance(allow_sw_power_cap_transition, bool):
        raise ValueError("allow_sw_power_cap_transition must be a boolean")
    if allow_sw_power_cap_transition:
        return tuple(sorted(_PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS))
    return (required,)


def timing_mode_policy(
    *,
    required_pstate: str,
    required_active_throttle_reasons: int,
    max_sm_clock_delta_mhz: float,
    allow_sw_power_cap_transition: bool = False,
) -> dict[str, object]:
    """Return the exact, fail-closed GPU mode policy bound into timing results."""

    if required_pstate != "P1":
        raise ValueError("required_pstate must be P1")
    required_throttle_reasons = _validate_required_active_throttle_reasons(
        required_active_throttle_reasons
    )
    allowed_throttle_reasons = _allowed_active_throttle_reasons(
        required_throttle_reasons,
        allow_sw_power_cap_transition=allow_sw_power_cap_transition,
    )
    if (
        not isinstance(max_sm_clock_delta_mhz, (int, float))
        or isinstance(max_sm_clock_delta_mhz, bool)
        or not math.isfinite(float(max_sm_clock_delta_mhz))
        or float(max_sm_clock_delta_mhz) <= 0.0
        or float(max_sm_clock_delta_mhz) > 60.0
    ):
        raise ValueError("max_sm_clock_delta_mhz must be in (0, 60]")
    return {
        "schema": GPU_TIMING_MODE_POLICY_SCHEMA,
        "required_pstate": required_pstate,
        "required_active_throttle_reasons": required_throttle_reasons,
        "active_throttle_reasons_match": (
            "supported-sw-power-cap-transition"
            if allow_sw_power_cap_transition
            else "exact"
        ),
        "allowed_observed_active_throttle_reasons": list(allowed_throttle_reasons),
        "allow_sw_power_cap_transition": allow_sw_power_cap_transition,
        "permitted_required_active_throttle_reasons": sorted(
            _PERMITTED_REQUIRED_ACTIVE_THROTTLE_REASONS
        ),
        "required_memory_clock_equality": True,
        "max_sm_clock_delta_mhz": float(max_sm_clock_delta_mhz),
    }


def _validate_timing_mode_stability(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    required_pstate: str,
    required_active_throttle_reasons: int,
    max_sm_clock_delta_mhz: float,
    allow_sw_power_cap_transition: bool = False,
) -> dict[str, object]:
    before_fields = before.get("fields")
    after_fields = after.get("fields")
    if not isinstance(before_fields, Mapping) or not isinstance(after_fields, Mapping):
        raise AssertionError("timing mode snapshots lack physical-GPU fields")
    stable_fields = (
        "index",
        "uuid",
        "persistence_mode",
        "compute_mode",
        "power.limit",
    )
    required_throttle_reasons = _validate_required_active_throttle_reasons(
        required_active_throttle_reasons
    )
    allowed_throttle_reasons = _allowed_active_throttle_reasons(
        required_throttle_reasons,
        allow_sw_power_cap_transition=allow_sw_power_cap_transition,
    )
    before_throttle_reasons = _mode_active_throttle_reasons(before)
    after_throttle_reasons = _mode_active_throttle_reasons(after)
    if (
        before_throttle_reasons not in allowed_throttle_reasons
        or after_throttle_reasons not in allowed_throttle_reasons
    ):
        if allow_sw_power_cap_transition:
            raise AssertionError(
                "timing observed an unsupported active clock-throttle reasons "
                "mask: "
                f"allowed={[hex(value) for value in allowed_throttle_reasons]}, "
                f"before={before_throttle_reasons:#x}, "
                f"after={after_throttle_reasons:#x}"
            )
        raise AssertionError(
            "timing requires the exact active clock-throttle reasons mask "
            f"{required_throttle_reasons:#x}: "
            f"before={before_throttle_reasons:#x}, "
            f"after={after_throttle_reasons:#x}"
        )
    if any(
        before_fields.get(field) != after_fields.get(field) for field in stable_fields
    ):
        raise AssertionError("physical GPU mode changed across timing")
    if (
        before_fields.get("pstate") != required_pstate
        or after_fields.get("pstate") != required_pstate
    ):
        raise AssertionError(
            "timing requires stable "
            f"{required_pstate}: before={before_fields.get('pstate')}, "
            f"after={after_fields.get('pstate')}"
        )
    before_sm = _mode_clock_mhz(before, "clocks.current.sm")
    after_sm = _mode_clock_mhz(after, "clocks.current.sm")
    before_memory = _mode_clock_mhz(before, "clocks.current.memory")
    after_memory = _mode_clock_mhz(after, "clocks.current.memory")
    sm_delta = abs(after_sm - before_sm)
    if sm_delta > max_sm_clock_delta_mhz:
        raise AssertionError(
            "SM clock changed across timing: "
            f"before={before_sm} MHz, after={after_sm} MHz, "
            f"limit={max_sm_clock_delta_mhz} MHz"
        )
    if before_memory != after_memory:
        raise AssertionError(
            "memory clock changed across timing: "
            f"before={before_memory} MHz, after={after_memory} MHz"
        )
    before_ns = before.get("captured_unix_ns")
    after_ns = after.get("captured_unix_ns")
    if (
        not isinstance(before_ns, int)
        or isinstance(before_ns, bool)
        or not isinstance(after_ns, int)
        or isinstance(after_ns, bool)
        or after_ns <= before_ns
    ):
        raise AssertionError("timing mode snapshot timestamps are not ordered")
    return {
        "schema": _ABBA_MODE_STABILITY_SCHEMA,
        "required_pstate": required_pstate,
        "required_memory_clock_equality": True,
        "max_sm_clock_delta_mhz": max_sm_clock_delta_mhz,
        "observed_sm_clock_delta_mhz": sm_delta,
        "observed_before_sm_clock_mhz": before_sm,
        "observed_after_sm_clock_mhz": after_sm,
        "observed_memory_clock_mhz": before_memory,
        "required_active_throttle_reasons": required_throttle_reasons,
        "allowed_observed_active_throttle_reasons": list(allowed_throttle_reasons),
        "allow_sw_power_cap_transition": allow_sw_power_cap_transition,
        "active_throttle_reasons_transition_observed": (
            before_throttle_reasons != after_throttle_reasons
        ),
        "observed_before_active_throttle_reasons": before_throttle_reasons,
        "observed_after_active_throttle_reasons": after_throttle_reasons,
        "stable_identity_and_mode_fields": list(stable_fields),
        "passed": True,
    }


def _balanced_duration_precondition(
    graphs: Mapping[str, torch.cuda.CUDAGraph],
    *,
    labels: tuple[str, str],
    minimum_cycles: int,
    minimum_seconds: float,
    maximum_seconds: float,
    stream: torch.cuda.Stream,
    flush: Callable[[], None] | None,
    mode_snapshot: Callable[[], Mapping[str, object]] | None,
    required_pstate: str | None,
    required_active_throttle_reasons: int = 0,
    allow_sw_power_cap_transition: bool = False,
) -> tuple[dict[str, object], Mapping[str, object] | None]:
    if minimum_cycles < 0 or minimum_seconds < 0.0:
        raise ValueError("preconditioning cycle/time minima must be nonnegative")
    if maximum_seconds <= 0.0 or maximum_seconds < minimum_seconds:
        raise ValueError("maximum preconditioning duration is invalid")
    required_throttle_reasons = _validate_required_active_throttle_reasons(
        required_active_throttle_reasons
    )
    allowed_throttle_reasons = _allowed_active_throttle_reasons(
        required_throttle_reasons,
        allow_sw_power_cap_transition=allow_sw_power_cap_transition,
    )
    if mode_snapshot is None and (
        required_pstate is not None or required_throttle_reasons != 0
    ):
        raise ValueError("GPU timing-mode requirements need a physical-GPU callback")
    orders = (
        (labels[0], labels[1], labels[1], labels[0]),
        (labels[1], labels[0], labels[0], labels[1]),
    )
    completed_cycles = 0
    active_seconds = 0.0
    mode_probes: list[Mapping[str, object]] = []
    mode_before: Mapping[str, object] | None = None
    batch_cycles = max(1, min(16, minimum_cycles or 1))
    batch_cycle_counts: list[int] = []
    batch_active_seconds: list[float] = []
    seconds_per_cycle: float | None = None
    while True:
        remaining_cycles = max(0, minimum_cycles - completed_cycles)
        current_batch_cycles = max(
            1, min(batch_cycles, remaining_cycles or batch_cycles)
        )
        remaining_budget = maximum_seconds - active_seconds
        if seconds_per_cycle is not None:
            if seconds_per_cycle >= remaining_budget:
                raise RuntimeError(
                    "duration preconditioning cannot safely continue within its "
                    f"maximum: observed={active_seconds:.6f}s, "
                    f"maximum={maximum_seconds:.6f}s"
                )
            safe_cycles = max(
                1,
                math.floor(0.8 * remaining_budget / seconds_per_cycle),
            )
            current_batch_cycles = min(current_batch_cycles, safe_cycles)
        batch_started = time.monotonic()
        with torch.cuda.stream(stream):
            for cycle_offset in range(current_batch_cycles):
                order = orders[(completed_cycles + cycle_offset) & 1]
                for label in order:
                    if flush is not None:
                        flush()
                    graphs[label].replay()
        stream.synchronize()
        batch_seconds = time.monotonic() - batch_started
        active_seconds += batch_seconds
        completed_cycles += current_batch_cycles
        batch_cycle_counts.append(current_batch_cycles)
        batch_active_seconds.append(batch_seconds)
        seconds_per_cycle = batch_seconds / current_batch_cycles
        if active_seconds > maximum_seconds:
            raise RuntimeError(
                "duration preconditioning exceeded its maximum: "
                f"observed={active_seconds:.6f}s, maximum={maximum_seconds:.6f}s"
            )
        if completed_cycles < minimum_cycles or active_seconds < minimum_seconds:
            # Grow from measured runtime rather than jumping directly to 1,024
            # cycles. This keeps long or cold-L2 graphs inside maximum_seconds
            # while still reaching the duration floor efficiently for short
            # kernels.
            remaining_cycles = max(0, minimum_cycles - completed_cycles)
            remaining_seconds = max(0.0, minimum_seconds - active_seconds)
            remaining_budget = maximum_seconds - active_seconds
            if (
                remaining_cycles
                and seconds_per_cycle * remaining_cycles > remaining_budget
            ):
                raise RuntimeError(
                    "duration preconditioning cannot meet its cycle minimum "
                    "within the maximum: "
                    f"remaining_cycles={remaining_cycles}, "
                    f"observed={active_seconds:.6f}s, "
                    f"maximum={maximum_seconds:.6f}s"
                )
            cycles_for_duration = (
                math.ceil(remaining_seconds / seconds_per_cycle)
                if remaining_seconds > 0.0
                else 1
            )
            batch_cycles = min(
                1024,
                max(
                    batch_cycles * 2,
                    min(cycles_for_duration, batch_cycles * 4),
                ),
            )
            continue
        if mode_snapshot is not None:
            probe = mode_snapshot()
            mode_probes.append(probe)
            fields = probe.get("fields")
            observed_pstate = (
                fields.get("pstate") if isinstance(fields, Mapping) else None
            )
            observed_throttle_reasons = _mode_active_throttle_reasons(probe)
            if (
                required_pstate is not None and observed_pstate != required_pstate
            ) or observed_throttle_reasons not in allowed_throttle_reasons:
                if active_seconds >= maximum_seconds:
                    raise RuntimeError(
                        "target graph did not reach the required GPU timing mode "
                        f"after {active_seconds:.3f}s: "
                        f"pstate={observed_pstate!r}, "
                        f"active_throttle_reasons={observed_throttle_reasons:#x}, "
                        "allowed="
                        f"{[hex(value) for value in allowed_throttle_reasons]}"
                    )
                # Once the duration/cycle minima are met, probe the requested
                # mode after small balanced batches instead of replaying a
                # potentially expensive 1,024-cycle block.
                batch_cycles = min(16, batch_cycles)
                continue
            mode_before = probe
        break
    replay_count = completed_cycles * len(orders[0]) // 2
    return (
        {
            "policy": "balanced_abba_target_graph_duration",
            "minimum_cycles": minimum_cycles,
            "minimum_active_seconds": minimum_seconds,
            "maximum_active_seconds": maximum_seconds,
            "completed_cycles": completed_cycles,
            "observed_active_seconds": active_seconds,
            "batch_cycle_counts": batch_cycle_counts,
            "batch_active_seconds": batch_active_seconds,
            "target_graph_replays_by_label": {
                labels[0]: replay_count,
                labels[1]: replay_count,
            },
            "cold_l2_flush_before_every_replay": flush is not None,
            "flush_inside_timed_interval": False,
            "required_pstate": required_pstate,
            "required_active_throttle_reasons": required_throttle_reasons,
            "allowed_observed_active_throttle_reasons": list(allowed_throttle_reasons),
            "allow_sw_power_cap_transition": allow_sw_power_cap_transition,
            "mode_probes": mode_probes,
        },
        mode_before,
    )


def time_conditions(
    graphs: Mapping[str, torch.cuda.CUDAGraph],
    *,
    labels: tuple[str, str],
    precondition: int,
    warmup: int,
    cycles: int,
    event_batch_cycles: int,
    stream: torch.cuda.Stream,
    cold_l2: bool,
    l2_flush_bytes: int,
    replays_per_reported_sample: int = 1,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    mode_snapshot: Callable[[], Mapping[str, object]],
    required_pstate: str,
    max_sm_clock_delta_mhz: float,
    required_active_throttle_reasons: int = 0,
    allow_sw_power_cap_transition: bool = False,
) -> dict[str, object]:
    if (
        not isinstance(precondition, int)
        or isinstance(precondition, bool)
        or precondition < 1
    ):
        raise ValueError("precondition must be a positive integer")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 1:
        raise ValueError("warmup must be a positive integer")
    if (
        not isinstance(precondition_seconds, (int, float))
        or isinstance(precondition_seconds, bool)
        or not math.isfinite(float(precondition_seconds))
        or float(precondition_seconds) < 5.0
    ):
        raise ValueError("precondition_seconds must be at least 5 seconds")
    if (
        not isinstance(maximum_precondition_seconds, (int, float))
        or isinstance(maximum_precondition_seconds, bool)
        or not math.isfinite(float(maximum_precondition_seconds))
        or float(maximum_precondition_seconds) < float(precondition_seconds)
        or float(maximum_precondition_seconds) > 60.0
    ):
        raise ValueError(
            "maximum_precondition_seconds must cover the minimum and be at most 60"
        )
    if not callable(mode_snapshot):
        raise ValueError("mode_snapshot must be an explicit physical-GPU callback")
    mode_policy = timing_mode_policy(
        required_pstate=required_pstate,
        required_active_throttle_reasons=required_active_throttle_reasons,
        max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
        allow_sw_power_cap_transition=allow_sw_power_cap_transition,
    )
    required_throttle_reasons = int(mode_policy["required_active_throttle_reasons"])
    order = (labels[0], labels[1], labels[1], labels[0])
    condition_flushes: list[tuple[str, Callable[[], None] | None]] = [("warm_l2", None)]
    if cold_l2:
        condition_flushes.append(("cold_l2", make_l2_flush_fn(True, l2_flush_bytes)))
    results: dict[str, object] = {}
    for condition, flush in condition_flushes:
        # Allocate, record, synchronize, and query every timing event before
        # target-graph duration/P1 conditioning. Event creation itself is lazy
        # and must not introduce an idle or initialization gap after the final
        # warmup/mode snapshot or inside a reported replay interval.
        prepared_event_pool = _prepare_abba_event_pool(
            cycles=cycles,
            event_batch_cycles=event_batch_cycles,
            replays_per_reported_sample=replays_per_reported_sample,
            stream=stream,
        )
        preconditioning, _ = _balanced_duration_precondition(
            graphs,
            labels=labels,
            minimum_cycles=precondition,
            minimum_seconds=precondition_seconds,
            maximum_seconds=maximum_precondition_seconds,
            stream=stream,
            flush=flush,
            mode_snapshot=mode_snapshot,
            required_pstate=required_pstate,
            required_active_throttle_reasons=required_throttle_reasons,
            allow_sw_power_cap_transition=allow_sw_power_cap_transition,
        )
        with torch.cuda.stream(stream):
            for warm_cycle in range(warmup):
                cycle_order = order if warm_cycle % 2 == 0 else tuple(reversed(order))
                for label in cycle_order:
                    if flush is not None:
                        flush()
                    graphs[label].replay()
        stream.synchronize()
        mode_before = mode_snapshot() if mode_snapshot is not None else None
        before = allocator_counters()
        timing = time_abba(
            graphs,
            labels=labels,
            cycles=cycles,
            event_batch_cycles=event_batch_cycles,
            stream=stream,
            flush=flush,
            prepared_event_pool=prepared_event_pool,
            replays_per_reported_sample=replays_per_reported_sample,
        )
        after = allocator_counters()
        mode_after = mode_snapshot() if mode_snapshot is not None else None
        if before != after:
            raise AssertionError(
                f"CUDA allocator state changed during {condition} timing: "
                f"before={before}, after={after}"
            )
        condition_result: dict[str, object] = {
            "cold_l2": flush is not None,
            "l2_flush_bytes": (
                resolve_l2_flush_bytes(l2_flush_bytes) if flush is not None else 0
            ),
            "preconditioning": preconditioning,
            "allocator_before": before,
            "allocator_after": after,
            "allocator_stable": True,
            "timings": timing,
        }
        if mode_before is not None and mode_after is not None:
            if required_pstate is None:
                raise AssertionError("mode snapshots require an explicit pstate policy")
            condition_result.update(
                {
                    "gpu_mode_before_timing": mode_before,
                    "gpu_mode_after_timing": mode_after,
                    "gpu_mode_stability": _validate_timing_mode_stability(
                        mode_before,
                        mode_after,
                        required_pstate=required_pstate,
                        required_active_throttle_reasons=(required_throttle_reasons),
                        max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
                        allow_sw_power_cap_transition=(allow_sw_power_cap_transition),
                    ),
                }
            )
        results[condition] = condition_result
    return results


def _single_arm_runtime_environment() -> tuple[str, str]:
    """Return raw and comparison hashes for the complete benchmark controls."""

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
    raw_payload: dict[str, object] = {
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
    raw_sha256 = json_sha256(raw_payload)
    comparison_payload = json.loads(json.dumps(raw_payload))
    comparison_variables = comparison_payload["set_variables"]
    if not isinstance(comparison_variables, dict):
        raise AssertionError("runtime environment set-variable map changed type")
    path_states: dict[str, dict[str, object]] = {}
    for name in _RUNTIME_ENVIRONMENT_OPERATIONAL_PATH_EXCEPTIONS:
        raw_value = comparison_variables.pop(name, None)
        if raw_value is None:
            path_states[name] = {"status": "missing"}
        elif name == "CUTE_DSL_LIBS":
            path_states[name] = {
                "status": "set",
                "library_basenames": [
                    Path(component).name
                    for component in str(raw_value).split(":")
                    if component
                ],
            }
        else:
            path_states[name] = {"status": "set"}
    comparison_payload["operational_path_exceptions"] = {
        "policy": "values-normalized; set/missing state remains exact",
        "names": list(_RUNTIME_ENVIRONMENT_OPERATIONAL_PATH_EXCEPTIONS),
        "states": path_states,
    }
    return raw_sha256, json_sha256(comparison_payload)


def _single_arm_cutlass_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for package in _CUTLASS_PACKAGE_NAMES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    return packages


def source_ptxas_version(provenance: Mapping[str, Any]) -> str:
    """Read the source PTXAS version bound to one exact cache artifact."""

    manifest_path = Path(str(provenance["manifest_path"]))
    sidecar_path = manifest_path.with_name(f"{manifest_path.stem}.ptx.json")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"exact-object frontend PTX sidecar is unavailable: {sidecar_path}: {exc}"
        ) from exc
    if (
        sidecar.get("schema") != "sparkinfer.cute.frontend_ptx.v3"
        or sidecar.get("cache_key") != provenance["cache_key"]
        or sidecar.get("compile_spec_hash") != provenance["compile_spec_hash"]
        or sidecar.get("package_fingerprint") != provenance["package_fingerprint"]
    ):
        raise RuntimeError(f"frontend PTX sidecar binding mismatch: {sidecar_path}")
    compile_manifest = sidecar.get("compile_manifest")
    if (
        not isinstance(compile_manifest, dict)
        or compile_manifest.get("sha256") != provenance["manifest_sha256"]
    ):
        raise RuntimeError(
            f"frontend PTX sidecar manifest hash mismatch: {sidecar_path}"
        )
    source_ptxas = sidecar.get("source_ptxas")
    version = source_ptxas.get("version") if isinstance(source_ptxas, dict) else None
    if not isinstance(version, str) or not version:
        raise RuntimeError(
            f"frontend PTX sidecar lacks source PTXAS version: {sidecar_path}"
        )
    return version


def single_graph_topology(graph: torch.cuda.CUDAGraph) -> dict[str, object]:
    topology = graph_topology(graph)
    signature = topology_signature(topology)
    return {
        "node_count": int(signature["node_count"]),
        "kernel_node_count": int(signature["kernel_node_count"]),
        "topology_sha256": json_sha256(signature),
    }


def _prepare_single_graph_event_pool(
    *,
    replays: int,
    event_batch_replays: int,
    replays_per_reported_sample: int,
    stream: torch.cuda.Stream,
) -> _PreparedSingleGraphEventPool:
    """Fully initialize one fixed timing-event pool before target conditioning."""

    if not isinstance(replays, int) or isinstance(replays, bool) or replays < 1:
        raise ValueError("replays must be a positive integer")
    if (
        not isinstance(event_batch_replays, int)
        or isinstance(event_batch_replays, bool)
        or event_batch_replays < 1
    ):
        raise ValueError("event_batch_replays must be a positive integer")
    if (
        not isinstance(replays_per_reported_sample, int)
        or isinstance(replays_per_reported_sample, bool)
        or replays_per_reported_sample < 1
    ):
        raise ValueError("replays_per_reported_sample must be a positive integer")
    batch_replay_capacity = min(replays, event_batch_replays)
    pair_count = batch_replay_capacity * replays_per_reported_sample
    event_pairs = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(pair_count)
    ]
    with torch.cuda.stream(stream):
        for start, end in event_pairs:
            start.record(stream)
            end.record(stream)
    stream.synchronize()
    prewarm_elapsed_us = [
        start.elapsed_time(end) * 1000.0 for start, end in event_pairs
    ]
    if any(
        not isinstance(value, float) or not math.isfinite(value) or value < 0.0
        for value in prewarm_elapsed_us
    ):
        raise AssertionError(
            "single-graph CUDA event-pool prewarm emitted invalid elapsed time"
        )
    event_handles = [
        int(handle)
        for pair in event_pairs
        for handle in (pair[0].cuda_event, pair[1].cuda_event)
    ]
    if len(set(event_handles)) != len(event_handles):
        raise AssertionError("single-graph CUDA event pool has duplicate handles")
    batch_count = math.ceil(replays / event_batch_replays)
    return _PreparedSingleGraphEventPool(
        event_pairs=event_pairs,
        metadata={
            "schema": _ABBA_EVENT_POOL_SCHEMA,
            "allocation_phase": "before_reported_samples",
            "prewarm_phase": "before_reported_samples",
            "prewarm_each_event": True,
            "one_pair_per_inner_replay": True,
            "event_creation_inside_sample_schedule": False,
            "initialized_before_target_graph_preconditioning": True,
            "reuse_boundary": "after_stream_synchronize_and_elapsed_query",
            "event_batch_replays": event_batch_replays,
            "batch_replay_capacity": batch_replay_capacity,
            "pair_count": pair_count,
            "event_count": len(event_handles),
            "unique_event_handle_count": len(set(event_handles)),
            "event_handle_sha256": json_sha256(event_handles),
            "prewarm_elapsed_query_count": len(prewarm_elapsed_us),
            "prewarm_elapsed_sha256": json_sha256(prewarm_elapsed_us),
            "batch_count": batch_count,
            "reuse_count": max(0, batch_count - 1),
        },
        replays=replays,
        event_batch_replays=event_batch_replays,
        replays_per_reported_sample=replays_per_reported_sample,
    )


def _single_graph_duration_precondition(
    graph: torch.cuda.CUDAGraph,
    *,
    minimum_replays: int,
    minimum_seconds: float,
    maximum_seconds: float,
    stream: torch.cuda.Stream,
    flush: Callable[[], None] | None,
    mode_snapshot: Callable[[], Mapping[str, object]],
    required_pstate: str,
    required_active_throttle_reasons: int,
) -> dict[str, object]:
    required_throttle_reasons = _validate_required_active_throttle_reasons(
        required_active_throttle_reasons
    )
    completed_replays = 0
    active_seconds = 0.0
    mode_probes: list[Mapping[str, object]] = []
    batch_replays = 1024
    while True:
        remaining_replays = max(0, minimum_replays - completed_replays)
        current_batch_replays = max(
            1, min(batch_replays, remaining_replays or batch_replays)
        )
        batch_started = time.monotonic()
        with torch.cuda.stream(stream):
            for _ in range(current_batch_replays):
                if flush is not None:
                    flush()
                graph.replay()
        stream.synchronize()
        active_seconds += time.monotonic() - batch_started
        completed_replays += current_batch_replays
        if active_seconds > maximum_seconds:
            raise RuntimeError(
                "single-graph duration preconditioning exceeded its maximum: "
                f"observed={active_seconds:.6f}s, maximum={maximum_seconds:.6f}s"
            )
        if completed_replays < minimum_replays or active_seconds < minimum_seconds:
            continue
        probe = mode_snapshot()
        mode_probes.append(probe)
        fields = probe.get("fields")
        observed_pstate = fields.get("pstate") if isinstance(fields, Mapping) else None
        observed_throttle_reasons = _mode_active_throttle_reasons(probe)
        if (
            observed_pstate != required_pstate
            or observed_throttle_reasons != required_throttle_reasons
        ):
            continue
        break
    return {
        "policy": "single_exact_target_graph_duration",
        "minimum_replays": minimum_replays,
        "minimum_active_seconds": minimum_seconds,
        "maximum_active_seconds": maximum_seconds,
        "completed_replays": completed_replays,
        "observed_active_seconds": active_seconds,
        "target_graph_replays": completed_replays,
        "cold_l2_flush_before_every_replay": flush is not None,
        "flush_inside_timed_interval": False,
        "required_pstate": required_pstate,
        "required_active_throttle_reasons": required_throttle_reasons,
        "mode_probes": mode_probes,
    }


def time_single_graph_conditions(
    graph: torch.cuda.CUDAGraph,
    *,
    precondition: int,
    warmup: int,
    replays: int,
    stream: torch.cuda.Stream,
    l2_flush_bytes: int,
    replays_per_reported_sample: int = 1,
    event_batch_replays: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    mode_snapshot: Callable[[], Mapping[str, object]],
    required_pstate: str,
    max_sm_clock_delta_mhz: float,
    required_active_throttle_reasons: int = 0,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, int]]]:
    """Time one graph only; no unused comparison arm is instantiated or replayed."""

    if (
        not isinstance(precondition, int)
        or isinstance(precondition, bool)
        or precondition < 2_000
        or not isinstance(warmup, int)
        or isinstance(warmup, bool)
        or warmup < 100
        or not isinstance(replays, int)
        or isinstance(replays, bool)
        or replays < 1_000
    ):
        raise ValueError(
            "single-arm timing requires precondition>=2000, warmup>=100, replays>=1000"
        )
    if (
        not isinstance(replays_per_reported_sample, int)
        or isinstance(replays_per_reported_sample, bool)
        or replays_per_reported_sample < 1
    ):
        raise ValueError("replays_per_reported_sample must be a positive integer")
    if (
        not isinstance(event_batch_replays, int)
        or isinstance(event_batch_replays, bool)
        or event_batch_replays < 1
    ):
        raise ValueError("event_batch_replays must be a positive integer")
    if (
        not isinstance(precondition_seconds, (int, float))
        or isinstance(precondition_seconds, bool)
        or not math.isfinite(float(precondition_seconds))
        or float(precondition_seconds) < 5.0
    ):
        raise ValueError("precondition_seconds must be at least 5 seconds")
    if (
        not isinstance(maximum_precondition_seconds, (int, float))
        or isinstance(maximum_precondition_seconds, bool)
        or not math.isfinite(float(maximum_precondition_seconds))
        or float(maximum_precondition_seconds) < float(precondition_seconds)
        or float(maximum_precondition_seconds) > 60.0
    ):
        raise ValueError(
            "maximum_precondition_seconds must cover the minimum and be at most 60"
        )
    if not callable(mode_snapshot):
        raise ValueError("mode_snapshot must be an explicit physical-GPU callback")
    mode_policy = timing_mode_policy(
        required_pstate=required_pstate,
        required_active_throttle_reasons=required_active_throttle_reasons,
        max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
    )
    required_throttle_reasons = int(mode_policy["required_active_throttle_reasons"])
    resolved_flush_bytes = resolve_l2_flush_bytes(l2_flush_bytes)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_cache_bytes = int(properties.L2_cache_size)
    if l2_cache_bytes <= 0 or resolved_flush_bytes < 2 * l2_cache_bytes:
        raise ValueError(
            "single-arm cold-L2 capacity must be at least twice physical L2: "
            f"flush={resolved_flush_bytes}, l2={l2_cache_bytes}"
        )
    cold_flush = make_l2_flush_fn(True, resolved_flush_bytes)
    if cold_flush is None:
        raise AssertionError("cold-L2 flush construction unexpectedly returned None")

    conditions: dict[str, dict[str, object]] = {}
    allocation_records: dict[str, dict[str, int]] = {}
    for name, flush in (("warm_l2", None), ("cold_l2", cold_flush)):
        inner_samples_us: list[list[float]] = []
        samples: list[float] = []
        prepared_event_pool = _prepare_single_graph_event_pool(
            replays=replays,
            event_batch_replays=event_batch_replays,
            replays_per_reported_sample=replays_per_reported_sample,
            stream=stream,
        )
        preconditioning = _single_graph_duration_precondition(
            graph,
            minimum_replays=precondition,
            minimum_seconds=float(precondition_seconds),
            maximum_seconds=float(maximum_precondition_seconds),
            stream=stream,
            flush=flush,
            mode_snapshot=mode_snapshot,
            required_pstate=required_pstate,
            required_active_throttle_reasons=required_throttle_reasons,
        )
        before = allocator_counters()
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                if flush is not None:
                    flush()
                graph.replay()
        stream.synchronize()
        mode_before = mode_snapshot()
        event_pairs = prepared_event_pool.event_pairs
        observed_batches = 0
        for batch_start in range(0, replays, event_batch_replays):
            batch_groups: list[list[int]] = []
            next_pair = 0
            with torch.cuda.stream(stream):
                for _ in range(
                    batch_start, min(replays, batch_start + event_batch_replays)
                ):
                    event_indices: list[int] = []
                    for _ in range(replays_per_reported_sample):
                        if next_pair >= len(event_pairs):
                            raise AssertionError(
                                "single-graph CUDA event pool exhausted within batch"
                            )
                        start, end = event_pairs[next_pair]
                        event_indices.append(next_pair)
                        next_pair += 1
                        if flush is not None:
                            flush()
                        start.record(stream)
                        graph.replay()
                        end.record(stream)
                    batch_groups.append(event_indices)
            stream.synchronize()
            for event_indices in batch_groups:
                inner = [
                    event_pairs[index][0].elapsed_time(event_pairs[index][1]) * 1000.0
                    for index in event_indices
                ]
                if any(
                    not isinstance(sample, float)
                    or not math.isfinite(sample)
                    or sample <= 0
                    for sample in inner
                ):
                    raise AssertionError(
                        f"single-arm {name} emitted a nonpositive timing"
                    )
                inner_samples_us.append(inner)
                samples.append(math.fsum(inner) / len(inner))
            observed_batches += 1
        if observed_batches != prepared_event_pool.metadata["batch_count"]:
            raise AssertionError("single-graph CUDA event-pool batch count changed")
        mode_after = mode_snapshot()
        after = allocator_counters()
        if before != after:
            raise AssertionError(
                f"single-arm {name} replay changed allocator counters: {before}->{after}"
            )
        conditions[name] = {
            "l2_flushed": flush is not None,
            "l2_flush_bytes": resolved_flush_bytes if flush is not None else 0,
            "preconditioning": preconditioning,
            "event_pool": dict(prepared_event_pool.metadata),
            "gpu_mode_before_timing": mode_before,
            "gpu_mode_after_timing": mode_after,
            "gpu_mode_stability": _validate_timing_mode_stability(
                mode_before,
                mode_after,
                required_pstate=required_pstate,
                required_active_throttle_reasons=required_throttle_reasons,
                max_sm_clock_delta_mhz=float(max_sm_clock_delta_mhz),
            ),
            "replays_per_reported_sample": replays_per_reported_sample,
            "aggregation": {
                "reported_sample": "arithmetic_mean_us",
                "inner_event_bracketing": "independent_per_graph_replay",
                "inner_schedule": "same_exact_graph_replay_per_repetition",
                "flush_before_every_inner_replay": flush is not None,
                "flush_inside_timed_interval": False,
            },
            "inner_samples_us": inner_samples_us,
            "inner_sample_count": replays * replays_per_reported_sample,
            "samples_us": samples,
        }
        allocation_records[name] = {
            "allocated_bytes_before": before["allocated"],
            "allocated_bytes_after": after["allocated"],
            "reserved_bytes_before": before["reserved"],
            "reserved_bytes_after": after["reserved"],
        }
    return conditions, allocation_records


def exact_artifact_evidence(
    provenance: Mapping[str, Any],
    *,
    verification_before: Mapping[str, Any],
    verification_after: Mapping[str, Any],
) -> dict[str, object]:
    """Freeze one timed cache object and its before/after verification."""

    compile_spec_json = provenance.get("compile_spec_json")
    if not isinstance(compile_spec_json, str) or not compile_spec_json:
        raise RuntimeError("exact artifact has no serialized compile specification")
    if hashlib.sha256(compile_spec_json.encode()).hexdigest() != provenance.get(
        "compile_spec_hash"
    ):
        raise RuntimeError("exact artifact compile-spec hash is inconsistent")
    toolchain = provenance.get("toolchain")
    if not isinstance(toolchain, (dict, list)) or not toolchain:
        raise RuntimeError("exact artifact has no toolchain identity")
    manifest_path = Path(str(provenance["manifest_path"])).resolve()
    sidecar_path = manifest_path.with_name(f"{manifest_path.stem}.ptx.json")
    if verification_before != verification_after:
        raise RuntimeError("exact artifact changed between pre/post verification")
    expected_verification = {
        "passed": True,
        "manifest_sha256": provenance["manifest_sha256"],
        "object_sha256": provenance["object_sha256"],
        "object_bytes": provenance["object_bytes"],
    }
    if dict(verification_before) != expected_verification:
        raise RuntimeError("exact artifact verification does not match provenance")
    return {
        "cache_root": str(Path(str(provenance["cache"])).resolve()),
        "cache_key": provenance["cache_key"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": provenance["manifest_sha256"],
        "frontend_ptx_sidecar_path": str(sidecar_path),
        "frontend_ptx_sidecar_sha256": sha256_file(sidecar_path),
        "object_path": str(Path(str(provenance["object_path"])).resolve()),
        "object_sha256": provenance["object_sha256"],
        "object_bytes": provenance["object_bytes"],
        "compile_spec_hash": provenance["compile_spec_hash"],
        "compile_spec_json": compile_spec_json,
        "semantic_key": provenance["semantic_key"],
        "kernel_id": provenance["kernel_id"],
        "package_fingerprint": provenance["package_fingerprint"],
        "toolchain": toolchain,
        "toolchain_sha256": json_sha256(toolchain),
        "verification_before": dict(verification_before),
        "verification_after": dict(verification_after),
    }


def _load_hashed_json(path: Path, hash_field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read frozen JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"frozen JSON is not an object: {path}")
    recorded = value.get(hash_field)
    payload = {key: item for key, item in value.items() if key != hash_field}
    if recorded != json_sha256(payload):
        raise RuntimeError(f"frozen JSON canonical hash mismatch: {path}")
    return value


def _normalized_mode_snapshot(snapshot: Mapping[str, Any]) -> dict[str, object]:
    fields = snapshot.get("fields")
    if snapshot.get("available") is not True or not isinstance(fields, Mapping):
        raise RuntimeError(f"GPU mode snapshot is unavailable: {snapshot}")
    return {
        "available": True,
        "torch_uuid": snapshot["torch_uuid"],
        "nvidia_smi_uuid": snapshot["nvidia_smi_uuid"],
        "captured_unix_ns": snapshot["captured_unix_ns"],
        "fields": dict(fields),
    }


def _validate_single_arm_source_owned_nodes(
    value: object,
    *,
    case_id: str,
    repo_root: Path,
    kernel_node_count: int,
) -> list[dict[str, object]]:
    """Validate and source-bind every non-exact CUDA-graph kernel node."""

    if not isinstance(value, list):
        raise RuntimeError(f"{case_id}: source-owned kernel nodes are not a list")
    normalized: list[dict[str, object]] = []
    previous_node_index = -1
    required_fields = {
        "node_index",
        "role",
        "implementation",
        "kernel_name",
        "kernel_name_sha256",
        "grid",
        "block",
        "dynamic_smem_bytes",
        "source_files",
    }
    for record_index, raw_record in enumerate(value):
        if not isinstance(raw_record, Mapping) or set(raw_record) != required_fields:
            raise RuntimeError(
                f"{case_id}: malformed source-owned kernel node {record_index}"
            )
        record = dict(raw_record)
        node_index = record["node_index"]
        if (
            not isinstance(node_index, int)
            or isinstance(node_index, bool)
            or node_index < 0
            or node_index >= kernel_node_count
            or node_index <= previous_node_index
        ):
            raise RuntimeError(
                f"{case_id}: source-owned kernel ordinals are not ordered/in-range"
            )
        previous_node_index = node_index
        role = record["role"]
        if not isinstance(role, str) or not role:
            raise RuntimeError(f"{case_id}: source-owned kernel role is empty")
        if record["implementation"] not in ("torch_cuda", "triton"):
            raise RuntimeError(
                f"{case_id}: unsupported source-owned kernel implementation"
            )
        kernel_name = record["kernel_name"]
        if (
            not isinstance(kernel_name, str)
            or not kernel_name
            or record["kernel_name_sha256"]
            != hashlib.sha256(kernel_name.encode("utf-8")).hexdigest()
        ):
            raise RuntimeError(f"{case_id}: source-owned kernel name hash differs")
        for field in ("grid", "block"):
            dimensions = record[field]
            if (
                not isinstance(dimensions, list)
                or len(dimensions) != 3
                or any(
                    not isinstance(dimension, int)
                    or isinstance(dimension, bool)
                    or dimension <= 0
                    for dimension in dimensions
                )
            ):
                raise RuntimeError(f"{case_id}: malformed source-owned kernel {field}")
        dynamic_smem = record["dynamic_smem_bytes"]
        if (
            not isinstance(dynamic_smem, int)
            or isinstance(dynamic_smem, bool)
            or dynamic_smem < 0
        ):
            raise RuntimeError(
                f"{case_id}: malformed source-owned dynamic shared memory"
            )
        source_files = record["source_files"]
        if not isinstance(source_files, list) or not source_files:
            raise RuntimeError(f"{case_id}: source-owned kernel has no source files")
        source_paths: list[str] = []
        normalized_files: list[dict[str, str]] = []
        for raw_file in source_files:
            if not isinstance(raw_file, Mapping) or set(raw_file) != {"path", "sha256"}:
                raise RuntimeError(
                    f"{case_id}: malformed source-owned kernel source record"
                )
            relative_path = raw_file["path"]
            expected_sha256 = raw_file["sha256"]
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or Path(relative_path).is_absolute()
                or Path(relative_path).as_posix() != relative_path
                or ".." in Path(relative_path).parts
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
            ):
                raise RuntimeError(
                    f"{case_id}: invalid source-owned kernel source identity"
                )
            source_path = (repo_root / relative_path).resolve()
            try:
                source_path.relative_to(repo_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"{case_id}: source-owned kernel path escapes runtime root"
                ) from exc
            if not source_path.is_file() or sha256_file(source_path) != expected_sha256:
                raise RuntimeError(
                    f"{case_id}: source-owned kernel source differs from review"
                )
            source_paths.append(relative_path)
            normalized_files.append({"path": relative_path, "sha256": expected_sha256})
        if source_paths != sorted(set(source_paths)):
            raise RuntimeError(
                f"{case_id}: source-owned kernel source paths are not sorted/unique"
            )
        normalized.append({**record, "source_files": normalized_files})
    return normalized


def build_single_arm_e2e_result(
    *,
    family: str,
    arm: str,
    sequence_position: str,
    evidence_status: str,
    repo_root: Path,
    producer_path: Path,
    source_manifest_path: Path,
    contract_path: Path,
    started_unix_ns: int,
    finished_unix_ns: int,
    expected_physical_gpu: int,
    gpu_mode_before: Mapping[str, Any],
    gpu_mode_after: Mapping[str, Any],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    """Build the immutable RUN_SCHEMA envelope around one-arm GPU evidence."""

    expected_arm = {
        "a1": "baseline",
        "a2": "baseline",
        "b1": "current",
        "b2": "current",
    }.get(sequence_position)
    if expected_arm != arm:
        raise ValueError(f"sequence position {sequence_position!r} is not arm {arm!r}")
    if evidence_status not in ("final-source", "diagnostic-non-final"):
        raise ValueError(
            "evidence status must be explicitly final-source or diagnostic-non-final"
        )
    if finished_unix_ns <= started_unix_ns:
        raise ValueError("single-arm process timestamps are not ordered")
    repo_root = repo_root.resolve()
    producer_path = producer_path.resolve()
    source_manifest_path = source_manifest_path.resolve()
    contract_path = contract_path.resolve()
    source = _load_hashed_json(source_manifest_path, "manifest_sha256")
    contract = _load_hashed_json(contract_path, "contract_sha256")
    if source.get("side") != arm:
        raise RuntimeError("single-arm source manifest side differs from requested arm")
    source_runtime = source.get("runtime")
    if not isinstance(source_runtime, dict) or source_runtime.get("repo_root") != str(
        repo_root
    ):
        raise RuntimeError("single-arm worktree differs from frozen source runtime")
    runtime_package = source_runtime.get("sparkinfer_package")
    production = source.get("production")
    if not isinstance(runtime_package, dict) or not isinstance(production, dict):
        raise RuntimeError("single-arm source manifest lacks package trees")
    production_package = production.get("sparkinfer_package")
    if not isinstance(production_package, dict):
        raise RuntimeError("single-arm source manifest lacks production package")
    runtime_fingerprint = str(runtime_package.get("fingerprint", ""))
    observed_source_fingerprint = cute_compiler._sparkinfer_package_fingerprint()
    if not runtime_fingerprint or observed_source_fingerprint != runtime_fingerprint:
        raise RuntimeError("single-arm runtime differs from frozen source manifest")

    harness = contract.get("harness")
    harness_files = harness.get("files") if isinstance(harness, dict) else None
    if not isinstance(harness_files, list) or not harness_files:
        raise RuntimeError("single-arm contract lacks frozen harness files")
    for record in harness_files:
        if not isinstance(record, dict):
            raise RuntimeError("single-arm harness record is malformed")
        relative_path = record.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError("single-arm harness record lacks a path")
        observed_path = (repo_root / relative_path).resolve()
        try:
            observed_path.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError(
                f"single-arm harness path escapes the runtime root: {relative_path}"
            ) from exc
        if (
            not observed_path.is_file()
            or sha256_file(observed_path) != record.get("sha256")
            or observed_path.stat().st_size != record.get("size_bytes")
        ):
            raise RuntimeError(
                f"single-arm harness file differs from frozen contract: {relative_path}"
            )

    families = contract.get("families")
    family_contract = families.get(family) if isinstance(families, dict) else None
    if not isinstance(family_contract, dict):
        raise RuntimeError(f"family {family!r} is absent from frozen E2E contract")
    relative_producer = producer_path.relative_to(repo_root).as_posix()
    if family_contract.get("producer") != relative_producer or family_contract.get(
        "producer_sha256"
    ) != sha256_file(producer_path):
        raise RuntimeError("single-arm producer differs from frozen family harness")
    reviewed_cases = family_contract.get("cases")
    if not isinstance(reviewed_cases, list):
        raise RuntimeError("single-arm family contract has no reviewed cases")
    reviewed_case_bindings = {
        (
            case.get("case_id"),
            case.get("case_contract_sha256"),
            case.get("input_sha256"),
        )
        for case in reviewed_cases
        if isinstance(case, dict)
    }
    emitted_case_bindings = {
        (
            case.get("case_id"),
            case.get("case_contract_sha256"),
            case.get("input_sha256"),
        )
        for case in cases
    }
    if emitted_case_bindings != reviewed_case_bindings or len(cases) != len(
        reviewed_case_bindings
    ):
        raise RuntimeError("single-arm emitted case set differs from frozen contract")

    reviewed_by_case = {
        str(case["case_id"]): case
        for case in reviewed_cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    artifact_evidence: list[Mapping[str, Any]] = []
    case_object_bindings: list[list[dict[str, object]]] = []
    for case in cases:
        case_id = case.get("case_id")
        reviewed = reviewed_by_case.get(str(case_id))
        if reviewed is None:
            raise RuntimeError(f"single-arm case is not reviewed: {case_id!r}")
        compile_contract = reviewed.get("compile_artifact_contract")
        arm_contract = (
            compile_contract.get(arm) if isinstance(compile_contract, dict) else None
        )
        if not isinstance(arm_contract, dict) or set(arm_contract) != {
            "artifacts",
            "launch_plan",
            "source_owned_kernel_nodes",
        }:
            raise RuntimeError(f"{case_id}: missing {arm} compile contract")
        expected_artifacts = arm_contract.get("artifacts")
        expected_plan = arm_contract.get("launch_plan")
        expected_source_owned = arm_contract.get("source_owned_kernel_nodes")
        artifacts = case.get("artifacts")
        launch_plan = case.get("launch_plan")
        source_owned = case.get("source_owned_kernel_nodes")
        if (
            not isinstance(expected_artifacts, list)
            or not expected_artifacts
            or not isinstance(expected_plan, list)
            or not expected_plan
            or not isinstance(expected_source_owned, list)
            or not isinstance(artifacts, list)
            or not artifacts
            or not isinstance(launch_plan, list)
            or not launch_plan
            or not isinstance(source_owned, list)
        ):
            raise RuntimeError(f"{case_id}: artifact/launch evidence is incomplete")
        expected_by_role = {
            str(item.get("role")): item
            for item in expected_artifacts
            if isinstance(item, Mapping)
        }
        by_role: dict[str, Mapping[str, Any]] = {}
        object_identities: set[tuple[str, str]] = set()
        process_bindings: list[dict[str, object]] = []
        for binding in artifacts:
            if not isinstance(binding, Mapping) or set(binding) != {
                "role",
                "kernel_id",
                "compile_spec_hash",
                "object_sha256",
                "evidence",
            }:
                raise RuntimeError(f"{case_id}: malformed exact artifact binding")
            role = str(binding["role"])
            expected = expected_by_role.get(role)
            evidence = binding["evidence"]
            if (
                expected is None
                or role in by_role
                or not isinstance(evidence, Mapping)
                or any(
                    binding.get(field) != evidence.get(field)
                    for field in ("kernel_id", "compile_spec_hash", "object_sha256")
                )
                or any(
                    evidence.get(field) != expected.get(field)
                    for field in ("kernel_id", "compile_spec_hash", "compile_spec_json")
                )
            ):
                raise RuntimeError(
                    f"{case_id}: exact artifact binding differs from review"
                )
            if evidence.get("package_fingerprint") != runtime_fingerprint:
                raise RuntimeError(
                    "single-arm runtime, source manifest, and exact object "
                    "fingerprints differ"
                )
            expected_after = {
                "passed": True,
                "manifest_sha256": evidence.get("manifest_sha256"),
                "object_sha256": evidence.get("object_sha256"),
                "object_bytes": evidence.get("object_bytes"),
            }
            if (
                evidence.get("verification_before") != expected_after
                or evidence.get("verification_after") != expected_after
                or verify_artifact(evidence) != expected_after
            ):
                raise RuntimeError(
                    "single-arm exact artifact verification is inconsistent"
                )
            object_identity = (
                str(evidence.get("cache_key")),
                str(evidence.get("object_sha256")),
            )
            if object_identity in object_identities:
                raise RuntimeError(f"{case_id}: exact cache object is duplicated")
            object_identities.add(object_identity)
            by_role[role] = binding
            artifact_evidence.append(evidence)
            process_bindings.append(
                {
                    "role": role,
                    "kernel_id": binding["kernel_id"],
                    "compile_spec_hash": binding["compile_spec_hash"],
                    "object_sha256": binding["object_sha256"],
                }
            )
        if set(by_role) != set(expected_by_role):
            raise RuntimeError(f"{case_id}: exact artifact role coverage is incomplete")

        graph = case.get("graph")
        if not isinstance(graph, Mapping):
            raise RuntimeError(f"{case_id}: graph evidence is missing")
        graph_kernel_node_count = graph.get("kernel_node_count")
        if (
            not isinstance(graph_kernel_node_count, int)
            or isinstance(graph_kernel_node_count, bool)
            or graph_kernel_node_count < 1
        ):
            raise RuntimeError(f"{case_id}: graph kernel count is invalid")
        normalized_source_owned = _validate_single_arm_source_owned_nodes(
            source_owned,
            case_id=str(case_id),
            repo_root=repo_root,
            kernel_node_count=graph_kernel_node_count,
        )
        if normalized_source_owned != expected_source_owned:
            raise RuntimeError(
                f"{case_id}: source-owned graph nodes differ from review"
            )

        normalized_plan: list[dict[str, object]] = []
        multiplicities: dict[str, int] = {}
        used_roles: set[str] = set()
        previous_node_index = -1
        for binding in launch_plan:
            if not isinstance(binding, Mapping) or set(binding) != {
                "node_index",
                "artifact_role",
                "kernel_id",
                "compile_spec_hash",
                "object_sha256",
                "multiplicity_index",
            }:
                raise RuntimeError(f"{case_id}: malformed launch-plan binding")
            role = str(binding["artifact_role"])
            artifact = by_role.get(role)
            multiplicity = multiplicities.get(role, 0) + 1
            node_index = binding["node_index"]
            if (
                artifact is None
                or not isinstance(node_index, int)
                or isinstance(node_index, bool)
                or node_index < 0
                or node_index >= graph_kernel_node_count
                or node_index <= previous_node_index
                or binding["multiplicity_index"] != multiplicity
                or any(
                    binding.get(field) != artifact.get(field)
                    for field in ("kernel_id", "compile_spec_hash", "object_sha256")
                )
            ):
                raise RuntimeError(
                    f"{case_id}: launch order/multiplicity/object binding is invalid"
                )
            previous_node_index = node_index
            multiplicities[role] = multiplicity
            used_roles.add(role)
            normalized_plan.append(
                {key: value for key, value in binding.items() if key != "object_sha256"}
            )
        covered_node_indices = sorted(
            [int(binding["node_index"]) for binding in launch_plan]
            + [int(record["node_index"]) for record in normalized_source_owned]
        )
        if (
            normalized_plan != expected_plan
            or used_roles != set(by_role)
            or covered_node_indices != list(range(graph_kernel_node_count))
        ):
            raise RuntimeError(
                f"{case_id}: launch/source partition differs from review or graph topology"
            )
        case_object_bindings.append(
            sorted(process_bindings, key=lambda item: str(item["role"]))
        )

    arm_toolchains = contract.get("arm_toolchains")
    arm_toolchain = (
        arm_toolchains.get(arm) if isinstance(arm_toolchains, dict) else None
    )
    if not isinstance(arm_toolchain, dict):
        raise RuntimeError("single-arm contract lacks arm toolchain")
    cutlass_packages = _single_arm_cutlass_packages()
    if cutlass_packages != arm_toolchain.get("cutlass_packages"):
        raise RuntimeError(
            "single-arm CUTLASS package map differs from frozen contract"
        )
    ptxas_versions = {source_ptxas_version(artifact) for artifact in artifact_evidence}
    if ptxas_versions != {arm_toolchain.get("ptxas_version")}:
        raise RuntimeError("single-arm source PTXAS differs from frozen contract")
    ptxas_version = ptxas_versions.pop()

    error, driver_version = cuda_driver.cuDriverGetVersion()
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDriverGetVersion failed: {error}")
    raw_environment_sha256, comparison_environment_sha256 = (
        _single_arm_runtime_environment()
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    command = [str(Path(sys.executable).resolve()), *sys.argv]
    process_identity = json_sha256(
        {
            "pid": os.getpid(),
            "started_unix_ns": started_unix_ns,
            "command": command,
            "source_manifest_sha256": source["manifest_sha256"],
            "contract_sha256": contract["contract_sha256"],
            "case_objects": case_object_bindings,
        }
    )
    payload: dict[str, object] = {
        "schema": SINGLE_ARM_E2E_RUN_SCHEMA,
        "family": family,
        "arm": arm,
        "sequence_position": sequence_position,
        "evidence_status": evidence_status,
        "invocation": {
            "process_id": process_identity,
            "pid": os.getpid(),
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": finished_unix_ns,
            "command": command,
            "worktree": str(repo_root),
        },
        "source": {
            "manifest_sha256": source["manifest_sha256"],
            "manifest_artifact_sha256": sha256_file(source_manifest_path),
            "production_fingerprint": production_package["fingerprint"],
            "runtime_package_fingerprint": runtime_fingerprint,
        },
        "producer": {
            "path": relative_producer,
            "sha256": sha256_file(producer_path),
        },
        "harness_case_contract_sha256": contract["contract_sha256"],
        "cutlass_packages": cutlass_packages,
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_driver_version": str(driver_version),
            "ptxas_version": ptxas_version,
            "raw_environment_sha256": raw_environment_sha256,
            "comparison_environment_sha256": comparison_environment_sha256,
        },
        "gpu": {
            "physical_ordinal": expected_physical_gpu,
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "capability": list(torch.cuda.get_device_capability()),
            "l2_cache_bytes": int(properties.L2_cache_size),
            "mode_before": _normalized_mode_snapshot(gpu_mode_before),
            "mode_after": _normalized_mode_snapshot(gpu_mode_after),
        },
        "cases": cases,
    }
    return {**payload, "result_sha256": json_sha256(payload)}
