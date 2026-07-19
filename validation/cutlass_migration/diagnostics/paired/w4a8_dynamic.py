#!/usr/bin/env python3
"""Same-address CUDA-graph ABBA for exact cached W4A8 objects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import torch
from cuda.bindings import driver as cuda_driver

import b12x.cute.compiler as cute_compiler
from benchmarks.common import make_l2_flush_fn
from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
)
from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument,
    require_target_gpu,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    artifact_provenance,
    gpu_mode_snapshot,
    time_conditions,
    verify_artifact,
)
from validation.cutlass_migration.paths import REPO_ROOT
from tests.test_w4a8_dynamic_kernel import _run_w4a8_dynamic


_M128_SPEC_HASH = "4775fdec2a6715860a8c341a3fdbb755239f3b5cbd1916c50f32c6dd29eacce3"
_EXPERTS = 4
_HIDDEN = 256
_INTERMEDIATE = 128
_TOPK = 2


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    add_target_gpu_argument(parser)
    parser.add_argument("--a-cache", type=pathlib.Path, required=True)
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--b-cache", type=pathlib.Path, required=True)
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument("--spec-hash", default=_M128_SPEC_HASH)
    parser.add_argument("--m", type=int, default=129)
    parser.add_argument("--tile-m", type=int, choices=(16, 32, 64, 128), default=128)
    parser.add_argument(
        "--recipe", choices=("w4a8_mx", "w4a8_nvfp4"), default="w4a8_nvfp4"
    )
    parser.add_argument("--activation", choices=("silu", "relu2"), default="relu2")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--precondition", type=int, default=100)
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-precondition-seconds", type=float, default=60.0)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--event-batch-cycles", type=int, default=25)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    parser.add_argument(
        "--cold-l2", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_for_spec(
    cache: pathlib.Path, spec_hash: str
) -> tuple[pathlib.Path, dict[str, object]]:
    matches = []
    for path in cache.rglob("*.json"):
        manifest = json.loads(path.read_text())
        if (
            manifest.get("schema") == "b12x.cute.compile_manifest.v3"
            and manifest.get("compile_spec_hash") == spec_hash
            and manifest.get("object_sha256")
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one manifest for spec {spec_hash} in {cache}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_exact(cache: pathlib.Path, spec_hash: str):
    provenance = artifact_provenance(cache, spec_hash)
    cache_key = str(provenance["cache_key"])
    previous = os.environ.get("B12X_CUTE_COMPILE_CACHE_DIR")
    os.environ["B12X_CUTE_COMPILE_CACHE_DIR"] = str(cache)
    try:
        compiled = cute_compiler._load_cute_compile_from_disk(cache_key)
    finally:
        if previous is None:
            os.environ.pop("B12X_CUTE_COMPILE_CACHE_DIR", None)
        else:
            os.environ["B12X_CUTE_COMPILE_CACHE_DIR"] = previous
    if compiled is None:
        raise RuntimeError(f"failed to load exact object {provenance['object_path']}")
    manifest = json.loads(
        pathlib.Path(str(provenance["manifest_path"])).read_text(encoding="utf-8")
    )
    return compiled, {**provenance, "launch_metadata": manifest["launch_metadata"]}


def _graph_topology(graph: torch.cuda.CUDAGraph) -> dict[str, object]:
    raw_graph = graph.raw_cuda_graph()
    error, _, node_count = cuda_driver.cuGraphGetNodes(raw_graph)
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGraphGetNodes count failed: {error}")
    error, nodes, returned_nodes = cuda_driver.cuGraphGetNodes(raw_graph, node_count)
    if error != cuda_driver.CUresult.CUDA_SUCCESS or returned_nodes != node_count:
        raise RuntimeError(
            f"cuGraphGetNodes failed: {error}, {returned_nodes}/{node_count}"
        )
    node_indices = {int(node): index for index, node in enumerate(nodes)}
    node_metadata = []
    for index, node in enumerate(nodes):
        error, node_type = cuda_driver.cuGraphNodeGetType(node)
        if error != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphNodeGetType failed: {error}")
        metadata: dict[str, object] = {"index": index, "type": node_type.name}
        if node_type == cuda_driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            error, params = cuda_driver.cuGraphKernelNodeGetParams(node)
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuGraphKernelNodeGetParams failed: {error}")
            error, name = cuda_driver.cuFuncGetName(params.func)
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuFuncGetName failed: {error}")
            metadata.update(
                {
                    "kernel_name": name.decode(),
                    "grid": [params.gridDimX, params.gridDimY, params.gridDimZ],
                    "block": [params.blockDimX, params.blockDimY, params.blockDimZ],
                    "dynamic_smem_bytes": params.sharedMemBytes,
                }
            )
        node_metadata.append(metadata)
    error, _, _, _, edge_count = cuda_driver.cuGraphGetEdges(raw_graph)
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGraphGetEdges count failed: {error}")
    error, sources, destinations, _, returned_edges = cuda_driver.cuGraphGetEdges(
        raw_graph, edge_count
    )
    if error != cuda_driver.CUresult.CUDA_SUCCESS or returned_edges != edge_count:
        raise RuntimeError(
            f"cuGraphGetEdges failed: {error}, {returned_edges}/{edge_count}"
        )
    return {
        "node_count": node_count,
        "kernel_node_count": sum(
            node["type"] == "CU_GRAPH_NODE_TYPE_KERNEL" for node in node_metadata
        ),
        "nodes": node_metadata,
        "edges": [
            [node_indices[int(source)], node_indices[int(destination)]]
            for source, destination in zip(sources, destinations, strict=True)
        ],
    }


def _topology_signature(topology: dict[str, object]) -> dict[str, object]:
    return {
        "node_count": topology["node_count"],
        "kernel_node_count": topology["kernel_node_count"],
        "nodes": [
            {key: value for key, value in node.items() if key != "kernel_name"}
            for node in topology["nodes"]
        ],
        "edges": topology["edges"],
    }


def _capture(launch, stream: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        launch()
    stream.synchronize()
    return graph


def _correctness(output: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    actual = output.float()
    expected = reference.float()
    difference = actual - expected
    denominator = max(float(expected.norm()), 1e-12)
    relative_l2 = float(difference.norm()) / denominator
    cosine_denominator = max(float(actual.norm()) * denominator, 1e-24)
    cosine = float((actual * expected).sum()) / cosine_denominator
    finite = bool(torch.isfinite(actual).all())
    nonzero = int(torch.count_nonzero(actual))
    allclose = bool(torch.allclose(actual, expected, rtol=0.15, atol=0.20))
    passed = bool(
        finite
        and nonzero > 0
        and math.isfinite(relative_l2)
        and relative_l2 <= 0.03
        and math.isfinite(cosine)
        and cosine >= 0.999
        and allclose
    )
    metrics = {
        "passed": passed,
        "finite": finite,
        "nonzero": nonzero,
        "max_abs": float(difference.abs().max()),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "allclose": allclose,
    }
    if not passed:
        raise AssertionError(f"W4A8 oracle failure: {metrics}")
    return metrics


def _oracle_phase(
    phase: str,
    graphs: dict[str, torch.cuda.CUDAGraph],
    labels: tuple[str, str],
    output: torch.Tensor,
    reference: torch.Tensor,
    stream: torch.cuda.Stream,
    *,
    scenario: int,
    previous_outputs: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    outputs = {}
    metrics = {}
    replay_allocator_records = {}
    for label in labels:
        allocation_before = allocator_counters()
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            graphs[label].replay()
        stream.synchronize()
        allocation_after = allocator_counters()
        if allocation_after != allocation_before:
            raise AssertionError(
                f"{label} changed CUDA allocator state during scenario {scenario} replay"
            )
        if bool(torch.isnan(output).any()):
            raise AssertionError(f"{label} left poisoned values in output")
        outputs[label] = output.clone()
        metrics[label] = _correctness(output, reference)
        replay_allocator_records[label] = {
            "before": allocation_before,
            "after": allocation_after,
            "unchanged": True,
        }
    pair_difference = outputs[labels[1]].float() - outputs[labels[0]].float()
    arms_bit_exact = bool(torch.equal(outputs[labels[0]], outputs[labels[1]]))
    if not arms_bit_exact:
        raise AssertionError("W4A8 A/B outputs are not bit exact")
    output_sha256 = {label: _tensor_sha256(outputs[label]) for label in labels}
    changed_from_previous = None
    if previous_outputs is not None:
        changed_from_previous = {
            label: not torch.equal(previous_outputs[label], outputs[label])
            for label in labels
        }
        if not all(changed_from_previous.values()):
            raise AssertionError(
                "W4A8 live-input mutation did not change every arm's output"
            )
    return {
        "phase": phase,
        "scenario": scenario,
        "metrics": metrics,
        "poisoned_output_overwritten": True,
        "arms_bit_exact": arms_bit_exact,
        "arm_max_abs_difference": float(pair_difference.abs().max()),
        "output_sha256": output_sha256,
        "changed_output_from_previous_scenario": changed_from_previous,
        "zero_replay_allocations": True,
        "replay_allocator_records": replay_allocator_records,
    }, outputs


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(f"{tuple(tensor.shape)}:{tensor.dtype}".encode())
    digest.update(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )
    return digest.hexdigest()


def _tensor_hash_record(tensors: dict[str, torch.Tensor]) -> dict[str, object]:
    tensor_sha256 = {
        name: _tensor_sha256(tensor) for name, tensor in sorted(tensors.items())
    }
    aggregate_sha256 = hashlib.sha256(
        json.dumps(
            tensor_sha256,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "tensor_sha256": tensor_sha256,
        "aggregate_sha256": aggregate_sha256,
    }


def _mutate_live_inputs(
    live_inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, object], dict[str, object], dict[str, bool]]:
    required = {"x", "topk_ids", "topk_weights"}
    if set(live_inputs) != required:
        raise AssertionError(f"unexpected W4A8 live-input set: {sorted(live_inputs)}")
    before = _tensor_hash_record(live_inputs)
    before_weight_sha256 = _tensor_sha256(live_inputs["topk_weights"])
    live_inputs["x"].mul_(-0.75)
    live_inputs["topk_ids"].add_(1).remainder_(_EXPERTS)
    # For top-k=2, 1-w swaps the normalized route weights without allocating a
    # replacement tensor. The equality fallback covers an exactly symmetric
    # 0.5/0.5 input while retaining the same storage.
    live_inputs["topk_weights"].sub_(1.0).neg_()
    torch.cuda.synchronize()
    after = _tensor_hash_record(live_inputs)
    if before_weight_sha256 == _tensor_sha256(live_inputs["topk_weights"]):
        live_inputs["topk_weights"][:, 0].fill_(0.75)
        live_inputs["topk_weights"][:, 1].fill_(0.25)
        torch.cuda.synchronize()
        after = _tensor_hash_record(live_inputs)
    before_tensors = before["tensor_sha256"]
    after_tensors = after["tensor_sha256"]
    if not isinstance(before_tensors, dict) or not isinstance(after_tensors, dict):
        raise AssertionError("malformed W4A8 live-input hash record")
    changed = {
        name: before_tensors[name] != after_tensors[name] for name in sorted(required)
    }
    if not all(changed.values()):
        raise AssertionError(
            f"W4A8 live-input mutation did not change every live tensor: {changed}"
        )
    if before["aggregate_sha256"] == after["aggregate_sha256"]:
        raise AssertionError(
            "W4A8 live-input scenarios have identical aggregate hashes"
        )
    return before, after, changed


def _all_tensors(state: dict[str, object]) -> dict[str, torch.Tensor]:
    result = {}
    for group in ("live_inputs", "read_only_inputs", "mutable_allocations"):
        for name, tensor in state[group].items():
            result[f"{group}.{name}"] = tensor
    return result


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "missing"


def main() -> None:
    args = _args()
    if args.m <= 0:
        raise ValueError("--m must be positive")
    if args.cycles < 500:
        raise ValueError("--cycles must be at least 500 for this comparison")
    if args.precondition < 1 or args.warmup < 1 or args.event_batch_cycles < 1:
        raise ValueError("invalid precondition/warmup/event-batch settings")
    if args.cycles % 2:
        raise ValueError("--cycles must be even for balanced ABBA/BAAB timing")
    if args.replays_per_reported_sample < 1:
        raise ValueError("--replays-per-reported-sample must be positive")
    if args.precondition_seconds < 5.0:
        raise ValueError("release timing requires --precondition-seconds >= 5")
    if (
        args.maximum_precondition_seconds < args.precondition_seconds
        or args.maximum_precondition_seconds > 60.0
    ):
        raise ValueError(
            "release timing requires precondition-seconds <= "
            "maximum-precondition-seconds <= 60"
        )
    if args.max_sm_clock_delta_mhz != 60.0:
        raise ValueError("release timing requires --max-sm-clock-delta-mhz 60")
    if not args.cold_l2 and args.evidence_status == "final-source":
        raise ValueError("final-source evidence requires warm- and cold-L2 timing")
    require_target_gpu(args.expected_physical_gpu)
    gpu_mode_initial = gpu_mode_snapshot(args.expected_physical_gpu)
    labels = (args.a_label, args.b_label)
    if labels[0] == labels[1]:
        raise ValueError("arm labels must differ")

    compiled_a, provenance_a = _load_exact(args.a_cache, args.spec_hash)
    compiled_b, provenance_b = _load_exact(args.b_cache, args.spec_hash)
    provenance = {labels[0]: provenance_a, labels[1]: provenance_b}
    artifact_verification_before = {
        label: verify_artifact(record) for label, record in provenance.items()
    }
    if provenance_a["compile_spec_json"] != provenance_b["compile_spec_json"]:
        raise AssertionError("arm compile specifications differ")
    if provenance_a["launch_metadata"] != provenance_b["launch_metadata"]:
        raise AssertionError("arm launch metadata differ")
    compile_spec = json.loads(provenance_a["compile_spec_json"])
    compile_facts = dict(compile_spec["facts"])
    requested_facts = {
        "recipe": args.recipe,
        "activation": args.activation,
        "tile_m": args.tile_m,
        "experts": _EXPERTS,
        "hidden": _HIDDEN,
        "intermediate": _INTERMEDIATE,
        "top_k": _TOPK,
    }
    if compile_facts != requested_facts:
        raise AssertionError(
            f"requested specialization {requested_facts} does not match "
            f"cached object {compile_facts}"
        )

    output, reference, _, state = _run_w4a8_dynamic(
        recipe=args.recipe,
        activation=args.activation,
        E=_EXPERTS,
        m=args.m,
        K=_HIDDEN,
        n=_INTERMEDIATE,
        top_k=_TOPK,
        seed=args.seed,
        tile_m=args.tile_m,
        return_launcher=True,
        return_state=True,
        compiled_override=compiled_a,
    )
    torch.cuda.synchronize()
    relaunch_with = state["relaunch_with"]
    launches = {
        labels[0]: lambda: relaunch_with(compiled_a),
        labels[1]: lambda: relaunch_with(compiled_b),
    }

    live_inputs = state["live_inputs"]
    if not isinstance(live_inputs, dict) or not all(
        isinstance(tensor, torch.Tensor) for tensor in live_inputs.values()
    ):
        raise AssertionError("W4A8 helper returned malformed live inputs")
    read_only_inputs = state["read_only_inputs"]
    if not isinstance(read_only_inputs, dict) or not all(
        isinstance(tensor, torch.Tensor) for tensor in read_only_inputs.values()
    ):
        raise AssertionError("W4A8 helper returned malformed read-only inputs")
    mutable_allocations = state["mutable_allocations"]
    if not isinstance(mutable_allocations, dict) or not all(
        isinstance(tensor, torch.Tensor) for tensor in mutable_allocations.values()
    ):
        raise AssertionError("W4A8 helper returned malformed workspace allocations")
    all_tensors = _all_tensors(state)
    initial_pointers = {name: tensor.data_ptr() for name, tensor in all_tensors.items()}
    read_only_initial = {
        name: _tensor_sha256(tensor) for name, tensor in read_only_inputs.items()
    }
    scenario_0_before_timing = _tensor_hash_record(live_inputs)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = {label: _capture(launches[label], stream) for label in labels}
    topologies = {label: _graph_topology(graphs[label]) for label in labels}
    topology_equal = _topology_signature(topologies[labels[0]]) == _topology_signature(
        topologies[labels[1]]
    )
    if not topology_equal:
        raise AssertionError("arm CUDA graph topologies differ")

    correctness_pre, _ = _oracle_phase(
        "pre-timing-scenario-0",
        graphs,
        labels,
        output,
        reference,
        stream,
        scenario=0,
    )

    if args.cold_l2:
        # Materialize the stable flush-buffer capacity before the outer
        # allocation snapshot. The shared timer reuses this exact buffer.
        make_l2_flush_fn(True, args.l2_flush_bytes)
    allocation_before_timing = allocator_counters()
    conditions = time_conditions(
        graphs,
        labels=labels,
        precondition=args.precondition,
        warmup=args.warmup,
        cycles=args.cycles,
        event_batch_cycles=args.event_batch_cycles,
        stream=stream,
        cold_l2=args.cold_l2,
        l2_flush_bytes=args.l2_flush_bytes,
        replays_per_reported_sample=args.replays_per_reported_sample,
        precondition_seconds=args.precondition_seconds,
        maximum_precondition_seconds=args.maximum_precondition_seconds,
        mode_snapshot=lambda: gpu_mode_snapshot(args.expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
    )
    allocation_after_timing = allocator_counters()
    if allocation_after_timing != allocation_before_timing:
        raise AssertionError("CUDA allocator state changed during W4A8 timing")
    compile_spec_hashes = {label: args.spec_hash for label in labels}
    for condition in conditions.values():
        condition["compile_spec_hashes"] = compile_spec_hashes
        condition["all_graph_spec_hashes"] = [args.spec_hash]
    correctness_post, scenario_0_outputs = _oracle_phase(
        "post-timing-scenario-0",
        graphs,
        labels,
        output,
        reference,
        stream,
        scenario=0,
    )
    scenario_0_after_timing = _tensor_hash_record(live_inputs)
    if scenario_0_after_timing != scenario_0_before_timing:
        raise AssertionError("W4A8 scenario-0 live inputs changed during timing")

    mutation_before, scenario_1, changed_inputs = _mutate_live_inputs(live_inputs)
    if mutation_before != scenario_0_after_timing:
        raise AssertionError("W4A8 mutation did not begin from timed scenario 0")
    current_reference = state["current_reference"]
    if not callable(current_reference):
        raise AssertionError("W4A8 helper returned a non-callable reference")
    reference_scenario_1 = current_reference()
    torch.cuda.synchronize()
    correctness_live, _ = _oracle_phase(
        "post-timing-live-mutation-scenario-1",
        graphs,
        labels,
        output,
        reference_scenario_1,
        stream,
        scenario=1,
        previous_outputs=scenario_0_outputs,
    )
    changed_outputs = correctness_live["changed_output_from_previous_scenario"]
    if not isinstance(changed_outputs, dict) or not all(changed_outputs.values()):
        raise AssertionError("W4A8 live scenario did not change every arm output")
    correctness = [correctness_pre, correctness_post, correctness_live]
    scenario_0_tensor_sha256 = scenario_0_after_timing.get("tensor_sha256")
    scenario_1_tensor_sha256 = scenario_1.get("tensor_sha256")
    if not isinstance(scenario_0_tensor_sha256, dict) or not isinstance(
        scenario_1_tensor_sha256, dict
    ):
        raise AssertionError("W4A8 live-input tensor hash records are malformed")
    allocator_checks = {
        phase["phase"]: phase["replay_allocator_records"] for phase in correctness
    }

    final_pointers = {name: tensor.data_ptr() for name, tensor in all_tensors.items()}
    if final_pointers != initial_pointers:
        raise AssertionError("allocation addresses changed")
    read_only_final = {
        name: _tensor_sha256(tensor) for name, tensor in read_only_inputs.items()
    }
    if read_only_final != read_only_initial:
        raise AssertionError("read-only inputs changed")
    artifact_verification_after = {
        label: verify_artifact(record) for label, record in provenance.items()
    }
    if artifact_verification_after != artifact_verification_before:
        raise RuntimeError("exact W4A8 cache artifacts changed during benchmark")
    gpu_mode_final = gpu_mode_snapshot(args.expected_physical_gpu)

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    root = REPO_ROOT
    source_path = root / "b12x/moe/fused/dynamic.py"
    result = {
        "schema": "b12x.w4a8.dynamic.cache_abba.v2",
        "evidence_status": args.evidence_status,
        "command": [sys.executable, *sys.argv],
        "worktree": str(root),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "kernel_source_sha256": _sha256(source_path),
        "gpu": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_index": args.expected_physical_gpu,
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "")),
            "sms": props.multi_processor_count,
            "capability": list(torch.cuda.get_device_capability()),
        },
        "gpu_mode_initial": gpu_mode_initial,
        "gpu_mode_final": gpu_mode_final,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvidia_cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
        },
        "shape": {
            "m": args.m,
            "experts": _EXPERTS,
            "hidden": _HIDDEN,
            "intermediate": _INTERMEDIATE,
            "topk": _TOPK,
            "tile_m": args.tile_m,
            "recipe": args.recipe,
            "activation": args.activation,
        },
        "object_provenance": provenance,
        "artifact_verification_before": artifact_verification_before,
        "artifact_verification_after": artifact_verification_after,
        "same_addresses": {
            "verified": True,
            "allocation_pointers": initial_pointers,
            "binding": "both exact objects use one relaunch closure and allocation set",
        },
        "same_address_arms": True,
        "same_address_across_arms": True,
        "fixed_workspace": True,
        "fixed_workspace_capacity": True,
        "fixed_allocation": True,
        "workspace_capacity": {
            name: {
                "numel": tensor.numel(),
                "bytes": tensor.numel() * tensor.element_size(),
                "dtype": str(tensor.dtype),
                "address": tensor.data_ptr(),
            }
            for name, tensor in sorted(mutable_allocations.items())
        },
        "read_only_inputs_unchanged": True,
        "read_only_input_sha256": read_only_initial,
        "read_only_inputs": {
            "unchanged": True,
            "sha256_before": read_only_initial,
            "sha256_after": read_only_final,
            "timed_live_scenario_0": {
                "unchanged": True,
                "sha256_before": scenario_0_before_timing["tensor_sha256"],
                "sha256_after": scenario_0_tensor_sha256,
            },
        },
        "live_input_scenarios_distinct": True,
        "live_input_mutation_changed_input": True,
        "live_input_mutation_changed_output": True,
        "live_input_mutation": {
            "schema": "b12x-live-input-mutation-v1",
            "captured_graph_reused": True,
            "in_place": True,
            "same_addresses": True,
            "mutated_in_place": True,
            "mutation_outside_timed_interval": True,
            "scenarios_distinct": True,
            "changed_input": True,
            "changed_inputs": changed_inputs,
            "changed_output": True,
            "changed_outputs": changed_outputs,
            "scenario_0": scenario_0_after_timing,
            "scenario_1": scenario_1,
            "scenario_0_sha256": scenario_0_tensor_sha256,
            "scenario_1_sha256": scenario_1_tensor_sha256,
            "scenario_0_output_sha256": correctness_post["output_sha256"],
            "scenario_1_output_sha256": correctness_live["output_sha256"],
            "allocation_addresses_stable": True,
        },
        "cuda_graph_replay": True,
        "cuda_graph_topology": topologies,
        "cuda_graph_topology_equal": topology_equal,
        "poisoned_outputs_overwritten": True,
        "arms_bit_exact": True,
        "allocator_stable": True,
        "zero_replay_allocations": True,
        "allocator_checks": allocator_checks,
        "allocation_before_timing": allocation_before_timing,
        "allocation_after_timing": allocation_after_timing,
        "l2_policy": ("warm-and-cold" if args.cold_l2 else "warm-only-diagnostic"),
        "l2_flush_bytes_requested": args.l2_flush_bytes,
        "precondition_cycles": args.precondition,
        "precondition_seconds": args.precondition_seconds,
        "maximum_precondition_seconds": args.maximum_precondition_seconds,
        "postcapture_warmup_cycles": args.warmup,
        "timed_abba_cycles": args.cycles,
        "timed_samples_per_arm": args.cycles * 2,
        "event_batch_cycles": args.event_batch_cycles,
        "replays_per_reported_sample": args.replays_per_reported_sample,
        "required_pstate": "P1",
        "max_sm_clock_delta_mhz": args.max_sm_clock_delta_mhz,
        "correctness": correctness,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gpu": result["gpu"],
                "conditions": {
                    name: {
                        "summaries": {
                            label: {
                                key: value
                                for key, value in summary.items()
                                if key != "samples_us"
                            }
                            for label, summary in condition["timings"][
                                "summaries"
                            ].items()
                        },
                        "ratios_b_over_a": condition["timings"]["ratios_b_over_a"],
                    }
                    for name, condition in conditions.items()
                },
                "correctness": correctness,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
