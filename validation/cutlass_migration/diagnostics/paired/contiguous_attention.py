#!/usr/bin/env python3
"""Same-address CUDA-graph ABBA for cached contiguous-attention objects.

The two arms are immutable cache objects, so the CUDA context, input/output
addresses, graph topology, warmup, and timing order are shared.  Both the fixed
and live-metadata varlen compile specifications emitted by the migration corpus
are supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F

from benchmarks.common import make_l2_flush_fn
from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    artifact_provenance,
    gpu_mode_snapshot,
    time_conditions,
    verify_artifact,
)
from validation.cutlass_migration.paths import REPO_ROOT
import sparkinfer.cute.compiler as cute_compiler
from sparkinfer.cute.utils import current_cuda_stream, make_ptr


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    parser.add_argument("--a-cache", type=pathlib.Path, required=True)
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--b-cache", type=pathlib.Path, required=True)
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument("--spec-hash", required=True)
    parser.add_argument("--precondition", type=int, default=200)
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-precondition-seconds", type=float, default=60.0)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--event-batch-cycles", type=int, default=50)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    parser.add_argument(
        "--cold-l2", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _git_output(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_for_spec(
    cache: pathlib.Path, spec_hash: str
) -> tuple[pathlib.Path, dict]:
    matches: list[tuple[pathlib.Path, dict]] = []
    for path in cache.rglob("*.json"):
        if path.name.endswith(".ptx.json"):
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("compile_spec_hash") == spec_hash:
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one manifest for {spec_hash} in {cache}, got {len(matches)}"
        )
    return matches[0]


def _load_exact(cache: pathlib.Path, spec_hash: str) -> tuple[object, dict[str, Any]]:
    provenance = artifact_provenance(cache, spec_hash)
    previous = os.environ.get("SPARKINFER_CUTE_COMPILE_CACHE_DIR")
    os.environ["SPARKINFER_CUTE_COMPILE_CACHE_DIR"] = str(cache)
    try:
        compiled = cute_compiler._load_cute_compile_from_disk(
            str(provenance["cache_key"])
        )
    finally:
        if previous is None:
            os.environ.pop("SPARKINFER_CUTE_COMPILE_CACHE_DIR", None)
        else:
            os.environ["SPARKINFER_CUTE_COMPILE_CACHE_DIR"] = previous
    if compiled is None:
        raise RuntimeError(f"failed to load exact object {provenance['object_path']}")
    return compiled, provenance


def _legacy_fields(manifest: dict[str, Any]) -> tuple[str, list[Any]]:
    spec = json.loads(manifest["compile_spec_json"])
    facts = spec.get("facts")
    if (
        not isinstance(facts, list)
        or len(facts) != 4
        or facts[0] != "legacy"
        or not isinstance(facts[3], list)
    ):
        raise RuntimeError(
            "expected a legacy contiguous-attention compile specification"
        )
    values: dict[int, Any] = {}
    for field in facts[3]:
        if not isinstance(field, list) or len(field) != 3 or field[0] != "field":
            raise RuntimeError(f"malformed compile-spec field: {field!r}")
        name = str(field[1])
        if not name.startswith("arg"):
            raise RuntimeError(f"unexpected compile-spec field name: {name}")
        values[int(name[3:])] = field[2]
    ordered = [values[index] for index in range(len(values))]
    return str(spec["kernel"]), ordered


def _dtype_from_fact(value: Any) -> torch.dtype:
    rendered = json.dumps(value, sort_keys=True)
    if "bfloat16" in rendered:
        return torch.bfloat16
    if "float16" in rendered:
        return torch.float16
    raise TypeError(f"unsupported dtype fact: {value!r}")


def _cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise TypeError(dtype)


def _case_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    kernel_id, values = _legacy_fields(manifest)
    if kernel_id == "attention.contiguous.forward":
        if len(values) != 10:
            raise RuntimeError(f"unexpected fixed-attention fact count: {len(values)}")
        return {
            "kind": "fixed",
            "kernel_id": kernel_id,
            "q_shape": tuple(values[0]),
            "k_shape": tuple(values[1]),
            "v_shape": tuple(values[2]),
            "dtype": _dtype_from_fact(values[3]),
            "causal": bool(values[4]),
            "window": (int(values[5]), int(values[6])),
            "has_sink": bool(values[7]),
            "tile": (int(values[8]), int(values[9])),
        }
    if kernel_id == "attention.contiguous.varlen_forward":
        if len(values) != 14:
            raise RuntimeError(f"unexpected varlen-attention fact count: {len(values)}")
        return {
            "kind": "varlen",
            "kernel_id": kernel_id,
            "q_shape": tuple(values[0]),
            "k_shape": tuple(values[1]),
            "v_shape": tuple(values[2]),
            "cu_q_shape": tuple(values[3]),
            "cu_k_shape": tuple(values[4]),
            "dtype": _dtype_from_fact(values[5]),
            "causal": bool(values[6]),
            "window": (int(values[7]), int(values[8])),
            "has_sink": bool(values[9]),
            "max_seqlen_q": int(values[10]),
            "max_seqlen_k": int(values[11]),
            "tile": (int(values[12]), int(values[13])),
        }
    raise RuntimeError(f"unsupported kernel id {kernel_id!r}")


def _balanced_lengths(total: int, segments: int) -> tuple[int, ...]:
    if total < segments:
        raise ValueError(
            f"cannot split total={total} into {segments} nonempty segments"
        )
    base, extra = divmod(total, segments)
    return tuple(base + (index >= segments - extra) for index in range(segments))


def _offsets(lengths: tuple[int, ...]) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _alternate_lengths(
    total: int,
    segments: int,
    maximum: int,
) -> tuple[int, ...]:
    lengths = list(_balanced_lengths(total, segments))
    for donor in range(segments):
        if lengths[donor] <= 1:
            continue
        for receiver in range(segments):
            if receiver != donor and lengths[receiver] < maximum:
                lengths[donor] -= 1
                lengths[receiver] += 1
                return tuple(lengths)
    raise ValueError(
        f"cannot build a distinct valid varlen split for total={total}, "
        f"segments={segments}, maximum={maximum}"
    )


def _make_inputs(case: dict[str, Any]) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260718)
    tensors = {
        name: (
            torch.randn(
                tuple(case[f"{name}_shape"]),
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            .div_(4.0)
            .to(case["dtype"])
            .contiguous()
        )
        for name in ("q", "k", "v")
    }
    if case["kind"] == "varlen":
        q_segments = int(case["cu_q_shape"][0]) - 1
        k_segments = int(case["cu_k_shape"][0]) - 1
        if q_segments != k_segments:
            raise ValueError("benchmark requires matching q/k segment counts")
        tensors["cu_q"] = _offsets(
            _balanced_lengths(int(case["q_shape"][0]), q_segments)
        )
        tensors["cu_k"] = _offsets(
            _balanced_lengths(int(case["k_shape"][0]), k_segments)
        )
    return tensors


def _reference_segment(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window: tuple[int, int],
    sink: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    qf, kf, vf = q.float(), k.float(), v.float()
    groups = int(q.shape[-2]) // int(k.shape[-2])
    if groups != 1:
        kf = kf.repeat_interleave(groups, dim=-2)
        vf = vf.repeat_interleave(groups, dim=-2)
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale
    q_pos = torch.arange(q.shape[0], device=q.device).unsqueeze(1)
    k_pos = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    anchor = q_pos + int(k.shape[0]) - int(q.shape[0])
    keep = torch.ones((q.shape[0], k.shape[0]), dtype=torch.bool, device=q.device)
    if causal:
        keep &= k_pos <= anchor
    left, right = window
    if left != -1:
        keep &= k_pos >= anchor - left
    if right != -1:
        keep &= k_pos <= anchor + right
    scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
    if sink is not None:
        sink_logits = sink.float().view(q.shape[-2], 1, 1)
        scores = torch.cat(
            [scores, sink_logits.expand(q.shape[-2], q.shape[0], 1)], dim=-1
        )
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)[
            ..., : k.shape[0]
        ]
    else:
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.einsum("hqk,khd->qhd", probabilities, vf).to(q.dtype)


def _reference(
    case: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    sink: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    q, k, v = tensors["q"], tensors["k"], tensors["v"]
    if case["kind"] == "fixed":
        outputs = [
            _reference_segment(
                q[index],
                k[index],
                v[index],
                causal=case["causal"],
                window=case["window"],
                sink=sink,
                scale=scale,
            )
            for index in range(q.shape[0])
        ]
        return torch.stack(outputs)
    q_offsets = [int(value) for value in tensors["cu_q"].cpu().tolist()]
    k_offsets = [int(value) for value in tensors["cu_k"].cpu().tolist()]
    outputs = []
    for q0, q1, k0, k1 in zip(
        q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:], strict=True
    ):
        outputs.append(
            _reference_segment(
                q[q0:q1],
                k[k0:k1],
                v[k0:k1],
                causal=case["causal"],
                window=case["window"],
                sink=sink,
                scale=scale,
            )
        )
    return torch.cat(outputs)


def _runtime_args(
    case: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    output: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> tuple[Any, ...]:
    dtype = _cutlass_dtype(case["dtype"])
    args: list[Any] = [
        make_ptr(
            dtype, tensors[name].data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        for name in ("q", "k", "v")
    ]
    args.extend(
        [
            make_ptr(
                dtype, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float32, lse.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            ),
        ]
    )
    if case["kind"] == "varlen":
        args.extend(
            make_ptr(
                cutlass.Int32,
                tensors[name].data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            for name in ("cu_q", "cu_k")
        )
    args.extend(
        [
            make_ptr(
                cutlass.Float32,
                sink.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            float(scale),
            current_cuda_stream(),
        ]
    )
    return tuple(args)


def _graph_topology(graph: torch.cuda.CUDAGraph) -> dict[str, Any]:
    raw_graph = graph.raw_cuda_graph()
    error, _, node_count = cuda_driver.cuGraphGetNodes(raw_graph)
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGraphGetNodes count failed: {error}")
    error, nodes, returned = cuda_driver.cuGraphGetNodes(raw_graph, node_count)
    if error != cuda_driver.CUresult.CUDA_SUCCESS or returned != node_count:
        raise RuntimeError(f"cuGraphGetNodes failed: {error}, {returned}/{node_count}")
    metadata = []
    for index, node in enumerate(nodes):
        error, node_type = cuda_driver.cuGraphNodeGetType(node)
        if error != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphNodeGetType failed: {error}")
        item: dict[str, Any] = {"index": index, "type": node_type.name}
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
            item["type"] == "CU_GRAPH_NODE_TYPE_KERNEL" for item in metadata
        ),
        "nodes": metadata,
    }


def _topology_signature(topology: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_count": topology["node_count"],
        "kernel_node_count": topology["kernel_node_count"],
        "nodes": [
            {key: value for key, value in node.items() if key != "kernel_name"}
            for node in topology["nodes"]
        ],
    }


def main() -> None:
    args = _args()
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_device not in {"4", "5"}:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to physical GPU 4 or 5")
    expected_physical_gpu = int(visible_device)
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this benchmark requires SM120")
    if args.a_label == args.b_label:
        raise ValueError("A/B labels must differ")
    if min(args.precondition, args.warmup, args.event_batch_cycles) <= 0:
        raise ValueError("precondition, warmup, and event batch must be positive")
    if args.cycles < 500:
        raise ValueError("--cycles must be at least 500")
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
    gpu_mode_initial = gpu_mode_snapshot(expected_physical_gpu)

    root = REPO_ROOT
    compiled_a, manifest_a = _load_exact(args.a_cache.resolve(), args.spec_hash)
    compiled_b, manifest_b = _load_exact(args.b_cache.resolve(), args.spec_hash)
    artifact_verification_before = {
        args.a_label: verify_artifact(manifest_a),
        args.b_label: verify_artifact(manifest_b),
    }
    if manifest_a["compile_spec_json"] != manifest_b["compile_spec_json"]:
        raise RuntimeError("A/B compile specifications differ")
    case = _case_from_manifest(manifest_a)
    labels = (args.a_label, args.b_label)
    compiled = {labels[0]: compiled_a, labels[1]: compiled_b}
    tensors = _make_inputs(case)
    q = tensors["q"]
    output = torch.empty_like(q)
    if case["kind"] == "fixed":
        lse_shape = (q.shape[0], q.shape[2], q.shape[1])
    else:
        lse_shape = (q.shape[1], q.shape[0])
    lse = torch.empty(lse_shape, dtype=torch.float32, device="cuda")
    sink = (
        torch.linspace(-0.25, 0.5, q.shape[-2], dtype=torch.float32, device="cuda")
        if case["has_sink"]
        else torch.empty(1, dtype=torch.float32, device="cuda")
    )
    scale = float(q.shape[-1]) ** -0.5
    live_inputs = tensors
    read_only_inputs = {"sink": sink}
    scenario_0_sha256 = {
        name: _tensor_sha256(tensor) for name, tensor in live_inputs.items()
    }
    read_only_sha256_before = {
        name: _tensor_sha256(tensor) for name, tensor in read_only_inputs.items()
    }
    scenario_1_values = {
        name: (tensor.float().mul(-0.75).add(0.03125).to(tensor.dtype).contiguous())
        for name, tensor in tensors.items()
        if name in {"q", "k", "v"}
    }
    if case["kind"] == "varlen":
        q_segments = int(case["cu_q_shape"][0]) - 1
        k_segments = int(case["cu_k_shape"][0]) - 1
        scenario_1_values["cu_q"] = _offsets(
            _alternate_lengths(
                int(case["q_shape"][0]), q_segments, int(case["max_seqlen_q"])
            )
        )
        scenario_1_values["cu_k"] = _offsets(
            _alternate_lengths(
                int(case["k_shape"][0]), k_segments, int(case["max_seqlen_k"])
            )
        )

    fixed_pointers_before = {
        **{name: value.data_ptr() for name, value in tensors.items()},
        "output": output.data_ptr(),
        "lse": lse.data_ptr(),
        "sink": sink.data_ptr(),
    }

    def launch(label: str) -> None:
        cute_compiler.run_compiled(
            compiled[label], _runtime_args(case, tensors, output, lse, sink, scale)
        )

    stream = torch.cuda.Stream()
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for label in labels:
        with torch.cuda.stream(stream):
            launch(label)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            launch(label)
        stream.synchronize()
        graphs[label] = graph

    topologies = {label: _graph_topology(graph) for label, graph in graphs.items()}
    topology_equal = _topology_signature(topologies[labels[0]]) == _topology_signature(
        topologies[labels[1]]
    )
    if not topology_equal:
        raise AssertionError("A/B CUDA graph topology differs")

    allocator_checks: list[dict[str, Any]] = []

    def replay_scenario(
        scenario: str,
        expected: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        arm_outputs: dict[str, torch.Tensor] = {}
        arm_lse: dict[str, torch.Tensor] = {}
        scenario_correctness: dict[str, Any] = {}
        for label in labels:
            output.fill_(float("nan"))
            lse.fill_(float("nan"))
            counters_before = allocator_counters()
            graphs[label].replay()
            stream.synchronize()
            counters_after = allocator_counters()
            if counters_before != counters_after:
                raise AssertionError(
                    f"CUDA allocator state changed during {scenario} {label} replay"
                )
            allocator_checks.append(
                {
                    "scenario": scenario,
                    "label": label,
                    "before": counters_before,
                    "after": counters_after,
                    "unchanged": True,
                }
            )
            finite = bool(torch.isfinite(output).all()) and bool(
                torch.isfinite(lse).all()
            )
            if not finite:
                raise AssertionError(
                    f"{label} left non-finite output or LSE values in {scenario}"
                )
            torch.testing.assert_close(output, expected, rtol=2e-2, atol=3.5e-2)
            difference = output.float() - expected.float()
            scenario_correctness[label] = {
                "passed": True,
                "finite": True,
                "max_abs": float(difference.abs().max()),
                "rmse": float(torch.sqrt(torch.mean(difference.square()))),
                "cosine": float(
                    F.cosine_similarity(
                        output.float().flatten(), expected.float().flatten(), dim=0
                    )
                ),
            }
            arm_outputs[label] = output.clone()
            arm_lse[label] = lse.clone()
        if not torch.equal(arm_outputs[labels[0]], arm_outputs[labels[1]]):
            raise AssertionError(f"A/B outputs are not bit exact in {scenario}")
        if not torch.equal(arm_lse[labels[0]], arm_lse[labels[1]]):
            raise AssertionError(f"A/B LSE outputs are not bit exact in {scenario}")
        return scenario_correctness, arm_outputs, arm_lse

    expected_scenario_0 = _reference(
        case, tensors, sink if case["has_sink"] else None, scale
    )
    correctness_scenario_0_pre, scenario_0_outputs_pre, _ = replay_scenario(
        "scenario_0_pre", expected_scenario_0
    )

    if args.cold_l2:
        make_l2_flush_fn(True, args.l2_flush_bytes)
    allocator_before_timing = allocator_counters()
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
        mode_snapshot=lambda: gpu_mode_snapshot(expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
    )
    allocator_after_timing = allocator_counters()
    if allocator_before_timing != allocator_after_timing:
        raise AssertionError("CUDA allocator state changed during graph replay timing")
    compile_spec_hashes = {label: args.spec_hash for label in labels}
    for condition in conditions.values():
        condition["compile_spec_hashes"] = compile_spec_hashes
        condition["all_graph_spec_hashes"] = [args.spec_hash]

    scenario_0_sha256_after_timing = {
        name: _tensor_sha256(tensor) for name, tensor in live_inputs.items()
    }
    if scenario_0_sha256_after_timing != scenario_0_sha256:
        raise AssertionError("live input changed during scenario-0 timing")
    correctness_scenario_0_post, scenario_0_outputs_post, _ = replay_scenario(
        "scenario_0_post", expected_scenario_0
    )
    for label in labels:
        if not torch.equal(
            scenario_0_outputs_pre[label], scenario_0_outputs_post[label]
        ):
            raise AssertionError(
                f"{label} scenario-0 output changed across timing interval"
            )

    for name, tensor in live_inputs.items():
        tensor.copy_(scenario_1_values[name])
    scenario_1_sha256 = {
        name: _tensor_sha256(tensor) for name, tensor in live_inputs.items()
    }
    changed_inputs = {
        name: scenario_0_sha256[name] != scenario_1_sha256[name] for name in live_inputs
    }
    if not all(changed_inputs.values()):
        raise AssertionError(
            f"scenario-1 did not mutate every captured live input: {changed_inputs}"
        )
    expected_scenario_1 = _reference(
        case, tensors, sink if case["has_sink"] else None, scale
    )
    correctness_scenario_1, scenario_1_outputs, _ = replay_scenario(
        "scenario_1_live_mutation", expected_scenario_1
    )
    changed_outputs = {
        label: not torch.equal(
            scenario_0_outputs_post[label], scenario_1_outputs[label]
        )
        for label in labels
    }
    if not all(changed_outputs.values()):
        raise AssertionError(
            f"live input mutation did not change every arm output: {changed_outputs}"
        )

    read_only_sha256_after = {
        name: _tensor_sha256(tensor) for name, tensor in read_only_inputs.items()
    }
    if read_only_sha256_before != read_only_sha256_after:
        raise AssertionError("read-only input changed")
    fixed_pointers_after = {
        **{name: value.data_ptr() for name, value in tensors.items()},
        "output": output.data_ptr(),
        "lse": lse.data_ptr(),
        "sink": sink.data_ptr(),
    }
    if fixed_pointers_before != fixed_pointers_after:
        raise AssertionError("captured tensor addresses changed")

    artifact_verification_after = {
        args.a_label: verify_artifact(manifest_a),
        args.b_label: verify_artifact(manifest_b),
    }
    if artifact_verification_after != artifact_verification_before:
        raise RuntimeError("exact contiguous cache artifacts changed during benchmark")
    gpu_mode_final = gpu_mode_snapshot(expected_physical_gpu)

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    correctness = {
        "scenario_0_pre": correctness_scenario_0_pre,
        "scenario_0_post": correctness_scenario_0_post,
        "scenario_1_live_mutation": correctness_scenario_1,
    }
    result = {
        "schema": "sparkinfer.contiguous_attention.cache_abba.v2",
        "evidence_status": args.evidence_status,
        "provenance": {
            "command": [str(pathlib.Path(sys.executable).resolve()), *sys.argv],
            "git_commit": _git_output(root, "rev-parse", "HEAD"),
            "git_worktree": _git_output(root, "rev-parse", "--show-toplevel"),
            "git_status_short": _git_output(root, "status", "--short").splitlines(),
            "source_sha256": {
                "benchmark": _file_sha256(pathlib.Path(__file__).resolve()),
                "api": _file_sha256(root / "sparkinfer/attention/contiguous/api.py"),
                "forward": _file_sha256(root / "sparkinfer/attention/contiguous/forward.py"),
                "mask": _file_sha256(root / "sparkinfer/attention/contiguous/mask.py"),
                "compiler": _file_sha256(root / "sparkinfer/cute/compiler.py"),
            },
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "case": {
            **{key: value for key, value in case.items() if key != "dtype"},
            "dtype": str(case["dtype"]),
        },
        "labels": {"a": labels[0], "b": labels[1]},
        "manifests": {"a": manifest_a, "b": manifest_b},
        "artifact_verification_before": artifact_verification_before,
        "artifact_verification_after": artifact_verification_after,
        "gpu": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_index": expected_physical_gpu,
            "logical_device": torch.cuda.current_device(),
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "")),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "gpu_mode_initial": gpu_mode_initial,
        "gpu_mode_final": gpu_mode_final,
        "cuda_graph_replay": True,
        "cuda_graph_topology": topologies,
        "cuda_graph_topology_equal": topology_equal,
        "fixed_allocation": True,
        "fixed_workspace_capacity": True,
        "same_address_arms": True,
        "fixed_pointers": {
            "before": fixed_pointers_before,
            "after": fixed_pointers_after,
        },
        "read_only_inputs_unchanged": True,
        "read_only_inputs": {
            "unchanged": True,
            "sha256_before": read_only_sha256_before,
            "sha256_after": read_only_sha256_after,
            "timed_live_scenario_0": {
                "unchanged": True,
                "sha256_before": scenario_0_sha256,
                "sha256_after": scenario_0_sha256_after_timing,
            },
        },
        "live_input_scenarios_distinct": True,
        "live_input_mutation_changed_input": True,
        "live_input_mutation_changed_output": True,
        "live_input_mutation": {
            "captured_graph_reused": True,
            "in_place": True,
            "same_addresses": True,
            "scenarios_distinct": True,
            "changed_input": True,
            "changed_output": True,
            "scenario_0_sha256": scenario_0_sha256,
            "scenario_1_sha256": scenario_1_sha256,
            "changed_inputs": changed_inputs,
            "changed_outputs": changed_outputs,
            "scenario_0_output_sha256": {
                label: _tensor_sha256(value)
                for label, value in scenario_0_outputs_post.items()
            },
            "scenario_1_output_sha256": {
                label: _tensor_sha256(value)
                for label, value in scenario_1_outputs.items()
            },
            "allocation_addresses_stable": True,
        },
        "poisoned_outputs_overwritten": True,
        "arms_bit_exact": True,
        "correctness": correctness,
        "precondition": args.precondition,
        "precondition_seconds": args.precondition_seconds,
        "maximum_precondition_seconds": args.maximum_precondition_seconds,
        "max_sm_clock_delta_mhz": args.max_sm_clock_delta_mhz,
        "postcapture_warmup_cycles": args.warmup,
        "cycles": args.cycles,
        "event_batch_cycles": args.event_batch_cycles,
        "replays_per_reported_sample": args.replays_per_reported_sample,
        "cold_l2": args.cold_l2,
        "l2_flush_bytes": args.l2_flush_bytes,
        "allocator_stable_during_timing": True,
        "allocator_stable": True,
        "zero_replay_allocations": True,
        "allocator_before": allocator_before_timing,
        "allocator_after": allocator_after_timing,
        "allocator_checks": allocator_checks,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case": result["case"],
                "correctness": correctness,
                "ratios_b_over_a": {
                    name: condition["timings"]["ratios_b_over_a"]
                    for name, condition in conditions.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
