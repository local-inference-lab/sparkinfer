#!/usr/bin/env python3
"""Fail-closed exact-object CUDA-graph ABBA for W4A16 top-k sum."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping

import cutlass
import cutlass.cute as cute
import torch

from benchmarks.common import nvidia_smi_gpu_mode_snapshot, resolve_l2_flush_bytes
from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
)
from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument,
    require_target_gpu,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    graph_topology,
    load_exact,
    sha256_file,
    tensor_sha256,
    time_conditions,
    topology_signature,
    verify_artifact,
)
from validation.cutlass_migration.paths import CORE_ROOT, REPO_ROOT
import sparkinfer.cute.compiler as cute_compiler
from sparkinfer.cute.runtime_control import (
    freeze_kernel_resolution,
    kernel_resolution_frozen,
    unfreeze_kernel_resolution,
)
from sparkinfer.cute.utils import current_cuda_stream, make_ptr
from sparkinfer.moe.fused.w4a16.kernel import W4A16TopKSumCompileResult


_REQUIRED_DECODE_M = (1, 2, 4, 8, 23, 33, 80)
_REQUIRED_PREFILL_M = (8192, 16384, 24576, 32768)
_REQUIRED_M = (*_REQUIRED_DECODE_M, *_REQUIRED_PREFILL_M)
_CUTLASS_COMPONENTS = {
    "cutlass_dsl": "nvidia-cutlass-dsl",
    "cutlass_dsl_libs_base": "nvidia-cutlass-dsl-libs-base",
    "cutlass_dsl_libs_core": "nvidia-cutlass-dsl-libs-core",
    "cutlass_dsl_libs_cu12": "nvidia-cutlass-dsl-libs-cu12",
    "cutlass_dsl_libs_cu13": "nvidia-cutlass-dsl-libs-cu13",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    add_target_gpu_argument(parser)
    parser.add_argument("--a-cache", type=Path, required=True)
    parser.add_argument("--a-cache-key")
    parser.add_argument("--a-fingerprint", required=True)
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--a-cutlass-version", default="4.5.2")
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--b-cache-key")
    parser.add_argument("--b-fingerprint", required=True)
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument("--b-cutlass-version", default="4.6.0")
    parser.add_argument(
        "--m",
        type=int,
        action="append",
        help="repeat for the exact required matrix; omission selects the full matrix",
    )
    parser.add_argument("--hidden-size", type=int, default=2688)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--precondition", type=int, default=1000)
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-precondition-seconds", type=float, default=30.0)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--prefill-cycles", type=int, default=500)
    parser.add_argument("--prefill-threshold", type=int, default=1024)
    parser.add_argument("--event-batch-cycles", type=int, default=100)
    parser.add_argument(
        "--decode-replays-per-reported-sample",
        type=int,
        default=64,
        help="independently event-bracketed graph replays averaged per decode sample",
    )
    parser.add_argument(
        "--prefill-replays-per-reported-sample",
        type=int,
        default=1,
        help="independently event-bracketed graph replays averaged per prefill sample",
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument(
        "--balanced-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="required; alternate ABBA and BAAB cycles",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _exact_cutlass_toolchain(
    provenance: Mapping[str, Any], expected_version: str
) -> dict[str, object]:
    entries = {
        str(entry[0]): entry[1]
        for entry in provenance["toolchain"]
        if isinstance(entry, list) and len(entry) >= 2
    }
    missing_entries = sorted(set(_CUTLASS_COMPONENTS) - set(entries))
    if missing_entries:
        raise RuntimeError(
            f"manifest omits CUTLASS toolchain entries: {missing_entries}"
        )
    exact = {
        distribution: entries[manifest_name]
        for manifest_name, distribution in _CUTLASS_COMPONENTS.items()
    }
    if exact["nvidia-cutlass-dsl"] != expected_version:
        raise RuntimeError(
            "CUTLASS DSL version mismatch: "
            f"{exact['nvidia-cutlass-dsl']} != {expected_version}"
        )
    invalid = {
        name: version
        for name, version in exact.items()
        if version not in {expected_version, "missing"}
    }
    if invalid:
        raise RuntimeError(
            f"mixed CUTLASS package versions for {expected_version}: {invalid}"
        )
    return exact


def _verified_artifact(provenance: Mapping[str, Any]) -> dict[str, object]:
    return {"status": "verified", **verify_artifact(provenance)}


def _load_arm(
    cache: Path,
    *,
    cache_key: str | None,
    fingerprint: str,
    expected_cutlass: str,
    topk: int,
    hidden_size: int,
) -> tuple[object, dict[str, Any]]:
    expected_spec = cute_compiler.KernelCompileSpec.from_key(
        "moe.w4a16.topk_sum",
        1,
        ("w4a16_topk_sum", "bf16", topk, hidden_size),
    )
    compiled, loaded = load_exact(
        cache,
        expected_spec.hash_key,
        cache_key=cache_key,
    )
    if loaded["kernel_id"] != "moe.w4a16.topk_sum":
        raise RuntimeError(f"unexpected kernel id: {loaded['kernel_id']}")
    if loaded["package_fingerprint"] != fingerprint:
        raise RuntimeError(
            "package fingerprint mismatch: "
            f"{loaded['package_fingerprint']} != {fingerprint}"
        )
    provenance = dict(loaded)
    manifest = json.loads(
        Path(str(loaded["manifest_path"])).read_text(encoding="utf-8")
    )
    provenance["exact_cutlass_toolchain"] = _exact_cutlass_toolchain(
        loaded, expected_cutlass
    )
    provenance["expected_cutlass_dsl"] = expected_cutlass
    provenance["artifact_source_identity"] = {
        "package_fingerprint": loaded["package_fingerprint"],
        "target": manifest.get("target"),
        "target_identity": manifest.get("target_identity"),
        "semantic_key": manifest.get("semantic_key"),
        "compile_environment": manifest.get("compile_environment"),
        "manifest_sha256": loaded["manifest_sha256"],
        "object_sha256": loaded["object_sha256"],
    }
    return compiled, provenance


def _mode_snapshot(expected_physical_gpu: int) -> dict[str, object]:
    snapshot = nvidia_smi_gpu_mode_snapshot()
    if not snapshot.get("available"):
        raise RuntimeError(f"nvidia-smi GPU-mode snapshot unavailable: {snapshot}")
    fields = snapshot.get("fields")
    if not isinstance(fields, dict):
        raise RuntimeError(f"GPU-mode snapshot has no field mapping: {snapshot}")
    if str(fields.get("index")) != str(expected_physical_gpu):
        raise RuntimeError(
            "GPU-mode snapshot selected the wrong physical GPU: "
            f"{fields.get('index')} != {expected_physical_gpu}"
        )
    if fields.get("uuid") != snapshot.get("nvidia_smi_uuid"):
        raise RuntimeError(f"GPU-mode snapshot UUID mismatch: {snapshot}")
    return snapshot


@contextmanager
def _resolution_frozen() -> Iterator[None]:
    already_frozen = kernel_resolution_frozen()
    if not already_frozen:
        freeze_kernel_resolution("exact-object W4A16 top-k-sum ABBA")
    try:
        yield
    finally:
        if not already_frozen:
            unfreeze_kernel_resolution()


def _case_kind(m: int) -> str:
    return "decode" if m < 1024 else "prefill"


def _pointer_snapshot(
    fc2: torch.Tensor, output: torch.Tensor
) -> dict[str, dict[str, object]]:
    return {
        "fc2": {
            "address": fc2.data_ptr(),
            "shape": list(fc2.shape),
            "dtype": str(fc2.dtype),
            "capacity_bytes": fc2.numel() * fc2.element_size(),
        },
        "output": {
            "address": output.data_ptr(),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "capacity_bytes": output.numel() * output.element_size(),
        },
    }


def _correctness(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    finite = bool(torch.isfinite(actual).all())
    nonzero = int(torch.count_nonzero(actual))
    max_abs = float((actual.float() - expected.float()).abs().max())
    actual_flat = actual.float().reshape(-1)
    expected_flat = expected.float().reshape(-1)
    denominator = actual_flat.norm() * expected_flat.norm()
    cosine = float((actual_flat * expected_flat).sum() / denominator.clamp_min(1e-24))
    passed = finite and nonzero > 0 and max_abs == 0.0
    if not passed:
        raise AssertionError(
            f"top-k-sum correctness failed: finite={finite}, nonzero={nonzero}, "
            f"max_abs={max_abs}, cosine={cosine}"
        )
    return {
        "passed": True,
        "finite": finite,
        "nonzero": nonzero,
        "max_abs": max_abs,
        "cosine": cosine,
    }


def _run_case(
    *,
    m: int,
    hidden_size: int,
    topk: int,
    launches: Mapping[str, object],
    provenance: Mapping[str, Mapping[str, Any]],
    labels: tuple[str, str],
    precondition: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    warmup: int,
    cycles: int,
    event_batch_cycles: int,
    l2_flush_bytes: int,
    replays_per_reported_sample: int,
    expected_physical_gpu: int,
    max_sm_clock_delta_mhz: float,
) -> dict[str, object]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(91700 + m)
    fc2 = torch.randn(
        (m, topk, hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    output = torch.empty((m, hidden_size), dtype=torch.bfloat16, device="cuda")
    expected_f32 = fc2[:, 0, :].float()
    for route in range(1, topk):
        expected_f32.add_(fc2[:, route, :].float())
    expected = expected_f32.to(torch.bfloat16)
    del expected_f32
    pointers_before = _pointer_snapshot(fc2, output)
    input_sha256_before = tensor_sha256(fc2)
    scenario_0_input_sha256 = {"fc2": input_sha256_before}
    read_only_sha256_before = {"oracle_expected": tensor_sha256(expected)}
    artifacts_before = {
        label: _verified_artifact(provenance[label]) for label in labels
    }

    fc2_ptr = make_ptr(
        cutlass.BFloat16,
        fc2.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    output_ptr = make_ptr(
        cutlass.BFloat16,
        output.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    def run(label: str) -> None:
        launches[label](fc2_ptr, output_ptr, m, current_cuda_stream())

    stream = torch.cuda.Stream()
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    topologies: dict[str, dict[str, object]] = {}
    for label in labels:
        with torch.cuda.stream(stream):
            run(label)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            run(label)
        stream.synchronize()
        topology = graph_topology(graph)
        if int(topology["kernel_node_count"]) != 1:
            raise AssertionError(
                f"{label}: expected one exact-object graph kernel, got {topology}"
            )
        graphs[label] = graph
        topologies[label] = topology
    topology_equal = topology_signature(topologies[labels[0]]) == topology_signature(
        topologies[labels[1]]
    )
    if not topology_equal:
        raise AssertionError("top-k-sum arm CUDA graph topologies differ")

    replay_allocator_records: list[dict[str, object]] = []

    def replay_checked(
        label: str,
        expected_value: torch.Tensor,
        *,
        phase: str,
    ) -> tuple[dict[str, object], torch.Tensor]:
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
        stream.synchronize()
        allocator_before = allocator_counters()
        with torch.cuda.stream(stream):
            graphs[label].replay()
        stream.synchronize()
        allocator_after = allocator_counters()
        if allocator_after != allocator_before:
            raise AssertionError(
                f"{label}: CUDA allocator changed during top-k replay: "
                f"{allocator_before} != {allocator_after}"
            )
        if bool(torch.isnan(output).any()):
            raise AssertionError(f"{label} did not overwrite poisoned output")
        replay_allocator_records.append(
            {
                "phase": phase,
                "label": label,
                "before": allocator_before,
                "after": allocator_after,
            }
        )
        return _correctness(output, expected_value), output.clone()

    correctness: dict[str, dict[str, dict[str, object]]] = {
        "baseline": {},
        "mutated_live_input": {},
        "post_timing": {},
    }
    baseline_outputs: dict[str, torch.Tensor] = {}
    for label in labels:
        correctness["baseline"][label], baseline_outputs[label] = replay_checked(
            label, expected, phase="scenario_0"
        )
    if not torch.equal(baseline_outputs[labels[0]], baseline_outputs[labels[1]]):
        raise AssertionError("top-k-sum arms are not bit exact")
    scenario_0_output_sha256 = {
        label: tensor_sha256(baseline_outputs[label]) for label in labels
    }

    fc2.neg_()
    scenario_1_input_sha256 = {"fc2": tensor_sha256(fc2)}
    changed_inputs = {
        "fc2": scenario_1_input_sha256["fc2"] != scenario_0_input_sha256["fc2"]
    }
    if not all(changed_inputs.values()):
        raise AssertionError(f"scenario-1 top-k input did not change: {changed_inputs}")
    pointers_scenario_1 = _pointer_snapshot(fc2, output)
    if pointers_scenario_1 != pointers_before:
        raise AssertionError("top-k live input was not mutated in place")
    mutated_expected = -expected
    mutated_outputs: dict[str, torch.Tensor] = {}
    for label in labels:
        correctness["mutated_live_input"][label], mutated_outputs[label] = (
            replay_checked(label, mutated_expected, phase="scenario_1")
        )
    if not torch.equal(mutated_outputs[labels[0]], mutated_outputs[labels[1]]):
        raise AssertionError("mutated-input top-k-sum arms are not bit exact")
    scenario_1_output_sha256 = {
        label: tensor_sha256(mutated_outputs[label]) for label in labels
    }
    changed_outputs = {
        label: scenario_1_output_sha256[label] != scenario_0_output_sha256[label]
        for label in labels
    }
    if not all(changed_outputs.values()):
        raise AssertionError(
            f"live fc2 mutation did not change every top-k arm: {changed_outputs}"
        )
    fc2.neg_()
    if tensor_sha256(fc2) != input_sha256_before:
        raise AssertionError("live fc2 input was not restored before timing")
    del baseline_outputs, mutated_outputs

    timed_input_sha256_before = {"fc2": tensor_sha256(fc2)}
    conditions = time_conditions(
        graphs,
        labels=labels,
        precondition=precondition,
        warmup=warmup,
        cycles=cycles,
        event_batch_cycles=event_batch_cycles,
        stream=stream,
        cold_l2=True,
        l2_flush_bytes=l2_flush_bytes,
        replays_per_reported_sample=replays_per_reported_sample,
        precondition_seconds=precondition_seconds,
        maximum_precondition_seconds=maximum_precondition_seconds,
        mode_snapshot=lambda: _mode_snapshot(expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
    )
    timed_input_sha256_after = {"fc2": tensor_sha256(fc2)}
    if timed_input_sha256_after != timed_input_sha256_before:
        raise AssertionError("scenario-0 top-k input changed during timed replay")
    compile_spec_hashes = {
        label: str(provenance[label]["compile_spec_hash"]) for label in labels
    }
    all_graph_spec_hashes = sorted(set(compile_spec_hashes.values()))
    for condition in conditions.values():
        condition["compile_spec_hashes"] = compile_spec_hashes
        condition["all_graph_spec_hashes"] = all_graph_spec_hashes
        for label in labels:
            count = int(condition["timings"]["summaries"][label]["count"])
            if count < 1000:
                raise AssertionError(f"{label}: only {count} reported timing samples")

    post_timing_output_sha256: dict[str, str] = {}
    for label in labels:
        metric, post_timing_output = replay_checked(
            label, expected, phase="post_timing"
        )
        correctness["post_timing"][label] = metric
        post_timing_output_sha256[label] = tensor_sha256(post_timing_output)
        del post_timing_output
    pointers_after = _pointer_snapshot(fc2, output)
    if pointers_after != pointers_before:
        raise AssertionError("top-k-sum tensor pointers/capacities changed")
    input_sha256_after = tensor_sha256(fc2)
    if input_sha256_after != input_sha256_before:
        raise AssertionError("top-k-sum input changed during timed replay")
    read_only_sha256_after = {"oracle_expected": tensor_sha256(expected)}
    if read_only_sha256_after != read_only_sha256_before:
        raise AssertionError("top-k oracle tensor changed during benchmark")
    artifacts_after = {label: _verified_artifact(provenance[label]) for label in labels}
    if artifacts_after != artifacts_before:
        raise AssertionError("top-k-sum cached artifacts changed during benchmark")
    allocator_checks = list(replay_allocator_records)
    allocator_checks.extend(
        {
            "phase": f"timing_{condition_name}",
            "before": condition["allocator_before"],
            "after": condition["allocator_after"],
        }
        for condition_name, condition in conditions.items()
    )

    return {
        "shape": {"m": m, "topk": topk, "hidden_size": hidden_size},
        "serving_regime": _case_kind(m),
        "compile_spec_hashes": compile_spec_hashes,
        "all_graph_spec_hashes": all_graph_spec_hashes,
        "object_provenance": provenance,
        "artifact_verification_before": artifacts_before,
        "artifact_verification_after": artifacts_after,
        "stable_tensor_pointers": pointers_before,
        "same_address_arms": True,
        "same_address_across_arms": True,
        "graph_reuse": {
            "captured_graph_reused": True,
            "in_place": True,
            "same_addresses": True,
            "allocation_addresses_stable": True,
            "addresses_before": pointers_before,
            "addresses_scenario_1": pointers_scenario_1,
            "addresses_after": pointers_after,
        },
        "fixed_workspace_capacity": True,
        "fixed_allocation": True,
        "cuda_graph_replay": True,
        "cuda_graph_topology": topologies,
        "cuda_graph_topology_equal": True,
        "allocator_before": allocator_checks[0]["before"],
        "allocator_after": allocator_checks[0]["after"],
        "allocator_checks": allocator_checks,
        "allocator_stable": True,
        "zero_replay_allocations": True,
        "input_sha256": input_sha256_before,
        "input_immutable": True,
        "read_only_inputs_unchanged": True,
        "read_only_inputs_immutable": True,
        "read_only_inputs": {
            "unchanged": True,
            "sha256_before": read_only_sha256_before,
            "sha256_after": read_only_sha256_after,
            "timed_live_scenario_0": {
                "unchanged": True,
                "sha256_before": timed_input_sha256_before,
                "sha256_after": timed_input_sha256_after,
            },
        },
        "poisoned_outputs_overwritten": True,
        "poisoned_output_overwritten": True,
        "arms_bit_exact": True,
        "live_input_scenarios_distinct": True,
        "live_input_mutation_changed_input": True,
        "live_input_mutation_changed_output": True,
        "live_input_mutation": {
            "captured_graph_reused": True,
            "in_place": True,
            "same_addresses": True,
            "allocation_addresses_stable": True,
            "scenarios_distinct": True,
            "changed_input": True,
            "changed_inputs": changed_inputs,
            "changed_output": True,
            "changed_outputs": changed_outputs,
            "scenario_0_sha256": scenario_0_input_sha256,
            "scenario_1_sha256": scenario_1_input_sha256,
            "scenario_0_output_sha256": scenario_0_output_sha256,
            "scenario_1_output_sha256": scenario_1_output_sha256,
        },
        "post_timing_output_sha256": post_timing_output_sha256,
        "correctness": correctness,
        "precondition": precondition,
        "warmup": warmup,
        "cycles": cycles,
        "event_batch_cycles": event_batch_cycles,
        "replays_per_reported_sample": replays_per_reported_sample,
        "conditions": conditions,
    }


def main() -> None:
    args = _args()
    gpu = require_target_gpu(args.expected_physical_gpu)
    labels = (args.a_label, args.b_label)
    if labels[0] == labels[1]:
        raise ValueError("arm labels must differ")
    matrix = tuple(sorted(set(args.m or _REQUIRED_M)))
    if args.m is not None and len(args.m) != len(set(args.m)):
        raise ValueError("duplicate --m values are not allowed")
    if matrix != _REQUIRED_M:
        raise ValueError(
            f"--m must cover the exact decode+prefill matrix {_REQUIRED_M}; got {matrix}"
        )
    if args.topk <= 0 or args.hidden_size <= 0:
        raise ValueError("topk and hidden size must be positive")
    if args.cycles < 500 or args.prefill_cycles < 500:
        raise ValueError("decode and prefill cycles must each be at least 500")
    if args.precondition < 1 or args.warmup < 1 or args.event_batch_cycles < 1:
        raise ValueError("precondition, warmup, and event batch must be positive")
    if args.precondition_seconds < 5.0:
        raise ValueError("precondition seconds must be at least 5")
    if not (args.precondition_seconds <= args.maximum_precondition_seconds <= 60.0):
        raise ValueError("maximum precondition seconds must cover the minimum and <=60")
    if not 0.0 < args.max_sm_clock_delta_mhz <= 60.0:
        raise ValueError("maximum SM clock delta must be in (0, 60]")
    if (
        args.decode_replays_per_reported_sample < 1
        or args.prefill_replays_per_reported_sample < 1
    ):
        raise ValueError("replays per reported sample must be positive")
    if not args.balanced_order:
        raise ValueError("balanced ABBA/BAAB ordering is required")
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    if l2_flush_bytes <= 0:
        raise RuntimeError("cold-L2 evidence requires a positive flush capacity")

    initial_mode = _mode_snapshot(int(args.expected_physical_gpu))
    compile_cache_initial = cute_compiler.compile_cache_info()
    launch_a, provenance_a = _load_arm(
        args.a_cache,
        cache_key=args.a_cache_key,
        fingerprint=args.a_fingerprint,
        expected_cutlass=args.a_cutlass_version,
        topk=args.topk,
        hidden_size=args.hidden_size,
    )
    launch_b, provenance_b = _load_arm(
        args.b_cache,
        cache_key=args.b_cache_key,
        fingerprint=args.b_fingerprint,
        expected_cutlass=args.b_cutlass_version,
        topk=args.topk,
        hidden_size=args.hidden_size,
    )
    if provenance_a["compile_spec_json"] != provenance_b["compile_spec_json"]:
        raise AssertionError("top-k-sum arm compile specifications differ")
    artifacts = {labels[0]: provenance_a, labels[1]: provenance_b}
    launches = {
        labels[0]: W4A16TopKSumCompileResult(
            compiled=launch_a,
            m=0,
            topk=args.topk,
            hidden_size=args.hidden_size,
        ).compiled,
        labels[1]: W4A16TopKSumCompileResult(
            compiled=launch_b,
            m=0,
            topk=args.topk,
            hidden_size=args.hidden_size,
        ).compiled,
    }
    compile_cache_after_load = cute_compiler.compile_cache_info()
    if int(compile_cache_after_load["compile_misses"]) != int(
        compile_cache_initial["compile_misses"]
    ):
        raise AssertionError("loading exact top-k-sum objects triggered compilation")

    with _resolution_frozen():
        cases = [
            _run_case(
                m=m,
                hidden_size=args.hidden_size,
                topk=args.topk,
                launches=launches,
                provenance=artifacts,
                labels=labels,
                precondition=args.precondition,
                precondition_seconds=args.precondition_seconds,
                maximum_precondition_seconds=args.maximum_precondition_seconds,
                warmup=args.warmup,
                cycles=(
                    args.prefill_cycles if m >= args.prefill_threshold else args.cycles
                ),
                event_batch_cycles=args.event_batch_cycles,
                l2_flush_bytes=l2_flush_bytes,
                replays_per_reported_sample=(
                    args.prefill_replays_per_reported_sample
                    if m >= args.prefill_threshold
                    else args.decode_replays_per_reported_sample
                ),
                expected_physical_gpu=int(args.expected_physical_gpu),
                max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
            )
            for m in matrix
        ]
    compile_cache_final = cute_compiler.compile_cache_info()
    if int(compile_cache_final["compile_misses"]) != int(
        compile_cache_initial["compile_misses"]
    ):
        raise AssertionError("top-k-sum ABBA triggered a CUTLASS compilation")
    final_mode = _mode_snapshot(int(args.expected_physical_gpu))
    if int(final_mode["captured_unix_ns"]) <= int(initial_mode["captured_unix_ns"]):
        raise AssertionError("GPU-mode snapshots are not ordered")

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    source_paths = {
        "benchmark": Path(__file__).resolve(),
        "kernel": REPO_ROOT / "sparkinfer/moe/fused/w4a16/kernel.py",
        "compiler": REPO_ROOT / "sparkinfer/cute/compiler.py",
        "exact_cache_abba": CORE_ROOT / "exact_cache_abba.py",
        "gpu_scope": CORE_ROOT / "gpu_scope.py",
    }
    result = {
        "schema": "sparkinfer.w4a16.topk_sum.cache_abba.v1",
        "evidence_status": args.evidence_status,
        "command": [sys.executable, *sys.argv],
        "labels": {"a": labels[0], "b": labels[1]},
        "gpu": {
            **gpu,
            "logical_index": torch.cuda.current_device(),
            "sms": properties.multi_processor_count,
            "capability": list(torch.cuda.get_device_capability()),
        },
        "gpu_mode_initial": initial_mode,
        "gpu_mode_final": final_mode,
        "worktree": str(REPO_ROOT),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--short"),
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
        "cache": {
            "a": str(args.a_cache.resolve()),
            "a_fingerprint": args.a_fingerprint,
            "a_info": provenance_a,
            "b": str(args.b_cache.resolve()),
            "b_fingerprint": args.b_fingerprint,
            "b_info": provenance_b,
        },
        "required_decode_m": list(_REQUIRED_DECODE_M),
        "required_prefill_m": list(_REQUIRED_PREFILL_M),
        "matrix_m": list(matrix),
        "order": [labels[0], labels[1], labels[1], labels[0]],
        "alternate_order": [labels[1], labels[0], labels[0], labels[1]],
        "balanced_order": True,
        "event_batch_cycles": args.event_batch_cycles,
        "l2_flush_bytes": l2_flush_bytes,
        "no_recompilation": True,
        "compile_cache_initial": compile_cache_initial,
        "compile_cache_after_artifact_load": compile_cache_after_load,
        "compile_cache_final": compile_cache_final,
        "cases": cases,
        "all_correct": True,
        "all_graph_topologies_equal": True,
        "all_same_address_arms": True,
        "all_fixed_workspace_capacity": True,
        "all_allocator_stable": True,
        "all_zero_replay_allocations": True,
        "all_input_immutable": True,
        "all_read_only_inputs_immutable": True,
        "all_poisoned_outputs_overwritten": True,
        "all_arms_bit_exact": True,
        "all_live_input_scenarios_distinct": True,
        "all_live_input_mutations_changed_input": True,
        "all_live_input_mutations_changed_output": True,
        "all_evidence_gates_passed": True,
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
                "matrix_m": list(matrix),
                "all_evidence_gates_passed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
