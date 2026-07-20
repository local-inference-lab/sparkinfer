#!/usr/bin/env python3
"""Run one CUTLASS arm of the frozen contiguous-attention graph corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from validation.cutlass_migration.diagnostics.paired.contiguous_attention import (
    _alternate_lengths,
    _case_from_manifest,
    _load_exact,
    _make_inputs,
    _offsets,
    _reference,
    _runtime_args,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    exact_artifact_evidence,
    gpu_mode_snapshot,
    json_sha256,
    single_graph_topology,
    tensor_sha256,
    time_single_graph_conditions,
    verify_artifact,
)
from validation.cutlass_migration.core.single_arm_e2e import (
    ReviewedCaseBinding,
    add_single_arm_arguments,
    begin_single_arm_session,
    bind_exact_artifact,
    build_exact_launch_plan,
    finish_single_arm_session,
    verify_case_compile_contract,
)
import sparkinfer.cute.compiler as cute_compiler


FAMILY = "contiguous_attention"
ARTIFACT_ROLE = "contiguous-attention"
INPUT_SCHEMA = "sparkinfer.contiguous_attention.end_to_end_input.v1"
CORRECTNESS_GATES = (
    "finite-output-and-lse",
    "guard-canaries",
    "live-metadata-and-tensor-replay",
    "nonzero-output",
    "torch-segmented-attention-reference",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    spec_hash: str
    seed: int = 20_260_718

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.name}"

    @property
    def input_contract(self) -> dict[str, object]:
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "compile_spec_hash": self.spec_hash,
            "source": {
                "generator": "torch.cuda.Generator",
                "distribution": "randn-float32-divide-4-to-compiled-dtype",
                "seed": self.seed,
            },
            "live_scenario": "q-k-v affine mutation plus alternate varlen offsets",
        }


CASES = (
    CaseSpec(
        "fixed",
        "9809517d008eb5113056aca8f5db06a31cf7315bade4885e5d50ff3a2682cca7",
    ),
    CaseSpec(
        "typed-smem-boundary",
        "555844a77dd811f5d0671bb837ba04788eb0419cd823a8f2bde85d67b01d1ef4",
    ),
    CaseSpec(
        "varlen-live-metadata",
        "61073abba7f383df4334e5196e0c5ced260c4b18bd4b63edb94d037bccdcac7a",
    ),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    return parser.parse_args()


def _guarded_like(
    shape: tuple[int, ...], dtype: torch.dtype, *, guard_value: float
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = 1
    for extent in shape:
        payload *= extent
    guard = 256
    storage = torch.full(
        (payload + 2 * guard,),
        guard_value,
        dtype=dtype,
        device="cuda",
    )
    return storage, storage[guard : guard + payload].view(shape)


def _assert_guards(
    storage: torch.Tensor, payload_numel: int, guard_value: float
) -> None:
    guard = (storage.numel() - payload_numel) // 2
    if guard <= 0 or not (
        torch.all(storage[:guard] == guard_value).item()
        and torch.all(storage[-guard:] == guard_value).item()
    ):
        raise AssertionError("contiguous-attention output guard changed")


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    difference = actual.float() - expected.float()
    finite = bool(torch.isfinite(actual).all().item())
    nonzero_count = int(torch.count_nonzero(actual).item())
    cosine = float(
        F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    )
    if (
        not finite
        or nonzero_count <= 0
        or not torch.isfinite(torch.tensor(cosine)).item()
    ):
        raise AssertionError("contiguous-attention output is nonfinite or all zero")
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3.5e-2)
    return {
        "finite": finite,
        "nonzero_count": nonzero_count,
        "max_abs": float(difference.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "cosine": cosine,
        "output_sha256": tensor_sha256(actual),
    }


def _run_case(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    compiled: object,
    provenance: Mapping[str, Any],
    artifact_before: Mapping[str, Any],
    precondition: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    warmup: int,
    replays: int,
    event_batch_replays: int,
    expected_physical_gpu: int,
    max_sm_clock_delta_mhz: float,
    l2_flush_bytes: int,
    replays_per_reported_sample: int,
) -> dict[str, object]:
    verify_case_compile_contract(
        case_id=spec.case_id,
        reviewed=reviewed,
        arm=arm,
        role=ARTIFACT_ROLE,
        provenance=provenance,
    )
    case = _case_from_manifest(dict(provenance))
    tensors = _make_inputs(case)
    q = tensors["q"]
    if case["kind"] == "fixed":
        lse_shape = (q.shape[0], q.shape[2], q.shape[1])
    else:
        lse_shape = (q.shape[1], q.shape[0])
    output_storage, output = _guarded_like(tuple(q.shape), q.dtype, guard_value=123.0)
    lse_storage, lse = _guarded_like(
        tuple(lse_shape), torch.float32, guard_value=-456.0
    )
    sink = (
        torch.linspace(-0.25, 0.5, q.shape[-2], dtype=torch.float32, device="cuda")
        if case["has_sink"]
        else torch.empty(1, dtype=torch.float32, device="cuda")
    )
    scale = float(q.shape[-1]) ** -0.5
    initial_live_hashes = {
        name: tensor_sha256(tensor) for name, tensor in sorted(tensors.items())
    }
    initial_sink_hash = tensor_sha256(sink)
    read_only_inputs_sha256 = json_sha256({"sink": initial_sink_hash})
    fixed_pointers = {
        **{name: tensor.data_ptr() for name, tensor in tensors.items()},
        "sink": sink.data_ptr(),
        "output_storage": output_storage.data_ptr(),
        "output": output.data_ptr(),
        "lse_storage": lse_storage.data_ptr(),
        "lse": lse.data_ptr(),
    }
    scenario_1_values = {
        name: tensor.float().mul(-0.75).add(0.03125).to(tensor.dtype).contiguous()
        for name, tensor in tensors.items()
        if name in {"q", "k", "v"}
    }
    if case["kind"] == "varlen":
        scenario_1_values["cu_q"] = _offsets(
            _alternate_lengths(
                int(case["q_shape"][0]),
                int(case["cu_q_shape"][0]) - 1,
                int(case["max_seqlen_q"]),
            )
        )
        scenario_1_values["cu_k"] = _offsets(
            _alternate_lengths(
                int(case["k_shape"][0]),
                int(case["cu_k_shape"][0]) - 1,
                int(case["max_seqlen_k"]),
            )
        )

    def launch() -> None:
        cute_compiler.run_compiled(
            compiled, _runtime_args(case, tensors, output, lse, sink, scale)
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        launch()
    stream.synchronize()
    topology = single_graph_topology(graph)
    if topology["kernel_node_count"] != 1:
        raise AssertionError(f"{spec.case_id}: expected one graph kernel: {topology}")
    if (
        reviewed.get("_discovery") is not True
        and topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")

    def replay_checked(expected: torch.Tensor, poison: float) -> dict[str, object]:
        with torch.cuda.stream(stream):
            output.fill_(poison)
            lse.fill_(poison)
        stream.synchronize()
        before = allocator_counters()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        after = allocator_counters()
        if before != after:
            raise AssertionError(f"{spec.case_id}: correctness replay allocated")
        if not torch.isfinite(lse).all().item():
            raise AssertionError(f"{spec.case_id}: graph left nonfinite LSE values")
        _assert_guards(output_storage, output.numel(), 123.0)
        _assert_guards(lse_storage, lse.numel(), -456.0)
        return _metrics(output, expected)

    expected_0 = _reference(case, tensors, sink if case["has_sink"] else None, scale)
    scenario_0 = replay_checked(expected_0, float("nan"))
    scenario_0_repeat = replay_checked(expected_0, -321.0)
    if scenario_0_repeat != scenario_0:
        raise AssertionError(f"{spec.case_id}: repeated scenario-0 output changed")

    conditions, allocation_records = time_single_graph_conditions(
        graph,
        precondition=precondition,
        warmup=warmup,
        replays=replays,
        stream=stream,
        l2_flush_bytes=l2_flush_bytes,
        replays_per_reported_sample=replays_per_reported_sample,
        event_batch_replays=event_batch_replays,
        precondition_seconds=precondition_seconds,
        maximum_precondition_seconds=maximum_precondition_seconds,
        mode_snapshot=lambda: gpu_mode_snapshot(expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
    )
    scenario_0_post = replay_checked(expected_0, float("nan"))
    if scenario_0_post != scenario_0:
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    if {
        name: tensor_sha256(tensor) for name, tensor in sorted(tensors.items())
    } != initial_live_hashes:
        raise AssertionError(f"{spec.case_id}: timed live input changed")

    for name, value in scenario_1_values.items():
        tensors[name].copy_(value)
    live_hashes = {
        name: tensor_sha256(tensor) for name, tensor in sorted(tensors.items())
    }
    if any(live_hashes[name] == initial_live_hashes[name] for name in tensors):
        raise AssertionError(f"{spec.case_id}: live mutation missed an input")
    expected_1 = _reference(case, tensors, sink if case["has_sink"] else None, scale)
    scenario_1 = replay_checked(expected_1, float("nan"))
    if scenario_1["output_sha256"] == scenario_0["output_sha256"]:
        raise AssertionError(f"{spec.case_id}: live input did not change output")
    if tensor_sha256(sink) != initial_sink_hash:
        raise AssertionError(f"{spec.case_id}: read-only sink changed")
    if {
        **{name: tensor.data_ptr() for name, tensor in tensors.items()},
        "sink": sink.data_ptr(),
        "output_storage": output_storage.data_ptr(),
        "output": output.data_ptr(),
        "lse_storage": lse_storage.data_ptr(),
        "lse": lse.data_ptr(),
    } != fixed_pointers:
        raise AssertionError(f"{spec.case_id}: captured addresses changed")
    if single_graph_topology(graph) != topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    artifact = exact_artifact_evidence(
        provenance,
        verification_before=artifact_before,
        verification_after=verify_artifact(provenance),
    )
    artifacts = [bind_exact_artifact(role=ARTIFACT_ROLE, evidence=artifact)]
    launch_plan = build_exact_launch_plan(
        case_id=spec.case_id,
        reviewed=reviewed,
        arm=arm,
        artifacts=artifacts,
        observed_roles=(ARTIFACT_ROLE,),
    )
    allocation = allocation_records["warm_l2"]
    return {
        "case_id": spec.case_id,
        "case_contract_sha256": reviewed["case_contract_sha256"],
        "input_sha256": json_sha256(spec.input_contract),
        "artifacts": artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": [],
        "correctness": {
            "independent_oracle": True,
            "oracle": "torch-segmented-attention-fp32-softmax",
            "passed": True,
            "finite": scenario_0["finite"],
            "nonzero_count": scenario_0["nonzero_count"],
            "gates": {gate: True for gate in CORRECTNESS_GATES},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": read_only_inputs_sha256,
            "output_sha256": scenario_0["output_sha256"],
        },
        "graph": {
            "capture_passed": True,
            "replay_passed": True,
            "topology_stable": True,
            "addresses_stable": True,
            "live_input_changed_output": True,
            "poison_overwrite_passed": True,
            **topology,
        },
        "allocation": {
            "fixed_workspace_capacity": True,
            "workspace_capacity_bytes": 0,
            "stable_addresses": True,
            "allocator_stable": True,
            "zero_replay_allocations": True,
            **allocation,
            "condition_counters": allocation_records,
        },
        "conditions": conditions,
    }


def main() -> int:
    args = _args()
    producer_path = Path(__file__).resolve()
    session = begin_single_arm_session(
        args,
        family=FAMILY,
        producer_path=producer_path,
        bindings=tuple(
            ReviewedCaseBinding(
                case_id=spec.case_id,
                input_sha256=json_sha256(spec.input_contract),
                correctness_gates=CORRECTNESS_GATES,
            )
            for spec in CASES
        ),
    )
    loaded: list[tuple[CaseSpec, object, dict[str, Any], dict[str, object]]] = []
    for spec in CASES:
        compiled, provenance = _load_exact(args.cache.resolve(), spec.spec_hash)
        if provenance["package_fingerprint"] != session.runtime_fingerprint:
            raise RuntimeError("exact object and frozen runtime fingerprints differ")
        loaded.append((spec, compiled, provenance, verify_artifact(provenance)))
    cases = [
        _run_case(
            spec=spec,
            reviewed=session.reviewed_cases[spec.case_id],
            arm=session.arm,
            compiled=compiled,
            provenance=provenance,
            artifact_before=artifact_before,
            precondition=args.precondition,
            precondition_seconds=args.precondition_seconds,
            maximum_precondition_seconds=args.maximum_precondition_seconds,
            warmup=args.warmup,
            replays=args.replays,
            event_batch_replays=args.event_batch_replays,
            expected_physical_gpu=session.expected_physical_gpu,
            max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
            l2_flush_bytes=args.l2_flush_bytes,
            replays_per_reported_sample=args.replays_per_reported_sample,
        )
        for spec, compiled, provenance, artifact_before in loaded
    ]
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
