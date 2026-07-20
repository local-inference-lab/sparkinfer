#!/usr/bin/env python3
"""Run one CUTLASS arm of the frozen NSA indexer production graph corpus."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from validation.cutlass_migration.diagnostics.paired.nsa_indexer import (
    _DEFINITIONS,
    _build_runtime,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    exact_artifact_evidence,
    gpu_mode_snapshot,
    json_sha256,
    load_exact,
    pin_module_launches,
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


FAMILY = "nsa_indexer"
INPUT_SCHEMA = "sparkinfer.nsa_indexer.end_to_end_input.v1"
CORRECTNESS_GATES = (
    "finite-checked-regions",
    "independent-torch-reference",
    "live-input-replay",
    "nonzero-checked-regions",
    "poison-overwrite",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    all_specs: tuple[str, ...]

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.name}"

    @property
    def input_contract(self) -> dict[str, object]:
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "builder": self.name,
            "compile_spec_hashes": list(self.all_specs),
            "live_scenarios": [0, 1],
        }


CASES = tuple(
    CaseSpec(name, tuple(definition.all_specs))
    for name, definition in _DEFINITIONS.items()
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    return parser.parse_args()


def _role(spec: CaseSpec, spec_hash: str) -> str:
    return f"kernel-{spec.all_specs.index(spec_hash)}-{spec_hash[:12]}"


def _snapshot_hash(snapshot: Mapping[str, torch.Tensor]) -> str:
    return json_sha256(
        {name: tensor_sha256(tensor) for name, tensor in sorted(snapshot.items())}
    )


def _nonzero_count(value: object) -> int:
    if isinstance(value, Mapping):
        return sum(
            int(item)
            if key in {"nonzero", "nonzero_count"}
            and isinstance(item, int)
            and not isinstance(item, bool)
            else _nonzero_count(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(_nonzero_count(item) for item in value)
    return 0


def _capture(
    runtime: Any,
    compiled: Mapping[str, object],
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, tuple[str, ...]]:
    observed: list[str] = []
    with ExitStack() as stack:
        for module, specs in runtime.modules.items():
            stack.enter_context(
                pin_module_launches(
                    module,
                    {spec_hash: compiled[spec_hash] for spec_hash in specs},
                    observed,
                )
            )
        stack.enter_context(runtime.launch_context())
        with torch.cuda.stream(stream):
            runtime.launch()
        stream.synchronize()
        observed.clear()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            runtime.launch()
        stream.synchronize()
    if not observed or set(observed) != set(runtime.definition.all_specs):
        raise RuntimeError(
            f"production route observed {observed}, "
            f"expected {runtime.definition.all_specs}"
        )
    return graph, tuple(observed)


def _run_case(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    compiled: Mapping[str, object],
    provenance: Mapping[str, Mapping[str, Any]],
    artifact_before: Mapping[str, Mapping[str, Any]],
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
    for spec_hash in spec.all_specs:
        verify_case_compile_contract(
            case_id=spec.case_id,
            reviewed=reviewed,
            arm=arm,
            role=_role(spec, spec_hash),
            provenance=provenance[spec_hash],
        )

    runtime = _build_runtime(spec.name)
    if tuple(runtime.definition.all_specs) != spec.all_specs:
        raise RuntimeError(f"{spec.case_id}: runtime compile-spec set drifted")
    fixed_pointers = {
        name: tensor.data_ptr() for name, tensor in runtime.stable_tensors.items()
    }
    read_only_hashes = {
        name: tensor_sha256(tensor) for name, tensor in runtime.read_only.items()
    }
    read_only_inputs_sha256 = json_sha256(read_only_hashes)
    stream = torch.cuda.Stream()
    graph, observed_specs = _capture(runtime, compiled, stream)
    topology = single_graph_topology(graph)
    if int(topology["kernel_node_count"]) != len(observed_specs):
        raise RuntimeError(
            f"{spec.case_id}: production graph contains unbound kernel nodes: "
            f"topology={topology}, observed_specs={observed_specs}"
        )
    if (
        reviewed.get("_discovery") is not True
        and topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")

    def replay_checked() -> tuple[dict[str, object], dict[str, torch.Tensor]]:
        with torch.cuda.stream(stream):
            runtime.poison()
        stream.synchronize()
        before = allocator_counters()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        after = allocator_counters()
        if before != after:
            raise AssertionError(f"{spec.case_id}: correctness replay allocated")
        correctness = runtime.validate()
        snapshot = {name: tensor.clone() for name, tensor in runtime.snapshot().items()}
        if not snapshot:
            raise AssertionError(f"{spec.case_id}: correctness snapshot is empty")
        return correctness, snapshot

    runtime.install_scenario(0)
    torch.cuda.synchronize()
    scenario_0_inputs = {
        name: tensor_sha256(tensor) for name, tensor in runtime.live_inputs.items()
    }
    correctness_0, snapshot_0 = replay_checked()
    correctness_0_repeat, snapshot_0_repeat = replay_checked()
    if _snapshot_hash(snapshot_0_repeat) != _snapshot_hash(snapshot_0):
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
    correctness_0_post, snapshot_0_post = replay_checked()
    if _snapshot_hash(snapshot_0_post) != _snapshot_hash(snapshot_0):
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    if {
        name: tensor_sha256(tensor) for name, tensor in runtime.live_inputs.items()
    } != scenario_0_inputs:
        raise AssertionError(f"{spec.case_id}: timed live inputs changed")

    runtime.install_scenario(1)
    torch.cuda.synchronize()
    scenario_1_inputs = {
        name: tensor_sha256(tensor) for name, tensor in runtime.live_inputs.items()
    }
    if any(
        scenario_1_inputs[name] == scenario_0_inputs[name]
        for name in runtime.live_inputs
    ):
        raise AssertionError(f"{spec.case_id}: live mutation missed an input")
    correctness_1, snapshot_1 = replay_checked()
    if _snapshot_hash(snapshot_1) == _snapshot_hash(snapshot_0_post):
        raise AssertionError(f"{spec.case_id}: live input did not change output")
    if {
        name: tensor_sha256(tensor) for name, tensor in runtime.read_only.items()
    } != read_only_hashes:
        raise AssertionError(f"{spec.case_id}: read-only input changed")
    if {
        name: tensor.data_ptr() for name, tensor in runtime.stable_tensors.items()
    } != fixed_pointers:
        raise AssertionError(f"{spec.case_id}: captured addresses changed")
    if single_graph_topology(graph) != topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    artifacts = []
    for spec_hash in spec.all_specs:
        evidence = exact_artifact_evidence(
            provenance[spec_hash],
            verification_before=artifact_before[spec_hash],
            verification_after=verify_artifact(provenance[spec_hash]),
        )
        artifacts.append(
            bind_exact_artifact(role=_role(spec, spec_hash), evidence=evidence)
        )
    launch_plan = build_exact_launch_plan(
        case_id=spec.case_id,
        reviewed=reviewed,
        arm=arm,
        artifacts=artifacts,
        observed_roles=tuple(_role(spec, spec_hash) for spec_hash in observed_specs),
    )
    scenario_0_hash = _snapshot_hash(snapshot_0)
    nonzero_count = _nonzero_count(correctness_0)
    if nonzero_count <= 0:
        # Some top-k validators express correctness through exact indices rather
        # than a numeric nonzero field; the checked snapshot is still required
        # to contain a nonzero byte for the release-level nonzero gate.
        nonzero_count = sum(
            int(torch.count_nonzero(tensor).item()) for tensor in snapshot_0.values()
        )
    if nonzero_count <= 0:
        raise AssertionError(f"{spec.case_id}: checked output is all zero")
    workspace_capacity_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in runtime.stable_tensors.values()
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
            "oracle": str((runtime.contract or {}).get("oracle", "torch-reference")),
            "passed": True,
            "finite": True,
            "nonzero_count": nonzero_count,
            "gates": {gate: True for gate in CORRECTNESS_GATES},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": read_only_inputs_sha256,
            "output_sha256": scenario_0_hash,
            "scenario_0": correctness_0,
            "scenario_0_repeat": correctness_0_repeat,
            "scenario_0_post": correctness_0_post,
            "scenario_1": correctness_1,
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
            "workspace_capacity_bytes": workspace_capacity_bytes,
            "stable_addresses": True,
            "allocator_stable": True,
            "zero_replay_allocations": True,
            **allocation,
            "condition_counters": allocation_records,
        },
        "conditions": conditions,
    }


@torch.inference_mode()
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
    loaded: dict[
        str,
        tuple[
            dict[str, object],
            dict[str, dict[str, Any]],
            dict[str, dict[str, object]],
        ],
    ] = {}
    for spec in CASES:
        compiled: dict[str, object] = {}
        provenance: dict[str, dict[str, Any]] = {}
        artifact_before: dict[str, dict[str, object]] = {}
        for spec_hash in spec.all_specs:
            exact, record = load_exact(args.cache.resolve(), spec_hash)
            if record["package_fingerprint"] != session.runtime_fingerprint:
                raise RuntimeError(
                    "exact object and frozen runtime fingerprints differ"
                )
            compiled[spec_hash] = exact
            provenance[spec_hash] = record
            artifact_before[spec_hash] = verify_artifact(record)
        loaded[spec.name] = (compiled, provenance, artifact_before)
    cases = []
    for spec in CASES:
        compiled, provenance, artifact_before = loaded[spec.name]
        cases.append(
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
        )
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
