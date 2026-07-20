#!/usr/bin/env python3
"""Run one CUTLASS arm of the paired W4A8 dynamic E2E case.

The adapter retains the paired producer's M129/tile-M128 NVFP4-ReLU2 tail
case, independent GPU oracle, exact cached direct object, fixed allocation
set, poisoned output checks, and in-place live-input graph replay.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from validation.cutlass_migration.diagnostics.paired.w4a8_dynamic import (
    _EXPERTS,
    _HIDDEN,
    _INTERMEDIATE,
    _M128_SPEC_HASH,
    _TOPK,
    _all_tensors,
    _correctness,
    _mutate_live_inputs,
    _tensor_hash_record,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    exact_artifact_evidence,
    gpu_mode_snapshot,
    json_sha256,
    load_exact,
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
from tests.test_w4a8_dynamic_kernel import _run_w4a8_dynamic


FAMILY = "w4a8_dynamic"
ARTIFACT_ROLE = "direct-dynamic"
INPUT_SCHEMA = "sparkinfer.w4a8.dynamic.end_to_end_input.v1"
REPLAYS_PER_REPORTED_SAMPLE = 8
CORRECTNESS_GATES = (
    "finite",
    "gpu-reference",
    "live-input-response",
    "nonzero",
    "poison-overwrite",
    "quantization-semantics",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str = "direct-prefill-m129-tile128-nvfp4-relu2"
    spec_hash: str = _M128_SPEC_HASH
    m: int = 129
    tile_m: int = 128
    recipe: str = "w4a8_nvfp4"
    activation: str = "relu2"
    seed: int = 20_260_718

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.name}"

    @property
    def input_contract(self) -> dict[str, object]:
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "shape": {
                "m": self.m,
                "experts": _EXPERTS,
                "hidden": _HIDDEN,
                "intermediate": _INTERMEDIATE,
                "topk": _TOPK,
            },
            "specialization": {
                "activation": self.activation,
                "compile_spec_hash": self.spec_hash,
                "recipe": self.recipe,
                "tile_m": self.tile_m,
            },
            "source": {
                "generator": "torch.cuda.Generator",
                "seed": self.seed,
            },
            "oracle": {
                "accumulation": "independent-torch-w4a8-reference",
                "cosine_minimum": 0.999,
                "relative_l2_maximum": 0.03,
            },
        }


CASES = (CaseSpec(),)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument(
        "--replays-per-reported-sample",
        type=int,
        default=REPLAYS_PER_REPORTED_SAMPLE,
    )
    parser.add_argument(
        "--cache-key",
        help="optional exact cache key; otherwise the spec must resolve uniquely",
    )
    return parser.parse_args()


def _tensor_map(value: object, name: str) -> dict[str, torch.Tensor]:
    if (
        not isinstance(value, dict)
        or not value
        or not all(
            isinstance(key, str) and isinstance(tensor, torch.Tensor)
            for key, tensor in value.items()
        )
    ):
        raise AssertionError(f"W4A8 helper returned malformed {name}")
    return value


def _validate_replay(
    *,
    spec: CaseSpec,
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    output: torch.Tensor,
    reference: torch.Tensor,
    poison: float,
) -> tuple[dict[str, object], torch.Tensor]:
    with torch.cuda.stream(stream):
        output.fill_(poison)
    stream.synchronize()
    before = allocator_counters()
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    after = allocator_counters()
    if before != after:
        raise AssertionError(
            f"{spec.case_id}: correctness replay allocated: {before}->{after}"
        )
    if math.isnan(poison):
        residual_poison = int(torch.isnan(output).sum().item())
    else:
        residual_poison = int((output == poison).sum().item())
    if residual_poison:
        raise AssertionError(
            f"{spec.case_id}: graph left {residual_poison} poisoned outputs"
        )
    metrics = _correctness(output, reference)
    return {
        "allocator_before": before,
        "allocator_after": after,
        "metrics": metrics,
        "poison": "nan" if math.isnan(poison) else poison,
        "poisoned_elements_after": residual_poison,
    }, output.clone()


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
    compile_spec = json.loads(str(provenance["compile_spec_json"]))
    compile_facts = dict(compile_spec["facts"])
    expected_facts = {
        "recipe": spec.recipe,
        "activation": spec.activation,
        "tile_m": spec.tile_m,
        "experts": _EXPERTS,
        "hidden": _HIDDEN,
        "intermediate": _INTERMEDIATE,
        "top_k": _TOPK,
    }
    if compile_facts != expected_facts:
        raise RuntimeError(
            f"{spec.case_id}: exact object specialization differs: "
            f"expected={expected_facts}, observed={compile_facts}"
        )

    output, reference, _, state = _run_w4a8_dynamic(
        recipe=spec.recipe,
        activation=spec.activation,
        E=_EXPERTS,
        m=spec.m,
        K=_HIDDEN,
        n=_INTERMEDIATE,
        top_k=_TOPK,
        seed=spec.seed,
        tile_m=spec.tile_m,
        return_launcher=True,
        return_state=True,
        compiled_override=compiled,
    )
    torch.cuda.synchronize()
    if not isinstance(state, dict):
        raise AssertionError("W4A8 helper returned malformed state")
    relaunch_with = state.get("relaunch_with")
    current_reference = state.get("current_reference")
    if not callable(relaunch_with) or not callable(current_reference):
        raise AssertionError("W4A8 helper omitted exact relaunch/reference callbacks")
    live_inputs = _tensor_map(state.get("live_inputs"), "live inputs")
    read_only_inputs = _tensor_map(state.get("read_only_inputs"), "read-only inputs")
    mutable_allocations = _tensor_map(
        state.get("mutable_allocations"), "mutable allocations"
    )
    all_tensors = _all_tensors(state)
    initial_pointers = {name: tensor.data_ptr() for name, tensor in all_tensors.items()}
    read_only_initial = {
        name: tensor_sha256(tensor) for name, tensor in sorted(read_only_inputs.items())
    }
    read_only_inputs_sha256 = json_sha256(read_only_initial)
    scenario_0_input = _tensor_hash_record(live_inputs)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch() -> None:
        relaunch_with(compiled)

    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        launch()
    stream.synchronize()
    topology = single_graph_topology(graph)
    if (
        reviewed.get("_discovery") is not True
        and topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")

    scenario_0_replays = []
    scenario_0_output: torch.Tensor | None = None
    for poison in (math.nan, 997.0, -733.0):
        replay, scenario_0_output = _validate_replay(
            spec=spec,
            graph=graph,
            stream=stream,
            output=output,
            reference=reference,
            poison=poison,
        )
        scenario_0_replays.append(replay)
    if scenario_0_output is None:
        raise AssertionError(f"{spec.case_id}: no correctness replay ran")

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
    scenario_0_post, scenario_0_post_output = _validate_replay(
        spec=spec,
        graph=graph,
        stream=stream,
        output=output,
        reference=reference,
        poison=math.nan,
    )
    scenario_0_after = _tensor_hash_record(live_inputs)
    if scenario_0_after != scenario_0_input:
        raise AssertionError(f"{spec.case_id}: timed live inputs changed")

    mutation_before, scenario_1_input, changed_inputs = _mutate_live_inputs(live_inputs)
    if mutation_before != scenario_0_after or not all(changed_inputs.values()):
        raise AssertionError(f"{spec.case_id}: live-input mutation contract changed")
    reference_scenario_1 = current_reference()
    torch.cuda.synchronize()
    scenario_1_replays = []
    scenario_1_output: torch.Tensor | None = None
    for poison in (math.nan, 997.0, -733.0):
        replay, scenario_1_output = _validate_replay(
            spec=spec,
            graph=graph,
            stream=stream,
            output=output,
            reference=reference_scenario_1,
            poison=poison,
        )
        scenario_1_replays.append(replay)
    if scenario_1_output is None or torch.equal(
        scenario_0_post_output, scenario_1_output
    ):
        raise AssertionError(f"{spec.case_id}: live input did not change output")
    if {
        name: tensor.data_ptr() for name, tensor in all_tensors.items()
    } != initial_pointers:
        raise AssertionError(f"{spec.case_id}: graph addresses changed")
    read_only_final = {
        name: tensor_sha256(tensor) for name, tensor in sorted(read_only_inputs.items())
    }
    if read_only_final != read_only_initial:
        raise AssertionError(f"{spec.case_id}: read-only inputs changed")
    if single_graph_topology(graph) != topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    artifact_after = verify_artifact(provenance)
    artifact = exact_artifact_evidence(
        provenance,
        verification_before=artifact_before,
        verification_after=artifact_after,
    )
    artifacts = [bind_exact_artifact(role=ARTIFACT_ROLE, evidence=artifact)]
    observed_roles = (ARTIFACT_ROLE,) * int(topology["kernel_node_count"])
    launch_plan = build_exact_launch_plan(
        case_id=spec.case_id,
        reviewed=reviewed,
        arm=arm,
        artifacts=artifacts,
        observed_roles=observed_roles,
    )
    allocation = allocation_records["warm_l2"]
    workspace_capacity = sum(
        tensor.numel() * tensor.element_size()
        for tensor in mutable_allocations.values()
    )
    return {
        "case_id": spec.case_id,
        "case_contract_sha256": reviewed["case_contract_sha256"],
        "input_sha256": json_sha256(spec.input_contract),
        "artifacts": artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": [],
        "correctness": {
            "independent_oracle": True,
            "oracle": "torch-w4a8-dynamic-reference",
            "passed": True,
            "finite": True,
            "nonzero_count": int(torch.count_nonzero(scenario_0_output).item()),
            "gates": {gate: True for gate in CORRECTNESS_GATES},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": read_only_inputs_sha256,
            "output_sha256": tensor_sha256(scenario_0_post_output),
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
            "workspace_capacity_bytes": workspace_capacity,
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
    if args.replays_per_reported_sample < 1:
        raise ValueError("--replays-per-reported-sample must be positive")
    spec = CASES[0]
    producer_path = Path(__file__).resolve()
    session = begin_single_arm_session(
        args,
        family=FAMILY,
        producer_path=producer_path,
        bindings=(
            ReviewedCaseBinding(
                case_id=spec.case_id,
                input_sha256=json_sha256(spec.input_contract),
                correctness_gates=CORRECTNESS_GATES,
            ),
        ),
    )
    compiled, provenance = load_exact(
        args.cache,
        spec.spec_hash,
        cache_key=args.cache_key,
    )
    artifact_before = verify_artifact(provenance)
    case = _run_case(
        spec=spec,
        reviewed=session.reviewed_cases[spec.case_id],
        arm=args.arm,
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
    finish_single_arm_session(session, [case])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
