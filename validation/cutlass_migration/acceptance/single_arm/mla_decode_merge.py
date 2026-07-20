#!/usr/bin/env python3
"""Run one frozen CUTLASS arm of the SM120 MLA decode/merge graph corpus.

Each process loads objects from only one compiler/source arm.  The reviewed
matrix is the complete paired diagnostic matrix: every decode specialization
at rows 1/2 and both direct five-chunk merge specializations at rows
2/4/32/128.  The production launch path, independent GPU oracles, fixed
workspace, live-input mutation, poisoned outputs, and warm/cold graph timing
remain part of every case.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping

import torch

import validation.cutlass_migration.diagnostics.paired.mla_decode_merge as paired
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
import sparkinfer.attention.mla.kernel as mla_kernel
import sparkinfer.attention.mla.merge as mla_merge
import sparkinfer.cute.compiler as cute_compiler


FAMILY = "mla_decode_merge"
INPUT_SCHEMA = "sparkinfer.attention.mla.decode_merge.end_to_end_input.v1"
DECODE_ROLE = "mla-decode"
DECODE_MERGE_ROLE = "mla-decode-merge-one-chunk"
DIRECT_MERGE_ROLE = "mla-direct-merge-five-chunk"
SINK_MERGE_ROLE = "mla-sink-merge-five-chunk"
CORRECTNESS_GATES = (
    "allocator-stability",
    "finite-nonzero",
    "gpu-oracle",
    "live-input-mutation",
    "poison-overwrite",
    "read-only-input-immutability",
    "stable-addresses",
)


@dataclass(frozen=True)
class CaseSpec:
    paired_case: paired.Case
    rows: int

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.paired_case.name}/rows-{self.rows}"

    @property
    def input_contract(self) -> dict[str, object]:
        case = self.paired_case
        shape: dict[str, object] = {"rows": self.rows}
        if isinstance(case, paired.DecodeCase):
            shape.update(
                {
                    "kind": "decode",
                    "family": case.family,
                    "heads": 8,
                    "main_topk": 64 if case.family == "dsv4" else 128,
                    "extra_topk": 64 if case.has_extra else 0,
                    "per_token_lengths": case.per_token,
                    "forced_num_splits": 1,
                }
            )
        else:
            shape.update(
                {
                    "kind": "merge",
                    "heads": case.heads,
                    "chunks": case.chunks,
                    "with_sink": case.with_sink,
                }
            )
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "paired_case": case.name,
            "shape": shape,
            "scenario_contract": {
                "count": 2,
                "builder_module": "tests.test_attention_mla_unified_corpus"
                if isinstance(case, paired.DecodeCase)
                else "tests.test_attention_mla_merge",
                "inactive_topk_tail": "poisoned-minus-one",
            },
        }

    @property
    def role_specs(self) -> tuple[tuple[str, str], ...]:
        case = self.paired_case
        if isinstance(case, paired.DecodeCase):
            return (
                (DECODE_ROLE, case.decode_spec_hash),
                (DECODE_MERGE_ROLE, case.merge_spec_hash),
            )
        return (
            (
                SINK_MERGE_ROLE if case.with_sink else DIRECT_MERGE_ROLE,
                case.spec_hash,
            ),
        )


CASES = tuple(
    CaseSpec(case, rows)
    for case in paired._CASES.values()
    for rows in (
        paired._DEFAULT_DECODE_ROWS
        if isinstance(case, paired.DecodeCase)
        else paired._DEFAULT_MERGE_ROWS
    )
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    return parser.parse_args()


def _hashes(tensors: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(tensor) for name, tensor in tensors.items()}


def _pointers(tensors: Mapping[str, torch.Tensor]) -> dict[str, int]:
    return {name: tensor.data_ptr() for name, tensor in tensors.items()}


def _assert_pointers(
    tensors: Mapping[str, torch.Tensor], expected: Mapping[str, int]
) -> None:
    observed = _pointers(tensors)
    if observed != dict(expected):
        raise AssertionError(f"stable tensor pointers changed: {expected}->{observed}")


def _timing(
    *,
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    args: argparse.Namespace,
    expected_physical_gpu: int,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, int]]]:
    return time_single_graph_conditions(
        graph,
        precondition=args.precondition,
        warmup=args.warmup,
        replays=args.replays,
        stream=stream,
        l2_flush_bytes=args.l2_flush_bytes,
        replays_per_reported_sample=args.replays_per_reported_sample,
        event_batch_replays=args.event_batch_replays,
        precondition_seconds=args.precondition_seconds,
        maximum_precondition_seconds=args.maximum_precondition_seconds,
        mode_snapshot=lambda: gpu_mode_snapshot(expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
    )


def _artifact_bindings(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    loaded: Mapping[str, tuple[object, Mapping[str, Any], Mapping[str, Any]]],
    observed_roles: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    artifacts: list[dict[str, object]] = []
    for role, spec_hash in spec.role_specs:
        _, provenance, verification_before = loaded[spec_hash]
        verify_case_compile_contract(
            case_id=spec.case_id,
            reviewed=reviewed,
            arm=arm,
            role=role,
            provenance=provenance,
        )
        evidence = exact_artifact_evidence(
            provenance,
            verification_before=verification_before,
            verification_after=verify_artifact(provenance),
        )
        artifacts.append(bind_exact_artifact(role=role, evidence=evidence))
    return artifacts, build_exact_launch_plan(
        case_id=spec.case_id,
        reviewed=reviewed,
        arm=arm,
        artifacts=artifacts,
        observed_roles=observed_roles,
    )


def _decode_inputs(
    spec: CaseSpec,
    device: torch.device,
) -> dict[str, Any]:
    case = spec.paired_case
    if not isinstance(case, paired.DecodeCase):
        raise TypeError("decode setup received a merge case")
    heads = 8
    if case.family == "dsv4":
        main_width = 64
        extra_width = 64 if case.has_extra else 0
        inputs = paired._make_inputs(
            rows=spec.rows,
            heads=heads,
            main_width=main_width,
            extra_width=extra_width,
            per_token=case.per_token,
            device=device,
        )
        if inputs.main_lengths is None:
            raise AssertionError("DSV4 decode case requires per-token lengths")
        paired._poison_inactive_topk_tails(
            inputs.main_index_scenarios, inputs.main_length_scenarios
        )
        if case.has_extra:
            if (
                inputs.extra_cache is None
                or inputs.extra_indices is None
                or inputs.extra_index_scenarios is None
                or inputs.extra_lengths is None
                or inputs.extra_length_scenarios is None
            ):
                raise AssertionError("dual-cache decode inputs are incomplete")
            paired._poison_inactive_topk_tails(
                inputs.extra_index_scenarios, inputs.extra_length_scenarios
            )
        result: dict[str, Any] = {
            "expected": tuple(
                paired._reference(inputs, scenario)[0] for scenario in range(2)
            ),
            "q": inputs.q,
            "kv_cache": inputs.main_cache,
            "indices": inputs.main_indices,
            "lengths": inputs.main_lengths,
            "q_scenarios": inputs.q_scenarios,
            "index_scenarios": inputs.main_index_scenarios,
            "length_scenarios": inputs.main_length_scenarios,
            "extra_cache": inputs.extra_cache,
            "extra_indices": inputs.extra_indices,
            "extra_lengths": inputs.extra_lengths,
            "extra_index_scenarios": inputs.extra_index_scenarios,
            "extra_length_scenarios": inputs.extra_length_scenarios,
            "sm_scale": paired._SM_SCALE,
            "main_width": main_width,
            "extra_width": extra_width,
            "install": lambda scenario: paired._install_scenario(inputs, scenario),
            "immutable": {
                "kv_cache": inputs.main_cache,
                "q_scenario_0": inputs.q_scenarios[0],
                "q_scenario_1": inputs.q_scenarios[1],
                "indices_scenario_0": inputs.main_index_scenarios[0],
                "indices_scenario_1": inputs.main_index_scenarios[1],
                "lengths_scenario_0": inputs.main_length_scenarios[0],
                "lengths_scenario_1": inputs.main_length_scenarios[1],
            },
        }
        if inputs.extra_cache is not None:
            result["immutable"].update(
                {
                    "extra_cache": inputs.extra_cache,
                    "extra_indices_scenario_0": inputs.extra_index_scenarios[0],
                    "extra_indices_scenario_1": inputs.extra_index_scenarios[1],
                    "extra_lengths_scenario_0": inputs.extra_length_scenarios[0],
                    "extra_lengths_scenario_1": inputs.extra_length_scenarios[1],
                }
            )
        return result

    inputs_glm = paired._make_glm_inputs(
        rows=spec.rows,
        heads=heads,
        width=128,
        per_token=case.per_token,
        device=device,
    )
    if case.per_token:
        if inputs_glm.lengths is None:
            raise AssertionError("GLM per-token decode is missing lengths")
        paired._poison_inactive_topk_tails(
            inputs_glm.index_scenarios, inputs_glm.length_scenarios
        )
    return {
        "expected": tuple(
            paired._glm_reference(inputs_glm, scenario)[0] for scenario in range(2)
        ),
        "q": inputs_glm.q,
        "kv_cache": inputs_glm.launch_cache,
        "indices": inputs_glm.indices,
        "lengths": inputs_glm.lengths,
        "q_scenarios": inputs_glm.q_scenarios,
        "index_scenarios": inputs_glm.index_scenarios,
        "length_scenarios": inputs_glm.length_scenarios,
        "extra_cache": None,
        "extra_indices": None,
        "extra_lengths": None,
        "extra_index_scenarios": None,
        "extra_length_scenarios": None,
        "sm_scale": paired._GLM_SM_SCALE,
        "main_width": 128,
        "extra_width": 0,
        "install": lambda scenario: paired._install_glm_scenario(inputs_glm, scenario),
        "immutable": {
            "kv_cache": inputs_glm.launch_cache,
            "packed_tokens": inputs_glm.packed_tokens,
            "q_scenario_0": inputs_glm.q_scenarios[0],
            "q_scenario_1": inputs_glm.q_scenarios[1],
            "indices_scenario_0": inputs_glm.index_scenarios[0],
            "indices_scenario_1": inputs_glm.index_scenarios[1],
            "lengths_scenario_0": inputs_glm.length_scenarios[0],
            "lengths_scenario_1": inputs_glm.length_scenarios[1],
        },
    }


def _run_decode_case(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    loaded: Mapping[str, tuple[object, Mapping[str, Any], Mapping[str, Any]]],
    capture_roles: list[str],
    args: argparse.Namespace,
    expected_physical_gpu: int,
    device: torch.device,
) -> dict[str, object]:
    case = spec.paired_case
    if not isinstance(case, paired.DecodeCase):
        raise TypeError("decode runner received a merge case")
    state = _decode_inputs(spec, device)
    expected = state["expected"]
    if torch.allclose(expected[0], expected[1]):
        raise AssertionError(f"{spec.case_id}: oracle scenarios are not distinct")

    workspace = paired._make_decode_workspace(
        rows=2,
        heads=8,
        width=state["main_width"] + state["extra_width"],
        family=case.family,
        device=device,
    )
    output = torch.empty(
        (spec.rows, 8, paired._GLM_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    stable_tensors: dict[str, torch.Tensor] = {
        "q": state["q"],
        "kv_cache": state["kv_cache"],
        "indices": state["indices"],
        "output": output,
        "workspace": workspace.shared_scratch,
    }
    if state["lengths"] is not None:
        stable_tensors["lengths"] = state["lengths"]
    if state["extra_cache"] is not None:
        stable_tensors.update(
            {
                "extra_cache": state["extra_cache"],
                "extra_indices": state["extra_indices"],
                "extra_lengths": state["extra_lengths"],
            }
        )
    stable_pointers = _pointers(stable_tensors)
    immutable_before = _hashes(state["immutable"])

    def launch() -> torch.Tensor:
        return mla_kernel.run_unified_decode(
            q_all=state["q"],
            swa_k_cache=state["kv_cache"],
            swa_indices=state["indices"],
            swa_topk_lengths=state["lengths"],
            workspace=workspace,
            sm_scale=state["sm_scale"],
            swa_page_size=paired._PAGE_SIZE,
            indexed_k_cache=state["extra_cache"],
            indexed_indices=state["extra_indices"],
            indexed_topk_lengths=state["extra_lengths"],
            indexed_page_size=(
                paired._PAGE_SIZE if state["extra_cache"] is not None else None
            ),
            forced_num_splits=1,
            out=output,
        )

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        state["install"](0)
        warm_output = launch()
    stream.synchronize()
    if warm_output.data_ptr() != output.data_ptr():
        raise AssertionError(f"{spec.case_id}: launcher replaced caller output")
    capture_roles.clear()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        launch()
    stream.synchronize()
    observed_roles = tuple(capture_roles)
    if observed_roles != (DECODE_ROLE, DECODE_MERGE_ROLE):
        raise AssertionError(
            f"{spec.case_id}: unexpected exact launch roles {observed_roles}"
        )
    topology = single_graph_topology(graph)
    if topology["node_count"] != 2 or topology["kernel_node_count"] != 2:
        raise AssertionError(f"{spec.case_id}: unexpected topology {topology}")
    if (
        reviewed.get("_discovery") is not True
        and topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")

    timed_live: dict[str, torch.Tensor] = {
        "q": state["q"],
        "indices": state["indices"],
    }
    if state["lengths"] is not None:
        timed_live["lengths"] = state["lengths"]
    if state["extra_indices"] is not None:
        timed_live.update(
            {
                "extra_indices": state["extra_indices"],
                "extra_lengths": state["extra_lengths"],
            }
        )

    def replay_checked(scenario: int, poison: float) -> dict[str, object]:
        with torch.cuda.stream(stream):
            state["install"](scenario)
            output.fill_(poison)
        stream.synchronize()
        if not torch.equal(state["indices"], state["index_scenarios"][scenario]):
            raise AssertionError(f"{spec.case_id}: live indices differ from scenario")
        if state["lengths"] is not None:
            if not torch.equal(state["lengths"], state["length_scenarios"][scenario]):
                raise AssertionError(f"{spec.case_id}: live lengths differ")
            for row in range(spec.rows):
                length = int(state["lengths"][row].item())
                if not 0 < length <= state["main_width"]:
                    raise AssertionError(f"{spec.case_id}: invalid top-k length")
                if (
                    length < state["main_width"]
                    and int(state["indices"][row, length].item()) != -1
                ):
                    raise AssertionError(f"{spec.case_id}: top-k tail is not poisoned")
        if state["extra_indices"] is not None:
            if not torch.equal(
                state["extra_indices"], state["extra_index_scenarios"][scenario]
            ) or not torch.equal(
                state["extra_lengths"], state["extra_length_scenarios"][scenario]
            ):
                raise AssertionError(f"{spec.case_id}: live extra metadata differs")
        _assert_pointers(stable_tensors, stable_pointers)
        before = allocator_counters()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        after = allocator_counters()
        if before != after:
            raise AssertionError(f"{spec.case_id}: correctness replay allocated")
        paired._assert_output(
            output,
            expected[scenario],
            label=f"{spec.case_id} scenario={scenario}",
        )
        metrics = paired._correctness_metrics(output, expected[scenario])
        if not metrics["finite"] or int(metrics["nonzero"]) <= 0:
            raise AssertionError(f"{spec.case_id}: invalid decode output")
        return metrics

    scenario_0_a = replay_checked(0, float("nan"))
    scenario_0_b = replay_checked(0, -123.0)
    if scenario_0_a["sha256"] != scenario_0_b["sha256"]:
        raise AssertionError(f"{spec.case_id}: poison changed decode output")
    timed_live_before = _hashes(timed_live)
    conditions, allocation_records = _timing(
        graph=graph,
        stream=stream,
        args=args,
        expected_physical_gpu=expected_physical_gpu,
    )
    if _hashes(timed_live) != timed_live_before:
        raise AssertionError(f"{spec.case_id}: timed live input changed")
    scenario_0_post = replay_checked(0, float("nan"))
    if scenario_0_post["sha256"] != scenario_0_a["sha256"]:
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    scenario_1 = replay_checked(1, float("nan"))
    if scenario_1["sha256"] == scenario_0_a["sha256"]:
        raise AssertionError(f"{spec.case_id}: live input did not change output")
    live_scenario_1_hashes = _hashes(timed_live)
    if live_scenario_1_hashes == timed_live_before:
        raise AssertionError(f"{spec.case_id}: live input mutation was ineffective")
    if _hashes(state["immutable"]) != immutable_before:
        raise AssertionError(f"{spec.case_id}: immutable input changed")
    _assert_pointers(stable_tensors, stable_pointers)
    if single_graph_topology(graph) != topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    artifacts, launch_plan = _artifact_bindings(
        spec=spec,
        reviewed=reviewed,
        arm=arm,
        loaded=loaded,
        observed_roles=observed_roles,
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
            "oracle": "compressed-mla-fp32-gpu-reference",
            "passed": True,
            "finite": True,
            "nonzero_count": int(scenario_0_a["nonzero"]),
            "gates": {gate: True for gate in CORRECTNESS_GATES},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": json_sha256(immutable_before),
            "output_sha256": str(scenario_0_a["sha256"]),
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
            "workspace_capacity_bytes": workspace.shared_scratch.numel(),
            "stable_addresses": True,
            "allocator_stable": True,
            "zero_replay_allocations": True,
            **allocation,
            "condition_counters": allocation_records,
        },
        "conditions": conditions,
    }


def _run_merge_case(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    loaded: Mapping[str, tuple[object, Mapping[str, Any], Mapping[str, Any]]],
    capture_roles: list[str],
    args: argparse.Namespace,
    expected_physical_gpu: int,
    device: torch.device,
) -> dict[str, object]:
    case = spec.paired_case
    if not isinstance(case, paired.MergeCase):
        raise TypeError("merge runner received a decode case")
    base_problem = paired._make_fixed_merge_problem(
        rows=spec.rows,
        heads=case.heads,
        chunks=case.chunks,
        device=device,
    )
    problem = replace(base_problem, output=torch.empty_like(base_problem.output))
    scenarios = paired._make_merge_scenarios(
        rows=spec.rows,
        heads=case.heads,
        chunks=case.chunks,
        device=device,
    )
    sink_scenarios: tuple[torch.Tensor, torch.Tensor] | None = None
    live_sink: torch.Tensor | None = None
    if case.with_sink:
        sink_scenarios = (
            torch.linspace(-1.25, 0.75, case.heads, device=device),
            torch.linspace(0.9, -0.6, case.heads, device=device),
        )
        live_sink = torch.empty_like(sink_scenarios[0])
    expected = tuple(
        paired._split_merge_fp32_oracle(
            partials,
            lse,
            chunks=case.chunks,
            attn_sink=None if sink_scenarios is None else sink_scenarios[index],
        )
        for index, (partials, lse) in enumerate(scenarios)
    )
    if torch.allclose(expected[0], expected[1]):
        raise AssertionError(f"{spec.case_id}: merge scenarios are not distinct")
    binding = mla_merge.build_sparse_mla_split_decode_merge_binding(
        tmp_output=problem.tmp_output,
        tmp_lse=problem.tmp_lse,
        num_chunks_ptr=problem.num_chunks_ptr,
        output=problem.output,
        num_chunks=case.chunks,
        attn_sink=live_sink,
    )
    stable_tensors: dict[str, torch.Tensor] = {
        "tmp_output": problem.tmp_output,
        "tmp_lse": problem.tmp_lse,
        "num_chunks": problem.num_chunks_ptr,
        "output": problem.output,
    }
    if live_sink is not None:
        stable_tensors["attn_sink"] = live_sink
    stable_pointers = _pointers(stable_tensors)
    immutable: dict[str, torch.Tensor] = {
        f"partials_scenario_{index}": scenario[0]
        for index, scenario in enumerate(scenarios)
    }
    immutable.update(
        {
            f"lse_scenario_{index}": scenario[1]
            for index, scenario in enumerate(scenarios)
        }
    )
    if sink_scenarios is not None:
        immutable.update(
            {
                f"sink_scenario_{index}": sink
                for index, sink in enumerate(sink_scenarios)
            }
        )
    immutable_before = _hashes(immutable)

    def install(scenario: int) -> None:
        paired._install_merge_scenario(
            problem,
            partials=scenarios[scenario][0],
            lse=scenarios[scenario][1],
            live_sink=live_sink,
            source_sink=(None if sink_scenarios is None else sink_scenarios[scenario]),
        )

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        install(0)
        binding.run()
    stream.synchronize()
    capture_roles.clear()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        binding.run()
    stream.synchronize()
    observed_roles = tuple(capture_roles)
    expected_role = SINK_MERGE_ROLE if case.with_sink else DIRECT_MERGE_ROLE
    if observed_roles != (expected_role,):
        raise AssertionError(
            f"{spec.case_id}: unexpected exact launch roles {observed_roles}"
        )
    topology = single_graph_topology(graph)
    if topology["node_count"] != 1 or topology["kernel_node_count"] != 1:
        raise AssertionError(f"{spec.case_id}: unexpected topology {topology}")
    if (
        reviewed.get("_discovery") is not True
        and topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")

    live_tensors: dict[str, torch.Tensor] = {
        "tmp_output": problem.tmp_output,
        "tmp_lse": problem.tmp_lse,
        "num_chunks": problem.num_chunks_ptr,
    }
    if live_sink is not None:
        live_tensors["attn_sink"] = live_sink

    def replay_checked(scenario: int, poison: float) -> dict[str, object]:
        with torch.cuda.stream(stream):
            install(scenario)
            problem.output.fill_(poison)
        stream.synchronize()
        _assert_pointers(stable_tensors, stable_pointers)
        before = allocator_counters()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        after = allocator_counters()
        if before != after:
            raise AssertionError(f"{spec.case_id}: correctness replay allocated")
        if not bool(torch.isfinite(problem.output).all().item()):
            raise AssertionError(f"{spec.case_id}: merge output is non-finite")
        nonzero = int(torch.count_nonzero(problem.output).item())
        if nonzero <= 0:
            raise AssertionError(f"{spec.case_id}: merge output is zero")
        torch.testing.assert_close(
            problem.output.float(), expected[scenario], atol=1.5e-2, rtol=1.5e-2
        )
        return {
            "nonzero": nonzero,
            "sha256": tensor_sha256(problem.output),
            "max_abs": float(
                (problem.output.float() - expected[scenario]).abs().max().item()
            ),
        }

    scenario_0_a = replay_checked(0, float("nan"))
    scenario_0_b = replay_checked(0, -321.0)
    if scenario_0_a["sha256"] != scenario_0_b["sha256"]:
        raise AssertionError(f"{spec.case_id}: poison changed merge output")
    timed_live_before = _hashes(live_tensors)
    conditions, allocation_records = _timing(
        graph=graph,
        stream=stream,
        args=args,
        expected_physical_gpu=expected_physical_gpu,
    )
    if _hashes(live_tensors) != timed_live_before:
        raise AssertionError(f"{spec.case_id}: timed merge input changed")
    scenario_0_post = replay_checked(0, float("nan"))
    if scenario_0_post["sha256"] != scenario_0_a["sha256"]:
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    scenario_1 = replay_checked(1, float("nan"))
    if scenario_1["sha256"] == scenario_0_a["sha256"]:
        raise AssertionError(f"{spec.case_id}: live input did not change output")
    if _hashes(live_tensors) == timed_live_before:
        raise AssertionError(f"{spec.case_id}: merge mutation was ineffective")
    if _hashes(immutable) != immutable_before:
        raise AssertionError(f"{spec.case_id}: immutable merge input changed")
    _assert_pointers(stable_tensors, stable_pointers)
    if single_graph_topology(graph) != topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    artifacts, launch_plan = _artifact_bindings(
        spec=spec,
        reviewed=reviewed,
        arm=arm,
        loaded=loaded,
        observed_roles=observed_roles,
    )
    allocation = allocation_records["warm_l2"]
    workspace_bytes = sum(
        tensor.untyped_storage().nbytes()
        for tensor in (
            problem.tmp_output,
            problem.tmp_lse,
            problem.num_chunks_ptr,
            problem.output,
        )
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
            "oracle": "split-merge-fp32-gpu-reference",
            "passed": True,
            "finite": True,
            "nonzero_count": int(scenario_0_a["nonzero"]),
            "gates": {gate: True for gate in CORRECTNESS_GATES},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": json_sha256(immutable_before),
            "output_sha256": str(scenario_0_a["sha256"]),
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
            "workspace_capacity_bytes": workspace_bytes,
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
    if args.replays_per_reported_sample <= 0:
        raise ValueError("--replays-per-reported-sample must be positive")
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
    device = torch.device("cuda", 0)
    unique_specs = {spec_hash for spec in CASES for _, spec_hash in spec.role_specs}
    loaded: dict[str, tuple[object, Mapping[str, Any], Mapping[str, Any]]] = {}
    for spec_hash in sorted(unique_specs):
        compiled, provenance = load_exact(args.cache, spec_hash)
        if provenance["package_fingerprint"] != session.runtime_fingerprint:
            raise RuntimeError(f"{spec_hash}: exact object/runtime fingerprints differ")
        loaded[spec_hash] = (compiled, provenance, verify_artifact(provenance))
    for spec in CASES:
        case = spec.paired_case
        if isinstance(case, paired.DecodeCase):
            paired._validate_decode_manifest(
                dict(loaded[case.decode_spec_hash][1]), case
            )
            paired._validate_merge_manifest(
                dict(loaded[case.merge_spec_hash][1]),
                spec_hash=case.merge_spec_hash,
                static_num_chunks=1,
                with_sink=False,
            )
        else:
            paired._validate_merge_manifest(
                dict(loaded[case.spec_hash][1]),
                spec_hash=case.spec_hash,
                static_num_chunks=case.chunks,
                with_sink=case.with_sink,
            )

    role_by_spec = {
        spec_hash: role for spec in CASES for role, spec_hash in spec.role_specs
    }
    capture_roles: list[str] = []
    active_spec_hashes: set[str] = set()

    def exact_dispatch(
        _func,
        *,
        compile_spec,
        compile_args,
        runtime_args,
        compile_kwargs=None,
    ):
        del _func, compile_args, compile_kwargs
        spec_hash = str(compile_spec.hash_key)
        if spec_hash not in active_spec_hashes:
            raise RuntimeError(
                "production launch requested a spec outside the active reviewed case: "
                f"requested={spec_hash}, active={sorted(active_spec_hashes)}"
            )
        capture_roles.append(role_by_spec[spec_hash])
        return cute_compiler.run_compiled(loaded[spec_hash][0], runtime_args)

    original_decode_launch = mla_kernel.sparkinfer_launch
    original_merge_launch = mla_merge.sparkinfer_launch
    previous_glm_h8 = os.environ.get(paired._GLM_H8_NATIVE_ENV)
    mla_kernel.sparkinfer_launch = exact_dispatch
    mla_merge.sparkinfer_launch = exact_dispatch
    os.environ[paired._GLM_H8_NATIVE_ENV] = "1"
    try:
        cases: list[dict[str, object]] = []
        for spec in CASES:
            active_spec_hashes.clear()
            active_spec_hashes.update(spec_hash for _, spec_hash in spec.role_specs)
            common = {
                "spec": spec,
                "reviewed": session.reviewed_cases[spec.case_id],
                "arm": session.arm,
                "loaded": loaded,
                "capture_roles": capture_roles,
                "args": args,
                "expected_physical_gpu": session.expected_physical_gpu,
                "device": device,
            }
            if isinstance(spec.paired_case, paired.DecodeCase):
                cases.append(_run_decode_case(**common))
            else:
                cases.append(_run_merge_case(**common))
        active_spec_hashes.clear()
    finally:
        mla_kernel.sparkinfer_launch = original_decode_launch
        mla_merge.sparkinfer_launch = original_merge_launch
        if previous_glm_h8 is None:
            os.environ.pop(paired._GLM_H8_NATIVE_ENV, None)
        else:
            os.environ[paired._GLM_H8_NATIVE_ENV] = previous_glm_h8
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
