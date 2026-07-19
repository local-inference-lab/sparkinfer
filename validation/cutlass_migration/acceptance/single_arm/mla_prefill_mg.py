#!/usr/bin/env python3
"""Run one frozen CUTLASS arm of the complete SM120 MLA MG-prefill corpus.

The case set is exactly the paired producer's eleven compile specializations
crossed with rows 1/2/8/32/128/512/2048.  Every case uses the production
prefill dispatch, one exact cached object, caller-owned output/LSE/workspace,
independent GPU oracles, live-input mutation, and hardened warm/cold CUDA-graph
timing in a process that loads only the requested source/toolchain arm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

import validation.cutlass_migration.diagnostics.paired.mla_prefill_mg as paired
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
import b12x.cute.compiler as cute_compiler
from b12x.attention.mla.traits import ScaleFormat


FAMILY = "mla_prefill_mg"
ARTIFACT_ROLE = "mla-prefill-mg"
INPUT_SCHEMA = "b12x.attention.mla.prefill_mg.end_to_end_input.v1"
CORRECTNESS_GATES = (
    "allocator-stability",
    "boundary-heads",
    "finite-nonzero",
    "gpu-oracle",
    "live-input-mutation",
    "lse-oracle",
    "poison-overwrite",
    "read-only-input-immutability",
    "stable-addresses",
)


@dataclass(frozen=True)
class CaseSpec:
    paired_case: paired.PrefillSpec
    rows: int

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.paired_case.name}/rows-{self.rows}"

    @property
    def input_contract(self) -> dict[str, object]:
        case = self.paired_case
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "paired_case": case.name,
            "shape": {
                "rows": self.rows,
                "family": case.family,
                "heads": case.heads,
                "heads_per_cta": case.heads_per_cta or case.heads,
                "valid_hpb": case.valid_hpb,
                "pack_hilo_rows": case.pack_hilo_rows,
                "topk": case.topk,
                "extra_topk": case.extra_topk,
                "mg_n_hg": case.n_hg,
                "compute_mode": case.compute_mode,
                "scale_format": case.scale_format,
                "has_sink": case.has_sink,
            },
            "scenario_contract": {
                "count": 2,
                "builder_module": "tests.test_attention_mla_unified_corpus",
                "inactive_topk_tail": "poisoned-minus-one",
            },
        }


CASES = tuple(
    CaseSpec(case, rows)
    for case in paired._SPECS.values()
    for rows in paired._DEFAULT_ROWS
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


def _prepare_inputs(spec: CaseSpec, device: torch.device) -> dict[str, Any]:
    case = spec.paired_case
    if case.family == "dsv4":
        inputs = paired._make_inputs(
            rows=spec.rows,
            heads=case.heads,
            main_width=case.topk,
            extra_width=case.extra_topk,
            per_token=True,
            device=device,
        )
        if inputs.main_lengths is None:
            raise AssertionError("MG-prefill requires per-token lengths")
        paired._poison_inactive_topk_tails(
            inputs.main_index_scenarios, inputs.main_length_scenarios
        )
        if case.extra_topk:
            if (
                inputs.extra_cache is None
                or inputs.extra_indices is None
                or inputs.extra_index_scenarios is None
                or inputs.extra_lengths is None
                or inputs.extra_length_scenarios is None
            ):
                raise AssertionError("dual-cache prefill inputs are incomplete")
            paired._poison_inactive_topk_tails(
                inputs.extra_index_scenarios, inputs.extra_length_scenarios
            )
        attn_sink = (
            torch.linspace(-0.8, 0.6, case.heads, device=device)
            if case.has_sink
            else None
        )
        expected = tuple(
            paired._reference(inputs, scenario, attn_sink=attn_sink)
            for scenario in range(2)
        )
        immutable: dict[str, torch.Tensor] = {
            "kv_cache": inputs.main_cache,
            "q_scenario_0": inputs.q_scenarios[0],
            "q_scenario_1": inputs.q_scenarios[1],
            "indices_scenario_0": inputs.main_index_scenarios[0],
            "indices_scenario_1": inputs.main_index_scenarios[1],
            "lengths_scenario_0": inputs.main_length_scenarios[0],
            "lengths_scenario_1": inputs.main_length_scenarios[1],
        }
        if inputs.extra_cache is not None:
            immutable.update(
                {
                    "extra_cache": inputs.extra_cache,
                    "extra_indices_scenario_0": inputs.extra_index_scenarios[0],
                    "extra_indices_scenario_1": inputs.extra_index_scenarios[1],
                    "extra_lengths_scenario_0": inputs.extra_length_scenarios[0],
                    "extra_lengths_scenario_1": inputs.extra_length_scenarios[1],
                }
            )
        if attn_sink is not None:
            immutable["attn_sink"] = attn_sink
        return {
            "expected": expected,
            "expected_lse": tuple(item[1] / math.log(2.0) for item in expected),
            "q": inputs.q,
            "kv_cache": inputs.main_cache,
            "indices": inputs.main_indices,
            "lengths": inputs.main_lengths,
            "index_scenarios": inputs.main_index_scenarios,
            "length_scenarios": inputs.main_length_scenarios,
            "extra_cache": inputs.extra_cache,
            "extra_indices": inputs.extra_indices,
            "extra_lengths": inputs.extra_lengths,
            "extra_index_scenarios": inputs.extra_index_scenarios,
            "extra_length_scenarios": inputs.extra_length_scenarios,
            "attn_sink": attn_sink,
            "sm_scale": paired._SM_SCALE,
            "scale_format": None,
            "install": lambda scenario: paired._install_scenario(inputs, scenario),
            "immutable": immutable,
        }

    if case.family == "glm":
        inputs_glm = paired._make_glm_inputs(
            rows=spec.rows,
            heads=case.heads,
            width=case.topk,
            per_token=True,
            device=device,
        )
        if inputs_glm.lengths is None:
            raise AssertionError("GLM MG-prefill requires per-token lengths")
        paired._poison_inactive_topk_tails(
            inputs_glm.index_scenarios, inputs_glm.length_scenarios
        )
        expected = tuple(
            paired._glm_reference(inputs_glm, scenario) for scenario in range(2)
        )
        return {
            "expected": expected,
            "expected_lse": tuple(item[1] for item in expected),
            "q": inputs_glm.q,
            "kv_cache": inputs_glm.launch_cache,
            "indices": inputs_glm.indices,
            "lengths": inputs_glm.lengths,
            "index_scenarios": inputs_glm.index_scenarios,
            "length_scenarios": inputs_glm.length_scenarios,
            "extra_cache": None,
            "extra_indices": None,
            "extra_lengths": None,
            "extra_index_scenarios": None,
            "extra_length_scenarios": None,
            "attn_sink": None,
            "sm_scale": paired._GLM_SM_SCALE,
            "scale_format": None,
            "install": lambda scenario: paired._install_glm_scenario(
                inputs_glm, scenario
            ),
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

    if case.family != "glm-nvfp4":
        raise AssertionError(f"unexpected MG-prefill family {case.family!r}")
    inputs_nvfp4 = paired._make_nvfp4_glm_inputs(
        rows=spec.rows,
        heads=case.heads,
        width=case.topk,
        device=device,
    )
    paired._poison_inactive_topk_tails(
        inputs_nvfp4.index_scenarios, inputs_nvfp4.length_scenarios
    )
    expected = tuple(
        paired._nvfp4_glm_reference(inputs_nvfp4, scenario) for scenario in range(2)
    )
    return {
        "expected": expected,
        "expected_lse": tuple(item[1] for item in expected),
        "q": inputs_nvfp4.q,
        "kv_cache": inputs_nvfp4.launch_cache,
        "indices": inputs_nvfp4.indices,
        "lengths": inputs_nvfp4.lengths,
        "index_scenarios": inputs_nvfp4.index_scenarios,
        "length_scenarios": inputs_nvfp4.length_scenarios,
        "extra_cache": None,
        "extra_indices": None,
        "extra_lengths": None,
        "extra_index_scenarios": None,
        "extra_length_scenarios": None,
        "attn_sink": None,
        "sm_scale": paired._GLM_SM_SCALE,
        "scale_format": int(ScaleFormat.NVFP4_E4M3),
        "install": lambda scenario: paired._install_nvfp4_glm_scenario(
            inputs_nvfp4, scenario
        ),
        "immutable": {
            "kv_cache": inputs_nvfp4.launch_cache,
            "dequant_nope": inputs_nvfp4.dequant_nope,
            "rope": inputs_nvfp4.rope,
            "q_scenario_0": inputs_nvfp4.q_scenarios[0],
            "q_scenario_1": inputs_nvfp4.q_scenarios[1],
            "indices_scenario_0": inputs_nvfp4.index_scenarios[0],
            "indices_scenario_1": inputs_nvfp4.index_scenarios[1],
            "lengths_scenario_0": inputs_nvfp4.length_scenarios[0],
            "lengths_scenario_1": inputs_nvfp4.length_scenarios[1],
        },
    }


def _run_case(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    provenance: Mapping[str, Any],
    artifact_before: Mapping[str, Any],
    capture_roles: list[str],
    args: argparse.Namespace,
    expected_physical_gpu: int,
    device: torch.device,
) -> dict[str, object]:
    case = spec.paired_case
    verify_case_compile_contract(
        case_id=spec.case_id,
        reviewed=reviewed,
        arm=arm,
        role=ARTIFACT_ROLE,
        provenance=provenance,
    )
    state = _prepare_inputs(spec, device)
    expected = state["expected"]
    expected_lse = state["expected_lse"]
    if torch.allclose(expected[0][0], expected[1][0]):
        raise AssertionError(f"{spec.case_id}: oracle scenarios are not distinct")
    output = torch.empty(
        (spec.rows, case.heads, paired._GLM_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    lse_base2 = torch.empty((spec.rows, case.heads), dtype=torch.float32, device=device)
    fixed_workspace = torch.empty((1,), dtype=torch.uint8, device=device)
    stable_tensors: dict[str, torch.Tensor] = {
        "q": state["q"],
        "kv_cache": state["kv_cache"],
        "indices": state["indices"],
        "lengths": state["lengths"],
        "output": output,
        "lse": lse_base2,
        "workspace": fixed_workspace,
    }
    if state["extra_cache"] is not None:
        stable_tensors.update(
            {
                "extra_cache": state["extra_cache"],
                "extra_indices": state["extra_indices"],
                "extra_lengths": state["extra_lengths"],
            }
        )
    if state["attn_sink"] is not None:
        stable_tensors["attn_sink"] = state["attn_sink"]
    stable_pointers = _pointers(stable_tensors)
    immutable_before = _hashes(state["immutable"])

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return paired.run_unified_prefill(
            q=state["q"],
            kv_cache=state["kv_cache"],
            topk_indices=state["indices"],
            topk_length=state["lengths"],
            sm_scale=state["sm_scale"],
            page_block_size=paired._PAGE_SIZE,
            attn_sink=state["attn_sink"],
            output=output,
            lse_out=lse_base2,
            workspace=fixed_workspace,
            scale_format=state["scale_format"],
            extra_kv_cache=state["extra_cache"],
            extra_indices=state["extra_indices"],
            extra_topk_length=state["extra_lengths"],
            extra_page_block_size=(
                paired._PAGE_SIZE if state["extra_cache"] is not None else None
            ),
        )

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        state["install"](0)
        warm_output, warm_lse = launch()
    stream.synchronize()
    if (
        warm_output.data_ptr() != output.data_ptr()
        or warm_lse.data_ptr() != lse_base2.data_ptr()
    ):
        raise AssertionError(f"{spec.case_id}: launcher replaced caller output")
    capture_roles.clear()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        launch()
    stream.synchronize()
    observed_roles = tuple(capture_roles)
    if observed_roles != (ARTIFACT_ROLE,):
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

    timed_live: dict[str, torch.Tensor] = {
        "q": state["q"],
        "indices": state["indices"],
        "lengths": state["lengths"],
    }
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
            lse_base2.fill_(poison)
        stream.synchronize()
        paired._assert_live_topk_contract(
            live_indices=state["indices"],
            live_lengths=state["lengths"],
            expected_indices=state["index_scenarios"][scenario],
            expected_lengths=state["length_scenarios"][scenario],
            rows=spec.rows,
            topk=case.topk,
        )
        if state["extra_indices"] is not None:
            paired._assert_live_topk_contract(
                live_indices=state["extra_indices"],
                live_lengths=state["extra_lengths"],
                expected_indices=state["extra_index_scenarios"][scenario],
                expected_lengths=state["extra_length_scenarios"][scenario],
                rows=spec.rows,
                topk=case.extra_topk,
            )
        _assert_pointers(stable_tensors, stable_pointers)
        before = allocator_counters()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        after = allocator_counters()
        if before != after:
            raise AssertionError(f"{spec.case_id}: correctness replay allocated")
        if torch.isnan(output).any() or torch.isnan(lse_base2).any():
            raise AssertionError(f"{spec.case_id}: replay left poisoned output")
        paired._assert_output(
            output,
            expected[scenario][0],
            label=f"{spec.case_id} scenario={scenario}",
        )
        paired._assert_prefill_boundary_heads(
            output, expected[scenario][0], n_hg=case.n_hg
        )
        torch.testing.assert_close(
            lse_base2, expected_lse[scenario], atol=6.0e-2, rtol=2.0e-2
        )
        finite = bool(
            torch.isfinite(output).all().item()
            and torch.isfinite(lse_base2).all().item()
        )
        nonzero = int(torch.count_nonzero(output).item()) + int(
            torch.count_nonzero(lse_base2).item()
        )
        if not finite or nonzero <= 0:
            raise AssertionError(f"{spec.case_id}: invalid prefill output")
        return {
            "finite": finite,
            "nonzero": nonzero,
            "sha256": json_sha256(
                {"output": tensor_sha256(output), "lse": tensor_sha256(lse_base2)}
            ),
        }

    scenario_0_a = replay_checked(0, float("nan"))
    scenario_0_b = replay_checked(0, -123.0)
    if scenario_0_a["sha256"] != scenario_0_b["sha256"]:
        raise AssertionError(f"{spec.case_id}: poison changed prefill output")
    timed_live_before = _hashes(timed_live)
    conditions, allocation_records = time_single_graph_conditions(
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
    if _hashes(timed_live) != timed_live_before:
        raise AssertionError(f"{spec.case_id}: timed live input changed")
    scenario_0_post = replay_checked(0, float("nan"))
    if scenario_0_post["sha256"] != scenario_0_a["sha256"]:
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    scenario_1 = replay_checked(1, float("nan"))
    if scenario_1["sha256"] == scenario_0_a["sha256"]:
        raise AssertionError(f"{spec.case_id}: live input did not change output")
    if _hashes(timed_live) == timed_live_before:
        raise AssertionError(f"{spec.case_id}: live input mutation was ineffective")
    if _hashes(state["immutable"]) != immutable_before:
        raise AssertionError(f"{spec.case_id}: immutable input changed")
    _assert_pointers(stable_tensors, stable_pointers)
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
            "oracle": "compressed-mla-prefill-gpu-reference",
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
            "workspace_capacity_bytes": fixed_workspace.numel(),
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
    loaded: dict[str, tuple[object, Mapping[str, Any], Mapping[str, Any]]] = {}
    for case in paired._SPECS.values():
        compiled, provenance = load_exact(args.cache, case.spec_hash)
        paired._validate_spec_contract(dict(provenance), case)
        if provenance["package_fingerprint"] != session.runtime_fingerprint:
            raise RuntimeError(
                f"{case.spec_hash}: exact object/runtime fingerprints differ"
            )
        loaded[case.spec_hash] = (
            compiled,
            provenance,
            verify_artifact(provenance),
        )

    capture_roles: list[str] = []
    active_spec_hash: list[str | None] = [None]

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
        if spec_hash != active_spec_hash[0]:
            raise RuntimeError(
                "production launch requested a spec outside the active reviewed case: "
                f"requested={spec_hash}, active={active_spec_hash[0]}"
            )
        capture_roles.append(ARTIFACT_ROLE)
        return cute_compiler.run_compiled(loaded[spec_hash][0], runtime_args)

    original_launch = paired.prefill_mg.b12x_launch
    previous_gate = os.environ.get(paired._MG_GATE_ENV)
    paired.prefill_mg.b12x_launch = exact_dispatch
    os.environ[paired._MG_GATE_ENV] = "1"
    try:
        cases: list[dict[str, object]] = []
        device = torch.device("cuda", 0)
        for spec in CASES:
            active_spec_hash[0] = spec.paired_case.spec_hash
            cases.append(
                _run_case(
                    spec=spec,
                    reviewed=session.reviewed_cases[spec.case_id],
                    arm=session.arm,
                    provenance=loaded[spec.paired_case.spec_hash][1],
                    artifact_before=loaded[spec.paired_case.spec_hash][2],
                    capture_roles=capture_roles,
                    args=args,
                    expected_physical_gpu=session.expected_physical_gpu,
                    device=device,
                )
            )
        active_spec_hash[0] = None
    finally:
        paired.prefill_mg.b12x_launch = original_launch
        if previous_gate is None:
            os.environ.pop(paired._MG_GATE_ENV, None)
        else:
            os.environ[paired._MG_GATE_ENV] = previous_gate
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
