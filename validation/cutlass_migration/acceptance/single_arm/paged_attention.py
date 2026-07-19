#!/usr/bin/env python3
"""Run one CUTLASS arm of the frozen paged-attention production graph corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from validation.cutlass_migration.diagnostics.paired.paged_attention import (
    _CASES,
    _HEAD_DIM,
    _KV_HEADS,
    _MERGE_SPEC,
    _PAGE_SIZE,
    _Q_HEADS,
    _assert_correct,
    _make_fixed_graph_binding,
    _prepare_live_metadata_scenario,
    clear_attention_caches,
    make_paged_inputs,
    paged_api,
    paged_attention_forward,
    paged_attention_reference,
    quantize_paged_kv_cache_e4m3,
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


FAMILY = "paged_attention"
INPUT_SCHEMA = "b12x.paged_attention.end_to_end_input.v1"
CORRECTNESS_GATES = (
    "finite-output-and-lse",
    "guard-canaries",
    "live-page-table-and-length-replay",
    "nonzero-output",
    "torch-paged-attention-reference",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    q_len: int
    cache_len: int
    mode: str
    fp8_kv: bool
    disable_split_kv: bool
    spec_hashes: tuple[str, ...]

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.name}"

    @property
    def seed(self) -> int:
        return 20_260_719 + self.q_len + 17 * self.cache_len

    @property
    def input_contract(self) -> dict[str, object]:
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "shape": {
                "q_len": self.q_len,
                "cache_len": self.cache_len,
                "q_heads": _Q_HEADS,
                "kv_heads": _KV_HEADS,
                "head_dim": _HEAD_DIM,
                "page_size": _PAGE_SIZE,
            },
            "mode": self.mode,
            "fp8_kv": self.fp8_kv,
            "disable_split_kv": self.disable_split_kv,
            "compile_spec_hashes": list(self.spec_hashes),
            "seed": self.seed,
        }


CASES = tuple(
    CaseSpec(
        case.name,
        case.q_len,
        case.cache_len,
        case.mode,
        case.fp8_kv,
        case.disable_split_kv,
        tuple(case.spec_hashes),
    )
    for case in _CASES.values()
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    return parser.parse_args()


def _role(spec: CaseSpec, spec_hash: str) -> str:
    return f"kernel-{spec.spec_hashes.index(spec_hash)}-{spec_hash[:12]}"


def _guarded_output(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    guard = 256
    storage = torch.full(
        (q.numel() + 2 * guard,),
        123.0,
        dtype=q.dtype,
        device=q.device,
    )
    return storage, storage[guard : guard + q.numel()].view_as(q)


def _assert_output_guards(storage: torch.Tensor, output: torch.Tensor) -> None:
    guard = (storage.numel() - output.numel()) // 2
    if guard <= 0 or not (
        torch.all(storage[:guard] == 123.0).item()
        and torch.all(storage[-guard:] == 123.0).item()
    ):
        raise AssertionError("paged-attention output guard changed")


def _capture(
    *,
    compiled: Mapping[str, object],
    expected_specs: tuple[str, ...],
    launch: Any,
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, tuple[str, ...]]:
    observed: list[str] = []
    with pin_module_launches(paged_api, compiled, observed):
        with torch.cuda.stream(stream):
            launch()
        stream.synchronize()
        observed.clear()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            launch()
        stream.synchronize()
    if not observed or set(observed) != set(expected_specs):
        raise RuntimeError(
            f"production route observed {observed}, expected {expected_specs}"
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
    for spec_hash in spec.spec_hashes:
        verify_case_compile_contract(
            case_id=spec.case_id,
            reviewed=reviewed,
            arm=arm,
            role=_role(spec, spec_hash),
            provenance=provenance[spec_hash],
        )
    clear_attention_caches()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[spec.q_len],
        cache_seqlens=[spec.cache_len],
        page_size=_PAGE_SIZE,
        q_heads=_Q_HEADS,
        kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=spec.seed,
    )
    k_descale = None
    v_descale = None
    if spec.fp8_kv:
        k_cache, v_cache, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
            k_cache, v_cache, page_table, cache_seqlens
        )
        k_descale = k_descale.reshape(-1)
        v_descale = v_descale.reshape(-1)
    live_scenario = _prepare_live_metadata_scenario(
        case=_CASES[spec.name],
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    expected, expected_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    live_q_len = int(live_scenario["live_q_len"])
    live_expected, live_expected_lse = paged_attention_reference(
        q[:live_q_len],
        k_cache,
        v_cache,
        live_scenario["page_table"],
        live_scenario["cache_seqlens"],
        live_scenario["cu_seqlens_q"],
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    output_storage, output = _guarded_output(q)
    binding, scratch, _ = _make_fixed_graph_binding(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        output=output,
        mode=spec.mode,
        disable_split_kv=spec.disable_split_kv,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    if bool(binding.scratch.plan.split_kv) != (_MERGE_SPEC in spec.spec_hashes):
        raise RuntimeError(f"{spec.case_id}: planner split-K contract drifted")
    raw_live_inputs = {
        "page_table": binding.scratch.page_table,
        "cache_seqlens": binding.scratch.cache_seqlens,
        "cu_seqlens_q": binding.scratch.cu_seqlens_q,
    }
    if any(tensor is None for tensor in raw_live_inputs.values()):
        raise RuntimeError(f"{spec.case_id}: graph metadata was not materialized")
    live_inputs = {
        name: tensor for name, tensor in raw_live_inputs.items() if tensor is not None
    }
    scenario_0_values = {name: tensor.clone() for name, tensor in live_inputs.items()}
    scenario_0_hashes = {
        name: tensor_sha256(tensor) for name, tensor in live_inputs.items()
    }
    immutable_inputs = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "source_page_table": page_table,
        "source_cache_seqlens": cache_seqlens,
        "source_cu_seqlens_q": cu_seqlens_q,
        **({"k_descale": k_descale} if k_descale is not None else {}),
        **({"v_descale": v_descale} if v_descale is not None else {}),
    }
    immutable_hashes = {
        name: tensor_sha256(tensor) for name, tensor in immutable_inputs.items()
    }
    read_only_inputs_sha256 = json_sha256(immutable_hashes)
    stable_tensors = {
        **immutable_inputs,
        **{f"live_{name}": tensor for name, tensor in live_inputs.items()},
        "output_storage": output_storage,
        "output": output,
        "lse": binding.scratch.lse,
        **{f"scratch_{index}": tensor for index, tensor in enumerate(scratch)},
    }
    if any(tensor is None for tensor in stable_tensors.values()):
        raise RuntimeError(f"{spec.case_id}: stable tensor was not materialized")
    fixed_pointers = {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    }

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return paged_attention_forward(binding=binding)

    stream = torch.cuda.Stream()
    graph, observed_specs = _capture(
        compiled=compiled,
        expected_specs=spec.spec_hashes,
        launch=launch,
        stream=stream,
    )
    topology = single_graph_topology(graph)
    if int(topology["kernel_node_count"]) != len(observed_specs):
        raise RuntimeError(
            f"{spec.case_id}: production graph contains unbound kernel nodes"
        )
    if (
        reviewed.get("_discovery") is not True
        and topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")

    def replay_checked(
        *,
        expected_output: torch.Tensor,
        expected_lse_value: torch.Tensor,
        q_len: int,
    ) -> tuple[dict[str, object], str]:
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            binding.scratch.lse.fill_(float("nan"))
        stream.synchronize()
        before = allocator_counters()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        after = allocator_counters()
        if before != after:
            raise AssertionError(f"{spec.case_id}: correctness replay allocated")
        _assert_output_guards(output_storage, output)
        lse = binding.scratch.current_lse_view()[:q_len]
        metrics = _assert_correct(
            label=spec.case_id,
            output=output[:q_len],
            lse_base2=lse,
            expected=expected_output,
            expected_lse=expected_lse_value,
            fp8_kv=spec.fp8_kv,
        )
        return metrics, json_sha256(
            {"output": tensor_sha256(output[:q_len]), "lse": tensor_sha256(lse)}
        )

    correctness_0, output_hash_0 = replay_checked(
        expected_output=expected,
        expected_lse_value=expected_lse,
        q_len=spec.q_len,
    )
    correctness_0_repeat, output_hash_0_repeat = replay_checked(
        expected_output=expected,
        expected_lse_value=expected_lse,
        q_len=spec.q_len,
    )
    if output_hash_0_repeat != output_hash_0:
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
    correctness_0_post, output_hash_0_post = replay_checked(
        expected_output=expected,
        expected_lse_value=expected_lse,
        q_len=spec.q_len,
    )
    if output_hash_0_post != output_hash_0:
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    if {
        name: tensor_sha256(tensor) for name, tensor in live_inputs.items()
    } != scenario_0_hashes:
        raise AssertionError(f"{spec.case_id}: timed live metadata changed")

    scenario_1_values = {
        name: live_scenario[name]
        for name in ("page_table", "cache_seqlens", "cu_seqlens_q")
    }
    with torch.cuda.stream(stream):
        for name, tensor in live_inputs.items():
            tensor.copy_(scenario_1_values[name])
    stream.synchronize()
    scenario_1_hashes = {
        name: tensor_sha256(tensor) for name, tensor in live_inputs.items()
    }
    if (
        scenario_1_hashes["page_table"] == scenario_0_hashes["page_table"]
        or scenario_1_hashes["cache_seqlens"] == scenario_0_hashes["cache_seqlens"]
    ):
        raise AssertionError(f"{spec.case_id}: live metadata mutation was incomplete")
    correctness_1, output_hash_1 = replay_checked(
        expected_output=live_expected,
        expected_lse_value=live_expected_lse,
        q_len=live_q_len,
    )
    if output_hash_1 == output_hash_0:
        raise AssertionError(f"{spec.case_id}: live metadata did not change output")
    with torch.cuda.stream(stream):
        for name, tensor in live_inputs.items():
            tensor.copy_(scenario_0_values[name])
    stream.synchronize()
    if {
        name: tensor_sha256(tensor) for name, tensor in live_inputs.items()
    } != scenario_0_hashes:
        raise AssertionError(f"{spec.case_id}: live metadata did not restore")
    if {
        name: tensor_sha256(tensor) for name, tensor in immutable_inputs.items()
    } != immutable_hashes:
        raise AssertionError(f"{spec.case_id}: immutable input changed")
    if {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    } != fixed_pointers:
        raise AssertionError(f"{spec.case_id}: captured addresses changed")
    if single_graph_topology(graph) != topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    artifacts = []
    for spec_hash in spec.spec_hashes:
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
    nonzero_count = int(torch.count_nonzero(output[: spec.q_len]).item())
    if nonzero_count <= 0:
        raise AssertionError(f"{spec.case_id}: output is all zero")
    allocation = allocation_records["warm_l2"]
    workspace_capacity_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in scratch
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
            "oracle": "torch-paged-attention-reference",
            "passed": True,
            "finite": True,
            "nonzero_count": nonzero_count,
            "gates": {gate: True for gate in CORRECTNESS_GATES},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": read_only_inputs_sha256,
            "output_sha256": output_hash_0,
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
    cases = []
    for spec in CASES:
        compiled: dict[str, object] = {}
        provenance: dict[str, dict[str, Any]] = {}
        artifact_before: dict[str, dict[str, object]] = {}
        for spec_hash in spec.spec_hashes:
            exact, record = load_exact(args.cache.resolve(), spec_hash)
            if record["package_fingerprint"] != session.runtime_fingerprint:
                raise RuntimeError(
                    "exact object and frozen runtime fingerprints differ"
                )
            compiled[spec_hash] = exact
            provenance[spec_hash] = record
            artifact_before[spec_hash] = verify_artifact(record)
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
