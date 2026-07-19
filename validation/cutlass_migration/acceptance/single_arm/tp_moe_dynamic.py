#!/usr/bin/env python3
"""Run one CUTLASS arm of the frozen TP-MoE dynamic E2E corpus.

The two paired-producer prefill cases are retained verbatim: NVFP4 M128 and
materialized W4A8-MX M4096.  This process loads exactly one cached object per
case, routes the real planned/bound production entrypoint to that object, and
never loads or instantiates the comparison arm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from validation.cutlass_migration.diagnostics.paired.tp_moe_dynamic import (
    _CASES as PAIRED_CASES,
    _assert_live_contract,
    _assert_scenarios_distinct,
    _build_case_state,
    _correctness_metrics,
    _install_scenario,
    _legacy_compile_args,
    _runtime_compile_args,
    _scenario_input_sha256,
    _specialization_environment,
    _stable_tensors,
    _tensor_leaves,
    _validate_frozen_case_contract,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    exact_artifact_evidence,
    gpu_mode_snapshot,
    graph_topology,
    json_sha256,
    load_exact,
    sha256_file,
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
import b12x.integration.tp_moe as tp_moe


FAMILY = "tp_moe_dynamic"
ARTIFACT_ROLE = "dynamic"
INPUT_SCHEMA = "b12x.tp_moe.dynamic.end_to_end_input.v1"
CORRECTNESS_GATES = (
    "finite",
    "gpu-reference",
    "live-input-response",
    "nonzero",
    "poison-overwrite",
    "quantization-semantics",
)
NVFP4_REPLAYS_PER_REPORTED_SAMPLE = 8
W4A8_REPLAYS_PER_REPORTED_SAMPLE = 1
_TP_MOE_SOURCE = Path("b12x/integration/tp_moe.py")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    spec_hash: str
    m: int
    quant_mode: str
    experts: int
    hidden: int
    intermediate: int
    topk: int
    activation: str
    input_seeds: tuple[int, ...]
    weight_seed: int
    replays_per_reported_sample: int

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
                "experts": self.experts,
                "hidden": self.hidden,
                "intermediate": self.intermediate,
                "topk": self.topk,
            },
            "specialization": {
                "activation": self.activation,
                "compile_spec_hash": self.spec_hash,
                "quant_mode": self.quant_mode,
            },
            "source": {
                "input_seeds": list(self.input_seeds),
                "weight_seed": self.weight_seed,
            },
            "oracle": {
                "accumulation": "independent-torch-float32-moe-reference",
                "output_dtype": "torch.bfloat16",
            },
        }


CASES = (
    CaseSpec(
        name="nvfp4-prefill-m128",
        spec_hash=("0bc508dbd1653bc3e566d299b82587caa85c1c5c1c320e70edf24831724f9d02"),
        m=128,
        quant_mode="nvfp4",
        experts=4,
        hidden=512,
        intermediate=128,
        topk=2,
        activation="silu",
        input_seeds=(202, 203),
        weight_seed=201,
        replays_per_reported_sample=NVFP4_REPLAYS_PER_REPORTED_SAMPLE,
    ),
    CaseSpec(
        name="w4a8-mx-materialized-m4096",
        spec_hash=("155c03993f48e837eff1d8c8016fb88acdd1e7e940e0634140a3fdcad9a5ef27"),
        m=4096,
        quant_mode="w4a8_mx",
        experts=16,
        hidden=4096,
        intermediate=1024,
        topk=4,
        activation="silu",
        input_seeds=(3001, 3002),
        weight_seed=3000,
        replays_per_reported_sample=W4A8_REPLAYS_PER_REPORTED_SAMPLE,
    ),
)


def _source_file_record(repo_root: Path, relative_path: Path) -> dict[str, str]:
    source_path = (repo_root / relative_path).resolve()
    try:
        normalized = source_path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"source-owned kernel path escapes repo: {source_path}"
        ) from exc
    if not source_path.is_file():
        raise RuntimeError(f"source-owned kernel path does not exist: {source_path}")
    return {"path": normalized, "sha256": sha256_file(source_path)}


def _classify_graph_kernel_nodes(
    topology: Mapping[str, Any], *, repo_root: Path
) -> tuple[tuple[tuple[int, str], ...], list[dict[str, object]]]:
    raw_nodes = topology.get("nodes")
    if not isinstance(raw_nodes, list):
        raise RuntimeError("TP-MoE graph topology has no node list")
    kernel_nodes = [
        node
        for node in raw_nodes
        if isinstance(node, Mapping) and node.get("type") == "CU_GRAPH_NODE_TYPE_KERNEL"
    ]
    exact_nodes: list[tuple[int, str]] = []
    source_owned: list[dict[str, object]] = []
    source_files = [_source_file_record(repo_root, _TP_MOE_SOURCE)]
    source_role_counts: dict[str, int] = {}
    for kernel_ordinal, node in enumerate(kernel_nodes):
        kernel_name = str(node.get("kernel_name", ""))
        if kernel_name.startswith("kernel_cutlass_kernel_"):
            exact_nodes.append((kernel_ordinal, ARTIFACT_ROLE))
            continue
        if "direct_copy_kernel_cuda" in kernel_name:
            role_prefix = "torch-copy"
        elif "FillFunctor" in kernel_name:
            role_prefix = "torch-fill"
        else:
            raise RuntimeError(
                f"TP-MoE graph has unclassified kernel ordinal={kernel_ordinal}, "
                f"name={kernel_name!r}"
            )
        occurrence = source_role_counts.get(role_prefix, 0) + 1
        source_role_counts[role_prefix] = occurrence
        grid = node.get("grid")
        block = node.get("block")
        dynamic_smem = node.get("dynamic_smem_bytes")
        if (
            not isinstance(grid, list)
            or len(grid) != 3
            or not isinstance(block, list)
            or len(block) != 3
            or not isinstance(dynamic_smem, int)
            or isinstance(dynamic_smem, bool)
            or dynamic_smem < 0
        ):
            raise RuntimeError(f"malformed TP-MoE graph metadata for {kernel_name!r}")
        source_owned.append(
            {
                "node_index": kernel_ordinal,
                "role": f"{role_prefix}-{occurrence}",
                "implementation": "torch_cuda",
                "kernel_name": kernel_name,
                "kernel_name_sha256": hashlib.sha256(
                    kernel_name.encode("utf-8")
                ).hexdigest(),
                "grid": list(grid),
                "block": list(block),
                "dynamic_smem_bytes": dynamic_smem,
                "source_files": source_files,
            }
        )
    if len(exact_nodes) != 1:
        raise RuntimeError(
            f"TP-MoE graph must contain one exact CUTLASS node: {exact_nodes!r}"
        )
    covered_indices = sorted(
        [index for index, _ in exact_nodes]
        + [int(record["node_index"]) for record in source_owned]
    )
    if covered_indices != list(range(len(kernel_nodes))):
        raise RuntimeError(
            "TP-MoE exact/source-owned nodes do not partition the graph: "
            f"observed={covered_indices!r}, kernels={len(kernel_nodes)}"
        )
    return tuple(exact_nodes), source_owned


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument(
        "--nvfp4-replays-per-reported-sample",
        type=int,
        default=NVFP4_REPLAYS_PER_REPORTED_SAMPLE,
    )
    parser.add_argument(
        "--w4a8-replays-per-reported-sample",
        type=int,
        default=W4A8_REPLAYS_PER_REPORTED_SAMPLE,
    )
    parser.add_argument(
        "--cache-key",
        action="append",
        default=[],
        metavar="CASE=KEY",
        help="optional exact cache key for one case; repeat as needed",
    )
    return parser.parse_args()


def _cache_keys(raw: list[str]) -> dict[str, str]:
    valid_names = {spec.name for spec in CASES}
    result: dict[str, str] = {}
    for value in raw:
        if "=" not in value:
            raise ValueError("--cache-key must use CASE=KEY")
        name, key = value.split("=", 1)
        if name not in valid_names or name in result:
            raise ValueError(f"invalid or duplicate cache-key case {name!r}")
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError(f"cache key is not a lowercase SHA-256: {key!r}")
        result[name] = key
    return result


def _pointer_map(state: object) -> dict[str, int]:
    return {
        name: tensor.data_ptr()
        for name, tensor in _stable_tensors(state).items()  # type: ignore[arg-type]
    }


def _hash_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(tensor) for name, tensor in sorted(tensors.items())}


def _validate_replay(
    *,
    spec: CaseSpec,
    paired_case: object,
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    state: object,
    scenario: int,
    poison: float,
) -> tuple[dict[str, object], torch.Tensor]:
    output = state.output  # type: ignore[attr-defined]
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
    metrics = _correctness_metrics(
        output,
        state.references[scenario],  # type: ignore[attr-defined]
        paired_case,
    )
    return {
        "allocator_before": before,
        "allocator_after": after,
        "poison": "nan" if math.isnan(poison) else poison,
        "poisoned_elements_after": residual_poison,
        "metrics": metrics,
    }, output.clone()


def _run_case(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    repo_root: Path,
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
    paired_case = PAIRED_CASES[spec.name]
    if (
        paired_case.spec_hash != spec.spec_hash
        or paired_case.m != spec.m
        or paired_case.corpus_nodeid
        not in {
            "tests/test_cute_migration_moe_standard_corpus.py::"
            "test_standard_moe_dynamic_prefill_live_graph_oracle",
            "tests/test_w4a8_migration_corpus.py::"
            "test_w4a8_materialized_routing_phase1_phase2_matches_oracle_under_graph",
        }
    ):
        raise RuntimeError(f"{spec.case_id}: paired producer case contract changed")
    compile_args = _legacy_compile_args(provenance)
    _validate_frozen_case_contract(paired_case, compile_args)

    with _specialization_environment(compile_args):
        state = _build_case_state(paired_case, compile_args)
        torch.cuda.synchronize()
        _assert_scenarios_distinct(state)
        _install_scenario(state, 0)
        torch.cuda.synchronize()
        _assert_live_contract(state, 0, spec.experts)

        fixed_pointers = _pointer_map(state)
        immutable_tensors = _tensor_leaves(state.immutable_roots)
        immutable_initial = _hash_tensors(immutable_tensors)
        read_only_inputs_sha256 = json_sha256(immutable_initial)
        scenario_0_input = _scenario_input_sha256(state.scenarios[0])
        scenario_1_input = _scenario_input_sha256(state.scenarios[1])
        if scenario_0_input == scenario_1_input:
            raise AssertionError(f"{spec.case_id}: live scenarios are identical")

        dispatch_records: list[dict[str, object]] = []
        original_get_dynamic_kernel = tp_moe._get_dynamic_kernel

        def exact_get_dynamic_kernel(
            E: int,
            m: int,
            k: int,
            n: int,
            num_topk: int,
            max_rows: int,
            *,
            topk_ids_dtype: torch.dtype,
            fast_math: bool,
            mac_override: int | None = None,
            activation: str = "silu",
            quant_mode: str = "nvfp4",
            w4a8_repacked: bool = False,
            direct_routing: bool = False,
            share_input_across_experts: bool = False,
            deterministic_output: bool = False,
            swiglu_limit: float | None = None,
            swiglu_alpha: float | None = None,
            swiglu_beta: float | None = None,
        ):
            observed_args = _runtime_compile_args(
                E=E,
                m=m,
                k=k,
                n=n,
                num_topk=num_topk,
                topk_ids_dtype=topk_ids_dtype,
                fast_math=fast_math,
                activation=activation,
                quant_mode=quant_mode,
                w4a8_repacked=w4a8_repacked,
                direct_routing=direct_routing,
                share_input_across_experts=share_input_across_experts,
                deterministic_output=deterministic_output,
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
            )
            if observed_args != compile_args:
                raise RuntimeError(
                    f"{spec.case_id}: production dispatch specialization changed"
                )
            mac = (
                int(mac_override)
                if mac_override is not None
                else int(tp_moe._get_impl_mac("dynamic"))
            )
            dispatch_records.append(
                {
                    "m": int(m),
                    "max_active_clusters": mac,
                    "max_rows": int(max_rows),
                    "runtime_compile_args": observed_args,
                }
            )
            return compiled, mac

        tp_moe._get_dynamic_kernel = exact_get_dynamic_kernel
        try:

            def launch() -> None:
                output = tp_moe.b12x_moe_fp4(binding=state.binding)
                if output.data_ptr() != state.output.data_ptr():
                    raise AssertionError(f"{spec.case_id}: launcher replaced output")

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                launch()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                launch()
            stream.synchronize()
            if len(dispatch_records) != 2:
                raise AssertionError(
                    f"{spec.case_id}: expected eager+capture dispatches, got "
                    f"{len(dispatch_records)}"
                )
            full_topology = graph_topology(graph)
            topology = single_graph_topology(graph)
            if (
                reviewed.get("_discovery") is not True
                and topology != reviewed["graph_topology_contract"][arm]
            ):
                raise RuntimeError(
                    f"{spec.case_id}: graph topology differs from review"
                )
            exact_nodes, source_owned_nodes = _classify_graph_kernel_nodes(
                full_topology, repo_root=repo_root
            )
            if (
                reviewed.get("_discovery") is not True
                and source_owned_nodes
                != reviewed["compile_artifact_contract"][arm][
                    "source_owned_kernel_nodes"
                ]
            ):
                raise RuntimeError(
                    f"{spec.case_id}: source-owned graph nodes differ from review"
                )

            scenario_0_replays = []
            scenario_0_output: torch.Tensor | None = None
            for poison in (math.nan, 997.0, -733.0):
                record, scenario_0_output = _validate_replay(
                    spec=spec,
                    paired_case=paired_case,
                    graph=graph,
                    stream=stream,
                    state=state,
                    scenario=0,
                    poison=poison,
                )
                scenario_0_replays.append(record)
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
                paired_case=paired_case,
                graph=graph,
                stream=stream,
                state=state,
                scenario=0,
                poison=math.nan,
            )
            if (
                _scenario_input_sha256(
                    (state.live_a, state.live_topk_ids, state.live_topk_weights)
                )
                != scenario_0_input
            ):
                raise AssertionError(f"{spec.case_id}: timed live inputs changed")

            with torch.cuda.stream(stream):
                _install_scenario(state, 1)
            stream.synchronize()
            _assert_live_contract(state, 1, spec.experts)
            scenario_1_replays = []
            scenario_1_output: torch.Tensor | None = None
            for poison in (math.nan, 997.0, -733.0):
                record, scenario_1_output = _validate_replay(
                    spec=spec,
                    paired_case=paired_case,
                    graph=graph,
                    stream=stream,
                    state=state,
                    scenario=1,
                    poison=poison,
                )
                scenario_1_replays.append(record)
            if scenario_1_output is None:
                raise AssertionError(f"{spec.case_id}: no live-input replay ran")
            if torch.equal(scenario_0_post_output, scenario_1_output):
                raise AssertionError(
                    f"{spec.case_id}: live input did not change output"
                )
            new_error = float(
                (scenario_1_output.float() - state.references[1].float())
                .square()
                .mean()
                .sqrt()
                .item()
            )
            stale_error = float(
                (scenario_1_output.float() - state.references[0].float())
                .square()
                .mean()
                .sqrt()
                .item()
            )
            if not (math.isfinite(new_error) and new_error < stale_error):
                raise AssertionError(
                    f"{spec.case_id}: live output is not closer to the live oracle: "
                    f"new={new_error}, stale={stale_error}"
                )
            if _pointer_map(state) != fixed_pointers:
                raise AssertionError(f"{spec.case_id}: graph addresses changed")
            immutable_final = _hash_tensors(immutable_tensors)
            if immutable_final != immutable_initial:
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
            launch_plan = build_exact_launch_plan(
                case_id=spec.case_id,
                reviewed=reviewed,
                arm=arm,
                artifacts=artifacts,
                observed_roles=exact_nodes,
            )
            covered_indices = sorted(
                [int(binding["node_index"]) for binding in launch_plan]
                + [int(binding["node_index"]) for binding in source_owned_nodes]
            )
            if covered_indices != list(range(int(topology["kernel_node_count"]))):
                raise AssertionError(
                    f"{spec.case_id}: graph kernel coverage is incomplete"
                )
            allocation = allocation_records["warm_l2"]
            workspace_capacity = sum(
                tensor.numel() * tensor.element_size() for tensor in state.scratch
            )
            # The production specialization uses atomic output reduction.  The
            # actual output is validated on every replay, while this identity is
            # the deterministic BF16 independent-oracle value required to compare
            # separate repeat processes without misclassifying legal atomics.
            canonical_output_sha256 = tensor_sha256(
                state.references[0].to(dtype=state.output.dtype)
            )
            return {
                "case_id": spec.case_id,
                "case_contract_sha256": reviewed["case_contract_sha256"],
                "input_sha256": json_sha256(spec.input_contract),
                "artifacts": artifacts,
                "launch_plan": launch_plan,
                "source_owned_kernel_nodes": source_owned_nodes,
                "correctness": {
                    "independent_oracle": True,
                    "oracle": "torch-float32-moe-reference-canonical-bf16",
                    "passed": True,
                    "finite": True,
                    "nonzero_count": int(torch.count_nonzero(scenario_0_output).item()),
                    "gates": {gate: True for gate in CORRECTNESS_GATES},
                    "read_only_inputs_immutable": True,
                    "read_only_inputs_sha256": read_only_inputs_sha256,
                    "output_sha256": canonical_output_sha256,
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
        finally:
            tp_moe._get_dynamic_kernel = original_get_dynamic_kernel


def main() -> int:
    args = _args()
    cache_keys = _cache_keys(args.cache_key)
    replay_factors = {
        "nvfp4-prefill-m128": args.nvfp4_replays_per_reported_sample,
        "w4a8-mx-materialized-m4096": args.w4a8_replays_per_reported_sample,
    }
    if any(value < 1 for value in replay_factors.values()):
        raise ValueError("reported-sample replay factors must be positive")
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
    loaded = [
        (
            spec,
            *load_exact(
                args.cache,
                spec.spec_hash,
                cache_key=cache_keys.get(spec.name),
            ),
        )
        for spec in CASES
    ]
    artifacts_before = {
        spec.name: verify_artifact(provenance) for spec, _, provenance in loaded
    }
    cases = [
        _run_case(
            spec=spec,
            reviewed=session.reviewed_cases[spec.case_id],
            arm=args.arm,
            repo_root=session.repo_root,
            compiled=compiled,
            provenance=provenance,
            artifact_before=artifacts_before[spec.name],
            precondition=args.precondition,
            precondition_seconds=args.precondition_seconds,
            maximum_precondition_seconds=args.maximum_precondition_seconds,
            warmup=args.warmup,
            replays=args.replays,
            event_batch_replays=args.event_batch_replays,
            expected_physical_gpu=session.expected_physical_gpu,
            max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
            l2_flush_bytes=args.l2_flush_bytes,
            replays_per_reported_sample=replay_factors[spec.name],
        )
        for spec, compiled, provenance in loaded
    ]
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
