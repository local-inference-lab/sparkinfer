#!/usr/bin/env python3
"""Run one CUTLASS arm of the real W4A16 serving CUDA-graph corpus.

The production graph is kept intact.  Routed decode and prefill cases retain
their Triton route-packing nodes in graph capture; only the CUTLASS fused-MoE
and top-k-sum nodes are loaded from the selected arm's exact object cache.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from validation.cutlass_migration.diagnostics.paired.w4a16_serving import (
    _ACTIVATION,
    _EXPERTS,
    _FUSED_SPEC_HASHES,
    _HIDDEN,
    _INTERMEDIATE,
    _REQUIRED_DECODE_M,
    _REQUIRED_PREFILL_M,
    _TOPK,
    _TOPK_SPEC_HASH,
    _capture,
    _case_kind,
    _correctness,
    _fused_launch_from_manifest,
    _make_inputs,
    _make_source_weights,
    _oracle,
    _pointer_snapshot,
    _resolution_frozen,
    _specialization,
    _tensor_tree_sha256,
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
import sparkinfer.cute.compiler as cute_compiler
from sparkinfer.moe.fused.w4a16.host import (
    plan_w4a16_buffers,
    select_route_block_size_m,
)
import sparkinfer.moe.fused.w4a16.kernel as w4a16_kernel
from sparkinfer.moe.fused.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_modelopt_nvfp4_weights,
)


FAMILY = "w4a16_serving"
FUSED_ARTIFACT_ROLE = "fused-moe"
TOPK_ARTIFACT_ROLE = "topk-sum"
INPUT_SCHEMA = "sparkinfer.w4a16.serving.end_to_end_input.v1"
CASES_M = (*_REQUIRED_DECODE_M, *_REQUIRED_PREFILL_M)
COMMON_CORRECTNESS_GATES = (
    "torch-w4a16-reference",
    "cosine-at-least-0.99",
    "relative-l2-at-most-0.15",
    "finite",
    "nonzero",
    "guard-canaries",
)
DIRECT_CORRECTNESS_GATES = (
    *COMMON_CORRECTNESS_GATES,
    "production-direct-routes-in-fused-kernel",
)
ROUTED_CORRECTNESS_GATES = (
    *COMMON_CORRECTNESS_GATES,
    "production-route-pack-in-graph",
)
_ROUTE_PACK_SOURCE = Path("sparkinfer/moe/fused/w4a16/route_pack.py")
_ROUTE_PACK_ROLES = {
    "_pack_topk_routes_small_prefix_kernel": "route-pack-small-prefix",
    "_pack_topk_routes_prefix_kernel": "route-pack-prefix",
    "_pack_topk_routes_post_prefix_kernel": "route-pack-post-prefix",
    "_pack_topk_routes_sort_kernel": "route-pack-sort",
}


@dataclass(frozen=True)
class CaseSpec:
    m: int

    @property
    def serving_regime(self) -> str:
        return "decode" if self.m < 1_024 else "prefill"

    @property
    def policy(self) -> str:
        return _case_kind(self.m)

    @property
    def direct_topk_routes(self) -> bool:
        return self.m <= 6

    @property
    def fused_spec_hash(self) -> str:
        return _FUSED_SPEC_HASHES[_specialization(self.m, self.direct_topk_routes)]

    @property
    def correctness_gates(self) -> tuple[str, ...]:
        return (
            DIRECT_CORRECTNESS_GATES
            if self.direct_topk_routes
            else ROUTED_CORRECTNESS_GATES
        )

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.serving_regime}-{self.policy}-m{self.m}"

    @property
    def input_contract(self) -> dict[str, object]:
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "shape": {
                "m": self.m,
                "hidden_size": _HIDDEN,
                "intermediate_size": _INTERMEDIATE,
                "experts": _EXPERTS,
                "topk": _TOPK,
            },
            "policy": {
                "specialization": self.policy,
                "direct_topk_routes": self.direct_topk_routes,
                "production_route_pack_in_graph": not self.direct_topk_routes,
                "activation": _ACTIVATION,
                "fast_math": True,
            },
            "source": {
                "weight_generator": "torch.cuda.Generator",
                "weight_seed": 20_260_718,
                "activation_generator": "torch.cuda.Generator",
                "activation_seed": 20_260_719,
                "expert_weights": "one-generated-expert-replicated-128-times",
                "activation_rows": "one-generated-row-replicated-m-times",
                "route_ids": "(token + arange(0,topk*23,23)) % experts",
                "route_weights": [0.25, 0.25, 0.125, 0.125, 0.125, 0.125],
            },
            "oracle": {
                "implementation": "tests.w4a16_reference.moe_reference_w4a16",
                "minimum_row_cosine": 0.99,
                "maximum_row_relative_l2": 0.15,
            },
        }


CASES = tuple(CaseSpec(m) for m in CASES_M)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument("--decode-replays-per-reported-sample", type=int, default=1)
    parser.add_argument("--prefill-replays-per-reported-sample", type=int, default=1)
    parser.add_argument(
        "--topk-cache-key",
        help="optional exact top-k cache key when the spec resolves ambiguously",
    )
    return parser.parse_args()


def _guarded_output(
    buffers: object, spec: CaseSpec
) -> tuple[object, torch.Tensor, torch.Tensor]:
    guard_elements = 256
    payload_elements = spec.m * _HIDDEN
    storage = torch.full(
        (payload_elements + 2 * guard_elements,),
        123.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = storage[guard_elements : guard_elements + payload_elements].view(
        spec.m, _HIDDEN
    )
    return replace(buffers, output=output), storage, output


def _assert_output_guards(storage: torch.Tensor, payload_elements: int) -> None:
    guard_elements = (storage.numel() - payload_elements) // 2
    if guard_elements <= 0 or not (
        torch.all(storage[:guard_elements] == 123.0).item()
        and torch.all(storage[-guard_elements:] == 123.0).item()
    ):
        raise AssertionError("W4A16 serving output guard canary changed")


def _source_file_record(repo_root: Path, relative_path: Path) -> dict[str, str]:
    path = (repo_root / relative_path).resolve()
    try:
        normalized = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"source-owned kernel path escapes repo: {path}") from exc
    if not path.is_file():
        raise RuntimeError(f"source-owned kernel path does not exist: {path}")
    return {"path": normalized, "sha256": sha256_file(path)}


def _classify_graph_kernel_nodes(
    topology: Mapping[str, Any], *, repo_root: Path
) -> tuple[tuple[tuple[int, str], ...], list[dict[str, object]]]:
    raw_nodes = topology.get("nodes")
    if not isinstance(raw_nodes, list):
        raise RuntimeError("W4A16 serving graph topology has no node list")
    kernel_nodes = [
        node
        for node in raw_nodes
        if isinstance(node, Mapping) and node.get("type") == "CU_GRAPH_NODE_TYPE_KERNEL"
    ]
    exact_nodes: list[tuple[int, str]] = []
    source_owned: list[dict[str, object]] = []
    route_source = [_source_file_record(repo_root, _ROUTE_PACK_SOURCE)]
    for kernel_ordinal, node in enumerate(kernel_nodes):
        kernel_name = str(node.get("kernel_name", ""))
        if "W4A16FusedMoeKernel" in kernel_name:
            exact_nodes.append((kernel_ordinal, FUSED_ARTIFACT_ROLE))
            continue
        if "W4A16TopKSumKernel" in kernel_name:
            exact_nodes.append((kernel_ordinal, TOPK_ARTIFACT_ROLE))
            continue
        role = _ROUTE_PACK_ROLES.get(kernel_name)
        if role is None:
            raise RuntimeError(
                "W4A16 serving graph has an unclassified source-owned kernel: "
                f"ordinal={kernel_ordinal}, name={kernel_name!r}"
            )
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
            raise RuntimeError(f"malformed graph metadata for {kernel_name!r}")
        source_owned.append(
            {
                "node_index": kernel_ordinal,
                "role": role,
                "implementation": "triton",
                "kernel_name": kernel_name,
                "kernel_name_sha256": hashlib.sha256(
                    kernel_name.encode("utf-8")
                ).hexdigest(),
                "grid": list(grid),
                "block": list(block),
                "dynamic_smem_bytes": dynamic_smem,
                "source_files": sorted(route_source, key=lambda record: record["path"]),
            }
        )
    if [role for _, role in exact_nodes] != [
        FUSED_ARTIFACT_ROLE,
        TOPK_ARTIFACT_ROLE,
    ]:
        raise RuntimeError(
            "W4A16 serving graph does not contain exactly one ordered fused/top-k pair: "
            f"{exact_nodes}"
        )
    observed_indices = sorted(
        [index for index, _ in exact_nodes]
        + [int(record["node_index"]) for record in source_owned]
    )
    if observed_indices != list(range(len(kernel_nodes))):
        raise RuntimeError(
            "exact and source-owned kernel bindings do not partition the graph: "
            f"observed={observed_indices}, kernels={len(kernel_nodes)}"
        )
    return tuple(exact_nodes), source_owned


def _run_case(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    repo_root: Path,
    source: tuple[torch.Tensor, ...],
    prepared: object,
    prepared_hashes: Mapping[str, str],
    fused_compiled: object,
    fused_provenance: Mapping[str, Any],
    fused_before: Mapping[str, Any],
    topk_compiled: object,
    topk_provenance: Mapping[str, Any],
    topk_before: Mapping[str, Any],
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
    for role, provenance in (
        (FUSED_ARTIFACT_ROLE, fused_provenance),
        (TOPK_ARTIFACT_ROLE, topk_provenance),
    ):
        verify_case_compile_contract(
            case_id=spec.case_id,
            reviewed=reviewed,
            arm=arm,
            role=role,
            provenance=provenance,
        )

    x, topk_ids, topk_weights, prototype = _make_inputs(spec.m)
    raw_buffers = make_w4a16_packed_buffers(
        prepared,
        m=spec.m,
        topk=_TOPK,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    buffers, output_storage, output = _guarded_output(raw_buffers, spec)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sms = int(props.multi_processor_count)
    block_size = select_route_block_size_m(spec.m, _TOPK, _EXPERTS)
    plan = plan_w4a16_buffers(
        prepared,
        m=spec.m,
        topk=_TOPK,
        route_num_experts=_EXPERTS,
        sms=sms,
    )
    max_m_blocks = spec.m * _TOPK if spec.direct_topk_routes else int(plan.route_blocks)
    fused = _fused_launch_from_manifest(
        fused_compiled,
        fused_provenance,
        size_m=spec.m,
        max_m_blocks=max_m_blocks,
    )
    if (
        fused.hidden_size != _HIDDEN
        or fused.intermediate_size != _INTERMEDIATE
        or fused.num_experts != _EXPERTS
        or fused.top_k != _TOPK
        or fused.activation != _ACTIVATION
        or fused.moe_block_size != block_size
        or fused.direct_topk_routes != spec.direct_topk_routes
        or fused.tc_decode_fused_sum
    ):
        raise RuntimeError(f"{spec.case_id}: exact fused metadata mismatch: {fused}")
    topk_sum = w4a16_kernel.W4A16TopKSumCompileResult(
        compiled=topk_compiled,
        m=0,
        topk=_TOPK,
        hidden_size=_HIDDEN,
    )

    def launch() -> torch.Tensor:
        return w4a16_kernel.run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=_ACTIVATION,
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            fused_launch=fused,
            topk_sum_launch=topk_sum,
        )

    compile_misses_before = int(cute_compiler.compile_cache_info()["compile_misses"])
    stream = torch.cuda.Stream()
    dispatch_records: list[dict[str, object]] = []
    graph = _capture(
        launch,
        fused=fused,
        topk_sum=topk_sum,
        stream=stream,
        records=dispatch_records,
        debug_path=None,
    )
    compile_misses_after = int(cute_compiler.compile_cache_info()["compile_misses"])
    if compile_misses_after != compile_misses_before:
        raise AssertionError(f"{spec.case_id}: graph capture compiled a CUTLASS kernel")
    full_topology = graph_topology(graph)
    topology = single_graph_topology(graph)
    if (
        reviewed.get("_discovery") is not True
        and topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")
    exact_nodes, source_owned_nodes = _classify_graph_kernel_nodes(
        full_topology, repo_root=repo_root
    )
    reviewed_source_owned = reviewed["compile_artifact_contract"][arm].get(
        "source_owned_kernel_nodes"
    )
    if (
        reviewed.get("_discovery") is not True
        and source_owned_nodes != reviewed_source_owned
    ):
        raise RuntimeError(
            f"{spec.case_id}: source-owned graph nodes differ from review"
        )
    if spec.direct_topk_routes and source_owned_nodes:
        raise AssertionError(f"{spec.case_id}: direct route unexpectedly packed routes")
    if not spec.direct_topk_routes and not source_owned_nodes:
        raise AssertionError(f"{spec.case_id}: production route pack left the graph")

    baseline_input_hashes = {
        "x": tensor_sha256(x),
        "topk_ids": tensor_sha256(topk_ids),
        "topk_weights": tensor_sha256(topk_weights),
    }
    pointers_before = _pointer_snapshot(x, topk_ids, topk_weights, buffers, prepared)
    pointers_before["output_storage"] = {
        "address": output_storage.data_ptr(),
        "shape": list(output_storage.shape),
        "dtype": str(output_storage.dtype),
        "capacity_bytes": output_storage.numel() * output_storage.element_size(),
    }
    baseline_expected = _oracle(prototype, source)

    def replay_checked(expected: torch.Tensor, poison: float) -> dict[str, object]:
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
        if bool(torch.isnan(output).any().item()):
            raise AssertionError(f"{spec.case_id}: graph left poisoned output values")
        metric = _correctness(output, expected)
        _assert_output_guards(output_storage, output.numel())
        return {**metric, "output_sha256": tensor_sha256(output)}

    baseline: dict[str, object] | None = None
    for poison in (float("nan"), -321.0):
        baseline = replay_checked(baseline_expected, poison)
    if baseline is None:
        raise AssertionError(f"{spec.case_id}: baseline replay was not executed")

    x.neg_()
    topk_ids.add_(1).remainder_(_EXPERTS)
    topk_weights.mul_(0.5)
    mutated_input_hashes = {
        "x": tensor_sha256(x),
        "topk_ids": tensor_sha256(topk_ids),
        "topk_weights": tensor_sha256(topk_weights),
    }
    if any(
        mutated_input_hashes[name] == baseline_input_hashes[name]
        for name in baseline_input_hashes
    ):
        raise AssertionError(f"{spec.case_id}: captured input mutation was ineffective")
    mutated_expected = _oracle(-prototype, source) * 0.5
    mutated: dict[str, object] | None = None
    for poison in (float("nan"), 321.0):
        mutated = replay_checked(mutated_expected, poison)
    if mutated is None or mutated["output_sha256"] == baseline["output_sha256"]:
        raise AssertionError(f"{spec.case_id}: live input did not change output")

    x.neg_()
    topk_ids.sub_(1).remainder_(_EXPERTS)
    topk_weights.mul_(2.0)
    restored_input_hashes = {
        "x": tensor_sha256(x),
        "topk_ids": tensor_sha256(topk_ids),
        "topk_weights": tensor_sha256(topk_weights),
    }
    if restored_input_hashes != baseline_input_hashes:
        raise AssertionError(f"{spec.case_id}: live inputs did not restore exactly")

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
    post_timing = replay_checked(baseline_expected, float("nan"))
    if post_timing != baseline:
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    post_timing_input_hashes = {
        "x": tensor_sha256(x),
        "topk_ids": tensor_sha256(topk_ids),
        "topk_weights": tensor_sha256(topk_weights),
    }
    if post_timing_input_hashes != baseline_input_hashes:
        raise AssertionError(f"{spec.case_id}: timed inputs changed")
    pointers_after = _pointer_snapshot(x, topk_ids, topk_weights, buffers, prepared)
    pointers_after["output_storage"] = {
        "address": output_storage.data_ptr(),
        "shape": list(output_storage.shape),
        "dtype": str(output_storage.dtype),
        "capacity_bytes": output_storage.numel() * output_storage.element_size(),
    }
    if pointers_after != pointers_before:
        raise AssertionError(f"{spec.case_id}: tensor addresses/capacities changed")
    if single_graph_topology(graph) != topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    fused_after = verify_artifact(fused_provenance)
    topk_after = verify_artifact(topk_provenance)
    artifacts = [
        bind_exact_artifact(
            role=FUSED_ARTIFACT_ROLE,
            evidence=exact_artifact_evidence(
                fused_provenance,
                verification_before=fused_before,
                verification_after=fused_after,
            ),
        ),
        bind_exact_artifact(
            role=TOPK_ARTIFACT_ROLE,
            evidence=exact_artifact_evidence(
                topk_provenance,
                verification_before=topk_before,
                verification_after=topk_after,
            ),
        ),
    ]
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
        raise AssertionError(f"{spec.case_id}: graph kernel coverage is incomplete")

    read_only_inputs_sha256 = json_sha256(
        {
            "prepared": dict(prepared_hashes),
            "baseline_live_inputs": baseline_input_hashes,
            "oracle_row": tensor_sha256(baseline_expected),
        }
    )
    allocation = allocation_records["warm_l2"]
    workspace_capacity_bytes = sum(
        int(record["capacity_bytes"])
        for name, record in pointers_before.items()
        if name not in {"x", "topk_ids", "topk_weights", "output", "output_storage"}
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
            "oracle": "tests.w4a16_reference.moe_reference_w4a16",
            "passed": True,
            "finite": baseline["finite"],
            "nonzero_count": baseline["nonzero"],
            "gates": {gate: True for gate in spec.correctness_gates},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": read_only_inputs_sha256,
            "output_sha256": baseline["output_sha256"],
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
                correctness_gates=spec.correctness_gates,
            )
            for spec in CASES
        ),
    )

    fused_objects: dict[str, tuple[object, Mapping[str, Any], Mapping[str, Any]]] = {}
    for specialization, spec_hash in _FUSED_SPEC_HASHES.items():
        compiled, provenance = load_exact(args.cache, spec_hash)
        if provenance["kernel_id"] != "moe.w4a16.fused_moe":
            raise RuntimeError(f"{specialization}: exact object is not fused W4A16")
        if provenance["package_fingerprint"] != session.runtime_fingerprint:
            raise RuntimeError(
                f"{specialization}: exact object/runtime fingerprints differ"
            )
        fused_objects[specialization] = (
            compiled,
            provenance,
            verify_artifact(provenance),
        )
    topk_compiled, topk_provenance = load_exact(
        args.cache, _TOPK_SPEC_HASH, cache_key=args.topk_cache_key
    )
    if topk_provenance["kernel_id"] != "moe.w4a16.topk_sum":
        raise RuntimeError("exact top-k object has the wrong kernel id")
    if topk_provenance["package_fingerprint"] != session.runtime_fingerprint:
        raise RuntimeError("exact top-k object/runtime fingerprints differ")
    topk_before = verify_artifact(topk_provenance)

    source = _make_source_weights()
    prepared = prepare_w4a16_modelopt_nvfp4_weights(
        *source,
        activation=_ACTIVATION,
        params_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    prepared_hashes_before = _tensor_tree_sha256(prepared)
    with _resolution_frozen():
        cases = []
        for spec in CASES:
            specialization = _specialization(spec.m, spec.direct_topk_routes)
            fused_compiled, fused_provenance, fused_before = fused_objects[
                specialization
            ]
            cases.append(
                _run_case(
                    spec=spec,
                    reviewed=session.reviewed_cases[spec.case_id],
                    arm=session.arm,
                    repo_root=session.repo_root,
                    source=source,
                    prepared=prepared,
                    prepared_hashes=prepared_hashes_before,
                    fused_compiled=fused_compiled,
                    fused_provenance=fused_provenance,
                    fused_before=fused_before,
                    topk_compiled=topk_compiled,
                    topk_provenance=topk_provenance,
                    topk_before=topk_before,
                    precondition=args.precondition,
                    precondition_seconds=args.precondition_seconds,
                    maximum_precondition_seconds=args.maximum_precondition_seconds,
                    warmup=args.warmup,
                    replays=args.replays,
                    event_batch_replays=args.event_batch_replays,
                    expected_physical_gpu=session.expected_physical_gpu,
                    max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
                    l2_flush_bytes=args.l2_flush_bytes,
                    replays_per_reported_sample=(
                        args.decode_replays_per_reported_sample
                        if spec.serving_regime == "decode"
                        else args.prefill_replays_per_reported_sample
                    ),
                )
            )
    prepared_hashes_after = _tensor_tree_sha256(prepared)
    if prepared_hashes_after != prepared_hashes_before:
        raise AssertionError("prepared W4A16 weights/scales changed across corpus")
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
