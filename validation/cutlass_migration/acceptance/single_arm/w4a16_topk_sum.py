#!/usr/bin/env python3
"""Run one CUTLASS arm of the frozen W4A16 top-k-sum E2E graph corpus.

All eleven serving shapes are explicit: seven decode sizes and four prefill
sizes.  This process loads one exact object from one source/toolchain arm and
never imports or instantiates the comparison arm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cutlass
import cutlass.cute as cute
import torch

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
import sparkinfer.cute.compiler as cute_compiler
from sparkinfer.cute.utils import current_cuda_stream, make_ptr


FAMILY = "w4a16_topk_sum"
ARTIFACT_ROLE = "topk-sum"
INPUT_SCHEMA = "sparkinfer.w4a16.topk_sum.end_to_end_input.v1"
KERNEL_ID = "moe.w4a16.topk_sum"
HIDDEN_SIZE = 2_688
TOPK = 6
DECODE_M = (1, 2, 4, 8, 23, 33, 80)
PREFILL_M = (8_192, 16_384, 24_576, 32_768)
DECODE_REPLAYS_PER_REPORTED_SAMPLE = 64
PREFILL_REPLAYS_PER_REPORTED_SAMPLE = 1
CORRECTNESS_GATES = (
    "torch-bf16-route-sum-bit-exact",
    "finite",
    "nonzero",
    "guard-canaries",
)


@dataclass(frozen=True)
class CaseSpec:
    m: int

    @property
    def serving_regime(self) -> str:
        return "decode" if self.m < 1_024 else "prefill"

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.serving_regime}-m{self.m}"

    @property
    def seed(self) -> int:
        return 91_700 + self.m

    @property
    def input_contract(self) -> dict[str, object]:
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "shape": {"m": self.m, "topk": TOPK, "hidden_size": HIDDEN_SIZE},
            "source": {
                "generator": "torch.cuda.Generator",
                "distribution": "randn-bfloat16",
                "seed": self.seed,
            },
            "oracle": {
                "accumulation": "route-ordered-float32",
                "output_dtype": "torch.bfloat16",
            },
        }


CASES = tuple(CaseSpec(m) for m in (*DECODE_M, *PREFILL_M))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument(
        "--decode-replays-per-reported-sample",
        type=int,
        default=DECODE_REPLAYS_PER_REPORTED_SAMPLE,
    )
    parser.add_argument(
        "--prefill-replays-per-reported-sample",
        type=int,
        default=PREFILL_REPLAYS_PER_REPORTED_SAMPLE,
    )
    parser.add_argument(
        "--cache-key",
        help="optional exact cache key; otherwise the compile spec must resolve uniquely",
    )
    return parser.parse_args()


def _expected_compile_spec() -> cute_compiler.KernelCompileSpec:
    return cute_compiler.KernelCompileSpec.from_key(
        KERNEL_ID,
        1,
        ("w4a16_topk_sum", "bf16", TOPK, HIDDEN_SIZE),
    )


def _guarded_output(spec: CaseSpec) -> tuple[torch.Tensor, torch.Tensor]:
    guard_elements = 256
    payload_elements = spec.m * HIDDEN_SIZE
    storage = torch.full(
        (payload_elements + 2 * guard_elements,),
        123.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = storage[guard_elements : guard_elements + payload_elements].view(
        spec.m, HIDDEN_SIZE
    )
    return storage, output


def _assert_guards(storage: torch.Tensor, payload_elements: int) -> None:
    guard_elements = (storage.numel() - payload_elements) // 2
    if guard_elements <= 0:
        raise AssertionError("W4A16 top-k-sum output has no guard region")
    if not (
        torch.all(storage[:guard_elements] == 123.0).item()
        and torch.all(storage[-guard_elements:] == 123.0).item()
    ):
        raise AssertionError("W4A16 top-k-sum output guard changed")


def _reference(fc2: torch.Tensor) -> torch.Tensor:
    expected_f32 = fc2[:, 0, :].float()
    for route in range(1, TOPK):
        expected_f32.add_(fc2[:, route, :].float())
    return expected_f32.to(torch.bfloat16)


def _validate_output(
    *,
    spec: CaseSpec,
    output: torch.Tensor,
    storage: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, object]:
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    _assert_guards(storage, spec.m * HIDDEN_SIZE)
    finite = bool(torch.isfinite(output).all().item())
    nonzero_count = int(torch.count_nonzero(output).item())
    if not finite or nonzero_count <= 0:
        raise AssertionError(
            f"{spec.case_id}: invalid output finite={finite}, nonzero={nonzero_count}"
        )
    return {
        "finite": finite,
        "nonzero_count": nonzero_count,
        "output_sha256": tensor_sha256(output),
    }


def _replay_checked(
    *,
    spec: CaseSpec,
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    output: torch.Tensor,
    storage: torch.Tensor,
    expected: torch.Tensor,
    poison: float,
) -> dict[str, object]:
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
    return _validate_output(
        spec=spec,
        output=output,
        storage=storage,
        expected=expected,
    )


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
    generator = torch.Generator(device="cuda")
    generator.manual_seed(spec.seed)
    fc2 = torch.randn(
        (spec.m, TOPK, HIDDEN_SIZE),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    storage, output = _guarded_output(spec)
    expected = _reference(fc2)
    initial_input_sha256 = tensor_sha256(fc2)
    initial_expected_sha256 = tensor_sha256(expected)
    read_only_inputs_sha256 = json_sha256(
        {"fc2": initial_input_sha256, "oracle_expected": initial_expected_sha256}
    )
    fixed_pointers = {
        "fc2": fc2.data_ptr(),
        "output": output.data_ptr(),
        "output_storage": storage.data_ptr(),
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

    def launch() -> None:
        compiled(fc2_ptr, output_ptr, spec.m, current_cuda_stream())

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        launch()
    stream.synchronize()
    initial_topology = single_graph_topology(graph)
    if initial_topology["kernel_node_count"] != 1:
        raise AssertionError(
            f"{spec.case_id}: expected exactly one graph kernel: {initial_topology}"
        )
    if (
        reviewed.get("_discovery") is not True
        and initial_topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")

    scenario_0: dict[str, object] | None = None
    for poison in (float("nan"), -321.0):
        scenario_0 = _replay_checked(
            spec=spec,
            graph=graph,
            stream=stream,
            output=output,
            storage=storage,
            expected=expected,
            poison=poison,
        )
    if scenario_0 is None:
        raise AssertionError("scenario-0 replay was not executed")

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
    scenario_0_post = _replay_checked(
        spec=spec,
        graph=graph,
        stream=stream,
        output=output,
        storage=storage,
        expected=expected,
        poison=float("nan"),
    )
    if scenario_0_post != scenario_0:
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    if tensor_sha256(fc2) != initial_input_sha256:
        raise AssertionError(f"{spec.case_id}: timed input changed")

    fc2.neg_()
    live_input_sha256 = tensor_sha256(fc2)
    if live_input_sha256 == initial_input_sha256:
        raise AssertionError(f"{spec.case_id}: live input mutation was ineffective")
    live_expected = -expected
    live_output: dict[str, object] | None = None
    for poison in (float("nan"), 321.0):
        live_output = _replay_checked(
            spec=spec,
            graph=graph,
            stream=stream,
            output=output,
            storage=storage,
            expected=live_expected,
            poison=poison,
        )
    if (
        live_output is None
        or live_output["output_sha256"] == scenario_0["output_sha256"]
    ):
        raise AssertionError(f"{spec.case_id}: live input did not change output")
    if tensor_sha256(fc2) != live_input_sha256:
        raise AssertionError(f"{spec.case_id}: graph mutated live input")
    fc2.neg_()
    if tensor_sha256(fc2) != initial_input_sha256:
        raise AssertionError(f"{spec.case_id}: live input did not restore exactly")
    if {
        "fc2": fc2.data_ptr(),
        "output": output.data_ptr(),
        "output_storage": storage.data_ptr(),
    } != fixed_pointers:
        raise AssertionError(f"{spec.case_id}: tensor addresses changed")
    if single_graph_topology(graph) != initial_topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    # Recompute the exact initial read-only binding after the intentional live
    # mutation has been restored; this catches oracle or input corruption.
    if (
        json_sha256(
            {"fc2": tensor_sha256(fc2), "oracle_expected": tensor_sha256(expected)}
        )
        != read_only_inputs_sha256
    ):
        raise AssertionError(f"{spec.case_id}: read-only binding changed")
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
        observed_roles=(ARTIFACT_ROLE,),
    )
    allocation = allocation_records["warm_l2"]
    del stream
    return {
        "case_id": spec.case_id,
        "case_contract_sha256": reviewed["case_contract_sha256"],
        "input_sha256": json_sha256(spec.input_contract),
        "artifacts": artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": [],
        "correctness": {
            "independent_oracle": True,
            "oracle": "torch-route-ordered-float32-to-bfloat16",
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
            **initial_topology,
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
    expected_spec = _expected_compile_spec()
    compiled, provenance = load_exact(
        args.cache,
        expected_spec.hash_key,
        cache_key=args.cache_key,
    )
    if provenance["kernel_id"] != KERNEL_ID:
        raise RuntimeError(f"unexpected exact-object kernel: {provenance['kernel_id']}")
    if provenance["package_fingerprint"] != session.runtime_fingerprint:
        raise RuntimeError("exact object and frozen runtime fingerprints differ")
    artifact_before = verify_artifact(provenance)

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
            replays_per_reported_sample=(
                args.decode_replays_per_reported_sample
                if spec.serving_regime == "decode"
                else args.prefill_replays_per_reported_sample
            ),
        )
        for spec in CASES
    ]
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
