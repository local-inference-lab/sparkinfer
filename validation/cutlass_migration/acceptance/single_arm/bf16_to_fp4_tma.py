#!/usr/bin/env python3
"""Run one CUTLASS arm of the frozen BF16->FP4 E2E graph corpus.

The process loads and executes only the requested source/toolchain arm.  It
never loads an unused comparison object, and it emits one immutable result for
the separate-process A1/B1/B2/A2 release gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch

from validation.cutlass_migration.diagnostics.paired.bf16_to_fp4_tma import (
    _guarded,
    _load,
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
import b12x.cute.compiler as cute_compiler
from b12x.cute.fp4 import quantize_grouped_nvfp4_torch


FAMILY = "bf16_to_fp4_tma"
ARTIFACT_ROLE = "bf16-to-fp4"
INPUT_SCHEMA = "b12x.bf16_to_fp4_tma.end_to_end_input.v1"
CORRECTNESS_GATES = (
    "bit-exact-packed",
    "bit-exact-scale",
    "fp4-negative-zero-canonical",
    "fp8-scale-range",
    "guard-canaries",
    "torch-reference",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    M: int
    K: int
    global_scale: float
    seed: int

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.name}"

    @property
    def input_contract(self) -> dict[str, object]:
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "shape": {"M": self.M, "K": self.K},
            "global_scale": self.global_scale,
            "source": {
                "generator": "torch.cpu.Generator",
                "distribution": "randn-float32-divide-4-to-bfloat16",
                "seed": self.seed,
            },
        }


CASES = (
    CaseSpec("minimum-tile", 128, 128, 0.125, 92_056),
    CaseSpec("multi-k", 128, 256, 3.25, 92_184),
    CaseSpec("prefill-m128-k4096", 128, 4096, 0.5, 96_024),
    CaseSpec("prefill-m512-k4096", 512, 4096, 2500.0, 96_408),
    CaseSpec("prefill-m2048-k4096", 2048, 4096, 31.75, 97_944),
    CaseSpec("prefill-m128-k7168-subnormal", 128, 7168, 0.03125, 99_096),
    CaseSpec("prefill-m512-k7168", 512, 7168, 448.0, 99_480),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    parser.add_argument("--spec-mode", choices=("current", "v1-mk"), default="current")
    return parser.parse_args()


def _source(
    spec: CaseSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    source = (
        torch.randn((spec.M, spec.K), generator=generator, dtype=torch.float32)
        .div_(4.0)
        .to("cuda", dtype=torch.bfloat16)
        .contiguous()
    )
    global_scale = torch.tensor([spec.global_scale], dtype=torch.float32, device="cuda")
    row_counts = torch.tensor([spec.M], dtype=torch.int32, device="cuda")
    return source, global_scale, row_counts


def _reference(
    source: torch.Tensor,
    global_scale: torch.Tensor,
    row_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed, scale_view = quantize_grouped_nvfp4_torch(
        source.unsqueeze(0), row_counts, global_scale
    )
    scale = (
        scale_view.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8).reshape(-1)
    )
    return packed, scale


def _assert_guards(storage: torch.Tensor, payload_bytes: int, value: int) -> None:
    guard_bytes = (storage.numel() - payload_bytes) // 2
    if guard_bytes <= 0:
        raise AssertionError("guarded output has no canary region")
    if not (
        torch.all(storage[:guard_bytes] == value).item()
        and torch.all(storage[-guard_bytes:] == value).item()
    ):
        raise AssertionError("BF16->FP4 output guard canary changed")


def _validate_output(
    *,
    spec: CaseSpec,
    packed: torch.Tensor,
    scale: torch.Tensor,
    packed_storage: torch.Tensor,
    scale_storage: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    packed_reference, scale_reference = reference
    packed_actual = packed.view(1, spec.M, spec.K // 2).permute(1, 2, 0)
    torch.testing.assert_close(packed_actual, packed_reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(scale, scale_reference, rtol=0.0, atol=0.0)
    _assert_guards(packed_storage, packed.numel(), 0xD3)
    _assert_guards(scale_storage, scale.numel(), 0x6D)
    low_nibble = packed_actual & 0x0F
    high_nibble = packed_actual >> 4
    negative_zero_count = int(
        torch.count_nonzero(
            (((low_nibble & 0x07) == 0) & ((low_nibble & 0x08) != 0))
            | (((high_nibble & 0x07) == 0) & ((high_nibble & 0x08) != 0))
        ).item()
    )
    if negative_zero_count:
        raise AssertionError("BF16->FP4 output contains a negative-zero nibble")
    if not torch.all(scale <= 0x7E).item():
        raise AssertionError("BF16->FP4 scale exceeds finite E4M3 range")
    nonzero_count = int(torch.count_nonzero(packed_actual).item()) + int(
        torch.count_nonzero(scale).item()
    )
    if nonzero_count <= 0:
        raise AssertionError("BF16->FP4 output is all zero")
    return {
        "nonzero_count": nonzero_count,
        "output_sha256": json_sha256(
            {"packed": tensor_sha256(packed_actual), "scale": tensor_sha256(scale)}
        ),
    }


def _run_case(
    *,
    spec: CaseSpec,
    reviewed: dict[str, Any],
    arm: str,
    launch: object,
    provenance: dict[str, Any],
    artifact_before: dict[str, object],
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

    source, global_scale, row_counts = _source(spec)
    packed_reference, scale_reference = _reference(source, global_scale, row_counts)
    packed_storage, packed = _guarded(spec.M * spec.K // 2, 0xD3)
    scale_storage, scale = _guarded(spec.M * spec.K // 16, 0x6D)
    fixed_pointers = {
        "source": source.data_ptr(),
        "global_scale": global_scale.data_ptr(),
        "packed": packed.data_ptr(),
        "scale": scale.data_ptr(),
    }
    initial_input_hashes = {
        "source": tensor_sha256(source),
        "global_scale": tensor_sha256(global_scale),
        "row_counts": tensor_sha256(row_counts),
    }
    read_only_inputs_sha256 = json_sha256(initial_input_hashes)

    launch(source, global_scale, packed, scale)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        launch(source, global_scale, packed, scale)
    torch.cuda.synchronize()
    initial_topology = single_graph_topology(graph)
    expected_topology = reviewed["graph_topology_contract"][arm]
    if reviewed.get("_discovery") is not True and initial_topology != expected_topology:
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")

    scenario_0_output: dict[str, object] | None = None
    for poison in (0x5A, 0xA5):
        packed.fill_(poison)
        scale.fill_(poison)
        before = allocator_counters()
        graph.replay()
        torch.cuda.synchronize()
        after = allocator_counters()
        if before != after:
            raise AssertionError(f"{spec.case_id}: correctness replay allocated")
        scenario_0_output = _validate_output(
            spec=spec,
            packed=packed,
            scale=scale,
            packed_storage=packed_storage,
            scale_storage=scale_storage,
            reference=(packed_reference, scale_reference),
        )
    if scenario_0_output is None:
        raise AssertionError("poison correctness replay was not executed")

    stream = torch.cuda.Stream()
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
    del stream
    scenario_0_post = _validate_output(
        spec=spec,
        packed=packed,
        scale=scale,
        packed_storage=packed_storage,
        scale_storage=scale_storage,
        reference=(packed_reference, scale_reference),
    )
    if scenario_0_post != scenario_0_output:
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    if {
        "source": tensor_sha256(source),
        "global_scale": tensor_sha256(global_scale),
        "row_counts": tensor_sha256(row_counts),
    } != initial_input_hashes:
        raise AssertionError(f"{spec.case_id}: timed read-only input changed")

    source.mul_(-0.75).add_(0.03125)
    live_input_hash = tensor_sha256(source)
    if live_input_hash == initial_input_hashes["source"]:
        raise AssertionError(f"{spec.case_id}: live input mutation was ineffective")
    live_source_snapshot = live_input_hash
    live_reference = _reference(source, global_scale, row_counts)
    live_output: dict[str, object] | None = None
    for poison in (0x3C, 0xC3):
        packed.fill_(poison)
        scale.fill_(poison)
        before = allocator_counters()
        graph.replay()
        torch.cuda.synchronize()
        after = allocator_counters()
        if before != after:
            raise AssertionError(f"{spec.case_id}: live-input replay allocated")
        live_output = _validate_output(
            spec=spec,
            packed=packed,
            scale=scale,
            packed_storage=packed_storage,
            scale_storage=scale_storage,
            reference=live_reference,
        )
    if (
        live_output is None
        or live_output["output_sha256"] == scenario_0_output["output_sha256"]
    ):
        raise AssertionError(f"{spec.case_id}: live input did not change output")
    if tensor_sha256(source) != live_source_snapshot:
        raise AssertionError(f"{spec.case_id}: graph mutated its live source")
    if tensor_sha256(global_scale) != initial_input_hashes["global_scale"]:
        raise AssertionError(f"{spec.case_id}: graph mutated global scale")
    if {
        "source": source.data_ptr(),
        "global_scale": global_scale.data_ptr(),
        "packed": packed.data_ptr(),
        "scale": scale.data_ptr(),
    } != fixed_pointers:
        raise AssertionError(f"{spec.case_id}: graph addresses changed")
    if single_graph_topology(graph) != initial_topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    allocation = allocation_records["warm_l2"]
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
    return {
        "case_id": spec.case_id,
        "case_contract_sha256": reviewed["case_contract_sha256"],
        "input_sha256": json_sha256(spec.input_contract),
        "artifacts": artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": [],
        "correctness": {
            "independent_oracle": True,
            "oracle": "torch-nvfp4-reference",
            "passed": True,
            "finite": True,
            "nonzero_count": scenario_0_output["nonzero_count"],
            "gates": {gate: True for gate in CORRECTNESS_GATES},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": read_only_inputs_sha256,
            "output_sha256": scenario_0_output["output_sha256"],
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

    loaded: list[tuple[CaseSpec, object, dict[str, Any], dict[str, object]]] = []
    original_fingerprint = cute_compiler._b12x_package_fingerprint
    original_cache_dir = os.environ.get("B12X_CUTE_COMPILE_CACHE_DIR")
    try:
        for spec in CASES:
            launch, _, provenance = _load(
                args.cache,
                session.runtime_fingerprint,
                spec.M,
                spec.K,
                args.spec_mode,
            )
            loaded.append((spec, launch, provenance, verify_artifact(provenance)))
    finally:
        cute_compiler._b12x_package_fingerprint = original_fingerprint
        if original_cache_dir is None:
            os.environ.pop("B12X_CUTE_COMPILE_CACHE_DIR", None)
        else:
            os.environ["B12X_CUTE_COMPILE_CACHE_DIR"] = original_cache_dir

    cases = [
        _run_case(
            spec=spec,
            reviewed=session.reviewed_cases[spec.case_id],
            arm=args.arm,
            launch=launch,
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
        for spec, launch, provenance, artifact_before in loaded
    ]
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
