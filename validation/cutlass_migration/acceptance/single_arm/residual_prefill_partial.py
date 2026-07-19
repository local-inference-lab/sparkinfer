#!/usr/bin/env python3
"""Run one CUTLASS arm of the residual prefill-partial E2E graph corpus.

The closed matrix covers the two compact specializations whose register count
increased under CUTLASS 4.6 and the hidden-7168 block-M specialization with the
remaining frontend performance cliff.  Only the exact partial object is timed;
the composite serving route is a separate E2E family.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

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
from b12x.integration import residual_kernels


FAMILY = "residual_prefill_partial"
ARTIFACT_ROLE = "prefill-partial"
INPUT_SCHEMA = "b12x.residual_prefill_partial.end_to_end_input.v1"
TOKENS = 33
PARTIALS = 25
MIXES = 24
MHC_MULT = 4
GRAM_PAIRS = 10
SEED = 20_260_718
CORRECTNESS_GATES = (
    "torch-residual-reference",
    "torch-projection-partials-reference",
    "torch-gram-partials-reference",
    "torch-derived-finalize-reference",
    "finite",
    "nonzero",
    "guard-canaries",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    kernel_kind: str
    hidden_size: int
    split_k: int
    spec_hash: str

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.name}"

    @property
    def block_m(self) -> int:
        return 2 if self.kernel_kind == "block-m" else 1

    @property
    def tile_n(self) -> int | None:
        return 12 if self.kernel_kind == "block-m" else None

    @property
    def input_contract(self) -> dict[str, object]:
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "shape": {
                "tokens": TOKENS,
                "hidden_size": self.hidden_size,
                "split_k": self.split_k,
                "partials": PARTIALS,
                "mixes": MIXES,
                "mhc_mult": MHC_MULT,
            },
            "specialization": {
                "kernel_kind": self.kernel_kind,
                "block_m": self.block_m,
                "tile_n": self.tile_n,
                "compute_gram": True,
            },
            "source": {
                "generator": "torch.cpu.Generator",
                "distribution": "named-randn-divisors",
                "seed": SEED,
            },
        }


CASES = (
    CaseSpec(
        "compact-h4096-m33",
        "compact",
        4_096,
        64,
        "e4695fe84c9f8da938c967b071c57b6e6e957b4026e1c8412d13ee3ed9e9197c",
    ),
    CaseSpec(
        "compact-h7168-m33",
        "compact",
        7_168,
        112,
        "d26e69a632fc33843b7f6da1167605b844138bad91c830792a0d934545f70e29",
    ),
    CaseSpec(
        "block-m2-n12-h7168-m33",
        "block-m",
        7_168,
        112,
        "f821ebda70f4739da06417e0292e1c755106de2710b966b932b141bd4c7ca5fe",
    ),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    return parser.parse_args()


def _make_inputs(spec: CaseSpec) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    def randn(shape: tuple[int, ...], divisor: float) -> torch.Tensor:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .div_(divisor)
            .to("cuda")
            .contiguous()
        )

    hidden = spec.hidden_size
    return {
        "residual": randn((TOKENS, MHC_MULT, hidden), 3).to(torch.bfloat16),
        "x": randn((TOKENS, hidden), 4).to(torch.bfloat16),
        "prev_post": randn((TOKENS, MHC_MULT), 3),
        "prev_comb": randn((TOKENS, MHC_MULT, MHC_MULT), 4),
        "fn": randn((MIXES, MHC_MULT * hidden), 64),
        "scale": randn((3,), 3),
        "bias": randn((MIXES,), 5),
        "norm_weight": randn((hidden,), 2).to(torch.bfloat16),
    }


def _reference(inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    residual = inputs["residual"]
    x = inputs["x"]
    prev_post = inputs["prev_post"]
    prev_comb = inputs["prev_comb"]
    fn = inputs["fn"]
    scale = inputs["scale"]
    bias = inputs["bias"]
    norm_weight = inputs["norm_weight"]
    residual_out = (
        prev_post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (prev_comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(torch.bfloat16)
    flat = residual_out.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + 1e-6
    comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    y_fp32 = (pre.unsqueeze(-1) * residual_out.float()).sum(dim=1)
    y = (
        y_fp32.to(torch.bfloat16).float()
        * torch.rsqrt(y_fp32.square().mean(dim=-1, keepdim=True) + 1e-6)
        * norm_weight.float()
    ).to(torch.bfloat16)
    return residual_out, y, post, comb


def _guarded(
    payload_elements: int,
    *,
    dtype: torch.dtype,
    guard_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    guard_elements = 256
    storage = torch.full(
        (payload_elements + 2 * guard_elements,),
        guard_value,
        dtype=dtype,
        device="cuda",
    )
    return storage, storage[guard_elements : guard_elements + payload_elements]


def _assert_guard(
    storage: torch.Tensor,
    payload_elements: int,
    guard_value: float,
) -> None:
    guard_elements = (storage.numel() - payload_elements) // 2
    if guard_elements <= 0 or not (
        torch.all(storage[:guard_elements] == guard_value).item()
        and torch.all(storage[-guard_elements:] == guard_value).item()
    ):
        raise AssertionError("residual prefill-partial guard canary changed")


def _gram_reference(out: torch.Tensor) -> torch.Tensor:
    value = out.float()
    return torch.stack(
        (
            (value[:, 0] * value[:, 0]).sum(dim=-1),
            (value[:, 1] * value[:, 1]).sum(dim=-1),
            (value[:, 2] * value[:, 2]).sum(dim=-1),
            (value[:, 3] * value[:, 3]).sum(dim=-1),
            (value[:, 0] * value[:, 1]).sum(dim=-1),
            (value[:, 0] * value[:, 2]).sum(dim=-1),
            (value[:, 0] * value[:, 3]).sum(dim=-1),
            (value[:, 1] * value[:, 2]).sum(dim=-1),
            (value[:, 1] * value[:, 3]).sum(dim=-1),
            (value[:, 2] * value[:, 3]).sum(dim=-1),
        ),
        dim=-1,
    )


def _derived_finalize(
    *,
    spec: CaseSpec,
    out: torch.Tensor,
    partials: torch.Tensor,
    inputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projection = partials[:, 0, 1 : MIXES + 1]
    inv_rms = torch.rsqrt(
        partials[:, 0, 0:1] / float(MHC_MULT * spec.hidden_size) + 1e-6
    )
    mixes = projection * inv_rms
    scale = inputs["scale"]
    bias = inputs["bias"]
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + 1e-6
    comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)

    gram = partials[:, 1, :GRAM_PAIRS]
    p0, p1, p2, p3 = pre.unbind(dim=-1)
    sy2 = (
        p0.square() * gram[:, 0]
        + p1.square() * gram[:, 1]
        + p2.square() * gram[:, 2]
        + p3.square() * gram[:, 3]
        + 2 * p0 * p1 * gram[:, 4]
        + 2 * p0 * p2 * gram[:, 5]
        + 2 * p0 * p3 * gram[:, 6]
        + 2 * p1 * p2 * gram[:, 7]
        + 2 * p1 * p3 * gram[:, 8]
        + 2 * p2 * p3 * gram[:, 9]
    )
    y_fp32 = (pre.unsqueeze(-1) * out.float()).sum(dim=1)
    y = (
        y_fp32.to(torch.bfloat16).float()
        * torch.rsqrt(sy2.unsqueeze(-1) / float(spec.hidden_size) + 1e-6)
        * inputs["norm_weight"].float()
    ).to(torch.bfloat16)
    return y, post, comb


def _validate_output(
    *,
    spec: CaseSpec,
    inputs: Mapping[str, torch.Tensor],
    partials: torch.Tensor,
    partial_storage: torch.Tensor,
    out: torch.Tensor,
    out_storage: torch.Tensor,
    reference: tuple[torch.Tensor, ...],
) -> dict[str, object]:
    expected_out, expected_y, expected_post, expected_comb = reference
    torch.testing.assert_close(out, expected_out, rtol=0.0, atol=2e-2)
    projection_reference = F.linear(out.flatten(1).float(), inputs["fn"])
    torch.testing.assert_close(
        partials[:, 0, 1 : MIXES + 1],
        projection_reference,
        rtol=2e-3,
        atol=1e-1,
    )
    gram_reference = _gram_reference(out)
    torch.testing.assert_close(
        partials[:, 1, :GRAM_PAIRS],
        gram_reference,
        rtol=5e-4,
        atol=1.0,
    )
    torch.testing.assert_close(
        partials[:, 0, 0],
        gram_reference[:, :4].sum(dim=-1),
        rtol=5e-4,
        atol=1.0,
    )
    y, post, comb = _derived_finalize(
        spec=spec,
        out=out,
        partials=partials,
        inputs=inputs,
    )
    torch.testing.assert_close(y, expected_y, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(post, expected_post, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(comb, expected_comb, rtol=2e-4, atol=2e-4)
    _assert_guard(partial_storage, partials.numel(), 777.0)
    _assert_guard(out_storage, out.numel(), 123.0)
    checked = (
        out,
        partials[:, 0, : MIXES + 1],
        partials[:, 1, :GRAM_PAIRS],
        y,
        post,
        comb,
    )
    finite = all(bool(torch.isfinite(value).all().item()) for value in checked)
    nonzero_count = sum(int(torch.count_nonzero(value).item()) for value in checked)
    if not finite or nonzero_count <= 0:
        raise AssertionError(
            f"{spec.case_id}: invalid output finite={finite}, nonzero={nonzero_count}"
        )
    return {
        "finite": finite,
        "nonzero_count": nonzero_count,
        "output_sha256": json_sha256(
            {
                "out": tensor_sha256(out),
                "projection_partials": tensor_sha256(partials[:, 0, : MIXES + 1]),
                "gram_partials": tensor_sha256(partials[:, 1, :GRAM_PAIRS]),
            }
        ),
    }


def _runtime_args(
    *,
    spec: CaseSpec,
    inputs: Mapping[str, torch.Tensor],
    partials: torch.Tensor,
    out: torch.Tensor,
) -> tuple[object, ...]:
    return (
        residual_kernels._to_kernel_tensor(
            inputs["x"], residual_kernels.cutlass.BFloat16, dynamic_layout=True
        ),
        residual_kernels._to_kernel_tensor(
            inputs["residual"],
            residual_kernels.cutlass.BFloat16,
            dynamic_layout=True,
        ),
        residual_kernels._to_kernel_tensor(
            inputs["prev_post"],
            residual_kernels.cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        residual_kernels._to_kernel_tensor(
            inputs["prev_comb"],
            residual_kernels.cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        residual_kernels._to_kernel_tensor(
            inputs["fn"], residual_kernels.cutlass.Float32
        ),
        residual_kernels._to_kernel_tensor(
            partials,
            residual_kernels.cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        residual_kernels._to_kernel_tensor(
            out, residual_kernels.cutlass.BFloat16, dynamic_layout=True
        ),
        residual_kernels.Int32(TOKENS),
        residual_kernels.current_cuda_stream(),
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
    inputs = _make_inputs(spec)
    partial_storage, partial_flat = _guarded(
        TOKENS * spec.split_k * PARTIALS,
        dtype=torch.float32,
        guard_value=777.0,
    )
    partials = partial_flat.view(TOKENS, spec.split_k, PARTIALS)
    out_storage, out_flat = _guarded(
        TOKENS * MHC_MULT * spec.hidden_size,
        dtype=torch.bfloat16,
        guard_value=123.0,
    )
    out = out_flat.view(TOKENS, MHC_MULT, spec.hidden_size)
    reference = _reference(inputs)
    live_input_initial = tensor_sha256(inputs["x"])
    static_names = (
        "residual",
        "prev_post",
        "prev_comb",
        "fn",
        "scale",
        "bias",
        "norm_weight",
    )
    static_hashes = {name: tensor_sha256(inputs[name]) for name in static_names}
    read_only_inputs_sha256 = json_sha256({"x": live_input_initial, **static_hashes})
    all_tensors = {
        **inputs,
        "partials": partials,
        "partial_storage": partial_storage,
        "out": out,
        "out_storage": out_storage,
    }
    fixed_pointers = {name: value.data_ptr() for name, value in all_tensors.items()}

    def launch() -> None:
        cute_compiler.run_compiled(
            compiled,
            _runtime_args(
                spec=spec,
                inputs=inputs,
                partials=partials,
                out=out,
            ),
        )

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
            f"{spec.case_id}: expected one exact partial kernel: {initial_topology}"
        )
    if (
        reviewed.get("_discovery") is not True
        and initial_topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")

    def replay_checked(
        expected: tuple[torch.Tensor, ...], poison: float
    ) -> dict[str, object]:
        with torch.cuda.stream(stream):
            partials.fill_(poison)
            out.fill_(poison)
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
        return _validate_output(
            spec=spec,
            inputs=inputs,
            partials=partials,
            partial_storage=partial_storage,
            out=out,
            out_storage=out_storage,
            reference=expected,
        )

    scenario_0: dict[str, object] | None = None
    for poison in (float("nan"), -321.0):
        scenario_0 = replay_checked(reference, poison)
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
    scenario_0_post = replay_checked(reference, float("nan"))
    if scenario_0_post != scenario_0:
        raise AssertionError(f"{spec.case_id}: output changed across timing")
    if tensor_sha256(inputs["x"]) != live_input_initial:
        raise AssertionError(f"{spec.case_id}: timed live input changed")

    inputs["x"].mul_(-0.5).add_(0.0625)
    live_input_mutated = tensor_sha256(inputs["x"])
    if live_input_mutated == live_input_initial:
        raise AssertionError(f"{spec.case_id}: live mutation was ineffective")
    live_reference = _reference(inputs)
    live_output: dict[str, object] | None = None
    for poison in (float("nan"), 321.0):
        live_output = replay_checked(live_reference, poison)
    if (
        live_output is None
        or live_output["output_sha256"] == scenario_0["output_sha256"]
    ):
        raise AssertionError(f"{spec.case_id}: live input did not change output")
    if tensor_sha256(inputs["x"]) != live_input_mutated:
        raise AssertionError(f"{spec.case_id}: graph mutated live input")
    if {name: tensor_sha256(inputs[name]) for name in static_names} != static_hashes:
        raise AssertionError(f"{spec.case_id}: read-only inputs changed")
    if {
        name: value.data_ptr() for name, value in all_tensors.items()
    } != fixed_pointers:
        raise AssertionError(f"{spec.case_id}: tensor addresses changed")
    if single_graph_topology(graph) != initial_topology:
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
            "oracle": "torch-residual-projection-gram-derived-finalize",
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
            "workspace_capacity_bytes": partials.numel() * partials.element_size(),
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
    loaded: list[tuple[CaseSpec, object, Mapping[str, Any], Mapping[str, Any]]] = []
    for spec in CASES:
        compiled, provenance = load_exact(args.cache, spec.spec_hash)
        if provenance["package_fingerprint"] != session.runtime_fingerprint:
            raise RuntimeError(
                f"{spec.case_id}: exact object/runtime fingerprints differ"
            )
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
