#!/usr/bin/env python3
"""Run one frozen CUTLASS arm of the residual-composite E2E corpus.

The closed matrix mirrors
``validation.cutlass_migration.diagnostics.paired.residual_composite``: four
decode composites at one live token and six prefill composites at each of
33/384/1024/2048 live tokens.  Every production CUDA-graph kernel, including
the residual Gram finalizer, is loaded from the selected arm's exact cache.
No comparison arm is imported into the process and no isolated kernel route is
timed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch

import validation.cutlass_migration.diagnostics.paired.residual_composite as paired
from benchmarks.common import make_l2_flush_fn
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    exact_artifact_evidence,
    gpu_mode_snapshot,
    graph_topology,
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
from sparkinfer.integration import (
    SPARKINFERMHCScratchCaps,
    sparkinfer_mhc_post_pre,
    sparkinfer_mhc_pre,
    plan_mhc_scratch,
)
from sparkinfer.integration import residual_kernels
import sparkinfer.cute.compiler as cute_compiler


FAMILY = "residual_composite"
INPUT_SCHEMA = "sparkinfer.residual.composite.end_to_end_input.v1"
DECODE_TOKENS = (1,)
PREFILL_TOKENS = tuple(paired._PREFILL_TOKEN_DEFAULTS)
DECODE_REPLAYS_PER_REPORTED_SAMPLE = 8
PREFILL_REPLAYS_PER_REPORTED_SAMPLE = 1
GUARD_ELEMENTS = 256
CORRECTNESS_GATES = (
    "allocator-stability",
    "finite",
    "gpu-oracle",
    "guard-canaries",
    "live-input-mutation",
    "nonzero",
    "poison-overwrite",
    "read-only-input-immutability",
    "stable-addresses",
)


def _artifact_role(spec_hash: str) -> str:
    try:
        kernel_id = paired._KERNEL_ID_BY_SPEC[spec_hash]
    except KeyError as exc:
        raise RuntimeError(f"unmapped residual compile spec {spec_hash}") from exc
    prefix = "integration.residual."
    if not kernel_id.startswith(prefix):
        raise RuntimeError(f"unexpected residual kernel id {kernel_id!r}")
    return kernel_id.removeprefix(prefix)


@dataclass(frozen=True)
class CaseSpec:
    paired_case: paired.CaseDefinition
    tokens: int

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.paired_case.name}/tokens-{self.tokens}"

    @property
    def expected_m(self) -> int | None:
        if not self.paired_case.prefill:
            return None
        return max(paired._PREFILL_POLICY_MIN, self.tokens)

    @property
    def role_specs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (_artifact_role(spec_hash), spec_hash)
            for spec_hash in self.paired_case.all_specs
        )

    @property
    def input_contract(self) -> dict[str, object]:
        case = self.paired_case
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "paired_case": case.name,
            "shape": {
                "tokens": self.tokens,
                "hidden_size": case.hidden_size,
                "split_k": paired._SPLIT_K[case.hidden_size],
                "expected_m": self.expected_m,
                "mhc_mult": 4,
                "mixes": 24,
            },
            "specialization": {
                "route": case.route,
                "production_entrypoint": (
                    "sparkinfer_mhc_pre"
                    if case.route == "decode-pre"
                    else "sparkinfer_mhc_post_pre"
                ),
                "environment": dict(case.environment),
                "exact_artifacts": [
                    {"role": role, "compile_spec_hash": spec_hash}
                    for role, spec_hash in self.role_specs
                ],
            },
            "source": {
                "generator": "torch.cpu.Generator",
                "distribution": "named-randn-divisors",
                "seed": 2_026_071_900 + case.hidden_size + self.tokens,
                "scenario_count": 2,
            },
            "oracle": {
                "device": "gpu",
                "post": "independent-torch-float32-post-reference",
                "pre": "independent-torch-float32-pre-reference",
                "output_dtype": "torch.bfloat16",
            },
            "workspace": {
                "planner": "plan_mhc_scratch",
                "max_tokens": self.tokens
                if self.expected_m is None
                else self.expected_m,
            },
        }


CASES = tuple(
    CaseSpec(case, tokens)
    for case in paired._CASES.values()
    for tokens in (PREFILL_TOKENS if case.prefill else DECODE_TOKENS)
)

if len(CASES) != 28:
    raise AssertionError(f"expected 28 residual-composite cases, got {len(CASES)}")
if len({case.case_id for case in CASES}) != len(CASES):
    raise AssertionError("residual-composite case identifiers are not unique")
if len({json_sha256(case.input_contract) for case in CASES}) != len(CASES):
    raise AssertionError("residual-composite input identities are not unique")
if tuple(sorted(CORRECTNESS_GATES)) != CORRECTNESS_GATES:
    raise AssertionError("residual-composite correctness gates must remain sorted")


@dataclass(frozen=True)
class GuardedTensor:
    name: str
    storage: torch.Tensor
    payload: torch.Tensor
    guard_value: int | float


@dataclass
class RuntimeCase:
    spec: CaseSpec
    launch: Any
    install: Any
    expected: Any
    outputs: tuple[torch.Tensor, ...]
    output_names: tuple[str, ...]
    stable_tensors: Mapping[str, torch.Tensor]
    read_only_tensors: Mapping[str, torch.Tensor]
    scenario_tensors: Mapping[str, torch.Tensor]
    live_tensors: Mapping[str, torch.Tensor]
    guards: tuple[GuardedTensor, ...]
    scratch_contract: Mapping[str, object]


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
    return parser.parse_args()


def _guarded_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    guard_value: int | float,
) -> GuardedTensor:
    payload_elements = math.prod(shape)
    storage = torch.full(
        (payload_elements + 2 * GUARD_ELEMENTS,),
        guard_value,
        dtype=dtype,
        device=device,
    )
    payload = storage[GUARD_ELEMENTS : GUARD_ELEMENTS + payload_elements].view(shape)
    return GuardedTensor(name, storage, payload, guard_value)


def _assert_guards(guards: tuple[GuardedTensor, ...]) -> None:
    for guard in guards:
        prefix = guard.storage[:GUARD_ELEMENTS]
        suffix = guard.storage[-GUARD_ELEMENTS:]
        if not (
            torch.all(prefix == guard.guard_value).item()
            and torch.all(suffix == guard.guard_value).item()
        ):
            raise AssertionError(f"{guard.name}: guard canary changed")


def _hashes(tensors: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(tensor) for name, tensor in sorted(tensors.items())}


def _pointers(tensors: Mapping[str, torch.Tensor]) -> dict[str, int]:
    return {name: tensor.data_ptr() for name, tensor in sorted(tensors.items())}


def _assert_pointers(
    tensors: Mapping[str, torch.Tensor], expected: Mapping[str, int]
) -> None:
    observed = _pointers(tensors)
    if observed != dict(expected):
        raise AssertionError(f"stable tensor pointers changed: {expected}->{observed}")


def _build_runtime(spec: CaseSpec, *, device: torch.device) -> RuntimeCase:
    case = spec.paired_case
    tokens = spec.tokens
    hidden_size = case.hidden_size
    split_k = paired._SPLIT_K[hidden_size]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2_026_071_900 + hidden_size + tokens)
    residual_scenarios = tuple(
        paired._cpu_randn(
            (tokens, 4, hidden_size),
            generator=generator,
            divisor=3.0 + scenario,
            dtype=torch.bfloat16,
            device=device,
        )
        for scenario in range(2)
    )
    x_scenarios = tuple(
        paired._cpu_randn(
            (tokens, hidden_size),
            generator=generator,
            divisor=4.0 + scenario,
            dtype=torch.bfloat16,
            device=device,
        )
        for scenario in range(2)
    )
    residual = torch.empty_like(residual_scenarios[0])
    x = torch.empty_like(x_scenarios[0])
    fn = paired._cpu_randn(
        (24, 4 * hidden_size),
        generator=generator,
        divisor=64.0,
        dtype=torch.float32,
        device=device,
    )
    fn_bf16 = fn.to(torch.bfloat16).contiguous()
    pre_fn = fn.view(24, 4, hidden_size).sum(dim=1).contiguous()
    scale = paired._cpu_randn(
        (3,),
        generator=generator,
        divisor=3.0,
        dtype=torch.float32,
        device=device,
    )
    bias = paired._cpu_randn(
        (24,),
        generator=generator,
        divisor=5.0,
        dtype=torch.float32,
        device=device,
    )
    norm_weight = paired._cpu_randn(
        (hidden_size,),
        generator=generator,
        divisor=2.0,
        dtype=torch.bfloat16,
        device=device,
    )
    prev_post = (
        0.75
        + paired._cpu_randn(
            (tokens, 4),
            generator=generator,
            divisor=16.0,
            dtype=torch.float32,
            device=device,
        )
    ).contiguous()
    prev_comb = torch.softmax(
        paired._cpu_randn(
            (tokens, 4, 4),
            generator=generator,
            divisor=2.0,
            dtype=torch.float32,
            device=device,
        ),
        dim=1,
    ).contiguous()

    expected_m = spec.expected_m
    max_tokens = tokens if expected_m is None else expected_m
    plan = plan_mhc_scratch(
        SPARKINFERMHCScratchCaps(
            device=device,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            split_k=split_k,
        )
    )
    scratch_guards = tuple(
        _guarded_tensor(
            f"scratch-{index}",
            shape,
            dtype=dtype,
            device=device,
            guard_value=173,
        )
        for index, (shape, dtype) in enumerate(plan.shapes_and_dtypes())
    )
    output_guards = (
        _guarded_tensor(
            "out",
            (tokens, 4, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            guard_value=123.0,
        ),
        _guarded_tensor(
            "post",
            (tokens, 4),
            dtype=torch.float32,
            device=device,
            guard_value=124.0,
        ),
        _guarded_tensor(
            "comb",
            (tokens, 4, 4),
            dtype=torch.float32,
            device=device,
            guard_value=125.0,
        ),
        _guarded_tensor(
            "y",
            (tokens, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            guard_value=126.0,
        ),
    )
    out, post, comb, y = (guard.payload for guard in output_guards)
    scratch = tuple(guard.payload for guard in scratch_guards)
    binding = plan.bind(
        scratch=scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=y,
        post=post,
        comb=comb,
        out=out,
    )
    outputs = (out, post, comb, y)

    def install(scenario: int) -> None:
        x.copy_(x_scenarios[scenario])
        residual.copy_(residual_scenarios[scenario])

    def expected() -> tuple[torch.Tensor, ...]:
        if case.route == "decode-pre":
            residual_ref = x.unsqueeze(1).expand(-1, 4, -1)
            oracle_fn = fn
        else:
            residual_ref = paired._post_reference(x, residual, prev_post, prev_comb)
            oracle_fn = (
                fn_bf16.float() if case.route.startswith("prefill-bf16-") else fn
            )
        y_ref, post_ref, comb_ref = paired._pre_reference(
            residual_ref,
            oracle_fn,
            scale,
            bias,
            norm_weight,
        )
        return residual_ref, post_ref, comb_ref, y_ref

    def launch() -> tuple[torch.Tensor, ...]:
        if case.route == "decode-pre":
            result = sparkinfer_mhc_pre(
                x,
                pre_fn,
                scale,
                bias,
                rms_eps=1.0e-6,
                hc_eps=1.0e-6,
                sinkhorn_iters=20,
                norm_weight=norm_weight,
                norm_eps=1.0e-6,
                binding=binding,
            )
        else:
            result = sparkinfer_mhc_post_pre(
                x,
                residual,
                prev_post,
                prev_comb,
                fn,
                scale,
                bias,
                rms_eps=1.0e-6,
                hc_eps=1.0e-6,
                sinkhorn_iters=20,
                norm_weight=norm_weight,
                norm_eps=1.0e-6,
                fn_bf16=(fn_bf16 if case.route.startswith("prefill-bf16-") else None),
                binding=binding,
            )
        if tuple(value.data_ptr() for value in result) != tuple(
            value.data_ptr() for value in outputs
        ):
            raise AssertionError("production API replaced caller-owned outputs")
        return result

    stable_tensors = {
        "x": x,
        "residual": residual,
        "prev_post": prev_post,
        "prev_comb": prev_comb,
        "fn": fn,
        "fn_bf16": fn_bf16,
        "pre_fn": pre_fn,
        "scale": scale,
        "bias": bias,
        "norm_weight": norm_weight,
        "partials": binding.partials,
        **{guard.name: guard.payload for guard in scratch_guards},
        **{guard.name: guard.payload for guard in output_guards},
        **{
            f"{guard.name}-guard-storage": guard.storage
            for guard in (*scratch_guards, *output_guards)
        },
    }
    read_only_tensors = {
        name: stable_tensors[name]
        for name in (
            "prev_post",
            "prev_comb",
            "fn",
            "fn_bf16",
            "pre_fn",
            "scale",
            "bias",
            "norm_weight",
        )
    }
    scenario_tensors = {
        "residual_scenario_0": residual_scenarios[0],
        "residual_scenario_1": residual_scenarios[1],
        "x_scenario_0": x_scenarios[0],
        "x_scenario_1": x_scenarios[1],
    }
    return RuntimeCase(
        spec=spec,
        launch=launch,
        install=install,
        expected=expected,
        outputs=outputs,
        output_names=("residual", "post", "comb", "y"),
        stable_tensors=stable_tensors,
        read_only_tensors=read_only_tensors,
        scenario_tensors=scenario_tensors,
        live_tensors={"x": x, "residual": residual},
        guards=(*scratch_guards, *output_guards),
        scratch_contract={
            "planner": "plan_mhc_scratch",
            "max_tokens": max_tokens,
            "live_tokens": tokens,
            "expected_m": expected_m,
            "hidden_size": hidden_size,
            "split_k": split_k,
            "scratch_nbytes": plan.layout.nbytes,
            "scratch_shapes_and_dtypes": [
                {"shape": list(shape), "dtype": str(dtype)}
                for shape, dtype in plan.shapes_and_dtypes()
            ],
            "guard_elements_per_side": GUARD_ELEMENTS,
            "guarded_scratch_buffers": len(scratch_guards),
            "guarded_output_buffers": len(output_guards),
        },
    )


def _capture(
    *,
    runtime: RuntimeCase,
    compiled: Mapping[str, object],
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, tuple[str, ...]]:
    expected_specs = tuple(runtime.spec.paired_case.all_specs)
    observed: list[str] = []
    misses_before = int(cute_compiler.compile_cache_info()["compile_misses"])
    with (
        paired._environment(runtime.spec.paired_case.environment),
        pin_module_launches(residual_kernels, compiled, observed),
    ):
        with torch.cuda.stream(stream):
            runtime.launch()
        stream.synchronize()
        if tuple(observed) != expected_specs:
            raise RuntimeError(
                f"{runtime.spec.case_id}: eager exact launch order differs: "
                f"observed={observed}, expected={expected_specs}"
            )
        observed.clear()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            runtime.launch()
        stream.synchronize()
    misses_after = int(cute_compiler.compile_cache_info()["compile_misses"])
    if misses_after != misses_before:
        raise AssertionError(f"{runtime.spec.case_id}: exact capture compiled a kernel")
    if tuple(observed) != expected_specs:
        raise RuntimeError(
            f"{runtime.spec.case_id}: captured exact launch order differs: "
            f"observed={observed}, expected={expected_specs}"
        )
    return graph, tuple(observed)


def _poison_count(output: torch.Tensor, poison: float) -> int:
    if math.isnan(poison):
        return int(torch.isnan(output).sum().item())
    return int((output == poison).sum().item())


def _validate_replay(
    *,
    runtime: RuntimeCase,
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    scenario: int,
    poison: float,
    stable_pointers: Mapping[str, int],
) -> dict[str, object]:
    with torch.cuda.stream(stream):
        runtime.install(scenario)
        expected = runtime.expected()
    stream.synchronize()
    live_before = _hashes(runtime.live_tensors)
    with torch.cuda.stream(stream):
        for output in runtime.outputs:
            output.fill_(poison)
    stream.synchronize()
    allocator_before = allocator_counters()
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    allocator_after = allocator_counters()
    if allocator_after != allocator_before:
        raise AssertionError(
            f"{runtime.spec.case_id}: correctness replay allocated: "
            f"{allocator_before}->{allocator_after}"
        )
    _assert_pointers(runtime.stable_tensors, stable_pointers)
    _assert_guards(runtime.guards)
    live_after = _hashes(runtime.live_tensors)
    if live_after != live_before:
        raise AssertionError(f"{runtime.spec.case_id}: graph mutated live inputs")
    poison_counts = {
        name: _poison_count(output, poison)
        for name, output in zip(runtime.output_names, runtime.outputs, strict=True)
    }
    if any(poison_counts.values()):
        raise AssertionError(
            f"{runtime.spec.case_id}: graph left poisoned output elements: "
            f"{poison_counts}"
        )
    tolerances = (
        (0.0, 2.0e-2),
        (2.0e-4, 2.0e-4),
        (2.0e-4, 2.0e-4),
        (2.0e-2, 2.0e-2),
    )
    metrics = {
        name: paired._metrics(actual, reference, rtol=rtol, atol=atol)
        for name, actual, reference, (rtol, atol) in zip(
            runtime.output_names,
            runtime.outputs,
            expected,
            tolerances,
            strict=True,
        )
    }
    return {
        "scenario": scenario,
        "poison": "nan" if math.isnan(poison) else poison,
        "poisoned_elements_after": poison_counts,
        "allocator_before": allocator_before,
        "allocator_after": allocator_after,
        "zero_replay_allocations": True,
        "live_input_sha256": live_before,
        "outputs": metrics,
        "output_sha256": json_sha256(
            {name: value["sha256"] for name, value in metrics.items()}
        ),
    }


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


def _run_case(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    loaded: Mapping[str, tuple[object, Mapping[str, Any], Mapping[str, Any]]],
    args: argparse.Namespace,
    expected_physical_gpu: int,
) -> dict[str, object]:
    compiled = {spec_hash: loaded[spec_hash][0] for _, spec_hash in spec.role_specs}
    runtime = _build_runtime(
        spec,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    runtime.install(0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    stable_pointers = _pointers(runtime.stable_tensors)
    immutable_tensors = {
        **runtime.read_only_tensors,
        **runtime.scenario_tensors,
    }
    immutable_before = _hashes(immutable_tensors)
    read_only_inputs_sha256 = json_sha256(immutable_before)
    scenario_hashes = {
        "x": [
            immutable_before["x_scenario_0"],
            immutable_before["x_scenario_1"],
        ],
        "residual": [
            immutable_before["residual_scenario_0"],
            immutable_before["residual_scenario_1"],
        ],
    }
    if scenario_hashes["x"][0] == scenario_hashes["x"][1]:
        raise AssertionError(f"{spec.case_id}: x scenarios are identical")
    if (
        spec.paired_case.route != "decode-pre"
        and scenario_hashes["residual"][0] == scenario_hashes["residual"][1]
    ):
        raise AssertionError(f"{spec.case_id}: residual scenarios are identical")

    graph, observed_specs = _capture(
        runtime=runtime,
        compiled=compiled,
        stream=stream,
    )
    full_topology = graph_topology(graph)
    topology = single_graph_topology(graph)
    expected_nodes = len(spec.role_specs)
    if int(topology["kernel_node_count"]) != expected_nodes:
        raise RuntimeError(
            f"{spec.case_id}: expected {expected_nodes} exact graph kernels, "
            f"got {topology['kernel_node_count']}"
        )
    if (
        reviewed.get("_discovery") is not True
        and topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")
    source_owned_kernel_nodes: list[dict[str, object]] = []
    reviewed_source_owned = reviewed["compile_artifact_contract"][arm][
        "source_owned_kernel_nodes"
    ]
    if (
        reviewed.get("_discovery") is not True
        and reviewed_source_owned != source_owned_kernel_nodes
    ):
        raise RuntimeError(
            f"{spec.case_id}: reviewed source-owned graph nodes are not empty"
        )
    observed_roles = tuple(_artifact_role(spec_hash) for spec_hash in observed_specs)
    if observed_roles != tuple(role for role, _ in spec.role_specs):
        raise RuntimeError(f"{spec.case_id}: exact launch-role order changed")

    scenario_0_replays = [
        _validate_replay(
            runtime=runtime,
            graph=graph,
            stream=stream,
            scenario=0,
            poison=poison,
            stable_pointers=stable_pointers,
        )
        for poison in (math.nan, -321.0)
    ]
    if len({record["output_sha256"] for record in scenario_0_replays}) != 1:
        raise AssertionError(f"{spec.case_id}: poison changed scenario-0 output")

    with torch.cuda.stream(stream):
        runtime.install(0)
        make_l2_flush_fn(True, args.l2_flush_bytes)
    stream.synchronize()
    allocation_before_timing = allocator_counters()
    replays_per_sample = (
        args.prefill_replays_per_reported_sample
        if spec.paired_case.prefill
        else args.decode_replays_per_reported_sample
    )
    conditions, allocation_records = time_single_graph_conditions(
        graph,
        precondition=args.precondition,
        warmup=args.warmup,
        replays=args.replays,
        stream=stream,
        l2_flush_bytes=args.l2_flush_bytes,
        replays_per_reported_sample=replays_per_sample,
        event_batch_replays=args.event_batch_replays,
        precondition_seconds=args.precondition_seconds,
        maximum_precondition_seconds=args.maximum_precondition_seconds,
        mode_snapshot=lambda: gpu_mode_snapshot(expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
    )
    allocation_after_timing = allocator_counters()
    if allocation_after_timing != allocation_before_timing:
        raise AssertionError(
            f"{spec.case_id}: allocator changed across timing: "
            f"{allocation_before_timing}->{allocation_after_timing}"
        )

    scenario_0_post = _validate_replay(
        runtime=runtime,
        graph=graph,
        stream=stream,
        scenario=0,
        poison=math.nan,
        stable_pointers=stable_pointers,
    )
    if scenario_0_post["output_sha256"] != scenario_0_replays[0]["output_sha256"]:
        raise AssertionError(f"{spec.case_id}: scenario-0 output changed across timing")
    scenario_1_replays = [
        _validate_replay(
            runtime=runtime,
            graph=graph,
            stream=stream,
            scenario=1,
            poison=poison,
            stable_pointers=stable_pointers,
        )
        for poison in (math.nan, 321.0)
    ]
    if len({record["output_sha256"] for record in scenario_1_replays}) != 1:
        raise AssertionError(f"{spec.case_id}: poison changed scenario-1 output")
    if scenario_1_replays[0]["output_sha256"] == scenario_0_replays[0]["output_sha256"]:
        raise AssertionError(f"{spec.case_id}: live-input mutation changed no output")

    immutable_after = _hashes(immutable_tensors)
    if immutable_after != immutable_before:
        raise AssertionError(f"{spec.case_id}: read-only inputs changed")
    _assert_pointers(runtime.stable_tensors, stable_pointers)
    _assert_guards(runtime.guards)
    if single_graph_topology(graph) != topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")

    artifacts, launch_plan = _artifact_bindings(
        spec=spec,
        reviewed=reviewed,
        arm=arm,
        loaded=loaded,
        observed_roles=observed_roles,
    )
    if [binding["node_index"] for binding in launch_plan] != list(
        range(expected_nodes)
    ):
        raise AssertionError(f"{spec.case_id}: exact graph-node coverage is incomplete")
    kernel_nodes = [
        node
        for node in full_topology["nodes"]
        if node["type"] == "CU_GRAPH_NODE_TYPE_KERNEL"
    ]
    if len(kernel_nodes) != expected_nodes:
        raise AssertionError(f"{spec.case_id}: full topology kernel count changed")

    baseline_outputs = scenario_0_replays[0]["outputs"]
    finite = all(bool(value["finite"]) for value in baseline_outputs.values())
    nonzero_count = sum(int(value["nonzero"]) for value in baseline_outputs.values())
    if not finite or nonzero_count <= 0:
        raise AssertionError(
            f"{spec.case_id}: invalid finite/nonzero result: "
            f"finite={finite}, nonzero={nonzero_count}"
        )
    allocation = allocation_records["warm_l2"]
    return {
        "case_id": spec.case_id,
        "case_contract_sha256": reviewed["case_contract_sha256"],
        "input_sha256": json_sha256(spec.input_contract),
        "artifacts": artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": source_owned_kernel_nodes,
        "correctness": {
            "independent_oracle": True,
            "oracle": "torch-gpu-residual-post-pre-reference",
            "passed": True,
            "finite": finite,
            "nonzero_count": nonzero_count,
            "gates": {gate: True for gate in CORRECTNESS_GATES},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": read_only_inputs_sha256,
            "scenario_input_sha256": scenario_hashes,
            "scenario_0_replays": scenario_0_replays,
            "scenario_0_post_timing": scenario_0_post,
            "scenario_1_replays": scenario_1_replays,
            "output_sha256": scenario_0_replays[0]["output_sha256"],
        },
        "graph": {
            "capture_passed": True,
            "replay_passed": True,
            "topology_stable": True,
            "addresses_stable": True,
            "live_input_changed_output": True,
            "poison_overwrite_passed": True,
            "guard_canaries_passed": True,
            "observed_compile_spec_hashes": list(observed_specs),
            "observed_artifact_roles": list(observed_roles),
            "kernel_nodes": kernel_nodes,
            **topology,
        },
        "allocation": {
            "fixed_workspace_capacity": True,
            "workspace_capacity_bytes": int(runtime.scratch_contract["scratch_nbytes"]),
            "scratch_contract": dict(runtime.scratch_contract),
            "stable_addresses": True,
            "fixed_pointers": stable_pointers,
            "allocator_stable": True,
            "zero_replay_allocations": True,
            "allocation_before_timing": allocation_before_timing,
            "allocation_after_timing": allocation_after_timing,
            **allocation,
            "condition_counters": allocation_records,
        },
        "conditions": conditions,
    }


def main() -> int:
    args = _args()
    if (
        min(
            args.decode_replays_per_reported_sample,
            args.prefill_replays_per_reported_sample,
        )
        <= 0
    ):
        raise ValueError("replays per reported sample must be positive")
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
    unique_specs = tuple(
        dict.fromkeys(spec_hash for spec in CASES for _, spec_hash in spec.role_specs)
    )
    loaded: dict[str, tuple[object, Mapping[str, Any], Mapping[str, Any]]] = {}
    for spec_hash in unique_specs:
        compiled, provenance = load_exact(args.cache, spec_hash)
        if provenance["package_fingerprint"] != session.runtime_fingerprint:
            raise RuntimeError(
                f"{spec_hash}: exact object/runtime package fingerprints differ"
            )
        expected_kernel_id = paired._KERNEL_ID_BY_SPEC[spec_hash]
        if provenance["kernel_id"] != expected_kernel_id:
            raise RuntimeError(
                f"{spec_hash}: exact object kernel is {provenance['kernel_id']!r}, "
                f"expected {expected_kernel_id!r}"
            )
        loaded[spec_hash] = (compiled, provenance, verify_artifact(provenance))

    cases = [
        _run_case(
            spec=spec,
            reviewed=session.reviewed_cases[spec.case_id],
            arm=session.arm,
            loaded=loaded,
            args=args,
            expected_physical_gpu=session.expected_physical_gpu,
        )
        for spec in CASES
    ]
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
