"""Shared lifecycle for one-arm CUTLASS migration GPU evidence producers.

This module owns only the process envelope: frozen source/contract loading,
physical-GPU selection, reviewed-case binding, timestamps, mode snapshots, and
atomic result publication.  Family producers continue to own their inputs,
oracles, graph capture, live-input proofs, workspace policy, and exact-object
loading so those contracts remain visible and reviewable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
    validate_evidence_status,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    build_single_arm_e2e_result,
    gpu_mode_snapshot,
    json_sha256,
    sha256_file,
    verify_artifact,
)
from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument,
    require_target_gpu,
)
from validation.cutlass_migration.paths import REPO_ROOT
import b12x.cute.compiler as cute_compiler


FAMILY_DISCOVERY_SCHEMA = "b12x.cute.migration.family_discovery.v1"


@dataclass(frozen=True)
class ReviewedCaseBinding:
    """Producer-owned identity that must exactly match the frozen contract."""

    case_id: str
    input_sha256: str
    correctness_gates: tuple[str, ...]
    cross_arm_output_policy: str = "oracle-only"
    topology_review_disposition: str = "equal"
    topology_review_reason: str = ""


@dataclass(frozen=True)
class SingleArmSession:
    phase: str
    family: str
    arm: str
    sequence_position: str
    evidence_status: str
    expected_physical_gpu: int
    repo_root: Path
    producer_path: Path
    source_manifest_path: Path
    contract_path: Path | None
    output_path: Path
    started_unix_ns: int
    gpu_mode_before: Mapping[str, Any]
    runtime_fingerprint: str
    source_manifest: Mapping[str, Any]
    contract: Mapping[str, Any]
    reviewed_cases: Mapping[str, Mapping[str, Any]]
    bindings: Mapping[str, ReviewedCaseBinding]


def add_single_arm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the invariant CLI shared by every one-arm process producer."""

    add_target_gpu_argument(parser)
    add_evidence_status_argument(parser)
    parser.add_argument(
        "--phase",
        choices=("discover", "final"),
        default="final",
        help="discover exact graph/object identities, or run against a frozen contract",
    )
    parser.add_argument("--arm", choices=("baseline", "current"), required=True)
    parser.add_argument(
        "--sequence-position", choices=("a1", "b1", "b2", "a2"), required=True
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        help="required for --phase final and forbidden for --phase discover",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precondition", type=int, default=2_000)
    parser.add_argument(
        "--precondition-seconds",
        type=float,
        default=5.0,
        help="minimum target-graph activity per cache condition",
    )
    parser.add_argument(
        "--maximum-precondition-seconds",
        type=float,
        default=30.0,
        help="fail if the target graph cannot reach unthrottled P1 in time",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--replays", type=int, default=1_000)
    parser.add_argument("--event-batch-replays", type=int, default=25)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)


def load_hashed_json(path: Path, hash_field: str) -> dict[str, Any]:
    """Load one frozen JSON object and verify its embedded canonical hash."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read frozen JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"frozen JSON is not an object: {path}")
    recorded = value.get(hash_field)
    payload = {key: item for key, item in value.items() if key != hash_field}
    if recorded != json_sha256(payload):
        raise RuntimeError(f"frozen JSON canonical hash mismatch: {path}")
    return value


def _reviewed_cases(
    contract: Mapping[str, Any],
    *,
    family: str,
    arm: str,
    bindings: Sequence[ReviewedCaseBinding],
) -> dict[str, dict[str, Any]]:
    families = contract.get("families")
    family_contract = families.get(family) if isinstance(families, dict) else None
    raw_cases = (
        family_contract.get("cases") if isinstance(family_contract, dict) else None
    )
    if not isinstance(raw_cases, list):
        raise RuntimeError(f"frozen contract has no {family} cases")
    reviewed = {
        str(case.get("case_id")): case for case in raw_cases if isinstance(case, dict)
    }
    expected = {binding.case_id: binding for binding in bindings}
    if len(expected) != len(bindings):
        raise RuntimeError(f"{family}: producer case identifiers are not unique")
    if set(reviewed) != set(expected) or len(reviewed) != len(bindings):
        raise RuntimeError(
            f"frozen {family} case set differs from producer: "
            f"missing={sorted(set(expected) - set(reviewed))}, "
            f"unexpected={sorted(set(reviewed) - set(expected))}"
        )
    for case_id, binding in expected.items():
        record = reviewed[case_id]
        if record.get("input_sha256") != binding.input_sha256:
            raise RuntimeError(f"{case_id}: frozen input identity differs")
        if record.get("required_correctness_gates") != list(binding.correctness_gates):
            raise RuntimeError(f"{case_id}: correctness gate contract differs")
        compile_contract = record.get("compile_artifact_contract")
        arm_contract = (
            compile_contract.get(arm) if isinstance(compile_contract, dict) else None
        )
        if not isinstance(arm_contract, dict) or set(arm_contract) != {
            "artifacts",
            "launch_plan",
            "source_owned_kernel_nodes",
        }:
            raise RuntimeError(f"{case_id}: missing {arm} compile contract")
        artifacts = arm_contract["artifacts"]
        launch_plan = arm_contract["launch_plan"]
        source_owned = arm_contract["source_owned_kernel_nodes"]
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or not isinstance(launch_plan, list)
            or not launch_plan
            or not isinstance(source_owned, list)
        ):
            raise RuntimeError(f"{case_id}: empty {arm} artifact or launch contract")
    return reviewed


def _discovery_cases(
    *, family: str, arm: str, bindings: Sequence[ReviewedCaseBinding]
) -> dict[str, dict[str, Any]]:
    """Build producer-owned case identities without inventing compile facts.

    Discovery deliberately leaves the artifact, launch, and topology contracts
    empty.  Those facts are observed from the real graph and exact cache object,
    independently reviewed, and only then frozen by the E2E contract builder.
    """

    by_id = {binding.case_id: binding for binding in bindings}
    if len(by_id) != len(bindings):
        raise RuntimeError(f"{family}: producer case identifiers are not unique")
    reviewed: dict[str, dict[str, Any]] = {}
    for case_id, binding in by_id.items():
        if binding.cross_arm_output_policy not in {"bit-exact", "oracle-only"}:
            raise RuntimeError(f"{case_id}: invalid cross-arm output policy")
        if binding.topology_review_disposition not in {"equal", "changed-reviewed"}:
            raise RuntimeError(f"{case_id}: invalid topology review disposition")
        if (
            binding.topology_review_disposition == "equal"
            and binding.topology_review_reason
        ) or (
            binding.topology_review_disposition == "changed-reviewed"
            and not binding.topology_review_reason.strip()
        ):
            raise RuntimeError(f"{case_id}: invalid topology review reason")
        case_identity = {
            "family": family,
            "case_id": case_id,
            "input_sha256": binding.input_sha256,
            "required_correctness_gates": sorted(binding.correctness_gates),
            "cross_arm_output_policy": binding.cross_arm_output_policy,
            "topology_review": {
                "disposition": binding.topology_review_disposition,
                "reason": binding.topology_review_reason,
            },
        }
        reviewed[case_id] = {
            **case_identity,
            "case_contract_sha256": json_sha256(case_identity),
            "_discovery": True,
            "compile_artifact_contract": {
                arm: {
                    "artifacts": [],
                    "launch_plan": [],
                    "source_owned_kernel_nodes": [],
                }
            },
            "graph_topology_contract": {arm: None},
        }
    return reviewed


def begin_single_arm_session(
    args: argparse.Namespace,
    *,
    family: str,
    producer_path: Path,
    bindings: Sequence[ReviewedCaseBinding],
) -> SingleArmSession:
    """Open a fail-closed one-arm process session before loading any object."""

    started_unix_ns = time.time_ns()
    repo_root = REPO_ROOT
    evidence_status = validate_evidence_status(args.evidence_status)
    require_target_gpu(args.expected_physical_gpu)
    gpu_mode_before = gpu_mode_snapshot(args.expected_physical_gpu)
    # Force CUDA initialization before family code freezes pointer/counter state.
    torch.empty(1, dtype=torch.uint8, device="cuda")

    phase = str(getattr(args, "phase", "final"))
    if phase == "final" and args.contract is None:
        raise RuntimeError("--contract is required for --phase final")
    if phase == "discover" and args.contract is not None:
        raise RuntimeError("--contract is forbidden for --phase discover")
    source_manifest_path = Path(args.source_manifest).resolve()
    contract_path = Path(args.contract).resolve() if args.contract is not None else None
    source_manifest = load_hashed_json(source_manifest_path, "manifest_sha256")
    contract = (
        load_hashed_json(contract_path, "contract_sha256")
        if contract_path is not None
        else {}
    )
    if source_manifest.get("side") != args.arm:
        raise RuntimeError("source manifest side differs from requested arm")
    source_runtime = source_manifest.get("runtime")
    runtime_package = (
        source_runtime.get("b12x_package") if isinstance(source_runtime, dict) else None
    )
    if not isinstance(runtime_package, dict):
        raise RuntimeError("source manifest lacks runtime b12x package")
    runtime_fingerprint = str(runtime_package.get("fingerprint", ""))
    if cute_compiler._b12x_package_fingerprint() != runtime_fingerprint:
        raise RuntimeError("active b12x package differs from source manifest")
    reviewed = (
        _reviewed_cases(
            contract,
            family=family,
            arm=args.arm,
            bindings=bindings,
        )
        if phase == "final"
        else _discovery_cases(family=family, arm=args.arm, bindings=bindings)
    )
    binding_map = {binding.case_id: binding for binding in bindings}
    return SingleArmSession(
        phase=phase,
        family=family,
        arm=args.arm,
        sequence_position=args.sequence_position,
        evidence_status=evidence_status,
        expected_physical_gpu=args.expected_physical_gpu,
        repo_root=repo_root,
        producer_path=producer_path.resolve(),
        source_manifest_path=source_manifest_path,
        contract_path=contract_path,
        output_path=Path(args.output).resolve(),
        started_unix_ns=started_unix_ns,
        gpu_mode_before=gpu_mode_before,
        runtime_fingerprint=runtime_fingerprint,
        source_manifest=source_manifest,
        contract=contract,
        reviewed_cases=reviewed,
        bindings=binding_map,
    )


def verify_case_compile_contract(
    *,
    case_id: str,
    reviewed: Mapping[str, Any],
    arm: str,
    role: str,
    provenance: Mapping[str, Any],
) -> None:
    if reviewed.get("_discovery") is True:
        return
    arm_contract = reviewed["compile_artifact_contract"][arm]
    matches = [
        artifact
        for artifact in arm_contract["artifacts"]
        if isinstance(artifact, Mapping) and artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{case_id}: {role!r} is not one reviewed artifact role")
    compile_contract = matches[0]
    fields = ("kernel_id", "compile_spec_hash", "compile_spec_json")
    if any(provenance.get(field) != compile_contract.get(field) for field in fields):
        raise RuntimeError(
            f"{case_id}: {role} exact object differs from compile contract"
        )


def bind_exact_artifact(
    *,
    role: str,
    evidence: Mapping[str, Any],
) -> dict[str, object]:
    """Bind one verified cache object to a stable graph role."""

    required = ("kernel_id", "compile_spec_hash", "object_sha256")
    if not role or any(not isinstance(evidence.get(field), str) for field in required):
        raise RuntimeError(f"cannot bind malformed exact artifact role {role!r}")
    return {
        "role": role,
        "kernel_id": evidence["kernel_id"],
        "compile_spec_hash": evidence["compile_spec_hash"],
        "object_sha256": evidence["object_sha256"],
        "evidence": dict(evidence),
    }


def build_exact_launch_plan(
    *,
    case_id: str,
    reviewed: Mapping[str, Any],
    arm: str,
    artifacts: Sequence[Mapping[str, Any]],
    observed_roles: Sequence[str | tuple[int, str]],
) -> list[dict[str, object]]:
    """Bind exact objects to their full-graph kernel-node ordinals.

    A string-only sequence retains the convenient all-exact behavior and maps
    roles to contiguous ordinals.  Mixed source-owned graphs pass explicit
    ``(kernel_ordinal, role)`` pairs so Torch/Triton nodes are never renumbered
    away or falsely attributed to a CUTLASS cache object.
    """

    by_role = {str(artifact.get("role")): artifact for artifact in artifacts}
    if len(by_role) != len(artifacts) or not by_role:
        raise RuntimeError(f"{case_id}: exact artifact roles are empty or duplicated")
    multiplicities: dict[str, int] = {}
    plan: list[dict[str, object]] = []
    previous_node_index = -1
    for sequence_index, observed in enumerate(observed_roles):
        if isinstance(observed, str):
            node_index, role = sequence_index, observed
        elif (
            isinstance(observed, tuple)
            and len(observed) == 2
            and isinstance(observed[0], int)
            and not isinstance(observed[0], bool)
            and isinstance(observed[1], str)
        ):
            node_index, role = observed
        else:
            raise RuntimeError(f"{case_id}: malformed exact graph binding {observed!r}")
        if node_index < 0 or node_index <= previous_node_index:
            raise RuntimeError(
                f"{case_id}: exact graph ordinals must be nonnegative and "
                "strictly increasing"
            )
        previous_node_index = node_index
        artifact = by_role.get(role)
        if artifact is None:
            raise RuntimeError(f"{case_id}: launch references unknown role {role!r}")
        multiplicity = multiplicities.get(role, 0) + 1
        multiplicities[role] = multiplicity
        plan.append(
            {
                "node_index": node_index,
                "artifact_role": role,
                "kernel_id": artifact["kernel_id"],
                "compile_spec_hash": artifact["compile_spec_hash"],
                "object_sha256": artifact["object_sha256"],
                "multiplicity_index": multiplicity,
            }
        )
    if set(multiplicities) != set(by_role):
        raise RuntimeError(f"{case_id}: graph does not use every exact artifact")
    normalized = [
        {key: binding[key] for key in binding if key != "object_sha256"}
        for binding in plan
    ]
    if (
        reviewed.get("_discovery") is not True
        and normalized != reviewed["compile_artifact_contract"][arm]["launch_plan"]
    ):
        raise RuntimeError(
            f"{case_id}: observed launch order or multiplicity differs from review"
        )
    return plan


def _build_family_discovery_result(
    session: SingleArmSession,
    cases: list[dict[str, object]],
    *,
    gpu_mode_after: Mapping[str, Any],
    finished_unix_ns: int,
) -> dict[str, object]:
    """Validate and publish one family's non-circular discovery fragment."""

    if session.contract_path is not None or session.contract:
        raise RuntimeError("discovery must not consume a final E2E contract")
    if finished_unix_ns <= session.started_unix_ns:
        raise RuntimeError("discovery timestamps are not ordered")
    source_runtime = session.source_manifest.get("runtime")
    runtime_package = (
        source_runtime.get("b12x_package")
        if isinstance(source_runtime, Mapping)
        else None
    )
    if (
        not isinstance(source_runtime, Mapping)
        or source_runtime.get("repo_root") != str(session.repo_root)
        or not isinstance(runtime_package, Mapping)
        or runtime_package.get("fingerprint") != session.runtime_fingerprint
        or cute_compiler._b12x_package_fingerprint() != session.runtime_fingerprint
    ):
        raise RuntimeError("discovery runtime differs from its frozen source manifest")

    emitted = {str(case.get("case_id")): case for case in cases}
    if len(emitted) != len(cases) or set(emitted) != set(session.bindings):
        raise RuntimeError(
            f"{session.family}: discovery case set differs from the producer binding"
        )
    policies: list[dict[str, object]] = []
    for case_id in sorted(emitted):
        case = emitted[case_id]
        binding = session.bindings[case_id]
        reviewed = session.reviewed_cases[case_id]
        if case.get("input_sha256") != binding.input_sha256 or case.get(
            "case_contract_sha256"
        ) != reviewed.get("case_contract_sha256"):
            raise RuntimeError(f"{case_id}: discovery input identity differs")
        correctness = case.get("correctness")
        gates = correctness.get("gates") if isinstance(correctness, Mapping) else None
        expected_gates = set(binding.correctness_gates)
        if (
            not isinstance(correctness, Mapping)
            or correctness.get("independent_oracle") is not True
            or correctness.get("passed") is not True
            or correctness.get("finite") is not True
            or not isinstance(correctness.get("nonzero_count"), int)
            or int(correctness["nonzero_count"]) <= 0
            or correctness.get("read_only_inputs_immutable") is not True
            or not isinstance(gates, Mapping)
            or set(gates) != expected_gates
            or any(value is not True for value in gates.values())
        ):
            raise RuntimeError(f"{case_id}: discovery correctness gates did not pass")

        graph = case.get("graph")
        if not isinstance(graph, Mapping) or any(
            graph.get(field) is not True
            for field in (
                "capture_passed",
                "replay_passed",
                "topology_stable",
                "addresses_stable",
                "live_input_changed_output",
                "poison_overwrite_passed",
            )
        ):
            raise RuntimeError(f"{case_id}: discovery graph gates did not pass")
        kernel_node_count = graph.get("kernel_node_count")
        nodes = graph.get("nodes")
        if (
            not isinstance(kernel_node_count, int)
            or isinstance(kernel_node_count, bool)
            or kernel_node_count < 1
            or not isinstance(nodes, list)
            or len(nodes) != graph.get("node_count")
        ):
            raise RuntimeError(f"{case_id}: discovery graph topology is malformed")

        allocation = case.get("allocation")
        if not isinstance(allocation, Mapping) or any(
            allocation.get(field) is not True
            for field in (
                "fixed_workspace_capacity",
                "stable_addresses",
                "allocator_stable",
                "zero_replay_allocations",
            )
        ):
            raise RuntimeError(f"{case_id}: discovery allocation gates did not pass")

        artifacts = case.get("artifacts")
        launch_plan = case.get("launch_plan")
        source_owned = case.get("source_owned_kernel_nodes")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or not isinstance(launch_plan, list)
            or not launch_plan
            or not isinstance(source_owned, list)
        ):
            raise RuntimeError(f"{case_id}: discovery artifact bindings are incomplete")
        artifact_roles: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise RuntimeError(f"{case_id}: malformed discovery artifact")
            role = artifact.get("role")
            evidence = artifact.get("evidence")
            if (
                not isinstance(role, str)
                or not role
                or role in artifact_roles
                or not isinstance(evidence, Mapping)
                or evidence.get("package_fingerprint") != session.runtime_fingerprint
                or any(
                    artifact.get(field) != evidence.get(field)
                    for field in ("kernel_id", "compile_spec_hash", "object_sha256")
                )
            ):
                raise RuntimeError(f"{case_id}: exact discovery artifact is invalid")
            expected_verification = {
                "passed": True,
                "manifest_sha256": evidence.get("manifest_sha256"),
                "object_sha256": evidence.get("object_sha256"),
                "object_bytes": evidence.get("object_bytes"),
            }
            if (
                evidence.get("verification_before") != expected_verification
                or evidence.get("verification_after") != expected_verification
                or verify_artifact(evidence) != expected_verification
            ):
                raise RuntimeError(
                    f"{case_id}: exact artifact changed during discovery"
                )
            artifact_roles.add(role)

        exact_indices: list[int] = []
        used_roles: set[str] = set()
        previous = -1
        multiplicities: dict[str, int] = {}
        for binding_record in launch_plan:
            if not isinstance(binding_record, Mapping):
                raise RuntimeError(f"{case_id}: malformed discovery launch binding")
            node_index = binding_record.get("node_index")
            role = binding_record.get("artifact_role")
            multiplicity = multiplicities.get(str(role), 0) + 1
            if (
                not isinstance(node_index, int)
                or isinstance(node_index, bool)
                or node_index <= previous
                or node_index >= kernel_node_count
                or role not in artifact_roles
                or binding_record.get("multiplicity_index") != multiplicity
            ):
                raise RuntimeError(f"{case_id}: discovery launch plan is invalid")
            previous = node_index
            multiplicities[str(role)] = multiplicity
            used_roles.add(str(role))
            exact_indices.append(node_index)
        source_indices = [
            record.get("node_index")
            for record in source_owned
            if isinstance(record, Mapping)
        ]
        if (
            len(source_indices) != len(source_owned)
            or any(
                not isinstance(node_index, int) or isinstance(node_index, bool)
                for node_index in source_indices
            )
            or used_roles != artifact_roles
            or sorted([*exact_indices, *source_indices])
            != list(range(kernel_node_count))
        ):
            raise RuntimeError(f"{case_id}: discovery graph coverage is incomplete")

        policies.append(
            {
                "case_id": case_id,
                "input_sha256": binding.input_sha256,
                "required_correctness_gates": sorted(binding.correctness_gates),
                "cross_arm_output_policy": binding.cross_arm_output_policy,
                "topology_review": {
                    "disposition": binding.topology_review_disposition,
                    "reason": binding.topology_review_reason,
                },
            }
        )

    producer_relative = session.producer_path.relative_to(session.repo_root).as_posix()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload: dict[str, object] = {
        "schema": FAMILY_DISCOVERY_SCHEMA,
        "family": session.family,
        "arm": session.arm,
        "sequence_position": session.sequence_position,
        "evidence_status": session.evidence_status,
        "invocation": {
            "pid": os.getpid(),
            "started_unix_ns": session.started_unix_ns,
            "finished_unix_ns": finished_unix_ns,
        },
        "source": {
            "manifest_sha256": session.source_manifest["manifest_sha256"],
            "manifest_artifact_sha256": sha256_file(session.source_manifest_path),
            "runtime_package_fingerprint": session.runtime_fingerprint,
        },
        "producer": {
            "path": producer_relative,
            "sha256": sha256_file(session.producer_path),
        },
        "gpu": {
            "physical_ordinal": session.expected_physical_gpu,
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "capability": list(torch.cuda.get_device_capability()),
            "mode_before": dict(session.gpu_mode_before),
            "mode_after": dict(gpu_mode_after),
        },
        "case_policies": policies,
        "cases": [emitted[case_id] for case_id in sorted(emitted)],
    }
    return {**payload, "fragment_sha256": json_sha256(payload)}


def finish_single_arm_session(
    session: SingleArmSession,
    cases: list[dict[str, object]],
) -> dict[str, object]:
    """Close the GPU envelope and atomically publish one immutable result."""

    gpu_mode_after = gpu_mode_snapshot(session.expected_physical_gpu)
    finished_unix_ns = time.time_ns()
    if session.phase == "discover":
        result = _build_family_discovery_result(
            session,
            cases,
            gpu_mode_after=gpu_mode_after,
            finished_unix_ns=finished_unix_ns,
        )
        result_hash_field = "fragment_sha256"
    else:
        if session.contract_path is None:
            raise RuntimeError("final single-arm session has no E2E contract")
        result = build_single_arm_e2e_result(
            family=session.family,
            arm=session.arm,
            sequence_position=session.sequence_position,
            evidence_status=session.evidence_status,
            repo_root=session.repo_root,
            producer_path=session.producer_path,
            source_manifest_path=session.source_manifest_path,
            contract_path=session.contract_path,
            started_unix_ns=session.started_unix_ns,
            finished_unix_ns=finished_unix_ns,
            expected_physical_gpu=session.expected_physical_gpu,
            gpu_mode_before=session.gpu_mode_before,
            gpu_mode_after=gpu_mode_after,
            cases=cases,
        )
        result_hash_field = "result_sha256"
    session.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = session.output_path.with_name(
        f".{session.output_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(session.output_path)
    print(
        json.dumps(
            {
                "arm": session.arm,
                "case_count": len(cases),
                "family": session.family,
                "output": str(session.output_path),
                "phase": session.phase,
                result_hash_field: result[result_hash_field],
                "sequence_position": session.sequence_position,
            },
            sort_keys=True,
        )
    )
    return result
