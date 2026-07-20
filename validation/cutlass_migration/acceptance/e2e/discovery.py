#!/usr/bin/env python3
"""Assemble and explicitly review one offline E2E discovery artifact.

GPU family producers emit one independently hashed fragment.  ``assemble``
requires the closed production-family set for one source side and one physical
GPU, revalidates the producer's raw correctness/graph/allocation evidence,
freezes exact cache files by content, and writes a *pending* discovery.
``review`` repeats every disk/source/harness check before adding a nonempty
review identifier.  Only the reviewed output is accepted by ``contract.py``.

This module is deliberately offline: importing it does not import torch,
CUTLASS, or the GPU producer module that owns the fragment-schema constant.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from validation.cutlass_migration.acceptance.e2e.contract import (
    DISCOVERY_SCHEMA,
    HARNESS_TREES,
    ContractBuildError,
    _actual_file_record,
    _atomic_write_json,
    _exact_keys,
    _is_sha256,
    _load_json,
    _load_source,
    _nonnegative_int,
    _positive_int,
    _registry,
    _registry_sha256,
    _regular_file,
    _require,
    _snapshot_harness,
    _source_records_by_repo_path,
    _validate_case_discovery,
    _validate_discovery,
    _verify_harness_copy,
)
from validation.cutlass_migration.acceptance.e2e.index import (
    PHYSICAL_GPUS,
    POSITION_ARM,
    REQUIRED_FAMILIES,
    EndToEndValidationError,
    _canonical_sha256,
    _validate_gpu_mode_snapshot,
)
from validation.cutlass_migration.paths import REPO_ROOT


# Importing core.single_arm_e2e would import torch/CUTLASS and break the
# assembler's offline contract.  Keep this literal synchronized with the
# producer-exported FAMILY_DISCOVERY_SCHEMA constant.
FAMILY_DISCOVERY_SCHEMA = "sparkinfer.cute.migration.family_discovery.v1"

_FRAGMENT_FIELDS = {
    "schema",
    "family",
    "arm",
    "sequence_position",
    "evidence_status",
    "invocation",
    "source",
    "producer",
    "gpu",
    "case_policies",
    "cases",
    "fragment_sha256",
}
_RAW_CASE_FIELDS = {
    "case_id",
    "case_contract_sha256",
    "input_sha256",
    "artifacts",
    "launch_plan",
    "source_owned_kernel_nodes",
    "correctness",
    "graph",
    "allocation",
    "conditions",
}
_RAW_ARTIFACT_FIELDS = {
    "cache_root",
    "cache_key",
    "manifest_path",
    "manifest_sha256",
    "frontend_ptx_sidecar_path",
    "frontend_ptx_sidecar_sha256",
    "object_path",
    "object_sha256",
    "object_bytes",
    "compile_spec_hash",
    "compile_spec_json",
    "semantic_key",
    "kernel_id",
    "package_fingerprint",
    "toolchain",
    "toolchain_sha256",
    "verification_before",
    "verification_after",
}
_GPU_STABLE_FIELDS = (
    "index",
    "uuid",
    "persistence_mode",
    "compute_mode",
    "power.limit",
)
_PENDING_REVIEW = {
    "status": "pending",
    "review_id": "",
    "reviewed_unix_ns": 0,
}


def _canonical_payload_hash(value: Mapping[str, Any], hash_field: str) -> str:
    return _canonical_sha256(
        {key: item for key, item in value.items() if key != hash_field}
    )


def _load_family_fragment(path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    path = _regular_file(path, location="family discovery fragment")
    value = _exact_keys(_load_json(path), _FRAGMENT_FIELDS, str(path))
    _require(
        value["schema"] == FAMILY_DISCOVERY_SCHEMA,
        f"{path}: unsupported family discovery schema",
    )
    _require(
        value["fragment_sha256"] == _canonical_payload_hash(value, "fragment_sha256"),
        f"{path}: family discovery canonical hash differs",
    )
    return value, _actual_file_record(path, recorded_path=str(path))


def _index_fragments(
    records: Sequence[tuple[dict[str, Any], dict[str, object]]],
) -> dict[str, tuple[dict[str, Any], dict[str, object]]]:
    _require(
        len(records) == len(REQUIRED_FAMILIES),
        "discovery assembly requires exactly one fragment per required family",
    )
    indexed: dict[str, tuple[dict[str, Any], dict[str, object]]] = {}
    for value, artifact in records:
        family = value.get("family")
        _require(
            isinstance(family, str) and family in REQUIRED_FAMILIES,
            f"{artifact['path']}: unknown family {family!r}",
        )
        _require(
            family not in indexed,
            f"discovery assembly has duplicate family {family!r}",
        )
        indexed[family] = (value, artifact)
    _require(
        set(indexed) == set(REQUIRED_FAMILIES),
        "discovery fragment family set is incomplete",
    )
    return indexed


def _prepare_context(
    *, side: str, source_manifest: Path, harness_root: Path
) -> dict[str, Any]:
    _require(side in {"baseline", "current"}, f"invalid source side {side!r}")
    source = _load_source(source_manifest, side)
    harness = _snapshot_harness(harness_root)
    runtime_root = Path(source["manifest"]["runtime"]["repo_root"])
    _verify_harness_copy(
        harness,
        runtime_root=runtime_root,
        location=f"{side} source runtime",
    )
    registry = _registry()
    harness_by_path = {str(record["path"]): record for record in harness["files"]}
    for family, producer in registry.items():
        _require(
            producer in harness_by_path,
            f"{family}: registered producer is absent from the frozen harness",
        )
    return {
        "side": side,
        "source": source,
        "harness": harness,
        "registry": registry,
        "harness_by_path": harness_by_path,
        "source_by_path": _source_records_by_repo_path(source),
    }


def _validate_fragment_gpu(
    value: object,
    *,
    location: str,
    expected_physical_gpu: int,
    started_unix_ns: int,
    finished_unix_ns: int,
) -> dict[str, object]:
    gpu = _exact_keys(
        value,
        {
            "physical_ordinal",
            "name",
            "uuid",
            "capability",
            "mode_before",
            "mode_after",
        },
        location,
    )
    _require(
        gpu["physical_ordinal"] == expected_physical_gpu
        and expected_physical_gpu in PHYSICAL_GPUS,
        f"{location}: physical GPU differs",
    )
    _require(
        isinstance(gpu["name"], str)
        and bool(gpu["name"].strip())
        and isinstance(gpu["uuid"], str)
        and bool(gpu["uuid"].strip())
        and gpu["capability"] == [12, 0],
        f"{location}: expected one named physical SM120 GPU",
    )
    try:
        before = _validate_gpu_mode_snapshot(
            gpu["mode_before"],
            location=f"{location}.mode_before",
            physical_gpu=expected_physical_gpu,
            gpu_uuid=str(gpu["uuid"]),
        )
        after = _validate_gpu_mode_snapshot(
            gpu["mode_after"],
            location=f"{location}.mode_after",
            physical_gpu=expected_physical_gpu,
            gpu_uuid=str(gpu["uuid"]),
        )
    except EndToEndValidationError as exc:
        raise ContractBuildError(str(exc)) from exc
    before_ns = int(before["captured_unix_ns"])
    after_ns = int(after["captured_unix_ns"])
    _require(
        started_unix_ns <= before_ns < after_ns <= finished_unix_ns,
        f"{location}: process/GPU-mode timestamps are not ordered",
    )
    _require(
        all(
            before["fields"][field] == after["fields"][field]
            for field in _GPU_STABLE_FIELDS
        ),
        f"{location}: stable physical-GPU mode changed during discovery",
    )
    return {
        "physical_ordinal": expected_physical_gpu,
        "uuid": gpu["uuid"],
        "name": gpu["name"],
        "capability": gpu["capability"],
    }


def _freeze_file(
    raw_path: object,
    *,
    expected_sha256: object,
    expected_size_bytes: object | None,
    location: str,
) -> dict[str, object]:
    _require(
        isinstance(raw_path, str) and Path(raw_path).is_absolute(),
        f"{location}: path is not absolute",
    )
    path = _regular_file(Path(raw_path), location=location)
    record = _actual_file_record(path, recorded_path=str(path))
    _require(
        _is_sha256(expected_sha256) and record["sha256"] == expected_sha256,
        f"{location}: content SHA differs from raw evidence",
    )
    if expected_size_bytes is not None:
        _nonnegative_int(expected_size_bytes, f"{location}.size_bytes")
        _require(
            record["size_bytes"] == expected_size_bytes,
            f"{location}: byte count differs from raw evidence",
        )
    return record


def _transform_artifact(
    value: object,
    *,
    location: str,
    runtime_package_fingerprint: str,
) -> tuple[dict[str, object], dict[str, object]]:
    binding = _exact_keys(
        value,
        {"role", "kernel_id", "compile_spec_hash", "object_sha256", "evidence"},
        location,
    )
    evidence = _exact_keys(
        binding["evidence"], _RAW_ARTIFACT_FIELDS, f"{location}.evidence"
    )
    role = binding["role"]
    _require(
        isinstance(role, str) and bool(role), f"{location}: artifact role is empty"
    )
    for field in (
        "cache_key",
        "manifest_sha256",
        "frontend_ptx_sidecar_sha256",
        "object_sha256",
        "compile_spec_hash",
        "semantic_key",
        "package_fingerprint",
        "toolchain_sha256",
    ):
        _require(
            _is_sha256(evidence[field]), f"{location}.evidence.{field}: invalid SHA"
        )
    compile_spec_json = evidence["compile_spec_json"]
    try:
        parsed_compile_spec = json.loads(compile_spec_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContractBuildError(
            f"{location}: compile specification is invalid: {exc}"
        ) from exc
    _require(
        isinstance(compile_spec_json, str)
        and bool(compile_spec_json)
        and isinstance(parsed_compile_spec, dict)
        and hashlib.sha256(compile_spec_json.encode()).hexdigest()
        == evidence["compile_spec_hash"],
        f"{location}: compile specification hash differs",
    )
    _require(
        isinstance(evidence["kernel_id"], str)
        and bool(evidence["kernel_id"])
        and isinstance(evidence["toolchain"], (dict, list))
        and bool(evidence["toolchain"]),
        f"{location}: kernel/toolchain identity is empty",
    )
    _positive_int(evidence["object_bytes"], f"{location}.evidence.object_bytes")
    _require(
        _canonical_sha256(evidence["toolchain"]) == evidence["toolchain_sha256"],
        f"{location}: toolchain hash differs",
    )
    _require(
        evidence["package_fingerprint"] == runtime_package_fingerprint,
        f"{location}: artifact/source package fingerprints differ",
    )
    _require(
        all(
            binding[field] == evidence[field]
            for field in ("kernel_id", "compile_spec_hash", "object_sha256")
        ),
        f"{location}: artifact binding differs from its raw evidence",
    )

    cache_root_raw = evidence["cache_root"]
    _require(
        isinstance(cache_root_raw, str) and Path(cache_root_raw).is_absolute(),
        f"{location}: cache root is not absolute",
    )
    cache_root = Path(cache_root_raw).resolve()
    _require(
        cache_root.is_dir() and not Path(cache_root_raw).is_symlink(),
        f"{location}: cache root is not a regular directory",
    )
    cache_key = str(evidence["cache_key"])
    expected_manifest = cache_root / cache_key[:2] / f"{cache_key}.json"
    _require(
        Path(str(evidence["manifest_path"])).resolve() == expected_manifest,
        f"{location}: manifest does not occupy its exact cache-key path",
    )
    manifest = _freeze_file(
        evidence["manifest_path"],
        expected_sha256=evidence["manifest_sha256"],
        expected_size_bytes=None,
        location=f"{location}.manifest",
    )
    raw_manifest = _load_json(Path(str(manifest["path"])))
    for field in (
        "cache_key",
        "object_sha256",
        "object_bytes",
        "compile_spec_hash",
        "compile_spec_json",
        "semantic_key",
        "kernel_id",
        "package_fingerprint",
        "toolchain",
    ):
        _require(
            evidence[field] == raw_manifest.get(field),
            f"{location}: raw evidence {field} differs from the frozen manifest",
        )
    obj = _freeze_file(
        evidence["object_path"],
        expected_sha256=evidence["object_sha256"],
        expected_size_bytes=evidence["object_bytes"],
        location=f"{location}.object",
    )
    sidecar = _freeze_file(
        evidence["frontend_ptx_sidecar_path"],
        expected_sha256=evidence["frontend_ptx_sidecar_sha256"],
        expected_size_bytes=None,
        location=f"{location}.frontend_ptx_sidecar",
    )
    # contract._validate_cache_artifact validates the complete quartet.  Check
    # the otherwise-unreferenced raw frontend PTX is immutable and present too.
    _regular_file(
        Path(str(evidence["object_path"])).with_suffix(".ptx"),
        location=f"{location}.frontend_ptx",
    )
    expected_verification = {
        "passed": True,
        "manifest_sha256": evidence["manifest_sha256"],
        "object_sha256": evidence["object_sha256"],
        "object_bytes": evidence["object_bytes"],
    }
    for phase in ("verification_before", "verification_after"):
        verification = _exact_keys(
            evidence[phase], set(expected_verification), f"{location}.evidence.{phase}"
        )
        _require(
            verification == expected_verification,
            f"{location}: exact artifact {phase} differs",
        )
    transformed = {
        "role": role,
        "manifest": manifest,
        "object": obj,
        "frontend_ptx_sidecar": sidecar,
    }
    raw_identity = {
        "role": role,
        "kernel_id": evidence["kernel_id"],
        "compile_spec_hash": evidence["compile_spec_hash"],
        "compile_spec_json": compile_spec_json,
        "manifest_sha256": evidence["manifest_sha256"],
        "object_sha256": evidence["object_sha256"],
        "frontend_ptx_sidecar_sha256": evidence["frontend_ptx_sidecar_sha256"],
    }
    return transformed, raw_identity


def _validate_correctness(
    value: object, *, location: str, required_gates: Sequence[str]
) -> None:
    _require(isinstance(value, dict), f"{location}: expected an object")
    gates = value.get("gates")
    _require(
        value.get("independent_oracle") is True
        and isinstance(value.get("oracle"), str)
        and bool(str(value["oracle"]).strip())
        and "arm" not in str(value["oracle"]).lower()
        and "equal" not in str(value["oracle"]).lower()
        and value.get("passed") is True
        and value.get("finite") is True
        and isinstance(value.get("nonzero_count"), int)
        and not isinstance(value.get("nonzero_count"), bool)
        and int(value["nonzero_count"]) > 0
        and value.get("read_only_inputs_immutable") is True
        and _is_sha256(value.get("read_only_inputs_sha256"))
        and _is_sha256(value.get("output_sha256"))
        and isinstance(gates, dict)
        and set(gates) == set(required_gates)
        and all(item is True for item in gates.values()),
        f"{location}: correctness/oracle/input gates did not pass",
    )


def _validate_allocation(value: object, *, location: str) -> None:
    _require(isinstance(value, dict), f"{location}: expected an object")
    _require(
        all(
            value.get(field) is True
            for field in (
                "fixed_workspace_capacity",
                "stable_addresses",
                "allocator_stable",
                "zero_replay_allocations",
            )
        ),
        f"{location}: allocation/serving gates did not pass",
    )
    _nonnegative_int(
        value.get("workspace_capacity_bytes"), f"{location}.workspace_capacity_bytes"
    )
    counters = value.get("condition_counters")
    _require(
        isinstance(counters, dict) and set(counters) == {"warm_l2", "cold_l2"},
        f"{location}: allocation condition coverage differs",
    )
    expected_fields = {
        "allocated_bytes_before",
        "allocated_bytes_after",
        "reserved_bytes_before",
        "reserved_bytes_after",
    }
    for condition in ("warm_l2", "cold_l2"):
        record = _exact_keys(
            counters[condition],
            expected_fields,
            f"{location}.condition_counters.{condition}",
        )
        for field in expected_fields:
            _nonnegative_int(
                record[field], f"{location}.condition_counters.{condition}.{field}"
            )
        _require(
            record["allocated_bytes_before"] == record["allocated_bytes_after"]
            and record["reserved_bytes_before"] == record["reserved_bytes_after"],
            f"{location}: {condition} replay changed allocator counters",
        )
    _require(
        all(
            value.get(field) == counters["warm_l2"][field] for field in expected_fields
        ),
        f"{location}: summary allocation counters differ from warm-L2 evidence",
    )


def _transform_case(
    value: object,
    *,
    policy: Mapping[str, Any],
    location: str,
    family: str,
    side: str,
    runtime_package_fingerprint: str,
    source_by_path: Mapping[str, Mapping[str, Any]],
    harness_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    case = _exact_keys(value, _RAW_CASE_FIELDS, location)
    case_id = policy["case_id"]
    _require(
        case["case_id"] == case_id and case["input_sha256"] == policy["input_sha256"],
        f"{location}: raw case identity differs from its policy",
    )
    identity = {
        "family": family,
        "case_id": case_id,
        "input_sha256": policy["input_sha256"],
        "required_correctness_gates": policy["required_correctness_gates"],
        "cross_arm_output_policy": policy["cross_arm_output_policy"],
        "topology_review": policy["topology_review"],
    }
    _require(
        case["case_contract_sha256"] == _canonical_sha256(identity),
        f"{location}: producer-owned discovery case hash differs",
    )
    _validate_correctness(
        case["correctness"],
        location=f"{location}.correctness",
        required_gates=policy["required_correctness_gates"],
    )
    _validate_allocation(case["allocation"], location=f"{location}.allocation")
    _require(
        isinstance(case["conditions"], dict)
        and set(case["conditions"]) == {"warm_l2", "cold_l2"},
        f"{location}: warm/cold discovery conditions are incomplete",
    )
    graph = case["graph"]
    _require(isinstance(graph, dict), f"{location}.graph: expected an object")
    _require(
        all(
            graph.get(field) is True
            for field in (
                "capture_passed",
                "replay_passed",
                "topology_stable",
                "addresses_stable",
                "live_input_changed_output",
                "poison_overwrite_passed",
            )
        ),
        f"{location}: graph serving gates did not pass",
    )
    graph_contract = {
        field: graph.get(field)
        for field in ("node_count", "kernel_node_count", "nodes")
    }

    raw_artifacts = case["artifacts"]
    _require(
        isinstance(raw_artifacts, list) and raw_artifacts, f"{location}: no artifacts"
    )
    transformed_artifacts: list[dict[str, object]] = []
    raw_identities: dict[str, dict[str, object]] = {}
    for index, artifact in enumerate(raw_artifacts):
        transformed, raw_identity = _transform_artifact(
            artifact,
            location=f"{location}.artifacts[{index}]",
            runtime_package_fingerprint=runtime_package_fingerprint,
        )
        role = str(transformed["role"])
        _require(
            role not in raw_identities, f"{location}: duplicate artifact role {role!r}"
        )
        transformed_artifacts.append(transformed)
        raw_identities[role] = raw_identity

    raw_plan = case["launch_plan"]
    _require(isinstance(raw_plan, list) and raw_plan, f"{location}: empty launch plan")
    launch_plan: list[dict[str, object]] = []
    expected_plan: list[dict[str, object]] = []
    for index, raw_binding in enumerate(raw_plan):
        binding = _exact_keys(
            raw_binding,
            {
                "node_index",
                "artifact_role",
                "kernel_id",
                "compile_spec_hash",
                "object_sha256",
                "multiplicity_index",
            },
            f"{location}.launch_plan[{index}]",
        )
        role = binding["artifact_role"]
        raw_identity = raw_identities.get(str(role))
        _require(
            raw_identity is not None
            and binding["kernel_id"] == raw_identity["kernel_id"]
            and binding["compile_spec_hash"] == raw_identity["compile_spec_hash"]
            and binding["object_sha256"] == raw_identity["object_sha256"],
            f"{location}.launch_plan[{index}]: raw artifact identity differs",
        )
        launch_plan.append(
            {
                "node_index": binding["node_index"],
                "artifact_role": role,
                "multiplicity_index": binding["multiplicity_index"],
            }
        )
        expected_plan.append(
            {
                "node_index": binding["node_index"],
                "artifact_role": role,
                "kernel_id": binding["kernel_id"],
                "compile_spec_hash": binding["compile_spec_hash"],
                "multiplicity_index": binding["multiplicity_index"],
            }
        )

    transformed = {
        "case_id": case_id,
        "input_sha256": policy["input_sha256"],
        "required_correctness_gates": policy["required_correctness_gates"],
        "cross_arm_output_policy": policy["cross_arm_output_policy"],
        "topology_review": policy["topology_review"],
        "graph": graph_contract,
        "artifacts": transformed_artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": case["source_owned_kernel_nodes"],
    }
    normalized = _validate_case_discovery(
        transformed,
        location=f"{location}.frozen",
        side=side,
        runtime_package_fingerprint=runtime_package_fingerprint,
        source_by_path=source_by_path,
        harness_by_path=harness_by_path,
    )
    expected_artifacts = sorted(
        [
            {
                key: identity[key]
                for key in (
                    "role",
                    "kernel_id",
                    "compile_spec_hash",
                    "compile_spec_json",
                )
            }
            for identity in raw_identities.values()
        ],
        key=lambda item: str(item["role"]),
    )
    _require(
        normalized["compile_contract"]["artifacts"] == expected_artifacts
        and normalized["compile_contract"]["launch_plan"] == expected_plan,
        f"{location}: raw and manifest-derived compile identities differ",
    )
    expected_consensus = sorted(
        [
            {
                "role": identity["role"],
                "manifest_sha256": identity["manifest_sha256"],
                "object_sha256": identity["object_sha256"],
                "frontend_ptx_sidecar_sha256": identity["frontend_ptx_sidecar_sha256"],
            }
            for identity in raw_identities.values()
        ],
        key=lambda item: str(item["role"]),
    )
    _require(
        normalized["artifact_consensus"] == expected_consensus,
        f"{location}: raw and immutable artifact identities differ",
    )
    return transformed


def _validate_policy(value: object, *, location: str) -> dict[str, Any]:
    policy = _exact_keys(
        value,
        {
            "case_id",
            "input_sha256",
            "required_correctness_gates",
            "cross_arm_output_policy",
            "topology_review",
        },
        location,
    )
    _require(
        isinstance(policy["case_id"], str)
        and bool(policy["case_id"])
        and _is_sha256(policy["input_sha256"]),
        f"{location}: invalid case identity",
    )
    gates = policy["required_correctness_gates"]
    _require(
        isinstance(gates, list)
        and bool(gates)
        and gates == sorted(set(gates))
        and all(isinstance(gate, str) and bool(gate) for gate in gates),
        f"{location}: correctness gates are not sorted/unique/nonempty",
    )
    _require(
        policy["cross_arm_output_policy"] in {"bit-exact", "oracle-only"},
        f"{location}: invalid cross-arm output policy",
    )
    review = _exact_keys(
        policy["topology_review"],
        {"disposition", "reason"},
        f"{location}.topology_review",
    )
    _require(
        review["disposition"] in {"equal", "changed-reviewed"}
        and isinstance(review["reason"], str)
        and (
            (review["disposition"] == "equal" and review["reason"] == "")
            or (
                review["disposition"] == "changed-reviewed"
                and bool(review["reason"].strip())
            )
        ),
        f"{location}: topology review is inconsistent",
    )
    return policy


def _validate_family_fragment(
    fragment: Mapping[str, Any],
    *,
    artifact_path: str,
    family: str,
    expected_physical_gpu: int,
    context: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    location = artifact_path
    side = str(context["side"])
    _require(fragment["family"] == family, f"{location}: family differs")
    _require(fragment["arm"] == side, f"{location}: source side differs")
    position = fragment["sequence_position"]
    _require(
        isinstance(position, str) and POSITION_ARM.get(position) == side,
        f"{location}: sequence position does not belong to {side}",
    )
    _require(
        fragment["evidence_status"] == "final-source",
        f"{location}: discovery is not final-source evidence",
    )
    invocation = _exact_keys(
        fragment["invocation"],
        {"pid", "started_unix_ns", "finished_unix_ns"},
        f"{location}.invocation",
    )
    _positive_int(invocation["pid"], f"{location}.invocation.pid")
    started = _positive_int(
        invocation["started_unix_ns"], f"{location}.invocation.started_unix_ns"
    )
    finished = _positive_int(
        invocation["finished_unix_ns"], f"{location}.invocation.finished_unix_ns"
    )
    _require(finished > started, f"{location}: invocation timestamps are not ordered")

    source_binding = _exact_keys(
        fragment["source"],
        {"manifest_sha256", "manifest_artifact_sha256", "runtime_package_fingerprint"},
        f"{location}.source",
    )
    source = context["source"]
    expected_source = {
        key: source[key]
        for key in (
            "manifest_sha256",
            "manifest_artifact_sha256",
            "runtime_package_fingerprint",
        )
    }
    _require(
        source_binding == expected_source,
        f"{location}: fragment is not bound to the selected frozen source",
    )
    producer = _exact_keys(
        fragment["producer"], {"path", "sha256"}, f"{location}.producer"
    )
    expected_producer = context["registry"][family]
    producer_harness = context["harness_by_path"][expected_producer]
    _require(
        producer == {"path": expected_producer, "sha256": producer_harness["sha256"]},
        f"{location}: producer differs from the frozen registry/harness",
    )
    gpu = _validate_fragment_gpu(
        fragment["gpu"],
        location=f"{location}.gpu",
        expected_physical_gpu=expected_physical_gpu,
        started_unix_ns=started,
        finished_unix_ns=finished,
    )

    raw_policies = fragment["case_policies"]
    raw_cases = fragment["cases"]
    _require(
        isinstance(raw_policies, list)
        and raw_policies
        and isinstance(raw_cases, list)
        and raw_cases,
        f"{location}: family discovery has no cases",
    )
    policies = [
        _validate_policy(item, location=f"{location}.case_policies[{index}]")
        for index, item in enumerate(raw_policies)
    ]
    policy_ids = [str(policy["case_id"]) for policy in policies]
    _require(
        policy_ids == sorted(set(policy_ids)),
        f"{location}: case policies are not sorted/unique",
    )
    raw_by_id = {
        str(case.get("case_id")): case for case in raw_cases if isinstance(case, dict)
    }
    _require(
        len(raw_by_id) == len(raw_cases) and list(raw_by_id) == policy_ids,
        f"{location}: raw cases and policies are not the same sorted set",
    )
    cases = [
        _transform_case(
            raw_by_id[case_id],
            policy=policies[index],
            location=f"{location}.cases[{index}]",
            family=family,
            side=side,
            runtime_package_fingerprint=str(source["runtime_package_fingerprint"]),
            source_by_path=context["source_by_path"],
            harness_by_path=context["harness_by_path"],
        )
        for index, case_id in enumerate(policy_ids)
    ]
    return {
        "producer": expected_producer,
        "producer_sha256": producer_harness["sha256"],
        "cases": cases,
    }, gpu


def _write_temporary_json(value: Mapping[str, Any]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix="sparkinfer-discovery-", suffix=".json")
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, ensure_ascii=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _require_pending_review(value: Mapping[str, Any], *, location: str) -> None:
    review = _exact_keys(
        value.get("review"),
        {"status", "review_id", "reviewed_unix_ns"},
        f"{location}.review",
    )
    _require(review == _PENDING_REVIEW, f"{location}: discovery is not pending review")
    _require(
        value.get("discovery_sha256")
        == _canonical_payload_hash(value, "discovery_sha256"),
        f"{location}: pending discovery canonical hash differs",
    )


def _reviewed_value(
    value: Mapping[str, Any], review_id: str, reviewed_unix_ns: int
) -> dict[str, Any]:
    _require_pending_review(value, location="pending discovery")
    _require(
        isinstance(review_id, str) and bool(review_id.strip()),
        "review ID must be explicitly nonempty",
    )
    _positive_int(reviewed_unix_ns, "reviewed_unix_ns")
    reviewed = copy.deepcopy(dict(value))
    reviewed["review"] = {
        "status": "reviewed",
        "review_id": review_id.strip(),
        "reviewed_unix_ns": reviewed_unix_ns,
    }
    reviewed["discovery_sha256"] = _canonical_payload_hash(reviewed, "discovery_sha256")
    return reviewed


def _deep_validate_reviewed(
    value: Mapping[str, Any], *, context: Mapping[str, Any]
) -> None:
    temporary = _write_temporary_json(value)
    try:
        _validate_discovery(
            temporary,
            side=str(context["side"]),
            source=context["source"],
            harness=context["harness"],
            registry=context["registry"],
        )
    finally:
        temporary.unlink(missing_ok=True)


def _deep_validate_pending(
    value: Mapping[str, Any], *, context: Mapping[str, Any]
) -> None:
    _require_pending_review(value, location="pending discovery")
    synthetic = _reviewed_value(value, "offline-structural-validation", 1)
    _deep_validate_reviewed(synthetic, context=context)


def assemble_discovery(
    *,
    side: str,
    source_manifest: Path,
    harness_root: Path,
    expected_physical_gpu: int,
    fragments: Sequence[Path],
) -> dict[str, object]:
    """Return one fully validated pending discovery for a source/GPU pair."""

    context = _prepare_context(
        side=side,
        source_manifest=source_manifest,
        harness_root=harness_root,
    )
    indexed = _index_fragments([_load_family_fragment(path) for path in fragments])
    families: dict[str, object] = {}
    fragment_records: list[dict[str, object]] = []
    expected_gpu: dict[str, object] | None = None
    for family in REQUIRED_FAMILIES:
        fragment, fragment_artifact = indexed[family]
        family_record, gpu = _validate_family_fragment(
            fragment,
            artifact_path=str(fragment_artifact["path"]),
            family=family,
            expected_physical_gpu=expected_physical_gpu,
            context=context,
        )
        if expected_gpu is None:
            expected_gpu = gpu
        _require(
            gpu == expected_gpu,
            f"{family}: physical-GPU identity differs across fragments",
        )
        families[family] = family_record
        fragment_records.append(
            {
                "family": family,
                "fragment_sha256": fragment["fragment_sha256"],
                "artifact_sha256": fragment_artifact["sha256"],
                "artifact_size_bytes": fragment_artifact["size_bytes"],
            }
        )
    _require(expected_gpu is not None, "discovery has no GPU identity")
    source = context["source"]
    source_binding = {
        key: source[key]
        for key in (
            "manifest_sha256",
            "manifest_artifact_sha256",
            "runtime_package_fingerprint",
        )
    }
    discovery_id = _canonical_sha256(
        {
            "schema": FAMILY_DISCOVERY_SCHEMA,
            "side": side,
            "source": source_binding,
            "harness_tree_fingerprint": context["harness"]["tree_fingerprint"],
            "producer_registry_sha256": _registry_sha256(context["registry"]),
            "gpu": expected_gpu,
            "fragments": fragment_records,
        }
    )
    payload: dict[str, object] = {
        "schema": DISCOVERY_SCHEMA,
        "side": side,
        "discovery_id": discovery_id,
        "review": dict(_PENDING_REVIEW),
        "source": source_binding,
        "harness_tree_fingerprint": context["harness"]["tree_fingerprint"],
        "producer_registry_sha256": _registry_sha256(context["registry"]),
        "gpu": expected_gpu,
        "families": families,
    }
    discovery = {**payload, "discovery_sha256": _canonical_sha256(payload)}
    _deep_validate_pending(discovery, context=context)
    return discovery


def review_discovery(
    *,
    pending_path: Path,
    source_manifest: Path,
    harness_root: Path,
    review_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate one pending discovery and return its reviewed transition."""

    pending_path = _regular_file(pending_path, location="pending discovery")
    pending = _load_json(pending_path)
    side = pending.get("side")
    _require(side in {"baseline", "current"}, "pending discovery has an invalid side")
    context = _prepare_context(
        side=str(side),
        source_manifest=source_manifest,
        harness_root=harness_root,
    )
    _deep_validate_pending(pending, context=context)
    reviewed = _reviewed_value(pending, review_id, time.time_ns())
    _deep_validate_reviewed(reviewed, context=context)
    return reviewed, context


def _output_outside_harness(path: Path, harness_root: Path) -> None:
    resolved = path.resolve()
    root = harness_root.resolve()
    for tree_name in HARNESS_TREES:
        tree = root / tree_name
        _require(
            resolved != tree and tree not in resolved.parents,
            "discovery output must be outside the snapshotted harness trees",
        )


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="sparkinfer-discovery-selftest-") as raw_root:
        root = Path(raw_root)
        paths: list[Path] = []
        for index, family in enumerate(REQUIRED_FAMILIES):
            payload = {
                "schema": FAMILY_DISCOVERY_SCHEMA,
                "family": family,
                "arm": "current",
                "sequence_position": "b1",
                "evidence_status": "final-source",
                "invocation": {},
                "source": {},
                "producer": {},
                "gpu": {},
                "case_policies": [],
                "cases": [],
                "self_test_index": index,
            }
            # Remove the self-test-only discriminator while retaining unique
            # canonical payloads through the family field.
            payload.pop("self_test_index")
            value = {**payload, "fragment_sha256": _canonical_sha256(payload)}
            path = root / f"{family}.json"
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            paths.append(path)

        records = [_load_family_fragment(path) for path in paths]
        _index_fragments(records)

        missing_duplicate = [*records[:-1], records[0]]
        try:
            _index_fragments(missing_duplicate)
        except ContractBuildError:
            pass
        else:
            raise AssertionError("duplicate/missing family mutation passed")

        tampered = _load_json(paths[0])
        tampered["arm"] = "baseline"
        paths[0].write_text(
            json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            _load_family_fragment(paths[0])
        except ContractBuildError:
            pass
        else:
            raise AssertionError("fragment hash mutation passed")

        pending_payload = {"review": dict(_PENDING_REVIEW), "probe": "positive"}
        pending = {
            **pending_payload,
            "discovery_sha256": _canonical_sha256(pending_payload),
        }
        reviewed = _reviewed_value(pending, "self-test-review", 1)
        _require(
            reviewed["review"]
            == {
                "status": "reviewed",
                "review_id": "self-test-review",
                "reviewed_unix_ns": 1,
            },
            "positive review transition failed",
        )
        for bad_id in ("", "   "):
            try:
                _reviewed_value(pending, bad_id, 1)
            except ContractBuildError:
                pass
            else:
                raise AssertionError("empty review ID mutation passed")
        tampered_pending = {**pending, "probe": "tampered"}
        try:
            _reviewed_value(tampered_pending, "review", 1)
        except ContractBuildError:
            pass
        else:
            raise AssertionError("pending hash mutation passed")


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    assemble = subparsers.add_parser("assemble", help="emit a pending discovery")
    assemble.add_argument("--side", choices=("baseline", "current"), required=True)
    assemble.add_argument("--source-manifest", type=Path, required=True)
    assemble.add_argument("--harness-root", type=Path, default=REPO_ROOT)
    assemble.add_argument(
        "--expected-physical-gpu", type=int, choices=PHYSICAL_GPUS, required=True
    )
    assemble.add_argument("--fragment", type=Path, action="append", required=True)
    assemble.add_argument("--output", type=Path, required=True)

    review = subparsers.add_parser(
        "review", help="stamp an explicitly reviewed discovery"
    )
    review.add_argument("--input", type=Path, required=True)
    review.add_argument("--source-manifest", type=Path, required=True)
    review.add_argument("--harness-root", type=Path, default=REPO_ROOT)
    review.add_argument("--review-id", required=True)
    review.add_argument("--output", type=Path, required=True)

    subparsers.add_parser(
        "self-test", help="run offline positive/negative integrity probes"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    started = time.monotonic()
    try:
        if args.operation == "self-test":
            _self_test()
            print(
                "status=pass discovery_self_test=positive_and_negative", file=sys.stderr
            )
            return 0
        harness_root = args.harness_root.resolve()
        output = args.output.resolve()
        _output_outside_harness(output, harness_root)
        if args.operation == "assemble":
            discovery = assemble_discovery(
                side=args.side,
                source_manifest=args.source_manifest,
                harness_root=harness_root,
                expected_physical_gpu=args.expected_physical_gpu,
                fragments=args.fragment,
            )
        else:
            pending = args.input.resolve()
            _require(
                pending != output,
                "review output must not overwrite the pending artifact",
            )
            discovery, context = review_discovery(
                pending_path=pending,
                source_manifest=args.source_manifest,
                harness_root=harness_root,
                review_id=args.review_id,
            )
        _atomic_write_json(output, discovery)
        _require(
            _load_json(output) == discovery,
            "published discovery differs from the validated in-memory value",
        )
        if args.operation == "review":
            _validate_discovery(
                output,
                side=str(discovery["side"]),
                source=context["source"],
                harness=context["harness"],
                registry=context["registry"],
            )
        else:
            _require_pending_review(discovery, location=str(output))
    except (
        ContractBuildError,
        EndToEndValidationError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        print(f"discovery {args.operation} failed: {exc}", file=sys.stderr)
        return 1
    print(
        "status=pass "
        f"operation={args.operation} "
        f"review_status={discovery['review']['status']} "
        f"discovery_sha256={discovery['discovery_sha256']} "
        f"elapsed_seconds={time.monotonic() - started:.3f} "
        f"output={output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
