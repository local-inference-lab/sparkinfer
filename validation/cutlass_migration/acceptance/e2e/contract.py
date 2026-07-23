#!/usr/bin/env python3
"""Build the immutable CUTLASS migration end-to-end contract.

The final one-arm producers require a reviewed contract before they may load a
cache object or capture a graph.  Consequently, contract construction cannot
consume a final process result: that would make the evidence dependency
circular.  This builder instead consumes four independently hashed discovery
artifacts (baseline/current on physical GPUs 4 and 5).  Discovery records carry
the complete CUDA-graph node list and role mapping, while exact compile
identities are derived here from the referenced cache manifest, object, and
frontend-PTX sidecar rather than copied by hand.

The builder is deliberately offline.  It imports neither torch nor CUTLASS.
It snapshots the complete ``benchmarks``, ``tests``, and ``validation`` trees, binds that
harness to both runtime worktrees, verifies both immutable production-source
manifests against disk, requires cross-GPU discovery consensus, and finally
validates its own v3 output with the release-index contract validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from validation.cutlass_migration.acceptance.corpus.ptx_capture import (
    _validate_bound_artifacts,
)
from validation.cutlass_migration.acceptance.e2e.index import (
    CONTRACT_SCHEMA,
    EXPECTED_PACKAGES,
    PHYSICAL_GPUS,
    PRODUCTION_SOURCE_SCHEMA,
    REQUIRED_FAMILIES,
    EndToEndValidationError,
    _canonical_sha256,
    _sha256_file,
    _validate_compile_side_contract,
    _validate_contract,
    _validate_source_manifest,
    _validate_source_owned_kernel_nodes,
)
from validation.cutlass_migration.acceptance.e2e.readiness import REGISTRY
from validation.cutlass_migration.paths import REPO_ROOT


_REPO_ROOT = REPO_ROOT


DISCOVERY_SCHEMA = "sparkinfer.cute.migration.end_to_end_contract_discovery.v1"
COMPILE_MANIFEST_SCHEMA = "sparkinfer.cute.compile_manifest.v3"
FRONTEND_PTX_SCHEMA = "sparkinfer.cute.frontend_ptx.v3"
KERNEL_NODE_TYPE = "CU_GRAPH_NODE_TYPE_KERNEL"
HARNESS_TREES = ("benchmarks", "tests", "validation")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_ROLE_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_CUTLASS_TOOLCHAIN_NAMES = {
    "nvidia-cutlass-dsl": "cutlass_dsl",
    "nvidia-cutlass-dsl-libs-base": "cutlass_dsl_libs_base",
    "nvidia-cutlass-dsl-libs-core": "cutlass_dsl_libs_core",
    "nvidia-cutlass-dsl-libs-cu12": "cutlass_dsl_libs_cu12",
    "nvidia-cutlass-dsl-libs-cu13": "cutlass_dsl_libs_cu13",
}


class ContractBuildError(RuntimeError):
    """A discovery/source/harness invariant is not admissible."""


def _fail(message: str) -> None:
    raise ContractBuildError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _exact_keys(value: object, expected: set[str], location: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{location}: expected an object")
    actual = set(value)
    _require(
        actual == expected,
        f"{location}: fields differ; missing={sorted(expected - actual)!r}, "
        f"unexpected={sorted(actual - expected)!r}",
    )
    return value


def _positive_int(value: object, location: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{location}: expected a positive integer",
    )
    return int(value)


def _nonnegative_int(value: object, location: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{location}: expected a nonnegative integer",
    )
    return int(value)


def _load_json(path: Path, *, location: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSON {location or path}: {exc}")
    _require(isinstance(value, dict), f"{location or path}: expected a JSON object")
    return value


def _regular_file(path: Path, *, location: str) -> Path:
    resolved = path.resolve()
    _require(
        path.is_file() and not path.is_symlink(),
        f"{location}: not an immutable regular file: {path}",
    )
    return resolved


def _actual_file_record(path: Path, *, recorded_path: str) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": recorded_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _validate_file_reference(value: object, *, location: str) -> dict[str, object]:
    record = _exact_keys(value, {"path", "sha256", "size_bytes"}, location)
    raw_path = record["path"]
    _require(
        isinstance(raw_path, str) and Path(raw_path).is_absolute(),
        f"{location}.path: cache evidence path must be absolute",
    )
    _require(_is_sha256(record["sha256"]), f"{location}.sha256: invalid SHA")
    _nonnegative_int(record["size_bytes"], f"{location}.size_bytes")
    path = _regular_file(Path(raw_path), location=location)
    observed = _actual_file_record(path, recorded_path=str(path))
    expected = {**record, "path": str(Path(raw_path).resolve())}
    _require(observed == expected, f"{location}: immutable file identity differs")
    return expected


def _iter_tree_files(root: Path) -> list[Path]:
    _require(root.is_dir() and not root.is_symlink(), f"not a regular tree: {root}")
    paths: list[Path] = []
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"tree contains a symlink: {path}")
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        paths.append(path)
    paths.sort(key=lambda path: path.relative_to(root).as_posix())
    return paths


def _package_snapshot(repo_root: Path) -> dict[str, object]:
    package_root = repo_root / "sparkinfer"
    paths = _iter_tree_files(package_root)
    _require(paths, f"empty sparkinfer package tree: {package_root}")
    digest = hashlib.sha256()
    records: list[dict[str, object]] = []
    for path in paths:
        relative = path.relative_to(package_root).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    return {
        "root": "sparkinfer",
        "fingerprint": digest.hexdigest(),
        "records_sha256": _canonical_sha256(records),
        "file_count": len(records),
        "files": records,
    }


def _verify_source_endpoint(endpoint: Mapping[str, Any], *, location: str) -> None:
    repo_root = Path(str(endpoint["repo_root"])).resolve()
    observed = _package_snapshot(repo_root)
    _require(
        observed == endpoint["sparkinfer_package"],
        f"{location}: sparkinfer tree changed after its source manifest was frozen",
    )


def _load_source(path: Path, side: str) -> dict[str, Any]:
    path = _regular_file(path, location=f"{side} source manifest")
    try:
        validated = _validate_source_manifest(path, side)
    except EndToEndValidationError as exc:
        raise ContractBuildError(str(exc)) from exc
    manifest = validated["manifest"]
    _require(
        manifest["schema"] == PRODUCTION_SOURCE_SCHEMA,
        f"{path}: unsupported source schema",
    )
    _verify_source_endpoint(manifest["production"], location=f"{path}.production")
    _verify_source_endpoint(manifest["runtime"], location=f"{path}.runtime")
    return validated


def _snapshot_harness(harness_root: Path) -> dict[str, object]:
    harness_root = harness_root.resolve()
    records: list[dict[str, object]] = []
    for tree_name in HARNESS_TREES:
        tree = harness_root / tree_name
        paths = _iter_tree_files(tree)
        _require(paths, f"harness tree is empty: {tree}")
        for path in paths:
            records.append(
                _actual_file_record(
                    path,
                    recorded_path=path.relative_to(harness_root).as_posix(),
                )
            )
    records.sort(key=lambda record: str(record["path"]))
    paths = [str(record["path"]) for record in records]
    _require(paths == sorted(set(paths)), "harness paths are not unique")
    return {
        "files": records,
        "file_count": len(records),
        "tree_fingerprint": _canonical_sha256(records),
    }


def _verify_harness_copy(
    harness: Mapping[str, Any], *, runtime_root: Path, location: str
) -> None:
    observed = _snapshot_harness(runtime_root)
    _require(observed == harness, f"{location}: runtime harness differs from review")


def _registry() -> dict[str, str]:
    _require(
        set(REGISTRY) == set(REQUIRED_FAMILIES),
        "single-arm producer registry differs from the closed family set",
    )
    result: dict[str, str] = {}
    for family in REQUIRED_FAMILIES:
        producer = REGISTRY[family].single_arm
        _require(producer is not None, f"{family}: no single-arm producer is registered")
        path = Path(producer)
        _require(
            not path.is_absolute()
            and path.as_posix() == producer
            and ".." not in path.parts,
            f"{family}: producer path is not normalized repo-relative",
        )
        result[family] = producer
    return result


def _registry_sha256(registry: Mapping[str, str]) -> str:
    return _canonical_sha256(
        [{"family": family, "producer": registry[family]} for family in REQUIRED_FAMILIES]
    )


def _toolchain_packages(toolchain: object, *, location: str) -> dict[str, str]:
    _require(isinstance(toolchain, list) and toolchain, f"{location}: empty toolchain")
    entries: dict[str, object] = {}
    for index, raw_entry in enumerate(toolchain):
        _require(
            isinstance(raw_entry, list) and len(raw_entry) >= 2,
            f"{location}[{index}]: malformed toolchain entry",
        )
        name = raw_entry[0]
        _require(
            isinstance(name, str) and name and name not in entries,
            f"{location}[{index}]: duplicate/invalid toolchain name",
        )
        entries[name] = raw_entry[1]
    packages: dict[str, str] = {}
    for package, manifest_name in _CUTLASS_TOOLCHAIN_NAMES.items():
        value = entries.get(manifest_name, "missing")
        _require(
            isinstance(value, str) and value,
            f"{location}: invalid {manifest_name} version",
        )
        packages[package] = value
    return packages


def _validate_cache_artifact(
    value: object,
    *,
    location: str,
    side: str,
    runtime_package_fingerprint: str,
) -> dict[str, Any]:
    record = _exact_keys(
        value,
        {"role", "manifest", "object", "frontend_ptx_sidecar"},
        location,
    )
    role = record["role"]
    _require(
        isinstance(role, str) and bool(_ROLE_RE.fullmatch(role)),
        f"{location}.role: invalid artifact role",
    )
    manifest_ref = _validate_file_reference(
        record["manifest"], location=f"{location}.manifest"
    )
    object_ref = _validate_file_reference(
        record["object"], location=f"{location}.object"
    )
    ptx_ref = _validate_file_reference(
        record["frontend_ptx_sidecar"],
        location=f"{location}.frontend_ptx_sidecar",
    )
    manifest_path = Path(str(manifest_ref["path"]))
    object_path = Path(str(object_ref["path"]))
    ptx_path = Path(str(ptx_ref["path"]))
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("schema") == COMPILE_MANIFEST_SCHEMA,
        f"{location}: unsupported compile manifest schema",
    )
    _require(
        manifest.get("cache_format")
        == "sparkinfer_cute_compile_cache_v6_explicit_spec",
        f"{location}: exact explicit compile specification is required",
    )
    cache_key = manifest.get("cache_key")
    _require(_is_sha256(cache_key), f"{location}: invalid cache key")
    _require(
        manifest_path.name == f"{cache_key}.json"
        and manifest_path.parent.name == str(cache_key)[:2]
        and object_path == manifest_path.with_suffix(".o")
        and ptx_path == manifest_path.with_name(f"{manifest_path.stem}.ptx.json"),
        f"{location}: cache-key/manifest/object/PTX paths are not exact",
    )
    try:
        _validate_bound_artifacts(
            str(cache_key),
            object_path=object_path,
            manifest_path=manifest_path,
            ptx_path=object_path.with_suffix(".ptx"),
            sidecar_path=ptx_path,
            redisassemble=False,
        )
    except (OSError, RuntimeError) as exc:
        raise ContractBuildError(
            f"{location}: exact object/manifest/frontend-PTX quartet is invalid: {exc}"
        ) from exc
    _require(
        manifest.get("object_sha256") == object_ref["sha256"]
        and manifest.get("object_bytes") == object_ref["size_bytes"],
        f"{location}: object identity differs from compile manifest",
    )
    _require(
        manifest.get("package_fingerprint") == runtime_package_fingerprint,
        f"{location}: cache object and source-manifest fingerprints differ",
    )
    kernel_id = manifest.get("kernel_id")
    compile_spec_json = manifest.get("compile_spec_json")
    compile_spec_hash = manifest.get("compile_spec_hash")
    _require(
        isinstance(kernel_id, str)
        and bool(kernel_id)
        and isinstance(compile_spec_json, str)
        and bool(compile_spec_json)
        and _is_sha256(compile_spec_hash),
        f"{location}: compile identity is incomplete",
    )
    try:
        parsed_spec = json.loads(compile_spec_json)
    except json.JSONDecodeError as exc:
        raise ContractBuildError(f"{location}: invalid compile-spec JSON: {exc}") from exc
    _require(isinstance(parsed_spec, dict), f"{location}: compile spec is not an object")
    _require(
        parsed_spec.get("kernel") == kernel_id
        and hashlib.sha256(compile_spec_json.encode()).hexdigest() == compile_spec_hash,
        f"{location}: compile-spec hash/kernel binding differs",
    )
    semantic_payload = manifest.get("semantic_payload")
    _require(
        isinstance(semantic_payload, dict)
        and manifest.get("semantic_key") == _canonical_sha256(semantic_payload),
        f"{location}: semantic manifest hash differs",
    )
    launch_metadata = manifest.get("launch_metadata")
    _require(
        isinstance(launch_metadata, dict)
        and launch_metadata.get("status") == "exact"
        and isinstance(launch_metadata.get("launch_dynamic_smem_bytes"), dict)
        and launch_metadata["launch_dynamic_smem_bytes"],
        f"{location}: exact launch dynamic-SMEM metadata is required",
    )
    expected_artifact_evidence = {
        "cache_key": cache_key,
        "object_sha256": object_ref["sha256"],
        "launch_metadata": launch_metadata,
    }
    _require(
        manifest.get("artifact_evidence_sha256")
        == _canonical_sha256(expected_artifact_evidence),
        f"{location}: manifest artifact-evidence hash differs",
    )
    packages = _toolchain_packages(
        manifest.get("toolchain"), location=f"{location}.manifest.toolchain"
    )
    _require(
        packages == EXPECTED_PACKAGES[side],
        f"{location}: {side} CUTLASS packages are not exact",
    )

    sidecar = _load_json(ptx_path)
    _require(
        sidecar.get("schema") == FRONTEND_PTX_SCHEMA,
        f"{location}: unsupported frontend-PTX sidecar schema",
    )
    for field, expected in (
        ("cache_key", cache_key),
        ("package_fingerprint", runtime_package_fingerprint),
        ("kernel_id", kernel_id),
        ("compile_spec_hash", compile_spec_hash),
    ):
        _require(
            sidecar.get(field) == expected,
            f"{location}: PTX sidecar {field} binding differs",
        )
    compile_manifest = sidecar.get("compile_manifest")
    _require(
        isinstance(compile_manifest, dict)
        and Path(str(compile_manifest.get("path", ""))).resolve() == manifest_path
        and compile_manifest.get("schema") == COMPILE_MANIFEST_SCHEMA
        and compile_manifest.get("sha256") == manifest_ref["sha256"],
        f"{location}: PTX sidecar does not bind the exact compile manifest",
    )
    sidecar_object = sidecar.get("object")
    _require(
        isinstance(sidecar_object, dict)
        and Path(str(sidecar_object.get("path", ""))).resolve() == object_path
        and sidecar_object.get("sha256") == object_ref["sha256"]
        and sidecar_object.get("bytes") == object_ref["size_bytes"],
        f"{location}: PTX sidecar does not bind the exact object",
    )
    source_ptxas = sidecar.get("source_ptxas")
    ptxas_version = (
        source_ptxas.get("version") if isinstance(source_ptxas, dict) else None
    )
    _require(
        isinstance(ptxas_version, str) and bool(ptxas_version.strip()),
        f"{location}: source PTXAS version is empty",
    )
    return {
        "identity": {
            "role": role,
            "kernel_id": kernel_id,
            "compile_spec_hash": compile_spec_hash,
            "compile_spec_json": compile_spec_json,
        },
        "packages": packages,
        "ptxas_version": ptxas_version,
        "manifest_sha256": manifest_ref["sha256"],
        "object_sha256": object_ref["sha256"],
        "frontend_ptx_sidecar_sha256": ptx_ref["sha256"],
        "launch_dynamic_smem_bytes": launch_metadata[
            "launch_dynamic_smem_bytes"
        ],
    }


def _validate_graph(value: object, *, location: str) -> dict[str, Any]:
    graph = _exact_keys(value, {"node_count", "kernel_node_count", "nodes"}, location)
    node_count = _positive_int(graph["node_count"], f"{location}.node_count")
    kernel_node_count = _positive_int(
        graph["kernel_node_count"], f"{location}.kernel_node_count"
    )
    nodes = graph["nodes"]
    _require(
        isinstance(nodes, list) and len(nodes) == node_count,
        f"{location}: node list does not match node_count",
    )
    normalized_nodes: list[dict[str, object]] = []
    kernel_nodes: list[dict[str, object]] = []
    for index, raw_node in enumerate(nodes):
        node_location = f"{location}.nodes[{index}]"
        _require(isinstance(raw_node, dict), f"{node_location}: expected an object")
        node_type = raw_node.get("type")
        expected_fields = {"index", "type"}
        if node_type == KERNEL_NODE_TYPE:
            expected_fields.update(
                {"kernel_name", "grid", "block", "dynamic_smem_bytes"}
            )
        node = _exact_keys(raw_node, expected_fields, node_location)
        _require(node["index"] == index, f"{node_location}: raw node index differs")
        _require(
            isinstance(node_type, str) and bool(node_type),
            f"{node_location}: node type is empty",
        )
        if node_type == KERNEL_NODE_TYPE:
            _require(
                isinstance(node["kernel_name"], str) and bool(node["kernel_name"]),
                f"{node_location}: kernel name is empty",
            )
            for field in ("grid", "block"):
                dimensions = node[field]
                _require(
                    isinstance(dimensions, list)
                    and len(dimensions) == 3
                    and all(
                        isinstance(dimension, int)
                        and not isinstance(dimension, bool)
                        and dimension > 0
                        for dimension in dimensions
                    ),
                    f"{node_location}.{field}: expected three positive integers",
                )
            _nonnegative_int(
                node["dynamic_smem_bytes"],
                f"{node_location}.dynamic_smem_bytes",
            )
            kernel_nodes.append(node)
        normalized_nodes.append(node)
    _require(
        len(kernel_nodes) == kernel_node_count,
        f"{location}: observed kernel count differs from graph metadata",
    )
    signature = {
        "node_count": node_count,
        "kernel_node_count": kernel_node_count,
        "nodes": [
            {key: item for key, item in node.items() if key != "kernel_name"}
            for node in normalized_nodes
        ],
    }
    return {
        "contract": {
            "topology_sha256": _canonical_sha256(signature),
            "node_count": node_count,
            "kernel_node_count": kernel_node_count,
        },
        # The release contract intentionally ignores compiler-generated symbol
        # names in its topology hash.  Same-arm discoveries on GPUs 4 and 5
        # must nevertheless agree on every observed node field, including the
        # names, so retain the complete metadata until consensus is checked.
        "consensus": {
            "signature": signature,
            "nodes_with_kernel_names": normalized_nodes,
        },
        "kernel_nodes": kernel_nodes,
    }


def _source_records_by_repo_path(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    runtime = source["manifest"]["runtime"]["sparkinfer_package"]
    return {
        f"sparkinfer/{record['path']}": record
        for record in runtime["files"]
        if isinstance(record, dict)
    }


def _validate_source_owned_bindings(
    records: Sequence[Mapping[str, Any]],
    *,
    location: str,
    kernel_nodes: Sequence[Mapping[str, Any]],
    source_by_path: Mapping[str, Mapping[str, Any]],
    harness_by_path: Mapping[str, Mapping[str, Any]],
) -> None:
    for index, record in enumerate(records):
        record_location = f"{location}[{index}]"
        ordinal = int(record["node_index"])
        node = kernel_nodes[ordinal]
        for field in ("kernel_name", "grid", "block", "dynamic_smem_bytes"):
            _require(
                record[field] == node[field],
                f"{record_location}: source-owned {field} differs from graph node",
            )
        for source_file in record["source_files"]:
            path = str(source_file["path"])
            expected = source_by_path.get(path) or harness_by_path.get(path)
            _require(
                expected is not None and source_file["sha256"] == expected["sha256"],
                f"{record_location}: source file is not bound to source/harness: {path}",
            )


def _validate_case_discovery(
    value: object,
    *,
    location: str,
    side: str,
    runtime_package_fingerprint: str,
    source_by_path: Mapping[str, Mapping[str, Any]],
    harness_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    case = _exact_keys(
        value,
        {
            "case_id",
            "input_sha256",
            "required_correctness_gates",
            "cross_arm_output_policy",
            "topology_review",
            "graph",
            "artifacts",
            "launch_plan",
            "source_owned_kernel_nodes",
        },
        location,
    )
    case_id = case["case_id"]
    _require(
        isinstance(case_id, str) and bool(_CASE_ID_RE.fullmatch(case_id)),
        f"{location}: invalid case ID",
    )
    _require(_is_sha256(case["input_sha256"]), f"{location}: invalid input SHA")
    gates = case["required_correctness_gates"]
    _require(
        isinstance(gates, list)
        and bool(gates)
        and gates == sorted(set(gates))
        and all(isinstance(gate, str) and gate for gate in gates),
        f"{location}: correctness gates are not sorted/unique/nonempty",
    )
    _require(
        case["cross_arm_output_policy"] in {"bit-exact", "oracle-only"},
        f"{location}: invalid cross-arm output policy",
    )
    topology_review = _exact_keys(
        case["topology_review"], {"disposition", "reason"}, f"{location}.topology_review"
    )
    _require(
        topology_review["disposition"] in {"equal", "changed-reviewed"}
        and isinstance(topology_review["reason"], str),
        f"{location}: invalid topology review",
    )
    graph = _validate_graph(case["graph"], location=f"{location}.graph")
    kernel_node_count = int(graph["contract"]["kernel_node_count"])
    raw_artifacts = case["artifacts"]
    _require(
        isinstance(raw_artifacts, list) and bool(raw_artifacts),
        f"{location}: no exact cache artifacts",
    )
    artifacts = [
        _validate_cache_artifact(
            artifact,
            location=f"{location}.artifacts[{index}]",
            side=side,
            runtime_package_fingerprint=runtime_package_fingerprint,
        )
        for index, artifact in enumerate(raw_artifacts)
    ]
    by_role = {str(artifact["identity"]["role"]): artifact for artifact in artifacts}
    _require(
        len(by_role) == len(artifacts),
        f"{location}: duplicate exact artifact role",
    )
    compile_ids = {
        (
            str(artifact["identity"]["kernel_id"]),
            str(artifact["identity"]["compile_spec_hash"]),
        )
        for artifact in artifacts
    }
    _require(
        len(compile_ids) == len(artifacts),
        f"{location}: duplicate exact compile identity",
    )
    package_sets = {tuple(sorted(artifact["packages"].items())) for artifact in artifacts}
    ptxas_versions = {str(artifact["ptxas_version"]) for artifact in artifacts}
    _require(
        len(package_sets) == 1 and len(ptxas_versions) == 1,
        f"{location}: mixed exact-object toolchains",
    )

    raw_plan = case["launch_plan"]
    _require(isinstance(raw_plan, list) and raw_plan, f"{location}: empty launch plan")
    launch_plan: list[dict[str, object]] = []
    previous_node_index = -1
    multiplicities: dict[str, int] = {}
    used_roles: set[str] = set()
    for index, raw_binding in enumerate(raw_plan):
        binding_location = f"{location}.launch_plan[{index}]"
        binding = _exact_keys(
            raw_binding,
            {"node_index", "artifact_role", "multiplicity_index"},
            binding_location,
        )
        node_index = _nonnegative_int(
            binding["node_index"], f"{binding_location}.node_index"
        )
        _require(
            node_index < kernel_node_count and node_index > previous_node_index,
            f"{binding_location}: exact node ordinals are not ordered/in-range",
        )
        previous_node_index = node_index
        role = binding["artifact_role"]
        _require(
            isinstance(role, str) and role in by_role,
            f"{binding_location}: unknown exact artifact role",
        )
        multiplicity = multiplicities.get(role, 0) + 1
        _require(
            binding["multiplicity_index"] == multiplicity,
            f"{binding_location}: launch multiplicity differs",
        )
        multiplicities[role] = multiplicity
        used_roles.add(role)
        artifact = by_role[role]
        kernel_node = graph["kernel_nodes"][node_index]
        smem_by_symbol = artifact["launch_dynamic_smem_bytes"]
        observed_smem = kernel_node["dynamic_smem_bytes"]
        allowed_smem = smem_by_symbol.get(kernel_node["kernel_name"])
        _require(
            isinstance(allowed_smem, list)
            and observed_smem in allowed_smem,
            f"{binding_location}: graph symbol/dynamic-SMEM is not in exact manifest",
        )
        identity = artifact["identity"]
        launch_plan.append(
            {
                "node_index": node_index,
                "artifact_role": role,
                "kernel_id": identity["kernel_id"],
                "compile_spec_hash": identity["compile_spec_hash"],
                "multiplicity_index": multiplicity,
            }
        )
    _require(
        used_roles == set(by_role),
        f"{location}: one or more exact artifacts are unused",
    )
    try:
        source_owned = _validate_source_owned_kernel_nodes(
            case["source_owned_kernel_nodes"],
            location=f"{location}.source_owned_kernel_nodes",
            kernel_node_count=kernel_node_count,
        )
    except EndToEndValidationError as exc:
        raise ContractBuildError(str(exc)) from exc
    _validate_source_owned_bindings(
        source_owned,
        location=f"{location}.source_owned_kernel_nodes",
        kernel_nodes=graph["kernel_nodes"],
        source_by_path=source_by_path,
        harness_by_path=harness_by_path,
    )
    compile_contract = {
        "artifacts": sorted(
            [artifact["identity"] for artifact in artifacts],
            key=lambda artifact: str(artifact["role"]),
        ),
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": source_owned,
    }
    try:
        _validate_compile_side_contract(
            compile_contract,
            location=f"{location}.derived_compile_contract",
            kernel_node_count=kernel_node_count,
        )
    except EndToEndValidationError as exc:
        raise ContractBuildError(str(exc)) from exc
    artifact_consensus = sorted(
        [
            {
                "role": artifact["identity"]["role"],
                "manifest_sha256": artifact["manifest_sha256"],
                "object_sha256": artifact["object_sha256"],
                "frontend_ptx_sidecar_sha256": artifact[
                    "frontend_ptx_sidecar_sha256"
                ],
            }
            for artifact in artifacts
        ],
        key=lambda artifact: str(artifact["role"]),
    )
    return {
        "case_id": case_id,
        "input_sha256": case["input_sha256"],
        "required_correctness_gates": gates,
        "cross_arm_output_policy": case["cross_arm_output_policy"],
        "topology_review": topology_review,
        "topology": graph["contract"],
        "graph_consensus": graph["consensus"],
        "compile_contract": compile_contract,
        "artifact_consensus": artifact_consensus,
        "toolchain": {
            "cutlass_packages": dict(package_sets.pop()),
            "ptxas_version": ptxas_versions.pop(),
        },
    }


def _validate_discovery(
    path: Path,
    *,
    side: str,
    source: Mapping[str, Any],
    harness: Mapping[str, Any],
    registry: Mapping[str, str],
) -> dict[str, Any]:
    path = _regular_file(path, location=f"{side} discovery")
    value = _exact_keys(
        _load_json(path),
        {
            "schema",
            "side",
            "discovery_id",
            "review",
            "source",
            "harness_tree_fingerprint",
            "producer_registry_sha256",
            "gpu",
            "families",
            "discovery_sha256",
        },
        str(path),
    )
    _require(value["schema"] == DISCOVERY_SCHEMA, f"{path}: invalid discovery schema")
    _require(value["side"] == side, f"{path}: discovery side differs")
    _require(
        isinstance(value["discovery_id"], str) and bool(value["discovery_id"].strip()),
        f"{path}: discovery ID is empty",
    )
    payload = {key: item for key, item in value.items() if key != "discovery_sha256"}
    _require(
        value["discovery_sha256"] == _canonical_sha256(payload),
        f"{path}: discovery canonical hash differs",
    )
    review = _exact_keys(
        value["review"], {"status", "review_id", "reviewed_unix_ns"}, f"{path}.review"
    )
    _require(
        review["status"] == "reviewed"
        and isinstance(review["review_id"], str)
        and bool(review["review_id"].strip()),
        f"{path}: discovery has not been explicitly reviewed",
    )
    _positive_int(review["reviewed_unix_ns"], f"{path}.review.reviewed_unix_ns")
    source_binding = _exact_keys(
        value["source"],
        {
            "manifest_sha256",
            "manifest_artifact_sha256",
            "runtime_package_fingerprint",
        },
        f"{path}.source",
    )
    expected_source_binding = {
        key: source[key]
        for key in (
            "manifest_sha256",
            "manifest_artifact_sha256",
            "runtime_package_fingerprint",
        )
    }
    _require(
        source_binding == expected_source_binding,
        f"{path}: discovery is not bound to the selected source manifest",
    )
    _require(
        value["harness_tree_fingerprint"] == harness["tree_fingerprint"],
        f"{path}: discovery harness fingerprint differs",
    )
    _require(
        value["producer_registry_sha256"] == _registry_sha256(registry),
        f"{path}: discovery producer registry differs",
    )
    gpu = _exact_keys(
        value["gpu"],
        {"physical_ordinal", "uuid", "name", "capability"},
        f"{path}.gpu",
    )
    _require(
        gpu["physical_ordinal"] in PHYSICAL_GPUS
        and isinstance(gpu["uuid"], str)
        and bool(gpu["uuid"].strip())
        and isinstance(gpu["name"], str)
        and bool(gpu["name"].strip())
        and gpu["capability"] == [12, 0],
        f"{path}: discovery GPU is not one reviewed physical SM120 GPU",
    )
    families = value["families"]
    _require(
        isinstance(families, dict) and set(families) == set(REQUIRED_FAMILIES),
        f"{path}: discovery family set is incomplete",
    )
    harness_by_path = {
        str(record["path"]): record for record in harness["files"]
    }
    source_by_path = _source_records_by_repo_path(source)
    normalized_families: dict[str, Any] = {}
    all_case_ids: set[str] = set()
    toolchains: set[str] = set()
    for family in REQUIRED_FAMILIES:
        location = f"{path}.families.{family}"
        family_record = _exact_keys(
            families[family], {"producer", "producer_sha256", "cases"}, location
        )
        producer = registry[family]
        producer_harness = harness_by_path.get(producer)
        _require(
            family_record["producer"] == producer
            and producer_harness is not None
            and family_record["producer_sha256"] == producer_harness["sha256"],
            f"{location}: producer is not bound to the frozen registry/harness",
        )
        raw_cases = family_record["cases"]
        _require(isinstance(raw_cases, list) and raw_cases, f"{location}: no cases")
        cases = [
            _validate_case_discovery(
                case,
                location=f"{location}.cases[{index}]",
                side=side,
                runtime_package_fingerprint=str(source["runtime_package_fingerprint"]),
                source_by_path=source_by_path,
                harness_by_path=harness_by_path,
            )
            for index, case in enumerate(raw_cases)
        ]
        case_ids = [str(case["case_id"]) for case in cases]
        _require(
            case_ids == sorted(set(case_ids)),
            f"{location}: cases are not sorted/unique",
        )
        duplicates = all_case_ids.intersection(case_ids)
        _require(not duplicates, f"{location}: globally duplicate case IDs {duplicates!r}")
        all_case_ids.update(case_ids)
        for case in cases:
            toolchains.add(_canonical_sha256(case["toolchain"]))
        normalized_families[family] = {
            "producer": producer,
            "producer_sha256": family_record["producer_sha256"],
            "cases": cases,
        }
    _require(
        len(toolchains) == 1,
        f"{path}: discovery contains mixed arm toolchains across families",
    )
    first_case = normalized_families[REQUIRED_FAMILIES[0]]["cases"][0]
    return {
        "path": str(path),
        "artifact_sha256": _sha256_file(path),
        "discovery_id": value["discovery_id"],
        "gpu": gpu,
        "families": normalized_families,
        "toolchain": first_case["toolchain"],
    }


def _side_consensus(discoveries: Sequence[dict[str, Any]], *, side: str) -> dict[str, Any]:
    by_gpu = {int(discovery["gpu"]["physical_ordinal"]): discovery for discovery in discoveries}
    _require(
        len(by_gpu) == len(discoveries) and set(by_gpu) == set(PHYSICAL_GPUS),
        f"{side}: discovery coverage must be exactly physical GPUs {PHYSICAL_GPUS}",
    )
    ids = [str(discovery["discovery_id"]) for discovery in discoveries]
    artifacts = [str(discovery["artifact_sha256"]) for discovery in discoveries]
    _require(
        len(set(ids)) == len(ids) and len(set(artifacts)) == len(artifacts),
        f"{side}: discovery IDs/artifacts are reused",
    )
    reference = by_gpu[PHYSICAL_GPUS[0]]
    for gpu in PHYSICAL_GPUS[1:]:
        observed = by_gpu[gpu]
        _require(
            observed["families"] == reference["families"]
            and observed["toolchain"] == reference["toolchain"],
            f"{side}: GPU {gpu} discovery differs from GPU {PHYSICAL_GPUS[0]}",
        )
    return {
        "families": reference["families"],
        "toolchain": reference["toolchain"],
        "gpus": {gpu: by_gpu[gpu]["gpu"] for gpu in PHYSICAL_GPUS},
    }


def _merge_families(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, object]:
    families: dict[str, object] = {}
    for family in REQUIRED_FAMILIES:
        baseline_family = baseline["families"][family]
        current_family = current["families"][family]
        _require(
            baseline_family["producer"] == current_family["producer"]
            and baseline_family["producer_sha256"]
            == current_family["producer_sha256"],
            f"{family}: producer differs across arms",
        )
        baseline_cases = {
            str(case["case_id"]): case for case in baseline_family["cases"]
        }
        current_cases = {
            str(case["case_id"]): case for case in current_family["cases"]
        }
        _require(
            set(baseline_cases) == set(current_cases),
            f"{family}: case set differs across arms",
        )
        cases: list[dict[str, object]] = []
        for case_id in sorted(baseline_cases):
            old = baseline_cases[case_id]
            new = current_cases[case_id]
            for field in (
                "input_sha256",
                "required_correctness_gates",
                "cross_arm_output_policy",
                "topology_review",
            ):
                _require(
                    old[field] == new[field],
                    f"{case_id}: {field} differs across arms",
                )
            topology_review = old["topology_review"]
            signatures_equal = old["topology"] == new["topology"]
            if signatures_equal:
                _require(
                    topology_review == {"disposition": "equal", "reason": ""},
                    f"{case_id}: equal topology requires an empty equal review",
                )
            else:
                _require(
                    topology_review["disposition"] == "changed-reviewed"
                    and bool(str(topology_review["reason"]).strip()),
                    f"{case_id}: changed topology requires one reviewed reason",
                )
            case_payload = {
                "case_id": case_id,
                "input_sha256": old["input_sha256"],
                "required_correctness_gates": old["required_correctness_gates"],
                "cross_arm_output_policy": old["cross_arm_output_policy"],
                "graph_topology_contract": {
                    "disposition": topology_review["disposition"],
                    "reason": topology_review["reason"],
                    "baseline": old["topology"],
                    "current": new["topology"],
                },
                "compile_artifact_contract": {
                    "baseline": old["compile_contract"],
                    "current": new["compile_contract"],
                },
            }
            cases.append(
                {
                    **case_payload,
                    "case_contract_sha256": _canonical_sha256(case_payload),
                }
            )
        family_payload = {
            "producer": baseline_family["producer"],
            "producer_sha256": baseline_family["producer_sha256"],
            "cases": cases,
        }
        families[family] = {
            **family_payload,
            "family_contract_sha256": _canonical_sha256(family_payload),
        }
    return families


def build_contract(
    *,
    baseline_source_manifest: Path,
    current_source_manifest: Path,
    baseline_discoveries: Sequence[Path],
    current_discoveries: Sequence[Path],
    harness_root: Path,
    corpus_id: str,
    version: str,
) -> dict[str, object]:
    """Validate immutable inputs and return one canonical contract-v3 value."""

    _require(bool(corpus_id.strip()), "corpus_id must be nonempty")
    _require(bool(version.strip()), "version must be nonempty")
    baseline_source = _load_source(baseline_source_manifest, "baseline")
    current_source = _load_source(current_source_manifest, "current")
    harness = _snapshot_harness(harness_root)
    for side, source in (
        ("baseline", baseline_source),
        ("current", current_source),
    ):
        runtime_root = Path(source["manifest"]["runtime"]["repo_root"])
        _verify_harness_copy(
            harness, runtime_root=runtime_root, location=f"{side} source runtime"
        )
    registry = _registry()
    harness_by_path = {str(record["path"]): record for record in harness["files"]}
    for family, producer in registry.items():
        _require(
            producer in harness_by_path,
            f"{family}: registered producer is absent from the frozen harness",
        )
    baseline_records = [
        _validate_discovery(
            path,
            side="baseline",
            source=baseline_source,
            harness=harness,
            registry=registry,
        )
        for path in baseline_discoveries
    ]
    current_records = [
        _validate_discovery(
            path,
            side="current",
            source=current_source,
            harness=harness,
            registry=registry,
        )
        for path in current_discoveries
    ]
    baseline = _side_consensus(baseline_records, side="baseline")
    current = _side_consensus(current_records, side="current")
    for gpu in PHYSICAL_GPUS:
        old_gpu = baseline["gpus"][gpu]
        new_gpu = current["gpus"][gpu]
        _require(
            old_gpu == new_gpu,
            f"physical GPU {gpu}: hardware discovery identity differs across arms",
        )
    _require(
        baseline["toolchain"]["cutlass_packages"] == EXPECTED_PACKAGES["baseline"]
        and current["toolchain"]["cutlass_packages"] == EXPECTED_PACKAGES["current"],
        "derived arm CUTLASS package maps are not exact",
    )
    payload = {
        "schema": CONTRACT_SCHEMA,
        "corpus_id": corpus_id,
        "version": version,
        "harness": harness,
        "arm_toolchains": {
            "baseline": baseline["toolchain"],
            "current": current["toolchain"],
        },
        "required_families": list(REQUIRED_FAMILIES),
        "families": _merge_families(baseline, current),
    }
    return {**payload, "contract_sha256": _canonical_sha256(payload)}


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                value,
                temporary,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source-manifest", type=Path, required=True)
    parser.add_argument("--current-source-manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline-discovery",
        type=Path,
        action="append",
        required=True,
        help="repeat for the reviewed GPU-4 and GPU-5 baseline discoveries",
    )
    parser.add_argument(
        "--current-discovery",
        type=Path,
        action="append",
        required=True,
        help="repeat for the reviewed GPU-4 and GPU-5 current discoveries",
    )
    parser.add_argument(
        "--harness-root",
        type=Path,
        default=_REPO_ROOT,
        help="root whose complete benchmarks/tests/validation trees are frozen",
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--version", default="1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    harness_root = args.harness_root.resolve()
    for tree_name in HARNESS_TREES:
        tree = harness_root / tree_name
        if output == tree or tree in output.parents:
            parser.error("--output must be outside the snapshotted harness trees")
    return args


def main() -> int:
    args = _args()
    started = time.monotonic()
    try:
        contract = build_contract(
            baseline_source_manifest=args.baseline_source_manifest,
            current_source_manifest=args.current_source_manifest,
            baseline_discoveries=args.baseline_discovery,
            current_discoveries=args.current_discovery,
            harness_root=args.harness_root,
            corpus_id=args.corpus_id,
            version=args.version,
        )
        output = args.output.resolve()
        _atomic_write_json(output, contract)
        _validate_contract(output)
    except (ContractBuildError, EndToEndValidationError, OSError) as exc:
        print(f"end-to-end contract build failed: {exc}", file=sys.stderr)
        return 1
    print(
        "status=pass "
        f"families={len(REQUIRED_FAMILIES)} "
        f"physical_gpus={','.join(map(str, PHYSICAL_GPUS))} "
        f"contract_sha256={contract['contract_sha256']} "
        f"elapsed_seconds={time.monotonic() - started:.3f} "
        f"output={output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
