#!/usr/bin/env python3
"""Compare graph-replay samples from an A-B-B-A benchmark sequence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CUTLASS_PACKAGES = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
)
_CASE_CONTRACT_SCHEMA = "sparkinfer.graph_replay_case_contract.v1"
_RUNTIME_ENVIRONMENT_SCHEMA = "sparkinfer-runtime-environment-v1"
_RUNTIME_ENVIRONMENT_PREFIXES = (
    "SPARKINFER_",
    "CUTE_",
    "CUTLASS_",
    "CUDA_",
    "TORCH_",
    "PYTORCH_",
    "TRITON_",
    "NVCC_",
    "PTXAS_",
    "NCCL_",
)
_RUNTIME_ENVIRONMENT_EXPLICIT_CONTROLS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_PATH",
    "CUDA_CACHE_MAXSIZE",
    "CUDA_FORCE_PTX_JIT",
    "CUDA_DISABLE_PTX_JIT",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_TF32_OVERRIDE",
)
# These values are required to be stable within each arm but cannot be equal
# across a CUTLASS-version comparison that uses isolated venvs/caches.  No
# other environment field is normalized.
_RUNTIME_ENVIRONMENT_OPERATIONAL_PATH_EXCEPTIONS = (
    "SPARKINFER_CUTE_COMPILE_CACHE_DIR",
    "CUTE_DSL_CACHE_DIR",
    "CUTE_DSL_LIBS",
)


@dataclass(frozen=True)
class Run:
    path: Path
    provenance: dict[str, Any]
    samples: dict[str, list[float]]
    cases: dict[str, dict[str, Any]]


def _case_key(case: dict[str, Any]) -> str:
    return json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(_case_key(payload).encode("utf-8")).hexdigest()


def _validate_runtime_environment(
    path: Path,
    provenance: dict[str, Any],
) -> None:
    environment = provenance.get("runtime_environment")
    if not isinstance(environment, dict):
        raise ValueError(f"{path}: runtime environment provenance is missing")
    expected_fields = {
        "schema",
        "complete_set_variable_prefixes",
        "set_variables",
        "explicit_controls",
        "nvidia_enumeration",
        "sha256",
    }
    if set(environment) != expected_fields:
        raise ValueError(
            f"{path}: runtime environment fields differ from the canonical schema: "
            f"missing={sorted(expected_fields - set(environment))}, "
            f"unexpected={sorted(set(environment) - expected_fields)}"
        )
    if environment.get("schema") != _RUNTIME_ENVIRONMENT_SCHEMA:
        raise ValueError(f"{path}: unsupported runtime environment schema")
    if environment.get("complete_set_variable_prefixes") != list(
        _RUNTIME_ENVIRONMENT_PREFIXES
    ):
        raise ValueError(f"{path}: runtime environment prefix coverage is incomplete")

    set_variables = environment.get("set_variables")
    if not isinstance(set_variables, dict) or any(
        not isinstance(name, str)
        or not name.startswith(_RUNTIME_ENVIRONMENT_PREFIXES)
        or not isinstance(value, str)
        for name, value in set_variables.items()
    ):
        raise ValueError(f"{path}: runtime environment set-variable map is invalid")

    explicit_controls = environment.get("explicit_controls")
    if not isinstance(explicit_controls, dict) or set(explicit_controls) != set(
        _RUNTIME_ENVIRONMENT_EXPLICIT_CONTROLS
    ):
        raise ValueError(f"{path}: runtime environment explicit-control map is incomplete")
    for name in _RUNTIME_ENVIRONMENT_EXPLICIT_CONTROLS:
        entry = explicit_controls[name]
        if not isinstance(entry, dict) or entry.get("status") not in {"set", "missing"}:
            raise ValueError(f"{path}: invalid explicit environment control {name!r}")
        if entry["status"] == "set":
            if set(entry) != {"status", "value"} or not isinstance(
                entry["value"], str
            ):
                raise ValueError(f"{path}: invalid set environment control {name!r}")
            if name.startswith(_RUNTIME_ENVIRONMENT_PREFIXES) and set_variables.get(
                name
            ) != entry["value"]:
                raise ValueError(
                    f"{path}: explicit control {name!r} disagrees with set_variables"
                )
        else:
            if set(entry) != {"status"} or name in set_variables:
                raise ValueError(f"{path}: invalid missing environment control {name!r}")

    nvidia_enumeration = environment.get("nvidia_enumeration")
    if (
        not isinstance(nvidia_enumeration, dict)
        or nvidia_enumeration.get("policy") != "explicit-only"
        or nvidia_enumeration.get("included")
        != [
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "NVIDIA_TF32_OVERRIDE",
        ]
        or not isinstance(nvidia_enumeration.get("reason"), str)
        or not nvidia_enumeration["reason"]
    ):
        raise ValueError(f"{path}: NVIDIA environment enumeration policy is invalid")

    recorded_sha256 = environment.get("sha256")
    unhashed_environment = dict(environment)
    unhashed_environment.pop("sha256")
    computed_sha256 = _json_sha256(unhashed_environment)
    if (
        not isinstance(recorded_sha256, str)
        or not _SHA256_RE.fullmatch(recorded_sha256)
        or recorded_sha256 != computed_sha256
    ):
        raise ValueError(
            f"{path}: runtime environment SHA256 mismatch: "
            f"recorded={recorded_sha256!r} computed={computed_sha256}"
        )


def _runtime_environment_raw_sha256(run: Run) -> str:
    return str(run.provenance["runtime_environment"]["sha256"])


def _runtime_environment_comparison_payload(run: Run) -> dict[str, Any]:
    """Normalize only version/cache venv paths that cannot agree across A/B."""
    environment = json.loads(json.dumps(run.provenance["runtime_environment"]))
    environment.pop("sha256")
    set_variables = environment["set_variables"]
    path_states: dict[str, dict[str, Any]] = {}
    for name in _RUNTIME_ENVIRONMENT_OPERATIONAL_PATH_EXCEPTIONS:
        raw_value = set_variables.pop(name, None)
        if raw_value is None:
            path_states[name] = {"status": "missing"}
        elif name == "CUTE_DSL_LIBS":
            path_states[name] = {
                "status": "set",
                "library_basenames": [
                    Path(component).name
                    for component in raw_value.split(":")
                    if component
                ],
            }
        else:
            path_states[name] = {"status": "set"}
    environment["operational_path_exceptions"] = {
        "policy": "values-normalized; set/missing state remains exact",
        "names": list(_RUNTIME_ENVIRONMENT_OPERATIONAL_PATH_EXCEPTIONS),
        "states": path_states,
    }
    return environment


def _runtime_environment_comparison_sha256(run: Run) -> str:
    return _json_sha256(_runtime_environment_comparison_payload(run))


def _validate_case_contract(path: Path, line_number: int, case: dict[str, Any]) -> None:
    contract_sha = case.get("case_contract_sha256")
    input_sha = case.get("input_generation_sha256")
    input_seed = case.get("input_seed")
    if not isinstance(contract_sha, str) or not _SHA256_RE.fullmatch(contract_sha):
        raise ValueError(f"{path}:{line_number}: case lacks a SHA-256 case contract")
    if not isinstance(input_sha, str) or not _SHA256_RE.fullmatch(input_sha):
        raise ValueError(
            f"{path}:{line_number}: case lacks a SHA-256 input-generation identity"
        )
    if isinstance(input_seed, bool) or not isinstance(input_seed, int):
        raise ValueError(f"{path}:{line_number}: case lacks an integer input seed")
    unhashed_case = dict(case)
    unhashed_case.pop("case_contract_sha256")
    expected = _json_sha256(unhashed_case)
    if contract_sha != expected:
        raise ValueError(
            f"{path}:{line_number}: case contract hash mismatch: "
            f"recorded={contract_sha} computed={expected}"
        )


def _validate_provenance(path: Path, provenance: dict[str, Any]) -> None:
    dependencies = provenance.get("benchmark_dependencies_sha256")
    if (
        not isinstance(dependencies, dict)
        or not dependencies
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
            for name, digest in dependencies.items()
        )
    ):
        raise ValueError(f"{path}: invalid benchmark dependency hashes")
    benchmark_sha = provenance.get("benchmark_sha256")
    if (
        not isinstance(benchmark_sha, str)
        or not _SHA256_RE.fullmatch(benchmark_sha)
        or benchmark_sha not in dependencies.values()
    ):
        raise ValueError(f"{path}: benchmark SHA256 is missing from dependencies")
    config = provenance.get("benchmark_config")
    config_sha = provenance.get("benchmark_config_sha256")
    if not isinstance(config, dict) or not config:
        raise ValueError(f"{path}: benchmark config is missing")
    if not isinstance(config_sha, str) or config_sha != _json_sha256(config):
        raise ValueError(f"{path}: benchmark config SHA256 does not match config")
    case_contract = provenance.get("benchmark_case_contract")
    if not isinstance(case_contract, dict):
        raise ValueError(f"{path}: benchmark case contract is missing")
    recorded_case_contract_sha = case_contract.get("sha256")
    unhashed_case_contract = dict(case_contract)
    unhashed_case_contract.pop("sha256", None)
    if not isinstance(
        recorded_case_contract_sha, str
    ) or recorded_case_contract_sha != _json_sha256(unhashed_case_contract):
        raise ValueError(f"{path}: benchmark case contract SHA256 mismatch")
    identity_fields = case_contract.get("identity_fields")
    expected_cases = case_contract.get("expected")
    if (
        not isinstance(identity_fields, list)
        or not identity_fields
        or len(set(identity_fields)) != len(identity_fields)
        or any(not isinstance(field, str) or not field for field in identity_fields)
        or not isinstance(expected_cases, list)
        or not expected_cases
        or any(not isinstance(case, dict) for case in expected_cases)
    ):
        raise ValueError(f"{path}: invalid benchmark case contract")
    argv = provenance.get("argv")
    if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
        raise ValueError(f"{path}: argv must be a string list")
    for field in ("python", "torch", "torch_cuda"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ValueError(f"{path}: provenance field {field!r} is empty")
    runtime_packages = provenance.get("runtime_packages")
    if not isinstance(runtime_packages, dict) or not runtime_packages:
        raise ValueError(f"{path}: runtime package provenance is missing")
    cutlass = provenance.get("cutlass")
    if (
        not isinstance(cutlass, dict)
        or set(cutlass) != set(_CUTLASS_PACKAGES)
        or any(
            not isinstance(cutlass[package], str) or not cutlass[package]
            for package in _CUTLASS_PACKAGES
        )
    ):
        raise ValueError(f"{path}: full five-package CUTLASS provenance is missing")
    _validate_runtime_environment(path, provenance)


def _external_case_contract_payload(
    backend: str, provenance: dict[str, Any]
) -> dict[str, Any]:
    case_contract = dict(provenance["benchmark_case_contract"])
    case_contract.pop("sha256", None)
    return {
        "schema": _CASE_CONTRACT_SCHEMA,
        "backend": backend,
        "benchmark_config": provenance["benchmark_config"],
        "benchmark_case_contract": case_contract,
    }


def _read_external_case_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read case contract: {exc}") from exc
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: case contract must be a JSON object")
    if contract.get("schema") != _CASE_CONTRACT_SCHEMA:
        raise ValueError(f"{path}: unsupported case contract schema")
    if set(contract) != {
        "schema",
        "backend",
        "benchmark_config",
        "benchmark_case_contract",
    }:
        raise ValueError(f"{path}: case contract has missing or unexpected fields")
    if not isinstance(contract.get("backend"), str) or not contract["backend"]:
        raise ValueError(f"{path}: case contract backend is empty")
    if not isinstance(contract.get("benchmark_config"), dict):
        raise ValueError(f"{path}: benchmark_config must be an object")
    case_contract = contract.get("benchmark_case_contract")
    if not isinstance(case_contract, dict) or set(case_contract) != {
        "identity_fields",
        "expected",
    }:
        raise ValueError(f"{path}: benchmark_case_contract is invalid")
    identity_fields = case_contract.get("identity_fields")
    expected = case_contract.get("expected")
    if (
        not isinstance(identity_fields, list)
        or not identity_fields
        or len(set(identity_fields)) != len(identity_fields)
        or any(not isinstance(field, str) or not field for field in identity_fields)
        or not isinstance(expected, list)
        or not expected
        or any(not isinstance(case, dict) for case in expected)
    ):
        raise ValueError(f"{path}: benchmark_case_contract entries are invalid")
    return contract


def _validate_read_only_input_provenance(
    path: Path,
    line_number: int,
    provenance: object,
) -> None:
    if not isinstance(provenance, dict) or set(provenance) != {
        "schema",
        "tensor_sha256",
        "aggregate_sha256",
    }:
        raise ValueError(
            f"{path}:{line_number}: read-only input provenance is missing or invalid"
        )
    tensor_hashes = provenance.get("tensor_sha256")
    if (
        provenance.get("schema") != "sparkinfer-read-only-inputs-v1"
        or not isinstance(tensor_hashes, dict)
        or not tensor_hashes
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
            for name, digest in tensor_hashes.items()
        )
    ):
        raise ValueError(f"{path}:{line_number}: invalid read-only input hashes")
    expected_aggregate = _json_sha256(
        {
            "schema": "sparkinfer-read-only-inputs-v1",
            "tensor_sha256": tensor_hashes,
        }
    )
    if provenance.get("aggregate_sha256") != expected_aggregate:
        raise ValueError(
            f"{path}:{line_number}: read-only input aggregate SHA256 mismatch"
        )


def _validate_reference_correctness(
    path: Path, line_number: int, correctness: dict[str, Any]
) -> None:
    required = {
        "oracle",
        "passed",
        "finite",
        "nonzero",
        "max_abs",
        "relative_l2",
        "cosine",
        "allclose",
        "minimum_cosine",
        "maximum_relative_l2",
        "relative_tolerance",
        "absolute_tolerance",
        "read_only_inputs",
    }
    missing = sorted(required - set(correctness))
    if missing:
        raise ValueError(
            f"{path}:{line_number}: reference correctness lacks {missing!r}"
        )
    if correctness.get("oracle") != "torch-reference":
        raise ValueError(f"{path}:{line_number}: oracle is not torch-reference")
    if correctness.get("passed") is not True or correctness.get("finite") is not True:
        raise ValueError(f"{path}:{line_number}: reference correctness did not pass")
    if correctness.get("allclose") is not True:
        raise ValueError(f"{path}:{line_number}: reference allclose did not pass")
    nonzero = correctness.get("nonzero")
    if isinstance(nonzero, bool) or not isinstance(nonzero, int) or nonzero <= 0:
        raise ValueError(f"{path}:{line_number}: reference output is zero")
    metrics: dict[str, float] = {}
    for field in (
        "max_abs",
        "relative_l2",
        "cosine",
        "minimum_cosine",
        "maximum_relative_l2",
        "relative_tolerance",
        "absolute_tolerance",
    ):
        try:
            value = float(correctness[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid correctness {field}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"{path}:{line_number}: non-finite correctness {field}")
        metrics[field] = value
    if (
        metrics["max_abs"] < 0
        or metrics["relative_l2"] < 0
        or not -1.0 <= metrics["cosine"] <= 1.0
        or not -1.0 <= metrics["minimum_cosine"] <= 1.0
        or metrics["maximum_relative_l2"] < 0
        or metrics["relative_tolerance"] < 0
        or metrics["absolute_tolerance"] < 0
    ):
        raise ValueError(f"{path}:{line_number}: invalid correctness bounds")
    if metrics["relative_l2"] > metrics["maximum_relative_l2"]:
        raise ValueError(f"{path}:{line_number}: relative L2 exceeds recorded bound")
    if metrics["cosine"] < metrics["minimum_cosine"]:
        raise ValueError(f"{path}:{line_number}: cosine is below recorded bound")
    _validate_read_only_input_provenance(
        path,
        line_number,
        correctness.get("read_only_inputs"),
    )


def _read_run(
    path: Path,
    backend: str,
    *,
    minimum_warmup: int,
    minimum_replays: int,
    require_reference_oracle: bool,
) -> Run:
    provenance: list[dict[str, Any]] = []
    samples: dict[str, list[float]] = {}
    cases: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            if record.get("type") == "provenance":
                provenance.append(record)
                continue
            if record.get("type") != "graph-replay-samples":
                continue
            if record.get("backend") != backend:
                continue
            case = record.get("case")
            raw_samples = record.get("samples")
            correctness = record.get("correctness")
            if not isinstance(case, dict):
                raise ValueError(f"{path}:{line_number}: missing sample case")
            _validate_case_contract(path, line_number, case)
            if not isinstance(raw_samples, list) or not raw_samples:
                raise ValueError(f"{path}:{line_number}: missing raw replay samples")
            if record.get("unit") != "us":
                raise ValueError(
                    f"{path}:{line_number}: replay sample unit must be 'us'"
                )
            if record.get("count") != len(raw_samples):
                raise ValueError(
                    f"{path}:{line_number}: replay sample count field does not "
                    "match the raw sample list"
                )
            if (
                not isinstance(correctness, dict)
                or correctness.get("passed") is not True
            ):
                raise ValueError(
                    f"{path}:{line_number}: correctness was not recorded as passing"
                )
            if require_reference_oracle:
                _validate_reference_correctness(path, line_number, correctness)
            if correctness.get("finite") is False:
                raise ValueError(f"{path}:{line_number}: correctness is non-finite")
            if "nonzero" in correctness and int(correctness["nonzero"]) <= 0:
                raise ValueError(f"{path}:{line_number}: correctness output is zero")
            if correctness.get("allclose") is False:
                raise ValueError(f"{path}:{line_number}: correctness allclose failed")
            metric_values: dict[str, float] = {}
            for metric in (
                "max_abs",
                "relative_l2",
                "cosine",
                "minimum_cosine",
                "maximum_relative_l2",
            ):
                if metric in correctness:
                    try:
                        metric_value = float(correctness[metric])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{path}:{line_number}: invalid correctness {metric}"
                        ) from exc
                    if not math.isfinite(metric_value):
                        raise ValueError(
                            f"{path}:{line_number}: non-finite correctness {metric}"
                        )
                    metric_values[metric] = metric_value
            if (
                "relative_l2" in metric_values
                and "maximum_relative_l2" in metric_values
                and metric_values["relative_l2"] > metric_values["maximum_relative_l2"]
            ):
                raise ValueError(
                    f"{path}:{line_number}: relative L2 exceeds recorded bound"
                )
            if (
                "cosine" in metric_values
                and "minimum_cosine" in metric_values
                and metric_values["cosine"] < metric_values["minimum_cosine"]
            ):
                raise ValueError(
                    f"{path}:{line_number}: cosine is below recorded bound"
                )
            try:
                values = [float(value) for value in raw_samples]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid sample value") from exc
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(
                    f"{path}:{line_number}: samples must be finite and positive"
                )
            key = _case_key(case)
            if key in samples:
                raise ValueError(
                    f"{path}:{line_number}: duplicate case for {backend}: {key}"
                )
            samples[key] = values
            cases[key] = case
    if len(provenance) != 1:
        raise ValueError(
            f"{path}: expected one provenance record, found {len(provenance)}"
        )
    if not samples:
        raise ValueError(f"{path}: found no {backend!r} graph-replay samples")
    _validate_provenance(path, provenance[0])
    serving_mode = provenance[0].get("serving_mode")
    if (
        not isinstance(serving_mode, dict)
        or serving_mode.get("cuda_graph_replay") is not True
    ):
        raise ValueError(f"{path}: provenance does not describe CUDA graph replay")
    if (
        int(serving_mode.get("warmup", 0)) < minimum_warmup
        or int(serving_mode.get("replays", 0)) < minimum_replays
    ):
        raise ValueError(
            f"{path}: requires warmup >= {minimum_warmup} and "
            f"replays >= {minimum_replays}"
        )
    expected_replays = int(serving_mode["replays"])
    wrong_sample_counts = {
        key: len(values)
        for key, values in samples.items()
        if len(values) != expected_replays
    }
    if wrong_sample_counts:
        raise ValueError(
            f"{path}: raw sample counts do not match provenance replays="
            f"{expected_replays}: {wrong_sample_counts}"
        )
    case_contract = provenance[0]["benchmark_case_contract"]
    identity_fields = case_contract["identity_fields"]
    try:
        expected_case_identities = {
            _case_key({field: case[field] for field in identity_fields})
            for case in case_contract["expected"]
        }
        observed_case_identities = {
            _case_key({field: case[field] for field in identity_fields})
            for case in cases.values()
        }
    except KeyError as exc:
        raise ValueError(
            f"{path}: benchmark case contract field {exc.args[0]!r} is missing"
        ) from exc
    if len(expected_case_identities) != len(case_contract["expected"]):
        raise ValueError(f"{path}: benchmark case contract has duplicate identities")
    if expected_case_identities != observed_case_identities:
        missing = sorted(expected_case_identities - observed_case_identities)
        unexpected = sorted(observed_case_identities - expected_case_identities)
        raise ValueError(
            f"{path}: emitted cases do not equal the requested case contract: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return Run(path=path, provenance=provenance[0], samples=samples, cases=cases)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio(current: float, baseline: float) -> float:
    return current / baseline if baseline else math.inf


def _physical_gpu(run: Run) -> tuple[str, str, str, str]:
    gpu = run.provenance.get("gpu")
    if not isinstance(gpu, dict):
        return "", "", "", ""
    return (
        str(gpu.get("physical_index", "")),
        str(gpu.get("name", "")),
        str(gpu.get("uuid", "")),
        json.dumps(gpu.get("capability"), separators=(",", ":")),
    )


def _cutlass_version(run: Run) -> str:
    cutlass = run.provenance.get("cutlass")
    return (
        str(cutlass.get("nvidia-cutlass-dsl", "")) if isinstance(cutlass, dict) else ""
    )


def _parse_expected_cutlass_packages(values: list[str], label: str) -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw in values:
        package, separator, package_version = raw.partition("=")
        if not separator or package not in _CUTLASS_PACKAGES or not package_version:
            raise ValueError(
                f"{label} expects PACKAGE=VERSION for one of {_CUTLASS_PACKAGES}, "
                f"got {raw!r}"
            )
        if package in expected:
            raise ValueError(f"{label} repeats package {package!r}")
        expected[package] = package_version
    if values and set(expected) != set(_CUTLASS_PACKAGES):
        missing = sorted(set(_CUTLASS_PACKAGES) - set(expected))
        raise ValueError(f"{label} is missing packages: {missing}")
    return expected


def _cutlass_map(run: Run) -> dict[str, str]:
    return {
        package: str(run.provenance["cutlass"][package])
        for package in _CUTLASS_PACKAGES
    }


def _normalized_argv(run: Run) -> list[str]:
    raw = run.provenance.get("argv", [])
    normalized: list[str] = []
    skip_next = False
    for argument in raw:
        if skip_next:
            skip_next = False
            continue
        if argument == "--raw-samples-jsonl":
            skip_next = True
            continue
        if argument.startswith("--raw-samples-jsonl="):
            continue
        normalized.append(argument)
    if skip_next:
        raise ValueError(f"{run.path}: --raw-samples-jsonl lacks a path")
    return normalized


def _experiment_signature(run: Run) -> str:
    """Fields that must agree across A and B, excluding CUTLASS itself."""
    return json.dumps(
        {
            "normalized_argv": _normalized_argv(run),
            "benchmark_config": run.provenance.get("benchmark_config"),
            "benchmark_config_sha256": run.provenance.get("benchmark_config_sha256"),
            "benchmark_dependencies_sha256": run.provenance.get(
                "benchmark_dependencies_sha256"
            ),
            "benchmark_case_contract": run.provenance.get("benchmark_case_contract"),
            "python": run.provenance.get("python"),
            "torch": run.provenance.get("torch"),
            "torch_cuda": run.provenance.get("torch_cuda"),
            "runtime_packages": run.provenance.get("runtime_packages"),
            "runtime_environment": _runtime_environment_comparison_payload(run),
            "runtime_environment_comparison_sha256": (
                _runtime_environment_comparison_sha256(run)
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_signature(run: Run) -> str:
    return json.dumps(
        {
            "worktree": run.provenance.get("worktree"),
            "commit": run.provenance.get("commit"),
            "branch": run.provenance.get("branch"),
            "dirty_paths": run.provenance.get("dirty_paths"),
            "sparkinfer_package_fingerprint": run.provenance.get("sparkinfer_package_fingerprint"),
            "benchmark_sha256": run.provenance.get("benchmark_sha256"),
            "experiment": _experiment_signature(run),
            "cutlass": run.provenance.get("cutlass"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a1", type=Path)
    parser.add_argument("b1", type=Path)
    parser.add_argument("b2", type=Path)
    parser.add_argument("a2", type=Path)
    parser.add_argument("--backend", default="sparkinfer")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--require-exact-case-contract",
        type=Path,
        metavar="PATH",
        required=True,
        help=(
            "require every run's benchmark config and exact case set to equal "
            "this external reviewed JSON contract"
        ),
    )
    parser.add_argument(
        "--require-reference-oracle",
        action="store_true",
        help=(
            "require a finite, nonzero torch-reference oracle with allclose, "
            "cosine, relative-L2, and recorded threshold checks for every case"
        ),
    )
    source_mode = parser.add_mutually_exclusive_group(required=True)
    source_mode.add_argument(
        "--require-matching-source-fingerprint",
        action="store_true",
        help="compiler-only mode: require one identical source fingerprint across A/B",
    )
    source_mode.add_argument(
        "--expected-a-source-fingerprint",
        help=(
            "end-to-end mode: require this source fingerprint on A1/A2; also "
            "requires --expected-b-source-fingerprint"
        ),
    )
    parser.add_argument(
        "--expected-b-source-fingerprint",
        help="end-to-end mode: require this source fingerprint on B1/B2",
    )
    parser.add_argument("--expected-a-cutlass-version")
    parser.add_argument("--expected-b-cutlass-version")
    parser.add_argument(
        "--expected-a-cutlass-package",
        action="append",
        default=[],
        metavar="PACKAGE=VERSION",
        help=(
            "exact A package map; repeat once for each of the five CUTLASS "
            "packages and use VERSION=missing for an absent wheel"
        ),
    )
    parser.add_argument(
        "--expected-b-cutlass-package",
        action="append",
        default=[],
        metavar="PACKAGE=VERSION",
        help=(
            "exact B package map; repeat once for each of the five CUTLASS "
            "packages and use VERSION=missing for an absent wheel"
        ),
    )
    parser.add_argument(
        "--allowed-physical-gpu",
        action="append",
        default=[],
        metavar="INDEX",
        help="require the physical GPU index to be one of these values",
    )
    parser.add_argument(
        "--expected-capability",
        default="12.0",
        help="required CUDA compute capability as MAJOR.MINOR (default: 12.0)",
    )
    parser.add_argument(
        "--require-l2-flush",
        action="store_true",
        help="require every run to record L2 eviction before timed replay",
    )
    parser.add_argument(
        "--require-serving-contract",
        action="store_true",
        help=(
            "require stable tensor allocations and fixed/preplanned workspace "
            "capacity in every run"
        ),
    )
    parser.add_argument(
        "--minimum-warmup",
        type=int,
        default=5,
        help="minimum untimed graph warmup iterations required in every run",
    )
    parser.add_argument(
        "--minimum-replays",
        type=int,
        default=100,
        help="minimum raw timed graph replays required per case and run",
    )
    parser.add_argument(
        "--minimum-l2-flush-multiple",
        type=float,
        default=2.0,
        help=(
            "with --require-l2-flush, require the eviction allocation to be "
            "at least this multiple of the recorded physical L2 capacity"
        ),
    )
    parser.add_argument("--max-mean-regression-pct", type=float)
    parser.add_argument("--max-median-regression-pct", type=float)
    parser.add_argument("--max-p95-regression-pct", type=float)
    parser.add_argument(
        "--max-run-mean-drift-pct",
        type=float,
        help=(
            "fail when |A2/A1-1| or |B2/B1-1| exceeds this percentage; "
            "this prevents pooled A-B-B-A results from hiding positional drift"
        ),
    )
    args = parser.parse_args()
    if bool(args.expected_a_source_fingerprint) != bool(
        args.expected_b_source_fingerprint
    ):
        parser.error(
            "--expected-a-source-fingerprint and --expected-b-source-fingerprint "
            "are required together"
        )
    if args.require_matching_source_fingerprint and args.expected_b_source_fingerprint:
        parser.error("compiler-only and end-to-end source modes conflict")
    if args.minimum_warmup < 1:
        parser.error("--minimum-warmup must be positive")
    if args.minimum_replays < 1:
        parser.error("--minimum-replays must be positive")
    if not math.isfinite(args.minimum_l2_flush_multiple) or (
        args.minimum_l2_flush_multiple < 1.0
    ):
        parser.error("--minimum-l2-flush-multiple must be finite and at least 1")
    for option in (
        "max_mean_regression_pct",
        "max_median_regression_pct",
        "max_p95_regression_pct",
        "max_run_mean_drift_pct",
    ):
        value = getattr(args, option)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{option.replace('_', '-')} must be finite and nonnegative")
    try:
        expected_capability = [
            int(part) for part in args.expected_capability.split(".", 1)
        ]
    except ValueError:
        parser.error("--expected-capability must be MAJOR.MINOR")
    if len(expected_capability) != 2 or any(part < 0 for part in expected_capability):
        parser.error("--expected-capability must be MAJOR.MINOR")
    try:
        external_case_contract = _read_external_case_contract(
            args.require_exact_case_contract
        )
        expected_a_cutlass = _parse_expected_cutlass_packages(
            args.expected_a_cutlass_package, "--expected-a-cutlass-package"
        )
        expected_b_cutlass = _parse_expected_cutlass_packages(
            args.expected_b_cutlass_package, "--expected-b-cutlass-package"
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not expected_a_cutlass or not expected_b_cutlass:
        parser.error(
            "exact five-package A and B CUTLASS maps are required for comparison"
        )
    if args.expected_a_cutlass_version is not None and not expected_a_cutlass:
        parser.error(
            "--expected-a-cutlass-version requires the exact five-package A map"
        )
    if args.expected_b_cutlass_version is not None and not expected_b_cutlass:
        parser.error(
            "--expected-b-cutlass-version requires the exact five-package B map"
        )

    try:
        runs = [
            _read_run(
                path,
                args.backend,
                minimum_warmup=args.minimum_warmup,
                minimum_replays=args.minimum_replays,
                require_reference_oracle=args.require_reference_oracle,
            )
            for path in (args.a1, args.b1, args.b2, args.a2)
        ]
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    a1, b1, b2, a2 = runs

    a_fingerprints = {
        str(run.provenance.get("sparkinfer_package_fingerprint", "")) for run in (a1, a2)
    }
    b_fingerprints = {
        str(run.provenance.get("sparkinfer_package_fingerprint", "")) for run in (b1, b2)
    }
    if (
        len(a_fingerprints) != 1
        or not next(iter(a_fingerprints))
        or len(b_fingerprints) != 1
        or not next(iter(b_fingerprints))
    ):
        parser.error("A1/A2 and B1/B2 require uniform nonempty source fingerprints")
    a_fingerprint = next(iter(a_fingerprints))
    b_fingerprint = next(iter(b_fingerprints))
    if args.require_matching_source_fingerprint and a_fingerprint != b_fingerprint:
        parser.error("compiler-only A/B runs have different source fingerprints")
    if args.expected_a_source_fingerprint and (
        a_fingerprint != args.expected_a_source_fingerprint
        or b_fingerprint != args.expected_b_source_fingerprint
    ):
        parser.error(
            "end-to-end source fingerprints differ from the explicit A/B contract: "
            f"A={a_fingerprint!r} B={b_fingerprint!r}"
        )
    benchmark_hashes = {str(run.provenance.get("benchmark_sha256", "")) for run in runs}
    if len(benchmark_hashes) != 1 or not next(iter(benchmark_hashes)):
        parser.error("all A-B-B-A runs must have the same nonempty benchmark SHA256")
    a_environment_hashes = {
        _runtime_environment_raw_sha256(run) for run in (a1, a2)
    }
    b_environment_hashes = {
        _runtime_environment_raw_sha256(run) for run in (b1, b2)
    }
    if len(a_environment_hashes) != 1 or len(b_environment_hashes) != 1:
        parser.error(
            "raw runtime environment provenance must match exactly within "
            "A1/A2 and B1/B2"
        )
    environment_comparison_hashes = {
        _runtime_environment_comparison_sha256(run) for run in runs
    }
    if len(environment_comparison_hashes) != 1:
        parser.error(
            "A-B-B-A runtime environments differ outside the explicit "
            f"operational path exceptions "
            f"{_RUNTIME_ENVIRONMENT_OPERATIONAL_PATH_EXCEPTIONS}"
        )

    if (
        args.expected_a_cutlass_version is not None
        and _cutlass_version(a1) != args.expected_a_cutlass_version
    ):
        parser.error(
            f"A CUTLASS version is {_cutlass_version(a1)!r}, expected "
            f"{args.expected_a_cutlass_version!r}"
        )
    if (
        args.expected_b_cutlass_version is not None
        and _cutlass_version(b1) != args.expected_b_cutlass_version
    ):
        parser.error(
            f"B CUTLASS version is {_cutlass_version(b1)!r}, expected "
            f"{args.expected_b_cutlass_version!r}"
        )
    if expected_a_cutlass and _cutlass_map(a1) != expected_a_cutlass:
        parser.error(
            f"A CUTLASS package map is {_cutlass_map(a1)!r}, expected "
            f"{expected_a_cutlass!r}"
        )
    if expected_b_cutlass and _cutlass_map(b1) != expected_b_cutlass:
        parser.error(
            f"B CUTLASS package map is {_cutlass_map(b1)!r}, expected "
            f"{expected_b_cutlass!r}"
        )

    try:
        a_source_match = _source_signature(a1) == _source_signature(a2)
        b_source_match = _source_signature(b1) == _source_signature(b2)
        experiment_signatures = {_experiment_signature(run) for run in runs}
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if not a_source_match:
        parser.error("A1 and A2 source/toolchain provenance differs")
    if not b_source_match:
        parser.error("B1 and B2 source/toolchain provenance differs")
    if len(experiment_signatures) != 1:
        parser.error(
            "A-B-B-A benchmark configuration, dependencies, argv, runtime "
            "environment, or runtime toolchain differs outside CUTLASS"
        )
    if external_case_contract is not None:
        if external_case_contract["backend"] != args.backend:
            parser.error(
                f"external case contract backend is "
                f"{external_case_contract['backend']!r}, expected {args.backend!r}"
            )
        mismatched_contracts = [
            str(run.path)
            for run in runs
            if _external_case_contract_payload(args.backend, run.provenance)
            != external_case_contract
        ]
        if mismatched_contracts:
            parser.error(
                "runs differ from the external exact case/config contract: "
                + ", ".join(mismatched_contracts)
            )

    gpu_identities = {_physical_gpu(run) for run in runs}
    gpu_identity = next(iter(gpu_identities)) if len(gpu_identities) == 1 else None
    if (
        gpu_identity is None
        or gpu_identity[0] in {"", "None"}
        or not gpu_identity[1]
        or not gpu_identity[2]
        or gpu_identity[3] in {"", "null"}
    ):
        parser.error(
            f"all runs must name the same physical GPU: {sorted(gpu_identities)}"
        )
    if args.allowed_physical_gpu and gpu_identity[0] not in set(
        args.allowed_physical_gpu
    ):
        parser.error(
            f"physical GPU {gpu_identity[0]!r} is not allowed; expected one of "
            f"{sorted(set(args.allowed_physical_gpu))!r}"
        )
    expected_capability_json = json.dumps(expected_capability, separators=(",", ":"))
    if gpu_identity[3] != expected_capability_json:
        parser.error(
            f"GPU capability {gpu_identity[3]} does not match required "
            f"{expected_capability_json}"
        )
    serving_modes = {
        json.dumps(run.provenance["serving_mode"], sort_keys=True) for run in runs
    }
    if len(serving_modes) != 1:
        parser.error("A-B-B-A serving-mode provenance differs")
    if args.require_l2_flush:
        for run in runs:
            serving_mode = run.provenance["serving_mode"]
            gpu = run.provenance.get("gpu")
            if serving_mode.get("l2_flush") is not True:
                parser.error(f"{run.path}: L2 eviction was not enabled")
            if not isinstance(gpu, dict):
                parser.error(f"{run.path}: GPU provenance is missing")
            try:
                flush_bytes = int(serving_mode.get("l2_flush_bytes", 0))
                l2_bytes = int(gpu.get("l2_cache_bytes", 0))
            except (TypeError, ValueError):
                parser.error(f"{run.path}: invalid L2 capacity/flush byte count")
            required_flush_bytes = math.ceil(args.minimum_l2_flush_multiple * l2_bytes)
            if l2_bytes <= 0 or flush_bytes < required_flush_bytes:
                parser.error(
                    f"{run.path}: ineffective L2 eviction: flush_bytes="
                    f"{flush_bytes}, l2_cache_bytes={l2_bytes}, required="
                    f"{required_flush_bytes}"
                )
    if args.require_serving_contract and any(
        run.provenance["serving_mode"].get(field) is not True
        for run in runs
        for field in ("stable_allocations", "fixed_workspace_capacity")
    ):
        parser.error(
            "A-B-B-A runs did not all record stable allocations and fixed "
            "workspace capacity"
        )
    case_sets = [set(run.samples) for run in runs]
    if any(case_set != case_sets[0] for case_set in case_sets[1:]):
        parser.error("A-B-B-A case sets differ")

    fieldnames = [
        "backend",
        "case_json",
        "gpu_physical_index",
        "gpu_name",
        "gpu_uuid",
        "gpu_capability",
        "a_cutlass_version",
        "b_cutlass_version",
        "a_runtime_environment_sha256",
        "b_runtime_environment_sha256",
        "runtime_environment_comparison_sha256",
        "runtime_environment_operational_path_exceptions",
        "samples_a1",
        "samples_b1",
        "samples_b2",
        "samples_a2",
        "a1_mean_us",
        "b1_mean_us",
        "b2_mean_us",
        "a2_mean_us",
        "a_run_mean_drift_pct",
        "b_run_mean_drift_pct",
        "a_mean_us",
        "b_mean_us",
        "mean_ratio_b_over_a",
        "mean_delta_pct",
        "mean_increase",
        "a_median_us",
        "b_median_us",
        "median_ratio_b_over_a",
        "median_delta_pct",
        "median_increase",
        "a_p95_us",
        "b_p95_us",
        "p95_ratio_b_over_a",
        "p95_delta_pct",
        "p95_increase",
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    output = (
        args.output.open("w", newline="", encoding="utf-8")
        if args.output
        else sys.stdout
    )
    mean_failures = 0
    median_failures = 0
    p95_failures = 0
    run_drift_failures = 0
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        gpu_index, gpu_name, gpu_uuid, gpu_capability = _physical_gpu(a1)
        for key in sorted(case_sets[0]):
            a1_values = a1.samples[key]
            b1_values = b1.samples[key]
            b2_values = b2.samples[key]
            a2_values = a2.samples[key]
            a_values = [*a1_values, *a2_values]
            b_values = [*b1_values, *b2_values]
            a_mean = statistics.fmean(a_values)
            b_mean = statistics.fmean(b_values)
            a1_mean = statistics.fmean(a1_values)
            b1_mean = statistics.fmean(b1_values)
            b2_mean = statistics.fmean(b2_values)
            a2_mean = statistics.fmean(a2_values)
            a_run_mean_drift_pct = (_ratio(a2_mean, a1_mean) - 1.0) * 100.0
            b_run_mean_drift_pct = (_ratio(b2_mean, b1_mean) - 1.0) * 100.0
            a_median = statistics.median(a_values)
            b_median = statistics.median(b_values)
            a_p95 = _percentile(a_values, 0.95)
            b_p95 = _percentile(b_values, 0.95)
            mean_delta_pct = (_ratio(b_mean, a_mean) - 1.0) * 100.0
            median_delta_pct = (_ratio(b_median, a_median) - 1.0) * 100.0
            p95_delta_pct = (_ratio(b_p95, a_p95) - 1.0) * 100.0
            mean_failures += bool(
                args.max_mean_regression_pct is not None
                and mean_delta_pct > args.max_mean_regression_pct
            )
            median_failures += bool(
                args.max_median_regression_pct is not None
                and median_delta_pct > args.max_median_regression_pct
            )
            p95_failures += bool(
                args.max_p95_regression_pct is not None
                and p95_delta_pct > args.max_p95_regression_pct
            )
            run_drift_failures += bool(
                args.max_run_mean_drift_pct is not None
                and (
                    abs(a_run_mean_drift_pct) > args.max_run_mean_drift_pct
                    or abs(b_run_mean_drift_pct) > args.max_run_mean_drift_pct
                )
            )
            writer.writerow(
                {
                    "backend": args.backend,
                    "case_json": key,
                    "gpu_physical_index": gpu_index,
                    "gpu_name": gpu_name,
                    "gpu_uuid": gpu_uuid,
                    "gpu_capability": gpu_capability,
                    "a_cutlass_version": _cutlass_version(a1),
                    "b_cutlass_version": _cutlass_version(b1),
                    "a_runtime_environment_sha256": _runtime_environment_raw_sha256(
                        a1
                    ),
                    "b_runtime_environment_sha256": _runtime_environment_raw_sha256(
                        b1
                    ),
                    "runtime_environment_comparison_sha256": next(
                        iter(environment_comparison_hashes)
                    ),
                    "runtime_environment_operational_path_exceptions": json.dumps(
                        _RUNTIME_ENVIRONMENT_OPERATIONAL_PATH_EXCEPTIONS,
                        separators=(",", ":"),
                    ),
                    "samples_a1": len(a1_values),
                    "samples_b1": len(b1_values),
                    "samples_b2": len(b2_values),
                    "samples_a2": len(a2_values),
                    "a1_mean_us": a1_mean,
                    "b1_mean_us": b1_mean,
                    "b2_mean_us": b2_mean,
                    "a2_mean_us": a2_mean,
                    "a_run_mean_drift_pct": a_run_mean_drift_pct,
                    "b_run_mean_drift_pct": b_run_mean_drift_pct,
                    "a_mean_us": a_mean,
                    "b_mean_us": b_mean,
                    "mean_ratio_b_over_a": _ratio(b_mean, a_mean),
                    "mean_delta_pct": mean_delta_pct,
                    "mean_increase": b_mean > a_mean,
                    "a_median_us": a_median,
                    "b_median_us": b_median,
                    "median_ratio_b_over_a": _ratio(b_median, a_median),
                    "median_delta_pct": median_delta_pct,
                    "median_increase": b_median > a_median,
                    "a_p95_us": a_p95,
                    "b_p95_us": b_p95,
                    "p95_ratio_b_over_a": _ratio(b_p95, a_p95),
                    "p95_delta_pct": p95_delta_pct,
                    "p95_increase": b_p95 > a_p95,
                }
            )
    finally:
        if args.output:
            output.close()

    print(
        f"cases={len(case_sets[0])} gpu={_physical_gpu(a1)} "
        f"A={_cutlass_version(a1)} B={_cutlass_version(b1)} "
        f"runtime_environment={next(iter(environment_comparison_hashes))} "
        f"mean_threshold_failures={mean_failures} "
        f"median_threshold_failures={median_failures} "
        f"p95_threshold_failures={p95_failures} "
        f"run_drift_failures={run_drift_failures}",
        file=sys.stderr,
    )
    return int(
        bool(mean_failures or median_failures or p95_failures or run_drift_failures)
    )


if __name__ == "__main__":
    raise SystemExit(main())
