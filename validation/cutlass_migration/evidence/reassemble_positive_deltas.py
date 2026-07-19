#!/usr/bin/env python3
"""Reassemble every positive CuTe resource delta with one common PTXAS.

The normal corpus compares the exact cubins produced by each CUTLASS DSL
frontend.  Those cubins can name different CUTLASS-owned PTXAS builds.  This
tool is the causal follow-up: it validates the cache-key-bound frontend PTX for
both arms, selects the union of all positive resource and exact R/UR/P/UP
deltas, assembles both arms with the same requested PTXAS executable and flags,
and audits the resulting cubins without loading them on a GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from validation.cutlass_migration.acceptance.corpus.ptx_capture import (
    _ptxas_replay_argv,
)
from validation.cutlass_migration.evidence.kernel_resources import (
    _INSTRUCTION_RE,
    _LOCAL_LOAD_RE,
    _LOCAL_STORE_RE,
    _TEXT_KERNEL_SECTION_RE,
    _attribute_short,
    _attribute_words,
    _cubin_shared_section_bytes,
    _kernel_code,
    _kernel_info,
    _ptxas_metadata,
    _resource_values,
)
from validation.cutlass_migration.evidence.sass_register_sets import (
    _REGISTER_PATTERNS,
    _register_indices,
    _register_reconfiguration,
    _set_fields,
)
_SCHEMA = "b12x.cute.common_ptxas_positive_delta.v2"
_SUMMARY_SCHEMA = "b12x.cute.common_ptxas_positive_delta_summary.v2"
_RESOURCE_DELTA_SCHEMA = "b12x.cute.kernel_resource_delta.v4"
_SASS_DELTA_SCHEMA = "b12x.cute.sass_register_set_delta.v3"
_PTX_SIDECAR_SCHEMA = "b12x.cute.frontend_ptx.v3"
_MANIFEST_SCHEMA = "b12x.cute.compile_manifest.v3"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARCH_RE = re.compile(r"^\s*\.target\s+(\S+)", re.MULTILINE)
_CODE_POSITIVE_FIELDS = ("sass_instruction_delta", "code_bytes_delta")
_SPECIAL_POSITIVE_FIELDS = (
    "any_register_usage_increase",
    "driver_occupancy_decrease",
    "new_local_memory",
    "new_register_ceiling",
)
_SCALAR_METRICS = (
    "allocated_registers",
    "eiattr_max_register_count",
    "effective_register_index_ceiling",
    "register_reconfiguration_instruction_count",
    "register_reconfiguration_min_target",
    "register_reconfiguration_max_target",
    "frame_bytes",
    "min_stack_bytes",
    "local_load_instructions",
    "local_store_instructions",
    "cubin_shared_section_bytes",
    "threads_x",
    "threads_y",
    "threads_z",
    "threads_per_cta",
    "parameter_bytes",
    "sass_instructions",
    "code_bytes",
)


class EvidenceError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc


def _compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _read_csv(path: Path, schema_field: str, schema: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = [
                {key: (value or "").strip() for key, value in row.items()}
                for row in reader
            ]
    except OSError as exc:
        raise EvidenceError(f"cannot read CSV {path}: {exc}") from exc
    if not rows:
        raise EvidenceError(f"CSV is empty: {path}")
    if schema_field not in rows[0]:
        raise EvidenceError(f"{path} has no {schema_field} column")
    bad = [index for index, row in enumerate(rows, 2) if row[schema_field] != schema]
    if bad:
        raise EvidenceError(f"{path} has invalid {schema_field} on rows {bad}")
    return rows


def _true(value: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise EvidenceError(f"invalid boolean {value!r}")
    return value.lower() == "true"


def _positive_reasons(resource: dict[str, str], sass: dict[str, str]) -> list[str]:
    reasons: set[str] = set()
    for prefix, row in (("resource", resource), ("sass", sass)):
        for field, value in row.items():
            if field.endswith("_increase") and value:
                if _true(value):
                    reasons.add(f"{prefix}:{field}")
        for field in _SPECIAL_POSITIVE_FIELDS:
            value = row.get(field, "")
            if value and _true(value):
                reasons.add(f"{prefix}:{field}")
    for field in _CODE_POSITIVE_FIELDS:
        raw = resource.get(field, "")
        try:
            positive = bool(raw) and int(raw) > 0
        except ValueError as exc:
            raise EvidenceError(f"invalid resource delta {field}={raw!r}") from exc
        if positive:
            reasons.add(f"resource:{field}")
    return sorted(reasons)


def _identity(row: dict[str, str], *, kernel_field: str) -> tuple[str, str]:
    comparison_key = row.get("comparison_semantic_key", "")
    kernel = row.get(kernel_field, "")
    if not _SHA256_RE.fullmatch(comparison_key) or not kernel:
        raise EvidenceError(f"invalid comparison identity {(comparison_key, kernel)!r}")
    return comparison_key, kernel


def _pair_delta_rows(
    resource_rows: list[dict[str, str]], sass_rows: list[dict[str, str]]
) -> list[tuple[dict[str, str], dict[str, str], list[str]]]:
    resources: dict[tuple[str, str], dict[str, str]] = {}
    for row in resource_rows:
        if row.get("pairing") != "exact-comparison-semantic-kernel":
            raise EvidenceError(f"resource delta is not exactly paired: {row}")
        if row.get("baseline_kernel") != row.get("current_kernel"):
            raise EvidenceError("resource delta changed exact CUDA symbol")
        key = _identity(row, kernel_field="current_kernel")
        if key in resources:
            raise EvidenceError(f"duplicate resource delta identity {key!r}")
        if row.get("symbol_sha256") != _sha256_bytes(key[1].encode("utf-8")):
            raise EvidenceError(f"resource delta symbol hash mismatch for {key!r}")
        resources[key] = row

    sass: dict[tuple[str, str], dict[str, str]] = {}
    for row in sass_rows:
        key = _identity(row, kernel_field="kernel")
        if key in sass:
            raise EvidenceError(f"duplicate SASS delta identity {key!r}")
        sass[key] = row
    if set(resources) != set(sass):
        raise EvidenceError(
            "resource/SASS delta identity sets differ: "
            f"missing={sorted(set(resources) - set(sass))!r} "
            f"unexpected={sorted(set(sass) - set(resources))!r}"
        )

    selected = []
    for key in sorted(resources):
        resource = resources[key]
        sass_row = sass[key]
        for side in ("baseline", "current"):
            if resource.get(f"{side}_object_sha256") != sass_row.get(
                f"{side}_object_sha256"
            ):
                raise EvidenceError(f"{key!r}: {side} object hash differs by report")
        reasons = _positive_reasons(resource, sass_row)
        if reasons:
            selected.append((resource, sass_row, reasons))
    return selected


def _cache_paths(cache_dir: Path, cache_key: str) -> dict[str, Path]:
    if not _SHA256_RE.fullmatch(cache_key):
        raise EvidenceError(f"invalid cache key {cache_key!r}")
    base = cache_dir / cache_key[:2] / cache_key
    return {
        "object": base.with_suffix(".o"),
        "manifest": base.with_suffix(".json"),
        "ptx": base.with_suffix(".ptx"),
        "sidecar": base.with_suffix(".ptx.json"),
    }


def _tool_record(path: Path) -> dict[str, str]:
    if not path.is_absolute() or not path.is_file():
        raise EvidenceError(f"tool must be an existing absolute path: {path}")
    result = subprocess.run(
        [str(path), "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "executable": str(path),
        "realpath": str(path.resolve()),
        "sha256": _sha256(path.resolve()),
        "version_output": result.stdout.strip(),
        "version_output_sha256": _sha256_bytes(result.stdout.strip().encode("utf-8")),
    }


def _load_bound_ptx(
    cache_dir: Path,
    cache_key: str,
    row: dict[str, str],
    side: str,
    common_ptxas: dict[str, str],
) -> dict[str, Any]:
    paths = _cache_paths(cache_dir, cache_key)
    try:
        object_bytes = paths["object"].read_bytes()
        manifest_bytes = paths["manifest"].read_bytes()
        ptx_bytes = paths["ptx"].read_bytes()
    except OSError as exc:
        raise EvidenceError(f"missing bound artifact for {cache_key}: {exc}") from exc
    manifest = _read_json(paths["manifest"])
    sidecar = _read_json(paths["sidecar"])
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != _MANIFEST_SCHEMA
        or manifest.get("cache_key") != cache_key
    ):
        raise EvidenceError(f"invalid compile manifest for {cache_key}")
    if (
        not isinstance(sidecar, dict)
        or sidecar.get("schema") != _PTX_SIDECAR_SCHEMA
        or sidecar.get("cache_key") != cache_key
    ):
        raise EvidenceError(f"invalid frontend PTX sidecar for {cache_key}")
    raw_compile_spec_version = row.get(f"{side}_compile_spec_version")
    try:
        row_compile_spec_version = int(raw_compile_spec_version or "")
    except ValueError as exc:
        raise EvidenceError(
            f"{cache_key}: invalid {side} compile-spec version "
            f"{raw_compile_spec_version!r}"
        ) from exc
    expected_object_sha = row.get(f"{side}_object_sha256")
    checks = {
        "resource-object": expected_object_sha == _sha256_bytes(object_bytes),
        "manifest-object": manifest.get("object_sha256") == expected_object_sha,
        "sidecar-object": sidecar.get("object", {}).get("sha256")
        == expected_object_sha,
        "manifest-sha": sidecar.get("compile_manifest", {}).get("sha256")
        == _sha256_bytes(manifest_bytes),
        "ptx-sha": sidecar.get("ptx", {}).get("sha256") == _sha256_bytes(ptx_bytes),
        "semantic-key": sidecar.get("semantic_key")
        == row.get(f"{side}_semantic_key")
        == manifest.get("semantic_key"),
        "comparison-semantic-key": sidecar.get("comparison_semantic_key")
        == row.get("comparison_semantic_key"),
        "target": sidecar.get("target")
        == row.get(f"{side}_target")
        == manifest.get("target"),
        "kernel-id": sidecar.get("kernel_id")
        == row.get(f"{side}_kernel_id")
        == manifest.get("kernel_id"),
        "compile-spec-version": sidecar.get("compile_spec_version")
        == row_compile_spec_version
        == manifest.get("compile_spec_version"),
        "compile-spec-hash": sidecar.get("compile_spec_hash")
        == row.get(f"{side}_compile_spec_hash")
        == manifest.get("compile_spec_hash"),
        "compile-spec-json": sidecar.get("compile_spec_json")
        == row.get(f"{side}_compile_spec_json")
        == manifest.get("compile_spec_json"),
        "compile-kwargs-json": sidecar.get("compile_kwargs_json")
        == row.get(f"{side}_compile_kwargs_json")
        == manifest.get("compile_kwargs_json"),
        "package-fingerprint": sidecar.get("package_fingerprint")
        == row.get(f"{side}_package_fingerprint")
        == manifest.get("package_fingerprint"),
        "toolchain": sidecar.get("toolchain") == manifest.get("toolchain")
        and _compact_json(sidecar.get("toolchain"))
        == row.get(f"{side}_toolchain_json"),
        "compile-options": sidecar.get("compile_options")
        == manifest.get("compile_options")
        and _compact_json(sidecar.get("compile_options"))
        == row.get(f"{side}_compile_options_json"),
        "compile-environment": sidecar.get("compile_environment")
        == manifest.get("compile_environment")
        and _compact_json(sidecar.get("compile_environment"))
        == row.get(f"{side}_compile_environment_json"),
        "source-ptxas-version": sidecar.get("source_ptxas", {}).get("version")
        == row.get(f"{side}_ptxas_version"),
        "source-ptxas-flags": sidecar.get("source_ptxas", {}).get("flags")
        == row.get(f"{side}_ptxas_flags"),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise EvidenceError(f"{cache_key}: PTX provenance checks failed: {failed}")
    entrypoints = sidecar.get("ptx", {}).get("entrypoints")
    binding = sidecar.get("entrypoint_binding", {})
    kernel = row[f"{side}_kernel"]
    if (
        not isinstance(entrypoints, list)
        or not entrypoints
        or binding.get("status") != "exact"
        or entrypoints != binding.get("ptx_entrypoints")
        or entrypoints != binding.get("embedded_cubin_entrypoints")
        or kernel not in entrypoints
    ):
        raise EvidenceError(f"{cache_key}: no exact PTX binding for {kernel}")
    environment = sidecar.get("compile_environment")
    if isinstance(environment, list) and any(
        isinstance(entry, list)
        and entry
        and entry[0] in {"CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR"}
        for entry in environment
    ):
        raise EvidenceError(f"{cache_key}: dump controls leaked into semantic identity")
    recorded_common = sidecar.get("common_ptxas", {})
    for field in ("executable", "realpath", "sha256", "version_output_sha256"):
        if recorded_common.get(field) != common_ptxas[field]:
            raise EvidenceError(
                f"{cache_key}: recorded common PTXAS {field} differs from requested tool"
            )
    source_ptxas = sidecar.get("source_ptxas", {})
    try:
        replay_argv = _ptxas_replay_argv(str(source_ptxas.get("flags", "")))
    except RuntimeError as exc:
        raise EvidenceError(f"{cache_key}: invalid source PTXAS flags: {exc}") from exc
    if source_ptxas.get("flags_argv") != replay_argv:
        raise EvidenceError(f"{cache_key}: invalid recorded PTXAS replay argv")
    expected_template = [
        common_ptxas["executable"],
        *replay_argv,
        "{input_ptx}",
        "-o",
        "{output_cubin}",
    ]
    if recorded_common.get("command_argv_template") != expected_template:
        raise EvidenceError(f"{cache_key}: invalid common PTXAS command template")
    return {
        "cache_key": cache_key,
        "paths": {key: str(value) for key, value in paths.items()},
        "ptx_path": paths["ptx"],
        "ptx_sha256": _sha256_bytes(ptx_bytes),
        "ptx_bytes": len(ptx_bytes),
        "entrypoints": entrypoints,
        "source_ptxas": source_ptxas,
        "object_sha256": expected_object_sha,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "semantic_key": sidecar["semantic_key"],
        "comparison_semantic_key": sidecar["comparison_semantic_key"],
        "compile_spec_hash": sidecar["compile_spec_hash"],
        "toolchain": sidecar.get("toolchain"),
    }


def _audit_cubin(path: Path, nvdisasm: Path) -> dict[str, Any]:
    cubin_bytes = path.read_bytes()
    result = subprocess.run(
        [str(nvdisasm), str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    disassembly = result.stdout
    architecture_match = _ARCH_RE.search(disassembly)
    if architecture_match is None:
        raise EvidenceError(f"nvdisasm omitted architecture for {path}")
    ptxas_version, ptxas_flags = _ptxas_metadata(disassembly)
    kernels = sorted(
        {
            match.group("kernel")
            for match in _TEXT_KERNEL_SECTION_RE.finditer(disassembly)
        }
    )
    registers = _resource_values(disassembly, "EIATTR_REGCOUNT")
    frames = _resource_values(disassembly, "EIATTR_FRAME_SIZE")
    stacks = _resource_values(disassembly, "EIATTR_MIN_STACK_SIZE")
    if not kernels or set(registers) != set(kernels):
        raise EvidenceError(f"invalid CUDA entry-point/resource set in {path}")
    output: dict[str, Any] = {}
    for kernel in kernels:
        code = _kernel_code(disassembly, kernel)
        info = _kernel_info(disassembly, kernel)
        reqntid = _attribute_words(info, "EIATTR_REQNTID")
        offsets = [
            int(match.group("offset"), 16) for match in _INSTRUCTION_RE.finditer(code)
        ]
        if len(reqntid) != 3 or not code or not info or not offsets:
            raise EvidenceError(f"incomplete disassembly for {path}:{kernel}")
        reconfiguration = _register_reconfiguration(disassembly, kernel, code)
        allocated = registers[kernel]
        effective = max(
            allocated, int(reconfiguration["register_reconfiguration_max_target"])
        )
        register_fields: dict[str, Any] = {}
        for family in _REGISTER_PATTERNS:
            indices = _register_indices(code, family)
            if family == "r" and indices and indices[-1] >= effective:
                raise EvidenceError(
                    f"{path}:{kernel}: R{indices[-1]} exceeds effective ceiling {effective}"
                )
            register_fields.update(_set_fields(family, indices))
        output[kernel] = {
            "allocated_registers": allocated,
            "eiattr_max_register_count": _attribute_short(info, "EIATTR_MAXREG_COUNT"),
            **reconfiguration,
            "effective_register_index_ceiling": effective,
            "frame_bytes": frames.get(kernel, 0),
            "min_stack_bytes": stacks.get(kernel, 0),
            "local_load_instructions": len(_LOCAL_LOAD_RE.findall(code)),
            "local_store_instructions": len(_LOCAL_STORE_RE.findall(code)),
            "cubin_shared_section_bytes": _cubin_shared_section_bytes(
                disassembly, kernel
            ),
            "threads_x": reqntid[0],
            "threads_y": reqntid[1],
            "threads_z": reqntid[2],
            "threads_per_cta": reqntid[0] * reqntid[1] * reqntid[2],
            "parameter_bytes": _attribute_short(info, "EIATTR_CBANK_PARAM_SIZE"),
            "sass_instructions": len(offsets),
            "code_bytes": max(offsets) + 16,
            **register_fields,
        }
    return {
        "path": str(path),
        "sha256": _sha256_bytes(cubin_bytes),
        "bytes": len(cubin_bytes),
        "architecture": architecture_match.group(1),
        "ptxas_version": ptxas_version,
        "ptxas_flags": ptxas_flags,
        "entrypoints": kernels,
        "kernels": output,
        "nvdisasm_stdout_sha256": _sha256_bytes(disassembly.encode("utf-8")),
    }


def _assemble_module(
    bound: dict[str, Any],
    output_path: Path,
    ptxas: Path,
    nvdisasm: Path,
    expected_flags: list[str],
    expected_raw_flags: str,
    common_ptxas: dict[str, str],
) -> dict[str, Any]:
    command = [
        str(ptxas),
        *expected_flags,
        str(bound["ptx_path"]),
        "-o",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        raise EvidenceError(
            f"common PTXAS failed with exit {result.returncode}: "
            f"argv={command!r} output={result.stdout!r}"
        )
    audit = _audit_cubin(output_path, nvdisasm)
    if audit["entrypoints"] != bound["entrypoints"]:
        raise EvidenceError(
            f"common PTXAS entry points differ for {bound['cache_key']}: "
            f"source={bound['entrypoints']!r} common={audit['entrypoints']!r}"
        )
    if audit["ptxas_flags"] != expected_raw_flags:
        raise EvidenceError(
            f"common cubin PTXAS flags differ for {bound['cache_key']}: "
            f"expected={expected_raw_flags!r} observed={audit['ptxas_flags']!r}"
        )
    if audit["ptxas_version"] not in common_ptxas["version_output"]:
        raise EvidenceError(
            f"common cubin PTXAS version differs for {bound['cache_key']}: "
            f"{audit['ptxas_version']!r}"
        )
    return {
        "cache_key": bound["cache_key"],
        "ptx_sha256": bound["ptx_sha256"],
        "command_argv": command,
        "command_stdout": result.stdout,
        "command_stdout_sha256": _sha256_bytes(result.stdout.encode("utf-8")),
        "audit": audit,
    }


def _delta_metrics(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in _SCALAR_METRICS:
        old = int(baseline[field])
        new = int(current[field])
        result[field] = {"baseline": old, "current": new, "delta": new - old}
    register_usage_increase = False
    for family in _REGISTER_PATTERNS:
        indices_field = f"{family}_indices_json"
        old_indices = json.loads(str(baseline[indices_field]))
        new_indices = json.loads(str(current[indices_field]))
        family_record: dict[str, Any] = {
            "baseline_indices": old_indices,
            "current_indices": new_indices,
            "added_indices": sorted(set(new_indices) - set(old_indices)),
            "removed_indices": sorted(set(old_indices) - set(new_indices)),
        }
        for metric in ("count", "min", "max", "span", "index_span"):
            old = int(baseline[f"{family}_{metric}"])
            new = int(current[f"{family}_{metric}"])
            family_record[metric] = {
                "baseline": old,
                "current": new,
                "delta": new - old,
            }
            if metric in {"count", "max", "span", "index_span"} and new > old:
                register_usage_increase = True
        result[family] = family_record
    if int(current["allocated_registers"]) > int(baseline["allocated_registers"]):
        register_usage_increase = True
    result["any_register_usage_increase"] = register_usage_increase
    result["any_local_memory_increase"] = any(
        int(current[field]) > int(baseline[field])
        for field in (
            "frame_bytes",
            "min_stack_bytes",
            "local_load_instructions",
            "local_store_instructions",
        )
    )
    result["any_static_shared_memory_increase"] = int(
        current["cubin_shared_section_bytes"]
    ) > int(baseline["cubin_shared_section_bytes"])
    return result


def _flatten_row(record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "common_ptxas_delta_schema": _SCHEMA,
        "comparison_semantic_key": record["comparison_semantic_key"],
        "baseline_semantic_key": record["baseline_semantic_key"],
        "current_semantic_key": record["current_semantic_key"],
        "kernel": record["kernel"],
        "symbol_sha256": record["symbol_sha256"],
        "selection_reasons_json": json.dumps(
            record["selection_reasons"], separators=(",", ":")
        ),
        "baseline_cache_key": record["baseline"]["cache_key"],
        "current_cache_key": record["current"]["cache_key"],
        "baseline_ptx_sha256": record["baseline"]["ptx_sha256"],
        "current_ptx_sha256": record["current"]["ptx_sha256"],
        "common_ptxas_sha256": record["common_ptxas"]["sha256"],
        "common_ptxas_version_output_sha256": record["common_ptxas"][
            "version_output_sha256"
        ],
        "common_ptxas_flags_json": json.dumps(
            record["common_ptxas_flags"], separators=(",", ":")
        ),
        "baseline_common_cubin_sha256": record["baseline"]["common_cubin"]["audit"][
            "sha256"
        ],
        "current_common_cubin_sha256": record["current"]["common_cubin"]["audit"][
            "sha256"
        ],
        "common_any_register_usage_increase": str(
            record["common_delta"]["any_register_usage_increase"]
        ).lower(),
        "common_any_local_memory_increase": str(
            record["common_delta"]["any_local_memory_increase"]
        ).lower(),
        "common_any_static_shared_memory_increase": str(
            record["common_delta"]["any_static_shared_memory_increase"]
        ).lower(),
    }
    for metric in _SCALAR_METRICS:
        values = record["common_delta"][metric]
        row[f"baseline_common_{metric}"] = values["baseline"]
        row[f"current_common_{metric}"] = values["current"]
        row[f"common_{metric}_delta"] = values["delta"]
    for family in _REGISTER_PATTERNS:
        values = record["common_delta"][family]
        for name in (
            "baseline_indices",
            "current_indices",
            "added_indices",
            "removed_indices",
        ):
            row[f"common_{family}_{name}_json"] = json.dumps(
                values[name], separators=(",", ":")
            )
        for metric in ("count", "min", "max", "span", "index_span"):
            row[f"baseline_common_{family}_{metric}"] = values[metric]["baseline"]
            row[f"current_common_{family}_{metric}"] = values[metric]["current"]
            row[f"common_{family}_{metric}_delta"] = values[metric]["delta"]
    return row


def _ensure_empty(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise EvidenceError(f"output directory must be verified empty: {path}")
    else:
        path.mkdir(parents=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-delta", type=Path, required=True)
    parser.add_argument("--sass-delta", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--current-cache", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path, default=Path("/opt/cuda/bin/ptxas"))
    parser.add_argument("--nvdisasm", type=Path, default=Path("/opt/cuda/bin/nvdisasm"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        resource_delta = args.resource_delta.resolve()
        sass_delta = args.sass_delta.resolve()
        baseline_cache = args.baseline_cache.resolve()
        current_cache = args.current_cache.resolve()
        ptxas = args.ptxas.resolve()
        nvdisasm = args.nvdisasm.resolve()
        output_dir = args.output_dir.resolve()
        _ensure_empty(output_dir)
        baseline_dir = output_dir / "baseline"
        current_dir = output_dir / "current"
        baseline_dir.mkdir()
        current_dir.mkdir()

        resource_rows = _read_csv(
            resource_delta, "delta_report_schema", _RESOURCE_DELTA_SCHEMA
        )
        sass_rows = _read_csv(
            sass_delta, "sass_register_set_delta_schema", _SASS_DELTA_SCHEMA
        )
        selected = _pair_delta_rows(resource_rows, sass_rows)
        common_ptxas = _tool_record(ptxas)
        nvdisasm_record = _tool_record(nvdisasm)
        modules: dict[tuple[str, str], dict[str, Any]] = {}
        records: list[dict[str, Any]] = []

        for resource, sass, reasons in selected:
            del sass
            comparison_semantic_key = resource["comparison_semantic_key"]
            kernel = resource["current_kernel"]
            bound: dict[str, dict[str, Any]] = {}
            for side, cache_dir in (
                ("baseline", baseline_cache),
                ("current", current_cache),
            ):
                cache_key = resource[f"{side}_cache_key"]
                bound[side] = _load_bound_ptx(
                    cache_dir, cache_key, resource, side, common_ptxas
                )
            baseline_flags = bound["baseline"]["source_ptxas"]["flags_argv"]
            current_flags = bound["current"]["source_ptxas"]["flags_argv"]
            baseline_raw_flags = bound["baseline"]["source_ptxas"]["flags"]
            current_raw_flags = bound["current"]["source_ptxas"]["flags"]
            if baseline_flags != current_flags:
                raise EvidenceError(
                    f"{(comparison_semantic_key, kernel)!r}: source PTXAS flags differ: "
                    f"{baseline_flags!r} != {current_flags!r}"
                )
            if baseline_raw_flags != current_raw_flags:
                raise EvidenceError(
                    f"{(comparison_semantic_key, kernel)!r}: source PTXAS raw flags differ: "
                    f"{baseline_raw_flags!r} != {current_raw_flags!r}"
                )
            if not isinstance(baseline_flags, list) or any(
                not isinstance(value, str) or not value for value in baseline_flags
            ):
                raise EvidenceError("invalid PTXAS flag argv in capture sidecar")

            assembled: dict[str, dict[str, Any]] = {}
            for side, directory in (
                ("baseline", baseline_dir),
                ("current", current_dir),
            ):
                cache_key = bound[side]["cache_key"]
                module_key = (side, cache_key)
                if module_key not in modules:
                    modules[module_key] = _assemble_module(
                        bound[side],
                        directory / f"{cache_key}.cubin",
                        ptxas,
                        nvdisasm,
                        baseline_flags,
                        baseline_raw_flags,
                        common_ptxas,
                    )
                assembled[side] = modules[module_key]
            baseline_kernel = assembled["baseline"]["audit"]["kernels"].get(kernel)
            current_kernel = assembled["current"]["audit"]["kernels"].get(kernel)
            if baseline_kernel is None or current_kernel is None:
                raise EvidenceError(f"common PTXAS omitted selected kernel {kernel}")
            records.append(
                {
                    "schema": _SCHEMA,
                    "comparison_semantic_key": comparison_semantic_key,
                    "baseline_semantic_key": resource["baseline_semantic_key"],
                    "current_semantic_key": resource["current_semantic_key"],
                    "kernel": kernel,
                    "symbol_sha256": resource["symbol_sha256"],
                    "selection_reasons": reasons,
                    "common_ptxas": common_ptxas,
                    "common_ptxas_flags": baseline_flags,
                    "baseline": {
                        **{
                            key: value
                            for key, value in bound["baseline"].items()
                            if key != "ptx_path"
                        },
                        "common_cubin": assembled["baseline"],
                    },
                    "current": {
                        **{
                            key: value
                            for key, value in bound["current"].items()
                            if key != "ptx_path"
                        },
                        "common_cubin": assembled["current"],
                    },
                    "common_delta": _delta_metrics(baseline_kernel, current_kernel),
                }
            )

        summary = {
            "schema": _SUMMARY_SCHEMA,
            "reassembler": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "resource_delta": {
                "path": str(resource_delta),
                "sha256": _sha256(resource_delta),
            },
            "sass_delta": {"path": str(sass_delta), "sha256": _sha256(sass_delta)},
            "baseline_cache": str(baseline_cache),
            "current_cache": str(current_cache),
            "common_ptxas": common_ptxas,
            "nvdisasm": nvdisasm_record,
            "input_identity_count": len(resource_rows),
            "selected_positive_identity_count": len(records),
            "assembled_module_count": len(modules),
            "records": records,
        }
        json_path = output_dir / "common-ptxas-positive-deltas.json"
        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        csv_path = output_dir / "common-ptxas-positive-deltas.csv"
        flat_rows = [_flatten_row(record) for record in records]
        if flat_rows:
            with csv_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=list(flat_rows[0]))
                writer.writeheader()
                writer.writerows(flat_rows)
        else:
            csv_path.write_text("common_ptxas_delta_schema\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "selected_positive_identity_count": len(records),
                    "assembled_module_count": len(modules),
                    "json": str(json_path),
                    "csv": str(csv_path),
                },
                indent=2,
            )
        )
        return 0
    except (EvidenceError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
