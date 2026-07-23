#!/usr/bin/env python3
"""Audit SM120 CuTe DSL objects for registers and thread-local memory.

Run real GPU workloads with ``SPARKINFER_CUTE_COMPILE_CACHE_DIR`` pointed at a fresh
directory, then pass that directory here.  The sparkinfer object cache stores the
compiled CUDA ELF inside each host object, so this audit reads the exact cubins
that the workload launched instead of recompiling approximations.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from validation.cutlass_migration.core.comparison_identity import (
    comparison_semantic_key_from_manifest,
)
from validation.cutlass_migration.paths import EVIDENCE_ROOT


_CUDA_ELF_MAGIC = b"\x7fELF\x02\x01\x01\x41"
_TOOL_TIMEOUT_SECONDS = 60
_LOCAL_LOAD_RE = re.compile(r"\bLDL(?:\.[A-Z0-9]+)*\b")
_LOCAL_STORE_RE = re.compile(r"\bSTL(?:\.[A-Z0-9]+)*\b")
_UNIFORM_REGISTER_RE = re.compile(r"\bUR(?P<index>[0-9]+)\b")
_PREDICATE_REGISTER_RE = re.compile(r"\bP(?P<index>[0-9]+)\b")
_UNIFORM_PREDICATE_REGISTER_RE = re.compile(r"\bUP(?P<index>[0-9]+)\b")
_INSTRUCTION_RE = re.compile(r"^\s*/\*(?P<offset>[0-9a-fA-F]+)\*/.*;\s*$", re.MULTILINE)
_TEXT_KERNEL_SECTION_RE = re.compile(
    r"^//--------------------- \.text\.(?P<kernel>kernel_cutlass_kernel_\S+)",
    re.MULTILINE,
)
_CACHE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_SCHEMAS = frozenset(
    {
        # Current compiler namespace.
        "sparkinfer._lib.compile_manifest.v3",
        # Preserve auditability of manifests emitted before the compiler moved
        # from ``sparkinfer.cute`` to ``sparkinfer._lib``.
        "sparkinfer.cute.compile_manifest.v3",
    }
)
_RESOURCE_REPORT_SCHEMA = "sparkinfer.cute.kernel_resources.v4"
_CONTRACT_METADATA_SCHEMA = "sparkinfer.cute.resource_row_contract.v2"
_SPECIALIZATION_CONTRACT_FIELDS = [
    "kernel_id",
    "compile_spec_version",
    "compile_spec_hash",
    "compile_spec_json",
    "semantic_key",
    "comparison_semantic_key",
    "target",
    "compile_kwargs_json",
    "kernel",
]
_CUTLASS_PACKAGE_FIELDS = {
    "nvidia-cutlass-dsl": "cutlass_dsl_version",
    "nvidia-cutlass-dsl-libs-base": "cutlass_dsl_libs_base_version",
    "nvidia-cutlass-dsl-libs-core": "cutlass_dsl_libs_core_version",
    "nvidia-cutlass-dsl-libs-cu12": "cutlass_dsl_libs_cu12_version",
    "nvidia-cutlass-dsl-libs-cu13": "cutlass_dsl_libs_cu13_version",
}


@dataclass(frozen=True)
class CompileManifest:
    cache_key: str = ""
    status: str = "missing"
    schema: str = ""
    semantic_key: str = ""
    comparison_semantic_key: str = ""
    target: str = ""
    kernel_id: str = ""
    compile_spec_version: str = ""
    compile_spec_hash: str = ""
    compile_spec_json: str = ""
    compile_kwargs_json: str = ""
    package_fingerprint: str = ""
    python_version: str = ""
    torch_version: str = ""
    torch_cuda_version: str = ""
    cutlass_dsl_version: str = ""
    cutlass_dsl_libs_base_version: str = ""
    cutlass_dsl_libs_core_version: str = ""
    cutlass_dsl_libs_cu12_version: str = ""
    cutlass_dsl_libs_cu13_version: str = ""
    toolchain_json: str = ""
    compile_options_json: str = ""
    compile_environment_json: str = ""
    launch_metadata_status: str = "unknown"
    launch_metadata_source: str = ""
    launch_metadata_reason: str = "manifest-field-missing"
    launch_dynamic_smem_bytes: dict[str, tuple[int, ...]] | None = None


@dataclass(frozen=True)
class KernelResources:
    resource_report_schema: str
    object_file: str
    object_sha256: str
    cache_key: str
    manifest_status: str
    manifest_schema: str
    semantic_key: str
    comparison_semantic_key: str
    target: str
    kernel_id: str
    compile_spec_version: str
    compile_spec_hash: str
    compile_spec_json: str
    compile_kwargs_json: str
    package_fingerprint: str
    python_version: str
    torch_version: str
    torch_cuda_version: str
    cutlass_dsl_version: str
    cutlass_dsl_libs_base_version: str
    cutlass_dsl_libs_core_version: str
    cutlass_dsl_libs_cu12_version: str
    cutlass_dsl_libs_cu13_version: str
    toolchain_json: str
    compile_options_json: str
    compile_environment_json: str
    kernel: str
    architecture: str
    ptxas_version: str
    ptxas_flags: str
    threads_x: int
    threads_y: int
    threads_z: int
    threads_per_cta: int
    cubin_shared_section_bytes: int
    launch_dynamic_smem_bytes: int | None
    launch_dynamic_smem_count: int
    launch_dynamic_smem_values_json: str
    launch_dynamic_smem_status: str
    launch_metadata_source: str
    launch_metadata_reason: str
    occupancy_status: str
    occupancy_device_ordinal: int | None
    occupancy_gpu_name: str
    occupancy_gpu_uuid: str
    occupancy_active_ctas_per_sm: int | None
    occupancy_active_threads_per_sm: int | None
    driver_resource_validation_status: str
    driver_registers: int | None
    driver_local_bytes: int | None
    driver_static_shared_bytes: int | None
    driver_max_threads_per_block: int | None
    max_register_count: int
    parameter_bytes: int
    registers: int
    sass_uniform_registers_used: int
    sass_uniform_register_span: int
    sass_predicate_registers_used: int
    sass_predicate_register_span: int
    sass_uniform_predicate_registers_used: int
    sass_uniform_predicate_register_span: int
    frame_bytes: int
    min_stack_bytes: int
    local_load_instructions: int
    local_store_instructions: int
    sass_instructions: int
    code_bytes: int

    @property
    def register_ceiling(self) -> bool:
        return self.registers >= 255

    @property
    def local_memory(self) -> bool:
        return (
            self.frame_bytes > 0
            or self.min_stack_bytes > 0
            or self.local_load_instructions > 0
            or self.local_store_instructions > 0
        )


class _CudaOccupancyAudit:
    """Query the CUDA driver for the launched function's CTA residency bound."""

    def __init__(self, device_ordinal: int) -> None:
        from cuda.bindings import driver as cuda

        self.cuda = cuda
        self.device_ordinal = device_ordinal
        self._check(cuda.cuInit(0), "cuInit")
        self.device = self._value(cuda.cuDeviceGet(device_ordinal), "cuDeviceGet")
        self.context = self._value(
            cuda.cuCtxCreate(None, 0, self.device), "cuCtxCreate"
        )
        raw_name = self._value(
            cuda.cuDeviceGetName(256, self.device), "cuDeviceGetName"
        )
        self.gpu_name = (
            bytes(raw_name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        )
        uuid = self._value(cuda.cuDeviceGetUuid(self.device), "cuDeviceGetUuid")
        uuid_hex = bytes(uuid.bytes).hex()
        self.gpu_uuid = (
            f"GPU-{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-"
            f"{uuid_hex[16:20]}-{uuid_hex[20:]}"
        )

    @staticmethod
    def _check(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
        if not result or int(result[0]) != 0:
            status = result[0] if result else "no-result"
            raise ValueError(f"{operation} failed: {status}")
        return result[1:]

    @classmethod
    def _value(cls, result: tuple[Any, ...], operation: str) -> Any:
        values = cls._check(result, operation)
        if len(values) != 1:
            raise ValueError(f"{operation} returned {len(values)} values")
        return values[0]

    def load_module(self, cubin: bytes) -> Any:
        return self._value(self.cuda.cuModuleLoadData(cubin), "cuModuleLoadData")

    def unload_module(self, module: Any) -> None:
        self._check(self.cuda.cuModuleUnload(module), "cuModuleUnload")

    def kernel_attributes_and_occupancy(
        self,
        cubin: bytes,
        kernel: str,
        threads_per_cta: int,
        dynamic_smem_bytes: int,
    ) -> tuple[int, dict[str, int]]:
        module = self.load_module(cubin)
        try:
            function = self._value(
                self.cuda.cuModuleGetFunction(module, kernel.encode("utf-8")),
                "cuModuleGetFunction",
            )
            if dynamic_smem_bytes:
                self._check(
                    self.cuda.cuFuncSetAttribute(
                        function,
                        self.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        dynamic_smem_bytes,
                    ),
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                )
            attributes = {
                "registers": int(
                    self._value(
                        self.cuda.cuFuncGetAttribute(
                            self.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                            function,
                        ),
                        "cuFuncGetAttribute(NUM_REGS)",
                    )
                ),
                "local_bytes": int(
                    self._value(
                        self.cuda.cuFuncGetAttribute(
                            self.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            function,
                        ),
                        "cuFuncGetAttribute(LOCAL_SIZE_BYTES)",
                    )
                ),
                "static_shared_bytes": int(
                    self._value(
                        self.cuda.cuFuncGetAttribute(
                            self.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            function,
                        ),
                        "cuFuncGetAttribute(SHARED_SIZE_BYTES)",
                    )
                ),
                "max_threads_per_block": int(
                    self._value(
                        self.cuda.cuFuncGetAttribute(
                            self.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            function,
                        ),
                        "cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)",
                    )
                ),
            }
            active_ctas = int(
                self._value(
                    self.cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        function,
                        threads_per_cta,
                        dynamic_smem_bytes,
                    ),
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                )
            )
            return active_ctas, attributes
        finally:
            self.unload_module(module)

    def close(self) -> None:
        if self.context is not None:
            context, self.context = self.context, None
            self._check(self.cuda.cuCtxDestroy(context), "cuCtxDestroy")


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


def _toolchain_versions(toolchain: Any) -> dict[str, str]:
    versions: dict[str, str] = {}
    if not isinstance(toolchain, list):
        return versions
    for entry in toolchain:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        name = str(entry[0])
        if name == "python" and len(entry) >= 3:
            implementation = str(entry[1])
            raw_version = entry[2]
            if isinstance(raw_version, list):
                versions[name] = f"{implementation} {'.'.join(map(str, raw_version))}"
            else:
                versions[name] = f"{implementation} {raw_version}"
        else:
            versions[name] = str(entry[1])
    return versions


def _manifest_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"kind": "float", "repr": repr(value)}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, (tuple, list)):
        return [_manifest_json_value(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {
                key: _manifest_json_value(item) for key, item in sorted(value.items())
            }
        return {
            "kind": "mapping",
            "items": [
                [_manifest_json_value(key), _manifest_json_value(item)]
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ],
        }
    return {
        "kind": "repr",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": repr(value),
    }


def _semantic_target_key(target_key: Any) -> Any:
    if not isinstance(target_key, (tuple, list)) or not target_key:
        return _manifest_json_value(target_key)
    tag = target_key[0]
    if tag in {"method", "function"} and len(target_key) >= 2:
        fingerprint = target_key[1]
        if isinstance(fingerprint, (tuple, list)) and len(fingerprint) >= 2:
            result: dict[str, Any] = {
                "kind": tag,
                "module": str(fingerprint[0]),
                "qualname": str(fingerprint[1]),
            }
            if len(target_key) >= 3:
                result["state"] = _manifest_json_value(target_key[2])
            return result
    if tag == "callable_instance" and len(target_key) >= 4:
        call_fingerprint = target_key[3]
        result = {
            "kind": tag,
            "type": f"{target_key[1]}.{target_key[2]}",
        }
        if isinstance(call_fingerprint, (tuple, list)) and len(call_fingerprint) >= 2:
            result["call"] = f"{call_fingerprint[0]}.{call_fingerprint[1]}"
        else:
            result["call"] = _manifest_json_value(call_fingerprint)
        if len(target_key) >= 5:
            result["state"] = _manifest_json_value(target_key[4])
        return result
    if tag == "callable" and len(target_key) >= 3:
        return {"kind": tag, "type": f"{target_key[1]}.{target_key[2]}"}
    return _manifest_json_value(target_key)


def _semantic_payload_from_cache_payload(cache_payload: list[Any]) -> dict[str, Any]:
    if len(cache_payload) != 11:
        raise ValueError(f"explicit cache payload has {len(cache_payload)} fields")
    if cache_payload[0] != "sparkinfer_cute_compile_cache_v6_explicit_spec":
        raise ValueError(f"unsupported cache format {cache_payload[0]!r}")
    semantic: dict[str, Any] = {
        "cache_format": cache_payload[0],
        "target": _semantic_target_key(cache_payload[1]),
        "compile_spec_hash": cache_payload[5],
    }
    try:
        semantic["compile_spec"] = json.loads(str(cache_payload[6]))
    except (TypeError, ValueError, json.JSONDecodeError):
        semantic["compile_spec"] = str(cache_payload[6])
    if cache_payload[7]:
        semantic["compile_kwargs_hash"] = cache_payload[7]
        try:
            semantic["compile_kwargs"] = json.loads(str(cache_payload[8]))
        except (TypeError, ValueError, json.JSONDecodeError):
            semantic["compile_kwargs"] = str(cache_payload[8])
    semantic["compile_options"] = _manifest_json_value(cache_payload[9])
    semantic["compile_environment"] = _manifest_json_value(cache_payload[10])
    return semantic


def _manifest_integrity_status(
    raw: dict[str, Any],
    *,
    filename_key: str,
    object_bytes: bytes,
) -> str:
    if raw.get("schema") not in _MANIFEST_SCHEMAS:
        return "unsupported-schema"
    cache_key = raw.get("cache_key")
    if not isinstance(cache_key, str) or not _CACHE_KEY_RE.fullmatch(cache_key):
        return "invalid-cache-key"
    if filename_key and cache_key != filename_key:
        return "cache-key-mismatch"
    if raw.get("object_sha256") != hashlib.sha256(object_bytes).hexdigest():
        return "object-hash-mismatch"
    if raw.get("object_bytes") != len(object_bytes):
        return "object-size-mismatch"
    launch_metadata = raw.get("launch_metadata")
    artifact_evidence = {
        "cache_key": cache_key,
        "object_sha256": raw.get("object_sha256"),
        "launch_metadata": launch_metadata,
    }
    try:
        artifact_evidence_json = json.dumps(
            artifact_evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return "invalid-artifact-evidence"
    if hashlib.sha256(artifact_evidence_json.encode("utf-8")).hexdigest() != raw.get(
        "artifact_evidence_sha256"
    ):
        return "artifact-evidence-hash-mismatch"
    cache_payload_repr = raw.get("cache_payload_repr")
    if (
        not isinstance(cache_payload_repr, str)
        or hashlib.sha256(cache_payload_repr.encode("utf-8")).hexdigest() != cache_key
    ):
        return "cache-payload-hash-mismatch"
    try:
        repr_payload = ast.literal_eval(cache_payload_repr)
    except (SyntaxError, ValueError):
        return "invalid-cache-payload-repr"
    cache_payload = raw.get("cache_payload")
    if (
        not isinstance(cache_payload, list)
        or _manifest_json_value(repr_payload) != cache_payload
    ):
        return "cache-payload-repr-mismatch"
    try:
        semantic_payload = _semantic_payload_from_cache_payload(cache_payload)
    except ValueError:
        return "invalid-cache-payload"
    if raw.get("cache_format") != cache_payload[0]:
        return "cache-format-mismatch"
    if raw.get("package_fingerprint") != cache_payload[2]:
        return "package-fingerprint-mismatch"
    if not isinstance(cache_payload[2], str) or not _CACHE_KEY_RE.fullmatch(
        cache_payload[2]
    ):
        return "invalid-package-fingerprint"
    if raw.get("toolchain") != cache_payload[3]:
        return "toolchain-payload-mismatch"
    if raw.get("compile_spec_hash") != cache_payload[5]:
        return "compile-spec-payload-mismatch"
    compile_spec_json = raw.get("compile_spec_json")
    if compile_spec_json != cache_payload[6] or not isinstance(compile_spec_json, str):
        return "compile-spec-json-mismatch"
    if hashlib.sha256(compile_spec_json.encode("utf-8")).hexdigest() != raw.get(
        "compile_spec_hash"
    ):
        return "compile-spec-hash-mismatch"
    if (
        raw.get("compile_kwargs_hash", "") != cache_payload[7]
        or raw.get("compile_kwargs_json", "") != cache_payload[8]
    ):
        return "compile-kwargs-payload-mismatch"
    if cache_payload[8]:
        if (
            hashlib.sha256(str(cache_payload[8]).encode("utf-8")).hexdigest()
            != cache_payload[7]
        ):
            return "compile-kwargs-hash-mismatch"
    elif cache_payload[7]:
        return "compile-kwargs-hash-without-json"
    if raw.get("compile_options") != cache_payload[9]:
        return "compile-options-payload-mismatch"
    if raw.get("compile_environment") != cache_payload[10]:
        return "compile-environment-payload-mismatch"
    try:
        compile_spec = json.loads(compile_spec_json)
    except json.JSONDecodeError:
        return "invalid-compile-spec-json"
    if not isinstance(compile_spec, dict):
        return "invalid-compile-spec-document"
    if raw.get("kernel_id") != compile_spec.get("kernel"):
        return "kernel-id-mismatch"
    if raw.get("compile_spec_version") != compile_spec.get("version"):
        return "compile-spec-version-mismatch"
    if raw.get("semantic_payload") != semantic_payload:
        return "semantic-payload-mismatch"
    semantic_json = json.dumps(
        semantic_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if hashlib.sha256(semantic_json.encode("utf-8")).hexdigest() != raw.get(
        "semantic_key"
    ):
        return "semantic-key-mismatch"
    semantic_target = semantic_payload.get("target")
    if not isinstance(semantic_target, dict):
        return "invalid-semantic-target"
    if raw.get("target_identity") != semantic_target:
        return "target-identity-mismatch"
    expected_target = semantic_target.get("type")
    if expected_target is None:
        expected_target = ".".join(
            filter(
                None,
                (
                    semantic_target.get("module"),
                    semantic_target.get("qualname"),
                ),
            )
        )
    if raw.get("target") != expected_target:
        return "target-name-mismatch"
    return "ok"


def _read_compile_manifest(object_path: Path, object_bytes: bytes) -> CompileManifest:
    filename_key = object_path.stem if _CACHE_KEY_RE.fullmatch(object_path.stem) else ""
    manifest_path = object_path.with_suffix(".json")
    if not manifest_path.exists():
        return CompileManifest(cache_key=filename_key)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return CompileManifest(cache_key=filename_key, status="invalid-json")
    if not isinstance(raw, dict):
        return CompileManifest(cache_key=filename_key, status="invalid-document")

    schema = str(raw.get("schema", ""))
    cache_key = str(raw.get("cache_key", filename_key))
    status = _manifest_integrity_status(
        raw,
        filename_key=filename_key,
        object_bytes=object_bytes,
    )
    comparison_key = ""
    if status == "ok":
        try:
            comparison_key = comparison_semantic_key_from_manifest(raw)
        except ValueError:
            status = "invalid-comparison-identity"

    toolchain = raw.get("toolchain", [])
    versions = _toolchain_versions(toolchain)
    launch_metadata = raw.get("launch_metadata")
    launch_status = "unknown"
    launch_source = ""
    launch_reason = "manifest-field-missing"
    launch_dynamic_smem: dict[str, tuple[int, ...]] = {}
    if isinstance(launch_metadata, dict):
        launch_status = str(launch_metadata.get("status", "unknown"))
        launch_source = str(launch_metadata.get("source", ""))
        launch_reason = str(launch_metadata.get("reason", ""))
        raw_dynamic_smem = launch_metadata.get("launch_dynamic_smem_bytes", {})
        valid = isinstance(raw_dynamic_smem, dict)
        if valid:
            for kernel, raw_values in raw_dynamic_smem.items():
                if not isinstance(kernel, str) or not isinstance(raw_values, list):
                    valid = False
                    break
                if not raw_values or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in raw_values
                ):
                    valid = False
                    break
                launch_dynamic_smem[kernel] = tuple(raw_values)
        if launch_status == "exact" and (not valid or not launch_dynamic_smem):
            launch_status = "invalid"
            launch_reason = "invalid-exact-launch-dynamic-smem-map"
            launch_dynamic_smem.clear()
        if (
            launch_status == "exact"
            and launch_source != "cutlass-final-llvm-launch-config-field-2"
        ):
            launch_status = "invalid"
            launch_reason = "invalid-exact-launch-metadata-source"
            launch_dynamic_smem.clear()
    if status != "ok":
        launch_status = "untrusted-manifest"
        launch_reason = status
        launch_dynamic_smem.clear()
    return CompileManifest(
        cache_key=cache_key,
        status=status,
        schema=schema,
        semantic_key=str(raw.get("semantic_key", "")),
        comparison_semantic_key=comparison_key,
        target=str(raw.get("target", "")),
        kernel_id=str(raw.get("kernel_id", "")),
        compile_spec_version=str(raw.get("compile_spec_version", "")),
        compile_spec_hash=str(raw.get("compile_spec_hash", "")),
        compile_spec_json=str(raw.get("compile_spec_json", "")),
        compile_kwargs_json=str(raw.get("compile_kwargs_json", "")),
        package_fingerprint=str(raw.get("package_fingerprint", "")),
        python_version=versions.get("python", ""),
        torch_version=versions.get("torch", ""),
        torch_cuda_version=versions.get("torch_cuda", ""),
        cutlass_dsl_version=versions.get("cutlass_dsl", "missing"),
        cutlass_dsl_libs_base_version=versions.get("cutlass_dsl_libs_base", "missing"),
        cutlass_dsl_libs_core_version=versions.get("cutlass_dsl_libs_core", "missing"),
        cutlass_dsl_libs_cu12_version=versions.get("cutlass_dsl_libs_cu12", "missing"),
        cutlass_dsl_libs_cu13_version=versions.get("cutlass_dsl_libs_cu13", "missing"),
        toolchain_json=_compact_json(toolchain),
        compile_options_json=_compact_json(raw.get("compile_options")),
        compile_environment_json=_compact_json(raw.get("compile_environment")),
        launch_metadata_status=launch_status,
        launch_metadata_source=launch_source,
        launch_metadata_reason=launch_reason,
        launch_dynamic_smem_bytes=launch_dynamic_smem,
    )


def _section(disassembly: str, section: str) -> str:
    marker = f"//--------------------- {section} "
    start = disassembly.find(marker)
    if start < 0:
        return ""
    next_section = disassembly.find("//--------------------- ", start + len(marker))
    return disassembly[start:] if next_section < 0 else disassembly[start:next_section]


def _attribute_blocks(section: str, attribute: str) -> list[str]:
    pattern = re.compile(
        rf"^[ \t]*//----- nvinfo : {re.escape(attribute)}[ \t]*\r?$"
        rf"(?P<body>.*?)"
        rf"(?=^[ \t]*//----- nvinfo :|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    return [match.group("body") for match in pattern.finditer(section)]


def _attribute_block(section: str, attribute: str) -> str:
    blocks = _attribute_blocks(section, attribute)
    if len(blocks) > 1:
        raise ValueError(f"nvdisasm reported duplicate per-kernel {attribute} blocks")
    return blocks[0] if blocks else ""


def _resource_values(disassembly: str, attribute: str) -> dict[str, int]:
    """Parse one bounded global .nv.info resource-attribute block."""

    global_info = _section(disassembly, ".nv.info")
    if not global_info:
        raise ValueError("nvdisasm omitted the global .nv.info section")
    blocks = _attribute_blocks(global_info, attribute)
    if not blocks:
        raise ValueError(f"nvdisasm omitted the global {attribute} block")

    entries: dict[str, int] = {}
    for block_number, block in enumerate(blocks, start=1):
        pending_kernel: str | None = None
        for line in block.splitlines():
            index_match = re.search(r"\.word\s+index@\(([^)]+)\)", line)
            value_match = re.search(r"\.word\s+0x([0-9a-fA-F]+)", line)
            if index_match is not None:
                if pending_kernel is not None:
                    raise ValueError(
                        f"{attribute} block {block_number} has index "
                        f"{pending_kernel!r} without a value"
                    )
                pending_kernel = index_match.group(1)
            elif value_match is not None:
                if pending_kernel is None:
                    raise ValueError(
                        f"{attribute} block {block_number} has a value without "
                        "a kernel/function index"
                    )
                if pending_kernel in entries:
                    raise ValueError(
                        f"{attribute} has duplicate entry for {pending_kernel}"
                    )
                entries[pending_kernel] = int(value_match.group(1), 16)
                pending_kernel = None
        if pending_kernel is not None:
            raise ValueError(
                f"{attribute} block {block_number} has index {pending_kernel!r} "
                "without a value"
            )
    if not entries:
        raise ValueError(f"nvdisasm reported no entries in {attribute}")
    return entries


def _kernel_code(disassembly: str, kernel: str) -> str:
    marker = f"//--------------------- .text.{kernel}"
    start = disassembly.find(marker)
    if start < 0:
        return ""
    next_section = disassembly.find("//--------------------- ", start + len(marker))
    return disassembly[start:] if next_section < 0 else disassembly[start:next_section]


def _kernel_info(disassembly: str, kernel: str) -> str:
    marker = f"//--------------------- .nv.info.{kernel}"
    start = disassembly.find(marker)
    if start < 0:
        return ""
    next_section = disassembly.find("//--------------------- ", start + len(marker))
    return disassembly[start:] if next_section < 0 else disassembly[start:next_section]


def _attribute_words(kernel_info: str, attribute: str) -> list[int]:
    block = _attribute_block(kernel_info, attribute)
    return [int(value, 16) for value in re.findall(r"\.word\s+0x([0-9a-fA-F]+)", block)]


def _attribute_short(kernel_info: str, attribute: str) -> int:
    block = _attribute_block(kernel_info, attribute)
    match = re.search(r"\.short\s+0x([0-9a-fA-F]+)", block)
    return int(match.group(1), 16) if match else 0


def _sass_register_usage(code: str, pattern: re.Pattern[str]) -> tuple[int, int]:
    """Return distinct named operands and the highest-addressed register span.

    Cubin metadata exposes the exact allocated per-thread GPR count through
    ``EIATTR_REGCOUNT``, but it does not expose equivalent allocation counts
    for the uniform or predicate files.  Keep these SASS-derived measurements
    explicitly named: the distinct count records operand use and the span
    records ``max(index) + 1``.  Neither is presented as an occupancy count.
    """

    indices = {int(match.group("index")) for match in pattern.finditer(code)}
    return len(indices), max(indices, default=-1) + 1


def _cubin_shared_section_bytes(disassembly: str, kernel: str) -> int:
    marker = f"//--------------------- .nv.shared.{kernel}"
    start = disassembly.find(marker)
    if start < 0:
        return 0
    next_section = disassembly.find("//--------------------- ", start + len(marker))
    section = (
        disassembly[start:] if next_section < 0 else disassembly[start:next_section]
    )
    return sum(
        int(value, 0) for value in re.findall(r"\.zero\s+([0-9xa-fA-F]+)", section)
    )


def _ptxas_metadata(disassembly: str) -> tuple[str, str]:
    version_match = re.search(
        r'\.string\s+"(Cuda compilation tools,[^"]+)"', disassembly
    )
    flags_match = next(
        (
            match
            for match in re.finditer(r'\.string\s+"([^"\r\n]+)"', disassembly)
            if re.search(r"(?:^|\s)-O\s+\d+(?:\s|$)", match.group(1))
            and re.search(r"(?:^|\s)-arch\s+\S+", match.group(1))
        ),
        None,
    )
    return (
        version_match.group(1) if version_match else "",
        flags_match.group(1).strip() if flags_match else "",
    )


def _embedded_cuda_elf(object_bytes: bytes) -> bytes:
    """Extract the exact embedded ELF64 CUDA object, excluding wrapper bytes."""

    embedded_cubin_count = object_bytes.count(_CUDA_ELF_MAGIC)
    if embedded_cubin_count != 1:
        raise ValueError(
            f"expected exactly one embedded CUDA ELF, found {embedded_cubin_count}"
        )
    cubin_start = object_bytes.find(_CUDA_ELF_MAGIC)
    available = len(object_bytes) - cubin_start
    if available < 64:
        raise ValueError("embedded CUDA ELF header is truncated")

    # CUDA cubins are ELF64 little-endian objects. Extended table counts would
    # require consulting section zero; reject them instead of guessing an
    # extent that could absorb bytes belonging to the host-object wrapper.
    e_phoff = struct.unpack_from("<Q", object_bytes, cubin_start + 0x20)[0]
    e_shoff = struct.unpack_from("<Q", object_bytes, cubin_start + 0x28)[0]
    e_ehsize = struct.unpack_from("<H", object_bytes, cubin_start + 0x34)[0]
    e_phentsize = struct.unpack_from("<H", object_bytes, cubin_start + 0x36)[0]
    e_phnum = struct.unpack_from("<H", object_bytes, cubin_start + 0x38)[0]
    e_shentsize = struct.unpack_from("<H", object_bytes, cubin_start + 0x3A)[0]
    e_shnum = struct.unpack_from("<H", object_bytes, cubin_start + 0x3C)[0]
    if e_ehsize < 64:
        raise ValueError(f"embedded CUDA ELF has invalid header size {e_ehsize}")
    if e_phnum == 0xFFFF or e_shnum == 0:
        raise ValueError("embedded CUDA ELF uses unsupported extended table counts")
    if e_phnum and (not e_phoff or e_phentsize < 56):
        raise ValueError("embedded CUDA ELF has an invalid program-header table")
    if e_shnum and (not e_shoff or e_shentsize < 64):
        raise ValueError("embedded CUDA ELF has an invalid section-header table")

    def checked_end(offset: int, size: int, label: str) -> int:
        end = offset + size
        if offset < 0 or size < 0 or end < offset or end > available:
            raise ValueError(f"embedded CUDA ELF {label} is truncated")
        return end

    extent = checked_end(0, e_ehsize, "header")
    if e_phnum:
        extent = max(
            extent,
            checked_end(e_phoff, e_phentsize * e_phnum, "program-header table"),
        )
        for index in range(e_phnum):
            header = cubin_start + e_phoff + index * e_phentsize
            p_offset = struct.unpack_from("<Q", object_bytes, header + 0x08)[0]
            p_filesz = struct.unpack_from("<Q", object_bytes, header + 0x20)[0]
            extent = max(
                extent,
                checked_end(p_offset, p_filesz, f"program segment {index}"),
            )
    extent = max(
        extent,
        checked_end(e_shoff, e_shentsize * e_shnum, "section-header table"),
    )
    for index in range(e_shnum):
        header = cubin_start + e_shoff + index * e_shentsize
        sh_type = struct.unpack_from("<I", object_bytes, header + 0x04)[0]
        sh_offset = struct.unpack_from("<Q", object_bytes, header + 0x18)[0]
        sh_size = struct.unpack_from("<Q", object_bytes, header + 0x20)[0]
        if sh_type != 8:  # SHT_NOBITS occupies memory but has no file payload.
            extent = max(
                extent,
                checked_end(sh_offset, sh_size, f"section {index}"),
            )
    return object_bytes[cubin_start : cubin_start + extent]


def _audit_object(
    object_path: Path,
    nvdisasm: str,
    occupancy: _CudaOccupancyAudit | None = None,
) -> list[KernelResources]:
    object_bytes = object_path.read_bytes()
    object_sha256 = hashlib.sha256(object_bytes).hexdigest()
    manifest = _read_compile_manifest(object_path, object_bytes)
    cubin_bytes = _embedded_cuda_elf(object_bytes)

    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(cubin_bytes)
        cubin_file.flush()
        result = subprocess.run(
            [nvdisasm, cubin_file.name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_TOOL_TIMEOUT_SECONDS,
        )
    disassembly = result.stdout

    target_match = re.search(r"^\s*\.target\s+(\S+)", disassembly, re.MULTILINE)
    if target_match is None:
        raise ValueError("nvdisasm omitted the CUDA target architecture")
    architecture = target_match.group(1)
    ptxas_version, ptxas_flags = _ptxas_metadata(disassembly)
    if not ptxas_version:
        raise ValueError("nvdisasm omitted the PTXAS version metadata")
    if not ptxas_flags:
        raise ValueError("nvdisasm omitted the PTXAS flags metadata")
    registers = _resource_values(disassembly, "EIATTR_REGCOUNT")
    frames = _resource_values(disassembly, "EIATTR_FRAME_SIZE")
    stacks = _resource_values(disassembly, "EIATTR_MIN_STACK_SIZE")
    # Enumerate entry points from their executable text sections, independently
    # of resource metadata.  Building this set from REGCOUNT/FRAME/STACK would
    # let a changed nvdisasm format silently erase a kernel or turn a missing
    # field into an apparent zero/improvement.
    kernels = sorted(
        {
            match.group("kernel")
            for match in _TEXT_KERNEL_SECTION_RE.finditer(disassembly)
        }
    )
    if not kernels:
        raise ValueError("nvdisasm reported no CUDA kernel text entry points")
    if manifest.launch_metadata_status == "exact":
        launch_kernels = set(manifest.launch_dynamic_smem_bytes or {})
        if launch_kernels != set(kernels):
            missing_launch = sorted(set(kernels) - launch_kernels)
            unexpected_launch = sorted(launch_kernels - set(kernels))
            raise ValueError(
                "exact launch metadata kernel set differs from cubin entry points: "
                f"missing={missing_launch!r} unexpected={unexpected_launch!r}"
            )

    resource_kernel_sets = {
        "EIATTR_REGCOUNT": set(registers),
        "EIATTR_FRAME_SIZE": set(frames),
        "EIATTR_MIN_STACK_SIZE": set(stacks),
    }
    if resource_kernel_sets["EIATTR_REGCOUNT"] != set(kernels):
        missing = sorted(set(kernels) - resource_kernel_sets["EIATTR_REGCOUNT"])
        unexpected = sorted(resource_kernel_sets["EIATTR_REGCOUNT"] - set(kernels))
        raise ValueError(
            "EIATTR_REGCOUNT entries differ from accepted CUDA kernel text "
            f"entry points: missing={missing!r} unexpected={unexpected!r}"
        )
    for attribute, reported_kernels in resource_kernel_sets.items():
        missing = sorted(set(kernels) - reported_kernels)
        if missing:
            raise ValueError(
                f"nvdisasm omitted {attribute} for CUDA entry points: "
                + ", ".join(missing)
            )

    rows: list[KernelResources] = []
    for kernel in kernels:
        code = _kernel_code(disassembly, kernel)
        kernel_info = _kernel_info(disassembly, kernel)
        cubin_shared_section_bytes = _cubin_shared_section_bytes(disassembly, kernel)
        reqntid = _attribute_words(kernel_info, "EIATTR_REQNTID")
        if not code:
            raise ValueError(f"nvdisasm omitted executable text for {kernel}")
        if not kernel_info:
            raise ValueError(f"nvdisasm omitted .nv.info section for {kernel}")
        if len(reqntid) != 3 or any(value <= 0 for value in reqntid):
            raise ValueError(
                f"nvdisasm reported invalid EIATTR_REQNTID for {kernel}: {reqntid}"
            )
        threads_x, threads_y, threads_z = reqntid
        instruction_offsets = [
            int(match.group("offset"), 16) for match in _INSTRUCTION_RE.finditer(code)
        ]
        if not instruction_offsets:
            raise ValueError(f"nvdisasm reported no SASS instructions for {kernel}")
        if registers[kernel] <= 0:
            raise ValueError(
                f"nvdisasm reported nonpositive EIATTR_REGCOUNT for {kernel}: "
                f"{registers[kernel]}"
            )
        uniform_registers_used, uniform_register_span = _sass_register_usage(
            code, _UNIFORM_REGISTER_RE
        )
        predicate_registers_used, predicate_register_span = _sass_register_usage(
            code, _PREDICATE_REGISTER_RE
        )
        (
            uniform_predicate_registers_used,
            uniform_predicate_register_span,
        ) = _sass_register_usage(code, _UNIFORM_PREDICATE_REGISTER_RE)
        launch_dynamic_smem_values = (
            (manifest.launch_dynamic_smem_bytes or {}).get(kernel, ())
            if manifest.launch_metadata_status == "exact"
            else ()
        )
        if launch_dynamic_smem_values:
            unique_launch_dynamic_smem = set(launch_dynamic_smem_values)
            launch_dynamic_smem_status = (
                "exact" if len(unique_launch_dynamic_smem) == 1 else "exact-multiple"
            )
            launch_dynamic_smem_bytes = max(unique_launch_dynamic_smem)
        else:
            launch_dynamic_smem_status = (
                "missing-kernel"
                if manifest.launch_metadata_status == "exact"
                else manifest.launch_metadata_status
            )
            launch_dynamic_smem_bytes = None
        if occupancy is None:
            occupancy_status = "not-requested"
            occupancy_active_ctas_per_sm = None
            driver_attributes: dict[str, int] = {}
            driver_resource_validation_status = "not-requested"
        elif (
            launch_dynamic_smem_bytes is None or threads_x * threads_y * threads_z <= 0
        ):
            occupancy_status = "missing-launch-metadata"
            occupancy_active_ctas_per_sm = None
            driver_attributes = {}
            driver_resource_validation_status = "missing-launch-metadata"
        else:
            occupancy_status = "exact-driver-query"
            (
                occupancy_active_ctas_per_sm,
                driver_attributes,
            ) = occupancy.kernel_attributes_and_occupancy(
                cubin_bytes,
                kernel,
                threads_x * threads_y * threads_z,
                launch_dynamic_smem_bytes,
            )
            resource_mismatches = []
            if driver_attributes["registers"] != registers[kernel]:
                resource_mismatches.append("registers")
            if driver_attributes["local_bytes"] != frames[kernel]:
                resource_mismatches.append("local-bytes/frame-bytes")
            if (
                driver_attributes["max_threads_per_block"]
                < threads_x * threads_y * threads_z
            ):
                resource_mismatches.append("max-threads-per-block")
            driver_resource_validation_status = (
                "exact-match"
                if not resource_mismatches
                else "mismatch:" + ",".join(resource_mismatches)
            )
        rows.append(
            KernelResources(
                resource_report_schema=_RESOURCE_REPORT_SCHEMA,
                object_file=str(object_path),
                object_sha256=object_sha256,
                cache_key=manifest.cache_key,
                manifest_status=manifest.status,
                manifest_schema=manifest.schema,
                semantic_key=manifest.semantic_key,
                comparison_semantic_key=manifest.comparison_semantic_key,
                target=manifest.target,
                kernel_id=manifest.kernel_id,
                compile_spec_version=manifest.compile_spec_version,
                compile_spec_hash=manifest.compile_spec_hash,
                compile_spec_json=manifest.compile_spec_json,
                compile_kwargs_json=manifest.compile_kwargs_json,
                package_fingerprint=manifest.package_fingerprint,
                python_version=manifest.python_version,
                torch_version=manifest.torch_version,
                torch_cuda_version=manifest.torch_cuda_version,
                cutlass_dsl_version=manifest.cutlass_dsl_version,
                cutlass_dsl_libs_base_version=(manifest.cutlass_dsl_libs_base_version),
                cutlass_dsl_libs_core_version=(manifest.cutlass_dsl_libs_core_version),
                cutlass_dsl_libs_cu12_version=(manifest.cutlass_dsl_libs_cu12_version),
                cutlass_dsl_libs_cu13_version=(manifest.cutlass_dsl_libs_cu13_version),
                toolchain_json=manifest.toolchain_json,
                compile_options_json=manifest.compile_options_json,
                compile_environment_json=manifest.compile_environment_json,
                kernel=kernel,
                architecture=architecture,
                ptxas_version=ptxas_version,
                ptxas_flags=ptxas_flags,
                threads_x=threads_x,
                threads_y=threads_y,
                threads_z=threads_z,
                threads_per_cta=threads_x * threads_y * threads_z,
                cubin_shared_section_bytes=cubin_shared_section_bytes,
                launch_dynamic_smem_bytes=launch_dynamic_smem_bytes,
                launch_dynamic_smem_count=len(launch_dynamic_smem_values),
                launch_dynamic_smem_values_json=_compact_json(
                    launch_dynamic_smem_values
                ),
                launch_dynamic_smem_status=launch_dynamic_smem_status,
                launch_metadata_source=manifest.launch_metadata_source,
                launch_metadata_reason=manifest.launch_metadata_reason,
                occupancy_status=occupancy_status,
                occupancy_device_ordinal=(
                    occupancy.device_ordinal if occupancy is not None else None
                ),
                occupancy_gpu_name=(
                    occupancy.gpu_name if occupancy is not None else ""
                ),
                occupancy_gpu_uuid=(
                    occupancy.gpu_uuid if occupancy is not None else ""
                ),
                occupancy_active_ctas_per_sm=occupancy_active_ctas_per_sm,
                occupancy_active_threads_per_sm=(
                    occupancy_active_ctas_per_sm * threads_x * threads_y * threads_z
                    if occupancy_active_ctas_per_sm is not None
                    else None
                ),
                driver_resource_validation_status=(driver_resource_validation_status),
                driver_registers=driver_attributes.get("registers"),
                driver_local_bytes=driver_attributes.get("local_bytes"),
                driver_static_shared_bytes=driver_attributes.get("static_shared_bytes"),
                driver_max_threads_per_block=driver_attributes.get(
                    "max_threads_per_block"
                ),
                max_register_count=_attribute_short(kernel_info, "EIATTR_MAXREG_COUNT"),
                parameter_bytes=_attribute_short(
                    kernel_info, "EIATTR_CBANK_PARAM_SIZE"
                ),
                registers=registers[kernel],
                sass_uniform_registers_used=uniform_registers_used,
                sass_uniform_register_span=uniform_register_span,
                sass_predicate_registers_used=predicate_registers_used,
                sass_predicate_register_span=predicate_register_span,
                sass_uniform_predicate_registers_used=(
                    uniform_predicate_registers_used
                ),
                sass_uniform_predicate_register_span=(uniform_predicate_register_span),
                frame_bytes=frames.get(kernel, 0),
                min_stack_bytes=stacks.get(kernel, 0),
                local_load_instructions=len(_LOCAL_LOAD_RE.findall(code)),
                local_store_instructions=len(_LOCAL_STORE_RE.findall(code)),
                sass_instructions=len(instruction_offsets),
                code_bytes=(
                    max(instruction_offsets) + 16 if instruction_offsets else 0
                ),
            )
        )
    return rows


def _short_kernel_name(kernel: str) -> str:
    match = re.search(r"kernel_sparkinfer(.*?)_object_at", kernel)
    return match.group(1) if match else kernel


def _row_dict(row: KernelResources) -> dict[str, object]:
    return {
        **asdict(row),
        "register_ceiling": row.register_ceiling,
        "local_memory": row.local_memory,
    }


def _write_csv(rows: list[KernelResources], output) -> None:
    fieldnames = [
        *KernelResources.__dataclass_fields__,
        "register_ceiling",
        "local_memory",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(_row_dict(row))


def _write_report(rows: list[KernelResources], output, output_format: str) -> None:
    if output_format == "json":
        json.dump([_row_dict(row) for row in rows], output, indent=2)
        output.write("\n")
    elif output_format == "csv":
        _write_csv(rows, output)
    else:
        print(
            "status manifest launch-smem regs UR P UP frame stack LDL STL threads "
            "ctas/SM cubin-section-smem launch-dynamic-smem architecture "
            "kernel_id/target",
            file=output,
        )
        for row in rows:
            status = (
                "LOCAL"
                if row.local_memory
                else "CEIL"
                if row.register_ceiling
                else "ok"
            )
            print(
                f"{status:5} {row.manifest_status:18} "
                f"{row.launch_dynamic_smem_status:14} "
                f"{row.registers:4} {row.sass_uniform_register_span:2} "
                f"{row.sass_predicate_register_span:1} "
                f"{row.sass_uniform_predicate_register_span:2} "
                f"{row.frame_bytes:5} "
                f"{row.min_stack_bytes:5} {row.local_load_instructions:3} "
                f"{row.local_store_instructions:3} {row.threads_per_cta:7} "
                f"{str(row.occupancy_active_ctas_per_sm) if row.occupancy_active_ctas_per_sm is not None else '?':>7} "
                f"{row.cubin_shared_section_bytes:10} "
                f"{str(row.launch_dynamic_smem_bytes) if row.launch_dynamic_smem_bytes is not None else '?':>12} "
                f"{row.architecture:12} "
                f"{row.kernel_id or row.target or _short_kernel_name(row.kernel)}",
                file=output,
            )
        print(
            f"audited {len(rows)} kernels; "
            f"register ceiling={sum(row.register_ceiling for row in rows)}; "
            f"local memory={sum(row.local_memory for row in rows)}; "
            f"unknown launch dynamic SMEM="
            f"{sum(row.launch_dynamic_smem_status != 'exact' for row in rows)}; "
            f"unknown driver occupancy="
            f"{sum(row.occupancy_status != 'exact-driver-query' for row in rows)}; "
            f"unvalidated driver resources="
            f"{sum(row.driver_resource_validation_status != 'exact-match' for row in rows)}; "
            f"missing/invalid semantic manifest="
            f"{sum(row.manifest_status != 'ok' for row in rows)}",
            file=output,
        )


def _read_required_patterns(
    inline_patterns: list[str], pattern_files: list[Path]
) -> list[str]:
    patterns = [pattern.strip() for pattern in inline_patterns if pattern.strip()]
    for pattern_file in pattern_files:
        try:
            lines = pattern_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(
                f"cannot read coverage file {pattern_file}: {exc}"
            ) from exc
        patterns.extend(
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        )
    return sorted(set(patterns))


def _read_specialization_contract(
    path: Path,
) -> set[tuple[str, ...]]:
    try:
        source = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read specialization contract {path}: {exc}") from exc
    with source:
        reader = csv.DictReader(source, delimiter="\t")
        expected_fields = _SPECIALIZATION_CONTRACT_FIELDS
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"{path}: expected tab-separated columns {expected_fields}, "
                f"got {reader.fieldnames}"
            )
        entries: set[tuple[str, ...]] = set()
        identities: set[tuple[str, str]] = set()
        for line_number, row in enumerate(reader, 2):
            entry = tuple(row.get(field, "").strip() for field in expected_fields)
            (
                kernel_id,
                compile_spec_version,
                compile_spec_hash,
                compile_spec_json,
                semantic_key,
                comparison_semantic_key,
                target,
                compile_kwargs_json,
                kernel,
            ) = entry
            if not kernel_id:
                raise ValueError(f"{path}:{line_number}: kernel_id is empty")
            if not compile_spec_version:
                raise ValueError(f"{path}:{line_number}: compile_spec_version is empty")
            if not _CACHE_KEY_RE.fullmatch(compile_spec_hash):
                raise ValueError(
                    f"{path}:{line_number}: compile_spec_hash is not SHA-256"
                )
            if (
                hashlib.sha256(compile_spec_json.encode("utf-8")).hexdigest()
                != compile_spec_hash
            ):
                raise ValueError(f"{path}:{line_number}: compile_spec_hash mismatch")
            if not _CACHE_KEY_RE.fullmatch(semantic_key):
                raise ValueError(f"{path}:{line_number}: semantic_key is not SHA-256")
            if not _CACHE_KEY_RE.fullmatch(comparison_semantic_key):
                raise ValueError(
                    f"{path}:{line_number}: comparison_semantic_key is not SHA-256"
                )
            if not target:
                raise ValueError(f"{path}:{line_number}: target is empty")
            if not kernel:
                raise ValueError(f"{path}:{line_number}: kernel is empty")
            try:
                compile_spec = json.loads(compile_spec_json)
                if compile_kwargs_json:
                    json.loads(compile_kwargs_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid compile JSON") from exc
            if not isinstance(compile_spec, dict):
                raise ValueError(f"{path}:{line_number}: compile spec is not an object")
            if compile_spec.get("kernel") != kernel_id:
                raise ValueError(f"{path}:{line_number}: kernel_id disagrees with spec")
            if str(compile_spec.get("version", "")) != compile_spec_version:
                raise ValueError(
                    f"{path}:{line_number}: spec version disagrees with spec"
                )
            identity = (comparison_semantic_key, kernel)
            if identity in identities:
                raise ValueError(
                    f"{path}:{line_number}: duplicate resource-row identity {identity}"
                )
            identities.add(identity)
            entries.add(entry)
    if not entries:
        raise ValueError(f"{path}: resource-row contract is empty")
    return entries


def _read_contract_metadata(
    path: Path,
    *,
    contract_path: Path,
    corpus_driver: Path,
    shape_matrix: Path,
    source_inventory: Path,
    expected_corpus_id: str,
    expected_corpus_version: str,
    resource_rows: set[tuple[str, ...]],
) -> dict[str, int]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read contract metadata {path}: {exc}") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != _CONTRACT_METADATA_SCHEMA
    ):
        raise ValueError(f"{path}: unsupported contract metadata schema")
    if metadata.get("corpus_id") != expected_corpus_id:
        raise ValueError(f"{path}: corpus id differs from required value")
    if metadata.get("corpus_version") != expected_corpus_version:
        raise ValueError(f"{path}: corpus version differs from required value")
    contract = metadata.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: contract record is missing")
    if contract.get("sha256") != hashlib.sha256(contract_path.read_bytes()).hexdigest():
        raise ValueError(f"{path}: exact contract SHA-256 mismatch")
    if contract.get("fields") != _SPECIALIZATION_CONTRACT_FIELDS:
        raise ValueError(f"{path}: exact contract fields differ")
    semantic_count = len({entry[4] for entry in resource_rows})
    comparison_semantic_count = len({entry[5] for entry in resource_rows})
    expected_counts = {
        "resource_row_count": len(resource_rows),
        "semantic_key_count": semantic_count,
        "comparison_semantic_key_count": comparison_semantic_count,
        "object_count": int(contract.get("object_count", -1)),
    }
    if contract.get("resource_row_count") != expected_counts["resource_row_count"]:
        raise ValueError(f"{path}: resource-row count disagrees with contract")
    if contract.get("semantic_key_count") != expected_counts["semantic_key_count"]:
        raise ValueError(f"{path}: semantic-key count disagrees with contract")
    if (
        contract.get("comparison_semantic_key_count")
        != expected_counts["comparison_semantic_key_count"]
    ):
        raise ValueError(
            f"{path}: comparison-semantic-key count disagrees with contract"
        )
    if expected_counts["object_count"] <= 0:
        raise ValueError(f"{path}: invalid expected object count")
    reviewed = metadata.get("reviewed_artifacts")
    artifact_paths = {
        "contract_builder": EVIDENCE_ROOT / "specialization_contract.py",
        "corpus_driver": corpus_driver,
        "shape_matrix": shape_matrix,
        "source_inventory": source_inventory,
    }
    if not isinstance(reviewed, dict) or set(reviewed) != set(artifact_paths):
        raise ValueError(f"{path}: reviewed artifact set differs")
    for name, artifact_path in artifact_paths.items():
        record = reviewed.get(name)
        if not isinstance(record, dict) or not artifact_path.is_file():
            raise ValueError(f"{path}: reviewed artifact {name} is missing")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if record.get("sha256") != digest:
            raise ValueError(f"{path}: reviewed artifact {name} SHA-256 mismatch")
    origin_reports = metadata.get("origin_reports")
    if (
        not isinstance(origin_reports, list)
        or not origin_reports
        or any(
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not _CACHE_KEY_RE.fullmatch(str(record.get("sha256", "")))
            for record in origin_reports
        )
    ):
        raise ValueError(f"{path}: origin-report provenance is invalid")
    return expected_counts


def _parse_cutlass_package_map(values: list[str]) -> dict[str, str] | None:
    if not values:
        return None
    packages: dict[str, str] = {}
    for value in values:
        name, separator, version = value.partition("=")
        if not separator or not name or not version:
            raise ValueError(
                f"invalid CUTLASS package requirement {value!r}; expected NAME=VERSION"
            )
        if name not in _CUTLASS_PACKAGE_FIELDS:
            raise ValueError(f"unknown CUTLASS package {name!r}")
        if name in packages:
            raise ValueError(f"duplicate CUTLASS package requirement {name!r}")
        packages[name] = version
    missing = sorted(set(_CUTLASS_PACKAGE_FIELDS) - set(packages))
    if missing:
        raise ValueError(
            "exact CUTLASS package map is incomplete; missing " + ", ".join(missing)
        )
    return packages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="CuTe object-cache directories or individual .o files",
    )
    parser.add_argument("--format", choices=("table", "csv", "json"), default="table")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the report to this path instead of stdout",
    )
    parser.add_argument(
        "--fail-on-spill",
        action="store_true",
        help="exit nonzero when a kernel has a stack frame or LDL/STL",
    )
    parser.add_argument(
        "--fail-on-register-ceiling",
        action="store_true",
        help="exit nonzero when a kernel uses 255 registers",
    )
    parser.add_argument(
        "--require-semantic-manifest",
        action="store_true",
        help=(
            "exit nonzero when an object lacks a valid compile-manifest sidecar; "
            "old objects are reported as missing rather than inferred"
        ),
    )
    parser.add_argument(
        "--require-launch-dynamic-smem",
        action="store_true",
        help=(
            "exit nonzero unless every kernel has exact launch-time dynamic "
            "shared-memory metadata from the fresh compiled IR"
        ),
    )
    parser.add_argument(
        "--occupancy-device",
        type=int,
        metavar="ORDINAL",
        help=(
            "load each audited cubin on this CUDA device ordinal and query the "
            "driver's active-CTA-per-SM bound using its exact launch SMEM"
        ),
    )
    parser.add_argument(
        "--require-driver-occupancy",
        action="store_true",
        help="exit nonzero unless every row has an exact CUDA driver occupancy query",
    )
    parser.add_argument(
        "--require-driver-resource-validation",
        action="store_true",
        help=(
            "require CUDA driver registers/local bytes to agree with nvdisasm, "
            "record authoritative static SMEM, and require the driver max-thread "
            "limit to admit the CTA"
        ),
    )
    parser.add_argument(
        "--require-occupancy-gpu-uuid",
        help="require the occupancy device to have this exact GPU UUID",
    )
    parser.add_argument(
        "--require-cutlass-dsl-version",
        help="exit nonzero unless every manifest reports this exact DSL version",
    )
    parser.add_argument(
        "--require-cutlass-package",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help=(
            "require an exact CUTLASS distribution version; repeat exactly once "
            "for each of dsl, libs-base, libs-core, libs-cu12, and libs-cu13, "
            "using VERSION=missing for an absent distribution"
        ),
    )
    parser.add_argument(
        "--require-architecture",
        default="sm_120a",
        help=(
            "exit nonzero unless every cubin has this exact target architecture "
            "(default: sm_120a)"
        ),
    )
    parser.add_argument(
        "--require-cutlass-libs-base-version",
        help="exit nonzero unless every manifest reports this exact libs-base version",
    )
    parser.add_argument(
        "--require-cutlass-libs-core-version",
        help="exit nonzero unless every manifest reports this exact libs-core version",
    )
    parser.add_argument(
        "--require-cutlass-libs-cu12-version",
        help="exit nonzero unless every manifest reports this exact libs-cu12 version",
    )
    parser.add_argument(
        "--require-cutlass-libs-cu13-version",
        help="exit nonzero unless every manifest reports this exact libs-cu13 version",
    )
    parser.add_argument(
        "--require-kernel-id-pattern",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "require at least one valid semantic manifest whose kernel_id matches "
            "this shell-style glob; repeat for multiple required families"
        ),
    )
    parser.add_argument(
        "--require-kernel-id-pattern-file",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help=(
            "read required kernel_id globs from PATH, one per non-comment line; "
            "repeat for multiple coverage files"
        ),
    )
    parser.add_argument(
        "--require-kernel-symbol-pattern",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "require at least one audited CUDA entry-point symbol matching this "
            "shell-style glob; repeat for multiple required source entries"
        ),
    )
    parser.add_argument(
        "--require-kernel-symbol-pattern-file",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help=(
            "read required CUDA entry-point symbol globs from PATH, one per "
            "non-comment line; repeat for multiple coverage files"
        ),
    )
    parser.add_argument(
        "--require-exact-specialization-contract",
        type=Path,
        metavar="PATH",
        help=(
            "require the audited set of exact compile specs, semantic keys, "
            "targets, and CUDA entry-point symbols to equal the tab-separated "
            "resource-row contract in PATH"
        ),
    )
    parser.add_argument(
        "--require-exact-specialization-contract-metadata",
        type=Path,
        metavar="PATH",
        help="require reviewed corpus metadata binding the exact row contract",
    )
    parser.add_argument("--require-corpus-driver", type=Path)
    parser.add_argument("--require-shape-matrix", type=Path)
    parser.add_argument("--require-source-inventory", type=Path)
    parser.add_argument("--require-corpus-id")
    parser.add_argument("--require-corpus-version")
    args = parser.parse_args()

    try:
        required_kernel_patterns = _read_required_patterns(
            args.require_kernel_id_pattern,
            args.require_kernel_id_pattern_file,
        )
        required_kernel_symbol_patterns = _read_required_patterns(
            args.require_kernel_symbol_pattern,
            args.require_kernel_symbol_pattern_file,
        )
        required_specializations = (
            _read_specialization_contract(args.require_exact_specialization_contract)
            if args.require_exact_specialization_contract is not None
            else None
        )
        metadata_gate_values = (
            args.require_exact_specialization_contract_metadata,
            args.require_corpus_driver,
            args.require_shape_matrix,
            args.require_source_inventory,
            args.require_corpus_id,
            args.require_corpus_version,
        )
        if args.require_exact_specialization_contract is not None and not all(
            metadata_gate_values
        ):
            raise ValueError(
                "exact specialization contract requires metadata, corpus driver, "
                "shape matrix, source inventory, corpus id, and corpus version"
            )
        if args.require_exact_specialization_contract is None and any(
            metadata_gate_values
        ):
            raise ValueError(
                "contract metadata/artifact gates require "
                "--require-exact-specialization-contract"
            )
        required_contract_counts = (
            _read_contract_metadata(
                args.require_exact_specialization_contract_metadata,
                contract_path=args.require_exact_specialization_contract,
                corpus_driver=args.require_corpus_driver,
                shape_matrix=args.require_shape_matrix,
                source_inventory=args.require_source_inventory,
                expected_corpus_id=args.require_corpus_id,
                expected_corpus_version=args.require_corpus_version,
                resource_rows=required_specializations,
            )
            if required_specializations is not None
            else None
        )
        required_cutlass_packages = _parse_cutlass_package_map(
            args.require_cutlass_package
        )
    except ValueError as exc:
        parser.error(str(exc))

    nvdisasm = shutil.which("nvdisasm")
    if nvdisasm is None:
        parser.error(
            "nvdisasm is required (install the CUDA toolkit or add it to PATH)"
        )
    if (
        args.require_driver_occupancy or args.require_driver_resource_validation
    ) and args.occupancy_device is None:
        parser.error(
            "--require-driver-occupancy/--require-driver-resource-validation "
            "requires --occupancy-device"
        )

    occupancy: _CudaOccupancyAudit | None = None
    if args.occupancy_device is not None:
        try:
            occupancy = _CudaOccupancyAudit(args.occupancy_device)
        except (ImportError, ValueError) as exc:
            parser.error(f"cannot initialize CUDA occupancy audit: {exc}")
        if (
            args.require_occupancy_gpu_uuid
            and occupancy.gpu_uuid != args.require_occupancy_gpu_uuid
        ):
            occupancy.close()
            parser.error(
                f"occupancy device UUID is {occupancy.gpu_uuid!r}, expected "
                f"{args.require_occupancy_gpu_uuid!r}"
            )

    object_paths: set[Path] = set()
    for path in args.paths:
        if path.is_dir():
            object_paths.update(path.rglob("*.o"))
        elif path.suffix == ".o":
            object_paths.add(path)
        else:
            parser.error(f"not an object file or directory: {path}")
    if not object_paths:
        parser.error("no CuTe cache objects were found in the supplied paths")

    rows: list[KernelResources] = []
    failures: list[tuple[Path, str]] = []
    try:
        for object_path in sorted(object_paths):
            try:
                rows.extend(_audit_object(object_path, nvdisasm, occupancy))
            except (
                OSError,
                ValueError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                failures.append((object_path, str(exc)))
    finally:
        if occupancy is not None:
            try:
                occupancy.close()
            except ValueError as exc:
                failures.append((Path("<cuda-context>"), str(exc)))

    rows.sort(
        key=lambda row: (
            not row.local_memory,
            not row.register_ceiling,
            _short_kernel_name(row.kernel),
        )
    )
    if args.output is None:
        _write_report(rows, sys.stdout, args.format)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output:
            _write_report(rows, output, args.format)
        print(
            f"wrote {len(rows)} kernels from {len(object_paths)} objects "
            f"to {args.output}",
            file=sys.stderr,
        )

    for object_path, error in failures:
        print(f"error: {object_path}: {error}", file=sys.stderr)
    if failures:
        return 2
    if not rows:
        print("error: no CUDA kernels were audited", file=sys.stderr)
        return 2

    valid_kernel_ids = {
        row.kernel_id
        for row in rows
        if row.manifest_status == "ok"
        and row.semantic_key
        and row.comparison_semantic_key
        and row.kernel_id
    }
    missing_kernel_patterns = [
        pattern
        for pattern in required_kernel_patterns
        if not any(
            fnmatch.fnmatchcase(kernel_id, pattern) for kernel_id in valid_kernel_ids
        )
    ]
    if required_kernel_patterns:
        print(
            f"kernel coverage patterns={len(required_kernel_patterns)} "
            f"matched={len(required_kernel_patterns) - len(missing_kernel_patterns)} "
            f"missing={len(missing_kernel_patterns)}",
            file=sys.stderr,
        )
    for pattern in missing_kernel_patterns:
        print(
            f"error: required kernel_id pattern not covered: {pattern}", file=sys.stderr
        )
    valid_kernel_symbols = {
        row.kernel
        for row in rows
        if row.manifest_status == "ok"
        and row.semantic_key
        and row.comparison_semantic_key
        and row.kernel
    }
    missing_kernel_symbol_patterns = [
        pattern
        for pattern in required_kernel_symbol_patterns
        if not any(
            fnmatch.fnmatchcase(kernel, pattern) for kernel in valid_kernel_symbols
        )
    ]
    if required_kernel_symbol_patterns:
        print(
            f"kernel symbol coverage patterns={len(required_kernel_symbol_patterns)} "
            f"matched={len(required_kernel_symbol_patterns) - len(missing_kernel_symbol_patterns)} "
            f"missing={len(missing_kernel_symbol_patterns)}",
            file=sys.stderr,
        )
    for pattern in missing_kernel_symbol_patterns:
        print(
            f"error: required CUDA kernel symbol pattern not covered: {pattern}",
            file=sys.stderr,
        )
    specialization_contract_mismatch = False
    if required_specializations is not None:
        audited_identity_list: list[tuple[str, ...]] = []
        invalid_contract_rows: list[tuple[str, str]] = []
        for row in rows:
            entry = (
                row.kernel_id,
                row.compile_spec_version,
                row.compile_spec_hash,
                row.compile_spec_json,
                row.semantic_key,
                row.comparison_semantic_key,
                row.target,
                row.compile_kwargs_json,
                row.kernel,
            )
            required_values = entry[:7] + entry[8:]
            if row.manifest_status != "ok" or any(
                not value for value in required_values
            ):
                invalid_contract_rows.append((row.object_file, row.manifest_status))
            else:
                audited_identity_list.append(entry)
        audited_specializations = set(audited_identity_list)
        audited_machine_identities = [
            (entry[5], entry[8]) for entry in audited_identity_list
        ]
        duplicate_audited_identities = len(audited_machine_identities) - len(
            set(audited_machine_identities)
        )
        semantic_objects: dict[str, set[tuple[str, str, str]]] = {}
        cache_semantics: dict[str, set[str]] = {}
        for row in rows:
            if row.manifest_status == "ok" and row.semantic_key:
                semantic_objects.setdefault(row.semantic_key, set()).add(
                    (row.cache_key, row.object_sha256, row.object_file)
                )
                cache_semantics.setdefault(row.cache_key, set()).add(row.semantic_key)
        multi_object_semantics = {
            key: objects
            for key, objects in semantic_objects.items()
            if len(objects) != 1
        }
        multi_semantic_cache_keys = {
            key: semantics
            for key, semantics in cache_semantics.items()
            if len(semantics) != 1
        }
        object_identities = {
            object_identity
            for objects in semantic_objects.values()
            for object_identity in objects
        }
        object_count_mismatch = bool(
            required_contract_counts is not None
            and len(object_identities) != required_contract_counts["object_count"]
        )
        missing_specializations = sorted(
            required_specializations - audited_specializations
        )
        unexpected_specializations = sorted(
            audited_specializations - required_specializations
        )
        specialization_contract_mismatch = bool(
            missing_specializations
            or unexpected_specializations
            or duplicate_audited_identities
            or multi_object_semantics
            or multi_semantic_cache_keys
            or invalid_contract_rows
            or object_count_mismatch
        )
        print(
            f"specialization contract={len(required_specializations)} "
            f"audited={len(audited_specializations)} "
            f"missing={len(missing_specializations)} "
            f"unexpected={len(unexpected_specializations)} "
            f"duplicate={duplicate_audited_identities} "
            f"multi_object={len(multi_object_semantics)} "
            f"multi_semantic_cache={len(multi_semantic_cache_keys)} "
            f"invalid_rows={len(invalid_contract_rows)} "
            f"objects={len(object_identities)}/"
            f"{required_contract_counts['object_count'] if required_contract_counts else '?'}",
            file=sys.stderr,
        )
        if duplicate_audited_identities:
            print(
                "error: audited corpus contains duplicate exact resource-row "
                "identities",
                file=sys.stderr,
            )
        for object_file, manifest_status in invalid_contract_rows:
            print(
                "error: exact resource-row contract observed an invalid or "
                f"incomplete row: object={object_file!r} "
                f"manifest_status={manifest_status!r}",
                file=sys.stderr,
            )
        for semantic_key, objects in sorted(multi_object_semantics.items()):
            print(
                f"error: semantic key {semantic_key} maps to multiple objects: "
                f"{sorted(objects)}",
                file=sys.stderr,
            )
        for cache_key, semantics in sorted(multi_semantic_cache_keys.items()):
            print(
                f"error: cache key {cache_key} maps to multiple semantic keys: "
                f"{sorted(semantics)}",
                file=sys.stderr,
            )
        if object_count_mismatch:
            print(
                "error: audited object count differs from reviewed contract: "
                f"audited={len(object_identities)} "
                f"expected={required_contract_counts['object_count']}",
                file=sys.stderr,
            )
        for entry in missing_specializations:
            print(f"error: required specialization missing: {entry}", file=sys.stderr)
        for entry in unexpected_specializations:
            print(
                f"error: specialization absent from exact contract: {entry}",
                file=sys.stderr,
            )
    if args.fail_on_spill and any(row.local_memory for row in rows):
        return 1
    if args.fail_on_register_ceiling and any(row.register_ceiling for row in rows):
        return 1
    if args.require_semantic_manifest and any(
        row.manifest_status != "ok"
        or not row.semantic_key
        or not row.comparison_semantic_key
        for row in rows
    ):
        return 1
    if args.require_launch_dynamic_smem and any(
        row.launch_dynamic_smem_status != "exact" for row in rows
    ):
        return 1
    if args.require_driver_occupancy and any(
        row.occupancy_status != "exact-driver-query"
        or row.occupancy_active_ctas_per_sm is None
        or row.occupancy_active_ctas_per_sm <= 0
        for row in rows
    ):
        return 1
    if args.require_driver_resource_validation and any(
        row.driver_resource_validation_status != "exact-match" for row in rows
    ):
        return 1
    if args.require_cutlass_dsl_version and any(
        row.cutlass_dsl_version != args.require_cutlass_dsl_version for row in rows
    ):
        return 1
    if required_cutlass_packages is not None and any(
        getattr(row, _CUTLASS_PACKAGE_FIELDS[package]) != expected
        for package, expected in required_cutlass_packages.items()
        for row in rows
    ):
        return 1
    if args.require_architecture and any(
        row.architecture != args.require_architecture for row in rows
    ):
        return 1
    if args.require_cutlass_libs_base_version and any(
        row.cutlass_dsl_libs_base_version != args.require_cutlass_libs_base_version
        for row in rows
    ):
        return 1
    if args.require_cutlass_libs_core_version and any(
        row.cutlass_dsl_libs_core_version != args.require_cutlass_libs_core_version
        for row in rows
    ):
        return 1
    if args.require_cutlass_libs_cu12_version and any(
        row.cutlass_dsl_libs_cu12_version != args.require_cutlass_libs_cu12_version
        for row in rows
    ):
        return 1
    if args.require_cutlass_libs_cu13_version and any(
        row.cutlass_dsl_libs_cu13_version != args.require_cutlass_libs_cu13_version
        for row in rows
    ):
        return 1
    if (
        missing_kernel_patterns
        or missing_kernel_symbol_patterns
        or specialization_contract_mismatch
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
