"""Measurement-only final-PTX capture for the CUTLASS migration corpus.

CUTLASS writes retained PTX to a function-derived filename.  That filename is
not a compile-specialization identity and can be reused by a later compile.
The corpus therefore installs this module before collection and copies the PTX
while :func:`b12x.cute.compiler._store_cute_compile_to_disk` still has both the
compiled artifact and the exact b12x cache key.

This module is benchmark instrumentation.  It is never imported by b12x
production code and it does not alter a kernel, its compile options, or its
runtime arguments.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import tempfile
from contextlib import suppress
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

from validation.cutlass_migration.core.comparison_identity import (
    comparison_semantic_key_from_manifest,
)


_SCHEMA = "b12x.cute.frontend_ptx.v3"
_MANIFEST_SCHEMA = "b12x.cute.compile_manifest.v3"
_CAPTURE_ENV = "CORPUS_RETAIN_FRONTEND_PTX"
_PTXAS_ENV = "CORPUS_COMMON_PTXAS"
_NVDISASM_ENV = "CORPUS_NVDISASM"
_OPERATIONAL_COMPILE_ENV = frozenset({"CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR"})
_PTXAS_VALUE_FLAGS = frozenset(
    {
        "-arch",
        "--gpu-name",
        "-regUsageLevel",
        "--register-usage-level",
        "-maxrregcount",
        "--maxrregcount",
    }
)
_CUDA_ELF_MAGIC = b"\x7fELF\x02\x01\x01\x41"
_CUDA_ELF_MACHINE = 190
_ELF_EXECUTABLE = 2
_ELF64_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_ELF64_PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")
_ELF64_SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
_ELF_SECTION_NOBITS = 8
_ELF_EXTENDED_COUNT = 0xFFFF
_TOOL_VERSION_TIMEOUT_SECONDS = 15
_NVDISASM_TIMEOUT_SECONDS = 120
_PTX_ENTRY_RE = re.compile(
    r"^\s*(?:\.visible\s+)?\.entry\s+(?P<kernel>[^\s(]+)\s*\(", re.MULTILINE
)
_CUBIN_ENTRY_RE = re.compile(
    r"^//--------------------- \.text\.(?P<kernel>kernel_cutlass_kernel_\S+)",
    re.MULTILINE,
)
_PTXAS_VERSION_RE = re.compile(r'\.string\s+"(Cuda compilation tools,[^"]+)"')
_PTXAS_FLAGS_RE = re.compile(r'\.string\s+"([^"\r\n]+)"')
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOUND_MANIFEST_FIELDS = (
    "semantic_key",
    "target",
    "kernel_id",
    "compile_spec_version",
    "compile_spec_hash",
    "compile_spec_json",
    "compile_kwargs_json",
    "package_fingerprint",
    "toolchain",
    "compile_options",
    "compile_environment",
)
_SIDECAR_FIELDS = {
    "schema",
    "cache_key",
    "comparison_semantic_key",
    *_BOUND_MANIFEST_FIELDS,
    "object",
    "compile_manifest",
    "ptx",
    "source_ptxas",
    "common_ptxas",
    "nvdisasm",
    "entrypoint_binding",
}
_TOOL_RECORD_FIELDS = {
    "executable",
    "realpath",
    "sha256",
    "version_output",
    "version_output_sha256",
}
_MANIFEST_FIELDS = {
    "schema",
    "cache_key",
    "cache_format",
    "cache_payload_repr",
    "cache_payload",
    "object_sha256",
    "object_bytes",
    "semantic_key",
    "semantic_payload",
    "target",
    "target_identity",
    "package_fingerprint",
    "toolchain",
    "compile_options",
    "compile_environment",
    "launch_metadata",
    "artifact_evidence_sha256",
    "compile_spec_hash",
    "compile_spec_json",
    "compile_kwargs_hash",
    "compile_kwargs_json",
    "kernel_id",
    "compile_spec_version",
}

_INSTALLED = False
_CAPTURE_ERRORS: list[str] = []
_INSTALLATION_EVIDENCE: dict[str, Any] = {}


def enabled() -> bool:
    return os.environ.get(_CAPTURE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def installation_status() -> dict[str, Any]:
    """Return process-local evidence that the capture hook was installed."""

    return {
        "enabled": enabled(),
        "installed": _INSTALLED,
        "capture_error_count": len(_CAPTURE_ERRORS),
        **_INSTALLATION_EVIDENCE,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            tmp_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            with suppress(OSError):
                os.unlink(tmp_name)


def _observe_tool_record(executable: str, version_flag: str) -> dict[str, Any]:
    """Observe a tool directly, without allowing validation to reuse a cache."""

    path = Path(executable)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"tool must be an existing absolute path: {executable!r}")
    result = subprocess.run(
        [str(path), version_flag],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=_TOOL_VERSION_TIMEOUT_SECONDS,
    )
    version_output = result.stdout.strip()
    return {
        "executable": str(path),
        "realpath": str(path.resolve()),
        "sha256": _sha256_file(path.resolve()),
        "version_output": version_output,
        "version_output_sha256": _sha256_bytes(version_output.encode("utf-8")),
    }


@lru_cache(maxsize=4)
def _tool_record(executable: str, version_flag: str) -> dict[str, Any]:
    """Cache capture-time provenance; validation always bypasses this cache."""

    return _observe_tool_record(executable, version_flag)


def _source_ptxas_metadata(disassembly: str) -> tuple[str, str]:
    version_match = _PTXAS_VERSION_RE.search(disassembly)
    flags = ""
    for match in _PTXAS_FLAGS_RE.finditer(disassembly):
        candidate = match.group(1).strip()
        if re.search(r"(?:^|\s)-O\s+\d+(?:\s|$)", candidate) and re.search(
            r"(?:^|\s)-arch\s+\S+", candidate
        ):
            flags = candidate
            break
    if version_match is None or not flags:
        raise RuntimeError("nvdisasm omitted source PTXAS version or flags")
    _ptxas_replay_argv(flags)
    return version_match.group(1), flags


def _ptxas_replay_argv(flags: str) -> list[str]:
    """Return executable argv for the flags printed in a cubin comment.

    ``nvdisasm`` renders the optimization flag as ``-O 3`` even though PTXAS's
    short-option parser requires the executable spelling ``-O3``.  Preserve
    the raw comment separately, but canonicalize that one presentation detail
    for the recorded and replayed command.  Refuse positional operands because
    input/output paths are supplied by the common-PTXAS runner itself.
    """

    try:
        raw_argv = shlex.split(flags)
    except ValueError as exc:
        raise RuntimeError(f"invalid source PTXAS flags {flags!r}") from exc
    if not raw_argv:
        raise RuntimeError("source PTXAS flags are empty")
    replay_argv: list[str] = []
    index = 0
    while index < len(raw_argv):
        value = raw_argv[index]
        if value == "-O":
            if index + 1 >= len(raw_argv) or not raw_argv[index + 1].isdigit():
                raise RuntimeError(f"invalid PTXAS optimization flag in {flags!r}")
            replay_argv.append(f"-O{raw_argv[index + 1]}")
            index += 2
            continue
        if not value.startswith("-"):
            if not replay_argv or replay_argv[-1] not in _PTXAS_VALUE_FLAGS:
                raise RuntimeError(
                    f"unexpected positional PTXAS flag token {value!r} in {flags!r}"
                )
        replay_argv.append(value)
        index += 1
    return replay_argv


def _checked_elf_extent(offset: int, size: int, available: int, label: str) -> int:
    end = offset + size
    if offset < 0 or size < 0 or end < offset or end > available:
        raise RuntimeError(
            f"embedded CUDA ELF {label} exceeds cache object: "
            f"offset={offset} size={size} available={available}"
        )
    return end


def _embedded_cubin(object_bytes: bytes) -> tuple[bytes, int]:
    """Return exactly the embedded CUDA ELF bytes and their object offset.

    CUTLASS' exported host object appends Python launcher metadata after the
    embedded cubin.  Slicing from the CUDA ELF magic to end-of-object therefore
    hashes unrelated trailer bytes.  Derive the cubin's exact file extent from
    its ELF header, program headers, section headers, segments, and non-NOBITS
    sections, and reject unsupported extended-count encodings.
    """

    if object_bytes.count(_CUDA_ELF_MAGIC) != 1:
        raise RuntimeError("cache object does not contain exactly one CUDA ELF")
    cubin_offset = object_bytes.find(_CUDA_ELF_MAGIC)
    available = len(object_bytes) - cubin_offset
    if available < _ELF64_HEADER.size:
        raise RuntimeError("embedded CUDA ELF header is truncated")
    header = _ELF64_HEADER.unpack_from(object_bytes, cubin_offset)
    (
        ident,
        elf_type,
        machine,
        version,
        _entry,
        program_offset,
        section_offset,
        _flags,
        header_size,
        program_entry_size,
        program_count,
        section_entry_size,
        section_count,
        section_name_index,
    ) = header
    if not ident.startswith(_CUDA_ELF_MAGIC):
        raise RuntimeError("embedded CUDA ELF identity differs")
    if elf_type != _ELF_EXECUTABLE or machine != _CUDA_ELF_MACHINE or version != 1:
        raise RuntimeError(
            "embedded CUDA ELF has unsupported type/machine/version "
            f"{elf_type}/{machine}/{version}"
        )
    if header_size != _ELF64_HEADER.size:
        raise RuntimeError(f"embedded CUDA ELF header size differs: {header_size}")
    if program_count == _ELF_EXTENDED_COUNT:
        raise RuntimeError("embedded CUDA ELF uses an extended program-header count")
    if section_count == 0 or section_name_index == _ELF_EXTENDED_COUNT:
        raise RuntimeError("embedded CUDA ELF uses an extended section encoding")
    if section_name_index >= section_count:
        raise RuntimeError("embedded CUDA ELF section-name index is out of range")
    if program_count and program_offset == 0:
        raise RuntimeError("embedded CUDA ELF has no program-header offset")
    if section_offset == 0:
        raise RuntimeError("embedded CUDA ELF has no section-header offset")
    if program_count and program_entry_size != _ELF64_PROGRAM_HEADER.size:
        raise RuntimeError(
            f"embedded CUDA ELF program-header entry size differs: {program_entry_size}"
        )
    if section_entry_size != _ELF64_SECTION_HEADER.size:
        raise RuntimeError(
            f"embedded CUDA ELF section-header entry size differs: {section_entry_size}"
        )

    extent = header_size
    program_table_size = program_entry_size * program_count
    extent = max(
        extent,
        _checked_elf_extent(
            program_offset,
            program_table_size,
            available,
            "program-header table",
        ),
    )
    section_table_size = section_entry_size * section_count
    extent = max(
        extent,
        _checked_elf_extent(
            section_offset,
            section_table_size,
            available,
            "section-header table",
        ),
    )
    for index in range(program_count):
        entry_offset = cubin_offset + program_offset + index * program_entry_size
        program = _ELF64_PROGRAM_HEADER.unpack_from(object_bytes, entry_offset)
        file_offset = program[2]
        file_size = program[5]
        memory_size = program[6]
        if file_size > memory_size:
            raise RuntimeError(
                f"embedded CUDA ELF program segment {index} file size exceeds memory"
            )
        extent = max(
            extent,
            _checked_elf_extent(
                file_offset,
                file_size,
                available,
                f"program segment {index}",
            ),
        )
    for index in range(section_count):
        entry_offset = cubin_offset + section_offset + index * section_entry_size
        section = _ELF64_SECTION_HEADER.unpack_from(object_bytes, entry_offset)
        section_type = section[1]
        file_offset = section[4]
        file_size = section[5]
        if section_type == _ELF_SECTION_NOBITS:
            continue
        extent = max(
            extent,
            _checked_elf_extent(
                file_offset,
                file_size,
                available,
                f"section {index}",
            ),
        )
    return object_bytes[cubin_offset : cubin_offset + extent], cubin_offset


def _disassemble_cubin(cubin: bytes, nvdisasm: str) -> tuple[list[str], str, str]:
    with tempfile.NamedTemporaryFile(suffix=".cubin") as temporary:
        temporary.write(cubin)
        temporary.flush()
        result = subprocess.run(
            [nvdisasm, temporary.name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_NVDISASM_TIMEOUT_SECONDS,
        )
    entrypoints = sorted(
        {match.group("kernel") for match in _CUBIN_ENTRY_RE.finditer(result.stdout)}
    )
    if not entrypoints:
        raise RuntimeError("nvdisasm reported no CUDA entry points")
    ptxas_version, ptxas_flags = _source_ptxas_metadata(result.stdout)
    return entrypoints, ptxas_version, ptxas_flags


def _compiled_ptx(compiled: Any) -> tuple[bytes, dict[str, Any]]:
    artifacts = getattr(compiled, "artifacts", None)
    raw = getattr(artifacts, "PTX", None)
    if not raw:
        raise RuntimeError(
            "compiled CUTLASS artifact has no retained PTX; "
            "the corpus must set CUTE_DSL_KEEP=ptx"
        )
    if isinstance(raw, str) and ("\n" in raw or "\r" in raw):
        encoded = raw.encode("utf-8")
        return encoded, {
            "kind": "compiled-artifact-inline-text",
            "python_type": type(raw).__name__,
            "bytes": len(encoded),
        }
    if isinstance(raw, (bytes, bytearray, memoryview)):
        encoded = bytes(raw)
        return encoded, {
            "kind": "compiled-artifact-inline-bytes",
            "python_type": type(raw).__name__,
            "bytes": len(encoded),
        }
    if not isinstance(raw, (str, os.PathLike)):
        raise RuntimeError(
            f"compiled CUTLASS PTX has unsupported representation {type(raw).__name__}"
        )
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"retained CUTLASS PTX does not exist: {path}")
    encoded = path.read_bytes()
    return encoded, {
        "kind": "compiled-artifact-path",
        "python_type": type(raw).__name__,
        "path": str(path),
        "bytes": len(encoded),
    }


def _capture_one(
    compiler: Any,
    cache_key: str,
    compiled: Any,
) -> None:
    object_path = compiler._cache_object_path(cache_key)
    manifest_path = compiler._cache_manifest_path(cache_key)
    if not object_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"cache object/manifest missing for {cache_key}")
    object_bytes = object_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    manifest = _validate_compile_manifest(
        _load_json_document(manifest_bytes, "compile manifest"),
        cache_key=cache_key,
        object_bytes=object_bytes,
    )

    ptx_bytes, cutlass_artifact = _compiled_ptx(compiled)
    try:
        ptx_text = ptx_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("retained PTX is not UTF-8") from exc
    ptx_entrypoints = sorted(
        {match.group("kernel") for match in _PTX_ENTRY_RE.finditer(ptx_text)}
    )
    if not ptx_entrypoints:
        raise RuntimeError("retained PTX has no .entry definitions")

    common_ptxas = os.environ.get(_PTXAS_ENV, "/opt/cuda/bin/ptxas")
    nvdisasm = os.environ.get(_NVDISASM_ENV, "/opt/cuda/bin/nvdisasm")
    common_record = _tool_record(common_ptxas, "--version")
    nvdisasm_record = _tool_record(nvdisasm, "--version")
    cubin, embedded_cubin_offset = _embedded_cubin(object_bytes)
    (
        cubin_entrypoints,
        source_ptxas_version,
        source_ptxas_flags,
    ) = _disassemble_cubin(cubin, nvdisasm)
    if ptx_entrypoints != cubin_entrypoints:
        raise RuntimeError(
            f"PTX/cubin entry-point mismatch for {cache_key}: "
            f"ptx={ptx_entrypoints!r} cubin={cubin_entrypoints!r}"
        )

    ptx_path = object_path.with_suffix(".ptx")
    sidecar_path = object_path.with_suffix(".ptx.json")
    _atomic_write(ptx_path, ptx_bytes)
    source_ptxas_argv = _ptxas_replay_argv(source_ptxas_flags)
    sidecar = {
        "schema": _SCHEMA,
        "cache_key": cache_key,
        "comparison_semantic_key": comparison_semantic_key_from_manifest(manifest),
        "semantic_key": manifest.get("semantic_key", ""),
        "target": manifest.get("target", ""),
        "kernel_id": manifest.get("kernel_id", ""),
        "compile_spec_version": manifest.get("compile_spec_version", ""),
        "compile_spec_hash": manifest.get("compile_spec_hash", ""),
        "compile_spec_json": manifest.get("compile_spec_json", ""),
        "compile_kwargs_json": manifest.get("compile_kwargs_json", ""),
        "package_fingerprint": manifest.get("package_fingerprint", ""),
        "toolchain": manifest.get("toolchain"),
        "compile_options": manifest.get("compile_options"),
        "compile_environment": manifest.get("compile_environment"),
        "object": {
            "path": str(object_path),
            "sha256": manifest["object_sha256"],
            "bytes": len(object_bytes),
            "embedded_cubin_offset": embedded_cubin_offset,
            "embedded_cubin_bytes": len(cubin),
            "embedded_cubin_sha256": _sha256_bytes(cubin),
        },
        "compile_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_bytes(manifest_bytes),
            "schema": manifest["schema"],
        },
        "ptx": {
            "path": str(ptx_path),
            "sha256": _sha256_bytes(ptx_bytes),
            "bytes": len(ptx_bytes),
            "cutlass_artifact": cutlass_artifact,
            "entrypoints": ptx_entrypoints,
        },
        "source_ptxas": {
            "version": source_ptxas_version,
            "flags": source_ptxas_flags,
            "flags_argv": source_ptxas_argv,
        },
        "common_ptxas": {
            **common_record,
            "command_argv_template": [
                common_record["executable"],
                *source_ptxas_argv,
                "{input_ptx}",
                "-o",
                "{output_cubin}",
            ],
        },
        "nvdisasm": nvdisasm_record,
        "entrypoint_binding": {
            "status": "exact",
            "ptx_entrypoints": ptx_entrypoints,
            "embedded_cubin_entrypoints": cubin_entrypoints,
        },
    }
    encoded = (
        json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(sidecar_path, encoded)


def _record_error(cache_key: str, exc: Exception) -> None:
    _CAPTURE_ERRORS.append(f"{cache_key}: {type(exc).__name__}: {exc}")


def install() -> None:
    """Install capture and identity filtering before the first corpus compile."""

    global _INSTALLED
    global _INSTALLATION_EVIDENCE
    if not enabled() or _INSTALLED:
        return

    keep_tokens = {
        token.strip().lower()
        for token in os.environ.get("CUTE_DSL_KEEP", "").split(",")
        if token.strip()
    }
    keep_tokens.add("ptx")
    os.environ["CUTE_DSL_KEEP"] = ",".join(sorted(keep_tokens))
    dump_dir_raw = os.environ.get("CUTE_DSL_DUMP_DIR", "").strip()
    if not dump_dir_raw:
        raise RuntimeError("frontend PTX capture requires CUTE_DSL_DUMP_DIR")
    dump_dir = Path(dump_dir_raw)
    if not dump_dir.is_absolute():
        raise RuntimeError("CUTE_DSL_DUMP_DIR must be absolute for PTX capture")
    dump_dir.mkdir(parents=True, exist_ok=True)

    import b12x.cute.compiler as compiler

    cache_info = compiler.compile_cache_info()
    preinstall_activity = {
        field: int(cache_info.get(field, 0))
        for field in (
            "memory_cache_size",
            "memory_cache_hits",
            "memory_cache_misses",
            "disk_cache_hits",
            "compile_misses",
        )
    }
    if any(preinstall_activity.values()):
        raise RuntimeError(
            "frontend PTX capture was installed after compile-cache activity: "
            f"{preinstall_activity}"
        )

    original_environment_key = compiler._compile_environment_key

    @lru_cache(maxsize=1)
    def measurement_environment_key() -> tuple[tuple[str, str], ...]:
        return tuple(
            entry
            for entry in original_environment_key()
            if entry[0] not in _OPERATIONAL_COMPILE_ENV
        )

    compiler._compile_environment_key = measurement_environment_key
    compiler._static_compile_cache_context.cache_clear()

    original_store = compiler._store_cute_compile_to_disk

    @wraps(original_store)
    def store_with_ptx_capture(
        cache_key: str,
        compiled: Any,
        *,
        cache_payload: tuple[object, ...] | None = None,
        func: Any = None,
    ) -> None:
        original_store(
            cache_key,
            compiled,
            cache_payload=cache_payload,
            func=func,
        )
        try:
            _capture_one(compiler, cache_key, compiled)
        except Exception as exc:
            # b12x deliberately suppresses disk-cache persistence failures.
            # Retain the error here so the pytest plugin can make capture a
            # hard corpus gate at session finish.
            _record_error(cache_key, exc)

    compiler._store_cute_compile_to_disk = store_with_ptx_capture
    _INSTALLED = True
    _INSTALLATION_EVIDENCE = {
        "compile_cache_activity_before_install": preinstall_activity,
        "hook_target": "b12x.cute.compiler._store_cute_compile_to_disk",
    }


def _require_exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not an object")
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{label} fields differ: missing={sorted(expected - observed)} "
            f"unexpected={sorted(observed - expected)}"
        )
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object field {key!r}")
        value[key] = item
    return value


def _load_json_document(value: bytes | str, label: str) -> Any:
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not strict UTF-8 JSON") from exc


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
    if len(cache_payload) != 10:
        raise RuntimeError(f"explicit cache payload has {len(cache_payload)} fields")
    if cache_payload[0] != "b12x_cute_compile_cache_v5_explicit_spec":
        raise RuntimeError(f"unsupported cache format {cache_payload[0]!r}")
    semantic: dict[str, Any] = {
        "cache_format": cache_payload[0],
        "target": _semantic_target_key(cache_payload[1]),
        "compile_spec_hash": cache_payload[4],
    }
    semantic["compile_spec"] = _load_json_document(
        str(cache_payload[5]), "cache-payload compile spec"
    )
    if cache_payload[6]:
        semantic["compile_kwargs_hash"] = cache_payload[6]
        semantic["compile_kwargs"] = _load_json_document(
            str(cache_payload[7]), "cache-payload compile kwargs"
        )
    semantic["compile_options"] = _manifest_json_value(cache_payload[8])
    semantic["compile_environment"] = _manifest_json_value(cache_payload[9])
    return semantic


def _validate_launch_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("compile manifest launch metadata is not an object")
    status = value.get("status")
    expected_fields = {"status", "source", "launch_dynamic_smem_bytes"}
    if status == "unknown":
        expected_fields.add("reason")
    metadata = _require_exact_fields(value, expected_fields, "launch metadata")
    if status not in {"exact", "unknown"}:
        raise RuntimeError(f"unsupported launch-metadata status {status!r}")
    if not isinstance(metadata["source"], str) or not metadata["source"]:
        raise RuntimeError("launch metadata has no source")
    launches = metadata["launch_dynamic_smem_bytes"]
    if not isinstance(launches, dict):
        raise RuntimeError("launch dynamic-SMEM metadata is not an object")
    for kernel, sizes in launches.items():
        if not isinstance(kernel, str) or not kernel:
            raise RuntimeError("launch dynamic-SMEM metadata has an invalid kernel")
        if (
            not isinstance(sizes, list)
            or not sizes
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0
                for size in sizes
            )
        ):
            raise RuntimeError(
                f"launch dynamic-SMEM metadata is invalid for {kernel!r}"
            )
    if status == "exact" and not launches:
        raise RuntimeError("exact launch metadata has no kernels")
    if status == "unknown":
        if not isinstance(metadata["reason"], str) or not metadata["reason"]:
            raise RuntimeError("unknown launch metadata has no reason")
        if launches:
            raise RuntimeError("unknown launch metadata unexpectedly has kernel sizes")
    return metadata


def _validate_compile_manifest(
    manifest: Any,
    *,
    cache_key: str,
    object_bytes: bytes,
) -> dict[str, Any]:
    raw = _require_exact_fields(manifest, _MANIFEST_FIELDS, "compile manifest")
    if raw.get("schema") != _MANIFEST_SCHEMA:
        raise RuntimeError("invalid compile manifest schema")
    if raw.get("cache_key") != cache_key:
        raise RuntimeError("compile manifest cache key differs")
    if raw.get("object_sha256") != _sha256_bytes(object_bytes):
        raise RuntimeError("compile manifest object SHA-256 differs")
    if raw.get("object_bytes") != len(object_bytes):
        raise RuntimeError("compile manifest object byte count differs")

    cache_payload_repr = raw.get("cache_payload_repr")
    if not isinstance(cache_payload_repr, str):
        raise RuntimeError("compile manifest has no cache-payload repr")
    if _sha256_bytes(cache_payload_repr.encode("utf-8")) != cache_key:
        raise RuntimeError("compile manifest cache-payload SHA-256 differs")
    try:
        repr_payload = ast.literal_eval(cache_payload_repr)
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError("compile manifest cache-payload repr is invalid") from exc
    cache_payload = raw.get("cache_payload")
    if not isinstance(cache_payload, list):
        raise RuntimeError("compile manifest cache payload is not a list")
    if _manifest_json_value(repr_payload) != cache_payload:
        raise RuntimeError("compile manifest cache payload/repr differ")
    semantic_payload = _semantic_payload_from_cache_payload(cache_payload)

    if raw.get("cache_format") != cache_payload[0]:
        raise RuntimeError("compile manifest cache format differs from payload")
    package_fingerprint = cache_payload[2]
    if (
        not isinstance(package_fingerprint, str)
        or not _SHA256_RE.fullmatch(package_fingerprint)
        or raw.get("package_fingerprint") != package_fingerprint
    ):
        raise RuntimeError("compile manifest package fingerprint differs from payload")
    if raw.get("toolchain") != cache_payload[3]:
        raise RuntimeError("compile manifest toolchain differs from payload")
    if not isinstance(raw["toolchain"], list):
        raise RuntimeError("compile manifest toolchain is not a list")
    toolchain_names: list[str] = []
    for entry in raw["toolchain"]:
        if (
            not isinstance(entry, list)
            or len(entry) < 2
            or not isinstance(entry[0], str)
            or not entry[0]
        ):
            raise RuntimeError("compile manifest toolchain has an invalid entry")
        toolchain_names.append(entry[0])
    if len(toolchain_names) != len(set(toolchain_names)):
        raise RuntimeError("compile manifest toolchain repeats a component")
    compile_spec_hash = cache_payload[4]
    compile_spec_json = cache_payload[5]
    if (
        not isinstance(compile_spec_hash, str)
        or not _SHA256_RE.fullmatch(compile_spec_hash)
        or raw.get("compile_spec_hash") != compile_spec_hash
        or raw.get("compile_spec_json") != compile_spec_json
        or not isinstance(compile_spec_json, str)
        or _sha256_bytes(compile_spec_json.encode("utf-8")) != compile_spec_hash
    ):
        raise RuntimeError("compile manifest compile-spec identity differs")
    compile_spec = _load_json_document(compile_spec_json, "compile spec")
    if not isinstance(compile_spec, dict):
        raise RuntimeError("compile manifest compile spec is not an object")
    if set(compile_spec) != {"facts", "kernel", "version"}:
        raise RuntimeError("compile manifest compile-spec fields differ")
    if not isinstance(compile_spec["kernel"], str) or not compile_spec["kernel"]:
        raise RuntimeError("compile manifest compile spec has no kernel")
    if (
        not isinstance(compile_spec["version"], int)
        or isinstance(compile_spec["version"], bool)
        or compile_spec["version"] < 1
    ):
        raise RuntimeError("compile manifest compile spec has an invalid version")
    if raw.get("kernel_id") != compile_spec.get("kernel"):
        raise RuntimeError("compile manifest kernel id differs from compile spec")
    if raw.get("compile_spec_version") != compile_spec.get("version"):
        raise RuntimeError("compile manifest version differs from compile spec")

    compile_kwargs_hash = cache_payload[6]
    compile_kwargs_json = cache_payload[7]
    if (
        raw.get("compile_kwargs_hash") != compile_kwargs_hash
        or raw.get("compile_kwargs_json") != compile_kwargs_json
    ):
        raise RuntimeError("compile manifest compile kwargs differ from payload")
    if compile_kwargs_json:
        if (
            not isinstance(compile_kwargs_json, str)
            or not isinstance(compile_kwargs_hash, str)
            or not _SHA256_RE.fullmatch(compile_kwargs_hash)
            or _sha256_bytes(compile_kwargs_json.encode("utf-8")) != compile_kwargs_hash
        ):
            raise RuntimeError("compile manifest compile-kwargs identity differs")
        _load_json_document(compile_kwargs_json, "compile kwargs")
    elif compile_kwargs_hash:
        raise RuntimeError("compile manifest has a kwargs hash without JSON")
    if raw.get("compile_options") != cache_payload[8]:
        raise RuntimeError("compile manifest compile options differ from payload")
    if not isinstance(raw["compile_options"], list) or any(
        not isinstance(option, str) or not option for option in raw["compile_options"]
    ):
        raise RuntimeError("compile manifest compile options are invalid")
    if raw.get("compile_environment") != cache_payload[9]:
        raise RuntimeError("compile manifest compile environment differs from payload")
    compile_environment = raw["compile_environment"]
    if not isinstance(compile_environment, list) or any(
        not isinstance(entry, list)
        or len(entry) != 2
        or not isinstance(entry[0], str)
        or not entry[0]
        or not isinstance(entry[1], str)
        for entry in compile_environment
    ):
        raise RuntimeError("compile manifest compile environment is invalid")
    environment_names = [entry[0] for entry in compile_environment]
    if environment_names != sorted(set(environment_names)):
        raise RuntimeError("compile manifest compile environment is not canonical")

    if raw.get("semantic_payload") != semantic_payload:
        raise RuntimeError("compile manifest semantic payload differs")
    semantic_json = json.dumps(
        semantic_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    semantic_key = _sha256_bytes(semantic_json.encode("utf-8"))
    if raw.get("semantic_key") != semantic_key:
        raise RuntimeError("compile manifest semantic key differs")
    semantic_target = semantic_payload.get("target")
    if not isinstance(semantic_target, dict):
        raise RuntimeError("compile manifest semantic target is not an object")
    if raw.get("target_identity") != semantic_target:
        raise RuntimeError("compile manifest target identity differs")
    expected_target = semantic_target.get("type")
    if expected_target is None:
        expected_target = ".".join(
            filter(
                None,
                (semantic_target.get("module"), semantic_target.get("qualname")),
            )
        )
    if not expected_target or raw.get("target") != expected_target:
        raise RuntimeError("compile manifest target name differs")

    launch_metadata = _validate_launch_metadata(raw.get("launch_metadata"))
    artifact_evidence = {
        "cache_key": cache_key,
        "object_sha256": raw["object_sha256"],
        "launch_metadata": launch_metadata,
    }
    artifact_evidence_json = json.dumps(
        artifact_evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if raw.get("artifact_evidence_sha256") != _sha256_bytes(
        artifact_evidence_json.encode("utf-8")
    ):
        raise RuntimeError("compile manifest artifact-evidence SHA-256 differs")
    return raw


def _parse_ptx_entrypoints(ptx_bytes: bytes) -> list[str]:
    try:
        ptx_text = ptx_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("captured PTX is not UTF-8") from exc
    matches = [match.group("kernel") for match in _PTX_ENTRY_RE.finditer(ptx_text)]
    if not matches:
        raise RuntimeError("captured PTX has no .entry definitions")
    if len(matches) != len(set(matches)):
        raise RuntimeError("captured PTX repeats a .entry definition")
    return sorted(matches)


def _validate_tool_record(
    value: Any,
    label: str,
    *,
    observe_version: bool = True,
) -> dict[str, Any]:
    recorded = _require_exact_fields(value, _TOOL_RECORD_FIELDS, label)
    executable = recorded.get("executable")
    if not isinstance(executable, str) or not executable:
        raise RuntimeError(f"{label} has no executable")
    path = Path(executable)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"{label} executable is not an absolute file")
    version_output = recorded.get("version_output")
    if not isinstance(version_output, str) or not version_output:
        raise RuntimeError(f"{label} has no version output")
    for field in ("sha256", "version_output_sha256"):
        if not _SHA256_RE.fullmatch(str(recorded.get(field, ""))):
            raise RuntimeError(f"{label} {field} is not SHA-256")
    statically_observed = {
        **recorded,
        "realpath": str(path.resolve()),
        "sha256": _sha256_file(path.resolve()),
        "version_output_sha256": _sha256_bytes(version_output.encode("utf-8")),
    }
    if recorded != statically_observed:
        mismatches = sorted(
            field
            for field in _TOOL_RECORD_FIELDS
            if recorded.get(field) != statically_observed.get(field)
        )
        raise RuntimeError(f"{label} static tool provenance differs: {mismatches}")
    if not observe_version:
        return recorded
    # Deliberately bypass the capture-time LRU.  A validator in the same
    # process must observe tool replacement or a changed version result.
    observed = _observe_tool_record(executable, "--version")
    if recorded != observed:
        mismatches = sorted(
            field
            for field in _TOOL_RECORD_FIELDS
            if recorded.get(field) != observed.get(field)
        )
        raise RuntimeError(f"{label} tool provenance differs: {mismatches}")
    return observed


def _artifact_inventory(
    root: Path,
) -> tuple[dict[str, dict[str, Path]], list[str]]:
    artifacts: dict[str, dict[str, Path]] = {
        "object": {},
        "manifest": {},
        "ptx": {},
        "sidecar": {},
    }
    errors: list[str] = []
    suffixes = (
        ("sidecar", ".ptx.json"),
        ("object", ".o"),
        ("manifest", ".json"),
        ("ptx", ".ptx"),
    )
    locks: dict[str, Path] = {}
    if not root.is_dir() or root.is_symlink():
        return artifacts, [f"compile cache is not a directory: {root}"]
    try:
        root_entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError as exc:
        return artifacts, [f"cannot scan compile cache {root}: {exc}"]
    shard_entries: list[tuple[str, list[os.DirEntry[str]]]] = []
    for entry in root_entries:
        path = Path(entry.path)
        if (
            entry.is_symlink()
            or not entry.is_dir(follow_symlinks=False)
            or not re.fullmatch(r"[0-9a-f]{2}", entry.name)
        ):
            errors.append(f"unexpected compile-cache root entry: {path}")
            continue
        try:
            children = sorted(os.scandir(path), key=lambda child: child.name)
        except OSError as exc:
            errors.append(f"cannot scan compile-cache shard {path}: {exc}")
            continue
        if not children:
            errors.append(f"unexpected empty compile-cache shard: {path}")
            continue
        shard_entries.append((entry.name, children))

    for shard, children in shard_entries:
        for entry in children:
            path = Path(entry.path)
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                errors.append(f"unexpected non-file compile-cache entry: {path}")
                continue
            if path.parent.name != shard:
                errors.append(f"misplaced compile-cache entry: {path}")
                continue
            kind = ""
            cache_key = ""
            if path.name.endswith(".lock"):
                lock_key = path.name[: -len(".lock")]
                expected_relative = Path(lock_key[:2]) / path.name
                relative = path.relative_to(root)
                if not _SHA256_RE.fullmatch(lock_key) or relative != expected_relative:
                    errors.append(f"invalid compile-cache lock artifact: {path}")
                elif path.stat().st_size != 0:
                    errors.append(f"nonempty compile-cache lock artifact: {path}")
                elif lock_key in locks:
                    errors.append(
                        f"duplicate compile-cache locks for {lock_key}: "
                        f"{locks[lock_key]}, {path}"
                    )
                else:
                    locks[lock_key] = path
                continue
            for candidate_kind, suffix in suffixes:
                if path.name.endswith(suffix):
                    kind = candidate_kind
                    cache_key = path.name[: -len(suffix)]
                    break
            if not kind:
                errors.append(f"unexpected compile-cache artifact: {path}")
                continue
            if not _SHA256_RE.fullmatch(cache_key):
                errors.append(f"invalid {kind} artifact filename: {path}")
                continue
            expected_relative = Path(cache_key[:2]) / path.name
            relative = path.relative_to(root)
            if relative != expected_relative:
                errors.append(
                    f"misplaced {kind} artifact for {cache_key}: "
                    f"{relative} != {expected_relative}"
                )
                continue
            prior = artifacts[kind].setdefault(cache_key, path)
            if prior != path:
                errors.append(
                    f"duplicate {kind} artifacts for {cache_key}: {prior}, {path}"
                )
    all_keys = set().union(*(set(paths) for paths in artifacts.values()))
    for cache_key in sorted(all_keys):
        missing = sorted(
            kind for kind, paths in artifacts.items() if cache_key not in paths
        )
        if missing:
            errors.append(f"{cache_key}: missing bound artifacts {missing}")
    orphan_locks = sorted(set(locks) - all_keys)
    for cache_key in orphan_locks:
        errors.append(f"{cache_key}: orphan compile-cache lock {locks[cache_key]}")
    return artifacts, errors


def _validate_cutlass_artifact(value: Any, ptx_bytes: bytes) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("PTX CUTLASS artifact provenance is not an object")
    kind = value.get("kind")
    expected_fields = {"kind", "python_type", "bytes"}
    if kind == "compiled-artifact-path":
        expected_fields.add("path")
    if set(value) != expected_fields:
        raise RuntimeError("PTX CUTLASS artifact provenance fields differ")
    if kind not in {
        "compiled-artifact-inline-text",
        "compiled-artifact-inline-bytes",
        "compiled-artifact-path",
    }:
        raise RuntimeError(f"unsupported PTX CUTLASS artifact kind {kind!r}")
    if value.get("bytes") != len(ptx_bytes):
        raise RuntimeError("PTX CUTLASS artifact byte count differs")
    if not isinstance(value.get("python_type"), str) or not value["python_type"]:
        raise RuntimeError("PTX CUTLASS artifact has no Python type")
    if kind == "compiled-artifact-path" and (
        not isinstance(value.get("path"), str) or not value["path"]
    ):
        raise RuntimeError("PTX CUTLASS artifact has no source path")


def _validate_bound_artifacts(
    cache_key: str,
    *,
    object_path: Path,
    manifest_path: Path,
    ptx_path: Path,
    sidecar_path: Path,
    redisassemble: bool,
) -> tuple[str, list[str]]:
    object_bytes = object_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    ptx_bytes = ptx_path.read_bytes()
    manifest = _validate_compile_manifest(
        _load_json_document(manifest_bytes, "compile manifest"),
        cache_key=cache_key,
        object_bytes=object_bytes,
    )
    sidecar = _load_json_document(sidecar_path.read_bytes(), "PTX sidecar")
    object_sha256 = _sha256_bytes(object_bytes)

    sidecar = _require_exact_fields(sidecar, _SIDECAR_FIELDS, "PTX sidecar")
    if sidecar.get("schema") != _SCHEMA:
        raise RuntimeError("invalid PTX sidecar schema")
    if sidecar.get("cache_key") != cache_key:
        raise RuntimeError("PTX sidecar cache key differs")
    expected_comparison_key = comparison_semantic_key_from_manifest(manifest)
    if sidecar.get("comparison_semantic_key") != expected_comparison_key:
        raise RuntimeError("PTX sidecar comparison semantic key differs")
    mismatched_fields = sorted(
        field
        for field in _BOUND_MANIFEST_FIELDS
        if sidecar.get(field) != manifest.get(field)
    )
    if mismatched_fields:
        raise RuntimeError(
            f"PTX sidecar/manifest bound fields differ: {mismatched_fields}"
        )
    compile_environment = sidecar["compile_environment"]
    if not isinstance(compile_environment, list):
        raise RuntimeError("compile environment is not a list")
    leaked_controls = sorted(
        str(entry[0])
        for entry in compile_environment
        if isinstance(entry, list) and entry and entry[0] in _OPERATIONAL_COMPILE_ENV
    )
    if leaked_controls:
        raise RuntimeError(
            f"PTX retention controls leaked into compile identity: {leaked_controls}"
        )

    current_ptx_entrypoints = _parse_ptx_entrypoints(ptx_bytes)
    cubin, embedded_cubin_offset = _embedded_cubin(object_bytes)
    common_ptxas = _require_exact_fields(
        sidecar["common_ptxas"],
        _TOOL_RECORD_FIELDS | {"command_argv_template"},
        "common PTXAS",
    )
    expected_common_ptxas = os.environ.get(_PTXAS_ENV, "").strip()
    if expected_common_ptxas and common_ptxas.get("executable") != (
        expected_common_ptxas
    ):
        raise RuntimeError("common PTXAS differs from the requested executable")
    common_tool = _validate_tool_record(
        {field: common_ptxas[field] for field in _TOOL_RECORD_FIELDS},
        "common PTXAS",
        observe_version=redisassemble,
    )
    nvdisasm = _validate_tool_record(
        sidecar["nvdisasm"],
        "nvdisasm",
        observe_version=redisassemble,
    )
    expected_nvdisasm = os.environ.get(_NVDISASM_ENV, "").strip()
    if expected_nvdisasm and nvdisasm.get("executable") != expected_nvdisasm:
        raise RuntimeError("nvdisasm differs from the requested executable")
    binding = _require_exact_fields(
        sidecar["entrypoint_binding"],
        {"status", "ptx_entrypoints", "embedded_cubin_entrypoints"},
        "entry-point binding",
    )
    recorded_cubin_entrypoints = binding["embedded_cubin_entrypoints"]
    if (
        not isinstance(recorded_cubin_entrypoints, list)
        or not recorded_cubin_entrypoints
        or any(
            not isinstance(kernel, str) or not kernel
            for kernel in recorded_cubin_entrypoints
        )
        or recorded_cubin_entrypoints != sorted(set(recorded_cubin_entrypoints))
    ):
        raise RuntimeError("recorded cubin entry points are invalid")
    source_ptxas = _require_exact_fields(
        sidecar["source_ptxas"],
        {"version", "flags", "flags_argv"},
        "source PTXAS",
    )
    recorded_source_version = source_ptxas["version"]
    recorded_source_flags = source_ptxas["flags"]
    if (
        not isinstance(recorded_source_version, str)
        or not recorded_source_version
        or not isinstance(recorded_source_flags, str)
        or not recorded_source_flags
    ):
        raise RuntimeError("recorded source PTXAS provenance is invalid")
    recorded_source_argv = _ptxas_replay_argv(recorded_source_flags)
    if source_ptxas["flags_argv"] != recorded_source_argv:
        raise RuntimeError("recorded source PTXAS argv differs from flags")

    if redisassemble:
        (
            current_cubin_entrypoints,
            source_ptxas_version,
            source_ptxas_flags,
        ) = _disassemble_cubin(cubin, str(nvdisasm["executable"]))
    else:
        # The profiled child validates every stored byte/hash/identity without
        # spawning tool-version or disassembly subprocesses.  A later/default
        # validation must re-observe both tools and redisassemble.
        current_cubin_entrypoints = recorded_cubin_entrypoints
        source_ptxas_version = recorded_source_version
        source_ptxas_flags = recorded_source_flags
    if current_ptx_entrypoints != current_cubin_entrypoints:
        raise RuntimeError(
            "current PTX/cubin entry points differ: "
            f"{current_ptx_entrypoints!r} != {current_cubin_entrypoints!r}"
        )
    source_ptxas_argv = _ptxas_replay_argv(source_ptxas_flags)
    expected_source_ptxas = {
        "version": source_ptxas_version,
        "flags": source_ptxas_flags,
        "flags_argv": source_ptxas_argv,
    }
    if sidecar["source_ptxas"] != expected_source_ptxas:
        raise RuntimeError("source PTXAS provenance differs from current object")
    expected_template = [
        common_tool["executable"],
        *source_ptxas_argv,
        "{input_ptx}",
        "-o",
        "{output_cubin}",
    ]
    if common_ptxas["command_argv_template"] != expected_template:
        raise RuntimeError("common PTXAS command template differs")

    object_record = _require_exact_fields(
        sidecar["object"],
        {
            "path",
            "sha256",
            "bytes",
            "embedded_cubin_offset",
            "embedded_cubin_bytes",
            "embedded_cubin_sha256",
        },
        "sidecar object",
    )
    expected_object_record = {
        "path": str(object_path),
        "sha256": object_sha256,
        "bytes": len(object_bytes),
        "embedded_cubin_offset": embedded_cubin_offset,
        "embedded_cubin_bytes": len(cubin),
        "embedded_cubin_sha256": _sha256_bytes(cubin),
    }
    if object_record != expected_object_record:
        raise RuntimeError("sidecar object identity/provenance differs")
    manifest_record = _require_exact_fields(
        sidecar["compile_manifest"],
        {"path", "sha256", "schema"},
        "sidecar compile manifest",
    )
    expected_manifest_record = {
        "path": str(manifest_path),
        "sha256": _sha256_bytes(manifest_bytes),
        "schema": _MANIFEST_SCHEMA,
    }
    if manifest_record != expected_manifest_record:
        raise RuntimeError("sidecar compile-manifest identity/provenance differs")
    ptx_record = _require_exact_fields(
        sidecar["ptx"],
        {"path", "sha256", "bytes", "cutlass_artifact", "entrypoints"},
        "sidecar PTX",
    )
    _validate_cutlass_artifact(ptx_record["cutlass_artifact"], ptx_bytes)
    expected_ptx_record = {
        "path": str(ptx_path),
        "sha256": _sha256_bytes(ptx_bytes),
        "bytes": len(ptx_bytes),
        "cutlass_artifact": ptx_record["cutlass_artifact"],
        "entrypoints": current_ptx_entrypoints,
    }
    if ptx_record != expected_ptx_record:
        raise RuntimeError("sidecar PTX identity/provenance differs")
    expected_binding = {
        "status": "exact",
        "ptx_entrypoints": current_ptx_entrypoints,
        "embedded_cubin_entrypoints": current_cubin_entrypoints,
    }
    if binding != expected_binding:
        raise RuntimeError("sidecar entry-point binding differs from artifacts")
    return str(manifest["semantic_key"]), current_ptx_entrypoints


def validate_cache(
    cache_dir: Path,
    *,
    required: bool = False,
    redisassemble: bool = True,
) -> dict[str, Any]:
    """Validate complete one-to-one object/manifest/PTX capture coverage.

    ``redisassemble=False`` is only for a profiled child that cannot safely
    spawn tool-version or nvdisasm subprocesses.  It still validates the exact
    quartet bytes, manifest/cache identity, ELF extent, tool binaries, recorded
    entry points, hashes, and replay argv.  The default/final gate re-observes
    both tools and always redisassembles each cubin.
    """

    if not required and not enabled():
        return {"enabled": False, "status": "disabled", "object_count": 0}
    errors = list(_CAPTURE_ERRORS)
    # B12X_CUTE_COMPILE_CACHE_DIR is already the object-cache root.  Only the
    # fallback CUTE_DSL_CACHE_DIR path receives a b12x_object_cache suffix in
    # production compiler code.
    root = cache_dir.resolve()
    artifacts, inventory_errors = _artifact_inventory(root)
    errors.extend(inventory_errors)
    objects = artifacts["object"]
    if not objects:
        errors.append(f"no cache objects under {root}")
    semantic_entrypoints: set[tuple[str, str]] = set()
    complete_keys = set.intersection(*(set(paths) for paths in artifacts.values()))
    for cache_key in sorted(complete_keys):
        try:
            semantic_key, entries = _validate_bound_artifacts(
                cache_key,
                object_path=artifacts["object"][cache_key],
                manifest_path=artifacts["manifest"][cache_key],
                ptx_path=artifacts["ptx"][cache_key],
                sidecar_path=artifacts["sidecar"][cache_key],
                redisassemble=redisassemble,
            )
            for kernel in entries:
                identity = (semantic_key, kernel)
                if identity in semantic_entrypoints:
                    raise RuntimeError(f"duplicate semantic entry point {identity!r}")
                semantic_entrypoints.add(identity)
        except Exception as exc:
            errors.append(f"{cache_key}: {type(exc).__name__}: {exc}")
    return {
        "enabled": True,
        "redisassembled": redisassemble,
        "status": "ok" if not errors else "error",
        "object_count": len(objects),
        "artifact_counts": {
            kind: len(paths) for kind, paths in sorted(artifacts.items())
        },
        "semantic_entrypoint_count": len(semantic_entrypoints),
        "errors": errors,
    }
