"""Cross-toolchain comparison identities for CUTLASS migration evidence.

Production manifests and cache keys are intentionally raw: CUTLASS version,
package fingerprint, compile options, compile kwargs, and compile environment
must describe the exact object that was built.  This benchmark-only module
constructs a separate identity for pairing the same specialization across
CUTLASS 4.5.2 and 4.6.0.

Only known comparison-only differences are normalized:

* the 4.6 default ``rdc=false`` compile option;
* the same option inside ``__dsl_compile_options_key`` compile kwargs; and
* CUTLASS' package-owned ``libcute_dsl_runtime.so`` component in
  ``CUTE_DSL_LIBS``;
* ``CUTE_DSL_KEEP`` / ``CUTE_DSL_DUMP_DIR`` and the derived
  ``dump-ptx-path=...`` option, which only retain compiler diagnostics and do
  not change generated device code.

Every other byte of specialization input remains significant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


COMPARISON_SEMANTIC_SCHEMA = "sparkinfer.cute.cross_toolchain_comparison_identity.v1"
_OPERATIONAL_COMPILE_ENVIRONMENT = frozenset({"CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR"})
_OPERATIONAL_COMPILE_OPTION_PREFIXES = ("dump-ptx-path=",)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_document(value: str, *, field: str, allow_empty: bool = False) -> Any:
    if not value:
        if allow_empty:
            return None
        raise ValueError(f"{field} is empty")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is invalid JSON") from exc


def _normalize_options(value: object, *, field: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(option, str) or not option for option in value)
    ):
        raise ValueError(f"{field} is not a nonempty-string list")
    return sorted(
        option
        for option in value
        if option != "rdc=false"
        and not option.startswith(_OPERATIONAL_COMPILE_OPTION_PREFIXES)
    )


def normalize_comparison_compile_options(value: object) -> list[str]:
    """Remove options that only retain diagnostics for cross-arm comparison."""

    return _normalize_options(value, field="compile_options")


def _is_package_owned_cute_dsl_runtime(component: str) -> bool:
    candidate = Path(component)
    return (
        candidate.name == "libcute_dsl_runtime.so"
        and "nvidia_cutlass_dsl" in candidate.parts
    )


def normalize_comparison_compile_environment(value: object) -> list[list[str]]:
    """Normalize only CUTLASS' package runtime component for comparison."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("compile_environment is not a name/value list")
    normalized: list[list[str]] = []
    names: set[str] = set()
    for raw_entry in value:
        if (
            not isinstance(raw_entry, Sequence)
            or isinstance(raw_entry, (str, bytes))
            or len(raw_entry) != 2
            or not isinstance(raw_entry[0], str)
            or not raw_entry[0]
            or not isinstance(raw_entry[1], str)
        ):
            raise ValueError("compile_environment has an invalid entry")
        name, raw_value = raw_entry
        if name in names:
            raise ValueError(f"compile_environment repeats {name!r}")
        names.add(name)
        if name in _OPERATIONAL_COMPILE_ENVIRONMENT:
            continue
        if name != "CUTE_DSL_LIBS":
            normalized.append([name, raw_value])
            continue
        retained = [
            component
            for component in raw_value.split(os.pathsep)
            if not _is_package_owned_cute_dsl_runtime(component)
        ]
        if retained:
            normalized.append([name, os.pathsep.join(retained)])
    return sorted(normalized, key=lambda entry: entry[0])


def normalize_comparison_compile_kwargs(value: object) -> object:
    """Normalize only the DSL option provenance nested in compile kwargs."""

    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    if "__dsl_compile_options_key" in normalized:
        normalized["__dsl_compile_options_key"] = _normalize_options(
            normalized["__dsl_compile_options_key"],
            field="compile_kwargs.__dsl_compile_options_key",
        )
    return normalized


def comparison_semantic_payload(
    *,
    target: str,
    compile_spec_hash: str,
    compile_spec_json: str,
    compile_kwargs_json: str,
    compile_options: object,
    compile_environment: object,
) -> dict[str, object]:
    """Build the exact specialization payload used only for A/B pairing."""

    if not target:
        raise ValueError("target is empty")
    compile_spec = _json_document(compile_spec_json, field="compile_spec_json")
    if (
        hashlib.sha256(compile_spec_json.encode("utf-8")).hexdigest()
        != compile_spec_hash
    ):
        raise ValueError("compile_spec_hash does not match compile_spec_json")
    payload: dict[str, object] = {
        "schema": COMPARISON_SEMANTIC_SCHEMA,
        "target": target,
        "compile_spec_hash": compile_spec_hash,
        "compile_spec": compile_spec,
        "compile_options": normalize_comparison_compile_options(compile_options),
        "compile_environment": normalize_comparison_compile_environment(
            compile_environment
        ),
    }
    if compile_kwargs_json:
        compile_kwargs = _json_document(
            compile_kwargs_json,
            field="compile_kwargs_json",
        )
        payload["compile_kwargs"] = normalize_comparison_compile_kwargs(compile_kwargs)
    return payload


def comparison_semantic_key(
    *,
    target: str,
    compile_spec_hash: str,
    compile_spec_json: str,
    compile_kwargs_json: str,
    compile_options: object,
    compile_environment: object,
) -> str:
    """Hash a comparison payload without altering raw production identity."""

    payload = comparison_semantic_payload(
        target=target,
        compile_spec_hash=compile_spec_hash,
        compile_spec_json=compile_spec_json,
        compile_kwargs_json=compile_kwargs_json,
        compile_options=compile_options,
        compile_environment=compile_environment,
    )
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def comparison_semantic_key_from_manifest(manifest: Mapping[str, object]) -> str:
    return comparison_semantic_key(
        target=str(manifest.get("target", "")),
        compile_spec_hash=str(manifest.get("compile_spec_hash", "")),
        compile_spec_json=str(manifest.get("compile_spec_json", "")),
        compile_kwargs_json=str(manifest.get("compile_kwargs_json", "")),
        compile_options=manifest.get("compile_options"),
        compile_environment=manifest.get("compile_environment"),
    )


def comparison_semantic_key_from_resource_row(row: Mapping[str, object]) -> str:
    compile_options = _json_document(
        str(row.get("compile_options_json", "")),
        field="compile_options_json",
    )
    compile_environment = _json_document(
        str(row.get("compile_environment_json", "")),
        field="compile_environment_json",
    )
    return comparison_semantic_key(
        target=str(row.get("target", "")),
        compile_spec_hash=str(row.get("compile_spec_hash", "")),
        compile_spec_json=str(row.get("compile_spec_json", "")),
        compile_kwargs_json=str(row.get("compile_kwargs_json", "")),
        compile_options=compile_options,
        compile_environment=compile_environment,
    )


__all__ = [
    "COMPARISON_SEMANTIC_SCHEMA",
    "comparison_semantic_key",
    "comparison_semantic_key_from_manifest",
    "comparison_semantic_key_from_resource_row",
    "comparison_semantic_payload",
    "normalize_comparison_compile_options",
    "normalize_comparison_compile_environment",
    "normalize_comparison_compile_kwargs",
]
