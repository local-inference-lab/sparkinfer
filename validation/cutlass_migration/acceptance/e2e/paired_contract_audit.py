#!/usr/bin/env python3
"""Offline developer audit for CUTLASS migration ABBA producer contracts.

This is a static schema/emission check, not a CPU acceptance test. Migration
acceptance remains the GPU-only artifact validation performed by
``python -m validation.cutlass_migration acceptance release-index``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from validation.cutlass_migration.paths import CORE_ROOT, DATA_ROOT, E2E_ROOT, REPO_ROOT


DEFAULT_MATRIX = DATA_ROOT / "cute_migration_abba_producer_contracts.json"
DEFAULT_VALIDATOR = E2E_ROOT / "release_index.py"
_SHARED_TIMER_REQUIRED_TOKENS = (
    "initialized_before_target_graph_preconditioning",
    "balanced_abba_target_graph_duration",
    "required_active_throttle_reasons",
    "event_creation_inside_sample_schedule",
)
_SHARED_TIMER_MODULE = "validation.cutlass_migration.core.exact_cache_abba"
_SHARED_TIMER_FUNCTION = "time_conditions"
_SHARED_TIMER_REQUIRED_KEYWORDS = {
    "event_batch_cycles",
    "precondition_seconds",
    "maximum_precondition_seconds",
    "mode_snapshot",
    "required_pstate",
    "max_sm_clock_delta_mhz",
}


class _ModuleAnalysis:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.functions = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.imports: dict[str, tuple[str, str]] = {}
        for node in self.tree.body:
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            for imported in node.names:
                self.imports[imported.asname or imported.name] = (
                    node.module,
                    imported.name,
                )
        self.calls = {
            name: [node for node in ast.walk(function) if isinstance(node, ast.Call)]
            for name, function in self.functions.items()
        }
        self.event_creation_in_loop = {
            name
            for name, function in self.functions.items()
            if _function_creates_cuda_event_in_loop(function)
        }


_ANALYSIS_CACHE: dict[Path, _ModuleAnalysis] = {}


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _function_creates_cuda_event_in_loop(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    parents = {
        child: parent
        for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(function):
        if (
            not isinstance(node, ast.Call)
            or _dotted_name(node.func) != "torch.cuda.Event"
        ):
            continue
        current: ast.AST | None = parents.get(node)
        while current is not None and current is not function:
            if isinstance(current, (ast.For, ast.AsyncFor, ast.While)):
                return True
            current = parents.get(current)
    return False


def _analysis(path: Path) -> _ModuleAnalysis:
    resolved = path.resolve()
    if resolved not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[resolved] = _ModuleAnalysis(resolved)
    return _ANALYSIS_CACHE[resolved]


def _benchmark_module_path(module: str) -> Path | None:
    if not module.startswith("validation.cutlass_migration."):
        return None
    candidate = REPO_ROOT / (module.replace(".", "/") + ".py")
    return candidate.resolve() if candidate.is_file() else None


def _function_reaches_loop_event_creation(
    path: Path,
    function_name: str,
    *,
    seen: set[tuple[Path, str]] | None = None,
) -> list[str]:
    seen = set() if seen is None else seen
    key = (path.resolve(), function_name)
    if key in seen:
        return []
    seen.add(key)
    module = _analysis(path)
    if function_name in module.event_creation_in_loop:
        return [f"{path.name}:{function_name}"]
    for call in module.calls.get(function_name, []):
        called = _dotted_name(call.func)
        if called in module.functions:
            chain = _function_reaches_loop_event_creation(path, called, seen=seen)
        elif called in module.imports:
            imported_module, imported_name = module.imports[called]
            imported_path = _benchmark_module_path(imported_module)
            chain = (
                []
                if imported_path is None
                else _function_reaches_loop_event_creation(
                    imported_path, imported_name, seen=seen
                )
            )
        else:
            chain = []
        if chain:
            return [f"{path.name}:{function_name}", *chain]
    return []


def _reachable_local_functions(path: Path, start: str = "main") -> set[str]:
    module = _analysis(path)
    pending = [start] if start in module.functions else list(module.functions)
    reached: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        for call in module.calls.get(name, []):
            called = _dotted_name(call.func)
            if called in module.functions and called not in reached:
                pending.append(called)
    return reached


def _shared_timer_calls(path: Path) -> list[ast.Call]:
    module = _analysis(path)
    aliases = {
        alias
        for alias, imported in module.imports.items()
        if imported == (_SHARED_TIMER_MODULE, _SHARED_TIMER_FUNCTION)
    }
    return [
        call
        for function_name in _reachable_local_functions(path)
        for call in module.calls.get(function_name, [])
        if _dotted_name(call.func) in aliases
    ]


def _static_runtime_follow_up(path: Path) -> list[str]:
    follow_up: list[str] = []
    module = _analysis(path)
    unsafe_chain = _function_reaches_loop_event_creation(path, "main")
    if unsafe_chain:
        follow_up.append(
            "per_sample_cuda_event_creation_reachable:" + "->".join(unsafe_chain)
        )
    timer_calls = _shared_timer_calls(path)
    if not timer_calls:
        follow_up.append("timing_not_routed_through_shared_time_conditions")
    for index, call in enumerate(timer_calls):
        keywords = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
        missing = sorted(_SHARED_TIMER_REQUIRED_KEYWORDS - keywords)
        if missing:
            follow_up.append(
                f"shared_time_conditions_call_{index}_missing_release_controls:"
                + ",".join(missing)
            )
        required_pstate = next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "required_pstate"
            ),
            None,
        )
        if not (
            isinstance(required_pstate, ast.Constant) and required_pstate.value == "P1"
        ):
            follow_up.append(
                f"shared_time_conditions_call_{index}_required_pstate_not_literal_P1"
            )
    if path.name == "w4a16_serving.py":
        for index, call in enumerate(timer_calls):
            if not any(
                keyword.arg == "required_active_throttle_reasons"
                for keyword in call.keywords
            ):
                follow_up.append(
                    f"shared_time_conditions_call_{index}_missing_explicit_"
                    "active_throttle_reasons_policy"
                )
        cli_calls = [
            node
            for node in ast.walk(module.tree)
            if isinstance(node, ast.Call)
            and _dotted_name(node.func).endswith(".add_argument")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--required-active-throttle-reasons"
        ]
        if len(cli_calls) != 1:
            follow_up.append("missing_unique_active_throttle_reasons_cli")
        else:
            defaults = [
                keyword.value
                for keyword in cli_calls[0].keywords
                if keyword.arg == "default"
            ]
            if not (
                len(defaults) == 1
                and isinstance(defaults[0], ast.Constant)
                and defaults[0].value == 0
            ):
                follow_up.append("active_throttle_reasons_cli_default_not_zero")
    return follow_up


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _supported_schemas(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_SUPPORTED_ABBA_SCHEMAS"
            for target in node.targets
        ):
            continue
        call = node.value
        if not isinstance(call, ast.Call) or not call.args:
            break
        value = ast.literal_eval(call.args[0])
        return {str(item) for item in value}
    raise RuntimeError(f"{path}: _SUPPORTED_ABBA_SCHEMAS was not found")


def _load_matrix(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "sparkinfer.cute.migration.abba_producer_contract_matrix.v1":
        raise RuntimeError(f"{path}: unsupported matrix schema")
    return value


def main() -> int:
    args = _args()
    matrix_path = args.matrix.resolve()
    validator_path = args.validator.resolve()
    matrix = _load_matrix(matrix_path)
    producers = matrix.get("producers")
    if not isinstance(producers, list) or not producers:
        raise RuntimeError("producer matrix is empty")
    required_gates = matrix.get("required_release_gates")
    if not isinstance(required_gates, list) or len(required_gates) != len(
        set(required_gates)
    ):
        raise RuntimeError("required release gates are missing or duplicated")

    supported = _supported_schemas(validator_path)
    matrix_schemas = {str(item["schema"]) for item in producers}
    if matrix_schemas != supported:
        raise RuntimeError(
            "producer/validator schema coverage differs: "
            f"missing={sorted(supported - matrix_schemas)}, "
            f"extra={sorted(matrix_schemas - supported)}"
        )

    rows: list[dict[str, object]] = []
    shared_source = (CORE_ROOT / "exact_cache_abba.py").read_text(encoding="utf-8")
    missing_shared_tokens = [
        token for token in _SHARED_TIMER_REQUIRED_TOKENS if token not in shared_source
    ]
    if missing_shared_tokens:
        raise RuntimeError(
            "shared exact-cache timer omits release-control tokens: "
            f"{missing_shared_tokens}"
        )
    for item in producers:
        source_path = (REPO_ROOT / str(item["path"])).resolve()
        if not source_path.is_relative_to(REPO_ROOT) or not source_path.is_file():
            raise RuntimeError(f"invalid producer path: {source_path}")
        source = source_path.read_text(encoding="utf-8")
        source_bundle = f"{source}\n{shared_source}"
        schema = str(item["schema"])
        if schema not in source:
            raise RuntimeError(f"{source_path}: schema literal {schema!r} is absent")
        evidence_keys = item.get("evidence_keys")
        if not isinstance(evidence_keys, list) or not evidence_keys:
            raise RuntimeError(f"{source_path}: evidence key list is empty")
        missing_keys = [
            str(key) for key in evidence_keys if str(key) not in source_bundle
        ]
        if missing_keys:
            raise RuntimeError(
                f"{source_path}: required emission/check tokens are absent: "
                f"{missing_keys}"
            )
        declared_follow_up = item.get("runtime_follow_up")
        if not isinstance(declared_follow_up, list):
            raise RuntimeError(f"{source_path}: runtime_follow_up must be a list")
        follow_up = list(
            dict.fromkeys(
                [
                    *(str(value) for value in declared_follow_up),
                    *_static_runtime_follow_up(source_path),
                ]
            )
        )
        rows.append(
            {
                "producer": str(source_path.relative_to(REPO_ROOT)),
                "schema": schema,
                "serving_units": str(item["serving_units"]),
                "source_sha256": _sha256(source_path),
                "static_evidence_tokens": len(evidence_keys),
                "runtime_follow_up": follow_up,
            }
        )

    single_arm_path = CORE_ROOT / "exact_cache_abba.py"
    single_arm_chain = _function_reaches_loop_event_creation(
        single_arm_path,
        "time_single_graph_conditions",
    )
    single_arm_follow_up = (
        ["per_sample_cuda_event_creation_reachable:" + "->".join(single_arm_chain)]
        if single_arm_chain
        else []
    )
    runtime_follow_up_count = sum(bool(row["runtime_follow_up"]) for row in rows)
    runtime_follow_up_count += bool(single_arm_follow_up)
    result = {
        "schema": "sparkinfer.cute.migration.abba_producer_contract_audit.v1",
        "status": "pass" if runtime_follow_up_count == 0 else "fail",
        "matrix": {
            "path": str(matrix_path),
            "sha256": _sha256(matrix_path),
        },
        "validator": {
            "path": str(validator_path),
            "sha256": _sha256(validator_path),
        },
        "required_release_gates": required_gates,
        "shared_timer_required_tokens": list(_SHARED_TIMER_REQUIRED_TOKENS),
        "shared_timer_required_call_keywords": sorted(_SHARED_TIMER_REQUIRED_KEYWORDS),
        "producer_count": len(rows),
        "runtime_follow_up_count": runtime_follow_up_count,
        "single_arm_helper": {
            "path": str(single_arm_path.relative_to(REPO_ROOT)),
            "source_sha256": _sha256(single_arm_path),
            "runtime_follow_up": single_arm_follow_up,
        },
        "producers": rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
